"""lensing_fig14_16_simbaEA.py

Lensing+kSZ paper Figures 14 and 16 with SIMBA's ionized gas recomputed under
the corrected ElectronAbundance convention (see simba_ea_correction.py and
docs/unbound_gas/simba_electron_abundance_report.md).

Only SIMBA is stacked.  Every other curve, and the Figure 14 data points, are
read from the published figure data in ~/projects/DESIxHSC-Lensing/zenodo/
(read-only).  SIMBA is drawn twice: EA as read (dashed; the paper version) and
EA corrected (solid).  For Figure 16 the beam-compensated data points are
recomputed with SIMBA's row of the cached per-simulation beam factors replaced
by its corrected value; the other five rows are read from the cached file.

Checks printed on every run:
  * SIMBA as read, restacked here from the cached maps, against the zenodo
    SIMBA-100 curves (Fig 14 and Fig 16) and against SIMBA's cached beam factor;
  * the beam-compensated points rebuilt from the cached beam factors against
    the zenodo data points.

Nothing is written except the new figures and a small .npz of the plotted
SIMBA curves under ../figures/<YYYY-MM>/<MM-DD>/simba_test/.  Cache writes are
disabled (forbid_cache_writes), and the paper scripts' own outputs
(data/beam_compensated, data/beam_factors, figure_data) are not touched.

Usage (from scripts/, on a compute node):
    python simba_test/lensing_fig14_16_simbaEA.py --z 0.5
    python simba_test/lensing_fig14_16_simbaEA.py --z 0.26
    python simba_test/lensing_fig14_16_simbaEA.py --z 0.5 --skip-corrected   # smoke test
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

sys.path.append('../src/')
sys.path.append('lensing/')
sys.path.append('simba_test/')
from utils import arcmin_to_comoving, comoving_to_arcmin  # type: ignore
from stacker import SimulationStacker  # type: ignore
from snr import detection_snr  # type: ignore
import beam_compensated_ratio_v2 as bcr  # type: ignore  # read-only reuse of helpers
from simba_ea_correction import build_map, forbid_cache_writes  # type: ignore

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore  # noqa: F401 (needed by stacker internals)

ZENODO = Path.home() / 'projects' / 'DESIxHSC-Lensing' / 'zenodo'

# Per-redshift inputs: paper panel letter and the configs the paper scripts use.
PANELS = {
    0.26: dict(letter='a', fig14_config='configs/lensing/mass_ratio_data_z026.yaml',
               fig16_config='configs/lensing/beam_compensated_z026.yaml'),
    0.5:  dict(letter='b', fig14_config='configs/lensing/mass_ratio_data_z05.yaml',
               fig16_config='configs/lensing/beam_compensated_z05.yaml'),
}

_OMEGA_B_SIMBA_FALLBACK = 0.048   # as in compare_data_ratio.py / beam_compensated_ratio_v2.py
SIMBA_LABEL = 'SIMBA-100'


def read_zenodo_csv(path: Path, ycol: str) -> dict:
    """Read a zenodo figure CSV into {series: (theta, y, yerr)}."""
    out: dict = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            s = out.setdefault(row['series'], ([], [], []))
            s[0].append(float(row['theta_arcmin']))
            s[1].append(float(row[ycol]))
            s[2].append(float(row[f'{ycol}_err']) if row[f'{ycol}_err'] else np.nan)
    return {k: tuple(np.array(v) for v in vals) for k, vals in out.items()}


def stack_kwargs(stack: dict) -> dict:
    """stackMap keyword arguments from a paper config's ``stack`` block, never saving."""
    return dict(minRadius=stack.get('min_radius', 1.0), maxRadius=stack.get('max_radius', 6.0),
                numRadii=stack.get('num_radii', 9), projection=stack.get('projection', 'yz'),
                save=False, load=True, radDistance=stack.get('rad_distance', 1.0),
                mask=stack.get('mask_haloes', False), maskRad=stack.get('mask_radii', 3.0),
                use_subhalos=stack.get('use_subhalos', False),
                halo_abundance_target=stack.get('halo_abundance_target', None))


def ratio_of_means(p0: np.ndarray, p1: np.ndarray, factor: float) -> tuple:
    """Ratio of halo means times ``factor`` and its propagated standard error."""
    m0, m1 = p0.mean(axis=1), p1.mean(axis=1)
    e0 = p0.std(axis=1) / np.sqrt(p0.shape[1])
    e1 = p1.std(axis=1) / np.sqrt(p1.shape[1])
    r = m0 / m1 * factor
    return r, np.abs(r) * np.sqrt((e0 / m0) ** 2 + (e1 / m1) ** 2)


def simba_stacker(sim: dict, z: float) -> SimulationStacker:
    return SimulationStacker(sim['name'], sim['snapshot'], z=sim.get('redshift', z),
                             simType='SIMBA', feedback=sim['feedback'])


def simba_entry(config: dict) -> dict:
    for group in config['simulations']:
        if group['sim_type'] == 'SIMBA':
            return group['sims'][0]
    raise ValueError('No SIMBA entry in config')


def colour_table(config: dict) -> dict:
    """Series label -> colour, with the colourmap logic of the paper scripts."""
    colours = {}
    for i, group in enumerate(config['simulations']):
        cmap = matplotlib.colormaps[['plasma', 'twilight', 'hot'][i]]  # type: ignore[attr-defined]
        cols = cmap(np.linspace(0.2, 0.85, len(group['sims'])))
        for j, sim in enumerate(group['sims']):
            if group['sim_type'] == 'IllustrisTNG':
                colours[sim['name']] = cols[j]
            elif group['sim_type'] == 'FLAMINGO':
                colours[f"FLAMINGO {sim['feedback']}".replace('_', '-')] = \
                    bcr._FLAMINGO_COLOURS.get(sim['feedback'], cols[j])
            else:
                colours[SIMBA_LABEL] = cols[j]
    return colours


def omega_baryon(header, fallback: float) -> float:
    """OmegaBaryon from a snapshot header, or ``fallback`` as the paper scripts do."""
    try:
        return header['OmegaBaryon']
    except KeyError:
        return fallback


def r200m_reference(config: dict, z: float, abundance) -> tuple:
    """Mean R200m (arcmin) of the first IllustrisTNG sim's SHAM sample, as in the paper."""
    for group in config['simulations']:
        if group['sim_type'] == 'IllustrisTNG':
            sim = group['sims'][0]
            st = SimulationStacker(sim['name'], sim['snapshot'], z=z, simType='IllustrisTNG')
            cosmo = FlatLambdaCDM(H0=100 * st.header['HubbleParam'], Om0=st.header['Omega0'],
                                  Tcmb0=2.7255 * u.K, Ob0=omega_baryon(st.header, 0.0456))
            _, r200m_kpch = bcr.sham_parent_halo_stats(st, abundance)
            return comoving_to_arcmin(r200m_kpch, z, cosmo=cosmo), sim['name']
    return None, None


def draw(ax, zen: dict, colours: dict, simba_read: tuple, simba_corr: tuple,
         data: tuple, data_label: str, z: float, cosmo_ref, r200m, ylabel: str) -> None:
    """Draw one panel: zenodo curves for the other sims, SIMBA twice, data points."""
    theta = simba_read[0]
    for label, (th, y, err) in zen.items():
        if label == SIMBA_LABEL or label.startswith('DESI'):
            continue
        ax.plot(th, y, label=label, color=colours[label], lw=2, marker='o')
        ax.fill_between(th, y - err, y + err, color=colours[label], alpha=0.2)
    c = colours[SIMBA_LABEL]
    ax.plot(theta, simba_read[1], color=c, lw=2, ls='--', marker='o', mfc='none',
            label=r'SIMBA-100 (EA as read)')
    ax.plot(theta, simba_corr[1], color=c, lw=2.5, ls='-', marker='o',
            label=r'SIMBA-100 (EA corrected)')
    ax.fill_between(theta, simba_corr[1] - simba_corr[2], simba_corr[1] + simba_corr[2],
                    color=c, alpha=0.2)
    ax.errorbar(data[0], data[1], yerr=data[2], fmt='s', color='black', label=data_label,
                markersize=6, capsize=2)
    secax = ax.secondary_xaxis('top', functions=(
        lambda arcmin: arcmin_to_comoving(arcmin, z, cosmo_ref) / 1e3,
        lambda mpc_h: comoving_to_arcmin(mpc_h * 1e3, z, cosmo_ref)))
    secax.set_xlabel(r'R [comoving Mpc/h]')
    ax.axhline(1.0, color='k', ls='--', lw=1.5, label='_nolegend_')
    if r200m[0] is not None:
        ax.axvline(r200m[0], color='gray', ls=':', lw=2,
                   label=rf'$\langle R_{{200\mathrm{{m}}}} \rangle$ ({r200m[1]})')
    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 6.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=12)


def main(z: float, skip_corrected: bool) -> None:
    t0 = time.time()
    forbid_cache_writes()
    panel = PANELS[z]
    letter = panel['letter']

    with open(panel['fig14_config']) as f:
        c14 = yaml.safe_load(f)
    with open(panel['fig16_config']) as f:
        master = yaml.safe_load(f)
    cdir = Path(panel['fig16_config']).parent
    with open(cdir / master['beam_test_config']) as f:
        cbt = yaml.safe_load(f)
    with open(cdir / master['no_beam_config']) as f:
        cnb = yaml.safe_load(f)

    s14, sbt, snb = c14['stack'], cbt['stack'], cnb['stack']
    sim = simba_entry(c14)
    for cfg in (cbt, cnb):
        assert simba_entry(cfg) == sim, 'SIMBA entry differs between Fig 14 and Fig 16 configs'
    print(f"z={z}: SIMBA {sim}; Fig 14 config {panel['fig14_config']}; "
          f"Fig 16 configs {master['beam_test_config']}, {master['no_beam_config']}", flush=True)

    zen14 = read_zenodo_csv(ZENODO / f'fig14{letter}.csv', 'fgas_obs')
    zen16 = read_zenodo_csv(ZENODO / f'fig16{letter}.csv', 'fgas')

    # ---- SIMBA stacks: as read (cached maps) and corrected (in memory) ----
    # Every (projection, pixel, beam) combination stacked for ionized_gas below.
    ion_maps = {(s14['projection'], s14['pixel_size'], s14['beam_size']),
                (sbt['projection'], sbt['pixel_size'], sbt['beam_size']),
                (sbt['projection'], sbt['pixel_size_2'], sbt['beam_size_2']),
                (snb['projection'], snb['pixel_size'], snb['beam_size'])}
    results = {}
    for version in ('read', 'corr'):
        st = simba_stacker(sim, z)
        omega_b = omega_baryon(st.header, _OMEGA_B_SIMBA_FALLBACK)
        f_b = omega_b / st.header['Omega0']
        if version == 'corr' and not skip_corrected:
            # Numerator maps with the corrected EA; 'total' still comes from the cache.
            for proj, pix, beam in sorted(ion_maps, key=str):
                build_map(st, 'ionized_gas', st.z, proj, pix, beam, corrected=True)
            missing = [k for k in ion_maps if ('ionized_gas', st.z, *k) not in st.maps]
            assert not missing, f'corrected maps missing for {missing}'
        k14, kbt, knb = stack_kwargs(s14), stack_kwargs(sbt), stack_kwargs(snb)
        radii14, p14n = st.stackMap('ionized_gas', filterType=s14['filter_type'],
                                    pixelSize=s14['pixel_size'], beamSize=s14['beam_size'], **k14)
        _, p14d = st.stackMap(s14['particle_type_2'], filterType=s14['filter_type_2'],
                              pixelSize=s14['pixel_size_2'], beamSize=s14['beam_size_2'], **k14)
        _, pbtb = st.stackMap('ionized_gas', filterType=sbt['filter_type'],
                              pixelSize=sbt['pixel_size'], beamSize=sbt['beam_size'], **kbt)
        _, pbtn = st.stackMap('ionized_gas', filterType=sbt['filter_type_2'],
                              pixelSize=sbt['pixel_size_2'], beamSize=sbt['beam_size_2'], **kbt)
        radii, pnbn = st.stackMap('ionized_gas', filterType=snb['filter_type'],
                                  pixelSize=snb['pixel_size'], beamSize=snb['beam_size'], **knb)
        _, pnbd = st.stackMap(snb['particle_type_2'], filterType=snb['filter_type_2'],
                              pixelSize=snb['pixel_size_2'], beamSize=snb['beam_size_2'], **knb)
        theta = radii * snb.get('rad_distance', 1.0)
        # Both figures are drawn on this grid, so the Fig 14 and Fig 16 radii must agree.
        assert np.allclose(radii14 * s14.get('rad_distance', 1.0), theta), \
            'Fig 14 and Fig 16 radius grids differ'
        results[version] = dict(
            theta=theta,
            fig14=ratio_of_means(p14n, p14d, 1.0 / f_b),
            fig16=ratio_of_means(pnbn, pnbd, 1.0 / f_b),
            beam_factor=pbtb.mean(axis=1) / pbtn.mean(axis=1),
            n_halos=(p14n.shape[1], pnbn.shape[1]),
        )
        print(f"  SIMBA {version}: Omega_b={omega_b}, haloes (Fig14, Fig16) = {results[version]['n_halos']}",
              flush=True)
        if version == 'read':
            cosmo_ref = FlatLambdaCDM(H0=100 * st.header['HubbleParam'], Om0=st.header['Omega0'],
                                      Tcmb0=2.7255 * u.K, Ob0=omega_b)
        del st
    if skip_corrected:
        print('  [smoke test] --skip-corrected: "corrected" curves are the as-read ones')

    # ---- Check 1: as-read SIMBA against the published curves ----
    rd, cr = results['read'], results['corr']
    for name, zen, key in (('Fig 14', zen14, 'fig14'), ('Fig 16', zen16, 'fig16')):
        th, y, _ = zen[SIMBA_LABEL]
        assert np.allclose(th, rd['theta']), f'{name}: theta grid mismatch'
        print(f"  CHECK {name}: SIMBA as read vs zenodo, max |rel diff| = "
              f"{np.max(np.abs(rd[key][0] / y - 1)):.2e}")

    # ---- Figure 16 compensation with SIMBA's beam factor replaced ----
    bf_path = cbt.get('beam_factor', {}).get('npz_path', f"../data/beam_factors/beam_factor_z{sbt['redshift']}.npz")
    cached = bcr.load_beam_factor_npz(bf_path, cbt)
    if cached is None:
        raise RuntimeError(f'Cached beam factors {bf_path} missing or made with other settings')
    labels = [str(s) for s in cached['sim_labels']]
    i_simba = labels.index(SIMBA_LABEL)
    bf_rows_read = np.array(cached['beam_factor'], dtype=float)
    print(f"  CHECK beam factor: SIMBA as read vs cached row, max |rel diff| = "
          f"{np.max(np.abs(rd['beam_factor'] / bf_rows_read[i_simba] - 1)):.2e}")
    bf_rows_corr = bf_rows_read.copy()
    bf_rows_corr[i_simba] = cr['beam_factor']

    # Only the use_sim_scatter=False branch of beam_compensated_ratio_v2.py is reproduced.
    assert not master.get('compensation', {}).get('use_sim_scatter', False), \
        'use_sim_scatter=True is not implemented here'
    data = bcr.load_measurements_npz(master['plot']['data_path'])['source_bin_0']
    ratio_data, sigma_data, cov_data = data['ratio'], data['ratio_err'], data['ratio_cov_h']
    comp = {}
    for version, rows in (('read', bf_rows_read), ('corr', bf_rows_corr)):
        bf = rows.mean(axis=0)
        inv = 1.0 / bf
        comp[version] = dict(beam_factor=bf, R=ratio_data / bf, sigma=sigma_data * inv,
                             snr=detection_snr(ratio_data / bf, np.outer(inv, inv) * cov_data, null=1.0))
    zd = zen16['DESI x ACT x HSC (beam-corrected)']
    print(f"  CHECK Fig 16 data: rebuilt from cached beam factors vs zenodo, max |rel diff| = "
          f"{np.max(np.abs(comp['read']['R'] / zd[1] - 1)):.2e}")
    print(f"  Fig 16 data shift (corrected / as read): {np.round(comp['corr']['R'] / comp['read']['R'], 4)}")
    print(f"  Fig 16 detection SNR (null=1): as read {comp['read']['snr']:.2f}, corrected {comp['corr']['snr']:.2f}")
    print(f"  SIMBA beam factor corrected / as read: {np.round(cr['beam_factor'] / rd['beam_factor'], 4)}")
    print(f"  SIMBA Fig 14 corrected / as read: {np.round(cr['fig14'][0] / rd['fig14'][0], 4)}")
    print(f"  SIMBA Fig 16 corrected / as read: {np.round(cr['fig16'][0] / rd['fig16'][0], 4)}", flush=True)

    # ---- Figures ----
    now = datetime.now()
    out_dir = Path('../figures') / now.strftime('%Y-%m') / now.strftime('%m-%d') / 'simba_test'
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = '_SMOKETEST' if skip_corrected else ''
    r200m_14 = r200m_reference(c14, z, s14.get('halo_abundance_target'))
    r200m_16 = r200m_reference(cnb, z, snb.get('halo_abundance_target'))

    fig, ax = plt.subplots(figsize=(10, 8))
    d14 = zen14['DESI x ACT x HSC combined']
    draw(ax, zen14, colour_table(c14), (rd['theta'], *rd['fig14']), (cr['theta'], *cr['fig14']),
         d14, 'DESI x ACT x HSC combined', z, cosmo_ref, r200m_14, r'$f_{\rm gas}^{\rm obs}(R)$')
    fig.tight_layout()
    p14 = out_dir / f'fig14{letter}_simbaEA_z{z}{tag}.pdf'
    fig.savefig(p14, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    draw(ax, zen16, colour_table(cnb), (rd['theta'], *rd['fig16']), (cr['theta'], *cr['fig16']),
         (zd[0], comp['corr']['R'], comp['corr']['sigma']),
         r'DESI $\times$ ACT $\times$ HSC (beam-corrected, SIMBA EA corrected)',
         z, cosmo_ref, r200m_16, r'$f_{\rm gas}(R)$')
    fig.tight_layout()
    p16 = out_dir / f'fig16{letter}_simbaEA_z{z}{tag}.pdf'
    fig.savefig(p16, dpi=150)
    plt.close(fig)

    npz = out_dir / f'fig14{letter}_fig16{letter}_simbaEA_z{z}{tag}.npz'
    np.savez(npz, theta_arcmin=rd['theta'],
             fig14_simba_read=rd['fig14'][0], fig14_simba_read_err=rd['fig14'][1],
             fig14_simba_corr=cr['fig14'][0], fig14_simba_corr_err=cr['fig14'][1],
             fig16_simba_read=rd['fig16'][0], fig16_simba_read_err=rd['fig16'][1],
             fig16_simba_corr=cr['fig16'][0], fig16_simba_corr_err=cr['fig16'][1],
             beam_factor_simba_read=rd['beam_factor'], beam_factor_simba_corr=cr['beam_factor'],
             beam_factor_mean_read=comp['read']['beam_factor'], beam_factor_mean_corr=comp['corr']['beam_factor'],
             fig16_data_read=comp['read']['R'], fig16_data_read_err=comp['read']['sigma'],
             fig16_data_corr=comp['corr']['R'], fig16_data_corr_err=comp['corr']['sigma'],
             snr_read=comp['read']['snr'], snr_corr=comp['corr']['snr'],
             smoke_test=skip_corrected)
    print(f'Saved {p14}\nSaved {p16}\nSaved {npz}\nDone in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--z', type=float, required=True, choices=sorted(PANELS))
    parser.add_argument('--skip-corrected', action='store_true',
                        help='smoke test: skip the corrected map builds')
    args = parser.parse_args()
    main(args.z, args.skip_corrected)
