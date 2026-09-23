"""unbound_simbaEA_difference.py

Unbound gas paper: how SIMBA's ionized-gas, tau and tSZ results change when
ElectronAbundance is read in each particle's own convention (see
simba_ea_correction.py and docs/unbound_gas/simba_electron_abundance_report.md).

For each SIMBA run the quantity is computed twice with identical settings:
EA as read (cached fields/maps, i.e. what the paper uses) and EA corrected
(rebuilt in memory).  Settings follow the paper scripts:

  * ionized_gas / total fraction profiles as in Figure 2
    (make_ratios3x2.py with ratios_3x2_z05.yaml): 3D enclosed-sphere profiles on
    the 1000^3 grid, 2D cumulative and 2D CAP profiles on 0.5' maps with the
    1.6' beam, radii 200-4000 ckpc/h, host haloes 10^13.22-5e14 Msun/h;
  * tau CAP profiles as in the unmasked column of Figure 7
    (simulated_kSZ_masked.py with tau_z05_CAP_masked_flamingo.yaml);
  * tSZ CAP profiles as in the unmasked column of Figure 11
    (simulated_tSZ_masked.py with tSZ_z05_CAP_masked.yaml), m100n1024 only;
  * the global baryon budget (ionized gas, non-ionized gas, stars + BHs),
    straight from the particle data.

Outputs (new files only, under ../figures/<YYYY-MM>/<MM-DD>/simba_test/):
unbound_simbaEA_profiles.pdf, unbound_simbaEA_budget.pdf and
unbound_simbaEA_difference.npz.  Cache writes are disabled.

Usage (from scripts/, on a compute node):
    python simba_test/unbound_simbaEA_difference.py
    python simba_test/unbound_simbaEA_difference.py --sims m50n512_s50noagn --validate --tag SMOKETEST
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml

sys.path.append('../src/')
sys.path.append('unbound_gas/')
sys.path.append('simba_test/')
from loadIO import snap_path  # type: ignore
from halos import select_massive_halos  # type: ignore
from mask_utils import get_cutout_indices_3d, sum_over_cutouts  # type: ignore
import make_ratios3x2 as fig2  # type: ignore  # read-only reuse of the Figure 2 helpers
from simba_ea_correction import (build_map, build_field_3d, corrected_electron_abundance,  # type: ignore
                                 forbid_cache_writes, X_H)

FIG2_CONFIG = 'configs/unbound_gas/ratios_3x2_z05.yaml'
FIG7_CONFIG = 'configs/unbound_gas/tau_z05_CAP_masked_flamingo.yaml'
FIG11_CONFIG = 'configs/unbound_gas/tSZ_z05_CAP_masked.yaml'

SIMS = {  # label -> (name, snapshot, feedback)
    'm100n1024_s50':    ('m100n1024', 125, 's50'),
    'm50n512_s50noagn': ('m50n512', 125, 's50noagn'),
    'm50n512_s50nox':   ('m50n512', 125, 's50nox'),
    'm50n512_s50nofb':  ('m50n512', 125, 's50nofb'),
}
TSZ_SIMS = ('m100n1024_s50',)   # Figure 11 uses m100n1024 only
COLOURS = {'m100n1024_s50': '#1f77b4', 'm50n512_s50noagn': '#d62728',
           'm50n512_s50nox': '#2ca02c', 'm50n512_s50nofb': '#9467bd'}
TAU_KEY, TSZ_KEY = 'tau CAP', 'tSZ CAP'


def tex(label: str) -> str:
    return label.replace('_', ' ')


def shell_ratio(stacker, field_num, field_den, params: dict, omega_b: float) -> tuple:
    """Figure 2's 3D enclosed-sphere ratio (make_ratios3x2.compute_3d_profile_ratio),
    applied to given fields so the numerator can be an in-memory field."""
    n = params['n_pixels']
    kpc_per_pixel = stacker.header['BoxSize'] / field_num.shape[0]
    haloes = stacker.loadHalos()
    mask = select_massive_halos(haloes['GroupMass'], 10 ** 13.22, 5 * 1e14)   # as hardcoded in Fig 2
    pos = np.round(haloes['GroupPos'][mask] / kpc_per_pixel).astype(int) % n
    radii = np.linspace(params['min_radius_3d'], params['max_radius_3d'], params['num_radii_3d'])
    p0, p1 = [], []
    for r in radii:
        idx = get_cutout_indices_3d(field_num, pos, np.ones(pos.shape[0]) * r / kpc_per_pixel)
        p0.append(sum_over_cutouts(field_num, idx.copy()))
        p1.append(sum_over_cutouts(field_den, idx.copy()))
    ratio, err = fig2._profile_ratio_and_err(np.array(p0), np.array(p1), omega_b, stacker.header['Omega0'])
    return radii, ratio, err


def mean_profile(profiles: np.ndarray) -> tuple:
    return profiles.mean(axis=1), profiles.std(axis=1) / np.sqrt(profiles.shape[1])


def baryon_budget(stacker, name: str, snapshot: int) -> dict:
    """Baryon mass fractions from the particle data, EA as read and corrected."""
    fn = snap_path(stacker.simPath, snapshot, 'SIMBA', sim_name=name)
    conv = 2 * X_H / (1 + X_H)          # M_ion / M for fully ionised gas per unit n_e/n_H
    m_gas = m_ion_read = m_ion_corr = 0.0
    with h5py.File(fn, 'r') as f:
        g = f['PartType0']
        n = g['Masses'].shape[0]
        for s in range(0, n, 50_000_000):
            e = min(s + 50_000_000, n)
            m = g['Masses'][s:e].astype(np.float64)
            ea = g['ElectronAbundance'][s:e]
            ne = corrected_electron_abundance(ea, g['GrackleHI'][s:e], g['GrackleHII'][s:e])
            m_gas += m.sum()
            m_ion_read += np.sum(m * ea * conv)
            m_ion_corr += np.sum(m * ne.astype(np.float64) * conv)
        m_star = f['PartType4']['Masses'][:].astype(np.float64).sum() if 'PartType4' in f else 0.0
        m_bh = f['PartType5']['Masses'][:].astype(np.float64).sum() if 'PartType5' in f else 0.0
    m_bar = m_gas + m_star + m_bh
    return dict(gas=m_gas / m_bar, stars_bh=(m_star + m_bh) / m_bar,
                ion_read=m_ion_read / m_bar, ion_corr=m_ion_corr / m_bar,
                ion_over_gas_read=m_ion_read / m_gas, ion_over_gas_corr=m_ion_corr / m_gas)


def run_sim(label: str, cfg2: dict, cfg7: dict, cfg11: dict, validate: bool) -> dict:
    name, snapshot, feedback = SIMS[label]
    s2, s7, s11 = cfg2['stack'], cfg7['stack'], cfg11['stack']
    z = s2.get('redshift', 0.5)
    # The same stackers (built at the Fig 2 redshift) serve the Fig 7 and Fig 11 stacks.
    assert s7.get('redshift', 0.5) == z and s11.get('redshift', 0.5) == z, \
        'Fig 2, Fig 7 and Fig 11 configs use different redshifts'
    entry = dict(name=name, snapshot=snapshot, feedback=feedback)
    out = {}
    t0 = time.time()

    st = {v: fig2.setup_stacker(entry, 'SIMBA', z) for v in ('read', 'corr')}
    stacker_r, omega_b, cosmo, _ = st['read']
    stacker_c = st['corr'][0]
    inv_sim = lambda comov: fig2.comoving_to_arcmin(comov, z, cosmo)

    if validate:
        # Rebuild the as-read 2D ionized_gas map in memory (correction off) and
        # compare with the cached map: proves the loader wrapper is transparent.
        cached = stacker_r.makeMap('ionized_gas', z=stacker_r.z, projection=s2['projection'],
                                   beamSize=1.6, save=False, load=True, pixelSize=s2['pixel_size'])
        rebuilt = build_map(fig2.SimulationStacker(name, snapshot, z=z, simType='SIMBA', feedback=feedback),
                            'ionized_gas', z, s2['projection'], s2['pixel_size'], 1.6, corrected=False)
        rel = np.max(np.abs(rebuilt - cached)) / np.max(np.abs(cached))
        print(f"  CHECK {label}: in-memory as-read map vs cache, max |diff|/max = {rel:.2e}, "
              f"sum ratio = {rebuilt.sum() / cached.sum():.8f}", flush=True)
        out['validate_rel_diff'] = rel

    # ---- Figure 2: 3D enclosed-sphere ionized_gas / total ----
    params3d = dict(n_pixels=s2.get('n_pixels', 1000), min_radius_3d=s2.get('min_radius_3d', 200.0),
                    max_radius_3d=s2.get('max_radius_3d', 4000.0), num_radii_3d=s2.get('num_radii_3d', 11))
    n3 = params3d['n_pixels']
    f_tot = stacker_r.makeField('total', nPixels=n3, dim='3D', projection=s2['projection'], save=False, load=True)
    f_ion = stacker_r.makeField('ionized_gas', nPixels=n3, dim='3D', projection=s2['projection'], save=False, load=True)
    out['3D read'] = shell_ratio(stacker_r, f_ion, f_tot, params3d, omega_b)
    del f_ion
    f_ion = build_field_3d(stacker_c, 'ionized_gas', n3, corrected=True)
    out['3D corr'] = shell_ratio(stacker_c, f_ion, f_tot, params3d, omega_b)
    del f_ion, f_tot
    print(f"  {label}: 3D done ({time.time() - t0:.0f} s)", flush=True)

    # ---- Figure 2: 2D cumulative and CAP ionized_gas / total (0.5' maps, 1.6' beam) ----
    params2d = dict(pixel_size=s2.get('pixel_size', 0.5), rad_distance=s2.get('rad_distance', 1.0),
                    projection=s2['projection'], save_field=False, load_field=True,
                    subtract_mean=s2.get('subtract_mean', False))
    build_map(stacker_c, 'ionized_gas', stacker_c.z, s2['projection'], params2d['pixel_size'], 1.6, corrected=True)
    for col, ft, ft2 in (('2D cumulative', s2.get('filter_type_col1', 'cumulative'), s2.get('filter_type_2_col1', 'cumulative')),
                         ('2D CAP', s2.get('filter_type_col2', 'CAP'), s2.get('filter_type_2_col2', 'CAP'))):
        for v, stk in (('read', stacker_r), ('corr', stacker_c)):
            out[f'{col} {v}'] = fig2.compute_2d_profile_ratio(
                stk, 'ionized_gas', 'total', ft, ft2, params2d, omega_b,
                params3d['min_radius_3d'], params3d['max_radius_3d'], params3d['num_radii_3d'], inv_sim)
    print(f"  {label}: 2D fractions done ({time.time() - t0:.0f} s)", flush=True)

    # ---- Figure 7 (unmasked): tau CAP in muK arcmin^2 ----
    kw7 = dict(filterType=s7.get('filter_type', 'CAP'), minRadius=1.0, maxRadius=6.0,
               pixelSize=s7.get('pixel_size', 0.5), save=False, load=True,
               radDistance=s7.get('rad_distance', 1.0), use_subhalos=s7.get('use_subhalos', False),
               halo_abundance_target=s7.get('halo_abundance_target', 5e-4),
               halo_mass_avg=s7.get('halo_mass_avg', 10 ** 13.22),
               halo_mass_upper=s7.get('halo_mass_upper', 5 * 10 ** 14),
               projection=s7.get('projection', 'xy'), mask=False)
    build_map(stacker_c, 'tau', stacker_c.z, kw7['projection'], kw7['pixelSize'], 1.6, corrected=True)
    for v, stk in (('read', stacker_r), ('corr', stacker_c)):
        radii, prof = stk.stackMap('tau', **kw7)
        out[f'{TAU_KEY} {v}'] = (radii * kw7['radDistance'], *mean_profile(prof))
    print(f"  {label}: tau done ({time.time() - t0:.0f} s)", flush=True)

    # ---- Figure 11 (unmasked): tSZ CAP ----
    if label in TSZ_SIMS:
        kw11 = dict(filterType=s11.get('filter_type', 'CAP'), minRadius=s11.get('min_radius', 1.0),
                    maxRadius=s11.get('max_radius', 6.0), numRadii=s11.get('num_radii', 11),
                    pixelSize=s11.get('pixel_size', 0.5), save=False, load=True,
                    radDistance=s11.get('rad_distance', 1.0), projection=s11.get('projection', 'xz'), mask=False)
        build_map(stacker_c, 'tSZ', stacker_c.z, kw11['projection'], kw11['pixelSize'], 1.6, corrected=True)
        for v, stk in (('read', stacker_r), ('corr', stacker_c)):
            radii, prof = stk.stackMap('tSZ', **kw11)
            out[f'{TSZ_KEY} {v}'] = (radii * kw11['radDistance'], *mean_profile(prof))
        print(f"  {label}: tSZ done ({time.time() - t0:.0f} s)", flush=True)

    out['budget'] = baryon_budget(stacker_r, name, snapshot)
    print(f"  {label}: budget {out['budget']} ({time.time() - t0:.0f} s)", flush=True)
    for key in ('3D', '2D cumulative', '2D CAP', TAU_KEY, TSZ_KEY):
        if f'{key} read' in out:
            ratio = out[f'{key} corr'][1] / out[f'{key} read'][1]
            print(f"  {label} {key}: corrected/as read = {np.round(ratio, 4)}", flush=True)
    return out


def plot_profiles(results: dict, path: Path) -> None:
    cols = [('3D', r'$r$ [comoving Mpc/h]', r'$f_{\rm ion}/f_b$ (3D)', 1e-3),
            ('2D cumulative', r'$\theta$ [arcmin]', r'$f_{\rm ion}/f_b$ (2D cumulative)', 1.0),
            ('2D CAP', r'$\theta$ [arcmin]', r'$f_{\rm ion}/f_b$ (2D CAP)', 1.0),
            (TAU_KEY, r'$\theta$ [arcmin]', r'$\tau$ CAP [$\mu$K arcmin$^2$]', 1.0),
            (TSZ_KEY, r'$\theta$ [arcmin]', r'tSZ CAP [arcmin$^2$]', 1.0)]
    fig, axes = plt.subplots(2, len(cols), figsize=(5.2 * len(cols), 9), sharex='col',
                             gridspec_kw=dict(height_ratios=[2, 1]))
    for c, (key, xlabel, ylabel, xscale) in enumerate(cols):
        top, bot = axes[0, c], axes[1, c]
        for label, res in results.items():
            if f'{key} read' not in res:
                continue
            col = COLOURS[label]
            x, yr, er = res[f'{key} read']
            _, yc, ec = res[f'{key} corr']
            x = np.asarray(x) * xscale
            top.plot(x, yr, color=col, ls='--', lw=1.8, marker='o', mfc='none', ms=4)
            top.plot(x, yc, color=col, ls='-', lw=2.2, marker='o', ms=4, label=tex(label))
            top.fill_between(x, yc - ec, yc + ec, color=col, alpha=0.15)
            bot.plot(x, yc / yr, color=col, lw=2, marker='o', ms=4)
        top.set_ylabel(ylabel, fontsize=15)
        bot.set_xlabel(xlabel, fontsize=15)
        bot.set_ylabel('corrected / as read', fontsize=13)
        bot.axhline(1.0, color='k', lw=1, ls=':')
        for ax in (top, bot):
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=13)
    handles = [plt.Line2D([0], [0], color='gray', ls='--', marker='o', mfc='none', label='EA as read (paper)'),
               plt.Line2D([0], [0], color='gray', ls='-', marker='o', label='EA corrected')]
    axes[0, 0].legend(handles=handles, fontsize=12, loc='lower right')
    axes[0, 1].legend(fontsize=12, loc='lower right')
    fig.suptitle(r'SIMBA, $z=0.5$: ElectronAbundance read in each particle\textquoteright s own convention',
                 fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_budget(results: dict, path: Path) -> None:
    labels = list(results)
    fig, ax = plt.subplots(figsize=(9, 1.2 + 1.1 * len(labels)))
    ypos, yticks = [], []
    for i, label in enumerate(labels):
        b = results[label]['budget']
        for j, v in enumerate(('read', 'corr')):
            y = 2.6 * i + j
            ion = b[f'ion_{v}']
            neutral = b['gas'] - ion
            ax.barh(y, ion, color='#1f77b4', label='ionized gas' if (i, j) == (0, 0) else None)
            ax.barh(y, neutral, left=ion, color='#aec7e8', label='non-ionized gas' if (i, j) == (0, 0) else None)
            ax.barh(y, b['stars_bh'], left=b['gas'], color='#ff7f0e', label='stars + BHs' if (i, j) == (0, 0) else None)
            ax.text(1.01, y, f"ion/gas = {b[f'ion_over_gas_{v}']:.3f}", va='center', fontsize=11,
                    transform=ax.get_yaxis_transform())
            ypos.append(y)
            yticks.append(f"{tex(label)} ({'as read' if v == 'read' else 'corrected'})")
    ax.set_yticks(ypos)
    ax.set_yticklabels(yticks, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_xlabel('fraction of baryon mass', fontsize=13)
    ax.invert_yaxis()
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(sims: list, validate: bool, tag: str) -> None:
    t0 = time.time()
    forbid_cache_writes()
    cfgs = []
    for path in (FIG2_CONFIG, FIG7_CONFIG, FIG11_CONFIG):
        with open(path) as f:
            cfgs.append(yaml.safe_load(f))
    results = {}
    for i, label in enumerate(sims):
        print(f"\n=== {label} ===", flush=True)
        results[label] = run_sim(label, *cfgs, validate=validate and i == 0)

    now = datetime.now()
    out_dir = Path('../figures') / now.strftime('%Y-%m') / now.strftime('%m-%d') / 'simba_test'
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'_{tag}' if tag else ''
    p_prof = out_dir / f'unbound_simbaEA_profiles{suffix}.pdf'
    p_bud = out_dir / f'unbound_simbaEA_budget{suffix}.pdf'
    plot_profiles(results, p_prof)
    plot_budget(results, p_bud)

    arrays = {}
    for label, res in results.items():
        for key, val in res.items():
            k = f"{label}/{key}".replace(' ', '_')
            if isinstance(val, dict):
                for kk, vv in val.items():
                    arrays[f'{k}/{kk}'] = np.asarray(vv)
            elif isinstance(val, tuple):
                for name, vv in zip(('x', 'y', 'err'), val):
                    arrays[f'{k}/{name}'] = np.asarray(vv)
            else:
                arrays[k] = np.asarray(val)
    p_npz = out_dir / f'unbound_simbaEA_difference{suffix}.npz'
    np.savez(p_npz, **arrays)
    print(f'\nSaved {p_prof}\nSaved {p_bud}\nSaved {p_npz}\nDone in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sims', nargs='+', default=list(SIMS), choices=list(SIMS))
    parser.add_argument('--validate', action='store_true',
                        help='check that an in-memory as-read rebuild matches the cached map (first sim)')
    parser.add_argument('--tag', default='', help='suffix for output file names')
    args = parser.parse_args()
    main(args.sims, args.validate, args.tag)
