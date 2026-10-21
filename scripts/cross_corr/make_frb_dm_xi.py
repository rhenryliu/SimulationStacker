"""make_frb_dm_xi.py
====================
Simulated FRB dispersion-measure--galaxy cross-correlation, after Figure 5 of
McCarty et al. (arXiv:2608.11296), on single snapshots with the programme's
map-level estimator (``docs/cross_corr/decisions.md`` D-01).

The paper's estimator averages the FRB DM contrast delta_DM = DM - <DM> over
FRB--galaxy pairs in each radial bin and subtracts the same for randoms. On a
periodic projected map that is the galaxy-averaged annulus mean of the
mean-subtracted electron column, so with the electron overdensity
delta_e = N_e / <N_e> - 1 of the projected 'ionized_gas' map,

    xi_{g,deltaDM}(R) = (1 + z) * Nbar_e * <delta_e>_[a, b),

where <.>_[a, b) is the annulus mean over the log bin [a, b) averaged over the
galaxies, Nbar_e is the map's mean comoving free-electron column and (1 + z)
turns a comoving column into DM (DM = int n_e,phys dl / (1 + z)) at the
snapshot redshift, the constant-redshift approximation of the paper's
Appendix A. The box depth cancels between Nbar_e and delta_e. The 'NFW'
(no-feedback) curve replaces delta_e by the CDM overdensity delta_m
(m = total - baryon, D-02) with the same Nbar_e, as the paper multiplies the
dark-matter overdensity by the mean DM.

The annulus means are computed by FFT with the 'Sigma' kernels of
``rprofiles.build_aperture_kernel`` (pixel-centre membership, pixel-count
normalization) and errors come from its spatial block jackknife. The kernels
carry a 1/(pixel area) factor, removed here, so the stored amplitudes are
dimensionless annulus means. Fields: g = SHAM subhalos (NGP), e =
'ionized_gas' (proportional to the free-electron count, mapMaker.
ionized_gas_masses), m = 'total' - 'baryon'.

Predictions: ``docs/cross_corr/predictions/2026-10-20_frb_dm_xi.md``.

Usage
-----
    cd scripts/
    python cross_corr/make_frb_dm_xi.py -p configs/cross_corr/frb_dm_xi_z03.yaml
    python cross_corr/make_frb_dm_xi.py -p configs/cross_corr/frb_dm_xi_z03.yaml --plot-only
    python cross_corr/make_frb_dm_xi.py -p configs/cross_corr/frb_dm_xi_z03.yaml \\
        --check-against ../data/r_profiles/r_profiles_TNG300-1_80_yz.npz

``--check-against`` runs acceptance check A of the predictions file instead: the
annulus routine with the r-profile settings (arcmin apertures, fixed annulus
width) must reproduce that file's Sigma amplitudes Y_eg and Y_gm.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.fft
import yaml

sys.path.append('../src/')
import rprofiles as rp  # noqa: E402
from stacker import SimulationStacker  # noqa: E402
from utils import comoving_to_arcmin  # noqa: E402
from make_r_profiles import sim_label  # noqa: E402

# Physical constants, as in mapMaker.ionized_gas_masses.
MSUN_G = 1.989e33      # g
M_P_G = 1.6726e-24     # g
X_H = 0.76
MU_E = 2.0 / (1.0 + X_H)
KPC_CM = 3.0857e21     # cm
PC_CM = 3.0857e18      # cm; DM in pc cm^-3 is a column in cm^-2 divided by this
RHO_CRIT_H2 = 1.87847e-29   # g cm^-3, critical density today over h^2

# Fixed colours for the runs, keyed by label (FLAMINGO as in the unbound-gas
# figures, TNG300-1 as there).
_COLOURS = {
    'L1_m9_fiducial': '#B30000',
    'L1_m9_fgas-8sigma': '#FF7F0E',
    'L1_m9_Jet_fgas-4sigma': '#C71585',
    'TNG300-1': '#2C1F3F',
}
_LEGEND = {
    'L1_m9_fiducial': r'FLAMINGO L1\_m9',
    'L1_m9_fgas-8sigma': r'FLAMINGO fgas$-8\sigma$',
    'L1_m9_Jet_fgas-4sigma': r'FLAMINGO Jet\_fgas$-4\sigma$',
    'TNG300-1': 'TNG300-1',
}


def log_bin_edges(r_min: float, r_max: float, n_bins: int) -> np.ndarray:
    """Return ``n_bins`` logarithmically spaced, contiguous bins as edges.

    Args:
        r_min (float): Inner edge of the first bin (> 0).
        r_max (float): Outer edge of the last bin.
        n_bins (int): Number of bins.

    Returns:
        np.ndarray: Edges, shape ``(n_bins + 1,)``.
    """
    return np.geomspace(r_min, r_max, n_bins + 1)


def pixel_size_cmpc(boxsize_ckpc_h: float, n_pixels: int, h: float) -> float:
    """Pixel side of a projected map in comoving Mpc (no h).

    Args:
        boxsize_ckpc_h (float): Box side in comoving kpc/h.
        n_pixels (int): Pixels per side.
        h (float): Hubble parameter.

    Returns:
        float: Pixel side in comoving Mpc.
    """
    return boxsize_ckpc_h / n_pixels / h / 1000.0


def npz_name(label: str, snapshot: int, projection: str) -> str:
    """File name of one run's output, shared by the writer and ``--plot-only``."""
    return f'frb_dm_xi_{label}_{snapshot}_{projection}.npz'


def mean_electron_column(ion_map: np.ndarray, pixel_ckpc_h: float, h: float) -> float:
    """Mean comoving free-electron column of an 'ionized_gas' map.

    The map holds M_ion = N_e m_p mu_e per pixel in Msun/h
    (``mapMaker.ionized_gas_masses``), so N_e = M_ion / (m_p mu_e).

    Args:
        ion_map (np.ndarray): Projected 'ionized_gas' map, Msun/h per pixel.
        pixel_ckpc_h (float): Pixel side in comoving kpc/h.
        h (float): Hubble parameter.

    Returns:
        float: Mean electron column per comoving area, in cm^-2.
    """
    n_e_per_pixel = float(np.mean(ion_map, dtype=np.float64)) / h * MSUN_G / (M_P_G * MU_E)
    pixel_area_cm2 = (pixel_ckpc_h / h * KPC_CM) ** 2
    return n_e_per_pixel / pixel_area_cm2


def cosmic_electron_column(omega_b: float, h: float, depth_ckpc_h: float) -> float:
    """Comoving free-electron column through ``depth`` for fully ionized primordial gas.

    Args:
        omega_b (float): Baryon density parameter.
        h (float): Hubble parameter.
        depth_ckpc_h (float): Projection depth in comoving kpc/h.

    Returns:
        float: Electron column in cm^-2, Omega_b rho_crit / (mu_e m_p) times the depth.
    """
    n_e = omega_b * RHO_CRIT_H2 * h ** 2 / (MU_E * M_P_G)       # comoving cm^-3
    return n_e * depth_ckpc_h / h * KPC_CM


def dm_from_column(column_cm2, z: float):
    """Turn a comoving electron column at redshift z into DM in pc cm^-3.

    DM = int n_e,phys dl / (1 + z) with n_e,phys = (1 + z)^3 n_e,com and
    dl = dchi / (1 + z), so DM = (1 + z) * (comoving column).

    Args:
        column_cm2 (float or np.ndarray): Comoving column in cm^-2.
        z (float): Redshift.

    Returns:
        float or np.ndarray: DM in pc cm^-3.
    """
    return (1.0 + z) * np.asarray(column_cm2) / PC_CM


def annulus_means(delta_g: np.ndarray, fields: dict, pixel: float,
                  inner: np.ndarray, widths: np.ndarray, n_jk_side: int):
    """Galaxy-averaged annulus means of overdensity fields, with jackknife.

    For each annulus [inner_i, inner_i + widths_i) this is
    ``pixel**2 * < Sigma_kernel[delta_g] * delta_X >_map``: the rprofiles
    'Sigma' amplitude Y_gX with the kernel's 1/(pixel area) removed, which on
    a periodic map with <delta_X> = 0 equals the mean of delta_X over the
    annulus around each galaxy, averaged over the galaxies.

    Args:
        delta_g (np.ndarray): Galaxy overdensity map, square.
        fields (dict): Overdensity maps keyed by field label, same grid.
        pixel (float): Pixel side, in the units of ``inner`` and ``widths``.
        inner (np.ndarray): Inner annulus radii.
        widths (np.ndarray): Annulus widths (one per annulus).
        n_jk_side (int): Jackknife blocks per side.

    Returns:
        tuple: ``(mean, jk)`` dicts keyed like ``fields``: ``mean[k]`` has
        shape ``(n_annuli,)`` and ``jk[k]`` the leave-one-out realizations,
        shape ``(n_annuli, n_jk_side**2)``.
    """
    n_pixels = delta_g.shape[0]
    shape = delta_g.shape
    spec_g = scipy.fft.rfft2(delta_g, workers=rp._FFT_WORKERS)
    edges = rp.block_edges(n_pixels, n_jk_side)
    counts = rp.block_counts(edges)
    mean = {k: [] for k in fields}
    jk = {k: [] for k in fields}
    for a, w in zip(inner, widths):
        kspec = rp.kernel_spectrum(rp.build_aperture_kernel(n_pixels, pixel, float(a), 'Sigma',
                                                            dr=float(w)))
        filtered = rp.filtered_map(spec_g, kspec, shape) * pixel ** 2
        for k, delta in fields.items():
            sums = rp.block_sums(filtered * delta, edges)
            mean[k].append(sums.sum() / counts.sum())
            jk[k].append(rp.jackknife_realizations(sums, counts))
    return ({k: np.array(v) for k, v in mean.items()},
            {k: np.array(v) for k, v in jk.items()})


def load_overdensities(stacker, n_pixels: int, projection: str, ptype_e: str, verbose=True):
    """Load the cached maps and return delta_e, delta_m and the raw electron map.

    Args:
        stacker (SimulationStacker): Configured stacker.
        n_pixels (int): Pixels per side of the cached grid.
        projection (str): 'xy', 'xz' or 'yz'.
        ptype_e (str): Electron field ('ionized_gas').
        verbose (bool, optional): Print progress.

    Returns:
        tuple: ``(deltas, ion_map)`` with ``deltas`` keyed 'e' and 'm'.
    """
    def _field(ptype):
        if verbose:
            print(f'    loading {ptype} ...', flush=True)
        return stacker.makeField(ptype, nPixels=n_pixels, projection=projection,
                                 save=False, load=True)

    baryon = _field('baryon')
    total = _field('total')
    cdm = rp.derive_cdm_field(total, baryon, header=stacker.header, verbose=verbose)
    del total, baryon
    deltas = {'m': rp.to_overdensity(cdm)}
    del cdm
    ion_map = _field(ptype_e)
    deltas['e'] = rp.to_overdensity(ion_map)
    return deltas, ion_map


def process_simulation(sim_type: str, sim_entry: dict, cfg: dict, out_dir: Path, verbose=True):
    """Compute xi_{g,deltaDM} and its NFW counterpart for one run; write an npz.

    Args:
        sim_type (str): Simulation suite.
        sim_entry (dict): Config entry with ``name``, ``snapshot``, ``n_pixels``
            and optionally ``feedback`` and ``redshift``.
        cfg (dict): The ``stack`` config block.
        out_dir (Path): Output directory for the npz.
        verbose (bool, optional): Print progress.

    Returns:
        Path: The npz written.
    """
    name, snapshot = sim_entry['name'], sim_entry['snapshot']
    feedback = sim_entry.get('feedback')
    n_pixels = int(sim_entry['n_pixels'])
    label = sim_label(sim_type, name, feedback)
    projection = cfg.get('projection', 'yz')
    z_cfg = float(sim_entry.get('redshift', cfg.get('redshift', 0.3)))
    t0 = time.time()

    stacker = SimulationStacker(name, snapshot, nPixels=n_pixels, simType=sim_type,
                                feedback=feedback, z=z_cfg)
    z_true = float(np.asarray(stacker.header['Redshift']).ravel()[0])
    h = float(stacker.header['HubbleParam'])
    lbox = float(stacker.header['BoxSize'])             # ckpc/h
    pixel_ckpc_h = lbox / n_pixels
    pixel_cmpc = pixel_size_cmpc(lbox, n_pixels, h)

    edges = log_bin_edges(float(cfg['r_min_cmpc']), float(cfg['r_max_cmpc']), int(cfg['n_bins']))
    centres = np.sqrt(edges[:-1] * edges[1:])
    if verbose:
        print(f'\n{"=" * 70}\n{label} (snapshot {snapshot}, z = {z_true:.4f}, h = {h})')
        print(f'  grid {n_pixels}^2, pixel {pixel_cmpc:.4f} cMpc; bins {edges[0]:.2f}-'
              f'{edges[-1]:.1f} cMpc, innermost {np.diff(edges)[0] / pixel_cmpc:.2f} pixels wide')
        # Pixel membership is by pixel-centre radius, so an edge on a lattice
        # shell decides a whole shell by convention (rprofiles.lattice_boundary_margin).
        tight = [(float(e),) + rp.lattice_boundary_margin(float(e), pixel_cmpc) for e in edges]
        tight = [t for t in tight if t[1] < 0.01]
        if tight:
            print('  bin edges within 0.01 pixel of a lattice shell: ' + ', '.join(
                f'{e:.3f} cMpc ({s} px, {m:.1e} px away)' for e, m, s in tight))

    deltas, ion_map = load_overdensities(stacker, n_pixels, projection,
                                         cfg.get('particle_type_e', 'ionized_gas'), verbose)
    column = mean_electron_column(ion_map, pixel_ckpc_h, h)
    del ion_map
    omega_b = float(stacker.header['OmegaBaryon'])
    column_cosmic = cosmic_electron_column(omega_b, h, lbox)
    dm_mean = float(dm_from_column(column, z_true))
    if verbose:
        print(f'  mean electron column {column:.4e} cm^-2 = {column / column_cosmic:.4f} '
              f'of fully ionized Omega_b = {omega_b}; mean DM through the box {dm_mean:.2f} pc cm^-3')

    target = float(cfg.get('halo_abundance_target', 1e-3))
    parent_upper = cfg.get('parent_mass_upper', 5e14)
    parent_upper = None if parent_upper is None else float(parent_upper)
    subhalos = stacker.loadSubHalos()
    galaxies, halo_mask = rp.make_galaxy_field(stacker, projection, n_pixels, target,
                                               parent_mass_upper=parent_upper, subhalos=subhalos)
    mstar = subhalos['SubhaloMStar'][halo_mask] / h      # Msun
    del subhalos
    delta_g = rp.to_overdensity(galaxies)
    n_gal = int(galaxies.sum())
    del galaxies
    if verbose:
        print(f'  SHAM {target:g} (cMpc/h)^-3: {n_gal} galaxies, M* {mstar.min():.3e}-'
              f'{mstar.max():.3e} Msun (median {np.median(mstar):.3e})', flush=True)

    n_jk_side = int(cfg.get('n_jk_side', rp.N_JK_SIDE))
    mean, jk = annulus_means(delta_g, deltas, pixel_cmpc, edges[:-1], np.diff(edges), n_jk_side)

    # xi in pc cm^-3: the same constant for every realization, so forming it
    # per jackknife realization (README Sec. 4) is a linear rescaling.
    scale = float(dm_from_column(column, z_true))
    out = {
        'edges_cmpc': edges, 'centres_cmpc': centres,
        'delta_e_annulus': mean['e'], 'delta_m_annulus': mean['m'],
        'xi_e': scale * mean['e'], 'xi_e_jk': scale * jk['e'],
        'xi_nfw': scale * mean['m'], 'xi_nfw_jk': scale * jk['m'],
        'xi_e_err': rp.jackknife_error(scale * jk['e'], axis=1),
        'xi_nfw_err': rp.jackknife_error(scale * jk['m'], axis=1),
        'meta_label': label, 'meta_sim_type': sim_type, 'meta_snapshot': snapshot,
        'meta_redshift': z_true, 'meta_h': h, 'meta_projection': projection,
        'meta_n_pixels': n_pixels, 'meta_pixel_cmpc': pixel_cmpc, 'meta_boxsize_ckpc_h': lbox,
        'meta_mean_electron_column_cm2': column, 'meta_cosmic_electron_column_cm2': column_cosmic,
        'meta_mean_dm_box': dm_mean, 'meta_dm_per_unit_delta': scale,
        'meta_n_galaxies': n_gal, 'meta_abundance_target': target,
        'meta_parent_mass_upper': np.nan if parent_upper is None else parent_upper,
        'meta_mstar_min': mstar.min(), 'meta_mstar_median': np.median(mstar),
        'meta_n_jk': n_jk_side ** 2,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / npz_name(label, snapshot, projection)
    np.savez(out_path, **out)
    if verbose:
        print(f'  R [cMpc] = {np.array2string(centres, precision=3)}')
        print(f'  xi_e   = {np.array2string(out["xi_e"], precision=3)}')
        print(f'  xi_NFW = {np.array2string(out["xi_nfw"], precision=3)}')
        print(f'  ratio  = {np.array2string(out["xi_e"] / out["xi_nfw"], precision=3)}')
        print(f'  wrote {out_path} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    return out_path


def check_against_rprofiles(npz_path: str, sim_type: str, sim_entry: dict, cfg: dict) -> bool:
    """Acceptance check A: reproduce a committed r-profile file's Sigma amplitudes.

    Runs :func:`annulus_means` with that file's arcmin apertures, annulus width
    and SHAM settings, multiplies back the kernel's 1/(pixel area), and
    compares with its ``Y_eg_Sigma`` and ``Y_gm_Sigma``.

    Args:
        npz_path (str): A ``data/r_profiles/r_profiles_*.npz`` file.
        sim_type (str): Suite of the matching run.
        sim_entry (dict): Config entry of the matching run.
        cfg (dict): The ``stack`` config block.

    Returns:
        bool: True if every aperture agrees to better than 1e-9 relative and
        the galaxy count matches.
    """
    ref = np.load(npz_path)
    n_pixels = int(ref['meta_n_pixels'])
    stacker = SimulationStacker(sim_entry['name'], sim_entry['snapshot'], nPixels=n_pixels,
                                simType=sim_type, feedback=sim_entry.get('feedback'),
                                z=float(sim_entry.get('redshift', cfg.get('redshift', 0.3))))
    z_true = float(np.asarray(stacker.header['Redshift']).ravel()[0])
    pixel_arcmin = comoving_to_arcmin(float(stacker.header['BoxSize']), z_true,
                                      cosmo=stacker.cosmo) / n_pixels
    radii = ref['radii']
    dr = float(ref['meta_dr_arcmin'])
    deltas, _ = load_overdensities(stacker, n_pixels, str(ref['meta_projection']),
                                   cfg.get('particle_type_e', 'ionized_gas'))
    galaxies, _ = rp.make_galaxy_field(stacker, str(ref['meta_projection']), n_pixels,
                                       float(ref['meta_abundance_target']),
                                       parent_mass_upper=(None if np.isnan(float(ref['meta_parent_mass_upper']))
                                                          else float(ref['meta_parent_mass_upper'])))
    n_gal = int(galaxies.sum())
    n_jk_side = int(round(np.sqrt(int(ref['meta_n_jk']))))
    mean, _ = annulus_means(rp.to_overdensity(galaxies), deltas, pixel_arcmin, radii,
                            np.full(radii.shape, dr), n_jk_side)
    ok = n_gal == int(ref['meta_n_galaxies'])
    print(f'  pixel {pixel_arcmin:.10f} arcmin (file {float(ref["meta_pixel_arcmin"]):.10f}); '
          f'N_gal {n_gal} (file {int(ref["meta_n_galaxies"])})')
    for key, ref_key in (('e', 'Y_eg_Sigma'), ('m', 'Y_gm_Sigma')):
        new = mean[key] / pixel_arcmin ** 2
        rel = np.max(np.abs(new / ref[ref_key] - 1))
        ok &= rel < 1e-9
        print(f'  {ref_key}: max relative difference {rel:.2e}')
    print(f'CHECK A {"PASS" if ok else "FAIL"}')
    return bool(ok)


def plot(npz_paths, cfg_plot: dict):
    """Draw xi_{g,deltaDM}(R), labelled deltaDM as in the paper, for every run and the NFW curve.

    Args:
        npz_paths (list): npz files written by :func:`process_simulation`.
        cfg_plot (dict): The ``plot`` config block.

    Returns:
        Path: The figure written.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family': 'serif', 'text.usetex': True, 'font.size': 15})

    runs = [dict(np.load(p, allow_pickle=False)) for p in npz_paths]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, d in enumerate(runs):
        label = str(d['meta_label'])
        r, xi, err = d['centres_cmpc'], d['xi_e'], d['xi_e_err']
        tag = f'{_LEGEND.get(label, label.replace("_", chr(92) + "_"))} ($z={float(d["meta_redshift"]):.2f}$)'
        c = _COLOURS.get(label, f'C{i}')
        ax.plot(r, xi, marker='o', ms=4, lw=1.8, color=c, label=tag)
        ax.fill_between(r, xi - err, xi + err, color=c, alpha=0.2, lw=0)
    fid = [d for d in runs if str(d['meta_label']) == cfg_plot.get('nfw_run', 'L1_m9_fiducial')]
    if fid:
        d = fid[0]
        ax.plot(d['centres_cmpc'], d['xi_nfw'], ls='--', color='k', lw=1.8,
                label=f'NFW (gas traces DM, {_LEGEND.get(str(d["meta_label"]), "fiducial")})')
    ax.set_xscale('log')
    # Axes as in the paper's Figure 5: the y axis is xi_{g,deltaDM}, which the
    # paper labels deltaDM (the mean excess DM at separation r), on a linear scale.
    ax.set_yscale(cfg_plot.get('yscale', 'linear'))
    ax.set_xlabel(r'$r$ (cMpc)')
    ax.set_ylabel(r'$\delta{\rm DM}$ (pc cm$^{-3}$)')
    n_gal = ', '.join(f'{int(d["meta_n_galaxies"])}' for d in runs)
    ax.set_title(rf'BGS SHAM, $n = {float(runs[0]["meta_abundance_target"]):g}\,h^3$ cMpc$^{{-3}}$ '
                 rf'($N_g$ = {n_gal}); snapshots, {str(runs[0]["meta_projection"])} projection',
                 fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    now = datetime.now()
    fig_dir = Path(cfg_plot.get('fig_path', '../figures/')) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f'{cfg_plot.get("fig_name", "frb_dm_xi")}.{cfg_plot.get("fig_type", "pdf")}'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Figure saved to {out}')
    return out


def main(path2config: str, plot_only: bool = False, check_against: Optional[str] = None):
    """Run every simulation of the config, then plot; or run check A.

    Args:
        path2config (str): YAML config.
        plot_only (bool, optional): Only redraw from the npz files on disk.
        check_against (str, optional): An r-profile npz for acceptance check A;
            the run is the config entry with that file's label and snapshot.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)
    cfg, cfg_plot = config['stack'], config.get('plot', {})
    out_dir = Path(cfg_plot.get('npz_path', '../data/frb_dm_xi/'))

    entries = [(s['sim_type'], e) for s in config['simulations'] for e in s['sims']]
    if check_against is not None:
        ref = np.load(check_against)
        want = (str(ref['meta_label']), int(ref['meta_snapshot']))
        match = [(t, e) for t, e in entries
                 if (sim_label(t, e['name'], e.get('feedback')), int(e['snapshot'])) == want]
        if not match:
            raise ValueError(f'no config entry matches {want}')
        if not check_against_rprofiles(check_against, *match[0], cfg):
            sys.exit(1)
        return

    paths = []
    for sim_type, entry in entries:
        label = sim_label(sim_type, entry['name'], entry.get('feedback'))
        path = out_dir / npz_name(label, entry['snapshot'], cfg.get('projection', 'yz'))
        paths.append(path if plot_only else process_simulation(sim_type, entry, cfg, out_dir))
    plot(paths, cfg_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulated FRB DM-galaxy cross-correlation.')
    parser.add_argument('-p', '--path2config', default='configs/cross_corr/frb_dm_xi_z03.yaml')
    parser.add_argument('--plot-only', action='store_true',
                        help='Redraw from the npz files on disk instead of computing.')
    parser.add_argument('--check-against', default=None,
                        help='r-profile npz for acceptance check A (no figure).')
    args = parser.parse_args()
    print(f'Arguments: {vars(args)}')
    main(args.path2config, plot_only=args.plot_only, check_against=args.check_against)
