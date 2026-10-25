"""compare_data_ratio_lrg_bins.py

The kSZ-to-lensing ratio f_gas^obs in the four DESI LRG redshift bins, data and
FLAMINGO on a 2x2 grid, all with the v2 filter (option C, ``DSigma_lensing``):

* data: the v2.1 and v2.2 measurements (the f_gas versions: neither or both of
  the kSZ and the lensing carry the DESI completeness weights), read from the
  ratio NPZs that DESIxHSC-Lensing's compare_combined_self.py writes, one per
  lens bin and version. Each panel draws one source sample,
  ``lens_bins[].source_bin`` (0 = HSC combined, 4 = HSC bin 4);
* simulations: the FLAMINGO runs in ``simulations``, stacked as
  compare_data_ratio.py stacks them for the v2-filter Figure 14b
  (configs/lensing/dsigma_lensing/mass_ratio_data_z05.yaml), i.e. the beamed
  ``ionized_gas`` numerator over the unbeamed ``total`` denominator, ratio of
  means times Omega_m / Omega_b, with the standard error of the mean as a band.

Each lens bin names its FLAMINGO snapshot and the redshift that snapshot is
placed at (``sim_redshift``: angular projection and halo selection), as the
z~0.26 configs place snapshot 71 at z = 0.30. Bins sharing a snapshot share its
stack, so they show the same simulation curves.

Caching: per (run, snapshot) the stacks are reduced to their means and standard
errors and written to ``plot.cache_dir`` with the settings that made them. A
cache made with other settings is an error, never overwritten: move it, or give
the other settings their own ``plot.cache_dir``.

Usage (from scripts/)
-----
    # stack whatever is not cached, then plot:
    python lensing/compare_data_ratio_lrg_bins.py -p configs/lensing/dsigma_lensing/ratio_lrg_bins_v2.1_v2.2.yaml
    # stack and cache one (run, snapshot), no figure (for parallel runs):
    python lensing/compare_data_ratio_lrg_bins.py -p <config> --only Jet_fgas-4sigma 62
    # plot from the caches only:
    python lensing/compare_data_ratio_lrg_bins.py -p <config> --plot-only
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import argparse
from astropy.cosmology import Planck18
import astropy.units as u

sys.path.append('../src/')
from stacker import SimulationStacker  # type: ignore
from halos import select_halos  # type: ignore
import figure_data  # type: ignore

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore  # noqa: F401 (needed by stacker internals)

# ---------------------------------------------------------------------------
# Matplotlib style: matches DESIxHSC-Lensing's compare_combined_self.py, the
# data-only version of this figure (mathtext, no usetex)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "figure.titlesize": 20,
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
})

_OMEGA_B_FLAMINGO_FALLBACK = 0.0486  # header provides it; fallback should never trigger

# Fixed colours for the FLAMINGO feedback variants, keyed by feedback name.
# Keep in sync with compare_data_ratio.py / beam_compensated_ratio_v2.py /
# plot_beam_factors.py.
_FLAMINGO_COLOURS = {
    'L1_m9':           '#B30000',  # dark red (fiducial)
    'fgas-8sigma':     '#FF7F0E',  # orange
    'Jet_fgas-4sigma': '#C71585',  # magenta
}

# Stack settings that do not change the numbers, left out of the cache settings.
_CACHE_IGNORED_KEYS = ('load_field', 'save_field')

# Every stack key this script reads; any other key (e.g. mask_haloes) would be
# silently ignored, so it is an error instead.
_USED_STACK_KEYS = {
    'load_field', 'save_field', 'rad_distance', 'min_radius', 'max_radius', 'num_radii',
    'projection', 'filter_type', 'particle_type', 'pixel_size', 'beam_size', 'stamp_res',
    'filter_type_2', 'particle_type_2', 'pixel_size_2', 'beam_size_2', 'use_subhalos',
    'halo_abundance_target', 'halo_mass_upper',
}


def load_measurements_npz(path: str) -> dict:
    """Load a nested dict saved by the companion save_measurements_npz helper.

    The ``.npz`` file is expected to use keys of the form
    ``"outer_key/inner_key"``, which are reconstructed into a two-level dict.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        Nested ``{outer_key: {inner_key: np.ndarray}}`` mapping.
    """
    archive = np.load(path)
    out: dict = {}
    for k in archive.files:
        outer_key, inner_key = k.split("/", 1)
        out.setdefault(outer_key, {})[inner_key] = archive[k]
    return out


def sham_parent_halo_stats(stacker: SimulationStacker,
                           halo_abundance_target: Optional[float],
                           halo_mass_upper: float = 5e14) -> tuple:
    """Mean parent-halo mass (Msun/h) and R200m (comoving kpc/h) of the SHAM sample.

    Replicates the subhalo selection performed inside
    ``SimulationStacker.stack_on_array`` (the ``use_subhalos=True`` branch):
    subhalos are abundance-matched by stellar mass within a parent-mass
    pre-filter, then the GroupMass and GroupRad of their parent FoF groups are
    averaged.

    Args:
        stacker: Instantiated SimulationStacker.
        halo_abundance_target: Target number density in (cMpc/h)^-3.  If None,
            falls back to the stack_on_array default of 5e-4.
        halo_mass_upper: Upper parent-mass bound (Msun/h) for the pre-filter.

    Returns:
        Tuple ``(mean_mass, mean_R200m)`` with the mean parent-halo mass in
        Msun/h and the mean parent-halo R200m in comoving kpc/h.
    """
    if halo_abundance_target is None:
        halo_abundance_target = 5e-4
    subhalos    = stacker.loadSubHalos()
    parents     = stacker.loadHalos()
    parent_mass = parents['GroupMass'][subhalos['SubhaloGrNr']]
    valid       = np.where(parent_mass <= halo_mass_upper)[0]
    local_mask  = select_halos(subhalos['SubhaloMStar'][valid], 'abundance',
                               target_number=halo_abundance_target,
                               Lbox=stacker.header['BoxSize'])
    halo_mask   = valid[local_mask]
    parent_grnr = subhalos['SubhaloGrNr'][halo_mask]
    mean_mass   = np.mean(parents['GroupMass'][parent_grnr])   # Msun/h
    mean_R200m  = np.mean(parents['GroupRad'][parent_grnr])    # comoving kpc/h
    return mean_mass, mean_R200m


def snapshot_redshifts(lens_bins: list) -> dict:
    """Map each FLAMINGO snapshot used by the lens bins to the redshift it is placed at.

    Args:
        lens_bins: The config's ``lens_bins`` list.

    Returns:
        ``{snapshot: sim_redshift}``.

    Raises:
        ValueError: If two lens bins place the same snapshot at different redshifts.
    """
    out: dict = {}
    for b in lens_bins:
        snap, z = int(b['snapshot']), float(b['sim_redshift'])
        if out.setdefault(snap, z) != z:
            raise ValueError(f"Snapshot {snap} is placed at z = {out[snap]} and z = {z} "
                             "by different lens bins; one stack per snapshot is assumed.")
    return out


def _stack_settings(stack_config: dict, sim: dict, snapshot: int, sim_z: float) -> str:
    """Canonical string of everything that determines one cached stack.

    Args:
        stack_config: The config's ``stack`` section.
        sim: One entry of the FLAMINGO ``sims`` list (``name``, ``feedback``).
        snapshot: Snapshot number.
        sim_z: Redshift the snapshot is placed at.

    Returns:
        JSON string, keys sorted.
    """
    stack = {k: v for k, v in stack_config.items() if k not in _CACHE_IGNORED_KEYS}
    return json.dumps({'stack': stack, 'name': sim['name'], 'feedback': sim['feedback'],
                       'snapshot': snapshot, 'sim_redshift': sim_z}, sort_keys=True)


def _cache_path(cache_dir: str, sim: dict, snapshot: int, sim_z: float) -> Path:
    """Path of the cached stack of one (run, snapshot)."""
    return Path(cache_dir) / f"{sim['name']}_{sim['feedback']}_{snapshot}_z{sim_z:g}.npz"


def load_sim_cache(path: Path, settings: str) -> Optional[dict]:
    """Load a cached stack if it exists and was made with ``settings``.

    Args:
        path: Cache file path.
        settings: Expected settings string (see :func:`_stack_settings`).

    Returns:
        Dict of the cached arrays, or None if the file is missing.

    Raises:
        ValueError: If the file was made with other settings (it is never
            overwritten: it may belong to another config sharing ``cache_dir``).
    """
    if not path.exists():
        print(f"No cached stack at {path}")
        return None
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['settings']) != settings:
            raise ValueError(f"Cached stack {path} was made with other settings; move it or "
                             "set plot.cache_dir to another folder")
        out = {k: archive[k] for k in archive.files if k != 'settings'}
    print(f"Loaded cached stack {path}")
    return out


def save_sim_cache(path: Path, data: dict, settings: str) -> None:
    """Write one cached stack to ``path``, atomically.

    Writes to a temporary file in the same directory and renames it into place,
    so a failed run never leaves a truncated file behind.

    Args:
        path: Destination ``.npz`` path.
        data: Arrays to save.
        settings: Settings string stored alongside (see :func:`_stack_settings`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # The suffix must stay '.npz': np.savez appends '.npz' to any other name.
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp.npz')
    np.savez(tmp, settings=np.array(settings), **data)
    os.replace(tmp, path)
    print(f"Saved stack to {path}")


def stack_sim(stack_config: dict, sim: dict, snapshot: int, sim_z: float,
              verbose: bool = True) -> dict:
    """Stack the numerator and denominator maps of one FLAMINGO (run, snapshot).

    The same stackMap calls as compare_data_ratio.py (Figure 14b).

    Args:
        stack_config: The config's ``stack`` section.
        sim: One entry of the FLAMINGO ``sims`` list (``name``, ``feedback``).
        snapshot: Snapshot number.
        sim_z: Redshift the snapshot is placed at.
        verbose: Whether to print progress.

    Returns:
        Dict with ``radii`` (in rad_distance units), the per-radius means and
        standard errors of the numerator (``mean0``, ``err0``) and denominator
        (``mean1``, ``err1``) over the stacked haloes, ``n_halos``, the SHAM
        sample's ``mean_mass`` (Msun/h) and ``mean_R200m`` (comoving kpc/h),
        ``omega_b``, ``omega_m`` and the snapshot's ``header_redshift``.

    Raises:
        ValueError: If ``sim_z`` is not the snapshot's header redshift.
    """
    t0 = time.time()
    stacker = SimulationStacker(sim['name'], snapshot, z=sim_z, simType='FLAMINGO',
                                feedback=sim['feedback'])
    header_z = float(stacker.header['Redshift'])
    if abs(header_z - sim_z) > 1e-3:
        raise ValueError(f"FLAMINGO {sim['feedback']} snapshot {snapshot} is at z = {header_z}, "
                         f"not at sim_redshift = {sim_z}")
    try:
        omega_b = stacker.header['OmegaBaryon']
    except KeyError:
        omega_b = _OMEGA_B_FLAMINGO_FALLBACK
        print(f"  [warn] OmegaBaryon missing in the header; using fallback {omega_b}")

    # float(): PyYAML reads e.g. 5e-4 (no decimal point) as a string
    halo_abundance_target = stack_config.get('halo_abundance_target', None)
    if halo_abundance_target is not None:
        halo_abundance_target = float(halo_abundance_target)
    halo_mass_upper = float(stack_config.get('halo_mass_upper', 5e14))
    mean_mass, mean_R200m = sham_parent_halo_stats(stacker, halo_abundance_target,
                                                   halo_mass_upper)

    base_kwargs = dict(
        minRadius=stack_config.get('min_radius', 1.0),
        maxRadius=stack_config.get('max_radius', 6.0),
        numRadii=stack_config.get('num_radii', 9),
        projection=stack_config.get('projection', 'yz'),
        save=stack_config.get('save_field', True),
        load=stack_config.get('load_field', True),
        radDistance=stack_config.get('rad_distance', 1.0),
        use_subhalos=stack_config.get('use_subhalos', True),
        halo_abundance_target=halo_abundance_target,
        halo_mass_upper=halo_mass_upper,
    )
    # profiles shape: (n_radii, n_halos)
    radii0, profiles0 = stacker.stackMap(stack_config['particle_type'],
                                         filterType=stack_config['filter_type'],
                                         pixelSize=stack_config['pixel_size'],
                                         beamSize=stack_config['beam_size'],
                                         stampRes=stack_config.get('stamp_res', None),
                                         **base_kwargs)
    radii1, profiles1 = stacker.stackMap(stack_config['particle_type_2'],
                                         filterType=stack_config['filter_type_2'],
                                         pixelSize=stack_config['pixel_size_2'],
                                         beamSize=stack_config['beam_size_2'],
                                         **base_kwargs)
    if not (np.array_equal(radii0, radii1) and profiles0.shape == profiles1.shape):
        raise ValueError("Numerator and denominator stacks have different radii or haloes")

    n_halos = profiles0.shape[1]
    if verbose:
        print(f"  FLAMINGO {sim['feedback']} snapshot {snapshot} (z = {sim_z}): "
              f"N = {n_halos}, mean M = {mean_mass:.3e} Msun/h, "
              f"mean R200m = {mean_R200m:.1f} ckpc/h, {time.time() - t0:.0f} s")
    return dict(
        radii=radii0,
        mean0=np.mean(profiles0, axis=1),
        err0=np.std(profiles0, axis=1) / np.sqrt(n_halos),
        mean1=np.mean(profiles1, axis=1),
        err1=np.std(profiles1, axis=1) / np.sqrt(n_halos),
        n_halos=n_halos,
        mean_mass=mean_mass,
        mean_R200m=mean_R200m,
        omega_b=omega_b,
        omega_m=stacker.header['Omega0'],
        header_redshift=header_z,
    )


def get_sim_stack(stack_config: dict, cache_dir: str, sim: dict, snapshot: int,
                  sim_z: float, plot_only: bool) -> dict:
    """Return the cached stack of one (run, snapshot), stacking it if needed.

    Args:
        stack_config: The config's ``stack`` section.
        cache_dir: Directory of the cached stacks.
        sim: One entry of the FLAMINGO ``sims`` list.
        snapshot: Snapshot number.
        sim_z: Redshift the snapshot is placed at.
        plot_only: If True, never stack.

    Returns:
        The stack dict (see :func:`stack_sim`).

    Raises:
        FileNotFoundError: Under ``plot_only``, if no cache exists.
    """
    path = _cache_path(cache_dir, sim, snapshot, sim_z)
    settings = _stack_settings(stack_config, sim, snapshot, sim_z)
    data = load_sim_cache(path, settings)
    if data is None:
        if plot_only:
            raise FileNotFoundError(f"--plot-only: no cached stack at {path}")
        data = stack_sim(stack_config, sim, snapshot, sim_z)
        save_sim_cache(path, data, settings)
    return data


def sim_fgas(data: dict, rad_distance: float) -> tuple:
    """f_gas^obs of one stack: ratio of means over Omega_b / Omega_m.

    The standard errors of the two means are propagated through the ratio in
    quadrature, as in compare_data_ratio.py.

    Args:
        data: Stack dict (see :func:`stack_sim`).
        rad_distance: Arcmin per unit radius.

    Returns:
        ``(theta_arcmin, fgas, fgas_err)``.
    """
    f_baryon = data['omega_b'] / data['omega_m']
    fgas = data['mean0'] / data['mean1'] / f_baryon
    fgas_err = np.abs(fgas) * np.sqrt((data['err0'] / data['mean0'])**2
                                      + (data['err1'] / data['mean1'])**2)
    return data['radii'] * rad_distance, fgas, fgas_err


def main(path2config: str, only: Optional[list] = None, plot_only: bool = False) -> None:
    """Stack what is needed and make the four-panel data-versus-FLAMINGO figure.

    Args:
        path2config: Path to the YAML configuration file.
        only: ``[feedback, snapshot]``: stack and cache this one (run, snapshot)
            and return without plotting.
        plot_only: Plot from the cached stacks only; never stack.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_config = config['stack']
    data_config = config['data']
    plot_config = config.get('plot', {})
    lens_bins = config['lens_bins']
    rad_distance = stack_config.get('rad_distance', 1.0)
    cache_dir = plot_config.get('cache_dir', '../data/lrg_bins_fgas/')

    if (stack_config['particle_type'], stack_config['particle_type_2']) != ('ionized_gas', 'total'):
        raise ValueError("This figure is f_gas^obs: particle_type 'ionized_gas' over "
                         "particle_type_2 'total'")
    unknown = set(stack_config) - _USED_STACK_KEYS
    if unknown:
        raise ValueError(f"stack keys not used by this script: {sorted(unknown)}")
    sims = []
    for group in config['simulations']:
        if group['sim_type'] != 'FLAMINGO':
            raise ValueError(f"Only FLAMINGO runs are supported, got {group['sim_type']!r}")
        sims.extend(group['sims'])
    snap_z = snapshot_redshifts(lens_bins)

    # ---- One (run, snapshot) only: stack, cache, no figure ----
    if only is not None:
        feedback, snapshot = only[0], int(only[1])
        matches = [s for s in sims if s['feedback'] == feedback]
        if not matches or snapshot not in snap_z:
            raise ValueError(f"--only {feedback} {snapshot}: not a run and snapshot of this config "
                             f"(runs {[s['feedback'] for s in sims]}, snapshots {sorted(snap_z)})")
        get_sim_stack(stack_config, cache_dir, matches[0], snapshot, snap_z[snapshot], plot_only)
        return

    print("Lens bin -> FLAMINGO snapshot:")
    for b in lens_bins:
        print(f"  LRG bin {b['lens_bin']} (z_lens = {b['z_lens']:.3f}, source bin {b['source_bin']}): "
              f"snapshot {b['snapshot']} placed at z = {b['sim_redshift']}")

    stacks = {(s['feedback'], snap): get_sim_stack(stack_config, cache_dir, s, snap, z, plot_only)
              for s in sims for snap, z in snap_z.items()}

    # ---- Output path: figures/<year-month>/<month-day>/<fig_subdir>/ ----
    now = datetime.now()
    fig_path = (Path(plot_config.get('fig_path', '../figures/')) / now.strftime("%Y-%m")
                / now.strftime("%m-%d") / plot_config.get('fig_subdir', ''))
    fig_path.mkdir(parents=True, exist_ok=True)
    out_path = fig_path / f"ratio{plot_config['fig_name']}.{plot_config.get('fig_type', 'pdf')}"

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey='row', sharex='col')
    axes = axes.flatten()
    numbers = []
    cat_type = data_config.get('cat_type', 'LRG')

    for i, b in enumerate(lens_bins):
        ax = axes[i]
        lens_bin, z_lens, source_bin = b['lens_bin'], b['z_lens'], b['source_bin']
        title = f'DESI {cat_type} z bin {lens_bin} (z~{z_lens:.3f})'

        # ---- Data: one source sample, each version ----
        theta_data = None
        for version in data_config['versions']:
            npz_path = version['npz'].format(lens_bin=lens_bin)
            measurement = load_measurements_npz(npz_path)
            key = f'source_bin_{source_bin}'
            if key not in measurement:
                raise KeyError(f"{npz_path} has no {key} (has {sorted(measurement)})")
            d = measurement[key]
            theta, ratio, ratio_err = d['ksz_theta_arcmin'], d['ratio'], d['ratio_err']
            if theta_data is not None and not np.array_equal(theta, theta_data):
                raise ValueError(f"The versions' theta grids differ in lens bin {lens_bin}")
            theta_data = theta
            label = f"Data, {version['label']}"
            ax.errorbar(theta * version.get('x_offset', 1.0), ratio, yerr=ratio_err, fmt='s',
                        c='black', mfc='none' if version.get('open_marker', False) else 'black',
                        capsize=2, label=label, zorder=3)
            figure_data.record(title, label, theta_arcmin=theta, fgas_obs=ratio,
                               fgas_obs_err=ratio_err)
            numbers.append((f"{label}; lens bin {lens_bin}; source bin {source_bin}; {npz_path}",
                            theta, ratio, ratio_err))

        # ---- FLAMINGO ----
        for s in sims:
            data = stacks[(s['feedback'], b['snapshot'])]
            theta, fgas, fgas_err = sim_fgas(data, rad_distance)
            if not np.allclose(theta, theta_data):
                raise ValueError(f"Stack radii {theta} differ from the data's theta {theta_data} "
                                 f"in lens bin {lens_bin}")
            label = f"FLAMINGO {s['feedback']}".replace('_', '-')
            colour = _FLAMINGO_COLOURS.get(s['feedback'], None)
            ax.plot(theta, fgas, color=colour, lw=2, label=label, zorder=2)
            ax.fill_between(theta, fgas - fgas_err, fgas + fgas_err, color=colour, alpha=0.2,
                            lw=0, zorder=1)
            figure_data.record(title, label, theta_arcmin=theta, fgas_obs=fgas,
                               fgas_obs_err=fgas_err)
            numbers.append((f"{label}; lens bin {lens_bin}; snapshot {b['snapshot']} at "
                            f"z = {b['sim_redshift']}; N = {int(data['n_halos'])}; "
                            f"mean parent M = {float(data['mean_mass']):.4e} Msun/h",
                            theta, fgas, fgas_err))

        # ---- Panel cosmetics, as compare_combined_self.py ----
        source_label = 'HSC combined' if source_bin == 0 else f'HSC Bin {source_bin}'
        ax.text(0.04, 0.96, f"{source_label}\nFLAMINGO z = {b['sim_redshift']:.2f}",
                transform=ax.transAxes, ha='left', va='top', fontsize=12)
        ax.axhline(1.0, color='black', linestyle='--')
        if i in (2, 3):
            ax.set_xlabel(r'$\theta$ [arcmin]', fontsize=18)
        if i in (0, 2):
            ax.set_ylabel(r'$\frac{\Delta \Sigma_{\rm kSZ}}{\Delta \Sigma_{\rm lens}} \times '
                          r'\frac{\Omega_m}{\Omega_b}$', fontsize=18)
        ax.set_title(title)
        ax.grid()
        ax.set_ylim(*plot_config.get('ylim', [-0.35, 1.9]))
        # Secondary x-axis in comoving Mpc/h at the lens bin's redshift
        D_C = Planck18.comoving_transverse_distance(z_lens).to(u.Mpc).value
        scale = D_C * np.pi / 180.0 / 60.0 * Planck18.h  # arcmin -> cMpc/h
        secax = ax.secondary_xaxis('top', functions=(lambda x, s=scale: x * s,
                                                     lambda x, s=scale: x / s))
        if i in (0, 1):
            secax.set_xlabel(r'$r_p$ [cMpc/$h$]', fontsize=18)

    # One legend for all panels, outside the grid on the right, data first
    # (get_legend_handles_labels lists lines before errorbar containers)
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda k: not labels[k].startswith('Data'))
    plt.tight_layout()
    fig.legend([handles[k] for k in order], [labels[k] for k in order], loc='center left',
               bbox_to_anchor=(1.0, 0.5), fontsize=13)

    print(f"Saving figure to {out_path}")
    fig.savefig(out_path, bbox_inches='tight')
    figure_data.save(out_path)
    with open(fig_path / f"{out_path.stem}_numbers.txt", 'w') as f:
        for header, theta, value, err in numbers:
            f.write(f"# {header}; columns theta [arcmin], f, error\n")
            for row in zip(theta, value, err):
                f.write(' '.join(f'{x:.10e}' for x in row) + '\n')
    plt.close(fig)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='f_gas^obs in the four DESI LRG bins: v2.1 and v2.2 data with FLAMINGO.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/lensing/dsigma_lensing/ratio_lrg_bins_v2.1_v2.2.yaml',
                        help='Path to the YAML configuration file.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--only', nargs=2, metavar=('FEEDBACK', 'SNAPSHOT'), default=None,
                      help='Stack and cache this one (run, snapshot), then exit without plotting.')
    mode.add_argument('--plot-only', action='store_true',
                      help='Plot from the cached stacks only; never stack.')
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")
    main(**args)
