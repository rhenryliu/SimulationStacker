"""snapshot_comoving_ratio.py

Simulation-only companion to beam_compensated_ratio_v2.py: the stacked gas
fraction f_gas(R) = <DSigma_ionized_gas> / <DSigma_total> x Omega_m/Omega_b of
each simulation, shown side by side for the two lensing snapshots (z ~ 0.26 and
z = 0.5) on the same comoving radial grid.

beam_compensated_ratio_v2.py stacks on a fixed angular grid (1-6 arcmin, with a
0.75 arcmin DSigma annulus), which probes 0.38-2.3 cMpc/h at z = 0.5 but only
0.21-1.28 cMpc/h at z = 0.26, so the two figures cannot be compared radius by
radius.  Here both the radial grid and the annulus width ``dr`` are fixed in
comoving Mpc/h: the cached noBeam maps are loaded exactly as stackMap would,
then stacked with ``stack_on_array(radDistanceUnits='kpc/h')`` using each map's
true comoving pixel size.  Only the pixel resolution then differs between the
snapshots (0.2 arcmin is ~77 kpc/h at z = 0.5 and ~43 kpc/h at z = 0.26).

Everything else -- simulations, snapshots, per-sim redshift overrides (FLAMINGO
z = 0.30 stands in for z = 0.26), SHAM halo selection, pixel sizes, particle
and filter types -- is read from the noBeam configs listed under ``snapshots``,
as beam_compensated_ratio_v2.py reads its noBeam config.  No data are plotted
and no beam compensation is applied.

The (snapshot, simulation) stacks are independent single-core loops, so they run
in a pool of ``n_workers`` processes (1 runs them serially in this process).

Usage
-----
    python lensing/snapshot_comoving_ratio.py -p configs/lensing/snapshot_comoving_ratio.yaml
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import argparse
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

sys.path.append('../src/')
from utils import arcmin_to_comoving, comoving_to_arcmin  # type: ignore
from stacker import SimulationStacker  # type: ignore
from halos import select_halos  # type: ignore
import figure_data  # type: ignore

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore  # noqa: F401 (needed by stacker internals)

# ---------------------------------------------------------------------------
# Matplotlib style — matches beam_compensated_ratio_v2.py
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

_OMEGA_B_TNG_FALLBACK      = 0.0456
_OMEGA_B_SIMBA_FALLBACK    = 0.048
_OMEGA_B_FLAMINGO_FALLBACK = 0.0486  # header provides it; fallback should never trigger

# Fixed colours for the FLAMINGO feedback variants, keyed by feedback name.
# Keep in sync with beam_compensated_ratio_v2.py / compare_data_ratio.py.
_FLAMINGO_COLOURS = {
    'L1_m9':           '#B30000',  # dark red (fiducial)
    'fgas-8sigma':     '#FF7F0E',  # orange
    'Jet_fgas-4sigma': '#C71585',  # magenta
}

# Stacking unit: radii and dr are given in comoving Mpc/h.
_RAD_DISTANCE_KPCH = 1000.0


def _resolve_stacker(sim_type_name: str, sim: dict, redshift: float,
                     verbose: bool) -> tuple:
    """Instantiate a SimulationStacker and resolve Omega_b.

    Keep in sync with beam_compensated_ratio_v2.py.

    Args:
        sim_type_name: ``'IllustrisTNG'``, ``'SIMBA'`` or ``'FLAMINGO'``.
        sim: Single simulation entry from the YAML ``sims`` list.
        redshift: Snapshot redshift.
        verbose: Whether to print warnings.

    Returns:
        ``(stacker, sim_label, omega_b)``
    """
    # Per-sim redshift override: a sim entry may declare its own 'redshift'
    # (e.g. a FLAMINGO z=0.30 snapshot substituted into a z=0.26 comparison);
    # otherwise fall back to the config-level redshift passed in.
    z = sim.get('redshift', redshift)

    if sim_type_name == 'IllustrisTNG':
        stacker   = SimulationStacker(sim['name'], sim['snapshot'],
                                      z=z, simType=sim_type_name)
        sim_label = sim['name']
        try:
            omega_b = stacker.header['OmegaBaryon']
        except KeyError:
            omega_b = _OMEGA_B_TNG_FALLBACK
            if verbose:
                print(f"  [warn] OmegaBaryon missing in {sim_label} header; "
                      f"using fallback {_OMEGA_B_TNG_FALLBACK}")

    elif sim_type_name == 'SIMBA':
        stacker   = SimulationStacker(sim['name'], sim['snapshot'],
                                      z=z, simType=sim_type_name,
                                      feedback=sim['feedback'])
        sim_label = f"SIMBA-100"
        try:
            omega_b = stacker.header['OmegaBaryon']
        except KeyError:
            omega_b = _OMEGA_B_SIMBA_FALLBACK
            if verbose:
                print(f"  [warn] OmegaBaryon missing in {sim_label} header; "
                      f"using fallback {_OMEGA_B_SIMBA_FALLBACK}")

    elif sim_type_name == 'FLAMINGO':
        stacker   = SimulationStacker(sim['name'], sim['snapshot'],
                                      z=z, simType=sim_type_name,
                                      feedback=sim['feedback'])
        # '-' instead of '_' so labels render under usetex
        sim_label = f"FLAMINGO {sim['feedback']}".replace('_', '-')
        try:
            omega_b = stacker.header['OmegaBaryon']
        except KeyError:
            omega_b = _OMEGA_B_FLAMINGO_FALLBACK
            if verbose:
                print(f"  [warn] OmegaBaryon missing in {sim_label} header; "
                      f"using fallback {_OMEGA_B_FLAMINGO_FALLBACK}")

    else:
        raise ValueError(f"Unknown simulation type: {sim_type_name!r}. "
                         "Expected 'IllustrisTNG', 'SIMBA' or 'FLAMINGO'.")

    return stacker, sim_label, omega_b


def sham_parent_halo_stats(stacker: SimulationStacker,
                           halo_abundance_target: Optional[float],
                           halo_mass_upper: float = 5e14) -> tuple:
    """Mean parent-halo mass (Msun/h) and R200m (comoving kpc/h) of the SHAM sample.

    Replicates the subhalo selection performed inside
    ``SimulationStacker.stack_on_array`` (the ``use_subhalos=True`` branch).
    Keep in sync with beam_compensated_ratio_v2.py.

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


def _sim_colours(nb_config: dict) -> dict:
    """Map sim labels to colours, using the colourmap logic of beam_compensated_ratio_v2.py.

    Args:
        nb_config: Parsed noBeam YAML config.

    Returns:
        ``{sim_label: colour}`` for every simulation in the config.
    """
    colour_for_sim: dict = {}
    for i, sim_group in enumerate(nb_config['simulations']):
        cmap    = matplotlib.colormaps[['plasma', 'twilight', 'hot'][i]]  # type: ignore[attr-defined]
        n_sims  = len(sim_group['sims'])
        colours = cmap(np.linspace(0.2, 0.85, n_sims))
        for j, sim in enumerate(sim_group['sims']):
            if sim_group['sim_type'] == 'IllustrisTNG':
                label  = sim['name']
                colour = colours[j]
            elif sim_group['sim_type'] == 'FLAMINGO':
                label  = f"FLAMINGO {sim['feedback']}".replace('_', '-')
                colour = _FLAMINGO_COLOURS.get(sim['feedback'], colours[j])
            else:
                label  = f"SIMBA-100"
                colour = colours[j]
            colour_for_sim[label] = colour
    return colour_for_sim


def _panel_title(nb_config: dict) -> str:
    """Panel title giving the snapshot redshift and any per-sim overrides.

    Args:
        nb_config: Parsed noBeam YAML config.

    Returns:
        TeX title, e.g. ``'$z = 0.26$ (FLAMINGO: $z = 0.3$)'``.
    """
    z_panel   = nb_config['stack'].get('redshift', 0.5)
    overrides: dict = {}
    for sim_group in nb_config['simulations']:
        for sim in sim_group['sims']:
            if 'redshift' in sim and sim['redshift'] != z_panel:
                overrides.setdefault(sim_group['sim_type'], sim['redshift'])
    title = rf'$z = {z_panel}$'
    if overrides:
        title += ' (' + ', '.join(rf'{t}: $z = {z}$' for t, z in overrides.items()) + ')'
    return title


def stack_one(sim_type_name: str, sim: dict, nb_stack: dict, comoving: dict,
              verbose: bool = True) -> dict:
    """Stack one simulation at one snapshot on the comoving radial grid.

    Loads the numerator and denominator maps as stackMap would (same pixel size,
    beam, projection and caching) but stacks them in comoving kpc/h units, so the
    radii and the DSigma annulus width are the same comoving lengths at every
    redshift.

    Args:
        sim_type_name: ``'IllustrisTNG'``, ``'SIMBA'`` or ``'FLAMINGO'``.
        sim: Single simulation entry from the noBeam config's ``sims`` list.
        nb_stack: ``stack`` section of that snapshot's noBeam config.
        comoving: ``comoving`` section of the master config (radii and ``dr``
            in comoving Mpc/h).
        verbose: If True, print progress messages to stdout.

    Returns:
        Dict with ``label``, ``radii`` (cMpc/h), ``ratio`` and ``ratio_err``
        (the Omega_m/Omega_b-normalised profile ratio and its standard error),
        ``n_halos``, ``mean_mass`` (Msun/h), ``R200m`` (cMpc/h), the cosmology
        ``h``, ``Om0``, ``Ob0`` and the wall-clock ``elapsed`` in seconds.
    """
    t0 = time.time()
    stacker, sim_label, omega_b = _resolve_stacker(
        sim_type_name, sim, nb_stack.get('redshift', 0.5), verbose)
    tag = f"[z={stacker.z} {sim_label}]"

    projection            = nb_stack.get('projection', 'yz')
    halo_abundance_target = nb_stack.get('halo_abundance_target', None)
    halo_mass_upper       = nb_stack.get('halo_mass_upper', 5e14)  # stackMap default

    stack_kwargs = dict(
        minRadius        = comoving['min_radius'],
        maxRadius        = comoving['max_radius'],
        numRadii         = comoving['num_radii'],
        dr               = comoving['dr'],
        projection       = projection,
        radDistance      = _RAD_DISTANCE_KPCH,
        radDistanceUnits = 'kpc/h',
        use_subhalos     = nb_stack.get('use_subhalos', False),
        halo_abundance_target = halo_abundance_target,
        halo_mass_upper  = halo_mass_upper,
    )

    # (particle type, filter, pixel size, beam) for the numerator and denominator,
    # with the same defaults as beam_compensated_ratio_v2.py.
    terms = [
        (nb_stack.get('particle_type',   'ionized_gas'), nb_stack.get('filter_type',   'DSigma'),
         nb_stack.get('pixel_size',      0.2),           nb_stack.get('beam_size',     None)),
        (nb_stack.get('particle_type_2', 'total'),       nb_stack.get('filter_type_2', 'DSigma'),
         nb_stack.get('pixel_size_2',    0.2),           nb_stack.get('beam_size_2',   None)),
    ]

    means, errs = [], []
    for pType, filterType, pixelSize, beamSize in terms:
        if verbose:
            print(f"{tag} stacking {pType} ({filterType})")
        map_ = stacker.makeMap(pType, projection=projection, beamSize=beamSize,
                               save=nb_stack.get('save_field', True),
                               load=nb_stack.get('load_field', True),
                               pixelSize=pixelSize,
                               mask=nb_stack.get('mask_haloes', False),
                               maskRad=nb_stack.get('mask_radii', 3.0))
        radii, profiles = stacker.stack_on_array(map_, filterType=filterType, **stack_kwargs)
        del map_
        means.append(np.mean(profiles, axis=1))
        errs.append(np.std(profiles, axis=1) / np.sqrt(profiles.shape[1]))
        n_halos = profiles.shape[1]

    factor    = stacker.header['Omega0'] / omega_b
    ratio     = means[0] / means[1] * factor
    ratio_err = np.abs(ratio) * np.sqrt((errs[0] / means[0])**2 + (errs[1] / means[1])**2)

    mean_mass, R200m_kpch = sham_parent_halo_stats(
        stacker, halo_abundance_target, halo_mass_upper)
    if verbose:
        print(f"{tag} {n_halos} haloes, mean M = {mean_mass:.3e} Msun/h "
              f"(log10 = {np.log10(mean_mass):.3f}), "
              f"mean R200m = {R200m_kpch / 1e3:.3f} cMpc/h, "
              f"done in {time.time() - t0:.0f} s")

    return dict(
        label     = sim_label,
        radii     = radii,
        ratio     = ratio,
        ratio_err = ratio_err,
        n_halos   = n_halos,
        mean_mass = mean_mass,
        R200m     = R200m_kpch / 1e3,
        h         = stacker.header['HubbleParam'],
        Om0       = stacker.header['Omega0'],
        Ob0       = omega_b,
        elapsed   = time.time() - t0,
    )


def main(path2config: str, verbose: bool = True) -> None:
    """Stack every simulation at both snapshots and save the side-by-side figure.

    Args:
        path2config: Path to the master YAML configuration file.
        verbose: If True, print progress messages to stdout.
    """
    config_dir = Path(path2config).parent

    with open(path2config) as f:
        master = yaml.safe_load(f)

    comoving    = master['comoving']
    plot_config = master.get('plot', {})
    n_workers   = master.get('n_workers', 1)

    nb_configs = []
    for entry in master['snapshots']:
        with open(config_dir / entry['no_beam_config']) as f:
            nb_configs.append(yaml.safe_load(f))

    # Every panel must show the same quantity.
    quantity_keys = ('particle_type', 'particle_type_2', 'filter_type', 'filter_type_2')
    quantity = [tuple(nb['stack'].get(k) for k in quantity_keys) for nb in nb_configs]
    if len(set(quantity)) > 1:
        raise ValueError(f"noBeam configs disagree on {quantity_keys}: {quantity}")
    pType  = nb_configs[0]['stack'].get('particle_type',   'ionized_gas')
    pType2 = nb_configs[0]['stack'].get('particle_type_2', 'total')
    # Exported-data column for the plotted fraction, e.g. 'fgas' or 'fbaryon'.
    y_name = {'ionized_gas': 'fgas', 'baryon': 'fbaryon'}.get(pType, f'f_{pType}')

    # ---- Output path: figures/<year-month>/<month-day>/ ----
    now      = datetime.now()
    fig_path = (
        Path(plot_config.get('fig_path', '../figures/'))
        / now.strftime("%Y-%m")
        / now.strftime("%m-%d")
    )
    fig_path.mkdir(parents=True, exist_ok=True)

    fig_name        = plot_config.get('fig_name', 'z026_z05')
    fig_type        = plot_config.get('fig_type', 'pdf')
    plot_error_bars = plot_config.get('plot_error_bars', True)
    do_plot_r200m   = plot_config.get('plot_r200m', True)

    t0 = time.time()

    # ==========================================================================
    # Stack every (snapshot, simulation) pair
    # ==========================================================================
    # tasks[k] = (panel index, sim type, sim entry, noBeam stack section),
    # in config order so the legend order matches beam_compensated_ratio_v2.py.
    tasks = [(i, sim_group['sim_type'], sim, nb['stack'])
             for i, nb in enumerate(nb_configs)
             for sim_group in nb['simulations']
             for sim in sim_group['sims']]

    results: dict = {}
    n_workers = max(1, min(n_workers, len(tasks)))
    if n_workers == 1:
        for k, (_, sim_type_name, sim, nb_stack) in enumerate(tasks):
            results[k] = stack_one(sim_type_name, sim, nb_stack, comoving, verbose)
    else:
        print(f"Stacking {len(tasks)} (snapshot, simulation) pairs on {n_workers} processes")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(stack_one, sim_type_name, sim, nb_stack, comoving, verbose): k
                       for k, (_, sim_type_name, sim, nb_stack) in enumerate(tasks)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()

    # ==========================================================================
    # Figure: one panel per snapshot, shared comoving x and shared y
    # ==========================================================================
    n_panels  = len(nb_configs)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 8),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]

    for i, (ax, nb) in enumerate(zip(axes, nb_configs)):
        z_panel        = nb['stack'].get('redshift', 0.5)
        title          = _panel_title(nb)
        colour_for_sim = _sim_colours(nb)
        panel_results  = [(tasks[k][1], results[k]) for k in range(len(tasks)) if tasks[k][0] == i]

        print(f"\n{figure_data.plain(title)}  [{y_name} at R (cMpc/h) = "
              f"{np.array2string(panel_results[0][1]['radii'], precision=3)}]")

        R200m_ref: Optional[tuple] = None
        for sim_type_name, res in panel_results:
            colour = colour_for_sim[res['label']]
            ax.plot(res['radii'], res['ratio'],
                    label=res['label'], color=colour, lw=2, marker='o', ls='-')
            if plot_error_bars:
                ax.fill_between(res['radii'],
                                res['ratio'] - res['ratio_err'],
                                res['ratio'] + res['ratio_err'],
                                color=colour, alpha=0.2)
            figure_data.record(figure_data.plain(title), res['label'], R_cMpch=res['radii'], **{
                y_name: res['ratio'],
                f'{y_name}_err': res['ratio_err'] if plot_error_bars else None})

            if R200m_ref is None and sim_type_name == 'IllustrisTNG':
                R200m_ref = (res['R200m'], res['label'])
            print(f"  {res['label']:28s} N = {res['n_halos']:7d}  "
                  f"{y_name} = {np.array2string(res['ratio'], precision=3)}")

        # Vertical dotted line at mean comoving R200m of the first IllustrisTNG sim's halos.
        if do_plot_r200m and R200m_ref is not None:
            ax.axvline(R200m_ref[0], color='gray', ls=':', lw=2,
                       label=rf'$\langle R_{{200\mathrm{{m}}}} \rangle$ ({R200m_ref[1]})')

        # Top axis in arcmin at this panel's redshift, using the first sim's
        # cosmology as beam_compensated_ratio_v2.py does.
        ref       = panel_results[0][1]
        cosmo_ref = FlatLambdaCDM(H0=100 * ref['h'], Om0=ref['Om0'],
                                  Tcmb0=2.7255 * u.K, Ob0=ref['Ob0'])
        secax_x = ax.secondary_xaxis(
            'top',
            functions=(
                lambda mpc_h, z=z_panel, c=cosmo_ref: comoving_to_arcmin(mpc_h * 1e3, z, c),
                lambda arcmin, z=z_panel, c=cosmo_ref: arcmin_to_comoving(arcmin, z, c) / 1e3,
            ),
        )
        secax_x.set_xlabel(r'$\theta$ [arcmin]')

        ax.axhline(1.0, color='k', ls='--', lw=1.5, label='_nolegend_')
        ax.set_title(title)
        ax.set_xlabel(r'$R$ [comoving Mpc/$h$]')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(plot_config.get('ylabel', r'$f_{\rm gas}(R)$'))
    axes[0].set_xlim(0.0, comoving['max_radius'] + 0.2)
    axes[0].legend(loc='best')

    fig.tight_layout()

    out_stem = f'snapshot_comoving_{pType}_{pType2}_{fig_name}'
    out_path = fig_path / f'{out_stem}.{fig_type}'
    print(f'\nSaving figure to {out_path}')
    fig.savefig(out_path, dpi=150)  # type: ignore
    figure_data.save(out_path)
    plt.close(fig)

    print(f'Done. Elapsed: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot simulated gas fractions at two snapshots side by side on a common comoving radial grid.',
    )
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/lensing/snapshot_comoving_ratio.yaml',
        help='Path to the master YAML configuration file.',
    )
    args = vars(parser.parse_args())
    print(f"Config: {args['path2config']}")
    main(**args)
