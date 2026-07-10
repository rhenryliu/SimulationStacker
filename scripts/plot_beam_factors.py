"""plot_beam_factors.py

Plot the beam suppression factor for each simulation at both redshifts on a
single axes, to demonstrate that the factor is effectively simulation-independent.

The beam suppression factor is defined as:

    beam_factor_i(r) = <DSigma_ionized_gas(beamed)>(r)
                       / <DSigma_ionized_gas(no beam)>(r)

where "beamed" uses the ACT pixel/beam settings (0.5 / 1.6 arcmin) and
"no beam" uses a fine resolution with no smoothing (0.2 arcmin, no beam).

Simulations are distinguished by colour (same colourmap as compare_data_ratio.py).
Redshifts are distinguished by linestyle (solid = z=0.5, dashed = z=0.26).

Usage
-----
    python plot_beam_factors.py
    python plot_beam_factors.py --config-z05 configs/mass_ratio_beamTest_z05.yaml \\
                                --config-z026 configs/mass_ratio_beamTest_z026.yaml
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml
import argparse

sys.path.append('../src/')
from stacker import SimulationStacker  # type: ignore
from halos import select_halos  # type: ignore

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore  # noqa: F401 (needed by stacker internals)

# ---------------------------------------------------------------------------
# Matplotlib style — matches compare_data_ratio.py exactly
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

_OMEGA_B_TNG_FALLBACK   = 0.0456
_OMEGA_B_SIMBA_FALLBACK = 0.048

# Fixed colours for the FLAMINGO feedback variants, keyed by feedback name.
# The 'hot' colormap previously used gave Jet_fgas-4sigma a pale yellow that is
# illegible in print. Keep in sync with compare_data_ratio.py /
# beam_compensated_ratio_v2.py.
_FLAMINGO_COLOURS = {
    'L1_m9':           '#B30000',  # dark red (fiducial)
    'fgas-8sigma':     '#FF7F0E',  # orange
    'Jet_fgas-4sigma': '#C71585',  # magenta
}

# Redshift label and linestyle for each config slot.
_REDSHIFT_STYLES = {
    'z05':  {'ls': '-',  'label': r'$z = 0.5$'},
    'z026': {'ls': '--', 'label': r'$z = 0.26$'},
}


def sham_parent_halo_stats(stacker: SimulationStacker,
                           halo_abundance_target: Optional[float],
                           halo_mass_upper: float = 5e14) -> tuple:
    """Mean parent-halo mass (Msun/h) and R200m (comoving kpc/h) of the SHAM sample.

    Replicates the subhalo selection performed inside
    ``SimulationStacker.stack_on_array`` (the ``use_subhalos=True`` branch):
    subhalos are abundance-matched by stellar mass within a parent-mass
    pre-filter, then the GroupMass and GroupRad of their parent FoF groups are
    averaged.  ``halo_mass_upper`` defaults to the ``stackMap`` default (5e14)
    since the beamTest configs do not set it.

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


def main(config_z05: str, config_z026: str, verbose: bool = True) -> None:
    """Compute and plot beam suppression factors for both redshifts.

    Args:
        config_z05:  Path to the beamTest config for z=0.5.
        config_z026: Path to the beamTest config for z=0.26.
        verbose: If True, print progress messages to stdout.
    """
    configs = {
        'z05':  config_z05,
        'z026': config_z026,
    }

    # ---- Load both YAML configs ----
    loaded = {}
    for key, path in configs.items():
        with open(path) as f:
            loaded[key] = yaml.safe_load(f)

    # ---- Build sim_label → colour mapping from the z05 config ----
    # Both configs list the same simulations, so building from either is equivalent.
    colour_for_sim: dict = {}
    for i, sim_group in enumerate(loaded['z05']['simulations']):
        cmap    = matplotlib.colormaps[['plasma', 'twilight', 'hot'][i]]  # type: ignore[attr-defined]
        n_sims  = len(sim_group['sims'])
        colours = cmap(np.linspace(0.2, 0.85, n_sims))
        for j, sim in enumerate(sim_group['sims']):
            if sim_group['sim_type'] == 'IllustrisTNG':
                label  = sim['name']
                colour = colours[j]
            elif sim_group['sim_type'] == 'FLAMINGO':
                # '-' instead of '_' so labels render under usetex
                label  = f"FLAMINGO {sim['feedback']}".replace('_', '-')
                colour = _FLAMINGO_COLOURS.get(sim['feedback'], colours[j])
            else:
                # label = f"{sim['name']}_{sim['feedback']}"
                label  = f"SIMBA-100"
                colour = colours[j]
            colour_for_sim[label] = colour

    # ---- Figure output path (taken from z05 plot config as reference) ----
    plot_config = loaded['z05'].get('plot', {})
    now      = datetime.now()
    fig_path = (
        Path(plot_config.get('fig_path', '../figures/'))
        / now.strftime("%Y-%m")
        / now.strftime("%m-%d")
    )
    fig_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    t0 = time.time()

    # ==========================================================================
    # Loop over redshifts and simulations
    # ==========================================================================
    for z_key, config in loaded.items():
        stack = config['stack']
        style = _REDSHIFT_STYLES[z_key]

        redshift     = stack.get('redshift',        0.5)
        rad_distance = stack.get('rad_distance',    1.0)
        pType        = stack.get('particle_type',   'ionized_gas')
        filter_type  = stack.get('filter_type',     'DSigma')
        pixel_size   = stack.get('pixel_size',      0.5)
        beam_size    = stack.get('beam_size',        1.6)
        pType2       = stack.get('particle_type_2', 'ionized_gas')
        filter_type2 = stack.get('filter_type_2',   'DSigma')
        pixel_size_2 = stack.get('pixel_size_2',    0.2)
        beam_size_2  = stack.get('beam_size_2',     None)

        base_kwargs = dict(
            minRadius    = stack.get('min_radius',   1.0),
            maxRadius    = stack.get('max_radius',   6.0),
            numRadii     = stack.get('num_radii',    9),
            projection   = stack.get('projection',   'yz'),
            save         = stack.get('save_field',   True),
            load         = stack.get('load_field',   True),
            radDistance  = rad_distance,
            mask         = stack.get('mask_haloes',  False),
            maskRad      = stack.get('mask_radii',   3.0),
            use_subhalos = stack.get('use_subhalos', False),
            halo_abundance_target = stack.get('halo_abundance_target', None),
        )

        for sim_group in config['simulations']:
            sim_type_name = sim_group['sim_type']
            for sim in sim_group['sims']:
                # Per-sim redshift override: a sim entry may declare its own
                # 'redshift' (e.g. a FLAMINGO z=0.30 snapshot substituted into the
                # z=0.26 slot); otherwise fall back to this config's redshift.
                sim_z = sim.get('redshift', redshift)
                if sim_type_name == 'IllustrisTNG':
                    stacker   = SimulationStacker(sim['name'], sim['snapshot'],
                                                  z=sim_z, simType=sim_type_name)
                    sim_label = sim['name']
                elif sim_type_name == 'SIMBA':
                    stacker   = SimulationStacker(sim['name'], sim['snapshot'],
                                                  z=sim_z, simType=sim_type_name,
                                                  feedback=sim['feedback'])
                    # sim_label = f"{sim['name']}_{sim['feedback']}"
                    sim_label = f"SIMBA-100"
                elif sim_type_name == 'FLAMINGO':
                    stacker   = SimulationStacker(sim['name'], sim['snapshot'],
                                                  z=sim_z, simType=sim_type_name,
                                                  feedback=sim['feedback'])
                    sim_label = f"FLAMINGO {sim['feedback']}".replace('_', '-')
                else:
                    raise ValueError(f"Unknown sim type: {sim_type_name!r}")

                if verbose:
                    print(f"[{z_key}] Processing {sim_label} (z={sim_z})")

                # Mean parent-halo mass and R200m of the SHAM-selected sample.
                mean_mass, R200m_kpch = sham_parent_halo_stats(
                    stacker, stack.get('halo_abundance_target', None))
                if verbose:
                    print(f"  [{z_key}] {sim_label}: mean M = {mean_mass:.3e} Msun/h "
                          f"(log10 = {np.log10(mean_mass):.3f}), "
                          f"mean R200m = {R200m_kpch:.3f} comoving kpc/h")

                radii_b, profiles_b = stacker.stackMap(
                    pType,  filterType=filter_type,
                    pixelSize=pixel_size, beamSize=beam_size,
                    **base_kwargs)
                radii_n, profiles_n = stacker.stackMap(
                    pType2, filterType=filter_type2,
                    pixelSize=pixel_size_2, beamSize=beam_size_2,
                    **base_kwargs)

                # Ratio of halo-means — no baryon normalisation since pType == pType2.
                mean_b = np.mean(profiles_b, axis=1)
                mean_n = np.mean(profiles_n, axis=1)
                beam_factor = mean_b / mean_n

                # Error: standard error of the mean propagated through the ratio.
                err_b = np.std(profiles_b, axis=1) / np.sqrt(profiles_b.shape[1])
                err_n = np.std(profiles_n, axis=1) / np.sqrt(profiles_n.shape[1])
                beam_factor_err = beam_factor * np.sqrt(
                    (err_b / mean_b)**2 + (err_n / mean_n)**2)

                colour = colour_for_sim[sim_label]
                theta  = radii_b * rad_distance

                ax.plot(theta, beam_factor,
                        color=colour, lw=2, ls=style['ls'], marker='o')
                ax.fill_between(
                    theta,
                    beam_factor - beam_factor_err,
                    beam_factor + beam_factor_err,
                    color=colour, alpha=0.15,
                )

    # ==========================================================================
    # Legend: coloured entries for sims, grey entries for redshift linestyles
    # ==========================================================================
    handles = []
    for sim_label, colour in colour_for_sim.items():
        handles.append(Line2D([0], [0], color=colour, lw=2, ls='-', label=sim_label))
    for style in _REDSHIFT_STYLES.values():
        handles.append(Line2D([0], [0], color='gray', lw=2,
                               ls=style['ls'], label=style['label']))
    ax.legend(handles=handles, loc='lower right')

    # ==========================================================================
    # Axes cosmetics
    # ==========================================================================
    ax.axhline(1.0, color='k', ls='--', lw=1.5, label='_nolegend_')
    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(
        r'$\langle \Delta\Sigma_{\rm kSZ}^{\rm beamed} \rangle'
        r'\,/\,'
        r'\langle \Delta\Sigma_{\rm kSZ}^{\rm no\,beam} \rangle$'
    )
    max_radius   = loaded['z05']['stack'].get('max_radius', 6.0)
    rad_distance = loaded['z05']['stack'].get('rad_distance', 1.0)
    ax.set_xlim(0.0, max_radius * rad_distance + 0.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = fig_path / 'beam_factors_all_sims.pdf'
    print(f'Saving figure to {out_path}')
    fig.savefig(out_path, dpi=150) # type: ignore[union-attr] (Path has no `savefig` method, but fig does)
    plt.close(fig)

    print(f'Done. Elapsed: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot beam suppression factors for all simulations at z=0.5 and z=0.26.',
    )
    parser.add_argument(
        '--config-z05',
        type=str,
        default='./configs/mass_ratio_beamTest_z05.yaml',
        help='Path to the beamTest config for z=0.5.',
    )
    parser.add_argument(
        '--config-z026',
        type=str,
        default='./configs/mass_ratio_beamTest_z026.yaml',
        help='Path to the beamTest config for z=0.26.',
    )
    args = vars(parser.parse_args())
    main(**args)
