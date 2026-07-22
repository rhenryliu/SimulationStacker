"""demo_stack.py

Minimal, self-contained demo of halo stacking with SimulationStacker.

This is the "hello world" of the repo: it stacks one particle type on selected
halos with one filter (default: CAP) and produces a mean radial profile for each
configured simulation (TNG300-1, SIMBA, FLAMINGO). It is intentionally simple and
formatted like the other scripts in this folder (e.g. compare_data_ratio.py) so it
also serves as a template to copy from. For the richer, paper-figure workflows see
make_ratios3x2.py and friends.

Two stacking modes, selected by ``stack.mode`` in the config:
    - 'field': raw projected field, native units, radii in comoving kpc/h, no beam
      (simplest and fastest -- the default).
    - 'map':   beam-convolved map, radii in arcmin (closer to an observation).

Data location is resolved (in order) from ``stack.data_root`` in the config, the
``SIMSTACK_DATA_ROOT`` environment variable, or the built-in NERSC default; leave
all unset on NERSC, or set one to run where the data lives elsewhere.

Outputs (written to figures/<year-month>/<month-day>/):
    - a figure overlaying the mean stacked profile of each simulation;
    - an .npz holding [radii, mean, err] per simulation;
    - the same numbers printed to stdout.

Usage
-----
    python demo_stack.py -p configs/demo_stack.yaml
"""

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml

sys.path.append('../src/')
from stacker import SimulationStacker  # type: ignore

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore  # noqa: F401 (required for IllustrisTNG loading)

# ---------------------------------------------------------------------------
# Matplotlib style: Computer Modern / LaTeX-compatible serif fonts.
# Set "text.usetex" to False below if you do not have a LaTeX installation.
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.size": 18,
    "axes.titlesize": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
})


def _as_float(x: Optional[float]) -> Optional[float]:
    """Coerce a config value to float, passing None through unchanged.

    PyYAML parses unsigned scientific notation (e.g. ``5e14``, ``1.665e13``) as a
    *string* rather than a float -- a well-known gotcha. Casting here means the
    demo works regardless of how the user writes such numbers.

    Args:
        x: A number, a numeric string, or None.

    Returns:
        The value as a float, or None if x is None.
    """
    return None if x is None else float(x)


def _tex(s: str) -> str:
    """Escape underscores so raw names render as text under ``text.usetex``.

    Args:
        s: A label such as a particle type or simulation name.

    Returns:
        The label with '_' escaped to '\\_'.
    """
    return s.replace('_', r'\_')


def stack_one_simulation(sim_type: str, sim: dict, stack_cfg: dict,
                         verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a SimulationStacker for one simulation and return its stacked profile.

    Args:
        sim_type: Simulation suite, one of 'IllustrisTNG', 'SIMBA', 'FLAMINGO'.
        sim: Per-simulation config entry with keys 'name', 'snapshot', and (for
            SIMBA/FLAMINGO) 'feedback'. May carry a per-sim 'redshift' override.
        stack_cfg: The ``stack`` section of the config.
        verbose: If True, print progress. Defaults to True.

    Returns:
        Tuple ``(radii, mean, err)``: the 1D aperture radii (in units of
        rad_distance), the halo-averaged profile, and the standard error on the
        mean across halos.
    """
    mode = stack_cfg.get('mode', 'field')
    # Per-sim redshift override (e.g. substituting a nearby snapshot); otherwise
    # fall back to the config-level redshift.
    z = sim.get('redshift', stack_cfg.get('redshift', 0.0))

    # Construct the stacker. sim_root=None lets SimulationStacker resolve the data
    # root from SIMSTACK_DATA_ROOT / the default; a non-null data_root overrides it.
    stacker = SimulationStacker(
        sim=sim['name'],
        snapshot=sim['snapshot'],
        simType=sim_type,
        feedback=sim.get('feedback'),           # None for IllustrisTNG
        z=z,
        nPixels=stack_cfg.get('n_pixels', 1000),
        sim_root=stack_cfg.get('data_root'),    # None -> env var / default
    )

    pType = stack_cfg.get('particle_type', 'gas')
    filterType = stack_cfg.get('filter_type', 'CAP')

    # --- Halo selection: which sample do we stack on? --------------------------
    # 'sham' (default, more realistic): SubHalo Abundance Matching. Stack on
    #     GALAXIES (subhalos), selecting the most massive by stellar mass down to
    #     a target number density -- the physically correct choice, since real
    #     kSZ/lensing measurements stack on galaxies, not halo centres. Triggered
    #     by use_subhalos=True; subhalos are pre-filtered by parent halo mass
    #     (<= halo_mass_upper), then abundance-matched to halo_abundance_target.
    # 'massive' (halo centres): stack on the most massive FoF halo CENTRES whose
    #     cumulative average mass matches halo_mass_avg (capped at halo_mass_upper).
    hsel = stack_cfg.get('halo_selection', {})
    method = hsel.get('method', 'sham')
    mass_upper = _as_float(hsel.get('mass_upper', 5e14))
    if method == 'sham':
        halo_kwargs = dict(
            use_subhalos=True,
            halo_abundance_target=_as_float(hsel.get('abundance_target', 5e-4)),
            halo_mass_upper=mass_upper,   # parent-halo-mass pre-filter
        )
    elif method == 'massive':
        halo_kwargs = dict(
            use_subhalos=False,
            halo_mass_avg=_as_float(hsel.get('mass_avg', 10 ** 13.22)),
            halo_mass_upper=mass_upper,
        )
    else:
        raise ValueError(
            f"halo_selection.method must be 'sham' or 'massive', got: {method!r}")
    if verbose:
        print(f"  halo selection: {method}")

    # Shared stacking arguments (halo-selection kwargs added per method above).
    common = dict(
        pType=pType,
        filterType=filterType,
        projection=stack_cfg.get('projection', 'xy'),
        save=stack_cfg.get('save_field', True),
        load=stack_cfg.get('load_field', True),
        subtract_mean=stack_cfg.get('subtract_mean', False),
        **halo_kwargs,
    )

    if mode == 'field':
        p = stack_cfg.get('field', {})
        radii, profiles = stacker.stackField(
            minRadius=_as_float(p.get('min_radius', 0.1)),
            maxRadius=_as_float(p.get('max_radius', 6.0)),
            numRadii=int(p.get('num_radii', 15)),
            radDistance=_as_float(p.get('rad_distance', 1000.0)),  # kpc/h per radial unit
            **common,
        )
    elif mode == 'map':
        p = stack_cfg.get('map', {})
        radii, profiles = stacker.stackMap(
            minRadius=_as_float(p.get('min_radius', 0.5)),
            maxRadius=_as_float(p.get('max_radius', 6.0)),
            numRadii=int(p.get('num_radii', 12)),
            radDistance=_as_float(p.get('rad_distance', 1.0)),     # arcmin per radial unit
            beamSize=_as_float(p.get('beam_size', 1.6)),
            pixelSize=_as_float(p.get('pixel_size', 0.5)),
            z=z,
            **common,
        )
    else:
        raise ValueError(f"stack.mode must be 'field' or 'map', got: {mode!r}")

    # profiles has shape (num_radii, n_halos): average over the halo axis.
    if profiles.ndim < 2 or profiles.shape[1] == 0:
        raise RuntimeError(
            f"No halos selected for {sim_type} {sim['name']}; check the halo "
            "mass selection (halo_mass_avg / halo_mass_upper) in the config.")
    mean = profiles.mean(axis=1)
    err = profiles.std(axis=1) / np.sqrt(profiles.shape[1])

    if verbose:
        print(f"  -> stacked {profiles.shape[1]} halos, "
              f"{profiles.shape[0]} radial bins")

    return np.asarray(radii), mean, err


def main(path2config: str, verbose: bool = True) -> None:
    """Run the demo stack for every simulation in the config and save outputs.

    Args:
        path2config: Path to the YAML configuration file.
        verbose: If True, print progress. Defaults to True.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_cfg = config.get('stack', {})
    plot_cfg = config.get('plot', {})
    mode = stack_cfg.get('mode', 'field')
    redshift = stack_cfg.get('redshift', 0.5)
    pType = stack_cfg.get('particle_type', 'gas')
    filterType = stack_cfg.get('filter_type', 'CAP')
    plot_error_bars = plot_cfg.get('plot_error_bars', True)

    # x-axis units and scaling differ by mode.
    if mode == 'field':
        rad_distance = _as_float(stack_cfg.get('field', {}).get('rad_distance', 1000.0))
        # radii are in units of rad_distance (kpc/h). Show Mpc/h for readability.
        x_scale = rad_distance / 1000.0
        x_label = r'$R\ [\mathrm{Mpc}/h]$'
    else:
        rad_distance = _as_float(stack_cfg.get('map', {}).get('rad_distance', 1.0))
        x_scale = rad_distance
        x_label = r'$\theta\ [\mathrm{arcmin}]$'

    # ---- Output path: figures/<year-month>/<month-day>/ ----
    now = datetime.now()
    fig_path = (
        Path(plot_cfg.get('fig_path', '../figures/'))
        / now.strftime("%Y-%m")
        / now.strftime("%m-%d")
    )
    fig_path.mkdir(parents=True, exist_ok=True)
    fig_name = plot_cfg.get('fig_name', 'demo_stack')
    fig_type = plot_cfg.get('fig_type', 'png')

    fig, ax = plt.subplots(figsize=(10, 8))
    results = {}  # npz key -> [x, mean, err], saved alongside the figure
    t0 = time.time()

    for sim_group in config['simulations']:
        sim_type = sim_group['sim_type']
        for sim in sim_group['sims']:
            name = sim['name']
            fb = sim.get('feedback')
            has_fb = bool(fb) and fb != name
            # Human-readable label for the legend/stdout; underscore-joined key
            # for the .npz (e.g. 'SIMBA__m100n1024_s50'), easy to reference later.
            disp_label = f"{name} ({fb})" if has_fb else name
            key_label = f"{name}_{fb}" if has_fb else name
            if verbose:
                print(f"Stacking {sim_type}: {disp_label} ...")

            radii, mean, err = stack_one_simulation(
                sim_type, sim, stack_cfg, verbose=verbose)
            x = radii * x_scale

            ax.plot(x, mean, lw=2, label=_tex(f"{sim_type}: {disp_label}"))
            if plot_error_bars:
                ax.fill_between(x, mean - err, mean + err, alpha=0.25)

            # Print and stash the profile.
            print(f"  {disp_label}: profile ({filterType}, {pType})")
            for xi, mi, ei in zip(x, mean, err):
                print(f"    R={xi:10.4g}   value={mi:14.6e} +/- {ei:.2e}")
            results[f"{sim_type}__{key_label}"] = np.vstack([x, mean, err])

    ax.set_xlabel(x_label)
    ax.set_ylabel(f'{filterType}  [{_tex(pType)}]')
    ax.set_title(f'Demo stack: {_tex(pType)}, {filterType} filter, {mode} mode')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()

    # ---- Save figure and the raw profiles ----
    out_stem = f'{fig_name}_{pType}_{filterType}_{mode}_z{redshift}'
    fig_file = fig_path / f'{out_stem}.{fig_type}'
    npz_file = fig_path / f'{out_stem}.npz'
    print(f'Saving figure to {fig_file}')
    fig.savefig(fig_file, dpi=150)
    plt.close(fig)
    np.savez(npz_file, **results)

    print(f'Saved profiles to {npz_file}')
    print(f'Done. Elapsed: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Minimal halo-stacking demo for SimulationStacker.',
    )
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/demo_stack.yaml',
        help='Path to the YAML configuration file.',
    )
    args = vars(parser.parse_args())
    print(f"Config: {args['path2config']}")
    main(**args)
