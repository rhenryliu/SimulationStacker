"""make_pk_suppression.py
=======================
Compute and plot the matter power spectrum suppression

    SP(k) ≈ P_tot(k) / P_dm(k)

for each simulation listed in a standard config YAML.  Uses the
``get_field_baryon_suppression`` method added to ``SimulationStacker``,
which builds 3D density fields via the existing ``makeField`` infrastructure
and computes auto/cross power spectra with Pylians (Pk_library).

P_dm is the auto power spectrum of DM particles in the *same* hydro
simulation, used as a proxy for the dark-matter-only reference.

Usage
-----
    python make_pk_suppression.py -p configs/mass_ratio_data_z05.yaml
    python make_pk_suppression.py -p configs/mass_ratio_data_z05.yaml --grid 256 --threads 32

Config keys consumed
--------------------
    simulations          — list of sims (same schema as all other scripts)
    stack.redshift       — passed as z= to SimulationStacker
    plot.fig_path        — output directory
    plot.fig_name        — base filename (appended with '_pk_suppression')
    plot.fig_type        — file extension  (pdf, png, …)

All other config content is ignored.
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import yaml

sys.path.append('../src/')
from stacker import SimulationStacker

# ---------------------------------------------------------------------------
# Matplotlib style  (identical to other scripts in this repo)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex":      True,
    "mathtext.fontset": "cm",
    "font.size":        20,
    "axes.titlesize":   20,
    "axes.labelsize":   20,
    "xtick.labelsize":  20,
    "ytick.labelsize":  20,
    "legend.fontsize":  13,
})

# Colourmap assigned to each sim type — twilight for TNG, hsv for SIMBA.
_COLOURMAPS = {
    'IllustrisTNG': 'twilight',
    'SIMBA':        'hsv',
}

# OmegaBaryon defaults for sims that don't store it in their header.
_OMEGA_BARYON_ILLUSTRIS_DEFAULT = 0.0456
_OMEGA_BARYON_SIMBA_DEFAULT     = 0.048


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_stacker(sim: dict, sim_type_name: str, redshift: float):
    """Instantiate a SimulationStacker and return metadata.

    Parameters
    ----------
    sim : dict
        Single simulation entry from the YAML ``simulations`` block.
        Must contain ``name`` and ``snapshot``; SIMBA entries also need
        ``feedback``.
    sim_type_name : str
        ``'IllustrisTNG'`` or ``'SIMBA'``.
    redshift : float
        Target simulation redshift.

    Returns
    -------
    stacker : SimulationStacker
    sim_label : str
        Human-readable label for the legend.
    """
    sim_name = sim['name']
    snapshot = sim['snapshot']

    if sim_type_name == 'IllustrisTNG':
        stacker   = SimulationStacker(sim_name, snapshot, z=redshift,
                                      simType=sim_type_name)
        sim_label = sim_name

    elif sim_type_name == 'SIMBA':
        feedback  = sim['feedback']
        stacker   = SimulationStacker(sim_name, snapshot, z=redshift,
                                      simType=sim_type_name,
                                      feedback=feedback)
        sim_label = f"{sim_name} ({feedback})"

    else:
        raise ValueError(f"Unknown simulation type: {sim_type_name!r}")

    return stacker, sim_label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(path2config: str, grid: int, threads: int) -> None:
    """Generate the power spectrum suppression figure.

    Parameters
    ----------
    path2config : str
        Path to the YAML configuration file.
    grid : int
        3D field resolution (grid^3 voxels) passed to
        ``get_field_baryon_suppression``.
    threads : int
        Number of Pylians threads.
    """
    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_cfg = config.get('stack', {})
    plot_cfg  = config.get('plot',  {})

    redshift = stack_cfg.get('redshift', 0.5)

    now       = datetime.now()
    yr_string = now.strftime("%Y-%m")
    dt_string = now.strftime("%m-%d")
    fig_path  = Path(plot_cfg.get('fig_path', '../figures/')) / yr_string / dt_string
    fig_path.mkdir(parents=True, exist_ok=True)

    fig_name = plot_cfg.get('fig_name', 'output')
    fig_type = plot_cfg.get('fig_type', 'pdf')

    # ------------------------------------------------------------------
    # Build a flat list of (sim_type_name, sim_dict) pairs and per-suite
    # colour arrays so that sims within each suite share a colormap.
    # ------------------------------------------------------------------
    suite_entries = []  # list of (sim_type_name, sim_dict, colour)

    for suite in config['simulations']:
        sim_type_name = suite['sim_type']
        sims          = suite['sims']
        cmap_name     = _COLOURMAPS.get(sim_type_name, 'viridis')
        cmap          = matplotlib.colormaps[cmap_name]  # type: ignore
        colours       = cmap(np.linspace(0.2, 0.85, len(sims)))
        for sim, colour in zip(sims, colours):
            suite_entries.append((sim_type_name, sim, colour))

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.axhline(1.0, color='k', ls='--', lw=1.5, zorder=0)
    ax.set_xscale('log')
    ax.set_xlabel(r'$k \; [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$S_P(k) = P_\mathrm{tot}(k)\,/\,P_\mathrm{dm}(k)$')
    ax.set_title(
        r'Matter power spectrum suppression  ($z={:.2f}$)'.format(redshift),
        fontsize=18,
    )
    ax.grid(True, which='both', ls=':', alpha=0.4)

    # ------------------------------------------------------------------
    # Loop over simulations
    # ------------------------------------------------------------------
    t0 = time.time()

    for i, (sim_type_name, sim, colour) in enumerate(suite_entries):
        sim_name = sim['name']
        feedback_str = (f"  feedback={sim.get('feedback')}"
                        if sim_type_name == 'SIMBA' else '')
        print(f"\n[{i+1}/{len(suite_entries)}] {sim_type_name} — "
              f"{sim_name}{feedback_str}")

        stacker, sim_label = setup_stacker(sim, sim_type_name, redshift)

        print(f"  Running get_field_baryon_suppression "
              f"(grid={grid}, threads={threads})...")
        t1 = time.time()
        results = stacker.get_field_baryon_suppression(
            grid=grid, save=False, load=False, threads=threads,
        )
        print(f"  Done in {time.time() - t1:.1f}s")

        k      = results['k']        # h/Mpc
        P_dm   = results['P_dm']
        P_tot  = results['P_tot']
        SP_k   = P_tot / P_dm

        ax.plot(k, SP_k, label=sim_label, color=colour, lw=2)

    # ------------------------------------------------------------------
    # Finalise and save
    # ------------------------------------------------------------------
    ax.legend(loc='lower left', framealpha=0.85)
    fig.tight_layout()

    out_path = fig_path / f"{fig_name}_pk_suppression.{fig_type}"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to: {out_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot matter power spectrum suppression SP(k) for simulations in a config.'
    )
    parser.add_argument(
        '-p', '--config', required=True,
        help='Path to YAML config file (e.g. configs/mass_ratio_data_z05.yaml)',
    )
    parser.add_argument(
        '--grid', type=int, default=512,
        help='3D field grid resolution (default: 512)',
    )
    parser.add_argument(
        '--threads', type=int, default=1,
        help='Number of Pylians threads (default: 1; increase for batch jobs)',
    )
    args = parser.parse_args()
    main(args.config, args.grid, args.threads)
