"""Stacked kSZ profiles on the UNMASKED maps, one panel per simulation suite.

This is the "No Masking" column of ``simulated_kSZ_masked.py`` on its own,
laid out as a single row of panels instead of a column of a 4-wide grid: same
config file, same stacking parameters, same halo selection, same halo-sample
report. Run it when the masking comparison is not needed and only the
unmasked profiles are wanted.

Run from the scripts/ directory:
    python unbound_gas/simulated_kSZ.py -p configs/unbound_gas/tau_z05_CAP_masked_flamingo.yaml
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import matplotlib
import matplotlib.cm as cm

import time

from astropy.cosmology import FlatLambdaCDM, Planck18
import astropy.units as u

# Import packages

sys.path.append('../src/')
from stacker import SimulationStacker
from utils import arcmin_to_comoving, comoving_to_arcmin
# Sibling module in this directory (Python puts the running script's own
# directory on sys.path); must come after the '../src/' append above.
from halo_stats import sample_stats, format_stats, format_table, write_stats_file

sys.path.append('../../illustrisPython/')
import illustris_python as il # type: ignore

import yaml
import argparse
from pathlib import Path
from datetime import datetime


matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": True,
    "mathtext.fontset": "cm",
    # Base font sizes (adjust as desired)
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
})

# Fixed colours for the FLAMINGO feedback variants, keyed by feedback name.
# Kept in sync with simulated_kSZ_masked.py / compare_data_ratio.py so the
# same simulation is the same colour across every figure in the paper.
_FLAMINGO_COLOURS = {
    'L1_m9':           '#B30000',  # dark red (fiducial)
    'fgas-8sigma':     '#FF7F0E',  # orange
    'Jet_fgas-4sigma': '#C71585',  # magenta
}


def main(path2config, verbose=True):
    """Stack the unmasked maps and save the one-row comparison figure.

    Args:
        path2config (str): Path to the configuration file.
        verbose (bool, optional): If True, prints detailed information. Defaults to True.

    Raises:
        ValueError: If the configuration file names an unknown simulation type.
    """

    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_config = config.get('stack', {})
    plot_config = config.get('plot', {})

    # Stacking parameters
    redshift = stack_config.get('redshift', 0.5)
    filterType = stack_config.get('filter_type', 'CAP')
    loadField = stack_config.get('load_field', True)
    saveField = stack_config.get('save_field', True)
    radDistance = stack_config.get('rad_distance', 1.0)
    pType = stack_config.get('particle_type', 'tau')
    projection = stack_config.get('projection', 'xy')
    use_subhalos = stack_config.get('use_subhalos', False)

    pixelSize = stack_config.get('pixel_size', 0.5) # in arcmin

    # Halo-selection parameters. Defaults match stackMap's own defaults, so a
    # config that sets none of these keeps its existing sample.
    halo_abundance_target = stack_config.get('halo_abundance_target', 5e-4)
    halo_mass_avg = stack_config.get('halo_mass_avg', 10 ** (13.22))
    halo_mass_upper = stack_config.get('halo_mass_upper', 5 * 10 ** (14))

    # Plotting parameters
    now = datetime.now()
    yr_string = now.strftime("%Y-%m")
    dt_string = now.strftime("%m-%d")

    figPath = Path(plot_config.get('fig_path')) / yr_string / dt_string
    figPath.mkdir(parents=True, exist_ok=True)
    plotErrorBars = plot_config.get('plot_error_bars', True)
    figName = plot_config.get('fig_name', 'default_figure')
    figType = plot_config.get('fig_type', 'pdf')

    colourmaps = ['hsv', 'twilight', 'plasma']

    # One panel per simulation suite in the config, in config order. Deriving
    # the count from the config (rather than hardcoding 3) keeps the script
    # working when a suite is commented out of the YAML.
    nPanels = len(config['simulations'])
    fig, axes = plt.subplots(1, nPanels, figsize=(6.0 * nPanels, 5.5),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    t0 = time.time()
    stats_rows = []
    # Cosmology for the secondary axes, taken from the first simulation
    # instantiated. All the suites here share a Planck-like cosmology, so the
    # arcmin <-> ckpc/h mapping is common to every panel to well under a pixel.
    cosmo = None

    for panel_idx, sim_type in enumerate(config['simulations']):
        sim_type_name = sim_type['sim_type']
        ax = axes[panel_idx]

        # Wrap rather than index directly: nPanels comes from the config, so a
        # fourth suite would otherwise be an IndexError.
        colourmap = matplotlib.colormaps[colourmaps[panel_idx % len(colourmaps)]] # type: ignore

        sims = sim_type['sims']
        if sim_type_name == 'FLAMINGO':
            fallback = colourmap(np.linspace(0.2, 0.85, len(sims)))
            colours = [_FLAMINGO_COLOURS.get(s['feedback'], fallback[k])
                       for k, s in enumerate(sims)]
        elif sim_type_name in ('IllustrisTNG', 'SIMBA'):
            colours = colourmap(np.linspace(0.2, 0.85, len(sims)))
        else:
            raise ValueError(f"Unknown simulation type: {sim_type_name}")

        if verbose:
            print(f"\n=== Processing simulations of type: {sim_type_name} ===")

        for j, sim in enumerate(sims):
            sim_name = sim['name']
            snapshot = sim['snapshot']

            if verbose:
                print(f"Processing simulation: {sim_name}")

            if sim_type_name == 'IllustrisTNG':
                stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                            simType=sim_type_name)
            else:
                # feedback holds the SIMBA model, or the FLAMINGO variant
                # directory name ('L1_m9' is the fiducial run).
                feedback = sim['feedback']
                if verbose:
                    print(f"Processing feedback model: {feedback}")

                stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                            simType=sim_type_name,
                                            feedback=feedback)
                if sim_type_name == 'SIMBA':
                    sim_name = sim_name + '_' + feedback
                else:
                    # '-' instead of '_' so the label renders under usetex
                    sim_name = f"FLAMINGO {feedback}".replace('_', '-')

            if cosmo is None:
                cosmo = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'],
                                      Om0=stacker.header['Omega0'],
                                      Tcmb0=2.7255 * u.K)

            # Select the sample and report its halo masses, then stack on
            # exactly that sample.
            halo_mask, sim_stats = sample_stats(
                stacker, sim_name, use_subhalos=use_subhalos,
                halo_abundance_target=halo_abundance_target,
                halo_mass_avg=halo_mass_avg,
                halo_mass_upper=halo_mass_upper)
            stats_rows.append(sim_stats)
            if verbose:
                print(format_stats(sim_stats), flush=True)

            radii0, profiles0 = stacker.stackMap(pType, filterType=filterType, minRadius=1.0, maxRadius=6.0, pixelSize=pixelSize, # type: ignore
                                    save=saveField, load=loadField, radDistance=radDistance,
                                    use_subhalos=use_subhalos,
                                    halo_abundance_target=halo_abundance_target,
                                    halo_mass_avg=halo_mass_avg,
                                    halo_mass_upper=halo_mass_upper,
                                    halo_mask=halo_mask,
                                    projection=projection, mask=False, maskRad=None)

            profiles_plot = np.mean(profiles0, axis=1)
            ax.plot(radii0 * radDistance, profiles_plot, label=sim_name,
                    color=colours[j], lw=2, marker='o')
            if plotErrorBars:
                profiles_err = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
                ax.fill_between(radii0 * radDistance,
                                profiles_plot - profiles_err,
                                profiles_plot + profiles_err,
                                color=colours[j], alpha=0.2)

    # Observational data, overlaid on every panel (as in the last column of
    # simulated_kSZ_masked.py).
    if plot_config.get('plot_data', False):
        data = np.load(plot_config['data_path'])
        r_data = data['theta_arcmins']
        profile_data = data['signal']
        profile_err = data['noise']

        for ax in axes:
            ax.errorbar(r_data, profile_data, yerr=profile_err, fmt='s', color='k',
                        label=plot_config['data_label'], markersize=5, zorder=10)

    # Configure all panels
    T_CMB = 2.7255
    v_c = 300000 / 299792458
    k = 1 / (T_CMB * v_c * 1e6)

    def forward_arcmin(arcmin):
        return arcmin_to_comoving(arcmin, redshift, cosmo)

    def inverse_arcmin(comoving):
        return comoving_to_arcmin(comoving, redshift, cosmo)

    for panel_idx, sim_type in enumerate(config['simulations']):
        ax = axes[panel_idx]

        ax.set_xlabel('R [arcmin]')
        if panel_idx == 0:
            ax.set_ylabel(r'$T_{kSZ}$ [$\mu K \rm{arcmin}^2$]')

        secax_x = ax.secondary_xaxis('top',
                                     functions=(forward_arcmin, inverse_arcmin))
        secax_x.set_xlabel('R [ckpc/h]')

        # One secondary y-axis for the row, on the rightmost panel (the panels
        # share their y-axis).
        if panel_idx == nPanels - 1:
            secax = ax.secondary_yaxis('right',
                                       functions=(lambda y: y * k,
                                                  lambda y: y / k))
            secax.set_ylabel(r'$\tau_{\rm CAP} = T_{kSZ}/T_{CMB}\;\; c/v_{rms}$')

        ax.legend(loc='lower right', fontsize=12)
        ax.set_yscale('log')
        ax.set_xlim(0.0, 6.5)
        ax.grid(True)
        ax.set_title(sim_type['sim_type'])

    fig.suptitle(f'Stacked kSZ profiles, {filterType} filter, z={redshift}', fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.94))  # Leave space at the top for the title
    fig.savefig(figPath / f'{figName}_{pType}_z{redshift}_unmasked.{figType}', dpi=300) # type: ignore
    plt.close(fig)

    # Halo-sample summary: to stdout (so it lands in the SLURM .out) and to a
    # text file alongside the figure.
    selection = (f"SHAM on SubhaloMStar, target n = {halo_abundance_target} (cMpc/h)^-3, "
                 f"parent-mass cap {halo_mass_upper:.3e} Msun/h"
                 if use_subhalos else
                 f"mass cut, target <M> = {halo_mass_avg:.4e} Msun/h, "
                 f"upper bound {halo_mass_upper:.3e} Msun/h")
    preamble = [
        f'Halo samples for {figName}_{pType}_z{redshift}_unmasked.{figType}',
        f'config          : {path2config}',
        f'particle type   : {pType}    filter: {filterType}    projection: {projection}',
        f'redshift        : {redshift}',
        f'selection       : {selection}',
    ]
    table = format_table(stats_rows)
    print('\n' + '\n'.join(preamble) + '\n\n' + table + '\n', flush=True)
    write_stats_file(figPath / f'{figName}_{pType}_z{redshift}_unmasked_halo_masses.txt',
                     stats_rows, preamble=preamble)

    print('Done!!! time taken = ', time.time() - t0, ' seconds')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process config.')
    parser.add_argument('-p', '--path2config', type=str, default='./configs/unbound_gas/tau_z05_CAP_masked_flamingo.yaml', help='Path to the configuration file.')
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)
