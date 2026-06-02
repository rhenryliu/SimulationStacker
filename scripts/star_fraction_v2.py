"""
star_fraction_v2.py
===================
Produces a grouped bar chart of the baryonic star fraction with TWO bars per
simulation:

    1. Global star fraction        -- Stars / (gas + Stars + BH) summed over the
                                      whole simulation box.
    2. Within-R200m star fraction  -- the same ratio, but summed only over 3-D
                                      spheres of radius mean(GroupRad) around the
                                      selected (massive) halo sample.

The two bars are plotted side by side for each simulation, with the simulation
name centred underneath the pair.

The halo sample and the in-sphere particle selection mirror make_stackArea.py:
haloes are chosen with select_massive_halos() and the 3-D spheres are cut out via
get_cutout_indices_3d() / sum_over_cutouts().  For R200m the mean of the
selected sample's 'GroupRad' is used (Group_R_Mean200 for TNG,
virial_quantities.r200 for SIMBA).

Usage
-----
    python star_fraction_v2.py -p ./configs/star_fraction_v2.yaml
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# --- set default font to Computer Modern (with fallbacks) and tick fontsize ---
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

import time

# Import packages
sys.path.append('../src/')
from stacker import SimulationStacker
from halos import select_massive_halos
from mask_utils import get_cutout_indices_3d, sum_over_cutouts

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore

import yaml
import argparse
from pathlib import Path
from datetime import datetime


def star_fractions_for_sim(stacker, nPixels, projection, saveField, loadField,
                           halo_mass_avg, halo_mass_upper, verbose=True):
    """Compute the global and within-R200m star fractions for one simulation.

    The star fraction is defined as ``Stars / (gas + Stars + BH)`` (the fraction
    of total baryonic mass that is in stars).  The global value sums each 3-D
    field over the whole box; the within-R200m value sums only over 3-D spheres
    of radius ``mean(GroupRad)`` around the selected massive haloes.

    Args:
        stacker (SimulationStacker): Initialised stacker for the simulation.
        nPixels (int): Grid resolution for the 3-D fields.
        projection (str): Projection axis passed through to ``makeField``.
        saveField (bool): Cache newly computed fields to disk.
        loadField (bool): Load cached fields from disk when available.
        halo_mass_avg (float): Target average halo mass [M_sun/h] for the
            'massive' selection.
        halo_mass_upper (float): Upper halo-mass bound [M_sun/h] for the same
            selection.
        verbose (bool): If True, print progress information.

    Returns:
        tuple[float, float]: ``(global_fraction, r200m_fraction)``.
    """
    # Build 3-D fields for the three baryon components.
    fields = {}
    for pType in ('gas', 'Stars', 'BH'):
        if verbose:
            print(f"  Building 3D field: {pType}")
        fields[pType] = stacker.makeField(pType, nPixels=nPixels, dim='3D',
                                          projection=projection,
                                          save=saveField, load=loadField)

    # --- Global star fraction: sum each field over the whole box ---
    sum_gas = np.sum(fields['gas'])
    sum_stars = np.sum(fields['Stars'])
    sum_bh = np.sum(fields['BH'])
    global_fraction = sum_stars / (sum_gas + sum_stars + sum_bh)

    # --- Within-R200m star fraction: sum over 3-D spheres around haloes ---
    # Physical size of a single voxel [comoving kpc/h per pixel].
    kpcPerPixel = stacker.header['BoxSize'] / fields['Stars'].shape[0]

    # Load haloes and select the massive sample (identical to make_stackArea.py).
    haloes = stacker.loadHalos()
    halo_mask = select_massive_halos(haloes['GroupMass'], halo_mass_avg, halo_mass_upper)

    GroupRad = haloes['GroupRad'][halo_mask]                # R200m [comoving kpc/h]
    GroupPos_px = np.round(haloes['GroupPos'][halo_mask] / kpcPerPixel).astype(int) % nPixels
    n_haloes = len(GroupRad)

    # R200m = mean GroupRad of the selected sample, applied to every halo.
    R200m = GroupRad.mean()
    rr = np.ones(n_haloes) * R200m / kpcPerPixel           # sphere radius [pixels]

    if verbose:
        print(f"  Selected haloes: {n_haloes}")
        print(f"  mean R200m = {R200m:.1f} comoving kpc/h ({rr[0]:.2f} pixels)")

    mask_indices = get_cutout_indices_3d(fields['Stars'], GroupPos_px, rr)
    sum_gas_halo = sum_over_cutouts(fields['gas'], mask_indices).sum()
    sum_stars_halo = sum_over_cutouts(fields['Stars'], mask_indices).sum()
    sum_bh_halo = sum_over_cutouts(fields['BH'], mask_indices).sum()
    r200m_fraction = sum_stars_halo / (sum_gas_halo + sum_stars_halo + sum_bh_halo)

    return float(global_fraction), float(r200m_fraction)


def main(path2config, verbose=True):
    """Load config, compute the two star fractions per simulation, and save the
    grouped bar chart.

    Args:
        path2config (str): Path to the YAML configuration file.
        verbose (bool): If True, print progress messages.

    Raises:
        ValueError: If the configuration file is invalid or missing required fields.
    """

    with open(path2config) as f:
        config = yaml.safe_load(f)

    redshift = config['redshift']
    loadField = config.get('load_field', True)
    saveField = config.get('save_field', True)
    projection = config.get('projection', 'yz')
    nPixels = config.get('n_pixels', 1000)

    # Cast to float: PyYAML parses unsigned-exponent literals (e.g. '5.0e14') as
    # strings, which would crash deep in the halo-selection comparison.
    halo_mass_avg = float(config.get('halo_mass_avg', 10**13.22))    # M_sun/h
    halo_mass_upper = float(config.get('halo_mass_upper', 5e14))     # M_sun/h

    now = datetime.now()
    yr_string = now.strftime("%Y-%m")
    dt_string = now.strftime("%m-%d")

    figPath = Path(config.get('fig_path')) / yr_string / dt_string
    figPath.mkdir(parents=True, exist_ok=True)

    figName = config['fig_name']
    figType = config['fig_type']

    # plot_dict maps sim_name -> {'global': float, 'r200m': float}
    load_from_file = config.get('load', False)
    load_path = config.get('load_path', None)

    plot_dict = {}
    t0 = time.time()

    if load_from_file:
        if load_path is None:
            raise ValueError("Config parameter 'load' is True but 'load_path' is not specified.")
        load_path_obj = Path(load_path)
        if not load_path_obj.exists():
            raise FileNotFoundError(f"Load path does not exist: {load_path}")

        with open(load_path_obj, 'r') as f:
            plot_dict = yaml.safe_load(f)

        print(f"Loaded star fraction data from {load_path}")
    else:
        for sim_type in config['simulations']:
            sim_type_name = sim_type['sim_type']

            if verbose:
                print(f"Processing simulations of type: {sim_type_name}")

            for sim in sim_type['sims']:
                sim_name = sim['name']
                snapshot = sim['snapshot']

                if sim_type_name == 'IllustrisTNG':
                    stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                                simType=sim_type_name)
                    label = sim_name

                elif sim_type_name == 'SIMBA':
                    feedback = sim['feedback']
                    stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                                simType=sim_type_name,
                                                feedback=feedback)
                    label = sim_name + '_' + feedback
                else:
                    raise ValueError(f"Unknown simulation type: {sim_type_name}")

                if verbose:
                    print(f"Processing simulation: {label}")

                global_frac, r200m_frac = star_fractions_for_sim(
                    stacker, nPixels=nPixels, projection=projection,
                    saveField=saveField, loadField=loadField,
                    halo_mass_avg=halo_mass_avg, halo_mass_upper=halo_mass_upper,
                    verbose=verbose,
                )

                plot_dict[label] = {'global': global_frac, 'r200m': r200m_frac}
                print(f"{label}: Global = {global_frac:.4f}, Within R200m = {r200m_frac:.4f}")

    if verbose:
        print(f"Computed all star fractions in {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # Plotting: grouped bars (global + within-R200m) per simulation
    # -----------------------------------------------------------------------
    labels = list(plot_dict.keys())
    global_vals = [plot_dict[k]['global'] for k in labels]
    r200m_vals = [plot_dict[k]['r200m'] for k in labels]

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.tick_params(axis='both', which='both', labelsize=14)

    ax.bar(x - width / 2, global_vals, width=width, color='skyblue',
           edgecolor='none', alpha=0.8, label=r'$f_{\mathrm{star}}^{\mathrm{global}}$')
    ax.bar(x + width / 2, r200m_vals, width=width, color='salmon',
           edgecolor='none', alpha=0.8, label=r'$f_{\mathrm{star}}^{R_{200\mathrm{m}}}$')

    ax.set_xlabel('Simulation Suites', fontsize=18)
    ax.set_ylabel(r'Stellar Fraction (compared to total baryons.)', fontsize=18)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right')
    ax.grid(True)
    ax.set_title(f'Baryon Star Fraction, z = {redshift}', fontsize=18)
    ax.legend(loc='best', fontsize=20)

    # -----------------------------------------------------------------------
    # Print and save plot_dict
    # -----------------------------------------------------------------------
    print("Star fraction values (simulation: global, within R200m):")
    for name in labels:
        print(f"{name}: global={plot_dict[name]['global']:.6f}, "
              f"r200m={plot_dict[name]['r200m']:.6f}")

    if not load_from_file:
        out_dict_path = figPath / f"{figName}_z{redshift}_star_fraction.yaml"
        try:
            serializable_dict = {
                k: {'global': float(v['global']), 'r200m': float(v['r200m'])}
                for k, v in plot_dict.items()
            }
            with open(out_dict_path, "w") as f:
                yaml.safe_dump(serializable_dict, f)
            print(f"Saved star fraction dictionary to {out_dict_path}")
        except Exception as e:
            print(f"Failed to save star fraction dictionary: {e}")

    fig.tight_layout()
    fig.savefig(figPath / f'{figName}_z{redshift}.{figType}', dpi=300)  # type: ignore
    plt.close(fig)

    print('Done!!!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process config.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/star_fraction_v2.yaml',
                        help='Path to the configuration file.')
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)
