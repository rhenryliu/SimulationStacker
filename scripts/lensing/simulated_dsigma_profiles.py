"""Simulated Delta Sigma profiles against the HSC Y3 x DESI LRG lensing measurement.

One panel per simulation suite (SIMBA, IllustrisTNG, FLAMINGO), each overlaid
with the same observational profile.

The radial sampling is taken from the measurement rather than from the config:
the data file's ``rp`` column is a comoving, h-free length (Mpc), and each
simulation is stacked at exactly those separations, converted into its own
ckpc/h with its own Hubble parameter. Simulations and data therefore land on
identical x values by construction, and no arcmin<->length conversion (and so
no assumption about the effective lens redshift) enters the comparison.

Both axes are quoted in the units of a single reference Hubble parameter,
``plot.data_h`` -- the h of the cosmology the measurement was made with. Each
simulation's ckpc/h radii and Msun*h/(ckpc/h)^2 surface densities are rescaled
from its own h to that reference, so the three suites (h = 0.6774, 0.68, 0.704,
0.681) are directly comparable on one set of axes.

The halo sample is the same SHAM selection used by
``unbound_gas/simulated_kSZ.py``, so a selection tuned here can be carried over
to the kSZ comparison unchanged.

Run from the scripts/ directory:
    python lensing/simulated_dsigma_profiles.py -p configs/lensing/dsigma_profile_z05.yaml
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import time

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# Import packages

sys.path.append('../src/')
from utils import arcmin_to_comoving, comoving_to_arcmin  # type: ignore
from stacker import SimulationStacker  # type: ignore
from rprofiles import select_sham_subhalos  # type: ignore

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore  # noqa: F401 (needed by stacker internals)

import yaml
import argparse
from pathlib import Path
from datetime import datetime
from astropy.table import Table


# Fixed colours for the FLAMINGO feedback variants, keyed by feedback name.
# Kept in sync with unbound_gas/simulated_kSZ.py and compare_data_ratio.py so
# the same simulation is the same colour across every figure.
_FLAMINGO_COLOURS = {
    'L1_m9':           '#B30000',  # dark red (fiducial)
    'fgas-8sigma':     '#FF7F0E',  # orange
    'Jet_fgas-4sigma': '#C71585',  # magenta
}


def read_lensing_data(data_path, data_h):
    """Read the measured Delta Sigma profile and its radial binning.

    Args:
        data_path (str): Path to the dsigma output FITS table. ``rp`` is read
            in comoving Mpc (h-free) and ``ds`` in Msun/pc^2 (h-free), which is
            what ``dsigma.precompute`` writes when called with an astropy
            cosmology (see the TUNIT keywords on the table).
        data_h (float): Hubble parameter of the cosmology the measurement was
            made with. Sets the reference h of both plot axes.

    Returns:
        tuple: ``(rp_mpc, radii_kpch, profile, profile_err)`` with ``rp_mpc``
        the comoving separations in Mpc, ``radii_kpch`` the same separations in
        ckpc/h at the reference h, ``profile`` the excess surface density in
        Msun*h/(ckpc/h)^2 and ``profile_err`` its standard error.

    Raises:
        ValueError: If the ``rp`` bins are not evenly spaced. The simulations
            are stacked on a ``np.linspace`` between the first and last bin, so
            an uneven binning would silently put the two on different radii.
    """
    data = Table.read(data_path)  # type: ignore
    rp_mpc = np.asarray(data['rp'], dtype=float)

    spacings = np.diff(rp_mpc)
    if spacings.size == 0 or not np.allclose(spacings, spacings[0], rtol=1e-6):
        raise ValueError(
            "The data rp bins are not evenly spaced, so stack_on_array's "
            "np.linspace sampling cannot reproduce them. Got rp = "
            f"{rp_mpc}.")

    # Msun/pc^2 -> Msun/kpc^2 (x1e6), then physical -> h-units (Sigma_h =
    # Sigma_phys / h, since masses gain a factor h and areas a factor h^2).
    to_h_units = 1e6 / data_h
    profile = np.asarray(data['ds'], dtype=float) * to_h_units
    profile_err = np.sqrt(np.diag(np.asarray(data['cov'], dtype=float))) * to_h_units

    radii_kpch = rp_mpc * 1000.0 * data_h

    return rp_mpc, radii_kpch, profile, profile_err


def main(path2config, verbose=True):
    """Stack the simulated Delta Sigma profiles and save the comparison figure.

    Args:
        path2config (str): Path to the configuration file.
        verbose (bool, optional): If True, prints detailed information. Defaults to True.

    Raises:
        ValueError: If the configuration file names an unknown simulation type,
            or if plotting the data is disabled (the data sets the radial bins).
    """

    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_config = config.get('stack', {})
    plot_config = config.get('plot', {})

    # Stacking parameters
    redshift = stack_config.get('redshift', 0.5)
    filterType = stack_config.get('filter_type', 'DSigma')
    loadField = stack_config.get('load_field', True)
    saveField = stack_config.get('save_field', False)
    pType = stack_config.get('particle_type', 'total')
    projection = stack_config.get('projection', 'yz')
    pixelSize = stack_config.get('pixel_size', 0.2)
    dsigma_dr_arcmin = stack_config.get('dsigma_dr', None)

    # Halo-selection parameters, matching unbound_gas/simulated_kSZ.py.
    use_subhalos = stack_config.get('use_subhalos', True)
    halo_abundance_target = stack_config.get('halo_abundance_target', 5e-4)
    halo_mass_avg = stack_config.get('halo_mass_avg', 10 ** (13.22))
    halo_mass_upper = stack_config.get('halo_mass_upper', 5 * 10 ** (14))

    # Plotting parameters
    now = datetime.now()
    yr_string = now.strftime("%Y-%m")
    dt_string = now.strftime("%m-%d")

    figPath = Path(plot_config.get('fig_path', '../figures/')) / yr_string / dt_string
    figPath.mkdir(parents=True, exist_ok=True)
    plotErrorBars = plot_config.get('plot_error_bars', True)
    figName = plot_config.get('fig_name', 'default_figure')
    figType = plot_config.get('fig_type', 'png')
    data_h = plot_config.get('data_h', 0.6766)  # Planck18

    # The measurement defines the radial bins, so it is not optional here.
    if not plot_config.get('plot_data', False):
        raise ValueError("plot_data must be true: the data file sets the radial bins.")

    rp_mpc, radii_kpch, profile_data, profile_err = read_lensing_data(
        plot_config['data_path'], data_h)
    nRadii = len(rp_mpc)
    if verbose:
        print(f"Radial bins taken from {plot_config['data_path']}")
        print(f"  rp [comoving Mpc] : {np.round(rp_mpc, 4)}")
        print(f"  R  [ckpc/h, h={data_h}] : {np.round(radii_kpch, 1)}")

    colourmaps = ['hsv', 'twilight', 'plasma']

    # One panel per simulation suite in the config, in config order.
    nPanels = len(config['simulations'])
    fig, axes = plt.subplots(1, nPanels, figsize=(6.5 * nPanels, 6.0),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    t0 = time.time()
    for panel_idx, sim_type in enumerate(config['simulations']):
        sim_type_name = sim_type['sim_type']
        ax = axes[panel_idx]

        colourmap = matplotlib.colormaps[colourmaps[panel_idx % len(colourmaps)]]  # type: ignore

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
                sim_label = sim_name
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
                    sim_label = sim_name + '_' + feedback
                else:
                    sim_label = f"FLAMINGO {feedback}"

            h_sim = stacker.header['HubbleParam']
            cosmo = FlatLambdaCDM(H0=100 * h_sim, Om0=stacker.header['Omega0'],
                                  Tcmb0=2.7255 * u.K)

            # Field resolution: the same rule makeMap uses, so the cached
            # fields under products/2D/ are hit rather than rebuilt.
            theta_arcmin = comoving_to_arcmin(stacker.header['BoxSize'], redshift, cosmo=cosmo)
            nPixels = np.ceil(theta_arcmin / pixelSize).astype(int)

            # Stack at exactly the measured separations. rp is a comoving,
            # h-free length, so rp * h_sim is that same length in this
            # simulation's Mpc/h, which is the unit of radDistance = 1000 kpc/h.
            minRadius = rp_mpc[0] * h_sim
            maxRadius = rp_mpc[-1] * h_sim
            radDistance = 1000.0  # kpc/h per radial unit

            # Annulus width of the compensated kernel, in the same Mpc/h units.
            # Left at None the filter falls back to 3 pixels, which is a
            # resolution-dependent angular scale rather than a fixed one.
            if dsigma_dr_arcmin is None:
                dr = None
            else:
                dr = arcmin_to_comoving(dsigma_dr_arcmin, redshift, cosmo) / 1000.0

            if verbose:
                print(f"  theta_arcmin: {theta_arcmin:.1f}, nPixels: {nPixels}, h: {h_sim}")
                print(f"  minRadius: {minRadius:.4f} Mpc/h, maxRadius: {maxRadius:.4f} Mpc/h, "
                      f"dr: {dr if dr is None else round(dr, 4)} Mpc/h")

            # Select the sample once and hand the exact index array to
            # stackField, so the sample reported is the sample stacked.
            if use_subhalos:
                # Read both catalogues once and hand them in: left to itself
                # select_sham_subhalos reads them again, which for FLAMINGO is
                # a second full pass over the SOAP-HBT files.
                subhalos = stacker.loadSubHalos()
                parents = stacker.loadHalos()
                halo_mask = select_sham_subhalos(stacker, halo_abundance_target,
                                                 parent_mass_upper=halo_mass_upper,
                                                 subhalos=subhalos, parents=parents)
                parent_grnr = subhalos['SubhaloGrNr'][halo_mask]
                mean_mass = np.mean(parents['GroupMass'][parent_grnr])
                if verbose:
                    print(f"  SHAM sample: {halo_mask.size} subhaloes, "
                          f"<M_parent> = {mean_mass:.3e} Msun/h")
            else:
                halo_mask = None

            radii, profiles = stacker.stackField(
                pType, filterType=filterType,
                minRadius=minRadius, maxRadius=maxRadius, numRadii=nRadii,  # type: ignore
                save=saveField, load=loadField, radDistance=radDistance,
                nPixels=nPixels, projection=projection,
                use_subhalos=use_subhalos,
                halo_abundance_target=halo_abundance_target,
                halo_mass_avg=halo_mass_avg,
                halo_mass_upper=halo_mass_upper,
                halo_mask=halo_mask, dr=dr)

            # radii comes back in this simulation's Mpc/h at exactly rp_mpc *
            # h_sim; plot everything on the shared reference-h axis instead.
            radii_plot = radii / h_sim * 1000.0 * data_h
            if not np.allclose(radii_plot, radii_kpch, rtol=1e-6):
                raise ValueError(
                    "Simulated radii do not coincide with the measured rp "
                    f"bins: {radii_plot} vs {radii_kpch}")

            # Sigma is in Msun*h_sim/(ckpc/h_sim)^2; rescale to the reference h
            # (Sigma_phys = Sigma_sim * h_sim, Sigma_ref = Sigma_phys / data_h).
            profiles = profiles * h_sim / data_h

            profiles_plot = np.mean(profiles, axis=1)
            ax.plot(radii_plot, profiles_plot, label=sim_label,
                    color=colours[j], lw=2, marker='o')
            if plotErrorBars:
                profiles_sem = np.std(profiles, axis=1) / np.sqrt(profiles.shape[1])
                ax.fill_between(radii_plot,
                                profiles_plot - profiles_sem,
                                profiles_plot + profiles_sem,
                                color=colours[j], alpha=0.2)

    # Observational data, overlaid on every panel.
    for panel_idx, sim_type in enumerate(config['simulations']):
        ax = axes[panel_idx]
        ax.errorbar(radii_kpch, profile_data, yerr=profile_err, fmt='s', color='k',
                    label=plot_config['data_label'], markersize=5, zorder=10)

        ax.set_xlabel(f'R [ckpc/h], h = {data_h}', fontsize=16)
        if panel_idx == 0:
            ax.set_ylabel(rf'$\Delta \Sigma$({pType})  [M$_\odot h$ / (ckpc/h)$^2$]',
                          fontsize=16)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True)
        ax.set_title(sim_type['sim_type'], fontsize=16)

    selection = (f"SHAM on SubhaloMStar, target n = {halo_abundance_target} (cMpc/h)$^{{-3}}$"
                 if use_subhalos else
                 f"mass cut, target $<M>$ = {halo_mass_avg:.3e} Msun/h")
    fig.suptitle(rf'$\Delta \Sigma$ profiles, {filterType} filter, z={redshift} -- {selection}',
                 fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figPath / f'{figName}_{pType}_z{redshift}_{filterType}.{figType}', dpi=300)  # type: ignore
    plt.close(fig)

    print(f"\nSaved: {figPath / f'{figName}_{pType}_z{redshift}_{filterType}.{figType}'}")
    print('Done!!! Time taken: ', time.time() - t0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process config.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/lensing/dsigma_profile_z05.yaml',
                        help='Path to the configuration file.')
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)
