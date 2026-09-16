"""compare_cmb_fgas.py

Overlay the beam-compensated DESI x ACT x HSC gas fraction measurement with the
DESI LS x ACT DR6 gas fraction digitized from Hadzhiyska et al. (2025),
arXiv:2507.14136 Fig. 6 (top).

The beam-compensated points are recomputed exactly as in
beam_compensated_ratio_v2.py (Phases 1–2): the beam suppression factor

    beam_factor(r) = mean over sims of
                     <DSigma_ionized_gas(beamed)>(r) / <DSigma_ionized_gas(no beam)>(r)

is stacked from the beamTest simulations, and the measured ratio is divided by
it.  With ``compensation.load_beam_factor`` (default true) the factor is read
from the file plot_beam_factors.py writes, so no beamTest stacking is needed
unless that file is missing or was made with different beamTest settings.

With ``use_sims: true`` (default false) the simulation f_gas(R) curves are also
plotted, stacked exactly as in beam_compensated_ratio_v2.py Phase 3 from the
``stack`` and ``simulations`` sections of this script's own config (extra
gas-component ratios are not supported).  That stacking needs a compute node;
the figure name then gains a ``_sims`` suffix.

The digitized data are plotted with the symmetric error bar ``fgas_err``.
Both datasets sit on the same 1–6 arcmin grid, so they are shifted
horizontally by -/+ ``x_offset`` arcmin to keep the markers from overlapping.

Usage
-----
    python lensing/compare_cmb_fgas.py -p configs/lensing/compare_cmb_fgas_z05.yaml
"""

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
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

sys.path.append('../src/')
from utils import arcmin_to_comoving, comoving_to_arcmin  # type: ignore

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore  # noqa: F401 (needed by stacker internals)

# Reuse the stacker/Omega_b resolution and the nested-npz loader from the
# beam-compensation script so both scripts build the data points identically.
# (scripts/lensing/ is on sys.path because this script lives there.)
from beam_compensated_ratio_v2 import (  # type: ignore
    _FLAMINGO_COLOURS, _resolve_stacker, load_beam_factor_npz,
    load_measurements_npz, sham_parent_halo_stats)

# ---------------------------------------------------------------------------
# Matplotlib style — matches beam_compensated_ratio_v2.py exactly
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
    "legend.fontsize": 18,
})


def main(path2config: str, verbose: bool = True) -> None:
    """Recompute the beam-compensated data and overlay it with the digitized f_gas.

    Args:
        path2config: Path to the master YAML configuration file.
        verbose: If True, print progress messages to stdout.
    """
    config_dir = Path(path2config).parent

    with open(path2config) as f:
        master = yaml.safe_load(f)

    with open(config_dir / master['beam_test_config']) as f:
        bt_config = yaml.safe_load(f)

    # The simulation curves read 'stack' and 'simulations' from this config
    # itself, in the layout of beam_compensated_ratio_v2.py's no_beam_config.
    nb_config = master

    comp_config = master.get('compensation', {})
    plot_config = master.get('plot', {})
    use_sims    = master.get('use_sims', False)

    use_sim_scatter  = comp_config.get('use_sim_scatter', False)
    load_beam_factor = comp_config.get('load_beam_factor', True)

    # ---- Output path: figures/<year-month>/<month-day>/ ----
    now      = datetime.now()
    fig_path = (
        Path(plot_config.get('fig_path', '../figures/'))
        / now.strftime("%Y-%m")
        / now.strftime("%m-%d")
    )
    fig_path.mkdir(parents=True, exist_ok=True)

    fig_name = plot_config.get('fig_name', 'compare_cmb_fgas')
    fig_type = plot_config.get('fig_type', 'pdf')
    x_offset = plot_config.get('x_offset', 0.05)   # arcmin
    plot_error_bars = plot_config.get('plot_error_bars', True)
    do_plot_r200m   = plot_config.get('plot_r200m', True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Reference cosmology for the top (comoving) axis: taken from the first
    # beamTest simulation, which is the same sim beam_compensated_ratio_v2.py
    # uses (the first noBeam sim) in the z05 configs.
    cosmo_ref: Optional[FlatLambdaCDM] = None

    # R200m reference line: cached from the first IllustrisTNG sim (use_sims only).
    R200m_arcmin_ref: Optional[float] = None
    R200m_label: Optional[str] = None

    t0 = time.time()

    # ==========================================================================
    # Phase 1: compute beam compensation factor from beamTest simulations
    # (identical to beam_compensated_ratio_v2.py Phase 1)
    # ==========================================================================
    bt_stack = bt_config['stack']

    bt_redshift     = bt_stack.get('redshift',        0.5)
    bt_rad_distance = bt_stack.get('rad_distance',    1.0)
    bt_pType        = bt_stack.get('particle_type',   'ionized_gas')
    bt_filter_type  = bt_stack.get('filter_type',     'DSigma')
    bt_pixel_size   = bt_stack.get('pixel_size',      0.5)
    bt_beam_size    = bt_stack.get('beam_size',        1.6)
    bt_pType2       = bt_stack.get('particle_type_2', 'ionized_gas')
    bt_filter_type2 = bt_stack.get('filter_type_2',   'DSigma')
    bt_pixel_size_2 = bt_stack.get('pixel_size_2',    0.2)
    bt_beam_size_2  = bt_stack.get('beam_size_2',     None)

    bt_base_kwargs = dict(
        minRadius    = bt_stack.get('min_radius',   1.0),
        maxRadius    = bt_stack.get('max_radius',   6.0),
        numRadii     = bt_stack.get('num_radii',    9),
        projection   = bt_stack.get('projection',   'yz'),
        save         = bt_stack.get('save_field',   True),
        load         = bt_stack.get('load_field',   True),
        radDistance  = bt_rad_distance,
        mask         = bt_stack.get('mask_haloes',  False),
        maskRad      = bt_stack.get('mask_radii',   3.0),
        use_subhalos = bt_stack.get('use_subhalos', False),
        halo_abundance_target = bt_stack.get('halo_abundance_target', None),
    )

    # The cached file (written by plot_beam_factors.py) holds the same per-sim
    # ratios as the loop below; stack instead if it is missing or stale.
    bf_path = bt_config.get('beam_factor', {}).get(
        'npz_path', f'../data/beam_factors/beam_factor_z{bt_redshift}.npz')
    cached = load_beam_factor_npz(bf_path, bt_config) if load_beam_factor else None

    beam_factors: list = []   # one (n_radii,) array per simulation

    if cached is not None:
        beam_factors    = list(cached['beam_factor'])
        bt_theta_arcmin = cached['theta_arcmin']

        # No stacking, but the top axis still needs the first simulation's
        # cosmology; building its stacker only reads the snapshot header.
        first_group = bt_config['simulations'][0]
        stacker, _, omega_b = _resolve_stacker(
            first_group['sim_type'], first_group['sims'][0], bt_redshift, verbose)
        cosmo_ref = FlatLambdaCDM(
            H0=100 * stacker.header['HubbleParam'],
            Om0=stacker.header['Omega0'],
            Tcmb0=2.7255 * u.K,
            Ob0=omega_b,
        )
    else:
        bt_radii = None

        for sim_group in bt_config['simulations']:
            sim_type_name = sim_group['sim_type']
            for sim in sim_group['sims']:
                stacker, sim_label, omega_b = _resolve_stacker(
                    sim_type_name, sim, bt_redshift, verbose)

                # Cache the cosmology of the first simulation for the top axis.
                if cosmo_ref is None:
                    cosmo_ref = FlatLambdaCDM(
                        H0=100 * stacker.header['HubbleParam'],
                        Om0=stacker.header['Omega0'],
                        Tcmb0=2.7255 * u.K,
                        Ob0=omega_b,
                    )

                if verbose:
                    print(f"[beamTest] Processing {sim_label}")

                radii_b, profiles_b = stacker.stackMap(
                    bt_pType,  filterType=bt_filter_type,
                    pixelSize=bt_pixel_size, beamSize=bt_beam_size,
                    **bt_base_kwargs)
                radii_n, profiles_n = stacker.stackMap(
                    bt_pType2, filterType=bt_filter_type2,
                    pixelSize=bt_pixel_size_2, beamSize=bt_beam_size_2,
                    **bt_base_kwargs)

                # Ratio of halo-means for this simulation.  No baryon normalisation
                # since pType == pType2 (same particle, different resolution/beam).
                mean_b = np.mean(profiles_b, axis=1)   # (n_radii,)
                mean_n = np.mean(profiles_n, axis=1)
                beam_factors.append(mean_b / mean_n)

                if bt_radii is None:
                    bt_radii = radii_b

        # bt_radii * bt_rad_distance gives the x-axis in arcmin, matching the data.
        bt_theta_arcmin = bt_radii * bt_rad_distance

    # Mean across simulations — each sim contributes equally regardless of
    # halo count, so we average the per-sim ratios rather than pooling halos.
    beam_factor_arr = np.array(beam_factors)                      # (n_sims, n_radii)
    beam_factor     = np.mean(beam_factor_arr, axis=0)            # (n_radii,)

    if verbose:
        print(f"Beam factor (mean over {len(beam_factors)} sims): {beam_factor}")

    # ==========================================================================
    # Phase 2: compensate the data
    # (identical to beam_compensated_ratio_v2.py Phase 2, diagonal errors only)
    # ==========================================================================
    data       = load_measurements_npz(plot_config['data_path'])
    key        = 'source_bin_0'
    theta_data = data[key]['ksz_theta_arcmin']
    ratio_data = data[key]['ratio']
    sigma_data = data[key]['ratio_err']

    # Sanity check: beamTest radii should match the data theta grid (both 9-point,
    # 1–6 arcmin, equally spaced).
    if not np.allclose(bt_theta_arcmin, theta_data, rtol=1e-3):
        print("[warn] beamTest radii do not match data theta grid — "
              "the beam correction will be approximate.")
        print(f"  beamTest theta: {bt_theta_arcmin}")
        print(f"  data theta    : {theta_data}")

    R_compensated = ratio_data / beam_factor

    # Error on R_comp = R / bf is sigma / bf; optionally add the scatter of the
    # beam factor across simulations (diagonal of J cov_beam J^T).
    inv_bf        = 1.0 / beam_factor
    sigma_comp_sq = (sigma_data * inv_bf) ** 2
    if use_sim_scatter:
        cov_beam       = np.cov(beam_factor_arr, rowvar=False)   # (n_radii, n_radii)
        jac            = ratio_data / beam_factor ** 2            # d(R_comp)/d(bf)
        sigma_comp_sq += np.diag(np.outer(jac, jac) * cov_beam)
    sigma_compensated = np.sqrt(sigma_comp_sq)

    # ==========================================================================
    # Phase 3: load the digitized Hadzhiyska et al. (2025) f_gas
    # ==========================================================================
    # R is in arcmin; fgas is already normalised by Omega_b / Omega_m.
    # The symmetric fgas_err is used (the last point's upper bar is clipped in
    # the original figure, so fgas_err falls back to the lower half-bar there).
    cmb = np.load(plot_config['cmb_fgas_path'])
    theta_cmb = cmb['R']
    fgas_cmb  = cmb['fgas']
    sigma_cmb = cmb['fgas_err']

    if verbose:
        print(f"Loaded digitized f_gas from {plot_config['cmb_fgas_path']}")
        print(f"  theta [arcmin]: {theta_cmb}")

    # ==========================================================================
    # Phase 4 (optional): noBeam simulation profiles
    # (identical to beam_compensated_ratio_v2.py Phase 3, without extra ratios)
    # ==========================================================================
    if use_sims:
        nb_stack = nb_config['stack']

        nb_redshift     = nb_stack.get('redshift',        0.5)
        nb_rad_distance = nb_stack.get('rad_distance',    1.0)
        nb_pType        = nb_stack.get('particle_type',   'ionized_gas')
        nb_filter_type  = nb_stack.get('filter_type',     'DSigma')
        nb_pixel_size   = nb_stack.get('pixel_size',      0.2)
        nb_beam_size    = nb_stack.get('beam_size',       None)
        nb_pType2       = nb_stack.get('particle_type_2', 'total')
        nb_filter_type2 = nb_stack.get('filter_type_2',   'DSigma')
        nb_pixel_size_2 = nb_stack.get('pixel_size_2',    0.2)
        nb_beam_size_2  = nb_stack.get('beam_size_2',     None)

        nb_base_kwargs = dict(
            minRadius    = nb_stack.get('min_radius',   1.0),
            maxRadius    = nb_stack.get('max_radius',   6.0),
            numRadii     = nb_stack.get('num_radii',    9),
            projection   = nb_stack.get('projection',   'yz'),
            save         = nb_stack.get('save_field',   True),
            load         = nb_stack.get('load_field',   True),
            radDistance  = nb_rad_distance,
            mask         = nb_stack.get('mask_haloes',  False),
            maskRad      = nb_stack.get('mask_radii',   3.0),
            use_subhalos = nb_stack.get('use_subhalos', False),
            halo_abundance_target = nb_stack.get('halo_abundance_target', None),
        )

        # Pre-build sim_label → colour mapping from the noBeam config, using the
        # same colourmap logic as compare_data_ratio.py so colours are identical
        # for the same simulations.
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
                    label  = f"{sim['name']}_{sim['feedback']}"
                    label  = f"SIMBA-100"
                    colour = colours[j]
                colour_for_sim[label] = colour

        for sim_group in nb_config['simulations']:
            sim_type_name = sim_group['sim_type']
            for sim in sim_group['sims']:
                stacker, sim_label, omega_b = _resolve_stacker(
                    sim_type_name, sim, nb_redshift, verbose)

                cosmo = FlatLambdaCDM(
                    H0=100 * stacker.header['HubbleParam'],
                    Om0=stacker.header['Omega0'],
                    Tcmb0=2.7255 * u.K,
                    Ob0=omega_b,
                )
                if cosmo_ref is None:
                    cosmo_ref = cosmo

                # Mean parent-halo mass and R200m of the SHAM-selected sample.
                mean_mass, R200m_kpch = sham_parent_halo_stats(
                    stacker, nb_stack.get('halo_abundance_target', None))
                if verbose:
                    print(f"  {sim_label}: mean M = {mean_mass:.3e} Msun/h "
                          f"(log10 = {np.log10(mean_mass):.3f}), "
                          f"mean R200m = {R200m_kpch:.3f} comoving kpc/h")

                # Cache R200m (arcmin) from the first IllustrisTNG sim for the vline.
                if (do_plot_r200m and sim_type_name == 'IllustrisTNG'
                        and R200m_arcmin_ref is None):
                    R200m_arcmin_ref = comoving_to_arcmin(R200m_kpch, nb_redshift, cosmo=cosmo)
                    R200m_label = sim_label

                if verbose:
                    print(f"[noBeam] Processing {sim_label}")

                f_baryon = omega_b / stacker.header['Omega0']
                factor   = 1.0 / f_baryon
                colour   = colour_for_sim[sim_label]
                x_axis   = None   # set on first stack

                # Stack the shared denominator once and reuse for all ratios.
                radii1, profiles1 = stacker.stackMap(
                    nb_pType2, filterType=nb_filter_type2,
                    pixelSize=nb_pixel_size_2, beamSize=nb_beam_size_2,
                    **nb_base_kwargs)
                mean1 = np.mean(profiles1, axis=1)
                err1  = np.std(profiles1, axis=1) / np.sqrt(profiles1.shape[1])

                # ---- primary ratio (ionized_gas / total, solid line + marker) ----
                radii0, profiles0 = stacker.stackMap(
                    nb_pType, filterType=nb_filter_type,
                    pixelSize=nb_pixel_size, beamSize=nb_beam_size,
                    **nb_base_kwargs)
                x_axis = radii0 * nb_rad_distance
                mean0  = np.mean(profiles0, axis=1)

                profiles_plot = mean0 / mean1 * factor
                ax.plot(x_axis, profiles_plot,
                        label=sim_label, color=colour, lw=2, marker='o', ls='-')

                if plot_error_bars:
                    err0         = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
                    profiles_err = np.abs(profiles_plot) * np.sqrt(
                        (err0 / mean0)**2 + (err1 / mean1)**2)
                    ax.fill_between(x_axis,
                                    profiles_plot - profiles_err,
                                    profiles_plot + profiles_err,
                                    color=colour, alpha=0.2)

    # ==========================================================================
    # Phase 5: overlay the two datasets
    # ==========================================================================
    # Our beam-compensated measurement, shifted left by x_offset.
    ax.errorbar(
        theta_data - x_offset,
        R_compensated,
        yerr=sigma_compensated,
        fmt='s',
        color='black',
        label=plot_config.get('data_label',
                              r'DESI $\times$ ACT $\times$ HSC (beam-corrected)'),
        markersize=6,
        capsize=2,
    )

    # Digitized Hadzhiyska et al. (2025) measurement, shifted right by x_offset.
    # Green diamonds keep it distinct from the simulation curves (blue/purple/red
    # colours with circle markers).
    ax.errorbar(
        theta_cmb + x_offset,
        fgas_cmb,
        yerr=sigma_cmb,
        fmt='D',
        color='tab:blue',
        label=plot_config.get('cmb_fgas_label', 'Hadzhiyska et al. (2025)'),
        markersize=6,
        capsize=2,
    )

    # ==========================================================================
    # Figure cosmetics
    # ==========================================================================
    # Top axis in comoving Mpc/h at the beamTest redshift (z = 0.5 here).
    if cosmo_ref is not None:
        secax_x = ax.secondary_xaxis(
            'top',
            functions=(
                lambda arcmin: arcmin_to_comoving(arcmin, bt_redshift, cosmo_ref) / 1e3,
                lambda mpc_h:  comoving_to_arcmin(mpc_h * 1e3, bt_redshift, cosmo_ref),
            ),
        )
        secax_x.set_xlabel(r'R [comoving Mpc/h]')

    # f_gas = 1 corresponds to the cosmic baryon fraction (no feedback).
    ax.axhline(1.0, color='k', ls='--', lw=1.5, label='_nolegend_')

    # Vertical dotted line at mean R200m of the first IllustrisTNG sim's halos.
    if do_plot_r200m and R200m_arcmin_ref is not None:
        ax.axvline(R200m_arcmin_ref, color='gray', ls=':', lw=2,
                   label=rf'$\langle R_{{200\mathrm{{m}}}} \rangle$ ({R200m_label})')

    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(plot_config.get('ylabel', r'$f_{\rm gas}(R)$'))
    ax.set_xlim(0.0, bt_stack.get('max_radius', 6.0) * bt_rad_distance + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    fig.tight_layout()

    out_stem = f"{fig_name}{'_sims' if use_sims else ''}_z{bt_redshift}"
    out_path = fig_path / f'{out_stem}.{fig_type}'
    print(f'Saving figure to {out_path}')
    fig.savefig(out_path, dpi=150)  # type: ignore
    plt.close(fig)

    print(f'Done. Elapsed: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Overlay beam-compensated f_gas data with the digitized '
                    'Hadzhiyska et al. (2025) f_gas measurement.',
    )
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/lensing/compare_cmb_fgas_z05.yaml',
        help='Path to the master YAML configuration file.',
    )
    args = vars(parser.parse_args())
    print(f"Config: {args['path2config']}")
    main(**args)
