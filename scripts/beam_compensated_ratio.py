"""beam_compensated_ratio.py

Plot a beam-compensated gas fraction profile alongside no-beam simulation
profiles.

The ACT CMB map has a beam (FWHM ≈ 1.6 arcmin) and pixel scale (0.5 arcmin)
that suppress the kSZ signal in the stacked radial profile relative to the
lensing denominator.  This script removes that effect by:

  1. **Beam factor** (from ``beam_test_config``):
     For each simulation, compute

         beam_factor_i(r) = <DSigma_ionized_gas(beamed)>(r)
                            / <DSigma_ionized_gas(no beam)>(r)

     using a ratio of halo-means (not mean of per-halo ratios).  Then average
     across all simulations (each simulation weighted equally, regardless of
     halo count)::

         beam_factor(r) = mean_i [ beam_factor_i(r) ]

  2. **Compensate the data** (from ``plot.data_path``):

         R_compensated(r) = R_data(r) / beam_factor(r)

     Errors are propagated in quadrature.  If ``compensation.use_sim_scatter``
     is true, the standard deviation of beam_factor_i across simulations is
     also propagated as a systematic term.

  3. **Plot** the beam-compensated data alongside simulation profiles run
     without any beam (from ``no_beam_config``), using the same colours as
     compare_data_ratio.py for the same simulations.

Usage
-----
    python beam_compensated_ratio.py -p configs/beam_compensated_z05.yaml
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
from stacker import SimulationStacker  # type: ignore
from snr import detection_snr

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


def _resolve_stacker(sim_type_name: str, sim: dict, redshift: float,
                     verbose: bool) -> tuple:
    """Instantiate a SimulationStacker and resolve Omega_b.

    Args:
        sim_type_name: ``'IllustrisTNG'`` or ``'SIMBA'``.
        sim: Single simulation entry from the YAML ``sims`` list.
        redshift: Snapshot redshift.
        verbose: Whether to print warnings.

    Returns:
        ``(stacker, sim_label, omega_b)``
    """
    if sim_type_name == 'IllustrisTNG':
        stacker   = SimulationStacker(sim['name'], sim['snapshot'],
                                      z=redshift, simType=sim_type_name)
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
                                      z=redshift, simType=sim_type_name,
                                      feedback=sim['feedback'])
        sim_label = f"{sim['name']}_{sim['feedback']}"
        try:
            omega_b = stacker.header['OmegaBaryon']
        except KeyError:
            omega_b = _OMEGA_B_SIMBA_FALLBACK
            if verbose:
                print(f"  [warn] OmegaBaryon missing in {sim_label} header; "
                      f"using fallback {_OMEGA_B_SIMBA_FALLBACK}")

    else:
        raise ValueError(f"Unknown simulation type: {sim_type_name!r}. "
                         "Expected 'IllustrisTNG' or 'SIMBA'.")

    return stacker, sim_label, omega_b


def main(path2config: str, verbose: bool = True) -> None:
    """Run the beam-compensated gas fraction comparison and save the figure.

    Args:
        path2config: Path to the master YAML configuration file.
        verbose: If True, print progress messages to stdout.
    """
    config_dir = Path(path2config).parent

    with open(path2config) as f:
        master = yaml.safe_load(f)

    with open(config_dir / master['beam_test_config']) as f:
        bt_config = yaml.safe_load(f)

    with open(config_dir / master['no_beam_config']) as f:
        nb_config = yaml.safe_load(f)

    comp_config = master.get('compensation', {})
    plot_config = master.get('plot', {})

    use_sim_scatter = comp_config.get('use_sim_scatter', False)

    # ---- Output path: figures/<year-month>/<month-day>/ ----
    now      = datetime.now()
    fig_path = (
        Path(plot_config.get('fig_path', '../figures/'))
        / now.strftime("%Y-%m")
        / now.strftime("%m-%d")
    )
    fig_path.mkdir(parents=True, exist_ok=True)

    fig_name        = plot_config.get('fig_name', 'beam_compensated_ratio')
    fig_type        = plot_config.get('fig_type', 'pdf')
    plot_error_bars = plot_config.get('plot_error_bars', True)

    fig, ax = plt.subplots(figsize=(10, 8))
    cosmo_ref: Optional[FlatLambdaCDM] = None

    t0 = time.time()

    # ==========================================================================
    # Phase 1: compute beam compensation factor from beamTest simulations
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
    )

    beam_factors: list = []   # one (n_radii,) array per simulation
    bt_radii = None

    for sim_group in bt_config['simulations']:
        sim_type_name = sim_group['sim_type']
        for sim in sim_group['sims']:
            stacker, sim_label, _ = _resolve_stacker(
                sim_type_name, sim, bt_redshift, verbose)

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

    # Mean across simulations — each sim contributes equally regardless of
    # halo count, so we average the per-sim ratios rather than pooling halos.
    beam_factor_arr    = np.array(beam_factors)                      # (n_sims, n_radii)
    beam_factor        = np.mean(beam_factor_arr, axis=0)            # (n_radii,)
    beam_factor_scatter = np.std(beam_factor_arr, axis=0, ddof=1)    # (n_radii,)

    if verbose:
        print(f"Beam factor (mean over {len(beam_factors)} sims): {beam_factor}")

    # ==========================================================================
    # Phase 2: compensate the data
    # ==========================================================================
    # bt_radii * bt_rad_distance gives the x-axis in arcmin, matching the data.
    bt_theta_arcmin = bt_radii * bt_rad_distance

    data       = load_measurements_npz(plot_config['data_path'])
    key        = 'source_bin_0'
    theta_data = data[key]['ksz_theta_arcmin']
    ratio_data = data[key]['ratio']
    sigma_data = data[key]['ratio_err']
    cov_data   = data[key]['ratio_cov_h']

    # Sanity check: beamTest radii should match the data theta grid (both 9-point,
    # 1–6 arcmin, equally spaced).
    if not np.allclose(bt_theta_arcmin, theta_data, rtol=1e-3):
        print("[warn] beamTest radii do not match data theta grid — "
              "the beam correction will be approximate.")
        print(f"  beamTest theta: {bt_theta_arcmin}")
        print(f"  data theta    : {theta_data}")

    R_compensated = ratio_data / beam_factor

    # Propagate covariance: D @ cov_data @ D^T where D = diag(1/beam_factor).
    inv_bf = 1.0 / beam_factor
    cov_compensated = np.outer(inv_bf, inv_bf) * cov_data

    sigma_comp_sq = (sigma_data * inv_bf) ** 2
    if use_sim_scatter:
        # Full (n_radii, n_radii) beam factor covariance across simulations.
        cov_beam    = np.cov(beam_factor_arr, rowvar=False)   # (n_radii, n_radii)
        jac         = ratio_data / beam_factor ** 2            # d(R_comp)/d(bf)
        cov_scatter = np.outer(jac, jac) * cov_beam
        cov_compensated += cov_scatter
        sigma_comp_sq   += np.diag(cov_scatter)
    sigma_compensated = np.sqrt(sigma_comp_sq)

    # ==========================================================================
    # Phase 3: noBeam simulation profiles
    # ==========================================================================
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
    )

    # Pre-build sim_label → colour mapping from the noBeam config, using the
    # same colourmap logic as compare_data_ratio.py so colours are identical
    # for the same simulations.
    colour_for_sim: dict = {}
    for i, sim_group in enumerate(nb_config['simulations']):
        cmap    = matplotlib.colormaps[['plasma', 'twilight'][i]]  # type: ignore[attr-defined]
        n_sims  = len(sim_group['sims'])
        colours = cmap(np.linspace(0.2, 0.85, n_sims))
        for j, sim in enumerate(sim_group['sims']):
            if sim_group['sim_type'] == 'IllustrisTNG':
                label = sim['name']
            else:
                label = f"{sim['name']}_{sim['feedback']}"
            colour_for_sim[label] = colours[j]

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

            if verbose:
                print(f"[noBeam] Processing {sim_label}")

            radii0, profiles0 = stacker.stackMap(
                nb_pType,  filterType=nb_filter_type,
                pixelSize=nb_pixel_size, beamSize=nb_beam_size,
                **nb_base_kwargs)
            radii1, profiles1 = stacker.stackMap(
                nb_pType2, filterType=nb_filter_type2,
                pixelSize=nb_pixel_size_2, beamSize=nb_beam_size_2,
                **nb_base_kwargs)

            # ionized_gas / total: normalise by baryon fraction (same logic as
            # compare_data_ratio.py when pType='ionized_gas', pType2='total').
            f_baryon      = omega_b / stacker.header['Omega0']
            factor        = 1.0 / f_baryon
            mean0         = np.mean(profiles0, axis=1)
            mean1         = np.mean(profiles1, axis=1)
            profiles_plot = mean0 / mean1 * factor

            colour = colour_for_sim[sim_label]
            ax.plot(
                radii0 * nb_rad_distance,
                profiles_plot,
                label=sim_label,
                color=colour,
                lw=2,
                marker='o',
            )

            if plot_error_bars:
                err0         = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
                err1         = np.std(profiles1, axis=1) / np.sqrt(profiles1.shape[1])
                profiles_err = np.abs(profiles_plot) * np.sqrt(
                    (err0 / mean0)**2 + (err1 / mean1)**2)
                ax.fill_between(
                    radii0 * nb_rad_distance,
                    profiles_plot - profiles_err,
                    profiles_plot + profiles_err,
                    color=colour,
                    alpha=0.2,
                )

    # ==========================================================================
    # Phase 4: overlay beam-compensated data
    # ==========================================================================
    ax.errorbar(
        theta_data,
        R_compensated,
        yerr=sigma_compensated,
        fmt='s',
        color='black',
        label=r'DESI $\times$ ACT $\times$ HSC (beam-corrected)',
        markersize=6,
        capsize=2,
    )

    # ==========================================================================
    # Phase 5: Print out the SNR of the beam-compensated data detection, using the full covarianc
    # ==========================================================================
    snr_feedback = detection_snr(R_compensated, cov_compensated, null=1.0)
    print(f"Beam-compensated data detection SNR (relative to null=1): {snr_feedback:.2f}")

    # ==========================================================================
    # Figure cosmetics — matches compare_data_ratio.py
    # ==========================================================================
    if cosmo_ref is not None:
        secax_x = ax.secondary_xaxis(
            'top',
            functions=(
                lambda arcmin: arcmin_to_comoving(arcmin, nb_redshift, cosmo_ref),
                lambda kpc_h:  comoving_to_arcmin(kpc_h,  nb_redshift, cosmo_ref),
            ),
        )
        secax_x.set_xlabel('R [comoving kpc/h]')

    ax.axhline(1.0, color='k', ls='--', lw=1.5, label='_nolegend_')
    ax.set_xlabel('R [arcmin]')
    ax.set_ylabel(
        r'$\frac{\langle \Delta \Sigma_{\rm kSZ} \rangle}'
        r'{\langle \Delta \Sigma_{\rm lens} \rangle} \times \frac{\Omega_m}{\Omega_b}$'
    )
    ax.set_xlim(0.0, nb_stack.get('max_radius', 6.0) * nb_rad_distance + 0.5)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_stem = f'beam_compensated_{nb_pType}_{nb_pType2}_{fig_name}_z{nb_redshift}'
    out_path = fig_path / f'{out_stem}.{fig_type}'
    print(f'Saving figure to {out_path}')
    fig.savefig(out_path, dpi=150) # type: ignore
    plt.close(fig)

    print(f'Done. Elapsed: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot beam-compensated gas fraction ratio against no-beam simulations.',
    )
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/beam_compensated_z05.yaml',
        help='Path to the master YAML configuration file.',
    )
    args = vars(parser.parse_args())
    print(f"Config: {args['path2config']}")
    main(**args)
