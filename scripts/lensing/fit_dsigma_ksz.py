"""Fit the SHAM number density on lensing, then predict kSZ from the same haloes.

For each simulation the SHAM target number density ``halo_abundance_target`` is
swept over a grid; the Delta Sigma profile is stacked at each density and
compared to the measured lensing profile by chi-squared. The chi-squared
minimum is interpolated to a continuous density n_best, a SHAM sample is
selected at exactly that density, and *that one sample* -- the same index array
into the subhalo catalogue -- provides both the Delta Sigma curve in the top row
and the kSZ curve in the bottom row. The two rows are therefore a like-for-like
pair, and the bottom row is a prediction, not a fit: no free parameter is
adjusted against the kSZ data.

The figure is 3 columns (one per simulation suite) by 2 rows:

    top row     Delta Sigma [Msun/pc^2] against comoving separation
                [ckpc/h at the measurement's reference h], the n_best stack
                solid, the swept grid densities faded, the bins excluded from
                the chi-squared shaded.
    bottom row  kSZ [uK arcmin^2] for the same n_best sample, against the
                angular separation it is measured at [arcmin].

Each row therefore carries the radial variable native to its own observable;
the two are related by a fixed proportionality at the lens redshift.

Because ``select_sham_subhalos`` keeps the top-N stellar-mass-ranked subhaloes,
the densities in the sweep are exactly nested: the lower-density sample is the
head of the higher-density one. The curves therefore trace a systematic shift
in the halo mass mix rather than independent sample noise.

Run from the scripts/ directory:
    python lensing/fit_dsigma_ksz.py -p configs/lensing/fit_dsigma_ksz_z05.yaml
    python lensing/fit_dsigma_ksz.py -p configs/lensing/fit_dsigma_ksz_z05.yaml --replot
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.cosmology import FlatLambdaCDM, Planck18
from scipy import stats
import astropy.units as u

sys.path.append('../src/')
from stacker import SimulationStacker  # type: ignore
from rprofiles import select_sham_subhalos  # type: ignore
from snr import apply_hartlap  # type: ignore
from utils import comoving_to_arcmin, arcmin_to_comoving, ksz_from_delta_sigma  # type: ignore

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore  # noqa: F401 (needed by stacker internals)


# Fixed colours for the FLAMINGO feedback variants, keyed by feedback name.
# Kept in sync with unbound_gas/simulated_kSZ.py, compare_data_ratio.py and
# lensing/simulated_dsigma_profiles.py.
_FLAMINGO_COLOURS = {
    'L1_m9':           '#B30000',  # dark red (fiducial)
    'fgas-8sigma':     '#FF7F0E',  # orange
    'Jet_fgas-4sigma': '#C71585',  # magenta
}

_COLOURMAPS = ['hsv', 'twilight', 'plasma']


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def uK_per_msun_pc2(z_lens, cosmology=Planck18):
    """Scale factor taking Delta Sigma [Msun/pc^2] to the file's ``ds_uK``.

    ``compare_combined_self.py`` builds ``ds_uK`` as
    ``-ksz_from_delta_sigma(ds, z_l, delta_sigma_is_comoving=True)``. That
    conversion is linear in Delta Sigma, so calibrating it on 1 Msun/pc^2 gives
    the exact factor and the inversion is exact rather than approximate.

    Args:
        z_lens (float): Effective lens redshift used to build the file.
        cosmology (astropy.cosmology, optional): Cosmology used to build the
            file. Defaults to Planck18, matching the upstream pipeline.

    Returns:
        float: Microkelvin per (Msun/pc^2), positive.
    """
    unit = np.array([1.0]) * u.Msun / u.pc ** 2
    return -float(ksz_from_delta_sigma(unit, z_lens, cosmology=cosmology,
                                       delta_sigma_is_comoving=True)[0])


def theta_to_rp(theta_arcmin, z_lens, cosmology=Planck18):
    """Convert angular bins to comoving transverse separation.

    The measurement's rp bins were built as theta * D_M(z_lens) with Planck18,
    so this reproduces the upstream dsigma output file's own rp column exactly.

    Args:
        theta_arcmin (float or np.ndarray): Angular bins in arcmin.
        z_lens (float): Effective lens redshift.
        cosmology (astropy.cosmology, optional): Defaults to Planck18.

    Returns:
        float or np.ndarray: Comoving transverse separation in Mpc (h-free).
    """
    return ((np.asarray(theta_arcmin) * u.arcmin).to(u.rad).value
            * cosmology.comoving_transverse_distance(z_lens).to(u.Mpc).value)


def read_lensing_data(plot_config):
    """Read the lensing measurement and put it back in native Msun/pc^2.

    Args:
        plot_config (dict): The config's ``plot`` block. Uses ``ds_data_path``,
            ``ds_data_key`` and ``data_z_lens``.

    Returns:
        dict: With keys ``theta`` (arcmin), ``rp`` (comoving Mpc, h-free),
        ``ds`` (Msun/pc^2), ``cov`` (Msun/pc^2 squared, no Hartlap correction
        applied) and ``err`` (the square root of its diagonal).

    Raises:
        KeyError: If ``ds_data_key`` is not present in the file.
    """
    path = plot_config['ds_data_path']
    key = plot_config['ds_data_key']
    z_lens = plot_config['data_z_lens']

    archive = np.load(path)
    if f'{key}/ds_uK' not in archive.files:
        raise KeyError(f"{key!r} not found in {path}. Available prefixes: "
                       f"{sorted({f.split('/')[0] for f in archive.files})}")

    theta = np.asarray(archive[f'{key}/ksz_theta_arcmin'], dtype=float)
    scale = uK_per_msun_pc2(z_lens)
    ds = np.asarray(archive[f'{key}/ds_uK'], dtype=float) / scale
    cov = np.asarray(archive[f'{key}/ds_uK_cov'], dtype=float) / scale ** 2

    rp = theta_to_rp(theta, z_lens)

    return {'theta': theta, 'rp': rp, 'ds': ds, 'cov': cov,
            'err': np.sqrt(np.diag(cov)), 'uK_scale': scale}


def read_ksz_data(plot_config):
    """Read the kSZ measurement overlaid on the bottom row.

    Args:
        plot_config (dict): The config's ``plot`` block. Uses ``ksz_data_path``.

    Returns:
        dict: With keys ``theta`` (arcmin), ``signal`` and ``err``
        (uK arcmin^2). Only diagonal errors are available, which is why the
        bottom row carries no goodness-of-fit number.
    """
    archive = np.load(plot_config['ksz_data_path'])
    return {'theta': np.asarray(archive['theta_arcmins'], dtype=float),
            'signal': np.asarray(archive['signal'], dtype=float),
            'err': np.asarray(archive['noise'], dtype=float)}


def save_npz_atomic(path, data):
    """Write an npz by way of a temporary file, then rename it into place.

    ``np.savez`` writes in place and is not atomic. The checkpoint exists to
    survive the job being killed at the wall clock, so a kill landing mid-write
    would otherwise truncate the very file holding every simulation already
    finished. Renaming within the same directory is atomic on POSIX.

    Args:
        path (Path or str): Destination ``.npz``.
        data (dict): Arrays to save.
    """
    path = Path(path)
    # The suffix must stay '.npz': np.savez appends '.npz' to any other name,
    # which would leave the rename pointing at a file that was never written.
    tmp = path.with_name(path.name + '.tmp.npz')
    np.savez(tmp, **data)
    os.replace(tmp, path)


def config_fingerprint(config):
    """Hash of the configuration that determines the stacked and fitted results.

    ``--resume`` reuses cached profiles for simulations it does not restack, so
    the cache is only valid while the settings behind it are unchanged. Only the
    inputs that affect numbers are hashed; purely cosmetic plot keys are not, so
    a rename or a colour change does not throw away hours of stacking.

    Args:
        config (dict): The parsed configuration.

    Returns:
        str: Hex digest.
    """
    relevant = {
        'stack': config.get('stack', {}),
        'fit': config.get('fit', {}),
        'data': {k: config.get('plot', {}).get(k) for k in
                 ('ds_data_path', 'ds_data_key', 'data_z_lens', 'ksz_data_path')},
        'simulations': config.get('simulations', []),
    }
    blob = json.dumps(relevant, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def chi2_of_profile(model, data, cov, fit_mask, n_jk):
    """Chi-squared of a model profile against the measurement.

    The Hartlap correction is derived from the number of bins actually fitted,
    rather than reusing a correction computed for the full data vector.

    Args:
        model (np.ndarray): Simulated profile, same length as ``data``.
        data (np.ndarray): Measured profile.
        cov (np.ndarray): Measured covariance, uncorrected, shape (N, N).
        fit_mask (np.ndarray): Boolean mask selecting the fitted bins.
        n_jk (int): Number of jackknife fields behind ``cov``.

    Returns:
        float: Chi-squared over the fitted bins.
    """
    residual = (model - data)[fit_mask]
    cov_fit = apply_hartlap(cov[np.ix_(fit_mask, fit_mask)], n_jk)
    return float(residual @ np.linalg.inv(cov_fit) @ residual)


def fit_abundance(targets, chi2_values):
    """Interpolate the best-fit number density from a chi-squared grid.

    A parabola is fitted in log10(n) through the grid minimum and its two
    neighbours; its vertex is the best fit and its curvature gives the
    delta-chi-squared = 1 interval. When the minimum falls on a grid edge no
    parabola is available, so the edge value is returned and flagged rather
    than extrapolated -- extending the grid needs a restack, so this is
    reported instead of raised, to avoid discarding a long run.

    Args:
        targets (np.ndarray): Grid of number densities, ascending.
        chi2_values (np.ndarray): Chi-squared at each grid point.

    Returns:
        dict: ``n_best`` (float), ``chi2_best`` (float), ``log10_sigma`` (float
        or nan) and ``at_edge`` (bool).
    """
    targets = np.asarray(targets, dtype=float)
    chi2_values = np.asarray(chi2_values, dtype=float)
    # np.argmin returns the index of a NaN if one is present, and the `a <= 0`
    # test below does not catch a NaN curvature, so a single bad chi-squared
    # would propagate silently into n_best and into the stack that follows.
    if not np.all(np.isfinite(chi2_values)):
        raise ValueError(f"non-finite chi-squared in the sweep: {chi2_values}")
    k = int(np.argmin(chi2_values))

    if k == 0 or k == len(targets) - 1:
        return {'n_best': float(targets[k]), 'chi2_best': float(chi2_values[k]),
                'log10_sigma': float('nan'), 'at_edge': True}

    x = np.log10(targets[k - 1:k + 2])
    y = chi2_values[k - 1:k + 2]
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:  # not a minimum (can happen if the grid is too coarse)
        return {'n_best': float(targets[k]), 'chi2_best': float(chi2_values[k]),
                'log10_sigma': float('nan'), 'at_edge': False}

    x0 = -b / (2 * a)
    return {'n_best': float(10 ** x0),
            'chi2_best': float(a * x0 ** 2 + b * x0 + c),
            'log10_sigma': float(1.0 / np.sqrt(a)),
            'at_edge': False}


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------

def make_stacker(sim_type_name, sim, redshift):
    """Instantiate a stacker and build its display label.

    Args:
        sim_type_name (str): One of 'IllustrisTNG', 'SIMBA', 'FLAMINGO'.
        sim (dict): The config entry, with 'name', 'snapshot' and (for SIMBA
            and FLAMINGO) 'feedback'.
        redshift (float): Snapshot redshift.

    Returns:
        tuple: ``(stacker, label)``.

    Raises:
        ValueError: If ``sim_type_name`` is not a known suite.
    """
    if sim_type_name == 'IllustrisTNG':
        return (SimulationStacker(sim['name'], sim['snapshot'], z=redshift,
                                  simType=sim_type_name), sim['name'])
    if sim_type_name in ('SIMBA', 'FLAMINGO'):
        feedback = sim['feedback']
        stacker = SimulationStacker(sim['name'], sim['snapshot'], z=redshift,
                                    simType=sim_type_name, feedback=feedback)
        label = (f"{sim['name']}_{feedback}" if sim_type_name == 'SIMBA'
                 else f"FLAMINGO {feedback}")
        return stacker, label
    raise ValueError(f"Unknown simulation type: {sim_type_name}")


def field_pixels(stacker, redshift, pixel_size):
    """Number of pixels per side at a target angular pixel scale.

    Mirrors the rule ``makeMap`` uses, so the cached products under
    ``products/2D/`` are hit rather than rebuilt.

    Args:
        stacker (SimulationStacker): Provides the header.
        redshift (float): Redshift at which the box is projected.
        pixel_size (float): Target pixel scale in arcmin.

    Returns:
        int: Pixels per side.
    """
    cosmo = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'],
                          Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K)
    theta_arcmin = comoving_to_arcmin(stacker.header['BoxSize'], redshift, cosmo=cosmo)
    return int(np.ceil(theta_arcmin / pixel_size))


def stack_dsigma_at(stacker, target, *, subhalos, parents, halo_mass_upper,
                    lens, stack_kwargs, h_sim):
    """Select a SHAM sample at one number density and stack Delta Sigma on it.

    Args:
        stacker (SimulationStacker): The simulation.
        target (float): SHAM target number density in (cMpc/h)^-3.
        subhalos (dict): Pre-loaded subhalo catalogue.
        parents (dict): Pre-loaded halo catalogue.
        halo_mass_upper (float): Parent FoF mass cap in Msun/h.
        lens (dict): Output of :func:`read_lensing_data`, for the radius check.
        stack_kwargs (dict): Passed straight to ``stackField``.
        h_sim (float): The simulation's Hubble parameter.

    Returns:
        tuple: ``(halo_mask, mean_profile, sem, mean_parent_mass)`` with the
        profiles in Msun/pc^2.

    Raises:
        ValueError: If the stacked radii do not land on the measured rp bins.
    """
    halo_mask = select_sham_subhalos(stacker, target,
                                     parent_mass_upper=halo_mass_upper,
                                     subhalos=subhalos, parents=parents)
    if halo_mask.size == 0:
        raise ValueError(
            f"target n = {target:.3e} (cMpc/h)^-3 selected zero subhaloes in this "
            "box; the density grid reaches below what the box can support.")
    radii, profiles = stacker.stackField(halo_mask=halo_mask, **stack_kwargs)

    # radii come back at exactly rp * h_sim; guard the invariant the whole
    # comparison rests on.
    if not np.allclose(radii / h_sim, lens['rp'], rtol=1e-6):
        raise ValueError("simulated radii do not coincide with the measured rp "
                         f"bins: {radii / h_sim} vs {lens['rp']}")

    # Msun*h_sim/(ckpc/h_sim)^2 -> Msun/pc^2:
    #   Sigma_phys[Msun/kpc^2] = Sigma_sim * h_sim, then / 1e6.
    to_msun_pc2 = h_sim / 1e6
    mean_profile = np.mean(profiles, axis=1) * to_msun_pc2
    sem = np.std(profiles, axis=1) / np.sqrt(profiles.shape[1]) * to_msun_pc2
    parent_grnr = subhalos['SubhaloGrNr'][halo_mask]
    return (halo_mask, mean_profile, sem,
            float(np.mean(parents['GroupMass'][parent_grnr])))


def compute_results(config, cache_path=None, resume=None, verbose=True):
    """Sweep the number density, fit it on lensing, and stack kSZ at the best fit.

    Results are checkpointed to ``cache_path`` after every simulation, so a run
    that is cut short (the interactive QOS caps at 4 h) keeps everything already
    stacked and can be continued with ``--resume``. Resume works in two tiers:
    the density sweep, which dominates the cost, is reused whenever the cached
    grid matches the configured one, and only the comparatively cheap
    best-fit stacks are redone.

    Args:
        config (dict): The parsed configuration.
        cache_path (Path, optional): Where to checkpoint after each simulation.
            Defaults to None (no checkpointing).
        resume (dict, optional): Previously cached results. Simulations already
            present are copied over and not restacked. Defaults to None.
        verbose (bool, optional): Print progress. Defaults to True.

    Returns:
        dict: Flat, npz-serialisable mapping. Per simulation ``<label>/...``
        holds the sweep (``targets``, ``ds_grid`` of shape n_targets x n_radii,
        ``ds_sem``, ``chi2``, ``n_haloes``, ``mean_mass``), the fit
        (``n_best``, ``log10_sigma``, ``at_edge``, ``chi2_best`` from the
        parabola and ``chi2_best_actual`` from the stacked profile) and the
        n_best sample itself (``ds_best``, ``ds_best_sem``, ``n_best_haloes``,
        ``n_best_mean_mass``, ``ksz``, ``ksz_sem``); plus the shared ``theta``,
        ``ksz_theta``, ``ds_data``, ``ds_cov`` and ``labels``.
    """
    stack_config = config['stack']
    plot_config = config['plot']
    fit_config = config.get('fit', {})

    redshift = stack_config.get('redshift', 0.5)
    projection = stack_config.get('projection', 'yz')
    load_field = stack_config.get('load_field', True)
    save_field = stack_config.get('save_field', False)

    ptype = stack_config.get('particle_type', 'total')
    filter_type = stack_config.get('filter_type', 'DSigma')
    pixel_size = stack_config.get('pixel_size', 0.2)
    dsigma_dr_arcmin = stack_config.get('dsigma_dr', None)

    ksz_ptype = stack_config.get('ksz_particle_type', 'tau')
    ksz_filter = stack_config.get('ksz_filter_type', 'CAP')
    ksz_pixel = stack_config.get('ksz_pixel_size', 0.5)
    ksz_beam = stack_config.get('ksz_beam_size', 1.6)
    ksz_rmin = stack_config.get('ksz_min_radius', 1.0)
    ksz_rmax = stack_config.get('ksz_max_radius', 6.0)
    ksz_nrad = stack_config.get('ksz_num_radii', 9)

    halo_mass_upper = stack_config.get('halo_mass_upper', 5e14)
    default_targets = [float(t) for t in stack_config['abundance_targets']]

    r_min_arcmin = fit_config.get('r_min_arcmin', 2.25)
    n_jk = int(fit_config.get('n_jk_lens', 100))
    calibrate = fit_config.get('calibrate_dr_bias', False)
    calibration_dr = fit_config.get('calibration_dr', 0.05)
    calibration_sim = fit_config.get('calibration_sim', None)

    lens = read_lensing_data(plot_config)
    theta = lens['theta']
    fit_mask = theta >= r_min_arcmin
    if fit_mask.sum() < 3:
        raise ValueError(f"r_min_arcmin={r_min_arcmin} leaves only "
                         f"{int(fit_mask.sum())} bins; need at least 3.")

    if verbose:
        print(f"Lensing measurement: {plot_config['ds_data_path']}")
        print(f"  key            : {plot_config['ds_data_key']}")
        print(f"  uK per Msun/pc2: {lens['uK_scale']:.8f}")
        print(f"  theta [arcmin] : {theta}")
        print(f"  ds [Msun/pc^2] : {np.round(lens['ds'], 4)}")
        print(f"  fitted bins    : {int(fit_mask.sum())} of {len(theta)} "
              f"(theta >= {r_min_arcmin} arcmin)")
        print(f"  abundance grid : {default_targets}\n")

    fingerprint = config_fingerprint(config)
    if resume is not None and str(resume.get('config_fingerprint', '')) != fingerprint:
        print("  WARNING: the cache was written under a different configuration "
              "(or predates this check); ignoring it and restacking everything.")
        resume = None

    results = {'config_fingerprint': np.asarray(fingerprint),
               'targets': np.asarray(default_targets),
               'theta': theta,
               'ds_data': lens['ds'],
               'ds_err': lens['err'],
               'ds_cov': lens['cov'],
               'fit_mask': fit_mask,
               'r_min_arcmin': np.asarray(r_min_arcmin),
               'n_jk_lens': np.asarray(n_jk)}
    labels, panels = [], []

    for panel_idx, sim_type in enumerate(config['simulations']):
        sim_type_name = sim_type['sim_type']
        if verbose:
            print(f"=== {sim_type_name} ===")

        for sim in sim_type['sims']:
            stacker, label = make_stacker(sim_type_name, sim, redshift)
            labels.append(label)
            panels.append(panel_idx)

            h_sim = stacker.header['HubbleParam']
            cosmo = FlatLambdaCDM(H0=100 * h_sim, Om0=stacker.header['Omega0'],
                                  Tcmb0=2.7255 * u.K)

            # A simulation whose chi-squared minimum falls outside the shared
            # grid can carry its own, without restacking the others.
            targets = [float(t) for t in sim.get('abundance_targets', default_targets)]

            # Resume in two tiers. The density sweep is by far the expensive
            # part and depends only on the grid, so it is reused whenever the
            # cached grid matches; the cheap best-fit stacks are redone if the
            # cache predates them.
            cached = {k[len(label) + 1:]: v for k, v in (resume or {}).items()
                      if k.startswith(f'{label}/')}
            cached_targets = cached.get('targets', (resume or {}).get('targets'))
            sweep_usable = ('ds_grid' in cached and cached_targets is not None
                            and np.array_equal(np.asarray(cached_targets, dtype=float),
                                               np.asarray(targets, dtype=float)))
            if sweep_usable and 'ds_best' in cached:
                for key, value in cached.items():
                    results[f'{label}/{key}'] = value
                if 'ksz_theta' in resume:
                    results['ksz_theta'] = resume['ksz_theta']
                if verbose:
                    print(f"  {label}: resumed from cache, not restacked\n")
                continue
            if targets != default_targets and verbose:
                print(f"  {label}: using its own abundance grid {targets}")

            n_pixels = field_pixels(stacker, redshift, pixel_size)
            # rp is a comoving h-free length; rp * h_sim is that same length in
            # this simulation's Mpc/h, the unit of radDistance = 1000 kpc/h.
            min_radius = lens['rp'][0] * h_sim
            max_radius = lens['rp'][-1] * h_sim
            dr = (None if dsigma_dr_arcmin is None
                  else arcmin_to_comoving(dsigma_dr_arcmin, redshift, cosmo) / 1000.0)

            if verbose:
                print(f"  {label}: nPixels={n_pixels}, h={h_sim}, "
                      f"R=[{min_radius:.4f}, {max_radius:.4f}] Mpc/h")

            # Read both catalogues once for the whole density sweep.
            subhalos = stacker.loadSubHalos()
            parents = stacker.loadHalos()

            stack_kwargs = dict(pType=ptype, filterType=filter_type,
                                minRadius=min_radius, maxRadius=max_radius,
                                numRadii=len(theta), save=save_field,
                                load=load_field, radDistance=1000.0,
                                nPixels=n_pixels, projection=projection,
                                use_subhalos=True,
                                halo_mass_upper=halo_mass_upper, dr=dr)

            if sweep_usable:
                ds_grid = list(cached['ds_grid'])
                ds_sem = list(cached['ds_sem'])
                chi2_values = list(cached['chi2'])
                n_haloes = list(cached['n_haloes'])
                mean_mass = list(cached['mean_mass'])
                if verbose:
                    print(f"    sweep reused from cache ({len(targets)} densities)")
            else:
                ds_grid, ds_sem, chi2_values = [], [], []
                n_haloes, mean_mass = [], []
                for target in targets:
                    halo_mask, mean_profile, sem, m_par = stack_dsigma_at(
                        stacker, target, subhalos=subhalos, parents=parents,
                        halo_mass_upper=halo_mass_upper, lens=lens,
                        stack_kwargs=stack_kwargs, h_sim=h_sim)
                    n_haloes.append(halo_mask.size)
                    mean_mass.append(m_par)
                    ds_grid.append(mean_profile)
                    ds_sem.append(sem)
                    chi2_values.append(chi2_of_profile(mean_profile, lens['ds'],
                                                       lens['cov'], fit_mask, n_jk))
                    if verbose:
                        print(f"    n={target:.2e}: {halo_mask.size:7d} subhaloes, "
                              f"<M_parent>={m_par:.3e} Msun/h, "
                              f"chi2={chi2_values[-1]:.2f}")

            fit = fit_abundance(targets, chi2_values)
            best_idx = int(np.argmin(chi2_values))
            if verbose:
                edge = ' (AT GRID EDGE -- extend the grid)' if fit['at_edge'] else ''
                print(f"    -> best fit n = {fit['n_best']:.3e} "
                      f"(interpolated chi2={fit['chi2_best']:.2f}, "
                      f"dof={int(fit_mask.sum()) - 1}){edge}")

            # Stack at the fitted density itself, not at the nearest grid point.
            # Both rows of the figure then describe one and the same sample, so
            # the kSZ is a like-for-like prediction of the lensing selection.
            best_mask, ds_best, ds_best_sem, mass_best = stack_dsigma_at(
                stacker, fit['n_best'], subhalos=subhalos, parents=parents,
                halo_mass_upper=halo_mass_upper, lens=lens,
                stack_kwargs=stack_kwargs, h_sim=h_sim)
            chi2_best_actual = chi2_of_profile(ds_best, lens['ds'], lens['cov'],
                                               fit_mask, n_jk)
            if verbose:
                print(f"    n_best sample: {best_mask.size} subhaloes, "
                      f"<M_parent>={mass_best:.3e} Msun/h, "
                      f"chi2={chi2_best_actual:.2f} (stacked, not interpolated)")

            # Optional: measure the DSigma estimator's small-radius bias on this
            # simulation's real profile, rather than on the analytic toy model
            # that motivated r_min_arcmin.
            if calibrate and (calibration_sim in (None, label, sim['name'])):
                # The annulus must be at least a pixel wide, or it catches no
                # pixels at some radii and delta_sigma_kernel divides by zero.
                if calibration_dr < pixel_size:
                    raise ValueError(
                        f"fit.calibration_dr = {calibration_dr} arcmin is narrower "
                        f"than the {pixel_size} arcmin pixel; the annulus would be "
                        "empty at some radii. Use at least the pixel scale.")
                dr_narrow = arcmin_to_comoving(calibration_dr, redshift, cosmo) / 1000.0
                narrow_kwargs = dict(stack_kwargs, dr=dr_narrow, save=False)
                _, narrow = stacker.stackField(halo_mask=best_mask, **narrow_kwargs)
                ratio = ds_best / (np.mean(narrow, axis=1) * h_sim / 1e6)
                results[f'{label}/dr_bias'] = ratio
                if verbose:
                    print(f"    dr bias (dr={dsigma_dr_arcmin}' vs "
                          f"{calibration_dr}'): {np.round(ratio, 3)}")

            # kSZ prediction: the SAME halo sample, no free parameter refitted.
            ksz_radii, ksz_profiles = stacker.stackMap(
                ksz_ptype, filterType=ksz_filter,
                minRadius=ksz_rmin, maxRadius=ksz_rmax, numRadii=ksz_nrad,
                z=redshift, projection=projection, save=save_field,
                load=load_field, radDistance=1.0, pixelSize=ksz_pixel,
                beamSize=ksz_beam, use_subhalos=True,
                halo_mask=best_mask, halo_mass_upper=halo_mass_upper)

            results[f'{label}/targets'] = np.asarray(targets)
            results[f'{label}/ds_grid'] = np.asarray(ds_grid)
            results[f'{label}/ds_best'] = ds_best
            results[f'{label}/ds_best_sem'] = ds_best_sem
            results[f'{label}/n_best_haloes'] = np.asarray(best_mask.size)
            results[f'{label}/n_best_mean_mass'] = np.asarray(mass_best)
            results[f'{label}/chi2_best_actual'] = np.asarray(chi2_best_actual)
            results[f'{label}/ds_sem'] = np.asarray(ds_sem)
            results[f'{label}/chi2'] = np.asarray(chi2_values)
            results[f'{label}/n_best'] = np.asarray(fit['n_best'])
            results[f'{label}/log10_sigma'] = np.asarray(fit['log10_sigma'])
            results[f'{label}/at_edge'] = np.asarray(fit['at_edge'])
            results[f'{label}/chi2_best'] = np.asarray(fit['chi2_best'])
            results[f'{label}/n_haloes'] = np.asarray(n_haloes)
            results[f'{label}/mean_mass'] = np.asarray(mean_mass)
            results[f'{label}/ksz'] = np.mean(ksz_profiles, axis=1)
            results[f'{label}/ksz_sem'] = (np.std(ksz_profiles, axis=1)
                                           / np.sqrt(ksz_profiles.shape[1]))
            results['ksz_theta'] = ksz_radii

            if verbose:
                print(f"    kSZ stacked on the best-fit sample "
                      f"({best_mask.size} subhaloes)\n", flush=True)

            # Checkpoint: a run cut short by the wall clock keeps every
            # simulation already finished, and --resume picks up from here.
            if cache_path is not None:
                save_npz_atomic(cache_path,
                                dict(results, labels=np.asarray(labels),
                                     panels=np.asarray(panels)))

    results['labels'] = np.asarray(labels)
    results['panels'] = np.asarray(panels)
    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(results, config, fig_path):
    """Draw the 3 x 2 comparison and write it to disk.

    Args:
        results (dict): Output of :func:`compute_results`, or the reloaded cache.
        config (dict): The parsed configuration.
        fig_path (Path): Directory to write the figure into.

    Returns:
        Path: The written file.
    """
    plot_config = config['plot']
    stack_config = config['stack']
    ksz_data = read_ksz_data(plot_config)

    theta = results['theta']
    targets = results['targets']
    fit_mask = np.asarray(results['fit_mask'], dtype=bool)

    # The Delta Sigma row is plotted against comoving separation, its own
    # natural radial variable, rather than against the angular bins the kSZ row
    # uses. Every simulation was stacked at rp * h_sim, i.e. at one and the same
    # physical separation, so a single reference h puts them all on one axis.
    data_h = plot_config.get('data_h', 0.6766)
    z_lens = plot_config['data_z_lens']
    radii_kpch = theta_to_rp(theta, z_lens) * 1000.0 * data_h
    r_min_kpch = float(theta_to_rp(float(results['r_min_arcmin']), z_lens)) * 1000.0 * data_h
    labels = [str(s) for s in results['labels']]
    panels = np.asarray(results['panels'], dtype=int)
    sim_types = [s['sim_type'] for s in config['simulations']]
    n_panels = len(sim_types)

    # sharex='row', not True: the two rows now carry different radial variables.
    fig, axes = plt.subplots(2, n_panels, figsize=(6.0 * n_panels, 9.5),
                             sharex='row', sharey='row')
    # With a single suite subplots returns a 1-D array of length 2; reshape
    # explicitly rather than via atleast_2d, which would give (1, 2).
    axes = np.asarray(axes).reshape(2, n_panels)

    for panel_idx in range(n_panels):
        ax_ds, ax_ksz = axes[0, panel_idx], axes[1, panel_idx]
        members = [i for i, p in enumerate(panels) if p == panel_idx]

        cmap = matplotlib.colormaps[_COLOURMAPS[panel_idx % len(_COLOURMAPS)]]  # type: ignore
        fallback = cmap(np.linspace(0.2, 0.85, max(len(members), 2)))

        for k, i in enumerate(members):
            label = labels[i]
            feedback = label.replace('FLAMINGO ', '')
            colour = (_FLAMINGO_COLOURS[feedback]
                      if sim_types[panel_idx] == 'FLAMINGO'
                      and feedback in _FLAMINGO_COLOURS else fallback[k])

            ds_grid = results[f'{label}/ds_grid']
            sim_targets = results.get(f'{label}/targets', targets)
            n_best = float(results[f'{label}/n_best'])
            chi2_best = float(results[f'{label}/chi2_best_actual'])
            dof = int(fit_mask.sum()) - 1
            at_edge = bool(results[f'{label}/at_edge'])

            # The swept densities, faded, so the sensitivity is visible. The
            # solid curve is stacked at the fitted density itself, so it is not
            # one of these.
            for j in range(len(sim_targets)):
                ax_ds.plot(radii_kpch, ds_grid[j], color=colour, lw=1.0, alpha=0.30)

            ds_best = results[f'{label}/ds_best']
            sem = results[f'{label}/ds_best_sem']
            edge = r'$^{\rm edge}$' if at_edge else ''
            ax_ds.plot(radii_kpch, ds_best, color=colour, lw=2,
                       marker='o', label=(rf'{label}'
                                          '\n'
                                          rf'$n$={n_best:.2e}{edge}, '
                                          rf'$\chi^2$/dof={chi2_best:.1f}/{dof}'))
            ax_ds.fill_between(radii_kpch, ds_best - sem, ds_best + sem,
                               color=colour, alpha=0.2)

            ksz = results[f'{label}/ksz']
            ksz_sem = results[f'{label}/ksz_sem']
            ax_ksz.plot(results['ksz_theta'], ksz, color=colour, lw=2,
                        marker='o', label=label)
            ax_ksz.fill_between(results['ksz_theta'], ksz - ksz_sem, ksz + ksz_sem,
                                color=colour, alpha=0.2)

        # Measurements
        ax_ds.errorbar(radii_kpch, results['ds_data'], yerr=results['ds_err'],
                       fmt='s', color='k', markersize=5, zorder=10,
                       label=plot_config.get('ds_data_label', 'lensing'))
        ax_ksz.errorbar(ksz_data['theta'], ksz_data['signal'], yerr=ksz_data['err'],
                        fmt='s', color='k', markersize=5, zorder=10,
                        label=plot_config.get('ksz_data_label', 'kSZ'))

        # Shade the bins excluded from the chi-squared.
        if not fit_mask.all():
            ax_ds.axvspan(0.0, r_min_kpch, color='grey', alpha=0.12, zorder=0)

        for ax in (ax_ds, ax_ksz):
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.3)
        ax_ds.set_xlim(0.75 * radii_kpch[0], 1.05 * radii_kpch[-1])
        ax_ksz.set_xlim(0.5, 6.5)
        ax_ds.set_title(sim_types[panel_idx], fontsize=16)
        ax_ds.legend(loc='upper right', fontsize=9)
        ax_ksz.legend(loc='lower right', fontsize=9)
        ax_ds.set_xlabel(rf'$R$ [ckpc/$h$], $h$ = {data_h}', fontsize=15)
        ax_ksz.set_xlabel(r'$\theta$ [arcmin]', fontsize=15)

        if panel_idx == 0:
            ax_ds.set_ylabel(r'$\Delta \Sigma$ [M$_\odot$ pc$^{-2}$]', fontsize=15)
            ax_ksz.set_ylabel(r'$T_{\rm kSZ}$ [$\mu$K arcmin$^2$]', fontsize=15)

    fig.suptitle('SHAM density fitted on lensing (top), kSZ predicted for the '
                 'same haloes (bottom)', fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig_name = plot_config.get('fig_name', 'fit_dsigma_ksz')
    fig_type = plot_config.get('fig_type', 'png')
    out = fig_path / f"{fig_name}.{fig_type}"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def print_summary(results, config):
    """Print the per-simulation fit summary to stdout.

    Also reports the best-fit density under two alternative inner radial cuts,
    as a robustness check. Those cost nothing: the profiles are already stacked,
    only the chi-squared is recomputed.

    Args:
        results (dict): Output of :func:`compute_results`.
        config (dict): The parsed configuration.
    """
    theta = results['theta']
    targets = results['targets']
    n_jk = int(results['n_jk_lens'])
    labels = [str(s) for s in results['labels']]
    dof = int(np.asarray(results['fit_mask'], dtype=bool).sum()) - 1

    header = (f"{'simulation':22s} {'n_best':>10s} {'+/- dex':>8s} "
              f"{'chi2/dof':>10s} {'PTE':>7s} {'N_halo':>8s} {'<M_par>':>10s}")
    print('\n' + header)
    print('-' * len(header))
    for label in labels:
        n_best = float(results[f'{label}/n_best'])
        sigma = float(results[f'{label}/log10_sigma'])
        chi2_best = float(results[f'{label}/chi2_best_actual'])
        flag = ' *edge*' if bool(results[f'{label}/at_edge']) else ''
        pte = float(stats.chi2.sf(chi2_best, dof)) if dof > 0 else float('nan')
        print(f"{label:22s} {n_best:10.3e} {sigma:8.3f} "
              f"{chi2_best:5.1f}/{dof:<4d} {pte:7.4f} "
              f"{int(results[f'{label}/n_best_haloes']):8d} "
              f"{float(results[f'{label}/n_best_mean_mass']):10.3e}{flag}")

    print('\nRobustness of the best-fit density to the inner radial cut:')
    for r_min in (float(results['r_min_arcmin']), 1.625, 1.0):
        mask = theta >= r_min
        if mask.sum() < 3:
            continue
        row = []
        for label in labels:
            chi2_values = [chi2_of_profile(p, results['ds_data'], results['ds_cov'],
                                           mask, n_jk)
                           for p in results[f'{label}/ds_grid']]
            sim_targets = results.get(f'{label}/targets', targets)
            row.append(f"{label}={fit_abundance(sim_targets, chi2_values)['n_best']:.2e}")
        print(f"  theta >= {r_min:5.3f}': " + '  '.join(row))

    # Flag simulations whose own scatter is not negligible against the data
    # errors, since the fit treats them as noiseless theory.
    print('\nSimulation standard error vs measurement error (fitted bins):')
    mask = np.asarray(results['fit_mask'], dtype=bool)
    for label in labels:
        ratio = results[f'{label}/ds_best_sem'][mask] / results['ds_err'][mask]
        note = '  <-- comparable to the data error' if ratio.max() > 0.5 else ''
        print(f"  {label:22s} max sim_sem/data_err = {ratio.max():.2f}{note}")


# ---------------------------------------------------------------------------

def main(path2config, replot=False, resume=False, verbose=True):
    """Run the fit and write the figure.

    Args:
        path2config (str): Path to the configuration file.
        replot (bool, optional): Reuse the cached profiles instead of
            restacking. Defaults to False.
        resume (bool, optional): Restack only the simulations missing from the
            cache. Defaults to False.
        verbose (bool, optional): Print progress. Defaults to True.

    Raises:
        FileNotFoundError: If ``--replot`` is given but no cache exists.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)

    plot_config = config['plot']
    now = datetime.now()
    fig_path = (Path(plot_config.get('fig_path', '../figures/'))
                / now.strftime('%Y-%m') / now.strftime('%m-%d'))
    fig_path.mkdir(parents=True, exist_ok=True)

    npz_path = Path(plot_config.get(
        'npz_path', f"../data/fit_dsigma_ksz/{plot_config.get('fig_name')}.npz"))
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if replot and resume:
        print('--resume is ignored when --replot is given.')
    if replot:
        if not npz_path.exists():
            raise FileNotFoundError(
                f"--replot given but no cache at {npz_path}. Run once without it.")
        print(f'Loading cached profiles from {npz_path}')
        with np.load(npz_path, allow_pickle=False) as archive:
            results = {k: archive[k] for k in archive.files}
        if str(results.get('config_fingerprint', '')) != config_fingerprint(config):
            print('  WARNING: this cache was written under a different '
                  'configuration; the figure will not match the config it is '
                  'labelled with. Re-run without --replot to refresh it.')
    else:
        previous = None
        if resume and npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as archive:
                previous = {k: archive[k] for k in archive.files}
            print(f'Resuming from {npz_path}')
        results = compute_results(config, cache_path=npz_path, resume=previous,
                                  verbose=verbose)
        # compute_results already checkpointed after the last simulation.
        print(f'Cached profiles to {npz_path}')

    print_summary(results, config)
    out = make_figure(results, config, fig_path)
    print(f'\nSaved: {out}')
    print('Done!!! Time taken: ', time.time() - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit the SHAM density on lensing, predict kSZ from the same haloes.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/lensing/fit_dsigma_ksz_z05.yaml',
                        help='Path to the configuration file.')
    parser.add_argument('--replot', action='store_true',
                        help='Reuse the cached profiles instead of restacking.')
    parser.add_argument('--resume', action='store_true',
                        help='Restack only the simulations missing from the cache.')
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")

    main(**args)
