"""simulated_dsigma_ratio_masked.py
===================================
The kSZ masking figure (simulated_kSZ_masked.py) recast in the observable of
the companion lensing paper, the ratio of DeltaSigma-filtered profiles

    f_n(theta) = <DeltaSigma_ion^(n)(theta)> / <DeltaSigma_tot(theta)> / (Omega_b / Omega_m),

for the gas within n = 1, 2, 3 x R200m of the stacked galaxies' host haloes
(columns 1-3) and for all the gas (column 4, with the Liu et al. 2026
measurement). The denominator is the unmasked total-matter profile in every
column, as lensing measures it, so the columns differ only through the
numerator.

Denominator: the unconvolved 0.2' total-matter field. Numerator, chosen by
``stack.numerator``:

- ``'tau_map'`` (default): the beam-convolved tau maps (0.5' pixels, 1.6'
  beam) -- the masked ones built by precompute_masked_sham_hosts.py around the
  hosts of the lensing-fit SHAM galaxies, and the unmasked one -- converted to
  ionized-gas mass. Per particle tau = sigma_T N_e / A_pix and
  M_ion = N_e m_p mu_e (mapMaker.make_sz_field, mapMaker.ionized_gas_masses),
  so a tau map times

      K = A_pix,phys * m_p * mu_e * h / (sigma_T * M_sun)

  is the ionized_gas map in Msun/h per pixel; the script checks this against
  the cached unmasked ionized_gas map. This is compare_data_ratio.py's
  convention (beam on the kSZ side only, as in the measurement), shown with
  the measured ratio as it is.
- ``'ionized_gas_field'``: unconvolved ionized-gas fields at the denominator's
  0.2' pixels -- the masked ones built by precompute_masked_particles.py
  (particles kept in exact spheres around the same hosts, binned like the
  cached fields), and the cached unmasked field. This is the beam-compensated
  comparison of beam_compensated_ratio_v2.py, shown with the beam-compensated
  measurement. With ``beam_factor.beamed_npz`` (the profiles of a 'tau_map'
  run) the script also reports the beam factor of these samples, beamed over
  unbeamed unmasked numerator, against the published one.

Both are filtered with the 'DSigma' kernel (0.75' annulus) on the
measurement's 9 radii from 1' to 6', in the yz projection of the cached
total-matter fields.

Every map is loaded by explicit path, and the stacks go through
``stack_on_array`` rather than ``stackMap``:

- ``stackMap(mask=True)`` rebuilds a missing masked map around the mass-cut
  sample and saves it at the shared path; here a missing or mismatched masked
  map is an error (``halo_stats.require_masked_map`` for the tau maps, the same
  test on the ``.sample.json`` record for the ionized-gas fields).
- ``stackMap`` multiplies tau profiles by T_CMB v/c and the pixel area (muK
  units); ``stack_on_array`` returns the bare kernel sums.
- ``makeMap`` loads ``..._map.npy`` whatever the beam requested, since the map
  cache path does not record the beam; the denominator is read from the field
  file (``..._yz.npy``) instead.

Error bars: standard error of the mean of each profile, combined in quadrature
(as in compare_data_ratio.py). Numerator and denominator are stacked on the
same galaxies and are positively correlated, so this overestimates the error
of the ratio.

Known offset between the columns ('tau_map' only): the masked maps are
projections of the 3D TSC tau cubes (precompute_masked_sham_hosts.py), the
unmasked map is binned in 2D. Projecting the unmasked cube the same way raises
the DeltaSigma stack by 2-4% at 1', ~1% at 1.6' and <0.6% beyond (SIMBA-100,
Illustris-1, TNG100-1; 2026-10-11), with the map sums equal to 1e-6. So at the
smallest radii the masked columns can sit slightly above the unmasked one for
that reason alone, as in the kSZ figure, which shares the maps' construction.
In 'ionized_gas_field' mode all four columns share one binning.

Outputs, in ``<fig_path>/<YYYY-MM>/<MM-DD>/``:
``<fig_name>_z<z>_masking_comparison.<ext>``, the profiles
(``..._profiles.npz``) and the halo-sample table (``..._halo_masses.txt``).
The profiles hold ``radii_arcmin`` and, per simulation ``<suite>/<name>[/<feedback>]``:
``<col>_{ratio,ratio_err,num_mean,num_sem}`` for col in mask1-3 and unmasked,
``den_{mean,sem}``, ``n_objects``, ``abundance``, ``rad_mean_ckpch``,
``omega_b_over_omega_m``; 'tau_map' runs add ``K_tau`` and ``K_check_offset``,
beam-factor reports add ``beam_factor`` per simulation and
``beam_factor/{beam_factor_mean,beam_factor_published,R_compensated_lensfit}``.

Usage, from the scripts/ directory:
    python unbound_gas/simulated_dsigma_ratio_masked.py -p configs/unbound_gas/dsigma_ratio_z05_masked.yaml --workers 12
    # redraw from the profiles of an earlier run, no stacking:
    python unbound_gas/simulated_dsigma_ratio_masked.py -p ... --from-npz <..._profiles.npz>
"""

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.cosmology import FlatLambdaCDM

sys.path.append('../src/')
from stacker import SimulationStacker  # type: ignore
from loadIO import _get_data_filepath  # type: ignore
from utils import arcmin_to_comoving, comoving_to_arcmin, flamingo_label  # type: ignore
# Sibling module in this directory (Python puts the running script's own
# directory on sys.path); must come after the '../src/' append above.
from halo_stats import (sample_stats, format_table, write_stats_file, fit_label,
                        load_fitted_abundances, map_geometry, require_masked_map)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
})

# Fixed colours for the FLAMINGO feedback variants, keyed by feedback name.
# Kept in sync with simulated_kSZ_masked.py / make_fgas_profiles.py.
_FLAMINGO_COLOURS = {
    'L1_m9':           '#B30000',  # dark red (fiducial)
    'fgas-8sigma':     '#FF7F0E',  # orange
    'Jet_fgas-4sigma': '#C71585',  # magenta
    'Mstar-1sigma':             '#17BECF',  # cyan
    'Mstar-1sigma_fgas-4sigma': '#2E8B57',  # sea green
}

# Omega_b fallbacks, as in make_fgas_profiles.py: Illustris-1 does not store
# OmegaBaryon in its header, SIMBA never does.
_OMEGA_BARYON_ILLUSTRIS_DEFAULT = 0.0456
_OMEGA_BARYON_SIMBA_DEFAULT = 0.048

# Constants of mapMaker.make_sz_field and mapMaker.ionized_gas_masses; K below
# must use the same values for the tau -> ionized-gas conversion to be exact.
_M_P = 1.6726e-24  # g, proton mass
_SIGMA_T = 6.6524587158e-29 * 1.e2 ** 2  # cm^2, Thomson cross section
_SOLAR_MASS = 1.989e33  # g
_X_H = 0.76  # primordial hydrogen mass fraction
_MU_E = 2.0 / (1.0 + _X_H)  # mean mass per free electron (fully ionized), in m_p
_KPC_TO_CM = ((1. * u.kpc).to(u.cm)).value  # type: ignore

# Tolerance on the pixel-wise check of K against the cached ionized_gas map;
# the median offset is <= 2e-7 for every run, and every pixel within ~1e-6
# (float rounding; 2026-10-11).
_K_RTOL = 1e-5


def make_stacker(sim_type_name, sim, redshift):
    """Instantiate a simulation and name it as the kSZ masking figure does.

    Args:
        sim_type_name (str): 'IllustrisTNG', 'SIMBA' or 'FLAMINGO'.
        sim (dict): Config entry with 'name', 'snapshot' and, for SIMBA and
            FLAMINGO, 'feedback'.
        redshift (float): Redshift of the projection.

    Returns:
        tuple: (stacker, table label, legend label, Omega_b). The table label is
        plain text (halo-mass table, profiles npz); the legend label is TeX.
    """
    name, snapshot = sim['name'], sim['snapshot']
    if sim_type_name == 'IllustrisTNG':
        stacker = SimulationStacker(name, snapshot, z=redshift, simType=sim_type_name)
        try:
            omega_b = stacker.header['OmegaBaryon']
        except KeyError:
            omega_b = _OMEGA_BARYON_ILLUSTRIS_DEFAULT
        return stacker, name, name, omega_b
    if sim_type_name == 'SIMBA':
        stacker = SimulationStacker(name, snapshot, z=redshift, simType=sim_type_name,
                                    feedback=sim['feedback'])
        label = f"{name}_{sim['feedback']}"
        return stacker, label, label, _OMEGA_BARYON_SIMBA_DEFAULT
    if sim_type_name == 'FLAMINGO':
        feedback = sim['feedback']
        stacker = SimulationStacker(name, snapshot, z=redshift, simType=sim_type_name,
                                    feedback=feedback)
        # '-' instead of '_' so the table name is plain text; the row label
        # already names the suite, so the legend label does not.
        return (stacker, f"FLAMINGO {feedback}".replace('_', '-'),
                flamingo_label(feedback, prefix=False), stacker.header['OmegaBaryon'])
    raise ValueError(f"Unknown simulation type: {sim_type_name}")


def sim_cosmology(stacker):
    """Flat LCDM cosmology of a simulation, for arcmin <-> comoving conversions."""
    return FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'],
                         Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K)


def tau_to_ionized_mass(stacker, n_pixels):
    """Ionized-gas mass per pixel (Msun/h) per unit optical depth, for one map.

    A pixel of a tau map holds sigma_T N_e / A_pix with A_pix its physical area
    (mapMaker.make_sz_field), and the ionized-gas mass of the same electrons is
    N_e m_p mu_e (mapMaker.ionized_gas_masses), so the factor is
    A_pix m_p mu_e / sigma_T, converted from grams to Msun/h. The scale factor
    is the one make_sz_field uses, 1 / (1 + stacker.z).

    Args:
        stacker (SimulationStacker): Provides the header and redshift.
        n_pixels (int): Pixels per side of the map.

    Returns:
        float: K in Msun/h.
    """
    h = stacker.header['HubbleParam']
    a = 1. / (1. + stacker.z)
    pix_area_cm2 = (a * stacker.header['BoxSize'] * _KPC_TO_CM / h / n_pixels) ** 2
    return pix_area_cm2 * _M_P * _MU_E * h / (_SIGMA_T * _SOLAR_MASS)


def check_conversion(stacker, tau_map, n_pixels, projection, k_factor):
    """Check K against the cached unmasked ionized_gas map, pixel by pixel.

    Args:
        stacker (SimulationStacker): The simulation.
        tau_map (np.ndarray): The unmasked tau map being stacked.
        n_pixels (int): Its pixels per side.
        projection (str): Its projection.
        k_factor (float): K from :func:`tau_to_ionized_mass`.

    Returns:
        float or None: Median of ionized_gas / (K tau) - 1 over the pixels with
        non-negligible tau, or None if no ionized_gas map of this geometry is
        cached.

    Raises:
        ValueError: If the median differs from 0 by more than ``_K_RTOL``.
    """
    path = _get_data_filepath(stacker.simType, stacker.sim, stacker.snapshot, stacker.feedback,
                              'ionized_gas', n_pixels, projection, 'map', '2D', False, 2.0,
                              stacker.base_path)
    if not path.exists():
        print(f"  [K check] no cached {path.name}; K not checked")
        return None
    ion = np.load(path)
    good = tau_map > 1e-3 * np.median(tau_map[tau_map > 0])
    offset = float(np.median(ion[good] / tau_map[good]) / k_factor - 1)
    print(f"  [K check] median ionized_gas / (K tau) - 1 = {offset:.2e} ({path.name})")
    if abs(offset) > _K_RTOL:
        raise ValueError(f"tau -> ionized-gas factor off by {offset:.2e} against {path}")
    return offset


def map_paths(stacker, p, density):
    """Paths of the maps one simulation is stacked on, checked before any stacking.

    Never builds anything: a missing map is an error. Each masked map must
    have been built around the hosts of the SHAM galaxies at ``density``
    (``halo_stats.require_masked_map`` for the tau maps; the same test on the
    record for the ionized-gas fields), with the beam and the parent-mass cap
    of this run (its ``.sample.json`` record).

    Args:
        stacker (SimulationStacker): The simulation.
        p (dict): The stacking parameters of main().
        density (float): SHAM number density in (cMpc/h)^-3.

    Returns:
        dict: 'den' (total-matter field), 'unmasked' (tau map) and
        'mask<n>' (masked tau maps) -> path, plus 'meta' -> {column: sample
        record of the masked map}.

    Raises:
        FileNotFoundError: If a map is missing.
        ValueError: If a masked map was built for another sample, beam or cap.
    """
    z, proj = p['redshift'], p['projection']
    n_den, _ = map_geometry(stacker, z, p['pixel_size_2'])
    n_num, _ = map_geometry(stacker, z, p['pixel_size'])

    def path(p_type, n, data_type, mask=False, mask_rad=2.0):
        return _get_data_filepath(stacker.simType, stacker.sim, stacker.snapshot, stacker.feedback,
                                  p_type, n, proj, data_type, '2D', mask, mask_rad, stacker.base_path)

    # Unbeamed products are read from field files: the map cache path does not
    # record the beam, so a '_map.npy' at this size could be beam-convolved.
    num_type = 'map' if p['numerator'] == 'tau_map' else 'field'
    paths = {'den': path(p['particle_type_2'], n_den, 'field'),
             'unmasked': path(p['particle_type'], n_num, num_type), 'meta': {}}
    for key in ('den', 'unmasked'):
        if not paths[key].exists():
            raise FileNotFoundError(f"{paths[key]} is missing; refusing to build it here")
    for mask_rad in p['mask_radii']:
        col = f"mask{mask_rad:.0f}"
        paths[col] = path(p['particle_type'], n_num, num_type, True, mask_rad)
        if num_type == 'map':
            require_masked_map(stacker, p['particle_type'], proj, mask_rad, z, p['pixel_size'],
                               density)
        elif not (paths[col].exists() and Path(str(paths[col]) + '.sample.json').exists()):
            raise FileNotFoundError(f"{paths[col]} or its sample record is missing; build it "
                                    f"with unbound_gas/precompute_masked_particles.py first")
        with open(str(paths[col]) + '.sample.json') as f:
            meta = json.load(f)
        if num_type == 'field' and not (meta.get('sample') == 'sham-hosts'
                                        and np.isclose(meta.get('density', np.nan), density)
                                        and np.isclose(meta.get('pixel_size', np.nan), p['pixel_size'])):
            raise ValueError(f"{paths[col]} was built around sample {meta.get('sample')!r} at "
                             f"n = {meta.get('density')} and {meta.get('pixel_size')}' pixels; "
                             f"this run stacks SHAM at n = {density} on {p['pixel_size']}' pixels")
        if not (np.isclose(meta.get('beam', np.nan), p['beam_size'])
                and np.isclose(meta.get('halo_mass_upper', np.nan), p['halo_mass_upper'])):
            raise ValueError(f"{paths[col]} was built with beam {meta.get('beam')}' and parent-mass "
                             f"cap {meta.get('halo_mass_upper')}; this run uses {p['beam_size']}' "
                             f"and {p['halo_mass_upper']:.3e}")
        paths['meta'][col] = meta
    return paths


def _stack(stacker, array, halo_mask, p, pixel_size):
    """DeltaSigma profiles of one map around the selected galaxies, (n_radii, n_galaxies).

    ``pixel_size`` is the map's nominal pixel size in arcmin (stack_on_array
    takes the true one from the array shape for this filter).
    """
    return stacker.stack_on_array(
        array, filterType='DSigma', minRadius=p['min_radius'], maxRadius=p['max_radius'],
        numRadii=p['num_radii'], projection=p['projection'], radDistance=1.0,
        radDistanceUnits='arcmin', use_subhalos=True, halo_mass_avg=p['halo_mass_avg'],
        halo_mass_upper=p['halo_mass_upper'], z=p['redshift'], pixelSize=pixel_size,
        halo_mask=halo_mask, dr=p['annulus'])


def _mean_sem(profiles):
    """Mean over the galaxies and its standard error."""
    return (np.mean(profiles, axis=1),
            np.std(profiles, axis=1) / np.sqrt(profiles.shape[1]))


def stack_run(task):
    """Stack one simulation: the denominator and the four numerator columns.

    Runs in a worker process when ``--workers`` > 1, so it takes and returns
    plain data only.

    Args:
        task (dict): 'sim_type', 'sim', 'density' (SHAM density in
            (cMpc/h)^-3) and 'params' (the stacking parameters of main()).

    Returns:
        dict: Arrays and scalars for the profiles npz, under 'out' (keys
        without the simulation prefix), plus 'stats' (halo_stats.sample_stats).
    """
    t0 = time.time()
    p = task['params']
    stacker, table_label, _, omega_b = make_stacker(task['sim_type'], task['sim'], p['redshift'])
    tag = table_label
    density = task['density']
    print(f"[{tag}] SHAM density n = {density:.4e} (cMpc/h)^-3", flush=True)
    halo_mask, stats = sample_stats(stacker, table_label, use_subhalos=True,
                                    halo_abundance_target=density,
                                    halo_mass_avg=p['halo_mass_avg'],
                                    halo_mass_upper=p['halo_mass_upper'])
    out = {'n_objects': int(len(halo_mask)), 'abundance': density,
           'rad_mean_ckpch': stats['rad_mean'],
           'omega_b_over_omega_m': omega_b / stacker.header['Omega0']}
    paths = map_paths(stacker, p, density)
    for col, meta in paths['meta'].items():
        # The density matches (map_paths); the galaxy count must too.
        if meta.get('n_galaxies') != len(halo_mask):
            raise ValueError(f"{paths[col]} was built around {meta.get('n_galaxies')} galaxies; "
                             f"this run stacks {len(halo_mask)}")

    # Denominator: the unconvolved total-matter field.
    radii, den = _stack(stacker, np.load(paths['den']), halo_mask, p, p['pixel_size_2'])
    den_mean, den_sem = _mean_sem(den)
    out['den_mean'], out['den_sem'] = den_mean, den_sem
    print(f"[{tag}] denominator stacked ({paths['den'].name}), {time.time() - t0:.0f} s", flush=True)

    # Numerator: tau maps (0.5', beam) converted to ionized-gas mass by K, or
    # ionized-gas fields (already in Msun/h per pixel).
    n_num, _ = map_geometry(stacker, p['redshift'], p['pixel_size'])
    if p['numerator'] == 'tau_map':
        k_factor = tau_to_ionized_mass(stacker, n_num)
        out['K_tau'] = k_factor
    else:
        k_factor = 1.0
    for col in [f"mask{r:.0f}" for r in p['mask_radii']] + ['unmasked']:
        num_map = np.load(paths[col])
        if col == 'unmasked' and p['numerator'] == 'tau_map':
            out['K_check_offset'] = check_conversion(stacker, num_map, n_num, p['projection'],
                                                     k_factor)
        radii_num, num = _stack(stacker, num_map, halo_mask, p, p['pixel_size'])
        if not np.allclose(radii_num, radii):
            raise ValueError("numerator and denominator radii differ")
        num_mean, num_sem = _mean_sem(num * k_factor)  # Msun/h per arcmin^2
        omega_ratio = out['omega_b_over_omega_m']
        ratio = num_mean / den_mean / omega_ratio
        # Standard errors in quadrature, written so a zero numerator gives no NaN.
        err = np.sqrt((num_sem / den_mean) ** 2
                      + (num_mean * den_sem / den_mean ** 2) ** 2) / omega_ratio
        out.update({f'{col}_num_mean': num_mean, f'{col}_num_sem': num_sem,
                    f'{col}_ratio': ratio, f'{col}_ratio_err': err})
        print(f"[{tag}] {col} stacked ({paths[col].name}), {time.time() - t0:.0f} s", flush=True)
    out['radii_arcmin'] = radii
    return {'out': out, 'stats': stats}


def main(path2config, from_npz=None, workers=1):
    """Stack (or redraw) and plot the DeltaSigma-ratio masking figure.

    Args:
        path2config (str): Path to the YAML configuration file.
        from_npz (str, optional): Profiles file of an earlier run
            (``..._profiles.npz``); if given, the figure is redrawn from it
            instead of selecting and stacking. Defaults to None.
        workers (int, optional): Simulations stacked in parallel processes.
            Defaults to 1.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)
    stack_config = config.get('stack', {})
    plot_config = config.get('plot', {})

    params = {
        'redshift': stack_config.get('redshift', 0.5),
        'projection': stack_config.get('projection', 'yz'),
        # 'tau_map' (beamed tau maps x K) or 'ionized_gas_field' (unbeamed).
        'numerator': stack_config.get('numerator', 'tau_map'),
        'particle_type': stack_config.get('particle_type', 'tau'),
        'pixel_size': stack_config.get('pixel_size', 0.5),
        # Beam of the numerator maps, checked against the masked maps' records.
        'beam_size': stack_config.get('beam_size', 1.6),
        'particle_type_2': stack_config.get('particle_type_2', 'total'),
        'pixel_size_2': stack_config.get('pixel_size_2', 0.2),
        # Radii and annulus in arcmin.
        'min_radius': stack_config.get('min_radius', 1.0),
        'max_radius': stack_config.get('max_radius', 6.0),
        'num_radii': stack_config.get('num_radii', 9),
        'annulus': stack_config.get('annulus', 0.75),
        'mask_radii': [float(r) for r in stack_config.get('mask_radii', [1.0, 2.0, 3.0])],
        # SHAM parent-mass cap; the mass target is not used by SHAM but is
        # passed on as simulated_kSZ_masked.py does.
        'halo_mass_avg': stack_config.get('halo_mass_avg', 10 ** 13.22),
        'halo_mass_upper': stack_config.get('halo_mass_upper', 5 * 10 ** 14),
    }
    expected_ptype = {'tau_map': 'tau', 'ionized_gas_field': 'ionized_gas'}.get(params['numerator'])
    if expected_ptype is None:
        raise ValueError(f"stack.numerator must be 'tau_map' or 'ionized_gas_field', "
                         f"got {params['numerator']!r}")
    if params['particle_type'] != expected_ptype:
        raise ValueError(f"numerator {params['numerator']!r} needs particle_type "
                         f"{expected_ptype!r}, got {params['particle_type']!r}")
    redshift = params['redshift']
    # The masked maps are built around the hosts of the lensing-fit SHAM
    # galaxies, so the SHAM densities must come from the same fits.
    fit_path = stack_config.get('abundance_from_fit')
    if fit_path is None:
        raise ValueError("stack.abundance_from_fit is required: the masked maps are built around "
                         "the hosts of the lensing-fit SHAM galaxies")
    fitted_abundances = load_fitted_abundances(fit_path)

    now = datetime.now()
    figPath = Path(plot_config.get('fig_path', '../figures/')) / now.strftime("%Y-%m") / now.strftime("%m-%d")
    figPath.mkdir(parents=True, exist_ok=True)
    figName = plot_config.get('fig_name', 'dsigma_ratio_yz_masked')
    figType = plot_config.get('fig_type', 'pdf')
    plotErrorBars = plot_config.get('plot_error_bars', True)
    rowHeight = plot_config.get('row_height', 4.5)
    stem = f'{figName}_z{redshift}'

    # Simulations in config order; a simulation listed in two rows is stacked once.
    entries = []  # (row index, position in row, sim_type, sim, out_key)
    for row_idx, suite in enumerate(config['simulations']):
        for j, sim in enumerate(suite['sims']):
            out_key = '/'.join(str(k) for k in (suite['sim_type'], sim['name'], sim.get('feedback'))
                               if k is not None)
            entries.append((row_idx, j, suite['sim_type'], sim, out_key))

    # Beam-factor report (unbeamed numerator only): inputs checked here, before
    # any stacking. A redraw reuses the factors stored in its profiles file.
    bf_config = stack_config.get('beam_factor')
    if bf_config and params['numerator'] != 'ionized_gas_field':
        print("Note: stack.beam_factor is used only with numerator 'ionized_gas_field'; ignored")
        bf_config = None
    beamed = published = None
    if bf_config:
        radii_cfg = np.linspace(params['min_radius'], params['max_radius'], params['num_radii'])
        with np.load(bf_config['published_npz']) as f:
            published = dict(f)
        if not np.allclose(published['theta_arcmin'], radii_cfg):
            raise ValueError(f"{bf_config['published_npz']} has other radii than this run")
        missing = set(bf_config['average_over']) - {e[-1] for e in entries}
        if missing:
            raise KeyError(f"beam_factor.average_over names runs not in this config: {sorted(missing)}")
        if not from_npz:
            with np.load(bf_config['beamed_npz']) as f:
                beamed = dict(f)
            if not np.allclose(beamed['radii_arcmin'], radii_cfg):
                raise ValueError(f"{bf_config['beamed_npz']} has other radii than this run")
            for _, _, sim_type_name, sim, out_key in entries:
                # K_tau: the file is a 'tau_map' run; the density: same galaxies.
                if f'{out_key}/K_tau' not in beamed:
                    raise ValueError(f"{bf_config['beamed_npz']} has no 'tau_map' profiles of {out_key}")
                if not np.isclose(beamed[f'{out_key}/abundance'],
                                  fitted_abundances[fit_label(sim_type_name, sim)]):
                    raise ValueError(f"{bf_config['beamed_npz']} stacked {out_key} at another SHAM density")

    t0 = time.time()
    results = {}  # out_key -> dict of npz entries (without the key prefix)
    stats_rows = []
    if from_npz:
        if Path(from_npz).name != f'{stem}_profiles.npz':
            raise ValueError(f"--from-npz {from_npz!r} is not this config's profiles file "
                             f"({stem + '_profiles.npz'!r})")
        npz_in = dict(np.load(from_npz))
        print(f"Redrawing from {from_npz} (no selection or stacking)")
        for *_, out_key in entries:
            prefix = out_key + '/'
            results[out_key] = {k[len(prefix):]: v for k, v in npz_in.items() if k.startswith(prefix)}
            results[out_key]['radii_arcmin'] = npz_in['radii_arcmin']
    else:
        tasks, seen = [], set()
        for _, _, sim_type_name, sim, out_key in entries:
            if out_key in seen:
                continue
            seen.add(out_key)
            label = fit_label(sim_type_name, sim)
            if label not in fitted_abundances:
                raise KeyError(f"No lensing fit for {label!r} in {fit_path}; "
                               f"fitted: {sorted(fitted_abundances)}")
            tasks.append({'key': out_key, 'sim_type': sim_type_name, 'sim': sim,
                          'density': fitted_abundances[label], 'params': params})
        # Check every map before stacking anything, so a missing or mismatched
        # one stops the run here rather than after hours of stacking.
        for task in tasks:
            map_paths(make_stacker(task['sim_type'], task['sim'], redshift)[0], params,
                      task['density'])
        print(f"All maps present and matching their samples ({len(tasks)} simulations)", flush=True)
        if workers > 1:
            # spawn, not fork: the parent may already hold numba/OpenMP threads.
            ctx = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                done = list(pool.map(stack_run, tasks))
        else:
            done = [stack_run(task) for task in tasks]
        for task, res in zip(tasks, done):
            results[task['key']] = res['out']
            stats_rows.append(res['stats'])

    radii = next(iter(results.values()))['radii_arcmin']
    # Beam factor of these samples: the beamed unmasked numerator of a
    # 'tau_map' run over this run's, per simulation, against the published one
    # (the mean over the runs of the lensing paper). Nothing here may fail
    # before the profiles are saved, so a sample mismatch is only reported.
    if bf_config and (beamed is not None or all('beam_factor' in results[e[-1]] for e in entries)):
        print(f"\nBeam factor, beamed / unbeamed unmasked numerator (theta = {radii} arcmin):")
        for *_, out_key in entries:
            res = results[out_key]
            if beamed is not None:
                res['beam_factor'] = beamed[f'{out_key}/unmasked_num_mean'] / res['unmasked_num_mean']
                if int(beamed[f'{out_key}/n_objects']) != int(res['n_objects']):
                    print(f"  WARNING: {out_key} has {int(beamed[f'{out_key}/n_objects'])} galaxies "
                          f"in the beamed run and {int(res['n_objects'])} here")
            print(f"  {out_key:40s} {np.array2string(res['beam_factor'], precision=3)}")
        b_mean = np.mean([results[k]['beam_factor'] for k in bf_config['average_over']], axis=0)
        print(f"  mean over {len(bf_config['average_over'])} runs           "
              f"{np.array2string(b_mean, precision=3)}")
        print(f"  published                                {np.array2string(published['beam_factor'], precision=3)}")
        r_lensfit = published['ratio'] / b_mean
        print(f"  data / this mean - published compensated, in published sigma: "
              f"{np.array2string((r_lensfit - published['R_compensated']) / published['sigma_compensated'], precision=2)}")
        results['beam_factor'] = {'beam_factor_mean': b_mean,
                                  'beam_factor_published': published['beam_factor'],
                                  'R_compensated_lensfit': r_lensfit}
    elif bf_config:
        print("Note: the profiles file holds no beam factors; not reported")

    # Profiles and halo-sample table first, so a plotting failure loses no
    # stacking (redraw with --from-npz). A redraw never overwrites its source.
    profiles_out = {'radii_arcmin': radii}
    if 'beam_factor' in results:
        profiles_out.update({f'beam_factor/{k}': v for k, v in results.pop('beam_factor').items()})
    for *_, out_key in entries:
        for k, v in results[out_key].items():
            if k != 'radii_arcmin' and v is not None:
                profiles_out[f'{out_key}/{k}'] = v
    npz_path = figPath / f'{stem}_profiles.npz'
    if from_npz and npz_path.resolve() == Path(from_npz).resolve():
        print(f"Not rewriting {npz_path}: it is the --from-npz source")
    else:
        np.savez(npz_path, **profiles_out)
    fig_file = figPath / f'{stem}_masking_comparison.{figType}'
    if from_npz:
        print(f"Halo samples: as in {str(from_npz).replace('_profiles.npz', '_halo_masses.txt')}")
    else:
        numerator = (f"{params['particle_type']} x K (ionized gas)" if params['numerator'] == 'tau_map'
                     else f"{params['particle_type']} fields, masked at particle level")
        preamble = [
            f'Halo samples for {fig_file.name}',
            f'config          : {path2config}',
            f"numerator       : {numerator}, {params['pixel_size']}' pixels, "
            f"{params['beam_size']}' beam; masks around the SHAM hosts",
            f"denominator     : {params['particle_type_2']}, {params['pixel_size_2']}' pixels, no beam",
            f"filter          : DSigma, annulus {params['annulus']}'    projection: {params['projection']}",
            f'redshift        : {redshift}',
            f"selection       : SHAM on SubhaloMStar, n fitted on lensing per simulation ({fit_path}), "
            f"parent-mass cap {params['halo_mass_upper']:.3e} Msun/h",
        ] + [f"SHAM density    : {s['label']}: {results[t['key']]['abundance']:.4e} (cMpc/h)^-3"
             for s, t in zip(stats_rows, tasks)]
        print('\n' + '\n'.join(preamble) + '\n\n' + format_table(stats_rows) + '\n', flush=True)
        write_stats_file(figPath / f'{stem}_halo_masses.txt', stats_rows, preamble=preamble)

    # ------------------------------------------------------------------
    # Figure: one row per suite, columns = mask radii, then unmasked.
    # ------------------------------------------------------------------
    columns = [f"mask{r:.0f}" for r in params['mask_radii']] + ['unmasked']
    nRows, nCols = len(config['simulations']), len(columns)
    fig, axes = plt.subplots(nRows, nCols, figsize=(18, rowHeight * nRows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    row_rad_arcmin = {}  # mean host R200m of each row's first simulation, arcmin
    cosmo_axis = None  # cosmology of the comoving top axis: the first IllustrisTNG run's
    for row_idx, j, sim_type_name, sim, out_key in entries:
        suite = config['simulations'][row_idx]
        # Colours as in simulated_kSZ_masked.py.
        if sim_type_name == 'FLAMINGO':
            fallback = matplotlib.colormaps['plasma'](np.linspace(0.2, 0.85, len(suite['sims'])))  # type: ignore
            colour = _FLAMINGO_COLOURS.get(sim['feedback'], fallback[j])
        else:
            cmap = matplotlib.colormaps[{'SIMBA': 'hsv', 'IllustrisTNG': 'twilight'}[sim_type_name]]  # type: ignore
            colour = cmap(np.linspace(0.2, 0.85, len(suite['sims'])))[j]
        stacker, _, legend_label, _ = make_stacker(sim_type_name, sim, redshift)
        cosmo = sim_cosmology(stacker)
        if cosmo_axis is None and sim_type_name == 'IllustrisTNG':
            cosmo_axis = cosmo
        res = results[out_key]
        if row_idx not in row_rad_arcmin:
            row_rad_arcmin[row_idx] = comoving_to_arcmin(float(res['rad_mean_ckpch']), redshift, cosmo)
        for col_idx, col in enumerate(columns):
            ax = axes[row_idx, col_idx]
            ratio, err = res[f'{col}_ratio'], res[f'{col}_ratio_err']
            ax.plot(radii, ratio, label=legend_label, color=colour, lw=2, marker='o')
            if plotErrorBars:
                ax.fill_between(radii, ratio - err, ratio + err, color=colour, alpha=0.2)
    if cosmo_axis is None:
        cosmo_axis = sim_cosmology(make_stacker(entries[0][2], entries[0][3], redshift)[0])

    # The measurement, on the unmasked column only (as in the kSZ figure). The
    # default keys are those of the raw ratio file; plot.data_keys overrides
    # them (e.g. for the beam-compensated file).
    if plot_config.get('plot_data', False):
        data = np.load(plot_config['data_path'])
        key = plot_config.get('data_key', 'source_bin_0')
        dk = {'theta': f'{key}/ksz_theta_arcmin', 'value': f'{key}/ratio', 'error': f'{key}/ratio_err'}
        dk.update(plot_config.get('data_keys', {}))
        for row_idx in range(nRows):
            axes[row_idx, -1].errorbar(data[dk['theta']], data[dk['value']],
                                       yerr=data[dk['error']], fmt='s', color='k',
                                       label=plot_config.get('data_label', 'data'),
                                       markersize=5, zorder=10)

    def forward_arcmin(arcmin):
        return arcmin_to_comoving(arcmin, redshift, cosmo_axis)

    def inverse_arcmin(comoving):
        return comoving_to_arcmin(comoving, redshift, cosmo_axis)

    for row_idx in range(nRows):
        for col_idx, col in enumerate(columns):
            ax = axes[row_idx, col_idx]
            ax.axhline(1.0, color='gray', ls=':', lw=1.5, zorder=0)
            if col != 'unmasked':
                ax.axvline(row_rad_arcmin[row_idx] * params['mask_radii'][col_idx],
                           color='k', linestyle='--', lw=1)
            if row_idx == nRows - 1:
                ax.set_xlabel('R [arcmin]')
            if col_idx == 0:
                ax.set_ylabel(plot_config.get('ylabel', r'$f_{\rm gas}^{\rm obs}$'))
            if row_idx == 0:
                secax_x = ax.secondary_xaxis('top', functions=(forward_arcmin, inverse_arcmin))
                secax_x.set_xlabel('R [ckpc/h]')
                ax.set_title(r'No Masking' if col == 'unmasked' else
                             f'Masked ($R_{{mask}} = {params["mask_radii"][col_idx]:.0f} R_{{200m}}$)')
            ax.set_xlim(0.0, params['max_radius'] + 0.5)
            if plot_config.get('ylim') is not None:
                ax.set_ylim(*plot_config['ylim'])
            ax.grid(True)
        # Each row's legend (simulations and the data) goes in plot.legend_column
        # (default: the unmasked column, as in the kSZ figure).
        handles, labels = axes[row_idx, -1].get_legend_handles_labels()
        axes[row_idx, plot_config.get('legend_column', -1)].legend(
            handles, labels, loc=plot_config.get('legend_loc', 'upper left'), fontsize=12)

    showSuptitle = plot_config.get('suptitle', False)
    if showSuptitle:
        fig.suptitle(rf'Stacked $\Delta\Sigma$ ratio profiles, z={redshift}', fontsize=22)
    fig.tight_layout(rect=(0.03, 0, 1, 0.97 if showSuptitle else 1))
    for row_idx, suite in enumerate(config['simulations']):
        bbox = axes[row_idx, 0].get_position()
        fig.text(0.02, 0.5 * (bbox.y0 + bbox.y1), suite.get('row_label', suite['sim_type']),
                 fontsize=20, va='center', rotation=90, ha='center')
    fig.savefig(fig_file, dpi=300)  # type: ignore
    plt.close(fig)
    print(f"Figure: {fig_file}")
    print(f'Done!!! time taken = {time.time() - t0:.0f} seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeltaSigma-ratio version of the kSZ masking figure.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/unbound_gas/dsigma_ratio_z05_masked.yaml',
                        help='Path to the configuration file.')
    parser.add_argument('--from-npz', type=str, default=None,
                        help='Redraw from the profiles .npz of an earlier run instead of stacking.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Simulations stacked in parallel processes.')
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")
    main(**args)
