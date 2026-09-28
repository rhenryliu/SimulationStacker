"""make_dog_calibration.py
========================
Round 3A, Stage 3 (``docs/cross_corr/open-items.md`` O-02): the calibration
factor for a difference-of-Gaussians (DoG) kernel, formalism Eq. (22),

    W_DoG(theta) = G(theta; sigma1) - G(theta; sigma2),
    K(k) = exp(-k^2 sigma1^2 / 2) - exp(-k^2 sigma2^2 / 2),

which is compensated (``K(0) = 0``) and has a strictly positive window
(``K(k) > 0`` for ``k > 0``).  Under a positive window the window term of
formalism Eq. (48) is a genuine covariance that cannot change sign, and the
filtered coefficients obey Cauchy-Schwarz.  Formalism §4.5 proposes the DoG
as the second, independent test of whether ``C_F - 1`` is window smearing.

This is a standalone test script (the kernel is not added to ``src/``): it
reuses the round-two field loading and galaxy sample of
``make_calibration_factor.py`` and the payload format of its
``flatten_for_npz``, so the output keys (``C_b_<F>``, ``Cerr_b_<F>``,
``x_t_b_<F>``, ...) mean exactly what they mean in
``data/cross_corr_C/calibration_*.npz``.  The DoG grid is a fixed-ratio
family, ``sigma2 = q sigma1`` with ``sigma1`` log-spaced; the "radii" axis of
the payload is ``sigma1`` in arcmin, and filters are named ``DoG_q=<q>``.
Comparison with DSigma is made at matched ``k_50`` in
``round3a_dog_analysis.py``, so no matching rule between ``sigma1`` and an
aperture is needed.

Usage
-----
    cd scripts/
    python cross_corr/make_dog_calibration.py \
        -p configs/cross_corr/round3a_z05.yaml [--sim TNG300-1]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.append('../src/')
sys.path.append(str(Path(__file__).resolve().parent))

import rprofiles as rp
import round3a_lib as lib
from stacker import SimulationStacker
from utils import comoving_to_arcmin

from make_calibration_factor import (GAS_FIELDS, add_convention_t,
                                     calibration_factors, flatten_for_npz,
                                     load_fields_and_baryon_fraction)
from make_r_profiles import sim_label


def dog_grid(r3cfg):
    """Return the DoG widths and ratios from the ``round3a`` config block.

    Args:
        r3cfg (dict): The ``round3a`` config block.

    Returns:
        tuple: ``(sigma1, ratios)``, arcmin and dimensionless.
    """
    sigma1 = np.geomspace(float(r3cfg['dog_sigma1_min']),
                          float(r3cfg['dog_sigma1_max']),
                          int(r3cfg['dog_n_sigma']))
    return sigma1, [float(q) for q in r3cfg['dog_ratios']]


def dog_filters(deltas, pixel_arcmin, sigma1, ratios, nbar_pix, f_b,
                n_jk_side=4):
    """DoG amplitudes for every width and ratio, both matter conventions.

    Args:
        deltas (dict): Overdensity maps keyed 'g', 'e', 'b', 'm'.
        pixel_arcmin (float): Pixel size, arcmin.
        sigma1 (np.ndarray): Inner widths, arcmin.
        ratios (sequence): ``sigma2/sigma1`` values.
        nbar_pix (float): Mean galaxies per pixel.
        f_b (float): Map-derived baryon fraction.
        n_jk_side (int, optional): Jackknife blocks per side.

    Returns:
        tuple: ``(filters, zero_lags)`` where ``filters`` has the layout of
        ``make_calibration_factor.build_filters`` (``{name: {'Y', 'Y_jk',
        'mask'}}``) and ``zero_lags`` maps each name to the kernel's central
        values, one per width.
    """
    n = next(iter(deltas.values())).shape[0]
    filters, zero_lags = {}, {}
    for q in ratios:
        name = f'DoG_q={q:g}'
        specs = [lib.dog_kernel_spectrum(n, pixel_arcmin, float(s), q * s)
                 for s in sigma1]
        k0s = [lib.zero_lag(sp, n) for sp in specs]
        amps = lib.filtered_amplitudes(deltas, specs, k0s, nbar_pix=nbar_pix,
                                       n_jk_side=n_jk_side)
        del specs
        entry = {'Y': dict(amps['Y']), 'Y_jk': dict(amps['Y_jk']),
                 'mask': np.ones(len(sigma1), dtype=bool)}
        add_convention_t(entry['Y'], f_b)
        add_convention_t(entry['Y_jk'], f_b)
        filters[name] = entry
        zero_lags[name] = np.asarray(k0s)
    return filters, zero_lags


def process_simulation(sim_type, sim_entry, cfg, r3cfg, verbose=True):
    """Run the DoG sweep for one simulation.

    Args:
        sim_type (str): Simulation suite.
        sim_entry (dict): One entry of the config ``sims`` list.
        cfg (dict): The ``stack`` config block.
        r3cfg (dict): The ``round3a`` config block.
        verbose (bool, optional): Print progress.

    Returns:
        list: Paths written.
    """
    name = sim_entry['name']
    snapshot = sim_entry['snapshot']
    feedback = sim_entry.get('feedback')
    n_pixels = sim_entry['n_pixels']
    label = sim_label(sim_type, name, feedback)

    z_cfg = float(sim_entry.get('redshift', cfg.get('redshift', 0.5)))
    stacker = SimulationStacker(name, snapshot, nPixels=n_pixels,
                                simType=sim_type, feedback=feedback, z=z_cfg)
    z_true = float(np.asarray(stacker.header['Redshift']).ravel()[0])
    lbox = float(stacker.header['BoxSize'])
    theta_arcmin = comoving_to_arcmin(lbox, z_true, cosmo=stacker.cosmo)
    pixel_arcmin = theta_arcmin / n_pixels

    sigma1, ratios = dog_grid(r3cfg)
    n_jk_side = int(cfg.get('n_jk_side', rp.N_JK_SIDE))
    target = float(cfg.get('halo_abundance_target', 5e-4))
    upper = cfg.get('parent_mass_upper', 5e14)
    upper = None if upper is None else float(upper)
    out_dir = Path(r3cfg.get('out_path', '../data/cross_corr_C/round3a/'))

    if verbose:
        print(f'\n{"=" * 70}\n{label}  ({sim_type}, snapshot {snapshot})\n'
              f'{"=" * 70}')
        print(f'  pixel = {pixel_arcmin:.5f} arcmin; sigma1 = '
              f'{np.array2string(sigma1, precision=3)}; q = {ratios}')

    subhalos = stacker.loadSubHalos()
    written = []
    for projection in cfg.get('projections', ['yz']):
        t0 = time.time()
        deltas, f_b = load_fields_and_baryon_fraction(
            stacker, n_pixels, projection, cfg, verbose=verbose)
        galaxies, halo_mask = rp.make_galaxy_field(
            stacker, projection, n_pixels, target, parent_mass_upper=upper,
            subhalos=subhalos)
        n_gal = int(halo_mask.size)
        nbar_pix = galaxies.sum() / float(n_pixels * n_pixels)
        deltas['g'] = rp.to_overdensity(galaxies)
        del galaxies
        if verbose:
            print(f'    {n_gal} SHAM galaxies; DoG sweep ...', flush=True)

        filters, zero_lags = dog_filters(deltas, pixel_arcmin, sigma1, ratios,
                                         nbar_pix, f_b, n_jk_side=n_jk_side)
        del deltas

        meta = {'label': label, 'sim_type': sim_type, 'sim_name': str(name),
                'feedback': str(feedback), 'snapshot': snapshot,
                'projection': projection, 'redshift': z_true,
                'n_pixels': n_pixels, 'pixel_arcmin': pixel_arcmin,
                'boxsize_ckpc_h': lbox, 'n_galaxies': n_gal,
                'nbar_pix': nbar_pix, 'abundance_target': target, 'f_b': f_b,
                'n_jk': n_jk_side ** 2,
                'filters': np.array(sorted(filters), dtype=object),
                'gas_fields': np.array(GAS_FIELDS, dtype=object),
                'radii_are': 'sigma1_arcmin'}
        payload = flatten_for_npz(filters, sigma1, f_b, meta)
        payload['sigma1'] = sigma1
        for fname, k0 in zero_lags.items():
            q = float(fname.split('=')[1])
            payload[f'sigma2_{fname}'] = q * sigma1
            payload[f'k0_{fname}'] = k0

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir /
                    f'dog_calibration_{label}_{snapshot}_{projection}.npz')
        tmp_path = out_path.with_name(out_path.stem + '.tmp.npz')
        np.savez(tmp_path, **payload)
        tmp_path.replace(out_path)
        written.append(out_path)

        if verbose:
            print(f'    saved {out_path} ({time.time() - t0:.0f} s)')
            for fname in sorted(filters):
                C = calibration_factors(filters[fname]['Y'], gas='b')['C']
                print(f'      {fname:10s} C_b(sigma1) = '
                      f'{np.array2string(C, precision=4)}')
    return written


def main(path2config, sim=None, feedback=None, verbose=True):
    """Run the DoG sweep for every simulation in a config.

    Args:
        path2config (str): Path to the YAML configuration file.
        sim (str, optional): Only this simulation name.
        feedback (str, optional): Only this feedback variant.
        verbose (bool, optional): Print progress.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)
    cfg = config.get('stack', {})
    r3cfg = config.get('round3a', {})
    t_start = time.time()
    written = []
    for suite in config['simulations']:
        for entry in suite['sims']:
            if sim is not None and entry['name'] != sim:
                continue
            if feedback is not None and entry.get('feedback') != feedback:
                continue
            written.extend(process_simulation(suite['sim_type'], entry, cfg,
                                              r3cfg, verbose=verbose))
    print(f'\n{"=" * 70}\nWrote {len(written)} file(s) in '
          f'{(time.time() - t_start) / 60:.1f} minutes:')
    for path in written:
        print(f'  {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Round 3A Stage 3: the DoG calibration factor.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/round3a_z05.yaml')
    parser.add_argument('--sim', type=str, default=None)
    parser.add_argument('--feedback', type=str, default=None)
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
