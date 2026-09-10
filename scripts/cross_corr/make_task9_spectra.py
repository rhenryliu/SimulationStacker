"""make_task9_spectra.py
======================
The 2D power spectra Task 9 needs, and the ``k_50`` mapping from aperture to
wavenumber.

Task 9 of ``docs/cross_correlation_notes_v0.2_addendum.md`` asks for a lower
panel showing the suppression, "overlaid with the directly measured
``P_tt^hydro / P_mm^hydro`` from the same box, plotted against ``k_50(R;F)`` so
that filters are compared at matched wavenumber rather than matched aperture."
Two things in that sentence cannot come from ``data/cross_corr_C/*.npz``:

1. **The unfiltered suppression** ``P_tt(k)/P_mm(k)``.  The filtered analogue
   ``Y_tt/Y_mm`` *is* in the npz, but overlaying it would make the comparison
   trivially true -- addendum Section 5.5 calls that the "identity check"
   (rung 1), whereas the overlay is meant to be the "suppression check"
   (rung 4).  So the spectra have to be measured from the maps.

2. **``k_50(R;F)``**, the wavenumber at which the signed cumulative
   contribution to the filtered amplitude crosses half.  The addendum's
   Section 1.4 table computes this for ``P_2D ∝ k^n``, but its own Appendix B
   shows the answer moving by a factor 1.8 between ``n = -1`` and ``n = -1.5``,
   so the power-law approximation is not good enough to label an axis with.
   Here it is computed against the **measured** CDM spectrum of each run and
   the **pixelized** kernel normalization, matching the amplitudes the
   ``.npz`` actually contains.

Both are per-run quantities and both need the cached projected fields, so this
runs on a compute node and caches its output; ``plot_task9.py`` then works from
the cache on a login node.

Cross-spectra are measured here rather than in ``theory.measured_p2d_from_map``
(which is auto-only) so that ``src/`` stays untouched; the binning and
normalization follow that function exactly.

Usage
-----
    cd scripts/
    python cross_corr/make_task9_spectra.py \
        -p configs/cross_corr/calibration_z05.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.fft
import yaml

sys.path.append('../src/')
sys.path.append(str(Path(__file__).resolve().parent))

import kernels as kn
import rprofiles as rp
import theory as th
from stacker import SimulationStacker
from utils import comoving_to_arcmin

from make_calibration_factor import (GAS_FIELDS, YT_USABLE_FRACTION,
                                     calibration_radii, radius_index,
                                     load_fields_and_baryon_fraction)
from make_r_profiles import sim_label


#: Fractions of the cumulative kernel response to report.  k_50 is the
#: addendum's matched-wavenumber axis; k_05 and k_95 give the width.
QUANTILES = (0.05, 0.50, 0.95)


def cross_p2d(delta_a, delta_b, pixel_arcmin, n_bins=400):
    """Measure the isotropic 2D cross power spectrum of two maps.

    The auto case reduces to ``theory.measured_p2d_from_map`` exactly; the
    binning, area normalization and wavenumber grid are copied from it so the
    two are directly comparable.

    Args:
        delta_a (np.ndarray): Square zero-mean overdensity map.
        delta_b (np.ndarray): Second map, same grid.
        pixel_arcmin (float): Angular pixel size, arcmin.
        n_bins (int, optional): Number of wavenumber bins.  Defaults to 400.

    Returns:
        tuple: ``(k, P_2D)`` with ``k`` in 1/arcmin and ``P_2D`` in arcmin^2.
        The cross spectrum is real by construction for real input maps; the
        imaginary part is discarded after the conjugate product.

    Raises:
        ValueError: If the maps are not square or do not share a grid.
    """
    delta_a = np.asarray(delta_a, dtype=np.float64)
    delta_b = np.asarray(delta_b, dtype=np.float64)
    if delta_a.shape != delta_b.shape:
        raise ValueError(f'Shape mismatch: {delta_a.shape} vs {delta_b.shape}.')
    if delta_a.ndim != 2 or delta_a.shape[0] != delta_a.shape[1]:
        raise ValueError(f'Maps must be square, got {delta_a.shape}.')

    n = delta_a.shape[0]
    area = (n * pixel_arcmin) ** 2

    fa = scipy.fft.rfft2(delta_a, workers=-1)
    fb = fa if delta_b is delta_a else scipy.fft.rfft2(delta_b, workers=-1)
    power = np.real(fa * np.conj(fb)) * area / float(n * n) ** 2

    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=pixel_arcmin)
    ky = 2.0 * np.pi * np.fft.rfftfreq(n, d=pixel_arcmin)
    kk = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)

    edges = np.linspace(0.0, kk.max(), n_bins + 1)
    idx = np.clip(np.digitize(kk.ravel(), edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    k_sum = np.bincount(idx, weights=kk.ravel(), minlength=n_bins)
    p_sum = np.bincount(idx, weights=power.ravel(), minlength=n_bins)

    good = counts > 0
    return k_sum[good] / counts[good], p_sum[good] / counts[good]


def filter_kernel_ft(k, R, filter_name, dr, r0=None, rmax=None):
    """Return the harmonic-space kernel of one named filter variant.

    Wraps ``kernels.py`` and adds the Park et al. (2021) Y transform, which
    that module does not carry:

        W_Y(k; R, Rmax) = W_Sigma(k; R) - W_Sigma(k; Rmax)

    the harmonic-space image of the map-level identity the calibration sweep
    uses (addendum Eq. A4).

    Args:
        k (np.ndarray): Wavenumbers, 1/arcmin.
        R (float): Aperture radius, arcmin.
        filter_name (str): ``'Sigma'``, ``'DSigma'``, ``'Upsilon_R0=<r0>'`` or
            ``'Ytransform_Rmax=<rmax>'``.
        dr (float): Annulus width, arcmin.
        r0 (float, optional): Upsilon reference radius; parsed from the name
            when omitted.
        rmax (float, optional): Y-transform reference radius; parsed from the
            name when omitted.

    Returns:
        np.ndarray: Kernel evaluated at ``k``.

    Raises:
        ValueError: If ``filter_name`` is not recognized.
    """
    if filter_name == 'Sigma':
        return kn.w_sigma(k, R, dr)
    if filter_name == 'DSigma':
        return kn.w_dsigma(k, R, dr)
    if filter_name.startswith('Upsilon'):
        ref = float(filter_name.split('=')[1]) if r0 is None else r0
        return kn.w_upsilon(k, R, dr, ref)
    if filter_name.startswith('Ytransform'):
        ref = float(filter_name.split('=')[1]) if rmax is None else rmax
        return kn.w_sigma(k, R, dr) - kn.w_sigma(k, ref, dr)
    raise ValueError(f'Unrecognized filter {filter_name!r}.')


def response_quantiles(k, p_2d, R, filter_name, dr, quantiles=QUANTILES):
    """Locate where a filter's response to a given spectrum accumulates.

    The filtered amplitude is ``Y(R) = int k dk/(2 pi) P(k) W(k;R)``, so the
    contribution density is ``k P(k) W(k;R)``.  This integrates it cumulatively
    and reports the wavenumbers at which the *signed* cumulative crosses the
    requested fractions of the total.

    The signed convention matters: compensated kernels change sign, so the
    cumulative is not monotonic and the "median" wavenumber is where the
    running integral first reaches half its final value.  This is the same
    definition the addendum's Section 1.4 uses, evaluated here against a
    measured spectrum rather than a power law.

    Args:
        k (np.ndarray): Wavenumbers, 1/arcmin, strictly increasing.
        p_2d (np.ndarray): Power spectrum at ``k``, arcmin^2.
        R (float): Aperture radius, arcmin.
        filter_name (str): Filter variant name.
        dr (float): Annulus width, arcmin.
        quantiles (sequence, optional): Fractions to locate.  Defaults to
            :data:`QUANTILES`.

    Returns:
        np.ndarray: Wavenumbers at the requested fractions, 1/arcmin; NaN
        where the total amplitude is too close to zero for the fraction to be
        meaningful (which happens for Upsilon at ``R = R0`` and for the Y
        transform at ``R = Rmax``, both identically zero by construction).
    """
    k = np.asarray(k, dtype=np.float64)
    good = k > 0
    k = k[good]
    p_2d = np.asarray(p_2d, dtype=np.float64)[good]

    integrand = k * p_2d * filter_kernel_ft(k, R, filter_name, dr)
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(k))])
    total = cumulative[-1]

    # Upsilon at R = R0 and the Y transform at R = Rmax give a kernel that is
    # identically zero, so total, cumulative and their scale are all exactly
    # 0.0.  A purely relative threshold collapses to 0.0 < 0.0 there and falls
    # through to a 0/0 divide, so the absolute scale is tested first.
    out = np.full(len(quantiles), np.nan)
    scale = float(np.max(np.abs(cumulative), initial=0.0))
    if not np.isfinite(total) or scale == 0.0 or abs(total) < 1e-12 * scale:
        return out

    frac = cumulative / total
    for i, q in enumerate(quantiles):
        hits = np.flatnonzero(frac >= q)
        if hits.size:
            out[i] = k[min(hits[0], len(k) - 1)]
    return out


def process_simulation(sim_type, sim_entry, cfg, out_dir, verbose=True):
    """Measure the spectra and k-quantiles for one simulation.

    Args:
        sim_type (str): Simulation suite.
        sim_entry (dict): One entry of the config ``sims`` list.
        cfg (dict): The ``stack`` config block.
        out_dir (pathlib.Path): Output directory.
        verbose (bool, optional): Print progress.  Defaults to True.

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

    radii, r0_list, rmax_list = calibration_radii(cfg)
    dr = float(cfg.get('dr_arcmin', rp.DR_ARCMIN))

    filter_names = (['Sigma', 'DSigma']
                    + [f'Upsilon_R0={r:g}' for r in r0_list]
                    + [f'Ytransform_Rmax={r:g}' for r in rmax_list])

    if verbose:
        print(f'\n{"=" * 70}\n{label}  ({sim_type}, snapshot {snapshot})\n'
              f'{"=" * 70}')
        print(f'  z = {z_true:.4f}, box = {lbox / 1000:.1f} cMpc/h, '
              f'pixel = {pixel_arcmin:.5f} arcmin')

    written = []
    for projection in cfg.get('projections', ['yz']):
        t0 = time.time()
        if verbose:
            print(f'\n  --- projection {projection} ---', flush=True)

        deltas, f_b = load_fields_and_baryon_fraction(
            stacker, n_pixels, projection, cfg, verbose=verbose)

        # Convention T map, built explicitly here rather than recombined:
        # the point of this script is the UNfiltered spectrum ratio, and the
        # recombination identity is only proven for filtered amplitudes.
        delta_t = (1.0 - f_b) * deltas['m'] + f_b * deltas['b']

        if verbose:
            print('    measuring 2D spectra ...', flush=True)
        payload = {'radii': radii}
        spectra = {}
        for tag, (a, b) in (('mm', ('m', 'm')), ('tt', (None, None)),
                            ('bm', ('b', 'm')), ('em', ('e', 'm')),
                            ('bb', ('b', 'b'))):
            if tag == 'tt':
                k, p = cross_p2d(delta_t, delta_t, pixel_arcmin)
            else:
                k, p = cross_p2d(deltas[a], deltas[b], pixel_arcmin)
            spectra[tag] = p
            payload[f'P2D_{tag}'] = p
        payload['k_arcmin'] = k
        del delta_t, deltas

        # Convert the wavenumber axis to h/Mpc for the figures.  1 arcmin
        # subtends comoving_to_arcmin^-1; k[h/Mpc] = k[1/arcmin] / (Mpc per
        # arcmin).
        mpc_per_arcmin = (lbox / theta_arcmin) / 1000.0
        payload['k_h_mpc'] = k / mpc_per_arcmin
        payload['mpc_per_arcmin'] = mpc_per_arcmin

        if verbose:
            print('    locating response quantiles ...', flush=True)
        for fname in filter_names:
            q = np.full((len(radii), len(QUANTILES)), np.nan)
            for i, R in enumerate(radii):
                q[i] = response_quantiles(k, spectra['mm'], float(R), fname, dr)
            payload[f'kquant_{fname}'] = q                    # 1/arcmin
            payload[f'kquant_hmpc_{fname}'] = q / mpc_per_arcmin

        payload['quantiles'] = np.asarray(QUANTILES)
        for key, value in {
                'label': label, 'sim_type': sim_type, 'snapshot': snapshot,
                'projection': projection, 'redshift': z_true,
                'n_pixels': n_pixels, 'pixel_arcmin': pixel_arcmin,
                'boxsize_ckpc_h': lbox, 'f_b': f_b, 'dr_arcmin': dr,
                'filters': np.array(filter_names, dtype=object)}.items():
            payload[f'meta_{key}'] = np.array(value)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'task9_spectra_{label}_{snapshot}_{projection}.npz'
        np.savez(out_path, **payload)
        written.append(out_path)
        if verbose:
            print(f'    saved {out_path}  ({time.time() - t0:.1f} s)')
            ratio = spectra['tt'] / spectra['mm']
            sel = (payload['k_h_mpc'] > 0.3) & (payload['k_h_mpc'] < 10)
            print(f'      P_tt/P_mm over k=0.3-10 h/Mpc: '
                  f'{np.nanmin(ratio[sel]):.3f} - {np.nanmax(ratio[sel]):.3f}')
            for fname in ('DSigma', 'Sigma'):
                q = payload[f'kquant_hmpc_{fname}'][0]
                print(f"      k_50(R=1') for {fname}: {q[1]:.3f} h/Mpc "
                      f'(k_05={q[0]:.3f}, k_95={q[2]:.3f})')

    return written


def main(path2config, sim=None, feedback=None, verbose=True):
    """Run the spectrum measurement for every simulation in a config.

    Args:
        path2config (str): Path to the YAML configuration file.
        sim (str, optional): Only process this simulation name.
        feedback (str, optional): Only process this feedback variant.
        verbose (bool, optional): Print progress.  Defaults to True.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)
    cfg = config.get('stack', {})
    out_dir = Path(config.get('plot', {}).get('npz_path',
                                              '../data/cross_corr_C/'))

    t_start = time.time()
    written = []
    for suite in config['simulations']:
        for entry in suite['sims']:
            if sim is not None and entry['name'] != sim:
                continue
            if feedback is not None and entry.get('feedback') != feedback:
                continue
            written.extend(process_simulation(suite['sim_type'], entry, cfg,
                                              out_dir, verbose=verbose))

    print(f'\n{"=" * 70}')
    print(f'Wrote {len(written)} file(s) in '
          f'{(time.time() - t_start) / 60:.1f} minutes:')
    for path in written:
        print(f'  {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Measure the 2D spectra and k_50 mapping Task 9 needs.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/calibration_z05.yaml')
    parser.add_argument('--sim', type=str, default=None)
    parser.add_argument('--feedback', type=str, default=None)
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
