"""make_ck_spectra.py
===================
Round 3A, Stage 2 (``docs/cross_corr/open-items.md`` O-01, with O-05 and
O-06): the harmonic-space calibration factor

    C(k) = P_bm(k) P_gm(k) / (P_mm(k) P_gb(k)),

and the exact split of the filtered calibration factor ``C_F`` into a window
part and a mediation part (formalism Eq. 48; ``round3a_lib`` for the
algebra):

    C_F = W_F M_F,   M_F = Y_gb^med / Y_gb,   Y_gb^med = int dmu P_bm P_gm/P_mm.

``M_F`` is identically 1 when mediation holds at every ``k`` (``C(k) = 1``),
so it isolates mediation failure; ``W_F`` is the window term that is present
even under exact mediation.

What is computed, per run and projection, from the cached maps of the
round-two sweep (``make_calibration_factor.py``):

- the same CDM, baryon and electron overdensity maps, and the same fiducial
  SHAM galaxy map, plus three more galaxy samples: the other number density
  (``alt``, for O-06) and the fiducial sample split at its median parent-halo
  mass (``lo``, ``hi``, for O-05);
- every auto and cross spectrum among ``{m, b, e}`` and each galaxy sample,
  in linear ``|k|`` bins over the full Fourier plane;
- for every kernel -- Sigma and DSigma at the 19 calibration apertures, and
  the difference-of-Gaussians (DoG) grid of Stage 3 -- the **exact**
  filtered amplitudes as sums over Fourier modes (Parseval), the mediated
  galaxy-gas amplitudes ``Y_gX^med`` for ``X`` in {b, e} in both matter
  conventions (only the smooth ``eta(k) = P_Xm/P_mm`` is binned), and the
  doubly filtered amplitudes of formalism Eq. (40) for the fiducial sample;
- the binned kernel weights ``w(bin) = sum K``, for the binned cross-check.

**Regression check.**  For the fiducial sample the exact amplitudes of the
Sigma and DSigma kernels must reproduce the committed
``data/cross_corr_C/calibration_*.npz`` to rounding, pair by pair and aperture
by aperture, and the SHAM sample size must match.  That confirms the fields,
the galaxy catalogue and the kernels are the round-two ones (the standing
hazard of ``records/r_profiles_implementation_plan.md`` U7).  The result is
stored in the output and printed; the analysis refuses a failed run.

Nothing in ``src/`` is modified and no existing output is overwritten: files
go to ``round3a.out_path`` as ``ck_spectra_<label>_<snapshot>_<proj>.npz``.

Usage
-----
    cd scripts/
    python cross_corr/make_ck_spectra.py -p configs/cross_corr/round3a_z05.yaml
    python cross_corr/make_ck_spectra.py -p configs/cross_corr/round3a_z05.yaml \
        --sim TNG300-1
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

import rprofiles as rp
import round3a_lib as lib
from stacker import SimulationStacker
from tools import hist2d_numba_seq
from utils import comoving_to_arcmin

from make_calibration_factor import (add_convention_t, calibration_radii,
                                     load_fields_and_baryon_fraction)
from make_r_profiles import sim_label


#: Continuous-field pairs, named as in the calibration npz.
CONTINUOUS_PAIRS = (('m', 'm'), ('b', 'm'), ('b', 'b'), ('e', 'm'),
                    ('b', 'e'), ('e', 'e'))

#: Gas fields and matter conventions for the mediated amplitudes.
GAS_FIELDS = ('b', 'e')
CONVENTIONS = ('C', 'T')

#: Galaxy-sample order in the outputs.
SAMPLE_ORDER = ('fid', 'alt', 'lo', 'hi')

#: Relative tolerance of the regression check against round two.
REGRESSION_RTOL = 1e-9


def pair_name(a, b):
    """Name of a pair in the npz, e.g. ``('b', 'm') -> 'bm'``.

    Galaxy samples are written ``'g:<sample>'`` internally and appear as
    ``g`` in the name with the sample as a suffix (``'gm_fid'``), so the
    fiducial names match the calibration npz (``bg``, ``eg``, ``gm``, ``gg``).

    Args:
        a (str): First field.
        b (str): Second field.

    Returns:
        str: The pair name.
    """
    sample = None
    letters = []
    for f in (a, b):
        if f.startswith('g:'):
            sample = f[2:]
            letters.append('g')
        else:
            letters.append(f)
    name = ''.join(sorted(letters))
    return f'{name}_{sample}' if sample else name


def deposit_galaxies(subhalos, indices, projection, n_pixels, lbox):
    """NGP galaxy count map, exactly as ``rprofiles.make_galaxy_field``.

    Args:
        subhalos (dict): Subhalo catalogue with ``'SubhaloPos'``.
        indices (np.ndarray): Catalogue indices of the sample.
        projection (str): 'xy', 'xz' or 'yz'.
        n_pixels (int): Pixels per side.
        lbox (float): Box size, ckpc/h.

    Returns:
        np.ndarray: Count map, float64, ``(n_pixels, n_pixels)``.
    """
    pos2d = rp.project_positions(subhalos['SubhaloPos'][indices], projection)
    tracks = np.ascontiguousarray(
        np.array([pos2d[:, 0], pos2d[:, 1]], dtype=np.float64))
    bins = np.array([n_pixels, n_pixels], dtype=np.int64)
    ranges = np.array([[0.0, lbox], [0.0, lbox]], dtype=np.float64)
    weights = np.ones(tracks.shape[1], dtype=np.float64)
    return hist2d_numba_seq(tracks, bins, ranges, weights=weights)


def build_samples(stacker, subhalos, projection, n_pixels, cfg, r3cfg,
                  verbose=True):
    """Build the four galaxy samples as NGP count maps.

    Args:
        stacker (SimulationStacker): Configured stacker.
        subhalos (dict): Pre-loaded subhalo catalogue.
        projection (str): Projection.
        n_pixels (int): Pixels per side.
        cfg (dict): The ``stack`` config block (fiducial sample settings).
        r3cfg (dict): The ``round3a`` config block.
        verbose (bool, optional): Print progress.

    Returns:
        dict: Sample name to ``{'map', 'n_gal', 'nbar_pix', 'target',
        'description', 'parent_mass_range'}``.

    Raises:
        RuntimeError: If the re-deposited fiducial map differs from
            ``make_galaxy_field``'s, which would break the regression check.
    """
    lbox = float(stacker.header['BoxSize'])
    target = float(cfg.get('halo_abundance_target', 5e-4))
    upper = cfg.get('parent_mass_upper', 5e14)
    upper = None if upper is None else float(upper)
    npix2 = float(n_pixels * n_pixels)
    samples = {}

    fid_map, fid_idx = rp.make_galaxy_field(
        stacker, projection, n_pixels, target, parent_mass_upper=upper,
        subhalos=subhalos)
    if not np.array_equal(deposit_galaxies(subhalos, fid_idx, projection,
                                           n_pixels, lbox), fid_map):
        raise RuntimeError('Re-deposited fiducial galaxy map differs from '
                           'rprofiles.make_galaxy_field.')
    samples['fid'] = {'map': fid_map, 'target': target,
                      'description': f'SHAM {target:g} (cMpc/h)^-3, as round two'}

    alt = r3cfg.get('alt_abundance_target')
    if alt is not None:
        alt_map, _ = rp.make_galaxy_field(
            stacker, projection, n_pixels, float(alt),
            parent_mass_upper=upper, subhalos=subhalos)
        samples['alt'] = {'map': alt_map, 'target': float(alt),
                          'description': f'SHAM {float(alt):g} (cMpc/h)^-3'}

    if r3cfg.get('mass_split', True):
        parents = stacker.loadHalos()
        pmass = np.asarray(parents['GroupMass'])[
            np.asarray(subhalos['SubhaloGrNr'])[fid_idx]]
        lo, hi = lib.split_by_parent_mass(fid_idx, pmass)
        lookup = dict(zip(fid_idx.tolist(), pmass.tolist()))
        for name, idx in (('lo', lo), ('hi', hi)):
            m = np.array([lookup[i] for i in idx.tolist()])
            samples[name] = {
                'map': deposit_galaxies(subhalos, idx, projection, n_pixels,
                                        lbox),
                'target': target,
                'description': (f'{name} half of fid by parent FoF mass, '
                                f'{m.min():.3e}-{m.max():.3e} Msun/h'),
                'parent_mass_range': (float(m.min()), float(m.max()))}

    for name, s in samples.items():
        s['n_gal'] = int(round(s['map'].sum()))
        s['nbar_pix'] = s['map'].sum() / npix2
        if verbose:
            print(f"    sample {name:3s}: {s['n_gal']:7d} galaxies  "
                  f"({s['description']})", flush=True)
    return samples


def kernel_list(n_pixels, pixel_arcmin, radii, dr, sigma1, ratios):
    """Yield every kernel as ``(filter, index, spectrum, zero_lag)``.

    Base filters use the pixelized kernels of ``rprofiles`` (their centre
    value is the self-pair ``K(0)`` of ``compute_Y_matrix``); the DoG kernels
    are defined in Fourier space (``round3a_lib.dog_kernel_spectrum``).

    Args:
        n_pixels (int): Pixels per side.
        pixel_arcmin (float): Pixel size, arcmin.
        radii (np.ndarray): Aperture grid, arcmin.
        dr (float): Annulus width, arcmin.
        sigma1 (np.ndarray): DoG inner widths, arcmin.
        ratios (sequence): DoG ``sigma2/sigma1`` values.

    Yields:
        tuple: ``(filter_name, index, spectrum, zero_lag)``.
    """
    for filt in ('Sigma', 'DSigma'):
        for i, R in enumerate(radii):
            kern = rp.build_aperture_kernel(n_pixels, pixel_arcmin, float(R),
                                            filt, dr)
            k0 = float(kern[0, 0])
            spec = np.real(scipy.fft.rfft2(kern, workers=lib.FFT_WORKERS))
            del kern
            yield filt, i, spec, k0
    for q in ratios:
        for i, s1 in enumerate(sigma1):
            spec = lib.dog_kernel_spectrum(n_pixels, pixel_arcmin, float(s1),
                                           float(q) * float(s1))
            yield f'DoG_q={q:g}', i, spec, lib.zero_lag(spec, n_pixels)


def process_simulation(sim_type, sim_entry, cfg, r3cfg, verbose=True):
    """Measure spectra and exact amplitudes for one simulation.

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
    mpc_per_arcmin = (lbox / theta_arcmin) / 1000.0

    radii, _, _ = calibration_radii(cfg)
    dr = float(cfg.get('dr_arcmin', rp.DR_ARCMIN))
    sigma1 = np.geomspace(float(r3cfg['dog_sigma1_min']),
                          float(r3cfg['dog_sigma1_max']),
                          int(r3cfg['dog_n_sigma']))
    ratios = [float(q) for q in r3cfg['dog_ratios']]
    width = float(r3cfg.get('k_bin_width', 0.02))
    cal_dir = Path(r3cfg.get('calibration_npz_path', '../data/cross_corr_C/'))
    out_dir = Path(r3cfg.get('out_path', '../data/cross_corr_C/round3a/'))

    if verbose:
        print(f'\n{"=" * 70}\n{label}  ({sim_type}, snapshot {snapshot})\n'
              f'{"=" * 70}')
        print(f'  z = {z_true:.4f}, pixel = {pixel_arcmin:.5f} arcmin, '
              f'grid {n_pixels}^2, k-bin width {width} /arcmin')

    # Resolution guard, as in make_calibration_factor: every aperture must be
    # resolved before the fields and catalogues are loaded.
    for guard in radii:
        rp.build_aperture_kernel(n_pixels, pixel_arcmin, float(guard),
                                 'DSigma', dr)

    subhalos = stacker.loadSubHalos()
    written = []
    for projection in cfg.get('projections', ['yz']):
        t0 = time.time()
        deltas, f_b = load_fields_and_baryon_fraction(
            stacker, n_pixels, projection, cfg, verbose=verbose)
        samples = build_samples(stacker, subhalos, projection, n_pixels, cfg,
                                r3cfg, verbose=verbose)
        binning = lib.ModeBinning(n_pixels, pixel_arcmin, width)

        F = {k: scipy.fft.rfft2(deltas.pop(k), workers=lib.FFT_WORKERS)
             for k in ('m', 'b', 'e')}
        for s in SAMPLE_ORDER:
            if s in samples:
                F[f'g:{s}'] = scipy.fft.rfft2(
                    rp.to_overdensity(samples[s].pop('map')),
                    workers=lib.FFT_WORKERS)
        del deltas
        sample_names = [s for s in SAMPLE_ORDER if s in samples]
        if verbose:
            print(f'    FFTs done ({time.time() - t0:.0f} s); '
                  f'{binning.n_bins} k bins', flush=True)

        # Pairs: continuous, then (g_s, m), (g_s, b), (g_s, e), (g_s, g_s).
        pairs = list(CONTINUOUS_PAIRS)
        for s in sample_names:
            g = f'g:{s}'
            pairs += [(g, 'm'), (g, 'b'), (g, 'e'), (g, g)]

        # Binned spectra and the weighted per-mode products, stacked into
        # one matrix so each kernel costs one matrix-vector product.
        payload = {}
        n_modes = int(np.prod(binning.shape))
        rows = [pair_name(a, b) for a, b in pairs]
        med_rows = [f'{X}_{conv}_{s}' for s in sample_names
                    for X in GAS_FIELDS for conv in CONVENTIONS]
        # Memory: 38 rows x n^2/2 modes x 8 bytes, about 30 GB at n = 14015
        # (FLAMINGO z ~ 0.30), 12 GB at n = 8869; peak RSS stays well inside a
        # 512 GB CPU node.
        stack = np.empty((len(rows) + len(med_rows), n_modes))
        spectra = {}
        norm = binning.area / float(n_pixels) ** 4
        for i, (a, b) in enumerate(pairs):
            re = np.real(F[a] * np.conj(F[b]))
            with np.errstate(invalid='ignore', divide='ignore'):
                p = binning.bin_sum(re) * norm / binning.counts
            p[binning.counts == 0] = np.nan
            spectra[rows[i]] = p
            payload[f'P2D_{rows[i]}'] = p
            stack[i] = binning.weighted(re).ravel()
            del re
        for s in sample_names:
            payload[f'shot_{s}'] = np.array(binning.area / samples[s]['n_gal'])

        # eta for each gas field and convention, from the continuous spectra.
        cont = {rp._pair_key(a, b): spectra[pair_name(a, b)]
                for a, b in CONTINUOUS_PAIRS}
        add_convention_t(cont, f_b)
        eta = {}
        for X in GAS_FIELDS:
            for conv, mkey in (('C', 'm'), ('T', 't')):
                num = cont[rp._pair_key(X, mkey)]
                den = cont[rp._pair_key(mkey, mkey)]
                with np.errstate(invalid='ignore', divide='ignore'):
                    e = np.where(den > 0, num / den, 0.0)
                eta[(X, conv)] = np.where(np.isfinite(e), e, 0.0)
                payload[f'eta_{X}_{conv}'] = eta[(X, conv)]

        # Mediated rows: eta(|k|) Re[F_g F_mconv*], with F_t = f_m F_m + f_b F_b.
        row_of = {r: i for i, r in enumerate(rows)}
        j = len(rows)
        for s in sample_names:
            gm = stack[row_of[f'gm_{s}']]
            gb = stack[row_of[f'bg_{s}']]
            for X in GAS_FIELDS:
                for conv in CONVENTIONS:
                    g_mat = gm if conv == 'C' else (1.0 - f_b) * gm + f_b * gb
                    stack[j] = binning.per_mode(eta[(X, conv)]).ravel() * g_mat
                    j += 1
        del F
        if verbose:
            print(f'    spectra and mode products ready '
                  f'({time.time() - t0:.0f} s, stack '
                  f'{stack.nbytes / 1e9:.1f} GB)', flush=True)

        # Exact amplitudes, kernel by kernel.  The doubly filtered rows are
        # copied once so each kernel's K^2 product reads a contiguous block.
        r2_rows = [row_of[r] for r in ('bm', 'mm', 'bb', 'gm_fid', 'gg_fid',
                                       'bg_fid')]
        stack2 = stack[r2_rows]
        n4 = float(n_pixels) ** 4
        out_Y, out_Y2, out_w, out_k0 = {}, {}, {}, {}
        filter_sizes = {'Sigma': len(radii), 'DSigma': len(radii)}
        filter_sizes.update({f'DoG_q={q:g}': len(sigma1) for q in ratios})
        for filt, size in filter_sizes.items():
            out_Y[filt] = np.empty((stack.shape[0], size))
            out_Y2[filt] = np.empty((len(r2_rows), size))
            out_w[filt] = np.empty((size, binning.n_bins))
            out_k0[filt] = np.empty(size)
        k2zero = {f: np.empty(n) for f, n in filter_sizes.items()}

        for filt, i, spec, k0 in kernel_list(n_pixels, pixel_arcmin, radii,
                                             dr, sigma1, ratios):
            flat = spec.ravel()
            out_Y[filt][:, i] = (stack @ flat) / n4
            out_Y2[filt][:, i] = (stack2 @ (flat * flat)) / n4
            out_w[filt][i] = binning.bin_sum(spec)
            out_k0[filt][i] = k0
            k2zero[filt][i] = lib.zero_lag(spec * spec, n_pixels)
        del stack, stack2
        if verbose:
            print(f'    exact amplitudes done ({time.time() - t0:.0f} s)',
                  flush=True)

        # Self-pairs: subtract K(0)/nbar from every galaxy auto (and
        # sum K^2 / (n^2 nbar) from the doubly filtered one).
        for filt in filter_sizes:
            for s in sample_names:
                r = row_of[f'gg_{s}']
                out_Y[filt][r] -= out_k0[filt] / samples[s]['nbar_pix']
            r2 = r2_rows.index(row_of['gg_fid'])
            out_Y2[filt][r2] -= k2zero[filt] / samples['fid']['nbar_pix']

            for r, rname in enumerate(rows):
                payload[f'Yx_{rname}_{filt}'] = out_Y[filt][r]
            for r, rname in enumerate(med_rows):
                payload[f'Ymed_{rname}_{filt}'] = out_Y[filt][len(rows) + r]
            for r, idx in enumerate(r2_rows):
                payload[f'Y2_{rows[idx]}_{filt}'] = out_Y2[filt][r]
            payload[f'w_{filt}'] = out_w[filt]
            payload[f'k0_{filt}'] = out_k0[filt]

        # Regression against round two (fiducial sample, base filters).
        reg = regression_check(cal_dir, label, snapshot, projection, radii,
                               payload, samples['fid'], verbose=verbose)
        payload.update(reg)

        payload.update({
            'radii': radii, 'dog_sigma1': sigma1,
            'dog_ratios': np.array(ratios), 'k_edges': binning.edges,
            'k_mean': binning.k_mean, 'counts': binning.counts})
        meta = {'label': label, 'sim_type': sim_type, 'sim_name': str(name),
                'feedback': str(feedback), 'snapshot': snapshot,
                'projection': projection, 'redshift': z_true,
                'n_pixels': n_pixels, 'pixel_arcmin': pixel_arcmin,
                'area_arcmin2': binning.area, 'boxsize_ckpc_h': lbox,
                'mpc_per_arcmin': mpc_per_arcmin, 'f_b': f_b, 'dr_arcmin': dr,
                'k_bin_width': width,
                'samples': np.array(sample_names, dtype=object),
                'filters': np.array(list(filter_sizes), dtype=object),
                'gas_fields': np.array(GAS_FIELDS, dtype=object)}
        for s in sample_names:
            meta[f'n_gal_{s}'] = samples[s]['n_gal']
            meta[f'nbar_pix_{s}'] = samples[s]['nbar_pix']
            meta[f'target_{s}'] = samples[s]['target']
            meta[f'description_{s}'] = samples[s]['description']
        for key, value in meta.items():
            payload[f'meta_{key}'] = np.array(value)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'ck_spectra_{label}_{snapshot}_{projection}.npz'
        tmp_path = out_path.with_name(out_path.stem + '.tmp.npz')
        np.savez(tmp_path, **payload)
        tmp_path.replace(out_path)
        written.append(out_path)
        if verbose:
            print(f'    saved {out_path} ({time.time() - t0:.0f} s)')
    return written


def regression_check(cal_dir, label, snapshot, projection, radii, payload,
                     fid, verbose=True):
    """Compare the fiducial exact amplitudes with the committed round two.

    Args:
        cal_dir (pathlib.Path): Directory of ``calibration_*.npz``.
        label (str): Run label.
        snapshot (int): Snapshot.
        projection (str): Projection.
        radii (np.ndarray): Aperture grid of this run.
        payload (dict): Output payload holding ``Yx_*`` arrays.
        fid (dict): The fiducial sample record.
        verbose (bool, optional): Print the result.

    Returns:
        dict: ``reg_*`` entries: per-pair worst relative differences, the
        sample checks, and ``reg_pass``.
    """
    path = cal_dir / f'calibration_{label}_{snapshot}_{projection}.npz'
    out = {'reg_reference': np.array(str(path))}
    if not path.exists():
        out['reg_pass'] = np.array(False)
        if verbose:
            print(f'    REGRESSION: reference {path} missing')
        return out
    ref = np.load(path, allow_pickle=True)
    ok = bool(np.allclose(ref['radii'], radii, rtol=0, atol=1e-9))
    ok &= int(ref['meta_n_galaxies']) == fid['n_gal']
    ok &= bool(np.isclose(float(ref['meta_nbar_pix']), fid['nbar_pix'],
                          rtol=1e-14, atol=0))
    out['reg_sample_match'] = np.array(ok)
    worst_all = 0.0
    for filt in ('Sigma', 'DSigma'):
        for pair in ('mm', 'bm', 'bb', 'em', 'be', 'ee'):
            mine = payload[f'Yx_{pair}_{filt}']
            theirs = ref[f'Y_{pair}_{filt}']
            worst = float(np.max(np.abs(mine / theirs - 1.0)))
            out[f'reg_maxrel_{pair}_{filt}'] = np.array(worst)
            worst_all = max(worst_all, worst)
        for pair in ('gm', 'bg', 'eg', 'gg'):
            mine = payload[f'Yx_{pair}_fid_{filt}']
            theirs = ref[f'Y_{pair}_{filt}']
            worst = float(np.max(np.abs(mine / theirs - 1.0)))
            out[f'reg_maxrel_{pair}_{filt}'] = np.array(worst)
            worst_all = max(worst_all, worst)
    out['reg_maxrel_all'] = np.array(worst_all)
    passed = ok and worst_all < REGRESSION_RTOL
    out['reg_pass'] = np.array(passed)
    if verbose:
        print(f'    REGRESSION vs round two: sample match {ok}, worst '
              f'relative difference {worst_all:.2e} -> '
              f'{"PASS" if passed else "FAIL"}', flush=True)
    return out


def main(path2config, sim=None, feedback=None, verbose=True):
    """Run Stage 2 for every simulation in a config.

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
        description='Round 3A Stage 2: C(k) and the exact window/mediation '
                    'split from cached maps.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/round3a_z05.yaml')
    parser.add_argument('--sim', type=str, default=None)
    parser.add_argument('--feedback', type=str, default=None)
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
