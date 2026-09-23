"""compute_pk_local.py
====================
Local version of the alpha model for the matter power spectrum section of the
unbound gas paper: the non-ionized baryons (neutral gas, stars, BH) are laid
out like the *local* ionized gas (or DM) within a radius R, instead of like
the box-wide ionized gas as in ``compute_pk_components.py`` /
``make_pk_alpha.py`` (which are unchanged by this script).

Operator (config block ``local``; everything else as in ``pk_common``). For a
moved component e (its field rho_e) and a target template T (ionized gas or
DM),

    m_e(x) = rho_T(x) * [W_R * (rho_e / (W_R * rho_T))](x) ,

i.e. every cell spreads its mass of e over the kernel around it, in
proportion to T there (a transport: mass is conserved exactly and, for the
sphere, moves no further than R). W_R is a normalised kernel: a sharp sphere
of radius R (``tophat``, discretised in real space) or a Gaussian with
per-axis width sigma = gaussian_sigma_factor * R (``gaussian``). Large-scale
power is therefore unchanged up to O((kR)^2); m_e -> rho_e as R -> 0, and on
small scales m_e approaches the global model as R grows. The operator is
linear in rho_e, so moving several components together equals the sum of
moving them one by one. Cells with no target mass within reach
([W_R * rho_T] <= 1e-3 of its mean; essentially never for R >= 2 cells) keep
their mass in place; the final rescaling to the exact mass of e only removes
FFT round-off (both are recorded).

Moved sets ('all' = neutral + stars + BH, 'stars', 'neutral'), targets
('ionized', 'dm'), kernels and radii come from the config; radii below
``min_radius_cells`` grid cells are skipped for that simulation.

Power spectra use a numba estimator that keeps every FFT in memory (Pylians
cannot fit FLAMINGO's extra fields at 2000^3). It reproduces Pylians exactly:
same k bins (Pylians drops the k=0 bin), same independent-mode counting, TSC
deconvolution, and power normalisation (L/n^2)^3; checked to 2e-7 on random
fields, and each run compares its original-component spectra with the
``*_Pk_components_*`` file from compute_pk_components.py.

Outputs (new files next to the 3D caches; existing ones skipped unless
--overwrite):
  <stem>_Pk_local_orig_<n>.npz
      k, Nmodes, components, means, P (5 x 5 x Nk) from this estimator, and
      max_rel_diff_vs_pylians.
  <stem>_Pk_local_<kernel>_R<R>_<n>.npz   (one per kernel and radius)
      k, R, kernel, sigma (Gaussian width or 0), and for every target t and
      moved set s: P_auto__<t>__<s> (Nk), P_cross__<t>__<s> (5 x Nk, with the
      original components), renorm__<t>__<s>, kept_frac__<t>__<s>.

Run from the scripts/ directory on a whole CPU node:
    python unbound_gas/compute_pk_local.py -p configs/unbound_gas/pk_components_z05.yaml \
        --sims 'm100n1024'
    # identity check (W = delta function => m_e == rho_e), no files written:
    python unbound_gas/compute_pk_local.py -p ... --sims m100n1024 --identity-check
"""

import argparse
import gc
import time

import numba
import numpy as np
import scipy.fft as sfft

from pk_common import (COMPONENTS, box_size_mpc, load_config, load_field,
                       save_npz_atomic, select_sims, sim_label, spectra_path)

MOVED_SETS = {'all': ['neutral_gas', 'Stars', 'BH'], 'stars': ['Stars'],
              'neutral': ['neutral_gas']}
TARGETS = {'ionized': 'ionized_gas', 'dm': 'DM'}
FLOOR = 1e-3  # relative to the mean of the smoothed target: below it, mass stays in place.
# (At 1e-6 the ratio rho_e / S_T reached ~1e6 in near-empty DM cells and float32
# FFT round-off on that field leaked ~0.5% into P; see the identity check.)


# ---------------------------------------------------------------------------
# Power spectrum estimator (Pylians-compatible)
# ---------------------------------------------------------------------------

def tsc_window(n: int) -> np.ndarray:
    """Per-axis TSC window sinc(pi s/n)^3 for signed FFT index s (Pylians convention)."""
    i = np.arange(n)
    s = np.where(i > n // 2, i - n, i).astype(np.float64)
    x = np.pi * s / n
    w = np.ones(n)
    nz = x != 0
    w[nz] = (np.sin(x[nz]) / x[nz]) ** 3
    return w


@numba.njit(parallel=True, cache=True)
def _power_sums(A, B, n, nbins, wmas):
    """Mode-summed Re(A_i B_j^*)/W^2 in integer-|k| bins, Pylians mode counting.

    A: (na, n, n, n//2+1) complex64; B: (nb, n, n, n//2+1) complex64.
    Returns sums (nbins, na, nb), k sums (nbins), mode counts (nbins).
    """
    nh = n // 2 + 1
    middle = n // 2
    na, nb = A.shape[0], B.shape[0]
    acc = np.zeros((n, nbins, na, nb))
    kacc = np.zeros((n, nbins))
    nacc = np.zeros((n, nbins))
    for ix in numba.prange(n):
        kx = ix - n if ix > middle else ix
        for iy in range(n):
            ky = iy - n if iy > middle else iy
            for iz in range(nh):
                if iz == 0 or (iz == middle and n % 2 == 0):
                    if kx < 0:
                        continue
                    elif kx == 0 or (kx == middle and n % 2 == 0):
                        if ky < 0:
                            continue
                kk = np.sqrt(kx * kx + ky * ky + iz * iz)
                b = int(kk)
                corr = 1.0 / (wmas[ix] * wmas[iy] * wmas[iz]) ** 2
                kacc[ix, b] += kk
                nacc[ix, b] += 1.0
                for i in range(na):
                    a = A[i, ix, iy, iz]
                    for j in range(nb):
                        c = B[j, ix, iy, iz]
                        acc[ix, b, i, j] += (a.real * c.real + a.imag * c.imag) * corr
    return acc.sum(0), kacc.sum(0), nacc.sum(0)


def cross_power(A: np.ndarray, B: np.ndarray, box: float, scale_a, scale_b):
    """Auto/cross power spectra of raw FFTs A (na) x B (nb) of density fields.

    FFT(delta) = FFT(rho) / mean(rho) except at k=0, which is dropped, so the
    overdensity spectra follow from the raw FFTs scaled by 1/mean.

    Args:
        A, B: stacks of raw half-complex FFTs, shape (na|nb, n, n, n//2+1).
        box: box size [Mpc/h].
        scale_a, scale_b: 1/mean of each field (arrays of length na, nb).

    Returns:
        k [h/Mpc] (Nk), Nmodes (Nk), P (na, nb, Nk) [(Mpc/h)^3], for
        1 <= |k|/k_F <= n/2 (Nyquist), matching compute_pk_components.py.
    """
    n = A.shape[1]
    nbins = int(np.sqrt(3) * (n // 2)) + 2
    acc, ka, na_ = _power_sums(A, B, n, nbins, tsc_window(n))
    idx = np.arange(nbins)
    sel = (na_ > 0) & (idx >= 1)
    k = ka[sel] / na_[sel] * 2 * np.pi / box
    keep = k <= np.pi * n / box
    P = acc[sel] / na_[sel][:, None, None] * (box / n ** 2) ** 3
    P = P * np.outer(scale_a, scale_b)[None]
    return k[keep], na_[sel][keep], np.moveaxis(P[keep], 0, -1)


# ---------------------------------------------------------------------------
# Local redistribution operator
# ---------------------------------------------------------------------------

def kernel_fft(kind: str, R: float, n: int, box: float, sigma_factor: float) -> np.ndarray:
    """Half-complex FFT (real, float32) of a unit-sum smoothing kernel.

    Args:
        kind: 'tophat' (sphere of radius R, discretised on the grid: cells whose
            centre-to-centre distance is <= R; always at least the central
            cell) or 'gaussian' (analytic exp(-k^2 sigma^2 / 2), sigma =
            sigma_factor * R), or 'delta' (identity; for the identity check).
        R: radius [Mpc/h].
        n: grid size; box: box size [Mpc/h].
    """
    s = np.fft.fftfreq(n, d=1.0 / n)
    sz = np.fft.rfftfreq(n, d=1.0 / n)
    if kind == 'delta':
        return np.ones((n, n, n // 2 + 1), dtype=np.float32)
    if kind == 'gaussian':
        sigma = sigma_factor * R
        kf = 2 * np.pi / box
        g = [np.exp(-0.5 * (kf * sigma * v) ** 2).astype(np.float32) for v in (s, s, sz)]
        return g[0][:, None, None] * g[1][None, :, None] * g[2][None, None, :]
    if kind == 'tophat':
        cell = box / n
        r = int(np.ceil(R / cell))
        d = np.arange(-r, r + 1) * cell
        dist2 = d[:, None, None] ** 2 + d[None, :, None] ** 2 + d[None, None, :] ** 2
        ball = dist2 <= R ** 2
        w = np.zeros((n, n, n), dtype=np.float32)
        ii = np.nonzero(ball)
        w[tuple((np.asarray(v) - r) % n for v in ii)] = 1.0
        w /= w.sum()
        Wk = sfft.rfftn(w, workers=-1).real.astype(np.float32)
        del w
        return Wk
    raise ValueError(kind)


def smooth(F: np.ndarray, Wk: np.ndarray, n: int, inplace: bool = False) -> np.ndarray:
    """Real-space field of the raw FFT F convolved with the kernel Wk.

    With inplace=True, F is overwritten (saves one full-size temporary).
    """
    if inplace:
        F *= Wk
        tmp = F
    else:
        tmp = F * Wk
    out = sfft.irfftn(tmp, s=(n, n, n), workers=-1)
    del tmp
    return out.astype(np.float32, copy=False)


def moved_field(rho_e: np.ndarray, rho_T: np.ndarray, S_T: np.ndarray, Wk: np.ndarray,
                n: int, mass_e: float):
    """Transport the mass of e within R so that it follows the target T.

    Every cell y spreads its mass rho_e(y) over the kernel around y, in
    proportion to rho_T there:

        m_e(x) = rho_T(x) * [W_R * (rho_e / S_T)](x),   S_T = W_R * rho_T ,

    which conserves mass exactly and (for the sphere) moves no mass further
    than R. Cells with no target mass within reach (S_T <= FLOOR x mean) keep
    their mass in place.

    Args:
        rho_e: field to move (real; overwritten and consumed); rho_T: target
            field (real); S_T: W_R * rho_T; Wk: kernel FFT; n: grid size;
            mass_e: total mass of e.

    Returns:
        The moved field (float32) and diagnostics (renorm, kept_frac): the
        final rescaling to mass_e (1 up to FFT round-off) and the fraction of
        e's mass left in place.
    """
    q = rho_e
    floor = FLOOR * float(np.mean(S_T, dtype=np.float64))
    mask = S_T <= floor
    kept_idx = np.flatnonzero(mask)
    kept_val = q.ravel()[kept_idx].copy()
    np.divide(q, S_T, out=q, where=~mask)
    np.copyto(q, np.float32(0.0), where=mask)
    del mask
    Fq = sfft.rfftn(q, workers=-1)
    del q
    m = smooth(Fq, Wk, n, inplace=True)    # W_R * (rho_e / S_T)
    del Fq
    m *= rho_T
    m.ravel()[kept_idx] += kept_val
    total = float(np.sum(m, dtype=np.float64))
    renorm = mass_e / total
    m *= np.float32(renorm)
    return m, renorm, float(np.sum(kept_val, dtype=np.float64)) / mass_e


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def component_ffts(entry: dict) -> tuple:
    """Raw FFTs of all five components (float32 fields) and their means.

    Real-space fields are not kept: at FLAMINGO's 2000^3 each costs 32 GB, and
    a target field is recovered from its FFT when needed.
    """
    n = entry['n_pixels']
    nh = n // 2 + 1
    F = np.empty((len(COMPONENTS), n, n, nh), dtype=np.complex64)
    means = np.zeros(len(COMPONENTS))

    gas = load_field(entry, 'gas')
    ion = load_field(entry, 'ionized_gas')
    means[1] = np.mean(ion, dtype=np.float64)
    F[1] = sfft.rfftn(ion, workers=-1)
    neu = gas - ion
    means[2] = np.mean(neu, dtype=np.float64)
    F[2] = sfft.rfftn(neu, workers=-1)
    del neu
    dm = load_field(entry, 'total')
    dm -= gas
    del gas
    for i, p_type in ((3, 'Stars'), (4, 'BH')):
        f = load_field(entry, p_type)
        dm -= f
        means[i] = np.mean(f, dtype=np.float64)
        F[i] = sfft.rfftn(f, workers=-1)
        del f
    means[0] = np.mean(dm, dtype=np.float64)
    F[0] = sfft.rfftn(dm, workers=-1)
    del dm, ion
    return F, means


def run_sim(entry: dict, config: dict, overwrite: bool, identity: bool) -> None:
    """Compute original and moved-field spectra for one simulation."""
    lc = config['local']
    n = entry['n_pixels']
    box = box_size_mpc(entry)
    cell = box / n
    idx = {c: i for i, c in enumerate(COMPONENTS)}

    # Cheap resume: skip the (expensive) field loading and FFTs when every
    # output of this simulation already exists.
    if not identity and not overwrite:
        wanted = [spectra_path(entry, f"local_{kern}_R{float(R):g}")
                  for kern in lc['kernels'] for R in lc['radii']
                  if float(R) >= lc['min_radius_cells'] * cell]
        if spectra_path(entry, 'local_orig').exists() and all(p.exists() for p in wanted):
            print("  all outputs exist, skipping")
            return

    pyl_path = spectra_path(entry, 'components')
    if not pyl_path.exists():
        raise FileNotFoundError(f"missing {pyl_path} (run compute_pk_components.py first; "
                                "it is the Pylians reference for this estimator)")

    t0 = time.time()
    F, means = component_ffts(entry)
    inv = 1.0 / means
    print(f"  fields and FFTs in {time.time() - t0:.0f} s; means "
          + ", ".join(f"{c}={m:.4e}" for c, m in zip(COMPONENTS, means)))

    # Original spectra with this estimator, checked against Pylians.
    orig_path = spectra_path(entry, 'local_orig')
    t0 = time.time()
    k, nmodes, P = cross_power(F, F, box, inv, inv)
    pyl = np.load(pyl_path)
    if len(pyl['k']) != len(k) or not np.allclose(pyl['k'], k, rtol=1e-10):
        raise RuntimeError("k bins differ from the Pylians components file")
    dev = float(np.max(np.abs(P / pyl['P'] - 1.0)))
    print(f"  original spectra in {time.time() - t0:.0f} s; max rel diff vs Pylians = {dev:.2e}")
    if dev > 1e-4:
        raise RuntimeError(f"estimator disagrees with Pylians (max rel diff {dev:.2e})")
    if not identity and (overwrite or not orig_path.exists()):
        save_npz_atomic(orig_path, k=k, Nmodes=nmodes, components=np.array(COMPONENTS),
                        means=means, P=P, max_rel_diff_vs_pylians=dev,
                        box_mpc=box, n_pixels=n)

    if identity:
        runs = [('delta', 0.0)]
    else:
        runs = [(kern, float(R)) for kern in lc['kernels'] for R in lc['radii']]
    for kern, R in runs:
        if kern != 'delta' and R < lc['min_radius_cells'] * cell:
            print(f"  skip {kern} R={R} Mpc/h: below {lc['min_radius_cells']} cells ({cell:.3f} Mpc/h)")
            continue
        out = spectra_path(entry, f"local_{kern}_R{R:g}")
        if not identity and out.exists() and not overwrite:
            print(f"  exists, skipping: {out}")
            continue
        t1 = time.time()
        sigma = lc['gaussian_sigma_factor'] * R if kern == 'gaussian' else 0.0
        Wk = kernel_fft(kern, R, n, box, lc['gaussian_sigma_factor'])
        res = dict(k=k, R=R, kernel=kern, sigma=sigma, box_mpc=box, n_pixels=n)
        for tname in lc['targets']:
            iT = idx[TARGETS[tname]]
            rho_T = sfft.irfftn(F[iT], s=(n, n, n), workers=-1).astype(np.float32, copy=False)
            S_T = smooth(F[iT], Wk, n)
            for sname in lc['moved']:
                members = [idx[c] for c in MOVED_SETS[sname]]
                if len(members) > 1:  # sum in place: F[members] would copy them all first
                    F_e = F[members[0]].copy()
                    for i in members[1:]:
                        F_e += F[i]
                else:
                    F_e = F[members[0]]  # view, not modified
                mass_e = float(sum(means[i] for i in members)) * n ** 3
                rho_e = sfft.irfftn(F_e, s=(n, n, n), workers=-1).astype(np.float32, copy=False)
                del F_e
                m, renorm, ffrac = moved_field(rho_e, rho_T, S_T, Wk, n, mass_e)
                del rho_e
                Fm = sfft.rfftn(m, workers=-1)[None]
                del m
                mean_m = mass_e / n ** 3
                _, _, Pm = cross_power(Fm, Fm, box, [1 / mean_m], [1 / mean_m])
                _, _, Px = cross_power(Fm, F, box, [1 / mean_m], inv)
                del Fm
                gc.collect()
                key = f"{tname}__{sname}"
                res[f"P_auto__{key}"] = Pm[0, 0]
                res[f"P_cross__{key}"] = Px[0]
                res[f"renorm__{key}"] = renorm
                res[f"kept_frac__{key}"] = ffrac
                if identity:
                    ref = np.einsum('i,j,ijk->k', means[members], means[members], P[np.ix_(members, members)]) \
                        / sum(means[members]) ** 2
                    d = float(np.max(np.abs(Pm[0, 0] / ref - 1.0)))
                    print(f"    identity {key}: max |P(m,m)/P(e,e) - 1| = {d:.2e}, renorm {renorm:.8f}")
            del S_T, rho_T
            gc.collect()
        print(f"  {kern} R={R:g} Mpc/h (sigma={sigma:.3f}) in {time.time() - t1:.0f} s; renorm "
              + ", ".join(f"{k_[8:]}={v:.5f}" for k_, v in res.items() if k_.startswith('renorm__')))
        if not identity:
            save_npz_atomic(out, **res)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None,
                        help="restrict to these entries (label, name or feedback)")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--identity-check', action='store_true',
                        help="run only a delta-function kernel and check m_e == rho_e; writes nothing")
    args = parser.parse_args()

    config = load_config(args.path2config)
    numba.set_num_threads(int(config['pk'].get('threads', numba.get_num_threads())))
    for entry in select_sims(config, args.sims):
        print(f"\n===== {sim_label(entry)} (grid {entry['n_pixels']}^3) =====")
        run_sim(entry, config, args.overwrite, args.identity_check)
        gc.collect()


if __name__ == '__main__':
    main()
