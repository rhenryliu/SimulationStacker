"""round3a_lib.py
===============
Pure functions shared by the Round 3A scripts: is the departure of the
filtered calibration factor from unity window smearing or mediation failure?
(``docs/cross_corr/open-items.md`` O-01 to O-10, formalism §8.4 and §4.5.)

Nothing here reads or writes files, so every function is testable on
synthetic inputs (``tests/test_round3a.py``).  The drivers are
``make_ck_spectra.py`` and ``make_dog_calibration.py`` (compute node) and the
``round3a_*.py`` analysis scripts (login node).

Conventions shared with ``rprofiles.py`` and the round-two sweep:

- every field is an overdensity ``delta = X/<X> - 1`` on a periodic square
  grid of ``n`` pixels of ``pixel`` arcmin;
- a kernel spectrum ``K(k)`` is the ``rfft2`` of the sampled real-space
  kernel ``W(theta)`` (units arcmin^-2), so a filtered amplitude is

      Y_ab = (1/n^2) sum_x (K * delta_a)(x) delta_b(x)
           = (1/n^4) sum_{k, full plane} K(k) Re[F_a(k) F_b*(k)],

  which is larger than the continuum integral by ``1/pixel^2`` (the
  "1/pixArea wart" of ``filter_specification.md`` §5, which cancels in every
  ratio used here);
- a binned spectrum is ``P(k) = <Re F_a F_b*> area / n^4`` in arcmin^2, the
  normalization of ``make_task9_spectra.cross_p2d``, so that the binned
  filtered amplitude is ``(1/area) sum_bins w(bin) P(bin)`` with
  ``w(bin) = sum_{modes in bin} K``.
"""

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import scipy.fft

#: Worker count for ``scipy.fft``, as in ``rprofiles``.
FFT_WORKERS = -1


# ---------------------------------------------------------------------------
# Output hygiene
# ---------------------------------------------------------------------------

def save_npz_atomic(path, **arrays) -> Path:
    """Write an ``.npz`` through a temporary file, then rename it into place.

    A killed or failed run leaves the previous file untouched rather than a
    truncated one.

    Args:
        path (str or pathlib.Path): Destination, ending in ``.npz``.
        **arrays: Arrays to save, as for ``np.savez``.

    Returns:
        pathlib.Path: The destination.
    """
    path = Path(path)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return path


def write_text_atomic(path, text: str) -> Path:
    """Write a text file through a temporary file, then rename it into place.

    Args:
        path (str or pathlib.Path): Destination.
        text (str): Contents.

    Returns:
        pathlib.Path: The destination.
    """
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)
    return path


def require_regression_pass(npz, path) -> None:
    """Refuse a ``ck_spectra`` file whose regression check failed.

    Args:
        npz (np.lib.npyio.NpzFile): The loaded file.
        path (str or pathlib.Path): Its path, for the message.

    Raises:
        SystemExit: If ``reg_pass`` is missing or false.
    """
    if 'reg_pass' not in npz.files or not bool(npz['reg_pass']):
        worst = (float(npz['reg_maxrel_all']) if 'reg_maxrel_all' in npz.files
                 else float('nan'))
        raise SystemExit(f'{path} failed its regression check against round '
                         f'two (worst {worst:.2e}); refusing to interpret it.')


# ---------------------------------------------------------------------------
# The rfft2 mode grid
# ---------------------------------------------------------------------------

def rfft_wavenumbers(n_pixels: int, pixel_arcmin: float) -> np.ndarray:
    """Return ``|k|`` on the ``rfft2`` grid of an ``n x n`` map.

    Args:
        n_pixels (int): Pixels per side.
        pixel_arcmin (float): Pixel size, arcmin.

    Returns:
        np.ndarray: ``|k|`` in 1/arcmin, shape ``(n, n//2 + 1)``.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(n_pixels, d=pixel_arcmin)
    ky = 2.0 * np.pi * np.fft.rfftfreq(n_pixels, d=pixel_arcmin)
    return np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)


def rfft_multiplicity(n_pixels: int) -> np.ndarray:
    """Return how many full-plane modes each ``rfft2`` column stands for.

    The ``ky = 0`` column (and the ``ky = n/2`` column when ``n`` is even)
    holds its own conjugate partners; every other column represents both
    ``k`` and ``-k``.

    Args:
        n_pixels (int): Pixels per side.

    Returns:
        np.ndarray: Shape ``(n//2 + 1,)``, entries 1 or 2, to broadcast along
        the last axis of an ``rfft2`` array.
    """
    mult = np.full(n_pixels // 2 + 1, 2.0)
    mult[0] = 1.0
    if n_pixels % 2 == 0:
        mult[-1] = 1.0
    return mult


class ModeBinning:
    """Linear ``|k|`` bins over the full Fourier plane of one map grid.

    Precomputes, once per grid, the bin index of every ``rfft2`` mode and the
    full-plane multiplicity, so that binned spectra and binned kernel weights
    are sums over exactly the same modes.

    Attributes:
        n_pixels (int): Pixels per side.
        pixel_arcmin (float): Pixel size, arcmin.
        area (float): Map area, arcmin^2.
        edges (np.ndarray): Bin edges, 1/arcmin.
        counts (np.ndarray): Full-plane modes per bin.
        k_mean (np.ndarray): Mode-weighted mean ``|k|`` per bin, NaN if empty.
    """

    def __init__(self, n_pixels: int, pixel_arcmin: float, width: float):
        """Build the binning.

        Args:
            n_pixels (int): Pixels per side.
            pixel_arcmin (float): Pixel size, arcmin.
            width (float): Bin width in 1/arcmin.

        Raises:
            ValueError: If ``width`` is not positive.
        """
        if not width > 0:
            raise ValueError(f'width must be positive, got {width!r}.')
        self.n_pixels = int(n_pixels)
        self.pixel_arcmin = float(pixel_arcmin)
        self.area = (self.n_pixels * self.pixel_arcmin) ** 2
        kk = rfft_wavenumbers(self.n_pixels, self.pixel_arcmin)
        n_bins = int(np.floor(kk.max() / width)) + 1
        self.edges = np.arange(n_bins + 1, dtype=np.float64) * width
        self.n_bins = n_bins
        self.shape = kk.shape
        self._mult = rfft_multiplicity(self.n_pixels)
        idx = np.floor(kk / width).astype(np.int64)
        self._idx = np.clip(idx, 0, n_bins - 1).ravel()
        self.counts = self.bin_sum(np.ones_like(kk))
        with np.errstate(invalid='ignore', divide='ignore'):
            self.k_mean = self.bin_sum(kk) / self.counts
        self.k_mean[self.counts == 0] = np.nan

    def bin_sum(self, values: np.ndarray) -> np.ndarray:
        """Sum an ``rfft2``-shaped array over each bin, full plane.

        Args:
            values (np.ndarray): Real array of shape ``(n, n//2 + 1)``.

        Returns:
            np.ndarray: Per-bin sums, shape ``(n_bins,)``.
        """
        weighted = (np.asarray(values, dtype=np.float64)
                    * self._mult[None, :]).ravel()
        return np.bincount(self._idx, weights=weighted,
                           minlength=self.n_bins)

    def cross_power(self, F_a: np.ndarray, F_b: np.ndarray) -> np.ndarray:
        """Binned cross spectrum of two maps from their ``rfft2``.

        Args:
            F_a (np.ndarray): ``rfft2`` of the first overdensity map.
            F_b (np.ndarray): ``rfft2`` of the second (may be ``F_a``).

        Returns:
            np.ndarray: ``P(k)`` in arcmin^2 per bin, NaN in empty bins.
        """
        re = np.real(F_a * np.conj(F_b))
        norm = self.area / float(self.n_pixels) ** 4
        with np.errstate(invalid='ignore', divide='ignore'):
            p = self.bin_sum(re) * norm / self.counts
        p[self.counts == 0] = np.nan
        return p

    def full_plane_sum(self, values: np.ndarray) -> float:
        """Sum an ``rfft2``-shaped array over the full Fourier plane.

        Args:
            values (np.ndarray): Real array of shape ``(n, n//2 + 1)``.

        Returns:
            float: The sum, each column weighted by its multiplicity.
        """
        return float(np.sum(np.asarray(values, dtype=np.float64)
                            * self._mult[None, :]))

    def weighted(self, values: np.ndarray) -> np.ndarray:
        """Return ``values`` times the full-plane multiplicity.

        Args:
            values (np.ndarray): Real array of shape ``(n, n//2 + 1)``.

        Returns:
            np.ndarray: Same shape, float64.
        """
        return np.asarray(values, dtype=np.float64) * self._mult[None, :]

    def per_mode(self, per_bin: np.ndarray) -> np.ndarray:
        """Broadcast a per-bin function back onto every ``rfft2`` mode.

        Args:
            per_bin (np.ndarray): One value per bin, shape ``(n_bins,)``.

        Returns:
            np.ndarray: Shape ``(n, n//2 + 1)``, piecewise constant in bins.
        """
        return np.asarray(per_bin, dtype=np.float64)[self._idx].reshape(
            self.shape)


def exact_amplitude(kernel_spec: np.ndarray, weighted_product: np.ndarray,
                    n_pixels: int) -> float:
    """Filtered amplitude as an exact sum over Fourier modes (Parseval).

    ``Y_ab = (1/n^4) sum_{k, full plane} K(k) Re[F_a F_b*]`` equals the
    map-level ``<(K * delta_a) delta_b>`` of ``rprofiles.compute_Y_matrix`` to
    rounding (without the galaxy self-pair term, which the caller subtracts).

    Args:
        kernel_spec (np.ndarray): Real kernel spectrum on the ``rfft2`` grid.
        weighted_product (np.ndarray): ``Re[F_a F_b*]`` already multiplied by
            the full-plane multiplicity (``ModeBinning.weighted``).
        n_pixels (int): Pixels per side.

    Returns:
        float: The amplitude.
    """
    return float(np.dot(np.ravel(kernel_spec), np.ravel(weighted_product))
                 ) / float(n_pixels) ** 4


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def dog_kernel_spectrum(n_pixels: int, pixel_arcmin: float, sigma1: float,
                        sigma2: float) -> np.ndarray:
    """Difference-of-Gaussians kernel spectrum on the ``rfft2`` grid.

    Formalism Eq. (22): ``W_DoG = G(sigma1) - G(sigma2)`` with unit-normalized
    Gaussians, so ``K(k) = [exp(-k^2 sigma1^2/2) - exp(-k^2 sigma2^2/2)] /
    pixel^2`` in the pipeline normalization.  Defined directly in Fourier
    space, so ``K(0) = 0`` and ``K(k) > 0`` for every ``k != 0`` hold exactly
    (a compensated kernel with a strictly positive window).

    Args:
        n_pixels (int): Pixels per side.
        pixel_arcmin (float): Pixel size, arcmin.
        sigma1 (float): Inner Gaussian width, arcmin.
        sigma2 (float): Outer Gaussian width, arcmin; must exceed ``sigma1``.

    Returns:
        np.ndarray: Real kernel spectrum, shape ``(n, n//2 + 1)``.

    Raises:
        ValueError: If the widths are not ``0 < sigma1 < sigma2``.
    """
    if not 0.0 < sigma1 < sigma2:
        raise ValueError(
            f'Need 0 < sigma1 < sigma2, got {sigma1!r}, {sigma2!r}.')
    kk = rfft_wavenumbers(n_pixels, pixel_arcmin)
    k2 = kk * kk
    spec = np.exp(-0.5 * k2 * sigma1 ** 2) - np.exp(-0.5 * k2 * sigma2 ** 2)
    return spec / pixel_arcmin ** 2


def dog_window(k: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    """Continuum harmonic DoG window, formalism Eq. (22).

    Args:
        k (np.ndarray): Wavenumbers, 1/arcmin.
        sigma1 (float): Inner width, arcmin.
        sigma2 (float): Outer width, arcmin.

    Returns:
        np.ndarray: ``exp(-k^2 s1^2/2) - exp(-k^2 s2^2/2)``.
    """
    k = np.asarray(k, dtype=np.float64)
    return np.exp(-0.5 * (k * sigma1) ** 2) - np.exp(-0.5 * (k * sigma2) ** 2)


def dog_zero_lag(sigma1: float, sigma2: float) -> float:
    """Continuum ``W_DoG(0) = (sigma1^-2 - sigma2^-2) / (2 pi)``, arcmin^-2.

    Args:
        sigma1 (float): Inner width, arcmin.
        sigma2 (float): Outer width, arcmin.

    Returns:
        float: The central value, formalism Eq. (23).
    """
    return (sigma1 ** -2 - sigma2 ** -2) / (2.0 * np.pi)


def zero_lag(kernel_spec: np.ndarray, n_pixels: int) -> float:
    """Real-space kernel value at lag zero from its spectrum.

    ``W(0) = (1/n^2) sum_{k, full plane} K(k)``.  This is the ``K(0)`` of the
    self-pair term ``K(0)/nbar_pix`` that ``rprofiles.compute_Y_matrix``
    subtracts from the galaxy auto.  Applied to ``K^2`` it gives
    ``sum_x W(x)^2``, the self-pair term of the doubly filtered auto.

    Args:
        kernel_spec (np.ndarray): Real kernel spectrum on the ``rfft2`` grid.
        n_pixels (int): Pixels per side.

    Returns:
        float: Kernel value at the origin, arcmin^-2 in pipeline units.
    """
    mult = rfft_multiplicity(n_pixels)
    total = float(np.sum(np.asarray(kernel_spec, dtype=np.float64)
                         * mult[None, :]))
    return total / float(n_pixels) ** 2


def response_quantiles_values(k: np.ndarray, p_2d: np.ndarray,
                              window: np.ndarray,
                              quantiles: Sequence[float] = (0.05, 0.5, 0.95)
                              ) -> np.ndarray:
    """Where a kernel's signed response to a spectrum accumulates.

    Same definition as ``make_task9_spectra.response_quantiles`` (formalism
    Eq. 64), but taking the kernel values directly, so any kernel -- the DoG
    included -- can be placed on the same ``k_50`` axis.

    Args:
        k (np.ndarray): Wavenumbers, strictly increasing, 1/arcmin.
        p_2d (np.ndarray): Spectrum at ``k``.
        window (np.ndarray): Kernel ``W(k)`` at ``k``.
        quantiles (sequence, optional): Fractions to locate.

    Returns:
        np.ndarray: Wavenumbers at the requested fractions; NaN where the
        total response is zero.
    """
    k = np.asarray(k, dtype=np.float64)
    good = k > 0
    k = k[good]
    integrand = k * np.asarray(p_2d, dtype=np.float64)[good] \
        * np.asarray(window, dtype=np.float64)[good]
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(k))])
    total = cumulative[-1]
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


# ---------------------------------------------------------------------------
# Map-level filtered amplitudes for arbitrary kernel spectra
# ---------------------------------------------------------------------------

def filtered_amplitudes(deltas: Dict[str, np.ndarray],
                        kernel_specs: Sequence[np.ndarray],
                        zero_lags: Sequence[float],
                        nbar_pix: float = None,
                        galaxy_key: str = 'g',
                        n_jk_side: int = 4) -> dict:
    """``rprofiles.compute_Y_matrix`` for an arbitrary list of kernels.

    Same operations in the same order as ``compute_Y_matrix`` -- correlate
    each field with the kernel once, multiply by every other field, reduce to
    per-block sums, subtract the galaxy self-pair term ``K(0)/nbar_pix`` --
    so a pixelized DSigma kernel reproduces its amplitudes exactly.  Used for
    the DoG sweep, whose kernels ``compute_Y_matrix`` does not build.

    Args:
        deltas (dict): Field key to overdensity map, all the same square
            shape.
        kernel_specs (sequence): Kernel spectra as multiplied in
            ``rprofiles.filtered_map`` (real for symmetric kernels).
        zero_lags (sequence): Real-space kernel value at lag zero, one per
            kernel, for the self-pair subtraction.
        nbar_pix (float, optional): Mean galaxies per pixel; required if
            ``galaxy_key`` is in ``deltas``.
        galaxy_key (str, optional): Key of the discrete field.
        n_jk_side (int, optional): Jackknife blocks per side.

    Returns:
        dict: ``{'Y': {pair: (n_kernels,)}, 'Y_jk': {pair: (n_jk,
        n_kernels)}, 'n_jk': int}``, pairs sorted as in ``rprofiles``.

    Raises:
        ValueError: On shape mismatch, or a galaxy field without
            ``nbar_pix``.
    """
    import rprofiles as rp  # deferred: keeps this module importable alone

    keys = sorted(deltas)
    shape = deltas[keys[0]].shape
    if shape[0] != shape[1] or any(deltas[k].shape != shape for k in keys):
        raise ValueError('All maps must be square and share one shape.')
    if galaxy_key in deltas and nbar_pix is None:
        raise ValueError('nbar_pix is required for the galaxy self-pairs.')
    if len(kernel_specs) != len(zero_lags):
        raise ValueError('One zero-lag value is needed per kernel.')

    edges = rp.block_edges(shape[0], n_jk_side)
    counts = rp.block_counts(edges)
    n_jk = len(counts)
    pairs = [rp._pair_key(a, b) for i, a in enumerate(keys) for b in keys[i:]]
    raw = {p: np.empty((len(kernel_specs), n_jk)) for p in pairs}
    spectra = {k: scipy.fft.rfft2(deltas[k], workers=FFT_WORKERS)
               for k in keys}

    for ik, (kspec, k0) in enumerate(zip(kernel_specs, zero_lags)):
        for ia, a in enumerate(keys):
            fmap = rp.filtered_map(spectra[a], kspec, shape)
            for b in keys[ia:]:
                sums = rp.block_sums(fmap * deltas[b], edges)
                if a == galaxy_key and b == galaxy_key:
                    sums = sums - float(k0) / nbar_pix * counts
                raw[rp._pair_key(a, b)][ik] = sums
            del fmap

    Y = {p: raw[p].sum(axis=1) / counts.sum() for p in pairs}
    Y_jk = {p: rp.jackknife_realizations(raw[p], counts).T for p in pairs}
    return {'Y': Y, 'Y_jk': Y_jk, 'n_jk': n_jk}


# ---------------------------------------------------------------------------
# The window/mediation split (formalism Eq. 48)
# ---------------------------------------------------------------------------

def binned_amplitude(w: np.ndarray, p: np.ndarray, area: float) -> np.ndarray:
    """Filtered amplitude from binned kernel weights and a binned spectrum.

    ``Y = (1/area) sum_bins w(bin) P(bin)``; empty bins (NaN spectrum) carry
    zero weight and are skipped.

    Args:
        w (np.ndarray): Kernel weights, shape ``(..., n_bins)``.
        p (np.ndarray): Spectrum, shape ``(n_bins,)``.
        area (float): Map area, arcmin^2.

    Returns:
        np.ndarray: Amplitudes, shape ``w.shape[:-1]``.
    """
    p = np.where(np.isfinite(p), p, 0.0)
    return (np.asarray(w, dtype=np.float64) @ p) / area


def mediated_spectrum(p_xm: np.ndarray, p_gm: np.ndarray,
                      p_mm: np.ndarray) -> np.ndarray:
    """The galaxy-gas spectrum exact mediation would predict.

    Under ``delta_X = eta delta_m + eps`` with ``<eps delta_g> = 0``,
    ``P_gX = eta P_gm = P_Xm P_gm / P_mm`` (formalism Eq. 46-47).

    Args:
        p_xm (np.ndarray): Gas-matter spectrum.
        p_gm (np.ndarray): Galaxy-matter spectrum.
        p_mm (np.ndarray): Matter auto spectrum.

    Returns:
        np.ndarray: ``P_Xm P_gm / P_mm``, NaN where ``P_mm <= 0``.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(p_mm > 0, p_xm * p_gm / p_mm, np.nan)


def split_from_amplitudes(Y_xm, Y_gm, Y_mm, Y_gx, Y_med
                          ) -> Dict[str, np.ndarray]:
    """Split the filtered calibration factor into window and mediation parts.

    With ``Y_gX^med = int dmu P_Xm P_gm / P_mm``, the galaxy-gas amplitude
    that exact mediation would predict,

        C_F = Y_Xm Y_gm / (Y_mm Y_gX)
            = [Y_Xm Y_gm / (Y_mm Y_gX^med)] * [Y_gX^med / Y_gX]
            =            W_F                *        M_F

    ``W_F = <eta><beta>/<eta beta>`` is the window term of formalism Eq. (48),
    present even under exact mediation.  ``M_F`` is identically 1 when
    ``C(k) = 1`` at every ``k``, so its departure from 1 is mediation failure
    seen through the filter.  ``C_F = W_F M_F`` holds exactly.

    Args:
        Y_xm, Y_gm, Y_mm, Y_gx (array-like): Filtered amplitudes.
        Y_med (array-like): The mediated galaxy-gas amplitude.

    Returns:
        dict: ``C``, ``W`` and ``M``, NaN where a denominator vanishes.
    """
    Y_xm, Y_gm, Y_mm, Y_gx, Y_med = (np.asarray(a, dtype=np.float64)
                                     for a in (Y_xm, Y_gm, Y_mm, Y_gx, Y_med))
    with np.errstate(invalid='ignore', divide='ignore'):
        C = np.where(Y_mm * Y_gx != 0, Y_xm * Y_gm / (Y_mm * Y_gx), np.nan)
        M = np.where(Y_gx != 0, Y_med / Y_gx, np.nan)
        W = np.where(Y_mm * Y_med != 0, Y_xm * Y_gm / (Y_mm * Y_med), np.nan)
    return {'C': C, 'W': W, 'M': M}


def window_mediation_split(w: np.ndarray, p_xm: np.ndarray, p_gm: np.ndarray,
                           p_mm: np.ndarray, p_gx: np.ndarray,
                           area: float) -> Dict[str, np.ndarray]:
    """The split of :func:`split_from_amplitudes` from binned quantities.

    Every amplitude is ``(1/area) sum_bins w P``.  This approximates the
    exact per-mode sums to the extent that the kernel is constant over the
    modes of a bin; it is the cross-check of the exact route that
    ``make_ck_spectra.py`` takes, and the route for any kernel whose exact
    sums were not stored.

    Args:
        w (np.ndarray): Kernel weights, shape ``(n_R, n_bins)``.
        p_xm, p_gm, p_mm, p_gx (np.ndarray): Binned spectra, ``(n_bins,)``.
        area (float): Map area, arcmin^2.

    Returns:
        dict: ``Y_Xm``, ``Y_gm``, ``Y_mm``, ``Y_gX``, ``Y_gX_med``, ``C``,
        ``W``, ``M``, each shape ``(n_R,)``.
    """
    out = {'Y_Xm': binned_amplitude(w, p_xm, area),
           'Y_gm': binned_amplitude(w, p_gm, area),
           'Y_mm': binned_amplitude(w, p_mm, area),
           'Y_gX': binned_amplitude(w, p_gx, area),
           'Y_gX_med': binned_amplitude(
               w, mediated_spectrum(p_xm, p_gm, p_mm), area)}
    out.update(split_from_amplitudes(out['Y_Xm'], out['Y_gm'], out['Y_mm'],
                                     out['Y_gX'], out['Y_gX_med']))
    return out


def derived_filter_weights(w_base: np.ndarray, radii: np.ndarray, kind: str,
                           ref: float, atol: float = 1e-9) -> np.ndarray:
    """Kernel weights of Upsilon or the Y transform from a base filter's.

    The kernels are linear in the base kernel, so their bin weights are the
    same linear combinations as the amplitudes in
    ``make_calibration_factor.assemble_derived_filter``:
    ``Upsilon(R; R0) = DSigma(R) - (R0/R)^2 DSigma(R0)`` and
    ``Y(R; Rmax) = Sigma(R) - Sigma(Rmax)``.

    Args:
        w_base (np.ndarray): Base-filter weights, ``(n_R, n_bins)``.
        radii (np.ndarray): Aperture grid, arcmin.
        kind (str): ``'Upsilon'`` or ``'Ytransform'``.
        ref (float): ``R0`` or ``Rmax``, arcmin; must be on the grid.
        atol (float, optional): Tolerance for locating ``ref``.

    Returns:
        np.ndarray: Weights of the derived filter, ``(n_R, n_bins)``.

    Raises:
        ValueError: If ``kind`` is unknown or ``ref`` is off the grid.
    """
    radii = np.asarray(radii, dtype=np.float64)
    hits = np.flatnonzero(np.abs(radii - float(ref)) <= atol)
    if hits.size == 0:
        raise ValueError(f'Reference radius {ref} is not on the grid.')
    idx = int(hits[0])
    if kind == 'Upsilon':
        scale = (float(ref) / radii) ** 2
    elif kind == 'Ytransform':
        scale = np.ones_like(radii)
    else:
        raise ValueError(f"kind must be 'Upsilon' or 'Ytransform': {kind!r}")
    return w_base - scale[:, None] * w_base[idx][None, :]


def rebin_spectra(counts: np.ndarray, k_mean: np.ndarray,
                  spectra: Dict[str, np.ndarray], new_edges: np.ndarray
                  ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Merge fine linear bins into coarser (e.g. logarithmic) bins.

    Spectra are averaged with the mode counts as weights, so a ratio formed
    after rebinning is a ratio of averaged spectra, not an average of noisy
    per-bin ratios.

    Args:
        counts (np.ndarray): Modes per fine bin.
        k_mean (np.ndarray): Mean ``|k|`` per fine bin (NaN if empty).
        spectra (dict): Name to fine-binned spectrum.
        new_edges (np.ndarray): Coarse edges, same units as ``k_mean``.

    Returns:
        tuple: ``(k_coarse, spectra_coarse)``; empty coarse bins are dropped.
    """
    good = (counts > 0) & np.isfinite(k_mean)
    which = np.digitize(k_mean[good], new_edges) - 1
    n_new = len(new_edges) - 1
    inside = (which >= 0) & (which < n_new)
    c = counts[good][inside]
    which = which[inside]
    csum = np.bincount(which, weights=c, minlength=n_new)
    keep = csum > 0
    k_c = np.bincount(which, weights=c * k_mean[good][inside],
                      minlength=n_new)[keep] / csum[keep]
    out = {}
    for name, p in spectra.items():
        vals = np.asarray(p, dtype=np.float64)[good][inside]
        out[name] = np.bincount(which, weights=c * vals,
                                minlength=n_new)[keep] / csum[keep]
    return k_c, out


def split_by_parent_mass(indices: np.ndarray, parent_mass: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Split a galaxy sample into halves by parent-halo mass.

    Ranks by parent mass with a stable sort, so ties (satellites sharing a
    host) fall on one side deterministically; the low half gets
    ``floor(n/2)`` galaxies.

    Args:
        indices (np.ndarray): Catalogue indices of the sample.
        parent_mass (np.ndarray): Parent-halo mass of each, same length.

    Returns:
        tuple: ``(low, high)`` index arrays, disjoint, covering the sample.

    Raises:
        ValueError: If the lengths differ or the sample has fewer than two.
    """
    indices = np.asarray(indices)
    parent_mass = np.asarray(parent_mass, dtype=np.float64)
    if indices.shape != parent_mass.shape or indices.size < 2:
        raise ValueError('Need matching arrays with at least two entries.')
    order = np.argsort(parent_mass, kind='stable')
    half = indices.size // 2
    return indices[order[:half]], indices[order[half:]]


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------

def loglog_slope(radii: np.ndarray, values: np.ndarray
                 ) -> Tuple[float, np.ndarray]:
    """Least-squares slope of ``ln|values|`` against ``ln radii``.

    Args:
        radii (np.ndarray): Positive abscissae.
        values (np.ndarray): Same sign throughout (a power law).

    Returns:
        tuple: ``(global_slope, local_slopes)``, the latter between adjacent
        points.

    Raises:
        ValueError: If ``values`` change sign or contain zeros.
    """
    radii = np.asarray(radii, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if not (np.all(values > 0) or np.all(values < 0)):
        raise ValueError('values must have one sign for a log slope.')
    lx = np.log(radii)
    ly = np.log(np.abs(values))
    slope = float(np.polyfit(lx, ly, 1)[0])
    return slope, np.diff(ly) / np.diff(lx)


def loo_linear_prediction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Predict each point from a straight line fitted to the others.

    Args:
        x (np.ndarray): Abscissae, at least three points.
        y (np.ndarray): Ordinates.

    Returns:
        np.ndarray: Leave-one-out predictions of ``y``.

    Raises:
        ValueError: With fewer than three points.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        raise ValueError('Need at least three points.')
    out = np.empty_like(y)
    for i in range(x.size):
        keep = np.arange(x.size) != i
        b, a = np.polyfit(x[keep], y[keep], 1)
        out[i] = a + b * x[i]
    return out


def predict_from_fit(x_fit: np.ndarray, y_fit: np.ndarray,
                     x_new: float) -> float:
    """Fit a straight line and evaluate it at one point.

    Args:
        x_fit (np.ndarray): Abscissae of the fit, at least two points.
        y_fit (np.ndarray): Ordinates of the fit.
        x_new (float): Where to evaluate.

    Returns:
        float: The prediction.
    """
    b, a = np.polyfit(np.asarray(x_fit, float), np.asarray(y_fit, float), 1)
    return float(a + b * x_new)


def interp_log_k(k_src: np.ndarray, y_src: np.ndarray,
                 k_new: np.ndarray) -> np.ndarray:
    """Interpolate ``y`` in ``ln k``, NaN outside the source range.

    The source need not be sorted (``k_50`` falls with aperture); NaNs in the
    source are dropped.

    Args:
        k_src (np.ndarray): Source wavenumbers.
        y_src (np.ndarray): Source values.
        k_new (np.ndarray): Target wavenumbers.

    Returns:
        np.ndarray: Interpolated values at ``k_new``.
    """
    k_src = np.asarray(k_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    ok = np.isfinite(k_src) & np.isfinite(y_src) & (k_src > 0)
    if ok.sum() < 2:
        return np.full(np.shape(k_new), np.nan)
    order = np.argsort(k_src[ok])
    lk = np.log(k_src[ok][order])
    ys = y_src[ok][order]
    k_new = np.asarray(k_new, dtype=np.float64)
    out = np.full(k_new.shape, np.nan)
    inside = np.isfinite(k_new) & (k_new > 0)
    lk_new = np.log(np.where(inside, k_new, 1.0))
    inside &= (lk_new >= lk[0]) & (lk_new <= lk[-1])
    out[inside] = np.interp(lk_new[inside], lk, ys)
    return out


def paired_correlation(jk_a: np.ndarray, jk_b: np.ndarray) -> np.ndarray:
    """Pearson correlation of two jackknife stacks, aperture by aperture.

    Args:
        jk_a (np.ndarray): Leave-one-out values, ``(n_jk, n_R)``.
        jk_b (np.ndarray): Same shape, second run.

    Returns:
        np.ndarray: Correlation per aperture, NaN where either is constant.
    """
    a = np.asarray(jk_a, dtype=np.float64)
    b = np.asarray(jk_b, dtype=np.float64)
    da = a - a.mean(axis=0)
    db = b - b.mean(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (da * db).sum(axis=0) / np.sqrt(
            (da * da).sum(axis=0) * (db * db).sum(axis=0))


# ---------------------------------------------------------------------------
# Stage 4: the 3D component spectra
# ---------------------------------------------------------------------------

#: Baryonic components of the unbound-gas 3D spectra, in their stored order.
BARYON_COMPONENTS = ('ionized_gas', 'neutral_gas', 'Stars', 'BH')


def component_bookkeeping(P: np.ndarray, means: np.ndarray,
                          components: Sequence[str]) -> Dict[str, object]:
    """Build the CDM/baryon/total spectra from the 3D component matrix.

    ``P[i, j]`` are auto/cross spectra of the component overdensities, and
    ``means`` the component mean densities on the same grid.  With mass
    weights ``w_i = rho_i / rho_b`` over the baryonic components,
    ``delta_b = sum_i w_i delta_i`` and ``delta_t = f_m delta_DM + f_b
    delta_b`` exactly (formalism Eq. 4 and 67), so

        P_bm = sum_i w_i P_{i,DM},  P_bb = sum_ij w_i w_j P_ij,
        P_tt = f_m^2 P_mm + 2 f_m f_b P_bm + f_b^2 P_bb,
        x_i = P_{i,DM} / P_{DM,DM},  x_b = sum_i w_i x_i.

    Args:
        P (np.ndarray): Component spectra, ``(n_c, n_c, n_k)``.
        means (np.ndarray): Component means, ``(n_c,)``.
        components (sequence): Component names; must include ``'DM'`` and
            every name in :data:`BARYON_COMPONENTS`.

    Returns:
        dict: ``f_b``, ``weights`` (baryon mass weights by name), ``P_mm``,
        ``P_bm``, ``P_bb``, ``P_tt``, ``x`` (``P_bm/P_mm``) and ``x_comp``
        (per-component ``x_i``).

    Raises:
        KeyError: If a required component is missing.
    """
    comps = list(components)
    i_dm = comps.index('DM')
    i_b = [comps.index(c) for c in BARYON_COMPONENTS]
    means = np.asarray(means, dtype=np.float64)
    rho_b = means[i_b].sum()
    f_b = rho_b / means.sum()
    f_m = 1.0 - f_b
    w = means[i_b] / rho_b
    P_mm = P[i_dm, i_dm]
    P_bm = np.einsum('i,ik->k', w, P[i_b, i_dm])
    P_bb = np.einsum('i,j,ijk->k', w, w, P[np.ix_(i_b, i_b)])
    P_tt = f_m * f_m * P_mm + 2.0 * f_m * f_b * P_bm + f_b * f_b * P_bb
    with np.errstate(invalid='ignore', divide='ignore'):
        x_comp = {c: P[comps.index(c), i_dm] / P_mm
                  for c in BARYON_COMPONENTS}
        x = P_bm / P_mm
    return {'f_b': f_b,
            'weights': dict(zip(BARYON_COMPONENTS, w)),
            'P_mm': P_mm, 'P_bm': P_bm, 'P_bb': P_bb, 'P_tt': P_tt,
            'x': x, 'x_comp': x_comp}
