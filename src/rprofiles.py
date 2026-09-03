"""r-profile computation via periodic FFT aperture filtering.

Implements Task 1 of ``docs/cross_correlation_notes.md`` as specified in
``docs/r_profiles_task1_spec.md``: the filtered cross-correlation
coefficients

    r_XY(R; F) = Y_XY / sqrt(Y_XX * Y_YY),

where every filtered amplitude is computed by ONE operation on periodic 2D
maps -- correlate field X (as an overdensity) with the pixelized aperture
kernel, multiply pixelwise by field Y (as an overdensity), take the map mean:

    Y_XY(R; F) = < F_R[delta_X] * delta_Y >_map .

Stamp-stacking a filtered map at galaxy positions is mathematically identical
to this map-level average on a periodic box, so the same routine produces the
galaxy-crossed pairs AND the field-field pairs (bm, mm, bb) that the existing
stamp-based stacker cannot reach.

Conventions (fixed by the spec, do not vary):

- Every field enters as ``delta = X / <X> - 1``.
- Aperture kernels reproduce :func:`filters.delta_sigma_kernel`'s pixel-count
  normalization exactly (disk ``+1/(pixArea*N_disk)``, annulus
  ``-1/(pixArea*N_ann)``, membership by pixel-centre radius).
- ``Sigma`` is the annulus mean over ``[R, R+dr]`` (positive, uncompensated);
  ``DSigma`` is the compensated disk-minus-annulus filter; ``Upsilon`` is the
  linear combination ``DSigma(R) - (r0/R)**2 * DSigma(r0)``, formed at the
  amplitude level rather than by extra convolutions.
- The galaxy auto-correlation gets an analytic self-pair subtraction of
  ``K(0) / nbar_pix``, with ``nbar_pix`` the mean galaxy count per pixel.
- Errors come from a spatial ``n_jk_side x n_jk_side`` block jackknife.  The
  three Y's entering each r are measured on the same map and are strongly
  correlated, so r is formed PER jackknife realization and the spread is
  taken afterwards -- never Gaussian propagation of marginal Y errors.
- No beam anywhere: the r's are intrinsic field properties.

Typical use::

    deltas = {'g': dg, 'e': de, 'b': db, 'm': dm}
    Ymat = compute_Y_matrix(deltas, pixel_arcmin, nbar_pix=nbar)
    prof = r_profiles(Ymat)
    prof['r'][('g', 'b')]['DSigma']        # r_gb(R) for the DSigma filter
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.fft

from halos import select_halos
from tools import hist2d_numba_seq

# All transforms go through scipy.fft with workers=-1 rather than numpy.fft,
# which is single-threaded.  Several of the cached production grids have large
# prime factors (14015 = 5 x 2803, 4822 = 2 x 2411) and fall back to Bluestein's
# algorithm, where the multithreading is worth an order of magnitude or more:
# measured 42x on a 4822^2 grid, 9x on 2674^2.  utils.fft_smoothed_map already
# uses scipy.fft the same way.
_FFT_WORKERS = -1

# ---------------------------------------------------------------------------
# Fixed numerical conventions (spec, "Fixed numerical conventions")
# ---------------------------------------------------------------------------

#: Base filters computed by convolution.  'Upsilon' is derived from 'DSigma'.
BASE_FILTERS: Tuple[str, ...] = ('Sigma', 'DSigma')

#: All filters reported.
FILTERS: Tuple[str, ...] = ('Sigma', 'DSigma', 'Upsilon')

#: Nine linear aperture bins over 1'-6', matching the existing pipeline.
APERTURES_ARCMIN: np.ndarray = np.linspace(1.0, 6.0, 9)

#: Annulus width of the compensated filter, in arcmin (f_gas paper convention).
DR_ARCMIN: float = 0.75

#: Upsilon reference radius, in arcmin.
R0_ARCMIN: float = 1.0

#: Jackknife blocks per side (4x4 = 16 leave-one-out patches).
N_JK_SIDE: int = 4


# ---------------------------------------------------------------------------
# Field preparation
# ---------------------------------------------------------------------------

def to_overdensity(field: np.ndarray) -> np.ndarray:
    """Convert a positive field to the overdensity ``field / <field> - 1``.

    The mean is accumulated in float64 regardless of the input dtype, so
    float32 caches do not lose precision in the normalization.

    Args:
        field (np.ndarray): 2D field of a non-negative quantity (mass,
            electron count, galaxy count) per pixel.

    Returns:
        np.ndarray: Overdensity field of the same shape, dtype float64, with
        a mean of zero to machine precision.

    Raises:
        ValueError: If the field mean is not strictly positive, which would
            make the normalization meaningless.
    """
    mean = float(np.mean(field, dtype=np.float64))
    if not mean > 0.0:
        raise ValueError(
            f"Cannot form an overdensity from a field with mean {mean!r}; "
            "the mean must be strictly positive."
        )
    return np.asarray(field, dtype=np.float64) / mean - 1.0


def derive_cdm_field(total_field: np.ndarray, baryon_field: np.ndarray,
                     header: Optional[dict] = None,
                     tolerance: float = 0.05,
                     verbose: bool = True) -> np.ndarray:
    """Derive the CDM-only projected map as ``total - baryon``.

    ``mapMaker.make_combined_field`` builds ``'total'`` as gas + DM + Stars +
    BH and ``'baryon'`` as gas + Stars + BH on identical grids, so the
    difference is the DM (PartType1) map exactly, up to float64 round-off.
    This avoids a full DM particle sweep whenever both composite fields are
    already cached.

    The derived CDM fraction is checked against the box cosmology
    ``1 - OmegaBaryon/Omega0``.  The check is a warning rather than an error
    because suites with massive neutrinos (FLAMINGO) carry a neutrino
    contribution in ``Omega0`` that is absent from the particle maps, so a
    per-cent-level offset there is expected and harmless.

    Args:
        total_field (np.ndarray): Cached ``'total'`` field (gas+DM+Stars+BH).
        baryon_field (np.ndarray): Cached ``'baryon'`` field (gas+Stars+BH),
            same shape and grid as ``total_field``.
        header (dict, optional): Simulation header providing ``'OmegaBaryon'``
            and ``'Omega0'`` for the consistency check.  Defaults to None,
            which skips the check.
        tolerance (float, optional): Relative tolerance on the CDM mass
            fraction before a warning is emitted.  Defaults to 0.05.
        verbose (bool, optional): If True, print the measured and expected
            CDM fractions.  Defaults to True.

    Returns:
        np.ndarray: CDM-only projected field, dtype float64.

    Raises:
        ValueError: If the two fields have different shapes, or if the
            derived field is negative by more than float64 round-off (which
            would mean the two caches are not a matched pair).
    """
    if total_field.shape != baryon_field.shape:
        raise ValueError(
            f"total/baryon shape mismatch: {total_field.shape} vs "
            f"{baryon_field.shape}; the two caches must share a grid."
        )

    cdm = np.asarray(total_field, dtype=np.float64) - np.asarray(
        baryon_field, dtype=np.float64)

    # Round-off can only produce values of order eps*total, never a physically
    # negative cell; anything larger means the caches are not a matched pair.
    worst = float(cdm.min())
    if worst < 0.0:
        scale = float(np.max(np.abs(total_field))) * 1e-12
        if worst < -scale:
            raise ValueError(
                f"Derived CDM field has a significantly negative cell "
                f"({worst:.6e}); 'total' and 'baryon' are not a matched pair."
            )
        cdm = np.clip(cdm, 0.0, None)

    if header is not None and 'OmegaBaryon' in header and 'Omega0' in header:
        f_cdm = float(np.mean(cdm, dtype=np.float64)
                      / np.mean(total_field, dtype=np.float64))
        f_expect = 1.0 - float(header['OmegaBaryon']) / float(header['Omega0'])
        if verbose:
            print(f"    CDM mass fraction: measured {f_cdm:.6f}, "
                  f"cosmology {f_expect:.6f} "
                  f"(delta {f_cdm - f_expect:+.2e})")
        if abs(f_cdm - f_expect) > tolerance * f_expect:
            warnings.warn(
                f"Derived CDM fraction {f_cdm:.4f} differs from the box "
                f"cosmology {f_expect:.4f} by more than {tolerance:.0%}. "
                "Expected at the per-cent level for suites with massive "
                "neutrinos; investigate if larger.",
                stacklevel=2,
            )

    return cdm


# ---------------------------------------------------------------------------
# Aperture kernels
# ---------------------------------------------------------------------------

def build_aperture_kernel(n_pixels: int, pixel_arcmin: float, R: float,
                          filter_type: str, dr: float = DR_ARCMIN
                          ) -> np.ndarray:
    """Build a periodic aperture kernel on the full map grid.

    The kernel is centred at lag (0, 0) with periodic wraparound (the
    ``numpy.fft`` convention), so correlating a map with it evaluates the
    aperture filter at every pixel simultaneously.  Pixel membership is by
    pixel-centre radius ``r = pixel_arcmin * hypot(i, j)`` over integer lags
    ``(i, j)``, and the weights reproduce
    :func:`filters.delta_sigma_kernel`'s pixel-count normalization exactly:

    - ``'DSigma'``: ``+1/(pixArea*N_disk)`` on ``r < R`` and
      ``-1/(pixArea*N_ann)`` on ``R <= r < R+dr``.  Sums to zero
      (compensated) to machine precision.
    - ``'Sigma'``: ``+1/(pixArea*N_ann)`` on ``R <= r < R+dr`` only, i.e. the
      annulus-mean (local surface density) estimate, positive weights,
      uncompensated.

    Only a small stamp of half-width ``ceil((R+dr)/pixel_arcmin)`` is
    evaluated and then embedded into the full grid, so memory scales with the
    output grid rather than with an intermediate radius array.

    Args:
        n_pixels (int): Number of pixels per side of the (square) map.
        pixel_arcmin (float): Angular size of one pixel, in arcmin.
        R (float): Aperture radius, in arcmin.
        filter_type (str): One of ``'Sigma'``, ``'DSigma'``.
        dr (float, optional): Annulus width in arcmin.  Defaults to
            :data:`DR_ARCMIN`.

    Returns:
        np.ndarray: Kernel of shape ``(n_pixels, n_pixels)``, dtype float64.

    Raises:
        ValueError: If ``filter_type`` is not a base filter, if ``dr`` or
            ``R`` is not positive, if the kernel support does not fit inside
            the periodic box, or if the disk or annulus contains no pixels.
            The last case is the resolution guard demanded by the spec: an
            under-resolved aperture must raise, never silently degrade.
    """
    if filter_type not in BASE_FILTERS:
        raise ValueError(
            f"filter_type must be one of {BASE_FILTERS}, got {filter_type!r}. "
            "('Upsilon' is derived from 'DSigma' amplitudes, not convolved.)"
        )
    if not R > 0:
        raise ValueError(f"R must be positive, got {R!r}.")
    if not dr > 0:
        raise ValueError(f"dr must be positive, got {dr!r}.")
    if not pixel_arcmin > 0:
        raise ValueError(f"pixel_arcmin must be positive, got {pixel_arcmin!r}.")

    R_out = R + dr
    half = int(np.ceil(R_out / pixel_arcmin))
    if 2 * half + 1 > n_pixels:
        raise ValueError(
            f"Aperture R+dr={R_out:.3f}' needs a stamp of {2*half+1} pixels, "
            f"which does not fit in an {n_pixels}-pixel periodic box."
        )

    lags = np.arange(-half, half + 1, dtype=np.float64)
    rr = np.hypot(lags[:, None], lags[None, :]) * pixel_arcmin

    disk = rr < R
    annulus = (rr >= R) & (rr < R_out)
    n_disk = int(disk.sum())
    n_ann = int(annulus.sum())

    if n_ann == 0:
        raise ValueError(
            f"Empty annulus for R={R:.3f}', dr={dr:.3f}' at "
            f"pixel_arcmin={pixel_arcmin:.4f}: the aperture is unresolved. "
            "Increase the map resolution."
        )
    if filter_type == 'DSigma' and n_disk == 0:
        raise ValueError(
            f"Empty disk for R={R:.3f}' at pixel_arcmin={pixel_arcmin:.4f}: "
            "the aperture is unresolved. Increase the map resolution."
        )

    pix_area = pixel_arcmin ** 2
    stamp = np.zeros_like(rr)
    if filter_type == 'DSigma':
        stamp[disk] = +1.0 / (pix_area * n_disk)
        stamp[annulus] = -1.0 / (pix_area * n_ann)
    else:  # 'Sigma'
        stamp[annulus] = +1.0 / (pix_area * n_ann)

    kernel = np.zeros((n_pixels, n_pixels), dtype=np.float64)
    idx = np.arange(-half, half + 1) % n_pixels
    kernel[np.ix_(idx, idx)] = stamp
    return kernel


def highpass_field(delta: np.ndarray, pixel_arcmin: float,
                   cut_arcmin: float) -> np.ndarray:
    """Zero every Fourier mode with a wavelength longer than ``cut_arcmin``.

    Used to test how much of a filtered amplitude comes from modes that a
    smaller simulation box cannot represent.  The annulus-mean (``'Sigma'``)
    kernel is uncompensated -- its transform tends to 1 as k tends to 0, since
    ``W_ann(k; R1, R2) = 2[R2 J1(kR2) - R1 J1(kR1)] / (k(R2^2 - R1^2))`` and
    ``J1(x) -> x/2`` -- so ``Y_Sigma`` integrates power down to the box
    fundamental and is not comparable between boxes of different size.  The
    compensated ``'DSigma'`` and ``'Upsilon'`` kernels have transforms that
    vanish at k = 0 and should be insensitive to this cut.

    The k = 0 mode is left at zero, so the output keeps zero mean.

    Args:
        delta (np.ndarray): 2D overdensity map on a square periodic grid.
        pixel_arcmin (float): Angular pixel size in arcmin.
        cut_arcmin (float): Wavelength cut in arcmin.  Modes with
            ``|k| < 2*pi/cut_arcmin`` are removed.

    Returns:
        np.ndarray: High-pass filtered map, same shape, dtype float64.

    Raises:
        ValueError: If ``delta`` is not a square 2D map, if ``cut_arcmin`` is
            not positive, or if it exceeds the map size, in which case it
            falls below the fundamental mode and would remove no power.
    """
    if not cut_arcmin > 0:
        raise ValueError(f"cut_arcmin must be positive, got {cut_arcmin!r}.")
    if delta.ndim != 2 or delta.shape[0] != delta.shape[1]:
        raise ValueError(f"delta must be a square 2D map, got shape "
                         f"{delta.shape}.")
    n = delta.shape[0]
    box_arcmin = n * pixel_arcmin
    if cut_arcmin > box_arcmin:
        # A cut longer than the box sits below the fundamental mode, so it
        # would remove nothing but the (already zero) k=0 bin.  That is
        # always a configuration error rather than a meaningful request.
        raise ValueError(
            f"cut_arcmin={cut_arcmin:.3f}' exceeds the map size "
            f"{box_arcmin:.3f}', so it lies below the fundamental mode and "
            f"would remove no power."
        )

    k_cut = 2.0 * np.pi / cut_arcmin
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=pixel_arcmin)
    ky = 2.0 * np.pi * np.fft.rfftfreq(n, d=pixel_arcmin)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2

    spec = scipy.fft.rfft2(np.asarray(delta, dtype=np.float64),
                           workers=_FFT_WORKERS)
    spec[k2 < k_cut ** 2] = 0.0
    return scipy.fft.irfft2(spec, s=(n, n), workers=_FFT_WORKERS)


def lattice_boundary_margin(edge_arcmin: float, pixel_arcmin: float
                            ) -> Tuple[float, float]:
    """Return how close an aperture edge sits to a realizable lattice shell.

    Pixel membership is decided by the strict test ``r < edge`` on pixel-centre
    radii, so when ``edge / pixel_arcmin`` coincides with an achievable lattice
    distance ``sqrt(i^2 + j^2)`` an entire shell of pixels sits exactly on the
    boundary.  Membership of that shell then flips under an arbitrarily small
    change of convention -- for example between this module's true
    arcmin-per-pixel and the slightly different effective pixel scale of
    ``SimulationStacker.stack_on_array``'s linspace stamp grid.

    This is a discretization degeneracy, not an under-resolution problem: it
    appears at isolated radii rather than growing smoothly as the aperture
    shrinks.  The cached production grids are exactly 0.2 arcmin/pixel, so the
    1 arcmin aperture (5.0 pixels), the 3 arcmin annulus edge (15.0 pixels) and
    the 6 arcmin aperture (30.0 pixels) are all degenerate, each with a
    12-pixel shell on the boundary.

    Because every Y entering an r is filtered with the same kernel, the effect
    largely cancels in the ratio: measured on TNG300-1 it moves the DSigma
    amplitude at R=1' by 6.1 per cent but the coefficient r by only 0.6 per
    cent.

    Args:
        edge_arcmin (float): Aperture edge radius in arcmin (either ``R`` or
            ``R + dr``).
        pixel_arcmin (float): Angular pixel size in arcmin.

    Returns:
        tuple: ``(margin_pixels, shell_pixels)`` where ``margin_pixels`` is the
        distance in pixels from the edge to the nearest realizable lattice
        shell, and ``shell_pixels`` is the number of pixels in that shell.  A
        margin near zero means the aperture is boundary-degenerate.
    """
    t = edge_arcmin / pixel_arcmin
    reach = int(np.ceil(t)) + 1
    lags = np.arange(-reach, reach + 1)
    sq = lags[:, None] ** 2 + lags[None, :] ** 2
    dist = np.sqrt(np.unique(sq))
    idx = int(np.argmin(np.abs(dist - t)))
    nearest = float(dist[idx])
    shell = int(np.count_nonzero(np.isclose(np.sqrt(sq), nearest, rtol=0,
                                            atol=1e-9)))
    return abs(t - nearest), shell


def degenerate_apertures(radii: Sequence[float], pixel_arcmin: float,
                         dr: float = DR_ARCMIN,
                         margin_pixels: float = 1e-3) -> Dict[float, list]:
    """Identify apertures whose disk or annulus edge lands on a lattice shell.

    Args:
        radii (sequence): Aperture radii in arcmin.
        pixel_arcmin (float): Angular pixel size in arcmin.
        dr (float, optional): Annulus width in arcmin.  Defaults to
            :data:`DR_ARCMIN`.
        margin_pixels (float, optional): Edges closer than this (in pixels) to
            a realizable lattice distance count as degenerate.  Defaults to
            1e-3.

    Returns:
        dict: Mapping of aperture radius to a list of
        ``(edge_name, margin_pixels, shell_pixels)`` for each degenerate edge.
        Apertures with no degenerate edge are absent.
    """
    out: Dict[float, list] = {}
    for R in radii:
        flagged = []
        for name, edge in (('disk', float(R)), ('annulus', float(R) + dr)):
            margin, shell = lattice_boundary_margin(edge, pixel_arcmin)
            if margin < margin_pixels:
                flagged.append((name, margin, shell))
        if flagged:
            out[float(R)] = flagged
    return out


def upsilon_defined_mask(radii: Sequence[float], r0: float = R0_ARCMIN,
                         atol: float = 1e-12) -> np.ndarray:
    """Return the aperture bins where ``Upsilon`` carries information.

    ``Upsilon(R; R0) = DSigma(R) - (R0/R)^2 DSigma(R0)`` is the Baldauf et al.
    (2010) estimator, whose whole purpose is to null everything below ``R0``.
    It is therefore meaningful only for ``R > R0``:

    - at ``R = R0`` it vanishes identically, so the coefficient
      ``Y_ab / sqrt(Y_aa Y_bb)`` is a genuine 0/0;
    - below ``R0`` the factor ``(R0/R)^2`` exceeds one and the reference term
      over-subtracts.  The amplitude is finite and the coefficient is defined,
      but it is not the quantity the estimator is about -- in the production
      runs it simply flips sign, giving ``r ~ -1``.

    Neither case is special-cased inside :func:`compute_Y_matrix`, which
    returns the raw algebra; this mask is what the figures, the Gate A metrics
    and the diagnostics use to drop the bins.

    Args:
        radii (sequence): Aperture radii in arcmin.
        r0 (float, optional): Upsilon reference radius in arcmin.  Defaults to
            :data:`R0_ARCMIN`.
        atol (float, optional): Absolute tolerance on the ``R == r0``
            comparison, in arcmin.  Defaults to 1e-12.

    Returns:
        np.ndarray: Boolean mask over ``radii``, True where ``R > r0``.
    """
    r = np.asarray(radii, dtype=np.float64)
    return r > float(r0) + atol


def filtered_map(field: np.ndarray, kernel_spectrum: np.ndarray,
                 shape: Tuple[int, int]) -> np.ndarray:
    """Apply an aperture kernel to a field by periodic FFT correlation.

    Evaluates ``F(x) = sum_l K(l) * field(x + l)`` at every pixel, i.e. the
    correlation (not convolution) that a stamp filter performs when centred
    at ``x``.  The kernel spectrum must already be conjugated -- see
    :func:`kernel_spectrum`.

    Args:
        field (np.ndarray): Real-space field, or its ``rfft2`` if it is
            already complex.
        kernel_spectrum (np.ndarray): Conjugated ``rfft2`` of the kernel, as
            returned by :func:`kernel_spectrum`.
        shape (tuple): Output shape ``(n_pixels, n_pixels)``.

    Returns:
        np.ndarray: Filtered map of shape ``shape``, dtype float64.
    """
    spec = (field if np.iscomplexobj(field)
            else scipy.fft.rfft2(field, workers=_FFT_WORKERS))
    return scipy.fft.irfft2(spec * kernel_spectrum, s=shape,
                            workers=_FFT_WORKERS)


def kernel_spectrum(kernel: np.ndarray) -> np.ndarray:
    """Return the conjugated ``rfft2`` of a kernel, ready for correlation.

    Args:
        kernel (np.ndarray): Real-space kernel from
            :func:`build_aperture_kernel`.

    Returns:
        np.ndarray: ``conj(rfft2(kernel))``, complex128.
    """
    return np.conj(scipy.fft.rfft2(kernel, workers=_FFT_WORKERS))


# ---------------------------------------------------------------------------
# Jackknife bookkeeping
# ---------------------------------------------------------------------------

def block_edges(n_pixels: int, n_side: int = N_JK_SIDE) -> np.ndarray:
    """Return the pixel indices splitting an axis into ``n_side`` blocks.

    Args:
        n_pixels (int): Number of pixels along the axis.
        n_side (int, optional): Number of blocks per side.  Defaults to
            :data:`N_JK_SIDE`.

    Returns:
        np.ndarray: Integer edges of length ``n_side + 1``, starting at 0 and
        ending at ``n_pixels``.  Blocks differ by at most one pixel when
        ``n_pixels`` is not divisible by ``n_side``.
    """
    return np.linspace(0, n_pixels, n_side + 1).astype(int)


def block_sums(product_map: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Sum a product map over each spatial jackknife block.

    Args:
        product_map (np.ndarray): 2D map of ``F_R[delta_X] * delta_Y``.
        edges (np.ndarray): Block edges from :func:`block_edges`.

    Returns:
        np.ndarray: Per-block sums, shape ``(n_side**2,)``, in row-major
        block order.
    """
    n_side = len(edges) - 1
    out = np.empty(n_side * n_side, dtype=np.float64)
    for i in range(n_side):
        for j in range(n_side):
            out[i * n_side + j] = product_map[
                edges[i]:edges[i + 1], edges[j]:edges[j + 1]
            ].sum(dtype=np.float64)
    return out


def block_counts(edges: np.ndarray) -> np.ndarray:
    """Return the pixel count of each spatial jackknife block.

    Args:
        edges (np.ndarray): Block edges from :func:`block_edges`.

    Returns:
        np.ndarray: Per-block pixel counts, shape ``(n_side**2,)``, float64.
    """
    n_side = len(edges) - 1
    widths = np.diff(edges).astype(np.float64)
    return np.outer(widths, widths).ravel()


def jackknife_realizations(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Form leave-one-out means from per-block sums and counts.

    Args:
        sums (np.ndarray): Per-block sums, shape ``(..., n_blocks)``.
        counts (np.ndarray): Per-block pixel counts, shape ``(n_blocks,)``.

    Returns:
        np.ndarray: Leave-one-out means, same shape as ``sums``; entry ``i``
        excludes block ``i``.
    """
    total_sum = sums.sum(axis=-1, keepdims=True)
    total_count = counts.sum()
    return (total_sum - sums) / (total_count - counts)


def jackknife_error(realizations: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the delete-one jackknife standard error.

    Args:
        realizations (np.ndarray): Leave-one-out estimates.
        axis (int, optional): Axis enumerating the realizations.
            Defaults to 0.

    Returns:
        np.ndarray: Jackknife error ``sqrt((M-1)/M * sum (x_i - xbar)^2)``,
        with the realization axis removed.  NaN realizations propagate.
    """
    m = realizations.shape[axis]
    mean = np.mean(realizations, axis=axis, keepdims=True)
    return np.sqrt((m - 1) / m * np.sum((realizations - mean) ** 2, axis=axis))


# ---------------------------------------------------------------------------
# Filtered amplitudes
# ---------------------------------------------------------------------------

def _pair_key(a: str, b: str) -> Tuple[str, str]:
    """Return the canonical (order-independent) key for a field pair.

    ``Y_XY == Y_YX`` for the radially symmetric kernels used here, so pairs
    are stored once under a sorted key.

    Args:
        a (str): First field key.
        b (str): Second field key.

    Returns:
        tuple: ``(a, b)`` sorted alphabetically.
    """
    return (a, b) if a <= b else (b, a)


def compute_Y_matrix(deltas: Dict[str, np.ndarray],
                     pixel_arcmin: float,
                     radii: Sequence[float] = APERTURES_ARCMIN,
                     dr: float = DR_ARCMIN,
                     r0: float = R0_ARCMIN,
                     nbar_pix: Optional[float] = None,
                     galaxy_key: str = 'g',
                     n_jk_side: int = N_JK_SIDE,
                     verbose: bool = False) -> dict:
    """Compute every filtered cross-amplitude and its jackknife realizations.

    For each base filter and aperture, the routine correlates each field with
    the aperture kernel once, multiplies pixelwise by every other field, and
    reduces to per-block sums.  Because ``Y_XY == Y_YX``, only unordered pairs
    are stored.  ``Upsilon`` amplitudes are assembled afterwards as
    ``Y_DSigma(R) - (r0/R)**2 * Y_DSigma(r0)``, per jackknife realization,
    rather than by additional convolutions.

    The galaxy auto-correlation receives the analytic self-pair (shot-noise)
    subtraction ``K(0) / nbar_pix``: an NGP-deposited point process
    contributes a zero-lag delta ``1/nbar_pix`` to the pixel-pair correlation
    that the compensated filter does not remove.  The correction is uniform
    across the map, so it is applied identically to the full-map amplitude
    and to every jackknife realization.  It vanishes automatically for
    ``Sigma``, whose kernel is zero at lag 0.

    Args:
        deltas (dict): Mapping of field key to overdensity map.  All maps must
            be square and share a shape.  Field keys are arbitrary strings;
            the convention used by the Task 1 scripts is ``'g'`` (galaxies),
            ``'e'`` (ionized gas / electrons), ``'b'`` (all baryons) and
            ``'m'`` (CDM).
        pixel_arcmin (float): Angular size of one pixel, in arcmin.
        radii (sequence, optional): Aperture radii in arcmin.  Defaults to
            :data:`APERTURES_ARCMIN`.
        dr (float, optional): Annulus width in arcmin.  Defaults to
            :data:`DR_ARCMIN`.
        r0 (float, optional): Upsilon reference radius in arcmin.  Defaults
            to :data:`R0_ARCMIN`.
        nbar_pix (float, optional): Mean galaxy count per pixel, used for the
            self-pair subtraction.  Required if ``galaxy_key`` is present in
            ``deltas``.  Defaults to None.
        galaxy_key (str, optional): Key of the discrete (point-process) field
            in ``deltas``.  Defaults to ``'g'``.
        n_jk_side (int, optional): Jackknife blocks per side.  Defaults to
            :data:`N_JK_SIDE`.
        verbose (bool, optional): If True, print progress per aperture.
            Defaults to False.

    Returns:
        dict: With keys

        - ``'radii'``: the aperture radii, shape ``(n_radii,)``.
        - ``'pixel_arcmin'``, ``'dr'``, ``'r0'``, ``'n_jk'``: the settings used.
        - ``'fields'``: sorted list of field keys.
        - ``'Y'``: ``Y[filter][(a, b)]`` full-map amplitudes, shape
          ``(n_radii,)``.
        - ``'Y_jk'``: ``Y_jk[filter][(a, b)]`` leave-one-out amplitudes,
          shape ``(n_jk, n_radii)``.

    Raises:
        ValueError: If the maps are not square, do not share a shape, or if
            the galaxy field is present without ``nbar_pix``.
    """
    keys = sorted(deltas)
    if not keys:
        raise ValueError("deltas is empty; nothing to correlate.")

    shape = deltas[keys[0]].shape
    n_pixels = shape[0]
    if shape[0] != shape[1]:
        raise ValueError(f"Maps must be square, got shape {shape}.")
    for k in keys:
        if deltas[k].shape != shape:
            raise ValueError(
                f"Field {k!r} has shape {deltas[k].shape}, expected {shape}; "
                "all fields must share one grid."
            )
    if galaxy_key in deltas and nbar_pix is None:
        raise ValueError(
            f"Field {galaxy_key!r} is a discrete point process; nbar_pix "
            "(mean galaxies per pixel) is required for the self-pair "
            "subtraction."
        )

    radii = np.asarray(radii, dtype=np.float64)
    edges = block_edges(n_pixels, n_jk_side)
    counts = block_counts(edges)
    n_jk = len(counts)

    # DSigma is additionally needed at r0 to build Upsilon.  Evaluate it once
    # by appending r0 to the DSigma radius list when it is not already there.
    ds_radii = list(radii)
    r0_index = next((i for i, r in enumerate(ds_radii)
                     if np.isclose(r, r0, rtol=0, atol=1e-12)), None)
    if r0_index is None:
        ds_radii.append(float(r0))
        r0_index = len(ds_radii) - 1

    pairs = [(_pair_key(a, b))
             for i, a in enumerate(keys) for b in keys[i:]]

    # Per-filter, per-pair block sums: shape (n_radii_for_filter, n_jk).
    raw: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {}
    filter_radii = {'Sigma': list(radii), 'DSigma': ds_radii}

    spectra = {k: scipy.fft.rfft2(deltas[k], workers=_FFT_WORKERS)
               for k in keys}

    for filt in BASE_FILTERS:
        rad_list = filter_radii[filt]
        raw[filt] = {p: np.empty((len(rad_list), n_jk), dtype=np.float64)
                     for p in pairs}
        for ir, R in enumerate(rad_list):
            kern = build_aperture_kernel(n_pixels, pixel_arcmin, R, filt, dr)
            k_zero = float(kern[0, 0])
            kspec = kernel_spectrum(kern)
            del kern

            for ia, a in enumerate(keys):
                fmap = filtered_map(spectra[a], kspec, shape)
                for b in keys[ia:]:
                    sums = block_sums(fmap * deltas[b], edges)
                    if a == galaxy_key and b == galaxy_key:
                        # Self-pairs: uniform per-pixel shot-noise term.
                        sums = sums - k_zero / nbar_pix * counts
                    raw[filt][_pair_key(a, b)][ir] = sums
                del fmap
            del kspec
            if verbose:
                print(f"      {filt} R={R:.3f}' done")

    # Reduce to full-map amplitudes and jackknife realizations.
    Y: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {}
    Y_jk: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {}
    for filt in BASE_FILTERS:
        Y[filt] = {}
        Y_jk[filt] = {}
        for p in pairs:
            sums = raw[filt][p]                       # (n_rad, n_jk)
            Y[filt][p] = sums.sum(axis=1) / counts.sum()
            Y_jk[filt][p] = jackknife_realizations(sums, counts).T  # (n_jk, n_rad)

    # Upsilon: linear combination of DSigma amplitudes, per realization.
    Y['Upsilon'] = {}
    Y_jk['Upsilon'] = {}
    scale = (r0 / radii) ** 2
    n_rad = len(radii)
    for p in pairs:
        ds_full = Y['DSigma'][p]
        ds_jk = Y_jk['DSigma'][p]
        Y['Upsilon'][p] = ds_full[:n_rad] - scale * ds_full[r0_index]
        Y_jk['Upsilon'][p] = (ds_jk[:, :n_rad]
                              - scale[None, :] * ds_jk[:, r0_index][:, None])

    # Trim the DSigma arrays back to the requested radii (drop the extra r0).
    for p in pairs:
        Y['DSigma'][p] = Y['DSigma'][p][:n_rad]
        Y_jk['DSigma'][p] = Y_jk['DSigma'][p][:, :n_rad]

    return {
        'radii': radii,
        'pixel_arcmin': float(pixel_arcmin),
        'dr': float(dr),
        'r0': float(r0),
        'n_jk': n_jk,
        'fields': keys,
        'Y': Y,
        'Y_jk': Y_jk,
    }


def get_Y(Ymat: dict, filt: str, a: str, b: str, jackknife: bool = False
          ) -> np.ndarray:
    """Look up a filtered amplitude regardless of field order.

    Args:
        Ymat (dict): Output of :func:`compute_Y_matrix`.
        filt (str): Filter name, one of :data:`FILTERS`.
        a (str): First field key.
        b (str): Second field key.
        jackknife (bool, optional): If True, return the leave-one-out
            realizations instead of the full-map amplitude.  Defaults to
            False.

    Returns:
        np.ndarray: Amplitude of shape ``(n_radii,)``, or ``(n_jk, n_radii)``
        if ``jackknife`` is True.
    """
    store = Ymat['Y_jk'] if jackknife else Ymat['Y']
    return store[filt][_pair_key(a, b)]


# ---------------------------------------------------------------------------
# Cross-correlation coefficients
# ---------------------------------------------------------------------------

def _coefficient(Y_ab: np.ndarray, Y_aa: np.ndarray, Y_bb: np.ndarray
                 ) -> np.ndarray:
    """Return ``Y_ab / sqrt(Y_aa * Y_bb)``, NaN where the denominator is invalid.

    Args:
        Y_ab (np.ndarray): Cross amplitude.
        Y_aa (np.ndarray): First auto amplitude.
        Y_bb (np.ndarray): Second auto amplitude.

    Returns:
        np.ndarray: Cross-correlation coefficient, same shape as ``Y_ab``.
        Entries where ``Y_aa * Y_bb <= 0`` (possible for compensated filters
        in the noise-dominated regime) are NaN rather than an error.
    """
    denom_sq = Y_aa * Y_bb
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(denom_sq > 0.0, Y_ab / np.sqrt(denom_sq), np.nan)
    return out


def r_profiles(Ymat: dict,
               pairs: Sequence[Tuple[str, str]] = (('g', 'b'), ('b', 'm'),
                                                   ('g', 'e'), ('e', 'm')),
               ratios: Sequence[Tuple[Tuple[str, str], Tuple[str, str]]] = (
                   (('b', 'm'), ('g', 'b')),
                   (('e', 'm'), ('g', 'e')),
               )) -> dict:
    """Form cross-correlation coefficients and their ratios with jackknife errors.

    Every coefficient and every ratio is formed PER jackknife realization
    before the spread is taken.  This is essential: the three amplitudes
    entering one r are measured on the same map and are strongly correlated,
    so propagating marginal Y errors Gaussianly would badly misestimate the
    uncertainty (and would miss the partial cancellation that makes the ratio
    ``r_bm/r_gb`` the well-behaved object of the study).

    Args:
        Ymat (dict): Output of :func:`compute_Y_matrix`.
        pairs (sequence, optional): Field pairs whose coefficients to report.
            Defaults to the four Task 1 coefficients ``r_gb``, ``r_bm``,
            ``r_ge`` and ``r_em``.
        ratios (sequence, optional): Pairs of pairs ``(numerator,
            denominator)`` whose coefficient ratio to report.  Defaults to
            ``r_bm/r_gb`` and its electron analogue ``r_em/r_ge``.

    Returns:
        dict: With keys

        - ``'radii'``: aperture radii, shape ``(n_radii,)``.
        - ``'r'``: ``r[(a, b)][filter]``, shape ``(n_radii,)``.
        - ``'r_err'``: jackknife errors, same shape.
        - ``'r_jk'``: ``r_jk[(a, b)][filter]``, shape ``(n_jk, n_radii)``.
        - ``'ratio'``, ``'ratio_err'``, ``'ratio_jk'``: same layout, keyed by
          ``(numerator_pair, denominator_pair)``.
    """
    out_r: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    out_err: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    out_jk: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    available = set(Ymat['fields'])
    for (a, b) in pairs:
        if a not in available or b not in available:
            continue
        out_r[(a, b)] = {}
        out_err[(a, b)] = {}
        out_jk[(a, b)] = {}
        for filt in FILTERS:
            full = _coefficient(get_Y(Ymat, filt, a, b),
                                get_Y(Ymat, filt, a, a),
                                get_Y(Ymat, filt, b, b))
            jk = _coefficient(get_Y(Ymat, filt, a, b, jackknife=True),
                              get_Y(Ymat, filt, a, a, jackknife=True),
                              get_Y(Ymat, filt, b, b, jackknife=True))
            out_r[(a, b)][filt] = full
            out_jk[(a, b)][filt] = jk
            out_err[(a, b)][filt] = jackknife_error(jk, axis=0)

    out_ratio: Dict[Tuple, Dict[str, np.ndarray]] = {}
    out_ratio_err: Dict[Tuple, Dict[str, np.ndarray]] = {}
    out_ratio_jk: Dict[Tuple, Dict[str, np.ndarray]] = {}
    for num, den in ratios:
        if num not in out_r or den not in out_r:
            continue
        key = (num, den)
        out_ratio[key] = {}
        out_ratio_err[key] = {}
        out_ratio_jk[key] = {}
        for filt in FILTERS:
            with np.errstate(invalid='ignore', divide='ignore'):
                full = out_r[num][filt] / out_r[den][filt]
                jk = out_jk[num][filt] / out_jk[den][filt]
            out_ratio[key][filt] = full
            out_ratio_jk[key][filt] = jk
            out_ratio_err[key][filt] = jackknife_error(jk, axis=0)

    return {
        'radii': Ymat['radii'],
        'r': out_r,
        'r_err': out_err,
        'r_jk': out_jk,
        'ratio': out_ratio,
        'ratio_err': out_ratio_err,
        'ratio_jk': out_ratio_jk,
    }


# ---------------------------------------------------------------------------
# Galaxy field construction
# ---------------------------------------------------------------------------

def project_positions(positions: np.ndarray, projection: str) -> np.ndarray:
    """Select the two transverse coordinates for a projection.

    Uses exactly the axis convention of ``SimulationStacker.stack_on_array``
    and ``mapMaker.make_mass_field``, so galaxy maps land on the same grid as
    the particle maps.

    Args:
        positions (np.ndarray): Positions of shape ``(N, 3)``.
        projection (str): One of ``'xy'``, ``'xz'``, ``'yz'``.

    Returns:
        np.ndarray: Transverse positions of shape ``(N, 2)``.

    Raises:
        NotImplementedError: If the projection is not recognised.
    """
    if projection == 'xy':
        return positions[:, :2]
    if projection == 'xz':
        return positions[:, [0, 2]]
    if projection == 'yz':
        return positions[:, 1:]
    raise NotImplementedError('Projection type not implemented: ' + projection)


def select_sham_subhalos(stacker, target_number: float,
                         parent_mass_upper: Optional[float] = 5e14,
                         subhalos: Optional[dict] = None,
                         parents: Optional[dict] = None) -> np.ndarray:
    """Select a SHAM galaxy sample, matching the existing stacking pipeline.

    Replicates the ``use_subhalos=True`` branch of
    ``SimulationStacker.stack_on_array`` exactly: optionally restrict to
    subhalos whose parent FoF halo is below ``parent_mass_upper``, then rank
    the survivors by stellar mass (``SubhaloMStar``) and keep the top N
    matching the target number density (Reddick et al. 2013 SHAM).

    The parent-mass pre-filter is part of the pipeline default
    (``stackMap(halo_mass_upper=5e14)``), so the f_gas paper's samples carry
    it; reproducing it here is what makes the integration test's "identical
    SHAM sample" true.

    Args:
        stacker (SimulationStacker): Provides the catalogues and the header.
        target_number (float): Target number density in ``(cMpc/h)^-3``.
        parent_mass_upper (float, optional): Upper bound on the parent FoF
            mass in Msun/h.  Pass None to disable the pre-filter.  Defaults
            to 5e14.
        subhalos (dict, optional): Pre-loaded subhalo catalogue, to avoid a
            second expensive read.  Defaults to None (loaded internally).
        parents (dict, optional): Pre-loaded halo catalogue, likewise.  Only
            consulted when ``parent_mass_upper`` is not None.  Defaults to
            None (loaded internally).

    Returns:
        np.ndarray: Integer indices into the subhalo catalogue, sorted by
        decreasing stellar mass.
    """
    if subhalos is None:
        subhalos = stacker.loadSubHalos()
    mstar = subhalos['SubhaloMStar']

    if parent_mass_upper is not None:
        if parents is None:
            parents = stacker.loadHalos()
        parent_mass = parents['GroupMass'][subhalos['SubhaloGrNr']]
        valid = np.where(parent_mass <= parent_mass_upper)[0]
        local = select_halos(mstar[valid], 'abundance',
                             target_number=target_number,
                             Lbox=stacker.header['BoxSize'])
        return valid[local]

    return select_halos(mstar, 'abundance',
                        target_number=target_number,
                        Lbox=stacker.header['BoxSize'])


def make_galaxy_field(stacker, projection: str, n_pixels: int,
                      target_number: float,
                      parent_mass_upper: Optional[float] = 5e14,
                      subhalos: Optional[dict] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Build an NGP galaxy count map from the SHAM-selected subhalo sample.

    Deposition is nearest-grid-point via :func:`tools.hist2d_numba_seq` on the
    same pixel edges as the particle maps (``[0, BoxSize)`` split into
    ``n_pixels`` bins).  NGP rather than TSC is deliberate: the point-process
    shot noise is subtracted analytically in :func:`compute_Y_matrix`, and a
    smoothing window would alias into the aperture kernel.

    Args:
        stacker (SimulationStacker): Provides the catalogues and the header.
        projection (str): One of ``'xy'``, ``'xz'``, ``'yz'``.
        n_pixels (int): Number of pixels per side, matching the particle maps.
        target_number (float): SHAM target number density in ``(cMpc/h)^-3``.
        parent_mass_upper (float, optional): Parent FoF mass bound in Msun/h.
            Defaults to 5e14.
        subhalos (dict, optional): Pre-loaded subhalo catalogue.  Defaults to
            None.

    Returns:
        tuple: ``(count_map, halo_mask)`` where ``count_map`` is the float64
        galaxy count per pixel, shape ``(n_pixels, n_pixels)``, and
        ``halo_mask`` is the integer index array of the selected subhalos
        (needed to stack the identical sample through the legacy stamp route).

    Raises:
        ValueError: If the SHAM selection is empty, which would make the
            galaxy overdensity undefined.
    """
    if subhalos is None:
        subhalos = stacker.loadSubHalos()
    halo_mask = select_sham_subhalos(
        stacker, target_number, parent_mass_upper=parent_mass_upper,
        subhalos=subhalos)

    if halo_mask.size == 0:
        raise ValueError(
            f"SHAM selection is empty for target_number={target_number!r} "
            f"in a box of {stacker.header['BoxSize']:.1f} ckpc/h."
        )

    pos2d = project_positions(subhalos['SubhaloPos'][halo_mask], projection)
    lbox = float(stacker.header['BoxSize'])

    tracks = np.ascontiguousarray(
        np.array([pos2d[:, 0], pos2d[:, 1]], dtype=np.float64))
    bins = np.array([n_pixels, n_pixels], dtype=np.int64)
    ranges = np.array([[0.0, lbox], [0.0, lbox]], dtype=np.float64)
    # Explicit unit weights: hist2d_numba_seq only broadcasts unit weight when
    # len(weights) == 1, and indexes weights[t] otherwise, so a length-N array
    # of ones is the unambiguous way to request counts.
    weights = np.ones(tracks.shape[1], dtype=np.float64)

    count_map = hist2d_numba_seq(tracks, bins, ranges, weights=weights)
    return count_map, halo_mask
