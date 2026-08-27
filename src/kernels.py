"""Analytic harmonic-space aperture kernels for the filtered amplitudes.

Implements Sec. 5.3 of ``docs/cross_correlation_notes.md``: every filter used
in this programme is linear, so each filtered amplitude is a single integral
of the projected power spectrum against an analytic kernel,

    Y(R) = int (k dk / 2 pi) P_2D(k) W(k; R).

The kernels are

    W_disk(k; R)      = 2 J1(kR) / (kR)
    W_ann(k; R1, R2)  = 2 [R2 J1(kR2) - R1 J1(kR1)] / (k (R2^2 - R1^2))
    W_DSigma(k; R)    = W_disk(k; R) - W_ann(k; R, R + dr)
    W_Upsilon(k; R)   = W_DSigma(k; R) - (R0/R)^2 W_DSigma(k; R0)

Both W_disk and W_ann tend to 1 as k tends to 0 (since J1(x) -> x/2), so the
compensated combinations vanish there.  That is the formal statement of why
DSigma and Upsilon are insensitive to the largest modes in a box while the
annulus mean is not -- the measurement behind the Gate B filter freeze, see
``scripts/cross_corr/check_filter_compensation.py``.

Normalization matters when comparing against the pipeline's measured
amplitudes.  ``filters.delta_sigma_kernel``, which ``rprofiles`` reproduces,
normalizes its pixel weights by ``1/(pixArea * N)`` rather than ``1/N``, so a
measured Y carries an extra factor of ``1/pixArea`` relative to the integral
above.  This cancels in every cross-correlation coefficient, but must be
restored explicitly when validating theory against measurement --
:func:`normalize_to_pipeline` does that, and
``docs/filter_specification.md`` records the convention.

All radii and wavenumbers must share reciprocal units: pass ``k`` in
``1/arcmin`` with radii in arcmin, or ``k`` in ``h/Mpc`` with radii in Mpc/h.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import j1

#: Below this value of ``x = kR`` the series expansion of ``2 J1(x)/x`` is used
#: instead of the direct ratio, which loses precision as x -> 0.
_SMALL_X = 1e-4


def _two_j1_over_x(x: np.ndarray) -> np.ndarray:
    """Return ``2 J1(x) / x``, stable at and near ``x = 0``.

    The direct ratio is 0/0 at the origin and cancels badly just above it.
    For small argument the series ``2 J1(x)/x = 1 - x^2/8 + x^4/192 - ...``
    is used instead.

    Args:
        x (np.ndarray): Argument, any shape, non-negative.

    Returns:
        np.ndarray: ``2 J1(x)/x``, equal to 1 at ``x = 0``.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = np.abs(x) < _SMALL_X
    xs = x[small]
    out[small] = 1.0 - xs ** 2 / 8.0 + xs ** 4 / 192.0
    xb = x[~small]
    out[~small] = 2.0 * j1(xb) / xb
    return out


def w_disk(k: np.ndarray, R: float) -> np.ndarray:
    """Transform of the normalized disk mean of radius ``R``.

    Args:
        k (np.ndarray): Wavenumbers, reciprocal to the units of ``R``.
        R (float): Disk radius.

    Returns:
        np.ndarray: ``2 J1(kR)/(kR)``, which tends to 1 as k tends to 0.

    Raises:
        ValueError: If ``R`` is not positive.
    """
    if not R > 0:
        raise ValueError(f"R must be positive, got {R!r}.")
    return _two_j1_over_x(np.asarray(k, dtype=np.float64) * R)


def w_annulus(k: np.ndarray, R1: float, R2: float) -> np.ndarray:
    """Transform of the normalized annulus mean over ``[R1, R2]``.

    Expressed as an area-weighted difference of two disk means, which is
    numerically better behaved near ``k = 0`` than the direct Bessel
    difference:

        W_ann = [R2^2 W_disk(k; R2) - R1^2 W_disk(k; R1)] / (R2^2 - R1^2).

    Args:
        k (np.ndarray): Wavenumbers, reciprocal to the units of the radii.
        R1 (float): Inner radius.
        R2 (float): Outer radius, strictly greater than ``R1``.

    Returns:
        np.ndarray: Kernel transform, tending to 1 as k tends to 0.

    Raises:
        ValueError: If the radii are not ``0 <= R1 < R2``.
    """
    if not (R2 > R1 >= 0):
        raise ValueError(f"Require 0 <= R1 < R2, got R1={R1!r}, R2={R2!r}.")
    k = np.asarray(k, dtype=np.float64)
    outer = R2 ** 2 * _two_j1_over_x(k * R2)
    inner = (R1 ** 2 * _two_j1_over_x(k * R1) if R1 > 0
             else np.zeros_like(outer))
    return (outer - inner) / (R2 ** 2 - R1 ** 2)


def w_sigma(k: np.ndarray, R: float, dr: float) -> np.ndarray:
    """Transform of the annulus-mean ("Sigma") filter over ``[R, R+dr]``.

    Retained as a diagnostic only.  This kernel is uncompensated -- it tends
    to 1 rather than 0 at ``k = 0`` -- so the amplitude it produces integrates
    power down to the fundamental mode of whatever box or survey volume it is
    measured in, and is not comparable between volumes of different size.

    Args:
        k (np.ndarray): Wavenumbers.
        R (float): Inner radius of the annulus.
        dr (float): Annulus width.

    Returns:
        np.ndarray: Kernel transform.
    """
    return w_annulus(k, R, R + dr)


def w_dsigma(k: np.ndarray, R: float, dr: float) -> np.ndarray:
    """Transform of the compensated disk-minus-annulus filter.

    Args:
        k (np.ndarray): Wavenumbers.
        R (float): Aperture radius.
        dr (float): Width of the compensating annulus, ``[R, R+dr]``.

    Returns:
        np.ndarray: ``W_disk(k; R) - W_ann(k; R, R+dr)``, vanishing at
        ``k = 0``.
    """
    return w_disk(k, R) - w_annulus(k, R, R + dr)


def w_upsilon(k: np.ndarray, R: float, dr: float, r0: float) -> np.ndarray:
    """Transform of the Baldauf et al. (2010) filter with reference radius r0.

    Args:
        k (np.ndarray): Wavenumbers.
        R (float): Aperture radius.
        dr (float): Annulus width.
        r0 (float): Reference radius below which information is nulled.

    Returns:
        np.ndarray: ``W_DSigma(k; R) - (r0/R)^2 W_DSigma(k; r0)``.  Identically
        zero when ``R == r0``, which is why the coefficient is undefined in
        that aperture bin.

    Raises:
        ValueError: If ``R`` or ``r0`` is not positive.
    """
    if not r0 > 0:
        raise ValueError(f"r0 must be positive, got {r0!r}.")
    return w_dsigma(k, R, dr) - (r0 / R) ** 2 * w_dsigma(k, r0, dr)


def aperture_kernel_ft(k: np.ndarray, R: float, filter_type: str,
                       dr: float = 0.75, r0: float = 1.0) -> np.ndarray:
    """Dispatch to the transform of the requested aperture filter.

    Args:
        k (np.ndarray): Wavenumbers, reciprocal to the units of the radii.
        R (float): Aperture radius.
        filter_type (str): One of ``'Sigma'``, ``'DSigma'``, ``'Upsilon'``.
        dr (float, optional): Annulus width.  Defaults to 0.75.
        r0 (float, optional): Upsilon reference radius.  Defaults to 1.0.

    Returns:
        np.ndarray: Kernel transform evaluated at ``k``.

    Raises:
        ValueError: If ``filter_type`` is not recognised.
    """
    if filter_type == 'Sigma':
        return w_sigma(k, R, dr)
    if filter_type == 'DSigma':
        return w_dsigma(k, R, dr)
    if filter_type == 'Upsilon':
        return w_upsilon(k, R, dr, r0)
    raise ValueError(
        f"filter_type must be 'Sigma', 'DSigma' or 'Upsilon', got "
        f"{filter_type!r}."
    )


def bin_averaged_kernel_ft(k: np.ndarray, R_lo: float, R_hi: float,
                           filter_type: str, dr: float = 0.75,
                           r0: float = 1.0, n_sub: int = 32) -> np.ndarray:
    """Average a kernel transform over an aperture bin, weighting by area.

    The theory note (Sec. 5.3) asks for bin averaging to be applied to the
    kernels rather than approximated at bin centres.  Note that the pipeline
    as built evaluates every filter at a *point* radius rather than over a
    bin, in both simulation and theory, so this routine is not needed for
    internal consistency; it exists for the data chain, where an aperture bin
    of finite width may be unavoidable.

    Args:
        k (np.ndarray): Wavenumbers.
        R_lo (float): Lower edge of the aperture bin.
        R_hi (float): Upper edge of the aperture bin.
        filter_type (str): Filter name.
        dr (float, optional): Annulus width.  Defaults to 0.75.
        r0 (float, optional): Upsilon reference radius.  Defaults to 1.0.
        n_sub (int, optional): Number of sub-radii in the quadrature.
            Defaults to 32.

    Returns:
        np.ndarray: Area-weighted mean of the kernel transform over the bin.

    Raises:
        ValueError: If the bin edges are not ``0 < R_lo < R_hi``.
    """
    if not (R_hi > R_lo > 0):
        raise ValueError(
            f"Require 0 < R_lo < R_hi, got R_lo={R_lo!r}, R_hi={R_hi!r}.")
    edges = np.linspace(R_lo, R_hi, n_sub + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    # Area weighting: annuli of equal radial width carry area proportional to
    # their mean radius, which is the natural weight for a stacked aperture.
    weights = mids * np.diff(edges)
    weights /= weights.sum()
    stack = np.stack([aperture_kernel_ft(k, float(R), filter_type, dr, r0)
                      for R in mids])
    return np.tensordot(weights, stack, axes=(0, 0))


def normalize_to_pipeline(y_continuum: np.ndarray,
                          pixel_size: float) -> np.ndarray:
    """Convert a continuum amplitude to the pipeline's pixel normalization.

    ``filters.delta_sigma_kernel`` weights pixels by ``1/(pixArea * N)`` where
    a pure mean would use ``1/N``, so every amplitude the pipeline reports is
    larger than the continuum integral by ``1/pixArea``.  The factor cancels in
    every coefficient ``r = Y_XY / sqrt(Y_XX Y_YY)`` and so never mattered for
    Tasks 1 to 3, but it must be applied to compare a theory amplitude against
    a measured one.

    Args:
        y_continuum (np.ndarray): Amplitude from the ``int k dk/(2 pi)``
            integral, dimensionless.
        pixel_size (float): Linear pixel size, in the units the measurement
            used for its ``pixel_size`` argument (arcmin in this pipeline).

    Returns:
        np.ndarray: Amplitude on the pipeline's normalization.

    Raises:
        ValueError: If ``pixel_size`` is not positive.
    """
    if not pixel_size > 0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size!r}.")
    return np.asarray(y_continuum, dtype=np.float64) / pixel_size ** 2


def filtered_amplitude(k: np.ndarray, p_2d: np.ndarray, R: float,
                       filter_type: str, dr: float = 0.75, r0: float = 1.0,
                       pixel_size: Optional[float] = None) -> float:
    """Integrate a projected power spectrum against an aperture kernel.

    Evaluates ``Y(R) = int (k dk / 2 pi) P_2D(k) W(k; R)`` by trapezoidal
    quadrature on the supplied grid.  The grid must be fine enough to resolve
    the kernel's oscillations, which have period ``2 pi / R`` in k, and must
    extend well past ``k ~ 1/R``; :func:`recommended_k_grid` builds one.

    Args:
        k (np.ndarray): Wavenumbers, strictly increasing.
        p_2d (np.ndarray): Projected power spectrum at those wavenumbers, in
            units reciprocal to ``k`` squared.
        R (float): Aperture radius, in units reciprocal to ``k``.
        filter_type (str): Filter name.
        dr (float, optional): Annulus width.  Defaults to 0.75.
        r0 (float, optional): Upsilon reference radius.  Defaults to 1.0.
        pixel_size (float, optional): If given, the result is converted to the
            pipeline's pixel normalization via :func:`normalize_to_pipeline`.
            Defaults to None, which returns the continuum amplitude.

    Returns:
        float: The filtered amplitude.
    """
    k = np.asarray(k, dtype=np.float64)
    w = aperture_kernel_ft(k, R, filter_type, dr, r0)
    integrand = k * np.asarray(p_2d, dtype=np.float64) * w / (2.0 * np.pi)
    y = float(np.trapz(integrand, k))
    if pixel_size is not None:
        return float(normalize_to_pipeline(y, pixel_size))
    return y


def recommended_k_grid(R_min: float, R_max: float, n_per_period: int = 64,
                       k_min_factor: float = 1e-3,
                       k_max_factor: float = 200.0) -> np.ndarray:
    """Build a k grid fine and wide enough for the aperture kernels.

    The kernels oscillate with period ``2 pi / R`` in k, so the largest
    aperture sets the required sampling and the smallest sets how far the grid
    must extend.

    Args:
        R_min (float): Smallest aperture radius.
        R_max (float): Largest aperture radius.
        n_per_period (int, optional): Samples per oscillation period of the
            largest aperture.  Defaults to 64.
        k_min_factor (float, optional): Lower limit as a fraction of
            ``1/R_max``.  Defaults to 1e-3.
        k_max_factor (float, optional): Upper limit as a multiple of
            ``1/R_min``.  Defaults to 200.

    Returns:
        np.ndarray: Linearly spaced wavenumbers.  Linear rather than
        logarithmic spacing is used because the integrand oscillates on a
        fixed period in k, which log spacing under-samples at high k.

    Raises:
        ValueError: If the radii are not ``0 < R_min <= R_max``.
    """
    if not (R_max >= R_min > 0):
        raise ValueError(
            f"Require 0 < R_min <= R_max, got {R_min!r}, {R_max!r}.")
    k_min = k_min_factor / R_max
    k_max = k_max_factor / R_min
    n = int(np.ceil((k_max - k_min) * R_max * n_per_period / (2.0 * np.pi)))
    return np.linspace(k_min, k_max, max(n, 4096))
