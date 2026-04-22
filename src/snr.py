"""snr.py — Detection SNR utilities for kSZ, lensing, and combined fgas measurements.

Pipeline-specific constants:
    - Lensing (dsigma): jackknife, N_jk = 100, Hartlap correction applies
      (alpha = 89/99 ≈ 0.899 for p = 9 bins).
    - kSZ (ThumbStack): bootstrap, N_boot = 10000, Hartlap correction
      negligible (alpha > 0.999) and not applied.
"""

import numpy as np
from scipy import stats


# ── Pipeline constants ────────────────────────────────────────────────────────

N_JK_LENS  = 100     # dsigma jackknife fields (compute_jackknife_fields(..., 100))
N_BOOT_KSZ = 10_000  # ThumbStack bootstrap samples (self.nSamples = 10000)


# ── Core utilities ────────────────────────────────────────────────────────────

def hartlap_factor(n_resample: int, n_bins: int) -> float:
    """Compute the Hartlap et al. (2007) bias-correction factor for a sample covariance precision matrix.

    The inverse of a sample covariance matrix estimated from N resamplings is
    a biased estimator of the precision matrix. The Hartlap factor corrects
    this multiplicatively: C^{-1}_unbiased = alpha * C^{-1}_sample.

    Valid for both jackknife and bootstrap estimators, though the correction
    is only practically significant when N is small relative to p.

    Reference: Hartlap, Simon & Schneider 2007, A&A 464, 399.

    Args:
        n_resample: Number of resamplings (jackknife fields or bootstrap draws)
            used to estimate the covariance matrix.
        n_bins: Number of data points (radial bins) in the data vector.

    Returns:
        Scalar alpha in (0, 1].

    Raises:
        ValueError: If n_resample <= n_bins + 2, making the correction
            ill-defined (precision matrix estimate is singular or negative).
    """
    if n_resample <= n_bins + 2:
        raise ValueError(
            f"Hartlap correction requires n_resample > n_bins + 2, "
            f"got n_resample={n_resample}, n_bins={n_bins}."
        )
    return (n_resample - n_bins - 2) / (n_resample - 1)


def apply_hartlap(cov: np.ndarray, n_resample: int) -> np.ndarray:
    """Return a rescaled covariance matrix whose inverse equals the Hartlap-corrected precision matrix.

    Dividing the covariance by alpha is equivalent to multiplying the precision
    matrix by alpha, but avoids inverting the matrix twice. Use this form when
    passing a corrected covariance to a downstream propagation step rather than
    inverting it directly.

    For the kSZ bootstrap covariance (n_resample = 10000, n_bins = 9),
    alpha > 0.999 and the correction is negligible. For the lensing jackknife
    covariance (n_resample = 100, n_bins = 9), alpha ≈ 0.899 and the
    correction reduces the SNR by approximately 5%.

    Args:
        cov: Sample covariance matrix of shape (n, n).
        n_resample: Number of resamplings used to estimate ``cov``.

    Returns:
        Rescaled covariance matrix of shape (n, n).
    """
    alpha = hartlap_factor(n_resample, cov.shape[0])
    return cov / alpha


def detection_snr(
    data: np.ndarray,
    cov: np.ndarray,
    null: float = 0.0,
) -> float:
    """Compute detection SNR as sqrt((d - null)^T C^{-1} (d - null)).

    This is the global matched-filter statistic. When null=0.0 it tests
    detection vs. zero signal; when null=1.0 it tests whether fgas is
    consistent with the no-feedback hypothesis.

    The caller is responsible for passing a Hartlap-corrected covariance
    (via apply_hartlap) when appropriate. This function simply inverts
    whatever covariance it receives.

    Args:
        data: Measured data vector of shape (n,).
        cov: Covariance matrix of shape (n, n). Should already be
            Hartlap-corrected if applicable.
        null: Value subtracted from ``data`` before computing the statistic.
            Use 0.0 for kSZ and lensing detection; 1.0 for fgas vs. the
            no-feedback null hypothesis.

    Returns:
        Detection significance in units of sigma.
    """
    delta = data - null
    prec = np.linalg.inv(cov)
    return float(np.sqrt(delta @ prec @ delta))

    


def null_test_pte(
    data: np.ndarray,
    cov: np.ndarray,
) -> dict[str, float]:
    """Perform a null test on a data vector against the zero model.

    Computes chi-squared = d^T C^{-1} d and the corresponding PTE
    (Probability To Exceed) under the null hypothesis that the true
    signal is zero. The degrees of freedom equals the number of data
    points since the null model has no free parameters.

    Args:
        data: 1D array of measured values of length N.
        cov: 2D covariance matrix of shape (N, N).

    Returns:
        Dictionary with keys:
            - 'chi2': Observed chi-squared value.
            - 'dof': Degrees of freedom (= N).
            - 'chi2_per_dof': Reduced chi-squared (should be ~1 for a good null).
            - 'pte': Probability To Exceed. Values near 0.5 indicate
              consistency with zero; values near 0 or 1 are suspicious.
    """
    cov_inv = np.linalg.inv(cov)
    chi2 = float(data @ cov_inv @ data)
    dof = len(data)

    return {
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
        "pte": stats.chi2.sf(chi2, df=dof), # type: ignore
    }