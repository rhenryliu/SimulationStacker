"""Theory-side filtered amplitudes: the P(k) -> Y(R) chain.

Implements the harmonic-space route of ``docs/cross_correlation_notes.md``
Sec. 5.3 for Task 4: a non-linear matter power spectrum from halofit, projected
to two dimensions, integrated against the analytic aperture kernels of
:mod:`kernels`.

Projection convention (see ``docs/filter_specification.md``): for validating
against a periodic simulation box the projection is the full box depth, for
which

    P_2D(k_perp) = P_3D(k_perp) / L,

with ``L`` the box side.  This follows from ``delta_2D = (1/L) int dz
delta_3D``, which selects the ``k_z = 0`` mode.  It is why a measured Y scales
as ``1/L`` and is not, on its own, a physical observable -- the amplitudes of
TNG300-1 and FLAMINGO differ by very nearly their box ratio.  The data chain
uses a different projection (the Pi_max cylinder that Y_gg is measured in) and
is deliberately kept separate.

Two approximations are made here and must be carried into any error budget:

1. **CDM versus total matter.** halofit returns the *total* matter power
   spectrum, while the pipeline's ``m`` field is CDM only.  The difference is
   an O(f_b) effect and is not negligible.  It is measurable directly in the
   simulations, since both the CDM and the total maps exist, and
   ``scripts/cross_corr/check_theory_transfer.py`` reports it rather than
   leaving it implicit.
2. **Hydrodynamic back-reaction on the CDM.** The convention adopted for now
   is ``P_mm^hydro-CDM / P_mm^DMO = 1``.  This is known to be wrong at the
   1-2 per cent level (van Daalen et al. 2011; Chisari et al. 2018) and is
   adopted only because no DMO run is available on disk; measuring it would
   require downloading one.  Recorded here so it is booked, not forgotten.

Neither ``n_s`` nor ``sigma8`` appears in any simulation header, so both come
from the published cosmology of each suite via :data:`SIMULATION_COSMOLOGIES`.
Those values are literature lookups, not data, and are flagged as such.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

import kernels as kn

#: Published cosmological parameters per simulation suite.
#:
#: WARNING: these are literature values, not read from the simulation files --
#: no snapshot header carries ``n_s`` or ``sigma8``.  ``Omega_m``, ``Omega_b``
#: and ``h`` *are* in the headers and should be taken from there; only ``n_s``
#: and ``sigma8`` need this table.  Verify against the suite's release paper
#: before quoting any absolute theory amplitude.
SIMULATION_COSMOLOGIES: Dict[str, Dict[str, float]] = {
    # Planck 2015 XIII, the IllustrisTNG cosmology.
    'IllustrisTNG': {'n_s': 0.9667, 'sigma8': 0.8159, 'm_nu': 0.0},
    # FLAMINGO fiducial: the DES Y3 '3x2pt + all external' cosmology.
    # Note this run has massive neutrinos (M_nu = 0.06 eV, in the header),
    # which halofit must be told about.
    'FLAMINGO': {'n_s': 0.967, 'sigma8': 0.807, 'm_nu': 0.06},
}


def halofit_power(h: float, omega_m: float, omega_b: float, z: float,
                  n_s: float, sigma8: float, m_nu: float = 0.0,
                  k_min: float = 1e-3, k_max: float = 50.0,
                  n_k: int = 800, non_linear: bool = True
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Return the matter power spectrum from CAMB, optionally with halofit.

    Args:
        h (float): Dimensionless Hubble parameter.
        omega_m (float): Total matter density parameter.
        omega_b (float): Baryon density parameter.
        z (float): Redshift.
        n_s (float): Scalar spectral index.
        sigma8 (float): Present-day linear amplitude. The primordial
            amplitude is solved for so the spectrum has this sigma8
            before the non-linear step is applied.
        m_nu (float, optional): Summed neutrino mass in eV.  Defaults to 0.
        k_min (float, optional): Minimum wavenumber in h/Mpc.  Defaults to 1e-3.
        k_max (float, optional): Maximum wavenumber in h/Mpc.  Defaults to 50.
        n_k (int, optional): Number of log-spaced samples.  Defaults to 800.
        non_linear (bool, optional): Apply halofit.  Defaults to True.

    Returns:
        tuple: ``(k, P)`` with ``k`` in h/Mpc and ``P`` in (Mpc/h)^3.

    Raises:
        ImportError: If CAMB is not installed.

    Note:
        This is the *total* matter spectrum, not CDM only.  See the module
        docstring.
    """
    try:
        import camb
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            'CAMB is required for the theory chain. It is present in the '
            'cosmodesi_dr1 environment; activate it first.') from exc

    omch2 = (omega_m - omega_b) * h ** 2
    ombh2 = omega_b * h ** 2
    if m_nu > 0:
        # CAMB counts the neutrino contribution inside omega_m, so remove it
        # from the CDM budget rather than adding to the total.
        omch2 -= m_nu / 93.14

    _AS_TRIAL = 2.1e-9
    params = camb.set_params(H0=100.0 * h, ombh2=ombh2, omch2=omch2,
                             mnu=m_nu, ns=n_s, As=_AS_TRIAL)
    # z=0 is requested alongside the target so sigma8 can be normalized there.
    # De-duplicate: passing [0.0, 0.0] makes CAMB's ODE integrator fail with
    # an opaque "Error in dverk" rather than a useful message.
    redshifts = sorted({round(float(z), 8), 0.0}, reverse=True)
    params.set_matter_power(redshifts=redshifts, kmax=k_max * 1.5)

    # Normalize the amplitude BEFORE the non-linear step, not after.  halofit's
    # mapping is not homogeneous in the input amplitude, so rescaling its
    # output by (sigma8_target/sigma8_actual)^2 is only exact in the linear
    # regime; done post hoc it biases P(k) by up to ~0.9 per cent near
    # k ~ 1 h/Mpc for a 0.8 per cent sigma8 mismatch.  One extra linear CAMB
    # evaluation removes the approximation entirely.
    params.NonLinear = camb.model.NonLinear_none
    sigma8_trial = camb.get_results(params).get_sigma8_0()
    params.InitPower.set_params(As=_AS_TRIAL * (sigma8 / sigma8_trial) ** 2,
                                ns=n_s)

    params.NonLinear = (camb.model.NonLinear_both if non_linear
                        else camb.model.NonLinear_none)
    results = camb.get_results(params)

    kh, z_out, pk = results.get_matter_power_spectrum(
        minkh=k_min, maxkh=k_max, npoints=n_k)

    # CAMB returns the redshift axis in INCREASING order regardless of the
    # order the redshifts were requested in, so the requested slice must be
    # selected by value. Taking pk[0] silently returns z=0 instead, which for
    # z=0.5 overestimates the power by a factor of about 2.3 -- large enough
    # to look like a catastrophic halofit failure rather than an indexing bug.
    z_out = np.atleast_1d(np.asarray(z_out, dtype=np.float64))
    iz = int(np.argmin(np.abs(z_out - z)))
    if abs(z_out[iz] - z) > 1e-6:
        raise RuntimeError(
            f'CAMB returned no spectrum at z={z}; got {z_out.tolist()}.')
    return kh, pk[iz]


def project_periodic_box(p_3d: np.ndarray, box_mpc_h: float) -> np.ndarray:
    """Project a 3D power spectrum through the full depth of a periodic box.

    For ``delta_2D = (1/L) int_0^L dz delta_3D`` the transverse Fourier mode
    picks out ``k_z = 0``, giving ``P_2D(k_perp) = P_3D(k_perp) / L``.

    Args:
        p_3d (np.ndarray): 3D power spectrum in (Mpc/h)^3.
        box_mpc_h (float): Box side length in Mpc/h.

    Returns:
        np.ndarray: Projected 2D power spectrum in (Mpc/h)^2.

    Raises:
        ValueError: If the box size is not positive.
    """
    if not box_mpc_h > 0:
        raise ValueError(f"box_mpc_h must be positive, got {box_mpc_h!r}.")
    return np.asarray(p_3d, dtype=np.float64) / box_mpc_h


def amplitudes_from_p2d(k_mpc_h: np.ndarray, p_2d: np.ndarray,
                        radii_mpc_h: Sequence[float], filter_type: str,
                        dr_mpc_h: float, r0_mpc_h: float,
                        pixel_arcmin: Optional[float] = None,
                        resample: bool = True) -> np.ndarray:
    """Integrate a projected spectrum against the aperture kernels.

    The aperture kernels oscillate with a fixed period ``2 pi / R`` in k, which
    a logarithmically spaced grid -- what CAMB returns -- under-samples at high
    k.  By default the spectrum is therefore resampled onto the linear grid of
    :func:`kernels.recommended_k_grid` (log-log interpolated, and clipped to
    the input range so nothing is extrapolated) before quadrature.  Without
    this the amplitude at the largest apertures drifts by a few per mille with
    the number of CAMB samples, which would otherwise be mistaken for model
    error.

    Args:
        k_mpc_h (np.ndarray): Wavenumbers in h/Mpc, strictly increasing.
        p_2d (np.ndarray): Projected power spectrum in (Mpc/h)^2.
        radii_mpc_h (sequence): Aperture radii in Mpc/h.
        filter_type (str): Filter name.
        dr_mpc_h (float): Annulus width in Mpc/h.
        r0_mpc_h (float): Upsilon reference radius in Mpc/h.
        pixel_arcmin (float, optional): If given, convert the result to the
            pipeline's ``1/pixArea`` normalization.  Defaults to None.
        resample (bool, optional): Resample onto a kernel-resolving linear
            grid first.  Defaults to True.

    Returns:
        np.ndarray: Filtered amplitudes, one per radius.
    """
    k_mpc_h = np.asarray(k_mpc_h, dtype=np.float64)
    p_2d = np.asarray(p_2d, dtype=np.float64)

    if resample:
        radii = np.asarray(radii_mpc_h, dtype=np.float64)
        grid = kn.recommended_k_grid(float(radii.min()), float(radii.max()))
        grid = grid[(grid >= k_mpc_h[0]) & (grid <= k_mpc_h[-1])]
        if grid.size >= 16:
            positive = p_2d > 0
            # Log-log interpolation: power spectra are close to power laws
            # over any narrow k range, so this is far more faithful than
            # linear interpolation of a steeply falling function.
            p_2d = np.exp(np.interp(np.log(grid),
                                    np.log(k_mpc_h[positive]),
                                    np.log(p_2d[positive])))
            k_mpc_h = grid

    return np.array([
        kn.filtered_amplitude(k_mpc_h, p_2d, float(R), filter_type,
                              dr=dr_mpc_h, r0=r0_mpc_h,
                              pixel_size=pixel_arcmin)
        for R in radii_mpc_h
    ])


def measured_p2d_from_map(delta: np.ndarray, pixel_arcmin: float,
                          n_bins: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Measure the isotropic 2D power spectrum of an overdensity map.

    Used to validate the transfer chain without any cosmological model: the
    map's own spectrum, pushed through the analytic kernel, must reproduce the
    amplitude the pipeline measures by real-space filtering of the same map.

    Args:
        delta (np.ndarray): Square 2D overdensity map, zero mean.
        pixel_arcmin (float): Angular pixel size in arcmin.
        n_bins (int, optional): Number of wavenumber bins.  Defaults to 400.

    Returns:
        tuple: ``(k, P_2D)`` with ``k`` in 1/arcmin and ``P_2D`` in arcmin^2.

    Raises:
        ValueError: If the map is not square.
    """
    import scipy.fft
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[0] != delta.shape[1]:
        raise ValueError(f'delta must be a square 2D map, got {delta.shape}.')
    n = delta.shape[0]
    area = (n * pixel_arcmin) ** 2

    spec = scipy.fft.rfft2(delta, workers=-1)
    power = (np.abs(spec) ** 2) * area / float(n * n) ** 2

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


def cosmology_for(sim_type: str, header: dict,
                  overrides: Optional[dict] = None) -> Dict[str, float]:
    """Assemble the halofit parameters for a simulation.

    ``Omega_m``, ``Omega_b`` and ``h`` come from the snapshot header, which is
    authoritative.  ``n_s`` and ``sigma8`` are not stored in any header and are
    taken from :data:`SIMULATION_COSMOLOGIES`, i.e. from the literature.

    Args:
        sim_type (str): Suite name, a key of :data:`SIMULATION_COSMOLOGIES`.
        header (dict): Normalized simulation header.
        overrides (dict, optional): Explicit parameter overrides, which take
            precedence over both sources.  Defaults to None.

    Returns:
        dict: Keys ``h``, ``omega_m``, ``omega_b``, ``n_s``, ``sigma8``,
        ``m_nu``.

    Raises:
        KeyError: If the suite has no entry and the missing parameters are not
            supplied via ``overrides``.
    """
    if 'OmegaBaryon' not in header:
        warnings.warn(
            "Header carries no 'OmegaBaryon'; falling back to the Planck "
            "value 0.0486. Pass an override if that is not this simulation's "
            "baryon density.", stacklevel=2)
    params = {
        'h': float(header['HubbleParam']),
        'omega_m': float(header['Omega0']),
        'omega_b': float(header.get('OmegaBaryon', 0.0486)),
    }
    if sim_type not in SIMULATION_COSMOLOGIES and not (
            overrides and {'n_s', 'sigma8'} <= set(overrides)):
        raise KeyError(
            f"No published n_s/sigma8 recorded for sim_type {sim_type!r}; "
            f"add an entry to SIMULATION_COSMOLOGIES or pass overrides."
        )
    params.update(SIMULATION_COSMOLOGIES.get(sim_type, {}))
    if overrides:
        params.update(overrides)
    return params
