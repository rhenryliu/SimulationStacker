"""Tests for the Round 3A code, ``scripts/cross_corr/round3a_lib.py`` and its
drivers.

Round 3A asks whether the filtered calibration factor's departure from unity
is window smearing or mediation failure.  Everything it concludes rests on
four pieces of machinery, and a mistake in any of them would be silent:

1. **Exact Parseval amplitudes.**  The regression check against the committed
   round-two amplitudes computes ``Y_ab`` as a sum over Fourier modes.  It must
   equal ``rprofiles.compute_Y_matrix`` to rounding, galaxy self-pairs included.
2. **Binned amplitudes and the split.**  ``C_F = W_F M_F`` is an exact
   identity at the binned level; ``M_F`` must be exactly 1 under constructed
   mediation and exactly ``c`` when ``C(k) = c``.
3. **The DoG kernel.**  Compensated, strictly positive window, the right
   central value for the self-pair term, and swept by a path that reproduces
   ``compute_Y_matrix`` exactly when handed a pixelized DSigma kernel.
4. **The wiring.**  Which spectrum feeds which slot of ``C(k)``, and the
   Convention T recombination, tested with distinct numbers so a swap shows.

Run with::

    cd tests/
    pytest test_round3a.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.fft

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts' / 'cross_corr'))

import kernels as kn  # noqa: E402
import rprofiles as rp  # noqa: E402
import make_calibration_factor as mcf  # noqa: E402
import make_task9_spectra as mts  # noqa: E402
import round3a_lib as lib  # noqa: E402

PIXEL = 0.2
N = 256
DR = 0.75
RADII = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.0, 5.0])
WIDTH = 0.02


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def fields():
    """Correlated overdensity maps plus a Poisson galaxy field.

    Same construction as ``test_calibration_factor.synthetic_deltas``: one
    Gaussian realization smoothed on different scales plus independent noise,
    so the fields are correlated without being proportional.

    Returns:
        tuple: ``(deltas, nbar_pix, f_b, n_gal)``.
    """
    rng = np.random.default_rng(20260927)
    kx = np.fft.fftfreq(N)[:, None]
    ky = np.fft.rfftfreq(N)[None, :]
    k = np.hypot(kx, ky)
    white = np.fft.rfft2(rng.standard_normal((N, N)))

    def smoothed(scale, noise_amp):
        field = np.fft.irfft2(white * np.exp(-0.5 * (k * scale) ** 2),
                              s=(N, N))
        field = field + noise_amp * rng.standard_normal((N, N))
        return np.exp(0.3 * field / field.std())

    mass_m = smoothed(6.0, 0.05)
    mass_b = smoothed(14.0, 0.10)
    mass_e = 0.8 * mass_b + 0.2 * smoothed(20.0, 0.10)
    f_b = float(mass_b.mean() / (mass_m.mean() + mass_b.mean()))
    deltas = {'m': rp.to_overdensity(mass_m),
              'b': rp.to_overdensity(mass_b),
              'e': rp.to_overdensity(mass_e)}
    lam = 0.05 * (1.0 + 2.0 * deltas['m'])
    counts = rng.poisson(np.clip(lam, 1e-6, None)).astype(float)
    deltas['g'] = rp.to_overdensity(counts)
    return deltas, counts.sum() / counts.size, f_b, int(counts.sum())


@pytest.fixture(scope='module')
def binning():
    """Linear bins of width 0.02/arcmin over the synthetic grid."""
    return lib.ModeBinning(N, PIXEL, WIDTH)


@pytest.fixture(scope='module')
def ffts(fields):
    """``rfft2`` of every synthetic field."""
    deltas = fields[0]
    return {k: scipy.fft.rfft2(v) for k, v in deltas.items()}


@pytest.fixture(scope='module')
def Ymat(fields):
    """``compute_Y_matrix`` on the synthetic fields: the reference."""
    deltas, nbar_pix, _, _ = fields
    return rp.compute_Y_matrix(deltas, PIXEL, radii=RADII, dr=DR,
                               nbar_pix=nbar_pix, n_jk_side=2)


def pixel_kernel_spec(R, filt):
    """Real ``rfft2`` of a pixelized base-filter kernel, and its centre.

    Args:
        R (float): Aperture radius, arcmin.
        filt (str): 'Sigma' or 'DSigma'.

    Returns:
        tuple: ``(spectrum, kernel[0, 0])``.
    """
    kern = rp.build_aperture_kernel(N, PIXEL, R, filt, DR)
    return np.real(scipy.fft.rfft2(kern)), float(kern[0, 0])


# ---------------------------------------------------------------------------
# 1. Mode binning and exact Parseval amplitudes
# ---------------------------------------------------------------------------

class TestModeBinning:
    """Bins must cover the full plane and preserve total power."""

    def test_counts_cover_the_full_plane(self, binning):
        assert binning.counts.sum() == N * N

    def test_binned_power_obeys_parseval(self, fields, ffts, binning):
        delta = fields[0]['m']
        p = binning.cross_power(ffts['m'], ffts['m'])
        total = np.nansum(binning.counts * p)
        assert total == pytest.approx(binning.area * np.mean(delta ** 2),
                                      rel=1e-10)

    def test_invalid_width_raises(self):
        with pytest.raises(ValueError):
            lib.ModeBinning(N, PIXEL, 0.0)


class TestExactParsevalAmplitudes:
    """The regression machinery must reproduce ``compute_Y_matrix``."""

    @pytest.mark.parametrize('filt', ['Sigma', 'DSigma'])
    @pytest.mark.parametrize('pair', [('m', 'm'), ('b', 'm'), ('g', 'm'),
                                      ('b', 'g'), ('e', 'm')])
    def test_cross_pairs(self, ffts, binning, Ymat, filt, pair):
        a, b = pair
        wre = binning.weighted(np.real(ffts[a] * np.conj(ffts[b])))
        for i, R in enumerate(RADII):
            spec, _ = pixel_kernel_spec(R, filt)
            y = lib.exact_amplitude(spec, wre, N)
            ref = Ymat['Y'][filt][rp._pair_key(a, b)][i]
            assert y == pytest.approx(ref, rel=1e-10, abs=1e-14)

    @pytest.mark.parametrize('filt', ['Sigma', 'DSigma'])
    def test_galaxy_auto_with_self_pairs(self, fields, ffts, binning, Ymat,
                                         filt):
        nbar = fields[1]
        wre = binning.weighted(np.real(ffts['g'] * np.conj(ffts['g'])))
        for i, R in enumerate(RADII):
            spec, k0 = pixel_kernel_spec(R, filt)
            y = lib.exact_amplitude(spec, wre, N) - k0 / nbar
            ref = Ymat['Y'][filt][('g', 'g')][i]
            assert y == pytest.approx(ref, rel=1e-9, abs=1e-12)

    def test_exact_mediated_amplitude_under_per_mode_mediation(self, fields,
                                                               ffts, binning):
        """Build ``delta_b = eta(|k|) delta_m`` exactly, eta constant per bin.

        Then ``P_gb = eta P_gm`` mode by mode, so the mediated amplitude built
        from the binned ``eta = P_bm/P_mm`` must equal the measured ``Y_gb``
        at every aperture, i.e. ``M_F = 1`` to rounding.
        """
        eta_bin = 0.5 + 0.4 * np.cos(np.arange(binning.n_bins) / 40.0)
        F_b = binning.per_mode(eta_bin) * ffts['m']
        eta_meas = (binning.cross_power(F_b, ffts['m'])
                    / binning.cross_power(ffts['m'], ffts['m']))
        good = binning.counts > 0
        np.testing.assert_allclose(eta_meas[good], eta_bin[good], rtol=1e-10)
        eta_meas[~good] = 0.0
        w_gb = binning.weighted(np.real(ffts['g'] * np.conj(F_b)))
        w_med = binning.weighted(binning.per_mode(eta_meas)
                                 * np.real(ffts['g'] * np.conj(ffts['m'])))
        for R in RADII:
            spec, _ = pixel_kernel_spec(R, 'DSigma')
            y_gb = lib.exact_amplitude(spec, w_gb, N)
            y_med = lib.exact_amplitude(spec, w_med, N)
            assert y_med / y_gb == pytest.approx(1.0, rel=1e-9)

    def test_zero_lag_from_spectrum_is_the_kernel_centre(self, binning):
        for R in (1.0, 3.5):
            spec, k0 = pixel_kernel_spec(R, 'DSigma')
            assert lib.zero_lag(spec, N) == pytest.approx(k0, rel=1e-12)


class TestBinnedAmplitudes:
    """Binned amplitudes approximate the exact ones; shot noise is exact."""

    def test_binned_close_to_exact(self, ffts, binning, Ymat):
        """Only approximate: within a bin the pixelized kernel is slightly
        anisotropic and the per-mode power fluctuates, and on this 256^2 grid
        a bin holds tens of modes, so the two correlate at the per-cent level
        (worst where the amplitude is smallest).  Production grids hold
        ~10^2-10^3 times more modes per bin; there the check is the closure
        test of ``make_ck_spectra.py`` against its own threshold."""
        p = binning.cross_power(ffts['b'], ffts['m'])
        for i, R in enumerate(RADII):
            spec, _ = pixel_kernel_spec(R, 'DSigma')
            w = binning.bin_sum(spec)
            y = lib.binned_amplitude(w, p, binning.area)
            ref = Ymat['Y']['DSigma'][('b', 'm')][i]
            assert y == pytest.approx(ref, rel=3e-2)

    def test_flat_shot_noise_reproduces_the_self_pair_term(self, fields,
                                                           binning):
        _, nbar, _, n_gal = fields
        p_shot = np.full(binning.n_bins, binning.area / n_gal)
        p_shot[binning.counts == 0] = np.nan
        for R in (1.0, 2.25, 5.0):
            spec, k0 = pixel_kernel_spec(R, 'DSigma')
            w = binning.bin_sum(spec)
            y = lib.binned_amplitude(w, p_shot, binning.area)
            assert y == pytest.approx(k0 / nbar, rel=1e-10)


# ---------------------------------------------------------------------------
# 2. The window/mediation split
# ---------------------------------------------------------------------------

class TestSplit:
    """``C_F = W_F M_F`` and the two limits that define the pieces."""

    @pytest.fixture
    def random_inputs(self):
        rng = np.random.default_rng(7)
        n_bins = 60
        w = rng.normal(size=(5, n_bins))        # signed, like DSigma's
        p_mm = rng.uniform(1.0, 2.0, n_bins)
        p_xm = rng.uniform(0.3, 1.0, n_bins) * p_mm
        p_gm = rng.uniform(1.0, 3.0, n_bins) * p_mm
        p_gx = rng.uniform(0.5, 1.5, n_bins) * p_xm * p_gm / p_mm
        return w, p_xm, p_gm, p_mm, p_gx

    def test_identity(self, random_inputs):
        out = lib.window_mediation_split(*random_inputs, area=3.0)
        np.testing.assert_allclose(out['C'], out['W'] * out['M'], rtol=1e-12)

    def test_split_from_amplitudes_guards_zero_denominators(self):
        out = lib.split_from_amplitudes([1.0, 1.0], [2.0, 2.0], [3.0, 0.0],
                                        [4.0, 4.0], [5.0, 5.0])
        assert out['C'][0] == pytest.approx(2.0 / 12.0)
        assert out['M'][0] == pytest.approx(5.0 / 4.0)
        assert np.isnan(out['C'][1]) and np.isnan(out['W'][1])

    def test_exact_mediation_gives_unit_mediation_factor(self, random_inputs):
        w, p_xm, p_gm, p_mm, _ = random_inputs
        p_gx = p_xm * p_gm / p_mm
        out = lib.window_mediation_split(w, p_xm, p_gm, p_mm, p_gx, area=3.0)
        np.testing.assert_allclose(out['M'], 1.0, rtol=1e-12)
        np.testing.assert_allclose(out['W'], out['C'], rtol=1e-12)

    def test_constant_C_of_k_is_recovered(self, random_inputs):
        w, p_xm, p_gm, p_mm, _ = random_inputs
        c = 1.07
        p_gx = p_xm * p_gm / p_mm / c
        out = lib.window_mediation_split(w, p_xm, p_gm, p_mm, p_gx, area=3.0)
        np.testing.assert_allclose(out['M'], c, rtol=1e-12)

    def test_constant_eta_has_no_window_term(self, random_inputs):
        w, _, p_gm, p_mm, p_gx = random_inputs
        p_xm = 0.6 * p_mm
        out = lib.window_mediation_split(w, p_xm, p_gm, p_mm, p_gx, area=3.0)
        np.testing.assert_allclose(out['W'], 1.0, rtol=1e-12)

    def test_empty_bins_are_skipped(self, random_inputs):
        w, p_xm, p_gm, p_mm, p_gx = random_inputs
        p_xm = p_xm.copy()
        p_xm[:3] = np.nan
        w = w.copy()
        w[:, :3] = 0.0
        out = lib.window_mediation_split(w, p_xm, p_gm, p_mm, p_gx, area=3.0)
        assert np.all(np.isfinite(out['C']))

    def test_derived_filter_weights_match_the_amplitude_rule(self, ffts,
                                                             binning):
        p = binning.cross_power(ffts['g'], ffts['m'])
        w = np.array([binning.bin_sum(pixel_kernel_spec(R, 'DSigma')[0])
                      for R in RADII])
        y = lib.binned_amplitude(w, p, binning.area)
        w_ups = lib.derived_filter_weights(w, RADII, 'Upsilon', 1.0)
        y_ups = lib.binned_amplitude(w_ups, p, binning.area)
        np.testing.assert_allclose(y_ups, y - (1.0 / RADII) ** 2 * y[0],
                                   rtol=1e-12, atol=1e-14)
        w_sig = np.array([binning.bin_sum(pixel_kernel_spec(R, 'Sigma')[0])
                          for R in RADII])
        y_sig = lib.binned_amplitude(w_sig, p, binning.area)
        w_yt = lib.derived_filter_weights(w_sig, RADII, 'Ytransform', 5.0)
        np.testing.assert_allclose(
            lib.binned_amplitude(w_yt, p, binning.area), y_sig - y_sig[-1],
            rtol=1e-12, atol=1e-14)

    def test_derived_filter_reference_must_be_on_grid(self):
        with pytest.raises(ValueError):
            lib.derived_filter_weights(np.ones((3, 4)), np.array([1., 2., 3.]),
                                       'Upsilon', 1.5)


# ---------------------------------------------------------------------------
# 3. The DoG kernel and the generic sweep
# ---------------------------------------------------------------------------

class TestDoGKernel:
    """Compensated, strictly positive window, correct central value."""

    def test_compensated_and_positive(self):
        spec = lib.dog_kernel_spectrum(N, PIXEL, 0.6, 1.2)
        assert spec[0, 0] == 0.0
        rest = spec.copy()
        rest[0, 0] = 1.0
        assert np.all(rest > 0.0)

    def test_zero_lag_matches_the_continuum(self, binning):
        s1, s2 = 0.6, 1.2
        spec = lib.dog_kernel_spectrum(N, PIXEL, s1, s2)
        assert lib.zero_lag(spec, N) == pytest.approx(
            lib.dog_zero_lag(s1, s2), rel=1e-6)

    def test_real_space_kernel_is_the_sampled_continuum_kernel(self):
        s1, s2 = 0.6, 1.2
        kern = scipy.fft.irfft2(lib.dog_kernel_spectrum(N, PIXEL, s1, s2),
                                s=(N, N))
        for lag in (0, 2, 5):
            r = lag * PIXEL
            cont = (np.exp(-r ** 2 / (2 * s1 ** 2)) / (2 * np.pi * s1 ** 2)
                    - np.exp(-r ** 2 / (2 * s2 ** 2)) / (2 * np.pi * s2 ** 2))
            assert kern[lag, 0] == pytest.approx(cont, rel=1e-6)

    def test_constant_field_filters_to_zero(self):
        spec = lib.dog_kernel_spectrum(N, PIXEL, 0.6, 1.2)
        out = rp.filtered_map(np.ones((N, N)), spec, (N, N))
        assert np.max(np.abs(out)) < 1e-10

    def test_window_function_matches_the_grid_kernel(self):
        k = np.linspace(0.0, 5.0, 11)
        w = lib.dog_window(k, 0.6, 1.2)
        assert w[0] == 0.0 and np.all(w[1:] > 0.0)

    def test_invalid_widths_raise(self):
        with pytest.raises(ValueError):
            lib.dog_kernel_spectrum(N, PIXEL, 1.2, 0.6)


class TestGenericSweep:
    """``filtered_amplitudes`` must be ``compute_Y_matrix`` for any kernel."""

    def test_reproduces_compute_Y_matrix_for_pixelized_dsigma(self, fields,
                                                               Ymat):
        deltas, nbar, _, _ = fields
        specs, k0s = [], []
        for R in RADII:
            kern = rp.build_aperture_kernel(N, PIXEL, R, 'DSigma', DR)
            specs.append(rp.kernel_spectrum(kern))
            k0s.append(float(kern[0, 0]))
        out = lib.filtered_amplitudes(deltas, specs, k0s, nbar_pix=nbar,
                                      n_jk_side=2)
        for pair, ref in Ymat['Y']['DSigma'].items():
            np.testing.assert_allclose(out['Y'][pair], ref, rtol=1e-13,
                                       atol=1e-15)
            np.testing.assert_allclose(out['Y_jk'][pair],
                                       Ymat['Y_jk']['DSigma'][pair],
                                       rtol=1e-13, atol=1e-15)

    def test_poisson_galaxy_auto_is_null_after_self_pairs(self):
        rng = np.random.default_rng(11)
        counts = rng.poisson(0.2, size=(N, N)).astype(float)
        deltas = {'g': rp.to_overdensity(counts)}
        nbar = counts.sum() / counts.size
        grid = lib.ModeBinning(N, PIXEL, WIDTH)
        specs, k0s = [], []
        for s1 in (0.6, 1.0):
            spec = lib.dog_kernel_spectrum(N, PIXEL, s1, 2 * s1)
            specs.append(spec)
            k0s.append(lib.zero_lag(spec, N))
        out = lib.filtered_amplitudes(deltas, specs, k0s, nbar_pix=nbar,
                                      n_jk_side=4)
        y = out['Y'][('g', 'g')]
        err = rp.jackknife_error(out['Y_jk'][('g', 'g')], axis=0)
        assert np.all(np.abs(y) < 3.0 * err)

    def test_galaxy_field_without_density_raises(self, fields):
        deltas = {'g': fields[0]['g']}
        with pytest.raises(ValueError):
            lib.filtered_amplitudes(deltas, [np.ones((N, N // 2 + 1))], [0.0])


# ---------------------------------------------------------------------------
# 4. Wiring
# ---------------------------------------------------------------------------

class TestWiring:
    """Distinct numbers in every slot, so a swapped key cannot pass."""

    def test_harmonic_C_through_calibration_factors(self):
        P = {('m', 'm'): np.array([5.0]), ('b', 'm'): np.array([2.0]),
             ('g', 'm'): np.array([3.0]), ('b', 'g'): np.array([7.0]),
             ('b', 'b'): np.array([11.0]), ('g', 'g'): np.array([13.0]),
             ('e', 'm'): np.array([17.0]), ('e', 'g'): np.array([19.0]),
             ('e', 'e'): np.array([23.0]), ('b', 'e'): np.array([29.0])}
        mcf.add_convention_t(P, 0.2)
        fac_b = mcf.calibration_factors(P, gas='b')
        fac_e = mcf.calibration_factors(P, gas='e')
        assert fac_b['C'][0] == pytest.approx(2.0 * 3.0 / (5.0 * 7.0))
        assert fac_e['C'][0] == pytest.approx(17.0 * 3.0 / (5.0 * 19.0))
        assert fac_b['x'][0] == pytest.approx(2.0 / 5.0)

    def test_convention_t_spectra_match_a_direct_total_map(self, fields, ffts,
                                                           binning):
        deltas, _, f_b, _ = fields
        delta_t = (1.0 - f_b) * deltas['m'] + f_b * deltas['b']
        F_t = scipy.fft.rfft2(delta_t)
        P = {}
        for a, b in (('m', 'm'), ('b', 'm'), ('b', 'b'), ('g', 'm'),
                     ('b', 'g'), ('e', 'm'), ('b', 'e')):
            P[rp._pair_key(a, b)] = binning.cross_power(ffts[a], ffts[b])
        mcf.add_convention_t(P, f_b)
        good = binning.counts > 0
        np.testing.assert_allclose(
            P[('g', 't')][good], binning.cross_power(ffts['g'], F_t)[good],
            rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            P[('t', 't')][good], binning.cross_power(F_t, F_t)[good],
            rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Small numerical helpers used by the analysis scripts."""

    def test_loglog_slope_of_a_power_law(self):
        radii = np.linspace(1.0, 9.75, 15)
        slope, local = lib.loglog_slope(radii, 3.0 * radii ** -2)
        assert slope == pytest.approx(-2.0, abs=1e-12)
        np.testing.assert_allclose(local, -2.0, atol=1e-12)
        slope_neg, _ = lib.loglog_slope(radii, -3.0 * radii ** -1.5)
        assert slope_neg == pytest.approx(-1.5, abs=1e-12)

    def test_loglog_slope_rejects_a_sign_change(self):
        with pytest.raises(ValueError):
            lib.loglog_slope(np.array([1.0, 2.0]), np.array([1.0, -1.0]))

    def test_leave_one_out_is_exact_on_a_line(self):
        x = np.array([0.4, 0.5, 0.6, 0.7])
        np.testing.assert_allclose(lib.loo_linear_prediction(x, 1 + 2 * x),
                                   1 + 2 * x, rtol=1e-12)
        assert lib.predict_from_fit(x[:3], 1 + 2 * x[:3], 0.9) == \
            pytest.approx(2.8)

    def test_interp_log_k_handles_unsorted_source(self):
        k_src = np.array([5.0, 2.0, 1.0])
        y_src = np.log(k_src)
        out = lib.interp_log_k(k_src, y_src, np.array([3.0, 0.5, 6.0]))
        assert out[0] == pytest.approx(np.log(3.0))
        assert np.isnan(out[1]) and np.isnan(out[2])

    def test_paired_correlation(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(400, 3))
        np.testing.assert_allclose(lib.paired_correlation(a, 2 * a + 1), 1.0)
        b = rng.normal(size=(400, 3))
        assert np.all(np.abs(lib.paired_correlation(a, b)) < 0.2)

    def test_split_by_parent_mass(self):
        idx = np.arange(10, 21)
        mass = np.array([5, 1, 9, 3, 7, 2, 8, 4, 6, 10, 0], dtype=float)
        lo, hi = lib.split_by_parent_mass(idx, mass)
        assert lo.size == 5 and hi.size == 6
        assert set(lo) | set(hi) == set(idx) and not set(lo) & set(hi)
        lookup = dict(zip(idx, mass))
        assert max(lookup[i] for i in lo) <= min(lookup[i] for i in hi)

    def test_response_quantiles_match_the_task9_definition(self):
        k = np.linspace(0.005, 20.0, 8000)
        p = k ** -1.0
        for R in (1.0, 3.0):
            ours = lib.response_quantiles_values(k, p, kn.w_dsigma(k, R, DR))
            ref = mts.response_quantiles(k, p, R, 'DSigma', DR)
            np.testing.assert_allclose(ours, ref, rtol=0, atol=0)

    def test_rebin_preserves_count_weighted_means(self):
        counts = np.array([1.0, 3.0, 0.0, 2.0])
        k_mean = np.array([1.0, 2.0, np.nan, 4.0])
        p = {'a': np.array([10.0, 20.0, np.nan, 40.0])}
        k_c, out = lib.rebin_spectra(counts, k_mean, p,
                                     np.array([0.5, 2.5, 5.0]))
        np.testing.assert_allclose(k_c, [(1 + 6) / 4.0, 4.0])
        np.testing.assert_allclose(out['a'], [(10 + 60) / 4.0, 40.0])


class TestComponentBookkeeping:
    """Eq. (4) and Eq. (67) on synthetic component maps."""

    def test_total_and_component_identities(self):
        rng = np.random.default_rng(5)
        n = 64
        names = ['DM', 'ionized_gas', 'neutral_gas', 'Stars', 'BH']
        base = rng.normal(size=(n, n))
        masses = [np.exp(0.2 * (base + a * rng.normal(size=(n, n))))
                  * scale for a, scale in zip((0.1, 0.5, 0.9, 1.3, 2.0),
                                              (5.0, 0.8, 0.05, 0.1, 0.01))]
        means = np.array([m.mean() for m in masses])
        deltas = [m / m.mean() - 1.0 for m in masses]
        grid = lib.ModeBinning(n, PIXEL, 0.1)
        F = [scipy.fft.rfft2(d) for d in deltas]
        P = np.array([[grid.cross_power(F[i], F[j]) for j in range(5)]
                      for i in range(5)])
        out = lib.component_bookkeeping(P, means, names)
        total = sum(masses)
        F_t = scipy.fft.rfft2(total / total.mean() - 1.0)
        good = grid.counts > 0
        np.testing.assert_allclose(out['P_tt'][good],
                                   grid.cross_power(F_t, F_t)[good],
                                   rtol=1e-10, atol=1e-14)
        x_sum = sum(out['weights'][c] * out['x_comp'][c]
                    for c in lib.BARYON_COMPONENTS)
        np.testing.assert_allclose(out['x'][good], x_sum[good], rtol=1e-10)


# ---------------------------------------------------------------------------
# 6. Driver wiring: the DoG payload and the analysis split
# ---------------------------------------------------------------------------

class TestDoGDriver:
    """``make_dog_calibration`` must reach the round-two payload format."""

    @pytest.fixture(scope='class')
    def payload(self, fields):
        import make_dog_calibration as mdc
        deltas, nbar, f_b, _ = fields
        sigma1 = np.array([0.6, 1.0, 1.6])
        filters, k0s = mdc.dog_filters(deltas, PIXEL, sigma1, [2.0, 1.5],
                                       nbar, f_b, n_jk_side=2)
        meta = {'label': 'synthetic', 'radii_are': 'sigma1_arcmin'}
        return mcf.flatten_for_npz(filters, sigma1, f_b, meta), filters, k0s

    def test_keys_and_suppression_only_for_baryons(self, payload):
        out, _, _ = payload
        for name in ('DoG_q=2', 'DoG_q=1.5'):
            assert f'C_b_{name}' in out and f'C_e_{name}' in out
            assert f'Cerr_b_{name}' in out and f'S_b_{name}' in out
            assert f'S_e_{name}' not in out

    def test_C_is_the_four_amplitude_ratio(self, payload):
        out, filters, _ = payload
        Y = filters['DoG_q=2']['Y']
        ratio = Y[('b', 'm')] * Y[('g', 'm')] / (Y[('m', 'm')] * Y[('b', 'g')])
        np.testing.assert_allclose(out['C_b_DoG_q=2'], ratio, rtol=1e-12)

    def test_sweep_matches_exact_parseval(self, fields, ffts, binning,
                                          payload):
        _, filters, k0s = payload
        _, nbar, _, _ = fields
        for i, s1 in enumerate((0.6, 1.0, 1.6)):
            spec = lib.dog_kernel_spectrum(N, PIXEL, s1, 2.0 * s1)
            for a, b in (('b', 'm'), ('g', 'm'), ('g', 'g')):
                wre = binning.weighted(np.real(ffts[a] * np.conj(ffts[b])))
                y = lib.exact_amplitude(spec, wre, N)
                if a == b == 'g':
                    y -= k0s['DoG_q=2'][i] / nbar
                assert y == pytest.approx(
                    filters['DoG_q=2']['Y'][rp._pair_key(a, b)][i],
                    rel=1e-9, abs=1e-13)


class TestAnalysisSplitWiring:
    """``round3a_ck_analysis.split`` reads the right keys in the right slots."""

    @pytest.fixture
    def fake(self):
        rng = np.random.default_rng(13)
        radii = np.array([1.0, 2.0, 3.0])
        d = {'radii': radii, 'dog_sigma1': np.array([0.5]),
             'meta_f_b': np.array(0.16)}
        for pair in ('mm', 'bm', 'bb', 'em', 'be', 'ee', 'gm_fid', 'bg_fid',
                     'eg_fid', 'gg_fid'):
            d[f'Yx_{pair}_DSigma'] = rng.uniform(0.5, 2.0, 3)
        for X in ('b', 'e'):
            for conv in ('C', 'T'):
                d[f'Ymed_{X}_{conv}_fid_DSigma'] = rng.uniform(0.5, 2.0, 3)
        return d

    def test_convention_C_baryons(self, fake):
        import round3a_ck_analysis as cka
        s = cka.split(fake, 'DSigma', 'fid', 'b', 'C')
        Y = lambda p: fake[f'Yx_{p}_DSigma']
        np.testing.assert_allclose(
            s['C'], Y('bm') * Y('gm_fid') / (Y('mm') * Y('bg_fid')))
        np.testing.assert_allclose(
            s['M'], fake['Ymed_b_C_fid_DSigma'] / Y('bg_fid'))
        np.testing.assert_allclose(s['W'] * s['M'], s['C'])

    def test_convention_T_electrons(self, fake):
        import round3a_ck_analysis as cka
        s = cka.split(fake, 'DSigma', 'fid', 'e', 'T')
        Y = lambda p: fake[f'Yx_{p}_DSigma']
        fb = 0.16
        fm = 1.0 - fb
        y_et = fm * Y('em') + fb * Y('be')
        y_gt = fm * Y('gm_fid') + fb * Y('bg_fid')
        y_tt = fm * fm * Y('mm') + 2 * fm * fb * Y('bm') + fb * fb * Y('bb')
        np.testing.assert_allclose(s['C'], y_et * y_gt / (y_tt * Y('eg_fid')))
        np.testing.assert_allclose(
            s['M'], fake['Ymed_e_T_fid_DSigma'] / Y('eg_fid'))

    def test_upsilon_is_derived_and_masked(self, fake):
        import round3a_ck_analysis as cka
        s = cka.split(fake, 'Upsilon_R0=1', 'fid', 'b', 'C')
        assert np.isnan(s['C'][0]) and np.all(np.isfinite(s['C'][1:]))
        Y = lambda p: (fake[f'Yx_{p}_DSigma']
                       - (1.0 / fake['radii']) ** 2 * fake[f'Yx_{p}_DSigma'][0])
        np.testing.assert_allclose(
            s['C'][1:], (Y('bm') * Y('gm_fid') / (Y('mm') * Y('bg_fid')))[1:])

    def test_binned_split_reads_the_binned_spectra(self):
        """The independent binned route feeds each spectrum to its slot."""
        import round3a_ck_analysis as cka
        rng = np.random.default_rng(29)
        fb, area = 0.16, 50.0
        d = {'radii': np.array([1.0, 2.0, 3.0]), 'meta_f_b': np.array(fb),
             'meta_area_arcmin2': np.array(area),
             'w_DSigma': rng.uniform(0.1, 1.0, (3, 6))}
        for pair in ('mm', 'bm', 'bb', 'em', 'be', 'gm_fid', 'bg_fid',
                     'eg_fid'):
            d[f'P2D_{pair}'] = rng.uniform(0.5, 2.0, 6)
        P = lambda p: d[f'P2D_{p}']
        Y = lambda p: d['w_DSigma'] @ p / area
        s = cka.binned_split(d, 'DSigma', 'fid', 'b', 'C')
        np.testing.assert_allclose(
            s['C'], Y(P('bm')) * Y(P('gm_fid'))
            / (Y(P('mm')) * Y(P('bg_fid'))))
        np.testing.assert_allclose(
            s['M'], Y(P('bm') * P('gm_fid') / P('mm')) / Y(P('bg_fid')))
        fm = 1.0 - fb
        p_et = fm * P('em') + fb * P('be')
        p_gt = fm * P('gm_fid') + fb * P('bg_fid')
        p_tt = fm * fm * P('mm') + 2 * fm * fb * P('bm') + fb * fb * P('bb')
        s = cka.binned_split(d, 'DSigma', 'fid', 'e', 'T')
        np.testing.assert_allclose(
            s['C'], Y(p_et) * Y(p_gt) / (Y(p_tt) * Y(P('eg_fid'))))
        np.testing.assert_allclose(
            s['M'], Y(p_et * p_gt / p_tt) / Y(P('eg_fid')))


class TestOutputHygiene:
    """Atomic writes and the regression guard used by every analysis script."""

    def test_atomic_npz_and_text(self, tmp_path):
        out = lib.save_npz_atomic(tmp_path / 'a.npz', x=np.arange(3))
        np.testing.assert_array_equal(np.load(out)['x'], np.arange(3))
        txt = lib.write_text_atomic(tmp_path / 'a.txt', 'hello')
        assert txt.read_text() == 'hello'
        assert not list(tmp_path.glob('*.tmp*'))

    def test_regression_guard(self, tmp_path):
        good = lib.save_npz_atomic(tmp_path / 'g.npz', reg_pass=np.array(True),
                                   reg_maxrel_all=np.array(1e-13))
        lib.require_regression_pass(np.load(good), good)
        bad = lib.save_npz_atomic(tmp_path / 'b.npz', reg_pass=np.array(False),
                                  reg_maxrel_all=np.array(1e-3))
        with pytest.raises(SystemExit):
            lib.require_regression_pass(np.load(bad), bad)
        missing = lib.save_npz_atomic(tmp_path / 'm.npz', x=np.array(1))
        with pytest.raises(SystemExit):
            lib.require_regression_pass(np.load(missing), missing)
