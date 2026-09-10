"""Acceptance tests for the Task 1 r-profile machinery (src/rprofiles.py).

Implements the four acceptance tests of ``docs/r_profiles_task1_spec.md``
on synthetic maps.  No simulation data is required, so this suite runs
anywhere the cosmodesi environment runs:

    cd tests/
    pytest test_rprofiles.py -v

The tests pin the properties the science depends on:

1. the FFT aperture kernel reproduces a direct stamp application of
   ``filters.delta_sigma_kernel`` to machine precision;
2. the map-level average ``<F[X] * delta_g>`` equals the stamp-stacked mean at
   galaxy positions, which is what licenses computing galaxy-crossed and
   field-field amplitudes with one routine;
3. the analytic self-pair subtraction takes an unclustered Poisson galaxy
   auto-correlation to zero within its jackknife error;
4. the cross-correlation coefficient of two fields with known mixing is
   recovered, both as an exact algebraic identity and against the
   statistically expected value.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import rprofiles as rp
from filters import delta_sigma_kernel, upsilon


# ---------------------------------------------------------------------------
# Synthetic-map helpers
# ---------------------------------------------------------------------------

N_PIX = 512
PIXEL_ARCMIN = 0.25
TEST_RADII = np.array([1.0, 3.5, 6.0])

#: Aperture grid for the Upsilon checks, mirroring the production spacing so
#: that for the frozen r0 = 2 arcmin it straddles the reference radius: 1.0 and
#: 1.625 lie below it, 2.25 immediately above it (where Upsilon is a small
#: difference of comparable DSigma amplitudes), and 3.5, 6.0 well above.
UPSILON_TEST_RADII = np.array([1.0, 1.625, 2.25, 3.5, 6.0])


def smooth_gaussian_field(n_pixels, correlation_pixels, rng):
    """Return a zero-mean smooth Gaussian random field of unit variance.

    White noise low-pass filtered in Fourier space, giving a field with
    spatial correlations on the requested scale so the aperture filters see
    genuine structure rather than pure pixel noise.

    Args:
        n_pixels (int): Pixels per side.
        correlation_pixels (float): Gaussian smoothing scale, in pixels.
        rng (np.random.Generator): Random generator.

    Returns:
        np.ndarray: Field of shape (n_pixels, n_pixels), zero mean, unit
        standard deviation.
    """
    white = rng.standard_normal((n_pixels, n_pixels))
    kx = np.fft.fftfreq(n_pixels)[:, None]
    ky = np.fft.fftfreq(n_pixels)[None, :]
    k2 = kx ** 2 + ky ** 2
    window = np.exp(-0.5 * k2 * (2 * np.pi * correlation_pixels) ** 2)
    field = np.fft.ifft2(np.fft.fft2(white) * window).real
    field -= field.mean()
    return field / field.std()


def lognormal_map(n_pixels, correlation_pixels, rng, sigma=0.8):
    """Return a strictly positive lognormal map with non-trivial structure.

    Args:
        n_pixels (int): Pixels per side.
        correlation_pixels (float): Gaussian correlation scale, in pixels.
        rng (np.random.Generator): Random generator.
        sigma (float, optional): Log-field amplitude.  Defaults to 0.8.

    Returns:
        np.ndarray: Positive map of shape (n_pixels, n_pixels).
    """
    return np.exp(sigma * smooth_gaussian_field(n_pixels, correlation_pixels, rng))


def stamp_cutout(field, centre, half):
    """Extract a periodic square stamp centred on a pixel.

    Mirrors ``SimulationStacker.cutout_2d_periodic`` but is written out here
    so the unit test does not depend on the heavy stacker import chain.

    Args:
        field (np.ndarray): 2D periodic map.
        centre (tuple): ``(row, col)`` centre pixel.
        half (int): Stamp half-width in pixels.

    Returns:
        np.ndarray: Stamp of shape ``(2*half+1, 2*half+1)``.
    """
    rows = (np.arange(-half, half + 1) + centre[0]) % field.shape[0]
    cols = (np.arange(-half, half + 1) + centre[1]) % field.shape[1]
    return field[np.ix_(rows, cols)]


def defined_mask(filter_type, radii, r0=None):
    """Mask the aperture bins where a coefficient is mathematically defined.

    ``Upsilon(R0; R0) = DSigma(R0) - (R0/R0)^2 * DSigma(R0)`` vanishes
    identically, so every Upsilon amplitude is exactly zero at the reference
    radius and the coefficient there is a genuine 0/0.  That bin carries no
    information and is excluded from comparisons rather than special-cased in
    the library.

    Args:
        filter_type (str): Filter name.
        radii (np.ndarray): Aperture radii in arcmin.
        r0 (float, optional): Upsilon reference radius.  Defaults to
            :data:`rprofiles.R0_ARCMIN`.

    Returns:
        np.ndarray: Boolean mask over ``radii``.
    """
    if r0 is None:
        r0 = rp.R0_ARCMIN
    if filter_type != 'Upsilon':
        return np.ones(len(radii), dtype=bool)
    return ~np.isclose(np.asarray(radii, dtype=float), r0, rtol=0, atol=1e-12)


def stamp_radius_grid(half, pixel_arcmin):
    """Return the exact pixel-centre radius grid of a stamp, in arcmin.

    Exact integer lags are used rather than a linspace over the stamp extent:
    the linspace convention of ``SimulationStacker.radial_distance_grid``
    rounds the half-width, and that rounding is precisely the sub-pixel
    discrepancy the NERSC integration test is meant to expose.  Mixing it in
    here would contaminate a test of the kernel algebra.

    Args:
        half (int): Stamp half-width in pixels.
        pixel_arcmin (float): Angular pixel size in arcmin.

    Returns:
        np.ndarray: Radii of shape ``(2*half+1, 2*half+1)``, in arcmin.
    """
    lags = np.arange(-half, half + 1, dtype=np.float64)
    return np.hypot(lags[:, None], lags[None, :]) * pixel_arcmin


# ---------------------------------------------------------------------------
# Acceptance test 1 -- kernel versus a direct stamp reimplementation
# ---------------------------------------------------------------------------

class TestKernelMatchesStamp:
    """The FFT kernel must reproduce filters.delta_sigma_kernel exactly."""

    @pytest.mark.parametrize('filter_type', ['DSigma', 'Sigma'])
    def test_matches_stamp_at_random_centres(self, filter_type):
        rng = np.random.default_rng(12345)
        field = lognormal_map(N_PIX, 6.0, rng)
        centres = rng.integers(0, N_PIX, size=(20, 2))

        worst = 0.0
        for R in TEST_RADII:
            kern = rp.build_aperture_kernel(N_PIX, PIXEL_ARCMIN, R, filter_type)
            fmap = rp.filtered_map(field, rp.kernel_spectrum(kern), field.shape)

            half = int(np.ceil((R + rp.DR_ARCMIN) / PIXEL_ARCMIN))
            r_grid = stamp_radius_grid(half, PIXEL_ARCMIN)

            for cx, cy in centres:
                cut = stamp_cutout(field, (cx, cy), half)
                if filter_type == 'DSigma':
                    ref = delta_sigma_kernel(cut, r_grid, R, dr=rp.DR_ARCMIN,
                                             pixel_size=PIXEL_ARCMIN)
                else:
                    # Sigma is the positive annulus mean; build it from the
                    # same masks delta_sigma_kernel uses for its annulus.
                    ann = (r_grid >= R) & (r_grid < R + rp.DR_ARCMIN)
                    ref = float(cut[ann].sum()
                                / (PIXEL_ARCMIN ** 2 * ann.sum()))
                worst = max(worst, abs(fmap[cx, cy] - ref))

        assert worst < 1e-12, (
            f'{filter_type}: max |FFT - stamp| = {worst:.3e}, expected < 1e-12'
        )

    def test_dsigma_kernel_is_compensated(self):
        kern = rp.build_aperture_kernel(N_PIX, PIXEL_ARCMIN, 3.0, 'DSigma')
        assert abs(kern.sum()) < 1e-12 * np.abs(kern).sum()

    def test_sigma_kernel_normalization(self):
        kern = rp.build_aperture_kernel(N_PIX, PIXEL_ARCMIN, 3.0, 'Sigma')
        # A positive annulus mean integrates a uniform field of value 1 to
        # 1/pixArea, the surface-density convention of delta_sigma_kernel.
        assert np.all(kern >= 0.0)
        assert kern.sum() == pytest.approx(1.0 / PIXEL_ARCMIN ** 2, rel=1e-12)

    def test_unresolved_aperture_raises(self):
        # 1 arcmin aperture with 4 arcmin pixels: the disk is empty.
        with pytest.raises(ValueError, match='unresolved|Empty'):
            rp.build_aperture_kernel(N_PIX, 4.0, 1.0, 'DSigma')

    def test_kernel_too_large_for_box_raises(self):
        with pytest.raises(ValueError, match='does not fit'):
            rp.build_aperture_kernel(16, PIXEL_ARCMIN, 6.0, 'DSigma')


# ---------------------------------------------------------------------------
# Acceptance test 2 -- correlation equals the stamp-stacked galaxy mean
# ---------------------------------------------------------------------------

class TestCorrelationEqualsStampStack:
    """<F[X] * delta_g> must equal the mean filtered value at galaxy pixels."""

    @staticmethod
    def _setup(rng, n_gal=400):
        field = lognormal_map(N_PIX, 6.0, rng)
        delta_x = rp.to_overdensity(field)

        pix = rng.integers(0, N_PIX, size=(n_gal, 2))
        counts = np.zeros((N_PIX, N_PIX), dtype=np.float64)
        np.add.at(counts, (pix[:, 0], pix[:, 1]), 1.0)
        nbar = n_gal / float(N_PIX * N_PIX)
        delta_g = counts / nbar - 1.0
        return delta_x, delta_g, pix, nbar

    def test_dsigma_matches_stamp_mean(self):
        rng = np.random.default_rng(7)
        delta_x, delta_g, pix, nbar = self._setup(rng)

        Ymat = rp.compute_Y_matrix({'x': delta_x, 'g': delta_g},
                                   PIXEL_ARCMIN, radii=TEST_RADII,
                                   nbar_pix=nbar)

        for ir, R in enumerate(TEST_RADII):
            kern = rp.build_aperture_kernel(N_PIX, PIXEL_ARCMIN, R, 'DSigma')
            fmap = rp.filtered_map(delta_x, rp.kernel_spectrum(kern),
                                   delta_x.shape)
            stamp_mean = fmap[pix[:, 0], pix[:, 1]].mean()
            got = rp.get_Y(Ymat, 'DSigma', 'x', 'g')[ir]
            assert got == pytest.approx(stamp_mean, rel=0, abs=1e-13), (
                f'R={R}: map average {got:.12e} != stamp mean '
                f'{stamp_mean:.12e}'
            )

    @pytest.mark.parametrize('r0', [1.0, 1.5, 2.0])
    def test_upsilon_matches_stamp_filter(self, r0):
        """The FFT Upsilon must equal ``filters.upsilon`` stacked on stamps.

        Parametrized over the reference radius because the production config
        moved R0 from 1 to 2 arcmin, and because the aperture grid deliberately
        straddles R0: :data:`UPSILON_TEST_RADII` contains radii below R0, and
        (for r0 = 2) one immediately above it, where Upsilon is a small
        difference of two comparable DSigma amplitudes and any discrepancy
        between the two routes would be amplified most.
        """
        rng = np.random.default_rng(8)
        delta_x, delta_g, pix, nbar = self._setup(rng)

        radii = UPSILON_TEST_RADII
        Ymat = rp.compute_Y_matrix({'x': delta_x, 'g': delta_g},
                                   PIXEL_ARCMIN, radii=radii,
                                   r0=r0, nbar_pix=nbar)

        half = int(np.ceil((max(radii.max(), r0) + rp.DR_ARCMIN)
                           / PIXEL_ARCMIN))
        r_grid = stamp_radius_grid(half, PIXEL_ARCMIN)
        cutouts = [stamp_cutout(delta_x, (cx, cy), half) for cx, cy in pix]

        for ir, R in enumerate(radii):
            ref = np.mean([
                upsilon(cut, r_grid, R, r0=r0, dr=rp.DR_ARCMIN,
                        pixel_size=PIXEL_ARCMIN)
                for cut in cutouts
            ])
            got = rp.get_Y(Ymat, 'Upsilon', 'x', 'g')[ir]
            assert got == pytest.approx(ref, rel=0, abs=1e-12), (
                f'r0={r0}, R={R}: Upsilon map average {got:.12e} != stamp '
                f'mean {ref:.12e}'
            )


# ---------------------------------------------------------------------------
# Acceptance test 3 -- Poisson galaxy auto-correlation is null
# ---------------------------------------------------------------------------

class TestPoissonGalaxyAuto:
    """Self-pair subtraction must null an unclustered galaxy auto-correlation."""

    def test_ygg_consistent_with_zero(self):
        rng = np.random.default_rng(0)
        n_gal = 20000
        pos = rng.uniform(0.0, float(N_PIX), size=(n_gal, 2))
        counts = np.zeros((N_PIX, N_PIX), dtype=np.float64)
        idx = np.floor(pos).astype(int) % N_PIX
        np.add.at(counts, (idx[:, 0], idx[:, 1]), 1.0)

        nbar = n_gal / float(N_PIX * N_PIX)
        delta_g = counts / nbar - 1.0

        radii = np.array([1.0, 5.0])
        Ymat = rp.compute_Y_matrix({'g': delta_g}, PIXEL_ARCMIN, radii=radii,
                                   nbar_pix=nbar)

        y = rp.get_Y(Ymat, 'DSigma', 'g', 'g')
        y_jk = rp.get_Y(Ymat, 'DSigma', 'g', 'g', jackknife=True)
        err = rp.jackknife_error(y_jk, axis=0)

        for ir, R in enumerate(radii):
            assert abs(y[ir]) < 3.0 * err[ir], (
                f"R={R}': |Y_gg| = {abs(y[ir]):.4e} exceeds 3 sigma = "
                f'{3 * err[ir]:.4e} after self-pair subtraction'
            )

    def test_subtraction_is_what_nulls_it(self):
        """Without the correction the same measurement is strongly non-zero."""
        rng = np.random.default_rng(0)
        n_gal = 20000
        pos = rng.uniform(0.0, float(N_PIX), size=(n_gal, 2))
        counts = np.zeros((N_PIX, N_PIX), dtype=np.float64)
        idx = np.floor(pos).astype(int) % N_PIX
        np.add.at(counts, (idx[:, 0], idx[:, 1]), 1.0)
        nbar = n_gal / float(N_PIX * N_PIX)
        delta_g = counts / nbar - 1.0

        R = 1.0
        kern = rp.build_aperture_kernel(N_PIX, PIXEL_ARCMIN, R, 'DSigma')
        fmap = rp.filtered_map(delta_g, rp.kernel_spectrum(kern), delta_g.shape)
        uncorrected = float(np.mean(fmap * delta_g))

        Ymat = rp.compute_Y_matrix({'g': delta_g}, PIXEL_ARCMIN,
                                   radii=np.array([R]), nbar_pix=nbar)
        corrected = rp.get_Y(Ymat, 'DSigma', 'g', 'g')[0]
        y_jk = rp.get_Y(Ymat, 'DSigma', 'g', 'g', jackknife=True)
        err = rp.jackknife_error(y_jk, axis=0)[0]

        assert abs(uncorrected) > 10.0 * err, (
            'The uncorrected Poisson auto-correlation should be dominated by '
            f'shot noise, got {uncorrected:.4e} against sigma {err:.4e}'
        )
        assert abs(corrected) < abs(uncorrected)


# ---------------------------------------------------------------------------
# Acceptance test 4 -- known mixing recovers the cross-correlation coefficient
# ---------------------------------------------------------------------------

class TestKnownMixingRecoversR:
    """r of f2 = a*f1 + noise must match the analytic value."""

    # 4 arcmin correlation length: a realistic amount of large-scale power, so
    # the positive-weight Sigma filter has a strictly positive auto-amplitude
    # across 1'-6' the way a real projected cosmological field does.  With a
    # much shorter correlation length Y_Sigma decays into the noise by 6' and
    # the coefficient becomes legitimately undefined.
    CORRELATION_PIXELS = 16.0

    @classmethod
    def _build(cls, rng, a=0.7, noise_amp=0.5):
        d1 = smooth_gaussian_field(N_PIX, cls.CORRELATION_PIXELS, rng)
        dn = noise_amp * smooth_gaussian_field(N_PIX, cls.CORRELATION_PIXELS, rng)
        d2 = a * d1 + dn
        # These are already zero-mean overdensity-like fields.
        return d1, dn, d2, a, noise_amp

    def test_exact_algebraic_identity(self):
        """r must equal the identity implied by the realized amplitudes."""
        rng = np.random.default_rng(2024)
        d1, dn, d2, a, _ = self._build(rng)

        Ymat = rp.compute_Y_matrix({'1': d1, 'n': dn, '2': d2},
                                   PIXEL_ARCMIN, radii=TEST_RADII)
        prof = rp.r_profiles(Ymat, pairs=[('1', '2')], ratios=[])

        for filt in rp.FILTERS:
            y11 = rp.get_Y(Ymat, filt, '1', '1')
            y1n = rp.get_Y(Ymat, filt, '1', 'n')
            ynn = rp.get_Y(Ymat, filt, 'n', 'n')
            # Y_12 = a*Y_11 + Y_1n and Y_22 = a^2 Y_11 + 2a Y_1n + Y_nn hold
            # exactly at the realization level because the filter is linear.
            with np.errstate(invalid='ignore'):
                expected = ((a * y11 + y1n)
                            / np.sqrt(y11 * (a ** 2 * y11
                                             + 2 * a * y1n + ynn)))
            got = prof['r'][('1', '2')][filt]

            mask = defined_mask(filt, TEST_RADII)
            assert np.all(np.isfinite(got[mask])), (
                f'{filt}: r has non-finite entries where it should be '
                f'defined: {got}'
            )
            assert np.allclose(got[mask], expected[mask],
                               rtol=1e-10, atol=1e-12), (
                f'{filt}: r={got}, algebraic identity={expected}'
            )

    def test_matches_statistical_expectation(self):
        """r must sit within a few jackknife sigma of a/sqrt(a^2+noise^2)."""
        rng = np.random.default_rng(99)
        d1, dn, d2, a, noise_amp = self._build(rng)

        Ymat = rp.compute_Y_matrix({'1': d1, '2': d2}, PIXEL_ARCMIN,
                                   radii=TEST_RADII)
        prof = rp.r_profiles(Ymat, pairs=[('1', '2')], ratios=[])

        # d1 and dn are independent unit-variance fields with the same power
        # spectrum, so every linear filter sees the same mixing ratio.
        expected = a / np.sqrt(a ** 2 + noise_amp ** 2)

        for filt in rp.FILTERS:
            mask = defined_mask(filt, TEST_RADII)
            got = prof['r'][('1', '2')][filt][mask]
            err = prof['r_err'][('1', '2')][filt][mask]
            assert np.all(np.isfinite(got)), f'{filt}: non-finite r {got}'
            assert np.all(err > 0), f'{filt}: non-positive jackknife error {err}'
            deviation = np.abs(got - expected) / err
            assert np.all(deviation < 4.0), (
                f'{filt}: r={got} deviates from expected {expected:.4f} by '
                f'{deviation} jackknife sigma'
            )

    def test_perfect_correlation_gives_unity(self):
        """A field correlated with itself must give exactly r = 1."""
        rng = np.random.default_rng(5)
        d1 = smooth_gaussian_field(N_PIX, self.CORRELATION_PIXELS, rng)
        Ymat = rp.compute_Y_matrix({'1': d1, '2': d1.copy()}, PIXEL_ARCMIN,
                                   radii=TEST_RADII)
        prof = rp.r_profiles(Ymat, pairs=[('1', '2')], ratios=[])
        for filt in rp.FILTERS:
            mask = defined_mask(filt, TEST_RADII)
            assert np.allclose(prof['r'][('1', '2')][filt][mask], 1.0,
                               atol=1e-10)

    def test_upsilon_vanishes_at_reference_radius(self):
        """Upsilon(R0; R0) is identically zero, so its coefficient is 0/0.

        This is a property of the estimator, not a defect: the reference
        radius carries no information by construction.  Pinning it here means
        the production scripts can rely on that bin being NaN rather than a
        spurious finite value.
        """
        rng = np.random.default_rng(31)
        d1, _, d2, _, _ = self._build(rng)
        radii = np.array([rp.R0_ARCMIN, 3.0])
        Ymat = rp.compute_Y_matrix({'1': d1, '2': d2}, PIXEL_ARCMIN,
                                   radii=radii)
        assert rp.get_Y(Ymat, 'Upsilon', '1', '2')[0] == pytest.approx(0.0,
                                                                       abs=1e-14)
        prof = rp.r_profiles(Ymat, pairs=[('1', '2')], ratios=[])
        assert np.isnan(prof['r'][('1', '2')]['Upsilon'][0])
        assert np.isfinite(prof['r'][('1', '2')]['Upsilon'][1])


# ---------------------------------------------------------------------------
# Supporting behaviour relied on by the production scripts
# ---------------------------------------------------------------------------

class TestLatticeBoundaryDegeneracy:
    """Pin the discretization degeneracy that the integration test exposed.

    Pixel membership uses a strict ``r < edge`` test, so when an aperture edge
    coincides with a realizable lattice distance an entire shell of pixels sits
    on the boundary and flips membership under an arbitrarily small change of
    convention.  This is what makes the FFT route and the legacy stamp route
    disagree at a handful of specific radii while agreeing to machine precision
    everywhere else.
    """

    # The production grids are exactly 0.2 arcmin/pixel.
    PRODUCTION_PIXEL = 0.2

    def test_detects_the_production_degenerate_apertures(self):
        """At 0.2 arcmin/pixel, R=1', R=2.25' and R=6' are degenerate.

        R=1' -> disk edge at 5.0 px, R=2.25' -> annulus edge at 3.0/0.2 = 15.0
        px, R=6' -> disk edge at 30.0 px.  Each of 5, 15 and 30 is a distance
        realizable on the integer lattice in more than one way, so each carries
        a 12-pixel shell.
        """
        flagged = rp.degenerate_apertures(rp.APERTURES_ARCMIN,
                                          self.PRODUCTION_PIXEL)
        assert sorted(flagged) == pytest.approx([1.0, 2.25, 6.0])
        assert flagged[1.0][0][0] == 'disk'
        assert flagged[2.25][0][0] == 'annulus'
        assert flagged[6.0][0][0] == 'disk'
        for R in (1.0, 2.25, 6.0):
            for _, margin, shell in flagged[R]:
                assert margin < 1e-9
                assert shell == 12

    def test_non_degenerate_apertures_are_not_flagged(self):
        clean = [1.625, 2.875, 3.5, 4.125, 4.75, 5.375]
        flagged = rp.degenerate_apertures(clean, self.PRODUCTION_PIXEL)
        assert flagged == {}

    def test_margin_is_the_distance_to_the_nearest_shell(self):
        # 5 pixels exactly: zero margin, shell of 12 -- (+-5,0), (0,+-5),
        # (+-3,+-4), (+-4,+-3).
        margin, shell = rp.lattice_boundary_margin(1.0, 0.2)
        assert margin == pytest.approx(0.0, abs=1e-12)
        assert shell == 12
        # Halfway between shells: a clearly non-degenerate boundary.
        margin, _ = rp.lattice_boundary_margin(1.0, 1.0 / 5.5)
        assert margin > 0.1

    def test_degeneracy_shifts_amplitude_but_not_the_coefficient(self):
        """The amplitude moves under the tie; the coefficient barely does.

        This is the property that makes the Task 1 deliverable robust: every Y
        entering a coefficient is filtered with the same kernel, so a shell of
        pixels entering or leaving the disk largely cancels in the ratio.
        """
        rng = np.random.default_rng(4242)
        d1 = smooth_gaussian_field(N_PIX, 16.0, rng)
        d2 = 0.8 * d1 + 0.4 * smooth_gaussian_field(N_PIX, 16.0, rng)

        radii = np.array([1.0])
        # Membership is the strict test rr < R, so at exactly 0.2 the shell
        # sits at rr == 1.0 and is EXCLUDED; a pixel marginally smaller puts it
        # at rr < 1.0 and INCLUDES it.  That is precisely the real situation:
        # the true production pixel is 0.19998 arcmin while stack_on_array's
        # stamp grid is spaced 0.2 exactly.
        pix_excl = 0.2
        pix_incl = 0.2 * (1 - 1e-4)
        assert rp.degenerate_apertures(radii, pix_excl)

        n_excl = int((rp.build_aperture_kernel(N_PIX, pix_excl, 1.0,
                                               'DSigma') > 0).sum())
        n_incl = int((rp.build_aperture_kernel(N_PIX, pix_incl, 1.0,
                                               'DSigma') > 0).sum())
        assert n_incl - n_excl == 12, (
            f'Expected the 12-pixel shell to flip membership, got '
            f'{n_excl} -> {n_incl} disk pixels'
        )

        y_a = rp.compute_Y_matrix({'a': d1, 'b': d2}, pix_excl, radii=radii)
        y_b = rp.compute_Y_matrix({'a': d1, 'b': d2}, pix_incl, radii=radii)

        amp_shift = abs(rp.get_Y(y_b, 'DSigma', 'a', 'b')[0]
                        / rp.get_Y(y_a, 'DSigma', 'a', 'b')[0] - 1.0)
        r_a = rp.r_profiles(y_a, pairs=[('a', 'b')],
                            ratios=[])['r'][('a', 'b')]['DSigma'][0]
        r_b = rp.r_profiles(y_b, pairs=[('a', 'b')],
                            ratios=[])['r'][('a', 'b')]['DSigma'][0]
        r_shift = abs(r_b / r_a - 1.0)

        # A smooth synthetic field moves less than the real, much steeper gas
        # profile does (0.8 per cent here against 6 per cent measured on
        # TNG300-1); the floor only guarantees the test exercises the effect.
        assert amp_shift > 1e-3, (
            f'The shell flip should move the amplitude measurably, got '
            f'{amp_shift:.3e}; the test is not exercising the effect'
        )
        assert r_shift < 0.3 * amp_shift, (
            f'Expected the coefficient to be far more stable than the '
            f'amplitude, got amplitude shift {amp_shift:.3e} and coefficient '
            f'shift {r_shift:.3e}'
        )


class TestFilterCompensation:
    """Compensated filters must be insensitive to the largest modes.

    The annulus-mean ('Sigma') kernel is uncompensated: its transform
    W_ann(k; R1, R2) = 2[R2 J1(kR2) - R1 J1(kR1)] / (k(R2^2 - R1^2)) tends to 1
    as k tends to 0, so the filtered amplitude integrates power down to the box
    fundamental and is not comparable between boxes of different size.  The
    DSigma and Upsilon kernels are compensated, with transforms vanishing at
    k = 0.  This is the reason the production filter set is {DSigma, Upsilon}.
    """

    def test_kernel_transforms_have_the_expected_k_to_zero_limit(self):
        """W_Sigma(k->0) -> 1/pixArea-normalized constant; W_DSigma(k->0) -> 0."""
        pix = 0.25
        for R in (1.0, 3.0):
            for filt, expected_zero in (('Sigma', False), ('DSigma', True)):
                kern = rp.build_aperture_kernel(N_PIX, pix, R, filt)
                # The k=0 element of the transform is just the kernel sum.
                total = kern.sum()
                if expected_zero:
                    assert abs(total) < 1e-12 * np.abs(kern).sum(), (
                        f'{filt} at R={R} should be compensated, sum={total:.3e}')
                else:
                    assert total == pytest.approx(1.0 / pix ** 2, rel=1e-12), (
                        f'{filt} at R={R} should integrate a uniform field to '
                        f'1/pixArea, got {total:.6e}')

    def test_highpass_leaves_compensated_amplitudes_alone(self):
        """Removing the longest modes must barely move DSigma/Upsilon.

        Sigma, by contrast, should shift substantially -- which is exactly why
        it cannot be compared between simulation boxes of different size.
        """
        rng = np.random.default_rng(2718)
        # Correlation length well below the cut, so there is genuine power on
        # both sides of it.
        d1 = smooth_gaussian_field(N_PIX, 8.0, rng)
        d2 = 0.85 * d1 + 0.35 * smooth_gaussian_field(N_PIX, 8.0, rng)
        pix = 0.25
        radii = np.array([2.0, 4.0])
        # Cut at half the map, i.e. 16x the largest aperture.  The compensated
        # filters are only insensitive when the cut sits well above the
        # aperture scale, since their transforms vanish at k=0 but only
        # polynomially: measured separation between Sigma and DSigma is 135x at
        # this ratio, 27x at 8x, and 5x at 4x.  Production is safer still --
        # TNG300-1's 205 cMpc/h box is 535 arcmin against a largest aperture of
        # 9.75 arcmin, a ratio of 55.
        cut = 0.5 * N_PIX * pix

        full = rp.compute_Y_matrix({'a': d1, 'b': d2}, pix, radii=radii)
        cut_mat = rp.compute_Y_matrix(
            {'a': rp.highpass_field(d1, pix, cut),
             'b': rp.highpass_field(d2, pix, cut)}, pix, radii=radii)

        shifts = {}
        for filt in rp.FILTERS:
            mask = defined_mask(filt, radii)
            a = rp.get_Y(full, filt, 'a', 'b')[mask]
            b = rp.get_Y(cut_mat, filt, 'a', 'b')[mask]
            shifts[filt] = float(np.max(np.abs(b / a - 1.0)))

        assert shifts['DSigma'] < 1e-3, (
            f"DSigma is compensated and should be insensitive to the low-k "
            f"cut, got {shifts['DSigma']:.3e}")
        assert shifts['Upsilon'] < 1e-3, (
            f"Upsilon is compensated and should be insensitive to the low-k "
            f"cut, got {shifts['Upsilon']:.3e}")
        assert shifts['Sigma'] > 30 * shifts['DSigma'], (
            f"Sigma is uncompensated and should be far more sensitive than "
            f"DSigma, got Sigma={shifts['Sigma']:.3e} vs "
            f"DSigma={shifts['DSigma']:.3e}")

    def test_highpass_preserves_zero_mean_and_removes_the_modes(self):
        rng = np.random.default_rng(11)
        d = smooth_gaussian_field(256, 8.0, rng)
        pix = 0.25
        cut = 0.25 * 256 * pix
        out = rp.highpass_field(d, pix, cut)
        assert abs(out.mean()) < 1e-12
        spec = np.abs(np.fft.rfft2(out))
        kx = 2 * np.pi * np.fft.fftfreq(256, d=pix)
        ky = 2 * np.pi * np.fft.rfftfreq(256, d=pix)
        k2 = kx[:, None] ** 2 + ky[None, :] ** 2
        assert spec[k2 < (2 * np.pi / cut) ** 2].max() < 1e-10

    def test_highpass_rejects_a_cut_larger_than_the_map(self):
        rng = np.random.default_rng(12)
        d = smooth_gaussian_field(64, 4.0, rng)
        with pytest.raises(ValueError, match='exceeds the map size'):
            rp.highpass_field(d, 0.25, 1e4)


class TestSupportingBehaviour:

    def test_jackknife_blocks_tile_the_map(self):
        edges = rp.block_edges(N_PIX, 4)
        counts = rp.block_counts(edges)
        assert len(counts) == 16
        assert counts.sum() == N_PIX * N_PIX

    def test_jackknife_blocks_handle_indivisible_grids(self):
        edges = rp.block_edges(2674, 4)
        counts = rp.block_counts(edges)
        assert counts.sum() == 2674 * 2674
        assert edges[0] == 0 and edges[-1] == 2674

    def test_jackknife_realizations_leave_one_out(self):
        sums = np.array([[1.0, 2.0, 3.0, 4.0]])
        counts = np.array([1.0, 1.0, 1.0, 1.0])
        got = rp.jackknife_realizations(sums, counts)
        # Excluding block 0 leaves (2+3+4)/3 = 3.
        assert got[0, 0] == pytest.approx(3.0)
        assert got[0, 3] == pytest.approx(2.0)

    def test_to_overdensity_is_zero_mean(self):
        rng = np.random.default_rng(1)
        field = lognormal_map(64, 4.0, rng)
        delta = rp.to_overdensity(field)
        assert abs(delta.mean()) < 1e-12
        assert delta.min() > -1.0 - 1e-12

    def test_to_overdensity_rejects_empty_field(self):
        with pytest.raises(ValueError, match='strictly positive'):
            rp.to_overdensity(np.zeros((8, 8)))

    def test_derive_cdm_field_is_exact(self):
        rng = np.random.default_rng(3)
        baryon = lognormal_map(64, 4.0, rng)
        cdm_true = 5.0 * lognormal_map(64, 4.0, rng)
        total = baryon + cdm_true
        got = rp.derive_cdm_field(total, baryon, header=None, verbose=False)
        assert np.allclose(got, cdm_true, rtol=1e-12, atol=0)

    def test_derive_cdm_field_rejects_mismatched_pair(self):
        rng = np.random.default_rng(4)
        baryon = lognormal_map(64, 4.0, rng)
        # 'total' that is smaller than the baryons cannot be a matched pair.
        with pytest.raises(ValueError, match='negative cell'):
            rp.derive_cdm_field(0.5 * baryon, baryon, header=None,
                                verbose=False)

    def test_Y_is_symmetric_under_field_swap(self):
        rng = np.random.default_rng(11)
        d1 = smooth_gaussian_field(N_PIX, 6.0, rng)
        d2 = smooth_gaussian_field(N_PIX, 6.0, rng)
        Ymat = rp.compute_Y_matrix({'a': d1, 'b': d2}, PIXEL_ARCMIN,
                                   radii=np.array([2.0]))
        assert (rp.get_Y(Ymat, 'DSigma', 'a', 'b')
                == pytest.approx(rp.get_Y(Ymat, 'DSigma', 'b', 'a')))

    def test_galaxy_field_requires_nbar(self):
        rng = np.random.default_rng(6)
        d = smooth_gaussian_field(64, 4.0, rng)
        with pytest.raises(ValueError, match='nbar_pix'):
            rp.compute_Y_matrix({'g': d}, PIXEL_ARCMIN,
                                radii=np.array([2.0]))

    def test_mismatched_shapes_rejected(self):
        rng = np.random.default_rng(6)
        with pytest.raises(ValueError, match='share one grid'):
            rp.compute_Y_matrix(
                {'a': smooth_gaussian_field(64, 4.0, rng),
                 'b': smooth_gaussian_field(32, 4.0, rng)},
                PIXEL_ARCMIN, radii=np.array([2.0]))


# ---------------------------------------------------------------------------
# Upsilon: the amplitude-level combination against every alternative route
# ---------------------------------------------------------------------------

class TestUpsilonConstruction:
    """Upsilon is built as a linear combination of DSigma *amplitudes*.

    ``compute_Y_matrix`` never convolves an Upsilon kernel; it forms
    ``Y_DSigma(R) - (R0/R)^2 Y_DSigma(R0)`` after the fact.  That is exact only
    because the filtered amplitude is linear in the kernel, so the same answer
    must come out of every other way of arranging the same algebra:

    - convolving a single composite kernel
      ``K_DSigma(R) - (R0/R)^2 K_DSigma(R0)`` once;
    - summing the kernel against the field directly in real space, with no FFT
      at all;
    - integrating the analytic transform ``kernels.w_upsilon`` against the
      measured 2D power spectrum.

    The first two pin the algebra and the FFT round-off; the third pins the
    pixelization of the aperture at the frozen reference radius.
    """

    @staticmethod
    def _composite_upsilon_kernel(n_pixels, pixel_arcmin, R, r0, dr):
        """Build a single real-space Upsilon kernel by combining two DSigmas.

        Args:
            n_pixels (int): Pixels per side.
            pixel_arcmin (float): Angular pixel size in arcmin.
            R (float): Aperture radius in arcmin.
            r0 (float): Reference radius in arcmin.
            dr (float): Annulus width in arcmin.

        Returns:
            np.ndarray: Kernel of shape ``(n_pixels, n_pixels)``.
        """
        k_r = rp.build_aperture_kernel(n_pixels, pixel_arcmin, R, 'DSigma', dr)
        k_0 = rp.build_aperture_kernel(n_pixels, pixel_arcmin, r0, 'DSigma', dr)
        return k_r - (r0 / R) ** 2 * k_0

    @pytest.mark.parametrize('r0', [1.0, 2.0])
    def test_composite_kernel_equals_amplitude_combination(self, r0):
        """One convolution with the composite kernel == two combined amplitudes.

        This is the step the amplitude-level shortcut could have got wrong, and
        the only one: everything else in the Upsilon path is shared with
        DSigma, which has its own stamp comparison above.
        """
        rng = np.random.default_rng(20)
        delta_x = rp.to_overdensity(lognormal_map(N_PIX, 6.0, rng))
        delta_y = rp.to_overdensity(lognormal_map(N_PIX, 6.0, rng))

        radii = UPSILON_TEST_RADII
        Ymat = rp.compute_Y_matrix({'x': delta_x, 'y': delta_y},
                                   PIXEL_ARCMIN, radii=radii, r0=r0)
        got = rp.get_Y(Ymat, 'Upsilon', 'x', 'y')

        worst = 0.0
        for ir, R in enumerate(radii):
            kern = self._composite_upsilon_kernel(N_PIX, PIXEL_ARCMIN, R, r0,
                                                  rp.DR_ARCMIN)
            fmap = rp.filtered_map(delta_x, rp.kernel_spectrum(kern),
                                   delta_x.shape)
            ref = float(np.mean(fmap * delta_y))
            # Scale by the DSigma amplitude rather than by Upsilon itself: at
            # R just above R0, Upsilon is a small difference of comparable
            # numbers and a relative tolerance on it would be meaningless.
            scale = abs(rp.get_Y(Ymat, 'DSigma', 'x', 'y')[ir])
            worst = max(worst, abs(got[ir] - ref) / scale)

        assert worst < 1e-12, (
            f'r0={r0}: composite-kernel Upsilon differs from the '
            f'amplitude-level combination by {worst:.3e} of the DSigma '
            'amplitude'
        )

    @pytest.mark.parametrize('r0', [1.0, 2.0])
    def test_fft_equals_direct_real_space_sum(self, r0):
        """The FFT is only a fast exact convolution -- verify against the sum.

        The filtered map is defined as ``F(x) = sum_l K(l) delta(x + l)``.
        Evaluating that sum directly at a handful of centres removes the FFT
        entirely and bounds its round-off, including the cancellation incurred
        by Upsilon's reference subtraction near R0.
        """
        rng = np.random.default_rng(21)
        delta_x = rp.to_overdensity(lognormal_map(N_PIX, 6.0, rng))
        centres = rng.integers(0, N_PIX, size=(8, 2))

        worst = 0.0
        for R in UPSILON_TEST_RADII:
            kern = self._composite_upsilon_kernel(N_PIX, PIXEL_ARCMIN, R, r0,
                                                  rp.DR_ARCMIN)
            fmap = rp.filtered_map(delta_x, rp.kernel_spectrum(kern),
                                   delta_x.shape)

            half = int(np.ceil((max(R, r0) + rp.DR_ARCMIN) / PIXEL_ARCMIN))
            # The kernel is stored with fft wraparound; fold it back to a
            # centred stamp so it can be dotted against a centred cutout.
            idx = np.arange(-half, half + 1) % N_PIX
            stamp_kernel = kern[np.ix_(idx, idx)]

            for cx, cy in centres:
                cut = stamp_cutout(delta_x, (cx, cy), half)
                direct = float(np.sum(stamp_kernel * cut))
                worst = max(worst, abs(fmap[cx, cy] - direct))

        # Absolute, because the filtered map crosses zero: the amplitudes here
        # are O(1), so this is a relative statement at the 1e-11 level.
        assert worst < 1e-11, (
            f'r0={r0}: FFT and direct real-space sum differ by {worst:.3e}'
        )

    @staticmethod
    def _axis_transform(n, pixel, R, r0, dr):
        """Return ``(k, pixArea * FFT(composite Upsilon kernel))`` on one axis.

        Follows the convention of ``test_kernels.TestAnalyticMatchesPixelized``:
        ``build_aperture_kernel`` weights pixels by ``1/(pixArea*N)``, so the
        discrete transform carries an extra ``1/pixArea`` relative to the
        normalized continuum kernel, and one axis of the 2D transform is taken
        rather than an azimuthal average (the staircase aperture is not exactly
        isotropic, and the axis slice is the unambiguous comparison).

        Args:
            n (int): Pixels per side.
            pixel (float): Angular pixel size in arcmin.
            R (float): Aperture radius in arcmin.
            r0 (float): Reference radius in arcmin.
            dr (float): Annulus width in arcmin.

        Returns:
            tuple: ``(k_axis, discrete_transform)``, both shape ``(n//2 + 1,)``.
        """
        kern = (rp.build_aperture_kernel(n, pixel, R, 'DSigma', dr)
                - (r0 / R) ** 2
                * rp.build_aperture_kernel(n, pixel, r0, 'DSigma', dr))
        spec = np.fft.rfft2(kern).real * pixel ** 2
        return 2.0 * np.pi * np.fft.rfftfreq(n, d=pixel), spec[0, :]

    @pytest.mark.parametrize('r0', [1.0, 2.0])
    @pytest.mark.parametrize('R', [3.5, 6.0])
    def test_pixelized_kernel_matches_the_analytic_transform(self, r0, R):
        """The pixelized Upsilon kernel must match ``kernels.w_upsilon``.

        The Task 4 theory chain integrates the analytic transform while the
        simulation chain convolves the pixelized kernel, so the two must agree
        wherever the Limber integral has its support.  ``test_kernels.py``
        pins this for Sigma and DSigma; Upsilon was never covered, and it is
        the filter that inherits the reference-radius discretization at every
        aperture.  The production configs have since moved R0 from 1 to 2
        arcmin, which changes which kernel that reference term is.

        Band and tolerance follow ``test_kernels.TestAnalyticMatchesPixelized``:
        ``kR < 1``, where the transform peaks and carries the amplitude for any
        realistic power spectrum.  The departure beyond that is the documented
        pixelization floor, exercised by the convergence test below.
        """
        import kernels as kn

        n, pixel = 512, 0.25
        k, discrete = self._axis_transform(n, pixel, R, r0, rp.DR_ARCMIN)
        analytic = kn.w_upsilon(k, R, rp.DR_ARCMIN, r0)

        band = (k > 0) & (k * R < 1.0)
        worst = float(np.max(np.abs(discrete[band] - analytic[band])))
        assert worst < 1e-2, (
            f'r0={r0}, R={R}: max |discrete - analytic| = {worst:.3e} for '
            f'kR < 1'
        )

    @pytest.mark.parametrize('r0', [1.0, 2.0])
    def test_upsilon_pixelization_error_converges(self, r0):
        """Halving the pixel must halve the departure from the continuum kernel.

        This is what identifies the residual as pixelization rather than an
        error in the composite kernel: a genuine algebra mistake would not
        shrink with resolution.  As for the other filters it is the 0.75 arcmin
        annulus that limits it, not the disk.
        """
        import kernels as kn

        R = 3.5
        errs = []
        for n, pixel in ((512, 0.25), (1024, 0.125), (2048, 0.0625)):
            k, discrete = self._axis_transform(n, pixel, R, r0, rp.DR_ARCMIN)
            analytic = kn.w_upsilon(k, R, rp.DR_ARCMIN, r0)
            band = (k > 0) & (k * R < 3.0)
            errs.append(float(np.max(np.abs(discrete[band]
                                            - analytic[band]))))

        assert all(a > b for a, b in zip(errs, errs[1:])), (
            f'r0={r0}: pixelization error should fall with the pixel, got '
            f'{errs}'
        )
        assert errs[-1] < 0.4 * errs[0], (
            f'r0={r0}: four-fold refinement should shrink the error well '
            f'below half, got {errs}'
        )

    def test_upsilon_is_zero_below_and_at_the_reference_radius_only_at_r0(self):
        """Upsilon is exactly zero at R0 and finite (wrong-signed) below it.

        Below R0 the ``(R0/R)^2`` reference term is larger than one and
        over-subtracts, so Upsilon carries no meaningful information there.
        This test pins the behaviour that
        :func:`rprofiles.upsilon_defined_mask` exists to hide.
        """
        rng = np.random.default_rng(22)
        delta_x = rp.to_overdensity(lognormal_map(N_PIX, 6.0, rng))
        delta_y = rp.to_overdensity(lognormal_map(N_PIX, 6.0, rng))

        r0 = 2.0
        radii = np.array([1.0, 1.625, 2.0, 2.25, 3.5])
        Ymat = rp.compute_Y_matrix({'x': delta_x, 'y': delta_y},
                                   PIXEL_ARCMIN, radii=radii, r0=r0)
        y = rp.get_Y(Ymat, 'Upsilon', 'x', 'y')
        ds = rp.get_Y(Ymat, 'DSigma', 'x', 'y')

        assert y[2] == pytest.approx(0.0, abs=1e-15 * abs(ds[2]) + 1e-300)
        assert np.all(np.abs(y[[0, 1]]) > 0.0)

        mask = rp.upsilon_defined_mask(radii, r0)
        assert list(mask) == [False, False, False, True, True]


# ---------------------------------------------------------------------------
# The Park et al. (2021) Y transform
# ---------------------------------------------------------------------------

class TestYTransformConstruction:
    """The Y transform is assembled, not convolved -- so prove it is exact.

    ``rprofiles.assemble_ytransform`` builds ``Y(R; Rmax) = Sigma(R) -
    Sigma(Rmax)`` from amplitudes already measured, on the operator identity
    ``F^Y_R = F^Sigma_R - F^Sigma_Rmax``.  That shortcut is the kind that fails
    *silently*: a wrong reference index or a mis-shaped jackknife broadcast
    still yields finite, smooth, plausible amplitudes.  Each check below
    therefore tests it against the thing it replaces.
    """

    def test_amplitudes_match_a_directly_built_composite_kernel(self):
        """The assembled amplitude equals one convolution with Sigma_R - Sigma_Rmax.

        This is the claim that makes the shortcut legitimate, and it is the
        addendum's Appendix A requirement that the transform be a direct
        map-level filter rather than a reconstruction.
        """
        rng = np.random.default_rng(101)
        delta_x = rp.to_overdensity(lognormal_map(N_PIX, 6.0, rng))
        delta_y = rp.to_overdensity(lognormal_map(N_PIX, 6.0, rng))
        radii = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rmax = 5.0

        Ymat = rp.compute_Y_matrix({'x': delta_x, 'y': delta_y},
                                   PIXEL_ARCMIN, radii=radii)
        Y, _, _ = rp.assemble_ytransform(Ymat, rmax)
        got = Y[('x', 'y')]

        ref_kernel = rp.build_aperture_kernel(N_PIX, PIXEL_ARCMIN, rmax,
                                              'Sigma', rp.DR_ARCMIN)
        direct = np.empty(len(radii))
        for i, R in enumerate(radii):
            composite = rp.build_aperture_kernel(
                N_PIX, PIXEL_ARCMIN, float(R), 'Sigma', rp.DR_ARCMIN
            ) - ref_kernel
            fmap = rp.filtered_map(delta_x, rp.kernel_spectrum(composite),
                                   delta_x.shape)
            direct[i] = float(np.mean(fmap * delta_y, dtype=np.float64))

        scale = np.abs(direct).max()
        assert np.allclose(got, direct, rtol=0, atol=1e-12 * scale), (
            f'assembled vs composite-kernel Y transform: worst absolute '
            f'difference {np.abs(got - direct).max():.3e} against amplitude '
            f'scale {scale:.3e}'
        )

    def test_the_transform_vanishes_identically_at_rmax(self):
        """``Y(Rmax; Rmax) = 0`` exactly, which is why that bin is masked.

        The coefficient there is a genuine 0/0, the mirror image of Upsilon at
        ``R = R0``.
        """
        rng = np.random.default_rng(102)
        delta_x = rp.to_overdensity(lognormal_map(N_PIX, 6.0, rng))
        delta_y = rp.to_overdensity(lognormal_map(N_PIX, 6.0, rng))
        radii = np.array([1.0, 2.0, 4.0])
        rmax = 4.0

        Ymat = rp.compute_Y_matrix({'x': delta_x, 'y': delta_y},
                                   PIXEL_ARCMIN, radii=radii)
        Y, Y_jk, _ = rp.assemble_ytransform(Ymat, rmax)

        sigma = rp.get_Y(Ymat, 'Sigma', 'x', 'y')
        assert Y[('x', 'y')][-1] == pytest.approx(
            0.0, abs=1e-15 * abs(sigma[-1]) + 1e-300)
        assert np.all(np.abs(Y_jk[('x', 'y')][:, -1]) <= 1e-15
                      * np.abs(rp.get_Y(Ymat, 'Sigma', 'x', 'y',
                                        jackknife=True)[:, -1]) + 1e-300)
        assert np.all(np.abs(Y[('x', 'y')][:-1]) > 0.0)

    def test_jackknife_realizations_are_differenced_per_realization(self):
        """Each leave-one-out estimate is differenced against its own Rmax value.

        Differencing the full-map value out of every realization instead would
        still produce a smooth error bar, and a wrong one: the two apertures
        are measured on the same map and are strongly correlated, which is
        exactly the cancellation the addendum's Appendix A trap 5 is about.
        """
        rng = np.random.default_rng(103)
        delta_x = rp.to_overdensity(lognormal_map(N_PIX, 6.0, rng))
        delta_y = rp.to_overdensity(lognormal_map(N_PIX, 6.0, rng))
        radii = np.array([1.0, 2.0, 3.0])
        rmax = 3.0

        Ymat = rp.compute_Y_matrix({'x': delta_x, 'y': delta_y},
                                   PIXEL_ARCMIN, radii=radii, n_jk_side=3)
        _, Y_jk, _ = rp.assemble_ytransform(Ymat, rmax)
        sigma_jk = rp.get_Y(Ymat, 'Sigma', 'x', 'y', jackknife=True)

        expected = sigma_jk - sigma_jk[:, -1][:, None]
        assert np.array_equal(Y_jk[('x', 'y')], expected)

        # And the naive alternative is measurably different, so the test has
        # teeth rather than passing on a degenerate case.
        naive = sigma_jk - rp.get_Y(Ymat, 'Sigma', 'x', 'y')[-1]
        assert not np.allclose(Y_jk[('x', 'y')], naive)

    def test_a_reference_radius_off_the_grid_raises(self):
        """An interpolated reference radius is an error, not an approximation."""
        rng = np.random.default_rng(104)
        delta_x = rp.to_overdensity(lognormal_map(128, 6.0, rng))
        Ymat = rp.compute_Y_matrix({'x': delta_x}, PIXEL_ARCMIN,
                                   radii=np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match='not on the aperture grid'):
            rp.assemble_ytransform(Ymat, 2.5)

    def test_an_unknown_base_filter_raises(self):
        """The base filter must be one the amplitudes were measured with."""
        rng = np.random.default_rng(105)
        delta_x = rp.to_overdensity(lognormal_map(128, 6.0, rng))
        Ymat = rp.compute_Y_matrix({'x': delta_x}, PIXEL_ARCMIN,
                                   radii=np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match='is not in Ymat'):
            rp.assemble_ytransform(Ymat, 3.0, base='NotAFilter')


class TestYTransformDefinedMask:
    """The mask is the single definition of which Y-transform bins survive."""

    def test_drops_the_band_at_and_above_the_usable_fraction(self):
        radii = np.array([1.0, 2.0, 4.0, 4.8, 5.0, 6.0])
        mask = rp.ytransform_defined_mask(radii, 6.0)
        # 0.8 * 6.0 = 4.8, and the test is strict `<`, so 4.8 itself is out.
        assert list(mask) == [True, True, True, False, False, False]

    def test_the_production_configuration_keeps_the_expected_bins(self):
        """R0 = 1' and Rmax = 6' on the production grid, pinned.

        Both are already aperture-grid points, which is why the grid does not
        move when they are adopted; this pins the bin counts the figures show.
        """
        base = np.linspace(1.0, 6.0, 9)
        step = base[1] - base[0]
        radii = np.concatenate(
            [base, np.arange(base[-1] + step, 10.0 + 0.5 * step, step)])

        assert np.any(np.isclose(radii, 1.0))
        assert np.any(np.isclose(radii, 6.0))

        in_data = radii <= 6.0 + 1e-9
        upsilon = rp.upsilon_defined_mask(radii, 1.0)
        ytr = rp.ytransform_defined_mask(radii, 6.0)

        assert int((upsilon & in_data).sum()) == 8
        assert int((ytr & in_data).sum()) == 7
        assert radii[ytr][-1] == pytest.approx(4.75)

    def test_rejects_a_non_positive_reference_radius(self):
        radii = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match='rmax must be positive'):
            rp.ytransform_defined_mask(radii, 0.0)
        with pytest.raises(ValueError, match='fraction must be positive'):
            rp.ytransform_defined_mask(radii, 6.0, fraction=0.0)


class TestRProfilesFilterSelection:
    """``r_profiles(filters=...)`` must not change the existing default."""

    def _ymat(self, seed=106):
        rng = np.random.default_rng(seed)
        deltas = {k: rp.to_overdensity(lognormal_map(256, 6.0, rng))
                  for k in ('g', 'm')}
        return rp.compute_Y_matrix(deltas, PIXEL_ARCMIN,
                                   radii=np.array([1.0, 2.0, 4.0]),
                                   nbar_pix=1.0, n_jk_side=2)

    def test_the_default_is_exactly_the_module_filter_set(self):
        """An unmodified compute_Y_matrix result must reproduce FILTERS.

        The default was a module constant and is now read off the amplitude
        dict; if those two ever diverge, every existing caller silently
        changes what it reports.
        """
        Ymat = self._ymat()
        assert tuple(Ymat['Y']) == rp.FILTERS

        got = rp.r_profiles(Ymat, pairs=(('g', 'm'),), ratios=())
        expected = rp.r_profiles(Ymat, pairs=(('g', 'm'),), ratios=(),
                                 filters=rp.FILTERS)
        for filt in rp.FILTERS:
            assert np.array_equal(got['r'][('g', 'm')][filt],
                                  expected['r'][('g', 'm')][filt],
                                  equal_nan=True)
            assert np.array_equal(got['r_err'][('g', 'm')][filt],
                                  expected['r_err'][('g', 'm')][filt],
                                  equal_nan=True)

    def test_an_injected_filter_is_reported(self):
        """The whole point: an appended derived filter needs no new algebra."""
        Ymat = self._ymat()
        (Ymat['Y']['Ytransform'],
         Ymat['Y_jk']['Ytransform'], _) = rp.assemble_ytransform(Ymat, 4.0)

        prof = rp.r_profiles(Ymat, pairs=(('g', 'm'),), ratios=())
        assert 'Ytransform' in prof['r'][('g', 'm')]
        assert prof['r'][('g', 'm')]['Ytransform'].shape == (3,)

        # Formed from the amplitudes, not copied from another filter.
        coeff = prof['r'][('g', 'm')]['Ytransform']
        with np.errstate(invalid='ignore', divide='ignore'):
            expected = (rp.get_Y(Ymat, 'Ytransform', 'g', 'm')
                        / np.sqrt(rp.get_Y(Ymat, 'Ytransform', 'g', 'g')
                                  * rp.get_Y(Ymat, 'Ytransform', 'm', 'm')))
        finite = np.isfinite(coeff) & np.isfinite(expected)
        assert finite.any()
        assert np.allclose(coeff[finite], expected[finite])
