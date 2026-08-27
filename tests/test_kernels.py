"""Tests for the analytic harmonic-space aperture kernels (src/kernels.py).

The point of these tests is the bridge between the two representations of the
same filter: the pixelized kernel that ``rprofiles`` actually applies to maps,
and the analytic transform that the theory chain integrates against a power
spectrum.  If those two disagree, a theory amplitude cannot be compared to a
measured one, which is the whole content of Task 4's first bullet.

Run:

    cd tests/
    pytest test_kernels.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import kernels as kn
import rprofiles as rp


class TestKernelLimits:
    """Analytic limits that the compensation argument rests on."""

    def test_disk_and_annulus_tend_to_unity_at_zero_k(self):
        assert kn.w_disk(np.array([0.0]), 3.0)[0] == pytest.approx(1.0)
        assert kn.w_annulus(np.array([0.0]), 3.0, 3.75)[0] == pytest.approx(1.0)

    def test_compensated_kernels_vanish_at_zero_k(self):
        """This is the formal statement behind the Gate B filter freeze."""
        assert kn.w_dsigma(np.array([0.0]), 3.0, 0.75)[0] == pytest.approx(
            0.0, abs=1e-14)
        assert kn.w_upsilon(np.array([0.0]), 3.0, 0.75, 1.0)[0] == (
            pytest.approx(0.0, abs=1e-14))

    def test_sigma_does_not_vanish_at_zero_k(self):
        """The annulus mean is uncompensated, hence box-size sensitive."""
        assert kn.w_sigma(np.array([0.0]), 3.0, 0.75)[0] == pytest.approx(1.0)

    def test_small_x_series_matches_direct_ratio(self):
        """The series branch must join the Bessel branch smoothly."""
        x = np.array([0.9e-4, 1.1e-4, 1e-3, 1e-2, 1.0, 10.0])
        from scipy.special import j1
        direct = 2.0 * j1(x) / x
        assert np.allclose(kn._two_j1_over_x(x), direct, rtol=1e-10)

    def test_upsilon_is_identically_zero_at_the_reference_radius(self):
        k = np.linspace(0.0, 5.0, 64)
        assert np.allclose(kn.w_upsilon(k, 1.0, 0.75, 1.0), 0.0, atol=1e-14)

    def test_dispatcher_matches_the_named_kernels(self):
        k = np.linspace(0.0, 4.0, 33)
        assert np.allclose(kn.aperture_kernel_ft(k, 2.0, 'DSigma', 0.75),
                           kn.w_dsigma(k, 2.0, 0.75))
        assert np.allclose(kn.aperture_kernel_ft(k, 2.0, 'Upsilon', 0.75, 1.0),
                           kn.w_upsilon(k, 2.0, 0.75, 1.0))
        with pytest.raises(ValueError, match='filter_type'):
            kn.aperture_kernel_ft(k, 2.0, 'nonsense')


class TestAnalyticMatchesPixelized:
    """The analytic transform must reproduce the pixelized kernel's transform.

    ``rprofiles.build_aperture_kernel`` weights pixels by ``1/(pixArea*N)``,
    so the discrete transform carries an extra ``1/pixArea`` relative to the
    normalized continuum kernel; multiplying by ``pixArea`` removes it.  What
    is left is a pure pixelization difference, which must vanish as the
    aperture becomes well resolved.
    """

    N_PIX = 512
    PIXEL = 0.25

    def _discrete_transform(self, R, filter_type, dr=0.75):
        """Return (k, pixArea * FFT(discrete kernel)) along one axis."""
        kern = rp.build_aperture_kernel(self.N_PIX, self.PIXEL, R,
                                        filter_type, dr)
        spec = np.fft.rfft2(kern).real * self.PIXEL ** 2
        k_axis = 2.0 * np.pi * np.fft.rfftfreq(self.N_PIX, d=self.PIXEL)
        return k_axis, spec[0, :]

    @pytest.mark.parametrize('filter_type', ['Sigma', 'DSigma'])
    @pytest.mark.parametrize('R', [2.0, 4.0])
    def test_transforms_agree_at_low_kR(self, filter_type, R):
        """Agreement must be excellent where the integral has its support.

        The kernel transform peaks at kR of order unity and decays beyond, so
        kR < 3 carries most of the amplitude for any realistic power spectrum.
        """
        k, discrete = self._discrete_transform(R, filter_type)
        analytic = kn.aperture_kernel_ft(k, R, filter_type, dr=0.75)
        band = (k > 0) & (k * R < 1.0)
        worst = np.max(np.abs(discrete[band] - analytic[band]))
        # The bound is set by the compensating annulus, not by the disk: at
        # dr = 0.75 arcmin and a 0.25 arcmin pixel the annulus spans only 3
        # pixels, and it is that thin ring the staircase resolves worst.
        # Measured here: 7.2e-3 (Sigma, R=2) down to 1.0e-3 (DSigma, R=4).
        assert worst < 1e-2, (
            f'{filter_type} at R={R}: max |discrete - analytic| = {worst:.3e} '
            f'for kR < 1'
        )

    def test_pixelization_departure_grows_with_kR(self):
        """Document where the pixelized kernel leaves the continuum one.

        A discrete aperture is a staircase, not a circle: at R=4 arcmin with
        0.25 arcmin pixels the disk holds 793 pixels against the continuum's
        804.2, a 1.4 per cent deficit.  That mismatch is invisible at low kR
        and grows into the oscillatory regime.  It sets the accuracy floor of
        the harmonic-space theory chain against the pipeline's pixelized
        measurement, so it is pinned here rather than left to be rediscovered.
        """
        k, discrete = self._discrete_transform(4.0, 'DSigma')
        analytic = kn.aperture_kernel_ft(k, 4.0, 'DSigma', dr=0.75)
        err = np.abs(discrete - analytic)
        bands = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 12.0)]
        worst = []
        for lo, hi in bands:
            sel = (k * 4.0 >= lo) & (k * 4.0 < hi) & (k > 0)
            worst.append(np.max(err[sel]) if sel.any() else np.nan)
        # Strictly increasing: the departure is a growing function of kR.
        assert all(a < b for a, b in zip(worst, worst[1:])), (
            f'Pixelization error should grow with kR, got {worst}')
        assert worst[0] < 1e-2, (
            f'Per-cent agreement expected for kR < 1, got {worst[0]:.3e}')

    def test_annulus_resolution_sets_the_agreement(self):
        """The thin compensating annulus, not the disk, limits the agreement.

        Halving the pixel doubles the number of pixels across the 0.75 arcmin
        annulus and improves the agreement roughly in proportion, in every
        filter and at every aperture tested. This is what fixes the accuracy
        floor of the theory chain: in production the pixel is 0.2 arcmin, so
        the annulus spans 3.75 pixels at every aperture, and the innermost
        aperture (R = 1 arcmin, only 5 pixels in radius) is the worst case.
        """
        for filter_type in ('Sigma', 'DSigma'):
            for R in (2.0, 4.0):
                errs = []
                for pixel in (0.25, 0.125):
                    n = 512
                    kern = rp.build_aperture_kernel(n, pixel, R, filter_type,
                                                    0.75)
                    spec = np.fft.rfft2(kern).real * pixel ** 2
                    k_axis = 2.0 * np.pi * np.fft.rfftfreq(n, d=pixel)
                    analytic = kn.aperture_kernel_ft(k_axis, R, filter_type,
                                                     dr=0.75)
                    band = (k_axis > 0) & (k_axis * R < 1.0)
                    errs.append(np.max(np.abs(spec[0, :][band]
                                              - analytic[band])))
                assert errs[1] < errs[0], (
                    f'{filter_type} at R={R}: doubling the annulus resolution '
                    f'should improve agreement, got {errs}')

    def test_agreement_improves_with_resolution(self):
        """Halving the pixel must bring the two representations closer."""
        R, filter_type = 3.0, 'DSigma'
        errors = []
        for pixel in (0.4, 0.2, 0.1):
            n = 512
            kern = rp.build_aperture_kernel(n, pixel, R, filter_type, 0.75)
            spec = np.fft.rfft2(kern).real * pixel ** 2
            k_axis = 2.0 * np.pi * np.fft.rfftfreq(n, d=pixel)
            analytic = kn.aperture_kernel_ft(k_axis, R, filter_type, dr=0.75)
            band = (k_axis > 0) & (k_axis < 2.0 / R)
            errors.append(np.max(np.abs(spec[0, :][band] - analytic[band])))
        assert errors[1] < errors[0] and errors[2] < errors[1], (
            f'Pixelization error should shrink with pixel size, got {errors}')

    def test_discrete_zero_mode_matches_the_analytic_limit(self):
        """Compensation holds exactly in the discrete kernel too."""
        for R in (1.0, 3.0, 6.0):
            kern = rp.build_aperture_kernel(self.N_PIX, self.PIXEL, R,
                                            'DSigma', 0.75)
            assert abs(kern.sum()) < 1e-12 * np.abs(kern).sum()
            kern = rp.build_aperture_kernel(self.N_PIX, self.PIXEL, R,
                                            'Sigma', 0.75)
            assert kern.sum() * self.PIXEL ** 2 == pytest.approx(1.0,
                                                                 rel=1e-12)


class TestFilteredAmplitude:
    """The k-space integral must reproduce a real-space measurement."""

    def test_matches_a_direct_map_measurement(self):
        """Y from the kernel integral must equal Y measured on a map.

        Uses a Gaussian random field with a known input power spectrum, so
        both routes describe the same field and any disagreement is a genuine
        failure of the transfer chain rather than of the field model.
        """
        n, pixel = 1024, 0.25
        rng = np.random.default_rng(20260826)

        # Build a field with a smooth, band-limited power spectrum.
        kx = 2.0 * np.pi * np.fft.fftfreq(n, d=pixel)
        ky = 2.0 * np.pi * np.fft.rfftfreq(n, d=pixel)
        kk = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
        k0 = 1.0
        p_in = 40.0 * np.exp(-0.5 * (kk / k0) ** 2)

        area = (n * pixel) ** 2
        white = np.fft.rfft2(rng.standard_normal((n, n)))
        delta = np.fft.irfft2(white * np.sqrt(p_in / area), s=(n, n))

        # Measure the map's own 2D power spectrum, then push it through the
        # analytic kernel.  Using the realized spectrum rather than the input
        # one removes sample variance from the comparison.
        spec = np.fft.rfft2(delta)
        p_meas = (np.abs(spec) ** 2) * area / (n * n) ** 2
        kbins = np.linspace(0, kk.max(), 600)
        idx = np.digitize(kk.ravel(), kbins) - 1
        counts = np.bincount(idx, minlength=len(kbins))
        sums = np.bincount(idx, weights=p_meas.ravel(), minlength=len(kbins))
        ok = counts > 0
        kcen = np.array([kk.ravel()[idx == i].mean()
                         for i in np.where(ok)[0]])
        pcen = sums[ok] / counts[ok]

        Ymat = rp.compute_Y_matrix({'a': delta}, pixel,
                                   radii=np.array([2.0, 4.0]))
        for i, R in enumerate([2.0, 4.0]):
            measured = rp.get_Y(Ymat, 'DSigma', 'a', 'a')[i]
            predicted = kn.filtered_amplitude(kcen, pcen, R, 'DSigma',
                                              dr=0.75, pixel_size=pixel)
            assert predicted == pytest.approx(measured, rel=0.05), (
                f'R={R}: kernel integral {predicted:.6e} vs map measurement '
                f'{measured:.6e}'
            )

    def test_pipeline_normalization_is_one_over_pixel_area(self):
        y = kn.normalize_to_pipeline(np.array([2.0]), 0.5)
        assert y[0] == pytest.approx(2.0 / 0.25)
        with pytest.raises(ValueError, match='pixel_size'):
            kn.normalize_to_pipeline(np.array([1.0]), 0.0)

    def test_recommended_grid_resolves_the_kernel(self):
        k = kn.recommended_k_grid(1.0, 6.0)
        assert k[0] > 0 and np.all(np.diff(k) > 0)
        # At least a few samples per oscillation of the largest aperture.
        assert np.max(np.diff(k)) < 2.0 * np.pi / 6.0 / 8.0
        with pytest.raises(ValueError, match='R_min'):
            kn.recommended_k_grid(6.0, 1.0)


class TestBinAveraging:

    def test_bin_average_of_a_narrow_bin_matches_the_point_kernel(self):
        k = np.linspace(0.0, 3.0, 128)
        point = kn.aperture_kernel_ft(k, 3.0, 'DSigma', 0.75)
        binned = kn.bin_averaged_kernel_ft(k, 2.999, 3.001, 'DSigma', 0.75)
        assert np.allclose(point, binned, atol=1e-6)

    def test_bin_average_differs_for_a_wide_bin(self):
        k = np.linspace(0.0, 3.0, 128)
        point = kn.aperture_kernel_ft(k, 3.0, 'DSigma', 0.75)
        binned = kn.bin_averaged_kernel_ft(k, 2.0, 4.0, 'DSigma', 0.75)
        assert not np.allclose(point, binned, atol=1e-3)

    def test_rejects_inverted_bins(self):
        with pytest.raises(ValueError, match='R_lo'):
            kn.bin_averaged_kernel_ft(np.array([1.0]), 3.0, 2.0, 'DSigma')


class TestTheoryPower:
    """Guards on the halofit call in :mod:`theory`."""

    def test_requested_redshift_is_the_one_returned(self):
        """CAMB orders its redshift axis increasing, whatever order was asked.

        Taking ``pk[0]`` therefore returns z=0 rather than the requested
        redshift, which at z=0.5 overstates the power by a factor of about
        2.3.  That is large enough to masquerade as a halofit failure, so the
        selection is pinned here: the z=0.5 spectrum must be well below the
        z=0 one at fixed k.
        """
        import theory as th
        k_half, p_half = th.halofit_power(
            h=0.6774, omega_m=0.3089, omega_b=0.0486, z=0.5,
            n_s=0.9667, sigma8=0.8159, n_k=200, k_max=10.0)
        k_zero, p_zero = th.halofit_power(
            h=0.6774, omega_m=0.3089, omega_b=0.0486, z=0.0,
            n_s=0.9667, sigma8=0.8159, n_k=200, k_max=10.0)
        ratio = np.interp(1.0, k_half, p_half) / np.interp(1.0, k_zero, p_zero)
        assert 0.3 < ratio < 0.7, (
            f'P(k=1, z=0.5)/P(k=1, z=0) = {ratio:.3f}; a value near 1 means '
            f'the redshift slice was not selected correctly')

    def test_sigma8_normalization_hits_the_target(self):
        """The returned spectrum must actually have the requested sigma8.

        A post-hoc rescaling of the halofit output would pass a simple ratio
        test by construction, since both calls would share one underlying
        amplitude. This instead checks the normalization against an
        independent linear calculation, so a regression to post-hoc scaling
        would be caught.
        """
        import theory as th
        target = 0.75
        k, p = th.halofit_power(h=0.6774, omega_m=0.3089, omega_b=0.0486,
                                z=0.0, n_s=0.9667, sigma8=target, n_k=600,
                                k_max=20.0, non_linear=False)
        # sigma8 is the rms in 8 Mpc/h spheres of the LINEAR field.
        x = k * 8.0
        window = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
        var = np.trapz(k ** 2 * p * window ** 2, k) / (2.0 * np.pi ** 2)
        assert np.sqrt(var) == pytest.approx(target, rel=2e-2), (
            f'requested sigma8={target}, spectrum integrates to '
            f'{np.sqrt(var):.4f}')

    def test_quadrature_is_converged(self):
        """The amplitude must not depend on how finely CAMB sampled k.

        The aperture kernel oscillates on a fixed period in k, which CAMB's
        log-spaced grid under-samples at high k; amplitudes_from_p2d resamples
        onto a linear grid to remove that sensitivity. Without the resampling
        the largest aperture drifts by a few per mille between n_k=800 and
        n_k=1600, which would masquerade as model error.
        """
        import theory as th
        out = []
        for n_k in (400, 1600):
            k, p3 = th.halofit_power(h=0.6774, omega_m=0.3089, omega_b=0.0486,
                                     z=0.5, n_s=0.9667, sigma8=0.8159,
                                     n_k=n_k, k_max=30.0)
            p2 = th.project_periodic_box(p3, 205.0)
            out.append(th.amplitudes_from_p2d(k, p2, [0.4, 2.3], 'DSigma',
                                              0.29, 0.4))
        rel = np.abs(out[1] / out[0] - 1.0)
        assert np.all(rel < 1e-3), (
            f'Amplitude still depends on the CAMB sampling: {rel}')

    def test_box_projection_divides_by_the_depth(self):
        import theory as th
        p = np.array([100.0, 50.0])
        assert np.allclose(th.project_periodic_box(p, 205.0), p / 205.0)
        with pytest.raises(ValueError, match='box_mpc_h'):
            th.project_periodic_box(p, 0.0)

    def test_cosmology_lookup_refuses_unknown_suites(self):
        import theory as th
        header = {'HubbleParam': 0.7, 'Omega0': 0.3, 'OmegaBaryon': 0.05}
        with pytest.raises(KeyError, match='n_s/sigma8'):
            th.cosmology_for('NotASuite', header)
        got = th.cosmology_for('IllustrisTNG', header)
        assert got['h'] == 0.7 and 'sigma8' in got
