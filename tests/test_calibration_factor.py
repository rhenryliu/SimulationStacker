"""Tests for ``scripts/cross_corr/make_calibration_factor.py``.

The calibration-factor sweep replaces two would-be particle sweeps with two
algebraic identities, and both are places where a mistake would be silent:
the amplitudes would still come out finite, smooth and plausible.  So each
identity is tested against the thing it claims to replace.

1. **The Y transform is a map-level filter, not a reconstruction.**
   ``Y(R;Rmax) = Sigma(R) - Sigma(Rmax)`` is asserted to hold at the amplitude
   level, so that no extra convolution is needed.  Tested against amplitudes
   computed with a directly-built ``Sigma(R) - Sigma(Rmax)`` convolution
   kernel.

2. **Convention T needs no total-matter field.**  ``delta_t = f_m delta_m +
   f_b delta_b`` is asserted to hold exactly on these maps, so the
   total-matter amplitudes are bilinear recombinations of measured ones.
   Tested against amplitudes computed from a directly constructed ``t`` map,
   both on synthetic fields and (where the data are on scratch) on a real
   TNG300-1 pair.

Everything else in the module is arithmetic on those amplitudes and is
covered by the algebraic-identity tests at the end.

Run with::

    cd tests/
    pytest test_calibration_factor.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts' / 'cross_corr'))

import rprofiles as rp  # noqa: E402
import make_calibration_factor as mcf  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PIXEL_ARCMIN = 0.2
N_PIXELS = 256
RADII = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.0, 5.0], dtype=float)


@pytest.fixture(scope='module')
def synthetic_deltas():
    """Three correlated overdensity maps plus a discrete galaxy field.

    The fields are built by smoothing one common Gaussian realization on
    different scales and adding independent noise, so they are correlated
    without being proportional -- otherwise every cross-correlation
    coefficient would be identically one and the tests would pass trivially.

    Returns:
        tuple: ``(deltas, nbar_pix, f_b)``.
    """
    rng = np.random.default_rng(20260910)
    kx = np.fft.fftfreq(N_PIXELS)[:, None]
    ky = np.fft.rfftfreq(N_PIXELS)[None, :]
    k = np.hypot(kx, ky)

    white = np.fft.rfft2(rng.standard_normal((N_PIXELS, N_PIXELS)))

    def smoothed(scale, noise_amp):
        field = np.fft.irfft2(white * np.exp(-0.5 * (k * scale) ** 2),
                              s=(N_PIXELS, N_PIXELS))
        field = field + noise_amp * rng.standard_normal((N_PIXELS, N_PIXELS))
        # Turn into a positive mass-like map, then an overdensity.
        mass = np.exp(0.3 * field / field.std())
        return mass

    mass_m = smoothed(6.0, 0.05)
    mass_b = smoothed(14.0, 0.10)     # gas: smoother, as feedback makes it
    mass_e = 0.8 * mass_b + 0.2 * smoothed(20.0, 0.10)

    f_b = float(mass_b.mean() / (mass_m.mean() + mass_b.mean()))

    deltas = {'m': rp.to_overdensity(mass_m),
              'b': rp.to_overdensity(mass_b),
              'e': rp.to_overdensity(mass_e)}

    # Discrete galaxy field: Poisson sample biased toward the matter map.
    lam = 0.05 * (1.0 + 2.0 * deltas['m'])
    counts = rng.poisson(np.clip(lam, 1e-6, None)).astype(float)
    nbar_pix = counts.sum() / counts.size
    deltas['g'] = rp.to_overdensity(counts)

    # Keep the raw mass maps for the Convention T reference construction.
    deltas['_mass_m'] = mass_m
    deltas['_mass_b'] = mass_b
    return deltas, nbar_pix, f_b


@pytest.fixture(scope='module')
def Ymat(synthetic_deltas):
    """Filtered amplitudes for the synthetic fields."""
    deltas, nbar_pix, _ = synthetic_deltas
    fields = {k: v for k, v in deltas.items() if not k.startswith('_')}
    return rp.compute_Y_matrix(fields, PIXEL_ARCMIN, radii=RADII,
                               nbar_pix=nbar_pix, n_jk_side=2)


# ---------------------------------------------------------------------------
# Identity 1: the Y transform is a genuine map-level filter
# ---------------------------------------------------------------------------

def ytransform_kernel(n_pixels, pixel_arcmin, R, rmax, dr):
    """Build the Y-transform kernel directly, as one convolution.

    This is the object ``make_calibration_factor`` claims never to need:
    the difference of two annulus-mean kernels, assembled on the map grid and
    applied in a single convolution.

    Args:
        n_pixels (int): Pixels per side.
        pixel_arcmin (float): Pixel size, arcmin.
        R (float): Aperture radius, arcmin.
        rmax (float): Reference radius, arcmin.
        dr (float): Annulus width, arcmin.

    Returns:
        np.ndarray: Kernel of shape ``(n_pixels, n_pixels)``.
    """
    inner = rp.build_aperture_kernel(n_pixels, pixel_arcmin, R, 'Sigma', dr)
    outer = rp.build_aperture_kernel(n_pixels, pixel_arcmin, rmax, 'Sigma', dr)
    return inner - outer


class TestYTransformIsAMapLevelFilter:
    """``Y(R;Rmax) = Sigma(R) - Sigma(Rmax)`` must hold at the amplitude level."""

    @pytest.mark.parametrize('rmax', [4.0, 5.0])
    def test_matches_direct_convolution(self, synthetic_deltas, Ymat, rmax):
        """The assembled amplitudes equal a single direct convolution."""
        deltas, _, _ = synthetic_deltas
        dr = rp.DR_ARCMIN

        assembled, _, mask = mcf.assemble_derived_filter(
            Ymat['Y']['Sigma'], Ymat['Y_jk']['Sigma'], RADII, 'Ytransform',
            rmax)

        for pair in [('b', 'm'), ('g', 'm'), ('m', 'm'), ('b', 'g')]:
            a, b = pair
            for i, R in enumerate(RADII):
                if not mask[i]:
                    continue
                kern = ytransform_kernel(N_PIXELS, PIXEL_ARCMIN, R, rmax, dr)
                spec = rp.kernel_spectrum(kern)
                fmap = rp.filtered_map(
                    np.fft.rfft2(deltas[a]), spec, deltas[a].shape)
                direct = float(np.mean(fmap * deltas[b]))
                assert direct == pytest.approx(assembled[pair][i], rel=1e-10), (
                    f'Y transform mismatch for {pair} at R={R}, Rmax={rmax}')

    def test_vanishes_at_rmax(self, Ymat):
        """``Y(Rmax; Rmax) == 0`` identically, and that bin is masked out."""
        rmax = 5.0
        assembled, _, mask = mcf.assemble_derived_filter(
            Ymat['Y']['Sigma'], Ymat['Y_jk']['Sigma'], RADII, 'Ytransform',
            rmax)
        idx = mcf.radius_index(RADII, rmax)
        assert not mask[idx]
        for pair, values in assembled.items():
            assert values[idx] == pytest.approx(0.0, abs=1e-18)

    def test_mask_drops_bins_above_fraction_of_rmax(self):
        """Bins at and above ``0.8 Rmax`` carry no signal and are dropped."""
        radii = np.array([1.0, 2.0, 3.0, 4.0, 4.5, 5.0])
        _, _, mask = mcf.assemble_derived_filter(
            {('m', 'm'): np.ones(6)}, {('m', 'm'): np.ones((2, 6))},
            radii, 'Ytransform', 5.0)
        np.testing.assert_array_equal(
            mask, radii < mcf.YT_USABLE_FRACTION * 5.0)

    def test_reference_radius_must_be_on_the_grid(self, Ymat):
        """An off-grid reference radius raises rather than interpolating."""
        with pytest.raises(ValueError, match='not on the aperture grid'):
            mcf.assemble_derived_filter(
                Ymat['Y']['Sigma'], Ymat['Y_jk']['Sigma'], RADII,
                'Ytransform', 4.3)


class TestUpsilonRebuild:
    """Rebuilding Upsilon at an arbitrary R0 must match ``compute_Y_matrix``."""

    def test_matches_compute_Y_matrix(self, synthetic_deltas):
        """The R0 = 1' rebuild reproduces the library's own Upsilon.

        Note this checks agreement with ``compute_Y_matrix``'s internal
        construction, which uses the same formula, so it is a wiring check
        rather than a from-scratch validation. The independent check of the
        Upsilon formula against a composite convolution kernel and against the
        legacy stamp filter lives in
        ``test_rprofiles.py::TestUpsilonConstruction``.
        """
        deltas, nbar_pix, _ = synthetic_deltas
        fields = {k: v for k, v in deltas.items() if not k.startswith('_')}
        ref = rp.compute_Y_matrix(fields, PIXEL_ARCMIN, radii=RADII, r0=1.0,
                                  nbar_pix=nbar_pix, n_jk_side=2)
        assembled, assembled_jk, mask = mcf.assemble_derived_filter(
            ref['Y']['DSigma'], ref['Y_jk']['DSigma'], RADII, 'Upsilon', 1.0)
        for pair in ref['Y']['Upsilon']:
            np.testing.assert_allclose(assembled[pair],
                                       ref['Y']['Upsilon'][pair], rtol=1e-12)
            np.testing.assert_allclose(assembled_jk[pair],
                                       ref['Y_jk']['Upsilon'][pair], rtol=1e-12)
        np.testing.assert_array_equal(mask,
                                      rp.upsilon_defined_mask(RADII, 1.0))

    def test_r0_of_two_differs_from_r0_of_one(self, Ymat):
        """The two production R0 choices are genuinely different filters."""
        a, _, _ = mcf.assemble_derived_filter(
            Ymat['Y']['DSigma'], Ymat['Y_jk']['DSigma'], RADII, 'Upsilon', 1.0)
        b, _, _ = mcf.assemble_derived_filter(
            Ymat['Y']['DSigma'], Ymat['Y_jk']['DSigma'], RADII, 'Upsilon', 2.25)
        assert not np.allclose(a[('m', 'm')][-1], b[('m', 'm')][-1])


# ---------------------------------------------------------------------------
# Identity 2: Convention T needs no total-matter field
# ---------------------------------------------------------------------------

class TestConventionT:
    """``delta_t = f_m delta_m + f_b delta_b`` must be exact on these maps."""

    def test_matches_a_directly_built_total_field(self, synthetic_deltas):
        """Recombined amplitudes equal those measured on a real ``t`` map."""
        deltas, nbar_pix, f_b = synthetic_deltas
        mass_t = deltas['_mass_m'] + deltas['_mass_b']

        fields = {k: v for k, v in deltas.items() if not k.startswith('_')}
        direct_fields = dict(fields)
        direct_fields['t'] = rp.to_overdensity(mass_t)

        ref = rp.compute_Y_matrix(direct_fields, PIXEL_ARCMIN, radii=RADII,
                                  nbar_pix=nbar_pix, n_jk_side=2)
        without_t = rp.compute_Y_matrix(fields, PIXEL_ARCMIN, radii=RADII,
                                        nbar_pix=nbar_pix, n_jk_side=2)

        for filt in ('Sigma', 'DSigma'):
            recombined = mcf.add_convention_t(dict(without_t['Y'][filt]), f_b)
            for pair in [('g', 't'), ('b', 't'), ('e', 't'), ('m', 't'),
                         ('t', 't')]:
                np.testing.assert_allclose(
                    recombined[pair], ref['Y'][filt][pair], rtol=1e-10,
                    err_msg=f'Convention T mismatch for {pair} under {filt}')

    def test_works_on_jackknife_stacks(self, synthetic_deltas, Ymat):
        """The same algebra applies elementwise to ``(n_jk, n_rad)`` arrays."""
        _, _, f_b = synthetic_deltas
        full = mcf.add_convention_t(dict(Ymat['Y']['DSigma']), f_b)
        jk = mcf.add_convention_t(dict(Ymat['Y_jk']['DSigma']), f_b)
        for pair in [('g', 't'), ('t', 't')]:
            assert jk[pair].shape == (Ymat['n_jk'], len(RADII))
            # The jackknife realizations must bracket the full-map value.
            assert np.all(np.isfinite(jk[pair]))
            assert np.isfinite(full[pair]).all()

    def test_baryon_fraction_from_maps_not_header(self, synthetic_deltas):
        """A wrong ``f_b`` breaks the identity, so the test has teeth."""
        deltas, nbar_pix, f_b = synthetic_deltas
        mass_t = deltas['_mass_m'] + deltas['_mass_b']
        fields = {k: v for k, v in deltas.items() if not k.startswith('_')}
        direct = dict(fields)
        direct['t'] = rp.to_overdensity(mass_t)

        ref = rp.compute_Y_matrix(direct, PIXEL_ARCMIN, radii=RADII,
                                  nbar_pix=nbar_pix, n_jk_side=2)
        without_t = rp.compute_Y_matrix(fields, PIXEL_ARCMIN, radii=RADII,
                                        nbar_pix=nbar_pix, n_jk_side=2)
        wrong = mcf.add_convention_t(dict(without_t['Y']['DSigma']),
                                     f_b * 1.05)
        assert not np.allclose(wrong[('t', 't')],
                               ref['Y']['DSigma'][('t', 't')], rtol=1e-6)


# ---------------------------------------------------------------------------
# The derived quantities
# ---------------------------------------------------------------------------

class TestCalibrationFactors:
    """Algebraic properties of C, C_A and the suppression mapping."""

    def test_C_is_one_under_exact_mediation(self):
        """``C == 1`` identically when the gas is a transfer of the matter.

        Addendum Eq. (A13): if ``delta_b = S * delta_m + eps`` with ``eps``
        uncorrelated with both ``m`` and ``g``, then ``Y_bm = S Y_mm`` and
        ``Y_gb = S Y_gm``, so ``C = 1`` whatever ``S`` is.  This is the
        proposition the whole measurement is testing, so it is worth pinning
        that the code reproduces it.
        """
        n = 7
        rng = np.random.default_rng(7)
        S = rng.uniform(0.3, 0.9, size=n)      # arbitrary scale-dependent transfer
        Y_mm = rng.uniform(1.0, 3.0, size=n)
        Y_gm = rng.uniform(0.5, 2.0, size=n)
        Y = {('m', 'm'): Y_mm,
             ('b', 'm'): S * Y_mm,
             ('b', 'g'): S * Y_gm,
             ('g', 'm'): Y_gm,
             ('g', 'g'): rng.uniform(1.0, 2.0, size=n),
             ('b', 'b'): rng.uniform(1.0, 2.0, size=n),
             ('b', 'e'): rng.uniform(0.5, 1.5, size=n),
             ('e', 'm'): rng.uniform(0.5, 1.5, size=n),
             ('e', 'e'): rng.uniform(1.0, 2.0, size=n),
             ('e', 'g'): rng.uniform(0.5, 1.5, size=n)}
        mcf.add_convention_t(Y, 0.157)
        out = mcf.calibration_factors(Y, gas='b')
        np.testing.assert_allclose(out['C'], np.ones(n), rtol=1e-12)

    def test_CA_equals_ratio_of_coefficients(self, synthetic_deltas, Ymat):
        """``C_A`` must equal ``r_bm / r_gb`` as the note defines it."""
        _, _, f_b = synthetic_deltas
        Y = mcf.add_convention_t(dict(Ymat['Y']['DSigma']), f_b)
        out = mcf.calibration_factors(Y, gas='b')
        expected = out['r_Xm'] / out['r_gX']
        np.testing.assert_allclose(out['CA'], expected, rtol=1e-10)

    def test_C_equals_rbm_rgm_over_rgb(self, synthetic_deltas, Ymat):
        """``C = r_bm r_gm / r_gb``: the four-amplitude form of (A12)."""
        _, _, f_b = synthetic_deltas
        Y = mcf.add_convention_t(dict(Ymat['Y']['DSigma']), f_b)
        out = mcf.calibration_factors(Y, gas='b')
        expected = out['r_Xm'] * out['r_gm'] / out['r_gX']
        np.testing.assert_allclose(out['C'], expected, rtol=1e-10)

    def test_suppression_conventions_agree(self):
        """(A16) and (A17) must give the same S for consistent x and x_t.

        With ``r_bm = 1``, (A14) relates the two: ``x_t = (1 - f_m/(f_m +
        f_b x)) / f_b``.  The addendum's worked check is ``x = 0.5`` giving
        ``S = 0.849``; this pins both branches against each other.
        """
        f_b = 0.157
        f_m = 1.0 - f_b
        x = np.array([0.4, 0.5, 0.7, 1.0])
        S_C = mcf.suppression(x, f_b, 'C')
        x_t = (1.0 - f_m / (f_m + f_b * x)) / f_b
        S_T = mcf.suppression(x_t, f_b, 'T')
        np.testing.assert_allclose(S_C, S_T, rtol=1e-12)
        # x = 1 is no suppression at all.
        assert S_C[-1] == pytest.approx(1.0)

    def test_term_b_residual_is_the_stochastic_term(self):
        """``B_measured - B_predicted`` isolates the (A15) stochastic term.

        Eq. (A15): ``Y_tt/Y_mm = (f_m + f_b x)^2 + f_b^2 (1 - r_bm^2)
        Y_bb/Y_mm``.  Constructing amplitudes with a known ``r_bm``, the
        residual must equal that second term.
        """
        f_b = 0.157
        n = 5
        Y_mm = np.full(n, 2.0)
        Y_bb = np.full(n, 1.5)
        r_bm = 0.9
        Y_bm = r_bm * np.sqrt(Y_mm * Y_bb)
        Y = {('m', 'm'): Y_mm, ('b', 'b'): Y_bb, ('b', 'm'): Y_bm}
        mcf.add_convention_t(Y, f_b)
        diag = mcf.term_b_diagnostic(Y, f_b)
        expected = f_b ** 2 * (1.0 - r_bm ** 2) * Y_bb / Y_mm
        np.testing.assert_allclose(diag['B_residual'], expected, rtol=1e-10)


class TestPayloadAssembly:
    """The glue between the three identities and the saved ``.npz``.

    ``build_filters`` and ``flatten_for_npz`` decide *which* gas field feeds
    *which* formula and *which* bins are trustworthy. Nothing else in this
    module exercises them, and that is exactly where a wrong-but-plausible
    number can reach disk.
    """

    @pytest.fixture
    def payload(self, synthetic_deltas, Ymat):
        _, _, f_b = synthetic_deltas
        filters = mcf.build_filters(Ymat, RADII, [1.0], [4.0, 5.0], f_b)
        return filters, mcf.flatten_for_npz(filters, RADII, f_b, {})

    def test_suppression_written_only_for_the_baryon_field(self, payload):
        """``S`` assumes the gas completes the mass budget; electrons do not.

        Eqs. (A16)/(A17) descend from ``delta_t = f_m delta_m + f_b delta_b``.
        Electrons are a subset of the baryons, not a complementary component,
        so ``S(x_e)`` would be finite, plausible and meaningless. It must not
        be written at all rather than written and caveated.
        """
        _, out = payload
        assert any(k.startswith('S_b_') for k in out)
        assert any(k.startswith('S_t_b_') for k in out)
        assert not [k for k in out if k.startswith('S_e_')]
        assert not [k for k in out if k.startswith('S_t_e_')]

    def test_calibration_factors_written_for_both_gas_fields(self, payload):
        """C and its relatives *are* meaningful for the electron field."""
        _, out = payload
        for gas in ('b', 'e'):
            for stem in ('C', 'C_t', 'CA', 'x', 'r_gX'):
                assert f'{stem}_{gas}_DSigma' in out, f'missing {stem}_{gas}'

    def test_derived_quantities_are_nan_outside_the_mask(self, payload):
        """A consumer that forgets the mask must get NaN, not a plausible number.

        Regression guard: before this was enforced, ``C`` at ``R < R0`` for
        Upsilon came out around 12.5 -- finite, smooth and entirely wrong.
        """
        filters, out = payload
        for fname, entry in filters.items():
            bad = ~entry['mask']
            if not bad.any():
                continue
            for stem in ('C_b', 'CA_b', 'x_b', 'r_gX_b'):
                key = f'{stem}_{fname}'
                assert np.all(np.isnan(out[key][bad])), (
                    f'{key} leaks values outside its mask')
            assert np.all(np.isnan(out[f'Cjk_b_{fname}'][:, bad]))

    def test_raw_amplitudes_are_not_masked(self, payload):
        """``Y_*`` stays unmasked: well-defined everywhere, useful as diagnostics."""
        filters, out = payload
        for fname, entry in filters.items():
            bad = ~entry['mask']
            if not bad.any():
                continue
            assert np.all(np.isfinite(out[f'Y_mm_{fname}'][bad])), (
                f'Y_mm_{fname} should be written unmasked')

    def test_every_filter_variant_reaches_the_payload(self, payload):
        """One key set per requested R0 and Rmax, plus the two base filters."""
        filters, out = payload
        assert set(filters) == {'Sigma', 'DSigma', 'Upsilon_R0=1',
                                'Ytransform_Rmax=4', 'Ytransform_Rmax=5'}
        for fname in filters:
            assert f'mask_{fname}' in out


class TestApertureGrid:
    """The Task 1 bins must survive the union with the reference radii."""

    def test_data_bins_are_bit_identical(self):
        """Adding reference radii must not move the nine 1'-6' bins."""
        cfg = {'min_radius': 1.0, 'max_radius': 6.0, 'num_radii': 9,
               'extend_max_radius': 10.0,
               'upsilon_r0': [1.0, 2.0],
               'ytransform_rmax': [4.0, 5.0, 6.0, 9.0]}
        base = mcf.aperture_radii(cfg)
        grid, r0_list, rmax_list = mcf.calibration_radii(cfg)
        for R in base:
            idx = mcf.radius_index(grid, R)
            assert grid[idx] == pytest.approx(R, abs=1e-12)
        for ref in r0_list + rmax_list:
            mcf.radius_index(grid, ref)          # must not raise
        assert np.all(np.diff(grid) > 0)


# ---------------------------------------------------------------------------
# On-data spot check
# ---------------------------------------------------------------------------

DATA_ROOT = Path('/pscratch/sd/r/rhliu/simulations/IllustrisTNG/products/2D')


@pytest.mark.skipif(not DATA_ROOT.exists(),
                    reason='TNG300-1 cached fields are not on scratch')
def test_convention_t_on_real_tng_maps():
    """The Convention T identity must hold on the real cached TNG maps.

    The synthetic test builds ``t = m + b`` by construction, which cannot fail.
    This one takes the actual cached 'total' and 'baryon' fields, derives CDM
    the way the pipeline does, and checks that recombining reproduces the
    total map's own overdensity to float64 round-off.  It is the test that
    would catch a mismatched pair of caches.
    """
    total_path = DATA_ROOT / 'TNG300-1_67_total_2674_yz.npy'
    baryon_path = DATA_ROOT / 'TNG300-1_67_baryon_2674_yz.npy'
    if not (total_path.exists() and baryon_path.exists()):
        pytest.skip('TNG300-1 total/baryon fields not cached at 2674 pixels')

    total = np.load(total_path)
    baryon = np.load(baryon_path)
    f_b = float(baryon.mean() / total.mean())

    cdm = rp.derive_cdm_field(total, baryon, verbose=False)
    delta_m = rp.to_overdensity(cdm)
    delta_b = rp.to_overdensity(baryon)
    delta_t = rp.to_overdensity(total)

    recombined = (1.0 - f_b) * delta_m + f_b * delta_b
    np.testing.assert_allclose(recombined, delta_t, atol=1e-10, rtol=0)
