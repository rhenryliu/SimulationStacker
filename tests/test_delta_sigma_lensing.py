"""Known-answer tests for the lensing-bin ΔΣ filter (filters.delta_sigma_lensing_weights and
SimulationStacker.stack_on_array(filterType='DSigma_lensing')), and for its Σ counterpart
(filters.sigma_lensing_weights, filterType='Sigma_lensing') at the end of the file.

The reference is a 2D Gaussian surface density, whose area-weighted ΔΣ over a bin [a, b] is
exact: with x = R^2 / 2 s^2 and F(x) = Ein(x) + e^-x (Ein(x) = E1(x) + ln x + γ),

    <ΔΣ>_[a,b] = M_tot [F(x_b) - F(x_a)] / (π (b^2 - a^2)).

The maps hold the exact mass per pixel (pixel-integrated Gaussian), as binned particle maps do.
The only approximation left in the filter is treating that mass as uniform over each pixel, so
the error falls as the pixel size squared.
"""

import os
import sys

import numpy as np
import pytest
from scipy.special import erf, exp1

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import filters  # noqa: E402
from utils import bins_from_geomean_monotonic  # noqa: E402

EDGES = bins_from_geomean_monotonic(np.linspace(1.0, 6.0, 9), pick='geom')[0]


def sig(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def exact_bin_average(a, b, s):
    """Area-weighted mean of ΔΣ over [a, b] for a unit-mass Gaussian of width s."""
    def F(r):
        x = r**2 / (2 * s**2)
        return exp1(x) + np.log(x) + np.euler_gamma + np.exp(-x)
    return (F(b) - F(a)) / (np.pi * (b**2 - a**2))


def pixel_mass(coords_row, coords_col, centre, pix, s):
    """Exact mass of a unit-mass Gaussian in each pixel of a grid (pixel centres given)."""
    def frac(t, t0):
        c = np.sqrt(2) * s
        return 0.5 * (erf((t + pix / 2 - t0) / c) - erf((t - pix / 2 - t0) / c))
    return np.outer(frac(coords_row, centre[0]), frac(coords_col, centre[1]))


def lensing_profile(W, mass):
    return np.tensordot(W, mass, axes=([1, 2], [0, 1]))


def test_pixel_overlaps_tile_the_disk():
    pix, half = 0.2, 40
    c = np.arange(-half, half + 1) * pix
    x0 = np.repeat(c - pix / 2 - 0.07, c.size)
    y0 = np.tile(c - pix / 2 + 0.03, c.size)
    for R in (0.05, 0.33, 1.0, 6.3):
        total = filters.pixel_disk_overlap(x0, x0 + pix, y0, y0 + pix, R).sum()
        assert total == pytest.approx(np.pi * R**2, rel=1e-12)


def test_uniform_sheet_cancels():
    """Compensation: a uniform background contributes nothing, for any halo offset."""
    for pix, half, off in ((0.2, 35, (-0.13, -0.05)), (0.5, 15, (-0.4, -0.01))):
        W = filters.delta_sigma_lensing_weights(half, pix, EDGES, offset=off)
        sheet = W.sum(axis=(1, 2)) * pix**2
        scale = np.abs(W).sum(axis=(1, 2)) * pix**2
        assert np.all(np.abs(sheet) <= 1e-12 * scale)


def test_point_mass_is_exact():
    """Mass in a pixel that lies inside every bin's inner edge: <ΔΣ> = 2 ln(b/a) / (π (b²-a²))."""
    half, pix = 35, 0.2
    W = filters.delta_sigma_lensing_weights(half, pix, EDGES, offset=(-0.1, -0.1))
    m = np.zeros((2 * half + 1,) * 2)
    m[half, half] = 1.0
    a, b = EDGES[:-1], EDGES[1:]
    np.testing.assert_allclose(lensing_profile(W, m), 2 * np.log(b / a) / (np.pi * (b**2 - a**2)),
                               rtol=1e-12)


@pytest.mark.parametrize('fwhm', [0.5, 1.2, 2.5, 6.0])
def test_gaussian_converges_with_pixel_size(fwhm):
    s = sig(fwhm)
    exact = np.array([exact_bin_average(a, b, s) for a, b in zip(EDGES[:-1], EDGES[1:])])
    errors = []
    for pix in (0.2, 0.1, 0.05):
        half = int(np.ceil(7.0 / pix))
        off = (-0.37 * pix, -0.81 * pix)
        c = np.arange(-half, half + 1) * pix
        W = filters.delta_sigma_lensing_weights(half, pix, EDGES, offset=off)
        errors.append(np.max(np.abs(lensing_profile(W, pixel_mass(c, c, off, pix, s)) / exact - 1)))
    assert errors[0] < 0.03                          # 0.2' maps: within 3 per cent
    assert errors[2] < 0.002                         # 0.05': within 0.2 per cent
    assert errors[2] < errors[1] < errors[0]         # and converging


def test_quadrature_is_converged():
    W256 = filters.delta_sigma_lensing_weights(35, 0.2, EDGES, offset=(-0.1, -0.15))
    W1024 = filters.delta_sigma_lensing_weights(35, 0.2, EDGES, offset=(-0.1, -0.15), n_quad=1024)
    assert np.abs(W256 - W1024).max() <= 1e-5 * np.abs(W1024).max()


def test_cutout_too_small_raises():
    with pytest.raises(ValueError):
        filters.delta_sigma_lensing_weights(30, 0.2, EDGES, offset=(-0.19, 0.0))   # reach 5.91 < 6.34


def test_map_pixel_convention():
    """The centring in stack_on_array assumes mapMaker's binning, binned_statistic_2d over
    [0, BoxSize] (mapMaker.py), in which pixel i spans [i d, (i+1) d), first coordinate on rows."""
    from scipy.stats import binned_statistic_2d
    box, n = 10.0, 7
    pts = np.random.default_rng(3).uniform(0, box, (2, 500))
    binned = binned_statistic_2d(pts[0], pts[1], np.ones(pts.shape[1]), statistic='sum',
                                 bins=n, range=[[0, box], [0, box]]).statistic
    idx = np.floor(pts / (box / n)).astype(int)
    expected = np.zeros((n, n))
    np.add.at(expected, (idx[0], idx[1]), 1.0)
    np.testing.assert_array_equal(binned, expected)


def _stacker_on_gaussians(units):
    """A stub SimulationStacker and a periodic map of nine unit-mass Gaussians (FWHM 2.5 radius
    units) at random sub-pixel positions, with exact mass per pixel; 0.1-unit pixels."""
    import astropy.units as u
    import stacker as stk
    from astropy.cosmology import FlatLambdaCDM
    from utils import comoving_to_arcmin

    s_obj = stk.SimulationStacker.__new__(stk.SimulationStacker)
    if units == 'kpc/h':
        box = 400.0
        s_obj.header = {'BoxSize': box}
        size = box                                   # map side in radius units
    else:                                            # arcmin, as stackMap uses
        box, h, om, z = 40000.0, 0.7, 0.3, 0.5
        s_obj.header = {'BoxSize': box, 'HubbleParam': h, 'Omega0': om}
        s_obj.z = z
        cosmo = FlatLambdaCDM(H0=100 * h, Om0=om, Tcmb0=2.7255 * u.K)   # as stack_on_array
        size = comoving_to_arcmin(box, z, cosmo=cosmo)
    npix = int(np.ceil(size / 0.1))
    pix = size / npix
    rng = np.random.default_rng(7)
    s = sig(2.5)
    frac = np.linspace(0.1, 0.9, 3)
    pos = np.array([(fy, fz) for fy in frac for fz in frac]) * size + rng.uniform(0, 5 * pix, (9, 2))
    centres = (np.arange(npix) + 0.5) * pix
    array = np.zeros((npix, npix))
    for y, z in pos:                                 # each Gaussian painted near its halo
        rows = (int(y / pix) + np.arange(-200, 201)) % npix
        cols = (int(z / pix) + np.arange(-200, 201)) % npix
        array[np.ix_(rows, cols)] += pixel_mass(centres[rows], centres[cols], (y, z), pix, s)
    pos3d = np.column_stack([np.zeros(9), pos * box / size])   # halo positions in box units
    s_obj.loadHalos = lambda: {'GroupMass': np.ones(9), 'GroupPos': pos3d}
    return s_obj, array, s


@pytest.mark.parametrize('units', ['kpc/h', 'arcmin'])
def test_stack_on_array_centres_haloes_correctly(units):
    """End to end through stack_on_array. A wrong offset sign or a swapped axis would shift
    every stack centre by up to two pixels and fail the tolerance."""
    s_obj, array, s = _stacker_on_gaussians(units)
    radii, prof = s_obj.stack_on_array(array, filterType='DSigma_lensing', minRadius=1.0,
                                       maxRadius=6.0, numRadii=9, projection='yz',
                                       radDistance=1.0, radDistanceUnits=units,
                                       halo_mask=np.arange(9))
    exact = np.array([exact_bin_average(a, b, s) for a, b in zip(EDGES[:-1], EDGES[1:])])
    assert prof.shape == (9, 9)
    # 0.1 pixels: the uniform-pixel error is <= 0.25 per cent (FWHM 2.5); offset classes add
    # a residual miscentring of at most 1/8 pixel.
    assert np.max(np.abs(prof / exact[:, None] - 1)) < 0.004


def test_stack_on_array_rejects_degenerate_radius_grid():
    s_obj, array, _ = _stacker_on_gaussians('kpc/h')
    with pytest.raises(ValueError):
        s_obj.stack_on_array(array, filterType='DSigma_lensing', minRadius=1.0, maxRadius=6.0,
                             numRadii=2, projection='yz', radDistance=1.0,
                             radDistanceUnits='kpc/h', halo_mask=np.arange(9))


# ---------------------------------------------------------------------------
# 'Sigma_lensing': the area-weighted mean of Σ over each lensing bin. For a unit-mass
# Gaussian, M(<R) = 1 - exp(-R^2 / 2 s^2), so
#     <Σ>_[a,b] = [exp(-a^2 / 2 s^2) - exp(-b^2 / 2 s^2)] / (π (b^2 - a^2)).
# Σ falls by orders of magnitude across a bin in a Gaussian's tail, where the uniform-pixel
# approximation cannot follow it, so the relative checks keep the bins holding at least 1% of
# the largest bin average. Profiles narrower than about two pixels (FWHM 0.5 on 0.2 pixels
# is off by >100% in its one non-negligible bin) are outside what the filter can resolve.
# ---------------------------------------------------------------------------

def exact_sigma_bin_average(a, b, s):
    """Area-weighted mean of Σ over [a, b] for a unit-mass Gaussian of width s."""
    return (np.exp(-a**2 / (2 * s**2)) - np.exp(-b**2 / (2 * s**2))) / (np.pi * (b**2 - a**2))


def test_sigma_uniform_sheet_is_exact():
    """A uniform sheet gives back its density in every bin, for any halo offset."""
    for pix, half, off in ((0.2, 35, (-0.13, -0.05)), (0.5, 15, (-0.4, -0.01))):
        W = filters.sigma_lensing_weights(half, pix, EDGES, offset=off)
        np.testing.assert_allclose(W.sum(axis=(1, 2)) * pix**2, 1.0, rtol=1e-12)


def test_sigma_point_masses_are_exact():
    """Mass in a pixel inside the inner edge contributes nothing; mass in a pixel wholly inside
    bin [a, b] contributes 1 / (π (b² - a²)) to that bin only."""
    half, pix = 35, 0.2
    W = filters.sigma_lensing_weights(half, pix, EDGES, offset=(0.0, 0.0))
    m = np.zeros((2 * half + 1,) * 2)
    m[half, half] = 1.0                              # spans r <= 0.14 < EDGES[0]
    np.testing.assert_allclose(lensing_profile(W, m), 0.0, atol=1e-15)
    m[:] = 0.0
    m[half + 5, half] = 1.0                          # spans 0.9 <= r <= 1.105, inside [0.741, 1.350]
    expected = np.zeros(len(EDGES) - 1)
    expected[0] = 1.0 / (np.pi * (EDGES[1]**2 - EDGES[0]**2))
    np.testing.assert_allclose(lensing_profile(W, m), expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize('fwhm', [1.2, 2.5, 6.0])
def test_sigma_gaussian_converges_with_pixel_size(fwhm):
    s = sig(fwhm)
    exact = exact_sigma_bin_average(EDGES[:-1], EDGES[1:], s)
    keep = exact >= 1e-2 * exact.max()
    errors = []
    for pix in (0.2, 0.1, 0.05):
        half = int(np.ceil(7.0 / pix))
        off = (-0.37 * pix, -0.81 * pix)
        c = np.arange(-half, half + 1) * pix
        W = filters.sigma_lensing_weights(half, pix, EDGES, offset=off)
        errors.append(np.max(np.abs(lensing_profile(W, pixel_mass(c, c, off, pix, s)) / exact - 1)[keep]))
    # Measured at 0.05': 5.5e-3 (FWHM 1.2), 9.5e-4 (2.5), 1.2e-4 (6.0); a change to EDGES or to
    # the 1% bin cut changes these.
    assert errors[2] < 0.006                         # 0.05': within 0.6 per cent
    assert errors[0] > 10 * errors[2]                # converging as the pixel size squared
    assert errors[2] < errors[1] < errors[0]
    if fwhm >= 2.5:
        assert errors[0] < 0.02                      # 0.2' maps: within 2 per cent


def test_sigma_plus_dsigma_is_mean_sigma_within():
    """<ΔΣ> + <Σ> is the bin average of Σ̄(<R), [Ein(x_b) - Ein(x_a)] / (π (b² - a²)) for a
    unit-mass Gaussian (x = R² / 2 s², Ein(x) = E1(x) + ln x + γ); pins the two weight builders to
    each other."""
    s, pix = sig(2.5), 0.05
    half = int(np.ceil(7.0 / pix))
    off = (-0.37 * pix, -0.81 * pix)
    c = np.arange(-half, half + 1) * pix
    m = pixel_mass(c, c, off, pix, s)
    total = lensing_profile(filters.delta_sigma_lensing_weights(half, pix, EDGES, offset=off)
                            + filters.sigma_lensing_weights(half, pix, EDGES, offset=off), m)
    def ein(r):
        x = r**2 / (2 * s**2)
        return exp1(x) + np.log(x) + np.euler_gamma
    exact = (ein(EDGES[1:]) - ein(EDGES[:-1])) / (np.pi * (EDGES[1:]**2 - EDGES[:-1]**2))
    np.testing.assert_allclose(total, exact, rtol=2e-3)


def test_sigma_cutout_too_small_raises():
    with pytest.raises(ValueError):
        filters.sigma_lensing_weights(30, 0.2, EDGES, offset=(-0.19, 0.0))   # reach 5.91 < 6.34


@pytest.mark.parametrize('units, rad_distance', [('kpc/h', 1.0), ('arcmin', 1.0), ('kpc/h', 2.0)])
def test_sigma_stack_on_array_centres_haloes_correctly(units, rad_distance):
    """End to end through stack_on_array, as for 'DSigma_lensing'. A wrong offset sign or a
    swapped axis would shift every stack centre by up to two pixels and fail the tolerance;
    radDistance = 2 (radii halved) would fail if the bin edges were not scaled by it."""
    s_obj, array, s = _stacker_on_gaussians(units)
    radii, prof = s_obj.stack_on_array(array, filterType='Sigma_lensing',
                                       minRadius=1.0 / rad_distance, maxRadius=6.0 / rad_distance,
                                       numRadii=9, projection='yz', radDistance=rad_distance,
                                       radDistanceUnits=units, halo_mask=np.arange(9))
    exact = exact_sigma_bin_average(EDGES[:-1], EDGES[1:], s)
    keep = exact >= 1e-2 * exact.max()
    assert prof.shape == (9, 9)
    # 0.1 pixels: the uniform-pixel error is <= 0.4 per cent in these bins (FWHM 2.5); offset
    # classes add a residual miscentring of at most 1/8 pixel.
    assert np.max(np.abs(prof / exact[:, None] - 1)[keep]) < 0.006
    assert np.max(np.abs(prof - exact[:, None])) < 2e-3 * exact.max()


# ---------------------------------------------------------------------------------------------
# Step 1c: ThumbStack-style stamps (stack_on_array(stampRes=...), 'DSigma_lensing' only).
# The data's kSZ is filtered after ThumbStack bilinearly interpolates the 0.5' map onto a 0.25'
# stamp centred on each galaxy; stampRes makes the simulations do the same.

def _arcmin_stacker_on_lattice(fwhm, pix_target, n_side=8, spacing=18.0, beam=None, seed=None):
    """A stub SimulationStacker (arcmin units, as stackMap uses) and a periodic map of
    n_side^2 unit-mass Gaussians, `spacing` arcmin apart, with exact mass per pixel; optionally
    smoothed with utils.fft_smoothed_map, as makeMap does. The sub-pixel offsets form an
    n_side x n_side lattice, or are uniform random with `seed` (not symmetric between rows and
    columns, so a swapped axis moves the stack centre)."""
    import astropy.units as u
    import stacker as stk
    from astropy.cosmology import FlatLambdaCDM
    from utils import comoving_to_arcmin, fft_smoothed_map

    h, om, z = 0.7, 0.3, 0.5
    cosmo = FlatLambdaCDM(H0=100 * h, Om0=om, Tcmb0=2.7255 * u.K)       # as stack_on_array
    target = n_side * spacing                                            # arcmin
    box = target / comoving_to_arcmin(1.0, z, cosmo=cosmo)               # kpc/h
    s_obj = stk.SimulationStacker.__new__(stk.SimulationStacker)
    s_obj.header = {'BoxSize': box, 'HubbleParam': h, 'Omega0': om}
    s_obj.z = z
    size = comoving_to_arcmin(box, z, cosmo=cosmo)
    npix = int(round(size / pix_target))
    pix = size / npix
    s = sig(fwhm)
    u_sub = (np.arange(n_side) + 0.5) / n_side
    j = np.arange(n_side)
    pos = np.array([((jy + 0.5) * size / n_side, (jz + 0.5) * size / n_side)
                    for jy in j for jz in j])
    if seed is None:
        sub = np.array([(a, b) for a in u_sub for b in u_sub])
    else:
        sub = np.random.default_rng(seed).uniform(0, 1, (n_side**2, 2))
    pos = np.floor(pos / pix) * pix + sub * pix
    centres = (np.arange(npix) + 0.5) * pix
    reach = int(np.ceil(8 * s / pix)) + 1
    array = np.zeros((npix, npix))
    for y, zz in pos:
        rows = (int(y / pix) + np.arange(-reach, reach + 1)) % npix
        cols = (int(zz / pix) + np.arange(-reach, reach + 1)) % npix
        array[np.ix_(rows, cols)] += pixel_mass(centres[rows], centres[cols], (y, zz), pix, s)
    if beam is not None:
        array = np.asarray(fft_smoothed_map(array, beam, pixel_size_arcmin=pix), dtype=float)
    pos3d = np.column_stack([np.zeros(len(pos)), pos * box / size])
    s_obj.loadHalos = lambda: {'GroupMass': np.ones(len(pos)), 'GroupPos': pos3d}
    return s_obj, array, len(pos)


def _stack_lensing(s_obj, array, n, **kw):
    _, prof = s_obj.stack_on_array(array, filterType='DSigma_lensing', minRadius=1.0,
                                   maxRadius=6.0, numRadii=9, projection='yz', radDistance=1.0,
                                   radDistanceUnits='arcmin', halo_mask=np.arange(n), **kw)
    return prof


def test_interpolated_stamp_reproduces_a_plane_exactly():
    """Bilinear interpolation of pixel means reproduces a linear field exactly, so the stamp
    must equal the plane at each stamp point: a half-pixel or axis error shows at once."""
    import stacker as stk
    npix, alpha, beta, gamma = 64, 0.3, 0.7, -1.1
    centres = np.arange(npix) + 0.5                    # pixel i spans [i, i + 1)
    plane = alpha + beta * centres[:, None] + gamma * centres[None, :]
    for position in ((31.37, 30.82), (29.0, 33.5), (32.95, 32.05)):
        stamp = stk.SimulationStacker.interpolated_stamp(plane, position, step=0.5, half=10)
        k = np.arange(-10, 11) * 0.5
        expected = alpha + beta * (position[0] + k[:, None]) + gamma * (position[1] + k[None, :])
        np.testing.assert_allclose(stamp, expected, rtol=0, atol=1e-12)


def test_stamp_uniform_sheet_cancels():
    s_obj, array, n = _arcmin_stacker_on_lattice(2.0, 0.5, n_side=2)
    sheet = np.ones_like(array)
    prof = _stack_lensing(s_obj, sheet, n, stampRes=0.25)
    scale = np.abs(_stack_lensing(s_obj, array, n, stampRes=0.25)).max()
    assert np.max(np.abs(prof)) < 1e-12 * max(scale, 1.0)


def test_stamp_converges_to_exact_on_fine_maps():
    """Pixel means plus bilinear interpolation smooth the profile slightly, an error that falls
    as the pixel size squared (FWHM 2.5: -0.38 per cent at 1' on 0.1' maps, -0.095 on 0.05').
    This catches gross errors (a half-pixel shift); the beamed-Gaussian test below pins the
    centring more finely."""
    exact = np.array([exact_bin_average(a, b, sig(2.5)) for a, b in zip(EDGES[:-1], EDGES[1:])])
    err = []
    for pix in (0.1, 0.05):
        s_obj, array, n = _arcmin_stacker_on_lattice(2.5, pix, n_side=3, seed=5)
        err.append(_stack_lensing(s_obj, array, n, stampRes=pix / 2).mean(axis=1) / exact - 1)
    assert np.max(np.abs(err[1])) < 1.5e-3
    assert 3.5 < err[0][0] / err[1][0] < 4.5


def test_stamp_step1c_factor_on_beamed_gaussian():
    """Known answer: stamp / direct filter at R0 = 1', ..., 6' for unit-mass Gaussians of FWHM 2'
    with exact mass per 0.5' pixel, smoothed by fft_smoothed_map with a 1.6' beam, 0.25' stamps.
    The expected values are the mean over 400 objects whose sub-pixel offsets form a 20 x 20
    lattice (DESIxHSC-Lensing scratch/gaussian_step1c.py, 2026-10-22; an independent
    implementation checked against ThumbStack's extractStamp). Here 64 objects at random
    offsets reproduce them to 2e-3; a quarter-pixel centring error or a swapped axis does not."""
    expected = np.array([0.9646, 0.9739, 0.9830, 0.9904, 0.9964, 0.9989, 0.9998, 1.0000, 1.0000])
    s_obj, array, n = _arcmin_stacker_on_lattice(2.0, 0.5, n_side=8, beam=1.6, seed=11)
    direct = _stack_lensing(s_obj, array, n).mean(axis=1)
    stamp = _stack_lensing(s_obj, array, n, stampRes=0.25).mean(axis=1)
    np.testing.assert_allclose(stamp / direct, expected, rtol=0, atol=2e-3)


def test_stamp_rejects_unsupported_settings():
    s_obj, array, n = _arcmin_stacker_on_lattice(2.0, 0.5, n_side=2)
    with pytest.raises(ValueError):
        s_obj.stack_on_array(array, filterType='DSigma', minRadius=1.0, maxRadius=6.0,
                             numRadii=9, projection='yz', radDistance=1.0,
                             radDistanceUnits='arcmin', halo_mask=np.arange(n), stampRes=0.25)
    with pytest.raises(ValueError):
        _stack_lensing(s_obj, array, n, stampRes=0.0)
    s_kpc, arr_kpc, s = _stacker_on_gaussians('kpc/h')
    with pytest.raises(ValueError):
        s_kpc.stack_on_array(arr_kpc, filterType='DSigma_lensing', minRadius=1.0, maxRadius=6.0,
                             numRadii=9, projection='yz', radDistance=1.0,
                             radDistanceUnits='kpc/h', halo_mask=np.arange(9), stampRes=0.25)
