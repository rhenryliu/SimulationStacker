"""Known-answer tests for the ring surface-density filter (filters.sigma_ring and
SimulationStacker.stack_on_array(filterType='Sigma')).

Σ(R) is the mean surface density of the pixels whose centres lie in [R, R + dr), the annulus
the 'DSigma' filter (filters.delta_sigma_kernel) subtracts, so ΔΣ(R) = Σ̄(<R) - Σ(R) exactly.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import filters  # noqa: E402


def _grid(half, pix):
    """Radial distance grid of a (2 half + 1)^2 cutout with pixel size pix, centred on a pixel."""
    c = np.arange(-half, half + 1) * pix
    return np.sqrt(c[:, None]**2 + c[None, :]**2)


def test_uniform_map():
    """A uniform map of m per pixel has Σ = m / pix^2 in every ring."""
    pix = 0.2
    rr = _grid(60, pix)
    m = np.full(rr.shape, 3.7)
    for R in (0.3, 1.0, 4.4, 10.0):
        assert filters.sigma_ring(m, rr, R, dr=0.75, pixel_size=pix) == pytest.approx(3.7 / pix**2,
                                                                                    rel=1e-12)


def test_mass_in_one_ring():
    """Mass only in [2, 2.75): Σ = 1 / pix^2 there and 0 in the rings either side."""
    pix = 0.2
    rr = _grid(40, pix)
    m = ((rr >= 2.0) & (rr < 2.75)).astype(float)
    assert filters.sigma_ring(m, rr, 2.0, dr=0.75, pixel_size=pix) == pytest.approx(1 / pix**2,
                                                                                  rel=1e-12)
    assert filters.sigma_ring(m, rr, 1.0, dr=0.75, pixel_size=pix) == 0.0
    assert filters.sigma_ring(m, rr, 2.75, dr=0.75, pixel_size=pix) == 0.0


def test_decomposes_delta_sigma():
    """On a random map, the 'DSigma' kernel equals the disk mean minus this ring, to rounding."""
    pix, dr = 0.2, 0.75
    rr = _grid(60, pix)
    m = np.random.default_rng(11).lognormal(size=rr.shape)
    for R in (0.52, 1.51, 5.47, 10.4):
        disk_mean = m[rr < R].sum() / (pix**2 * np.sum(rr < R))
        ds = filters.delta_sigma_kernel(m, rr, R, dr=dr, pixel_size=pix)
        sig = filters.sigma_ring(m, rr, R, dr=dr, pixel_size=pix)
        assert ds == pytest.approx(disk_mean - sig, rel=1e-12, abs=1e-12 * disk_mean)


def test_edges_on_exact_pixel_distances():
    """On an integer lattice the edges R = 3 and R + dr = 5 fall on pixel distances, so the
    half-open [R, R + dr) convention (that of the 'DSigma' annulus) is tested exactly."""
    rr = _grid(20, 1.0)
    m = np.random.default_rng(3).lognormal(size=rr.shape)
    assert np.sum(rr == 3.0) > 0 and np.sum(rr == 5.0) > 0
    sig = filters.sigma_ring(m, rr, 3.0, dr=2.0, pixel_size=1.0)
    assert sig == pytest.approx(m[(rr >= 3.0) & (rr < 5.0)].mean(), rel=1e-14)
    assert filters.delta_sigma_kernel(m, rr, 3.0, dr=2.0, pixel_size=1.0) == \
        pytest.approx(m[rr < 3.0].mean() - sig, rel=1e-12)


def test_bad_inputs_raise():
    """A non-positive width and a ring with no pixel centres are refused."""
    rr = _grid(10, 0.2)
    m = np.ones(rr.shape)
    for dr in (0.0, -0.5):
        with pytest.raises(ValueError, match='dr must be positive'):
            filters.sigma_ring(m, rr, 1.0, dr=dr, pixel_size=0.2)
    with pytest.raises(ValueError, match='No pixels'):
        filters.sigma_ring(m, rr, 5.0, dr=0.75, pixel_size=0.2)   # ring beyond the cutout


@pytest.mark.parametrize('rad_distance, dr', [(1.0, 0.75), (1.0, None), (2.0, 0.375), (2.0, None)])
def test_stack_on_array_matches_direct_ring(rad_distance, dr):
    """End to end through stack_on_array, on a random periodic map. The reference takes the ring
    pixels from the whole map, so a cutout too small for the outermost ring [6.53, 7.28) kpc/h
    (the sizing a plain int(maxRadius + 1) would give at radDistance = 1) changes the result and
    fails. Radii and dr are in radDistance units and the pixel area in kpc/h, so radDistance = 2
    tests the units; dr = None is stack_on_array's 3-pixel fallback. No ring edge falls on a
    pixel distance, so float rounding cannot move a pixel across one."""
    import stacker as stk

    box, npix = 40.0, 400                       # 0.1 kpc/h pixels, radDistance 1 kpc/h = 10 pixels
    d = box / npix
    s_obj = stk.SimulationStacker.__new__(stk.SimulationStacker)
    s_obj.header = {'BoxSize': box}
    rng = np.random.default_rng(5)
    array = rng.lognormal(size=(npix, npix))
    # Halo pixels, two near the box edges to exercise the periodic wrap; the positions sit
    # 0.2 pixels off the pixel corner so np.round picks that pixel.
    loc = np.array([[200, 200], [3, 390], [395, 7], [120, 260]])
    pos3d = np.column_stack([np.zeros(len(loc)), (loc + 0.2) * d])
    s_obj.loadHalos = lambda: {'GroupMass': np.ones(len(loc)), 'GroupPos': pos3d}

    radii, prof = s_obj.stack_on_array(array, filterType='Sigma', minRadius=1.03 / rad_distance,
                                       maxRadius=6.53 / rad_distance, numRadii=5, projection='yz',
                                       radDistance=rad_distance, radDistanceUnits='kpc/h',
                                       halo_mask=np.arange(len(loc)), dr=dr)
    assert prof.shape == (5, len(loc))
    dr_used = 3 * d / rad_distance if dr is None else dr

    idx = np.arange(npix)
    for h, (i0, j0) in enumerate(loc):
        di = (idx - i0 + npix // 2) % npix - npix // 2       # periodic pixel offsets
        dj = (idx - j0 + npix // 2) % npix - npix // 2
        dist = np.sqrt(di[:, None]**2 + dj[None, :]**2) * d / rad_distance
        for k, R in enumerate(radii):
            ring = (dist >= R) & (dist < R + dr_used)
            assert prof[k, h] == pytest.approx(array[ring].sum() / (d**2 * ring.sum()), rel=1e-12)
