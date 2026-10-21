"""Known-answer tests for scripts/cross_corr/make_frb_dm_xi.py.

- The annulus routine recovers electron excesses painted around galaxies on a
  periodic map, exactly, in every log bin.
- Its wiring matches rprofiles.compute_Y_matrix's 'Sigma' amplitude once the
  kernel's 1/(pixel area) is multiplied back.
- It equals an explicit loop over galaxies, and its jackknife realizations are
  the leave-one-block-out means, on a grid with unequal blocks as in production.
- Units: a fully ionized map built from the cosmic baryon density in Msun/h per
  (ckpc/h)^2 gives the electron column computed in g cm^-3 (catches h-factor
  errors in either function); mean_electron_column inverts
  mapMaker.ionized_gas_masses (with literal constants); the parsec and the
  pixel size are checked against independent values.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts' / 'cross_corr'))

import rprofiles as rp  # noqa: E402
import make_frb_dm_xi as fx  # noqa: E402


def _painted_maps(n=240, pix=0.1, ring=(0.55, 0.95), excess=2.5):
    """Galaxy count map and electron map with a ring of excess around each galaxy.

    Galaxies sit on pixels at least 6 units apart (periodically), so the painted
    rings, and every annulus tested, never overlap another galaxy's. Ring
    membership uses the integer-lag radius from the galaxy pixel, as the
    'Sigma' kernel does; no pixel distance falls on a ring edge.
    """
    sites = [(20, 30), (90, 200), (150, 100), (215, 15), (60, 120), (180, 180)]
    counts = np.zeros((n, n))
    ion = np.ones((n, n))
    idx = np.arange(n)
    for i0, j0 in sites:
        counts[i0, j0] += 1.0
        di = (idx - i0 + n // 2) % n - n // 2
        dj = (idx - j0 + n // 2) % n - n // 2
        r = np.hypot(di[:, None], dj[None, :]) * pix
        ion[(r >= ring[0]) & (r < ring[1])] += excess
    return counts, ion, pix


def test_log_bin_edges():
    e = fx.log_bin_edges(0.3, 30.0, 15)
    assert e.shape == (16,)
    assert e[0] == pytest.approx(0.3) and e[-1] == pytest.approx(30.0)
    np.testing.assert_allclose(e[1:] / e[:-1], 100 ** (1 / 15), rtol=1e-12)


def test_annulus_means_recover_painted_ring():
    counts, ion, pix = _painted_maps()
    mean_ion = ion.mean()
    fields = {'e': rp.to_overdensity(ion), 'flat': np.zeros_like(ion)}
    inner = np.array([0.55, 1.25, 2.05])           # painted ring, then two empty annuli
    widths = np.array([0.40, 0.50, 0.70])
    mean, jk = fx.annulus_means(rp.to_overdensity(counts), fields, pix, inner, widths, 4)
    expected = np.array([(1 + 2.5) / mean_ion - 1, 1 / mean_ion - 1, 1 / mean_ion - 1])
    np.testing.assert_allclose(mean['e'], expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mean['flat'], 0.0, atol=1e-15)
    assert jk['e'].shape == (3, 16)


def test_annulus_means_wiring_matches_compute_Y_matrix():
    """pixel**2 x the rprofiles 'Sigma' amplitude Y_eg, on a random map."""
    rng = np.random.default_rng(4)
    n, pix = 128, 0.2
    counts = rng.poisson(0.05, size=(n, n)).astype(float)
    ion = rng.lognormal(size=(n, n))
    dg, de = rp.to_overdensity(counts), rp.to_overdensity(ion)
    radii = np.array([1.0, 1.6, 2.4])
    Ymat = rp.compute_Y_matrix({'g': dg, 'e': de}, pix, radii=radii, dr=0.5, r0=1.0,
                               nbar_pix=counts.mean())
    mean, _ = fx.annulus_means(dg, {'e': de}, pix, radii, np.full(3, 0.5), 4)
    np.testing.assert_allclose(mean['e'] / pix ** 2, rp.get_Y(Ymat, 'Sigma', 'g', 'e'),
                               rtol=1e-10)


def test_annulus_means_match_pair_loop():
    """An explicit loop over galaxies (pixels holding several count several times)."""
    rng = np.random.default_rng(7)
    n, pix, a, w = 90, 0.3, 0.95, 0.8                # edges 3.17 and 5.83 pixels: no lattice ties
    counts = rng.poisson(0.03, size=(n, n)).astype(float)
    counts[10, 10], counts[50, 70] = 2.0, 4.0
    de = rp.to_overdensity(rng.lognormal(size=(n, n)))
    mean, _ = fx.annulus_means(rp.to_overdensity(counts), {'e': de}, pix, np.array([a]),
                               np.array([w]), 4)
    idx = np.arange(n)
    total, weight = 0.0, 0.0
    for i0, j0 in zip(*np.nonzero(counts)):
        di = (idx - i0 + n // 2) % n - n // 2
        dj = (idx - j0 + n // 2) % n - n // 2
        r = np.hypot(di[:, None], dj[None, :]) * pix
        ring = (r >= a) & (r < a + w)
        total += counts[i0, j0] * de[ring].mean()
        weight += counts[i0, j0]
    assert mean['e'][0] == pytest.approx(total / weight, rel=1e-10)


def test_jackknife_leave_one_block_out():
    """Realization b is the mean of the product map without block b (unequal blocks)."""
    rng = np.random.default_rng(2)
    n, pix = 90, 0.3                                 # 90 -> blocks of 22 and 23 pixels
    dg = rp.to_overdensity(rng.poisson(0.02, size=(n, n)).astype(float))
    de = rp.to_overdensity(rng.lognormal(size=(n, n)))
    _, jk = fx.annulus_means(dg, {'e': de}, pix, np.array([0.9]), np.array([0.8]), 4)
    kern = rp.build_aperture_kernel(n, pix, 0.9, 'Sigma', dr=0.8)
    prod = rp.filtered_map(dg, rp.kernel_spectrum(kern), (n, n)) * pix ** 2 * de
    ed = rp.block_edges(n, 4)
    for b, (i, j) in enumerate(np.ndindex(4, 4)):
        keep = np.ones((n, n), bool)
        keep[ed[i]:ed[i + 1], ed[j]:ed[j + 1]] = False
        assert jk['e'][0, b] == pytest.approx(prod[keep].mean(), rel=1e-10)


def test_dm_from_column():
    import astropy.units as u
    pc_cm = (1 * u.pc).to(u.cm).value                 # independent of fx.PC_CM
    assert fx.dm_from_column(pc_cm, 0.0) == pytest.approx(1.0, rel=1e-4)
    assert fx.dm_from_column(2 * pc_cm, 0.3) == pytest.approx(2.6, rel=1e-4)


def test_pixel_size_cmpc():
    assert fx.pixel_size_cmpc(205000.0, 4822, 0.6774) == pytest.approx(0.062758, rel=1e-4)
    assert fx.pixel_size_cmpc(681000.0, 14015, 0.681) == pytest.approx(1000.0 / 14015, rel=1e-9)


def test_electron_column_inverts_mapmaker():
    """mapMaker.ionized_gas_masses -> mean_electron_column gives N_e per comoving area."""
    import types
    import mapMaker
    h, xe, mass_1e10, pixel = 0.68, 1.0, 2.0, 50.0    # mass in 1e10 Msun/h; pixel in ckpc/h
    stacker = types.SimpleNamespace(header={'HubbleParam': h}, simType='IllustrisTNG')
    m_ion = mapMaker.ionized_gas_masses({'Masses': np.array([mass_1e10]),
                                         'ElectronAbundance': np.array([xe])}, stacker)
    n_e = xe * 0.76 * mass_1e10 * 1e10 * (1.989e33 / h) / 1.6726e-24   # literals, not fx's
    column = fx.mean_electron_column(m_ion.reshape(1, 1), pixel, h)
    assert column == pytest.approx(n_e / (pixel / h * 3.0857e21) ** 2, rel=1e-9)


def test_electron_column_unit_chains_agree():
    """Msun/h per pixel -> cm^-2 against Omega_b rho_crit / (mu_e m_p) in g cm^-3."""
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM

    h, omega_b, pixel, depth = 0.681, 0.0486, 71.4, 681000.0     # ckpc/h
    rho_crit = FlatLambdaCDM(H0=100 * h, Om0=0.3).critical_density0
    rho_h_units = rho_crit.to(u.Msun / u.kpc ** 3).value / h ** 2   # (Msun/h) / (kpc/h)^3
    m_ion = omega_b * rho_h_units * pixel ** 2 * depth               # Msun/h per pixel
    column = fx.mean_electron_column(np.full((4, 4), m_ion), pixel, h)
    assert column == pytest.approx(fx.cosmic_electron_column(omega_b, h, depth), rel=1e-3)
