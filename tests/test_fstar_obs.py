"""Tests for the observation-based stellar bands (scripts/unbound_gas/compute_fstar_obs.py).

Synthetic regions and particles only (no simulation data): the uniform
stellar scale that brings an aggregate to a target (below and above s = 1,
with and without regions capped by their ionized gas, unreachable targets),
the s > 1 power-spectrum formula, and the change field built from the
solver's coefficients with halo_transfer's floor functions (conservation per
region, the aggregate after, and the uncapped case equal to -(s - 1) D).

Run:

    cd tests/
    pytest test_fstar_obs.py -v
"""

import os
import sys

import numpy as np
import pytest
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'unbound_gas'))

import halo_transfer as ht
import compute_fstar_obs as cfo

BOX = 1000.0
N3, N2 = 16, 32


def _total(ms, mi, m_fixed, s):
    """Stellar mass after scale s, by the definition (independent of the solver)."""
    c = np.full(len(ms), s - 1.0) if s <= 1 else np.minimum(s - 1.0, mi / ms)
    return float(np.sum(ms * (1 + c)) + m_fixed)


class TestSolveScale:

    def setup_method(self):
        rng = np.random.default_rng(7)
        self.ms = rng.uniform(1.0, 10.0, 200)
        self.mi = self.ms * rng.uniform(0.05, 5.0, 200)  # M_ion / M* from 0.05 to 5
        self.fixed = 3.0

    def test_below_one(self):
        target = 0.4 * self.ms.sum() + self.fixed
        s, coef, capped = cfo.solve_scale(self.ms, self.mi, self.fixed, target)
        assert s == pytest.approx(0.4, rel=1e-14)
        assert np.allclose(coef, s - 1.0) and not capped.any()

    def test_above_one_uncapped(self):
        # every region can supply 1.04 M*: s = 1.04 is uncapped
        mi = self.ms * 2.0
        target = 1.04 * self.ms.sum() + self.fixed
        s, coef, capped = cfo.solve_scale(self.ms, mi, self.fixed, target)
        assert s == pytest.approx(1.04, rel=1e-13)
        assert not capped.any() and np.allclose(coef, 0.04)

    @pytest.mark.parametrize('factor', [1.3, 1.5, 2.5, 3.3])   # at most ~3.55 (all ionized gas)
    def test_above_one_capped(self, factor):
        target = factor * self.ms.sum() + self.fixed
        s, coef, capped = cfo.solve_scale(self.ms, self.mi, self.fixed, target)
        # the aggregate hits the target, and s agrees with a root finder
        assert _total(self.ms, self.mi, self.fixed, s) == pytest.approx(target, rel=1e-12)
        s_ref = brentq(lambda x: _total(self.ms, self.mi, self.fixed, x) - target, 1.0, 10.0,
                       xtol=1e-14)
        assert s == pytest.approx(s_ref, rel=1e-10)
        r = self.mi / self.ms
        assert np.all(coef <= r * (1 + 1e-15))                  # never more than the ionized gas
        assert np.allclose(coef[capped], r[capped])             # capped: all of it
        assert np.allclose(coef[~capped], s - 1.0)              # others: the common s - 1
        assert np.all(r[capped] < s - 1.0) and np.all(r[~capped] >= s - 1.0)
        assert capped.any() == bool(r.min() < s - 1.0)
        assert capped.any() or factor < 1.3

    def test_all_ionized_gas(self):
        target = self.ms.sum() + self.mi.sum() + self.fixed
        s, coef, capped = cfo.solve_scale(self.ms, self.mi, self.fixed, target)
        assert np.allclose(coef, self.mi / self.ms)
        assert _total(self.ms, self.mi, self.fixed, s) == pytest.approx(target, rel=1e-12)

    def test_unreachable(self):
        with pytest.raises(ValueError):
            cfo.solve_scale(self.ms, self.mi, self.fixed, 1.01 * (self.ms.sum() + self.mi.sum()) + self.fixed)
        with pytest.raises(ValueError):
            cfo.solve_scale(self.ms, self.mi, self.fixed, 0.5 * self.fixed)
        with pytest.raises(ValueError):
            cfo.solve_scale(np.array([1.0, 0.0]), np.array([1.0, 1.0]), 0.0, 1.0)

    def test_solve_ends_sources(self):
        targets = {'fstar': {'low': 0.1, 'high': 0.3}, 'mstar_m200m': {'low': 0.01, 'high': 0.04}}
        denoms = {'fstar': (self.ms.sum() + self.fixed) / 0.2, 'mstar_m200m': (self.ms.sum() + self.fixed) / 0.02}
        ends = cfo.solve_ends(targets, self.ms, self.mi, self.fixed, denoms, float(self.ms.sum()))
        assert ends['fstar__low']['source'] == 's0' and ends['fstar__low']['s'] < 1
        assert ends['mstar_m200m__high']['source'] == 'pass'   # s ~ 2 caps the M_ion/M* = 0.05 regions
        for key, e in ends.items():
            opt = key.split('__')[0]
            after = _total(self.ms, self.mi, self.fixed, e['s'])
            assert after / denoms[opt] == pytest.approx(e['target'], rel=1e-12)


def test_p_at_scale():
    rng = np.random.default_rng(1)
    P, PmD, PDD = rng.uniform(1, 2, (3, 10))
    for s in (0.0, 0.3, 1.0):
        assert np.allclose(cfo.p_at_scale(P, PmD, PDD, s), ht.p_of_scale(P, PmD, PDD, s))
    assert np.allclose(cfo.p_at_scale(P, PmD, PDD, 2.5), P - 3.0 * PmD + 2.25 * PDD)
    with pytest.raises(ValueError):
        cfo.p_at_scale(P, PmD, PDD, -0.1)


def _blk(rng, n, labels, mass, **extra):
    blk = dict(pos=rng.uniform(0, BOX, (n, 3)).astype(np.float32), mass=mass.astype(np.float32),
               labels={'v': labels.astype(np.int32)})
    blk['pix'] = ht.pixel_index_2d(blk['pos'], N2, BOX, 'yz')
    blk.update(extra)
    return blk


@pytest.fixture
def store():
    """Four regions above the cut with M_ion / M* of 20, 5, 0.5 and 8 (the
    third caps first), and one halo below the cut."""
    rng = np.random.default_rng(11)
    halo_mass = np.array([2e13, 1e13, 1e13, 5e13, 1e11])
    s_lab = np.repeat([0, 1, 2, 3, 4, -1], [10, 20, 40, 10, 5, 6])
    stars = _blk(rng, len(s_lab), s_lab, np.full(len(s_lab), 1e9))
    g_lab = np.repeat([0, 1, 2, 3, 4, -1], [250, 125, 25, 100, 20, 30])
    m_gas = np.full(len(g_lab), 1e9)
    m_ion = 0.8 * m_gas
    gas = _blk(rng, len(g_lab), g_lab, m_ion, m_gas=m_gas.astype(np.float32),
               sf=np.zeros(len(g_lab), dtype=bool))
    winds = dict(mass=np.full(2, 5e8, dtype=np.float32), labels={'v': np.array([0, -1], dtype=np.int32)})
    bh = dict(mass=np.full(2, 2e8, dtype=np.float32), labels={'v': np.array([1, 3], dtype=np.int32)})
    return {'Stars': [stars], 'gas': [gas], 'Winds': [winds], 'BH': [bh]}, halo_mass


def _end_field(st, hm, target_f):
    """The compute script's steps for one fstar end: solve, then build H and the 2D maps."""
    cut = 1e13
    bary = ht.halo_baryons(st, 'v', hm, cut)
    has = bary['mbaryon'] > 0
    act = (bary['mstar'] > 0) & (bary['mion'] > 0)
    rows = np.flatnonzero(act)
    fixed = float(bary['mstar'][has & ~act].sum())
    s, c, cap = cfo.solve_scale(bary['mstar'][rows], bary['mion'][rows], fixed,
                                target_f * bary['mbaryon'][has].sum())
    coef = np.zeros(len(hm))
    coef[rows] = c
    capped = np.zeros(len(hm), dtype=bool)
    capped[rows] = cap
    no_stars = has & ~(bary['mstar'] > 0)
    no_ion = has & (bary['mstar'] > 0) & ~(bary['mion'] > 0)
    info = dict(dm=coef * bary['mstar'], raised=coef > 0, capped=capped, no_stars=no_stars,
                no_ion=no_ion)
    H, d3 = ht.scaled_transfer_field(st, 'v', hm, cut, coef, info, bary, np.nan, N3, BOX)
    S2, G2, d2 = ht.scaled_transfer_maps_2d(st, 'v', hm, cut, coef, info, bary, np.nan, N2)
    return s, coef, capped, bary, has, H, d3, S2, G2, d2


class TestEndField:

    def test_capped_end(self, store):
        st, hm = store
        s, coef, capped, bary, has, H, d3, S2, G2, d2 = _end_field(st, hm, 0.3)
        assert s > 1 and capped[2] and not capped[[0, 1, 3]].any()
        after = (bary['mstar'] + coef * bary['mstar'])[has].sum() / bary['mbaryon'][has].sum()
        assert after == pytest.approx(0.3, rel=1e-12)
        assert d3['fstar_after'] == pytest.approx(0.3, rel=1e-12)
        assert d3['max_removed_over_mion'] == pytest.approx(1.0, rel=1e-6)   # region 2: all of it
        assert max(d3['max_halo_cons_stars'], d3['max_halo_cons_gas']) < 1e-6
        assert abs(d3['sum_H_rel']) < 1e-5
        assert S2.sum() == pytest.approx(d3['mstar_added'], rel=1e-6)
        assert G2.sum() == pytest.approx(d3['mstar_added'], rel=1e-6)
        assert S2.min() >= 0 and G2.min() >= 0
        assert coef[4] == 0.0                                        # below the cut

    def test_uncapped_end_is_minus_scaled_D(self, store):
        st, hm = store
        # a small raise keeps every region uncapped: H = -(s - 1) D
        s, coef, capped, bary, has, H, d3, S2, G2, d2 = _end_field(st, hm, 0.155)
        assert 1 < s < 1.5 and not capped.any()
        D, _ = ht.transfer_field(st, 'v', hm, 1e13, N3, BOX)
        assert np.allclose(H, -(s - 1.0) * D, rtol=1e-5, atol=1e-6 * np.abs(D).max())
