"""Tests for the stellar-fraction floor (src/halo_transfer.py).

Synthetic particles and haloes only (no simulation data): per-region baryon
budgets with winds and black holes, the floor coefficients (raised, capped,
untouched, no-star and no-ionized-gas regions), and the 3D and 2D change
fields (conservation per halo and in total, the particle weights, and the
relation to the s = 0 transfer field).

Run:

    cd tests/
    pytest test_fstar.py -v
"""

import os
import sys

import numpy as np
import pytest
from scipy.stats import binned_statistic_2d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import halo_transfer as ht

BOX = 1000.0
N3, N2 = 16, 32


def _blk(rng, n, labels, mass, **extra):
    blk = dict(pos=rng.uniform(0, BOX, (n, 3)).astype(np.float32), mass=mass.astype(np.float32),
               labels={'v': labels.astype(np.int32)})
    blk['pix'] = ht.pixel_index_2d(blk['pos'], N2, BOX, 'yz')
    blk.update(extra)
    return blk


@pytest.fixture
def store():
    """Six haloes: 0 star-poor (raised), 1 star-rich (untouched), 2 gas-poor
    (capped), 3 no stars, 4 no ionized gas, 5 below the mass cut."""
    rng = np.random.default_rng(3)
    halo_mass = np.array([1e13, 1e13, 1e13, 1e13, 1e13, 1e10])
    # stars (true): per halo counts; mass 1e9 each
    s_lab = np.repeat([0, 1, 2, 4, 5, -1], [10, 200, 10, 10, 5, 7])
    stars = _blk(rng, len(s_lab), s_lab, np.full(len(s_lab), 1e9))
    # gas: m_gas 1e9 each; ionized fraction 0.8, except halo 2 (tiny) and 4 (none)
    g_lab = np.repeat([0, 1, 2, 3, 4, 5, -1], [300, 100, 100, 50, 80, 20, 30])
    m_gas = np.full(len(g_lab), 1e9)
    m_ion = 0.8 * m_gas
    m_ion[g_lab == 2] = 1e6
    m_ion[g_lab == 4] = 0.0
    gas = _blk(rng, len(g_lab), g_lab, m_ion, m_gas=m_gas.astype(np.float32),
               sf=np.zeros(len(g_lab), dtype=bool))
    winds = dict(mass=np.full(3, 5e8, dtype=np.float32), labels={'v': np.array([0, 1, -1], dtype=np.int32)})
    bh = dict(mass=np.full(2, 2e8, dtype=np.float32), labels={'v': np.array([0, 1], dtype=np.int32)})
    return {'Stars': [stars], 'gas': [gas], 'Winds': [winds], 'BH': [bh]}, halo_mass


class TestBaryons:

    def test_budget(self, store):
        st, hm = store
        b = ht.halo_baryons(st, 'v', hm, 1e11)
        assert b['mstar'][0] == pytest.approx(10e9)
        assert b['mgas'][0] == pytest.approx(300e9)
        assert b['mion'][0] == pytest.approx(240e9)
        assert b['mwind'][0] == pytest.approx(5e8)
        assert b['mbh'][1] == pytest.approx(2e8)
        assert b['mbaryon'][0] == pytest.approx(10e9 + 300e9 + 5e8 + 2e8)
        assert b['mbaryon'][5] == 0.0          # below the cut


class TestFloor:
    T = 0.3

    def test_coefficients(self, store):
        st, hm = store
        b = ht.halo_baryons(st, 'v', hm, 1e11)
        coef, info = ht.floor_coefficients(b, self.T)
        f = b['mstar'] / np.where(b['mbaryon'] > 0, b['mbaryon'], 1)
        # halo 0: raised exactly to the target
        assert info['raised'][0] and not info['capped'][0]
        assert (b['mstar'][0] + info['dm'][0]) / b['mbaryon'][0] == pytest.approx(self.T, rel=1e-12)
        assert coef[0] == pytest.approx(info['dm'][0] / b['mstar'][0])
        # halo 1 is above the target: untouched
        assert f[1] > self.T and coef[1] == 0 and not info['raised'][1]
        # halo 2: target needs more than its ionized gas -> capped at all of it
        assert info['capped'][2] and info['dm'][2] == pytest.approx(b['mion'][2])
        # halo 3 has no stars, halo 4 no ionized gas: unchanged
        assert info['no_stars'][3] and coef[3] == 0
        assert info['no_ion'][4] and coef[4] == 0
        # below the cut
        assert coef[5] == 0

    def test_field_conserves_and_matches_weights(self, store):
        st, hm = store
        b = ht.halo_baryons(st, 'v', hm, 1e11)
        coef, info = ht.floor_coefficients(b, self.T)
        H, d = ht.scaled_transfer_field(st, 'v', hm, 1e11, coef, info, b, self.T, N3, BOX)
        assert abs(d['sum_H_rel']) < 1e-6                    # float32 TSC weights
        assert d['max_halo_cons_stars'] < 1e-12 and d['max_halo_cons_gas'] < 1e-12
        assert d['max_removed_over_mion'] <= 1.0 + 1e-12
        assert abs(d['min_fstar_after_minus_target']) < 1e-12
        assert d['n_raised'] == 2 and d['n_capped'] == 1 and d['n_no_stars'] == 1 and d['n_no_ion'] == 1
        # direct deposit of the particle weights
        s, g = st['Stars'][0], st['gas'][0]
        ratio = np.where(b['mion'] > 0, b['mstar'] / np.where(b['mion'] > 0, b['mion'], 1), 0)
        sl, gl = s['labels']['v'], g['labels']['v']
        ss = (sl >= 0) & (coef[np.maximum(sl, 0)] > 0)
        gg = (gl >= 0) & (coef[np.maximum(gl, 0)] > 0)
        ref = np.zeros((N3,) * 3)
        ht.deposit(ref, s['pos'][ss].copy(), coef[sl[ss]] * s['mass'][ss].astype(np.float64), BOX)
        ht.deposit(ref, g['pos'][gg].copy(), -coef[gl[gg]] * ratio[gl[gg]] * g['mass'][gg].astype(np.float64), BOX)
        assert np.allclose(H, ref, rtol=1e-6, atol=1e-6 * np.abs(ref).max())

    def test_maps(self, store):
        st, hm = store
        b = ht.halo_baryons(st, 'v', hm, 1e11)
        coef, info = ht.floor_coefficients(b, self.T)
        S, G, d = ht.scaled_transfer_maps_2d(st, 'v', hm, 1e11, coef, info, b, self.T, N2)
        assert S.min() >= 0 and G.min() >= 0
        assert abs(d['sum_stars_rel']) < 1e-12 and abs(d['sum_gas_rel']) < 1e-12
        s = st['Stars'][0]
        sl = s['labels']['v']
        ss = (sl >= 0) & (coef[np.maximum(sl, 0)] > 0)
        refS = binned_statistic_2d(s['pos'][ss, 1], s['pos'][ss, 2],
                                   coef[sl[ss]] * s['mass'][ss].astype(np.float64), 'sum',
                                   bins=[N2, N2], range=[[0, BOX], [0, BOX]]).statistic
        assert np.allclose(S, refS, rtol=1e-12, atol=0)

    def test_unit_coefficient_is_minus_the_s0_transfer(self, store):
        """coef = 1 in every active region adds M*_h of stars and removes it from
        the ionized gas: exactly minus the s = 0 transfer field D."""
        st, hm = store
        b = ht.halo_baryons(st, 'v', hm, 1e11)
        active = (b['mstar'] > 0) & (b['mion'] > 0)
        coef = active.astype(float)
        info = dict(dm=b['mstar'] * coef, raised=active, capped=np.zeros(len(hm), bool),
                    no_stars=np.zeros(len(hm), bool), no_ion=np.zeros(len(hm), bool))
        H, _ = ht.scaled_transfer_field(st, 'v', hm, 1e11, coef, info, b, 0.0, N3, BOX)
        D, _ = ht.transfer_field(st, 'v', hm, 1e11, N3, BOX)
        assert np.allclose(H, -D, rtol=1e-6, atol=1e-6 * np.abs(D).max())
