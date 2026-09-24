"""Tests for the projected stellar-transfer maps and the lensing ratio.

Synthetic particles, haloes and maps only (no simulation data): the 2D pixel
index against the cached fields' ``binned_statistic_2d`` binning, the per-halo
bookkeeping of ``halo_transfer.transfer_maps_2d`` (and its agreement with the
3D ``transfer_field``), the linearity of the DSigma stack behind
``compute_stellar_maps.f_of_scale``, the stacked halo sample against
``stack_on_array``'s own selection, and the lensing-settings resolution.

Run:

    cd tests/
    pytest test_stellar_maps.py -v
"""

import os
import sys

import numpy as np
import pytest
from scipy.stats import binned_statistic_2d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'unbound_gas'))

import halo_transfer as ht
import compute_stellar_maps as csm


def _block(pos, mass, labels, m_gas=None):
    blk = dict(pos=pos.astype(np.float32), mass=mass.astype(np.float32),
               labels={'v': labels.astype(np.int32)})
    if m_gas is not None:
        blk['m_gas'] = m_gas.astype(np.float32)
        blk['sf'] = np.zeros(len(mass), dtype=bool)
    return blk


class TestPixelIndex:

    @pytest.mark.parametrize('projection', ['yz', 'xy', 'xz'])
    def test_matches_binned_statistic(self, projection):
        """bincount over the index = the cached fields' binned_statistic_2d sum, bit for bit."""
        rng = np.random.default_rng(0)
        box, n = 7.0, 13
        edges = np.linspace(0, box, n + 1)
        pos = rng.uniform(0, box, (50000, 3))
        pos = np.concatenate([pos, np.stack([edges, edges[::-1], edges], 1),
                              [[box, 0.0, box], [0.0, box, 0.0]]]).astype(np.float32)
        w = rng.uniform(0, 1, len(pos))
        a, b = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}[projection]
        ref = binned_statistic_2d(pos[:, a], pos[:, b], w, 'sum', bins=[n, n],
                                  range=[[0, box], [0, box]]).statistic
        pix = ht.pixel_index_2d(pos, n, box, projection)
        assert pix.dtype == np.int32
        got = np.bincount(pix, weights=w, minlength=n * n).reshape(n, n)
        assert np.array_equal(got, ref)

    def test_outside_box_raises(self):
        pos = np.array([[1.0, 1.0, 1.0], [1.0, 11.0, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            ht.pixel_index_2d(pos, 4, 10.0, 'yz')

    def test_empty(self):
        assert len(ht.pixel_index_2d(np.zeros((0, 3), dtype=np.float32), 4, 10.0)) == 0


class TestTransferMaps:
    N = 32

    @pytest.fixture
    def store(self):
        rng = np.random.default_rng(1)
        box, n_halo = 1000.0, 4
        halo_mass = np.array([1e13, 1e12, 1e11, 1e10])
        s_lab = rng.integers(-1, n_halo, 400)
        g_lab = rng.integers(-1, n_halo - 1, 3000)    # halo 3: stars but no gas
        g_mass = rng.uniform(1e8, 3e8, 3000)
        m_ion = g_mass * rng.uniform(0.0, 1.0, 3000)
        st = _block(rng.uniform(0, box, (400, 3)), rng.uniform(1e8, 1e9, 400), s_lab)
        gs = _block(rng.uniform(0, box, (3000, 3)), m_ion, g_lab, m_gas=g_mass)
        store = {'Stars': [st], 'gas': [gs]}
        for p in store:
            for blk in store[p]:
                blk['pix'] = ht.pixel_index_2d(blk['pos'], self.N, box, 'yz')
        return store, halo_mass, box

    def test_conservation_and_kept_haloes(self, store):
        st, halo_mass, _ = store
        A, B, diag = ht.transfer_maps_2d(st, 'v', halo_mass, 1e10, self.N)
        assert A.shape == B.shape == (self.N, self.N)
        assert A.min() >= 0 and B.min() >= 0
        assert abs(diag['sum_added_rel']) < 1e-12
        assert abs(diag['sum_removed_rel']) < 1e-12
        assert diag['max_halo_cons'] < 1e-12
        assert diag['n_haloes_kept_noion'] == 1       # halo 3 keeps its stars
        lab = st['Stars'][0]['labels']['v']
        m = st['Stars'][0]['mass'].astype(np.float64)
        assert diag['mstar_moved'] == pytest.approx(m[(lab >= 0) & (lab < 3)].sum(), rel=1e-12)

    def test_maps_are_the_particle_weights(self, store):
        """B = the moved stars and A = f_h m_ion, binned on the 2D grid."""
        st, halo_mass, box = store
        A, B, _ = ht.transfer_maps_2d(st, 'v', halo_mass, 1e11, self.N)
        s, g = st['Stars'][0], st['gas'][0]
        n_halo = len(halo_mass)
        sl = ht.restrict_labels(s['labels']['v'], halo_mass, 1e11)
        gl = ht.restrict_labels(g['labels']['v'], halo_mass, 1e11)
        f, active, _ = ht.transfer_factors(ht.halo_totals(sl, s['mass'], n_halo),
                                           ht.halo_totals(gl, g['mass'], n_halo))
        ss = (sl >= 0) & active[np.maximum(sl, 0)]
        gg = (gl >= 0) & active[np.maximum(gl, 0)]
        rng2 = [[0, box], [0, box]]
        refB = binned_statistic_2d(s['pos'][ss, 1], s['pos'][ss, 2], s['mass'][ss].astype(np.float64),
                                   'sum', bins=[self.N] * 2, range=rng2).statistic
        refA = binned_statistic_2d(g['pos'][gg, 1], g['pos'][gg, 2],
                                   f[gl[gg]] * g['mass'][gg].astype(np.float64),
                                   'sum', bins=[self.N] * 2, range=rng2).statistic
        assert np.allclose(B, refB, rtol=1e-12, atol=0)
        assert np.allclose(A, refA, rtol=1e-12, atol=0)

    def test_same_bookkeeping_as_3d(self, store):
        """Moved mass and active haloes agree with transfer_field for every cut."""
        st, halo_mass, box = store
        for cut in (1e10, 1e12, 1e13):
            _, _, d2 = ht.transfer_maps_2d(st, 'v', halo_mass, cut, self.N)
            _, d3 = ht.transfer_field(st, 'v', halo_mass, cut, 16, box)
            assert d2['n_haloes_active'] == d3['n_haloes_active']
            assert d2['mstar_moved'] == d3['mstar_moved']


class TestLensingRatio:

    def _stacker(self, n_halo, box):
        """A SimulationStacker with a synthetic header and halo catalogue."""
        from stacker import SimulationStacker
        st = SimulationStacker.__new__(SimulationStacker)
        rng = np.random.default_rng(5)
        st.header = {'BoxSize': box, 'HubbleParam': 0.7, 'Omega0': 0.3}
        st.z = 0.5
        pos = rng.uniform(0, box, (n_halo, 3))
        st.loadHalos = lambda: {'GroupMass': np.full(n_halo, 1e13), 'GroupPos': pos}
        return st

    def test_stack_is_linear_and_f_of_scale_exact(self):
        """Stacking the explicitly changed maps gives f_of_scale of the separate stacks."""
        rng = np.random.default_rng(6)
        box, n, n_halo = 100000.0, 400, 12
        st = self._stacker(n_halo, box)
        maps = {k: rng.lognormal(size=(n, n)) for k in ('N', 'T', 'A', 'B')}
        kw = dict(filterType='DSigma', minRadius=1.0, maxRadius=6.0, numRadii=9, projection='yz',
                  radDistance=1.0, radDistanceUnits='arcmin', z=0.5, pixelSize=0.2,
                  halo_mask=np.arange(n_halo))

        def mean_stack(arr):
            return st.stack_on_array(arr, **kw)[1].mean(axis=1)

        m = {k: mean_stack(v) for k, v in maps.items()}
        factor = 0.3 / 0.048
        for s in (1.0, 0.5, 0.0):
            t = 1.0 - s
            direct = (mean_stack(maps['N'] + t * maps['A'])
                      / mean_stack(maps['T'] + t * (maps['A'] - maps['B'])) * factor)
            formula = csm.f_of_scale(m['N'], m['T'], m['A'], m['B'], s, factor)
            assert np.allclose(direct, formula, rtol=1e-12, atol=0)
        assert np.array_equal(csm.f_of_scale(m['N'], m['T'], m['A'], m['B'], 1.0, factor),
                              m['N'] / m['T'] * factor)



class TestHaloSample:
    """stack_stellar_maps.halo_sample picks stack_on_array's rows, in its order."""

    def _stacker(self):
        from stacker import SimulationStacker
        rng = np.random.default_rng(7)
        box, n_par, n_sub = 100000.0, 400, 3000
        st = SimulationStacker.__new__(SimulationStacker)
        st.header = {'BoxSize': box, 'HubbleParam': 0.7, 'Omega0': 0.3}
        st.z = 0.5
        # bottom-heavy masses, so the 'massive' selection can reach its 10^13.22 target
        halos = {'GroupMass': 10 ** rng.normal(12.0, 0.8, n_par),
                 'GroupPos': rng.uniform(0, box, (n_par, 3))}
        subs = {'SubhaloMStar': 10 ** rng.uniform(8, 12, n_sub),
                'SubhaloGrNr': rng.integers(0, n_par, n_sub),
                'SubhaloPos': rng.uniform(0, box, (n_sub, 3))}
        st.loadHalos = lambda: halos
        st.loadSubHalos = lambda: subs
        return st

    @pytest.mark.parametrize('stack_cfg, overrides', [
        ({'use_subhalos': True, 'halo_abundance_target': 5e-4}, None),
        ({'use_subhalos': True, 'halo_abundance_target': 5e-4}, {'halo_mass_upper': None}),
        ({'use_subhalos': False}, None),
    ])
    def test_same_rows_as_stack_on_array(self, stack_cfg, overrides):
        import stack_stellar_maps as ssm
        st = self._stacker()
        stack = csm.resolve_stack_settings(stack_cfg, overrides)
        mask = ssm.halo_sample(st, stack)
        assert len(mask) > 0
        arr = np.random.default_rng(8).lognormal(size=(300, 300))
        kw = dict(filterType='DSigma', minRadius=1.0, maxRadius=6.0, numRadii=9, projection='yz',
                  radDistance=1.0, radDistanceUnits='arcmin', z=0.5, pixelSize=0.2,
                  use_subhalos=stack['use_subhalos'], halo_mass_avg=stack['halo_mass_avg'],
                  halo_mass_upper=stack['halo_mass_upper'],
                  halo_abundance_target=stack['halo_abundance_target'])
        own = st.stack_on_array(arr, halo_mask=None, **kw)[1]
        ours = st.stack_on_array(arr, halo_mask=mask, **kw)[1]
        assert np.array_equal(own, ours)


class TestStackSettings:

    def test_defaults_are_the_lensing_scripts(self):
        s = csm.resolve_stack_settings({'use_subhalos': True, 'halo_abundance_target': '1.0e-3'})
        assert s['halo_abundance_target'] == 1e-3          # string from PyYAML cast
        assert s['halo_mass_upper'] == 5e14 and s['pixel_size'] == 0.2
        assert (s['min_radius'], s['max_radius'], s['num_radii']) == (1.0, 6.0, 9)
        assert csm.resolve_stack_settings({})['halo_abundance_target'] == 5e-4

    def test_overrides(self):
        s = csm.resolve_stack_settings({'halo_abundance_target': 5e-4},
                                       {'halo_abundance_target': 2e-4})
        assert s['halo_abundance_target'] == 2e-4
        with pytest.raises(KeyError):
            csm.resolve_stack_settings({}, {'no_such_key': 1})

    @pytest.mark.parametrize('bad', [{'beam_size': 1.6}, {'particle_type': 'baryon'},
                                     {'filter_type': 'CAP'}, {'pixel_size_2': 0.5},
                                     {'mask_haloes': True}])
    def test_unsupported_settings_raise(self, bad):
        with pytest.raises(ValueError):
            csm.resolve_stack_settings(bad)
