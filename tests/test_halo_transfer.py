"""Tests for the halo-level stellar-to-ionized-gas transfer (src/halo_transfer.py).

Synthetic particles and haloes only (no simulation data): the aperture labels
against a brute-force mass-priority assignment with periodic distances, the
nesting of labels in the halo mass cut, the halo-finder ID mappings, halo-by-
halo mass conservation of the transfer field, and the quadratic P(s).

Run:

    cd tests/
    pytest test_halo_transfer.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import halo_transfer as ht


def brute_force_labels(pos, hpos, radius, priority, box):
    """Most-massive containing halo with minimum-image distances (reference)."""
    d = pos[:, None, :] - hpos[None, :, :]
    d -= box * np.round(d / box)
    inside = (d ** 2).sum(-1) <= radius[None, :] ** 2
    inside &= (radius > 0)[None, :]
    out = np.full(len(pos), -1)
    order = np.argsort(-priority, kind='stable')
    for p in range(len(pos)):
        hit = order[inside[p, order]]
        if len(hit):
            out[p] = hit[0]
    return out


@pytest.fixture
def random_setup():
    rng = np.random.default_rng(42)
    box = 100.0
    hpos = rng.uniform(0, box, (60, 3))
    hpos[0] = [0.5, 99.5, 50.0]          # straddles two periodic faces
    radius = rng.uniform(1.0, 12.0, 60)
    radius[5] = 0.0                      # no aperture: claims nothing
    mass = 10 ** rng.uniform(11, 14, 60)
    pos = rng.uniform(0, box, (20000, 3)).astype(np.float32)
    return pos, hpos, radius, mass, box


class TestApertureLabels:

    def test_matches_brute_force(self, random_setup):
        pos, hpos, radius, mass, box = random_setup
        lab = ht.ApertureLabeller(hpos, radius, mass, box).labels(pos)
        ref = brute_force_labels(pos.astype(np.float64), hpos, radius, mass, box)
        assert np.array_equal(lab, ref)
        assert np.any(lab >= 0) and np.any(lab < 0)

    def test_small_hash_grid_matches(self, random_setup):
        """Same answer when the hash grid is coarse (full-range insertion path)."""
        pos, hpos, radius, mass, box = random_setup
        lab = ht.ApertureLabeller(hpos, radius, mass, box, max_hash=2).labels(pos)
        ref = brute_force_labels(pos.astype(np.float64), hpos, radius, mass, box)
        assert np.array_equal(lab, ref)

    def test_overlap_goes_to_most_massive_and_wraps(self):
        box = 10.0
        hpos = np.array([[0.2, 5.0, 5.0], [9.6, 5.0, 5.0]])   # 0.6 apart across x = 0
        radius = np.array([1.0, 1.0])
        mass = np.array([1e12, 1e13])
        pos = np.array([[9.9, 5.0, 5.0], [0.9, 5.0, 5.0], [1.1, 5.0, 5.0],
                        [5.0, 5.0, 5.0]], dtype=np.float32)
        lab = ht.ApertureLabeller(hpos, radius, mass, box).labels(pos)
        # in both -> massive halo 1; 0.9 is 1.3 from halo 1 -> halo 0; 1.1 in halo 0 only
        assert lab.tolist() == [1, 0, 0, -1]

    def test_nested_in_mass_cut(self, random_setup):
        """Masking low-cut labels == labelling with only the haloes above the higher cut."""
        pos, hpos, radius, mass, box = random_setup
        cut_lo, cut_hi = 1e11, 1e13
        lo = ht.ApertureLabeller(hpos, radius, mass, box,
                                 rows=np.flatnonzero(mass >= cut_lo)).labels(pos)
        hi = ht.ApertureLabeller(hpos, radius, mass, box,
                                 rows=np.flatnonzero(mass >= cut_hi)).labels(pos)
        assert np.array_equal(ht.restrict_labels(lo, mass, cut_hi), hi)


class TestMembershipLabels:

    def test_group_ends(self):
        ends = np.cumsum([3, 0, 2]).astype(np.int64)      # group 1 is empty
        lab = ht.labels_from_group_ends(np.arange(7, dtype=np.int64), ends)
        assert lab.tolist() == [0, 0, 0, 2, 2, -1, -1]

    def test_fof_ids(self):
        central_ids = np.array([40, -1, 7, 12])            # row 1 is hostless
        lookup = ht.make_fof_lookup(central_ids)
        ids = np.array([7, 12, 40, 5, ht.FLAMINGO_NO_GROUP, -1])
        assert ht.labels_from_fof_ids(ids, lookup).tolist() == [2, 3, 0, -1, -1, -1]

    def test_duplicate_central_raises(self):
        with pytest.raises(ValueError):
            ht.make_fof_lookup(np.array([3, 3]))


def _block(pos, mass, labels, m_gas=None):
    blk = dict(pos=pos.astype(np.float32), mass=mass.astype(np.float32),
               labels={'v': labels.astype(np.int32)})
    if m_gas is not None:
        blk['m_gas'] = m_gas.astype(np.float32)
        blk['sf'] = np.zeros(len(mass), dtype=bool)
    return blk


class TestTransferField:
    # Grid of 64^3. tsc_parallel races (loses mass) when its parallel stripes
    # are narrower than ~3 cells; halo_transfer.deposit caps the thread count
    # at n // 8 so these tests hold whatever NUMBA_NUM_THREADS is.
    N = 64

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
        return {'Stars': [st], 'gas': [gs]}, halo_mass, box

    def test_conservation_and_kept_haloes(self, store):
        st, halo_mass, box = store
        D, diag = ht.transfer_field(st, 'v', halo_mass, 1e10, self.N, box)
        assert abs(diag['sum_D_rel']) < 1e-6          # float32 TSC weights
        assert diag['max_halo_cons'] < 1e-12
        assert diag['max_halo_removed'] < 1e-12
        assert diag['n_haloes_kept_noion'] == 1       # halo 3
        lab = st['Stars'][0]['labels']['v']
        m = st['Stars'][0]['mass'].astype(np.float64)
        assert diag['mstar_kept_noion'] == pytest.approx(m[lab == 3].sum())
        assert diag['mstar_moved'] == pytest.approx(m[(lab >= 0) & (lab < 3)].sum())

    def test_mass_cut_selects_haloes(self, store):
        st, halo_mass, box = store
        _, diag = ht.transfer_field(st, 'v', halo_mass, 1e12, self.N, box)
        lab = st['Stars'][0]['labels']['v']
        m = st['Stars'][0]['mass'].astype(np.float64)
        assert diag['n_haloes_selected'] == 2
        assert diag['mstar_moved'] == pytest.approx(m[(lab == 0) | (lab == 1)].sum())

    def test_field_is_added_minus_removed(self, store):
        """D equals the direct deposit of the per-particle weights."""
        st, halo_mass, box = store
        D, _ = ht.transfer_field(st, 'v', halo_mass, 1e10, self.N, box)
        s, g = st['Stars'][0], st['gas'][0]
        n_halo = len(halo_mass)
        mstar = ht.halo_totals(s['labels']['v'], s['mass'], n_halo)
        mion = ht.halo_totals(g['labels']['v'], g['mass'], n_halo)
        f, active, _ = ht.transfer_factors(mstar, mion)
        ref = np.zeros((self.N,) * 3)
        sl, gl = s['labels']['v'], g['labels']['v']
        ss = (sl >= 0) & active[np.maximum(sl, 0)]
        gg = (gl >= 0) & active[np.maximum(gl, 0)]
        ht.deposit(ref, s['pos'][ss].copy(), -s['mass'][ss].astype(np.float64), box)
        ht.deposit(ref, g['pos'][gg].copy(), f[gl[gg]] * g['mass'][gg], box)
        assert np.allclose(D, ref, rtol=1e-6, atol=1e-6 * np.abs(ref).max())


class TestPofScale:

    def test_quadratic_identity(self):
        rng = np.random.default_rng(3)
        rho = rng.lognormal(size=(16, 16, 16))
        D = rng.normal(size=(16, 16, 16))
        D -= D.mean()
        mean = rho.mean()

        def power(a, b):
            return np.real(np.fft.rfftn(a) * np.conj(np.fft.rfftn(b)))

        dm, dd = rho / mean - 1, D / mean
        P_mm, P_mD, P_DD = power(dm, dm), power(dm, dd), power(dd, dd)
        for s in (1.0, 0.5, 0.0):
            direct = power(dm + (1 - s) * dd, dm + (1 - s) * dd)
            assert np.allclose(ht.p_of_scale(P_mm, P_mD, P_DD, s), direct)
        assert np.array_equal(ht.p_of_scale(P_mm, P_mD, P_DD, 1.0), P_mm)

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            ht.p_of_scale(1.0, 0.0, 0.0, 1.5)
