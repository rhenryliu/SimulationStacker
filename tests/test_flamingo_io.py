"""Phase 1 verification gate for FLAMINGO I/O support.

Validates that the FLAMINGO branches in loadIO.py / stacker.py deliver data in
the pipeline's internal conventions (positions in ckpc/h, halo masses in
Msun/h, particle masses in 1e10 Msun/h, BoxSize in ckpc/h), and that global
mass densities reproduce the box cosmology (Omega_cdm, Omega_b).

Requires the FLAMINGO L1_m9 download on Perlmutter scratch; all tests skip if
the data is not present. Run from the repo root or tests/ directory:

    pytest tests/test_flamingo_io.py -v

Note: the global-density test streams particle masses over all 64 snapshot
chunks (~50 GB of I/O) and takes a few minutes.
"""
import os
import sys
import glob

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loadIO import (snap_path, load_flamingo_header, load_halos, load_subhalos,
                    load_subset, _get_data_filepath)

FLAMINGO_BASE = '/pscratch/sd/r/rhliu/simulations/FLAMINGO/'
SIM = 'L1_m9'
SNAPSHOT = 67
VARIANTS = ['L1_m9', 'fgas-8sigma', 'Jet_fgas-4sigma']

# Reference values for the fiducial L1_m9 box at snapshot 67 (z=0.5),
# cross-checked against the SOAP catalog in tests/test_flamingo.ipynb.
H_EXPECTED = 0.681
BOXSIZE_CKPC_H = 1000.0 * 1000.0 * H_EXPECTED  # 1 cGpc in ckpc/h
N_SUBHALOS = 17_677_657
N_CENTRALS = 13_401_874
MAX_M200M_MSUN = 1.651e15       # most massive halo, Msun (no h)
MAX_R200M_CMPC = 3.691          # its radius, comoving Mpc (no h)
MEAN_GAS_MASS_MSUN = 1.099e9    # mean gas particle mass, Msun (no h)
DM_MASS_1E10MSUN = 0.56500919   # InitialMassTable[1], 1e10 Msun (no h)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(FLAMINGO_BASE + SIM),
    reason='FLAMINGO data not available on this system')


def sim_path(variant=SIM):
    return FLAMINGO_BASE + SIM + '/' + variant + '/'


@pytest.fixture(scope='module')
def header():
    vf = snap_path(sim_path(), SNAPSHOT, 'FLAMINGO')
    return load_flamingo_header(vf)


class TestPaths:
    def test_virtual_file_exists(self):
        vf = snap_path(sim_path(), SNAPSHOT, 'FLAMINGO')
        assert vf.endswith('flamingo_0067.hdf5')
        assert os.path.isfile(vf)

    def test_folder_path_has_64_chunks(self):
        folder = snap_path(sim_path(), SNAPSHOT, 'FLAMINGO', path_only=True)
        chunks = sorted(glob.glob(folder + 'flamingo_*.hdf5'))
        assert len(chunks) == 64

    def test_all_variants_present(self):
        for variant in VARIANTS:
            vf = snap_path(sim_path(variant), SNAPSHOT, 'FLAMINGO')
            assert os.path.isfile(vf), variant


class TestHeader:
    def test_boxsize_in_ckpc_h(self, header):
        assert np.isclose(header['BoxSize'], BOXSIZE_CKPC_H, rtol=1e-6)
        assert np.isscalar(header['BoxSize'])

    def test_cosmology(self, header):
        assert np.isclose(header['HubbleParam'], H_EXPECTED)
        assert np.isclose(header['Omega0'], 0.304611, rtol=1e-4)
        assert np.isclose(header['OmegaBaryon'], 0.0486, rtol=1e-4)
        assert np.isclose(header['Redshift'], 0.5, atol=1e-4)
        assert np.isclose(header['Time'], 1.0 / 1.5, rtol=1e-6)

    def test_masstable_zero_and_counts_not_overflowed(self, header):
        # All-zero MassTable means every ptype has a per-particle mass field
        # (so the TNG DM MassTable trick must never trigger for FLAMINGO).
        assert np.all(header['MassTable'] == 0.0)
        # The virtual file's NumPart_Total overflows 32 bits; the normalized
        # header must carry the true (HighWord-corrected) counts.
        assert header['NumPart_Total'][0] == 5_364_099_453
        assert header['NumPart_Total'][1] == 5_832_000_000


class TestHaloCatalog:
    @pytest.fixture(scope='class')
    def halos(self, header):
        return load_halos(sim_path(), SNAPSHOT, 'FLAMINGO', header=header)

    def test_centrals_only(self, halos):
        assert len(halos['GroupMass']) == N_CENTRALS

    def test_mass_units_msun_h(self, halos):
        # Most massive halo: 1.651e15 Msun -> *h in Msun/h
        assert np.isclose(halos['GroupMass'].max(), MAX_M200M_MSUN * H_EXPECTED, rtol=2e-3)

    def test_radius_units_ckpc_h(self, halos):
        i = np.argmax(halos['GroupMass'])
        assert np.isclose(halos['GroupRad'][i], MAX_R200M_CMPC * 1000.0 * H_EXPECTED, rtol=2e-3)

    def test_positions_within_box(self, halos):
        assert halos['GroupPos'].min() >= 0.0
        assert halos['GroupPos'].max() <= BOXSIZE_CKPC_H * (1 + 1e-6)
        # Positions genuinely span the box (i.e. not left in cMpc)
        assert halos['GroupPos'].max() > 0.9 * BOXSIZE_CKPC_H


class TestSubhaloCatalog:
    @pytest.fixture(scope='class')
    def subhalos(self, header):
        return load_subhalos(sim_path(), SNAPSHOT, 'FLAMINGO', header=header)

    def test_no_filtering(self, subhalos):
        assert len(subhalos['SubhaloMass']) == N_SUBHALOS

    def test_grnr_indexes_centrals_catalog(self, subhalos, header):
        # SubhaloGrNr must be a valid index into load_halos' centrals-only arrays
        assert subhalos['SubhaloGrNr'].min() >= 0
        assert subhalos['SubhaloGrNr'].max() < N_CENTRALS

    def test_centrals_map_to_own_row(self, subhalos, header):
        # For central subhalos, GroupPos[SubhaloGrNr] must be their own position.
        halos = load_halos(sim_path(), SNAPSHOT, 'FLAMINGO', header=header)
        rng = np.random.default_rng(0)
        idx = rng.choice(N_SUBHALOS, size=2000, replace=False)
        pos_h = halos['GroupPos'][subhalos['SubhaloGrNr'][idx]]
        pos_s = subhalos['SubhaloPos'][idx]
        dist = np.linalg.norm(pos_h - pos_s, axis=1)
        # Centrals: exact match. Satellites: within their host halo
        # (< ~10 Mpc/h separation, generous for the largest clusters).
        assert np.median(dist) < 1e-3          # most subhalos are centrals
        assert dist.max() < 10_000.0           # ckpc/h

    def test_stellar_masses_sane(self, subhalos):
        mstar = subhalos['SubhaloMStar']
        assert mstar.min() >= 0.0
        # Massive galaxies exist but nothing absurd (units check: Msun/h)
        assert 1e12 < mstar.max() < 1e14


class TestParticleChunk:
    @pytest.fixture(scope='class')
    def chunk0(self):
        folder = snap_path(sim_path(), SNAPSHOT, 'FLAMINGO', path_only=True)
        return folder + 'flamingo_0067.0.hdf5'

    def test_gas_contract(self, chunk0, header):
        particles = load_subset(sim_path(), SNAPSHOT, 'FLAMINGO', 'gas', chunk0,
                                header=header)
        coords, masses = particles['Coordinates'], particles['Masses']
        # Positions in ckpc/h, inside the box
        assert coords.min() >= 0.0
        assert coords.max() <= BOXSIZE_CKPC_H * (1 + 1e-6)
        assert coords.max() > 1000.0  # would fail if left in cMpc
        # Masses in 1e10 Msun/h
        expected = MEAN_GAS_MASS_MSUN / 1e10 * H_EXPECTED
        assert np.isclose(masses.mean(), expected, rtol=0.05)
        # Mass is conserved when binned exactly as mapMaker does it
        field, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=100,
                                     range=[[0, BOXSIZE_CKPC_H], [0, BOXSIZE_CKPC_H]],
                                     weights=masses)
        # (sum in float64: float32 accumulation loses precision at 8e7 particles)
        assert np.isclose(field.sum(), masses.sum(dtype=np.float64), rtol=1e-7)

    def test_dm_has_per_particle_masses(self, chunk0, header):
        particles = load_subset(sim_path(), SNAPSHOT, 'FLAMINGO', 'DM', chunk0,
                                header=header)
        masses = particles['Masses']
        # DM masses scatter ~0.5% around InitialMassTable[1] (they are not
        # exactly uniform in FLAMINGO); check the mean and the spread.
        assert np.isclose(masses.mean(), DM_MASS_1E10MSUN * H_EXPECTED, rtol=5e-3)
        assert masses.std() / masses.mean() < 0.02
        assert 'ParticleIDs' not in particles  # TNG trick must not trigger

    def test_bh_masses_alias(self, chunk0, header):
        # FLAMINGO BHs have no 'Masses' dataset; loader must alias DynamicalMasses
        particles = load_subset(sim_path(), SNAPSHOT, 'FLAMINGO', 'BH', chunk0,
                                header=header)
        assert 'Masses' in particles
        assert particles['Masses'].min() > 0.0


class TestGlobalDensity:
    """Sum particle masses over all 64 chunks and compare to the cosmology.

    This is the strongest end-to-end unit check: any stray factor of h, 1e10,
    or a missed particle population shifts the recovered Omega values.
    """

    @pytest.fixture(scope='class')
    def mass_budget_msun(self, header):
        from astropy.cosmology import FlatLambdaCDM
        import astropy.units as u
        h = header['HubbleParam']
        cosmo = FlatLambdaCDM(H0=100 * h * u.km / u.s / u.Mpc, Om0=header['Omega0'])
        rho_crit0 = cosmo.critical_density0.to(u.Msun / u.Mpc ** 3).value  # Msun/Mpc^3
        v_box = 1000.0 ** 3  # cMpc^3
        return rho_crit0 * v_box  # Msun for Omega=1

    @pytest.fixture(scope='class')
    def summed_masses(self, header):
        folder = snap_path(sim_path(), SNAPSHOT, 'FLAMINGO', path_only=True)
        chunks = sorted(glob.glob(folder + 'flamingo_*.hdf5'))
        totals = {}
        for p_type in ['gas', 'DM', 'Stars', 'BH']:
            total = 0.0
            for chunk in chunks:
                particles = load_subset(sim_path(), SNAPSHOT, 'FLAMINGO', p_type,
                                        chunk, header=header, keys=['Masses'])
                total += np.sum(particles['Masses'], dtype=np.float64)
            # Loader units are 1e10 Msun/h -> native Msun (no h) for comparison
            totals[p_type] = total * 1e10 / header['HubbleParam']
        return totals

    def test_dm_density_matches_omega_cdm(self, summed_masses, mass_budget_msun, header):
        omega_cdm = header['Omega0'] - header['OmegaBaryon']
        recovered = summed_masses['DM'] / mass_budget_msun
        assert np.isclose(recovered, omega_cdm, rtol=5e-3), \
            f'recovered Omega_cdm={recovered:.5f}, expected {omega_cdm:.5f}'

    def test_baryon_density_matches_omega_b(self, summed_masses, mass_budget_msun, header):
        baryons = summed_masses['gas'] + summed_masses['Stars'] + summed_masses['BH']
        recovered = baryons / mass_budget_msun
        assert np.isclose(recovered, header['OmegaBaryon'], rtol=0.02), \
            f'recovered Omega_b={recovered:.5f}, expected {header["OmegaBaryon"]:.5f}'


class TestStackerIntegration:
    @pytest.fixture(scope='class')
    def stacker(self):
        from stacker import SimulationStacker
        return SimulationStacker(sim=SIM, snapshot=SNAPSHOT, nPixels=100,
                                 simType='FLAMINGO', feedback='fgas-8sigma', z=0.5)

    def test_header_normalized(self, stacker):
        assert np.isclose(stacker.header['BoxSize'], BOXSIZE_CKPC_H, rtol=1e-6)
        assert np.isclose(stacker.h, H_EXPECTED)
        assert np.isclose(stacker.cosmo.Om0, 0.304611, rtol=1e-4)

    def test_snap_paths(self, stacker):
        assert os.path.isfile(stacker.snapPath())
        folder = stacker.snapPath(pathOnly=True)
        assert len(glob.glob(folder + 'flamingo_*.hdf5')) == 64

    def test_bad_feedback_rejected(self):
        from stacker import SimulationStacker
        with pytest.raises(AssertionError):
            SimulationStacker(sim=SIM, snapshot=SNAPSHOT, simType='FLAMINGO',
                              feedback='s50', z=0.5)

    def test_cache_filename_includes_variant(self):
        path = _get_data_filepath('FLAMINGO', SIM, SNAPSHOT, 'fgas-8sigma',
                                  'gas', 1000, projection='xy', data_type='field', dim='2D')
        assert path.name == 'L1_m9_fgas-8sigma_67_gas_1000_xy.npy'
        assert 'FLAMINGO/products/2D' in str(path)
