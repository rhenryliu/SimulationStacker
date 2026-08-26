"""Phase 2 verification gate for FLAMINGO SZ fields.

Validates the FLAMINGO branch of mapMaker.make_sz_field, which uses SWIFT's
precomputed ComptonYParameters / ElectronNumberDensities instead of the
TNG-style ElectronAbundance / InternalEnergy derivation:

1. unit test of _flamingo_sz_weights on synthetic particles (exact algebra,
   including the no-sqrt(a) velocity convention);
2. physics consistency on real cluster gas: ComptonYParameters must equal
   sigma_T * (k_B T / m_e c^2) * n_e * V, validating the particle-volume and
   unit chain used for tau/kSZ;
3. data-level check: summed particle Compton-y within R200m of the most
   massive halo must reproduce the SOAP SO/200_mean/ComptonY value;
4. integration test: make_sz_field run on a single chunk conserves the total
   Compton-y of that chunk.

Run: pytest tests/test_flamingo_sz.py -v
"""
import os
import sys
import glob

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

FLAMINGO_BASE = '/pscratch/sd/r/rhliu/simulations/FLAMINGO/'
SIM = 'L1_m9'
SNAPSHOT = 67
SIM_PATH = FLAMINGO_BASE + SIM + '/' + SIM + '/'
VIRTUAL_FILE = SIM_PATH + 'snapshots/flamingo_0067/flamingo_0067.hdf5'
SOAP_FILE = SIM_PATH + 'SOAP-HBT/halo_properties_0067.hdf5'

MPC_TO_CM = 3.0856775814913673e24
SIGMA_T = 6.6524587158e-29 * 1.e2**2  # cm^2
K_B = 1.3807e-16                       # erg/K
M_E = 9.10938356e-28                   # g
C_LIGHT = 29979245800.                 # cm/s

pytestmark = pytest.mark.skipif(
    not os.path.isdir(FLAMINGO_BASE + SIM),
    reason='FLAMINGO data not available on this system')


def read_cells(f, ptype, cell_idx, fields):
    """Read fields for all particles in the given spatial cells (see notebook)."""
    offsets = f['Cells/OffsetsInFile/' + ptype][:][cell_idx]
    counts = f['Cells/Counts/' + ptype][:][cell_idx]
    order = np.argsort(offsets)
    offsets, counts = offsets[order], counts[order]
    out = {field: np.concatenate([f[f'{ptype}/{field}'][o:o + c]
                                  for o, c in zip(offsets, counts)])
           for field in fields}
    return out


@pytest.fixture(scope='module')
def target_halo():
    """Most massive halo: centre (cMpc), R200m (cMpc), SOAP ComptonY (cm^2)."""
    with h5py.File(SOAP_FILE, 'r') as f:
        m200m = f['SO/200_mean/TotalMass'][:]
        t = int(np.argmax(m200m))
        return {
            'pos': f['InputHalos/HaloCentre'][t],
            'R200m': float(f['SO/200_mean/SORadius'][t]),
            'ComptonY_cm2': float(f['SO/200_mean/ComptonY'][t]) * MPC_TO_CM**2,
        }


@pytest.fixture(scope='module')
def halo_gas(target_halo):
    """Gas particle fields in the spatial cells around the target halo."""
    with h5py.File(VIRTUAL_FILE, 'r') as f:
        centres = f['Cells/Centres'][:]
        cell_size = f['Cells/Meta-data'].attrs['size']
        box = float(f['Header'].attrs['BoxSize'][0])
        d = centres - target_halo['pos']
        d -= box * np.round(d / box)
        margin = 2.0 * target_halo['R200m']
        cells = np.where(np.all(np.abs(d) <= margin + cell_size / 2, axis=1))[0]
        gas = read_cells(f, 'PartType0', cells,
                         ['Coordinates', 'Masses', 'Densities', 'Temperatures',
                          'ComptonYParameters', 'ElectronNumberDensities'])
        gas['box'] = box
        gas['a'] = float(f['Header'].attrs['Scale-factor'][0])
    return gas


class TestWeightsHelper:
    """Exact algebra of _flamingo_sz_weights on synthetic particles."""

    def test_synthetic_particle(self):
        from mapMaker import _flamingo_sz_weights
        h, a = 0.681, 2.0 / 3.0
        pix_area = 3.7e45  # arbitrary physical cm^2
        particles = {
            'Masses': np.array([2.0 * h]),               # loader units: native 2.0
            'Densities': np.array([0.5]),                # native -> V_com = 4 Mpc^3
            'ComptonYParameters': np.array([1.5e-3]),    # physical Mpc^2
            'ElectronNumberDensities': np.array([2.5e70]),  # physical Mpc^-3
            'Velocities': np.array([[300.0, -150.0, 0.0]]),  # peculiar km/s
        }
        dY, b, tau = _flamingo_sz_weights(particles, h, a, pix_area)

        assert np.isclose(dY[0], 1.5e-3 * MPC_TO_CM**2 / pix_area, rtol=1e-12)
        N_e = 2.5e70 * 4.0 * a**3
        assert np.isclose(tau[0], SIGMA_T * N_e / pix_area, rtol=1e-12)
        # Velocity convention: b = tau * v/c with v the RAW velocity (peculiar
        # km/s, no sqrt(a)), c in cm/s -- matching the TNG branch convention.
        assert np.allclose(b[0], tau[0] * particles['Velocities'][0] / C_LIGHT, rtol=1e-12)
        assert b.shape == (1, 3)

    def test_star_forming_particles_contribute_zero(self):
        from mapMaker import _flamingo_sz_weights
        particles = {
            'Masses': np.array([0.1]),
            'Densities': np.array([0.5]),
            'ComptonYParameters': np.array([0.0]),        # 0 for star-forming gas
            'ElectronNumberDensities': np.array([0.0]),   # 0 for star-forming gas
            'Velocities': np.array([[100.0, 0.0, 0.0]]),
        }
        dY, b, tau = _flamingo_sz_weights(particles, 0.681, 2. / 3., 1e45)
        assert dY[0] == 0.0 and tau[0] == 0.0 and np.all(b[0] == 0.0)


class TestPhysicsConsistency:
    def test_compton_y_equals_sigmaT_Te_ne_V(self, halo_gas):
        """y_i = sigma_T * (k_B T_i / m_e c^2) * n_e_i * V_i for hot cluster gas.

        Validates the electron-count construction N_e = n_e * (M/D) * a^3 that
        the tau/kSZ fields rely on, against SWIFT's own ComptonYParameters.
        """
        a = halo_gas['a']
        sel = (halo_gas['ElectronNumberDensities'] > 0) & (halo_gas['Temperatures'] > 1e6)
        assert sel.sum() > 10000  # plenty of hot gas in a 1.65e15 Msun cluster

        V_phys_cm3 = (halo_gas['Masses'][sel].astype(np.float64)
                      / halo_gas['Densities'][sel]) * a**3 * MPC_TO_CM**3
        ne_cm3 = halo_gas['ElectronNumberDensities'][sel] / MPC_TO_CM**3
        y_pred_cm2 = (SIGMA_T * K_B * halo_gas['Temperatures'][sel]
                      / (M_E * C_LIGHT**2) * ne_cm3 * V_phys_cm3)
        y_stored_cm2 = halo_gas['ComptonYParameters'][sel] * MPC_TO_CM**2

        ratio = y_pred_cm2 / y_stored_cm2
        assert abs(np.median(ratio) - 1.0) < 0.05, f'median ratio {np.median(ratio):.4f}'

    def test_ionized_gas_mass_construction(self, halo_gas):
        """Hot cluster gas is fully ionized: M_ion = N_e * m_p * mu_e ~ M_gas.

        Validates the FLAMINGO ionized_gas branch of make_mass_field, which
        uses the same N_e = n_e * (M/D) * a^3 electron-count construction.
        """
        m_p = 1.6726e-24
        solar_mass = 1.989e33
        mu_e = 2.0 / (1.0 + 0.76)
        a = halo_gas['a']

        sel = (halo_gas['ElectronNumberDensities'] > 0) & (halo_gas['Temperatures'] > 1e7)
        M_native = halo_gas['Masses'][sel].astype(np.float64)          # 1e10 Msun
        V_phys = M_native / halo_gas['Densities'][sel] * a**3          # physical Mpc^3
        Ne = halo_gas['ElectronNumberDensities'][sel] * V_phys         # electron count
        M_ion_native = Ne * m_p * mu_e / solar_mass / 1e10             # 1e10 Msun

        ratio = M_ion_native / M_native
        assert abs(np.median(ratio) - 1.0) < 0.1, \
            f'median ionized fraction {np.median(ratio):.4f} for hot cluster gas'

    def test_halo_compton_y_matches_soap(self, halo_gas, target_halo):
        """Summed particle y within R200m reproduces SOAP SO/200_mean/ComptonY."""
        d = halo_gas['Coordinates'] - target_halo['pos']
        d -= halo_gas['box'] * np.round(d / halo_gas['box'])
        r = np.linalg.norm(d, axis=1)
        y_sum_cm2 = (halo_gas['ComptonYParameters'][r < target_halo['R200m']].sum()
                     * MPC_TO_CM**2)
        assert np.isclose(y_sum_cm2, target_halo['ComptonY_cm2'], rtol=0.01), \
            f'particles {y_sum_cm2:.4e} vs SOAP {target_halo["ComptonY_cm2"]:.4e}'


class TestMakeSZFieldIntegration:
    """Run make_sz_field on a single chunk and check y conservation."""

    @pytest.fixture(scope='class')
    def single_chunk_field(self):
        import mapMaker
        from stacker import SimulationStacker

        chunk0 = SIM_PATH + 'snapshots/flamingo_0067/swift_snapshot_0067/flamingo_0067.0.hdf5'
        stacker = SimulationStacker(sim=SIM, snapshot=SNAPSHOT, nPixels=200,
                                    simType='FLAMINGO', feedback=SIM, z=0.5)

        real_glob = glob.glob
        orig = mapMaker.glob.glob
        try:
            mapMaker.glob.glob = lambda pattern: [chunk0] if 'flamingo_' in pattern else real_glob(pattern)
            field = mapMaker.make_sz_field(stacker, 'tSZ', nPixels=200, projection='xy')
        finally:
            mapMaker.glob.glob = orig
        return field, chunk0, stacker

    def test_tsz_field_conserves_chunk_y(self, single_chunk_field):
        field, chunk0, stacker = single_chunk_field
        assert field.shape == (200, 200)
        assert np.all(np.isfinite(field))
        assert field.min() >= 0.0

        with h5py.File(chunk0, 'r') as f:
            y_total_cm2 = np.sum(f['PartType0/ComptonYParameters'][:], dtype=np.float64) * MPC_TO_CM**2

        a = 1.0 / 1.5
        h = stacker.h
        kpc_to_cm = MPC_TO_CM / 1000.0
        pix_area_cm2 = (a * stacker.header['BoxSize'] * (kpc_to_cm / h) / 200) ** 2
        assert np.isclose(field.sum() * pix_area_cm2, y_total_cm2, rtol=1e-6), \
            f'field total {field.sum() * pix_area_cm2:.5e} vs chunk total {y_total_cm2:.5e}'
