"""simba_ea_correction.py

In-memory correction of SIMBA's ``ElectronAbundance`` (EA) for the
``scripts/simba_test`` impact study.

SIMBA snapshots store EA in two normalisations: dust-free particles carry
Grackle's n_e m_p / rho (with the Grackle species as mass fractions, so
GrackleHI + GrackleHII = X_H), while dust-bearing particles carry n_e / n_H
(with the species per H nucleus, GrackleHI + GrackleHII = 1).  Dividing by
GrackleHI + GrackleHII recovers n_e / n_H in both cases; the cap at the fully
ionised primordial value 1 + 2y handles ~2e-7 of the gas mass whose EA and
species were written by different code paths.  See
``docs/unbound_gas/simba_electron_abundance_report.md``.

Nothing in ``src/`` is edited.  :func:`simba_ea_corrected` temporarily wraps
``mapMaker.load_subset`` so that the existing field builders (``make_mass_field``
for ionized_gas, ``make_sz_field`` for tau/kSZ/tSZ) see the corrected EA;
everything else in those code paths is unchanged.  :func:`forbid_cache_writes`
replaces ``save_data`` everywhere it is imported, so a script using this module
cannot write to the field/map caches on scratch.
"""

import contextlib
import sys

import numpy as np

sys.path.append('../src/')
import loadIO     # type: ignore
import mapMaker   # type: ignore
import stacker as stacker_module  # type: ignore

X_H = 0.76
Y_HE_PER_H = (1.0 - X_H) / (4.0 * X_H)   # n_He / n_H = 0.0789
NE_CAP = 1.0 + 2.0 * Y_HE_PER_H          # fully ionised n_e / n_H = 1.1579

_original_load_subset = mapMaker.load_subset


def corrected_electron_abundance(ea: np.ndarray, grackle_hi: np.ndarray,
                                 grackle_hii: np.ndarray) -> np.ndarray:
    """Return n_e / n_H for SIMBA gas from the stored EA and Grackle H species.

    Args:
        ea: Stored ``ElectronAbundance``.
        grackle_hi: ``GrackleHI`` (mass fraction or per-H, matching ``ea``).
        grackle_hii: ``GrackleHII``.

    Returns:
        n_e / n_H, capped at 1 + 2y, in the dtype of ``ea``.
    """
    h_sum = grackle_hi.astype(np.float64) + grackle_hii.astype(np.float64)
    ne_nh = np.minimum(ea.astype(np.float64) / h_sum, NE_CAP)
    return ne_nh.astype(ea.dtype)


def _corrected_load_subset(sim_path, snapshot, sim_type, p_type, snap_path,
                           header=None, keys=None, sim_name=None):
    """``load_subset`` with EA replaced by n_e / n_H for SIMBA gas."""
    particles = _original_load_subset(sim_path, snapshot, sim_type, p_type, snap_path,
                                      header=header, keys=keys, sim_name=sim_name)
    if sim_type == 'SIMBA' and keys is not None and 'ElectronAbundance' in keys:
        species = _original_load_subset(sim_path, snapshot, sim_type, p_type, snap_path,
                                        header=header, keys=['GrackleHI', 'GrackleHII'],
                                        sim_name=sim_name)
        ea = particles['ElectronAbundance']
        particles['ElectronAbundance'] = corrected_electron_abundance(
            ea, species['GrackleHI'], species['GrackleHII'])
        print(f"  [simba_ea] corrected EA for {len(ea)} particles: "
              f"mean {np.mean(ea, dtype=np.float64):.4f} -> "
              f"{np.mean(particles['ElectronAbundance'], dtype=np.float64):.4f}", flush=True)
    return particles


@contextlib.contextmanager
def simba_ea_corrected(enabled: bool = True):
    """Context manager: field builders inside it see the corrected SIMBA EA.

    Args:
        enabled: If False, the loader is left untouched (used to validate that
            an in-memory rebuild reproduces the cached as-read field).
    """
    if enabled:
        mapMaker.load_subset = _corrected_load_subset
    try:
        yield
    finally:
        mapMaker.load_subset = _original_load_subset


def _refuse_save(*args, **kwargs):
    raise RuntimeError("simba_test: cache writes are disabled (save_data called).")


def forbid_cache_writes() -> None:
    """Make every ``save_data`` reachable from the stacker raise instead of writing."""
    for module in (loadIO, mapMaker, stacker_module):
        module.save_data = _refuse_save


def build_map(stacker, pType: str, z: float, projection: str, pixelSize: float,
              beamSize, corrected: bool) -> np.ndarray:
    """Build a (beam-convolved) map in memory and inject it into ``stacker.maps``.

    ``stackMap`` then finds it under its own cache key and neither loads nor
    saves anything for this particle type.

    Args:
        stacker: SimulationStacker instance (SIMBA).
        pType: Particle type, e.g. 'ionized_gas', 'tau', 'tSZ'.
        z: Redshift passed to ``stackMap``.
        projection: 'xy', 'xz' or 'yz'.
        pixelSize: Pixel size in arcmin.
        beamSize: Beam FWHM in arcmin, or None.
        corrected: Apply the EA correction while building.

    Returns:
        The map (also stored in ``stacker.maps``).
    """
    with simba_ea_corrected(corrected):
        map_ = stacker.makeMap(pType, z=z, projection=projection, beamSize=beamSize,
                               save=False, load=False, pixelSize=pixelSize)
    stacker.maps[(pType, z, projection, pixelSize, beamSize)] = map_
    return map_


def build_field_3d(stacker, pType: str, nPixels: int, corrected: bool) -> np.ndarray:
    """Build a 3D field in memory (no cache read or write).

    Args:
        stacker: SimulationStacker instance.
        pType: Particle type.
        nPixels: Grid size per side.
        corrected: Apply the EA correction while building.

    Returns:
        The 3D field.
    """
    with simba_ea_corrected(corrected):
        return stacker.makeField(pType, nPixels=nPixels, dim='3D', save=False, load=False)
