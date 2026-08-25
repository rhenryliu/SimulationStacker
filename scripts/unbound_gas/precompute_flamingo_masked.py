"""precompute_flamingo_masked.py

Build and cache the FLAMINGO SZ maps needed by simulated_tSZ_masked.py (Fig 9)
and simulated_kSZ_masked.py (Fig 7), which both run with ``load_field: true``
and therefore expect every map to be on disk already.

For one (feedback variant, particle type) pair this writes, under
``<data root>/FLAMINGO/products/``:

- ``2D/L1_m9_{feedback}_{snapshot}_{pType}_{nPixels}_{projection}_map.npy``
  the unmasked, beam-convolved map (the "No Masking" column);
- ``3D/L1_m9_{feedback}_{snapshot}_{pType}_{nPixels}.npy`` the intermediate
  cubic field, built once on the first masked call and reloaded for the rest;
- ``2D/masked/..._map_masked{R}R200c.npy`` one map per masking radius.

Sizing at the default 0.5 arcmin pixel size: the FLAMINGO L1_m9 box
(1000 cMpc = 681,000 ckpc/h) gives nPixels = 3548, so the cubic field is
3548**3 = 4.5e10 cells. That is 179 GB as float32 (see mapMaker.FIELD_3D_DTYPE)
plus a 45 GB boolean mask, i.e. ~223 GB peak -- it needs a full 512 GB
Perlmutter CPU node, and the 3D cache is ~179 GB on disk per pair.

Each step is skipped when its output already exists, so a job that runs out of
walltime can simply be resubmitted. The one exception is a 3D cache truncated
by a job killed mid-write: ``np.load`` may then raise something other than the
ValueError that mapMaker retries on, so delete such a file by hand before
resubmitting.

Usage
-----
    python unbound_gas/precompute_flamingo_masked.py --ptype tSZ --projection xz \
        --feedback L1_m9

The companion runCPU_flamingo_masked.sh submits all six pairs as a job array.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.append('../src/')
import mapMaker  # type: ignore
import stacker as stacker_module  # type: ignore
from stacker import SimulationStacker  # type: ignore

# Projection each figure stacks along, matching the YAML configs:
# tSZ_z05_CAP_masked.yaml uses 'xz', tau_z05_CAP_masked_flamingo.yaml uses 'xy'.
_DEFAULT_PROJECTION = {'tSZ': 'xz', 'tau': 'xy'}


class _TruncatedGlob:
    """Stand-in for mapMaker's ``glob`` module that returns only the first N hits.

    Used solely by the ``--max-chunks`` validation mode: it makes a full run of
    the real code path finish in minutes by reading a couple of the 64 FLAMINGO
    chunk files instead of all of them. The resulting fields are physically
    meaningless and must never be saved as products.
    """

    def __init__(self, real_glob_module, max_chunks):
        self._real = real_glob_module
        self._max_chunks = max_chunks

    def glob(self, pattern):
        return sorted(self._real.glob(pattern))[:self._max_chunks]

    def __getattr__(self, name):
        return getattr(self._real, name)


def _disable_all_writes():
    """Replace save_data with a no-op everywhere the pipeline calls it.

    ``--no-save`` alone is not enough: ``makeField`` passes ``save3D=True`` into
    ``create_masked_field`` unconditionally, so the cubic intermediate would
    still be written -- and under ``--max-chunks`` that file would be a partial
    field sitting in the products cache under a perfectly ordinary name, silently
    poisoning any later run at the same pixel size. Stubbing save_data in both
    module namespaces makes validation mode provably write-free.
    """
    def _no_write(data, *args, **kwargs):
        shape = getattr(data, 'shape', '?')
        print(f"    [validation mode] suppressed write of array {shape}",
              flush=True)

    mapMaker.save_data = _no_write
    stacker_module.save_data = _no_write


def precompute(feedback, pType, projection=None, sim='L1_m9', snapshot=67,
               redshift=0.5, pixelSize=0.5, beamSize=1.6,
               maskRadii=(1.0, 2.0, 3.0), skip_unmasked=False,
               save=True, max_chunks=None, verbose=True):
    """Build and cache one FLAMINGO variant's unmasked and masked maps.

    Args:
        feedback (str): FLAMINGO variant directory name, one of 'L1_m9'
            (fiducial), 'fgas-8sigma', 'Jet_fgas-4sigma'.
        pType (str): Particle type, 'tSZ' or 'tau'.
        projection (str, optional): Projection direction. Defaults to None,
            which resolves to the projection the corresponding figure uses.
        sim (str, optional): Simulation name. Defaults to 'L1_m9'.
        snapshot (int, optional): Snapshot number. Defaults to 67 (z=0.5).
        redshift (float, optional): Snapshot redshift. Defaults to 0.5.
        pixelSize (float, optional): Map pixel size in arcmin. Defaults to 0.5.
        beamSize (float, optional): Beam FWHM in arcmin. Defaults to 1.6, the
            stackMap default the plotting scripts rely on.
        maskRadii (tuple of float, optional): Masking radii in units of R200c.
            Defaults to (1.0, 2.0, 3.0), the three masked columns.
        skip_unmasked (bool, optional): If True, skip the no-masking map.
            Defaults to False.
        save (bool, optional): If False, computed maps are not written to the
            products cache. Validation only. Defaults to True.
        max_chunks (int, optional): If set, read only the first N of the 64
            FLAMINGO snapshot chunk files. Validation only -- the resulting
            fields are physically meaningless. Defaults to None (all chunks).
        verbose (bool, optional): If True, print per-step timings.

    Returns:
        None
    """
    if projection is None:
        projection = _DEFAULT_PROJECTION[pType]

    if max_chunks is not None:
        if save:
            raise ValueError(
                "max_chunks builds fields from a subset of the snapshot; "
                "refusing to run with save=True so partial fields cannot be "
                "written into the products cache. Pass --no-save.")
        print(f"[validation mode] reading only the first {max_chunks} chunk "
              f"file(s); fields are NOT physically meaningful", flush=True)
        mapMaker.glob = _TruncatedGlob(mapMaker.glob, max_chunks)
        _disable_all_writes()

    stacker = SimulationStacker(sim, snapshot, z=redshift,
                                simType='FLAMINGO', feedback=feedback)

    t_start = time.time()

    if not skip_unmasked:
        if verbose:
            print(f"\n=== {feedback} / {pType} / {projection}: unmasked map ===",
                  flush=True)
        t0 = time.time()
        stacker.makeMap(pType, projection=projection, beamSize=beamSize,
                        pixelSize=pixelSize, save=save, load=True, mask=False)
        print(f"    unmasked map done in {time.time() - t0:.1f} s", flush=True)
        gc.collect()

    for maskRad in maskRadii:
        if verbose:
            print(f"\n=== {feedback} / {pType} / {projection}: "
                  f"masked at {maskRad} R200c ===", flush=True)
        t0 = time.time()
        stacker.makeMap(pType, projection=projection, beamSize=beamSize,
                        pixelSize=pixelSize, save=save, load=True,
                        mask=True, maskRad=maskRad)
        print(f"    masked ({maskRad} R200c) done in {time.time() - t0:.1f} s",
              flush=True)
        # The cubic field and its boolean mask are ~223 GB combined; drop them
        # before the next radius reloads the field from the 3D cache.
        gc.collect()

    print(f"\nAll done for {feedback} / {pType}: "
          f"{time.time() - t_start:.1f} s total", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Precompute FLAMINGO unmasked and masked SZ maps.')
    parser.add_argument('--feedback', type=str, required=True,
                        choices=['L1_m9', 'fgas-8sigma', 'Jet_fgas-4sigma'],
                        help='FLAMINGO variant directory name.')
    parser.add_argument('--ptype', type=str, required=True,
                        choices=['tSZ', 'tau'], dest='pType',
                        help='Particle type to build.')
    parser.add_argument('--projection', type=str, default=None,
                        choices=['xy', 'xz', 'yz'],
                        help="Projection; defaults to 'xz' for tSZ, 'xy' for tau.")
    parser.add_argument('--sim', type=str, default='L1_m9')
    parser.add_argument('--snapshot', type=int, default=67)
    parser.add_argument('--redshift', type=float, default=0.5)
    parser.add_argument('--pixel-size', type=float, default=0.5,
                        dest='pixelSize', help='Map pixel size in arcmin.')
    parser.add_argument('--beam-size', type=float, default=1.6,
                        dest='beamSize', help='Beam FWHM in arcmin.')
    parser.add_argument('--mask-radii', type=float, nargs='*',
                        default=[1.0, 2.0, 3.0], dest='maskRadii',
                        help='Masking radii in units of R200c.')
    parser.add_argument('--skip-unmasked', action='store_true',
                        dest='skip_unmasked',
                        help='Skip the no-masking map.')
    parser.add_argument('--no-save', action='store_false', dest='save',
                        help='Do not write anything to the products cache. '
                             'Required with --max-chunks.')
    parser.add_argument('--max-chunks', type=int, default=None,
                        dest='max_chunks',
                        help='Validation only: read just the first N of the 64 '
                             'snapshot chunk files so a full run of the real '
                             'code path finishes in minutes. Implies --no-save.')
    args = vars(parser.parse_args())
    print(f"Arguments: {args}", flush=True)

    precompute(**args)
