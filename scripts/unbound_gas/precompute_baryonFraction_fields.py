"""precompute_baryonFraction_fields.py

Build and cache the fields that make_baryonFraction.py (unbound gas paper,
Figure 5) expects to find on disk, for one (simulation, particle type, dim)
task per invocation.

Why this is needed
------------------
``baryonFraction_z05.yaml`` runs with ``load_field: true`` at 0.2 arcmin pixels
(2D) and ``n_pixels: 1000`` (3D) over six simulations, and an audit of
``$SCRATCH/simulations/*/products/`` found the following gaps:

- **2D @ 0.2 arcmin**: ``gas``, ``baryon``, ``ionized_gas`` and ``total`` are
  cached for all six, but ``Stars`` and ``BH`` are cached for none.
- **3D @ 1000**: complete for TNG300-1, Illustris-1 and SIMBA m100n1024, but the
  three FLAMINGO variants are missing ``ionized_gas``.

``neutral_gas`` is missing everywhere too, but the figure derives it as
``gas - ionized_gas`` (see ``make_baryonFraction.resolve_stack_types``), so it
is only worth generating to warm the cache for other scripts -- hence the
optional third group in the companion runner.

Without this precompute the figure would silently recompute each missing field
inline, turning a plotting run into a multi-hour particle-reading job.

What one task writes
--------------------
Under ``<data root>/{simType}/products/``:

- ``--dim 2D``: ``2D/..._{nPixels}_{projection}.npy``, the unconvolved projected
  field.  It is written via ``makeMap(beamSize=0)`` rather than ``makeField`` so
  that ``nPixels`` is derived from the box's angular size exactly as the figure
  derives it -- the two must agree or the figure will not find the file.
- ``--dim 3D``: ``3D/..._{nPixels}.npy``, the cubic field.  The 3D cache
  filename carries no projection, so one file serves all three.

Each task skips itself when its output already exists, so an array job that
hits the walltime can simply be resubmitted.

Usage
-----
    python unbound_gas/precompute_baryonFraction_fields.py \
        --simtype FLAMINGO --sim L1_m9 --snapshot 67 --feedback L1_m9 \
        --ptype ionized_gas --dim 3D --n-pixels 1000

    python unbound_gas/precompute_baryonFraction_fields.py \
        --simtype IllustrisTNG --sim TNG300-1 --snapshot 67 \
        --ptype Stars --dim 2D --pixel-size 0.2

The companion runCPU_baryonFraction_precompute.sh submits the full task list as
a job array.
"""

import argparse
import gc
import sys
import time

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

sys.path.append('../src/')
from stacker import SimulationStacker  # type: ignore
from loadIO import _get_data_filepath  # type: ignore
from utils import comoving_to_arcmin  # type: ignore


def map_n_pixels(stacker: SimulationStacker, z: float, pixelSize: float) -> int:
    """Return the 2D grid size ``makeMap`` would derive for this pixel size.

    Mirrors ``SimulationStacker.makeMap``: the box's comoving side is converted
    to an angular size at redshift ``z`` and divided by the requested pixel
    size, rounding up.  Duplicated here only so the existence check can name the
    output file without building it; if ``makeMap`` ever changes its sizing this
    must follow.

    Parameters
    ----------
    stacker : SimulationStacker
    z : float
        Redshift at which the box is projected.
    pixelSize : float
        Target pixel size [arcmin].

    Returns
    -------
    int
        Number of pixels per side.
    """
    cosmo = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'],
                          Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K)
    theta_arcmin = comoving_to_arcmin(stacker.header['BoxSize'], z, cosmo=cosmo)
    return int(np.ceil(theta_arcmin / pixelSize))


def main(args: argparse.Namespace) -> int:
    """Build and cache one field, skipping the work if it is already on disk.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        0 on success (including a skip).
    """
    t0 = time.time()
    stacker = SimulationStacker(args.sim, args.snapshot, z=args.redshift,
                                simType=args.simtype, feedback=args.feedback)

    if args.dim == '3D':
        nPixels = args.n_pixels
    else:
        nPixels = map_n_pixels(stacker, args.redshift, args.pixel_size)

    target = _get_data_filepath(
        args.simtype, args.sim, args.snapshot, args.feedback, args.ptype,
        nPixels, projection=args.projection, data_type='field', dim=args.dim,
        base_path=stacker.base_path,
    )
    print(f"Target: {target}")
    print(f"  simType={args.simtype} sim={args.sim} feedback={args.feedback} "
          f"snapshot={args.snapshot} ptype={args.ptype} dim={args.dim} "
          f"nPixels={nPixels} projection={args.projection}")

    if target.exists() and not args.overwrite:
        print(f"  Already present ({target.stat().st_size / 1e9:.1f} GB); skipping.")
        return 0

    if args.dim == '3D':
        stacker.makeField(args.ptype, nPixels=nPixels, dim='3D',
                          projection=args.projection,
                          save=args.save, load=True)
    else:
        # beamSize=0 -> makeMap skips the convolution and saves the raw field,
        # which is exactly what the figure loads when beam_size: 0.
        stacker.makeMap(args.ptype, z=args.redshift, projection=args.projection,
                        pixelSize=args.pixel_size, beamSize=0,
                        save=args.save, load=True)

    gc.collect()
    print(f"  Done in {time.time() - t0:.1f}s -> {target}")
    if args.save and not target.exists():
        raise RuntimeError(f"Expected output {target} was not written.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Precompute one cached field for make_baryonFraction.py.')
    parser.add_argument('--simtype', type=str, required=True,
                        choices=['IllustrisTNG', 'SIMBA', 'FLAMINGO'],
                        help='Simulation suite.')
    parser.add_argument('--sim', type=str, required=True,
                        help="Simulation name, e.g. 'TNG300-1' or 'L1_m9'.")
    parser.add_argument('--snapshot', type=int, required=True,
                        help='Snapshot number.')
    parser.add_argument('--feedback', type=str, default=None,
                        help="Feedback variant; required for SIMBA and FLAMINGO.")
    parser.add_argument('--ptype', type=str, required=True,
                        help="Particle type, e.g. 'Stars', 'BH', 'ionized_gas'.")
    parser.add_argument('--dim', type=str, default='2D', choices=['2D', '3D'],
                        help="'2D' projected field or '3D' cubic field.")
    parser.add_argument('--projection', type=str, default='yz',
                        choices=['xy', 'xz', 'yz'],
                        help="Projection axis (2D only; 3D filenames carry none).")
    parser.add_argument('--pixel-size', type=float, default=0.2,
                        help='2D pixel size in arcmin. Default 0.2.')
    parser.add_argument('--n-pixels', type=int, default=1000,
                        help='3D grid size per side. Default 1000.')
    parser.add_argument('--redshift', type=float, default=0.5,
                        help='Snapshot redshift. Default 0.5.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Rebuild even if the output already exists.')
    parser.add_argument('--no-save', action='store_false', dest='save',
                        help='Build but do not write (validation runs).')
    args = parser.parse_args()
    print(f"Arguments: {vars(args)}")
    sys.exit(main(args))
