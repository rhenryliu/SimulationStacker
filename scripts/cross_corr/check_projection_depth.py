"""check_projection_depth.py
==========================
Task 4, third bullet, simulation-side: how do the filtered amplitudes and the
coefficients respond to the depth of the line-of-sight projection?

The measurement pipeline projects through the entire simulation box, so a
measured Y carries the box depth: ``P_2D = P_3D / L``, and the TNG300-1 and
FLAMINGO amplitudes differ by very nearly their box ratio.  The data chain
projects something else entirely -- ``Y_gg`` comes from cylinders of
half-length ``Pi_max = 100 h^-1 Mpc``, while the kSZ integrates the whole line
of sight.  Since Eq. (4) of the theory note combines ``Y_gb / sqrt(Y_gg Y_mm)``,
a net one power of Y, those conventions do not cancel the way they do in every
coefficient.

This script slices the cached 3D fields into slabs of varying thickness,
projects each slab, and reports how the amplitudes and the coefficients move.
The amplitudes are expected to scale roughly as the inverse depth; the
question worth answering is how much of that survives into the coefficients,
which is what the calibration actually delivers.

The data-side recheck of the RSD and Pi_max treatment for ``Y_gg`` belongs to
Phase 4, once the pair-count pipeline exists; this is its simulation-side
proxy.

Usage
-----
    cd scripts/
    python cross_corr/check_projection_depth.py --sim TNG300-1 --snapshot 67
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append('../src/')
import rprofiles as rp
from loadIO import _get_data_filepath
from stacker import SimulationStacker
from utils import comoving_to_arcmin

#: Fractions of the box depth to project through.
DEPTH_FRACTIONS = (0.125, 0.25, 0.5, 1.0)

#: Filters reported.  Sigma is excluded: it is not in the frozen set and its
#: amplitude is not comparable between projection depths in the first place.
FILTERS = ('DSigma', 'Upsilon')


def load_3d(sim_type, sim, snapshot, feedback, ptype, grid):
    """Memory-map a cached 3D field.

    Args:
        sim_type (str): Simulation suite.
        sim (str): Simulation name.
        snapshot (int): Snapshot number.
        feedback (str or None): Feedback variant.
        ptype (str): Particle type.
        grid (int): Grid size per side.

    Returns:
        np.ndarray: Memory-mapped array, shape ``(grid, grid, grid)``.

    Raises:
        FileNotFoundError: If the cached field does not exist.
    """
    path = _get_data_filepath(sim_type, sim, snapshot, feedback, ptype, grid,
                             projection='xy', data_type='field', dim='3D')
    if not Path(path).exists():
        raise FileNotFoundError(f'No cached 3D field at {path}')
    return np.load(path, mmap_mode='r')


def slab_projections(field, n_slabs, axis=2):
    """Project a 3D field into ``n_slabs`` disjoint slabs along one axis.

    Reading through a memory map slab by slab keeps peak memory at one 2D map
    plus one slab rather than the whole cube.

    Args:
        field (np.ndarray): Memory-mapped 3D field.
        n_slabs (int): Number of equal slabs; must divide the grid.
        axis (int, optional): Projection axis.  Defaults to 2.

    Returns:
        list: One 2D float64 array per slab.
    """
    n = field.shape[axis]
    edges = np.linspace(0, n, n_slabs + 1).astype(int)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sl = [slice(None)] * 3
        sl[axis] = slice(lo, hi)
        out.append(np.asarray(field[tuple(sl)], dtype=np.float64).sum(axis=axis))
    return out


def main(sim, snapshot, sim_type, feedback, grid, redshift, min_radius):
    """Measure the depth dependence of the amplitudes and coefficients.

    Args:
        sim (str): Simulation name.
        snapshot (int): Snapshot number.
        sim_type (str): Simulation suite.
        feedback (str or None): Feedback variant.
        grid (int): Cached 3D grid size.
        redshift (float): Config redshift (the header value is authoritative).
        min_radius (float): Smallest aperture to report, in arcmin.
    """
    stacker = SimulationStacker(sim, snapshot, nPixels=grid, simType=sim_type,
                               feedback=feedback, z=redshift)
    z_true = float(np.asarray(stacker.header['Redshift']).ravel()[0])
    lbox = float(stacker.header['BoxSize'])
    theta = comoving_to_arcmin(lbox, z_true, cosmo=stacker.cosmo)
    pixel_arcmin = theta / grid

    radii = rp.APERTURES_ARCMIN[rp.APERTURES_ARCMIN >= min_radius]
    print(f'{sim} snapshot {snapshot} (z={z_true:.4f}), 3D grid {grid}^3')
    print(f'  box {lbox / 1000:.0f} cMpc/h, cell {pixel_arcmin:.4f} arcmin '
          f'({lbox / grid:.0f} ckpc/h)')
    print(f'  apertures {radii.min():.2f}-{radii.max():.2f} arcmin; the '
          f'smallest is {radii.min() / pixel_arcmin:.1f} cells')

    total = load_3d(sim_type, sim, snapshot, feedback, 'total', grid)
    baryon = load_3d(sim_type, sim, snapshot, feedback, 'baryon', grid)
    ion = load_3d(sim_type, sim, snapshot, feedback, 'ionized_gas', grid)

    results = {}
    for frac in DEPTH_FRACTIONS:
        n_slabs = int(round(1.0 / frac))
        depth_mpc = lbox / 1000.0 * frac
        tot_s = slab_projections(total, n_slabs)
        bar_s = slab_projections(baryon, n_slabs)
        ion_s = slab_projections(ion, n_slabs)

        y_acc, r_acc = [], []
        for t, b, e in zip(tot_s, bar_s, ion_s):
            cdm = rp.derive_cdm_field(t, b, verbose=False)
            deltas = {'m': rp.to_overdensity(cdm),
                      'e': rp.to_overdensity(e)}
            Ymat = rp.compute_Y_matrix(deltas, pixel_arcmin, radii=radii)
            prof = rp.r_profiles(Ymat, pairs=[('e', 'm')], ratios=[])
            y_acc.append({f: rp.get_Y(Ymat, f, 'm', 'm') for f in FILTERS})
            r_acc.append({f: prof['r'][('e', 'm')][f] for f in FILTERS})
        results[frac] = {
            'depth': depth_mpc,
            'n_slabs': n_slabs,
            'Y': {f: np.mean([a[f] for a in y_acc], axis=0) for f in FILTERS},
            'r': {f: np.mean([a[f] for a in r_acc], axis=0) for f in FILTERS},
        }
        print(f'  depth {depth_mpc:6.1f} cMpc/h ({n_slabs} slabs) done',
              flush=True)

    ref = results[1.0]
    for filt in FILTERS:
        print(f'\n{"=" * 70}\n{filt}: amplitude and coefficient versus '
              f'projection depth\n{"=" * 70}')
        headers = [f"{results[f]['depth']:.0f} Mpc/h" for f in DEPTH_FRACTIONS]
        print(f"  {'R':>6} " + "".join(f'{h:>22}' for h in headers))
        print(f"  {'':6} " + "".join(f"{'Y ratio / r':>22}"
                                     for _ in DEPTH_FRACTIONS))
        for i, R in enumerate(radii):
            cells = []
            for f in DEPTH_FRACTIONS:
                yr = results[f]['Y'][filt][i] / ref['Y'][filt][i]
                rv = results[f]['r'][filt][i]
                cells.append(f'{yr:9.3f} / {rv:9.4f}')
            print(f'  {R:6.2f} ' + "".join(f'{c:>22}' for c in cells))
        # How much does the coefficient move between the thinnest and the
        # full-depth projection?  This is the number that matters, since the
        # amplitude convention is fixed by the specification but the
        # coefficient is what the calibration delivers.
        with np.errstate(invalid='ignore'):
            drift = np.abs(results[min(DEPTH_FRACTIONS)]['r'][filt]
                           / ref['r'][filt] - 1.0)
        finite = drift[np.isfinite(drift)]
        if finite.size:
            print(f'  worst |r(thinnest)/r(full) - 1| = '
                  f'{np.max(finite):.3e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Projection-depth dependence of amplitudes and r.')
    parser.add_argument('--sim', type=str, default='TNG300-1')
    parser.add_argument('--snapshot', type=int, default=67)
    parser.add_argument('--sim-type', dest='sim_type', type=str,
                        default='IllustrisTNG')
    parser.add_argument('--feedback', type=str, default=None)
    parser.add_argument('--grid', type=int, default=1000,
                        help='Cached 3D grid size.')
    parser.add_argument('--redshift', type=float, default=0.5)
    parser.add_argument('--min-radius', dest='min_radius', type=float,
                        default=2.0,
                        help='Smallest aperture in arcmin; the coarse 3D grid '
                             'under-resolves 1 arcmin.')
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
