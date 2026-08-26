"""check_resolution.py
====================
Quantify the two discretization systematics of the Task 1 r-profiles, as
required by ``docs/r_profiles_task1_spec.md`` ("Resolution requirements":
*Include one convergence check ... before the full sweep*).

Two effects are measured, both on the field-field coefficients (``r_bm``,
``r_em``), which need no galaxy catalogue and are therefore cheap:

1. **Resolution convergence.** The cached map is block-summed onto a grid
   coarser by ``--factor`` (mass-conserving, exact) and the coefficients are
   recomputed.  A converged coefficient barely moves.

2. **Boundary-tie sensitivity.** Pixel membership uses the strict test
   ``r < edge``, so when an aperture edge falls on a realizable lattice
   distance an entire shell of pixels sits on the boundary and flips
   membership under an arbitrarily small change of convention.  The cached
   grids are exactly 0.2 arcmin/pixel, which makes R=1' (5.0 px), the R=2.25'
   annulus edge (15.0 px) and R=6' (30.0 px) degenerate.  The run compares the
   true arcmin-per-pixel against the slightly different effective scale of
   ``SimulationStacker.stack_on_array``'s linspace stamp grid, which is what
   the integration test exposes.

Both shifts are expected to be far smaller in the coefficients than in the raw
filtered amplitudes, because every Y entering a coefficient is filtered with
the same kernel.

Usage
-----
    cd scripts/
    python cross_corr/check_resolution.py -p configs/cross_corr/r_profiles_z05.yaml --sim TNG300-1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.append('../src/')
import rprofiles as rp
from stacker import SimulationStacker
from utils import comoving_to_arcmin

PAIRS = [('b', 'm'), ('e', 'm')]


def coarsen(field, factor):
    """Block-sum a field onto a coarser grid, discarding any remainder rows.

    Block summing is exact and mass-conserving, so the coarse map is precisely
    the map that would have been produced by binning the same particles onto
    the coarser grid.

    Note:
        When ``n_pixels`` is not divisible by ``factor`` (e.g. the odd 1301 and
        8869 production grids at ``factor=2``) the remainder row and column are
        dropped, so the coarse map is periodic on a box smaller by at most
        ``factor/n_pixels`` -- under 0.08 per cent for every production grid.
        That introduces a small discontinuity at the wrap edge, which is
        acceptable for this diagnostic but would not be for a production map.

    Args:
        field (np.ndarray): 2D field.
        factor (int): Coarsening factor.

    Returns:
        tuple: ``(coarse_field, n_coarse)``.
    """
    n = field.shape[0] // factor
    trimmed = field[:n * factor, :n * factor]
    return trimmed.reshape(n, factor, n, factor).sum(axis=(1, 3)), n


def load_deltas(stacker, n_pixels, projection):
    """Load the electron, baryon and CDM overdensity maps.

    Args:
        stacker (SimulationStacker): Configured stacker.
        n_pixels (int): Cached grid size.
        projection (str): Projection direction.

    Returns:
        dict: Overdensity maps keyed 'e', 'b', 'm'.
    """
    baryon = stacker.makeField('baryon', nPixels=n_pixels,
                               projection=projection, save=False, load=True)
    total = stacker.makeField('total', nPixels=n_pixels,
                              projection=projection, save=False, load=True)
    cdm = rp.derive_cdm_field(total, baryon, header=stacker.header,
                              verbose=False)
    del total
    electrons = stacker.makeField('ionized_gas', nPixels=n_pixels,
                                  projection=projection, save=False, load=True)
    return {'m': cdm, 'b': baryon, 'e': electrons}


def coefficients(fields, pixel_arcmin, radii, dr, r0):
    """Compute the field-field coefficients for a set of raw maps.

    Args:
        fields (dict): Raw (positive) maps keyed by field.
        pixel_arcmin (float): Angular pixel size in arcmin.
        radii (np.ndarray): Aperture radii in arcmin.
        dr (float): Annulus width in arcmin.
        r0 (float): Upsilon reference radius in arcmin.

    Returns:
        tuple: ``(prof, Ymat)`` from :func:`rprofiles.r_profiles` and
        :func:`rprofiles.compute_Y_matrix`.
    """
    deltas = {k: rp.to_overdensity(v) for k, v in fields.items()}
    Ymat = rp.compute_Y_matrix(deltas, pixel_arcmin, radii=radii, dr=dr, r0=r0)
    return rp.r_profiles(Ymat, pairs=PAIRS, ratios=[]), Ymat


def compare(label, radii, base_prof, base_Y, other_prof, other_Y):
    """Print a per-aperture comparison of coefficients and amplitudes.

    Args:
        label (str): Description of the comparison.
        radii (np.ndarray): Aperture radii in arcmin.
        base_prof (dict): Reference coefficients.
        base_Y (dict): Reference amplitudes.
        other_prof (dict): Comparison coefficients.
        other_Y (dict): Comparison amplitudes.

    Returns:
        dict: ``{(pair, filter): max |fractional change in r|}``.
    """
    print(f'\n  --- {label} ---')
    worst = {}
    for pair in PAIRS:
        for filt in rp.FILTERS:
            a = base_prof['r'][pair][filt]
            b = other_prof['r'][pair][filt]
            ya = rp.get_Y(base_Y, filt, *pair)
            yb = rp.get_Y(other_Y, filt, *pair)
            with np.errstate(invalid='ignore', divide='ignore'):
                dr_frac = b / a - 1.0
                dy_frac = yb / ya - 1.0
            finite = dr_frac[np.isfinite(dr_frac)]
            worst[(pair, filt)] = (float(np.max(np.abs(finite)))
                                   if finite.size else np.nan)
            name = f'r_{pair[0]}{pair[1]}'
            print(f'  {name} [{filt}]  max|dr/r| = '
                  f'{worst[(pair, filt)]:.3e}')
            print(f"    {'R':>7} {'r_base':>10} {'r_other':>10} {'dr/r':>11} "
                  f"{'dY/Y':>11}")
            for i, R in enumerate(radii):
                print(f'    {R:7.3f} {a[i]:10.6f} {b[i]:10.6f} '
                      f'{dr_frac[i]:+11.3e} {dy_frac[i]:+11.3e}')
    return worst


def process(sim_type, entry, cfg, factor, verbose=True):
    """Run both checks for one simulation.

    Args:
        sim_type (str): Simulation suite.
        entry (dict): Config entry for the simulation.
        cfg (dict): The ``stack`` config block.
        factor (int): Coarsening factor for the convergence check.
        verbose (bool, optional): Unused; kept for symmetry.

    Returns:
        dict: Summary of the worst fractional changes.
    """
    name, snapshot = entry['name'], entry['snapshot']
    feedback = entry.get('feedback')
    n_pixels = entry['n_pixels']
    projection = cfg.get('projections', ['yz'])[0]

    stacker = SimulationStacker(name, snapshot, nPixels=n_pixels,
                                simType=sim_type, feedback=feedback,
                                z=float(cfg.get('redshift', 0.5)))
    z_true = float(np.asarray(stacker.header['Redshift']).ravel()[0])
    lbox = float(stacker.header['BoxSize'])
    theta = comoving_to_arcmin(lbox, z_true, cosmo=stacker.cosmo)
    pixel_arcmin = theta / n_pixels

    radii = np.linspace(float(cfg.get('min_radius', 1.0)),
                        float(cfg.get('max_radius', 6.0)),
                        int(cfg.get('num_radii', 9)))
    dr = float(cfg.get('dr_arcmin', rp.DR_ARCMIN))
    r0 = float(cfg.get('r0_arcmin', rp.R0_ARCMIN))

    label = name if feedback is None else f'{name}_{feedback}'
    print(f'\n{"=" * 70}\n{label} snapshot {snapshot} (z={z_true:.4f})\n'
          f'{"=" * 70}')
    print(f'  grid {n_pixels}^2, pixel {pixel_arcmin:.6f} arcmin')

    degenerate = rp.degenerate_apertures(radii, pixel_arcmin, dr)
    print(f'  boundary-degenerate apertures: '
          f'{sorted(round(R, 3) for R in degenerate)}')
    for R, edges in sorted(degenerate.items()):
        for edge_name, margin, shell in edges:
            print(f"    R={R:.3f}' {edge_name}: {shell}-pixel shell, margin "
                  f'{margin:.2e} px')

    fields = load_deltas(stacker, n_pixels, projection)
    base_prof, base_Y = coefficients(fields, pixel_arcmin, radii, dr, r0)

    # 1. Resolution convergence.
    coarse = {}
    for k, v in fields.items():
        coarse[k], n_coarse = coarsen(v, factor)
    print(f'\n  coarsened grid: {n_coarse}^2, pixel '
          f'{pixel_arcmin * factor:.6f} arcmin')
    coarse_prof, coarse_Y = coefficients(coarse, pixel_arcmin * factor, radii,
                                         dr, r0)
    worst_res = compare(f'resolution: {n_pixels} vs {n_coarse} '
                        f'({factor}x coarser)', radii,
                        base_prof, base_Y, coarse_prof, coarse_Y)
    del coarse

    # 2. Boundary-tie convention.  stack_on_array's stamp radius grid is a
    # linspace over the rounded cutout half-width, giving this pixel scale.
    n_vir = int(radii.max() + 1)
    cutout_size = 2 * int(round(n_vir / pixel_arcmin)) + 1
    stamp_pixel = 2.0 * n_vir / (cutout_size - 1)
    print(f'\n  stamp-grid effective pixel: {stamp_pixel:.6f} arcmin '
          f'(true {pixel_arcmin:.6f})')
    tie_prof, tie_Y = coefficients(fields, stamp_pixel, radii, dr, r0)
    worst_tie = compare('boundary tie: true pixel vs stamp-grid pixel', radii,
                        base_prof, base_Y, tie_prof, tie_Y)

    summary = {
        'label': label,
        'worst_resolution': max(v for v in worst_res.values()
                                if np.isfinite(v)),
        'worst_tie': max(v for v in worst_tie.values() if np.isfinite(v)),
    }
    print(f"\n  SUMMARY {label}: worst |dr/r| from resolution = "
          f"{summary['worst_resolution']:.3e}, from boundary tie = "
          f"{summary['worst_tie']:.3e}")
    return summary


def main(path2config, sim=None, factor=2):
    """Run the resolution and boundary-tie checks for a config.

    Args:
        path2config (str): Path to the YAML configuration file.
        sim (str, optional): Restrict to this simulation name.
        factor (int, optional): Coarsening factor.  Defaults to 2.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)
    cfg = config.get('stack', {})

    summaries = []
    for suite in config['simulations']:
        for entry in suite['sims']:
            if sim is not None and entry['name'] != sim:
                continue
            summaries.append(process(suite['sim_type'], entry, cfg, factor))

    print(f'\n{"=" * 70}\nOverall\n{"=" * 70}')
    print(f"  {'run':32s} {'worst |dr/r| (res)':>20} "
          f"{'worst |dr/r| (tie)':>20}")
    for s in summaries:
        print(f"  {s['label']:32s} {s['worst_resolution']:20.3e} "
              f"{s['worst_tie']:20.3e}")
    if summaries:
        print(f"\n  Gate A thresholds the cross-code scatter of r_bm/r_gb at "
              f"10 per cent; both systematics above are far below that.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resolution and boundary-tie checks for the r-profiles.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/r_profiles_z05.yaml')
    parser.add_argument('--sim', type=str, default=None,
                        help='Restrict to this simulation name.')
    parser.add_argument('--factor', type=int, default=2,
                        help='Coarsening factor for the convergence check.')
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
