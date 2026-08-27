"""check_theory_transfer.py
=========================
Task 4 deliverable: build and validate the theory chain for Y_mm.

The comparison is decomposed so that separate error sources cannot hide inside
one another:

**A. Transfer chain.** Measure the CDM map's own 2D power spectrum, push it
through the analytic aperture kernel of Sec. 5.3, and compare with the
amplitude the pipeline obtains by real-space filtering the same map.  No
cosmological model enters, so any disagreement is a failure of the harmonic
transfer itself -- pixelization of the aperture, k-grid resolution, or the
``1/pixArea`` normalization convention.

**B. CDM versus total matter.** halofit returns the total matter spectrum
while the pipeline's ``m`` field is CDM only.  Both maps exist, so the size of
that conflation is measured rather than assumed.

**C. halofit accuracy.** Replace the measured spectrum with halofit at the
simulation's cosmology and compare again.  What is left, after A and B are
known, is the error of the non-linear model itself.

Two approximations are adopted and reported in the output rather than buried:
``P_mm^hydro-CDM / P_mm^DMO = 1`` (no DMO run is available on disk; the true
value is 1-2 per cent, van Daalen et al. 2011, Chisari et al. 2018), and the
``n_s``/``sigma8`` values, which no simulation header carries and which are
therefore literature lookups.

Usage
-----
    cd scripts/
    python cross_corr/check_theory_transfer.py -p configs/cross_corr/r_profiles_z05.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.append('../src/')
import kernels as kn
import rprofiles as rp
import theory as th
from stacker import SimulationStacker
from utils import arcmin_to_comoving, comoving_to_arcmin
sys.path.append(str(Path(__file__).parent))
from make_r_profiles import aperture_radii, sim_label

#: Filters carried through the theory chain.  Sigma is included as a
#: diagnostic only; it is not in the frozen set (see Gate B).
FILTERS = ('DSigma', 'Upsilon')


def load_matter_maps(stacker, n_pixels, projection):
    """Load the CDM and total-matter overdensity maps.

    Args:
        stacker (SimulationStacker): Configured stacker.
        n_pixels (int): Cached grid size.
        projection (str): Projection direction.

    Returns:
        tuple: ``(delta_cdm, delta_total)``.
    """
    baryon = stacker.makeField('baryon', nPixels=n_pixels,
                               projection=projection, save=False, load=True)
    total = stacker.makeField('total', nPixels=n_pixels,
                              projection=projection, save=False, load=True)
    cdm = rp.derive_cdm_field(total, baryon, header=stacker.header,
                              verbose=False)
    del baryon
    return rp.to_overdensity(cdm), rp.to_overdensity(total)


def process(sim_type, entry, cfg, radii_arcmin):
    """Run the three-way comparison for one simulation.

    Args:
        sim_type (str): Simulation suite.
        entry (dict): Config entry.
        cfg (dict): The ``stack`` config block.
        radii_arcmin (np.ndarray): Aperture radii in arcmin.

    Returns:
        dict: Summary of the worst fractional differences.
    """
    name, snapshot = entry['name'], entry['snapshot']
    feedback = entry.get('feedback')
    n_pixels = entry['n_pixels']
    projection = cfg.get('projections', ['yz'])[0]
    dr_arcmin = float(cfg.get('dr_arcmin', rp.DR_ARCMIN))
    r0_arcmin = float(cfg.get('r0_arcmin', rp.R0_ARCMIN))
    label = sim_label(sim_type, name, feedback)

    stacker = SimulationStacker(name, snapshot, nPixels=n_pixels,
                                simType=sim_type, feedback=feedback,
                                z=float(cfg.get('redshift', 0.5)))
    z_true = float(np.asarray(stacker.header['Redshift']).ravel()[0])
    lbox_kpc_h = float(stacker.header['BoxSize'])
    lbox_mpc_h = lbox_kpc_h / 1000.0
    theta = comoving_to_arcmin(lbox_kpc_h, z_true, cosmo=stacker.cosmo)
    pixel_arcmin = theta / n_pixels

    print(f'\n{"=" * 74}\n{label} snapshot {snapshot} (z={z_true:.4f})\n'
          f'{"=" * 74}')
    print(f'  box {lbox_mpc_h:.0f} cMpc/h = {theta:.1f} arcmin, '
          f'pixel {pixel_arcmin:.5f} arcmin')

    delta_cdm, delta_total = load_matter_maps(stacker, n_pixels, projection)

    # Amplitudes the pipeline measures by real-space filtering.
    Y_cdm = rp.compute_Y_matrix({'m': delta_cdm}, pixel_arcmin,
                                radii=radii_arcmin, dr=dr_arcmin, r0=r0_arcmin)
    Y_tot = rp.compute_Y_matrix({'m': delta_total}, pixel_arcmin,
                                radii=radii_arcmin, dr=dr_arcmin, r0=r0_arcmin)

    # A. Transfer chain, using the map's own spectrum. Work in arcmin
    # throughout so no distance conversion enters this comparison.
    k_arcmin, p2d_arcmin = th.measured_p2d_from_map(delta_cdm, pixel_arcmin)
    del delta_cdm, delta_total

    # C. halofit at the simulation's own cosmology, projected through the box.
    cosmo_params = th.cosmology_for(sim_type, stacker.header)
    k_mpc, p3d = th.halofit_power(z=z_true, **cosmo_params)
    p2d_mpc = th.project_periodic_box(p3d, lbox_mpc_h)
    radii_mpc = arcmin_to_comoving(radii_arcmin, z_true,
                                   cosmo=stacker.cosmo) / 1000.0
    dr_mpc = float(arcmin_to_comoving(dr_arcmin, z_true,
                                      cosmo=stacker.cosmo)) / 1000.0
    r0_mpc = float(arcmin_to_comoving(r0_arcmin, z_true,
                                      cosmo=stacker.cosmo)) / 1000.0
    print(f"  halofit: n_s={cosmo_params['n_s']}, "
          f"sigma8={cosmo_params['sigma8']}, m_nu={cosmo_params['m_nu']} eV "
          f'(n_s/sigma8 are literature values, not from the header)')

    summary = {'label': label, 'A': [], 'B': [], 'C': []}
    for filt in FILTERS:
        measured_cdm = rp.get_Y(Y_cdm, filt, 'm', 'm')
        measured_tot = rp.get_Y(Y_tot, filt, 'm', 'm')
        transfer = th.amplitudes_from_p2d(
            k_arcmin, p2d_arcmin, radii_arcmin, filt, dr_arcmin, r0_arcmin,
            pixel_arcmin=pixel_arcmin)
        halofit = th.amplitudes_from_p2d(
            k_mpc, p2d_mpc, radii_mpc, filt, dr_mpc, r0_mpc,
            pixel_arcmin=pixel_arcmin)

        with np.errstate(invalid='ignore', divide='ignore'):
            a = transfer / measured_cdm - 1.0
            b = measured_tot / measured_cdm - 1.0
            c = halofit / measured_cdm - 1.0

        print(f'\n  {filt}:')
        print(f"    {'R':>7} {'measured':>12} {'A transfer':>12} "
              f"{'B tot/CDM':>11} {'C halofit':>11}")
        for i, R in enumerate(radii_arcmin):
            if not np.isfinite(measured_cdm[i]):
                continue
            print(f'    {R:7.3f} {measured_cdm[i]:12.5e} {a[i]:+12.3e} '
                  f'{b[i]:+11.3e} {c[i]:+11.3e}')
        for key, arr in (('A', a), ('B', b), ('C', c)):
            finite = arr[np.isfinite(arr)]
            worst = float(np.max(np.abs(finite))) if finite.size else np.nan
            summary[key].append((filt, worst))
            print(f'    worst |{key}| = {worst:.3e}')
    return summary


def main(path2config, sim=None):
    """Run the theory-transfer validation for every simulation in a config.

    Args:
        path2config (str): Path to the YAML configuration file.
        sim (str, optional): Restrict to this simulation name.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)
    cfg = config.get('stack', {})
    radii = aperture_radii(cfg)

    summaries = []
    for suite in config['simulations']:
        for entry in suite['sims']:
            if sim is not None and entry['name'] != sim:
                continue
            summaries.append(process(suite['sim_type'], entry, cfg, radii))

    print(f'\n{"=" * 74}\nSummary: worst fractional difference against the '
          f'measured CDM amplitude\n{"=" * 74}')
    print('  A = transfer chain (map spectrum -> analytic kernel), no model')
    print('  B = total matter instead of CDM (halofit returns total matter)')
    print('  C = halofit instead of the measured spectrum')
    print(f"\n  {'run':26s} {'filter':9s} {'A':>11} {'B':>11} {'C':>11}")
    for s in summaries:
        for i, (filt, _) in enumerate(s['A']):
            print(f"  {s['label'] if i == 0 else '':26s} {filt:9s} "
                  f"{s['A'][i][1]:11.3e} {s['B'][i][1]:11.3e} "
                  f"{s['C'][i][1]:11.3e}")
    print('\n  Adopted approximation: P_mm^hydro-CDM / P_mm^DMO = 1.')
    print('  Known to be wrong at the 1-2 per cent level (van Daalen 2011,')
    print('  Chisari 2018); adopted because no DMO run is on disk.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate the P(k) -> Y(R) theory chain against the maps.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/r_profiles_z05.yaml')
    parser.add_argument('--sim', type=str, default=None,
                        help='Restrict to this simulation name.')
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
