"""check_filter_compensation.py
=============================
Task 2 deliverable: show quantitatively why the frozen filter set is
{DSigma, Upsilon} and not Sigma.

The annulus-mean ("Sigma") kernel is uncompensated.  From the theory note's
Sec. 5.3, its transform is

    W_ann(k; R1, R2) = 2[R2 J1(kR2) - R1 J1(kR1)] / (k (R2^2 - R1^2)),

and since J1(x) -> x/2 as x -> 0, W_ann(k -> 0) -> 1.  A Sigma-filtered
amplitude therefore integrates power all the way down to the box fundamental
mode, which differs between simulations: 2*pi/(205 cMpc/h) for TNG300-1 against
2*pi/(681 cMpc/h) for FLAMINGO L1_m9.  Y_Sigma is consequently not the same
quantity in the two boxes.  The compensated DSigma and Upsilon kernels have
W(k -> 0) -> 0 and are immune.

The r's are ratios, in which the effect partly cancels, so the question this
script answers is not whether the mechanism exists (it provably does) but how
much of it survives into the coefficients.  For each simulation it recomputes
the coefficients after removing every mode longer than the smallest box in the
comparison, and reports the shift.  A large shift for Sigma and a negligible
one for DSigma/Upsilon is the evidence for the filter freeze.

It also reports whether removing those modes from FLAMINGO moves its Sigma
coefficient towards TNG300-1's, which tests whether the box difference
actually explains the cross-code disagreement seen for Sigma in Task 1.

Usage
-----
    cd scripts/
    python cross_corr/check_filter_compensation.py -p configs/cross_corr/r_profiles_z05.yaml
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
sys.path.append(str(Path(__file__).parent))
from make_r_profiles import aperture_radii, sim_label

#: Coefficients measured before and after the cut.  The galaxy-crossed pairs
#: are included so the test covers r_bm/r_gb, which is the quantity Gate A
#: actually thresholds, rather than only the clean field-field coefficients.
PAIRS = [('b', 'm'), ('e', 'm'), ('g', 'b'), ('g', 'e')]

#: Ratios measured before and after the cut, as (numerator, denominator).
RATIOS = [(('b', 'm'), ('g', 'b')), (('e', 'm'), ('g', 'e'))]

#: Comoving side length of the smallest box in the comparison, cMpc/h.
#: TNG300-1 sets the largest mode common to every retained simulation.
REFERENCE_BOX_CMPC_H = 205.0


def load_fields(stacker, n_pixels, projection):
    """Load the electron, baryon and CDM maps for one run.

    Args:
        stacker (SimulationStacker): Configured stacker.
        n_pixels (int): Cached grid size.
        projection (str): Projection direction.

    Returns:
        dict: Raw (positive) maps keyed 'e', 'b', 'm'.
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


def coefficients(deltas, pixel_arcmin, radii, dr, r0, nbar_pix):
    """Compute the coefficients and ratios from prepared overdensity maps.

    The same ``nbar_pix`` is used for the full and the high-pass filtered
    maps.  That is correct to well below the precision of this test: the cut
    removes only modes with ``k < 2*pi/cut``, a fraction of order
    ``(k_cut/k_Nyquist)^2 ~ 1e-6`` of all modes, so the white shot-noise
    spectrum the self-pair term removes is essentially untouched.

    Args:
        deltas (dict): Overdensity maps, including the galaxy field 'g'.
        pixel_arcmin (float): Angular pixel size in arcmin.
        radii (np.ndarray): Aperture radii in arcmin.
        dr (float): Annulus width in arcmin.
        r0 (float): Upsilon reference radius in arcmin.
        nbar_pix (float): Mean galaxy count per pixel.

    Returns:
        tuple: ``(prof, Ymat)``.
    """
    Ymat = rp.compute_Y_matrix(deltas, pixel_arcmin, radii=radii, dr=dr, r0=r0,
                               nbar_pix=nbar_pix)
    return rp.r_profiles(Ymat, pairs=PAIRS, ratios=RATIOS), Ymat


def process(sim_type, entry, cfg, radii, verbose=True):
    """Measure the low-k sensitivity of every filter for one simulation.

    Args:
        sim_type (str): Simulation suite.
        entry (dict): Config entry for the simulation.
        cfg (dict): The ``stack`` config block.
        radii (np.ndarray): Aperture radii in arcmin.
        verbose (bool, optional): Print the per-aperture tables.

    Returns:
        dict: Summary with the worst fractional shift per filter, plus the
        full and cut coefficient arrays for the cross-code comparison.
    """
    name, snapshot = entry['name'], entry['snapshot']
    feedback = entry.get('feedback')
    n_pixels = entry['n_pixels']
    projection = cfg.get('projections', ['yz'])[0]
    dr = float(cfg.get('dr_arcmin', rp.DR_ARCMIN))
    r0 = float(cfg.get('r0_arcmin', rp.R0_ARCMIN))

    stacker = SimulationStacker(name, snapshot, nPixels=n_pixels,
                                simType=sim_type, feedback=feedback,
                                z=float(cfg.get('redshift', 0.5)))
    z_true = float(np.asarray(stacker.header['Redshift']).ravel()[0])
    lbox = float(stacker.header['BoxSize'])
    theta = comoving_to_arcmin(lbox, z_true, cosmo=stacker.cosmo)
    pixel_arcmin = theta / n_pixels

    # The reference box, expressed as an angular scale at this redshift.
    cut_arcmin = comoving_to_arcmin(REFERENCE_BOX_CMPC_H * 1000.0, z_true,
                                    cosmo=stacker.cosmo)

    label = sim_label(sim_type, name, feedback)
    print(f'\n{"=" * 72}\n{label} snapshot {snapshot} (z={z_true:.4f})\n'
          f'{"=" * 72}')
    print(f'  box {lbox / 1000:.0f} cMpc/h = {theta:.1f} arcmin, '
          f'grid {n_pixels}^2, pixel {pixel_arcmin:.5f} arcmin')
    print(f'  removing modes longer than {cut_arcmin:.1f} arcmin '
          f'(= {REFERENCE_BOX_CMPC_H:.0f} cMpc/h, the smallest retained box)')
    print(f'  largest aperture {radii.max():.2f} arcmin; cut/aperture ratio '
          f'{cut_arcmin / radii.max():.1f}')

    fields = load_fields(stacker, n_pixels, projection)
    deltas = {k: rp.to_overdensity(v) for k, v in fields.items()}
    del fields

    galaxies, halo_mask = rp.make_galaxy_field(
        stacker, projection, n_pixels,
        float(cfg.get('halo_abundance_target', 5e-4)),
        parent_mass_upper=(None if cfg.get('parent_mass_upper') is None
                           else float(cfg['parent_mass_upper'])))
    nbar_pix = galaxies.sum() / float(n_pixels * n_pixels)
    deltas['g'] = rp.to_overdensity(galaxies)
    del galaxies
    print(f'  {halo_mask.size} SHAM galaxies (nbar = {nbar_pix:.4e} per pixel)')

    full_prof, full_Y = coefficients(deltas, pixel_arcmin, radii, dr, r0,
                                     nbar_pix)

    cut_deltas = {k: rp.highpass_field(v, pixel_arcmin, cut_arcmin)
                  for k, v in deltas.items()}
    del deltas
    cut_prof, cut_Y = coefficients(cut_deltas, pixel_arcmin, radii, dr, r0,
                                   nbar_pix)
    del cut_deltas

    summary = {'label': label, 'radii': radii, 'worst': {},
               'r_full': {}, 'r_cut': {},
               'ratio_full': {}, 'ratio_cut': {}}
    for filt in rp.FILTERS:
        worst_Y = 0.0
        worst_r = 0.0
        for pair in PAIRS:
            ya = rp.get_Y(full_Y, filt, *pair)
            yb = rp.get_Y(cut_Y, filt, *pair)
            ra = full_prof['r'][pair][filt]
            rb = cut_prof['r'][pair][filt]
            with np.errstate(invalid='ignore', divide='ignore'):
                dy = np.abs(yb / ya - 1.0)
                dr_ = np.abs(rb / ra - 1.0)
            worst_Y = max(worst_Y, float(np.nanmax(dy)))
            worst_r = max(worst_r, float(np.nanmax(dr_)))
        summary['worst'][filt] = {'Y': worst_Y, 'r': worst_r}
        summary['r_full'][filt] = full_prof['r'][('e', 'm')][filt]
        summary['r_cut'][filt] = cut_prof['r'][('e', 'm')][filt]
        gate = (('b', 'm'), ('g', 'b'))
        summary['ratio_full'][filt] = full_prof['ratio'][gate][filt]
        summary['ratio_cut'][filt] = cut_prof['ratio'][gate][filt]
        with np.errstate(invalid='ignore', divide='ignore'):
            d_ratio = np.abs(summary['ratio_cut'][filt]
                             / summary['ratio_full'][filt] - 1.0)
        summary['worst'][filt]['ratio'] = float(np.nanmax(d_ratio))
        print(f'  {filt:8s} worst |dY/Y| = {worst_Y:9.3e}   '
              f'worst |dr/r| = {worst_r:9.3e}   '
              f"worst |d(r_bm/r_gb)| = {summary['worst'][filt]['ratio']:9.3e}")

    if verbose:
        print(f"\n  r_em before and after the cut:")
        print(f"    {'R':>7} " + " ".join(f'{f:>22s}' for f in rp.FILTERS))
        for i, R in enumerate(radii):
            cells = []
            for filt in rp.FILTERS:
                cells.append(f"{summary['r_full'][filt][i]:9.4f}->"
                             f"{summary['r_cut'][filt][i]:9.4f}")
            print(f'    {R:7.3f} ' + " ".join(f'{c:>22s}' for c in cells))
    return summary


def main(path2config):
    """Run the compensation check for every simulation in a config.

    Args:
        path2config (str): Path to the YAML configuration file.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)
    cfg = config.get('stack', {})

    radii = aperture_radii(cfg)

    summaries = []
    for suite in config['simulations']:
        for entry in suite['sims']:
            summaries.append(process(suite['sim_type'], entry, cfg, radii))

    print(f'\n{"=" * 72}\nSummary: sensitivity to modes longer than '
          f'{REFERENCE_BOX_CMPC_H:.0f} cMpc/h\n{"=" * 72}')
    print(f"  {'run':26s} " + " ".join(f'{f + " (Y / r)":>24s}'
                                       for f in rp.FILTERS))
    for s in summaries:
        cells = [f"{s['worst'][f]['Y']:9.2e} /{s['worst'][f]['r']:9.2e}"
                 for f in rp.FILTERS]
        print(f"  {s['label']:26s} " + " ".join(f'{c:>24s}' for c in cells))

    # Does removing FLAMINGO's extra large-scale modes move its Sigma
    # coefficient towards TNG300-1's?  If so, the box difference explains the
    # cross-code disagreement Task 1 saw for Sigma.
    tng = next((s for s in summaries if s['label'].startswith('TNG')), None)
    flam = next((s for s in summaries
                 if s['label'] == 'L1_m9_fiducial'), None)
    if flam is None:
        flam = next((s for s in summaries if 'L1_m9' in s['label']),
                    None)
    if tng is not None and flam is not None:
        print(f'\n  Does the box difference explain the cross-code '
              f'disagreement?')
        print(f'    Both simulations are cut at the same physical scale, so '
              f'this is')
        print(f'    a like-for-like comparison; TNG is cut at its own box '
              f'size, which')
        print(f'    is a verified no-op.')
        for quantity, fk, ck in (('r_em', 'r_full', 'r_cut'),
                                 ('r_bm/r_gb (Gate A)', 'ratio_full',
                                  'ratio_cut')):
            print(f'    {quantity}:')
            for filt in rp.FILTERS:
                with np.errstate(invalid='ignore'):
                    before = np.abs(flam[fk][filt] / tng[fk][filt] - 1)
                    after = np.abs(flam[ck][filt] / tng[ck][filt] - 1)
                print(f'      {filt:8s} max|FLAM/TNG - 1| = '
                      f'{np.nanmax(before):.3e} before, '
                      f'{np.nanmax(after):.3e} after')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quantify low-k (box-size) sensitivity of each filter.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/r_profiles_z05.yaml')
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
