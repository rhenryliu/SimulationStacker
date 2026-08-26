"""make_r_profiles.py
====================
Compute the Task 1 cross-correlation coefficients r_gb, r_bm, r_ge, r_em and
the ratio r_bm/r_gb for the filters {Sigma, DSigma, Upsilon(R0=1')} over nine
linear aperture bins in 1'-6', for every simulation listed in a YAML config.

One ``.npz`` is written per (simulation, projection).  ``plot_r_profiles.py``
turns those into the Singh et al. (2020) Fig. 1 analogue and the Gate A
cross-simulation scatter metrics.

Fields, following ``docs/r_profiles_task1_spec.md``:

    g  SHAM-selected subhalos, NGP-deposited      (galaxies)
    e  'ionized_gas'                              (free electrons)
    b  'baryon'  = gas + stars + BH               (all baryons)
    m  'total' - 'baryon'                         (CDM)

The CDM map is derived by subtraction rather than by a separate DM particle
sweep: ``mapMaker.make_combined_field`` builds 'total' as gas+DM+stars+BH and
'baryon' as gas+stars+BH on identical grids, so the difference is the DM map
up to float64 round-off.  The derived CDM mass fraction is checked against the
box cosmology and reported for every run.

Usage
-----
    cd scripts/
    python cross_corr/make_r_profiles.py -p configs/cross_corr/r_profiles_z05.yaml
    python cross_corr/make_r_profiles.py -p configs/cross_corr/r_profiles_z05.yaml --sim TNG300-1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.append('../src/')
import rprofiles as rp
from stacker import SimulationStacker
from utils import comoving_to_arcmin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sim_label(sim_type, name, feedback):
    """Build a filename- and legend-safe label for a simulation run.

    Args:
        sim_type (str): 'IllustrisTNG', 'SIMBA' or 'FLAMINGO'.
        name (str): Simulation name.
        feedback (str or None): Feedback variant, if any.

    Returns:
        str: Label such as 'TNG300-1' or 'L1_m9_fgas-8sigma'.
    """
    if feedback is None:
        return str(name)
    if sim_type == 'FLAMINGO' and feedback == name:
        return f'{name}_fiducial'
    return f'{name}_{feedback}'


def load_component_fields(stacker, n_pixels, projection, cfg, verbose=True):
    """Load the cached projected fields and derive the CDM map.

    Args:
        stacker (SimulationStacker): Configured stacker.
        n_pixels (int): Pixels per side of the cached grid.
        projection (str): 'xy', 'xz' or 'yz'.
        cfg (dict): The ``stack`` config block.
        verbose (bool, optional): Print progress. Defaults to True.

    Returns:
        dict: Overdensity maps keyed 'e', 'b', 'm'.

    Raises:
        ValueError: If a required cached field is missing (raised by
            ``loadData``); the message names the expected file.
    """
    load = cfg.get('load_field', True)
    save = cfg.get('save_field', False)
    ptype_e = cfg.get('particle_type_e', 'ionized_gas')
    ptype_b = cfg.get('particle_type_b', 'baryon')

    def _field(ptype):
        if verbose:
            print(f'    loading {ptype} ...', flush=True)
        return stacker.makeField(ptype, nPixels=n_pixels,
                                 projection=projection, save=save, load=load)

    deltas = {}

    baryon = _field(ptype_b)
    total = _field('total')
    cdm = rp.derive_cdm_field(total, baryon, header=stacker.header,
                              verbose=verbose)
    del total
    deltas['m'] = rp.to_overdensity(cdm)
    del cdm
    deltas['b'] = rp.to_overdensity(baryon)
    del baryon

    electrons = _field(ptype_e)
    deltas['e'] = rp.to_overdensity(electrons)
    del electrons

    return deltas


def flatten_for_npz(prof, Ymat, meta):
    """Flatten the nested result dicts into a flat ``np.savez`` payload.

    Args:
        prof (dict): Output of :func:`rprofiles.r_profiles`.
        Ymat (dict): Output of :func:`rprofiles.compute_Y_matrix`.
        meta (dict): Scalar/str metadata to record alongside the arrays.

    Returns:
        dict: Flat mapping of names to arrays, safe for ``np.savez``.
    """
    out = {'radii': prof['radii']}

    for (a, b), per_filter in prof['r'].items():
        for filt, values in per_filter.items():
            out[f'r_{a}{b}_{filt}'] = values
            out[f'rerr_{a}{b}_{filt}'] = prof['r_err'][(a, b)][filt]
            out[f'rjk_{a}{b}_{filt}'] = prof['r_jk'][(a, b)][filt]

    for (num, den), per_filter in prof['ratio'].items():
        tag = f'{num[0]}{num[1]}_over_{den[0]}{den[1]}'
        for filt, values in per_filter.items():
            out[f'ratio_{tag}_{filt}'] = values
            out[f'ratioerr_{tag}_{filt}'] = prof['ratio_err'][(num, den)][filt]
            out[f'ratiojk_{tag}_{filt}'] = prof['ratio_jk'][(num, den)][filt]

    for filt, pairs in Ymat['Y'].items():
        for (a, b), values in pairs.items():
            out[f'Y_{a}{b}_{filt}'] = values

    for key, value in meta.items():
        out[f'meta_{key}'] = np.array(value)

    return out


def process_simulation(sim_type, sim_entry, cfg, out_dir, verbose=True):
    """Compute and save the r-profiles for one simulation, all projections.

    Args:
        sim_type (str): Simulation suite.
        sim_entry (dict): One entry of the config ``sims`` list.  Requires
            ``name``, ``snapshot`` and ``n_pixels``; ``feedback`` and
            ``redshift`` are optional.
        cfg (dict): The ``stack`` config block.
        out_dir (pathlib.Path): Directory for the output ``.npz`` files.
        verbose (bool, optional): Print progress. Defaults to True.

    Returns:
        list: Paths of the written ``.npz`` files.
    """
    name = sim_entry['name']
    snapshot = sim_entry['snapshot']
    feedback = sim_entry.get('feedback')
    n_pixels = sim_entry['n_pixels']
    label = sim_label(sim_type, name, feedback)

    z_cfg = float(sim_entry.get('redshift', cfg.get('redshift', 0.5)))
    stacker = SimulationStacker(name, snapshot, nPixels=n_pixels,
                                simType=sim_type, feedback=feedback, z=z_cfg)

    # The snapshot header is authoritative for the angular scale; the config
    # value is only a cross-check (SimulationStacker already warns loudly on a
    # mismatch, which would bias every aperture in this task).
    z_true = float(np.asarray(stacker.header['Redshift']).ravel()[0])
    lbox = float(stacker.header['BoxSize'])
    theta_arcmin = comoving_to_arcmin(lbox, z_true, cosmo=stacker.cosmo)
    pixel_arcmin = theta_arcmin / n_pixels

    radii = np.linspace(float(cfg.get('min_radius', 1.0)),
                        float(cfg.get('max_radius', 6.0)),
                        int(cfg.get('num_radii', 9)))
    dr = float(cfg.get('dr_arcmin', rp.DR_ARCMIN))
    r0 = float(cfg.get('r0_arcmin', rp.R0_ARCMIN))
    n_jk_side = int(cfg.get('n_jk_side', rp.N_JK_SIDE))
    target = float(cfg.get('halo_abundance_target', 5e-4))
    parent_upper = cfg.get('parent_mass_upper', 5e14)
    parent_upper = None if parent_upper is None else float(parent_upper)

    if verbose:
        print(f'\n{"=" * 70}')
        print(f'{label}  ({sim_type}, snapshot {snapshot})')
        print(f'{"=" * 70}')
        print(f'  z (header) = {z_true:.4f}   [config {z_cfg}]')
        print(f'  box = {lbox / 1000:.1f} cMpc/h = {theta_arcmin:.1f} arcmin')
        print(f'  grid = {n_pixels}^2, pixel = {pixel_arcmin:.5f} arcmin '
              f'({lbox / n_pixels:.1f} ckpc/h)')
        print(f'  smallest aperture {radii.min():.2f} arcmin spans '
              f'{radii.min() / pixel_arcmin:.1f} pixels')

    # Report apertures whose disk or annulus edge lands exactly on a lattice
    # shell.  Membership of that shell flips under an arbitrarily small change
    # of convention, which is a real (if small) discretization systematic on
    # the filtered amplitudes.  It largely cancels in the coefficients, but the
    # affected radii should be visible in the log rather than discovered later.
    degenerate = rp.degenerate_apertures(radii, pixel_arcmin, dr)
    if degenerate and verbose:
        print('  boundary-degenerate apertures (lattice shell exactly on an '
              'edge):')
        for R, edges in sorted(degenerate.items()):
            for edge_name, margin, shell in edges:
                print(f"    R={R:.3f}' {edge_name} edge: {shell} pixels sit "
                      f'{margin:.1e} pixels from the boundary')

    # Guard: an under-resolved aperture must raise before the field loads.
    # r0 is checked as well as the smallest aperture, since a config may set
    # r0 below min_radius.
    for guard_radius in (float(radii.min()), r0):
        rp.build_aperture_kernel(n_pixels, pixel_arcmin, guard_radius,
                                 'DSigma', dr)

    subhalos = stacker.loadSubHalos()

    written = []
    for projection in cfg.get('projections', ['yz']):
        t0 = time.time()
        if verbose:
            print(f'\n  --- projection {projection} ---', flush=True)

        deltas = load_component_fields(stacker, n_pixels, projection, cfg,
                                       verbose=verbose)

        galaxies, halo_mask = rp.make_galaxy_field(
            stacker, projection, n_pixels, target,
            parent_mass_upper=parent_upper, subhalos=subhalos)
        n_gal = int(halo_mask.size)
        nbar_pix = galaxies.sum() / float(n_pixels * n_pixels)
        deltas['g'] = rp.to_overdensity(galaxies)
        del galaxies
        if verbose:
            print(f'    {n_gal} SHAM galaxies '
                  f'(nbar = {nbar_pix:.4e} per pixel, '
                  f'{n_gal / (lbox / 1000) ** 3:.3e} (cMpc/h)^-3)',
                  flush=True)

        if verbose:
            print('    computing filtered amplitudes ...', flush=True)
        Ymat = rp.compute_Y_matrix(deltas, pixel_arcmin, radii=radii, dr=dr,
                                   r0=r0, nbar_pix=nbar_pix,
                                   n_jk_side=n_jk_side)
        prof = rp.r_profiles(Ymat)
        del deltas

        meta = {
            'label': label,
            'sim_type': sim_type,
            'sim_name': str(name),
            'feedback': str(feedback),
            'snapshot': snapshot,
            'projection': projection,
            'redshift': z_true,
            'n_pixels': n_pixels,
            'pixel_arcmin': pixel_arcmin,
            'boxsize_ckpc_h': lbox,
            'n_galaxies': n_gal,
            'nbar_pix': nbar_pix,
            'abundance_target': target,
            'parent_mass_upper': np.nan if parent_upper is None else parent_upper,
            'dr_arcmin': dr,
            'r0_arcmin': r0,
            'n_jk': Ymat['n_jk'],
            # Per-aperture flag: True where a lattice shell sits on the disk or
            # annulus edge, so the amplitude carries a convention-dependent
            # discretization shift (small in the coefficients, see
            # rprofiles.lattice_boundary_margin).
            'boundary_degenerate': np.array(
                [float(R) in degenerate for R in radii], dtype=bool),
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'r_profiles_{label}_{snapshot}_{projection}.npz'
        np.savez(out_path, **flatten_for_npz(prof, Ymat, meta))
        written.append(out_path)

        if verbose:
            print(f'    saved {out_path}  ({time.time() - t0:.1f} s)')
            for filt in rp.FILTERS:
                r_gb = prof['r'][('g', 'b')][filt]
                r_bm = prof['r'][('b', 'm')][filt]
                ratio = prof['ratio'][(('b', 'm'), ('g', 'b'))][filt]
                with np.errstate(invalid='ignore'):
                    print(f'      {filt:8s} r_gb[{np.nanmin(r_gb):.3f},'
                          f'{np.nanmax(r_gb):.3f}]  '
                          f'r_bm[{np.nanmin(r_bm):.3f},{np.nanmax(r_bm):.3f}]  '
                          f'ratio[{np.nanmin(ratio):.3f},'
                          f'{np.nanmax(ratio):.3f}]')

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(path2config, sim=None, feedback=None, verbose=True):
    """Run the r-profile computation for every simulation in a config.

    Args:
        path2config (str): Path to the YAML configuration file.
        sim (str, optional): If given, only process simulations with this
            name.  Defaults to None (all).
        feedback (str, optional): If given, only process this feedback
            variant.  Defaults to None (all).
        verbose (bool, optional): Print progress. Defaults to True.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)

    cfg = config.get('stack', {})
    plot_cfg = config.get('plot', {})
    out_dir = Path(plot_cfg.get('npz_path', '../data/r_profiles/'))

    t_start = time.time()
    written = []
    for suite in config['simulations']:
        sim_type = suite['sim_type']
        for entry in suite['sims']:
            if sim is not None and entry['name'] != sim:
                continue
            if feedback is not None and entry.get('feedback') != feedback:
                continue
            written.extend(
                process_simulation(sim_type, entry, cfg, out_dir,
                                   verbose=verbose))

    print(f'\n{"=" * 70}')
    print(f'Wrote {len(written)} file(s) in '
          f'{(time.time() - t_start) / 60:.1f} minutes:')
    for path in written:
        print(f'  {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute Task 1 r-profiles from cached projected fields.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/r_profiles_z05.yaml',
                        help='Path to the YAML configuration file.')
    parser.add_argument('--sim', type=str, default=None,
                        help='Only process this simulation name.')
    parser.add_argument('--feedback', type=str, default=None,
                        help='Only process this feedback variant.')
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
