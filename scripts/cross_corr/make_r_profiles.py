"""make_r_profiles.py
====================
Compute the Task 1 cross-correlation coefficients r_gb, r_bm, r_gm, r_ge, r_em
and the ratio r_bm/r_gb for the filter set

    {Sigma, DSigma, Upsilon(R0 = r0_arcmin), Y(Rmax = ytransform_rmax)}

over nine linear aperture bins in 1'-6' plus a diagnostic extension, for every
simulation listed in a YAML config.

One ``.npz`` is written per (simulation, projection).  ``plot_r_profiles.py``
turns those into the Singh et al. (2020) Fig. 1 analogue and the Gate A
cross-simulation scatter metrics.

Fields, following ``docs/r_profiles_task1_spec.md``:

    g  SHAM-selected subhalos, NGP-deposited      (galaxies)
    e  'ionized_gas'                              (free electrons)
    b  'baryon'  = gas + stars + BH               (all baryons)
    m  'total' - 'baryon'                         (CDM)

Two additions to the original Task 1 deliverable, both of which are
post-processing of what ``rprofiles.compute_Y_matrix`` already returns:

- **r_gm**, the galaxy-matter coefficient.  ``compute_Y_matrix`` measures every
  unordered field pair, so ``Y_gm`` was always present; only the coefficient
  was not being formed.  It is the third leg of the addendum's calibration
  factor ``C = r_bm r_gm / r_gb`` (``cross_correlation_notes_v0.2_addendum.md``
  Eq. A12), and the one that ``r_bm/r_gb`` alone cannot show.
- **The Park et al. (2021) Y transform**, ``Y(R; Rmax) = Sigma(R) -
  Sigma(Rmax)``, assembled by ``rprofiles.assemble_ytransform`` as an exact
  linear combination of Sigma amplitudes -- the direct map-level filter of the
  addendum's Appendix A, not the quadrature reconstruction of its Section 2.
  Built from Sigma but compensated, so the ``k -> 0`` response that
  disqualified Sigma at Gate B cancels in the difference.

Both R0 and Rmax must be aperture-grid points, because the constructions read
the amplitude there rather than interpolating it.  Any that are missing are
unioned into the grid and the addition is logged: apertures are convolved
independently, so appending one never moves the nine data-matched bins, but it
must never happen silently either.

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
import warnings
from pathlib import Path

import numpy as np
import yaml

sys.path.append('../src/')
import rprofiles as rp
from stacker import SimulationStacker
from utils import comoving_to_arcmin


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

#: Field pairs whose coefficients are reported.  ``('g', 'm')`` is the addendum
#: addition: with r_gb and r_bm it completes the calibration factor
#: ``C = r_bm r_gm / r_gb`` (v0.2 addendum Eq. A12).
PAIRS = (('g', 'b'), ('b', 'm'), ('g', 'm'), ('g', 'e'), ('e', 'm'))

#: Coefficient ratios reported, ``(numerator, denominator)``.  This is the
#: Route A transfer of the v0.1 note, Eq. (4), and the Gate A statistic.
RATIOS = ((('b', 'm'), ('g', 'b')), (('e', 'm'), ('g', 'e')))

#: Key under which the Park et al. Y transform is stored.  Deliberately not
#: a bare 'Y': the theory documents reserve that glyph for the amplitudes
#: ``Y_ab``, and the two must stay distinguishable in the saved payload.
YTRANSFORM_KEY = 'Ytransform'


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


def aperture_radii(cfg):
    """Build the aperture grid: the data-matched bins plus any extension.

    The bins over ``[min_radius, max_radius]`` are the observationally
    accessible range and are held fixed, so results there never shift when the
    extension changes.  ``extend_max_radius`` appends further bins above
    ``max_radius`` at the same spacing, as a diagnostic of the large-aperture
    behaviour of the coefficients (simulations carry no beam, so nothing stops
    us going above the data range; going *below* 1 arcmin is not useful
    because the data are hard-limited by resolution there).

    Args:
        cfg (dict): The ``stack`` config block.  Uses ``min_radius``,
            ``max_radius``, ``num_radii`` and the optional
            ``extend_max_radius``.

    Returns:
        np.ndarray: Aperture radii in arcmin, strictly increasing.
    """
    base = np.linspace(float(cfg.get('min_radius', 1.0)),
                       float(cfg.get('max_radius', 6.0)),
                       int(cfg.get('num_radii', 9)))
    extend_to = cfg.get('extend_max_radius')
    if extend_to is None:
        return base
    if float(extend_to) <= base[-1]:
        warnings.warn(
            f"extend_max_radius={extend_to} is not above max_radius="
            f"{base[-1]}; no extension bins will be added.", stacklevel=2)
        return base
    step = base[1] - base[0]
    # Half-step tolerance so the endpoint is included when it lands on a bin.
    extension = np.arange(base[-1] + step, float(extend_to) + 0.5 * step, step)
    return np.concatenate([base, extension])


def union_reference_radii(radii, references, atol=1e-9, verbose=True):
    """Ensure every filter reference radius is an aperture-grid point.

    ``Upsilon`` reads ``DSigma`` at ``R0`` and the Y transform reads ``Sigma``
    at ``Rmax``; neither interpolates, so a reference radius that is not on the
    grid is an error rather than something to approximate.  Apertures are
    convolved independently, so appending one leaves every existing bin
    bit-identical -- but it does change the length of every saved array, and it
    puts an off-cadence point in the plotted grid, so the addition is reported
    rather than made silently.

    Args:
        radii (np.ndarray): Aperture grid in arcmin, strictly increasing.
        references (sequence): Reference radii in arcmin, in any order, with
            None entries ignored.
        atol (float, optional): Absolute tolerance, arcmin, for deciding that
            a reference is already on the grid.  Defaults to 1e-9.
        verbose (bool, optional): Report any addition.  Defaults to True.

    Returns:
        np.ndarray: The grid, with any missing reference radii inserted in
        sorted order.  Returned unchanged (and identical object contents) when
        every reference is already present.
    """
    radii = np.asarray(radii, dtype=np.float64)
    missing = []
    for ref in references:
        if ref is None:
            continue
        ref = float(ref)
        if not np.any(np.abs(radii - ref) <= atol):
            missing.append(ref)

    if not missing:
        if verbose:
            print('  every filter reference radius is already an aperture; '
                  'the grid is unchanged')
        return radii

    out = np.unique(np.concatenate([radii, np.asarray(missing, dtype=float)]))
    if verbose:
        print(f'  reference radii not on the grid, appended: '
              f'{sorted(set(missing))} -> grid grows '
              f'{len(radii)} to {len(out)} apertures')
    return out


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

    dr = float(cfg.get('dr_arcmin', rp.DR_ARCMIN))
    r0 = float(cfg.get('r0_arcmin', rp.R0_ARCMIN))
    rmax = float(cfg.get('ytransform_rmax', 6.0))
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
        print(f"  Upsilon R0 = {r0:g}', Y transform Rmax = {rmax:g}'")

    radii = union_reference_radii(aperture_radii(cfg), (r0, rmax),
                                  verbose=verbose)
    upsilon_mask = rp.upsilon_defined_mask(radii, r0)
    ytransform_mask = rp.ytransform_defined_mask(radii, rmax)

    if verbose:
        print(f'  {len(radii)} apertures {radii.min():.3f}-{radii.max():.3f} '
              f'arcmin; smallest spans {radii.min() / pixel_arcmin:.1f} pixels')
        # Both derived filters null a band of the grid by construction, and
        # which bins survive is the first thing to check when a curve looks
        # short.  Report it here rather than leaving it to the plotting step.
        in_data = radii <= float(cfg.get('max_radius', 6.0)) + 1e-9
        print(f"  usable bins, data range: Upsilon "
              f"{int((upsilon_mask & in_data).sum())}/{int(in_data.sum())}, "
              f"Y transform "
              f"{int((ytransform_mask & in_data).sum())}/{int(in_data.sum())}")

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
    # Every aperture is checked, not merely the smallest: the reference radii
    # are now part of the grid, and an unresolved one would poison every
    # derived-filter bin rather than a single point.
    for guard_radius in radii:
        rp.build_aperture_kernel(n_pixels, pixel_arcmin, float(guard_radius),
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
        del deltas

        # The Park et al. Y transform costs no convolution: it is an exact
        # linear combination of the Sigma amplitudes just measured, formed per
        # jackknife realization.  Injecting it here rather than inside
        # compute_Y_matrix keeps rprofiles.FILTERS -- and therefore every other
        # consumer of the library -- unchanged.
        (Ymat['Y'][YTRANSFORM_KEY],
         Ymat['Y_jk'][YTRANSFORM_KEY], _) = rp.assemble_ytransform(Ymat, rmax)

        prof = rp.r_profiles(Ymat, pairs=PAIRS, ratios=RATIOS)

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
            'ytransform_rmax': rmax,
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
        # KNOWN GAP: this writes straight to the final path rather than to a
        # temporary one followed by an atomic os.replace.  A crash or an
        # interactive-QOS preemption part-way through would therefore leave a
        # truncated file where the previous good cache was -- and these .npz
        # are git-tracked products that double as the input to
        # plot_r_profiles.py, so there is no second copy to fall back on
        # besides git history.  Left as-is deliberately for now; fix by
        # writing to `out_path.with_suffix('.npz.tmp')` and then
        # `os.replace(tmp, out_path)`, which is atomic on POSIX.
        np.savez(out_path, **flatten_for_npz(prof, Ymat, meta))
        written.append(out_path)

        if verbose:
            print(f'    saved {out_path}  ({time.time() - t0:.1f} s)')
            # Report each filter only over the bins it actually informs.  The
            # derived filters are finite outside their mask but are not the
            # quantity they are named after there -- Upsilon flips to r ~ -1
            # below R0 -- so an unmasked min/max would advertise a number that
            # means nothing.
            masks = {'Upsilon': upsilon_mask, YTRANSFORM_KEY: ytransform_mask}
            for filt in Ymat['Y']:
                use = masks.get(filt, np.ones(len(radii), dtype=bool))
                if not use.any():
                    print(f'      {filt:11s} no usable aperture')
                    continue

                def _span(values, use=use):
                    with np.errstate(invalid='ignore'), \
                            warnings.catch_warnings():
                        warnings.filterwarnings('ignore',
                                                message='All-NaN slice')
                        return np.nanmin(values[use]), np.nanmax(values[use])

                lo_gb, hi_gb = _span(prof['r'][('g', 'b')][filt])
                lo_bm, hi_bm = _span(prof['r'][('b', 'm')][filt])
                lo_gm, hi_gm = _span(prof['r'][('g', 'm')][filt])
                lo_ra, hi_ra = _span(
                    prof['ratio'][(('b', 'm'), ('g', 'b'))][filt])
                print(f'      {filt:11s} r_gb[{lo_gb:.3f},{hi_gb:.3f}]  '
                      f'r_bm[{lo_bm:.3f},{hi_bm:.3f}]  '
                      f'r_gm[{lo_gm:.3f},{hi_gm:.3f}]  '
                      f'ratio[{lo_ra:.3f},{hi_ra:.3f}]')

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
