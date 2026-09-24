"""compute_pk_stellar.py
======================
Halo-level stellar-to-ionized-gas transfer for the matter power spectrum
section of the unbound gas paper: how the baryonic suppression of P(k)
responds when part or all of the stellar mass in haloes is instead ionized
gas laid out like the halo's own ionized gas (``src/halo_transfer.py`` has
the model; ``make_pk_alpha.py`` and ``make_pk_local.py`` are unchanged).

For each simulation of ``configs/unbound_gas/pk_components_z05.yaml`` and
each configuration of its ``stellar`` block -- method (halo-finder
membership 'fof', or apertures of x R200m 'ap<x>') and halo mass cut
M_min (FoF GroupMass) -- this builds the zero-mass transfer field D (all
selected stellar mass moved, s = 0) from the particles and measures

    P_mm (cached total field), P_mD, P_DD ,

from which P_mm(s) = P_mm + 2 (1 - s) P_mD + (1 - s)^2 P_DD for every
stellar scale s (fraction of the selected stellar mass kept as stars;
evaluated by make_pk_stellar.py). Stars are PartType4 without the TNG/
Illustris wind-phase particles; ionized gas is the pipeline's definition
(mapMaker.ionized_gas_masses). Haloes with stars but no ionized gas keep
their stars (reported).

One pass over the snapshot reads the stars and gas once and labels them for
every requested method at the lowest M_min (higher cuts follow by masking:
membership trivially, apertures because of the mass-priority rule). The
fields are then built and measured one configuration at a time.

Spectra use the numba estimator of compute_pk_local.py (identical to
Pylians: same k bins, mode counting, TSC deconvolution); the estimator's
P_mm is checked against the Pylians P_total of the components file, or of the
DMO spectra file where no components file exists (z ~ 0.26). Without a
cached Stars field the negative-stellar-mass check is skipped.

Validation written to every file (see the docs for thresholds): total mass
conservation of D on the grid; per-halo conservation of the particle
weights; the removed stars against the cached Stars field (negative mass);
per-halo stellar mass from the particles against the halo catalogue (fof,
TNG/Illustris/SIMBA); and, for the first configuration of each simulation,
the spectrum of the explicitly built field rho_m + (1 - s) D at s = 0 and
0.5 against the quadratic formula.

Outputs (new files next to the 3D caches; existing ones skipped unless
--overwrite), one per method variant:
  <stem>_Pk_stellar_<variant>_<n>.npz    (<variant> = fof, ap1, ap2, ...)
      k, Nmodes, P_mm, mean_total, box/particle totals, and for every mass
      cut tag M<log10 M_min> (e.g. M11): P_mD__<tag>, P_DD__<tag>, and
      diagnostics diag__<tag>__<name>; per-halo masses at the lowest cut.

Run from the scripts/ directory on a whole CPU node:
    python unbound_gas/compute_pk_stellar.py -p configs/unbound_gas/pk_components_z05.yaml \
        --sims m100n1024
    # smoke test: first 2 chunks only, nothing saved
    python unbound_gas/compute_pk_stellar.py -p ... --sims TNG300-1 --max-chunks 2
"""

import argparse
import gc
import time

import numba
import numpy as np
import scipy.fft as sfft

from compute_pk_local import cross_power
from pk_common import (COMPONENTS, box_size_mpc, field_path, load_config, load_field,
                       save_npz_atomic, select_sims, sim_label, spectra_path)

import halo_transfer as ht


def variants_of(config: dict, only=None) -> list:
    """Method variants from the config's ``stellar`` block.

    Returns:
        list: Dicts with 'name' ('fof', 'ap1', 'ap2', ...), 'method' and, for
        apertures, 'x' (radius in units of GroupRad = R200m).
    """
    sc = config['stellar']
    out = []
    if 'membership' in sc['methods']:
        out.append(dict(name='fof', method='membership', x=0.0))
    if 'aperture' in sc['methods']:
        for x in sc['aperture_radii']:
            out.append(dict(name=f"ap{float(x):g}", method='aperture', x=float(x)))
    if only:
        out = [v for v in out if v['name'] in only]
    return out


def mass_tag(m: float) -> str:
    """Key tag of a halo mass cut, e.g. 1e11 -> 'M11'."""
    return f"M{np.log10(float(m)):g}"


def axpy_slabs(y: np.ndarray, a: float, x: np.ndarray) -> None:
    """y += a * x in place, slab by slab (no full-size temporary)."""
    a = np.float32(a) if y.dtype == np.float32 else a
    for i in range(y.shape[0]):
        y[i] += a * x[i]


def run_sim(entry: dict, config: dict, only, overwrite: bool, max_chunks) -> None:
    """Transfer fields and spectra of every configuration for one simulation."""
    from stacker import SimulationStacker

    sc = config['stellar']
    save = max_chunks is None
    variants = variants_of(config, only)
    if save and not overwrite:
        variants = [v for v in variants
                    if not spectra_path(entry, f"stellar_{v['name']}").exists()]
    if not variants:
        print("  all outputs exist, skipping")
        return
    cuts = sorted(float(m) for m in sc['halo_mass_min'])
    n = entry['n_pixels']
    box_mpc = box_size_mpc(entry)

    st = SimulationStacker(entry['name'], entry['snapshot'], simType=entry['sim_type'],
                           feedback=entry.get('feedback'), z=entry.get('redshift', 0.5))
    box = float(st.header['BoxSize'])
    t0 = time.time()
    halos = st.loadHalos()
    gmass = np.asarray(halos['GroupMass'], dtype=np.float64)
    print(f"  {len(gmass):,} haloes; {np.sum(gmass >= cuts[0]):,} above {cuts[0]:.0e} Msun/h "
          f"({time.time() - t0:.0f} s)")

    # ---- one particle pass for every variant
    t0 = time.time()
    store = ht.collect_particles(st, variants, halos, cuts[0], max_chunks=max_chunks)
    tot = store['totals']
    print(f"  particle pass in {time.time() - t0:.0f} s; true stars {tot['mstar']:.4e}, "
          f"winds {tot['mwind']:.3e}, ionized gas {tot['mion']:.4e} Msun/h")

    # ---- original spectrum with the numba estimator, checked against Pylians.
    # The Pylians reference is P_total of the components file, or of the DMO
    # spectra file where no components file exists (e.g. at z ~ 0.26).
    comp_path = spectra_path(entry, 'components')
    ref_kind = 'components' if comp_path.exists() else 'dmo'
    comp = np.load(spectra_path(entry, ref_kind))
    t0 = time.time()
    F = np.empty((2, n, n, n // 2 + 1), dtype=np.complex64)
    field = load_field(entry, 'total')
    mean_total = float(np.mean(field, dtype=np.float64))
    F[0] = sfft.rfftn(field, workers=-1)
    del field
    inv = 1.0 / mean_total
    k, nmodes, P = cross_power(F[:1], F[:1], box_mpc, [inv], [inv])
    P_mm = P[0, 0]
    if len(comp['k']) != len(k) or not np.allclose(comp['k'], k, rtol=1e-10):
        raise RuntimeError(f"k bins differ from the Pylians {ref_kind} spectra file")
    dev = float(np.max(np.abs(P_mm / comp['P_total'] - 1.0)))
    print(f"  P_mm in {time.time() - t0:.0f} s; max rel diff vs Pylians P_total "
          f"({ref_kind} file) = {dev:.2e}")
    if dev > 1e-4:
        raise RuntimeError(f"estimator disagrees with Pylians (max rel diff {dev:.2e})")
    # Cached Stars mass (includes TNG/Illustris winds): from the components
    # file, else from the Stars cache when it exists (read below), else NaN.
    has_stars_cache = field_path(entry['sim_type'], entry['name'], entry['snapshot'],
                                 entry.get('feedback'), 'Stars', n).exists()
    mstar_cache = (float(comp['means'][COMPONENTS.index('Stars')]) * n ** 3
                   if ref_kind == 'components' else np.nan)
    if not has_stars_cache:
        print("  no Stars cache: the negative-stellar-mass check is skipped")

    cat_mstar = None
    if any(v['method'] == 'membership' for v in variants):
        from loadIO import load_halo_stellar_masses
        cat_mstar = load_halo_stellar_masses(st.simPath, st.snapshot, st.simType,
                                             sim_name=st.sim, header=st.header)

    explicit_done = False
    for v in variants:
        res = dict(k=k, Nmodes=nmodes, P_mm=P_mm, max_rel_diff_vs_pylians=dev,
                   pylians_reference=ref_kind,
                   mean_total=mean_total, mass_total=mean_total * n ** 3,
                   box_mpc=box_mpc, n_pixels=n, variant=v['name'], method=v['method'],
                   x=v['x'], halo_mass_min=np.array(cuts),
                   tags=np.array([mass_tag(m) for m in cuts]),
                   mstar_box=tot['mstar'], mwind_box=tot['mwind'], mgas_box=tot['mgas'],
                   mion_box=tot['mion'], m2_stars_box=tot['m2_stars'], m2_gas_box=tot['m2_gas'],
                   mstar_cache=mstar_cache, n_chunks_read=store['n_chunks'])
        for i, mcut in enumerate(cuts):
            tag = mass_tag(mcut)
            t1 = time.time()
            first = i == 0
            stars = load_field(entry, 'Stars') if first and has_stars_cache else None
            if stars is not None and not np.isfinite(mstar_cache):
                mstar_cache = float(np.sum(stars, dtype=np.float64))
                res['mstar_cache'] = mstar_cache
            D, diag = ht.transfer_field(store, v['name'], gmass, mcut, n, box,
                                        stars_field=stars, keep_halo_table=first)
            del stars
            D32 = D.astype(np.float32)
            del D
            gc.collect()

            # Membership check against the catalogue (fof, lowest cut).
            mstar_h = diag.pop('mstar_h_all', None)
            if first and v['method'] == 'membership' and cat_mstar is not None:
                sel = (gmass >= mcut) & (cat_mstar > 0)
                rel = np.abs(mstar_h[sel] / cat_mstar[sel] - 1.0)
                res['catalogue_check_median'] = float(np.median(rel))
                res['catalogue_check_p99'] = float(np.percentile(rel, 99))
                res['catalogue_check_massweighted'] = float(
                    np.sum(np.abs(mstar_h[sel] - cat_mstar[sel])) / np.sum(cat_mstar[sel]))
                print(f"    catalogue check (per-halo M*): median {res['catalogue_check_median']:.1e}, "
                      f"99% {res['catalogue_check_p99']:.1e}, mass-weighted "
                      f"{res['catalogue_check_massweighted']:.1e}")
            table = diag.pop('halo_table', None)
            if table is not None:
                res['halo_rows'] = table['rows']
                res['halo_mstar'] = table['mstar'].astype(np.float32)
                res['halo_mion'] = table['mion'].astype(np.float32)
                res['halo_gmass'] = gmass[table['rows']].astype(np.float32)

            # Explicit field check (first configuration of the simulation):
            # build rho_m + (1 - s) D and measure its spectrum directly.
            direct = {}
            if not explicit_done:
                field = load_field(entry, 'total')
                coef = 0.0  # current multiple of D in `field`
                for s in (0.0, 0.5):
                    axpy_slabs(field, (1.0 - s) - coef, D32)
                    coef = 1.0 - s
                    F[1] = sfft.rfftn(field, workers=-1)
                    _, _, Pd = cross_power(F[1:2], F[1:2], box_mpc, [inv], [inv])
                    direct[s] = Pd[0, 0]
                del field
                explicit_done = True

            F[1] = sfft.rfftn(D32, workers=-1)
            del D32
            _, _, Px = cross_power(F[1:2], F, box_mpc, [inv], [inv, inv])
            P_mD, P_DD = Px[0, 0], Px[0, 1]
            res[f"P_mD__{tag}"] = P_mD
            res[f"P_DD__{tag}"] = P_DD
            for s, Pd in direct.items():
                d = float(np.max(np.abs(Pd / ht.p_of_scale(P_mm, P_mD, P_DD, s) - 1.0)))
                res[f"explicit_check_s{s:g}__{tag}"] = d
                print(f"    explicit field check s={s:g}: max |P_direct/P_formula - 1| = {d:.2e}")
            for key, val in diag.items():
                res[f"diag__{tag}__{key}"] = val
            q = ht.p_of_scale(P_mm, P_mD, P_DD, 0.0) / P_mm - 1.0
            print(f"  [{v['name']} {tag}] {time.time() - t1:.0f} s: {diag['n_haloes_active']:,} haloes, "
                  f"moved {diag['mstar_moved'] / tot['mstar']:.3f} of the stars "
                  f"(kept, no ionized gas: {diag['mstar_kept_noion'] / tot['mstar']:.1e}); "
                  f"sum D / moved = {diag['sum_D_rel']:.1e}; max per-halo cons. {diag['max_halo_cons']:.1e}; "
                  f"neg. stars {diag.get('neg_star_mass', 0.0) / max(diag['mstar_moved'], 1e-30):.1e}; "
                  f"Q(s=0) at k[0] {100 * q[0]:+.4f}%, k~5 {100 * q[np.argmin(np.abs(k - 5))]:+.2f}%",
                  flush=True)
            gc.collect()
        if save:
            save_npz_atomic(spectra_path(entry, f"stellar_{v['name']}"), **res)
    del F, store
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None,
                        help="restrict to these entries (label, name or feedback)")
    parser.add_argument('--variants', nargs='*', default=None,
                        help="restrict to these method variants (fof, ap1, ap2, ...)")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max-chunks', type=int, default=None,
                        help="smoke test: read only the first N snapshot chunks; nothing is saved")
    args = parser.parse_args()

    config = load_config(args.path2config)
    numba.set_num_threads(int(config['pk'].get('threads', numba.get_num_threads())))
    for entry in select_sims(config, args.sims):
        print(f"\n===== {sim_label(entry)} (grid {entry['n_pixels']}^3) =====", flush=True)
        run_sim(entry, config, args.variants, args.overwrite, args.max_chunks)
        gc.collect()


if __name__ == '__main__':
    main()
