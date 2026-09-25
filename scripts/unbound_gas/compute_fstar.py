"""compute_fstar.py
================
Stellar-fraction floor for the unbound gas paper's P(k) section: the high-f*
end of the S(k) and lensing bands of make_pk_fstar.py.

For each method variant (halo-finder membership 'fof', apertures 'ap<x>' of
x R200m) and halo mass cut of the config's ``stellar`` block, every selected
halo region whose star fraction

    f*_h = M*_h / (M*_h + M_wind,h + M_gas,h + M_BH,h)

(true stars over all baryons in the region; TNG/Illustris wind particles
count as gas) is below the method's target f*_T (``fstar.targets``) gains
dM_h = f*_T M_b,h - M*_h of stars, laid out like its stars and taken from its
ionized gas in proportion to the ionized mass; regions at or above the target
are unchanged (a floor). Regions with too little ionized gas convert all of
it (capped); regions without stars or ionized gas stay unchanged
(``halo_transfer.floor_coefficients``). Baryons are conserved region by region.

With H the change (stars added minus ionized gas removed; TSC, 3D grid of the
config) and S, G the 2D maps of the stars added and the ionized gas removed
(the lensing grid, as compute_stellar_maps.py):

    P_mm(f*_T) = P_mm + 2 P_mH + P_HH                           (exact)
    f(theta)   = [N - dS_G] / [T + dS_S - dS_G] * Omega_m/Omega_b   (stack_fstar_maps.py)

The f* = 0 end is the existing s = 0 transfer (compute_pk_stellar.py,
compute_stellar_maps.py, stack_stellar_maps.py), which this does not redo.

One pass over the stars, gas, winds and BH per simulation (at the lowest cut,
labelled for every method), then for each configuration the 3D field and its
spectra and the two 2D maps. The spectra estimator is compute_pk_local.py's
(identical to Pylians), checked against the Pylians P_total of the components
or DMO spectra file; the first configuration also gets the explicit-field
check (spectrum of rho_m + H against the formula).

Outputs (new files; a variant is skipped when its spectra file and all its
maps exist, unless --overwrite):
  products/3D/<stem>_Pk_fstar_<variant>_<n>.npz
      k, Nmodes, P_mm, target, per tag P_mH__<tag>, P_HH__<tag> and the
      floor bookkeeping diag__<tag>__* (star fractions before and after,
      regions raised / capped / without stars or ionized gas, conservation)
  products/2D/<stem>_fstar<T>_stars_<variant>_<tag>_<n2>_yz.npy   stars added
  products/2D/<stem>_fstar<T>_gas_<variant>_<tag>_<n2>_yz.npy     ionized gas removed
  products/2D/<stem>_fstar_maps_<n2>_yz.npz                      2D bookkeeping

Run from the scripts/ directory on a whole CPU node (runINT_fstar.sh):
    python unbound_gas/compute_fstar.py -p configs/unbound_gas/pk_fstar_z05.yaml --sims m100n1024
    # smoke test: first 2 chunks only, nothing saved
    python unbound_gas/compute_fstar.py -p ... --sims TNG300-1 --max-chunks 2
"""

import argparse
import gc
import time
from pathlib import Path

import numba
import numpy as np
import scipy.fft as sfft

from compute_pk_local import cross_power
from compute_pk_stellar import axpy_slabs, mass_tag, variants_of
from compute_stellar_maps import (field2d_path, lensing_settings, lensing_sim,
                                  map_n_pixels)
from pk_common import (load_config, load_field, save_npy_atomic, save_npz_atomic,
                       select_sims, sim_label, spectra_path)

import halo_transfer as ht


def targets_of(config: dict) -> dict:
    """Target star fraction per method variant name ('fof', 'ap1', ...)."""
    return {str(k): float(v) for k, v in config['fstar']['targets'].items()}


def fstar_map_path(entry: dict, kind: str, target: float, variant: str, tag: str,
                   n: int, projection: str) -> Path:
    """Floor map: kind 'stars' (stars added) or 'gas' (ionized gas removed)."""
    return field2d_path(entry, f"fstar{target:g}_{kind}_{variant}_{tag}", n, projection)


def fstar_maps_diag_path(entry: dict, n: int, projection: str) -> Path:
    """Per-simulation bookkeeping file of the floor maps."""
    return field2d_path(entry, 'fstar_maps', n, projection).with_suffix('.npz')


def run_sim(entry: dict, config: dict, only, overwrite: bool, max_chunks) -> None:
    """Floor fields, spectra and maps of every configuration for one simulation."""
    from stacker import SimulationStacker

    targets = targets_of(config)
    lens = lensing_settings(config)
    lsim, z = lensing_sim(lens, entry)
    if lsim is None:
        raise ValueError(f"{sim_label(entry)} is not in the lensing config {lens['config_path']}")
    proj = lens['stack']['projection']
    st = SimulationStacker(entry['name'], entry['snapshot'], simType=entry['sim_type'],
                           feedback=entry.get('feedback'), z=z)
    box = float(st.header['BoxSize'])
    box_mpc = box / 1000.0
    n3 = int(entry['n_pixels'])
    n2 = map_n_pixels(st, z, lens['stack']['pixel_size'])
    for p_type in ('ionized_gas', 'total'):
        if not field2d_path(entry, p_type, n2, proj).exists():
            raise FileNotFoundError(f"no cached {p_type} map at n = {n2} (z = {z})")
    print(f"  3D grid {n3}^3; lensing grid z = {z}, {n2}^2 ({proj})")

    save = max_chunks is None
    cuts = sorted(float(m) for m in config['stellar']['halo_mass_min'])
    tags = [mass_tag(m) for m in cuts]
    variants = variants_of(config, only)
    missing = [v['name'] for v in variants if v['name'] not in targets]
    if missing:
        raise KeyError(f"no fstar.targets for {missing}")

    def done(v):
        T = targets[v['name']]
        return (spectra_path(entry, f"fstar_{v['name']}").exists()
                and all(fstar_map_path(entry, k, T, v['name'], t, n2, proj).exists()
                        for k in ('stars', 'gas') for t in tags))
    if save and not overwrite:
        variants = [v for v in variants if not done(v)]
    if not variants:
        print("  all outputs exist, skipping")
        return

    t0 = time.time()
    halos = st.loadHalos()
    gmass = np.asarray(halos['GroupMass'], dtype=np.float64)
    print(f"  {len(gmass):,} haloes; {np.sum(gmass >= cuts[0]):,} above {cuts[0]:.0e} Msun/h "
          f"({time.time() - t0:.0f} s)")

    # ---- one pass: stars, gas, winds, BH; then each kept particle's 2D pixel
    t0 = time.time()
    store = ht.collect_particles(st, variants, halos, cuts[0], max_chunks=max_chunks, baryons=True)
    tot = store['totals']
    print(f"  particle pass in {time.time() - t0:.0f} s; true stars {tot['mstar']:.4e}, winds "
          f"{tot['mwind']:.3e}, gas {tot['mgas']:.4e}, BH {tot['mbh']:.3e} Msun/h", flush=True)
    t0 = time.time()
    for p_type in ('Stars', 'gas'):
        for blk in store[p_type]:
            blk['pix'] = ht.pixel_index_2d(blk['pos'], n2, box, proj)
            blk.pop('sf', None)
    gc.collect()
    print(f"  pixel indices in {time.time() - t0:.0f} s", flush=True)

    # ---- P_mm with the numba estimator, checked against Pylians
    comp_path = spectra_path(entry, 'components')
    ref_kind = 'components' if comp_path.exists() else 'dmo'
    ref = np.load(spectra_path(entry, ref_kind))
    F = np.empty((2, n3, n3, n3 // 2 + 1), dtype=np.complex64)
    field = load_field(entry, 'total')
    mean_total = float(np.mean(field, dtype=np.float64))
    F[0] = sfft.rfftn(field, workers=-1)
    del field
    inv = 1.0 / mean_total
    k, nmodes, P = cross_power(F[:1], F[:1], box_mpc, [inv], [inv])
    P_mm = P[0, 0]
    if len(ref['k']) != len(k) or not np.allclose(ref['k'], k, rtol=1e-10):
        raise RuntimeError(f"k bins differ from the Pylians {ref_kind} spectra file")
    dev = float(np.max(np.abs(P_mm / ref['P_total'] - 1.0)))
    print(f"  P_mm: max rel diff vs Pylians P_total ({ref_kind} file) = {dev:.2e}")
    if dev > 1e-4:
        raise RuntimeError(f"estimator disagrees with Pylians (max rel diff {dev:.2e})")

    dpath = fstar_maps_diag_path(entry, n2, proj)
    res2 = {}
    if save and dpath.exists():  # keep the numbers of the variants not rebuilt now
        with np.load(dpath) as old:
            res2 = {kk: old[kk] for kk in old.files}
    res2.update(n_pixels=n2, projection=proj, z=z, box=box, halo_mass_min=np.array(cuts),
                tags=np.array(tags))

    explicit_done = False
    for v in variants:
        T = targets[v['name']]
        s0_path = spectra_path(entry, f"stellar_{v['name']}")
        s0 = np.load(s0_path) if s0_path.exists() else None
        res = dict(k=k, Nmodes=nmodes, P_mm=P_mm, max_rel_diff_vs_pylians=dev,
                   pylians_reference=ref_kind, mean_total=mean_total,
                   mass_total=mean_total * n3 ** 3, box_mpc=box_mpc, n_pixels=n3,
                   variant=v['name'], method=v['method'], x=v['x'], target=T,
                   halo_mass_min=np.array(cuts), tags=np.array(tags),
                   mstar_box=tot['mstar'], mwind_box=tot['mwind'], mgas_box=tot['mgas'],
                   mion_box=tot['mion'], mbh_box=tot['mbh'], n_chunks_read=store['n_chunks'])
        for mcut, tag in zip(cuts, tags):
            t1 = time.time()
            bary = ht.halo_baryons(store, v['name'], gmass, mcut)
            coef, info = ht.floor_coefficients(bary, T)
            # Same regions as the s = 0 run: its moved stellar mass is the stars
            # of the regions with stars and ionized gas.
            if s0 is not None and max_chunks is None:
                act = (bary['mstar'] > 0) & (bary['mion'] > 0)
                m0 = float(s0[f"diag__{tag}__mstar_moved"])
                res[f"check_mstar_vs_s0__{tag}"] = (float(bary['mstar'][act].sum()) - m0) / m0

            H, d3 = ht.scaled_transfer_field(store, v['name'], gmass, mcut, coef, info, bary, T,
                                             n3, box)
            H32 = H.astype(np.float32)
            del H
            gc.collect()
            direct = None
            if not explicit_done:
                field = load_field(entry, 'total')
                axpy_slabs(field, 1.0, H32)
                F[1] = sfft.rfftn(field, workers=-1)
                del field
                _, _, Pd = cross_power(F[1:2], F[1:2], box_mpc, [inv], [inv])
                direct = Pd[0, 0]
                explicit_done = True
            F[1] = sfft.rfftn(H32, workers=-1)
            del H32
            _, _, Px = cross_power(F[1:2], F, box_mpc, [inv], [inv, inv])
            P_mH, P_HH = Px[0, 0], Px[0, 1]
            res[f"P_mH__{tag}"] = P_mH
            res[f"P_HH__{tag}"] = P_HH
            if direct is not None:
                res[f"explicit_check__{tag}"] = float(np.max(np.abs(direct / (P_mm + 2 * P_mH + P_HH) - 1)))
            for key, val in d3.items():
                res[f"diag__{tag}__{key}"] = val

            S2, G2, d2 = ht.scaled_transfer_maps_2d(store, v['name'], gmass, mcut, coef, info,
                                                   bary, T, n2)
            d2['min_stars_added'] = float(S2.min())
            d2['min_gas_removed'] = float(G2.min())
            if save:
                save_npy_atomic(fstar_map_path(entry, 'stars', T, v['name'], tag, n2, proj), S2)
                save_npy_atomic(fstar_map_path(entry, 'gas', T, v['name'], tag, n2, proj), G2)
            del S2, G2
            for key in ('sum_stars_rel', 'sum_gas_rel', 'max_halo_cons_stars', 'max_halo_cons_gas',
                        'mstar_added', 'min_stars_added', 'min_gas_removed'):
                res2[f"diag__{v['name']}__{tag}__{key}"] = d2[key]
            res2[f"target__{v['name']}"] = T

            q = (2 * P_mH + P_HH) / P_mm
            i5 = int(np.argmin(np.abs(k - 5)))
            print(f"  [{v['name']} {tag}] {time.time() - t1:.0f} s: f* {d3['fstar_sim']:.3f} -> "
                  f"{d3['fstar_after']:.3f} (target {T:g}); {d3['n_raised']:,} raised "
                  f"({d3['n_capped']:,} capped), {d3['n_no_stars']:,} without stars, "
                  f"{d3['n_no_ion']:,} without ionized gas; stars added "
                  f"{d3['mstar_added'] / tot['mstar']:.3f} of the stars; sum H / added "
                  f"{d3['sum_H_rel']:.1e}; per-halo {max(d3['max_halo_cons_stars'], d3['max_halo_cons_gas']):.1e}; "
                  f"2D sums {d2['sum_stars_rel']:.1e}/{d2['sum_gas_rel']:.1e}; vs s=0 regions "
                  f"{res.get(f'check_mstar_vs_s0__{tag}', np.nan):.1e}; "
                  f"explicit {res.get(f'explicit_check__{tag}', np.nan):.1e}; "
                  f"P/P_mm - 1 at k[0] {100 * q[0]:+.4f}%, k~5 {100 * q[i5]:+.2f}%", flush=True)
            gc.collect()
        if save:
            save_npz_atomic(spectra_path(entry, f"fstar_{v['name']}"), **res)
    if save:
        save_npz_atomic(dpath, **res2)
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
    for key in ('fstar', 'lensing'):
        if key not in config:
            raise SystemExit(f"{args.path2config} has no '{key}' block")
    numba.set_num_threads(int(config['pk'].get('threads', numba.get_num_threads())))
    for entry in select_sims(config, args.sims):
        print(f"\n===== {sim_label(entry)} (3D grid {entry['n_pixels']}^3) =====", flush=True)
        run_sim(entry, config, args.variants, args.overwrite, args.max_chunks)
        gc.collect()


if __name__ == '__main__':
    main()
