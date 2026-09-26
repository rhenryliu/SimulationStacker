"""compute_fstar_obs.py
=====================
Observation-based stellar bands for the unbound gas paper's P(k) section: the
ends of the S(k) and lensing f_gas(theta) bands of make_pk_fstar_obs.py
(options a and c of NOTES/unbound_gas/handoff_physical_stellar_band.md).

Regions: the 1 R200m apertures (mass priority) of the haloes with FoF mass at
or above the cut, i.e. the s = 0 transfer's 'ap1' regions. Every region's true
stars are scaled by one factor s per simulation and end (a population
rescale), the difference exchanged with the region's ionized gas (new stars
laid out like its stars, gas in proportion to its ionized mass), so that an
aggregate over the regions hits a target (``fstar_obs.targets``):

    fstar        sum M* / sum M_b = T ,  M_b = M* + M_wind + M_gas + M_BH  (option a)
    mstar_m200m  sum M* / sum M200m = T   (catalogue SO/200_mean mass)       (option c)

For s <= 1 each region with stars and ionized gas keeps s M*_h. For s > 1 it
gains min(s - 1, M_ion,h / M*_h) M*_h: a region converts at most all of its
ionized gas (capped), and ``solve_scale`` finds the s that still hits the
target. Regions with stars but no ionized gas keep their stars (as in the
s = 0 transfer); the baryons of every region are conserved.

An end without a capped region is exactly the s = 0 transfer at scale s,

    P_mm(s) = P_mm + 2 (1 - s) P_mD + (1 - s)^2 P_DD ,   f(theta) = f_of_scale(N, T, A, B, s)

from the existing files (source 's0'; also for s > 1, where no region's
ionized gas goes negative). A capped end (source 'pass') gets its own change
field: one pass over the stars, gas, winds and BH, then the 3D field H (TSC)
and its spectra and the 2D maps of the stars added and the ionized gas
removed (halo_transfer's floor functions with the per-region coefficients),
as compute_fstar.py:

    P_mm(end) = P_mm + 2 P_mH + P_HH                                     (exact)
    f(theta)  = [N - dS_G] / [T + dS_S - dS_G] * Omega_m/Omega_b   (stack_fstar_obs_maps.py)

Inputs: the s = 0 file products/3D/<stem>_Pk_stellar_<variant>_<n>.npz (P_mm,
P_mD, P_DD, the per-region M* and M_ion table, bookkeeping), the f* floor
file <stem>_Pk_fstar_<variant>_<n>.npz (the regions' baryon sum) and the halo
catalogue. With a pass, s is re-solved from the pass's own float64 region
sums, which are checked against those files.

Outputs (a simulation is skipped when its file and all its maps exist, unless
--overwrite):
  products/3D/<stem>_Pk_fstar_obs_<variant>_<n>.npz
      region sums and the simulation's aggregates; per end <option>__<end>:
      target, s, source, regions capped, the aggregate after; for 'pass' ends
      P_mH, P_HH and the bookkeeping; the checks
  products/2D/<stem>_fstarobs_<option>_<end>_{stars,gas}_<variant>_<tag>_<n2>_yz.npy
      (capped ends only)

Run from the scripts/ directory; a pass needs a whole CPU node (runINT_fstar_obs.sh):
    python unbound_gas/compute_fstar_obs.py -p configs/unbound_gas/pk_fstar_obs_z05.yaml --sims Illustris-1
    # which ends need a pass (reads the files and the catalogue; nothing saved)
    python unbound_gas/compute_fstar_obs.py -p ... --solve-only
    # smoke test: first 2 chunks only, nothing saved
    python unbound_gas/compute_fstar_obs.py -p ... --sims TNG300-1 --max-chunks 2
"""

import argparse
import gc
import time
from pathlib import Path

import numba
import numpy as np
import scipy.fft as sfft

from compute_pk_local import cross_power
from compute_pk_stellar import axpy_slabs, mass_tag
from compute_stellar_maps import field2d_path, lensing_settings, lensing_sim, map_n_pixels
from pk_common import (load_config, load_field, save_npy_atomic, save_npz_atomic,
                       select_sims, sim_label, spectra_path)

import halo_transfer as ht

OPTIONS = ('fstar', 'mstar_m200m')
ENDS = ('low', 'high')


def obs_settings(config: dict) -> dict:
    """The ``fstar_obs`` block: region variant, mass cut and targets.

    Returns:
        dict: 'variant' (dict as compute_pk_stellar.variants_of: name 'ap<x>',
        method 'aperture', x), 'mass_min', 'tag', 'targets' ({option: {'low',
        'high'}}).
    """
    b = config['fstar_obs']
    x = float(b['aperture_radius'])
    mcut = float(b['halo_mass_min'])  # PyYAML reads 1.0e13 as a string
    targets = {}
    for opt in OPTIONS:
        lo, hi = (float(t) for t in b['targets'][opt])
        if not 0.0 < lo < hi:
            raise ValueError(f"fstar_obs.targets.{opt} must be 0 < low < high, got {lo}, {hi}")
        targets[opt] = {'low': lo, 'high': hi}
    return dict(variant=dict(name=f"ap{x:g}", method='aperture', x=x), mass_min=mcut,
                tag=mass_tag(mcut), targets=targets)


def end_keys() -> list:
    """Keys '<option>__<end>' of the four ends."""
    return [f"{o}__{e}" for o in OPTIONS for e in ENDS]


def solve_scale(mstar, mion, m_fixed: float, target: float):
    """Uniform stellar scale s bringing the regions' stellar mass to a target.

    Region h (with stars and ionized gas) ends with M*_h (1 + c_h), where
    c_h = s - 1 for s <= 1 and c_h = min(s - 1, r_h), r_h = M_ion,h / M*_h,
    for s > 1: a region converts at most all of its ionized gas (capped). The
    total sum_h M*_h (1 + c_h) + m_fixed is piecewise linear and increasing in
    s, so the solution is exact (no iteration).

    Args:
        mstar (np.ndarray): M*_h > 0 of the regions that can change.
        mion (np.ndarray): Their ionized-gas masses, > 0.
        m_fixed (float): Stellar mass that stays unchanged (regions with
            stars but no ionized gas).
        target (float): Total stellar mass wanted (changing + fixed).

    Returns:
        tuple: (s, coef, capped): the scale, c_h per region (float64) and the
        capped regions (c_h = r_h < s - 1).

    Raises:
        ValueError: If a region has no stars or ionized gas, or the target is
            below m_fixed (s < 0) or above sum(M* + M_ion) + m_fixed.
    """
    ms = np.asarray(mstar, dtype=np.float64)
    mi = np.asarray(mion, dtype=np.float64)
    if len(ms) == 0 or np.any(ms <= 0) or np.any(mi <= 0):
        raise ValueError("every changing region needs stars and ionized gas")
    total = ms.sum()
    need = float(target) - float(m_fixed)
    if need < 0:
        raise ValueError(f"target {target:.4e} is below the fixed stellar mass {m_fixed:.4e}")
    if need <= total:
        s = need / total
        return s, np.full(len(ms), s - 1.0), np.zeros(len(ms), dtype=bool)
    add = need - total
    r = mi / ms
    order = np.argsort(r, kind='stable')
    rs, mo = r[order], ms[order]
    mr = mo * rs
    # added mass at x = s - 1 in [r_{j-1}, r_j]: regions i < j capped (m_i r_i),
    # regions i >= j gain x m_i
    before = np.concatenate(([0.0], np.cumsum(mr)[:-1]))
    tail = total - np.concatenate(([0.0], np.cumsum(mo)[:-1]))
    added_at = before + rs * tail
    if add > mr.sum() * (1.0 + 1e-12):
        raise ValueError(f"target {target:.4e} needs more than all the regions' ionized gas "
                         f"(at most {m_fixed + total + mr.sum():.4e})")
    j = int(np.searchsorted(added_at, add, side='left'))
    x = rs[-1] if j >= len(rs) else (add - before[j]) / tail[j]
    coef = np.minimum(x, r)
    return 1.0 + x, coef, r < x


def p_at_scale(P_mm, P_mD, P_DD, s: float):
    """P_mm + 2 (1 - s) P_mD + (1 - s)^2 P_DD for any s >= 0.

    Unlike ``halo_transfer.p_of_scale`` (0 <= s <= 1), s > 1 is allowed: the
    field rho_m + (1 - s) D is then the uniform stellar increase, exact as long
    as no region's ionized gas goes negative ((s - 1) M*_h <= M_ion,h), which
    the caller must have checked (an uncapped end).
    """
    if s < 0:
        raise ValueError(f"stellar scale must be >= 0, got {s}")
    t = 1.0 - s
    return P_mm + 2.0 * t * P_mD + t * t * P_DD


def obs_path(entry: dict, variant: str) -> Path:
    """Per-simulation results file (products/3D)."""
    return spectra_path(entry, f"fstar_obs_{variant}")


def obs_map_path(entry: dict, key: str, kind: str, variant: str, tag: str, n: int,
                 projection: str) -> Path:
    """Map of a capped end: kind 'stars' (stars added) or 'gas' (ionized gas removed)."""
    opt, end = key.split('__')
    return field2d_path(entry, f"fstarobs_{opt}_{end}_{kind}_{variant}_{tag}", n, projection)


def reference_sums(entry: dict, variant: str, tag: str, mass_min: float, gmass) -> dict:
    """Region sums of the s = 0 and f* floor files for one configuration.

    ``gmass`` (the catalogue's FoF masses, float64) selects the table's
    regions above the cut (the stored float32 copy could round across it).

    Returns:
        dict: 'P_mm', 'k', the s = 0 bookkeeping ('mstar_assigned', 'mstar_moved',
        'mstar_kept_noion', 'n_active'), the f* file's 'mstar_regions',
        'mbaryon_regions', 'n_regions', and the per-region table of the
        active regions above the cut ('rows', 'mstar', 'mion'; float32 as stored).
    """
    p0 = spectra_path(entry, f"stellar_{variant}")
    pf = spectra_path(entry, f"fstar_{variant}")
    for p in (p0, pf):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing (the s = 0 and f* floor runs come first)")
    with np.load(p0) as f:
        if float(np.min(f['halo_mass_min'])) > mass_min:
            raise ValueError(f"{p0}: its region table starts above the cut {mass_min:.0e}")
        keep = np.asarray(gmass, dtype=np.float64)[f['halo_rows']] >= mass_min
        out = dict(k=f['k'], P_mm=f['P_mm'],
                   mstar_assigned=float(f[f"diag__{tag}__mstar_assigned"]),
                   mstar_moved=float(f[f"diag__{tag}__mstar_moved"]),
                   mstar_kept_noion=float(f[f"diag__{tag}__mstar_kept_noion"]),
                   n_active=int(f[f"diag__{tag}__n_haloes_active"]),
                   rows=f['halo_rows'][keep], mstar=f['halo_mstar'][keep], mion=f['halo_mion'][keep])
    with np.load(pf) as f:
        out.update(mstar_regions=float(f[f"diag__{tag}__mstar_regions"]),
                   mbaryon_regions=float(f[f"diag__{tag}__mbaryon_regions"]),
                   n_regions=int(f[f"diag__{tag}__n_regions"]))
    if len(out['rows']) != out['n_active']:
        raise RuntimeError(f"{p0}: {len(out['rows'])} table regions above the cut, "
                           f"{out['n_active']} active in the bookkeeping")
    return out


def m200m_sum(halos: dict, mass_min: float, rows=None) -> tuple:
    """Sum of the catalogue M200m over the haloes above the cut (or given rows).

    Raises:
        ValueError: If the catalogue has no M200m, or a selected halo has none.

    Returns:
        tuple: (sum in Msun/h, number of haloes).
    """
    m200 = halos.get('GroupMass_m200m')
    if m200 is None:
        raise ValueError("the halo catalogue has no M200m (GroupMass_m200m)")
    m200 = np.asarray(m200, dtype=np.float64)
    if rows is None:
        rows = np.flatnonzero(np.asarray(halos['GroupMass'], dtype=np.float64) >= mass_min)
    vals = m200[rows]
    bad = ~(np.isfinite(vals) & (vals > 0))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} selected haloes have no M200m")
    return float(vals.sum()), len(rows)


def solve_ends(targets: dict, mstar, mion, m_fixed: float, denoms: dict, m_moved: float) -> dict:
    """s, coefficients and source of every end.

    An uncapped end takes s from the float64 sums (m_moved = sum of the
    changing regions' stars) rather than the per-region table.

    Returns:
        dict: {'<option>__<end>': dict(target, target_mass, s, coef, capped,
        capped_frac (share of the changing stars in capped regions), source)}.
    """
    ms = np.asarray(mstar, dtype=np.float64)
    out = {}
    for opt in OPTIONS:
        for end in ENDS:
            T = targets[opt][end]
            tm = T * denoms[opt]
            s, coef, capped = solve_scale(mstar, mion, m_fixed, tm)
            if not capped.any():
                s = (tm - m_fixed) / m_moved
                coef = np.full(len(coef), s - 1.0)
            out[f"{opt}__{end}"] = dict(target=T, target_mass=tm, s=s, coef=coef, capped=capped,
                                        capped_frac=float(ms[capped].sum() / ms.sum()),
                                        source='pass' if capped.any() else 's0')
    return out


def _print_ends(ends: dict, current: dict) -> None:
    for key, e in ends.items():
        opt = key.split('__')[0]
        print(f"    {key:20s} target {e['target']:g} (sim {current[opt]:.4f}): s = {e['s']:.4f} "
              f"[{e['source']}]; {int(e['capped'].sum()):,} regions capped "
              f"({e['capped_frac']:.3f} of the changing stars)", flush=True)


def solve_from_files(entry: dict, obs: dict, halos: dict) -> dict:
    """Every end's s and source from the s = 0 and f* files and the catalogue (no pass).

    Args:
        entry (dict): Simulation entry.
        obs (dict): From ``obs_settings``.
        halos (dict): ``SimulationStacker.loadHalos()`` output.

    Returns:
        dict: 'ref' (``reference_sums``), 'm200m', 'n_haloes', 'denoms' and
        'current' (per option), 'ends' (``solve_ends``), 'mstar'/'mion' (the
        changing regions' table, float64), 'checks'.
    """
    v, tag, mcut = obs['variant'], obs['tag'], obs['mass_min']
    gmass = np.asarray(halos['GroupMass'], dtype=np.float64)
    ref = reference_sums(entry, v['name'], tag, mcut, gmass)
    m200, n_sel = m200m_sum(halos, mcut)
    denoms = {'fstar': ref['mbaryon_regions'], 'mstar_m200m': m200}
    current = {o: ref['mstar_regions'] / denoms[o] for o in OPTIONS}
    ms_t = ref['mstar'].astype(np.float64)
    mi_t = ref['mion'].astype(np.float64)
    checks = dict(
        table_vs_moved=float(ms_t.sum() / ref['mstar_moved'] - 1.0),
        s0_vs_fstar_regions=float(ref['mstar_assigned'] / ref['mstar_regions'] - 1.0))
    ends = solve_ends(obs['targets'], ms_t, mi_t, ref['mstar_kept_noion'], denoms, ref['mstar_moved'])
    return dict(ref=ref, m200m=m200, n_haloes=n_sel, denoms=denoms, current=current, ends=ends,
                mstar=ms_t, mion=mi_t, checks=checks)


def run_sim(entry: dict, config: dict, solve_only: bool, overwrite: bool, max_chunks) -> None:
    """Ends, and the fields, spectra and maps of the capped ends, for one simulation."""
    from stacker import SimulationStacker

    obs = obs_settings(config)
    v, tag, mcut = obs['variant'], obs['tag'], obs['mass_min']
    name = v['name']
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
    save = max_chunks is None and not solve_only

    t0 = time.time()
    halos = st.loadHalos()
    gmass = np.asarray(halos['GroupMass'], dtype=np.float64)
    sol = solve_from_files(entry, obs, halos)
    ref, m200, n_sel, current, ends, checks = (sol[kk] for kk in ('ref', 'm200m', 'n_haloes', 'current',
                                                                  'ends', 'checks'))
    ms_t, mi_t = sol['mstar'], sol['mion']
    print(f"  {n_sel:,} haloes >= {mcut:.0e} Msun/h; {ref['n_active']:,} regions with stars and "
          f"ionized gas, {ref['n_regions']:,} with baryons ({time.time() - t0:.0f} s)")
    print(f"  sim: f* = {current['fstar']:.4f}, M*/M200m = {current['mstar_m200m']:.4f}; uncapped up "
          f"to s = {1.0 + float(np.min(mi_t / ms_t)):.3f}; table vs moved {checks['table_vs_moved']:+.1e}, "
          f"s = 0 vs f* stars {checks['s0_vs_fstar_regions']:+.1e}")
    _print_ends(ends, current)
    pass_keys = [kk for kk, e in ends.items() if e['source'] == 'pass']
    if solve_only:
        return

    out_path = obs_path(entry, name)
    if save and not overwrite and out_path.exists() and all(
            obs_map_path(entry, kk, kind, name, tag, n2, proj).exists()
            for kk in pass_keys for kind in ('stars', 'gas')):
        print("  all outputs exist, skipping")
        return

    res = dict(k=ref['k'], P_mm=ref['P_mm'], variant=name, x=v['x'], tag=tag, halo_mass_min=mcut,
               z=z, n_pixels=n3, n_pixels_2d=n2, projection=proj, n_haloes=n_sel,
               n_active=ref['n_active'], n_regions=ref['n_regions'],
               mstar_regions=ref['mstar_regions'], mbaryon_regions=ref['mbaryon_regions'],
               m200m_sum=m200, mstar_moved=ref['mstar_moved'], mstar_fixed=ref['mstar_kept_noion'],
               fstar_sim=current['fstar'], mstar_m200m_sim=current['mstar_m200m'],
               s_uncapped_max=1.0 + float(np.min(mi_t / ms_t)), ends=np.array(end_keys()),
               **{f"check_{kk}": val for kk, val in checks.items()})

    if pass_keys:
        # ---- one pass: stars, gas, winds, BH (labels of this variant only)
        t0 = time.time()
        store = ht.collect_particles(st, [v], halos, mcut, max_chunks=max_chunks, baryons=True)
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

        bary = ht.halo_baryons(store, name, gmass, mcut)
        has = bary['mbaryon'] > 0
        act = (bary['mstar'] > 0) & (bary['mion'] > 0)
        rows = np.flatnonzero(act)
        ms_p, mi_p = bary['mstar'][rows], bary['mion'][rows]
        m_fixed = float(bary['mstar'][has & ~act].sum())
        if max_chunks is None:
            # the same regions and budgets as the s = 0 and f* runs
            if not np.array_equal(rows, ref['rows']):
                raise RuntimeError("the pass's regions with stars and ionized gas differ from the s = 0 run")
            p_denoms = {'fstar': float(bary['mbaryon'][has].sum()), 'mstar_m200m': m200}
            res.update(
                check_pass_moved_vs_s0=float(ms_p.sum() / ref['mstar_moved'] - 1.0),
                check_pass_fixed_vs_s0=float(m_fixed - ref['mstar_kept_noion']),
                check_pass_mbaryon_vs_fstar=float(p_denoms['fstar'] / ref['mbaryon_regions'] - 1.0),
                check_pass_table_mstar=float(np.max(np.abs(ms_p / ms_t - 1.0))),
                check_pass_table_mion=float(np.max(np.abs(mi_p / mi_t - 1.0))))
            print("  pass vs s = 0 / f* files: moved {check_pass_moved_vs_s0:+.1e}, fixed "
                  "{check_pass_fixed_vs_s0:+.1e} Msun/h, baryons {check_pass_mbaryon_vs_fstar:+.1e}, "
                  "per-region M* {check_pass_table_mstar:.1e}, M_ion {check_pass_table_mion:.1e} "
                  "(float32 table)".format(**res))
        else:
            # smoke test: the regions are only partly read, so their M*/M200m
            # is set to the files' value (reachable targets, realistic s)
            p_denoms = {'fstar': float(bary['mbaryon'][has].sum()),
                        'mstar_m200m': float(bary['mstar'][has].sum()) / current['mstar_m200m']}
            print(f"  smoke test: {int(has.sum()):,} regions reached; their M*/M200m set to the files' value")
        # s from the pass's float64 sums (the table is float32)
        p_current = {o: float(bary['mstar'][has].sum()) / p_denoms[o] for o in OPTIONS}
        p_ends = solve_ends(obs['targets'], ms_p, mi_p, m_fixed, p_denoms, float(ms_p.sum()))
        _print_ends(p_ends, p_current)
        if max_chunks is None:
            for kk in end_keys():
                # only a region within float32 rounding of the threshold could
                # flip the decision; the pass's float64 one is used
                agree = p_ends[kk]['source'] == ends[kk]['source']
                if not agree:
                    print(f"  WARNING {kk}: capping differs between the pass ({p_ends[kk]['source']}) "
                          f"and the float32 table ({ends[kk]['source']}); using the pass")
                res[f"check_source_agree__{kk}"] = agree
                res[f"check_s_pass_vs_files__{kk}"] = float(p_ends[kk]['s'] / ends[kk]['s'] - 1.0)
            for kk in end_keys():
                ends[kk] = p_ends[kk]  # float64 values for every end
            pass_keys = [kk for kk, e in ends.items() if e['source'] == 'pass']
        else:
            ends = p_ends
            pass_keys = [kk for kk, e in ends.items() if e['source'] == 'pass'] or ['fstar__high']
            if ends['fstar__high']['source'] != 'pass':
                print("  smoke test: no capped end in the partial data; building fstar__high anyway")

        # ---- P_mm with the numba estimator, checked against the s = 0 file's
        F = np.empty((2, n3, n3, n3 // 2 + 1), dtype=np.complex64)
        field = load_field(entry, 'total')
        mean_total = float(np.mean(field, dtype=np.float64))
        F[0] = sfft.rfftn(field, workers=-1)
        del field
        inv = 1.0 / mean_total
        k, nmodes, P = cross_power(F[:1], F[:1], box_mpc, [inv], [inv])
        P_mm = P[0, 0]
        if len(ref['k']) != len(k) or not np.allclose(ref['k'], k, rtol=1e-10):
            raise RuntimeError("k bins differ from the s = 0 file")
        dev = float(np.max(np.abs(P_mm / ref['P_mm'] - 1.0)))
        print(f"  P_mm: max rel diff vs the s = 0 file = {dev:.2e}")
        if dev > 1e-4:
            raise RuntimeError(f"estimator disagrees with the s = 0 file (max rel diff {dev:.2e})")
        # P_mm_pass goes with the pass ends' P_mH, P_HH; P_mm (the s = 0
        # file's) with P_mD, P_DD
        res.update(Nmodes=nmodes, P_mm_pass=P_mm, check_pmm_vs_s0=dev, mean_total=mean_total,
                   n_chunks_read=store['n_chunks'])

        n_halo = len(gmass)
        # the unchanged regions with baryons, for _floor_diag's bookkeeping
        no_stars = has & ~(bary['mstar'] > 0)
        no_ion = has & (bary['mstar'] > 0) & ~(bary['mion'] > 0)
        explicit_done = False
        for kk in pass_keys:
            e = ends[kk]
            t1 = time.time()
            coef = np.zeros(n_halo)
            coef[rows] = e['coef']
            capped = np.zeros(n_halo, dtype=bool)
            capped[rows] = e['capped']
            dm = coef * bary['mstar']
            info = dict(dm=dm, raised=coef > 0, capped=capped, no_stars=no_stars, no_ion=no_ion)
            H, d3 = ht.scaled_transfer_field(store, name, gmass, mcut, coef, info, bary, np.nan,
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
            res[f"P_mH__{kk}"] = P_mH
            res[f"P_HH__{kk}"] = P_HH
            if direct is not None:
                res[f"explicit_check__{kk}"] = float(np.max(np.abs(direct / (P_mm + 2 * P_mH + P_HH) - 1)))
            opt = kk.split('__')[0]
            after = float((bary['mstar'][has] + dm[has]).sum()) / p_denoms[opt]
            res[f"check_target__{kk}"] = after / e['target'] - 1.0
            for key, val in d3.items():
                if key not in ('f_target', 'min_fstar_after_minus_target'):
                    res[f"diag__{kk}__{key}"] = val

            S2, G2, d2 = ht.scaled_transfer_maps_2d(store, name, gmass, mcut, coef, info, bary,
                                                   np.nan, n2)
            res[f"mapdiag__{kk}__sum_stars_rel"] = d2['sum_stars_rel']
            res[f"mapdiag__{kk}__sum_gas_rel"] = d2['sum_gas_rel']
            res[f"mapdiag__{kk}__min_stars_added"] = float(S2.min())
            res[f"mapdiag__{kk}__min_gas_removed"] = float(G2.min())
            if save:
                save_npy_atomic(obs_map_path(entry, kk, 'stars', name, tag, n2, proj), S2)
                save_npy_atomic(obs_map_path(entry, kk, 'gas', name, tag, n2, proj), G2)
            del S2, G2
            q = (2 * P_mH + P_HH) / P_mm
            i5 = int(np.argmin(np.abs(k - 5)))
            print(f"  [{kk}] {time.time() - t1:.0f} s: s = {e['s']:.4f}, aggregate after / target - 1 = "
                  f"{res[f'check_target__{kk}']:+.1e}; stars added {d3['mstar_added']:.4e} Msun/h "
                  f"({d3['n_capped']:,} capped); removed/M_ion max {d3['max_removed_over_mion']:.4f}; "
                  f"sum H / added {d3['sum_H_rel']:.1e}; per-halo "
                  f"{max(d3['max_halo_cons_stars'], d3['max_halo_cons_gas']):.1e}; 2D sums "
                  f"{d2['sum_stars_rel']:.1e}/{d2['sum_gas_rel']:.1e}; explicit "
                  f"{res.get(f'explicit_check__{kk}', np.nan):.1e}; P/P_mm - 1 at k[0] "
                  f"{100 * q[0]:+.4f}%, k~5 {100 * q[i5]:+.2f}%", flush=True)
            gc.collect()
        del F, store
        gc.collect()

    for kk, e in ends.items():
        res[f"target__{kk}"] = e['target']
        res[f"s__{kk}"] = e['s']
        res[f"source__{kk}"] = e['source']
        res[f"n_capped__{kk}"] = int(e['capped'].sum())
        res[f"capped_frac__{kk}"] = e['capped_frac']
    if save:
        save_npz_atomic(out_path, **res)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None,
                        help="restrict to these entries (label, name or feedback)")
    parser.add_argument('--solve-only', action='store_true',
                        help="report s and the source of every end; no pass, nothing saved")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max-chunks', type=int, default=None,
                        help="smoke test: read only the first N snapshot chunks; nothing is saved")
    args = parser.parse_args()

    config = load_config(args.path2config)
    for key in ('fstar_obs', 'lensing'):
        if key not in config:
            raise SystemExit(f"{args.path2config} has no '{key}' block")
    if not args.solve_only:
        numba.set_num_threads(int(config['pk'].get('threads', numba.get_num_threads())))
    for entry in select_sims(config, args.sims):
        print(f"\n===== {sim_label(entry)} (3D grid {entry['n_pixels']}^3) =====", flush=True)
        run_sim(entry, config, args.solve_only, args.overwrite, args.max_chunks)
        gc.collect()


if __name__ == '__main__':
    main()
