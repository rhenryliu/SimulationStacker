"""validate_pk_caches.py
======================
Check the cached 3D fields that ``compute_pk_components.py`` builds its five
mass components from (see ``pk_common``), for every simulation in
``configs/unbound_gas/pk_components_z05.yaml``. Nothing is written except the
report on stdout.

Checks per simulation (grid n^3, cached gas, ionized_gas, Stars, BH, total):

1. Every field is finite and non-negative.
2. Derived DM = total - gas - Stars - BH is non-negative up to float32
   rounding. Clearly negative cells mean the total and component caches do
   not come from exactly the same build; fails if their (negative) mass
   exceeds 1e-6 of the box total, since DM is never cached at these grids.
3. Mass budget against the header cosmology: sum(total) / (rho_crit V) vs
   Omega_m, and the baryon share sum(gas + Stars + BH) / sum(total) vs
   Omega_b / Omega_m.
4. Ionized vs total gas: global ratio, and the fraction of cells where the
   ionized mass exceeds the gas mass (neutral_gas < 0 beyond rounding).
5. Each component's 3D total mass vs the cached unmasked 2D field of the same
   component (binned by an independent code path, binned_statistic_2d).
6. Optional (--rebuild): re-bin one component from the particles with the
   current mapMaker code and compare it to the cache (relative L2 norm and
   total). The decisive end-to-end test is compute_pk_components.py --fresh.
7. Where a DM_512 cache exists, the power spectrum of the derived DM vs that of
   DM_512 at k below half the 512-grid Nyquist frequency.

Run from the scripts/ directory on a whole CPU node:
    python unbound_gas/validate_pk_caches.py -p configs/unbound_gas/pk_components_z05.yaml \
        --rebuild Stars --rebuild-sims TNG300-1 Illustris-1 m100n1024 'L1_m9 (L1_m9)'
"""

import argparse
import glob
import sys
import time

import numpy as np

from pk_common import (RHO_CRIT_H, field_path, header_of, load_config,
                       load_field, select_sims, sim_label)

sys.path.append('../src/')
from loadIO import resolve_data_root  # noqa: E402

CACHED = ['gas', 'ionized_gas', 'Stars', 'BH', 'total']


def report(ok: bool, msg: str, failures: list) -> None:
    """Print one check result and record failures."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        failures.append(msg)


def check_2d_totals(entry: dict, sums3d: dict, failures: list) -> None:
    """Compare 3D component totals with unmasked 2D field caches (check 5)."""
    root = resolve_data_root(None)
    stem = (f"{entry['name']}_{entry['feedback']}_{entry['snapshot']}"
            if entry.get('feedback') else f"{entry['name']}_{entry['snapshot']}")
    for p_type, s3 in sums3d.items():
        # Unmasked field caches only: <stem>_<ptype>_<n>_<proj>.npy (no _map/_masked).
        # The 2D grid differs from the 3D one (e.g. 2674 vs 1000) and the total
        # mass is independent of it, so any unmasked cache is a valid reference.
        pattern = f"{root}{entry['sim_type']}/products/2D/{stem}_{p_type}_[0-9]*_[xyz][xyz].npy"
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  [skip] no unmasked 2D cache for {p_type}")
            continue
        s2 = float(np.sum(np.load(files[-1], mmap_mode='r'), dtype=np.float64))
        rel = s2 / s3 - 1.0
        report(abs(rel) < 1e-4,
               f"{p_type}: 3D total vs 2D cache {files[-1].split('/')[-1]}: rel diff {rel:+.2e}",
               failures)


def check_rebuild(entry: dict, p_type: str, cached: np.ndarray, failures: list) -> None:
    """Re-bin one component from particles and compare to its cache (check 6).

    The rebuild accumulates in the cache's on-disk dtype (older caches are
    float64, current mapMaker.FIELD_3D_DTYPE is float32), so the comparison is
    like for like: float32 accumulation alone differs from float64 by ~1e-3 of
    the peak in dense cells. The module global is restored afterwards.
    """
    import mapMaker
    from stacker import SimulationStacker
    disk_dtype = np.load(field_path(entry['sim_type'], entry['name'], entry['snapshot'],
                                    entry.get('feedback'), p_type, entry['n_pixels']),
                         mmap_mode='r').dtype
    st = SimulationStacker(entry['name'], entry['snapshot'], simType=entry['sim_type'],
                           feedback=entry.get('feedback'), z=entry.get('redshift', 0.5))
    saved_dtype = mapMaker.FIELD_3D_DTYPE
    mapMaker.FIELD_3D_DTYPE = disk_dtype.type
    try:
        t0 = time.time()
        fresh = mapMaker.create_field(st, p_type, entry['n_pixels'], 'xy', dim='3D', load=False)
    finally:
        mapMaker.FIELD_3D_DTYPE = saved_dtype
    print(f"  rebuilt {p_type} in {time.time() - t0:.0f} s, accumulating in {fresh.dtype} "
          f"(cache on disk: {disk_dtype})")
    diff = fresh.astype(np.float32) - cached
    rel_l2 = float(np.sqrt(np.sum(diff.astype(np.float64) ** 2) /
                           np.sum(cached.astype(np.float64) ** 2)))
    maxdiff = float(np.max(np.abs(diff))) / float(np.max(cached))
    del diff
    rel_sum = float(np.sum(fresh, dtype=np.float64) / np.sum(cached, dtype=np.float64) - 1.0)
    # Like-for-like accumulation dtype (see docstring); the relative L2 norm is
    # the measure that matters for P(k). Tolerances leave room for the
    # non-deterministic summation order of the threaded TSC.
    report(rel_l2 < 1e-4 and abs(rel_sum) < 1e-5,
           f"rebuilt {p_type} vs cache: rel L2 diff = {rel_l2:.2e}, rel sum diff = "
           f"{rel_sum:+.2e} (max|diff|/max = {maxdiff:.2e})", failures)


def check_dm512(entry: dict, dm: np.ndarray, box_mpc: float, threads: int,
                failures: list) -> None:
    """Compare the derived DM power spectrum with a cached DM_512 field (check 7)."""
    path512 = field_path(entry['sim_type'], entry['name'], entry['snapshot'],
                         entry.get('feedback'), 'DM', 512)
    if not path512.exists():
        print("  [skip] no DM_512 cache")
        return
    import Pk_library as PKL
    n = dm.shape[0]
    d = dm / np.float32(np.mean(dm, dtype=np.float64)) - np.float32(1.0)
    pk_n = PKL.Pk(d, box_mpc, axis=0, MAS='TSC', threads=threads, verbose=False)
    del d
    d512 = np.array(np.load(path512), dtype=np.float32)
    d512 = d512 / np.float32(np.mean(d512, dtype=np.float64)) - np.float32(1.0)
    pk_512 = PKL.Pk(d512, box_mpc, axis=0, MAS='TSC', threads=threads, verbose=False)
    kmax = 0.5 * np.pi * 512 / box_mpc
    sel = (pk_512.k3D <= kmax)
    ratio = np.interp(pk_512.k3D[sel], pk_n.k3D, pk_n.Pk[:, 0]) / pk_512.Pk[sel, 0]
    dev = float(np.max(np.abs(ratio - 1.0)))
    report(dev < 0.02,
           f"P(k) derived DM ({n}^3) vs DM_512 for k <= {kmax:.2f} h/Mpc: "
           f"max |ratio-1| = {dev:.3e}", failures)


def validate(entry: dict, config: dict, rebuild: list, threads: int) -> list:
    """Run all checks for one simulation; return the list of failed checks."""
    failures = []
    hdr = header_of(entry)
    box_mpc = hdr['BoxSize'] / 1000.0
    n = entry['n_pixels']
    print(f"\n===== {sim_label(entry)}: grid {n}^3, box {box_mpc:.1f} Mpc/h =====")

    fields, sums = {}, {}
    for p_type in CACHED:
        f = load_field(entry, p_type)
        finite = bool(np.isfinite(f).all())
        fmin = float(f.min())
        sums[p_type] = float(np.sum(f, dtype=np.float64))
        report(finite and fmin >= 0.0,
               f"{p_type}: finite={finite}, min={fmin:.3e}, sum={sums[p_type]:.6e} M_sun/h",
               failures)
        fields[p_type] = f

    # 2. derived DM, evaluated slab by slab to avoid a second full-size temporary.
    # Pass/fail is on the mass in clearly negative cells relative to the box
    # total, not on the cell count: the SIMBA-100 caches have ~400 gas-only void
    # cells where the cached total sits a few per cent below the cached gas
    # (the caches come from different builds), yet a full particle rebuild of
    # all five components reproduces every component spectrum to < 1e-4 at
    # k <= 10 h/Mpc (compute_pk_components.py --fresh, Sep 2026).
    tol = 1e-5  # relative to the local total; float32 rounding is ~1e-7
    n_bad, worst, dm_sum, neg_mass = 0, 0.0, 0.0, 0.0
    for i in range(n):
        tot = fields['total'][i]
        dm = tot - fields['gas'][i] - fields['Stars'][i] - fields['BH'][i]
        dm_sum += float(np.sum(dm, dtype=np.float64))
        neg = dm < -tol * tot
        n_bad += int(neg.sum())
        neg_mass += float(np.sum(dm[neg], dtype=np.float64))
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(tot > 0, dm / tot, 0.0)
        worst = min(worst, float(r.min()))
    neg_frac = -neg_mass / sums['total']
    report(neg_frac < 1e-6,
           f"derived DM: negative mass / total = {neg_frac:.2e} ({n_bad} cells below "
           f"-{tol:g} x total; min(DM/total) = {worst:.3e})", failures)

    # 3. mass budget
    om = float(hdr['Omega0'])
    ob = hdr.get('OmegaBaryon')
    ob_src = 'header'
    if ob is None:
        ob = config['pk']['omega_baryon_fallback'].get(entry['name'])
        ob_src = 'config fallback'
    expected_tot = om * RHO_CRIT_H * box_mpc ** 3
    r_tot = sums['total'] / expected_tot
    report(abs(r_tot - 1.0) < 1e-3,
           f"sum(total) / (Omega_m rho_crit V) = {r_tot:.6f} (Omega_m = {om:.4f})", failures)
    fb = (sums['gas'] + sums['Stars'] + sums['BH']) / sums['total']
    if ob is not None:
        report(abs(fb / (float(ob) / om) - 1.0) < 1e-3,
               f"baryon share {fb:.5f} vs Omega_b/Omega_m = {float(ob) / om:.5f} "
               f"(Omega_b = {float(ob)} from {ob_src})", failures)
    else:
        print(f"  [info] baryon share {fb:.5f}; no Omega_b available")
    print(f"  [info] DM share {dm_sum / sums['total']:.5f}")

    # 4. ionized vs total gas
    ion_frac = sums['ionized_gas'] / sums['gas']
    over = 0
    for i in range(n):
        g = fields['gas'][i]
        over += int(np.sum(fields['ionized_gas'][i] > g * (1 + tol) + 1e-30))
    report(ion_frac <= 1.0 + 1e-6,
           f"global ionized/gas = {ion_frac:.5f}; cells with ionized > gas: "
           f"{over} ({over / n ** 3:.2e} of all)", failures)
    # Where ionized > gas (e.g. metal-enriched, fully ionized gas converted with
    # a fixed X_H = 0.76), the derived neutral_gas is negative; report its mass.
    neg_neu = 0.0
    for i in range(n):
        d = fields['gas'][i] - fields['ionized_gas'][i]
        neg_neu += float(np.sum(d[d < 0], dtype=np.float64))
    print(f"  [info] negative neutral_gas mass = {-neg_neu / sums['gas']:.4f} of all gas "
          f"(net neutral = {1 - ion_frac:.4f} of gas)")
    baryons = sums['gas'] + sums['Stars'] + sums['BH']
    print(f"  [info] global baryon budget: ionized {sums['ionized_gas'] / baryons:.4f}, "
          f"neutral {(sums['gas'] - sums['ionized_gas']) / baryons:.4f}, "
          f"stars {sums['Stars'] / baryons:.4f}, BH {sums['BH'] / baryons:.5f}")

    # 5. 2D totals
    check_2d_totals(entry, {p: sums[p] for p in CACHED}, failures)

    # 6. rebuild
    for p_type in rebuild:
        check_rebuild(entry, p_type, fields[p_type], failures)

    # 7. derived DM spectrum vs DM_512 (build DM in place in the total buffer)
    dm = fields.pop('total')
    for p_type in ('gas', 'Stars', 'BH'):
        dm -= fields.pop(p_type)
    fields.clear()
    check_dm512(entry, dm, box_mpc, threads, failures)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None,
                        help="restrict to these entries (label, name or feedback)")
    parser.add_argument('--rebuild', nargs='*', default=[],
                        help="components to re-bin from particles and compare (e.g. Stars)")
    parser.add_argument('--rebuild-sims', nargs='*', default=None,
                        help="restrict --rebuild to these entries (default: all selected)")
    args = parser.parse_args()

    config = load_config(args.path2config)
    threads = int(config['pk'].get('threads', 1))
    summary = {}
    rebuild_for = {sim_label(e) for e in select_sims(config, args.rebuild_sims)}
    for entry in select_sims(config, args.sims):
        t0 = time.time()
        rebuild = args.rebuild if sim_label(entry) in rebuild_for else []
        summary[sim_label(entry)] = validate(entry, config, rebuild, threads)
        print(f"  ({time.time() - t0:.0f} s)")

    print("\n########## summary ##########")
    n_fail = 0
    for label, fails in summary.items():
        print(f"{label:28s} {'OK' if not fails else f'{len(fails)} FAILED'}")
        for f in fails:
            print(f"    - {f}")
        n_fail += len(fails)
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
