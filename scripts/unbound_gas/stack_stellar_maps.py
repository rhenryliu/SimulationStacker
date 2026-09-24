"""stack_stellar_maps.py
======================
Stacked DSigma profiles for the lensing observable under the halo-level
stellar-to-ionized-gas transfer (bottom panel of make_pk_stellar.py's S(k)
band figures).

For each simulation of the config that is also in the lensing config named in
its ``lensing`` block, this stacks on one fixed halo sample -- the lensing
script's (``lensing/beam_compensated_ratio_v2.py``; e.g. SHAM subhaloes at the
configured abundance), with its settings
(``compute_stellar_maps.resolve_stack_settings``):

  N, T  the cached ionized_gas and total maps, through
        ``SimulationStacker.stackMap`` exactly as the lensing script does;
  A, B  the transfer maps of every configuration (compute_stellar_maps.py),
        through ``stack_on_array`` on the same haloes;

and saves the halo means, from which make_pk_stellar.py evaluates

    f(s) = [N + t A] / [T + t (A - B)] * Omega_m / Omega_b ,   t = 1 - s .

The stacked sample does not change with s (the same observed galaxies), and
it is separate from the transfer's halo selection (every host halo above the
mass cut, box-wide). Another sample: the config's ``lensing.overrides`` and
``lensing.sample_name`` (its own output file; the maps are reused).

Validation (saved):
  - ``stack_on_array`` on the cached ionized_gas / total maps with the sample
    selected here equals ``stackMap``'s profile (same map, haloes and order);
  - the explicitly built maps ionized_gas + A/2 and total + (A - B)/2 of the
    first configuration stack to the linear combination (s = 0.5);
  - the map bookkeeping of compute_stellar_maps.py is copied in (``mapdiag__*``).

Output (products/2D/, rewritten on every run; one file per simulation and sample):
  <stem>_lensing_stellar_<sample_name>.npz
      theta_arcmin, N_mean/N_std, T_mean/T_std, f_sim (s = 1), omega_b,
      omega_m, n_haloes, halo_rows, z, settings, and per configuration
      <variant>__<tag>: A_mean__*, A_std__*, B_mean__*, B_std__*.

The stacks run in a process pool, one map per task (pure-Python halo loop,
~1 ms per halo and map; FLAMINGO has ~1.6e5 haloes at z ~ 0.5, ~3.2e5 at 0.3).

Run from the scripts/ directory (a whole node for FLAMINGO; runINT_stellar_maps.sh):
    python unbound_gas/stack_stellar_maps.py -p configs/unbound_gas/pk_components_z05.yaml \
        --sims m100n1024
    # quick check on the first 50 haloes, nothing saved (fine on a login node
    # for the small boxes)
    python unbound_gas/stack_stellar_maps.py -p ... --sims Illustris-1 --max-halos 50 --nproc 4
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from compute_pk_stellar import mass_tag, variants_of
from compute_stellar_maps import (f_of_scale, field2d_path, lensing_settings, lensing_sim,
                                  map_n_pixels, map_path, maps_diag_path, omega_b_of,
                                  sample_settings, stack_path)
from pk_common import load_config, save_npz_atomic, select_sims, sim_label

# Set in the parent before the pool forks (read-only in the workers).
_ST = None       # SimulationStacker
_MASK = None     # stacked halo rows
_ARRAY_KW = None  # stack_on_array arguments
_MAP_KW = None   # stackMap arguments


def halo_sample(st, stack: dict) -> np.ndarray:
    """Rows stacked by ``SimulationStacker.stack_on_array`` for these settings.

    Replicates its selection (checked against stackMap in every run): with
    subhaloes, abundance matching on SubhaloMStar among subhaloes whose parent
    GroupMass <= halo_mass_upper; otherwise the 'massive' host selection.
    """
    from halos import select_halos
    box = st.header['BoxSize']
    if stack['use_subhalos']:
        sub = st.loadSubHalos()
        mstar = sub['SubhaloMStar']
        if stack['halo_mass_upper'] is None:
            return select_halos(mstar, 'abundance', target_number=stack['halo_abundance_target'],
                                Lbox=box)
        parent_mass = st.loadHalos()['GroupMass'][sub['SubhaloGrNr']]
        valid = np.where(parent_mass <= stack['halo_mass_upper'])[0]
        local = select_halos(mstar[valid], 'abundance', target_number=stack['halo_abundance_target'],
                             Lbox=box)
        return valid[local]
    return select_halos(st.loadHalos()['GroupMass'], 'massive',
                        target_average_mass=stack['halo_mass_avg'],
                        upper_mass_bound=stack['halo_mass_upper'])


def _task(spec):
    """Stack one map: ('stackMap', key, pType) or ('array', key, [(path, coef), ...])."""
    kind, key, payload = spec
    t0 = time.time()
    if kind == 'stackMap':
        radii, prof = _ST.stackMap(payload, **_MAP_KW)
    else:
        arr = None
        for path, coef in payload:
            a = np.load(path)
            if arr is None:
                arr = a if coef == 1.0 else coef * a
            else:
                arr += a if coef == 1.0 else coef * a
            del a
        radii, prof = _ST.stack_on_array(arr, halo_mask=_MASK, **_ARRAY_KW)
        del arr
    return key, radii, prof.mean(axis=1), prof.std(axis=1), prof.shape[1], time.time() - t0


def _rel(x, ref) -> float:
    return float(np.max(np.abs(x - ref)) / np.max(np.abs(ref)))


def run_sim(entry: dict, config: dict, nproc, max_halos) -> None:
    """Stacks of the cached and transfer maps for one simulation."""
    global _ST, _MASK, _ARRAY_KW, _MAP_KW
    from stacker import SimulationStacker

    lens = lensing_settings(config)
    stack = lens['stack']
    lsim, z = lensing_sim(lens, entry)
    if lsim is None:
        print(f"  not in the lensing config {lens['config_path']}; skipping")
        return
    proj = stack['projection']
    st = SimulationStacker(entry['name'], entry['snapshot'], simType=entry['sim_type'],
                           feedback=entry.get('feedback'), z=z)
    n = map_n_pixels(st, z, stack['pixel_size'])
    ion, tot = (field2d_path(entry, p, n, proj) for p in ('ionized_gas', 'total'))
    for p in (ion, tot):
        if not p.exists():
            raise FileNotFoundError(f"no cached map {p} (z = {z}, n = {n})")

    tags = [mass_tag(m) for m in sorted(float(m) for m in config['stellar']['halo_mass_min'])]
    configs, missing = [], []
    for v in variants_of(config):
        for tag in tags:
            ok = all(map_path(entry, k, v['name'], tag, n, proj).exists() for k in ('added', 'removed'))
            (configs if ok else missing).append((v['name'], tag))
    if missing:
        print(f"  no transfer maps for {missing} (run compute_stellar_maps.py); not stacked")

    t0 = time.time()
    mask = halo_sample(st, stack)
    if max_halos:
        mask = mask[:max_halos]
    print(f"  z = {z}, {n}^2 pixels ({proj}); {len(mask):,} haloes stacked "
          f"({time.time() - t0:.0f} s)", flush=True)

    _ST, _MASK = st, mask
    _ARRAY_KW = dict(filterType=stack['filter_type'], minRadius=stack['min_radius'],
                     maxRadius=stack['max_radius'], numRadii=stack['num_radii'], projection=proj,
                     radDistance=stack['rad_distance'], radDistanceUnits='arcmin',
                     use_subhalos=stack['use_subhalos'], halo_mass_avg=stack['halo_mass_avg'],
                     halo_mass_upper=stack['halo_mass_upper'],
                     halo_abundance_target=stack['halo_abundance_target'], z=z,
                     pixelSize=stack['pixel_size'])
    _MAP_KW = dict(filterType=stack['filter_type'], minRadius=stack['min_radius'],
                   maxRadius=stack['max_radius'], numRadii=stack['num_radii'], projection=proj,
                   save=False, load=True, radDistance=stack['rad_distance'],
                   pixelSize=stack['pixel_size'], beamSize=None, mask=False,
                   subtract_mean=False, use_subhalos=stack['use_subhalos'],
                   halo_abundance_target=stack['halo_abundance_target'],
                   halo_mass_avg=stack['halo_mass_avg'], halo_mass_upper=stack['halo_mass_upper'])

    tasks = [] if max_halos else [('stackMap', 'N_map', 'ionized_gas'), ('stackMap', 'T_map', 'total')]
    tasks += [('array', 'N', [(ion, 1.0)]), ('array', 'T', [(tot, 1.0)])]
    for v, tag in configs:
        tasks.append(('array', f"A__{v}__{tag}", [(map_path(entry, 'added', v, tag, n, proj), 1.0)]))
        tasks.append(('array', f"B__{v}__{tag}", [(map_path(entry, 'removed', v, tag, n, proj), 1.0)]))
    if configs:  # explicit maps at s = 0.5 for the first configuration
        a0, b0 = (map_path(entry, k, *configs[0], n, proj) for k in ('added', 'removed'))
        tasks.append(('array', 'N_explicit', [(ion, 1.0), (a0, 0.5)]))
        tasks.append(('array', 'T_explicit', [(tot, 1.0), (a0, 0.5), (b0, -0.5)]))

    nproc = min(len(tasks), nproc or (os.cpu_count() or 1))
    print(f"  {len(tasks)} stacks on {nproc} processes", flush=True)
    t0 = time.time()
    out = {}
    with mp.get_context('fork').Pool(nproc) as pool:
        for key, radii, mean, std, nh, dt in pool.imap_unordered(_task, tasks):
            out[key] = dict(radii=radii, mean=mean, std=std, n=nh)
            print(f"    {key}: {dt:.0f} s", flush=True)
    print(f"  stacks done in {time.time() - t0:.0f} s", flush=True)

    radii = out['N']['radii']
    for key, o in out.items():
        if not np.array_equal(o['radii'], radii):
            raise RuntimeError(f"radii of {key} differ")
        if key.endswith('_map'):
            continue
        if o['n'] != len(mask):
            raise RuntimeError(f"{key} stacked {o['n']} haloes, expected {len(mask)}")
    N = out['N_map'] if 'N_map' in out else out['N']
    T = out['T_map'] if 'T_map' in out else out['T']
    omega_b = omega_b_of(st)
    omega_m = float(st.header['Omega0'])
    factor = omega_m / omega_b
    res = dict(theta_arcmin=radii * stack['rad_distance'], radii=radii,
               N_mean=N['mean'], N_std=N['std'], T_mean=T['mean'], T_std=T['std'],
               f_sim=N['mean'] / T['mean'] * factor, omega_b=omega_b, omega_m=omega_m,
               factor=factor, n_haloes=len(mask), halo_rows=np.asarray(mask),
               z=z, n_pixels=n, projection=proj, settings=sample_settings(stack, z),
               sample_name=lens['sample_name'], lensing_config=lens['config_path'],
               configs=np.array([f"{v}__{t}" for v, t in configs]))

    # Validation: the lensing script's own stacks against the sample selected here.
    if 'N_map' in out:
        if N['n'] != len(mask) or T['n'] != len(mask):
            raise RuntimeError(f"stackMap stacked {N['n']} / {T['n']} haloes, the sample "
                               f"selected here has {len(mask)}")
        res['check_stackmap_N'] = _rel(out['N']['mean'], N['mean'])
        res['check_stackmap_T'] = _rel(out['T']['mean'], T['mean'])
        print(f"  stack_on_array vs stackMap: N {res['check_stackmap_N']:.1e}, "
              f"T {res['check_stackmap_T']:.1e}")
        # A and B are stacked on the sample selected here; if it differed from
        # stackMap's own selection, f(s) would mix two samples -- refuse to save.
        if max(res['check_stackmap_N'], res['check_stackmap_T']) > 1e-12:
            raise RuntimeError("the halo sample selected here does not reproduce stackMap's "
                               "profiles; halo_sample() is out of sync with stack_on_array")
    for v, tag in configs:
        c = f"{v}__{tag}"
        for k in ('A', 'B'):
            res[f"{k}_mean__{c}"] = out[f"{k}__{c}"]['mean']
            res[f"{k}_std__{c}"] = out[f"{k}__{c}"]['std']
    if configs:
        c = f"{configs[0][0]}__{configs[0][1]}"
        A0, B0 = out[f"A__{c}"]['mean'], out[f"B__{c}"]['mean']
        res['explicit_config'] = c
        res['explicit_check_N'] = _rel(out['N_explicit']['mean'], out['N']['mean'] + 0.5 * A0)
        res['explicit_check_T'] = _rel(out['T_explicit']['mean'], out['T']['mean'] + 0.5 * (A0 - B0))
        print(f"  explicit maps at s = 0.5 ({c}) vs linear combination: "
              f"N {res['explicit_check_N']:.1e}, T {res['explicit_check_T']:.1e}")

    dpath = maps_diag_path(entry, n, proj)
    if dpath.exists():
        with np.load(dpath) as d:
            for k in d.files:
                if k.startswith('diag__'):
                    res['map' + k] = d[k]

    th = res['theta_arcmin']
    print(f"  f(s=1) at theta = {th[0]:g}, {th[-1]:g} arcmin: {res['f_sim'][0]:.4f}, {res['f_sim'][-1]:.4f}")
    for v, tag in configs:
        c = f"{v}__{tag}"
        f0 = f_of_scale(N['mean'], T['mean'], res[f"A_mean__{c}"], res[f"B_mean__{c}"], 0.0, factor)
        print(f"    f(s=0) {v} {tag}: {f0[0]:.4f}, {f0[-1]:.4f}")
    if max_halos:
        print("  --max-halos: nothing saved")
    else:
        save_npz_atomic(stack_path(entry, lens['sample_name']), **res)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None,
                        help="restrict to these entries (label, name or feedback)")
    parser.add_argument('--nproc', type=int, default=None,
                        help="worker processes (default: one per stack, at most the CPU count)")
    parser.add_argument('--max-halos', type=int, default=None,
                        help="quick check: stack only the first N haloes of the sample; nothing saved")
    args = parser.parse_args()

    config = load_config(args.path2config)
    if 'lensing' not in config:
        raise SystemExit(f"{args.path2config} has no 'lensing' block")
    for entry in select_sims(config, args.sims):
        print(f"\n===== {sim_label(entry)} =====", flush=True)
        run_sim(entry, config, args.nproc, args.max_halos)


if __name__ == '__main__':
    main()
