"""stack_fstar_maps.py
====================
Lensing stacks of the stellar-fraction floor maps (compute_fstar.py): the
high-f* end of make_pk_fstar.py's f_gas(theta) band.

For each simulation of the config, the 2D maps of the stars added (S) and the
ionized gas removed (G) by the floor, for every configuration, are stacked
with the lensing script's DSigma settings on the same fixed halo sample as
the f* = 0 stacks of stack_stellar_maps.py (checked: identical rows, same
settings). With N, T the stored halo-mean DSigma of the cached ionized_gas and
total maps,

    f(theta) = [N - S_G] / [T + S_S - S_G] * Omega_m / Omega_b .

Validation (saved): the halo rows and settings equal the stored lensing
stacks; the explicit maps ionized_gas - G/2 and total + (S - G)/2 of the first
configuration stack to the linear combination; the 2D bookkeeping of
compute_fstar.py is copied in (``mapdiag__*``).

Output (products/2D/, rewritten on every run):
  <stem>_lensing_fstar_<sample_name>.npz
      theta_arcmin, target__<cfg>, S_mean__<cfg>, S_std__<cfg>, G_mean__<cfg>,
      G_std__<cfg> (<cfg> = <variant>__<tag>), n_haloes, settings, checks.

Run from the scripts/ directory (a whole node for FLAMINGO; runINT_fstar.sh):
    python unbound_gas/stack_fstar_maps.py -p configs/unbound_gas/pk_fstar_z05.yaml --sims m100n1024
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

import stack_stellar_maps as ssm
from compute_fstar import fstar_map_path, fstar_maps_diag_path, targets_of
from compute_pk_stellar import mass_tag, variants_of
from compute_stellar_maps import (field2d_path, lensing_settings, lensing_sim, map_n_pixels,
                                  sample_settings, stack_path)
from pk_common import load_config, save_npz_atomic, select_sims, sim_label


def fstar_stack_path(entry: dict, sample_name: str):
    """Path of the floor stacks of one halo sample."""
    p = stack_path(entry, sample_name)
    return p.with_name(p.name.replace('_lensing_stellar_', '_lensing_fstar_'))


def run_sim(entry: dict, config: dict, nproc) -> None:
    """Stacks of the floor maps for one simulation."""
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
    targets = targets_of(config)
    tags = [mass_tag(m) for m in sorted(float(m) for m in config['stellar']['halo_mass_min'])]
    configs, missing = [], []
    for v in variants_of(config):
        for tag in tags:
            ok = all(fstar_map_path(entry, k, targets[v['name']], v['name'], tag, n, proj).exists()
                     for k in ('stars', 'gas'))
            (configs if ok else missing).append((v['name'], tag))
    if missing:
        print(f"  no floor maps for {missing} (run compute_fstar.py); not stacked")
    if not configs:
        return

    # The f* = 0 end uses the stored lensing stacks: same sample, same settings.
    ref_path = stack_path(entry, lens['sample_name'])
    if not ref_path.exists():
        raise FileNotFoundError(f"{ref_path} missing (run stack_stellar_maps.py first)")
    with np.load(ref_path) as ref:
        ref_rows, ref_settings = ref['halo_rows'], str(ref['settings'])
        N_ref, T_ref = ref['N_mean'], ref['T_mean']
    if ref_settings != sample_settings(stack, z):
        raise ValueError(f"{ref_path} was stacked with other settings than the config's lensing block")
    t0 = time.time()
    mask = ssm.halo_sample(st, stack)
    if not np.array_equal(mask, ref_rows):
        raise RuntimeError("the halo sample differs from the stored lensing stacks")
    print(f"  z = {z}, {n}^2 pixels ({proj}); {len(mask):,} haloes, identical to the stored "
          f"f* = 0 stacks ({time.time() - t0:.0f} s)", flush=True)

    # Reuse stack_stellar_maps' worker (ssm._task): it reads these module
    # globals, which the 'fork' pool below hands to the workers.
    ssm._ST, ssm._MASK = st, mask
    ssm._ARRAY_KW = dict(filterType=stack['filter_type'], minRadius=stack['min_radius'],
                         maxRadius=stack['max_radius'], numRadii=stack['num_radii'],
                         projection=proj, radDistance=stack['rad_distance'],
                         radDistanceUnits='arcmin', use_subhalos=stack['use_subhalos'],
                         halo_mass_avg=stack['halo_mass_avg'],
                         halo_mass_upper=stack['halo_mass_upper'],
                         halo_abundance_target=stack['halo_abundance_target'], z=z,
                         pixelSize=stack['pixel_size'])
    tasks = []
    for v, tag in configs:
        T = targets[v]
        tasks.append(('array', f"S__{v}__{tag}", [(fstar_map_path(entry, 'stars', T, v, tag, n, proj), 1.0)]))
        tasks.append(('array', f"G__{v}__{tag}", [(fstar_map_path(entry, 'gas', T, v, tag, n, proj), 1.0)]))
    v0, t0_ = configs[0]
    s0, g0 = (fstar_map_path(entry, k, targets[v0], v0, t0_, n, proj) for k in ('stars', 'gas'))
    tasks.append(('array', 'N_explicit', [(ion, 1.0), (g0, -0.5)]))
    tasks.append(('array', 'T_explicit', [(tot, 1.0), (s0, 0.5), (g0, -0.5)]))

    nproc = min(len(tasks), nproc or (os.cpu_count() or 1))
    print(f"  {len(tasks)} stacks on {nproc} processes", flush=True)
    t1 = time.time()
    out = {}
    with mp.get_context('fork').Pool(nproc) as pool:
        for key, radii, mean, std, nh, dt in pool.imap_unordered(ssm._task, tasks):
            out[key] = dict(radii=radii, mean=mean, std=std, n=nh)
            print(f"    {key}: {dt:.0f} s", flush=True)
    print(f"  stacks done in {time.time() - t1:.0f} s", flush=True)
    for key, o in out.items():
        if o['n'] != len(mask):
            raise RuntimeError(f"{key} stacked {o['n']} haloes, expected {len(mask)}")

    radii = out[f"S__{v0}__{t0_}"]['radii']
    res = dict(theta_arcmin=radii * stack['rad_distance'], radii=radii, n_haloes=len(mask),
               z=z, n_pixels=n, projection=proj, settings=sample_settings(stack, z),
               sample_name=lens['sample_name'], lensing_config=lens['config_path'],
               configs=np.array([f"{v}__{t}" for v, t in configs]))
    for v, tag in configs:
        c = f"{v}__{tag}"
        res[f"target__{c}"] = targets[v]
        for kk in ('S', 'G'):
            res[f"{kk}_mean__{c}"] = out[f"{kk}__{c}"]['mean']
            res[f"{kk}_std__{c}"] = out[f"{kk}__{c}"]['std']
    c0 = f"{v0}__{t0_}"
    S0, G0 = res[f"S_mean__{c0}"], res[f"G_mean__{c0}"]
    res['explicit_config'] = c0
    res['explicit_check_N'] = ssm._rel(out['N_explicit']['mean'], N_ref - 0.5 * G0)
    res['explicit_check_T'] = ssm._rel(out['T_explicit']['mean'], T_ref + 0.5 * (S0 - G0))
    print(f"  explicit maps ({c0}) vs linear combination: N {res['explicit_check_N']:.1e}, "
          f"T {res['explicit_check_T']:.1e}")
    dpath = fstar_maps_diag_path(entry, n, proj)
    if dpath.exists():
        with np.load(dpath) as d:
            for kk in d.files:
                if kk.startswith('diag__'):
                    res['map' + kk] = d[kk]
    save_npz_atomic(fstar_stack_path(entry, lens['sample_name']), **res)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None,
                        help="restrict to these entries (label, name or feedback)")
    parser.add_argument('--nproc', type=int, default=None,
                        help="worker processes (default: one per stack, at most the CPU count)")
    args = parser.parse_args()

    config = load_config(args.path2config)
    for key in ('fstar', 'lensing'):
        if key not in config:
            raise SystemExit(f"{args.path2config} has no '{key}' block")
    for entry in select_sims(config, args.sims):
        print(f"\n===== {sim_label(entry)} =====", flush=True)
        run_sim(entry, config, args.nproc)


if __name__ == '__main__':
    main()
