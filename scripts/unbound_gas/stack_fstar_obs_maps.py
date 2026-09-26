"""stack_fstar_obs_maps.py
========================
Lensing stacks of the capped ends of the observation-based stellar bands
(compute_fstar_obs.py): the 2D maps of the stars added (S) and the ionized gas
removed (G), stacked with the lensing script's DSigma settings on the same
fixed halo sample as the s = 0 stacks of stack_stellar_maps.py (checked:
identical rows, same settings). With N, T the stored halo-mean DSigma of the
cached ionized_gas and total maps,

    f(theta) = [N - S_G] / [T + S_S - S_G] * Omega_m / Omega_b .

Ends without a capped region need no stacks (make_pk_fstar_obs.py takes them
from the s = 0 stacks at stellar scale s).

Validation (saved): the halo rows and settings equal the stored lensing
stacks; the explicit maps ionized_gas - G/2 and total + (S - G)/2 of the
first end stack to the linear combination; compute_fstar_obs.py's 2D
bookkeeping is copied in (``mapdiag__*``).

Output (products/2D/, rewritten on every run):
  <stem>_lensing_fstar_obs_<sample_name>.npz
      theta_arcmin, ends, s__<end>, S_mean__<end>, S_std__<end>, G_mean__<end>,
      G_std__<end> (<end> = <option>__<low|high>), n_haloes, settings, checks.

Run from the scripts/ directory (a whole node for FLAMINGO; runINT_fstar_obs.sh):
    python unbound_gas/stack_fstar_obs_maps.py -p configs/unbound_gas/pk_fstar_obs_z05.yaml --sims Illustris-1
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

import stack_stellar_maps as ssm
from compute_fstar_obs import obs_map_path, obs_path, obs_settings
from compute_stellar_maps import (field2d_path, lensing_settings, lensing_sim, map_n_pixels,
                                  sample_settings, stack_path)
from pk_common import load_config, save_npz_atomic, select_sims, sim_label


def obs_stack_path(entry: dict, sample_name: str):
    """Path of the capped ends' stacks of one halo sample."""
    p = stack_path(entry, sample_name)
    return p.with_name(p.name.replace('_lensing_stellar_', '_lensing_fstar_obs_'))


def run_sim(entry: dict, config: dict, nproc) -> None:
    """Stacks of the capped ends' maps for one simulation."""
    from stacker import SimulationStacker

    obs = obs_settings(config)
    name, tag = obs['variant']['name'], obs['tag']
    lens = lensing_settings(config)
    stack = lens['stack']
    lsim, z = lensing_sim(lens, entry)
    if lsim is None:
        print(f"  not in the lensing config {lens['config_path']}; skipping")
        return
    rpath = obs_path(entry, name)
    if not rpath.exists():
        print(f"  no {rpath.name} (run compute_fstar_obs.py); not stacked")
        return
    with np.load(rpath) as f:
        ends = [str(e) for e in f['ends'] if str(f[f"source__{e}"]) == 'pass']
        svals = {e: float(f[f"s__{e}"]) for e in ends}
        mapdiag = {kk: f[kk] for kk in f.files if kk.startswith('mapdiag__')}
    if not ends:
        print("  no capped end: nothing to stack")
        return
    proj = stack['projection']
    st = SimulationStacker(entry['name'], entry['snapshot'], simType=entry['sim_type'],
                           feedback=entry.get('feedback'), z=z)
    n = map_n_pixels(st, z, stack['pixel_size'])
    missing = [e for e in ends if not all(obs_map_path(entry, e, k, name, tag, n, proj).exists()
                                          for k in ('stars', 'gas'))]
    if missing:
        raise FileNotFoundError(f"maps of {missing} missing (rerun compute_fstar_obs.py)")
    ion, tot = (field2d_path(entry, p, n, proj) for p in ('ionized_gas', 'total'))

    # Same sample and settings as the stored s = 0 stacks.
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
          f"s = 0 stacks ({time.time() - t0:.0f} s)", flush=True)

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
    for e in ends:
        tasks.append(('array', f"S__{e}", [(obs_map_path(entry, e, 'stars', name, tag, n, proj), 1.0)]))
        tasks.append(('array', f"G__{e}", [(obs_map_path(entry, e, 'gas', name, tag, n, proj), 1.0)]))
    e0 = ends[0]
    s0, g0 = (obs_map_path(entry, e0, k, name, tag, n, proj) for k in ('stars', 'gas'))
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

    radii = out[f"S__{e0}"]['radii']
    res = dict(theta_arcmin=radii * stack['rad_distance'], radii=radii, n_haloes=len(mask),
               z=z, n_pixels=n, projection=proj, settings=sample_settings(stack, z),
               sample_name=lens['sample_name'], lensing_config=lens['config_path'],
               variant=name, tag=tag, ends=np.array(ends))
    for e in ends:
        res[f"s__{e}"] = svals[e]
        for kk in ('S', 'G'):
            res[f"{kk}_mean__{e}"] = out[f"{kk}__{e}"]['mean']
            res[f"{kk}_std__{e}"] = out[f"{kk}__{e}"]['std']
    S0, G0 = res[f"S_mean__{e0}"], res[f"G_mean__{e0}"]
    res['explicit_end'] = e0
    res['explicit_check_N'] = ssm._rel(out['N_explicit']['mean'], N_ref - 0.5 * G0)
    res['explicit_check_T'] = ssm._rel(out['T_explicit']['mean'], T_ref + 0.5 * (S0 - G0))
    print(f"  explicit maps ({e0}) vs linear combination: N {res['explicit_check_N']:.1e}, "
          f"T {res['explicit_check_T']:.1e}")
    res.update(mapdiag)
    save_npz_atomic(obs_stack_path(entry, lens['sample_name']), **res)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None,
                        help="restrict to these entries (label, name or feedback)")
    parser.add_argument('--nproc', type=int, default=None,
                        help="worker processes (default: one per stack, at most the CPU count)")
    args = parser.parse_args()

    config = load_config(args.path2config)
    for key in ('fstar_obs', 'lensing'):
        if key not in config:
            raise SystemExit(f"{args.path2config} has no '{key}' block")
    for entry in select_sims(config, args.sims):
        print(f"\n===== {sim_label(entry)} =====", flush=True)
        run_sim(entry, config, args.nproc)


if __name__ == '__main__':
    main()
