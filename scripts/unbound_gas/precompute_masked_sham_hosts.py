"""precompute_masked_sham_hosts.py

Build the masked SZ maps read by the unbound gas paper's masking figures
(simulated_kSZ_masked.py, simulated_tSZ_masked.py) around the host haloes of
the galaxies those figures stack.

For one simulation and particle type:

1. select the SHAM galaxies at the simulation's lensing-fitted number density
   (``halo_stats.select_sample`` with the density from
   ``lensing/fit_dsigma_ksz.py``'s npz), exactly as the figure scripts do;
2. take their unique parent FoF haloes (``SubhaloGrNr``);
3. for each masking radius n, keep only the gas within n x R200m of each host
   (its own ``GroupRad``, centred on the loader's ``GroupPos`` -- CAESAR
   ``pos`` for SIMBA), project, and convolve with the 1.6 arcmin beam, through
   the same ``mapMaker.create_masked_field`` + ``utils.fft_smoothed_map`` path
   as ``SimulationStacker.makeMap``;
4. save the map at the standard masked-map cache path
   (``..._map_masked{n}R200c.npy``; the "R200c" in the name is historical).

IMPORTANT: the cache path does not encode the halo sample. Once this script
has run, those paths hold masks around the SHAM hosts at the lensing-fitted
density (unbound gas paper decision, 2026-09-29), not around the mass-cut
sample that ``SimulationStacker.makeField(mask=True)`` would build. A JSON file
next to each map (``<map>.sample.json``) records the sample, and a figure
script run with ``load_field: true`` loads the map as it is. Do not let a
figure script rebuild a missing masked map: makeField would silently use the
mass-cut sample and save it at the same path.

Each map is written to ``<map>.tmp.npy`` and renamed only after it is
complete. An existing map is skipped unless ``--overwrite`` is given; this
script does not move anything to the trash -- the runners
(runCPU_masked_sham_hosts*.sh) move the old maps to the scratch trash before
calling it, and ``--overwrite`` should not be used on a map that has not been
moved away first. The script stops before any work if the cached 3D cube is
missing (create_masked_field would otherwise rebuild it from the particles).

Memory: the 3D cube is reloaded for every radius (create_masked_field
multiplies it in place). FLAMINGO needs a full CPU node per (variant, ptype):
a 3548^3 float32 cube is 179 GB plus a 45 GB boolean mask.

Usage (from the scripts/ directory):
    python unbound_gas/precompute_masked_sham_hosts.py --sim-type SIMBA \\
        --name m100n1024 --snapshot 125 --feedback s50 --ptype tau
    # validation / preview: build in memory, compare, save nothing
    python unbound_gas/precompute_masked_sham_hosts.py ... --dry-run
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append('../src/')
from stacker import SimulationStacker  # type: ignore
from mapMaker import create_masked_field  # type: ignore
from loadIO import _get_data_filepath  # type: ignore
from utils import fft_smoothed_map  # type: ignore
from halos import select_halos  # type: ignore
# Sibling module in this directory; after the '../src/' append above.
from halo_stats import fit_label, load_fitted_abundances, map_geometry, select_sample

# Projection each figure stacks along (tau_z05_CAP_masked_flamingo.yaml and
# tSZ_z05_CAP_masked.yaml).
_DEFAULT_PROJECTION = {'tau': 'xy', 'tSZ': 'xz'}
_DEFAULT_FITS = ['../data/fit_dsigma_ksz/fit_dsigma_ksz_z05.npz',
                 '../data/fit_dsigma_ksz/fit_dsigma_ksz_z05_simba50.npz']


def mask_hosts(stacker, sample, density, mass_upper=5e14, mass_avg=10 ** 13.22):
    """Indices into the halo catalogue of the haloes to keep gas around.

    Args:
        stacker (SimulationStacker): The simulation.
        sample (str): 'sham-hosts' (unique parents of the SHAM galaxies at
            ``density``) or 'masscut' (the mass-cut sample makeField uses;
            for validation only).
        density (float): SHAM number density in (cMpc/h)^-3.
        mass_upper (float): Parent / halo mass cap in Msun/h.
        mass_avg (float): Mass-cut target mean mass in Msun/h.

    Returns:
        tuple: (host indices, dict of sample statistics).
    """
    haloes = stacker.loadHalos()
    if sample == 'masscut':
        idx = np.asarray(select_halos(haloes['GroupMass'], 'massive',
                                      target_average_mass=mass_avg, upper_mass_bound=mass_upper))
        return idx, dict(n_galaxies=None, n_hosts=int(idx.size))
    subs = stacker.loadSubHalos()
    gal = select_sample(stacker, use_subhalos=True, halo_abundance_target=density,
                        halo_mass_avg=mass_avg, halo_mass_upper=mass_upper,
                        haloes=haloes, subhalos=subs)
    hosts = np.unique(np.asarray(subs['SubhaloGrNr'], dtype=np.int64)[gal])
    return hosts, dict(n_galaxies=int(np.size(gal)), n_hosts=int(hosts.size))


def build(stacker, ptype, projection, hosts, mask_rad, z, pixel_size, beam):
    """Beam-convolved map keeping only the gas within mask_rad x R200m of the hosts."""
    n, arcpp = map_geometry(stacker, z, pixel_size)
    h = stacker.loadHalos()
    cat = {'GroupMass': h['GroupMass'][hosts],
           'GroupRad': h['GroupRad'][hosts] * mask_rad,
           'GroupPos': h['GroupPos'][hosts]}
    f2d = create_masked_field(stacker, ptype, n, cat, projection=projection, save3D=False,
                              load3D=True, base_path=stacker.base_path)
    return fft_smoothed_map(f2d, beam, pixel_size_arcmin=arcpp), n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--sim-type', required=True, choices=['IllustrisTNG', 'SIMBA', 'FLAMINGO'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--snapshot', type=int, required=True)
    ap.add_argument('--feedback', default=None)
    ap.add_argument('--ptype', required=True, choices=['tau', 'tSZ'])
    ap.add_argument('--projection', default=None)
    ap.add_argument('--radii', type=float, nargs='+', default=[1.0, 2.0, 3.0])
    ap.add_argument('--fit', nargs='+', default=_DEFAULT_FITS,
                    help='lensing-fit npz file(s) of lensing/fit_dsigma_ksz.py')
    ap.add_argument('--sample', default='sham-hosts', choices=['sham-hosts', 'masscut'])
    ap.add_argument('--halo-mass-avg', type=float, default=10 ** 13.22,
                    help='mass-cut target mean mass [Msun/h] (masscut sample only)')
    ap.add_argument('--halo-mass-upper', type=float, default=5e14,
                    help='parent / halo mass cap [Msun/h], as the figure configs')
    ap.add_argument('--z', type=float, default=0.5)
    ap.add_argument('--pixel-size', type=float, default=0.5)
    ap.add_argument('--beam', type=float, default=1.6)
    ap.add_argument('--dry-run', action='store_true', help='build in memory, save nothing')
    ap.add_argument('--compare-to', default=None,
                    help='dry-run only: .npy map to compare the first radius against')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    projection = args.projection or _DEFAULT_PROJECTION[args.ptype]
    st = SimulationStacker(args.name, args.snapshot, z=args.z, simType=args.sim_type,
                           feedback=args.feedback)
    sim = {'name': args.name, 'feedback': args.feedback}
    density = None
    if args.sample == 'sham-hosts':
        fitted = load_fitted_abundances([f for f in args.fit if Path(f).exists()])
        label = fit_label(args.sim_type, sim)
        if label not in fitted:
            raise KeyError(f"No lensing fit for {label!r} in {args.fit}; fitted: {sorted(fitted)}")
        density = fitted[label]
    n_pix, _ = map_geometry(st, args.z, args.pixel_size)
    cube = _get_data_filepath(args.sim_type, args.name, args.snapshot, args.feedback, args.ptype,
                              n_pix, projection, 'field', '3D', False, 2.0, st.base_path)
    if not cube.exists():
        raise FileNotFoundError(f"no cached 3D cube {cube}; refusing to rebuild it from the particles")
    hosts, info = mask_hosts(st, args.sample, density, mass_upper=args.halo_mass_upper,
                             mass_avg=args.halo_mass_avg)
    print(f"{args.sim_type} {args.name} {args.feedback or ''} {args.ptype} ({projection}): "
          f"sample={args.sample} n={density} -> {info} ({time.time() - t0:.0f} s)", flush=True)

    for mrad in args.radii:
        n, _ = map_geometry(st, args.z, args.pixel_size)
        final = _get_data_filepath(args.sim_type, args.name, args.snapshot, args.feedback, args.ptype,
                                   n, projection, 'map', '2D', True, mrad, st.base_path)
        if final.exists() and not (args.overwrite or args.dry_run):
            print(f"  exists, skipping: {final}")
            continue
        t1 = time.time()
        m, n = build(st, args.ptype, projection, hosts, mrad, args.z, args.pixel_size, args.beam)
        print(f"  {mrad:g}x R200m built in {time.time() - t1:.0f} s; sum = {m.sum():.6e}", flush=True)
        if args.dry_run:
            if args.compare_to and mrad == args.radii[0]:
                ref = np.load(args.compare_to)
                print(f"  vs {args.compare_to}: max|diff|/max = "
                      f"{np.max(np.abs(m - ref)) / np.max(np.abs(ref)):.3e}")
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(final.stem + '.tmp.npy')
        np.save(tmp, m)
        os.replace(tmp, final)
        meta = dict(sample=args.sample, density=density, fit_files=args.fit, radius_R200m=mrad,
                    centre='loader GroupPos (CAESAR pos for SIMBA)', ptype=args.ptype,
                    projection=projection, pixel_size=args.pixel_size, beam=args.beam, z=args.z,
                    halo_mass_upper=args.halo_mass_upper,
                    halo_mass_avg=args.halo_mass_avg if args.sample == 'masscut' else None,
                    written=datetime.datetime.now().isoformat(timespec='seconds'), **info)
        with open(str(final) + '.sample.json', 'w') as f:
            json.dump(meta, f, indent=1)
        print(f"  saved {final}", flush=True)
    print(f"Done in {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
