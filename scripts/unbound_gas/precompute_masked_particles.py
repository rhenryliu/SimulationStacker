"""precompute_masked_particles.py

Build masked 2D ionized-gas maps by masking the gas particles themselves: keep
the particles within n x R200m of the host haloes of the SHAM galaxies (exact
periodic spheres, ``halo_transfer.ApertureLabeller``), and bin
them in 2D exactly as the cached unmasked fields are binned
(``mapMaker.make_mass_field``: ``ionized_gas_masses`` weights,
``binned_statistic_2d`` sums over [0, BoxSize)). All radii come from one pass
over the gas.

This is the route for maps whose 3D cube would not fit: the cube-based masked
maps (precompute_masked_sham_hosts.py) need a cube at the map's resolution,
which for FLAMINGO at 0.2' is 8869^3. It also shares its binning with the
unmasked 2D fields, so ``--keep-all`` (no mask) reproduces the cached unmasked
field, which the script checks.

SHAM density: by default the simulation's lensing-fitted density, read from
the ``--fit`` npz files of lensing/fit_dsigma_ksz.py (the three z = 0.5 fits if
``--fit`` is not given); or a fixed density for every simulation with
``--density`` (e.g. 5e-4 for the DESI LRGs at z = 0.5, 1e-3 for the BGS at
z ~ 0.26, as in Liu et al. 2026). The two options exclude each other.

Differences from the cube-based masks: the spheres are exact (the cube masks
round the host centres to a cell and keep whole cells), and particles are
binned to the nearest pixel rather than spread by TSC.

Output, for the default no-beam maps: the standard 2D masked-field path
(``..._ionized_gas_<n>_<proj>_masked<r>R200c.npy``; the "R200c" is historical)
with a ``.sample.json`` record of the sample, each written via a temporary
file, the record first; a map counts as done only with its record. As with
the tau maps, the path does not encode the sample: the record holds the SHAM
density and where it came from (``density_source``: 'fit' or 'fixed'), and the
stacking script checks it. Beam-convolved maps (tests) are only written to
``--out-dir``, never to the cache.

Keep-all check: with ``--keep-all`` and a complete pass, a keep-all map that
differs from the cached unmasked map by more than 1e-10 (relative) stops the
script with exit code 1 before anything is saved.

Usage (from the scripts/ directory):
    python unbound_gas/precompute_masked_particles.py --sim-type SIMBA --name m100n1024 \\
        --snapshot 125 --feedback s50
    # at a fixed SHAM density instead of the lensing fit:
    python unbound_gas/precompute_masked_particles.py ... --density 5e-4
    # check against the cached unmasked field, save nothing:
    python unbound_gas/precompute_masked_particles.py ... --keep-all --dry-run
    # test maps on the 0.5' beamed grid, written elsewhere:
    python unbound_gas/precompute_masked_particles.py ... --pixel-size 0.5 --beam 1.6 \\
        --keep-all --out-dir /pscratch/sd/r/rhliu/claude_jobs/particle_mask_test
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binned_statistic_2d

sys.path.append('../src/')
from stacker import SimulationStacker  # type: ignore
from loadIO import _get_data_filepath, load_subset  # type: ignore
from mapMaker import ionized_gas_keys, ionized_gas_masses  # type: ignore
from halo_transfer import ApertureLabeller, _chunks  # type: ignore
from utils import fft_smoothed_map  # type: ignore
# Sibling modules in this directory; after the '../src/' append above.
from halo_stats import fit_label, load_fitted_abundances, map_geometry
from precompute_masked_sham_hosts import mask_hosts

_DEFAULT_FITS = ['../data/fit_dsigma_ksz/fit_dsigma_ksz_z05.npz',
                 '../data/fit_dsigma_ksz/fit_dsigma_ksz_z05_simba50.npz',
                 '../data/fit_dsigma_ksz/fit_dsigma_ksz_z05_mstar.npz']
_AXES = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}  # map axes, as make_mass_field
_KEEP_ALL_RTOL = 1e-10  # keep-all vs cached map (same particles; summation order only)


def _done(path, density):
    """A map is done only with its sample record (a crash between the two leaves neither usable).

    The path does not encode the sample, so a finished map built at another
    SHAM density is an error rather than a map to skip.
    """
    record = Path(str(path) + '.sample.json')
    if not (path.exists() and record.exists()):
        return False
    meta = json.loads(record.read_text())
    if not np.isclose(meta.get('density', np.nan), density):
        raise ValueError(f"{path} was built at n = {meta.get('density')} "
                         f"({meta.get('density_source', 'fit')}); this run is at n = {density}: "
                         f"move the old map aside first, or pass --overwrite")
    return True


def main() -> None:
    """Parse the arguments, build the masked maps of one simulation and save them."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--sim-type', required=True, choices=['IllustrisTNG', 'SIMBA', 'FLAMINGO'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--snapshot', type=int, required=True)
    ap.add_argument('--feedback', default=None)
    ap.add_argument('--projection', default='yz', choices=list(_AXES))
    ap.add_argument('--radii', type=float, nargs='+', default=[1.0, 2.0, 3.0])
    sample = ap.add_mutually_exclusive_group()
    sample.add_argument('--fit', nargs='+', default=None,
                        help='lensing-fit npz file(s) of lensing/fit_dsigma_ksz.py '
                             '(default: the three z = 0.5 fits)')
    sample.add_argument('--density', type=float, default=None,
                        help='fixed SHAM number density [(cMpc/h)^-3] instead of a lensing fit')
    ap.add_argument('--halo-mass-upper', type=float, default=5e14,
                    help='SHAM parent-mass cap [Msun/h], as the figure configs')
    ap.add_argument('--z', type=float, default=0.5)
    ap.add_argument('--pixel-size', type=float, default=0.2, help='arcmin')
    ap.add_argument('--beam', type=float, default=0.0, help="FWHM in arcmin; > 0 needs --out-dir")
    ap.add_argument('--keep-all', action='store_true',
                    help='also bin every particle and compare with the cached unmasked map')
    ap.add_argument('--out-dir', default=None, help='write here instead of the cache (tests)')
    ap.add_argument('--max-chunks', type=int, default=None,
                    help='read only the first chunks, for timing; saves nothing')
    ap.add_argument('--dry-run', action='store_true', help='build in memory, save nothing')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.beam > 0 and args.out_dir is None and not args.dry_run:
        ap.error('beam-convolved maps are test products: give --out-dir or --dry-run')
    if args.density is not None and not (np.isfinite(args.density) and args.density > 0):
        ap.error(f'--density must be positive and finite, got {args.density}')
    save = not (args.dry_run or args.max_chunks is not None)

    t0 = time.time()
    st = SimulationStacker(args.name, args.snapshot, z=args.z, simType=args.sim_type,
                           feedback=args.feedback)
    if args.density is not None:
        density, density_source, fit_files = args.density, 'fixed', []
    else:
        # The default list may name fits that do not exist yet; explicit ones must.
        fit_files = ([f for f in _DEFAULT_FITS if Path(f).exists()] if args.fit is None
                     else args.fit)
        fitted = load_fitted_abundances(fit_files)
        label = fit_label(args.sim_type, {'name': args.name, 'feedback': args.feedback})
        if label not in fitted:
            raise KeyError(f"No lensing fit for {label!r} in {fit_files}; fitted: {sorted(fitted)}")
        density, density_source = fitted[label], 'fit'
    n_pix, arcpp = map_geometry(st, args.z, args.pixel_size)
    box = float(st.header['BoxSize'])

    def out_path(mask_rad):
        """Where the map of one radius (None: no mask) goes."""
        if args.out_dir is not None:
            tag = 'all' if mask_rad is None else f'masked{mask_rad:g}'
            stem = _get_data_filepath(args.sim_type, args.name, args.snapshot, args.feedback,
                                      'ionized_gas', n_pix, args.projection, 'field', '2D',
                                      False, 2.0, st.base_path).stem
            return Path(args.out_dir) / f'{stem}_px{args.pixel_size:g}_beam{args.beam:g}_{tag}.npy'
        return _get_data_filepath(args.sim_type, args.name, args.snapshot, args.feedback,
                                  'ionized_gas', n_pix, args.projection, 'field', '2D',
                                  True, mask_rad, st.base_path)

    radii = [r for r in args.radii
             if args.overwrite or not save or not _done(out_path(r), density)]
    for r in sorted(set(args.radii) - set(radii)):
        print(f"  exists, skipping: {out_path(r)}")
    if not radii and not args.keep_all:
        print("Nothing to do")
        return

    hosts, info = mask_hosts(st, 'sham-hosts', density, mass_upper=args.halo_mass_upper)
    h = st.loadHalos()
    labellers = {r: ApertureLabeller(h['GroupPos'], r * np.asarray(h['GroupRad'], dtype=np.float64),
                                     h['GroupMass'], box, rows=hosts) for r in radii}
    print(f"{args.sim_type} {args.name} {args.feedback or ''} ionized_gas ({args.projection}, "
          f"{n_pix}^2 at {args.pixel_size}', beam {args.beam}'): n={density} ({density_source}) -> {info} "
          f"({time.time() - t0:.0f} s)", flush=True)

    # Bin numbers once per chunk (binned_statistic_2d's own 'sum' is a bincount
    # over them, so the keep-all map reproduces make_mass_field), then one
    # bincount per radius over the kept particles.
    n_flat = (n_pix + 2) ** 2  # binned_statistic's grid, with an outlier row/column each side
    flat = {r: np.zeros(n_flat) for r in radii}
    if args.keep_all:
        flat[None] = np.zeros(n_flat)
    ia, ib = _AXES[args.projection]
    chunks = _chunks(st)
    if args.max_chunks is not None:
        chunks = chunks[:args.max_chunks]
    keys = ['Coordinates'] + ionized_gas_keys(st.simType)
    n_part = 0
    for ic, (_, path) in enumerate(chunks):
        part = load_subset(st.simPath, st.snapshot, st.simType, 'gas', snap_path=path,
                           header=st.header, sim_name=st.sim, keys=keys)
        co = part['Coordinates']
        if len(co) == 0:
            continue
        mass = ionized_gas_masses(part, st)  # Msun/h
        binnumber = binned_statistic_2d(co[:, ia], co[:, ib], mass, 'count',
                                        bins=[n_pix, n_pix], range=[[0, box], [0, box]]).binnumber
        # The labeller wants positions wrapped into [0, box); the binning above
        # uses the coordinates as loaded, as make_mass_field does.
        pos = np.mod(np.asarray(co, dtype=np.float64), box).astype(np.float32)
        for r, lab in labellers.items():
            keep = lab.labels(pos) >= 0
            flat[r] += np.bincount(binnumber[keep], weights=mass[keep], minlength=n_flat)
        if args.keep_all:
            flat[None] += np.bincount(binnumber, weights=mass, minlength=n_flat)
        n_part += len(co)
        del part, co, mass, binnumber, pos
        if ic % 10 == 0 or ic == len(chunks) - 1:
            print(f"  chunk {ic + 1}/{len(chunks)}: {n_part:.3e} gas particles, "
                  f"{time.time() - t0:.0f} s", flush=True)

    maps = {}
    for r, arr in flat.items():
        m2d = arr.reshape(n_pix + 2, n_pix + 2)[1:-1, 1:-1]
        maps[r] = fft_smoothed_map(m2d, args.beam, pixel_size_arcmin=arcpp) if args.beam > 0 else m2d

    if args.keep_all and args.max_chunks is None:
        # Cached unmasked map of the same geometry: the field without a beam,
        # the map with one (that path does not record the beam).
        ref_path = _get_data_filepath(args.sim_type, args.name, args.snapshot, args.feedback,
                                      'ionized_gas', n_pix, args.projection,
                                      'map' if args.beam > 0 else 'field', '2D', False, 2.0,
                                      st.base_path)
        if ref_path.exists():
            ref = np.load(ref_path)
            d_sum = maps[None].sum() / ref.sum() - 1
            d_max = np.max(np.abs(maps[None] - ref)) / np.max(np.abs(ref))
            print(f"  [keep-all check] vs {ref_path.name}: sum ratio - 1 = {d_sum:.2e}, "
                  f"max|diff|/max = {d_max:.2e}", flush=True)
            if max(abs(d_sum), d_max) > _KEEP_ALL_RTOL:
                print("  [keep-all check] FAILED: not saving", flush=True)
                sys.exit(1)
        else:
            print(f"  [keep-all check] no cached {ref_path.name}")
    for r in radii:
        print(f"  {r:g}x R200m: sum = {maps[r].sum():.6e}"
              + (f" ({maps[r].sum() / maps[None].sum():.4f} of all)" if None in maps else ''))

    if not save:
        print(f"Saved nothing ({'max-chunks' if args.max_chunks is not None else 'dry run'}); "
              f"done in {time.time() - t0:.0f} s")
        return
    targets = list(radii) + ([None] if args.keep_all and args.out_dir is not None else [])
    for r in targets:
        final = out_path(r)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(final.stem + '.tmp.npy')
        np.save(tmp, maps[r])
        if r is not None:
            # Record first, each via a temporary file: a map is done only with
            # its record, so a crash leaves at worst a record to overwrite.
            meta = dict(sample='sham-hosts', density=density, density_source=density_source,
                        fit_files=fit_files, radius_R200m=r,
                        method='particles (exact periodic spheres, nearest-pixel binning)',
                        centre='loader GroupPos (CAESAR pos for SIMBA)', ptype='ionized_gas',
                        projection=args.projection, pixel_size=args.pixel_size, beam=args.beam,
                        z=args.z, n_pixels=n_pix, halo_mass_upper=args.halo_mass_upper,
                        n_gas_particles=n_part,
                        written=datetime.datetime.now().isoformat(timespec='seconds'), **info)
            side = Path(str(final) + '.sample.json')
            side_tmp = side.with_name(side.name + '.tmp')
            side_tmp.write_text(json.dumps(meta, indent=1))
            os.replace(side_tmp, side)
        os.replace(tmp, final)
        print(f"  saved {final}", flush=True)
    print(f"Done in {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
