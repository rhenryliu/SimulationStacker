"""test_simba_centres_masked.py

Test of the SIMBA halo centre in the masked DeltaSigma-ratio figures
(simulated_dsigma_ratio_masked.py with the particle-masked ionized-gas fields
of precompute_masked_particles.py).

SIMBA's halo centre in the loader is CAESAR ``halo_data/pos``, a centre of
mass (user decision 2026-09-29: keep it). SIMBA-100 shows a much larger
fraction of its signal from outside 1 R200m at small apertures than the other
simulations. This test asks whether that comes from the masks being centred
on ``pos`` rather than on the central galaxy, by also centring them on CAESAR
``halo_data/minpotpos`` (the potential minimum, on the central galaxy).

One pass over the gas of one SIMBA run builds the 1, 2, 3 x R200m particle
masks around the unique hosts of the SHAM galaxies at the given density,
centred (a) on ``pos``, exactly as precompute_masked_particles.py, and (b) on
``minpotpos``; both are binned like the cached unbeamed fields and stacked
exactly as simulated_dsigma_ratio_masked.py does. (a) must reproduce the
masked profiles of the figure's profiles file (``--profiles``), which checks
the test itself. Also reports the host offsets |pos - minpotpos| / R200m.
Nothing is written to the products cache.

Outputs, in ../figures/YYYY-MM/MM-DD/: test_simba_centres_<run>_n<density>.npz
(radii, outside fractions per centre and radius, offsets) and the printed table.

Usage (from the scripts/ directory; one SIMBA-100 pass takes a few minutes, so
run it on a compute node):
    python unbound_gas/test_simba_centres_masked.py --profiles \\
        ../figures/2026-10/10-12/dsigma_ratio_yz_masked_nobeam_n5e-4_z0.5_profiles.npz
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import binned_statistic_2d

sys.path.append('../src/')
from loadIO import load_subset, load_as_dict, _get_data_filepath  # type: ignore
from mapMaker import ionized_gas_keys, ionized_gas_masses  # type: ignore
from halo_transfer import ApertureLabeller, _chunks  # type: ignore
# Sibling modules in this directory; after the '../src/' append above.
from halo_stats import map_geometry, select_sample
from precompute_masked_sham_hosts import mask_hosts
from simulated_dsigma_ratio_masked import make_stacker, _stack

_AXES = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}  # map axes, as make_mass_field


def main() -> None:
    """Build the two sets of masked maps of one SIMBA run, stack them and report."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--name', default='m100n1024')
    ap.add_argument('--snapshot', type=int, default=125)
    ap.add_argument('--feedback', default='s50')
    ap.add_argument('--z', type=float, default=0.5)
    ap.add_argument('--density', type=float, default=5e-4, help='SHAM density [(cMpc/h)^-3]')
    ap.add_argument('--radii', type=float, nargs='+', default=[1.0, 2.0, 3.0])
    ap.add_argument('--profiles', required=True,
                    help="the figure's profiles npz with this run at this density (the check of (a))")
    args = ap.parse_args()

    t0 = time.time()
    sim = {'name': args.name, 'snapshot': args.snapshot, 'feedback': args.feedback}
    key = f'SIMBA/{args.name}/{args.feedback}'
    st = make_stacker('SIMBA', sim, args.z)[0]
    box = float(st.header['BoxSize'])
    hosts, info = mask_hosts(st, 'sham-hosts', args.density, mass_upper=5e14)
    h = st.loadHalos()
    cat = load_as_dict(f'{st.simPath}catalogs/{st.sim}_{st.snapshot}.hdf5', 'halo_data')
    # minpotpos in the units of the loader's GroupPos (CAESAR pos x h, kpc/h).
    centres = {'pos': np.asarray(h['GroupPos'], dtype=np.float64),
               'minpot': np.asarray(cat['minpotpos'], dtype=np.float64) * st.header['HubbleParam']}
    rad = np.asarray(h['GroupRad'], dtype=np.float64)
    d = centres['pos'][hosts] - centres['minpot'][hosts]
    d -= box * np.round(d / box)
    off = np.linalg.norm(d, axis=1) / rad[hosts]
    print(f"SIMBA {args.name} {args.feedback}, n = {args.density}: {info}; host offsets "
          f"|pos - minpotpos| / R200m: median {np.median(off):.3f}, 16-84% "
          f"{np.percentile(off, 16):.3f}-{np.percentile(off, 84):.3f}, fraction > 0.5: "
          f"{np.mean(off > 0.5):.3f} ({time.time() - t0:.0f} s)", flush=True)

    # Particle pass, as precompute_masked_particles.py (binning from the
    # coordinates as loaded, labelling on positions wrapped into [0, box)).
    n_pix, _ = map_geometry(st, args.z, 0.2)
    labellers = {(c, r): ApertureLabeller(centres[c], r * rad, h['GroupMass'], box, rows=hosts)
                 for c in centres for r in args.radii}
    n_flat = (n_pix + 2) ** 2
    flat = {k: np.zeros(n_flat) for k in labellers}
    ia, ib = _AXES['yz']
    keys = ['Coordinates'] + ionized_gas_keys(st.simType)
    for ic, (_, path) in enumerate(_chunks(st)):
        part = load_subset(st.simPath, st.snapshot, st.simType, 'gas', snap_path=path,
                           header=st.header, sim_name=st.sim, keys=keys)
        co = part['Coordinates']
        if len(co) == 0:
            continue
        mass = ionized_gas_masses(part, st)  # Msun/h
        binnumber = binned_statistic_2d(co[:, ia], co[:, ib], mass, 'count', bins=[n_pix, n_pix],
                                        range=[[0, box], [0, box]]).binnumber
        pos = np.mod(np.asarray(co, dtype=np.float64), box).astype(np.float32)
        for k, lab in labellers.items():
            keep = lab.labels(pos) >= 0
            flat[k] += np.bincount(binnumber[keep], weights=mass[keep], minlength=n_flat)
        print(f"  chunk {ic + 1}: {len(co):.3e} gas particles, {time.time() - t0:.0f} s", flush=True)
    maps = {k: v.reshape(n_pix + 2, n_pix + 2)[1:-1, 1:-1] for k, v in flat.items()}

    # Stacks, as simulated_dsigma_ratio_masked.stack_run (the denominator
    # cancels in the outside fraction, so only the numerators are needed).
    p = dict(min_radius=1.0, max_radius=6.0, num_radii=9, projection='yz', annulus=0.75,
             halo_mass_avg=10 ** 13.22, halo_mass_upper=5e14, redshift=args.z)
    halo_mask = select_sample(st, use_subhalos=True, halo_abundance_target=args.density,
                              halo_mass_avg=p['halo_mass_avg'], halo_mass_upper=p['halo_mass_upper'])
    ion = np.load(_get_data_filepath('SIMBA', st.sim, st.snapshot, st.feedback, 'ionized_gas', n_pix,
                                     'yz', 'field', '2D', False, 2.0, st.base_path))
    radii, unm = _stack(st, ion, halo_mask, p, 0.2)
    unm_mean = unm.mean(axis=1)
    with np.load(args.profiles) as ref:
        ref = {k: ref[k] for k in ref.files if k.startswith(key + '/')}
    print(f"\nradii [arcmin] {np.array2string(radii, precision=3)}; {len(halo_mask)} galaxies")
    print(f"(a) check: unmasked numerator vs {args.profiles}: max relative difference "
          f"{np.max(np.abs(unm_mean / ref[key + '/unmasked_num_mean'] - 1)):.1e}")
    out = {'radii_arcmin': radii, 'offsets_R200m': off, 'unmasked_num_mean': unm_mean}
    for (c, r), m in maps.items():
        _, num = _stack(st, m, halo_mask, p, 0.2)
        num_mean = num.mean(axis=1)
        out[f'{c}/mask{r:.0f}_num_mean'] = num_mean
        out[f'{c}/out{r:.0f}'] = 1 - num_mean / unm_mean
        check = ''
        if c == 'pos':
            check = (f"   (a) check vs the profiles file: max relative difference "
                     f"{np.max(np.abs(num_mean / ref[f'{key}/mask{r:.0f}_num_mean'] - 1)):.1e}")
        print(f"out{r:.0f}  centre {c:6s} {np.array2string(1 - num_mean / unm_mean, precision=3)}{check}",
              flush=True)

    now = datetime.now()
    fig_dir = Path('../figures') / now.strftime('%Y-%m') / now.strftime('%m-%d')
    fig_dir.mkdir(parents=True, exist_ok=True)
    npz = fig_dir / f'test_simba_centres_{args.name}_{args.feedback}_n{args.density:g}.npz'
    np.savez(npz, **out)
    print(f"Saved {npz}; done in {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
