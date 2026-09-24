"""compute_stellar_maps.py
========================
Projected maps of the halo-level stellar-to-ionized-gas transfer, for the
lensing observable in the bottom panel of make_pk_stellar.py's S(k) band
figures (unbound gas paper).

The observable is the beam-free simulation curve of
``lensing/beam_compensated_ratio_v2.py``:

    f(theta) = <DSigma[ionized_gas]>(theta) / <DSigma[total]>(theta) * Omega_m/Omega_b ,

a ratio of halo means of the compensated DSigma filter on the cached 2D maps
(0.2 arcmin pixels, no beam). The filter is linear in the map. With the maps
of the ionized-gas mass added (A) and the stellar mass removed (B) by the
transfer at s = 0 (``src/halo_transfer.py``: the same haloes, methods and
mass cuts as compute_pk_stellar.py, applied over the whole box), every
stellar scale s follows exactly (t = 1 - s; N, T = the stacked ionized_gas
and total maps):

    f(s) = [N + t DSigma_A] / [T + t (DSigma_A - DSigma_B)] * Omega_m/Omega_b .

This script builds A and B for every configuration (method variant x mass
cut) on the grid of the cached ionized_gas / total maps, from one pass over
the stars and gas; ``stack_stellar_maps.py`` stacks them. The grid, redshift
and projection come from the lensing config named in the ``lensing`` block of
the config (its redshift, not the P(k) one: the map grid depends on it).
Binning: the cached fields' ``binned_statistic_2d`` (``ht.pixel_index_2d``).

Outputs (new files next to the cached 2D fields; existing ones skipped unless
--overwrite):
  <stem>_stellar_added_<variant>_<tag>_<n>_<proj>.npy     A, Msun/h per pixel (float64)
  <stem>_stellar_removed_<variant>_<tag>_<n>_<proj>.npy   B, Msun/h per pixel (float64)
  <stem>_stellar_maps_<n>_<proj>.npz                      bookkeeping and validation
Validation (in the .npz and the log): sum(A) = sum(B) = moved stellar mass;
per-halo conservation; the moved mass and number of active haloes against the
3D run (``*_Pk_stellar_<variant>_*.npz``); the cached 2D Stars map minus B
(negative mass; only where such a map exists).

The helpers below (lensing settings, file paths, f(s)) are shared with
``stack_stellar_maps.py`` and ``make_pk_stellar.py``.

Run from the scripts/ directory on a whole CPU node (runINT_stellar_maps.sh):
    python unbound_gas/compute_stellar_maps.py -p configs/unbound_gas/pk_components_z05.yaml \
        --sims m100n1024
    # smoke test: first 2 chunks only, nothing saved
    python unbound_gas/compute_stellar_maps.py -p ... --sims TNG300-1 --max-chunks 2
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from compute_pk_stellar import mass_tag, variants_of
from pk_common import (load_config, save_npy_atomic, save_npz_atomic, select_sims,
                       sim_label, spectra_path)

import halo_transfer as ht
from loadIO import _get_data_filepath, resolve_data_root

# Omega_b fallbacks of lensing/beam_compensated_ratio_v2.py (_resolve_stacker),
# used when a snapshot header has no OmegaBaryon. Keep in sync with that script.
_OMEGA_B_FALLBACK = {'IllustrisTNG': 0.0456, 'SIMBA': 0.048, 'FLAMINGO': 0.0486}


# ---------------------------------------------------------------------------
# Lensing settings and file paths (shared with stack_stellar_maps.py and
# make_pk_stellar.py)
# ---------------------------------------------------------------------------

def resolve_stack_settings(nb_stack: dict, overrides=None) -> dict:
    """Stacking settings of the lensing noBeam config, as the lensing script uses them.

    The defaults are those of ``beam_compensated_ratio_v2.py`` (its ``.get``
    fallbacks) and of ``SimulationStacker.stackMap`` for the arguments the
    script does not pass (halo_mass_avg, halo_mass_upper, subtract_mean).

    Args:
        nb_stack: The ``stack`` section of the lensing noBeam config.
        overrides: Optional replacements (same keys), e.g. another
            ``halo_abundance_target``.

    Returns:
        dict: The resolved settings.

    Raises:
        KeyError: If an override is not a known key.
        ValueError: If the settings are not a linear, beam-free DSigma stack
            of ionized_gas over total (what the f(s) algebra assumes).
    """
    g = nb_stack.get
    s = dict(
        redshift=g('redshift', 0.5), rad_distance=g('rad_distance', 1.0),
        particle_type=g('particle_type', 'ionized_gas'), particle_type_2=g('particle_type_2', 'total'),
        filter_type=g('filter_type', 'DSigma'), filter_type_2=g('filter_type_2', 'DSigma'),
        pixel_size=g('pixel_size', 0.2), pixel_size_2=g('pixel_size_2', 0.2),
        beam_size=g('beam_size', None), beam_size_2=g('beam_size_2', None),
        min_radius=g('min_radius', 1.0), max_radius=g('max_radius', 6.0), num_radii=g('num_radii', 9),
        projection=g('projection', 'yz'), mask_haloes=g('mask_haloes', False),
        use_subhalos=g('use_subhalos', False), halo_abundance_target=g('halo_abundance_target', None),
        # stackMap defaults (beam_compensated_ratio_v2.py does not pass these)
        halo_mass_avg=10 ** 13.22, halo_mass_upper=5e14, subtract_mean=False,
    )
    for key, val in (overrides or {}).items():
        if key not in s:
            raise KeyError(f"unknown lensing override {key!r}; known: {sorted(s)}")
        s[key] = val
    # stack_on_array's fallback for a missing abundance target
    if s['halo_abundance_target'] is None:
        s['halo_abundance_target'] = 5e-4
    # PyYAML reads e.g. '1.0e11' (no exponent sign) as a string
    for key in ('redshift', 'rad_distance', 'pixel_size', 'pixel_size_2', 'min_radius',
                'max_radius', 'halo_abundance_target', 'halo_mass_avg'):
        s[key] = float(s[key])
    if s['halo_mass_upper'] is not None:
        s['halo_mass_upper'] = float(s['halo_mass_upper'])
    s['num_radii'] = int(s['num_radii'])
    for b in ('beam_size', 'beam_size_2'):
        if s[b] not in (None, 0, 0.0):
            raise ValueError(f"{b} = {s[b]}: only beam-free stacks are supported")
        s[b] = None
    if (s['particle_type'], s['particle_type_2']) != ('ionized_gas', 'total'):
        raise ValueError("the lensing stack must be ionized_gas over total")
    if s['filter_type'] != 'DSigma' or s['filter_type_2'] != 'DSigma':
        raise ValueError("only the DSigma filter is supported (for both profiles)")
    if s['pixel_size'] != s['pixel_size_2']:
        raise ValueError("both profiles must use the same pixel size")
    if s['mask_haloes'] or s['subtract_mean']:
        raise ValueError("masked or mean-subtracted maps are not supported")
    return s


def lensing_settings(config: dict) -> dict:
    """The ``lensing`` block of a stellar config, with its lensing config read.

    Returns:
        dict: 'config_path', 'config' (parsed lensing noBeam config), 'stack'
        (``resolve_stack_settings``), 'data' (beam-compensated data .npz path
        or None), 'sample_name'.
    """
    lb = config['lensing']
    path = Path(lb['config'])
    lcfg = load_config(str(path))
    return dict(config_path=str(path), config=lcfg,
                stack=resolve_stack_settings(lcfg['stack'], lb.get('overrides')),
                data=lb.get('data'), sample_name=str(lb.get('sample_name', 'lens')))


def lensing_sim(lens: dict, entry: dict):
    """The lensing config's simulation matching a stellar-config entry.

    Returns:
        tuple: (sim dict, redshift used by the lensing script), or (None, None)
        if the simulation is not in the lensing config.
    """
    for grp in lens['config']['simulations']:
        if grp['sim_type'] != entry['sim_type']:
            continue
        for s in grp['sims']:
            if (s['name'] == entry['name'] and int(s['snapshot']) == int(entry['snapshot'])
                    and s.get('feedback') == entry.get('feedback')):
                return s, float(s.get('redshift', lens['stack']['redshift']))
    return None, None


def map_n_pixels(stacker, z: float, pixel_size: float) -> int:
    """Pixels per side of a map at redshift z, exactly as ``SimulationStacker.makeMap``."""
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM
    from utils import comoving_to_arcmin
    cosmo = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'], Om0=stacker.header['Omega0'],
                          Tcmb0=2.7255 * u.K)
    theta_arcmin = comoving_to_arcmin(stacker.header['BoxSize'], z, cosmo=cosmo)
    return int(np.ceil(theta_arcmin / pixel_size))


def field2d_path(entry: dict, p_type: str, n: int, projection: str) -> Path:
    """Path of a 2D field of a stellar-config entry (the cache naming convention)."""
    return _get_data_filepath(entry['sim_type'], entry['name'], entry['snapshot'],
                              entry.get('feedback'), p_type, n, projection=projection,
                              data_type='field', dim='2D')


def map_path(entry: dict, kind: str, variant: str, tag: str, n: int, projection: str) -> Path:
    """Path of a transfer map: kind 'added' (A) or 'removed' (B)."""
    return field2d_path(entry, f"stellar_{kind}_{variant}_{tag}", n, projection)


def maps_diag_path(entry: dict, n: int, projection: str) -> Path:
    """Path of the per-simulation bookkeeping file of the transfer maps."""
    p = field2d_path(entry, 'stellar_maps', n, projection)
    return p.with_suffix('.npz')


def stack_path(entry: dict, sample_name: str) -> Path:
    """Path of the stacked profiles of one halo sample (stack_stellar_maps.py)."""
    if entry.get('feedback'):
        stem = f"{entry['name']}_{entry['feedback']}_{entry['snapshot']}"
    else:
        stem = f"{entry['name']}_{entry['snapshot']}"
    return (Path(resolve_data_root(None)) / entry['sim_type'] / 'products' / '2D'
            / f"{stem}_lensing_stellar_{sample_name}.npz")


def sample_settings(stack: dict, z: float) -> str:
    """Canonical string of everything that sets a stacked profile (stored and compared)."""
    return json.dumps(dict(stack, z_sim=z), sort_keys=True)


def omega_b_of(stacker) -> float:
    """Omega_b as the lensing script resolves it (header, else its fallback)."""
    try:
        return float(stacker.header['OmegaBaryon'])
    except KeyError:
        return _OMEGA_B_FALLBACK[stacker.simType]


def f_of_scale(N, T, A, B, s: float, factor: float):
    """Lensing ratio at stellar scale s: [N + t A] / [T + t (A - B)] * factor, t = 1 - s.

    Args:
        N, T: Halo-mean DSigma of the ionized_gas and total maps.
        A, B: Halo-mean DSigma of the added (ionized gas) and removed (stars)
            maps of one configuration at s = 0.
        s: Stellar scale (1 = simulation).
        factor: Omega_m / Omega_b.
    """
    t = 1.0 - s
    return (N + t * A) / (T + t * (A - B)) * factor


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

def _moved_3d(entry: dict, variant: str) -> dict:
    """{tag: (moved stellar mass, active haloes)} of the 3D run, if its file exists."""
    path = spectra_path(entry, f"stellar_{variant}")
    if not path.exists():
        return {}
    with np.load(path) as f:
        return {str(t): (float(f[f"diag__{t}__mstar_moved"]), int(f[f"diag__{t}__n_haloes_active"]))
                for t in f['tags']}


def _neg_mass_2d(stars2d: np.ndarray, removed: np.ndarray) -> tuple:
    """Sum and pixel count of min(stars2d - removed, 0), row block by row block."""
    neg, npix = 0.0, 0
    for i in range(0, removed.shape[0], 1024):
        d = np.asarray(stars2d[i:i + 1024], dtype=np.float64) - removed[i:i + 1024]
        m = d < 0
        neg += float(d[m].sum())
        npix += int(m.sum())
    return neg, npix


def run_sim(entry: dict, config: dict, only, overwrite: bool, max_chunks) -> None:
    """Transfer maps of every configuration for one simulation."""
    from stacker import SimulationStacker

    lens = lensing_settings(config)
    lsim, z = lensing_sim(lens, entry)
    if lsim is None:
        print(f"  not in the lensing config {lens['config_path']}; skipping")
        return
    proj = lens['stack']['projection']
    st = SimulationStacker(entry['name'], entry['snapshot'], simType=entry['sim_type'],
                           feedback=entry.get('feedback'), z=z)
    box = float(st.header['BoxSize'])
    n = map_n_pixels(st, z, lens['stack']['pixel_size'])
    for p_type in ('ionized_gas', 'total'):
        if not field2d_path(entry, p_type, n, proj).exists():
            raise FileNotFoundError(f"no cached {p_type} map {field2d_path(entry, p_type, n, proj)} "
                                    f"(z = {z}, n = {n}): wrong redshift or pixel size?")
    print(f"  lensing grid: z = {z}, {n}^2 pixels ({proj}), cached ionized_gas/total maps found")

    save = max_chunks is None
    cuts = sorted(float(m) for m in config['stellar']['halo_mass_min'])
    tags = [mass_tag(m) for m in cuts]
    # (variant, tag) configurations to build: all of them with --overwrite or in
    # a smoke test, else only those whose A or B map is missing.
    todo = {(v['name'], t) for v in variants_of(config, only) for t in tags
            if not save or overwrite
            or not all(map_path(entry, k, v['name'], t, n, proj).exists() for k in ('added', 'removed'))}
    variants = [v for v in variants_of(config, only) if any((v['name'], t) in todo for t in tags)]
    if not variants:
        print("  all maps exist, skipping")
        return

    t0 = time.time()
    halos = st.loadHalos()
    gmass = np.asarray(halos['GroupMass'], dtype=np.float64)
    print(f"  {len(gmass):,} haloes; {np.sum(gmass >= cuts[0]):,} above {cuts[0]:.0e} Msun/h "
          f"({time.time() - t0:.0f} s)")

    # ---- one particle pass for every variant, then each kept particle's pixel
    t0 = time.time()
    store = ht.collect_particles(st, variants, halos, cuts[0], max_chunks=max_chunks)
    tot = store['totals']
    print(f"  particle pass in {time.time() - t0:.0f} s; true stars {tot['mstar']:.4e}, "
          f"winds {tot['mwind']:.3e}, ionized gas {tot['mion']:.4e} Msun/h", flush=True)
    t0 = time.time()
    for p_type in ('Stars', 'gas'):
        for blk in store[p_type]:
            blk['pix'] = ht.pixel_index_2d(blk['pos'], n, box, proj)
            for key in ('pos', 'm_gas', 'sf'):  # not needed for the maps
                blk.pop(key, None)
    gc.collect()
    print(f"  pixel indices in {time.time() - t0:.0f} s", flush=True)

    stars_path = field2d_path(entry, 'Stars', n, proj)
    stars2d = np.load(stars_path, mmap_mode='r') if stars_path.exists() else None
    if stars2d is None:
        print("  no cached 2D Stars map: the negative-stellar-mass check is skipped")

    dpath = maps_diag_path(entry, n, proj)
    res = {}
    if save and dpath.exists():
        # keep the numbers of the configurations not rebuilt now; those rebuilt
        # below replace their own diag__ keys
        with np.load(dpath) as old:
            res = {k: old[k] for k in old.files}
    res.update(n_pixels=n, projection=proj, z=z, box=box, lensing_config=lens['config_path'],
               halo_mass_min=np.array(cuts), tags=np.array(tags),
               mstar_box=tot['mstar'], mwind_box=tot['mwind'], mgas_box=tot['mgas'],
               mion_box=tot['mion'], n_chunks_read=store['n_chunks'])
    for v in variants:
        ref = _moved_3d(entry, v['name'])
        for mcut, tag in zip(cuts, tags):
            if (v['name'], tag) not in todo:
                print(f"  [{v['name']} {tag}] maps exist, skipping")
                continue
            t1 = time.time()
            A, B, diag = ht.transfer_maps_2d(store, v['name'], gmass, mcut, n)
            diag['min_added'] = float(A.min())
            diag['min_removed'] = float(B.min())
            if stars2d is not None:
                diag['neg_star_mass'], diag['neg_star_pixels'] = _neg_mass_2d(stars2d, B)
            if tag in ref and max_chunks is None:
                m3, n3 = ref[tag]
                diag['moved_vs_3d'] = (diag['mstar_moved'] - m3) / m3 if m3 > 0 else 0.0
                diag['active_vs_3d'] = diag['n_haloes_active'] - n3
            if save:
                save_npy_atomic(map_path(entry, 'added', v['name'], tag, n, proj), A)
                save_npy_atomic(map_path(entry, 'removed', v['name'], tag, n, proj), B)
            del A, B
            for key, val in diag.items():
                res[f"diag__{v['name']}__{tag}__{key}"] = val
            moved = max(diag['mstar_moved'], 1e-30)
            print(f"  [{v['name']} {tag}] {time.time() - t1:.0f} s: {diag['n_haloes_active']:,} haloes, "
                  f"moved {diag['mstar_moved'] / tot['mstar']:.3f} of the stars; "
                  f"sum A / moved - 1 = {diag['sum_added_rel']:.1e}, sum B / moved - 1 = "
                  f"{diag['sum_removed_rel']:.1e}; per-halo {diag['max_halo_cons']:.1e}; "
                  f"neg. stars {diag.get('neg_star_mass', 0.0) / moved:.1e}; "
                  f"vs 3D run: moved {diag.get('moved_vs_3d', np.nan):.1e}, "
                  f"active {diag.get('active_vs_3d', 'n/a')}", flush=True)
            gc.collect()
    if save:
        save_npz_atomic(dpath, **res)
    del store
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
    if 'lensing' not in config:
        raise SystemExit(f"{args.path2config} has no 'lensing' block")
    for entry in select_sims(config, args.sims):
        print(f"\n===== {sim_label(entry)} =====", flush=True)
        run_sim(entry, config, args.variants, args.overwrite, args.max_chunks)
        gc.collect()


if __name__ == '__main__':
    main()
