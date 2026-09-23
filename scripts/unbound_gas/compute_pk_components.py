"""compute_pk_components.py
==========================
Auto and cross power spectra of the five mass components (DM, ionized_gas,
neutral_gas, Stars, BH; see ``pk_common``) of each hydro simulation in
``configs/unbound_gas/pk_components_z05.yaml``, plus the spectra of the total
matter field and of the DMO reference run.

Because P_mm is bilinear in the component weights, these 15 spectra give the
matter power spectrum exactly for any linear redistribution of mass between
the components -- e.g. every alpha in the model
rho_b(alpha) = rhobar_b [alpha u_ion + (1 - alpha) u_else] -- without
rebuilding a field (``make_pk_alpha.py`` does that algebra).

Outputs (new files next to the 3D caches; existing outputs are skipped unless
--overwrite):

  <stem>_Pk_components_<n>.npz
      k [h/Mpc], Nmodes, components (names), means (box-mean density of each
      component, M_sun/h per voxel), P (5 x 5 x Nk, symmetric, (Mpc/h)^3),
      P_total (spectrum of the cached total field; must equal
      w^T P w with w = means / sum(means)), box_mpc, n_pixels.
  <stem>_Pk_dmo_<n>.npz  (only when the DMO field exists)
      k, Nmodes, P_total, P_dmo, P_total_dmo (cross), box_mpc, n_pixels.

All spectra use Pylians with TSC deconvolution (matching the TSC-binned
fields), monopole only, truncated at the grid Nyquist frequency.

Run from the scripts/ directory on a whole CPU node:
    python unbound_gas/compute_pk_components.py -p configs/unbound_gas/pk_components_z05.yaml \
        --sims TNG300-1
"""

import argparse
import gc
import time

import numpy as np

from pk_common import (COMPONENTS, box_size_mpc, load_config, load_dmo_field,
                       load_field, save_npz_atomic, select_sims, sim_label,
                       spectra_path, to_overdensity)

import Pk_library as PKL


def build_components_fresh(entry: dict) -> list:
    """Bin the five component fields from the particles, ignoring the caches.

    Validation mode (--fresh): DM is binned directly instead of derived as
    total - gas - Stars - BH, so comparing its spectra with the cache-based
    ones tests the caches where it matters. Nothing is saved to the field
    cache. Uses the current mapMaker code (float32 accumulation).
    """
    from stacker import SimulationStacker
    from mapMaker import create_field
    st = SimulationStacker(entry['name'], entry['snapshot'], simType=entry['sim_type'],
                           feedback=entry.get('feedback'), z=entry.get('redshift', 0.5))
    n = entry['n_pixels']
    out = {}
    for p_type in ('DM', 'gas', 'ionized_gas', 'Stars', 'BH'):
        t0 = time.time()
        out[p_type] = create_field(st, p_type, n, 'xy', dim='3D', load=False).astype(np.float32)
        print(f"  fresh {p_type} binned in {time.time() - t0:.0f} s")
    neutral = out['gas'] - out['ionized_gas']
    del out['gas']
    return [out['DM'], out['ionized_gas'], neutral, out['Stars'], out['BH']]


def build_components(entry: dict) -> list:
    """Assemble the five component fields (float32), in COMPONENTS order.

    Peak memory is five full grids: the total field is turned into DM in place
    and the gas field is released once DM and neutral gas no longer need it.
    """
    gas = load_field(entry, 'gas')
    ion = load_field(entry, 'ionized_gas')
    neutral = gas - ion
    dm = load_field(entry, 'total')
    dm -= gas
    del gas
    stars = load_field(entry, 'Stars')
    dm -= stars
    bh = load_field(entry, 'BH')
    dm -= bh
    return [dm, ion, neutral, stars, bh]


def nyquist_cut(k: np.ndarray, box_mpc: float, n: int) -> np.ndarray:
    """Boolean mask selecting k <= k_Nyquist = pi n / L."""
    return k <= np.pi * n / box_mpc


def component_spectra(entry: dict, threads: int, fresh: bool = False) -> dict:
    """Compute the component auto/cross spectra and the total spectrum.

    Args:
        entry: Simulation entry.
        threads: Pylians threads.
        fresh: Bin the components from particles (validation mode) instead of
            assembling them from the cached fields; P_total is then the
            spectrum of their sum.
    """
    box = box_size_mpc(entry)
    n = entry['n_pixels']

    t0 = time.time()
    fields = build_components_fresh(entry) if fresh else build_components(entry)
    total_fresh = np.sum(fields, axis=0, dtype=np.float32) if fresh else None
    means = np.array([to_overdensity(f) for f in fields])
    print(f"  components assembled in {time.time() - t0:.0f} s; box means "
          + ", ".join(f"{c}={m:.4e}" for c, m in zip(COMPONENTS, means)))
    if np.any(means <= 0):
        raise ValueError(f"non-positive component mean: {dict(zip(COMPONENTS, means))}")

    t0 = time.time()
    X = PKL.XPk(fields, box, axis=0, MAS=['TSC'] * len(fields), threads=threads)
    print(f"  XPk of {len(fields)} fields in {time.time() - t0:.0f} s")
    sel = nyquist_cut(X.k3D, box, n)
    nc = len(COMPONENTS)
    P = np.zeros((nc, nc, sel.sum()))
    pair = 0
    for i in range(nc):
        P[i, i] = X.Pk[sel, 0, i]
        for j in range(i + 1, nc):  # Pylians orders pairs (0,1), (0,2), ..., (1,2), ...
            P[i, j] = P[j, i] = X.XPk[sel, 0, pair]
            pair += 1
    k, nmodes = X.k3D[sel], X.Nmodes3D[sel]
    del X, fields
    gc.collect()

    t0 = time.time()
    tot = total_fresh if fresh else load_field(entry, 'total')
    to_overdensity(tot)
    Pt = PKL.Pk(tot, box, axis=0, MAS='TSC', threads=threads, verbose=False)
    P_total = Pt.Pk[nyquist_cut(Pt.k3D, box, n), 0]
    del tot, Pt
    gc.collect()
    print(f"  P_total in {time.time() - t0:.0f} s")

    # Algebra check: the mass-weighted component sum must reproduce P_total.
    w = means / means.sum()
    P_sum = np.einsum('i,j,ijk->k', w, w, P)
    dev = float(np.max(np.abs(P_sum / P_total - 1.0)))
    print(f"  check: max |w^T P w / P_total - 1| = {dev:.2e}")
    if dev > 1e-4:
        raise RuntimeError(f"component sum does not reproduce P_total (max dev {dev:.2e})")

    return dict(k=k, Nmodes=nmodes, components=np.array(COMPONENTS), means=means,
                P=P, P_total=P_total, box_mpc=box, n_pixels=n)


def dmo_spectra(entry: dict, threads: int) -> dict:
    """Compute P_total, P_DMO and their cross spectrum."""
    box = box_size_mpc(entry)
    n = entry['n_pixels']
    t0 = time.time()
    dmo = load_dmo_field(entry)
    to_overdensity(dmo)
    tot = load_field(entry, 'total')
    to_overdensity(tot)
    X = PKL.XPk([tot, dmo], box, axis=0, MAS=['TSC', 'TSC'], threads=threads)
    sel = nyquist_cut(X.k3D, box, n)
    out = dict(k=X.k3D[sel], Nmodes=X.Nmodes3D[sel], P_total=X.Pk[sel, 0, 0],
               P_dmo=X.Pk[sel, 0, 1], P_total_dmo=X.XPk[sel, 0, 0],
               box_mpc=box, n_pixels=n)
    del X, tot, dmo
    gc.collect()
    r = out['P_total_dmo'] / np.sqrt(out['P_total'] * out['P_dmo'])
    print(f"  DMO spectra in {time.time() - t0:.0f} s; cross-correlation "
          f"r(k) at the lowest 3 k: {np.round(r[:3], 5).tolist()}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None,
                        help="restrict to these entries (label, name or feedback)")
    parser.add_argument('--what', nargs='+', default=['components', 'dmo'],
                        choices=['components', 'dmo'])
    parser.add_argument('--overwrite', action='store_true',
                        help="recompute spectra files that already exist")
    parser.add_argument('--fresh', action='store_true',
                        help="validation: bin the components from particles instead of the "
                             "caches (DM binned directly); writes *_Pk_components_fresh_*.npz")
    args = parser.parse_args()

    config = load_config(args.path2config)
    threads = int(config['pk'].get('threads', 1))

    for entry in select_sims(config, args.sims):
        print(f"\n===== {sim_label(entry)} (grid {entry['n_pixels']}^3) =====")
        if 'components' in args.what:
            path = spectra_path(entry, 'components_fresh' if args.fresh else 'components')
            if path.exists() and not args.overwrite:
                print(f"  exists, skipping: {path}")
            else:
                save_npz_atomic(path, **component_spectra(entry, threads, fresh=args.fresh))
        if 'dmo' in args.what:
            path = spectra_path(entry, 'dmo')
            if path.exists() and not args.overwrite:
                print(f"  exists, skipping: {path}")
            else:
                try:
                    save_npz_atomic(path, **dmo_spectra(entry, threads))
                except FileNotFoundError as e:
                    print(f"  DMO spectra skipped: {e}")


if __name__ == '__main__':
    main()
