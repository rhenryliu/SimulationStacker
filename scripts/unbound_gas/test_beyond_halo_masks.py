"""test_beyond_halo_masks.py

Two exploratory tests of the kSZ masking analysis of the unbound gas paper
(user request 2026-09-29: figures folder only, not in the paper yet). Both
stack the CAP-filtered tau maps (xy, 0.5 arcmin, 1.6 arcmin beam) around the
lensing-fit SHAM galaxies of the kSZ masking figure, and build every masked
map in memory from the cached 3D tau cube (nothing is written to the products
cache).

``diffuse``: where the gas outside the stacked hosts sits. Besides the
unmasked map and the map keeping only the gas within 1 x R200m of the stacked
galaxies' hosts (the kSZ figure's first column), keep the gas within
1 x R200m of ALL haloes above M_FoF >= 1e12 (and 1e11) Msun/h. Then
    S_U - S_hosts : gas outside the stacked hosts;
    S_all - S_hosts: of which in other haloes above the threshold;
    S_U - S_all   : of which outside every such halo (diffuse, or in smaller
                    haloes).

``los``: how much of the signal comes from far along the line of sight.
Project the (unmasked, and 1 x R200m host-masked) cube through slabs of depth
D along z instead of the whole box; each galaxy is stacked on the slab whose
centre is nearest to it (slabs overlap by half, so a galaxy is >= D/4 from the
slab edges). A tau map carries no velocity weighting, so the full-depth
profile includes correlated gas far along the line of sight that a
velocity-weighted kSZ stack would partly average out.

Outputs (figures/YYYY-MM/MM-DD/): test_<mode>_<sim>.npz and test_<mode>.pdf.

Usage (from scripts/):
    python unbound_gas/test_beyond_halo_masks.py diffuse --sims TNG100-1 Illustris-1
    python unbound_gas/test_beyond_halo_masks.py los --sims TNG300-1 --depths 50000 100000
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('../src/')
from stacker import SimulationStacker  # type: ignore
from loadIO import _get_data_filepath  # type: ignore
from mask_utils import get_cutout_mask_3d  # type: ignore
from utils import fft_smoothed_map  # type: ignore
# Sibling module in this directory; after the '../src/' append above.
from halo_stats import fit_label, load_fitted_abundances, map_geometry, select_sample

SIMS = {
    'TNG100-1': ('IllustrisTNG', 'TNG100-1', 67, None),
    'TNG300-1': ('IllustrisTNG', 'TNG300-1', 67, None),
    'Illustris-1': ('IllustrisTNG', 'Illustris-1', 103, None),
    'SIMBA-100': ('SIMBA', 'm100n1024', 125, 's50'),
    'FLAMINGO_L1_m9': ('FLAMINGO', 'L1_m9', 67, 'L1_m9'),
    'FLAMINGO_fgas-8sigma': ('FLAMINGO', 'L1_m9', 67, 'fgas-8sigma'),
    'FLAMINGO_Jet_fgas-4sigma': ('FLAMINGO', 'L1_m9', 67, 'Jet_fgas-4sigma'),
}
FITS = ['../data/fit_dsigma_ksz/fit_dsigma_ksz_z05.npz',
        '../data/fit_dsigma_ksz/fit_dsigma_ksz_z05_simba50.npz']
Z, PIX, BEAM, PROJ = 0.5, 0.5, 1.6, 'xy'
STACK_KW = dict(filterType='CAP', minRadius=1.0, maxRadius=6.0, numRadii=11, projection=PROJ,
                radDistance=1.0, radDistanceUnits='arcmin', z=Z, pixelSize=PIX,
                use_subhalos=True, halo_mass_upper=5e14)


def load_cube(st):
    """Cached unmasked 3D tau cube (float), as create_masked_field loads it."""
    n, _ = map_geometry(st, Z, PIX)
    path = _get_data_filepath(st.simType, st.sim, st.snapshot, st.feedback, 'tau', n, PROJ,
                              'field', '3D', False, 2.0, st.base_path)
    if not path.exists():
        raise FileNotFoundError(f"no cached 3D cube {path}")
    return np.load(path), n


def sphere_mask(cube, st, h, rows, n, factor=1.0):
    """Boolean cube: within factor x R200m of the given halo rows (catalogue ``h``)."""
    kpp = st.header['BoxSize'] / n
    pos = np.round(h['GroupPos'][rows] / kpp).astype(int)
    return get_cutout_mask_3d(cube, pos, h['GroupRad'][rows] * factor / kpp)


def project(cube, arcpp, z_slice=None, mask=None, chunk=64):
    """Beam-convolved xy map of the cube, optionally only the cells z_slice along z
    and only where ``mask`` is True. Summed in z-chunks, so a masked projection
    never materialises a second full cube (179 GB for FLAMINGO)."""
    zs = np.arange(cube.shape[2]) if z_slice is None else np.asarray(z_slice)
    acc = np.zeros(cube.shape[:2], dtype=np.float64)
    for i in range(0, zs.size, chunk):
        idx = zs[i:i + chunk]
        part = cube[:, :, idx]
        if mask is not None:
            part = np.where(mask[:, :, idx], part, 0)
        acc += np.sum(part, axis=2, dtype=np.float64)
    return fft_smoothed_map(acc, BEAM, pixel_size_arcmin=arcpp)


def stack(st, arr, rows, density):
    """Mean and standard error of the CAP profile of the SHAM galaxies ``rows``."""
    radii, prof = st.stack_on_array(arr, halo_abundance_target=density, halo_mask=rows, **STACK_KW)
    return radii, np.mean(prof, axis=1), np.std(prof, axis=1) / np.sqrt(prof.shape[1])


def threshold_key(thr):
    """Unambiguous result key of a halo-mass threshold, e.g. 1e12 -> 'all_1e12'."""
    return 'all_' + f'{thr:.0e}'.replace('+', '')


def sample(st, label):
    """Lensing-fit SHAM galaxies (subhalo rows), their host rows, and the density."""
    stype, name, _, fb = SIMS[label]
    density = load_fitted_abundances([f for f in FITS if Path(f).exists()])[
        fit_label(stype, {'name': name, 'feedback': fb})]
    subs = st.loadSubHalos()
    gal = select_sample(st, use_subhalos=True, halo_abundance_target=density,
                        halo_mass_upper=5e14, subhalos=subs)
    hosts = np.unique(np.asarray(subs['SubhaloGrNr'], dtype=np.int64)[gal])
    return np.asarray(gal), hosts, density, subs


def run_diffuse(label, thresholds):
    stype, name, snap, fb = SIMS[label]
    st = SimulationStacker(name, snap, z=Z, simType=stype, feedback=fb)
    gal, hosts, density, _ = sample(st, label)
    cube, n = load_cube(st)
    _, arcpp = map_geometry(st, Z, PIX)
    h = st.loadHalos()
    out = {'density': density, 'n_gal': gal.size, 'n_hosts': hosts.size}
    radii, out['U'], out['U_err'] = stack(st, project(cube, arcpp), gal, density)
    out['radii'] = radii
    m = sphere_mask(cube, st, h, hosts, n)
    radii, out['hosts'], out['hosts_err'] = stack(st, project(cube, arcpp, mask=m), gal, density)
    gmass = np.asarray(h['GroupMass'])
    for thr in sorted(thresholds, reverse=True):
        rows = np.flatnonzero(gmass >= thr)
        m |= sphere_mask(cube, st, h, rows, n)  # hosts plus all haloes above thr
        key = threshold_key(thr)
        out[f'n_{key}'] = rows.size
        radii, out[key], out[f'{key}_err'] = stack(st, project(cube, arcpp, mask=m), gal, density)
    return out


def run_los(label, depths):
    stype, name, snap, fb = SIMS[label]
    st = SimulationStacker(name, snap, z=Z, simType=stype, feedback=fb)
    gal, hosts, density, subs = sample(st, label)
    cube, n = load_cube(st)
    _, arcpp = map_geometry(st, Z, PIX)
    box = st.header['BoxSize']
    kpp = box / n
    zgal = np.mod(np.asarray(subs['SubhaloPos'])[gal, 2], box)
    m = sphere_mask(cube, st, st.loadHalos(), hosts, n)
    out = {'density': density, 'n_gal': gal.size}
    for tag, mask in (('U', None), ('hosts', m)):
        radii, out[f'{tag}_full'], out[f'{tag}_full_err'] = stack(st, project(cube, arcpp, mask=mask), gal, density)
        out['radii'] = radii
        for d in depths:
            nd = max(int(round(d / kpp)), 2)
            stride = max(nd // 2, 1)
            starts = np.arange(0, n, stride)
            centres = (starts + nd / 2.0) * kpp
            dist = np.abs(((zgal[:, None] - centres[None, :]) + box / 2) % box - box / 2)
            owner = np.argmin(dist, axis=1)
            tot, w = None, 0
            for k, s0 in enumerate(starts):
                rows = gal[owner == k]
                if rows.size == 0:
                    continue
                idx = np.arange(s0, s0 + nd) % n
                _, prof, _ = stack(st, project(cube, arcpp, z_slice=idx, mask=mask), rows, density)
                tot = prof * rows.size if tot is None else tot + prof * rows.size
                w += rows.size
            out[f'{tag}_D{int(d)}'] = tot / w
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('mode', choices=['diffuse', 'los'])
    ap.add_argument('--sims', nargs='+', default=['TNG100-1'])
    ap.add_argument('--thresholds', type=float, nargs='+', default=[1e12, 1e11])
    ap.add_argument('--depths', type=float, nargs='+', default=[50000.0, 100000.0],
                    help='slab depths in ckpc/h (los mode)')
    args = ap.parse_args()
    now = datetime.now()
    fig_path = Path('../figures') / now.strftime('%Y-%m') / now.strftime('%m-%d')
    fig_path.mkdir(parents=True, exist_ok=True)
    results = {}
    for label in args.sims:
        t0 = time.time()
        res = run_diffuse(label, args.thresholds) if args.mode == 'diffuse' else run_los(label, args.depths)
        np.savez(fig_path / f'test_{args.mode}_{label}.npz', **res)
        results[label] = res
        print(f"{label}: done in {time.time() - t0:.0f} s", flush=True)

    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4), squeeze=False)
    for ax, (label, r) in zip(axes[0], results.items()):
        x = r['radii']
        if args.mode == 'diffuse':
            ax.plot(x, 1 - r['hosts'] / r['U'], 'k-o', label='outside stacked hosts')
            for key in sorted(k for k in r if k.startswith('all_') and not k.endswith('_err')):
                ax.plot(x, 1 - r[key] / r['U'], '-o', label=f'outside hosts and haloes $>{key[4:]}$')
        else:
            for tag, ls in (('U', '-'), ('hosts', '--')):
                ax.plot(x, r[f'{tag}_full'], 'k' + ls, label=f'{tag}, full box')
                for d in args.depths:
                    ax.plot(x, r[f'{tag}_D{int(d)}'], ls, label=f'{tag}, D = {d / 1000:.0f} Mpc/h')
        ax.set_title(label)
        ax.set_xlabel('R [arcmin]')
        ax.grid(True)
    axes[0, 0].set_ylabel('fraction of CAP signal' if args.mode == 'diffuse' else 'CAP tau profile')
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_path / f'test_{args.mode}_{"_".join(results)}.pdf')
    print('saved', fig_path)


if __name__ == '__main__':
    main()
