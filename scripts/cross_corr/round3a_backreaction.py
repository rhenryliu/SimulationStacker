"""round3a_backreaction.py
=========================
Round 3A, Stage 4: the one step of the suppression mapping that was adopted
rather than measured -- the hydrodynamic back-reaction on the CDM,

    P_mm^hydro(k) = P_tt^DMO(k)        (formalism Eq. 62, ledger row 13;
                                        open-items O-14)

-- measured at z ~ 0.5 for the four runs, now that DMO references are on
disk.  Also the full chain against the true suppression
``S(k) = P_tt^hydro / P_tt^DMO``, and the component split of formalism
Eq. (67), ``x_b = w_e x_e + w_n x_n + w_star x_star (+ w_BH x_BH)``.

Inputs are the unbound-gas pipeline's 3D products (user-approved dependency,
2026-09-27): ``<stem>_Pk_components_<n>.npz`` (auto/cross spectra of the
DM, ionized-gas, neutral-gas, stellar and black-hole overdensities on one TSC
grid, TSC-deconvolved, neutrinos excluded) and ``<stem>_Pk_dmo_<n>.npz``
(``P_dmo``, the matched DMO run's DM spectrum), both under
``products/3D`` on scratch (the hard-coded ``/pscratch`` root is the
repository's known technical debt).  The CDM here is ``DM = total - gas -
Stars - BH`` on the same grid, the same definition as the 2D maps.  The 3D
grids reach ``k_Nyq`` = 9.2 h/Mpc (FLAMINGO, 2000^3) and 15.3 h/Mpc
(TNG300-1, 1000^3).

Writes a figure under ``figures/<yyyy-mm>/<mm-dd>/`` and the numbers to
``data/cross_corr_C/round3a/stage4_backreaction.{npz,txt}``.

Usage
-----
    cd scripts/
    python cross_corr/round3a_backreaction.py
"""

import argparse
import io
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('../src/')
sys.path.append(str(Path(__file__).resolve().parent))

import round3a_lib as lib

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'text.usetex': False,
    'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
    'legend.fontsize': 8,
})

#: label -> (suite directory, file stem, grid size, ck_spectra label)
RUNS = {'TNG300-1': ('IllustrisTNG', 'TNG300-1_67', 1000, 'TNG300-1'),
        'FLA fid': ('FLAMINGO', 'L1_m9_L1_m9_67', 2000, 'L1_m9_fiducial'),
        'FLA Jet': ('FLAMINGO', 'L1_m9_Jet_fgas-4sigma_67', 2000,
                    'L1_m9_Jet_fgas-4sigma'),
        'FLA fgas-8σ': ('FLAMINGO', 'L1_m9_fgas-8sigma_67', 2000,
                        'L1_m9_fgas-8sigma')}
COLOURS = dict(zip(RUNS, ('0.25', 'tab:blue', 'tab:orange', 'tab:red')))
K_REPORT = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0)


def load_run(sim_root, suite, stem, n):
    """Load one run's component and DMO spectra and build the chain.

    Args:
        sim_root (pathlib.Path): Root of the simulation products.
        suite (str): Suite directory.
        stem (str): File stem.
        n (int): Grid size in the file names.

    Returns:
        dict: ``k``, bookkeeping from ``round3a_lib.component_bookkeeping``,
        ``P_total``, ``P_dmo``, ``B``, ``S_true``, ``S_int``, ``S_model``
        and ``k_nyq``.
    """
    base = sim_root / suite / 'products' / '3D'
    comp = np.load(base / f'{stem}_Pk_components_{n}.npz', allow_pickle=True)
    dmo = np.load(base / f'{stem}_Pk_dmo_{n}.npz', allow_pickle=True)
    if not np.allclose(comp['k'], dmo['k']):
        raise ValueError(f'{stem}: component and DMO k bins differ.')
    book = lib.component_bookkeeping(comp['P'], comp['means'],
                                     [str(c) for c in comp['components']])
    f_b = book['f_b']
    out = dict(book)
    out['k'] = comp['k']
    out['P_total'] = comp['P_total']
    out['P_dmo'] = dmo['P_dmo']
    out['B'] = book['P_mm'] / dmo['P_dmo']
    out['S_true'] = comp['P_total'] / dmo['P_dmo']
    out['S_int'] = comp['P_total'] / book['P_mm']
    out['S_model'] = ((1.0 - f_b) + f_b * book['x']) ** 2
    out['k_nyq'] = np.pi * float(comp['n_pixels']) / float(comp['box_mpc'])
    return out


def x_2d(ck_path):
    """``x(k) = P_bm/P_mm`` from the 2D maps, in logarithmic bins.

    Args:
        ck_path (pathlib.Path): A ``ck_spectra`` file.

    Returns:
        tuple: ``(k in h/Mpc, x)``.
    """
    d = np.load(ck_path, allow_pickle=True)
    lib.require_regression_pass(d, ck_path)
    edges = np.logspace(-1.3, 1.3, 40)
    kc, sp = lib.rebin_spectra(d['counts'], d['k_mean'] / float(
        d['meta_mpc_per_arcmin']), {'bm': d['P2D_bm'], 'mm': d['P2D_mm']},
        edges)
    return kc, sp['bm'] / sp['mm']


def at(k, y, kk):
    """Value of ``y`` at the bin nearest ``kk``.

    Args:
        k (np.ndarray): Wavenumbers.
        y (np.ndarray): Values.
        kk (float): Target wavenumber.

    Returns:
        float: ``y`` at the nearest bin.
    """
    return float(y[int(np.argmin(np.abs(k - kk)))])


def report(runs, x2):
    """Print the Stage 4 numbers.

    Args:
        runs (dict): label -> output of :func:`load_run`.
        x2 (dict): label -> ``(k, x)`` from the 2D maps.
    """
    print('=' * 78)
    print('Round 3A, Stage 4: back-reaction and the suppression chain, '
          'z ~ 0.5 (3D)')
    print('=' * 78)
    print('\n[acceptance] bookkeeping: P_tt = f_m^2 P_mm + 2 f_m f_b P_bm + '
          'f_b^2 P_bb against the stored total, max relative difference '
          'for k <= 5 h/Mpc:')
    for label, r in runs.items():
        sel = r['k'] <= 5.0
        worst = np.max(np.abs(r['P_tt'][sel] / r['P_total'][sel] - 1.0))
        print(f"  {label:12s} {worst:.1e}   (f_b = {r['f_b']:.5f}, k_Nyq = "
              f"{r['k_nyq']:.1f} h/Mpc)")

    head = ''.join(f'{k:>8.1f}' for k in K_REPORT)
    print(f'\n[O-14] back-reaction B(k) = P_mm^hydro / P_DMO       k = {head}')
    for label, r in runs.items():
        print(f'  {label:12s}' + ' ' * 28 + ''.join(
            f"{at(r['k'], r['B'], k):8.3f}" for k in K_REPORT))
    print(f'\n[chain] S_true = P_tt^hydro/P_DMO; S_int = P_tt/P_mm (what the '
          f'estimator targets); (f_m + f_b x)^2 from the 3D x')
    for label, r in runs.items():
        for key, name in (('S_true', 'S_true'), ('S_int', 'S_int'),
                          ('S_model', '(fm+fb x)^2')):
            print(f'  {label:12s} {name:12s}' + ' ' * 16 + ''.join(
                f"{at(r['k'], r[key], k):8.3f}" for k in K_REPORT))
        err = r['S_int'] / r['S_true'] - 1.0
        print(f'  {label:12s} S_int/S_true-1' + ' ' * 14 + ''.join(
            f"{at(r['k'], err, k):+8.3f}" for k in K_REPORT))

    print('\n[Eq. 67] baryon mass weights and per-component x_i = '
          'P_{i,DM}/P_DM,DM at k = 1, 3, 5 h/Mpc:')
    for label, r in runs.items():
        w = r['weights']
        print(f"  {label:12s} weights: ionized {w['ionized_gas']:.3f}, "
              f"neutral {w['neutral_gas']:.3f}, stars {w['Stars']:.3f}, "
              f"BH {w['BH']:.4f}")
        for kk in (1.0, 3.0, 5.0):
            xs = {c: at(r['k'], r['x_comp'][c], kk)
                  for c in lib.BARYON_COMPONENTS}
            print(f"  {'':12s} k={kk:3.0f}: x_ion {xs['ionized_gas']:.3f}  "
                  f"x_neu {xs['neutral_gas']:.3f}  x_star {xs['Stars']:.3f}  "
                  f"x_BH {xs['BH']:.3f}  ->  x_b {at(r['k'], r['x'], kk):.3f}"
                  f"; x_b/x_ion {at(r['k'], r['x'], kk) / xs['ionized_gas']:.3f}")

    print('\n[P9] 3D x(k) against the 2D maps\' x(k), max |dx| over '
          '0.3 <= k <= 5 h/Mpc (threshold 0.02):')
    for label, r in runs.items():
        k2, xx = x2[label]
        sel = (k2 >= 0.3) & (k2 <= 5.0) & np.isfinite(xx)
        x3 = np.interp(np.log(k2[sel]), np.log(r['k']), r['x'])
        print(f'  {label:12s} {np.max(np.abs(x3 - xx[sel])):.4f}')


def fig(runs, x2, fig_dir):
    """Back-reaction, the suppression chain, and the component split.

    Args:
        runs (dict): label -> output of :func:`load_run`.
        x2 (dict): label -> ``(k, x)`` from the 2D maps.
        fig_dir (pathlib.Path): Output directory.

    Returns:
        pathlib.Path: The figure.
    """
    fig_, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for label, r in runs.items():
        k, c = r['k'], COLOURS[label]
        sel = k <= r['k_nyq']
        axes[0].plot(k[sel], r['B'][sel], color=c, label=label)
        axes[1].plot(k[sel], r['S_true'][sel], '-', color=c, label=label)
        axes[1].plot(k[sel], r['S_int'][sel], '--', color=c)
        axes[1].plot(k[sel], r['S_model'][sel], ':', color=c)
        k2, xx = x2[label]
        axes[2].plot(k2, xx, 'o', color=c, ms=3)
        axes[2].plot(k[sel], r['x'][sel], '-', color=c, label=f'{label}: x_b')
        axes[2].plot(k[sel], r['x_comp']['ionized_gas'][sel], '--', color=c,
                     lw=1)
    axes[0].axhspan(0.98, 1.02, color='0.9', zorder=-1)
    axes[0].axhline(1.0, color='k', lw=0.8)
    axes[0].set_ylabel(r'$P^{\rm hydro}_{mm}/P^{\rm DMO}$')
    axes[0].set_title('O-14: back-reaction (grey: the booked 1-2%)')
    axes[1].set_ylabel('suppression')
    axes[1].set_title(r'solid $S$; dashed $P_{tt}/P_{mm}$; dotted '
                      r'$(f_m+f_bx)^2$')
    axes[2].set_ylabel(r'$x = P_{bm}/P_{mm}$')
    axes[2].set_title('solid 3D $x_b$, dashed 3D $x_{\\rm ion}$, dots 2D $x_b$')
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlim(0.1, 12)
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
    axes[0].legend()
    fig_.suptitle('Stage 4, z ≈ 0.5: the DMO reference closes the chain')
    fig_.tight_layout()
    path = fig_dir / 'round3a_s4_backreaction.png'
    fig_.savefig(path, dpi=140)
    plt.close(fig_)
    return path


def main(sim_root, r3_dir, fig_root, verbose=True):
    """Run Stage 4.

    Args:
        sim_root (str): Root of the simulation products.
        r3_dir (str): Round 3A output directory (for the 2D spectra, and the
            numbers written here).
        fig_root (str): Root of the dated figure directories.
        verbose (bool, optional): Print the report.
    """
    sim_root, r3_dir = Path(sim_root), Path(r3_dir)
    now = datetime.now()
    fig_dir = Path(fig_root) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs, x2 = {}, {}
    for label, (suite, stem, n, cklabel) in RUNS.items():
        runs[label] = load_run(sim_root, suite, stem, n)
        x2[label] = x_2d(r3_dir / f'ck_spectra_{cklabel}_67_yz.npz')
    path = fig(runs, x2, fig_dir)
    buf = io.StringIO()
    with redirect_stdout(buf):
        report(runs, x2)
    text = buf.getvalue()
    if verbose:
        print(text)
        print(f'Wrote {path}')
    flat = {}
    for label, r in runs.items():
        for key in ('k', 'P_mm', 'P_bm', 'P_bb', 'P_tt', 'P_total', 'P_dmo',
                    'B', 'S_true', 'S_int', 'S_model', 'x', 'f_b', 'k_nyq'):
            flat[f'{label}__{key}'] = np.asarray(r[key])
        for c in lib.BARYON_COMPONENTS:
            flat[f'{label}__x_{c}'] = r['x_comp'][c]
            flat[f'{label}__w_{c}'] = np.asarray(r['weights'][c])
    lib.save_npz_atomic(r3_dir / 'stage4_backreaction.npz', **flat)
    lib.write_text_atomic(r3_dir / 'stage4_backreaction.txt', text)
    print(f"Wrote {r3_dir / 'stage4_backreaction.txt'} and .npz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Round 3A Stage 4: back-reaction and the chain at z~0.5.')
    parser.add_argument('--sim-root', default='/pscratch/sd/r/rhliu/simulations')
    parser.add_argument('--r3-dir', default='../data/cross_corr_C/round3a/')
    parser.add_argument('--fig-root', default='../figures/')
    args = parser.parse_args()
    main(args.sim_root, args.r3_dir, args.fig_root)
