"""round3a_dog_analysis.py
========================
Round 3A, Stage 3 analysis (``docs/cross_corr/open-items.md`` O-02; prediction
P7): does a compensated kernel with a strictly positive window -- the
difference of Gaussians (DoG) of formalism Eq. (22) -- shrink ``|C - 1|`` and
its feedback ordering, compared with DSigma at matched ``k_50``?

Reads ``data/cross_corr_C/round3a/dog_calibration_*.npz`` (the map-level DoG
sweep with jackknife errors), the round-two ``calibration_*.npz`` (DSigma),
and ``ck_spectra_*.npz`` (for ``k_50`` against each run's measured CDM
spectrum, and for the exact Parseval DoG amplitudes, which must agree with the
sweep).  Formalism §4.5's rule: "substantial shrinkage" means at least a
factor two at matched ``k_50`` in all four runs.

Writes a figure under ``figures/<yyyy-mm>/<mm-dd>/`` and the numbers to
``data/cross_corr_C/round3a/stage3_dog_analysis.{npz,txt}``.

Usage
-----
    cd scripts/
    python cross_corr/round3a_dog_analysis.py
"""

import argparse
import io
import sys
import warnings
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
from round3a_ck_analysis import RUNS, COLOURS, SAMPLES, k50, split

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'text.usetex': False,
    'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
    'legend.fontsize': 8,
})

DOG_FILTERS = ('DoG_q=2', 'DoG_q=1.5')


def load(r3_dir, cal_dir, zkey):
    """Load the DoG, round-two and spectra files of one redshift sample.

    Args:
        r3_dir (pathlib.Path): ``data/cross_corr_C/round3a``.
        cal_dir (pathlib.Path): ``data/cross_corr_C``.
        zkey (str): ``'z05'`` or ``'z026'``.

    Returns:
        dict: label to ``{'dog', 'cal', 'ck'}`` npz handles.
    """
    out = {}
    for label, _ in RUNS:
        snap = SAMPLES[zkey]['TNG' if label.startswith('TNG') else 'FLA']
        stem = f'{label}_{snap}_yz.npz'
        ck = np.load(r3_dir / f'ck_spectra_{stem}', allow_pickle=True)
        lib.require_regression_pass(ck, r3_dir / f'ck_spectra_{stem}')
        out[label] = {
            'dog': np.load(r3_dir / f'dog_calibration_{stem}',
                           allow_pickle=True),
            'cal': np.load(cal_dir / f'calibration_{stem}', allow_pickle=True),
            'ck': ck}
    return out


def compare(runs, gas='b'):
    """DoG against DSigma at matched ``k_50``, per run.

    Args:
        runs (dict): Output of :func:`load`.
        gas (str, optional): Gas field.

    Returns:
        dict: per label and DoG filter: ``k50_dog``, ``C_dog``, ``err_dog``,
        ``C_ds_at``, ``err_ds_at`` (DSigma interpolated to the DoG ``k_50``),
        ``ratio`` (``|C_dog-1| / |C_ds-1|``), ``sweep_vs_exact`` (worst
        relative difference between the map-level sweep and the Parseval
        amplitudes), and ``k50_ds1`` (DSigma's ``k_50`` at 1').
    """
    out = {}
    for label, r in runs.items():
        k_ds = k50(r['ck'], 'DSigma')
        c_ds = r['cal'][f'C_{gas}_DSigma']
        e_ds = r['cal'][f'Cerr_{gas}_DSigma']
        for filt in DOG_FILTERS:
            k_dog = k50(r['ck'], filt)
            c_dog = r['dog'][f'C_{gas}_{filt}']
            exact = split(r['ck'], filt, gas=gas)['C']
            c_at = lib.interp_log_k(k_ds, c_ds, k_dog)
            e_at = lib.interp_log_k(k_ds, e_ds, k_dog)
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = np.abs(c_dog - 1.0) / np.abs(c_at - 1.0)
            out[(label, filt)] = {
                'k50_dog': k_dog, 'C_dog': c_dog,
                'err_dog': r['dog'][f'Cerr_{gas}_{filt}'],
                'C_ds_at': c_at, 'err_ds_at': e_at, 'ratio': ratio,
                'sweep_vs_exact': float(np.nanmax(np.abs(c_dog / exact - 1))),
                'k50_ds1': float(k_ds[0]),
                'sigma1': r['dog']['sigma1']}
    return out


def report(results, zkeys):
    """Print the Stage 3 numbers.

    Args:
        results (dict): ``{zkey: compare(...)}``.
        zkeys (iterable): Redshift keys.
    """
    print('=' * 78)
    print('Round 3A, Stage 3: DoG against DSigma at matched k_50 (baryons, '
          'Convention C)')
    print('=' * 78)
    for zkey in zkeys:
        res = results[zkey]
        print(f'\n---------------- z ≈ {SAMPLES[zkey]["z"]} ----------------')
        print('\n[check] map-level DoG sweep vs exact Parseval DoG C, worst '
              'relative difference:')
        print('  ' + ', '.join(
            f"{s}: {max(res[(l, f)]['sweep_vs_exact'] for f in DOG_FILTERS):.1e}"
            for l, s in RUNS))
        for filt in DOG_FILTERS:
            print(f"\n[P7] {filt}: at each sigma1, k_50, C_DoG ± jk, DSigma at "
                  f"the same k_50, and |C_DoG-1|/|C_DS-1| (shrinkage if <= 0.5)")
            for label, short in RUNS:
                r = res[(label, filt)]
                print(f"  {short:12s} (DSigma 1' k_50 = {r['k50_ds1']:.2f} "
                      'h/Mpc)')
                for i, s1 in enumerate(r['sigma1']):
                    print(f"    sigma1 {s1:5.3f}': k50 {r['k50_dog'][i]:5.2f}  "
                          f"C_DoG {r['C_dog'][i]:.4f}±{r['err_dog'][i]:.4f}  "
                          f"C_DS {r['C_ds_at'][i]:.4f}  ratio "
                          f"{r['ratio'][i]:.2f}")
        print("\n[P7] Summary over the k_50 range the DoG shares with DSigma "
              "(1'-9.75'): median |C_DoG-1|/|C_DS-1|, points showing "
              'shrinkage (ratio <= 0.5), and the highest-k matched point')
        for filt in DOG_FILTERS:
            for label, short in RUNS:
                r = res[(label, filt)]
                ok = np.isfinite(r['ratio'])
                if not ok.any():
                    print(f'  {filt:10s} {short:12s} no k_50 overlap with '
                          'DSigma; skipped')
                    continue
                ratios = r['ratio'][ok]
                top = int(np.flatnonzero(ok)[0])    # sigma1 ascends: highest k
                print(f"  {filt:10s} {short:12s} median {np.median(ratios):5.2f}"
                      f"; shrinkage at {int(np.sum(ratios <= 0.5))}/"
                      f"{ratios.size}; highest matched k_50 "
                      f"{r['k50_dog'][top]:.2f}: C_DoG {r['C_dog'][top]:.3f} "
                      f"vs C_DS {r['C_ds_at'][top]:.3f}")
        print('\n[P7] Verdict (formalism §4.5): substantial shrinkage = ratio '
              '<= 0.5 at matched k_50 in all four runs, at the same sigma1:')
        for filt in DOG_FILTERS:
            stack = np.vstack([res[(label, filt)]['ratio'] for label, _ in RUNS])
            common = np.all(np.isfinite(stack), axis=0)
            hits = np.all(stack[:, common] <= 0.5, axis=0)
            print(f'  {filt}: {int(hits.sum())} of {int(common.sum())} '
                  'sigma1 values with all four runs matched')
        print('\n[P7] Feedback ordering at fixed sigma1 (FLAMINGO fid, Jet, '
              'fgas-8σ): C_DoG, and DSigma at the same k_50')
        fla = [l for l, _ in RUNS if not l.startswith('TNG')]
        for filt in DOG_FILTERS:
            sig = res[(fla[0], filt)]['sigma1']
            for i in range(len(sig)):
                cd = [res[(l, filt)]['C_dog'][i] for l in fla]
                cs = [res[(l, filt)]['C_ds_at'][i] for l in fla]
                mono = cd[0] < cd[1] < cd[2] or cd[0] < cd[2]
                print(f"  {filt:10s} sigma1 {sig[i]:5.3f}': DoG "
                      + ' '.join(f'{c:.3f}' for c in cd)
                      + ('   (fid lowest)' if mono and cd[0] == min(cd)
                         else '   (fid not lowest)')
                      + '  | DSigma ' + ' '.join(f'{c:.3f}' for c in cs))


def fig(results, fig_dir):
    """``C`` against ``k_50`` for DSigma and the two DoG families.

    Args:
        results (dict): ``{zkey: compare(...)}``, with the DSigma curves.
        fig_dir (pathlib.Path): Output directory.

    Returns:
        pathlib.Path: The figure.
    """
    fig_, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)
    for row, zkey in enumerate(results):
        for col, (label, short) in enumerate(RUNS):
            ax = axes[row, col]
            ds = results[zkey][('__ds__', label)]
            ax.errorbar(ds['k50'], ds['C'], yerr=ds['err'], fmt='s-',
                        color='k', ms=3, label=r'$\Delta\Sigma$')
            for filt, ls in zip(DOG_FILTERS, ('o--', '^:')):
                r = results[zkey][(label, filt)]
                ax.errorbar(r['k50_dog'], r['C_dog'], yerr=r['err_dog'],
                            fmt=ls, color=COLOURS[label], ms=3,
                            label=filt.replace('_', ' '))
            ax.axhline(1.0, color='0.5', lw=0.8)
            ax.set_xscale('log')
            ax.set_title(f'{short}, z≈{SAMPLES[zkey]["z"]}')
            if row == 1:
                ax.set_xlabel(r'$k_{50}$ [$h$/Mpc]')
        axes[row, 0].set_ylabel(r'$C_{\mathcal{F}}$ (baryons)')
    axes[0, 0].legend()
    fig_.suptitle('O-02: a positive-window kernel (DoG) against '
                  r'$\Delta\Sigma$ at matched $k_{50}$')
    fig_.tight_layout()
    path = fig_dir / 'round3a_s3_dog.png'
    fig_.savefig(path, dpi=140)
    plt.close(fig_)
    return path


def main(r3_dir, cal_dir, fig_root, verbose=True):
    """Run the Stage 3 analysis.

    Args:
        r3_dir (str): Round 3A output directory.
        cal_dir (str): Round-two calibration directory.
        fig_root (str): Root of the dated figure directories.
        verbose (bool, optional): Print the report.
    """
    r3_dir, cal_dir = Path(r3_dir), Path(cal_dir)
    now = datetime.now()
    fig_dir = Path(fig_root) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for zkey in SAMPLES:
            runs = load(r3_dir, cal_dir, zkey)
            res = compare(runs)
            for label, r in runs.items():
                res[('__ds__', label)] = {
                    'k50': k50(r['ck'], 'DSigma'),
                    'C': r['cal']['C_b_DSigma'],
                    'err': r['cal']['Cerr_b_DSigma']}
            results[zkey] = res
        path = fig(results, fig_dir)
        buf = io.StringIO()
        with redirect_stdout(buf):
            report(results, SAMPLES)
    text = buf.getvalue()
    if verbose:
        print(text)
        print(f'Wrote {path}')
    flat = {}
    for zkey, res in results.items():
        for key, value in res.items():
            name = f"{zkey}__{'__'.join(key)}"
            for sub, arr in value.items():
                flat[f'{name}__{sub}'] = np.asarray(arr)
    lib.save_npz_atomic(r3_dir / 'stage3_dog_analysis.npz', **flat)
    lib.write_text_atomic(r3_dir / 'stage3_dog_analysis.txt', text)
    print(f"Wrote {r3_dir / 'stage3_dog_analysis.txt'} and .npz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Round 3A Stage 3: DoG against DSigma at matched k_50.')
    parser.add_argument('--r3-dir', default='../data/cross_corr_C/round3a/')
    parser.add_argument('--cal-dir', default='../data/cross_corr_C/')
    parser.add_argument('--fig-root', default='../figures/')
    args = parser.parse_args()
    main(args.r3_dir, args.cal_dir, args.fig_root)
