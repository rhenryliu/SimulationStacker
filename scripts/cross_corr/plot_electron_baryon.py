"""plot_electron_baryon.py
=========================
Task 3 deliverable: the electron-versus-baryon split.

The kSZ stack measures the free-electron field ``e``, while the suppression
algebra of the theory note (Eqs. 1-7) is written for all baryons ``b``.  This
script reports, from the ``.npz`` files written by ``make_r_profiles.py``:

1. the correction factor ``Y_gb / Y_ge(R)`` per simulation and filter, i.e. how
   much of the baryon signal the electron field misses;
2. the two candidate framings side by side -- targeting ``P_bm/P_mm`` with a
   calibrated electron-to-baryon transfer, or targeting ``P_em/P_mm`` directly
   -- judged by which of ``r_bm/r_gb`` and ``r_em/r_ge`` is the more stable
   across codes, which is what decides the cleaner deliverable.

Neutral gas is not separated out: ``baryon - ionized_gas`` lumps neutral gas,
stars and black holes together, and isolating the stellar part would need new
particle sweeps.  The aggregate split is what is reported here.

Only the compensated filters are used for the recommendation.  Sigma is
retained in the tables as a diagnostic but is not comparable between boxes of
different size -- see ``check_filter_compensation.py``.

Usage
-----
    cd scripts/
    python cross_corr/plot_electron_baryon.py -p configs/cross_corr/r_profiles_z05.yaml
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.append('../src/')
import rprofiles as rp
sys.path.append(str(Path(__file__).parent))
from plot_r_profiles import load_runs, representative_runs, series

#: The frozen filter set.  Sigma is reported but excluded from decisions.
FROZEN_FILTERS = ('DSigma', 'Upsilon')


def correction_factor(run, filt):
    """Return Y_gb / Y_ge(R), the electron-to-baryon correction.

    Args:
        run (dict): One entry from :func:`plot_r_profiles.load_runs`.
        filt (str): Filter name.

    Returns:
        np.ndarray: Correction factor per aperture.
    """
    d = run['data']
    with np.errstate(invalid='ignore', divide='ignore'):
        return d[f'Y_bg_{filt}'] / d[f'Y_eg_{filt}']


def make_figure(runs, out_path):
    """Plot the correction factor and the two candidate ratio framings.

    Args:
        runs (list): Runs from :func:`plot_r_profiles.load_runs`.
        out_path (pathlib.Path): Output figure path.
    """
    n_cols = len(rp.FILTERS)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.6 * n_cols, 7.0),
                             sharex=True, squeeze=False)
    colours = matplotlib.colormaps['viridis'](
        np.linspace(0.05, 0.9, max(len(runs), 1)))

    for j, filt in enumerate(rp.FILTERS):
        ax = axes[0][j]
        ax.axhline(1.0, color='k', ls='--', lw=1.2)
        for k, run in enumerate(runs):
            v = correction_factor(run, filt)
            good = np.isfinite(v)
            ax.plot(run['radii'][good], v[good], color=colours[k], lw=1.8,
                    marker='o', ms=3.5,
                    label=run['label'] if j == 0 else None)
        ax.set_title(filt if filt != 'Upsilon' else r"$\Upsilon(R_0=1')$")
        if j == 0:
            ax.set_ylabel(r'$Y_{gb}/Y_{ge}$')
        ax.grid(alpha=0.3)

        ax = axes[1][j]
        ax.axhline(1.0, color='k', ls='--', lw=1.2)
        for k, run in enumerate(runs):
            b, _ = series(run, ('b', 'm'), ('g', 'b'), filt)
            e, _ = series(run, ('e', 'm'), ('g', 'e'), filt)
            gb = np.isfinite(b)
            ax.plot(run['radii'][gb], b[gb], color=colours[k], lw=1.8,
                    label='baryon route' if (j == 0 and k == 0) else None)
            ge = np.isfinite(e)
            ax.plot(run['radii'][ge], e[ge], color=colours[k], lw=1.4,
                    ls=':', label='electron route' if (j == 0 and k == 0)
                    else None)
        if j == 0:
            ax.set_ylabel(r'$r_{bm}/r_{gb}$  and  $r_{em}/r_{ge}$')
        ax.set_xlabel(r'$R$ [arcmin]')
        ax.grid(alpha=0.3)

    h0, l0 = axes[0][0].get_legend_handles_labels()
    h1, l1 = axes[1][0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc='upper center',
               ncol=min(len(runs) + 2, 6), frameon=True,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('Task 3: electrons versus all baryons', y=1.06, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to: {out_path}')


def report(runs, out_path, data_max=None):
    """Print and save the Task 3 numbers and the framing recommendation.

    Args:
        runs (list): Runs from :func:`plot_r_profiles.load_runs`.
        out_path (pathlib.Path): Destination text file.
        data_max (float, optional): Largest observationally accessible
            aperture in arcmin.  The recommendation uses only apertures at or
            below it, since the extension above is a diagnostic and is noisy
            in the smaller box.  Defaults to None (use every aperture).
    """
    lines = []

    def emit(text=''):
        print(text)
        lines.append(text)

    reps = representative_runs(runs)
    radii = runs[0]['radii']
    in_data = (np.ones(len(radii), dtype=bool) if data_max is None
               else radii <= data_max + 1e-9)

    emit('=' * 78)
    emit('Task 3: electrons versus all baryons')
    emit('=' * 78)
    emit()
    emit('b = gas + stars + BH ("baryon"); e = ionized gas ("ionized_gas").')
    emit('Neutral gas is not separated from stars; the split reported is the')
    emit('aggregate b - e.')
    emit()

    emit('Correction factor Y_gb / Y_ge(R)')
    for filt in rp.FILTERS:
        emit(f'  {filt}:')
        emit(f"    {'run':26s} " + " ".join(f'{R:7.2f}' for R in radii))
        for run in runs:
            v = correction_factor(run, filt)
            emit(f"    {run['label']:26s} "
                 + " ".join(f'{x:7.3f}' for x in v))
        emit()

    emit('Cross-code stability of the two candidate framings')
    emit('  (spread across code families of the ratio each framing calibrates)')
    emit()
    verdict = {}
    for filt in rp.FILTERS:
        emit(f'  {filt}:')
        emit(f"    {'R':>7} {'baryon route':>14} {'electron route':>16}")
        B = np.vstack([series(r, ('b', 'm'), ('g', 'b'), filt)[0]
                       for r in reps])
        E = np.vstack([series(r, ('e', 'm'), ('g', 'e'), filt)[0]
                       for r in reps])
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sb = np.nanstd(B, axis=0, ddof=1) / np.abs(np.nanmean(B, axis=0))
            se = np.nanstd(E, axis=0, ddof=1) / np.abs(np.nanmean(E, axis=0))
        for i, R in enumerate(radii):
            tag = '' if in_data[i] else '   (extension)'
            emit(f'    {R:7.2f} {sb[i]:14.4f} {se[i]:16.4f}{tag}')
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            wb, we = np.nanmax(sb[in_data]), np.nanmax(se[in_data])
        verdict[filt] = (wb, we)
        emit(f'    worst over the data range   {wb:.4f} (baryon) '
             f'{we:.4f} (electron)')
        emit()

    emit('Recommendation, over the data range, compensated filters only.')
    emit('(Sigma amplitudes are not comparable between boxes of different')
    emit('size -- see check_filter_compensation.py -- though its coefficients')
    emit('largely are.)')
    for filt in FROZEN_FILTERS:
        wb, we = verdict[filt]
        if not (np.isfinite(wb) and np.isfinite(we)):
            emit(f'  {filt}: indeterminate')
            continue
        # Each of wb and we is itself a spread over only len(reps) code
        # families -- for two families it is one pairwise difference divided
        # by sqrt(2), with no error bar at all. Comparing two such numbers
        # cannot establish a preference at any level worth acting on, so the
        # verdict is always reported as nominal and the margin is quoted.
        spread = abs(wb - we) / max(wb, we)
        better = 'electron (P_em/P_mm)' if we < wb else 'baryon (P_bm/P_mm)'
        if spread < 0.10:
            emit(f'  {filt}: baryon route {wb:.4f} vs electron route '
                 f'{we:.4f} -> INDISTINGUISHABLE ({spread:.1%} apart)')
        else:
            emit(f'  {filt}: baryon route {wb:.4f} vs electron route '
                 f'{we:.4f} -> nominally favours {better} ({spread:.1%} '
                 f'apart), but see the caveat below')
    emit()
    emit(f'CAVEAT: each spread above is taken over {len(reps)} code '
         f'famil{"y" if len(reps) == 1 else "ies"}. With so few, neither '
         f'number carries an')
    emit('error bar, and a difference of a few tens of per cent between them')
    emit('is not evidence for either framing. Treat every verdict on this')
    emit('table as nominal.')
    emit()
    emit('Where the two framings are indistinguishable on stability, the')
    emit('deciding evidence is the size and code-dependence of the')
    emit('electron-to-baryon correction Y_gb/Y_ge tabulated above: a large,')
    emit('feedback-dependent correction argues for targeting the electron')
    emit('field directly, which is what the kSZ actually measures.')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f'\nReport written to: {out_path}')


def main(path2config):
    """Produce the Task 3 figure and report for one config.

    Args:
        path2config (str): Path to the YAML configuration file.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)

    plot_cfg = config.get('plot', {})
    npz_dir = Path(plot_cfg.get('npz_path', '../data/r_profiles/'))
    fig_name = plot_cfg.get('fig_name', 'r_profiles')
    fig_type = plot_cfg.get('fig_type', 'pdf')

    wanted = {(suite['sim_type'], entry['name'],
               str(entry.get('feedback')), entry['snapshot'])
              for suite in config['simulations']
              for entry in suite['sims']}
    runs = load_runs(npz_dir, wanted=wanted)
    if not runs:
        raise SystemExit(f'No matching r_profiles_*.npz in {npz_dir}. '
                         'Run make_r_profiles.py first.')

    now = datetime.now()
    out_dir = (Path(plot_cfg.get('fig_path', '../figures/'))
               / now.strftime('%Y-%m') / now.strftime('%m-%d'))
    make_figure(runs, out_dir / f'{fig_name}_electron_baryon.{fig_type}')
    print()
    report(runs, out_dir / f'{fig_name}_electron_baryon.txt',
           data_max=config.get('stack', {}).get('max_radius'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Task 3: electron versus baryon split and framing choice.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/r_profiles_z05.yaml')
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
