"""plot_r_profiles.py
====================
Turn the ``.npz`` files written by ``make_r_profiles.py`` into the Singh et al.
(2020) Fig. 1 analogue and the Gate A decision metrics of
``docs/cross_correlation_notes.md`` (Sec. 6, Task 1).

Figures (one per field definition, baryons and electrons):

    rows:    r_gb(R),  r_bm(R),  r_bm/r_gb(R)
    columns: Sigma,    DSigma,   Upsilon(R0 = 1')
    curves:  one per simulation, with its within-projection jackknife band

Metrics printed and written alongside the figure:

    - max_R |r - 1| per coefficient, filter and simulation;
    - the cross-simulation scatter of r_bm/r_gb at each aperture, which is the
      quantity Gate A thresholds at 10 per cent.

Usage
-----
    cd scripts/
    python cross_corr/plot_r_profiles.py -p configs/cross_corr/r_profiles_z05.yaml
"""

import argparse
import glob
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

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'text.usetex': False,      # no LaTeX in the cosmodesi environment
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 9,
})

#: Rows of the figure: (numerator pair, denominator pair or None, label).
BARYON_ROWS = [
    (('g', 'b'), None, r'$r_{gb}$'),
    (('b', 'm'), None, r'$r_{bm}$'),
    (('b', 'm'), ('g', 'b'), r'$r_{bm}\,/\,r_{gb}$'),
]

ELECTRON_ROWS = [
    (('g', 'e'), None, r'$r_{ge}$'),
    (('e', 'm'), None, r'$r_{em}$'),
    (('e', 'm'), ('g', 'e'), r'$r_{em}\,/\,r_{ge}$'),
]


def code_family(sim_type, sim_name):
    """Return the code family a run belongs to, for cross-code statistics.

    Gate A is explicitly a cross-*code* test (``docs/cross_correlation_notes.md``
    Sec. 6: "Validation must therefore be cross-code ... not merely
    cross-parameter within one code"), so the FLAMINGO feedback variants count
    as one family, and Illustris and IllustrisTNG count as two despite sharing
    a ``sim_type`` -- they are different galaxy-formation models.

    Args:
        sim_type (str): Suite name from the run metadata.
        sim_name (str): Simulation name from the run metadata.

    Returns:
        str: Code family label.
    """
    if sim_type == 'IllustrisTNG':
        return 'Illustris' if sim_name.startswith('Illustris') else 'TNG'
    return sim_type


def is_fiducial_variant(sim_type, sim_name, feedback):
    """Return whether a run is its suite's fiducial feedback variant.

    Each suite spells its fiducial differently: FLAMINGO names the directory
    after the box (``L1_m9``), SIMBA calls it ``s50``, and the IllustrisTNG
    runs carry no feedback variant at all.

    Args:
        sim_type (str): Suite name.
        sim_name (str): Simulation name.
        feedback (str): Feedback variant, or the string ``'None'``.

    Returns:
        bool: True if this is the suite's fiducial run.
    """
    if sim_type == 'FLAMINGO':
        return feedback == sim_name
    if sim_type == 'SIMBA':
        return feedback == 's50'
    return feedback in ('None', '')


def representative_runs(runs):
    """Pick exactly one run per code family for the cross-code statistic.

    Gate A thresholds the scatter *between codes*, so each family contributes
    once.  The fiducial feedback variant represents its family where one is
    identifiable; otherwise the first member does, so a family is never
    dropped silently.

    Args:
        runs (list): Runs from :func:`load_runs`.

    Returns:
        list: One run per code family, ordered by family name.
    """
    by_family = {}
    for run in runs:
        by_family.setdefault(run['family'], []).append(run)
    reps = []
    for family in sorted(by_family):
        members = by_family[family]
        fiducial = [m for m in members if m['is_fiducial']]
        reps.append(fiducial[0] if fiducial else members[0])
    return reps


def load_runs(npz_dir, wanted=None):
    """Load the r-profile ``.npz`` files belonging to one config.

    Args:
        npz_dir (pathlib.Path): Directory holding ``r_profiles_*.npz``.
        wanted (set, optional): Set of ``(sim_type, sim_name, feedback,
            snapshot)`` tuples to keep.  Matching on the full key rather than
            on the snapshot alone matters because every config writes into the
            same directory, so a stale or unrelated run that happens to reuse a
            snapshot number would otherwise be pulled silently into the
            figures and the Gate A statistics.  Defaults to None (keep all).

    Returns:
        list: One dict per run, sorted by code family then label.
    """
    runs = []
    for path in sorted(glob.glob(str(npz_dir / 'r_profiles_*.npz'))):
        data = np.load(path, allow_pickle=False)
        key = (str(data['meta_sim_type']), str(data['meta_sim_name']),
               str(data['meta_feedback']), int(data['meta_snapshot']))
        if wanted is not None and key not in wanted:
            continue
        runs.append({
            'path': path,
            'label': str(data['meta_label']),
            'sim_type': key[0],
            'sim_name': key[1],
            'feedback': key[2],
            'family': code_family(key[0], key[1]),
            'is_fiducial': is_fiducial_variant(key[0], key[1], key[2]),
            'snapshot': key[3],
            'projection': str(data['meta_projection']),
            'redshift': float(data['meta_redshift']),
            'n_galaxies': int(data['meta_n_galaxies']),
            'radii': data['radii'],
            'data': data,
        })
    return sorted(runs, key=lambda r: (r['family'], r['label']))


def series(run, num, den, filt):
    """Extract a coefficient or coefficient ratio and its jackknife error.

    Args:
        run (dict): One entry from :func:`load_runs`.
        num (tuple): Numerator field pair, e.g. ``('b', 'm')``.
        den (tuple or None): Denominator field pair for a ratio, or None for
            a bare coefficient.
        filt (str): Filter name.

    Returns:
        tuple: ``(values, errors)``, each of shape ``(n_radii,)``.
    """
    d = run['data']
    if den is None:
        return d[f'r_{num[0]}{num[1]}_{filt}'], d[f'rerr_{num[0]}{num[1]}_{filt}']
    tag = f'{num[0]}{num[1]}_over_{den[0]}{den[1]}'
    return d[f'ratio_{tag}_{filt}'], d[f'ratioerr_{tag}_{filt}']


def make_figure(runs, rows, out_path, title, show_errors=True):
    """Draw the Singh Fig. 1 analogue for one field definition.

    Args:
        runs (list): Runs from :func:`load_runs`.
        rows (list): Row specification, e.g. :data:`BARYON_ROWS`.
        out_path (pathlib.Path): Output figure path.
        title (str): Figure suptitle.
        show_errors (bool, optional): Shade the jackknife band.  Defaults to
            True.
    """
    n_rows, n_cols = len(rows), len(rp.FILTERS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols,
                                                      3.4 * n_rows),
                             sharex=True, sharey='row', squeeze=False)
    colours = matplotlib.colormaps['viridis'](
        np.linspace(0.05, 0.9, max(len(runs), 1)))

    for i, (num, den, row_label) in enumerate(rows):
        for j, filt in enumerate(rp.FILTERS):
            ax = axes[i][j]
            ax.axhline(1.0, color='k', ls='--', lw=1.2, zorder=1)
            for k, run in enumerate(runs):
                radii = run['radii']
                values, errors = series(run, num, den, filt)
                # Upsilon(R0; R0) is identically zero, so its coefficient at
                # the reference radius is a genuine 0/0.  Drop it rather than
                # plotting a gap-filled line through a meaningless point.
                good = np.isfinite(values)
                if not good.any():
                    continue
                ax.plot(radii[good], values[good], color=colours[k], lw=1.8,
                        marker='o', ms=3.5,
                        label=run['label'] if (i == 0 and j == 0) else None)
                if show_errors:
                    ax.fill_between(radii[good],
                                    (values - errors)[good],
                                    (values + errors)[good],
                                    color=colours[k], alpha=0.18, lw=0)
            if i == 0:
                ax.set_title(filt if filt != 'Upsilon'
                             else r"$\Upsilon(R_0=1')$")
            if j == 0:
                ax.set_ylabel(row_label)
            if i == n_rows - 1:
                ax.set_xlabel(r'$R$ [arcmin]')
            ax.grid(alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(runs), 6),
               frameon=True, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, y=1.045, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to: {out_path}')


def gate_a_metrics(runs, rows, data_max=None):
    """Compute the Task 1 deliverable metrics.

    Args:
        runs (list): Runs from :func:`load_runs`.
        rows (list): Row specification.
        data_max (float, optional): Largest observationally accessible
            aperture in arcmin.  Statistics are reported separately for
            apertures at or below it and for the diagnostic extension above
            it, and the Gate A verdict uses the data range only.  Defaults to
            None, which treats every aperture as in range.

    Returns:
        dict: ``{'max_dev': {...}, 'scatter': {...}}`` where ``max_dev`` maps
        ``(label, coefficient, filter)`` to ``max_R |r - 1|`` and ``scatter``
        maps ``filter`` to the per-radius cross-simulation standard deviation
        of the ratio row.
    """
    max_dev = {}
    for num, den, _ in rows:
        name = (f'{num[0]}{num[1]}' if den is None
                else f'{num[0]}{num[1]}/{den[0]}{den[1]}')
        for filt in rp.FILTERS:
            for run in runs:
                values, _ = series(run, num, den, filt)
                with np.errstate(invalid='ignore'):
                    dev = np.nanmax(np.abs(values - 1.0))
                max_dev[(run['label'], name, filt)] = float(dev)

    # Scatter of the ratio row -- the Gate A quantity.
    #
    # Gate A is a cross-CODE threshold ("Validation must therefore be
    # cross-code ... not merely cross-parameter within one code", theory note
    # Sec. 6), so the primary statistic keeps one representative run per code
    # family.  Pooling the three FLAMINGO feedback variants into it would let a
    # single code's parameter sweep dominate a statistic meant to measure
    # code-to-code disagreement.  The all-run scatter is computed too and
    # reported separately as the feedback-inclusive diagnostic.
    ratio_rows = [r for r in rows if r[1] is not None]
    scatter = {}
    if ratio_rows and runs:
        num, den, _ = ratio_rows[0]
        subsets = {
            'cross-code': representative_runs(runs),
            'all-runs': list(runs),
        }
        for subset_name, subset in subsets.items():
            if len(subset) < 2:
                continue
            scatter[subset_name] = {}
            for filt in rp.FILTERS:
                stack = np.vstack([series(run, num, den, filt)[0]
                                   for run in subset])
                # The Upsilon reference-radius bin is all-NaN by construction,
                # so nanmean/nanstd legitimately reduce an empty slice there.
                with np.errstate(invalid='ignore'), \
                        warnings.catch_warnings():
                    warnings.filterwarnings('ignore',
                                            message='Mean of empty slice')
                    warnings.filterwarnings('ignore',
                                            message='Degrees of freedom <= 0')
                    mean = np.nanmean(stack, axis=0)
                    std = np.nanstd(stack, axis=0, ddof=1)
                    frac = std / np.abs(mean)
                    radii = subset[0]['radii']
                    in_data = (np.ones(len(radii), dtype=bool) if data_max is
                               None else radii <= data_max + 1e-9)
                    scatter[subset_name][filt] = {
                        'radii': radii,
                        'mean': mean,
                        'std': std,
                        'frac_std': frac,
                        'in_data': in_data,
                        'worst_data': (np.nanmax(frac[in_data])
                                       if in_data.any() else np.nan),
                        'worst_ext': (np.nanmax(frac[~in_data])
                                      if (~in_data).any() else np.nan),
                        'members': [r['label'] for r in subset],
                    }
    return {'max_dev': max_dev, 'scatter': scatter}


def report_metrics(metrics, runs, rows, out_path):
    """Print the Gate A metrics and write them to a text file.

    Args:
        metrics (dict): Output of :func:`gate_a_metrics`.
        runs (list): Runs from :func:`load_runs`.
        rows (list): Row specification.
        out_path (pathlib.Path): Destination text file.
    """
    lines = []

    def emit(text=''):
        print(text)
        lines.append(text)

    emit('=' * 78)
    emit('Task 1 metrics')
    emit('=' * 78)
    emit()
    emit('Runs:')
    for run in runs:
        emit(f"  {run['label']:24s} snap {run['snapshot']:3d}  "
             f"z={run['redshift']:.4f}  proj={run['projection']}  "
             f"N_gal={run['n_galaxies']}")
    emit()

    emit('max_R |r - 1|  (deliverable metric 1)')
    names = []
    for num, den, _ in rows:
        names.append(f'{num[0]}{num[1]}' if den is None
                     else f'{num[0]}{num[1]}/{den[0]}{den[1]}')
    header = f"  {'simulation':24s}" + ''.join(
        f'{f"{n} [{f}]":>22s}' for n in names for f in rp.FILTERS)
    emit(header)
    for run in runs:
        row = f"  {run['label']:24s}"
        for n in names:
            for f in rp.FILTERS:
                row += f'{metrics["max_dev"][(run["label"], n, f)]:22.4f}'
        emit(row)
    emit()

    subset_titles = {
        'cross-code': ('Scatter of the ratio across CODES  '
                       '(deliverable metric 2, the Gate A quantity)'),
        'all-runs': ('Scatter of the ratio across ALL runs  '
                     '(diagnostic: includes feedback variants within a code, '
                     'so this is NOT the Gate A statistic)'),
    }
    for subset_name in ('cross-code', 'all-runs'):
        if subset_name not in metrics['scatter']:
            continue
        per_filter = metrics['scatter'][subset_name]
        emit(subset_titles[subset_name])
        any_filt = next(iter(per_filter.values()))
        emit(f"  members ({len(any_filt['members'])}): "
             f"{', '.join(any_filt['members'])}")
        for filt, s in per_filter.items():
            emit(f'  {filt}:')
            emit(f"    {'R [arcmin]':>11}  {'mean':>10}  {'std':>10}  "
                 f"{'frac. std':>10}")
            for R, m, sd, fs, ind in zip(s['radii'], s['mean'], s['std'],
                                         s['frac_std'], s['in_data']):
                tag = '' if ind else '   (extension)'
                emit(f'    {R:11.3f}  {m:10.4f}  {sd:10.4f}  {fs:10.4f}{tag}')
            worst = s['worst_data']
            emit(f'    worst fractional scatter, data range: {worst:.4f}')
            if np.isfinite(s['worst_ext']):
                emit(f'    worst fractional scatter, extension: '
                     f"{s['worst_ext']:.4f}")
            if np.isfinite(worst) and subset_name == 'cross-code':
                if worst <= 0.10:
                    verdict = 'PASS  (<= 10%: fixed-transfer route)'
                elif worst <= 0.20:
                    verdict = 'MARGINAL  (10-20%: parametrized-r route)'
                else:
                    verdict = 'FAIL  (> 20%: revisit the estimator)'
                emit(f'    Gate A on this filter (data range): {verdict}')
            emit()

    emit('Note: apertures above the config max_radius are a diagnostic')
    emit('extension, not observationally accessible, and the Gate A verdict')
    emit('uses the data range only.')
    emit('Note: Upsilon is identically zero at R = R0 = 1 arcmin by')
    emit('construction, so its coefficient is undefined (NaN) in that bin and')
    emit('is excluded from both the curves and these metrics.')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f'\nMetrics written to: {out_path}')


def main(path2config, verbose=True):
    """Build the figures and metrics for one config.

    Args:
        path2config (str): Path to the YAML configuration file.
        verbose (bool, optional): Unused placeholder for CLI symmetry.
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
        raise SystemExit(
            f'No matching r_profiles_*.npz found in {npz_dir} for the runs in '
            f'{path2config}. Run make_r_profiles.py first.')
    if len(runs) < len(wanted):
        missing = wanted - {(r['sim_type'], r['sim_name'], r['feedback'],
                             r['snapshot']) for r in runs}
        print(f'WARNING: {len(missing)} configured run(s) have no .npz yet: '
              f'{sorted(missing)}')

    now = datetime.now()
    fig_dir = (Path(plot_cfg.get('fig_path', '../figures/'))
               / now.strftime('%Y-%m') / now.strftime('%m-%d'))

    make_figure(runs, BARYON_ROWS, fig_dir / f'{fig_name}_baryon.{fig_type}',
                'Baryon-matter cross-correlation coefficients '
                '(b = gas + stars + BH)')
    make_figure(runs, ELECTRON_ROWS,
                fig_dir / f'{fig_name}_tau.{fig_type}',
                'Tau-matter cross-correlation coefficients '
                '(e = ionized gas)')

    print()
    data_max = config.get('stack', {}).get('max_radius')
    metrics = gate_a_metrics(runs, BARYON_ROWS, data_max=data_max)
    report_metrics(metrics, runs, BARYON_ROWS,
                   fig_dir / f'{fig_name}_metrics.txt')

    print()
    metrics_e = gate_a_metrics(runs, ELECTRON_ROWS, data_max=data_max)
    report_metrics(metrics_e, runs, ELECTRON_ROWS,
                   fig_dir / f'{fig_name}_metrics_electron.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Task 1 r-profiles and report the Gate A metrics.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/r_profiles_z05.yaml',
                        help='Path to the YAML configuration file.')
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
