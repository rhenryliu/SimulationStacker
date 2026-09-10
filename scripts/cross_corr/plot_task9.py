"""plot_task9.py
==============
Task 9 of ``docs/cross_correlation_notes_v0.2_addendum.md``: the figure that
shows what the calibration factor is worth.

Per filter, a two-panel column:

**Upper** -- three curves against aperture:

    (i)   the truth,     Y_bm / Y_mm            measured directly from the maps
    (ii)  Route A, C_A=1, Y_gb / sqrt(Y_gg Y_mm)        (addendum Eq. A10)
    (iii) Route B, C=1,   Y_gb / Y_gm                   (addendum Eq. A11)

The gap between (i) and each of (ii), (iii) *is* the calibration factor, so
this single panel answers both "what is the correction" and "how wrong is the
uncorrected observable".

**Lower** -- the same three converted to suppression through (A16),
``S = (f_m + f_b x)^2``, overlaid with the directly measured
``P_tt(k) / P_mm(k)`` of the same box, and plotted against ``k_50(R;F)`` so
that filters are compared at matched wavenumber rather than matched aperture.
``k_50`` and the measured spectra come from ``make_task9_spectra.py``.

House style, per the addendum's closing note on the many-line problem: two to
three lines per panel, one panel per filter, the FLAMINGO feedback variants as
a shaded envelope with the fiducial as the exemplar, and TNG300-1 as a thin
dotted cross-code witness.

Usage
-----
    cd scripts/
    python cross_corr/plot_task9.py -p configs/cross_corr/calibration_z05.yaml
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
sys.path.append(str(Path(__file__).resolve().parent))

from make_calibration_factor import suppression

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    # Computer Modern rather than dejavuserif: in dejavuserif the
    # \Upsilon macro renders pixel-identically to an upright Latin Y,
    # which makes the Baldauf filter indistinguishable from the Park
    # et al. Y transform and from the Y_ab amplitudes.  'cm' draws the
    # forked Upsilon.
    'mathtext.fontset': 'cm',
    'text.usetex': False,      # no LaTeX in the cosmodesi environment
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 9,
})

#: Filters shown, in the addendum's Task 9 order, with display labels.
PANEL_FILTERS = [
    ('Sigma', r'$\Sigma$'),
    ('DSigma', r'$\Delta\Sigma$'),
    ('Upsilon_R0=1', r"$\Upsilon(R_0=1')$"),
    ('Ytransform_Rmax=5', r"$Y(R_{\max}=5')$"),
]

#: The three estimator curves: key, label, colour.
CURVES = [
    ('truth', r'truth  $Y_{bm}/Y_{mm}$', '#222222'),
    ('routeA', r'Route A, $C_A{=}1$', '#c1272d'),
    ('routeB', r'Route B, $C{=}1$', '#0072b2'),
]

#: Observationally accessible aperture range, arcmin.
DATA_RANGE = (1.0, 6.0)


def estimator_curves(npz, filt):
    """Extract the three estimator curves for one run and filter.

    Args:
        npz (np.lib.npyio.NpzFile): One ``calibration_*.npz``.
        filt (str): Filter variant name.

    Returns:
        dict: ``{'truth', 'routeA', 'routeB'}`` each an array over apertures,
        NaN outside the filter's mask.  Returns None if the filter is absent.
    """
    if f'Y_mm_{filt}' not in npz.files:
        return None
    mask = npz[f'mask_{filt}']
    Y_mm = npz[f'Y_mm_{filt}']
    Y_bm = npz[f'Y_bm_{filt}']
    Y_gb = npz[f'Y_bg_{filt}']
    Y_gm = npz[f'Y_gm_{filt}']
    Y_gg = npz[f'Y_gg_{filt}']

    with np.errstate(invalid='ignore', divide='ignore'):
        truth = np.where(Y_mm != 0, Y_bm / Y_mm, np.nan)
        # Route A needs sqrt(Y_gg Y_mm); compensated filters can drive either
        # auto negative, in which case the estimator is genuinely undefined.
        denom_a = Y_gg * Y_mm
        routeA = np.where(denom_a > 0, Y_gb / np.sqrt(denom_a), np.nan)
        routeB = np.where(Y_gm != 0, Y_gb / Y_gm, np.nan)

    out = {}
    for key, values in (('truth', truth), ('routeA', routeA),
                        ('routeB', routeB)):
        v = np.asarray(values, dtype=np.float64).copy()
        v[~mask] = np.nan
        out[key] = v
    return out


def load_runs(npz_dir, snapshots):
    """Load the calibration and spectrum files for one redshift sample.

    Args:
        npz_dir (pathlib.Path): Directory holding both npz families.
        snapshots (set): Snapshot numbers belonging to this sample.

    Returns:
        list: One dict per run, with 'label', 'is_fiducial', 'is_tng', 'cal',
        'spec', 'f_b'.
    """
    runs = []
    for path in sorted(glob.glob(str(npz_dir / 'calibration_*.npz'))):
        cal = np.load(path, allow_pickle=True)
        snapshot = int(cal['meta_snapshot'])
        if snapshot not in snapshots:
            continue
        label = str(cal['meta_label'])
        spec_path = npz_dir / (f'task9_spectra_{label}_{snapshot}_'
                               f"{str(cal['meta_projection'])}.npz")
        if not spec_path.exists():
            warnings.warn(
                f'No spectra for {label} (expected {spec_path.name}); run '
                'make_task9_spectra.py first. Skipping this run.',
                stacklevel=2)
            continue
        sim_type = str(cal['meta_sim_type'])
        feedback = str(cal['meta_feedback'])
        sim_name = str(cal['meta_sim_name'])
        runs.append({
            'label': label,
            'sim_type': sim_type,
            'is_tng': sim_type == 'IllustrisTNG',
            'is_fiducial': (feedback == sim_name if sim_type == 'FLAMINGO'
                            else feedback in ('None', '')),
            'cal': cal,
            'spec': np.load(spec_path, allow_pickle=True),
            'f_b': float(cal['meta_f_b']),
        })
    return runs


def envelope(values):
    """Return the elementwise min and max across a list of curves.

    Args:
        values (list): Arrays of identical shape.

    Returns:
        tuple: ``(lo, hi)``, NaN-safe.
    """
    stack = np.vstack(values)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)


def make_figure(runs, out_path, title):
    """Draw the Task 9 two-row figure.

    Args:
        runs (list): Output of :func:`load_runs`.
        out_path (pathlib.Path): Destination file.
        title (str): Figure suptitle.

    Returns:
        pathlib.Path: The path written.
    """
    flamingo = [r for r in runs if not r['is_tng']]
    fiducial = next((r for r in flamingo if r['is_fiducial']),
                    flamingo[0] if flamingo else None)
    tng = next((r for r in runs if r['is_tng']), None)
    if fiducial is None:
        raise ValueError('No FLAMINGO run found; cannot draw the envelope.')

    n_col = len(PANEL_FILTERS)
    fig, axes = plt.subplots(2, n_col, figsize=(4.3 * n_col, 8.0),
                             squeeze=False)

    radii = fiducial['cal']['radii']
    in_range = (radii >= DATA_RANGE[0]) & (radii <= DATA_RANGE[1])

    for col, (filt, flabel) in enumerate(PANEL_FILTERS):
        ax_top, ax_bot = axes[0][col], axes[1][col]
        fid_curves = estimator_curves(fiducial['cal'], filt)
        if fid_curves is None:
            for ax in (ax_top, ax_bot):
                ax.text(0.5, 0.5, f'{flabel}\nnot available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            continue

        # Carry each run alongside its curves.  Filtering a separate list would
        # shift the alignment as soon as one variant lacks this filter (a stale
        # cache from a selective --sim/--feedback rerun, say), and the lower
        # panel would then use the wrong run's f_b with no error raised.
        variant_pairs = [(estimator_curves(r['cal'], filt), r)
                         for r in flamingo]
        variant_pairs = [(v, r) for v, r in variant_pairs if v is not None]
        variants = [v for v, _ in variant_pairs]
        tng_curves = estimator_curves(tng['cal'], filt) if tng else None

        f_b = fiducial['f_b']
        kq = fiducial['spec'][f'kquant_hmpc_{filt}'][:, 1]      # k_50, h/Mpc

        for key, label, colour in CURVES:
            y = fid_curves[key]
            ax_top.plot(radii[in_range], y[in_range], color=colour, lw=2.0,
                        label=label, zorder=3)
            if len(variants) > 1:
                lo, hi = envelope([v[key] for v in variants])
                ax_top.fill_between(radii[in_range], lo[in_range],
                                    hi[in_range], color=colour, alpha=0.18,
                                    lw=0, zorder=1)
            if tng_curves is not None:
                ax_top.plot(radii[in_range], tng_curves[key][in_range],
                            color=colour, lw=1.1, ls=':', zorder=2)

            # Lower panel: the same three, mapped to suppression, against k_50.
            s = suppression(y, f_b, 'C')
            order = np.argsort(kq)
            sel = order[in_range[order]]
            ax_bot.plot(kq[sel], s[sel], color=colour, lw=2.0, label=label,
                        zorder=3)
            if len(variants) > 1:
                lo, hi = envelope([suppression(v[key], r['f_b'], 'C')
                                   for v, r in variant_pairs])
                ax_bot.fill_between(kq[sel], lo[sel], hi[sel], color=colour,
                                    alpha=0.18, lw=0, zorder=1)
            if tng_curves is not None:
                s_tng = suppression(tng_curves[key], tng['f_b'], 'C')
                k_tng = tng['spec'][f'kquant_hmpc_{filt}'][:, 1]
                o = np.argsort(k_tng)
                st = o[in_range[o]]
                ax_bot.plot(k_tng[st], s_tng[st], color=colour, lw=1.1,
                            ls=':', zorder=2)

        # The unfiltered suppression of the same box: the thing the filtered
        # estimates are supposed to reproduce (addendum Sec. 5.5, rung 4).
        # The window is set by the filter's own k_50 range, padded, so that the
        # comparison is legible: showing the full measured spectrum would
        # compress the estimator curves into a tenth of the axis.
        k_spec = fiducial['spec']['k_h_mpc']
        ratio = fiducial['spec']['P2D_tt'] / fiducial['spec']['P2D_mm']
        finite_kq = kq[in_range][np.isfinite(kq[in_range])]
        if finite_kq.size:
            k_lo, k_hi = 0.25 * finite_kq.min(), 4.0 * finite_kq.max()
        else:
            k_lo, k_hi = 0.1, 30.0
        show = (k_spec > k_lo) & (k_spec < k_hi)
        ax_bot.set_xlim(k_lo, k_hi)
        ax_bot.plot(k_spec[show], ratio[show], color='#009e73', lw=1.6,
                    ls='--', zorder=4, label=r'measured $P_{tt}/P_{mm}$')
        if len(flamingo) > 1:
            lo, hi = envelope([np.interp(k_spec, r['spec']['k_h_mpc'],
                                         r['spec']['P2D_tt']
                                         / r['spec']['P2D_mm'])
                               for r in flamingo])
            ax_bot.fill_between(k_spec[show], lo[show], hi[show],
                                color='#009e73', alpha=0.15, lw=0, zorder=0)

        ax_top.set_title(flabel)
        ax_top.set_xlabel(r"$R$ [arcmin]")
        ax_bot.set_xlabel(r'$k_{50}(R;\mathcal{F})$  [$h\,$Mpc$^{-1}$]')
        ax_bot.set_xscale('log')
        ax_top.axhline(1.0, color='0.6', lw=0.8, ls='-', zorder=0)
        ax_bot.axhline(1.0, color='0.6', lw=0.8, ls='-', zorder=0)
        ax_top.grid(alpha=0.25)
        ax_bot.grid(alpha=0.25)
        if col == 0:
            ax_top.set_ylabel(r'$Y_{bm}/Y_{mm}$  (and its estimators)')
            ax_bot.set_ylabel(r'$S = (f_m + f_b x)^2$')
            ax_top.legend(frameon=False, loc='best')
            ax_bot.legend(frameon=False, loc='best')

    fig.suptitle(title, y=0.995)
    fig.text(0.5, 0.005,
             'solid + band: FLAMINGO fiducial with the feedback-variant '
             'envelope   |   dotted: TNG300-1',
             ha='center', fontsize=10, color='0.35')
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def report_metrics(runs):
    """Print the numbers the figure is meant to convey.

    Args:
        runs (list): Output of :func:`load_runs`.
    """
    print('\nA/truth and B/truth are the calibration factors the estimator '
          'needs (1 = no correction).')
    print('smear = S(truth)/[P_tt/P_mm at k_50] - 1 is the window-smearing '
          'residual: addendum Sec. 5.5 rung 3,')
    print('       i.e. how well the real-space ratio can be read as a k-space '
          'ratio at matched wavenumber.')
    print(f"\n{'filter':20s} {'run':22s} {'A/truth':>9s} {'B/truth':>9s} "
          f"{'S truth':>9s} {'S routeB':>9s} {'P_tt/P_mm@k50':>14s} "
          f"{'smear':>8s}")
    for filt, _ in PANEL_FILTERS:
        for run in runs:
            c = estimator_curves(run['cal'], filt)
            if c is None:
                continue
            radii = run['cal']['radii']
            sel = (radii >= DATA_RANGE[0]) & (radii <= DATA_RANGE[1])
            f_b = run['f_b']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                a = np.nanmean(c['routeA'][sel] / c['truth'][sel])
                b = np.nanmean(c['routeB'][sel] / c['truth'][sel])
                s_t = np.nanmean(suppression(c['truth'][sel], f_b, 'C'))
                s_b = np.nanmean(suppression(c['routeB'][sel], f_b, 'C'))
                kq = run['spec'][f'kquant_hmpc_{filt}'][:, 1][sel]
                ratio = np.interp(
                    kq, run['spec']['k_h_mpc'],
                    run['spec']['P2D_tt'] / run['spec']['P2D_mm'])
                direct = np.nanmean(ratio)
            smear = s_t / direct - 1.0 if np.isfinite(direct) else np.nan
            print(f'{filt:20s} {run["label"][:22]:22s} {a:9.3f} {b:9.3f} '
                  f'{s_t:9.3f} {s_b:9.3f} {direct:14.3f} {smear:+8.3f}')


def main(path2config, verbose=True):
    """Draw the Task 9 figure for the sample named by a config.

    Args:
        path2config (str): Path to the YAML configuration file.
        verbose (bool, optional): Print metrics.  Defaults to True.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)
    plot_cfg = config.get('plot', {})
    npz_dir = Path(plot_cfg.get('npz_path', '../data/cross_corr_C/'))
    snapshots = {int(e['snapshot']) for s in config['simulations']
                 for e in s['sims']}

    runs = load_runs(npz_dir, snapshots)
    if not runs:
        raise SystemExit(
            f'No runs found in {npz_dir} for snapshots {sorted(snapshots)}. '
            'Run make_calibration_factor.py and make_task9_spectra.py first.')

    z_mean = np.mean([float(r['cal']['meta_redshift']) for r in runs])
    now = datetime.now()
    fig_dir = (Path(plot_cfg.get('fig_path', '../figures/'))
               / now.strftime('%Y-%m') / now.strftime('%m-%d'))
    name = plot_cfg.get('fig_name', 'calibration')
    out_path = fig_dir / f'task9_{name}.png'

    make_figure(runs, out_path,
                f'Task 9: the calibration factor as a gap between curves '
                f'($z \\approx {z_mean:.2f}$)')
    print(f'Wrote {out_path}')
    if verbose:
        report_metrics(runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Draw the Task 9 estimator/suppression figure.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/calibration_z05.yaml')
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
