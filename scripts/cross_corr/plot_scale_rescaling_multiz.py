"""plot_scale_rescaling_multiz.py
==============================
The comoving versus physical scale test of ``x_F(R) = Y_bm / Y_mm`` at every
redshift slice on disk: z ~ 0.26 (FLAMINGO 0.30), 0.5, 0.75 and 1.0.  It
extends ``plot_scale_rescaling.py``, which compares two of them, to the
snapshots listed per run in the config.

For each run the curves are plotted against arcmin, comoving Mpc/h and
physical Mpc/h, and the one-parameter family ``s = R_com (1+z)^beta`` is
scanned for the rescaling under which they overlap best (beta = 0 comoving,
beta = -1 physical), as in the two-snapshot script.  The curve extraction,
jackknife, beta search and non-linear scale are imported from it unchanged.

Generalization to N snapshots (user, 2026-10-05):

- **Overlap metric.**  The rms of ``x_i - x_j`` over all N(N-1)/2 snapshot
  pairs, on a log-spaced grid across the interval of s that *every* curve
  covers.  For N = 2 this is exactly the two-snapshot metric.  For any N it
  equals ``sqrt(2N/(N-1))`` times the rms scatter about the mean curve (the
  population rms, divisor N), so it measures how universal the whole set is,
  not one pair.
- **Comparison window.**  The common interval moves with beta, towards larger
  s (where x flattens towards 1) as beta rises, so rms values at different
  candidates are not over the same range of scales.  The report prints the
  span ``s_hi / s_lo`` of the window for each candidate.
- **Jackknife.**  Formed per realization, pairing the realizations of all
  snapshots patch for patch: every snapshot uses the same 4x4 grid on the
  same yz projection of the same box.
- **Equivalent beta of a candidate scale** ``R_*(z)``: minus the least-squares
  slope of ``ln R_*`` against ``ln(1+z)`` over the snapshots, which is the
  two-point formula for N = 2.  Arcmin, ``R_200c`` and ``R_nl`` are not exact
  power laws in ``1+z``, so the largest residual of that fit is printed too.
- **Arcmin.**  Its row of the figure, and its overlap rms in the report, use
  the apertures in arcmin directly.  With more than two redshifts no single
  beta maps arcmin onto a common rescaling of s, so the arcmin beta is only
  reported.
- **Range of beta.**  The interval every curve covers shrinks quickly with
  beta across z = 0.26-1.0.  At the R_nl beta (~1.4) it spans less than
  ``min_overlap_factor``, so the report prints NaN for that candidate.

Caveat: the Delta Sigma annulus width is fixed at 0.75 arcmin, so in comoving
units it grows with redshift: 0.16 (TNG300-1, z = 0.26) or 0.18 (FLAMINGO,
z = 0.30), 0.29 (z ~ 0.5), 0.40-0.41 (z ~ 0.75) and 0.50 cMpc/h (z ~ 1.0), a
factor of ~3 across the set against ~1.7 for the two-snapshot figure.  Even an
exactly universal P_bm/P_mm would not make the aperture curves coincide in any
single coordinate, and the larger spread here makes that limit tighter.  The
report prints the width per snapshot.

Usage
-----
    cd scripts/
    python cross_corr/plot_scale_rescaling_multiz.py \
        -p configs/cross_corr/scale_rescaling_multiz.yaml
"""

import argparse
import glob
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import yaml

sys.path.append('../src/')
sys.path.append(str(Path(__file__).resolve().parent))

from stacker import SimulationStacker
import theory as th

# Reused verbatim so the curve extraction, jackknife, beta search and R_nl
# cannot drift from the two-snapshot figure.  Importing the module also
# applies its rcParams.
from plot_scale_rescaling import (FILTER_LABELS, ROWS, RUN_TITLES, best_beta,
                                  coordinate, jk_sigma, nonlinear_radius,
                                  snapshot_curve)

#: Line colour per redshift slot, lowest z (light) to highest (dark): one
#: blue ramp, validated as an ordinal palette on a white surface.
Z_COLOURS = ('#86b6ef', '#3987e5', '#1c5cab', '#0d366b')

#: Line style and marker per redshift slot, in the same order, so the
#: snapshots stay distinguishable without colour.
Z_LINESTYLES = (':', '-.', '--', '-')
Z_MARKERS = ('o', 's', '^', 'D')

#: Candidate coordinates in report order.
CANDIDATES = ('arcmin', 'physical', 'R200c', 'comoving', 'R_nl', 'best')

#: Suite names in the shared legend, where a slot's redshift differs by suite.
SUITE_NAMES = {'IllustrisTNG': 'TNG', 'FLAMINGO': 'FLAMINGO'}


def load_runs(npz_dir, runs_cfg, filt, radius_range):
    """Load every listed snapshot of every requested run.

    Args:
        npz_dir (pathlib.Path): Directory holding ``calibration_*.npz``.
        runs_cfg (list): Config entries ``{'label', 'snapshots'}``, in
            plotting order.
        filt (str): Filter variant name.
        radius_range (sequence): ``(min, max)`` apertures to keep, arcmin.

    Returns:
        list: One dict per run with ``label``, ``meta`` (the highest-z file)
        and ``snaps``, the snapshot curves in order of decreasing redshift.

    Raises:
        SystemExit: If fewer than two snapshots are listed, or a listed
            snapshot does not have exactly one file on disk.
    """
    found = {}
    for path in sorted(glob.glob(str(npz_dir / 'calibration_*.npz'))):
        cal = np.load(path, allow_pickle=True)
        key = (str(cal['meta_label']), int(cal['meta_snapshot']))
        found.setdefault(key, []).append(cal)

    runs = []
    for entry in runs_cfg:
        label = entry['label']
        snapshots = sorted({int(s) for s in entry['snapshots']})
        if len(snapshots) < 2:
            raise SystemExit(f'{label}: list at least two snapshots.')
        cals = []
        for snapshot in snapshots:
            hits = found.get((label, snapshot), [])
            if not hits:
                raise SystemExit(
                    f'{label} snapshot {snapshot}: no file in {npz_dir}. '
                    'Run make_calibration_factor.py first.')
            if len(hits) > 1:
                raise SystemExit(
                    f'{label} snapshot {snapshot}: {len(hits)} files in '
                    f'{npz_dir} (one per projection?); expected 1.')
            cals.append(hits[0])
        cals.sort(key=lambda c: float(c['meta_redshift']), reverse=True)
        runs.append({
            'label': label,
            'meta': cals[0],
            'snaps': [snapshot_curve(c, filt, radius_range) for c in cals],
        })
    return runs


def snap_coordinate(snap, beta):
    """Return the apertures in arcmin if ``beta`` is None, else ``s``."""
    return snap['radii'] if beta is None else coordinate(snap, beta)


def overlap_rms(curves, n_grid, min_factor):
    """rms difference over all pairs of curves, on the interval all cover.

    Args:
        curves (list): ``(s, x)`` per snapshot, ``s`` increasing.
        n_grid (int): Log-spaced comparison points.
        min_factor (float): Minimum ``s_max / s_min`` of the overlap.

    Returns:
        tuple: ``(rms, s_lo, s_hi)``; ``rms`` is NaN if the common overlap
        is too short or any curve is non-finite there.
    """
    lo = max(s[0] for s, _ in curves)
    hi = min(s[-1] for s, _ in curves)
    if not hi >= lo * min_factor:
        return np.nan, lo, hi
    ln_grid = np.linspace(np.log(lo), np.log(hi), n_grid)
    xs = [np.interp(ln_grid, np.log(s), x) for s, x in curves]
    diff = np.array([xs[i] - xs[j]
                     for i, j in combinations(range(len(xs)), 2)])
    return float(np.sqrt(np.mean(diff ** 2))), lo, hi


def mismatch(run, beta, cfg, jk=None):
    """Overlap rms of all snapshots of one run at one beta.

    Args:
        run (dict): One entry of :func:`load_runs`.
        beta (float or None): Rescaling exponent; None for arcmin.
        cfg (dict): Configuration, for the overlap settings.
        jk (int, optional): Jackknife realization; None for the full sample.

    Returns:
        float: The all-pairs rms of ``x_i - x_j``.
    """
    curves = [(snap_coordinate(snap, beta),
               snap['x'] if jk is None else snap['x_jk'][jk])
              for snap in run['snaps']]
    return overlap_rms(curves, int(cfg.get('n_overlap_grid', 50)),
                       float(cfg.get('min_overlap_factor', 2.0)))[0]


def equivalent_beta(scales, zs):
    """Least-squares beta for which ``R_com (1+z)^beta ∝ R_com / R_*(z)``.

    Args:
        scales (sequence): A comoving scale ``R_*`` at each redshift.
        zs (sequence): The redshifts.

    Returns:
        tuple: ``(beta, max_resid)``: the exponent (-1 for a fixed physical
        scale) and the largest ``|ln R_*|`` residual about the fitted power
        law, zero for two snapshots.
    """
    ln_a = np.log(1.0 + np.asarray(zs, dtype=np.float64))
    ln_r = np.log(np.asarray(scales, dtype=np.float64))
    slope, intercept = np.polyfit(ln_a, ln_r, 1)
    resid = ln_r - (intercept + slope * ln_a)
    return -float(slope), float(np.max(np.abs(resid)))


_RNL_CACHE = {}


def candidate_betas(run):
    """Equivalent beta of every candidate coordinate for one run.

    ``R_200c`` uses the header cosmology through ``stacker.cosmo``; ``R_nl``
    uses CAMB linear P(k) with n_s and sigma8 from the literature table in
    :data:`theory.SIMULATION_COSMOLOGIES`, as in the two-snapshot script.

    Args:
        run (dict): One entry of :func:`load_runs`.

    Returns:
        tuple: ``(cand, resid)``.  ``cand`` maps 'arcmin', 'physical',
        'comoving', 'R200c' and 'R_nl' to beta, plus ``R_nl_z``, R_nl in
        cMpc/h per snapshot.  ``resid`` maps 'arcmin', 'R200c' and 'R_nl' to
        the largest power-law residual of :func:`equivalent_beta`.
    """
    snaps = run['snaps']
    meta = run['meta']
    feedback = meta['meta_feedback'].item()
    if feedback in (None, 'None', ''):
        feedback = None
    sim_type = str(meta['meta_sim_type'])
    stacker = SimulationStacker(str(meta['meta_sim_name']),
                                snaps[0]['snapshot'],
                                nPixels=int(meta['meta_n_pixels']),
                                simType=sim_type, feedback=feedback,
                                z=snaps[0]['z'])

    def r200c_comoving(z):
        # Comoving R_200c at fixed mass, up to a constant.
        return (1.0 + z) * stacker.cosmo.efunc(z) ** (-2.0 / 3.0)

    params = th.cosmology_for(sim_type, stacker.header)

    def r_nl(z):
        key = (tuple(sorted(params.items())), round(z, 6))
        if key not in _RNL_CACHE:
            k, pk = th.halofit_power(z=z, non_linear=False, **params)
            _RNL_CACHE[key] = nonlinear_radius(k, pk)
        return _RNL_CACHE[key]

    zs = [snap['z'] for snap in snaps]
    cand = {'physical': -1.0, 'comoving': 0.0}
    resid = {}
    cand['arcmin'], resid['arcmin'] = equivalent_beta(
        [snap['mpc_per_arcmin'] for snap in snaps], zs)
    cand['R200c'], resid['R200c'] = equivalent_beta(
        [r200c_comoving(z) for z in zs], zs)
    cand['R_nl_z'] = [r_nl(z) for z in zs]
    cand['R_nl'], resid['R_nl'] = equivalent_beta(cand['R_nl_z'], zs)
    return cand, resid


def analyse(run, cfg):
    """Scan beta, fit beta_hat with jackknife errors, and evaluate candidates.

    Adds ``betas``, ``scan``, ``beta_hat``, ``beta_err``, ``beta_on_edge``,
    ``cand``, ``resid``, ``D`` (candidate -> (rms, jackknife error)) and
    ``window`` (candidate -> ``s_hi / s_lo`` of the common interval) to
    ``run``.

    Args:
        run (dict): One entry of :func:`load_runs`.
        cfg (dict): Configuration.

    Raises:
        ValueError: If the snapshots have different numbers of jackknife
            realizations.
    """
    betas = np.arange(float(cfg.get('beta_min', -4.0)),
                      float(cfg.get('beta_max', 4.0)) + 1e-9,
                      float(cfg.get('beta_step', 0.01)))
    n_jk = run['snaps'][0]['x_jk'].shape[0]
    if any(snap['x_jk'].shape[0] != n_jk for snap in run['snaps']):
        raise ValueError(f"{run['label']}: the snapshots have different "
                         'numbers of jackknife realizations.')

    run['betas'] = betas
    run['scan'] = np.array([mismatch(run, b, cfg) for b in betas])
    beta_hat, _, on_edge = best_beta(lambda b: mismatch(run, b, cfg), betas)
    # best_beta flags only the ends of the grid.  A minimum next to a beta
    # where the common overlap is too short (NaN beyond it) is an edge too.
    if np.any(np.isfinite(run['scan'])):
        i = int(np.nanargmin(run['scan']))
        neighbours = run['scan'][[max(i - 1, 0), min(i + 1, len(betas) - 1)]]
        on_edge = on_edge or not np.all(np.isfinite(neighbours))
    beta_jk = np.array([best_beta(lambda b: mismatch(run, b, cfg, jk=i),
                                  betas)[0] for i in range(n_jk)])
    run['beta_hat'] = beta_hat
    run['beta_err'] = float(jk_sigma(beta_jk))
    run['beta_on_edge'] = on_edge

    cand, resid = candidate_betas(run)
    cand['best'] = beta_hat
    run['cand'] = cand
    run['resid'] = resid
    run['D'] = {}
    run['window'] = {}
    for key in CANDIDATES:
        beta = None if key == 'arcmin' else cand[key]
        d_jk = [mismatch(run, beta, cfg, jk=i) for i in range(n_jk)]
        run['D'][key] = (mismatch(run, beta, cfg), float(jk_sigma(d_jk)))
        _, s_lo, s_hi = overlap_rms(
            [(snap_coordinate(snap, beta), snap['x']) for snap in run['snaps']],
            2, 0.0)
        run['window'][key] = s_hi / s_lo


def z_styles(n):
    """Colour, line style and marker per snapshot, highest redshift first.

    Args:
        n (int): Number of snapshots.

    Returns:
        list: ``(colour, linestyle, marker)`` for each snapshot in order of
        decreasing redshift, taken from the dark end of the ramp.

    Raises:
        ValueError: If there are more snapshots than redshift slots.
    """
    if n > len(Z_COLOURS):
        raise ValueError(f'{n} snapshots but only {len(Z_COLOURS)} redshift '
                         'styles are defined.')
    slots = range(len(Z_COLOURS) - 1, len(Z_COLOURS) - 1 - n, -1)
    return [(Z_COLOURS[i], Z_LINESTYLES[i], Z_MARKERS[i]) for i in slots]


def slot_labels(runs):
    """Legend label per redshift slot, shared by runs with equal snapshot counts.

    Runs from different suites can sit at slightly different redshifts in
    the same slot (TNG300-1 z = 0.26 against FLAMINGO z = 0.30), so a slot
    whose values differ names each suite's.

    Args:
        runs (list): Runs with the same number of snapshots.

    Returns:
        list: One label per slot, highest redshift first.
    """
    labels = []
    for k in range(len(runs[0]['snaps'])):
        by_z = {}
        for run in runs:
            sim_type = str(run['meta']['meta_sim_type'])
            by_z.setdefault(f"{run['snaps'][k]['z']:.2f}", []).append(
                SUITE_NAMES.get(sim_type, sim_type))
        if len(by_z) == 1:
            labels.append(f'$z$ = {next(iter(by_z))}')
        else:
            labels.append('$z$ = ' + ', '.join(
                f"{z} ({'/'.join(dict.fromkeys(suites))})"
                for z, suites in by_z.items()))
    return labels


def make_figure(runs, cfg, out_path, filt):
    """x_F at every redshift against each coordinate, one column per run.

    Same layout as the two-snapshot figure: each coordinate row is its own
    subfigure with one x label under it.  The shaded band is the interval
    every curve covers, over which the all-pairs rms is taken.  When every
    run has the same number of snapshots, one redshift legend sits above the
    top row, since four entries inside a panel collide with the curves.

    Args:
        runs (list): Analysed runs.
        cfg (dict): Configuration.
        out_path (pathlib.Path): Output PNG path.
        filt (str): Filter variant name, for the title.
    """
    n_runs = len(runs)
    fig = plt.figure(figsize=(3.7 * n_runs, 3.3 * len(ROWS)),
                     constrained_layout=True)
    subfigs = fig.subfigures(len(ROWS), 1, hspace=0.08)

    xs = np.concatenate([snap['x'] for run in runs for snap in run['snaps']])
    pad = 0.04 * (np.nanmax(xs) - np.nanmin(xs))
    ylim = (np.nanmin(xs) - pad, np.nanmax(xs) + pad)
    n_grid = int(cfg.get('n_overlap_grid', 50))
    min_factor = float(cfg.get('min_overlap_factor', 2.0))
    shared_legend = len({len(run['snaps']) for run in runs}) == 1

    for row, (subfig, (kind, xlabel)) in enumerate(zip(subfigs, ROWS)):
        axes = subfig.subplots(1, n_runs, sharex=True, sharey=True,
                               squeeze=False)[0]
        for col, (ax, run) in enumerate(zip(axes, runs)):
            beta = None if kind == 'arcmin' else run['cand'][kind]
            curves = []
            styles = z_styles(len(run['snaps']))
            for snap, (colour, ls, marker) in zip(run['snaps'], styles):
                s = snap_coordinate(snap, beta)
                curves.append((s, snap['x']))
                err = jk_sigma(snap['x_jk'])
                ax.fill_between(s, snap['x'] - err, snap['x'] + err,
                                color=colour, alpha=0.25, lw=0)
                ax.plot(s, snap['x'], color=colour, ls=ls, lw=1.6,
                        marker=marker, ms=3.0,
                        label=f"$z = {snap['z']:.2f}$")
            rms, s_lo, s_hi = overlap_rms(curves, n_grid, min_factor)
            ax.axvspan(s_lo, s_hi, color='0.92', zorder=0, lw=0)
            ax.text(0.97, 0.05, rf'rms $\Delta x$ = {rms:.4f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=9)
            ax.set_xscale('log')
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.25, lw=0.5)
            if row == 0:
                ax.set_title(RUN_TITLES.get(run['label'], run['label']))
                if not shared_legend:
                    ax.legend(loc='upper left', frameon=False)
            elif kind == 'comoving' and col == 0:
                # Here the shading does not fill the panel, so the key shows.
                ax.legend(handles=[Patch(facecolor='0.92', edgecolor='0.6',
                                         label='range of rms')],
                          loc='upper left', frameon=False)
        axes[0].set_ylabel(r'$Y_{bm}/Y_{mm}$')
        subfig.supxlabel(xlabel)

    if shared_legend:
        n = len(runs[0]['snaps'])
        handles = [Line2D([], [], color=colour, ls=ls, lw=1.6, marker=marker,
                          ms=4.0) for colour, ls, marker in z_styles(n)]
        subfigs[0].legend(handles, slot_labels(runs),
                          loc='outside upper center', ncol=n, frameon=False,
                          fontsize=10)

    fig.suptitle(rf'$Y_{{bm}}/Y_{{mm}}$ ({FILTER_LABELS.get(filt, filt)}); '
                 r'rms $\Delta x$ over all redshift pairs', fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def report(runs):
    """Per-run candidates, their overlap rms and the fitted beta.

    Args:
        runs (list): Analysed runs.

    Returns:
        list: Lines of the report.
    """
    lines = []
    lines.append('Equivalent beta of each coordinate, s = R_com (1+z)^beta '
                 '(0 comoving, -1 physical); least squares over the snapshots')
    lines.append(f"{'run':24s} {'z (high to low)':>24s} "
                 + ' '.join(f'{k:>9s}' for k in CANDIDATES))
    for run in runs:
        zs = '/'.join(f"{snap['z']:.3f}" for snap in run['snaps'])
        lines.append(f"{run['label']:24s} {zs:>24s} "
                     + ' '.join(f"{run['cand'][k]:9.3f}" for k in CANDIDATES))

    lines.append('')
    lines.append('Largest |ln R_*| residual about the power law in (1+z) '
                 'behind each equivalent beta')
    for run in runs:
        lines.append(f"  {run['label']:24s} "
                     + '  '.join(f'{k} {v:.4f}'
                                 for k, v in run['resid'].items()))

    lines.append('')
    lines.append('Delta Sigma annulus width (fixed in arcmin) in cMpc/h, '
                 'per snapshot (high z to low)')
    for run in runs:
        dr = float(run['meta']['meta_dr_arcmin'])
        lines.append(f"  {run['label']:24s} "
                     + '  '.join(f"z={snap['z']:.3f}: "
                                 f"{dr * snap['mpc_per_arcmin']:.3f}"
                                 for snap in run['snaps']))

    lines.append('')
    lines.append('Overlap rms of x_i - x_j over all snapshot pairs '
                 '(jackknife error), per coordinate')
    lines.append(f"{'run':24s} " + ' '.join(f'{k:>17s}' for k in CANDIDATES))
    for run in runs:
        lines.append(f"{run['label']:24s} "
                     + ' '.join(f"{run['D'][k][0]:8.4f}({run['D'][k][1]:.4f})"
                                for k in CANDIDATES))

    lines.append('')
    lines.append('Span s_hi/s_lo of the common window each rms is taken over '
                 '(below min_overlap_factor -> NaN rms)')
    lines.append(f"{'run':24s} " + ' '.join(f'{k:>17s}' for k in CANDIDATES))
    for run in runs:
        lines.append(f"{run['label']:24s} "
                     + ' '.join(f"{run['window'][k]:17.2f}"
                                for k in CANDIDATES))

    lines.append('')
    lines.append('Best-fit beta (jackknife error); R_nl in cMpc/h per snapshot '
                 '(sigma_lin = 1, literature n_s/sigma8), high z to low')
    for run in runs:
        edge = '  ** at the edge of the scan **' if run['beta_on_edge'] else ''
        rnl = ' / '.join(f'{r:.3f}' for r in run['cand']['R_nl_z'])
        lines.append(f"  {run['label']:24s} beta_hat = {run['beta_hat']:+.3f} "
                     f"+/- {run['beta_err']:.3f}   R_nl = {rnl}{edge}")

    # One beta for every run: minimize the summed squared rms.
    betas = runs[0]['betas']
    total = np.sum([run['scan'] ** 2 for run in runs], axis=0)
    if np.any(np.isfinite(total)):
        i = int(np.nanargmin(total))
        lines.append('')
        lines.append(f'  common beta over all runs (grid, step '
                     f"{betas[1] - betas[0]:.2f}): {betas[i]:+.2f}")
    return lines


def main(path2config):
    """Draw the multi-redshift scale-rescaling figure and write the report.

    Args:
        path2config (str): Path to the YAML configuration file.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)
    plot_cfg = config.get('plot', {})
    filt = config.get('filter', 'DSigma')
    npz_dir = Path(plot_cfg.get('npz_path', '../data/cross_corr_C/'))
    radius_range = config.get('radius_range_arcmin', [1.0, 9.75])

    runs = load_runs(npz_dir, config['runs'], filt, radius_range)
    for run in runs:
        analyse(run, config)

    now = datetime.now()
    fig_dir = (Path(plot_cfg.get('fig_path', '../figures/'))
               / now.strftime('%Y-%m') / now.strftime('%m-%d'))
    name = plot_cfg.get('fig_name', 'scale_rescaling_multiz')
    out_path = fig_dir / f'{name}_{filt}.png'

    make_figure(runs, config, out_path, filt)
    print(f'Wrote {out_path}')
    lines = report(runs)
    txt_path = out_path.with_suffix('.txt')
    txt_path.write_text('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))
    print(f'\nWrote {txt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Y_bm/Y_mm at every listed redshift against arcmin, '
                    'comoving and physical scale, and fit a rescaling.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/scale_rescaling_multiz.yaml')
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
