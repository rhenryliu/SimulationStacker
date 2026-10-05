"""plot_scale_rescaling.py
==========================
Is the filtered ratio ``x_F(R) = Y_bm / Y_mm`` more universal across redshift
in comoving or in physical units?  Uros's question of 2026-09-27, testing the
hypothesis that the baryon-matter decorrelation sits at a fixed physical rather
than a fixed comoving scale.

For each run the z ~ 0.5 and z ~ 0.26 curves (FLAMINGO's low-z snapshot is
z = 0.30) are plotted against

1. the aperture in arcmin, as measured;
2. comoving Mpc/h, ``R_com = R_arcmin * chi(z)``;
3. physical Mpc/h, ``R_phys = R_com / (1+z)``.

All three belong to the one-parameter family ``s = R_com (1+z)^beta``:
comoving is beta = 0, physical beta = -1, and arcmin the run-specific beta for
which ``chi(z_hi)/chi(z_lo) = ((1+z_lo)/(1+z_hi))^beta``.  The script also
scans beta for the best overlap and prints it (not plotted), alongside two
physically motivated scales converted to an equivalent beta: ``R_200c`` at
fixed halo mass (comoving ``R ∝ (1+z) E(z)^{-2/3}``) and the non-linear scale
``R_nl(z)``, defined by ``sigma_lin(R_nl, z) = 1`` for a top-hat.  ``R_200m``
at fixed mass is constant in comoving units (beta = 0).

Overlap metric: the rms of ``x_hi - x_lo`` on a log-spaced grid across the
interval of s that both curves cover, each curve interpolated linearly in
ln s.  It is formed per jackknife realization (D-11), pairing the realizations
of the two snapshots patch for patch: both use the same 4x4 spatial grid on
the same yz projection of the same box.

Caveat: the Delta Sigma annulus width is fixed at 0.75 arcmin, so in comoving
units the filter is wider at z ~ 0.5 (0.29 cMpc/h) than at low z (0.16-0.18
cMpc/h).  Even an exactly universal P_bm/P_mm would therefore not make the
aperture curves coincide exactly in any single coordinate.

Usage
-----
    cd scripts/
    python cross_corr/plot_scale_rescaling.py -p configs/cross_corr/scale_rescaling.yaml
"""

import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import yaml
from scipy.integrate import trapezoid
from scipy.optimize import brentq, minimize_scalar

sys.path.append('../src/')

from stacker import SimulationStacker
import theory as th

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'text.usetex': False,      # no LaTeX in the cosmodesi environment
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
})

#: Panel titles per run label.
RUN_TITLES = {
    'TNG300-1': 'TNG300-1',
    'L1_m9_fiducial': 'FLAMINGO fiducial',
    'L1_m9_Jet_fgas-4sigma': r'FLAMINGO Jet_fgas$-4\sigma$',
    'L1_m9_fgas-8sigma': r'FLAMINGO fgas$-8\sigma$',
}

#: Display labels for filter variant names.
FILTER_LABELS = {'DSigma': r'$\Delta\Sigma$'}

#: Colour and line style of the higher- and lower-redshift snapshot.
Z_STYLE = {'hi': ('#c1272d', '-'), 'lo': ('#0072b2', '--')}

#: Coordinate rows: key, x-axis label.
ROWS = [
    ('arcmin', r'$R$ [arcmin]'),
    ('comoving', r'$R_{\rm com}$ [$h^{-1}$cMpc]'),
    ('physical', r'$R_{\rm phys} = R_{\rm com}/(1+z)$ [$h^{-1}$pMpc]'),
]


def jk_sigma(samples):
    """Leave-one-out jackknife standard deviation along axis 0.

    Args:
        samples (np.ndarray): Realizations along the first axis.

    Returns:
        np.ndarray: The jackknife error, NaN-safe.
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = np.sum(np.isfinite(samples), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sqrt((n - 1) / n
                       * np.nansum((samples - np.nanmean(samples, axis=0)) ** 2,
                                   axis=0))


def snapshot_curve(cal, filt, radius_range):
    """Extract x_F = Y_bm / Y_mm and its distance conversion for one snapshot.

    Args:
        cal (np.lib.npyio.NpzFile): One ``calibration_*.npz``.
        filt (str): Filter variant name.
        radius_range (sequence): ``(min, max)`` apertures to keep, arcmin.

    Returns:
        dict: ``z``, ``snapshot``, ``radii`` (arcmin), ``x``, ``x_jk``
        (realizations x apertures) and ``mpc_per_arcmin`` (cMpc/h).

    Raises:
        KeyError: If the filter is absent from the file.
    """
    if f'x_b_{filt}' not in cal.files:
        raise KeyError(f'Filter {filt!r} not in {cal.fid.name}.')
    radii = np.asarray(cal['radii'], dtype=np.float64)
    keep = (np.asarray(cal[f'mask_{filt}'], dtype=bool)
            & (radii >= radius_range[0]) & (radii <= radius_range[1]))
    order = np.argsort(radii)
    keep = keep[order]

    # x_b = Y_bm / Y_mm as make_calibration_factor.py wrote it, NaN where
    # Y_mm vanishes.
    x = np.asarray(cal[f'x_b_{filt}'], dtype=np.float64)
    x_jk = np.asarray(cal[f'xjk_b_{filt}'], dtype=np.float64)

    # cMpc/h per arcmin exactly as make_calibration_factor.py converted it:
    # the simulation's own cosmology at the snapshot's true redshift.
    theta_box = float(cal['meta_pixel_arcmin']) * int(cal['meta_n_pixels'])
    mpc_per_arcmin = float(cal['meta_boxsize_ckpc_h']) / 1000.0 / theta_box

    return {
        'z': float(cal['meta_redshift']),
        'snapshot': int(cal['meta_snapshot']),
        'radii': radii[order][keep],
        'x': x[order][keep],
        'x_jk': x_jk[:, order][:, keep],
        'mpc_per_arcmin': mpc_per_arcmin,
    }


def load_pairs(npz_dir, labels, filt, radius_range, snapshots=None):
    """Load both snapshots of every requested run.

    Args:
        npz_dir (pathlib.Path): Directory holding ``calibration_*.npz``.
        labels (list): Run labels, in plotting order.
        filt (str): Filter variant name.
        radius_range (sequence): ``(min, max)`` apertures to keep, arcmin.
        snapshots (dict, optional): Run label -> the two snapshot numbers to
            use, for when the directory holds more snapshots of a run.  None,
            or a label absent from it, keeps every file of that run.

    Returns:
        list: One dict per run with ``label``, ``meta`` (the higher-z file)
        and the snapshot curves ``hi`` and ``lo`` (by redshift).

    Raises:
        SystemExit: If a run does not have exactly two snapshots selected.
    """
    snapshots = snapshots or {}
    by_label = {}
    for path in sorted(glob.glob(str(npz_dir / 'calibration_*.npz'))):
        cal = np.load(path, allow_pickle=True)
        label = str(cal['meta_label'])
        wanted = snapshots.get(label)
        if wanted is not None and int(cal['meta_snapshot']) not in wanted:
            continue
        by_label.setdefault(label, []).append(cal)

    runs = []
    for label in labels:
        cals = by_label.get(label, [])
        if len(cals) != 2:
            raise SystemExit(
                f'{label}: expected 2 snapshots in {npz_dir}, found '
                f'{len(cals)}. Run make_calibration_factor.py first, or '
                "name the pair under 'snapshots' in the config.")
        cals.sort(key=lambda c: float(c['meta_redshift']), reverse=True)
        runs.append({
            'label': label,
            'meta': cals[0],
            'hi': snapshot_curve(cals[0], filt, radius_range),
            'lo': snapshot_curve(cals[1], filt, radius_range),
        })
    return runs


def coordinate(snap, beta):
    """Return ``s = R_com (1+z)^beta`` in cMpc/h for one snapshot."""
    return snap['radii'] * snap['mpc_per_arcmin'] * (1.0 + snap['z']) ** beta


def overlap_rms(s_a, x_a, s_b, x_b, n_grid, min_factor):
    """rms difference of two curves over the interval both cover.

    Args:
        s_a, x_a (np.ndarray): First curve, ``s`` increasing.
        s_b, x_b (np.ndarray): Second curve, ``s`` increasing.
        n_grid (int): Log-spaced comparison points.
        min_factor (float): Minimum ``s_max / s_min`` of the overlap.

    Returns:
        tuple: ``(rms, s_lo, s_hi)``; ``rms`` is NaN if the overlap is too
        short or either curve is non-finite there.
    """
    lo = max(s_a[0], s_b[0])
    hi = min(s_a[-1], s_b[-1])
    if not hi >= lo * min_factor:
        return np.nan, lo, hi
    ln_grid = np.linspace(np.log(lo), np.log(hi), n_grid)
    diff = (np.interp(ln_grid, np.log(s_a), x_a)
            - np.interp(ln_grid, np.log(s_b), x_b))
    return float(np.sqrt(np.mean(diff ** 2))), lo, hi


def mismatch(run, beta, cfg, jk=None):
    """Overlap rms of the two snapshots at one beta.

    Args:
        run (dict): One entry of :func:`load_pairs`.
        beta (float): Rescaling exponent.
        cfg (dict): Configuration, for the overlap settings.
        jk (int, optional): Jackknife realization; None for the full sample.

    Returns:
        float: The rms of ``x_hi - x_lo``.
    """
    hi, lo = run['hi'], run['lo']
    x_hi = hi['x'] if jk is None else hi['x_jk'][jk]
    x_lo = lo['x'] if jk is None else lo['x_jk'][jk]
    return overlap_rms(coordinate(hi, beta), x_hi, coordinate(lo, beta), x_lo,
                       int(cfg.get('n_overlap_grid', 50)),
                       float(cfg.get('min_overlap_factor', 2.0)))[0]


def best_beta(fun, betas):
    """Minimize ``fun`` on a grid, then refine within one grid step.

    Args:
        fun (callable): beta -> mismatch.
        betas (np.ndarray): Scan grid.

    Returns:
        tuple: ``(beta_hat, fun(beta_hat), on_edge)``.
    """
    values = np.array([fun(b) for b in betas])
    if not np.any(np.isfinite(values)):
        return np.nan, np.nan, False
    i = int(np.nanargmin(values))
    step = betas[1] - betas[0]
    res = minimize_scalar(fun, method='bounded',
                          bounds=(max(betas[i] - step, betas[0]),
                                  min(betas[i] + step, betas[-1])),
                          options={'xatol': 1e-5})
    beta_hat = float(res.x) if res.fun <= values[i] else float(betas[i])
    on_edge = i in (0, len(betas) - 1)
    return beta_hat, float(fun(beta_hat)), on_edge


def equivalent_beta(scale_hi, scale_lo, z_hi, z_lo):
    """The beta for which ``R_com (1+z)^beta ∝ R_com / R_*(z)``.

    Args:
        scale_hi, scale_lo (float): A comoving scale ``R_*`` at each redshift.
        z_hi, z_lo (float): The two redshifts.

    Returns:
        float: Equivalent exponent; -1 for a fixed physical scale.
    """
    return -np.log(scale_hi / scale_lo) / np.log((1.0 + z_hi) / (1.0 + z_lo))


def nonlinear_radius(k, pk):
    """Comoving top-hat radius with ``sigma(R) = 1``.

    Args:
        k (np.ndarray): Log-spaced wavenumbers, h/Mpc.
        pk (np.ndarray): Linear power spectrum, (Mpc/h)^3.

    Returns:
        float: ``R_nl`` in cMpc/h.
    """
    lnk = np.log(k)

    def ln_sigma(ln_r):
        x = k * np.exp(ln_r)
        # Series below x = 0.01, where the closed form cancels catastrophically.
        w = np.where(x > 1e-2, 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3,
                     1.0 - x ** 2 / 10.0)
        return 0.5 * np.log(trapezoid(k ** 3 * pk * w ** 2, lnk)
                            / (2.0 * np.pi ** 2))

    # Brackets R_nl (a few cMpc/h at z ~ 0.3-0.5) with wide margin.
    return float(np.exp(brentq(ln_sigma, np.log(0.05), np.log(50.0))))


_RNL_CACHE = {}


def candidate_betas(run):
    """Equivalent beta of every candidate coordinate for one run.

    ``R_200c`` uses the header cosmology through ``stacker.cosmo``; ``R_nl``
    uses CAMB linear P(k) with n_s and sigma8 from the literature table in
    :data:`theory.SIMULATION_COSMOLOGIES` (no header carries them).

    Args:
        run (dict): One entry of :func:`load_pairs`.

    Returns:
        dict: beta for 'arcmin', 'physical', 'comoving', 'R200c' and 'R_nl',
        plus ``R_nl_hi``/``R_nl_lo`` in cMpc/h.
    """
    hi, lo = run['hi'], run['lo']
    meta = run['meta']
    feedback = meta['meta_feedback'].item()
    if feedback in (None, 'None', ''):
        feedback = None
    sim_type = str(meta['meta_sim_type'])
    stacker = SimulationStacker(str(meta['meta_sim_name']), hi['snapshot'],
                                nPixels=int(meta['meta_n_pixels']),
                                simType=sim_type, feedback=feedback,
                                z=hi['z'])

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

    z_hi, z_lo = hi['z'], lo['z']
    return {
        'arcmin': equivalent_beta(hi['mpc_per_arcmin'], lo['mpc_per_arcmin'],
                                  z_hi, z_lo),
        'physical': -1.0,
        'comoving': 0.0,
        'R200c': equivalent_beta(r200c_comoving(z_hi), r200c_comoving(z_lo),
                                 z_hi, z_lo),
        'R_nl': equivalent_beta(r_nl(z_hi), r_nl(z_lo), z_hi, z_lo),
        'R_nl_hi': r_nl(z_hi),
        'R_nl_lo': r_nl(z_lo),
    }


def analyse(run, cfg):
    """Scan beta, fit beta_hat with jackknife errors, and evaluate candidates.

    Adds ``betas``, ``scan``, ``beta_hat``, ``beta_err``, ``beta_on_edge``,
    ``cand`` and ``D`` (candidate -> (rms, jackknife error)) to ``run``.

    Args:
        run (dict): One entry of :func:`load_pairs`.
        cfg (dict): Configuration.
    """
    betas = np.arange(float(cfg.get('beta_min', -4.0)),
                      float(cfg.get('beta_max', 4.0)) + 1e-9,
                      float(cfg.get('beta_step', 0.01)))
    n_jk = run['hi']['x_jk'].shape[0]
    if run['lo']['x_jk'].shape[0] != n_jk:
        raise ValueError(f"{run['label']}: the two snapshots have different "
                         'numbers of jackknife realizations.')

    run['betas'] = betas
    run['scan'] = np.array([mismatch(run, b, cfg) for b in betas])
    beta_hat, _, on_edge = best_beta(lambda b: mismatch(run, b, cfg), betas)
    beta_jk = np.array([best_beta(lambda b: mismatch(run, b, cfg, jk=i),
                                  betas)[0] for i in range(n_jk)])
    run['beta_hat'] = beta_hat
    run['beta_err'] = float(jk_sigma(beta_jk))
    run['beta_on_edge'] = on_edge

    cand = candidate_betas(run)
    cand['best'] = beta_hat
    run['cand'] = cand
    run['D'] = {}
    for key in ('arcmin', 'physical', 'comoving', 'R200c', 'R_nl', 'best'):
        d_jk = [mismatch(run, cand[key], cfg, jk=i) for i in range(n_jk)]
        run['D'][key] = (mismatch(run, cand[key], cfg), float(jk_sigma(d_jk)))


def make_figure(runs, cfg, out_path, filt):
    """x_F at both redshifts against each coordinate, one column per run.

    Each coordinate row is its own subfigure, with one x label centred under
    it and a gap before the next row, so a row's label cannot be read as the
    title of the row below.

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

    xs = np.concatenate([run[s]['x'] for run in runs for s in ('hi', 'lo')])
    pad = 0.04 * (np.nanmax(xs) - np.nanmin(xs))
    ylim = (np.nanmin(xs) - pad, np.nanmax(xs) + pad)
    n_grid = int(cfg.get('n_overlap_grid', 50))
    min_factor = float(cfg.get('min_overlap_factor', 2.0))

    for row, (subfig, (kind, xlabel)) in enumerate(zip(subfigs, ROWS)):
        axes = subfig.subplots(1, n_runs, sharex=True, sharey=True,
                               squeeze=False)[0]
        for col, (ax, run) in enumerate(zip(axes, runs)):
            beta = run['cand'][kind]
            coords = {}
            for which in ('hi', 'lo'):
                snap = run[which]
                s = (snap['radii'] if kind == 'arcmin'
                     else coordinate(snap, beta))
                coords[which] = s
                colour, ls = Z_STYLE[which]
                err = jk_sigma(snap['x_jk'])
                ax.fill_between(s, snap['x'] - err, snap['x'] + err,
                                color=colour, alpha=0.25, lw=0)
                ax.plot(s, snap['x'], color=colour, ls=ls, lw=1.8,
                        marker='o', ms=3.5,
                        label=f"$z = {snap['z']:.2f}$")
            # The overlap metric is invariant to a common rescaling of s, so
            # the arcmin row's value is D(beta_arcmin), computed directly here.
            rms, s_lo, s_hi = overlap_rms(coords['hi'], run['hi']['x'],
                                          coords['lo'], run['lo']['x'],
                                          n_grid, min_factor)
            ax.axvspan(s_lo, s_hi, color='0.92', zorder=0, lw=0)
            ax.text(0.97, 0.05, rf'rms $\Delta x$ = {rms:.4f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=9)
            ax.set_xscale('log')
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.25, lw=0.5)
            if row == 0:
                ax.set_title(RUN_TITLES.get(run['label'], run['label']))
                ax.legend(loc='upper left', frameon=False)
            elif kind == 'comoving' and col == 0:
                # Here the shading does not fill the panel, so the key shows.
                ax.legend(handles=[Patch(facecolor='0.92', edgecolor='0.6',
                                         label='range of rms')],
                          loc='upper left', frameon=False)
        axes[0].set_ylabel(r'$Y_{bm}/Y_{mm}$')
        subfig.supxlabel(xlabel)

    fig.suptitle(rf'$Y_{{bm}}/Y_{{mm}}$ ({FILTER_LABELS.get(filt, filt)})',
                 fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def report(runs):
    """Print the per-run candidates, their overlap rms and the fitted beta.

    Args:
        runs (list): Analysed runs.
    """
    keys = ('arcmin', 'physical', 'R200c', 'comoving', 'R_nl', 'best')
    print('\nEquivalent beta of each coordinate, s = R_com (1+z)^beta '
          '(0 comoving, -1 physical)')
    print(f"{'run':24s} {'z_hi':>6s} {'z_lo':>6s} "
          + ' '.join(f'{k:>9s}' for k in keys))
    for run in runs:
        print(f"{run['label']:24s} {run['hi']['z']:6.3f} {run['lo']['z']:6.3f} "
              + ' '.join(f"{run['cand'][k]:9.3f}" for k in keys))

    print('\nOverlap rms of x_hi - x_lo (jackknife error), per coordinate')
    print(f"{'run':24s} " + ' '.join(f'{k:>17s}' for k in keys))
    for run in runs:
        print(f"{run['label']:24s} "
              + ' '.join(f"{run['D'][k][0]:8.4f}({run['D'][k][1]:.4f})"
                         for k in keys))

    print('\nBest-fit beta (jackknife error); R_nl in cMpc/h (sigma_lin = 1, '
          'literature n_s/sigma8)')
    for run in runs:
        edge = '  ** at the edge of the scan **' if run['beta_on_edge'] else ''
        c = run['cand']
        print(f"  {run['label']:24s} beta_hat = {run['beta_hat']:+.3f} "
              f"+/- {run['beta_err']:.3f}   R_nl = {c['R_nl_hi']:.3f} "
              f"(z_hi) / {c['R_nl_lo']:.3f} (z_lo){edge}")

    # One beta for every run: minimize the summed squared rms.
    betas = runs[0]['betas']
    total = np.sum([run['scan'] ** 2 for run in runs], axis=0)
    if np.any(np.isfinite(total)):
        i = int(np.nanargmin(total))
        print(f'\n  common beta over all runs (grid, step '
              f"{betas[1] - betas[0]:.2f}): {betas[i]:+.2f}")


def main(path2config):
    """Draw the scale-rescaling figure and print the overlap metrics.

    Args:
        path2config (str): Path to the YAML configuration file.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)
    plot_cfg = config.get('plot', {})
    filt = config.get('filter', 'DSigma')
    npz_dir = Path(plot_cfg.get('npz_path', '../data/cross_corr_C/'))
    radius_range = config.get('radius_range_arcmin', [1.0, 9.75])

    runs = load_pairs(npz_dir, config['runs'], filt, radius_range,
                      config.get('snapshots'))
    for run in runs:
        analyse(run, config)

    now = datetime.now()
    fig_dir = (Path(plot_cfg.get('fig_path', '../figures/'))
               / now.strftime('%Y-%m') / now.strftime('%m-%d'))
    name = plot_cfg.get('fig_name', 'scale_rescaling')
    out_path = fig_dir / f'{name}_{filt}.png'

    make_figure(runs, config, out_path, filt)
    print(f'Wrote {out_path}')
    report(runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Y_bm/Y_mm at two redshifts against arcmin, '
                    'comoving and physical scale, and fit a rescaling.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/scale_rescaling.yaml')
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
