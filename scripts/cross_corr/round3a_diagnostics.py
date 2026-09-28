"""round3a_diagnostics.py
=======================
Round 3A, Stage 1: the diagnostics that need no new computation, read from
the committed round-two outputs ``data/cross_corr_C/calibration_*.npz`` and
``task9_spectra_*.npz`` (``docs/cross_corr/open-items.md``):

- **O-10**: ``C_F`` against ``k_50(R;F)`` for all eight filters on one axis;
  the spread between filters at matched ``k_50`` (the collapse test of
  formalism §8.4); mean ``|C-1|`` on matched aperture ranges; and the
  addendum's prediction 3 (Sigma's advantage at matched ``k_50``).
- **O-03**: ``C`` against ``x`` per aperture; a straight line fitted to the
  three FLAMINGO variants predicts TNG300-1.
- **O-04**: the matter-crossed electron-to-baryon correction ``Y_bm/Y_em``
  beside the galaxy-crossed ``Y_gb/Y_ge``, with per-realization jackknife
  errors.
- **O-09**: the log-slope of ``Y_gm^(DSigma)`` over 1'-9.75', and how much of
  the DSigma amplitude Upsilon keeps.
- **D-05 input**: the electron and baryon calibration factors split into the
  cross-code (TNG300-1 against FLAMINGO fiducial) and cross-feedback
  (FLAMINGO fiducial against its variants) axes, with jackknife errors --
  paired, realization by realization, for the FLAMINGO variants.
- **P8**: whether the FLAMINGO variants share initial conditions, from the
  correlation of their leave-one-out realizations.

Scored against ``docs/cross_corr/predictions/2026-09-27_round3a.md``.
Writes figures under ``figures/<yyyy-mm>/<mm-dd>/`` and the numbers to
``data/cross_corr_C/round3a/stage1_diagnostics.{npz,txt}``.

Usage
-----
    cd scripts/
    python cross_corr/round3a_diagnostics.py
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

import rprofiles as rp
import round3a_lib as lib

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    # Computer Modern draws the forked Upsilon; dejavuserif renders it as Y.
    'mathtext.fontset': 'cm', 'text.usetex': False,
    'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
    'legend.fontsize': 8,
})

#: Runs, in display order: label in the npz file names, short name.
RUNS = (('TNG300-1', 'TNG300-1'),
        ('L1_m9_fiducial', 'FLA fid'),
        ('L1_m9_Jet_fgas-4sigma', 'FLA Jet'),
        ('L1_m9_fgas-8sigma', 'FLA fgas-8σ'))
FLA = ('L1_m9_fiducial', 'L1_m9_Jet_fgas-4sigma', 'L1_m9_fgas-8sigma')

#: Snapshots per redshift sample.
SAMPLES = {'z05': {'TNG': 67, 'FLA': 67, 'z': '0.5'},
           'z026': {'TNG': 80, 'FLA': 71, 'z': '0.26/0.30'}}

FILTERS = ('DSigma', 'Sigma', 'Upsilon_R0=1', 'Upsilon_R0=2',
           'Ytransform_Rmax=4', 'Ytransform_Rmax=5', 'Ytransform_Rmax=6',
           'Ytransform_Rmax=9')
LABELS = {'DSigma': r'$\Delta\Sigma$', 'Sigma': r'$\Sigma$',
          'Upsilon_R0=1': r"$\Upsilon(1')$", 'Upsilon_R0=2': r"$\Upsilon(2')$",
          'Ytransform_Rmax=4': r"$Y(4')$", 'Ytransform_Rmax=5': r"$Y(5')$",
          'Ytransform_Rmax=6': r"$Y(6')$", 'Ytransform_Rmax=9': r"$Y(9')$"}
COLOURS = dict(zip(FILTERS, ('k', 'tab:blue', 'tab:red', 'tab:orange',
                             'tab:green', 'tab:olive', 'tab:cyan',
                             'tab:purple')))
DATA_RANGE = (1.0, 6.0)


def load_sample(npz_dir, zkey):
    """Load the calibration and Task 9 files of one redshift sample.

    Args:
        npz_dir (pathlib.Path): ``data/cross_corr_C``.
        zkey (str): ``'z05'`` or ``'z026'``.

    Returns:
        dict: label to ``{'cal', 'spec'}`` npz handles.
    """
    runs = {}
    for label, _ in RUNS:
        snap = SAMPLES[zkey]['TNG' if label.startswith('TNG') else 'FLA']
        runs[label] = {
            'cal': np.load(npz_dir / f'calibration_{label}_{snap}_yz.npz',
                           allow_pickle=True),
            'spec': np.load(npz_dir / f'task9_spectra_{label}_{snap}_yz.npz',
                            allow_pickle=True)}
    return runs


def data_mask(radii):
    """Apertures inside the data range 1'-6'.

    Args:
        radii (np.ndarray): Aperture grid, arcmin.

    Returns:
        np.ndarray: Boolean mask.
    """
    return (radii >= DATA_RANGE[0] - 1e-9) & (radii <= DATA_RANGE[1] + 1e-9)


# ---------------------------------------------------------------------------
# O-10: collapse at matched k_50, matched aperture ranges, prediction 3
# ---------------------------------------------------------------------------

def collapse(runs, gas='b'):
    """Spread between each filter's ``C_F`` and DSigma's at matched ``k_50``.

    Each filter's curve is interpolated in ``ln k_50`` onto DSigma's
    ``k_50`` points over every aperture where both are defined.

    Args:
        runs (dict): Output of :func:`load_sample`.
        gas (str, optional): Gas field.

    Returns:
        dict: ``{label: {filter: (max |dC|, n points)}}``.
    """
    out = {}
    for label, run in runs.items():
        cal, spec = run['cal'], run['spec']
        k_ds = spec['kquant_hmpc_DSigma'][:, 1]
        c_ds = cal[f'C_{gas}_DSigma']
        out[label] = {}
        for filt in FILTERS[1:]:
            on_ds = lib.interp_log_k(spec[f'kquant_hmpc_{filt}'][:, 1],
                                     cal[f'C_{gas}_{filt}'], k_ds)
            diff = np.abs(on_ds - c_ds)
            n = int(np.isfinite(diff).sum())
            out[label][filt] = (float(np.nanmax(diff)) if n else np.nan, n)
    return out


def matched_ranges(runs, gas='b'):
    """Mean ``|C-1|`` of DSigma and each filter on their common usable bins.

    Args:
        runs (dict): Output of :func:`load_sample`.
        gas (str, optional): Gas field.

    Returns:
        dict: ``{filter: (DSigma value, filter value, n bins)}``, averaged
        over runs and bins, within 1'-6'.
    """
    out = {}
    for filt in FILTERS[1:]:
        ds, ff, n = [], [], 0
        for run in runs.values():
            cal = run['cal']
            radii = cal['radii']
            common = (data_mask(radii) & cal['mask_DSigma']
                      & cal[f'mask_{filt}'])
            ds.append(np.abs(cal[f'C_{gas}_DSigma'][common] - 1.0))
            ff.append(np.abs(cal[f'C_{gas}_{filt}'][common] - 1.0))
            n = int(common.sum())
        out[filt] = (float(np.mean(np.concatenate(ds))),
                     float(np.mean(np.concatenate(ff))), n)
    return out


def prediction3(runs, gas='b'):
    """Sigma against DSigma at matched aperture and at matched ``k_50``.

    Matched aperture: mean ``|C-1|`` over 1'-6'.  Matched ``k_50``: Sigma's
    points whose ``k_50`` lies inside DSigma's range (all apertures), against
    DSigma interpolated to the same ``k_50``.

    Args:
        runs (dict): Output of :func:`load_sample`.
        gas (str, optional): Gas field.

    Returns:
        dict: ``{'R': (Sigma, DSigma), 'k50': (Sigma, DSigma, n)}``,
        averaged over runs.
    """
    rs, rd, ks, kd, n = [], [], [], [], 0
    for run in runs.values():
        cal, spec = run['cal'], run['spec']
        dm = data_mask(cal['radii'])
        rs.append(np.abs(cal[f'C_{gas}_Sigma'][dm] - 1.0))
        rd.append(np.abs(cal[f'C_{gas}_DSigma'][dm] - 1.0))
        k_sig = spec['kquant_hmpc_Sigma'][:, 1]
        ds_at = lib.interp_log_k(spec['kquant_hmpc_DSigma'][:, 1],
                                 cal[f'C_{gas}_DSigma'], k_sig)
        ok = np.isfinite(ds_at) & np.isfinite(cal[f'C_{gas}_Sigma'])
        ks.append(np.abs(cal[f'C_{gas}_Sigma'][ok] - 1.0))
        kd.append(np.abs(ds_at[ok] - 1.0))
        n += int(ok.sum())
    mean = lambda a: float(np.mean(np.concatenate(a))) if a else np.nan
    return {'R': (mean(rs), mean(rd)), 'k50': (mean(ks), mean(kd), n)}


# ---------------------------------------------------------------------------
# O-03: C against x
# ---------------------------------------------------------------------------

def c_versus_x(runs, filt='DSigma', gas='b'):
    """Fit ``C = a + b x`` to the FLAMINGO variants and predict TNG300-1.

    Args:
        runs (dict): Output of :func:`load_sample`.
        filt (str, optional): Filter.
        gas (str, optional): Gas field.

    Returns:
        dict: per-aperture arrays ``radii``, ``err_tng`` (TNG300-1 minus
        prediction), ``spread_fb`` (max-min over the variants), ``slope``,
        ``pass`` (``|err| < spread/2``), and ``loo_rms`` (leave-one-out over
        all four runs).
    """
    cal0 = runs['TNG300-1']['cal']
    radii = cal0['radii']
    keep = data_mask(radii) & cal0[f'mask_{filt}']
    rows = {k: [] for k in ('radii', 'err_tng', 'spread_fb', 'slope', 'pass',
                            'loo_rms')}
    for i in np.flatnonzero(keep):
        x = {lab: float(runs[lab]['cal'][f'x_{gas}_{filt}'][i])
             for lab, _ in RUNS}
        c = {lab: float(runs[lab]['cal'][f'C_{gas}_{filt}'][i])
             for lab, _ in RUNS}
        xf = np.array([x[l] for l in FLA])
        cf = np.array([c[l] for l in FLA])
        pred = lib.predict_from_fit(xf, cf, x['TNG300-1'])
        spread = float(cf.max() - cf.min())
        err = c['TNG300-1'] - pred
        allx = np.array([x[l] for l, _ in RUNS])
        allc = np.array([c[l] for l, _ in RUNS])
        loo = lib.loo_linear_prediction(allx, allc) - allc
        rows['radii'].append(radii[i])
        rows['err_tng'].append(err)
        rows['spread_fb'].append(spread)
        rows['slope'].append(float(np.polyfit(xf, cf, 1)[0]))
        rows['pass'].append(abs(err) < 0.5 * spread)
        rows['loo_rms'].append(float(np.sqrt(np.mean(loo ** 2))))
    return {k: np.array(v) for k, v in rows.items()}


# ---------------------------------------------------------------------------
# O-04: the electron-to-baryon correction
# ---------------------------------------------------------------------------

def e_to_b(runs, filt='DSigma'):
    """Matter- and galaxy-crossed ``b/e`` amplitude ratios with errors.

    Args:
        runs (dict): Output of :func:`load_sample`.
        filt (str, optional): Filter.

    Returns:
        dict: ``{label: {'radii', 'matter', 'matter_err', 'galaxy',
        'galaxy_err', 'total'}}`` where ``total`` is the Convention T
        ``Y_bt/Y_et``.
    """
    out = {}
    for label, run in runs.items():
        cal = run['cal']
        ratio = {}
        for tag, (num, den) in (('matter', ('bm', 'em')),
                                ('galaxy', ('bg', 'eg'))):
            ratio[tag] = cal[f'Y_{num}_{filt}'] / cal[f'Y_{den}_{filt}']
            jk = cal[f'Yjk_{num}_{filt}'] / cal[f'Yjk_{den}_{filt}']
            ratio[f'{tag}_err'] = rp.jackknife_error(jk, axis=0)
        ratio['total'] = cal[f'Y_bt_{filt}'] / cal[f'Y_et_{filt}']
        ratio['radii'] = cal['radii']
        out[label] = ratio
    return out


# ---------------------------------------------------------------------------
# O-09: the DSigma log-slope and the Upsilon near-cancellation
# ---------------------------------------------------------------------------

def slopes(runs):
    """Log-slopes of the DSigma amplitudes, and Upsilon's surviving share.

    Args:
        runs (dict): Output of :func:`load_sample`.

    Returns:
        dict: ``{label: {'slope_<pair>', 'local_gm', 'radii',
        'ups_share_<r0>', 'err_inflation_<r0>'}}``.  ``ups_share`` is
        ``Y_gm^(Upsilon)/Y_gm^(DSigma)``; ``err_inflation`` is the
        jackknife error of ``C^(Upsilon)`` over that of ``C^(DSigma)``.
    """
    out = {}
    for label, run in runs.items():
        cal = run['cal']
        radii = cal['radii']
        res = {'radii': radii}
        for pair in ('gm', 'bg', 'mm', 'bm'):
            y = cal[f'Y_{pair}_DSigma']
            s, local = lib.loglog_slope(radii, y)
            res[f'slope_{pair}'] = s
            if pair == 'gm':
                res['local_gm'] = local
        for r0 in ('1', '2'):
            filt = f'Upsilon_R0={r0}'
            with np.errstate(invalid='ignore', divide='ignore'):
                share = cal[f'Y_gm_{filt}'] / cal['Y_gm_DSigma']
                infl = cal[f'Cerr_b_{filt}'] / cal['Cerr_b_DSigma']
            share = np.where(cal[f'mask_{filt}'], share, np.nan)
            res[f'ups_share_{r0}'] = share
            res[f'err_inflation_{r0}'] = infl
        out[label] = res
    return out


# ---------------------------------------------------------------------------
# D-05 input and P8: the scatter split by axis
# ---------------------------------------------------------------------------

def axis_split(runs, filt='DSigma', gas='b'):
    """Cross-code and cross-feedback differences of ``C`` with errors.

    Cross-code: TNG300-1 against FLAMINGO fiducial, errors added in
    quadrature (independent volumes).  Cross-feedback: FLAMINGO fiducial
    against each variant, with the jackknife error of the per-realization
    difference (the variants share a volume and a block layout).

    Args:
        runs (dict): Output of :func:`load_sample`.
        filt (str, optional): Filter.
        gas (str, optional): Gas field.

    Returns:
        dict: arrays over the data-range usable apertures: ``radii``,
        ``code``, ``code_err``, ``fb_<variant>``, ``fb_<variant>_err``
        (fractional, relative to the pair mean), and the realization
        correlations ``rho_code``, ``rho_<variant>``.
    """
    get = lambda lab, key: runs[lab]['cal'][f'{key}_{gas}_{filt}']
    radii = runs['TNG300-1']['cal']['radii']
    keep = data_mask(radii) & runs['TNG300-1']['cal'][f'mask_{filt}']
    out = {'radii': radii[keep]}
    ct, cf = get('TNG300-1', 'C')[keep], get('L1_m9_fiducial', 'C')[keep]
    out['code'] = (ct - cf) / (0.5 * (ct + cf))
    out['code_err'] = np.hypot(get('TNG300-1', 'Cerr')[keep],
                               get('L1_m9_fiducial', 'Cerr')[keep]) / cf
    jk_t = get('TNG300-1', 'Cjk')[:, keep]
    jk_f = get('L1_m9_fiducial', 'Cjk')[:, keep]
    out['rho_code'] = lib.paired_correlation(jk_t, jk_f)
    for lab, tag in (('L1_m9_fgas-8sigma', 'f8'),
                     ('L1_m9_Jet_fgas-4sigma', 'jet')):
        cv = get(lab, 'C')[keep]
        jk_v = get(lab, 'Cjk')[:, keep]
        out[f'fb_{tag}'] = (cv - cf) / (0.5 * (cv + cf))
        out[f'fb_{tag}_err'] = rp.jackknife_error(jk_v - jk_f, axis=0) / cf
        out[f'rho_{tag}'] = lib.paired_correlation(jk_f, jk_v)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_collapse(samples, fig_dir):
    """``C_F`` against ``k_50`` for every filter, one panel per run.

    Args:
        samples (dict): ``{zkey: runs}``.
        fig_dir (pathlib.Path): Output directory.

    Returns:
        pathlib.Path: The figure.
    """
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharey='row')
    for row, (zkey, runs) in enumerate(samples.items()):
        for col, (label, short) in enumerate(RUNS):
            ax = axes[row, col]
            cal, spec = runs[label]['cal'], runs[label]['spec']
            for filt in FILTERS:
                k = spec[f'kquant_hmpc_{filt}'][:, 1]
                c = cal[f'C_b_{filt}']
                ok = np.isfinite(k) & np.isfinite(c)
                ax.plot(k[ok], c[ok], 'o-', ms=3, lw=1.2 if filt != 'DSigma'
                        else 2.2, color=COLOURS[filt], label=LABELS[filt])
            ax.axhline(1.0, color='0.6', lw=0.8)
            ax.set_xscale('log')
            ax.set_title(f'{short}, z≈{SAMPLES[zkey]["z"]}')
            if row == 1:
                ax.set_xlabel(r'$k_{50}(R;\mathcal{F})$ [$h$/Mpc]')
            if col == 0:
                ax.set_ylabel(r'$C_{\mathcal{F}}$ (baryons)')
    axes[0, 0].legend(ncol=2)
    fig.suptitle('O-10: the calibration factor against the wavenumber each '
                 'filter probes')
    fig.tight_layout()
    path = fig_dir / 'round3a_s1_collapse_k50.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_c_vs_x(samples, fig_dir, radii_show=(1.0, 2.25, 3.5, 6.0)):
    """``C`` against ``x`` at four apertures, with the FLAMINGO line.

    Args:
        samples (dict): ``{zkey: runs}``.
        fig_dir (pathlib.Path): Output directory.
        radii_show (tuple, optional): Apertures to show, arcmin.

    Returns:
        pathlib.Path: The figure.
    """
    fig, axes = plt.subplots(2, len(radii_show), figsize=(16, 7.5))
    marks = dict(zip([l for l, _ in RUNS], ('s', 'o', '^', 'D')))
    for row, (zkey, runs) in enumerate(samples.items()):
        radii = runs['TNG300-1']['cal']['radii']
        for col, R in enumerate(radii_show):
            ax = axes[row, col]
            i = int(np.argmin(np.abs(radii - R)))
            xs, cs = [], []
            for label, short in RUNS:
                x = runs[label]['cal']['x_b_DSigma'][i]
                c = runs[label]['cal']['C_b_DSigma'][i]
                e = runs[label]['cal']['Cerr_b_DSigma'][i]
                ax.errorbar(x, c, yerr=e, fmt=marks[label], label=short)
                if label in FLA:
                    xs.append(x)
                    cs.append(c)
            b, a = np.polyfit(xs, cs, 1)
            xx = np.linspace(min(xs + [runs['TNG300-1']['cal']['x_b_DSigma'][i]]),
                             max(xs + [runs['TNG300-1']['cal']['x_b_DSigma'][i]]),
                             10)
            ax.plot(xx, a + b * xx, 'k--', lw=1)
            ax.set_title(f"R = {radii[i]:.3g}', z≈{SAMPLES[zkey]['z']}")
            ax.set_xlabel(r'$x_{\Delta\Sigma} = Y_{bm}/Y_{mm}$')
            if col == 0:
                ax.set_ylabel(r'$C^{(\Delta\Sigma)}$')
    axes[0, 0].legend()
    fig.suptitle('O-03: C against x; dashed: straight line through the '
                 'three FLAMINGO variants')
    fig.tight_layout()
    path = fig_dir / 'round3a_s1_c_vs_x.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_e_to_b(eb, fig_dir):
    """Matter- and galaxy-crossed ``b/e`` ratios against aperture.

    Args:
        eb (dict): ``{zkey: e_to_b(...)}``.
        fig_dir (pathlib.Path): Output directory.

    Returns:
        pathlib.Path: The figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colours = dict(zip([l for l, _ in RUNS],
                       ('0.3', 'tab:blue', 'tab:orange', 'tab:red')))
    for ax, (zkey, res) in zip(axes, eb.items()):
        for label, short in RUNS:
            r = res[label]
            ax.errorbar(r['radii'], r['matter'], yerr=r['matter_err'],
                        color=colours[label], lw=1.8, label=f'{short}: '
                        r'$Y_{bm}/Y_{em}$')
            ax.plot(r['radii'], r['galaxy'], '--', color=colours[label],
                    lw=1.2, label=r'$Y_{gb}/Y_{ge}$')
        ax.axvspan(*DATA_RANGE, color='0.93', zorder=-1)
        ax.set_xlabel('R [arcmin]')
        ax.set_title(f'z≈{SAMPLES[zkey]["z"]}, ' r'$\Delta\Sigma$')
    axes[0].set_ylabel('baryon / electron amplitude')
    axes[0].legend(ncol=2, fontsize=7)
    fig.suptitle('O-04: the electron-to-baryon correction, matter-crossed '
                 '(solid) and galaxy-crossed (dashed)')
    fig.tight_layout()
    path = fig_dir / 'round3a_s1_e_to_b.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_slopes(sl, fig_dir):
    """``Y_gm^(DSigma)`` log-log, and Upsilon's share of it.

    Args:
        sl (dict): ``{zkey: slopes(...)}``.
        fig_dir (pathlib.Path): Output directory.

    Returns:
        pathlib.Path: The figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colours = dict(zip([l for l, _ in RUNS],
                       ('0.3', 'tab:blue', 'tab:orange', 'tab:red')))
    for zkey, res in sl.items():
        ls = '-' if zkey == 'z05' else ':'
        for label, short in RUNS:
            r = res[label]
            axes[0].plot(r['radii'][1:], r['local_gm'], ls,
                         color=colours[label],
                         label=f'{short}, z≈{SAMPLES[zkey]["z"]}')
            axes[1].plot(r['radii'], r['ups_share_1'], ls,
                         color=colours[label])
    axes[0].axhline(-2.0, color='k', lw=0.8)
    axes[0].set_xlabel("R [arcmin] (upper edge of the pair)")
    axes[0].set_ylabel(r'local $d\ln Y^{(\Delta\Sigma)}_{gm}/d\ln R$')
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.0, color='k', lw=0.8)
    axes[1].set_xlabel('R [arcmin]')
    axes[1].set_ylabel(r"$Y^{(\Upsilon(1'))}_{gm}/Y^{(\Delta\Sigma)}_{gm}$")
    fig.suptitle('O-09: the DSigma log-slope, and the share of the DSigma '
                 'amplitude Upsilon keeps')
    fig.tight_layout()
    path = fig_dir / 'round3a_s1_slopes.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_axis_split(splits, fig_dir):
    """Cross-code and cross-feedback differences of ``C`` by gas field.

    Args:
        splits (dict): ``{(zkey, gas): axis_split(...)}``.
        fig_dir (pathlib.Path): Output directory.

    Returns:
        pathlib.Path: The figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for (zkey, gas), s in splits.items():
        ax = axes[0 if zkey == 'z05' else 1, 0 if gas == 'b' else 1]
        ax.errorbar(s['radii'], s['code'], yerr=s['code_err'], fmt='s-',
                    color='0.3', label='TNG300-1 − FLA fid')
        ax.errorbar(s['radii'], s['fb_f8'], yerr=s['fb_f8_err'], fmt='o-',
                    color='tab:red', label='FLA fgas-8σ − fid')
        ax.errorbar(s['radii'], s['fb_jet'], yerr=s['fb_jet_err'], fmt='^-',
                    color='tab:orange', label='FLA Jet − fid')
        ax.axhline(0.0, color='k', lw=0.8)
        ax.set_title(f'z≈{SAMPLES[zkey]["z"]}, gas = '
                     f'{"baryons" if gas == "b" else "electrons"}')
    for ax in axes[1]:
        ax.set_xlabel('R [arcmin]')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\Delta C^{(\Delta\Sigma)}/C$')
    axes[0, 0].legend()
    fig.suptitle('D-05 input: the calibration-factor scatter split into '
                 'code and feedback axes')
    fig.tight_layout()
    path = fig_dir / 'round3a_s1_axis_split.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(samples, results):
    """Print the Stage 1 numbers, scored against the predictions file.

    Args:
        samples (dict): ``{zkey: runs}``.
        results (dict): The computed diagnostics.
    """
    print('=' * 78)
    print('Round 3A, Stage 1 diagnostics (committed round-two outputs)')
    print('=' * 78)
    for zkey in samples:
        z = SAMPLES[zkey]['z']
        print(f'\n---------------- z ≈ {z} ----------------')

        print('\n[O-10] Collapse: max |C_F - C_DSigma| at matched k_50 '
              '(all apertures; P4 threshold 0.03)')
        col = results[f'collapse_{zkey}']
        print('  ' + ' ' * 20 + ''.join(f'{s:>14s}' for _, s in RUNS))
        for filt in FILTERS[1:]:
            print(f'  {filt:20s}' + ''.join(
                f'{col[l][filt][0]:10.3f}({col[l][filt][1]:2d})'
                for l, _ in RUNS))

        print('\n[O-10] Mean |C-1| on matched aperture ranges, 1\'-6\' '
              '(DSigma | filter | bins)')
        for filt, (d, f, n) in results[f'matched_{zkey}'].items():
            print(f'  {filt:20s} {d:7.3f} | {f:7.3f} | {n:2d}   '
                  f'ratio filter/DSigma = {f / d:5.2f}')

        p3 = results[f'pred3_{zkey}']
        print('\n[Prediction 3] Sigma vs DSigma, mean |C-1|:')
        print(f'  matched aperture (1\'-6\'): Sigma {p3["R"][0]:.3f}, '
              f'DSigma {p3["R"][1]:.3f}')
        print(f'  matched k_50 ({p3["k50"][2]} points): Sigma '
              f'{p3["k50"][0]:.3f}, DSigma {p3["k50"][1]:.3f}')

        cx = results[f'cx_{zkey}']
        print('\n[O-03] C = a + b x from the FLAMINGO variants, predicting '
              'TNG300-1 (DSigma, baryons)')
        print(f"  {'R':>6s} {'TNG-pred':>9s} {'fb spread':>10s} "
              f"{'slope b':>8s} {'LOO rms':>8s} pass(|err|<spread/2)")
        for i, R in enumerate(cx['radii']):
            print(f"  {R:6.3f} {cx['err_tng'][i]:+9.4f} "
                  f"{cx['spread_fb'][i]:10.4f} {cx['slope'][i]:8.3f} "
                  f"{cx['loo_rms'][i]:8.4f} {bool(cx['pass'][i])}")

        eb = results[f'eb_{zkey}']
        print('\n[O-04] b/e ratios, DSigma: matter-crossed Y_bm/Y_em '
              '(jk err) | galaxy-crossed Y_gb/Y_ge | Conv. T Y_bt/Y_et')
        radii = eb['TNG300-1']['radii']
        for R in (1.0, 2.25, 3.5, 6.0, 9.75):
            i = int(np.argmin(np.abs(radii - R)))
            print(f"  R = {radii[i]:5.3f}': " + '; '.join(
                f"{s} {eb[l]['matter'][i]:.3f}±{eb[l]['matter_err'][i]:.3f}"
                f" | {eb[l]['galaxy'][i]:.3f} | {eb[l]['total'][i]:.3f}"
                for l, s in RUNS))

        sl = results[f'slopes_{zkey}']
        print('\n[O-09] Global log-slope over 1\'-9.75\' of the DSigma '
              'amplitudes (P6 range for Y_gm: [-2.2, -1.8])')
        for label, short in RUNS:
            r = sl[label]
            print(f"  {short:12s} Y_gm {r['slope_gm']:+.3f}  Y_gb "
                  f"{r['slope_bg']:+.3f}  Y_mm {r['slope_mm']:+.3f}  Y_bm "
                  f"{r['slope_bm']:+.3f}")
        print("  Upsilon(1') share of Y_gm^(DSigma) and C-error inflation "
              "at R = 2.25', 6', 9.75':")
        for label, short in RUNS:
            r = sl[label]
            idx = [int(np.argmin(np.abs(r['radii'] - R)))
                   for R in (2.25, 6.0, 9.75)]
            print(f"  {short:12s} share " + ' '.join(
                f"{r['ups_share_1'][i]:+.3f}" for i in idx)
                + '   inflation ' + ' '.join(
                f"{r['err_inflation_1'][i]:6.1f}" for i in idx))

        print('\n[D-05] Axis split of C (DSigma); max |dC/C| over 1\'-6\' and '
              'its significance (jackknife)')
        for gas in ('b', 'e'):
            s = results[f'split_{zkey}_{gas}']
            for key, tag in (('code', 'code (TNG-fid)'),
                             ('fb_f8', 'feedback (f8-fid)'),
                             ('fb_jet', 'feedback (Jet-fid)')):
                i = int(np.nanargmax(np.abs(s[key])))
                print(f"  gas={gas} {tag:20s} max {abs(s[key][i]):.3f} at "
                      f"R={s['radii'][i]:.3f}' ({abs(s[key][i]) / s[key + '_err'][i]:6.1f} sigma)")
        s = results[f'split_{zkey}_b']
        print('\n[P8] Correlation of leave-one-out C_b realizations, median '
              "over 1'-6' (P8: > 0.9 for the variants, |rho| < 0.5 across "
              'codes)')
        print(f"  fid vs fgas-8σ {np.nanmedian(s['rho_f8']):+.3f}; fid vs Jet "
              f"{np.nanmedian(s['rho_jet']):+.3f}; TNG vs fid "
              f"{np.nanmedian(s['rho_code']):+.3f}")


def main(npz_dir, out_dir, fig_root, verbose=True):
    """Compute, plot and report the Stage 1 diagnostics.

    Args:
        npz_dir (str): Directory of the committed round-two npz.
        out_dir (str): Directory for the Stage 1 numbers.
        fig_root (str): Root of the dated figure directories.
        verbose (bool, optional): Print the report.
    """
    npz_dir, out_dir = Path(npz_dir), Path(out_dir)
    now = datetime.now()
    fig_dir = Path(fig_root) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    fig_dir.mkdir(parents=True, exist_ok=True)
    samples = {z: load_sample(npz_dir, z) for z in SAMPLES}

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for zkey, runs in samples.items():
            results[f'collapse_{zkey}'] = collapse(runs)
            results[f'matched_{zkey}'] = matched_ranges(runs)
            results[f'pred3_{zkey}'] = prediction3(runs)
            results[f'cx_{zkey}'] = c_versus_x(runs)
            results[f'eb_{zkey}'] = e_to_b(runs)
            results[f'slopes_{zkey}'] = slopes(runs)
            for gas in ('b', 'e'):
                results[f'split_{zkey}_{gas}'] = axis_split(runs, gas=gas)

        paths = [fig_collapse(samples, fig_dir),
                 fig_c_vs_x(samples, fig_dir),
                 fig_e_to_b({z: results[f'eb_{z}'] for z in samples},
                            fig_dir),
                 fig_slopes({z: results[f'slopes_{z}'] for z in samples},
                            fig_dir),
                 fig_axis_split({(z, g): results[f'split_{z}_{g}']
                                 for z in samples for g in ('b', 'e')},
                                fig_dir)]

        buf = io.StringIO()
        with redirect_stdout(buf):
            report(samples, results)
    text = buf.getvalue()
    if verbose:
        print(text)
        for p in paths:
            print(f'Wrote {p}')

    out_dir.mkdir(parents=True, exist_ok=True)
    lib.write_text_atomic(out_dir / 'stage1_diagnostics.txt', text)
    flat = {}
    for key, value in results.items():
        if key.startswith(('cx_', 'split_')):
            for sub, arr in value.items():
                flat[f'{key}__{sub}'] = np.asarray(arr)
        elif key.startswith('slopes_'):
            for label, res in value.items():
                for sub, arr in res.items():
                    flat[f'{key}__{label}__{sub}'] = np.asarray(arr)
        elif key.startswith('eb_'):
            for label, res in value.items():
                for sub, arr in res.items():
                    flat[f'{key}__{label}__{sub}'] = np.asarray(arr)
    lib.save_npz_atomic(out_dir / 'stage1_diagnostics.npz', **flat)
    print(f"Wrote {out_dir / 'stage1_diagnostics.txt'} and .npz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Round 3A Stage 1 diagnostics from committed outputs.')
    parser.add_argument('--npz-dir', default='../data/cross_corr_C/')
    parser.add_argument('--out-dir', default='../data/cross_corr_C/round3a/')
    parser.add_argument('--fig-root', default='../figures/')
    args = parser.parse_args()
    main(args.npz_dir, args.out_dir, args.fig_root)
