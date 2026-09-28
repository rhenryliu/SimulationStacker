"""round3a_ck_analysis.py
=======================
Round 3A, Stage 2 analysis: reads ``data/cross_corr_C/round3a/ck_spectra_*.npz``
(written by ``make_ck_spectra.py``) and answers:

- **O-01 / P1**: is ``C_F - 1`` window smearing or mediation failure?  The
  exact split ``C_F = W_F M_F`` per filter, run, redshift, gas field and
  matter convention, and ``C(k)`` itself over the filters' response range.
- **P4 (second part)**: does ``M_F`` collapse across filters at matched
  ``k_50`` while ``W_F`` does not?
- **O-05 / P2**: is ``C(k)`` the same for every galaxy sample (the parent-mass
  halves and the other number density)?
- **O-06 / P3**: does the ``z ~ 0.26`` cross-code gap follow the number
  density or the snapshot?
- The doubly filtered coefficients of formalism Eq. (40), which are bounded by
  one for any kernel.

Upsilon and the Y transform are linear in their base kernels, so their exact
amplitudes -- including the mediated ones -- are the same linear combinations
of the Sigma and DSigma amplitudes used in ``make_calibration_factor``.  The
calibration factors are formed by ``make_calibration_factor.calibration_factors``
after ``add_convention_t``, i.e. through the round-two wiring.

Refuses any input whose regression check against round two failed.  Writes
figures under ``figures/<yyyy-mm>/<mm-dd>/`` and the numbers to
``data/cross_corr_C/round3a/stage2_ck_analysis.{npz,txt}``.

Usage
-----
    cd scripts/
    python cross_corr/round3a_ck_analysis.py
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

import kernels as kn
import rprofiles as rp
import round3a_lib as lib
from make_calibration_factor import add_convention_t, calibration_factors

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'text.usetex': False,
    'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
    'legend.fontsize': 8,
})

RUNS = (('TNG300-1', 'TNG300-1'),
        ('L1_m9_fiducial', 'FLA fid'),
        ('L1_m9_Jet_fgas-4sigma', 'FLA Jet'),
        ('L1_m9_fgas-8sigma', 'FLA fgas-8σ'))
COLOURS = dict(zip([l for l, _ in RUNS],
                   ('0.25', 'tab:blue', 'tab:orange', 'tab:red')))
SAMPLES = {'z05': {'TNG': 67, 'FLA': 67, 'z': '0.5',
                   'fid': 'LRG-like 5e-4', 'alt': 'BGS-like 1e-3'},
           'z026': {'TNG': 80, 'FLA': 71, 'z': '0.26/0.30',
                    'fid': 'BGS-like 1e-3', 'alt': 'LRG-like 5e-4'}}
FILTERS = ('DSigma', 'Sigma', 'Upsilon_R0=1', 'Upsilon_R0=2',
           'Ytransform_Rmax=4', 'Ytransform_Rmax=5', 'Ytransform_Rmax=6',
           'Ytransform_Rmax=9', 'DoG_q=2', 'DoG_q=1.5')
DATA_RANGE = (1.0, 6.0)
# Acceptance tolerances.  CLOSURE_TOL is the pre-registered closure threshold
# of predictions/2026-09-27_round3a.md (binned-route C_F against the committed
# C_F over CLOSURE_RANGE); it also bounds the binned-vs-exact M_F of the
# derived filters, whose subtractions amplify the binning error.  BINNED_M_TOL
# bounds that M_F check for the stored kernels, about twice the binning floor
# seen on the production grids.
CLOSURE_TOL = 5e-3
CLOSURE_RANGE = (1.0, 9.75)
BINNED_M_TOL = 2e-3


# ---------------------------------------------------------------------------
# Amplitudes per filter
# ---------------------------------------------------------------------------

def base_and_rule(filt):
    """Return the base filter and the derivation rule of a filter name.

    Args:
        filt (str): Filter name.

    Returns:
        tuple: ``(base, kind, ref)``; ``kind`` is None for a stored filter.
    """
    if filt.startswith('Upsilon'):
        return 'DSigma', 'Upsilon', float(filt.split('=')[1])
    if filt.startswith('Ytransform'):
        return 'Sigma', 'Ytransform', float(filt.split('=')[1])
    return filt, None, None


def derive(values, radii, kind, ref):
    """Apply the Upsilon or Y-transform linear combination to an array.

    Args:
        values (np.ndarray): Base-filter values over ``radii``.
        radii (np.ndarray): Aperture grid, arcmin.
        kind (str): ``'Upsilon'`` or ``'Ytransform'``.
        ref (float): ``R0`` or ``Rmax``.

    Returns:
        np.ndarray: Derived-filter values.
    """
    idx = int(np.flatnonzero(np.abs(radii - ref) <= 1e-9)[0])
    scale = (ref / radii) ** 2 if kind == 'Upsilon' else np.ones_like(radii)
    return values - scale * values[idx]


def filter_mask(filt, axis):
    """Usable bins of a filter, as in round two.

    Args:
        filt (str): Filter name.
        axis (np.ndarray): Aperture radii (or DoG widths).

    Returns:
        np.ndarray: Boolean mask.
    """
    _, kind, ref = base_and_rule(filt)
    if kind == 'Upsilon':
        return rp.upsilon_defined_mask(axis, ref)
    if kind == 'Ytransform':
        # Deliberately the rule of make_calibration_factor's
        # assemble_derived_filter, so the masks match round two bin for bin;
        # rp.ytransform_defined_mask differs only for an aperture exactly on
        # 0.8 Rmax, which no grid here has.
        return axis < 0.8 * ref
    return np.ones(len(axis), dtype=bool)


def amplitudes(d, filt, sample):
    """Exact amplitudes of one filter and galaxy sample, both conventions.

    Args:
        d (np.lib.npyio.NpzFile): One ``ck_spectra`` file.
        filt (str): Filter name.
        sample (str): Galaxy sample.

    Returns:
        tuple: ``(Y, Ymed, axis)``: the amplitude dict keyed like
        ``make_calibration_factor`` (Convention T pairs added), the mediated
        amplitudes ``{(X, conv): array}``, and the radius or width axis.
    """
    base, kind, ref = base_and_rule(filt)
    radii = d['radii']
    axis = d['dog_sigma1'] if base.startswith('DoG') else radii

    def get(key):
        v = d[f'{key}_{base}']
        return derive(v, radii, kind, ref) if kind else np.asarray(v)

    Y = {('m', 'm'): get('Yx_mm'), ('b', 'm'): get('Yx_bm'),
         ('b', 'b'): get('Yx_bb'), ('e', 'm'): get('Yx_em'),
         ('b', 'e'): get('Yx_be'), ('e', 'e'): get('Yx_ee'),
         ('g', 'm'): get(f'Yx_gm_{sample}'), ('b', 'g'): get(f'Yx_bg_{sample}'),
         ('e', 'g'): get(f'Yx_eg_{sample}'), ('g', 'g'): get(f'Yx_gg_{sample}')}
    add_convention_t(Y, float(d['meta_f_b']))
    Ymed = {(X, c): get(f'Ymed_{X}_{c}_{sample}')
            for X in ('b', 'e') for c in ('C', 'T')}
    return Y, Ymed, axis


def split(d, filt, sample='fid', gas='b', conv='C'):
    """``C_F``, ``W_F`` and ``M_F`` for one filter, sample, gas, convention.

    Args:
        d (np.lib.npyio.NpzFile): One ``ck_spectra`` file.
        filt (str): Filter name.
        sample (str, optional): Galaxy sample.
        gas (str, optional): 'b' or 'e'.
        conv (str, optional): 'C' or 'T'.

    Returns:
        dict: ``axis``, ``mask``, ``C``, ``W``, ``M`` (NaN outside the mask).
    """
    Y, Ymed, axis = amplitudes(d, filt, sample)
    fac = calibration_factors(Y, gas=gas)
    C = fac['C' if conv == 'C' else 'C_t']
    Y_gx = Y[rp._pair_key('g', gas)]
    with np.errstate(invalid='ignore', divide='ignore'):
        M = Ymed[(gas, conv)] / Y_gx
        W = C / M
    mask = filter_mask(filt, axis)
    out = {'axis': axis, 'mask': mask}
    for key, v in (('C', C), ('W', W), ('M', M)):
        v = np.asarray(v, dtype=float).copy()
        v[~mask] = np.nan
        out[key] = v
    return out


def binned_split(d, filt, sample='fid', gas='b', conv='C'):
    """The split rebuilt from binned spectra and binned kernel weights.

    An independent route to ``C_F``, ``W_F`` and ``M_F``.  It shares no code
    with the exact route's mediated amplitudes: Convention T is formed here by
    ``add_convention_t`` on the binned spectra, not by the per-mode
    ``f_m F_m + f_b F_b`` product of ``make_ck_spectra.py``.  The two agree to
    the binning approximation (about 1e-3 on the production grids), so a
    wiring slip in either shows up as a per-cent-level disagreement.

    Args:
        d (np.lib.npyio.NpzFile): One ``ck_spectra`` file.
        filt (str): Filter name.
        sample (str, optional): Galaxy sample.
        gas (str, optional): 'b' or 'e'.
        conv (str, optional): 'C' or 'T'.

    Returns:
        dict: As ``round3a_lib.window_mediation_split``.
    """
    base, kind, ref = base_and_rule(filt)
    w = np.asarray(d[f'w_{base}'], dtype=float)
    if kind:
        w = lib.derived_filter_weights(w, d['radii'], kind, ref)
    P = {('m', 'm'): d['P2D_mm'], ('b', 'm'): d['P2D_bm'],
         ('b', 'b'): d['P2D_bb'], ('e', 'm'): d['P2D_em'],
         ('b', 'e'): d['P2D_be'], ('g', 'm'): d[f'P2D_gm_{sample}'],
         ('b', 'g'): d[f'P2D_bg_{sample}'], ('e', 'g'): d[f'P2D_eg_{sample}']}
    P = {k: np.asarray(v, dtype=float) for k, v in P.items()}
    add_convention_t(P, float(d['meta_f_b']))
    mkey = 'm' if conv == 'C' else 't'
    return lib.window_mediation_split(
        w, P[rp._pair_key(gas, mkey)], P[rp._pair_key('g', mkey)],
        P[(mkey, mkey)], P[rp._pair_key('g', gas)],
        float(d['meta_area_arcmin2']))


def doubly_filtered(d, filt):
    """Doubly filtered coefficients (formalism Eq. 40) for the fiducial sample.

    Args:
        d (np.lib.npyio.NpzFile): One ``ck_spectra`` file.
        filt (str): 'Sigma', 'DSigma' or a DoG filter.

    Returns:
        dict: ``r_bm``, ``r_gm``, ``r_gb`` and ``C`` from ``Y2``.
    """
    y = {p: d[f'Y2_{p}_{filt}'] for p in ('bm', 'mm', 'bb', 'gm_fid',
                                          'gg_fid', 'bg_fid')}
    root = lambda a, b: np.sqrt(np.where(a * b > 0, a * b, np.nan))
    return {'r_bm': y['bm'] / root(y['bb'], y['mm']),
            'r_gm': y['gm_fid'] / root(y['gg_fid'], y['mm']),
            'r_gb': y['bg_fid'] / root(y['gg_fid'], y['bb']),
            'C': y['bm'] * y['gm_fid'] / (y['mm'] * y['bg_fid'])}


# ---------------------------------------------------------------------------
# k_50 and C(k)
# ---------------------------------------------------------------------------

def k50(d, filt):
    """Median response wavenumber of every aperture of a filter, h/Mpc.

    Analytic kernels against this run's measured CDM spectrum, the
    definition of ``make_task9_spectra`` (formalism Eq. 64).

    Args:
        d (np.lib.npyio.NpzFile): One ``ck_spectra`` file.
        filt (str): Filter name.

    Returns:
        np.ndarray: ``k_50`` per aperture (or DoG width), h/Mpc.
    """
    good = (d['counts'] > 0) & np.isfinite(d['P2D_mm'])
    k = d['k_mean'][good]
    p = d['P2D_mm'][good]
    dr = float(d['meta_dr_arcmin'])
    base, kind, ref = base_and_rule(filt)
    out = []
    if base.startswith('DoG'):
        q = float(base.split('=')[1])
        for s1 in d['dog_sigma1']:
            out.append(lib.response_quantiles_values(
                k, p, lib.dog_window(k, s1, q * s1))[1])
    else:
        for R in d['radii']:
            if base == 'DSigma':
                w = kn.w_dsigma(k, R, dr)
                if kind:
                    w = w - (ref / R) ** 2 * kn.w_dsigma(k, ref, dr)
            else:
                w = kn.w_sigma(k, R, dr)
                if kind:
                    w = w - kn.w_sigma(k, ref, dr)
            out.append(lib.response_quantiles_values(k, p, w)[1])
    return np.asarray(out) / float(d['meta_mpc_per_arcmin'])


def harmonic_C(d, samples=('fid',), gas='b', conv='C', per_decade=12):
    """``C(k)`` in logarithmic bins, spectra averaged before the ratio.

    Args:
        d (np.lib.npyio.NpzFile): One ``ck_spectra`` file.
        samples (sequence, optional): Galaxy samples.
        gas (str, optional): 'b' or 'e'.
        conv (str, optional): 'C' or 'T'.
        per_decade (int, optional): Log bins per decade.

    Returns:
        tuple: ``(k in h/Mpc, {sample: C(k)})``.
    """
    names = ['mm', 'bm', 'bb', 'em', 'be']
    for s in samples:
        names += [f'gm_{s}', f'bg_{s}', f'eg_{s}']
    kmin = np.nanmin(d['k_mean'][d['counts'] > 0])
    kmax = np.nanmax(d['k_mean'])
    edges = np.logspace(np.log10(kmin * 0.999), np.log10(kmax * 1.001),
                        int(np.ceil(np.log10(kmax / kmin) * per_decade)) + 1)
    kc, sp = lib.rebin_spectra(d['counts'], d['k_mean'],
                               {n: d[f'P2D_{n}'] for n in names}, edges)
    f_b = float(d['meta_f_b'])
    out = {}
    for s in samples:
        P = {('m', 'm'): sp['mm'], ('b', 'm'): sp['bm'], ('b', 'b'): sp['bb'],
             ('e', 'm'): sp['em'], ('b', 'e'): sp['be'],
             ('g', 'm'): sp[f'gm_{s}'], ('b', 'g'): sp[f'bg_{s}'],
             ('e', 'g'): sp[f'eg_{s}'],
             # Autos of g and e are not needed for C; placeholders keep
             # calibration_factors' coefficient slots finite.
             ('g', 'g'): np.ones_like(sp['mm']),
             ('e', 'e'): np.ones_like(sp['mm'])}
        add_convention_t(P, f_b)
        fac = calibration_factors(P, gas=gas)
        out[s] = fac['C' if conv == 'C' else 'C_t']
    return kc / float(d['meta_mpc_per_arcmin']), out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(npz_dir, zkey):
    """Load the ``ck_spectra`` files of one redshift sample.

    Args:
        npz_dir (pathlib.Path): ``data/cross_corr_C/round3a``.
        zkey (str): ``'z05'`` or ``'z026'``.

    Returns:
        dict: label to npz handle.

    Raises:
        SystemExit: If a file is missing or failed its regression check.
    """
    out = {}
    for label, _ in RUNS:
        snap = SAMPLES[zkey]['TNG' if label.startswith('TNG') else 'FLA']
        path = npz_dir / f'ck_spectra_{label}_{snap}_yz.npz'
        if not path.exists():
            raise SystemExit(f'Missing {path}; run make_ck_spectra.py first.')
        d = np.load(path, allow_pickle=True)
        lib.require_regression_pass(d, path)
        out[label] = d
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_ck(data, fig_dir, ranges):
    """``C(k)`` of the fiducial sample, both conventions, both redshifts.

    Args:
        data (dict): ``{zkey: {label: npz}}``.
        fig_dir (pathlib.Path): Output directory.
        ranges (dict): ``{zkey: (k05, k95)}`` of DSigma at 1', h/Mpc.

    Returns:
        pathlib.Path: The figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (zkey, runs) in zip(axes, data.items()):
        for label, short in RUNS:
            for conv, ls in (('C', '-'), ('T', ':')):
                k, C = harmonic_C(runs[label], conv=conv)
                ax.plot(k, C['fid'], ls, color=COLOURS[label],
                        label=short if conv == 'C' else None)
        ax.axvspan(*ranges[zkey], color='0.9', zorder=-1,
                   label=r"$\Delta\Sigma(1')$: $k_{05}$-$k_{95}$")
        ax.axhline(1.0, color='k', lw=0.8)
        ax.set_xscale('log')
        ax.set_xlim(0.1, 30)
        ax.set_ylim(0.8, 1.4)
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_title(f'z≈{SAMPLES[zkey]["z"]} ({SAMPLES[zkey]["fid"]})')
    axes[0].set_ylabel(r'$C(k) = P_{bm}P_{gm}/(P_{mm}P_{gb})$')
    axes[0].legend()
    fig.suptitle('O-01: the harmonic-space calibration factor (solid: '
                 'Convention C; dotted: Convention T)')
    fig.tight_layout()
    path = fig_dir / 'round3a_s2_Ck.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_split(data, k50s, fig_dir, filters=('DSigma', 'Sigma', 'DoG_q=2')):
    """``C_F``, ``W_F`` and ``M_F`` against ``k_50`` for three filters.

    Args:
        data (dict): ``{zkey: {label: npz}}``.
        k50s (dict): ``{(zkey, label, filt): k_50 array}``.
        fig_dir (pathlib.Path): Output directory.
        filters (tuple, optional): Filters to show.

    Returns:
        pathlib.Path: The figure.
    """
    fig, axes = plt.subplots(2, len(filters), figsize=(16, 8.5), sharey=True)
    for row, (zkey, runs) in enumerate(data.items()):
        for col, filt in enumerate(filters):
            ax = axes[row, col]
            for label, short in RUNS:
                s = split(runs[label], filt)
                k = k50s[(zkey, label, filt)]
                ax.plot(k, s['M'], '-', color=COLOURS[label], lw=2,
                        label=f'{short}: M' if col == 0 else None)
                ax.plot(k, s['W'], '--', color=COLOURS[label], lw=1.2,
                        label='W' if (col == 0 and label == 'TNG300-1')
                        else None)
                ax.plot(k, s['C'], ':', color=COLOURS[label], lw=1.2,
                        label='C = W M' if (col == 0 and label == 'TNG300-1')
                        else None)
            ax.axhline(1.0, color='k', lw=0.8)
            ax.set_xscale('log')
            ax.set_title(f'{filt}, z≈{SAMPLES[zkey]["z"]}')
            if row == 1:
                ax.set_xlabel(r'$k_{50}$ [$h$/Mpc]')
        axes[row, 0].set_ylabel('factor (baryons, Convention C)')
    axes[0, 0].legend(ncol=2)
    fig.suptitle('O-01: the filtered calibration factor split into its '
                 'window part W and mediation part M')
    fig.tight_layout()
    path = fig_dir / 'round3a_s2_split.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_samples(data, fig_dir):
    """``C(k)`` for the four galaxy samples, per run.

    Args:
        data (dict): ``{zkey: {label: npz}}``.
        fig_dir (pathlib.Path): Output directory.

    Returns:
        pathlib.Path: The figure.
    """
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)
    styles = {'fid': ('k', '-'), 'alt': ('tab:green', '-.'),
              'lo': ('tab:blue', '--'), 'hi': ('tab:red', '--')}
    for row, (zkey, runs) in enumerate(data.items()):
        for col, (label, short) in enumerate(RUNS):
            ax = axes[row, col]
            k, C = harmonic_C(runs[label], samples=('fid', 'alt', 'lo', 'hi'))
            for s, (c, ls) in styles.items():
                name = {'fid': SAMPLES[zkey]['fid'], 'alt': SAMPLES[zkey]['alt'],
                        'lo': 'fid, low parent mass', 'hi': 'fid, high parent mass'}[s]
                ax.plot(k, C[s], ls, color=c, label=name)
            ax.axhline(1.0, color='0.5', lw=0.8)
            ax.set_xscale('log')
            ax.set_xlim(0.1, 30)
            ax.set_ylim(0.8, 1.5)
            ax.set_title(f'{short}, z≈{SAMPLES[zkey]["z"]}')
            if row == 1:
                ax.set_xlabel(r'$k$ [$h$/Mpc]')
        axes[row, 0].set_ylabel(r'$C(k)$')
    axes[0, 0].legend()
    fig.suptitle('O-05: mediation requires one C(k) for every galaxy sample')
    fig.tight_layout()
    path = fig_dir / 'round3a_s2_samples.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_density_swap(o6, fig_dir):
    """Cross-code ``dC/C`` for each (snapshot, density), baryons and electrons.

    Args:
        o6 (dict): ``{(zkey, sample, gas): (radii, dC/C)}``.
        fig_dir (pathlib.Path): Output directory.

    Returns:
        pathlib.Path: The figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    styles = {('z05', 'fid'): ('tab:blue', '-'), ('z05', 'alt'): ('tab:blue', '--'),
              ('z026', 'fid'): ('tab:red', '-'), ('z026', 'alt'): ('tab:red', '--')}
    for ax, gas in zip(axes, ('b', 'e')):
        for (zkey, s), (c, ls) in styles.items():
            radii, rel = o6[(zkey, s, gas)]
            ax.plot(radii, rel, ls, color=c, marker='o', ms=3,
                    label=f'z≈{SAMPLES[zkey]["z"]}, {SAMPLES[zkey][s]}')
        ax.axhline(0.0, color='k', lw=0.8)
        ax.set_xlabel('R [arcmin]')
        ax.set_title(f'gas = {"baryons" if gas == "b" else "electrons"}, '
                     r'$\Delta\Sigma$')
    axes[0].set_ylabel(r'$(C_{\rm TNG} - C_{\rm FLA\,fid})/C$')
    axes[0].legend()
    fig.suptitle('O-06: does the cross-code gap follow the snapshot or the '
                 'number density?')
    fig.tight_layout()
    path = fig_dir / 'round3a_s2_density_swap.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyse(data, cal_dir):
    """Compute every Stage 2 number.

    Args:
        data (dict): ``{zkey: {label: npz}}``.
        cal_dir (pathlib.Path): Round-two calibration npz directory.

    Returns:
        dict: Results keyed by test.
    """
    res = {}
    for zkey, runs in data.items():
        # k_50 of every filter per run.
        for label in runs:
            for filt in FILTERS:
                res[('k50', zkey, label, filt)] = k50(runs[label], filt)
        # DSigma(1') response range, fiducial run.
        good = (runs['L1_m9_fiducial']['counts'] > 0)
        d = runs['L1_m9_fiducial']
        k = d['k_mean'][good]
        q = lib.response_quantiles_values(
            k, d['P2D_mm'][good], kn.w_dsigma(k, 1.0, float(d['meta_dr_arcmin'])))
        res[('range1', zkey)] = (q[0] / float(d['meta_mpc_per_arcmin']),
                                 q[2] / float(d['meta_mpc_per_arcmin']))

        for label, d in runs.items():
            for filt in FILTERS:
                for gas in ('b', 'e'):
                    for conv in ('C', 'T'):
                        res[('split', zkey, label, filt, gas, conv)] = split(
                            d, filt, gas=gas, conv=conv)
            for filt in ('Sigma', 'DSigma', 'DoG_q=2', 'DoG_q=1.5'):
                res[('r2', zkey, label, filt)] = doubly_filtered(d, filt)
            k_h, Cs = harmonic_C(d, samples=('fid', 'alt', 'lo', 'hi'))
            res[('Ck', zkey, label)] = (k_h, Cs)
            k_h, Ce = harmonic_C(d, samples=('fid',), gas='e')
            res[('Ck_e', zkey, label)] = (k_h, Ce['fid'])
            # Closure: exact C_F against the committed round-two C_F.
            snap = int(d['meta_snapshot'])
            cal = np.load(cal_dir / f'calibration_{label}_{snap}_yz.npz',
                          allow_pickle=True)
            worst = 0.0
            for filt in FILTERS[:8]:
                s = res[('split', zkey, label, filt, 'b', 'C')]
                ref = cal[f'C_b_{filt}']
                ok = np.isfinite(ref) & np.isfinite(s['C'])
                worst = max(worst, float(np.max(np.abs(s['C'][ok] / ref[ok]
                                                       - 1.0))))
            res[('closure', zkey, label)] = worst

            # Pre-registered closure: the binned-route C_F against the
            # committed C_F, stored and derived filters apart.
            radii = d['radii']
            in_range = ((radii >= CLOSURE_RANGE[0] - 1e-9)
                        & (radii <= CLOSURE_RANGE[1] + 1e-9))
            worst_bc = [0.0, 0.0]
            for filt in FILTERS[:8]:
                bn = binned_split(d, filt, 'fid', 'b', 'C')
                ref = np.asarray(cal[f'C_b_{filt}'], dtype=float)
                ok = (in_range & filter_mask(filt, radii) & np.isfinite(ref)
                      & np.isfinite(bn['C']))
                if ok.any():
                    i = 0 if base_and_rule(filt)[1] is None else 1
                    worst_bc[i] = max(worst_bc[i], float(np.max(np.abs(
                        bn['C'][ok] / ref[ok] - 1.0))))
            res[('closure_binned', zkey, label)] = tuple(worst_bc)

            # Independent cross-check of the mediated amplitudes: the binned
            # route against the exact one, every filter, gas and convention.
            worst_base, worst_derived = 0.0, 0.0
            for filt in FILTERS:
                for gas in ('b', 'e'):
                    for conv in ('C', 'T'):
                        ex = res[('split', zkey, label, filt, gas, conv)]
                        bn = binned_split(d, filt, 'fid', gas, conv)
                        ok = (ex['mask'] & np.isfinite(ex['M'])
                              & np.isfinite(bn['M']))
                        dm = float(np.max(np.abs(bn['M'][ok] - ex['M'][ok])))
                        if base_and_rule(filt)[1] is None:
                            worst_base = max(worst_base, dm)
                        else:
                            worst_derived = max(worst_derived, dm)
            res[('binned_check', zkey, label)] = (worst_base, worst_derived)

        # O-06: cross-code dC/C, fiducial and swapped densities.
        for sample in ('fid', 'alt'):
            for gas in ('b', 'e'):
                st = split(runs['TNG300-1'], 'DSigma', sample, gas)
                sf = split(runs['L1_m9_fiducial'], 'DSigma', sample, gas)
                radii = runs['TNG300-1']['radii']
                keep = ((radii >= DATA_RANGE[0] - 1e-9)
                        & (radii <= DATA_RANGE[1] + 1e-9))
                rel = (st['C'] - sf['C']) / (0.5 * (st['C'] + sf['C']))
                res[('o6', zkey, sample, gas)] = (radii[keep], rel[keep])
    return res


def report(data, res):
    """Print the Stage 2 numbers, scored against the predictions file.

    Args:
        data (dict): ``{zkey: {label: npz}}``.
        res (dict): Output of :func:`analyse`.
    """
    print('=' * 78)
    print('Round 3A, Stage 2: C(k) and the window/mediation split')
    print('=' * 78)
    for zkey, runs in data.items():
        z = SAMPLES[zkey]['z']
        print(f'\n---------------- z ≈ {z} ----------------')
        print('\n[acceptance] regression vs round two (worst relative diff) '
              'and exact-C_F closure vs committed C_F:')
        for label, short in RUNS:
            d = runs[label]
            print(f"  {short:12s} regression {float(d['reg_maxrel_all']):.1e} "
                  f"(pass {bool(d['reg_pass'])}); C_F closure "
                  f"{res[('closure', zkey, label)]:.1e}")
        print('\n[acceptance] binned route vs exact route for M_F (independent '
              'code paths; every filter, gas and convention), worst |dM| over '
              f'usable bins; threshold {BINNED_M_TOL:g} for the stored '
              f'kernels, {CLOSURE_TOL:g} for the derived filters:')
        for label, short in RUNS:
            wb, wd = res[('binned_check', zkey, label)]
            print(f'  {short:12s} Sigma/DSigma/DoG {wb:.1e} '
                  f"({'PASS' if wb < BINNED_M_TOL else 'FAIL'}); "
                  f'Upsilon/Y transform {wd:.1e} '
                  f"({'PASS' if wd < CLOSURE_TOL else 'FAIL'})")
        print('\n[acceptance] pre-registered closure: binned-route C_F against '
              f"the committed C_F over {CLOSURE_RANGE[0]:g}'-"
              f"{CLOSURE_RANGE[1]:g}' (baryons, Convention C), threshold "
              f'{CLOSURE_TOL:g}; the split itself uses the exact route:')
        for label, short in RUNS:
            wb, wd = res[('closure_binned', zkey, label)]
            print(f'  {short:12s} Sigma/DSigma {wb:.1e} '
                  f"({'PASS' if wb < CLOSURE_TOL else 'FAIL'}); "
                  f'Upsilon/Y transform {wd:.1e} '
                  f"({'PASS' if wd < CLOSURE_TOL else 'FAIL'})")

        k05, k95 = res[('range1', zkey)]
        print(f"\n[P1] DSigma, baryons, Convention C, R = 1'.  f_W = ln W/ln C "
              f"(window >= 0.8, mediation <= 0.2).  C(k) over "
              f"DSigma(1')'s k05-k95 = {k05:.2f}-{k95:.2f} h/Mpc:")
        for label, short in RUNS:
            s = res[('split', zkey, label, 'DSigma', 'b', 'C')]
            C, W, M = s['C'][0], s['W'][0], s['M'][0]
            fW = np.log(W) / np.log(C)
            k, Cs = res[('Ck', zkey, label)]
            sel = (k >= k05) & (k <= k95)
            ck = Cs['fid'][sel]
            print(f"  {short:12s} C {C:.4f} = W {W:.4f} x M {M:.4f}; f_W "
                  f"{fW:+.2f}; C(k) in range: mean {np.mean(ck):.3f}, "
                  f"max |C-1| {np.max(np.abs(ck - 1)):.3f}")

        print('\n[O-01] M_F and W_F over the data range, DSigma, baryons, '
              'Convention C:')
        radii = runs['TNG300-1']['radii']
        show = [i for i, R in enumerate(radii)
                if R in (1.0, 1.625, 2.25, 3.5, 4.75, 6.0)]
        print('  R      ' + ''.join(f'{radii[i]:>14.3f}' for i in show))
        for label, short in RUNS:
            s = res[('split', zkey, label, 'DSigma', 'b', 'C')]
            print(f'  {short:12s}M ' + ''.join(f'{s["M"][i]:14.4f}' for i in show))
            print(f'  {"":12s}W ' + ''.join(f'{s["W"][i]:14.4f}' for i in show))

        print("\n[O-01] At R = 1' (DoG: smallest sigma1), baryons: C / W / M "
              'by filter and convention (FLA fid | TNG300-1)')
        for filt in FILTERS:
            parts = []
            for label in ('L1_m9_fiducial', 'TNG300-1'):
                for conv in ('C', 'T'):
                    s = res[('split', zkey, label, filt, 'b', conv)]
                    i = int(np.flatnonzero(s['mask'])[0])
                    parts.append(f"{conv}:{s['C'][i]:.3f}/{s['W'][i]:.3f}/"
                                 f"{s['M'][i]:.3f}")
            print(f'  {filt:18s} ' + ' | '.join(parts))

        print('\n[O-01] Electrons, DSigma, Convention C at 1\': C / W / M')
        for label, short in RUNS:
            s = res[('split', zkey, label, 'DSigma', 'e', 'C')]
            print(f"  {short:12s} {s['C'][0]:.4f} / {s['W'][0]:.4f} / "
                  f"{s['M'][0]:.4f}")

        print('\n[P4b] Spread of M_F and W_F across Sigma, DSigma and DoG_q=2 '
              'at matched k_50 (baryons, Conv. C); P4 threshold 0.01 on M:')
        for label, short in RUNS:
            k_ds = res[('k50', zkey, label, 'DSigma')]
            sd = res[('split', zkey, label, 'DSigma', 'b', 'C')]
            dm, dw = [], []
            for filt in ('Sigma', 'DoG_q=2'):
                sf = res[('split', zkey, label, filt, 'b', 'C')]
                kf = res[('k50', zkey, label, filt)]
                dm.append(np.abs(lib.interp_log_k(kf, sf['M'], k_ds)
                                 - sd['M']))
                dw.append(np.abs(lib.interp_log_k(kf, sf['W'], k_ds)
                                 - sd['W']))
            print(f"  {short:12s} max |dM| {np.nanmax(np.concatenate(dm)):.4f}"
                  f"   max |dW| {np.nanmax(np.concatenate(dw)):.4f}")

        print('\n[P2 / O-05] C(k) by galaxy sample (baryons, Conv. C) at '
              'k = 1, 2, 3.5, 5, 7 h/Mpc; and max |C_lo - C_hi| over 3-7 h/Mpc')
        for label, short in RUNS:
            k, Cs = res[('Ck', zkey, label)]
            idx = [int(np.argmin(np.abs(k - kk))) for kk in (1, 2, 3.5, 5, 7)]
            sel = (k >= 3) & (k <= 7)
            diff = Cs['lo'][sel] - Cs['hi'][sel]
            line = '  '.join(f"{s}:" + ','.join(f'{Cs[s][i]:.3f}' for i in idx)
                             for s in ('fid', 'alt', 'lo', 'hi'))
            print(f'  {short:12s} {line}')
            print(f'  {"":12s} C_lo - C_hi over 3-7 h/Mpc: mean '
                  f'{np.mean(diff):+.3f}, max |.| {np.max(np.abs(diff)):.3f}')

        print("\n[P3 / O-06] Cross-code (C_TNG - C_fid)/C, DSigma: at 1' and "
              "max over 1'-6'")
        for sample in ('fid', 'alt'):
            for gas in ('b', 'e'):
                radii6, rel = res[('o6', zkey, sample, gas)]
                i = int(np.nanargmax(np.abs(rel)))
                print(f"  {SAMPLES[zkey][sample]:14s} gas={gas}: at 1' "
                      f"{rel[0]:+.3f}; max |.| {abs(rel[i]):.3f} at "
                      f"{radii6[i]:.3f}'")

        print('\n[Eq. 40] Doubly filtered coefficients at the smallest scale '
              '(bounded by 1 for any kernel) and C~:')
        for label, short in RUNS:
            parts = []
            for filt in ('DSigma', 'Sigma', 'DoG_q=2'):
                r2 = res[('r2', zkey, label, filt)]
                parts.append(f"{filt}: r_bm {r2['r_bm'][0]:.3f} r_gm "
                             f"{r2['r_gm'][0]:.3f} r_gb {r2['r_gb'][0]:.3f} "
                             f"C~ {r2['C'][0]:.3f}")
            print(f'  {short:12s} ' + '; '.join(parts))


def main(npz_dir, cal_dir, out_dir, fig_root, verbose=True):
    """Run the Stage 2 analysis.

    Args:
        npz_dir (str): Directory of the ``ck_spectra`` files.
        cal_dir (str): Directory of the round-two calibration files.
        out_dir (str): Output directory for the numbers.
        fig_root (str): Root of the dated figure directories.
        verbose (bool, optional): Print the report.
    """
    npz_dir, cal_dir, out_dir = Path(npz_dir), Path(cal_dir), Path(out_dir)
    now = datetime.now()
    fig_dir = Path(fig_root) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = {z: load(npz_dir, z) for z in SAMPLES}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        res = analyse(data, cal_dir)
        k50s = {(z, l, f): res[('k50', z, l, f)] for z in data
                for l, _ in RUNS for f in FILTERS}
        paths = [fig_ck(data, fig_dir, {z: res[('range1', z)] for z in data}),
                 fig_split(data, k50s, fig_dir),
                 fig_samples(data, fig_dir),
                 fig_density_swap({(z, s, g): res[('o6', z, s, g)]
                                   for z in data for s in ('fid', 'alt')
                                   for g in ('b', 'e')}, fig_dir)]
        buf = io.StringIO()
        with redirect_stdout(buf):
            report(data, res)
    text = buf.getvalue()
    if verbose:
        print(text)
        for p in paths:
            print(f'Wrote {p}')

    flat = {}
    for key, value in res.items():
        name = '__'.join(str(k) for k in key)
        if key[0] == 'split':
            for sub in ('axis', 'C', 'W', 'M'):
                flat[f'{name}__{sub}'] = value[sub]
        elif key[0] == 'r2':
            for sub, arr in value.items():
                flat[f'{name}__{sub}'] = arr
        elif key[0] == 'Ck':
            flat[f'{name}__k'] = value[0]
            for s, arr in value[1].items():
                flat[f'{name}__{s}'] = arr
        elif key[0] == 'Ck_e':
            flat[f'{name}__k'], flat[f'{name}__fid'] = value
        elif key[0] == 'o6':
            flat[f'{name}__radii'], flat[f'{name}__rel'] = value
        else:
            flat[name] = np.asarray(value)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib.save_npz_atomic(out_dir / 'stage2_ck_analysis.npz', **flat)
    lib.write_text_atomic(out_dir / 'stage2_ck_analysis.txt', text)
    print(f"Wrote {out_dir / 'stage2_ck_analysis.txt'} and .npz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Round 3A Stage 2 analysis: C(k) and the split.')
    parser.add_argument('--npz-dir', default='../data/cross_corr_C/round3a/')
    parser.add_argument('--cal-dir', default='../data/cross_corr_C/')
    parser.add_argument('--out-dir', default='../data/cross_corr_C/round3a/')
    parser.add_argument('--fig-root', default='../figures/')
    args = parser.parse_args()
    main(args.npz_dir, args.cal_dir, args.out_dir, args.fig_root)
