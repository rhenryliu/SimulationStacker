"""make_pk_fstar_obs_grid.py
==========================
Alternative layouts of the unbound gas paper's P(k)-section figure (S(k) and
the lensing gas fraction with the observation-based stellar bands), drawn
from the same products as make_pk_fstar_obs_column.py (make_pk_fstar_obs's
``analyse``); that script and its single-column figure are unchanged.

Layouts (``--layout``):
  A  2x2, by suite: IllustrisTNG (left) and FLAMINGO (right); S(k) on top,
     f_gas(theta) below; each simulation with its rescaled ends and band, as
     in the column figure. Full width.
  B  2x2, absolute values (left: S(k), f_gas(theta) of every simulation, and
     the measurement) and the stellar rescaling as differences from each
     simulation (right: Delta S(k), Delta f_gas(theta), bands between the low
     and high ends). ``--data-band`` shades the measurement's +-1 sigma in the
     Delta f_gas panel. Full width.
  C  the column figure, thinned: bands without dashed/dotted outlines, no
     markers, k from ``--kmin``, legend above the panels. One column.
  D  2x2: B's left column (every simulation, and the measurement), and on the
     right C's panels (each simulation with the band between its rescaled
     ends, no measurement); rows share their y axis. ``--edges thin`` draws
     the band edges as thin lines in the simulation's colour (default: none).
     Full width.

Output: <fig_path>/YYYY-MM/MM-DD/<fig_name>_<option>_grid_<layout>[_databand|_edges].<ext>

Run from the scripts/ directory (light; login node is fine):
    python unbound_gas/make_pk_fstar_obs_grid.py -p configs/unbound_gas/pk_fstar_obs_z05_lensfit.yaml --layout B
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import compute_fstar_obs as cfo
import make_pk_fstar_obs as mfo
import make_pk_fstar_obs_column as col  # labels, aggregate symbols and column fonts
import make_pk_stellar as mps  # colours and axis style
from compute_stellar_maps import lensing_settings
from pk_common import load_config, select_sims


def _ends(r: dict, option: str):
    """Low and high ends of one option for a result."""
    return r['ends'][f"{option}__low"], r['ends'][f"{option}__high"]


def _style_keys(T_lo: float, T_hi: float, sym: str, fmt: str, lens_data, delta=False):
    """Legend handles for the line styles (and the measurement)."""
    h = [plt.Line2D([], [], color='gray', lw=1.3),
         plt.Line2D([], [], color='gray', lw=0.9, ls='--'),
         plt.Line2D([], [], color='gray', lw=0.9, ls=':')]
    lab = ['simulation', rf'${sym}={format(T_lo, fmt)}$', rf'${sym}={format(T_hi, fmt)}$']
    if lens_data is not None:
        h.append(plt.Line2D([], [], color='k', marker='s', ls='', ms=3))
        lab.append(r'DESI$\times$ACT$\times$HSC')
    return h, lab


def _k_axis(ax, kmin: float, kmax: float) -> None:
    mps._style_axis(ax)
    ax.set_xlim(kmin, kmax)


def _f_axis(ax, lens_xmax: float) -> None:
    ax.grid(True, which='major', color='0.8', lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0.0, lens_xmax)
    ax.set_xlabel(r'$\theta\;[\mathrm{arcmin}]$', labelpad=1)


def _data(ax, lens_data) -> None:
    if lens_data is not None:
        ax.errorbar(lens_data['theta'], lens_data['f'], yerr=lens_data['err'], fmt='s',
                    color='k', ms=3, capsize=1.5, elinewidth=0.8, zorder=5)


def _sim_with_band(ax, x, y0, ylo, yhi, colour, marker=True, outlines=True):
    """A simulation (solid) with its rescaled ends and the band between them."""
    ax.plot(x, y0, color=colour, lw=1.3, marker='o' if marker else None, ms=2.5)
    if outlines:
        ax.plot(x, ylo, color=colour, lw=0.9, ls='--')
        ax.plot(x, yhi, color=colour, lw=0.9, ls=':')
    ax.fill_between(x, ylo, yhi, color=colour, alpha=0.2, lw=0)


def layout_A(have, option, kmin, kmax, lens_data, lens_xmax, path):
    sym, own, fmt = col.AGG_SYMBOL[option], _own(option), _fmt(option)
    groups = [('IllustrisTNG', [r for r in have if r['sim_type'] == 'IllustrisTNG']),
              ('FLAMINGO', [r for r in have if r['sim_type'] == 'FLAMINGO'])]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharey='row',
                             gridspec_kw=dict(hspace=0.32, wspace=0.08))
    for c, (title, rs) in enumerate(groups):
        ax1, ax2 = axes[0, c], axes[1, c]
        for r in rs:
            colour = mps.sim_colour(r)
            el, eh = _ends(r, option)
            sel = (r['k'] >= kmin) & (r['k'] <= kmax)
            _sim_with_band(ax1, r['k'][sel], r['S0'][sel], el['S'][sel], eh['S'][sel], colour,
                           marker=False)
            ax1.plot([], [], color=colour, lw=1.3,
                     label=rf"{col.paper_label(r)} (${sym}={format(r['obs'][own], fmt)}$)")
            L = r['lens']
            _sim_with_band(ax2, L['theta'], L['f'], el['f'], eh['f'], colour)
        _data(ax2, lens_data)
        ax1.axhline(1.0, color='k', lw=0.7)
        ax2.axhline(1.0, color='k', lw=0.7)
        _k_axis(ax1, kmin, kmax)
        ax1.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$', labelpad=1)
        ax1.set_title(title)
        _f_axis(ax2, lens_xmax)
        ax1.legend(loc='lower left', framealpha=0.85, handlelength=1.4, borderpad=0.4,
                   labelspacing=0.25)
    axes[0, 0].set_ylabel(r'$S(k) = P_{\rm mm}/P_{\rm DMO}$')
    axes[1, 0].set_ylabel(r'$f_{\rm gas}(\theta)\,/\,(\Omega_b/\Omega_m)$')
    T_lo, T_hi = _ends(have[0], option)[0]['target'], _ends(have[0], option)[1]['target']
    h, lab = _style_keys(T_lo, T_hi, sym, fmt, lens_data)
    fig.legend(h, lab, loc='upper center', ncol=len(h), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"saved {path}")


def layout_B(have, option, kmin, kmax, lens_data, lens_xmax, path, data_band=False):
    sym, own, fmt = col.AGG_SYMBOL[option], _own(option), _fmt(option)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6),
                             gridspec_kw=dict(hspace=0.32, wspace=0.28))
    (aS, adS), (af, adf) = axes
    for r in have:
        colour = mps.sim_colour(r)
        el, eh = _ends(r, option)
        sel = (r['k'] >= kmin) & (r['k'] <= kmax)
        k = r['k'][sel]
        aS.plot(k, r['S0'][sel], color=colour, lw=1.3,
                label=rf"{col.paper_label(r)} (${sym}={format(r['obs'][own], fmt)}$)")
        adS.plot(k, el['S'][sel] - r['S0'][sel], color=colour, lw=0.9, ls='--')
        adS.plot(k, eh['S'][sel] - r['S0'][sel], color=colour, lw=0.9, ls=':')
        adS.fill_between(k, el['S'][sel] - r['S0'][sel], eh['S'][sel] - r['S0'][sel],
                         color=colour, alpha=0.2, lw=0)
        L = r['lens']
        af.plot(L['theta'], L['f'], color=colour, lw=1.3, marker='o', ms=2.5)
        adf.plot(L['theta'], el['f'] - L['f'], color=colour, lw=0.9, ls='--')
        adf.plot(L['theta'], eh['f'] - L['f'], color=colour, lw=0.9, ls=':')
        adf.fill_between(L['theta'], el['f'] - L['f'], eh['f'] - L['f'], color=colour,
                         alpha=0.2, lw=0)
    _data(af, lens_data)
    if data_band and lens_data is not None:
        adf.fill_between(lens_data['theta'], -lens_data['err'], lens_data['err'],
                         color='0.6', alpha=0.25, lw=0, zorder=0)
        adf.plot([], [], color='0.6', lw=5, alpha=0.4, label=r'measurement $\pm 1\sigma$')
        adf.legend(loc='upper right', framealpha=0.85, handlelength=1.4)
    aS.axhline(1.0, color='k', lw=0.7)
    af.axhline(1.0, color='k', lw=0.7)
    adS.axhline(0.0, color='k', lw=0.7)
    adf.axhline(0.0, color='k', lw=0.7)
    for ax in (aS, adS):
        _k_axis(ax, kmin, kmax)
        ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$', labelpad=1)
    for ax in (af, adf):
        _f_axis(ax, lens_xmax)
    aS.set_ylabel(r'$S(k) = P_{\rm mm}/P_{\rm DMO}$')
    adS.set_ylabel(r'$\Delta S(k)$ (rescaled $-$ simulation)')
    af.set_ylabel(r'$f_{\rm gas}(\theta)\,/\,(\Omega_b/\Omega_m)$')
    adf.set_ylabel(r'$\Delta f_{\rm gas}(\theta)\,/\,(\Omega_b/\Omega_m)$')
    aS.set_title('simulations')
    adS.set_title('stellar rescaling')
    T_lo, T_hi = _ends(have[0], option)[0]['target'], _ends(have[0], option)[1]['target']
    hs, ls = aS.get_legend_handles_labels()
    h, lab = _style_keys(T_lo, T_hi, sym, fmt, lens_data)
    fig.legend(hs + h, ls + lab, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"saved {path}")


def layout_C(have, option, kmin, kmax, lens_data, lens_xmax, path):
    sym, own, fmt = col.AGG_SYMBOL[option], _own(option), _fmt(option)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 5.6),
                                   gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.3))
    for r in have:
        colour = mps.sim_colour(r)
        el, eh = _ends(r, option)
        sel = (r['k'] >= kmin) & (r['k'] <= kmax)
        _sim_with_band(ax1, r['k'][sel], r['S0'][sel], el['S'][sel], eh['S'][sel], colour,
                       marker=False, outlines=False)
        ax1.plot([], [], color=colour, lw=1.3,
                 label=rf"{col.paper_label(r)} (${sym}={format(r['obs'][own], fmt)}$)")
        L = r['lens']
        _sim_with_band(ax2, L['theta'], L['f'], el['f'], eh['f'], colour, marker=False,
                       outlines=False)
    _data(ax2, lens_data)
    ax1.axhline(1.0, color='k', lw=0.7)
    ax2.axhline(1.0, color='k', lw=0.7)
    _k_axis(ax1, kmin, kmax)
    ax1.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$', labelpad=1)
    ax1.set_ylabel(r'$S(k) = P_{\rm mm}/P_{\rm DMO}$')
    _f_axis(ax2, lens_xmax)
    ax2.set_ylabel(r'$f_{\rm gas}(\theta)\,/\,(\Omega_b/\Omega_m)$')
    T_lo, T_hi = _ends(have[0], option)[0]['target'], _ends(have[0], option)[1]['target']
    hs, ls = ax1.get_legend_handles_labels()
    band = plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.3, lw=0)
    h = [band] + ([plt.Line2D([], [], color='k', marker='s', ls='', ms=3)] if lens_data is not None else [])
    lab = [rf'${sym}={format(T_lo, fmt)}$ to ${format(T_hi, fmt)}$'] + \
          ([r'DESI$\times$ACT$\times$HSC'] if lens_data is not None else [])
    fig.legend(hs + h, ls + lab, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.99))
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"saved {path}")


def layout_D(have, option, kmin, kmax, lens_data, lens_xmax, path, edges='none'):
    sym, own, fmt = col.AGG_SYMBOL[option], _own(option), _fmt(option)
    # 7.0 x 4.5 in, the paper figure (2026-10-08; was 5.6 in tall).
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharey='row',
                             gridspec_kw=dict(hspace=0.32, wspace=0.08))
    (aS, bS), (af, bf) = axes
    for r in have:
        colour = mps.sim_colour(r)
        el, eh = _ends(r, option)
        sel = (r['k'] >= kmin) & (r['k'] <= kmax)
        k, L = r['k'][sel], r['lens']
        aS.plot(k, r['S0'][sel], color=colour, lw=1.3,
                label=rf"{col.paper_label(r)} (${sym}={format(r['obs'][own], fmt)}$)")
        af.plot(L['theta'], L['f'], color=colour, lw=1.3, marker='o', ms=2.5)
        for ax, x, y0, ylo, yhi in ((bS, k, r['S0'][sel], el['S'][sel], eh['S'][sel]),
                                    (bf, L['theta'], L['f'], el['f'], eh['f'])):
            ax.plot(x, y0, color=colour, lw=1.3)
            ax.fill_between(x, ylo, yhi, color=colour, alpha=0.2, lw=0)
            if edges == 'thin':
                ax.plot(x, ylo, color=colour, lw=0.4)
                ax.plot(x, yhi, color=colour, lw=0.4)
    _data(af, lens_data)
    for ax in (aS, bS, af, bf):
        ax.axhline(1.0, color='k', lw=0.7)
    for ax in (aS, bS):
        _k_axis(ax, kmin, kmax)
        ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$', labelpad=1)
    for ax in (af, bf):
        _f_axis(ax, lens_xmax)
    aS.set_ylabel(r'$S(k) = P_{\rm mm}/P_{\rm DMO}$')
    af.set_ylabel(r'$f_{\rm gas}(\theta)\,/\,(\Omega_b/\Omega_m)$')
    aS.set_title('simulations', fontsize=9)
    bS.set_title('stellar rescaling', fontsize=9)
    T_lo, T_hi = _ends(have[0], option)[0]['target'], _ends(have[0], option)[1]['target']
    hs, ls = aS.get_legend_handles_labels()
    h = [plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.3, lw=0)]
    lab = [rf'${sym}={format(T_lo, fmt)}$ to ${format(T_hi, fmt)}$']
    if lens_data is not None:
        h.append(plt.Line2D([], [], color='k', marker='s', ls='', ms=3))
        lab.append(r'DESI$\times$ACT$\times$HSC')
    fig.legend(hs + h, ls + lab, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.92))
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"saved {path}")


def _own(option: str) -> str:
    return 'fstar_sim' if option == 'fstar' else 'mstar_m200m_sim'


def _fmt(option: str) -> str:
    return '.2f' if option == 'fstar' else '.3f'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--layout', choices=['A', 'B', 'C', 'D'], required=True)
    parser.add_argument('--option', default='fstar', choices=list(cfo.OPTIONS))
    parser.add_argument('--kmin', type=float, default=0.1, help="smallest k plotted [h/Mpc] (default 0.1)")
    parser.add_argument('--kmax', type=float, default=5.0, help="largest k plotted [h/Mpc] (default 5)")
    parser.add_argument('--data-band', action='store_true',
                        help="layout B: shade the measurement's +-1 sigma in the Delta f_gas panel")
    parser.add_argument('--edges', choices=['none', 'thin'], default='none',
                        help="layout D: band edges as thin lines (default: none)")
    parser.add_argument('--ext', default=None, help="file type (default: the config's fig_type)")
    args = parser.parse_args()

    config = load_config(args.path2config)
    for key in ('fstar_obs', 'lensing'):
        if key not in config:
            raise SystemExit(f"{args.path2config} has no '{key}' block")
    plot_cfg = config['plot']
    now = datetime.now()
    out_dir = Path(plot_cfg['fig_path']) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.ext or plot_cfg.get('fig_type', 'pdf')
    lens = lensing_settings(config)
    lens_xmax = lens['stack']['max_radius'] * lens['stack']['rad_distance'] + 0.5
    lens_data = None
    if lens['data'] and Path(lens['data']).exists():
        with np.load(lens['data']) as dd:
            lens_data = dict(theta=dd['theta_arcmin'], f=dd['R_compensated'], err=dd['sigma_compensated'])

    results = [r for r in (mfo.analyse(e, config, False) for e in select_sims(config))
               if r is not None]
    have = [r for r in results if f"{args.option}__low" in r['ends']]
    # The legends name one pair of targets: every simulation must share it.
    targets = {(e[0]['target'], e[1]['target']) for e in (_ends(r, args.option) for r in have)}
    if len(targets) > 1:
        raise SystemExit(f"simulations have different {args.option} targets: {sorted(targets)}")
    missing = [r['label'] for r in have if r.get('lens') is None
               or any(e.get('f') is None for e in _ends(r, args.option))]
    if not have or missing:
        raise SystemExit(f"no results, or lensing stacks missing for {missing}")
    suffix = ('_databand' if (args.layout == 'B' and args.data_band) else
              '_edges' if (args.layout == 'D' and args.edges == 'thin') else '')
    path = out_dir / f"{plot_cfg['fig_name']}_{args.option}_grid_{args.layout}{suffix}.{ext}"
    if args.layout == 'A':
        layout_A(have, args.option, args.kmin, args.kmax, lens_data, lens_xmax, path)
    elif args.layout == 'B':
        layout_B(have, args.option, args.kmin, args.kmax, lens_data, lens_xmax, path,
                 data_band=args.data_band)
    elif args.layout == 'C':
        layout_C(have, args.option, args.kmin, args.kmax, lens_data, lens_xmax, path)
    else:
        layout_D(have, args.option, args.kmin, args.kmax, lens_data, lens_xmax, path,
                 edges=args.edges)


if __name__ == '__main__':
    main()
