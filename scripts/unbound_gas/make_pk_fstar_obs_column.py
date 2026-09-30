"""make_pk_fstar_obs_column.py
============================
Single-column (REVTeX reprint, 3.4 in wide) version of one make_pk_fstar_obs.py
band figure, for the unbound gas paper: S(k) = P_mm/P_DMO (top) and the
lensing gas fraction f_gas(theta) / (Omega_b/Omega_m) (bottom) for every
simulation (solid), with its stars rescaled so the aggregate hits the low
(dashed) and high (dotted) target of one option, the band between filled,
and the beam-compensated DESI x ACT x HSC points in the lower panel.

Same data as make_pk_fstar_obs.py (its ``analyse``); only the layout
differs: no title (the caption carries it), paper-style simulation labels
with each simulation's own aggregate stellar fraction, fonts sized for a
column. The table of make_pk_fstar_obs.py is not repeated.

Output: <fig_path>/YYYY-MM/MM-DD/<fig_name>_<option>_S_<variant>_<tag>_column.<ext>

Run from the scripts/ directory (light; login node is fine):
    python unbound_gas/make_pk_fstar_obs_column.py -p configs/unbound_gas/pk_fstar_obs_z05.yaml
    python unbound_gas/make_pk_fstar_obs_column.py -p ... --option mstar_m200m --ext png
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import compute_fstar_obs as cfo
import make_pk_fstar_obs as mfo
import make_pk_stellar as mps  # colours and axis style
from compute_stellar_maps import lensing_settings
from pk_common import load_config, select_sims

# Column-sized fonts (the imported scripts set poster-sized ones).
matplotlib.rcParams.update({
    "font.size":       8,
    "axes.labelsize":  9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.3,
})

# Simulation labels as in the paper's other figures (Fig. 4 caption names).
PAPER_LABEL = {
    ('IllustrisTNG', 'TNG300-1', None): 'TNG300-1',
    ('IllustrisTNG', 'Illustris-1', None): 'Illustris-1',
    # FLAMINGO names as utils.flamingo_label, written for mathtext (no usetex here).
    ('FLAMINGO', 'L1_m9', 'L1_m9'): 'FLAMINGO L1_m9',
    ('FLAMINGO', 'L1_m9', 'fgas-8sigma'): r'FLAMINGO fgas$-8\sigma$',
    ('FLAMINGO', 'L1_m9', 'Jet_fgas-4sigma'): r'FLAMINGO Jet_fgas$-4\sigma$',
    ('FLAMINGO', 'L1_m9', 'Mstar-1sigma'): r'FLAMINGO M$_\ast-\sigma$',
    ('FLAMINGO', 'L1_m9', 'Mstar-1sigma_fgas-4sigma'): r'FLAMINGO M$_\ast-\sigma$_fgas$-4\sigma$',
}
# Symbol of each option's aggregate in the legends.
AGG_SYMBOL = {'fstar': r'\bar{f}_\star', 'mstar_m200m': r'\bar{M}_\star/\bar{M}_{200m}'}


def paper_label(r: dict) -> str:
    """Paper-style label; falls back to the pipeline label."""
    return PAPER_LABEL.get((r['sim_type'], r['name'], r['feedback']), r['label'])


def fig_column(results: list, option: str, kmax: float, path: Path, lens_data,
               lens_xmax: float, no_band=frozenset()) -> None:
    """Two stacked panels, one column wide.

    Simulations whose (sim_type, name, feedback) is in ``no_band`` are drawn as
    the simulation alone, without the rescaled ends and their band.
    """
    lo, hi = f"{option}__low", f"{option}__high"
    have = [r for r in results if lo in r['ends']]
    if not have:
        raise SystemExit("no results to plot")
    T_lo, T_hi = have[0]['ends'][lo]['target'], have[0]['ends'][hi]['target']
    sym = AGG_SYMBOL[option]
    own = 'fstar_sim' if option == 'fstar' else 'mstar_m200m_sim'
    fmt = '.2f' if option == 'fstar' else '.3f'
    colours = [mps.sim_colour(r) for r in have]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 5.3),
                                   gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.3))
    for r, col in zip(have, colours):
        sel = r['k'] <= kmax
        k = r['k'][sel]
        el, eh = r['ends'][lo], r['ends'][hi]
        ax1.plot(k, r['S0'][sel], color=col, lw=1.3,
                 label=rf"{paper_label(r)} (${sym}={format(r['obs'][own], fmt)}$)")
        if (r['sim_type'], r['name'], r['feedback']) in no_band:
            continue
        ax1.plot(k, el['S'][sel], color=col, lw=0.9, ls='--')
        ax1.plot(k, eh['S'][sel], color=col, lw=0.9, ls=':')
        ax1.fill_between(k, el['S'][sel], eh['S'][sel], color=col, alpha=0.2, lw=0)
    ax1.axhline(1.0, color='k', lw=0.7)
    mps._style_axis(ax1)
    mps._k_range(ax1, have, kmax)
    ax1.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$', labelpad=1)
    ax1.set_ylabel(r'$S(k) = P_{\rm mm}/P_{\rm DMO}$')
    ax1.legend(loc='lower left', framealpha=0.85, handlelength=1.6, borderpad=0.4,
               labelspacing=0.3)

    for r, col in zip(have, colours):
        L, el, eh = r.get('lens'), r['ends'][lo], r['ends'][hi]
        banded = (r['sim_type'], r['name'], r['feedback']) not in no_band
        if L is None or (banded and (el.get('f') is None or eh.get('f') is None)):
            raise SystemExit(f"{r['label']}: lensing stacks missing")
        th = L['theta']
        ax2.plot(th, L['f'], color=col, lw=1.3, marker='o', ms=2.5)
        if not banded:
            continue
        ax2.plot(th, el['f'], color=col, lw=0.9, ls='--')
        ax2.plot(th, eh['f'], color=col, lw=0.9, ls=':')
        ax2.fill_between(th, el['f'], eh['f'], color=col, alpha=0.2, lw=0)
    h = [plt.Line2D([], [], color='gray', lw=1.3, marker='o', ms=2.5),
         plt.Line2D([], [], color='gray', lw=0.9, ls='--'),
         plt.Line2D([], [], color='gray', lw=0.9, ls=':')]
    lab = ['simulation', rf'${sym}={format(T_lo, fmt)}$', rf'${sym}={format(T_hi, fmt)}$']
    if lens_data is not None:
        h.append(ax2.errorbar(lens_data['theta'], lens_data['f'], yerr=lens_data['err'], fmt='s',
                              color='k', ms=3, capsize=1.5, elinewidth=0.8, zorder=5))
        lab.append(r'DESI$\times$ACT$\times$HSC')
    ax2.axhline(1.0, color='k', lw=0.7)
    ax2.grid(True, which='major', color='0.8', lw=0.6)
    ax2.set_axisbelow(True)
    ax2.set_xlim(0.0, lens_xmax)
    # headroom for a two-column legend clear of the curves and error bars
    ax2.set_ylim(top=max(ax2.get_ylim()[1], 1.3))
    ax2.set_xlabel(r'$\theta\;[\mathrm{arcmin}]$', labelpad=1)
    ax2.set_ylabel(r'$f_{\rm gas}(\theta)\,/\,(\Omega_b/\Omega_m)$')
    ax2.legend(h, lab, loc='upper left', ncol=2, framealpha=0.85, handlelength=1.6,
               borderpad=0.4, labelspacing=0.3, columnspacing=1.0)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--option', default='fstar', choices=list(cfo.OPTIONS))
    parser.add_argument('--kmax', type=float, default=5.0, help="largest k plotted [h/Mpc] (default 5)")
    parser.add_argument('--ext', default=None, help="file type (default: the config's fig_type)")
    args = parser.parse_args()

    config = load_config(args.path2config)
    for key in ('fstar_obs', 'lensing'):
        if key not in config:
            raise SystemExit(f"{args.path2config} has no '{key}' block")
    obs = cfo.obs_settings(config)
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
    else:
        print(f"no beam-compensated data at {lens['data']}; the lensing panel has no data points")

    results = [r for r in (mfo.analyse(e, config, False) for e in select_sims(config))
               if r is not None]
    # Simulations drawn without their rescaled ends (config: bands: false).
    no_band = frozenset((e['sim_type'], e['name'], e.get('feedback'))
                        for e in config['simulations'] if not e.get('bands', True))
    name = (f"{plot_cfg['fig_name']}_{args.option}_S_{obs['variant']['name']}_{obs['tag']}"
            f"_column.{ext}")
    fig_column(results, args.option, args.kmax, out_dir / name, lens_data, lens_xmax, no_band)


if __name__ == '__main__':
    main()
