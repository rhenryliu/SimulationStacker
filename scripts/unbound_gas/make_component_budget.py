"""make_component_budget.py

The unbound gas paper's component-budget figure as one grid: a row per
simulation and three columns,

    (a) 3D spherical shells: the fraction of the baryons in each shell
        contributed by each component (make_baryonFraction.py, shell mode);
    (b) 3D cumulative: M_i(<r) / M_tot(<r) / (Omega_b/Omega_m)
        (make_stackArea.py, 3D panel);
    (c) 2D Delta Sigma: Delta Sigma_i / Delta Sigma_tot / (Omega_b/Omega_m)
        (make_stackArea.py, 2D panel).

Plotting only: the profiles are read from the npz files the two production
scripts write next to their figures (``<fig_name>_3D_shell_baryonFraction_profiles.npz``
and ``<fig_name>_z<z>_stackArea_profiles.npz``), so nothing is restacked.

Run from the scripts/ directory:
    python unbound_gas/make_component_budget.py -p configs/unbound_gas/component_budget_z05.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

matplotlib.use('Agg')
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
})

# Component colours as in make_stackArea.py / make_baryonFraction.py (plasma
# 0-0.8 over the four components, in this order).
_COMPONENTS = ['ionized_gas', 'neutral_gas', 'Stars', 'BH']
_COMPONENT_LABELS = {'ionized_gas': 'Ionized gas', 'neutral_gas': 'Neutral gas',
                     'Stars': 'Stars', 'BH': 'Black holes'}

_TITLES = [
    '(a) 3D shells\nfraction of the baryons in each shell',
    '(b) 3D cumulative\n' r'$M_i(<r)/M_{\rm tot}(<r)/(\Omega_b/\Omega_m)$',
    '(c) 2D $\\Delta\\Sigma$, no beam\n' r'$\Delta\Sigma_i/\Delta\Sigma_{\rm tot}/(\Omega_b/\Omega_m)$',
]


def _colours(labels):
    """Component colours, matched by name to the production figures."""
    cols = matplotlib.colormaps['plasma'](np.linspace(0.0, 0.8, len(_COMPONENTS)))  # type: ignore
    return [cols[_COMPONENTS.index(lab)] for lab in labels]


def _check_labels(labels, where):
    unknown = [lab for lab in labels if lab not in _COMPONENTS]
    if unknown:
        raise ValueError(f"{where}: unknown components {unknown}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    args = parser.parse_args()
    with open(args.path2config) as f:
        cfg = yaml.safe_load(f)

    shells = np.load(cfg['baryonfraction_npz'], allow_pickle=False)
    area = np.load(cfg['stackarea_npz'], allow_pickle=False)
    rows = cfg['simulations']  # [{key: 'suite/name[/feedback]', label: ...}]
    for r in rows:
        for arr, kind in ((shells, '3D_fractions'), (area, '3D_fractions'), (area, '2D_fractions')):
            if f"{r['key']}/{kind}" not in arr.files:
                raise KeyError(f"{r['key']}/{kind} missing from the npz files")

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(cfg.get('width', 7.1), cfg.get('row_height', 1.05) * n + 0.9),
                             sharex='col', sharey=True, squeeze=False)
    handles = {}
    for i, r in enumerate(rows):
        key = r['key']
        # (a) shells: stacked bars, one per shell
        labs = [str(x) for x in shells[f'{key}/3D_labels']]
        _check_labels(labs, f'{key} shells')
        edges = shells[f'{key}/3D_edges']
        frac = shells[f'{key}/3D_fractions']
        left, widths = edges[:-1], np.diff(edges)
        bottom = np.zeros_like(left)
        for lab, fr, col in zip(labs, frac, _colours(labs)):
            h = axes[i, 0].bar(left, fr, width=widths, bottom=bottom, align='edge', color=col,
                               alpha=0.8, linewidth=0.0)
            handles.setdefault(lab, h)
            bottom = bottom + fr
        # (b) 3D cumulative and (c) 2D Delta Sigma: stacked areas
        for c, pre in ((1, '3D'), (2, '2D')):
            labs = [str(x) for x in area[f'{key}/{pre}_labels']]
            _check_labels(labs, f'{key} {pre}')
            axes[i, c].stackplot(area[f'{key}/{pre}_x'], area[f'{key}/{pre}_fractions'],
                                 colors=_colours(labs), alpha=0.8)
            axes[i, c].axhline(1.0, color='k', ls='--', lw=1)
        axes[i, 0].text(0.03, 0.06, r['label'], transform=axes[i, 0].transAxes, fontsize=8.5,
                        va='bottom', ha='left',
                        bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='gray',
                                  alpha=0.85))
        for ax in axes[i]:
            ax.grid(True, lw=0.4)
            ax.set_ylim(0.0, cfg.get('ymax', 1.08))
    for c, title in enumerate(_TITLES):
        axes[0, c].set_title(title, fontsize=8.5)
    axes[-1, 0].set_xlabel(r'$r$ [comoving kpc/$h$]')
    axes[-1, 1].set_xlabel(r'$r$ [comoving kpc/$h$]')
    axes[-1, 2].set_xlabel(r'$\theta$ [arcmin]')
    axes[0, 0].set_xlim(0.0, float(cfg.get('rmax_ckpch', 4000.0)))
    axes[0, 1].set_xlim(0.0, float(cfg.get('rmax_ckpch', 4000.0)))
    axes[0, 2].set_xlim(0.0, float(cfg.get('thetamax_arcmin', 10.5)))
    order = [c for c in _COMPONENTS if c in handles]
    fig.legend([handles[c] for c in order], [_COMPONENT_LABELS[c] for c in order],
               loc='lower center', ncol=len(order), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.25, w_pad=0.6)

    now = datetime.now()
    out = Path(cfg.get('fig_path', '../figures/')) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg.get('fig_name', 'component_budget')}.{cfg.get('fig_type', 'pdf')}"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == '__main__':
    sys.exit(main())
