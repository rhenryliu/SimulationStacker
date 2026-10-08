"""make_kSZ_outside_fraction.py

Figures (exploratory at z = 0.5; the z ~ 0.26 grid is in the unbound gas
paper's appendix) re-expressing the kSZ masking figure
(simulated_kSZ_masked.py) as the fraction of the CAP-filtered signal that comes
from gas outside the retained spheres,

    f_out(theta; n) = 1 - S_n(theta) / S_unmasked(theta),

where S_n is the stacked profile of the map that keeps only the gas within
n x R200m of the stacked galaxies' hosts. The profiles are read from the npz
that simulated_kSZ_masked.py writes next to its figure; nothing is restacked
and the production script is not touched.

Two figures are written to ``<fig_path>/<YYYY-MM>/<MM-DD>/``:

- ``<fig_name>_grid.<ext>``: the masking figure's grid, one row per entry of
  ``rows`` in the config. Columns 1-3 show f_out for n = 1, 2, 3; column 4
  shows the unmasked profiles with the data, as in the masking figure.
- ``<fig_name>_summary.<ext>``: one panel per aperture in ``summary_theta``,
  f_out against n for every simulation.

No error bands: the npz holds the mean and standard error of each profile, but
S_n and S_unmasked are stacked on the same galaxies and are correlated, so the
error of their ratio cannot be derived from them. f_out can be negative at
small apertures: removing gas outside the spheres lowers the CAP ring more
than the disk.

Run from the scripts/ directory:
    python unbound_gas/make_kSZ_outside_fraction.py -p configs/unbound_gas/kSZ_outside_fraction_z05.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.append('../src/')
from utils import flamingo_label  # type: ignore

matplotlib.use('Agg')
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
})

# Colours as in simulated_kSZ_masked.py: fixed FLAMINGO colours, SIMBA sampled
# from a colour map over the sims of its row (except SIMBA-100), and the
# IllustrisTNG runs fixed to their three-run-row colours.
_FLAMINGO_COLOURS = {
    'L1_m9':           '#B30000',  # dark red (fiducial)
    'fgas-8sigma':     '#FF7F0E',  # orange
    'Jet_fgas-4sigma': '#C71585',  # magenta
    'Mstar-1sigma':             '#17BECF',  # cyan
    'Mstar-1sigma_fgas-4sigma': '#2E8B57',  # sea green
}
_SUITE_CMAPS = {'SIMBA': 'hsv', 'IllustrisTNG': 'twilight', 'FLAMINGO': 'plasma'}
_SIMBA100_COLOUR = matplotlib.colormaps['hsv'](0.85)  # type: ignore
# IllustrisTNG runs: the colours of the three-run row, whatever the row holds.
_TNG_COLOURS = dict(zip(['TNG100-1', 'TNG300-1', 'Illustris-1'],
                        matplotlib.colormaps['twilight'](np.linspace(0.2, 0.85, 3))))  # type: ignore
_SIMBA_NAMES = {'m100n1024/s50': 'SIMBA-100', 'm50n512/s50noagn': 'SIMBA-50 no-AGN',
                'm50n512/s50nox': 'SIMBA-50 no-X-ray', 'm50n512/s50nofb': 'SIMBA-50 no-feedback',
                'm50n512/s50nojet': 'SIMBA-50 no-jet', 'm50n512/s50': 'SIMBA-50'}


def sim_label(key: str) -> str:
    """Legend label of an npz key 'suite/name[/feedback]'."""
    suite, rest = key.split('/', 1)
    if suite == 'FLAMINGO':
        return flamingo_label(rest.split('/')[-1], prefix=False)
    if suite == 'SIMBA':
        return _SIMBA_NAMES.get(rest, rest)
    return rest


def row_colours(row: dict) -> list:
    """Colours of the simulations of one row (which may mix suites).

    Each run takes its suite's colour map, sampled over that suite's runs in
    the row; FLAMINGO runs, SIMBA-100 and the IllustrisTNG runs have fixed
    colours instead (SIMBA-100 the magenta it has as the last of the four
    SIMBA runs, the IllustrisTNG runs those of the three-run row).
    """
    keys = row['sims']
    suites = [k.split('/')[0] for k in keys]
    colours = []
    for key, suite in zip(keys, suites):
        same = [k for k, s in zip(keys, suites) if s == suite]
        cmap = matplotlib.colormaps[_SUITE_CMAPS[suite]]  # type: ignore
        colour = cmap(np.linspace(0.2, 0.85, len(same)))[same.index(key)]
        if suite == 'FLAMINGO':
            colour = _FLAMINGO_COLOURS.get(key.split('/')[-1], colour)
        elif key == 'SIMBA/m100n1024/s50':
            colour = _SIMBA100_COLOUR
        elif suite == 'IllustrisTNG':
            colour = _TNG_COLOURS.get(key.split('/')[-1], colour)
        colours.append(colour)
    return colours


def outside_fraction(prof, key: str, n: int) -> np.ndarray:
    """1 - S_n / S_unmasked for one simulation."""
    return 1.0 - prof[f'{key}/mask{n}_mean'] / prof[f'{key}/unmasked_mean']


def fig_grid(prof, rows: list, data, cfg: dict, path: Path) -> None:
    """Rows of the masking figure: f_out for n = 1-3, then the unmasked profiles."""
    theta = prof['radii_arcmin']
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, cfg.get('row_height', 3.6) * n_rows),
                             sharex=True)
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        cols = row_colours(row)
        for key, col in zip(row['sims'], cols):
            for n in (1, 2, 3):
                axes[r, n - 1].plot(theta, outside_fraction(prof, key, n), color=col, lw=2,
                                    marker='o', ms=4, label=sim_label(key))
            axes[r, 3].plot(theta, prof[f'{key}/unmasked_mean'], color=col, lw=2, marker='o',
                            ms=4, label=sim_label(key))
        if data is not None:
            axes[r, 3].errorbar(data['theta'], data['signal'], yerr=data['err'], fmt='s',
                                color='k', ms=5, zorder=10, label=cfg.get('data_label', 'data'))
        for c in range(3):
            ax = axes[r, c]
            ax.axhline(0.0, color='k', lw=0.8)
            ax.set_ylim(cfg.get('fraction_ylim', [-0.2, 1.0]))
            ax.grid(True)
            if c > 0:
                ax.tick_params(labelleft=False)
        axes[r, 0].set_ylabel(r'$1 - S_n/S_{\rm unmasked}$')
        axes[r, 3].set_yscale('log')
        axes[r, 3].grid(True)
        axes[r, 3].legend(loc='lower right', fontsize=11)
        axes[r, 3].yaxis.set_label_position('right')
        axes[r, 3].yaxis.tick_right()
        axes[r, 3].set_ylabel(r'$T_{\rm kSZ}$ [$\mu$K arcmin$^2$]')
        axes[r, 0].text(-0.32, 0.5, row['title'], transform=axes[r, 0].transAxes, rotation=90,
                        va='center', ha='center', fontsize=18)
    for c in range(3):
        axes[0, c].set_title(rf'gas outside ${c + 1}\,R_{{200m}}$')
    axes[0, 3].set_title('No masking')
    for ax in axes[-1]:
        ax.set_xlabel(r'$\theta$ [arcmin]')
        ax.set_xlim(0.0, 6.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def fig_summary(prof, rows: list, cfg: dict, path: Path) -> None:
    """f_out against n at a few apertures, every simulation in one panel per aperture."""
    theta = prof['radii_arcmin']
    apertures = cfg.get('summary_theta', [2.0, 4.0, 6.0])
    fig, axes = plt.subplots(1, len(apertures), figsize=(5.2 * len(apertures), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    seen = set()
    for row in rows:
        for key, col in zip(row['sims'], row_colours(row)):
            if key in seen:  # a simulation listed in two rows is drawn once
                continue
            seen.add(key)
            for ax, th in zip(axes, apertures):
                f = [np.interp(th, theta, outside_fraction(prof, key, n)) for n in (1, 2, 3)]
                ax.plot([1, 2, 3], f, color=col, lw=1.8, marker='o', ms=5, label=sim_label(key))
    for ax, th in zip(axes, apertures):
        ax.set_title(rf'$\theta = {th:g}$ arcmin')
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel(r'retained radius $n$ [$R_{200m}$]')
        ax.axhline(0.0, color='k', lw=0.8)
        ax.grid(True)
    axes[0].set_ylabel(r'$1 - S_n/S_{\rm unmasked}$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(5, len(labels)), fontsize=12,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def print_table(prof, rows: list, apertures) -> None:
    """f_out at the summary apertures, for the log."""
    theta = prof['radii_arcmin']
    print(f"{'simulation':42s} " + ' '.join(f"n={n},{th:g}'" for n in (1, 2, 3) for th in apertures))
    seen = set()
    for row in rows:
        for key in row['sims']:
            if key in seen:
                continue
            seen.add(key)
            vals = [np.interp(th, theta, outside_fraction(prof, key, n))
                    for n in (1, 2, 3) for th in apertures]
            print(f"{key:42s} " + ' '.join(f"{v:7.3f}" for v in vals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    args = parser.parse_args()
    with open(args.path2config) as f:
        cfg = yaml.safe_load(f)
    prof = np.load(cfg['profiles_npz'])
    rows = cfg['rows']
    missing = [k for row in rows for k in row['sims'] if f'{k}/unmasked_mean' not in prof.files]
    if missing:
        raise KeyError(f"not in {cfg['profiles_npz']}: {missing}")
    data = None
    if cfg.get('data_path'):
        with np.load(cfg['data_path']) as d:
            data = dict(theta=d['theta_arcmins'], signal=d['signal'], err=d['noise'])
    now = datetime.now()
    out = Path(cfg.get('fig_path', '../figures/')) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    out.mkdir(parents=True, exist_ok=True)
    ext = cfg.get('fig_type', 'pdf')
    name = cfg.get('fig_name', 'kSZ_outside_fraction')
    fig_grid(prof, rows, data, cfg, out / f'{name}_grid.{ext}')
    fig_summary(prof, rows, cfg, out / f'{name}_summary.{ext}')
    print_table(prof, rows, cfg.get('summary_theta', [2.0, 4.0, 6.0]))


if __name__ == '__main__':
    main()
