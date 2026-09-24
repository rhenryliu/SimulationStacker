"""plot_pk_suppression.py

Presentation plot of the baryonic suppression of the matter power spectrum,

    S(k) = P_mm(k) / P_DMO(k),

for every simulation in the config, each referenced to the dark-matter-only run
with the same initial conditions.

Nothing is computed here: the spectra are read from the ``*_Pk_dmo_<n>.npz``
files that unbound_gas/compute_pk_components.py writes to
``<root>/<SimType>/products/3D/``. ``P_total`` is the hydro run's total-matter
auto spectrum and ``P_dmo`` the DMO run's, on the same grid and k bins, so
S(k) is the same quantity as S(alpha_0) in unbound_gas/make_pk_alpha.py.

Simulations are grouped by suite in the same order as
configs/lensing/mass_ratio_noBeam_z05.yaml, so colours and labels match the
lensing paper figures.

Usage
-----
    python lensing/plot_pk_suppression.py -p configs/lensing/pk_suppression_z05.yaml
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml

sys.path.append('../src/')
from loadIO import resolve_data_root  # type: ignore

# ---------------------------------------------------------------------------
# Matplotlib style — matches beam_compensated_ratio_v2.py
# ---------------------------------------------------------------------------
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
    "legend.fontsize": 14,
})

# Fixed colours for the FLAMINGO feedback variants, keyed by feedback name.
# Keep in sync with beam_compensated_ratio_v2.py / compare_data_ratio.py.
_FLAMINGO_COLOURS = {
    'L1_m9':           '#B30000',  # dark red (fiducial)
    'fgas-8sigma':     '#FF7F0E',  # orange
    'Jet_fgas-4sigma': '#C71585',  # magenta
}

# k values [h/Mpc] at which S(k) is printed to stdout.
_K_PRINT = [1.0, 2.0, 5.0, 10.0]


def sim_entries(config: dict) -> list:
    """Flatten the config's simulation groups into labelled, coloured entries.

    Colours and labels follow the lensing figures (beam_compensated_ratio_v2.py):
    group i takes colourmap ['plasma', 'twilight', 'hot'][i], sampled on
    linspace(0.2, 0.85, n_sims), and FLAMINGO variants use fixed colours.

    Args:
        config: Parsed YAML config.

    Returns:
        List of dicts with keys ``sim_type``, ``sim``, ``label`` and ``colour``.
    """
    entries = []
    for i, sim_group in enumerate(config['simulations']):
        cmap    = matplotlib.colormaps[['plasma', 'twilight', 'hot'][i]]  # type: ignore[attr-defined]
        n_sims  = len(sim_group['sims'])
        colours = cmap(np.linspace(0.2, 0.85, n_sims))
        for j, sim in enumerate(sim_group['sims']):
            if sim_group['sim_type'] == 'IllustrisTNG':
                label  = sim['name']
                colour = colours[j]
            elif sim_group['sim_type'] == 'FLAMINGO':
                label  = f"FLAMINGO {sim['feedback']}".replace('_', '-')
                colour = _FLAMINGO_COLOURS.get(sim['feedback'], colours[j])
            else:
                label  = "SIMBA-100"
                colour = colours[j]
            entries.append(dict(sim_type=sim_group['sim_type'], sim=sim,
                                label=label, colour=colour))
    return entries


def dmo_spectra_path(sim_type: str, sim: dict) -> Path:
    """Path of the hydro/DMO spectra file written by compute_pk_components.py."""
    if sim.get('feedback'):
        stem = f"{sim['name']}_{sim['feedback']}_{sim['snapshot']}"
    else:
        stem = f"{sim['name']}_{sim['snapshot']}"
    return (Path(resolve_data_root(None)) / sim_type / 'products' / '3D'
            / f"{stem}_Pk_dmo_{sim['n_pixels']}.npz")


def load_suppression(sim_type: str, sim: dict) -> tuple:
    """Load k [h/Mpc] and S(k) = P_total / P_dmo for one simulation.

    Raises:
        FileNotFoundError: If the spectra file does not exist.
    """
    path = dmo_spectra_path(sim_type, sim)
    if not path.exists():
        raise FileNotFoundError(f"missing spectra {path} (run unbound_gas/compute_pk_components.py)")
    d = np.load(path)
    return d['k'], d['P_total'] / d['P_dmo']


def main(path2config: str) -> None:
    """Plot S(k) for every simulation in the config and save the figure."""
    with open(path2config) as f:
        config = yaml.safe_load(f)
    plot_config = config['plot']
    kmax = config['pk'].get('kmax', 10.0)

    # ---- Output path: figures/<year-month>/<month-day>/ ----
    now      = datetime.now()
    fig_path = (
        Path(plot_config.get('fig_path', '../figures/'))
        / now.strftime("%Y-%m")
        / now.strftime("%m-%d")
    )
    fig_path.mkdir(parents=True, exist_ok=True)
    fig_name  = plot_config.get('fig_name', 'pk_suppression')
    fig_types = plot_config.get('fig_type', 'pdf')
    if isinstance(fig_types, str):
        fig_types = [fig_types]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    print(f"{'simulation':26s} " + ' '.join(f"S(k={kp:g})" for kp in _K_PRINT))
    for e in sim_entries(config):
        k, S = load_suppression(e['sim_type'], e['sim'])
        sel = k <= kmax
        ax.plot(k[sel], S[sel], color=e['colour'], lw=2.5, label=e['label'])
        # Nearest k bin; '--' beyond the grid's Nyquist (as make_pk_alpha.py's table).
        vals = [f"{S[np.argmin(np.abs(np.log(k / kp)))]:9.3f}" if kp <= k.max()
                else f"{'--':>9s}" for kp in _K_PRINT]
        print(f"{e['label']:26s} " + ' '.join(vals))

    ax.axhline(1.0, color='k', lw=1, ls='--')
    ax.set_xscale('log')
    ax.grid(True, which='major', color='0.8', lw=0.8)
    ax.grid(True, which='minor', axis='x', color='0.9', lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$S(k) = P_{\rm hydro}(k)/P_{\rm DMO}(k)$')
    ax.text(0.97, 0.95, r'$z \simeq 0.5$', transform=ax.transAxes, ha='right', va='top')
    ax.legend(loc='lower left', framealpha=0.85)
    fig.tight_layout()

    for ext in fig_types:
        out_path = fig_path / f'{fig_name}.{ext}'
        fig.savefig(out_path, dpi=plot_config.get('dpi', 150))  # type: ignore
        print(f"saved {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the baryonic P(k) suppression S(k) of each simulation.')
    parser.add_argument('-p', '--path2config', required=True, help='path to the YAML config')
    args = parser.parse_args()
    main(args.path2config)
