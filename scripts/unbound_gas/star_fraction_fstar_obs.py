"""star_fraction_fstar_obs.py
==========================
Stellar-fraction bar chart (one-off, for slides) with the aggregate stellar
fraction of the P(k) section's observation-based bands, for the simulations of
a pk_fstar_obs config. Two bars per simulation, in the style of
star_fraction_v2.py:

    1. Global   f* = M* / (M* + M_wind + M_gas + M_BH) over the whole box.
    2. Haloes   f̄* = the same ratio summed over the 1 R200m apertures (mass
                priority) of the haloes with FoF mass at or above the cut,
                i.e. the f̄* quoted in the legend of make_pk_fstar_obs_grid.py.

Winds (TNG/Illustris wind-phase particles) count as gas. Plotting only: both
values are read from existing files, nothing is recomputed:
  products/3D/<stem>_Pk_fstar_obs_<variant>_<n>.npz   f̄* ('fstar_sim'), compute_fstar_obs.py
  products/3D/<stem>_Pk_fstar_<variant>_<n>.npz       box totals, compute_fstar.py

Output: <fig_path>/YYYY-MM/MM-DD/star_fraction_fstar_obs_z<z>.png and a yaml
of the values in star_fraction_v2.py's format ({label: {global, r200m}}).

Run from the scripts/ directory (light; login node is fine):
    python unbound_gas/star_fraction_fstar_obs.py -p configs/unbound_gas/pk_fstar_obs_z05_n5e-4.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

import compute_fstar_obs as cfo
from pk_common import load_config, sim_label, spectra_path

sys.path.append('../src/')
from utils import flamingo_label  # noqa: E402

# Same fonts as star_fraction_v2.py (set after the imports, which may set their own).
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": True,
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})


def bar_label(entry: dict) -> str:
    """Tick label of a simulation (usetex-safe), as in star_fraction_v2.py."""
    if entry['sim_type'] == 'FLAMINGO':
        return flamingo_label(entry['feedback'])
    return entry['name']


def fractions_for_sim(entry: dict, obs: dict) -> dict:
    """Global and halo stellar fractions of one simulation from its files.

    Args:
        entry (dict): Simulation entry of the pk_fstar_obs config.
        obs (dict): ``compute_fstar_obs.obs_settings`` of that config.

    Returns:
        dict: 'global', 'halo' (f̄*), 'n_haloes', 'z'.

    Raises:
        FileNotFoundError: If either file is missing.
        KeyError: If the f* file has no entry for the config's mass cut.
        ValueError: If the f* obs file was made with another region definition
            or mass cut, or the f* file was regenerated after it (their f̄*
            differ).
    """
    name, tag = obs['variant']['name'], obs['tag']
    p_obs = spectra_path(entry, f"fstar_obs_{name}")
    p_fst = spectra_path(entry, f"fstar_{name}")
    for p in (p_obs, p_fst):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing (compute_fstar.py / compute_fstar_obs.py first)")
    with np.load(p_obs) as f:
        if (str(f['variant']) != name or str(f['tag']) != tag
                or not np.isclose(float(f['halo_mass_min']), obs['mass_min'], rtol=1e-12, atol=0.0)):
            raise ValueError(f"{p_obs}: made for {f['variant']}, {f['tag']}, cut "
                             f"{float(f['halo_mass_min']):.3e}; config wants {name}, {tag}, "
                             f"{obs['mass_min']:.3e}")
        halo = float(f['fstar_sim'])
        n_haloes = int(f['n_haloes'])
        z = float(f['z'])
    with np.load(p_fst) as f:
        halo_fst = float(f[f"diag__{tag}__fstar_sim"])
        mstar, mwind = float(f['mstar_box']), float(f['mwind_box'])
        mgas, mbh = float(f['mgas_box']), float(f['mbh_box'])
    if not np.isclose(halo, halo_fst, rtol=1e-10, atol=0.0):
        raise ValueError(f"{sim_label(entry)}: f̄* {halo:.6f} ({p_obs.name}) vs "
                         f"{halo_fst:.6f} ({p_fst.name})")
    return {'global': mstar / (mstar + mwind + mgas + mbh), 'halo': halo,
            'n_haloes': n_haloes, 'z': z}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True,
                        help='pk_fstar_obs config (its simulations and fstar_obs block).')
    args = parser.parse_args()

    config = load_config(args.path2config)
    obs = cfo.obs_settings(config)
    x, mcut = obs['variant']['x'], obs['mass_min']

    labels, vals = [], []
    for entry in config['simulations']:
        labels.append(bar_label(entry))
        vals.append(fractions_for_sim(entry, obs))
    zs = {round(v['z'], 3) for v in vals}
    if len(zs) != 1:
        raise ValueError(f"the simulations' files are at different redshifts: {sorted(zs)}")
    z = zs.pop()

    print(f"{'simulation':40s} {'f_halo':>8s} {'f_global':>9s} {'n_haloes':>9s}")
    for entry, v in zip(config['simulations'], vals):
        print(f"{sim_label(entry):40s} {v['halo']:8.4f} {v['global']:9.4f} {v['n_haloes']:9,d}")

    # ---- plot: grouped bars (global + haloes) per simulation
    xs = np.arange(len(labels))
    width = 0.4
    aper = r'R_{200\mathrm{m}}' if x == 1.0 else rf'{x:g}\,R_{{200\mathrm{{m}}}}'
    halo_lab = (rf'$\bar{{f}}_\star$ (haloes $\geq 10^{{{np.log10(mcut):g}}}\,M_\odot/h$, '
                rf'$<{aper}$)')

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.tick_params(axis='both', which='both', labelsize=14)
    ax.bar(xs - width / 2, [v['global'] for v in vals], width=width, color='skyblue',
           edgecolor='none', alpha=0.8, label=r'$f_{\star}^{\mathrm{global}}$')
    ax.bar(xs + width / 2, [v['halo'] for v in vals], width=width, color='salmon',
           edgecolor='none', alpha=0.8, label=halo_lab)
    ax.set_xlabel('Simulation Suites', fontsize=18)
    ax.set_ylabel('Stellar Fraction (compared to total baryons)', fontsize=18)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, ha='right')
    ax.grid(True)
    ax.set_title(f'Baryon Star Fraction, z = {z:g}', fontsize=18)
    ax.legend(loc='best', fontsize=18)

    plot = config.get('plot', {})
    now = datetime.now()
    out_dir = Path(plot.get('fig_path', '../figures/')) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"star_fraction_fstar_obs_z{z:g}"

    # values in star_fraction_v2.py's format (its 'r200m' key = the halo bar), in plot order
    with open(out_dir / f"{stem}.yaml", 'w', encoding='utf-8') as f:
        f.write(f"# star_fraction_fstar_obs.py -p {args.path2config}: global f* and the f̄* of "
                f"haloes >= {mcut:.1e} Msun/h within {x:g} R200m\n")
        yaml.safe_dump({lab: {'global': float(v['global']), 'r200m': float(v['halo'])}
                        for lab, v in zip(labels, vals)}, f, sort_keys=False, allow_unicode=True)
    print(f"saved {out_dir / f'{stem}.yaml'}")

    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"saved {out_dir / f'{stem}.png'}")


if __name__ == '__main__':
    main()
