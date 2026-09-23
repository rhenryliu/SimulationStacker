"""make_pk_alpha.py
=================
Figures and tables for the matter power spectrum section of the unbound gas
paper: how much the non-ionized baryons (neutral gas, stars, black holes)
matter for the baryonic suppression of P(k).

Reads the spectra written by ``compute_pk_components.py`` and evaluates the
matter power spectrum for mass-conserving redistributions of the baryons.
With component weights c_i (mass fractions of the total matter; c_i = w_i for
the actual simulation) and component spectra P_ij,

    P_mm(c) = sum_ij c_i c_j P_ij .

The alpha model of the section lays all baryon mass out on two templates,

    rho_b(alpha) = rhobar_b [ alpha u_ion + (1 - alpha) u_else ],

with u_x = rho_x / rhobar_x and u_else the mass-weighted mix of neutral gas,
stars and BH: c_ion = alpha f_b and c_e = (1 - alpha) f_b w_e / w_else. At
alpha_0 = w_ion / f_b (the global ionized fraction of the baryons) this is
exactly the simulation; at alpha = 1 every baryon is distributed like the
ionized gas. The DM field is held fixed.

Quantities:
  Q(k)  = P_mm(alpha=1) / P_mm(alpha_0) - 1   -- reference-free headline
  S(k)  = P_mm / P_ref, with P_ref the DMO run when its spectra exist, else
          the DM of the hydro run (stand-in, flagged in legends and tables)
  dS(k) = S(1) - S(alpha_0) = [P_mm(1) - P_mm(alpha_0)] / P_ref  -- absolute
          change in the suppression (preferred over a fraction of 1 - S,
          which diverges on large scales where the suppression vanishes)

Outputs in <fig_path>/YYYY-MM/MM-DD/:
  <fig_name>_alpha.<ext>        S(k) bands (alpha_0 -> 1) and Q(k), with the
                                per-component split
  <fig_name>_targets.<ext>      Q(k) when the non-ionized mass is laid out
                                like the ionized gas, like the DM, or
                                uniformly (one panel per simulation)
  <fig_name>_dmo_check.<ext>    S(k) with the DMO reference vs the hydro-DM
                                stand-in, and the total x DMO correlation r(k)
  <fig_name>_table.txt          budget fractions and the quantities above at
                                the k values in pk.k_table

Run from the scripts/ directory (light; login node is fine):
    python unbound_gas/make_pk_alpha.py -p configs/unbound_gas/pk_components_z05.yaml
    python unbound_gas/make_pk_alpha.py -p configs/unbound_gas/pk_components_z05.yaml \
        --sims 'L1_m9 (L1_m9)' 'L1_m9 (fgas-8sigma)' 'L1_m9 (Jet_fgas-4sigma)'
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pk_common import COMPONENTS, load_config, select_sims, sim_label, spectra_path

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex":      True,
    "mathtext.fontset": "cm",
    "font.size":        16,
    "axes.titlesize":   16,
    "axes.labelsize":   18,
    "xtick.labelsize":  15,
    "ytick.labelsize":  15,
    "legend.fontsize":  11,
})

# One colourmap per suite, as in the other unbound-gas figures (FLAMINGO has no
# established one; viridis follows make_pk_suppression.py's fallback).
_COLOURMAPS = {'IllustrisTNG': 'twilight', 'SIMBA': 'hsv', 'FLAMINGO': 'viridis'}

I_DM, I_ION, I_NEU, I_ST, I_BH = (COMPONENTS.index(c) for c in
                                  ['DM', 'ionized_gas', 'neutral_gas', 'Stars', 'BH'])
ELSE = [I_NEU, I_ST, I_BH]


# ---------------------------------------------------------------------------
# Model algebra
# ---------------------------------------------------------------------------

def p_of(c: np.ndarray, P: np.ndarray) -> np.ndarray:
    """P_mm for component weights c: sum_ij c_i c_j P_ij (per k)."""
    return np.einsum('i,j,ijk->k', c, c, P)


def weights_alpha(w: np.ndarray, alpha: float) -> np.ndarray:
    """Component weights of the alpha model (see module docstring)."""
    f_b = w[1:].sum()
    w_else = w[ELSE].sum()
    if w_else <= 0:
        raise ValueError("no non-ionized baryon mass: the alpha model is undefined")
    c = np.zeros_like(w)
    c[I_DM] = w[I_DM]
    c[I_ION] = alpha * f_b
    c[ELSE] = (1.0 - alpha) * f_b * w[ELSE] / w_else
    return c


def weights_move(w: np.ndarray, moved: list, target: str) -> np.ndarray:
    """Weights with the mass of the ``moved`` components redistributed.

    Args:
        w: Actual component weights.
        moved: Indices of the components whose mass is moved.
        target: 'ionized' (laid out like the ionized gas), 'dm' (like the DM)
            or 'uniform' (spread uniformly: contributes no fluctuations).
    """
    c = w.copy()
    m = c[moved].sum()
    c[moved] = 0.0
    if target == 'ionized':
        c[I_ION] += m
    elif target == 'dm':
        c[I_DM] += m
    elif target != 'uniform':
        raise ValueError(target)
    return c


def analyse(entry: dict) -> dict:
    """Load one simulation's spectra and evaluate every model quantity."""
    comp = np.load(spectra_path(entry, 'components'))
    k, P, means = comp['k'], comp['P'], comp['means']
    w = means / means.sum()
    f_b = w[1:].sum()
    res = dict(label=sim_label(entry), sim_type=entry['sim_type'], k=k,
               w=w, alpha0=w[I_ION] / f_b,
               frac=dict(ionized=w[I_ION] / f_b, neutral=w[I_NEU] / f_b,
                         stars=w[I_ST] / f_b, BH=w[I_BH] / f_b))

    P0 = p_of(w, P)
    res['P0'] = P0
    res['P_alpha1'] = p_of(weights_alpha(w, 1.0), P)
    res['Q'] = res['P_alpha1'] / P0 - 1.0
    res['Q_comp'] = {name: p_of(weights_move(w, [i], 'ionized'), P) / P0 - 1.0
                     for name, i in [('neutral', I_NEU), ('stars', I_ST), ('BH', I_BH)]}
    res['Q_target'] = {t: p_of(weights_move(w, ELSE, t), P) / P0 - 1.0
                       for t in ('ionized', 'dm', 'uniform')}

    # Reference: DMO when available (same k bins), else the hydro DM auto spectrum.
    res['P_hydroDM'] = P[I_DM, I_DM]
    dmo_path = spectra_path(entry, 'dmo')
    if dmo_path.exists():
        dmo = np.load(dmo_path)
        if not np.allclose(dmo['k'], k):
            raise ValueError(f"k bins differ between {dmo_path} and the components file")
        res['P_ref'], res['ref_is_dmo'] = dmo['P_dmo'], True
        res['r_dmo'] = dmo['P_total_dmo'] / np.sqrt(dmo['P_total'] * dmo['P_dmo'])
    else:
        res['P_ref'], res['ref_is_dmo'] = res['P_hydroDM'], False
    res['S0'] = P0 / res['P_ref']
    res['S1'] = res['P_alpha1'] / res['P_ref']
    res['dS'] = res['S1'] - res['S0']
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def colours_for(results: list) -> list:
    """One colour per simulation, from its suite's colourmap."""
    out = [None] * len(results)
    for suite, cmap_name in _COLOURMAPS.items():
        idx = [i for i, r in enumerate(results) if r['sim_type'] == suite]
        cols = matplotlib.colormaps[cmap_name](np.linspace(0.2, 0.85, max(len(idx), 1)))  # type: ignore
        for i, c in zip(idx, cols):
            out[i] = c
    return out


def ref_tag(r: dict) -> str:
    """Legend suffix flagging the hydro-DM stand-in reference."""
    return '' if r['ref_is_dmo'] else r' $^\dagger$'


def fig_alpha(results: list, colours: list, kmax: float, path: Path) -> None:
    """S(k) bands between alpha_0 and 1 (top) and Q(k) with components (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.05))
    for r, col in zip(results, colours):
        sel = r['k'] <= kmax
        k = r['k'][sel]
        ax1.plot(k, r['S0'][sel], color=col, lw=2,
                 label=rf"{r['label']}{ref_tag(r)}, $\alpha_0={r['alpha0']:.3f}$")
        ax1.plot(k, r['S1'][sel], color=col, lw=1.5, ls='--')
        ax1.fill_between(k, r['S0'][sel], r['S1'][sel], color=col, alpha=0.2, lw=0)
        ax2.plot(k, 100 * r['Q'][sel], color=col, lw=2)
        ax2.plot(k, 100 * r['Q_comp']['stars'][sel], color=col, lw=1, ls='--')
        ax2.plot(k, 100 * r['Q_comp']['neutral'][sel], color=col, lw=1, ls=':')
    ax1.axhline(1.0, color='k', lw=1)
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$S(k) = P_{\rm mm}/P_{\rm DMO}$')
    ax1.legend(loc='lower left', framealpha=0.85)
    ax1.text(0.98, 0.95, r'solid: simulation ($\alpha_0$); dashed: all baryons as ionized gas ($\alpha=1$)',
             transform=ax1.transAxes, ha='right', va='top', fontsize=10)
    ax2.axhline(0.0, color='k', lw=1)
    ax2.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    ax2.set_ylabel(r'$P_{\rm mm}(\alpha{=}1)/P_{\rm mm}(\alpha_0) - 1\;[\%]$')
    ax2.plot([], [], color='gray', lw=2, label='all non-ionized')
    ax2.plot([], [], color='gray', lw=1, ls='--', label='stars only')
    ax2.plot([], [], color='gray', lw=1, ls=':', label='neutral gas only')
    ax2.legend(loc='lower left', framealpha=0.85)
    if any(not r['ref_is_dmo'] for r in results):
        ax1.text(0.02, 0.02 + 0.06 * (len(results) + 1), r'$^\dagger$ reference: DM of the hydro run',
                 transform=ax1.transAxes, fontsize=10)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


def fig_targets(results: list, colours: list, kmax: float, path: Path) -> None:
    """Q(k) for the three layouts of the non-ionized mass, one panel per sim."""
    n = len(results)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), sharex=True,
                             sharey=True, squeeze=False)
    styles = {'ionized': ('-', 'like ionized gas'), 'dm': ('--', 'like DM'),
              'uniform': (':', 'uniform')}
    for ax, r, col in zip(axes.flat, results, colours):
        sel = r['k'] <= kmax
        for t, (ls, lab) in styles.items():
            ax.plot(r['k'][sel], 100 * r['Q_target'][t][sel], color=col, ls=ls, lw=2, label=lab)
        ax.axhline(0.0, color='k', lw=1)
        ax.set_xscale('log')
        ax.set_title(r['label'], fontsize=14)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\Delta P_{\rm mm}/P_{\rm mm}\;[\%]$')
    axes.flat[0].legend(title='non-ionized baryons laid out', fontsize=10, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


def fig_dmo_check(results: list, colours: list, kmax: float, path: Path) -> None:
    """S(k) with the DMO reference vs the hydro-DM stand-in, and r(k)."""
    have = [(r, c) for r, c in zip(results, colours) if r['ref_is_dmo']]
    if not have:
        print("no DMO spectra yet; skipping the DMO check figure")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.5, 1], hspace=0.05))
    for r, col in have:
        sel = r['k'] <= kmax
        k = r['k'][sel]
        ax1.plot(k, r['S0'][sel], color=col, lw=2, label=rf"{r['label']}: $P_{{\rm mm}}/P_{{\rm DMO}}$")
        ax1.plot(k, (r['P0'] / r['P_hydroDM'])[sel], color=col, lw=1.5, ls='--',
                 label=rf"{r['label']}: $P_{{\rm mm}}/P_{{\rm DM,hydro}}$")
        ax2.plot(k, 1.0 - r['r_dmo'][sel], color=col, lw=2)
    ax1.axhline(1.0, color='k', lw=1)
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$S(k)$')
    ax1.legend(fontsize=9, loc='lower left', framealpha=0.85)
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    ax2.set_ylabel(r'$1 - r_{\rm mm\times DMO}(k)$')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def write_table(results: list, k_table: list, path: Path) -> None:
    """Write budget fractions and model quantities at k_table to a text file."""
    lines = []
    lines.append("# Global baryon budget (fractions of all baryon mass in the box)")
    lines.append(f"{'simulation':28s} {'alpha0(ion)':>11s} {'neutral':>9s} {'stars':>9s} {'BH':>9s}  reference")
    for r in results:
        fr = r['frac']
        lines.append(f"{r['label']:28s} {fr['ionized']:11.4f} {fr['neutral']:9.4f} "
                     f"{fr['stars']:9.4f} {fr['BH']:9.5f}  "
                     f"{'DMO run' if r['ref_is_dmo'] else 'hydro DM (stand-in)'}")
    lines.append("")
    lines.append("# Per simulation, at each k [h/Mpc]: S(alpha0), S(1), Q = P(1)/P(alpha0)-1 [%],")
    lines.append("# Q with only stars / only neutral gas / only BH moved to the ionized template [%],")
    lines.append("# Q with all non-ionized mass laid out like DM / uniformly [%], and the absolute")
    lines.append("# change in the suppression, dS = S(1) - S(alpha0).")
    hdr = (f"{'k':>6s} {'S0':>8s} {'S1':>8s} {'Q':>8s} {'Q_star':>8s} {'Q_neu':>8s} "
           f"{'Q_BH':>8s} {'Q_likeDM':>9s} {'Q_unif':>8s} {'dS':>8s}")
    for r in results:
        lines.append(f"\n## {r['label']}  (reference: {'DMO' if r['ref_is_dmo'] else 'hydro DM'})")
        lines.append(hdr)
        for kt in k_table:
            if kt > r['k'].max():
                continue
            i = int(np.argmin(np.abs(np.log(r['k'] / kt))))
            lines.append(
                f"{r['k'][i]:6.2f} {r['S0'][i]:8.4f} {r['S1'][i]:8.4f} {100 * r['Q'][i]:8.3f} "
                f"{100 * r['Q_comp']['stars'][i]:8.3f} {100 * r['Q_comp']['neutral'][i]:8.3f} "
                f"{100 * r['Q_comp']['BH'][i]:8.4f} {100 * r['Q_target']['dm'][i]:9.3f} "
                f"{100 * r['Q_target']['uniform'][i]:8.3f} {r['dS'][i]:+8.4f}")
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text)
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None,
                        help="restrict to these entries (label, name or feedback)")
    parser.add_argument('--kmax', type=float, default=10.0,
                        help="largest k plotted [h/Mpc] (default 10)")
    parser.add_argument('--suffix', default='',
                        help="extra tag appended to the output file names")
    args = parser.parse_args()

    config = load_config(args.path2config)
    plot_cfg = config['plot']
    now = datetime.now()
    out_dir = Path(plot_cfg['fig_path']) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = plot_cfg['fig_name'] + (f"_{args.suffix}" if args.suffix else '')
    ext = plot_cfg.get('fig_type', 'pdf')

    results = []
    for entry in select_sims(config, args.sims):
        if not spectra_path(entry, 'components').exists():
            print(f"no component spectra for {sim_label(entry)}; skipping")
            continue
        results.append(analyse(entry))
    if not results:
        raise SystemExit("no spectra found")

    colours = colours_for(results)
    fig_alpha(results, colours, args.kmax, out_dir / f"{stem}_alpha.{ext}")
    fig_targets(results, colours, args.kmax, out_dir / f"{stem}_targets.{ext}")
    fig_dmo_check(results, colours, args.kmax, out_dir / f"{stem}_dmo_check.{ext}")
    write_table(results, config['pk'].get('k_table', [1.0]), out_dir / f"{stem}_table.txt")


if __name__ == '__main__':
    main()
