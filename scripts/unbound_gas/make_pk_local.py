"""make_pk_local.py
=================
Figures and tables for the local redistribution model of the unbound gas
paper's power spectrum section (spectra from ``compute_pk_local.py``; the
global alpha model of ``make_pk_alpha.py`` is shown alongside as the
R -> infinity reference and is not modified).

For a moved set s (all non-ionized = neutral_gas + Stars + BH, or stars only,
or neutral gas only), target T (local ionized gas or local DM), kernel and
radius R, the matter field is the simulation's with the components of s
replaced by their transported field m_s (same total mass):

    P_local = c^T P_ext c ,  c = weights of (DM, ionized, neutral, stars, BH, m_s)

with the moved components' own weights set to zero and m_s carrying their
mass. Quantities (as in make_pk_alpha.py):

    Q_local(k)  = P_local / P_mm - 1          (reference-free)
    dS_local(k) = (P_local - P_mm) / P_DMO    (absolute change in S = P/P_DMO)

The global model's Q and dS are recomputed here from the same estimator's
original spectra (``*_Pk_local_orig_*``), identical to make_pk_alpha.py's up to
~1e-7. SIMBA is labelled provisional while its ElectronAbundance convention is
investigated (NOTES/unbound_gas/simba_electron_abundance_handoff.md).

Outputs in <fig_path>/YYYY-MM/MM-DD/ (fig_name from the config):
  <fig_name>_local_S_<kernel>_R<R>_<target>.<ext>
                                      S(k) bands for all simulations at one R
                                      (--band-radius/--band-kernel; layout of
                                      make_pk_alpha's _alpha figure), dS below
  <fig_name>_local_S_grid_<kernel>_<target>.<ext>
                                      per simulation: S(k) and dS for every R
                                      and the global model
  <fig_name>_local_Q_<target>.<ext>   Q(k): one panel per simulation, one curve
                                      per R (solid sphere, dashed Gaussian),
                                      black = global model; all non-ionized moved
  <fig_name>_local_split.<ext>        Q(k) at one R (--split-radius, sphere):
                                      all / stars only / neutral only, both targets
  <fig_name>_local_table.txt          Q and dS at pk.k_table for every
                                      configuration, large-scale Q, diagnostics

Run from the scripts/ directory (light; login node is fine):
    python unbound_gas/make_pk_local.py -p configs/unbound_gas/pk_components_z05.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import make_pk_alpha as mpa  # reuses the global-model algebra and plot style
from pk_common import load_config, select_sims, sim_label, spectra_path

MOVED_MEMBERS = {'all': [mpa.I_NEU, mpa.I_ST, mpa.I_BH], 'stars': [mpa.I_ST],
                 'neutral': [mpa.I_NEU]}
KERNEL_STYLE = {'tophat': '-', 'gaussian': '--'}
TARGET_LABEL = {'ionized': 'like local ionized gas', 'dm': 'like local DM'}


def label_of(entry: dict) -> str:
    """Legend label; SIMBA flagged provisional (ElectronAbundance convention)."""
    lab = sim_label(entry)
    return lab + ' (provisional)' if entry['sim_type'] == 'SIMBA' else lab


def p_local(P: np.ndarray, w: np.ndarray, members: list, P_auto: np.ndarray,
            P_cross: np.ndarray) -> np.ndarray:
    """P_mm with the ``members`` components replaced by their transported field.

    Args:
        P: original component spectra (5, 5, Nk); w: component weights (5).
        members: indices of the moved components.
        P_auto: auto spectrum of the moved field (Nk).
        P_cross: its cross spectra with the five originals (5, Nk).
    """
    nc = len(w)
    Pext = np.zeros((nc + 1, nc + 1, P.shape[-1]))
    Pext[:nc, :nc] = P
    Pext[nc, :nc] = Pext[:nc, nc] = P_cross
    Pext[nc, nc] = P_auto
    c = np.append(w, w[members].sum())
    c[members] = 0.0
    return np.einsum('i,j,ijk->k', c, c, Pext)


def analyse(entry: dict, config: dict) -> dict:
    """Global and local model quantities for one simulation."""
    lc = config['local']
    orig = np.load(spectra_path(entry, 'local_orig'))
    k, P, means = orig['k'], orig['P'], orig['means']
    w = means / means.sum()
    dmo = np.load(spectra_path(entry, 'dmo'))
    if len(dmo['k']) != len(k) or not np.allclose(dmo['k'], k, rtol=1e-10):
        raise ValueError(f"k bins differ between the DMO and local spectra of {sim_label(entry)}")
    P0 = mpa.p_of(w, P)
    Pg = mpa.p_of(mpa.weights_alpha(w, 1.0), P)
    res = dict(label=label_of(entry), sim_type=entry['sim_type'], k=k, P0=P0,
               P_dmo=dmo['P_dmo'], nmodes=orig['Nmodes'], kF=2 * np.pi / float(orig['box_mpc']),
               alpha0=w[mpa.I_ION] / w[1:].sum(), S0=P0 / dmo['P_dmo'],
               S_global=Pg / dmo['P_dmo'],
               Q_global=Pg / P0 - 1.0, dS_global=(Pg - P0) / dmo['P_dmo'], local={})
    for kern in lc['kernels']:
        for R in lc['radii']:
            path = spectra_path(entry, f"local_{kern}_R{float(R):g}")
            if not path.exists():
                continue
            f = np.load(path)
            for t in lc['targets']:
                for s in lc['moved']:
                    key = f"{t}__{s}"
                    Pl = p_local(P, w, MOVED_MEMBERS[s], f[f"P_auto__{key}"], f[f"P_cross__{key}"])
                    res['local'][(kern, float(R), t, s)] = dict(
                        S=Pl / dmo['P_dmo'], Q=Pl / P0 - 1.0, dS=(Pl - P0) / dmo['P_dmo'],
                        renorm=float(f[f"renorm__{key}"]), kept=float(f[f"kept_frac__{key}"]))
    return res


def fig_Q(results: list, config: dict, target: str, kmax: float, path: Path) -> None:
    """Q(k) per simulation for every R and kernel (all non-ionized moved)."""
    radii = [float(R) for R in config['local']['radii']]
    cols = matplotlib.colormaps['plasma'](np.linspace(0.1, 0.8, len(radii)))  # type: ignore
    n = len(results)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), sharex=True,
                             sharey=True, squeeze=False)
    for ax, r in zip(axes.flat, results):
        sel = r['k'] <= kmax
        ax.plot(r['k'][sel], 100 * r['Q_global'][sel], color='k', lw=2, label=r'global ($R\to\infty$)')
        for R, col in zip(radii, cols):
            for kern, ls in KERNEL_STYLE.items():
                d = r['local'].get((kern, R, target, 'all'))
                if d is not None:
                    ax.plot(r['k'][sel], 100 * d['Q'][sel], color=col, ls=ls, lw=1.5,
                            label=rf'$R={R:g}$' if kern == 'tophat' else None)
        ax.axhline(0.0, color='gray', lw=0.8)
        ax.set_xscale('log')
        ax.set_title(r['label'], fontsize=13)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$P_{\rm mm}^{\rm moved}/P_{\rm mm} - 1\;[\%]$')
    h, lab = axes.flat[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color='gray', ls=ls) for ls in KERNEL_STYLE.values()]
    lab += ['sphere', 'Gaussian']
    axes.flat[0].legend(h, lab, fontsize=9, title=rf'non-ionized baryons {TARGET_LABEL[target]}; $R$ [$h^{{-1}}$Mpc]',
                        title_fontsize=9, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


def fig_split(results: list, R: float, kmax: float, path: Path) -> None:
    """Q(k) at one sphere radius: all / stars only / neutral only, both targets."""
    n = len(results)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), sharex=True,
                             sharey=True, squeeze=False)
    styles = {'all': '-', 'stars': '--', 'neutral': ':'}
    cols = {'ionized': 'C0', 'dm': 'C3'}
    for ax, r in zip(axes.flat, results):
        sel = r['k'] <= kmax
        for t, col in cols.items():
            for s, ls in styles.items():
                d = r['local'].get(('tophat', R, t, s))
                if d is not None:
                    ax.plot(r['k'][sel], 100 * d['Q'][sel], color=col, ls=ls, lw=1.5)
        ax.axhline(0.0, color='gray', lw=0.8)
        ax.set_xscale('log')
        ax.set_title(r['label'], fontsize=13)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$P_{\rm mm}^{\rm moved}/P_{\rm mm} - 1\;[\%]$')
    h = [plt.Line2D([], [], color=c) for c in cols.values()] + \
        [plt.Line2D([], [], color='gray', ls=ls) for ls in styles.values()]
    lab = [TARGET_LABEL[t] for t in cols] + ['all non-ionized', 'stars only', 'neutral only']
    axes.flat[0].legend(h, lab, fontsize=9, title=rf'sphere, $R={R:g}\,h^{{-1}}$Mpc',
                        title_fontsize=9, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


KERNEL_NAME = {'tophat': 'sphere', 'gaussian': 'Gaussian'}


def fig_S_band(results: list, R: float, kern: str, target: str, kmax: float,
               path: Path) -> None:
    """S(k) bands, simulation -> local model at one R (layout of make_pk_alpha's figure).

    Top: S = P_mm/P_DMO for the simulation (solid) and with all non-ionized
    baryons moved within R (dashed), shaded between; the global model is dotted.
    Bottom: the absolute change dS = S(moved) - S(simulation), local (solid)
    and global (dotted).
    """
    colours = mpa.colours_for(results)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.05))
    for r, col in zip(results, colours):
        d = r['local'].get((kern, R, target, 'all'))
        if d is None:
            print(f"  {r['label']}: no {kern} R={R:g} spectra; left out of the band figure")
            continue
        sel = r['k'] <= kmax
        k = r['k'][sel]
        ax1.plot(k, r['S0'][sel], color=col, lw=2, label=r['label'])
        ax1.plot(k, d['S'][sel], color=col, lw=1.5, ls='--')
        ax1.plot(k, r['S_global'][sel], color=col, lw=1, ls=':')
        ax1.fill_between(k, r['S0'][sel], d['S'][sel], color=col, alpha=0.2, lw=0)
        ax2.plot(k, d['dS'][sel], color=col, lw=2)
        ax2.plot(k, r['dS_global'][sel], color=col, lw=1, ls=':')
    ax1.axhline(1.0, color='k', lw=1)
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$S(k) = P_{\rm mm}/P_{\rm DMO}$')
    ax1.legend(loc='lower left', framealpha=0.85)
    ax1.set_title(rf'non-ionized baryons moved within $R={R:g}\,h^{{-1}}$Mpc '
                  rf'({KERNEL_NAME[kern]}), {TARGET_LABEL[target]}', fontsize=13)
    ax2.axhline(0.0, color='k', lw=1)
    ax2.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    ax2.set_ylabel(r'$\Delta S = S_{\rm moved} - S$')
    ax2.plot([], [], color='gray', lw=2, label='simulation (solid, top)')
    ax2.plot([], [], color='gray', lw=1.5, ls='--', label=rf'local, $R={R:g}$ (dashed top, solid bottom)')
    ax2.plot([], [], color='gray', lw=1, ls=':', label=r'global model ($R\to\infty$)')
    ax2.legend(loc='lower left', framealpha=0.85, fontsize=10)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


def fig_S_grid(results: list, config: dict, kern: str, target: str, kmax: float,
               path: Path) -> None:
    """Per simulation: S(k) (top) and dS (bottom) for every R, plus the global model."""
    radii = [float(R) for R in config['local']['radii']]
    cols = matplotlib.colormaps['plasma'](np.linspace(0.1, 0.8, len(radii)))  # type: ignore
    n = len(results)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(5 * ncol, 5.5 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.22, wspace=0.08)
    first, firstD = None, None
    for i, r in enumerate(results):
        row, col = divmod(i, ncol)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[1.6, 1], hspace=0.05)
        axS = fig.add_subplot(inner[0], sharex=first, sharey=first)
        axD = fig.add_subplot(inner[1], sharex=axS, sharey=firstD)
        if first is None:
            first, firstD = axS, axD
        sel = r['k'] <= kmax
        k = r['k'][sel]
        axS.plot(k, r['S0'][sel], color='k', lw=2, label='simulation')
        axS.plot(k, r['S_global'][sel], color='k', lw=1.2, ls=':', label=r'global ($R\to\infty$)')
        axD.plot(k, r['dS_global'][sel], color='k', lw=1.2, ls=':')
        for R, c in zip(radii, cols):
            d = r['local'].get((kern, R, target, 'all'))
            if d is not None:
                axS.plot(k, d['S'][sel], color=c, lw=1.5, label=rf'$R={R:g}$')
                axD.plot(k, d['dS'][sel], color=c, lw=1.5)
        axS.axhline(1.0, color='gray', lw=0.8)
        axD.axhline(0.0, color='gray', lw=0.8)
        axS.set_xscale('log')
        axS.set_title(r['label'], fontsize=13)
        plt.setp(axS.get_xticklabels(), visible=False)
        if col == 0:
            axS.set_ylabel(r'$S(k)$')
            axD.set_ylabel(r'$\Delta S$')
        else:
            plt.setp(axS.get_yticklabels(), visible=False)
            plt.setp(axD.get_yticklabels(), visible=False)
        if i + ncol >= n:  # nothing below this panel
            axD.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
        else:
            plt.setp(axD.get_xticklabels(), visible=False)
        if i == 0:
            axS.legend(fontsize=9, loc='lower left',
                       title=rf'{KERNEL_NAME[kern]}, {TARGET_LABEL[target]}; $R$ [$h^{{-1}}$Mpc]',
                       title_fontsize=9)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


def write_table(results: list, config: dict, path: Path) -> None:
    """Q and dS at pk.k_table for every configuration, plus diagnostics."""
    lc = config['local']
    k_table = config['pk'].get('k_table', [1.0])
    lines = ["# Local redistribution model. Q = P_moved/P_mm - 1 [%], dS = S_moved - S [absolute].",
             "# Columns: global model (R = infinity), then kernel/R [Mpc/h] (th = sphere, ga = Gaussian).",
             "# 'large-scale' rows: mode-weighted mean over k <= 3 k_F (should be ~0 for local models).",
             "# SIMBA is provisional (ElectronAbundance convention under investigation)."]
    for r in results:
        confs = sorted({(kern, R) for (kern, R, _, _) in r['local']},
                       key=lambda c: (c[0] != 'tophat', c[1]))
        names = ['global'] + [f"{'th' if kern == 'tophat' else 'ga'}{R:g}" for kern, R in confs]
        for t in lc['targets']:
            for s in lc['moved']:
                lines.append(f"\n## {r['label']} | moved: {s} | {TARGET_LABEL[t]}")
                lines.append(f"{'k':>6s} {'qty':>3s} " + " ".join(f"{nm:>8s}" for nm in names))
                for kt in k_table:
                    if kt > r['k'].max():
                        continue
                    i = int(np.argmin(np.abs(np.log(r['k'] / kt))))
                    for qty, scale, fmt in (('Q', 100, '8.3f'), ('dS', 1, '+8.4f')):
                        vals = [(r['Q_global'] if qty == 'Q' else r['dS_global'])[i] * scale
                                if s == 'all' else np.nan]
                        vals += [r['local'][(kern, R, t, s)][qty][i] * scale for kern, R in confs]
                        lines.append(f"{r['k'][i]:6.2f} {qty:>3s} " +
                                     " ".join(f"{v:{fmt}}" if np.isfinite(v) else f"{'-':>8s}" for v in vals))
                ls = r['k'] <= 3 * r['kF'] * 1.0001
                wls = r['nmodes'][ls]
                vals = [np.average(r['Q_global'][ls], weights=wls) * 100 if s == 'all' else np.nan]
                vals += [np.average(r['local'][(kern, R, t, s)]['Q'][ls], weights=wls) * 100
                         for kern, R in confs]
                lines.append(f"{'large-scale':>10s} " + " ".join(f"{v:8.3f}" if np.isfinite(v) else f"{'-':>8s}" for v in vals))
                diag = [f"{nm}: renorm-1={r['local'][(kern, R, t, s)]['renorm'] - 1:+.1e}, "
                        f"kept={r['local'][(kern, R, t, s)]['kept']:.1e}"
                        for nm, (kern, R) in zip(names[1:], confs)]
                lines.append("# diagnostics: " + "; ".join(diag))
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None)
    parser.add_argument('--kmax', type=float, default=10.0)
    parser.add_argument('--split-radius', type=float, default=1.0,
                        help="sphere radius [Mpc/h] for the per-component figure (default 1)")
    parser.add_argument('--band-radius', type=float, default=1.0,
                        help="R [Mpc/h] of the S(k) band figure (default 1)")
    parser.add_argument('--band-kernel', default='tophat', choices=['tophat', 'gaussian'],
                        help="kernel of the S(k) band figure (default tophat)")
    parser.add_argument('--suffix', default='')
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
        if not spectra_path(entry, 'local_orig').exists():
            print(f"no local spectra for {sim_label(entry)}; skipping")
            continue
        results.append(analyse(entry, config))
    if not results:
        raise SystemExit("no local spectra found")

    for t in config['local']['targets']:
        fig_Q(results, config, t, args.kmax, out_dir / f"{stem}_local_Q_{t}.{ext}")
        fig_S_band(results, args.band_radius, args.band_kernel, t, args.kmax,
                   out_dir / f"{stem}_local_S_{args.band_kernel}_R{args.band_radius:g}_{t}.{ext}")
        for kern in config['local']['kernels']:
            fig_S_grid(results, config, kern, t, args.kmax,
                       out_dir / f"{stem}_local_S_grid_{kern}_{t}.{ext}")
    fig_split(results, args.split_radius, args.kmax, out_dir / f"{stem}_local_split.{ext}")
    write_table(results, config, out_dir / f"{stem}_local_table.txt")


if __name__ == '__main__':
    main()
