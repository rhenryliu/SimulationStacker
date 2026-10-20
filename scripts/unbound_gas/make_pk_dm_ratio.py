"""make_pk_dm_ratio.py
====================
How closely does the dark matter of a hydro run follow its dark-matter-only
(DMO) twin, and what does using it as the reference of the suppression
S(k) = P_mm / P_DMO cost?

For each simulation in the config(s), from spectra already on disk:

    R(k)  = P_DM,hydro / P_DMO
    dS(k) = P_mm / P_DM,hydro - P_mm / P_DMO  = S(k) [1 / R(k) - 1]

P_DM,hydro is the auto spectrum of the hydro run's DM overdensity (derived as
total - gas - Stars - BH; see ``pk_common``), P_DMO that of the matched DMO run
on the same grid and k bins, and P_mm the hydro run's total matter. dS is the
error in S(k) when the hydro DM stands in for the DMO run: dS > 0 means the
stand-in understates the suppression.

When ``compute_pk_components.py --what dmcross`` has been run, also the
cross-correlation coefficient of the two DM fields,

    r(k) = P_DM,hydro x DMO / sqrt(P_DM,hydro P_DMO),

plotted as 1 - r. No spectrum has shot noise subtracted (Pylians' default, as
in the other P(k) scripts).

Inputs (``pk_common.spectra_path``), per simulation:
  <stem>_Pk_components_<n>.npz   P[DM, DM], P_total and the component spectra
  <stem>_Pk_dmo_<n>.npz          P_dmo
  <stem>_Pk_dmcross_<n>.npz      optional: P_dm, P_dmo, P_dm_dmo

Outputs in <fig_path>/YYYY-MM/MM-DD/ (fig_name from the first config):
  <fig_name>_dm_ratio.<ext>        R(k)
  <fig_name>_dm_dS.<ext>           dS(k)
  <fig_name>_dm_r.<ext>            1 - r(k) (only if dmcross spectra exist)
  <fig_name>_dm_S.<ext> and .png   S(k) with the DMO reference (solid) and the
                                   hydro-DM reference (dotted)
  <fig_name>_dm_ratio_table.txt    R, S, dS (and 1 - r) at pk.k_table, and
                                   large-scale diagnostics

The large-scale diagnostics (mode-weighted means over k <= --k-large, or the
lowest bin alone when the box has no bin there) separate
a difference between the two runs from one inside the hydro run: R, S, the
hydro DM against the hydro total matter, and the hydro baryons (all non-DM
components, mass-weighted) against the hydro DM, as an amplitude ratio
sqrt(P_bb / P_cc) and the correlation coefficient as 1 - r_bc.

Run from the scripts/ directory (light; login node is fine):
    python unbound_gas/make_pk_dm_ratio.py -p configs/unbound_gas/pk_components_z05.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import make_pk_stellar as mps  # plot style (via make_pk_alpha), colours, grid
from pk_common import load_config, select_sims, sim_label, spectra_path


def analyse(entry: dict) -> dict:
    """Load one simulation's spectra and evaluate R, dS and (if present) r."""
    comp = np.load(spectra_path(entry, 'components'))
    dmo = np.load(spectra_path(entry, 'dmo'))
    k = comp['k']
    if str(comp['components'][0]) != 'DM':
        raise ValueError(f"first component of {sim_label(entry)} is not DM")
    if dmo['k'].shape != k.shape or not np.allclose(dmo['k'], k):
        raise ValueError(f"k bins differ between the components and DMO files of {sim_label(entry)}")
    P = comp['P']
    P_dm, P_dmo, P_mm = P[0, 0], dmo['P_dmo'], comp['P_total']
    dev = float(np.max(np.abs(dmo['P_total'] / P_mm - 1.0)))
    if dev > 1e-4:
        raise ValueError(f"P_total differs between the components and DMO files of "
                         f"{sim_label(entry)} (max dev {dev:.2e})")

    # Hydro baryons: mass-weighted sum of the non-DM components.
    w = comp['means'] / comp['means'].sum()
    wb = w[1:] / w[1:].sum()
    P_bb = np.einsum('i,j,ijk->k', wb, wb, P[1:, 1:])
    P_bc = np.einsum('i,ik->k', wb, P[1:, 0])

    res = dict(label=sim_label(entry), sim_type=entry['sim_type'], name=entry['name'],
               feedback=entry.get('feedback'), k=k, Nmodes=comp['Nmodes'],
               R=P_dm / P_dmo, S=P_mm / P_dmo, dS=P_mm / P_dm - P_mm / P_dmo,
               dm_vs_mm=P_dm / P_mm, b_amp=np.sqrt(P_bb / P_dm), r_bc=P_bc / np.sqrt(P_bb * P_dm),
               r=None, cross_checks=None)

    cross_path = spectra_path(entry, 'dmcross')
    if cross_path.exists():
        x = np.load(cross_path)
        if x['k'].shape != k.shape or not np.allclose(x['k'], k):
            raise ValueError(f"k bins differ between {cross_path} and the components file")
        res['r'] = x['P_dm_dmo'] / np.sqrt(x['P_dm'] * x['P_dmo'])
        # The dmcross auto spectra must reproduce the ones already on disk.
        res['cross_checks'] = (float(np.max(np.abs(x['P_dm'] / P_dm - 1.0))),
                               float(np.max(np.abs(x['P_dmo'] / P_dmo - 1.0))))
        if max(res['cross_checks']) > 1e-4:
            raise ValueError(f"{cross_path} does not reproduce the components/DMO spectra "
                             f"(max dev {max(res['cross_checks']):.2e})")
    else:
        print(f"no DM cross spectra for {res['label']} ({cross_path.name})")
    return res


def _plot(results: list, key: str, ylabel: str, ref: Optional[float], kmax: float, path: Path,
          transform=None, logy: bool = False) -> None:
    """One curve per simulation of results[key] (optionally transformed) against k."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        sel = r['k'] <= kmax
        y = r[key] if transform is None else transform(r[key])
        ax.plot(r['k'][sel], y[sel], color=mps.sim_colour(r), lw=2, label=r['label'])
    if ref is not None:
        ax.axhline(ref, color='k', lw=1, ls='--')
    mps._style_axis(ax)
    if logy:
        ax.set_yscale('log')
    mps._k_range(ax, results, kmax)
    ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', framealpha=0.85)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


def fig_S(results: list, kmax: float, paths: list) -> None:
    """S(k) with the DMO reference (solid) and the hydro-DM reference (dotted)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        sel = r['k'] <= kmax
        col = mps.sim_colour(r)
        ax.plot(r['k'][sel], r['S'][sel], color=col, lw=2, label=r['label'])
        ax.plot(r['k'][sel], (r['S'] + r['dS'])[sel], color=col, lw=2, ls=':')
    ax.axhline(1.0, color='k', lw=1, ls='--')
    mps._style_axis(ax)
    mps._k_range(ax, results, kmax)
    ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$S(k) = P_{\rm mm}/P_{\rm ref}$')
    sims = ax.legend(loc='lower left', framealpha=0.85)
    ax.add_artist(sims)
    style = [Line2D([], [], color='k', lw=2, ls='-', label=r'$P_{\rm ref} = P_{\rm DMO}$'),
             Line2D([], [], color='k', lw=2, ls=':', label=r'$P_{\rm ref} = P_{\rm DM,hydro}$')]
    ax.legend(handles=style, loc='center left', framealpha=0.85)
    for path in paths:
        fig.savefig(path, bbox_inches='tight', dpi=200)
        print(f"saved {path}")
    plt.close(fig)


def _large_bins(r: dict, k_large: float) -> np.ndarray:
    """Bins with k <= k_large, or the lowest bin alone if there is none."""
    sel = r['k'] <= k_large
    if not sel.any():
        sel[0] = True
    return sel


def _mean_large(r: dict, values: np.ndarray, k_large: float) -> float:
    """Mode-weighted mean of ``values`` over ``_large_bins``."""
    sel = _large_bins(r, k_large)
    return float(np.average(values[sel], weights=r['Nmodes'][sel]))


def write_table(results: list, k_table: list, k_large: float, path: Path) -> None:
    """Write R, S, dS (and 1 - r) at k_table plus the large-scale diagnostics."""
    lines = [f"# Large scales: mode-weighted means over k <= {k_large:g} h/Mpc (lowest bin if none)",
             "# R = P_DM,hydro/P_DMO; S = P_mm/P_DMO; DM/mm = P_DM,hydro/P_mm (inside the hydro run);",
             "# b_amp = sqrt(P_bb/P_DM,hydro) and 1 - r_bc: hydro baryons against hydro DM.",
             f"{'simulation':28s} {'bins':>4s} {'modes':>8s} {'R':>8s} {'S':>8s} "
             f"{'DM/mm':>8s} {'b_amp':>8s} {'1-r_bc':>9s} {'1-r':>9s}"]
    for r in results:
        sel = _large_bins(r, k_large)
        one_minus_r = (f"{_mean_large(r, 1.0 - r['r'], k_large):9.2e}"
                       if r['r'] is not None else f"{'-':>9s}")
        lines.append(f"{r['label']:28s} {sel.sum():4d} {int(r['Nmodes'][sel].sum()):8d} "
                     f"{_mean_large(r, r['R'], k_large):8.5f} {_mean_large(r, r['S'], k_large):8.5f} "
                     f"{_mean_large(r, r['dm_vs_mm'], k_large):8.5f} "
                     f"{_mean_large(r, r['b_amp'], k_large):8.5f} "
                     f"{_mean_large(r, 1.0 - r['r_bc'], k_large):9.2e} {one_minus_r}")
    lines.append("")
    lines.append("# Per simulation, at each k [h/Mpc] (nearest bin): R, S = P_mm/P_DMO,")
    lines.append("# S with the hydro-DM reference, dS = S_hydroDM - S (> 0: stand-in understates")
    lines.append("# the suppression), and 1 - r of the hydro DM x DMO fields.")
    hdr = f"{'k':>6s} {'R':>8s} {'S':>8s} {'S_hydroDM':>9s} {'dS':>8s} {'1-r':>9s}"
    for r in results:
        lines.append(f"\n## {r['label']}")
        if r['cross_checks'] is not None:
            lines.append(f"# dmcross check: max |P_dm/P[DM,DM] - 1| = {r['cross_checks'][0]:.1e}, "
                         f"max |P_dmo(dmcross)/P_dmo(DMO file) - 1| = {r['cross_checks'][1]:.1e}")
        lines.append(hdr)
        for kt in k_table:
            if kt > r['k'].max():
                continue
            i = int(np.argmin(np.abs(np.log(r['k'] / kt))))
            one_minus_r = f"{1.0 - r['r'][i]:9.2e}" if r['r'] is not None else f"{'-':>9s}"
            lines.append(f"{r['k'][i]:6.2f} {r['R'][i]:8.4f} {r['S'][i]:8.4f} "
                         f"{r['S'][i] + r['dS'][i]:9.4f} {r['dS'][i]:+8.4f} {one_minus_r}")
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text)
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', nargs='+', required=True,
                        help="one or more configs; fig_path, fig_name and k_table come from the first")
    parser.add_argument('--sims', nargs='*', default=None,
                        help="restrict to these entries (label, name or feedback)")
    parser.add_argument('--kmax', type=float, default=10.0,
                        help="largest k plotted [h/Mpc] (default 10)")
    parser.add_argument('--k-large', type=float, default=0.1,
                        help="upper k of the large-scale diagnostics [h/Mpc] (default 0.1)")
    parser.add_argument('--suffix', default='',
                        help="extra tag appended to the output file names")
    args = parser.parse_args()

    configs = [load_config(p) for p in args.path2config]
    plot_cfg = configs[0]['plot']
    now = datetime.now()
    out_dir = Path(plot_cfg['fig_path']) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = plot_cfg['fig_name'] + (f"_{args.suffix}" if args.suffix else '')
    ext = plot_cfg.get('fig_type', 'pdf')

    results = []
    for config in configs:
        for entry in select_sims(config, args.sims):
            if not (spectra_path(entry, 'components').exists()
                    and spectra_path(entry, 'dmo').exists()):
                print(f"no components or DMO spectra for {sim_label(entry)}; skipping")
                continue
            results.append(analyse(entry))
    if not results:
        raise SystemExit("no spectra found")

    _plot(results, 'R', r'$P_{\rm DM,hydro}(k)/P_{\rm DMO}(k)$', 1.0, args.kmax,
          out_dir / f"{stem}_dm_ratio.{ext}")
    _plot(results, 'dS', r'$\Delta S(k) = P_{\rm mm}/P_{\rm DM,hydro} - P_{\rm mm}/P_{\rm DMO}$',
          0.0, args.kmax, out_dir / f"{stem}_dm_dS.{ext}")
    have_r = [r for r in results if r['r'] is not None]
    if have_r:
        _plot(have_r, 'r', r'$1 - r_{\rm DM,hydro\times DMO}(k)$', None, args.kmax,
              out_dir / f"{stem}_dm_r.{ext}", transform=lambda x: 1.0 - x, logy=True)
    fig_S(results, args.kmax, [out_dir / f"{stem}_dm_S.{ext}", out_dir / f"{stem}_dm_S.png"])
    write_table(results, configs[0]['pk'].get('k_table', [1.0]), args.k_large,
                out_dir / f"{stem}_dm_ratio_table.txt")


if __name__ == '__main__':
    main()
