"""make_pk_stellar.py
===================
Figures and tables for the halo-level stellar-to-ionized-gas transfer of the
unbound gas paper's power spectrum section (spectra from
``compute_pk_stellar.py``; the global and local models of
``make_pk_alpha.py`` / ``make_pk_local.py`` are shown alongside for context
and are not modified).

For a method variant (halo-finder membership 'fof', or apertures 'ap<x>' of
x R200m), halo mass cut M_min and stellar scale s (fraction of the selected
stellar mass kept as stars; the rest is ionized gas laid out like each
halo's ionized gas):

    P_mm(s) = P_mm + 2 (1 - s) P_mD + (1 - s)^2 P_DD      (exact)
    Q(s)    = P_mm(s) / P_mm - 1                         (= P_modified/P_original - 1)
    S(s)    = P_mm(s) / P_DMO ,  dS(s) = S(s) - S(1)

The context curves move all stars: 'global' = like the box-wide ionized gas
(make_pk_alpha's stars-only variant), 'local R=1' = transported within a
1 Mpc/h sphere to follow the local ionized gas (make_pk_local's 'stars' set).
Both use the cached Stars field, which for TNG/Illustris includes wind-phase
particles (~8% of Illustris-1's; excluded from the halo-level transfer).

Outputs in <fig_path>/YYYY-MM/MM-DD/ (fig_name from the config):
  <fig_name>_stellar_scales_<variant>_<tag>.<ext>
                                  Q(k) for every s of stellar.stellar_scales,
                                  one panel per simulation (--scale-variant,
                                  --scale-tag; default fof, lowest cut)
  <fig_name>_stellar_methods.<ext>
                                  Q(k) at s = 0 for every method and mass cut
  <fig_name>_stellar_S_<variant>_<tag>.<ext>
                                  S(k) from s = 1 to s = 0 for all simulations
                                  with dS below (layout of make_pk_alpha's
                                  _alpha figure), one per method variant
                                  (--band-tag; default lowest cut)
  <fig_name>_stellar_table.txt    budgets, Q and dS at pk.k_table, large-scale
                                  Q, and every validation number

Run from the scripts/ directory (light; login node is fine):
    python unbound_gas/make_pk_stellar.py -p configs/unbound_gas/pk_components_z05.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import make_pk_alpha as mpa  # plot style, colours and the global-model algebra
import make_pk_local as mpl  # local-model spectra and the SIMBA label
from compute_pk_stellar import mass_tag, variants_of
from pk_common import load_config, select_sims, sim_label, spectra_path

import halo_transfer as ht

VARIANT_COLOUR = {'fof': 'C0', 'ap1': 'C1', 'ap2': 'C2'}
TAG_STYLE = ['-', '--', ':']


def variant_label(name: str) -> str:
    """Legend label of a method variant."""
    if name == 'fof':
        return 'halo-finder membership'
    return rf"aperture $r<{name[2:]}\,R_{{\rm 200m}}$"


def tag_label(tag: str) -> str:
    """Legend label of a mass-cut tag, e.g. 'M11'."""
    return rf"$M_{{\rm FoF}}\geq10^{{{tag[1:]}}}\,h^{{-1}}M_\odot$"


def analyse(entry: dict, config: dict) -> dict:
    """Stellar-transfer quantities (and context models) for one simulation."""
    scales = [float(s) for s in config['stellar']['stellar_scales']]
    dmo = np.load(spectra_path(entry, 'dmo'))
    comp = np.load(spectra_path(entry, 'components'))
    k = comp['k']
    if not np.allclose(dmo['k'], k, rtol=1e-10):
        raise ValueError(f"k bins differ between the DMO and components spectra of {sim_label(entry)}")
    P_dmo = dmo['P_dmo']
    n = int(comp['n_pixels'])
    means = comp['means']
    w = means / means.sum()
    P0g = mpa.p_of(w, comp['P'])
    Pg = mpa.p_of(mpa.weights_move(w, [mpa.I_ST], 'ionized'), comp['P'])
    res = dict(label=mpl.label_of(entry), sim_type=entry['sim_type'], k=k, P_dmo=P_dmo,
               nmodes=comp['Nmodes'], kF=2 * np.pi / float(comp['box_mpc']), scales=scales,
               mstar_cache=float(means[mpa.I_ST]) * n ** 3,
               mbaryon=float(means[1:].sum()) * n ** 3,
               Q_global=Pg / P0g - 1.0, dS_global=(Pg - P0g) / P_dmo, cfg={})
    try:
        loc = mpl.analyse(entry, config)['local'].get(('tophat', 1.0, 'ionized', 'stars'))
    except (FileNotFoundError, KeyError):
        loc = None
    res['Q_local'] = None if loc is None else loc['Q']
    res['dS_local'] = None if loc is None else loc['dS']

    for v in variants_of(config):
        path = spectra_path(entry, f"stellar_{v['name']}")
        if not path.exists():
            continue
        f = np.load(path)
        if not np.allclose(f['k'], k, rtol=1e-10):
            raise ValueError(f"k bins differ in {path}")
        P_mm = f['P_mm']
        res['S0'] = P_mm / P_dmo
        res['P_mm'] = P_mm
        res['pylians_dev'] = float(f['max_rel_diff_vs_pylians'])
        for key in ('mstar_box', 'mwind_box', 'mion_box', 'mass_total', 'box_mpc'):
            res[key] = float(f[key])
        for tag in f['tags']:
            tag = str(tag)
            P_mD, P_DD = f[f"P_mD__{tag}"], f[f"P_DD__{tag}"]
            diag = {kk.split('__', 2)[2]: float(f[kk]) for kk in f.files
                    if kk.startswith(f"diag__{tag}__")}
            Ps = {s: ht.p_of_scale(P_mm, P_mD, P_DD, s) for s in scales}
            d = dict(Q={s: Ps[s] / P_mm - 1.0 for s in scales},
                     S={s: Ps[s] / P_dmo for s in scales},
                     dS={s: (Ps[s] - P_mm) / P_dmo for s in scales}, diag=diag)
            for s in (0.0, 0.5):
                kk = f"explicit_check_s{s:g}__{tag}"
                if kk in f.files:
                    d[f'explicit_s{s:g}'] = float(f[kk])
            if v['method'] == 'membership' and 'catalogue_check_median' in f.files:
                d['catalogue'] = (float(f['catalogue_check_median']), float(f['catalogue_check_p99']),
                                  float(f['catalogue_check_massweighted']))
            # Poisson term change at s = 0 (V sum(m^2) / M^2), relative to P_mm at Nyquist.
            dsum = diag['sn_S1'] + diag['sn_S2'] - diag['sn_S3']
            d['dPsn_rel'] = res['box_mpc'] ** 3 * dsum / res['mass_total'] ** 2 / P_mm[-1]
            res['cfg'][(v['name'], tag)] = d
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _panel_grid(n: int):
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), sharex=True,
                             sharey=True, squeeze=False)
    return fig, axes


def _finish_grid(fig, axes, n, ylabel, path):
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


def _context(ax, r, sel, which='Q'):
    ax.plot(r['k'][sel], 100 * r[f'{which}_global'][sel] if which == 'Q' else r[f'{which}_global'][sel],
            color='k', ls=':', lw=1.2)
    if r[f'{which}_local'] is not None:
        ax.plot(r['k'][sel], 100 * r[f'{which}_local'][sel] if which == 'Q' else r[f'{which}_local'][sel],
                color='gray', ls='-.', lw=1.2)


def fig_scales(results: list, variant: str, tag: str, kmax: float, path: Path) -> None:
    """Q(k) = P_mm(s)/P_mm - 1 for every stellar scale s, one panel per simulation."""
    have = [r for r in results if (variant, tag) in r['cfg']]
    if not have:
        print(f"no {variant} {tag} spectra; skipping {path.name}")
        return
    scales = [s for s in have[0]['scales'] if s < 1.0]
    cols = matplotlib.colormaps['viridis'](np.linspace(0.85, 0.1, len(scales)))  # type: ignore
    fig, axes = _panel_grid(len(have))
    for ax, r in zip(axes.flat, have):
        sel = r['k'] <= kmax
        d = r['cfg'][(variant, tag)]
        for s, c in zip(scales, cols):
            ax.plot(r['k'][sel], 100 * d['Q'][s][sel], color=c, lw=1.8, label=rf'$s={s:g}$')
        _context(ax, r, sel)
        ax.axhline(0.0, color='gray', lw=0.8)
        ax.set_xscale('log')
        ax.set_title(r['label'], fontsize=13)
    h, lab = axes.flat[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color='k', ls=':'), plt.Line2D([], [], color='gray', ls='-.')]
    lab += ['all stars, global', r'all stars, local $R=1$']
    axes.flat[0].legend(h, lab, fontsize=9, loc='lower left',
                        title=f"{variant_label(variant)}, {tag_label(tag)}", title_fontsize=9)
    _finish_grid(fig, axes, len(have), r'$P_{\rm mm}(s)/P_{\rm mm} - 1\;[\%]$', path)


def fig_methods(results: list, s: float, kmax: float, path: Path) -> None:
    """Q(k) at one stellar scale for every method variant and mass cut."""
    fig, axes = _panel_grid(len(results))
    tags = sorted({t for r in results for (_, t) in r['cfg']}, key=lambda t: float(t[1:]))
    names = [nm for nm in VARIANT_COLOUR if any(nm == v for r in results for (v, _) in r['cfg'])]
    for ax, r in zip(axes.flat, results):
        sel = r['k'] <= kmax
        for nm in names:
            for tag, ls in zip(tags, TAG_STYLE):
                d = r['cfg'].get((nm, tag))
                if d is not None:
                    ax.plot(r['k'][sel], 100 * d['Q'][s][sel], color=VARIANT_COLOUR[nm], ls=ls, lw=1.6)
        _context(ax, r, sel)
        ax.axhline(0.0, color='gray', lw=0.8)
        ax.set_xscale('log')
        ax.set_title(r['label'], fontsize=13)
    h = [plt.Line2D([], [], color=VARIANT_COLOUR[nm]) for nm in names] + \
        [plt.Line2D([], [], color='gray', ls=ls) for ls in TAG_STYLE[:len(tags)]] + \
        [plt.Line2D([], [], color='k', ls=':'), plt.Line2D([], [], color='gray', ls='-.')]
    lab = [variant_label(nm) for nm in names] + [tag_label(t) for t in tags] + \
        ['all stars, global', r'all stars, local $R=1$']
    axes.flat[0].legend(h, lab, fontsize=8, loc='lower left', title=rf'$s={s:g}$', title_fontsize=9)
    _finish_grid(fig, axes, len(results), r'$P_{\rm mm}(s)/P_{\rm mm} - 1\;[\%]$', path)


def fig_S_band(results: list, variant: str, tag: str, kmax: float, path: Path) -> None:
    """S(k) from the simulation (s = 1) to all selected stars moved (s = 0), dS below."""
    have = [r for r in results if (variant, tag) in r['cfg']]
    if not have:
        print(f"no {variant} {tag} spectra; skipping {path.name}")
        return
    colours = mpa.colours_for(have)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.05))
    for r, col in zip(have, colours):
        d = r['cfg'][(variant, tag)]
        sel = r['k'] <= kmax
        k = r['k'][sel]
        s_lo = min(d['S'])
        ax1.plot(k, r['S0'][sel], color=col, lw=2, label=r['label'])
        ax1.plot(k, d['S'][s_lo][sel], color=col, lw=1.5, ls='--')
        ax1.fill_between(k, r['S0'][sel], d['S'][s_lo][sel], color=col, alpha=0.2, lw=0)
        ax2.plot(k, d['dS'][s_lo][sel], color=col, lw=2)
        ax2.plot(k, r['dS_global'][sel], color=col, lw=1, ls=':')
    ax1.axhline(1.0, color='k', lw=1)
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$S(k) = P_{\rm mm}/P_{\rm DMO}$')
    ax1.legend(loc='lower left', framealpha=0.85)
    ax1.set_title(f"stars moved to ionized gas: {variant_label(variant)}, {tag_label(tag)}",
                  fontsize=13)
    ax2.axhline(0.0, color='k', lw=1)
    ax2.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    ax2.set_ylabel(r'$\Delta S = S(s{=}0) - S$')
    ax2.plot([], [], color='gray', lw=2, label=r'simulation (solid, top); $s=0$ (dashed top, solid bottom)')
    ax2.plot([], [], color='gray', lw=1, ls=':', label='all stars like the global ionized gas')
    ax2.legend(loc='lower left', framealpha=0.85, fontsize=10)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def _large_scale(r, q):
    ls = r['k'] <= 3 * r['kF'] * 1.0001
    return float(np.average(q[ls], weights=r['nmodes'][ls]))


def write_table(results: list, config: dict, path: Path) -> None:
    """Budgets, Q and dS at pk.k_table, large-scale Q and validation numbers."""
    k_table = config['pk'].get('k_table', [1.0])
    L = ["# Halo-level stellar-to-ionized-gas transfer (compute_pk_stellar.py).",
         "# s = fraction of the selected stellar mass kept as stars (1 = simulation).",
         "# Q(s) = P_mm(s)/P_mm - 1 [%]; dS(s) = S(s) - S [absolute]; S = P_mm/P_DMO.",
         "# Configurations: <variant>/<tag>: fof = halo-finder membership, ap<x> = x R200m apertures;",
         "#   M<n> = FoF GroupMass >= 10^n Msun/h. 'global' / 'local1': all (cached) stars laid out",
         "#   like the box-wide ionized gas / moved within 1 Mpc/h (sphere) like the local ionized gas.",
         "# 'large-scale': mode-weighted mean over k <= 3 k_F."]
    for r in results:
        confs = sorted(r['cfg'], key=lambda c: (list(VARIANT_COLOUR).index(c[0])
                                                if c[0] in VARIANT_COLOUR else 9, float(c[1][1:])))
        L.append(f"\n## {r['label']}")
        if not confs:
            L.append("(no stellar-transfer spectra)")
            continue
        fstar = r['mstar_cache'] / r['mbaryon']
        L.append(f"# budget [Msun/h]: true stars {r['mstar_box']:.4e}; wind particles {r['mwind_box']:.3e} "
                 f"({r['mwind_box'] / (r['mstar_box'] + r['mwind_box']):.4f} of PartType4); cached Stars "
                 f"{r['mstar_cache']:.4e}; stellar share of baryons (cache) {fstar:.4f}")
        L.append(f"{'config':>9s} {'N_active':>9s} {'f_moved':>8s} {'f_kept':>8s} {'f_SF':>7s} "
                 f"{'f_p50':>7s} {'f_p90':>7s} {'f_p99':>7s} {'>1':>6s} {'f*(s=0)':>8s}")
        for c in confs:
            dg = r['cfg'][c]['diag']
            L.append(f"{c[0] + '/' + c[1]:>9s} {int(dg['n_haloes_active']):9d} "
                     f"{dg['mstar_moved'] / r['mstar_box']:8.4f} {dg['mstar_kept_noion'] / r['mstar_box']:8.1e} "
                     f"{dg['sf_frac_added']:7.4f} {dg['f_p50']:7.3f} {dg['f_p90']:7.3f} {dg['f_p99']:7.2f} "
                     f"{dg['frac_moved_f_gt1']:6.3f} {(r['mstar_cache'] - dg['mstar_moved']) / r['mbaryon']:8.4f}")
        L.append("#   f_moved/f_kept: moved / kept-in-place (no ionized gas) stellar mass over all true stars;")
        L.append("#   f_SF: share of the added mass placed on star-forming gas; f_p*: M*-weighted percentiles")
        L.append("#   of M*_h/M_ion,h; '>1': moved mass in haloes with M*_h > M_ion,h; f*(s=0): stellar share")
        L.append("#   of the baryons after the transfer (cached Stars minus moved).")
        names = [f"{c[0]}/{c[1]}" for c in confs] + ['global', 'local1']
        for s, qty, scale, fmt in ((0.0, 'Q', 100, '8.3f'), (0.0, 'dS', 1, '+8.4f'), (0.5, 'Q', 100, '8.3f')):
            L.append(f"# {qty}(s={s:g})")
            L.append(f"{'k':>11s} " + " ".join(f"{nm:>8s}" for nm in names))
            ctx = [r[f'{qty}_global'], r[f'{qty}_local']] if s == 0.0 else [None, None]
            for kt in k_table:
                if kt > r['k'].max():
                    continue
                i = int(np.argmin(np.abs(np.log(r['k'] / kt))))
                vals = [r['cfg'][c][qty][s][i] * scale for c in confs]
                vals += [np.nan if x is None else x[i] * scale for x in ctx]
                L.append(f"{r['k'][i]:11.2f} " + " ".join(f"{v:{fmt}}" if np.isfinite(v) else f"{'-':>8s}"
                                                          for v in vals))
            if qty == 'Q':
                vals = [_large_scale(r, r['cfg'][c]['Q'][s]) * 100 for c in confs]
                vals += [np.nan if x is None else _large_scale(r, x) * 100 for x in ctx]
                L.append(f"{'large-scale':>11s} " + " ".join(f"{v:8.4f}" if np.isfinite(v) else f"{'-':>8s}"
                                                             for v in vals))
        L.append("# validation")
        L.append(f"#   estimator vs Pylians P_total: {r['pylians_dev']:.1e}")
        for c in confs:
            d = r['cfg'][c]
            dg = d['diag']
            parts = [f"sum D/moved {dg['sum_D_rel']:+.1e}", f"per-halo {dg['max_halo_cons']:.1e}",
                     f"dP_SN(s=0)/P(k_Nyq) {d['dPsn_rel']:+.1e}"]
            if 'neg_star_mass' in dg:
                parts.append(f"neg. stars {dg['neg_star_mass'] / max(dg['mstar_moved'], 1e-30):+.1e} "
                             f"({int(dg['neg_star_cells'])} cells)")
            for s in (0.0, 0.5):
                if f'explicit_s{s:g}' in d:
                    parts.append(f"explicit s={s:g} {d[f'explicit_s{s:g}']:.1e}")
            if 'catalogue' in d:
                parts.append("catalogue M*_h median/99%/mass-weighted "
                             + "/".join(f"{x:.1e}" for x in d['catalogue']))
            L.append(f"#   {c[0]}/{c[1]}: " + "; ".join(parts))
    path.write_text("\n".join(L) + "\n")
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None)
    parser.add_argument('--kmax', type=float, default=10.0)
    parser.add_argument('--scale-variant', default='fof',
                        help="method variant of the stellar-scale figure (default fof)")
    parser.add_argument('--scale-tag', default=None,
                        help="mass-cut tag of the stellar-scale figure, e.g. M11 (default lowest cut)")
    parser.add_argument('--band-tag', default=None,
                        help="mass-cut tag of the S(k) band figures (default lowest cut)")
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()

    config = load_config(args.path2config)
    plot_cfg = config['plot']
    now = datetime.now()
    out_dir = Path(plot_cfg['fig_path']) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = plot_cfg['fig_name'] + (f"_{args.suffix}" if args.suffix else '')
    ext = plot_cfg.get('fig_type', 'pdf')
    lowest = mass_tag(min(float(m) for m in config['stellar']['halo_mass_min']))

    results = []
    for entry in select_sims(config, args.sims):
        if not any(spectra_path(entry, f"stellar_{v['name']}").exists() for v in variants_of(config)):
            print(f"no stellar-transfer spectra for {sim_label(entry)}; skipping")
            continue
        results.append(analyse(entry, config))
    if not results:
        raise SystemExit("no stellar-transfer spectra found")

    s_min = min(float(s) for s in config['stellar']['stellar_scales'])
    tag = args.scale_tag or lowest
    fig_scales(results, args.scale_variant, tag, args.kmax,
               out_dir / f"{stem}_stellar_scales_{args.scale_variant}_{tag}.{ext}")
    fig_methods(results, s_min, args.kmax, out_dir / f"{stem}_stellar_methods.{ext}")
    btag = args.band_tag or lowest
    for v in variants_of(config):
        fig_S_band(results, v['name'], btag, args.kmax, out_dir / f"{stem}_stellar_S_{v['name']}_{btag}.{ext}")
    write_table(results, config, out_dir / f"{stem}_stellar_table.txt")


if __name__ == '__main__':
    main()
