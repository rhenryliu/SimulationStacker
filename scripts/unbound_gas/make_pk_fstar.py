"""make_pk_fstar.py
================
Figures and table of the stellar-fraction bands (unbound gas paper, P(k)
section). For each method variant (halo-finder membership 'fof', apertures
'ap<x>' of x R200m) and halo mass cut: S(k) = P_mm/P_DMO (top) and the lensing
f_gas(theta) of lensing/beam_compensated_ratio_v2.py (bottom) for every
simulation, as

    the simulation itself                                   (solid),
    all selected stars converted to ionized gas, f* = 0     (dashed; the s = 0
        transfer of compute_pk_stellar.py / stack_stellar_maps.py),
    every selected region's star fraction raised to at least the method's
        target f*_T (fstar.targets)                         (dotted;
        compute_fstar.py / stack_fstar_maps.py),

with the band between f* = 0 and the floor filled, and the beam-compensated
data in the lower panel. The legend gives each simulation's own star fraction
f* = M*/(M* + M_wind + M_gas + M_BH) summed over the configuration's regions.
Simulations in plot.exclude_sims (matched like --sims) are left out of the
figures but kept in the table.

Outputs in <fig_path>/YYYY-MM/MM-DD/ (fig_name from the config):
  <fig_name>_fstar_S_<variant>_<tag>.<ext>   one per method variant and mass cut
  <fig_name>_fstar_table.txt                 star fractions, bookkeeping, S and
                                             f_gas at selected k and theta, and
                                             every validation number

Run from the scripts/ directory (light; login node is fine):
    python unbound_gas/make_pk_fstar.py -p configs/unbound_gas/pk_fstar_z05.yaml
    python unbound_gas/make_pk_fstar.py -p configs/unbound_gas/pk_fstar_z026.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import make_pk_stellar as mps  # style, colours, labels and the f* = 0 (s = 0) end
from compute_fstar import targets_of
from compute_pk_stellar import mass_tag, variants_of
from compute_stellar_maps import lensing_settings, lensing_sim, sample_settings, stack_path
from pk_common import load_config, select_sims, sim_label, spectra_path
from stack_fstar_maps import fstar_stack_path


def analyse(entry: dict, config: dict) -> dict:
    """The simulation, f* = 0 and floor quantities of one simulation.

    Returns:
        dict: make_pk_stellar.analyse's result (S0, the s = 0 end under
        'cfg', the s = 0 lensing ratio under 'lens') plus 'fs' (per
        (variant, tag): the floor's S(k), target and bookkeeping) and
        'lens_fs' (per (variant, tag): the floor's lensing ratio and checks).
    """
    r = mps.analyse(entry, config, lensing=True)
    if 'P_mm' not in r:
        raise FileNotFoundError(f"{sim_label(entry)}: no _Pk_stellar_ spectra for the f* = 0 end "
                                "(run compute_pk_stellar.py first)")
    r['fs'], r['lens_fs'] = {}, {}
    for v in variants_of(config):
        path = spectra_path(entry, f"fstar_{v['name']}")
        if not path.exists():
            continue
        with np.load(path) as f:
            if not np.allclose(f['k'], r['k'], rtol=1e-10):
                raise ValueError(f"k bins differ in {path}")
            P_mm = f['P_mm']
            pmm_dev = float(np.max(np.abs(P_mm / r['P_mm'] - 1)))
            for tag in f['tags']:
                tag = str(tag)
                P_hi = P_mm + 2 * f[f"P_mH__{tag}"] + f[f"P_HH__{tag}"]
                diag = {kk.split('__', 2)[2]: float(f[kk]) for kk in f.files
                        if kk.startswith(f"diag__{tag}__")}
                d = dict(S=P_hi / r['P_dmo'], Q=P_hi / P_mm - 1, target=float(f['target']),
                         diag=diag, pmm_dev=pmm_dev)
                for kk in (f"explicit_check__{tag}", f"check_mstar_vs_s0__{tag}"):
                    if kk in f.files:
                        d[kk.split('__')[0]] = float(f[kk])
                r['fs'][(v['name'], tag)] = d
    lens = lensing_settings(config)
    lsim, z = lensing_sim(lens, entry)
    p0, pf = stack_path(entry, lens['sample_name']), fstar_stack_path(entry, lens['sample_name'])
    if lsim is None or not (p0.exists() and pf.exists()):
        return r
    with np.load(p0) as a, np.load(pf) as b:
        if str(b['settings']) != sample_settings(lens['stack'], z):
            raise ValueError(f"{pf} was stacked with other settings than the config's lensing "
                             "block gives; rerun stack_fstar_maps.py")
        N, T, fac = a['N_mean'], a['T_mean'], float(a['factor'])
        checks = {kk: float(b[kk]) for kk in ('explicit_check_N', 'explicit_check_T')}
        for c in b['configs']:
            v, tag = str(c).split('__')
            S, G = b[f"S_mean__{c}"], b[f"G_mean__{c}"]
            pre = f"mapdiag__{v}__{tag}__"
            r['lens_fs'][(v, tag)] = dict(
                theta=b['theta_arcmin'], f=(N - G) / (T + S - G) * fac, checks=checks,
                mapdiag={kk[len(pre):]: float(b[kk]) for kk in b.files if kk.startswith(pre)})
    return r


def fig_fstar_band(results: list, variant: str, tag: str, kmax: float, path: Path,
                   z_label, lens_data, lens_xmax: float) -> None:
    """S(k) (top) and f_gas(theta) (bottom): simulation, f* = 0 and the floor."""
    have = [r for r in results if (variant, tag) in r['cfg'] and (variant, tag) in r['fs']]
    if not have:
        print(f"no {variant} {tag} floor spectra; skipping {path.name}")
        return
    T = have[0]['fs'][(variant, tag)]['target']
    colours = [mps.sim_colour(r) for r in have]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10.5),
                                   gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.28))
    for r, col in zip(have, colours):
        d0, dh = r['cfg'][(variant, tag)], r['fs'][(variant, tag)]
        sel = r['k'] <= kmax
        k = r['k'][sel]
        s0, sh = d0['S'][0.0][sel], dh['S'][sel]
        fsim = dh['diag']['fstar_sim']
        ax1.plot(k, r['S0'][sel], color=col, lw=2, label=rf"{r['label']} ($f_\star={fsim:.2f}$)")
        ax1.plot(k, s0, color=col, lw=1.5, ls='--')
        ax1.plot(k, sh, color=col, lw=1.5, ls=':')
        ax1.fill_between(k, s0, sh, color=col, alpha=0.2, lw=0)
    ax1.axhline(1.0, color='k', lw=1)
    mps._style_axis(ax1)
    mps._k_range(ax1, have, kmax)
    ax1.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    ax1.set_ylabel(r'$S(k) = P_{\rm mm}/P_{\rm DMO}$')
    ax1.legend(loc='lower left', framealpha=0.85)
    title = f"stellar fraction of halo baryons: {mps.variant_label(variant)}, {mps.tag_label(tag)}"
    ax1.set_title(f"{z_label}\n{title}" if z_label else title, fontsize=13)

    missing = []
    for r, col in zip(have, colours):
        L, Lf = r.get('lens'), r['lens_fs'].get((variant, tag))
        if L is None or Lf is None or (variant, tag) not in L['cfg']:
            missing.append(r['label'])
            continue
        th = L['theta']
        f1, f0, fh = L['cfg'][(variant, tag)]['f'][1.0], L['cfg'][(variant, tag)]['f'][0.0], Lf['f']
        ax2.plot(th, f1, color=col, lw=2, marker='o', ms=4)
        ax2.plot(th, f0, color=col, lw=1.5, ls='--')
        ax2.plot(th, fh, color=col, lw=1.5, ls=':')
        ax2.fill_between(th, f0, fh, color=col, alpha=0.2, lw=0)
    if missing:
        print(f"  {variant} {tag}: no lensing stacks for {', '.join(missing)} (bottom panel)")
    h = [plt.Line2D([], [], color='gray', lw=2, marker='o', ms=4),
         plt.Line2D([], [], color='gray', lw=1.5, ls='--'),
         plt.Line2D([], [], color='gray', lw=1.5, ls=':')]
    lab = ['simulation', r'$f_\star=0$', rf'$f_\star\geq{T:g}$']
    if lens_data is not None:
        h.append(ax2.errorbar(lens_data['theta'], lens_data['f'], yerr=lens_data['err'], fmt='s',
                              color='k', ms=6, capsize=2, zorder=5))
        lab.append(r'DESI $\times$ ACT $\times$ HSC (beam-corrected)')
    ax2.axhline(1.0, color='k', lw=1)
    ax2.grid(True, which='major', color='0.8', lw=0.8)
    ax2.set_axisbelow(True)
    ax2.set_xlim(0.0, lens_xmax)
    ax2.set_xlabel(r'$\theta\;[\mathrm{arcmin}]$')
    ax2.set_ylabel(r'$f_{\rm gas}(\theta)$')
    ax2.legend(h, lab, loc='best', framealpha=0.85, fontsize=10)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {path}")


def write_table(results: list, config: dict, path: Path) -> None:
    """Star fractions, floor bookkeeping, S and f_gas at selected k and theta, validation."""
    k_table = [1.0, 5.0]
    L = ["# Stellar-fraction bands (compute_fstar.py, stack_fstar_maps.py; f* = 0 end from the s = 0 transfer).",
         "# f* = M*/(M* + M_wind + M_gas + M_BH) over the configuration's regions (true stars; winds count",
         "#   as gas). Floor: every region below the target f*_T gains stars (laid out like its stars) from",
         "#   its ionized gas; regions above are unchanged. Configurations <variant>/<tag> as in the s tables.",
         "# f*_sim / f*_floor: the regions' star fraction in the simulation / after the floor;",
         "# raised, capped, no-star, no-ion: numbers of regions; added: stellar mass added / all true stars;",
         "# m_nostar, m_noion: baryonic mass of the unchanged regions below target / all regions' baryons."]
    if config['plot'].get('z_label'):
        L.insert(1, f"# Redshift: {config['plot']['z_label']}")
    for r in results:
        confs = sorted(r['fs'], key=lambda c: (list(mps.VARIANT_COLOUR).index(c[0])
                                               if c[0] in mps.VARIANT_COLOUR else 9, float(c[1][1:])))
        L.append(f"\n## {r['label']}")
        if not confs:
            L.append("(no floor spectra)")
            continue
        L.append(f"{'config':>9s} {'f*_T':>5s} {'f*_sim':>7s} {'f*_floor':>8s} {'raised':>8s} {'capped':>7s} "
                 f"{'no-star':>7s} {'no-ion':>7s} {'added':>7s} {'m_nostar':>9s} {'m_noion':>8s} {'wind/b':>7s} {'BH/b':>7s}")
        for c in confs:
            d = r['fs'][c]['diag']
            mb = d['mbaryon_regions']
            L.append(f"{c[0] + '/' + c[1]:>9s} {d['f_target']:5.2f} {d['fstar_sim']:7.4f} {d['fstar_after']:8.4f} "
                     f"{int(d['n_raised']):8d} {int(d['n_capped']):7d} {int(d['n_no_stars']):7d} "
                     f"{int(d['n_no_ion']):7d} {d['mstar_added'] / r['mstar_box']:7.3f} "
                     f"{d['mbaryon_no_stars'] / mb:9.1e} {d['mbaryon_no_ion'] / mb:8.1e} "
                     f"{d['mwind_regions'] / mb:7.4f} {d['mbh_regions'] / mb:7.1e}")
        L.append("# S(k): simulation / f*=0 / floor, and the floor's S - S_sim")
        L.append(f"{'config':>9s} " + " ".join(f"{'k=' + format(kt, 'g'):>30s}" for kt in k_table))
        for c in confs:
            vals = []
            for kt in k_table:
                i = int(np.argmin(np.abs(np.log(r['k'] / kt))))
                s1, s0, sh = r['S0'][i], r['cfg'][c]['S'][0.0][i], r['fs'][c]['S'][i]
                vals.append(f"{s1:.4f}/{s0:.4f}/{sh:.4f} {sh - s1:+.4f}")
            L.append(f"{c[0] + '/' + c[1]:>9s} " + " ".join(f"{v:>30s}" for v in vals))
        if r.get('lens') is not None and r['lens_fs']:
            L.append("# f_gas(theta): simulation / f*=0 / floor, and the floor's f - f_sim")
            th = r['lens']['theta']
            ii = [0, len(th) - 1]
            L.append(f"{'config':>9s} " + " ".join(f"{'theta=' + format(th[i], 'g') + chr(39):>30s}" for i in ii))
            for c in confs:
                if c not in r['lens_fs'] or c not in r['lens']['cfg']:
                    continue
                f1, f0, fh = (r['lens']['cfg'][c]['f'][1.0], r['lens']['cfg'][c]['f'][0.0],
                              r['lens_fs'][c]['f'])
                L.append(f"{c[0] + '/' + c[1]:>9s} " + " ".join(
                    f"{f'{f1[i]:.4f}/{f0[i]:.4f}/{fh[i]:.4f} {fh[i] - f1[i]:+.4f}':>30s}" for i in ii))
        L.append("# validation")
        first = r['fs'][confs[0]]
        L.append(f"#   P_mm of the floor files vs the s = 0 files: {first['pmm_dev']:.1e}")
        for c in confs:
            d, fs = r['fs'][c]['diag'], r['fs'][c]
            parts = [f"sum H/added {d['sum_H_rel']:+.1e}",
                     f"per-halo {max(d['max_halo_cons_stars'], d['max_halo_cons_gas']):.1e}",
                     f"removed/M_ion max {d['max_removed_over_mion']:.3f}",
                     f"f*_after - f*_T (raised, uncapped) min {d['min_fstar_after_minus_target']:+.1e}"]
            if 'check_mstar_vs_s0' in fs:
                parts.append(f"regions vs s=0 run {fs['check_mstar_vs_s0']:+.1e}")
            if 'explicit_check' in fs:
                parts.append(f"explicit field {fs['explicit_check']:.1e}")
            md = r['lens_fs'].get(c, {}).get('mapdiag', {})
            if md:
                parts.append(f"2D sums {md['sum_stars_rel']:+.1e}/{md['sum_gas_rel']:+.1e}")
            L.append(f"#   {c[0]}/{c[1]}: " + "; ".join(parts))
        lc = next(iter(r['lens_fs'].values()), None)
        if lc is not None:
            L.append(f"#   lensing explicit maps vs linear combination: N {lc['checks']['explicit_check_N']:.1e}, "
                     f"T {lc['checks']['explicit_check_T']:.1e}")
    path.write_text("\n".join(L) + "\n")
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None)
    parser.add_argument('--kmax', type=float, default=5.0, help="largest k plotted [h/Mpc] (default 5)")
    parser.add_argument('--band-tag', default=None,
                        help="mass-cut tag of the band figures, e.g. M11 (default: every cut)")
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()

    config = load_config(args.path2config)
    for key in ('fstar', 'lensing'):
        if key not in config:
            raise SystemExit(f"{args.path2config} has no '{key}' block")
    plot_cfg = config['plot']
    now = datetime.now()
    out_dir = Path(plot_cfg['fig_path']) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = plot_cfg['fig_name'] + (f"_{args.suffix}" if args.suffix else '')
    ext = plot_cfg.get('fig_type', 'pdf')
    lens = lensing_settings(config)
    lens_xmax = lens['stack']['max_radius'] * lens['stack']['rad_distance'] + 0.5
    lens_data = None
    if lens['data'] and Path(lens['data']).exists():
        with np.load(lens['data']) as dd:
            lens_data = dict(theta=dd['theta_arcmin'], f=dd['R_compensated'], err=dd['sigma_compensated'])
    else:
        print(f"no beam-compensated data at {lens['data']}; the lensing panel has no data points")
    targets_of(config)  # validates the fstar block

    excluded = set(plot_cfg.get('exclude_sims') or [])
    results, fig_results = [], []
    for entry in select_sims(config, args.sims):
        if not any(spectra_path(entry, f"fstar_{v['name']}").exists() for v in variants_of(config)):
            print(f"no floor spectra for {sim_label(entry)}; skipping")
            continue
        results.append(analyse(entry, config))
        if {sim_label(entry), entry['name'], entry.get('feedback')} & excluded:
            print(f"{sim_label(entry)}: left out of the figures (plot.exclude_sims), kept in the table")
        else:
            fig_results.append(results[-1])
    if not results:
        raise SystemExit("no floor spectra found")
    if not fig_results:
        raise SystemExit("every simulation is in plot.exclude_sims")

    all_tags = [mass_tag(m) for m in sorted(float(m) for m in config['stellar']['halo_mass_min'])]
    for btag in ([args.band_tag] if args.band_tag else all_tags):
        for v in variants_of(config):
            fig_fstar_band(fig_results, v['name'], btag, args.kmax,
                           out_dir / f"{stem}_fstar_S_{v['name']}_{btag}.{ext}",
                           plot_cfg.get('z_label'), lens_data, lens_xmax)
    write_table(results, config, out_dir / f"{stem}_fstar_table.txt")


if __name__ == '__main__':
    main()
