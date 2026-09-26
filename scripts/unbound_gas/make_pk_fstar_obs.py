"""make_pk_fstar_obs.py
=====================
Figures and table of the observation-based stellar bands (unbound gas paper,
P(k) section; compute_fstar_obs.py). For each option -- the aggregate stellar
fraction of the halo baryons f* = sum M* / sum (M* + M_wind + M_gas + M_BH)
('fstar', option a) or sum M* / sum M200m ('mstar_m200m', option c), over the
1 R200m regions of the haloes above the cut -- S(k) = P_mm/P_DMO (top) and the
lensing f_gas(theta) of lensing/beam_compensated_ratio_v2.py (bottom) for
every simulation, as

    the simulation itself                                   (solid),
    its stars rescaled so the aggregate hits the low target  (dashed),
    ... and the high target                                  (dotted),

with the band between the two ends filled and the beam-compensated data in
the lower panel. The legend gives each simulation's own f* and M*/M200m and
the stellar scales s of the two ends. An end without a capped region is the
s = 0 transfer at scale s (P_mm + 2(1-s) P_mD + (1-s)^2 P_DD, f_of_scale on
the s = 0 lensing stacks); a capped end uses its own field (P_mm + 2 P_mH +
P_HH, stack_fstar_obs_maps.py's stacks).

Outputs in <fig_path>/YYYY-MM/MM-DD/ (fig_name from the config):
  <fig_name>_<option>_S_<variant>_<tag>.<ext>   one per option
  <fig_name>_table.txt                          aggregates, s per end, capping,
                                                S and f_gas at selected k and
                                                theta, every validation number

With --preview, simulations without a compute_fstar_obs.py file are solved
from the s = 0 and f* files and the catalogue, and only their uncapped ends
are drawn (no band where an end is missing); nothing is saved but the figures
(suffix 'preview' unless --suffix is given).

Run from the scripts/ directory (light; login node is fine):
    python unbound_gas/make_pk_fstar_obs.py -p configs/unbound_gas/pk_fstar_obs_z05.yaml
    python unbound_gas/make_pk_fstar_obs.py -p configs/unbound_gas/pk_fstar_obs_z026.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import compute_fstar_obs as cfo
import make_pk_local as mpl  # the SIMBA label
import make_pk_stellar as mps  # style, colours and labels
from compute_stellar_maps import (f_of_scale, lensing_settings, lensing_sim, sample_settings,
                                  stack_path)
from pk_common import load_config, select_sims, sim_label, spectra_path
from stack_fstar_obs_maps import obs_stack_path

OPTION_LABEL = {'fstar': r'f_\star', 'mstar_m200m': r'M_\star/M_{\rm 200m}'}
OPTION_TITLE = {'fstar': 'aggregate stellar fraction of the halo baryons',
                'mstar_m200m': r'aggregate $M_\star/M_{\rm 200m}$'}


def _solve_preview(entry: dict, obs: dict, z: float) -> dict:
    """compute_fstar_obs's ends from the files alone (no pass; --preview).

    ``z`` is the lensing config's redshift of the simulation, as in
    compute_fstar_obs.run_sim.
    """
    from stacker import SimulationStacker
    st = SimulationStacker(entry['name'], entry['snapshot'], simType=entry['sim_type'],
                           feedback=entry.get('feedback'), z=z)
    sol = cfo.solve_from_files(entry, obs, st.loadHalos())
    out = dict(fstar_sim=sol['current']['fstar'], mstar_m200m_sim=sol['current']['mstar_m200m'],
               mbaryon_regions=sol['ref']['mbaryon_regions'], m200m_sum=sol['m200m'],
               n_haloes=sol['n_haloes'], preview=True, checks={}, ends={})
    for kk, e in sol['ends'].items():
        out['ends'][kk] = dict(target=e['target'], s=e['s'], source=e['source'],
                               n_capped=int(e['capped'].sum()), capped_frac=e['capped_frac'])
    return out


def _load_obs(path: Path) -> dict:
    """compute_fstar_obs.py's per-simulation file."""
    with np.load(path) as f:
        out = dict(fstar_sim=float(f['fstar_sim']), mstar_m200m_sim=float(f['mstar_m200m_sim']),
                   mbaryon_regions=float(f['mbaryon_regions']), m200m_sum=float(f['m200m_sum']),
                   n_haloes=int(f['n_haloes']), s_uncapped_max=float(f['s_uncapped_max']),
                   preview=False, ends={},
                   checks={kk[len('check_'):]: (bool(f[kk]) if f[kk].dtype == bool else float(f[kk]))
                           for kk in f.files if kk.startswith('check_')})
        P_pass = f['P_mm_pass'] if 'P_mm_pass' in f.files else None
        for kk in f['ends']:
            kk = str(kk)
            e = dict(target=float(f[f"target__{kk}"]), s=float(f[f"s__{kk}"]),
                     source=str(f[f"source__{kk}"]), n_capped=int(f[f"n_capped__{kk}"]),
                     capped_frac=float(f[f"capped_frac__{kk}"]))
            if e['source'] == 'pass':
                e['P'] = P_pass + 2 * f[f"P_mH__{kk}"] + f[f"P_HH__{kk}"]
                # the end key itself holds '__': strip the whole prefix
                pre, mpre = f"diag__{kk}__", f"mapdiag__{kk}__"
                e['diag'] = {q[len(pre):]: float(f[q]) for q in f.files if q.startswith(pre)}
                e['mapdiag'] = {q[len(mpre):]: float(f[q]) for q in f.files if q.startswith(mpre)}
                for q in (f"explicit_check__{kk}", f"check_target__{kk}"):
                    if q in f.files:
                        e[q.split('__')[0]] = float(f[q])
            out['ends'][kk] = e
    return out


def analyse(entry: dict, config: dict, preview: bool):
    """The simulation and the four ends of one simulation, or None without results.

    Returns:
        dict or None: label/colour keys, 'k', 'S0' (simulation), 'obs'
        (aggregates, checks), 'ends' ({key: s, source, S (or None), f (or
        None), ...}), 'lens' (theta, simulation f, checks) or None.
    """
    obs = cfo.obs_settings(config)
    name, tag = obs['variant']['name'], obs['tag']
    lens = lensing_settings(config)
    lsim, z = lensing_sim(lens, entry)
    rpath = cfo.obs_path(entry, name)
    if rpath.exists():
        o = _load_obs(rpath)
    elif preview:
        if lsim is None:
            raise ValueError(f"{sim_label(entry)} is not in the lensing config {lens['config_path']}")
        print(f"{sim_label(entry)}: no {rpath.name}; preview from the s = 0 and f* files")
        o = _solve_preview(entry, obs, z)
    else:
        print(f"{sim_label(entry)}: no {rpath.name}; skipping (run compute_fstar_obs.py, or --preview)")
        return None

    dmo = np.load(spectra_path(entry, 'dmo'))
    with np.load(spectra_path(entry, f"stellar_{name}")) as f:
        k, P_mm = f['k'], f['P_mm']
        P_mD, P_DD = f[f"P_mD__{tag}"], f[f"P_DD__{tag}"]
    if not np.allclose(dmo['k'], k, rtol=1e-10):
        raise ValueError(f"k bins differ between the DMO and s = 0 spectra of {sim_label(entry)}")
    P_dmo = dmo['P_dmo']
    r = dict(label=mpl.label_of(entry), sim_type=entry['sim_type'], name=entry['name'],
             feedback=entry.get('feedback'), k=k, S0=P_mm / P_dmo, obs=o, ends={}, lens=None)
    for kk, e in o['ends'].items():
        d = dict(e)
        if e['source'] == 's0':
            d['S'] = cfo.p_at_scale(P_mm, P_mD, P_DD, e['s']) / P_dmo
        elif 'P' in e:
            d['S'] = e['P'] / P_dmo
        else:
            d['S'] = None  # preview: capped end not computed yet
        r['ends'][kk] = d

    p0 = stack_path(entry, lens['sample_name'])
    if lsim is None or not p0.exists():
        print(f"  {sim_label(entry)}: no lensing stacks ({p0.name})")
        return r
    with np.load(p0) as a:
        if str(a['settings']) != sample_settings(lens['stack'], z):
            raise ValueError(f"{p0} was stacked with other settings than the config's lensing block")
        N, T, fac = a['N_mean'], a['T_mean'], float(a['factor'])
        c = f"{name}__{tag}"
        A, B = a[f"A_mean__{c}"], a[f"B_mean__{c}"]
        r['lens'] = dict(theta=a['theta_arcmin'], f=f_of_scale(N, T, A, B, 1.0, fac), factor=fac,
                         checks={})
    pf = obs_stack_path(entry, lens['sample_name'])
    stk = None
    if pf.exists():
        stk = np.load(pf)
        if str(stk['settings']) != sample_settings(lens['stack'], z):
            raise ValueError(f"{pf} was stacked with other settings than the config's lensing block")
        r['lens']['checks'] = {q: float(stk[q]) for q in ('explicit_check_N', 'explicit_check_T')}
    for kk, d in r['ends'].items():
        if d['source'] == 's0':
            d['f'] = f_of_scale(N, T, A, B, d['s'], fac)
        elif stk is not None and f"S_mean__{kk}" in stk.files:
            if not np.isclose(float(stk[f"s__{kk}"]), d['s'], rtol=1e-12):
                raise ValueError(f"{pf}: {kk} was stacked for s = {float(stk[f's__{kk}'])}, "
                                 f"the results file has {d['s']}; rerun stack_fstar_obs_maps.py")
            S, G = stk[f"S_mean__{kk}"], stk[f"G_mean__{kk}"]
            d['f'] = (N - G) / (T + S - G) * fac
        else:
            d['f'] = None
    if stk is not None:
        stk.close()
    return r


def fig_obs_band(results: list, option: str, variant: str, tag: str, kmax: float, path: Path,
                 z_label, lens_data, lens_xmax: float) -> None:
    """S(k) (top) and f_gas(theta) (bottom): simulation and the two ends of one option."""
    lo, hi = f"{option}__low", f"{option}__high"
    have = [r for r in results if lo in r['ends']]
    if not have:
        print(f"no results; skipping {path.name}")
        return
    T_lo, T_hi = have[0]['ends'][lo]['target'], have[0]['ends'][hi]['target']
    colours = [mps.sim_colour(r) for r in have]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10.5),
                                   gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.28))
    for r, col in zip(have, colours):
        sel = r['k'] <= kmax
        k = r['k'][sel]
        o, el, eh = r['obs'], r['ends'][lo], r['ends'][hi]
        lab = (rf"{r['label']}: $f_\star={o['fstar_sim']:.2f}$, "
               rf"$M_\star/M_{{\rm 200m}}={o['mstar_m200m_sim']:.3f}$; "
               rf"$s={el['s']:.2f}$ / ${eh['s']:.2f}$")
        ax1.plot(k, r['S0'][sel], color=col, lw=2, label=lab)
        for e, ls in ((el, '--'), (eh, ':')):
            if e['S'] is not None:
                ax1.plot(k, e['S'][sel], color=col, lw=1.5, ls=ls)
        if el['S'] is not None and eh['S'] is not None:
            ax1.fill_between(k, el['S'][sel], eh['S'][sel], color=col, alpha=0.2, lw=0)
    ax1.axhline(1.0, color='k', lw=1)
    mps._style_axis(ax1)
    mps._k_range(ax1, have, kmax)
    ax1.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
    ax1.set_ylabel(r'$S(k) = P_{\rm mm}/P_{\rm DMO}$')
    ax1.legend(loc='lower left', framealpha=0.85, fontsize=9)
    title = f"{OPTION_TITLE[option]}: {mps.variant_label(variant)}, {mps.tag_label(tag)}"
    ax1.set_title(f"{z_label}\n{title}" if z_label else title, fontsize=13)

    missing = []
    for r, col in zip(have, colours):
        L, el, eh = r.get('lens'), r['ends'][lo], r['ends'][hi]
        if L is None:
            missing.append(r['label'])
            continue
        th = L['theta']
        ax2.plot(th, L['f'], color=col, lw=2, marker='o', ms=4)
        for e, ls in ((el, '--'), (eh, ':')):
            if e.get('f') is not None:
                ax2.plot(th, e['f'], color=col, lw=1.5, ls=ls)
        if el.get('f') is not None and eh.get('f') is not None:
            ax2.fill_between(th, el['f'], eh['f'], color=col, alpha=0.2, lw=0)
    if missing:
        print(f"  {option}: no lensing stacks for {', '.join(missing)} (bottom panel)")
    sym = OPTION_LABEL[option]
    h = [plt.Line2D([], [], color='gray', lw=2, marker='o', ms=4),
         plt.Line2D([], [], color='gray', lw=1.5, ls='--'),
         plt.Line2D([], [], color='gray', lw=1.5, ls=':')]
    lab = ['simulation', rf'${sym}={T_lo:g}$', rf'${sym}={T_hi:g}$']
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


def _fmt(x, spec):
    return 'n/a' if x is None else format(x, spec)


def write_table(results: list, config: dict, path: Path) -> None:
    """Aggregates, s and capping per end, S and f_gas at selected k and theta, validation."""
    obs = cfo.obs_settings(config)
    k_table = [1.0, 5.0]
    L = ["# Observation-based stellar bands (compute_fstar_obs.py, stack_fstar_obs_maps.py).",
         f"# Regions: {obs['variant']['name']} (x = {obs['variant']['x']:g} R200m, mass priority), "
         f"FoF mass >= {obs['mass_min']:.0e} Msun/h.",
         "# Every region's true stars x s (one s per simulation and end), exchanged with its ionized gas;",
         "#   s > 1: a region gains at most all its ionized gas (capped), s solved so the aggregate hits",
         "#   the target. f* = sum M* / sum (M* + M_wind + M_gas + M_BH); M*/M200m = sum M* / sum M200m",
         "#   (catalogue SO mass). b/cosmic = sum M_b / (Omega_b/Omega_m sum M200m).",
         "# source s0: exact from the s = 0 transfer at scale s; pass: own field (capped regions).",
         "# capped: regions converting all their ionized gas (share of the changing stars)."]
    if config['plot'].get('z_label'):
        L.insert(1, f"# Redshift: {config['plot']['z_label']}")
    keys = cfo.end_keys()
    for r in results:
        o = r['obs']
        L.append(f"\n## {r['label']}" + ('  [PREVIEW: capped ends not computed]' if o['preview'] else ''))
        fac = r['lens']['factor'] if r.get('lens') else None
        bc = (o['mbaryon_regions'] * fac / o['m200m_sum']) if fac else None
        L.append(f"haloes {o['n_haloes']:,}; f* = {o['fstar_sim']:.4f}; M*/M200m = {o['mstar_m200m_sim']:.4f}; "
                 f"b/cosmic = {_fmt(bc, '.3f')}; uncapped up to s = {_fmt(o.get('s_uncapped_max'), '.3f')}")
        L.append(f"{'end':>20s} {'target':>7s} {'s':>8s} {'source':>6s} {'capped':>8s} {'share':>7s}")
        for kk in keys:
            e = r['ends'][kk]
            L.append(f"{kk:>20s} {e['target']:7.3f} {e['s']:8.4f} {e['source']:>6s} "
                     f"{e['n_capped']:8,d} {e['capped_frac']:7.4f}")
        L.append("# S(k): simulation, each end, and the end's S - S_sim")
        L.append(f"{'end':>20s} " + " ".join(f"{'k=' + format(kt, 'g'):>24s}" for kt in k_table))
        for kk in keys:
            e, vals = r['ends'][kk], []
            for kt in k_table:
                i = int(np.argmin(np.abs(np.log(r['k'] / kt))))
                if e['S'] is None:
                    vals.append('n/a')
                else:
                    vals.append(f"{r['S0'][i]:.4f}/{e['S'][i]:.4f} {e['S'][i] - r['S0'][i]:+.4f}")
            L.append(f"{kk:>20s} " + " ".join(f"{v:>24s}" for v in vals))
        if r.get('lens') is not None:
            th = r['lens']['theta']
            ii = [0, len(th) - 1]
            L.append("# f_gas(theta): simulation, each end, and the end's f - f_sim")
            L.append(f"{'end':>20s} " + " ".join(f"{'theta=' + format(th[i], 'g') + chr(39):>24s}" for i in ii))
            f1 = r['lens']['f']
            for kk in keys:
                fe = r['ends'][kk].get('f')
                L.append(f"{kk:>20s} " + " ".join(
                    f"{'n/a' if fe is None else f'{f1[i]:.4f}/{fe[i]:.4f} {fe[i] - f1[i]:+.4f}':>24s}"
                    for i in ii))
        L.append("# validation")
        if o['checks']:
            L.append("#   " + "; ".join(f"{kk} {v}" if isinstance(v, bool) else f"{kk} {v:+.1e}"
                                     for kk, v in sorted(o['checks'].items())))
        for kk in keys:
            e = r['ends'][kk]
            if e['source'] != 'pass' or 'diag' not in e:
                continue
            d, md = e['diag'], e.get('mapdiag', {})
            parts = [f"aggregate/target-1 {e.get('check_target', np.nan):+.1e}",
                     f"sum H/added {d['sum_H_rel']:+.1e}",
                     f"per-halo {max(d['max_halo_cons_stars'], d['max_halo_cons_gas']):.1e}",
                     f"removed/M_ion max {d['max_removed_over_mion']:.4f}"]
            if 'explicit_check' in e:
                parts.append(f"explicit field {e['explicit_check']:.1e}")
            if md:
                parts.append(f"2D sums {md['sum_stars_rel']:+.1e}/{md['sum_gas_rel']:+.1e}; "
                             f"min maps {md['min_stars_added']:.1e}/{md['min_gas_removed']:.1e}")
            L.append(f"#   {kk}: " + "; ".join(parts))
        lc = (r.get('lens') or {}).get('checks')
        if lc:
            L.append(f"#   lensing explicit maps vs linear combination: N {lc['explicit_check_N']:.1e}, "
                     f"T {lc['explicit_check_T']:.1e}")
    path.write_text("\n".join(L) + "\n")
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None)
    parser.add_argument('--kmax', type=float, default=5.0, help="largest k plotted [h/Mpc] (default 5)")
    parser.add_argument('--preview', action='store_true',
                        help="solve simulations without results from the files; draw the uncapped ends")
    parser.add_argument('--suffix', default=None)
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
    suffix = args.suffix if args.suffix is not None else ('preview' if args.preview else '')
    stem = plot_cfg['fig_name'] + (f"_{suffix}" if suffix else '')
    ext = plot_cfg.get('fig_type', 'pdf')
    lens = lensing_settings(config)
    lens_xmax = lens['stack']['max_radius'] * lens['stack']['rad_distance'] + 0.5
    lens_data = None
    if lens['data'] and Path(lens['data']).exists():
        with np.load(lens['data']) as dd:
            lens_data = dict(theta=dd['theta_arcmin'], f=dd['R_compensated'], err=dd['sigma_compensated'])
    else:
        print(f"no beam-compensated data at {lens['data']}; the lensing panel has no data points")

    results = [r for r in (analyse(e, config, args.preview) for e in select_sims(config, args.sims))
               if r is not None]
    if not results:
        raise SystemExit("no results found")
    for opt in cfo.OPTIONS:
        fig_obs_band(results, opt, obs['variant']['name'], obs['tag'], args.kmax,
                     out_dir / f"{stem}_{opt}_S_{obs['variant']['name']}_{obs['tag']}.{ext}",
                     plot_cfg.get('z_label'), lens_data, lens_xmax)
    if not args.preview:
        write_table(results, config, out_dir / f"{stem}_table.txt")


if __name__ == '__main__':
    main()
