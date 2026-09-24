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

The context models move all stars: 'global' = like the box-wide ionized gas
(make_pk_alpha's stars-only variant; in the table only, not in the figures),
'local R=1' = transported within a 1 Mpc/h sphere to follow the local ionized
gas (make_pk_local's 'stars' set; figures and table). Both use the cached
Stars field, which for TNG/Illustris includes wind-phase particles (~8% of
Illustris-1's; excluded from the halo-level transfer).

Outputs in <fig_path>/YYYY-MM/MM-DD/ (fig_name from the config):
  <fig_name>_stellar_scales_<variant>_<tag>.<ext>
                                  Q(k) for every s of stellar.stellar_scales,
                                  one panel per simulation, one figure per
                                  mass cut (--scale-variant, default fof;
                                  --scale-tag, default every cut)
  <fig_name>_stellar_methods.<ext>
                                  Q(k) at s = 0 for every method and mass cut
  <fig_name>_stellar_S_<variant>_<tag>.<ext>
                                  S(k) from s = 1 to s = 0 for all simulations,
                                  one per method variant and mass cut
                                  (--band-tag, default every cut); below it the
                                  lensing f_gas(theta) under the same transfer
                                  (--bottom lensing, the default with a
                                  `lensing` config block) or dS on a shared k
                                  axis (--bottom dS, the earlier figure)
  <fig_name>_stellar_table.txt    budgets, Q and dS at pk.k_table, large-scale
                                  Q, and every validation number
  <fig_name>_stellar_lensing_table.txt
                                  f_gas(theta) at s = 1 and s = 0 and the
                                  lensing validation numbers (--bottom lensing)

The lensing panel is the beam-free simulation curve of
lensing/beam_compensated_ratio_v2.py, f = <DSigma[ionized_gas]>/<DSigma[total]>
* Omega_m/Omega_b on its fixed halo sample, with the maps changed by the same
box-wide transfer as the top panel (t = 1 - s):
f(s) = [N + t A] / [T + t (A - B)] * Omega_m/Omega_b (stacks from
stack_stellar_maps.py; the beam-compensated data points of the lensing script
as black squares).

Figures stop at k = 5 h/Mpc (--kmax), have grid lines, and colour the
simulations as the lensing P(k) suppression figure (lensing/plot_pk_suppression.py);
in the per-simulation panel figures colour encodes s or the method instead.

Without a components spectra file (e.g. z ~ 0.26, configs/unbound_gas/pk_stellar_z026.yaml)
the global context curve is omitted and the baryon budget comes from the
particle pass; without local-model spectra the local curve is omitted. An
optional plot.z_label titles the figures; simulations in plot.exclude_sims
(matched like --sims) are left out of the figures but kept in the tables.

Run from the scripts/ directory (light; login node is fine):
    python unbound_gas/make_pk_stellar.py -p configs/unbound_gas/pk_components_z05.yaml
    python unbound_gas/make_pk_stellar.py -p configs/unbound_gas/pk_stellar_z026.yaml
    # the earlier dS bottom panel (e.g. into a separate file name)
    python unbound_gas/make_pk_stellar.py -p ... --bottom dS --suffix dS
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
from compute_stellar_maps import (f_of_scale, lensing_settings, lensing_sim, sample_settings,
                                  stack_path)
from pk_common import load_config, select_sims, sim_label, spectra_path

import halo_transfer as ht

VARIANT_COLOUR = {'fof': 'C0', 'ap1': 'C1', 'ap2': 'C2'}
TAG_STYLE = ['-', '--', ':']

# Simulation colours of the lensing P(k) suppression figure
# (lensing/plot_pk_suppression.py, after the lensing paper figures): suite group
# i takes colourmap ['plasma', 'twilight'][i] sampled on linspace(0.2, 0.85, n)
# over the group's simulations (SIMBA: m100n1024; IllustrisTNG: TNG300-1,
# Illustris-1), and the FLAMINGO variants have fixed colours. Copied rather
# than imported: importing that script would override this script's rcParams.
_SUITE_COLOURS = {
    'SIMBA': ('plasma', ['m100n1024']),
    'IllustrisTNG': ('twilight', ['TNG300-1', 'Illustris-1']),
}
_FLAMINGO_COLOURS = {
    'L1_m9':           '#B30000',  # dark red (fiducial)
    'fgas-8sigma':     '#FF7F0E',  # orange
    'Jet_fgas-4sigma': '#C71585',  # magenta
}


def sim_colour(r: dict):
    """Colour of a simulation, as in the lensing P(k) suppression figure."""
    if r['sim_type'] == 'FLAMINGO':
        return _FLAMINGO_COLOURS.get(r['feedback'], 'k')
    cmap_name, names = _SUITE_COLOURS[r['sim_type']]
    cols = matplotlib.colormaps[cmap_name](np.linspace(0.2, 0.85, len(names)))  # type: ignore
    return cols[names.index(r['name'])] if r['name'] in names else 'k'


def _style_axis(ax) -> None:
    """Log k axis with the lensing figure's grid lines."""
    ax.set_xscale('log')
    ax.grid(True, which='major', color='0.8', lw=0.8)
    ax.grid(True, which='minor', axis='x', color='0.9', lw=0.5)
    ax.set_axisbelow(True)


def _k_range(ax, results: list, kmax: float) -> None:
    """Shared k range: from just below the smallest k plotted to exactly kmax.

    Set once after all panels are drawn (a limit set earlier would freeze the
    shared axis before the larger boxes' low-k points are plotted).
    """
    ax.set_xlim(min(float(r['k'][0]) for r in results) / 1.2, kmax)


def variant_label(name: str) -> str:
    """Legend label of a method variant."""
    if name == 'fof':
        return 'halo-finder membership'
    return rf"aperture $r<{name[2:]}\,R_{{\rm 200m}}$"


def tag_label(tag: str) -> str:
    """Legend label of a mass-cut tag, e.g. 'M11'."""
    return rf"$M_{{\rm FoF}}\geq10^{{{tag[1:]}}}\,h^{{-1}}M_\odot$"


def lensing_result(entry: dict, config: dict, scales: list):
    """Lensing f_gas(theta) for every configuration and s (stack_stellar_maps.py output).

    Returns:
        dict or None: 'theta', 'n_haloes', 'z', 'checks' and per configuration
        (variant, tag) the ratio f[s] and the map bookkeeping; None if the
        simulation is not in the lensing config or has no stacks.

    Raises:
        ValueError: If the stacks were made with other settings than the
            config's ``lensing`` block now gives (rerun stack_stellar_maps.py).
    """
    lens = lensing_settings(config)
    lsim, z = lensing_sim(lens, entry)
    if lsim is None:
        return None
    path = stack_path(entry, lens['sample_name'])
    if not path.exists():
        print(f"no lensing stacks for {sim_label(entry)} ({path.name})")
        return None
    with np.load(path) as f:
        if str(f['settings']) != sample_settings(lens['stack'], z):
            raise ValueError(f"{path} was stacked with other settings than the config's "
                             "lensing block gives; rerun stack_stellar_maps.py")
        N, T, factor = f['N_mean'], f['T_mean'], float(f['factor'])
        out = dict(theta=f['theta_arcmin'], n_haloes=int(f['n_haloes']), z=float(f['z']),
                   sample_name=str(f['sample_name']), cfg={},
                   checks={k: (str(f[k]) if k == 'explicit_config' else float(f[k]))
                           for k in ('check_stackmap_N', 'check_stackmap_T', 'explicit_config',
                                     'explicit_check_N', 'explicit_check_T') if k in f.files})
        for c in f['configs']:
            v, tag = str(c).split('__')
            A, B = f[f"A_mean__{c}"], f[f"B_mean__{c}"]
            pre = f"mapdiag__{v}__{tag}__"
            out['cfg'][(v, tag)] = dict(
                f={s: f_of_scale(N, T, A, B, s, factor) for s in scales},
                mapdiag={k[len(pre):]: float(f[k]) for k in f.files if k.startswith(pre)})
    return out


def analyse(entry: dict, config: dict, lensing: bool = False) -> dict:
    """Stellar-transfer quantities (and context models) for one simulation.

    With ``lensing``, also the lensing f_gas(theta) (``lensing_result``) as 'lens'.
    """
    scales = [float(s) for s in config['stellar']['stellar_scales']]
    dmo = np.load(spectra_path(entry, 'dmo'))
    comp_path = spectra_path(entry, 'components')
    # Without a components file (e.g. z ~ 0.26) there is no global context curve,
    # and the baryon budget comes from the particle pass (see below).
    comp = np.load(comp_path) if comp_path.exists() else None
    ref = comp if comp is not None else dmo
    k = ref['k']
    if not np.allclose(dmo['k'], k, rtol=1e-10):
        raise ValueError(f"k bins differ between the DMO and components spectra of {sim_label(entry)}")
    P_dmo = dmo['P_dmo']
    res = dict(label=mpl.label_of(entry), sim_type=entry['sim_type'], name=entry['name'],
               feedback=entry.get('feedback'), k=k, P_dmo=P_dmo,
               nmodes=ref['Nmodes'], kF=2 * np.pi / float(ref['box_mpc']), scales=scales,
               mstar_cache=np.nan, mbaryon=np.nan, budget_from_particles=comp is None,
               Q_global=None, dS_global=None, cfg={})
    if comp is not None:
        n = int(comp['n_pixels'])
        means = comp['means']
        w = means / means.sum()
        P0g = mpa.p_of(w, comp['P'])
        Pg = mpa.p_of(mpa.weights_move(w, [mpa.I_ST], 'ionized'), comp['P'])
        res.update(mstar_cache=float(means[mpa.I_ST]) * n ** 3,
                   mbaryon=float(means[1:].sum()) * n ** 3,
                   Q_global=Pg / P0g - 1.0, dS_global=(Pg - P0g) / P_dmo)
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
        if res['budget_from_particles']:
            # PartType4 mass: the Stars cache when it was read, else stars + winds
            # from the particle pass; baryons = PartType4 + gas (BH not included).
            mc = float(f['mstar_cache'])
            res['mstar_cache'] = mc if np.isfinite(mc) else res['mstar_box'] + res['mwind_box']
            res['mbaryon'] = res['mstar_box'] + res['mwind_box'] + float(f['mgas_box'])
    res['lens'] = lensing_result(entry, config, scales) if lensing else None
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


def _finish_grid(fig, axes, n, ylabel, path, z_label=None):
    if z_label:
        fig.suptitle(z_label, fontsize=16)
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
    # Only the local model is drawn; the global one is in the table only.
    if r[f'{which}_local'] is not None:
        ax.plot(r['k'][sel], 100 * r[f'{which}_local'][sel] if which == 'Q' else r[f'{which}_local'][sel],
                color='gray', ls='-.', lw=1.2)


def _context_legend(results):
    """Legend handles and labels of the context curves present in any result."""
    h, lab = [], []
    if any(r['Q_local'] is not None for r in results):
        h.append(plt.Line2D([], [], color='gray', ls='-.'))
        lab.append(r'all stars, local $R=1$')
    return h, lab


def fig_scales(results: list, variant: str, tag: str, kmax: float, path: Path,
               z_label=None) -> None:
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
        _style_axis(ax)
        ax.set_title(r['label'], fontsize=13)
    _k_range(axes.flat[0], have, kmax)
    h, lab = axes.flat[0].get_legend_handles_labels()
    hc, lc = _context_legend(have)
    axes.flat[0].legend(h + hc, lab + lc, fontsize=9, loc='lower left',
                        title=f"{variant_label(variant)}, {tag_label(tag)}", title_fontsize=9)
    _finish_grid(fig, axes, len(have), r'$P_{\rm mm}(s)/P_{\rm mm} - 1\;[\%]$', path, z_label)


def fig_methods(results: list, s: float, kmax: float, path: Path, z_label=None) -> None:
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
        _style_axis(ax)
        ax.set_title(r['label'], fontsize=13)
    _k_range(axes.flat[0], results, kmax)
    hc, lc = _context_legend(results)
    h = [plt.Line2D([], [], color=VARIANT_COLOUR[nm]) for nm in names] + \
        [plt.Line2D([], [], color='gray', ls=ls) for ls in TAG_STYLE[:len(tags)]] + hc
    lab = [variant_label(nm) for nm in names] + [tag_label(t) for t in tags] + lc
    title = r'selected stars $\to$ ionized gas' if s == 0.0 else rf'$s={s:g}$'
    axes.flat[0].legend(h, lab, fontsize=8, loc='lower left', title=title, title_fontsize=9)
    _finish_grid(fig, axes, len(results), r'$P_{\rm mm}(s)/P_{\rm mm} - 1\;[\%]$', path, z_label)


def _lensing_panel(ax, have: list, colours: list, variant: str, tag: str, lens_data,
                   xmax: float) -> None:
    """f_gas(theta) at s = 1 (solid, markers) and the lowest s (dashed), with the data."""
    missing = []
    for r, col in zip(have, colours):
        L = r.get('lens')
        if L is None or (variant, tag) not in L['cfg']:
            missing.append(r['label'])
            continue
        fs = L['cfg'][(variant, tag)]['f']
        s_lo = min(fs)
        th = L['theta']
        ax.plot(th, fs[1.0], color=col, lw=2, marker='o', ms=4)
        ax.plot(th, fs[s_lo], color=col, lw=1.5, ls='--')
        ax.fill_between(th, fs[1.0], fs[s_lo], color=col, alpha=0.2, lw=0)
    if missing:
        print(f"  {variant} {tag}: no lensing stacks for {', '.join(missing)} (bottom panel)")
    h = [plt.Line2D([], [], color='gray', lw=2, marker='o', ms=4),
         plt.Line2D([], [], color='gray', lw=1.5, ls='--')]
    lab = ['simulation', r'selected stars $\to$ ionized gas']
    if lens_data is not None:
        h.append(ax.errorbar(lens_data['theta'], lens_data['f'], yerr=lens_data['err'], fmt='s',
                             color='k', ms=6, capsize=2, zorder=5))
        lab.append(r'DESI $\times$ ACT $\times$ HSC (beam-corrected)')
    ax.axhline(1.0, color='k', lw=1)
    ax.grid(True, which='major', color='0.8', lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0.0, xmax)
    ax.set_xlabel(r'$\theta\;[\mathrm{arcmin}]$')
    ax.set_ylabel(r'$f_{\rm gas}(\theta)$')
    ax.legend(h, lab, loc='best', framealpha=0.85, fontsize=10)


def fig_S_band(results: list, variant: str, tag: str, kmax: float, path: Path,
               z_label=None, bottom: str = 'dS', lens_data=None, lens_xmax: float = 6.5) -> None:
    """S(k) from the simulation (s = 1) to all selected stars moved (s = 0).

    Below: the lensing f_gas(theta) under the same transfer (bottom='lensing';
    ``_lensing_panel``, own theta axis) or dS on the shared k axis (bottom='dS',
    the earlier figure).
    """
    have = [r for r in results if (variant, tag) in r['cfg']]
    if not have:
        print(f"no {variant} {tag} spectra; skipping {path.name}")
        return
    colours = [sim_colour(r) for r in have]
    if bottom == 'dS':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                                       gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.05))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10.5),
                                       gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.28))
    for r, col in zip(have, colours):
        d = r['cfg'][(variant, tag)]
        sel = r['k'] <= kmax
        k = r['k'][sel]
        s_lo = min(d['S'])
        ax1.plot(k, r['S0'][sel], color=col, lw=2, label=r['label'])
        ax1.plot(k, d['S'][s_lo][sel], color=col, lw=1.5, ls='--')
        ax1.fill_between(k, r['S0'][sel], d['S'][s_lo][sel], color=col, alpha=0.2, lw=0)
        if bottom == 'dS':
            ax2.plot(k, d['dS'][s_lo][sel], color=col, lw=2)
    ax1.axhline(1.0, color='k', lw=1)
    _style_axis(ax1)
    if bottom == 'dS':
        _style_axis(ax2)
    _k_range(ax1, have, kmax)
    ax1.set_ylabel(r'$S(k) = P_{\rm mm}/P_{\rm DMO}$')
    ax1.legend(loc='lower left', framealpha=0.85)
    title = f"stars moved to ionized gas: {variant_label(variant)}, {tag_label(tag)}"
    ax1.set_title(f"{z_label}\n{title}" if z_label else title, fontsize=13)
    if bottom == 'dS':
        ax2.axhline(0.0, color='k', lw=1)
        ax2.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
        ax2.set_ylabel(r'$\Delta S = S(s{=}0) - S$')
        ax2.plot([], [], color='gray', lw=2, label=r'simulation (solid, top); $s=0$ (dashed top, solid bottom)')
        ax2.legend(loc='lower left', framealpha=0.85, fontsize=10)
    else:
        ax1.set_xlabel(r'$k\;[h\,\mathrm{Mpc}^{-1}]$')
        _lensing_panel(ax2, have, colours, variant, tag, lens_data, lens_xmax)
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
    if config['plot'].get('z_label'):
        L.insert(1, f"# Redshift: {config['plot']['z_label']}")
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
        if r['budget_from_particles']:
            L.append("# (no components spectra: 'cached Stars' is the Stars cache or, without one, all PartType4")
            L.append("#  from the particle pass; baryons = PartType4 + gas, BH not included; 'global' column empty)")
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


def write_lensing_table(results: list, config: dict, path: Path) -> None:
    """f_gas(theta) at s = 1 and at the lowest s, and the lensing validation numbers."""
    lens = lensing_settings(config)
    L = ["# Lensing observable under the halo-level stellar transfer (stack_stellar_maps.py).",
         "# f(theta; s) = <DSigma[ionized_gas + t A]> / <DSigma[total + t (A - B)]> * Omega_m/Omega_b,",
         "#   t = 1 - s; A, B = ionized gas added / stars removed at s = 0 (compute_stellar_maps.py).",
         f"# Stacked sample (fixed for every s): '{lens['sample_name']}', settings of {lens['config_path']}"
         + (f" with overrides {config['lensing'].get('overrides')}" if config['lensing'].get('overrides') else ""),
         "# Configurations as in the P(k) table: <variant>/<tag>."]
    if config['plot'].get('z_label'):
        L.insert(1, f"# Redshift: {config['plot']['z_label']}")
    for r in results:
        Ls = r.get('lens')
        L.append(f"\n## {r['label']}")
        if Ls is None:
            L.append("(no lensing stacks)")
            continue
        confs = sorted(Ls['cfg'], key=lambda c: (list(VARIANT_COLOUR).index(c[0])
                                                 if c[0] in VARIANT_COLOUR else 9, float(c[1][1:])))
        L.append(f"# z = {Ls['z']:g}; {Ls['n_haloes']:,} haloes stacked")
        if not confs:
            L.append("(no transfer stacks)")
            continue
        s_lo = min(Ls['cfg'][confs[0]]['f'])
        names = ['s=1'] + [f"{c[0]}/{c[1]}" for c in confs]
        for qty in ('f', 'rel'):
            L.append(f"# f(s={s_lo:g})" if qty == 'f' else f"# f(s={s_lo:g}) / f(s=1) - 1 [%]")
            L.append(f"{'theta':>7s} " + " ".join(f"{nm:>8s}" for nm in names))
            f1 = Ls['cfg'][confs[0]]['f'][1.0]
            for i, th in enumerate(Ls['theta']):
                vals = [Ls['cfg'][c]['f'][s_lo][i] for c in confs]
                if qty == 'f':
                    row = [f1[i]] + vals
                    L.append(f"{th:7.3f} " + " ".join(f"{v:8.4f}" for v in row))
                else:
                    row = [100 * (v / f1[i] - 1.0) for v in vals]
                    L.append(f"{th:7.3f} {'':>8s} " + " ".join(f"{v:+8.2f}" for v in row))
        L.append("# validation")
        ch = Ls['checks']
        if 'check_stackmap_N' in ch:
            L.append(f"#   stack_on_array (this sample) vs stackMap (lensing path): N {ch['check_stackmap_N']:.1e}, "
                     f"T {ch['check_stackmap_T']:.1e}")
        if 'explicit_check_N' in ch:
            L.append(f"#   explicit maps at s = 0.5 ({ch['explicit_config']}) vs linear combination: "
                     f"N {ch['explicit_check_N']:.1e}, T {ch['explicit_check_T']:.1e}")
        for c in confs:
            md = Ls['cfg'][c]['mapdiag']
            if not md:
                continue
            parts = [f"sum A/moved-1 {md['sum_added_rel']:+.1e}", f"sum B/moved-1 {md['sum_removed_rel']:+.1e}",
                     f"per-halo {md['max_halo_cons']:.1e}"]
            if 'moved_vs_3d' in md:
                parts.append(f"moved vs 3D run {md['moved_vs_3d']:+.1e}, active haloes {int(md['active_vs_3d']):+d}")
            if 'neg_star_mass' in md:
                parts.append(f"neg. stars (2D Stars - B) {md['neg_star_mass'] / max(md['mstar_moved'], 1e-30):+.1e} "
                             f"({int(md['neg_star_pixels'])} pixels)")
            L.append(f"#   {c[0]}/{c[1]}: " + "; ".join(parts))
    path.write_text("\n".join(L) + "\n")
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None)
    parser.add_argument('--kmax', type=float, default=5.0,
                        help="largest k plotted [h/Mpc] (default 5)")
    parser.add_argument('--scale-variant', default='fof',
                        help="method variant of the stellar-scale figure (default fof)")
    parser.add_argument('--scale-tag', default=None,
                        help="mass-cut tag of the stellar-scale figure, e.g. M11 (default: every cut)")
    parser.add_argument('--band-tag', default=None,
                        help="mass-cut tag of the S(k) band figures (default: every cut)")
    parser.add_argument('--bottom', choices=['lensing', 'dS'], default=None,
                        help="bottom panel of the S(k) band figures: the lensing f_gas(theta) "
                             "(default with a 'lensing' config block) or dS (the earlier figure)")
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()

    config = load_config(args.path2config)
    bottom = args.bottom or ('lensing' if 'lensing' in config else 'dS')
    if bottom == 'lensing' and 'lensing' not in config:
        raise SystemExit(f"--bottom lensing needs a 'lensing' block in {args.path2config}")
    lens_data, lens_xmax = None, 6.5
    if bottom == 'lensing':
        lens = lensing_settings(config)
        lens_xmax = lens['stack']['max_radius'] * lens['stack']['rad_distance'] + 0.5
        if lens['data'] and Path(lens['data']).exists():
            with np.load(lens['data']) as dd:
                lens_data = dict(theta=dd['theta_arcmin'], f=dd['R_compensated'],
                                 err=dd['sigma_compensated'])
        else:
            print(f"no beam-compensated data at {lens['data']}; the lensing panel has no data points")
    plot_cfg = config['plot']
    now = datetime.now()
    out_dir = Path(plot_cfg['fig_path']) / now.strftime('%Y-%m') / now.strftime('%m-%d')
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = plot_cfg['fig_name'] + (f"_{args.suffix}" if args.suffix else '')
    ext = plot_cfg.get('fig_type', 'pdf')
    all_tags = [mass_tag(m) for m in sorted(float(m) for m in config['stellar']['halo_mass_min'])]

    # plot.exclude_sims: simulations left out of the figures (kept in the tables)
    excluded = set(plot_cfg.get('exclude_sims') or [])
    results, fig_results = [], []
    for entry in select_sims(config, args.sims):
        if not any(spectra_path(entry, f"stellar_{v['name']}").exists() for v in variants_of(config)):
            print(f"no stellar-transfer spectra for {sim_label(entry)}; skipping")
            continue
        results.append(analyse(entry, config, lensing=bottom == 'lensing'))
        if {sim_label(entry), entry['name'], entry.get('feedback')} & excluded:
            print(f"{sim_label(entry)}: left out of the figures (plot.exclude_sims), kept in the tables")
        else:
            fig_results.append(results[-1])
    if not results:
        raise SystemExit("no stellar-transfer spectra found")
    if not fig_results:
        raise SystemExit("every simulation is in plot.exclude_sims")

    s_min = min(float(s) for s in config['stellar']['stellar_scales'])
    z_label = plot_cfg.get('z_label')  # optional figure title, e.g. for z ~ 0.26
    for tag in ([args.scale_tag] if args.scale_tag else all_tags):
        fig_scales(fig_results, args.scale_variant, tag, args.kmax,
                   out_dir / f"{stem}_stellar_scales_{args.scale_variant}_{tag}.{ext}", z_label)
    fig_methods(fig_results, s_min, args.kmax, out_dir / f"{stem}_stellar_methods.{ext}", z_label)
    for btag in ([args.band_tag] if args.band_tag else all_tags):
        for v in variants_of(config):
            fig_S_band(fig_results, v['name'], btag, args.kmax,
                       out_dir / f"{stem}_stellar_S_{v['name']}_{btag}.{ext}", z_label,
                       bottom=bottom, lens_data=lens_data, lens_xmax=lens_xmax)
    write_table(results, config, out_dir / f"{stem}_stellar_table.txt")
    if bottom == 'lensing':
        write_lensing_table(results, config, out_dir / f"{stem}_stellar_lensing_table.txt")


if __name__ == '__main__':
    main()
