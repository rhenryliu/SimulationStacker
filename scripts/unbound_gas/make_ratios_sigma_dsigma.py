"""make_ratios_sigma_dsigma.py
============================
A variant of the unbound gas paper's Figure 1 (make_ratios3x2.py), kept out of
the draft: particle-type fraction profiles, normalised by the cosmic baryon
fraction (OmegaBaryon / OmegaMatter), from two 2D filters, with the grid
transposed so that each filter is a row.

Layout
------
Rows:    one per filter, from ``stack.rows`` (default ``['Sigma', 'DSigma']``),
         the same filter on the numerator and the denominator:
         'Sigma'  = Σ(R), the mean surface density in the ring [R, R + 0.75')
         'DSigma' = ΔΣ(R) = Σ̄(<R) - Σ(R), as Figure 1's ΔΣ column
Columns: one per simulation suite, in config order (IllustrisTNG, SIMBA,
         FLAMINGO), with the suite's legend below the bottom row.

Both rows stack the same unconvolved maps (``stack.pixel_size`` 0.2 arcmin,
``stack.beam_size`` 0: no beam), and the row titles, right of each row, say
so. The ring width is 0.75 arcmin, stack_on_array's default for arcmin radii
(stackMap does not pass one), so the Σ(R) ring is exactly the annulus the ΔΣ
filter subtracts; Σ(R) is plotted at the ring's inner edge R.

The radii, halo selection ('massive', mean mass 10^13.22), ratio and error are
those of make_ratios3x2.py, whose helpers this script imports. The error bands
are the standard errors of the halo means, propagated as if the two particle
types were independent (as in Figure 1); their covariance is dropped. The R200m line of
each column (that suite's first simulation) is the mean GroupRad of the same
selection; Figure 1 took it from its 3D column, which is not drawn here.

Usage (from scripts/)
---------------------
    python unbound_gas/make_ratios_sigma_dsigma.py -p configs/unbound_gas/ratios_sigma_dsigma_z05.yaml

``--from-npz <profiles.npz>`` redraws the figure from the profiles file of an
earlier run (written next to each figure) instead of stacking.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml

# Figure 1's helpers, colours and plot style (sets sys.path for ../src/ too).
import make_ratios3x2 as mr3
from utils import arcmin_to_comoving, comoving_to_arcmin
from halos import select_massive_halos

# Ring width of 'Sigma' and annulus width of 'DSigma' (arcmin): stack_on_array's
# default for arcmin radii, which stackMap cannot override.
_RING_WIDTH_ARCMIN = 0.75

# Row kinds: the filterType passed to stackMap and the row label.
_ROW_TITLES = {'Sigma': rf"2D $\Sigma(R)$, ring $[R, R + {_RING_WIDTH_ARCMIN:g}')$",
               'DSigma': r'2D $\Delta\Sigma(R)$'}

# Upper halo mass bound of the 'massive' selection. It must equal stackMap's
# default halo_mass_upper, which compute_2d_profile_ratio cannot override, or
# the R200m line no longer describes the stacked haloes; also hard-coded in
# make_ratios3x2's 3D column.
_HALO_MASS_UPPER = 5e14

# Short subscripts for the y labels; other particle types use their name.
_PTYPE_SUBSCRIPTS = {'ionized_gas': 'ion', 'total': 'tot', 'baryon': 'b', 'gas': 'gas'}


def mean_R200m(stacker, halo_mass_avg: float) -> float:
    """Mean R200m (comoving kpc/h) of the haloes the 2D stacks select.

    Args:
        stacker (SimulationStacker): Stacker of the simulation.
        halo_mass_avg (float): Target mean mass of the 'massive' selection, in
            the units of the catalogue's ``GroupMass``.

    Returns:
        float: Mean ``GroupRad`` (R200m, comoving kpc/h) of the selection.
    """
    haloes = stacker.loadHalos()
    halo_mask = select_massive_halos(haloes['GroupMass'], halo_mass_avg, _HALO_MASS_UPPER)
    return float(np.mean(haloes['GroupRad'][halo_mask]))


def _subscript(pType: str) -> str:
    """Subscript naming a particle type in the y labels."""
    return _PTYPE_SUBSCRIPTS.get(pType, pType.replace('_', r'\_'))


def configure_panel(ax, row_kind: str, is_top: bool, is_bottom: bool, is_left: bool,
                    pType: str, pType2: str, R200m_arcmin: float | None,
                    forward_arcmin, inverse_arcmin, xlim: float,
                    column_title: str, panel_label: str):
    """Apply the axis decorations to one panel.

    Args:
        ax (matplotlib.axes.Axes): The panel.
        row_kind (str): 'Sigma' or 'DSigma', for the y label.
        is_top, is_bottom, is_left (bool): Position of the panel in the grid.
        pType, pType2 (str): Numerator and denominator particle types.
        R200m_arcmin (float or None): R200m of the column's first simulation
            (arcmin), drawn as a vertical line.
        forward_arcmin, inverse_arcmin (callable): arcmin <-> comoving kpc/h for
            the column's first simulation, for the top axis.
        xlim (float): Upper x limit (arcmin).
        column_title (str): Suite name, drawn above the top row.
        panel_label (str): Subplot letter, e.g. ``'(a)'``.
    """
    ax.axhline(1.0, color='k', ls='--', lw=2)
    if R200m_arcmin is not None:
        ax.axvline(R200m_arcmin, color='gray', ls=':', lw=2, label=r'$R_{200\mathrm{m}}$')
    ax.set_xlim(0.0, xlim)
    ax.grid(True)

    if is_left:
        sym = r'\Sigma' if row_kind == 'Sigma' else r'\Delta\Sigma'
        ax.set_ylabel(
            rf'${sym}_{{\mathrm{{{_subscript(pType)}}}}} / {sym}_{{\mathrm{{{_subscript(pType2)}}}}}'
            r' \;/\; (\Omega_b / \Omega_m)$',
            fontsize=18,
        )
    if is_bottom:
        ax.set_xlabel('R [arcmin]', fontsize=18)
    if is_top:
        secax = ax.secondary_xaxis('top', functions=(forward_arcmin, inverse_arcmin))
        secax.set_xlabel('R [comoving kpc/h]', fontsize=18)
        ax.set_title(column_title, fontsize=18, fontweight='bold')

    ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
            fontsize=18, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))


def main(path2config: str, ptype: str = None, from_npz: str = None, verbose: bool = True):
    """Generate the Σ(R) / ΔΣ(R) fraction figure.

    Args:
        path2config (str): Path to the YAML configuration file.
        ptype (str, optional): Numerator particle type (overrides the config).
        from_npz (str, optional): Profiles file of an earlier run
            (``<fig_name>_<ptype>_profiles.npz``); if given, the profiles are
            read from it instead of being stacked.
        verbose (bool): If True, print progress and the plotted values.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)
    stack_cfg = config.get('stack', {})
    plot_cfg = config.get('plot', {})

    redshift = stack_cfg.get('redshift', 0.5)
    projection = stack_cfg.get('projection', 'yz')
    pType = ptype if ptype is not None else stack_cfg.get('particle_type', 'ionized_gas')
    pType2 = stack_cfg.get('particle_type_2', 'total')

    # Radial range in comoving kpc/h, converted to arcmin per simulation (as
    # Figure 1's 2D columns do with its 3D range).
    min_radius = float(stack_cfg.get('min_radius', 200.0))
    max_radius = float(stack_cfg.get('max_radius', 4000.0))
    num_radii = int(stack_cfg.get('num_radii', 11))
    halo_mass_avg = float(stack_cfg.get('halo_mass_avg', 10 ** 13.22))

    pixel_size = float(stack_cfg.get('pixel_size', 0.2))
    beam_size = stack_cfg.get('beam_size', 0)
    beam_size = 0 if beam_size in (None, 0) else float(beam_size)  # 0: no beam
    params_2d = {
        'halo_mass_avg': halo_mass_avg,
        'beam_size':     beam_size,
        'pixel_size':    pixel_size,
        'rad_distance':  float(stack_cfg.get('rad_distance', 1.0)),
        'projection':    projection,
        'save_field':    stack_cfg.get('save_field', True),
        'load_field':    stack_cfg.get('load_field', True),
        'subtract_mean': stack_cfg.get('subtract_mean', False),
    }

    rows = stack_cfg.get('rows', ['Sigma', 'DSigma'])
    unknown = [r for r in rows if r not in _ROW_TITLES]
    if unknown:
        raise ValueError(f"Unknown row kind(s) {unknown}; use 'Sigma' or 'DSigma'.")
    # Row labels name the filter and the maps both rows are stacked on.
    beam_str = rf"${float(beam_size):g}'$ beam" if beam_size else 'no beam'
    map_str = rf"${pixel_size:g}'$ pixels, {beam_str}"
    row_titles = {r: f'{_ROW_TITLES[r]}\n{map_str}' for r in rows}

    now = datetime.now()
    figPath = Path(plot_cfg.get('fig_path', '../figures/')) / now.strftime("%Y-%m") / now.strftime("%m-%d")
    figPath.mkdir(parents=True, exist_ok=True)
    figName = plot_cfg.get('fig_name', 'ratios_sigma_dsigma')
    figType = plot_cfg.get('fig_type', 'pdf')
    plot_error_bars = plot_cfg.get('plot_error_bars', True)

    # One column per suite, with Figure 1's colours.
    suites = []
    for suite in config['simulations']:
        name = suite['sim_type']
        sims = suite['sims']
        if name == 'FLAMINGO':
            fallback = matplotlib.colormaps['plasma'](np.linspace(0.2, 0.85, len(sims)))  # type: ignore
            colours = [mr3._FLAMINGO_COLOURS.get(s['feedback'], fallback[k]) for k, s in enumerate(sims)]
        elif name in mr3._COLOURMAPS:
            cmap = matplotlib.colormaps[mr3._COLOURMAPS[name]]  # type: ignore
            colours = list(cmap(np.linspace(0.2, 0.85, len(sims))))
            if name == 'SIMBA':
                colours = [mr3._SIMBA100_COLOUR if (s['name'], s.get('feedback')) == ('m100n1024', 's50')
                           else c for s, c in zip(sims, colours)]
            elif name == 'IllustrisTNG':
                colours = [mr3._TNG_COLOURS.get(s['name'], c) for s, c in zip(sims, colours)]
        else:
            raise ValueError(f"Unknown simulation type: {name!r}")
        # Column title and legend title; Figure 1 configs call it row_label.
        suites.append((name, sims, colours, suite.get('label', suite.get('row_label', name))))
    nRows, nCols = len(rows), len(suites)

    npz_in = None
    if from_npz:
        expected = f'{figName}_{pType}_profiles.npz'
        if Path(from_npz).name != expected:
            raise ValueError(f"--from-npz {from_npz!r} is not this config's profiles file ({expected!r})")
        npz_in = dict(np.load(from_npz))
        print(f"Redrawing from {from_npz} (no stacking)")

    # Panels of plot.panel_width x plot.panel_height, plus a strip below the
    # bottom row for the per-column legends (the longest, FLAMINGO's, has six
    # entries with the R200m line) and one on the right for the row titles.
    panel_width = float(plot_cfg.get('panel_width', 4.6))
    panel_height = float(plot_cfg.get('panel_height', 4.0))
    legend_height = float(plot_cfg.get('legend_height', 2.2))  # inches
    title_width = 0.9  # inches
    fig_width = panel_width * nCols + title_width
    fig_height = panel_height * nRows + legend_height
    fig, axes = plt.subplots(nRows, nCols, figsize=(fig_width, fig_height),
                             sharex=True, sharey='row', squeeze=False)

    # Per column: R200m (arcmin) and the arcmin <-> kpc/h converters of the
    # suite's first simulation.
    R200m_arcmin_col = [None] * nCols
    converters_col = [None] * nCols
    max_arcmin = 0.0
    profiles_out = {}
    t0 = time.time()

    for col_idx, (sim_type_name, sims, colours, _) in enumerate(suites):
        if verbose:
            print(f"\n{'='*60}\nSuite: {sim_type_name}  (column {col_idx})\n{'='*60}")
        for j, sim in enumerate(sims):
            if verbose:
                feedback_str = f"  feedback={sim.get('feedback')}" if 'feedback' in sim else ''
                print(f"\n  [{j+1}/{len(sims)}] {sim['name']}{feedback_str}")

            sim_z = float(sim.get('redshift', redshift))
            stacker, OmegaBaryon, cosmo, sim_label = mr3.setup_stacker(sim, sim_type_name, sim_z)

            def _fwd(arcmin, c=cosmo, z=sim_z): return arcmin_to_comoving(arcmin, z, c)
            def _inv(comov, c=cosmo, z=sim_z):  return comoving_to_arcmin(comov, z, c)

            sim_key = '/'.join(str(k) for k in (sim_type_name, sim['name'], sim.get('feedback'))
                               if k is not None)
            if npz_in is not None:
                if f'{sim_key}/R200m_kpch' not in npz_in:
                    raise KeyError(f"not in {from_npz}: {sim_key}/R200m_kpch")
                R200m_kpch = float(npz_in[f'{sim_key}/R200m_kpch'])
            else:
                R200m_kpch = mean_R200m(stacker, halo_mass_avg)
            profiles_out[f'{sim_key}/R200m_kpch'] = R200m_kpch
            if converters_col[col_idx] is None:
                converters_col[col_idx] = (_fwd, _inv)
                R200m_arcmin_col[col_idx] = comoving_to_arcmin(R200m_kpch, sim_z, cosmo)

            for row_idx, kind in enumerate(rows):
                if npz_in is not None:
                    needed = [f'{sim_key}/{kind}_{q}' for q in ('radii', 'ratio', 'err')]
                    missing = [k for k in needed if k not in npz_in]
                    if missing:
                        raise KeyError(f"not in {from_npz}: {missing}")
                    radii, ratio, err = (npz_in[k] for k in needed)
                else:
                    if verbose:
                        print(f"    Stacking {kind} (pixel {pixel_size}', beam {beam_size})...")
                    radii, ratio, err = mr3.compute_2d_profile_ratio(
                        stacker, pType, pType2, kind, kind, params_2d, OmegaBaryon,
                        min_radius, max_radius, num_radii, _inv)
                max_arcmin = max(max_arcmin, float(np.max(radii)))
                profiles_out[f'{sim_key}/{kind}_radii'] = radii
                profiles_out[f'{sim_key}/{kind}_ratio'] = ratio
                profiles_out[f'{sim_key}/{kind}_err'] = err

                mr3.plot_panel(axes[row_idx, col_idx], radii, ratio, err,
                               sim_label, colours[j], plot_error_bars)
                if verbose:
                    print(f"    [{kind}] R = {np.array2string(radii, precision=3)}")
                    print(f"    [{kind}] ratio = {np.array2string(ratio, precision=4)}")
                    print(f"    [{kind}] err = {np.array2string(err, precision=4)}")

    # Shared upper x limit (arcmin), padded so no profile is clipped.
    xlim = max_arcmin + 0.5
    panel_idx = 0
    for row_idx, kind in enumerate(rows):
        for col_idx, suite in enumerate(suites):
            fwd, inv = converters_col[col_idx]
            configure_panel(
                axes[row_idx, col_idx], kind,
                is_top=row_idx == 0, is_bottom=row_idx == nRows - 1, is_left=col_idx == 0,
                pType=pType, pType2=pType2, R200m_arcmin=R200m_arcmin_col[col_idx],
                forward_arcmin=fwd, inverse_arcmin=inv, xlim=xlim,
                column_title=suite[3], panel_label=f'({chr(ord("a") + panel_idx)})')
            panel_idx += 1

    # Lay out the panels above the legend strip and left of the row-title
    # strip; then the row titles right of the rightmost panels (the y labels
    # name the ratio) and one legend per column (handles from its bottom
    # panel) below that column.
    fig.tight_layout(rect=[0, legend_height / fig_height, 1 - title_width / fig_width, 1])  # type: ignore
    for row_idx, kind in enumerate(rows):
        axes[row_idx, -1].annotate(row_titles[kind], xy=(1.04, 0.5), xycoords='axes fraction',
                                   ha='left', va='center', rotation=270, fontsize=16,
                                   multialignment='center')
    for col_idx, suite in enumerate(suites):
        handles, labels = axes[-1, col_idx].get_legend_handles_labels()
        bbox = axes[-1, col_idx].get_position()
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=((bbox.x0 + bbox.x1) / 2, legend_height / fig_height),
                   frameon=True, fontsize=13, title=suite[3], title_fontsize=14)

    out_path = figPath / f'{figName}_{pType}.{figType}'
    fig.savefig(out_path, dpi=300)  # type: ignore
    plt.close(fig)
    profiles_out['row_kinds'] = np.array(rows)
    profiles_out['row_titles'] = np.array([row_titles[r] for r in rows])
    np.savez(figPath / f'{figName}_{pType}_profiles.npz', **profiles_out)

    print(f"\nFigure saved to: {out_path}")
    print(f"Total time: {(time.time() - t0) / 60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Sigma(R) / DeltaSigma(R) fraction figure.")
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/unbound_gas/ratios_sigma_dsigma_z05.yaml',
                        help='Path to the YAML configuration file.')
    parser.add_argument('--ptype', type=str, default=None,
                        help='Override the numerator particle type from the config.')
    parser.add_argument('--from-npz', type=str, default=None,
                        help='Redraw from the profiles .npz of an earlier run instead of stacking.')
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")
    main(**args)
