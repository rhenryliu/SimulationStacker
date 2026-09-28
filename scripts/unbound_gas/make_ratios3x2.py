"""make_ratios3x2.py
===================
Generate a grid of particle-type fraction profiles, normalised by the
cosmic baryon fraction (OmegaBaryon / OmegaMatter).

Layout
------
Rows:    one per simulation suite, in config order (IllustrisTNG, SIMBA,
         FLAMINGO); the default config is TNG on top, SIMBA below.
Columns: chosen by ``stack.columns`` (default ``['3d', 'col1', 'col2']``):
         '3d'   = 3D spherical profiles        (radius in comoving kpc/h)
         'col1' = 2D projected, filter_type_col1 (cumulative; arcmin)
         'col2' = 2D projected, filter_type_col2 (CAP; arcmin)
         'col3' = 2D projected, filter_type_col3 (DSigma; arcmin)

The 3D grid size may be set per simulation (``n_pixels`` next to ``name`` and
``snapshot``), falling back to ``stack.n_pixels``; FLAMINGO needs ~2000 to
approach the resolution the other suites get at 1000.

Usage
-----
    python unbound_gas/make_ratios3x2.py -p configs/unbound_gas/ratios_3x2_z05.yaml
"""

import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import yaml
import argparse

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.append('../src/')
from utils import arcmin_to_comoving, comoving_to_arcmin, flamingo_label
from stacker import SimulationStacker
from halos import select_massive_halos
from mask_utils import get_cutout_indices_3d, sum_over_cutouts

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex":      True,
    "mathtext.fontset": "cm",
    "font.size":        20,
    "axes.titlesize":   20,
    "axes.labelsize":   20,
    "xtick.labelsize":  20,
    "ytick.labelsize":  20,
    "legend.fontsize":  13,
})

# Colour maps used for the TNG and SIMBA suites.
_COLOURMAPS = {'IllustrisTNG': 'twilight', 'SIMBA': 'hsv'}

# Fixed colours for the FLAMINGO feedback variants, keyed by feedback name.
# Kept in sync with simulated_kSZ_masked.py so the same simulation is the same
# colour across every figure in the paper.
_FLAMINGO_COLOURS = {
    'L1_m9':           '#B30000',  # dark red (fiducial)
    'fgas-8sigma':     '#FF7F0E',  # orange
    'Jet_fgas-4sigma': '#C71585',  # magenta
}

# Column kinds (see module docstring) and their titles; the 2D titles name
# the filter set in the config.
_DEFAULT_COLUMNS = ['3d', 'col1', 'col2']
_FILTER_TITLES = {'cumulative': 'cumulative', 'CAP': 'CAP', 'DSigma': r'$\Delta\Sigma$'}

# Default OmegaBaryon for Illustris-1 (not stored in header).
_OMEGA_BARYON_ILLUSTRIS_DEFAULT = 0.0456
# Default OmegaBaryon for SIMBA (not stored in header).
_OMEGA_BARYON_SIMBA_DEFAULT = 0.048


# ===========================================================================
# Helper functions
# ===========================================================================

def setup_stacker(sim: dict, sim_type_name: str, redshift: float):
    """Instantiate a SimulationStacker and derive cosmological quantities.

    Parameters
    ----------
    sim : dict
        Single simulation entry from the YAML ``simulations`` block.
        Must contain ``name`` and ``snapshot``; SIMBA entries also need
        ``feedback``.
    sim_type_name : str
        ``'IllustrisTNG'``, ``'SIMBA'`` or ``'FLAMINGO'``.
    redshift : float
        Target simulation redshift.

    Returns
    -------
    stacker : SimulationStacker
    OmegaBaryon : float
    cosmo : FlatLambdaCDM
    sim_label : str
        Human-readable label for legends (includes feedback suffix for SIMBA).
    """
    sim_name = sim['name']
    snapshot = sim['snapshot']

    if sim_type_name == 'IllustrisTNG':
        stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                    simType=sim_type_name)
        try:
            OmegaBaryon = stacker.header['OmegaBaryon']
        except KeyError:
            OmegaBaryon = _OMEGA_BARYON_ILLUSTRIS_DEFAULT
        sim_label = sim_name

    elif sim_type_name == 'SIMBA':
        feedback = sim['feedback']
        stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                    simType=sim_type_name,
                                    feedback=feedback)
        OmegaBaryon = _OMEGA_BARYON_SIMBA_DEFAULT
        sim_label = f"{sim_name}_{feedback}"

    elif sim_type_name == 'FLAMINGO':
        # feedback holds the variant directory name ('L1_m9' = fiducial).
        feedback = sim['feedback']
        stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                    simType=sim_type_name,
                                    feedback=feedback)
        OmegaBaryon = stacker.header['OmegaBaryon']
        # The row's legend title already names the suite.
        sim_label = flamingo_label(feedback, prefix=False)

    else:
        raise ValueError(f"Unknown simulation type: {sim_type_name!r}")

    cosmo = FlatLambdaCDM(
        H0=100 * stacker.header['HubbleParam'],
        Om0=stacker.header['Omega0'],
        Tcmb0=2.7255 * u.K,
        Ob0=OmegaBaryon,
    )

    return stacker, OmegaBaryon, cosmo, sim_label


def _profile_ratio_and_err(profiles0: np.ndarray, profiles1: np.ndarray,
                            OmegaBaryon: float, Omega0: float):
    """Compute the baryon-fraction-normalised mean ratio and its propagated
    standard error from per-halo stacked profiles.

    Parameters
    ----------
    profiles0, profiles1 : ndarray of shape (n_radii, n_halos)
        Stacked profiles for the numerator and denominator particle types.
    OmegaBaryon : float
    Omega0 : float
        Total matter density parameter.

    Returns
    -------
    ratio : ndarray of shape (n_radii,)
    err   : ndarray of shape (n_radii,)
    """
    mean0 = np.mean(profiles0, axis=1)
    mean1 = np.mean(profiles1, axis=1)
    ratio = mean0 / mean1 / (OmegaBaryon / Omega0)

    # Propagate standard errors in quadrature.
    se0 = np.std(profiles0, axis=1) / np.sqrt(profiles0.shape[1])
    se1 = np.std(profiles1, axis=1) / np.sqrt(profiles1.shape[1])
    err = np.abs(ratio) * np.sqrt((se0 / mean0) ** 2 + (se1 / mean1) ** 2)

    return ratio, err


def compute_3d_profile_ratio(stacker: SimulationStacker,
                              pType: str, pType2: str,
                              params: dict,
                              OmegaBaryon: float):
    """Compute 3D spherical-shell fraction profiles.

    Builds 3D density fields with ``stacker.makeField``, selects halos by
    mass, and accumulates the enclosed mass in spherical apertures using
    ``get_cutout_indices_3d`` / ``sum_over_cutouts``.

    Parameters
    ----------
    stacker : SimulationStacker
    pType, pType2 : str
        Numerator and denominator particle types.
    params : dict
        Sub-dict of stack parameters (``n_pixels``, ``min_radius_3d``,
        ``max_radius_3d``, ``num_radii_3d``, ``projection``,
        ``save_field``, ``load_field``, ``subtract_mean``,
        ``halo_mass_min``, ``halo_mass_max``).
    OmegaBaryon : float

    Returns
    -------
    radii : ndarray   — comoving kpc/h
    ratio : ndarray
    err   : ndarray
    R200m : float     — mean R200m (mean-overdensity radius) for the selected
                        halos (comoving kpc/h)
    """
    nPixels     = params['n_pixels']
    minR        = params['min_radius_3d']
    maxR        = params['max_radius_3d']
    nRadii      = params['num_radii_3d']
    projection  = params['projection']
    save        = params['save_field']
    load        = params['load_field']
    sub_mean    = params['subtract_mean']
    # mass_min    = params['halo_mass_min']
    # mass_max    = params.get('halo_mass_max', None)
    mass_min    = 10 ** 13.22 
    mass_max    = 5 * 1e14  

    # Build 3D fields for both particle types.
    field0 = stacker.makeField(pType, nPixels=nPixels, dim='3D',
                               projection=projection, save=save, load=load)
    field0 = field0 - np.mean(field0) if sub_mean else field0

    field1 = stacker.makeField(pType2, nPixels=nPixels, dim='3D',
                               projection=projection, save=save, load=load)
    field1 = field1 - np.mean(field1) if sub_mean else field1

    # Physical scale: comoving kpc/h per pixel.
    kpc_per_pixel = stacker.header['BoxSize'] / field0.shape[0]

    # Halo selection by mass.
    haloes    = stacker.loadHalos()
    halo_mask = select_massive_halos(haloes['GroupMass'], mass_min, mass_max)
    GroupPos_masked = (
        np.round(haloes['GroupPos'][halo_mask] / kpc_per_pixel).astype(int) % nPixels
    )
    R200m = np.mean(haloes['GroupRad'][halo_mask])  # comoving kpc/h (mean overdensity)

    # Stack in spherical apertures at each radius.
    radii     = np.linspace(minR, maxR, nRadii)
    profiles0 = []
    profiles1 = []
    for r in radii:
        rr = np.ones(GroupPos_masked.shape[0]) * r / kpc_per_pixel
        idx = get_cutout_indices_3d(field0, GroupPos_masked, rr)
        profiles0.append(sum_over_cutouts(field0, idx.copy()))
        profiles1.append(sum_over_cutouts(field1, idx.copy()))

    profiles0 = np.array(profiles0)   # shape (n_radii, n_halos)
    profiles1 = np.array(profiles1)

    ratio, err = _profile_ratio_and_err(profiles0, profiles1,
                                        OmegaBaryon, stacker.header['Omega0'])
    return radii, ratio, err, R200m


def compute_2d_profile_ratio(stacker: SimulationStacker,
                              pType: str, pType2: str,
                              filterType: str, filterType2: str,
                              params: dict,
                              OmegaBaryon: float,
                              minR_com: float, maxR_com: float, nRadii: int,
                              inverse_arcmin):
    """Compute 2D projected fraction profiles via ``stackMap``.

    The radial range is supplied in comoving kpc/h (matching the 3D column) and
    converted to arcmin per-simulation via ``inverse_arcmin`` (the sim's own
    cosmology), so that every 2D profile reaches the same comoving extent.

    Parameters
    ----------
    stacker : SimulationStacker
    pType, pType2 : str
        Numerator and denominator particle types.
    filterType, filterType2 : str
        Filter applied when stacking (e.g. ``'CAP'``, ``'cumulative'``).
    params : dict
        Sub-dict of stack parameters (``pixel_size``, ``rad_distance``,
        ``projection``, ``save_field``, ``load_field``, ``subtract_mean``).
    OmegaBaryon : float
    minR_com, maxR_com : float
        Inner/outer stacking radius in comoving kpc/h.
    nRadii : int
        Number of radial bins.
    inverse_arcmin : callable
        comoving kpc/h → arcmin conversion for this simulation's cosmology.

    Returns
    -------
    radii : ndarray   — arcmin (scaled by ``rad_distance``)
    ratio : ndarray
    err   : ndarray
    """
    pixelSize   = params['pixel_size']
    radDistance = params['rad_distance']
    projection  = params['projection']
    save        = params['save_field']
    load        = params['load_field']
    sub_mean    = params['subtract_mean']

    # Convert the comoving radial range to arcmin using this sim's cosmology.
    minR = inverse_arcmin(minR_com)
    maxR = inverse_arcmin(maxR_com)

    radii0, profiles0 = stacker.stackMap(
        pType, filterType=filterType,
        minRadius=minR, maxRadius=maxR, numRadii=nRadii,
        save=save, load=load, radDistance=radDistance,
        pixelSize=pixelSize, projection=projection,
        subtract_mean=sub_mean,
    )
    radii1, profiles1 = stacker.stackMap(
        pType2, filterType=filterType2,
        minRadius=minR, maxRadius=maxR, numRadii=nRadii,
        save=save, load=load, radDistance=radDistance,
        pixelSize=pixelSize, projection=projection,
        subtract_mean=sub_mean,
    )

    ratio, err = _profile_ratio_and_err(profiles0, profiles1,
                                        OmegaBaryon, stacker.header['Omega0'])
    # radii0 and radii1 share the same x-axis (same stacking parameters).
    # stackMap returns radii in units of radDistance (the rr grid spans
    # +/- n_vir in multiples of radDistance), so multiply to get arcmin.
    return radii0 * radDistance, ratio, err


def plot_panel(ax, radii: np.ndarray, ratio: np.ndarray, err: np.ndarray,
               label: str, colour, plot_error_bars: bool):
    """Draw a single profile line with an optional shaded error band.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    radii : ndarray
    ratio, err : ndarray
    label : str
    colour : colour spec accepted by matplotlib
    plot_error_bars : bool
    """
    ax.plot(radii, ratio, label=label, color=colour, lw=2, marker='o')
    if plot_error_bars:
        ax.fill_between(radii, ratio - err, ratio + err,
                        color=colour, alpha=0.2)


def configure_subplot(ax, kind: str, title: str,
                      is_top: bool, is_bottom: bool, is_left: bool,
                      pType: str, pType2: str,
                      R200m_kpch: float | None,
                      R200m_arcmin: float | None,
                      forward_arcmin, inverse_arcmin,
                      xlim_2d: float,
                      panel_label: str):
    """Apply axis decorations to a single subplot panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    kind : str
        ``'3d'`` for the 3D column (x in comoving kpc/h); anything else is a
        2D column (x in arcmin).
    title : str
        Column title, drawn on the top row only.
    is_top, is_bottom, is_left : bool
        Position of the panel in the grid.
    pType, pType2 : str
        Particle type names used for y-axis label.
    R200m_kpch : float or None
        R200m (mean-overdensity radius) in comoving kpc/h for the vertical
        reference line (3D column only).
    R200m_arcmin : float or None
        R200m (mean-overdensity radius) in arcmin for the vertical reference
        line (2D columns only).
    forward_arcmin, inverse_arcmin : callable
        Conversion functions between arcmin and comoving kpc/h, used to add
        a secondary x-axis on the top row (2D columns).
    xlim_2d : float
        Upper limit for the 2D x-axis (arcmin), already scaled by
        ``rad_distance`` and padded to avoid clipping any profile.
    panel_label : str
        Subplot letter, e.g. ``'(a)'``.
    """
    is_3d = kind == '3d'

    # --- Horizontal reference line at unity ---
    ax.axhline(1.0, color='k', ls='--', lw=2)

    # --- R200m vertical reference line (mean-overdensity radius) ---
    if is_3d and R200m_kpch is not None:
        ax.axvline(R200m_kpch, color='gray', ls=':', lw=2, label=r'$R_{200\mathrm{m}}$')
    elif not is_3d and R200m_arcmin is not None:
        ax.axvline(R200m_arcmin, color='gray', ls=':', lw=2, label=r'$R_{200\mathrm{m}}$')

    # --- Axis limits ---
    ax.set_xlim(0.0, None if is_3d else xlim_2d)
    ax.grid(True)

    # --- Y axis label (left column only) ---
    if is_left:
        ax.set_ylabel(
            rf'$\frac{{\mathrm{{{pType}}}}}{{\mathrm{{{pType2}}}}} \;/\; (\Omega_b / \Omega_m)$',
            fontsize=18,
        )

    # --- X axis label (bottom row) and secondary axis + title (top row) ---
    if is_bottom:
        ax.set_xlabel('R [comoving kpc/h]' if is_3d else 'R [arcmin]', fontsize=18)
    if is_top:
        if is_3d:
            # 3D column is already in comoving kpc/h: no conversion needed.
            secax = ax.secondary_xaxis('top')
        else:
            secax = ax.secondary_xaxis('top',
                                       functions=(forward_arcmin, inverse_arcmin))
        secax.set_xlabel('R [comoving kpc/h]', fontsize=18)
        ax.set_title(title, fontsize=18)

    # --- Subplot panel label in top-left corner ---
    ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
            fontsize=18, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))


# ===========================================================================
# Main
# ===========================================================================

def main(path2config: str, ptype: str, verbose: bool = True):
    """Generate the particle-fraction ratio figure grid.

    Parameters
    ----------
    path2config : str
        Path to the YAML configuration file.
    ptype : str
        Particle type to plot (overrides config).
    verbose : bool
        If True, print progress messages to stdout.
    """
    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_cfg = config.get('stack', {})
    plot_cfg  = config.get('plot', {})

    # --- Shared stacking parameters ---
    redshift    = stack_cfg.get('redshift', 0.5)
    projection  = stack_cfg.get('projection', 'xy')
    save_field  = stack_cfg.get('save_field', True)
    load_field  = stack_cfg.get('load_field', True)
    subtract_mn = stack_cfg.get('subtract_mean', False)
    pType       = ptype if ptype is not None else stack_cfg.get('particle_type', 'ionized_gas')
    # pType       = stack_cfg.get('particle_type', 'ionized_gas')
    pType2      = stack_cfg.get('particle_type_2', 'total')

    # --- 3D column parameters ---
    params_3d = {
        'n_pixels':      stack_cfg.get('n_pixels', 1000),
        'min_radius_3d': stack_cfg.get('min_radius_3d', 200.0),
        'max_radius_3d': stack_cfg.get('max_radius_3d', 4000.0),
        'num_radii_3d':  stack_cfg.get('num_radii_3d', 11),
        'projection':    projection,
        'save_field':    save_field,
        'load_field':    load_field,
        'subtract_mean': subtract_mn,
        'halo_mass_min': stack_cfg.get('halo_mass_min', 10 ** 13.22),
        'halo_mass_max': stack_cfg.get('halo_mass_max', 5e14),
    }

    # --- 2D column parameters ---
    # The 2D columns reuse the 3D comoving radial range (min_radius_3d,
    # max_radius_3d, num_radii_3d), converted to arcmin per-simulation, so both
    # columns extend to the same comoving extent (4000 ckpc/h) as the 3D column.
    rad_distance = stack_cfg.get('rad_distance', 1.0)
    params_2d = {
        'pixel_size':    stack_cfg.get('pixel_size', 0.5),
        'rad_distance':  rad_distance,
        'projection':    projection,
        'save_field':    save_field,
        'load_field':    load_field,
        'subtract_mean': subtract_mn,
    }

    # Filter types for columns 1 (cumulative), 2 (CAP) and 3 (DSigma).
    ft_col1  = stack_cfg.get('filter_type_col1',   'cumulative')
    ft2_col1 = stack_cfg.get('filter_type_2_col1', 'cumulative')
    ft_col2  = stack_cfg.get('filter_type_col2',   'CAP')
    ft2_col2 = stack_cfg.get('filter_type_2_col2', 'CAP')
    ft_col3  = stack_cfg.get('filter_type_col3',   'DSigma')
    ft2_col3 = stack_cfg.get('filter_type_2_col3', 'DSigma')

    # --- Plotting parameters ---
    now       = datetime.now()
    yr_string = now.strftime("%Y-%m")
    dt_string = now.strftime("%m-%d")
    figPath   = Path(plot_cfg.get('fig_path', '../figures/')) / yr_string / dt_string
    figPath.mkdir(parents=True, exist_ok=True)

    figName        = plot_cfg.get('fig_name', 'ratios_3x2')
    figType        = plot_cfg.get('fig_type', 'pdf')
    plot_error_bars = plot_cfg.get('plot_error_bars', True)

    # Columns to draw, and the (filter, filter_2) pair of each 2D column.
    columns = stack_cfg.get('columns', _DEFAULT_COLUMNS)
    col_filters = {'col1': (ft_col1, ft2_col1), 'col2': (ft_col2, ft2_col2),
                   'col3': (ft_col3, ft2_col3)}
    unknown = [c for c in columns if c != '3d' and c not in col_filters]
    if unknown:
        raise ValueError(f"Unknown column kind(s) {unknown}; use '3d', 'col1', 'col2' or 'col3'.")
    col_titles = {'3d': '3D cumulative'}
    col_titles.update({k: f'2D {_FILTER_TITLES.get(ft, ft)}' for k, (ft, _) in col_filters.items()})

    # ------------------------------------------------------------------
    # One row per simulation suite, in config order.
    # ------------------------------------------------------------------
    suites = []
    for suite in config['simulations']:
        name = suite['sim_type']
        sims = suite['sims']
        if name == 'FLAMINGO':
            fallback = matplotlib.colormaps['plasma'](np.linspace(0.2, 0.85, len(sims)))  # type: ignore
            colours = [_FLAMINGO_COLOURS.get(s['feedback'], fallback[k])
                       for k, s in enumerate(sims)]
        elif name in _COLOURMAPS:
            cmap = matplotlib.colormaps[_COLOURMAPS[name]]  # type: ignore
            colours = cmap(np.linspace(0.2, 0.85, len(sims)))
        else:
            raise ValueError(f"Unknown simulation type: {name!r}")
        suites.append((name, sims, colours))
    nRows, nCols = len(suites), len(columns)

    # ------------------------------------------------------------------
    # Create figure: plot.panel_width x plot.panel_height per panel (default
    # 6 x 4.5 in, as in the original 18 x 9 in 3x2 grid), plus a strip on the
    # right for the per-row legends.
    # ------------------------------------------------------------------
    panel_width  = plot_cfg.get('panel_width', 6.0)   # inches
    panel_height = plot_cfg.get('panel_height', 4.5)  # inches
    legend_width = 2.8  # inches
    fig_width = panel_width * nCols + legend_width
    fig, axes = plt.subplots(nRows, nCols, figsize=(fig_width, panel_height * nRows),
                             sharex='col', sharey='row', squeeze=False)

    # R200m per row, taken from the first sim processed in each suite.
    R200m_kpch_per_row = [None] * nRows
    R200m_arcmin_per_row = [None] * nRows

    # Arcmin <-> comoving kpc/h conversion functions (set after the first stacker).
    forward_arcmin  = None
    inverse_arcmin  = None

    # Largest plotted arcmin radius across all sims/2D panels; used to set a
    # shared x-limit for the 2D columns (per-sim cosmologies map 4000 ckpc/h to
    # slightly different arcmin, and sharex='col' ties each column's rows).
    max_arcmin_2d = 0.0

    t0 = time.time()

    for row_idx, (sim_type_name, sims, colours) in enumerate(suites):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Suite: {sim_type_name}  (row {row_idx})")
            print(f"{'='*60}")

        for j, sim in enumerate(sims):
            sim_name = sim['name']
            if verbose:
                feedback_str = f"  feedback={sim.get('feedback')}" if 'feedback' in sim else ''
                print(f"\n  [{j+1}/{len(sims)}] {sim_name}{feedback_str}")

            # ---- Instantiate stacker ----
            stacker, OmegaBaryon, cosmo, sim_label = setup_stacker(
                sim, sim_type_name, redshift)

            # ---- Arcmin <-> kpc/h conversion ----
            # Per-sim converters (this sim's own cosmology) convert the 2D
            # stacking range to arcmin.  The global converters (first sim)
            # drive the shared secondary top axis in configure_subplot.
            def _make_converters(c, z):
                def _fwd(arcmin): return arcmin_to_comoving(arcmin, z, c)
                def _inv(comov):  return comoving_to_arcmin(comov,  z, c)
                return _fwd, _inv
            fwd_sim, inv_sim = _make_converters(cosmo, redshift)
            if forward_arcmin is None:
                forward_arcmin, inverse_arcmin = fwd_sim, inv_sim

            R200m_kpch = None
            for col_idx, kind in enumerate(columns):
                if kind == '3d':
                    if verbose:
                        print(f"    Computing 3D profiles...")
                    # Per-simulation 3D grid size, falling back to the global one.
                    params_sim = dict(params_3d,
                                      n_pixels=int(sim.get('n_pixels', params_3d['n_pixels'])))
                    radii, ratio, err, R200m_kpch = compute_3d_profile_ratio(
                        stacker, pType, pType2, params_sim, OmegaBaryon)
                else:
                    ft, ft2 = col_filters[kind]
                    if verbose:
                        print(f"    Computing 2D profiles (filter={ft}/{ft2})...")
                    radii, ratio, err = compute_2d_profile_ratio(
                        stacker, pType, pType2, ft, ft2, params_2d, OmegaBaryon,
                        params_3d['min_radius_3d'], params_3d['max_radius_3d'],
                        params_3d['num_radii_3d'], inv_sim)
                    # Track the largest plotted arcmin radius for the shared 2D x-limit.
                    max_arcmin_2d = max(max_arcmin_2d, float(np.max(radii)))

                plot_panel(axes[row_idx, col_idx], radii, ratio, err,
                           sim_label, colours[j], plot_error_bars)
                if verbose:
                    # Plotted values, for quoting in the text. The 2D columns
                    # share the 3D column's comoving radii (converted to arcmin).
                    print(f"    [{col_titles[kind]}] R = {np.array2string(radii, precision=3)}")
                    print(f"    [{col_titles[kind]}] ratio = {np.array2string(ratio, precision=4)}")
                    print(f"    [{col_titles[kind]}] err = {np.array2string(err, precision=4)}")

            # Cache R200m (comoving and arcmin) for the vline decoration.
            if R200m_kpch is not None and R200m_kpch_per_row[row_idx] is None:
                R200m_kpch_per_row[row_idx] = R200m_kpch
                R200m_arcmin_per_row[row_idx] = comoving_to_arcmin(R200m_kpch, redshift, cosmo)

    # ------------------------------------------------------------------
    # Axis decorations
    # ------------------------------------------------------------------
    # Shared upper x-limit (arcmin) for the 2D columns, padded to avoid clipping.
    xlim_2d = max_arcmin_2d + 0.5

    panel_idx = 0
    for row_idx in range(nRows):
        for col_idx, kind in enumerate(columns):
            configure_subplot(
                ax=axes[row_idx, col_idx],
                kind=kind,
                title=col_titles[kind],
                is_top=row_idx == 0,
                is_bottom=row_idx == nRows - 1,
                is_left=col_idx == 0,
                pType=pType,
                pType2=pType2,
                R200m_kpch=R200m_kpch_per_row[row_idx],
                R200m_arcmin=R200m_arcmin_per_row[row_idx],
                forward_arcmin=forward_arcmin,
                inverse_arcmin=inverse_arcmin,
                xlim_2d=xlim_2d,
                panel_label=f'({chr(ord("a") + panel_idx)})',
            )
            panel_idx += 1

    # -----------------------------------------------------------------------
    # Figure-level labels and layout
    # -----------------------------------------------------------------------
    # Row labels placed as text on the leftmost axes so that shared-y axes do
    # not duplicate the y-label on every panel
    for row_idx, (sim_type_name, _, _) in enumerate(suites):
        axes[row_idx, 0].annotate(sim_type_name, xy=(-0.25, 0.5), xycoords='axes fraction',
                                  ha='right', va='center', rotation=90, fontsize=14,
                                  fontweight='bold')

    # Lay out the panels first, then put one legend per row (handles from its
    # rightmost panel) in the strip to the right of that row.
    fig.tight_layout(rect=[0, 0, 1 - legend_width / fig_width, 1]) # type: ignore
    for row_idx, (sim_type_name, _, _) in enumerate(suites):
        handles, labels = axes[row_idx, -1].get_legend_handles_labels()
        bbox = axes[row_idx, -1].get_position()
        fig.legend(handles, labels,
                   loc='upper left', bbox_to_anchor=(bbox.x1 + 0.01, bbox.y1),
                   frameon=True, fontsize=13, title=sim_type_name,
                   title_fontsize=14)

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    out_path = figPath / f'{figName}_{pType}.{figType}'
    fig.savefig(out_path, dpi=300) # type: ignore
    plt.close(fig)

    elapsed = (time.time() - t0) / 60
    print(f"\nFigure saved to: {out_path}")
    print(f"Total time: {elapsed:.2f} minutes")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the particle-fraction ratio figure grid.")
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/unbound_gas/ratios_3x2_z05.yaml',
        help='Path to the YAML configuration file.',
    )
    parser.add_argument(
        '--ptype',
        type=str,
        default=None,
        help='Override particle type from config (e.g. "ionized_gas").',
    )
    args = vars(parser.parse_args())
    print(f"Arguments: {args}")
    main(**args)
