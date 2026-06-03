"""make_ratios3x2.py
===================
Generate a 3×2 figure of particle-type fraction profiles, normalised by the
cosmic baryon fraction (OmegaBaryon / OmegaMatter).

Layout
------
Rows:    top = TNG suite (TNG50, TNG100, TNG300, Illustris)
         bottom = SIMBA suite
Columns: col 0 = 3D spherical profiles  (radius in comoving kpc/h)
         col 1 = 2D projected, cumulative filter  (radius in arcmin)
         col 2 = 2D projected, CAP filter          (radius in arcmin)

Usage
-----
    python make_ratios3x2.py -p configs/ratios_3x2_z05.yaml
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
from utils import arcmin_to_comoving, comoving_to_arcmin
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

# Colour maps used for TNG (col 0 of simulations list) and SIMBA (col 1).
# _COLOURMAPS = ['hsv', 'twilight']
_COLOURMAPS = ['twilight', 'hsv']

# Subplot panel labels in reading order.
_PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

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
        ``'IllustrisTNG'`` or ``'SIMBA'``.
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


def configure_subplot(ax, row_idx: int, col_idx: int,
                      pType: str, pType2: str,
                      R200m_kpch: float | None,
                      R200m_arcmin: float | None,
                      forward_arcmin, inverse_arcmin,
                      xlim_2d: float,
                      suite_name: str, panel_label: str):
    """Apply axis decorations to a single subplot panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    row_idx, col_idx : int
        Position in the 2×3 grid.
    pType, pType2 : str
        Particle type names used for y-axis label.
    R200m_kpch : float or None
        R200m (mean-overdensity radius) in comoving kpc/h for the vertical
        reference line (col 0 only).
    R200m_arcmin : float or None
        R200m (mean-overdensity radius) in arcmin for the vertical reference
        line (cols 1, 2 only).
    forward_arcmin, inverse_arcmin : callable
        Conversion functions between arcmin and comoving kpc/h, used to add
        a secondary x-axis on the top row (cols 1, 2).
    xlim_2d : float
        Upper limit for the 2D x-axis (arcmin), already scaled by
        ``rad_distance`` and padded to avoid clipping any profile.
    suite_name : str
        Suite label used in the column 0 title (ignored for cols 1, 2).
    panel_label : str
        Subplot letter, e.g. ``'(a)'``.
    """
    # --- Horizontal reference line at unity ---
    ax.axhline(1.0, color='k', ls='--', lw=2)

    # --- R200m vertical reference line (mean-overdensity radius) ---
    if col_idx == 0 and R200m_kpch is not None:
        ax.axvline(R200m_kpch, color='gray', ls=':', lw=2, label=r'$R_{200\mathrm{m}}$')
    elif col_idx > 0 and R200m_arcmin is not None:
        ax.axvline(R200m_arcmin, color='gray', ls=':', lw=2, label=r'$R_{200\mathrm{m}}$')

    # --- Axis limits ---
    ax.set_xlim(0.0, None if col_idx == 0 else xlim_2d)
    ax.grid(True)

    # --- Y axis label (left column only) ---
    if col_idx == 0:
        ax.set_ylabel(
            rf'$\frac{{\mathrm{{{pType}}}}}{{\mathrm{{{pType2}}}}} \;/\; (\Omega_b / \Omega_m)$',
            fontsize=18,
        )

    # --- X axis labels and secondary axis ---
    col_titles = ['3D cumulative', '2D cumulative', '2D CAP']
    if col_idx == 0:
        # 3D column: x in comoving kpc/h
        if row_idx == 1:
            ax.set_xlabel('R [comoving kpc/h]', fontsize=18)
            # ax.legend(loc='lower right')
        else:
            # Secondary x-axis on top row (also in comoving kpc/h, no conversion needed)
            secax = ax.secondary_xaxis('top')
            secax.set_xlabel('R [comoving kpc/h]', fontsize=18)
            # ax.set_title(f'{suite_name} — {col_titles[col_idx]}', fontsize=18)
            ax.set_title(col_titles[col_idx], fontsize=18)
    else:
        # 2D columns: x in arcmin
        if row_idx == 1:
            ax.set_xlabel('R [arcmin]', fontsize=18)
            # ax.legend(loc='lower right')
        else:
            # Add secondary x-axis in comoving kpc/h on top row.
            secax = ax.secondary_xaxis('top',
                                       functions=(forward_arcmin, inverse_arcmin))
            secax.set_xlabel('R [comoving kpc/h]', fontsize=18)
            ax.set_title(col_titles[col_idx], fontsize=18)

    # --- Column titles for top row ---
    if row_idx == 0 and col_idx > 0:
        pass  # title already set above

    # --- Subplot panel label in top-left corner ---
    ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
            fontsize=18, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))


# ===========================================================================
# Main
# ===========================================================================

def main(path2config: str, ptype: str, verbose: bool = True):
    """Generate the 3×2 particle-fraction ratio figure.

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

    # Filter types for columns 1 (cumulative) and 2 (CAP).
    ft_col1  = stack_cfg.get('filter_type_col1',   'cumulative')
    ft2_col1 = stack_cfg.get('filter_type_2_col1', 'cumulative')
    ft_col2  = stack_cfg.get('filter_type_col2',   'CAP')
    ft2_col2 = stack_cfg.get('filter_type_2_col2', 'CAP')

    # --- Plotting parameters ---
    now       = datetime.now()
    yr_string = now.strftime("%Y-%m")
    dt_string = now.strftime("%m-%d")
    figPath   = Path(plot_cfg.get('fig_path', '../figures/')) / yr_string / dt_string
    figPath.mkdir(parents=True, exist_ok=True)

    figName        = plot_cfg.get('fig_name', 'ratios_3x2')
    figType        = plot_cfg.get('fig_type', 'pdf')
    plot_error_bars = plot_cfg.get('plot_error_bars', True)

    # ------------------------------------------------------------------
    # Identify TNG and SIMBA simulation lists from config.
    # Rows: TNG = row 0, SIMBA = row 1.
    # ------------------------------------------------------------------
    tng_sims   = None
    simba_sims = None
    for suite in config['simulations']:
        if suite['sim_type'] == 'IllustrisTNG':
            tng_sims   = suite['sims']
            tng_cmap   = matplotlib.colormaps[_COLOURMAPS[0]]  # type: ignore
            tng_colours = tng_cmap(np.linspace(0.2, 0.85, len(tng_sims)))
        elif suite['sim_type'] == 'SIMBA':
            simba_sims   = suite['sims']
            simba_cmap   = matplotlib.colormaps[_COLOURMAPS[1]]  # type: ignore
            simba_colours = simba_cmap(np.linspace(0.2, 0.85, len(simba_sims)))

    if tng_sims is None or simba_sims is None:
        raise ValueError("Config must contain both 'IllustrisTNG' and 'SIMBA' simulation entries.")

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex='col', sharey='row')

    # R200m placeholders (set during the first processed sim in each suite).
    R200m_kpch_tng   = None
    R200m_kpch_simba = None
    R200m_arcmin_tng   = None
    R200m_arcmin_simba = None

    # Arcmin ↔ comoving kpc/h conversion functions (set after first TNG stacker).
    forward_arcmin  = None
    inverse_arcmin  = None

    # Largest plotted arcmin radius across all sims/2D panels; used to set a
    # shared x-limit for the 2D columns (per-sim cosmologies map 4000 ckpc/h to
    # slightly different arcmin, and sharex='col' ties each column's rows).
    max_arcmin_2d = 0.0

    t0 = time.time()

    # ------------------------------------------------------------------
    # Loop over suites: row 0 = TNG, row 1 = SIMBA
    # ------------------------------------------------------------------
    suites = [
        ('IllustrisTNG', tng_sims,   tng_colours,   0),
        ('SIMBA',        simba_sims, simba_colours, 1),
    ]

    for sim_type_name, sims, colours, row_idx in suites:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Suite: {sim_type_name}  (row {row_idx})")
            print(f"{'='*60}")

        for j, sim in enumerate(sims):
            sim_name = sim['name']
            if verbose:
                feedback_str = f"  feedback={sim.get('feedback')}" if sim_type_name == 'SIMBA' else ''
                print(f"\n  [{j+1}/{len(sims)}] {sim_name}{feedback_str}")

            # ---- Instantiate stacker ----
            stacker, OmegaBaryon, cosmo, sim_label = setup_stacker(
                sim, sim_type_name, redshift)

            # ---- Arcmin ↔ kpc/h conversion ----
            # Per-sim converters (this sim's own cosmology) convert the 2D
            # stacking range to arcmin.  The global converters (first TNG sim)
            # drive the shared secondary top axis in configure_subplot.
            def _make_converters(c, z):
                def _fwd(arcmin): return arcmin_to_comoving(arcmin, z, c)
                def _inv(comov):  return comoving_to_arcmin(comov,  z, c)
                return _fwd, _inv
            fwd_sim, inv_sim = _make_converters(cosmo, redshift)
            if forward_arcmin is None:
                forward_arcmin, inverse_arcmin = fwd_sim, inv_sim

            # ==============================================================
            # Column 0 — 3D spherical profiles
            # ==============================================================
            if verbose:
                print(f"    Computing 3D profiles...")
            radii_3d, ratio_3d, err_3d, R200m_kpch = compute_3d_profile_ratio(
                stacker, pType, pType2, params_3d, OmegaBaryon)

            # Cache R200m for vline decoration.
            if sim_type_name == 'IllustrisTNG' and R200m_kpch_tng is None:
                R200m_kpch_tng = R200m_kpch
            if sim_type_name == 'SIMBA' and R200m_kpch_simba is None:
                R200m_kpch_simba = R200m_kpch

            plot_panel(axes[row_idx, 0], radii_3d, ratio_3d, err_3d,
                       sim_label, colours[j], plot_error_bars)

            # ==============================================================
            # Column 1 — 2D cumulative profiles
            # ==============================================================
            if verbose:
                print(f"    Computing 2D cumulative profiles (filter={ft_col1}/{ft2_col1})...")
            radii_2d_cum, ratio_2d_cum, err_2d_cum = compute_2d_profile_ratio(
                stacker, pType, pType2, ft_col1, ft2_col1, params_2d, OmegaBaryon,
                params_3d['min_radius_3d'], params_3d['max_radius_3d'],
                params_3d['num_radii_3d'], inv_sim)

            # Track the largest plotted arcmin radius for the shared 2D x-limit.
            max_arcmin_2d = max(max_arcmin_2d, float(np.max(radii_2d_cum)))

            # Cache R200m in arcmin.
            if sim_type_name == 'IllustrisTNG' and R200m_arcmin_tng is None:
                R200m_arcmin_tng = comoving_to_arcmin(R200m_kpch, redshift, cosmo)
            if sim_type_name == 'SIMBA' and R200m_arcmin_simba is None:
                R200m_arcmin_simba = comoving_to_arcmin(R200m_kpch, redshift, cosmo)

            plot_panel(axes[row_idx, 1], radii_2d_cum, ratio_2d_cum, err_2d_cum,
                       sim_label, colours[j], plot_error_bars)

            # ==============================================================
            # Column 2 — 2D CAP profiles
            # ==============================================================
            if verbose:
                print(f"    Computing 2D CAP profiles (filter={ft_col2}/{ft2_col2})...")
            radii_2d_cap, ratio_2d_cap, err_2d_cap = compute_2d_profile_ratio(
                stacker, pType, pType2, ft_col2, ft2_col2, params_2d, OmegaBaryon,
                params_3d['min_radius_3d'], params_3d['max_radius_3d'],
                params_3d['num_radii_3d'], inv_sim)

            # Track the largest plotted arcmin radius for the shared 2D x-limit.
            max_arcmin_2d = max(max_arcmin_2d, float(np.max(radii_2d_cap)))

            plot_panel(axes[row_idx, 2], radii_2d_cap, ratio_2d_cap, err_2d_cap,
                       sim_label, colours[j], plot_error_bars)

    # ------------------------------------------------------------------
    # Axis decorations
    # ------------------------------------------------------------------
    suite_names = ['IllustrisTNG', 'SIMBA']
    R200m_kpch_per_row   = [R200m_kpch_tng,   R200m_kpch_simba]
    R200m_arcmin_per_row = [R200m_arcmin_tng, R200m_arcmin_simba]

    # Shared upper x-limit (arcmin) for the 2D columns, padded to avoid clipping.
    xlim_2d = max_arcmin_2d + 0.5

    panel_idx = 0
    for row_idx, suite_name in enumerate(suite_names):
        for col_idx in range(3):
            configure_subplot(
                ax=axes[row_idx, col_idx],
                row_idx=row_idx,
                col_idx=col_idx,
                pType=pType,
                pType2=pType2,
                R200m_kpch=R200m_kpch_per_row[row_idx],
                R200m_arcmin=R200m_arcmin_per_row[row_idx],
                forward_arcmin=forward_arcmin,
                inverse_arcmin=inverse_arcmin,
                xlim_2d=xlim_2d,
                suite_name=suite_name,
                panel_label=_PANEL_LABELS[panel_idx],
            )
            panel_idx += 1

    # -----------------------------------------------------------------------
    # Figure-level labels and layout
    # -----------------------------------------------------------------------
    # Row labels placed as text on the leftmost axes so that shared-y axes do
    # not duplicate the y-label on every panel
    axes[0, 0].annotate('IllustrisTNG', xy=(-0.25, 0.5), xycoords='axes fraction',
                        ha='right', va='center', rotation=90, fontsize=14,
                        fontweight='bold')
    axes[1, 0].annotate('SIMBA', xy=(-0.25, 0.5), xycoords='axes fraction',
                        ha='right', va='center', rotation=90, fontsize=14,
                        fontweight='bold')

    # -----------------------------------------------------------------------
    # Legends positioned to the right of the figure
    # -----------------------------------------------------------------------
    # Collect handles and labels from the rightmost column for each row
    handles_tng, labels_tng = axes[0, 2].get_legend_handles_labels()
    handles_simba, labels_simba = axes[1, 2].get_legend_handles_labels()
    
    # Create legends on the right side: TNG on top, SIMBA on bottom
    legend_tng = fig.legend(handles_tng, labels_tng, 
                           loc='upper left', bbox_to_anchor=(0.89, 0.88),
                           frameon=True, fontsize=13, title='IllustrisTNG',
                           title_fontsize=14)
    legend_simba = fig.legend(handles_simba, labels_simba,
                             loc='upper left', bbox_to_anchor=(0.873, 0.48),
                             frameon=True, fontsize=13, title='SIMBA',
                             title_fontsize=14)

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    fig.tight_layout(rect=[0, 0, 0.90, 1]) # type: ignore
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
        description="Generate 3×2 particle-fraction ratio figure.")
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/ratios_3x2_z05.yaml',
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
