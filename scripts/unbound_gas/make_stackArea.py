"""
make_stackArea.py
=================
Produces two cumulative baryon-fraction stacked-area figures, one per stacking
method, each laid out as ``n_rows × n_cols`` panels (one per simulation):

    Figure 1 — 3D stacking: mass enclosed in spheres of radius r, summed
               directly over 3-D density fields.  X-axis in comoving kpc/h.
    Figure 2 — 2D stacking: filtered radial profiles of projected maps
               (``filter_type``, e.g. 'DSigma' or 'CAP').  X-axis in arcmin;
               secondary top axis in comoving kpc/h.

Panels are filled **column-major**, so with six simulations in two columns the
first three fill the left column and the last three the right -- for
stackArea_dsigma_z05.yaml, TNG300-1 / Illustris-1 / SIMBA m100n1024 on the left
and the three FLAMINGO variants on the right, the same panel positions as
Figure 5 (make_baryonFraction.py).

Each panel shows a stacked-area plot of the mean baryon-type profiles, each
divided by the total-matter profile and by the cosmic baryon fraction
Omega_b / Omega_m.  A halo holding exactly its cosmic share of baryons reaches
the dashed line at y = 1.  Unlike make_baryonFraction.py, the denominator is
the total matter field, so the stacked areas need not sum to 1.

With ``filter_type: 'DSigma'`` the 2D panels show
Delta Sigma_i / Delta Sigma_tot / (Omega_b / Omega_m): each component's share of
the excess surface density at R, not the mass fraction within R.  A component
whose stacked profile rises with R has Delta Sigma_i < 0, which a stacked area
cannot show; the script emits a RuntimeWarning naming any such component and
radius.

Usage
-----
    python unbound_gas/make_stackArea.py -p ./configs/unbound_gas/stackArea_dsigma_z05.yaml

The pre-FLAMINGO version (one 2x3 figure, 3D row above 2D row, CAP filter) is
frozen as archive/make_stackArea_v1.py; run it with
configs/unbound_gas/stackArea_z05.yaml to reproduce the old Figure 4.

Config file format
------------------
See configs/unbound_gas/stackArea_dsigma_z05.yaml for an annotated example and
``main`` for the full list of keys.  The top-level keys are ``stack``, ``plot``
and ``simulations`` (a flat list; SIMBA and FLAMINGO entries need ``feedback``).
A simulation entry may carry its own ``n_pixels``, overriding ``stack.n_pixels``
for its 3-D grid.

Resolution caveat
-----------------
The FLAMINGO L1_m9 box is 681,000 ckpc/h, so a 1000^3 grid gives 681 ckpc/h
voxels and a 2000^3 grid 340 ckpc/h, against a ~222 ckpc/h radial step for the
default radial grid.  The innermost FLAMINGO points of the 3D figure are
therefore voxel-limited (a RuntimeWarning says so); the 2D figure is unaffected.

Dependencies
------------
* SimulationStacker src/ package (stacker, halos, mask_utils, utils)
* make_baryonFraction.py in this directory (neutral_gas derivation helpers)
* illustris_python (on sys.path one level up)
* astropy, matplotlib, numpy, yaml
"""

import sys
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import cast

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import yaml
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# ---------------------------------------------------------------------------
# Internal package imports
# ---------------------------------------------------------------------------
sys.path.append('../src/')
from utils import arcmin_to_comoving, comoving_to_arcmin
from stacker import SimulationStacker
from halos import select_massive_halos
from mask_utils import get_cutout_indices_3d, sum_over_cutouts

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore

# neutral_gas = gas - ionized_gas, shared with Figure 5 so the two figures
# derive it identically.  Found via sys.path[0], this script's own directory.
from make_baryonFraction import resolve_stack_types, assemble_components

# ---------------------------------------------------------------------------
# Global matplotlib style — Computer Modern / LaTeX-compatible fonts
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

_YLABEL_3D = r'$M_i(<r) \, / \, M_{\rm tot}(<r) \, / \, (\Omega_b / \Omega_m)$'

# 2D y-labels, keyed by filter_type.  Used only when the numerator and the
# denominator share a filter; otherwise the generic label below applies.
_YLABEL_2D = {
    'DSigma':     r'$\Delta\Sigma_i \, / \, \Delta\Sigma_{\rm tot} \, / \, (\Omega_b / \Omega_m)$',
    'CAP':        r'${\rm CAP}_i \, / \, {\rm CAP}_{\rm tot} \, / \, (\Omega_b / \Omega_m)$',
    'cumulative': r'$M_i(<R) \, / \, M_{\rm tot}(<R) \, / \, (\Omega_b / \Omega_m)$',
}
_YLABEL_GENERIC = r'Baryon fraction $/ \, (\Omega_b / \Omega_m)$'

_TITLE_2D = {
    'DSigma':     r'$\Delta\Sigma$-filtered Baryon Fractions (2D)',
    'CAP':        'CAP-filtered Baryon Fractions (2D)',
    'cumulative': 'Cumulative Baryon Fractions (2D, disks)',
}

# Fractions below -_NEG_TOL are reported as genuinely negative rather than
# round-off (the derived neutral_gas carries float32 cancellation error of
# order 1e-7).
_NEG_TOL = 1e-6

# ---------------------------------------------------------------------------
# Helper: create a SimulationStacker and associated cosmology
# ---------------------------------------------------------------------------

def make_stacker(sim: dict, redshift: float):
    """Initialise a SimulationStacker for the given simulation entry and return
    the stacker together with its FlatLambdaCDM cosmology and OmegaBaryon.

    Parameters
    ----------
    sim : dict
        A single entry from the ``simulations`` list in the config file.
        Required keys: ``sim_type``, ``name``, ``snapshot``.
        SIMBA and FLAMINGO entries also require ``feedback``; for FLAMINGO it
        holds the variant directory name ('L1_m9' = fiducial).
    redshift : float
        Target redshift (used for cosmology and stacker initialisation).

    Returns
    -------
    stacker : SimulationStacker
    cosmo   : FlatLambdaCDM
    OmegaBaryon : float
    sim_label   : str  — human-readable label for the panels
    """
    sim_type = sim['sim_type']
    sim_name = sim['name']
    snapshot = sim['snapshot']

    if sim_type == 'IllustrisTNG':
        stacker = SimulationStacker(sim_name, snapshot, z=redshift, simType=sim_type)
        try:
            OmegaBaryon = stacker.header['OmegaBaryon']
        except KeyError:
            # Illustris-1 does not store OmegaBaryon; use the standard value
            OmegaBaryon = 0.0456
        sim_label = sim_name

    elif sim_type == 'SIMBA':
        feedback = sim['feedback']
        stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                    simType=sim_type, feedback=feedback)
        OmegaBaryon = 0.048  # Standard value for SIMBA runs
        # sim_label = f"{sim_name}_{feedback}"
        sim_label = "SIMBA-m100"

    elif sim_type == 'FLAMINGO':
        # feedback holds the FLAMINGO variant directory name ('L1_m9' = fiducial).
        feedback = sim['feedback']
        stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                    simType=sim_type, feedback=feedback)
        # '-' instead of '_' so the label renders under usetex (as in Figure 5).
        sim_label = f"FLAMINGO {feedback}".replace('_', '-')
        # load_flamingo_header maps the SWIFT cosmology onto TNG-style keys.
        # Its Omega0 is Omega_cdm + Omega_b with neutrinos excluded, matching the
        # 'total' field (gas + DM + Stars + BH), so Omega_b / Omega_m is
        # consistent with the denominator.
        OmegaBaryon = stacker.header['OmegaBaryon']

    else:
        raise ValueError(f"Unknown sim_type '{sim_type}'.  "
                         "Supported: 'IllustrisTNG', 'SIMBA', 'FLAMINGO'.")

    cosmo = FlatLambdaCDM(
        H0=100 * stacker.header['HubbleParam'],
        Om0=stacker.header['Omega0'],
        Tcmb0=2.7255 * u.K,
        Ob0=OmegaBaryon,
    )
    return stacker, cosmo, OmegaBaryon, sim_label


# ---------------------------------------------------------------------------
# Helper: flag negative components
# ---------------------------------------------------------------------------

def warn_negative_fractions(fractions, labels, radii, where):
    """Warn about components whose stacked fraction is negative.

    A stacked area cannot represent a negative layer: ``stackplot`` draws it
    downward from the running total, over the layers beneath, with no visual
    cue.  Delta Sigma components go negative wherever their stacked profile
    rises with R, so report them rather than let the figure mislead.

    Parameters
    ----------
    fractions : list[numpy.ndarray]
        One profile per component, as passed to ``stackplot``.
    labels : list[str]
        Component names, in the order of ``fractions``.
    radii : numpy.ndarray
        Radii of the profile points, in the plotted x-axis units.
    where : str
        Panel identifier for the message, e.g. ``'TNG300-1 (2D DSigma)'``.

    Warns
    -----
    RuntimeWarning
        Once per component that falls below ``-_NEG_TOL`` at any radius.
    """
    for label, frac in zip(labels, fractions):
        bad = np.flatnonzero(frac < -_NEG_TOL)
        if bad.size:
            warnings.warn(
                f"{where}: '{label}' is negative at R = "
                f"{np.round(radii[bad], 3).tolist()} (min {frac[bad].min():.3g}); "
                "the stacked area misrepresents it there.",
                RuntimeWarning, stacklevel=2,
            )


# ---------------------------------------------------------------------------
# 3-D stacking: radial profiles from 3-D density fields
# ---------------------------------------------------------------------------

def run_3d_stacking(stacker, OmegaBaryon, baryon_types, pType2,
                    nPixels, minRadius, maxRadius, nRadii,
                    projection, saveField, loadField,
                    ax, colours, radDistance,
                    halo_mass_avg=10**13.22, halo_mass_upper=5e14,
                    derive_neutral_gas=True, sim_label='', verbose=True):
    """Build 3-D density fields, cut out spheres around massive haloes, and plot
    the resulting baryon-fraction stacked-area profile.

    The radii are in comoving kpc/h (the native 3-D field unit).  A sphere of
    radius r is grown from each halo centre and the enclosed mass summed via
    ``get_cutout_indices_3d`` / ``sum_over_cutouts``.  A radius of exactly 0 is
    dropped: a zero-radius sphere collapses to the single central voxel, which
    is non-informative and misleading for a cumulative profile.

    Parameters
    ----------
    stacker    : SimulationStacker
    OmegaBaryon : float
    baryon_types : list[str]  — e.g. ['ionized_gas', 'neutral_gas', 'Stars', 'BH']
    pType2     : str          — particle type used as the total-mass denominator
    nPixels    : int          — grid resolution for 3-D fields
    minRadius  : float        — minimum stacking radius [comoving kpc/h]
    maxRadius  : float        — maximum stacking radius [comoving kpc/h]
    nRadii     : int          — number of radial bins
    projection : str          — projection axis for field ('xy', 'xz', 'yz') — passed
                                through to makeField (the 3-D field is not projected)
    saveField  : bool         — cache fields to disk
    loadField  : bool         — try to load cached fields from disk
    ax         : matplotlib Axes
    colours    : array-like   — one colour per baryon type
    radDistance : float       — multiplicative scaling applied to radii for x-axis
    halo_mass_avg : float     — target average halo mass [M_sun/h] for the 'massive'
                                selection.  Default 10**13.22.
    halo_mass_upper : float   — upper halo-mass bound [M_sun/h] for the same
                                selection.  Default 5e14.
    derive_neutral_gas : bool — if True, stack ``gas`` in place of ``neutral_gas``
                                and subtract ``ionized_gas`` afterwards (see
                                make_baryonFraction.resolve_stack_types).
                                Default True.
    sim_label  : str          — identifies the panel in warnings.
    verbose    : bool

    Warns
    -----
    RuntimeWarning
        If the voxel size exceeds the radial step (the inner radii are then
        voxel-limited), or if any component fraction is negative.
    """
    if not baryon_types:
        raise ValueError("baryon_types must contain at least one component.")

    stack_types = resolve_stack_types(baryon_types, derive_neutral_gas)

    # Build the 3-D field for each stacked particle type (numerators)
    baryon_fields = {}
    for pt in stack_types:
        if verbose:
            print(f"  Building 3D field: {pt}")
        baryon_fields[pt] = stacker.makeField(pt, nPixels=nPixels, dim='3D',
                                              projection=projection,
                                              save=saveField, load=loadField)

    # Build the total-mass 3-D field (common denominator for all baryon fractions)
    if verbose:
        print(f"  Building 3D field: {pType2} (total)")
    field_total = stacker.makeField(pType2, nPixels=nPixels, dim='3D',
                                    projection=projection,
                                    save=saveField, load=loadField)

    # Physical size of a single voxel [comoving kpc/h per pixel]
    kpcPerPixel = stacker.header['BoxSize'] / field_total.shape[0]
    if verbose:
        print(f"  kpcPerPixel = {kpcPerPixel:.3f}")

    # Surface an unresolved radial grid rather than letting it pass silently:
    # when the radial step is below a voxel, consecutive radii select the same
    # voxels and the innermost points carry no independent information.
    bin_width = (maxRadius - minRadius) / max(nRadii - 1, 1)
    if kpcPerPixel > bin_width:
        warnings.warn(
            f"3D voxel size ({kpcPerPixel:.0f} ckpc/h) exceeds the radial step "
            f"({bin_width:.0f} ckpc/h) for {sim_label or stacker.sim}: the inner "
            "radii are voxel-limited. Increase n_pixels or widen the radial grid.",
            RuntimeWarning, stacklevel=2,
        )

    # Load haloes and select massive ones (config-driven mass cuts)
    haloes = stacker.loadHalos()
    haloMass = haloes['GroupMass']
    halo_mask = select_massive_halos(haloMass, halo_mass_avg, halo_mass_upper)

    haloes['GroupMass'] = haloes['GroupMass'][halo_mask]
    haloes['GroupRad'] = haloes['GroupRad'][halo_mask]  # R200c [comoving kpc/h]
    GroupPos_px = np.round(haloes['GroupPos'][halo_mask] / kpcPerPixel).astype(int) % nPixels
    n_haloes = len(haloes['GroupMass'])

    if verbose:
        print(f"  Number of selected haloes: {n_haloes}")

    # Linearly-spaced radii at which to evaluate the stacked profiles.
    # Drop r=0: a zero-radius sphere collapses to the central voxel and is
    # non-informative/misleading for a cumulative profile.
    radii = np.linspace(minRadius, maxRadius, nRadii)  # [comoving kpc/h]
    radii = radii[radii > 0]

    # Accumulate profile arrays: shape will be (nRadii, nHalos) after stacking
    profiles_stacked = {pt: [] for pt in stack_types}
    profiles_total = []

    t0 = time.time()
    for r in radii:
        # Cutout radius converted to pixels (same for all haloes at a given r)
        rr = np.ones(n_haloes) * r / kpcPerPixel
        mask_indices = get_cutout_indices_3d(field_total, GroupPos_px, rr)
        for pt in stack_types:
            profiles_stacked[pt].append(sum_over_cutouts(baryon_fields[pt], mask_indices))
        profiles_total.append(sum_over_cutouts(field_total, mask_indices))
        if verbose:
            print(f"    r={r:.0f} kpc/h  elapsed={time.time()-t0:.1f}s")

    # Convert profile lists to arrays: (nRadii, nHalos)
    for pt in stack_types:
        profiles_stacked[pt] = np.array(profiles_stacked[pt]) # type: ignore
    profiles_total = np.array(profiles_total)

    # gas -> neutral_gas by subtraction (no-op when derive_neutral_gas is False)
    profiles_baryon = assemble_components(profiles_stacked, baryon_types, derive_neutral_gas)

    # Mean over haloes at each radius
    mean_total = np.mean(profiles_total, axis=1)  # (nRadii,)

    # Baryon fractions normalised by the cosmic baryon fraction so that a
    # uniformly baryon-traced field returns 1 everywhere
    fractions = []
    bt_labels = []
    for bt in baryon_types:
        mean_bt = np.mean(profiles_baryon[bt], axis=1)
        fractions.append(mean_bt / mean_total / (OmegaBaryon / stacker.header['Omega0']))
        bt_labels.append(bt)

    warn_negative_fractions(fractions, bt_labels, radii * radDistance,
                            f'{sim_label} (3D)')

    # Draw the stacked-area plot on the provided axis
    ax.stackplot(radii * radDistance, fractions, labels=bt_labels, alpha=0.8, colors=colours)


# ---------------------------------------------------------------------------
# 2-D stacking: radial profiles from projected 2-D maps
# ---------------------------------------------------------------------------

def run_2d_stacking(stacker, cosmo, OmegaBaryon, baryon_types, pType2,
                    filterType, filterType2, minRadius, maxRadius, nRadii,
                    projection, saveField, loadField, radDistance,
                    ax, colours, forward_arcmin, inverse_arcmin,
                    halo_mass_avg=10**13.22, halo_mass_upper=5e14,
                    pixelSize=0.5, beamSize=1.6, derive_neutral_gas=True,
                    sim_label='', verbose=True):
    """Stack 2-D projected maps and plot the resulting baryon-fraction profile.

    The stacking radii are expressed in arcmin (converted from the comoving kpc/h
    values in the config using the simulation cosmology).  The x-axis of the
    returned plot is in arcmin; the caller is responsible for adding a secondary
    comoving-kpc/h axis via ``forward_arcmin`` / ``inverse_arcmin``.  An r = 0
    radius (0 arcmin) is dropped from the plotted profile, mirroring the 3-D path.

    Parameters
    ----------
    stacker     : SimulationStacker
    cosmo       : FlatLambdaCDM  — simulation cosmology (used implicitly via
                                   forward_arcmin / inverse_arcmin passed in)
    OmegaBaryon : float
    baryon_types : list[str]
    pType2      : str           — particle type for the total-mass denominator map
    filterType  : str           — filter applied to baryon maps (e.g. 'DSigma', 'CAP')
    filterType2 : str           — filter applied to total-mass map
    minRadius   : float         — minimum radius [comoving kpc/h], converted to arcmin
    maxRadius   : float         — maximum radius [comoving kpc/h], converted to arcmin
    nRadii      : int
    projection  : str
    saveField   : bool
    loadField   : bool
    radDistance : float         — multiplicative scaling for the x-axis display
    ax          : matplotlib Axes
    colours     : array-like
    forward_arcmin  : callable  — arcmin → comoving kpc/h (for secondary axis)
    inverse_arcmin  : callable  — comoving kpc/h → arcmin (for secondary axis)
    halo_mass_avg : float       — target average halo mass [M_sun/h] forwarded to
                                  stackMap's 'massive' selection.  Default 10**13.22.
    halo_mass_upper : float     — upper halo-mass bound [M_sun/h] for the same
                                  selection.  Default 5e14.
    pixelSize   : float         — map pixel size [arcmin].  Default 0.5.
    beamSize    : float or None — Gaussian beam FWHM [arcmin]; 0 or None loads the
                                  raw (unconvolved) cached field.  Default 1.6.
    derive_neutral_gas : bool   — if True, stack ``gas`` in place of ``neutral_gas``
                                  and subtract ``ionized_gas`` afterwards.
                                  Default True.
    sim_label   : str           — identifies the panel in warnings.
    verbose     : bool
    """
    if not baryon_types:
        raise ValueError("baryon_types must contain at least one component.")

    # Convert stacking radii from comoving kpc/h → arcmin using the sim cosmology
    minRadius_arcmin = inverse_arcmin(minRadius)
    maxRadius_arcmin = inverse_arcmin(maxRadius)

    stack_types = resolve_stack_types(baryon_types, derive_neutral_gas)

    if verbose:
        print(f"  2D stacking: {minRadius_arcmin:.2f} – {maxRadius_arcmin:.2f} arcmin "
              f"(filter '{filterType}' / '{filterType2}', pixel {pixelSize} arcmin, "
              f"beam {beamSize})")

    # Stack the total-mass map (denominator)
    radii1, profiles_total = stacker.stackMap(
        pType2, filterType=filterType2,
        minRadius=minRadius_arcmin, maxRadius=maxRadius_arcmin, numRadii=nRadii,
        save=saveField, load=loadField, radDistance=radDistance,
        projection=projection,
        pixelSize=pixelSize, beamSize=beamSize,
        halo_mass_avg=halo_mass_avg, halo_mass_upper=halo_mass_upper,
    )

    # Stack each particle-type map (numerators)
    profiles_stacked = {}
    for pt in stack_types:
        if verbose:
            print(f"  Stacking 2D map: {pt}")
        t1 = time.time()
        radii0, profiles_stacked[pt] = stacker.stackMap(
            pt, filterType=filterType,
            minRadius=minRadius_arcmin, maxRadius=maxRadius_arcmin, numRadii=nRadii,
            save=saveField, load=loadField, radDistance=radDistance,
            projection=projection,
            pixelSize=pixelSize, beamSize=beamSize,
            halo_mass_avg=halo_mass_avg, halo_mass_upper=halo_mass_upper,
        )
        if verbose:
            print(f"    done in {time.time()-t1:.1f}s")

    # gas -> neutral_gas by subtraction (no-op when derive_neutral_gas is False)
    profiles_baryon = assemble_components(profiles_stacked, baryon_types, derive_neutral_gas)

    # Mean over haloes
    mean_total = np.mean(profiles_total, axis=1)  # (nRadii,)

    fractions = []
    bt_labels = []
    for bt in baryon_types:
        mean_bt = np.mean(profiles_baryon[bt], axis=1)
        fractions.append(mean_bt / mean_total / (OmegaBaryon / stacker.header['Omega0']))
        bt_labels.append(bt)

    # Drop r=0 (0 arcmin) so both panels start at the same physical radius;
    # inverse_arcmin(0) is exactly 0, so this removes the innermost point only
    # when min_radius == 0.
    keep = radii0 > 0
    fractions = [frac[keep] for frac in fractions]

    warn_negative_fractions(fractions, bt_labels, radii0[keep] * radDistance,
                            f'{sim_label} (2D {filterType})')

    # x-axis: radii in arcmin scaled by radDistance
    ax.stackplot(radii0[keep] * radDistance, fractions, labels=bt_labels, alpha=0.8, colors=colours)

    # Store the final arcmin radius so the caller can set xlim
    return maxRadius_arcmin


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _label_panel(ax, sim_label):
    """Put the simulation name in a tight box at the lower-left of a panel."""
    ax.text(0.03, 0.05, sim_label, transform=ax.transAxes, fontsize=18,
            va='bottom', ha='left',
            bbox=dict(boxstyle='square,pad=0.1', facecolor='white',
                      edgecolor='gray', alpha=0.85))


def main(path2config: str, verbose: bool = True):
    """Load config, run 3-D and 2-D stacking for each simulation, and save two
    figures (one for 3D stacking, one for 2D stacking).

    Panels are laid out on an ``n_rows × n_cols`` grid filled column-major, so
    six simulations in two columns put the first three on the left and the last
    three on the right.

    Parameters
    ----------
    path2config : str
        Path to the YAML configuration file.
    verbose : bool
        If True, print progress messages during stacking.

    Config keys under ``stack:``
    ---------------------------
    redshift, load_field, save_field, projection, rad_distance
        As in the other unbound-gas scripts.
    min_radius, max_radius, num_radii : float, float, int
        Linear radial grid [comoving kpc/h]; converted to arcmin for 2D.
    n_pixels : int, default 1000
        3-D grid size per side.  A simulation entry's own ``n_pixels``
        overrides it for that simulation.
    baryon_types : list[str], default ['ionized_gas', 'neutral_gas', 'Stars', 'BH']
        Numerator components, bottom to top in the stack.
    particle_type_2 : str, default 'total'
        Denominator particle type.
    filter_type, filter_type_2 : str, default 'CAP'
        2D filter for the numerators and for the denominator.
    pixel_size : float, default 0.5
        2D map pixel size [arcmin].
    beam_size : float or None, default 1.6
        2D Gaussian beam FWHM [arcmin]; 0 or null loads the raw cached fields.
    derive_neutral_gas : bool, default True
        Obtain ``neutral_gas`` as ``gas - ionized_gas`` (equal to float32
        round-off, see make_baryonFraction.resolve_stack_types) instead of
        stacking the ``neutral_gas`` field.
    halo_mass_avg : float, default 10**13.22
        Target average halo mass [M_sun/h] for the 'massive' selection,
        applied identically to the 3-D and 2-D stacks.
    halo_mass_upper : float, default 5e14
        Upper halo-mass bound [M_sun/h] for the same selection.

    Config keys under ``plot:``
    ---------------------------
    fig_path, fig_name, fig_type
        Output location; figures go to ``fig_path/YYYY-MM/MM-DD/``.
    n_cols : int, default 2 when more than three simulations are listed, else 1
        Number of panel columns; rows follow from the simulation count.
    """

    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_config = config.get('stack', {})
    plot_config  = config.get('plot', {})

    # -----------------------------------------------------------------------
    # Read stacking parameters (shared between 3-D and 2-D)
    # -----------------------------------------------------------------------
    redshift    = stack_config.get('redshift', 0.5)
    loadField   = stack_config.get('load_field', True)
    saveField   = stack_config.get('save_field', True)
    radDistance = stack_config.get('rad_distance', 1.0)
    baryon_types = stack_config.get('baryon_types', ['ionized_gas', 'neutral_gas', 'Stars', 'BH'])
    projection  = stack_config.get('projection', 'yz')
    # 2-D map geometry. Both keys used to be read but never passed on, so
    # stackMap's defaults (0.5 arcmin pixels, 1.6 arcmin beam) applied whatever
    # the config said; the defaults here keep old configs on those values.
    pixelSize   = float(stack_config.get('pixel_size', 0.5))   # arcmin
    beamSize    = stack_config.get('beam_size', 1.6)           # arcmin; 0/null -> no beam
    beamSize    = None if beamSize is None else float(beamSize)
    derive_neutral_gas = stack_config.get('derive_neutral_gas', True)

    filterType  = stack_config.get('filter_type', 'CAP')    # applied to baryon maps
    filterType2 = stack_config.get('filter_type_2', 'CAP')  # applied to total map
    pType2      = stack_config.get('particle_type_2', 'total')
    # The two keys default independently, so a config that sets only
    # filter_type would silently divide by a CAP-filtered denominator.
    if filterType != filterType2:
        warnings.warn(
            f"filter_type '{filterType}' differs from filter_type_2 '{filterType2}': "
            "the 2D panels divide differently filtered profiles, which has no clean "
            "interpretation.", RuntimeWarning, stacklevel=2,
        )

    minRadius   = stack_config.get('min_radius', 200.0)  # comoving kpc/h
    maxRadius   = stack_config.get('max_radius', 6000.0)
    nRadii      = stack_config.get('num_radii', 15)
    nPixels     = stack_config.get('n_pixels', 1000)     # pixels per side for 3-D fields

    # Cast to float: PyYAML parses unsigned-exponent literals (e.g. '5.0e14')
    # as strings, which would crash deep in the halo-selection comparison.
    halo_mass_avg   = float(stack_config.get('halo_mass_avg', 10**13.22))   # M_sun/h
    halo_mass_upper = float(stack_config.get('halo_mass_upper', 5e14))      # M_sun/h

    # -----------------------------------------------------------------------
    # Read plotting parameters
    # -----------------------------------------------------------------------
    now       = datetime.now()
    yr_string = now.strftime("%Y-%m")
    dt_string = now.strftime("%m-%d")

    figPath = Path(plot_config.get('fig_path', '../figures/')) / yr_string / dt_string
    figPath.mkdir(parents=True, exist_ok=True)
    figName = plot_config.get('fig_name', 'combined')
    figType = plot_config.get('fig_type', 'pdf')

    # Colour palette: one colour per baryon type, drawn from the 'plasma' colourmap
    colourmap = matplotlib.colormaps['plasma']  # type: ignore
    colours   = colourmap(np.linspace(0.0, 0.8, len(baryon_types)))

    sims = config['simulations']
    n_sims = len(sims)

    # -----------------------------------------------------------------------
    # Panel grid: n_rows × n_cols, filled column-major so that each column is a
    # contiguous block of the config's simulation list (e.g. the three FLAMINGO
    # variants together in the right-hand column), as in Figure 5.
    # -----------------------------------------------------------------------
    n_cols = int(plot_config.get('n_cols', 2 if n_sims > 3 else 1))
    n_rows = int(np.ceil(n_sims / n_cols))

    # sharex=True on the 3D figure: every panel spans the same comoving range.
    # The 2D figure deliberately does not share x — the arcmin extent of a fixed
    # comoving radius differs slightly between simulations' cosmologies, so each
    # panel keeps its own limits and its own secondary axis.
    fig_3d, _axes_3d = plt.subplots(n_rows, n_cols, figsize=(9, 9), sharey=True, sharex=True)
    fig_2d, _axes_2d = plt.subplots(n_rows, n_cols, figsize=(9, 9), sharey=True)
    # reshape rather than list(): plt.subplots collapses singleton dimensions,
    # so a 1-column or 1-row grid would otherwise not be indexable as [r, c].
    axes_3d = np.asarray(_axes_3d, dtype=object).reshape(n_rows, n_cols)
    axes_2d = np.asarray(_axes_2d, dtype=object).reshape(n_rows, n_cols)

    # Blank any unused cells (e.g. 5 simulations on a 3x2 grid) so an empty
    # frame does not masquerade as a panel with no data.
    for idx in range(n_sims, n_rows * n_cols):
        axes_3d[idx % n_rows, idx // n_rows].set_visible(False)
        axes_2d[idx % n_rows, idx // n_rows].set_visible(False)

    t_total = time.time()

    for idx, sim in enumerate(sims):
        row, col = idx % n_rows, idx // n_rows   # column-major fill
        sim_type = sim['sim_type']
        sim_name = sim['name']
        if verbose:
            print(f"\n=== Processing simulation [{idx+1}/{n_sims}]: {sim_name} ({sim_type}) ===")

        # Build the shared stacker and cosmology for this simulation
        stacker, cosmo, OmegaBaryon, sim_label = make_stacker(sim, redshift)

        # 3-D grid: a simulation entry may override stack.n_pixels, since the
        # FLAMINGO box is ~3x TNG300-1's and needs a finer grid for comparable voxels.
        sim_nPixels = int(sim.get('n_pixels', nPixels))

        # Arcmin ↔ comoving kpc/h conversion functions (sim-specific cosmology)
        def forward_arcmin(arcmin, _redshift=redshift, _cosmo=cosmo):
            return arcmin_to_comoving(arcmin, _redshift, _cosmo)

        def inverse_arcmin(comoving, _redshift=redshift, _cosmo=cosmo):
            return comoving_to_arcmin(comoving, _redshift, _cosmo)

        # -------------------------------------------------------------------
        # 3-D stacking
        # -------------------------------------------------------------------
        ax_3d = cast(Axes, axes_3d[row, col])
        if verbose:
            print(f"  [3D] starting (n_pixels = {sim_nPixels}) ...")
        run_3d_stacking(
            stacker=stacker,
            OmegaBaryon=OmegaBaryon,
            baryon_types=baryon_types,
            pType2=pType2,
            nPixels=sim_nPixels,
            minRadius=minRadius,
            maxRadius=maxRadius,
            nRadii=nRadii,
            projection=projection,
            saveField=saveField,
            loadField=loadField,
            ax=ax_3d,
            colours=colours,
            radDistance=radDistance,
            halo_mass_avg=halo_mass_avg,
            halo_mass_upper=halo_mass_upper,
            derive_neutral_gas=derive_neutral_gas,
            sim_label=sim_label,
            verbose=verbose,
        )
        # Style: 3-D subplot
        ax_3d.axhline(1.0, color='k', ls='--', lw=2)
        ax_3d.set_xlim(0.0, maxRadius * radDistance)
        ax_3d.grid(True)
        # Bottom ticks on every panel but no labels; the labels go on the top of
        # the top row and the bottom of the bottom row (set after the loop).
        ax_3d.tick_params(axis='x', bottom=True, labelbottom=False, top=True, labeltop=False)
        _label_panel(ax_3d, sim_label)

        # -------------------------------------------------------------------
        # 2-D map stacking
        # -------------------------------------------------------------------
        ax_2d = cast(Axes, axes_2d[row, col])
        if verbose:
            print("  [2D] starting ...")
        maxRadius_arcmin = run_2d_stacking(
            stacker=stacker,
            cosmo=cosmo,
            OmegaBaryon=OmegaBaryon,
            baryon_types=baryon_types,
            pType2=pType2,
            filterType=filterType,
            filterType2=filterType2,
            minRadius=minRadius,
            maxRadius=maxRadius,
            nRadii=nRadii,
            projection=projection,
            saveField=saveField,
            loadField=loadField,
            radDistance=radDistance,
            ax=ax_2d,
            colours=colours,
            forward_arcmin=forward_arcmin,
            inverse_arcmin=inverse_arcmin,
            halo_mass_avg=halo_mass_avg,
            halo_mass_upper=halo_mass_upper,
            pixelSize=pixelSize,
            beamSize=beamSize,
            derive_neutral_gas=derive_neutral_gas,
            sim_label=sim_label,
            verbose=verbose,
        )
        # Style: 2-D subplot
        ax_2d.axhline(1.0, color='k', ls='--', lw=2)
        ax_2d.set_xlim(0.0, maxRadius_arcmin * radDistance)
        ax_2d.grid(True)
        # Bottom x label and tick labels only on the bottom row of each column
        if row == n_rows - 1:
            ax_2d.set_xlabel('R [arcmin]')
            ax_2d.tick_params(axis='x', bottom=True, labelbottom=True, top=True, labeltop=False)
        else:
            ax_2d.tick_params(axis='x', bottom=True, labelbottom=False, top=True, labeltop=False)
        _label_panel(ax_2d, sim_label)

        # Secondary x-axis (comoving kpc/h) on the top row
        if row == 0:
            secax_x = ax_2d.secondary_xaxis('top', functions=(forward_arcmin, inverse_arcmin))
            secax_x.set_xlabel('R [comoving kpc/h]')

    # -----------------------------------------------------------------------
    # Figure-level labels, layout and output
    # -----------------------------------------------------------------------
    # 3D: x-axis label and ticks on the top of the top row and the bottom of the
    # bottom row, for every column.
    for col in range(n_cols):
        ax_top = cast(Axes, axes_3d[0, col])
        ax_top.xaxis.set_label_position('top')
        ax_top.set_xlabel('R [comoving kpc/h]')
        ax_top.tick_params(axis='x', bottom=True, labelbottom=False,
                           top=True, labeltop=True)
        if n_rows > 1:
            ax_bot = cast(Axes, axes_3d[n_rows - 1, col])
            ax_bot.xaxis.set_label_position('bottom')
            ax_bot.set_xlabel('R [comoving kpc/h]')
            ax_bot.tick_params(axis='x', bottom=True, labelbottom=True,
                               top=True, labeltop=False)

    # One y-label per figure: the ratio labels are too long for a single row of
    # a 3-row grid.
    ylabel_2d = (_YLABEL_2D.get(filterType, _YLABEL_GENERIC)
                 if filterType == filterType2 else _YLABEL_GENERIC)
    fig_3d.supylabel(_YLABEL_3D, fontsize=16)
    fig_2d.supylabel(ylabel_2d, fontsize=16)

    # One figure-level legend in a reserved strip at the bottom: the stacked
    # areas fill the axes, so a per-panel legend would cover data.
    def _add_legend(fig, axes):
        handles, labels = cast(Axes, axes[0, 0]).get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(baryon_types),
                   frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig_3d.suptitle(f'Cumulative Baryon Fractions (3D, spheres) at $z={redshift}$', fontsize=18)
    _add_legend(fig_3d, axes_3d)
    fig_3d.tight_layout(rect=(0, 0.05, 1, 1))
    out_3d = figPath / f'{figName}_z{redshift}_3D_stackArea.{figType}'
    print(f'Saving 3D figure to {out_3d}')
    fig_3d.savefig(out_3d, dpi=300)  # type: ignore
    plt.close(fig_3d)

    title_2d = _TITLE_2D.get(filterType, f'{filterType}-filtered Baryon Fractions (2D)')
    fig_2d.suptitle(f'{title_2d} at $z={redshift}$', fontsize=18)
    _add_legend(fig_2d, axes_2d)
    fig_2d.tight_layout(rect=(0, 0.05, 1, 1))
    out_2d = figPath / f'{figName}_z{redshift}_2D_{filterType}_stackArea.{figType}'
    print(f'Saving 2D figure to {out_2d}')
    fig_2d.savefig(out_2d, dpi=300)  # type: ignore
    plt.close(fig_2d)

    print(f'Done!  Total elapsed time: {time.time()-t_total:.1f}s')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Produce 3D and 2D cumulative baryon-fraction stacked-area '
                    'figures (normalised by total matter and Omega_b / Omega_m) '
                    'for multiple simulations.'
    )
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/unbound_gas/stackArea_dsigma_z05.yaml',
        help='Path to the YAML configuration file.',
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose progress output.',
    )
    args = parser.parse_args()
    print(f"Arguments: {vars(args)}")
    main(path2config=args.path2config, verbose=not args.quiet)
