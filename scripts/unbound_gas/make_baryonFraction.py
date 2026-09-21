"""
make_baryonFraction.py
======================
Produces two figures showing baryon-type fractions normalised by the total
baryonic mass (so all components always sum to 1):

    Figure 1 — 3D stacking, figsize (9, 9).  X-axis in comoving kpc/h.
    Figure 2 — 2D stacking, figsize (9, 9).  X-axis in arcmin; secondary top
               axis in comoving kpc/h.

Both figures are laid out as ``n_rows × n_cols`` panels filled **column-major**,
so with six simulations and the default two columns the first three fill the
left column and the last three the right.

Each figure can show either a cumulative or a differential quantity, and the
suptitle says which:

    3D   ``sphere: true``   mass within a sphere of radius R    (stacked area)
         ``sphere: false``  mass in the shell between edges     (stacked bar)
    2D   ``filter_type: 'cumulative'``  mass within a disk of radius R  (area)
         ``filter_type: 'ring'``        mass in the annulus between edges (bar)
         ``filter_type: 'CAP'``         compensated aperture filter      (area)

The mode is also folded into the output filename so cumulative and differential
runs do not overwrite each other.

Unlike make_stackArea.py, the denominator is the sum of all baryon-type
fields rather than the total matter field, so the stacked areas always
integrate to 1 and no Omega_b / Omega_m normalisation is applied.

Usage
-----
    python unbound_gas/make_baryonFraction.py -p ./configs/unbound_gas/baryonFraction_z05.yaml

Config file format
------------------
Same format as make_stackArea.py, plus the ``filter_type: 'ring'`` option,
``derive_neutral_gas`` and ``plot.n_cols``; see ``main`` for the full list.
``particle_type_2`` and ``filter_type_2`` are ignored; the denominator is
derived internally.

Resolution caveat
-----------------
The FLAMINGO L1_m9 box is 681,000 ckpc/h, so the 3D figure's ``n_pixels: 1000``
grid gives 681 ckpc/h voxels — coarser than the ~222 ckpc/h radial shells and
than the mean R200m of the selected sample (558 ckpc/h).  The FLAMINGO rows of
the **3D** figure are therefore unresolved below ~700 ckpc/h and should be read
as indicative only.  The 2D figure is unaffected (0.2 arcmin pixels).

Dependencies
------------
* SimulationStacker src/ package (stacker, halos, mask_utils, utils)
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

# ---------------------------------------------------------------------------
# Global matplotlib style
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
# Helper: create a SimulationStacker and associated cosmology
# ---------------------------------------------------------------------------

def make_stacker(sim: dict, redshift: float):
    """Initialise a SimulationStacker for the given simulation entry.

    Parameters
    ----------
    sim : dict
        A single entry from the ``simulations`` list in the config file.
        Required keys: ``sim_type``, ``name``, ``snapshot``.
        SIMBA and FLAMINGO entries also require ``feedback``; for FLAMINGO it
        holds the variant directory name ('L1_m9' = fiducial).
    redshift : float
        Target redshift.

    Returns
    -------
    stacker : SimulationStacker
    cosmo : FlatLambdaCDM
    sim_label : str
        Human-readable label for plot titles.
    """
    sim_type = sim['sim_type']
    sim_name = sim['name']
    snapshot = sim['snapshot']

    if sim_type == 'IllustrisTNG':
        stacker = SimulationStacker(sim_name, snapshot, z=redshift, simType=sim_type)
        sim_label = sim_name
        try:
            OmegaBaryon = stacker.header['OmegaBaryon']
        except KeyError:
            OmegaBaryon = 0.0456

    elif sim_type == 'SIMBA':
        feedback = sim['feedback']
        stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                    simType=sim_type, feedback=feedback)
        # sim_label = f"{sim_name}_{feedback}"
        sim_label = "SIMBA-m100"
        OmegaBaryon = 0.048

    elif sim_type == 'FLAMINGO':
        # feedback holds the FLAMINGO variant directory name ('L1_m9' = fiducial).
        feedback = sim['feedback']
        stacker = SimulationStacker(sim_name, snapshot, z=redshift,
                                    simType=sim_type, feedback=feedback)
        # '-' instead of '_' so the label renders under usetex (cf. star_fraction_v2.py).
        sim_label = f"FLAMINGO {feedback}".replace('_', '-')
        # load_flamingo_header normalises the SWIFT cosmology into TNG-style keys,
        # so OmegaBaryon is present; fall back to the DES Y3 value used elsewhere
        # in the repo if a variant ever omits it.
        try:
            OmegaBaryon = stacker.header['OmegaBaryon']
        except KeyError:
            OmegaBaryon = 0.0486

    else:
        raise ValueError(f"Unknown sim_type '{sim_type}'. "
                         "Supported: 'IllustrisTNG', 'SIMBA', 'FLAMINGO'.")

    cosmo = FlatLambdaCDM(
        H0=100 * stacker.header['HubbleParam'],
        Om0=stacker.header['Omega0'],
        Tcmb0=2.7255 * u.K,
        Ob0=OmegaBaryon,
    )
    return stacker, cosmo, sim_label


# ---------------------------------------------------------------------------
# Helper: divide-by-zero-safe baryon fractions
# ---------------------------------------------------------------------------

def safe_fractions(means, baryon_types):
    """Normalise per-component means by their sum, guarding empty radial bins.

    A radial bin whose total baryon mass is zero (e.g. the innermost shell when
    ``min_radius = 0``, which can contain no particles) would otherwise produce
    a silent ``0 / 0 = NaN`` for some components and a misleading uniform split
    for others.  Such bins are set to NaN for *every* component and a
    ``RuntimeWarning`` is emitted, so the stacked plot shows a visible gap there
    rather than a spurious value.

    Parameters
    ----------
    means : dict[str, numpy.ndarray]
        Per-baryon-type mean profile.  All arrays must share the same shape
        ``(n_bins,)``.
    baryon_types : list[str]
        Component keys into ``means``, in stacking order.

    Returns
    -------
    list[numpy.ndarray]
        One fraction array per baryon type, in the order of ``baryon_types``.
        Across components the fractions sum to 1 in bins with non-zero total
        baryon mass and are NaN in bins where the total is zero.
    """
    total = sum(means[bt] for bt in baryon_types)
    empty = total == 0.0
    if np.any(empty):
        warnings.warn(
            f"{int(np.count_nonzero(empty))} radial bin(s) have zero total "
            "baryon mass; their fractions are set to NaN.",
            RuntimeWarning, stacklevel=2,
        )
    # Avoid the 0/0 warning/NaN-from-division by dividing by 1 where empty,
    # then overwriting those bins with NaN explicitly.
    safe_total = np.where(empty, 1.0, total)
    return [np.where(empty, np.nan, means[bt] / safe_total) for bt in baryon_types]


# ---------------------------------------------------------------------------
# Helper: neutral gas by subtraction
# ---------------------------------------------------------------------------

def resolve_stack_types(baryon_types, derive_neutral_gas):
    """Map the plotted components onto the particle types actually stacked.

    ``neutral_gas`` is defined in ``mapMaker`` as the total gas mass minus the
    ionized part, so ``neutral_gas = gas - ionized_gas`` holds cell by cell.
    Both fields bin the same particles with masses that enter linearly, and
    every stacking filter is linear in the mass grid, so the identity survives
    all the way to the stacked profiles.  Deriving it therefore costs nothing
    and avoids a full gas-particle pass per simulation for a field that is not
    cached at 0.2 arcmin (nor in 3D for FLAMINGO).

    Verified on the cached Illustris-1 maps: ``max|gas - ionized - neutral|``
    is 4.4e-08 of the peak, i.e. float32 round-off.

    Parameters
    ----------
    baryon_types : list[str]
        Components to be plotted.
    derive_neutral_gas : bool
        If True, substitute ``gas`` for ``neutral_gas`` in the stacking list.

    Returns
    -------
    list[str]
        Particle types to stack, in the order of ``baryon_types``, de-duplicated.
        The de-duplication matters when the caller asks for both ``gas`` and
        ``neutral_gas``: the substitution would otherwise list ``gas`` twice and
        build or stack that field twice, which for an uncached FLAMINGO field is
        a wasted pass over 5.4e9 particles rather than a wasted dict write.

    Raises
    ------
    ValueError
        If the derivation is requested but ``ionized_gas`` is not among the
        components, leaving nothing to subtract.
    """
    if not derive_neutral_gas or 'neutral_gas' not in baryon_types:
        return list(dict.fromkeys(baryon_types))
    if 'ionized_gas' not in baryon_types:
        raise ValueError(
            "derive_neutral_gas requires 'ionized_gas' in baryon_types "
            "(neutral_gas is computed as gas - ionized_gas). Either add it or "
            "set derive_neutral_gas: false to stack the neutral_gas field directly."
        )
    return list(dict.fromkeys(
        'gas' if bt == 'neutral_gas' else bt for bt in baryon_types))


def assemble_components(stacked, baryon_types, derive_neutral_gas):
    """Rebuild the plotted components from the stacked particle types.

    Parameters
    ----------
    stacked : dict[str, numpy.ndarray]
        Profiles keyed by the particle types returned by ``resolve_stack_types``.
    baryon_types : list[str]
        Components to be plotted.
    derive_neutral_gas : bool
        Whether ``neutral_gas`` was substituted by ``gas`` when stacking.

    Returns
    -------
    dict[str, numpy.ndarray]
        Profiles keyed by ``baryon_types``.
    """
    if not derive_neutral_gas or 'neutral_gas' not in baryon_types:
        return {bt: stacked[bt] for bt in baryon_types}
    return {
        bt: (stacked['gas'] - stacked['ionized_gas']) if bt == 'neutral_gas'
        else stacked[bt]
        for bt in baryon_types
    }


# ---------------------------------------------------------------------------
# Plot titles: make the cumulative / differential distinction explicit
# ---------------------------------------------------------------------------

# 3D: keyed by the `sphere` config flag.
_TITLE_3D = {
    True:  'Cumulative Baryon Fractions (3D, spheres)',
    False: 'Differential Baryon Fractions (3D, spherical shells)',
}
_TAG_3D = {True: 'sphere', False: 'shell'}

# 2D: keyed by `filter_type`.  Filters not listed here fall back to a generic
# '<name>-filtered' title, which keeps the figure honest for the compensated
# filters ('DSigma', 'upsilon', ...) that are neither cumulative nor a ring.
_TITLE_2D = {
    'cumulative': 'Cumulative Baryon Fractions (2D, disks)',
    'ring':       'Differential Baryon Fractions (2D, annuli)',
    'CAP':        'CAP-filtered Baryon Fractions (2D)',
}


def title_2d(filterType):
    """Return the suptitle stem for a 2D ``filter_type``."""
    return _TITLE_2D.get(filterType, f'{filterType}-filtered Baryon Fractions (2D)')


# ---------------------------------------------------------------------------
# 3-D stacking
# ---------------------------------------------------------------------------

def run_3d_stacking(stacker, baryon_types, nPixels, minRadius, maxRadius, nRadii,
                    projection, saveField, loadField, ax, colours,
                    radDistance, sphere=True, halo_mass_avg=10**13.22,
                    halo_mass_upper=5e14, derive_neutral_gas=True,
                    dr=None, verbose=True):
    """Build 3-D density fields and plot the baryon-fraction profile.

    The denominator is the sum of all baryon fields so the plotted fractions
    sum to 1 at every radius (a composition decomposition).

    The representation depends on ``sphere``:

    * ``sphere=False`` (differential): the ``nRadii`` values of the radial grid
      are interpreted as bin *edges*, giving ``nRadii - 1`` spherical shells.
      Shell ``i`` spans ``[edges[i], edges[i+1]]`` and is computed as
      ``sphere(edges[i+1]) - sphere(edges[i])``.  Drawn as a stacked bar
      (histogram over radius), which reads as a per-shell quantity.
    * ``sphere=True`` (cumulative): the field is accumulated within a sphere of
      radius R at each grid value.  Drawn as a stacked area, which reads as an
      integral quantity.

    Parameters
    ----------
    stacker : SimulationStacker
    baryon_types : list[str]
        Component particle types, e.g. ['ionized_gas', 'neutral_gas', 'Stars', 'BH'].
    nPixels : int
        Grid resolution for 3-D fields.
    minRadius, maxRadius : float
        Radial-grid endpoints [comoving kpc/h].
    nRadii : int
        Number of radial-grid values (= number of bin edges; differential mode
        therefore produces ``nRadii - 1`` shells).
    projection : str
        Projection axis ('xy', 'xz', or 'yz').
    saveField, loadField : bool
        Cache fields to / from disk.
    ax : matplotlib.axes.Axes
    colours : array-like
        One colour per baryon type.
    radDistance : float
        Multiplicative scaling applied to radii for the x-axis.
    sphere : bool
        Cumulative spheres (stacked area) if True; differential shells from
        consecutive grid edges (stacked bar) if False.  Default True.
    halo_mass_avg : float
        Target average halo mass [M_sun/h] passed to ``select_massive_halos``.
        Default ``10**13.22``.  Matches the default used by ``stacker.stackMap``
        in the 2-D path so the two panels stack the same halo sample.
    halo_mass_upper : float
        Upper halo-mass bound [M_sun/h] for the same selection.  Default ``5e14``.
    derive_neutral_gas : bool
        If True, stack ``gas`` in place of ``neutral_gas`` and take the
        difference against ``ionized_gas`` afterwards.  See
        ``resolve_stack_types``.  Default True.
    dr : float, optional
        Deprecated and ignored.  Shell widths are derived from consecutive
        radial-grid edges, not from ``dr``.  Retained only for backward
        compatibility; supplying a non-zero value emits a ``DeprecationWarning``.
    verbose : bool

    Warns
    -----
    RuntimeWarning
        If the voxel size exceeds the radial bin width, i.e. the radial grid is
        finer than the field itself.  This is the case for FLAMINGO at
        ``n_pixels = 1000`` (681 ckpc/h voxels against ~222 ckpc/h shells).
    """
    if dr is not None and dr != 0.0:
        warnings.warn(
            "`dr` is deprecated and ignored; shell widths are derived from "
            "consecutive radial-grid edges. Remove `dr` from the config.",
            DeprecationWarning, stacklevel=2,
        )

    if not baryon_types:
        raise ValueError("baryon_types must contain at least one component.")

    stack_types = resolve_stack_types(baryon_types, derive_neutral_gas)

    baryon_fields = {}
    for pt in stack_types:
        if verbose:
            print(f"  Building 3D field: {pt}")
        baryon_fields[pt] = stacker.makeField(pt, nPixels=nPixels, dim='3D',
                                              projection=projection,
                                              save=saveField, load=loadField)

    # Use the first field to derive voxel size (all fields share the same grid)
    first_field = next(iter(baryon_fields.values()))
    kpcPerPixel = stacker.header['BoxSize'] / first_field.shape[0]
    if verbose:
        print(f"  kpcPerPixel = {kpcPerPixel:.3f}")

    # Surface an unresolved radial grid rather than letting it pass silently:
    # when a shell is thinner than a voxel, consecutive edges select the same
    # voxels and the innermost bins carry no independent information.
    bin_width = (maxRadius - minRadius) / max(nRadii - 1, 1)
    if kpcPerPixel > bin_width:
        warnings.warn(
            f"3D voxel size ({kpcPerPixel:.0f} ckpc/h) exceeds the radial bin "
            f"width ({bin_width:.0f} ckpc/h) for {stacker.sim}: the inner radial "
            "bins are unresolved. Increase n_pixels or widen the radial grid.",
            RuntimeWarning, stacklevel=2,
        )

    haloes = stacker.loadHalos()
    haloMass = haloes['GroupMass']
    halo_mask = select_massive_halos(haloMass, halo_mass_avg, halo_mass_upper)

    haloes['GroupMass'] = haloes['GroupMass'][halo_mask]
    haloes['GroupRad'] = haloes['GroupRad'][halo_mask]
    GroupPos_px = np.round(haloes['GroupPos'][halo_mask] / kpcPerPixel).astype(int) % nPixels
    n_haloes = len(haloes['GroupMass'])

    if verbose:
        print(f"  Number of selected haloes: {n_haloes}")

    # Radial-grid values are treated as bin EDGES -> (nRadii - 1) shells.
    edges = np.linspace(minRadius, maxRadius, nRadii)

    # Cumulative baryon mass within a sphere at each edge, per halo.
    # Computed once per edge and differenced for shells, so each edge's
    # cutout is evaluated only once (≈2x fewer cutout calls than computing
    # an inner and outer sphere per shell).
    cumulative = {pt: [] for pt in stack_types}
    t0 = time.time()
    for edge in edges:
        rr = np.full(n_haloes, edge / kpcPerPixel)
        mask_indices = get_cutout_indices_3d(first_field, GroupPos_px, rr)
        for pt in stack_types:
            cumulative[pt].append(
                sum_over_cutouts(baryon_fields[pt], mask_indices.copy())
            )
        if verbose:
            print(f"    r={edge:.0f} kpc/h  elapsed={time.time()-t0:.1f}s")
    for pt in stack_types:
        cumulative[pt] = np.array(cumulative[pt])  # (nRadii, n_haloes) # type: ignore

    # gas -> neutral_gas by subtraction (no-op when derive_neutral_gas is False)
    cumulative = assemble_components(cumulative, baryon_types, derive_neutral_gas)

    if sphere:
        # Cumulative composition -> stacked area (continuous / integral reading).
        # The r=0 edge encloses no mass, so its bin is empty and safe_fractions
        # sets it to NaN (a gap at the origin) instead of dividing 0/0.
        means = {bt: np.mean(cumulative[bt], axis=1) for bt in baryon_types}
        fractions = safe_fractions(means, baryon_types)
        ax.stackplot(edges * radDistance, fractions, labels=baryon_types,
                     alpha=0.8, colors=colours)
    else:
        # Differential composition -> stacked bar (per-shell / histogram reading).
        # Difference the cumulative sums between adjacent edges to get the
        # per-shell mass, then average over haloes.
        shells = {bt: cumulative[bt][1:] - cumulative[bt][:-1] # type: ignore
                  for bt in baryon_types}                       # (nRadii-1, n_haloes)
        means = {bt: np.mean(shells[bt], axis=1) for bt in baryon_types}
        # Shells with no enclosed particles (e.g. the innermost one when
        # min_radius=0) have zero total baryon mass; safe_fractions flags them
        # as NaN so the bar shows a gap rather than a spurious uniform split.
        fractions = safe_fractions(means, baryon_types)

        left = edges[:-1] * radDistance          # inner edge of each shell
        widths = np.diff(edges) * radDistance    # shell width (per bar)
        bottom = np.zeros_like(left)
        for bt, frac, colour in zip(baryon_types, fractions, colours):
            ax.bar(left, frac, width=widths, bottom=bottom,
                   align='edge', color=colour, alpha=0.8, label=bt,
                   edgecolor='white', linewidth=0.0)
            bottom = bottom + frac

# ---------------------------------------------------------------------------
# 2-D stacking
# ---------------------------------------------------------------------------

def run_2d_stacking(stacker, baryon_types, filterType, minRadius, maxRadius, nRadii,
                    projection, saveField, loadField, radDistance,
                    ax, colours, inverse_arcmin, forward_arcmin,
                    halo_mass_avg=10**13.22, halo_mass_upper=5e14,
                    pixelSize=0.5, beamSize=1.6, derive_neutral_gas=True,
                    verbose=True):
    """Stack 2-D projected maps and plot the baryon-fraction profile.

    The denominator is the sum of all stacked baryon maps so fractions sum to 1.

    The representation mirrors the 3-D path and depends on ``filterType``:

    * ``'ring'`` (differential): the ``nRadii`` values of the radial grid are
      interpreted as bin *edges*.  Maps are stacked with the ``'cumulative'``
      filter at every edge and consecutive edges are differenced, giving
      ``nRadii - 1`` annuli.  Drawn as a stacked bar, which reads as a
      per-annulus quantity.  This is the 2-D analogue of ``sphere=False``.
    * ``'cumulative'``: mass within a disk of radius R.  Stacked area.
    * anything else (``'CAP'``, ``'DSigma'``, ...): passed straight through to
      ``stacker.stackMap`` and drawn as a stacked area.

    Parameters
    ----------
    stacker : SimulationStacker
    baryon_types : list[str]
    filterType : str
        Filter applied to all maps: 'cumulative', 'ring', 'CAP', ...
    minRadius : float
        Minimum radius [comoving kpc/h], converted internally to arcmin.
    maxRadius : float
        Maximum radius [comoving kpc/h], converted internally to arcmin.
    nRadii : int
    projection : str
    saveField : bool
    loadField : bool
    radDistance : float
    ax : matplotlib.axes.Axes
    colours : array-like
    inverse_arcmin : callable
        comoving kpc/h → arcmin.
    forward_arcmin : callable
        arcmin → comoving kpc/h (for secondary axis).
    halo_mass_avg : float
        Target average halo mass [M_sun/h] for the 'massive' selection inside
        ``stacker.stackMap``.  Passed explicitly so the 2-D and 3-D panels stack
        the same halo sample.  Default ``10**13.22``.
    halo_mass_upper : float
        Upper halo-mass bound [M_sun/h] for the same selection.  Default ``5e14``.
    pixelSize : float
        Map pixel size [arcmin] passed to ``stacker.stackMap``.  Default 0.5.
    beamSize : float or None
        Gaussian beam FWHM [arcmin] applied before stacking; 0 or None disables
        the convolution.  Default 1.6.  Differential ('ring') runs want this off:
        at the usual grid the annuli are ~0.6 arcmin wide, narrower than a
        1.6 arcmin beam, so a beam-convolved ring measures the beam.
    derive_neutral_gas : bool
        If True, stack ``gas`` in place of ``neutral_gas`` and take the
        difference against ``ionized_gas`` afterwards.  See
        ``resolve_stack_types``.  Default True.
    verbose : bool

    Returns
    -------
    maxRadius_arcmin : float
        Maximum stacking radius in arcmin (used to set xlim on the caller).
    """
    if not baryon_types:
        raise ValueError("baryon_types must contain at least one component.")

    minRadius_arcmin = inverse_arcmin(minRadius)
    maxRadius_arcmin = inverse_arcmin(maxRadius)

    # 'ring' is not a stacker filter: it is the cumulative filter evaluated on
    # the radial-grid edges and then differenced, exactly as run_3d_stacking
    # builds shells out of cumulative spheres.
    stack_filter = 'cumulative' if filterType == 'ring' else filterType
    stack_types = resolve_stack_types(baryon_types, derive_neutral_gas)

    if verbose:
        print(f"  2D stacking: {minRadius_arcmin:.2f} – {maxRadius_arcmin:.2f} arcmin "
              f"(filter '{filterType}', pixel {pixelSize} arcmin, beam {beamSize})")

    profiles_stacked = {}
    radii_out = None
    for pt in stack_types:
        if verbose:
            print(f"  Stacking 2D map: {pt}")
        t1 = time.time()
        radii0, profiles_stacked[pt] = stacker.stackMap(
            pt, filterType=stack_filter,
            minRadius=minRadius_arcmin, maxRadius=maxRadius_arcmin, numRadii=nRadii,
            save=saveField, load=loadField, radDistance=radDistance,
            projection=projection,
            pixelSize=pixelSize, beamSize=beamSize,
            halo_mass_avg=halo_mass_avg, halo_mass_upper=halo_mass_upper,
        )
        if radii_out is None:
            radii_out = radii0
        if verbose:
            print(f"    done in {time.time()-t1:.1f}s")

    # gas -> neutral_gas by subtraction (no-op when derive_neutral_gas is False)
    profiles_baryon = assemble_components(profiles_stacked, baryon_types,
                                          derive_neutral_gas)

    # Denominator: sum of all baryon-component means.  Bins with zero total are
    # flagged as NaN by safe_fractions rather than dividing 0/0.
    if filterType == 'ring':
        # Differential composition -> stacked bar (per-annulus / histogram
        # reading).  profiles are (nRadii, n_haloes), so differencing along
        # axis 0 gives the mass in each annulus, per halo, before averaging.
        rings = {bt: profiles_baryon[bt][1:] - profiles_baryon[bt][:-1]
                 for bt in baryon_types}
        means = {bt: np.mean(rings[bt], axis=1) for bt in baryon_types}
        # Annuli with no enclosed particles (e.g. the innermost one when
        # min_radius=0) are flagged NaN so the bar shows a gap.
        fractions = safe_fractions(means, baryon_types)

        left = radii_out[:-1] * radDistance          # inner edge of each annulus
        widths = np.diff(radii_out) * radDistance    # annulus width (per bar)
        bottom = np.zeros_like(left)
        for bt, frac, colour in zip(baryon_types, fractions, colours):
            ax.bar(left, frac, width=widths, bottom=bottom,
                   align='edge', color=colour, alpha=0.8, label=bt,
                   edgecolor='white', linewidth=0.0)
            bottom = bottom + frac
    else:
        means = {bt: np.mean(profiles_baryon[bt], axis=1) for bt in baryon_types}
        fractions = safe_fractions(means, baryon_types)

        ax.stackplot(radii_out * radDistance, fractions, labels=baryon_types,
                     alpha=0.8, colors=colours)

    return maxRadius_arcmin


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(path2config: str, verbose: bool = True):
    """Load config, run 3-D and 2-D stacking for each simulation, and save two
    separate figures (one for 3D stacking, one for 2D stacking).

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
    sphere : bool, default True
        If True, each 3D radial bin accumulates all mass within a sphere of
        radius R (cumulative).  If False, each bin is the spherical shell
        between consecutive radial-grid edges (differential).
    filter_type : str, default 'CAP'
        2D filter.  'cumulative' (disks) and 'ring' (annuli between consecutive
        radial-grid edges) are the 2D analogues of ``sphere`` true/false; 'CAP'
        and the other stacker filters are passed straight through.
    pixel_size : float, default 0.5
        2D map pixel size [arcmin].
    beam_size : float, default 1.6
        2D Gaussian beam FWHM [arcmin]; 0 disables the convolution.  Set it to 0
        for ``filter_type: 'ring'`` — the annuli are narrower than a 1.6 arcmin
        beam, so a convolved ring measures the beam rather than the gas.
    derive_neutral_gas : bool, default True
        If True, obtain ``neutral_gas`` as ``gas - ionized_gas`` instead of
        stacking the ``neutral_gas`` field.  The two are equal to float32
        round-off (see ``resolve_stack_types``), and the derived form needs no
        ``neutral_gas`` field on disk — which matters because none is cached at
        0.2 arcmin, nor in 3D for FLAMINGO.  Set to False once those fields
        exist to stack them directly.
    halo_mass_avg : float, default 10**13.22
        Target average halo mass [M_sun/h] for the 'massive' halo selection,
        applied identically to the 3-D and 2-D stacks.
    halo_mass_upper : float, default 5e14
        Upper halo-mass bound [M_sun/h] for the same selection.
    dr : float, optional
        Deprecated and ignored.  Shell widths are derived from consecutive
        radial-grid edges.  Supplying it emits a ``DeprecationWarning``.

    Config keys under ``plot:``
    ---------------------------
    n_cols : int, default 2 when more than three simulations are listed, else 1
        Number of panel columns; rows follow from the simulation count.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)

    stack_config = config.get('stack', {})
    plot_config  = config.get('plot', {})

    # -----------------------------------------------------------------------
    # Read stacking parameters
    # -----------------------------------------------------------------------
    redshift     = stack_config.get('redshift', 0.5)
    loadField    = stack_config.get('load_field', True)
    saveField    = stack_config.get('save_field', True)
    radDistance  = stack_config.get('rad_distance', 1.0)
    baryon_types = stack_config.get('baryon_types', ['ionized_gas', 'neutral_gas', 'Stars', 'BH'])
    projection   = stack_config.get('projection', 'yz')
    filterType   = stack_config.get('filter_type', 'CAP')
    minRadius    = stack_config.get('min_radius', 200.0)   # comoving kpc/h
    maxRadius    = stack_config.get('max_radius', 6000.0)
    nRadii       = stack_config.get('num_radii', 15)
    nPixels      = stack_config.get('n_pixels', 1000)
    sphere       = stack_config.get('sphere', True)
    # 2D map geometry. Neither key was read before this, so stackMap's defaults
    # (0.5 arcmin pixels, 1.6 arcmin beam) were silently in force regardless of
    # what the config said.
    pixelSize    = float(stack_config.get('pixel_size', 0.5))   # arcmin
    beamSize     = stack_config.get('beam_size', 1.6)           # arcmin; 0 -> no beam
    beamSize     = None if beamSize is None else float(beamSize)
    derive_neutral_gas = stack_config.get('derive_neutral_gas', True)
    # Cast to float: PyYAML parses unsigned-exponent literals (e.g. '5.0e14')
    # as strings, which would crash deep in the halo-selection comparison.
    halo_mass_avg   = float(stack_config.get('halo_mass_avg', 10**13.22))   # M_sun/h
    halo_mass_upper = float(stack_config.get('halo_mass_upper', 5e14))      # M_sun/h
    # `dr` is deprecated and ignored; read only so a stale config still triggers
    # the DeprecationWarning emitted inside run_3d_stacking.
    dr           = stack_config.get('dr', None)

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

    colourmap = matplotlib.colormaps['plasma']  # type: ignore
    colours   = colourmap(np.linspace(0.0, 0.8, len(baryon_types)))

    sims   = config['simulations']
    n_sims = len(sims)

    # -----------------------------------------------------------------------
    # Panel grid: n_rows × n_cols, filled column-major so that each column is a
    # contiguous block of the config's simulation list (e.g. the three FLAMINGO
    # variants together in the right-hand column).
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

        stacker, cosmo, sim_label = make_stacker(sim, redshift)

        def forward_arcmin(arcmin, _redshift=redshift, _cosmo=cosmo):
            return arcmin_to_comoving(arcmin, _redshift, _cosmo)

        def inverse_arcmin(comoving, _redshift=redshift, _cosmo=cosmo):
            return comoving_to_arcmin(comoving, _redshift, _cosmo)

        # -------------------------------------------------------------------
        # 3-D stacking
        # -------------------------------------------------------------------
        ax_3d = cast(Axes, axes_3d[row, col])
        if verbose:
            print("  [3D] starting ...")
        run_3d_stacking(
            stacker=stacker,
            baryon_types=baryon_types,
            nPixels=nPixels,
            minRadius=minRadius,
            maxRadius=maxRadius,
            nRadii=nRadii,
            projection=projection,
            saveField=saveField,
            loadField=loadField,
            ax=ax_3d,
            colours=colours,
            radDistance=radDistance,
            sphere=sphere,
            halo_mass_avg=halo_mass_avg,
            halo_mass_upper=halo_mass_upper,
            derive_neutral_gas=derive_neutral_gas,
            dr=dr,
            verbose=verbose,
        )
        if col == 0:
            ax_3d.set_ylabel('Baryon fraction')
        ax_3d.set_xlim(0.0, maxRadius * radDistance)
        ax_3d.set_ylim(0.0, 1.0)
        ax_3d.grid(False)
        # Show bottom ticks on all panels but suppress labels; labels live on
        # the top of the top panel and bottom of the bottom panel (set after the loop).
        ax_3d.tick_params(axis='x', bottom=True, labelbottom=False, top=True, labeltop=False)
        # Sim label in a tight box at the lower-left corner
        ax_3d.text(0.03, 0.05, sim_label, transform=ax_3d.transAxes, fontsize=18,
                   va='bottom', ha='left',
                   bbox=dict(boxstyle='square,pad=0.1', facecolor='white',
                             edgecolor='gray', alpha=0.85))

        # -------------------------------------------------------------------
        # 2-D stacking
        # -------------------------------------------------------------------
        ax_2d = cast(Axes, axes_2d[row, col])
        if verbose:
            print("  [2D] starting ...")
        maxRadius_arcmin = run_2d_stacking(
            stacker=stacker,
            baryon_types=baryon_types,
            filterType=filterType,
            minRadius=minRadius,
            maxRadius=maxRadius,
            nRadii=nRadii,
            projection=projection,
            saveField=saveField,
            loadField=loadField,
            radDistance=radDistance,
            ax=ax_2d,
            colours=colours,
            inverse_arcmin=inverse_arcmin,
            forward_arcmin=forward_arcmin,
            halo_mass_avg=halo_mass_avg,
            halo_mass_upper=halo_mass_upper,
            pixelSize=pixelSize,
            beamSize=beamSize,
            derive_neutral_gas=derive_neutral_gas,
            verbose=verbose,
        )
        if col == 0:
            ax_2d.set_ylabel('Baryon fraction')
        ax_2d.set_xlim(0.0, maxRadius_arcmin * radDistance)
        ax_2d.set_ylim(0.0, 1.0)
        ax_2d.grid(True)
        # Bottom x label and ticks only on the bottom row of each column
        if row == n_rows - 1:
            ax_2d.set_xlabel('R [arcmin]')
            ax_2d.tick_params(axis='x', bottom=True, labelbottom=True, top=True, labeltop=False)
        else:
            ax_2d.tick_params(axis='x', bottom=True, labelbottom=False, top=True, labeltop=False)
        # Sim label in a tight box at the lower-left corner
        ax_2d.text(0.03, 0.05, sim_label, transform=ax_2d.transAxes, fontsize=18,
                   va='bottom', ha='left',
                   bbox=dict(boxstyle='square,pad=0.1', facecolor='white',
                             edgecolor='gray', alpha=0.85))
        
        # Secondary x-axis (comoving kpc/h) on top panel
        if row == 0:
            secax = ax_2d.secondary_xaxis('top', functions=(forward_arcmin, inverse_arcmin))
            secax.set_xlabel('R [comoving kpc/h]')

    # -----------------------------------------------------------------------
    # Finalise and save
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

    # One figure-level legend in a reserved strip at the bottom. A per-panel
    # legend has nowhere to go on a stacked composition plot: the areas fill the
    # axes, and on the 3x2 grid it collided with the lower-left simulation label
    # (the FLAMINGO variant names are long).
    def _add_legend(fig, axes):
        handles, labels = cast(Axes, axes[0, 0]).get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(baryon_types),
                   frameon=False, bbox_to_anchor=(0.5, 0.0))

    # Titles and filenames carry the mode, so a cumulative run and a differential
    # run of the same config are distinguishable both on the page and on disk.
    fig_3d.suptitle(f'{_TITLE_3D[bool(sphere)]} at $z={redshift}$', fontsize=18)
    _add_legend(fig_3d, axes_3d)
    fig_3d.tight_layout(rect=(0, 0.05, 1, 1))
    out_3d = figPath / f'{figName}_3D_{_TAG_3D[bool(sphere)]}_baryonFraction.{figType}'
    print(f'Saving 3D figure to {out_3d}')
    fig_3d.savefig(out_3d, dpi=300)  # type: ignore
    plt.close(fig_3d)

    fig_2d.suptitle(f'{title_2d(filterType)} at $z={redshift}$', fontsize=18)
    _add_legend(fig_2d, axes_2d)
    fig_2d.tight_layout(rect=(0, 0.05, 1, 1))
    out_2d = figPath / f'{figName}_2D_{filterType}_baryonFraction.{figType}'
    print(f'Saving 2D figure to {out_2d}')
    fig_2d.savefig(out_2d, dpi=300)  # type: ignore
    plt.close(fig_2d)

    print(f'Done!  Total elapsed time: {time.time()-t_total:.1f}s')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Produce 3D and 2D baryon-fraction stacked-area figures '
                    '(normalised by total baryons, not total matter) for multiple simulations.'
    )
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/unbound_gas/baryonFraction_z05.yaml',
        help='Path to the YAML configuration file.',
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose progress output.',
    )
    args = parser.parse_args()
    print(f"Arguments: {vars(args)}")
    main(path2config=args.path2config, verbose=not args.quiet)
