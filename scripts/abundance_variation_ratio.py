"""abundance_variation_ratio.py

Appendix figure: sensitivity of the simulated gas-fraction profiles to the
SHAM target number density ``halo_abundance_target``.

This mirrors Phase 3 of ``beam_compensated_ratio_v2.py`` -- the same six
simulations, the same stacking configuration, the same colours -- but drops the
beam compensation (Phases 1-2), the data overlay (Phase 4) and the SNR report
(Phase 5).  Instead the noBeam ratio

    f_gas(theta) = <DSigma_ionized_gas> / <DSigma_total> * Omega_m / Omega_b

is recomputed for several target number densities.  ``plot.mode`` selects how
that is shown:

``'absolute'`` (default)
    f_gas(theta) itself, on the same axes as the main-text figure.  The
    fiducial density is a solid line with markers and a shaded standard-error
    band (``plot.plot_error_bars``); the others are thinner and slightly
    faded, so each simulation reads as one band.  With six simulations and
    three densities this is 18 curves, but the density spread within a
    simulation is visibly much smaller than the spread between simulations --
    which is the point of the appendix.

``'deviation'``
    The fractional deviation from the fiducial density,

        100 * [ f_gas(theta | n) / f_gas(theta | n_fid) - 1 ]   [per cent]

    so the fiducial is the zero line by construction and only the non-fiducial
    densities draw lines (12 rather than 18).  A grey band shows the fractional
    statistical error of the kSZ x lensing measurement, as a yardstick.  That
    band is free to compute: with ``use_sim_scatter: false`` the beam
    compensation in ``beam_compensated_ratio_v2.py`` divides both ``ratio`` and
    ``ratio_err`` by the same ``beam_factor``, so sigma/R is identically
    unchanged by the compensation and can be read straight off the data file.
    It is a *fractional* error, so it is drawn in this mode only.

Both modes print the peak deviation per simulation to stdout, so the number for
the appendix text is available either way.  The cached profiles are
mode-independent: switching modes needs only ``--replot``, never a restack.

Halo selection is performed here rather than inside ``stackMap`` so that

  * the mean parent-halo mass reported in the summary table is provably the
    sample that was stacked, not an independent re-derivation, and
  * the subhalo catalogue is ranked by stellar mass once per simulation instead
    of once per (simulation, density, particle type).

Because every density slices the same stellar-mass-ranked list, the samples are
exactly nested: the n_fid/2 sample is the head of the n_fid sample.  The curves
therefore trace a systematic shift in the halo mass mix, not independent sample
noise.

Usage
-----
    python abundance_variation_ratio.py -p configs/abundance_variation_z05.yaml
    python abundance_variation_ratio.py -p configs/abundance_variation_z026.yaml
    python abundance_variation_ratio.py -p configs/abundance_variation_z05.yaml --replot
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import yaml
import argparse
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

sys.path.append('../src/')
from utils import arcmin_to_comoving, comoving_to_arcmin  # type: ignore
from stacker import SimulationStacker  # type: ignore

sys.path.append('../../illustrisPython/')
import illustris_python as il  # type: ignore  # noqa: F401 (needed by stacker internals)

# ---------------------------------------------------------------------------
# Matplotlib style -- matches beam_compensated_ratio_v2.py exactly
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern", "CMU Serif", "DejaVu Serif", "Times New Roman"],
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

_OMEGA_B_TNG_FALLBACK      = 0.0456
_OMEGA_B_SIMBA_FALLBACK    = 0.048
_OMEGA_B_FLAMINGO_FALLBACK = 0.0486  # header provides it; fallback should never trigger

# Fixed colours for the FLAMINGO feedback variants, keyed by feedback name.
# Keep in sync with compare_data_ratio.py / beam_compensated_ratio_v2.py /
# plot_beam_factors.py.
_FLAMINGO_COLOURS = {
    'L1_m9':           '#B30000',  # dark red (fiducial)
    'fgas-8sigma':     '#FF7F0E',  # orange
    'Jet_fgas-4sigma': '#C71585',  # magenta
}

# Linestyles assigned to the non-fiducial densities, in config order.  The
# fiducial density keeps the solid line used by the main-text figure.
_LINESTYLES         = [':', '--', '-.', (0, (3, 1, 1, 1))]
_FIDUCIAL_LINESTYLE = '-'

# Upper parent-mass bound for the SHAM pre-filter.  Matches the stackMap default
# used by the main-text figure; held fixed across densities so the samples stay
# exactly nested.
_HALO_MASS_UPPER = 5e14


def save_measurements_npz(path: str, data: dict) -> None:
    """Save a nested dict-of-dicts to an NPZ file without pickling.

    The input must have the structure ``{outer_key: {inner_key: array_like}}``.
    All inner values are converted to NumPy arrays and stored under keys of the
    form ``"outer_key/inner_key"``.

    Args:
        path: Output ``.npz`` file path.
        data: Nested dict of array-like values.

    Raises:
        TypeError: If ``data`` is not a dict-of-dicts.
        ValueError: If an inner value cannot be converted to a NumPy array.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    flat = {}
    for outer_key, inner in data.items():
        if not isinstance(inner, dict):
            raise TypeError(
                f"Value for {outer_key!r} must be a dict, got {type(inner)}")
        if '/' in outer_key:
            raise ValueError(f"Outer key must not contain '/': {outer_key!r}")
        for inner_key, value in inner.items():
            try:
                arr = np.asarray(value)
            except Exception as e:
                raise ValueError(
                    f"Could not convert {outer_key}/{inner_key} to array") from e
            flat[f"{outer_key}/{inner_key}"] = arr

    np.savez_compressed(path, **flat)


def load_measurements_npz(path: str) -> dict:
    """Load a nested dict saved by :func:`save_measurements_npz`.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        Nested ``{outer_key: {inner_key: np.ndarray}}`` mapping.
    """
    archive = np.load(path)
    out: dict = {}
    for k in archive.files:
        outer_key, inner_key = k.split("/", 1)
        out.setdefault(outer_key, {})[inner_key] = archive[k]
    return out


def _resolve_stacker(sim_type_name: str, sim: dict, redshift: float,
                     verbose: bool) -> tuple:
    """Instantiate a SimulationStacker and resolve Omega_b.

    Args:
        sim_type_name: ``'IllustrisTNG'``, ``'SIMBA'`` or ``'FLAMINGO'``.
        sim: Single simulation entry from the YAML ``sims`` list.
        redshift: Snapshot redshift.
        verbose: Whether to print warnings.

    Returns:
        ``(stacker, sim_label, omega_b)``
    """
    # Per-sim redshift override: a sim entry may declare its own 'redshift'
    # (e.g. a FLAMINGO z=0.30 snapshot substituted into a z=0.26 comparison);
    # otherwise fall back to the config-level redshift passed in.
    z = sim.get('redshift', redshift)

    if sim_type_name == 'IllustrisTNG':
        stacker   = SimulationStacker(sim['name'], sim['snapshot'],
                                      z=z, simType=sim_type_name)
        sim_label = sim['name']
        try:
            omega_b = stacker.header['OmegaBaryon']
        except KeyError:
            omega_b = _OMEGA_B_TNG_FALLBACK
            if verbose:
                print(f"  [warn] OmegaBaryon missing in {sim_label} header; "
                      f"using fallback {_OMEGA_B_TNG_FALLBACK}")

    elif sim_type_name == 'SIMBA':
        stacker   = SimulationStacker(sim['name'], sim['snapshot'],
                                      z=z, simType=sim_type_name,
                                      feedback=sim['feedback'])
        sim_label = f"SIMBA-100"
        try:
            omega_b = stacker.header['OmegaBaryon']
        except KeyError:
            omega_b = _OMEGA_B_SIMBA_FALLBACK
            if verbose:
                print(f"  [warn] OmegaBaryon missing in {sim_label} header; "
                      f"using fallback {_OMEGA_B_SIMBA_FALLBACK}")

    elif sim_type_name == 'FLAMINGO':
        stacker   = SimulationStacker(sim['name'], sim['snapshot'],
                                      z=z, simType=sim_type_name,
                                      feedback=sim['feedback'])
        # '-' instead of '_' so labels render under usetex
        sim_label = f"FLAMINGO {sim['feedback']}".replace('_', '-')
        try:
            omega_b = stacker.header['OmegaBaryon']
        except KeyError:
            omega_b = _OMEGA_B_FLAMINGO_FALLBACK
            if verbose:
                print(f"  [warn] OmegaBaryon missing in {sim_label} header; "
                      f"using fallback {_OMEGA_B_FLAMINGO_FALLBACK}")

    else:
        raise ValueError(f"Unknown simulation type: {sim_type_name!r}. "
                         "Expected 'IllustrisTNG', 'SIMBA' or 'FLAMINGO'.")

    return stacker, sim_label, omega_b


def abundance_masks(stacker: SimulationStacker, targets: list,
                    halo_mass_upper: float = _HALO_MASS_UPPER) -> list:
    """SHAM subhalo masks for several target number densities, in one pass.

    Replicates the ``use_subhalos=True`` selection inside
    ``SimulationStacker.stack_on_array`` exactly: subhalos are pre-filtered by
    parent FoF mass, ranked by stellar mass (``SubhaloMStar``), and the top
    ``int(n * V_box)`` are kept.  ``N_gal`` is computed from the full box volume
    regardless of the pre-filter, matching ``select_abundance_subhalos``.

    Ranking is done once and every target slices the same ordered list, so the
    resulting samples are exactly nested: a lower target number density yields a
    strict subset of a higher one.  Differences between densities are therefore
    a systematic shift in the halo mass mix rather than independent sample noise.

    Args:
        stacker: Instantiated SimulationStacker.
        targets: Target number densities in (cMpc/h)^-3.
        halo_mass_upper: Upper parent-mass bound (Msun/h) for the pre-filter.

    Returns:
        List of dicts, one per entry of ``targets`` and in the same order, with
        keys ``'mask'`` (integer indices into the subhalo catalogue),
        ``'n_halos'``, ``'mean_mass'`` (mean parent-halo mass, Msun/h) and
        ``'mean_R200m'`` (mean parent-halo R200m, comoving kpc/h).
    """
    subhalos    = stacker.loadSubHalos()
    parents     = stacker.loadHalos()
    parent_mass = parents['GroupMass'][subhalos['SubhaloGrNr']]
    valid       = np.where(parent_mass <= halo_mass_upper)[0]

    # Rank once; each target is a prefix of this ordering.
    order      = np.argsort(subhalos['SubhaloMStar'][valid])[::-1]
    box_volume = (stacker.header['BoxSize'] / 1e3) ** 3   # ckpc/h -> (cMpc/h)^3

    out = []
    for target in targets:
        n_gal       = int(target * box_volume)
        mask        = valid[order[:n_gal]]
        parent_grnr = subhalos['SubhaloGrNr'][mask]
        out.append({
            'mask':       mask,
            'n_halos':    mask.size,
            'mean_mass':  np.mean(parents['GroupMass'][parent_grnr]),
            'mean_R200m': np.mean(parents['GroupRad'][parent_grnr]),
        })
    return out


def build_colour_map(config: dict) -> dict:
    """Map simulation label to plot colour.

    Uses the same per-sim-type colourmap logic as ``compare_data_ratio.py`` and
    ``beam_compensated_ratio_v2.py`` so that a given simulation keeps the same
    colour across every figure in the paper.

    Args:
        config: The noBeam config dict (must contain a ``simulations`` list).

    Returns:
        ``{sim_label: colour}`` mapping.
    """
    colour_for_sim: dict = {}
    for i, sim_group in enumerate(config['simulations']):
        cmap    = matplotlib.colormaps[['plasma', 'twilight', 'hot'][i]]  # type: ignore[attr-defined]
        n_sims  = len(sim_group['sims'])
        colours = cmap(np.linspace(0.2, 0.85, n_sims))
        for j, sim in enumerate(sim_group['sims']):
            if sim_group['sim_type'] == 'IllustrisTNG':
                label  = sim['name']
                colour = colours[j]
            elif sim_group['sim_type'] == 'FLAMINGO':
                label  = f"FLAMINGO {sim['feedback']}".replace('_', '-')
                colour = _FLAMINGO_COLOURS.get(sim['feedback'], colours[j])
            else:
                label  = f"SIMBA-100"
                colour = colours[j]
            colour_for_sim[label] = colour
    return colour_for_sim


def _sim_key(sim_label: str) -> str:
    """NPZ-safe key for a simulation label (no spaces, no '/')."""
    return sim_label.replace(' ', '_')


def _entry_key(sim_label: str, target: float) -> str:
    """NPZ outer key for one (simulation, number density) pair."""
    return f"{_sim_key(sim_label)}__{target:.3e}"


def _target_label(target: float, fiducial: float) -> str:
    """LaTeX legend label giving the density and its multiple of the fiducial."""
    mantissa, exponent = f'{target:.1e}'.split('e')
    number = rf'{mantissa} \times 10^{{{int(exponent)}}}'

    ratio = target / fiducial
    if np.isclose(ratio, 0.5):
        multiple = r'\bar{n}_{\rm gal}/2'
    elif np.isclose(ratio, 2.0):
        multiple = r'2\,\bar{n}_{\rm gal}'
    elif np.isclose(ratio, 1.0):
        multiple = r'\bar{n}_{\rm gal}'
    else:
        multiple = rf'{ratio:.3g}\,\bar{{n}}_{{\rm gal}}'

    return rf'${number}$ (${multiple}$)'


def compute_results(nb_config: dict, targets: list, verbose: bool = True) -> dict:
    """Stack every simulation at every target number density.

    Args:
        nb_config: The noBeam config dict (stacking parameters + simulations).
        targets: Target number densities in (cMpc/h)^-3.
        verbose: If True, print progress messages to stdout.

    Returns:
        Nested dict ready for :func:`save_measurements_npz`, with one entry per
        ``(simulation, density)`` pair plus a ``'meta'`` entry.

    Raises:
        ValueError: If the noBeam config does not enable ``use_subhalos``.
    """
    nb_stack = nb_config['stack']

    if not nb_stack.get('use_subhalos', False):
        raise ValueError(
            "The noBeam config must set use_subhalos: true -- "
            "halo_abundance_target only has meaning for SHAM subhalo selection.")

    nb_redshift     = nb_stack.get('redshift',        0.5)
    nb_rad_distance = nb_stack.get('rad_distance',    1.0)
    nb_pType        = nb_stack.get('particle_type',   'ionized_gas')
    nb_filter_type  = nb_stack.get('filter_type',     'DSigma')
    nb_pixel_size   = nb_stack.get('pixel_size',      0.2)
    nb_beam_size    = nb_stack.get('beam_size',       None)
    nb_pType2       = nb_stack.get('particle_type_2', 'total')
    nb_filter_type2 = nb_stack.get('filter_type_2',   'DSigma')
    nb_pixel_size_2 = nb_stack.get('pixel_size_2',    0.2)
    nb_beam_size_2  = nb_stack.get('beam_size_2',     None)

    # Identical to beam_compensated_ratio_v2.py except that halo selection is
    # supplied explicitly via halo_mask instead of halo_abundance_target.
    nb_base_kwargs = dict(
        minRadius    = nb_stack.get('min_radius',   1.0),
        maxRadius    = nb_stack.get('max_radius',   6.0),
        numRadii     = nb_stack.get('num_radii',    9),
        projection   = nb_stack.get('projection',   'yz'),
        save         = nb_stack.get('save_field',   True),
        load         = nb_stack.get('load_field',   True),
        radDistance  = nb_rad_distance,
        mask         = nb_stack.get('mask_haloes',  False),
        maskRad      = nb_stack.get('mask_radii',   3.0),
        use_subhalos = True,
    )

    results: dict = {}
    summary: list = []          # (sim_label, target, n_halos, mean_mass, mean_R200m)
    theta: Optional[np.ndarray] = None
    cosmo_ref: Optional[FlatLambdaCDM] = None
    sim_keys: list = []

    for sim_group in nb_config['simulations']:
        sim_type_name = sim_group['sim_type']
        for sim in sim_group['sims']:
            stacker, sim_label, omega_b = _resolve_stacker(
                sim_type_name, sim, nb_redshift, verbose)

            # _resolve_stacker collapses some entries to a fixed label (every
            # SIMBA run becomes 'SIMBA-100'), so two active entries of the same
            # type would silently overwrite each other's results and colour.
            if _sim_key(sim_label) in sim_keys:
                raise ValueError(
                    f"Duplicate simulation label {sim_label!r} in the config. "
                    "Labels must be unique -- results would overwrite silently.")
            sim_keys.append(_sim_key(sim_label))

            if cosmo_ref is None:
                cosmo_ref = FlatLambdaCDM(
                    H0=100 * stacker.header['HubbleParam'],
                    Om0=stacker.header['Omega0'],
                    Tcmb0=2.7255 * u.K,
                    Ob0=omega_b,
                )

            # Omega_m / Omega_b, so the ratio is normalised by the cosmic
            # baryon fraction.  Cancels in the deviation from fiducial, but is
            # applied here so the stored f_gas is physically meaningful.
            factor = stacker.header['Omega0'] / omega_b

            if verbose:
                print(f"[noBeam] Processing {sim_label}")

            # One catalogue read and one stellar-mass ranking for all densities.
            masks = abundance_masks(stacker, targets)

            for target, info in zip(targets, masks):
                if verbose:
                    print(f"  {sim_label}: n = {target:.3e} (cMpc/h)^-3, "
                          f"N = {info['n_halos']}, "
                          f"mean M = {info['mean_mass']:.3e} Msun/h "
                          f"(log10 = {np.log10(info['mean_mass']):.3f}), "
                          f"mean R200m = {info['mean_R200m']:.3f} comoving kpc/h")

                # Denominator first, then numerator.  Both reuse the in-memory
                # map cached on the stacker (keyed on pType/z/projection/pixel/
                # beam only), so extra densities cost stacking but no map I/O.
                _, profiles_den = stacker.stackMap(
                    nb_pType2, filterType=nb_filter_type2,
                    pixelSize=nb_pixel_size_2, beamSize=nb_beam_size_2,
                    halo_mask=info['mask'], **nb_base_kwargs)
                radii, profiles_num = stacker.stackMap(
                    nb_pType, filterType=nb_filter_type,
                    pixelSize=nb_pixel_size, beamSize=nb_beam_size,
                    halo_mask=info['mask'], **nb_base_kwargs)

                mean_den = np.mean(profiles_den, axis=1)
                mean_num = np.mean(profiles_num, axis=1)
                fgas     = mean_num / mean_den * factor

                # Standard error on the halo mean, propagated through the
                # ratio.  Same formula as beam_compensated_ratio_v2.py: the
                # numerator/denominator correlation (both are stacked on the
                # same halos) is neglected there too, so the shaded bands in
                # the two figures mean the same thing.
                err_den  = np.std(profiles_den, axis=1) / np.sqrt(profiles_den.shape[1])
                err_num  = np.std(profiles_num, axis=1) / np.sqrt(profiles_num.shape[1])
                fgas_err = np.abs(fgas) * np.sqrt(
                    (err_num / mean_num) ** 2 + (err_den / mean_den) ** 2)

                if theta is None:
                    theta = radii * nb_rad_distance

                results[_entry_key(sim_label, target)] = {
                    'fgas':       fgas,
                    'fgas_err':   fgas_err,
                    'mean_num':   mean_num,
                    'mean_den':   mean_den,
                    'err_num':    err_num,
                    'err_den':    err_den,
                    'target':     target,
                    'n_halos':    info['n_halos'],
                    'mean_mass':  info['mean_mass'],
                    'mean_R200m': info['mean_R200m'],
                }
                summary.append((sim_label, target, info['n_halos'],
                                info['mean_mass'], info['mean_R200m']))

    if theta is None or cosmo_ref is None:
        raise ValueError("No simulations were processed -- check the config.")

    results['meta'] = {
        'abundance_targets': np.asarray(targets),
        'redshift':          nb_redshift,
        'theta':             theta,
        'cosmo_H0':          cosmo_ref.H0.value,
        'cosmo_Om0':         cosmo_ref.Om0,
        'sim_keys':          np.asarray(sim_keys),
    }

    if verbose:
        _print_summary_table(summary)

    return results


def _print_summary_table(summary: list) -> None:
    """Print the sim x density table of sample sizes and mean halo properties.

    Args:
        summary: List of ``(sim_label, target, n_halos, mean_mass, mean_R200m)``.
    """
    print("\n" + "=" * 86)
    print("SHAM sample properties")
    print("=" * 86)
    print(f"{'simulation':<26}{'n [(cMpc/h)^-3]':>17}{'N_halos':>10}"
          f"{'<M> [Msun/h]':>16}{'log10<M>':>10}{'<R200m>':>11}")
    print("-" * 86)
    for sim_label, target, n_halos, mean_mass, mean_R200m in summary:
        print(f"{sim_label:<26}{target:>17.3e}{n_halos:>10d}"
              f"{mean_mass:>16.4e}{np.log10(mean_mass):>10.3f}"
              f"{mean_R200m:>11.1f}")
    print("=" * 86 + "\n")


def _print_deviation_table(deviations: dict) -> None:
    """Print the peak fractional deviation from fiducial, per simulation.

    Args:
        deviations: ``{sim_label: {target: deviation array in per cent}}``.
    """
    print("=" * 62)
    print("Fractional deviation from the fiducial number density")
    print("=" * 62)
    print(f"{'simulation':<26}{'n [(cMpc/h)^-3]':>17}{'max |dev| [%]':>17}")
    print("-" * 62)
    worst = 0.0
    for sim_label, per_target in deviations.items():
        for target, dev in per_target.items():
            peak  = np.max(np.abs(dev))
            worst = max(worst, peak)
            print(f"{sim_label:<26}{target:>17.3e}{peak:>17.2f}")
    print("-" * 62)
    print(f"Largest deviation across all simulations and radii: {worst:.2f} %")
    print("=" * 62 + "\n")


def make_figure(results: dict, nb_config: dict, plot_config: dict,
                targets: list, fiducial: float, out_path: Path,
                verbose: bool = True) -> None:
    """Draw and save the fractional-deviation figure.

    Args:
        results: Output of :func:`compute_results` (or the loaded ``.npz``).
        nb_config: The noBeam config dict, used for colours and axis limits.
        plot_config: The ``plot`` section of the master config.
        targets: Target number densities in (cMpc/h)^-3.
        fiducial: The fiducial target number density.
        out_path: Full path (with suffix) to write the figure to.
        verbose: If True, print progress messages to stdout.
    """
    nb_stack     = nb_config['stack']
    nb_redshift  = float(np.asarray(results['meta']['redshift']))
    theta        = np.asarray(results['meta']['theta'])
    rad_distance = nb_stack.get('rad_distance', 1.0)

    cosmo_ref = FlatLambdaCDM(
        H0=float(np.asarray(results['meta']['cosmo_H0'])),
        Om0=float(np.asarray(results['meta']['cosmo_Om0'])),
        Tcmb0=2.7255 * u.K,
    )

    mode = plot_config.get('mode', 'absolute')
    if mode not in ('absolute', 'deviation'):
        raise ValueError(
            f"plot.mode must be 'absolute' or 'deviation', got {mode!r}.")

    colour_for_sim = build_colour_map(nb_config)

    # Linestyle per density, assigned in config order; the fiducial stays solid.
    non_fiducial = [t for t in targets if not np.isclose(t, fiducial)]
    if len(non_fiducial) > len(_LINESTYLES):
        raise ValueError(
            f"{len(non_fiducial)} non-fiducial densities but only "
            f"{len(_LINESTYLES)} linestyles are defined.")
    linestyle_for_target = dict(zip(non_fiducial, _LINESTYLES))
    linestyle_for_target[fiducial] = _FIDUCIAL_LINESTYLE

    fig, ax = plt.subplots(figsize=tuple(plot_config.get('figsize', (8, 6))))

    # ---- grey band: fractional statistical error of the measurement ----
    # Deviation mode only: the band is a *fractional* error, so it has no
    # meaning on an absolute f_gas axis.  In absolute mode the sim-to-sim
    # spread is itself the yardstick.
    #
    # sigma/R is unchanged by beam compensation when use_sim_scatter is false,
    # since both ratio and ratio_err are divided by the same beam_factor.
    frac_err = None
    if mode == 'deviation':
        if 'data_path' not in plot_config:
            print("[warn] no plot.data_path in the config -- "
                  "the measurement error band will not be drawn.")
        else:
            data       = load_measurements_npz(plot_config['data_path'])
            data_key   = plot_config.get('data_key', 'source_bin_0')
            theta_data = data[data_key]['ksz_theta_arcmin']
            frac_err   = np.abs(
                data[data_key]['ratio_err'] / data[data_key]['ratio']) * 100

            if not np.allclose(theta_data, theta, rtol=1e-3):
                print("[warn] data theta grid does not match the simulation "
                      "radii -- the error band will be drawn on its own grid.")
                print(f"  simulation theta: {theta}")
                print(f"  data theta      : {theta_data}")

            ax.fill_between(theta_data, -frac_err, frac_err,
                            color='0.85', lw=0, zorder=0)
            # Crisp edges, so the band still reads as an envelope where it runs
            # off-panel.
            for sign in (+1, -1):
                ax.plot(theta_data, sign * frac_err, color='0.65', lw=1,
                        label='_nolegend_', zorder=1)

    # ---- curves ----
    deviations: dict = {}
    peak_deviation   = 0.0
    fgas_min, fgas_max = np.inf, -np.inf

    # Shaded standard error on the fiducial curve only, as in the main-text
    # figure.  Bands on all three densities would be unreadable, and the
    # nested samples make them a poor guide to the density differences anyway.
    plot_error_bars = plot_config.get('plot_error_bars', True)
    missing_err     = False

    for sim_label, colour in colour_for_sim.items():
        fiducial_key = _entry_key(sim_label, fiducial)
        if fiducial_key not in results:
            raise KeyError(
                f"No cached result for {fiducial_key!r}. The cache was built "
                "from a different config -- re-run without --replot.")
        fgas_fid = np.asarray(results[fiducial_key]['fgas'])
        fgas_min = min(fgas_min, float(fgas_fid.min()))
        fgas_max = max(fgas_max, float(fgas_fid.max()))

        if mode == 'absolute':
            ax.plot(theta, fgas_fid, color=colour, lw=2, marker='o',
                    markersize=3.5, ls=_FIDUCIAL_LINESTYLE,
                    label='_nolegend_', zorder=3)

            if plot_error_bars:
                if 'fgas_err' not in results[fiducial_key]:
                    missing_err = True
                else:
                    err = np.asarray(results[fiducial_key]['fgas_err'])
                    ax.fill_between(theta, fgas_fid - err, fgas_fid + err,
                                    color=colour, alpha=0.2, lw=0, zorder=1)
                    fgas_min = min(fgas_min, float((fgas_fid - err).min()))
                    fgas_max = max(fgas_max, float((fgas_fid + err).max()))

        deviations[sim_label] = {}
        for target in non_fiducial:
            key = _entry_key(sim_label, target)
            if key not in results:
                raise KeyError(
                    f"No cached result for {key!r}. The cache was built from a "
                    "different config -- re-run without --replot.")
            fgas = np.asarray(results[key]['fgas'])
            dev  = 100.0 * (fgas / fgas_fid - 1.0)
            fgas_min = min(fgas_min, float(fgas.min()))
            fgas_max = max(fgas_max, float(fgas.max()))

            deviations[sim_label][target] = dev
            peak_deviation = max(peak_deviation, float(np.max(np.abs(dev))))

            if mode == 'absolute':
                # Thinner and slightly faded, so each simulation reads as one
                # band rather than three competing lines.
                ax.plot(theta, fgas, color=colour, lw=1.5, alpha=0.75,
                        ls=linestyle_for_target[target], label='_nolegend_',
                        zorder=3)
            else:
                ax.plot(theta, dev, color=colour, lw=2, marker='o',
                        markersize=3.5, ls=linestyle_for_target[target],
                        label='_nolegend_', zorder=3)

    if missing_err:
        print("[warn] the cache has no 'fgas_err' -- it predates the shaded "
              "error bands, so they were skipped. Re-run without --replot.")

    # Printed in both modes -- the peak deviation is the number for the text,
    # whether or not it is the quantity being plotted.
    if verbose:
        _print_deviation_table(deviations)

    # ---- axes ----
    # Absolute mode marks the cosmic baryon fraction, as the main-text figure
    # does; deviation mode marks the fiducial density.
    ax.axhline(1.0 if mode == 'absolute' else 0.0,
               color='k', ls='--', lw=1.5, label='_nolegend_', zorder=2)

    secax_x = ax.secondary_xaxis(
        'top',
        functions=(
            lambda arcmin: arcmin_to_comoving(arcmin, nb_redshift, cosmo_ref),
            lambda kpc_h:  comoving_to_arcmin(kpc_h,  nb_redshift, cosmo_ref),
        ),
    )
    secax_x.set_xlabel(r'R [comoving kpc/h]')

    default_ylabel = (r'$f_{\rm gas}(R)$' if mode == 'absolute'
                      else r'$\Delta f_{\rm gas} / f_{\rm gas}$ [\%]')
    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(plot_config.get('ylabel', default_ylabel))
    ax.set_xlim(0.0, nb_stack.get('max_radius', 6.0) * rad_distance + 0.5)

    if mode == 'absolute':
        ylim = plot_config.get('ylim', None)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            # Headroom at the top so the simulation legend has guaranteed empty
            # space: the curves rise steeply, so 'best' placement is unreliable
            # and a fixed upper-left box would sit on top of TNG300-1.
            span = fgas_max - fgas_min
            ax.set_ylim(fgas_min - 0.08 * span, fgas_max + 0.45 * span)
    else:
        # Symmetric y-limits sized to the curves, not to the (much wider) error
        # band -- the band runs off-panel rather than crushing the curves.  The
        # floor keeps the *tightest* part of the band on-panel, so it reads as
        # an envelope the curves sit inside rather than a background fill.
        ylim_percent = plot_config.get('ylim_percent', None)
        if ylim_percent is None:
            ylim_percent = max(10.0, 1.3 * peak_deviation)
            if frac_err is not None:
                ylim_percent = max(ylim_percent, 1.25 * frac_err.min())
        ax.set_ylim(-ylim_percent, ylim_percent)

    ax.grid(True, alpha=0.3)

    # ---- two-part legend: simulation colours, then density linestyles ----
    sim_handles = [Line2D([0], [0], color=colour, lw=2, ls='-', label=label)
                   for label, colour in colour_for_sim.items()]
    leg1 = ax.legend(handles=sim_handles,
                     loc=plot_config.get('legend_loc_sims', 'upper left'),
                     ncol=2, fontsize=12, framealpha=0.9)
    ax.add_artist(leg1)

    # Absolute mode draws the fiducial too, so it needs a key entry for it.
    density_targets = targets if mode == 'absolute' else non_fiducial
    density_handles = [
        Line2D([0], [0], color='gray', lw=2, ls=linestyle_for_target[t],
               label=_target_label(t, fiducial))
        for t in density_targets
    ]
    if frac_err is not None:
        density_handles.append(Patch(
            facecolor='0.85',
            label=rf'DESI $\times$ ACT $\times$ HSC $1\sigma$ '
                  rf'({frac_err.min():.0f}--{frac_err.max():.0f}\%)'))
    ax.legend(handles=density_handles,
              loc=plot_config.get(
                  'legend_loc_density',
                  'lower right' if mode == 'absolute' else 'lower left'),
              fontsize=12,
              title=r'$\bar{n}_{\rm gal}$ $[h^3\,{\rm cMpc}^{-3}]$',
              title_fontsize=12, framealpha=0.9)

    fig.tight_layout()
    print(f'Saving figure to {out_path}')
    fig.savefig(out_path, dpi=150)  # type: ignore[union-attr]
    plt.close(fig)


def main(path2config: str, replot: bool = False, verbose: bool = True) -> None:
    """Run the number-density variation test and save the figure.

    Args:
        path2config: Path to the master YAML configuration file.
        replot: If True, reuse the cached ``.npz`` instead of restacking.
        verbose: If True, print progress messages to stdout.
    """
    config_dir = Path(path2config).parent

    with open(path2config) as f:
        master = yaml.safe_load(f)

    with open(config_dir / master['no_beam_config']) as f:
        nb_config = yaml.safe_load(f)

    plot_config = master.get('plot', {})
    targets     = [float(t) for t in master['abundance_targets']]
    fiducial    = float(master['fiducial_target'])

    if not any(np.isclose(t, fiducial) for t in targets):
        raise ValueError(
            f"fiducial_target {fiducial:.3e} is not among abundance_targets "
            f"{targets}.")

    nb_stack    = nb_config['stack']
    nb_redshift = nb_stack.get('redshift', 0.5)
    nb_pType    = nb_stack.get('particle_type',   'ionized_gas')
    nb_pType2   = nb_stack.get('particle_type_2', 'total')

    fig_name = plot_config.get('fig_name', 'abundance_variation')
    fig_type = plot_config.get('fig_type', 'pdf')
    mode     = plot_config.get('mode', 'absolute')
    out_stem = f'{fig_name}_{nb_pType}_{nb_pType2}'
    if mode == 'deviation':
        # Suffixed so flipping the toggle does not overwrite the default figure.
        # The cache is mode-independent, so switching needs only --replot.
        out_stem += '_deviation'

    # ---- Output paths ----
    # Canonical cache: stable path, so --replot finds it on any later day.
    npz_path = Path(plot_config.get(
        'npz_path', f'../data/abundance_variation/{fig_name}.npz'))
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Figure (and a provenance copy of the cache): figures/<year-month>/<month-day>/
    now      = datetime.now()
    fig_path = (
        Path(plot_config.get('fig_path', '../figures/'))
        / now.strftime("%Y-%m")
        / now.strftime("%m-%d")
    )
    fig_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if replot:
        if not npz_path.exists():
            raise FileNotFoundError(
                f"--replot given but no cache at {npz_path}. "
                "Run once without --replot first.")
        print(f'Loading cached profiles from {npz_path}')
        results = load_measurements_npz(str(npz_path))
    else:
        results = compute_results(nb_config, targets, verbose=verbose)
        save_measurements_npz(str(npz_path), results)
        print(f'Saved profiles to {npz_path}')
        save_measurements_npz(str(fig_path / f'{out_stem}.npz'), results)
        print(f'Saved provenance copy to {fig_path / f"{out_stem}.npz"}')

    make_figure(results, nb_config, plot_config, targets, fiducial,
                fig_path / f'{out_stem}.{fig_type}', verbose=verbose)

    print(f'Done (z={nb_redshift}). Elapsed: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot the sensitivity of the simulated gas-fraction ratio '
                    'to the SHAM target number density.',
    )
    parser.add_argument(
        '-p', '--path2config',
        type=str,
        default='./configs/abundance_variation_z05.yaml',
        help='Path to the master YAML configuration file.',
    )
    parser.add_argument(
        '--replot',
        action='store_true',
        help='Reuse the cached .npz instead of restacking (fast cosmetics loop).',
    )
    args = vars(parser.parse_args())
    print(f"Config: {args['path2config']}")
    main(**args)
