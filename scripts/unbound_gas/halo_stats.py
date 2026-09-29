"""Halo-sample selection and mass statistics for the kSZ figure scripts.

Shared by ``simulated_kSZ_masked.py`` (the 4-column masking grid) and
``simulated_kSZ.py`` (the unmasked row), so both report the same numbers in
the same format.

The point of :func:`sample_stats` is that it returns BOTH the index array and
the statistics: the caller hands the index array straight to
``stackMap(halo_mask=...)``, so the sample that is reported is provably the
sample that is stacked, and the (large) catalogues are read once per
simulation instead of once per masking column.

Imported by sibling scripts as ``from halo_stats import sample_stats`` --
Python puts the running script's own directory on ``sys.path``, so this
resolves when the scripts are launched from ``scripts/`` as
``python unbound_gas/simulated_kSZ.py``. It must be imported AFTER the
``sys.path.append('../src/')`` line, since it imports from ``src/``.
"""

import json
import warnings
from pathlib import Path

import numpy as np

from halos import select_halos  # type: ignore
from rprofiles import select_sham_subhalos  # type: ignore


# Name of the top-hat mass in each suite's own catalogue, used to label the
# headline column. SIMBA is absent: CAESAR records no top-hat mass, so those
# rows fall back to the FoF total.
_TOPHAT_LABEL = {
    'IllustrisTNG': 'M_TopHat200',
    'FLAMINGO': 'M_BN98',
}

_FALLBACK_LABEL = 'M_FoF'

# Column order of the summary table, as (stats key, header, unit-less format).
_TABLE_COLUMNS = (
    ('mass_fof', 'M_FoF'),
    ('mass_m200m', 'M_200m'),
    ('mass_m200c', 'M_200c'),
    ('mass_tophat', 'M_TopHat'),
)


def _mean_valid(masses):
    """Mean of a mass array, ignoring NaN, with the count of valid entries.

    Args:
        masses (np.ndarray or None): Masses of the selected sample, possibly
            containing NaN where the catalogue did not compute the quantity,
            or None where the suite does not record it at all.

    Returns:
        tuple: ``(mean, n_valid)``. ``(nan, 0)`` if masses is None or every
        entry is NaN.
    """
    if masses is None:
        return float('nan'), 0
    valid = np.isfinite(masses)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return float('nan'), 0
    return float(np.nanmean(masses)), n_valid


def map_geometry(stacker, z, pixel_size):
    """Pixels per side and exact arcmin per pixel of a map, as ``makeMap`` computes them.

    Args:
        stacker (SimulationStacker): Provides the header (box size, cosmology).
        z (float): Redshift of the map.
        pixel_size (float): Nominal pixel size in arcmin.

    Returns:
        tuple: (nPixels, arcmin per pixel).
    """
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM
    from utils import comoving_to_arcmin  # type: ignore
    cosmo = FlatLambdaCDM(H0=100 * stacker.header['HubbleParam'],
                          Om0=stacker.header['Omega0'], Tcmb0=2.7255 * u.K)
    theta = comoving_to_arcmin(stacker.header['BoxSize'], z, cosmo=cosmo)
    n = int(np.ceil(theta / pixel_size))
    return n, theta / n


def require_masked_map(stacker, p_type, projection, mask_rad, z, pixel_size, density):
    """Refuse to stack a masked map that was not built around the stacked sample.

    The masked-map cache path does not encode the halo sample, and a missing
    map would be rebuilt by ``SimulationStacker.makeField`` around the
    mass-cut sample and saved at that path. Since 2026-09-29 the cached
    masked maps of the unbound gas figures are built by
    precompute_masked_sham_hosts.py around the host haloes of the SHAM
    galaxies at the lensing-fitted density, with a ``.sample.json`` record.

    Args:
        stacker (SimulationStacker): The simulation.
        p_type (str): Particle type of the map (e.g. 'tau', 'tSZ').
        projection (str): Projection of the map.
        mask_rad (float): Masking radius in units of R200m.
        z (float): Redshift of the map.
        pixel_size (float): Nominal pixel size in arcmin.
        density (float or None): SHAM number density being stacked, or None
            for a mass-cut stack.

    Raises:
        FileNotFoundError: If the map or its sample record is missing.
        ValueError: If the map was built around a different sample.
    """
    from loadIO import _get_data_filepath  # type: ignore
    n, _ = map_geometry(stacker, z, pixel_size)
    path = _get_data_filepath(stacker.simType, stacker.sim, stacker.snapshot, stacker.feedback,
                              p_type, n, projection, 'map', '2D', True, mask_rad, stacker.base_path)
    meta_path = Path(str(path) + '.sample.json')
    if not (path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"{path} or its sample record is missing; build it with "
                                f"unbound_gas/precompute_masked_sham_hosts.py first")
    with open(meta_path) as f:
        meta = json.load(f)
    if (density is None or meta.get('sample') != 'sham-hosts'
            or not np.isclose(meta.get('density') or -1.0, density)):
        raise ValueError(f"{path} was built around sample {meta.get('sample')!r} at n = "
                         f"{meta.get('density')}, but this run stacks "
                         f"{'the mass-cut sample' if density is None else f'SHAM at n = {density}'}")


def fit_label(sim_type_name, sim):
    """Label under which lensing/fit_dsigma_ksz.py stores a simulation's fit.

    Mirrors the labels built by ``make_stacker`` in that script.

    Args:
        sim_type_name (str): One of 'IllustrisTNG', 'SIMBA', 'FLAMINGO'.
        sim (dict): The config entry, with 'name' and (for SIMBA and FLAMINGO)
            'feedback'.

    Returns:
        str: e.g. 'TNG300-1', 'm100n1024_s50' or 'FLAMINGO L1_m9'.
    """
    if sim_type_name == 'SIMBA':
        return f"{sim['name']}_{sim['feedback']}"
    if sim_type_name == 'FLAMINGO':
        return f"FLAMINGO {sim['feedback']}"
    return sim['name']


def load_fitted_abundances(fit_paths):
    """Read the SHAM densities fitted on lensing by lensing/fit_dsigma_ksz.py.

    Args:
        fit_paths (str or list of str): Path(s) to that script's results npz.
            Several files (e.g. a separate fit of runs missing from the first)
            are merged; a simulation fitted in more than one file is an error,
            so a density can never be picked silently from the wrong fit.

    Returns:
        dict: Best-fit number density n_best in (cMpc/h)^-3, keyed by
        :func:`fit_label`.

    Raises:
        ValueError: If the same simulation label appears in two files.
    """
    if isinstance(fit_paths, (str, bytes)) or hasattr(fit_paths, '__fspath__'):
        fit_paths = [fit_paths]
    fitted = {}
    for fit_path in fit_paths:
        with np.load(fit_path) as fit:
            for label in fit['labels']:
                label = str(label)
                if label in fitted:
                    raise ValueError(f"{label!r} is fitted in more than one of {fit_paths}")
                fitted[label] = float(fit[f'{label}/n_best'])
    return fitted


def select_sample(stacker, use_subhalos=False, halo_abundance_target=5e-4,
                  halo_mass_avg=10 ** 13.22, halo_mass_upper=5 * 10 ** 14,
                  haloes=None, subhalos=None):
    """Reproduce the halo/subhalo selection that ``stack_on_array`` performs.

    Mirrors the selection branch of
    :meth:`stacker.SimulationStacker.stack_on_array` exactly, so the returned
    index array can be passed back in as ``halo_mask`` without changing the
    stacked sample.

    Args:
        stacker (SimulationStacker): Provides the catalogues and the header.
        use_subhalos (bool, optional): If True, select subhalos by stellar-mass
            abundance matching; otherwise select halos by the 'massive'
            cumulative-average criterion. Defaults to False.
        halo_abundance_target (float, optional): Target number density in
            (cMpc/h)^-3 for the SHAM branch. None falls back to 5e-4, matching
            ``stack_on_array``. Defaults to 5e-4.
        halo_mass_avg (float, optional): Target average halo mass (Msun/h) for
            the 'massive' branch. None triggers the deprecated fixed-bin
            selection, again matching ``stack_on_array``. Defaults to 10^13.22.
        halo_mass_upper (float, optional): Upper mass bound (Msun/h). In the
            SHAM branch this is the parent-FoF-mass pre-filter. Defaults to
            5e14.
        haloes (dict, optional): Pre-loaded halo catalogue, to avoid a repeat
            read. Defaults to None (loaded internally).
        subhalos (dict, optional): Pre-loaded subhalo catalogue. Defaults to
            None (loaded internally, and only when use_subhalos is True).

    Returns:
        np.ndarray: Integer indices into the subhalo catalogue when
        use_subhalos is True, otherwise into the halo catalogue.
    """
    if halo_abundance_target is None:
        halo_abundance_target = 5e-4

    if haloes is None:
        haloes = stacker.loadHalos()

    if use_subhalos:
        if subhalos is None:
            subhalos = stacker.loadSubHalos()
        return select_sham_subhalos(stacker, halo_abundance_target,
                                    parent_mass_upper=halo_mass_upper,
                                    subhalos=subhalos, parents=haloes)

    if halo_mass_avg is None:
        # Deprecated fixed-bin path, kept so the reported sample still matches
        # whatever stack_on_array would have selected.
        warnings.warn("halo_mass_avg is None, using legacy halo selection method.",
                      DeprecationWarning, stacklevel=2)
        return select_halos(haloes['GroupMass'], 'binned', ind=2)

    return select_halos(haloes['GroupMass'], 'massive',
                        target_average_mass=halo_mass_avg,
                        upper_mass_bound=halo_mass_upper)


def sample_stats(stacker, label, use_subhalos=False, halo_abundance_target=5e-4,
                 halo_mass_avg=10 ** 13.22, halo_mass_upper=5 * 10 ** 14):
    """Select the stacking sample and summarise its halo masses.

    Every mass is averaged over the STACKED OBJECTS, not over unique halos: in
    the SHAM branch a halo hosting several selected galaxies contributes once
    per galaxy, which is the weighting the stacked profile actually carries.

    Args:
        stacker (SimulationStacker): The simulation to summarise.
        label (str): Display name for this simulation, used in the report.
        use_subhalos (bool, optional): See :func:`select_sample`.
        halo_abundance_target (float, optional): See :func:`select_sample`.
        halo_mass_avg (float, optional): See :func:`select_sample`.
        halo_mass_upper (float, optional): See :func:`select_sample`.

    Returns:
        tuple: ``(halo_mask, stats)``. ``halo_mask`` is the index array to pass
        to ``stackMap(halo_mask=...)``. ``stats`` is a dict with keys 'label',
        'n_selected', 'n_unique_halos', 'headline_key', 'headline_mass',
        'headline_valid', 'mass_fof', 'mass_m200m', 'mass_m200c',
        'mass_tophat' (each a ``(mean, n_valid)`` tuple in Msun/h), 'rad_mean'
        (mean R200m in ckpc/h) and 'mstar_mean' (Msun/h, or None outside the
        SHAM branch).
    """
    haloes = stacker.loadHalos()
    subhalos = stacker.loadSubHalos() if use_subhalos else None

    halo_mask = select_sample(stacker, use_subhalos=use_subhalos,
                              halo_abundance_target=halo_abundance_target,
                              halo_mass_avg=halo_mass_avg,
                              halo_mass_upper=halo_mass_upper,
                              haloes=haloes, subhalos=subhalos)

    if use_subhalos:
        # Each selected galaxy is attributed the mass of its parent FoF halo.
        rows = np.asarray(subhalos['SubhaloGrNr'], dtype=np.int64)[halo_mask]
        mstar_mean = float(np.mean(subhalos['SubhaloMStar'][halo_mask]))
    else:
        rows = np.asarray(halo_mask, dtype=np.int64)
        mstar_mean = None

    def masses_for(key):
        """Selected-sample values of one mass key, or None if unavailable."""
        arr = haloes.get(key)
        return None if arr is None else arr[rows]

    # GroupMass keeps the catalogues' 0.0 "not computed" sentinel on purpose
    # (loadIO.load_halos explains why: NaN would break select_massive_halos'
    # argsort ranking). Averaging is a different job, so apply the same
    # zero-means-missing convention here rather than folding a sentinel into
    # the mean. FLAMINGO's hostless centrals are the case this guards against;
    # they do not currently reach either sample, but a silent zero would be
    # indistinguishable from a real measurement if that ever changed.
    fof_masses = haloes['GroupMass'][rows]
    fof_masses = np.where(fof_masses == 0.0, np.nan, fof_masses)

    stats = {
        'label': label,
        'n_selected': int(len(halo_mask)),
        'n_unique_halos': int(len(np.unique(rows))),
        'mass_fof': _mean_valid(fof_masses),
        'mass_m200m': _mean_valid(masses_for('GroupMass_m200m')),
        'mass_m200c': _mean_valid(masses_for('GroupMass_m200c')),
        'mass_tophat': _mean_valid(masses_for('GroupMass_TopHat')),
        'rad_mean': float(np.mean(haloes['GroupRad'][rows])),
        'mstar_mean': mstar_mean,
    }

    # Headline mass: the top-hat definition where the suite records one, the
    # FoF total where it does not (SIMBA).
    if haloes.get('GroupMass_TopHat') is not None:
        stats['headline_key'] = _TOPHAT_LABEL.get(stacker.simType, 'M_TopHat')
        stats['headline_mass'], stats['headline_valid'] = stats['mass_tophat']
    else:
        stats['headline_key'] = _FALLBACK_LABEL
        stats['headline_mass'], stats['headline_valid'] = stats['mass_fof']

    return halo_mask, stats


def _fmt_mass(entry, n_selected):
    """Format a ``(mean, n_valid)`` mass entry for the summary table."""
    mean, n_valid = entry
    if n_valid == 0:
        return f'{"n/a":>12s}'
    text = f'{mean:.4e}'
    if n_valid < n_selected:
        text += '*'
    return f'{text:>12s}'


def format_stats(stats):
    """One-line progress report for a single simulation.

    Args:
        stats (dict): As returned by :func:`sample_stats`.

    Returns:
        str: A single line, ready to print.
    """
    mass = stats['headline_mass']
    log_text = 'n/a' if not np.isfinite(mass) else f'{np.log10(mass):.3f}'
    line = (f"  [halo sample] {stats['label']}: N = {stats['n_selected']}, "
            f"<{stats['headline_key']}> = {mass:.4e} Msun/h (log10 = {log_text})")
    if stats['headline_valid'] < stats['n_selected']:
        line += (f"  [{stats['n_selected'] - stats['headline_valid']} of "
                 f"{stats['n_selected']} not computed in the catalogue, excluded]")
    if stats['headline_key'] == _FALLBACK_LABEL:
        line += '  [no top-hat mass in this catalogue; FoF total shown]'
    return line


def format_table(rows):
    """Render the summary table for every simulation processed.

    Args:
        rows (list): Stats dicts, in the order they should be listed.

    Returns:
        str: Multi-line table, with no trailing newline.
    """
    header = (f"{'simulation':<32s} {'N_sel':>8s} {'headline':>12s} "
              f"{'<M_head>':>12s} " +
              ' '.join(f'{"<" + name + ">":>12s}' for _, name in _TABLE_COLUMNS) +
              f" {'<R200m>':>9s} {'<M_star>':>12s}")
    lines = [header, '-' * len(header)]
    for stats in rows:
        mstar = ('n/a' if stats['mstar_mean'] is None
                 else f"{stats['mstar_mean']:.4e}")
        lines.append(
            f"{stats['label']:<32s} {stats['n_selected']:>8d} "
            f"{stats['headline_key']:>12s} "
            f"{_fmt_mass((stats['headline_mass'], stats['headline_valid']), stats['n_selected'])} " +
            ' '.join(_fmt_mass(stats[key], stats['n_selected'])
                     for key, _ in _TABLE_COLUMNS) +
            f" {stats['rad_mean']:>9.1f} {mstar:>12s}"
        )
    lines.append('')
    lines.append('Masses in Msun/h, averaged over the stacked objects (a halo hosting several')
    lines.append('selected galaxies counts once per galaxy). <R200m> in comoving kpc/h.')
    lines.append('M_FoF is the friends-of-friends total; M_200m / M_200c / M_TopHat are')
    lines.append('spherical-overdensity masses (top-hat = Bryan & Norman 1998, i.e. TNG')
    lines.append("Group_M_TopHat200 and FLAMINGO SO/BN98). 'n/a' means the catalogue does not")
    lines.append("record that definition; a trailing '*' means some selected halos had it")
    lines.append('uncomputed and were excluded from that average.')
    return '\n'.join(lines)


def write_stats_file(path, rows, preamble=()):
    """Write the summary table to a text file next to the figure.

    Args:
        path (pathlib.Path): Destination file path.
        rows (list): Stats dicts, in listing order.
        preamble (iterable of str, optional): Lines written above the table,
            e.g. the config and filter settings. Defaults to ().

    Returns:
        None
    """
    with open(path, 'w') as f:
        for line in preamble:
            f.write(line + '\n')
        if preamble:
            f.write('\n')
        f.write(format_table(rows) + '\n')
    print(f'Wrote halo-sample statistics to: {path}')
