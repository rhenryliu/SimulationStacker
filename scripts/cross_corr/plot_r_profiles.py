"""plot_r_profiles.py
====================
Turn the ``.npz`` files written by ``make_r_profiles.py`` into the Singh et al.
(2020) Fig. 1 analogue and the Gate A decision metrics of
``docs/cross_correlation_notes.md`` (Sec. 6, Task 1).

Figures (one per field definition, baryons and electrons):

    rows:    r_gb(R),  r_bm(R),  r_gm(R),  C(R) = r_bm r_gm / r_gb
    columns: Sigma,    DSigma,   Upsilon(R0),  Y(Rmax)
    curves:  one per simulation, with its within-projection jackknife band

Both reference radii are read from each run's own metadata, never assumed.

The bottom row is the calibration factor of the v0.2 addendum Eq. (A12), whose
Sec. 4.3 argues it is what should be plotted "in place of the individual r's,
which are convention-free only in combination and which are not bounded by
unity".  The three rows above it are its three legs; ``r_gm`` is the same curve
in the baryon and electron figures -- it touches neither field -- but appears
in both so each figure stands alone.  The electron figure's C is the analogue
``r_em r_gm / r_ge``.

Nothing here is recomputed from the maps: every quantity is assembled from the
coefficients and their jackknife stacks already in the ``.npz``, per jackknife
realization.

Metrics printed and written alongside the figure:

    - max_R |x - 1| per quantity, filter and simulation;
    - the cross-simulation scatter, at each aperture, of BOTH C and the Task 1
      statistic r_bm/r_gb.  Gate A's verdict is taken on C, per addendum
      Task 10; r_bm/r_gb is reported without a verdict because it is what
      tasks_1_to_4_record.md and the U6 R0 scan are quoted against.

Usage
-----
    cd scripts/
    python cross_corr/plot_r_profiles.py -p configs/cross_corr/r_profiles_z05.yaml
"""

import argparse
import glob
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.append('../src/')
import rprofiles as rp

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    # Computer Modern rather than dejavuserif: in dejavuserif the
    # \Upsilon macro renders pixel-identically to an upright Latin Y,
    # which makes the Baldauf filter indistinguishable from the Park
    # et al. Y transform and from the Y_ab amplitudes.  'cm' draws the
    # forked Upsilon.
    'mathtext.fontset': 'cm',
    'text.usetex': False,      # no LaTeX in the cosmodesi environment
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 9,
})

#: Rows of the figure: (numerator, denominator or None, label).  Each of the
#: first two entries is a field pair, or a tuple of field pairs whose
#: coefficients are multiplied.
#:
#: The bottom row is the calibration factor
#:
#:     C = r_bm r_gm / r_gb = Y_bm Y_gm / (Y_mm Y_gb)
#:
#: of the v0.2 addendum Eq. (A12), whose Sec. 4.3 argues it is what should be
#: plotted "in place of the individual r's, which are convention-free only in
#: combination and which are not bounded by unity".  It equals unity
#: identically whenever the galaxies' correlation with the gas is entirely
#: mediated by the matter field, whatever the feedback does, so its departure
#: from one measures direct galaxy-gas stochasticity.  The three rows above it
#: are exactly its three legs.
BARYON_ROWS = [
    (('g', 'b'), None, r'$r_{gb}$'),
    (('b', 'm'), None, r'$r_{bm}$'),
    (('g', 'm'), None, r'$r_{gm}$'),
    ((('b', 'm'), ('g', 'm')), (('g', 'b'),),
     r'$r_{bm}\,r_{gm}\,/\,r_{gb}$'),
]

ELECTRON_ROWS = [
    (('g', 'e'), None, r'$r_{ge}$'),
    (('e', 'm'), None, r'$r_{em}$'),
    (('g', 'm'), None, r'$r_{gm}$'),
    ((('e', 'm'), ('g', 'm')), (('g', 'e'),),
     r'$r_{em}\,r_{gm}\,/\,r_{ge}$'),
]

#: Cross-simulation scatter statistics, as ``(numerator, denominator, label,
#: carries_verdict)``.  These are deliberately NOT derived from the figure rows.
#: Gate A is re-taken on ``C`` per addendum Task 10, but ``r_bm/r_gb`` is the
#: statistic every number in ``tasks_1_to_4_record.md`` and the U6 R0 scan is
#: quoted against, so it stays in the report without a verdict rather than
#: disappearing when it stopped being a row.
BARYON_SCATTER = [
    ((('b', 'm'), ('g', 'm')), (('g', 'b'),),
     'C = r_bm r_gm / r_gb', True),
    ((('b', 'm'),), (('g', 'b'),),
     'r_bm / r_gb  (Task 1 statistic)', False),
]

ELECTRON_SCATTER = [
    ((('e', 'm'), ('g', 'm')), (('g', 'e'),),
     'C = r_em r_gm / r_ge', True),
    ((('e', 'm'),), (('g', 'e'),),
     'r_em / r_ge  (Task 1 statistic)', False),
]

#: Column order.  Only the filters actually present in the loaded runs are
#: drawn, so a directory of pre-Y-transform ``.npz`` still plots.
FILTER_ORDER = ('Sigma', 'DSigma', 'Upsilon', 'Ytransform')

#: Column titles.  All four are mathtext, deliberately: with a plain Latin Y
#: column beside them, an ASCII 'DSigma' next to a typeset ``$Y(R_{max})$``
#: invites exactly the filter/amplitude confusion the theory documents devote a
#: notation table to.  'cm' is selected above so the forked Upsilon is
#: distinguishable from the Park et al. Y.
STATIC_FILTER_LABELS = {'Sigma': r'$\Sigma$', 'DSigma': r'$\Delta\Sigma$'}


def code_family(sim_type, sim_name):
    """Return the code family a run belongs to, for cross-code statistics.

    Gate A is explicitly a cross-*code* test (``docs/cross_correlation_notes.md``
    Sec. 6: "Validation must therefore be cross-code ... not merely
    cross-parameter within one code"), so the FLAMINGO feedback variants count
    as one family, and Illustris and IllustrisTNG count as two despite sharing
    a ``sim_type`` -- they are different galaxy-formation models.

    Args:
        sim_type (str): Suite name from the run metadata.
        sim_name (str): Simulation name from the run metadata.

    Returns:
        str: Code family label.
    """
    if sim_type == 'IllustrisTNG':
        return 'Illustris' if sim_name.startswith('Illustris') else 'TNG'
    return sim_type


def is_fiducial_variant(sim_type, sim_name, feedback):
    """Return whether a run is its suite's fiducial feedback variant.

    Each suite spells its fiducial differently: FLAMINGO names the directory
    after the box (``L1_m9``), SIMBA calls it ``s50``, and the IllustrisTNG
    runs carry no feedback variant at all.

    Args:
        sim_type (str): Suite name.
        sim_name (str): Simulation name.
        feedback (str): Feedback variant, or the string ``'None'``.

    Returns:
        bool: True if this is the suite's fiducial run.
    """
    if sim_type == 'FLAMINGO':
        return feedback == sim_name
    if sim_type == 'SIMBA':
        return feedback == 's50'
    return feedback in ('None', '')


def representative_runs(runs):
    """Pick exactly one run per code family for the cross-code statistic.

    Gate A thresholds the scatter *between codes*, so each family contributes
    once.  The fiducial feedback variant represents its family where one is
    identifiable; otherwise the first member does, so a family is never
    dropped silently.

    Args:
        runs (list): Runs from :func:`load_runs`.

    Returns:
        list: One run per code family, ordered by family name.
    """
    by_family = {}
    for run in runs:
        by_family.setdefault(run['family'], []).append(run)
    reps = []
    for family in sorted(by_family):
        members = by_family[family]
        fiducial = [m for m in members if m['is_fiducial']]
        reps.append(fiducial[0] if fiducial else members[0])
    return reps


def load_runs(npz_dir, wanted=None):
    """Load the r-profile ``.npz`` files belonging to one config.

    Args:
        npz_dir (pathlib.Path): Directory holding ``r_profiles_*.npz``.
        wanted (set, optional): Set of ``(sim_type, sim_name, feedback,
            snapshot)`` tuples to keep.  Matching on the full key rather than
            on the snapshot alone matters because every config writes into the
            same directory, so a stale or unrelated run that happens to reuse a
            snapshot number would otherwise be pulled silently into the
            figures and the Gate A statistics.  Defaults to None (keep all).

    Returns:
        list: One dict per run, sorted by code family then label.
    """
    runs = []
    for path in sorted(glob.glob(str(npz_dir / 'r_profiles_*.npz'))):
        data = np.load(path, allow_pickle=False)
        key = (str(data['meta_sim_type']), str(data['meta_sim_name']),
               str(data['meta_feedback']), int(data['meta_snapshot']))
        if wanted is not None and key not in wanted:
            continue
        runs.append({
            'path': path,
            'label': str(data['meta_label']),
            'sim_type': key[0],
            'sim_name': key[1],
            'feedback': key[2],
            'family': code_family(key[0], key[1]),
            'is_fiducial': is_fiducial_variant(key[0], key[1], key[2]),
            'snapshot': key[3],
            'projection': str(data['meta_projection']),
            'redshift': float(data['meta_redshift']),
            'n_galaxies': int(data['meta_n_galaxies']),
            'radii': data['radii'],
            # Both reference radii are per run, not global constants: the
            # configs have moved R0 between 1 and 2 arcmin.  Read them rather
            # than assuming, so a mixed directory cannot mislabel a curve.
            'r0': (float(data['meta_r0_arcmin'])
                   if 'meta_r0_arcmin' in data.files else rp.R0_ARCMIN),
            # None for a run written before the Y transform existed; the
            # column is then simply absent rather than mismasked.
            'rmax': (float(data['meta_ytransform_rmax'])
                     if 'meta_ytransform_rmax' in data.files else None),
            'filters': [f for f in FILTER_ORDER
                        if f'r_gb_{f}' in data.files],
            'data': data,
        })
    return sorted(runs, key=lambda r: (r['family'], r['label']))


def common_filters(runs):
    """Return the filters every loaded run carries, in :data:`FILTER_ORDER`.

    A column is drawn only where every curve can be drawn: a filter present in
    some runs and not others would produce a panel whose cross-simulation
    scatter is taken over a different set of simulations than its neighbours,
    which is exactly the kind of silent inhomogeneity the Gate A statistic must
    not have.

    Args:
        runs (list): Runs from :func:`load_runs`.

    Returns:
        list: Filter names common to every run.

    Raises:
        SystemExit: If the runs share no filter at all, which means the
            directory holds nothing this script can plot.
    """
    if not runs:
        return []
    shared = [f for f in FILTER_ORDER
              if all(f in run['filters'] for run in runs)]
    if not shared:
        raise SystemExit(
            'The loaded runs share no filter; expected some of '
            f'{FILTER_ORDER}. Re-run make_r_profiles.py.')

    dropped = sorted({f for run in runs for f in run['filters']}
                     - set(shared))
    if dropped:
        print(f'WARNING: filter(s) {dropped} are missing from at least one '
              'run and are therefore not plotted. Re-run make_r_profiles.py '
              'for every run in the config to restore the column.')
    return shared


def every_run_has(runs, num, den, filters):
    """Return whether every run carries the coefficients a quantity needs.

    Both the full-map array and the jackknife stack are required: ``series``
    reads both, and the error bar is formed from the stack.

    Args:
        runs (list): Runs from :func:`load_runs`.
        num (tuple): Numerator field pair or tuple of pairs.
        den (tuple or None): Denominator field pair or tuple of pairs.
        filters (list): Filter columns, from :func:`common_filters`.

    Returns:
        bool: True if every run can supply the quantity for every filter.
    """
    needed = as_pairs(num) + as_pairs(den)
    return all(f'r_{a}{b}_{filt}' in run['data'].files
               and f'rjk_{a}{b}_{filt}' in run['data'].files
               for run in runs for (a, b) in needed for filt in filters)


def common_rows(runs, rows, filters):
    """Drop rows whose field pair is missing from any loaded run.

    The companion to :func:`common_filters`, for the other axis of the figure.
    ``data/r_profiles/`` is a shared, accumulating cache -- every config writes
    into it -- so a run predating a coefficient can sit beside a current one.
    A missing *filter* already degrades to a dropped column with a warning; a
    missing *pair* would instead raise ``KeyError`` out of ``series`` deep
    inside the figure or the metrics, which is a worse failure for the same
    cause.  ``r_gm`` is the pair this actually applies to: every ``.npz``
    written before it was added carries ``Y_gm`` but no ``r_gm``.

    Args:
        runs (list): Runs from :func:`load_runs`.
        rows (list): Row specification, e.g. :data:`BARYON_ROWS`.
        filters (list): Filter columns, from :func:`common_filters`.

    Returns:
        list: The rows every run can supply, in the given order.
    """
    kept = []
    for num, den, label in rows:
        if every_run_has(runs, num, den, filters):
            kept.append((num, den, label))
        else:
            print(f'WARNING: row {label} needs a coefficient missing from at '
                  'least one run and is dropped. Re-run make_r_profiles.py '
                  'for every run in the config to restore the row.')
    return kept


def common_stats(runs, stats, filters):
    """Drop scatter statistics whose coefficients are missing from any run.

    The companion to :func:`common_rows` for the Gate A statistics, which are
    deliberately independent of the figure rows and so are never seen by it.
    They reach :func:`series` by the same route and must degrade the same way:
    without this, a run lacking ``r_gm`` writes both figures and then raises
    ``KeyError`` inside :func:`gate_a_metrics`, before either metrics file is
    written.  That is not hypothetical -- the Illustris-1 and SIMBA ``.npz``
    predate ``r_gm`` and are still in ``data/r_profiles/``, excluded from the
    current configs but a config edit away from being loaded again.

    Args:
        runs (list): Runs from :func:`load_runs`.
        stats (list): Scatter statistics as ``(num, den, label,
            carries_verdict)``, e.g. :data:`BARYON_SCATTER`.
        filters (list): Filter columns, from :func:`common_filters`.

    Returns:
        list: The statistics every run can supply, in the given order.
    """
    kept = []
    for num, den, label, carries_verdict in stats:
        if every_run_has(runs, num, den, filters):
            kept.append((num, den, label, carries_verdict))
        else:
            print(f'WARNING: scatter statistic "{label}" needs a coefficient '
                  'missing from at least one run and is dropped from the '
                  'Gate A report. Re-run make_r_profiles.py for every run in '
                  'the config to restore it.')
    return kept


def as_pairs(spec):
    """Normalize a row's numerator or denominator to a tuple of field pairs.

    Accepts the single-pair form ``('b', 'm')`` used throughout the original
    row specification, the multi-pair form ``(('b', 'm'), ('g', 'm'))`` needed
    for the calibration factor, and None for an absent denominator.

    Args:
        spec (tuple or None): Field pair, tuple of field pairs, or None.

    Returns:
        tuple: Tuple of ``(a, b)`` field pairs, empty for None.
    """
    if spec is None or len(spec) == 0:
        return ()
    if isinstance(spec[0], str):
        if len(spec) != 2:
            raise ValueError(
                f'A field pair must have exactly two entries, got {spec!r}. '
                'Pass a tuple of pairs to multiply several coefficients.')
        return (spec,)
    return tuple(spec)


def series(run, num, den, filt):
    """Extract a coefficient, ratio or product, and its jackknife error.

    Every quantity here is formed **per jackknife realization** and the spread
    taken afterwards, never by Gaussian propagation of marginal errors.  That
    is essential rather than stylistic: the coefficients entering one product
    are measured on the same map and are strongly correlated, so propagating
    marginal errors would badly misestimate the uncertainty and would miss the
    partial cancellation that makes the combination well behaved
    (``filter_specification.md`` Sec. 8; addendum Appendix A trap 5).

    Values are recomputed from the stored ``r_*``/``rjk_*`` arrays rather than
    read from the stored ``rerr_``/``ratio_``/``ratioerr_`` arrays, so that one
    code path serves bare coefficients, ratios and products alike.  This is a
    no-op for the cases that have a stored counterpart: across all production
    runs and filters the recomputation reproduces them to exactly zero.

    Args:
        run (dict): One entry from :func:`load_runs`.
        num (tuple): Numerator field pair ``('b', 'm')``, or tuple of pairs
            ``(('b', 'm'), ('g', 'm'))`` whose coefficients are multiplied.
        den (tuple or None): Denominator field pair or tuple of pairs, or None.
        filt (str): Filter name.

    Returns:
        tuple: ``(values, errors)``, each of shape ``(n_radii,)``.  Both
        derived filters are masked to NaN outside the band they inform:

        - Upsilon at ``R <= R0``, where it nulls everything below its
          reference radius (:func:`rprofiles.upsilon_defined_mask`);
        - the Y transform at ``R >= 0.8 Rmax``, where it vanishes into the
          reference annulus (:func:`rprofiles.ytransform_defined_mask`).

        Every consumer here already drops NaN, so masking once at the source
        keeps the figures and the Gate A metrics consistent -- they cannot
        disagree about which bins exist.
    """
    d = run['data']

    num_pairs = as_pairs(num)
    den_pairs = as_pairs(den)
    if not num_pairs:
        raise ValueError('series needs at least one numerator field pair; '
                         f'got num={num!r}.')

    a, b = num_pairs[0]
    values = d[f'r_{a}{b}_{filt}']
    jk = d[f'rjk_{a}{b}_{filt}']
    with np.errstate(invalid='ignore', divide='ignore'):
        for a, b in num_pairs[1:]:
            values = values * d[f'r_{a}{b}_{filt}']
            jk = jk * d[f'rjk_{a}{b}_{filt}']
        for a, b in den_pairs:
            # NaN rather than +/-inf on an exact zero, matching
            # rprofiles._coefficient.  An inf would survive every nanmax and
            # nanmean downstream and be reported as though it were a number.
            den_full = d[f'r_{a}{b}_{filt}']
            den_jk = d[f'rjk_{a}{b}_{filt}']
            values = np.where(den_full != 0.0, values / den_full, np.nan)
            jk = np.where(den_jk != 0.0, jk / den_jk, np.nan)
    errors = rp.jackknife_error(jk, axis=0)

    defined = None
    if filt == 'Upsilon':
        defined = rp.upsilon_defined_mask(run['radii'], run['r0'])
    elif filt == 'Ytransform' and run.get('rmax') is not None:
        defined = rp.ytransform_defined_mask(run['radii'], run['rmax'])

    if defined is not None:
        values = np.where(defined, values, np.nan)
        errors = np.where(defined, errors, np.nan)
    return values, errors


def _reference_label(runs, key, symbol, argument):
    """Build a derived filter's column title from the runs' own metadata.

    Args:
        runs (list): Entries from :func:`load_runs`.
        key (str): Run key holding the reference radius, ``'r0'`` or
            ``'rmax'``.
        symbol (str): Mathtext for the filter, e.g. ``r'\\Upsilon'``.
        argument (str): Mathtext for the argument, e.g. ``r'R_0'``.

    Returns:
        str: A mathtext label such as ``$\\Upsilon(R_0=1')$``.  If the runs
        disagree on the reference radius -- which would make the column
        incomparable across curves -- the label says so rather than quietly
        picking one.
    """
    values = sorted({round(float(r[key]), 6) for r in runs
                     if r.get(key) is not None})
    if len(values) != 1:
        return (f'${symbol}$ (MIXED ${argument}$: '
                + ', '.join(f"{v:g}'" for v in values) + ')')
    return f'${symbol}({argument}=' + f'{values[0]:g}' + r"')$"


def upsilon_label(runs):
    """Build the Upsilon (Baldauf et al. 2010) column title.

    Args:
        runs (list): Entries from :func:`load_runs`.

    Returns:
        str: e.g. ``$\\Upsilon(R_0=1')$``.
    """
    return _reference_label(runs, 'r0', r'\Upsilon', 'R_0')


def ytransform_label(runs):
    """Build the Park et al. (2021) Y-transform column title.

    Plain italic Y with an explicit ``R_max`` argument, per the notation table
    of ``docs/cross_correlation_notes_v0.3_response.md``: a radial argument
    marks a filter, field subscripts mark an amplitude, and the forked Upsilon
    distinguishes the Baldauf filter from this one.

    Args:
        runs (list): Entries from :func:`load_runs`.

    Returns:
        str: e.g. ``$Y(R_{max}=6')$``.
    """
    return _reference_label(runs, 'rmax', 'Y', r'R_{\rm max}')


def column_label(filt, runs):
    """Return the column title for one filter.

    Args:
        filt (str): Filter name.
        runs (list): Entries from :func:`load_runs`.

    Returns:
        str: Mathtext column title.
    """
    if filt == 'Upsilon':
        return upsilon_label(runs)
    if filt == 'Ytransform':
        return ytransform_label(runs)
    return STATIC_FILTER_LABELS.get(filt, filt)


def make_figure(runs, rows, filters, out_path, title, show_errors=True):
    """Draw the Singh Fig. 1 analogue for one field definition.

    Args:
        runs (list): Runs from :func:`load_runs`.
        rows (list): Row specification, e.g. :data:`BARYON_ROWS`.
        filters (list): Filter columns, from :func:`common_filters`.
        out_path (pathlib.Path): Output figure path.
        title (str): Figure suptitle.
        show_errors (bool, optional): Shade the jackknife band.  Defaults to
            True.
    """
    n_rows, n_cols = len(rows), len(filters)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols,
                                                      3.4 * n_rows),
                             sharex=True, sharey='row', squeeze=False)
    colours = matplotlib.colormaps['viridis'](
        np.linspace(0.05, 0.9, max(len(runs), 1)))

    for i, (num, den, row_label) in enumerate(rows):
        for j, filt in enumerate(filters):
            ax = axes[i][j]
            ax.axhline(1.0, color='k', ls='--', lw=1.2, zorder=1)
            for k, run in enumerate(runs):
                radii = run['radii']
                values, errors = series(run, num, den, filt)
                # Each derived filter nulls a band of the grid -- Upsilon
                # below R0, the Y transform below Rmax -- where the amplitude
                # is zero or over-subtracted and the coefficient meaningless.
                # `series` masks those bins to NaN; drop them rather than
                # plotting a gap-filled line through them.
                good = np.isfinite(values)
                if not good.any():
                    continue
                ax.plot(radii[good], values[good], color=colours[k], lw=1.8,
                        marker='o', ms=3.5,
                        label=run['label'] if (i == 0 and j == 0) else None)
                if show_errors:
                    ax.fill_between(radii[good],
                                    (values - errors)[good],
                                    (values + errors)[good],
                                    color=colours[k], alpha=0.18, lw=0)
            if i == 0:
                ax.set_title(column_label(filt, runs))
            if j == 0:
                ax.set_ylabel(row_label)
            if i == n_rows - 1:
                ax.set_xlabel(r'$R$ [arcmin]')
            ax.grid(alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(runs), 6),
               frameon=True, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, y=1.045, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to: {out_path}')


def quantity_name(num, den):
    """Build a compact plain-text name for a coefficient, ratio or product.

    Args:
        num (tuple): Numerator field pair or tuple of pairs.
        den (tuple or None): Denominator field pair or tuple of pairs.

    Returns:
        str: e.g. ``'gb'``, ``'bm/gb'`` or ``'bm.gm/gb'``.
    """
    top = '.'.join(f'{a}{b}' for a, b in as_pairs(num))
    bottom = '.'.join(f'{a}{b}' for a, b in as_pairs(den))
    return f'{top}/{bottom}' if bottom else top


def gate_a_metrics(runs, rows, filters, stats, data_max=None):
    """Compute the Task 1 deliverable metrics.

    Args:
        runs (list): Runs from :func:`load_runs`.
        rows (list): Row specification, used for ``max_dev``.
        filters (list): Filter columns, from :func:`common_filters`.
        stats (list): Scatter statistics as ``(num, den, label,
            carries_verdict)``, e.g. :data:`BARYON_SCATTER`.  Deliberately
            independent of ``rows``: the Gate A quantity is a scientific
            choice, not a consequence of what the figure happens to draw.
        data_max (float, optional): Largest observationally accessible
            aperture in arcmin.  Statistics are reported separately for
            apertures at or below it and for the diagnostic extension above
            it, and the Gate A verdict uses the data range only.  Defaults to
            None, which treats every aperture as in range.

    Returns:
        dict: ``{'max_dev': {...}, 'scatter': {...}}`` where ``max_dev`` maps
        ``(label, quantity, filter)`` to ``max_R |x - 1|`` and ``scatter`` maps
        ``statistic label`` to ``subset`` to ``filter`` to its per-radius
        cross-simulation spread.
    """
    max_dev = {}
    for num, den, _ in rows:
        name = quantity_name(num, den)
        for filt in filters:
            for run in runs:
                values, _ = series(run, num, den, filt)
                with np.errstate(invalid='ignore'):
                    dev = np.nanmax(np.abs(values - 1.0))
                max_dev[(run['label'], name, filt)] = float(dev)

    # Scatter of the ratio row -- the Gate A quantity.
    #
    # Gate A is a cross-CODE threshold ("Validation must therefore be
    # cross-code ... not merely cross-parameter within one code", theory note
    # Sec. 6), so the primary statistic keeps one representative run per code
    # family.  Pooling the three FLAMINGO feedback variants into it would let a
    # single code's parameter sweep dominate a statistic meant to measure
    # code-to-code disagreement.  The all-run scatter is computed too and
    # reported separately as the feedback-inclusive diagnostic.
    scatter = {}
    subsets = {
        'cross-code': representative_runs(runs) if runs else [],
        'all-runs': list(runs),
    }
    for num, den, stat_label, carries_verdict in stats:
        scatter[stat_label] = {'carries_verdict': carries_verdict}
        for subset_name, subset in subsets.items():
            if len(subset) < 2:
                continue
            scatter[stat_label][subset_name] = {}
            for filt in filters:
                stack = np.vstack([series(run, num, den, filt)[0]
                                   for run in subset])
                # The Upsilon bins at R <= R0 are all-NaN by construction, so
                # nanmean/nanstd legitimately reduce an empty slice there.
                with np.errstate(invalid='ignore'), \
                        warnings.catch_warnings():
                    warnings.filterwarnings('ignore',
                                            message='Mean of empty slice')
                    warnings.filterwarnings('ignore',
                                            message='Degrees of freedom <= 0')
                    # The Y transform's whole diagnostic extension is masked
                    # when Rmax sits at the top of the data range, so the
                    # extension-only reduction below is legitimately an
                    # all-NaN slice rather than a symptom.
                    warnings.filterwarnings('ignore',
                                            message='All-NaN slice encountered')
                    mean = np.nanmean(stack, axis=0)
                    std = np.nanstd(stack, axis=0, ddof=1)
                    frac = std / np.abs(mean)
                    radii = subset[0]['radii']
                    in_data = (np.ones(len(radii), dtype=bool) if data_max is
                               None else radii <= data_max + 1e-9)
                    scatter[stat_label][subset_name][filt] = {
                        'radii': radii,
                        'mean': mean,
                        'std': std,
                        'frac_std': frac,
                        'in_data': in_data,
                        'worst_data': (np.nanmax(frac[in_data])
                                       if in_data.any() else np.nan),
                        'worst_ext': (np.nanmax(frac[~in_data])
                                      if (~in_data).any() else np.nan),
                        'members': [r['label'] for r in subset],
                    }
    return {'max_dev': max_dev, 'scatter': scatter}


def report_metrics(metrics, runs, rows, filters, out_path, data_max=None):
    """Print the Gate A metrics and write them to a text file.

    Args:
        metrics (dict): Output of :func:`gate_a_metrics`.
        runs (list): Runs from :func:`load_runs`.
        rows (list): Row specification.
        filters (list): Filter columns, from :func:`common_filters`.
        out_path (pathlib.Path): Destination text file.
        data_max (float, optional): Largest observationally accessible
            aperture in arcmin, used only to flag a Y-transform reference
            radius that sits outside it.  Defaults to None.
    """
    lines = []

    def emit(text=''):
        print(text)
        lines.append(text)

    emit('=' * 78)
    emit('Task 1 metrics')
    emit('=' * 78)
    emit()
    emit('Runs:')
    for run in runs:
        emit(f"  {run['label']:24s} snap {run['snapshot']:3d}  "
             f"z={run['redshift']:.4f}  proj={run['projection']}  "
             f"N_gal={run['n_galaxies']}")
    emit()

    emit('max_R |x - 1|  (deliverable metric 1)')
    names = [quantity_name(num, den) for num, den, _ in rows]
    # One block per quantity rather than one wide row: with four quantities
    # and four filters a single row runs past 370 characters and stops being
    # readable in a terminal or a diff.
    for n in names:
        emit(f'  {n}:')
        emit(f"    {'simulation':24s}" + ''.join(f'{f:>14s}'
                                                 for f in filters))
        for run in runs:
            row = f"    {run['label']:24s}"
            for f in filters:
                row += f'{metrics["max_dev"][(run["label"], n, f)]:14.4f}'
            emit(row)
        emit()

    for stat_label, per_subset in metrics['scatter'].items():
        carries_verdict = per_subset.get('carries_verdict', False)
        emit('=' * 78)
        emit(f'Cross-simulation scatter of  {stat_label}')
        if carries_verdict:
            emit('  (deliverable metric 2, the Gate A quantity -- addendum '
                 'Task 10 re-takes')
            emit('   Gate A on this calibration factor rather than on '
                 'the Task 1 statistic')
            emit('   also reported in this file)')
        else:
            emit('  (reported for continuity with tasks_1_to_4_record.md and '
                 'the U6 R0 scan,')
            emit('   which quote this statistic; it carries no Gate A verdict '
                 'here)')
        emit('=' * 78)
        subset_titles = {
            'cross-code': 'Across CODES (one representative run per family)',
            'all-runs': ('Across ALL runs (diagnostic: includes feedback '
                         'variants within a code,'
                         '\n  so this is NOT the Gate A statistic)'),
        }
        for subset_name in ('cross-code', 'all-runs'):
            if subset_name not in per_subset:
                continue
            per_filter = per_subset[subset_name]
            emit(subset_titles[subset_name])
            any_filt = next(iter(per_filter.values()))
            emit(f"  members ({len(any_filt['members'])}): "
                 f"{', '.join(any_filt['members'])}")
            for filt, s in per_filter.items():
                emit(f'  {filt}:')
                emit(f"    {'R [arcmin]':>11}  {'mean':>10}  {'std':>10}  "
                     f"{'frac. std':>10}")
                for R, m, sd, fs, ind in zip(s['radii'], s['mean'], s['std'],
                                             s['frac_std'], s['in_data']):
                    tag = '' if ind else '   (extension)'
                    emit(f'    {R:11.3f}  {m:10.4f}  {sd:10.4f}  '
                         f'{fs:10.4f}{tag}')
                worst = s['worst_data']
                emit(f'    worst fractional scatter, data range: {worst:.4f}')
                if np.isfinite(s['worst_ext']):
                    emit(f'    worst fractional scatter, extension: '
                         f"{s['worst_ext']:.4f}")
                if (np.isfinite(worst) and subset_name == 'cross-code'
                        and carries_verdict):
                    if worst <= 0.10:
                        verdict = 'PASS  (<= 10%: fixed-transfer route)'
                    elif worst <= 0.20:
                        verdict = 'MARGINAL  (10-20%: parametrized-r route)'
                    else:
                        verdict = 'FAIL  (> 20%: revisit the estimator)'
                    emit(f'    Gate A on this filter (data range): {verdict}')
                emit()

    emit('Note: apertures above the config max_radius are a diagnostic')
    emit('extension, not observationally accessible, and the Gate A verdict')
    emit('uses the data range only.')
    if 'Upsilon' in filters:
        r0_values = sorted({round(float(r['r0']), 6) for r in runs})
        r0_text = ', '.join(f'{v:g}' for v in r0_values)
        emit(f'Note: Upsilon nulls everything below its reference radius R0 = '
             f'{r0_text} arcmin.')
        emit('It is identically zero at R = R0 (a genuine 0/0 in the '
             'coefficient)')
        emit('and over-subtracted below it, so every bin with R <= R0 is '
             'masked to')
        emit('NaN and excluded from both the curves and these metrics.')
        if len(r0_values) != 1:
            emit('WARNING: the runs do not share one R0, so the Upsilon '
                 'column is')
            emit('not comparable across curves.')

    if 'Ytransform' in filters:
        rmax_values = sorted({round(float(r['rmax']), 6) for r in runs
                              if r.get('rmax') is not None})
        rmax_text = ', '.join(f'{v:g}' for v in rmax_values)
        emit(f'Note: the Park et al. (2021) Y transform Y(R; Rmax) = Sigma(R) '
             f'- Sigma(Rmax),')
        emit(f'Rmax = {rmax_text} arcmin, vanishes identically at R = Rmax and '
             'carries almost')
        emit(f'no signal just below it, so bins with R >= '
             f'{rp.YT_USABLE_FRACTION:g} Rmax are masked to NaN')
        emit('and excluded from both the curves and these metrics (addendum '
             'Secs. 2.5, 8.2).')
        if len(rmax_values) != 1:
            emit('WARNING: the runs do not share one Rmax, so the Y-transform')
            emit('column is not comparable across curves.')
        if rmax_values and data_max is not None and max(rmax_values) > data_max:
            emit(f'WARNING: Rmax = {max(rmax_values):g} arcmin lies above the '
                 f'data range ({data_max:g} arcmin),')
            emit('so on real data this filter would reference an aperture the')
            emit('kSZ stack does not measure (v0.3 response Sec. 4.2).')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f'\nMetrics written to: {out_path}')


def main(path2config, verbose=True):
    """Build the figures and metrics for one config.

    Args:
        path2config (str): Path to the YAML configuration file.
        verbose (bool, optional): Unused placeholder for CLI symmetry.
    """
    with open(path2config) as f:
        config = yaml.safe_load(f)

    plot_cfg = config.get('plot', {})
    npz_dir = Path(plot_cfg.get('npz_path', '../data/r_profiles/'))
    fig_name = plot_cfg.get('fig_name', 'r_profiles')
    fig_type = plot_cfg.get('fig_type', 'pdf')

    wanted = {(suite['sim_type'], entry['name'],
               str(entry.get('feedback')), entry['snapshot'])
              for suite in config['simulations']
              for entry in suite['sims']}

    runs = load_runs(npz_dir, wanted=wanted)
    if not runs:
        raise SystemExit(
            f'No matching r_profiles_*.npz found in {npz_dir} for the runs in '
            f'{path2config}. Run make_r_profiles.py first.')
    if len(runs) < len(wanted):
        missing = wanted - {(r['sim_type'], r['sim_name'], r['feedback'],
                             r['snapshot']) for r in runs}
        print(f'WARNING: {len(missing)} configured run(s) have no .npz yet: '
              f'{sorted(missing)}')

    filters = common_filters(runs)
    baryon_rows = common_rows(runs, BARYON_ROWS, filters)
    electron_rows = common_rows(runs, ELECTRON_ROWS, filters)
    print(f'Filter columns: {filters}')
    print(f'Rows: baryon {[r[2] for r in baryon_rows]}, '
          f'electron {[r[2] for r in electron_rows]}')

    now = datetime.now()
    fig_dir = (Path(plot_cfg.get('fig_path', '../figures/'))
               / now.strftime('%Y-%m') / now.strftime('%m-%d'))

    make_figure(runs, baryon_rows, filters,
                fig_dir / f'{fig_name}_baryon.{fig_type}',
                'Baryon-matter cross-correlation coefficients '
                '(b = gas + stars + BH)')
    make_figure(runs, electron_rows, filters,
                fig_dir / f'{fig_name}_tau.{fig_type}',
                'Tau-matter cross-correlation coefficients '
                '(e = ionized gas)')

    print()
    data_max = config.get('stack', {}).get('max_radius')
    metrics = gate_a_metrics(runs, baryon_rows, filters,
                             common_stats(runs, BARYON_SCATTER, filters),
                             data_max=data_max)
    report_metrics(metrics, runs, baryon_rows, filters,
                   fig_dir / f'{fig_name}_metrics.txt', data_max=data_max)

    print()
    metrics_e = gate_a_metrics(
        runs, electron_rows, filters,
        common_stats(runs, ELECTRON_SCATTER, filters),
        data_max=data_max)
    report_metrics(metrics_e, runs, electron_rows, filters,
                   fig_dir / f'{fig_name}_metrics_electron.txt',
                   data_max=data_max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot Task 1 r-profiles and report the Gate A metrics.')
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/cross_corr/r_profiles_z05.yaml',
                        help='Path to the YAML configuration file.')
    args = vars(parser.parse_args())
    print(f'Arguments: {args}')
    main(**args)
