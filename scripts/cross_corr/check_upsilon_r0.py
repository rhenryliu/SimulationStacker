"""check_upsilon_r0.py
====================
Scan the Upsilon reference radius R0 and report what it costs and what it buys.

``Upsilon(R; R0) = DSigma(R) - (R0/R)^2 DSigma(R0)`` nulls all information
below R0, so R0 is a scale cut with two opposing effects on the Task 1
deliverable ``r_bm/r_gb``:

- raising R0 removes the small-scale, feedback-sensitive modes that drive the
  cross-code disagreement, which should *reduce* the Gate A scatter;
- raising R0 also deletes every aperture bin at or below it, and suppresses the
  amplitude in the bins just above it (Upsilon vanishes continuously as
  R -> R0), so the surviving bins are fewer and, near R0, are a small
  difference of comparable numbers in which any systematic is magnified.

The production configs moved R0 from 1 to 2 arcmin without a measurement of
that trade-off.  This script supplies it.

The scan is cheap because Upsilon never needs its own convolution: it is a
linear combination of DSigma amplitudes, so the DSigma amplitudes are measured
ONCE on a radius grid containing every candidate R0, and each R0 is then
assembled arithmetically from them.  This is exactly what
``rprofiles.compute_Y_matrix`` does internally for a single R0, and is verified
against a direct convolution and against the stamp filter in
``tests/test_rprofiles.py::TestUpsilonConstruction``.

Usage
-----
    cd scripts/
    python cross_corr/check_upsilon_r0.py -p configs/cross_corr/r_profiles_z05.yaml
    python cross_corr/check_upsilon_r0.py -p configs/cross_corr/r_profiles_z05.yaml \
        --r0 1.0 1.5 2.0 2.5
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.append('../src/')
import rprofiles as rp
from stacker import SimulationStacker
from utils import comoving_to_arcmin
sys.path.append(str(Path(__file__).parent))
from make_r_profiles import aperture_radii, load_component_fields, sim_label
from plot_r_profiles import code_family, is_fiducial_variant

#: Reference radii scanned by default, in arcmin.  1.0 is the original Task 1
#: value, 2.0 the current production one.
DEFAULT_R0 = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)

#: The Gate A quantity and the coefficients it is built from.
GATE_PAIRS = [('b', 'm'), ('g', 'b'), ('e', 'm'), ('g', 'e')]
GATE_RATIOS = [(('b', 'm'), ('g', 'b')), (('e', 'm'), ('g', 'e'))]
GATE_KEY = (('b', 'm'), ('g', 'b'))


def reassemble_upsilon(Ymat, r0, radii_out):
    """Rebuild a Y matrix with Upsilon formed at a different reference radius.

    ``compute_Y_matrix`` measures Sigma and DSigma by convolution and derives
    Upsilon as ``Y_DSigma(R) - (r0/R)^2 Y_DSigma(r0)``.  Because that step is
    pure arithmetic on the amplitudes, a whole scan over r0 can reuse one set
    of measured DSigma amplitudes.

    Args:
        Ymat (dict): Output of :func:`rprofiles.compute_Y_matrix`, measured on
            a radius grid that contains ``r0`` and all of ``radii_out``.
        r0 (float): Upsilon reference radius, in arcmin.
        radii_out (np.ndarray): Aperture radii to report, in arcmin.

    Returns:
        dict: A Y matrix in the same layout as ``Ymat``, restricted to
        ``radii_out`` and with ``'Upsilon'`` built at ``r0``.

    Raises:
        ValueError: If ``r0`` or any requested aperture is absent from the
            measured radius grid.
    """
    radii_all = np.asarray(Ymat['radii'], dtype=np.float64)

    def _index(value):
        hits = np.flatnonzero(np.isclose(radii_all, value, rtol=0, atol=1e-9))
        if hits.size == 0:
            raise ValueError(
                f'Radius {value!r} is not on the measured grid {radii_all}; '
                'compute_Y_matrix must be called with every candidate r0 '
                'included in `radii`.')
        return int(hits[0])

    i0 = _index(r0)
    idx = np.array([_index(R) for R in radii_out])
    radii_out = np.asarray(radii_out, dtype=np.float64)
    scale = (r0 / radii_out) ** 2

    Y, Y_jk = {}, {}
    for filt in rp.BASE_FILTERS:
        Y[filt] = {p: v[idx] for p, v in Ymat['Y'][filt].items()}
        Y_jk[filt] = {p: v[:, idx] for p, v in Ymat['Y_jk'][filt].items()}

    Y['Upsilon'] = {}
    Y_jk['Upsilon'] = {}
    for p, ds in Ymat['Y']['DSigma'].items():
        Y['Upsilon'][p] = ds[idx] - scale * ds[i0]
    for p, ds in Ymat['Y_jk']['DSigma'].items():
        Y_jk['Upsilon'][p] = ds[:, idx] - scale[None, :] * ds[:, i0][:, None]

    out = dict(Ymat)
    out.update({'radii': radii_out, 'r0': float(r0), 'Y': Y, 'Y_jk': Y_jk})
    return out


def upsilon_snr(Ymat, pair):
    """Return |Y_Upsilon| / jackknife error per aperture for one field pair.

    Upsilon's amplitude vanishes as R approaches R0, so it is worth asking
    separately whether the bins just above R0 are actually noisy or merely
    small.  The jackknife error shrinks with the amplitude -- both are
    dominated by the same difference of DSigma amplitudes -- so this ratio
    answers that directly rather than leaving it to be inferred from an error
    bar on a coefficient.

    Args:
        Ymat (dict): A Y matrix from :func:`reassemble_upsilon`.
        pair (tuple): Field pair, e.g. ``('b', 'm')``.

    Returns:
        np.ndarray: ``|Y| / sigma_jk``, shape ``(n_radii,)``.
    """
    y = rp.get_Y(Ymat, 'Upsilon', *pair)
    err = rp.jackknife_error(rp.get_Y(Ymat, 'Upsilon', *pair, jackknife=True),
                             axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.abs(y) / err


def process_simulation(sim_type, entry, cfg, radii, r0_list, verbose=True):
    """Measure the DSigma amplitudes once and scan R0 for one simulation.

    Args:
        sim_type (str): Simulation suite.
        entry (dict): One entry of the config ``sims`` list.
        cfg (dict): The ``stack`` config block.
        radii (np.ndarray): Reported aperture radii, in arcmin.
        r0_list (sequence): Candidate reference radii, in arcmin.
        verbose (bool, optional): Print progress.  Defaults to True.

    Returns:
        dict: Per-R0 coefficients, ratios, errors and amplitude SNR, plus the
        run's identifying metadata.
    """
    name, snapshot = entry['name'], entry['snapshot']
    feedback = entry.get('feedback')
    n_pixels = entry['n_pixels']
    projection = cfg.get('projections', ['yz'])[0]
    dr = float(cfg.get('dr_arcmin', rp.DR_ARCMIN))
    label = sim_label(sim_type, name, feedback)

    stacker = SimulationStacker(name, snapshot, nPixels=n_pixels,
                                simType=sim_type, feedback=feedback,
                                z=float(cfg.get('redshift', 0.5)))
    z_true = float(np.asarray(stacker.header['Redshift']).ravel()[0])
    lbox = float(stacker.header['BoxSize'])
    pixel_arcmin = comoving_to_arcmin(lbox, z_true,
                                      cosmo=stacker.cosmo) / n_pixels

    print(f'\n{"=" * 72}\n{label} snapshot {snapshot} (z={z_true:.4f})\n'
          f'{"=" * 72}')
    print(f'  grid {n_pixels}^2, pixel {pixel_arcmin:.5f} arcmin, '
          f'dr {dr} arcmin')

    # One measured grid covering both the reported apertures and every
    # candidate reference radius.
    grid = np.unique(np.concatenate([np.asarray(radii, dtype=np.float64),
                                     np.asarray(r0_list, dtype=np.float64)]))
    print(f'  measuring DSigma at {len(grid)} radii, '
          f'scanning {len(r0_list)} reference radii')

    # Fail fast: build every kernel and discard it, so an under-resolved
    # aperture or one too large for the periodic box raises here rather than
    # after the multi-gigabyte field load below.
    for guard in grid:
        rp.build_aperture_kernel(n_pixels, pixel_arcmin, float(guard),
                                 'DSigma', dr)

    deltas = load_component_fields(stacker, n_pixels, projection, cfg,
                                   verbose=verbose)
    galaxies, halo_mask = rp.make_galaxy_field(
        stacker, projection, n_pixels,
        float(cfg.get('halo_abundance_target', 5e-4)),
        parent_mass_upper=(None if cfg.get('parent_mass_upper') is None
                           else float(cfg['parent_mass_upper'])))
    nbar_pix = galaxies.sum() / float(n_pixels * n_pixels)
    deltas['g'] = rp.to_overdensity(galaxies)
    del galaxies
    print(f'  {halo_mask.size} SHAM galaxies '
          f'(nbar = {nbar_pix:.4e} per pixel)', flush=True)

    print('  computing filtered amplitudes ...', flush=True)
    Ymat = rp.compute_Y_matrix(deltas, pixel_arcmin, radii=grid, dr=dr,
                               r0=float(r0_list[0]), nbar_pix=nbar_pix,
                               n_jk_side=int(cfg.get('n_jk_side',
                                                     rp.N_JK_SIDE)))
    del deltas

    # Free consistency guard: compute_Y_matrix was asked for r0_list[0], so its
    # own Upsilon block is the reassembly for that reference radius on the full
    # grid.  Reproducing it exactly confirms the scan is the production
    # computation amortized, not an approximation to it.  Costs no FFTs.
    check = reassemble_upsilon(Ymat, float(r0_list[0]), Ymat['radii'])
    for pair, ref in Ymat['Y']['Upsilon'].items():
        got = check['Y']['Upsilon'][pair]
        if not np.array_equal(got, ref):
            raise AssertionError(
                f'reassemble_upsilon disagrees with compute_Y_matrix for '
                f'{pair} at r0={r0_list[0]}: max |diff| = '
                f'{np.max(np.abs(got - ref)):.3e}')
    print('  reassembly reproduces compute_Y_matrix bit-for-bit', flush=True)

    out = {
        'label': label,
        'family': code_family(sim_type, name),
        'is_fiducial': is_fiducial_variant(sim_type, name, str(feedback)),
        'radii': np.asarray(radii, dtype=np.float64),
        'pixel_arcmin': pixel_arcmin,
        'redshift': z_true,
        'r0': {},
    }
    for r0 in r0_list:
        sub = reassemble_upsilon(Ymat, float(r0), radii)
        prof = rp.r_profiles(sub, pairs=GATE_PAIRS, ratios=GATE_RATIOS)
        defined = rp.upsilon_defined_mask(radii, r0)
        out['r0'][float(r0)] = {
            'defined': defined,
            'ratio': np.where(defined, prof['ratio'][GATE_KEY]['Upsilon'],
                              np.nan),
            'ratio_err': np.where(defined,
                                  prof['ratio_err'][GATE_KEY]['Upsilon'],
                                  np.nan),
            'r_bm': np.where(defined, prof['r'][('b', 'm')]['Upsilon'], np.nan),
            'r_gb': np.where(defined, prof['r'][('g', 'b')]['Upsilon'], np.nan),
            'snr_bm': np.where(defined, upsilon_snr(sub, ('b', 'm')), np.nan),
            'snr_gb': np.where(defined, upsilon_snr(sub, ('g', 'b')), np.nan),
        }
    return out


def cross_code_scatter(runs, r0, data_max):
    """Return the Gate A cross-code scatter of r_bm/r_gb at one R0.

    One run per code family contributes, as in ``plot_r_profiles`` -- pooling
    a single code's feedback variants would let its parameter sweep drive a
    statistic defined as cross-code.

    Args:
        runs (list): Outputs of :func:`process_simulation`.
        r0 (float): Reference radius, in arcmin.
        data_max (float): Largest observationally accessible aperture, arcmin.

    Returns:
        dict: ``radii``, per-aperture ``frac_std``, the worst value inside the
        data range, and the number of contributing apertures.
    """
    by_family = {}
    for run in runs:
        by_family.setdefault(run['family'], []).append(run)
    reps = []
    for family in sorted(by_family):
        members = by_family[family]
        fiducial = [m for m in members if m['is_fiducial']]
        reps.append(fiducial[0] if fiducial else members[0])

    radii = reps[0]['radii']
    stack = np.vstack([r['r0'][float(r0)]['ratio'] for r in reps])
    # Bins at or below R0 are all-NaN by construction, so nanmean/nanstd
    # legitimately reduce an empty slice there.
    with np.errstate(invalid='ignore', divide='ignore'), \
            warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        warnings.filterwarnings('ignore', message='Degrees of freedom <= 0')
        warnings.filterwarnings('ignore',
                                message='All-NaN slice encountered')
        mean = np.nanmean(stack, axis=0) if len(reps) > 1 else stack[0]
        std = (np.nanstd(stack, axis=0, ddof=1) if len(reps) > 1
               else np.full(len(radii), np.nan))
        frac = std / np.abs(mean)
    in_data = radii <= data_max + 1e-9
    usable = in_data & np.isfinite(frac)

    # Upsilon vanishes as R -> R0, so the first surviving bin is a small
    # difference of comparable DSigma amplitudes and magnifies anything that
    # differs between codes.  Reporting the statistic with and without it
    # separates an artefact of the scale cut from genuine cross-code
    # disagreement.
    trimmed = usable.copy()
    first = np.flatnonzero(usable)
    if first.size:
        trimmed[first[0]] = False

    return {
        'radii': radii,
        'n_codes': len(reps),
        'mean': mean,
        'frac_std': frac,
        'in_data': in_data,
        'worst_data': float(np.max(frac[usable])) if usable.any() else np.nan,
        'worst_trimmed': (float(np.max(frac[trimmed])) if trimmed.any()
                          else np.nan),
        'first_bin': float(radii[first[0]]) if first.size else np.nan,
        'n_bins': int(usable.sum()),
    }


def verdict(worst):
    """Return the Gate A verdict string for a cross-code scatter.

    Args:
        worst (float): Worst fractional scatter inside the data range.

    Returns:
        str: Human-readable verdict.
    """
    if not np.isfinite(worst):
        return 'UNDEFINED (no usable aperture)'
    if worst <= 0.10:
        return 'PASS      (<= 10%: fixed-transfer route)'
    if worst <= 0.20:
        return 'MARGINAL  (10-20%: parametrized-r route)'
    return 'FAIL      (> 20%: revisit the estimator)'


def report(runs, r0_list, data_max, out_path):
    """Print the scan and write it to a text file.

    Args:
        runs (list): Outputs of :func:`process_simulation`.
        r0_list (sequence): Candidate reference radii, arcmin.
        data_max (float): Largest observationally accessible aperture, arcmin.
        out_path (pathlib.Path): Destination text file.
    """
    lines = []

    def emit(text=''):
        print(text)
        lines.append(text)

    emit('=' * 78)
    emit('Upsilon reference-radius scan')
    emit('=' * 78)
    emit(f'Runs: {", ".join(r["label"] for r in runs)}')
    emit(f'Data range: R <= {data_max:g} arcmin')
    emit()
    emit('Gate A: cross-code scatter of r_bm/r_gb (Upsilon), data range only')
    emit(f'  {"R0 [arcmin]":>11}  {"bins":>5}  {"worst":>8}  '
         f'{"1st bin":>8}  {"w/o 1st":>8}  verdict')
    summary = {}
    for r0 in r0_list:
        s = cross_code_scatter(runs, r0, data_max)
        summary[float(r0)] = s
        emit(f'  {r0:11.3f}  {s["n_bins"]:5d}  {s["worst_data"]:8.4f}  '
             f'{s["first_bin"]:8.3f}  {s["worst_trimmed"]:8.4f}  '
             f'{verdict(s["worst_data"])}')
    emit()
    emit(f'({summary[float(r0_list[0])]["n_codes"]} code families contribute; '
         'bins counts the apertures above R0 inside the data range.')
    emit(' "1st bin" is the smallest surviving aperture and "w/o 1st" is the')
    emit(' worst scatter excluding it: Upsilon vanishes as R -> R0, so that')
    emit(' bin magnifies any cross-code difference and can drive the')
    emit(' statistic on its own.)')
    emit()

    for r0 in r0_list:
        s = summary[float(r0)]
        emit('-' * 78)
        emit(f'R0 = {r0:g} arcmin')
        emit('-' * 78)
        emit(f'  {"R":>7}  {"mean ratio":>11}  {"frac. scatter":>13}   '
             + '  '.join(f'{r["label"][:16]:>16}' for r in runs))
        for i, R in enumerate(s['radii']):
            if not np.isfinite(s['frac_std'][i]) and not np.any(
                    [np.isfinite(r['r0'][float(r0)]['ratio'][i])
                     for r in runs]):
                continue
            tag = '' if s['in_data'][i] else '  (ext)'
            row = (f'  {R:7.3f}  {s["mean"][i]:11.4f}  '
                   f'{s["frac_std"][i]:13.4f}   '
                   + '  '.join(f'{r["r0"][float(r0)]["ratio"][i]:16.4f}'
                               for r in runs))
            emit(row + tag)
        emit()
        emit('  Amplitude signal-to-noise |Y_Upsilon| / sigma_jk '
             '(b-m / g-b pairs):')
        emit(f'  {"R":>7}   ' + '  '.join(f'{r["label"][:16]:>16}'
                                          for r in runs))
        for i, R in enumerate(s['radii']):
            cells = []
            any_finite = False
            for r in runs:
                a = r['r0'][float(r0)]['snr_bm'][i]
                b = r['r0'][float(r0)]['snr_gb'][i]
                if np.isfinite(a) or np.isfinite(b):
                    any_finite = True
                cells.append(f'{a:7.1f}/{b:<8.1f}')
            if any_finite:
                emit(f'  {R:7.3f}   ' + '  '.join(cells))
        emit()

    emit('=' * 78)
    emit('Reading this table')
    emit('=' * 78)
    emit('Raising R0 trades bins for cleanliness: it removes the small-scale')
    emit('modes that drive cross-code disagreement, but deletes every aperture')
    emit('at or below R0 and suppresses the amplitude just above it, where')
    emit('Upsilon vanishes continuously.  The "bins" column and the SNR table')
    emit('are the cost; the scatter column is the benefit.  A scatter that')
    emit('falls only because the noisiest bins were deleted is not a gain.')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n')
    print(f'\nScan written to: {out_path}')
    return summary


def make_figure(runs, r0_list, summary, data_max, out_path):
    """Plot r_bm/r_gb against aperture, one panel per candidate R0.

    Args:
        runs (list): Outputs of :func:`process_simulation`.
        r0_list (sequence): Candidate reference radii, arcmin.
        summary (dict): Per-R0 scatter from :func:`cross_code_scatter`.
        data_max (float): Largest observationally accessible aperture, arcmin.
        out_path (pathlib.Path): Destination figure path.
    """
    n = len(r0_list)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), sharey=True,
                             squeeze=False)
    colours = matplotlib.colormaps['viridis'](
        np.linspace(0.05, 0.9, max(len(runs), 1)))

    for j, r0 in enumerate(r0_list):
        ax = axes[0][j]
        ax.axhline(1.0, color='k', ls='--', lw=1.1)
        ax.axvline(r0, color='0.5', ls=':', lw=1.2)
        for k, run in enumerate(runs):
            d = run['r0'][float(r0)]
            good = np.isfinite(d['ratio'])
            if not good.any():
                continue
            ax.plot(run['radii'][good], d['ratio'][good], color=colours[k],
                    lw=1.7, marker='o', ms=3.2,
                    label=run['label'] if j == 0 else None)
            ax.fill_between(run['radii'][good],
                            (d['ratio'] - d['ratio_err'])[good],
                            (d['ratio'] + d['ratio_err'])[good],
                            color=colours[k], alpha=0.16, lw=0)
        ax.axvspan(data_max, ax.get_xlim()[1], color='0.85', alpha=0.5, lw=0)
        worst = summary[float(r0)]['worst_data']
        ax.set_title(f"$R_0={r0:g}'$\nscatter {worst:.3f}"
                     if np.isfinite(worst) else f"$R_0={r0:g}'$")
        ax.set_xlabel(r'$R$ [arcmin]')
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(r'$r_{bm}\,/\,r_{gb}$   ($\Upsilon$)')

    # Legend below the panels: the per-panel titles carry two lines each, so a
    # top-anchored legend lands on top of them.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(runs), 6),
               frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle('Upsilon reference-radius scan: the Gate A ratio '
                 '(grey band = above the data range)', y=1.02, fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure written to: {out_path}')


def main(path2config, r0_list=None, verbose=True):
    """Run the reference-radius scan for every simulation in a config.

    Args:
        path2config (str): Path to the YAML configuration file.
        r0_list (sequence, optional): Candidate reference radii in arcmin.
            Defaults to None, meaning :data:`DEFAULT_R0`.
        verbose (bool, optional): Print progress.  Defaults to True.
    """
    with open(path2config) as handle:
        config = yaml.safe_load(handle)
    cfg = config['stack']
    plot_cfg = config.get('plot', {})

    radii = aperture_radii(cfg)
    data_max = float(cfg.get('max_radius', 6.0))
    r0_list = sorted(float(v) for v in (r0_list or DEFAULT_R0))

    runs = []
    for block in config['simulations']:
        for entry in block['sims']:
            runs.append(process_simulation(block['sim_type'], entry, cfg,
                                           radii, r0_list, verbose=verbose))

    if not runs:
        raise ValueError(f'No simulations found in {path2config}')

    name = plot_cfg.get('fig_name', 'r_profiles')
    fig_dir = Path(plot_cfg.get('fig_path', '../figures/'))
    summary = report(runs, r0_list, data_max,
                     fig_dir / f'{name}_upsilon_r0_scan.txt')
    make_figure(runs, r0_list, summary, data_max,
                fig_dir / f'{name}_upsilon_r0_scan.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--path2config', '-p', required=True,
                        help='Path to the YAML configuration file.')
    parser.add_argument('--r0', nargs='+', type=float, default=None,
                        help='Candidate Upsilon reference radii, in arcmin.')
    args = parser.parse_args()
    main(args.path2config, r0_list=args.r0)
