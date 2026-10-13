"""compare_sham_density.py

Numbers for the switch of the unbound gas paper's SHAM figures from the
lensing-fit densities to the fixed densities of Liu et al. 2026 (5e-4
(cMpc/h)^-3 at z = 0.5, 1e-3 at z ~ 0.26), read from the files the figure
scripts save. Nothing is stacked or plotted; every number traces to the files
named in the output.

Subcommands:

- ``ratio``: a profiles file of simulated_dsigma_ratio_masked.py (Figs. 6, 9).
  Per simulation: galaxies stacked, mean host R200m (ckpc/h and arcmin at the
  simulation's redshift), the masked ratios f_n and the unmasked one at the
  first and last radius, and the outside fractions 1 - f_n / f_unmasked
  (= 1 - <DSigma_ion^(n)> / <DSigma_ion>, the denominators being the same).
  Optionally against an earlier profiles file (``--old``), the companion's
  published simulated curves (``--companion``, a release CSV with columns
  panel, series, theta_arcmin, fgas, fgas_err) and the measurement of the
  config's ``plot.data_path`` ((f_sim - data) / sigma).
- ``fgas``: a profiles file of make_fgas_profiles.py (Fig. 7). Per simulation:
  f_gas(SHAM) - f_gas(mass cut) at every radius and its largest size within
  6' and over all radii; optionally against an earlier file (``--old``).
- ``pk``: a table of make_pk_fstar_obs.py (Fig. 5). Per simulation, the
  lensing f_gas at the first and last radius and the shift of each f* end, the
  shifts at the first radius in units of the measurement's uncertainty there
  (``--data``, sigma_compensated), and the spread of f_gas between the
  simulations; optionally against an earlier table (``--old``).

Usage (from the scripts/ directory):
    python unbound_gas/compare_sham_density.py ratio -p configs/unbound_gas/dsigma_ratio_z05_masked_nobeam_n5e-4.yaml \\
        --new <..._profiles.npz> --old <lensing-fit ..._profiles.npz> --companion <fig16b.csv> --out <report.txt>
    python unbound_gas/compare_sham_density.py fgas --new <..._2D_profiles.npz> --old <...> --out <report.txt>
    python unbound_gas/compare_sham_density.py pk --new <..._table.txt> --old <...> \\
        --data ../data/beam_compensated/beam_compensated_ionized_gas_total_z0.5.npz --out <report.txt>
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.append('../src/')
from utils import comoving_to_arcmin  # type: ignore
# Sibling module in this directory; after the '../src/' append above.
from simulated_dsigma_ratio_masked import make_stacker, sim_cosmology

# Release-CSV series names of the companion's six runs -> profiles-file keys.
_COMPANION_SERIES = {
    'SIMBA-100': 'SIMBA/m100n1024/s50',
    'TNG300-1': 'IllustrisTNG/TNG300-1',
    'Illustris-1': 'IllustrisTNG/Illustris-1',
    'FLAMINGO L1-m9': 'FLAMINGO/L1_m9/L1_m9',
    'FLAMINGO fgas-8sigma': 'FLAMINGO/L1_m9/fgas-8sigma',
    'FLAMINGO Jet-fgas-4sigma': 'FLAMINGO/L1_m9/Jet_fgas-4sigma',
}


class Report:
    """Collects the printed lines, to write them to a file at the end."""

    def __init__(self) -> None:
        self.lines = []

    def __call__(self, line: str = '') -> None:
        print(line)
        self.lines.append(line)

    def save(self, path: str | None) -> None:
        if path:
            Path(path).write_text('\n'.join(self.lines) + '\n')
            print(f"\nWrote {path}")


def _arr(a, fmt='{:+.3f}'):
    return '[' + ' '.join(fmt.format(x) for x in np.atleast_1d(a)) + ']'


def _runs(npz):
    """Simulation keys of a profiles file, in file order."""
    keys = []
    for k in npz.files:
        if '/' in k and not k.startswith('beam_factor/'):
            run = k.rsplit('/', 1)[0]
            if run not in keys:
                keys.append(run)
    return keys


def _outside(npz, run, n):
    """1 - f_n / f_unmasked = 1 - <num_n> / <num_unmasked> at every radius."""
    return 1.0 - npz[f'{run}/mask{n}_num_mean'] / npz[f'{run}/unmasked_num_mean']


def ratio(args: argparse.Namespace, rep: Report) -> None:
    """Figs. 6 and 9: masked DeltaSigma ratios and outside fractions."""
    with open(args.path2config) as f:
        config = yaml.safe_load(f)
    z_default = config.get('stack', {}).get('redshift', 0.5)
    # Rounded as the profiles keys are (simulated_dsigma_ratio_masked.py: f"mask{r:.0f}").
    mask_radii = [round(float(r)) for r in config.get('stack', {}).get('mask_radii', [1, 2, 3])]
    entries = {}
    for suite in config['simulations']:
        for sim in suite['sims']:
            key = '/'.join(str(k) for k in (suite['sim_type'], sim['name'], sim.get('feedback'))
                           if k is not None)
            entries[key] = (suite['sim_type'], sim, sim.get('redshift', z_default))
    new = np.load(args.new)
    old = np.load(args.old) if args.old else None
    th = new['radii_arcmin']
    rep(f"# ratio: {args.new}")
    rep(f"# config: {args.path2config}")
    if old is not None:
        rep(f"# old:    {args.old}")
        if not np.allclose(old['radii_arcmin'], th):
            raise ValueError("the two profiles files have different radii")
    rep(f"# radii [arcmin]: {_arr(th, '{:.3f}')}")
    rep(f"# outside fraction out_n = 1 - f_n / f_unmasked, at {th[0]:g}' and {th[-1]:g}'")
    rep()
    hdr = (f"{'run':34s} {'n_gal':>8s} {'n':>10s} {'R200m':>7s} {'R200m':>6s}  "
           f"{'f_unm':>12s}  " + '  '.join(f"{'out' + str(n):>11s}" for n in mask_radii))
    rep(hdr)
    rep(f"{'':34s} {'':>8s} {'':>10s} {'ckpc/h':>7s} {'arcmin':>6s}  "
        f"{'1st / last':>12s}  " + '  '.join(f"{'1st / last':>11s}" for _ in mask_radii))
    for run in _runs(new):
        sim_type, sim, z = entries[run]
        stacker = make_stacker(sim_type, sim, z)[0]
        rad = float(new[f'{run}/rad_mean_ckpch'])
        rad_arcmin = comoving_to_arcmin(rad, z, sim_cosmology(stacker))
        f_unm = new[f'{run}/unmasked_ratio']
        outs = [_outside(new, run, n) for n in mask_radii]
        n_obj = int(new[f'{run}/n_objects'])
        rep(f"{run:34s} {n_obj:8d} {float(new[f'{run}/abundance']):10.3e} {rad:7.1f} {rad_arcmin:6.3f}  "
            f"{f_unm[0]:5.3f} / {f_unm[-1]:5.3f}  "
            + '  '.join(f"{o[0]:4.2f} / {o[-1]:4.2f}" for o in outs))
    rep()
    rep("# outside fractions at every radius")
    for run in _runs(new):
        for n in mask_radii:
            rep(f"{run:34s} out{n} {_arr(_outside(new, run, n), '{:.3f}')}")
    for n in mask_radii:
        o = np.array([_outside(new, run, n) for run in _runs(new)])
        rep(f"# range over runs, out{n}: {th[0]:g}' {o[:, 0].min():.2f}-{o[:, 0].max():.2f}; "
            f"{th[-1]:g}' {o[:, -1].min():.2f}-{o[:, -1].max():.2f}")

    if old is not None:
        rep()
        rep("# against --old (runs in both): old n, old out_n at 1st / last radius, new - old")
        for run in _runs(new):
            if f'{run}/mask{mask_radii[0]}_num_mean' not in old:
                rep(f"{run:34s} not in --old")
                continue
            parts = []
            for n in mask_radii:
                o_old, o_new = _outside(old, run, n), _outside(new, run, n)
                parts.append(f"out{n} {o_old[0]:4.2f} / {o_old[-1]:4.2f} "
                             f"(new-old {o_new[0] - o_old[0]:+.2f} / {o_new[-1] - o_old[-1]:+.2f})")
            rep(f"{run:34s} n_old {float(old[f'{run}/abundance']):.3e}  " + '  '.join(parts))
            rep(f"{'':34s} f_unmasked new - old {_arr(new[f'{run}/unmasked_ratio'] - old[f'{run}/unmasked_ratio'])}")

    if args.companion:
        rep()
        rep(f"# unmasked column against the companion's published curves: {args.companion}")
        with open(args.companion) as f:
            rows = list(csv.DictReader(f))
        for series, run in _COMPANION_SERIES.items():
            sel = [r for r in rows if r['series'] == series]
            if not sel or f'{run}/unmasked_ratio' not in new:
                rep(f"{series:26s} missing ({'CSV' if not sel else 'profiles file'})")
                continue
            if len(sel) != len(th):
                rep(f"{series:26s} {len(sel)} rows in the CSV for {len(th)} radii")
                continue
            th_c = np.array([float(r['theta_arcmin']) for r in sel])
            f_c = np.array([float(r['fgas']) for r in sel])
            e_c = np.array([float(r['fgas_err']) for r in sel])
            if not np.allclose(th_c, th):
                rep(f"{series:26s} other radii in the CSV")
                continue
            f_n, e_n = new[f'{run}/unmasked_ratio'], new[f'{run}/unmasked_ratio_err']
            rep(f"{series:26s} max|f - f_pub| = {np.max(np.abs(f_n - f_c)):.2e} "
                f"(relative {np.max(np.abs(f_n / f_c - 1)):.2e}); "
                f"max|err - err_pub| = {np.max(np.abs(e_n - e_c)):.2e}")

    data_path = config.get('plot', {}).get('data_path')
    if data_path and config.get('plot', {}).get('plot_data', False):
        dk = {'theta': 'theta_arcmin', 'value': 'R_compensated', 'error': 'sigma_compensated'}
        dk.update(config['plot'].get('data_keys', {}))
        d = np.load(data_path)
        if np.allclose(d[dk['theta']], th):
            rep()
            rep(f"# unmasked column against the measurement ({data_path}): (f_sim - data) / sigma")
            rep(f"# data {_arr(d[dk['value']], '{:.3f}')} +- {_arr(d[dk['error']], '{:.3f}')}")
            for run in _runs(new):
                pull = (new[f'{run}/unmasked_ratio'] - d[dk['value']]) / d[dk['error']]
                rep(f"{run:34s} {_arr(pull, '{:+.1f}')}")


def fgas(args: argparse.Namespace, rep: Report) -> None:
    """Fig. 7: SHAM minus mass cut, and the new SHAM panel against an older one."""
    new = np.load(args.new)
    old = np.load(args.old) if args.old else None
    r = new['radii']
    in6 = r <= 6.0 + 1e-9
    rep(f"# fgas: {args.new}")
    if old is not None:
        rep(f"# old:  {args.old}")
        if not np.allclose(old['radii'], r):
            raise ValueError("the two profiles files have different radii")
    rep(f"# radii [arcmin]: {_arr(r, '{:.2f}')}")
    rep()
    rep("# d = f_gas(SHAM) - f_gas(mass cut), in units of Omega_b/Omega_m")
    for run in _runs(new):
        d = new[f'{run}/fgas_sham'] - new[f'{run}/fgas_masscut']
        rep(f"{run:40s} n = {float(new[f'{run}/abundance']):.3e}  max|d| <=6': {np.max(np.abs(d[in6])):.3f}  "
            f"all: {np.max(np.abs(d)):.3f}  d = {_arr(d)}")
    if old is not None:
        rep()
        rep("# against --old: mass-cut panel new - old (expected 0), SHAM panel new - old")
        for run in _runs(new):
            if f'{run}/fgas_sham' not in old:
                rep(f"{run:40s} not in --old")
                continue
            dm = new[f'{run}/fgas_masscut'] - old[f'{run}/fgas_masscut']
            ds = new[f'{run}/fgas_sham'] - old[f'{run}/fgas_sham']
            d_old = old[f'{run}/fgas_sham'] - old[f'{run}/fgas_masscut']
            rep(f"{run:40s} n_old = {float(old[f'{run}/abundance']):.3e}  max|mass cut| {np.max(np.abs(dm)):.1e}  "
                f"SHAM max|new-old| <=6': {np.max(np.abs(ds[in6])):.3f} all: {np.max(np.abs(ds)):.3f}  "
                f"new-old = {_arr(ds)}")
            rep(f"{'':40s} old: max|SHAM - mass cut| <=6': {np.max(np.abs(d_old[in6])):.3f} "
                f"all: {np.max(np.abs(d_old)):.3f}")


_PK_ROW = re.compile(r'^\s*(fstar__low|fstar__high)\s+([\d.]+)/([\d.]+)\s+([+-][\d.]+)'
                     r'\s+([\d.]+)/([\d.]+)\s+([+-][\d.]+)\s*$')


def _pk_table(path):
    """{run: {end: (f_sim_1st, f_end_1st, shift_1st, f_sim_last, f_end_last, shift_last)}}
    from the f_gas(theta) block of each simulation of a make_pk_fstar_obs.py table."""
    out, run, in_fgas = {}, None, False
    for line in Path(path).read_text().splitlines():
        if line.startswith('## '):
            run, in_fgas = line[3:].strip(), False
            out[run] = {}
        elif line.startswith('# f_gas(theta)'):
            in_fgas = True
        elif line.startswith('#'):
            in_fgas = False
        elif in_fgas and run is not None:
            m = _PK_ROW.match(line)
            if m:
                out[run][m.group(1)] = tuple(float(x) for x in m.groups()[1:])
    return {k: v for k, v in out.items() if v}


def pk(args: argparse.Namespace, rep: Report) -> None:
    """Fig. 5: shifts of the lensing f_gas from the stellar rescaling."""
    sigma = float(np.load(args.data)['sigma_compensated'][0])
    rep(f"# pk: {args.new}")
    rep(f"# sigma of the innermost measured point: {sigma:.4f} ({args.data})")
    tables = [('new', _pk_table(args.new))]
    if args.old:
        rep(f"# old: {args.old}")
        tables.append(('old', _pk_table(args.old)))
    for tag, tab in tables:
        rep()
        rep(f"# {tag}: f_sim and f* ends at the first / last radius, shift at the first radius in sigma")
        for run, ends in tab.items():
            lo, hi = ends['fstar__low'], ends['fstar__high']
            rep(f"{run:34s} f_sim {lo[0]:.4f} / {lo[3]:.4f}  f*=0.05 {lo[1]:.4f} ({lo[2]:+.4f} = {lo[2] / sigma:+.2f} sigma)  "
                f"f*=0.30 {hi[1]:.4f} ({hi[2]:+.4f} = {hi[2] / sigma:+.2f} sigma)")
        shifts = [(abs(e[2]) / sigma, run, end) for run, ends in tab.items() for end, e in ends.items()]
        f1 = [ends['fstar__low'][0] for ends in tab.values()]
        best = max(shifts)
        rep(f"# {tag}: largest |shift| at the first radius {best[0]:.2f} sigma ({best[1]}, {best[2]}); "
            f"f_sim at the first radius {min(f1):.3f}-{max(f1):.3f} over {len(f1)} runs")
    if args.old:
        rep()
        rep("# f_sim new - old at the first / last radius")
        new, old = tables[0][1], tables[1][1]
        for run in new:
            if run in old:
                rep(f"{run:34s} {new[run]['fstar__low'][0] - old[run]['fstar__low'][0]:+.4f} / "
                    f"{new[run]['fstar__low'][3] - old[run]['fstar__low'][3]:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='what', required=True)
    a = sub.add_parser('ratio', help='Figs. 6 and 9 (simulated_dsigma_ratio_masked.py profiles)')
    a.add_argument('-p', '--path2config', required=True, help="the profiles file's config")
    a.add_argument('--new', required=True)
    a.add_argument('--old', default=None)
    a.add_argument('--companion', default=None, help="release CSV of the companion's curves")
    b = sub.add_parser('fgas', help='Fig. 7 (make_fgas_profiles.py profiles)')
    b.add_argument('--new', required=True)
    b.add_argument('--old', default=None)
    c = sub.add_parser('pk', help='Fig. 5 (make_pk_fstar_obs.py table)')
    c.add_argument('--new', required=True)
    c.add_argument('--old', default=None)
    c.add_argument('--data', required=True, help='beam-compensated measurement npz')
    for p in (a, b, c):
        p.add_argument('--out', default=None, help='also write the output here')
    args = ap.parse_args()
    rep = Report()
    {'ratio': ratio, 'fgas': fgas, 'pk': pk}[args.what](args, rep)
    rep.save(args.out)


if __name__ == '__main__':
    main()
