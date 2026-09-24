"""pk_dmo_checks.py
================
Helpers for lensing/runCPU_pk_dmo_z026.sh, the job that makes the hydro/DMO
spectra (S(k) = P_total / P_DMO) behind the dotted z ~ 0.26 curves of
lensing/plot_pk_suppression.py. Reads a config in the schema of
configs/unbound_gas/pk_components_z05.yaml (e.g. configs/lensing/pk_dmo_z026.yaml).

Subcommands:

    tasks   print "sim_type name snapshot feedback n_pixels redshift" for each
            entry whose 3D `total` cache is missing (feedback '-' if none);
            the runner builds these with precompute_baryonFraction_fields.py
    mass    check each entry's `total` cache: its sum must equal
            Omega_m * rho_crit * V to 1e-3, which catches a hydro snapshot
            with a missing or unreadable chunk (build_dmo_field.py does the
            same check for the DMO fields)
    report  print S(k) and the hydro/DMO cross-correlation r(k) from each
            entry's *_Pk_dmo_<n>.npz; exit non-zero if any file is missing

Run from the scripts/ directory:
    python lensing/pk_dmo_checks.py -p configs/lensing/pk_dmo_z026.yaml report
"""

import argparse
import sys

import numpy as np

sys.path.append('unbound_gas/')
from pk_common import (RHO_CRIT_H, field_path, header_of, load_config,  # noqa: E402
                       select_sims, sim_label, spectra_path)

# k values [h/Mpc] at which `report` prints S(k) (as plot_pk_suppression.py).
K_PRINT = [1.0, 2.0, 5.0, 10.0]

# Tolerance of the mass check (as build_dmo_field.py).
MASS_TOL = 1e-3


def total_path(entry: dict):
    """Path of an entry's cached 3D `total` field."""
    return field_path(entry['sim_type'], entry['name'], entry['snapshot'],
                      entry.get('feedback'), 'total', entry['n_pixels'])


def tasks(entries: list) -> int:
    """Print one build task per entry whose `total` cache is missing."""
    for e in entries:
        if total_path(e).exists():
            print(f"# {sim_label(e)}: total cache present, nothing to build", file=sys.stderr)
            continue
        print(e['sim_type'], e['name'], e['snapshot'], e.get('feedback') or '-',
              e['n_pixels'], e.get('redshift', 0.5))
    return 0


def mass(entries: list) -> int:
    """Check sum(total) against Omega_m rho_crit V; return 1 on any failure."""
    failed = 0
    for e in entries:
        path = total_path(e)
        if not path.exists():
            print(f"FAIL {sim_label(e)}: missing {path}")
            failed += 1
            continue
        hdr = header_of(e)
        box_mpc = float(hdr['BoxSize']) / 1000.0
        expected = float(hdr['Omega0']) * RHO_CRIT_H * box_mpc ** 3  # M_sun/h
        total = float(np.sum(np.load(path, mmap_mode='r'), dtype=np.float64))
        ratio = total / expected
        ok = abs(ratio - 1.0) < MASS_TOL
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {sim_label(e)} snapshot {e['snapshot']}: "
              f"sum(total) / (Omega_m rho_crit V) = {ratio:.6f} "
              f"(Omega_m = {float(hdr['Omega0']):.4f}, box {box_mpc:.1f} Mpc/h)")
    return int(failed > 0)


def report(entries: list) -> int:
    """Print S(k) and r(k) for each entry; return 1 if any spectra file is missing."""
    missing = 0
    print(f"{'simulation':28s} {'snap':>4s} {'S(k_min)':>9s} "
          + ' '.join(f"{f'S(k={kp:g})':>9s}" for kp in K_PRINT) + "  r(k) lowest 3 k")
    for e in entries:
        path = spectra_path(e, 'dmo')
        if not path.exists():
            print(f"{sim_label(e):28s} {e['snapshot']:>4d}  MISSING {path}")
            missing += 1
            continue
        d = np.load(path)
        k, S = d['k'], d['P_total'] / d['P_dmo']
        r = d['P_total_dmo'] / np.sqrt(d['P_total'] * d['P_dmo'])
        vals = [f"{S[np.argmin(np.abs(np.log(k / kp)))]:9.4f}" if kp <= k.max()
                else f"{'--':>9s}" for kp in K_PRINT]
        print(f"{sim_label(e):28s} {e['snapshot']:>4d} {S[0]:9.4f} " + ' '.join(vals)
              + f"  {np.round(r[:3], 5).tolist()}")
    return int(missing > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None,
                        help="restrict to these entries (label, name or feedback)")
    parser.add_argument('command', choices=['tasks', 'mass', 'report'])
    args = parser.parse_args()

    entries = select_sims(load_config(args.path2config), args.sims)
    if args.sims and not entries:
        # A mistyped --sims would otherwise make every stage a silent no-op.
        sys.exit(f"--sims {args.sims} matched no entry of {args.path2config}")
    sys.exit({'tasks': tasks, 'mass': mass, 'report': report}[args.command](entries))


if __name__ == '__main__':
    main()
