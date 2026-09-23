"""build_dmo_field.py
===================
Build the 3D DM density field of each dark-matter-only (DMO) reference run in
``configs/unbound_gas/pk_components_z05.yaml``, on the same grid as its hydro
counterpart, for the P_hydro / P_DMO suppression in the power spectrum section.

``SimulationStacker`` cannot open these runs (its feedback-variant assertions
reject 'dm' and 'L1_m9_DMO', and the FLAMINGO DMO download is nested one level
deeper than the hydro runs), so this script hands ``mapMaker.make_mass_field``
a minimal stand-in exposing only what that function reads. The binning code,
TSC kernel, unit conversions (``loadIO.load_subset``) and float32 grid are
therefore exactly those of the cached hydro fields.

Output (new files; an existing output is never overwritten):
    <root>/<SimType>/products/3D/<DMO filename per loadIO convention>, e.g.
    FLAMINGO/products/3D/L1_m9_L1_m9_DMO_67_DM_2000.npy
    IllustrisTNG/products/3D/TNG300-1-Dark_67_DM_1000.npy

As a completeness check the summed field must equal Omega_m * rho_crit * V to
within 1e-3 (every run here has at most 128 roughly equal chunk files, so a
missing chunk would show as a deficit of >= 1/128); otherwise nothing is
saved. The DMO box size must also match the hydro run's, since both fields are
later transformed with the hydro box size.

Run from the scripts/ directory on a whole CPU node:
    python unbound_gas/build_dmo_field.py -p configs/unbound_gas/pk_components_z05.yaml \
        --sims 'L1_m9 (L1_m9)'
"""

import argparse
import sys
import time

import h5py
import numpy as np

from pk_common import (RHO_CRIT_H, dmo_entry, field_path, header_of, load_config,
                       save_npy_atomic, select_sims, sim_label)

sys.path.append('../src/')
from loadIO import load_flamingo_header, resolve_data_root  # noqa: E402
from mapMaker import make_mass_field  # noqa: E402
from verify_snapshot_files import check_snapshot  # noqa: E402


class _DMOSnapshot:
    """Stand-in for SimulationStacker with the attributes make_mass_field reads.

    Attributes mirror SimulationStacker: ``simType``, ``sim``, ``snapshot``,
    ``simPath``, ``header`` (TNG-style keys, BoxSize in ckpc/h) and ``z``.
    """

    def __init__(self, sim_type: str, name: str, snapshot: int, chunk_glob_root: str,
                 header: dict):
        self.simType = sim_type
        self.sim = name
        self.snapshot = snapshot
        self.simPath = chunk_glob_root
        self.header = header
        self.z = float(header['Redshift'])
        self._root = chunk_glob_root

    def snapPath(self, chunkNum=0, pathOnly=False):  # noqa: N802 (mirrors stacker API)
        """Return the location make_mass_field globs chunk files from.

        make_mass_field globs ``<this> + 'snap_*.hdf5'`` for TNG,
        ``<this>`` itself for SIMBA (a single file) and
        ``<this> + 'flamingo_*.hdf5'`` for FLAMINGO.
        """
        return self._root


def make_dmo_snapshot(d: dict) -> _DMOSnapshot:
    """Build the stand-in for one DMO run from its config entry.

    Args:
        d: DMO entry from ``pk_common.dmo_entry`` (sim_type, name, snapshot,
            feedback, n_pixels, path relative to the data root).
    """
    root = resolve_data_root(None)
    path = root + d['path']
    if d['sim_type'] == 'IllustrisTNG':
        with h5py.File(f"{path}snap_{d['snapshot']:03d}.0.hdf5", 'r') as f:
            header = dict(f['Header'].attrs.items())
        glob_root = path
    elif d['sim_type'] == 'SIMBA':
        with h5py.File(path, 'r') as f:
            header = dict(f['Header'].attrs.items())
        glob_root = path
    elif d['sim_type'] == 'FLAMINGO':
        header = load_flamingo_header(f"{path}flamingo_{d['snapshot']:04d}.hdf5")
        glob_root = f"{path}swift_snapshot_{d['snapshot']:04d}/"
    else:
        raise ValueError(f"unknown sim_type {d['sim_type']}")
    return _DMOSnapshot(d['sim_type'], d['name'], d['snapshot'], glob_root, header)


def snapshot_complete(d: dict) -> bool:
    """True if every file of a Gadget-format DMO snapshot is on disk and intact.

    Uses verify_snapshot_files.check_snapshot (file count and particle totals
    against the header). FLAMINGO's SWIFT chunks are not checked here; the
    mass check in build() catches a missing chunk there.
    """
    if d['sim_type'] == 'FLAMINGO':
        return True
    path = resolve_data_root(None) + d['path']
    pattern = f"{path}snap_{d['snapshot']:03d}.*.hdf5" if d['sim_type'] == 'IllustrisTNG' else path
    try:
        check_snapshot(pattern)
    except SystemExit as e:
        print(f"snapshot not complete yet, skipping: {e}")
        return False
    return True


def build(d: dict, hydro_box: float) -> None:
    """Build, check and save the DM field of one DMO run (skips if present).

    Args:
        d: DMO entry from ``pk_common.dmo_entry``.
        hydro_box: BoxSize (ckpc/h) of the hydro counterpart; the DMO header
            must agree to 1e-6, because compute_pk_components.py transforms
            both fields with the hydro box size.
    """
    out = field_path(d['sim_type'], d['name'], d['snapshot'], d['feedback'], 'DM',
                     d['n_pixels'])
    if out.exists():
        print(f"already present, skipping: {out}")
        return
    if not snapshot_complete(d):
        return

    snap = make_dmo_snapshot(d)
    hdr = snap.header
    if abs(float(hdr['BoxSize']) / hydro_box - 1.0) > 1e-6:
        sys.exit(f"DMO BoxSize {hdr['BoxSize']} != hydro BoxSize {hydro_box}; not building {out}")
    print(f"DMO {d['name']} {d.get('feedback') or ''} snapshot {d['snapshot']}: "
          f"z={snap.z:.4f}, BoxSize={hdr['BoxSize']:.1f} ckpc/h, "
          f"Omega0={float(hdr['Omega0']):.4f}, grid {d['n_pixels']}^3")

    t0 = time.time()
    field = make_mass_field(snap, 'DM', nPixels=d['n_pixels'], projection='xy', dim='3D')
    print(f"binned in {time.time() - t0:.0f} s; dtype {field.dtype}")

    volume = (hdr['BoxSize'] / 1000.0) ** 3  # (Mpc/h)^3
    expected = float(hdr['Omega0']) * RHO_CRIT_H * volume  # M_sun/h
    total = float(np.sum(field, dtype=np.float64))
    ratio = total / expected
    print(f"mass check: sum = {total:.6e} M_sun/h, Omega_m rho_crit V = {expected:.6e}, "
          f"ratio = {ratio:.6f}")
    if abs(ratio - 1.0) > 1e-3:
        sys.exit(f"mass check failed (ratio {ratio:.6f}); not saving {out}")

    save_npy_atomic(out, field)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-p', '--path2config', required=True)
    parser.add_argument('--sims', nargs='*', default=None,
                        help="restrict to these hydro entries (label, name or feedback)")
    args = parser.parse_args()

    config = load_config(args.path2config)
    seen = set()
    for entry in select_sims(config, args.sims):
        d = dmo_entry(entry)
        key = (d['sim_type'], d['name'], d['feedback'], d['snapshot'], d['n_pixels'])
        if key in seen:
            continue  # the FLAMINGO variants share one DMO run
        seen.add(key)
        print(f"\n===== DMO reference for {sim_label(entry)} =====")
        build(d, float(header_of(entry)['BoxSize']))


if __name__ == '__main__':
    main()
