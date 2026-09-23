"""verify_snapshot_files.py
==========================
Integrity checks for downloaded Gadget-format HDF5 snapshot files (IllustrisTNG,
Illustris, SIMBA), used by ``fetch_dmo_snapshots.sh`` for the dark-matter-only
(DMO) reference runs of the power-spectrum section.

Three subcommands:

    check    <file>                 verify one file, exit non-zero if broken
    install  <staged> <dest>        verify a staged download, then atomically
                                    move it onto <dest> (replacing any file there)
    snapshot <glob>                 verify that the files matching <glob> form a
                                    complete snapshot: file count equals the
                                    header's NumFilesPerSnapshot and the summed
                                    NumPart_ThisFile equals NumPart_Total

A file passes ``check`` if it opens, every particle group announced in
``NumPart_ThisFile`` exists, every dataset in it has that many rows, and the
last row of every dataset can be read. Truncated downloads fail at the open
(HDF5 compares its stored end-of-file address with the real file size) or at
the last-row read.

Run from the scripts/ directory, e.g.:
    python unbound_gas/verify_snapshot_files.py snapshot \
        '/pscratch/sd/r/rhliu/simulations/IllustrisTNG/TNG300-1-Dark/output/snapdir_067/snap_067.*.hdf5'
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np


def check_file(path: str) -> np.ndarray:
    """Verify that one snapshot file is complete and readable.

    Args:
        path: Path to the HDF5 snapshot file.

    Returns:
        The file's ``NumPart_ThisFile`` array (int64, one entry per particle type).

    Raises:
        OSError: If HDF5 cannot open the file (e.g. it is truncated).
        ValueError: If a particle group or dataset is missing or has the wrong
            number of rows.
    """
    with h5py.File(path, 'r') as f:
        npart = np.asarray(f['Header'].attrs['NumPart_ThisFile'], dtype=np.int64)
        for ptype, n in enumerate(npart):
            if n == 0:
                continue
            group = f'PartType{ptype}'
            if group not in f:
                raise ValueError(f"{path}: header lists {n} particles of type "
                                 f"{ptype} but group {group} is missing")
            for key, ds in f[group].items():
                if not isinstance(ds, h5py.Dataset):
                    continue
                if ds.shape[0] != n:
                    raise ValueError(f"{path}: {group}/{key} has {ds.shape[0]} "
                                     f"rows, header says {n}")
                _ = ds[-1]  # forces a read of the final chunk
    return npart


def install(staged: str, dest: str) -> None:
    """Verify a staged download and move it onto its final path.

    ``os.replace`` is atomic on a single filesystem, so ``dest`` is never left
    half-written: it holds either the old file or the verified new one.

    Args:
        staged: Path of the completed download in the staging directory.
        dest: Final path; an existing file there is replaced.
    """
    npart = check_file(staged)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.replace(staged, dest)
    print(f"installed {dest} (NumPart_ThisFile={npart.tolist()})")


def check_snapshot(pattern: str) -> None:
    """Verify that the files matching ``pattern`` form a complete snapshot.

    Args:
        pattern: Glob matching every file of one snapshot.

    Raises:
        SystemExit: With a non-zero code if any file is broken, a file is
            missing, or the particle totals disagree with the header.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no files match {pattern}")

    summed = None
    bad = []
    for path in files:
        try:
            npart = check_file(path)
        except (OSError, ValueError, KeyError) as e:
            bad.append(f"{os.path.basename(path)}: {e}")
            continue
        summed = npart if summed is None else summed + npart
    if summed is None:
        sys.exit(f"INCOMPLETE: none of the {len(files)} files matching {pattern} is readable")

    # Header totals from any readable file (every chunk carries the same ones).
    good = next(p for p in files if os.path.basename(p) not in
                {b.split(':')[0] for b in bad})
    with h5py.File(good, 'r') as f:
        hdr = f['Header'].attrs
        n_files = int(hdr['NumFilesPerSnapshot'])
        total = np.asarray(hdr['NumPart_Total'], dtype=np.int64)
        if 'NumPart_Total_HighWord' in hdr:
            total = total + (np.asarray(hdr['NumPart_Total_HighWord'],
                                        dtype=np.int64) << 32)

    print(f"{pattern}: {len(files)} files (header expects {n_files}); "
          f"particles {summed.tolist()} (header total {total.tolist()})")
    problems = []
    if bad:
        problems.append(f"{len(bad)} unreadable file(s): " + "; ".join(bad[:5]))
    if len(files) != n_files:
        problems.append(f"file count {len(files)} != NumFilesPerSnapshot {n_files}")
    if not np.array_equal(summed, total):
        problems.append("summed NumPart_ThisFile != NumPart_Total")
    if problems:
        sys.exit("INCOMPLETE: " + " | ".join(problems))
    print("COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('check')
    p.add_argument('file')
    p = sub.add_parser('install')
    p.add_argument('staged')
    p.add_argument('dest')
    p = sub.add_parser('snapshot')
    p.add_argument('pattern')
    args = parser.parse_args()

    if args.cmd == 'check':
        print(f"{args.file}: OK, NumPart_ThisFile={check_file(args.file).tolist()}")
    elif args.cmd == 'install':
        install(args.staged, args.dest)
    else:
        check_snapshot(args.pattern)


if __name__ == '__main__':
    main()
