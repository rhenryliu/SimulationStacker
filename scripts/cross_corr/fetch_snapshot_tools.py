"""fetch_snapshot_tools.py
========================
Manifest and integrity checks for downloading the z ~ 0.75 and z = 1.0
snapshots of the cross_corr runs; driven by ``fetch_snapshots_z075_z10.sh``.

What is fetched (snapshots checked against the servers on 2026-09-28):

    FLAMINGO L1_m9, fgas-8sigma, Jet_fgas-4sigma
        57 (z = 1.00) and 62 (z = 0.75). The two feedback variants have no
        full snapshot at 63 (z = 0.70), so z = 0.75 is the matched slice.
        Whole snapshot directory (virtual file, swift_snapshot_, membership_,
        xray_) plus the SOAP-HBT catalogue.
    FLAMINGO L1_m9_DMO
        57 and 62: virtual file, swift_snapshot_, membership_ (the server has
        no X-ray files for DMO) plus the SOAP-HBT catalogue.
    TNG300-1
        57 (z = 0.7574), a mini snapshot, whole; and 50 (z = 0.9973), a full
        snapshot fetched as a field subset: the mini-snapshot field list
        (InternalEnergyOld is not served) plus NeutralHydrogenAbundance, with
        the tracers. Group catalogues for both.
    TNG300-1-Dark
        57 (mini, the only form it exists in) and 50 (full), whole, with their
        group catalogues.

The on-disk layout copies what is already there. FLAMINGO hydro files go at
the server's relative path under DATA_ROOT. FLAMINGO DMO snapshot files go at
the nested path the earlier tar downloads produced,
``.../L1_m9_DMO/snapshots/flamingo_NNNN/FLAMINGO/L1_m9/L1_m9_DMO/snapshots/flamingo_NNNN/``,
with the SOAP file flat under ``L1_m9_DMO/SOAP-HBT/``. TNG files go to
``output/snapdir_NNN/snap_NNN.i.hdf5`` and
``output/groups_NNN/fof_subhalo_tab_NNN.i.hdf5``.

Manifest lines are tab-separated:
``dataset component url dest size resume auth kind``. ``size`` is the byte
count from the Durham listing, or 0 for TNG (the job asks the server);
``resume`` is 1 only where the server honours range requests (TNG whole files;
not TNG field subsets and not Durham, both checked 2026-09-28).

Subcommands (run from ``scripts/``):

    manifest <out>                 write the full manifest, print totals and
                                   any destination that already exists;
                                   --flamingo-variants / --flamingo-snaps /
                                   --no-flamingo-dmo / --no-tng select another
                                   set (defaults: the set above)
    smoke <manifest> <out>         one file of each kind, for a smoke test
    check <file> <kind>            verify one file
    install <staged> <dest> <kind> verify, then move onto dest; never
                                   overwrites an existing file
    complete <manifest> [dataset ...]
                                   every manifest file present, plus snapshot
                                   and catalogue totals

Kinds: ``gadget`` (particle chunks of TNG and SWIFT: header row counts, last
row of every dataset), ``groupcat`` (TNG group catalogue chunks), ``generic``
(opens, last row of every non-virtual dataset).
"""

import argparse
import glob
import json
import os
import sys
import urllib.request
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'unbound_gas'))
import verify_snapshot_files as vsf  # noqa: E402

DATA_ROOT = os.environ.get('SIMSTACK_DATA_ROOT',
                           '/pscratch/sd/r/rhliu/simulations').rstrip('/')
DURHAM = 'https://dataweb.cosma.dur.ac.uk:8443/hdfstream'
TNG_API = 'https://www.tng-project.org/api'
TNG_API_HEADER = os.environ.get('TNG_API_HEADER',
                                os.path.expanduser('~/.config/tng/api_header'))

FLAMINGO_HYDRO = ['L1_m9', 'fgas-8sigma', 'Jet_fgas-4sigma']
FLAMINGO_SNAPS = [57, 62]

# TNG300-1 snapshot 50 subset: the mini-snapshot field list (as in the local
# snapdir_080) without InternalEnergyOld, which the API rejects, plus
# NeutralHydrogenAbundance. Black holes and tracers are taken whole.
TNG50_SUBSET = (
    'gas=Coordinates,Density,ElectronAbundance,GFM_Metallicity,InternalEnergy,'
    'Masses,NeutralHydrogenAbundance,ParticleIDs,StarFormationRate,Velocities'
    '&dm=Coordinates,ParticleIDs,Velocities'
    '&stars=Coordinates,GFM_InitialMass,GFM_Metallicity,GFM_StellarFormationTime,'
    'Masses,ParticleIDs,Potential,StellarHsml,Velocities'
    '&bhs=all&tracers=all')

# (dataset, simulation, snapshot, field-subset query or None)
TNG_RUNS = [('tng300_57', 'TNG300-1', 57, None),
            ('tng300_50', 'TNG300-1', 50, TNG50_SUBSET),
            ('tngdark_57', 'TNG300-1-Dark', 57, None),
            ('tngdark_50', 'TNG300-1-Dark', 50, None)]

# Components fetched by the smoke test: one file each.
SMOKE = OrderedDict([
    ('tng300_50', ['snapdir', 'groups']),
    ('tng300_57', ['snapdir']),
    ('flam_L1_m9_57', ['virtual', 'swift_snapshot', 'membership', 'xray']),
    ('flamdmo_57', ['virtual', 'membership', 'SOAP-HBT']),
])


# --------------------------------------------------------------------------
# Per-file checks
# --------------------------------------------------------------------------

def check_generic(path: str) -> int:
    """Open an HDF5 file and read the last row of every non-virtual dataset.

    Virtual datasets are skipped: their sources resolve only once the file is
    in its final directory, and ``complete`` checks those sources instead.

    Args:
        path: HDF5 file.

    Returns:
        Number of datasets read.
    """
    count = [0]

    def visit(_name, obj):
        if isinstance(obj, h5py.Dataset) and not obj.is_virtual:
            if obj.shape and obj.shape[0] > 0:
                _ = obj[-1]
            elif not obj.shape:
                _ = obj[()]
            count[0] += 1

    with h5py.File(path, 'r') as f:
        f.visititems(visit)
    return count[0]


def check_groupcat(path: str) -> None:
    """Verify one TNG group-catalogue chunk.

    Args:
        path: ``fof_subhalo_tab_NNN.i.hdf5`` file.

    Raises:
        ValueError: If a Group or Subhalo dataset has the wrong number of rows.
    """
    check_generic(path)
    with h5py.File(path, 'r') as f:
        hdr = f['Header'].attrs
        for group, key in (('Group', 'Ngroups_ThisFile'),
                           ('Subhalo', 'Nsubgroups_ThisFile')):
            n = int(hdr[key])
            if n == 0:
                continue
            for name, ds in f[group].items():
                if ds.shape[0] != n:
                    raise ValueError(f'{path}: {group}/{name} has {ds.shape[0]} '
                                     f'rows, header {key} = {n}')


def check(path: str, kind: str) -> None:
    """Dispatch a single-file check on its kind.

    Args:
        path: File to verify.
        kind: ``gadget``, ``groupcat`` or ``generic``.
    """
    if kind == 'gadget':
        vsf.check_file(path)
    elif kind == 'groupcat':
        check_groupcat(path)
    elif kind == 'generic':
        check_generic(path)
    else:
        raise ValueError(f'unknown kind {kind}')


def install(staged: str, dest: str, kind: str) -> None:
    """Verify a staged download and move it onto its final path.

    Refuses to replace an existing file, so data already on disk is never
    overwritten. ``os.rename`` is atomic on one filesystem.

    Args:
        staged: Completed download in the staging tree.
        dest: Final path, which must not exist yet.
        kind: File kind passed to :func:`check`.
    """
    if os.path.exists(dest):
        sys.exit(f'refusing to overwrite existing {dest}')
    check(staged, kind)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.rename(staged, dest)
    print(f'installed {dest}')


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def _component(dest: str) -> str:
    """Short component label of a destination path (for smoke selection)."""
    parent = os.path.basename(os.path.dirname(dest))
    for prefix in ('snapdir', 'groups', 'swift_snapshot', 'membership', 'xray'):
        if parent.startswith(prefix + '_'):
            return prefix
    if parent == 'SOAP-HBT':
        return 'SOAP-HBT'
    if parent.startswith('flamingo_') and os.path.basename(dest).startswith('flamingo_'):
        return 'virtual'
    return parent


def _walk_remote(directory, prefix=''):
    """Yield ``(relative path, size)`` for every file under a remote directory."""
    import hdfstream
    for name in directory.keys():
        obj = directory[name]
        if isinstance(obj, hdfstream.RemoteDirectory):
            yield from _walk_remote(obj, prefix + name + '/')
        else:
            yield prefix + name, int(obj.size)


def _flamingo_lines(variants: list = FLAMINGO_HYDRO, snaps: list = FLAMINGO_SNAPS,
                    dmo: bool = True) -> list:
    """Manifest lines for every FLAMINGO file (sizes from the server listing).

    Args:
        variants: Hydro variant directory names under ``FLAMINGO/L1_m9/``.
        snaps: Snapshot numbers.
        dmo: Also list ``L1_m9_DMO`` at the same snapshots.
    """
    import hdfstream
    hdfstream.disable_progress(True)
    runs = [(f'flam_{v}_{s}', v, s) for s in snaps for v in variants]
    if dmo:
        runs += [(f'flamdmo_{s}', 'L1_m9_DMO', s) for s in snaps]
    lines = []
    for dataset, variant, snap in runs:
        snap_dir = f'FLAMINGO/L1_m9/{variant}/snapshots/flamingo_{snap:04d}'
        soap = f'FLAMINGO/L1_m9/{variant}/SOAP-HBT/halo_properties_{snap:04d}.hdf5'
        if variant == 'L1_m9_DMO':
            local_dir = f'{DATA_ROOT}/{snap_dir}/{snap_dir}'  # nested, as on disk
        else:
            local_dir = f'{DATA_ROOT}/{snap_dir}'
        remote = hdfstream.open(DURHAM, snap_dir)
        for rel, size in _walk_remote(remote):
            dest = f'{local_dir}/{rel}'
            kind = 'gadget' if '/swift_snapshot_' in dest else 'generic'
            lines.append((dataset, _component(dest), f'{DURHAM}/download/{snap_dir}/{rel}',
                          dest, size, 0, 'none', kind))
        soap_dir = hdfstream.open(DURHAM, os.path.dirname(soap))
        size = int(soap_dir[os.path.basename(soap)].size)
        dest = f'{DATA_ROOT}/{soap}'
        lines.append((dataset, 'SOAP-HBT', f'{DURHAM}/download/{soap}', dest, size,
                      0, 'none', 'generic'))
    return lines


def _tng_headers():
    """HTTP headers carrying the TNG API key, read from the header file."""
    with open(TNG_API_HEADER) as f:
        name, value = f.read().strip().split(':', 1)
    return {name.strip(): value.strip()}


def _tng_json(path: str, headers: dict):
    req = urllib.request.Request(f'{TNG_API}/{path}', headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _tng_lines():
    """Manifest lines for every TNG file; sizes are left to the job (0)."""
    headers = _tng_headers()
    lines = []
    for dataset, sim, snap, subset in TNG_RUNS:
        out = f'{DATA_ROOT}/IllustrisTNG/{sim}/output'
        for kind_api, subdir, stem in (('snapshot', f'snapdir_{snap:03d}', f'snap_{snap:03d}'),
                                       ('groupcat', f'groups_{snap:03d}',
                                        f'fof_subhalo_tab_{snap:03d}')):
            urls = _tng_json(f'{sim}/files/{kind_api}-{snap}/', headers)['files']
            for url in urls:
                url = url.replace('http://', 'https://', 1)
                chunk = url.rsplit('/', 1)[1].split('.')[1]
                dest = f'{out}/{subdir}/{stem}.{chunk}.hdf5'
                if kind_api == 'snapshot':
                    kind = 'gadget'
                    resume = 0 if subset else 1
                    if subset:
                        url = f'{url}?{subset}'
                else:
                    kind, resume = 'groupcat', 1
                lines.append((dataset, _component(dest), url, dest, 0, resume, 'tng', kind))
    return lines


def write_manifest(out: str, variants: list = FLAMINGO_HYDRO, snaps: list = FLAMINGO_SNAPS,
                   dmo: bool = True, tng: bool = True) -> None:
    """Write the full manifest and summarize it.

    The defaults give the z ~ 0.75 / z = 1.0 set this module was written for.

    Args:
        out: Output TSV path.
        variants: FLAMINGO hydro variants to list.
        snaps: FLAMINGO snapshot numbers.
        dmo: Also list FLAMINGO L1_m9_DMO at those snapshots.
        tng: Also list the TNG runs of ``TNG_RUNS``.
    """
    lines = _flamingo_lines(variants, snaps, dmo) + (_tng_lines() if tng else [])
    with open(out, 'w') as f:
        for line in lines:
            f.write('\t'.join(str(x) for x in line) + '\n')

    print(f'wrote {len(lines)} lines to {out}')
    print(f'{"dataset":22s} {"files":>6s} {"known bytes":>16s}')
    per = OrderedDict()
    for line in lines:
        n, b = per.get(line[0], (0, 0))
        per[line[0]] = (n + 1, b + int(line[4]))
    for dataset, (n, b) in per.items():
        print(f'{dataset:22s} {n:6d} {b / 1e12:13.3f} TB')
    headers = _tng_headers() if tng else None
    for dataset, sim, snap, subset in (TNG_RUNS if tng else []):
        meta = _tng_json(f'{sim}/snapshots/{snap}/', headers)
        tag = 'subset of' if subset else 'whole'
        print(f'{dataset:22s} API: snapshot {meta["filesize_snapshot"] / 1e12:.3f} TB ({tag}), '
              f'groupcat {meta["filesize_groupcat"] / 1e9:.1f} GB, z = {meta["redshift"]:.4f}')

    existing = [line[3] for line in lines if os.path.exists(line[3])]
    if existing:
        print(f'WARNING: {len(existing)} destination(s) already exist, e.g. {existing[:3]}')
    else:
        print('no destination exists yet')


def write_smoke(manifest: str, out: str) -> None:
    """Pick one manifest line per smoke component.

    Args:
        manifest: Full manifest TSV.
        out: Smoke manifest TSV.
    """
    picked = OrderedDict()
    with open(manifest) as f:
        for raw in f:
            cols = raw.rstrip('\n').split('\t')
            key = (cols[0], cols[1])
            if cols[0] in SMOKE and cols[1] in SMOKE[cols[0]] and key not in picked:
                picked[key] = raw
    with open(out, 'w') as f:
        f.writelines(picked.values())
    for (dataset, comp), raw in picked.items():
        print(f'{dataset:16s} {comp:15s} {raw.split(chr(9))[3]}')


# --------------------------------------------------------------------------
# Completeness
# --------------------------------------------------------------------------

def _complete_tng(snapdir: str, groupsdir: str) -> list:
    """Snapshot and group-catalogue totals for one TNG snapshot."""
    problems = []
    snap = os.path.basename(snapdir).split('_')[1]
    try:
        vsf.check_snapshot(f'{snapdir}/snap_{snap}.*.hdf5')
    except SystemExit as e:
        problems.append(f'{snapdir}: {e}')
    files = sorted(glob.glob(f'{groupsdir}/fof_subhalo_tab_{snap}.*.hdf5'))
    n_groups = n_subs = 0
    for path in files:
        with h5py.File(path, 'r') as f:
            hdr = f['Header'].attrs
            n_groups += int(hdr['Ngroups_ThisFile'])
            n_subs += int(hdr['Nsubgroups_ThisFile'])
            totals = (int(hdr['NumFiles']), int(hdr['Ngroups_Total']),
                      int(hdr['Nsubgroups_Total']), float(hdr['Redshift']))
    if not files:
        return problems + [f'{groupsdir}: no group catalogue files']
    print(f'{groupsdir}: {len(files)} files (expects {totals[0]}), groups {n_groups} '
          f'({totals[1]}), subhalos {n_subs} ({totals[2]}), z = {totals[3]:.4f}')
    if (len(files), n_groups, n_subs) != totals[:3]:
        problems.append(f'{groupsdir}: group catalogue incomplete')
    return problems


def _complete_flamingo(snapdir: str) -> list:
    """Particle totals, companion row counts and virtual sources for one snapshot.

    Args:
        snapdir: Directory holding ``flamingo_NNNN.hdf5``.
    """
    problems = []
    snap = os.path.basename(snapdir).split('_')[1]
    virtual = f'{snapdir}/flamingo_{snap}.hdf5'
    try:
        vsf.check_snapshot(f'{snapdir}/swift_snapshot_{snap}/flamingo_{snap}.*.hdf5')
    except SystemExit as e:
        problems.append(f'{snapdir}: {e}')

    # Every file the virtual file maps must exist (relative to its directory).
    sources = set()

    def visit(_name, obj):
        if isinstance(obj, h5py.Dataset) and obj.is_virtual:
            sources.update(m.file_name for m in obj.virtual_sources())

    with h5py.File(virtual, 'r') as f:
        f.visititems(visit)
        z = float(np.asarray(f['Header'].attrs['Redshift']).ravel()[0])
    missing = sorted(s for s in sources if not os.path.exists(os.path.join(snapdir, s)))
    print(f'{virtual}: z = {z:.4f}, {len(sources)} source files, {len(missing)} missing')
    if missing:
        problems.append(f'{virtual}: {len(missing)} missing sources, e.g. {missing[:3]}')

    # Membership and X-ray chunks must match their particle chunk row by row.
    for chunk in sorted(glob.glob(f'{snapdir}/swift_snapshot_{snap}/flamingo_{snap}.*.hdf5')):
        i = chunk.rsplit('.', 2)[1]
        with h5py.File(chunk, 'r') as f:
            npart = np.asarray(f['Header'].attrs['NumPart_ThisFile'], dtype=np.int64)
        for comp in ('membership', 'xray'):
            path = f'{snapdir}/{comp}_{snap}/{comp}_{snap}.{i}.hdf5'
            if not os.path.exists(path):
                continue  # absent companions are caught by the source check
            with h5py.File(path, 'r') as f:
                for group in (k for k in f.keys() if k.startswith('PartType')):
                    n = npart[int(group[len('PartType'):])]
                    for name, ds in f[group].items():
                        if ds.shape[0] != n:
                            problems.append(f'{path}: {group}/{name} has {ds.shape[0]} '
                                            f'rows, particle chunk has {n}')
    return problems


def complete(manifest: str, datasets: list) -> None:
    """Check that the listed datasets were fetched completely.

    Args:
        manifest: Full manifest TSV.
        datasets: Dataset names to check; all in the manifest if empty.

    Raises:
        SystemExit: Non-zero if anything is missing or inconsistent.
    """
    with open(manifest) as f:
        all_rows = [raw.rstrip('\n').split('\t') for raw in f]
    unknown = sorted(set(datasets) - {r[0] for r in all_rows})
    if unknown:
        sys.exit(f'datasets not in {manifest}: {unknown}')
    rows = [r for r in all_rows if not datasets or r[0] in datasets]
    problems = []
    for dataset in OrderedDict.fromkeys(r[0] for r in rows):
        dests = [r[3] for r in rows if r[0] == dataset]
        absent = [d for d in dests if not os.path.exists(d)]
        print(f'\n== {dataset}: {len(dests) - len(absent)} of {len(dests)} files present')
        if absent:
            problems.append(f'{dataset}: {len(absent)} files absent, e.g. {absent[:2]}')
            continue
        if dataset.startswith('tng'):
            snapdir = next(os.path.dirname(d) for d in dests if '/snapdir_' in d)
            groupsdir = next(os.path.dirname(d) for d in dests if '/groups_' in d)
            problems += _complete_tng(snapdir, groupsdir)
        else:
            virtual = next(r[3] for r in rows if r[0] == dataset and r[1] == 'virtual')
            problems += _complete_flamingo(os.path.dirname(virtual))
    if problems:
        sys.exit('INCOMPLETE:\n  ' + '\n  '.join(problems))
    print('\nALL COMPLETE')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('manifest')
    p.add_argument('out')
    p.add_argument('--flamingo-variants', nargs='+', default=FLAMINGO_HYDRO)
    p.add_argument('--flamingo-snaps', nargs='+', type=int, default=FLAMINGO_SNAPS)
    p.add_argument('--no-flamingo-dmo', action='store_true')
    p.add_argument('--no-tng', action='store_true')
    p = sub.add_parser('smoke')
    p.add_argument('manifest')
    p.add_argument('out')
    p = sub.add_parser('check')
    p.add_argument('file')
    p.add_argument('kind')
    p = sub.add_parser('install')
    p.add_argument('staged')
    p.add_argument('dest')
    p.add_argument('kind')
    p = sub.add_parser('complete')
    p.add_argument('manifest')
    p.add_argument('datasets', nargs='*')
    args = parser.parse_args()

    if args.cmd == 'manifest':
        write_manifest(args.out, args.flamingo_variants, args.flamingo_snaps,
                       dmo=not args.no_flamingo_dmo, tng=not args.no_tng)
    elif args.cmd == 'smoke':
        write_smoke(args.manifest, args.out)
    elif args.cmd == 'check':
        check(args.file, args.kind)
        print(f'{args.file}: OK ({args.kind})')
    elif args.cmd == 'install':
        install(args.staged, args.dest, args.kind)
    else:
        complete(args.manifest, args.datasets)


if __name__ == '__main__':
    main()
