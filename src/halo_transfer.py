"""halo_transfer.py
=================
Halo-level transfer of stellar mass into ionized gas, for sensitivity tests of
the matter power spectrum to the stellar baryon fraction.

For a halo region h with star particles S_h and gas particles G_h,

    M*_h = sum_{j in S_h} m*_j ,   M_ion,h = sum_{i in G_h} m_ion,i ,

a stellar scale s keeps s m*_j of every star and adds
(1 - s) M*_h m_ion,i / M_ion,h to each gas particle: the removed stellar mass
follows the halo's existing ionized-gas distribution, and mass is conserved
halo by halo. Nothing outside the selected haloes changes. m_ion is the
pipeline's ionized-gas mass (``mapMaker.ionized_gas_masses``).

Because the change is linear in (1 - s), the matter field is
rho_m(s) = rho_m + (1 - s) D with the zero-mass transfer field

    D = TSC[ sum_h (M*_h / M_ion,h) m_ion,i ] - TSC[ sum_h m*_j ] ,

deposited with the same TSC as the cached fields, and

    P_mm(s) = P_mm + 2 (1 - s) P_mD + (1 - s)^2 P_DD      (``p_of_scale``)

exactly, with delta_D = D / rhobar_m. One D per configuration gives every s.
Haloes whose region has stars but no ionized gas keep their stars (reported).

The two methods differ only in how particles are assigned to haloes (labels
are rows of ``SimulationStacker.loadHalos()``; -1 = no halo):

- ``membership``: the halo finder's membership (``MembershipLabeller``): FoF
  groups (IllustrisTNG/Illustris), CAESAR haloes (SIMBA), FoF groups mapped
  to their SOAP-HBT central (FLAMINGO). Satellites belong to their host.
- ``aperture``: particles within x * R200m (``GroupRad``) of ``GroupPos``
  (``aperture_labels``); where apertures overlap, the particle belongs to the
  most massive halo (``GroupMass``) that contains it (mass priority, periodic
  minimum-image distances). Every particle is modified at most once, and the
  labels are nested in the halo mass cut (``restrict_labels``).

``transfer_maps_2d`` is the projected counterpart for stacked profiles: 2D
maps of the ionized-gas mass added and the stellar mass removed at s = 0,
binned like the cached 2D fields.

The stellar-fraction floor (``halo_baryons``, ``floor_coefficients``,
``scaled_transfer_field``, ``scaled_transfer_maps_2d``) is the opposite
direction with a per-halo amount: each region whose star fraction (true
stars over all baryons) is below a target gains stars, laid out like its
stars and taken from its ionized gas, up to the target.
"""

import glob
import os
import time
import warnings

import h5py
import numba
import numpy as np
from scipy.stats import binned_statistic_2d

from loadIO import (load_caesar_particle_halos, load_flamingo_central_fof_ids,
                    load_flamingo_fof_ids, load_fof_group_ends, load_subset)
from mapMaker import ionized_gas_keys, ionized_gas_masses

try:
    from abacusnbody.analysis.tsc import tsc_parallel  # type: ignore
except ImportError:
    tsc_parallel = None

NO_HALO = -1
FLAMINGO_NO_GROUP = 2147483647  # FOFGroupIDs of particles in no group
_PTYPE_INDEX = {'gas': 0, 'Stars': 4, 'BH': 5}


# ---------------------------------------------------------------------------
# Labels: halo-finder membership
# ---------------------------------------------------------------------------

def labels_from_group_ends(global_index, group_ends):
    """Group of each particle in a FoF-ordered snapshot.

    Args:
        global_index (np.ndarray): Global particle indices (int64).
        group_ends (np.ndarray): Cumulative particle count per group
            (``loadIO.load_fof_group_ends``).

    Returns:
        np.ndarray: int32 group index, NO_HALO beyond the last group.
    """
    idx = np.searchsorted(group_ends, global_index, side='right')
    return np.where(idx < len(group_ends), idx, NO_HALO).astype(np.int32)


def make_fof_lookup(central_fof_ids):
    """Sorted lookup from FoF group ID to halo row (see ``labels_from_fof_ids``).

    Args:
        central_fof_ids (np.ndarray): FoF ID of each halo row (negative = none).

    Returns:
        tuple: (sorted IDs, their rows).
    """
    rows = np.flatnonzero(central_fof_ids >= 0)
    order = np.argsort(central_fof_ids[rows], kind='stable')
    sorted_ids = central_fof_ids[rows][order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError("a FoF group has more than one central")
    return sorted_ids, rows[order]


def labels_from_fof_ids(fof_ids, lookup):
    """Halo row of each particle from its FoF group ID.

    Args:
        fof_ids (np.ndarray): FoF group ID per particle.
        lookup (tuple): From ``make_fof_lookup``.

    Returns:
        np.ndarray: int32 halo row; NO_HALO for particles in no group or in a
        group without a catalogued central.
    """
    sorted_ids, rows = lookup
    if len(sorted_ids) == 0:
        return np.full(len(fof_ids), NO_HALO, dtype=np.int32)
    pos = np.clip(np.searchsorted(sorted_ids, fof_ids), 0, len(sorted_ids) - 1)
    match = (sorted_ids[pos] == fof_ids) & (fof_ids >= 0) & (fof_ids != FLAMINGO_NO_GROUP)
    return np.where(match, rows[pos], NO_HALO).astype(np.int32)


class MembershipLabeller:
    """Halo-finder membership of particles, as rows of ``stacker.loadHalos()``.

    Args:
        stacker (SimulationStacker): The simulation.
    """

    def __init__(self, stacker):
        self.st = stacker
        self._cache = {}

    def _setup(self, p_type):
        if p_type in self._cache:
            return self._cache[p_type]
        st = self.st
        if st.simType == 'IllustrisTNG':
            val = load_fof_group_ends(st.simPath, st.snapshot, p_type)
        elif st.simType == 'SIMBA':
            n = int(np.asarray(st.header['NumPart_Total'])[_PTYPE_INDEX[p_type]])
            val = load_caesar_particle_halos(st.simPath, st.snapshot, st.sim, p_type, n)
        elif st.simType == 'FLAMINGO':
            if 'lookup' not in self._cache:
                self._cache['lookup'] = make_fof_lookup(
                    load_flamingo_central_fof_ids(st.simPath, st.snapshot))
            val = self._cache['lookup']
        else:
            raise NotImplementedError(st.simType)
        self._cache[p_type] = val
        return val

    def labels(self, p_type, chunk, global_start, n):
        """Membership labels of one chunk's particles of a type.

        Args:
            p_type (str): 'gas', 'Stars' or 'BH'.
            chunk (int): Chunk index (FLAMINGO membership file).
            global_start (int): Global index of the chunk's first particle of
                the type (TNG/Illustris ordering; SIMBA has one chunk).
            n (int): Number of particles of the type in the chunk.

        Returns:
            np.ndarray: int32 halo row per particle, NO_HALO if none.
        """
        val = self._setup(p_type)
        if self.st.simType == 'IllustrisTNG':
            return labels_from_group_ends(global_start + np.arange(n, dtype=np.int64), val)
        if self.st.simType == 'SIMBA':
            return val[global_start:global_start + n]
        ids = load_flamingo_fof_ids(self.st.simPath, self.st.snapshot, p_type, chunk)
        if len(ids) != n:
            raise RuntimeError(f"FLAMINGO membership chunk {chunk} ({p_type}) has {len(ids)} "
                               f"particles, the snapshot chunk {n}")
        return labels_from_fof_ids(ids, val)


# ---------------------------------------------------------------------------
# Labels: radial apertures (mass priority, periodic)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _axis_cells(p, r, cell, nh):
    """Hash-cell range [lo, hi] covered by [p - r, p + r] on one axis (full if >= nh)."""
    lo = int(np.floor((p - r) / cell))
    hi = int(np.floor((p + r) / cell))
    if hi - lo + 1 >= nh:
        return 0, nh - 1
    return lo, hi


@numba.njit(cache=True)
def _build_hash(hp, hr, cell, nh):
    """CSR lists of the haloes whose sphere bounding box overlaps each hash cell.

    Haloes are inserted in array order, so each cell's list keeps that order
    (the caller passes haloes sorted by decreasing priority).
    """
    ncell = nh * nh * nh
    counts = np.zeros(ncell + 1, dtype=np.int64)
    for h in range(hp.shape[0]):
        r = hr[h] + 1e-6 * cell  # pad: the cell ranges must be a superset of the sphere
        lx, hx = _axis_cells(hp[h, 0], r, cell, nh)
        ly, hy = _axis_cells(hp[h, 1], r, cell, nh)
        lz, hz = _axis_cells(hp[h, 2], r, cell, nh)
        for i in range(lx, hx + 1):
            for j in range(ly, hy + 1):
                for k in range(lz, hz + 1):
                    counts[((i % nh) * nh + (j % nh)) * nh + (k % nh) + 1] += 1
    offsets = np.cumsum(counts)
    members = np.empty(offsets[-1], dtype=np.int32)
    fill = offsets[:-1].copy()
    for h in range(hp.shape[0]):
        r = hr[h] + 1e-6 * cell
        lx, hx = _axis_cells(hp[h, 0], r, cell, nh)
        ly, hy = _axis_cells(hp[h, 1], r, cell, nh)
        lz, hz = _axis_cells(hp[h, 2], r, cell, nh)
        for i in range(lx, hx + 1):
            for j in range(ly, hy + 1):
                for k in range(lz, hz + 1):
                    c = ((i % nh) * nh + (j % nh)) * nh + (k % nh)
                    members[fill[c]] = h
                    fill[c] += 1
    return offsets, members


@numba.njit(parallel=True, cache=True)
def _query_hash(pos, hp, hr2, offsets, members, cell, nh, box):
    """First halo (in list order) whose sphere contains each particle, or -1."""
    n = pos.shape[0]
    out = np.full(n, -1, dtype=np.int32)
    half = 0.5 * box
    for p in numba.prange(n):
        x = np.float64(pos[p, 0])
        y = np.float64(pos[p, 1])
        z = np.float64(pos[p, 2])
        c = ((int(x / cell) % nh) * nh + (int(y / cell) % nh)) * nh + (int(z / cell) % nh)
        for q in range(offsets[c], offsets[c + 1]):
            h = members[q]
            dx = x - hp[h, 0]
            dy = y - hp[h, 1]
            dz = z - hp[h, 2]
            if dx > half:
                dx -= box
            elif dx < -half:
                dx += box
            if dy > half:
                dy -= box
            elif dy < -half:
                dy += box
            if dz > half:
                dz -= box
            elif dz < -half:
                dz += box
            if dx * dx + dy * dy + dz * dz <= hr2[h]:
                out[p] = h
                break
    return out


class ApertureLabeller:
    """Aperture labels with mass priority, for a fixed halo set and radius factor.

    A particle belongs to the halo of highest ``priority`` whose sphere of
    radius ``radius`` (periodic minimum-image distance) contains it. Haloes
    with non-positive or non-finite radius claim nothing.

    Args:
        halo_pos (np.ndarray): (H, 3) centres, same units as ``box``.
        radius (np.ndarray): (H,) aperture radii.
        priority (np.ndarray): (H,) priority (the halo mass); ties go to the
            lower row.
        box (float): Periodic box size.
        rows (np.ndarray, optional): Halo rows to consider (default: all).
        max_hash (int): Maximum hash cells per side.
    """

    def __init__(self, halo_pos, radius, priority, box, rows=None, max_hash=512):
        rows = np.arange(len(radius)) if rows is None else np.asarray(rows)
        r = np.asarray(radius, dtype=np.float64)[rows]
        ok = np.isfinite(r) & (r > 0)
        rows, r = rows[ok], r[ok]
        order = np.argsort(-np.asarray(priority, dtype=np.float64)[rows], kind='stable')
        self.rows = rows[order].astype(np.int64)
        self.box = float(box)
        self.hp = np.mod(np.asarray(halo_pos, dtype=np.float64)[self.rows], self.box)
        hr = r[order]
        self.hr2 = hr * hr
        if len(hr):
            nh = int(np.clip(self.box / (2.0 * np.median(hr)), 1, max_hash))
        else:
            nh = 1
        self.nh = nh
        self.cell = self.box / nh  # the hash grid must tile the box exactly
        self.offsets, self.members = _build_hash(self.hp, hr, self.cell, nh)

    def labels(self, pos):
        """Halo row of each particle (int32; NO_HALO outside every aperture).

        Args:
            pos (np.ndarray): (N, 3) positions wrapped into [0, box).
        """
        if len(self.rows) == 0 or len(pos) == 0:
            return np.full(len(pos), NO_HALO, dtype=np.int32)
        idx = _query_hash(pos, self.hp, self.hr2, self.offsets, self.members,
                          self.cell, self.nh, self.box)
        return np.where(idx >= 0, self.rows[np.maximum(idx, 0)], NO_HALO).astype(np.int32)


def restrict_labels(labels, halo_mass, mass_min):
    """Drop labels of haloes below a mass cut.

    For aperture labels built with mass priority over the haloes above a lower
    cut, this equals labelling with the higher cut directly: the owner of a
    particle is the most massive halo containing it, so if it falls below the
    new cut no halo above the cut contains the particle.

    Args:
        labels (np.ndarray): Halo rows (NO_HALO = none).
        halo_mass (np.ndarray): Mass per halo row.
        mass_min (float): Cut (inclusive).

    Returns:
        np.ndarray: int32 labels.
    """
    out = labels.astype(np.int32, copy=True)
    has = out >= 0
    out[has] = np.where(halo_mass[out[has]] >= mass_min, out[has], NO_HALO)
    return out


# ---------------------------------------------------------------------------
# Transfer bookkeeping
# ---------------------------------------------------------------------------

def halo_totals(labels, masses, n_halo):
    """Sum of particle masses per halo row (float64).

    Args:
        labels (np.ndarray): Halo rows (NO_HALO ignored).
        masses (np.ndarray): Particle masses.
        n_halo (int): Number of halo rows.
    """
    m = labels >= 0
    return np.bincount(labels[m], weights=masses[m].astype(np.float64), minlength=n_halo)


def transfer_factors(mstar_h, mion_h):
    """Per-halo ratio f_h = M*_h / M_ion,h of moved stellar to ionized-gas mass.

    Args:
        mstar_h (np.ndarray): Stellar mass per halo region.
        mion_h (np.ndarray): Ionized-gas mass per halo region.

    Returns:
        tuple: (f, active, kept): f (float64; 0 where nothing moves), active
        (stars moved), kept (stars but no ionized gas: stars stay in place).
    """
    active = (mstar_h > 0) & (mion_h > 0)
    kept = (mstar_h > 0) & ~(mion_h > 0)
    f = np.zeros(len(mstar_h), dtype=np.float64)
    f[active] = mstar_h[active] / mion_h[active]
    return f, active, kept


def p_of_scale(P_mm, P_mD, P_DD, s):
    """Matter power spectrum at stellar scale s.

    P_mm(s) = P_mm + 2 (1 - s) P_mD + (1 - s)^2 P_DD, exact for the field
    rho_m + (1 - s) D (delta_D = D / rhobar_m).

    Args:
        P_mm, P_mD, P_DD (np.ndarray): Spectra per k.
        s (float): Fraction of the selected stellar mass kept as stars,
            0 <= s <= 1 (1 = the simulation, 0 = all moved).

    Returns:
        np.ndarray: P_mm(s).
    """
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"stellar scale must be in [0, 1], got {s}")
    t = 1.0 - s
    return P_mm + 2.0 * t * P_mD + t * t * P_DD


def deposit(grid, pos, weights, box):
    """Add a TSC deposit of weighted particles to ``grid`` in place.

    Same function and cell convention as the cached 3D fields
    (``mapMaker.make_mass_field``); periodic. Note ``tsc_parallel`` casts
    each weight to the position dtype (float32 for the stored particles).

    The thread count is capped here: ``tsc_parallel`` splits the grid into
    stripes (about n / (2 nthread) cells wide) scattered in parallel, and
    stripes narrower than ~3 cells let two threads add to the same cell at
    once, silently losing mass (seen at 2-cell stripes). With nthread=-1 it
    would use ``numba.config.NUMBA_NUM_THREADS`` (fixed at start-up, e.g.
    256 on a CPU node without the runner's export), which gives 2-cell
    stripes at n = 1000. Capping at n // 8 keeps stripes >= 4 cells.
    """
    if tsc_parallel is None:
        raise ImportError("abacusutils (abacusnbody.analysis.tsc) is required for TSC deposits")
    if len(pos) == 0:
        return grid
    nthread = max(1, min(numba.config.NUMBA_NUM_THREADS, grid.shape[0] // 8))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # float64 grid/weights advisory
        tsc_parallel(pos, grid, box, weights=weights, nthread=nthread)
    return grid


# ---------------------------------------------------------------------------
# Particle pass
# ---------------------------------------------------------------------------

def _is_true_star(particles, sim_type):
    """Stars proper: TNG/Illustris PartType4 also holds wind-phase particles
    (GFM_StellarFormationTime <= 0), which are gas in transit, not stars."""
    if sim_type == 'IllustrisTNG':
        return particles['GFM_StellarFormationTime'] > 0
    return np.ones(len(particles['Masses']), dtype=bool)


def _chunks(stacker):
    """(chunk index, file path) of the snapshot files, in particle order."""
    if stacker.simType == 'IllustrisTNG':
        n = int(stacker.header['NumFilesPerSnapshot'])
        return [(i, stacker.snapPath(chunkNum=i)) for i in range(n)]
    if stacker.simType == 'SIMBA':
        return [(0, stacker.snapPath())]
    if stacker.simType == 'FLAMINGO':
        files = glob.glob(stacker.snapPath(pathOnly=True) + 'flamingo_*.hdf5')
        idx = [int(os.path.basename(f).split('.')[1]) for f in files]
        return sorted(zip(idx, files))
    raise NotImplementedError(stacker.simType)


def _chunk_counts(path):
    with h5py.File(path, 'r') as f:
        return np.asarray(f['Header'].attrs['NumPart_ThisFile'], dtype=np.int64)


def collect_particles(stacker, variants, halos, mass_min, max_chunks=None,
                      block_size=100_000_000, verbose=True, baryons=False):
    """Read stars and gas once and keep those assigned to a halo by any variant.

    Args:
        stacker (SimulationStacker): The simulation.
        variants (list): Dicts with 'name' and 'method' ('membership' or
            'aperture'; aperture also 'x', the radius in units of GroupRad).
        halos (dict): ``stacker.loadHalos()`` output.
        mass_min (float): Lowest halo mass cut (GroupMass, Msun/h); labels of
            smaller haloes are dropped here (higher cuts: ``restrict_labels``).
        max_chunks (int, optional): Read only the first chunks (smoke tests).
        block_size (int): Kept particles are merged into blocks of about this
            size (fewer, larger TSC calls).
        verbose (bool): Print progress.
        baryons (bool): Also keep the masses and labels (no positions) of the
            assigned TNG/Illustris wind particles ('Winds') and black holes
            ('BH'), for per-region baryon budgets (``halo_baryons``).

    Returns:
        dict: 'Stars' and 'gas' -> list of blocks; each block has 'pos'
        (float32, wrapped), 'mass' (float32 Msun/h: stellar mass, or
        ionized-gas mass for gas), 'labels' ({variant name: int32}), and for
        gas 'm_gas' (float32 Msun/h) and 'sf' (bool, star-forming). With
        ``baryons``, also 'Winds' and 'BH' blocks ('mass', 'labels'). Also
        'totals': box sums over every particle read (true stars, winds, gas,
        ionized gas, and sums of squared masses; with ``baryons`` also BH).
    """
    st = stacker
    box = float(st.header['BoxSize'])
    gmass = np.asarray(halos['GroupMass'], dtype=np.float64)
    labellers = {}
    for v in variants:
        if v['method'] == 'membership':
            labellers[v['name']] = MembershipLabeller(st)
        elif v['method'] == 'aperture':
            rows = np.flatnonzero(gmass >= mass_min)
            radius = v['x'] * np.asarray(halos['GroupRad'], dtype=np.float64)
            labellers[v['name']] = ApertureLabeller(halos['GroupPos'], radius, gmass, box, rows=rows)
        else:
            raise ValueError(v['method'])
    sfr_key = 'StarFormationRates' if st.simType == 'FLAMINGO' else 'StarFormationRate'
    keys = {'Stars': ['Coordinates', 'Masses'] + (['GFM_StellarFormationTime']
                                                   if st.simType == 'IllustrisTNG' else []),
            'gas': ['Coordinates'] + ionized_gas_keys(st.simType) + [sfr_key],
            'BH': ['Coordinates', 'Masses']}
    read_types = ('Stars', 'gas', 'BH') if baryons else ('Stars', 'gas')
    kept_types = read_types + ('Winds',) if baryons else read_types

    totals = {k: 0.0 for k in ('mstar', 'mwind', 'mgas', 'mion', 'm2_stars', 'm2_gas')}
    if baryons:
        totals['mbh'] = 0.0
    out = {p: [] for p in kept_types}
    pending = {p: [] for p in kept_types}

    def flush(p_type):
        if pending[p_type]:
            blk = {k: (np.concatenate([b[k] for b in pending[p_type]]) if k != 'labels' else
                       {n: np.concatenate([b['labels'][n] for b in pending[p_type]])
                        for n in pending[p_type][0]['labels']})
                   for k in pending[p_type][0]}
            out[p_type].append(blk)
            pending[p_type].clear()

    chunks = _chunks(st)[:max_chunks] if max_chunks else _chunks(st)
    start = {p: 0 for p in read_types}
    t0 = time.time()
    for ic, (chunk, path) in enumerate(chunks):
        counts = _chunk_counts(path)
        for p_type in read_types:
            n = int(counts[_PTYPE_INDEX[p_type]])
            g0 = start[p_type]
            start[p_type] += n
            if n == 0:
                continue
            part = load_subset(st.simPath, st.snapshot, st.simType, p_type, snap_path=path,
                               header=st.header, sim_name=st.sim, keys=keys[p_type])
            mass = part['Masses'].astype(np.float64) * 1e10  # Msun/h
            if len(mass) != n:
                raise RuntimeError(f"chunk {chunk}: {len(mass)} {p_type} read, header says {n}")
            pos = np.mod(np.asarray(part['Coordinates'], dtype=np.float64), box).astype(np.float32)
            labels = {}
            for name, lab in labellers.items():
                if isinstance(lab, MembershipLabeller):
                    l_ = lab.labels(p_type, chunk, g0, n)
                    labels[name] = restrict_labels(l_, gmass, mass_min)
                else:
                    labels[name] = lab.labels(pos)
            if p_type == 'Stars':
                star = _is_true_star(part, st.simType)
                totals['mstar'] += float(mass[star].sum())
                totals['mwind'] += float(mass[~star].sum())
                totals['m2_stars'] += float(np.sum(mass[star] ** 2))
                assigned = np.any([l_ >= 0 for l_ in labels.values()], axis=0)
                keep = star & assigned
                blk = dict(pos=pos[keep], mass=mass[keep].astype(np.float32),
                           labels={k_: l_[keep] for k_, l_ in labels.items()})
                if baryons:
                    # masses and labels only (never deposited): kept in small
                    # per-chunk blocks, merged once at the end
                    wind = ~star & assigned
                    if wind.any():
                        pending['Winds'].append(dict(mass=mass[wind].astype(np.float32),
                                                     labels={k_: l_[wind] for k_, l_ in labels.items()}))
            elif p_type == 'BH':
                totals['mbh'] += float(mass.sum())
                keep = np.any([l_ >= 0 for l_ in labels.values()], axis=0)
                blk = dict(mass=mass[keep].astype(np.float32),
                           labels={k_: l_[keep] for k_, l_ in labels.items()})
            else:
                mion = ionized_gas_masses(part, st)
                totals['mgas'] += float(mass.sum())
                totals['mion'] += float(mion.sum())
                totals['m2_gas'] += float(np.sum(mass ** 2))
                keep = np.any([l_ >= 0 for l_ in labels.values()], axis=0)
                blk = dict(pos=pos[keep], mass=mion[keep].astype(np.float32),
                           m_gas=mass[keep].astype(np.float32),
                           sf=np.asarray(part[sfr_key])[keep] > 0,
                           labels={k_: l_[keep] for k_, l_ in labels.items()})
            pending[p_type].append(blk)
            if sum(len(b['mass']) for b in pending[p_type]) >= block_size:
                flush(p_type)
            del part, mass, pos, labels
        if verbose and (ic % 25 == 0 or ic == len(chunks) - 1):
            kept = {p: sum(len(b['mass']) for b in out[p] + pending[p]) for p in out}
            print(f"  chunk {ic + 1}/{len(chunks)} read ({time.time() - t0:.0f} s); kept "
                  f"{kept['Stars']:,} stars, {kept['gas']:,} gas", flush=True)
    for p_type in out:
        flush(p_type)
    out['totals'] = totals
    out['n_chunks'] = len(chunks)
    return out


# ---------------------------------------------------------------------------
# Transfer field
# ---------------------------------------------------------------------------

def _neg_mass(stars_field, grid):
    """Sum and cell count of min(stars_field + grid, 0), slab by slab."""
    neg, ncell = 0.0, 0
    for i in range(grid.shape[0]):
        d = stars_field[i].astype(np.float64) + grid[i]
        m = d < 0
        neg += float(d[m].sum())
        ncell += int(m.sum())
    return neg, ncell


def _weighted_percentiles(values, weights, q):
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    cw /= cw[-1]
    return np.interp(np.asarray(q) / 100.0, cw, values[order])


def transfer_field(store, name, halo_mass, mass_min, n_pixels, box, stars_field=None,
                   keep_halo_table=False):
    """Build the zero-mass transfer field D of one configuration (s = 0).

    Args:
        store (dict): From ``collect_particles``.
        name (str): Variant name (key of the stored labels).
        halo_mass (np.ndarray): GroupMass per halo row (Msun/h).
        mass_min (float): Halo mass cut (Msun/h).
        n_pixels (int): Grid size per side.
        box (float): Box size (kpc/h).
        stars_field (np.ndarray, optional): Cached Stars field on the same
            grid; if given, the removed stars are checked against it
            (cells where the modified stellar field would be negative).
        keep_halo_table (bool): Also return per-halo masses of the active
            haloes.

    Returns:
        tuple: (D, diag): D is float64 (n, n, n) in Msun/h per cell;
        diag holds the bookkeeping and conservation diagnostics.
    """
    n_halo = len(halo_mass)
    mstar_h = np.zeros(n_halo)
    mion_h = np.zeros(n_halo)
    for blk in store['Stars']:
        mstar_h += halo_totals(restrict_labels(blk['labels'][name], halo_mass, mass_min),
                               blk['mass'], n_halo)
    for blk in store['gas']:
        mion_h += halo_totals(restrict_labels(blk['labels'][name], halo_mass, mass_min),
                              blk['mass'], n_halo)
    f, active, kept = transfer_factors(mstar_h, mion_h)

    grid = np.zeros((n_pixels,) * 3, dtype=np.float64)
    removed_h = np.zeros(n_halo)
    s3 = 0.0
    for blk in store['Stars']:
        lab = restrict_labels(blk['labels'][name], halo_mass, mass_min)
        sel = lab >= 0
        sel[sel] = active[lab[sel]]
        w = blk['mass'][sel].astype(np.float64)
        removed_h += np.bincount(lab[sel], weights=w, minlength=n_halo)
        s3 += float(np.sum(w ** 2))
        deposit(grid, blk['pos'][sel], -w, box)
    moved = float(mstar_h[active].sum())
    dep_stars = float(grid.sum(dtype=np.float64))
    diag = {}
    if stars_field is not None:
        diag['neg_star_mass'], diag['neg_star_cells'] = _neg_mass(stars_field, grid)

    added_h = np.zeros(n_halo)
    s1 = s2 = added_sf = 0.0
    for blk in store['gas']:
        lab = restrict_labels(blk['labels'][name], halo_mass, mass_min)
        sel = lab >= 0
        sel[sel] = active[lab[sel]]
        w = f[lab[sel]] * blk['mass'][sel].astype(np.float64)
        added_h += np.bincount(lab[sel], weights=w, minlength=n_halo)
        s1 += float(np.sum(2.0 * blk['m_gas'][sel].astype(np.float64) * w))
        s2 += float(np.sum(w ** 2))
        added_sf += float(w[blk['sf'][sel]].sum())
        deposit(grid, blk['pos'][sel], w, box)

    sel_h = halo_mass >= mass_min
    cons = np.abs(added_h[active] - mstar_h[active]) / mstar_h[active] if active.any() else np.zeros(1)
    rem = np.abs(removed_h[active] - mstar_h[active]) / mstar_h[active] if active.any() else np.zeros(1)
    fa = f[active]
    pct = _weighted_percentiles(fa, mstar_h[active], [50, 90, 99]) if active.any() else [np.nan] * 3
    diag.update(
        n_haloes_selected=int(sel_h.sum()),
        n_haloes_with_stars=int(((mstar_h > 0) & sel_h).sum()),
        n_haloes_active=int(active.sum()),
        n_haloes_kept_noion=int(kept.sum()),
        mstar_assigned=float(mstar_h.sum()),
        mstar_moved=moved,
        mstar_kept_noion=float(mstar_h[kept].sum()),
        mion_regions=float(mion_h[active].sum()),
        max_halo_cons=float(cons.max()),
        max_halo_removed=float(rem.max()),
        sum_D=float(grid.sum(dtype=np.float64)),
        dep_rel_err_stars=(dep_stars + moved) / moved if moved > 0 else 0.0,
        sf_frac_added=added_sf / moved if moved > 0 else 0.0,
        f_p50=float(pct[0]), f_p90=float(pct[1]), f_p99=float(pct[2]),
        frac_moved_f_gt1=float(mstar_h[active][fa > 1].sum() / moved) if moved > 0 else 0.0,
        sn_S1=s1, sn_S2=s2, sn_S3=s3,
    )
    diag['sum_D_rel'] = diag['sum_D'] / moved if moved > 0 else 0.0
    if keep_halo_table:
        rows = np.flatnonzero(active)
        diag['halo_table'] = dict(rows=rows.astype(np.int32), mstar=mstar_h[rows],
                                  mion=mion_h[rows])
        diag['mstar_h_all'] = mstar_h
    return grid, diag


# ---------------------------------------------------------------------------
# Projected (2D) transfer maps
# ---------------------------------------------------------------------------

_PROJECTION_AXES = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}


def pixel_index_2d(pos, n_pixels, box, projection='yz'):
    """Flat pixel index (ix * n + iy) of each particle on a projected 2D grid.

    The binning of the cached 2D fields (``mapMaker.make_mass_field``: a
    ``binned_statistic_2d`` sum over [0, box] with n x n bins, the right edge
    in the last bin). The bin numbers come from that same scipy call, so a
    ``bincount`` over this index reproduces its sum map exactly.

    Args:
        pos (np.ndarray): (N, 3) positions in [0, box] (wrapped).
        n_pixels (int): Pixels per side.
        box (float): Box size, in the units of ``pos``.
        projection (str): 'xy', 'xz' or 'yz' (the two kept axes, in order).

    Returns:
        np.ndarray: Flat indices, shape (N,); int32 (int64 if n^2 needs it).

    Raises:
        ValueError: If a particle lies outside [0, box] (the cached fields
            would drop it).
    """
    a, b = _PROJECTION_AXES[projection]
    n = int(n_pixels)
    dtype = np.int32 if n * n < 2 ** 31 else np.int64
    if len(pos) == 0:
        return np.zeros(0, dtype=dtype)
    res = binned_statistic_2d(pos[:, a], pos[:, b], None, 'count', bins=[n, n],
                              range=[[0, box], [0, box]])
    ix, iy = np.divmod(res.binnumber, n + 2)  # bin numbers include the outlier bins
    ix -= 1
    iy -= 1
    if ix.min() < 0 or ix.max() >= n or iy.min() < 0 or iy.max() >= n:
        raise ValueError("particles outside [0, box]")
    return (ix * n + iy).astype(dtype)


def transfer_maps_2d(store, name, halo_mass, mass_min, n_pixels):
    """Projected maps of one configuration's transfer at s = 0.

    The 2D counterpart of ``transfer_field``: the same per-halo bookkeeping
    (regions from the stored labels at the cut, f_h = M*_h / M_ion,h, haloes
    without ionized gas keep their stars), binned like the cached 2D fields
    (every stored block needs a 'pix' index from ``pixel_index_2d``). The
    projected transfer field is added - removed, so for any stellar scale s
    (t = 1 - s) the maps follow exactly:

        ionized_gas(s) = ionized_gas + t * added ,
        total(s)       = total + t * (added - removed) .

    Args:
        store (dict): From ``collect_particles``, with 'pix' in every block.
        name (str): Variant name (key of the stored labels).
        halo_mass (np.ndarray): GroupMass per halo row (Msun/h).
        mass_min (float): Halo mass cut (Msun/h).
        n_pixels (int): Pixels per side of the grid 'pix' refers to.

    Returns:
        tuple: (added, removed, diag): the ionized-gas mass added and the
        stellar mass removed, float64 (n, n) in Msun/h per pixel, both >= 0;
        diag holds the bookkeeping and conservation numbers.
    """
    n_halo = len(halo_mass)
    n = int(n_pixels)
    mstar_h = np.zeros(n_halo)
    mion_h = np.zeros(n_halo)
    for blk in store['Stars']:
        mstar_h += halo_totals(restrict_labels(blk['labels'][name], halo_mass, mass_min),
                               blk['mass'], n_halo)
    for blk in store['gas']:
        mion_h += halo_totals(restrict_labels(blk['labels'][name], halo_mass, mass_min),
                              blk['mass'], n_halo)
    f, active, kept = transfer_factors(mstar_h, mion_h)

    maps, per_halo = {}, {}
    for p_type, key in (('Stars', 'removed'), ('gas', 'added')):
        flat = np.zeros(n * n)
        h = np.zeros(n_halo)
        for blk in store[p_type]:
            lab = restrict_labels(blk['labels'][name], halo_mass, mass_min)
            sel = lab >= 0
            sel[sel] = active[lab[sel]]
            w = blk['mass'][sel].astype(np.float64)
            if p_type == 'gas':
                w *= f[lab[sel]]
            h += np.bincount(lab[sel], weights=w, minlength=n_halo)
            flat += np.bincount(blk['pix'][sel], weights=w, minlength=n * n)
        maps[key] = flat.reshape(n, n)
        per_halo[key] = h

    moved = float(mstar_h[active].sum())
    ms = mstar_h[active]
    cons = np.abs(per_halo['added'][active] - ms) / ms if active.any() else np.zeros(1)
    rem = np.abs(per_halo['removed'][active] - ms) / ms if active.any() else np.zeros(1)
    sum_added = float(maps['added'].sum())
    sum_removed = float(maps['removed'].sum())
    diag = dict(
        n_haloes_selected=int((halo_mass >= mass_min).sum()),
        n_haloes_active=int(active.sum()),
        n_haloes_kept_noion=int(kept.sum()),
        mstar_moved=moved,
        mstar_kept_noion=float(mstar_h[kept].sum()),
        sum_added=sum_added,
        sum_removed=sum_removed,
        sum_added_rel=(sum_added - moved) / moved if moved > 0 else 0.0,
        sum_removed_rel=(sum_removed - moved) / moved if moved > 0 else 0.0,
        max_halo_cons=float(cons.max()),
        max_halo_removed=float(rem.max()),
    )
    return maps['added'], maps['removed'], diag


# ---------------------------------------------------------------------------
# Stellar-fraction floor (per-halo scaled transfer)
# ---------------------------------------------------------------------------

def halo_baryons(store, name, halo_mass, mass_min):
    """Baryonic mass per halo region of one configuration.

    Needs a store from ``collect_particles(..., baryons=True)``.

    Args:
        store (dict): Particle store.
        name (str): Variant name (key of the stored labels).
        halo_mass (np.ndarray): GroupMass per halo row (Msun/h).
        mass_min (float): Halo mass cut (Msun/h).

    Returns:
        dict: float64 arrays over halo rows (Msun/h): 'mstar' (true stars),
        'mwind' (TNG/Illustris wind particles), 'mgas' (all gas), 'mion'
        (ionized gas), 'mbh' (black holes) and 'mbaryon' (stars + winds +
        gas + BH). Zero for haloes below the cut.
    """
    n = len(halo_mass)
    spec = (('mstar', 'Stars', 'mass'), ('mion', 'gas', 'mass'), ('mgas', 'gas', 'm_gas'),
            ('mwind', 'Winds', 'mass'), ('mbh', 'BH', 'mass'))
    out = {}
    for key, p_type, field in spec:
        tot = np.zeros(n)
        for blk in store[p_type]:
            tot += halo_totals(restrict_labels(blk['labels'][name], halo_mass, mass_min),
                               blk[field], n)
        out[key] = tot
    out['mbaryon'] = out['mstar'] + out['mwind'] + out['mgas'] + out['mbh']
    return out


def floor_coefficients(bary, f_target):
    """Stars to add so that every region's star fraction is at least f_target.

    With f*_h = M*_h / M_b,h (``halo_baryons``: true stars over stars + winds
    + gas + BH), a region below the target gains dM_h = f_target M_b,h - M*_h
    of stars, laid out like its stars and taken from its ionized gas in
    proportion to the ionized mass; regions at or above the target are
    unchanged (a floor). A region with less ionized gas than dM_h converts all
    of it (capped); regions without stars (no stellar template) or without
    ionized gas stay unchanged.

    Args:
        bary (dict): From ``halo_baryons``.
        f_target (float): Target star fraction.

    Returns:
        tuple: (coef, info): coef_h = dM_h / M*_h (0 where unchanged), so the
        change is coef_h m*_j on each star and -coef_h (M*_h / M_ion,h) m_ion,i
        on each gas particle; info has 'dm' and the boolean arrays 'raised',
        'capped', 'no_stars', 'no_ion' (over regions with baryons).
    """
    ms, mi, mb = bary['mstar'], bary['mion'], bary['mbaryon']
    need = f_target * mb - ms
    below = (mb > 0) & (need > 0)
    no_stars = below & ~(ms > 0)
    no_ion = below & (ms > 0) & ~(mi > 0)
    raised = below & (ms > 0) & (mi > 0)
    dm = np.zeros(len(ms))
    dm[raised] = np.minimum(need[raised], mi[raised])
    coef = np.zeros(len(ms))
    coef[raised] = dm[raised] / ms[raised]
    return coef, dict(dm=dm, raised=raised, capped=raised & (need > mi),
                      no_stars=no_stars, no_ion=no_ion)


def _scaled_weights(store, name, halo_mass, mass_min, coef, bary):
    """Per-block weights of the scaled transfer: yields (p_type, block, sel, labels, w).

    Stars gain w = coef_h m*_j; ionized gas loses coef_h (M*_h / M_ion,h)
    m_ion,i, returned as a positive w (the removed mass).
    """
    ratio = np.zeros(len(halo_mass))
    ok = bary['mion'] > 0
    ratio[ok] = bary['mstar'][ok] / bary['mion'][ok]
    for p_type in ('Stars', 'gas'):
        for blk in store[p_type]:
            lab = restrict_labels(blk['labels'][name], halo_mass, mass_min)
            sel = lab >= 0
            sel[sel] = coef[lab[sel]] > 0
            ls = lab[sel]
            w = blk['mass'][sel].astype(np.float64) * coef[ls]
            if p_type == 'gas':
                w *= ratio[ls]
            yield p_type, blk, sel, ls, w


def _floor_diag(bary, coef, info, f_target, per_halo):
    """Bookkeeping of one floor configuration (shared by the 3D and 2D builds)."""
    dm, raised, capped = info['dm'], info['raised'], info['capped']
    ms, mb, mi = bary['mstar'], bary['mbaryon'], bary['mion']
    has = mb > 0
    moved = float(dm.sum())
    r = raised
    cons = [np.abs(per_halo[k][r] - dm[r]) / dm[r] if r.any() else np.zeros(1)
            for k in ('Stars', 'gas')]
    f_after = np.where(has, (ms + dm) / np.where(has, mb, 1.0), 0.0)
    ok = r & ~capped
    return dict(
        f_target=float(f_target),
        n_regions=int(has.sum()),
        n_raised=int(r.sum()), n_capped=int(capped.sum()),
        n_no_stars=int(info['no_stars'].sum()), n_no_ion=int(info['no_ion'].sum()),
        mbaryon_regions=float(mb[has].sum()), mstar_regions=float(ms[has].sum()),
        mwind_regions=float(bary['mwind'][has].sum()), mbh_regions=float(bary['mbh'][has].sum()),
        mgas_regions=float(bary['mgas'][has].sum()), mion_regions=float(mi[has].sum()),
        fstar_sim=float(ms[has].sum() / mb[has].sum()) if has.any() else np.nan,
        fstar_after=float((ms[has] + dm[has]).sum() / mb[has].sum()) if has.any() else np.nan,
        mstar_added=moved,
        mstar_added_capped=float(dm[capped].sum()),
        mbaryon_no_stars=float(mb[info['no_stars']].sum()),
        mbaryon_no_ion=float(mb[info['no_ion']].sum()),
        max_halo_cons_stars=float(cons[0].max()), max_halo_cons_gas=float(cons[1].max()),
        max_removed_over_mion=float((dm[r] / mi[r]).max()) if r.any() else 0.0,
        min_fstar_after_minus_target=float((f_after[ok] - f_target).min()) if ok.any() else 0.0,
    )


def scaled_transfer_field(store, name, halo_mass, mass_min, coef, info, bary, f_target,
                          n_pixels, box):
    """3D change field of the stellar-fraction floor: stars added minus ionized gas removed.

    Args:
        store (dict): From ``collect_particles(..., baryons=True)``.
        name (str): Variant name.
        halo_mass (np.ndarray): GroupMass per halo row (Msun/h).
        mass_min (float): Halo mass cut (Msun/h).
        coef, info: From ``floor_coefficients``.
        bary (dict): From ``halo_baryons``.
        f_target (float): The target (for the diagnostics).
        n_pixels (int): Grid size per side.
        box (float): Box size (kpc/h).

    Returns:
        tuple: (H, diag): H float64 (n, n, n) in Msun/h per cell (TSC, as the
        cached fields), zero total mass; diag from ``_floor_diag`` plus the
        grid sums.
    """
    n_halo = len(halo_mass)
    grid = np.zeros((n_pixels,) * 3, dtype=np.float64)
    per_halo = {'Stars': np.zeros(n_halo), 'gas': np.zeros(n_halo)}
    for p_type, blk, sel, ls, w in _scaled_weights(store, name, halo_mass, mass_min, coef, bary):
        per_halo[p_type] += np.bincount(ls, weights=w, minlength=n_halo)
        deposit(grid, blk['pos'][sel], w if p_type == 'Stars' else -w, box)
    diag = _floor_diag(bary, coef, info, f_target, per_halo)
    moved = diag['mstar_added']
    diag['sum_H'] = float(grid.sum(dtype=np.float64))
    diag['sum_H_rel'] = diag['sum_H'] / moved if moved > 0 else 0.0
    return grid, diag


def scaled_transfer_maps_2d(store, name, halo_mass, mass_min, coef, info, bary, f_target,
                            n_pixels):
    """2D maps of the stellar-fraction floor: stars added and ionized gas removed.

    Same weights as ``scaled_transfer_field``, binned like the cached 2D
    fields (every block needs a 'pix' index from ``pixel_index_2d``). For the
    lensing ratio, ionized_gas -> ionized_gas - gas_removed and
    total -> total + stars_added - gas_removed.

    Returns:
        tuple: (stars_added, gas_removed, diag): float64 (n, n) in Msun/h per
        pixel, both >= 0; diag from ``_floor_diag`` plus the map sums.
    """
    n_halo = len(halo_mass)
    n = int(n_pixels)
    maps = {'Stars': np.zeros(n * n), 'gas': np.zeros(n * n)}
    per_halo = {'Stars': np.zeros(n_halo), 'gas': np.zeros(n_halo)}
    for p_type, blk, sel, ls, w in _scaled_weights(store, name, halo_mass, mass_min, coef, bary):
        per_halo[p_type] += np.bincount(ls, weights=w, minlength=n_halo)
        maps[p_type] += np.bincount(blk['pix'][sel], weights=w, minlength=n * n)
    diag = _floor_diag(bary, coef, info, f_target, per_halo)
    moved = diag['mstar_added']
    diag['sum_stars_added'] = float(maps['Stars'].sum())
    diag['sum_gas_removed'] = float(maps['gas'].sum())
    diag['sum_stars_rel'] = (diag['sum_stars_added'] - moved) / moved if moved > 0 else 0.0
    diag['sum_gas_rel'] = (diag['sum_gas_removed'] - moved) / moved if moved > 0 else 0.0
    return maps['Stars'].reshape(n, n), maps['gas'].reshape(n, n), diag
