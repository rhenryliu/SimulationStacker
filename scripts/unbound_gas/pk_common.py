"""pk_common.py
=============
Shared helpers for the matter power spectrum (component) scripts of the unbound
gas paper: ``build_dmo_field.py``, ``validate_pk_caches.py``,
``compute_pk_components.py`` and ``make_pk_alpha.py``. All read
``configs/unbound_gas/pk_components_z05.yaml``.

Each hydro simulation is split into five mass components on one 3D TSC grid,
built from the cached fields in ``<root>/<SimType>/products/3D/``::

    DM          = total - gas - Stars - BH
    ionized_gas   (cached)
    neutral_gas = gas - ionized_gas
    Stars         (cached)
    BH            (cached)

The DM field is derived rather than binned because no DM cache exists at the
grids used; TSC deposition is linear in the particle weights, so on a common
grid the subtraction is exact up to floating-point rounding.
"""

import os
import sys
from pathlib import Path

import numpy as np
import yaml

# src/ located relative to this file, so the helpers import both from scripts/
# (the repository convention) and from elsewhere (e.g. tests).
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from loadIO import _get_data_filepath, resolve_data_root  # noqa: E402

# Order of the components in every spectra file and weight vector.
COMPONENTS = ['DM', 'ionized_gas', 'neutral_gas', 'Stars', 'BH']

# Critical density today in (M_sun/h) / (Mpc/h)^3 (= 2.775e11 h^2 M_sun/Mpc^3).
RHO_CRIT_H = 2.77536627e11


def load_config(path: str) -> dict:
    """Read the YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def sim_label(entry: dict) -> str:
    """Human-readable label of a simulation entry, e.g. 'L1_m9 (fgas-8sigma)'."""
    if entry.get('feedback'):
        return f"{entry['name']} ({entry['feedback']})"
    return entry['name']


def select_sims(config: dict, only=None) -> list:
    """Return the simulation entries, optionally filtered by label or name.

    Args:
        config: Parsed configuration.
        only: Optional list of labels (``sim_label``), names, or feedback
            variants; an entry is kept if any of the three matches.

    Returns:
        List of simulation entry dicts.
    """
    sims = config['simulations']
    if not only:
        return sims
    keep = set(only)
    return [s for s in sims
            if {sim_label(s), s['name'], s.get('feedback')} & keep]


def dmo_entry(entry: dict) -> dict:
    """DMO reference run of a hydro entry, with suite and grid filled in."""
    d = dict(entry['dmo'])
    d.setdefault('sim_type', entry['sim_type'])
    d.setdefault('feedback', None)
    d['n_pixels'] = entry['n_pixels']
    return d


def field_path(sim_type: str, name: str, snapshot: int, feedback, p_type: str,
               n_pixels: int) -> Path:
    """Path of a cached 3D field, using the repository's filename convention."""
    return _get_data_filepath(sim_type, name, snapshot, feedback, p_type,
                              n_pixels, data_type='field', dim='3D')


def load_field(entry: dict, p_type: str) -> np.ndarray:
    """Load a cached 3D field of a hydro entry as a float32 array.

    float64 caches (the 1000^3 TNG/Illustris/SIMBA ones) are cast to float32,
    which halves memory; the relative rounding (~6e-8) is far below anything
    the power spectra resolve.

    Raises:
        FileNotFoundError: If the cache does not exist.
    """
    path = field_path(entry['sim_type'], entry['name'], entry['snapshot'],
                      entry.get('feedback'), p_type, entry['n_pixels'])
    if not path.exists():
        raise FileNotFoundError(f"missing cache {path}")
    arr = np.load(path, mmap_mode='r')
    return np.array(arr, dtype=np.float32)  # copies (and casts) into memory


def load_dmo_field(entry: dict) -> np.ndarray:
    """Load the DMO reference DM field of a hydro entry (float32)."""
    d = dmo_entry(entry)
    path = field_path(d['sim_type'], d['name'], d['snapshot'], d['feedback'],
                      'DM', d['n_pixels'])
    if not path.exists():
        raise FileNotFoundError(f"missing DMO field {path} (run build_dmo_field.py)")
    return np.array(np.load(path, mmap_mode='r'), dtype=np.float32)


def box_size_mpc(entry: dict) -> float:
    """Box size of a hydro entry in Mpc/h, from the snapshot header."""
    return header_of(entry)['BoxSize'] / 1000.0


def header_of(entry: dict) -> dict:
    """Snapshot header of a hydro entry, normalized to TNG-style keys.

    Uses SimulationStacker (header-only; cheap) so FLAMINGO headers get the
    same normalization as everywhere else in the pipeline.
    """
    from stacker import SimulationStacker
    st = SimulationStacker(entry['name'], entry['snapshot'],
                           simType=entry['sim_type'],
                           feedback=entry.get('feedback'),
                           z=entry.get('redshift', 0.5))
    return st.header


def spectra_path(entry: dict, kind: str) -> Path:
    """Path of a spectra file for a hydro entry.

    Args:
        entry: Simulation entry.
        kind: 'components' (the 15 component spectra) or 'dmo' (total, DMO and
            their cross spectrum).
    """
    root = resolve_data_root(None)
    if entry.get('feedback'):
        stem = f"{entry['name']}_{entry['feedback']}_{entry['snapshot']}"
    else:
        stem = f"{entry['name']}_{entry['snapshot']}"
    return Path(root) / entry['sim_type'] / 'products' / '3D' / \
        f"{stem}_Pk_{kind}_{entry['n_pixels']}.npz"


def save_npz_atomic(path: Path, **arrays) -> None:
    """Write an .npz via a temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)
    print(f"saved {path}")


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Write an .npy via a temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as fh:
        np.save(fh, array)
    os.replace(tmp, path)
    print(f"saved {path}")


def to_overdensity(field: np.ndarray) -> float:
    """Convert a density field to its overdensity in place.

    Args:
        field: float32 array, overwritten with field/mean - 1.

    Returns:
        The mean of the input field (float64), needed for the mass weights.
    """
    mean = float(np.mean(field, dtype=np.float64))
    field /= np.float32(mean)
    field -= np.float32(1.0)
    return mean
