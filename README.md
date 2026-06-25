## SimulationStacker

A simple repo for working with halo stacking on cosmological simulations.

## Installation

`src/` is not an installed package — scripts add `../src/` to `sys.path`, so no
`pip install` of this repo itself is required. You only need the dependencies.

There are two supported ways to get the dependencies:

### 1. NERSC / cosmodesi (the validated environment)

On Perlmutter, the dependencies come from the cosmodesi DR1 environment with a
virtualenv layered on top. Both lines are required:

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
```

`requirements-freeze.txt` is a full `pip freeze` snapshot of this exact stack
(Python 3.10.13). It is a reference/lock for the NERSC environment only — it
includes the entire cosmodesi/DESI conda base (some packages are DESI-only and not
on PyPI), so it will **not** `pip install` cleanly on other machines.

### 2. Portable (other machines)

Use the curated, import-derived requirements. Targets Python 3.10:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the core library and the main analysis scripts with
clean PyPI dependencies (numpy, scipy, matplotlib, astropy, h5py, pandas, numba,
PyYAML, asdf, abacusutils).

Some dependencies are not clean PyPI installs (compiled, system-library, or
GitHub-only) and are split into `requirements-optional.txt`. Install the ones you
need:

```bash
pip install -r requirements-optional.txt
```

Notes on the optional dependencies:

- **`illustris_python`** — required for any IllustrisTNG loading; installed from
  GitHub. SIMBA-only workflows do not need it (SIMBA data is read from HDF5
  directly).
- **`nbodykit`** — currently a hard import for the core library (`stacker` →
  `mapMaker` → an unguarded `from nbodykit.lab import ...` in `src/tools.py`),
  even though it is only actually used by the power-spectrum scripts. It is a
  legacy package with a non-trivial build.
- **`Pylians`** (`Pk_library` / `MAS_library`) — only used by the power-spectrum
  paths (guarded by `try/except` in `src/stacker.py`); needs a C/Cython toolchain.
- **`hdf5plugin`** — needed by the 3D masking scripts.
- **`mpi4py`, `pixell`** — only used by `scripts/get_AP_sz_marvin.py`. `mpi4py`
  must be built against the system MPI; `pixell` needs a Fortran compiler.

### Regenerating the requirements files

`requirements.txt` and `requirements-optional.txt` are curated by hand from the
project's actual imports (a blind `pip freeze` would pull in the whole inherited
cosmodesi base). Update a pin by reading the installed version from the active
environment, e.g. `python -c "import importlib.metadata as m; print(m.version('numpy'))"`.

`requirements-freeze.txt` is regenerated from the activated NERSC environment with
`pip freeze` (then re-add the descriptive header).
