# SimulationStacker

A Python toolkit for **halo stacking on cosmological simulations**
(IllustrisTNG, SIMBA, and FLAMINGO). It builds 2D/3D projected fields from
particle data, stacks them on selected halos with aperture filters, and produces
radial profiles for comparison between simulations (and against observations).

Typical uses: gas/baryon fraction profiles, weak-lensing ΔΣ, and SZ (tSZ / kSZ /
τ) profiles, stacked over a chosen halo sample.

If you just want to see it work, jump to [Quickstart](#quickstart-your-first-stack).

## What's in here

```
src/            core library (imported by scripts via ../src on sys.path)
  stacker.py      SimulationStacker: the main entry point (fields, maps, stacking, caching)
  mapMaker.py     low-level field creation from particles (per particle type)
  filters.py      stacking filters: CAP, cumulative, DSigma, upsilon, ...
  loadIO.py       all TNG/SIMBA/FLAMINGO catalog + particle I/O; data-root resolution
  halos.py        select_halos(): 'binned', 'massive', 'abundance' (SHAM) halo selection
  tools.py        numba-JIT histogram / TSC binning (performance-critical)
  utils.py, field_utils.py, mask_utils.py, snr.py   cosmology, masking, SNR helpers
scripts/        analysis + figure scripts, each driven by a YAML config in configs/
  demo_stack.py   >>> the minimal, documented example — start here <<<
  configs/        YAML configs for every script
tests/          pytest suite (FLAMINGO I/O + SZ) plus interactive notebooks
```

`src/` is **not** an installed package — scripts add `../src/` to `sys.path`. See
[CLAUDE.md](CLAUDE.md) for a fuller architecture tour and the exact unit
conventions per simulation.

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
on PyPI, and a few lines are local editable installs), so it will **not**
`pip install` cleanly on other machines.

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

Notes on the optional dependencies (all are now guarded with `try/except`, so the
core library imports without them — you only need each for the workflow that uses it):

- **`illustris_python`** — required for any IllustrisTNG loading; installed from
  GitHub. SIMBA- or FLAMINGO-only workflows do not need it (both read HDF5
  directly). TNG code paths raise a clear error if it is missing.
- **`abacusnbody`** (abacusutils) — provides `tsc_parallel`, used only for **3D**
  (`dim='3D'`) field creation. 2D fields/maps and the demo do not need it.
- **`nbodykit`** — used only by the power-spectrum utilities (`src/tools.py`,
  `src/field_utils.py`, `scripts/powerspectra/`). A legacy package with a
  non-trivial build.
- **`Pylians`** (`Pk_library` / `MAS_library`) — only used by the power-spectrum
  paths in `src/stacker.py`; needs a C/Cython toolchain.
- **`hdf5plugin`** — needed by the 3D masking scripts.
- **`mpi4py`, `pixell`** — only used by `scripts/get_AP_sz_marvin.py`.

### Regenerating the requirements files

`requirements.txt` and `requirements-optional.txt` are curated by hand from the
project's actual imports (a blind `pip freeze` would pull in the whole inherited
cosmodesi base). Update a pin by reading the installed version from the active
environment, e.g. `python -c "import importlib.metadata as m; print(m.version('numpy'))"`.

`requirements-freeze.txt` is regenerated from the activated NERSC environment with
`pip freeze` (then re-add the descriptive header).

## Quickstart: your first stack

The fastest way in is the documented demo. From the `scripts/` directory:

```bash
cd scripts/
python demo_stack.py -p configs/demo_stack.yaml
```

It stacks one particle type (default: `gas`) with one filter (default: `CAP`) on
a selected halo sample, for TNG300-1, SIMBA, and FLAMINGO at z = 0.5, and writes a
figure + an `.npz` of the profiles to a dated folder `figures/<YYYY-MM>/<MM-DD>/`
(the same convention as the other scripts). Edit `configs/demo_stack.yaml` to
change the particle type, filter, mode, or simulation list. (On a first run, any
field not already cached is computed from particles, which can take a while —
FLAMINGO especially; comment out blocks in the config to start small.)

`demo_stack.py` is formatted like the other scripts in `scripts/` — same import
layout, `matplotlib` style block (`text.usetex=True`; set it to `False` if you
have no LaTeX), dated output folders, and nested YAML config — so it doubles as a
template to copy from.

Equivalently, the core API in a few lines:

```python
import sys; sys.path.insert(0, 'src')          # or run from scripts/ with '../src'
from stacker import SimulationStacker

s = SimulationStacker(sim='TNG300-1', snapshot=67, simType='IllustrisTNG',
                      z=0.5, nPixels=1000)

# radii (in units of radDistance), profiles has shape (numRadii, n_halos)
radii, profiles = s.stackField('gas', filterType='CAP',
                               minRadius=0.1, maxRadius=6.0, numRadii=15,
                               radDistance=1000.0,           # kpc/h per radial unit
                               halo_mass_avg=1.665e13)        # 'massive' selection
mean_profile = profiles.mean(axis=1)
```

## How stacking works

**Two workflows.** Both select halos, cut out a periodic patch around each, apply
a filter at a set of radii, and average over halos:

| Workflow | Method | Units | Beam |
|----------|--------|-------|------|
| **Fields** | `makeField` / `stackField` | native, radii in comoving kpc/h | none |
| **Maps**   | `makeMap` / `stackMap`     | beam-convolved, radii in arcmin | Gaussian FWHM |

The demo's `stack.mode` selects between them (`'field'` is simplest; `'map'` is
closer to an observation). Both ultimately call `stack_on_array`, which also
stacks any 2D array you pass in directly.

**Filters** (`filterType`, in [src/filters.py](src/filters.py)): `'CAP'` (Circular
Aperture Photometry, the default), `'cumulative'` (total enclosed mass),
`'DSigma'` (excess surface density ΔΣ), `'upsilon'`, `'ringring'`.

**Halo selection** (`select_halos` in [src/halos.py](src/halos.py)), three methods:
- `'massive'` — the most massive halos whose cumulative average mass matches
  `halo_mass_avg` (capped at `halo_mass_upper`). This is the default used by
  `stackField`/`stackMap`.
- `'binned'` — a fixed halo-mass bin.
- `'abundance'` — subhalo abundance matching (SHAM) on stellar mass to a target
  number density (`use_subhalos=True`, `halo_abundance_target`).

**Particle types** (`pType`): `gas`, `DM`, `Stars`, `BH`, `total`, `baryon`,
`ionized_gas`, and the SZ types `tSZ`, `kSZ`, `tau`. Unit conventions differ per
simulation and are normalized at load time in [loadIO.py](src/loadIO.py); see the
Unit Conventions table in [CLAUDE.md](CLAUDE.md).

## Data & paths

Simulation data is read from a **base directory** resolved in this order:

1. an explicit `sim_root=` argument to `SimulationStacker(...)`
   (the demo exposes this as `stack.data_root` in the config), then
2. the `SIMSTACK_DATA_ROOT` environment variable, then
3. the built-in NERSC default `/pscratch/sd/r/rhliu/simulations/`.

On NERSC you can leave all of these unset. Elsewhere, set one, e.g.:

```bash
export SIMSTACK_DATA_ROOT=/path/to/your/simulations
```

Beneath that root the layout is `<root>/<SimType>/<sim>/...` for raw data and
`<root>/<SimType>/products/{2D,3D}/` for cached fields/maps (`.npy`). Use
`load=True` / `save=True` on `makeField`/`makeMap` to control the cache;
`SimulationStacker` also caches in memory in `s.fields` / `s.maps`.

The three suites and their variants (feedback models, snapshots) are listed in
[CLAUDE.md](CLAUDE.md#supported-simulations). Data acquisition helpers for SIMBA
live in `get_sims/`.

## Running off NERSC

This repo grew on one NERSC account; a few things are still tied to it. The core
library now imports and runs anywhere once dependencies and the data root are set,
but be aware of the following when handing it off:

- **Data root** — set `SIMSTACK_DATA_ROOT` (or pass `sim_root=`). The old
  hardcoded `/pscratch/...` path remains only as the fallback default.
- **Gitignored files do not travel with a clone.** `.gitignore` excludes `data/`,
  `figures/`, `external/`, `Outputs_Perlmutter/`, and `*.sh` (except two lensing
  plot scripts). So the SLURM submit scripts (`scripts/runCPU*.sh`), the
  `data/*.npz` referenced by some configs via `../data/...`, and `external/`
  helpers are **not** in the repo — copy them over separately if you need them.
- **SLURM scripts** hardcode NERSC specifics: `#SBATCH -A desi/-C cpu`, a personal
  `--mail-user`, and the two `source ...cosmodesi...` lines. Templatize these for
  your scheduler/account. The demo needs no SLURM — it runs interactively.
- **Working-directory assumption** — the analysis scripts use
  `sys.path.append('../src/')` and must be run from `scripts/` (this includes
  `demo_stack.py`). The pytest suite uses a `__file__`-relative path instead and
  is CWD-independent.
- **Some peripheral scripts and notebooks** (`scripts/get_AP_sz_marvin.py`,
  `scripts/powerspectra/*`, several `tests/*.ipynb`, some `configs/*.yaml`
  `data_path:` entries) carry absolute paths to NERSC/other-machine locations and
  will need editing before they run elsewhere.
- **YAML numeric gotcha** — PyYAML parses unsigned scientific notation like
  `5e14` or `1.665e13` as a *string*, not a float. `demo_stack.py` casts such
  config values defensively; if you write your own script, do the same (or use a
  signed exponent, `5.0e+14`).

## Testing

FLAMINGO I/O and SZ-field support has a pytest suite (skips automatically if the
FLAMINGO data is not on scratch):

```bash
pytest tests/test_flamingo_io.py tests/test_flamingo_sz.py -v
```

`test_flamingo_io.py` includes a global-density test that streams ~50 GB of
particle masses (a few minutes). Most other tests are interactive Jupyter
notebooks in `tests/` (e.g. `test_3D_fields.ipynb`, `test_flamingo.ipynb`).
