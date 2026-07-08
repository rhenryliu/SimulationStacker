# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

SimulationStacker is a Python toolkit for halo stacking analysis on cosmological simulations (IllustrisTNG, SIMBA and FLAMINGO). It creates 2D/3D projected fields from particle data, applies stacking filters, and generates comparison figures between simulations and observations.

## Python Environment

This repo uses the **cosmodesi DR1** environment on NERSC, with a virtualenv layered on top:

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
```

Both lines are required. The virtualenv (`cosmodesi_dr1`) holds any extra packages installed on top of the base cosmodesi stack. Do not suggest `conda activate` or `module load python` for this project.

**Important:** if the `cosmodesi_dr1` environment is not activated, the bare `python` command resolves to the system `/usr/bin/python`, which is Python 2.7 and will fail on modern syntax (e.g. f-strings). When the environment is not active, use `python3` explicitly. Once `cosmodesi_dr1` is activated, `python` points to the correct Python 3 interpreter and this is not an issue.

## Running Scripts

Scripts are run from the `scripts/` directory with a YAML config file:
```bash
cd scripts/
python make_ratios3x2.py -p configs/ratios_3x2_z05.yaml
```

SLURM batch submission:
```bash
sbatch scripts/runCPU.sh
```

## Testing

Most tests are Jupyter notebooks in `tests/`. Validate changes interactively:
```bash
jupyter notebook tests/test_3D_fields.ipynb
```

FLAMINGO I/O and SZ-field support has a proper pytest suite (skips automatically
if the FLAMINGO data is not on scratch):
```bash
pytest tests/test_flamingo_io.py tests/test_flamingo_sz.py -v
```
`test_flamingo_io.py` includes a global-density test that streams ~50 GB of
particle masses (a few minutes). `tests/test_flamingo.ipynb` is the interactive
demo of the FLAMINGO data layout and h5py access patterns.

## Architecture

**`src/` is not an installed package** — scripts import by adding `../src/` to `sys.path`. The core library has these layers:

1. **`stacker.py`** — `SimulationStacker` class, the main entry point. Holds simulation metadata (sim name, snapshot, redshift, nPixels), orchestrates field creation and stacking, and manages caching via HDF5 files.

2. **`mapMaker.py`** — Low-level field creation from particle data. Called by `stacker.makeField()`. Supports particle types: `gas`, `dm`, `stars`, `bh`, `tSZ`, `kSZ`, `tau`, `ionized_gas`, `baryon`, `total`. Experimental types: `neutral_gas`, `tau_DM`.

3. **`filters.py`** — Stacking filter functions: `CAP` (Circular Aperture Profile, the primary method), `DSigma`/`delta_sigma` (excess surface mass density), `delta_sigma_ring`, `total_mass`, `upsilon`. These operate on 2D projected fields and return radial profiles.

4. **`loadIO.py`** — Data I/O for TNG, SIMBA and FLAMINGO group catalogs and particle data. Differences between the sims are handled here (e.g., halo radius field names and unit conventions differ). Uses `illustris_python` for TNG; reads HDF5 directly for SIMBA and FLAMINGO. For FLAMINGO, halo catalogs come from the SOAP-HBT file (`load_halos` keeps centrals only, using `SO/200_mean`; `load_subhalos` keeps all rows, using `BoundSubhalo` masses), `load_flamingo_header` normalizes the SWIFT header to TNG-style conventions, and `_convert_flamingo_particles` applies the no-h-to-h unit conversion at load time.

5. **`tools.py`** — Numba-JIT-compiled histogram binning (`hist2d_numba_seq`, `numba_tsc_3D`). Performance-critical; avoid touching unless necessary.

6. **`field_utils.py`**, **`utils.py`**, **`mask_utils.py`**, **`halos.py`** — Supporting utilities for field statistics, cosmology, masking, and halo selection. `halos.py` exposes a unified `select_halos()` dispatcher with three methods: `'binned'` (fixed mass bin), `'massive'` (highest-mass halos matching a target average mass), and `'abundance'` (SHAM using subhalo stellar mass `SubhaloMStar`, following Reddick et al. 2013).

7. **`snr.py`** — SNR and statistics utilities: `hartlap_factor`, `apply_hartlap` (Hartlap et al. 2007 bias correction for covariance inversion), `detection_snr` (sqrt of χ² against a null), `null_test_pte` (chi-squared null test with PTE). Constants `N_JK_LENS=100` and `N_BOOT_KSZ=10000` encode the jackknife/bootstrap resample counts used in the pipeline.

8. **`SZstacker.py`** — Deprecated SZ-specific stacker subclass; prefer the parent `SimulationStacker` for new work.

## Key Public API: SimulationStacker

```python
s = SimulationStacker(sim='TNG300-1', snapshot=67, nPixels=1000,
                      simType='IllustrisTNG', feedback=None, z=0.5)
```

`simType` is one of `'IllustrisTNG'`, `'SIMBA'`, `'FLAMINGO'`. `feedback` is required for SIMBA and FLAMINGO; for FLAMINGO it is the variant directory name (`'L1_m9'` = fiducial, `'fgas-8sigma'`, `'Jet_fgas-4sigma'`), e.g. `SimulationStacker(sim='L1_m9', snapshot=67, simType='FLAMINGO', feedback='fgas-8sigma', z=0.5)`.

The two main workflows are **fields** (simulation-native units, kpc/h radii) and **maps** (beam-convolved, arcmin radii):

| Method | Purpose |
|--------|---------|
| `makeField(pType, nPixels, projection, save, load, dim)` | Create raw 2D/3D projected field from particles |
| `makeMap(pType, z, projection, beamSize, pixelSize, save, load)` | Create beam-convolved 2D map |
| `stackField(pType, filterType, minRadius, maxRadius, numRadii, ...)` | Stack field and return radial profile |
| `stackMap(pType, filterType, minRadius, maxRadius, numRadii, ...)` | Stack beam-convolved map |
| `stack_on_array(array, filterType, radDistance, radDistanceUnits, ...)` | Stack any 2D array directly |
| `setField(pType, field_, nPixels, projection)` | Inject a precomputed field |

`filterType` options: `'CAP'` (default), `'cumulative'`, `'DSigma'`, `'ringring'`, `'DSigma_ring'`, `'upsilon'`, `'DSigma_mccarthy'`.

## Data Caching

Precomputed fields and maps are cached as `.npy` files under:
```
/pscratch/sd/r/rhliu/simulations/{IllustrisTNG|SIMBA|FLAMINGO}/products/{2D|3D}/
```
Filename convention: `{sim}_{snapshot}_{pType}_{nPixels}_{projection}.npy` for TNG; SIMBA and FLAMINGO insert the feedback variant: `{sim}_{feedback}_{snapshot}_{pType}_{nPixels}_{projection}.npy`

Use `load=True` / `save=True` in `makeField()`/`makeMap()` to control read/write. Results are also cached in-memory in `stacker.fields` and `stacker.maps`.

## Unit Conventions

| Quantity | IllustrisTNG | SIMBA | FLAMINGO (native) |
|----------|-------------|-------|-------------------|
| Mass | 10^10 M☉/h | M☉/h | 10^10 M☉ (**no h**) |
| Positions | ckpc/h | ckpc/h | cMpc (**no h**) |
| Halo radius field | `Group_R_Mean200` | `virial_quantities.r200c` | `SO/200_mean/SORadius` |
| DM particle mass | from `MassTable[1]` (no per-particle field) | per-particle field | per-particle field |

FLAMINGO's no-h units are converted at load time in `loadIO.py` (positions ×1000·h → ckpc/h; particle masses ×h → 10^10 M☉/h; catalog masses ×1e10·h → M☉/h), so downstream code sees the same conventions for all three suites. The FLAMINGO snapshot header is also normalized (`load_flamingo_header`): `BoxSize` in ckpc/h, cosmology under `HubbleParam`/`Omega0`/`OmegaBaryon`. Note FLAMINGO velocities are peculiar km/s (no √a factor, unlike Gadget-style TNG/SIMBA), and FLAMINGO black holes use `DynamicalMasses` (aliased to `Masses` by the loader).

SZ fields (tSZ, kSZ, τ) are computed from gas `ElectronAbundance`, `InternalEnergy`, `Density`, and `Velocities`, and use CGS constants defined in `mapMaker.py`. FLAMINGO has no `ElectronAbundance`/`InternalEnergy`; its SZ and `ionized_gas` fields instead use the precomputed `ComptonYParameters` (tSZ) and `ElectronNumberDensities` (τ/kSZ/ionized gas electron counts) — see `mapMaker._flamingo_sz_weights`. Both fields are zero for star-forming particles.

## Configuration System

All analysis scripts read YAML configs. Key config sections:

```yaml
stack:
  redshift: 0.5
  projection: 'yz'              # 'xy', 'xz', or 'yz'
  load_field: true               # Load cached HDF5 field
  save_field: true               # Save computed field to HDF5
  particle_type: 'gas'
  n_pixels: 1000
  halo_mass_min: 1.665e13        # Msun
  filter_type: 'CAP'             # 'CAP', 'cumulative', 'DSigma', 'DSigma_ring', etc.
  filter_type_2: 'CAP'           # Filter for denominator ptype in ratio plots
  dim: '2D'                      # '2D' (beam-convolved map) or '3D' (spherical stacking)
  subtract_mean: false           # Subtract field mean before stacking
  halo_mass_avg: 1.665e13        # Target average mass for 'massive' halo selection (Msun)
  halo_mass_upper: 5.0e14        # Upper mass bound for 'massive' selection (Msun)
  halo_abundance_target: 5.0e-4  # Target number density for SHAM selection (cMpc/h)^{-3}
plot:
  fig_path: '../figures/'
  fig_name: 'output_name'
  extra_sim_ratios: []           # Additional numerator ptypes to overlay in ratio plots
simulations:
  - sim_type: 'IllustrisTNG'
    sims:
      - name: 'TNG300-1'
        snapshot: 67
  - sim_type: 'SIMBA'
    sims:
      - name: 'm100n1024'
        snapshot: 125
        feedback: 's50'
```

## Supported Simulations

- **IllustrisTNG**: `TNG300-1`, `TNG300-2`, `TNG100-1`, `TNG100-2`, `TNG50-1`, `Illustris-1`, `Illustris-2`
- **SIMBA**: `m50n512`, `m100n1024` with feedback variants `s50`, `s50nox`, `s50noagn`, `s50nofb`, `s50nojet`
- **FLAMINGO**: `L1_m9` (1 cGpc box) with feedback variants `L1_m9` (fiducial), `fgas-8sigma`, `Jet_fgas-4sigma`; only snapshot 67 (z=0.5) is downloaded. Snapshots are SWIFT virtual HDF5 files (64 chunk files each + halo-membership files); halo catalogs are SOAP-HBT. Read with plain h5py — **swiftsimio does not work in the cosmodesi environment** (needs NumPy ≥ 2.0).

Simulation data lives at hardcoded paths in `stacker.py` under `/pscratch/sd/r/rhliu/simulations/` — this is known technical debt, do not refactor without being asked.

## Pre-push Code Review Hook

`.claude/hooks/pre-push` runs an autonomous Claude code review before pushes. It is opt-in:
```bash
REVIEW=1 git push
```
Without `REVIEW=1`, the hook exits immediately without reviewing.
