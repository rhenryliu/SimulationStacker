# Implementation Plan: Task 1 r-profile computation

Companion to `docs/cross_correlation_notes.md` (theory) and
`docs/r_profiles_task1_spec.md` (engineering spec). This plan maps the spec
onto the repo as it actually exists at commit `f56aba8`. Nothing here is
implemented yet; no existing file has been modified.

Status of every claim below: **read from source in this session**; nothing has
been executed except the arcmin/pixel-scale calculations in Section 4 (run in
the cosmodesi environment with each simulation's header cosmology).

---

## 1. File-by-file change list

### 1.1 New module: `src/rprofiles.py`

Single new library module; **no changes to any existing `src/` file are
required** (Section 2 shows every reuse point resolves to an existing public
function). Theory-note Phase 0 item 2 ("extend SimulationStacker to output
CDM-only and total-baryon maps") is already satisfied: `makeField('DM')` and
`makeField('baryon')` exist today.

Proposed public API (Google-style stubs; all maps are square periodic 2D
arrays, all radii in arcmin):

```python
"""r-profile computation via periodic FFT aperture filtering.

Implements the Task 1 estimator of docs/r_profiles_task1_spec.md:
Y_XY(R; F) = < F_R[delta_X] * delta_Y >_map on periodic projected maps,
with kernels that reproduce filters.delta_sigma_kernel's pixel-count
normalization exactly, spatial 4x4 block jackknife, and analytic
self-pair subtraction for the galaxy auto-correlation.
"""

FILTERS = ('Sigma', 'DSigma', 'Upsilon')   # Upsilon derived from DSigma
APERTURES_ARCMIN = np.linspace(1.0, 6.0, 9)
DR_ARCMIN = 0.75
R0_ARCMIN = 1.0
N_JK_SIDE = 4                              # 4x4 = 16 leave-one-out patches


def build_aperture_kernel(n_pixels: int, pixel_arcmin: float, R: float,
                          filter_type: str, dr: float = DR_ARCMIN) -> np.ndarray:
    """Build a periodic aperture kernel on the full map grid.

    Kernel lives at lag (0, 0) with periodic wrap (np.fft convention).
    Pixel membership is by pixel-centre radius r = pixel_arcmin *
    hypot(i, j) with i, j integer lags, matching the radial grids used by
    filters.delta_sigma_kernel on stamp cutouts. Normalization is the
    pixel-count convention of delta_sigma_kernel:
      'DSigma': +1/(pixArea*N_disk) on r < R, -1/(pixArea*N_ann) on
                R <= r < R+dr (compensated; sums to exactly 0).
      'Sigma' : +1/(pixArea*N_ann) on R <= r < R+dr (annulus mean,
                positive, uncompensated).

    Raises:
        ValueError: If the disk or annulus contains zero pixels
            (resolution guard; never degrade silently).
    """

def filtered_map(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return F_R[field] via periodic FFT convolution (rfft2/irfft2)."""

def to_overdensity(field: np.ndarray) -> np.ndarray:
    """Return field/mean(field) - 1 with a float64 mean (spec Sec. 'Fixed
    numerical conventions')."""

def block_means(product_map: np.ndarray, n_side: int = N_JK_SIDE) -> np.ndarray:
    """Per-block sums and counts of F[X]*Y, reduced to the 16 block means
    needed for leave-one-out jackknife. Returns shape (n_side**2,)."""

def jackknife_realizations(block_sums, block_counts) -> np.ndarray:
    """Leave-one-out means: Y_i = (total - block_i)/(N - n_i), shape (16,)."""

def compute_Y_matrix(deltas: dict[str, np.ndarray], pixel_arcmin: float,
                     radii: np.ndarray = APERTURES_ARCMIN,
                     dr: float = DR_ARCMIN, r0: float = R0_ARCMIN,
                     nbar_pix: float | None = None,
                     galaxy_key: str = 'g') -> dict:
    """Compute all filtered cross-amplitudes and their jackknife blocks.

    For every unordered field pair (X, Y) in deltas, every filter in
    {Sigma, DSigma}, and every aperture radius: convolve X with the
    kernel, multiply pixelwise by Y, record the mean and the 16
    leave-one-out block means. Upsilon amplitudes are formed as the
    linear combination Y_ds(R) - (r0/R)**2 * Y_ds(r0) at the Y level
    (per jackknife realization), never by extra convolutions.

    The galaxy auto (galaxy_key, galaxy_key) receives analytic self-pair
    subtraction: K(0) * pixArea / nbar_pix per aperture (spec, verified
    numerically there), applied to the mean and each jackknife block.

    Returns:
        dict with 'radii', 'filters', and per-pair arrays
        Y[(X, Y)][filter] of shape (n_radii,) plus
        Y_jk[(X, Y)][filter] of shape (16, n_radii).
    """

def r_profiles(Ymat: dict, pairs=(('g','b'), ('b','m'), ('g','e'), ('e','m'))
               ) -> dict:
    """Form r_XY = Y_XY / sqrt(Y_XX * Y_YY) and r_bm/r_gb.

    CRITICAL (spec): every r and the ratio r_bm/r_gb are formed PER
    jackknife realization; errors are the jackknife spread of the r's,
    never Gaussian propagation of marginal Y errors.

    Returns:
        dict: r[pair][filter] (n_radii,), r_err[pair][filter],
        ratio['bm_over_gb'][filter] and its error.
    """

def make_galaxy_field(stacker, projection: str, n_pixels: int,
                      target_number: float,
                      parent_mass_upper: float | None = 5e14
                      ) -> tuple[np.ndarray, np.ndarray]:
    """NGP-deposit the SHAM-selected subhalo sample onto the field grid.

    Replicates stack_on_array's use_subhalos=True selection exactly
    (stacker.py:607-621): loadSubHalos(), optional parent-FoF-mass
    pre-filter via SubhaloGrNr against loadHalos()['GroupMass'] <=
    parent_mass_upper, then select_halos(SubhaloMStar, 'abundance',
    target_number=..., Lbox=header['BoxSize']). Deposition is NGP via
    tools.hist2d_numba_seq with bins=[n_pixels]*2 and
    ranges=[[0, BoxSize]]*2, identical to the particle-map pixel edges
    (binned_statistic_2d convention: pixel i covers
    [i, i+1)*BoxSize/n_pixels). No smoothing window (spec: NGP, not TSC).

    Returns:
        (count_map, halo_mask): float64 galaxy count map and the integer
        index array of selected subhalos (for the integration test and
        for reproducibility metadata).
    """
```

Internal efficiency layout (not API): precompute `rfft2` of each of the 4
overdensity fields once; per kernel (9 radii x 2 base filters = 18) do one
`irfft2` per field (72 total); accumulate per-pair block sums streaming so no
more than one filtered map is alive at a time.

### 1.2 Modifications to existing files

**None.** `src/`, `scripts/` existing files, and `tests/` existing files are
untouched. The only repo additions are the new files listed here.

### 1.3 New scripts and configs

Following the paper-folder convention (`scripts/<paper>/` mirrored by
`scripts/configs/<paper>/`), a new paper folder `cross_corr` (name is a
proposal — see Open Question O1):

| Path | Purpose |
|---|---|
| `scripts/cross_corr/make_r_profiles.py` | Compute one (sim, sample): build/load the 4 fields per projection via `stacker.makeField(..., save=True, load=True)`, build the galaxy map, run `compute_Y_matrix` + `r_profiles`, write one `.npz` per (sim, sample, projection) to `../data/r_profiles/`. CLI: `-p <yaml> [--sim NAME] [--feedback FB]` so SLURM jobs shard by simulation (pattern: `abundance_variation_ratio.py` npz-cache convention). |
| `scripts/cross_corr/plot_r_profiles.py` | Read the npz caches, produce the Singh et al. (2020) Fig. 1 analogue (one panel per filter, one curve per simulation, within-projection jackknife bands + across-projection scatter) and print/save the Gate A metrics: `max_R |r-1|` per filter, per-radius cross-simulation scatter of `r_bm/r_gb`. |
| `scripts/cross_corr/runCPU_rprofiles.sh` | `regular`-QOS batch runner, one node, submitted from `scripts/` (`sbatch cross_corr/runCPU_rprofiles.sh`), log to `../Outputs_Perlmutter/`, cosmodesi env sourcing copied from `unbound_gas/runCPU.sh`. A `debug`-QOS smoke variant with SIMBA (smallest full-production box) first. |
| `scripts/configs/cross_corr/r_profiles_z05.yaml` | LRG-like sample: `redshift: 0.5`, `halo_abundance_target: 5.0e-4`, sims TNG300-1/67, Illustris-1/103, m100n1024-s50/125, FLAMINGO L1_m9 x {L1_m9, fgas-8sigma, Jet_fgas-4sigma}/67. Per-sim `n_pixels` key inside each `sims:` entry (resolution differs per box; Section 4). |
| `scripts/configs/cross_corr/r_profiles_z026.yaml` | BGS-like sample: `redshift: 0.26`, `halo_abundance_target: 1.0e-3`, sims TNG300-1/80, Illustris-1/116, m100n1024-s50/136. No FLAMINGO (snapshot 71 not downloaded; spec). |

Config schema follows the existing `stack:`/`plot:`/`simulations:` split; new
keys under `stack:`: `projections: ['xy','xz','yz']`, `min_radius: 1.0`,
`max_radius: 6.0`, `num_radii: 9`, `dr_arcmin: 0.75`, `r0_arcmin: 1.0`,
`halo_abundance_target`, `parent_mass_upper: 5.0e14`; under `plot:`:
`npz_path: '../data/r_profiles/'`, `fig_path`, `fig_name`.

### 1.4 New tests

| Path | Contents |
|---|---|
| `tests/test_rprofiles.py` | The four synthetic acceptance tests (Section 3). No simulation data; runs anywhere the cosmodesi env runs. |
| `tests/test_rprofiles_integration.py` | The NERSC Y_gb cross-check vs the existing stamp route (Section 3.5), gated with the `test_flamingo_io.py` skip pattern. |

---

## 2. Reuse points: what the spec names vs what the repo has

| Spec names | Found in repo | How it will be called / notes |
|---|---|---|
| `delta_sigma_kernel` normalization | `filters.delta_sigma_kernel(mass_grid, r_grid, r, dr=0.5, pixel_size=1)` at `src/filters.py:155`. Disk `r_grid < r` gets `+1/(pixArea*N_disk)`; annulus `r <= r_grid < r+dr` gets `-1/(pixArea*N_ann)`; returns `sum(mass*kernel)`. | `build_aperture_kernel` reproduces these masks/weights on integer-pixel-lag radii. Note the **default `dr=0.5` differs from the pipeline value**: `stack_on_array` passes `dr=0.75` (arcmin) explicitly (`stacker.py:724`); the new module hardcodes 0.75 per spec. Used directly (not reimplemented) as the reference in acceptance test 1. |
| `upsilon` | `filters.upsilon(mass_grid, r_grid, r, r0=1.0, dr=0.5, pixel_size=1)` at `src/filters.py:211` = `delta_sigma_kernel(R) - (r0/r)**2 * delta_sigma_kernel(r0)`. | Reproduced as the same linear combination at the Y level. Reference implementation for test 2. |
| `halos.select_halos` | `select_halos(halo_masses, method, **kwargs)` at `src/halos.py:113`; `'abundance'` route needs `target_number` ((cMpc/h)^-3) and `Lbox` (ckpc/h), ranks by the passed array (we pass `SubhaloMStar`), returns integer indices. Signature is simType-independent. | Called inside `make_galaxy_field` exactly as `stack_on_array` does, **including the parent-FoF-mass pre-filter** (`GroupMass[SubhaloGrNr] <= 5e14`) that `stack_on_array` applies before abundance matching whenever `halo_mass_upper` is not None (`stacker.py:608-617`; `stackMap` default is `5e14`, and `scripts/lensing/compare_data_ratio.py` confirms the f_gas figures use that default). Reproducing it is required for the integration test's "identical SHAM sample". |
| `loadSubHalos` | `stacker.loadSubHalos()` -> `loadIO.load_subhalos(sim_path, snapshot, sim_type, sim_name, header)`. Uniform keys across all three simTypes: `SubhaloPos` (ckpc/h, (N,3)), `SubhaloMStar` (Msun/h), `SubhaloMass`, `SubhaloGrNr` (index into the `load_halos` catalogue; for FLAMINGO already remapped to centrals-only rank, `loadIO.py:298-304`). | Positions -> pixel via the same axis slices as `stack_on_array` (`xy`->:2, `xz`->[0,2], `yz`->1:) and NGP histogram on `[0, BoxSize)`. |
| `tools.hist2d_numba_seq` | `hist2d_numba_seq(tracks, bins, ranges, weights=np.empty(0), dtype=np.float32)` at `src/tools.py:18`; `tracks` shape (2, N); returns float64 (bins[0], bins[1]). njit: pass `bins`/`ranges` as arrays, not lists. Empty `weights` -> unit counts. | Galaxy NGP deposition: `hist2d_numba_seq(np.array([x, y]), np.array([n, n]), np.array([[0., L], [0., L]]))`. Sample sizes are tiny (4k–160k galaxies) so `np.histogram2d` would also do; spec says reuse this, so we do. |
| "existing arcmin/comoving conversion (f_gas Eq. 27)" | `utils.comoving_to_arcmin(L_com_kpch, z, cosmo=Planck18)` and `utils.arcmin_to_comoving(theta_arcmin, z, cosmo=Planck18)` at `src/utils.py:232/256`. | `pixel_arcmin = comoving_to_arcmin(BoxSize, z, cosmo=stacker.cosmo) / n_pixels`. **Must pass `cosmo=stacker.cosmo` explicitly** — the Planck18 default is not the simulation cosmology; `stack_on_array` builds the sim cosmology from the header (`stacker.py:643-647`) and we match that. |
| `snr.py` statistics helpers | `hartlap_factor(n_resample, n_bins)`, `apply_hartlap(cov, n_resample)`, `detection_snr`, `null_test_pte` (`src/snr.py`). | Marginal reuse only: the Task 1 deliverable is per-radius jackknife spreads, no covariance inversion. If `plot_r_profiles.py` ever quotes a chi^2 of r vs 1, it applies `hartlap_factor(16, n_bins)` — note 16 jackknife patches with 9 bins makes an invertible-covariance statement fragile (16 <= 9+2 fails `hartlap_factor`'s validity check for the full 9-bin vector); per-radius errors are the deliverable, as the spec says. |
| `stack_on_array` (validation hook) | `stacker.stack_on_array(array, filterType='DSigma', minRadius, maxRadius, numRadii, projection, radDistance=1.0, radDistanceUnits='arcmin', z, pixelSize, use_subhalos, halo_abundance_target, halo_mask=None)` at `src/stacker.py:556`. Accepts a precomputed `halo_mask` (skips selection) — this is the clean hook for stacking the identical sample. Centres cutouts at `np.round(pos_ckpch / (BoxSize/nPixels))` (integer pixel, nearest-edge convention) and builds stamp radial grids by `linspace` over the rounded cutout half-width — both are the sub-pixel conventions the spec expects to show up in the integration-test residuals. | Integration test calls `stack_on_array` directly on the raw tau field (avoiding `stackMap`'s tau->microK conversion at `stacker.py:439-445`), with `halo_mask=` the indices returned by `make_galaxy_field`, `radDistanceUnits='arcmin'`, `pixelSize` = true arcmin/pixel. |
| `'dm'`, `'ionized_gas'`, `'baryon'` pTypes | `mapMaker.create_field` dispatch (`mapMaker.py:310`): mass types go to `make_mass_field`, `'total'`/`'baryon'` to `make_combined_field`, SZ to `make_sz_field`. **The CDM pType is `'DM'`, not `'dm'`** (cache filenames use `DM`; spec's lowercase name does not exist — conflict C1). `'ionized_gas'` exists for all three simTypes: TNG/SIMBA electron count from `ElectronAbundance` (`mapMaker.py:710-715`), FLAMINGO from `ElectronNumberDensities * V_phys` (`mapMaker.py:698-708`, zero for star-forming particles — documented bias). `'baryon'` = gas + Stars + BH for **all three** simTypes (`make_combined_field`, `mapMaker.py:774-779`, simType-agnostic; SIMBA maps Stars/BH to PartType4/5 at `loadIO.py:391-394`, FLAMINGO at `loadIO.py:405-409` with BH `DynamicalMasses` aliased to `Masses`). So yes — stars and BH are included everywhere; note the theory note defines b as gas+stars (no BH) — conflict C5 (negligible mass, but must be stated). |
| 3D cross machinery (optional add-on) | `stacker._compute_pk_3D(field, field_b=None, BoxSize, grid, threads)` and `_compute_corr_3D(...)` exist (`stacker.py:1028/1064`) and support cross-spectra via `PKL.XPk`. Requires Pylians (`HAS_PK_LIBRARY` guard). | Out of core scope; a later `make_r3d.py` can call them unchanged. Note `field_utils.py` (`calc_power`, `make_cross_corr*`) is MTNG-legacy code requiring abacusutils/nbodykit — do **not** route through it. |
| Field caching | `makeField(pType, nPixels, projection, save, load)` -> `loadIO.save_data/load_data`, filenames keyed by nPixels, so new resolutions coexist with the existing 1000-pixel caches; `'baryon'` reuses per-component caches at the same nPixels (`make_combined_field` load path). | Production script uses `save=True, load=True` throughout. |

Does **not** exist as assumed / notable gaps:

- No existing map-level FFT-filter operation anywhere in the repo (the
  `fft_smoothed_map` beam code in `utils.py` is Gaussian-only). The kernel
  convolution machinery is genuinely new code in `rprofiles.py`.
- No existing jackknife utility (lensing jackknife lives in the external
  dsigma pipeline; `snr.py` only consumes covariances). Written fresh.
- `field_utils.py` does not provide arcmin conversions (it is unrelated MTNG
  legacy); everything needed is in `utils.py`.

---

## 3. Test plan

### 3.1–3.4 `tests/test_rprofiles.py` (synthetic, no simulation data)

Header pattern: `sys.path.insert(0, .../src)` as in `test_flamingo_io.py`.
Fixed RNG seeds. Grid: 512x512, `pixel_arcmin` chosen so apertures span
5–40 pixels. Test map: exp(Gaussian random field) ("lognormal") minus mean.

1. **Kernel vs stamp** (`test_kernel_matches_stamp`): for ~20 random integer
   centres, compare `filtered_map(field, K)[cx, cy]` against
   `filters.delta_sigma_kernel(cutout, r_grid, R, dr=0.75, pixel_size=pix)`
   where `cutout = SimulationStacker.cutout_2d_periodic(field, (cx, cy), L)`
   and `r_grid` is built from **exact integer pixel lags** times
   `pixel_arcmin` (not `radial_distance_grid`'s linspace, which rounds the
   half-width — that rounding is precisely the integration-test residual and
   must not contaminate this unit test). Assert max abs diff < 1e-12 for
   both DSigma and Sigma kernels at 3 radii. Also assert
   `build_aperture_kernel` raises `ValueError` when `R < pixel_arcmin`
   (empty disk).
2. **Convolution == stamp stack at galaxy pixels**
   (`test_correlate_equals_stamp_mean`): deposit N=500 random galaxies at
   integer pixels (NGP), form `delta_g`; assert
   `mean(F[delta_X] * delta_g) == mean_g(F[delta_X](x_g)) - mean_map(F[delta_X])`
   to float precision (~1e-13 relative), for DSigma and Upsilon. The
   `mean_map` term is exactly 0 for the compensated kernels (kernel sums to
   zero identically) — assert that too.
3. **Poisson Y_gg null** (`test_poisson_ygg_null`): unclustered Poisson
   galaxies (nbar_pix ~ 0.01), compute Y_gg with self-pair subtraction
   `K(0) * pixArea / nbar_pix`; assert |Y_gg| < 3 sigma_jk at two apertures
   (one small, one large), for DSigma. Repeat over ~5 seeds to keep the
   flake rate down while honouring the 3-sigma criterion.
4. **Known mixing recovers r** (`test_known_mixing_r`): `f1` = GRF,
   `f2 = a*f1 + noise` with known a and noise amplitude; compute
   `r_12(R)` per filter; compare against the analytic
   `r = a*sqrt(Y_11) / sqrt(a^2*Y_11 + Y_nn)` evaluated from the realized
   `Y_11`, `Y_nn` (same kernels, so exact up to the f1–noise cross term);
   assert agreement within the jackknife errors (and that the jackknife
   r-spread is finite and sensible).

### 3.5 `tests/test_rprofiles_integration.py` (NERSC data, marked/skipped)

Gate (module-level `pytestmark`, following `test_flamingo_io.py:45-47`):

```python
TNG_BASE = '/pscratch/sd/r/rhliu/simulations/IllustrisTNG/TNG300-1/'
pytestmark = pytest.mark.skipif(
    not os.path.isdir(TNG_BASE), reason='TNG300-1 data not available')
```

plus an in-test `pytest.skip` if the cached tau field at the production
nPixels is absent (`loadData` raising `ValueError`) — the test must never
trigger an hours-long field computation.

Content: TNG300-1 snapshot 67, z=0.5, one projection ('yz', matching the
f_gas configs). Load the cached tau 2D field at production nPixels; build the
SHAM sample once via `make_galaxy_field` (returning `halo_mask`); route A =
`rprofiles.compute_Y_matrix` on `to_overdensity(tau)` x `delta_g`; route B =
`stacker.stack_on_array(tau_field, filterType='DSigma', minRadius=1,
maxRadius=6, numRadii=9, radDistance=1.0, radDistanceUnits='arcmin',
pixelSize=true_arcmin_per_pixel, halo_mask=halo_mask)` mean over halos,
converted to the overdensity convention (divide by `mean(tau)`; subtract the
map-mean term, which is zero for DSigma). Assert relative agreement < 1e-2
per bin; **print** the per-bin residuals (they document the
integer-pixel-centring and cutout-rounding conventions; spec: documented,
not hidden). One extra assertion: Upsilon route A equals the
`filters.upsilon` stamp combination to the same tolerance.

Run commands:

```bash
cd tests/
pytest test_rprofiles.py -v                    # anywhere, seconds
pytest test_rprofiles_integration.py -v        # NERSC login node, needs cached tau field
```

---

## 4. Resolution and memory budget

Pixel scales computed this session with each simulation's own header
cosmology (`FlatLambdaCDM(H0=100h, Om0=Omega0)`), via
`utils.comoving_to_arcmin`:

| Sim (sample) | Box | 1' in Mpc/h | Proposed nPixels | pixel (ckpc/h) | pixel (arcmin) | px per 1' aperture |
|---|---|---|---|---|---|---|
| TNG300-1 (z=0.5) | 205 Mpc/h | 0.383 | **2048** | 100 | 0.261 | 3.8 |
| TNG300-1 (z=0.5, convergence) | | | **4096** | 50 | 0.131 | 7.7 |
| TNG300-1 (z=0.26) | | 0.213 | **4096** | 50 | 0.235 | 4.2 |
| Illustris-1 (z=0.5) | 75 Mpc/h | 0.388 | **2048** | 37 | 0.094 | 10.6 |
| Illustris-1 (z=0.26) | | 0.214 | **2048** | 37 | 0.171 | 5.8 |
| SIMBA m100n1024 (z=0.5) | 100 Mpc/h | 0.385 | **2048** | 49 | 0.127 | 7.9 |
| SIMBA m100n1024 (z=0.26) | | 0.213 | **2048** | 49 | 0.229 | 4.4 |
| FLAMINGO L1_m9 x3 (z=0.5) | 681 Mpc/h | 0.384 | **5000** (spec) | 136 | 0.355 | 2.8 |
| FLAMINGO (proposed convergence) | | | **8192** | 83 | 0.217 | 4.6 |

Two flags: (i) TNG300 at z=0.26 needs 4096, not 2048 — at 2048 the pixel is
0.471' and the 1' aperture holds only ~2 pixels across (the spec's ">= 2048"
was calibrated at z=0.5). (ii) FLAMINGO at the spec's ~5000 is the coarsest
configuration in the sweep (2.8 px per smallest aperture, annulus dr=0.75' ~
2 px wide); I propose adding a FLAMINGO 5000-vs-8192 convergence check
alongside the spec's TNG300 2048-vs-4096 one (Open Question O3).

Memory (2D float64 maps): 2048^2 = 32 MB, 4096^2 = 128 MB, 5000^2 = 200 MB,
8192^2 = 512 MB. Peak resident in `compute_Y_matrix`: 4 field maps + 4 rfft2
buffers (~half size, complex128) + 1 filtered map + 1 product buffer, i.e.
< 10 map-equivalents: **< 2 GB at 5000, < 5 GB at 8192** — trivial against a
512 GB CPU node; also fine on a login node *for the FFT stage only* (field
creation is a job, Section 6).

FFT cost per (sim, sample, projection): 4 forward rfft2 + 72 irfft2 (18
kernels x 4 fields; Upsilon adds none). At 5000^2 an rfft2 is O(seconds)
multithreaded: **minutes end-to-end**; at 8192^2 still < ~30 min. Negligible
next to field creation.

Field creation (the real cost; per projection, per sim, 5 particle sweeps —
DM, gas-mass, ionized_gas, Stars, BH; 'baryon' assembles from cached
components at the same nPixels):

| Sim | particles/sweep (order) | I/O per full sweep | estimate per projection |
|---|---|---|---|
| TNG300-1 | ~1.6e10 (DM), ~1.5e10 (gas) | O(0.5 TB) coords+masses | few hours (all 5 sweeps) |
| Illustris-1 | ~1.8e9 x2 | O(60 GB) | < 1 h |
| SIMBA m100n1024 | ~1e9 x2 | O(40 GB) | < 1 h |
| FLAMINGO L1_m9 (each variant) | 5.4e9 gas, 5.8e9 DM | O(300 GB) | few hours |

These are I/O-volume estimates, **not measured**; the existing product cache
proves TNG300/FLAMINGO full-box 2D sweeps complete inside normal job walltimes
(they were run at nPixels=1000/3548 before; `binned_statistic_2d` cost is
resolution-independent). All 3 projections triple the sweep cost since
`makeField` does one projection per call; accepted as-is rather than touching
`mapMaker` (minimal-diff rule). Budget: one `regular`-QOS single-CPU-node job
per (sim, sample), `--time` 4–8 h for TNG300/FLAMINGO, 2 h for
Illustris/SIMBA; total 9 (sim, sample) combinations (6 at z=0.5, 3 at
z=0.26). Disk for new caches: < 10 GB total on scratch (5000^2 x 5 types x 3
projections x 3 FLAMINGO variants = 9 GB is the largest block; 8192
convergence adds ~1.5 GB for one variant).

---

## 5. Open questions and conflicts (not silently resolved)

**C1 — pType name.** Spec says CDM = pType `'dm'`; the repo pType is `'DM'`
(dispatch and cache filenames). Plan uses `'DM'`. Flagging because the spec
said to.

**C2 — Sigma filter definition.** Theory note Sec. 2.2 defines Sigma(R) as
the azimuthally averaged surface density *at* R (and Sec. 6/Task 2 discusses
disk means); the spec fixes Sigma = **annulus mean over [R, R+0.75']**,
positive, uncompensated. Spec is authoritative for implementation; adopted.
The Fig. 1 morphology comparison to Singh's Sigma-based coefficient should
carry a caption note that our Sigma is a local-annulus estimate, not
Sigma-bar(<R).

**C3 — "9 linear bins" vs point radii.** The existing pipeline evaluates
filters at the 9 points `np.linspace(1, 6, 9)` (min_radius/max_radius/
num_radii = 1/6/9 in the f_gas configs); the theory note Sec. 5.3 asks for
kernel bin-averaging over aperture bins "unless a Singh-style test shows it
negligible". Plan follows the pipeline (point radii = the 9 linspace values),
because the spec pins "matching the existing pipeline"; the bin-averaging
question belongs to Task 4 (theory transfer), not Task 1. Flagged so it is a
decision, not an oversight.

**C4 — SHAM parent-mass pre-filter.** The spec's galaxy-sample definition
("method='abundance', target 5e-4 / 1e-3") does not mention the parent-mass
cut, but the existing pipeline's `use_subhalos=True` path applies
`GroupMass[parent] <= 5e14` **before** abundance matching whenever
`halo_mass_upper` is set — and `stackMap`'s default (used by the f_gas
figures) sets it to 5e14. To make the integration test's "identical SHAM
sample" true and to match the f_gas paper samples, `make_galaxy_field`
defaults to `parent_mass_upper=5e14` (config-overridable, `null` to
disable). **Assumption to confirm with the user.**

**C5 — 'baryon' includes BH.** Theory note defines b = ionized + neutral gas
+ stars; repo `'baryon'` = gas + Stars + **BH** (all simTypes; FLAMINGO BH
via DynamicalMasses). BH mass fraction is negligible for these samples, but
the field definition frozen at Gate B should say so explicitly. Plan uses
repo `'baryon'` unchanged.

**C6 — FLAMINGO electron fields zero for star-forming gas.** Carried into
`'ionized_gas'` and any tau-based validation for FLAMINGO (documented bias in
`mapMaker`); affects e-fields only, not b/m. No action for Task 1 beyond a
metadata note in the npz.

**C7 — resolution at z=0.26 and FLAMINGO** (Section 4): TNG300 z=0.26 needs
4096 (spec's ">=2048" is insufficient there); FLAMINGO 5000 is marginal and
gets a proposed convergence check. Spec's numbers amended rather than
followed blindly — flagged per the spec's own instruction.

**O1 — folder name.** New paper folder `scripts/cross_corr/` +
`scripts/configs/cross_corr/` proposed (third paper alongside `lensing/`,
`unbound_gas/`). Trivial to rename before anything lands.

**O2 — Illustris-1 z quirk.** Illustris-1 snapshot 116 is the z~=0.26 config
convention already in use (`ratios_3x2_z026.yaml`); reused as-is. The
`_Z_MATCH_TOL` warning in `SimulationStacker.__init__` will police any
mismatch at runtime.

**O3 — FLAMINGO convergence run.** Add 5000-vs-8192 on the fiducial L1_m9
(one projection) to the spec's TNG300 2048-vs-4096 check? Costs one extra
FLAMINGO gas+DM+Stars+BH sweep (~1.5 GB cache, a few node-hours). Recommended;
awaiting user confirmation.

**O4 — theory-note Eq. 4 footnote.** The note itself flags that its source
discussion had `Y_gb/(Y_mm*Y_gg)` without the square root; Eq. 4 (with sqrt)
is what the r-definitions imply and what `r_profiles` implements. No code
impact for Task 1 (we only produce r's), but recorded since the two documents
are the chain of authority.

---

## 6. Implementation order

1. **`src/rprofiles.py`** — kernels, convolution, Y matrix, self-pair
   subtraction, jackknife, r's, `make_galaxy_field`. Docstrings per repo
   convention (Google style, type hints, Canadian English).
2. **`tests/test_rprofiles.py`** — the four synthetic acceptance tests.
   Run on a login node (`cd tests/ && pytest test_rprofiles.py -v`); these
   are small-grid FFTs, safely within login-node limits.
3. **`tests/test_rprofiles_integration.py`** — written now, expected to skip
   until the TNG300 cached fields exist at production nPixels.
4. ⛔ **CHECKPOINT — pytest suite green.** Show the pytest output; commit the
   module + tests as one unit (code-reviewer subagent on `git diff HEAD`
   first, per CLAUDE.md). **No NERSC job is submitted before this point.**
5. **`scripts/cross_corr/make_r_profiles.py` + configs + runner.** Login-node
   syntax/import check (`python -m py_compile`, `--help`), then a
   `debug`-QOS smoke run on SIMBA m100n1024 z=0.5 (smallest production sweep)
   with one projection.
6. **Field precompute + Task 1 on TNG300-1 alone** (theory note Phase 1 step
   4): one `regular`-QOS job (nPixels 2048, then the 4096 convergence
   repeat, one projection). Run the integration test once the tau cache
   exists. Compare 2048-vs-4096 r's; freeze plot conventions with
   `plot_r_profiles.py`.
7. **Full sweep** — remaining 8 (sim, sample) combos, 3 projections each,
   ≤ 4 queued jobs at a time (CLAUDE.md), job IDs reported; then the Fig. 1
   analogue + Gate A scatter metrics.

---

## Addendum: implementation notes (what changed against this plan)

Written during implementation on branch `task1-r-profiles`. The plan above is
left as approved; every deviation from it is recorded here.

### A1. No particle sweeps are needed — the CDM map is derived by subtraction

The plan budgeted node-hours for DM field creation. In fact `'total'` and
`'baryon'` are already cached for every production run, and
`mapMaker.make_combined_field` builds them from the same component maps on the
same grid (`'total'` = gas+DM+stars+BH, `'baryon'` = gas+stars+BH), so

    CDM = total - baryon

exactly, up to float64 round-off. Verified on TNG300-1 snapshot 67 at
2674^2: the derived CDM mass fraction is 0.842668 against the box cosmology
`1 - Ob/Om` = 0.842668 (six digits), with zero negative pixels. Implemented as
`rprofiles.derive_cdm_field`, which re-runs that check for every simulation
and reports it. The check is a warning, not an error, because FLAMINGO's
`Omega0` includes a neutrino contribution absent from the particle maps.

Consequence: **Task 1 in the `yz` projection needs no particle sweep at all.**

### A2. Resolution comes from the existing caches, not the plan's grid sizes

Section 4 proposed 2048/4096/5000. Every production run already has a cached
`yz` field at ~0.2 arcmin/pixel, which is finer than the spec's minimum
everywhere and costs nothing:

| Run | snapshot (z) | nPixels | arcmin/pixel | px per 1' aperture |
|---|---|---|---|---|
| TNG300-1 | 67 (0.5030) / 80 (0.2613) | 2674 / 4822 | 0.200 / 0.200 | 5.0 |
| Illustris-1 | 103 (0.5030) / 116 (0.2613) | 966 / 1752 | 0.200 / 0.200 | 5.0 |
| SIMBA m100n1024 s50 | 125 (0.4904) / 136 (0.2668) | 1301 / 2349 | 0.200 / 0.198 | 5.0 |
| FLAMINGO L1_m9 x3 | 67 (0.5000) / 71 (0.3000) | 8869 / 14015 | 0.200 / 0.201 | 5.0 |

This supersedes conflict C7: the spec's ">= 2048 / ~5000" concern and the
plan's z=0.26 amendment are both moot, since every configuration resolves the
smallest aperture with five pixels.

### A3. FLAMINGO snapshot 71 is available — the BGS sweep is in scope

The spec states that only snapshot 67 is downloaded and that the BGS-like
sweep is out of scope. That is no longer true: snapshot 71 (z = 0.3000) is
present for all three variants with SOAP-HBT catalogues and cached projected
fields. z = 0.30 is exactly the FLAMINGO BGS redshift the theory note asks
for (Sec. 6), so `r_profiles_z026.yaml` includes all six runs.

### A4. Only `yz` is cached; `xy`/`xz` remain unrun

The spec asks for three projections. Only `yz` exists on disk, and `xy`/`xz`
would each need a full particle sweep per field type per simulation (order
days of node-hours across the twelve simulation-samples, dominated by
TNG300-1 and the three FLAMINGO variants). Gate A is a cross-*code* test,
which `yz` alone delivers; the across-projection scatter is the secondary
diagnostic. **Not run, pending approval.**

### A5. FFTs go through `scipy.fft` with `workers=-1`

`numpy.fft` is single-threaded, and several cached grids have large prime
factors that force Bluestein's algorithm (14015 = 5 x 2803, 4822 = 2 x 2411,
1301 prime). Measured on a login node: 42x speedup at 4822^2, 9x at 2674^2.
`utils.fft_smoothed_map` already uses `scipy.fft` this way.

### A6. Two behaviours pinned by the tests that the plan did not anticipate

- **`Upsilon(R0; R0) = 0` identically**, so its coefficient at the reference
  radius is a genuine 0/0. The library returns NaN there; the figures and the
  Gate A metrics exclude that bin. Pinned by
  `test_upsilon_vanishes_at_reference_radius`.
- **The positive-weight `Sigma` filter can have a negative auto-amplitude**
  for a field with little large-scale power, which also makes `r` undefined.
  Real projected fields are not in that regime, but the synthetic test fields
  had to be given a realistic (4 arcmin) correlation length for the
  coefficient to be defined across 1'-6'.

### A7. Test-suite location

`tests/` had been emptied into `notebooks/` before implementation, so
`tests/` now contains only the two new files. `notebooks/test_flamingo_io.py`
and `notebooks/test_flamingo_sz.py` are genuine pytest suites that were moved
with the notebooks; they are left where they are rather than moved back
without approval.

### A8. Confirmed as planned

- C4 (SHAM parent-mass pre-filter at 5e14 Msun/h, reproducing
  `stackMap`'s default and hence the f_gas paper's sample) is implemented as
  the config-overridable default `parent_mass_upper`.
- C1 (`'DM'`, not `'dm'`) is moot given A1.
- C5 (`'baryon'` includes BH in all three suites) is unchanged and documented
  in the config.

### A9. Discretization systematics, measured (post-review)

A code review of the implementation ran the integration test and found the FFT
route and the legacy stamp route disagreeing by 6.1 per cent at R=1' while
agreeing to ~1e-15 at most other radii. Investigating that pattern produced
two findings worth recording, and one new committed script,
`scripts/cross_corr/check_resolution.py`.

**Boundary-tie degeneracy.** Pixel membership uses the strict test
`r < edge`, so when an aperture edge coincides with a realizable lattice
distance `sqrt(i^2 + j^2)`, an entire shell of pixels sits on the boundary and
flips membership under an arbitrarily small change of convention. The affected
radii are predicted exactly by `rprofiles.degenerate_apertures`: at a 0.2
arcmin pixel the disk edge of R=1' (5.0 px), the annulus edge of R=2.25'
(3.0'/0.2 = 15.0 px) and the disk edge of R=6' (30.0 px) each carry a
12-pixel shell — and those are precisely the three radii where the two routes
disagreed. This is a discretization degeneracy at isolated radii, not a
smooth under-resolution effect.

**The coefficients are much more robust than the amplitudes.** Every Y
entering an r is filtered with the same kernel, so the shell flip largely
cancels in the ratio: on TNG300-1 it moves the DSigma *amplitude* at R=1' by
6.1 per cent but the *coefficient* by 0.6 per cent.

**Measured budget** (worst |dr/r| over all apertures, filters and the
field-field pairs r_bm and r_em), from `check_resolution.py`:

| Run (z ~ 0.5) | 2x coarser grid | pixel-scale convention |
|---|---|---|
| TNG300-1 (2674 -> 1337) | 1.14e-2 | 1.53e-2 |
| Illustris-1 (966 -> 483) | 1.83e-2 | 1.68e-2 |
| SIMBA m100n1024 (1301 -> 650) | 2.41e-2 | 3.98e-3 |

The 2x-coarsening column is a deliberately aggressive bound; the residual at
the production resolution is a fraction of it. Both columns are far below the
10 per cent Gate A threshold on the cross-code scatter of r_bm/r_gb.

**Upsilon is the sensitive filter.** Every number above is dominated by
Upsilon, which subtracts `(R0/R)^2 * DSigma(R0)` and therefore inherits the
R0 = 1' discretization at *every* radius. This is the same reference-term
amplification the theory note predicts for the physics (Sec. 6, Task 1), here
acting on the discretization instead. Sigma and DSigma stay at the few-per-mille
level. The integration test gives Upsilon its own documented error budget
rather than a blanket tolerance.

**Redshift convention.** The production script and the integration test both
take the angular scale from the snapshot header, not from the config: TNG300-1
snapshot 67 is z = 0.5030, not 0.5. The cached grids were *sized* at z = 0.5
(hence 2674 pixels for a nominal 0.2 arcmin), so at the true redshift the
pixel is 0.19893 arcmin and none of the production apertures is
boundary-degenerate. `stack_on_array` quantizes its own stamp radius grid to
`n_vir / round(n_vir / pixel)` = 0.2 exactly, which is the residual difference
between the two routes; the FFT route is the more accurate of the two.

### A10. Other review fixes applied

- `runCPU_rprofiles_debug.sh` now propagates the pytest exit status. It
  previously ended on an unconditional `echo`, so the job reported
  `COMPLETED 0:0` with a failing integration test — and an
  `--dependency=afterok` on it was satisfied anyway.
- `plot_r_profiles.py` now reports the Gate A scatter over one representative
  run per *code family* (Illustris / TNG / SIMBA / FLAMINGO), with the
  feedback-inclusive all-run scatter printed separately as a diagnostic.
  Pooling the three FLAMINGO variants into the Gate A number would let one
  code's parameter sweep drive a statistic the theory note defines as
  cross-code.
- `load_runs` now matches on `(sim_type, sim_name, feedback, snapshot)`
  instead of the bare snapshot integer, so a stale or unrelated `.npz` sharing
  a snapshot number cannot be pulled silently into a figure.
- `tests/conftest.py` registers the `integration` marker; the `r0` kernel
  guard is checked alongside `min_radius`; the module docstring's self-pair
  formula is corrected to `K(0) / nbar_pix`.

### A11. Incidental finding in the existing `stack_on_array` DSigma route

Isolating the integration-test residuals surfaced a small internal
inconsistency in the legacy stamp route, which predates this work:

`SimulationStacker.stack_on_array` builds its stamp radius grid with
`radial_distance_grid(cutout, (-n_vir, n_vir))`, a linspace over the *rounded*
cutout half-width, so the grid's effective pixel scale is
`n_vir / round(n_vir / pixel)` -- exactly 0.2 arcmin for TNG300-1 snapshot 67
at 2674 pixels. It then passes `pixel_size=arcminPerPixel`, the *true* scale
(0.19893 arcmin), to `filters.delta_sigma_kernel` for the `1/pixArea`
normalization. Membership is therefore decided on one grid and the amplitude
normalized by another, biasing the absolute DSigma amplitude by
`(0.19893/0.2)^2 - 1 = -1.07` per cent at this grid size.

Impact is limited, which is presumably why it has gone unnoticed:

- It is a pure multiplicative factor, so it cancels exactly in any ratio of
  two stacks on the same grid -- including the f_gas paper's kSZ/lensing ratio
  and every coefficient in this task.
- It affects only absolute DSigma amplitudes.

Nothing was changed in `stacker.py`; the integration test simply mirrors the
convention so it compares kernel algebra rather than this bookkeeping. Flagged
here for a decision rather than fixed, since `stacker.py` is shared with the
f_gas figures.

### A12. Expected limitation: small boxes are shot-noise limited in r_gb

The SHAM sample size is fixed by the target number density and the box volume,
`N = target * (L/1000)^3`:

| Run | L [cMpc/h] | N at 5e-4 (z~0.5) | N at 1e-3 (z~0.26) |
|---|---|---|---|
| FLAMINGO L1_m9 | 681 | 157,910 | 315,821 |
| TNG300-1 | 205 | 4,307 | 8,615 |
| SIMBA m100n1024 | 100 | 500 | 1,000 |
| Illustris-1 | 75 | 210 | 421 |

A synthetic check on an Illustris-1-sized grid indicates that at N ~ 200-500
the galaxy auto-correlation Y_gg is shot-noise dominated to the point where it
can go negative after the (unbiased) self-pair subtraction, leaving r_gb
undefined; at N ~ 4000 it is measurable but with large jackknife errors. The
synthetic test used a weakly clustered field and is therefore pessimistic --
real SHAM galaxies are strongly biased, which raises Y_gg relative to shot
noise -- so the production jackknife errors are the number that matters.

The consequence to watch for in the results: **r_bm is a field-field
coefficient and is measured at high signal-to-noise in every box, but r_gb --
and hence the Gate A ratio r_bm/r_gb -- may be dominated by shot noise in
Illustris-1 and SIMBA.** If so, the cross-code Gate A statistic rests mainly
on FLAMINGO and TNG300-1, which would weaken exactly the cross-family
discipline the theory note asks for (Sec. 6). Options, none of which should be
chosen silently:

1. Keep the matched number density and report Illustris-1/SIMBA r_gb with
   their (large) errors, letting the error bars carry the message.
2. Raise the target density for the small boxes, accepting that the samples
   are then not matched to DESI across suites.
3. Restrict the galaxy-crossed part of Gate A to the boxes that support it,
   and use the small boxes only for r_bm.

---

## Results (job 57626842, 12 simulation-samples, yz, 17 min wall-clock)

All twelve runs completed; per-run sanity checks passed (derived CDM mass
fraction matches the box cosmology to 1e-15 for TNG/Illustris/SIMBA and 1e-7
for FLAMINGO, the expected massive-neutrino offset; SHAM number densities
reproduce the target to <0.5 per cent).

### R1. r_bm is measured cleanly everywhere; r_gb is not near unity

`r_bm` is a field-field coefficient and is measured at very high
signal-to-noise in every box: 0.89-1.00 with jackknife errors <= 0.008, and
1.001/1.009/1.001 at R = 1/3.5/6 arcmin for the Sigma filter on FLAMINGO. The
"expectation of near-unity" of the theory note (Sec. 3) holds well for r_bm.

`r_gb` does **not** sit near unity. On FLAMINGO (157,910 galaxies, so
statistically solid) it is 1.34 -> 1.06 -> 1.00 across 1'-6' for Sigma, but
1.71 -> 1.45 -> 1.29 for DSigma. Values above 1 are permitted here: these are
filtered amplitudes `Y = sum_k Whard(k) P(k)` and the compensated kernel
changes sign, so no Cauchy-Schwarz bound applies.

Consequently `r_bm/r_gb` is **0.5-0.85 for DSigma/Upsilon and 0.78-1.0 for
Sigma**, not ~1. For the estimator this is not fatal -- Eq. (4) uses the ratio
as a calibrated transfer, so what matters is its cross-code stability, not its
proximity to 1 -- but it does mean the transfer is a large correction rather
than a small one.

### R2. The filter morphology reproduces Singh et al. (2020) Fig. 1

The Sigma-based coefficient deviates from unity only at small R and returns to
1.000 by 6 arcmin; the compensated DSigma and Upsilon carry the deviation out
to the largest aperture. This is exactly the trade-off the theory note
predicts (Sec. 6, Task 1), now measured for the gas field, which had no clean
precedent.

> **Correction (superseded by Task 2, section T2 below).** This section
> originally concluded that "Sigma is the best-behaved filter on every metric
> here". That conclusion was wrong. The annulus-mean kernel is uncompensated,
> so `Y_Sigma` integrates power down to the box fundamental mode, which
> differs between the 205 cMpc/h TNG300-1 box and the 681 cMpc/h FLAMINGO box;
> Sigma is therefore not the same quantity in the two simulations and its
> apparent superiority is contaminated by that difference. The compensated
> filters are immune. Gate A rests on DSigma and Upsilon alone. See T2.

### R3. Gate A: inconclusive, not failed -- the small boxes are shot-noise limited

The naive four-code scatter of `r_bm/r_gb` is 3-55 per cent and would read as
a Gate A failure. It is not a physics result. The per-run jackknife errors on
the ratio are *as large as or larger than* the scatter itself: at z~0.5 for
DSigma the four-code scatter is 0.185 while the median statistical error on a
single run is 0.546. Illustris-1 (210 galaxies) and SIMBA m100n1024 (500
galaxies) simply cannot measure `Y_gg` at these apertures.

Restricting to the two boxes that can (FLAMINGO L1_m9 fiducial with 157,910
galaxies and TNG300-1 with 4,307), the cross-code scatter collapses to

| filter | z ~ 0.5 | z ~ 0.26/0.30 |
|---|---|---|
| Sigma | 0.008 - 0.074 | 0.013 - 0.083 |
| DSigma | 0.003 - 0.081 | 0.007 - 0.112 |
| Upsilon | 0.020 - 0.089 | 0.006 - 0.132 |

i.e. mostly under 10 per cent, which is the Gate A fixed-transfer criterion.
The three FLAMINGO feedback variants agree with each other at the ~4 per cent
level on the ratio, so feedback sensitivity is mild.

**Two codes is not a cross-code validation.** Singh et al.'s Appendix A lesson
is precisely that priors calibrated on one family bias another. The honest
status is therefore: the estimator looks viable on the evidence available, but
Gate A cannot be declared passed until Illustris and SIMBA contribute a
galaxy sample large enough to measure `Y_gg`. That is a sample-size problem,
not a pipeline problem, and the options are in A12.

### R4. The self-pair subtraction dominates Y_gg at small R

On FLAMINGO the analytic shot-noise term removed from `Y_gg` is 6.9x the
retained signal at R=1', 2.8x at 3.5' and 1.9x at 6'. The subtraction is exact
for a point process (each object contributes exactly one self-pair, so the
term is exactly `K(0)/nbar_pix` regardless of clustering), and the pytest
suite verifies it nulls an unclustered Poisson sample. But it means `r_gb` at
small apertures is a difference of two comparable numbers, so any error in the
shot-noise model propagates straight into it. This mirrors, on the simulation
side, exactly the caveat the theory note raises for the data measurement of
`Y_gg` (Sec. 5.1, "Self-pairs" and "Fibre incompleteness").

---

## Tasks 2 and 3 (job 57633393, four retained simulations, 27 min)

Scope set by the 2026-08-26 decisions: SIMBA and Illustris-1 dropped, xy/xz
not run, apertures extended above 6 arcmin but not below 1 arcmin, R0 kept at
1 arcmin, per-radius jackknife errors only (no covariance), CMB noise out of
scope. ANTILLES and CAMELS deferred; CAMELS is ill-suited anyway because of
its halo-mass limit.

### T2. The filter set: the compensation argument, measured

The annulus-mean ("Sigma") kernel is uncompensated. From the theory note's
Sec. 5.3, `W_ann(k; R1, R2) = 2[R2 J1(kR2) - R1 J1(kR1)] / (k(R2^2 - R1^2))`,
and since `J1(x) -> x/2`, `W_ann(k -> 0) -> 1`. So `Y_Sigma` integrates power
down to the box fundamental, which differs between the 205 cMpc/h TNG300-1 box
and the 681 cMpc/h FLAMINGO box. `W_DSigma(k -> 0) -> 0` by compensation.

`check_filter_compensation.py` removes every mode longer than 205 cMpc/h --
the modes TNG300-1 cannot represent -- and remeasures. At z ~ 0.5:

Worst fractional shift, as amplitude Y / coefficient r / the Gate A ratio
`r_bm/r_gb`:

| run | Sigma | DSigma | Upsilon |
|---|---|---|---|
| TNG300-1 | 5.6e-16 / 6.7e-16 / 6.7e-16 | 6.7e-16 / 1.8e-15 / 1.8e-15 | 6.7e-16 / 1.1e-15 / 1.1e-15 |
| L1_m9 fiducial | 8.7e-2 / 6.1e-3 / 4.9e-3 | 1.4e-4 / 2.1e-5 / 2.0e-5 | 1.4e-4 / 2.5e-5 / 2.0e-5 |
| L1_m9 fgas-8sigma | 8.6e-2 / 6.6e-3 / 5.1e-3 | 1.5e-4 / 2.1e-5 / 1.9e-5 | 1.5e-4 / 2.7e-5 / 2.0e-5 |
| L1_m9 Jet_fgas-4sigma | 8.6e-2 / 5.8e-3 / 4.6e-3 | 1.5e-4 / 1.9e-5 / 1.7e-5 | 1.6e-4 / 2.2e-5 / 1.6e-5 |

TNG300-1 shifting by machine precision is the sanity check: at its own box
scale there is nothing to remove.

**The mechanism is confirmed at the amplitude level and is large.** `Y_Sigma`
moves by 8.7 per cent, against 0.014 per cent for the compensated filters -- a
factor of 600. Any cross-box use of Sigma *amplitudes* is invalid at the
per-cent level, which matters for the Task 4 theory transfer of `Y_mm`.

**It largely cancels in the coefficients.** `r_Sigma` moves by 0.6 per cent
and the Gate A ratio by 0.5 per cent, against 2e-5 for the compensated
filters. The cancellation happens because the removed large-scale modes have
`r(k) ~ 1` and so contribute almost equally to the cross and to both autos.
The compensated filters remain roughly 250 times less sensitive even here.

**And it does not explain any of the cross-code disagreement.** Cutting both
simulations at the same physical scale leaves the FLAMINGO-versus-TNG300-1
difference essentially untouched, on every filter and on both the clean
field-field coefficient and the metric Gate A actually uses:

| quantity | filter | before cut | after cut |
|---|---|---|---|
| `r_em` | Sigma | 1.489e-2 | 1.561e-2 |
| `r_em` | DSigma | 7.363e-2 | 7.363e-2 |
| `r_em` | Upsilon | 1.575e-2 | 1.575e-2 |
| `r_bm/r_gb` | Sigma | 1.140e-1 | 1.145e-1 |
| `r_bm/r_gb` | DSigma | 3.546e-1 | 3.545e-1 |
| `r_bm/r_gb` | Upsilon | 4.417e-1 | 4.417e-1 |

(The `r_bm/r_gb` values are maxima over all fifteen apertures and are
dominated by the unreliable 9.75 arcmin bin discussed in T5; the point here is
the before/after comparison, which is null.) **The box difference does not
explain the cross-code disagreement seen in Task 1, for any filter.**

So the filter freeze to {DSigma, Upsilon} stands, but the justification is not
the one that motivated it. It rests on two things:

1. **Sigma amplitudes do not port between boxes** (8.7 per cent), which is
   disqualifying for anything that compares or calibrates `Y_Sigma` across
   simulations or against a theory prediction -- the Task 4 transfer of
   `Y_mm` in particular.
2. **The measurement-side argument of the note**: an uncompensated disk mean
   is not measured on the lensing side, and on the kSZ side it reintroduces
   large-scale CMB noise.

It does *not* rest on Sigma's coefficients being contaminated. They largely
are not, and the cross-code disagreement that made Sigma look best in Task 1
survives the cut intact -- so that disagreement is astrophysical or
statistical in origin, not geometric.

A practical corollary for any future use of the compensated filters: their
insensitivity holds only while the removed scale sits well above the aperture.
Measured on synthetic fields, the Sigma-to-DSigma separation is 135x when the
cut is 16x the aperture, 27x at 8x, and 5x at 4x. Production sits at 55x
(TNG300-1's 535 arcmin box against a 9.75 arcmin largest aperture).

The point-mass completion of Task 2(b) was **not** implemented and remains on
the list: Singh et al.'s claim that Sigma-based coefficients localize better
is therefore still untested here, and is the open "discussion item with Uros".

### T3. Electrons versus all baryons

`b` = gas + stars + BH, `e` = ionized gas. Neutral gas is not separated from
stars, by decision; the split is the aggregate `b - e`.

Correction factor `Y_gb / Y_ge(R)` at z ~ 0.5, at R = 1, 3.5 and 9.75 arcmin:

| run | Sigma | DSigma | Upsilon |
|---|---|---|---|
| TNG300-1 | 1.003 / 1.011 / 1.023 | 1.206 / 1.047 / 1.023 | - / 1.002 / 1.000 |
| L1_m9 fiducial | 1.024 / 1.028 / 1.040 | 1.611 / 1.111 / 1.057 | - / 1.015 / 1.021 |
| L1_m9 fgas-8sigma | 1.056 / 1.023 / 1.039 | 2.268 / 1.205 / 1.069 | - / 1.058 / 1.027 |
| L1_m9 Jet_fgas-4sigma | 1.041 / 1.024 / 1.030 | 1.748 / 1.149 / 1.062 | - / 1.041 / 1.027 |

The electron field misses a large fraction of the baryon signal at small
apertures under the compensated filter -- 21 per cent for TNG300-1 and 127 per
cent for FLAMINGO fgas-8sigma at R = 1 arcmin -- and the correction is
strongly scale-dependent, decaying to a few per cent by 9.75 arcmin. Crucially
it is also strongly **feedback**-dependent: at R = 1 arcmin the DSigma
correction spans 1.61 to 2.27 across the three FLAMINGO variants, a factor of
1.4 within a single code.

On cross-code stability of the ratio each framing would calibrate, the two are
close. Worst over the data range: at z ~ 0.5 DSigma gives 0.081 for the baryon
route against 0.094 for the electron route (14 per cent apart, nominally
favouring baryons) and Upsilon 0.089 against 0.092 (2 per cent apart); at
z ~ 0.26 all four agree within 8 per cent of each other. **None of these
differences is meaningful**: each spread is itself taken over only two code
families, so it is a single pairwise difference divided by sqrt(2) with no
error bar, and comparing two such numbers cannot establish a preference.
**The choice cannot be made on stability.**

The deciding evidence is the correction factor itself. Targeting `P_bm/P_mm`
requires carrying an electron-to-baryon transfer that is large and varies by
~40 per cent between feedback variants of one code at the innermost aperture;
targeting `P_em/P_mm` needs no such transfer, because the electron field is
what the kSZ measures. **Recommendation: target `P_em/P_mm`**, and treat the
stellar and neutral terms as a separate, externally constrained contribution
in the suppression mapping, which is the first of the note's two framings.

### T4. Gate A on the four retained simulations

Over the observational range 1-6 arcmin, with the two code families:

| filter | z ~ 0.5 | z ~ 0.26/0.30 |
|---|---|---|
| Sigma | 0.074 (PASS) | 0.083 (PASS) |
| DSigma | 0.081 (PASS) | 0.112 (MARGINAL) |
| Upsilon | 0.089 (PASS) | 0.132 (MARGINAL) |

**These are sample standard deviations over two members, which for N = 2 is
just the pairwise difference divided by sqrt(2).** The raw
TNG300-1-versus-FLAMINGO differences are a factor sqrt(2) larger: 10.5, 11.4
and 12.6 per cent at z ~ 0.5, and 11.7, 15.9 and 18.7 per cent at z ~ 0.26.
The verdict therefore straddles the note's 10 per cent boundary depending on
which statistic is quoted, and is consistently worse at the lower redshift.
The honest reading is **borderline between the fixed-transfer and
parametrized-r routes**, on an estimate from two code families that cannot
support an error bar.

### T5. What the aperture extension showed

The ratio `r_bm/r_gb` rises monotonically with aperture -- 0.53 at 1 arcmin to
about 0.85 by 9 arcmin for DSigma -- consistent with approaching unity on
large scales as the theory note expects, and confirming that the departure
from unity is a small-scale phenomenon.

The largest bin is not trustworthy in the smaller box. TNG300-1's `Y_gg` falls
anomalously at 9.75 arcmin (4.18 to 2.04, roughly halving, where FLAMINGO
declines smoothly), which pushes its ratio from 0.83 to 0.63. The cause is
sample size: with 4,307 galaxies the analytic self-pair subtraction removes 73
per cent of the raw `Y_gg` at that aperture, so the residual is a difference
of comparable numbers. All reported statistics are therefore split into the
data range and the diagnostic extension, and the Gate A verdict uses the data
range only.

---

## Task 4 (job 57638497, four simulations, 3 min)

Scope set by the 2026-08-26 decisions: `P_mm^hydro-CDM / P_mm^DMO = 1` adopted
rather than measured (no DMO run on disk; downloading one would take over a
day); the full box depth used for simulation validation and the `Pi_max`
cylinder reserved for the data chain; the RSD/`Pi_max` recheck for `Y_gg`
deferred to Phase 4 with a simulation-side proxy in its place; Gate B closed.
The Phase 0 specification that all of this belongs in is now written, at
`docs/filter_specification.md`.

### T6. Two bugs the decomposition caught

Splitting the comparison into A (transfer chain, no cosmological model), B
(CDM versus total matter) and C (halofit) was what made the errors findable.
The first run gave C = +129 per cent, a factor 2.3 -- far too large to be
halofit error, and since A was small the fault had to be in the theory branch.

1. **CAMB returns its redshift axis in increasing order**, whatever order the
   redshifts were requested in, so `pk[0]` was the z = 0 spectrum rather than
   z = 0.503. At these redshifts that overstates the power by ~2.3x. The slice
   is now selected by matching the requested value.
2. Requesting z = 0 made the internal redshift list `[0.0, 0.0]`, and CAMB's
   ODE integrator fails on duplicates with an opaque "Error in dverk". The
   list is now de-duplicated.

A code review found two more, both sub-per-cent but both real:

3. **sigma8 was applied after halofit rather than before.** halofit's mapping
   is not homogeneous in the input amplitude, so rescaling its output by
   `(sigma8_target/sigma8_actual)^2` is exact only in the linear regime; for
   TNG300-1's 0.8 per cent amplitude mismatch that biased P(k) by up to 0.9
   per cent near k ~ 1 h/Mpc. The primordial amplitude is now solved for
   before the non-linear step. The original test could not have caught this,
   since both of its calls shared one underlying amplitude and the ratio held
   by construction; it is replaced by an independent integral of the linear
   spectrum against the 8 Mpc/h top hat.
4. **CAMB's log-spaced k under-samples the oscillating kernel.** The aperture
   kernels oscillate on a fixed period `2 pi / R` in k, so log spacing thins
   out exactly where it should not; the amplitude at the largest apertures
   drifted by ~0.3 per cent with the number of CAMB samples. Spectra are now
   resampled onto a kernel-resolving linear grid, with a convergence test.

All four are pinned by regression tests in `tests/test_kernels.py`.

### T7. The theory chain validates at the few-per-cent level

Worst fractional difference against the measured CDM amplitude, over the full
aperture range:

| run | filter | A transfer | B tot/CDM | C halofit |
|---|---|---|---|---|
| TNG300-1 | DSigma | 1.7e-2 | 8.9e-2 | 4.6e-2 |
| TNG300-1 | Upsilon | 1.8e-2 | 3.0e-2 | 6.4e-2 |
| L1_m9 fiducial | DSigma | 2.1e-2 | 1.34e-1 | 5.6e-2 |
| L1_m9 fiducial | Upsilon | 2.2e-2 | 8.4e-2 | 6.7e-2 |
| L1_m9 fgas-8sigma | DSigma | 2.0e-2 | 1.69e-1 | 4.8e-2 |
| L1_m9 fgas-8sigma | Upsilon | 2.2e-2 | 1.44e-1 | 5.5e-2 |
| L1_m9 Jet_fgas-4sigma | DSigma | 2.1e-2 | 1.67e-1 | 4.6e-2 |
| L1_m9 Jet_fgas-4sigma | Upsilon | 2.2e-2 | 1.32e-1 | 5.4e-2 |

**A, the transfer chain, is 1.7-2.2 per cent everywhere** and mostly far
better than that away from the innermost aperture. This is the accuracy with
which the harmonic-space route of Sec. 5.3 reproduces the real-space filtering
the pipeline performs, with no cosmological model involved. It is limited by
the pixelization of the aperture, and specifically by the 0.75 arcmin annulus,
which spans only 3.75 pixels at the production resolution (see
`tests/test_kernels.py::TestAnalyticMatchesPixelized`).

**C, halofit, is 4.6-6.7 per cent** and does not vary much between suites,
which is the expected accuracy of the non-linear model at these scales.

### T8. The dominant theory-side systematic is CDM versus total matter

**B is the largest term in the table, at 3 to 17 per cent, and it is
feedback-dependent.** halofit returns the *total* matter power spectrum while
the estimator defines `m` as CDM. Ordered by feedback strength:

| run | B (DSigma) |
|---|---|
| TNG300-1 | 8.9 per cent |
| L1_m9 fiducial | 13.4 per cent |
| L1_m9 Jet_fgas-4sigma | 16.7 per cent |
| L1_m9 fgas-8sigma | 16.9 per cent |

That ordering is physical: stronger feedback ejects more gas, so the total
matter field departs further from the CDM field. The consequence for the
programme is that this is **not a fixed transfer** -- it is a
feedback-dependent one, and it is an order of magnitude larger than the 1-2
per cent hydro-CDM versus DMO back-reaction we chose to set to unity. Booking
the back-reaction while ignoring this would be the wrong priority.

Two ways out, neither chosen here: predict the CDM-only spectrum rather than
the total (CAMB can do the linear CDM spectrum, but halofit's non-linear
correction is formulated for total matter), or redefine `m` as total matter
throughout, which changes the estimator's meaning and would have to be agreed
with the theory note.

### T9. The coefficients are insensitive to projection depth

`check_projection_depth.py` slices the TNG300-1 3D fields into slabs of 26,
51, 102 and 205 cMpc/h and reprojects. The amplitudes behave exactly as the
`P_2D = P_3D/L` scaling predicts: going from the full box to one eighth of it
raises `Y_mm` by a factor 7.4 to 7.8, against the 8 that exact inverse-depth
scaling would give.

The coefficients do not move at all:

| filter | worst \|r(26 cMpc/h) / r(205 cMpc/h) - 1\| |
|---|---|
| DSigma | 3.8e-4 |
| Upsilon | 1.6e-3 |

**Over a factor of eight in projection depth the coefficient shifts by less
than 0.2 per cent.** This is the result that licenses the whole calibration
strategy: the amplitudes are convention-dependent and do not port between a
simulation box and a `Pi_max` cylinder, but `r_bm/r_gb` -- the only thing the
simulations are asked to deliver -- is essentially independent of the
projection. The mismatch between the kSZ's full-line-of-sight integration and
the clustering's 100 h^-1 Mpc cylinder therefore threatens the amplitudes in
Eq. (4), which must be made consistent by construction, but not the calibrated
transfer.

Caveat: this used the 1000^3 cached 3D grid, whose 0.53 arcmin cells
under-resolve the smallest apertures, so the study covers R >= 2.25 arcmin
only. FLAMINGO has no cached 3D CDM or ionized-gas field, so the study is
TNG300-1 only.
