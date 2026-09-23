# Matter power spectrum suppression by baryon component (unbound gas paper)

**Status as of 2026-09-22.** Code in `scripts/unbound_gas/`, config
`scripts/configs/unbound_gas/pk_components_z05.yaml`. Commits `435d9ea`
(DMO downloads), `cc17c7e` (component pipeline) and the local-model commit
(`compute_pk_local.py`, `make_pk_local.py`, `runINT_pk_local.sh`). Detailed working log:
`NOTES/unbound_gas/` (untracked).

## 1. Question and models

How much do the baryons the SZ effects cannot see — stars, neutral gas, black
holes — matter for the baryonic suppression of the matter power spectrum? All
at z = 0.5 for TNG300-1, Illustris-1, SIMBA m100n1024 (s50) and FLAMINGO
L1_m9 (fiducial, fgas-8sigma, Jet_fgas-4sigma).

Each simulation's 3D TSC grid is split into five mass components:

| component | built from the cached fields as |
|---|---|
| DM | total − gas − Stars − BH (exact: TSC is linear, same grid) |
| ionized_gas | cached (electron count × m_p μ_e, X_H = 0.76) |
| neutral_gas | gas − ionized_gas |
| Stars, BH | cached |

With component weights c_i (mass fractions of all matter; c_i = w_i for the
simulation) and auto/cross spectra P_ij of the component overdensities,
**P_mm(c) = Σ_ij c_i c_j P_ij** exactly, for any linear redistribution of mass
between components. Two families of redistribution:

- **Global (alpha) model** — `make_pk_alpha.py`. All baryon mass on two
  templates: ρ_b(α) = ρ̄_b [α u_ion + (1 − α) u_else], u_x = ρ_x / ρ̄_x, u_else
  the mass-weighted mix of neutral gas, stars and BH. α₀ = ρ̄_ion / ρ̄_b (the
  box-wide ionized share of the baryons) reproduces the simulation; α = 1 lays
  every baryon out like the ionized gas. P_mm(α) is exactly quadratic in α.
  Because the templates have different large-scale biases, this also changes
  P_mm on the largest scales (−0.6 to −1.25%; SIMBA +2.6%).
- **Local model** — `compute_pk_local.py` / `make_pk_local.py`. Each moved
  component e is transported within a radius R so that it follows a target T
  (local ionized gas or local DM):
  m_e = ρ_T · [W_R ∗ (ρ_e / (W_R ∗ ρ_T))].
  Every cell spreads its mass over the kernel around it in proportion to ρ_T:
  mass is conserved exactly and, for the sphere, moves at most R, so power on
  scales k ≪ 1/R is unchanged. R → 0 is the simulation; large R approaches the
  global model on small scales.

Quantities reported (reference P_DMO = DM field of the matched DMO run, same
grid):

| symbol | definition | note |
|---|---|---|
| S(k) | P_mm / P_DMO | suppression |
| Q(k) | P_mm(moved) / P_mm(simulation) − 1 | reference-free headline |
| ΔS(k) | S(moved) − S(simulation) | absolute change in the suppression; ΔS = S · Q |

(ΔS replaced an earlier "share of the total suppression", [S − S(moved)] / [1 − S],
which diverges on large scales where 1 − S → 0.)

## 2. Inputs

- Cached 3D fields `<root>/<SimType>/products/3D/`: 1000³ for
  TNG/Illustris/SIMBA, 2000³ float32 for FLAMINGO (k_Nyq = 15.3, 41.9, 31.4,
  9.2 h/Mpc for TNG300-1, Illustris-1, SIMBA-100, FLAMINGO).
- DMO runs (same initial conditions; checked: hydro × DMO correlation → 1 on
  large scales):

| hydro | DMO snapshot | how obtained |
|---|---|---|
| TNG300-1 67 | `IllustrisTNG/TNG300-1-Dark/output/snapdir_067/` | TNG API (`fetch_dmo_snapshots.sh`) |
| Illustris-1 103 | `IllustrisTNG/Illustris-1-Dark/output/snapdir_103/` | TNG API |
| SIMBA m100n1024 125 | `SIMBA/m100n1024/dm/snapshots/snap_m100n1024_009.hdf5` (z = 0.4904) | public http, simba.roe.ac.uk |
| FLAMINGO 67 (all variants) | `FLAMINGO/L1_m9/L1_m9_DMO/snapshots/flamingo_0067/FLAMINGO/L1_m9/L1_m9_DMO/snapshots/flamingo_0067/` (nested) | pre-existing |

## 3. Scripts (run from `scripts/`)

| script | does | runs on | output |
|---|---|---|---|
| `fetch_dmo_snapshots.sh` | downloads the DMO snapshots; staged, size-checked, verified, atomic install; `DATASETS="simba illustris tng300"` | `sbatch` (xfer QOS, `-C cron`) | snapshot files |
| `verify_snapshot_files.py` | `check` / `install` / `snapshot` integrity checks for Gadget-format HDF5 | login | stdout |
| `pk_common.py` | shared helpers: config, component list, cache and spectra paths, atomic saves | — | — |
| `build_dmo_field.py` | DM field of each DMO run on the hydro grid via `mapMaker.make_mass_field` (stand-in object); requires a complete snapshot, box size = hydro's, mass = Ω_m ρ_crit V to 1e-3 | CPU node | `<stem>_DM_<n>.npy` |
| `validate_pk_caches.py` | checks on the cached fields: finiteness, derived-DM negative mass, mass budget vs cosmology, ionized ≤ gas, 3D vs 2D totals, optional particle rebuild (`--rebuild`, like-for-like dtype), derived DM vs `DM_512` | CPU node | stdout |
| `compute_pk_components.py` | 15 component spectra + P_total (Pylians, TSC); DMO spectra (P_total, P_DMO, cross); `--fresh` rebins from particles | CPU node | `<stem>_Pk_components_<n>.npz`, `<stem>_Pk_dmo_<n>.npz` |
| `make_pk_alpha.py` | global model: S bands α₀ → 1, Q, per-component Q, alternative layouts (like DM, uniform), ΔS, DMO vs hydro-DM check | login | `<fig>_alpha/_targets/_dmo_check.pdf`, `_table.txt` |
| `compute_pk_local.py` | local model spectra for every kernel × R × target × moved set; own Pylians-compatible estimator; `--identity-check` | CPU node | `<stem>_Pk_local_orig_<n>.npz`, `<stem>_Pk_local_<kernel>_R<R>_<n>.npz` |
| `make_pk_local.py` | local model: S(k) bands at one R for all sims with ΔS below (`--band-radius`, `--band-kernel`; layout of `make_pk_alpha`'s figure), per-sim S(k)/ΔS grid over R with the global model, Q per R and kernel, per-component split at one R, table incl. large-scale Q and diagnostics | login | `<fig>_local_S_<kernel>_R<R>_<target>.pdf`, `_local_S_grid_<kernel>_<target>.pdf`, `_local_Q_<target>.pdf`, `_local_split.pdf`, `_local_table.txt` |
| `runINT_pk_components.sh` | stages `validate dmo spectra` in one allocation | `salloc` | logs in `../Outputs_Perlmutter/` |
| `runINT_pk_local.sh` | `compute_pk_local.py`, one simulation per node | `salloc -N ≤4` | logs in `../Outputs_Perlmutter/` |

`<stem>` = `<sim>_<snap>` (TNG/Illustris) or `<sim>_<feedback>_<snap>`
(SIMBA/FLAMINGO), in `<root>/<SimType>/products/3D/`. Figures go to
`../figures/YYYY-MM/MM-DD/`.

### Config (`pk_components_z05.yaml`)

- `pk.threads`, `pk.omega_baryon_fallback` (mass-budget checks only),
  `pk.k_table` (table wavenumbers).
- `simulations[]`: `sim_type`, `name`, `snapshot`, `feedback`, `n_pixels`,
  `dmo` (name, snapshot, feedback, path relative to the data root; the
  FLAMINGO variants share one DMO entry).
- `local` (read only by the two local-model scripts): `radii` [Mpc/h],
  `min_radius_cells` (2: FLAMINGO skips R = 0.5), `kernels` (`tophat`,
  `gaussian`), `gaussian_sigma_factor` (0.447 = same rms radius as the
  sphere; 1.0 = W(k) = exp(−k²R²/2) convention), `targets` (`ionized`, `dm`),
  `moved` (`all` = neutral + stars + BH, `stars`, `neutral`).

### Spectra file contents

- `_Pk_components_`: `k` [h/Mpc], `Nmodes`, `components`, `means` (box mean
  per component, M⊙/h per voxel), `P` (5 × 5 × Nk), `P_total`, `box_mpc`,
  `n_pixels`.
- `_Pk_dmo_`: `k`, `Nmodes`, `P_total`, `P_dmo`, `P_total_dmo`.
- `_Pk_local_orig_`: as `_Pk_components_` from the local estimator, plus
  `max_rel_diff_vs_pylians`.
- `_Pk_local_<kernel>_R<R>_`: `k`, `R`, `kernel`, `sigma`, and per target t and
  moved set s: `P_auto__<t>__<s>` (Nk), `P_cross__<t>__<s>` (5 × Nk, with the
  original components), `renorm__<t>__<s>`, `kept_frac__<t>__<s>`.

All spectra: monopole, TSC-deconvolved, k from the fundamental mode to the grid
Nyquist frequency, identical k bins across files.

## 4. Validation summary

- Mass budgets vs Ω_m ρ_crit V and Ω_b/Ω_m to ≤ 6e-5; 3D vs 2D totals ≤ 5e-5.
- Algebra: Σ w_i w_j P_ij = P_total to ≤ 1.3e-7.
- SIMBA-100 end-to-end particle rebuild (`--fresh`) reproduces every spectrum
  to < 7e-5 at k ≤ 10 h/Mpc despite ~400 inconsistent void cells in the old
  caches.
- FLAMINGO Stars rebuild bit-identical to its cache.
- Local estimator = Pylians to 2e-7; identity kernel reproduces P(e, e) to
  ≤ 3e-6; transport conserves mass to 1e-7.

## 5. Caveats

- **SIMBA is provisional.** `ElectronAbundance` appears to be n_e m_p/ρ rather
  than n_e/n_H for ~65% of its gas; if so, most of SIMBA's "neutral gas" is
  ionized. Under investigation (separate session).
- TNG300-1 / Illustris-1 3D Stars caches hold ~4e-5 less stellar mass than the
  current code builds (provenance unknown; effect on P_mm < 1e-4).
- Fixed X_H = 0.76 makes ionized > gas in metal-enriched cells (5% of cells in
  TNG300-1, 21% in Illustris-1; negative neutral mass ≤ 0.12% of gas).
- The hydro run's own DM is not an adequate reference: it underestimates the
  suppression at k = 5 h/Mpc by up to ~0.07; all results use DMO runs.
- FLAMINGO neutrinos are excluded from both hydro and DMO fields.

## 6. Results snapshot (global model, 2026-09-22)

| sim | α₀ | S(k≈1) | S(k≈5) | Q(k≈1) | Q(k≈5) | Q, stars only (k≈5) | Q, largest scales |
|---|---|---|---|---|---|---|---|
| TNG300-1 | 0.955 | 0.992 | 0.892 | −1.0% | −3.8% | −3.6% | −0.6% |
| Illustris-1 | 0.861 | 0.913 | 0.740 | −2.2% | −8.5% | −7.8% | −1.0% |
| SIMBA-100 (prov.) | 0.769 | 0.945 | 0.772 | +2.5% | −3.3% | −5.3% | +2.6% |
| FLAMINGO L1_m9 | 0.915 | 0.978 | 0.862 | −2.2% | −7.2% | −6.2% | −1.2% |
| fgas-8sigma | 0.909 | 0.936 | 0.806 | −2.6% | −7.9% | −6.9% | −1.25% |
| Jet_fgas-4sigma | 0.917 | 0.927 | 0.805 | −2.2% | −6.4% | −5.5% | −1.0% |

## 7. Results snapshot (local model, 2026-09-22)

Figures: `pk_components_z05_local_S_tophat_R1_{ionized,dm}.pdf` (S bands, all
simulations) and `pk_components_z05_local_S_grid_{tophat,gaussian}_{ionized,dm}.pdf`
(S and ΔS versus R per simulation). All non-ionized baryons moved **like the
local ionized gas**, sphere kernel, Q at the k closest to 1 and 5 h/Mpc; "large scales" is the mode-weighted mean
over k ≤ 3 k_F. Full tables (both kernels, all R, both targets, per component):
`figures/2026-09/09-22/pk_components_z05_local_table.txt`.

| sim | Q(k≈1) global | Q(k≈1) R=1 | Q(k≈5) global | Q(k≈5) R=1 | Q(k≈5) R=4 | ΔS(k≈5) R=1 | Q, large scales: global / R=1 / R=4 | Q(k≈5) R=1, like local DM |
|---|---|---|---|---|---|---|---|---|
| TNG300-1 | −1.00% | −0.03% | −3.8% | −2.7% | −3.4% | −0.024 | −0.58 / −0.000 / +0.002% | −1.7% |
| Illustris-1 | −2.17% | −0.11% | −8.5% | −7.1% | −8.3% | −0.053 | −0.99 / −0.002 / +0.086% | −3.0% |
| SIMBA-100 (prov.) | +2.50% | +0.16% | −3.3% | −3.7% | −4.0% | −0.029 | +2.69 / +0.009 / +0.160% | −1.3% |
| FLAMINGO L1_m9 | −2.17% | −0.05% | −7.2% | −5.2% | −6.8% | −0.045 | −1.18 / −0.000 / +0.000% | −2.3% |
| fgas-8sigma | −2.64% | −0.09% | −7.9% | −6.1% | −7.7% | −0.049 | −1.23 / −0.000 / +0.000% | −2.4% |
| Jet_fgas-4sigma | −2.17% | −0.06% | −6.4% | −4.6% | −6.1% | −0.037 | −1.00 / +0.001 / +0.000% | −1.8% |

- Local transport removes the large-scale shift of the global model (≤ 0.01%
  for R ≤ 1 Mpc/h). Most of the global model's Q at k ≈ 1 h/Mpc was that shift.
- At k ≳ 5 h/Mpc the effect is robust: Q grows towards the global value as R
  increases, and is dominated by the stars
  (`pk_components_z05_local_split.pdf`).
- Laying the mass out like the local DM instead gives 0.35–0.63 of the "like
  ionized gas" effect at k ≈ 5 (R = 1 Mpc/h, sphere).
- The Gaussian kernel (σ = R/√5) moves mass less far than the sphere of the
  same R and gives smaller |Q| at fixed R.

Implementation choices worth knowing: transport ("scatter") form, not the
gather form ρ_T [W∗ρ_e]/[W∗ρ_T] (which needs a 0.2–2.5% global rescaling on
test fields and does not bound the displacement); target floor 1e-3 of the
mean smoothed target (cells below it keep their mass); spectra by a numba
estimator that reproduces Pylians exactly, because Pylians holds every FFT of
its input list and FLAMINGO's extra fields would not fit in memory (measured
peak 361 GB per FLAMINGO variant).

## 8. Reproduce

```bash
cd scripts/
sbatch unbound_gas/fetch_dmo_snapshots.sh                     # DMO snapshots
salloc -q interactive -C cpu -N 1 -t 4:00:00 -A desi bash unbound_gas/runINT_pk_components.sh
python unbound_gas/make_pk_alpha.py -p configs/unbound_gas/pk_components_z05.yaml
salloc -q interactive -C cpu -N 4 -t 2:00:00 -A desi bash unbound_gas/runINT_pk_local.sh
python unbound_gas/make_pk_local.py -p configs/unbound_gas/pk_components_z05.yaml
```

Measured cost: FLAMINGO component spectra ~27 min per variant (376 GB peak);
local model ~6 min per 1000³ simulation (46 GB peak) and ~37 min per FLAMINGO
variant (361 GB peak).
