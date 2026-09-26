# Observation-based stellar bands: S(k) and the lensing f_gas(θ) with the stars rescaled to observed aggregates (unbound gas paper)

**Status as of 2026-09-26** (runs of 2026-09-25, all ten simulation-redshift
pairs done). Code: `scripts/unbound_gas/{compute_fstar_obs.py,
stack_fstar_obs_maps.py, make_pk_fstar_obs.py, runINT_fstar_obs.sh}`, configs
`scripts/configs/unbound_gas/pk_fstar_obs_{z05,z026}.yaml`, tests
`tests/test_fstar_obs.py`. Nothing in `src/` changed: the capped ends use the
floor functions of `src/halo_transfer.py` (`halo_baryons`,
`scaled_transfer_field`, `scaled_transfer_maps_2d`). Builds on the s = 0
transfer (`pk_stellar_transfer.md`) and the f* floor runs (`pk_fstar_transfer.md`),
whose files it reads; their configs and scripts are unchanged. Working log:
`NOTES/unbound_gas/session_log_2026-09-25_fstar_obs.md` (untracked).

## 1. Question and parameterisation

The f* bands of `pk_fstar_transfer.md` run from f* = 0 to a floor set near the
most star-rich simulation (Illustris-1), and scale every star of a region
together; both ends lie outside what observations allow for groups and
clusters. Here the ends are set by observed **aggregate** stellar quantities of
the haloes with FoF mass ≥ 1e13 M⊙/h, within 1 R200m (the `ap1` regions: own
R200m apertures, mass priority where they overlap):

- **option (a), f\***: f* = ΣM* / Σ(M* + M_wind + M_gas + M_BH) (true stars;
  winds count as gas), targets **0.05** (low) and **0.30** (high);
- **option (c), M\*/M200m**: ΣM* / ΣM200m (catalogue SO mass,
  `GroupMass_m200m`), targets **0.01** and **0.04**.

Each end is a **population rescale**: every region's stars × one factor s per
simulation and end, exchanged with the region's ionized gas (new stars laid
out like its stars, gas in proportion to its ionized mass), with s solved so
that the aggregate hits the target. The simulation's halo-to-halo scatter and
mass trend are kept. Where s > 1, a region gains at most all of its ionized
gas (**capped**), and s is re-solved so that the aggregate still hits the
target. Baryons are conserved region by region; nothing outside the regions
changes. The same targets are used at z ≈ 0.5 and z ≈ 0.26.

The literature behind the numbers (`NOTES/unbound_gas/lit_stellar_fractions_groups_2026-09-25.md`,
a subagent review not re-verified): no direct constraint on f* at R200m
(the gas beyond R500c is unmeasured in groups); M*/M200m ≈ 0.01–0.04. The
targets are the user's preliminary choices (2026-09-25) and may be revised.

## 2. Definitions (user decisions 2026-09-25)

| item | choice |
|---|---|
| how a target is applied | population rescale (one s per simulation and end), not a per-halo floor/cap |
| regions | 1 R200m apertures (`ap1`), FoF mass ≥ 1e13 M⊙/h |
| f* | true stars / (true stars + winds + all gas + BH), as `pk_fstar_transfer.md` |
| M200m | catalogue SO/200_mean mass of the haloes above the cut |
| exchange | stars ↔ ionized gas (`mapMaker.ionized_gas_masses`); new stars like the existing stars |
| s > 1, too little ionized gas | the region converts all of it (capped); s re-solved |
| regions with stars but no ionized gas | unchanged (as the s = 0 transfer) |
| simulations | TNG300-1, Illustris-1, FLAMINGO L1_m9 / fgas-8sigma / Jet_fgas-4sigma; no SIMBA |
| figures | one S(k) + lensing f_gas(θ) figure per option and redshift; no f* = 0 line |

## 3. Algebra

For the regions h with stars and ionized gas (M*_h, M_ion,h > 0), an end with
scale s changes each star by c_h m*_j and each gas particle's ionized mass by
−c_h (M*_h / M_ion,h) m_ion,i, with

    c_h = s − 1                         (s ≤ 1)
    c_h = min(s − 1, M_ion,h / M*_h)    (s > 1; = M_ion,h / M*_h: capped),

and s solves Σ_h M*_h (1 + c_h) + M*_fixed = T × (ΣM_b or ΣM200m), a piecewise
linear equation solved exactly (`compute_fstar_obs.solve_scale`).

- **Uncapped end** (every s ≤ 1, and s > 1 while (s − 1) M*_h ≤ M_ion,h in
  every region): the change is (1 − s) D, D the s = 0 transfer field, so

      P_mm = P_mm + 2(1 − s) P_mD + (1 − s)² P_DD ,
      f(θ) = [N + tA] / [T + t(A − B)] × Ω_m/Ω_b ,  t = 1 − s,

  exactly, from the existing s = 0 spectra and lensing stacks (source `s0`).
- **Capped end**: its own zero-mass change field H (TSC) and 2D maps S (stars
  added) and G (ionized gas removed) from one particle pass (source `pass`):

      P_mm = P_mm + 2 P_mH + P_HH ,   f(θ) = [N − ΔΣ_G] / [T + ΔΣ_S − ΔΣ_G] × Ω_m/Ω_b .

## 4. Scripts (run from `scripts/`)

| script | does | runs on | output |
|---|---|---|---|
| `compute_fstar_obs.py` | s and source of every end from the s = 0 file (per-region M*, M_ion table), the f* file (ΣM_b) and the catalogue (ΣM200m); for capped ends one pass (stars, gas, winds, BH), s re-solved from the pass's float64 sums, then H and its spectra and the 2D maps; `--solve-only` | login (no capped end) / CPU node | `<stem>_Pk_fstar_obs_ap1_<n>.npz` (3D), `<stem>_fstarobs_<option>_<end>_{stars,gas}_ap1_M13_<n2>_yz.npy` (2D) |
| `stack_fstar_obs_maps.py` | ΔΣ stacks of S and G on the lensing sample (same rows and settings as the s = 0 stacks) | CPU node | `<stem>_lensing_fstar_obs_<sample>.npz` (2D) |
| `make_pk_fstar_obs.py` | figures and table; `--preview` draws the uncapped ends before any pass | login | `<fig>_{fstar,mstar_m200m}_S_ap1_M13`, `<fig>_table.txt` |
| `runINT_fstar_obs.sh` | the first two, one run per node; `RUNS="<z>:<sim> ..."`, `STAGES`, `EXTRA` | `salloc -N ≤ 4` | logs `fstar_obs-<job>_<z>_<sim>.out` |

Measured cost (one node per run): FLAMINGO compute 45–48 min (pass 40–42 min)
plus stacks 2–3 min, peak 150–192 GB; TNG300-1 z ≈ 0.26 compute 27 min (pass
16 min) plus stacks 1.5 min, 149 GB; Illustris-1 8–9 min. TNG300-1 z ≈ 0.5
needs no pass (login node, ~1 min). New
data: 30 maps (25 GB, products/2D), 10 results files (products/3D), 9 stack
files (products/2D). Jobs (2026-09-25): 58880422 (smoke: Illustris-1 both z,
FLAMINGO 2 chunks), 58880592 (FLAMINGO ×3 both z, TNG300-1 z ≈ 0.26).

### Files and modes

Results file `<stem>_Pk_fstar_obs_ap1_<n>.npz` (products/3D), one per
simulation and snapshot:

| keys | content |
|---|---|
| `ends` | `fstar__low`, `fstar__high`, `mstar_m200m__low`, `mstar_m200m__high` |
| `target__<end>`, `s__<end>`, `source__<end>`, `n_capped__<end>`, `capped_frac__<end>` | the target, the solved scale, `'s0'` or `'pass'`, the capped regions and their share of the changing stars |
| `fstar_sim`, `mstar_m200m_sim`, `mstar_regions`, `mbaryon_regions`, `m200m_sum`, `mstar_moved`, `mstar_fixed`, `n_haloes`, `n_active`, `n_regions`, `s_uncapped_max` | the simulation's aggregates and region sums |
| `k`, `P_mm` | the s = 0 file's (they go with its P_mD, P_DD for the `s0` ends) |
| `P_mm_pass`, `P_mH__<end>`, `P_HH__<end>`, `Nmodes` | pass runs only: the pass's own P_mm and the capped ends' spectra |
| `diag__<end>__*`, `mapdiag__<end>__*` | bookkeeping of the capped ends (`halo_transfer._floor_diag` without its per-halo target check; 2D sums and minima) |
| `check_*`, `explicit_check__<end>`, `check_target__<end>` | the validation numbers of §5 |

Stack file `<stem>_lensing_fstar_obs_lens.npz` (products/2D): `ends` (capped
only), `s__<end>`, `S_mean__<end>`, `G_mean__<end>` (and `_std`), the sample
settings, `explicit_check_{N,T}` and the copied `mapdiag__*`. The uncapped
ends need no stacks: `make_pk_fstar_obs.py` takes them from the s = 0 stacks
(`<stem>_lensing_stellar_lens.npz`, configuration `ap1__M13`).

Modes of `compute_fstar_obs.py`:

- default: solve from the files; without a capped end, save the results file
  (no pass; light); with one, run the pass and build the capped ends.
- `--solve-only`: print s, source and capping of every end; nothing saved.
- `--max-chunks N` (smoke test): read N chunks; the regions are only partly
  read, so their M*/M200m is set to the files' value (to keep realistic s);
  the ends capped in the partial data are built (`fstar__high` if none is);
  nothing saved.
- `--overwrite`: rebuild even if the results file and the capped ends' maps
  exist (otherwise skipped).

Known limitations:

- The skip check uses the file-based capping (float32 table), the saved
  results the pass's (float64). They agreed everywhere here; if they ever
  differ (a region within float32 rounding of the threshold), reruns without
  `--overwrite` would not skip.
- `make_pk_fstar_obs.py --preview` solves from the files for simulations
  without a results file and draws only their uncapped ends; its figures get
  the suffix `preview`.
- The runner `cd`s to a hardcoded repository path (as `runINT_fstar.sh`).

## 5. Validation

All 10 (simulation, redshift) runs; every number is in the `_table.txt` files
and the output files.

| check (what it tests) | result |
|---|---|
| s = 0 file's region table vs its moved stellar mass; s = 0 vs f* files' region stars | ≤ 4.2e-9 (float32 table); ≤ 4.4e-16 |
| pass vs files: regions with stars and ionized gas (same rows), moved and fixed stars, ΣM_b | identical rows; 0 relative difference |
| pass vs float32 table, per region (M*, M_ion) | ≤ 6.0e-8 |
| capping decision, pass (float64) vs table (float32); s | identical everywhere; s ≤ 4.8e-9 |
| aggregate after the change vs target (capped ends) | ≤ 3.3e-16 |
| P_mm of the pass vs the s = 0 file | identical (0) |
| explicit field ρ_m + H vs P_mm + 2 P_mH + P_HH (first capped end per run) | ≤ 1.3e-7 |
| Σ H / stars added; per-region conservation; ionized gas removed ≤ M_ion | ≤ 1.1e-10; ≤ 1.5e-13; max ratio 1 (the capped regions) |
| Σ S, Σ G vs stars added (2D); maps ≥ 0 | ≤ 8.2e-15; min 0 |
| lensing: same halo rows and settings as the s = 0 stacks; explicit maps ionized_gas − G/2, total + (S − G)/2 vs the linear combination | identical rows; ≤ 5.1e-14 |
| pytest `tests/test_fstar_obs.py` (12 synthetic tests: solver vs a root finder, capping, unreachable targets, s > 1 formula, H = −(s − 1) D when uncapped) | pass |

The files of 2026-09-25 hold zeros in the diagnostics
`diag__<end>__{n_no_stars, n_no_ion, mbaryon_no_stars, mbaryon_no_ion}` (the
script passed all-False masks then; fixed afterwards, nothing else affected).
The true values: every selected region has stars and ionized gas, except one
FLAMINGO fiducial halo at z = 0.5 with gas but no stars.

## 6. Results

Figures in `figures/2026-09/09-25/`: `pk_fstar_obs_{z05,z026}_{fstar,mstar_m200m}_S_ap1_M13.pdf`
and tables `pk_fstar_obs_{z05,z026}_table.txt`.

The simulations (1 R200m, ≥ 1e13; z ≈ 0.5 / 0.26): f* = 0.097 / 0.088 (TNG300-1),
0.338 / 0.361 (Illustris-1), 0.183 / 0.175, 0.261 / 0.248, 0.210 / 0.200
(FLAMINGO fiducial, fgas-8sigma, Jet_fgas-4sigma); M*/M200m = 0.0131 / 0.0120,
0.0272 / 0.0261, 0.0216 / 0.0209, 0.0237 / 0.0229, 0.0197 / 0.0192; baryons
within R200m / cosmic 0.86, 0.48, 0.74, 0.57, 0.59 (z ≈ 0.5). All five lie
inside the M*/M200m range, and their f* spread (×3.5) comes mostly from the
gas they keep within R200m.

Stellar scale s of the low / high ends, and the change of S(k ≈ 5) and of
f_gas(1′) at each end (end − simulation), z ≈ 0.5:

| sim | (a) s | (a) ΔS(k≈5) | (a) Δf(1′) | (c) s | (c) ΔS(k≈5) | (c) Δf(1′) |
|---|---|---|---|---|---|---|
| TNG300-1 | 0.52 / 3.10 | −0.008 / +0.035 | +0.027 / −0.115 | 0.77 / 3.06 | −0.004 / +0.034 | +0.013 / −0.113 |
| Illustris-1 | 0.15 / 0.89 | −0.024 / −0.003 | +0.059 / +0.008 | 0.37 / 1.48 | −0.018 / +0.014 | +0.044 / −0.032 |
| FLAMINGO L1_m9 | 0.27 / 1.64 | −0.020 / +0.018 | +0.054 / −0.046 | 0.46 / 1.85 | −0.015 / +0.024 | +0.040 / −0.062 |
| fgas-8sigma | 0.19 / 1.15 | −0.023 / +0.004 | +0.057 / −0.011 | 0.42 / 1.69 | −0.017 / +0.020 | +0.041 / −0.048 |
| Jet_fgas-4sigma | 0.24 / 1.43 | −0.016 / +0.009 | +0.049 / −0.027 | 0.51 / 2.03 | −0.011 / +0.022 | +0.031 / −0.064 |

z ≈ 0.26: (a) TNG300-1 −0.007 / +0.041 (Δf(1′) +0.012 / −0.064), Illustris-1
−0.025 / −0.005, FLAMINGO −0.018 to −0.025 / +0.007 to +0.021; (c) TNG300-1
−0.003 / +0.040, Illustris-1 −0.018 / +0.017, FLAMINGO −0.011 to −0.018 /
+0.024 to +0.028. Capped regions (share of the changing stars): at most 0.12% except at
the (c) high end of Illustris-1 (5% at z ≈ 0.5, 26% at z ≈ 0.26),
fgas-8sigma (1.6% / 3.5%) and Jet_fgas-4sigma (0.9% / 1.7%).

Reading:

- **The bands are narrower than the f* = 0 → floor bands** of
  `pk_fstar_transfer.md` at the same configuration (1 R200m, ≥ 1e13, z ≈ 0.5:
  S(k≈5) width 0.021–0.042 for (a) and 0.031–0.038 for (c), vs 0.041–0.077
  there; about half for (a), half to three quarters for (c)).
- **(c) gives similar widths in every simulation** (ΔS(k≈5) 0.031–0.038 at
  z ≈ 0.5): every simulation starts inside the M*/M200m range, and a ×4
  range in stellar mass is ×4 for all. **(a) depends on the gas**: TNG300-1
  (star-poor, gas-rich) must triple its stars to reach f* = 0.30, while
  Illustris-1 (gas-poor) is above 0.30 already, so both of its ends lower
  the stars and the band lies below the simulation.
- The stellar bands are small compared with the spread between the
  simulations (S(k≈5) from 0.74 to 0.89 at z ≈ 0.5), but not negligible
  for a single simulation: the high end of TNG300-1 (s ≈ 3) removes about a
  third of its suppression at k ≈ 5.
- As for the other stellar models, more stars means more small-scale power
  and a lower lensing f_gas at every θ.

## 7. Reproduce

```bash
cd scripts/
python unbound_gas/compute_fstar_obs.py -p configs/unbound_gas/pk_fstar_obs_z05.yaml --solve-only
python unbound_gas/compute_fstar_obs.py -p configs/unbound_gas/pk_fstar_obs_z05.yaml --sims TNG300-1   # no pass needed
salloc -q interactive -C cpu -N 4 -t 2:30:00 -A desi --no-shell
SLURM_JOB_ID=<id> SLURM_JOB_NUM_NODES=4 bash unbound_gas/runINT_fstar_obs.sh      # default RUNS: the capped runs
python unbound_gas/make_pk_fstar_obs.py -p configs/unbound_gas/pk_fstar_obs_z05.yaml
python unbound_gas/make_pk_fstar_obs.py -p configs/unbound_gas/pk_fstar_obs_z026.yaml
cd ../tests && pytest test_fstar_obs.py -v
```

## 8. Open items

- Targets are preliminary (user, 2026-09-25): f* 0.05–0.30, M*/M200m 0.01–0.04.
  After changing them, `compute_fstar_obs.py --solve-only` shows which ends
  are capped; rerun with `--overwrite` (on the login node when no end is
  capped, else `runINT_fstar_obs.sh` with `EXTRA=--overwrite`), then
  `make_pk_fstar_obs.py`.
- Options (b) envelope + ICL only and (d) IMF-like uniform scaling are
  deferred (`NOTES/unbound_gas/handoff_physical_stellar_band.md`).
