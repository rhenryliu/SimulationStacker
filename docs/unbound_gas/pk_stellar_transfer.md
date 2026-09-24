# Halo-level stellar-to-ionized-gas transfer and the P(k) suppression (unbound gas paper)

**Status as of 2026-09-24.** Code in `src/halo_transfer.py` and
`scripts/unbound_gas/{compute_pk_stellar.py, make_pk_stellar.py, runINT_pk_stellar.sh}`,
config block `stellar:` in `scripts/configs/unbound_gas/pk_components_z05.yaml`,
tests in `tests/test_halo_transfer.py`; small additions to `src/mapMaker.py`
(ionized-mass helper, behaviour unchanged) and `src/loadIO.py` (membership
readers). z ≈ 0.5 committed as `30a6e34`; spectra exist for all six
simulations (§5). z ≈ 0.26 (FLAMINGO z = 0.30) added the same day for five
simulations, SIMBA excluded (§9; config `configs/unbound_gas/pk_stellar_z026.yaml`).
The bottom panel of the `_stellar_S_` figures shows the lensing f_gas(θ) of
`lensing/beam_compensated_ratio_v2.py` under the same transfer (§10;
`compute_stellar_maps.py`, `stack_stellar_maps.py`, `lensing:` config blocks).
Companion to `pk_suppression_components.md` (the global and
local models, unchanged). Working logs:
`NOTES/unbound_gas/session_log_2026-09-23_pk_stellar_transfer.md` and
`session_log_2026-09-24_stellar_lensing_observable.md` (untracked).

## 1. Question and model

The kSZ effect constrains the ionized gas; the baryonic mass locked in stars
is less directly constrained. How much does the baryonic suppression of the
matter power spectrum depend on it? Counterfactual: convert some or all of the
stellar mass associated with haloes into ionized gas, conserving baryonic
mass halo by halo, and lay it out like that halo's existing ionized gas. This
is a sensitivity test, not a feedback model.

For a halo region h with star particles S_h and gas particles G_h,

    M*_h = Σ_{S_h} m*_j ,   M_ion,h = Σ_{G_h} m_ion,i .

The **stellar scale s** is the fraction of the selected stellar mass kept as
stars (s = 1: the simulation; s = 0: all selected stellar mass converted).
Every star keeps s m*_j, and every gas particle of the region gains
(1 − s) M*_h m_ion,i / M_ion,h. The mass moved out of the stars equals the mass
added to the gas, region by region; nothing outside the selected regions
changes, and nothing is renormalised globally. The matter field is

    ρ_m(s) = ρ_m + (1 − s) D ,   D = TSC[Σ_h (M*_h/M_ion,h) m_ion,i] − TSC[Σ_h m*_j] ,

with D a zero-mass *transfer field* (s = 0) deposited with the same TSC as the
cached fields. With δ_D = D / ρ̄_m (ρ̄_m unchanged),

    P_mm(s) = P_mm + 2 (1 − s) P_mD + (1 − s)² P_DD      (exact),

so one D per configuration gives every s (the halo-level analogue of the
α-model algebra). Quantities, as in the companion doc (P_DMO = the matched DMO
run):

| symbol | definition |
|---|---|
| Q(k; s) | P_mm(s) / P_mm − 1 (= P_modified / P_original − 1) |
| S(k; s) | P_mm(s) / P_DMO |
| ΔS(k; s) | S(s) − S(1) |

Two ways to define the regions (the **only** difference between the methods;
same haloes, stars, gas, mass cuts, s and grid):

- **Halo-finder membership (`fof`).** The finder's own particle membership.
  Each particle belongs to at most one region, and satellites belong to their
  host (their stars follow the host's ionized gas).
- **Radial aperture (`ap1`, `ap2`).** All particles within x R200m of the halo
  centre, x = 1, 2. Overlaps: a particle belongs to the **most massive**
  selected halo whose aperture contains it (mass priority; periodic
  minimum-image distances). Every particle is modified at most once, and the
  labels are nested in the mass cut (raising the cut only unassigns
  particles), so all cuts come from one labelling.

Why not the existing models: the global α model lays the stars out like the
box-wide ionized gas, whose large-scale bias differs from the stars', so it
shifts P even as k → 0 (−0.6 to −1.25%; SIMBA +2.6%). The local model is
halo-agnostic and has a free transport radius and kernel.

## 2. Definitions and data model

| item | choice |
|---|---|
| haloes | rows of `SimulationStacker.loadHalos()`: FoF groups (TNG/Illustris), CAESAR haloes (SIMBA), SOAP-HBT centrals (FLAMINGO); hosts only |
| mass cuts | FoF `GroupMass` ≥ 1e11, 1e12, 1e13 M⊙/h (the quantity the repo's halo selection uses) |
| aperture radius, centre | `GroupRad` = R200m (`Group_R_Mean200` / CAESAR `virial_quantities.r200` / SOAP `SO/200_mean/SORadius`), `GroupPos`; haloes with R200m ≤ 0 claim nothing |
| stars | PartType4 **without** TNG/Illustris wind-phase particles (`GFM_StellarFormationTime ≤ 0`); SIMBA and FLAMINGO PartType4 are all stars |
| ionized gas | the pipeline's definition, `mapMaker.ionized_gas_masses`: M_ion = N_e m_p μ_e, μ_e = 2/(1+X_H), X_H = 0.76 (TNG/Illustris/SIMBA: 0.864 x_e M_gas; FLAMINGO: from `ElectronNumberDensities`, zero for star-forming gas) |
| regions with stars but no ionized gas | stars stay in place (reported; ≤ 1.8e-4 of the stars, see §5) |
| periodic boundaries | membership is not geometric; aperture distances are minimum-image; TSC deposits wrap |
| particle masses | variable in all suites; everything is mass-weighted |

Membership sources (checked against the data, 2026-09-23):

| suite | membership | gotcha |
|---|---|---|
| TNG300-1, Illustris-1 | snapshots are FoF-ordered: group g holds global indices [ends[g−1], ends[g]), ends = cumsum(`GroupLenType`); chunk files read in numeric order | `GroupMassType[:,4]` excludes winds, the cached `Stars` field includes them (~8% of Illustris-1's, 0.5% of TNG300-1's) |
| SIMBA | CAESAR `halo_data/lists/{slist,glist}` with `*_start/_end` (disjoint) | the snapshot's `HaloID` is a different numbering |
| FLAMINGO | `FOFGroupIDs` from `membership_0067/membership_0067.{i}.hdf5` (chunk-aligned), mapped to the central via SOAP `InputHalos/HBTplus/HostFOFId` (one central per FoF group) | the raw chunks' `FOFGroupIDs` come from a different (on-the-fly) FoF run; "no group" = 2147483647 |

Numerics: stored particle positions and masses are float32 (tsc_parallel
casts weights to the position dtype); per-halo sums are float64; D is
accumulated in float64 and cast to float32 for the FFT, like the other
fields. `halo_transfer.deposit` caps tsc_parallel's thread count at n/8: its
parallel stripes must be ≳ 3 cells wide, and at 2-cell stripes (e.g. 256
threads on a 1000³ grid) it silently loses mass.

Shot noise: nothing in the pipeline subtracts it. The transfer changes the
Poisson term V Σm²/M² by ≤ 5e-5 of P at the Nyquist frequency (FLAMINGO; ≤ 6e-6 for the others; §4); no
treatment change is needed.

## 3. Scripts (run from `scripts/`)

| script | does | runs on | output |
|---|---|---|---|
| `src/halo_transfer.py` | membership and aperture labels, per-halo bookkeeping, `collect_particles` (one pass over stars and gas), `transfer_field` (D and diagnostics), `p_of_scale`; `pixel_index_2d`, `transfer_maps_2d` (the 2D maps A, B of §10) | — | — |
| `compute_pk_stellar.py` | per simulation: one particle pass, then for each method × mass cut D and its spectra (numba estimator of `compute_pk_local.py`, identical to Pylians); validation (§4); `--variants`, `--overwrite`, `--max-chunks N` (smoke test, nothing saved) | CPU node | `<stem>_Pk_stellar_<variant>_<n>.npz` |
| `make_pk_stellar.py` | Q for every s, method comparison at s = 0, S bands with the lensing f_gas(θ) below (§10; `--bottom dS` restores the earlier ΔS panel) (the Q-for-every-s and S-band figures once per mass cut), tables; the local stars-only model as context in the figures, local and global in the table | login | `<fig>_stellar_scales_<variant>_<tag>`, `_stellar_methods`, `_stellar_S_<variant>_<tag>`, `_stellar_table.txt`, `_stellar_lensing_table.txt` |
| `runINT_pk_stellar.sh` | `compute_pk_stellar.py`, one simulation per node; `SIMS`, `EXTRA`, `CONFIG` env vars | `salloc -N ≤ 4` | logs in `../Outputs_Perlmutter/` |
| `compute_stellar_maps.py` | per simulation: one particle pass, then for each method × mass cut the 2D maps of the ionized gas added (A) and stars removed (B) at s = 0 on the lensing map grid (§10); `--variants`, `--overwrite`, `--max-chunks N` | CPU node | `<stem>_stellar_{added,removed}_<variant>_<tag>_<n>_yz.npy`, `<stem>_stellar_maps_<n>_yz.npz` (products/2D) |
| `stack_stellar_maps.py` | ΔΣ stacks of the cached ionized_gas/total maps and of A, B on the lensing sample, in a process pool; `--max-halos N` (quick check, nothing saved) | CPU node (small boxes: login) | `<stem>_lensing_stellar_<sample_name>.npz` (products/2D) |
| `runINT_stellar_maps.sh` | both of the above, one simulation per node; `SIMS`, `STAGES`, `EXTRA`, `CONFIG` | `salloc -N ≤ 4` | logs `stellar_maps-<job>_<sim>.out` |

Config (`stellar:` block; ignored by every other script): `methods`
(`membership`, `aperture`), `aperture_radii` (x, units of R200m),
`halo_mass_min` (FoF GroupMass cuts, M⊙/h), `stellar_scales` (plots only).
Optional `plot.z_label` titles the figures; `plot.exclude_sims` (matched like
`--sims`) leaves simulations out of the figures but not the tables (z ≈ 0.5:
SIMBA, for now). Figures stop at k = 5 h/Mpc
(`--kmax`), have grid lines, and colour the simulations as the lensing P(k)
suppression figure (`lensing/plot_pk_suppression.py`: twilight for TNG300-1 /
Illustris-1, plasma for SIMBA, fixed FLAMINGO colours); in the per-simulation
panel figures colour encodes s (`_scales_`) or the method (`_methods_`). Where no `_Pk_components_` file
exists (z ≈ 0.26), `compute_pk_stellar.py` checks its estimator against the
Pylians `P_total` of the `_Pk_dmo_` file, skips the negative-stellar-mass check
when there is no Stars cache, and `make_pk_stellar.py` leaves the table's
global column empty and takes the baryon budget from the particle pass (PartType4 + gas; BH,
1e-4–3e-3 of the baryons, not included).

Spectra file `<stem>_Pk_stellar_<variant>_<n>.npz` (variant `fof`, `ap1`,
`ap2`; tag `M11`, `M12`, `M13` = log10 of the cut): `k`, `Nmodes`, `P_mm`
(estimator, cached total field), `mean_total`, `mass_total`, box totals of the
particle pass (`mstar_box` true stars, `mwind_box`, `mgas_box`, `mion_box`,
sums of squared masses), `mstar_cache`; per tag `P_mD__<tag>`, `P_DD__<tag>`
and `diag__<tag>__<name>` (haloes selected/active/kept, moved and kept
stellar mass, conservation numbers, share of added mass on star-forming gas,
M*-weighted percentiles of M*_h/M_ion,h, shot-noise sums); validation numbers
(§4); and per-halo `halo_rows`, `halo_mstar`, `halo_mion`, `halo_gmass` of the
active haloes at the lowest cut. Same k bins as every other spectra file.

## 4. Validation

All six simulations; every number is in `_stellar_table.txt` and in the
spectra files. "moved" = stellar mass moved at s = 0.

| check (what it tests) | TNG300-1 | Illustris-1 | SIMBA | FLAMINGO ×3 |
|---|---|---|---|---|
| estimator P_mm vs Pylians P_total (s = 1 reproduces the simulation) | 9.3e-8 | 7.2e-8 | 8.1e-8 | ≤ 1.2e-7 |
| explicit field ρ_m + (1−s)D vs the quadratic formula, s = 0 and 0.5 | ≤ 3.6e-8 | ≤ 9.7e-8 | ≤ 7.7e-8 | ≤ 4.7e-8 |
| total and baryon mass: Σ D on the grid / moved | ≤ 6e-11 | ≤ 3e-11 | ≤ 4e-10 | ≤ 9e-12 |
| per-halo conservation, max over haloes of \|added − moved\| / M*_h | 2e-13 | 1e-13 | 1e-11 | ≤ 7e-14 |
| membership mapping: per-halo M*_h vs catalogue (median / 99%) | 2e-8 / 5e-8 | 7e-7 / 4e-6 | 1.4e-7 / 2.2e-6 | — (no FoF stellar mass in SOAP) |
| modified stars vs cached Stars: negative mass / moved | −3.5e-5 | −3.6e-5 | −3e-7 | ≤ −1.9e-5 |
| change of the Poisson term at s = 0, / P(k_Nyq) | ≤ 4e-7 | ≤ 1e-6 | ≤ 6e-6 | ≤ 5e-5 |
| large scales: mode-weighted Q(s=0) over k ≤ 3 k_F, all 9 configurations | ≤ 5e-4 % | ≤ 9e-3 % | ≤ 1.3e-3 % | ≤ 6e-4 % |
| same for the global stars-only model (for comparison) | −0.62% | −1.37% | −0.65% | −0.88 to −1.08% |

- s = 1 is the simulation by construction (D enters with (1 − s)); the
  non-trivial parts are the estimator check and the explicit-field check.
- DM, neutral gas and BH are untouched by construction; D is zero outside the
  selected regions (up to the TSC cloud).
- The small negative stellar residuals are not mass loss: the stored
  positions are float32 (~1e-4 of a cell in the large boxes), so the removed
  stars' TSC weights differ very slightly from the cache's; TNG/Illustris
  also carry the caches' ~4e-5 stellar deficit (companion doc §5).
- pytest (`tests/test_halo_transfer.py`, 12 synthetic tests): aperture labels
  equal a brute-force mass-priority assignment (including periodic faces and
  a coarse hash grid), nesting in the mass cut, group-end and FoF-ID
  mappings, per-halo conservation, the kept-halo fallback, D = direct
  deposit, and the quadratic identity.
- `mapMaker.ionized_gas_masses` is bitwise identical to the previous inline
  formula on one chunk each of TNG300-1, Illustris-1, SIMBA and FLAMINGO.

## 5. Results snapshot (2026-09-24)

Figures in `figures/2026-09/09-24/`: `pk_components_z05_stellar_scales_fof_M{11,12,13}.pdf`
(Q for s = 0.75 … 0, one per mass cut), `_stellar_methods.pdf` (every method
and cut at s = 0), `_stellar_S_{fof,ap1,ap2}_M{11,12,13}.pdf` (S bands with
ΔS, one per method and cut), and `_stellar_table.txt`. "global" = all (cached) stars like the box-wide ionized
gas (table only; removed from the figures at the user's request, 2026-09-24);
"local R=1" = all stars transported within a 1 Mpc/h sphere like the local
ionized gas (make_pk_local's `stars` set; dash-dot in the figures).

Stars moved at s = 0 (fraction of all true stars), membership, by cut:

| sim | ≥ 1e11 | ≥ 1e12 | ≥ 1e13 | stellar share of baryons: sim → fof/M11, s = 0 |
|---|---|---|---|---|
| TNG300-1 | 0.985 | 0.789 | 0.370 | 0.0283 → 0.0006 |
| Illustris-1 | 0.936 | 0.700 | 0.312 | 0.0588 → 0.0081 (winds stay) |
| SIMBA-100 (prov.) | 0.976 | 0.729 | 0.326 | 0.0329 → 0.0008 |
| FLAMINGO L1_m9 | 0.996 | 0.721 | 0.305 | 0.0549 → 0.0002 |
| fgas-8sigma | 0.996 | 0.696 | 0.286 | 0.0606 → 0.0002 |
| Jet_fgas-4sigma | 0.986 | 0.628 | 0.253 | 0.0578 → 0.0008 |

Apertures move 0.84–0.92 (1 R200m) and 0.92–0.99 (2 R200m) of the stars at
≥ 1e11. Haloes with stars but no ionized gas: none in TNG/Illustris/SIMBA,
≤ 1.8e-4 of the stars in FLAMINGO.

Q(s = 0) [%] at k ≈ 1 and 5 h/Mpc (cut ≥ 1e11 unless noted):

| sim | fof k≈1 | fof k≈5 | fof ≥1e13, k≈5 | 1 R200m k≈5 | 2 R200m k≈5 | local R=1, k≈5 | global k≈1 | global k≈5 |
|---|---|---|---|---|---|---|---|---|
| TNG300-1 | −0.01 | −1.54 | −1.39 | −1.98 | −2.76 | −2.51 | −1.25 | −3.63 |
| Illustris-1 | −0.05 | −3.49 | −3.09 | −4.14 | −6.34 | −6.49 | −2.76 | −7.75 |
| SIMBA-100 (prov.) | −0.05 | −2.73 | −2.51 | −1.64 | −3.33 | −4.12 | −1.61 | −5.29 |
| FLAMINGO L1_m9 | −0.04 | −2.96 | −2.64 | −3.45 | −5.03 | −4.52 | −2.04 | −6.21 |
| fgas-8sigma | −0.06 | −3.42 | −3.08 | −3.84 | −5.67 | −5.32 | −2.42 | −6.89 |
| Jet_fgas-4sigma | −0.03 | −2.49 | −2.21 | −2.92 | −4.37 | −4.01 | −1.99 | −5.55 |

ΔS(k ≈ 5, s = 0): membership −0.014 (TNG300-1), −0.026 (Illustris-1), −0.021
(SIMBA), −0.026 / −0.028 / −0.020 (FLAMINGO fiducial / fgas-8sigma /
Jet_fgas-4sigma); 2 R200m −0.025 to −0.047; global stars-only −0.032 to −0.057.

Reading:

- **No large-scale offset.** Q → 0 as k → 0 in every configuration (§4),
  unlike the global model; at k ≈ 1 the halo-level effect is ≤ 0.06%
  (membership) and ≤ 0.5% (2 R200m).
- **The resolved effect is set by group and cluster stars.** Haloes ≥ 1e13
  hold 25–37% of the stars but give 88–92% of the ≥ 1e11 effect at k ≈ 5
  (78–82% at k ≈ 10 for the 1000³ runs): in lower-mass haloes the stars and
  ionized gas sit within ~1–2 grid cells, so moving mass inside them hardly
  changes P at k ≲ 10 h/Mpc.
- **Nearly linear in the converted mass:** Q(s = 0.5) / Q(s = 0) = 0.50–0.51 at
  k ≈ 5; the (1 − s)² P_DD term is small there.
- **Method dependence** (the region's extent sets how far the mass moves):
  1 R200m gives 1.1–1.3× the membership Q at k ≈ 5 (SIMBA the exception at
  0.6×, not investigated); 2 R200m gives 1.2–1.8×. Relative to the global
  stars-only model, membership gives 43–52% and 2 R200m 63–82% of its Q at
  k ≈ 5. The local R = 1 Mpc/h field transport lies within 25% of the 2 R200m
  result.

## 6. Caveats

- **SIMBA is provisional** (ElectronAbundance convention, see the companion
  doc): the per-particle ionized-gas template inside each halo would change
  under the correction.
- **Star-forming gas counts as ionized in TNG/Illustris/SIMBA** (x_e ≈ 1) but
  not in FLAMINGO. Share of the added mass placed on star-forming gas
  (membership, ≥ 1e11): 1.9% TNG300-1, 10.7% Illustris-1, 3.7% SIMBA, 0
  FLAMINGO. That mass stays near the removed stars.
- **Gas-poor haloes:** moved mass in haloes with M*_h > M_ion,h (their ionized
  gas more than doubles at s = 0), membership ≥ 1e11: 1% TNG300-1, 39%
  Illustris-1, 28% SIMBA, 31% / 49% / 37% FLAMINGO; M*-weighted median of
  M*_h/M_ion,h 0.19 (TNG300-1) to 0.99 (fgas-8sigma).
- **Winds** are excluded from the moved stars but included in the context
  models (global, local R=1), which move the cached Stars field: for
  Illustris-1 those models move ~8% more mass.
- **Resolution:** transfers within ≲ 1–2 cells are invisible below the Nyquist
  frequency (FLAMINGO: 0.34 Mpc/h cells, k_Nyq = 9.2 h/Mpc), so the low-mass
  haloes' stars barely enter; this is a property of the grid, not the model.
- Regions: membership includes FoF bridges and outskirts; with mass priority a
  2 R200m aperture absorbs neighbouring galaxies into the bigger halo's
  region. FLAMINGO stars in FoF groups with no SOAP central stay in place
  (part of 1 − f_moved). Aperture centres are `GroupPos` as elsewhere in the
  repo (SIMBA CAESAR `pos`).
- **tsc_parallel thread hazard (possible pre-existing effect).** Stripes
  narrower than ~3 cells lose mass silently; stripe width ≈ n / (2 ×
  NUMBA_NUM_THREADS at start-up). All runs here used 128 threads (3.9- and
  7.8-cell stripes) and `deposit` now caps the thread count. Hypothesis, not
  verified: 1000³ caches built without the thread export (256 threads →
  2-cell stripes) would carry small deficits like the TNG300-1/Illustris-1
  Stars caches' ~4e-5 and the ~400 inconsistent SIMBA void cells
  (companion doc §4–5); the FLAMINGO 2000³ rebuild was bit-identical.
  Supporting (not conclusive): against the TNG300-1 snap-80 Stars cache, built
  on 2026-09-23 with the 128-thread export, the negative-stellar residual is
  −5e-6 of the moved mass (§9), against −3.5e-5 for the old snap-67 cache.

## 7. Reproduce

```bash
cd scripts/
salloc -q interactive -C cpu -N 4 -t 2:30:00 -A desi --no-shell
SLURM_JOB_ID=<id> SLURM_JOB_NUM_NODES=4 \
    SIMS="L1_m9 fgas-8sigma Jet_fgas-4sigma TNG300-1 Illustris-1 m100n1024" \
    bash unbound_gas/runINT_pk_stellar.sh
python unbound_gas/make_pk_stellar.py -p configs/unbound_gas/pk_components_z05.yaml
cd ../tests && pytest test_halo_transfer.py -v      # 12 synthetic tests, ~20 s
```

Measured cost (one node each; 9 configurations): FLAMINGO ~72 min per variant
(particle pass 48 min, 220–228 GB peak); TNG300-1 70 min (pass 29 min for
600 chunks, ~4e9 kept gas particles, 240 GB peak); Illustris-1 17 min
(184 GB); SIMBA-100 5 min (121 GB). A smoke test: `EXTRA="--max-chunks 2"`.

z ≈ 0.26 (§9):

```bash
SLURM_JOB_ID=<id> SLURM_JOB_NUM_NODES=4 CONFIG=configs/unbound_gas/pk_stellar_z026.yaml \
    SIMS="Illustris-1 L1_m9 fgas-8sigma Jet_fgas-4sigma TNG300-1" \
    bash unbound_gas/runINT_pk_stellar.sh
python unbound_gas/make_pk_stellar.py -p configs/unbound_gas/pk_stellar_z026.yaml
```

Lensing panel (§10), per redshift (`CONFIG` as above for z ≈ 0.26; SIMBA only
at z ≈ 0.5):

```bash
salloc -q interactive -C cpu -N 4 -t 1:45:00 -A desi --no-shell
SLURM_JOB_ID=<id> SLURM_JOB_NUM_NODES=4 \
    SIMS="L1_m9 fgas-8sigma Jet_fgas-4sigma TNG300-1 Illustris-1 m100n1024" \
    bash unbound_gas/runINT_stellar_maps.sh
python unbound_gas/make_pk_stellar.py -p configs/unbound_gas/pk_components_z05.yaml
cd ../tests && pytest test_stellar_maps.py -v       # 19 synthetic tests, ~25 s
```

Another stacking sample: set `lensing.overrides` (e.g.
`{halo_abundance_target: 2.0e-4}`) and a new `lensing.sample_name` in the
config, then run only the stacks (`STAGES=stack`; the maps are reused) and
`make_pk_stellar.py`. The earlier ΔS bottom panel: `make_pk_stellar.py
--bottom dS`.

## 8. Open items (2026-09-24)

- Which configuration(s) go in the paper: membership ≥ 1e11 as the baseline
  with the 1–2 R200m apertures as the spread is one option; the mass-cut
  result (groups and clusters dominate) is worth a sentence either way.
- SIMBA: recompute with `compute_pk_stellar.py --sims m100n1024 --overwrite`
  once the ElectronAbundance correction is adopted (it would enter through
  `mapMaker.ionized_gas_masses`), and drop the "(provisional)" label.
- The tsc_parallel hypothesis for the old 1000³ caches (§6) is untested; a
  rebuild of one cache at 256 vs 128 threads would settle it.
- The SIMBA 1 R200m < membership inversion is not understood (CAESAR halo
  extent or centre definition are candidates).
- z ≈ 0.26 has no global or local context curves and no SIMBA (user decisions
  2026-09-24): computing them needs the missing ionized-gas and component
  caches (§9).
- Lensing panel (§10): the stacked sample is the lensing SHAM sample for now
  (user decision 2026-09-24, may change: `lensing.overrides`); only
  ionized_gas/total (a baryon/total variant would need one more stack per
  simulation, the maps are reused); SIMBA provisional as above.

## 9. z ≈ 0.26 (FLAMINGO z = 0.30)

Same analysis at the lensing low-z snapshots, config
`configs/unbound_gas/pk_stellar_z026.yaml` (snapshots and DMO references as in
`configs/lensing/pk_dmo_z026.yaml`, whose run wrote the hydro `total`/`Stars`
caches and the `_Pk_dmo_` spectra used here):

| sim | snapshot | z | DMO reference | notes |
|---|---|---|---|---|
| TNG300-1 | 80 | 0.2613 | TNG300-1-Dark 80 | a TNG mini snapshot; every field used exists and the FoF ordering holds (catalogue check below) |
| Illustris-1 | 116 | 0.2613 | Illustris-1-Dark 116 | no Stars cache: negative-stellar check skipped |
| FLAMINGO ×3 | 71 | 0.30 | L1_m9_DMO 71 | the only low-z snapshot with a DMO run (72, z = 0.25, is fiducial-only and has none) |

Left out (user decisions, 2026-09-24): SIMBA (its public DM-only run has no
snapshot near z = 0.27, so no S(k)); the global and local context curves (no
component or local-model spectra exist at these snapshots).

Validation (all 45 configurations): estimator vs the DMO file's Pylians
P_total ≤ 1.2e-7; explicit field vs formula ≤ 8.4e-8; Σ D / moved ≤ 1e-10;
per-halo conservation ≤ 2e-13; catalogue M*_h median 2e-8 (TNG300-1) and
7e-7 (Illustris-1); negative stellar mass −6e-6 (TNG300-1), ≤ −1.9e-5
(FLAMINGO); Poisson change ≤ 5e-5 of P(k_Nyq); large-scale Q ≤ 0.014%
(largest for Illustris-1's apertures; ≤ 5e-4 % for TNG300-1 and FLAMINGO). Stars kept for lack of ionized
gas: none in TNG/Illustris, ≤ 2.6e-4 of the stars in FLAMINGO.

Figures in `figures/2026-09/09-24/`: `pk_stellar_z026_stellar_scales_fof_M{11,12,13}.pdf`,
`_stellar_methods.pdf`, `_stellar_S_{fof,ap1,ap2}_M{11,12,13}.pdf`, `_stellar_table.txt`.

Q(s = 0) [%] at k ≈ 5 h/Mpc, z ≈ 0.26 (z ≈ 0.5 in brackets); cut ≥ 1e11
unless noted:

| sim | membership | membership ≥1e13 | 1 R200m | 2 R200m | ΔS(k≈5), membership |
|---|---|---|---|---|---|
| TNG300-1 | −1.54 (−1.54) | −1.43 (−1.39) | −2.02 (−1.98) | −2.67 (−2.76) | −0.014 (−0.014) |
| Illustris-1 | −3.97 (−3.49) | −3.65 (−3.09) | −4.55 (−4.14) | −6.45 (−6.34) | −0.027 (−0.026) |
| FLAMINGO L1_m9 | −3.09 (−2.96) | −2.85 (−2.64) | −3.65 (−3.45) | −5.05 (−5.03) | −0.027 (−0.026) |
| fgas-8sigma | −3.69 (−3.42) | −3.42 (−3.08) | −4.18 (−3.84) | −5.79 (−5.67) | −0.030 (−0.028) |
| Jet_fgas-4sigma | −2.67 (−2.49) | −2.44 (−2.21) | −3.15 (−2.92) | −4.43 (−4.37) | −0.022 (−0.020) |

Stars moved (membership ≥ 1e11 / ≥ 1e13): 0.988/0.422 (TNG300-1),
0.940/0.353 (Illustris-1), 0.996/0.343, 0.996/0.323, 0.986/0.290 (FLAMINGO).
Stellar share of the baryons (particle-pass budget): 0.0305, 0.0637 (PartType4
incl. winds), 0.0596, 0.0659, 0.0622.

Reading: the halo-level effect barely changes between z ≈ 0.5 and z ≈ 0.26.
With the membership regions it grows by 0 (TNG300-1) to 14% (Illustris-1) at
k ≈ 5 while the stellar mass grows by 7–11%. With the 2 R200m apertures it
changes by ≤ 3.3%. Everything found at z ≈ 0.5 holds:
- no large-scale offset;
- ≥ 1e13 haloes give 91–93% of the effect at k ≈ 5;
- Q(s = 0.5)/Q(s = 0) = 0.50;
- the regions rank the same (1 R200m 1.1–1.3×, 2 R200m 1.6–1.7× membership).

Cost: FLAMINGO 72–74 min per variant (223–226 GB), TNG300-1 74 min (264 GB),
Illustris-1 17 min (123 GB); allocation 58827894, 4 nodes, 1 h 31 min.


## 10. The lensing observable under the transfer (bottom panel of the S-band figures)

The bottom panel of every `_stellar_S_<variant>_<tag>` figure shows the
beam-free simulation curve of `lensing/beam_compensated_ratio_v2.py`, the
stacked ionized-gas to total mass ratio

    f(θ) = ⟨ΔΣ[ionized_gas]⟩(θ) / ⟨ΔΣ[total]⟩(θ) × Ω_m/Ω_b ,

a ratio of halo means of the compensated ΔΣ filter (`filters.delta_sigma_kernel`)
on the cached 2D maps (0.2′ pixels, yz projection, θ = 1–6′ in 9 steps), and
how it changes under the **same box-wide transfer** as the top panel (every
host halo above the figure's mass cut, the figure's method). The filter is a
sum of map × kernel, so it is linear in the map. With A = the ionized-gas
mass added and B = the stellar mass removed at s = 0 (2D maps), N and T the
halo-mean ΔΣ of the cached ionized_gas and total maps, and t = 1 − s,

    f(s) = [N + t ΔΣ_A] / [T + t (ΔΣ_A − ΔΣ_B)] × Ω_m/Ω_b      (exact),

so one (A, B) pair per configuration gives every s (f is a ratio, not linear
in s). The panel shows s = 1 (solid, markers) and s = 0 (dashed) with the band
between, and the lensing script's beam-compensated measurement
(`../data/beam_compensated/beam_compensated_ionized_gas_total_z{0.5,0.26}.npz`,
black squares). The simulation curves carry no error bands (user decision).

Definitions (user decisions 2026-09-24):

| item | choice |
|---|---|
| stacked sample | the lensing script's, **fixed for every s** (the same observed galaxies): SHAM on `SubhaloMStar` among subhaloes with parent GroupMass ≤ 5e14 M⊙/h (`stackMap` default), abundance 5e-4 (z ≈ 0.5) / 1e-3 (h/Mpc)³ (z ≈ 0.26); stacked at `SubhaloPos` |
| transfer | exactly the P(k) one (§1–2): per configuration, A = Σ f_h m_ion,i and B = Σ m*_j over the active haloes, from the same particle pass and bookkeeping |
| grid, redshift | the lensing config's (`lensing.config`): z = 0.5 / 0.26 for TNG300-1 and Illustris-1 (not 0.2613: the map grid depends on z), FLAMINGO 0.5 / 0.3; nPixels as `SimulationStacker.makeMap` |
| binning | the cached 2D fields' `binned_statistic_2d` sum (each kept particle's pixel from the same scipy call, `halo_transfer.pixel_index_2d`) |
| Ω_b | snapshot header, else the lensing script's fallback (Illustris-1: 0.0456) |
| simulations | as the top panel: six computed at z ≈ 0.5, five at z ≈ 0.26 (no SIMBA); SIMBA is left out of the z ≈ 0.5 figures for now (`plot.exclude_sims`, user decision 2026-09-24), its numbers stay in the tables |

Config (`lensing:` block in `pk_components_z05.yaml` / `pk_stellar_z026.yaml`):
`config` (the lensing noBeam config: settings, redshifts, sample), `data`,
`sample_name` (output tag) and `overrides` (any stacking setting, e.g.
`halo_abundance_target`). The maps do not depend on the sample, so another
sample needs only the stacking step; each sample's stacks sit in their own
file, and `make_pk_stellar.py` refuses stacks whose stored settings differ
from the config's.

Files (products/2D, new): `<stem>_stellar_added_<variant>_<tag>_<n>_yz.npy`,
`<stem>_stellar_removed_<variant>_<tag>_<n>_yz.npy` (float64, M⊙/h per pixel;
18 per simulation), `<stem>_stellar_maps_<n>_yz.npz` (bookkeeping),
`<stem>_lensing_stellar_<sample_name>.npz` (θ, N, T, per configuration ΔΣ_A
and ΔΣ_B halo means and scatter, Ω_b, Ω_m, halo rows, settings, checks).

### Validation

All 54 configurations at z ≈ 0.5 (six simulations × 9); z ≈ 0.26 below.

| check (what it tests) | result |
|---|---|
| the sample selected here + `stack_on_array` vs the lensing script's `stackMap`, cached ionized_gas and total maps (same haloes, order, map; s = 1 reproduces the lensing curve) | 0 (identical; the script refuses to save otherwise) |
| explicitly built maps ionized_gas + A/2 and total + (A − B)/2 stacked vs the linear combination (s = 0.5, fof ≥ 1e11) | ≤ 3.0e-14 |
| Σ A / moved − 1, Σ B / moved − 1 (2D mass conservation) | ≤ 5.5e-15, ≤ 1.1e-15 |
| per-halo conservation of the weights | ≤ 1.3e-11 |
| moved stellar mass and active haloes vs the 3D run (`_Pk_stellar_` files) | identical (0, 0) in every configuration |
| cached 2D Stars map − B, negative mass / moved | TNG300-1 −6.4e-5, Illustris-1 −8.4e-6, SIMBA −3e-10, FLAMINGO ≤ −3.9e-4 |
| pytest `tests/test_stellar_maps.py` (19 synthetic tests) | pass |

- The negative residual is float32 rounding of the stored positions, not lost
  mass: on one chunk each, float32 wrapped positions put 1.2–1.5e-4 (TNG300-1)
  and 5.0e-4 (FLAMINGO) of the stellar mass one pixel (0.2′) away from its
  float64 pixel, which bounds it; A is shifted the same way. Irrelevant for ΔΣ
  at 1–6′. The cached 2D Stars maps include TNG/Illustris winds, which B does not.
- Tests: the 2D pixel index reproduces `binned_statistic_2d` bit for bit
  (edges included); A and B equal the binned particle weights and share
  `transfer_field`'s bookkeeping; the stacked sample equals `stack_on_array`'s
  own selection (three branches); stacking the changed maps equals `f_of_scale`
  of the separate stacks (s = 1, 0.5, 0).

### Results (z ≈ 0.5)

Figures `figures/2026-09/09-24/pk_components_z05_stellar_S_{fof,ap1,ap2}_M{11,12,13}.pdf`
(bottom panel), numbers in `_stellar_lensing_table.txt` (f at every θ for s = 1
and s = 0, and the relative change). A one-time PNG set of all figures of both
redshifts is in `figures/2026-09/09-24/png/` (200 dpi). Data (beam-compensated): 0.32 ± 0.06 at
1′, 0.46 ± 0.31 at 6′.

f(s = 1) and Δf = f(s = 0) − f(s = 1), at θ = 1′ / 6′:

| sim (haloes stacked) | f 1′ | f 6′ | fof ≥1e11 | fof ≥1e13 | 1 R200m ≥1e11 | 2 R200m ≥1e11 |
|---|---|---|---|---|---|---|
| TNG300-1 (4,307) | 0.517 | 0.899 | +0.093 / +0.065 | +0.077 / +0.056 | +0.071 / +0.056 | +0.048 / +0.061 |
| Illustris-1 (210) | 0.131 | 0.708 | +0.150 / +0.126 | +0.101 / +0.110 | +0.118 / +0.102 | +0.049 / +0.126 |
| SIMBA-100 (500; prov.; tables only) | 0.290 | 0.658 | +0.116 / +0.081 | +0.083 / +0.064 | +0.086 / +0.055 | +0.051 / +0.070 |
| FLAMINGO L1_m9 (157,910) | 0.302 | 0.799 | +0.128 / +0.109 | +0.101 / +0.089 | +0.101 / +0.094 | +0.056 / +0.103 |
| fgas-8sigma (157,910) | 0.163 | 0.671 | +0.127 / +0.119 | +0.096 / +0.096 | +0.103 / +0.101 | +0.045 / +0.112 |
| Jet_fgas-4sigma (157,910) | 0.221 | 0.668 | +0.116 / +0.103 | +0.087 / +0.081 | +0.093 / +0.088 | +0.049 / +0.097 |

Reading:

- **Converting the stars raises f everywhere**, most at small θ: by +0.09 to
  +0.15 at 1′ (membership ≥ 1e11), i.e. +18% (TNG300-1) to +115% (Illustris-1,
  whose inner gas fraction is lowest), 1.5–2.5 × the data error there; by +0.06
  to +0.13 (+7 to +18%) at 6′, well inside the data error.
- **Method ranking at small θ is the reverse of P(k):** at 1′ the 2 R200m
  apertures change f least (+0.05) and membership most; at 6′ the methods agree
  to within ~0.03. Presumably (not checked) the larger the region, the more of the
  added gas lands outside the inner apertures; P(k), in contrast, grows with
  how far the mass moves.
- **Mass cut:** haloes ≥ 1e13 give 65–85% of the ≥ 1e11 change at 1′ (P(k):
  ~90% at k ≈ 5); the stacked galaxies' own, lower-mass haloes matter more here.
- The s = 1 curves are the lensing figure's (identical stacks).

### Results (z ≈ 0.26; FLAMINGO z = 0.30)

Same analysis with `pk_stellar_z026.yaml` (lensing config
`mass_ratio_noBeam_z026.yaml`: abundance 1e-3 (h/Mpc)³, TNG300-1/Illustris-1
stacked at z = 0.26 on the 4822² / 1752² grids, FLAMINGO at 0.3 on 14015²; no
SIMBA). Validation over the 45 configurations as at z ≈ 0.5: stackMap identity
0; explicit s = 0.5 ≤ 3.6e-14; Σ A, Σ B vs moved ≤ 1.1e-14; per-halo ≤ 2e-13;
moved mass and active haloes identical to the 3D run; no 2D Stars caches at
these snapshots, so no negative-stellar check. Figures
`pk_stellar_z026_stellar_S_*`, table `pk_stellar_z026_stellar_lensing_table.txt`.
Data: 0.29 ± 0.05 at 1′, 0.86 ± 0.29 at 6′.

| sim (haloes stacked) | f 1′ | f 6′ | fof ≥1e11 | fof ≥1e13 | 1 R200m ≥1e11 | 2 R200m ≥1e11 |
|---|---|---|---|---|---|---|
| TNG300-1 (8,615) | 0.290 | 0.837 | +0.063 / +0.073 | +0.041 / +0.063 | +0.044 / +0.063 | +0.030 / +0.062 |
| Illustris-1 (421) | 0.069 | 0.318 | +0.114 / +0.149 | +0.053 / +0.126 | +0.086 / +0.114 | +0.038 / +0.092 |
| FLAMINGO L1_m9 (315,821) | 0.173 | 0.689 | +0.097 / +0.120 | +0.056 / +0.100 | +0.075 / +0.106 | +0.039 / +0.104 |
| fgas-8sigma (315,821) | 0.098 | 0.495 | +0.103 / +0.132 | +0.053 / +0.109 | +0.081 / +0.114 | +0.035 / +0.107 |
| Jet_fgas-4sigma (315,821) | 0.132 | 0.538 | +0.093 / +0.114 | +0.047 / +0.092 | +0.073 / +0.098 | +0.036 / +0.095 |

The picture is the same as at z ≈ 0.5, with two differences: the lower-mass
sample (twice the abundance) makes ≥ 1e13 haloes matter less at 1′ (45–65% of
the ≥ 1e11 change), and the absolute change at 6′ (+0.06 to +0.15) now matches
or exceeds that at 1′ (+0.06 to +0.11). Relative to f(s = 1): +22% (TNG300-1)
to +165% (Illustris-1) at 1′, +9 to +47% at 6′; 1.2–2.2 × the data error at 1′.

Cost (both redshifts): FLAMINGO ~70 min per variant on one node (particle
pass 48 min, pixel index 2.5 min, 9 configurations ~15 min, stacks 100 s at
z ≈ 0.5 / 180 s at 0.3 on 23 processes); TNG300-1 ~70 min (pass 28 min, pixel
index 9.5 min for ~4e9 particles, configurations 1.5–5 min each, stacks 24 s);
Illustris-1 ~17 min; SIMBA 4 min. Peak memory (sacct MaxRSS): TNG300-1
247–293 GB (the pass keeps ~4e9 gas particles), FLAMINGO 98–111 GB,
Illustris-1 ~105 GB, stacks ≤ 76 GB.
Allocations: 58832867 (smoke), 58832967 (z ≈ 0.5, and z ≈ 0.26 Illustris-1),
58834395 (z ≈ 0.26).
