# Halo-level stellar-to-ionized-gas transfer and the P(k) suppression (unbound gas paper)

**Status as of 2026-09-24.** Code in `src/halo_transfer.py` and
`scripts/unbound_gas/{compute_pk_stellar.py, make_pk_stellar.py, runINT_pk_stellar.sh}`,
config block `stellar:` in `scripts/configs/unbound_gas/pk_components_z05.yaml`,
tests in `tests/test_halo_transfer.py`; small additions to `src/mapMaker.py`
(ionized-mass helper, behaviour unchanged) and `src/loadIO.py` (membership
readers). z ≈ 0.5 committed as `30a6e34`; spectra exist for all six
simulations (§5). z ≈ 0.26 (FLAMINGO z = 0.30) added the same day for five
simulations, SIMBA excluded (§9; config `configs/unbound_gas/pk_stellar_z026.yaml`).
Companion to `pk_suppression_components.md` (the global and
local models, unchanged). Working log:
`NOTES/unbound_gas/session_log_2026-09-23_pk_stellar_transfer.md` (untracked).

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
| `src/halo_transfer.py` | membership and aperture labels, per-halo bookkeeping, `collect_particles` (one pass over stars and gas), `transfer_field` (D and diagnostics), `p_of_scale` | — | — |
| `compute_pk_stellar.py` | per simulation: one particle pass, then for each method × mass cut D and its spectra (numba estimator of `compute_pk_local.py`, identical to Pylians); validation (§4); `--variants`, `--overwrite`, `--max-chunks N` (smoke test, nothing saved) | CPU node | `<stem>_Pk_stellar_<variant>_<n>.npz` |
| `make_pk_stellar.py` | Q for every s, method comparison at s = 0, S bands with ΔS, table; the global and local stars-only models as context | login | `<fig>_stellar_scales_<variant>_<tag>`, `_stellar_methods`, `_stellar_S_<variant>_<tag>`, `_stellar_table.txt` |
| `runINT_pk_stellar.sh` | `compute_pk_stellar.py`, one simulation per node; `SIMS`, `EXTRA`, `CONFIG` env vars | `salloc -N ≤ 4` | logs in `../Outputs_Perlmutter/` |

Config (`stellar:` block; ignored by every other script): `methods`
(`membership`, `aperture`), `aperture_radii` (x, units of R200m),
`halo_mass_min` (FoF GroupMass cuts, M⊙/h), `stellar_scales` (plots only).
Optional `plot.z_label` titles the figures. Where no `_Pk_components_` file
exists (z ≈ 0.26), `compute_pk_stellar.py` checks its estimator against the
Pylians `P_total` of the `_Pk_dmo_` file, skips the negative-stellar-mass check
when there is no Stars cache, and `make_pk_stellar.py` omits the global context
curve and takes the baryon budget from the particle pass (PartType4 + gas; BH,
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

Figures in `figures/2026-09/09-24/`: `pk_components_z05_stellar_scales_fof_M11.pdf`
(Q for s = 0.75 … 0), `_stellar_methods.pdf` (every method and cut at s = 0),
`_stellar_S_{fof,ap1,ap2}_M11.pdf` (S bands with ΔS), and
`_stellar_table.txt`. "global" = all (cached) stars like the box-wide ionized
gas; "local R=1" = all stars transported within a 1 Mpc/h sphere like the
local ionized gas (make_pk_local's `stars` set).

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
  curves (global, local R=1), which move the cached Stars field: for
  Illustris-1 those curves move ~8% more mass.
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

Figures in `figures/2026-09/09-24/`: `pk_stellar_z026_stellar_scales_fof_M11.pdf`,
`_stellar_methods.pdf`, `_stellar_S_{fof,ap1,ap2}_M11.pdf`, `_stellar_table.txt`.

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

