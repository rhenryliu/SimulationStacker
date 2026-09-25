# Stellar-fraction bands: S(k) and the lensing f_gas(θ) between f* = 0 and a stellar-fraction floor (unbound gas paper)

**Status as of 2026-09-25.** Code: `src/halo_transfer.py` (stellar-fraction
floor functions; `collect_particles(..., baryons=True)`),
`scripts/unbound_gas/{compute_fstar.py, stack_fstar_maps.py, make_pk_fstar.py, runINT_fstar.sh}`,
configs `scripts/configs/unbound_gas/pk_fstar_{z05,z026}.yaml`, tests
`tests/test_fstar.py`. Builds on the halo-level stellar transfer
(`pk_stellar_transfer.md`, whose s = 0 outputs give the f* = 0 end); the
configs and scripts of those figures are unchanged. Working log:
`NOTES/unbound_gas/session_log_2026-09-24_stellar_lensing_observable.md`
(§11 on; untracked).

## 1. Question and parameterisation

The simulations start from different stellar fractions, especially inside
haloes (the bar plot of `star_fraction_v2.py`). Instead of the stellar scale
s of `pk_stellar_transfer.md` (a fraction of each simulation's own stars), the
figures here use an absolute **stellar fraction of the halo baryons**,

    f*_h = M*_h / M_b,h ,   M_b,h = M*_h + M_wind,h + M_gas,h + M_BH,h ,

per halo region h (the regions of the transfer: halo-finder membership, or
apertures of 1 or 2 × each halo's own R200m; every host halo above the mass
cut). Each figure shows, per simulation,

- the simulation itself (solid);
- **f* = 0** (dashed): all selected stars converted to ionized gas laid out
  like the region's ionized gas, i.e. exactly the s = 0 transfer (its spectra
  and lensing stacks are reused);
- a **stellar-fraction floor** f* ≥ f*_T (dotted): every region below the
  target gains dM_h = f*_T M_b,h − M*_h of stars, laid out like the region's
  stars (each star scaled by 1 + dM_h/M*_h) and taken from its ionized gas in
  proportion to the ionized mass; regions at or above the target are
  unchanged;

with the band between f* = 0 and the floor filled. Baryons are conserved
region by region and nothing outside the regions changes. Each region's
stellar mass is ordered f* = 0 ≤ simulation ≤ floor, so the simulation lies
between the two ends halo by halo. The curves are not ordered by
construction: f_gas(θ) of the simulation lies in the band everywhere, and S(k)
does too except at a few k (mostly k ≈ 0.5–1 h/Mpc, 40 of the 99
simulation-configurations) where both ends are slightly below it, by at most
2.4e-5 in S — invisible in the figures.

## 2. Definitions (user decisions 2026-09-24)

| item | choice |
|---|---|
| high end | a floor (B2): regions below the target are raised to it, regions above untouched (not every region set to the target) |
| stars | true stars (TNG/Illustris wind-phase particles excluded, as in the transfer) |
| wind particles | counted as **gas** (they are decoupled wind-phase gas cells stored in PartType4); in the denominator, never converted |
| black holes | in the denominator (`BH`, dynamical masses); never converted |
| gas | all PartType0 gas in the denominator; only the ionized gas (`mapMaker.ionized_gas_masses`) is converted |
| where the new stars go | like the region's existing stars |
| region | the transfer region of the method (membership / own R200m / own 2 R200m), mass priority where apertures overlap |
| targets f*_T | membership 0.5, 1 R200m 0.4, 2 R200m 0.25 (near the most star-rich simulation in each region type) |
| too little ionized gas | all of it converted (capped), reported |
| no stars / no ionized gas | region unchanged, reported |
| figures | the S(k) + lensing band figures, all 9 configurations, both redshifts; SIMBA computed at z ≈ 0.5 but not plotted (`plot.exclude_sims`) |

This star fraction differs from `star_fraction_v2.py`'s, which sums the
cached 1000³ fields (PartType4 including winds) in spheres of the
**sample-mean** R200m around the 'massive' halo sample.

## 3. Algebra

With coef_h = dM_h / M*_h (0 for unchanged regions), the floor changes each
star by +coef_h m*_j and each gas particle's ionized mass by
−coef_h (M*_h / M_ion,h) m_ion,i — minus coef_h times the region's s = 0
transfer weights. Its 3D change field H (TSC, as the cached fields; zero total
mass) gives

    P_mm(floor) = P_mm + 2 P_mH + P_HH                       (exact),

and with S, G the 2D maps of the stars added and the ionized gas removed, and
N, T the halo-mean ΔΣ of the cached ionized_gas and total maps on the lensing
sample,

    f(θ; floor) = [N − ΔΣ_G] / [T + ΔΣ_S − ΔΣ_G] × Ω_m/Ω_b .

The f* = 0 end is P_mm + 2 P_mD + P_DD and [N + ΔΣ_A]/[T + ΔΣ_A − ΔΣ_B] × Ω_m/Ω_b
of `pk_stellar_transfer.md`. (A floor is not linear in the target, so
intermediate f* values would need their own fields.)

## 4. Scripts (run from `scripts/`)

| script | does | runs on | output |
|---|---|---|---|
| `src/halo_transfer.py` | `collect_particles(..., baryons=True)` also keeps the labelled wind and BH masses; `halo_baryons`, `floor_coefficients`, `scaled_transfer_field` (3D H), `scaled_transfer_maps_2d` (2D S, G) | — | — |
| `compute_fstar.py` | per simulation: one pass over stars, gas, winds and BH; per configuration the floor's H and its spectra (estimator of `compute_pk_local.py`) and the 2D maps on the lensing grid; validation (§5) | CPU node | `<stem>_Pk_fstar_<variant>_<n>.npz` (3D), `<stem>_fstar<T>_{stars,gas}_<variant>_<tag>_<n2>_yz.npy`, `<stem>_fstar_maps_<n2>_yz.npz` (2D) |
| `stack_fstar_maps.py` | ΔΣ stacks of S and G on the lensing sample (must equal the stored f* = 0 stacks' rows and settings) | CPU node | `<stem>_lensing_fstar_<sample>.npz` (2D) |
| `make_pk_fstar.py` | the band figures and a table | login | `<fig>_fstar_S_<variant>_<tag>`, `<fig>_fstar_table.txt` |
| `runINT_fstar.sh` | `compute_fstar.py` then `stack_fstar_maps.py`, one simulation per node; `SIMS`, `STAGES`, `EXTRA`, `CONFIG` | `salloc -N ≤ 4` | logs `fstar-<job>_<sim>.out` |

## 5. Validation

All 99 configurations (six simulations × 9 at z ≈ 0.5, five × 9 at z ≈ 0.26);
every number is in the `_fstar_table.txt` files and the output files.

| check (what it tests) | result |
|---|---|
| regions: the pass's stellar mass of the regions with stars and ionized gas vs the s = 0 run's moved mass (same labels) | identical (0) in every configuration |
| P_mm of the new estimator vs Pylians P_total (components / DMO file); vs the s = 0 files | ≤ 1.2e-7; identical |
| explicit field ρ_m + H vs P_mm + 2 P_mH + P_HH (first configuration per simulation) | ≤ 7.4e-8 |
| Σ H / stars added (3D, float32 TSC weights) | ≤ 6.7e-10 |
| per-halo conservation (stars added and gas removed vs dM_h) | ≤ 2.5e-11 |
| Σ S, Σ G vs stars added (2D) | ≤ 9.7e-15 |
| raised, uncapped regions end exactly at the target | min(f*_after − f*_T) = −6e-17 |
| gas removed ≤ the region's ionized gas | by construction; max ratio 1 (the capped regions) |
| lensing: same halo rows and settings as the f* = 0 stacks; explicit maps ionized_gas − G/2, total + (S − G)/2 vs the linear combination | identical rows; ≤ 4.7e-14 |
| pytest `tests/test_fstar.py` (5 synthetic tests, incl. H = −D at coef = 1) | pass |

Region bookkeeping to keep in mind (tables, per configuration):

- **FLAMINGO has ~5 million star-less regions above 1e11 M⊙/h** (low-mass
  haloes with a few gas particles and no star particle), holding 10–12% of
  the regions' baryons at the ≥ 1e11 cut for the fiducial and fgas-8sigma
  variants (8–12%; Jet_fgas-4sigma 2–4%). They stay unchanged (no stellar
  template), so FLAMINGO's floor reaches only f* ≈ 0.45–0.50 (membership),
  0.38–0.42 (1 R200m) and 0.24–0.26 (2 R200m) in aggregate at ≥ 1e11; at
  ≥ 1e12 such regions hold ≤ 1.1e-3 of the baryons and at ≥ 1e13 none. Other
  suites: ≤ 0.16% of the baryons.
- **Capped regions** (the target needs more than all their ionized gas):
  at ≥ 1e11, Illustris-1 3,804 of 11,776 raised membership regions (z ≈ 0.5),
  SIMBA 2,075 of 27,540, FLAMINGO ~1e5; none at ≥ 1e13.
- Winds are 0.2–1.5% of Illustris-1's region baryons and ≤ 0.06% of
  TNG300-1's (none in SIMBA/FLAMINGO); BH 0.04–0.1% (TNG/Illustris), 0.1–0.3%
  (SIMBA), 0.6–1.6% (FLAMINGO, dynamical masses).

## 6. Results

Figures in `figures/2026-09/09-25/`: `pk_fstar_z05_fstar_S_{fof,ap1,ap2}_M{11,12,13}.pdf`
and `pk_fstar_z026_fstar_S_*.pdf`; tables `pk_fstar_z05_fstar_table.txt`,
`pk_fstar_z026_fstar_table.txt` (f*, bookkeeping, S at k = 1 and 5 and f_gas
at 1′ and 6′ for the three states, and the validation numbers).

Own star fraction f*_sim and the floor's aggregate f* (≥ 1e11), with the
change of S at k ≈ 5 and of f_gas at 1′ at the f* = 0 end / the floor end,
z ≈ 0.5:

| sim | membership (T = 0.5) | 1 R200m (T = 0.4) | 2 R200m (T = 0.25) |
|---|---|---|---|
| TNG300-1 | 0.14 → 0.50; ΔS −0.014 / +0.047; Δf +0.09 / −0.23 | 0.11 → 0.40; −0.018 / +0.063; +0.07 / −0.18 | 0.07 → 0.25; −0.025 / +0.066; +0.05 / −0.11 |
| Illustris-1 | 0.31 → 0.51; −0.026 / +0.008; +0.15 / −0.02 | 0.29 → 0.43; −0.031 / +0.013; +0.12 / −0.01 | 0.19 → 0.27; −0.047 / +0.019; +0.05 / −0.01 |
| SIMBA-100 (prov.; tables only) | 0.22 → 0.51; −0.021 / +0.049; +0.12 / −0.11 | 0.21 → 0.42; −0.013 / +0.016; +0.09 / −0.06 | 0.12 → 0.26; −0.026 / +0.032; +0.05 / −0.05 |
| FLAMINGO L1_m9 | 0.27 → 0.45; −0.026 / +0.038; +0.13 / −0.12 | 0.26 → 0.38; −0.030 / +0.045; +0.10 / −0.08 | 0.16 → 0.24; −0.043 / +0.045; +0.06 / −0.05 |
| fgas-8sigma | 0.33 → 0.46; −0.028 / +0.024; +0.13 / −0.05 | 0.32 → 0.39; −0.031 / +0.028; +0.10 / −0.03 | 0.20 → 0.25; −0.046 / +0.030; +0.05 / −0.02 |
| Jet_fgas-4sigma | 0.33 → 0.49; −0.020 / +0.025; +0.12 / −0.07 | 0.31 → 0.42; −0.023 / +0.030; +0.09 / −0.05 | 0.19 → 0.26; −0.035 / +0.030; +0.05 / −0.03 |

z ≈ 0.26 (same columns): TNG300-1 0.13 → 0.50 (−0.014 / +0.053; +0.06 / −0.14),
0.11 → 0.40, 0.07 → 0.25; Illustris-1 0.35 → 0.52 (−0.027 / +0.007;
+0.11 / −0.01), 0.32 → 0.45, 0.21 → 0.29; FLAMINGO 0.29–0.35 → 0.46–0.50
(membership), 0.26–0.33 → 0.39–0.42, 0.16–0.20 → 0.24–0.26.

Reading:

- **The simulations sit at very different places in the band.** TNG300-1's
  regions are star-poor (f* = 0.07–0.14), so the floor multiplies their
  stellar mass by 3.2–4.5 and moves S(k ≈ 5) by +0.045 to +0.074 and f_gas(1′)
  by −0.06 to −0.23 (all 18 configurations), 2–4 times its f* = 0 shift
  (ΔS −0.012 to −0.025, Δf +0.02 to +0.09). Illustris-1 is already at or
  near the targets (0.19–0.35), so its floor barely changes anything
  (ΔS ≤ +0.02, Δf ≥ −0.02) and its band is almost all on the f* = 0 side.
  FLAMINGO and SIMBA are in between.
- **More stars means more small-scale power:** the floor raises S at
  k ≈ 5 in all 99 configurations (stars are more concentrated than the ionized gas they replace); for
  TNG300-1 with the 2 R200m floor at z ≈ 0.26, S exceeds 1 around
  k ≈ 2 h/Mpc (up to 1.0036).
- **The lensing ratio moves the opposite way:** in all 99 configurations the
  floor lowers f_gas at every θ (less ionized gas, more concentrated total
  mass) and f* = 0 raises it.
- Region size: with the 2 R200m apertures the f* = 0 end moves S(k ≈ 5)
  most (as for s) and the floor moves f_gas(1′) least (lower target, gas
  taken from a larger region); the floor's S shift (+0.02 to +0.07) is
  comparable to or larger than the membership one (+0.007 to +0.053).

## 7. Reproduce

Needs the s = 0 outputs of `pk_stellar_transfer.md` (spectra, maps, lensing
stacks) for the same configurations and lensing sample.

```bash
cd scripts/
salloc -q interactive -C cpu -N 4 -t 2:30:00 -A desi --no-shell
SLURM_JOB_ID=<id> SLURM_JOB_NUM_NODES=4 \
    SIMS="L1_m9 fgas-8sigma Jet_fgas-4sigma TNG300-1" bash unbound_gas/runINT_fstar.sh
#   then Illustris-1 and m100n1024 (z ~ 0.5); CONFIG=configs/unbound_gas/pk_fstar_z026.yaml for z ~ 0.26
python unbound_gas/make_pk_fstar.py -p configs/unbound_gas/pk_fstar_z05.yaml
python unbound_gas/make_pk_fstar.py -p configs/unbound_gas/pk_fstar_z026.yaml
cd ../tests && pytest test_fstar.py -v        # 5 synthetic tests, ~20 s
```

Measured cost per simulation (one node, 9 configurations; compute + stacks):
FLAMINGO ~80–90 min (particle pass 49 min, 3D field + spectra + 2D maps
~3–4 min per configuration), peak 228–250 GB; TNG300-1 100–120 min (pass
27–29 min, pixel index ~10 min), peak 264–294 GB; Illustris-1 ~25 min; SIMBA
~6 min. Stacks 1–3.5 min, ≤ 76 GB. New data: 198 maps (116 GB, products/2D),
33 spectra files (products/3D), 22 bookkeeping/stack files (products/2D).
Jobs (2026-09-24/25): 58854581 (smoke; SIMBA z ≈ 0.5, Illustris-1 both z),
58854783 (z ≈ 0.5), 58855280 (z ≈ 0.26).

## 8. Open items

- FLAMINGO's star-less haloes (10–12% of the region baryons at ≥ 1e11 for two
  variants) stay at f* = 0 in the floor: raising them would need a stellar
  template (e.g. their ionized gas, or a model profile) — not done.
- The floor is not linear in the target, so each target is its own set of
  fields; a finer f* grid (e.g. several dotted lines) costs one pass each
  unless several targets are built in one pass.
- Targets and the 1 R200m value (0.4) are the user's choices of 2026-09-24;
  SIMBA stays out of the figures while its ionized gas is provisional.
- **Follow-up (separate task, not started, 2026-09-25):** a physically
  motivated band. This band scales all stars of a region together, while the
  poorly constrained component is the central galaxy's envelope + ICL, and its
  ends lie outside the observed range for ≥ 1e13 haloes. A split of the stars
  within R200m (≥ 1e13: ~17–35% central within 30 pkpc, ~23–33% envelope +
  ICL, ~41–51% satellites) and a literature review (R200m stellar fraction of
  the baryons unconstrained because of the gas beyond R500c; suggested ceiling
  ~0.30) are in `NOTES/unbound_gas/`; options and context in
  `NOTES/unbound_gas/handoff_physical_stellar_band.md` (untracked).
