# Simulation Programme Record: Tasks 1 to 4

**Status as of 2026-08-26.** Branch `task1-r-profiles`, four commits, merged to
`main`. Gate A open, **Gate B closed**, Gate C not reached.

---

## 0. What this document is

A standalone, step-by-step record of the simulation-side work for the
model-independent baryon–matter cross-correlation programme: what was built,
what was measured, what was deliberately skipped, and what comes next. It is
written so that a reader — or a fresh session — can pick up the work without
having seen the conversation that produced it.

**Companion documents.** This record is self-contained, but three other files
carry detail it summarizes:

| document | role |
|---|---|
| `cross_correlation_notes.md` | The theory note. Authoritative for the estimator and the physics. Defines Tasks 1–6, Phases 0–7 and Gates A–C. |
| `r_profiles_task1_spec.md` | The Task 1 engineering spec. Authoritative for implementation decisions. |
| `filter_specification.md` | The Phase 0 "single source of truth" for filters, discretization, normalization and projection. Written during this work; **closes Gate B**. |
| `r_profiles_implementation_plan.md` | The long-form implementation plan plus a running results log (sections A1–A12, R1–R4, T1–T9). Where the raw numbers live. |

---

## 1. Background: what the simulations are being asked for

Baryonic feedback redistributes gas and suppresses the total matter power
spectrum by tens of per cent at k ≳ 0.5 h/Mpc. The programme aims to measure
the quantity that controls that suppression at leading order, the
baryon–matter cross-correlation `P_bm/P_mm`, as model-independently as the
data allow.

It combines two threads. **Singh et al. (2020, MNRAS 491, 51;
arXiv:1811.06499)** removed galaxy bias from galaxy–galaxy lensing by
substituting `Υ_gm = ρ̄_m r_cc sqrt(Υ_mm Υ_gg)`: measured clustering becomes
part of the model, theory supplies `Υ_mm`, and all residual stochasticity is
compressed into one cross-correlation coefficient `r_cc`, calibrated on mocks.
**The f_gas paper (Liu 2026, in prep.)** applies a common ΔΣ aperture filter to
the velocity-weighted kSZ stack (ACT DR6 × DESI DR2) and to the lensing shear
field (HSC Y3), and takes the bin-by-bin ratio.

This programme applies the Singh substitution to the **kSZ side** instead of
the lensing side. Galaxy–galaxy lensing then drops out of the estimator
entirely and the unobservable baryon auto-correlation cancels, leaving

    Y_bm / Y_mm = (r_bm / r_gb) · Y_gb / sqrt(Y_gg · Y_mm)         (Eq. 4)

Everything on the right is measured except `Y_mm`, which is theory, and the
ratio `r_bm/r_gb`, which carries all the astrophysical stochasticity. **That
ratio is the one thing the simulations must deliver.** Because the baryon
term enters the suppression with weight `f_b ≈ 0.156`, a 20 per cent error on
it propagates to only ~3 per cent on the transfer function `T = sqrt(P_tt/P_mm)`
and ~6 per cent on the power ratio (Eq. 7 of the note). The estimator is
forgiving of calibration error in a way a direct measurement would not be.

**The deliverable of Task 1 is the direct analogue of Singh et al. Fig. 1:**
`r` versus aperture over 1′–6′, one curve per simulation, one panel per filter.

---

## 2. Easy-to-follow summary of each step

1. **Reconnaissance.** Read the theory note and the Task 1 spec, then read the
   whole of `src/` to find what could be reused. Wrote an implementation plan
   naming every reuse point and every conflict found.
2. **Built the Task 1 machinery** (`src/rprofiles.py`): one operation computes
   every filtered amplitude — FFT-correlate a field with the pixelized
   aperture kernel, multiply by a second field, take the map mean.
3. **Tested it** against the four acceptance tests in the spec, on synthetic
   maps, plus an integration test against the existing stamp-based stacker on
   real data.
4. **Ran Task 1** on twelve simulation-samples. Produced the Singh Fig. 1
   analogue and the Gate A metrics.
5. **Discovered the small boxes could not support the measurement** — SIMBA and
   Illustris-1 yield only 500 and 210 galaxies — and, on instruction, dropped
   them, leaving four runs: TNG300-1 and three FLAMINGO variants.
6. **Task 2**, reframed: instead of the point-mass completion, *measured*
   whether the Σ filter is comparable between simulation boxes. Froze the
   filter set to {ΔΣ, Υ}.
7. **Task 3**: measured the electron-versus-baryon correction and recommended
   targeting `P_em/P_mm` directly.
8. **Extended the apertures** above 6′ as a large-aperture diagnostic, keeping
   the observational 1′–6′ bins bit-identical.
9. **Task 4**: built the harmonic-space theory chain (halofit → projection →
   `Y_mm`) and validated it against the simulations in three separable pieces.
10. **Wrote the Phase 0 filter specification** that the note called for but
    which had never existed, and **closed Gate B**.

Every stage below records what was done, what came out, and what was skipped.

---

## 3. Stage-by-stage record

### Stage 0 — Reconnaissance and planning

**Done.** Read `cross_correlation_notes.md` and `r_profiles_task1_spec.md`,
then read `stacker.py`, `filters.py`, `mapMaker.py`, `halos.py`, `loadIO.py`,
`tools.py`, `utils.py`, `field_utils.py` and `snr.py` in full, plus a
representative config-driven script and the existing test conventions.
Produced `r_profiles_implementation_plan.md`.

**Results.** Every reuse point the spec named resolved to a real function,
with three corrections: the CDM particle type is `'DM'` not `'dm'`;
`utils.comoving_to_arcmin` defaults to Planck18 and must be passed the
simulation's own cosmology; and `filters.delta_sigma_kernel`'s default annulus
width is 0.5′ while the pipeline passes 0.75′ explicitly. Three things did
*not* exist and had to be written: any map-level FFT filter, any jackknife
utility, and any arcmin conversion in `field_utils.py` (which is unrelated
MillenniumTNG legacy code).

**Skipped.** Nothing.

### Stage 1 — Task 1 machinery

**Done.** Wrote `src/rprofiles.py`. The core decision, taken from the spec: all
amplitudes come from one operation on periodic 2D maps,

    Y_XY(R; F) = ⟨ F_R[δ_X] · δ_Y ⟩_map

because stamp-stacking a filtered map at galaxy positions is mathematically
identical to this map-level average on a periodic box. That reaches the
field–field pairs (`bm`, `mm`, `bb`) the existing stamp stacker cannot produce,
and the galaxy-crossed case doubles as the validation hook.

Fixed conventions: every field enters as `δ = X/⟨X⟩ − 1`; kernels reproduce
`filters.delta_sigma_kernel`'s pixel-count normalization exactly; errors come
from a 4×4 block jackknife with **`r` formed per realization**, never by
Gaussian propagation of marginal `Y` errors; the discrete galaxy field carries
an analytic self-pair subtraction `K(0)/n̄_pix`; no beam anywhere.

**Results — three findings that changed the plan:**

- **The CDM map is exact by subtraction.** `mapMaker.make_combined_field`
  builds `'total'` as gas+DM+stars+BH and `'baryon'` as gas+stars+BH on
  identical grids, so `CDM = total − baryon` exactly. Verified on TNG300-1:
  the derived baryon fraction is 0.157332 against Ω_b/Ω_m = 0.157332, with
  zero negative pixels. **No DM particle sweep was ever needed.**
- **Every production run already had cached fields at ~0.2 arcmin/pixel**, finer
  than the spec's minimum, so the whole sweep cost FFTs rather than node-days.
- **FLAMINGO snapshot 71 (z = 0.30) is downloaded**, so the BGS-like sample the
  spec declared out of scope was in scope after all.

Also switched all transforms to `scipy.fft` with `workers=-1`: several cached
grids have large prime factors (14015 = 5 × 2803) that force Bluestein's
algorithm, where multithreading was worth 42× at 4822² and 9× at 2674².

**Skipped.** No modification to any existing file. The theory note's Phase 0
item 2 ("extend SimulationStacker to output CDM-only and total-baryon maps")
was already satisfied by `makeField`.

### Stage 2 — Testing

**Done.** `tests/test_rprofiles.py` implements the spec's four acceptance
tests on synthetic maps: the FFT kernel against an independent stamp
reimplementation of `delta_sigma_kernel` (< 1e-12); `⟨F[X]·δ_g⟩` equal to the
stamp-stacked galaxy mean; a Poisson galaxy auto nulled by the self-pair
subtraction; and a known field mixing recovering `r` as an exact algebraic
identity. `tests/test_rprofiles_integration.py` cross-checks against
`SimulationStacker.stack_on_array` on real TNG300-1 data with a real
4307-galaxy SHAM sample, and skips when the data is absent.

**Results.** 32 synthetic tests and 4 integration tests pass. Two behaviours
surfaced that were not anticipated and are now pinned:

- **Υ(R₀; R₀) ≡ 0** by construction, so its coefficient at the reference
  radius is a genuine 0/0. That bin is excluded everywhere rather than
  special-cased.
- **A boundary-tie degeneracy.** Membership uses the strict test `r < edge`, so
  when an aperture edge lands on a realizable lattice distance
  `sqrt(i²+j²)` a whole shell of pixels sits on the boundary and flips under an
  arbitrarily small convention change. At 0.2′ pixels this hits R = 1′ (5.0 px),
  the R = 2.25′ annulus edge (15.0 px) and R = 6′ (30.0 px), each a 12-pixel
  shell. Lattice arithmetic predicts exactly those three radii and no others,
  which is precisely the pattern the integration test showed. The effect is
  ~6 per cent on an amplitude and ~0.6 per cent on a coefficient.

**Incidental finding, not fixed.** `stack_on_array` normalizes ΔΣ by the *true*
pixel area but bins radii on a *quantized* linspace grid, biasing absolute ΔΣ
amplitudes by ~1 per cent at production resolution. It cancels in any ratio,
including the f_gas paper's kSZ/lensing ratio, so it was flagged rather than
changed — `stacker.py` is shared with the f_gas figures.

### Stage 3 — Task 1 production sweep

**Done.** Twelve simulation-samples (six runs × two redshift samples), `yz`
projection, 17 minutes on one CPU node. SHAM samples follow the f_gas paper:
subhalos ranked by `SubhaloMStar`, parent FoF mass ≤ 5×10¹⁴ M☉/h, target
number density 5×10⁻⁴ (cMpc/h)⁻³ for the LRG-like sample and 1×10⁻³ for
BGS-like.

**Results.**

- **`r_bm` is measured cleanly everywhere**: 0.89–1.00 with jackknife errors
  ≤ 0.008. The note's "expectation of near-unity" holds well for it.
- **`r_gb` is not near unity**: on FLAMINGO it runs 1.34 → 1.00 across 1′–6′
  for Σ, but 1.71 → 1.29 for ΔΣ. Values above 1 are permitted — these are
  filtered amplitudes against a sign-changing compensated kernel, so no
  Cauchy–Schwarz bound applies. Singh et al. Fig. 1 likewise shows
  `r_cc^(Υ)` reaching ~1.3.
- **`r_bm/r_gb` is therefore 0.5–0.85 for ΔΣ/Υ and 0.78–1.0 for Σ** — a large
  calibrated transfer, not a small correction. For the estimator this is not
  fatal, since Eq. (4) uses the ratio as a transfer; what matters is its
  cross-code stability.
- **The filter morphology reproduces Singh et al. Fig. 1.** The Σ coefficient's
  deviation is localized at small R and returns to unity by 6′, while the
  compensated filters carry it to the largest aperture — the same trade-off
  the note predicts, now measured for the gas field, which had no precedent.
- **The self-pair subtraction dominates `Y_gg` at small R**: on FLAMINGO it
  removes 6.9× the retained signal at R = 1′ and 1.9× at 6′, so `r_gb` there
  is a difference of comparable numbers. This mirrors, on the simulation side,
  the caveat the note raises for the data measurement of `Y_gg` (Sec. 5.1).

**A conclusion later retracted.** This stage reported that "Σ is the best
filter on every metric". Stage 5 showed that conclusion was contaminated; see
Stage 5 and the correction note in `r_profiles_implementation_plan.md` R2.

### Stage 4 — Restriction to four simulations

**Done.** The naive four-code Gate A scatter was 3–55 per cent, which would read
as a failure. It was not a physics result: the per-run jackknife errors were as
large as the scatter itself (for ΔΣ at z ≈ 0.5, scatter 0.185 against a median
single-run error of 0.546). The cause is sample size, fixed by volume:

| run | box (cMpc/h) | N at 5e-4 | N at 1e-3 |
|---|---|---|---|
| FLAMINGO L1_m9 | 681 | 157,910 | 315,821 |
| TNG300-1 | 205 | 4,307 | 8,615 |
| SIMBA m100n1024 | 100 | 500 | 1,000 |
| Illustris-1 | 75 | 210 | 421 |

SIMBA and Illustris-1 simply cannot measure `Y_gg` over 1′–6′. On instruction
they were dropped, leaving **four runs: TNG300-1 plus the three FLAMINGO
variants** (fiducial, fgas−8σ, Jet_fgas−4σ).

**Consequence, and it is a real cost.** Cross-*code* families fall from four to
two (TNG and FLAMINGO). Singh et al.'s Appendix A lesson is exactly that priors
calibrated on one mock family biased results on an independent family; with two
families that cross-family discipline cannot be exercised.

### Stage 5 — Task 2: the filter set

**The task was reframed.** The note offers two options: (a) restrict to ΔΣ and
Υ, or (b) reconstruct Σ from ΔΣ by point-mass completion (the analogue of
Singh et al. Eqs. 29–30) and quantify the induced bias versus `R_min`. Option
(a) was chosen, on the physical argument that the annulus mean is not
compensated and so its value depends on box size, whereas ΔΣ and Υ are. Task 2
therefore became: **turn that argument into a measurement.**

**Done.** The argument is provable from the note's own kernels (Sec. 5.3):

    W_ann(k; R₁,R₂) = 2[R₂J₁(kR₂) − R₁J₁(kR₁)] / (k(R₂²−R₁²)) → 1  as k → 0
    W_ΔΣ(k → 0) → 0                                                (compensated)

So `Y_Σ` integrates power down to the fundamental mode of whatever volume it is
measured in. `scripts/cross_corr/check_filter_compensation.py` removes every
mode longer than 205 cMpc/h — the modes TNG300-1's box cannot represent — and
remeasures.

**Results.**

- **The mechanism is confirmed and large at the amplitude level.** `Y_Σ` shifts
  by **8.7 per cent** against **0.014 per cent** for the compensated filters, a
  factor of ~600. TNG300-1 shifts by 5.6e-16 at its own box scale, the sanity
  check.
- **But it largely cancels in the coefficients.** `r_Σ` moves 0.6 per cent and
  the Gate A ratio 0.5 per cent, because the removed modes have `r(k) ≈ 1` and
  contribute almost equally to cross and autos.
- **And it explains none of the cross-code disagreement.** Cutting both
  simulations at the same physical scale leaves the difference untouched:
  for `r_bm/r_gb`, Σ 0.1140 → 0.1145, ΔΣ 0.3546 → 0.3545, Υ 0.4417 → 0.4417.

**Conclusion.** The freeze to {ΔΣ, Υ} stands, but on different grounds than
motivated it: Σ *amplitudes* do not port between volumes (disqualifying for the
Task 4 theory transfer), plus the note's measurement-side argument that an
uncompensated disk mean is not measured on the lensing side and reintroduces
large-scale CMB noise on the kSZ side. It does **not** rest on Σ's coefficients
being contaminated — they largely are not.

**Skipped.** The point-mass completion, option (b). Singh et al.'s claim that
Σ-based coefficients localize better therefore remains untested here; it is
the open "discussion item with Uroš" the note names.

### Stage 6 — Task 3: electrons versus baryons

The kSZ stack measures free electrons `e`, while the suppression algebra
(Eqs. 1–7) is written for all baryons `b`.

**Done.** Measured the correction `Y_gb/Y_ge(R)` and compared the cross-code
stability of the two candidate framings. Aggregate split only: `baryon −
ionized_gas` lumps neutral gas, stars and black holes together, and isolating
the stellar part would need new particle sweeps.

**Results.** The correction under ΔΣ at R = 1′:

| run | Y_gb/Y_ge at 1′ | at 3.5′ | at 9.75′ |
|---|---|---|---|
| TNG300-1 | 1.21 | 1.05 | 1.02 |
| L1_m9 fiducial | 1.61 | 1.11 | 1.06 |
| L1_m9 Jet_fgas−4σ | 1.75 | 1.15 | 1.06 |
| L1_m9 fgas−8σ | 2.27 | 1.21 | 1.07 |

It is large, strongly scale-dependent, and **varies by 40 per cent between
feedback variants of a single code**. Cross-code stability cannot separate the
two framings — every difference is within noise for two families — so the
decision rests on the correction factor itself.

**Recommendation: target `P_em/P_mm`**, the first of the note's two framings,
treating stellar and neutral terms as a separate externally constrained
contribution. Targeting `P_bm/P_mm` would require carrying a large,
feedback-dependent transfer; targeting the electron field needs none, because
that is what the kSZ measures.

**Skipped.** The neutral-gas/stellar separation. The note cares about it —
FLAMINGO carries roughly twice TNG's stellar mass for these samples (f_gas
paper Figs. 16–17) — but it needs `Stars` field sweeps for four runs.

### Stage 7 — Aperture extension

**Done.** Appended bins above 6′ at the existing 0.625′ spacing, to 9.75′,
holding the nine data-matched 1′–6′ bins bit-identical. Not extended below 1′,
where the data are hard-limited by resolution.

**Results.** The ratio `r_bm/r_gb` rises monotonically with aperture — 0.53 at
1′ to ~0.85 by 9′ for ΔΣ — consistent with approaching unity on large scales as
the theory expects, confirming the departure from unity is a small-scale
phenomenon.

**A caveat the extension exposed.** TNG300-1's `Y_gg` falls anomalously at
9.75′ (4.18 → 2.04, roughly halving, where FLAMINGO declines smoothly), pushing
its ratio from 0.83 to 0.63. With 4,307 galaxies the self-pair subtraction
removes 73 per cent of the raw `Y_gg` there. All reported statistics now split
the data range from the diagnostic extension, and **the Gate A verdict uses the
data range only.**

### Stage 8 — Task 4: the theory chain for `Y_mm`

**Done.** Built `src/kernels.py` (the analytic Sec. 5.3 transforms),
`src/theory.py` (halofit → projection → `Y`), and two validation scripts. The
comparison is decomposed so error sources cannot hide inside one another:

- **A — transfer chain.** The CDM map's own 2D power spectrum pushed through the
  analytic kernel, compared with the pipeline's real-space filtering of the
  same map. No cosmological model enters.
- **B — CDM versus total matter.** halofit returns total matter; the estimator's
  `m` is CDM. Both maps exist, so this is measured, not assumed.
- **C — halofit accuracy.** The measured spectrum replaced by halofit.

The projection relation was verified rather than trusted: for a fully projected
periodic box `δ_2D = (1/L)∫dz δ_3D` selects the `k_z = 0` mode, giving
`P_2D = P_3D/L`. Measured directly on a TNG300-1 3D field, `P_2D·L/P_3D` =
0.98–1.05 over k = 0.1–6 h/Mpc.

**Results, worst fractional difference against the measured CDM amplitude:**

| run | filter | A transfer | B tot/CDM | C halofit |
|---|---|---|---|---|
| TNG300-1 | ΔΣ | 1.7e-2 | 8.9e-2 | 4.6e-2 |
| TNG300-1 | Υ | 1.8e-2 | 3.0e-2 | 6.4e-2 |
| L1_m9 fiducial | ΔΣ | 2.1e-2 | 1.34e-1 | 5.6e-2 |
| L1_m9 fiducial | Υ | 2.2e-2 | 8.4e-2 | 6.7e-2 |
| L1_m9 fgas−8σ | ΔΣ | 2.0e-2 | 1.69e-1 | 4.8e-2 |
| L1_m9 fgas−8σ | Υ | 2.2e-2 | 1.44e-1 | 5.5e-2 |
| L1_m9 Jet_fgas−4σ | ΔΣ | 2.1e-2 | 1.67e-1 | 4.6e-2 |
| L1_m9 Jet_fgas−4σ | Υ | 2.2e-2 | 1.32e-1 | 5.4e-2 |

- **A is 1.7–2.2 per cent**, limited by pixelization of the 0.75′ annulus, which
  spans only 3.75 pixels at production resolution. This is the accuracy floor
  of any harmonic-space theory prediction against this pixelized measurement.
- **C is 4.6–6.7 per cent**, the expected accuracy of the non-linear model.
- **B is the largest term, 3–17 per cent, and it scales with feedback strength**
  (8.9 per cent TNG300-1, 13.4 fiducial, 16.7 Jet, 16.9 fgas−8σ). Stronger
  feedback drives total matter further from CDM. **This is an order of magnitude
  above the 1–2 per cent hydro-CDM/DMO back-reaction, and unlike it, it is
  feedback-dependent and cannot be absorbed into a fixed transfer.**

**Projection-depth study.** Reprojecting the cached 3D fields in slabs of
26–205 cMpc/h: amplitudes scale as the inverse depth as predicted (a factor
7.4–7.8 against 8), while the coefficients move by only **3.8e-4 (ΔΣ) and
1.6e-3 (Υ)**. Over a factor of eight in depth the coefficient shifts by under
0.2 per cent. This licenses the whole calibration strategy — the mismatch
between the kSZ's full-line-of-sight integration and the clustering's
Π_max = 100 h⁻¹Mpc cylinder threatens the *amplitudes* in Eq. (4), which must
be made consistent by construction, but not the transfer the simulations
deliver.

**Skipped, by decision.** `P_mm^hydro-CDM / P_mm^DMO = 1` is adopted rather
than measured: no DMO run is on disk and downloading one would take over a day.
Known to be wrong at 1–2 per cent (van Daalen et al. 2011; Chisari et al.
2018), and now demonstrably subdominant to B. Also skipped: the data-side
RSD/Π_max recheck for `Y_gg`, which belongs to Phase 4 once the pair-count
pipeline exists; the depth study is its simulation-side proxy.

### Stage 9 — The Phase 0 specification and Gate B

**Done.** Wrote `filter_specification.md`, the "single source of truth for
data, theory, and simulation code paths" that the note's Phase 0 item 1 called
for and which had never been written. Task 4 is the first task that genuinely
needed it, because it is the first to compare absolute amplitudes rather than
ratios. It records the filter definitions, the aperture grid, the
discretization and its boundary degeneracy, the `1/pixArea` normalization
wart, **both projection conventions**, self-pair exclusion, the jackknife, and
a table of every adopted approximation.

---

## 4. The gates

The theory note (Sec. 8) defines three decision points.

### Gate A — is the estimator viable as posed? **OPEN**

*Criterion.* Is the cross-code scatter of `r_bm/r_gb` ≲ 10 per cent (fixed
transfer plus prior width), 10–20 per cent (parametrized-`r` route, the
analogue of Singh et al. Eq. 26 with sim-derived priors, marginalized), or
larger/strongly scale-dependent (estimator not viable as posed)?

*Status.* Over the observational range 1′–6′, with the two remaining code
families:

| filter | z ≈ 0.5 | z ≈ 0.26/0.30 |
|---|---|---|
| ΔΣ | 0.081 | 0.112 |
| Υ | 0.089 | 0.132 |

**These are sample standard deviations over N = 2, which is just the pairwise
difference divided by √2.** The raw TNG-versus-FLAMINGO differences are a
factor √2 larger: 11.4 and 12.6 per cent at z ≈ 0.5, 15.9 and 18.7 per cent at
z ≈ 0.26. The verdict therefore straddles the 10 per cent boundary depending on
which statistic is quoted, and is consistently worse at the lower redshift.

*Honest reading:* **borderline between the fixed-transfer and parametrized-`r`
routes, on an estimate from two code families that cannot support an error
bar.** Feedback sensitivity within FLAMINGO is mild (≤ 7 per cent), which is
encouraging for the estimator's premise. Gate A is recorded as-is and the work
proceeded, on instruction.

### Gate B — filter set and field definition frozen. **CLOSED**

*Decision.* **Filter set {ΔΣ, Υ(R₀ = 1′)}.** Σ is computed and reported as a
diagnostic but is not part of the analysis. **Field definition: target
`P_em/P_mm`**, the ionized-gas field crossed with CDM.

Evidence in Stages 5 and 6; recorded in `filter_specification.md` §1.
Everything downstream uses exactly these.

### Gate C — internal consistency of the data measurement. **NOT REACHED**

Belongs to Phase 6, after the data-side `Y_gg` measurement exists. It tests
source-bin splits, field splits and filter choices at the level of the
statistical errors.

---

## 5. What was skipped, and why

| skipped | reason | cost of the gap |
|---|---|---|
| `xy` and `xz` projections | Only `yz` is cached; the others need full particle sweeps across all runs, days of node-hours | Across-projection scatter unmeasured; Gate A is a cross-*code* test, which `yz` delivers |
| SIMBA and Illustris-1 | 500 and 210 SHAM galaxies; jackknife errors exceeded the cross-code scatter | Cross-code families drop from four to two; Singh Appendix A discipline cannot be exercised |
| Point-mass completion (Task 2b) | Filter set frozen to {ΔΣ, Υ}, so Σ reconstruction is moot | Singh's claim that Σ-based coefficients localize better is untested |
| Neutral-gas / stellar separation | Needs new `Stars` sweeps for four runs | Only the aggregate `b − e` split is known |
| DMO runs | Over a day to download | `P_hydroCDM/P_DMO = 1` assumed; 1–2 per cent, now shown subdominant to the CDM/total-matter term |
| CMB-noise treatment of Σ | Out of scope by instruction | The measurement-side half of Task 2 is unquantified |
| Data-side RSD / Π_max recheck | Pair-count pipeline does not exist yet | Deferred to Phase 4; depth study is the proxy |
| Covariance across aperture bins | 16 jackknife regions cannot support it (Hartlap needs more resamplings than bins + 2) | Per-radius errors only; would need 6×6 or 8×8 regions |
| Apertures below 1′ | Data are resolution-limited there | None for the data application |

---

## 6. Defects found and how

Recorded because the method that found them is reusable.

| defect | where | how found |
|---|---|---|
| Aperture boundary-tie degeneracy | discretization | Integration test showed 6.1 per cent at R=1′ but 1e-15 elsewhere; lattice arithmetic predicted exactly the affected radii |
| `stack_on_array` normalizes by true pixel area but bins on a quantized grid (~1 per cent) | pre-existing, `stacker.py` | Isolating integration-test residuals; flagged, not changed |
| SLURM runner reported success with a failing test | `runCPU_rprofiles_debug.sh` | Job showed `COMPLETED 0:0` while pytest had failed; an `afterok` dependency was satisfied anyway |
| Gate A statistic pooled cross-code with cross-feedback scatter | `plot_r_profiles.py` | Code review against the note's cross-code definition |
| CAMB returns its redshift axis in *increasing* order, so `pk[0]` was z=0 | `theory.py` | The A/B/C decomposition: C came out at +129 per cent, too large for halofit, and A was small, so the fault had to be local |
| Duplicate redshifts at z=0 crash CAMB's integrator | `theory.py` | A regression test written for the bug above |
| σ8 applied *after* halofit rather than before (~0.9 per cent) | `theory.py` | Code review; the original test could not have caught it and was replaced |
| Log-spaced k under-samples the oscillating kernel (~0.3 per cent) | `theory.py` | Code review |

**The decomposition into A/B/C is what made the CAMB bug findable.** A single
end-to-end comparison would have shown a factor-2.3 disagreement with no way to
tell whether the model, the transfer or the measurement was at fault.

---

## 7. Numbers at a glance

**Coefficients (z ≈ 0.5, four runs, `yz`):** `r_bm` = 0.89–1.00 (errors
≤ 0.008); `r_gb` = 1.25–2.1; `r_bm/r_gb` = 0.53 at 1′ rising to ~0.85 by 9′.

**Gate A (1′–6′, two code families):** ΔΣ 0.081 / 0.112, Υ 0.089 / 0.132 at
z ≈ 0.5 / 0.26. Pairwise differences √2 larger.

**Filter compensation:** `Y_Σ` shifts 8.7 per cent under a box-scale low-k cut,
`Y_ΔΣ` 0.014 per cent; `r_Σ` shifts only 0.6 per cent.

**Electron/baryon:** `Y_gb/Y_ge` at 1′ under ΔΣ = 1.21 (TNG) to 2.27 (fgas−8σ).

**Theory chain:** transfer 1.7–2.2 per cent, CDM-vs-total 3–17 per cent,
halofit 4.6–6.7 per cent.

**Projection depth:** amplitudes ∝ 1/L; coefficients stable to 3.8e-4 (ΔΣ).

**Discretization systematics on `r`:** ≤ 2.4 per cent worst case from a 2×
resolution change; ≤ 1.5 per cent from the pixel-scale convention.

---

## 8. Next steps

Ordered by what unblocks the most.

1. **Resolve the CDM-versus-total-matter term (B, 3–17 per cent).** This is now
   the dominant theory-side systematic and it is feedback-dependent. Two
   routes, neither taken: predict a CDM-only non-linear spectrum (CAMB gives
   the linear CDM spectrum, but halofit's correction is formulated for total
   matter), or redefine `m` as total matter throughout — which changes what the
   estimator means and must be agreed against the theory note.
2. **Strengthen Gate A.** It rests on two code families. ANTILLES is the
   natural addition; CAMELS is ill-suited because of its halo-mass limit. Until
   then the Singh Appendix A cross-family discipline is unexercised.
3. **Decide whether the small boxes can be recovered** by raising their SHAM
   number density, accepting that the samples are then not matched to DESI
   across suites.
4. **Phase 4 — the data-side `Y_gg`.** Build the pair-count pipeline (theory
   note Sec. 5.1) on DESI DR2 BGS and LRG bin 1 with the frozen filter, and run
   the fibre-incompleteness campaign; the 1′–6′ range sits at or below the
   fibre patrol scale. Recheck RSD and Π_max at these apertures.
5. **Phase 5 — kSZ systematics.** Resolve the Ondaro-Mallea et al. (2026)
   velocity-reconstruction suppression (10–20 per cent, weakly
   feedback-dependent) and whether it is scale-dependent over 1′–6′; carry over
   the beam treatment from the f_gas paper.
6. **Optional, cheap:** the point-mass completion, to answer Singh's
   localization claim; the `xy`/`xz` projections, if across-projection scatter
   is wanted; more jackknife regions, if a covariance is needed.

---

## 9. File map

**Library.** `src/rprofiles.py` (amplitudes, kernels, jackknife, SHAM galaxy
maps), `src/kernels.py` (analytic harmonic kernels), `src/theory.py`
(halofit → projection → `Y`). No existing `src/` file was modified.

**Scripts**, all run from `scripts/`:

| script | purpose |
|---|---|
| `cross_corr/make_r_profiles.py` | Task 1: compute coefficients, write one `.npz` per run |
| `cross_corr/plot_r_profiles.py` | Singh Fig. 1 analogue and the Gate A metrics |
| `cross_corr/plot_electron_baryon.py` | Task 3 |
| `cross_corr/check_filter_compensation.py` | Task 2 |
| `cross_corr/check_theory_transfer.py` | Task 4 A/B/C validation |
| `cross_corr/check_projection_depth.py` | Task 4 depth study |
| `cross_corr/check_resolution.py` | Resolution and boundary-tie systematics |
| `cross_corr/runCPU_rprofiles.sh`, `runCPU_task4.sh` | SLURM runners |

**Configs.** `scripts/configs/cross_corr/r_profiles_z05.yaml` and
`r_profiles_z026.yaml`.

**Tests.** `tests/test_rprofiles.py` (32), `tests/test_kernels.py` (25),
`tests/test_rprofiles_integration.py` (4, skips without data).

**Outputs.** `data/r_profiles/*.npz` (committed); figures and metrics under
`figures/<yyyy-mm>/<mm-dd>/` (gitignored, regenerable).

**Reproduce everything:**

```bash
cd scripts/
sbatch cross_corr/runCPU_rprofiles.sh    # Tasks 1-3
sbatch cross_corr/runCPU_task4.sh        # Task 4
cd ../tests/ && pytest -q                # 57 tests, no simulation data needed
```
