# Simulation Programme Record: Tasks 7 to 10

**Status as of 2026-09-10.** Branch `main`, two commits (`ec8f129`, `82526e3`).
**Gate A passes** on the new statistic, **Gate B reopened** on the field
definition, Gate C not reached.

---

## 0. What this document is

A standalone, step-by-step record of the second block of simulation work on the
model-independent baryon–matter cross-correlation programme: the calibration
factor `C`, the Park et al. Y transform, the re-taken gates, and the Task 9
figure. It is the direct successor to `tasks_1_to_4_record.md` and is written
so that a reader — or a fresh session — can pick up the work without having
seen the conversation that produced it.

**Companion documents.**

| document | role |
|---|---|
| `cross_correlation_notes.md` | The v0.1 theory note. Defines Tasks 1–6, Phases 0–7 and Gates A–C. |
| `cross_correlation_notes_v0.2_addendum.md` | Uroš's response. Retires point-mass marginalization, adds the Y transform and Route B, and defines Tasks 7–10 and five falsifiable predictions. **This is what the present work answers.** |
| `cross_correlation_notes_v0.3_response.md` | The reply to that addendum, written from the results recorded here. Authoritative for the scientific conclusions. |
| `tasks_1_to_4_record.md` | The predecessor record. Gate A open, Gate B closed. |
| `filter_specification.md` | The Phase 0 single source of truth for filters and discretization. |

### A note on the three things called "Y"

Three distinct objects in this programme are written with a Y-like glyph, and
in the `dejavuserif` math font the `\Upsilon` macro renders *pixel-identically*
to an upright Latin `Y`, so the figures could not distinguish them. The
cross-correlation figures therefore use Computer Modern (`mathtext.fontset =
'cm'`), which draws the forked ϒ. The convention, following the addendum's
Section 1 notation table:

| written | is | read it as |
|---|---|---|
| $\Upsilon(R;R_0)$ | the Baldauf et al. (2010) **filter** | forked ϒ, always with an $R_0$ argument |
| $Y(R;R_{\max})$ | the Park et al. (2021) **filter** | plain italic Y, always with an $R_{\max}$ argument |
| $Y_{\alpha\beta}$ | a filtered **amplitude** ($Y_{gb}$, $Y_{mm}$, $Y_{bm}$, …) | plain italic Y, always with *field subscripts* and never a radial argument |

So a radial argument means a filter and a field subscript means an amplitude;
the Baldauf filter is additionally distinguished by its glyph. In code the
three are unambiguous already and are left alone: the filter keys are
`'Upsilon_R0=<r0>'` and `'Ytransform_Rmax=<rmax>'`, and amplitudes are
`Y_<pair>_<filter>`.

---

## 1. Background: what was being asked for

`tasks_1_to_4_record.md` left Gate A open. The Task 1 statistic
`r_bm/r_gb` came out at 0.53–0.85 — a large calibrated transfer, not a small
correction — with a cross-code scatter of 8–13 per cent that straddled the
10 per cent fixed-transfer threshold, measured on two code families whose
per-run errors were comparable to the scatter itself.

The v0.2 addendum reframed the problem. It observed that both estimator routes'
correction factors collapse to one ratio of four filtered amplitudes,

    C = r_bm r_gm / r_gb = Y_bm Y_gm / (Y_mm Y_gb)                     (A12)

and proved (its Eq. A13) that **`C = 1` identically whenever the galaxies'
correlation with the gas is entirely mediated by the matter field**, whatever
the feedback does. Deviations from unity therefore measure direct galaxy–gas
stochasticity, a one-halo effect. It added a second estimator route needing no
clustering and no theory spectrum,

    Y_bm/Y_mm = (r_bm r_gm / r_gb) · Y_gb/Y_gm                         (A11)

a fourth filter (the Park et al. 2021 Y transform), and four new tasks:

- **Task 7** — measure `C` for all runs, samples and filters, both conventions.
- **Task 8** — implement the Y transform as a direct map-level filter and scan
  `R_max`.
- **Task 9** — the two-panel figure showing truth against both uncorrected
  estimators, and the suppression against measured `P_tt/P_mm` at matched
  `k_50`.
- **Task 10** — re-take Gate A on `C` rather than on `r_bm/r_gb`.

It also stated five predictions in advance, so the runs would be a test rather
than a fit.

---

## 2. Easy-to-follow summary of each step

1. **Read the three documents, then answered Task 7 before writing any code.**
   `compute_Y_matrix` computes every unordered field pair, so `Y_gm` was
   already sitting in the committed Task 1 `.npz`. `C` was computable
   immediately, for four runs, two redshifts and three filters.
2. **Found the same shortcut for two more tasks.** Convention T is an exact
   recombination of measured amplitudes; the Y transform is an exact linear
   combination of `Sigma` amplitudes. Neither needed a new field sweep.
3. **Agreed the scope** with three questions: a standalone script leaving
   `src/` untouched, both `R0 = 1'` and `2'` carried, amplitudes and metrics
   before figures.
4. **Built `make_calibration_factor.py`** and a test module that checks each
   of the three algebraic identities against the thing it replaces.
5. **Smoke-tested in `debug`**, then verified all 30 shared amplitude arrays
   were bit-identical to the committed Task 1 output.
6. **Ran the sweep** — first submitted to `regular`, cancelled on instruction,
   moved to `interactive`; hit and diagnosed a nested-`srun` deadlock before
   the documented `salloc … bash runner` form worked.
7. **Discovered the filter comparison is under-powered.** With four runs a
   scatter estimate carries 41 per cent uncertainty, so seven of eight filters
   are statistically tied.
8. **Wrote the v0.3 response**, then verified every number in it against the
   files and corrected four overstatements it contained.
9. **Code review found a Critical defect** — meaningless suppression arrays
   written for the electron field — which forced a regeneration of all eight
   data products. Committed as `ec8f129`.
10. **Built Task 9**: measured the unfiltered spectra and `k_50`, drew both
    figures, fixed two more review findings, committed as `82526e3`.

Every stage below records what was done, what came out, and what was skipped.

---

## 3. Stage-by-stage record

### Stage 0 — Reconnaissance, and Task 7 answered from existing files

**Done.** Read `cross_correlation_notes.md`, the v0.2 addendum and
`tasks_1_to_4_record.md` in full, then `rprofiles.py`, `kernels.py`,
`theory.py`, `make_r_profiles.py` and `check_theory_transfer.py`.

**Results — three findings that removed most of the planned work:**

- **`Y_gm` was already in every committed `.npz`.** `compute_Y_matrix` loops
  over all unordered pairs of `{g, e, b, m}` and `make_r_profiles.py` saves all
  ten. `C` was therefore computable for four runs × two redshifts × three
  filters with no new computation, and came out at 1.13 → 1.01 over 1′–6′ for
  ΔΣ with a cross-run scatter of 1.8–4.5 per cent.
- **Convention T needs no total-matter field.** Because the CDM map is
  *defined* as `total − baryon`, `delta_t = f_m delta_m + f_b delta_b` holds
  exactly, so `Y_tt`, `Y_gt`, `Y_bt` are bilinear recombinations of measured
  amplitudes.
- **The Y transform needs no new convolution.** `F^Y_R = F^Sigma_R −
  F^Sigma_Rmax` as a map-level operator, so its amplitudes are a linear
  combination of `Sigma` amplitudes — the same trick `compute_Y_matrix` already
  uses to build `Upsilon` from `DSigma`.

**Also verified, against `check_theory_transfer.py`:** the addendum's
Section 4.6 flag is correct. Stage 8's "term B" is `Y_tt/Y_mm − 1` measured on
one hydrodynamic box; it is negative, reproduces the reported values, and
matches `(f_m + f_b x)^2 − 1` to under one per cent relative in all four runs.
It is the signal, not a theory-side systematic.

**Skipped.** Nothing.

### Stage 1 — Scope decisions

**Done.** Three questions put before any edit, per the two-gate working rule.
Answers: a **new standalone script** leaving `src/rprofiles.py` and
`make_r_profiles.py` untouched; **both `R0 = 1'` and `R0 = 2'`** carried, since
the production configs moved to 2′ (commit `91e39d7`) while every addendum
prediction assumes 1′; **amplitudes and metrics first**, figures deferred.

**Results.** The standalone-script choice turned out to matter: it is what
allows `data/r_profiles/*.npz` to stay reproducible bit-for-bit, which became
the regression test in Stage 3.

**Skipped.** Nothing.

### Stage 2 — The calibration machinery

**Done.** Wrote `scripts/cross_corr/make_calibration_factor.py` (849 lines).
One `compute_Y_matrix` call per (run, projection) over a 19-aperture grid — the
Task 1 grid unioned with the reference radii `{2′}` for Upsilon and
`{4′, 5′, 6′, 9′}` for the Y transform — then everything else as
post-processing:

- `Upsilon(R;R0) = DSigma(R) − (R0/R)² DSigma(R0)` for each `R0`;
- `Y(R;Rmax) = Sigma(R) − Sigma(Rmax)` for each `Rmax`, masked at
  `R ≥ 0.8 Rmax`;
- Convention T by recombination, with **`f_b` taken from the maps**, never from
  the header: `OmegaBaryon/Omega0` is wrong at the per-cent level for FLAMINGO,
  whose `Omega0` carries a neutrino contribution absent from the particle maps;
- `C`, `C_A`, `x` and the coefficients for gas `∈ {b, e}` in both conventions,
  every one formed **per jackknife realization**.

**Results.** The whole sweep costs 19 apertures × 2 base filters × 4 fields per
run, against 15 apertures for Task 1 — a 27 per cent increase for eight filter
variants and two conventions.

**Skipped.** The `T_bp` reconstruction path of the addendum's Section 2. It is
needed only for a lensing leg that does not exist in the simulation pipeline,
and the addendum's own Section 2.4 concedes Route A never needs it.

### Stage 3 — Testing and the regression check

**Done.** `tests/test_calibration_factor.py`, 22 tests. The design rests on
three identities that would fail *silently* — the amplitudes would still be
finite, smooth and plausible — so each is tested against the thing it replaces:

| identity | tested against |
|---|---|
| Y transform as a linear combination | a directly-built `Sigma(R) − Sigma(Rmax)` convolution kernel |
| Upsilon rebuilt at arbitrary `R0` | `compute_Y_matrix`'s own Upsilon |
| `delta_t = f_m delta_m + f_b delta_b` | amplitudes from a directly built `t` map, synthetic **and** on the real cached TNG maps |

**Results.** All pass. The on-data Convention T check holds to `1e-10` on the
real caches, which is the test that would catch a mismatched `total`/`baryon`
pair. A deliberate 5 per cent perturbation of `f_b` breaks the identity, so the
test has teeth.

**The regression check that justified the standalone-script choice:** all 30
amplitude arrays shared with `data/r_profiles/r_profiles_TNG300-1_67_yz.npz`
are bit-identical, worst relative difference exactly zero. Appending four
reference radii did not move the nine data-matched bins.

**Skipped.** Nothing.

### Stage 4 — Running the sweep, and two SLURM lessons

**Done.** Tiered as the working rules require: `py_compile` and an aperture-grid
preview on the login node, then a reduced run in `debug` (job **58150753**,
TNG300-1 only, `COMPLETED 0:0`, 88 s, 22.7 GB peak), then the full sweep.

**Results.** Two operational lessons worth recording because both cost time:

- **The `regular` queue was the wrong choice.** Job **58150915** was submitted
  and sat pending; it was cancelled on instruction and the work moved to the
  `interactive` QOS, which turned around in seconds. For a 28-minute FFT job on
  cached fields, `interactive` is the right default.
- **Nested `srun` deadlocks.** Allocating with `salloc --no-shell` and then
  wrapping the runner in an outer `srun` left only the `extern` step alive
  (job **58151138**); the runner creates its own job step. The documented house
  form — `salloc -q interactive -C cpu -N 1 -t 1:00:00 -A desi bash
  cross_corr/runINT_calibration.sh` — is the one that works, and it releases
  the allocation automatically when the runner exits.

The full sweep (job **58151144**) took **27.9 minutes**: 7.6 for `z ≈ 0.5`,
19.7 for `z ≈ 0.26/0.30` where the FLAMINGO grids are 14015².

**Skipped.** The `xy` and `xz` projections, as in Task 1: only `yz` is cached.

### Stage 5 — Task 7: the calibration factor

**Done.** `C(R)` for four runs, two samples, eight filter variants, two
conventions, two gas fields, with jackknife errors from 16 blocks.

**Results.**

`C` for ΔΣ at `z ≈ 0.5`, over the data range:

| `R` | TNG300-1 | L1_m9 fid | fgas−8σ | Jet_fgas−4σ | scatter |
|---|---|---|---|---|---|
| 1.000′ | 1.114 ± 0.006 | 1.105 ± 0.001 | 1.166 ± 0.001 | 1.138 ± 0.002 | 2.4% |
| 3.500′ | 1.020 ± 0.005 | 1.010 ± 0.001 | 1.080 ± 0.002 | 1.066 ± 0.002 | 3.3% |
| 6.000′ | 1.001 ± 0.006 | 0.993 ± 0.002 | 1.025 ± 0.002 | 1.030 ± 0.003 | 1.8% |

- **`C` is close to unity as well as stable.** Mean `|C−1|` = 0.052 over 1′–6′
  for ΔΣ at `z ≈ 0.5`, against a 22–45 per cent transfer for `r_bm/r_gb`.
- **Jackknife errors are 0.001–0.01**, an order of magnitude below the
  cross-run scatter, so unlike the Stage 4 situation of the previous record the
  scatter is a real physical difference between runs rather than noise.
- **Replacing `r_bm/r_gb` with `C` collapses the cross-code disagreement** for
  ΔΣ at `z ≈ 0.5` from 10.9 to **1.3 per cent**.
- **But not at the other redshift.** At `z ≈ 0.26/0.30` the same cross-code
  comparison gives 9.7 per cent — inside the threshold, seven times worse, and
  unexplained.

**Skipped.** Nothing in Task 7 itself.

### Stage 6 — Task 8: the Y transform

**Done.** Implemented as a direct map-level filter with `R_max ∈ {4′, 5′, 6′,
9′}`, masked at `R ≥ 0.8 R_max`.

**Results.** Mean `|C−1|` / worst cross-run scatter over 1′–6′:

| filter | bins | `z ≈ 0.5` | `z ≈ 0.26` |
|---|---|---|---|
| ΔΣ | 12 | 0.052 / 0.047 | 0.100 / 0.059 |
| Σ | 12 | 0.051 / 0.042 | 0.057 / 0.060 |
| Υ(1′) | 11 | 0.078 / 0.083 | 0.059 / 0.096 |
| Υ(2′) | 9 | 0.123 / 0.044 | 0.068 / 0.085 |
| Y(4′) | 5 | 0.116 / 0.049 | 0.066 / 0.078 |
| Y(5′) | 6 | 0.122 / 0.046 | 0.076 / 0.074 |
| Y(6′) | 9 | 0.118 / 0.064 | 0.093 / 0.071 |
| Y(9′) | 12 | 0.086 / 0.046 | 0.094 / 0.068 |

- The addendum's fiducial `R_max = 5'` leaves only **six** usable bins in
  1′–6′, as its Section 8.2 predicted.
- **Larger `R_max` is better** on both columns, which argues for 9′ — but 9′ is
  outside the data range, so on real data the transform would reference an
  aperture the kSZ stack does not measure. That obstacle is not in the
  addendum and it weakens the Y transform independently of its performance.

**Skipped.** The `T_bp` versus direct-measurement validation (Stage 2), and the
`check_filter_compensation.py` re-run with the Y transform included. The latter
is cheap and remains open; the prediction to test is that the Y transform
shifts far less than Σ's 8.7 per cent under a box-scale low-`k` cut but
noticeably more than ΔΣ's 0.014 per cent.

### Stage 7 — The statistical-power finding

**Done.** Before reading anything into the filter ordering, tested whether the
differences in the scatter column are significant. With `N = 4` runs the
sampling uncertainty on a standard deviation is `sigma/sqrt(2(N−1))` = **41 per
cent relative**.

**Results.** **Seven of eight filters are within 1σ of the best** at both
redshifts. The only filter this measurement can disfavour is `Υ(R0 = 1')`, at
1.7σ (`z ≈ 0.5`) and 1.1σ (`z ≈ 0.26`).

This is the most consequential methodological result of the session: the
four-run design is sufficient to establish that Gate A passes, but **not to
rank filters within it**. Any filter recommendation has to rest on the
quantities that *are* resolved — mean `|C−1|` (where the ΔΣ-versus-Y-transform
gap is a factor of two), usable bin count (exact), and the `C → 1` limit.

**Skipped.** Nothing. This stage was analysis of existing output.

### Stage 8 — The v0.3 response document, and its own corrections

**Done.** Wrote `docs/cross_correlation_notes_v0.3_response.md` (484 lines),
then verified every quantitative claim in it against the `.npz` files.

**Results — the verification caught four overstatements in the draft**, all
now corrected:

| draft claim | actual | fix |
|---|---|---|
| cross-feedback exceeds cross-code "for every filter at both redshifts" | 14 of 16 cells; two counterexamples | verdict qualified, full table added |
| "ΔΣ wins on every metric" | Σ has smaller `|C−1|` at `z ≈ 0.5`; ΔΣ has the largest at `z ≈ 0.26` | recommendation re-based on stability and bin count |
| ΔΣ bolded as best `|C−1|` at `z ≈ 0.5` | Σ is 0.051 against ΔΣ 0.052 | bolding removed |
| B residual `2.5e−3` | `2.9e−3` over both redshifts | corrected, and the addendum's `6e−4` bound recorded as ~5× too tight |

**Skipped.** Nothing.

### Stage 9 — Code review, and a Critical defect

**Done.** Ran the `code-reviewer` subagent on the unit at the commit gate.

**Results.** One Critical, one Warning, both real:

- **Critical: meaningless suppression arrays for the electron field.**
  `S_e_*` and `S_t_e_*` were being written into every `.npz`. Eqs. (A16)/(A17)
  descend from `delta_t = f_m delta_m + f_b delta_b`, which requires the gas
  field to *complete* the mass budget against the CDM. Electrons are a strict
  subset of the baryons, not a complementary component, so `S(x_e)` is finite,
  smooth and meaningless. Now gated on `SUPPRESSION_FIELD = 'b'`. `C`, `C_A`
  and `x` remain written for both gas fields — those *are* meaningful for
  electrons and Section 5.2 of the response depends on them.
- **Warning: masked bins written as ordinary numbers.** The reviewer
  demonstrated `C = 12.5` sitting at `R < R0` for Upsilon with nothing but a
  separate boolean to flag it. Derived quantities are now NaN outside their
  mask; raw `Y_*` amplitudes stay unmasked, since those are well-defined
  everywhere and are useful diagnostics.

**Why 17 tests missed the Critical item:** `build_filters` and
`flatten_for_npz` — the glue deciding which gas field feeds which formula —
had no test at all. Five were added, taking the module to 22.

All eight `.npz` were regenerated (job **58152212**, 26.3 min) and every number
in the response document re-verified against the new files. **None changed**:
the document quotes `C` only within its mask, and its suppression figures are
analytic propagation through (A18), not the `S_*` arrays.

Committed as **`ec8f129`**.

### Stage 10 — Task 9: the figure

**Done.** Two new scripts. `make_task9_spectra.py` measures the 2D auto and
cross spectra from the cached maps and computes `k_50(R;F)`;
`plot_task9.py` draws the two-row figure. Run together in job **58152609**
(2.1 min — the fields were still in page cache).

Two things could not come from the calibration `.npz`, and the reasoning is
worth recording:

- **`P_tt(k)/P_mm(k)` unfiltered.** The filtered analogue `Y_tt/Y_mm` *is* in
  the npz, but overlaying it would make the lower panel trivially true — that
  is rung 1 of the addendum's Section 5.5 ladder (identity check) dressed as
  rung 4 (suppression check). So the spectra were measured from the maps.
- **`k_50`**, computed against each run's *measured* CDM spectrum and the
  *pixelized* kernel rather than the power-law approximation of Section 1.4,
  whose own Appendix B shows `k_50` moving by a factor 1.8 between `n = −1` and
  `n = −1.5`.

**Results.**

- **`response_quantiles` reproduces the addendum's Appendix B to three
  decimals** on a power law (ΔΣ at 1′: 1.943 against 1.943; Σ at 1′: 0.373
  against 0.372; Y(5′) at 1′: 0.557 against 0.557), including the Y-transform
  kernel that `src/kernels.py` does not carry. Independent validation of the
  machinery where it overlaps the addendum.
- **Route B with no correction is nearly unbiased for ΔΣ**: `B/truth` = 0.92–0.98
  at `z ≈ 0.5` and 0.88–0.93 at `z ≈ 0.26`. **Route A with no correction is off
  by 36–54 per cent.**
- **The window-smearing residual favours ΔΣ**, and this is resolved where the
  scatter is not: `S(truth)` against measured `P_tt/P_mm` at matched `k_50` is
  **+0.8 to +1.7 per cent for ΔΣ** against +1.9 to +2.9 for Υ(1′) and the Y
  transform at `z ≈ 0.5`.

**A second review** found two more defects, both fixed before commit (Section 6).
Committed as **`82526e3`**.

**Skipped.** The `use_sim_scatter` and beam checks of the addendum's
Section 8.1; the covariance treatment of Section 8.4.

---

## 4. The gates

### Gate A — is the estimator viable as posed? **PASSES, fixed-transfer route**

*Criterion.* Cross-code scatter of the calibration factor ≲ 10 per cent (fixed
transfer plus prior width), 10–20 per cent (parametrized route), or larger
(not viable as posed).

*Status.* Every filter, both redshifts, is inside the fixed-transfer threshold.
Worst case across the whole grid is `Υ(R0 = 1')` at `z ≈ 0.26`, 9.6 per cent;
ΔΣ is 4.7 and 5.9 per cent at the two redshifts. Propagated through (A18), a
6 per cent error on `C` gives **1.9 per cent on `S`** and 0.9 per cent on `T`.

*Honest reading:* **passes, and by a clearer margin than the Task 1 statistic
ever offered** — but on the same two code families, and Stage 7 shows the four
runs cannot rank filters within the pass. Where Task 1's Gate A verdict was
limited by jackknife errors comparable to the signal, this one is limited by
having four samples of a population.

### Gate B — filter set and field definition. **REOPENED**

The addendum reopened Gate B on the filter set. The measurement reopens it on
the **field definition** instead.

*Filter set.* Recommend `{ΔΣ}` fiducial with `Υ` retained as the
point-mass-nulling cross-check, and **the Y transform dropped** — not on the
scores, which Stage 7 shows are mostly not significant, but because its best
configuration (`R_max = 9'`) references an aperture outside the data range.
If `Υ` is retained it should be at `R0 = 2'`, since `Υ(1')` is the one filter
this measurement *can* disfavour.

*Field definition.* Stage 6 of the previous record froze the target as
`P_em/P_mm` on the argument that the electron field needs no correction because
it is what the kSZ measures. That argument is about the *observable*; the
calibration points the other way:

| filter | gas = baryons | gas = electrons |
|---|---|---|
| ΔΣ | 0.052 / **0.047** | 0.074 / **0.095** |
| Υ(1′) | 0.078 / 0.083 | 0.118 / 0.067 |
| Y(9′) | 0.086 / 0.046 | 0.102 / 0.053 |

(mean `|C−1|` / max cross-run scatter, `z ≈ 0.5`.) For the fiducial filter the
electron target carries **twice** the scatter of the baryon target. Two code
families cannot adjudicate this, which is the same limitation Stage 6 hit.

### Gate C — internal consistency of the data measurement. **NOT REACHED**

Unchanged. Belongs to Phase 6, after the data-side `Y_gg` measurement exists.

---

## 5. The five predictions, scored

| # | prediction | verdict |
|---|---|---|
| 1 | Y transform beats Υ on `\|C−1\|` | **falsified** at `z ≈ 0.5` (0.086–0.122 against 0.078); **reverses** at `z ≈ 0.26` |
| 2 | Y and Υ beat ΔΣ | **falsified on `\|C−1\|`** at `z ≈ 0.5`; **untestable on scatter** (Stage 7) |
| 3 | Σ's advantage shrinks at matched `k_50` | **now testable** — the `k_50` machinery exists, but the Σ-versus-ΔΣ comparison at matched `k_50` was not made |
| 4 | `C → 1` at large `R` for every filter | **holds for ΔΣ, Σ, Y; fails for Υ** |
| 5 | `C` more stable across feedback than codes | **falsified, backwards**, in 14 of 16 cells |

On **4**, Υ plateaus at `C = 0.885–0.940` at 9.75′ where ΔΣ reaches
0.990–1.014. This is *not* the bug the addendum suspected: it is independent of
`R0` and the other three filters converge. `Υ(R) = ΔΣ(R) − (R0/R)² ΔΣ(R0)`, and
although `(R0/R)² ≈ 0.01` at 9.75′, `ΔΣ(1')` exceeds `ΔΣ(9.75')` by roughly two
orders of magnitude, so the reference term stays order-unity at every aperture.
Υ never converges to ΔΣ over the accessible range.

On **5**, this is the important one. The addendum's (A13) says feedback enters
only through `S(k)`, which cancels identically, so `C` should be feedback-blind.
It is not: cross-feedback is 9.4 per cent against 1.3 cross-code for ΔΣ at
`z ≈ 0.5`. **The mediation hypothesis fails specifically in the strong-feedback
runs** — physically sensible, since AGN process the gas around the very
galaxies being stacked in a way the total matter field does not record. The
consequence is that the addendum's Section 4.4 reframing (that the residual is
a sample-matching problem, fixable by matching `w_gg` between suites) does not
survive: the residual is astrophysical and tracks feedback strength. It stays
tolerable — 9.4 per cent maps to 3.0 per cent on `S` — but it should be booked
as a feedback-dependent prior.

---

## 6. Defects found and how

Recorded because the method that found them is reusable.

| defect | where | how found |
|---|---|---|
| `S_e_*`/`S_t_e_*` suppression written for the electron field, where (A16)/(A17) do not apply — finite, smooth, meaningless | `make_calibration_factor.flatten_for_npz` | code-reviewer, by driving `build_filters`/`flatten_for_npz` directly rather than reading them |
| Masked bins saved as ordinary numbers (`C = 12.5` at `R < R0` for Υ) | same | code-reviewer, by inspecting the saved payload against the mask |
| `add_convention_t` demanded the galaxy/electron pairs even when only `Y_tt` was wanted | `make_calibration_factor` | a test failing on a deliberately minimal amplitude dict |
| Summary table labelled its endpoint columns `1'`/`6'` when they were the first and last *usable* bin, which differs per filter | `make_calibration_factor.report_run` | reading the debug smoke-test output and not recognizing a number |
| `zip(variants, flamingo)` mis-pairs curves with the wrong run's `f_b` once any variant lacks a filter | `plot_task9.make_figure` | code-reviewer, reasoning about a partial-rerun scenario |
| Degenerate-total guard evaluated `0.0 < 0.0` and fell through to a `0/0` divide | `make_task9_spectra.response_quantiles` | code-reviewer, by executing the Υ(`R = R0`) case with warnings enabled |
| Four overstatements in the response document | `cross_correlation_notes_v0.3_response.md` | re-deriving every quoted number from the `.npz` after drafting |
| Filter ranking read into a statistic that cannot support it | analysis | asking whether a 0.042-against-0.047 difference is significant at `N = 4` |

**The pattern worth carrying forward:** two of the three code defects lived in
the *glue* — the functions that decide which quantity feeds which formula — not
in the algebra, which was tested. The algebra had three dedicated identity
tests and none of them failed. Test the wiring, not only the mathematics.

---

## 7. Numbers at a glance

**Calibration factor (ΔΣ, 1′–6′):** `C` = 1.11 → 1.00 at `z ≈ 0.5`, mean
`|C−1|` = 0.052, scatter 4.7 per cent; 0.100 and 5.9 per cent at `z ≈ 0.26`.

**Against Task 1:** `r_bm/r_gb` cross-code 10.9 per cent → `C` cross-code
**1.3 per cent** (`z ≈ 0.5`); 9.7 per cent at `z ≈ 0.26`.

**Gate A:** every filter inside the fixed-transfer threshold; worst 9.6 per cent
(`Υ(1')`, `z ≈ 0.26`). 6 per cent on `C` → 1.9 per cent on `S`.

**Statistical power:** scatter estimates from `N = 4` carry 41 per cent
uncertainty; 7 of 8 filters tied at both redshifts.

**Cross-feedback against cross-code:** feedback larger in 14 of 16 cells.

**Term B:** negative, matches `(f_m + f_b x)² − 1` to under 1 per cent relative
in all four runs. Stochastic residual `2.9e−3`, about 5× the addendum's quoted
`6e−4`.

**Electron against baryon:** ΔΣ scatter 0.095 against 0.047.

**Task 9:** Route B uncorrected `B/truth` = 0.92–0.98 (ΔΣ, `z ≈ 0.5`); Route A
1.39–1.54. Window smearing +0.8 to +1.7 per cent (ΔΣ) against +1.9 to +2.9
(Υ, Y).

**Cost:** 27.9 min for the sweep, 26.3 for the regeneration, 2.1 for Task 9;
one CPU node, `interactive` QOS throughout.

---

## 8. What was skipped, and why

| skipped | reason | cost of the gap |
|---|---|---|
| A third code family (ANTILLES) | not on disk; out of scope for this session | Gate A still rests on two families; the Singh Appendix A discipline stays unexercised |
| `xy` and `xz` projections | only `yz` is cached | across-projection scatter unmeasured, as in Task 1 |
| `T_bp` reconstruction validation | needed only for a lensing leg that does not exist in simulations | the Route B lensing path is unvalidated at the discretization level |
| `check_filter_compensation.py` with the Y transform | cheap, simply not run | the addendum's box-scale prediction for the Y transform is untested |
| Prediction 3 (Σ against ΔΣ at matched `k_50`) | the machinery now exists but the comparison was not drawn | one of five predictions unscored |
| Bin-to-bin covariance | 16 jackknife regions cannot support it | per-radius errors only; Υ and the Y transform correlate bins by construction |
| Beam, `C_beam` per filter | out of scope; `C` is an intrinsic field property | filter-dependent beam correction still to be recomputed for whatever filter is adopted |
| DMO runs | over a day to download, as in Task 4 | `P_hydroCDM/P_DMO = 1` still assumed |
| The electron/baryon decision | two code families cannot adjudicate | Gate B left open on the field definition |

---

## 9. Next steps

Ordered by what unblocks the most.

1. **Cancel the CDM-only non-linear spectrum work.** Next step 1 of
   `tasks_1_to_4_record.md` was to resolve term B; Stage 0 shows B is the
   signal. The correct Convention C chain is halofit(DMO) → `Y_mm^hydro CDM`
   with only the 1–2 per cent back-reaction, as v0.1 Section 4 already said.
2. **Settle the filter set**, which needs either a third code family or the
   matched-`k_50` comparison (prediction 3). Not more apertures and not more
   jackknife regions — Stage 7 shows neither would help.
3. **Re-open the Stage 6 electron/baryon decision** with the Section 5.2
   numbers in hand.
4. **Diagnose the `z ≈ 0.26` cross-code degradation** (1.3 → 9.7 per cent for
   ΔΣ). SHAM matching at the higher number density, or the FLAMINGO `z = 0.30`
   against TNG's `z = 0.26` mismatch, are the two candidates.
5. **Book the feedback dependence of `C`** as a prior width rather than as a
   sample-matching artefact.
6. **Phase 4 onward** unchanged from the previous record: the data-side `Y_gg`,
   the fibre-incompleteness campaign, and the Ondaro-Mallea velocity
   reconstruction question.

---

## 10. File map

**Library.** No `src/` file was created or modified. `rprofiles.py`,
`kernels.py` and `theory.py` are used read-only; the Y-transform harmonic
kernel and the cross-spectrum estimator live in the scripts rather than in
`src/`, so that `data/r_profiles/*.npz` stays reproducible.

**Scripts**, all run from `scripts/`:

| script | purpose |
|---|---|
| `cross_corr/make_calibration_factor.py` | Tasks 7, 8, 10: every filter's amplitudes, `C`, `C_A`, `x`, `S`, both conventions |
| `cross_corr/make_task9_spectra.py` | Task 9: measured 2D auto/cross spectra and the `k_50` mapping |
| `cross_corr/plot_task9.py` | Task 9: the two-row figure and its metrics |
| `cross_corr/runINT_calibration.sh` | interactive runner for the sweep |
| `cross_corr/runINT_task9.sh` | interactive runner for Task 9 |
| `cross_corr/runCPU_calibration.sh` | batch equivalent; superseded by the interactive runner |

**Configs.** `scripts/configs/cross_corr/calibration_z05.yaml` and
`calibration_z026.yaml`, copied verbatim from the Task 1 pair apart from the
`upsilon_r0` and `ytransform_rmax` blocks and the output path.

**Tests.** `tests/test_calibration_factor.py` (22; no simulation data needed
except one on-data spot check that skips without scratch).

**Outputs.** `data/cross_corr_C/calibration_*.npz` (8, committed) and
`task9_spectra_*.npz` (8, committed); figures under
`figures/<yyyy-mm>/<mm-dd>/` (gitignored, regenerable).

**Commits.** `ec8f129` (Tasks 7, 8, 10 and the v0.2/v0.3 documents),
`82526e3` (Task 9).

**Reproduce everything:**

```bash
cd scripts/
salloc -q interactive -C cpu -N 1 -t 1:00:00 -A desi bash cross_corr/runINT_calibration.sh
salloc -q interactive -C cpu -N 1 -t 1:00:00 -A desi bash cross_corr/runINT_task9.sh
cd ../tests/ && pytest test_calibration_factor.py -q     # 22 tests
```
