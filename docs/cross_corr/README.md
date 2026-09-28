# kSZ × clustering baryon–matter cross-correlation programme: docs map

**Status as of 2026-09-21; file layout as of 2026-09-27.** This file is the entry point for every session, human or agent. It says which document to believe for what, where the programme stands, and the rules that code changes must follow. It contains no derivations and copies no numbers it does not need; numbers live in the task records and are cited from there.

Companion files at the same level: `decisions.md` (what has been decided, with status) and `open-items.md` (what has been proposed and not yet done).

---

## 1. One-paragraph orientation

The programme measures $x = P_{bm}/P_{mm}$, the baryon–matter cross-correlation that controls the baryonic suppression of the matter power spectrum, using a kSZ stack around DESI galaxies combined with either galaxy clustering plus a theory matter spectrum (Route A) or galaxy–galaxy lensing (Route B). Both routes need one simulation-calibrated factor, $C = Y_{bm}Y_{gm}/(Y_{mm}Y_{gb})$, and the whole simulation effort exists to establish that $C$ is near 1 and stable across codes and feedback models. It is: $C = 1.11 \to 1.00$ over $1'$ to $6'$ at $z\approx0.5$ with 4.7 per cent cross-run scatter for $\Delta\Sigma$, which propagates to about 1.9 per cent on the suppression. Gate A passes. Gate B is reopened on the electron-versus-baryon target field. Gate C is not reached.

---

## 2. Which document to believe

| document | authoritative for | superseded on | status |
|---|---|---|---|
| `records/tasks_7_to_10_record.md` | every round-two number; gate verdicts; code defects found; commits `ec8f129`, `82526e3`; §11 adds `dba0a68` and `6dbbebd` | nothing | **current, append-only** |
| `records/tasks_1_to_4_record.md` | every round-one number; theory-chain validation; SHAM sample definitions; `data/r_profiles/*.npz` provenance | its Next step 1 (term B, cancelled); its Gate A verdict | **current for numbers, append-only** |
| `records/r_profiles_implementation_plan.md` | Task 1 build log and raw round-one results (A1–A12, R1–R4, T1–T9); the $\Upsilon$ verification, the $R_0$ scan and the $R_0 = 1'$ decision, the stale FLAMINGO r-profiles (U1–U7) | its §1–6, the plan as first written (deviations recorded in A1–A12) | **current for numbers, append-only** |
| `mathematical_formalism.md` | derivations, equation numbers (1)–(70), notation, assumptions ledger (§A), symbol table (§B) | nothing | current; regenerated deliberately, not patched |
| `programme_synthesis.md` | narrative in three passes; provenance table §III.I | its matched-bin figure was corrected 2026-09-20 | current; regenerated deliberately |
| `filter_specification.md` | filter definitions, aperture grid, discretization, normalization, projection, self-pair and error conventions (§2–8) | §1 (Gate B verdict and target field: D-05, D-10); its $R_0 = 1'$ conflicts with D-10's $\Upsilon(R_0=2')$, unresolved; see the status note at its top | current except as noted |
| `archive/cross_correlation_notes_v0.3_response.md` | round-two scientific conclusions as first drawn | Task 9, which it reports unattempted but which was completed; "mediation fails" reading, now one of two live readings (formalism §8.4) | superseded |
| `archive/cross_correlation_notes_v0.2_addendum.md` | Route B, the calibration factor, the mediation proposition, the four-filter analysis, Tasks 7–10 as specified, five advance predictions | predictions 1, 2, 5 falsified; the circular mediation test; the $6\times10^{-4}$ bound; $R_{\max}=5'$ as fiducial | superseded |
| `archive/cross_correlation_notes.md` (v0.1) | the original estimator, physics, Tasks 1–6, Phases 0–7, Gates A–C as designed | point-mass marginalization (retired); near-unity expectation for $r$ (does not hold for compensated filters); $m$ = CDM as the only convention | superseded |
| `archive/r_profiles_task1_spec.md` | the Task 1 engineering spec as first posed | three projections (only `yz` is cached), a CDM particle sweep (CDM is total minus baryon), snapshot 71 out of scope (in scope), its resolution requirements (the existing caches were used): `records/r_profiles_implementation_plan.md` A1–A4 | superseded |

When two documents disagree, the later row in this table wins. When a record and any other document disagree on a number, the record wins.

---

## 3. Where the programme stands

| gate | verdict | basis | caveat |
|---|---|---|---|
| A: transfer viable | **passes**, fixed-transfer route | $C$ cross-run scatter 4.2–9.6% across filters and redshifts; $\Delta\Sigma$ at $z\approx0.5$: 4.7% | a scatter from $N=4$ runs carries 41% relative uncertainty; $\Delta\Sigma$ passes cleanly ([2.8, 6.6]%), $\Upsilon(1')$ at $z\approx0.26$ (9.6%) is undetermined against the 10% threshold |
| B: filter and field frozen | **reopened** on the field | electron target carries $2\times$ the cross-run scatter of the baryon target | filter set provisionally $\{\Delta\Sigma\}$ fiducial, $\Upsilon(R_0=2')$ cross-check, Park et al. $Y$ transform dropped; four runs cannot rank filters |
| C: data-side consistency | not reached | needs Phase 4 ($Y_{gg}$ pair counts) first | |

**Headline numbers** (all from `records/tasks_7_to_10_record.md`, $\Delta\Sigma$, $z\approx0.5$ unless stated):

- $C$: 1.11 at $1'$ → 1.00 at $6'$; mean $|C-1| = 0.052$; jackknife errors an order of magnitude below cross-run scatter.
- Cross-code agreement: 10.9% (old $r_{bm}/r_{gb}$) → 1.3% ($C$). At $z\approx0.26$: 9.7%, unexplained.
- Cross-feedback scatter: 9.4%, larger than cross-code in 14 of 16 filter/redshift cells; monotone in feedback strength (1.105, 1.138, 1.166 at $1'$ for fiducial, Jet, fgas$-8\sigma$).
- Route B uncorrected ($C=1$): 0.92–0.98 of truth. Route A uncorrected: 1.39–1.54.
- Window smearing: $+0.8$ to $+1.7$% ($\Delta\Sigma$) vs $+1.9$ to $+2.9$% ($\Upsilon$, $Y$ transform).
- Round-one "term B" = $Y_{tt}/Y_{mm}-1$ matches $(f_m+f_bx)^2-1$ to $<1$%: it is the signal, not a systematic.
- Electron vs baryon target scatter: 0.095 vs 0.047. $Y_{gb}/Y_{ge}$ at $1'$: 1.21 (TNG) to 2.27 (fgas$-8\sigma$).
- Propagation: 6% on $C$ → 1.9% on $S$, 0.9% on $T=\sqrt S$.

**Runs:** TNG300-1; FLAMINGO L1_m9 fiducial, fgas$-8\sigma$, Jet_fgas$-4\sigma$. Two code families only. SIMBA and Illustris-1 dropped (too few SHAM galaxies).

---

## 4. Rules for code and analysis changes

Violating any of these has already cost a round. Each is justified in the formalism at the cited section.

**Notation** (synthesis §III.A; formalism §5.1, §9.1)
- $Y_{\alpha\beta}$ with field subscripts is a filtered amplitude. $Y(R;R_{\max})$ with radial arguments is the Park et al. transform, a filter. Never a bare "$Y$" for the filter.
- $x(k)=P_{bm}/P_{mm}$ is harmonic-space; $x_{\mathcal F}(R)=Y_{bm}/Y_{mm}$ is what the estimator delivers; $x^{\rm T}$, $x^{\rm T}_{\mathcal F}$ are the total-matter versions. Never a bare $x$ for the filtered ratio.
- Mediation transfer is $\eta(k)$, cross bias is $\beta(k)$. Not $S(k)$, not $b(k)$.
- Field labels: $g$, $e$ (ionized gas), $b$ (all baryons), $m$ (CDM), $t$ (total). $m \equiv t - b$ on the same grid, never a separate particle sweep.

**Conventions**
- Convention C ($m$ = CDM) for Route A. Convention T ($m \to t$) for Route B. Never mixed (formalism §9.2). Confirmation from Uroš pending (`open-items.md` O-19).
- $f_b$ from map means, never from the header (formalism §2.3; FLAMINGO $\Omega_0$ carries neutrinos).
- `SUPPRESSION_FIELD = 'b'`. $S(x_e)$ is meaningless; $C$, $C_A$, $x_e$ for electrons are still written (formalism §9.8).
- Every derived quantity ($r$, $C$, $x$, $S$) is formed **per jackknife realization**, never by Gaussian propagation of marginal errors (formalism §6.5).
- Masks: $\Upsilon(R_0;R_0)\equiv0$, exclude $R=R_0$. Park et al. $Y$ transform: exclude $R\ge0.8R_{\max}$.
- Aperture grid: the 9 linear bins over $1'$–$6'$ are bit-frozen for the $f_{\rm gas}$ paper. Any transform that integrates over $\Delta\Sigma$ (the $Y$ transform, a DoG built from $\Delta\Sigma$) needs a **log-spaced** grid; on the linear grid errors reach 24% (formalism §4.4, archive v0.2 §2.2).
- Any $\Sigma$-from-$\Delta\Sigma$ reconstruction requires the **local** $\Delta\Sigma$; feeding it the pipeline's annulus-mean $\Delta\Sigma$ biases by 43% at $1'$ (formalism §4.4).

**Data products**
- `data/r_profiles/*.npz` are frozen at `dba0a68` (2026-09-10), the last of three regenerations after round one (`91e39d7`, `a07d613`, `dba0a68`; see `records/r_profiles_implementation_plan.md` U3–U7 and `records/tasks_7_to_10_record.md` §11). Round two added `data/cross_corr_C/` without touching `src/`.
- All ten unordered pairs of $\{g,e,b,m\}$ are saved; $Y_{gm}$, $Y_{bm}$, $Y_{mm}$, $Y_{em}$ need no recomputation.
- Numbers are cited from the records, never copied into other documents without "as of DATE, from RECORD".

**Process**
- State predictions in a dated file before a run. Round two was a test rather than a fit because of this.
- An agent finishing a task appends to the relevant record and updates §3 above and `decisions.md`. It does not edit `mathematical_formalism.md` or `programme_synthesis.md`; those are regenerated on request.
- Test the wiring, not only the algebra. Two of three round-two defects were in the functions deciding which quantity feeds which formula; the algebra had three identity tests and passed all of them.

---

## 5. Code and data map

| path | role |
|---|---|
| `src/rprofiles.py` | map-level amplitudes $Y_{\alpha\beta}$, kernels, jackknife, SHAM map construction; `compute_Y_matrix` over all ten pairs |
| `src/kernels.py` | analytic harmonic kernels (formalism §4.2) |
| `src/theory.py` | halofit → $Y_{mm}$ chain (Route A only) |
| `scripts/cross_corr/make_calibration_factor.py` | round-two sweep: $C$, $C_A$, $x$, $S$ per run/sample/filter |
| `scripts/cross_corr/make_task9_spectra.py`, `plot_task9.py` | 2D spectra from cached maps; the Task 9 figure |
| `scripts/cross_corr/make_r_profiles.py`, `plot_r_profiles.py` | round-one r-profiles and the Task 1 figure; since `6dbbebd` its bottom row is $C$ and its Gate A statistic is the cross-code scatter of $C$ |
| `scripts/cross_corr/plot_electron_baryon.py` | Task 3, the $e$ versus $b$ correction |
| `scripts/cross_corr/check_filter_compensation.py`, `check_theory_transfer.py`, `check_projection_depth.py`, `check_resolution.py` | Task 2 box-scale test; Task 4 A/B/C decomposition and depth study; resolution and boundary-tie checks |
| `scripts/cross_corr/check_upsilon_r0.py` | the $R_0$ scan (`records/r_profiles_implementation_plan.md` U6) |
| `scripts/cross_corr/runINT_*.sh`, `runCPU_*.sh` | SLURM runners, submitted from `scripts/`; `runCPU_calibration.sh` is superseded by `runINT_calibration.sh` |
| `scripts/configs/cross_corr/` | `r_profiles_z05.yaml`, `r_profiles_z026.yaml`, `calibration_z05.yaml`, `calibration_z026.yaml` |
| `tests/test_calibration_factor.py` | identity tests; extend for any new kernel |
| `tests/test_rprofiles.py`, `test_kernels.py`, `test_rprofiles_integration.py` | round-one unit tests; the integration test skips without the scratch data |
| `data/r_profiles/*.npz` | round-one amplitudes, frozen at `dba0a68` |
| `data/cross_corr_C/` | round-two outputs |

---

## 6. Directory layout

```
docs/cross_corr/
  README.md                       this file: map, status, rules
  decisions.md                    decision log with status tags
  open-items.md                   proposals with tests and pass criteria
  mathematical_formalism.md       human reference, equations (1)–(70)
  programme_synthesis.md          human narrative in three passes
  filter_specification.md         Phase 0 conventions; status note at its top
  records/
    tasks_1_to_4_record.md               append-only, ground truth for round one
    tasks_7_to_10_record.md              append-only, ground truth for round two
    r_profiles_implementation_plan.md    append-only, Task 1 build log and U1–U7
  archive/
    cross_correlation_notes.md               v0.1, superseded 2026-09
    cross_correlation_notes_v0.2_addendum.md superseded 2026-09
    cross_correlation_notes_v0.3_response.md superseded 2026-09
    r_profiles_task1_spec.md                 superseded 2026-08
```

Each archived file gets a two-line header: `SUPERSEDED. See docs/cross_corr/README.md §2. Kept for provenance.` Nothing in `archive/` is to be treated as current. Documents written before 2026-09-27 cite each other by bare filename; §2 gives every file's location.

---

## 7. Reading order

- **Agent picking up a task:** this file → `decisions.md` → the relevant `open-items.md` entry → the record it cites → the formalism section it cites.
- **Human re-entering after a gap:** `programme_synthesis.md` Part I → this file §3 → `open-items.md`.
- **Anyone needing a derivation or a number:** `mathematical_formalism.md` by equation number; the records by task and stage.
