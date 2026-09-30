# kSZ × clustering baryon–matter cross-correlation programme: docs map

**Status as of 2026-09-27 (Round 3A); file layout as of 2026-09-27.** This file is the entry point for every session, human or agent. It says which document to believe for what, where the programme stands, and the rules that code changes must follow. It contains no derivations and copies no numbers it does not need; numbers live in the task records and are cited from there.

Companion files at the same level: `decisions.md` (what has been decided, with status) and `open-items.md` (what has been proposed and not yet done).

---

## 1. One-paragraph orientation

The programme measures $x = P_{bm}/P_{mm}$, the baryon–matter cross-correlation that controls the baryonic suppression of the matter power spectrum, using a kSZ stack around DESI galaxies combined with either galaxy clustering plus a theory matter spectrum (Route A) or galaxy–galaxy lensing (Route B). Both routes need one simulation-calibrated factor, $C = Y_{bm}Y_{gm}/(Y_{mm}Y_{gb})$, and the whole simulation effort exists to establish that $C$ is near 1 and stable across codes and feedback models. It is: $C = 1.11 \to 1.00$ over $1'$ to $6'$ at $z\approx0.5$ with 4.7 per cent cross-run scatter for $\Delta\Sigma$, which propagates to about 1.9 per cent on the suppression. Gate A passes. Gate B is reopened on the electron-versus-baryon target field. Gate C is not reached. Round 3A (2026-09-27) split $C-1$ into a window term that barely changes with feedback and a genuine mediation failure that carries the feedback ordering, and measured the back-reaction at $z\approx0.5$.

---

## 2. Which document to believe

| document | authoritative for | superseded on | status |
|---|---|---|---|
| `records/round3a_record.md` | every Round 3A number: $C(k)$, the split $C_{\mathcal F} = W_{\mathcal F}M_{\mathcal F}$, the DoG kernel, the back-reaction and the true suppression at $z\approx0.5$, O-03/04/05/06/09/10; the Round 3A predictions scored | nothing | **current, append-only** |
| `predictions/2026-09-27_round3a.md` | the Round 3A predictions and acceptance thresholds, as written before the runs | nothing (scored in `records/round3a_record.md` §4) | frozen |
| `records/tasks_7_to_10_record.md` | every round-two number; gate verdicts; code defects found; commits `ec8f129`, `82526e3`; §11 adds `dba0a68` and `6dbbebd` | the explanation of $\Upsilon$'s $C$ plateau at $9.75'$ (Stage 6, prediction 4): corrected in `records/round3a_record.md` | **current, append-only** |
| `records/tasks_1_to_4_record.md` | every round-one number; theory-chain validation; SHAM sample definitions; `data/r_profiles/*.npz` provenance | its Next step 1 (term B, cancelled); its Gate A verdict | **current for numbers, append-only** |
| `records/r_profiles_implementation_plan.md` | Task 1 build log and raw round-one results (A1–A12, R1–R4, T1–T9); the $\Upsilon$ verification, the $R_0$ scan and the $R_0 = 1'$ decision, the stale FLAMINGO r-profiles (U1–U7) | its §1–6, the plan as first written (deviations recorded in A1–A12) | **current for numbers, append-only** |
| `mathematical_formalism.md` | derivations, equation numbers (1)–(70), notation, assumptions ledger (§A), symbol table (§B) | nothing | current; regenerated deliberately, not patched |
| `programme_synthesis.md` | narrative in three passes; provenance table §III.I | its matched-bin figure was corrected 2026-09-20. As of 2026-09-27 (Round 3A): its reading of the feedback ordering as mediation failure (II.6, II.9, III.C) is refined by D-19; its $\Upsilon(R_0=2')$ (III.F, III.G) is now $1'$ (D-10); its two-sample test (III.H item 3) holds only in harmonic space (O-05); the "two orders of magnitude" behind $\Upsilon$'s plateau (III.D) is 17–19 | current; regenerated deliberately |
| `filter_specification.md` | filter definitions, aperture grid, discretization, normalization, projection, self-pair and error conventions (§2–8) | §1 (Gate B verdict and target field: D-05, D-10); see the status note at its top | current except as noted |
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
| B: filter and field frozen | **reopened** on the field | electron target carries $2\times$ the cross-run scatter of the baryon target, mainly on the feedback axis (Round 3A) | filter set provisionally $\{\Delta\Sigma\}$ fiducial, $\Upsilon(R_0=1')$ cross-check (D-10), Park et al. $Y$ transform dropped; a positive-window DoG does not beat $\Delta\Sigma$ (Round 3A); four runs cannot rank filters |
| C: data-side consistency | not reached | needs Phase 4 ($Y_{gg}$ pair counts) first | |

**Headline numbers** (all from `records/tasks_7_to_10_record.md`, $\Delta\Sigma$, $z\approx0.5$ unless stated):

- $C$: 1.11 at $1'$ → 1.00 at $6'$; mean $|C-1| = 0.052$; jackknife errors an order of magnitude below cross-run scatter.
- Cross-code agreement: 10.9% (old $r_{bm}/r_{gb}$) → 1.3% ($C$). At $z\approx0.26$: 9.7%, unexplained in round two; narrowed down in Round 3A (below).
- Cross-feedback scatter: 9.4%, larger than cross-code in 14 of 16 filter/redshift cells; monotone in feedback strength (1.105, 1.138, 1.166 at $1'$ for fiducial, Jet, fgas$-8\sigma$).
- Route B uncorrected ($C=1$): 0.92–0.98 of truth. Route A uncorrected: 1.39–1.54.
- Window smearing: $+0.8$ to $+1.7$% ($\Delta\Sigma$) vs $+1.9$ to $+2.9$% ($\Upsilon$, $Y$ transform).
- Round-one "term B" = $Y_{tt}/Y_{mm}-1$ matches $(f_m+f_bx)^2-1$ to $<1$%: it is the signal, not a systematic.
- Electron vs baryon target scatter: 0.095 vs 0.047. $Y_{gb}/Y_{ge}$ at $1'$: 1.21 (TNG) to 2.27 (fgas$-8\sigma$).
- Propagation: 6% on $C$ → 1.9% on $S$, 0.9% on $T=\sqrt S$.

**Round 3A** (from `records/round3a_record.md`, as of 2026-09-27; $\Delta\Sigma$, $z\approx0.5$ unless stated):

- Split at $1'$, $C_{\mathcal F} = W_{\mathcal F}M_{\mathcal F}$: window $W$ = 1.03–1.06, barely changing across the FLAMINGO variants (spread 0.005); mediation $M$ = 1.05–1.10, carrying the whole feedback ordering. D-19 resolved.
- $C(k)$ = 1.05–1.12 over $\Delta\Sigma(1')$'s response range: mediation genuinely fails. Galaxies in lower-mass hosts reach 1.2–1.36; those in hosts above $\sim10^{13}\,M_\odot/h$ are close to mediated in TNG300-1 and FLAMINGO fiducial, and 6–10 per cent off in the strong-feedback runs.
- The $z\approx0.26$ cross-code gap (7.1% and 9.3% at $1'$ for the two densities, relative to the two runs' mean $C$) follows the snapshot, not the SHAM number density, and is absent for electrons (0.6–1.6% at $1'$).
- Difference of Gaussians: median $|C-1|$ 1.1–4.2 times $\Delta\Sigma$'s at matched $k_{50}$; no width passes the shrinkage test of formalism §4.5 in all four runs.
- Back-reaction $P^{\rm hydro}_{mm}/P^{\rm DMO}$ = 0.975–1.018 over $0.5\le k\le7\,h/$Mpc; enters $S$ one-to-one (D-21).
- $e\to b$: $Y_{bm}/Y_{em}$ at $1'$ = 1.16–1.85; $x_b/x_{\rm ion}$ at $k=5\,h/$Mpc = 1.2–2.3, 86–96 per cent of the step from stars.
- $\Delta\Sigma$ log-slope $-1.3$ ($z\approx0.5$) and $-1.2$ ($z\approx0.26$), not $-2$; $\Upsilon(1')$ keeps 50–83 per cent of the amplitude. A $C(x)$ line through the FLAMINGO variants does not predict TNG300-1, so the feedback prior (D-14) stands.

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
- The twelve $z\approx0.5$ and $z\approx0.26/0.30$ files in `data/r_profiles/` are frozen at `dba0a68` (2026-09-10), the last of three regenerations after round one (`91e39d7`, `a07d613`, `dba0a68`; see `records/r_profiles_implementation_plan.md` U3–U7 and `records/tasks_7_to_10_record.md` §11). Round two added `data/cross_corr_C/` without touching `src/`.
- The same folder also holds the $z\approx0.75$ and $z\approx1.0$ r-profiles added on 2026-09-30 (TNG300-1 snapshots 57 and 50, FLAMINGO snapshots 62 and 57; configs `r_profiles_z075.yaml`, `r_profiles_z10.yaml`; runners `runCPU_rprofiles_highz_fields.sh`, `runCPU_rprofiles_highz.sh`). They are not part of the frozen set and have no record entry yet.
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
| `scripts/configs/cross_corr/` | `r_profiles_z05.yaml`, `r_profiles_z026.yaml`, `r_profiles_z075.yaml`, `r_profiles_z10.yaml`, `calibration_z05.yaml`, `calibration_z026.yaml` |
| `tests/test_calibration_factor.py` | identity tests; extend for any new kernel |
| `tests/test_rprofiles.py`, `test_kernels.py`, `test_rprofiles_integration.py` | round-one unit tests; the integration test skips without the scratch data |
| `scripts/cross_corr/round3a_lib.py` | Round 3A shared functions: exact Parseval amplitudes, the window/mediation split $C_{\mathcal F} = W_{\mathcal F}M_{\mathcal F}$, the DoG kernel, the 3D component bookkeeping |
| `scripts/cross_corr/make_ck_spectra.py`, `make_dog_calibration.py`, `runCPU_round3a.sh` | Round 3A compute: spectra, $C(k)$, the exact split and the regression check against round two; the DoG sweep |
| `scripts/cross_corr/round3a_diagnostics.py`, `round3a_ck_analysis.py`, `round3a_dog_analysis.py`, `round3a_backreaction.py` | Round 3A analysis, Stages 1–4 (login node) |
| `tests/test_round3a.py` | Round 3A tests: Parseval amplitudes against `compute_Y_matrix`, the split's limits, the DoG kernel, the wiring |
| `data/r_profiles/*.npz` | round-one amplitudes, frozen at `dba0a68`; plus the $z\approx0.75$ and $z\approx1.0$ files (2026-09-30), outside the frozen set |
| `data/cross_corr_C/` | round-two outputs; `round3a/` holds Round 3A's |

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
  predictions/
    2026-09-27_round3a.md                dated predictions, written before Round 3A ran
  records/
    tasks_1_to_4_record.md               append-only, ground truth for round one
    tasks_7_to_10_record.md              append-only, ground truth for round two
    round3a_record.md                    append-only, ground truth for Round 3A
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
