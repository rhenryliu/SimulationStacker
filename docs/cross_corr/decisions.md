# Decision log

**As of 2026-09-27** (Round 3A: D-05, D-10, D-14, D-19 updated; D-21 added). One entry per decision. Status is one of **settled**, **provisional** (settled unless a named open item overturns it), **open**, **reopened**, **superseded**. Entries are never deleted; a reversed decision gets its status changed and a pointer to what replaced it. Evidence points at the record or document that holds the numbers; nothing here is a number's primary home.

Read with `README.md` (map and status) and `open-items.md` (what could change these).

---

### D-01 The estimator is a map-level average, not a stamp stack
- **Status:** settled
- **Date:** round one (early 2026-09)
- **Decision:** every $Y_{\alpha\beta}(R;\mathcal F) = \langle \mathcal F_R[\delta_\alpha]\,\delta_\beta\rangle_{\rm map}$ is computed as an FFT-filtered map multiplied by a second map and averaged over pixels, on a periodic box.
- **Why:** identical to stacking for galaxy-crossed pairs (formalism §5.3) and the only way to reach the field–field pairs $bm$, $mm$, $bb$ that the calibration needs.
- **Evidence:** `records/tasks_1_to_4_record.md`; validated against `stack_on_array`.

### D-02 The CDM field is defined as total minus baryon; $f_b$ from map means
- **Status:** settled
- **Date:** round one
- **Decision:** $\rho_m \equiv \rho_t - \rho_b$ on the same grid, no separate DM particle sweep; $f_b = \bar\rho_b/\bar\rho_t$ from the maps, never `OmegaBaryon/Omega0`.
- **Why:** makes $\delta_t = f_m\delta_m + f_b\delta_b$ exact (formalism Eq. 4); the header value is wrong at the per-cent level for FLAMINGO because $\Omega_0$ includes neutrinos.
- **Evidence:** identity verified to $10^{-10}$; a 5 per cent perturbation of $f_b$ breaks it (`records/tasks_7_to_10_record.md`).

### D-03 SIMBA-100 and Illustris-1 dropped
- **Status:** settled
- **Date:** round one
- **Decision:** the run set is TNG300-1 plus FLAMINGO L1_m9 fiducial, fgas$-8\sigma$, Jet_fgas$-4\sigma$.
- **Why:** 500 and 210 SHAM galaxies respectively; their jackknife errors exceeded the cross-code scatter, so a naive four-code Gate A read as a 3 to 55 per cent failure that was sample size, not physics.
- **Evidence:** `records/tasks_1_to_4_record.md`.

### D-04 The $\Sigma$ filter is not used for amplitudes that meet theory
- **Status:** settled
- **Date:** round one, Gate B; reaffirmed 2026-09-20
- **Decision:** $\Sigma$ (annulus mean) is excluded from any amplitude compared against a theory prediction or ported between volumes. It may be computed as a diagnostic.
- **Why:** uncompensated; its amplitude depends on the box scale (8.7 per cent shift under a box-scale low-$k$ cut vs 0.014 per cent for compensated filters), and its median wavenumber moves by a factor 1.8 with the assumed spectral slope (formalism §4.2, synthesis §III.D).
- **Note:** $\Sigma$ scored well on $|C-1|$ at $z\approx0.5$ in round two; that does not reopen this, because the objection is to the amplitudes, which the theory transfer needs, not to the coefficients.

### D-05 Target field: electron ($e$) in round one; reopened in round two
- **Status:** **reopened**; current lean is the baryon target $b$
- **Date:** round one (Gate B, chose $e$); reopened in `records/tasks_7_to_10_record.md` (electron-versus-baryon comparison)
- **Decision as it stood:** target $P_{em}/P_{mm}$ because the kSZ measures electrons and it needs no $e\to b$ transfer.
- **Why reopened:** the electron target carries twice the cross-run scatter of the baryon target (0.095 vs 0.047, $\Delta\Sigma$, $z\approx0.5$). The suppression algebra also requires the baryon field, not a subset of it (formalism §9.8), so the $e\to b$ step is not optional under either choice; the question is only where it is calibrated.
- **Blocked on:** `open-items.md` O-07 (O-04 done).
- **Round 3A evidence (2026-09-27, `records/round3a_record.md`):** the "twice the scatter" comes mainly from the feedback axis. Across FLAMINGO variants the electron $C$ moves 0.178 against 0.089 for baryons ($z\approx0.5$). Across codes the electron target is worse at $z\approx0.5$ (0.032 against 0.013) and better at $z\approx0.26$ (0.037 against 0.093). The electrons' mediation factor at $1'$ (1.10–1.28) exceeds the baryons' (1.05–1.10). The $e\to b$ step, $x_b/x_{\rm ion}$ = 1.2–2.3 at $k=5\,h/$Mpc, is dominated by stars (86–96 per cent of $x_b - x_{\rm ion}$), which the stellar mass function constrains externally. The evidence favours the baryon target with an externally constrained stellar term, but the decision is the user's.

### D-06 Point-mass marginalization retired
- **Status:** settled
- **Date:** 2026-09-02 (U. Seljak), formalized in v0.2
- **Decision:** no point-mass nuisance parameter or $\Sigma$ reconstruction with one; Task 2b of v0.1 is closed.
- **Why:** Prat et al. (2023) show point-mass marginalization with an infinite prior, mode projection and $\Upsilon$ are equivalent. Cite it correctly: the equivalence is in their Figs. 3–4 and Appendix A at the posterior level, not Fig. 1, and it is a statement about information for cosmological parameters at 6–8 Mpc$/h$ scale cuts, not about filter behaviour at $1'$–$6'$.
- **Evidence:** `archive/cross_correlation_notes_v0.2_addendum.md` §3.

### D-07 Route B adopted; the four-amplitude $C$ is the calibration factor
- **Status:** settled
- **Date:** 2026-09-03 (U. Seljak); confirmed by round two
- **Decision:** $x_{\mathcal F} = C\cdot Y_{gb}/Y_{gm}$ with $C = Y_{bm}Y_{gm}/(Y_{mm}Y_{gb})$ is a live route alongside Route A. $C$ is the primary reported quantity; the individual $r$'s are diagnostics only, since they are not bounded by unity for compensated filters (formalism §7).
- **Why:** $Y_{bb}$ and $Y_{gg}$ cancel from $C$, so it is convention-free and numerically well conditioned; it replaced a 22–45 per cent transfer with a 0–11 per cent one and collapsed the cross-code disagreement from 10.9 to 1.3 per cent at $z\approx0.5$.
- **Evidence:** `records/tasks_7_to_10_record.md`.

### D-08 Convention C for Route A, Convention T for Route B
- **Status:** provisional; confirmation from U. Seljak pending (O-19)
- **Date:** recommended in `records/tasks_7_to_10_record.md`; formalized 2026-09-20
- **Decision:** Route A keeps $m$ = CDM so that halofit on a DMO cosmology models $Y_{mm}$. Route B sets $m\to t$ so its observable is exactly the kSZ-to-lensing ratio the $f_{\rm gas}$ paper measures, with no theory spectrum.
- **Why:** lensing measures total matter; using Convention C for Route B would require a feedback-dependent $Y_{gt}/Y_{gm}$ correction. $C$ and $C^{\rm T}$ agree to about one per cent (formalism §9.2).

### D-09 Round-one "term B" is the signal; CDM-only non-linear prescription cancelled
- **Status:** settled
- **Date:** round two
- **Decision:** the 3–17 per cent "CDM versus total matter" term of round one is $Y_{tt}/Y_{mm}-1$, which matches $(f_m+f_bx)^2-1$ to under one per cent in all four runs. It is the suppression itself. The correct Convention C theory chain is halofit(DMO) $\to Y^{\rm hydro\,CDM}_{mm}$ with only the 1–2 per cent back-reaction.
- **Supersedes:** `records/tasks_1_to_4_record.md` Next step 1.

### D-10 Filter set: $\Delta\Sigma$ fiducial, $\Upsilon(R_0=1')$ cross-check, Park et al. $Y$ transform dropped
- **Status:** provisional
- **Date:** `records/tasks_7_to_10_record.md` recommendation; caveats added 2026-09-20; $R_0$ set to $1'$ 2026-09-27
- **Decision:** as stated.
- **$R_0$ (2026-09-27, user decision):** the cross-check is $\Upsilon(R_0 = 1')$, as in the $R_0$ scan (`records/r_profiles_implementation_plan.md` U6), the r-profile configs and `filter_specification.md` §2; the round-two record's $2'$ recommendation is superseded. On the pooled four-run scatter of $C$, $\Upsilon(1')$ was the one filter round two could disfavour (1.7σ at $z\approx0.5$).
- **Round 3A (2026-09-27):** O-09 finds the $\Delta\Sigma$ log-slope near $-1.3$, not $-2$, so $\Upsilon(1')$ keeps 50–83 per cent of the $\Delta\Sigma$ amplitude over $2.25'$–$9.75'$ and is not a near-cancellation: keep it, with that caveat. O-10 finds that no filter ranking survives comparison at matched $k_{50}$ as a collapse. O-02 finds that a positive-window DoG does not shrink $|C-1|$: it is 1.1–4.2 times $\Delta\Sigma$'s at matched $k_{50}$, because most of $C-1$ is mediation failure, which no kernel removes; $\Delta\Sigma$ stays fiducial. See `records/round3a_record.md`.
- **Why:** $\Delta\Sigma$ has the smallest window smearing ($+0.8$ to $+1.7$ per cent), $C\to1$ at large $R$, and the most usable bins. The $Y$ transform loses to both incumbents on $|C-1|$ at $z\approx0.5$ and its best configuration ($R_{\max}=9'$) references an aperture outside the data range.
- **Caveats that could overturn it:** four runs cannot rank filters (41 per cent sampling uncertainty on any scatter); the $|C-1|$ comparison was not on matched aperture ranges (factor 1.5, not 2.3); $\Upsilon$ may be a near-cancellation at the measured log-slope, which would argue against retaining it (O-09); a positive-window DoG may dominate $\Delta\Sigma$ outright (O-02). See formalism §4.5.

### D-11 Every derived quantity is formed per jackknife realization
- **Status:** settled
- **Date:** round one (Task 1 convention); enforced in `make_calibration_factor.py`
- **Decision:** $r$, $C$, $x$, $S$ are computed inside each of the 16 block-deleted realizations and the scatter taken; never Gaussian propagation of marginal errors on the amplitudes.
- **Why:** the amplitudes entering $C$ are strongly correlated; propagation misestimates the error by a large factor (formalism §6.5).

### D-12 Mask rules
- **Status:** settled
- **Decision:** exclude $R=R_0$ for $\Upsilon$ (identically zero) and $R\ge0.8R_{\max}$ for the Park et al. $Y$ transform (vanishes at $R_{\max}$, uninformative near it).

### D-13 `SUPPRESSION_FIELD = 'b'`
- **Status:** settled
- **Date:** round two commit gate (critical defect fix)
- **Decision:** $S$ is computed only from the baryon field. $S(x_e)$ arrays are not written. $C$, $C_A$, $x$ for electrons are still written.
- **Why:** the suppression algebra descends from $\delta_t = f_m\delta_m + f_b\delta_b$, which requires the gas field to complete the mass budget; electrons are a subset (formalism §9.8).

### D-14 Feedback dependence of $C$ booked as a prior width
- **Status:** provisional (O-03 failed, 2026-09-27)
- **Date:** `records/tasks_7_to_10_record.md` Next step 5
- **Decision as it stands:** treat the 9.4 per cent cross-feedback scatter as a prior on $C$.
- **Why open:** the dependence is monotone in feedback strength and may be a window artefact rather than physics (formalism §8.4); if $C$ regresses tightly on $x$ across runs the prior can be replaced by a relation.
- **Round 3A (2026-09-27):** O-03 fails. A straight line $C(x)$ through the three FLAMINGO variants mispredicts TNG300-1 by $+0.023$ to $+0.071$ at every aperture over $1'$–$6'$ at $z\approx0.5$, more than half the cross-feedback spread, so no one-parameter relation transfers across codes; the prior stands. See `records/round3a_record.md`.

### D-15 Notation
- **Status:** settled
- **Date:** 2026-09-20/21
- **Decision:** $Y_{\alpha\beta}$ (field subscripts) is an amplitude; $Y(R;R_{\max})$ (radial arguments) is the Park et al. transform; $x(k)$, $x_{\mathcal F}(R)$, $x^{\rm T}$, $x^{\rm T}_{\mathcal F}$ as in formalism §9.1; mediation transfer $\eta(k)$, cross bias $\beta(k)$. The earlier $S(k)$ and $b(k)$ for those two are retired.
- **Why:** the collisions with the suppression $S$ and the baryon label $b$, and between the filtered and harmonic $x$, were the source of the confusion in every review pass.

### D-16 Aperture grids
- **Status:** settled
- **Decision:** the 9 linear bins over $1'$–$6'$ with $\delta R = 0.75'$ are bit-frozen for the $f_{\rm gas}$ paper. Any transform that integrates $\Delta\Sigma$ over radius uses a log-spaced grid; any $\Sigma$-from-$\Delta\Sigma$ reconstruction uses the local, not annulus-mean, $\Delta\Sigma$.
- **Why:** 24 per cent reconstruction error and 10–22 per cent point-mass leakage on the linear grid; 43 per cent bias from the annulus-mean input (formalism §4.4; archive v0.2 §2.2–2.3).

### D-17 $R_{\max} = 5'$ as the $Y$-transform fiducial
- **Status:** superseded by D-10
- **Date:** set in v0.2; superseded in round two
- **Why:** the $R_{\max}$ scan preferred $9'$, outside the data range, and the transform was dropped.

### D-18 Gate A verdict
- **Status:** settled, with a recorded caveat
- **Date:** `records/tasks_7_to_10_record.md`; caveat 2026-09-20
- **Decision:** Gate A passes on the fixed-transfer route on two code families.
- **Caveat:** any scatter from $N=4$ carries $1/\sqrt{2(N-1)} = 41$ per cent relative uncertainty, and the four runs are one code plus three feedback variants of another, so they are not exchangeable draws. $\Delta\Sigma$ at 4.7 per cent passes cleanly; $\Upsilon(1')$ at $z\approx0.26$ at 9.6 per cent is undetermined against the 10 per cent threshold. The two-axis decomposition (cross-code, cross-feedback) is the primary statistic, not the pooled four-run scatter.

### D-19 "Mediation fails in the strong-feedback runs" is not a decision
- **Status:** resolved 2026-09-27 (Round 3A): both readings hold, for different parts of $C_{\mathcal F}-1$
- **Resolution:** the exact split $C_{\mathcal F} = W_{\mathcal F}M_{\mathcal F}$ (`records/round3a_record.md`, Stage 2) gives, for $\Delta\Sigma$ at $1'$, a window term of $+3$ to $+8$ per cent and a mediation term of $+4$ to $+16$ per cent. The window term is nearly feedback-independent (1.051, 1.053, 1.056 for FLAMINGO fid, Jet, fgas$-8\sigma$ at $z\approx0.5$). The monotone feedback ordering is carried entirely by $M_{\mathcal F}$ (1.051, 1.081, 1.104). $C(k)$ itself is 1.05–1.12 over $\Delta\Sigma(1')$'s response range, and far from sample-invariant (up to 1.36 for galaxies in lower-mass hosts). Mediation genuinely fails, most plausibly through halo-mass-dependent gas depletion. The window term explains the sign reversals between filters.
- **Date:** asserted in `archive/cross_correlation_notes_v0.3_response.md`; demoted 2026-09-20
- **What stands:** the measured $C_{\mathcal F}$ departs from 1 by about 5 per cent, monotone in feedback strength.
- **Why open:** the filtered $C$ departs from 1 even under exact mediation, through a window term that also predicts the monotone feedback dependence (formalism §8.4). Two readings are live; O-01 and O-02 separate them.

### D-20 Process
- **Status:** settled
- **Decision:** predictions are written to a dated file before any run. An agent finishing a task appends to the relevant record and updates `README.md` §3 and this file; it does not edit the formalism or synthesis, which are regenerated on request. Numbers live in the records and are cited elsewhere with a date and source. Superseded documents move to `archive/` with a header line and are never deleted.

### D-21 The back-reaction is measured at $z\approx0.5$, not adopted
- **Status:** settled at $z\approx0.5$; open at $z\approx0.26$
- **Date:** 2026-09-27 (`records/round3a_record.md`, Stage 4)
- **Decision:** the suppression is $S = (f_m+f_bx)^2 \times B(k)$ with the measured back-reaction $B = P^{\rm hydro}_{mm}/P^{\rm DMO}$, rather than the adopted $B = 1$ of formalism Eq. (62). At $z\approx0.5$, $B$ lies within $\pm2.5$ per cent of unity for all four runs over $0.5\le k\le7\,h/$Mpc, with a run-dependent sign.
- **Why:** DMO references for all four runs are on disk since 2026-09-24, with 3D component spectra at $z\approx0.5$ from the unbound-gas pipeline. $B$ enters $S$ one-to-one, not suppressed by $f_b$, so at the per-cent level it is as large as the whole calibration budget (6 per cent on $C$ gives 1.9 per cent on $S$).
- **Open:** $z\approx0.26$ needs the hydro component spectra at snapshots 71 (FLAMINGO) and 80 (TNG300-1); the DMO spectra exist.
