# Open items

**As of 2026-09-21.** Everything proposed and not yet done, ordered by priority. Each entry says what it decides, how to do it, what counts as pass or fail, what it costs, and where it came from. Provenance distinguishes items in the task records (agreed next steps) from items raised in review (proposals, not decisions). An item is closed by appending its outcome to the relevant record and updating `decisions.md`; it is not deleted here, its status changes.

Priority 1: cheap and decides something that is currently blocking. Priority 2: cheap or moderate, informative. Priority 3: needed before data or expensive. Priority 4: deferred.

---

## Priority 1

### O-01 Direct $C(k)$ from the measured spectra
- **Status:** proposed 2026-09-20 (review), unimplemented
- **Decides:** D-19, and through it D-14. Whether the 5 per cent departure of $C_{\mathcal F}$ from 1 and its feedback dependence are window smearing or genuine mediation failure.
- **How:** compute $C(k) = P_{bm}P_{gm}/(P_{mm}P_{gb})$ from the 2D spectra `make_task9_spectra.py` already measures; the galaxy pairs $P_{gm}$, $P_{gb}$ need adding. Plot per run against $k$, and overlay the four filters' $C_{\mathcal F}$ against $k_{50}(R;\mathcal F)$.
- **Pass / fail:** $C(k)\approx1$ while $C_{\mathcal F}$ deviates → window term; respond with a narrower window (O-02), not a prior. $C(k)$ deviates and tracks feedback → mediation genuinely fails; D-14 stands. If one $C(k)$ underlies everything, the four filters' curves collapse onto it when plotted against $k_{50}$.
- **Cost:** hours; post-processing plus one spectrum pair.
- **Refs:** formalism §8.4, Eq. (48); synthesis §III.H item 2.

### O-02 Difference-of-Gaussians kernel
- **Status:** proposed 2026-09-21 (review), unimplemented
- **Decides:** D-19 from the other side; and whether D-10's fiducial should change.
- **How:** add $W_{\rm DoG}$ (formalism Eq. 22) to `rprofiles.py` with $\sigma_1,\sigma_2$ matched to $\Delta\Sigma$'s $k_{50}$ at each aperture (e.g. $\sigma_1=0.58'$, $\sigma_2=1.15'$ for the $1'$-equivalent). Rerun the Task 7 sweep. Extend `tests/test_calibration_factor.py` with $\hat W_{\rm DoG}(0)=0$ and $\hat W_{\rm DoG}>0$ checks.
- **Pass / fail:** under a positive window $r\le1$ strictly and the window term cannot flip sign. If $|C-1|$ and its monotone feedback dependence shrink substantially, the deviation was the window; if they persist at 5–10 per cent, it is the central galaxy–gas coupling and a smooth point-mass-blind kernel (difference of ring-Gaussians) is next.
- **Cost:** one kernel, one sweep. Lensing-leg implementation is the aperture-mass estimator with $Q_{\rm DoG}$ (formalism Eq. 26), catalogue-level; existing $1'$–$6'$ shear grid supports DoGs to the $2'$-equivalent.
- **Refs:** formalism §4.5.

### O-03 Regress $C$ on $x$ across runs
- **Status:** proposed 2026-09-18 (review of v0.3), unimplemented
- **Decides:** D-14. Cross-feedback scatter (9.4 per cent) is the dominant calibration term.
- **How:** from `data/cross_corr_C/`, plot $C(R)$ against $x_{\mathcal F}(R)$ per aperture across the four runs. If a tight one-parameter relation exists, adopt $C = C(x)$ and solve self-consistently (one Newton step suffices since $C\approx1$).
- **Pass / fail:** residual scatter about the relation well below 9.4 per cent → replaces the prior. No relation → D-14 stands.
- **Cost:** post-processing only.
- **Refs:** synthesis §III.H item 2.

### O-19 Confirm Convention T for Route B with U. Seljak
- **Status:** open, communication
- **Decides:** D-08 from provisional to settled.
- **How:** state that "$Y_{gm}$" in his 2026-09-03 message is read as the lensing total-matter amplitude $Y_{gt}$, and that Route B is run entirely in Convention T with no theory spectrum.
- **Cost:** one message.

---

## Priority 2

### O-04 Plot the matter-crossed $e\to b$ correction $Y_{bm}/Y_{em}$
- **Status:** proposed 2026-09-20 (review), unimplemented
- **Decides:** input to D-05 (O-07). The chain needs $Y_{bm}/Y_{em}$; round one reported the galaxy-crossed $Y_{gb}/Y_{ge}$ (1.21–2.27 at $1'$), which is not the same number and should be the larger.
- **How:** both amplitudes are already in `data/r_profiles/*.npz`; one division, one figure beside the Task 3 figure.
- **Cost:** minutes.
- **Refs:** formalism §9.8.

### O-05 Two-sample invariance test of $Y_{gb}/Y_{gm}$
- **Status:** proposed 2026-09-18 (review), unimplemented
- **Decides:** an independent test of mediation (D-19) that does not use $Y_{mm}$ or $Y_{bm}$, and that runs on observables alone, so it doubles as a data-side null test.
- **How:** mediation requires $Y_{gb}/Y_{gm} = \eta$ with no galaxy-sample dependence. Build two SHAM samples at one snapshot (split by halo mass, or two number densities) and compare the ratio at matched aperture and filter.
- **Pass / fail:** ratios agree within jackknife → mediation holds at that level. Disagree → direct galaxy–gas coupling, strongest in the high-mass split.
- **Cost:** one extra SHAM sample per run; existing sweep.
- **Refs:** synthesis §III.H item 3.

### O-06 Separate redshift from sample in the cross-code split
- **Status:** proposed 2026-09-18 (review), unimplemented
- **Decides:** why cross-code agreement is 1.3 per cent at $z\approx0.5$ but 9.7 per cent at $z\approx0.26$, and the $|C-1|$ reversal between redshifts. Also subsumes the snapshot-mismatch question (FLAMINGO $z=0.30$ vs TNG $z=0.26$).
- **How:** recompute $C$ at $z\approx0.5$ with the BGS-like number density ($1\times10^{-3}$), and at $z\approx0.26/0.30$ with the LRG-like density. If the 9.7 per cent follows the density, it is SHAM matching; if it follows the snapshot, it is redshift.
- **Cost:** two SHAM samples, existing sweep. Far cheaper and more decisive than a third code.
- **Refs:** synthesis §III.H item 4; `records/tasks_7_to_10_record.md` (cross-code discussion).

### O-07 Reopen the electron-versus-baryon target decision
- **Status:** open (`records/tasks_7_to_10_record.md` Next step 2)
- **Decides:** D-05.
- **How:** with O-04 in hand, weigh: the electron target's $2\times$ scatter; the physical reading that the extra scatter is the codes' subgrid partition of baryons into stars, cold and ionized gas, which differs far more between TNG and FLAMINGO than the total baryon distribution does; and that the $e\to b$ step is externally constrainable through the stellar mass function whereas the electron field's code dependence is not.
- **Pass / fail:** a decision, recorded in `decisions.md` D-05.
- **Refs:** synthesis §III.H item 5.

### O-08 Check the lensing-leg ring weight in the $f_{\rm gas}$ code
- **Status:** proposed 2026-09-21 (review), unimplemented
- **Decides:** whether the kSZ and lensing legs apply the same functional. The pipeline's annulus-mean $\Delta\Sigma$ is the ring integral of $Q\,\Delta\Sigma_{\rm local}$, not the flat annulus average of $\gamma_t$; for a locally flat $\Delta\Sigma$ the two differ by a factor 1.66 at $R=1'$, $\delta R=0.75'$.
- **How:** read the lensing estimator; determine whether it applies the duality weight $Q$ or a flat average. If flat, the $f_{\rm gas}$ ratio's inner bins carry a gradient-term bias that the simulation forward model may or may not reproduce.
- **Cost:** code reading; possibly a correction.
- **Refs:** formalism §4.5, Eq. (25).

### O-09 Measure the $\Delta\Sigma$ log-slope; test whether $\Upsilon$ is a near-cancellation
- **Status:** proposed 2026-09-18 (review), unimplemented
- **Decides:** whether to keep $\Upsilon(2')$ as the cross-check in D-10. If the slope is near $-2$ over $1'$–$10'$, $\Upsilon$ subtracts an order-unity fraction of the signal at every radius, and a point-mass cross-check that removes the signal is not a cross-check.
- **How:** fit the log-slope of $Y^{(\Delta\Sigma)}_{gm}$ over $1'$–$9.75'$; report $\Upsilon/\Delta\Sigma$ amplitude against $R$; check that the jackknife errors on $C^{(\Upsilon)}$ are inflated by the same factor, as they must be if the amplitude is a few per cent of $\Delta\Sigma$'s.
- **Pass / fail:** slope $\approx-2$ → retire $\Upsilon$; replace the cross-check with a forward-modelled central stellar component. Slope $\approx-1.5$ → $\Upsilon$ subtracts ~30 per cent at $9.75'$; keep with that caveat.
- **Cost:** post-processing.
- **Refs:** synthesis §III.D, §III.H item 6.

---

## Priority 3

### O-10 Report $|C-1|$ on matched ranges and at matched $k_{50}$; score prediction 3
- **Status:** partially done (Task 9 built the $k_{50}$ machinery); comparison not made
- **Decides:** whether any filter ranking survives fair comparison. The quoted $\Delta\Sigma$-vs-$Y(5')$ gap is 1.5, not 2.3, on matched bins.
- **How:** recompute mean $|C-1|$ on the intersection of usable bins per filter pair; plot $C$ against $k_{50}(R;\mathcal F)$ for all filters on one axis.
- **Refs:** synthesis §III.D; `records/tasks_7_to_10_record.md` Next step 3.

### O-11 Third code family
- **Status:** open (`records/tasks_7_to_10_record.md` Next step 4)
- **Decides:** the Singh et al. Appendix A discipline (priors from one family bias another at 2–3 per cent) cannot be exercised with two families. Also the only way to raise $N$ above four for any scatter statistic.
- **How:** SIMBA at a larger box, or another suite with $\ge$ several thousand SHAM galaxies at the LRG-like density.
- **Cost:** high. Run O-06 first; it may explain the $z\approx0.26$ discrepancy without this.

### O-12 Beam correction for the adopted filter
- **Status:** open; needed before any data application
- **Decides:** $C_{\rm beam}(R;\mathcal F)$ is filter-dependent and does not transfer from the $\Delta\Sigma$ version (formalism Eq. 65).
- **How:** forward-model the ACT beam into simulated maps and refilter; or, for a DoG, use the analytic composition $\sigma_i^2\to\sigma_i^2+\sigma_b^2$ (formalism Eq. 24). Check `use_sim_scatter` behaviour with any new filter.

### O-13 Sky-side projection: Limber and redshift evolution
- **Status:** open, unquantified beyond geometry
- **Decides:** ledger row 19. The box result $P_{\rm 2D}=P_{\rm 3D}/L$ is exact; on the sky the data's $x_{\mathcal F}$ averages over $k$ and $z$. Geometric spread is $\pm0.18$ in $\ln k$ across LRG bin 1 (4 per cent window broadening); evolution of $x(k,z)$ across $0.4<z<0.6$ is unmeasured.
- **How:** compute $C$ and $x_{\mathcal F}$ at two snapshots bracketing the sample and interpolate; compare with the single-snapshot calibration.
- **Refs:** formalism §3.2, ledger row 19.

### O-15 Phase 4: data-side $Y_{gg}$ and fibre incompleteness
- **Status:** open, unchanged from v0.1 (Route A only)
- **Decides:** whether Route A is viable at $1'$–$6'$, below the DESI fibre patrol radius.
- **How:** pair-count $w_{gg}(r_p)$ with the same filter; quantify incompleteness at arcminute scales; it enters as $Y_{gg}^{-1/2}$ and mimics a scale-dependent $r$.

### O-16 Velocity-reconstruction suppression on the kSZ leg
- **Status:** open, unchanged (Ondaro-Mallea et al. 2026)
- **Decides:** a 10–20 per cent multiplicative factor on $Y_{gb}$ with no cancellation now that lensing is out of Route A; whether it is scale-dependent over $1'$–$6'$ is unresolved.

### O-22 Snapshot mismatch, FLAMINGO $z=0.30$ vs TNG $z=0.26$
- **Status:** open; folded into O-06
- **Decides:** one of two candidate explanations for the cross-code degradation at the lower redshift.

---

## Priority 4

### O-14 DMO run for the back-reaction
- **Status:** deferred (over a day to download)
- **Decides:** ledger row 13; the only assumption in formalism §9 that is neither exact nor measured (1–2 per cent).

### O-17 Bin-to-bin covariance
- **Status:** open
- **Decides:** 16 jackknife regions cannot support one; needed for any $\chi^2$ over apertures, and more for filters that correlate bins by construction.

### O-18 Second projection axis
- **Status:** open
- **Decides:** across-projection scatter is unmeasured; only `yz` is cached.

### O-20 Box-scale compensation test for the Park et al. $Y$ transform
- **Status:** open, low priority since the transform was dropped (D-10)
- **How:** rerun `check_filter_compensation.py` with the transform; prediction is a shift between $\Delta\Sigma$'s 0.014 per cent and $\Sigma$'s 8.7 per cent, in proportion to the low-$k$ coefficients.

### O-21 Validate a $\Sigma$-from-$\Delta\Sigma$ reconstruction on the lensing leg
- **Status:** open, low priority; the DoG's aperture-mass duality (O-02) makes it unnecessary for that kernel
- **How:** apply $\mathbf T_{\rm bp}$ to simulated $\Delta\Sigma$ binned as the data and compare with the directly measured $Y(R;R_{\max})$; log grid required.
