# Round 3A record: window smearing or mediation failure?

**Status as of 2026-09-27.** Branch `main`; commits listed in §10. Predictions, written before any run: `../predictions/2026-09-27_round3a.md`. The operational log of the session (commands, job IDs, dead ends) is `NOTES/cross_corr/session_log_2026-09-27_round3a.md`, which is not committed.

## Summary

Round 3A asked why the calibration factor $C$ departs from 1, and answered it.

- **Both effects are real.** The filtered $C$ splits exactly into a window part $W$ and a mediation part $M$. For $\Delta\Sigma$ at $1'$, $W$ adds 3–8 per cent and $M$ adds 4–16 per cent. The window part barely changes across the FLAMINGO feedback variants (spread 0.005 at $z\approx0.5$); the feedback ordering of $C$ lives entirely in $M$. Decision D-19 is resolved.
- **Mediation genuinely fails.** The harmonic-space $C(k)$ is 1.05–1.12 over $\Delta\Sigma(1')$'s response range, and 1.2–1.36 for galaxies in lower-mass hosts. The gas around the stacked galaxies is more depleted than the matter field predicts, most plausibly because depletion depends on halo mass.
- **A positive-window filter does not help.** The difference of Gaussians gives 1.1–4 times $\Delta\Sigma$'s $|C-1|$ at matched wavenumber. $\Delta\Sigma$'s convergence $C\to1$ at large aperture is partly its signed window cancelling mediation failure. $\Delta\Sigma$ stays the fiducial filter.
- **The $z\approx0.26$ cross-code gap follows the snapshot, not the galaxy number density**, and it is absent for electrons.
- **The back-reaction is now measured at $z\approx0.5$**, within $\pm2.5$ per cent. It enters the suppression one-to-one, so it is as large as the whole calibration budget (new decision D-21).
- **The electron-to-baryon step** is 1.2–2.3 at $k=5\,h/$Mpc, and stars carry 86–96 per cent of it. The electron target's doubled scatter comes mainly from the feedback axis.
- **Smaller results.**
  - The $\Delta\Sigma$ log-slope is $-1.3$, not $-2$, so $\Upsilon(1')$ keeps 50–83 per cent of the amplitude and stays as the cross-check.
  - A $C(x)$ relation does not transfer across codes, so the feedback prior stays.
  - The round-two explanation of $\Upsilon$'s plateau is corrected.
- **The acceptance checks passed on the route the results use.** The new code reproduces round two's amplitudes to $1.5\times10^{-12}$ and its $C_{\mathcal F}$ to $5\times10^{-12}$. An independent binned route agrees to about $10^{-3}$ for the directly computed kernels; for the derived $\Upsilon$ it reaches 0.6 per cent in one run, just outside the pre-registered 0.5 per cent, which does not touch the split (§4).

---

## 0. What this document is

A standalone record of Round 3A of the simulation programme: what was asked, what was built, what came out, what was skipped, and what comes next. It succeeds `tasks_7_to_10_record.md` and is written so that a reader, or a fresh session, can pick up the work without the conversation that produced it. It is append-only, like the other records: corrections go at the end.

Companion documents: `../README.md` (map, status, rules), `../decisions.md` (decisions D-01 to D-21), `../open-items.md`, `../mathematical_formalism.md` (equation numbers below refer to it), and the predictions file above.

---

## 1. The question

Round two measured the calibration factor $C_{\mathcal F} = Y_{bm}Y_{gm}/(Y_{mm}Y_{gb})$ at 1.11 falling to 1.00 over $1'$–$6'$ ($\Delta\Sigma$, $z\approx0.5$), ordered by feedback strength across the FLAMINGO variants. The mediation proposition (Eq. 46–47) says the harmonic-space $C(k)$ is exactly 1 whenever the galaxies' correlation with the gas is carried by the matter field. The round-two reading was that mediation fails in the strong-feedback runs. The formalism (§8.4, Eq. 48) then showed that the *filtered* $C_{\mathcal F}$ departs from 1 even under exact mediation, through a window term that predicts the same feedback ordering. Two readings were live (decision D-19). Round 3A separates them.

The separation is an exact identity. With $\eta = P_{bm}/P_{mm}$, $\beta = P_{gm}/P_{mm}$, and the galaxy-gas amplitude that exact mediation would predict,

$$
Y^{\rm med}_{gb}(R) = \int d\mu_R\,\frac{P_{bm}P_{gm}}{P_{mm}},
$$

the filtered factor splits as

$$
C_{\mathcal F} = \underbrace{\frac{Y_{bm}Y_{gm}}{Y_{mm}Y^{\rm med}_{gb}}}_{W_{\mathcal F}\,=\,\langle\eta\rangle\langle\beta\rangle/\langle\eta\beta\rangle}\;\times\;\underbrace{\frac{Y^{\rm med}_{gb}}{Y_{gb}}}_{M_{\mathcal F}} .
$$

$W_{\mathcal F}$ is the window term of Eq. (48). $M_{\mathcal F}$ is identically 1 when $C(k) = 1$ at every $k$, so its departure from 1 is mediation failure seen through the filter. Round 3A measures both, for every filter, run and redshift, together with $C(k)$ itself.

The round also takes the cheap items that the formalism and the review had queued (O-03, O-04, O-05, O-06, O-09, O-10), the difference-of-Gaussians (DoG) test of §4.5 (O-02), and, since DMO references came on disk during the unbound-gas work, the back-reaction (O-14).

---

## 2. Easy-to-follow summary of each step

1. **Wrote the predictions file before any run** (Stage 0).
2. **Ran the diagnostics that need no new computation** on the login node, from the committed round-two files (Stage 1).
3. **Built the machinery for exact Fourier-mode sums**, after a unit test showed the binned approximation too coarse on small grids. Covered it with unit tests (56 now) and smoke-tested it on TNG300-1 in a debug job.
4. **Ran $C(k)$, the split and the DoG sweep** for all four runs at both redshifts in three debug jobs, about 15 minutes each. Every regression check against round two passed.
5. **Analysed $C(k)$ and the split** (Stage 2) and **the DoG** (Stage 3) on the login node.
6. **Measured the back-reaction and the chain at $z\approx0.5$** from the unbound-gas 3D products (Stage 4).
7. **Code review.** No Critical findings. Three warnings fixed before any commit: atomic writes, regression guards in every consumer, and an independent cross-check of the mediation factor. A second pass on those fixes found no Critical either; its two warnings (an unflagged half of the new cross-check, and a summary line that could crash on an empty overlap) were fixed, and the pre-registered closure was added to the Stage 2 report for all eight run-snapshots.
8. **Updated the docs**: `decisions.md` (D-05, D-10, D-14, D-19, new D-21), `open-items.md`, `README.md`, `filter_specification.md`, and the one-line DMO statements in the formalism and the synthesis (user decision). Then this record.

---

## 3. Stage-by-stage record

### Stage 0 — Predictions

**Done.** `../predictions/2026-09-27_round3a.md`, written before any Round 3A number was computed: nine predictions (P1–P9), each with a pass/fail threshold, plus three acceptance checks that had to pass before anything was interpreted. It lists separately the results already seen during planning (the electron/baryon split by axis, a first look at O-04, the 3D back-reaction, the committed r-profile $C$), so those are not scored as predictions.

### Stage 1 — Diagnostics from the committed files

**Done.** `scripts/cross_corr/round3a_diagnostics.py`, on the login node, reads only `data/cross_corr_C/calibration_*.npz` and `task9_spectra_*.npz`. Numbers in `data/cross_corr_C/round3a/stage1_diagnostics.txt`.

**O-10, the collapse test.** If $C_{\mathcal F}\approx C(k_{50})$, the eight filters' curves would fall on one curve when plotted against $k_{50}(R;\mathcal F)$. They do not: at matched $k_{50}$ each filter differs from $\Delta\Sigma$ by 0.04–0.21 (every run, both redshifts), against the 0.03 threshold of P4. On matched aperture ranges, round two's comparison is reproduced: $Y(5')$ has $1.55\times$ $\Delta\Sigma$'s mean $|C-1|$ on its six bins at $z\approx0.5$ (0.122 against 0.079), as the synthesis stated; at $z\approx0.26$ every filter has a smaller $|C-1|$ than $\Delta\Sigma$ on matched bins (0.48–0.93 times).

**Prediction 3** of the v0.2 addendum holds at both redshifts. At matched aperture over $1'$–$6'$, $\Sigma$ and $\Delta\Sigma$ have mean $|C-1|$ of 0.051 and 0.052 ($z\approx0.5$) and 0.057 and 0.100 ($z\approx0.26$). At matched $k_{50}$ they have 0.065 and 0.010 ($z\approx0.5$), and 0.051 and 0.034 ($z\approx0.26$): $\Sigma$'s advantage disappears, because at $1'$ it probes $k\approx1\,h/$Mpc, where $\Delta\Sigma$ at large aperture is already at $C\approx1$.

**O-03, $C$ against $x$.** A straight line $C = a + b\,x_{\mathcal F}$ through the three FLAMINGO variants, at each aperture, predicts TNG300-1 too low by $+0.023$ to $+0.071$ at every aperture over $1'$–$6'$ at $z\approx0.5$, always more than half the cross-feedback spread. At $z\approx0.26$ it fails at $1'$–$2.25'$ and passes from $2.875'$. No one-parameter relation transfers across codes, so the feedback prior of D-14 stands.

**O-04, the electron-to-baryon correction.** The matter-crossed ratio the chain needs, $Y_{bm}/Y_{em}$ ($\Delta\Sigma$), with per-realization jackknife errors:

| $R$ | TNG300-1 | FLA fid | FLA Jet | FLA fgas$-8\sigma$ |
|---|---|---|---|---|
| $1'$, $z\approx0.5$ | 1.162 ± 0.002 | 1.492 ± 0.001 | 1.566 ± 0.001 | 1.854 ± 0.002 |
| $2.25'$ | 1.066 | 1.184 | 1.222 | 1.322 |
| $6'$ | 1.031 | 1.072 | 1.083 | 1.100 |
| $1'$, $z\approx0.26$ | 1.354 ± 0.005 | 1.932 ± 0.003 | 2.079 ± 0.004 | 2.635 ± 0.005 |

The galaxy-crossed $Y_{gb}/Y_{ge}$ of round one (1.21–2.27 at $1'$, $z\approx0.5$) is larger at small aperture, as O-04 expected, and the two converge by $3.5'$. The Convention T version $Y_{bt}/Y_{et}$ is within 1–2 per cent of $Y_{bm}/Y_{em}$.

**O-09, the $\Delta\Sigma$ log-slope.** The global slope of $Y^{(\Delta\Sigma)}_{gm}$ over $1'$–$9.75'$ is $-1.27$ to $-1.32$ at $z\approx0.5$ and $-1.17$ to $-1.19$ at $z\approx0.26$, not $-2$. $\Upsilon(1')$ therefore keeps 50 per cent of the $\Delta\Sigma$ amplitude at $2.25'$, 73–74 per cent at $6'$ and 80–83 per cent at $9.75'$; it is not a near-cancellation. By O-09's rule: keep $\Upsilon$ as the cross-check, with that caveat.

**A correction to the round-two record.** `tasks_7_to_10_record.md` (Stage 6 and prediction 4) explained $\Upsilon$'s $C$ plateau at $9.75'$ by $\Delta\Sigma(1')$ exceeding $\Delta\Sigma(9.75')$ "by roughly two orders of magnitude", making the reference term order-unity. The measured ratio is 17–19 for $Y_{gm}$, 7–12 for $Y_{gb}$, about 6 for $Y_{mm}$ and 3–5 for $Y_{bm}$, so $\Upsilon$ subtracts only 3–20 per cent from each amplitude at $9.75'$. The plateau comes from those fractions being *different* for the four amplitudes, because their log-slopes differ ($-0.5$ to $-1.3$). Multiplying round two's $C^{(\Delta\Sigma)}(9.75')$ by $\prod(1-f)$ over the four amplitudes reproduces the measured $C^{(\Upsilon)}(9.75')$ to four decimals in every run. That agreement is an algebraic identity, so it confirms only the arithmetic; the physical content is in the slopes.

**The electron and baryon targets, split by axis (input to D-05).** The round-two statement that the electron target carries twice the scatter is a pooled four-run number. Split into its axes ($\Delta\Sigma$, largest $|\Delta C|/C$ over $1'$–$6'$, with jackknife significance; the FLAMINGO differences use paired realizations):

| | cross-code (TNG300-1 − fid) | cross-feedback (fgas$-8\sigma$ − fid) |
|---|---|---|
| baryons, $z\approx0.5$ | 0.013 (2.4σ) | 0.089 (61σ) |
| electrons, $z\approx0.5$ | 0.032 (5.7σ) | 0.178 (97σ) |
| baryons, $z\approx0.26$ | 0.093 at $1'$ (12σ) | 0.073 (38σ) |
| electrons, $z\approx0.26$ | 0.037 (7.9σ) | 0.125 (67σ) |

The electron target is twice as sensitive to feedback within one code. Across codes it is worse than the baryon target at $z\approx0.5$ and better at $z\approx0.26$, where the baryon gap sits almost entirely in the $1'$ bin.

**P8, shared initial conditions.** Pre-registered on $C$: the median correlation of leave-one-out realizations between FLAMINGO fiducial and its variants is 0.70–0.79 ($z\approx0.5$) and 0.84–0.90 ($z\approx0.26$), below the 0.9 threshold; TNG300-1 against fiducial is $-0.02$ and $-0.34$. As scored, P8 fails. A post-hoc check on an amplitude dominated by cosmic variance settles the question it was meant to answer: the realizations of $Y^{(\Delta\Sigma)}_{mm}$ correlate at 1.000 between the variants and at $-0.2$ across codes. The variants share initial conditions; $C$'s realizations decorrelate because the shared cosmic variance largely cancels in the four-amplitude ratio, leaving feedback-specific fluctuations. Cross-feedback differences can therefore use paired jackknife errors, as the table above does.

### Stage 2 — $C(k)$ and the exact split (O-01, O-05, O-06)

**Done.** `scripts/cross_corr/make_ck_spectra.py` on a compute node, then `round3a_ck_analysis.py` on the login node. Numbers in `data/cross_corr_C/round3a/stage2_ck_analysis.txt`.

For each run and snapshot, the compute script rebuilds the round-two CDM, baryon and electron overdensity maps and the fiducial SHAM galaxy map, and adds three more galaxy samples:
- `alt`, the other number density at the same snapshot (O-06);
- `lo` and `hi`, the fiducial sample split at its median parent FoF mass (O-05).

It measures every auto and cross spectrum in linear $|k|$ bins of $0.02\,$arcmin$^{-1}$ over the full Fourier plane. For 58 kernels (pixelized $\Sigma$ and $\Delta\Sigma$ at the 19 calibration apertures, and the 20 DoG kernels of Stage 3) it computes the **exact** filtered amplitudes as sums over Fourier modes (Parseval), and the mediated amplitude $Y^{\rm med}_{gX}$ for $X \in \{b, e\}$ in both matter conventions. In $Y^{\rm med}_{gX}$ only the smooth $\eta(k) = P_{Xm}/P_{mm}$ is binned. $\Upsilon$ and the $Y$ transform follow as the round-two linear combinations of these amplitudes.

**A design change made during testing.** The first version formed every amplitude from binned spectra and binned kernel weights. On the $256^2$ test grid that approximation is only good to 1–2 per cent where the amplitude is small: within a bin, the slight anisotropy of the pixelized kernel correlates with the mode-to-mode scatter of the power, and finer bins do not help. The split is therefore computed from exact per-mode sums. The binned route is kept as a cross-check. On the production grids, over all eight run-snapshots, it reproduces the committed $C_{\mathcal F}$ to $1.4\times10^{-3}$ for $\Sigma$ and $\Delta\Sigma$ and to $6\times10^{-3}$ for the derived filters, and the exact $M_{\mathcal F}$ to $9.2\times10^{-4}$ for $\Sigma$, $\Delta\Sigma$ and the DoG ($5.1\times10^{-3}$ for the derived filters). Both checks are printed, with pass/fail flags, at the top of the Stage 2 report.

**Acceptance.**
- The exact amplitudes of the fiducial sample reproduce the committed round-two `calibration_*.npz`. Every one of the ten field pairs of the $\Sigma$ and $\Delta\Sigma$ filters agrees at every aperture, to a worst relative difference of $1.5\times10^{-12}$ across all eight run-snapshots.
- The SHAM samples are identical (e.g. 157,910 galaxies for FLAMINGO at $5\times10^{-4}$).
- The rebuilt $C_{\mathcal F}$ matches the committed values to $5\times10^{-12}$ for all eight round-two filters.

The fields, the galaxy catalogue and the kernels are therefore exactly the round-two ones.

**$C(k)$ itself** (fiducial sample, baryons, Convention C; logarithmic bins):

| run | $k=1$ | 2 | 3.5 | 5 | $7\,h/$Mpc |
|---|---|---|---|---|---|
| TNG300-1, $z\approx0.5$ | 1.011 | 1.040 | 1.068 | 1.090 | 1.119 |
| FLA fid | 1.009 | 1.035 | 1.048 | 1.049 | 1.050 |
| FLA Jet | 1.049 | 1.087 | 1.082 | 1.074 | 1.070 |
| FLA fgas$-8\sigma$ | 1.047 | 1.111 | 1.109 | 1.089 | 1.078 |
| TNG300-1, $z\approx0.26$ | 1.013 | 1.045 | 1.099 | 1.114 | 1.154 |
| FLA fid ($z=0.30$) | 1.018 | 1.055 | 1.092 | 1.090 | 1.059 |
| FLA Jet | 1.065 | 1.100 | 1.100 | 1.069 | 1.028 |
| FLA fgas$-8\sigma$ | 1.068 | 1.126 | 1.124 | 1.080 | 1.035 |

$C(k)\to1$ on large scales and rises above 1 from $k\approx0.5\,h/$Mpc. Mediation fails in harmonic space, by 5–12 per cent over $\Delta\Sigma(1')$'s response range ($k_{05}$–$k_{95}$ = 1.8–7.4 $h/$Mpc at $z\approx0.5$). In the strong-feedback runs it already fails by 5 per cent at $k=1\,h/$Mpc. $C>1$ means $P_{gb} < \eta P_{gm}$: the gas around the stacked galaxies is more depleted than the matter field predicts. Convention T gives the same $C(k)$ to within one per cent over $1\le k\le7.5\,h/$Mpc (0.3–1.0 per cent, largest in the strong-feedback runs).

**The split at $1'$** ($\Delta\Sigma$, baryons, Convention C):

| run | $C_{\mathcal F}$ | $W_{\mathcal F}$ | $M_{\mathcal F}$ | $f_W = \ln W/\ln C$ |
|---|---|---|---|---|
| TNG300-1, $z\approx0.5$ | 1.114 | 1.028 | 1.084 | 0.25 |
| FLA fid | 1.105 | 1.051 | 1.051 | 0.50 |
| FLA Jet | 1.138 | 1.053 | 1.081 | 0.40 |
| FLA fgas$-8\sigma$ | 1.166 | 1.056 | 1.104 | 0.36 |
| TNG300-1, $z\approx0.26$ | 1.249 | 1.077 | 1.160 | 0.33 |
| FLA fid ($z=0.30$) | 1.138 | 1.067 | 1.067 | 0.50 |
| FLA Jet | 1.109 | 1.066 | 1.041 | 0.61 |
| FLA fgas$-8\sigma$ | 1.105 | 1.050 | 1.053 | 0.49 |

Two things settle the question round two left open (D-19).

- **Both effects are real and comparable at $1'$.** The window term contributes $+3$ to $+8$ per cent, the mediation term $+4$ to $+16$ per cent.
- **The feedback ordering is mediation, not window.** Across the FLAMINGO variants at $z\approx0.5$, $W$ is 1.051, 1.053, 1.056, a spread of 0.005. $M$ is 1.051, 1.081, 1.104, so the monotone ordering 1.105, 1.138, 1.166 of $C_{\mathcal F}$ is carried entirely by $M$. Round two's reading, that mediation fails more in the strong-feedback runs, is right about the ordering. The formalism's window term is right that a feedback-independent part of $C_{\mathcal F}-1$ at $1'$ has nothing to do with mediation.

At larger apertures the window term turns negative (0.98–0.99 by $3.5'$–$6'$) while $M$ stays above 1. In the strong-feedback runs, $C_{\mathcal F}\to1$ at $6'$ is therefore partly a cancellation ($M = 1.044$, $W = 0.981$–$0.987$ at $z\approx0.5$), not a return to mediation.

**The window term explains the sign reversals between filters.** At the smallest usable aperture, $\Sigma$, $\Upsilon$ and the $Y$ transform all have $C_{\mathcal F}<1$ at $z\approx0.5$ (0.85–0.94). In every case that comes from $W$ (0.83–0.92), while their $M$ stays at 1.00–1.04. Formalism §8.4 said the reversal was possible only because the measure is signed; this confirms it and gives its size.

**Electrons.** The same split for the ionized gas at $1'$, $z\approx0.5$: $M_e$ = 1.120 (TNG300-1), 1.103, 1.166, 1.279 (FLAMINGO fid, Jet, fgas$-8\sigma$); $W_e$ = 1.03–1.12. Mediation fails more for the electrons than for all baryons, and more steeply with feedback. That is the harmonic-space origin of the electron target's doubled cross-feedback scatter (Stage 1).

**P4, second part.** At matched $k_{50}$ across $\Sigma$, $\Delta\Sigma$ and the DoG ($q=2$), $M_{\mathcal F}$ differs by 0.011–0.026 ($z\approx0.5$) and 0.020–0.045 ($z\approx0.26$), against 0.06–0.10 for $W_{\mathcal F}$. The mediation factor is much closer to a filter-independent function of $k_{50}$ than the window factor, but it misses the pre-registered 0.01. That is expected with hindsight: $M_{\mathcal F}$ is a weighted mean of a $C(k)$ that varies across each window, and the windows differ in width.

**O-05, sample invariance, done in harmonic space.** Mediation requires the same $C(k)$ for every galaxy sample. As posed in the review (and in the synthesis, §III.H item 3), the aperture-space version tests $Y_{gb}/Y_{gm}$ for two samples. Even under exact mediation that ratio is $\langle\eta\beta\rangle/\langle\beta\rangle$, which depends on each sample's $\beta(k)$, so it cannot separate the two readings; it is done here on the spectra. The fiducial sample split at its median parent mass ($z\approx0.5$; TNG300-1 halves $2.3\times10^{12}$–$1.65\times10^{13}$ and $1.65\times10^{13}$–$4.1\times10^{14}\,M_\odot/h$):

| run | sample | $k=1$ | 2 | 3.5 | 5 | $7\,h/$Mpc |
|---|---|---|---|---|---|---|
| TNG300-1 | low mass | 1.034 | 1.138 | 1.325 | 1.323 | 1.325 |
| | high mass | 1.006 | 1.015 | 0.999 | 1.009 | 1.041 |
| FLA fid | low mass | 1.027 | 1.178 | 1.272 | 1.202 | 1.142 |
| | high mass | 1.004 | 1.002 | 0.985 | 0.996 | 1.013 |
| FLA Jet | low mass | 1.108 | 1.202 | 1.140 | 1.051 | 1.013 |
| | high mass | 1.034 | 1.061 | 1.064 | 1.083 | 1.097 |
| FLA fgas$-8\sigma$ | low mass | 1.119 | 1.357 | 1.216 | 1.101 | 1.052 |
| | high mass | 1.029 | 1.060 | 1.076 | 1.084 | 1.090 |

Sample invariance fails by far more than the per-cent level: $C(k)$ reaches 1.2–1.36 for the galaxies in lower-mass hosts. Hosts above $\sim\!10^{13}\,M_\odot/h$ are close to mediated in TNG300-1 and FLAMINGO fid, and 6–10 per cent off in the strong-feedback runs. Contrary to the high-mass expectation stated in O-05, and as the predictions file expected, the deviation is carried by the lower-mass hosts. The natural reading is that gas depletion depends on halo mass and the galaxies select halos: the residual gas field correlates with the galaxies through the host mass, without any appeal to the galaxies' own AGN. By comparison, swapping the number density moves $C(k)$ by at most 2–5 per cent over $1\le k\le7.5\,h/$Mpc.

**O-06, what the $z\approx0.26$ cross-code gap follows.** The cross-code difference $(C_{\rm TNG}-C_{\rm fid})/C$ of $\Delta\Sigma$, baryons, at $1'$:

| snapshot | LRG-like ($5\times10^{-4}$) | BGS-like ($10^{-3}$) |
|---|---|---|
| $z\approx0.5$ | $+0.008$ | $+0.004$ |
| $z\approx0.26/0.30$ | $+0.071$ | $+0.093$ |

The gap follows the snapshot, not the number density, so SHAM matching is not its cause. It is also a baryon-field effect: the electron $C$ at $z\approx0.26$ differs across codes by only 0.6–1.6 per cent at $1'$. Its $M$ carries it (TNG300-1 $M = 1.160$ against FLAMINGO fid 1.067 at $1'$; $W$ 1.077 against 1.067). Two candidates remain, and this round cannot separate them: the FLAMINGO $z=0.30$ against TNG300-1 $z=0.26$ snapshot mismatch (O-22), and a genuine difference in the codes' stellar and neutral components at low redshift (the gap is absent for electrons).

**The doubly filtered coefficients (formalism Eq. 40).** With the positive measure $\hat W^2$:
- **Continuous fields:** $\tilde r_{bm}$ = 0.89–0.99, bounded by one as Cauchy-Schwarz requires.
- **Galaxy coefficients:** $\tilde r_{gm}$ = 2.2–3.3 and $\tilde r_{gb}$ = 1.2–2.7 at the smallest scale.

**This does not contradict the bound.** It is a property of the galaxy auto after the Poisson shot noise is subtracted. The Fourier coefficient behaves the same way:
- with the shot noise included, $P_{gm}/\sqrt{P_{gg}P_{mm}}$ is 0.34–0.93 at every $k$;
- subtracting it gives 1.02 at $0.1\,h/$Mpc, rising to 3 at $7\,h/$Mpc.

The reason is that the galaxies' clustering power beyond Poisson is only 5–20 per cent of the Poisson level above $2\,h/$Mpc. That is the sub-Poisson stochasticity of a mostly-central sample of massive halos (halo exclusion). Formalism §7.3's statement that $|\tilde r|\le1$ strictly therefore holds for continuous fields only. For $r_{gb}$ and $r_{gm}$, the excess over one comes from the galaxy auto (the formalism's "fourth mechanism"), not from the sign-changing measure. This matters for Route A, which divides by $\sqrt{Y_{gg}}$. It does not matter for Route B, where $Y_{gg}$ cancels.

### Stage 3 — The difference-of-Gaussians kernel (O-02)

**Done.** The DoG test is a standalone script, `scripts/cross_corr/make_dog_calibration.py`, on a compute node; it is not added to `src/`. It is then compared with $\Delta\Sigma$ in `round3a_dog_analysis.py`. Numbers are in `data/cross_corr_C/round3a/stage3_dog_analysis.txt`.

**The kernel.** Formalism Eq. (22), defined in Fourier space, $\hat W = e^{-k^2\sigma_1^2/2}-e^{-k^2\sigma_2^2/2}$ in the pipeline normalization, so that $\hat W(0)=0$ and $\hat W>0$ for $k>0$ hold exactly. Two fixed-ratio families, $\sigma_2 = q\sigma_1$ with $q = 2$ and $1.5$, are run over ten log-spaced widths $\sigma_1 = 0.5'$–$5'$. They are compared with $\Delta\Sigma$ at matched $k_{50}$, measured against each run's CDM spectrum, so no rule matching $\sigma_1$ to an aperture is needed.

**The sweep.** The amplitudes, with 16-block jackknife errors, go through a generic re-implementation of `compute_Y_matrix`. A unit test shows it reproduces `compute_Y_matrix` bit for bit when handed the pixelized $\Delta\Sigma$ kernels. The payload goes through round two's `flatten_for_npz`, so the keys mean what they mean in the round-two files, with no $S$ written for electrons. The map-level sweep and Stage 2's exact Fourier sums agree to $\le1.5\times10^{-11}$ on every DoG amplitude.

**Result: the positive window does not shrink $C-1$.** Median of $|C^{\rm DoG}-1|/|C^{(\Delta\Sigma)}-1|$ over the $k_{50}$ range the two share:

| | TNG300-1 | FLA fid | FLA Jet | FLA fgas$-8\sigma$ |
|---|---|---|---|---|
| $q=2$, $z\approx0.5$ | 4.2 | 3.8 | 1.8 | 1.8 |
| $q=1.5$, $z\approx0.5$ | 2.8 | 3.0 | 1.8 | 1.7 |
| $q=2$, $z\approx0.26$ | 2.2 | 1.6 | 1.4 | 1.1 |
| $q=1.5$, $z\approx0.26$ | 2.3 | 1.6 | 1.3 | 1.1 |

No width shrinks $|C-1|$ by the factor of two that formalism §4.5 required in all four runs. At $z\approx0.5$ no point shrinks it at all. At $z\approx0.26$ the DoG is closer to 1 than $\Delta\Sigma$ only at its smallest widths in FLAMINGO ($k_{50}\approx7\,h/$Mpc: 1.04, 1.02, 0.98 against 1.14, 1.11, 1.10), where $C(k)$ itself turns back towards 1. The FLAMINGO feedback ordering (fiducial lowest) persists at every width at $z\approx0.5$ except one; at $z\approx0.26$ it holds for $\sigma_1\ge1.1'$–$1.4'$ and reverses at the smallest widths.

**Why.** From Stage 2, both factors of the DoG's $C_{\mathcal F}$ are positive-window averages:
- $W^{\rm DoG}$ = 1.00–1.09 at every width, bar the eight exceptions below;
- $M^{\rm DoG}$ = 1.00–1.13.

Nothing cancels. $\Delta\Sigma$'s signed window makes its $W$ fall below 1 at large aperture (0.98–0.99), partly cancelling $M>1$. So $\Delta\Sigma$'s apparent convergence $C\to1$ over $4'$–$6'$ is partly that cancellation, not a return to mediation. The DoG shows the uncancelled size.

**The sign of the DoG window term.** Under a positive window, $W = \langle\eta\rangle\langle\beta\rangle/\langle\eta\beta\rangle$ exceeds 1 whenever $\eta$ falls and $\beta$ rises across the window. It does so in 632 of 640 cases (run, $q$, $\sigma_1$, gas field, convention). The eight exceptions (0.956–0.995) are all FLAMINGO fgas$-8\sigma$ and Jet at $z\approx0.26$, at the smallest widths, for the baryons only. There $\eta(k)$ has a minimum near $7.5\,h/$Mpc and rises again (0.37 to 0.52 by $19\,h/$Mpc in Jet, $z=0.30$) as the stars take over the baryon field. The premise of the prediction, a monotone $\eta$, fails at the highest $k$.

**Consequence for the filter decision (D-10).** A positive window buys the properties formalism §4.5 lists: Cauchy-Schwarz for continuous fields, no pixelization floor, a beam that composes analytically. It does not buy a smaller calibration factor, because the dominant part of $C-1$ is mediation failure, which every kernel sees. $\Delta\Sigma$ remains the fiducial filter.

### Stage 4 — The back-reaction and the suppression chain at $z\approx0.5$

**Done.** `scripts/cross_corr/round3a_backreaction.py`, on the login node. It reads the unbound-gas pipeline's 3D products (a dependency approved on 2026-09-27): auto and cross spectra of the DM, ionized-gas, neutral-gas, stellar and black-hole overdensities on one TSC grid (TSC-deconvolved, neutrinos excluded, DM $=$ total $-$ gas $-$ stars $-$ BH as in the 2D maps), and the matched DMO run's spectrum. Grids reach $k_{\rm Nyq} = 9.2\,h/$Mpc (FLAMINGO, $2000^3$) and $15.3\,h/$Mpc (TNG300-1, $1000^3$). Numbers in `data/cross_corr_C/round3a/stage4_backreaction.txt`.

**Acceptance.** Rebuilding the total-matter spectrum from the components, $P_{tt} = f_m^2P_{mm}+2f_mf_bP_{bm}+f_b^2P_{bb}$, reproduces the stored total to $1.3\times10^{-7}$ (TNG300-1) and $\le4\times10^{-8}$ (FLAMINGO) for $k\le5\,h/$Mpc, so the component bookkeeping is right. P9: the 3D $x(k)$ agrees with the 2D maps' $x(k)$ (from Stage 2's spectra) to 0.002–0.005 over $0.3\le k\le5\,h/$Mpc, inside the 0.02 threshold. Two independent pipelines, different grids and mass assignments, give the same $x$.

**The back-reaction** $B(k) = P^{\rm hydro}_{mm}/P^{\rm DMO}$ (formalism Eq. 62, ledger row 13):

| run | $k=0.5$ | 1 | 2 | 3 | 5 | $7\,h/$Mpc |
|---|---|---|---|---|---|---|
| TNG300-1 | 1.001 | 1.005 | 1.011 | 1.010 | 0.995 | 0.977 |
| FLA fid | 1.005 | 1.010 | 1.017 | 1.018 | 1.013 | 1.012 |
| FLA Jet | 1.003 | 1.002 | 1.000 | 0.994 | 0.981 | 0.975 |
| FLA fgas$-8\sigma$ | 1.004 | 1.005 | 1.002 | 0.995 | 0.982 | 0.979 |

Within $\pm2.5$ per cent everywhere, as booked, but with a run-dependent sign: the hydro CDM clusters more than the DMO matter in the fiducial run at all $k$, less in the strong-feedback runs above $3\,h/$Mpc.

**The chain against the true suppression.** The estimator targets $S_{\rm int} = P_{tt}/P_{mm}$ within the hydro run, which $(f_m+f_bx)^2$ matches to 0.002. The true suppression is $S = P^{\rm hydro}_{tt}/P^{\rm DMO}$, and $S_{\rm int}/S - 1 = 1/B - 1$ is $-1.8$ to $+2.5$ per cent over $0.5\le k\le7\,h/$Mpc. At $k=5\,h/$Mpc: $-1.3$ per cent (fiducial), $+0.5$ (TNG300-1), $+1.9$ (Jet), $+1.8$ (fgas$-8\sigma$). This error is not suppressed by $f_b$: it enters $S$ one-to-one, so it is as large as the whole calibration budget (6 per cent on $C$ gives 1.9 per cent on $S$), and 5–8 per cent of the suppression depth $1-S$ at $k\approx5\,h/$Mpc. It is now measured rather than assumed at $z\approx0.5$ (decision D-21). At $z\approx0.26$ the DMO spectra exist but the hydro component spectra at snapshots 71 and 80 do not, so it remains booked there.

**The component split of Eq. (67).** $x_b = \sum_i w_ix_i$ over ionized gas, neutral gas, stars and black holes, exactly. At $k = 5\,h/$Mpc:

| run | $w_{\rm ion}$ | $w_\star$ | $x_{\rm ion}$ | $x_{\rm neutral}$ | $x_\star$ | $x_b$ | $x_b/x_{\rm ion}$ |
|---|---|---|---|---|---|---|---|
| TNG300-1 | 0.955 | 0.028 | 0.544 | 0.749 | 4.44 | 0.658 | 1.21 |
| FLA fid | 0.915 | 0.055 | 0.301 | 1.082 | 3.63 | 0.513 | 1.70 |
| FLA Jet | 0.917 | 0.058 | 0.225 | 0.904 | 2.99 | 0.408 | 1.81 |
| FLA fgas$-8\sigma$ | 0.909 | 0.061 | 0.177 | 0.978 | 3.47 | 0.407 | 2.30 |

Stars carry 3–6 per cent of the baryon mass, but their cross-spectrum with the CDM is 3.0–4.4 times the CDM's own at these scales, so in FLAMINGO they supply 39–52 per cent of $x_b$ at $k = 5\,h/$Mpc (fiducial: $0.055\times3.63 = 0.20$ of 0.51), and 19 per cent in TNG300-1. Of the step itself, $x_b - x_{\rm ion} = \sum_{i\ne{\rm ion}} w_i(x_i - x_{\rm ion})$, the stars carry 86–96 per cent, the neutral gas 3–10 and the black holes 1–5. The electron-to-baryon step in harmonic space, $x_b/x_{\rm ion} = 1.2$–2.3 at $k=5$, matches the aperture-space O-04 numbers in size and ordering. It is dominated by the stellar term, which the stellar mass function constrains externally, as O-07 argued.

---

## 4. The predictions, scored

Against `../predictions/2026-09-27_round3a.md`, whose thresholds were fixed before the runs.

| # | prediction | outcome | verdict |
|---|---|---|---|
| — | acceptance: regression $<10^{-9}$; closure $<0.5$ per cent; 3D bookkeeping $<10^{-3}$ | $1.5\times10^{-12}$; closure $5\times10^{-12}$ on the exact route (all eight filters), and on the binned route $\le1.4\times10^{-3}$ for $\Sigma$ and $\Delta\Sigma$ but up to $6.0\times10^{-3}$ for $\Upsilon$ (FLAMINGO fgas$-8\sigma$, $z\approx0.26$); $1.3\times10^{-7}$ | **passed** on the exact route, which the split uses; the binned route misses the 0.5 per cent closure for $\Upsilon$ in one run-snapshot |
| P1 | at $1'$ ($\Delta\Sigma$, $z\approx0.5$) window if $f_W\ge0.8$, mediation if $f_W\le0.2$; prior: mixed, $M-1$ in $[0.03, 0.08]$ ordered by feedback | $f_W$ = 0.25–0.50; $M-1$ = 0.051, 0.081, 0.104 (fid, Jet, fgas$-8\sigma$), 0.084 (TNG300-1); $C(k)-1$ up to 0.05–0.12, ordered | **mixed, as the prior said**; the prior's $M$ range held for fid and (just) Jet, not for fgas$-8\sigma$ or TNG300-1 |
| P2 | the low-mass half deviates most: $C_{\rm lo}-C_{\rm hi}\ge0.02$ at 3–7 $h/$Mpc (FLA fid) | $+0.13$ to $+0.29$ | **held**; O-05's high-mass expectation is not borne out |
| P3 | the $z\approx0.26$ gap stays with the snapshot ($\ge0.05$ at $z\approx0.26$ with the LRG-like density, $\le0.02$ at $z\approx0.5$ with the BGS-like one); electrons within 0.04 in all four | $+0.071$ and $+0.004$; electrons 0.024–0.059 | **held**, except the electron sub-claim fails once (0.059, $z\approx0.5$, BGS-like) |
| P4 | no collapse at matched $k_{50}$ (spread $>0.03$); $M$ collapses within 0.01 across $\Sigma$, $\Delta\Sigma$, DoG while $W$ does not | spread 0.04–0.21; $M$ 0.011–0.045, $W$ 0.06–0.10 | first part **held**; second part **failed** ($M$ is far closer to filter-independent than $W$, but not to 0.01) |
| — | addendum prediction 3: $\Sigma$'s advantage shrinks at matched $k_{50}$ | yes, at both redshifts | **held** |
| P5 | the FLAMINGO $C(x)$ line predicts TNG300-1 within half the cross-feedback spread; prior: fails at $1'$, passes by $3'$ | fails at every aperture at $z\approx0.5$; at $z\approx0.26$ fails to $2.25'$, passes from $2.875'$ | **failed**; the prior was right at $z\approx0.26$ only |
| P6 | $\Delta\Sigma$ log-slope in $[-2.2,-1.8]$ | $-1.17$ to $-1.32$ | **falsified**; $\Upsilon$ is not a near-cancellation |
| P7 | the DoG does not remove $C-1$: within a factor 1.5 of $\Delta\Sigma$ at matched $k_{50}$, ordering persists, $W^{\rm DoG}>1$ everywhere | no shrinkage anywhere by §4.5's rule; the DoG is 1.1–4.2 times *worse* (median); ordering persists except at the smallest widths at $z\approx0.26$; $W^{\rm DoG}\le1$ in 8 of 640 cases | main claim **held**; "within 1.5" failed on the worse side; "$W>1$ everywhere" failed where $\eta$ turns up at high $k$ |
| P8 | FLAMINGO variants' $C$ realizations correlate at $\rho>0.9$; TNG300-1 $\lvert\rho\rvert<0.5$ | 0.70–0.90; $-0.02$, $-0.34$ | **failed as posed**; the premise (shared initial conditions) is confirmed post hoc by $Y_{mm}$, $\rho = 1.000$ |
| P9 | 3D and 2D $x(k)$ agree within 0.02 | 0.002–0.005 | **held** |

---

## 5. What this changes

**Decisions** (`../decisions.md`):
- **D-19** is resolved: both readings hold, for different parts of $C_{\mathcal F}-1$.
- **D-14** (the feedback prior on $C$) stands. O-03 failed, and the feedback dependence is genuine mediation failure, not a window artefact.
- **D-10**: $R_0 = 1'$ by user decision. $\Upsilon$ is kept (O-09), the DoG does not beat $\Delta\Sigma$ (O-02), and $\Delta\Sigma$ stays fiducial.
- **D-21** is new: the back-reaction is measured at $z\approx0.5$.
- **D-05** gains evidence, and the decision stays the user's.

**Open items** (`../open-items.md`): O-01, O-02, O-03, O-04, O-05, O-06, O-09 and O-10 are done; O-14 is done at $z\approx0.5$.

**For the calibration strategy**, three consequences.

1. **$C$ depends strongly on which halos host the sample.** Splitting one SHAM sample by parent mass moves $C(k)$ by up to 30 per cent. Matching the simulated and data samples on number density alone is therefore not enough in general. It happens to work at $z\approx0.5$ (1.3 per cent cross-code) and not at $z\approx0.26$ (7–9 per cent). A calibration aimed at data needs the host-mass distribution matched, for instance through the sample's own lensing amplitude.
2. **The two parts of $C_{\mathcal F}$ behave differently.** The window part is filter-dependent but nearly feedback-independent, and is fixed by $\eta(k)$ and $\beta(k)$. The physics uncertainty sits in the mediation part. This favours the harmonic-space forward model of formalism §9.6 (option 2), which models $C(k)$ and computes the window exactly, over an aperture-by-aperture $C_{\mathcal F}$ with a prior.
3. **The galaxy auto after shot-noise subtraction is small at these scales** (sub-Poisson). Every quantity that divides by $\sqrt{Y_{gg}}$ — Route A, $C_A$, $r_{gb}$, $r_{gm}$ — inherits that fragility, and Route B does not.

---

## 6. Issues found, and how

| issue | where | how found |
|---|---|---|
| Binned amplitudes only 1–2 per cent accurate on small grids | the first design of the split | a unit test against `compute_Y_matrix` on a $256^2$ grid; replaced by exact per-mode sums |
| `salloc ... bash runner` and `bash -n` denied by this session's permission settings | the runner | switched to a dual-use batch script (`runCPU_round3a.sh`); the interactive-only draft went to the trash |
| The Stage 3 summary picked DoG points outside $\Delta\Sigma$'s $k_{50}$ range, so NaNs | `round3a_dog_analysis.py` | reading the output; replaced by medians over the shared range |
| Code review: analysis outputs written non-atomically; two consumers did not refuse a failed regression; the mediated amplitudes had no independent check | the analysis scripts | the `code-reviewer` subagent; all three fixed, and the new binned cross-check agrees to $\le9.2\times10^{-4}$ |
| Second review pass: the derived filters' half of the binned cross-check was printed but never flagged; the DoG summary could crash if a run's $k_{50}$ grid missed $\Delta\Sigma$'s | `round3a_ck_analysis.py`, `round3a_dog_analysis.py` | the `code-reviewer` subagent, on the fixes; both fixed, a wiring test for the binned route added, and the rerun reports identical apart from the new flags and closure lines |
| Three imprecisions in this record's draft | this record | re-deriving the numbers: Convention T's $C(k)$ agrees to 1 per cent, not 0.5; the density swap moves $C(k)$ by up to 5 per cent, not 4; the stars carry 86–96 per cent of the electron-to-baryon step (39–52 per cent of $x_b$ in FLAMINGO), not "about 40 per cent" of it |
| The round-two explanation of $\Upsilon$'s plateau ("two orders of magnitude") | `tasks_7_to_10_record.md`, Stage 6 | Stage 1's amplitude ratios (17–19) |
| $\lvert\tilde r\rvert\le1$ holds for continuous fields only | formalism §7.3 | Stage 2's doubly filtered galaxy coefficients, confirmed on the Fourier coefficients |
| The pre-registered binned-route closure had been run on TNG300-1 at $z\approx0.5$ only | the acceptance check | rerun on all eight run-snapshots before committing: $\Upsilon$ reaches 0.6 per cent in FLAMINGO fgas$-8\sigma$ at $z\approx0.26$; the split uses the exact route, so no result changes |

---

## 7. Numbers at a glance

**Split at $1'$** ($\Delta\Sigma$, baryons, $z\approx0.5$): $W$ = 1.03–1.06, $M$ = 1.05–1.10; FLAMINGO $W$ spread 0.005, $M$ spread 0.053.

**$C(k)$** over 1.8–7.4 $h/$Mpc ($z\approx0.5$): 1.05–1.12. Low-mass hosts: up to 1.36. High-mass hosts: $\le1.10$.

**Cross-code gap at $1'$**: $z\approx0.5$: 0.4–0.8 per cent; $z\approx0.26$: 7–9 per cent, for either number density.

**DoG against $\Delta\Sigma$** at matched $k_{50}$: $|C-1|$ ratio 1.1–4.2 (median); no width shrinks it by half in all four runs.

**Back-reaction** at $z\approx0.5$: 0.975–1.018 over $0.5\le k\le7\,h/$Mpc, i.e. $-1.8$ to $+2.5$ per cent on $S$.

**$e\to b$**: $Y_{bm}/Y_{em}$ at $1'$ = 1.16–1.85 ($z\approx0.5$); $x_b/x_{\rm ion}$ at $k=5$ = 1.21–2.30, 86–96 per cent of the step from stars.

**$\Delta\Sigma$ log-slope**: $-1.3$ ($z\approx0.5$), $-1.2$ ($z\approx0.26$); $\Upsilon(1')$ keeps 50–83 per cent.

**Cost**: Stage 2 and Stage 3 together took 43 minutes of debug-QOS wall time over three one-node jobs, with peak memory 46–78 GiB. Stages 1 and 4 run in seconds on a login node.

---

## 8. What was skipped, and why

| skipped | reason | cost of the gap |
|---|---|---|
| Errors on $C(k)$ per $k$ | agreed for the first pass | $C(k)$ differences between runs carry no error bars; the FLAMINGO variants share initial conditions, so their differences are much less noisy than their values |
| The back-reaction at $z\approx0.26$ | the hydro component spectra at snapshots 71 and 80 do not exist | D-21 open there |
| A FLAMINGO fiducial rerun at snapshot 72 ($z=0.25$) | its 2D maps do not exist | O-22 unresolved: snapshot mismatch against the codes' stellar components |
| The point-mass-blind ring-Gaussian kernel (formalism §4.5) | the DoG outcome shows $C-1$ is mostly mediation, which no kernel removes | low |
| The D-05 decision | it is the user's | Gate B stays open on the field |
| Second projection, third code, bin covariance | unchanged from round two | unchanged |

---

## 9. Next steps

Ordered by what unblocks the most.

1. **Decide D-05** (electron or baryon target) with this round's evidence. The baryon target with an externally constrained stellar term is favoured, but the call is the user's.
2. **Test the host-mass dependence of the calibration directly.** Rebuild the calibration sample matched on host-halo mass rather than on number density, and ask how much of the $z\approx0.26$ cross-code gap and of the cross-feedback spread survives. If it largely disappears, the calibration should be conditioned on a measurable host-mass proxy.
3. **Move to a harmonic-space forward model** (formalism §9.6, option 2), with $C(k)$ as the calibrated object and the window computed exactly.
4. **Close D-21 at $z\approx0.26$** (hydro component spectra at snapshots 71 and 80) and **O-22** (FLAMINGO fiducial at snapshot 72).
5. **Regenerate the formalism and the synthesis** to fold this round in, when the user asks:
   - formalism: §4.5 (the DoG outcome), §7.3 (the bound), §8.4 (the window term now measured), and ledger rows 10, 11 and 13;
   - synthesis: its mediation reading.

---

## 10. File map and reproduction

**Library.** Nothing in `src/` changed except four documentation paths in docstrings and comments (user decision). The Round 3A functions live in `scripts/cross_corr/round3a_lib.py`, so `data/r_profiles/` and `data/cross_corr_C/calibration_*.npz` stay reproducible.

| path | purpose |
|---|---|
| `scripts/cross_corr/round3a_lib.py` | exact Fourier-mode amplitudes, the split, the DoG kernel, the 3D bookkeeping, output hygiene |
| `scripts/cross_corr/make_ck_spectra.py` | Stage 2 compute: spectra, exact and mediated amplitudes, regression check |
| `scripts/cross_corr/make_dog_calibration.py` | Stage 3 compute: the DoG sweep with jackknife errors |
| `scripts/cross_corr/round3a_diagnostics.py` | Stage 1 |
| `scripts/cross_corr/round3a_ck_analysis.py` | Stage 2 analysis |
| `scripts/cross_corr/round3a_dog_analysis.py` | Stage 3 analysis |
| `scripts/cross_corr/round3a_backreaction.py` | Stage 4 |
| `scripts/cross_corr/runCPU_round3a.sh` | batch runner (debug QOS by default; also runs under `salloc`) |
| `scripts/configs/cross_corr/round3a_z05.yaml`, `round3a_z026.yaml` | the round-two `stack` and `simulations` blocks verbatim, plus a `round3a` block |
| `tests/test_round3a.py` | 56 tests |
| `data/cross_corr_C/round3a/` | `ck_spectra_*` and `dog_calibration_*` (8 each), `stage{1..4}_*.{npz,txt}`; 9.3 MB |
| `figures/2026-09/09-28/round3a_s{1,2,3,4}_*.png` | figures from the final runs (gitignored, regenerable); `09-27/` holds identical first versions of s1, s2, s4 |

**Jobs** (debug QOS, one CPU node each, all `COMPLETED 0:0`):

| job | contents | wall time | peak memory (`seff`) |
|---|---|---|---|
| 58997433 | smoke test, TNG300-1 at $z\approx0.5$ | 2 min | 25 GiB |
| 58997723 | $z\approx0.5$, both stages | 13.6 min | 46 GiB |
| 58997725 | $z\approx0.26$, spectra | 15.7 min | 78 GiB |
| 58997727 | $z\approx0.26$, DoG | 13.6 min | 61 GiB |
| 58999586 | verification, TNG300-1 at $z\approx0.5$: `make_ck_spectra.py` as committed (a comment and the resolution guard were added after the runs above) reproduces `ck_spectra_TNG300-1_67_yz.npz` bit for bit, all 278 arrays | 2.3 min | 25 GiB |

**Commits** (user-approved 2026-09-28, not pushed): `d9fe05a` (the `src/` documentation paths); `c33c30a` (code, tests, configs and `data/cross_corr_C/round3a/`). This record and the other documentation changes are in the commit that follows `c33c30a`.

**Reproduce everything:**

```bash
cd scripts/
sbatch --export=ALL,CONFIGS=z05 cross_corr/runCPU_round3a.sh
sbatch --export=ALL,CONFIGS=z026,STAGES=ck cross_corr/runCPU_round3a.sh
sbatch --export=ALL,CONFIGS=z026,STAGES=dog cross_corr/runCPU_round3a.sh
python cross_corr/round3a_diagnostics.py      # Stage 1 (login node)
python cross_corr/round3a_ck_analysis.py      # Stage 2
python cross_corr/round3a_dog_analysis.py     # Stage 3
python cross_corr/round3a_backreaction.py     # Stage 4
cd ../tests/ && pytest test_round3a.py -q     # 56 tests
```
