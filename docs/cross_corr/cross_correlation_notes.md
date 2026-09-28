# Model-Independent Baryon-Matter Cross-Correlations from kSZ and Galaxy Clustering: Theory Note and Simulation Validation Plan

**R. Henry Liu** (with U. Seljak)
*Working document, v0.1*

---

## 1. Introduction and Motivation

Baryonic feedback is a leading systematic for weak lensing cosmology: energy injection from AGN and supernovae redistributes gas within and beyond haloes, suppressing the total matter power spectrum by up to tens of per cent at $k \gtrsim 0.5\,h\,\mathrm{Mpc}^{-1}$ (Chisari et al. 2018; van Daalen et al. 2020). The goal of the programme described here is to measure the quantity that controls this suppression at leading order, the baryon-matter cross-correlation $P_{bm}/P_{mm}$, in a form that is as close to model-independent as the data permit.

This note builds on two existing methodological threads:

1. **Singh et al. (2020, MNRAS 491, 51; arXiv:1811.06499).** Cosmological constraints from galaxy-galaxy lensing without explicit galaxy bias modelling. The key move is the substitution $\Upsilon_{gm} = \bar{\rho}_m\, r_{cc}\sqrt{\Upsilon_{mm}\Upsilon_{gg}}$: the measured galaxy clustering $\Upsilon_{gg}$ becomes part of the model, theory supplies $\Upsilon_{mm}$, and all residual non-linear stochasticity is compressed into a single cross-correlation coefficient $r_{cc}(r_p)$, parametrized with priors calibrated on mock catalogues. The bias never appears.

2. **Liu (2026, in prep.; the $f_{\rm gas}$ paper).** A common $\Delta\Sigma$ aperture filter applied to both the velocity-weighted kSZ maps (ACT DR6 $\times$ DESI DR2) and the galaxy-galaxy lensing shear field (HSC Y3), whose bin-by-bin ratio yields the projected gas fraction $f_{\rm gas}^{\rm obs}(R)$ with no profile fitting.

The present proposal combines the two: apply the Singh et al. substitution to the **kSZ side** rather than the lensing side. The target becomes the filtered baryon-matter cross-correlation ratio $Y_{bm}/Y_{mm}$, constructed from the measured kSZ stack $Y_{gb}$, the measured galaxy clustering $Y_{gg}$, and a theory prediction for $Y_{mm}$. As shown in Section 3, **galaxy-galaxy lensing drops out of the estimator entirely**, and the unobservable baryon auto-correlation $Y_{bb}$ cancels algebraically. The only simulation-calibrated ingredient is the ratio of two cross-correlation coefficients, $r_{bm}/r_{gb}$.

Three properties make this attractive:

- **Error suppression by $f_b$.** The baryon-matter cross-correlation enters the total-matter power suppression with weight $\sim f_b \approx 0.156$ (Section 4). A 20% calibration error on $r_{bm}/r_{gb}$ propagates to only $\sim 3\%$ on the total-matter transfer function ($\sim 6\%$ on power). The estimator tolerates calibration uncertainty that would be fatal in a direct measurement.
- **Footprint.** The $f_{\rm gas}$ measurement is bottlenecked by the DESI $\cap$ HSC overlap (546 deg$^2$). The new estimator requires only DESI clustering and the kSZ stack, both available over the full DESI $\cap$ ACT overlap (6,709 deg$^2$). The limiting lensing statistics are replaced by high-S/N clustering plus theory, an order-of-magnitude gain in usable area for the cross-correlation target.
- **Directness.** $Y_{bm}/Y_{mm}$ is the filtered configuration-space analogue of $P_{bm}/P_{mm}$, the object that appears in the field-level decomposition of the suppression. This shortens the inference chain relative to the $f_{\rm gas} \to \Delta P/P$ mapping (SP(k)-style), and the two routes cross-check each other.

The immediate deliverable is **not** a data measurement. It is a simulation study, in the spirit of Fig. 1 of Singh et al. (2020): compute $r_{gb}(R)$ and $r_{bm}(R)$ across hydrodynamical simulations spanning feedback prescriptions, for the three filters $\{\Sigma, \Delta\Sigma, \Upsilon(R_0 = 1')\}$, over the observationally accessible range $1'$-$6'$, and determine whether the ratio $r_{bm}/r_{gb}$ is (i) close to unity and (ii) stable across codes and feedback models. If yes, the estimator inherits the model-independence of the measured inputs up to a small, well-quantified transfer.

---

## 2. Notation and Filtered Observables

### 2.1 Fields

- $g$: galaxies (DESI BGS or LRG samples; SHAM-selected analogues in simulations).
- $b$: baryons. **Caution:** the kSZ traces free electrons only. We write $e$ for the ionized-gas field and $b$ for all baryons (ionized + neutral gas + stars) when the distinction matters; Task 3 quantifies the difference.
- $m$: dark matter (CDM). This follows the notation of the source discussion; note it differs from the $f_{\rm gas}$ paper, where $\Omega_m$ denotes total matter.
- $t = m + b$: total matter, the field relevant for lensing and for $P(k)$ suppression.

Mass fractions: $f_b \equiv \Omega_b/(\Omega_b + \Omega_{\rm dm}) \approx 0.156$ (Planck 2018), $f_m = 1 - f_b \approx 0.844$, so that at the field level

$$\delta_t = f_m\,\delta_m + f_b\,\delta_b. \tag{1}$$

### 2.2 Filters

All observables are projected (2D) fields filtered at aperture $R$ with one of:

- $\Sigma(R)$: azimuthally averaged surface density (or filtered map value) at $R$.
- $\Delta\Sigma(R) = \bar{\Sigma}(<R) - \bar{\Sigma}(R, R + \delta R)$: the compensated disk-minus-annulus filter of the $f_{\rm gas}$ paper (its Eqs. 14-16), $\delta R = 0.75'$.
- $\Upsilon(R; R_0) = \Delta\Sigma(R) - (R_0/R)^2\,\Delta\Sigma(R_0)$: the Baldauf et al. (2010) estimator, which nulls all information below $R_0$. Fiducial $R_0 = 1'$.

Apertures: 9 linear bins over $1'$-$6'$, matching the existing pipeline.

### 2.3 Generic filtered two-point amplitudes

Let $Y_{XY}(R; \mathcal{F})$ denote the stacked filtered cross-correlation between fields $X$ and $Y$ with filter $\mathcal{F} \in \{\Sigma, \Delta\Sigma, \Upsilon\}$. Concretely:

- $Y_{gb}$: the velocity-weighted kSZ stack with filter $\mathcal{F}$, converted to gas surface density units. **Measured** (existing pipeline; ACT DR6 $\times$ DESI DR2). Subject to the beam and to velocity-reconstruction normalization (Section 6, Task 5).
- $Y_{gg}$: the filter applied to the projected galaxy surface density field, stacked on the same galaxy sample. **Measured** (DESI clustering; construction detailed in Section 5). High S/N; this is the quantity that replaces bias modelling, exactly as $w_{gg}$ did in Singh et al. (2020).
- $Y_{mm}$: the filtered dark-matter auto-correlation. **Theory** (halofit / N-body emulator, plus the exact filter and binning transfer).
- $Y_{gm}$, $Y_{gt}$: galaxy-matter cross from lensing. **Not required** by the estimator; retained as an external cross-check. Note lensing strictly measures $Y_{gt}$, not $Y_{gm}$.
- $Y_{bb}$, $Y_{bm}$: unobservable directly; the first cancels, the second is the target.

---

## 3. The Estimator

Define the cross-correlation coefficients at each aperture $R$ and for each filter:

$$r_{gb}(R) \equiv \frac{Y_{gb}}{\sqrt{Y_{gg}\,Y_{bb}}}, \qquad r_{bm}(R) \equiv \frac{Y_{bm}}{\sqrt{Y_{bb}\,Y_{mm}}}. \tag{2}$$

Solve the first for $Y_{bb} = Y_{gb}^2 / (r_{gb}^2\, Y_{gg})$ and substitute into the second:

$$Y_{bm} = \frac{r_{bm}}{r_{gb}}\; Y_{gb}\,\sqrt{\frac{Y_{mm}}{Y_{gg}}}, \tag{3}$$

so that the target ratio is

$$\boxed{\;\frac{Y_{bm}}{Y_{mm}} = \frac{r_{bm}}{r_{gb}} \cdot \frac{Y_{gb}}{\sqrt{Y_{gg}\,Y_{mm}}}\;} \tag{4}$$

(Note: the source discussion wrote $Y_{gb}/(Y_{mm}Y_{gg})$; the square root in Eq. 4 follows from the algebra above and is dimensionally required.)

Structural observations:

1. **$Y_{bb}$ cancels.** The baryon auto-correlation, which no current observable measures cleanly at these scales, never needs to be known.
2. **Lensing is not required.** $Y_{gm}$ appears nowhere in Eq. (4). The estimator runs on the full kSZ footprint. Lensing re-enters only as a consistency test: the ratio of the existing $f_{\rm gas}^{\rm obs} \propto Y_{gb}/Y_{gt}$ estimator to Eq. (4) isolates $r_{gt}\sqrt{Y_{mm}Y_{gg}}/Y_{gt}$-type combinations, i.e., a measurement-level check on the galaxy-matter coefficient that Singh et al. calibrated from mocks.
3. **The model-dependence boundary is explicit.** Everything on the right of Eq. (4) is measured except (i) $Y_{mm}$ from theory, which carries non-linear matter power accuracy (percent-level at these scales, testable against N-body), and (ii) the ratio $r_{bm}/r_{gb}$, which carries all astrophysical stochasticity and is the object of the simulation programme. This mirrors the Singh et al. structure, where $r_{cc}$ carried the analogous burden and shifted $S_8$ by only $\sim 0.3\sigma$ even when set to 1 everywhere.
4. **Expectation of near-unity.** On large scales all fields trace the same long-wavelength modes and both $r$'s $\to 1$; deviations are confined to scales where non-linear stochasticity, feedback-driven gas displacement, and satellite/halo-exclusion effects enter. Whether $1'$-$6'$ (roughly $0.3$-$2\,\mathrm{cMpc}$ for BGS, $0.5$-$3.5\,\mathrm{cMpc}$ for LRG bin 1) is "large scale" in this sense is precisely the question. There is also a partial-cancellation argument for the **ratio**: both coefficients share the factor $Y_{bb}^{-1/2}$ and both involve the gas field, so code-to-code variations may partially cancel in $r_{bm}/r_{gb}$ even where the individual coefficients deviate from unity. This should be tested, not assumed.

---

## 4. Connection to Power Suppression and Error Budget

From Eq. (1), the total-matter auto-spectrum decomposes as

$$P_{tt} = f_m^2 P_{mm} + 2 f_m f_b P_{bm} + f_b^2 P_{bb}. \tag{5}$$

Writing $x \equiv P_{bm}/P_{mm}$ and assuming $r_{bm} \approx 1$ so that $P_{bb} \approx x^2 P_{mm}$,

$$\frac{P_{tt}}{P_{mm}} \approx (f_m + f_b x)^2, \qquad T \equiv \sqrt{P_{tt}/P_{mm}} = f_m + f_b x. \tag{6}$$

Error propagation for $x \approx 1$:

$$\frac{\delta T}{T} \approx f_b\,\frac{\delta x}{x} \approx 0.156\,\frac{\delta x}{x}, \qquad \frac{\delta (P_{tt}/P_{mm})}{P_{tt}/P_{mm}} \approx 2 f_b\,\frac{\delta x}{x}. \tag{7}$$

So a **20% error on $r_{bm}/r_{gb}$ (hence on $x$) yields $\approx 3\%$ on the transfer function $T$ and $\approx 6\%$ on the power ratio**; if instead only the cross term is perturbed with $P_{bb}$ held fixed, the weight is $2 f_m f_b \approx 0.26$, giving $\approx 5\%$ on power. Under every bookkeeping the error is suppressed by roughly one power of $f_b$. This is the quantitative sense in which the estimator is forgiving: even a crude calibration of the $r$ ratio delivers a competitive suppression constraint, and a 5-10% calibration delivers a percent-level one.

Two caveats on the mapping itself:

- Eq. (5) is exact for 3D power spectra of a given hydro simulation. Our observable is a **filtered projected** analogue; the mapping from $Y_{bm}/Y_{mm}(R)$ to $x(k)$ requires either the window formalism or, more robustly, the same forward-modelled cross-suite regression already planned for the $f_{\rm gas} \to \Delta P/P$ programme, now with $Y_{bm}/Y_{mm}$ as the (more direct) input statistic.
- The suppression usually quoted in the literature is $P_{tt}^{\rm hydro}/P_{tt}^{\rm DMO}$. Connecting $P_{mm}$ (hydro CDM auto) to the gravity-only theory prediction introduces the baryonic back-reaction on the CDM itself, a $\sim 1$-$2\%$ effect at these scales (van Daalen et al. 2011; Chisari et al. 2018). Task 4 pins down which convention $r_{bm}$ is defined against and keeps it consistent between calibration and application.

---

## 5. Computing $Y_{gg}$ (and the Matching Theory Transfer)

$Y_{gg}$ is the one estimator input that is neither in the existing $f_{\rm gas}$ pipeline nor a standard published data product at arcminute scales, so its construction is specified here. The guiding principle throughout: the discrete filter operation applied to the data must be reproduced exactly in the theory transfer for $Y_{mm}$ and in the simulation calibration of $r_{bm}/r_{gb}$; because the $r$'s are ratios, any filter-implementation mismatch between data and simulations aliases directly into a spurious calibration error.

### 5.1 Measurement: pair counts, never the power spectrum

$Y_{gg}$ is the kSZ stacking operation with the CMB map replaced by the galaxy number density field. The chain, following Singh et al. (2020, Secs. 2.2-2.4) with our filter substituted at the final step:

1. **Measure $\xi_{gg}(r_p, \Pi)$**, the 3D galaxy two-point correlation function in transverse/line-of-sight separation bins, with the Landy-Szalay estimator $\xi_{gg} = (DD - 2DR + RR)/RR$ using systematics-weighted DESI galaxies and the matching randoms.
2. **Project:** $w_{gg}(r_p) = \sum_\Pi \Delta\Pi\, \xi_{gg}(r_p, \Pi)$ with $\Pi_{\max} = 100\,h^{-1}$Mpc (Kaiser correction for the finite-$\Pi_{\max}$ RSD residual to be rechecked at our scales; Singh et al. found it negligible at theirs).
3. **Promote to a surface density:** $\Sigma_{gg}(r_p) \propto w_{gg}(r_p)$. The proportionality constant is pure convention (Singh et al. use $\bar{\rho}_m$ to match lensing units); see Section 5.4.
4. **Apply the filter:** compute $\bar{\Sigma}(<R) = (2/R^2)\int_0^R r'\,\Sigma(r')\,dr'$ and the annulus mean $\bar{\Sigma}(R, R+\delta R)$, then $\Delta\Sigma_{gg}(R) = \bar{\Sigma}(<R) - \bar{\Sigma}(R, R+\delta R)$ and $\Upsilon_{gg}(R; R_0) = \Delta\Sigma_{gg}(R) - (R_0/R)^2 \Delta\Sigma_{gg}(R_0)$.

Steps 1-2 can equivalently be replaced by a map-level operation (project the galaxy field in a redshift slice, stack the filtered map on the lens positions), which is geometrically identical to the kSZ measurement; the cylinder pair-count route has better line-of-sight noise properties and is the default. Covariance: jackknife over the same 100 $k$-means regions as the other observables, so the joint covariance of $(Y_{gb}, Y_{gg})$ is internally consistent.

**Filter-definition caveat.** Singh et al.'s $\Delta\Sigma$ is $\bar{\Sigma}(<R) - \Sigma(R)$ with the *local* value at $R$; the $f_{\rm gas}$ paper filter uses the *annulus mean* over $[R, R+\delta R]$ with $\delta R = 0.75'$. The annulus mean equals $\Sigma(R)$ only up to a term involving the profile gradient across the annulus, which is not automatically negligible in the inner bins at our aperture range. All $Y$'s in this programme use the annulus-mean definition, matching the existing kSZ pipeline; the Singh formulas are the template, not the specification.

**Self-pairs.** Each lens galaxy contributes $1/(\pi R^2)$ to its own disk mean and nothing to the annulus; the compensated filter does not remove this because it is not a constant background. Self-pairs must be excluded explicitly from the counts. With self-pairs excluded, shot noise does not bias the mean $Y_{gg}$ but does enter its covariance.

**Fibre incompleteness.** The range $1'$-$6'$ sits at and below the DESI fibre patrol scale at the relevant redshifts, where pair completeness is suppressed. Pairwise-inverse-probability weights and/or angular upweighting from the alternate-MTL realizations must be applied and validated at these specific angular scales (Task 5); a scale-dependent completeness error in $Y_{gg}$ enters the estimator as $Y_{gg}^{-1/2}$ and mimics a scale-dependent $r$.

### 5.2 Theory: configuration-space chain

For $Y_{mm}$, and for validating the measured $Y_{gg}$ against theory expectations, the configuration-space route mirrors the measurement: $P(k) \to \xi(r)$ by Hankel transform, $\to w_p(r_p) = \int_{-\Pi_{\max}}^{\Pi_{\max}} d\Pi\, \xi\!\left(\sqrt{r_p^2 + \Pi^2}\right)$ with the same $\Pi_{\max}$ and Kaiser factor as the data, $\to$ steps 3-4 above. This is the Baldauf et al. (2010) / Singh et al. (2020) computation with our filter substituted.

### 5.3 Theory: harmonic-space chain (preferred)

All filters here are linear, so each filtered amplitude is a single integral of the projected (Limber) power spectrum against an analytic kernel:

$$Y(R) = \int \frac{k\,dk}{2\pi}\; P_{2D}(k)\; \hat{W}(k; R),$$

with disk-mean kernel $\hat{W}_{\rm disk}(k; R) = 2J_1(kR)/(kR)$, annulus-mean kernel

$$\hat{W}_{\rm ann}(k; R_1, R_2) = \frac{2\left[R_2 J_1(kR_2) - R_1 J_1(kR_1)\right]}{k\,(R_2^2 - R_1^2)},$$

and therefore

$$\hat{W}_{\Delta\Sigma}(k; R) = \frac{2J_1(kR)}{kR} - \hat{W}_{\rm ann}(k; R, R+\delta R), \qquad \hat{W}_{\Upsilon} = \hat{W}_{\Delta\Sigma}(k; R) - \frac{R_0^2}{R^2}\,\hat{W}_{\Delta\Sigma}(k; R_0).$$

$\hat{W}_{\Delta\Sigma}(k \to 0) \to 0$ expresses the compensation. This shares machinery with the existing analytic (Hankel-space) beam cross-check, makes finite-$\delta R$, pixelization, and bin-averaging effects explicit, and is the recommended implementation for $Y_{mm}$. Bin averaging over the 9 aperture bins is applied to the kernels, not approximated at bin centres, unless a Singh-style Appendix D test shows the difference is negligible.

### 5.4 Normalization convention

All $Y$'s are defined as mean-density-normalized projected overdensity amplitudes, so every $Y$ is dimensionless, the $r$'s are convention-free ratios, and the kSZ temperature-to-optical-depth conversion stays quarantined in $\alpha_{\rm conv}$ exactly as in the $f_{\rm gas}$ paper. Any convention works provided it is applied identically in data, theory, and simulations; this one minimizes bookkeeping.

---

## 6. Simulation Validation Programme

Simulations: the six runs already in the pipeline (Illustris-1, TNG300-1, SIMBA-100, FLAMINGO L1_m9 fiducial, fgas$-8\sigma$, Jet_fgas$-4\sigma$), with SHAM-selected samples matched to BGS ($z = 0.26$; FLAMINGO at $z = 0.30$) and LRG bin 1 ($z = 0.5$) as in the $f_{\rm gas}$ paper. ANTILLES and CAMELS to be added later for density in feedback space once the machinery exists.

### Task 1: The $r$ profiles (the Fig. 1 analogue)

For each simulation, sample, and filter $\mathcal{F} \in \{\Sigma, \Delta\Sigma, \Upsilon(R_0 = 1')\}$, compute from the projected maps:

- $r_{gb}(R)$, using the SHAM galaxy sample and the ionized-gas map;
- $r_{bm}(R)$, using the ionized-gas and CDM maps;
- the ratio $r_{bm}/r_{gb}(R)$.

Produce the direct analogue of Singh et al. (2020) Fig. 1: $r$ versus $R$ over $1'$-$6'$, one curve per simulation, one panel per filter. Deliverable metrics: $\max_R |r - 1|$ per filter; cross-simulation scatter of $r_{bm}/r_{gb}$ at each $R$; scale dependence of the ratio.

Morphological expectation from Singh et al.: the $\Sigma$-based coefficient deviates from unity only at small $R$ (deviations are localized), while $\Upsilon$ carries the deviation out to larger $R$ through the $\Delta\Sigma(R_0)$ reference term (their Fig. 1 shows $r_{cc}^{(\Upsilon)}$ reaching $\sim 1.3$ where $r_{cc}^{(\Sigma)} \lesssim 1.1$). The same trade-off should appear here, and the filter recommendation should come out of this task, not go into it. Note the gas field is smoother than the galaxy field (pressure, feedback), so the morphology of $r_{gb}$ has no clean precedent; this is the measurement.

Decision criterion: if the cross-code scatter of $r_{bm}/r_{gb}$ is $\lesssim 10\%$ over the usable aperture range, Eq. (7) bounds the induced suppression error at $\lesssim 1.6\%$ (transfer level) and the estimator is viable with a fixed transfer plus prior width, following the Singh et al. "small"/"wide" prior strategy. If the scatter is larger or strongly scale-dependent, the parametrized-$r$ route (their Eq. 26 analogue with sim-derived priors, marginalized) is the fallback.

Cross-family discipline (lesson from Singh et al. Appendix A): priors derived from one mock family biased results on an independent family at the aggressive scale cut ($r_0 = 1\,h^{-1}$Mpc) at the 2-3% level. Validation must therefore be cross-code (Illustris vs TNG vs SIMBA vs FLAMINGO), not merely cross-parameter within one code.

### Task 2: The $\Sigma$ filter and point-mass completion

We measure $\Delta\Sigma$-filtered quantities; the plain $\Sigma$ filter is not directly measured on the lensing side and, on the kSZ side, an uncompensated disk mean reintroduces large-scale CMB noise. Two options to evaluate:

(a) Restrict the analysis to $\Delta\Sigma$ and $\Upsilon$, both measurable on all inputs. Simplest and likely sufficient if Task 1 shows acceptable $r$ behaviour for these filters.

(b) Reconstruct $\Sigma$ from $\Delta\Sigma$ via enclosed-mass completion: since $\Delta\Sigma(R) = \bar{\Sigma}(<R) - \Sigma(R)$, knowledge of $\Delta\Sigma$ at $R > R_{\min}$ plus a single nuisance parameter for the enclosed mass below $R_{\min}$, treated as a point mass contributing $\propto R^{-2}$, determines $\Sigma(R)$. This is the exact analogue of the $\Sigma_0$/$\Delta\Sigma_0$ construction in Singh et al. (their Eqs. 29-30), where the nuisance absorbed most small-scale modelling error. Task: implement in simulations, quantify the bias on $r^{(\Sigma)}$'s induced by the point-mass approximation as a function of $R_{\min}$. **Discussion item with Uroš:** whether the localization advantage of $\Sigma$-based $r$'s (per Singh Fig. 1) justifies the extra nuisance parameter.

### Task 3: Electrons versus baryons

The kSZ stack measures $Y_{ge}$ (free electrons), while Eqs. (1)-(7) require $b$ = all baryons. In each simulation, compute both $r_{ge}, r_{em}$ and $r_{gb}, r_{bm}$, and the correction factor $Y_{gb}/Y_{ge}(R)$. The stellar contribution is centrally concentrated and code-dependent (FLAMINGO carries roughly twice the stellar mass of TNG for these halo samples, per the $f_{\rm gas}$ paper Figs. 16-17); SIMBA's on-the-fly molecular partitioning removes gas from the ionized budget. Decide between two framings:

- Target $P_{em}/P_{mm}$ (ionized-gas cross) and add stellar + neutral terms in the suppression mapping separately, with external constraints (stellar mass functions, 21 cm);
- Target $P_{bm}/P_{mm}$ and absorb the electron-to-baryon step into a simulation-calibrated, code-dependent transfer.

Recommendation pending Task 3 results: carry both, report the split systematically, and let the cross-suite scatter of each decide which is the cleaner deliverable.

### Task 4: The theory side, $Y_{mm}$

- Compute $Y_{mm}(R; \mathcal{F})$ from halofit/emulator $P_{mm}(k)$ through the exact filter, projection, and bin-averaging transfer; validate against the gravity-only counterparts of the hydro suites (or their DMO pairs where available).
- Quantify the back-reaction gap $P_{mm}^{\rm hydro\,CDM}/P_{mm}^{\rm DMO}$ over the relevant scales in each suite, and fix the convention for $r_{bm}$ (hydro-CDM based) consistently between calibration and application. Expected $1$-$2\%$; must be booked, not ignored, given the percent-level error budget of Eq. (7).
- Note the Kaiser/RSD and $\Pi_{\max}$ treatment for $Y_{gg}$: the projected clustering must use a line-of-sight projection consistent with what the theory transfer assumes (Section 5.2); Singh et al. found RSD corrections negligible for $\Pi_{\max} = 100\,h^{-1}$Mpc at their scales, but our apertures are smaller and this should be rechecked.

### Task 5: Measurement-side systematics inherited from the $f_{\rm gas}$ pipeline

- **Beam.** The ACT beam suppresses $Y_{gb}$ only; unlike the $f_{\rm gas}$ ratio, nothing on the denominator side shares it, so the same treatment applies: forward-model the beam into the simulated $Y_{gb}$ (preferred, as in the $f_{\rm gas}$ paper Sec. V B) or apply the feedback-independent $C_{\rm beam}(R)$ compensation.
- **Velocity reconstruction.** The Ondaro-Mallea et al. (2026, arXiv:2607.23339) non-linear suppression of the stacked signal (10-20% for realistic reconstructions retaining non-linear/RSD information, weakly feedback-dependent) enters $Y_{gb}$ directly and no longer has any chance of partial cancellation against a lensing denominator. Whether it is flat or scale-dependent over $1'$-$6'$ must be resolved (their figures; Boryana) before the error budget is finalized; the forward-modelling route, running the DESI reconstruction on FLAMINGO mocks, handles either case and would make the mock $Y_{gb}$ and data $Y_{gb}$ commensurable by construction.
- **$Y_{gg}$ at arcminute scales.** DESI fibre-assignment incompleteness affects small angular separations; pairwise-inverse-probability or angular upweighting corrections need evaluation over $1'$-$6'$ specifically, since this range sits at or below the fibre patrol scale at the relevant redshifts.

### Task 6: Forecast

Propagate the Task 1 cross-suite scatter of $r_{bm}/r_{gb}$ through Eq. (7), combine with the statistical covariance of $Y_{gb}$ (existing bootstrap machinery) scaled to the full 6,709 deg$^2$ footprint and the $Y_{gg}$, $Y_{mm}$ uncertainties, and produce a projected error on $T(k)$-level suppression. Compare against: (i) the SP(k)-mediated $f_{\rm gas}$ route; (ii) published joint kSZ + lensing constraints (Bigwood et al. 2024; McCarthy et al. 2025; Siegel et al. 2025).

---

## 7. Relation to the Existing $f_{\rm gas}$ Measurement

The $f_{\rm gas}$ paper stands on its own as the model-independent gas-fraction measurement. This programme reuses its components with one substitution:

| Quantity | $f_{\rm gas}$ paper | This programme |
|---|---|---|
| kSZ stack $Y_{gb}$ | numerator | numerator (unchanged pipeline) |
| Lensing $Y_{gt}$ | denominator | dropped (cross-check only) |
| Clustering $Y_{gg}$ | unused | denominator (measured) |
| Theory $Y_{mm}$ | unused | denominator (modelled) |
| Calibrated transfer | $C_{\rm beam}$ | $C_{\rm beam}$ + $r_{bm}/r_{gb}$ |
| Footprint | 546 deg$^2$ | 6,709 deg$^2$ |

The two estimators measure different physical ratios ($f_{\rm gas} \sim$ gas over total matter; Eq. 4 $\sim$ gas-CDM cross over CDM auto) and their consistency, where the footprints overlap, is itself a test of the $r$ calibrations. Neither supersedes the other; the pair brackets the model-dependence from both sides.

## 8. End-to-End Execution Plan

Phases are ordered by dependency; gates are explicit decision points. Phases 1-2 are simulation-only and carry no data risk; nothing downstream of Gate A should be built before Gate A passes.

### Phase 0: Conventions and infrastructure

1. Write a one-page filter specification: annulus-mean $\Delta\Sigma$ with $\delta R = 0.75'$, $\Upsilon(R_0 = 1')$, 9 aperture bins over $1'$-$6'$, pixel-counting area convention, bin-averaging rule, self-pair exclusion, and the dimensionless normalization of Section 5.4. This document is the single source of truth for data, theory, and simulation code paths.
2. Extend `SimulationStacker` to output CDM-only projected maps alongside the existing total-matter and ionized-gas maps; add total-baryon (gas + stars) maps for Task 3.
3. Implement the three filters as one shared map-level operation, unit-tested against the analytic kernels of Section 5.3 on a known input profile.

### Phase 1: Simulation $r$ profiles (Task 1)

4. Run Task 1 on TNG300-1 alone (fastest turnaround): $r_{gb}(R)$, $r_{bm}(R)$, $r_{bm}/r_{gb}(R)$ for all three filters, both samples. Fix plot conventions on this output.
5. Sweep the remaining five simulations; produce the Singh Fig. 1 analogue and the cross-suite scatter metrics.
6. **Gate A:** is the cross-code scatter of $r_{bm}/r_{gb}$ $\lesssim 10\%$ (fixed-transfer route), $10$-$20\%$ (parametrized-$r$ route with marginalized priors), or larger/strongly scale-dependent (estimator not viable as posed; return to Uroš with the diagnosis)? The $f_b$ suppression of Eq. (7) means even the middle outcome delivers a percent-level suppression constraint.

### Phase 2: Design decisions in simulations (Tasks 2-3)

7. Task 2: implement the $\Sigma$ point-mass completion in simulations; quantify the induced bias on $r^{(\Sigma)}$ versus $R_{\min}$. Decide with Uroš whether $\Sigma$ enters the final analysis or the filter set is $\{\Delta\Sigma, \Upsilon\}$ only.
8. Task 3: compute the electron-versus-baryon split ($r_{ge}$ vs $r_{gb}$, $Y_{gb}/Y_{ge}(R)$) per code; fix the target ($P_{em}/P_{mm}$ with external stellar/neutral terms, or $P_{bm}/P_{mm}$ with a calibrated transfer) based on which shows smaller cross-suite scatter.
9. **Gate B:** filter set and field definition frozen. Everything downstream uses exactly these.

### Phase 3: Theory pipeline (Task 4)

10. Implement $Y_{mm}$ via the harmonic-space chain (Section 5.3) with the frozen filter set; validate against DMO/gravity-only simulation counterparts to percent level.
11. Book the hydro-CDM versus DMO back-reaction gap per suite; fix the $r_{bm}$ convention consistently.
12. Recheck the RSD/$\Pi_{\max}$ residual for the $Y_{gg}$ transfer at our apertures.

### Phase 4: Data measurement of $Y_{gg}$

13. Build the pair-count pipeline (Section 5.1) on DESI DR2 BGS and LRG bin 1 with systematics weights and the frozen filter; jackknife covariance on the shared 100 regions.
14. Fibre-incompleteness campaign: apply PIP/angular-upweighting corrections; validate on mocks with realistic fibre assignment; quantify the residual scale-dependent error over $1'$-$6'$ explicitly.
15. Null and consistency tests: randoms-based nulls; NGC/SGC split (the LRG bin 2 lesson); comparison of cylinder pair-count and map-level implementations.

### Phase 5: kSZ-side systematics closure (Task 5)

16. Resolve the Ondaro-Mallea radius-dependence question (their figures; Boryana). If scale-dependent, run the DESI reconstruction on FLAMINGO mocks and adopt full forward-modelling of $Y_{gb}$; if flat, fold into the multiplicative budget with $v_{\rm los}$ and $r_V$.
17. Carry over the beam treatment: forward-model the beam into simulated $Y_{gb}$ (preferred) with `use_sim_scatter` enabled, or apply $C_{\rm beam}(R)$ with its prior width.

### Phase 6: Combine

18. Assemble Eq. (4) per aperture bin: measured $Y_{gb}$ (existing pipeline, full 6,709 deg$^2$ footprint), measured $Y_{gg}$ (Phase 4), theory $Y_{mm}$ (Phase 3), calibrated $r_{bm}/r_{gb}$ with its Gate A prior (Phase 1). Propagate the block covariance (kSZ bootstrap $\times$ clustering jackknife; treat cross-covariance between $Y_{gb}$ and $Y_{gg}$, which share the lens sample, via the jackknife rather than assuming independence).
19. Consistency test against $f_{\rm gas}^{\rm obs}$ on the DESI $\cap$ HSC $\cap$ ACT overlap: the ratio of the two estimators isolates the galaxy-matter coefficient and validates the $r$ calibration with data.
20. **Gate C:** internal consistency (source-bin splits, field splits, filter choices) at the level of the statistical errors.

### Phase 7: Suppression mapping and paper (Task 6)

21. Cross-suite regression from the filtered $Y_{bm}/Y_{mm}(R)$ (beam-convolved observable space) to $T(k)$ / $P_{tt}/P_{mm}$ suppression, extending to ANTILLES for feedback-space density; quantify sufficiency and residual scatter, per the Eq. (7) budget.
22. Forecast and final error budget; comparison against the SP(k)-mediated $f_{\rm gas}$ route and published joint kSZ + lensing constraints; write up.

### Standing parallel items

- Ondaro-Mallea resolution (step 16) can start immediately; it also feeds the $f_{\rm gas}$ paper's Section IV C 1 before its submission.
- The Fig. 1 analogue (step 5) plus the Task 2/Task 3 decisions (steps 7-8) are the agenda for the next meeting with Uroš.

---

## References

- Baldauf T., Smith R. E., Seljak U., Mandelbaum R., 2010, PRD 81, 063531
- Bigwood L., et al., 2024, MNRAS 534, 655
- Chisari N. E., et al., 2018, MNRAS 480, 3962
- Liu R. H., 2026, in preparation (the $f_{\rm gas}$ paper) and companion simulation paper
- McCarthy I. G., et al., 2025, MNRAS 540, 143 (arXiv:2410.19905)
- Ondaro-Mallea L., Angulo R. E., Hadzhiyska B., Schaye J., 2026, arXiv:2607.23339
- Salcido J., et al., 2023, MNRAS 523, 2247 (SP(k))
- Siegel J., et al., 2025, arXiv:2509.10455; 2025, arXiv:2512.02954
- Singh S., Mandelbaum R., Seljak U., Rodríguez-Torres S., Slosar A., 2020, MNRAS 491, 51 (arXiv:1811.06499)
- van Daalen M. P., Schaye J., Booth C. M., Dalla Vecchia C., 2011, MNRAS 415, 3649
- van Daalen M. P., McCarthy I. G., Schaye J., 2020, MNRAS 491, 2424