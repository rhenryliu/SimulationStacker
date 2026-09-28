# The Programme, Explained

**A synthesis of `cross_correlation_notes.md` (v0.1), `tasks_1_to_4_record.md`, `cross_correlation_notes_v0.2_addendum.md`, `cross_correlation_notes_v0.3_response.md` and `tasks_7_to_10_record.md`.**
Status as of 2026-09-10. R. Henry Liu, with U. Seljak.

---

## How to read this

Three passes over the same material, each complete on its own.

- **Part I** is the whole programme in about a page. If you read nothing else, read this.
- **Part II** explains each idea in plain terms, one short section per idea, in the order the physics runs. Each section ends with a pointer into Part III.
- **Part III** carries the algebra, the tables, and the complete ledger of assumptions.

| if you want | go to |
|---|---|
| the goal and current status | Part I |
| why galaxy bias forced this design | II.3, II.4 |
| what the estimator actually is | II.5, III.B |
| why the calibration factor is nearly 1 | II.6, III.C |
| why there are four aperture filters | II.7, III.D |
| how an aperture profile becomes a $P(k)$ suppression | II.8, III.E |
| what the simulations found | II.9, III.F |
| **every assumption, with its size** | **II.10, III.G** |
| what is still open, in priority order | III.H |
| which document is authoritative for what | III.I |

---

# Part I. The whole thing in one page

**The goal.** Baryonic feedback moves gas out of haloes and suppresses the total matter power spectrum by up to tens of per cent at $k \gtrsim 0.5\,h\,$Mpc$^{-1}$. That suppression is the leading systematic for weak lensing cosmology. The quantity that controls it at leading order is the baryon-matter cross-correlation $P_{bm}/P_{mm}$. This programme measures that quantity in a way that depends on hydrodynamic simulations as little as the data allow.

**The observables.** Three surveys, each giving one projected two-point amplitude at aperture $R$:

| symbol | what it is | source |
|---|---|---|
| $Y_{gb}$ | velocity-weighted kSZ stack, i.e. the gas around galaxies | ACT DR6 $\times$ DESI DR2, 6,709 deg$^2$ |
| $Y_{gm}$ | galaxy-galaxy lensing, i.e. the total matter around the same galaxies | HSC Y3, 546 deg$^2$ overlap |
| $Y_{gg}$ | projected galaxy clustering, the same filter on the galaxy field | DESI DR2, not yet built |

**The chain.** The target is $x \equiv Y_{bm}/Y_{mm}$, the filtered version of $P_{bm}/P_{mm}$. Two algebraically equivalent routes reach it, differing in which observables they use:

$$
x = \underbrace{\frac{r_{bm}}{r_{gb}}}_{\text{from simulations}} \cdot \frac{Y_{gb}}{\sqrt{Y_{gg}\,Y_{mm}}} \qquad \text{(Route A: kSZ + clustering + theory, 6,709 deg}^2\text{)}
$$

$$
x = \underbrace{\frac{r_{bm}\,r_{gm}}{r_{gb}}}_{\equiv\, C,\ \text{from simulations}} \cdot \frac{Y_{gb}}{Y_{gm}} \qquad \text{(Route B: kSZ / lensing, 546 deg}^2\text{)}
$$

Route B's observable, $Y_{gb}/Y_{gm}$, is exactly what the $f_{\rm gas}$ paper already measures. Then $x$ maps to the power suppression through $S = (f_m + f_b x)^2$ with $f_b \approx 0.157$, so errors on the calibration are suppressed by roughly one power of $f_b$: a 6 per cent error on $C$ gives 1.9 per cent on $S$.

**What the simulations must deliver.** Only the calibration factor. Everything else is measured or modelled. So the entire simulation programme exists to answer one question: is $C$ close to 1, and is it stable across codes and feedback models?

**Where it stands.**

- **Gate A passes.** $C = 1.11 \to 1.00$ over $1'$ to $6'$ at $z\approx0.5$, with a cross-run scatter of 4.7 per cent for $\Delta\Sigma$. The old statistic $r_{bm}/r_{gb}$ was a 22 to 45 per cent transfer with 8 to 13 per cent scatter that straddled the viability threshold. Replacing it with the four-amplitude $C$ collapses the cross-code disagreement from 10.9 to 1.3 per cent at $z\approx0.5$.
- **Route B's raw observable is already nearly unbiased.** With no correction at all, $Y_{gb}/Y_{gm}$ recovers the truth to within 2 to 8 per cent. Route A's raw observable is off by 36 to 54 per cent. This is the single strongest result in the programme.
- **Gate B is reopened**, not on the filter set as expected but on the field definition: the electron target carries twice the cross-run scatter of the baryon target.
- **Gate C is not reached**, and Phase 4 (the data-side $Y_{gg}$) has not started.

**The five things that could still break it.**

1. Only two code families, TNG and FLAMINGO. Singh et al.'s Appendix A lesson is that priors calibrated on one family bias another, and that discipline is unexercised.
2. $C$ depends on feedback strength, so it is not the feedback-blind quantity the theory suggested. It is currently booked as a prior width.
3. The cross-code agreement is 1.3 per cent at $z\approx0.5$ but 9.7 per cent at $z\approx0.26$, and nobody knows why.
4. The electron-versus-baryon target is unresolved and two code families cannot adjudicate it.
5. Route A still needs a $Y_{gg}$ measurement at arcminute scales, below the DESI fibre patrol radius, which has not been attempted.

---

# Part II. The ideas, plainly

## II.1 The physics problem

Dark matter only simulations predict $P(k)$ to sub-per-cent accuracy. Real matter contains baryons, and AGN and supernovae push gas out of haloes and into the surrounding medium. Removing gas from the centre of a halo lowers the small-scale power, by up to tens of per cent at $k \gtrsim 0.5\,h\,$Mpc$^{-1}$. Different simulation codes disagree about how big the effect is, because they disagree about feedback, so you cannot simply model it away.

Write the total matter field as a mass-weighted sum of the CDM and the baryons,

$$
\delta_t = f_m\,\delta_m + f_b\,\delta_b, \qquad f_b = \Omega_b/\Omega_m \approx 0.157 .
$$

Squaring this shows that the suppression is controlled at leading order by how well the baryon field still traces the matter field, that is, by the ratio $x = P_{bm}/P_{mm}$. If the gas were undisturbed, $x = 1$ and there is no suppression. If feedback smooths the gas out relative to the dark matter, $x < 1$ at small scales and the power drops.

So: **measure $x$, get the suppression.** That is the whole idea.

*Detail: §III.E.*

## II.2 What the three surveys actually give you

**The kSZ effect.** Free electrons moving with a bulk velocity Doppler-shift CMB photons that scatter off them. The temperature shift is proportional to the electron momentum along the line of sight, so stacking CMB temperature at galaxy positions, weighted by each galaxy's reconstructed line-of-sight velocity, gives the projected free-electron distribution around galaxies. This is the only observable that sees the gas directly, without assuming it is in hydrostatic equilibrium or emitting X-rays. It measures the galaxy-gas cross-correlation, $Y_{gb}$.

**Galaxy-galaxy lensing.** Background galaxy shapes are sheared by foreground mass. The tangential shear around lenses gives the excess surface density $\Delta\Sigma$, which is the galaxy-total matter cross-correlation, $Y_{gm}$. Note the field is total matter, not CDM.

**Galaxy clustering.** The projected correlation function of the lens sample itself, $w_p(r_p)$, filtered the same way, gives $Y_{gg}$.

**The $f_{\rm gas}$ paper** takes the kSZ stack and the lensing signal, applies one common aperture filter to both, and takes the bin-by-bin ratio. That ratio is a projected gas fraction, with no profile fitting anywhere. It is the existing measurement this programme builds on.

*Detail: §III.A.*

## II.3 Why you cannot just do the obvious thing

The obvious route to $x = P_{bm}/P_{mm}$ is to divide the kSZ stack by something that gives you $P_{mm}$. But every galaxy-crossed observable carries the galaxy bias $b$ and its scale dependence, and at arcminute scales the bias is not a constant, is not predictable from theory, and is not separable from the astrophysics you are trying to measure. Modelling it means importing a halo model, which reintroduces exactly the model dependence the programme is trying to avoid.

## II.4 The Singh trick

Singh et al. (2020) removed the bias from galaxy-galaxy lensing without modelling it. The move is to notice that if you have the galaxy-matter cross **and** the galaxy auto, the bias appears in both, and the combination

$$
r_{gm} \equiv \frac{Y_{gm}}{\sqrt{Y_{gg}\,Y_{mm}}}
$$

is bias-free by construction. All the residual non-linear stochasticity is compressed into that one coefficient, which is close to 1, is calibrated on mocks, and shifted their $S_8$ by only about $0.3\sigma$ even when set to 1 everywhere. The measured clustering becomes part of the model; theory supplies the matter auto; nothing else is modelled.

**This programme applies the same substitution to the kSZ side instead of the lensing side.** Doing so has two structural consequences that are worth stating plainly, because they are the reason the design is attractive:

1. The **baryon auto-correlation $Y_{bb}$ cancels algebraically.** Nothing measures the gas auto-correlation cleanly at these scales, and you never need to know it.
2. **Lensing drops out entirely**, so the measurement is no longer bottlenecked by the DESI $\cap$ HSC overlap. The footprint goes from 546 to 6,709 deg$^2$.

*Detail: §III.B.*

## II.5 Two routes

**Route A** is the original design (v0.1 note, Eq. 4). Feed it the kSZ stack, the galaxy clustering, and a theory prediction for the matter auto:

$$
x = \frac{r_{bm}}{r_{gb}} \cdot \frac{Y_{gb}}{\sqrt{Y_{gg}\,Y_{mm}}} .
$$

Big footprint, but it needs a $Y_{gg}$ measurement that does not exist yet, at angular scales below the DESI fibre patrol radius, plus a non-linear matter power prediction good to a few per cent.

**Route B** came out of Uroš's 2026-09-03 message and is in the v0.2 addendum (Eq. A11). Substitute the definition of $r_{gm}$ into Route A and the clustering and theory terms both disappear:

$$
x = \frac{r_{bm}\,r_{gm}}{r_{gb}} \cdot \frac{Y_{gb}}{Y_{gm}} .
$$

The observable is now the kSZ-to-lensing ratio, which the $f_{\rm gas}$ paper already measures. No new data pipeline, no theory spectrum, no DMO run. The cost is that the footprint reverts to 546 deg$^2$.

The two are algebraically the same statement, so they are not competitors. They fail differently, and their agreement where the footprints overlap is a null test of the $Y_{gg}$ and $Y_{mm}$ legs.

*Detail: §III.B.*

## II.6 The calibration factor, and when it is exactly 1

Both routes need a correction factor from simulations. Route B's is

$$
C \equiv \frac{r_{bm}\,r_{gm}}{r_{gb}} = \frac{Y_{bm}\,Y_{gm}}{Y_{mm}\,Y_{gb}} .
$$

Note the second form: written out, $C$ is just a ratio of four filtered amplitudes. Both auto-correlations, $Y_{bb}$ and $Y_{gg}$, cancel out. Every normalization convention cancels with them, including the kSZ temperature-to-optical-depth conversion and the projection depth. This is why $C$ is a far better-behaved object than the quantity it replaced.

**The condition for $C = 1$.** Suppose the gas field is any scale-dependent smoothed version of the matter field, plus noise that is uncorrelated with the galaxies:

$$
\delta_b = S(k)\,\delta_m + \epsilon, \qquad \langle \epsilon\,\delta_m\rangle = \langle \epsilon\,\delta_g\rangle = 0 .
$$

Then $C = 1$ identically, at every $k$, for **any** $S(k)$. Feedback can be arbitrarily strong and arbitrarily scale-dependent and $C$ does not move. The plain-language version: **$C = 1$ whenever the galaxies' correlation with the gas is entirely mediated by the matter field.**

So $C - 1$ is not a generic fudge factor. It measures one specific thing: the direct galaxy-gas connection that the matter field does not carry. Physically, that is the fact that the gas you are stacking on has been processed by the AGN of the very galaxies you are stacking on, and the total matter field does not record that history. It is a one-halo, centrally concentrated effect, which is why excising the halo centre should push $C$ toward 1.

**Measured:** $C$ runs from 1.11 at $1'$ to 1.00 at $6'$, with a mean $|C-1|$ of 0.052. So mediation is a good approximation but not an exact one, and the residual grows with feedback strength.

*Detail: §III.C.*

## II.7 Why there are four aperture filters

Everything is measured at an aperture $R$ rather than a wavenumber $k$, so a filter has to be chosen. Four are in play, and the differences between them matter because the coefficients deviate from 1 only at small scales, so a filter that reaches down to small scales drags that deviation into every aperture.

| filter | definition | property that matters |
|---|---|---|
| $\Sigma(R)$ | annulus mean over $[R, R+0.75']$ | depends only on scales $\ge R$, but **uncompensated**: its amplitude depends on the box size |
| $\Delta\Sigma(R)$ | disk mean minus annulus mean | compensated, but its disk mean integrates over **all** scales below $R$ |
| $\Upsilon(R;R_0)$ | $\Delta\Sigma(R) - (R_0/R)^2\Delta\Sigma(R_0)$ | nulls everything below $R_0$, and nulls a central point mass exactly |
| $Y(R;R_{\max})$ | $\Sigma(R) - \Sigma(R_{\max})$ | the Park et al. (2021) transform: local **and** compensated |

The uncompensated filter $\Sigma$ was frozen out early because an amplitude that depends on box size cannot be compared against a theory prediction. The v0.2 addendum added the Park et al. $Y$ transform because it is the only one with all three good properties at once, and predicted it would win.

**It did not.** At $z\approx0.5$ the $Y$ transform is beaten by both incumbents on $|C-1|$, and its best configuration references an aperture at $9'$, outside the data range, where the kSZ stack has no measurement and where large-scale CMB noise would be worst. The current recommendation is $\Delta\Sigma$ as fiducial with $\Upsilon$ retained as a cross-check, and the $Y$ transform dropped.

One caution about that comparison, raised in review and not yet acted on: the filters are compared at matched aperture, but at $R = 1'$ and $z = 0.5$, $\Delta\Sigma$ responds around $k \approx 5\,h/$Mpc while the $Y$ transform responds around $k \approx 1.5$. Comparing them at fixed $R$ therefore compares different physical scales.

*Detail: §III.D.*

## II.8 From an aperture profile to a power suppression

The measured quantity is $x(R)$, a filtered real-space ratio. The thing you want is $S(k)$, the power suppression. Three steps:

1. **Within a simulation the algebra is exact.** Because $\delta_t = f_m \delta_m + f_b \delta_b$ is a linear field identity and the filters are linear, the same relation holds for the filtered amplitudes: $Y_{tt}/Y_{mm} = (f_m + f_b x)^2 + f_b^2 (1-r_{bm}^2)\,Y_{bb}/Y_{mm}$, written out in §III.E. The second term is measured to be below $3\times10^{-3}$, so in practice $Y_{tt}/Y_{mm} = (f_m+f_b x)^2$.
2. **Getting from hydro to DMO** needs one assumption, that the hydrodynamic CDM auto-spectrum matches the gravity-only total-matter spectrum. That back-reaction is a known 1 to 2 per cent effect and is booked, not measured, because no DMO run is on disk.
3. **Getting from $R$ to $k$** needs the window formalism or a cross-suite regression. This is where the filter choice re-enters, since a narrow window distorts less. Measured: the window-smearing residual is $+0.8$ to $+1.7$ per cent for $\Delta\Sigma$ against $+1.9$ to $+2.9$ for $\Upsilon$ and the $Y$ transform.

The payoff of the whole design is the $f_b$ weighting. Since $S = (f_m + f_b x)^2$ with $f_b \approx 0.157$,

$$
\frac{\delta S}{S} = \frac{2 f_b}{f_m + f_b x}\,\delta x \approx 0.31\,\delta x .
$$

A 20 per cent calibration error gives 6 per cent on the suppression, and the measured 6 per cent gives 1.9 per cent. **The estimator tolerates calibration error that would be fatal in a direct measurement.**

*Detail: §III.E.*

## II.9 What the simulations found

Four runs survive: TNG300-1 plus three FLAMINGO variants (fiducial, fgas$-8\sigma$, Jet_fgas$-4\sigma$). SIMBA and Illustris-1 were dropped because their boxes yield only 500 and 210 SHAM galaxies, and their jackknife errors exceeded the signal.

**The old statistic was marginal.** $r_{bm}/r_{gb}$ came out at 0.53 rising to 0.85, a large transfer rather than a small correction, with cross-code scatter of 8 to 13 per cent straddling the 10 per cent viability threshold.

**The new statistic is not.** $C$ is close to 1, converges to 1 with aperture, has jackknife errors an order of magnitude below the cross-run scatter, and passes Gate A with room. Its cross-code disagreement is 1.3 per cent at $z\approx0.5$.

**Three findings that were not expected:**

- **Four runs cannot rank filters.** A scatter estimated from $N=4$ carries 41 per cent sampling uncertainty, so seven of eight filters are statistically tied. The only filter the measurement can disfavour is $\Upsilon(R_0 = 1')$. Any filter recommendation has to rest on something the measurement actually resolves.
- **$C$ tracks feedback strength**, which the mediation argument said it should not. Cross-feedback scatter exceeds cross-code scatter in 14 of 16 cells. The residual is astrophysical, not a sample-matching artefact.
- **The dominant "theory systematic" from the previous round was the signal.** Stage 8 of the earlier record flagged a 3 to 17 per cent "CDM versus total matter" term as the largest theory-side error. It is negative, scales with feedback strength, and matches $(f_m + f_b x)^2 - 1$ to under one per cent. It is the suppression itself. The planned CDM-only non-linear prescription work is cancelled.

*Detail: §III.F.*

## II.10 What is assumed, in plain terms

Grouped by how much they could hurt. Full ledger with sizes and provenance in §III.G.

**Exact or verified, not really assumptions.** The field decomposition $\delta_t = f_m\delta_m + f_b\delta_b$ holds exactly in the simulations because the CDM map is *defined* as total minus baryon (with $f_b$ taken from the maps, never from the header, since FLAMINGO's $\Omega_0$ carries a neutrino contribution the particle maps do not). The projection relation $P_{\rm 2D} = P_{\rm 3D}/L$ was measured, not trusted. The filter algebra was tested against the thing it replaces in every case.

**Booked with a known size.** Hydro-CDM versus DMO back-reaction, 1 to 2 per cent, adopted as 1. The $r_{bm}\approx1$ step in the suppression mapping, worth $3\times10^{-3}$. Pixelization of the $0.75'$ annulus, which sets a 1.7 to 2.2 per cent floor on any harmonic-space theory prediction. halofit accuracy, 4.6 to 6.7 per cent, Route A only. The calibration transfer itself, 4.7 to 9.6 per cent.

**Not yet tested at all.** The ACT beam, which is filter-dependent and has to be recomputed for whichever filter is adopted. The velocity-reconstruction suppression of the kSZ signal, 10 to 20 per cent and possibly scale-dependent, which enters $Y_{gb}$ with no cancellation now that lensing is gone. DESI fibre incompleteness at $1'$ to $6'$, which sits at or below the patrol radius. RSD and $\Pi_{\max}$ for the clustering leg. Bin-to-bin covariance, which 16 jackknife regions cannot support.

**Choices that could be wrong rather than merely imprecise.** The electron-versus-baryon target, currently frozen on electrons but now contradicted by the calibration scatter. Two code families rather than three or more. One projection axis. A snapshot mismatch, FLAMINGO at $z=0.30$ against TNG at $z=0.26$, in the lower-redshift sample. SHAM samples as stand-ins for DESI selection.

*Detail: §III.G.*

---

# Part III. Detail

## III.A Notation and conventions

**Fields.** $g$ galaxies (DESI BGS or LRG bin 1; SHAM analogues in simulations), $e$ ionized gas (what the kSZ actually sees), $b$ all baryons (ionized plus neutral gas plus stars), $m$ CDM, $t = m + b$ total matter.

**Two conventions for the matter leg**, which must never be mixed:

- **Convention C**, $m = $ CDM. Used by the v0.1 note and by Route A, because theory can supply $Y_{mm}$: the hydrodynamic CDM auto matches the gravity-only total-matter spectrum to 1 to 2 per cent, so halofit on a DMO cosmology is the right model for it.
- **Convention T**, $m \to t = $ total matter. Recommended for Route B, because lensing measures total matter and no theory spectrum enters at all. Measured to agree with Convention C to about one per cent, and it needs no new simulation product since every Convention T amplitude is a recombination of measured ones.

**Filtered amplitudes.** $Y_{\alpha\beta}(R;\mathcal{F}) = \langle \mathcal{F}_R[\delta_\alpha]\,\delta_\beta\rangle$, one number per aperture, filter and field pair. In simulations this is computed as a map-level average on a periodic box, which is mathematically identical to stamp-stacking a filtered map at galaxy positions and additionally reaches the field-field pairs no stamp stacker can produce. Every field enters as $\delta = X/\langle X\rangle - 1$, so all $Y$'s are dimensionless and all $r$'s are convention-free.

**Careful with the letter $Y$.** With field subscripts it is a filtered amplitude. With radial arguments and no subscripts, $Y(R;R_{\max})$, it is the Park et al. transform, one of the filter choices. Where both appear the filter is a superscript: $Y^{(\Upsilon)}_{gb}$, $C^{(Y)}$.

**Apertures.** Nine linear bins over $1'$ to $6'$, matching the $f_{\rm gas}$ pipeline, extended to $9.75'$ as a diagnostic and unioned with reference radii $\{2', 4', 5', 6', 9'\}$ for the Task 7 to 10 sweep, giving 19. Annulus width $\delta R = 0.75'$ throughout. Geometry: $1' = 0.383$ cMpc$/h$ at $z=0.5$, $0.243$ at $z=0.30$ (FLAMINGO) and $0.213$ at $z=0.26$ (TNG), so the range is roughly $0.4$ to $2.3$ cMpc$/h$ for the LRG-like sample and $0.21$ to $1.5$ cMpc$/h$ for the BGS-like one.

**Errors.** A $4\times4$ block jackknife, 16 regions, with every derived quantity formed **per realization**, never by Gaussian propagation of marginal errors on the amplitudes. Sixteen regions cannot support a bin-to-bin covariance (Hartlap needs more resamplings than bins plus two), so per-radius errors only.

**Filter-definition caveat.** Singh et al.'s $\Delta\Sigma$ uses the local $\Sigma(R)$; this programme uses the annulus mean over $[R, R+\delta R]$, matching the kSZ pipeline. They differ by a profile-gradient term that is not negligible in the inner bins. The Singh formulas are the template, not the specification, and any reconstruction of $\Sigma$ from $\Delta\Sigma$ must respect which convention it was built on.

## III.B The estimator algebra

Define at each aperture and for each filter

$$
r_{gb} = \frac{Y_{gb}}{\sqrt{Y_{gg}Y_{bb}}}, \qquad r_{bm} = \frac{Y_{bm}}{\sqrt{Y_{bb}Y_{mm}}}, \qquad r_{gm} = \frac{Y_{gm}}{\sqrt{Y_{gg}Y_{mm}}} .
$$

Solving the first for $Y_{bb}$ and substituting into the second gives **Route A**:

$$
\frac{Y_{bm}}{Y_{mm}} = \frac{r_{bm}}{r_{gb}}\cdot\frac{Y_{gb}}{\sqrt{Y_{gg}\,Y_{mm}}} .
$$

Substituting $Y_{mm}^{1/2} = Y_{gm}/(r_{gm}Y_{gg}^{1/2})$ into that gives **Route B**:

$$
\frac{Y_{bm}}{Y_{mm}} = \frac{r_{bm}\,r_{gm}}{r_{gb}}\cdot\frac{Y_{gb}}{Y_{gm}} .
$$

Written out, the two calibration factors are

$$
C = \frac{r_{bm}r_{gm}}{r_{gb}} = \frac{Y_{bm}Y_{gm}}{Y_{mm}Y_{gb}}, \qquad
C_A = \frac{r_{bm}}{r_{gb}} = \frac{Y_{bm}\sqrt{Y_{gg}}}{Y_{gb}\sqrt{Y_{mm}}} .
$$

$Y_{bb}$ cancels from both; $Y_{gg}$ cancels from $C$ as well. Two practical consequences: $C$ is finite in every bin of every filter, whereas $C_A$ is undefined in one $\Upsilon(R_0=2')$ bin where $Y_{gg}$ goes negative (a compensated filter against a shot-noise-subtracted galaxy auto); and $C$ is insensitive to every normalization convention, which is why the projection-depth study found the coefficients stable to $4\times10^{-4}$ while the amplitudes scaled as $1/L$.

**A note on the individual $r$'s.** They are not bounded by 1. Cauchy-Schwarz applies mode by mode in Fourier space, but a filtered amplitude integrates the spectrum against a sign-changing kernel, so the denominator can be small. Measured $r_{gb}$ reaches 2.1, and Singh et al. Fig. 1 likewise shows $r_{cc}^{(\Upsilon)}\sim1.3$. Only the four-amplitude combination $C$ has a convention-free meaning, which is why it, and not the individual coefficients, is the object to report.

## III.C The mediation proposition

**Statement.** If $\delta_b(\mathbf{k}) = S(k)\,\delta_m(\mathbf{k}) + \epsilon(\mathbf{k})$ with $\langle\epsilon\delta_m\rangle = \langle\epsilon\delta_g\rangle = 0$, then $P_{bm} = S P_{mm}$ and $P_{gb} = S P_{gm}$, so

$$
C(k) = \frac{P_{bm}P_{gm}}{P_{mm}P_{gb}} = \frac{S P_{mm}\cdot P_{gm}}{P_{mm}\cdot S P_{gm}} = 1
$$

identically, for any $S(k)$. Equivalently, mediation is the statement $r_{gb} = r_{gm}\,r_{bm}$.

**Two caveats.**

- For *filtered* amplitudes, $C = 1$ additionally requires that $S(k)$ and $b(k) \equiv P_{gm}/P_{mm}$ do not both vary across the kernel's support. The residual is a window-weighted covariance of the two, and it shrinks as the kernel narrows.
- The proposition says what to look for, not how big the violation is.

**Measured.** $|C-1|$ is 0.052 on average for $\Delta\Sigma$ at $z\approx0.5$, converging to $\le 0.03$ by $6'$. Within FLAMINGO the deviation is monotone in feedback strength at essentially every aperture: fiducial 1.105, Jet_fgas$-4\sigma$ 1.138, fgas$-8\sigma$ 1.166 at $R=1'$. That ordering is the clean physical signal, and it is the direct galaxy-gas stochasticity the proposition identifies as the residual: AGN in the stacked haloes process the gas around those particular galaxies in a way the total matter field does not record.

**One framing to drop.** The v0.2 addendum proposed measuring $r_{gm}$ and checking whether it equals $r_{gb}/r_{bm}$ as a test of mediation. Since $C \equiv r_{bm}r_{gm}/r_{gb}$, that check *is* $C = 1$, computed from the same four amplitudes on the same maps. It is redundant rather than vacuous ($C \ne 1$ does falsify mediation), but it is not an independent test, and the value of $C$ was always the deliverable.

## III.D Filters and their $k$-space response

**Definitions**, with $\Sigma(R)$ the annulus mean over $[R, R+\delta R]$ and $\delta R = 0.75'$:

$$
\Delta\Sigma(R) = \bar\Sigma(0,R) - \Sigma(R), \qquad
\Upsilon(R;R_0) = \Delta\Sigma(R) - \frac{R_0^2}{R^2}\Delta\Sigma(R_0), \qquad
Y(R;R_{\max}) = \Sigma(R) - \Sigma(R_{\max}) .
$$

**Real-space support.** $\Sigma(R)$ depends on $\xi(r)$ only for $r \ge R$, because $\Sigma(R) = \bar\rho\int d\Pi\,\xi(\sqrt{R^2+\Pi^2})$. $\Delta\Sigma$ depends on all $r$ through the disk mean. $\Upsilon$ depends on $r \ge R_0$. The $Y$ transform's $R$-dependent part depends only on $r \ge R$.

**Compensation.** With $\hat W_{\rm ann}(k;R_1,R_2) = 2[R_2J_1(kR_2)-R_1J_1(kR_1)]/[k(R_2^2-R_1^2)] \to 1$ as $k\to0$, the annulus mean is uncompensated and integrates power down to the fundamental mode of whatever volume it is measured in. That is why $\Sigma$ amplitudes do not port between boxes: removing every mode longer than TNG300-1's box shifts $Y_\Sigma$ by 8.7 per cent against 0.014 per cent for the compensated filters. It largely cancels in the coefficients (0.6 per cent), so the disqualification rests on the amplitudes, which the theory transfer needs.

**Where each filter lives in $k$.** For a power-law projected spectrum, the wavenumbers carrying the central 90 per cent of the response, converted at $z=0.5$:

| filter | $R$ | $k_{50}$ [$h/$Mpc] | width $k_{95}/k_{05}$ |
|---|---|---|---|
| $\Sigma$ | $1'$ | 0.97 | 21.2 |
| $\Delta\Sigma$ | $1'$ | 5.07 | 3.36 |
| $\Delta\Sigma$ | $3'$ | 1.99 | 3.25 |
| $\Upsilon(1')$ | $3'$ | 1.65 | 3.02 |
| $Y(5')$ | $1'$ | 1.45 | 3.29 |

Two readings. $\Sigma$ is six to seven times broader in $\ln k$ than any compensated filter, and its median wavenumber moves by a factor 1.8 when the assumed spectral slope changes from $-1$ to $-1.5$, which is the same instability the box-cut test found. And at fixed aperture the filters probe wavenumbers differing by up to a factor of five, so a filter comparison at matched $R$ is partly a relabelling of scales.

**Why $\Upsilon$ fails to converge.** $C^{(\Upsilon)}$ plateaus at 0.885 to 0.940 at $9.75'$ where $\Delta\Sigma$ reaches 0.990 to 1.014. This is not a bug: $\Delta\Sigma(1')$ exceeds $\Delta\Sigma(9.75')$ by roughly two orders of magnitude, so even though $(R_0/R)^2 \approx 0.01$ there, the reference term stays order unity at every aperture and $\Upsilon$ never converges to $\Delta\Sigma$. Raised in review and not yet acted on: that ratio implies a log-slope near $-2$, and since $\Upsilon$ subtracts a mode scaling as $R^{-2}$, at that slope the subtraction removes an order-unity fraction of the signal at every radius. If confirmed by directly measuring the slope, that is a stronger argument than non-convergence, and it would say that any estimator projecting out the $1/R^2$ mode removes most of the signal in this aperture range, unlike the regime Prat et al. (2023) analysed.

**Why the $Y$ transform was dropped.** It has the best structural properties, being both local and compensated, but at $z\approx0.5$ it loses to both incumbents on $|C-1|$, and its best configuration ($R_{\max}=9'$) references an aperture outside the data range where the uncompensated large-scale noise it is meant to avoid would be worst. Raised in review: the $|C-1|$ comparison is not on matched aperture ranges, since $\Delta\Sigma$'s 0.052 averages over twelve bins to $6'$ while $Y(5')$'s 0.122 averages over six bins stopping at $3.5'$ (the mask is $R \ge 0.8R_{\max}$), exactly where $|C-1|$ is largest for every filter. Recomputing $\Delta\Sigma$ on the same six bins gives about 0.08, so the gap is a factor of roughly 1.5 rather than 2.3. The ranking survives; the margin does not.

## III.E Suppression mapping

**Exact, no assumptions:**

$$
P_{tt} = f_m^2P_{mm} + 2f_mf_bP_{bm} + f_b^2P_{bb}, \qquad\text{hence}\qquad
\frac{P_{tt}}{P_{mm}} = (f_m + f_bx)^2 + f_b^2(1-r_{bm}^2)\frac{P_{bb}}{P_{mm}} .
$$

Both relations transfer exactly to the filtered amplitudes, because the field decomposition and the filters are both linear. The stochastic second term is measured across all filters, runs and redshifts at worst $2.9\times10^{-3}$, which enters $S$ at the 0.3 per cent level. (The v0.2 addendum quoted a bound of $6\times10^{-4}$; that was arithmetic done at $r_{bm}=0.95$ under a stated assumption of $r_{bm}\ge0.89$, and is wrong by a factor of two on its own inputs before the measurement is considered.)

**Suppression**, adopting $P_{mm}^{\rm hydro} = P_{tt}^{\rm DMO}$:

$$
S = (f_m + f_bx)^2 \quad\text{(Convention C)}, \qquad
S = \frac{f_m^2}{(1-f_bx_t)^2} \quad\text{(Convention T, } x_t = P_{bt}/P_{tt}) .
$$

The two agree: $x = 0.5$ gives $S = 0.849$ either way.

**Error propagation.** $\delta S/S = 2f_b\,\delta x/(f_m+f_bx) \approx 0.31\,\delta x$, so 6 per cent on $C$ gives 1.9 per cent on $S$ and 0.9 per cent on $T = \sqrt{S}$; 20 per cent gives 6 per cent and 3 per cent.

**The four-rung validation ladder**, kept separate so a failure is diagnosable, in the spirit of the A/B/C decomposition that found the CAMB bug in the previous round:

1. Identity check: both sides of Route B from the same maps. Tests code only.
2. Kernel check: the map-measured ratio against the same ratio predicted by pushing the measured 2D spectra through the analytic kernel. Holds at 1.7 to 2.2 per cent, limited by pixelization.
3. Localization check: the filtered ratio against $x(k)$ at $k_{50}(R;\mathcal{F})$. This is the window smearing, measured at $+0.8$ to $+1.7$ per cent for $\Delta\Sigma$ against $+1.9$ to $+2.9$ for $\Upsilon$ and the $Y$ transform.
4. Suppression check: $S$ from the measured $x$ against the directly measured $P_{tt}/P_{mm}$ in the same box. Closes without a DMO run; only the final step to $P_{tt}^{\rm DMO}$ needs one.

## III.F The simulation programme and what it found

**Runs.** TNG300-1 (205 cMpc$/h$ box, 4,307 SHAM galaxies at the LRG-like density) plus FLAMINGO L1_m9 fiducial, fgas$-8\sigma$ and Jet_fgas$-4\sigma$ (681 cMpc$/h$, 157,910). SIMBA (500 galaxies) and Illustris-1 (210) were dropped in the first round: their jackknife errors exceeded the cross-code scatter, so a naive four-code Gate A read as a 3 to 55 per cent failure that was sample size, not physics. Samples follow the $f_{\rm gas}$ paper: subhalos ranked by stellar mass, parent FoF mass $\le 5\times10^{14}\,M_\odot/h$, target density $5\times10^{-4}$ (cMpc$/h)^{-3}$ for LRG-like and $1\times10^{-3}$ for BGS-like. One projection, `yz`, is cached.

**Round one (Tasks 1 to 4).** $r_{bm} = 0.89$ to 1.00 with errors $\le0.008$, so the near-unity expectation holds for it. $r_{gb}$ does not: 1.25 to 2.1. The ratio $r_{bm}/r_{gb}$ is 0.53 rising to 0.85, a large transfer, with cross-code scatter of 8 to 13 per cent (raw pairwise differences $\sqrt2$ larger) straddling the 10 per cent threshold. Gate A left open. Gate B closed on filter set $\{\Delta\Sigma, \Upsilon(1')\}$ and target $P_{em}/P_{mm}$. The theory chain was validated in three separable pieces, transfer 1.7 to 2.2 per cent, CDM-versus-total 3 to 17 per cent, halofit 4.6 to 6.7 per cent, and the decomposition is what made a CAMB redshift-ordering bug findable.

**Round two (Tasks 7 to 10).** $C$ for $\Delta\Sigma$ at $z\approx0.5$:

| $R$ | TNG300-1 | L1_m9 fid | fgas$-8\sigma$ | Jet_fgas$-4\sigma$ | scatter |
|---|---|---|---|---|---|
| $1.000'$ | $1.114\pm0.006$ | $1.105\pm0.001$ | $1.166\pm0.001$ | $1.138\pm0.002$ | 2.4% |
| $3.500'$ | $1.020\pm0.005$ | $1.010\pm0.001$ | $1.080\pm0.002$ | $1.066\pm0.002$ | 3.3% |
| $6.000'$ | $1.001\pm0.006$ | $0.993\pm0.002$ | $1.025\pm0.002$ | $1.030\pm0.003$ | 1.8% |

Jackknife errors are an order of magnitude below the cross-run scatter, so unlike the previous round the scatter is a real physical difference between runs rather than noise.

**Headline numbers.**

| quantity | value |
|---|---|
| $C$, $\Delta\Sigma$, $1'$ to $6'$, $z\approx0.5$ | $1.11 \to 1.00$, mean $\vert C-1\vert = 0.052$, scatter 4.7% |
| same at $z\approx0.26$ | mean 0.100, scatter 5.9% |
| cross-code, $r_{bm}/r_{gb}$ against $C$ | 10.9% $\to$ **1.3%** at $z\approx0.5$; 9.7% at $z\approx0.26$ |
| cross-feedback | 9.4% ($\Delta\Sigma$, $z\approx0.5$) |
| Route B uncorrected, ratio to truth | 0.92 to 0.98 |
| Route A uncorrected, ratio to truth | 1.39 to 1.54 |
| window smearing | $+0.8$ to $+1.7$% ($\Delta\Sigma$) against $+1.9$ to $+2.9$% ($\Upsilon$, $Y$) |
| electron against baryon target, scatter | 0.095 against 0.047 |
| $Y_{gb}/Y_{ge}$ at $1'$ | 1.21 (TNG) to 2.27 (fgas$-8\sigma$) |
| Gate A propagation | 6% on $C$ $\to$ **1.9% on $S$**, 0.9% on $T$ |

**The five advance predictions, scored.** Stated in the v0.2 addendum before the runs, so the exercise was a test rather than a fit.

| # | prediction | verdict |
|---|---|---|
| 1 | $Y$ transform beats $\Upsilon$ on $\vert C-1\vert$ | falsified at $z\approx0.5$; reverses at $z\approx0.26$ |
| 2 | $Y$ and $\Upsilon$ beat $\Delta\Sigma$ | falsified on $\vert C-1\vert$; untestable on scatter |
| 3 | $\Sigma$'s advantage shrinks at matched $k_{50}$ | machinery now exists, comparison not made |
| 4 | $C\to1$ at large $R$ for every filter | holds for $\Delta\Sigma$, $\Sigma$, $Y$; fails for $\Upsilon$ |
| 5 | $C$ more stable across feedback than across codes | falsified, backwards, in 14 of 16 cells |

**Statistical power.** A standard deviation from $N=4$ carries $\sigma/\sqrt{2(N-1)} = 41$ per cent relative uncertainty. Seven of eight filters are within $1\sigma$ of the best at both redshifts. Applied consistently to the Gate A threshold as well, which the record does not do: $\Delta\Sigma$ at 4.7 per cent is $[2.8, 6.6]$ per cent and passes cleanly, while $\Upsilon(1')$ at $z\approx0.26$ at 9.6 per cent is $[5.7, 13.5]$ per cent and is undetermined against a 10 per cent threshold. Note also that the pooled four-run scatter mixes one code with three feedback variants of a second, so the runs are not exchangeable draws; the two-axis decomposition into cross-code and cross-feedback is the sounder primary statistic.

**Gate status.**

- **Gate A: passes**, fixed-transfer route, on two code families. Limited by having four samples of a population rather than by measurement noise, which is a different and better limitation than the previous round's.
- **Gate B: reopened** on the field definition. Filter set recommended as $\{\Delta\Sigma\}$ fiducial with $\Upsilon(R_0=2')$ retained, $Y$ transform dropped. Field definition unresolved: two code families cannot adjudicate it.
- **Gate C: not reached.** Belongs to Phase 6, after the data-side $Y_{gg}$ exists.

## III.G The complete assumptions ledger

### Exact by construction or verified by measurement

| assumption | status |
|---|---|
| $\delta_t = f_m\delta_m + f_b\delta_b$ | exact in simulations: the CDM map is *defined* as total minus baryon. Verified to $10^{-10}$ on real caches; a deliberate 5 per cent perturbation of $f_b$ breaks it, so the test has teeth |
| $f_b$ from the maps, not the header | required: `OmegaBaryon/Omega0` is wrong at the per-cent level for FLAMINGO, whose $\Omega_0$ carries a neutrino contribution absent from the particle maps |
| $P_{\rm 2D} = P_{\rm 3D}/L$ for a fully projected periodic box | measured at 0.98 to 1.05 over $k = 0.1$ to $6\,h/$Mpc, not assumed |
| map-level average equals stamp-stacking on a periodic box | tested against the existing stamp stacker on real data |
| $Y$ transform as a linear combination of $\Sigma$ amplitudes | tested against a directly built $\Sigma(R)-\Sigma(R_{\max})$ kernel |
| $\Upsilon$ rebuilt at arbitrary $R_0$ | tested against the pipeline's own $\Upsilon$ |
| projection depth does not affect coefficients | amplitudes scale as $1/L$; coefficients stable to $4\times10^{-4}$ ($\Delta\Sigma$) over a factor of eight in depth |

### Adopted with a known size

| assumption | size | note |
|---|---|---|
| $P_{mm}^{\rm hydro\,CDM}/P_{mm}^{\rm DMO} = 1$ | 1 to 2% | no DMO run on disk; over a day to download. Known wrong, booked |
| $r_{bm}\approx1$ in the suppression mapping | $2.9\times10^{-3}$ on $P_{tt}/P_{mm}$ | measured across all filters, runs and redshifts |
| pixelization of the $0.75'$ annulus | 1.7 to 2.2% | the annulus spans 3.75 pixels at production resolution; this is the floor on any harmonic-space theory prediction against this measurement |
| halofit non-linear accuracy | 4.6 to 6.7% | Route A only; Route B needs no theory spectrum |
| the calibration transfer $C$ itself | 4.7 to 9.6% | propagates to 1.5 to 3.0% on $S$ |
| discretization and resolution | $\le2.4$% on a coefficient from a $2\times$ resolution change; $\le1.5$% from the pixel-scale convention | |
| aperture boundary ties | $\sim6$% on an amplitude, $\sim0.6$% on a coefficient, at $R = 1'$, $2.25'$, $6'$ only | lattice arithmetic predicts exactly those radii |
| self-pair subtraction in $Y_{gg}$ | removes $6.9\times$ the retained signal at $1'$, $1.9\times$ at $6'$ | so $r_{gb}$ at small $R$ is a difference of comparable numbers. Mirrors the data-side caveat |

### Not yet tested

| gap | why it matters |
|---|---|
| ACT beam per filter | $C_{\rm beam}$ is filter-dependent and does not transfer between filters. $C$ as measured is an intrinsic field property with no beam anywhere |
| velocity-reconstruction suppression | 10 to 20 per cent on $Y_{gb}$, weakly feedback-dependent (Ondaro-Mallea et al. 2026). With lensing gone from Route A there is no partial cancellation. Whether it is scale-dependent over $1'$ to $6'$ is unresolved |
| DESI fibre incompleteness | $1'$ to $6'$ sits at or below the fibre patrol scale. Enters Route A as $Y_{gg}^{-1/2}$ and mimics a scale-dependent $r$ |
| RSD and $\Pi_{\max}$ for $Y_{gg}$ | Singh et al. found it negligible at their scales; ours are smaller and it is unchecked. The projection-depth study is the simulation-side proxy |
| bin-to-bin covariance | 16 jackknife regions cannot support one. $\Upsilon$ and the $Y$ transform correlate bins by construction more than $\Delta\Sigma$ does |
| box-scale compensation test for the $Y$ transform | cheap, simply not run |
| $\Sigma$ against $\Delta\Sigma$ at matched $k_{50}$ | prediction 3, unscored |
| reconstruction of $\Sigma$ from $\Delta\Sigma$ for a lensing leg | needed only for Route B's lensing path; unvalidated at the discretization level |

### Choices that could be wrong rather than imprecise

| choice | current state |
|---|---|
| electron ($e$) versus baryon ($b$) target | frozen on $P_{em}/P_{mm}$ in round one, because that is what the kSZ measures and it needs no transfer. Round two contradicts it: the electron target carries twice the cross-run scatter for $\Delta\Sigma$. Reopened, unresolved |
| filter set | $\{\Delta\Sigma\}$ fiducial, $\Upsilon(2')$ retained, $Y$ transform dropped. But four runs cannot rank filters, so this rests on bin count, the $C\to1$ limit and the out-of-range $R_{\max}$, not on the scatter |
| Convention C or T | C for Route A, T recommended for Route B. They agree to about one per cent |
| two code families | TNG and FLAMINGO. Singh et al.'s Appendix A found that priors from one mock family biased an independent family at the 2 to 3 per cent level, and that discipline cannot be exercised with two |
| one projection axis | only `yz` is cached; across-projection scatter unmeasured |
| snapshot mismatch | FLAMINGO $z=0.30$ against TNG $z=0.26$ for the BGS-like sample. One of the two candidate explanations for the cross-code degradation at that redshift |
| SHAM samples | stand-ins for DESI selection, matched on number density, not on clustering or on $r_{gm}$ |
| stellar and neutral gas | only the aggregate $b - e$ split is known; isolating stars needs new particle sweeps. FLAMINGO carries roughly twice TNG's stellar mass for these samples |

## III.H Open items, in priority order

Merging the record's next steps with points raised in review of the v0.3 document. Items marked **[review]** are proposals not yet in the record.

1. **Cancel the CDM-only non-linear spectrum work.** Term B is the signal, confirmed by direct measurement. The correct Convention C chain is halofit(DMO) $\to Y_{mm}^{\rm hydro\,CDM}$ with only the 1 to 2 per cent back-reaction.
2. **[review] Regress $C$ against $x$ across the four runs.** Since $C$ tracks feedback strength and so does $x$, a tight $C(x)$ relation would replace the 9.4 per cent cross-feedback prior with the residual scatter about the relation, and it converges in one Newton step given $C \approx 1$. Cross-feedback is currently the dominant term in the calibration budget. Everything needed is already in the output files, so this is post-processing.
3. **[review] Test mediation independently.** Mediation requires $Y_{gb}/Y_{gm}$ to be identical for two different galaxy samples at the same snapshot and aperture, since both equal $S(k)$. That involves neither $Y_{mm}$ nor $Y_{bm}$, so it is new information, and it runs on observables alone, making it a data-side null test of the assumption the calibration rests on. Splitting by halo mass would sharpen it.
4. **[review] Separate redshift from sample.** Recompute $C$ at $z\approx0.5$ with the BGS-like number density. This distinguishes SHAM matching from the snapshot mismatch in the 1.3 versus 9.7 per cent cross-code split and in the $|C-1|$ reversal between redshifts, and it is far cheaper and more decisive than adding a third code.
5. **Reopen the Stage 6 electron/baryon decision** with the round-two scatter numbers. **[review]** A physical reading worth adding: the extra scatter on the electron target is most likely the codes' subgrid partition of baryons into stars, cold gas and ionized gas, which differs far more between TNG and FLAMINGO than the total baryon distribution does. That favours the baryon target for a stronger reason than scatter alone, since the $e \to b$ step is separately constrainable through the stellar mass function whereas the code dependence of the electron field is not.
6. **Settle the filter set**, which needs either a third code family or the matched-$k_{50}$ comparison. **[review]** Also compare $|C-1|$ on matched aperture ranges, and measure the $\Delta\Sigma$ log-slope directly to settle whether $\Upsilon$ is a near-cancellation at this slope. If it is, retaining $\Upsilon$ as a point-mass cross-check is the wrong call, since you cannot null the point mass without nulling a signal nearly parallel to it; a forward-modelled central stellar component is the right check instead.
7. **Book the feedback dependence of $C$** as a prior width rather than a sample-matching artefact, unless item 2 removes the need.
8. **Phase 4 onward, unchanged:** the data-side $Y_{gg}$ pair-count pipeline, the fibre-incompleteness campaign, and the Ondaro-Mallea velocity-reconstruction question.

## III.I Provenance

| document | authoritative for | superseded on |
|---|---|---|
| `cross_correlation_notes.md` (v0.1) | the estimator, the physics, Tasks 1 to 6, Phases 0 to 7, Gates A to C | point-mass marginalization (retired); the $6\times10^{-4}$ stochastic bound; the "near-unity expectation" for $r$, which does not apply to compensated filters |
| `tasks_1_to_4_record.md` | the round-one implementation, the theory chain, Gate B round one | its next step 1 (term B), cancelled; its Gate A verdict, superseded |
| `cross_correlation_notes_v0.2_addendum.md` | Route B, the calibration factor $C$, the mediation proposition, the filter analysis, Tasks 7 to 10, the five predictions | three of five predictions falsified; the circular mediation test; the stochastic bound; $R_{\max}=5'$ as fiducial |
| `cross_correlation_notes_v0.3_response.md` | the scientific conclusions of round two | Task 9, which it reports as unattempted but which was completed afterwards |
| `tasks_7_to_10_record.md` | the round-two implementation, all round-two numbers, the gates as they now stand | current |

**Code.** `src/rprofiles.py` (amplitudes, kernels, jackknife, SHAM maps), `src/kernels.py` (analytic harmonic kernels), `src/theory.py` (halofit to $Y$). Round two added `scripts/cross_corr/make_calibration_factor.py`, `make_task9_spectra.py`, `plot_task9.py` and `tests/test_calibration_factor.py`, with no `src/` file modified, so `data/r_profiles/*.npz` stays reproducible bit for bit. Commits `ec8f129` and `82526e3`.

**A lesson worth carrying.** Two of the three code defects found at the round-two commit gate lived in the glue, the functions deciding which quantity feeds which formula, not in the algebra, which had three dedicated identity tests and passed all of them. Test the wiring, not only the mathematics.
