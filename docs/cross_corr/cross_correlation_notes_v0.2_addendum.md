# Localizing Filters and the Direct-Ratio Estimator

**Addendum v0.2 to `cross_correlation_notes.md`**
R. Henry Liu (with U. Seljak), September 2026

This addendum supersedes Sections 2.2, 3 and parts of 4 and 6 of the v0.1 note. It responds to four messages from Uroš (2026-09-02 11:16, 2026-09-03 03:44, 2026-09-03 05:00) and to the Task 1-4 results recorded in `tasks_1_to_4_record.md`. Section numbering continues from the parent note where possible; new tasks are numbered 7 onward.

---

## 0. What changed

1. **Point-mass marginalization is retired** (Task 2b). Prat et al. (2023) show that point-mass marginalization with an infinite prior, mode projection, and $\Upsilon$ are equivalent, so reconstructing $\Sigma$ with a point-mass nuisance buys nothing that $\Upsilon$ does not already deliver. Section 3 below records what that paper actually demonstrates, which is narrower than the headline.
2. **A fourth filter enters the programme**: the $Y$ transform of Park et al. (2021), $Y(R;R_{\max}) = \Sigma(R) - \Sigma(R_{\max})$. It is compensated (unlike $\Sigma$), fully local from below (unlike $\Delta\Sigma$), and references the *largest* aperture rather than the smallest (unlike $\Upsilon$). Section 1-2.
3. **A second estimator route** connects directly to the quantity the $f_{\rm gas}$ paper already measures: $Y_{bm}/Y_{mm} = (r_{bm} r_{gm}/r_{gb})\,Y_{gb}/Y_{gm}$. It needs no clustering measurement and no theory $Y_{mm}$. Section 4.
4. **The calibration factor has an exact interpretation.** In both routes the simulation-calibrated factor is a ratio of four filtered amplitudes, and it equals unity identically whenever the galaxy-gas correlation is entirely mediated by the matter field. Deviations from unity measure direct galaxy-gas stochasticity, which is a one-halo, centrally concentrated effect. This is the theoretical content behind Uroš's expectation that the factor is "close to 1, especially once we excise the central part". Section 4.4.
5. **The mapping to power suppression is tightened.** The linearized Eq. (7) of v0.1 is replaced by exact expressions; the residual stochastic term is shown to be below $10^{-3}$. Section 5.

Two corrections to v0.1 are recorded in passing: the near-unity expectation for $r$ does not apply to compensated filters (Section 1.5), and the "CDM versus total matter" term $B$ identified in Stage 8 of the task record is probably the signal rather than a systematic (Section 4.6).

---

## 1. The filter family

**Notation: the letter $Y$ carries two unrelated meanings in this programme, and the second is new here.**

| written as | means |
|---|---|
| $Y_{\alpha\beta}(R;\mathcal{F})$, with **field subscripts** $\alpha,\beta \in \{g,b,e,m,t\}$ | the **filtered two-point amplitude** of v0.1 Section 2.3, $\langle\mathcal{F}_R[\delta_\alpha]\,\delta_\beta\rangle$: one number per aperture, filter and field pair. $Y_{gb}$ is the kSZ stack, $Y_{gm}$ the lensing signal, $Y_{gg}$ the clustering, $Y_{mm}$ the theory term. |
| $Y(R;R_{\max})$, with **radial arguments and no subscripts** | the **Park et al. (2021) $Y$ transform**, one of the four available choices of the filter $\mathcal{F}$, defined in (A3) below. |

So $\Upsilon$ and the Park et al. $Y$ transform are *filters*, on the same footing as $\Sigma$ and $\Delta\Sigma$, while $Y_{gb}$, $Y_{gm}$, $Y_{gg}$, $Y_{mm}$ and $Y_{bm}$ are *measured or predicted numbers* that never refer to a filter. Where both appear at once the filter is a superscript, as in $Y^{(\Upsilon)}_{gb}$ or $C^{(Y)}$. In prose the transform is called "the Park et al. $Y$ transform" on first use and "the $Y$ transform" thereafter, never a bare $Y$. Note also that v0.1 Section 2.3 writes the generic amplitude as $Y_{XY}$, whose second index collides with the same letter; $Y_{\alpha\beta}$ is used here instead.

### 1.1 Definitions

Throughout, $\Sigma(R)$ denotes the pipeline's annulus mean $\bar\Sigma(R, R+\delta R)$ with $\delta R = 0.75'$, matching the $f_{\rm gas}$ convention; where the distinction from the local value matters it is flagged explicitly (Section 2.4 shows it matters a great deal). The four filters are

$$
\Sigma(R) \equiv \bar\Sigma(R, R+\delta R), \qquad \bar\Sigma(R_1,R_2) = \frac{2}{R_2^2-R_1^2}\int_{R_1}^{R_2}\Sigma(R')\,R'\,dR' ,
$$

$$
\Delta\Sigma(R) = \bar\Sigma(0,R) - \Sigma(R), \tag{A1}
$$

$$
\Upsilon(R;R_0) = \Delta\Sigma(R) - \frac{R_0^2}{R^2}\,\Delta\Sigma(R_0), \tag{A2}
$$

$$
Y(R;R_{\max}) = \Sigma(R) - \Sigma(R_{\max}). \tag{A3}
$$

$\Upsilon$ is the Baldauf et al. (2010) annular differential surface density; $Y(R;R_{\max})$ is the Park et al. (2021) localizing transform, written here in the form Prat et al. (2023) give as their Eq. (12). Fiducial choices: $R_0 = 1'$, $R_{\max} = 5'$ (Uroš's suggestion, and see Section 8.2).

### 1.2 Which scales each filter sees

The three-dimensional correlation function enters $\Sigma(R)$ only through $r \geq R$, since $\Sigma(R) = \bar\rho\int d\Pi\,\xi(\sqrt{R^2+\Pi^2})$. This gives a clean ordering of real-space support:

| filter | depends on $\xi(r)$ for | nulls the $1/R^2$ mode | compensated, $\hat W(k\to 0) \to 0$ |
|---|---|---|---|
| $\Sigma$ | $r \geq R$ | yes (trivially: a central point mass contributes nothing at $R>0$) | **no** |
| $\Delta\Sigma$ | all $r$ | no | yes |
| $\Upsilon(R;R_0)$ | $r \geq R_0$ | yes | yes |
| $Y(R;R_{\max})$ | $r \geq \min(R, R_{\max})$, and the $R$-dependent part only $r \geq R$ | yes | **yes** |

The Park et al. $Y$ transform is the only filter in the set with all three desirable properties. This is the structural reason to test it, and it resolves the Gate B tension directly: $\Sigma$ was frozen out in Stage 5 because it is uncompensated, so its amplitudes do not port between volumes and it readmits large-scale CMB and atmospheric noise on the kSZ side. The $Y$ transform is a difference of two annulus means, so the uncompensated $k \to 0$ response cancels identically while the localization that made $\Sigma$ attractive is retained.

### 1.3 Harmonic kernels

With the generic filtered amplitude $Y_{\alpha\beta}(R) = \int \frac{k\,dk}{2\pi} P^{\alpha\beta}_{\rm 2D}(k)\,\hat W(k;R)$ and $\hat W_{\rm ann}(k;R_1,R_2) = 2[R_2 J_1(kR_2) - R_1 J_1(kR_1)]/[k(R_2^2-R_1^2)]$,

$$
\hat W_{\Sigma}(k;R) = \hat W_{\rm ann}(k;R,R+\delta R), \qquad \hat W_{Y}(k;R) = \hat W_{\Sigma}(k;R) - \hat W_{\Sigma}(k;R_{\max}). \tag{A4}
$$

Expanding $\hat W_{\rm ann} = 1 - k^2(R_1^2+R_2^2)/8 + O(k^4)$ gives the leading low-$k$ response of each compensated filter:

$$
\begin{aligned}
\hat W_{\Delta\Sigma} &\simeq \frac{(R+\delta R)^2}{8}\,k^2, \\
\hat W_{\Upsilon} &\simeq \frac{(R+\delta R)^2 - (R_0/R)^2(R_0+\delta R)^2}{8}\,k^2, \\
\hat W_{Y} &\simeq \frac{R_{\max}^2 + (R_{\max}+\delta R)^2 - R^2 - (R+\delta R)^2}{8}\,k^2.
\end{aligned}
\tag{A5}
$$

All three vanish as $k^2$, but the coefficients differ sharply. At $R = 1'$ with $R_{\max} = 5'$, $\delta R = 0.75'$, the coefficients are $0.383$ arcmin$^2$ for $\Delta\Sigma$ and $6.75$ arcmin$^2$ for the $Y$ transform: **the $Y$ transform has $17.6\times$ more low-$k$ response than $\Delta\Sigma$ at the same aperture.** That is the price of real-space locality, and it has three consequences worth booking: sensitivity to the simulation box scale, sensitivity to large-scale CMB and atmospheric noise in the kSZ stack, and a larger two-halo contribution.

### 1.4 Where each filter actually lives in $k$

Taking $P_{\rm 2D}(k)\propto k^{n}$ and integrating $k^2 P(k)\hat W(k;R)$, the table below gives the wavenumbers at which the signed cumulative contribution to the amplitude crosses 5, 50 and 95 per cent. Values are for $n=-1$; the $n=-1.5$ case is quoted in Appendix B and is nearly identical for the compensated filters and very different for $\Sigma$, which is itself the diagnostic. Conversion uses a flat $\Lambda$CDM cosmology ($\Omega_m = 0.3111$, $h=0.6766$), for which $1' = 0.383\,$cMpc$/h$ at $z=0.5$ and $0.243\,$cMpc$/h$ at $z=0.30$.

| filter | $R$ | $k_{50}$ [$h/$Mpc, $z{=}0.5$] | $k_{95}/k_{05}$ | $k_{50}R$ |
|---|---|---|---|---|
| $\Sigma$ | $1'$ | 0.97 | 21.2 | 0.37 |
| $\Sigma$ | $3'$ | 0.40 | 20.6 | 0.46 |
| $\Delta\Sigma$ | $1'$ | 5.07 | 3.36 | 1.94 |
| $\Delta\Sigma$ | $3'$ | 1.99 | 3.25 | 2.29 |
| $\Upsilon(1')$ | $2'$ | 2.17 | 2.98 | 1.66 |
| $\Upsilon(1')$ | $3'$ | 1.65 | 3.02 | 1.90 |
| $Y(5')$ | $1'$ | 1.45 | 3.29 | 0.56 |
| $Y(5')$ | $3'$ | 0.94 | 2.95 | 1.08 |

Three readings:

- **$\Sigma$ is a factor of 6-7 broader in $\ln k$ than any compensated filter.** Its response extends to $k_{05} \approx 0.036/R$, i.e. down to the box scale, which is the quantitative version of the Stage 5 finding that $Y_\Sigma$ shifts by 8.7 per cent under a box-scale low-$k$ cut.
- **At fixed aperture the four filters probe different wavenumbers**, by up to a factor of five. $\Delta\Sigma$ at $1'$ sits at $k \approx 5\,h/$Mpc; the $Y$ transform at $1'$ sits at $k \approx 1.5\,h/$Mpc; $\Sigma$ at $1'$ sits at $k \approx 1\,h/$Mpc. Any statement of the form "filter F gives $r$ closer to unity at $R=1'$" therefore conflates genuine localization with a relabelling of scales. **The Stage 3 conclusion that $\Sigma$ outperforms $\Delta\Sigma$ is at least partly this artefact.** When comparing filters, plot against $k_{50}(R;\mathcal{F})$, not against $R$.
- $\Delta\Sigma$, $\Upsilon$ and the $Y$ transform have comparable logarithmic widths ($k_{95}/k_{05} \approx 3$). This matters for Section 4.4: window width, not just window centre, controls one of the two ways the calibration factor departs from unity.

### 1.5 A correction to v0.1: $r$ is not bounded by unity for compensated filters

Section 3, point 4 of v0.1 states an expectation that the $r$'s approach unity on large scales. That is a statement about the Fourier-space coefficient $r_{ab}(k) = P_{ab}/\sqrt{P_{aa}P_{bb}}$, for which Cauchy-Schwarz gives $|r| \le 1$ mode by mode. It does **not** transfer to the filtered coefficients. Writing $Y_{ab} = \int \frac{k\,dk}{2\pi}P_{ab}\hat W$, the bound $|\int P_{ab}\hat W| \le \int\sqrt{P_{aa}P_{bb}}\,|\hat W|$ involves $|\hat W|$, whereas the denominator $\sqrt{Y_{aa}Y_{bb}}$ involves the signed $\hat W$, which is small when the kernel changes sign. Filtered coefficients above unity are therefore permitted and expected. The measured $r_{gb}^{(\Delta\Sigma)} = 1.25$-$2.1$ and Singh et al.'s $r_{cc}^{(\Upsilon)} \sim 1.3$ are both consequences of this, not anomalies.

The practical implication is that individual $r$'s are diagnostics only. The object with a convention-free meaning is the four-amplitude ratio of Section 4.3, and that is what should be plotted as the primary deliverable.

---

## 2. Reconstructing $\Sigma$ from $\Delta\Sigma$

### 2.1 The identity

Differentiating $\bar\Sigma(0,R)R^2 = 2\int_0^R \Sigma R' dR'$ gives

$$
\frac{d\bar\Sigma(0,R)}{d\ln R} = 2[\Sigma(R) - \bar\Sigma(0,R)] = -2\,\Delta\Sigma(R), \tag{A6}
$$

from which, for any $R < R_{\max}$,

$$
\boxed{\;\Sigma(R) - \Sigma(R_{\max}) = \Delta\Sigma(R_{\max}) - \Delta\Sigma(R) + 2\int_R^{R_{\max}}\Delta\Sigma(R')\,d\ln R'\;} \tag{A7}
$$

This is exactly the expression Uroš wrote on 9/2. It is the integration by parts of Prat et al. Eq. (12),

$$
Y(R;R_{\max}) = \int_R^{R_{\max}} d\ln R'\left[2\Delta\Sigma(R') + \frac{d\Delta\Sigma(R')}{d\ln R'}\right], \tag{A8}
$$

since $\int_R^{R_{\max}} (d\Delta\Sigma/d\ln R')\,d\ln R' = \Delta\Sigma(R_{\max}) - \Delta\Sigma(R)$. So the answer to "why don't they integrate the $\Delta\Sigma$ term" is that Eq. (12) does contain the integral; what it does not do is remove the derivative analytically. Verified numerically to machine precision (Appendix B).

### 2.2 Why the by-parts form is the one to implement

Prat et al. discretize Eq. (A8) as $\mathbf{Y} = (2\mathbf{S} + \mathbf{S}\mathbf{D})\Delta\boldsymbol{\Sigma}$ with $\mathbf{S}$ a trapezoidal log-integration matrix and $\mathbf{D}$ a finite-difference matrix. The by-parts form replaces $\mathbf{S}\mathbf{D}$ with the exact boundary operator $\mathbf{B}$, $(\mathbf{B}v)_i = v_{\rm last} - v_i$:

$$
\mathbf{T}_{\rm Prat} = 2\mathbf{S} + \mathbf{S}\mathbf{D}, \qquad \mathbf{T}_{\rm bp} = 2\mathbf{S} + \mathbf{B}. \tag{A9}
$$

Both are linear, both are unbiased in the continuum limit, and both propagate noise correctly through $\mathbf{C}\to\mathbf{T}\mathbf{C}\mathbf{T}^{\rm T}$, so the difference is not statistical. It is a discretization difference, and on coarse grids it is not small. Tested on a cored mock profile (Appendix B):

| grid | worst reconstruction error, $\mathbf{T}_{\rm Prat}$ | worst error, $\mathbf{T}_{\rm bp}$ |
|---|---|---|
| 9 linear bins over $1'$-$6'$, truncated at $R_{\max}=5'$ | 24 per cent | 25 per cent |
| 9 log-spaced nodes over $1'$-$5'$ | 2.2 per cent | 0.33 per cent |
| 15 log-spaced nodes over $1'$-$5'$ | 0.7 per cent | 0.1 per cent |

Two conclusions. **The current linear 9-bin aperture grid cannot support a data-vector-level $Y$ transform**; errors reach tens of per cent as $R \to R_{\max}$, where $Y(R;R_{\max}) \to 0$ and relative errors necessarily blow up. On a log grid the by-parts form is uniformly accurate at the 0.3 per cent level, while $\mathbf{T}_{\rm Prat}$ degrades toward $R_{\max}$ because the finite-difference stencil is worst exactly where the signal is smallest.

The mode-nulling property degrades the same way. Applying each transform to a pure $1/R^2$ vector, the residual leakage relative to the reconstructed signal is 22 per cent ($\mathbf{T}_{\rm Prat}$) and 10 per cent ($\mathbf{T}_{\rm bp}$) on the linear grid, falling to 2.2 per cent on a 9-node log grid and 0.7 per cent on a 15-node log grid. $\Upsilon$, by contrast, nulls the point mass *exactly* at the discrete level, since it is a two-point combination with no quadrature in it. That robustness is a genuine argument in $\Upsilon$'s favour and should be weighed against the $Y$ transform's better localization.

### 2.3 The annulus-mean trap

Identity (A7) is derived for $\Delta\Sigma = \bar\Sigma(0,R) - \Sigma(R)$ with the **local** $\Sigma(R)$. The pipeline measures $\Delta\Sigma_{\rm ann} = \bar\Sigma(0,R) - \bar\Sigma(R,R+\delta R)$. Feeding $\Delta\Sigma_{\rm ann}$ into (A7) does not reconstruct $\bar\Sigma(R,R+\delta R) - \bar\Sigma(R_{\max},R_{\max}+\delta R)$; it reconstructs it plus $2\int_R^{R_{\max}}[\Sigma - \bar\Sigma_{\rm ann}]\,d\ln R'$, which accumulates over the whole integration range. On the mock profile the error is **43 per cent at $R=1'$ and 18 per cent at $R=4'$** (Appendix B). This is not a subtlety that can be neglected; it is the largest single number in this addendum.

### 2.4 Consequence: do not reconstruct where you can measure

The $Y$ transform exists because galaxy-galaxy lensing measures only $\Delta\Sigma$; $\Sigma$ is not an observable of the shear field. That constraint applies to exactly one of the legs in this programme:

| leg | is $\Sigma$ directly available? | route to $Y(R;R_{\max})$ |
|---|---|---|
| $Y_{gb}$, kSZ stack | yes, an annulus mean of the filtered map | measure two annuli and difference them |
| $Y_{gg}$, clustering | yes, from $w_{gg}(r_p)$ | as above |
| $Y_{mm}$, theory | yes, kernel (A4) | analytic |
| all simulation amplitudes | yes, `rprofiles.py` map-level operation | as above |
| $Y_{gm}$ (or $Y_{gt}$), lensing | **no** | reconstruct via (A7), on a log grid, with a matched $\Delta\Sigma$ convention |

So for Route A (Section 4.1) the transform matrix is never needed at all: define the $Y$ transform by (A3) and measure it directly on every leg. Only Route B, which uses the lensing denominator, needs the reconstruction, and there the right implementation is either (i) match the $\Delta\Sigma$ convention by shrinking $\delta R$ until $\bar\Sigma_{\rm ann}\to\Sigma$ to the required accuracy, or (ii) build the transform in harmonic space: find coefficients $a_i$ such that $\sum_i a_i \hat W_{\Delta\Sigma}(k;R_i) \approx \hat W_{Y}(k;R)$ over the $k$ range that carries the signal, which is a small well-posed least-squares problem and automatically respects whatever $\Delta\Sigma$ convention the data were measured with.

### 2.5 Covariance and rank

$\mathbf{T}$ annihilates the $1/R^2$ mode by construction, so $\mathbf{T}\mathbf{C}\mathbf{T}^{\rm T}$ is rank-deficient by one and cannot be inverted directly. Prat et al. handle this with a Tegmark (1997) pseudo-inverse and report it as their least numerically stable method. For our purposes, where the filtered amplitudes enter as a ratio rather than through a likelihood over the full data vector, this is mostly a covariance-propagation issue rather than an inversion problem, but it must not be ignored if a $\chi^2$ is ever formed. The same applies to $\Upsilon$, where the $R = R_0$ bin is identically zero and is already excluded in the current pipeline; for the $Y$ transform the analogous exclusion is $R = R_{\max}$, and in practice bins with $R \gtrsim 0.8\,R_{\max}$ carry little signal (Section 8.2).

---

## 3. Point-mass marginalization: what Prat et al. actually show

For citation accuracy, since the claim will end up in a paper:

- **Fig. 1** of Prat et al. (2023) visualizes the transformed data vectors for $\gamma_t$, $\Upsilon$, the $Y$ transform and their new "project-out" estimator. It does **not** include point-mass marginalization, and the paper says so explicitly, because that method modifies only the inverse covariance and leaves the data vector untouched.
- The equivalence of point-mass marginalization and $\Upsilon$ is established at the **posterior** level in their Figs. 3 (LSST Y1 simulated) and 4 (DES Y3 data), and analytically in their Appendix A, which shows that point-mass marginalization with an infinite prior is identical to projecting out the $1/R^2$ mode. Their conclusion is that removing that mode is the only operation that matters for cosmological parameters.
- **That conclusion should not be imported wholesale into our regime.** It is a statement about information content for $\Omega_m$ and $S_8$ in a $2\times2$pt analysis with scale cuts of 6-8 Mpc$/h$, where small scales are being discarded. We work at $1'$-$6'$, i.e. $0.4$-$2.3$ cMpc$/h$ at $z=0.5$, the statistic itself is the deliverable rather than a step toward a posterior, and Stage 3 already measured that the filters give materially different $r$ values at fixed aperture. There is no contradiction, but the reason must be stated when citing.

**Decision.** Task 2b (point-mass completion of $\Sigma$) is retired. Its motivation, recovering $\Sigma$-like localization from $\Delta\Sigma$ data, is better served by the $Y$ transform, which needs no nuisance parameter and is compensated.

---

## 4. The estimator family

### 4.1 Route A: clustering plus theory (v0.1, Eq. 4)

$$
\frac{Y_{bm}}{Y_{mm}} = \frac{r_{bm}}{r_{gb}}\cdot\frac{Y_{gb}}{\sqrt{Y_{gg}\,Y_{mm}}}. \tag{A10}
$$

Uroš's 9/2 message wrote $Y_{gb}/(Y_{mm}Y_{gg})$ and his 9/3 03:44 correction restored the square root, matching v0.1 Eq. (4). Inputs: measured $Y_{gb}$ (kSZ, full $6{,}709$ deg$^2$), measured $Y_{gg}$ (DESI clustering, not yet built), theory $Y_{mm}$, simulation-calibrated $r_{bm}/r_{gb}$.

### 4.2 Route B: the direct ratio (Uroš, 9/3 05:00)

Substituting $Y_{mm}^{1/2} = Y_{gm}/(r_{gm}Y_{gg}^{1/2})$, which follows from the definition $r_{gm} = Y_{gm}/\sqrt{Y_{gg}Y_{mm}}$, into (A10):

$$
\boxed{\;\frac{Y_{bm}}{Y_{mm}} = \frac{r_{bm}\,r_{gm}}{r_{gb}}\cdot\frac{Y_{gb}}{Y_{gm}}\;} \tag{A11}
$$

$Y_{gb}/Y_{gm}$ is, up to the electron-to-baryon step and the beam, exactly the ratio the $f_{\rm gas}$ paper measures. Route B therefore turns an existing, reviewed measurement into a statement about $P_{bm}/P_{mm}$ with no new data pipeline: no $Y_{gg}$, no fibre-incompleteness campaign, no theory spectrum. The cost is the footprint, which reverts to DESI $\cap$ HSC $\cap$ ACT.

The two routes are complementary rather than competing, and they fail differently: Route A is limited by clustering systematics at arcminute scales and by non-linear matter power accuracy; Route B is limited by lensing statistics and shares the $f_{\rm gas}$ paper's shear systematics. Their ratio is $\sqrt{Y_{mm}Y_{gg}}\,r_{gm}/Y_{gm}$, i.e. $1$ by construction, so consistency between them is a null test of the $Y_{gg}$ and $Y_{mm}$ legs.

### 4.3 The calibration factor is one ratio of four amplitudes

Both routes' correction factors collapse. Writing them out,

$$
\begin{aligned}
C &\equiv \frac{r_{bm}\,r_{gm}}{r_{gb}} = \frac{Y_{bm}\,Y_{gm}}{Y_{mm}\,Y_{gb}}, \\
C_A &\equiv \frac{r_{bm}}{r_{gb}} = \frac{Y_{bm}}{Y_{mm}}\cdot\frac{\sqrt{Y_{gg}\,Y_{mm}}}{Y_{gb}}.
\end{aligned}
\tag{A12}
$$

Note that $Y_{bb}$ and $Y_{gg}$ both cancel out of $C$ entirely. Three consequences:

1. **Eq. (A11) is an algebraic identity, not a model.** Plotting its two sides against each other in simulations tests the code, not the physics. The physics is in the *value* of $C$ and in whether it is stable across codes, feedback models, galaxy samples and redshifts.
2. **$C$ is free of normalization conventions.** Every constant prefactor, including $\bar\rho_m$, the kSZ temperature-to-optical-depth conversion, and the projection depth, cancels in the four-fold ratio. This is why Stage 8's projection-depth study found the coefficients stable to $4\times10^{-4}$ while the amplitudes scaled as $1/L$.
3. **$C$ should be the primary plotted quantity**, in place of the individual $r$'s, which are convention-free only in combination and which are not bounded by unity (Section 1.5).

### 4.4 What makes $C$ equal to one

**Proposition.** Suppose the baryon field can be written as

$$
\delta_b(\mathbf{k}) = S(k)\,\delta_m(\mathbf{k}) + \epsilon(\mathbf{k}), \qquad \langle\epsilon\,\delta_m\rangle = \langle\epsilon\,\delta_g\rangle = 0,
$$

for an arbitrary scale-dependent transfer $S(k)$. Then $P_{bm} = S P_{mm}$ and $P_{gb} = S P_{gm}$, and

$$
C(k) = \frac{P_{bm}P_{gm}}{P_{mm}P_{gb}} = \frac{S P_{mm}\,P_{gm}}{P_{mm}\,S P_{gm}} = 1 \quad \text{identically, at every } k. \tag{A13}
$$

The content is a conditional-independence statement: **$C = 1$ whenever the galaxies' correlation with the gas is entirely mediated by the matter field.** No assumption is made about $S(k)$, so arbitrarily strong, arbitrarily scale-dependent feedback leaves $C = 1$ as long as the gas displacement is a deterministic functional of the matter field plus noise uncorrelated with the galaxy sample. This is the precise version of Uroš's expectation that "$r_{gm}/r_{bm}$ will be relatively flat, possibly close to 1"; the mediation hypothesis is equivalent to $r_{gb} = r_{gm}r_{bm}$.

$C \neq 1$ therefore measures the *direct* galaxy-gas connection not carried by the total matter field: the fact that the gas around the specific galaxies being stacked has been processed by those galaxies' own AGN and supernova histories, which the matter field does not fully record. That is a one-halo, centrally concentrated effect, which is exactly why excising the halo centre should push $C$ toward unity, and why the same operation removes the stellar contribution, which is likewise central.

Two caveats, both important.

**Filtered $C$ is not Fourier $C$.** For filtered amplitudes, $C = 1$ requires in addition that $S(k)$ and $b(k) \equiv P_{gm}/P_{mm}$ do not both vary across the kernel's support, since

$$
C_{\mathcal{F}} = \frac{\langle S P_{mm}\rangle_{\hat W}\,\langle b P_{mm}\rangle_{\hat W}}{\langle P_{mm}\rangle_{\hat W}\,\langle S b P_{mm}\rangle_{\hat W}} = 1 - \frac{\mathrm{Cov}_{\hat W}(S,b)}{\langle S\rangle_{\hat W}\,\langle b\rangle_{\hat W}} + \dots
$$

with $\langle\cdot\rangle_{\hat W}$ the $k\hat W P_{mm}$-weighted average. Narrower kernels suppress this term, which is a second, independent reason to prefer the compensated filters over $\Sigma$ (Section 1.4: $k_{95}/k_{05} \approx 3$ versus $\approx 21$).

**The proposition says nothing about the size of the violation.** It identifies what to look for, not how big it is. That is the measurement.

**An immediate consistency test using existing numbers.** Under mediation, $C_A = r_{bm}/r_{gb} = 1/r_{gm}$. Stage 3 measured $r_{bm}/r_{gb} = 0.53$ at $1'$ rising to $0.85$ at $9'$ for $\Delta\Sigma$, which predicts $r_{gm}^{(\Delta\Sigma)} \approx 1.9$ at $1'$ and $\approx 1.2$ at $9'$. Independently, $r_{gb} = 1.25$-$2.1$ and $r_{bm} = 0.89$-$1.00$, so $r_{gb}/r_{bm} \approx 1.7/0.9 \approx 1.9$ at $1'$. These are consistent, which is encouraging but not yet a test, since it is the same numbers rearranged. **The test is to measure $r_{gm}$ directly and check whether it equals $r_{gb}/r_{bm}$.** If it does, $C \approx 1$ and Route B needs almost no calibration; if $r_{gm}$ comes out near unity instead, then $C \approx 0.53$ at $1'$ and the transfer is large. This is Task 7, and it is cheap: `rprofiles.py` already computes $Y_{XY}$ for arbitrary field pairs, and the galaxy and matter maps both exist.

**A reframing of Gate A, if mediation holds.** Under mediation the Route A factor is $1/r_{gm}$, which is a galaxy-matter stochasticity quantity, not a feedback quantity. Its cross-code scatter would then be driven by differences in the SHAM galaxy samples between codes rather than by feedback prescription. That is consistent with what Stage 3 and Gate A found: feedback sensitivity within FLAMINGO is mild ($\leq 7$ per cent) while the TNG-versus-FLAMINGO difference is 11-13 per cent. If that reading survives Task 7, the Gate A verdict changes character: the residual scatter is a sample-matching problem, addressable by matching $w_{gg}$ or $r_{gm}$ between suites, rather than an irreducible astrophysical uncertainty. It also opens the option of *measuring* $r_{gm}$ from data on the DESI $\cap$ HSC overlap, following Singh et al. exactly, and applying it over the full ACT footprint.

### 4.5 Which matter field: $m$ = CDM or $m$ = total

v0.1 sets $m = $ CDM and $t = m + b$. Lensing measures $t$, not $m$. Uroš's phrase "you are already measuring $Y_{gb}/Y_{gm}$" therefore identifies $Y_{gm}$ with the lensing denominator, i.e. with $Y_{gt}$. Two self-consistent conventions exist and they must not be mixed:

**Convention C ($m$ = CDM), for Route A.** Theory supplies $Y_{mm}$. The relevant statement is that the hydrodynamic CDM auto-spectrum matches the gravity-only total-matter spectrum to 1-2 per cent (van Daalen et al. 2011; Chisari et al. 2018), so halofit or an emulator run on a DMO cosmology is the right model for $Y_{mm}$, up to that back-reaction. Route A should stay in this convention.

**Convention T ($m \to t$ = total matter), for Route B.** Every quantity in (A11) is then measured: $Y_{gb}$ from kSZ, $Y_{gt}$ from lensing, and the target becomes $Y_{bt}/Y_{tt}$. No theory spectrum enters, so neither halofit accuracy (Stage 8's term $C$, 4.6-6.7 per cent) nor the CDM-versus-total question arises. Section 5.3 gives the suppression mapping in this convention, which is as clean as in Convention C.

**Recommendation:** run Route B entirely in Convention T and say so explicitly, since it removes the single largest theory-side term at no physical cost. Confirm with Uroš, since his "$m$" is ambiguous between the two. If instead Route B is forced into Convention C, a feedback-dependent factor $Y_{gt}/Y_{gm}$ reappears and has to be calibrated, which defeats the purpose.

### 4.6 A flag on Stage 8's term $B$

Stage 8 reports a "CDM versus total matter" term of 3-17 per cent, scaling with feedback strength (8.9 per cent for TNG300-1, 16.9 per cent for fgas$-8\sigma$), and lists it as the dominant theory-side systematic. But $Y_{tt}^{\rm hydro}/Y_{mm}^{\rm hydro}$ is, by Eq. (A15) below, approximately $(f_m + f_b x)^2$, which for strong feedback is exactly a 15-17 per cent suppression. **The ordering of $B$ across feedback variants is the ordering of the signal.** If that identification is right, $B$ is not an error to be resolved but the quantity being measured, and the correct theory chain in Convention C is halofit(DMO) $\to Y_{mm}^{\rm hydro\,CDM}$ with only the 1-2 per cent back-reaction correction, exactly as v0.1 Section 4 states. Next step 1 of the task record would then be answered without either of the two routes it proposes. This needs a look at `theory.py` to confirm what $B$ is actually differencing, but it should be checked before spending effort on a CDM-only non-linear prescription.

---

## 5. Mapping to power suppression

### 5.1 Exact relations

With $\delta_t = f_m\delta_m + f_b\delta_b$, $f_b = \Omega_b/\Omega_m \approx 0.157$, define $x \equiv P_{bm}/P_{mm}$ and $x_t \equiv P_{bt}/P_{tt}$. Then, with no assumptions at all,

$$
P_{tt} = f_m P_{mt} + f_b P_{bt} \;\Longrightarrow\; f_m\frac{P_{mt}}{P_{tt}} + f_b x_t = 1, \tag{A14}
$$

$$
\frac{P_{tt}}{P_{mm}} = (f_m + f_b x)^2 + f_b^2\,(1 - r_{bm}^2)\,\frac{P_{bb}}{P_{mm}}. \tag{A15}
$$

The second term in (A15) is the only place the gas stochasticity enters. With $r_{bm}\geq 0.89$ as measured in Stage 3 and $P_{bb}/P_{mm}\sim x^2 \sim 0.25$ in the strong-feedback regime, it contributes $\lesssim 6\times10^{-4}$. **It is negligible at the target precision, so $P_{tt}/P_{mm} = (f_m + f_b x)^2$ can be treated as exact**, which is a stronger statement than the v0.1 Eq. (6) approximation and does not require assuming $r_{bm}\approx 1$, only bounding its departure.

### 5.2 Suppression

$S(k) \equiv P_{tt}^{\rm hydro}/P_{tt}^{\rm DMO}$. Adopting $P_{mm}^{\rm hydro} = P_{tt}^{\rm DMO}$ (the 1-2 per cent back-reaction, booked separately),

$$
S = (f_m + f_b x)^2 \quad\text{(Convention C)}. \tag{A16}
$$

### 5.3 Convention T

Using (A14) with $P_{mt}/P_{tt} = 1/(f_m + f_b x)$, which follows when $r_{bm}=1$,

$$
\boxed{\;S = \frac{f_m^2}{(1 - f_b x_t)^2}\;} \quad \text{(Convention T)}
\tag{A17}
$$

and, without assuming $r_{bm} = 1$, the exact relation is

$$
S = \frac{f_m\,(f_m + f_b x)}{1 - f_b x_t}.
$$

Check: $x_t = 1$ gives $S = 1$; $x_t = 0.542$ gives $S = 0.849$, against $0.850$ from the exact (A15) at $x = 0.5$. So the Convention T mapping is as accurate as the Convention C one and requires no theory spectrum, no DMO run, and no CDM/total conversion. Only the back-reaction assumption remains.

### 5.4 Error propagation

$$
\begin{aligned}
\frac{\delta S}{S} &= \frac{2f_b}{f_m + f_b x}\,\delta x \approx 0.31\,\delta x \qquad (x \approx 1), \\
\frac{\delta S}{S} &= \frac{2f_b}{1 - f_b x_t}\,\delta x_t \approx 0.37\,\delta x_t.
\end{aligned}
\tag{A18}
$$

A 20 per cent error on $C$ propagates to $\approx 6$ per cent on $S$ and $\approx 3$ per cent on $T = \sqrt{S}$, reproducing v0.1 Eq. (7). The $f_b$ suppression is what makes a large but well-characterized transfer tolerable.

### 5.5 What "reasonable agreement with expectations" means operationally

Uroš's request is to check that the filtered ratio reproduces the power-spectrum suppression. That is a four-rung ladder, and it is worth keeping the rungs separate so a failure is diagnosable, in the same spirit as the A/B/C decomposition that found the CAMB bug:

1. **Identity check.** Both sides of (A11) from the same maps. Must agree to numerical precision. Tests code only.
2. **Kernel check.** $Y_{bm}/Y_{mm}(R)$ measured from maps, against the same ratio predicted by pushing the measured 2D spectra $P_{bm}(k)$, $P_{mm}(k)$ through kernel (A4). This is Stage 8's test A extended to the cross-spectrum and should hold at the same 1.7-2.2 per cent.
3. **Localization check.** $Y_{bm}/Y_{mm}(R)$ against $x(k)$ evaluated at $k_{50}(R;\mathcal{F})$ from Section 1.4. The discrepancy is the window smearing, and it is the quantity that decides whether the real-space ratio can be read as a $k$-space ratio or needs the full forward model. Expect this to be smallest for the narrowest kernels.
4. **Suppression check.** $S$ from (A16) or (A17) using the measured $x$, against the directly measured $P_{tt}^{\rm hydro}/P_{mm}^{\rm hydro}$ in the same box. This closes without a DMO run; only the final step to $P_{tt}^{\rm DMO}$ needs one, and that step is the 1-2 per cent back-reaction.

---

## 6. Tasks

### Task 7: $r_{gm}$ and the calibration factor $C$ (highest priority)

Measure $Y_{gm}$ (and $Y_{gt}$) alongside the existing $Y_{gb}$, $Y_{bm}$, $Y_{mm}$, $Y_{gg}$ for all four runs, both samples, all filters. Deliverables:

- $C(R) = Y_{bm}Y_{gm}/(Y_{mm}Y_{gb})$ per filter per run, with jackknife errors formed per realization as in Task 1.
- The mediation test: $r_{gb}$ against $r_{gm}r_{bm}$, bin by bin.
- Both conventions: $C$ with $m$ = CDM and $C_t = Y_{bt}Y_{gt}/(Y_{tt}Y_{gb})$ with $m \to t$.

Pass criteria, mirroring Gate A: cross-code scatter of $C$ below 10 per cent over $1'$-$6'$ supports a fixed transfer with a prior width; 10-20 per cent supports a parametrized route; larger or strongly scale-dependent sends it back for diagnosis. Additionally, report $|C - 1|$, since a $C$ that is both stable *and* near unity is a qualitatively stronger result than one that is merely stable.

Machinery: no new code beyond adding the $gm$ and $gt$ pairs to the existing sweep. Cost is minutes.

### Task 8: the Park et al. $Y$ transform

Implement $Y(R;R_{\max}) = \Sigma(R) - \Sigma(R_{\max})$ as a direct map-level filter, not as a reconstruction, with $R_{\max} = 5'$ fiducial and $R_{\max} \in \{4', 5', 6', 9'\}$ as a sensitivity scan. Then:

- Re-run `check_filter_compensation.py` with the $Y$ transform included. Prediction: it shifts far less than $\Sigma$'s 8.7 per cent under a box-scale low-$k$ cut, but noticeably more than $\Delta\Sigma$'s 0.014 per cent, in proportion to the low-$k$ coefficients of (A5), which is a factor of 17.6 at $R=1'$.
- Re-run the aperture sweep with the $Y$ transform and add it to the $C(R)$ deliverable of Task 7.
- Validate the reconstruction path separately, since Route B needs it for the lensing leg: apply $\mathbf{T}_{\rm bp}$ to simulated $\Delta\Sigma$ binned exactly as the data, and compare against the directly measured $Y(R;R_{\max})$. This is the only clean way to bound the discretization and annulus-mean errors of Section 2.

### Task 9: the plots Uroš asked for

Per filter $\mathcal{F} \in \{\Sigma, \Delta\Sigma, \Upsilon(1'), Y(5')\}$, per run, per sample, a two-panel figure:

- **Upper:** three curves. (i) the truth, $Y_{bm}/Y_{mm}(R)$ measured directly; (ii) the Route A estimator with $C_A$ set to unity, $Y_{gb}/\sqrt{Y_{gg}Y_{mm}}$; (iii) the Route B estimator with $C$ set to unity, $Y_{gb}/Y_{gm}$. The gap between (i) and each of (ii), (iii) *is* the calibration factor, so this single panel answers both "what is the correction" and "how wrong is the uncorrected observable".
- **Lower:** the same three converted to suppression via (A16), overlaid with the directly measured $P_{tt}^{\rm hydro}/P_{mm}^{\rm hydro}$ from the same box, plotted against $k_{50}(R;\mathcal{F})$ so that filters are compared at matched wavenumber rather than matched aperture.

Given the many-line problem, the house style applies: small multiples of 2-3 lines per panel, one panel per filter, with the feedback variants as an envelope plus the fiducial as an exemplar.

### Task 10: revisit Gate A

Re-take the Gate A decision on $C$ rather than on $r_{bm}/r_{gb}$, in both conventions, and with the $Y$ transform in the filter set. Gate B (filter set frozen to $\{\Delta\Sigma, \Upsilon\}$) is reopened by this addendum and should be reclosed as $\{\Delta\Sigma, \Upsilon, Y\}$ or a subset, on the Task 7-8 evidence.

---

## 7. Predictions, and what would falsify them

Stated in advance so that the simulation runs are a test rather than a fit.

1. **The $Y$ transform outperforms $\Upsilon$ on $|C-1|$.** Reason: $\Upsilon$ references $\Delta\Sigma(R_0=1')$, the aperture where the coefficients deviate most from unity and where the ACT beam bites hardest, and imports it into every radius through the $R_0^2/R^2$ factor. The $Y$ transform references $\Sigma(R_{\max})$, the aperture where deviations are smallest. Falsified if $C^{(Y)}$ is no closer to unity than $C^{(\Upsilon)}$ at matched $k_{50}$.
2. **The $Y$ transform and $\Upsilon$ outperform $\Delta\Sigma$.** This is Uroš's expectation and the reason for the whole exercise. Falsified if $C^{(\Delta\Sigma)}$ is already flat and near unity, in which case nothing is gained and the simplest filter wins.
3. **$\Sigma$'s apparent Stage 3 advantage shrinks when compared at matched $k_{50}$** rather than matched $R$, because at $R=1'$ it is probing $k \approx 1\,h/$Mpc where $\Delta\Sigma$ probes $k\approx5\,h/$Mpc.
4. **$C \to 1$ at large $R$ for every filter**, since mediation must hold in the linear regime. Failure of this limit indicates a bug, most likely in the self-pair subtraction, which Stage 3 already found removes $6.9\times$ the retained $Y_{gg}$ signal at $R=1'$ and which TNG300-1's anomalous $9.75'$ behaviour suggests is fragile.
5. **$C$ is more stable across feedback variants than across codes**, because feedback enters through $S(k)$, which cancels identically in (A13), whereas the galaxy sample enters through $b(k)$, which does not cancel unless mediation is exact.

---

## 8. Open issues

### 8.1 The beam, per filter

$C_{\rm beam}(R)$ is filter-dependent and must be recomputed for the $Y$ transform; it does not transfer from the $\Delta\Sigma$ version. The forward-modelling route (convolve the simulated map with the beam, then filter) is unaffected, since both the beam and the filters are linear operations on the map and commute in the sense that matters: filtering a beam-convolved map is the correct forward model either way. `use_sim_scatter` behaviour should be checked with the new filter.

One asymmetry is worth noting: $\Upsilon$ explicitly adds the $R_0 = 1'$ bin, the most beam-suppressed point in the data vector, into every aperture. The $Y$ transform's reference bin sits at $R_{\max}$, where the beam correction is smallest. If the beam correction carries a prior width, $\Upsilon$ propagates the widest one everywhere and the $Y$ transform the narrowest.

### 8.2 Choosing $R_{\max}$

$Y(R_{\max}) \equiv 0$, so the bins near $R_{\max}$ are uninformative and must be excluded, exactly as $R = R_0$ is excluded for $\Upsilon$. For a flat $\Delta\Sigma$, which is what the kSZ data show, (A7) gives $Y(R) = 2\,\Delta\Sigma\,\ln(R_{\max}/R)$: the amplitude at $R = 4'$ is 14 per cent of that at $R=1'$ for $R_{\max}=5'$. With the current $1'$-$6'$ range and $R_{\max}=5'$, the usable range is roughly $1'$-$3.5'$. Enlarging $R_{\max}$ widens the usable range and increases the dynamic range, but also increases the low-$k$ coefficient in (A5) and therefore the large-scale noise and box-scale sensitivity. This is a genuine optimization, and Task 8's $R_{\max}$ scan should settle it rather than fixing $5'$ by assumption.

The same relation, $Y \simeq 2\Delta\Sigma\ln(R_{\max}/R)$, explains what Uroš meant by the transform behaving "very differently" for a flat $\Delta\Sigma$: a constant input maps to a logarithmically declining output, so the shape of the kSZ $Y(R;R_{\max})$ profile is generated almost entirely by the transform rather than by the data. That is a caution as much as a feature, and it is worth checking that the profile carries information beyond the amplitude of $\Delta\Sigma$ before reading structure into it.

### 8.3 Aperture grid

Section 2.2 shows that a data-vector-level $Y$ transform needs log-spaced apertures. The current 9 linear bins over $1'$-$6'$ are matched to the $f_{\rm gas}$ paper and are bit-frozen for that purpose. If Route B is pursued with the $Y$ transform, either (i) the lensing leg is remeasured on a log grid, or (ii) the harmonic-space transform of Section 2.4 is used, which does not care about the grid. Direct $\Sigma$ measurement on the other legs is unaffected either way.

### 8.4 Carried over unchanged from v0.1

Electron versus baryon (Task 3 recommends targeting $P_{em}/P_{mm}$; note that under (A13) the electron-to-baryon step is itself a mediation question and the same cancellation applies if the neutral and stellar components are also mediated by the matter field, which for stars they plainly are not, hence the centre excision argument); velocity-reconstruction suppression (Ondaro-Mallea et al. 2026); fibre incompleteness, which affects Route A only; the covariance across aperture bins, which 16 jackknife regions cannot support and which the $Y$ transform needs more than $\Delta\Sigma$ does because the transform correlates bins by construction.

---

## Appendix A: what the implementation has to get right

A specification rather than code, since the module belongs in `SimulationStacker` next to `rprofiles.py`, `kernels.py` and `theory.py` and should follow their conventions. Required surface:

- the four kernels of Section 1.3, reducing to the pixel-count normalization of `filters.delta_sigma_kernel` exactly, so that the harmonic and map-level paths remain comparable at the level Stage 8's test A established (1.7-2.2 per cent);
- the Park et al. $Y$ transform as a **direct map-level filter** built from two annulus means, not as a reconstruction from $\Delta\Sigma$;
- both discretizations of Section 2.2, $\mathbf{T}_{\rm Prat} = 2\mathbf{S}+\mathbf{S}\mathbf{D}$ and $\mathbf{T}_{\rm bp} = 2\mathbf{S}+\mathbf{B}$, needed for the lensing leg only;
- the calibration factor $C = Y_{bm}Y_{gm}/(Y_{mm}Y_{gb})$ of Section 4.3, and the suppression mappings (A15) to (A18) in both conventions.

Five traps, each of which a fresh implementation will otherwise fall into, with sizes measured in Appendix B:

1. $\mathbf{T}$ must be fed a $\Delta\Sigma$ built on the **local** $\Sigma(R)$. Feeding it the pipeline's annulus-mean $\Delta\Sigma$ biases the reconstruction by 43 per cent at $1'$ (Section 2.3).
2. $\mathbf{T}$ needs a **log-spaced** aperture grid. The current 9 linear bins give reconstruction errors up to 24 per cent and leak the $1/R^2$ mode at the 10-22 per cent level (Section 2.2).
3. Prefer $\mathbf{T}_{\rm bp}$: identical continuum limit, roughly half the point-mass leakage, and flat 0.3 per cent accuracy where $\mathbf{T}_{\rm Prat}$ degrades to 2.2 per cent near $R_{\max}$ (Section 2.2).
4. Exclude $R = R_0$ for $\Upsilon$ and $R = R_{\max}$ for the $Y$ transform, both identically zero by construction; in practice drop $R \gtrsim 0.8\,R_{\max}$ (Sections 2.5 and 8.2).
5. Form $C$ **per jackknife realization**, never by Gaussian propagation of marginal errors on the four amplitudes, matching the convention already fixed for Task 1 in `rprofiles.py`.

## Appendix B: numerical values

Computed with `kernel_diagnostics.py` and `reconstruction_test.py`.

**Low-$k$ coefficients $c_2$ in $\hat W \simeq \hat W(0) + c_2 k^2$, arcmin$^2$, $\delta R = 0.75'$, $R_0=1'$, $R_{\max}=5'$:**

| filter | $R=1'$ | $R=2'$ | $R=3'$ | $R=4'$ | $\hat W(0)$ |
|---|---|---|---|---|---|
| $\Sigma$ | $-0.508$ | $-1.445$ | $-2.883$ | $-4.820$ | 1 |
| $\Delta\Sigma$ | $0.383$ | $0.945$ | $1.758$ | $2.820$ | 0 |
| $\Upsilon(1')$ | $0$ | $0.850$ | $1.715$ | $2.796$ | 0 |
| $Y(5')$ | $6.750$ | $5.812$ | $4.375$ | $2.437$ | 0 |

**Response quantiles, $P_{\rm 2D}\propto k^{n}$, $k$ in arcmin$^{-1}$:**

| filter | $R$ | $k_{05}$ ($n{=}{-}1$) | $k_{50}$ | $k_{95}$ | $k_{50}$ ($n{=}{-}1.5$) |
|---|---|---|---|---|---|
| $\Sigma$ | $1'$ | 0.036 | 0.372 | 0.762 | 0.204 |
| $\Delta\Sigma$ | $1'$ | 0.811 | 1.943 | 2.727 | 1.827 |
| $\Delta\Sigma$ | $3'$ | 0.323 | 0.763 | 1.049 | 0.723 |
| $\Upsilon(1')$ | $2'$ | 0.363 | 0.830 | 1.083 | 0.846 |
| $Y(5')$ | $1'$ | 0.234 | 0.557 | 0.769 | 0.550 |
| $Y(5')$ | $3'$ | 0.158 | 0.359 | 0.466 | 0.367 |

$\Sigma$'s $k_{50}$ moves by a factor 1.8 between the two spectral slopes while every compensated filter moves by under 6 per cent. That instability, not the box-cut test alone, is the reason $\Sigma$ amplitudes do not port.

**Reconstruction accuracy** (cored mock, $\Sigma \propto [1+(R/1.5')^2]^{-0.6}$, $R_{\max}=5'$): continuum identity exact to $3\times10^{-15}$; discretized errors as tabulated in Section 2.2; annulus-mean mismatch 43 per cent at $1'$, 28 per cent at $2'$, 22 per cent at $3'$, 18 per cent at $4'$.

## References

Additions to the v0.1 list:

- Baldauf T., Smith R. E., Seljak U., Mandelbaum R., 2010, PRD 81, 063531
- MacCrann N., et al., 2020, MNRAS 491, 5498
- Park Y., Rozo E., Krause E., 2021, PRL 126, 021301
- Prat J., Zacharegkas G., Park Y., et al. (DES Collaboration), 2023, MNRAS, arXiv:2212.03734
- Tegmark M., 1997, PRD 55, 5895
- Seljak U., 1998, ApJ 503, 492 (mode projection equals marginalization, their Appendix A reference)
