# Mathematical Formalism

**From a simulation box of particles to the matter power suppression.**
Companion to `programme_synthesis.md`. Every step stated as algebra, with derivations, and every approximation flagged at the point it enters.

---

## 1. The chain

$$
\text{particles} \;\to\; \delta_\alpha(\mathbf{x}) \;\to\; \delta_\alpha(\boldsymbol\theta) \;\to\; Y_{\alpha\beta}(R;\mathcal{F}) \;\to\; r_{\alpha\beta},\; C \;\to\; x \;\to\; S(k)
$$

| step | section | exact? |
|---|---|---|
| particles to 3D density fields | §2 | exact given the mass-assignment scheme |
| species algebra $\delta_t = f_m\delta_m + f_b\delta_b$ | §2.3 | **exact by construction** |
| 3D to 2D projection | §3 | exact for a fully projected periodic box |
| filter as convolution, $Y_{\alpha\beta}$ | §4, §5 | exact; two equivalent representations |
| pixelization | §6 | approximate, 1.7 to 2.2 per cent |
| $Y \to r \to C$ | §7, §8 | exact algebra, but $r$ is not a correlation coefficient |
| $x \to S$ | §9 | exact up to $3\times10^{-3}$ plus the back-reaction |
| $R \to k$ | §9.6 | the one genuinely lossy step |

Symbols are collected in Appendix B.

---

## 2. From particles to density fields

### 2.1 Mass assignment

A periodic cube of side $L$ holds particles of species $p \in \{\text{gas}, \text{DM}, \text{stars}, \text{BH}\}$ with masses $m_i$ at positions $\mathbf{x}_i$. Each species is deposited on a grid with a mass-assignment kernel $M$ (NGP, CIC or higher):

$$
\rho_p(\mathbf{x}) = \sum_{i \in p} m_i\, M(\mathbf{x} - \mathbf{x}_i) .
\tag{1}
$$

In Fourier space $M$ multiplies every field by the same $\hat M(\mathbf{k})$, so every measured spectrum carries a factor $\hat M^2(k)$. Because that factor sits **inside** the $k$-integrals of §5, it does not cancel from $r$ or $C$ exactly: it reweights the window, and cancels only to the extent that the spectra in numerator and denominator have similar shapes across it. For CIC at $0.2'$ pixels, $\hat M^2 \approx 0.95$ at the upper edge of the $\Delta\Sigma$ window at $R=1'$, so on absolute amplitudes it is not negligible and must be deconvolved or matched before a measured amplitude is compared against theory. In the four-fold ratio $C$ it is a second-order correction.

### 2.2 The composite fields

$$
\rho_t = \rho_{\rm gas} + \rho_{\rm DM} + \rho_\star + \rho_{\rm BH}, \qquad
\rho_b = \rho_{\rm gas} + \rho_\star + \rho_{\rm BH}, \qquad
\rho_m \equiv \rho_t - \rho_b ,
\tag{2}
$$

with $\rho_e$ the ionized-gas component, a strict subset of $\rho_{\rm gas}$ and therefore of $\rho_b$. Defining the CDM field by subtraction rather than by a separate particle sweep is what makes §2.3 exact, and it was verified on TNG300-1: the derived baryon fraction is $0.157332$ against $\Omega_b/\Omega_m = 0.157332$, with no negative pixels.

Overdensities are normalized by each field's **own** mean,

$$
\delta_\alpha(\mathbf{x}) = \frac{\rho_\alpha(\mathbf{x})}{\bar\rho_\alpha} - 1 ,
\qquad \alpha \in \{g, e, b, m, t\} .
\tag{3}
$$

### 2.3 The species identity

Because each $\delta$ carries its own normalization, the mass sum $\rho_t = \rho_m + \rho_b$ does **not** give $\delta_t = \delta_m + \delta_b$. Substituting (3) into (2):

$$
\delta_t = \frac{\rho_m + \rho_b}{\bar\rho_m + \bar\rho_b} - 1
= \frac{\bar\rho_m(1+\delta_m) + \bar\rho_b(1+\delta_b)}{\bar\rho_m + \bar\rho_b} - 1
= f_m\delta_m + f_b\delta_b ,
\tag{4}
$$

$$
f_b \equiv \frac{\bar\rho_b}{\bar\rho_t}, \qquad f_m \equiv \frac{\bar\rho_m}{\bar\rho_t} = 1 - f_b .
$$

**This is exact, not an approximation**, and it is the hinge of the whole suppression mapping. Two conditions: $\rho_m$ must be defined as $\rho_t - \rho_b$ on the identical grid, and $f_b$ must be the ratio of map means. Using $\Omega_b/\Omega_0$ from the simulation header breaks it at the per-cent level for FLAMINGO, whose $\Omega_0$ carries a neutrino contribution that the particle maps do not. The on-data check of (4) holds to $10^{-10}$; a deliberate 5 per cent perturbation of $f_b$ breaks it.

### 2.4 The galaxy field

$g$ is the one discrete field. SHAM selection ranks subhalos by stellar mass, cuts parent FoF mass at $\le 5\times10^{14}\,M_\odot/h$, and takes the top $N$ to hit a target number density ($5\times10^{-4}$ (cMpc$/h)^{-3}$ for the LRG-like sample, $1\times10^{-3}$ for BGS-like). Then

$$
n(\mathbf{x}) = \sum_{i=1}^{N}\delta^3_{\rm D}(\mathbf{x}-\mathbf{x}_i), \qquad
\delta_g = \frac{n}{\bar n} - 1, \qquad \bar n = N/L^3 .
\tag{5}
$$

Discreteness is not a detail here. It is the sole origin of the self-pair term of §6.4, which at $R=1'$ is several times larger than the signal it sits on.

---

## 3. Projection

### 3.1 Definition

Project the full box along one axis:

$$
\delta_\alpha(\boldsymbol\theta) \equiv \frac{1}{L}\int_0^L dz\; \delta_\alpha(\boldsymbol\theta, z) .
\tag{6}
$$

The $1/L$ makes the projected field a dimensionless overdensity rather than a surface density, which is the normalization convention that keeps all $Y$'s dimensionless and all $r$'s convention-free.

### 3.2 $P_{\rm 2D} = P_{\rm 3D}/L$

Expand in the box's discrete modes, $\delta(\mathbf{x}) = \sum_{\mathbf{k}}\delta_{\mathbf{k}}e^{i\mathbf{k}\cdot\mathbf{x}}$, with $\langle\delta^{\alpha}_{\mathbf{k}}\delta^{\beta\,*}_{\mathbf{k}'}\rangle = \delta_{\mathbf{k}\mathbf{k}'}P^{\alpha\beta}_{\rm 3D}(k)/V$ and $V=L^3$. Then

$$
\delta_\alpha(\boldsymbol\theta) = \sum_{\mathbf{k}}\delta^\alpha_{\mathbf{k}}e^{i\mathbf{k}_\perp\cdot\boldsymbol\theta}\,\frac{1}{L}\int_0^L dz\, e^{ik_zz}
= \sum_{\mathbf{k}_\perp}\delta^\alpha_{(\mathbf{k}_\perp,\,0)}e^{i\mathbf{k}_\perp\cdot\boldsymbol\theta} ,
\tag{7}
$$

since the $z$ integral is $\delta_{k_z 0}$. **Full projection is a selection of the $k_z=0$ plane, with no smoothing at all.** Hence

$$
\langle|\delta^{\rm 2D}_{\mathbf{k}_\perp}|^2\rangle = \frac{P_{\rm 3D}(k_\perp)}{V} \equiv \frac{P_{\rm 2D}(k_\perp)}{A},\quad A = L^2
\quad\Longrightarrow\quad
P^{\alpha\beta}_{\rm 2D}(k) = \frac{P^{\alpha\beta}_{\rm 3D}(k)}{L} .
\tag{8}
$$

Measured directly on a TNG300-1 field, $P_{\rm 2D}L/P_{\rm 3D} = 0.98$ to $1.05$ over $k = 0.1$ to $6\,h/$Mpc. The residual is discretization, not physics.

Equation (8) is the box analogue of the Limber approximation, and it is *exact* here where Limber is not, because there is no redshift kernel and no curved sky. On data the corresponding statement carries Limber's error.

### 3.3 Angular units

Transverse comoving separation maps to angle through $\theta = r_\perp/\chi(z)$, so $k_\theta = \chi k_\perp$ and

$$
1' = \chi(z)\times\frac{\pi}{10800}\ \ \text{cMpc}/h
= 0.383\ \text{cMpc}/h \ (z=0.5), \qquad 0.243\ (z=0.30,\ \text{FLAMINGO}), \qquad 0.213\ (z=0.26,\ \text{TNG}).
\tag{9}
$$

**Depth mismatch.** The projection depth $L$ is a free parameter: the kSZ integrates the full line of sight, the clustering uses a $\Pi_{\max} = 100\,h^{-1}$Mpc cylinder. From (8) every amplitude scales as $1/L$, so depth must be matched by construction when amplitudes are combined. It does not threaten the calibration: reprojecting in slabs of 26 to 205 cMpc$/h$ moves amplitudes by the predicted factor 7.4 to 7.8 against 8, while the coefficients move by $3.8\times10^{-4}$ ($\Delta\Sigma$) and $1.6\times10^{-3}$ ($\Upsilon$).

---

## 4. Filters

### 4.1 A filter is a convolution

Each filter is convolution with an azimuthally symmetric kernel $W_R(\theta)$:

$$
f^R_\alpha(\boldsymbol\theta) \equiv (W_R * \delta_\alpha)(\boldsymbol\theta) = \int d^2\theta'\; W_R(\theta')\,\delta_\alpha(\boldsymbol\theta-\boldsymbol\theta') .
\tag{10}
$$

Its harmonic kernel is the Hankel transform

$$
\hat W_R(k) = \int d^2\theta\; W_R(\theta)\,e^{-i\mathbf{k}\cdot\boldsymbol\theta} = 2\pi\int_0^\infty \theta\,d\theta\; W_R(\theta)J_0(k\theta) .
\tag{11}
$$

Two properties follow immediately and are used throughout:

$$
\textbf{compensation:} \quad \int d^2\theta\, W_R = 0 \iff \hat W_R(0) = 0 .
\tag{12}
$$

$$
\textbf{point-mass response:} \quad \delta_\alpha \supset a\,\delta^2_{\rm D}(\boldsymbol\theta-\boldsymbol\theta_i)\ \text{at each stacking centre} \;\Rightarrow\; \Delta Y_{g\alpha} = a\,W_R(0) .
\tag{13}
$$

Here $a$ has dimensions of area, a projected mass divided by the mean surface density, and the response is that of the stacked, galaxy-crossed amplitude. A filter that vanishes at the origin is blind to a central point mass. No expansion or scale-cut argument is needed; it is a statement about the support of $W_R$.

### 4.2 The two building blocks

Disk mean over $\theta<R$, and annulus mean over $[R_1,R_2]$:

$$
W_{\rm disk}(\theta;R) = \frac{\Theta(R-\theta)}{\pi R^2}, \qquad
W_{\rm ann}(\theta;R_1,R_2) = \frac{\Theta(\theta-R_1)\Theta(R_2-\theta)}{\pi(R_2^2-R_1^2)} .
\tag{14}
$$

Using $\int_0^R \theta J_0(k\theta)\,d\theta = R J_1(kR)/k$, which follows from $\frac{d}{dx}[xJ_1(x)] = xJ_0(x)$:

$$
\hat W_{\rm disk}(k;R) = \frac{2J_1(kR)}{kR}, \qquad
\hat W_{\rm ann}(k;R_1,R_2) = \frac{2\left[R_2J_1(kR_2) - R_1J_1(kR_1)\right]}{k\,(R_2^2-R_1^2)} .
\tag{15}
$$

Both are unit-normalized, $\hat W(0)=1$, so both are uncompensated. The small-$k$ expansion, from $J_1(x) = x/2 - x^3/16 + O(x^5)$, is

$$
\hat W_{\rm ann}(k;R_1,R_2) = 1 - \frac{R_1^2+R_2^2}{8}k^2 + O(k^4) ,
\tag{16}
$$

of which $\hat W_{\rm disk}$ is the case $R_1=0$.

### 4.3 The four filters

Write $\Sigma(R)$ for the annulus mean over $[R, R+\delta R]$ with $\delta R = 0.75'$, the pipeline convention.

$$
\begin{aligned}
\Sigma(R) &= \bar\Sigma(R, R+\delta R), \\
\Delta\Sigma(R) &= \bar\Sigma(0,R) - \Sigma(R), \\
\Upsilon(R;R_0) &= \Delta\Sigma(R) - \frac{R_0^2}{R^2}\Delta\Sigma(R_0), \\
Y(R;R_{\max}) &= \Sigma(R) - \Sigma(R_{\max}).
\end{aligned}
\tag{17}
$$

Harmonic kernels follow by linearity from (15). Real-space kernels are worth writing out, because they make every structural claim visible at a glance:

| filter | $W_R(\theta)$ | support | $\hat W(0)$ | $W_R(0)$ |
|---|---|---|---|---|
| $\Sigma$ | $+\frac{1}{\pi[(R+\delta R)^2-R^2]}$ on $[R,R{+}\delta R]$ | one ring | 1 | 0 |
| $\Delta\Sigma$ | $+\frac{1}{\pi R^2}$ on $\theta<R$; $-\frac{1}{\pi[(R+\delta R)^2-R^2]}$ on the ring | disk out to $R{+}\delta R$ | 0 | $\frac{1}{\pi R^2}$ |
| $\Upsilon(R;R_0)$ | $0$ for $\theta<R_0$; $+$ ring at $R_0$; $+\frac{1}{\pi R^2}$ on $[R_0{+}\delta R, R]$; $-$ ring at $R$ | $[R_0, R{+}\delta R]$ | 0 | 0 |
| $Y(R;R_{\max})$ | $+$ ring at $R$; $-$ ring at $R_{\max}$ | two rings | 0 | 0 |

The $\Upsilon$ entry is the one to verify, since it is the cleanest proof of what $\Upsilon$ does. For $\theta<R_0$ only the two disk terms contribute:

$$
W_\Upsilon(\theta<R_0) = \frac{1}{\pi R^2} - \frac{R_0^2}{R^2}\cdot\frac{1}{\pi R_0^2} = 0 .
\tag{18}
$$

Exactly zero, identically in $R$. $\Upsilon$ does not look at the map inside $R_0$, so by (13) it cannot see a central point mass, a stellar component, or anything else confined there. The $Y$ transform reaches the same conclusion more directly: it is a difference of two rings, so $W_Y(0)=0$ and $\hat W_Y(0) = 1-1 = 0$, making it the only filter in the set that is both local and compensated.

### 4.4 Reconstructing $\Sigma$ from $\Delta\Sigma$

Needed only for a lensing leg, where the shear field gives $\Delta\Sigma$ and never $\Sigma$. Differentiating $\bar\Sigma(0,R)R^2 = 2\int_0^R\Sigma R'dR'$ gives

$$
\frac{d\bar\Sigma(0,R)}{d\ln R} = 2[\Sigma(R)-\bar\Sigma(0,R)] = -2\Delta\Sigma(R) ,
\tag{19}
$$

and integrating from $R$ to $R_{\max}$,

$$
\Sigma(R)-\Sigma(R_{\max}) = \Delta\Sigma(R_{\max}) - \Delta\Sigma(R) + 2\int_R^{R_{\max}}\Delta\Sigma(R')\,d\ln R' .
\tag{20}
$$

Exact, and it is the integration by parts of Prat et al. (2023) Eq. (12). Note (19) holds for the **local** $\Sigma(R)$; feeding it the annulus-mean $\Delta\Sigma$ biases the reconstruction by tens of per cent.

### 4.5 Filter design: what the sharp edges cost, and a smooth alternative

Every kernel in (17) has a step edge in $\theta$, so its harmonic kernel has algebraic tails ($J_1(kR)/kR\sim k^{-3/2}$) and changes sign. That single fact is the origin of several separate entries in the ledger: the signed measure of (30), mechanism 3 of §7.2 and hence $r_{gb}$ at 1.9 and the sign flips of $C$ between filters, the window term of §8.4, the pixelization floor of §6.3, and the boundary ties. At $R=1'$, 42 per cent of the absolute weight in the $\Delta\Sigma$ window sits in negative lobes, so its apparent narrowness ($k_{95}/k_{05}=3.4$ in the table below) is partly cancellation.

**A constraint on the design space.** Evaluating the inverse of (11) at the origin,

$$
W_R(0) = \frac{1}{2\pi}\int_0^\infty k\,\hat W_R(k)\,dk .
\tag{21}
$$

If $\hat W_R\ge0$ everywhere then $W_R(0)>0$, and conversely a kernel that vanishes at the origin must have a sign-changing transform. **A positive harmonic window and immunity to a central point mass are mutually exclusive.** This is the localization programme of §4.3 (null the centre) in direct tension with the window-behaviour programme of §7 and §8 (make $r$ a genuine correlation, make $C_{\mathcal F}$ track $C(k)$). $\Delta\Sigma$ sits on the worst branch of the fork: $W(0)=1/\pi R^2\ne0$, so it sees the point mass, *and* its transform changes sign.

**The difference of Gaussians.** The natural smooth counterpart of $\Delta\Sigma$ replaces the sharp disk and ring by unit-normalized Gaussians of widths $\sigma_1<\sigma_2$:

$$
W_{\rm DoG}(\theta) = \frac{e^{-\theta^2/2\sigma_1^2}}{2\pi\sigma_1^2} - \frac{e^{-\theta^2/2\sigma_2^2}}{2\pi\sigma_2^2},
\qquad
\hat W_{\rm DoG}(k) = e^{-k^2\sigma_1^2/2} - e^{-k^2\sigma_2^2/2} .
\tag{22}
$$

It is compensated, $\hat W(0)=0$, and for every $k>0$ the narrower Gaussian dominates, so $\hat W_{\rm DoG}>0$: **a compensated kernel with a strictly positive window.** Its scales follow in closed form,

$$
\theta_0^2 = \frac{2\sigma_1^2\sigma_2^2}{\sigma_2^2-\sigma_1^2}\ln\frac{\sigma_2^2}{\sigma_1^2}, \qquad
k_*^2 = \frac{2}{\sigma_2^2-\sigma_1^2}\ln\frac{\sigma_2^2}{\sigma_1^2}, \qquad
\hat W_{\rm DoG}\simeq\frac{\sigma_2^2-\sigma_1^2}{2}k^2, \qquad
W_{\rm DoG}(0) = \frac{\sigma_1^{-2}-\sigma_2^{-2}}{2\pi},
\tag{23}
$$

for the zero crossing in real space, the peak wavenumber, the low-$k$ coefficient in the sense of (16), and the point-mass response in the sense of (13). Matched to the $\Delta\Sigma$ window at $R=1'$ (equal $k_{50}$, power-law spectrum, $k$ in arcmin$^{-1}$):

| kernel | $k_{05}$ | $k_{50}$ | $k_{95}$ | negative weight | $W(0)$ [arcmin$^{-2}$] | low-$k$ coeff. [arcmin$^2$] | real-space extent |
|---|---|---|---|---|---|---|---|
| $\Delta\Sigma$, $R=1'$ | 0.81 | 1.94 | 2.73 | 42% | 0.32 | 0.38 | hard edge at $1.75'$ |
| DoG, $\sigma_1=0.58'$, $\sigma_2=1.15'$ | 0.72 | 1.94 | 3.89 | 0 | 0.36 | 0.50 | $\vert W\vert<0.3\%$ of peak beyond $3.5'$ |
| DoG, $\sigma_1=0.65'$, $\sigma_2=0.98'$ | 0.74 | 1.94 | 3.66 | 0 | 0.21 | 0.27 | similar |
| log-Gaussian band-pass, $\Delta\ln k=0.3$ | 1.30 | 2.13 | 3.48 | 0 | $>0$ | rings | |

The DoG keeps $\Delta\Sigma$'s point-mass sensitivity and large-scale response, so nothing is lost on either count, and buys four things: $d\mu_R\ge0$, so Cauchy-Schwarz holds, $r\le1$ strictly and mechanism 3 disappears; $\langle\cdot\rangle_\nu$ becomes a genuine probability average, so $\sigma_s^2\ge0$, $x_{\mathcal F}$ lies within the range of $x(k)$, and the window term of §8.4 is a true covariance with a definite sign; Gaussian rather than algebraic tails, so the pixelization floor and the boundary ties of §6.3 essentially vanish; and, because a Gaussian convolved with a Gaussian is a Gaussian, a Gaussian beam of width $\sigma_b$ composes analytically,

$$
\big(W_{\rm DoG}*B_{\sigma_b}\big)(\theta;\sigma_1,\sigma_2) = W_{\rm DoG}\!\left(\theta;\sqrt{\sigma_1^2+\sigma_b^2},\,\sqrt{\sigma_2^2+\sigma_b^2}\right),
\tag{24}
$$

so the beam correction of §10.2 is exact rather than forward-modelled, and the kernel can be specified post-beam. At the $1'$-equivalent scale $\sigma_1<\sigma_b\approx0.68'$, so the beam sets the effective core, which is physics rather than a defect. The cost is the loss of the aperture-photometry reading of the $f_{\rm gas}$ ratio as gas in a disk over mass in a disk, which the $x\to S$ chain never used.

If the objective is instead to excise the centre, (21) says the window must be signed, and the best available option is a *smooth* point-mass-blind kernel such as a difference of two ring-Gaussians $\propto\theta^2e^{-\theta^2/2\sigma^2}$ at different $\sigma$, which removes the algebraic tails while keeping $W(0)=0$.

**The lensing leg: aperture-mass duality.** The kSZ map, the clustering and the simulations accept any kernel directly, but shear delivers $\Delta\Sigma$, never $\Sigma$. For any **compensated** kernel this is no obstacle. Write $\Sigma = \bar\Sigma(<\theta)-\Delta\Sigma$, integrate the first term by parts against the cumulative kernel $F(\theta)=\int_{|\theta'|<\theta}W\,d^2\theta'$, and use (19). The boundary term at infinity is $F(\infty)\bar\Sigma(<\infty)$, which vanishes because $F(\infty)=\int W=0$, and the same cancellation removes any uniform mass sheet. The result is the classical aperture-mass identity (Schneider 1996; Schneider et al. 1998):

$$
\int d^2\theta\;W(\theta)\,\Sigma(\theta) = \int d^2\theta\;Q(\theta)\,\Delta\Sigma(\theta),
\qquad
Q(\theta) \equiv \bar W(<\theta) - W(\theta) = \frac{2}{\theta^2}\int_0^\theta\theta'W(\theta')\,d\theta' - W(\theta) .
\tag{25}
$$

This is also the precise reason $\Sigma$ itself was never available from lensing while every compensated kernel is. For the DoG, $\int_{|\theta'|<\theta}G_i = 1-e^{-\theta^2/2\sigma_i^2}$ gives the shear-side kernel in closed form,

$$
Q_{\rm DoG}(\theta) = \frac{e^{-\theta^2/2\sigma_2^2}-e^{-\theta^2/2\sigma_1^2}}{\pi\theta^2}
- \frac{e^{-\theta^2/2\sigma_1^2}}{2\pi\sigma_1^2} + \frac{e^{-\theta^2/2\sigma_2^2}}{2\pi\sigma_2^2},
\tag{26}
$$

with $Q_{\rm DoG}(0)=0$ exactly, as any regular shear kernel must have, $Q_{\rm DoG}\ge0$ everywhere, a peak near $1.03'$ for the $1'$-equivalent kernel, and a Gaussian tail of width $\sigma_2$. So the DoG-filtered lensing amplitude is a **positively weighted average of the tangential shear**, and (25) was verified numerically to $5\times10^{-15}$ on a profile that includes a mass sheet. The clean implementation is at the catalogue level, replacing the annulus indicator in the pair sum by $Q_{\rm DoG}(\theta_{ls})$, which is the standard aperture-mass estimator; bin weights on an existing $\Delta\Sigma$ vector also work, with the annulus-mean-versus-local issue of §4.4 now second order because $Q$ is smooth. All three legs then use the same $W$; the lensing leg merely reaches it through $Q$.

The current pipeline is the special case $W=$ disk$(R)$ minus ring$(R,R{+}\delta R)$, whose $Q$ is supported on the ring alone. One consequence worth confirming in the $f_{\rm gas}$ lensing code: the pipeline's annulus-mean $\Delta\Sigma$ is the ring integral of $Q\,\Delta\Sigma_{\rm local}$, not the flat annulus average of $\gamma_t$, and for a profile with locally flat $\Delta\Sigma$ the ring integral of $Q$ at $R=1'$, $\delta R=0.75'$ is $1.66$, not $1$. That is the filter-definition caveat of §4.4 with a number attached, and the same gradient term that made the annulus-mean input to (20) fail by 43 per cent.

**Reach in the shear data.** $Q_{\rm DoG}$ needs $\gamma_t$ out to roughly $3\sigma_2$:

| DoG matched to $\Delta\Sigma$ at | $\sigma_2$ | 95% of weight inside | 99% inside | fraction inside the current $6'$ |
|---|---|---|---|---|
| $1'$ | $1.15'$ | $2.8'$ | $3.4'$ | 1.000 |
| $1.7'$ | $1.96'$ | $4.7'$ | $5.8'$ | 0.992 |
| $2'$ | $2.31'$ | $5.5'$ | $6.7'$ | 0.972 |
| $3'$ | $3.46'$ | $7.2'$ | $7.8'$ | 0.822 |

The existing $1'$ to $6'$ grid supports DoGs to the $2'$-equivalent; larger ones need shear to $8'$ to $10'$, which HSC provides and the $f_{\rm gas}$ grid does not. Because $Q(0)=0$ and $Q$ is small inside $\sim0.5\sigma_1$, the innermost region, where the beam, fibre collisions and small-scale shear systematics live, is automatically downweighted.

**The experiment this suggests.** A DoG is one new kernel in `rprofiles.py`, and running it settles the open question of §8.4 from the other side: if $|C-1|$ and its monotone feedback dependence shrink substantially under a positive window, the deviation was the window term; if they persist at 5 to 10 per cent, the residual is the central galaxy-gas coupling, and the smooth point-mass-blind kernel is the next thing to try.

---

## 5. The amplitudes $Y_{\alpha\beta}$

### 5.1 Definition and two representations

$$
\boxed{\;Y_{\alpha\beta}(R;\mathcal{F}) \equiv \big\langle f^R_\alpha(\boldsymbol\theta)\,\delta_\beta(\boldsymbol\theta)\big\rangle_{\rm map}\;}
\tag{27}
$$

**Real-space form.** Substituting (10) and using statistical homogeneity, with $w_{\alpha\beta}(\theta) = \langle\delta_\alpha(\boldsymbol\theta'+\boldsymbol\theta)\delta_\beta(\boldsymbol\theta')\rangle$,

$$
Y_{\alpha\beta}(R) = \int d^2\theta'\;W_R(\theta')\,w_{\alpha\beta}(\theta') = 2\pi\int_0^\infty \theta\,d\theta\;W_R(\theta)\,w_{\alpha\beta}(\theta) .
\tag{28}
$$

$Y$ is the aperture kernel used as a radial weight on the projected correlation function.

**Harmonic form.** With $\langle\tilde\delta_\alpha(\mathbf{k})\tilde\delta^*_\beta(\mathbf{k}')\rangle = (2\pi)^2\delta^2_{\rm D}(\mathbf{k}-\mathbf{k}')P^{\alpha\beta}_{\rm 2D}(k)$ and Parseval,

$$
Y_{\alpha\beta}(R) = \int\frac{d^2k}{(2\pi)^2}\,\hat W_R(k)P^{\alpha\beta}_{\rm 2D}(k)
= \int\frac{k\,dk}{2\pi}\,\hat W_R(k)\,P^{\alpha\beta}_{\rm 2D}(k) \;\equiv\; \int d\mu_R\; P^{\alpha\beta}_{\rm 2D} ,
\tag{29}
$$

the last step being the azimuthal average. The measure

$$
d\mu_R(k) \equiv \frac{k\,dk}{2\pi}\,\hat W_R(k)
\tag{30}
$$

is the object to keep in mind for the rest of this document. **It is not positive.** Every compensated filter's $\hat W$ changes sign by construction, and even the annulus kernel of $\Sigma$ oscillates beyond its first zero (near $k\approx1.7$ arcmin$^{-1}$ at $R=1'$), so $d\mu_R$ is a signed measure, and most of the counterintuitive behaviour in §7 traces back to that one fact.

### 5.2 Symmetry

From (28), $w_{\alpha\beta}(\theta) = w_{\beta\alpha}(\theta)$ for isotropic fields, so

$$
Y_{\alpha\beta} = Y_{\beta\alpha} .
\tag{31}
$$

Filtering the gas and correlating with galaxies equals filtering the galaxies and correlating with gas. Four fields therefore give ten unordered pairs rather than sixteen ordered ones, which is exactly what `compute_Y_matrix` computes and what made $Y_{gm}$, $Y_{bm}$ and $Y_{mm}$ available in the round-one outputs before Task 7 was posed. Computing a pair both ways is a free numerical check.

### 5.3 The stack is the same computation

Write $\delta_g$ from (5). Its filtered map is $f^R_g = \frac{1}{\bar n}\sum_i W_R(\boldsymbol\theta-\boldsymbol\theta_i) - \hat W_R(0)$, and for any field $\alpha$,

$$
Y_{g\alpha} = \big\langle f^R_\alpha\,\delta_g\big\rangle
= \frac{1}{N}\sum_{i=1}^{N} f^R_\alpha(\boldsymbol\theta_i) \;-\; \big\langle f^R_\alpha\big\rangle ,
\tag{32}
$$

and the second term vanishes identically since $\langle f^R_\alpha\rangle = \hat W_R(0)\langle\delta_\alpha\rangle = 0$. **The map-level average against the galaxy overdensity is exactly the stack of the filtered map at galaxy positions.** The map-level form is the implementation because it additionally reaches the field-field pairs $bm$, $mm$, $bb$, for which no positions exist and no stack is definable.

On the sky, $\langle f^R_\alpha\rangle$ is not exactly zero once masks and an incomplete footprint enter, and the compensated filters are what keep the residual small.

---

## 6. Discretization: what the code computes

### 6.1 The estimator

On an $N_{\rm pix}^2$ grid at roughly $0.2'$ per pixel,

$$
\hat Y_{\alpha\beta}(R) = \frac{1}{N_{\rm pix}^2}\sum_{\rm pix} f^R_\alpha[{\rm pix}]\;\delta_\beta[{\rm pix}],
\qquad
f^R_\alpha = \mathrm{IFFT}\big[\mathrm{FFT}(\delta_\alpha)\cdot\hat W^{\rm pix}_R\big] .
\tag{33}
$$

By discrete Parseval this equals, up to normalization, $\sum_{\mathbf{k}}\hat W^{\rm pix}_R(\mathbf{k})\,\tilde\delta_\alpha\tilde\delta^*_\beta$, so the FFT is an implementation choice, not a different statistic. Each field is transformed once; each aperture costs one multiply and one inverse transform.

### 6.2 The pixelized kernel

$\hat W^{\rm pix}$ is the transform of the **discrete, pixel-counted** kernel, built by testing pixel-centre distances against the aperture edges and normalizing by pixel count. It is not (15). The gap between them is the reason the Task 4 theory comparison was decomposed into three separable pieces rather than run end to end.

### 6.3 Two discretization errors, with sizes

**Annulus resolution.** The $0.75'$ annulus spans 3.75 pixels at production resolution. This sets a **1.7 to 2.2 per cent floor** on any analytic-kernel prediction compared against the pixelized measurement. It is the accuracy limit of the whole harmonic-space theory chain, independent of cosmology. A smooth kernel (§4.5) removes it, along with the boundary ties below.

**Boundary ties.** Membership uses the strict test $r < \text{edge}$. When an edge coincides with a realizable lattice distance $\sqrt{i^2+j^2}$, an entire shell of pixels sits on the boundary and flips under an arbitrarily small convention change. At $0.2'$ pixels this occurs at $R = 1'$ (5.0 px), the $2.25'$ annulus edge (15.0 px) and $R = 6'$ (30.0 px), each a 12-pixel shell, and nowhere else. Lattice arithmetic predicts exactly those three radii, which is the pattern the integration test found. Roughly 6 per cent on an amplitude, 0.6 per cent on a coefficient.

### 6.4 Self-pairs

Only $g$ is discrete, so only $Y_{gg}$ has this term. Working from (5) with $A=L^2$ and $\bar n = N/A$:

$$
Y_{gg} = \frac{1}{A\bar n^2}\sum_{i,j}W_R(\theta_{ij}) - \hat W_R(0)
= \underbrace{\frac{W_R(0)}{\bar n}}_{\text{self-pairs}} + \underbrace{\frac{1}{A\bar n^2}\sum_{i\neq j}W_R(\theta_{ij})}_{\text{clustering}} - \hat W_R(0) .
\tag{34}
$$

The $i=j$ term is $N W_R(0)/(A\bar n^2) = W_R(0)/\bar n$. It is shot noise, not clustering, and must be removed analytically. For $\Delta\Sigma$ and $\Upsilon$ the third term vanishes by compensation; for $\Sigma$ it does not.

Its size is $W_{\rm disk}(0)/\bar n = 1/(\pi R^2\bar n_{\rm 2D})$. For the FLAMINGO LRG-like sample, $\bar n_{\rm 2D} = 5\times10^{-4}\times681 = 0.34$ per (cMpc$/h)^2$ and $\pi R^2 = 0.46$ (cMpc$/h)^2$ at $1'$, giving

$$
\frac{1}{\pi R^2\bar n_{\rm 2D}} \approx 6.4 ,
\tag{35}
$$

in the dimensionless units of (3). The record reports the same term as $6.9\times$ the *retained* clustering signal at $1'$, which puts the retained $Y_{gg}$ near 0.9 and the raw stack near 7.3, roughly 87 per cent shot noise. **So $Y_{gg}$ at small $R$ is a difference of two comparable numbers**, which is why $r_{gb}$ and $r_{gm}$ carry wide error bands there while $r_{bm}$ does not, and why TNG300-1's 4,307 galaxies produce the anomaly at $9.75'$. It is the simulation-side mirror of the fibre-incompleteness problem awaiting the data measurement.

### 6.5 Errors

A $4\times4$ block jackknife over the map, with **every derived quantity formed per realization**:

$$
\hat\sigma^2[q] = \frac{N_J-1}{N_J}\sum_{J=1}^{N_J}\left(q^{(J)} - \bar q\right)^2,
\qquad q^{(J)} = q\big(Y^{(J)}_{\alpha\beta}\big) .
\tag{36}
$$

Never $\sigma^2[q] = \sum_i(\partial q/\partial Y_i)^2\sigma^2[Y_i]$: the four amplitudes entering $C$ are strongly correlated, and Gaussian propagation of marginal errors would misestimate it, here by a large overestimate, because the shared sample variance cancels in the ratio. With $N_J=16$ a bin-to-bin covariance is unavailable, since Hartlap requires more resamplings than bins plus two.

---

## 7. The coefficients $r_{\alpha\beta}$

### 7.1 Definition, and why it is not a correlation coefficient

$$
r_{\alpha\beta}(R;\mathcal{F}) \equiv \frac{Y_{\alpha\beta}}{\sqrt{Y_{\alpha\alpha}Y_{\beta\beta}}} .
\tag{37}
$$

A genuine correlation coefficient of $A = f^R_\alpha$ and $B = \delta_\beta$ would be $\langle AB\rangle/\sqrt{\langle A^2\rangle\langle B^2\rangle}$. Here the numerator is $\langle AB\rangle$, but the denominator is **not**: $Y_{\alpha\alpha} = \langle f^R_\alpha\delta_\alpha\rangle$ carries one power of $\hat W$, whereas $\langle (f^R_\alpha)^2\rangle$ would carry $\hat W^2$. So (37) is a **ratio of window averages**, while the Fourier coefficient

$$
r_{\alpha\beta}(k) = \frac{P_{\alpha\beta}(k)}{\sqrt{P_{\alpha\alpha}(k)P_{\beta\beta}(k)}}, \qquad |r_{\alpha\beta}(k)|\le1
\tag{38}
$$

is an average of ratios. Nothing bounds (37) by unity.

### 7.2 Three mechanisms for $r^{\mathcal{F}}\neq1$

Substituting (38) into (37),

$$
r^{\mathcal{F}}_{\alpha\beta} = \frac{\displaystyle\int d\mu_R\; r_{\alpha\beta}(k)\sqrt{P_{\alpha\alpha}P_{\beta\beta}}}{\sqrt{\displaystyle\int d\mu_R\,P_{\alpha\alpha}\;\int d\mu_R\,P_{\beta\beta}}} .
\tag{39}
$$

1. **Decorrelation**, $r_{\alpha\beta}(k)<1$. The physics, and the only one of the three that is.
2. **Shape mismatch.** Set $r_{\alpha\beta}(k)\equiv1$ and (39) is still not 1. For a positive measure Cauchy-Schwarz gives $\int\sqrt{P_{\alpha\alpha}P_{\beta\beta}}\,d\mu \le \sqrt{\int P_{\alpha\alpha}d\mu\int P_{\beta\beta}d\mu}$, with equality only if $P_{\alpha\alpha}\propto P_{\beta\beta}$ across the window. Two perfectly correlated fields with different spectral shapes give $r^{\mathcal{F}}<1$.
3. **Sign-changing measure.** $d\mu_R$ is not positive, so the Cauchy-Schwarz step of mechanism 2 fails and $r^{\mathcal{F}}>1$ becomes possible. Whichever field carries the most small-scale power suffers the most cancellation in its own auto, shrinking $Y_{\alpha\alpha}$ and inflating any $r$ containing it. This is the only route to $r>1$, and it is why $r_{gb}$ reaches 1.9 and why Singh et al. Fig. 1 shows $r^{(\Upsilon)}_{cc}\sim1.3$. A kernel with a positive window (§4.5) removes it entirely, at the price of seeing the central point mass.

For $r_{gb}$ and $r_{gm}$ there is a fourth: the self-pair residual of §6.4 in the denominator.

**Diagnostic.** Excursions *below* unity can be physical; excursions *above* unity cannot. So the observation that $r_{bm}$ dips below 1 for $\Delta\Sigma$ but rises above 1 for $\Sigma$ and $\Upsilon$, on identical fields in identical runs, localizes the latter entirely in mechanism 3.

### 7.3 A bounded alternative

Filtering both fields restores the bound:

$$
\tilde r_{\alpha\beta}(R) = \frac{\langle f^R_\alpha f^R_\beta\rangle}{\sqrt{\langle (f^R_\alpha)^2\rangle\langle (f^R_\beta)^2\rangle}}
= \frac{\int d\mu^{(2)}_R P_{\alpha\beta}}{\sqrt{\int d\mu^{(2)}_R P_{\alpha\alpha}\int d\mu^{(2)}_R P_{\beta\beta}}},
\qquad d\mu^{(2)}_R \propto \hat W_R^2\,k\,dk \ge 0 .
\tag{40}
$$

Now the measure is positive definite, Cauchy-Schwarz applies, and $|\tilde r|\le1$ strictly. This is not the observable, since the kSZ delivers the $Y$ form, but it is cheap in simulations and separates mechanism 1 from mechanisms 2 and 3 in a single comparison.

---

## 8. The calibration factor $C$

### 8.1 Both routes

From the definitions (37), solve $r_{gb}$ for the unobservable baryon auto, $Y_{bb} = Y_{gb}^2/(r_{gb}^2Y_{gg})$, and substitute into $r_{bm}$:

$$
Y_{bm} = r_{bm}\sqrt{Y_{bb}Y_{mm}} = \frac{r_{bm}}{r_{gb}}\,Y_{gb}\sqrt{\frac{Y_{mm}}{Y_{gg}}}
\quad\Longrightarrow\quad
\boxed{\;x_{\mathcal F} \equiv \frac{Y_{bm}}{Y_{mm}} = \frac{r_{bm}}{r_{gb}}\cdot\frac{Y_{gb}}{\sqrt{Y_{gg}Y_{mm}}}\;}
\tag{41}
$$

which is **Route A**. Now use $r_{gm}$ to eliminate the theory term, $Y^{1/2}_{mm} = Y_{gm}/(r_{gm}Y^{1/2}_{gg})$:

$$
\boxed{\;x_{\mathcal F} = \frac{r_{bm}r_{gm}}{r_{gb}}\cdot\frac{Y_{gb}}{Y_{gm}}\;}
\tag{42}
$$

which is **Route B**. The two are the same statement; they differ only in which observables carry it. The subscript $\mathcal F$ records that this is the filtered, aperture-space ratio; its relation to the harmonic-space $x(k)$ that the suppression needs is the subject of §9.

### 8.2 The calibration factors collapse

$$
C \equiv \frac{r_{bm}r_{gm}}{r_{gb}}
= \frac{Y_{bm}}{\sqrt{Y_{bb}Y_{mm}}}\cdot\frac{Y_{gm}}{\sqrt{Y_{gg}Y_{mm}}}\cdot\frac{\sqrt{Y_{gg}Y_{bb}}}{Y_{gb}}
= \frac{Y_{bm}\,Y_{gm}}{Y_{mm}\,Y_{gb}} ,
\tag{43}
$$

$$
C_A \equiv \frac{r_{bm}}{r_{gb}} = \frac{Y_{bm}\sqrt{Y_{gg}}}{Y_{gb}\sqrt{Y_{mm}}} .
\tag{44}
$$

$Y_{bb}$ leaves both; $Y_{gg}$ leaves $C$ as well. Three consequences:

- **$C$ is convention-free.** Any rescaling $\delta_\alpha\to\lambda_\alpha\delta_\alpha$, and any $k$-independent common factor such as the projection depth, cancels exactly in the four-fold ratio; a $k$-dependent common factor such as the mass-assignment window cancels only approximately (§2.1). This is why the depth study found coefficients stable to $4\times10^{-4}$ while amplitudes scaled as $1/L$.
- **$C$ is numerically better conditioned.** The self-pair-damaged $Y_{gg}$ is gone, as are the auto-spectra most damaged by the negative lobes of §7.2. $C$ runs 1.11 to 1.00 while its ingredients run 0.92 to 1.9.
- **$C$ is always defined.** $C_A$ is undefined in one $\Upsilon(R_0=2')$ bin where $Y_{gg}$ goes negative.

**$C$ is the gap between curves.** Since the uncorrected Route B estimator is $Y_{gb}/Y_{gm}$,

$$
C(R) = \frac{x_{\rm true}(R)}{x_{\rm Route\,B}(R)\big|_{C=1}}, \qquad
C_A(R) = \frac{x_{\rm true}(R)}{x_{\rm Route\,A}(R)\big|_{C_A=1}} .
\tag{45}
$$

### 8.3 The mediation proposition

**Statement.** If the baryon field is any linear, possibly scale-dependent, transform of the matter field plus stochasticity uncorrelated with both the matter and the galaxies,

$$
\delta_b(\mathbf{k}) = \eta(k)\,\delta_m(\mathbf{k}) + \epsilon(\mathbf{k}), \qquad
\langle\epsilon\delta_m\rangle = \langle\epsilon\delta_g\rangle = 0 ,
\tag{46}
$$

then $P_{bm} = \eta P_{mm}$ and $P_{gb} = \eta P_{gm}$, so

$$
C(k) = \frac{P_{bm}P_{gm}}{P_{mm}P_{gb}} = \frac{\eta P_{mm}\cdot P_{gm}}{P_{mm}\cdot \eta P_{gm}} = 1
\tag{47}
$$

identically, at every $k$, for **any** $\eta(k)$. Equivalently, mediation is the statement $r_{gb} = r_{gm}r_{bm}$.

The first condition in (46) is a definition, not an assumption: choosing $\eta = P_{bm}/P_{mm}$ makes $\langle\epsilon\delta_m\rangle=0$ automatically. The entire physical content is the second condition, that the residual gas field is uncorrelated with the galaxies.

Feedback enters only through $\eta$, and $\eta$ cancels. So $C-1$ measures one specific thing: the **direct** galaxy-gas connection not carried by the matter field, which is the gas around the stacked galaxies having been processed by those galaxies' own AGN in a way the total matter field does not record.

### 8.4 The filtered version is not the Fourier version

Insert (46) into (43) with $\beta(k)\equiv P_{gm}/P_{mm}$, the galaxy-matter cross bias, and define the $d\mu_R P_{mm}$-weighted average $\langle\cdot\rangle$:

$$
C_{\mathcal{F}} = \frac{\int d\mu\,\eta P_{mm}\;\int d\mu\,\beta P_{mm}}{\int d\mu\,P_{mm}\;\int d\mu\,\eta\beta P_{mm}}
= \frac{\langle \eta\rangle\langle \beta\rangle}{\langle \eta\beta\rangle}
= \frac{1}{1 + \mathrm{Cov}(\eta,\beta)/\langle \eta\rangle\langle \beta\rangle} .
\tag{48}
$$

This is exact, not a first-order expansion. **Even under exact mediation, the filtered $C$ departs from 1** unless $\eta(k)$ or $\beta(k)$ is constant across the window. The residual shrinks as the window narrows, and since $d\mu_R$ is signed, $\mathrm{Cov}$ here is not a true covariance and can take either sign.

Two things follow that matter for interpreting the measurement. (i) The deviation $C-1$ has two possible origins, genuine mediation failure and window smearing, and they are not distinguished by measuring $C_{\mathcal{F}}$ alone. (ii) Stronger feedback steepens $\eta(k)$, which raises $|\mathrm{Cov}(\eta,\beta)|$, so (48) **also** predicts a monotone feedback dependence, which is the measured ordering 1.105, 1.138, 1.166 across fiducial, Jet_fgas$-4\sigma$, fgas$-8\sigma$ at $R=1'$. (iii) The sign is the expected one for $\Delta\Sigma$: $\eta$ falls with $k$ as feedback smooths the gas while $\beta$ rises with $k$ as galaxies cluster more strongly than matter inside haloes, so $\mathrm{Cov}(\eta,\beta)<0$ and (48) gives $C_{\mathcal{F}}>1$, which is what $\Delta\Sigma$ shows. That the sign reverses for $\Sigma$, $\Upsilon$ and the $Y$ transform is possible only because $d\mu_R$ is signed, and is itself evidence that the window term is not small.

**The separating test.** Compute $C(k) = P_{bm}P_{gm}/(P_{mm}P_{gb})$ directly from the measured 2D spectra. If $C(k)\approx1$ while $C_{\mathcal{F}}(R)$ deviates, the deviation is entirely (48) and the correct response is a narrower window, not a feedback prior. If $C(k)$ itself deviates and tracks feedback, mediation genuinely fails. A corollary: if a single $C(k)$ underlies everything, the filters' $C_{\mathcal{F}}$ curves must collapse onto one curve when plotted against $k_{50}(R;\mathcal{F})$ rather than $R$. A second, independent test is to recompute $C$ with a positive-window kernel (§4.5), under which the window term becomes a true covariance and cannot flip sign between filters.

---

## 9. From $x$ to the suppression $S$

### 9.1 Notation for this section

Two ratios of two-point functions are in play, and the difference between them is the difference between what the estimator delivers and what the suppression needs:

$$
x(k) \equiv \frac{P_{bm}(k)}{P_{mm}(k)}, \qquad
x_{\mathcal F}(R) \equiv \frac{Y_{bm}(R;\mathcal F)}{Y_{mm}(R;\mathcal F)} .
\tag{49}
$$

$x(k)$ lives in harmonic space, one value per wavenumber, and is the quantity that controls the suppression (§9.3). $x_{\mathcal F}(R)$ lives in aperture space, one value per aperture and filter, and is what the estimators (41) and (42) deliver. Neither is a constant. $x(k)$ tends to 1 at low $k$, where gas and matter share the same long-wavelength modes, and falls at high $k$ where feedback has smoothed the gas relative to the dark matter; $x_{\mathcal F}(R)$ inherits that trend, running from about $0.56$ at $1'$ to $0.92$ at $6'$ for $\Delta\Sigma$ at $z\approx0.5$. This section is about how the two are related and how each maps onto $S$.

Each ratio also exists in two versions, depending on which field the label $m$ stands for (§9.2): $x$ and $x_{\mathcal F}$ as written, with $m$ the CDM; and $x^{\rm T}$, $x^{\rm T}_{\mathcal F}$ with $m$ replaced by the total matter $t$. A bare $x$ always means the CDM version.

| symbol | definition | lives in | delivered by |
|---|---|---|---|
| $x(k)$ | $P_{bm}/P_{mm}$ | harmonic space, per $k$ | nothing directly; it is what $S(k)$ needs |
| $x_{\mathcal F}(R)$ | $Y_{bm}/Y_{mm}$ | aperture space, per $R$ and $\mathcal F$ | Route A, Eq. (41) |
| $x^{\rm T}(k)$ | $P_{bt}/P_{tt}$ | harmonic space | nothing directly |
| $x^{\rm T}_{\mathcal F}(R)$ | $Y_{bt}/Y_{tt}$ | aperture space | Route B, Eq. (53) |

**The window average.** For any function $g(k)$ define

$$
\langle g\rangle_{\nu_R} \equiv \frac{\int d\mu_R(k)\,P_{mm}(k)\,g(k)}{\int d\mu_R(k)\,P_{mm}(k)}
= \frac{1}{Y_{mm}(R)}\int d\mu_R\,P_{mm}\,g ,
\tag{50}
$$

with $d\mu_R = (k\,dk/2\pi)\hat W_R(k)$ from (30). The weight $d\mu_R P_{mm}$ is the contribution of each wavenumber to $Y_{mm}(R)$, so $\langle g\rangle_{\nu_R}$ is $g$ averaged over the wavenumbers the filter responds to at aperture $R$, weighted by how strongly it responds there. The reason to introduce it is a two-line identity. Multiply and divide the integrand of $Y_{bm}$ by $P_{mm}$:

$$
x_{\mathcal F}(R) = \frac{\int d\mu_R\,P_{bm}}{\int d\mu_R\,P_{mm}}
= \frac{\int d\mu_R\,P_{mm}\,\big(P_{bm}/P_{mm}\big)}{\int d\mu_R\,P_{mm}}
= \langle x\rangle_{\nu_R} .
\tag{51}
$$

**The aperture-space ratio is the window average of the harmonic-space one.** If $x(k)$ were constant over the filter's support, $x_{\mathcal F}$ would equal it exactly; it is not, and the mismatch is quantified in §9.6. Two cautions. The weight integrates to 1 but is **signed** for every compensated filter, since $\hat W_R$ changes sign, so $\langle\cdot\rangle_\nu$ is not a probability average and $\langle x\rangle_\nu$ is not bounded by the range of $x$. And the weight contains $P_{mm}$, so the same filter at the same aperture averages over slightly different effective ranges in different runs.

### 9.2 The suppression, and the two conventions

**What is being predicted.** The quantity weak lensing cosmology needs is the ratio of the total-matter power spectrum with baryonic physics to the one without,

$$
S(k) \equiv \frac{P^{\rm hydro}_{tt}(k)}{P^{\rm DMO}_{tt}(k)}, \qquad T(k)\equiv\sqrt{S(k)} ,
\tag{52}
$$

defined for total matter because that is what shear responds to. It is a function of $k$, equal to 1 on large scales and falling to 0.8 or below at $k\gtrsim5\,h/$Mpc for strong feedback. Every $S$ in the rest of this section is this object.

**What "convention" means.** Nothing in the derivation of (41) to (44) used the fact that $m$ is the CDM. The identities hold for any four fields, so replacing every $m$ by $t$ gives a second, equally exact estimator, whose target is $x^{\rm T}_{\mathcal F}$ rather than $x_{\mathcal F}$:

$$
x^{\rm T}_{\mathcal F} \equiv \frac{Y_{bt}}{Y_{tt}} = C^{\rm T}\cdot\frac{Y_{gb}}{Y_{gt}}, \qquad
C^{\rm T} \equiv \frac{Y_{bt}\,Y_{gt}}{Y_{tt}\,Y_{gb}} .
\tag{53}
$$

The two choices are bookkeeping, not physics: the same simulation box yields the same $S$ through either, and the calibration factors $C$ and $C^{\rm T}$ agree to about one per cent. What differs is what has to be supplied from outside, and that decides which one to use where.

- **Convention C**, $m$ = CDM. The v0.1 note's choice, and the right one for **Route A**, because Route A needs a theory prediction for $Y_{mm}$, and the CDM field of a hydrodynamic run clusters like the total-matter field of a gravity-only run to 1 to 2 per cent (§9.5), so halofit on a DMO cosmology is a legitimate model for it. Route B in Convention C would be the wrong pairing: lensing measures $Y_{gt}$, not $Y_{gm}$, and converting one to the other needs a feedback-dependent factor, which defeats the purpose.
- **Convention T**, $m\to t$. The right choice for **Route B**, because its observable $Y_{gb}/Y_{gt}$ is exactly the kSZ-to-lensing ratio the $f_{\rm gas}$ paper measures, with no conversion at all. No theory spectrum enters anywhere. The price is that the suppression has to be expressed through $x^{\rm T}$ rather than $x$, which is §9.4.

On the simulation side Convention T costs nothing. $t$ is not a fifth field but a recombination, so every Convention T amplitude follows from the measured ones by (4) and the bilinearity of $Y$:

$$
Y_{gt} = f_mY_{gm}+f_bY_{gb}, \qquad Y_{bt} = f_mY_{bm}+f_bY_{bb}, \qquad Y_{tt} = f_m^2Y_{mm}+2f_mf_bY_{bm}+f_b^2Y_{bb} ,
\tag{54}
$$

with $f_b$ from the map means. These are exact and were verified against a directly built $t$ map to $10^{-10}$.

The mediation proposition of §8.3 carries over with $m\to t$ in its statement. The two hypotheses, gas mediated by the CDM and gas mediated by the total matter, differ only through the stochastic term, which is why $C$ and $C^{\rm T}$ track each other.

### 9.3 Convention C: the exact decomposition

Squaring the species identity (4) and using bilinearity of the two-point function,

$$
P_{tt} = f_m^2P_{mm} + 2f_mf_bP_{bm} + f_b^2P_{bb} .
\tag{55}
$$

Dividing by $P_{mm}$, writing $P_{bm} = xP_{mm}$, and completing the square with $P_{bb} = P_{bm}^2/(r_{bm}^2P_{mm})$,

$$
\frac{P_{tt}}{P_{mm}} = f_m^2 + 2f_mf_bx + f_b^2\frac{P_{bb}}{P_{mm}}
= (f_m + f_bx)^2 + f_b^2(1-r_{bm}^2)\frac{P_{bb}}{P_{mm}} .
\tag{56}
$$

The first term is the whole story; the second is the gas stochasticity, suppressed by $f_b^2\approx0.025$.

**The same identity holds for the filtered amplitudes.** It is not obtained by filtering (56), whose square would not commute with the window average, but by filtering (55), which is linear. Substitute $\delta_t = f_m\delta_m+f_b\delta_b$ into $Y_{tt} = \langle (W_R*\delta_t)\,\delta_t\rangle$. The filter is linear and $f_m$, $f_b$ are constants, so they pass through it:

$$
Y_{tt} = f_m^2\big\langle (W_R*\delta_m)\delta_m\big\rangle + 2f_mf_b\big\langle (W_R*\delta_b)\delta_m\big\rangle + f_b^2\big\langle (W_R*\delta_b)\delta_b\big\rangle
= f_m^2Y_{mm}+2f_mf_bY_{bm}+f_b^2Y_{bb} ,
$$

using $Y_{mb}=Y_{bm}$ from (31). Dividing by $Y_{mm}$ and completing the square,

$$
\frac{Y_{tt}}{Y_{mm}} = (f_m + f_bx_{\mathcal{F}})^2 + f_b^2\left(\frac{Y_{bb}}{Y_{mm}} - x_{\mathcal{F}}^2\right) .
\tag{57}
$$

This holds at every aperture separately, on a single map, with no ensemble average, because the coefficients in (4) do not depend on scale. The scale dependence of the amplitudes between $1'$ and $10'$ plays no role in it: (57) is an identity aperture by aperture exactly as (56) is wavenumber by wavenumber.

The filtered second term is measured across all filters, runs and redshifts at worst $2.9\times10^{-3}$, entering $S$ at the 0.3 per cent level, so $Y_{tt}/Y_{mm} = (f_m+f_bx_{\mathcal F})^2$ is effectively exact. That measured term contains both the filtered stochasticity and a small piece from the non-commutation of the square, $f_b^2\mathrm{Var}_\nu(x)$, which §9.6 isolates. This is also the identity that showed the round-one "term B" to be the signal: $B\equiv Y_{tt}/Y_{mm}-1$ is negative, scales with feedback, and matches $(f_m+f_bx_{\mathcal F})^2-1$ to under one per cent in all four runs.

### 9.4 Convention T: the same suppression from $x^{\rm T}$

**What $x^{\rm T}$ measures.** From (4), $P_{tt} = f_mP_{mt}+f_bP_{bt}$ exactly. Dividing by $P_{tt}$,

$$
f_m\,\frac{P_{mt}}{P_{tt}} + f_b\,x^{\rm T} = 1 \qquad\text{exactly, with}\quad x^{\rm T}\equiv\frac{P_{bt}}{P_{tt}} .
\tag{58}
$$

So the total-matter power splits into a CDM-crossed share and a baryon-crossed share, and $f_bx^{\rm T}$ is the baryons' share. If baryons traced the total matter perfectly, $x^{\rm T}=1$ and their share would equal their mass fraction $f_b$; $x^{\rm T}<1$ means the baryons contribute less to the clustering at that $k$ than their mass would suggest, which is what feedback does. Compare $x$, which measures the baryons against the CDM alone: $x^{\rm T}$ measures them against the mixture they are part of.

**Relation between the two.** If $r_{bm}=1$, so that $\delta_b = x\,\delta_m$ mode by mode, then $\delta_t = (f_m+f_bx)\,\delta_m$, giving $P_{bt} = x(f_m+f_bx)P_{mm}$ and $P_{tt} = (f_m+f_bx)^2P_{mm}$, hence

$$
x^{\rm T} = \frac{x}{f_m+f_bx}, \qquad x = \frac{f_m\,x^{\rm T}}{1-f_bx^{\rm T}} .
\tag{59}
$$

$x=1$ gives $x^{\rm T}=1$; $x=0.5$ gives $x^{\rm T}=0.542$. The two are monotonically related and either can be the working variable, provided the mapping to $S$ is the matching one.

**Suppression from $x^{\rm T}$.** Since $P_{mt}/P_{mm} = f_m+f_bx$ exactly and $P_{mt}/P_{tt} = (1-f_bx^{\rm T})/f_m$ from (58), their ratio is

$$
\frac{P_{tt}}{P_{mm}} = \frac{P_{mt}/P_{mm}}{P_{mt}/P_{tt}} = \frac{f_m\,(f_m+f_bx)}{1-f_bx^{\rm T}} \qquad\text{exactly.}
\tag{60}
$$

Eliminating $x$ with (59) under $r_{bm}=1$,

$$
\boxed{\;\frac{P_{tt}}{P_{mm}} = \frac{f_m^2}{(1-f_bx^{\rm T})^2}\;}
\tag{61}
$$

Check: $x^{\rm T}=1$ gives 1; $x^{\rm T}=0.542$ gives $0.849$, identical to (56) at $x=0.5$ and $r_{bm}=1$. Convention T therefore reaches the same suppression with **no theory spectrum, no DMO run and no CDM-to-total conversion**. Like (56), (61) transfers to the filtered amplitudes with $x^{\rm T}\to x^{\rm T}_{\mathcal F}$, up to a stochastic term of the same size as before.

### 9.5 The back-reaction

Everything above gives $P^{\rm hydro}_{tt}/P^{\rm hydro}_{mm}$, a ratio of two spectra from the **same** hydrodynamic run. The suppression (52) compares against a gravity-only run. Connecting them requires

$$
\frac{P^{\rm hydro}_{mm}}{P^{\rm DMO}_{tt}} = 1 ,
\tag{62}
$$

that is, that the CDM in the hydrodynamic run clusters as the total matter does in the gravity-only run. This is violated at the 1 to 2 per cent level, because baryons pushed out of haloes drag on the dark-matter potential, and it is currently adopted rather than measured, since no DMO run is on disk. With (62), $S = P^{\rm hydro}_{tt}/P^{\rm hydro}_{mm}$, and (56), (57), (60) and (61) become statements about $S$ directly. **This is the only assumption in §9 that is neither exact nor measured.**

### 9.6 The lossy step: from $x_{\mathcal F}(R)$ to $x(k)$

Everything so far is exact at fixed $k$ or at fixed $R$. The one step that loses information is going between them, and it is the reverse of (51): the estimator delivers $\langle x\rangle_{\nu_R}$, and $S(k)$ needs $x(k)$.

Expand $x$ in $s=\ln k$ about $\bar s\equiv\langle s\rangle_\nu$, the window-weighted mean of $\ln k$, which is the effective wavenumber the filter probes at aperture $R$. The linear term vanishes by the definition of $\bar s$, leaving

$$
\langle x\rangle_\nu = x(\bar s) + \tfrac12\,x''(\bar s)\,\sigma^2_s + O(\mu_3), \qquad \sigma^2_s\equiv\mathrm{Var}_\nu(s),
\tag{63}
$$

with $\sigma_s$ the width of the window in $\ln k$. The correction is the **window smearing**: the estimator returns $x$ at the effective wavenumber plus a curvature term proportional to the window's width. It was measured at $+0.8$ to $+1.7$ per cent for $\Delta\Sigma$ against $+1.9$ to $+2.9$ for $\Upsilon$ and the $Y$ transform. Because $d\nu$ is signed for compensated filters, $\bar s$ and $\sigma_s^2$ are not a genuine mean and variance and $\sigma_s^2$ can even be negative, which is why the effective wavenumber is defined operationally as the median of the signed cumulative response,

$$
\int_0^{k_{50}} d\mu_R\,P_{mm} = \tfrac12\int_0^\infty d\mu_R\,P_{mm} ,
\tag{64}
$$

rather than as a moment. Representative values at $z=0.5$, for a power-law $P_{\rm 2D}\propto k^{-1}$ (the pipeline evaluates (64) against each run's measured spectrum instead): $k_{50}R$ runs from 1.9 at $R=1'$ to 2.4 at $4'$ for $\Delta\Sigma$, giving $5.1\,h/$Mpc at $1'$; about $1.5\,h/$Mpc for $Y(5')$ at $1'$; about $1.0$ for $\Sigma$. **At fixed aperture the filters probe wavenumbers differing by up to a factor of five**, so comparing filters at matched $R$ is partly a relabelling of scales.

**A second, smaller non-commutation.** Because $S$ is quadratic in $x$,

$$
\big\langle (f_m+f_bx)^2\big\rangle_\nu - \big(f_m+f_b\langle x\rangle_\nu\big)^2 = f_b^2\,\mathrm{Var}_\nu(x) ,
\tag{65}
$$

so window-averaging and mapping to suppression do not commute. This is the term that moves between the two pieces of (57) relative to the window average of (56). With $x$ varying by $\pm0.1$ across one window it is $2.5\times10^{-4}$: the same $f_b^2$ that suppresses everything else suppresses the nonlinearity of the map.

Three ways to undo the window, in increasing honesty: assign $x(k_{50}(R)) = x_{\mathcal F}(R)$ and accept (63) as the error; forward-model a parametrized $x(k)$ through $\hat W_R$ and fit the aperture data; or regress across simulation suites from $\{x_{\mathcal F}(R)\}$ to $S(k)$ directly, which is robust but reimports simulation dependence.

### 9.7 Error propagation

From (56) with the stochastic term dropped, $S=(f_m+f_bx)^2$, so

$$
\frac{\delta S}{S} = \frac{2f_b}{f_m+f_bx}\,\delta x \;\approx\; 0.31\,\delta x \quad (x\approx1),
\qquad \frac{\delta T}{T} = \tfrac12\frac{\delta S}{S} ;
\tag{66}
$$

the Convention T form (61) gives $\delta S/S = 2f_b\,\delta x^{\rm T}/(1-f_bx^{\rm T})\approx0.37\,\delta x^{\rm T}$. Since $x_{\mathcal F} = C\cdot(\text{observable})$, a fractional error on $C$ is a fractional error on $x$. Six per cent on $C$ gives 1.9 per cent on $S$; twenty per cent gives 6 per cent. **One power of $f_b$ is the entire reason the design tolerates a simulation-calibrated transfer.**

### 9.8 Why the electron field cannot be substituted here

For any component $\alpha$ of the baryons write $x_\alpha\equiv P_{\alpha m}/P_{mm}$, so that $x = x_b$. Equations (55) to (61) descend entirely from (4), which requires the gas field to **complete the mass budget** against the CDM. Electrons are a strict subset of the baryons, not a complementary component, so $\rho_e+\rho_m\neq\rho_t$ and $S(x_e)$ is finite, smooth and meaningless. That is why `SUPPRESSION_FIELD` is pinned to `'b'`; $C$, $C_A$ and $x_e$ remain meaningful and are still written.

What is true is the mass-weighted split, with $w_e+w_n+w_\star=1$ the fractions of the baryon mass in ionized gas, neutral gas, and stars plus black holes:

$$
\delta_b = w_e\delta_e + w_n\delta_n + w_\star\delta_\star
\quad\Longrightarrow\quad
x_b = w_ex_e + w_nx_n + w_\star x_\star ,
\tag{67}
$$

which follows from (4)'s normalization argument applied to the sub-components. $x_e$ is one term of three. The others are not small at small aperture: stars sit at halo centres and so are far more clustered relative to the matter field than the gas, giving $x_\star\gg x_e$ there even for $w_\star$ of a few per cent. If $x_b>x_e$ at small scales, as this argues and as the galaxy-crossed measurement below supports, then **using $x_e$ in place of $x_b$ overestimates the suppression**, by more for stronger feedback and smaller aperture. The measured galaxy-crossed correction $Y_{gb}/Y_{ge}$ at $1'$ runs 1.21 (TNG) to 2.27 (fgas$-8\sigma$), collapsing to 1.02 to 1.07 by $9.75'$, exactly as (67) predicts once feedback evacuates gas from the centre while leaving the stars.

The quantity the chain actually needs is the **matter**-crossed ratio $Y_{bm}/Y_{em}$, not the galaxy-crossed one, and it is already on disk as a ratio of two saved amplitudes.

---

## 10. The data side

### 10.1 What is measured and what is calibrated

$$
x^{\rm obs}(R) = \underbrace{C(R)}_{\text{simulations}}\cdot\underbrace{\frac{Y_{gb}^{\rm obs}}{Y_{gm}^{\rm obs}}}_{\text{kSZ}/\text{lensing}}
\qquad\text{or}\qquad
x^{\rm obs}(R) = C_A(R)\cdot\frac{Y_{gb}^{\rm obs}}{\sqrt{Y_{gg}^{\rm obs}Y_{mm}^{\rm th}}} .
\tag{68}
$$

Note that **none of the three $r$'s is a pure data quantity**. $r_{gb}$ and $r_{bm}$ both contain the unobservable $Y_{bb}$, and the stack delivers $Y_{gb}$, not $r_{gb}$. $r_{gm} = Y_{gm}/\sqrt{Y_{gg}Y_{mm}}$ is measurable in principle but only with a theory $Y_{mm}$, from lensing, clustering and theory, which is precisely the Singh et al. programme; and under mediation $C_A = 1/r_{gm}$, so a data-side $r_{gm}$ would be a direct check on the Route A transfer. $r_{bm}$ is unobservable in principle.

### 10.2 Multiplicative factors on the kSZ leg

The kSZ observable is a temperature, and

$$
Y^{\rm obs}_{gb} = \alpha_{\rm conv}\,\mathcal{R}_v\int d\mu_R\,B(k)\,P_{gb}(k) ,
\tag{69}
$$

with $\alpha_{\rm conv}$ the temperature-to-optical-depth conversion, $\mathcal{R}_v$ the velocity-reconstruction response (10 to 20 per cent suppression, weakly feedback-dependent, scale dependence unresolved), and $B(k)$ the instrument beam. A caution: the convention-freedom of §8.2 applies to $C$ computed in simulations, where all four amplitudes share one normalization. It does **not** apply to the observable ratio in (68), which carries $\alpha_{\rm conv}$ and $\mathcal{R}_v$ explicitly. With lensing removed from Route A there is no partial cancellation of either.

**The beam is filter-dependent.** From (69), the multiplicative correction is a ratio of two window integrals,

$$
C_{\rm beam}(R;\mathcal{F}) = \frac{\int d\mu_R\,P_{gb}}{\int d\mu_R\,B\,P_{gb}} ,
\tag{70}
$$

which depends on $\hat W_R$ and therefore does not transfer between filters. It also depends weakly on the shape of $P_{gb}$, which is what "feedback-independent $C_{\rm beam}$" approximates away. Forward-modelling the beam into the simulated $Y_{gb}$ avoids both issues, and is legitimate because the beam and the filter are both linear operations on the map and so commute in the only sense that matters.

---

## Appendix A. Assumptions, keyed to equations

| # | assumption | where | status |
|---|---|---|---|
| 1 | $\rho_m \equiv \rho_t-\rho_b$ on one grid; $f_b$ from map means | (2), (4) | exact; verified to $10^{-10}$ |
| 2 | mass-assignment window cancels in ratios | (1) | exact for $r$, $C$; not for absolute amplitudes |
| 3 | $P_{\rm 2D}=P_{\rm 3D}/L$ | (8) | exact in principle; measured at 0.98 to 1.05 |
| 4 | depth matching between legs | (8), (9) | amplitudes $\propto1/L$; coefficients stable to $4\times10^{-4}$ |
| 5 | pixelized kernel $\neq$ analytic kernel | (33) | 1.7 to 2.2 per cent floor |
| 6 | boundary ties | §6.3 | $\sim6$% on amplitudes at $R=1'$, $2.25'$, $6'$ only |
| 7 | analytic self-pair subtraction | (34) | exact in form; leaves $Y_{gg}$ a small residual at $1'$ |
| 8 | jackknife with $N_J=16$, $q$ per realization | (36) | no bin-to-bin covariance available |
| 9 | $r^{\mathcal{F}}$ is not bounded by 1 | (39) | structural; interpret with care |
| 10 | mediation, $C=1$ | (46), (47) | **not assumed**; the filtered $C$ departs from 1 by about 5 per cent, and whether that is mediation failure or the window term of (48) is untested |
| 11 | window smearing in $C_{\mathcal{F}}$ | (48) | present even under exact mediation; untested |
| 12 | $r_{bm}\approx1$ in the suppression map | (56), (61) | $\le2.9\times10^{-3}$, measured |
| 13 | back-reaction $P^{\rm hydro\,CDM}_{mm}=P^{\rm DMO}_{tt}$ | (62) | 1 to 2 per cent, adopted not measured |
| 14 | window smearing in $x$ | (63) | $+0.8$ to $+2.9$ per cent, filter-dependent |
| 15 | filtering and $S(x)$ do not commute | (65) | $f_b^2\mathrm{Var}(x)\sim2.5\times10^{-4}$ |
| 16 | electron is not a mass-budget complement | (67) | structural; $S(x_e)$ is meaningless |
| 17 | $\alpha_{\rm conv}$, $\mathcal{R}_v$, $B(k)$ | (69), (70) | all multiplicative, all on the kSZ leg, none cancelling |
| 18 | halofit for $Y^{\rm th}_{mm}$ | (68) | 4.6 to 6.7 per cent, Route A only |
| 19 | sky-side projection is Limber, not the exact slice of (8) | §3.2, (51) | the data's $x_{\mathcal F}$ averages over $k$ and $z$; geometric spread $\pm0.18$ in $\ln k$ across LRG bin 1 against a window half-width of $\pm0.61$, about 4 per cent broadening; evolution of $x(k,z)$ across the sample unmeasured |
| 20 | sharp-edged kernels | (17), (21) | source of rows 5, 6, 9 and 11; a positive-window kernel (22) removes 5, 6 and 9 and makes 11 a true covariance, at the cost of point-mass sensitivity equal to $\Delta\Sigma$'s |

## Appendix B. Symbols

| symbol | meaning |
|---|---|
| $\alpha,\beta$ | field labels, $\in\{g,e,b,m,t\}$ |
| $\delta_\alpha$ | overdensity, normalized by that field's own mean |
| $f_b, f_m$ | mass fractions from the map means, $f_b\approx0.157$ |
| $W_R(\theta),\ \hat W_R(k)$ | filter kernel in real and harmonic space |
| $W_{\rm DoG},\ \sigma_1<\sigma_2$ | difference-of-Gaussians kernel, Eq. (22); positive window, compensated |
| $Q(\theta)$ | shear-side kernel dual to $W$, Eq. (25); $Q_{\rm DoG}$ in Eq. (26) |
| $d\mu_R$ | $(k\,dk/2\pi)\hat W_R(k)$, a **signed** measure |
| $d\nu_R$ | $d\mu_R P_{mm}$, normalized: the weight in $\langle x\rangle$ |
| $Y_{\alpha\beta}(R;\mathcal{F})$ | filtered two-point amplitude, Eq. (27) |
| $w_{\alpha\beta}(\theta)$ | projected angular correlation function |
| $r_{\alpha\beta}$ | $Y_{\alpha\beta}/\sqrt{Y_{\alpha\alpha}Y_{\beta\beta}}$, not a correlation coefficient |
| $\tilde r_{\alpha\beta}$ | doubly filtered coefficient, genuinely bounded by 1 |
| $C,\ C_A$ | calibration factors, Eqs. (43), (44) |
| $x(k)$ | $P_{bm}/P_{mm}$, harmonic space, Convention C |
| $x_{\mathcal F}(R)$ | $Y_{bm}/Y_{mm}$, aperture space; what the estimator delivers |
| $x^{\rm T},\ x^{\rm T}_{\mathcal F}$ | the same two with $m\to t$, Convention T |
| $\langle g\rangle_{\nu_R}$ | window average of $g(k)$, Eq. (50); $x_{\mathcal F}=\langle x\rangle_\nu$ |
| $S(k),\ T$ | suppression $P^{\rm hydro}_{tt}/P^{\rm DMO}_{tt}$, and $\sqrt{S}$ |
| $\eta(k)$ | the mediation transfer, $\delta_b = \eta\delta_m+\epsilon$, Eq. (46) |
| $\beta(k)$ | $P_{gm}/P_{mm}$, the galaxy-matter cross bias entering (48) |
| $a$ | projected point-mass amplitude in (13), dimensions of area |
| $k_{50}(R;\mathcal{F})$ | effective wavenumber, Eq. (64) |

In earlier discussion the mediation transfer was written $S(k)$ and the cross bias $b(k)$; both are renamed here so that $S$ means only the suppression and $b$ only the baryon field.
