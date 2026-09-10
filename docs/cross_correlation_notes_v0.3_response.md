# The Calibration Factor, Measured

**Response v0.3 to `cross_correlation_notes_v0.2_addendum.md`**
R. Henry Liu (with U. Seljak), September 2026

The addendum proposed four new tasks and stated five falsifiable predictions.
Tasks 7, 8 and 10 are now done for all four retained runs, both galaxy samples
and eight filter variants. This document reports what came out, scores the
predictions, and re-takes Gates A and B on the new statistic.

The short version: **the calibration factor $C$ is a far better-behaved object
than $r_{bm}/r_{gb}$, and Gate A passes on it comfortably.** But the addendum's
filter argument is not supported: the $Y$ transform is beaten by both
incumbents on $|C-1|$ at $z\approx0.5$, and its best configuration references an
aperture outside the data range. With only four runs the Gate A scatter cannot
rank the filters at all. And the prediction that mattered most — that $C$ would
be stable across feedback and vary across codes — comes out backwards in 14 of
16 cases.

---

## 0. What changed

1. **Gate A passes, on every filter, at both redshifts.** The cross-run scatter
   of $C$ is 4.2–9.6 per cent, inside the fixed-transfer threshold, where the
   Task 1 statistic $r_{bm}/r_{gb}$ straddled it. Section 3. Jackknife errors
   are 0.001–0.01, so the run-to-run differences are physical rather than
   measurement noise — though with only four runs the *scatter estimate* itself
   carries 41 per cent uncertainty, which is why it cannot rank the filters
   (Section 4.1).
2. **$C$ is close to unity as well as stable**: mean $|C-1| = 0.052$ over
   $1'$–$6'$ for $\Delta\Sigma$ at $z \approx 0.5$ (0.100 at $z\approx0.26$),
   against a 22–45 per cent transfer for $r_{bm}/r_{gb}$. This is the
   "qualitatively stronger result" the addendum asked for. Section 3.
3. **Predictions 1, 2 and 5 are falsified**, 1 and 2 on $|C-1|$ at
   $z\approx0.5$ only, 5 in 14 of 16 (filter, redshift) cells. Section 6.
   **The scatter-based half of predictions 1 and 2 is not testable** with four
   runs, which is itself a finding. Section 4.1.
4. **Gate B is reopened on the field definition, not only the filter set.** The
   electron target frozen in Stage 6 carries twice the cross-run scatter of the
   baryon target. Section 5.2.
5. **Addendum Section 4.6 is confirmed by direct measurement**: Stage 8's
   "term B" is the signal. Section 7.1.

Two numerical corrections to the addendum, and one circular test, are
recorded in Section 7.

---

## 1. What was run

`scripts/cross_corr/make_calibration_factor.py`, one
`rprofiles.compute_Y_matrix` call per (run, projection) over 19 apertures, then
everything else as post-processing. Nothing in `src/` and nothing in
`make_r_profiles.py` was modified, so `data/r_profiles/*.npz` is untouched.

**Three linearity facts made the run cheap**, and each is tested against the
thing it replaces in `tests/test_calibration_factor.py` (17 tests, all passing):

| claim | replaces | test |
|---|---|---|
| $Y(R;R_{\max}) = \Sigma(R) - \Sigma(R_{\max})$ at the amplitude level | a second convolution per aperture | against a directly-built $\Sigma(R)-\Sigma(R_{\max})$ kernel |
| $\Upsilon$ at any $R_0$ on the grid is a recombination of $\Delta\Sigma$ | a separate sweep per $R_0$ | against `compute_Y_matrix`'s own Upsilon |
| $\delta_t = f_m\delta_m + f_b\delta_b$ exactly | a total-matter field sweep | against amplitudes from a directly built $t$ map, synthetic **and** on the real cached TNG maps |

The third deserves emphasis: because the CDM map is *defined* as
`total - baryon`, the identity is exact for $f_b = \langle b\rangle/\langle
t\rangle$ measured on the maps. It is **not** exact for
$\Omega_b/\Omega_m$ from the header, which differs at the per-cent level for
FLAMINGO because `Omega0` carries a neutrino contribution the particle maps do
not. The map-derived value is used throughout.

**Regression check.** All 30 amplitude arrays shared with the committed Task 1
output are bit-identical (worst relative difference exactly $0$). Appending the
four reference radii did not move the nine data-matched bins.

**Cost.** 27.9 minutes in one interactive allocation, one CPU node: 7.6 min for
$z\approx0.5$, 19.7 for $z\approx0.26/0.30$ where the FLAMINGO grids are
$14015^2$. Runner: `cross_corr/runINT_calibration.sh`.

---

## 2. What was measured

For gas field $X \in \{b, e\}$ and matter field $m$ (CDM, Convention C) or
$t$ (total, Convention T):

$$C = \frac{Y_{Xm}\,Y_{gm}}{Y_{mm}\,Y_{gX}}, \qquad
C_A = \frac{r_{Xm}}{r_{gX}} = \frac{Y_{Xm}\sqrt{Y_{gg}}}{Y_{gX}\sqrt{Y_{mm}}},
\qquad x = \frac{Y_{Xm}}{Y_{mm}},$$

with the suppression $S$ from (A16)/(A17), for eight filters —
$\Sigma$, $\Delta\Sigma$, $\Upsilon(R_0 = 1', 2')$ and
$Y(R_{\max} = 4', 5', 6', 9')$ — every quantity formed **per jackknife
realization**, per the addendum's Appendix A trap 5. Note $Y_{XX}$ cancels
out of $C_A$ algebraically, which is worth knowing when propagating errors.

---

## 3. Task 7: the calibration factor

### 3.1 $C(R)$ for the fiducial filter

$z \approx 0.5$, gas = baryons, Convention C, jackknife errors from 16 blocks:

| $R$ [arcmin] | TNG300-1 | L1_m9 fid | fgas−8σ | Jet_fgas−4σ | scatter |
|---|---|---|---|---|---|
| 1.000 | 1.114 ± 0.006 | 1.105 ± 0.001 | 1.166 ± 0.001 | 1.138 ± 0.002 | 2.4% |
| 1.625 | 1.047 ± 0.006 | 1.061 ± 0.001 | 1.155 ± 0.002 | 1.110 ± 0.002 | 4.5% |
| 2.250 | 1.029 ± 0.004 | 1.035 ± 0.001 | 1.130 ± 0.002 | 1.091 ± 0.002 | 4.5% |
| 2.875 | 1.024 ± 0.005 | 1.019 ± 0.001 | 1.101 ± 0.003 | 1.076 ± 0.002 | 3.8% |
| 3.500 | 1.020 ± 0.005 | 1.010 ± 0.001 | 1.080 ± 0.002 | 1.066 ± 0.002 | 3.3% |
| 4.125 | 1.007 ± 0.003 | 1.002 ± 0.001 | 1.062 ± 0.002 | 1.055 ± 0.002 | 3.0% |
| 4.750 | 1.003 ± 0.006 | 0.997 ± 0.001 | 1.047 ± 0.002 | 1.045 ± 0.002 | 2.6% |
| 5.375 | 1.003 ± 0.006 | 0.994 ± 0.001 | 1.035 ± 0.002 | 1.037 ± 0.003 | 2.2% |
| 6.000 | 1.001 ± 0.006 | 0.993 ± 0.002 | 1.025 ± 0.002 | 1.030 ± 0.003 | 1.8% |

Two things to note. The jackknife errors are **0.001–0.006**, an order of
magnitude below the cross-run scatter, so unlike the Stage 4 situation — where
per-run errors exceeded the scatter and the naive Gate A verdict was
meaningless — the scatter here is a real physical difference between runs.
And $C \to 1$ smoothly with aperture, reaching 1.8 per cent scatter and
$|C-1| \le 0.03$ by $6'$.

### 3.2 Against the Task 1 statistic

Over $1'$–$6'$ at $z\approx0.5$, $\Delta\Sigma$:

| statistic | cross-**code** (TNG vs FLA-fid) | cross-**feedback** (fid vs fgas−8σ) | typical value |
|---|---|---|---|
| $r_{bm}/r_{gb}$ (Task 1) | 10.9% | 10.4% | 0.55 → 0.78 |
| $C$ (this work) | **1.3%** | 9.4% | 1.11 → 1.00 |

Replacing the Route A factor with the four-amplitude $C$ collapses the
cross-code disagreement by a factor of eight and converts a 22–45 per cent
transfer into a 1–13 per cent one. This is the addendum's central claim, and it
holds.

**One qualification, which matters.** The 1.3 per cent cross-code agreement is
a $z\approx0.5$ result. At $z\approx0.26/0.30$ the same comparison gives
**9.7 per cent** for $\Delta\Sigma$ — still inside the Gate A threshold, but
seven times worse, and the redshift split is the same direction Stage 3 and
Gate A already reported for $r_{bm}/r_{gb}$. Whether that is the SHAM sample
matching degrading at the higher number density, or the FLAMINGO $z=0.30$
snapshot against TNG's $z=0.26$, is not resolved here. The headline
cross-code collapse should be quoted for the LRG-like sample only.

---

## 4. Task 8: the Park et al. $Y$ transform

Implemented as a direct map-level filter, per the addendum's Appendix A. The
$R_{\max}$ scan, $z\approx0.5$, mean $|C-1|$ and worst cross-run scatter over
$1'$–$6'$:

| filter | usable bins | mean $|C-1|$ | max scatter | Gate A |
|---|---|---|---|---|
| $\Delta\Sigma$ | 12 | 0.052 | 0.047 | fixed |
| $\Sigma$ | 12 | 0.051 | 0.042 | fixed |
| $\Upsilon(R_0=1')$ | 11 | 0.078 | 0.083 | fixed |
| $\Upsilon(R_0=2')$ | 9 | 0.123 | 0.044 | fixed |
| $Y(R_{\max}=4')$ | 5 | 0.116 | 0.049 | fixed |
| $Y(R_{\max}=5')$ | 6 | 0.122 | 0.046 | fixed |
| $Y(R_{\max}=6')$ | 9 | 0.118 | 0.064 | fixed |
| $Y(R_{\max}=9')$ | 12 | 0.086 | 0.046 | fixed |

And at $z\approx0.26/0.30$, where the ranking is **not** the same:

| filter | usable bins | mean $|C-1|$ | max scatter |
|---|---|---|---|
| $\Delta\Sigma$ | 12 | 0.100 | 0.059 |
| $\Sigma$ | 12 | 0.057 | 0.060 |
| $\Upsilon(R_0=1')$ | 11 | 0.059 | 0.096 |
| $\Upsilon(R_0=2')$ | 9 | 0.068 | 0.085 |
| $Y(R_{\max}=4')$ | 5 | 0.066 | 0.078 |
| $Y(R_{\max}=5')$ | 6 | 0.076 | 0.074 |
| $Y(R_{\max}=6')$ | 9 | 0.093 | 0.071 |
| $Y(R_{\max}=9')$ | 12 | 0.094 | 0.068 |

### 4.1 The scatter column cannot rank the filters

Before reading anything into the ordering: the scatter is a sample standard
deviation over **four** runs, whose own sampling uncertainty is
$\sigma/\sqrt{2(N-1)} = 41$ per cent relative. Testing each filter against the
best in its column:

| | $z\approx0.5$ | $z\approx0.26/0.30$ |
|---|---|---|
| filters within $1\sigma$ of the best scatter | **7 of 8** | **7 of 8** |
| only filter significantly worse | $\Upsilon(R_0=1')$, $1.7\sigma$ | $\Upsilon(R_0=1')$, $1.1\sigma$ |

So the honest statement is that **this measurement does not discriminate
between the filters on Gate A scatter at all**, with the single exception that
$\Upsilon(R_0=1')$ is disfavoured at both redshifts. The apparent ordering —
$\Sigma$ 0.042 against $\Delta\Sigma$ 0.047 at $z\approx0.5$, for instance — is
noise. Any filter recommendation has to rest on something else.

That "something else" exists, and is resolved:

- **Mean $|C-1|$ differences are large enough to matter.** At $z\approx0.5$,
  $\Delta\Sigma$ and $\Sigma$ sit at 0.05 while every $Y$-transform variant with
  $R_{\max}\le6'$ sits at 0.12 — a factor of two, far outside anything the
  four-run sampling can produce.
- **Usable bin count is exact**, not estimated: 12 for $\Delta\Sigma$, 5–6 for
  the addendum's fiducial $Y(5')$.
- **The $C\to1$ limit** (Section 6, prediction 4) is a clean qualitative split.

**Reading the two tables together on $|C-1|$**, then: $\Delta\Sigma$ has the
smallest at $z\approx0.5$, but at $z\approx0.26$ it is the *largest* in the set,
with $\Sigma$ and $\Upsilon(1')$ roughly half of it. That reversal is not
explained here and is the one result in this document I would most want a third
code to arbitrate.

### 4.2 Choosing $R_{\max}$

The addendum's fiducial $R_{\max}=5'$ leaves only six usable bins in $1'$–$6'$,
as its Section 8.2 predicted. Larger $R_{\max}$ improves both columns — at
$z\approx0.26$ the $|C-1|$ trend across $4'\to9'$ is monotone within the noise,
and the bin count doubles — which argues for $9'$ over $5'$.

But $9'$ is outside the data range. On real data the transform would reference
an aperture the kSZ stack does not measure, and $\Sigma(9')$ is exactly where
the uncompensated large-scale CMB and atmospheric noise the addendum worries
about in its Section 1.3 would be worst. That is a genuine obstacle the
addendum does not address, and it means the $Y$ transform's best configuration
here is not one the measurement could adopt.

$\Sigma$ is reported for completeness. Its coefficients behave well here, as
Stage 5 already found, but the Stage 5 disqualification stands on the amplitude
argument: uncompensated amplitudes do not port between volumes, which the theory
transfer needs.

---

## 5. Task 10: the gates, re-taken

### 5.1 Gate A — **PASSES, fixed-transfer route**

Every filter, both redshifts, is inside the 10 per cent fixed-transfer
threshold. Worst case across the whole grid is $\Upsilon(R_0=1')$ at
$z\approx0.26$, at 9.6 per cent; $\Delta\Sigma$ is 4.7 and 5.9 per cent at the
two redshifts. Propagated through (A18), a 6 per cent error on $C$ gives
**1.9 per cent on $S$** and 0.9 per cent on $T$.

The caveat from the task record is unchanged and must be restated: this is two
code families. It is now a *better-measured* two-family result, not a
three-family one — and Section 4.1 shows that four runs are enough to establish
that Gate A passes, but not enough to rank the filters within it.

### 5.2 Gate B — **reopened, and not where the addendum expected**

The addendum reopened Gate B on the filter set. The measurement reopens it on
the **field definition** instead.

Mean $|C-1|$ / max cross-run scatter, $z\approx0.5$, $1'$–$6'$:

| filter | gas = baryons | gas = electrons |
|---|---|---|
| $\Delta\Sigma$ | 0.052 / **0.047** | 0.074 / **0.095** |
| $\Upsilon(1')$ | 0.078 / 0.083 | 0.118 / 0.067 |
| $Y(9')$ | 0.086 / 0.046 | 0.102 / 0.053 |

For the fiducial filter the electron target carries **twice** the cross-run
scatter of the baryon target. Stage 6 froze the target as $P_{em}/P_{mm}$ on
the argument that the electron field needs no correction factor because it is
what the kSZ measures. That argument is about the *observable*; this is about
the *calibration*, and they point in opposite directions. The addendum's
Section 8.4 anticipated the tension — the electron-to-baryon step is itself a
mediation question, and stars are plainly not mediated by the matter field —
but the size now favours carrying the baryon target with its transfer.

**Recommendation: reopen the Stage 6 decision.** Not resolve it here; the two
framings differ by more than the two-family scatter can adjudicate, which is
the same limitation Stage 6 hit.

### 5.3 Filter set: recommend $\{\Delta\Sigma\}$ fiducial, $\Upsilon$ retained

Given Section 4.1, this recommendation rests only on the things the measurement
actually resolves:

- $\Delta\Sigma$ ties for the **most usable bins** (12) and is the only
  compensated filter with a clean $C\to1$ limit (Section 6, prediction 4).
- Its $|C-1|$ is the smallest at $z\approx0.5$ by a factor of two over every
  $Y$-transform variant with $R_{\max}\le6'$, a gap well outside the noise.
- It is the incumbent, frozen at Gate B and used for the $f_{\rm gas}$ paper, so
  keeping it costs nothing in re-validation.

**Nothing in the measurement supports adding the $Y$ transform**, and its best
configuration ($R_{\max}=9'$) references an aperture outside the data range.
$\Upsilon$ is worth retaining as the point-mass-nulling cross-check, where its
exact discrete nulling (addendum Section 2.2) is a real advantage that $C$ does
not capture — but $R_0=2'$ rather than $1'$, since $\Upsilon(1')$ is the one
filter this measurement *can* disfavour, at both redshifts.

What would change this: a third code family, or the matched-$k_{50}$ comparison
of Task 9. Not more apertures, and not more jackknife regions.

---

## 6. The five predictions, scored

| # | prediction | verdict |
|---|---|---|
| 1 | $Y$ transform beats $\Upsilon$ on $\|C-1\|$ | **falsified** at $z\approx0.5$ (0.086–0.122 vs 0.078); reversed at $z\approx0.26$ |
| 2 | $Y$ and $\Upsilon$ beat $\Delta\Sigma$ | **falsified on $\|C-1\|$** at $z\approx0.5$; **untestable on scatter** (Section 4.1) |
| 3 | $\Sigma$'s advantage shrinks at matched $k_{50}$ | **untested** — see below |
| 4 | $C\to1$ at large $R$ for every filter | **holds for $\Delta\Sigma$, $\Sigma$, $Y$; fails for $\Upsilon$** |
| 5 | $C$ more stable across feedback than across codes | **falsified, backwards** in 14/16 cells |

**On 1 and 2**, two caveats, both of which cut against a firm verdict. The
addendum's own applies — this is matched aperture, not matched $k_{50}$. And
the scatter column cannot separate the filters at all with four runs
(Section 4.1), so predictions framed on stability are simply not testable at
present power. What *is* testable is $|C-1|$, where the $z\approx0.5$ gap is a
factor of two and clearly falsifies both predictions — but note that
prediction 1 reverses sign at $z\approx0.26$, where $Y(4')$ at 0.066 does beat
$\Upsilon(2')$ at 0.068 and roughly ties $\Upsilon(1')$ at 0.059. The verdicts
above are therefore weaker than a bare "falsified" suggests, and I would not
retire the $Y$ transform on this evidence alone — the case for dropping it
(Section 4.2) rests on the out-of-range $R_{\max}$, not on these scores.

**On 3**, honestly: not tested. Doing it properly needs $k_{50}$ computed
against the *measured* $P_{\rm 2D}$ and the *pixelized* kernel, not the
power-law approximation of the addendum's Section 1.4 — whose own Appendix B
shows $\Sigma$'s $k_{50}$ moving by a factor 1.8 between $n=-1$ and $n=-1.5$.
That is Task 9 work and is not attempted here.

**On 4**, $\Upsilon$ plateaus at $C = 0.885$–$0.940$ at $9.75'$ where
$\Delta\Sigma$ reaches $0.990$–$1.014$. This is *not* the bug the addendum
suspected: it is independent of $R_0$ (both $1'$ and $2'$ show it) and the other
three filters converge. The explanation is the addendum's own Section 8.1 point.
$\Upsilon(R) = \Delta\Sigma(R) - (R_0/R)^2\Delta\Sigma(R_0)$, and although
$(R_0/R)^2 \approx 0.01$ at $9.75'$, $\Delta\Sigma(1')$ exceeds
$\Delta\Sigma(9.75')$ by roughly two orders of magnitude, so the reference term
stays order-unity at every aperture. $\Upsilon$ never converges to
$\Delta\Sigma$ over the accessible range.

**On 5**, this is the important one. Cross-feedback exceeds cross-code in
**14 of 16** (filter, redshift) cells — 9.4 vs 1.3 per cent for $\Delta\Sigma$
at $z\approx0.5$, 16.4 vs 3.2 for $\Upsilon(1')$. The two exceptions are
$\Delta\Sigma$ at $z\approx0.26$ (9.7 code vs 7.5 feedback) and
$\Upsilon(R_0=2')$ at $z\approx0.5$ (11.2 vs 6.8), and both are cases where the
cross-code term is anomalously *large* rather than the feedback term small, so
they do not rescue the prediction. Full table:

| filter | $z\approx0.5$ code / feedback | $z\approx0.26$ code / feedback |
|---|---|---|
| $\Delta\Sigma$ | 1.3% / 9.4% | 9.7% / 7.5% |
| $\Sigma$ | 3.9% / 7.9% | 4.0% / 10.2% |
| $\Upsilon(1')$ | 3.2% / 16.4% | 7.4% / 16.2% |
| $\Upsilon(2')$ | 11.2% / 6.8% | 6.2% / 14.5% |
| $Y(9')$ | 6.1% / 8.5% | 5.3% / 12.1% |

The addendum's Eq. (A13) argument says feedback enters only
through $S(k)$, which cancels identically, so $C$ should be feedback-blind.
It is not. **The mediation hypothesis fails specifically in the strong-feedback
runs**, which is physically sensible: AGN in the stacked haloes process the gas
around those particular galaxies in a way the total matter field does not
record, and that is exactly the direct galaxy–gas stochasticity the proposition
identifies as the residual.

The consequence matters for how the transfer is used. The addendum's Section
4.4 offered an optimistic reframing — that the residual scatter is a
sample-matching problem, fixable by matching $w_{gg}$ or $r_{gm}$ between
suites. That reframing does not survive: the residual is astrophysical and
tracks feedback strength. It remains tolerable (9.4 per cent → 3.0 per cent on
$S$), but it should be booked as a feedback-dependent prior, not as something a
better SHAM match would remove.

---

## 7. Corrections to the addendum

### 7.1 Section 4.6 is right: term B is the signal — confirmed

`check_theory_transfer.py` computes `B = Y_tt/Y_mm - 1` from two maps of the
same hydrodynamic box. Reconstructing $Y_{tt}$ from the measured amplitudes:

| run | worst $|B|$ measured | Stage 8 reported | sign | $(f_m + f_b x)^2 - 1$ |
|---|---|---|---|---|
| TNG300-1, $\Delta\Sigma$ | 8.868e−2 | 8.9e−2 | negative | −8.997e−2 |
| L1_m9 fiducial | 1.318e−1 | 1.34e−1 | negative | −1.327e−1 |
| fgas−8σ | 1.668e−1 | 1.69e−1 | negative | −1.677e−1 |
| Jet_fgas−4σ | 1.652e−1 | 1.67e−1 | negative | −1.660e−1 |

B is negative, reproduces the Stage 8 values, and matches
$(f_m + f_b x)^2 - 1$ to under one per cent relative in all four runs. **It is
the suppression, not a theory-side systematic.** Next step 1 of
`tasks_1_to_4_record.md` — building a CDM-only non-linear prescription to
resolve B — should be cancelled. The correct Convention C chain is
halofit(DMO) → $Y_{mm}^{\rm hydro\,CDM}$ with only the 1–2 per cent
back-reaction, as v0.1 Section 4 says.

### 7.2 Section 5.1's stochastic bound is ~5× too tight

The addendum bounds the second term of (A15) below $6\times10^{-4}$. Measured
across all eight filters, four runs and both redshifts, the worst residual is
$2.9\times10^{-3}$, roughly five times the quoted bound. Still negligible at the
target precision — it enters $S$ at the 0.3 per cent level against the ~6 per
cent from a 20 per cent error on $C$ — but the number should be corrected.

### 7.3 The Section 4.4 consistency test is circular

The addendum proposes measuring $r_{gm}$ and checking whether it equals
$r_{gb}/r_{bm}$. Since $C \equiv r_{bm}r_{gm}/r_{gb}$ identically, that check
*is* $C = 1$, computed from the same four amplitudes on the same maps. It
cannot test mediation independently — it only reports $C$, which is the
measurement. The framing should be dropped; nothing is lost, because the value
of $C$ was always the deliverable.

### 7.4 Convention T is marginally better and costs nothing

$C_t$ against $C$, $z\approx0.5$, $\Delta\Sigma$: mean $|C-1|$ 0.051 vs 0.052,
max scatter 0.047 vs 0.047. The two conventions agree to about one per cent
throughout. The addendum's recommendation — run Route B in Convention T, since
it removes the theory spectrum, the DMO run and the CDM/total conversion at no
physical cost — is confirmed, and it needs no new simulation product: every
Convention T amplitude is a recombination of measured ones.

### 7.5 A small asymmetry between the routes

$C_A$ (Route A) is undefined in one of 36 $\Upsilon(R_0=2')$ bins, where
$Y_{gg}$ goes negative — a compensated filter against a shot-noise-subtracted
galaxy auto. $C$ (Route B) is finite in every bin of every filter, because
$Y_{gg}$ cancels out of it. Minor, but it is one more structural reason to
prefer the four-amplitude form.

---

## 8. What this does not settle

Carried forward unchanged, and none of it is addressed by the present run:

- **Two code families.** ANTILLES remains the natural addition. The
  Singh Appendix A cross-family discipline is still unexercised, and the new
  finding that the residual is feedback-driven makes a third *code* less urgent
  than a wider feedback span — which FLAMINGO already provides and ANTILLES
  would extend.
- **`yz` projection only**; across-projection scatter unmeasured.
- **No bin-to-bin covariance.** 16 jackknife regions cannot support one, and
  the $Y$ transform and $\Upsilon$ correlate bins by construction more than
  $\Delta\Sigma$ does.
- **No beam.** The $r$'s and $C$ here are intrinsic field properties;
  $C_{\rm beam}$ is filter-dependent and would have to be recomputed for any
  filter that enters the analysis.
- **Prediction 3 and the whole matched-$k_{50}$ comparison** (Task 9).
- **The electron/baryon decision** (Section 5.2), which the two-family scatter
  cannot adjudicate.

## 9. Recommended next steps

1. **Cancel** the CDM-only non-linear spectrum work (Section 7.1).
2. **Adopt $\Delta\Sigma$** as the fiducial filter and Convention T for Route B;
   retain $\Upsilon$ as a cross-check; **drop the $Y$ transform** unless the
   matched-$k_{50}$ comparison overturns Sections 4 and 6.
3. **Re-open the Stage 6 electron/baryon decision** with the Section 5.2
   numbers in hand.
4. **Task 9** next, since it is the one place the filter conclusions could still
   move.
5. **Book the feedback dependence of $C$** as a prior width, not as a
   sample-matching artefact.

## Files

**New.** `scripts/cross_corr/make_calibration_factor.py`,
`scripts/configs/cross_corr/calibration_z05.yaml`, `calibration_z026.yaml`,
`scripts/cross_corr/runINT_calibration.sh`, `tests/test_calibration_factor.py`
(17 tests). No existing file modified.

**Outputs.** `data/cross_corr_C/calibration_{run}_{snapshot}_yz.npz`, eight
files, carrying every filter's amplitudes and jackknife stacks, $C$, $C_A$, $x$
and $S$ in both conventions for both gas fields, and the term-B diagnostic.

**Reproduce:**

```bash
cd scripts/
salloc -q interactive -C cpu -N 1 -t 1:00:00 -A desi bash cross_corr/runINT_calibration.sh
cd ../tests/ && pytest test_calibration_factor.py -q
```

## References

As v0.2, plus no additions.
