# Filter and Normalization Specification

Phase 0, item 1 of `docs/cross_correlation_notes.md` Sec. 8: the single source
of truth for the filter definitions, the discretization, and the normalization
and projection conventions, shared by the data, theory and simulation code
paths. Everything downstream of Gate B uses exactly what is written here.

Companion documents: `cross_correlation_notes.md` (theory, authoritative for
the estimator), `r_profiles_task1_spec.md` (the Task 1 engineering spec) and
`r_profiles_implementation_plan.md` (what was built and measured).

---

## 1. Gate B: the frozen filter set and field definition

**Filter set: {ΔΣ, Υ(R₀ = 1′)}.** Σ is computed and reported as a diagnostic
but is not part of the analysis.

**Field definition: the target is `P_em/P_mm`** — the ionized-gas (free
electron) field crossed with CDM — rather than `P_bm/P_mm` with a calibrated
electron-to-baryon transfer.

Both decisions and the evidence behind them are in
`r_profiles_implementation_plan.md` sections T2 and T3. In brief: the
annulus-mean kernel is uncompensated, so its *amplitude* integrates power down
to the fundamental mode of whatever volume it is measured in and does not port
between boxes (8.7 per cent shift under the measured test), which is
disqualifying for the theory transfer even though its *coefficients* are
largely unaffected; and the electron-to-baryon correction is large and varies
by 40 per cent between feedback variants of a single code, so targeting the
field the kSZ actually measures avoids carrying it.

---

## 2. Filter definitions

With Σ̄(<R) the disk mean and Σ̄(R, R+δR) the annulus mean:

    ΔΣ(R)      = Σ̄(<R) − Σ̄(R, R + δR),        δR = 0.75 arcmin
    Υ(R; R₀)   = ΔΣ(R) − (R₀/R)² ΔΣ(R₀),      R₀ per config (see below)
    Σ(R)       = Σ̄(R, R + δR)                  [diagnostic only]
    Y(R; Rmax) = Σ(R) − Σ(Rmax)                [diagnostic only]

The last is the Park et al. (2021) transform added by the v0.2 addendum. It is
**not** part of the frozen set: the v0.3 response Sec. 5.3 recommends against
adopting it, because its best-performing configuration (Rmax = 9′) references
an aperture outside the data range. It is computed and plotted so the filter
comparison stays visible, on `Rmax = 6′` — the top of the data range, and
already an aperture-grid point, so adopting it moves no bin.

The annulus mean, not the local value Σ(R), is the second term of ΔΣ. This
matches the existing kSZ pipeline; Singh et al.'s formulas use the local value
and are the template, not the specification.

**Υ carries information only above R₀.** That is the point of the filter — it
nulls everything below its reference radius — and it makes every bin with
R ≤ R₀ unusable, in two distinct ways:

- at R = R₀ the amplitude is identically zero, so the coefficient is a genuine
  0/0;
- below R₀ the factor (R₀/R)² exceeds one and the reference term
  over-subtracts. The amplitude is finite and the coefficient is defined, but
  it is not the estimator's quantity. In the production runs it simply flips
  sign: measured at R₀ = 2′, both `r_gb` and `r_bm` came out near −1 at
  R = 1′ and 1.625′.

**The Y transform is the mirror image**, at the other end of the grid:
`Y(Rmax; Rmax) ≡ 0`, and just below Rmax it is a small difference of two
comparable annulus means carrying almost no signal. Bins with R ≥ 0.8·Rmax are
therefore dropped, following the addendum's Secs. 2.5 and 8.2.

`rprofiles.compute_Y_matrix` returns the raw algebra and does not special-case
any of this. `rprofiles.upsilon_defined_mask(radii, r0)` and
`rprofiles.ytransform_defined_mask(radii, rmax)` are the single definitions of
which bins survive, and `plot_r_profiles.series` applies them once so the
curves and the Gate A metrics can never disagree about it.

Both reference radii are **not** global constants: they are read from each
run's `meta_r0_arcmin` and `meta_ytransform_rmax`. R₀ moved from 1′ to 2′ in
commit 91e39d7 and is now **back at 1′**, on the evidence of the R₀ scan in
`r_profiles_implementation_plan.md` U6: 1′ gives the smallest cross-code
scatter (0.090, the only PASS at z ~ 0.5, against 0.143 at 2′), the most usable
bins (8 against 7 over 1′–6′), and is the one R₀ whose statistic survives the
trim unchanged. This restores agreement with Sec. 1 above, which never stopped
specifying R₀ = 1′.

## 3. Aperture grid

    data-matched:  9 linear bins, R = 1.0 to 6.0 arcmin, spacing 0.625 arcmin
    extension:     6.625 to 9.75 arcmin at the same spacing (diagnostic)

The data-matched bins are held fixed: changing the extension must never move
them. Apertures below 1 arcmin are not used, because the data are
resolution-limited there.

Filters are evaluated at **point radii**, not averaged over bin widths, in
simulation and in theory alike. `kernels.bin_averaged_kernel_ft` exists for the
data chain if a finite bin width becomes unavoidable there.

## 4. Discretization

- **Membership** is by pixel-centre radius, with the strict test `r < edge`.
  Disk: `r < R`. Annulus: `R <= r < R + δR`.
- **Weights** follow `filters.delta_sigma_kernel`: `+1/(pixArea · N_disk)` in
  the disk and `−1/(pixArea · N_ann)` in the annulus, with `N` the *pixel
  count*, not the geometric area. The compensated kernel therefore sums to
  exactly zero.
- **An unresolved aperture must raise**, never silently degrade: an empty disk
  or annulus is an error.
- **Boundary degeneracy.** When an aperture edge coincides with a realizable
  lattice distance `sqrt(i² + j²)` an entire shell of pixels sits on the
  boundary and flips membership under an arbitrarily small change of
  convention. At a 0.2 arcmin pixel this affects R = 1′ (5.0 px), the R = 2.25′
  annulus edge (15.0 px) and R = 6′ (30.0 px), each a 12-pixel shell.
  `rprofiles.degenerate_apertures` predicts the affected radii and the
  production run logs them. The effect is ~6 per cent on an amplitude and
  ~0.6 per cent on a coefficient.
- **Resolution.** The pixel must be well below the smallest aperture. All
  production grids are ~0.2 arcmin, so R = 1′ spans 5 pixels and the 0.75′
  annulus spans 3.75 pixels. The annulus, not the disk, is the limiting
  element.

## 5. Normalization

Every field enters as a dimensionless overdensity, `δ = X/⟨X⟩ − 1`, so every
amplitude is dimensionless in the sense of Sec. 5.4 of the theory note.

**One wart, recorded because it does not cancel everywhere.** The weights of
Sec. 4 carry `1/pixArea` where a plain mean would not, so a measured amplitude
is larger than the continuum integral

    Y(R) = ∫ (k dk / 2π) P_2D(k) Ŵ(k; R)

by exactly `1/pixArea`. This cancels in every coefficient
`r = Y_XY / sqrt(Y_XX Y_YY)` and so never mattered for Tasks 1 to 3. It must be
restored when comparing theory to measurement; `kernels.normalize_to_pipeline`
does that.

## 6. Projection convention

This is the convention that Task 4 forced into the open, and the one most
easily got wrong, because it is invisible in every coefficient and decisive in
Eq. (4).

**Amplitudes carry the projection depth.** For a fully projected periodic box,
`δ_2D = (1/L) ∫ dz δ_3D` selects the `k_z = 0` mode, giving

    P_2D(k⊥) = P_3D(k⊥) / L.

Verified directly on a TNG300-1 3D field: `P_2D · L / P_3D` = 0.98 to 1.05 over
k = 0.1 to 6 h/Mpc. It is also why the measured `Y_mm` of TNG300-1 and
FLAMINGO differ by 3.28 to 3.40 against their box ratio of 3.32.

Two projections are therefore defined, and they are **not** interchangeable:

1. **Simulation validation** uses the full box depth, `P_2D = P_3D / L`, with
   `L` the box side of the simulation in question. This is what
   `theory.project_periodic_box` implements and what
   `check_theory_transfer.py` validates against.
2. **The data chain** uses the cylinder that `Y_gg` is measured in,
   `Π_max = 100 h⁻¹ Mpc` (theory note Sec. 5.1), and the kSZ side must be
   converted to match rather than the reverse, because the kSZ integrates the
   entire line of sight.

Since Eq. (4) combines `Y_gb / sqrt(Y_gg Y_mm)` — a net one power of Y — the
convention does not cancel there as it does in the coefficients. Every Y
entering Eq. (4) must use projection convention 2, consistently.

**The coefficients are the portable product.** `r_bm/r_gb` is a ratio in which
both the `1/pixArea` factor of Sec. 5 and the projection depth of this section
cancel. That is why the simulation calibration transfers to the data even
though the amplitudes do not.

## 7. Self-pair exclusion

The galaxy field is a discrete point process deposited by nearest grid point.
Each object contributes exactly one self-pair, so the shot-noise term removed
from the galaxy auto-correlation is

    K(0) / n̄_pix,

with `n̄_pix` the mean galaxy count per pixel. This is exact for any point
process, independent of clustering. It is verified in the test suite by
nulling an unclustered Poisson sample.

It is not a small correction: on FLAMINGO it removes 6.9 times the retained
signal at R = 1′ and 1.9 times at 6′, so `Y_gg` at small apertures is a
difference of comparable numbers and any error in the shot-noise model
propagates directly into `r_gb`. The same caveat applies on the data side
(theory note Sec. 5.1, self-pairs and fibre incompleteness).

## 7a. The coefficients are not bounded by one

`r = Y_XY / sqrt(Y_XX Y_YY)` looks like a correlation coefficient, and the
measured `r_bm` sits slightly *above* unity — 1.006 to 1.013 across the
apertures for TNG300-1, for Σ and Υ alike. That is not a bug and not noise.

Cauchy-Schwarz bounds `|r| ≤ 1` only when the map
`(X, Y) ↦ Y_XY` is a positive-semidefinite bilinear form, which requires
`Ŵ(k) ≥ 0` for every k. None of the three kernels satisfies that: measured over
k = 0 to 40 arcmin⁻¹ at R = 1′, `Ŵ` is negative over 51 per cent of the range
for Σ, 45 per cent for ΔΣ and 50 per cent for Υ. All three oscillate, because
they are built from `J₁`.

So `r > 1` is permitted by the algebra and should not be read as a failure of
the measurement. Singh et al. (2020) Fig. 1 shows the same thing on the lensing
side, with `r_cc` reaching ≈ 1.3 for Υ. What the quantity remains is a
well-defined, convention-free ratio of filtered amplitudes — which is all the
estimator of Eq. (4) needs.

## 8. Errors

Spatial block jackknife, 4×4 = 16 leave-one-out patches on the product map.
Every coefficient and every ratio is formed **per jackknife realization**
before the spread is taken. The three amplitudes entering one coefficient are
measured on the same map and are strongly correlated, so Gaussian propagation
of marginal errors would badly misestimate the uncertainty and would miss the
partial cancellation that makes the ratio well behaved.

Sixteen regions support per-radius error bars but not the inversion of a
covariance across the aperture bins; a covariance would need more regions
(Hartlap requires more resamplings than bins + 2).

## 9. Adopted approximations

Recorded so they are booked rather than forgotten.

| approximation | status | size |
|---|---|---|
| `P_mm^hydro-CDM / P_mm^DMO = 1` | adopted; no DMO run on disk | 1-2 per cent (van Daalen 2011, Chisari 2018) |
| halofit returns total matter, the pipeline's `m` is CDM | measured, not assumed | up to 8.9 per cent on `Y_mm` (ΔΣ, TNG300-1) |
| `n_s`, `sigma8` from published cosmologies | no header carries them | literature values, see `theory.SIMULATION_COSMOLOGIES` |
| no beam anywhere | deliberate; the r's are intrinsic field properties | beam forward-modelling stays in the f_gas machinery |
| `yz` projection only | `xy`/`xz` need full particle sweeps | across-projection scatter unmeasured |
