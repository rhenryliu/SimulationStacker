# Task 1 Engineering Spec: r-profile computation in SimulationStacker

Companion to `docs/cross_correlation_notes.md` (the theory note), which is
authoritative for the estimator and physics. This document is authoritative
for implementation decisions. If either conflicts with repo reality, flag it
in the plan rather than silently deviating.

## Objective

Compute r_gb(R), r_bm(R), r_ge(R), r_em(R) and the ratio r_bm/r_gb for
filters {Sigma, DSigma, Upsilon(R0=1')} over 9 linear aperture bins in
1'-6', for all six production simulations, three projections each.
Deliverable: the Singh et al. (2020) Fig. 1 analogue plus per-radius
cross-simulation scatter of r_bm/r_gb (Gate A of the theory note, Sec. 8).

## Core algorithmic decision (settled, do not re-derive)

All Y_XY(R; F) are computed by ONE operation on periodic 2D maps:
FFT-convolve field X (as overdensity) with the pixelized aperture kernel,
multiply pixelwise by field Y (as overdensity), take the map mean.

Rationale: stamp-stacking a filtered map at galaxy positions is
mathematically identical to <F_R[map] * delta_g> on the periodic box, so
this route computes galaxy-crossed AND field-field (bm, mm, bb) pairs
uniformly. The field-field pairs are the ones the existing stacker cannot
produce; the galaxy-crossed case doubles as the validation hook against
`stackMap`.

Numerically verified facts (synthetic-map tests, to be ported to pytest):
- Kernel built with `delta_sigma_kernel`'s pixel-count normalization
  (disk +1/(pixArea*N_disk), annulus -1/(pixArea*N_ann), membership by
  pixel-centre radius) matches the stamp filter to machine precision
  (max abs diff ~1e-16 on lognormal test maps).
- <F[X] * delta_g> equals the stamp-stacked mean at pixel-aligned galaxy
  positions to float precision (NGP-deposited count map).
- Analytic self-pair subtraction for the galaxy auto, subtracting
  K(0) * pixArea / nbar_pix per aperture, takes Poisson (unclustered)
  Y_gg to within ~1 sigma of zero.

## Fixed numerical conventions

- Overdensity convention: every field enters as delta = X/<X> - 1;
  all Y's dimensionless (theory note Sec. 5.4).
- Filter parameters: dr = 0.75 arcmin annulus; Upsilon R0 = 1 arcmin;
  radii = 9 linear bins over 1-6 arcmin; arcmin-to-comoving conversion via
  the existing comoving transverse distance helper (the f_gas paper Eq. 27
  convention), evaluated at the snapshot redshift.
- Sigma filter = annulus mean over [R, R+dr] (the local-Sigma estimate),
  positive weights only, uncompensated.
- NO beam anywhere in this task: the r's are intrinsic field properties.
  Beam forward-modelling stays in the existing f_gas machinery.
- Galaxy maps: NGP deposition of SHAM-selected subhalo positions
  (method='abundance', target 5e-4 (cMpc/h)^-3 for the LRG-like sample,
  1e-3 for BGS-like), same grid and projection as the particle maps.
  NGP, not TSC: the point process shot noise is subtracted analytically
  and a smoothing window would alias into the aperture kernel.
- CDM field = pType 'dm'; ionized gas = 'ionized_gas'; total baryons =
  'baryon' (verify stars and BH are included for all three simTypes).

## Errors

Spatial patch jackknife, 4x4 blocks on the product map, leave-one-out.
CRITICAL: the three Y's entering each r are measured on the same map and
are strongly correlated; form r per jackknife realization and take the
spread. Never propagate marginal Y errors Gaussianly.
The three projections (xy, xz, yz) are quasi-independent realizations;
report both within-projection jackknife and across-projection scatter.
Known limitation, acceptable for Gate A: patch jackknife underestimates
cosmic variance at large apertures; Gate A is about cross-CODE scatter,
which this does not affect.

## Resolution requirements

Pixel size must be well below the smallest aperture (1 arcmin at the
snapshot redshift). Production nPixels=1000 on TNG300 (~0.2 Mpc/h/pixel)
is marginal at z=0.5; use >= 2048 for TNG300/Illustris/SIMBA and ~5000
for FLAMINGO L1 (1 cGpc). Kernel construction must raise on an empty disk
or annulus, never silently degrade. Include one convergence check
(2048 vs 4096 on TNG300-1) before the full sweep.

## Acceptance tests (pytest, synthetic maps, no simulation data needed)

1. Convolution kernel vs a direct stamp reimplementation of
   `delta_sigma_kernel` at random centres: max abs diff < 1e-12.
2. correlate(field, galaxy_count_map) == mean of the filtered field at
   galaxy pixels, to float precision, for DSigma and Upsilon.
3. Poisson galaxy auto: |Y_gg| < 3 sigma after self-pair subtraction at
   two apertures.
4. r of two fields constructed with known mixing (f2 = a*f1 + noise)
   recovers the analytic r within jackknife errors.

## Integration test (needs NERSC data)

Y_gb via this module vs the existing `stackMap('tau', ...)` route (no
beam) on the identical SHAM sample, TNG300-1 snapshot 67: agreement to
sub-percent. Residuals isolate sub-pixel-centring and cutout-edge
conventions and should be documented, not hidden. Note `stack_on_array`
centres cutouts at integer pixels; small discrepancies are diagnostic.

## Repo constraints

- Follow CLAUDE.md: cosmodesi environment, YAML-config script pattern,
  do not refactor hardcoded /pscratch paths, do not touch tools.py numba
  internals. Google-style docstrings throughout.
- Reuse, do not duplicate: `halos.select_halos`, `loadSubHalos`,
  `tools.hist2d_numba_seq` (galaxy map deposition), the existing
  arcmin/comoving conversion, `snr.py` statistics helpers.
- FLAMINGO: only snapshot 67 (z=0.5) is downloaded; the BGS-like sweep at
  z~0.3 requires snapshot 71 and is out of scope until downloaded.
- The existing 3D machinery (`_compute_pk_3D`/`_compute_corr_3D` with
  field_b) already supports cross-spectra; a 3D r_bm(k) script is a cheap
  optional add-on for connecting to the suppression mapping, not part of
  the core deliverable.