# FRB DM–galaxy cross-correlation in the simulations: record

**Append-only.** Ground truth for the simulated McCarty et al. (arXiv:2608.11296) Figure 5. Predictions, written before the run: `predictions/2026-10-20_frb_dm_xi.md`.

## 1. Task and method (2026-10-20)

Simulate the paper's Figure 5, $\xi_{g,\delta{\rm DM}}(R)$ in pc cm$^{-3}$ over 15 log bins in 0.3–30 cMpc, with this programme's map-level estimator (D-01) on single snapshots:

$$
\xi_{g,\delta{\rm DM}} = (1+z)\,\bar N_e\,\langle\delta_e\rangle_{[a,b)},\qquad \xi^{\rm NFW} = (1+z)\,\bar N_e\,\langle\delta_m\rangle_{[a,b)},
$$

with $\langle\cdot\rangle_{[a,b)}$ the annulus mean around the galaxies (the `rprofiles` $\Sigma$ kernel with $R=a$, $dr=b-a$, times the pixel area), $\bar N_e$ the mean comoving free-electron column of the projected `ionized_gas` map, $e$ = `ionized_gas`, $m = t - b$ (D-02). The paper's pair estimator minus randoms is the annulus mean of the mean-subtracted DM map; the $(1+z)$ is its Appendix A constant-redshift approximation. Derivation and conventions in the module docstring of `scripts/cross_corr/make_frb_dm_xi.py`.

User decisions (2026-10-20), see `decisions.md` D-22: the repo's BGS SHAM sample ($10^{-3}\,(h^{-1}{\rm cMpc})^{-3}$, parent cap $5\times10^{14}\,M_\odot/h$, satellites included), no stellar-mass sub-cuts; FLAMINGO L1_m9 fiducial, fgas$-8\sigma$, Jet_fgas$-4\sigma$ at snapshot 71 ($z=0.300$) and TNG300-1 at snapshot 80 ($z=0.261$, no $z=0.3$ snapshot on disk); `yz`; no data overlay.

## 2. Run

| item | value |
|---|---|
| job | 60078819, interactive QOS, 1 node; COMPLETED 0:0 in 5 min 18 s: check A step 1 min 09 s (MaxRSS 24.7 GiB), full-run step 4 min 01 s (MaxRSS 35.6 GiB); per run 0.4 min (TNG300-1) and 1.1–1.2 min (each FLAMINGO run) |
| log | `/pscratch/sd/r/rhliu/logs/frb_dm_xi-20261020-231046.log` |
| runner, config | `scripts/cross_corr/runINT_frb_dm_xi.sh`, `scripts/configs/cross_corr/frb_dm_xi_z03.yaml` |
| outputs | `data/frb_dm_xi/frb_dm_xi_{TNG300-1_80, L1_m9_fiducial_71, L1_m9_fgas-8sigma_71, L1_m9_Jet_fgas-4sigma_71}_yz.npz`; figure `figures/2026-10/10-21/frb_dm_xi_z03.pdf`, redrawn on 2026-10-21 from the npz files with the paper's axes ($\delta$DM in pc cm$^{-3}$, i.e. $\xi_{g,\delta{\rm DM}}$, on a linear scale, against $r$ in cMpc) |
| grids | TNG300-1 $4822^2$ (0.0628 cMpc pixels; innermost bin 1.72 pixels wide); FLAMINGO $14015^2$ (0.0714 cMpc; 1.51 pixels) |
| samples | TNG300-1 8,615 galaxies, $M_*$ $9.0\times10^{10}$–$3.4\times10^{12}\,M_\odot$ (median $1.40\times10^{11}$); FLAMINGO 315,821 each, minimum $M_*$ 1.66 / 1.72 / 1.46 $\times10^{11}\,M_\odot$ (fiducial / fgas / Jet), medians 2.40 / 2.46 / 2.05 $\times10^{11}$ |
| errors | $4\times4$ spatial block jackknife |

## 3. Acceptance checks

| check | result |
|---|---|
| A. Regression against `data/r_profiles/r_profiles_TNG300-1_80_yz.npz` ($\Sigma$ amplitudes $Y_{eg}$, $Y_{gm}$, 15 apertures $1'$–$9.75'$, $dr=0.75'$) | max relative difference $4.4\times10^{-16}$ for both; $N_g$ 8,615 in both; **PASS** |
| B. Mean electron column over the fully ionized cosmic value $\Omega_b\rho_{\rm crit}/(\mu_em_p)$ through the box | TNG300-1 0.952; FLAMINGO 0.914 / 0.907 / 0.915; all in [0.70, 1.00]; **PASS** |
| C. `tests/test_frb_dm_xi.py`: painted rings recovered exactly; wiring against `compute_Y_matrix`; equality with an explicit galaxy loop; jackknife realizations equal leave-one-block-out means on unequal blocks; the electron column from a cosmic-density map in Msun/h per (ckpc/h)$^2$ equals the one computed in g cm$^{-3}$; `mean_electron_column` inverts `mapMaker.ionized_gas_masses` (literal constants); the parsec (astropy) and the pixel size | 5 passed at the run; 9 after the code review (four tests added, 2026-10-20); **PASS** |
| D. $\xi^{\rm NFW}$(1 cMpc), fiducial, interpolated linearly in $\ln R$ | 43.4 pc cm$^{-3}$ (42.7 log-log), in [3, 60]; **PASS** |
| Closure (added in review, not a prediction): mean DM through each box rebuilt from $\Omega_b\rho_{\rm crit}/(\mu_e m_p) = 0.2228$ pc cm$^{-3}$ per cMpc (FLAMINGO; 0.2204 for TNG300-1's $\Omega_b h^2$), the check B fraction, the box depth and $(1+z)$ | 264.684 and 80.059 pc cm$^{-3}$, equal to the stored `meta_mean_dm_box` |

## 4. Results

Bin centres (geometric) 0.35, 0.475, 0.646, 0.879, 1.19, 1.62, 2.21, 3.00, 4.08, 5.54, 7.54, 10.2, 13.9, 18.9, 25.7 cMpc. Values from the npz files; $\xi$ in pc cm$^{-3}$.

| run | $\xi_e$ at 0.35 | at 1.19 | at 25.7 | $\xi_e/\xi^{\rm NFW}$ at 0.35 | at 0.88 | max (where) | at 13.9–25.7 |
|---|---|---|---|---|---|---|---|
| FLAMINGO fiducial | $90.3\pm2.1$ | $36.2\pm1.0$ | $2.07\pm0.29$ | 0.625 | 0.980 | 1.064 (1.62) | 0.964–0.965 |
| FLAMINGO Jet | $67.9\pm1.5$ | $30.6\pm0.8$ | $2.02\pm0.27$ | 0.494 | 0.821 | 1.015 (3.00) | 0.966–0.969 |
| FLAMINGO fgas$-8\sigma$ | $60.4\pm1.4$ | $32.1\pm0.9$ | $2.07\pm0.29$ | 0.428 | 0.802 | 1.061 (2.21) | 0.963–0.966 |
| TNG300-1 | $129.2\pm10.6$ | $36.2\pm3.6$ | $2.15\pm0.55$ | 0.846 | 1.069 | 1.080 (1.19) | 0.986–0.989 |

Jackknife errors on $\xi_e/\xi^{\rm NFW}$ (formed per realization) are 0.001–0.005 for FLAMINGO and 0.002–0.009 for TNG300-1. The fgas$-8\sigma$/fiducial ratio of $\xi_e$ rises from 0.669 (0.35 cMpc) through 1.007 (2.21) to a peak of 1.028 (4.08) and is 0.998–1.004 beyond 12 cMpc; Jet/fiducial rises from 0.752 to 1.007 (5.54) and is 0.975–0.987 beyond 12 cMpc. All $\xi$ are positive.

## 5. Predictions scored

| prediction | outcome |
|---|---|
| P1: $\xi_e/\xi^{\rm NFW}$ within $\pm0.10$ of 1 in every bin above 10 cMpc | **pass**: 0.963–0.969 (FLAMINGO), 0.986–0.989 (TNG300-1) in the three bins wholly above 10 cMpc. Prior 0.93–1.00 held. |
| P2: innermost-bin $\xi_e/\xi^{\rm NFW} < 0.85$ in every hydro run | **pass**, narrowly for TNG300-1 (0.846). Prior 0.4–0.7 held for FLAMINGO (0.43–0.63), not for TNG300-1. |
| P3: fiducial $>$ Jet $>$ fgas$-8\sigma$ in $\xi_e$ over 0.3–3 cMpc | **fail**: holds in the first 3 bins (to 0.75 cMpc) only; fgas$-8\sigma$ exceeds Jet from 0.88 cMpc and the fiducial from 2.21 cMpc. Note: the predictions file says "8 of the 10 bins", but 0.3–3 cMpc holds 7 whole bins (8 with the straddling one); the outcome fails under any count. |
| P4: fgas/fiducial further from 1 at 0.3–1 cMpc than at 10–30 cMpc | **pass**: 0.67–0.81 against 0.998–1.004. Prior held. |
| P5: TNG300-1 $\xi_e/\xi^{\rm NFW}$ above fiducial's below 1 cMpc | **pass** in all four bins (0.85–1.07 against 0.63–0.98). |

## 6. Observations (provisional readings, not tested)

- **Electrons exceed the NFW expectation at 1–3 cMpc in every run** (peak $\xi_e/\xi^{\rm NFW}$ 1.015 for Jet, 1.06–1.08 for the others, beyond the jackknife errors), after the central deficit. A reading: gas removed from the inner haloes sits just outside them. Not tested here; the CDM field itself is also feedback-modified (back-reaction, D-21), which enters the denominator.
- **Artefact checks on that excess (2026-10-20).** The ratio is independent of $\bar N_e$ and of the kernel normalization, and both fields pass through the same kernel, so neither the unit conversion nor the aperture discretization can produce it. The committed r-profiles (`data/r_profiles/r_profiles_*_{80,71}_yz.npz`, same SHAM sample, fixed $0.75'$ annuli instead of log bins) give the same peak in $Y_{eg}/Y_{gm}$ ($\Sigma$): 1.082 (TNG300-1, near 1.2 cMpc), 1.066 (fiducial, 1.6), 1.063 (fgas$-8\sigma$, 2.3), 1.017 (Jet, 2.5), so it is not a product of the new binning (the cMpc positions use an approximate flat $\Omega_m = 0.3$ conversion, for labelling only). All baryons show it too, slightly larger: $Y_{bg}/Y_{gm}$ peaks at 1.082–1.086 (1.041 for Jet). Both checks use the same estimator, so they do not exclude an artefact shared by every $\Sigma$ amplitude; none is known.
- **P3's failure has the same shape**: the strongest-feedback run has the most electrons at 2–5 cMpc. Same untested reading.
- **The large-scale plateau is 0.964 in FLAMINGO and 0.987 in TNG300-1, not 1.** Free electrons exclude stars and star-forming gas, which sit in haloes. A large-scale electron bias slightly below the CDM's is one reading; the 0.96 against 0.99 difference between codes is unexplained.
- **The plateau, a reading to test.** If all baryons trace the CDM on large scales ($\delta_b \to \delta_m$), removing a non-ionized fraction $f_n$ with bias $b_n$ gives a plateau of $(1 - f_n b_n)/(1 - f_n)$. Check B gives $f_n$ = 0.086 (FLAMINGO fiducial) and 0.048 (TNG300-1), so $b_n$ = 1.38 and 1.26: the cross-code difference would come mostly from $f_n$. Test: add $\langle\delta_b\rangle/\langle\delta_m\rangle$ (the `baryon` map is already loaded) and check it tends to 1 at large $R$. Not run.
- Cross-code amplitudes at large $R$ are uncompensated $\Sigma$-type amplitudes compared between a 205 cMpc/$h$ and a 1 cGpc box, so they carry D-04's box-scale caveat; the per-run ratios to NFW largely cancel it.

## 7. What this is not, and resolution limits

- **Pixelization at small $R$.** The innermost bins are 1.5 (FLAMINGO) and 1.7 (TNG300-1) pixels wide and hold about 47 and 61 pixels by area. The area-weighted radius of each annulus differs from the plotted geometric centre by about 2 per cent, plus bin-to-bin pixel jitter. The jackknife does not include this. It cancels in the ratio to NFW, not in the absolute $\xi$. Bin edges within 0.01 pixel of a lattice shell: at 0.754 cMpc (TNG300-1, a 4-pixel shell) and at most edges beyond 2.5 cMpc in both suites, where shells of 8–32 pixels are dense and hold a negligible share of each annulus. `make_frb_dm_xi.py` now logs them; it did not during job 60078819.
- **TNG300-1 errors at large $R$.** Its $4\times4$ jackknife blocks are 75.7 cMpc on a side against annuli reaching 30 cMpc, so its errors above about 15 cMpc are probably underestimated. FLAMINGO's blocks are 250 cMpc.
- **Code changed after the run.** The review (2026-10-20) added helpers, edge logging, a NaN guard in check A and plot fallbacks; none changes the stored numbers. The figure was redrawn from the npz files with the reviewed script (`--plot-only`).

Not a reproduction of the paper's lightcone measurement: single snapshots at $z = 0.30$ / $0.26$ against the paper's mean galaxy redshifts 0.17–0.22; no FRB-redshift weighting of pairs; a SHAM sample far sparser and more massive than the paper's 27.3 million BGS galaxies (all galaxies here have $M_* > 9\times10^{10}\,M_\odot$, so the paper's stellar-mass bins collapse to one); no data comparison. The figure compares simulations with each other.

## 8. Next steps (not started)

- A denser sample closer to the paper's "all BGS" (higher SHAM density, or the paper's lensing-mass-tuned stellar-mass cuts with the halo masses of their ref. [49]).
- A lower-redshift FLAMINGO snapshot (0.17–0.22), or the lightcone DM maps for a like-for-like comparison.
- A stamp-route cross-check of a few bins with the `'Sigma_lensing'` filter on mean-subtracted maps (D-01 equivalence is already established; this would only confirm the log-bin wiring).
