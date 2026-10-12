# SHAM samples at the companion paper's number densities (unbound gas paper)

**Status: not started.** Plan written 2026-10-12 on the Mac from a read-only
look at this repo (at `21138f0`) and at the paper (`~/overleaf/Unbound_Gas`,
at `47e2ac4`), for an agent on NERSC to carry out. Nothing here has been run.
Statements the Mac session could not check (scratch files, job outputs,
cached stacks) are marked **[verify]**.

## 0. How to use this plan

1. **Step 1 is exploration only** (Sec. 4). Check the facts below on NERSC,
   explore the code and the draft yourself, bring your questions and every
   assumption you would otherwise make to the user, and give them a concrete
   execution plan. **Do not edit code, move files, submit jobs or edit the
   paper until the user approves that plan.** You will have context this
   plan does not have; where you find that something here is wrong, say so.
2. **Stop at the checkpoint** (Sec. 9) after the runs, report, and wait for
   the user before touching the paper text.
3. Commit without pushing, in both repos (the user pushes by hand). Commit
   only your own files. Commit messages short and neutral, with no wording
   about steps, tiers, plans or agents (the user's preference for the paper
   repo; keep it here too).
4. Do not delete files: move them (Sec. 6) or leave them unreferenced.
5. Follow the global commit rules (code-reviewer subagent before code
   commits; latex-reviewer-diff and outside-reader for `.tex`).

## 1. Goal

The companion paper, Liu et al. 2026 (arXiv:2609.22466, the kSZ x
galaxy-galaxy lensing gas fraction; paper repo `~/overleaf/Weak_Lensing_kSZ`,
already public), compares the simulations with its measurement using SHAM
samples at fixed number densities: **5e-4 (cMpc/h)^-3 for the DESI LRGs at
z = 0.5 and 1e-3 for the BGS at z = 0.26** (stellar-mass ranking, parent FoF
haloes above 5e14 Msun/h removed before ranking; its Sec. VI and the configs
`configs/lensing/mass_ratio_{data,noBeam,beamTest}_{z05,z026}.yaml`). Its
beam compensation factor C_beam was also calibrated on those samples.

The unbound gas draft instead uses SHAM samples whose density is fitted per
simulation to the HSC lensing ("lensing-matched", Appendix C, Table II;
`lensing/fit_dsigma_ksz.py`, `data/fit_dsigma_ksz/*.npz`). The user does not
want this paper to contradict the companion. So:

- the SHAM figures of the main text and Appendix A move to the companion's
  densities;
- the kSZ masking figure (Fig. 6) is replaced by the masked Delta Sigma ratio
  (masked ionized gas over the unmasked total matter, stacked on the same
  galaxies), whose dependence on halo mass largely cancels, as the
  companion argues for the unmasked ratio;
- the lensing-matched samples survive only in the tSZ appendix (App. B),
  whose amplitudes depend steeply on halo mass.

## 2. Decisions taken (user, 2026-10-12)

1. **Code:** add a fixed-density option to the scripts that currently take
   the SHAM density only from lensing-fit files (Sec. 5). No fake fit npz.
2. **Fig. 6:** the no-beam version: masked ionized gas at 0.2' without beam,
   over the unmasked 0.2' total matter, with the beam-compensated
   measurement of Liu et al. 2026 on the unmasked column
   (`dsigma_ratio_z05_masked_nobeam.yaml`, commit `8f58cb4`). Fig. 9 follows
   (`dsigma_ratio_z026_masked_nobeam.yaml`, commit `21138f0`). The beamed
   tau-map version is not used.
3. **App. B (tSZ) stays on the lensing-matched samples.** A short
   description of the lensing fit and the z = 0.5 LRG columns of Table II
   move into App. B. App. C and the BGS columns of Table II are removed
   (commented out, Sec. 10.3). App. B's comparison of the tSZ and kSZ
   outside fractions goes (different samples and filters after this change).
   No tSZ reruns: the tSZ masked maps (`tSZ`, `xz`) are separate files.
4. **Robustness:** move the existing lensing-fit masked ionized-gas fields to
   a separate folder before rebuilding (Sec. 6). Compute the 5e-4 vs
   lensing-fit comparisons (f_gas of Fig. 7, and the masked fractions of
   Figs. 6 and 9) and **report them to the user only**, not in the paper.
   The new Fig. 7(b) uses 5e-4.
5. **Simulations:** keep every run of the current configs, including the
   SIMBA-50 runs and TNG100-1 (62 and 210 galaxies at 5e-4).
6. **Scope:** Figs. 5 (bottom row), 6, 7(b) and 9 move to 5e-4 (LRG,
   z = 0.5) / 1e-3 (BGS, z ~ 0.26). Figs. 1-4 and 8 stay on the mass cut.
   Fig. 10 (tSZ) is unchanged. The old Fig. 6 (CAP kSZ amplitudes vs
   Hadzhiyska et al.) and old Fig. 9 (CAP outside fractions) are replaced,
   not kept elsewhere. The halo-selection paragraph of the paper's
   `CLAUDE.md` and `AGENTS.md` is updated.
7. **Checkpoint** after the runs (Sec. 9); wait for the user before the text.
8. **Sec. IV C** is drafted in agent markup (`\arev`) for the user to rework.
9. **Repos on NERSC:** paper at `/global/homes/r/rhliu/overleaf/Unbound_Gas/`
   (`~/overleaf/Unbound_Gas`), this repo at
   `/global/u2/r/rhliu/projects/SimulationStacker`. Commit, do not push.

Smaller choices made by this plan (the user may change them before handoff):

- New configs and runners are **copies** with a density suffix (`_n5e-4`,
  `_n1e-3`); the lensing-fit configs stay as they are, so the old figures
  can still be redrawn with `--from-npz`.
- The moved maps go to `products/2D/masked/lensfit_sham_hosts/` next to the
  originals, with a `MANIFEST.txt` (Sec. 6).
- New paper figure files: `figures/dsigma_ratio_masked_z05.pdf` (Fig. 6) and
  `figures/dsigma_ratio_masked_z026.pdf` (Fig. 9). Figs. 5 and 7 replace
  their files in place (`pk_fstar_obs_z05_fstar_S_ap1_M13.pdf`,
  `fgas_profiles_z05_2D.pdf`), as earlier regenerations did. Old files stay
  in `figures/`, unreferenced.
- The figure labels `fig:kSZ_sim_profiles` and `fig:app_z026_ksz` are kept
  for the new figures, so the existing `\ref`s still resolve.
- The 1e-3 Fig. 9 config drops `max_objects`: at 1e-3 a FLAMINGO box holds
  about 3.2e5 galaxies, below the 4e5 subsample cap **[verify]**.
- The beam-factor block of the no-beam config is dropped from the 5e-4
  config: the published C_beam was computed on these same samples, and the
  block's `beamed_npz` (a lensing-fit tau-map run) would make the script
  stop on the density mismatch.

## 3. Background

### 3.1 Which sample each figure uses now (draft at `47e2ac4`)

| Fig. | label | sample now | after this change |
|---|---|---|---|
| 1, 2 | `fgas_gas_profiles`, `fgas_baryon_profiles` | mass cut | unchanged |
| 3 | `star_fraction` | mass cut | unchanged |
| 4 | `simulation_compare` | mass cut | unchanged |
| 5 | `stellar_pk` | top: rescaling of all haloes >= 1e13; bottom: lensing-matched SHAM | bottom: SHAM 5e-4 |
| 6 | `kSZ_sim_profiles` | CAP tau amplitudes, lensing-matched SHAM, xy | new: masked DSigma ratio, SHAM 5e-4, yz, no beam |
| 7 | `HOD_effects` | (a) mass cut, (b) lensing-matched SHAM | (b) SHAM 5e-4 |
| 8 | `app_z026_fgas` | mass cut (mean 10^13.24) | unchanged |
| 9 | `app_z026_ksz` | CAP outside fractions, BGS lensing-matched SHAM, no FLAMINGO | new: masked DSigma ratio, SHAM 1e-3, with FLAMINGO |
| 10 | `tSZ_sim_profiles` | lensing-matched SHAM (LRG fit) | unchanged |

Mass cut: FoF haloes below 5e14, most massive kept down to a mean FoF mass
of 10^13.22 (`halos.select_massive_halos`). SHAM: subhaloes in parents
<= 5e14 ranked by `SubhaloMStar`, top n x V kept, stacked on `SubhaloPos`
(`stacker.stack_on_array`, `halo_stats.select_sample`).

### 3.2 What 5e-4 selects at z = 0.5 (from `data/fit_dsigma_ksz/fit_dsigma_ksz_z05{,_simba50,_mstar}.npz`)

| run | galaxies | mean host FoF mass [1e13 Msun/h] | chi2 (6 dof) at 5e-4 | at fitted n |
|---|---|---|---|---|
| Illustris-1 | 210 | 2.77 | 114 | 1.93 (n = 1.20e-3) |
| TNG100-1 | 210 | 3.46 | 573 | 1.89 (2.68e-3) |
| TNG300-1 | 4307 | 3.52 | 558 | 2.03 (2.97e-3) |
| SIMBA-100 | 500 | 2.87 | 219 | 1.97 (1.70e-3) |
| SIMBA-50 (3 runs) | 62 each | 3.64-4.21 | 679-946 | 2.10-2.45 |
| FLAMINGO (5 runs) | 157,910 | 3.38-3.61 | 348-510 | 2.05-2.08 |

The mass-cut mean is 1.66e13. So the 5e-4 hosts are about twice as massive
as the mass-cut sample, against about 1.2 times for the lensing-matched
samples. The chi2 column is for your understanding: the user does **not**
want a "5e-4 is disfavoured by the lensing" statement in the paper. The BGS
fit grid has no 1e-3 point, so the z ~ 0.26 masses at 1e-3 are unknown.

### 3.3 Existing code for the new Figs. 6 and 9

- `unbound_gas/simulated_dsigma_ratio_masked.py` (`5d02be1`, `8f58cb4`,
  `21138f0`): `<DSigma_ion^(n)>/<DSigma_tot>/(Omega_b/Omega_m)` for the gas
  within 1, 2, 3 x R200m of the SHAM hosts and unmasked; numerator modes
  `tau_map` (beamed) and `ionized_gas_field` (no beam, used here). Checks
  each masked map's `.sample.json` against the stacked density.
  `--from-npz` redraws from a saved profiles file.
- `unbound_gas/precompute_masked_particles.py` (`8f58cb4`): masks the gas
  particles (exact periodic spheres around the unique parent haloes of the
  SHAM galaxies) and bins them like the cached 0.2' fields; `--keep-all`
  checks against the cached unmasked field; `--out-dir` writes test maps
  outside the cache.
- Runners: `runINT_masked_ionized_gas_particles.sh` (z = 0.5, 3 nodes,
  job 59702660, 49 min), `runINT_masked_ionized_gas_particles_z026.sh`
  (z ~ 0.26, 3 nodes, job 59716174, 54 min), `runINT_dsigma_ratio_masked.sh`
  (stacking, config as argument).
- **The gap:** the density comes only from lensing-fit npz files:
  `simulated_dsigma_ratio_masked.py` stops without `stack.abundance_from_fit`
  (~l.509), and `precompute_masked_particles.py` takes it from `--fit`.
  `make_fgas_profiles.py` (Fig. 7) and the P(k) chain (Fig. 5) already fall
  back to `halo_abundance_target` when `abundance_from_fit` is absent.

### 3.4 The cache-path collision

The masked-field cache path does not record the sample
(`..._ionized_gas_<n>_yz_masked<r>R200c.npy`; "R200c" is historical). A
5e-4 build writes exactly where the lensing-fit fields are. The script skips
existing maps unless `--overwrite`, and the stacking script refuses a map
whose `.sample.json` density differs, so nothing mixes silently; but a
rebuild with `--overwrite` would destroy the lensing-fit fields. Hence the
move in Sec. 6. The tau maps (`xy` for the old Fig. 6, `yz` for the beamed
ratio figure) and the tSZ maps (`xz`) are not touched by this plan.

## 4. Step 1: explore, ask, plan (no execution)

Check at least the following, then write your questions, assumptions and
execution plan for the user, and wait for approval.

- HEAD of both NERSC clones. The user commits on the Mac and pushes by
  hand, so a clone may be behind `21138f0` (this repo) or `47e2ac4` (the
  paper). If either is behind, ask the user to push or pull; do not pull on
  your own (other sessions may have uncommitted work there). Then
  `git log` since those commits: anything that changes this plan.
- The lensing-fit masked ionized-gas fields on scratch
  (`/pscratch/sd/r/rhliu/simulations/{IllustrisTNG,SIMBA,FLAMINGO}/products/2D/masked/`):
  list them with their `.sample.json` (`sample`, `density`, `method`,
  `z`, `pixel_size`, `beam`). Expect 12 runs x 3 radii at z = 0.5 and
  8 runs x 3 radii at z ~ 0.26. **[verify]**
- The saved profiles files of the lensing-fit no-beam runs (z = 0.5 and
  z ~ 0.26) and of the current Fig. 7, under `figures/2026-10/*/`, needed for
  the robustness comparison without restacking. **[verify]**
- The cached unmasked 0.2' `ionized_gas` and `total` fields (yz) for all
  runs at both redshifts. **[verify]**
- For Fig. 5: which `'lens'`-tagged (5e-4) lensing stacks exist from the
  pre-lensfit configs (`pk_components_z05.yaml`, `pk_fstar_obs_z05.yaml`),
  for which runs, and whether their stored settings match what the new
  configs would ask for (the scripts compare them and refuse mismatches).
  The FLAMINGO M*-sigma runs very likely have none. **[verify]**
- Where the numbers of the draft come from (Sec. 9): e.g. the "1.4 times"
  lensing shift (probably the `make_pk_fstar_obs.py` table), the satellite
  fraction (printed by `make_fgas_profiles.py`). **[verify]**
- Whether the companion's simulated f_gas curves (5e-4, yz, 0.2', 0.75'
  annulus) are stored anywhere; the unmasked column of the new Fig. 6 should
  reproduce them for the shared runs, a useful check. **[verify]**
- The draft: read Secs. III A, IV B 3, IV C, IV D, the Discussion and
  Appendices A-C yourself; the line numbers in Sec. 10 are from `47e2ac4`.
- Questions the Mac session already knows to ask the user:
  - whether the moved Table II keeps its chi2 column (in an earlier session
    the user said a chi2 was not necessary; the context was not clear);
  - whether App. B keeps the sentence that the fitted densities are 2.4-6
    times the LRG density, and the `[AUTHOR: explain why ...]` placeholder
    attached to it (both arguably restate the contradiction the user wants
    to avoid);
  - whether `make_unboundGas_plots.sh` (stale figure numbers) should be
    updated.

## 5. Step 2: fixed-density option (code)

- `unbound_gas/precompute_masked_particles.py`: add `--density FLOAT`,
  mutually exclusive with `--fit` (default behaviour unchanged). Record the
  source in the `.sample.json` (e.g. `density_source: 'fixed'` or the fit
  files; keep `sample: 'sham-hosts'` so `require_masked_map`-style checks
  still apply). Update the docstring, which says the maps surround the
  lensing-fit hosts.
- `unbound_gas/simulated_dsigma_ratio_masked.py`: accept
  `stack.halo_abundance_target` as the alternative to
  `stack.abundance_from_fit` (exactly one required); every run then stacks
  at that density. Update the error message and docstring.
- `precompute_masked_sham_hosts.py` is not needed (the beamed route is not
  used); leave it.
- Validation:
  - one small run (SIMBA-100 or TNG100-1) with `--density` set to its
    fitted value and `--out-dir` on scratch must reproduce the moved
    lensing-fit maps of that run (after Sec. 6). Read the density from the
    fit npz and pass it at full precision: the selection keeps
    `int(n * V)` galaxies, so a rounded value can change the sample by one
    galaxy. Compare within the script's own 1e-10 relative tolerance;
  - the default `--fit` path is unchanged (diff review; the lensing-fit
    configs still parse and select the same densities, e.g. a `--from-npz`
    redraw).
- Code-reviewer subagent, then commit.

## 6. Step 3: move the lensing-fit masked fields

- For each suite directory `.../products/2D/masked/`, move the
  `ionized_gas` `yz` masked fields **and their `.sample.json`** whose record
  says `sample: 'sham-hosts'` at a lensing-fit density (z = 0.5 and
  z ~ 0.26 sets) into `.../products/2D/masked/lensfit_sham_hosts/`.
- Move nothing else (no tau, tSZ, or other projections). Write
  `MANIFEST.txt` in each destination: source path, destination, density,
  z, written date, job.
- Use `mv` within the filesystem; verify counts and sizes afterwards.
- Afterwards the lensing-fit Fig. 6/9 configs fail their map check by
  design; redraw them with `--from-npz` if needed.

## 7. Step 4: build the masked fields at the companion densities

- New runners, copies of the two particle runners:
  `runINT_masked_ionized_gas_particles_n5e-4.sh` (z = 0.5, `--density 5e-4`)
  and `runINT_masked_ionized_gas_particles_z026_n1e-3.sh` (z ~ 0.26,
  `--density 1e-3`, FLAMINGO at snapshot 71, z = 0.30; TNG300-1 80,
  Illustris-1 116, SIMBA-100 136 at z = 0.26). Update their header comments
  (the "new files, nothing moved" note no longer holds). Keep the
  `--keep-all` checks on the small boxes.
- Same node layout as before is a reasonable start (the particle pass
  dominates: ~45 min per FLAMINGO variant); fewer hosts at 5e-4 / 1e-3 may
  make it faster.

## 8. Step 5: figures

1. **Fig. 6:** `configs/unbound_gas/dsigma_ratio_z05_masked_nobeam_n5e-4.yaml`
   from `dsigma_ratio_z05_masked_nobeam.yaml`: `halo_abundance_target: 5.0e-4`
   instead of `abundance_from_fit`, no `beam_factor` block, new `fig_name`.
   Run with `runINT_dsigma_ratio_masked.sh <config>`.
2. **Fig. 9:** `dsigma_ratio_z026_masked_nobeam_n1e-3.yaml` from the z026
   config: `halo_abundance_target: 1.0e-3`, no `max_objects` (after
   checking the counts), new `fig_name`.
3. **Fig. 7:** `fgas_profiles_z05_n5e-4.yaml` from `fgas_profiles_z05.yaml`
   without `abundance_from_fit` (its `halo_abundance_target` is already
   5e-4); new `fig_name`. Record the printed satellite fractions.
4. **Fig. 5 (bottom row):** `pk_components_z05_n5e-4.yaml` and
   `pk_fstar_obs_z05_n5e-4.yaml` from the `_lensfit` versions, with the
   lensing block's `abundance_from_fit` removed. Keep the lensing config
   `configs/lensing/mass_ratio_noBeam_z05_mstar.yaml` (5e-4, same stack
   block as `mass_ratio_noBeam_z05.yaml`, so existing `'lens'` stacks stay
   valid) and use `sample_name: 'lens'` to reuse the pre-lensfit 5e-4 stacks
   where their settings match. Compute the missing lensing stacks with
   `stack_stellar_maps.py` and `stack_fstar_obs_maps.py` (template:
   `runCPU_fig6_lensfit.sh`; the spectra and transfer maps are
   sample-independent and cached). Draw with
   `make_pk_fstar_obs_grid.py --layout D` (the paper's layout, sized in
   `cd97e49`) and write the table with `make_pk_fstar_obs.py`.
5. Do not copy anything into the paper yet.

## 9. Step 6: checkpoint report (STOP here)

Report to the user, and add a "Results" section to this file. For each item
give the draft's current claim, its location, and the new value; say plainly
where a qualitative claim no longer holds. Do not interpret or explain the
changes; report them.

- **A. Fig. 7 / Sec. IV D** (l.1157-1165, caption l.1181):
  - per run, the largest |f_gas(SHAM 5e-4) - f_gas(mass cut)| over 1-6'
    (claim: "within about 0.05 at all apertures except SIMBA-100");
  - SIMBA-100's offset (claim: SHAM above mass cut by 0.06-0.09 at 1-5');
  - satellite fractions (claim: 23-28%);
  - mean host FoF mass (claim: "somewhat more massive" than the mass cut).
- **B. Fig. 5 / Sec. IV B 3, abstract l.106, conclusions (ii) l.1285:**
  the largest lensing f_gas shift at 1' from the stellar rescaling, in
  units of the 0.06 uncertainty of the innermost measured point (claim:
  "up to about 1.4 times"); the spread across simulations at 1' (claim:
  0.15-0.53).
- **C. Fig. 6 / Sec. IV C, abstract l.109, conclusions (iii):** the outside
  fraction 1 - f_n/f_unmasked (equal to 1 - DSigma_ion^(n)/DSigma_ion,
  since the denominators are the same) per run at 1' and 6' for
  n = 1, 2, 3; mean host R200m in arcmin. Old claims, for the CAP filter
  on the lensing-matched samples: < 0.1 at 1' except SIMBA-100 (0.30); up
  to about three quarters at 6'; outside 3 R200m < 0.1 at 6'; Illustris-1
  0.74 and IllustrisTNG about half at 6'; R200m 1.3-1.5'; no clear ordering
  among the SIMBA-50 runs. To separate the filter from the sample, also
  give the lensing-fit no-beam values (from its saved profiles) next to the
  old CAP values.
- **D. Fig. 9 / App. A** (l.1448 paragraph): as C at z ~ 0.26, now with
  FLAMINGO (old claims: outside 3 R200m at most 0.09 at 6'; outside
  1 R200m 0.26-0.64 at 6'; R200m 1.5-1.75'); mean host FoF mass at 1e-3
  (replaces "(1.2-1.4)e13").
- **E. Robustness, for the user only:** f_gas of Fig. 7(b) at 5e-4 vs at
  the fitted densities, per run; the outside fractions at 5e-4 vs at the
  fitted densities, at both redshifts.
- **F. Consistency with the companion:** the unmasked column of the new
  Fig. 6 against the companion's published simulation curves for the
  shared runs (TNG300-1, Illustris-1, SIMBA-100, FLAMINGO L1_m9,
  fgas-8sigma, Jet_fgas-4sigma), if they can be found; and a description
  of where each run lies relative to the beam-compensated data.
- **G.** Anything else that surprised you, and the job IDs, run times and
  file paths of every product.

Then wait for the user.

## 10. Step 7: the paper text (only after the user's go-ahead)

### 10.1 Rules

- Load the `paper-edit` skill and follow it, and the paper's `CLAUDE.md`
  (agent markup `\arev{}` / `\areplace{old}{new}`, never touching `\HL{}`,
  `\SF{}`, `\rev{}` or `\replace{}`; struck text without long braced groups,
  `\mbox` around `\cite`/`\ref` inside struck text; `Author et al.~\cite{}`,
  never `\citet`; Canadian spelling).
- **The user has just spent two sessions and a manual pass making the prose
  sound like them**, and has proofread everything except Sec. IV C and
  IV B 2. In the proofread sections (abstract, introduction, III A,
  IV B 3, IV D, Discussion, App. A's first paragraphs) limit yourself to
  deletions, single-word removals ("lensing-matched"), number swaps and,
  where unavoidable, one short clause copied from the user's neighbouring
  wording. Do not reflow or "improve" anything else.
- Sec. IV C, the App. A kSZ paragraph and the Fig. 6/9 captions are rewritten
  in agent markup for the user to rework. For a whole paragraph, use the
  draft's existing pattern: comment out the old paragraph under a
  `% (agent) REPLACED: ... Previous version:` note and put the new text in
  `\arev{}`.
- Do not write scientific interpretations of open questions (why a fraction
  changed, why simulations differ); leave `[AUTHOR: ...]` placeholders, as
  the draft already does. Keep existing `[AUTHOR]` placeholders.
- Acronyms: the sentences to delete in III A define "LRGs" (l.352) and
  "HSC" (l.390). Re-define each at its new first live use (LRG: probably
  the mass-cut paragraph's "mass of DESI LRGs"; HSC: in App. B).

### 10.2 Edits by section (line numbers at `47e2ac4`; re-grep)

- **Abstract** (l.106, l.109): in the outside-gas sentence, "CAP-filtered"
  becomes "$\Delta\Sigma$-filtered" (one word: the outside fraction
  1 - f_n/f_unmasked equals that of the Delta Sigma-filtered kSZ signal,
  since the lensing denominator cancels); numbers per Sec. 9.
- **Introduction roadmap** (l.186): drop the App. C clause.
- **III A** (l.352-392):
  - opener: delete "The number density of this sample is chosen ... as
    described in Appendix~\ref{app:lensing_samples}." and "We refer to this
    sample as the SHAM, or lensing-matched, sample.";
  - SHAM paragraph: give n_g as 5e-4 (Mpc/h)^-3 "as in
    Ref.~\cite{Liu2026_baryons}" (approximately the DESI LRG density,
    `Zhou2023_DESI_LRG`, which App. C currently cites); keep the user's
    sentences on stellar-mass ranking, satellites, "a common number density
    selects haloes of different mass ..." and "For the gas fraction this has
    little effect ...";
  - "However, the amplitudes of the kSZ and tSZ profiles ... does not
    cancel." becomes about the tSZ profiles of App. B only (drop "kSZ and");
    replace "We therefore choose $n_{\rm g}$ separately ..." and
    "Appendix~\ref{app:lensing_samples} describes the fit ..." with one
    short sentence saying that for those profiles n_g is instead matched to
    the galaxy-galaxy lensing (App. B);
  - the masked-profiles paragraph is unchanged.
- **IV B 3** (l.975): drop "lensing-matched"; numbers per Sec. 9 (also in
  the abstract and conclusions (ii)).
- **IV C** (l.1038-1131, `\arev` throughout, not proofread by the user):
  adapt rather than rewrite. The S_n / (1 - S_n/S_unmasked) framing, the
  outside-fraction paragraphs and the painting-model paragraphs carry over
  with CAP replaced by Delta Sigma and the numbers of Sec. 9. What is new:
  the figure's y-axis is now the masked ionized gas over the unmasked total
  matter on the same galaxies, so the profile descriptions (written for CAP
  amplitudes) change; and the data comparison is now with the
  beam-compensated measurement of Liu et al.~\cite{Liu2026_baryons} instead
  of Hadzhiyska et al. The unmasked column is, by construction, the
  companion's own comparison (same sample, projection, pixels, annulus and
  radii): keep it to a sentence that points to the companion's discussion,
  and check any descriptive wording against the companion's Sec. VI (e.g.
  its statement that the fiducial FLAMINGO run tracks the data most
  closely); do not re-derive which simulation matches best. Keep the caveat
  that the simulated maps retain line-of-sight structure that
  velocity-weighted stacks partly decorrelate, if still accurate. The last
  paragraph's tSZ-vs-kSZ comparison goes. SIMBA-50: 62 galaxies each, not
  "a few hundred".
- **IV D** (l.1157-1165, caption l.1181): drop "lensing-matched" (l.1157);
  "its host haloes (Table~\ref{tab:lensing_samples}) are somewhat more
  massive" becomes the measured comparison without the table reference;
  satellite percentage and the 0.05 / 0.06-0.09 numbers per Sec. 9; caption
  "(b) uses ... at the lensing-matched number density of each simulation
  (Appendix~C)" becomes "at $n_{\rm g} = 5\times10^{-4}$". If the checkpoint
  showed a qualitative change, the user will have said how to handle it.
- **Discussion and conclusions:** (iii) (l.1285) "CAP-filtered" becomes
  "$\Delta\Sigma$-filtered", as in the abstract; the paragraph at l.1295-1304 opens with the old Fig. 6
  vs Hadzhiyska et al. comparison, which no longer exists: rewrite its first
  sentence, keep the rest where still accurate, flag anything doubtful.
- **App. A:** l.1421 "SHAM samples at the number density fitted, as in
  Appendix~C, to the galaxy--galaxy lensing of the DESI BGS ..." becomes
  "at $n_{\rm g} = 10^{-3}\,(\mathrm{Mpc}/h)^{-3}$, as in
  Ref.~\cite{Liu2026_baryons}"; replace "The converged fits select host
  haloes of mean FoF mass ..." with the 1e-3 mean mass or delete it; the kSZ
  paragraph (l.1448, agent text) and the Fig. 9 caption follow the new
  figure: FLAMINGO is now included ("We omit FLAMINGO ..." goes), the
  TNG300-1 non-convergence sentence goes, the reference to Table II goes;
  keep the `[AUTHOR]` placeholder.
- **App. B:** "lensing-matched SHAM galaxies (Appendix~C)" in the text and
  caption now points within App. B; add a short paragraph on the fit (the
  density grid; Delta Sigma of the total matter on unconvolved 0.2' maps
  with a 0.3' annulus; HSC Y3 DESI LRGs at 0.4 < z < 0.6, seven bins at
  theta >= 2.25', jackknife covariance; minimum of the interpolated chi2;
  host masses ~2e13; the McCarthy et al. precedent), and the table with the
  z = 0.5 LRG columns for the runs of the tSZ figure (Illustris-1, TNG100-1,
  TNG300-1, SIMBA-100, the five FLAMINGO runs). No "5e-4 is disfavoured"
  sentence. Delete the sentence comparing the tSZ outside fractions with the
  kSZ ones. Leave the `\HL` note on the data points and the data points
  themselves alone.
- **App. C and Table II:** put the whole appendix inside
  `\begin{comment} ... \end{comment}` with a `% (agent)` note saying where
  its content went; drop the BGS columns. First check it for `\HL{}` or
  `\SF{}` notes: a comment block would hide them, which the paper's rules
  forbid, so keep any you find live (move them to App. B and tell the user).
- **References:** every live `\ref{app:lensing_samples}` and
  `\ref{tab:lensing_samples}` must be updated; the build must show no
  undefined references or citations.
- **Paper `CLAUDE.md` and `AGENTS.md`** (keep in sync, `CLAUDE.md`
  authoritative): halo selection (ii) becomes SHAM at 5e-4 (LRG, z = 0.5)
  and 1e-3 (BGS, z ~ 0.26) as in the companion, with the lensing-matched
  samples only in the tSZ appendix; "the lensing fits of Appendix C" becomes
  Appendix B.
- **Figures:** copy the four new PDFs into `figures/` with the names of
  Sec. 2; update the `\includegraphics` lines of Figs. 6 and 9.
- `TODO.md` as the paper-edit skill says.

### 10.3 Build, review, commit

Build outside the repo (paper `CLAUDE.md`: `module load texlive/2024`,
`latexmk` into `$SCRATCH/latex-build/...`), check the log for undefined
references, then latex-reviewer-diff and outside-reader on the changed
sections, then commit (no push) and report the change list to the user.

## 11. Step 8: wrap up

Update this file's status line and add the job IDs, products and decisions
taken along the way; commit it with the configs and runners in this repo.
