# `DSigma_lensing`: the lensing-bin ΔΣ filter (pointer)

**The write-up lives in the DESIxHSC-Lensing repository, so there is one version to keep up
to date:** `docs/ksz_lensing_filter_consistency.md` in
<https://github.com/rHenryLiu/DESIxHSC-Lensing> (on NERSC:
`/global/homes/r/rhliu/projects/DESIxHSC-Lensing/docs/ksz_lensing_filter_consistency.md`).
It explains why the kSZ/lensing comparison needs this filter, derives it, gives the validation
and the remade Figure 14, and lists the next steps. Nothing below repeats it.

## What it is

`filterType='DSigma_lensing'` returns, for each stacking radius, the area-weighted average of
ΔΣ(R) = Σ̄(<R) − Σ(R) over a lensing bin: the quantity a weak-lensing ΔΣ measurement reports.
The bins are the edges whose geometric means are the stacking radii, built exactly as the
DESI × HSC lensing bins are. It is used as the simulation denominator of the f_gas comparison,
in place of the 0.75′ disk-minus-ring `'DSigma'` filter on total mass.

## Where it is in this repo

- `src/filters.py`: `delta_sigma_lensing_weights` (with `pixel_disk_overlap`) builds the
  per-pixel weights.
- `src/stacker.py`: `stack_on_array` applies them, centred on each halo's true position. With
  `stampRes` (arcmin; `stackMap` passes it, config key `stack.stamp_res`) it applies them instead on
  a stamp bilinearly interpolated from the map around each halo, as ThumbStack filters the data
  (step 1c; `SimulationStacker.interpolated_stamp`). The option C configs in
  `scripts/configs/lensing/dsigma_lensing/` set `stamp_res: 0.25` for the beamed numerator.
- `tests/test_delta_sigma_lensing.py`: known-answer tests.
- `scripts/configs/lensing/mass_ratio_data_{z05,z026}_dsigma_lensing.yaml`: Figure 14 with
  `filter_type_2: 'DSigma_lensing'`. The paper configs and `make_*_plots.sh` are unchanged.

## Not to be confused with

- `'DSigma'`: the disk-minus-ring filter specified in `docs/cross_corr/filter_specification.md`
  for the kSZ × clustering programme. This change does not alter it.
- `'DSigma_mccarthy'`: the deprecated per-pixel ΔΣ estimator in `filters.py`.
