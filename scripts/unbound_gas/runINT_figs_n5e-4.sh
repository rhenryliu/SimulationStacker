#! /bin/bash -l

# Unbound gas paper figures on the SHAM samples at the fixed densities of
# Liu et al. 2026 (5e-4 (cMpc/h)^-3 for the LRGs at z = 0.5, 1e-3 for the BGS
# at z ~ 0.26) instead of the lensing-fit densities. Stacking only: needs the
# masked fields of runINT_masked_ionized_gas_particles_n5e-4.sh and
# runINT_masked_ionized_gas_particles_z026_n1e-3.sh; every other map is cached.
# Two steps side by side on one node, half the node each:
#   (a) Fig. 7: make_fgas_profiles.py, fgas_profiles_z05_n5e-4.yaml (the
#       lensing-fit run, job 59140079, took 2 h 12 min at 36 GB);
#   (b) in order:
#       - Fig. 5: 'lens' (5e-4) stacks of the two FLAMINGO M*-sigma runs, the
#         only Fig. 5 runs without them. --sims is essential: the other runs'
#         'lens' stellar stacks hold the nine transfers of pk_components_z05.yaml,
#         and stack_stellar_maps.py would rewrite them with this config's one;
#         then the grid figure (layout D, as the paper) and the table;
#       - Fig. 6: simulated_dsigma_ratio_masked.py, dsigma_ratio_z05_masked_nobeam_n5e-4.yaml;
#       - Fig. 9: the same, dsigma_ratio_z026_masked_nobeam_n1e-3.yaml
#       (the lensing-fit stacks took 30 min each at 69-93 GB).
# One log per step: ../Outputs_Perlmutter/figs_n5e-4-<job>_{fgas,chain}.log.
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 03:30:00 -A desi -J figs_n5e-4 bash unbound_gas/runINT_figs_n5e-4.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python

export LOG=../Outputs_Perlmutter/figs_n5e-4-${SLURM_JOB_ID}
export CFG=configs/unbound_gas

fgas() {  # (a) Fig. 7
  OMP_NUM_THREADS=64 NUMBA_NUM_THREADS=64 NUMEXPR_MAX_THREADS=64 \
    python -u unbound_gas/make_fgas_profiles.py -p $CFG/fgas_profiles_z05_n5e-4.yaml
}
chain() {  # (b) Figs. 5, 6 and 9; stops at the first failure
  export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 NUMEXPR_MAX_THREADS=8
  local mstar="Mstar-1sigma Mstar-1sigma_fgas-4sigma"
  echo "########## Fig. 5: M*-sigma 'lens' stacks ##########" &&
  python -u unbound_gas/stack_stellar_maps.py -p $CFG/pk_components_z05_n5e-4.yaml --sims $mstar &&
  python -u unbound_gas/stack_fstar_obs_maps.py -p $CFG/pk_fstar_obs_z05_n5e-4.yaml --sims $mstar &&
  echo "########## Fig. 5: grid (layout D) and table ##########" &&
  python -u unbound_gas/make_pk_fstar_obs_grid.py -p $CFG/pk_fstar_obs_z05_n5e-4.yaml --layout D &&
  python -u unbound_gas/make_pk_fstar_obs.py -p $CFG/pk_fstar_obs_z05_n5e-4.yaml &&
  echo "########## Fig. 6 ##########" &&
  python -u unbound_gas/simulated_dsigma_ratio_masked.py -p $CFG/dsigma_ratio_z05_masked_nobeam_n5e-4.yaml --workers 12 &&
  echo "########## Fig. 9 ##########" &&
  python -u unbound_gas/simulated_dsigma_ratio_masked.py -p $CFG/dsigma_ratio_z026_masked_nobeam_n1e-3.yaml --workers 8
}
export -f fgas chain

srun -n 1 -c 128 --mem=220G --cpu-bind=cores bash -c 'fgas' > "${LOG}_fgas.log" 2>&1 &
pa=$!
srun -n 1 -c 128 --mem=220G --cpu-bind=cores bash -c 'chain' > "${LOG}_chain.log" 2>&1 &
pb=$!
wait $pa; ra=$?
wait $pb; rb=$?
echo "fgas: exit $ra; chain: exit $rb"
rc=$(( ra != 0 || rb != 0 ))
echo "########## EXIT CODE: $rc ##########"
exit $rc
