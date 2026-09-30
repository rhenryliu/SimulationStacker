#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --job-name=fit_bgs
#SBATCH -o ../Outputs_Perlmutter/fit_bgs-%j.out
#SBATCH -e ../Outputs_Perlmutter/fit_bgs-%j.err

# Appendix B of the unbound gas paper: SHAM densities fitted on the HSC Y3 x
# DESI BGS lensing profile (lensing/fit_dsigma_ksz.py).
#   node 1  TNG300-1 80, Illustris-1 116, SIMBA-100 136 (fit_dsigma_ksz_z026_bgs.yaml)
#   node 2  the five FLAMINGO runs at snapshot 71 (z = 0.30), one process each
#           (fit_dsigma_ksz_z030_bgs_<variant>.yaml; 4e5-galaxy subsamples)
# The stacking is a single-core loop, so the five FLAMINGO fits share a node.
# Needs the M*-sigma runs' z = 0.30 maps (runCPU_z026_products.sh).
# Submit from scripts/:  sbatch --dependency=afterok:<z026_products> unbound_gas/runCPU_fit_bgs.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 NUMEXPR_MAX_THREADS=16

LOG=../Outputs_Perlmutter/fit_bgs-${SLURM_JOB_ID}
flamingo() {
  local rc=0 pids=()
  for fb in L1_m9 fgas-8sigma Jet_fgas-4sigma Mstar-1sigma Mstar-1sigma_fgas-4sigma; do
    python -u lensing/fit_dsigma_ksz.py -p configs/lensing/fit_dsigma_ksz_z030_bgs_$fb.yaml \
      > ${LOG}_$fb.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || rc=1; done
  return $rc
}
export -f flamingo
export LOG
srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores \
  python -u lensing/fit_dsigma_ksz.py -p configs/lensing/fit_dsigma_ksz_z026_bgs.yaml > ${LOG}_small.log 2>&1 &
srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores bash -c "flamingo" > ${LOG}_flamingo.log 2>&1 &
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
