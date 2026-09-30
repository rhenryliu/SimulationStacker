#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --job-name=masks_mstar
#SBATCH -o ../Outputs_Perlmutter/masks_mstar-%j.out
#SBATCH -e ../Outputs_Perlmutter/masks_mstar-%j.err

# Masked tau (xy) and tSZ (xz) maps (1, 2, 3 x R200m) of the FLAMINGO
# stellar-mass variants at z = 0.5 around the host haloes of their lensing-fit
# SHAM galaxies (precompute_masked_sham_hosts.py), one variant per node. No
# maps exist at these paths yet, so nothing is moved to the trash. Needs the
# 3548^3 cubes (runCPU_mstar_cubes.sh) and the fits (runCPU_fit_mstar.sh).
# Submit from scripts/:
#   sbatch --dependency=afterok:<cubes>:<fit> unbound_gas/runCPU_masked_sham_hosts_mstar.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128

LOG=../Outputs_Perlmutter/masks_mstar-${SLURM_JOB_ID}
FIT=../data/fit_dsigma_ksz/fit_dsigma_ksz_z05_mstar.npz
variant() {
  local fb=$1 rc=0
  for pt in tau tSZ; do
    python -u unbound_gas/precompute_masked_sham_hosts.py --sim-type FLAMINGO --name L1_m9 \
      --snapshot 67 --feedback "$fb" --ptype "$pt" --fit "$FIT" || rc=1
  done
  return $rc
}
export -f variant
export FIT
for fb in Mstar-1sigma Mstar-1sigma_fgas-4sigma; do
  srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores bash -c "variant $fb" \
    > "${LOG}_${fb}.log" 2>&1 &
done
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
ls -la "$(realpath /pscratch/sd/r/rhliu)"/simulations/FLAMINGO/products/2D/masked/ | grep Mstar
echo "exit $rc"
exit $rc
