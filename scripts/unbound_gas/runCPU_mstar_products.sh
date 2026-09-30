#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=05:00:00
#SBATCH --nodes=4
#SBATCH --job-name=mstar_products
#SBATCH -o ../Outputs_Perlmutter/mstar_products-%j.out
#SBATCH -e ../Outputs_Perlmutter/mstar_products-%j.err

# Cached fields and maps of the two FLAMINGO stellar-mass variants
# (Mstar-1sigma, Mstar-1sigma_fgas-4sigma) at z = 0.5 (snapshot 67) that the
# unbound gas figures load, mirroring what is cached for L1_m9. One node per
# (variant, set):
#   set A  3D 2000^3 gas, ionized_gas, Stars, BH, then total and baryon (which
#          load the cached gas/Stars/BH; total bins the DM from the particles,
#          uncached); 0.5' raw gas, Stars, BH (yz), then the
#          1.6'-beam ionized_gas, baryon and total maps (Figs 2-3 columns 2-3)
#   set B  0.2' raw gas, ionized_gas, Stars, BH, then total and baryon (yz;
#          DSigma columns, Fig 4, Fig 9, lensing fit); the 1.6'-beam tau map in
#          yz (the kSZ row of lensing/fit_dsigma_ksz.py)
# Each task skips itself when its output exists, so the job can be resubmitted.
# Submit from scripts/:  sbatch unbound_gas/runCPU_mstar_products.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128

LOG=../Outputs_Perlmutter/mstar_products-${SLURM_JOB_ID}
P="python -u unbound_gas/precompute_baryonFraction_fields.py --simtype FLAMINGO --sim L1_m9 --snapshot 67 --redshift 0.5 --projection yz"

set_a() {  # variant
  local fb=$1 rc=0
  for pt in gas ionized_gas Stars BH total baryon; do
    $P --feedback "$fb" --ptype $pt --dim 3D --n-pixels 2000 || rc=1
  done
  for pt in gas Stars BH; do
    $P --feedback "$fb" --ptype $pt --dim 2D --pixel-size 0.5 || rc=1
  done
  for pt in ionized_gas baryon total; do
    $P --feedback "$fb" --ptype $pt --dim 2D --pixel-size 0.5 --beam-size 1.6 || rc=1
  done
  return $rc
}
set_b() {  # variant
  local fb=$1 rc=0
  for pt in gas ionized_gas Stars BH total baryon; do
    $P --feedback "$fb" --ptype $pt --dim 2D --pixel-size 0.2 || rc=1
  done
  $P --feedback "$fb" --ptype tau --dim 2D --pixel-size 0.5 --beam-size 1.6 || rc=1
  return $rc
}
export -f set_a set_b
export P
for fb in Mstar-1sigma Mstar-1sigma_fgas-4sigma; do
  for s in a b; do
    srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores bash -c "set_$s $fb" \
      > "${LOG}_${fb}_${s}.log" 2>&1 &
  done
done
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
