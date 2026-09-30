#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=8
#SBATCH --job-name=z026_products
#SBATCH -o ../Outputs_Perlmutter/z026_products-%j.out
#SBATCH -e ../Outputs_Perlmutter/z026_products-%j.err

# Cached fields and maps for the unbound gas paper's Appendix B (z ~ 0.26:
# TNG300-1 80, Illustris-1 116, SIMBA-100 136; FLAMINGO snapshot 71, z = 0.30)
# that are not on disk yet. The Appendix B f_gas figure stacks ionized_gas and
# total in 3D, CAP (0.5', 1.6' beam) and DSigma (0.2', no beam). One node per task:
#   tng      TNG300-1 80: 3D ionized_gas (1000^3); 1.6'-beam total map
#   flam_*   FLAMINGO L1_m9 / fgas-8sigma / Jet_fgas-4sigma 71: 3D ionized_gas
#            (2000^3); 1.6'-beam total map (0.5' raw gas, Stars, BH first)
#   newA_*   M*-sigma variants 71: 3D gas, ionized_gas, Stars, BH, total;
#            0.5' raw gas, Stars, BH; 1.6'-beam ionized_gas and total maps
#   newB_*   M*-sigma variants 71: 0.2' gas, ionized_gas, Stars, BH, total,
#            baryon; 1.6'-beam tau map in yz (kSZ row of the lensing fit)
# Illustris-1 116 and SIMBA-100 136 are complete. Tasks skip existing outputs.
# TNG uses redshift 0.26 (not the snapshot's 0.2613) to match the cached z ~ 0.26
# maps (e.g. 1929 pixels at 0.5'), as the lensing configs do.
# Submit from scripts/:  sbatch unbound_gas/runCPU_z026_products.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128

LOG=../Outputs_Perlmutter/z026_products-${SLURM_JOB_ID}
F="python -u unbound_gas/precompute_baryonFraction_fields.py --projection yz"
FL="$F --simtype FLAMINGO --sim L1_m9 --snapshot 71 --redshift 0.3"
tng() {
  local rc=0
  $F --simtype IllustrisTNG --sim TNG300-1 --snapshot 80 --redshift 0.26 --ptype ionized_gas \
     --dim 3D --n-pixels 1000 || rc=1
  $F --simtype IllustrisTNG --sim TNG300-1 --snapshot 80 --redshift 0.26 --ptype total \
     --dim 2D --pixel-size 0.5 --beam-size 1.6 || rc=1
  return $rc
}
flam() {  # variant
  local fb=$1 rc=0
  $FL --feedback "$fb" --ptype ionized_gas --dim 3D --n-pixels 2000 || rc=1
  for pt in gas Stars BH; do $FL --feedback "$fb" --ptype $pt --dim 2D --pixel-size 0.5 || rc=1; done
  $FL --feedback "$fb" --ptype total --dim 2D --pixel-size 0.5 --beam-size 1.6 || rc=1
  return $rc
}
newA() {  # variant
  local fb=$1 rc=0
  for pt in gas ionized_gas Stars BH total; do
    $FL --feedback "$fb" --ptype $pt --dim 3D --n-pixels 2000 || rc=1
  done
  for pt in gas Stars BH; do $FL --feedback "$fb" --ptype $pt --dim 2D --pixel-size 0.5 || rc=1; done
  for pt in ionized_gas total; do
    $FL --feedback "$fb" --ptype $pt --dim 2D --pixel-size 0.5 --beam-size 1.6 || rc=1
  done
  return $rc
}
newB() {  # variant
  local fb=$1 rc=0
  for pt in gas ionized_gas Stars BH total baryon; do
    $FL --feedback "$fb" --ptype $pt --dim 2D --pixel-size 0.2 || rc=1
  done
  $FL --feedback "$fb" --ptype tau --dim 2D --pixel-size 0.5 --beam-size 1.6 || rc=1
  return $rc
}
export -f tng flam newA newB
export F FL
run() { srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores bash -c "$1" > "${LOG}_$2.log" 2>&1 & }
run "tng" tng
for fb in L1_m9 fgas-8sigma Jet_fgas-4sigma; do run "flam $fb" "flam_$fb"; done
for fb in Mstar-1sigma Mstar-1sigma_fgas-4sigma; do
  run "newA $fb" "newA_$fb"
  run "newB $fb" "newB_$fb"
done
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
