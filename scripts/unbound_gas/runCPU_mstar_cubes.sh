#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=07:00:00
#SBATCH --nodes=4
#SBATCH --job-name=mstar_cubes
#SBATCH -o ../Outputs_Perlmutter/mstar_cubes-%j.out
#SBATCH -e ../Outputs_Perlmutter/mstar_cubes-%j.err

# SZ products of the two FLAMINGO stellar-mass variants at z = 0.5 for the
# masking figures: per (variant, ptype) one node builds the unmasked
# 1.6'-beam map (tau in xy, tSZ in xz; precompute_flamingo_masked.py with NO
# mask radii, so no mass-cut masked map is written) and the 3548^3 float32
# cube (makeField dim='3D'; 179 GB, the cache create_masked_field reloads).
# The masked maps themselves are built afterwards around the SHAM hosts by
# precompute_masked_sham_hosts.py, once the lensing fits exist.
# Submit from scripts/:  sbatch unbound_gas/runCPU_mstar_cubes.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128

LOG=../Outputs_Perlmutter/mstar_cubes-${SLURM_JOB_ID}
one() {  # variant ptype projection
  local fb=$1 pt=$2 proj=$3 rc=0
  python -u unbound_gas/precompute_flamingo_masked.py --feedback "$fb" --ptype "$pt" \
    --projection "$proj" --mask-radii || rc=1
  python -u unbound_gas/precompute_baryonFraction_fields.py --simtype FLAMINGO --sim L1_m9 \
    --snapshot 67 --redshift 0.5 --feedback "$fb" --ptype "$pt" --dim 3D --n-pixels 3548 \
    --projection "$proj" || rc=1
  return $rc
}
export -f one
for fb in Mstar-1sigma Mstar-1sigma_fgas-4sigma; do
  srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores bash -c "one $fb tau xy" \
    > "${LOG}_${fb}_tau.log" 2>&1 &
  srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores bash -c "one $fb tSZ xz" \
    > "${LOG}_${fb}_tSZ.log" 2>&1 &
done
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
