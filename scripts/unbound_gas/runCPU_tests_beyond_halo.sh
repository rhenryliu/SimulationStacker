#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --nodes=2
#SBATCH --job-name=tests_beyond_halo
#SBATCH -o ../Outputs_Perlmutter/tests_beyond_halo-%j.out
#SBATCH -e ../Outputs_Perlmutter/tests_beyond_halo-%j.err

# Exploratory tests of the kSZ masking analysis (test_beyond_halo_masks.py;
# user request 2026-09-29: figures folder only, not in the paper): where the
# gas outside the stacked hosts sits (other haloes vs diffuse), and how much of
# the tau profile comes from far along the line of sight. Node 1: the smaller
# boxes; node 2: FLAMINGO L1_m9 (diffuse, 1e12 threshold only: its 1e11 haloes
# are unresolved, and the 179 GB cube leaves no room for the slab test).
# Submit from scripts/:  sbatch unbound_gas/runCPU_tests_beyond_halo.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 NUMEXPR_MAX_THREADS=16

LOG=../Outputs_Perlmutter/tests_beyond_halo-${SLURM_JOB_ID}
T=unbound_gas/test_beyond_halo_masks.py
node1() {
  python -u $T diffuse --sims TNG100-1 Illustris-1 SIMBA-100 TNG300-1 > ${LOG}_diffuse.log 2>&1 &
  python -u $T los --sims TNG300-1 --depths 50000 100000 > ${LOG}_los_TNG300.log 2>&1 &
  python -u $T los --sims SIMBA-100 --depths 25000 50000 > ${LOG}_los_SIMBA100.log 2>&1 &
  wait
}
export -f node1
export LOG T
srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=none bash -c node1 &
srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=none python -u $T diffuse --sims FLAMINGO_L1_m9 \
  --thresholds 1e12 > ${LOG}_diffuse_FLAMINGO.log 2>&1 &
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
