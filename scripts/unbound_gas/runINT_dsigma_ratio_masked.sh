#! /bin/bash -l

# Interactive-QOS runner for simulated_dsigma_ratio_masked.py (the DeltaSigma
# ratio version of the kSZ masking figure). Stacking only: needs the yz masked
# tau maps of runINT_masked_tau_yz.sh; every other map is cached. The 12
# simulations are stacked in parallel worker processes (pure-Python per-galaxy
# loops, one core each); the FLAMINGO runs (0.6-0.76 million galaxies, five
# DeltaSigma stacks each) set the walltime.
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 2:00:00 -A desi -J dsigma_ratio_masked bash unbound_gas/runINT_dsigma_ratio_masked.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python

export OMP_NUM_THREADS=8
export NUMBA_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

echo "########## simulated_dsigma_ratio_masked.py (z = 0.5) ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u unbound_gas/simulated_dsigma_ratio_masked.py \
  -p configs/unbound_gas/dsigma_ratio_z05_masked.yaml --workers 12
rc=$?
echo "########## EXIT CODE: $rc ##########"
exit $rc
