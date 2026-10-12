#! /bin/bash -l

# Interactive-QOS runner for simulated_dsigma_ratio_masked.py (the DeltaSigma
# ratio version of the kSZ masking figure). Stacking only: needs the yz masked
# tau maps of runINT_masked_tau_yz.sh; every other map is cached. The 12
# simulations are stacked in parallel worker processes (pure-Python per-galaxy
# loops, one core each); the FLAMINGO runs (0.6-0.76 million galaxies, five
# DeltaSigma stacks each) set the walltime.
# The config is the first argument (default: the beamed figure); the
# beam-compensated one, dsigma_ratio_z05_masked_nobeam.yaml, needs the masked
# fields of runINT_masked_ionized_gas_particles.sh.
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 2:00:00 -A desi -J dsigma_ratio_masked bash unbound_gas/runINT_dsigma_ratio_masked.sh [config]

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python

export OMP_NUM_THREADS=8
export NUMBA_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

CONFIG=${1:-configs/unbound_gas/dsigma_ratio_z05_masked.yaml}
echo "########## simulated_dsigma_ratio_masked.py (z = 0.5): $CONFIG ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u unbound_gas/simulated_dsigma_ratio_masked.py \
  -p "$CONFIG" --workers 12
rc=$?
echo "########## EXIT CODE: $rc ##########"
exit $rc
