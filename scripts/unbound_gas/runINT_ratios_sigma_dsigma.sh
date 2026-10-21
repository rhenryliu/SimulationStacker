#! /bin/bash -l

# Interactive-QOS runner for make_ratios_sigma_dsigma.py (Figure 1 variant:
# Sigma(R) and DeltaSigma(R) rows on the unconvolved 0.2' maps; not in the
# draft). Stacking only, from the cached 0.2' fields; ~2-2.5 h, mostly the
# four FLAMINGO stacks per variant.
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 3:30:00 -A desi bash unbound_gas/runINT_ratios_sigma_dsigma.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

echo "########## make_ratios_sigma_dsigma.py (z = 0.5, ionized_gas) ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u unbound_gas/make_ratios_sigma_dsigma.py \
  -p configs/unbound_gas/ratios_sigma_dsigma_z05.yaml
rc=$?
echo "########## EXIT CODE: $rc ##########"
exit $rc
