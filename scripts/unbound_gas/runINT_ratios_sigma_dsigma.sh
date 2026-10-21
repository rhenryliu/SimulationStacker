#! /bin/bash -l

# Interactive-QOS runner for make_ratios_sigma_dsigma.py (Figure 1 variant:
# Sigma and DeltaSigma rows on the unconvolved 0.2' maps; not in the draft).
# Stacking only, from the cached 0.2' fields: the ring-Sigma config took 1 h
# (job 60067855), mostly the four FLAMINGO stacks per variant. The optional
# argument is the config path relative to scripts/ (default:
# configs/unbound_gas/ratios_sigma_dsigma_z05.yaml).
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 2:00:00 -A desi bash unbound_gas/runINT_ratios_sigma_dsigma.sh \
#     [configs/unbound_gas/ratios_sigma_lensing_dsigma_z05.yaml]

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1
CONFIG=${1:-configs/unbound_gas/ratios_sigma_dsigma_z05.yaml}
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

echo "########## make_ratios_sigma_dsigma.py -p $CONFIG ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u unbound_gas/make_ratios_sigma_dsigma.py \
  -p "$CONFIG"
rc=$?
echo "########## EXIT CODE: $rc ##########"
exit $rc
