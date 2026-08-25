#! /bin/bash -l

# Interactive-QOS runner for make_fgas_profiles.py (production f_gas figure,
# now with FLAMINGO). All maps are cached, so this is stacking-only.
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 2:30:00 -A desi bash unbound_gas/runINT_fgas_profiles.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

echo "########## make_fgas_profiles.py (z = 0.5, with FLAMINGO) ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u unbound_gas/make_fgas_profiles.py \
  -p configs/unbound_gas/fgas_profiles_z05.yaml
rc=$?
echo "########## EXIT CODE: $rc ##########"
exit $rc
