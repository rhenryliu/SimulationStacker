#! /bin/bash -l

# Interactive-QOS runner for make_hod_ratio.py (tau SHAM vs mass-cut ratio,
# now with FLAMINGO). First run builds the missing FLAMINGO tau yz beam maps
# (fgas-8sigma, Jet_fgas-4sigma; the fiducial one is already cached).
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 3:30:00 -A desi bash unbound_gas/runINT_hod_ratio.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

echo "########## make_hod_ratio.py (z = 0.5, with FLAMINGO) ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u unbound_gas/make_hod_ratio.py \
  -p configs/unbound_gas/hod_ratio_z05.yaml
rc=$?
echo "########## EXIT CODE: $rc ##########"
exit $rc
