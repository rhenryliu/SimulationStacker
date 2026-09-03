#! /bin/bash -l

# Interactive-QOS runner for check_upsilon_r0.py: the Upsilon reference-radius
# scan on both r-profile configs. All projected fields are cached, so this is
# FFT work only -- roughly 10 minutes at z ~ 0.5 (2674^2 and 8869^2 grids) and
# 20 at z ~ 0.26/0.30 (4822^2 and 14015^2).
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 1:30:00 -A desi bash cross_corr/runINT_upsilon_r0.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

R0_LIST="1.0 1.25 1.5 1.75 2.0 2.5"
status=0

echo "########## R0 scan, z ~ 0.5 ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u cross_corr/check_upsilon_r0.py \
  -p configs/cross_corr/r_profiles_z05.yaml --r0 $R0_LIST || status=1

echo "########## R0 scan, z ~ 0.26 / 0.30 ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u cross_corr/check_upsilon_r0.py \
  -p configs/cross_corr/r_profiles_z026.yaml --r0 $R0_LIST || status=1

echo "########## EXIT CODE: $status ##########"
exit $status
