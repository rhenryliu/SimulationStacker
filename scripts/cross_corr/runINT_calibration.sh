#! /bin/bash -l

# Interactive-QOS runner for make_calibration_factor.py: addendum Tasks 7, 8
# and 10, the calibration factor C for the four retained simulations
# (TNG300-1 + three FLAMINGO variants), yz projection, both redshift samples.
#
# All projected fields are cached, so this is FFT work only: one
# compute_Y_matrix call per (run, projection) over 19 apertures, against 15
# for the Task 1 sweep. TNG300-1 at z ~ 0.5 measured 21 s of filtering; the
# FLAMINGO grids (8869^2 at z ~ 0.5, 14015^2 at z ~ 0.30) dominate. Expect
# roughly 30 minutes for both configs.
#
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 1:00:00 -A desi bash cross_corr/runINT_calibration.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

status=0

echo "########## Calibration factor C, z ~ 0.5 (LRG-like) ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u \
  cross_corr/make_calibration_factor.py \
  -p configs/cross_corr/calibration_z05.yaml || status=1

echo "########## Calibration factor C, z ~ 0.26 / 0.30 (BGS-like) ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u \
  cross_corr/make_calibration_factor.py \
  -p configs/cross_corr/calibration_z026.yaml || status=1

echo "########## EXIT CODE: $status ##########"
exit $status
