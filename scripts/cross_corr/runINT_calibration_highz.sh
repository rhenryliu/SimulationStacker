#! /bin/bash -l

# Interactive-QOS runner for make_calibration_factor.py at the two new
# redshift slices: z ~ 0.75 (TNG300-1 57, FLAMINGO 62) and z ~ 1.0
# (TNG300-1 50, FLAMINGO 57), four runs each, yz projection.  Companion to
# runINT_calibration.sh (z ~ 0.5 and z ~ 0.26/0.30).
#
# All projected fields are cached (built by runCPU_rprofiles_highz_fields.sh),
# so this is FFT work only.  On 6328^2 and 5076^2 FLAMINGO grids the two
# configs took 4.7 and 4.5 minutes (job 59388014, 2026-10-05), about 11
# minutes wall in all.  A TNG300-1-only smoke step (~1 minute, mostly the
# subhalo catalogue) runs first; its output is rewritten by the full z ~ 0.75
# step that follows.
#
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 0:30:00 -A desi -J calib_highz \
#     bash cross_corr/runINT_calibration_highz.sh \
#     > ../Outputs_Perlmutter/calibration_highz-int-$(date +%Y%m%d-%H%M%S).log 2>&1

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

echo "=== python: $(which python) ==="
echo "=== started: $(date) ==="

echo "########## Smoke: TNG300-1 only, z ~ 0.75 ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u \
  cross_corr/make_calibration_factor.py \
  -p configs/cross_corr/calibration_z075.yaml --sim TNG300-1 || {
    echo "########## SMOKE FAILED: not running the full configs ##########"
    exit 1
  }

status=0

echo "########## Calibration factor C, z ~ 0.75 ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u \
  cross_corr/make_calibration_factor.py \
  -p configs/cross_corr/calibration_z075.yaml || status=1

echo "########## Calibration factor C, z ~ 1.0 ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u \
  cross_corr/make_calibration_factor.py \
  -p configs/cross_corr/calibration_z10.yaml || status=1

echo "=== finished: $(date) ==="
echo "########## EXIT CODE: $status ##########"
exit $status
