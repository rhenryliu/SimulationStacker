#! /bin/bash -l

# Addendum Tasks 7, 8 and 10: the calibration factor C for the four retained
# simulations (TNG300-1 + three FLAMINGO variants), yz projection, both
# redshift samples.
#
# Submit from scripts/ (all paths below are CWD-relative to it):
#   cd scripts/
#   sbatch cross_corr/runCPU_calibration.sh
#
# Cost: one compute_Y_matrix call per (run, projection) over 19 apertures,
# against 15 for the Task 1 sweep, which took 17 minutes for twelve
# sim-samples.  Eight sim-samples here, so ~25 minutes expected; the two-hour
# request is slack for the FLAMINGO 14015^2 grids at z = 0.30.  As with the
# Task 1 runner, the heavy step is loading the subhalo catalogues rather than
# the FFT work, which is why this is not a login-node job.

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=calibration
#SBATCH -o ../Outputs_Perlmutter/slurm-%x-%j.out
#SBATCH -e ../Outputs_Perlmutter/slurm-%x-%j.err

set -o pipefail

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

echo "=== python: $(which python) ==="
echo "=== started: $(date) ==="
status=0

echo
echo "############ Calibration factor C, z ~ 0.5 (LRG-like) ############"
python -u cross_corr/make_calibration_factor.py \
    -p configs/cross_corr/calibration_z05.yaml || status=1

echo
echo "############ Calibration factor C, z ~ 0.26 / 0.30 (BGS-like) ############"
python -u cross_corr/make_calibration_factor.py \
    -p configs/cross_corr/calibration_z026.yaml || status=1

echo
echo "=== finished: $(date) ==="
if [ $status -ne 0 ]; then
    echo "ONE OR MORE STEPS FAILED"
    exit 1
fi
exit 0
