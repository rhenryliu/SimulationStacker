#! /bin/bash -l

# Task 1 r-profiles: debug-QOS smoke test.
#
# Runs the smallest production box (SIMBA m100n1024, 1301^2) end to end --
# cached-field load, CDM derivation, SHAM galaxy map, FFT amplitudes,
# jackknife, npz write -- plus the integration test against the legacy stamp
# stacker on TNG300-1.  Confirms the pipeline works at scale before the full
# sweep goes into the regular queue.
#
# Submit from scripts/:
#   cd scripts/
#   sbatch cross_corr/runCPU_rprofiles_debug.sh

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --job-name=rprof_smoke
#SBATCH -o ../Outputs_Perlmutter/slurm-%x-%j.out
#SBATCH -e ../Outputs_Perlmutter/slurm-%x-%j.err

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

echo "=== python: $(which python) ==="
echo "=== started: $(date) ==="

echo
echo "############ 1. SIMBA smoke run (z=0.5) ############"
python -u cross_corr/make_r_profiles.py \
    -p configs/cross_corr/r_profiles_z05.yaml --sim m100n1024

echo
echo "############ 2. Integration test vs legacy stamp stacker ############"
cd ../tests && python -u -m pytest test_rprofiles_integration.py -v -s --no-header
pytest_status=$?

echo "=== finished: $(date) ==="

# Propagate the test result: without this the job's exit status is that of the
# final echo, so a failing gate would still report COMPLETED 0:0 and satisfy an
# --dependency=afterok for the full sweep.
if [ $pytest_status -ne 0 ]; then
    echo "INTEGRATION TEST FAILED (exit $pytest_status)"
fi
exit $pytest_status
