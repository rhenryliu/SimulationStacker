#! /bin/bash -l

# Interactive-QOS runner for the SIMBA ElectronAbundance impact study
# (docs/unbound_gas/simba_electron_abundance_report.md). Everything is in memory:
# the scripts disable cache writes and only write new figures/.npz files under
# ../figures/<YYYY-MM>/<MM-DD>/simba_test/.
#
# Launch from the scripts/ directory:
#   salloc -q interactive -C cpu -N 1 -t 0:30:00 -A desi bash simba_test/runINT_simbaEA.sh smoke
#   salloc -q interactive -C cpu -N 1 -t 2:30:00 -A desi bash simba_test/runINT_simbaEA.sh full
#
# smoke: unbound-gas figure for m50n512 s50noagn only, with the as-read
#        validation, plus the lensing z=0.5 figures without corrected builds.
# full : lensing Figs 14/16 at z=0.5 and z=0.26, then the unbound-gas figure
#        for m100n1024 s50 and m50n512 s50noagn/nox/nofb (with validation).

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:-full}
JOB=${SLURM_JOB_ID:-local}
LOGDIR=../Outputs_Perlmutter
mkdir -p "$LOGDIR"
FAILED=0

run_step () {
    local name=$1; shift
    local log="$LOGDIR/simbaEA-${JOB}_${name}.out"
    echo "  [$name] start $(date +%H:%M:%S) -> $log"
    srun -N 1 -n 1 -c 256 --cpu-bind=cores python -u "$@" > "$log" 2>&1
    local rc=$?
    echo "EXIT CODE: $rc" >> "$log"
    echo "  [$name] exit $rc at $(date +%H:%M:%S)"
    [ $rc -eq 0 ] || FAILED=$((FAILED + 1))
}

echo "########## SIMBA EA impact study ($MODE), job $JOB ##########"
which python

if [ "$MODE" = "smoke" ]; then
    run_step unbound_smoke simba_test/unbound_simbaEA_difference.py --sims m50n512_s50noagn --validate --tag SMOKETEST
    run_step lensing_smoke_z0.5 simba_test/lensing_fig14_16_simbaEA.py --z 0.5 --skip-corrected
elif [ "$MODE" = "full" ]; then
    run_step lensing_z0.5 simba_test/lensing_fig14_16_simbaEA.py --z 0.5
    run_step lensing_z0.26 simba_test/lensing_fig14_16_simbaEA.py --z 0.26
    run_step unbound simba_test/unbound_simbaEA_difference.py --validate
else
    echo "unknown mode: $MODE"; exit 2
fi

echo "failed steps: $FAILED"
exit $((FAILED > 0))
