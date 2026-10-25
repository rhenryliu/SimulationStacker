#! /bin/bash -l

# FLAMINGO stacks and the figure of lensing/compare_data_ratio_lrg_bins.py (f_gas^obs in the four
# DESI LRG bins, v2.1 and v2.2 data with FLAMINGO L1_m9 / fgas-8sigma / Jet_fgas-4sigma).
#
# One task = one (run, snapshot) stack, `--only FEEDBACK SNAPSHOT`, which loads the cached maps or
# builds them (save_field: true). All maps are cached since job 60256233 (2026-10-25), where each
# 62/57 task built its beamed 0.5' ionized_gas map in one particle pass (~42 min, <= 10.2 GB).
# Stacks already cached with the config's settings are skipped, so this runner is safe to rerun.
# The binning and the per-halo stacking loop are single-threaded, so tasks run NPAR at a time.
#
# STAGE=smoke : snapshot 67 only; its curves must equal those of the v2-filter Figure 14b
#               (configs/lensing/dsigma_lensing/mass_ratio_data_z05.yaml).
# STAGE=full  : every (run, snapshot), then the figure from the caches.
# DRYRUN=1    : print the task list and exit.
#
# Per-task logs, elapsed time and peak RSS go to ../Outputs_Perlmutter/lrg_bins-<jobid>-<stage>/.
# Run on a compute node, from scripts/:
#   salloc -q interactive -C cpu -N 1 -t 03:00:00 -A desi -J lrg_bins srun bash lensing/runINT_lrg_bins.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

STAGE=${STAGE:-full}
NPAR=${NPAR:-9}
export CONFIG=configs/lensing/dsigma_lensing/ratio_lrg_bins_v2.1_v2.2.yaml
export LOGDIR=../Outputs_Perlmutter/lrg_bins-${SLURM_JOB_ID:-local}-$STAGE

case $STAGE in
    smoke) SNAPS="67" ;;
    full)  SNAPS="67 62 57" ;;
    *) echo "unknown STAGE=$STAGE" >&2; exit 1 ;;
esac
tasks=$(for snap in $SNAPS; do
            for fb in L1_m9 fgas-8sigma Jet_fgas-4sigma; do echo "$fb $snap"; done
        done)

# run_task FEEDBACK SNAPSHOT
run_task () {
    local fb=$1 snap=$2
    local tag="${fb}_${snap}" rc secs kb
    /usr/bin/time -f "%e %M" -o "$LOGDIR/$tag.time" \
        python -u lensing/compare_data_ratio_lrg_bins.py -p "$CONFIG" --only "$fb" "$snap" \
            > "$LOGDIR/$tag.log" 2>&1
    rc=$?
    # The last line of the time file holds "elapsed maxrss"; a task killed by a
    # signal gets a "Command terminated" line first, or no file at all.
    secs=$(tail -n 1 "$LOGDIR/$tag.time" 2>/dev/null | awk 'NF==2 && $1+0==$1 {print $1}')
    kb=$(tail -n 1 "$LOGDIR/$tag.time" 2>/dev/null | awk 'NF==2 && $2+0==$2 {print $2}')
    if [ -n "$secs" ] && [ -n "$kb" ]; then
        printf '%-4s %-28s %7.1f min  peak %6.1f GB  (%s)\n' \
            "$([ $rc = 0 ] && echo OK || echo FAIL)" "$tag" \
            "$(echo "$secs / 60" | bc -l)" "$(echo "$kb / 1048576" | bc -l)" "$(date +%H:%M)"
    else
        printf '%-4s %-28s  time/peak n/a (exit %s)  (%s)\n' \
            "$([ $rc = 0 ] && echo OK || echo FAIL)" "$tag" "$rc" "$(date +%H:%M)"
    fi
    return $rc
}
export -f run_task

echo "=== stage=$STAGE  parallel=$NPAR  python=$(which python)  host=$(hostname)  started $(date) ==="
if [ "${DRYRUN:-0}" = 1 ]; then
    echo "$tasks"
    exit 0
fi
mkdir -p "$LOGDIR"
status=0
echo "$tasks" | xargs -P "$NPAR" -L 1 bash -c 'run_task "$@"' _ || status=1

if [ $status = 0 ] && [ "$STAGE" = full ]; then
    echo
    echo "########## figure ##########"
    python -u lensing/compare_data_ratio_lrg_bins.py -p "$CONFIG" --plot-only || status=1
fi

echo
echo "=== finished $(date); logs in $LOGDIR ==="
echo "########## EXIT CODE: $status ##########"
exit $status
