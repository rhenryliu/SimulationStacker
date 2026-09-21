#! /bin/bash -l

# Interactive-QOS runner for the make_baryonFraction.py (Figure 5) precompute.
#
# Same work as runCPU_baryonFraction_precompute.sh, but packed into a single
# salloc allocation instead of an sbatch array, because the interactive QOS has
# a much shorter queue wait. Interactive is salloc-only (sbatch cannot use it)
# and caps at 4 h / 4 nodes / 2 concurrent allocations.
#
# Each task gets a whole node via `srun -N1 -n1 --exclusive`, and at most
# $SLURM_JOB_NUM_NODES run at once. Task indices and the task table itself come
# from baryonFraction_tasks.sh, shared with the sbatch runner.
#
# The default TASKS order puts group B (the three FLAMINGO 3D ionized_gas builds,
# the long poles at roughly an hour each) first, so they start immediately on
# three nodes while group A's twelve short 2D builds fill the remaining node and
# then the nodes B frees.
#
# Launch from the scripts/ directory:
#   salloc -q interactive -C cpu -N 4 -t 4:00:00 -A desi \
#       bash unbound_gas/runINT_baryonFraction_precompute.sh
#
#   # a subset (e.g. only group C, the optional neutral_gas cache-warm):
#   TASKS="15 16 17 18 19 20 21 22 23" salloc -q interactive -C cpu -N 4 -t 4:00:00 \
#       -A desi bash unbound_gas/runINT_baryonFraction_precompute.sh
#
# Every task skips itself if its output already exists, so this is safe to rerun
# after a timeout. The exception is a cache truncated by a task killed mid-write:
# np.load raises on such a file, so delete it by hand before rerunning.

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

# Whole node per task: abacusnbody's TSC runs at nthread=-1, and srun's default
# binding would otherwise pin each task to a couple of hardware threads.
export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
# numexpr caps itself at NUMEXPR_MAX_THREADS (default 64) and prints a line
# starting with "Error." when OMP_NUM_THREADS exceeds it. Harmless, but it reads
# like a failure in the logs -- raise the cap to match.
export NUMEXPR_MAX_THREADS=128

source unbound_gas/baryonFraction_tasks.sh

# Groups B then A by default (long poles first); override with TASKS="...".
TASKS=${TASKS:-"12 13 14 0 1 2 3 4 5 6 7 8 9 10 11"}
NODES=${SLURM_JOB_NUM_NODES:-1}
JOB=${SLURM_JOB_ID:-local}
LOGDIR=../Outputs_Perlmutter
mkdir -p "$LOGDIR"

echo "########## baryonFraction precompute (interactive) ##########"
echo "allocation : job $JOB, $NODES node(s)"
echo "tasks      : $TASKS"
echo "logs       : $LOGDIR/bfrac_precompute_int-${JOB}_<task>.out"
echo "start      : $(date)"
echo

run_task () {
    local I=$1
    local LOG="$LOGDIR/bfrac_precompute_int-${JOB}_${I}.out"
    local FEEDBACK_ARG=()
    local rc
    if [ "${FEEDBACKS[$I]}" != "none" ]; then
        FEEDBACK_ARG=(--feedback "${FEEDBACKS[$I]}")
    fi
    {
        echo "Task $I: simtype=${SIMTYPES[$I]} sim=${SIMS[$I]} snapshot=${SNAPS[$I]}" \
             "feedback=${FEEDBACKS[$I]} ptype=${PTYPES[$I]} dim=${DIMS[$I]}"
        srun -N 1 -n 1 -c 256 --exclusive --cpu-bind=cores \
            python -u unbound_gas/precompute_baryonFraction_fields.py \
            --simtype "${SIMTYPES[$I]}" --sim "${SIMS[$I]}" --snapshot "${SNAPS[$I]}" \
            "${FEEDBACK_ARG[@]}" --ptype "${PTYPES[$I]}" --dim "${DIMS[$I]}" \
            --projection yz --pixel-size 0.2 --n-pixels 1000 --redshift 0.5
    } > "$LOG" 2>&1
    # Real exit status of the srun step (a brace group exits with the status of
    # its last command), recorded to a sidecar .rc file. Do not infer success by
    # grepping the log for a marker: the log also carries the task's own stdout,
    # so a marker could be spoofed by, or lost to, a change in what the Python
    # script prints -- and this step's whole job is to make cached .npy files
    # trustworthy for the figure that follows.
    rc=$?
    echo "EXIT CODE: $rc" >> "$LOG"
    echo "$rc" > "${LOG}.rc"
    if [ "$rc" = "0" ]; then
        echo "  [task $I] OK   -- $(grep -E '^  (Done in|Already present)' "$LOG" | tail -1)"
    else
        echo "  [task $I] FAIL (exit $rc) -- see $LOG"
    fi
}

# Throttle to one concurrent task per allocated node.
for I in $TASKS; do
    while [ "$(jobs -rp | wc -l)" -ge "$NODES" ]; do sleep 5; done
    echo "  [task $I] launching ($(date +%H:%M:%S))"
    run_task "$I" &
done
wait

echo
echo "finish     : $(date)"
echo "########## summary ##########"
FAILED=0
for I in $TASKS; do
    LOG="$LOGDIR/bfrac_precompute_int-${JOB}_${I}.out"
    if [ "$(cat "${LOG}.rc" 2>/dev/null)" = "0" ]; then
        printf "task %-2s OK   %s\n" "$I" "$(grep -E '^  (Done in|Already present)' "$LOG" | tail -1)"
    else
        printf "task %-2s FAIL %s\n" "$I" "$LOG"
        FAILED=$((FAILED + 1))
    fi
done
echo "failed tasks: $FAILED"
exit $((FAILED > 0))
