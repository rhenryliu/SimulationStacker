#! /bin/bash -l

# Precompute the cached fields make_baryonFraction.py (Figure 5) needs, as a
# job array of one (simulation, particle type, dim) task per index.
#
# The task table has three groups. Submit A+B before running the figure; C only
# warms the cache for other scripts and for flipping `derive_neutral_gas: false`
# later, so nothing waits on it.
#
#   0-11  A  2D Stars / BH at 0.2 arcmin for all six simulations
#   12-14 B  3D ionized_gas at n_pixels=1000 for the three FLAMINGO variants
#   15-23 C  neutral_gas (2D for all six, 3D for FLAMINGO) -- optional
#
# Every task skips itself if its output already exists, so a task killed by the
# walltime can just be resubmitted. The exception is a cache truncated by a job
# killed mid-write: delete such a file by hand before resubmitting.
#
# Whole 512 GB node per task: the FLAMINGO 3D field is 1000**3 float32 (4 GB)
# on top of ~6 GB of per-chunk particle buffers, and the numba-parallel binning
# is left at nthread=-1 -- without an explicit cpus-per-task srun would bind the
# task to a couple of hardware threads.
#
# %4 throttles the array to four concurrently running tasks, so this does not
# monopolise the queue.
#
# Submit from the scripts/ directory:
#   sbatch --array=0-14%4 unbound_gas/runCPU_baryonFraction_precompute.sh  # A+B
#   sbatch --array=15-23%4 unbound_gas/runCPU_baryonFraction_precompute.sh # C

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --job-name=bfrac_precompute
#SBATCH -o ../Outputs_Perlmutter/bfrac_precompute-%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

# Match runCPU_flamingo_masked.sh: give the whole node to the single task.
export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
# numexpr caps itself at NUMEXPR_MAX_THREADS (default 64) and prints a line
# starting with "Error." when OMP_NUM_THREADS exceeds it. Harmless, but it
# reads like a failure in the logs -- raise the cap to match.
export NUMEXPR_MAX_THREADS=128

# Task table lives in a shared file so the sbatch and salloc runners cannot
# drift apart. It defines SIMTYPES/SIMS/SNAPS/FEEDBACKS/PTYPES/DIMS.
# Both this source and the python invocation below are relative to the sbatch
# submission directory, so fail fast with a clear message rather than running on
# with empty task arrays if the job was submitted from somewhere other than
# scripts/ (runINT guards the same hazard with an explicit cd).
source unbound_gas/baryonFraction_tasks.sh || {
    echo "ERROR: cannot source unbound_gas/baryonFraction_tasks.sh --" \
         "submit this job from the scripts/ directory." >&2
    exit 1
}

I=${SLURM_ARRAY_TASK_ID}
SIMTYPE=${SIMTYPES[$I]}
SIM=${SIMS[$I]}
SNAP=${SNAPS[$I]}
FEEDBACK=${FEEDBACKS[$I]}
PTYPE=${PTYPES[$I]}
DIM=${DIMS[$I]}

echo "Task $I: simtype=$SIMTYPE sim=$SIM snapshot=$SNAP feedback=$FEEDBACK ptype=$PTYPE dim=$DIM"

FEEDBACK_ARG=()
if [ "$FEEDBACK" != "none" ]; then
    FEEDBACK_ARG=(--feedback "$FEEDBACK")
fi

srun --cpu-bind=cores python -u unbound_gas/precompute_baryonFraction_fields.py \
    --simtype "$SIMTYPE" --sim "$SIM" --snapshot "$SNAP" "${FEEDBACK_ARG[@]}" \
    --ptype "$PTYPE" --dim "$DIM" \
    --projection yz --pixel-size 0.2 --n-pixels 1000 --redshift 0.5
