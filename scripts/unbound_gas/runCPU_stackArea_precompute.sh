#! /bin/bash -l

# Precompute the 2000^3 FLAMINGO 3D fields make_stackArea.py (Figure 4) loads
# with stackArea_dsigma_z05.yaml, as a job array of one (variant, particle type)
# task per index. The TNG300-1 / Illustris-1 / SIMBA fields (1000^3) and all 2D
# fields (0.2 arcmin, no beam) are already cached, so only FLAMINGO needs this.
#
#   0-11  A  gas, ionized_gas, Stars, BH at 2000^3 for the three variants
#   12-14 B  total (= gas + DM + Stars + BH) at 2000^3 for the three variants
#
# B must start only after A has finished: make_combined_field loads the cached
# gas / Stars / BH components and bins only DM, but if a component is missing
# it rebuilds it in memory without saving it, i.e. a wasted full particle pass.
# Hence two submissions chained by a dependency.
#
# Every task skips itself if its output already exists, so a task killed by the
# walltime can simply be resubmitted. The exception is a cache truncated by a
# job killed mid-write: delete such a file by hand before resubmitting.
#
# Whole 512 GB node per task: a 2000^3 float32 field is 32 GB, and the 'total'
# build holds the running sum, one loaded component and the DM field at once
# (~100 GB); the numba/TSC binning runs at nthread=-1, so without an explicit
# cpus-per-task srun would bind the task to a couple of hardware threads.
#
# %4 throttles group A to four concurrently running tasks.
#
# Submit from the scripts/ directory:
#   A=$(sbatch --parsable --array=0-11%4 unbound_gas/runCPU_stackArea_precompute.sh)
#   sbatch --array=12-14%3 --dependency=afterok:$A unbound_gas/runCPU_stackArea_precompute.sh
#
# Debug-QOS smoke test (one real, short task: fiducial BH):
#   sbatch -q debug -t 00:30:00 --array=9 unbound_gas/runCPU_stackArea_precompute.sh

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --job-name=stackArea_precompute
#SBATCH -o ../Outputs_Perlmutter/stackArea_precompute-%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

# Match runCPU_baryonFraction_precompute.sh: give the whole node to the task.
export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
# numexpr caps itself at NUMEXPR_MAX_THREADS (default 64) and prints a line
# starting with "Error." when OMP_NUM_THREADS exceeds it. Harmless, but it
# reads like a failure in the logs -- raise the cap to match.
export NUMEXPR_MAX_THREADS=128

# The python invocation below is relative to the submission directory; fail
# fast rather than run on from the wrong place.
[ -f unbound_gas/precompute_baryonFraction_fields.py ] || {
    echo "ERROR: submit this job from the scripts/ directory." >&2
    exit 1
}

N_PIXELS=2000   # must match the FLAMINGO n_pixels in stackArea_dsigma_z05.yaml

#          |------------------------------ A: components ------------------------------|------------ B: total ------------|
FEEDBACKS=( L1_m9 fgas-8sigma Jet_fgas-4sigma  L1_m9       fgas-8sigma Jet_fgas-4sigma  L1_m9 fgas-8sigma Jet_fgas-4sigma  L1_m9 fgas-8sigma Jet_fgas-4sigma  L1_m9 fgas-8sigma Jet_fgas-4sigma )
PTYPES=(    gas   gas         gas              ionized_gas ionized_gas ionized_gas      Stars Stars       Stars            BH    BH          BH               total total       total )

I=${SLURM_ARRAY_TASK_ID}
FEEDBACK=${FEEDBACKS[$I]}
PTYPE=${PTYPES[$I]}

echo "Task $I: FLAMINGO L1_m9 feedback=$FEEDBACK ptype=$PTYPE dim=3D n_pixels=$N_PIXELS"

srun --cpu-bind=cores python -u unbound_gas/precompute_baryonFraction_fields.py \
    --simtype FLAMINGO --sim L1_m9 --snapshot 67 --feedback "$FEEDBACK" \
    --ptype "$PTYPE" --dim 3D --n-pixels "$N_PIXELS" --redshift 0.5
