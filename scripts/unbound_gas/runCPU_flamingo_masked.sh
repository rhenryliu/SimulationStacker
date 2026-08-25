#! /bin/bash -l

# Precompute the FLAMINGO maps needed for Figures 7 and 11 (the masked SZ
# profile grids). One array task per (feedback variant, particle type) pair.
#
# Each task needs a whole 512 GB CPU node: at 0.5 arcmin pixels the L1_m9 box
# gives nPixels = 3548, so the intermediate cubic field is 179 GB (float32)
# plus a 45 GB boolean mask. Each task also writes ~179 GB of 3D cache to
# scratch, ~1.07 TB across the six tasks.
#
# Every step skips itself if its output file already exists, so a task that
# hits the walltime can just be resubmitted.
#
# Submit from the scripts/ directory:
#   sbatch unbound_gas/runCPU_flamingo_masked.sh              # all six
#   sbatch --array=0-2 unbound_gas/runCPU_flamingo_masked.sh  # tSZ only

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --array=0-5
#SBATCH -o ../Outputs_Perlmutter/flamingo_masked-%A_%a.out # STDOUT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

# Give the whole node to the single task: the 3D field is built with
# abacusnbody's numba-parallel TSC, which is left at nthread=-1. Without an
# explicit cpus-per-task srun would bind the task to a couple of hardware
# threads and the 3548**3 binning would take days.
export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
# numexpr caps itself at NUMEXPR_MAX_THREADS (default 64) and prints a
# line starting with "Error." when OMP_NUM_THREADS exceeds it. Harmless,
# but it reads like a failure in the logs -- raise the cap to match.
export NUMEXPR_MAX_THREADS=128

# Array index -> (feedback, particle type). tSZ is stacked in 'xz' and tau in
# 'xy'; the driver picks the right projection from the particle type.
FEEDBACKS=(L1_m9 fgas-8sigma Jet_fgas-4sigma L1_m9 fgas-8sigma Jet_fgas-4sigma)
PTYPES=(tSZ tSZ tSZ tau tau tau)

FEEDBACK=${FEEDBACKS[$SLURM_ARRAY_TASK_ID]}
PTYPE=${PTYPES[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: feedback=$FEEDBACK ptype=$PTYPE"

srun --cpu-bind=cores python -u unbound_gas/precompute_flamingo_masked.py --feedback "$FEEDBACK" --ptype "$PTYPE"
