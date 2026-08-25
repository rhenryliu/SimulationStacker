#! /bin/bash -l

# Render the tSZ masked-profile grid with the CAP_ringring filter.
#
# No precompute needed: the filter is applied at stack time, so this reuses the
# tSZ 'xz' map products already cached for the CAP version of the figure.
#
# Debug QOS (30 min cap) is enough -- the CAP version of this same figure took
# 21.9 minutes. If it ever overruns, resubmit with --qos=regular --time=02:00:00.
#
# Submit from the scripts/ directory:
#   sbatch unbound_gas/runCPU_ringring_figure.sh

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH -o ../Outputs_Perlmutter/ringring_figure-%j.out # STDOUT
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
# numexpr caps itself at NUMEXPR_MAX_THREADS (default 64) and prints a
# line starting with "Error." when OMP_NUM_THREADS exceeds it. Harmless,
# but it reads like a failure in the logs -- raise the cap to match.
export NUMEXPR_MAX_THREADS=128

srun --cpu-bind=cores python -u unbound_gas/simulated_tSZ_masked.py \
  -p configs/unbound_gas/tSZ_z05_ringring_masked.yaml
rc=$?

if [ $rc -eq 0 ]; then
  echo "########## RINGRING FIGURE RENDERED ##########"
else
  echo "########## RENDER FAILED ##########"
fi
exit $rc
