#! /bin/bash -l

# Render Figures 7 (kSZ) and 11 (tSZ) with the FLAMINGO row included.
#
# These have to run in a batch job rather than on a login node. The stacking
# loop in stack_on_array is a per-halo Python loop, and the FLAMINGO mask
# selection yields 213,108 halos (vs 6,240 for TNG300-1) on a 3548^2 map, so
# each FLAMINGO panel takes several minutes and the full figure 1-2 hours.
# A login-node attempt was reaped after ~5 minutes of sustained CPU.
#
# Everything the scripts need is already cached (both configs set
# load_field: true), so this job is stacking and plotting only -- no particle
# reads. Single-threaded by nature; the node is mostly idle.
#
# Submit from the scripts/ directory:
#   sbatch unbound_gas/runCPU_flamingo_figures.sh

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH -o ../Outputs_Perlmutter/flamingo_figures-%j.out # STDOUT
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

rc=0

echo "########## Figure 11: tSZ ##########"
srun --cpu-bind=cores python -u unbound_gas/simulated_tSZ_masked.py \
  -p configs/unbound_gas/tSZ_z05_CAP_masked.yaml || rc=1

echo "########## Figure 7: kSZ ##########"
srun --cpu-bind=cores python -u unbound_gas/simulated_kSZ_masked.py \
  -p configs/unbound_gas/tau_z05_CAP_masked_flamingo.yaml || rc=1

if [ $rc -eq 0 ]; then
  echo "########## FIGURES RENDERED ##########"
else
  echo "########## RENDER FAILED ##########"
fi
exit $rc
