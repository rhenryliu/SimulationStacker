#! /bin/bash -l

# f_gas SHAM vs mass-cut ratio figure (make_fgas_hod_ratio.py) with the
# FLAMINGO variants included.
#
# The first run is dominated by building the three FLAMINGO 'total'
# beam-convolved maps (3548^2 at 0.5 arcmin, from ~1.1e10 particles per
# variant, est. ~2-3 h each); the cached ionized_gas 3548 yz maps are
# reused. Each map is saved to scratch as soon as it is built, so a
# timeout resumes from cache on resubmission. Stacking adds roughly an
# hour per FLAMINGO variant (213,108 mass-cut centrals + ~158k SHAM
# subhalos, per-halo Python loop on a 3548^2 map); TNG/SIMBA take
# minutes. Hence the 12 h wall clock; reruns with all maps cached are
# stacking-only (~2-3 h).
#
# Submit from the scripts/ directory:
#   sbatch unbound_gas/runCPU_fgas_hod.sh                       # regular queue
#   sbatch -q debug -t 00:30:00 unbound_gas/runCPU_fgas_hod.sh  # smoke test

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH -o ../Outputs_Perlmutter/fgas_hod-%j.out # STDOUT
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

echo "########## f_gas HOD ratio (z = 0.5, with FLAMINGO) ##########"
srun --cpu-bind=cores python -u unbound_gas/make_fgas_hod_ratio.py \
  -p configs/unbound_gas/fgas_hod_ratio_z05.yaml

echo "########## DONE ##########"
