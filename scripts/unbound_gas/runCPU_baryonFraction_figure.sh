#! /bin/bash -l

# Produce Figure 5 of the unbound gas paper (make_baryonFraction.py), six
# simulations on a 3x2 grid, from cached fields.
#
# Needs a whole 512 GB node rather than a share of one: the FLAMINGO variants
# run at n_pixels=2000, so run_3d_stacking holds four 2000**3 float32 fields
# (4 x 32 GB) at once, and at the outermost edge the per-halo cutout index list
# for ~220-230k haloes is a further ~38 GB. That sums to ~166 GB, but the
# measured peak was 196.6 GB (seff, 2026-09-21 run) -- budget from the latter.
#
# Runtime is dominated by the FLAMINGO 3D edge sweeps. At n_pixels=1000 those
# took ~160 s per variant; the 2000 grid makes every sphere 8x larger in voxel
# count, so budget roughly 25 min per variant on top of ~40 min for the 2D
# stacking and the three smaller suites.
#
# Submit from the scripts/ directory:
#   sbatch unbound_gas/runCPU_baryonFraction_figure.sh

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --job-name=bfrac_figure
#SBATCH -o ../Outputs_Perlmutter/bfrac_figure-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
# numexpr caps itself at NUMEXPR_MAX_THREADS (default 64) and prints a line
# starting with "Error." when OMP_NUM_THREADS exceeds it. Harmless, but it
# reads like a failure in the logs -- raise the cap to match.
export NUMEXPR_MAX_THREADS=128

echo "########## make_baryonFraction.py (six sims, 3x2) ##########"
echo "start: $(date)"
srun -n 1 -c 256 --cpu-bind=cores python -u unbound_gas/make_baryonFraction.py \
  -p configs/unbound_gas/baryonFraction_z05.yaml
rc=$?
echo "finish: $(date)"
echo "########## EXIT CODE: $rc ##########"
exit $rc
