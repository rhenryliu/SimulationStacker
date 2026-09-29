#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --job-name=fit_z026_ext
#SBATCH -o ../Outputs_Perlmutter/fit_z026_ext-%j.out
#SBATCH -e ../Outputs_Perlmutter/fit_z026_ext-%j.err

# Extended-grid rerun of the z ~ 0.26 lensing fit for TNG300-1, Illustris-1 and
# SIMBA-100 (fit_dsigma_ksz_z026.yaml, grid to 3.2e-2): the first run
# (job 59074639) ended at the 4e-3 grid edge for every simulation.
# Submit from scripts/:  sbatch unbound_gas/runCPU_fit_z026_ext.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128
srun -n 1 -c 256 --cpu-bind=cores python -u lensing/fit_dsigma_ksz.py \
  -p configs/lensing/fit_dsigma_ksz_z026.yaml --resume
