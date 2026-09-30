#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --job-name=ratios_z026
#SBATCH -o ../Outputs_Perlmutter/ratios_z026-%j.out
#SBATCH -e ../Outputs_Perlmutter/ratios_z026-%j.err

# Appendix B of the unbound gas paper: ionized gas fraction profiles at z ~ 0.26
# in 3D, CAP and Delta Sigma for the mass cut at the BGS mean host halo mass
# (configs/unbound_gas/ratios_z026_bgs.yaml). Needs runCPU_z026_products.sh.
# Submit from scripts/:  sbatch --dependency=afterok:<z026_products> unbound_gas/runCPU_ratios_z026.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=64 NUMBA_NUM_THREADS=64 NUMEXPR_MAX_THREADS=64
srun --nodes=1 --ntasks=1 -c 256 --cpu-bind=cores \
  python -u unbound_gas/make_ratios3x2.py -p configs/unbound_gas/ratios_z026_bgs.yaml --ptype ionized_gas
rc=$?
echo "exit $rc"
exit $rc
