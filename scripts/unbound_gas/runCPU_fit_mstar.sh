#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --job-name=fit_mstar
#SBATCH -o ../Outputs_Perlmutter/fit_mstar-%j.out
#SBATCH -e ../Outputs_Perlmutter/fit_mstar-%j.err

# Lensing fit of the SHAM number density (HSC Y3 x DESI LRG bin 1) for the
# FLAMINGO stellar-mass variants at z = 0.5 (lensing/fit_dsigma_ksz.py, same
# settings as fit_dsigma_ksz_z05.yaml) -> data/fit_dsigma_ksz/fit_dsigma_ksz_z05_mstar.npz.
# Needs the 0.2' total maps and the 1.6'-beam tau yz maps (runCPU_mstar_products.sh).
# Submit from scripts/:  sbatch --dependency=afterok:<products> unbound_gas/runCPU_fit_mstar.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128
srun --nodes=1 --ntasks=1 -c 256 --cpu-bind=cores \
  python -u lensing/fit_dsigma_ksz.py -p configs/lensing/fit_dsigma_ksz_z05_mstar.yaml
rc=$?
echo "exit $rc"
exit $rc
