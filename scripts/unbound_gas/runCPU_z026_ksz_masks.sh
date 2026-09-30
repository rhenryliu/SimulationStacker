#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --job-name=z026_ksz
#SBATCH -o ../Outputs_Perlmutter/z026_ksz-%j.out
#SBATCH -e ../Outputs_Perlmutter/z026_ksz-%j.err

# Appendix B tier B: kSZ masking at z ~ 0.26 for SIMBA-100 136, TNG300-1 80 and
# Illustris-1 116 around the hosts of the BGS lensing-fit SHAM galaxies: the
# unmasked 1.6'-beam tau maps in xy, the tau cubes (--build-cube) and the
# masked maps (1, 2, 3 x R200m), then the masking figure and its npz.
# Needs data/fit_dsigma_ksz/fit_dsigma_ksz_z026_bgs.npz (runCPU_fit_bgs.sh).
# Submit from scripts/:  sbatch --dependency=afterok:<fit_bgs> unbound_gas/runCPU_z026_ksz_masks.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128

FIT=../data/fit_dsigma_ksz/fit_dsigma_ksz_z026_bgs.npz
F="python -u unbound_gas/precompute_baryonFraction_fields.py --projection xy --redshift 0.26 --ptype tau --dim 2D --pixel-size 0.5 --beam-size 1.6"
M="python -u unbound_gas/precompute_masked_sham_hosts.py --ptype tau --projection xy --z 0.26 --fit $FIT --build-cube"
rc=0
srun --nodes=1 --ntasks=1 -c 256 --cpu-bind=cores bash -c "
  $F --simtype SIMBA --sim m100n1024 --snapshot 136 --feedback s50 &&
  $F --simtype IllustrisTNG --sim TNG300-1 --snapshot 80 &&
  $F --simtype IllustrisTNG --sim Illustris-1 --snapshot 116 &&
  $M --sim-type SIMBA --name m100n1024 --snapshot 136 --feedback s50 &&
  $M --sim-type IllustrisTNG --name TNG300-1 --snapshot 80 &&
  $M --sim-type IllustrisTNG --name Illustris-1 --snapshot 116 &&
  python -u unbound_gas/simulated_kSZ_masked.py -p configs/unbound_gas/tau_z026_CAP_masked_bgs.yaml" || rc=1
echo "exit $rc"
exit $rc
