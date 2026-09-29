#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=03:30:00
#SBATCH --nodes=4
#SBATCH --job-name=fit_z026
#SBATCH -o ../Outputs_Perlmutter/fit_z026-%j.out
#SBATCH -e ../Outputs_Perlmutter/fit_z026-%j.err

# Lensing-fit SHAM densities at z ~ 0.26 (HSC Y3 x DESI BGS) for the unbound
# gas paper's redshift appendix: TNG300-1 / Illustris-1 / SIMBA-100 at
# z = 0.26 on one node, and each FLAMINGO variant (snapshot 71, z = 0.30) on its
# own node; the FLAMINGO runs also build (and cache, as new files) their
# z = 0.30 tau maps for the kSZ row. Each config checkpoints to its own npz, so
# a rerun with --resume restacks only what is missing.
# Submit from scripts/:  sbatch unbound_gas/runCPU_fit_z026.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128

LOG=../Outputs_Perlmutter/fit_z026-${SLURM_JOB_ID}
for c in fit_dsigma_ksz_z026 fit_dsigma_ksz_z030_flamingo_L1_m9 \
         fit_dsigma_ksz_z030_flamingo_fgas-8sigma fit_dsigma_ksz_z030_flamingo_Jet_fgas-4sigma; do
  srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores python -u lensing/fit_dsigma_ksz.py \
    -p configs/lensing/$c.yaml --resume > ${LOG}_$c.log 2>&1 &
done
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
