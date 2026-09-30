#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=01:45:00
#SBATCH --nodes=8
#SBATCH --job-name=fstar_iter3
#SBATCH -o ../Outputs_Perlmutter/fstar_iter3-%j.out
#SBATCH -e ../Outputs_Perlmutter/fstar_iter3-%j.err

# Stellar fractions from the particles (star_fraction_v2.py, method: particles)
# for the unbound gas paper, iteration 3, one config per node:
#   star_fraction_particles_4/5       the FLAMINGO M*-sigma variants at z = 0.5 (Fig. 1)
#   star_fraction_particles_z026_1..6 Appendix B: TNG300-1 80, Illustris-1 116,
#                                     SIMBA-100 136, FLAMINGO x5 at snapshot 71
# Each writes <fig_name>_z<z>_star_fraction.yaml in the dated figures folder.
# Submit from scripts/:  sbatch unbound_gas/runCPU_fstar_iter3.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128

LOG=../Outputs_Perlmutter/fstar_iter3-${SLURM_JOB_ID}
for c in 4 5 z026_1 z026_2 z026_3 z026_4 z026_5 z026_6; do
  srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores python -u unbound_gas/star_fraction_v2.py \
    -p configs/unbound_gas/star_fraction_particles_$c.yaml > ${LOG}_$c.log 2>&1 &
done
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
