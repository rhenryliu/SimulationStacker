#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --nodes=3
#SBATCH --job-name=fstar_particles
#SBATCH -o ../Outputs_Perlmutter/fstar_particles-%j.out
#SBATCH -e ../Outputs_Perlmutter/fstar_particles-%j.err

# Unbound gas paper Figure 1 from the particles (star_fraction_v2.py,
# method: particles): three subsets, one per node (each holds one FLAMINGO
# variant). Each writes <fig_name>_z0.5_star_fraction.yaml in the dated
# figures folder; merge them into ../data/star_fraction/ and plot with
# configs/unbound_gas/star_fraction_particles.yaml.
# Submit from scripts/:  sbatch unbound_gas/runCPU_star_fraction_particles.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128

LOG=../Outputs_Perlmutter/fstar_particles-${SLURM_JOB_ID}
for i in 1 2 3; do
  srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores python -u unbound_gas/star_fraction_v2.py \
    -p configs/unbound_gas/star_fraction_particles_$i.yaml > ${LOG}_$i.log 2>&1 &
done
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
