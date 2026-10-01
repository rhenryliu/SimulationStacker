#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --job-name=fgas_prof_iter3
#SBATCH -o ../Outputs_Perlmutter/fgas_prof_iter3-%j.out
#SBATCH -e ../Outputs_Perlmutter/fgas_prof_iter3-%j.err

# Unbound gas paper, selection figure (make_fgas_profiles.py): Delta Sigma at
# 0.2' without beam, mass cut vs lensing-fit SHAM, eight runs. Resubmission of
# the 'sz' task of job 59117699, which timed out on the seventh run: the
# FLAMINGO SHAM samples (~7e5 galaxies) make the Delta Sigma stacks ~20 min each.
# Submit from scripts/:  sbatch unbound_gas/runCPU_fgas_profiles_iter3.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=64 NUMBA_NUM_THREADS=64 NUMEXPR_MAX_THREADS=64
srun --nodes=1 --ntasks=1 -c 256 --cpu-bind=cores \
  python -u unbound_gas/make_fgas_profiles.py -p configs/unbound_gas/fgas_profiles_z05.yaml
rc=$?
echo "exit $rc"
exit $rc
