#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --job-name=ratios_unconv
#SBATCH -o ../Outputs_Perlmutter/ratios_unconv-%j.out
#SBATCH -e ../Outputs_Perlmutter/ratios_unconv-%j.err

# Unbound gas paper f_gas / f_baryon figures (ratios_3x2_z05.yaml) with the
# DSigma column on unconvolved 0.2 arcmin maps (pixel_size_col3 / beam_size_col3).
# The 0.2 arcmin ionized_gas and baryon fields of the SIMBA-50 runs and the
# TNG100-1 baryon field are not cached yet and are built (and saved as new
# files) on first use. The two particle types need disjoint new fields, so
# they run side by side.
# Submit from scripts/:  sbatch unbound_gas/runCPU_ratios_unconvolved.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=64 NUMBA_NUM_THREADS=64 NUMEXPR_MAX_THREADS=64

LOG=../Outputs_Perlmutter/ratios_unconv-${SLURM_JOB_ID}
python -u unbound_gas/make_ratios3x2.py -p configs/unbound_gas/ratios_3x2_z05.yaml --ptype ionized_gas > ${LOG}_ionized_gas.log 2>&1 &
python -u unbound_gas/make_ratios3x2.py -p configs/unbound_gas/ratios_3x2_z05.yaml --ptype baryon > ${LOG}_baryon.log 2>&1 &
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
