#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --job-name=fig4_relabel
#SBATCH -o ../Outputs_Perlmutter/fig4_relabel-%j.out
#SBATCH -e ../Outputs_Perlmutter/fig4_relabel-%j.err

# Re-render the unbound gas paper's component-budget figure (panel a:
# make_baryonFraction.py 3D shells; panel b: make_stackArea.py DSigma) with the
# paper's simulation names (FLAMINGO L1_m9 / fgas-8sigma / Jet_fgas-4sigma,
# SIMBA-100). Same configs and cached fields as before; label-only change.
# The two run side by side (peak ~197 GB for make_baryonFraction; see
# runCPU_baryonFraction_figure.sh). Submit from scripts/:
#   sbatch unbound_gas/runCPU_fig4_relabel.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=64 NUMBA_NUM_THREADS=64 NUMEXPR_MAX_THREADS=64

LOG=../Outputs_Perlmutter/fig4_relabel-${SLURM_JOB_ID}
python -u unbound_gas/make_baryonFraction.py -p configs/unbound_gas/baryonFraction_z05.yaml > ${LOG}_shells.log 2>&1 &
python -u unbound_gas/make_stackArea.py -p configs/unbound_gas/stackArea_dsigma_z05.yaml > ${LOG}_dsigma.log 2>&1 &
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
