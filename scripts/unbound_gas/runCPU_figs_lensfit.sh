#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --job-name=figs_lensfit
#SBATCH -o ../Outputs_Perlmutter/figs_lensfit-%j.out
#SBATCH -e ../Outputs_Perlmutter/figs_lensfit-%j.err

# Unbound gas paper: kSZ and tSZ masking figures and the SHAM-vs-mass-cut
# f_gas figure on the lensing-fit SHAM samples (abundance_from_fit in their
# configs), with the masks rebuilt around the SHAM hosts. Refuses to start
# unless check_masked_maps.py passes (a missing masked map would otherwise be
# rebuilt with the mass-cut sample and saved). The three scripts are serial
# per-halo Python loops, so they run side by side on one node.
# Submit from scripts/ after the mask jobs:  sbatch unbound_gas/runCPU_figs_lensfit.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=32 NUMBA_NUM_THREADS=32 NUMEXPR_MAX_THREADS=32

KSZ=configs/unbound_gas/tau_z05_CAP_masked_flamingo.yaml
TSZ=configs/unbound_gas/tSZ_z05_CAP_masked.yaml
python unbound_gas/check_masked_maps.py -p $KSZ || exit 1
python unbound_gas/check_masked_maps.py -p $TSZ || exit 1

LOG=../Outputs_Perlmutter/figs_lensfit-${SLURM_JOB_ID}
python -u unbound_gas/simulated_kSZ_masked.py -p $KSZ > ${LOG}_kSZ.log 2>&1 &
python -u unbound_gas/simulated_tSZ_masked.py -p $TSZ > ${LOG}_tSZ.log 2>&1 &
python -u unbound_gas/make_fgas_profiles.py -p configs/unbound_gas/fgas_profiles_z05.yaml > ${LOG}_fgas.log 2>&1 &
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
