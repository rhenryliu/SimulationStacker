#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --job-name=fig6_lensfit
#SBATCH -o ../Outputs_Perlmutter/fig6_lensfit-%j.out
#SBATCH -e ../Outputs_Perlmutter/fig6_lensfit-%j.err

# Unbound gas paper, P(k) section figure: lensing-panel stacks on the
# lensing-fit SHAM sample ('lensfit'): the s = 0 stacks (stack_stellar_maps.py,
# ap1/M13 only) and the capped ends (stack_fstar_obs_maps.py). The spectra and
# transfer maps are sample-independent and already cached. Plot afterwards on
# the login node:
#   python unbound_gas/make_pk_fstar_obs_column.py -p configs/unbound_gas/pk_fstar_obs_z05_lensfit.yaml
# Submit from scripts/:  sbatch unbound_gas/runCPU_fig6_lensfit.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 NUMEXPR_MAX_THREADS=8

rc=0
# 'L1_m9' alone would select all three FLAMINGO variants; use the full label.
for sel in "TNG300-1" "Illustris-1" "L1_m9 (L1_m9)" "fgas-8sigma" "Jet_fgas-4sigma"; do
  echo "########## $sel ##########"
  python -u unbound_gas/stack_stellar_maps.py -p configs/unbound_gas/pk_components_z05_lensfit.yaml \
    --sims "$sel" || rc=1
  python -u unbound_gas/stack_fstar_obs_maps.py -p configs/unbound_gas/pk_fstar_obs_z05_lensfit.yaml \
    --sims "$sel" || rc=1
done
echo "exit $rc"
exit $rc
