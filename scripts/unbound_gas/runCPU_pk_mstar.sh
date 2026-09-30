#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=08:00:00
#SBATCH --nodes=2
#SBATCH --job-name=pk_mstar
#SBATCH -o ../Outputs_Perlmutter/pk_mstar-%j.out
#SBATCH -e ../Outputs_Perlmutter/pk_mstar-%j.err

# The P(k)-section chain (unbound gas paper, S(k) + lensing f_gas figure) for
# the FLAMINGO stellar-mass variants at z = 0.5, one variant per node, the
# stages in order:
#   1 compute_pk_components.py --what dmo   S(k) reference: P_total, P_DMO
#                                           (L1_m9_DMO field, shared)
#   2 compute_pk_stellar.py                 s-scaled spectra, ap1 / M >= 1e13
#   3 compute_stellar_maps.py + stack_stellar_maps.py   lensing stacks, 'lensfit'
#   4 compute_fstar.py (reduced config)     f* floor region sums, ap1 / 1e13
#   5 compute_fstar_obs.py + stack_fstar_obs_maps.py    capped ends + stacks
# Needs the 2000^3 fields (runCPU_mstar_products.sh) and the lensing fits
# (fit_dsigma_ksz_z05_mstar.npz). The paper figure draws these runs without
# their bands (bands: false). Submit from scripts/:
#   sbatch --dependency=afterok:<products>:<fit> unbound_gas/runCPU_pk_mstar.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128

LOG=../Outputs_Perlmutter/pk_mstar-${SLURM_JOB_ID}
CS=configs/unbound_gas/pk_components_z05_lensfit.yaml
CF=configs/unbound_gas/pk_fstar_z05_mstar.yaml
CO=configs/unbound_gas/pk_fstar_obs_z05_lensfit.yaml
chain() {  # variant (exact feedback name: select_sims matches it exactly)
  local fb=$1
  python -u unbound_gas/compute_pk_components.py -p $CS --sims "$fb" --what dmo || return 1
  python -u unbound_gas/compute_pk_stellar.py -p $CS --sims "$fb" || return 2
  python -u unbound_gas/compute_stellar_maps.py -p $CS --sims "$fb" || return 3
  OMP_NUM_THREADS=1 python -u unbound_gas/stack_stellar_maps.py -p $CS --sims "$fb" || return 4
  python -u unbound_gas/compute_fstar.py -p $CF --sims "$fb" || return 5
  python -u unbound_gas/compute_fstar_obs.py -p $CO --sims "$fb" || return 6
  OMP_NUM_THREADS=1 python -u unbound_gas/stack_fstar_obs_maps.py -p $CO --sims "$fb" || return 7
}
export -f chain
export CS CF CO
for fb in Mstar-1sigma Mstar-1sigma_fgas-4sigma; do
  srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores bash -c "chain $fb" \
    > "${LOG}_${fb}.log" 2>&1 &
done
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
