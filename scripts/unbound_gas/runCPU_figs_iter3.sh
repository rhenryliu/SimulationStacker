#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=05:00:00
#SBATCH --nodes=3
#SBATCH --job-name=figs_iter3
#SBATCH -o ../Outputs_Perlmutter/figs_iter3-%j.out
#SBATCH -e ../Outputs_Perlmutter/figs_iter3-%j.err

# Unbound gas paper, iteration 3: rerun the z = 0.5 figures with the FLAMINGO
# stellar-mass variants (all products cached beforehand). One node each:
#   ratios   make_ratios3x2.py, ionized gas and baryons (Figs. 2-3, 4 rows)
#   budget   make_stackArea.py and make_baryonFraction.py side by side (the
#            component-budget grid is then plotted from their npz on a login node)
#   sz       simulated_kSZ_masked.py, simulated_tSZ_masked.py, make_fgas_profiles.py
# Submit from scripts/:  sbatch unbound_gas/runCPU_figs_iter3.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=64 NUMBA_NUM_THREADS=64 NUMEXPR_MAX_THREADS=64

LOG=../Outputs_Perlmutter/figs_iter3-${SLURM_JOB_ID}
ratios() {
  local rc=0
  for pt in ionized_gas baryon; do
    python -u unbound_gas/make_ratios3x2.py -p configs/unbound_gas/ratios_3x2_z05.yaml --ptype $pt || rc=1
  done
  return $rc
}
budget() {
  python -u unbound_gas/make_stackArea.py -p configs/unbound_gas/stackArea_dsigma_z05.yaml \
    > ${LOG}_stackArea.log 2>&1 &
  local p1=$!
  python -u unbound_gas/make_baryonFraction.py -p configs/unbound_gas/baryonFraction_z05.yaml \
    > ${LOG}_baryonFraction.log 2>&1 &
  local p2=$!
  local rc=0
  wait $p1 || rc=1
  wait $p2 || rc=1
  return $rc
}
sz() {
  local rc=0
  python -u unbound_gas/simulated_kSZ_masked.py -p configs/unbound_gas/tau_z05_CAP_masked_flamingo.yaml || rc=1
  python -u unbound_gas/simulated_tSZ_masked.py -p configs/unbound_gas/tSZ_z05_CAP_masked.yaml || rc=1
  python -u unbound_gas/make_fgas_profiles.py -p configs/unbound_gas/fgas_profiles_z05.yaml || rc=1
  return $rc
}
export -f ratios budget sz
export LOG
for t in ratios budget sz; do
  srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores bash -c "$t" > "${LOG}_$t.log" 2>&1 &
done
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "exit $rc"
exit $rc
