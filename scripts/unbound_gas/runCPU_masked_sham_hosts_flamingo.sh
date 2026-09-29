#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=01:30:00
#SBATCH --nodes=3
#SBATCH --job-name=masks_sham_flamingo
#SBATCH -o ../Outputs_Perlmutter/masks_sham_flamingo-%j.out
#SBATCH -e ../Outputs_Perlmutter/masks_sham_flamingo-%j.err

# Regenerate the FLAMINGO masked maps of the unbound gas masking figures (kSZ
# tau xy, tSZ xz; 1, 2, 3 x R200m) around the host haloes of the lensing-fit
# SHAM galaxies (precompute_masked_sham_hosts.py): one feedback variant per
# node (the 3548^3 float32 cube is 179 GB plus a 45 GB mask, reloaded for every
# radius). First moves the current FLAMINGO masked maps (R200m, mass-cut
# sample, built 2026-09-26) to the scratch trash (user decision 2026-09-29).
# Submit from scripts/:
#   sbatch unbound_gas/runCPU_masked_sham_hosts_flamingo.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128

ROOT=$(realpath /pscratch/sd/r/rhliu)
TRASH=$ROOT/trash
for f in "$ROOT"/simulations/FLAMINGO/products/2D/masked/L1_m9_*_67_*_map_masked*R200c.npy; do
  [ -e "$f" ] || continue
  rel=$(realpath --relative-to="$ROOT" "$f")
  mkdir -p "$TRASH/$(dirname "$rel")"
  mv -v --backup=numbered "$f" "$TRASH/$rel"
done

LOG=../Outputs_Perlmutter/masks_sham_flamingo-${SLURM_JOB_ID}
variant() {  # one feedback variant on one node: tau, then tSZ
  local fb=$1 rc=0
  for pt in tau tSZ; do
    python -u unbound_gas/precompute_masked_sham_hosts.py --sim-type FLAMINGO --name L1_m9 \
      --snapshot 67 --feedback "$fb" --ptype "$pt" || rc=1
  done
  return $rc
}
export -f variant
for fb in L1_m9 fgas-8sigma Jet_fgas-4sigma; do
  srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores bash -c "variant $fb" \
    > "${LOG}_${fb}.log" 2>&1 &
done
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
ls -la "$ROOT"/simulations/FLAMINGO/products/2D/masked/
echo "exit $rc"
exit $rc
