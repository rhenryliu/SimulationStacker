#!/bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --job-name=masks_sham_small
#SBATCH -o ../Outputs_Perlmutter/masks_sham_small-%j.out
#SBATCH -e ../Outputs_Perlmutter/masks_sham_small-%j.err

# Regenerate the IllustrisTNG / Illustris / SIMBA masked maps of the unbound gas
# masking figures (kSZ tau xy, tSZ xz; 1, 2, 3 x R200m) around the host haloes
# of the lensing-fit SHAM galaxies (precompute_masked_sham_hosts.py).
#
# First moves ALL existing IllustrisTNG and SIMBA masked maps to the scratch
# trash (same filesystem, original path recreated, numbered backups): they were
# built on 2026-03-02 with Group_R_TopHat200 / CAESAR r200c radii around the
# mass-cut sample. User decision 2026-09-29: regenerate at the current paths,
# trash first, regenerate only the maps the figures use.
#
# Also runs a one-radius FLAMINGO dry run (nothing saved) to time the FLAMINGO
# jobs (runCPU_masked_sham_hosts_flamingo.sh). Submit from scripts/:
#   sbatch unbound_gas/runCPU_masked_sham_hosts.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 NUMEXPR_MAX_THREADS=8

ROOT=$(realpath /pscratch/sd/r/rhliu)
TRASH=$ROOT/trash
for suite in IllustrisTNG SIMBA; do
  for f in "$ROOT"/simulations/$suite/products/2D/masked/*_map_masked*R200c.npy; do
    [ -e "$f" ] || continue
    rel=$(realpath --relative-to="$ROOT" "$f")
    mkdir -p "$TRASH/$(dirname "$rel")"
    mv -v --backup=numbered "$f" "$TRASH/$rel"
  done
done

LOG=../Outputs_Perlmutter/masks_sham_small-${SLURM_JOB_ID}
P=unbound_gas/precompute_masked_sham_hosts.py
run() {  # tag, then precompute arguments
  local tag=$1; shift
  python -u $P "$@" > "${LOG}_${tag}.log" 2>&1
  local rc=$?
  echo "exit $rc $tag"
  return $rc
}
run noagn_tau   --sim-type SIMBA --name m50n512 --snapshot 125 --feedback s50noagn --ptype tau &
run nox_tau     --sim-type SIMBA --name m50n512 --snapshot 125 --feedback s50nox --ptype tau &
run nofb_tau    --sim-type SIMBA --name m50n512 --snapshot 125 --feedback s50nofb --ptype tau &
run s100_tau    --sim-type SIMBA --name m100n1024 --snapshot 125 --feedback s50 --ptype tau &
run s100_tSZ    --sim-type SIMBA --name m100n1024 --snapshot 125 --feedback s50 --ptype tSZ &
run tng100_tau  --sim-type IllustrisTNG --name TNG100-1 --snapshot 67 --ptype tau &
run tng100_tSZ  --sim-type IllustrisTNG --name TNG100-1 --snapshot 67 --ptype tSZ &
run tng300_tau  --sim-type IllustrisTNG --name TNG300-1 --snapshot 67 --ptype tau &
run tng300_tSZ  --sim-type IllustrisTNG --name TNG300-1 --snapshot 67 --ptype tSZ &
run ill_tau     --sim-type IllustrisTNG --name Illustris-1 --snapshot 103 --ptype tau &
run ill_tSZ     --sim-type IllustrisTNG --name Illustris-1 --snapshot 103 --ptype tSZ &
run flamingo_dry --sim-type FLAMINGO --name L1_m9 --snapshot 67 --feedback L1_m9 --ptype tau --radii 1 --dry-run &
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done
echo "ALL DONE"
ls -la "$ROOT"/simulations/IllustrisTNG/products/2D/masked/ "$ROOT"/simulations/SIMBA/products/2D/masked/
exit $rc
