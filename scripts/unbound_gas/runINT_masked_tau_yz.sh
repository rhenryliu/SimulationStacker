#! /bin/bash -l

# Masked tau maps in the yz projection (1, 2, 3 x R200m around the host haloes
# of the lensing-fit SHAM galaxies, 0.5' pixels, 1.6' beam) for the DeltaSigma
# ratio version of the kSZ masking figure (simulated_dsigma_ratio_masked.py).
# The xy maps of the kSZ figure are not touched: the yz maps are new files
# (the masked-map path carries the projection), built by
# precompute_masked_sham_hosts.py from the cached 3D tau cubes, whose path does
# not carry one -- so no particle pass. No yz masked map exists yet, so
# nothing is moved to the trash.
#
# Three nodes: the seven IllustrisTNG / Illustris / SIMBA runs side by side on
# the first, then one FLAMINGO variant; two FLAMINGO variants one after the
# other on each of the others (a 3548^3 float32 cube is 179 GB plus a 45 GB
# mask, reloaded for every radius; ~20 min per variant in job 59070382).
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 3 -t 01:20:00 -A desi -J masks_tau_yz bash unbound_gas/runINT_masked_tau_yz.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python

export LOG=../Outputs_Perlmutter/masks_tau_yz-${SLURM_JOB_ID}
export P=unbound_gas/precompute_masked_sham_hosts.py
export MSTAR_FIT=../data/fit_dsigma_ksz/fit_dsigma_ksz_z05_mstar.npz

small() {  # the small boxes side by side (default fits: z05 + simba50)
  export OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 NUMEXPR_MAX_THREADS=16
  local rc=0
  python -u $P --sim-type IllustrisTNG --name TNG100-1 --snapshot 67 --ptype tau --projection yz > "${LOG}_tng100.log" 2>&1 &
  python -u $P --sim-type IllustrisTNG --name TNG300-1 --snapshot 67 --ptype tau --projection yz > "${LOG}_tng300.log" 2>&1 &
  python -u $P --sim-type IllustrisTNG --name Illustris-1 --snapshot 103 --ptype tau --projection yz > "${LOG}_ill1.log" 2>&1 &
  python -u $P --sim-type SIMBA --name m100n1024 --snapshot 125 --feedback s50 --ptype tau --projection yz > "${LOG}_s100.log" 2>&1 &
  python -u $P --sim-type SIMBA --name m50n512 --snapshot 125 --feedback s50noagn --ptype tau --projection yz > "${LOG}_noagn.log" 2>&1 &
  python -u $P --sim-type SIMBA --name m50n512 --snapshot 125 --feedback s50nox --ptype tau --projection yz > "${LOG}_nox.log" 2>&1 &
  python -u $P --sim-type SIMBA --name m50n512 --snapshot 125 --feedback s50nofb --ptype tau --projection yz > "${LOG}_nofb.log" 2>&1 &
  for j in $(jobs -p); do wait "$j" || rc=1; done
  echo "small runs: exit $rc"
  return $rc
}

flamingo() {  # FLAMINGO variants one after the other; M*-sigma runs use their own fit
  export OMP_NUM_THREADS=128 NUMBA_NUM_THREADS=128 NUMEXPR_MAX_THREADS=128
  local rc=0 s fb fit
  for fb in "$@"; do
    fit=()
    case $fb in Mstar*) fit=(--fit "$MSTAR_FIT") ;; esac
    python -u $P --sim-type FLAMINGO --name L1_m9 --snapshot 67 --feedback "$fb" \
      --ptype tau --projection yz "${fit[@]}" > "${LOG}_${fb}.log" 2>&1
    s=$?
    [ $s -ne 0 ] && rc=1
    echo "FLAMINGO $fb: exit $s"
  done
  return $rc
}
export -f small flamingo

srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores \
  bash -c 'small; a=$?; flamingo Mstar-1sigma_fgas-4sigma; b=$?; exit $((a | b))' &
srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores \
  bash -c 'flamingo L1_m9 fgas-8sigma' &
srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores \
  bash -c 'flamingo Jet_fgas-4sigma Mstar-1sigma' &
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done

R=$(realpath /pscratch/sd/r/rhliu)/simulations
ls -la "$R"/IllustrisTNG/products/2D/masked/ "$R"/SIMBA/products/2D/masked/ "$R"/FLAMINGO/products/2D/masked/ | grep "tau.*_yz_map_masked"
echo "########## EXIT CODE: $rc ##########"
exit $rc
