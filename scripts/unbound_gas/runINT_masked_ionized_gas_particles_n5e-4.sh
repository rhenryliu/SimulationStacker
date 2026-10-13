#! /bin/bash -l

# Masked ionized-gas fields at 0.2' (yz, no beam; 1, 2, 3 x R200m around the
# host haloes of the SHAM galaxies at the fixed density n = 5e-4 (cMpc/h)^-3
# of Liu et al. 2026's LRG comparison) for the DeltaSigma ratio masking figure
# (dsigma_ratio_z05_masked_nobeam_n5e-4.yaml), built at particle level by
# precompute_masked_particles.py --density: as runINT_masked_ionized_gas_particles.sh,
# whose maps are at the lensing-fit densities. Both write the same cache paths:
# the lensing-fit maps were moved to masked/lensfit_sham_hosts/ (MANIFEST.txt
# there) before this run, and existing maps are skipped, never overwritten.
# TNG100-1 and SIMBA-100 also check their keep-all map against the cached 0.2'
# field.
#
# Same layout as the lensing-fit run (job 59702660: 49 min, 249 GB peak per
# node): three nodes, two FLAMINGO variants side by side on each of the first
# two, the fifth with TNG300-1 and the small boxes on the third.
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 3 -t 02:00:00 -A desi -J masks_ig_n5e-4 bash unbound_gas/runINT_masked_ionized_gas_particles_n5e-4.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python

export LOG=../Outputs_Perlmutter/masks_ig_n5e-4-${SLURM_JOB_ID}
export P=unbound_gas/precompute_masked_particles.py
export N=5e-4

run() {  # tag, threads, then precompute arguments; one log per run
  local tag=$1 nt=$2 s
  shift 2
  OMP_NUM_THREADS=$nt NUMBA_NUM_THREADS=$nt NUMEXPR_MAX_THREADS=$nt \
    python -u $P --density $N "$@" > "${LOG}_${tag}.log" 2>&1
  s=$?
  echo "$tag: exit $s"
  return $s
}
flamingo() {  # FLAMINGO variants side by side on one node
  local rc=0 fb
  for fb in "$@"; do
    run "$fb" 64 --sim-type FLAMINGO --name L1_m9 --snapshot 67 --feedback "$fb" &
  done
  for j in $(jobs -p); do wait "$j" || rc=1; done
  return $rc
}
rest() {  # the fifth FLAMINGO variant, TNG300-1 and the small boxes side by side
  local rc=0
  run Mstar-1sigma_fgas-4sigma 64 --sim-type FLAMINGO --name L1_m9 --snapshot 67 --feedback Mstar-1sigma_fgas-4sigma &
  run tng300 32 --sim-type IllustrisTNG --name TNG300-1 --snapshot 67 &
  run tng100 16 --sim-type IllustrisTNG --name TNG100-1 --snapshot 67 --keep-all &
  run ill1 16 --sim-type IllustrisTNG --name Illustris-1 --snapshot 103 &
  run s100 16 --sim-type SIMBA --name m100n1024 --snapshot 125 --feedback s50 --keep-all &
  run noagn 8 --sim-type SIMBA --name m50n512 --snapshot 125 --feedback s50noagn &
  run nox 8 --sim-type SIMBA --name m50n512 --snapshot 125 --feedback s50nox &
  run nofb 8 --sim-type SIMBA --name m50n512 --snapshot 125 --feedback s50nofb &
  for j in $(jobs -p); do wait "$j" || rc=1; done
  return $rc
}
export -f run flamingo rest

srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores bash -c 'flamingo L1_m9 fgas-8sigma' &
srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores bash -c 'flamingo Jet_fgas-4sigma Mstar-1sigma' &
srun --nodes=1 --ntasks=1 --exclusive -c 256 --cpu-bind=cores bash -c 'rest' &
rc=0
for j in $(jobs -p); do wait "$j" || rc=1; done

R=$(realpath /pscratch/sd/r/rhliu)/simulations
ls -la "$R"/IllustrisTNG/products/2D/masked/ "$R"/SIMBA/products/2D/masked/ "$R"/FLAMINGO/products/2D/masked/ \
  | grep -E "_(67|103|125)_ionized_gas.*_yz_masked"
echo "########## EXIT CODE: $rc ##########"
exit $rc
