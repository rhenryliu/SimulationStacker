#! /bin/bash -l

# Masked ionized-gas fields at 0.2' (yz, no beam; 1, 2, 3 x R200m around the
# host haloes of the SHAM galaxies at the fixed density n = 1e-3 (cMpc/h)^-3
# of Liu et al. 2026's BGS comparison) for the z ~ 0.26 DeltaSigma ratio
# masking figure (dsigma_ratio_z026_masked_nobeam_n1e-3.yaml), built at particle
# level by precompute_masked_particles.py --density: as
# runINT_masked_ionized_gas_particles_z026.sh, whose maps are at the BGS
# lensing-fit densities. TNG300-1 (80), Illustris-1 (116) and SIMBA-100 (136)
# are projected at z = 0.26, the five FLAMINGO runs (71) at z = 0.30. Both
# runners write the same cache paths: the lensing-fit maps were moved to
# masked/lensfit_sham_hosts/ (MANIFEST.txt there) before this run, and existing
# maps are skipped, never overwritten. Illustris-1 and SIMBA-100 also check
# their keep-all map against the cached 0.2' field.
#
# Same layout as the lensing-fit run (job 59716174: 54 min, 174 GB peak per
# node): three nodes, two FLAMINGO variants side by side on each of the first
# two, the fifth with the other three runs on the third.
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 3 -t 02:00:00 -A desi -J masks_ig_z026_n1e-3 bash unbound_gas/runINT_masked_ionized_gas_particles_z026_n1e-3.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python

export LOG=../Outputs_Perlmutter/masks_ig_z026_n1e-3-${SLURM_JOB_ID}
export P=unbound_gas/precompute_masked_particles.py
export N=1e-3

run() {  # tag, threads, then precompute arguments; one log per run
  local tag=$1 nt=$2 s
  shift 2
  OMP_NUM_THREADS=$nt NUMBA_NUM_THREADS=$nt NUMEXPR_MAX_THREADS=$nt \
    python -u $P --density $N "$@" > "${LOG}_${tag}.log" 2>&1
  s=$?
  echo "$tag: exit $s"
  return $s
}
flamingo() {  # FLAMINGO variants (snapshot 71, z = 0.30) side by side on one node
  local rc=0 fb
  for fb in "$@"; do
    run "$fb" 64 --sim-type FLAMINGO --name L1_m9 --snapshot 71 --feedback "$fb" --z 0.3 &
  done
  for j in $(jobs -p); do wait "$j" || rc=1; done
  return $rc
}
rest() {  # the fifth FLAMINGO variant and the z = 0.26 runs side by side
  local rc=0
  run Mstar-1sigma_fgas-4sigma 64 --sim-type FLAMINGO --name L1_m9 --snapshot 71 \
    --feedback Mstar-1sigma_fgas-4sigma --z 0.3 &
  run tng300 32 --sim-type IllustrisTNG --name TNG300-1 --snapshot 80 --z 0.26 &
  run ill1 16 --sim-type IllustrisTNG --name Illustris-1 --snapshot 116 --z 0.26 --keep-all &
  run s100 16 --sim-type SIMBA --name m100n1024 --snapshot 136 --feedback s50 --z 0.26 --keep-all &
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
  | grep -E "_(80|116|136|71)_ionized_gas.*_yz_masked"
echo "########## EXIT CODE: $rc ##########"
exit $rc
