#! /bin/bash -l

# Interactive-QOS runner for make_frb_dm_xi.py (simulated FRB DM-galaxy
# cross-correlation, after McCarty et al. Figure 5; predictions in
# docs/cross_corr/predictions/2026-10-20_frb_dm_xi.md). Acceptance check A
# (regression against the committed TNG300-1 r-profiles) runs first; the full
# run (four snapshots from the cached 0.2' fields) only if it passes; the
# whole job took 5 min 18 s (job 60078819).
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 0:30:00 -A desi bash cross_corr/runINT_frb_dm_xi.sh
# Do NOT wrap this in an outer srun: the runner creates its own job steps.

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1
CONFIG=configs/cross_corr/frb_dm_xi_z03.yaml

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
which python

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

echo "########## check A: r-profile regression (TNG300-1, snapshot 80) ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u cross_corr/make_frb_dm_xi.py -p "$CONFIG" \
  --check-against ../data/r_profiles/r_profiles_TNG300-1_80_yz.npz || { echo "check A failed"; exit 1; }

echo "########## make_frb_dm_xi.py -p $CONFIG ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u cross_corr/make_frb_dm_xi.py -p "$CONFIG"
rc=$?
echo "########## EXIT CODE: $rc ##########"
exit $rc
