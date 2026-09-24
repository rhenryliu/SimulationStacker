#! /bin/bash -l

# Interactive-QOS runner for compute_pk_stellar.py (halo-level stellar-to-
# ionized-gas transfer of the unbound gas paper's power spectrum section): one
# simulation per node, at most $SLURM_JOB_NUM_NODES at a time, in the order of
# $SIMS (FLAMINGO variants and TNG300-1 first: the long poles).
#
# Each simulation needs one pass over its stars and gas plus one transfer
# field and FFT per configuration (3 methods x 3 mass cuts). Peak memory is
# set by the kept particles (those in any halo region at the lowest cut) plus
# ~5 full grids -- one whole 512 GB node per simulation.
#
# Launch from the scripts/ directory:
#   salloc -q interactive -C cpu -N 4 -t 4:00:00 -A desi bash unbound_gas/runINT_pk_stellar.sh
#   SIMS="m100n1024" salloc -q interactive -C cpu -N 1 -t 1:00:00 -A desi \
#       bash unbound_gas/runINT_pk_stellar.sh
# Extra arguments for compute_pk_stellar.py go in $EXTRA, e.g.
#   EXTRA="--max-chunks 2" (smoke test, nothing saved) or EXTRA="--variants fof".
# Another config (e.g. z ~ 0.26) via $CONFIG:
#   CONFIG=configs/unbound_gas/pk_stellar_z026.yaml SIMS="TNG300-1 Illustris-1" \
#       salloc -q interactive -C cpu -N 2 -t 2:00:00 -A desi bash unbound_gas/runINT_pk_stellar.sh
#
# Finished method variants (one file each) are skipped unless --overwrite.

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
# numexpr caps itself at NUMEXPR_MAX_THREADS (default 64) and prints a line
# starting with "Error." when OMP_NUM_THREADS exceeds it. Harmless, but it
# reads like a failure in the logs -- raise the cap to match.
export NUMEXPR_MAX_THREADS=128

CONFIG=${CONFIG:-configs/unbound_gas/pk_components_z05.yaml}
SIMS=${SIMS:-"L1_m9 fgas-8sigma Jet_fgas-4sigma TNG300-1 Illustris-1 m100n1024"}
EXTRA=${EXTRA:-""}
NODES=${SLURM_JOB_NUM_NODES:-1}
JOB=${SLURM_JOB_ID:-local}
LOGDIR=../Outputs_Perlmutter
mkdir -p "$LOGDIR"

echo "########## P(k) stellar transfer (interactive) ##########"
echo "allocation : job $JOB, $NODES node(s); config: $CONFIG; sims: $SIMS; extra: $EXTRA"

run_sim () {
    local s=$1
    local log="$LOGDIR/pk_stellar-${JOB}_${s}.out"
    # 'L1_m9' as a --sims filter would select all three FLAMINGO variants
    # (it is their shared name); pass the fiducial by its full label instead.
    local sel=$s
    [ "$s" = "L1_m9" ] && sel="L1_m9 (L1_m9)"
    # shellcheck disable=SC2086
    srun -N 1 -n 1 -c 256 --exclusive --cpu-bind=cores \
        python -u unbound_gas/compute_pk_stellar.py -p "$CONFIG" --sims "$sel" $EXTRA > "$log" 2>&1
    local rc=$?
    echo "EXIT CODE: $rc" >> "$log"
    echo "$rc" > "${log}.rc"
    echo "  [$s] exit $rc at $(date +%H:%M:%S)"
}

for s in $SIMS; do
    while [ "$(jobs -rp | wc -l)" -ge "$NODES" ]; do sleep 5; done
    echo "  [$s] launching ($(date +%H:%M:%S))"
    run_sim "$s" &
done
wait

FAILED=0
for s in $SIMS; do
    [ "$(cat "$LOGDIR/pk_stellar-${JOB}_${s}.out.rc" 2>/dev/null)" = "0" ] || FAILED=$((FAILED + 1))
done
echo "failed: $FAILED"
exit $((FAILED > 0))
