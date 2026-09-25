#! /bin/bash -l

# Interactive-QOS runner for the stellar-fraction bands (make_pk_fstar.py): for
# each simulation, compute_fstar.py (one pass over its stars, gas, winds and
# BH, then the floor's 3D field and spectra and 2D maps for every
# configuration) and then stack_fstar_maps.py (the DSigma stacks of the maps,
# in a process pool), on one node; at most $SLURM_JOB_NUM_NODES simulations at
# a time, in the order of $SIMS (FLAMINGO variants and TNG300-1 first).
#
# Needs the s = 0 outputs (compute_pk_stellar.py, compute_stellar_maps.py,
# stack_stellar_maps.py) for the f* = 0 end and the stacking sample check.
# Peak memory: the particle pass (up to ~300 GB for TNG300-1) plus up to ~5
# full 3D grids (FLAMINGO 2000^3) -- one whole 512 GB node per simulation.
#
# Launch from the scripts/ directory:
#   salloc -q interactive -C cpu -N 4 -t 3:00:00 -A desi --no-shell
#   SLURM_JOB_ID=<id> SLURM_JOB_NUM_NODES=4 bash unbound_gas/runINT_fstar.sh
# Environment variables:
#   CONFIG  config with `fstar` and `lensing` blocks (default: z ~ 0.5), e.g.
#           CONFIG=configs/unbound_gas/pk_fstar_z026.yaml
#   SIMS    simulations (default: all six of the z ~ 0.5 config)
#   STAGES  "compute stack" (default), or one of them
#   EXTRA   extra arguments for compute_fstar.py, e.g.
#           EXTRA="--max-chunks 2" STAGES=compute (smoke test, nothing saved)
#
# Finished variants are skipped unless EXTRA="--overwrite"; the stacks are
# always rewritten.

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
# numexpr caps itself at NUMEXPR_MAX_THREADS (default 64) and prints a line
# starting with "Error." when OMP_NUM_THREADS exceeds it. Harmless, but it
# reads like a failure in the logs -- raise the cap to match.
export NUMEXPR_MAX_THREADS=128

CONFIG=${CONFIG:-configs/unbound_gas/pk_fstar_z05.yaml}
SIMS=${SIMS:-"L1_m9 fgas-8sigma Jet_fgas-4sigma TNG300-1 Illustris-1 m100n1024"}
STAGES=${STAGES:-"compute stack"}
EXTRA=${EXTRA:-""}
NODES=${SLURM_JOB_NUM_NODES:-1}
JOB=${SLURM_JOB_ID:-local}
LOGDIR=../Outputs_Perlmutter
mkdir -p "$LOGDIR"

echo "########## stellar-fraction floor: fields, maps and stacks (interactive) ##########"
echo "allocation : job $JOB, $NODES node(s); config: $CONFIG; sims: $SIMS; stages: $STAGES; extra: $EXTRA"

run_sim () {
    local s=$1
    local log="$LOGDIR/fstar-${JOB}_${s}.out"
    # 'L1_m9' as a --sims filter would select all three FLAMINGO variants
    # (it is their shared name); pass the fiducial by its full label instead.
    local sel=$s
    [ "$s" = "L1_m9" ] && sel="L1_m9 (L1_m9)"
    local rc=0
    : > "$log"
    if [[ " $STAGES " == *" compute "* ]]; then
        # shellcheck disable=SC2086
        srun -N 1 -n 1 -c 256 --exclusive --cpu-bind=cores \
            python -u unbound_gas/compute_fstar.py -p "$CONFIG" --sims "$sel" $EXTRA >> "$log" 2>&1
        rc=$?
        echo "COMPUTE EXIT CODE: $rc" >> "$log"
    fi
    if [ "$rc" = "0" ] && [[ " $STAGES " == *" stack "* ]]; then
        # one single-threaded process per stack
        OMP_NUM_THREADS=1 srun -N 1 -n 1 -c 256 --exclusive --cpu-bind=cores \
            python -u unbound_gas/stack_fstar_maps.py -p "$CONFIG" --sims "$sel" >> "$log" 2>&1
        rc=$?
        echo "STACK EXIT CODE: $rc" >> "$log"
    fi
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
    [ "$(cat "$LOGDIR/fstar-${JOB}_${s}.out.rc" 2>/dev/null)" = "0" ] || FAILED=$((FAILED + 1))
done
echo "failed: $FAILED"
exit $((FAILED > 0))
