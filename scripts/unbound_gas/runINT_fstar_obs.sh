#! /bin/bash -l

# Interactive-QOS runner for the observation-based stellar bands
# (make_pk_fstar_obs.py): for each (redshift, simulation) run,
# compute_fstar_obs.py (s of every end; for the capped ends one pass over the
# stars, gas, winds and BH, then the 3D field, spectra and 2D maps) and then
# stack_fstar_obs_maps.py (the DSigma stacks of those maps, in a process
# pool), on one node; at most $SLURM_JOB_NUM_NODES runs at a time, in the
# order of $RUNS. Log names carry the redshift, so both redshifts can share
# one allocation.
#
# Needs the s = 0 outputs (compute_pk_stellar.py, compute_stellar_maps.py,
# stack_stellar_maps.py) and the f* floor spectra (compute_fstar.py) of the
# same snapshots. Peak memory: the particle pass plus up to ~5 full 3D grids
# (FLAMINGO 2000^3) -- one whole 512 GB node per run.
#
# Launch from the scripts/ directory:
#   salloc -q interactive -C cpu -N 4 -t 2:30:00 -A desi --no-shell
#   SLURM_JOB_ID=<id> SLURM_JOB_NUM_NODES=4 bash unbound_gas/runINT_fstar_obs.sh
# Environment variables:
#   RUNS    "<z>:<sim>" pairs; <z> selects configs/unbound_gas/pk_fstar_obs_<z>.yaml
#           (z05 or z026). Default: the runs with a capped end
#           (compute_fstar_obs.py --solve-only, 2026-09-25), longest first;
#           TNG300-1 at z05 needs no pass (run it on the login node).
#   STAGES  "compute stack" (default), or one of them
#   EXTRA   extra arguments for compute_fstar_obs.py, e.g.
#           EXTRA="--max-chunks 2" STAGES=compute (smoke test, nothing saved)
#
# Finished runs are skipped unless EXTRA="--overwrite"; the stacks are
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

RUNS=${RUNS:-"z05:L1_m9 z05:fgas-8sigma z05:Jet_fgas-4sigma z026:L1_m9 z026:fgas-8sigma z026:Jet_fgas-4sigma z026:TNG300-1 z05:Illustris-1 z026:Illustris-1"}
STAGES=${STAGES:-"compute stack"}
EXTRA=${EXTRA:-""}
NODES=${SLURM_JOB_NUM_NODES:-1}
JOB=${SLURM_JOB_ID:-local}
LOGDIR=../Outputs_Perlmutter
mkdir -p "$LOGDIR"

echo "########## observation-based stellar bands: fields, maps and stacks (interactive) ##########"
echo "allocation : job $JOB, $NODES node(s); runs: $RUNS; stages: $STAGES; extra: $EXTRA"

run_one () {
    local z=${1%%:*}
    local s=${1#*:}
    local config="configs/unbound_gas/pk_fstar_obs_${z}.yaml"
    local log="$LOGDIR/fstar_obs-${JOB}_${z}_${s}.out"
    # 'L1_m9' as a --sims filter would select all three FLAMINGO variants
    # (it is their shared name); pass the fiducial by its full label instead.
    local sel=$s
    [ "$s" = "L1_m9" ] && sel="L1_m9 (L1_m9)"
    local rc=0
    : > "$log"
    if [ ! -f "$config" ]; then
        echo "no config $config" >> "$log"
        rc=2
    fi
    if [ "$rc" = "0" ] && [[ " $STAGES " == *" compute "* ]]; then
        # shellcheck disable=SC2086
        srun -N 1 -n 1 -c 256 --exclusive --cpu-bind=cores \
            python -u unbound_gas/compute_fstar_obs.py -p "$config" --sims "$sel" $EXTRA >> "$log" 2>&1
        rc=$?
        echo "COMPUTE EXIT CODE: $rc" >> "$log"
    fi
    if [ "$rc" = "0" ] && [[ " $STAGES " == *" stack "* ]]; then
        # one single-threaded process per stack
        OMP_NUM_THREADS=1 srun -N 1 -n 1 -c 256 --exclusive --cpu-bind=cores \
            python -u unbound_gas/stack_fstar_obs_maps.py -p "$config" --sims "$sel" >> "$log" 2>&1
        rc=$?
        echo "STACK EXIT CODE: $rc" >> "$log"
    fi
    echo "$rc" > "${log}.rc"
    echo "  [$z $s] exit $rc at $(date +%H:%M:%S)"
}

for r in $RUNS; do
    while [ "$(jobs -rp | wc -l)" -ge "$NODES" ]; do sleep 5; done
    echo "  [$r] launching ($(date +%H:%M:%S))"
    run_one "$r" &
done
wait

FAILED=0
for r in $RUNS; do
    [ "$(cat "$LOGDIR/fstar_obs-${JOB}_${r%%:*}_${r#*:}.out.rc" 2>/dev/null)" = "0" ] || FAILED=$((FAILED + 1))
done
echo "failed: $FAILED"
exit $((FAILED > 0))
