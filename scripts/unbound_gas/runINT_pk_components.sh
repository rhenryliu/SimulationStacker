#! /bin/bash -l

# Interactive-QOS runner for the matter power spectrum section of the unbound
# gas paper (configs/unbound_gas/pk_components_z05.yaml). Stages, in order:
#
#   validate  validate_pk_caches.py: consistency checks on the cached 3D fields,
#             re-binning Stars from particles for one simulation per suite
#   dmo       build_dmo_field.py: 3D DM field of each DMO reference run whose
#             snapshot is on disk (FLAMINGO 2000^3 ~20 min; skips built ones)
#   spectra   compute_pk_components.py: component + DMO spectra (skips done ones)
#
# The figures/tables (make_pk_alpha.py) are light and run on a login node.
#
# Memory: at FLAMINGO's 2000^3 grid each float32 field is 32 GB. The spectra
# stage holds the five component fields plus Pylians' five FFTs (~320 GB); the
# validation stage holds five fields (~160 GB). One whole 512 GB node each.
#
# Launch from the scripts/ directory:
#   salloc -q interactive -C cpu -N 1 -t 4:00:00 -A desi \
#       bash unbound_gas/runINT_pk_components.sh
#   # a subset of stages and simulations:
#   STAGES="spectra" SIMS="TNG300-1" salloc -q interactive -C cpu -N 1 -t 1:00:00 \
#       -A desi bash unbound_gas/runINT_pk_components.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

# Whole node: abacusnbody's TSC runs at nthread=-1 (numba), Pylians uses OpenMP.
export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
# numexpr caps itself at NUMEXPR_MAX_THREADS (default 64) and prints a line
# starting with "Error." when OMP_NUM_THREADS exceeds it. Harmless, but it
# reads like a failure in the logs -- raise the cap to match.
export NUMEXPR_MAX_THREADS=128

CONFIG=configs/unbound_gas/pk_components_z05.yaml
STAGES=${STAGES:-"validate dmo spectra"}
SIM_ARGS=()
if [ -n "${SIMS:-}" ]; then
    # shellcheck disable=SC2206  # word-split on purpose: SIMS is a list of names
    SIM_ARGS=(--sims $SIMS)
fi
JOB=${SLURM_JOB_ID:-local}
LOGDIR=../Outputs_Perlmutter
mkdir -p "$LOGDIR"

run_stage () {
    local name=$1; shift
    local log="$LOGDIR/pk_${name}-${JOB}.out"
    echo "  [$name] start $(date +%H:%M:%S), log $log"
    srun -N 1 -n 1 -c 256 --cpu-bind=cores python -u "$@" > "$log" 2>&1
    local rc=$?
    echo "EXIT CODE: $rc" >> "$log"
    echo "  [$name] exit $rc at $(date +%H:%M:%S)"
    return $rc
}

echo "########## P(k) components (interactive) ##########"
echo "allocation : job $JOB; stages: $STAGES; sims: ${SIMS:-all}"
FAILED=0
for stage in $STAGES; do
    case $stage in
        validate)
            run_stage validate unbound_gas/validate_pk_caches.py -p "$CONFIG" "${SIM_ARGS[@]}" \
                --rebuild Stars --rebuild-sims TNG300-1 Illustris-1 m100n1024 'L1_m9 (L1_m9)' \
                || FAILED=$((FAILED + 1)) ;;
        dmo)
            run_stage dmo unbound_gas/build_dmo_field.py -p "$CONFIG" "${SIM_ARGS[@]}" \
                || FAILED=$((FAILED + 1)) ;;
        spectra)
            run_stage spectra unbound_gas/compute_pk_components.py -p "$CONFIG" "${SIM_ARGS[@]}" \
                || FAILED=$((FAILED + 1)) ;;
        *) echo "unknown stage $stage"; FAILED=$((FAILED + 1)) ;;
    esac
done
echo "failed stages: $FAILED"
exit $((FAILED > 0))
