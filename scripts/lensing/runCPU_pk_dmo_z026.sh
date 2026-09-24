#! /bin/bash -l

# Batch runner for the z ~ 0.26 hydro/DMO power spectra behind the dotted
# curves of lensing/plot_pk_suppression.py (configs/lensing/pk_dmo_z026.yaml).
# Stages, in order; the job stops at the first failed stage:
#
#   fields  unbound_gas/precompute_baryonFraction_fields.py: 3D gas, Stars, BH
#           and then total fields of every entry whose `total` cache is
#           missing (components first, so `total` only has to bin the DM)
#   mass    lensing/pk_dmo_checks.py mass: sum(total) = Omega_m rho_crit V
#   dmo     unbound_gas/build_dmo_field.py: 3D DM field of each DMO run whose
#           snapshot is complete on disk (mass-checked; skips built ones)
#   spectra unbound_gas/compute_pk_components.py --what dmo
#   report  lensing/pk_dmo_checks.py report: S(k), r(k); fails if a
#           *_Pk_dmo_<n>.npz is missing
#
# Every stage skips outputs that already exist, so a resubmission after a
# timeout resumes where the last job stopped.
#
# Cost (from the z ~ 0.5 runs, one CPU node): FLAMINGO 2000^3 per variant gas
# 15 min, Stars 2.5, BH 1, total 18 (peak RSS 150-193 GB); FLAMINGO DMO field
# 18 min; FLAMINGO dmo spectra ~6 min each. TNG300-1-Dark DMO field 10 min,
# Illustris-1-Dark 3 min, 1000^3 spectra ~1 min. TNG300-1 hydro fields at
# 1000^3: not measured.
#
# Submit from the scripts/ directory, e.g.
#   # FLAMINGO (all three variants share the name L1_m9):
#   SIMS="L1_m9" sbatch -t 04:00:00 lensing/runCPU_pk_dmo_z026.sh
#   # TNG300-1 and Illustris-1, after the DMO download job finishes:
#   SIMS="TNG300-1 Illustris-1" sbatch -t 02:00:00 --dependency=afterok:<fetch job> \
#       lensing/runCPU_pk_dmo_z026.sh
#   # smoke test in debug:
#   STAGES="fields" PTYPES="BH" SIMS="fgas-8sigma TNG300-1" \
#       sbatch -q debug -t 00:20:00 lensing/runCPU_pk_dmo_z026.sh
# SIMS matches config names, feedback variants or labels; word-split, so a
# label with a space cannot be passed.

#SBATCH -A desi
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH --job-name=pk_dmo_z026
#SBATCH -o ../Outputs_Perlmutter/pk_dmo_z026-%j.out

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

# Whole node: abacusnbody's TSC runs at nthread=-1 (numba), Pylians uses OpenMP.
export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
# numexpr caps itself at NUMEXPR_MAX_THREADS (default 64) and prints a line
# starting with "Error." when OMP_NUM_THREADS exceeds it; raise the cap.
export NUMEXPR_MAX_THREADS=128

CONFIG=configs/lensing/pk_dmo_z026.yaml
STAGES=${STAGES:-"fields mass dmo spectra report"}
PTYPES=${PTYPES:-"gas Stars BH total"}
SIM_ARGS=()
if [ -n "${SIMS:-}" ]; then
    # shellcheck disable=SC2206  # word-split on purpose: SIMS is a list of names
    SIM_ARGS=(--sims $SIMS)
fi
JOB=${SLURM_JOB_ID:-local}
LOGDIR=../Outputs_Perlmutter
mkdir -p "$LOGDIR"

# run_py LOG ARGS...: one whole-node python step, appended to LOG. stdin is
# closed so srun never eats the task list of the loop it runs in.
run_py () {
    local log=$1; shift
    srun -N 1 -n 1 -c 256 --cpu-bind=cores python -u "$@" >> "$log" 2>&1 < /dev/null
}

stage_fields () {
    local log=$1 rc=0 simtype sim snap fb npix z pt
    local -a args
    local tasks
    tasks=$(mktemp "$LOGDIR/pkz026_fields-$JOB.tasks.XXXX")
    python lensing/pk_dmo_checks.py -p "$CONFIG" tasks "${SIM_ARGS[@]}" > "$tasks" 2>> "$log" \
        || { rm -f "$tasks"; return 1; }
    while read -r simtype sim snap fb npix z <&3; do
        for pt in $PTYPES; do
            args=(--simtype "$simtype" --sim "$sim" --snapshot "$snap" --ptype "$pt"
                  --dim 3D --n-pixels "$npix" --redshift "$z")
            [ "$fb" != "-" ] && args+=(--feedback "$fb")
            echo "  [fields] $sim ${fb#-} $snap $pt start $(date +%H:%M:%S)"
            if ! run_py "$log" unbound_gas/precompute_baryonFraction_fields.py "${args[@]}"; then
                echo "  [fields] $sim ${fb#-} $snap $pt FAILED"
                rc=1
            fi
        done
    done 3< "$tasks"
    rm -f "$tasks"
    return $rc
}

echo "########## hydro/DMO P(k) at z ~ 0.26 ##########"
echo "job $JOB; stages: $STAGES; sims: ${SIMS:-all}; field ptypes: $PTYPES"
for stage in $STAGES; do
    log="$LOGDIR/pkz026_${stage}-${JOB}.out"
    echo "[$stage] start $(date +%H:%M:%S), log $log"
    case $stage in
        fields)  stage_fields "$log" ;;
        mass)    run_py "$log" lensing/pk_dmo_checks.py -p "$CONFIG" mass "${SIM_ARGS[@]}" ;;
        dmo)     run_py "$log" unbound_gas/build_dmo_field.py -p "$CONFIG" "${SIM_ARGS[@]}" ;;
        spectra) run_py "$log" unbound_gas/compute_pk_components.py -p "$CONFIG" \
                     --what dmo "${SIM_ARGS[@]}" ;;
        report)  python lensing/pk_dmo_checks.py -p "$CONFIG" report "${SIM_ARGS[@]}" >> "$log" 2>&1 ;;
        *)       echo "unknown stage $stage"; false ;;
    esac
    rc=$?
    echo "EXIT CODE: $rc" >> "$log"
    echo "[$stage] exit $rc at $(date +%H:%M:%S)"
    if [ "$rc" != 0 ]; then
        echo "########## stopping: stage $stage failed ##########"
        exit 1
    fi
done
[ -f "$LOGDIR/pkz026_report-${JOB}.out" ] && cat "$LOGDIR/pkz026_report-${JOB}.out"
echo "########## all stages done ##########"
