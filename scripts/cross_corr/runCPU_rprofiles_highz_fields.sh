#! /bin/bash -l

# Build and cache the 0.2' yz projected fields that make_r_profiles.py needs at
# the two new redshift slices (configs r_profiles_z075.yaml, r_profiles_z10.yaml):
#
#   z ~ 0.75: TNG300-1 57 (z = 0.7574), FLAMINGO L1_m9 / fgas-8sigma / Jet_fgas-4sigma 62 (z = 0.75)
#   z ~ 1.0 : TNG300-1 50 (z = 0.9973), FLAMINGO ... 57 (z = 1.00)
#
# One task = one call of unbound_gas/precompute_baryonFraction_fields.py for a
# (run, snapshot, particle type), which sizes the grid as ceil(box angle / 0.2')
# at --redshift, writes the raw field through makeMap(beamSize=0) and skips
# fields that are already cached (so this runner is safe to resubmit). The
# --redshift passed is each snapshot's header value, the same value the configs
# carry per run.
#
# Order matters: mapMaker.make_combined_field builds 'baryon' and 'total' from
# their components but does not save the components, so the components (gas,
# DM, Stars, BH) are built and saved first and the combined fields, built
# last, only load them.
#
# STAGE=smoke : Stars and BH only (fast), every run -- exercises every new
#               snapshot reader and the save path. Submit with the debug QOS.
# STAGE=full  : stage 1 = gas, DM, ionized_gas, Stars, BH (cached ones skip),
#               stage 2 = baryon, total.
# DRYRUN=1    : print the task list and exit.
#
# The 2D binning (scipy binned_statistic_2d) is single-threaded, so tasks run
# NPAR at a time with one thread each; per-task logs, elapsed time and peak RSS
# go to ../Outputs_Perlmutter/rprofiles_highz_fields-<jobid>/.
#
# Submit from scripts/:
#   STAGE=smoke sbatch --qos=debug --time=00:30:00 cross_corr/runCPU_rprofiles_highz_fields.sh
#   STAGE=full  sbatch cross_corr/runCPU_rprofiles_highz_fields.sh

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --job-name=rprofiles_highz_fields
#SBATCH -o ../Outputs_Perlmutter/slurm-%x-%j.out
#SBATCH -e ../Outputs_Perlmutter/slurm-%x-%j.err

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

STAGE=${STAGE:-full}
NPAR=${NPAR:-12}
export LOGDIR=../Outputs_Perlmutter/rprofiles_highz_fields-${SLURM_JOB_ID:-local}

# simtype sim snapshot feedback redshift  (feedback '-' = none)
RUNS="IllustrisTNG TNG300-1 57 - 0.7574
IllustrisTNG TNG300-1 50 - 0.9973
FLAMINGO L1_m9 62 L1_m9 0.75
FLAMINGO L1_m9 62 fgas-8sigma 0.75
FLAMINGO L1_m9 62 Jet_fgas-4sigma 0.75
FLAMINGO L1_m9 57 L1_m9 1.0
FLAMINGO L1_m9 57 fgas-8sigma 1.0
FLAMINGO L1_m9 57 Jet_fgas-4sigma 1.0"

case $STAGE in
    smoke) STAGE_PTYPES=("Stars BH") ;;
    full)  STAGE_PTYPES=("gas DM ionized_gas Stars BH" "baryon total") ;;
    *) echo "unknown STAGE=$STAGE" >&2; exit 1 ;;
esac

# run_task SIMTYPE SIM SNAP FEEDBACK Z PTYPE
run_task () {
    local simtype=$1 sim=$2 snap=$3 fb=$4 z=$5 ptype=$6
    local tag="${sim}_${fb}_${snap}_${ptype}" rc secs kb
    local -a fbarg=()
    [ "$fb" != "-" ] && fbarg=(--feedback "$fb")
    /usr/bin/time -f "%e %M" -o "$LOGDIR/$tag.time" \
        python -u unbound_gas/precompute_baryonFraction_fields.py \
            --simtype "$simtype" --sim "$sim" --snapshot "$snap" "${fbarg[@]}" \
            --ptype "$ptype" --dim 2D --projection yz --pixel-size 0.2 \
            --redshift "$z" > "$LOGDIR/$tag.log" 2>&1
    rc=$?
    # The last line of the time file holds "elapsed maxrss"; a task killed by a
    # signal gets a "Command terminated" line first, or no file at all.
    secs=$(tail -n 1 "$LOGDIR/$tag.time" 2>/dev/null | awk 'NF==2 && $1+0==$1 {print $1}')
    kb=$(tail -n 1 "$LOGDIR/$tag.time" 2>/dev/null | awk 'NF==2 && $2+0==$2 {print $2}')
    if [ -n "$secs" ] && [ -n "$kb" ]; then
        printf '%-4s %-44s %7.1f min  peak %6.1f GB  (%s)\n' \
            "$([ $rc = 0 ] && echo OK || echo FAIL)" "$tag" \
            "$(echo "$secs / 60" | bc -l)" "$(echo "$kb / 1048576" | bc -l)" "$(date +%H:%M)"
    else
        printf '%-4s %-44s  time/peak n/a (exit %s)  (%s)\n' \
            "$([ $rc = 0 ] && echo OK || echo FAIL)" "$tag" "$rc" "$(date +%H:%M)"
    fi
    return $rc
}
export -f run_task

echo "=== stage=$STAGE  parallel=$NPAR  python=$(which python)  started $(date) ==="
mkdir -p "$LOGDIR"
status=0
for ptypes in "${STAGE_PTYPES[@]}"; do
    echo
    echo "########## particle types: $ptypes ##########"
    tasks=$(while read -r simtype sim snap fb z; do
                for p in $ptypes; do echo "$simtype $sim $snap $fb $z $p"; done
            done <<< "$RUNS")
    if [ "${DRYRUN:-0}" = 1 ]; then
        echo "$tasks"
        continue
    fi
    echo "$tasks" | xargs -P "$NPAR" -L 1 bash -c 'run_task "$@"' _ || status=1
    # Stage 2 loads the stage-1 components; stop if any of them failed.
    [ $status != 0 ] && { echo "stage failed; not continuing"; break; }
done

echo
echo "=== finished $(date); logs in $LOGDIR ==="
echo "########## EXIT CODE: $status ##########"
exit $status
