#! /bin/bash -l

# Runner for Round 3A of the cross-correlation programme (predictions:
# docs/cross_corr/predictions/2026-09-27_round3a.md):
#
#   ck   make_ck_spectra.py       C(k), the exact window/mediation split of C_F,
#                                 the O-05/O-06 galaxy samples, and the
#                                 regression check against round two
#   dog  make_dog_calibration.py  the difference-of-Gaussians C with jackknife
#                                 errors (O-02)
#
# for the four retained runs, yz, both redshift samples, from cached maps only.
#
# Optional variables (pass with sbatch --export=ALL,SIM=...,STAGES=...):
#   SIM=TNG300-1        restrict to one simulation name (smoke test)
#   FEEDBACK=L1_m9      restrict to one FLAMINGO variant (with SIM=L1_m9)
#   STAGES="ck dog"     stages to run (default: both)
#   CONFIGS="z05 z026"  configs to run (default: both)
#
# Submit from scripts/ (the log paths and every path below are relative to it).
# The defaults below are a debug-QOS smoke test; override for the full run:
#   cd scripts/
#   sbatch --export=ALL,SIM=TNG300-1,CONFIGS=z05 cross_corr/runCPU_round3a.sh
#   sbatch --qos=regular --time=02:00:00 cross_corr/runCPU_round3a.sh
# The same file also runs inside an interactive allocation:
#   salloc -q interactive -C cpu -N 1 -t 2:00:00 -A desi bash cross_corr/runCPU_round3a.sh

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --job-name=round3a
#SBATCH -o ../Outputs_Perlmutter/slurm-%x-%j.out
#SBATCH -e ../Outputs_Perlmutter/slurm-%x-%j.err

set -o pipefail

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

STAGES=${STAGES:-"ck dog"}
CONFIGS=${CONFIGS:-"z05 z026"}
SIM_ARG=()
if [ -n "${SIM:-}" ]; then
  SIM_ARG=(--sim "$SIM")
fi
if [ -n "${FEEDBACK:-}" ]; then
  SIM_ARG+=(--feedback "$FEEDBACK")
fi

echo "=== python: $(which python) ==="
echo "=== started: $(date); stages: ${STAGES}; configs: ${CONFIGS}; sim: ${SIM:-all} ==="
status=0
for z in $CONFIGS; do
  for stage in $STAGES; do
    case $stage in
      ck)  script=cross_corr/make_ck_spectra.py ;;
      dog) script=cross_corr/make_dog_calibration.py ;;
      *)   echo "Unknown stage: $stage"; status=1; continue ;;
    esac
    echo
    echo "############ Round 3A: ${stage}, ${z} ############"
    srun -n 1 -c 256 --cpu-bind=cores python -u "$script" \
      -p "configs/cross_corr/round3a_${z}.yaml" "${SIM_ARG[@]}" || status=1
  done
done

echo
echo "=== finished: $(date) ==="
if [ $status -ne 0 ]; then
    echo "ONE OR MORE STEPS FAILED"
    exit 1
fi
exit 0
