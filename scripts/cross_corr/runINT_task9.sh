#! /bin/bash -l

# Interactive-QOS runner for Task 9 of the v0.2 addendum: measure the 2D
# spectra and the k_50 aperture-to-wavenumber mapping, then draw the figure.
#
# The spectra step needs the cached projected fields (it measures
# P_tt(k)/P_mm(k) directly, which cannot come from the filtered amplitudes);
# the plotting step is pure post-processing but is run here too so the whole
# task completes in one allocation.
#
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 1:00:00 -A desi bash cross_corr/runINT_task9.sh

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

status=0

for z in z05 z026; do
  echo "########## Task 9 spectra, ${z} ##########"
  srun -n 1 -c 256 --cpu-bind=cores python -u \
    cross_corr/make_task9_spectra.py \
    -p configs/cross_corr/calibration_${z}.yaml || status=1
done

for z in z05 z026; do
  echo "########## Task 9 figure, ${z} ##########"
  srun -n 1 -c 256 --cpu-bind=cores python -u \
    cross_corr/plot_task9.py \
    -p configs/cross_corr/calibration_${z}.yaml || status=1
done

echo "########## EXIT CODE: $status ##########"
exit $status
