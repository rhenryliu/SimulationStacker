#! /bin/bash -l

# Interactive-QOS runner for the Task 1 r-profile sweep: the four retained
# simulations (TNG300-1 + three FLAMINGO variants), yz projection, both
# redshift samples, then the figures and Gate A metrics.
#
# All projected fields are cached, so this is FFT work only: one
# compute_Y_matrix call per (run, projection) over 15 apertures.  The FLAMINGO
# grids dominate (8869^2 at z ~ 0.5, 14015^2 at z ~ 0.30); expect roughly 30
# minutes for both configs.  Interactive rather than regular is deliberate: a
# half-hour FFT job on cached fields turns around in seconds here and sat
# pending in the regular queue (docs/tasks_7_to_10_record.md Stage 4).
#
# Launched from the scripts/ directory via:
#   salloc -q interactive -C cpu -N 1 -t 1:00:00 -A desi bash cross_corr/runINT_rprofiles.sh
#
# Do NOT wrap this in an outer srun: the runner creates its own job step, and
# nesting them deadlocks with only the extern step alive.

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

echo "=== python: $(which python) ==="
echo "=== started: $(date) ==="
status=0

echo
echo "########## r-profiles, z ~ 0.5 (LRG-like) ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u cross_corr/make_r_profiles.py \
  -p configs/cross_corr/r_profiles_z05.yaml || status=1

echo
echo "########## r-profiles, z ~ 0.26 / 0.30 (BGS-like) ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u cross_corr/make_r_profiles.py \
  -p configs/cross_corr/r_profiles_z026.yaml || status=1

# The figures and the Gate A metrics are regenerated here rather than left to
# a separate step: without them, a sweep that changes R0 or Rmax exits 0 while
# the figures on disk still show the previous configuration, with nothing to
# say so.
echo
echo "########## figures and Gate A metrics ##########"
srun -n 1 -c 256 --cpu-bind=cores python -u cross_corr/plot_r_profiles.py \
  -p configs/cross_corr/r_profiles_z05.yaml || status=1
srun -n 1 -c 256 --cpu-bind=cores python -u cross_corr/plot_r_profiles.py \
  -p configs/cross_corr/r_profiles_z026.yaml || status=1

echo
echo "=== finished: $(date) ==="
echo "########## EXIT CODE: $status ##########"
exit $status
