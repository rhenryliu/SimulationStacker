#! /bin/bash -l

# Task 1 r-profiles and figures at the two new redshift slices, from fields
# already cached by runCPU_rprofiles_highz_fields.sh:
#   configs/cross_corr/r_profiles_z075.yaml  (TNG300-1 57, FLAMINGO 62)
#   configs/cross_corr/r_profiles_z10.yaml   (TNG300-1 50, FLAMINGO 57)
#
# First checks that every field make_r_profiles.py loads (baryon, total,
# ionized_gas for each run) is on disk, and stops if one is missing: with
# load_field true and save_field false, make_r_profiles would otherwise rebuild
# it from particles without saving it, turning this FFT job into hours.
#
# Then make_r_profiles.py (npz into data/r_profiles/), plot_r_profiles.py
# (baryon and tau figures plus Gate A metrics) and plot_electron_baryon.py, for
# both configs.
#
# Submit from scripts/, after the field build:
#   sbatch --dependency=afterok:<fields job> cross_corr/runCPU_rprofiles_highz.sh

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --job-name=rprofiles_highz
#SBATCH -o ../Outputs_Perlmutter/slurm-%x-%j.out
#SBATCH -e ../Outputs_Perlmutter/slurm-%x-%j.err

cd /global/u2/r/rhliu/projects/SimulationStacker/scripts || exit 1

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

export OMP_NUM_THREADS=128
export NUMBA_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

CONFIGS="configs/cross_corr/r_profiles_z075.yaml configs/cross_corr/r_profiles_z10.yaml"

echo "=== python: $(which python) ==="
echo "=== started: $(date) ==="

echo
echo "########## required cached fields ##########"
python - $CONFIGS <<'EOF' || { echo "missing fields; not running"; exit 1; }
import sys
import yaml
sys.path.append('../src/')
from loadIO import _get_data_filepath

missing = []
for path in sys.argv[1:]:
    cfg = yaml.safe_load(open(path))
    stack = cfg['stack']
    ptypes = [stack.get('particle_type_b', 'baryon'), 'total',
              stack.get('particle_type_e', 'ionized_gas')]
    for suite in cfg['simulations']:
        for e in suite['sims']:
            for proj in stack.get('projections', ['yz']):
                for p in ptypes:
                    f = _get_data_filepath(suite['sim_type'], e['name'], e['snapshot'],
                                           e.get('feedback'), p, e['n_pixels'],
                                           projection=proj, data_type='field', dim='2D')
                    ok = f.exists()
                    print(f"{'ok     ' if ok else 'MISSING'} {f}")
                    if not ok:
                        missing.append(str(f))
sys.exit(1 if missing else 0)
EOF

status=0
for cfg in $CONFIGS; do
    echo
    echo "########## r-profiles: $cfg ##########"
    srun -n 1 -c 256 --cpu-bind=cores python -u cross_corr/make_r_profiles.py -p "$cfg" || status=1
done
for cfg in $CONFIGS; do
    echo
    echo "########## figures: $cfg ##########"
    srun -n 1 -c 256 --cpu-bind=cores python -u cross_corr/plot_r_profiles.py -p "$cfg" || status=1
    srun -n 1 -c 256 --cpu-bind=cores python -u cross_corr/plot_electron_baryon.py -p "$cfg" || status=1
done

echo
echo "=== finished: $(date) ==="
echo "########## EXIT CODE: $status ##########"
exit $status
