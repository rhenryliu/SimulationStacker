#! /bin/bash -l

# Task 1 r-profiles: full sweep over the production simulations.
#
# Submit from scripts/ (all paths below are CWD-relative to it):
#   cd scripts/
#   sbatch cross_corr/runCPU_rprofiles.sh
#
# The heavy step is not the FFT work (minutes per simulation on cached maps)
# but loading the subhalo catalogues -- illustris_python reads every field of
# the TNG300-1 group catalogue -- which is why this runs on a compute node
# rather than a shared login node.

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --job-name=rprofiles
#SBATCH -o ../Outputs_Perlmutter/slurm-%x-%j.out
#SBATCH -e ../Outputs_Perlmutter/slurm-%x-%j.err

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

echo "=== python: $(which python) ==="
echo "=== started: $(date) ==="

# LRG-like sample, z ~ 0.5
python -u cross_corr/make_r_profiles.py -p configs/cross_corr/r_profiles_z05.yaml

# BGS-like sample, z ~ 0.26 (FLAMINGO at z = 0.30)
python -u cross_corr/make_r_profiles.py -p configs/cross_corr/r_profiles_z026.yaml

# Figures and Gate A metrics
python -u cross_corr/plot_r_profiles.py -p configs/cross_corr/r_profiles_z05.yaml
python -u cross_corr/plot_r_profiles.py -p configs/cross_corr/r_profiles_z026.yaml

echo "=== finished: $(date) ==="
