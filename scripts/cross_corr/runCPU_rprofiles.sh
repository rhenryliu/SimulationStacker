#! /bin/bash -l

# Tasks 1-3 for the four retained simulations (TNG300-1 + three FLAMINGO
# variants), yz projection, apertures extended to 9.75 arcmin.
#
# Submit from scripts/ (all paths below are CWD-relative to it):
#   cd scripts/
#   sbatch cross_corr/runCPU_rprofiles.sh
#
# The heavy step is not the FFT work but loading the subhalo catalogues --
# illustris_python reads every field of the TNG300-1 group catalogue -- which
# is why this runs on a compute node rather than a shared login node.

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=rprofiles
#SBATCH -o ../Outputs_Perlmutter/slurm-%x-%j.out
#SBATCH -e ../Outputs_Perlmutter/slurm-%x-%j.err

set -o pipefail

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

echo "=== python: $(which python) ==="
echo "=== started: $(date) ==="
status=0

echo
echo "############ Task 1: r-profiles, z ~ 0.5 ############"
python -u cross_corr/make_r_profiles.py \
    -p configs/cross_corr/r_profiles_z05.yaml || status=1

echo
echo "############ Task 1: r-profiles, z ~ 0.26 / 0.30 ############"
python -u cross_corr/make_r_profiles.py \
    -p configs/cross_corr/r_profiles_z026.yaml || status=1

echo
echo "############ Task 1: figures and Gate A metrics ############"
python -u cross_corr/plot_r_profiles.py \
    -p configs/cross_corr/r_profiles_z05.yaml || status=1
python -u cross_corr/plot_r_profiles.py \
    -p configs/cross_corr/r_profiles_z026.yaml || status=1

echo
echo "############ Task 2: filter compensation / box-size sensitivity ############"
# The argument is a property of the kernels and the box geometry, so one
# redshift suffices to establish it.
python -u cross_corr/check_filter_compensation.py \
    -p configs/cross_corr/r_profiles_z05.yaml || status=1

echo
echo "############ Task 3: electron versus baryon ############"
python -u cross_corr/plot_electron_baryon.py \
    -p configs/cross_corr/r_profiles_z05.yaml || status=1
python -u cross_corr/plot_electron_baryon.py \
    -p configs/cross_corr/r_profiles_z026.yaml || status=1

echo
echo "=== finished: $(date) ==="
if [ $status -ne 0 ]; then
    echo "ONE OR MORE STEPS FAILED"
fi
exit $status
