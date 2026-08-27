#! /bin/bash -l
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=task4
#SBATCH -o ../Outputs_Perlmutter/slurm-%x-%j.out
#SBATCH -e ../Outputs_Perlmutter/slurm-%x-%j.err
set -o pipefail
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
echo "=== started: $(date) ==="; status=0
echo; echo "########## Task 4A/B/C: theory transfer, all four sims ##########"
python -u cross_corr/check_theory_transfer.py -p configs/cross_corr/r_profiles_z05.yaml || status=1
echo; echo "########## Task 4: projection-depth study (TNG300-1) ##########"
python -u cross_corr/check_projection_depth.py --sim TNG300-1 --snapshot 67 --grid 1000 || status=1
echo "=== finished: $(date) ==="
[ $status -ne 0 ] && echo "ONE OR MORE STEPS FAILED"
exit $status
