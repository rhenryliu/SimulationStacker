#! /bin/bash -l

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=04:30:00
#SBATCH --nodes=1
#SBATCH -o ../Outputs_Perlmutter/slurm-%j.out # STDOUT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

# Appendix figure: sensitivity of the simulated gas-fraction profiles to the
# SHAM target number density, at both redshifts.
#
# Runtime: the three densities sum to 3.5x a single fiducial pass, and the
# fiducial passes measured in slurm-55734073.out were 647 s (z=0.5) and 994 s
# (z=0.26).  So roughly 38 min + 58 min ~= 1 h 40 m total.  The debug QOS caps
# at 30 min and will not fit -- hence --qos=regular with a 2.5 h wall clock.
#
# Both runs are independent: if the second fails the first has already written
# its figure and its .npz cache.  Re-plotting afterwards is seconds, not hours:
#   python lensing/abundance_variation_ratio.py -p configs/lensing/abundance_variation_z05.yaml --replot
#
# Submit from the scripts/ directory:
#   sbatch lensing/runCPU_abundance.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

echo "=== abundance_variation_ratio.py: z = 0.5 (LRG) ==="
srun python -u lensing/abundance_variation_ratio.py -p configs/lensing/abundance_variation_z05.yaml

echo "=== abundance_variation_ratio.py: z = 0.26 (BGS) ==="
srun python -u lensing/abundance_variation_ratio.py -p configs/lensing/abundance_variation_z026.yaml

echo "=== Done ==="
