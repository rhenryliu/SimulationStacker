#! /bin/bash -l

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
## SBATCH --ntasks-per-node=1
#SBATCH -o ../Outputs_Perlmutter/make_lensing_plots-%j.out # STDOUT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

# Master figure script for the simulation figures of the lensing+kSZ paper
# (Figs 14-17; the observational figures come from the Weak_lensing project).
# Submit from the scripts/ directory:
#   sbatch lensing/make_lensing_plots.sh

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account m3058

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

echo "make_lensing_plots.sh"

# Figure 14a
srun python -u lensing/compare_data_ratio.py -p configs/lensing/mass_ratio_data_z026.yaml
# Figure 14b
srun python -u lensing/compare_data_ratio.py -p configs/lensing/mass_ratio_data_z05.yaml
# Figure 15
srun python -u lensing/plot_beam_factors.py --config-z05 configs/lensing/mass_ratio_beamTest_z05.yaml --config-z026 configs/lensing/mass_ratio_beamTest_z026.yaml
# Figure 16a (ionized gas only)
srun python -u lensing/beam_compensated_ratio_v2.py -p configs/lensing/beam_compensated_z026.yaml
# Figure 16b (ionized gas only)
srun python -u lensing/beam_compensated_ratio_v2.py -p configs/lensing/beam_compensated_z05.yaml
# Figure 17a (total baryons, data repeated for reference)
srun python -u lensing/beam_compensated_ratio_v2.py -p configs/lensing/beam_compensated_z026_baryon.yaml
# Figure 17b (total baryons, data repeated for reference)
srun python -u lensing/beam_compensated_ratio_v2.py -p configs/lensing/beam_compensated_z05_baryon.yaml
# Appendix: SHAM abundance-target sensitivity (both redshifts)
srun python -u lensing/abundance_variation_ratio.py -p configs/lensing/abundance_variation_z05.yaml
srun python -u lensing/abundance_variation_ratio.py -p configs/lensing/abundance_variation_z026.yaml
# Not a paper figure: testing only -- simulated Delta Sigma vs the HSC Y3 x DESI LRG measurement.
srun python -u lensing/simulated_dsigma_profiles.py -p configs/lensing/dsigma_profile_z05.yaml
