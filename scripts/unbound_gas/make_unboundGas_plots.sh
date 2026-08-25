#! /bin/bash -l

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
## SBATCH --ntasks-per-node=1
#SBATCH -o ../Outputs_Perlmutter/slurm-%j.out # STDOUT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

# Master figure script for the unbound gas paper. Figure numbers follow the
# current draft. Assumes all fields/maps are already cached on scratch (the
# FLAMINGO precompute is runCPU_flamingo_masked.sh); even so, the stacking in
# Figures 7 and 11 dominates the walltime (up to ~4 h together, see
# runCPU_flamingo_figures.sh), hence regular QOS rather than debug.
# Submit from the scripts/ directory:
#   sbatch unbound_gas/make_unboundGas_plots.sh

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account m3058

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

echo "make_unboundGas_plots.sh"

# Figure 1: stellar fractions
srun python -u unbound_gas/star_fraction.py -p configs/unbound_gas/star_fraction.yaml
# Figure 2: ionized gas fraction ratios (3x2)
srun python -u unbound_gas/make_ratios3x2.py -p configs/unbound_gas/ratios_3x2_z05.yaml --ptype ionized_gas
# Figure 3: baryon fraction ratios (3x2)
srun python -u unbound_gas/make_ratios3x2.py -p configs/unbound_gas/ratios_3x2_z05.yaml --ptype baryon
# Figure 4: cumulative baryon-component stacked areas
srun python -u unbound_gas/make_stackArea.py -p configs/unbound_gas/stackArea_z05.yaml
# Figure 5: non-cumulative baryon-component fractions
srun python -u unbound_gas/make_baryonFraction.py -p configs/unbound_gas/stackArea_z05.yaml
# Figure 6: 2D CAP f_gas profiles, SHAM vs mass-cut halo selection
srun python -u unbound_gas/make_fgas_profiles.py -p configs/unbound_gas/fgas_profiles_z05.yaml
# Figure 7: kSZ CAP profiles with progressive halo masking
srun python -u unbound_gas/simulated_kSZ_masked.py -p configs/unbound_gas/tau_z05_CAP_masked_flamingo.yaml
# Figure 8: masked-box gas distribution visualization
srun python -u unbound_gas/mask_haloes_visual.py -p configs/unbound_gas/mask_visual_z05.yaml
# Figure 9: gas fraction profiles in halo mass bins
srun python -u unbound_gas/make_ratios_mass_bins.py -p configs/unbound_gas/ratios_mass_bins_z05.yaml --ptype ionized_gas
# Figure 10: baryon fraction profiles in halo mass bins
srun python -u unbound_gas/make_ratios_mass_bins.py -p configs/unbound_gas/ratios_mass_bins_z05.yaml --ptype baryon
# Figure 11: tSZ CAP profiles with progressive halo masking
srun python -u unbound_gas/simulated_tSZ_masked.py -p configs/unbound_gas/tSZ_z05_CAP_masked.yaml
# tSZ CAP profiles, unmasked (companion to Figure 11)
srun python -u unbound_gas/simulated_tSZ_maps.py -p configs/unbound_gas/tSZ_z05_CAP.yaml
