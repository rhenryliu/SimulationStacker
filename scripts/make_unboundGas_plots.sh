#! /bin/bash -l

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
## SBATCH --ntasks-per-node=1
#SBATCH -o ../Outputs_Perlmutter/slurm-%j.out # STDOUT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account m3058

# source /global/common/software/desi/desi_environment.sh 23.1 # inherits it
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
# module unload desiutil
# module load desiutil/3.2.6


echo "simulated_maps.py"

# srun python -u compare_data_ratio.py -p ./configs/mass_ratio_beamTest_z05.yaml
# Figure 1
srun python -u star_fraction.py -p ./configs/star_fraction.yaml
# Figure 2
srun python -u make_ratios3x2.py -p ./configs/ratios_3x2_z05.yaml --ptype ionized_gas
# Figure 3
srun python -u make_ratios3x2.py -p ./configs/ratios_3x2_z05.yaml --ptype baryon
# Figure 4
srun python -u make_stackArea.py -p ./configs/stackArea_z05.yaml
# Figure 5
srun python -u make_baryonFraction.py -p ./configs/stackArea_z05.yaml
# Figure 6
srun python -u make_fgas_profiles.py -p ./configs/fgas_profiles_z05.yaml
# Figure 7
srun python -u simulated_kSZ_masked.py -p ./configs/tau_z05_CAP_masked.yaml
# Figure 8
# TODO: find figure 8 script and add it here
# Figure 9
srun python -u simulated_tSZ_masked.py -p ./configs/tSZ_z05_CAP_masked.yaml
# Figure 10 
srun python -u simulated_tSZ_maps.py -p ./configs/tSZ_z05_CAP.yaml
# Figure 11
srun python -u make_ratios_mass_bins.py -p ./configs/ratios_mass_bins_z05.yaml --ptype ionized_gas
# Figure 12
srun python -u make_ratios_mass_bins.py -p ./configs/ratios_mass_bins_z05.yaml --ptype baryon