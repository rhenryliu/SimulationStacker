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

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account desi

# source /global/common/software/desi/desi_environment.sh 23.1 # inherits it
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
# module unload desiutil
# module load desiutil/3.2.6


# echo "mass_profiles_sims.py"
# srun python -u mask_sims_3D.py -p ./configs/3D_masked_profiles_z00.yaml
# srun python -u mask_sims_3D.py -p ./configs/3D_masked_profiles_z05.yaml
# srun python -u make_3D_profiles2x2.py -p ./configs/3D_mass_ratio_z05.yaml
# srun python -u make_2D_ratios2x2.py -p ./configs/mass_ratio_z05.yaml
# srun python -u make_2D_ratios2x2.py -p ./configs/mass_ratio_z00_field.yaml
# srun python -u mass_profiles_sims.py
# srun python star_fraction.py -p ./configs/star_fraction.yaml
# srun python -u simulated_kSZ_masked.py -p ./configs/tau_z05_CAP_masked.yaml
# srun python -u simulated_kSZ.py -p ./configs/tau_z05_CAP.yaml
# srun python -u simulated_tSZ_masked.py -p ./configs/tSZ_z05_CAP_masked.yaml
# srun python -u simulated_tSZ.py -p ./configs/tSZ_z05_CAP.yaml
# srun python -u simulated_mass_ratio.py -p ./configs/mass_ratio_z05.yaml
# srun python -u simulated_mass_ratio.py -p ./configs/mass_ratio_data_z05.yaml
# srun python -u make_2D_profiles2x2.py -p ./configs/tau_profiles_z05.yaml
# srun python -u simulated_SZ_ratio.py -p ./configs/kSZ_ratio_z05_2.yaml
# srun python -u simulated_dsigma_profiles.py -p ./configs/dsigma_profile_z05.yaml
# srun python -u simulated_tSZ_maps_beamTest.py
# srun python -u make_ratios3x2.py -p ./configs/ratios_3x2_z026.yaml --ptype ionized_gas
# srun python -u make_ratios3x2.py -p ./configs/ratios_3x2_z026.yaml --ptype baryon
srun python -u unbound_gas/make_ratios_mass_bins.py -p configs/unbound_gas/ratios_mass_bins_z05.yaml --ptype baryon


# srun python -u mask_haloes_visual.py