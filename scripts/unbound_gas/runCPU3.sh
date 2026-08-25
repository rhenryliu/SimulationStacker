#! /bin/bash -l

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
## SBATCH --ntasks-per-node=1
#SBATCH -o ../Outputs_Perlmutter/runCPU3-%j.out # STDOUT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account m3058

# source /global/common/software/desi/desi_environment.sh 23.1 # inherits it
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
# module unload desiutil
# module load desiutil/3.2.6


echo "simulated_maps.py"
# srun python -u make_2D_profiles2x2.py -p ./configs/ionized_gas_profiles_z05.yaml
# srun python -u simulated_mass_ratio.py -p ./configs/ionized_gas_profiles_z05.yaml
# srun python -u make_3D_profiles2x2.py
# srun python -u simulated_tSZ_masked.py -p ./configs/tSZ_z05_CAP_masked.yaml
# srun python -u simulated_kSZ_masked.py -p ./configs/tau_z05_CAP_masked.yaml
# srun python -u compare_data_ratio.py -p ./configs/mass_ratio_data_z026.yaml
# srun python -u compare_data_ratio.py -p ./configs/mass_ratio_beamTest_z05.yaml
# srun python -u compare_data_ratio.py -p ./configs/mass_ratio_beamTest_z026.yaml
srun python -u unbound_gas/star_fraction_v2.py -p configs/unbound_gas/star_fraction_flamingo.yaml