#! /bin/bash -l

#SBATCH -A m3058
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
## SBATCH --ntasks-per-node=1
#SBATCH -o ../Outputs_Perlmutter/slurm-%j.out # STDOUT
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r.henryliu@berkeley.edu

# source /global/common/software/desi/desi_environment.sh 23.1 # inherits it
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# module unload desiutil
# module load desiutil/3.2.6


echo "simulated_maps.py"
# srun python -u simulated_SZ_maps.py -p ./configs/tau_z05_CAP.yaml
srun python -u simulated_SZ_masked.py -p ./configs/tau_z05_CAP_masked.yaml
# srun python -u simulated_SZ_maps.py -p ./configs/tSZ_z05_CAP.yaml
# srun python -u simulated_fields.py -p ./configs/field_z05.yaml
# srun python -u simulated_fields.py -p ./configs/field_z00.yaml