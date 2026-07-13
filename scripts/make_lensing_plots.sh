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

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account m3058

# source /global/common/software/desi/desi_environment.sh 23.1 # inherits it
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate
# module unload desiutil
# module load desiutil/3.2.6


echo "simulated_maps.py"

# Figure 14a
srun python -u compare_data_ratio.py -p ./configs/mass_ratio_data_z026.yaml
# Figure 14b
srun python -u compare_data_ratio.py -p ./configs/mass_ratio_data_z05.yaml
# Figure 15
srun python -u plot_beam_factors.py
# Figure 16a (ionized gas only)
srun python -u beam_compensated_ratio_v2.py -p ./configs/beam_compensated_z026.yaml
# Figure 16b (ionized gas only)
srun python -u beam_compensated_ratio_v2.py -p ./configs/beam_compensated_z05.yaml
# Figure 17a (total baryons, data repeated for reference)
srun python -u beam_compensated_ratio_v2.py -p ./configs/beam_compensated_z026_baryon.yaml
# Figure 17b (total baryons, data repeated for reference)
srun python -u beam_compensated_ratio_v2.py -p ./configs/beam_compensated_z05_baryon.yaml