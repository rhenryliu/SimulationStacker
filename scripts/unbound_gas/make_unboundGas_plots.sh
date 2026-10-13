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

# Master figure script for the unbound gas paper: one command per figure of the
# draft (~/overleaf/Unbound_Gas, Figs. 1-10, with Figs. 5-7 and 9 on the SHAM
# samples at the fixed densities of Liu et al. 2026, 2026-10-12). Every command
# writes to ../figures/YYYY-MM/MM-DD/; the copies in the paper's figures/ are
# made by hand, some under another name (noted below as "paper:").
#
# Assumes every field, map and stack is cached on scratch; the prerequisite
# runners are named per figure. Figs. 1, 2, 6-10 stack (Figs. 1-2 and 8
# about 2 h each, Figs. 7 and 10 about 1 h, Figs. 6 and 9 under 30 min each), hence
# regular QOS; the others only plot. Each stacking script
# redraws from an earlier run's profiles with --from-npz <..._profiles.npz>
# (the file of the same config, written next to the figure).
# Submit from the scripts/ directory:
#   sbatch unbound_gas/make_unboundGas_plots.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

echo "make_unboundGas_plots.sh"

# Figure 1: ionized gas fraction profiles, 3D / CAP / DSigma (mass cut).
# Fields: runCPU_baryonFraction_precompute.sh, runCPU_stackArea_precompute.sh,
# runCPU_mstar_products.sh, runCPU_ratios_unconvolved.sh.
srun python -u unbound_gas/make_ratios3x2.py -p configs/unbound_gas/ratios_3x2_z05.yaml --ptype ionized_gas
# Figure 2: baryon fraction profiles (same fields).
srun python -u unbound_gas/make_ratios3x2.py -p configs/unbound_gas/ratios_3x2_z05.yaml --ptype baryon
# Figure 3: stellar fractions (plotting only, from the merged yaml of the
# config's load_path: runCPU_star_fraction_particles.sh, runCPU_fstar_iter3.sh).
# paper: star_fraction_particles_z0.5.png -> star_fraction_z0.5.png
srun python -u unbound_gas/star_fraction_v2.py -p configs/unbound_gas/star_fraction_particles.yaml
# Figure 4: baryon component budget (plotting only, from the two npz files
# named in the config, written by make_baryonFraction.py with
# baryonFraction_z05.yaml and make_stackArea.py with stackArea_dsigma_z05.yaml;
# rerunning those writes new dated files, and the config's paths must follow).
srun python -u unbound_gas/make_component_budget.py -p configs/unbound_gas/component_budget_z05.yaml
# Figure 5: S(k) and the lensing f_gas under the stellar rescaling, SHAM at
# 5e-4 (plotting only, and its table). Needs the 'lens' stacks
# (stack_stellar_maps.py, stack_fstar_obs_maps.py; runINT_figs_n5e-4.sh for
# the M*-sigma runs) and the cached spectra and transfer maps
# (runINT_pk_components.sh, runINT_pk_stellar.sh, runINT_stellar_maps.sh,
# runINT_fstar.sh, runINT_fstar_obs.sh, runCPU_pk_mstar.sh).
# paper: the layout-D grid, pk_fstar_obs_z05_n5e-4_fstar_grid_D.pdf ->
# pk_fstar_obs_z05_fstar_S_ap1_M13.pdf (the paper's name happens to match the
# single-panel file make_pk_fstar_obs.py writes, which is not the paper figure).
srun python -u unbound_gas/make_pk_fstar_obs_grid.py -p configs/unbound_gas/pk_fstar_obs_z05_n5e-4.yaml --layout D
srun python -u unbound_gas/make_pk_fstar_obs.py -p configs/unbound_gas/pk_fstar_obs_z05_n5e-4.yaml
# Figure 6: masked DeltaSigma ratios, SHAM at 5e-4, z = 0.5. Masked fields:
# runINT_masked_ionized_gas_particles_n5e-4.sh.
# paper: dsigma_ratio_yz_masked_nobeam_n5e-4_z0.5_masking_comparison.pdf -> dsigma_ratio_masked_z05.pdf
srun python -u unbound_gas/simulated_dsigma_ratio_masked.py -p configs/unbound_gas/dsigma_ratio_z05_masked_nobeam_n5e-4.yaml --workers 12
# Figure 7: f_gas profiles, mass cut vs SHAM at 5e-4 (cached 0.2' fields).
# paper: fgas_profiles_z05_n5e-4_2D.pdf -> fgas_profiles_z05_2D.pdf
srun python -u unbound_gas/make_fgas_profiles.py -p configs/unbound_gas/fgas_profiles_z05_n5e-4.yaml
# Figure 8: ionized gas fraction profiles at z ~ 0.26 (mass cut). Fields:
# runCPU_z026_products.sh.
srun python -u unbound_gas/make_ratios3x2.py -p configs/unbound_gas/ratios_z026_bgs.yaml --ptype ionized_gas
# Figure 9: masked DeltaSigma ratios, SHAM at 1e-3, z ~ 0.26. Masked fields:
# runINT_masked_ionized_gas_particles_z026_n1e-3.sh.
# paper: dsigma_ratio_yz_masked_nobeam_n1e-3_z0.26_masking_comparison.pdf -> dsigma_ratio_masked_z026.pdf
srun python -u unbound_gas/simulated_dsigma_ratio_masked.py -p configs/unbound_gas/dsigma_ratio_z026_masked_nobeam_n1e-3.yaml --workers 8
# Figure 10: masked tSZ profiles, SHAM at the lensing-fit densities. Masked
# maps: runCPU_masked_sham_hosts.sh, runCPU_masked_sham_hosts_flamingo.sh,
# runCPU_masked_sham_hosts_mstar.sh. The guard stops the run if one is
# missing or built around another sample (simulated_tSZ_masked.py would
# rebuild it around the mass-cut sample, at the same path).
srun python -u unbound_gas/check_masked_maps.py -p configs/unbound_gas/tSZ_z05_CAP_masked.yaml &&
  srun python -u unbound_gas/simulated_tSZ_masked.py -p configs/unbound_gas/tSZ_z05_CAP_masked.yaml
