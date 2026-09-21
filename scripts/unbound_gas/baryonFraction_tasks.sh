#! /bin/bash -l

# Shared task table for the make_baryonFraction.py (Figure 5) precompute.
#
# Sourced by both runners so the two cannot drift apart:
#   runCPU_baryonFraction_precompute.sh  -- sbatch job array (regular QOS)
#   runINT_baryonFraction_precompute.sh  -- salloc (interactive QOS)
#
# Groups:
#   0-11  A  2D Stars / BH at 0.2 arcmin for all six simulations
#   12-14 B  3D ionized_gas at n_pixels=1000 for the three FLAMINGO variants
#   15-23 C  neutral_gas (2D for all six, 3D for FLAMINGO) -- optional cache-warm,
#            since make_baryonFraction.py derives neutral_gas as gas - ionized_gas
#
# 'none' in FEEDBACKS means the suite has no feedback variant.

#          |------------------------ A: 2D Stars / BH ------------------------|------- B: 3D ionized_gas -------|------------------------------ C: neutral_gas ------------------------------|
SIMTYPES=( IllustrisTNG IllustrisTNG IllustrisTNG IllustrisTNG SIMBA     SIMBA     FLAMINGO FLAMINGO FLAMINGO    FLAMINGO    FLAMINGO        FLAMINGO        FLAMINGO FLAMINGO    FLAMINGO        IllustrisTNG IllustrisTNG SIMBA     FLAMINGO FLAMINGO    FLAMINGO        FLAMINGO FLAMINGO    FLAMINGO )
SIMS=(     TNG300-1     TNG300-1     Illustris-1  Illustris-1  m100n1024 m100n1024 L1_m9    L1_m9    L1_m9       L1_m9       L1_m9           L1_m9           L1_m9    L1_m9       L1_m9           TNG300-1     Illustris-1  m100n1024 L1_m9    L1_m9       L1_m9           L1_m9    L1_m9       L1_m9    )
SNAPS=(    67           67           103          103          125       125       67       67       67          67          67              67              67       67          67              67           103          125       67       67          67              67       67          67       )
FEEDBACKS=( none        none         none         none         s50       s50       L1_m9    L1_m9    fgas-8sigma fgas-8sigma Jet_fgas-4sigma Jet_fgas-4sigma L1_m9    fgas-8sigma Jet_fgas-4sigma none         none         s50       L1_m9    fgas-8sigma Jet_fgas-4sigma L1_m9    fgas-8sigma Jet_fgas-4sigma )
PTYPES=(   Stars        BH           Stars        BH           Stars     BH        Stars    BH       Stars       BH          Stars           BH              ionized_gas ionized_gas ionized_gas  neutral_gas  neutral_gas  neutral_gas neutral_gas neutral_gas neutral_gas  neutral_gas neutral_gas neutral_gas )
DIMS=(     2D           2D           2D           2D           2D        2D        2D       2D       2D          2D          2D              2D              3D       3D          3D              2D           2D           2D        2D       2D          2D              3D       3D          3D       )
