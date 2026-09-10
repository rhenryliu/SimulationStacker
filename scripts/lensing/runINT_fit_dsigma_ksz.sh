#! /bin/bash -l

# Fit the SHAM number density on lensing, then predict kSZ from the same haloes.
#
# NOT a paper figure -- testing/diagnostic, for tuning the halo selection of the
# eventual joint lensing+kSZ comparison.
#
# All 'total' fields (0.2 arcmin/pixel) and 'tau' maps (0.5 arcmin/pixel, yz)
# are cached, so the cost is the SHAM stacking itself. FLAMINGO dominates and
# scales linearly with the number density: the 5-point grid sums to ~15.5x the
# n = 5e-4 cost, so budget ~2-2.5 h in total.
#
# The run checkpoints to the cache npz after every simulation, so if the 4 h
# interactive cap is hit nothing already stacked is lost:
#   python lensing/fit_dsigma_ksz.py -p configs/lensing/fit_dsigma_ksz_z05.yaml --resume
# and re-plotting from the cache is seconds, not hours:
#   python lensing/fit_dsigma_ksz.py -p configs/lensing/fit_dsigma_ksz_z05.yaml --replot
#
# Grab a node first (from the scripts/ directory):
#   salloc -q interactive -C cpu -N 1 -t 4:00:00 -A desi
# then:
#   bash lensing/runINT_fit_dsigma_ksz.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

srun -n 1 python -u lensing/fit_dsigma_ksz.py -p configs/lensing/fit_dsigma_ksz_z05.yaml "$@"

echo "=== Done ==="
