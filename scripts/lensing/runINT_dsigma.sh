#! /bin/bash -l

# Simulated Delta Sigma profiles vs the HSC Y3 x DESI LRG bin1 measurement.
#
# NOT a paper figure -- this is the testing/diagnostic figure that the halo
# selection for the eventual joint lensing+kSZ fit is tuned on.
#
# Fields are all cached at pixel_size 0.2 arcmin, so the cost is dominated by
# reading the FLAMINGO fields (3 x 629 MB) and SOAP subhalo catalogues, plus
# ~158k SHAM cutouts per FLAMINGO variant. Runs comfortably inside an
# interactive allocation; the debug QOS 30 min cap is tight.
#
# Grab a node first (from the scripts/ directory):
#   salloc -q interactive -C cpu -N 1 -t 2:00:00 -A desi
# then:
#   bash lensing/runINT_dsigma.sh

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh dr1
source ~/myenvs/cosmodesi_dr1/bin/activate

srun -n 1 python -u lensing/simulated_dsigma_profiles.py -p configs/lensing/dsigma_profile_z05.yaml

echo "=== Done ==="
