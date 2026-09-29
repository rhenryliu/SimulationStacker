"""check_masked_maps.py

Guard for the masking figures (simulated_kSZ_masked.py, simulated_tSZ_masked.py):
exit non-zero unless, for every simulation of the config, the unmasked map and
the three masked maps are cached, and each masked map's ``.sample.json``
(written by precompute_masked_sham_hosts.py) records masks around the SHAM
hosts at the density this config stacks. A figure script run with a missing
masked map would otherwise rebuild it with the mass-cut sample of
``SimulationStacker.makeField`` and save it at the same path.

Usage (from scripts/):
    python unbound_gas/check_masked_maps.py -p configs/unbound_gas/tau_z05_CAP_masked_flamingo.yaml
"""

import argparse
import json
import sys

import numpy as np
import yaml
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

sys.path.append('../src/')
from stacker import SimulationStacker  # type: ignore
from loadIO import _get_data_filepath  # type: ignore
from utils import comoving_to_arcmin  # type: ignore
# Sibling module in this directory; after the '../src/' append above.
from halo_stats import fit_label, load_fitted_abundances


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('-p', '--path2config', required=True)
    args = ap.parse_args()
    with open(args.path2config) as f:
        config = yaml.safe_load(f)
    stack = config['stack']
    z, ptype, proj = stack.get('redshift', 0.5), stack['particle_type'], stack['projection']
    pix = stack.get('pixel_size', 0.5)
    fit = stack.get('abundance_from_fit')
    fitted = load_fitted_abundances(fit) if fit is not None else None
    problems = []
    for suite in config['simulations']:
        for sim in suite['sims']:
            st = SimulationStacker(sim['name'], sim['snapshot'], z=z, simType=suite['sim_type'],
                                   feedback=sim.get('feedback'))
            cosmo = FlatLambdaCDM(H0=100 * st.header['HubbleParam'], Om0=st.header['Omega0'],
                                  Tcmb0=2.7255 * u.K)
            n = int(np.ceil(comoving_to_arcmin(st.header['BoxSize'], z, cosmo=cosmo) / pix))
            label = fit_label(suite['sim_type'], sim)
            density = fitted[label] if fitted is not None else stack.get('halo_abundance_target', 5e-4)
            for mask, mrad in ((False, None), (True, 1.0), (True, 2.0), (True, 3.0)):
                fp = _get_data_filepath(suite['sim_type'], sim['name'], sim['snapshot'], sim.get('feedback'),
                                        ptype, n, proj, 'map', '2D', mask, mrad if mask else 2.0, st.base_path)
                if not fp.exists():
                    problems.append(f"missing {fp}")
                    continue
                if mask:
                    meta_path = str(fp) + '.sample.json'
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                    except FileNotFoundError:
                        problems.append(f"no sample record {meta_path}")
                        continue
                    if meta.get('sample') != 'sham-hosts' or not np.isclose(meta.get('density') or -1, density):
                        problems.append(f"{fp}: sample {meta.get('sample')} n={meta.get('density')}, "
                                        f"config stacks n={density}")
            print(f"checked {label}", flush=True)
    if problems:
        print('\n'.join(['PROBLEMS:'] + problems))
        sys.exit(1)
    print('all masked maps present and built around the stacked SHAM hosts')


if __name__ == '__main__':
    main()
