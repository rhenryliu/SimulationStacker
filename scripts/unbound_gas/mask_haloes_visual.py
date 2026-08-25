"""mask_haloes_visual.py
========================
Figure 8 of the unbound gas paper: a 2x2 visualization of the projected gas
field of a simulation box, unmasked and after masking spheres of 1, 2 and 3
R_200m around every halo above a mass threshold.  The residual filamentary
web in the masked panels illustrates the diffuse gas that halo-centric
models miss.

For each simulation in the config this produces one figure with panels:

    [No Masking]  [R_mask = 1 R_200m]
    [R_mask = 2 R_200m]  [R_mask = 3 R_200m]

The 3D field is loaded from the scratch product cache when available (no
particle reads); otherwise it is computed once and saved. All four panels
project the same 3D field; the masked panels zero the spheres around the
selected haloes before projecting (note this is the complement of the
``create_masked_field`` convention, which keeps the spheres).

Usage
-----
    python unbound_gas/mask_haloes_visual.py -p configs/unbound_gas/mask_visual_z05.yaml

Run from the ``scripts/`` directory (paths in the config are relative to it).
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yaml

sys.path.append('../src/')
from stacker import SimulationStacker
from loadIO import load_data
from mask_utils import get_cutout_mask_3d

PROJECTION_AXIS = {'xy': 2, 'xz': 1, 'yz': 0}


def mass_label(mass_msun):
    """Return a LaTeX label like ``$10^{11}$`` for a mass threshold in M_sun.

    Args:
        mass_msun (float): Mass threshold in M_sun.

    Returns:
        str: LaTeX-formatted mass string (without the M_sun suffix).
    """
    exponent = np.log10(mass_msun)
    if np.isclose(exponent, round(exponent)):
        return rf'$10^{{{int(round(exponent))}}}$'
    return rf'${mass_msun:.2g}$'


def load_field_3D(stacker, pType, nPixels, projection):
    """Load the cached 3D field for the stacker, computing it if missing.

    Args:
        stacker (SimulationStacker): Configured stacker instance.
        pType (str): Particle type of the field.
        nPixels (int): Grid size per side.
        projection (str): Projection direction ('xy', 'xz' or 'yz').

    Returns:
        np.ndarray: The 3D field, shape (nPixels, nPixels, nPixels).
    """
    try:
        return load_data(stacker.simType, stacker.sim, stacker.snapshot,
                         stacker.feedback, pType, nPixels, projection,
                         data_type='field', dim='3D',
                         base_path=stacker.base_path)
    except ValueError as e:
        print(e)
        print('Computing the 3D field instead (this reads particle data)...')
        return stacker.makeField(pType, nPixels=nPixels, projection=projection,
                                 save=True, load=False, dim='3D')


def make_figure(stacker, stack_cfg, fig_path):
    """Render the 4-panel masking visualization for one simulation.

    Args:
        stacker (SimulationStacker): Configured stacker instance.
        stack_cfg (dict): The ``stack`` section of the config.
        fig_path (Path): Directory the figure is written to.
    """
    pType = stack_cfg.get('particle_type', 'gas')
    nPixels = int(stack_cfg.get('n_pixels', 1000))
    projection = stack_cfg.get('projection', 'yz')
    mask_radii = [float(r) for r in stack_cfg.get('mask_radii', [1, 2, 3])]
    if len(mask_radii) != 3:
        raise ValueError('The 2x2 panel layout expects exactly 3 mask_radii; '
                         f'got {len(mask_radii)}')
    mass_min_msun = float(stack_cfg.get('mass_min', 1.0e11))
    figName = stack_cfg.get('fig_name', 'mask_visual')

    h = stacker.header['HubbleParam']

    # Halo catalogue: keep every halo above the mass threshold. GroupMass is
    # in M_sun/h, the config threshold in M_sun, hence the factor h.
    haloes = stacker.loadHalos()
    sel = haloes['GroupMass'] >= mass_min_msun * h
    print(f"{stacker.sim}: masking {sel.sum()} of {sel.size} haloes "
          f">= {mass_min_msun:.3g} M_sun")

    # All four panels project the same 3D field, loaded once.
    field_3D = load_field_3D(stacker, pType, nPixels, projection)
    axis = PROJECTION_AXIS[projection]
    field_2D = np.sum(field_3D, axis=axis, dtype=np.float64)

    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    ax = ax.flatten()
    positive = field_2D[field_2D > 0]
    norm = LogNorm(vmin=positive.min(), vmax=positive.max())

    # Clip at vmin for display: LogNorm treats zeros (fully masked pixels)
    # as "bad" and would render them transparent instead of dark.
    ax[0].imshow(np.maximum(field_2D, norm.vmin), norm=norm)
    ax[0].set_title('No Masking')

    # Pixel conversion as in mapMaker.create_masked_field.
    kpcPerPixel = stacker.header['BoxSize'] / nPixels
    pos_pix = np.round(haloes['GroupPos'][sel] / kpcPerPixel).astype(int)
    rad_pix = haloes['GroupRad'][sel] / kpcPerPixel

    for i, rad in enumerate(mask_radii, start=1):
        # get_cutout_mask_3d is True inside the halo spheres; the repo's
        # create_masked_field KEEPS those (halo-painting mock for the masked
        # SZ profiles). This figure shows the opposite: haloes removed, the
        # diffuse web kept — hence the complement.
        cutout = get_cutout_mask_3d(field_3D, pos_pix, rad_pix * rad)
        masked_2D = np.sum(field_3D * ~cutout, axis=axis, dtype=np.float64)
        del cutout
        ax[i].imshow(np.maximum(masked_2D, norm.vmin), norm=norm)
        ax[i].set_title(rf'$R_{{\rm mask}} = {rad:g}\, R_{{200{{\rm m}}}}$')

    fig.suptitle(f'{stacker.sim} Simulation Box, pType {pType}, '
                 f'snapshot {stacker.snapshot}\n'
                 rf'Masking Haloes $\geq$ {mass_label(mass_min_msun)} $M_\odot$',
                 fontsize=15)
    fig.tight_layout()

    stem = f'{figName}_{pType}_{stacker.sim}'
    fig.savefig(fig_path / f'{stem}.pdf', dpi=300)
    fig.savefig(fig_path / f'{stem}.png', dpi=200)
    plt.close(fig)
    print(f'Saved {fig_path / stem}.pdf')


def main():
    """Parse the config and render one figure per configured simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--path2config', type=str,
                        default='./configs/unbound_gas/mask_visual_z05.yaml',
                        help='Path to the configuration file.')
    args = parser.parse_args()

    with open(args.path2config) as f:
        config = yaml.safe_load(f)

    stack_cfg = config['stack']
    plot_cfg = config.get('plot', {})
    stack_cfg.setdefault('fig_name', plot_cfg.get('fig_name', 'mask_visual'))
    redshift = float(stack_cfg.get('redshift', 0.5))

    now = datetime.now()
    fig_path = (Path(plot_cfg.get('fig_path', '../figures/'))
                / now.strftime('%Y-%m') / now.strftime('%m-%d'))
    fig_path.mkdir(parents=True, exist_ok=True)

    for suite in config['simulations']:
        sim_type = suite['sim_type']
        for sim in suite['sims']:
            stacker = SimulationStacker(sim['name'], sim['snapshot'],
                                        simType=sim_type,
                                        feedback=sim.get('feedback'),
                                        z=redshift)
            make_figure(stacker, stack_cfg, fig_path)

    print('Done!!')


if __name__ == '__main__':
    main()
