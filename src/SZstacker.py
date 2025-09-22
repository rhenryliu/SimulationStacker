# !pip install illustris_python
import sys

import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

# from abacusnbody.analysis.tsc import tsc_parallel
import time
import yaml
import glob
# import json
# import pprint 

# sys.path.append('../../illustrisPython/')
import illustris_python as il 

from tools import numba_tsc_3D, hist2d_numba_seq
from stacker import SimulationStacker
from utils import fft_smoothed_map
from halos import select_massive_halos, halo_ind
from filters import total_mass, delta_sigma, CAP, CAP_from_mass, DSigma_from_mass
from loadIO import snap_path, load_halos, load_subsets, load_subset, load_data, save_data
from mapMaker import create_field

class SZMapStacker(SimulationStacker):
    
    def __init__(self,
                 sim: str, 
                 snapshot: int, 
                 nPixels=2000, 
                 simType='IllustrisTNG', 
                 feedback=None, # Only for SIMBA
                 z=0.0):
        """Initialize the SZMapStacker.

        Args:
            sim (str): The simulation name.
            snapshot (int): The snapshot number.
            SZ (str): The SZ map type. Either 'tSZ', 'kSZ', or 'tau'.
            nPixels (int, optional): The number of pixels in the map. Defaults to 2000.
            simType (str, optional): The type of simulation. Defaults to 'IllustrisTNG'.
            feedback (_type_, optional): Feedback mechanism. Defaults to None.
        """
        
        super().__init__(sim, snapshot, nPixels, simType=simType, feedback=feedback, z=z)


    def makeMap(self, pType, z=None, projection='xy', beamsize=1.6, save=False, load=True, pixelSize=0.5):
        """Create a map from the simulation data.

        Args:
            pType (str): The type of particle to use for the map. Either 'tSZ', 'kSZ', or 'tau'.
                Note that in the case of 'kSZ', an optical depth (tau) map will be created instead of a velocity map.
            z (float, optional): The redshift to use for the map. Defaults to None.
            projection (str, optional): The projection to use for the map. Defaults to 'xy'.
            beamsize (float, optional): The size of the beam to use for the map. Defaults to 1.6.
            save (bool, optional): Whether to save the map to disk. Defaults to False.
            load (bool, optional): Whether to load the map from disk. Defaults to True.
            pixelSize (float, optional): The size of the pixels in the map. Defaults to 0.5.
        """
        
        if z is None:
            z = self.z
        
        # First define cosmology
        cosmo = FlatLambdaCDM(H0=100 * self.header['HubbleParam'], Om0=self.header['Omega0'], Tcmb0=2.7255 * u.K)

        # Get distance to the snapshot redshift
        dA = cosmo.angular_diameter_distance(z).to(u.kpc).value
        dA *= self.header['HubbleParam']  # Convert to kpc/h
        
        # Get the box size in angular units.
        theta_arcmin = np.degrees(self.header['BoxSize'] / dA) * 60  # Convert to arcminutes
        print(f"Map size at z={z}: {theta_arcmin:.2f} arcmin")

        # Round up to the nearest integer, pixel size is 0.5 arcmin as in ACT
        nPixels = np.ceil(theta_arcmin / pixelSize).astype(int)
        arcminPerPixel = theta_arcmin / nPixels  # Arcminutes per pixel, this is the true pixelSize after rounding.
        # beamsize_pixel = beamsize / arcminPerPixel  # Convert arcminutes to pixels
        
        

        # Now that we know the expected pixel size, we try to load the map first before computing it:
        if load:
            try:
                return self.loadData(pType, nPixels=nPixels, projection=projection, type='map')
            except ValueError as e:
                print(e)
                print("Computing the map instead...")    

        map_ = self.makeField(pType, nPixels=nPixels, projection=projection, save=False, load=load)

        # Convolve the map with a Gaussian beam (only if beamsize is not None)
        if beamsize is not None:
            map_ = fft_smoothed_map(map_, beamsize, pixel_size_arcmin=arcminPerPixel)

        if save:
            if self.simType == 'IllustrisTNG':
                saveName = self.sim + '_' + str(self.snapshot) + '_' + \
                    pType + '_' + str(nPixels) + '_' + projection + '_map'
                np.save(f'/pscratch/sd/r/rhliu/simulations/{self.simType}/products/2D/{saveName}.npy', map_)
            elif self.simType == 'SIMBA':
                saveName = self.sim + '_' + self.feedback + '_' + str(self.snapshot) + '_' + \
                    pType + '_' + str(nPixels) + '_' + projection + '_map'
                np.save(f'/pscratch/sd/r/rhliu/simulations/{self.simType}/products/2D/{saveName}.npy', map_)

        return map_

    
    def makeField(self, pType, nPixels=None, projection='xy', save=False, load=True):
        if nPixels is None:
            nPixels = self.nPixels
        return create_field(self, pType, nPixels, projection, save, load)

    def stackMap(self, pType, filterType='cumulative', minRadius=0.5, maxRadius=6.0, numRadii=11,
                 z=None, projection='xy', save=False, load=True, radDistance=1.0, pixelSize=0.5, 
                 halo_mass_avg=10**(13.22), halo_mass_upper=5*10**(14)):
        """Stack the map of a given particle type.

        Args:
            pType (str): Particle type to stack.
            filterType (str, optional): Type of filter to apply. Defaults to 'cumulative'.
            minRadius (float, optional): Minimum radius for stacking. Defaults to 0.2.
            maxRadius (float, optional): Maximum radius for stacking. Defaults to 6.0.
            numRadii (int, optional): Number of radial bins for stacking. Defaults to 11.
            z (float, optional): Redshift of the snapshot. Defaults to None, in which case self.z is used.
            projection (str, optional): Direction of the field projection. Defaults to 'xy'. Options are ['xy', 'yz', 'xz']
            save (bool, optional): If True, saves the stacked map to a file. Defaults to True.
            load (bool, optional): If True, loads the stacked map from a file if it exists. Defaults to True.
            radDistance (float, optional): Radial distance units for stacking. Defaults to 1 arcmin.
                Note there is no None option here as in stackField.
            pixelSize (float, optional): Size of each pixel in arcminutes. Defaults to 0.5.
            halo_mass_avg (float, optional): Average halo mass for selecting halos. Defaults to 10**(13.22).
            halo_mass_upper (float, optional): Upper mass bound for selecting halos. Defaults to None.

        Returns:
            radii, profiles: Stacked radial profiles (2D) and their corresponding radii (1D).
            
        TODO:
            Add a wrapper for automatic stacking along all 3 projections.
            Implement the DSigma filter for stacking.
        """

        if z is None:
            z = self.z
        
        # Load or create the map
        fieldKey = (pType, z, projection, pixelSize)
        if not (fieldKey in self.maps and self.maps[fieldKey] is not None):
            self.maps[fieldKey] = self.makeMap(pType, z=z, projection=projection,
                                               save=save, load=load, pixelSize=pixelSize)

        # Use the abstracted stacking function from parent class
        radii, profiles = self.stack_on_array(
            array=self.maps[fieldKey],
            filterType=filterType,
            minRadius=minRadius,
            maxRadius=maxRadius,
            numRadii=numRadii,
            projection=projection,
            radDistance=radDistance,
            radDistanceUnits='arcmin',
            halo_mass_avg=halo_mass_avg,
            halo_mass_upper=halo_mass_upper,
            z=z,
            pixelSize=pixelSize
        )
        
        # Unit Conversion specific to SZ maps:
        T_CMB = 2.7255
        if pType == 'tau':
            # In the case of the tau field, we want to do unit conversion from optical depth units to micro-Kelvin.
            # This is done by multiplying the tau field by T_CMB * (v/c)
            v_c = 300000 / 299792458 # velocity over speed of light.
            profiles = profiles * T_CMB * 1e6 * v_c # Convert to micro-Kelvin, the units for kSZ in data.
        elif pType == 'kSZ':
            # TODO: kSZ unit conversion
            pass
        elif pType == 'tSZ':
            # TODO: tSZ unit conversion
            pass
        
        return radii, profiles

