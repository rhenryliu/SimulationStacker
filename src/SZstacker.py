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
        """Create a map from the simulation data. 
        Much of the algo for this method comes from the SZ_TNG repo on Github authored by Boryana Hadzhiyska.

        Args:
            pType (str): The type of particle to use for the map. Either 'tSZ', 'kSZ', or 'tau'.
                Note that in the case of 'kSZ', an optical depth (tau) map will be created instead of a velocity map.
            z (float, optional): The redshift to use for the map. Defaults to None.
            nPixels: Size of the output map in pixels. Defaults to self.nPixels.
            save (bool, optional): Whether to save the map to disk. Defaults to False.
            load (bool, optional): Whether to load the map from disk. Defaults to True.
            pixelSize (float, optional): The size of the pixels in the map. Defaults to 0.5.
        """
        if nPixels is None:
            nPixels = self.nPixels
            
        if load:
            try:
                return self.loadData(pType, nPixels=nPixels, projection=projection, type='field')
            except ValueError as e:
                print(e)
                print("Computing the field instead...")

        # If loading fails, we then compute the necessary fields. Much of the code below is the same as the SZ_TNG repo.            
        # Define some necessary parameters: (Physical units in cgs)
        gamma = 5/3. # unitless. Adiabatic Index
        k_B = 1.3807e-16 # cgs (erg/K), Boltzmann constant
        m_p = 1.6726e-24 # g, mass of proton
        unit_c = 1.e10 # TNG faq is wrong (see README.md)
        # unit_c = 1.023**2*1.e10
        X_H = 0.76 # unitless, primordial hydrogen fraction
        sigma_T = 6.6524587158e-29*1.e2**2 # cm^2, thomson cross section
        m_e = 9.10938356e-28 # g, electron mass
        c = 29979245800. # cm/s, speed of light
        const = k_B*sigma_T/(m_e*c**2) # cgs (cm^2/K), constant for compton y computation
        kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
        solar_mass = 1.989e33 # g


        # sim params (same for MTNG and TNG)
        h = self.header['HubbleParam'] # Hubble Parameter
        unit_mass = 1.e10*(solar_mass/h)
        # unit_mass = 1.0 # msun/h, units already converted.
        unit_dens = 1.e10*(solar_mass/h)/(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
        # unit_dens = 1./(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
        unit_vol = (kpc_to_cm/h)**3 # cancels unit dens division from previous, so h doesn't matter neither does kpc


        # z = zs[snaps == snapshot] # TODO: Implement the automatic selection of snapshot from redshift (or vice versa)
        z = self.z
        a = 1./(1+self.z) # scale factor
        Lbox_hkpc = self.header['BoxSize'] # kpc/h
        
        
        # Now we compute the SZ fields iteratively, iterating over every snapshot chunk: (Note SIMBA only has one chunk)
        
        # Get chunks:        
        folderPath = self.snapPath(self.simType, pathOnly=True)
        if self.simType == 'IllustrisTNG':
            snaps = glob.glob(folderPath + 'snap_*.hdf5') # TODO: fix this for SIMBA (done I think)
        elif self.simType == 'SIMBA':
            # print('Folder Path:', folderPath + f'*_{self.snapshot}.hdf5')
            snaps = glob.glob(folderPath)
            # print('Snaps:', snaps)

        # Convert coordinates to pixel coordinates        
        gridSize = [nPixels, nPixels]
        minMax = [0, self.header['BoxSize']]
        field_total = np.zeros(gridSize)
        
        t0 = time.time()
        for i, snap in enumerate(snaps):

            particles = self.loadSubset(pType, snapPath=snap, keys=['Coordinates', 'Masses', 'ElectronAbundance', 'InternalEnergy', 'Density', 'Velocities'])

            Co = particles['Coordinates']
            EA = particles['ElectronAbundance']
            IE = particles['InternalEnergy']
            D = particles['Density']
            M = particles['Masses']
            V = particles['Velocities']

            
            # for each cell, compute its total volume (gas mass by gas density) and convert density units
            dV = M/D # ckpc/h^3 
            D *= unit_dens # g/ccm^3 # True for TNG and mixed for MTNG because of unit difference
            # unit_c = 1.e10 # TNG faq is wrong (see README.md)

            # obtain electron temperature, electron number density and velocity
            Te = (gamma - 1.)*IE/k_B * 4*m_p/(1 + 3*X_H + 4*X_H*EA) * unit_c # K
            ne = EA*X_H*D/m_p # ccm^-3 # True for TNG and mixed for MTNG because of unit difference
            Ve = V*np.sqrt(a) # km/s

            # compute the contribution to the y and b signals of each cell
            # ne*dV cancel unit length of simulation and unit_vol converts ckpc/h^3 to cm^3
            # both should be unitless (const*Te/d_A**2 is cm^2/cm^2; sigma_T/d_A^2 is unitless)
            dY = const*(ne*Te*dV)*unit_vol/(a*Lbox_hkpc*(kpc_to_cm/h)/nPixels)**2.#d_A**2 # Compton Y parameter
            b = sigma_T*(ne[:, None]*(Ve/c)*dV[:, None])*unit_vol/(a*Lbox_hkpc*(kpc_to_cm/h)/nPixels)**2.#d_A**2 # kSZ signal
            tau = sigma_T*(ne*dV)*unit_vol/(a*Lbox_hkpc*(kpc_to_cm/h)/nPixels)**2.#d_A**2 # Optical depth. This is what we use for

            # Now we make the fields:
            
            if projection == 'xy':
                coordinates = Co[:, :2]  # Take x and y coordinates
            elif projection == 'xz':
                coordinates = Co[:, [0, 2]] # Take x and z coordinates
            elif projection == 'yz':
                coordinates = Co[:, 1:] # Take y and z coordinates
            else:
                raise NotImplementedError('Projection type not implemented: ' + projection)

            if pType == 'tSZ':
                # field = hist2d_numba_seq(np.array([coordinates[:, 0], coordinates[:, 1]]), bins=gridSize, ranges=(minMax, minMax), weights=dY)
                result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], values=dY, 
                                            statistic='sum', bins=gridSize, range=[minMax, minMax]) # type: ignore
            elif pType == 'kSZ':
                # field = hist2d_numba_seq(np.array([coordinates[:, 0], coordinates[:, 1]]), bins=gridSize, ranges=(minMax, minMax), weights=b)
                result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], values=b, 
                                            statistic='sum', bins=gridSize, range=[minMax, minMax]) # type: ignore
            elif pType == 'tau':
                # field = hist2d_numba_seq(np.array([coordinates[:, 0], coordinates[:, 1]]), bins=gridSize, ranges=(minMax, minMax), weights=tau)
                result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], values=tau, 
                                            statistic='sum', bins=gridSize, range=[minMax, minMax]) # type: ignore
            else:
                raise ValueError('Particle type not recognized: ' + pType)
            
            field = result.statistic
            field_total += field

            if i % 10 == 0:
                print(f'Processed {i} snapshots, time elapsed: {time.time() - t0:.2f} seconds')
        
        
        print('hist2d time:', time.time() - t0)
        if save:
            if self.simType == 'IllustrisTNG':
                saveName = (self.sim + '_' + str(self.snapshot) + '_' + 
                            pType + '_' + str(nPixels) + '_' + projection)
                np.save(f'/pscratch/sd/r/rhliu/simulations/{self.simType}/products/2D/{saveName}.npy', field_total)
            elif self.simType == 'SIMBA':
                saveName = (self.sim + '_' + self.feedback + '_' + str(self.snapshot) + '_' +  # type: ignore
                            pType + '_' + str(nPixels) + '_' + projection)
                np.save(f'/pscratch/sd/r/rhliu/simulations/{self.simType}/products/2D/{saveName}.npy', field_total)

        return field_total
    
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

