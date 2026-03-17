# !pip install illustris_python
import sys

import numpy as np
import matplotlib.pyplot as plt
import h5py
import warnings

import scipy
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

# from tools import numba_tsc_3D, hist2d_numba_seq
from utils import fft_smoothed_map, comoving_to_arcmin
from halos import halo_ind, select_halos
from filters import total_mass, delta_sigma, CAP, CAP_ringring, CAP_from_mass, DSigma_from_mass, delta_sigma_mccarthy, delta_sigma_kernel, delta_sigma_ring, upsilon
from loadIO import load_subhalos, snap_path, load_halos, load_subsets, load_subset, load_data, save_data
from mapMaker import create_field, create_masked_field


class SimulationStacker(object):

    def __init__(self, 
                 sim: str, 
                 snapshot: int, 
                 nPixels=2000, 
                 simType='IllustrisTNG', 
                 feedback=None, # Only for SIMBA
                 z=0.0):
        """Initialize a SimulationStacker for field creation and halo stacking.

        Sets up simulation paths, loads the snapshot header, and initializes
        in-memory caches for fields and maps.

        Args:
            sim (str): Simulation name. IllustrisTNG options: 'TNG300-1',
                'TNG300-2', 'TNG100-1', 'TNG100-2', 'TNG50-1', 'Illustris-1'.
                SIMBA options: 'm50n512', 'm100n1024'.
            snapshot (int): Snapshot number in the simulation.
            nPixels (int, optional): Number of pixels per side of 2D fields,
                i.e. field shape will be (nPixels, nPixels). Defaults to 2000.
            simType (str, optional): Simulation code, one of
                ['IllustrisTNG', 'SIMBA']. Defaults to 'IllustrisTNG'.
            feedback (str, optional): Feedback model variant for SIMBA only.
                One of ['s50', 's50nox', 's50noagn', 's50nofb', 's50nojet'].
                Required if simType='SIMBA'. Defaults to None.
            z (float, optional): Redshift of the snapshot. Used for
                cosmological distance calculations in maps. Defaults to 0.0.

        Note:
            TODO: Implement automatic snapshot selection from redshift.
        """
        
        self.simType = simType
        if self.simType == 'IllustrisTNG':
            self.simPath = '/pscratch/sd/r/rhliu/simulations/IllustrisTNG/' + sim + '/output/'
        elif self.simType == 'SIMBA':
            self.simPath = '/pscratch/sd/r/rhliu/simulations/SIMBA/' + sim + '/' + feedback + '/' # type: ignore
            assert feedback in ['s50', 's50nox', 's50noagn', 's50nofb', 's50nojet']
        else:
            raise NotImplementedError('Simulation type not implemented')

        self.sim = sim
        self.snapshot = snapshot
        self.nPixels = nPixels
        self.feedback = feedback
        self.z = z
        
        # with h5py.File(il.snapshot.snapPath(self.simPath, self.snapshot), 'r') as f:
        with h5py.File(self.snapPath(), 'r') as f:
        
            self.header = dict(f['Header'].attrs.items())
        
        # self.Lbox = self.header['BoxSize'] # kpc/h
        self.h = self.header['HubbleParam'] # Hubble parameter

        # Define cosmology
        self.cosmo = FlatLambdaCDM(H0=100 * self.header['HubbleParam'], Om0=self.header['Omega0'], Tcmb0=2.7255 * u.K)
        
        # self.kpcPerPixel = self.Lbox / self.nPixels # technically kpc/h per pixel
        self.fields = {}
        self.maps = {}

    def makeField(self, pType, nPixels=None, projection='xy', save=False, load=True, 
                  mask=False, maskRad=3.0, mask_target_mass=10**(13.22), mask_upper_mass=5*10**(14),
                  base_path=None, dim='2D'):
        """Uses a histogram binning to make projected fields (either 2D or 3D) of a given particle type from the simulation.

        Args:
            pType (str): Particle Type. One of 'gas', 'DM', 'Stars', 'BH' for mass maps, 'tSZ', 'kSZ', or 'tau' for SZ maps,
                and 'total' for all masses combined.
            nPixels (int, optional): Number of pixels in each direction of the 2D Field. Defaults to self.nPixels.
            projection (str, optional): Direction of the field projection. Currently only 'xy' is implemented. Defaults to 'xy'.
            save (bool, optional): If True, saves the field to a file. Defaults to False.
            load (bool, optional): If True, loads the field from a file if it exists and returns the field. Defaults to True.
            mask (bool, optional): If True, masks out areas outside of haloes in the field. Defaults to False.
            maskRad (float, optional): Number of virial radii around each halo to keep unmasked. Only used if mask=True. 
                Defaults to 3x virial radii.
            base_path (str, optional): Base path for loading/saving data. Defaults to None, which uses the default path.
            dim (str, optional): Dimension of the field to create. Either '2D' or '3D'. Defaults to '2D'.

        Raises:
            NotImplementedError: If field is not one of the ones listed above.

        Returns:
            np.ndarray: 2D or 3D numpy array of the field for the given particle type.

        TODO: Handle saving and loading of the fields for the masked case.
        """
    
        if nPixels is None:
            nPixels = self.nPixels

        if load:
            try:
                return self.loadData(pType, nPixels=nPixels, projection=projection, type='field', 
                                     mask=mask, maskRad=maskRad, base_path=base_path, dim=dim)
            except ValueError as e:
                print(e)
                print("Computing the field instead...")
                
        if mask:
            haloes = self.loadHalos()
            haloMass = haloes['GroupMass']
            
            halo_mask = select_halos(haloMass, 'massive', target_average_mass=mask_target_mass, upper_mass_bound=mask_upper_mass)
            haloes['GroupMass'] = haloes['GroupMass'][halo_mask]
            haloes['GroupRad'] = haloes['GroupRad'][halo_mask] * maskRad # in kpc/h
            haloes['GroupPos'] = haloes['GroupPos'][halo_mask]

            field = create_masked_field(self, halo_cat=haloes, pType=pType, nPixels=nPixels, projection=projection,
                                        save3D=True, load3D=load, base_path=base_path, dim=dim) # TODO: make save3D and load3D configurable
        else:
            field = create_field(self, pType, nPixels, projection, dim=dim, load=load)
        
        if save:
            # TODO: Handle saving and loading of the fields for the masked case.
            save_data(field, self.simType, self.sim, self.snapshot, 
                      self.feedback, pType, nPixels, projection, 'field', 
                      mask=mask, maskRad=maskRad, base_path=base_path, dim=dim)

        return field

    def makeMap(self, pType, z=None, projection='xy', beamSize=1.6, save=False, load=True, 
                pixelSize=0.5, mask=False, maskRad=3.0, base_path=None):
        """Make a 2D map convolved with a beam for a given particle type.
        This is more realistic than makeField

        Args:
            pType (str): Particle Type. One of 'gas', 'DM', 'Stars', 'BH' for mass maps, 'tSZ', 'kSZ', or 'tau' for SZ maps,
                and 'total' for all masses combined.
            z (float, optional): Redshift of the snapshot. Defaults to None, in which case self.z is used.
            # nPixels (int, optional): Number of pixels in each direction of the 2D map. Defaults to self.nPixels.
            projection (str, optional): Direction of the map projection. Currently only 'xy' is implemented. Defaults to 'xy'.
            beamSize (float, optional): Size of the beam in arcminutes. Defaults to 1.6.
            save (bool, optional): If True, saves the map to a file. Defaults to False.
            load (bool, optional): If True, loads the map from a file if it exists and returns the map. Defaults to True.
            pixelSize (float, optional): The theoretical expected size of each pixel in arcminutes. Defaults to 0.5. arcminPerPixel overrides this to the exact size.
            mask (bool, optional): If True, masks out areas outside of haloes in the map. Defaults to False.
            maskRad (float, optional): Number of virial radii around each halo to keep unmasked. Only used if mask=True.
                Defaults to 3x virial radii.
            base_path (str, optional): Base path for loading/saving data. Defaults to None, which uses the default path.

        Returns:
            np.ndarray: 2D numpy array of the map for the given particle type.
        """        
        if z is None:
            z = self.z
        
        if beamSize == 0.0:
            # Zero beamsize is same as None, so we just change it to None here.
            beamSize = None

        if (beamSize is None) and save:
            print('Saving field instead of map, since beamSize is None.')
            saveField = True
            save = False # Don't save the map if we aren't convolving it, since it's just the same as the field.
        else:
            saveField = False
            
        # First define cosmology
        cosmo = FlatLambdaCDM(H0=100 * self.header['HubbleParam'], Om0=self.header['Omega0'], Tcmb0=2.7255 * u.K)

        # Get distance to the snapshot redshift
        # dA = cosmo.angular_diameter_distance(z).to(u.kpc).value
        # dA *= self.header['HubbleParam']  # Convert to kpc/h
        
        # Get the box size in angular units.
        # theta_arcmin = np.degrees(self.header['BoxSize'] / dA) * 60  # Convert to arcminutes # TODO: this is wrong for sure!! Change ASAP
        theta_arcmin = comoving_to_arcmin(self.header['BoxSize'], z, cosmo=cosmo)
        print(f"Box size: {self.header['BoxSize']} kpc/h , Map size at z={z}: {theta_arcmin:.2f} arcmin")

        # Round up to the nearest integer, pixel size is 0.5 arcmin as in ACT
        nPixels = np.ceil(theta_arcmin / pixelSize).astype(int)
        arcminPerPixel = theta_arcmin / nPixels  # Arcminutes per pixel, this is the true pixelSize after rounding.
        # beamSize_pixel = beamSize / arcminPerPixel  # Convert arcminutes to pixels

        
        # Now that we know the expected pixel size, we try to load the map first before computing it:
        if load:
            try:
                return self.loadData(pType, nPixels=nPixels, projection=projection, type='map', 
                                     mask=mask, maskRad=maskRad, base_path=base_path)
            except ValueError as e:
                print(e)
                print("Computing the map instead...")    
        
        # If we don't have the map pre-saved, we then make the map. 
        # Since this is before doing beam convolution, this step is fine to do using makeField.
        map_ = self.makeField(pType, nPixels=nPixels, projection=projection, save=saveField, load=load, 
                              mask=mask, maskRad=maskRad, base_path=base_path)

        # Convolve the map with a Gaussian beam (only if beamSize is not None)
        if beamSize is not None:
            map_ = fft_smoothed_map(map_, beamSize, pixel_size_arcmin=arcminPerPixel)

        if save:
            save_data(map_, self.simType, self.sim, self.snapshot, 
                      self.feedback, pType, nPixels, projection, 'map', 
                      mask=mask, maskRad=maskRad, base_path=base_path)


        return map_

    def convolveMap(self, map_, fwhm_arcmin, pixel_size_arcmin):
        """
        DEPRECIATED: Check fft_smoothed_map function in utils.py for new convolution code.
        Convolve the map with a Gaussian beam.

        Args:
            map_ (np.ndarray): 2D numpy array of the field for the given particle type.
            fwhm_arcmin (float): Full width at half maximum of the Gaussian beam in arcminutes.
            pixel_size_arcmin (float): Size of the pixel in arcminutes.

        Returns:
            np.ndarray: Convolved 2D numpy array.
        """
        
        sigma_pixels = fwhm_arcmin / (2.355 * pixel_size_arcmin)

        # Apply Gaussian filter
        convolved_map = gaussian_filter(map_, sigma=sigma_pixels, mode='wrap')
        
        return convolved_map
    
    def setMap(self, pType, map_, z=None, projection='xy', pixelSize=0.5):
        """Set a precomputed map for a given particle type.

        Args:
            pType (str): Particle type to set the map for.
            map_ (np.ndarray): 2D numpy array of the map.
            z (float, optional): Redshift of the snapshot. Defaults to None, in which case self.z is used.
            projection (str, optional): Direction of the field projection. Defaults to 'xy'. Options are ['xy', 'yz', 'xz']
        """

        if z is None:
            z = self.z

        self.maps[(pType, z, projection, pixelSize)] = map_

    def setField(self, pType, field_, nPixels=None, projection='xy'):
        """Set a precomputed field for a given particle type.

        Args:
            pType (str): Particle type to set the field for.
            field_ (np.ndarray): 2D numpy array of the field.
            nPixels (int, optional): Number of pixels in each direction of the 2D Field. Defaults to self.nPixels.
            projection (str, optional): Direction of the field projection. Currently only 'xy' is implemented. Defaults to 'xy'.
        """

        if nPixels is None:
            nPixels = self.nPixels

        self.fields[(pType, nPixels, projection)] = field_
    

    def stackMap(self, pType, filterType='cumulative', minRadius=0.5, maxRadius=6.0, numRadii=11,
                 z=None, projection='xy', save=False, load=True, radDistance=1.0, 
                 pixelSize=0.5, beamSize=1.6, 
                 mask=False, maskRad=3.0, subtract_mean=False, 
                 use_subhalos=False, halo_mass_avg=10**(13.22), halo_mass_upper=5*10**(14)):
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
            beamSize (float, optional): Size of the Gaussian beam in arcminutes, defaults to 1.6. If None, no beam convolution is applied.
            mask (bool, optional): If True, masks out areas outside of haloes in the map. Defaults to False.
            maskRad (float, optional): Number of virial radii around each halo to keep unmasked. Only used if mask=True.
                Defaults to 3x virial radii.
            subtract_mean (bool, optional): If True, subtracts the mean of the map before stacking. Defaults to False.
            use_subhalos (bool, optional): If True, uses subhalos in the stacking process. Defaults to False.
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
        fieldKey = (pType, z, projection, pixelSize, beamSize)
        if not (fieldKey in self.maps and self.maps[fieldKey] is not None):
            self.maps[fieldKey] = self.makeMap(pType, z=z, projection=projection, save=save, load=load,
                                               pixelSize=pixelSize, beamSize=beamSize, mask=mask, maskRad=maskRad)

        # If subtract_mean is True, subtract the mean of the map before stacking.
        if subtract_mean:
            map_mean = np.mean(self.maps[fieldKey])
            self.maps[fieldKey] -= map_mean

        # Use the abstracted stacking function
        radii, profiles = self.stack_on_array(
            array=self.maps[fieldKey],
            filterType=filterType,
            minRadius=minRadius,
            maxRadius=maxRadius,
            numRadii=numRadii,
            projection=projection,
            radDistance=radDistance,
            radDistanceUnits='arcmin',
            z=z,
            pixelSize=pixelSize,
            use_subhalos=use_subhalos,
            halo_mass_avg=halo_mass_avg,
            halo_mass_upper=halo_mass_upper,
        )
        
        # restore the mean if subtracted
        if subtract_mean:
            self.maps[fieldKey] += map_mean

        # Unit Conversion specific to SZ maps:
        T_CMB = 2.7255
        if pType == 'tau':
            # In the case of the tau field, we want to do unit conversion from optical depth units to micro-Kelvin.
            # This is done by multiplying the tau field by T_CMB * (v/c)
            v_c = 300000 / 299792458 # velocity over speed of light.
            pixArea = (pixelSize**2) # Convert to arcmin^2 units
            # factor = 1
            profiles = profiles * T_CMB * 1e6 * v_c * pixArea # Convert to micro-Kelvin, the units for kSZ in data.
        elif pType == 'kSZ':
            # TODO: kSZ unit conversion
            pixArea = (pixelSize**2) # Convert to arcmin^2 units
            profiles = profiles * pixArea # Convert to arcmin^2 units
            # pass
        elif pType == 'tSZ':
            # TODO: tSZ unit conversion
            # factor = (180.*60./np.pi)**2
            pixArea = (pixelSize**2) # Convert to arcmin^2 units
            profiles = profiles * pixArea # Convert to arcmin^2 units
            # pass
        else:
            # No unit conversion for other fields.
            pass
        
        return radii, profiles 

    def stackField(self, pType, filterType='cumulative', minRadius=0.1, maxRadius=4.5, numRadii=25,
                   projection='xy', nPixels=None, save=False, load=True, radDistance=1000.0, 
                   mask=False, maskRad=3.0, subtract_mean=False, 
                   use_subhalos=False, halo_mass_avg=10**(13.22), halo_mass_upper=5*10**(14)):
        """Do stacking on the computed field.

        Args:
            pType (str): Particle Type. One of 'gas', 'DM', or 'Stars'
            filterType (str, optional): Stacked Filter Types. One of ['cumulative', 'CAP', 'DSigma', 'upsilon', 'ringring']. Defaults to 'cumulative'.
            minRadius (float, optional): Minimum radius in kpc/h for the stacking. Defaults to 0.1.
            maxRadius (float, optional): Maximum radius in kpc/h for the stacking. Defaults to 4.5.
            numRadii (int, optional): Number of radial bins for the stacking. Defaults to 25.
            projection (str, optional): Direction of the field projection. Currently only 'xy' is implemented. Defaults to 'xy'.
            nPixels (int, optional): Number of pixels in each direction of the 2D Field. Defaults to self.nPixels.
            save (bool, optional): If True, saves the stacked field to a file. Defaults to True.
            load (bool, optional): If True, loads the stacked field from a file if it exists. Defaults to True.
            radDistance (float, optional): Radial distance units for stacking. Defaults to 1000 kpc/h (so converts to 1 Mpc/h).
                If None, uses the mean halo radius from the halo catalog.
            mask (bool, optional): If True, masks out areas outside of haloes in the field. Defaults to False.
            maskRad (float, optional): Number of virial radii around each halo to keep unmasked. Only used if mask=True. 
                Defaults to 3x virial radii.
            subtract_mean (bool, optional): If True, subtracts the mean of the field before stacking. Defaults to False.
            use_subhalos (bool, optional): If True, uses subhalos in the stacking. Defaults to False.
            halo_mass_avg (float, optional): Average halo mass for subhalo selection. Defaults to 10^(13.22).
            halo_mass_upper (float, optional): Upper halo mass limit for subhalo selection. Defaults to 5*10^(14).
        Raises:
            NotImplementedError: If pType is not one of the ones listed above.

        Returns:
            radii, profiles : 1D radii and 2D profiles for the stacked field.
        """

        if nPixels is None:
            nPixels = self.nPixels

        # Load or create the field
        fieldKey = (pType, nPixels, projection)
        if not (fieldKey in self.fields and self.fields[fieldKey] is not None):
            self.fields[fieldKey] = self.makeField(pType, nPixels=nPixels, projection=projection,
                                                   save=save, load=load, mask=mask, maskRad=maskRad)
        else:
            assert self.fields[fieldKey].shape == (nPixels, nPixels), \
                f"Field shape mismatch: {self.fields[fieldKey].shape} != {(nPixels, nPixels)}"

        # Handle radDistance = None case
        if radDistance is None:
            haloes = self.loadHalos()
            mass_min, mass_max, _ = halo_ind(2)
            halo_mask = np.where(np.logical_and((haloes['GroupMass'] > mass_min), (haloes['GroupMass'] < mass_max)))[0]
            radDistance = haloes['GroupRad'][halo_mask].mean()

        # If subtract_mean is True, subtract the mean of the map before stacking.
        if subtract_mean:
            field_mean = np.mean(self.fields[fieldKey])
            self.fields[fieldKey] -= field_mean

        # Use the abstracted stacking function
        radii, profiles = self.stack_on_array(
            array=self.fields[fieldKey],
            filterType=filterType,
            minRadius=minRadius,
            maxRadius=maxRadius,
            numRadii=numRadii,
            projection=projection,
            radDistance=radDistance,
            radDistanceUnits='kpc/h',
            use_subhalos=use_subhalos,
            halo_mass_avg=halo_mass_avg,
            halo_mass_upper=halo_mass_upper,
        )
        
        # restore the mean if subtracted
        # TODO: this may introduce weird numerics behaviour, check later
        if subtract_mean:
            self.fields[fieldKey] += field_mean

        # Apply post-processing for CAP filter
        # This is taken care of in the `stack_on_array` function now
        # if filterType == 'CAP':
        #     radii_CAP = np.linspace(minRadius, maxRadius, 25)
        #     cap_profiles = CAP_from_mass(radii_CAP, radii, profiles.mean(axis=1))
        #     return radii_CAP, cap_profiles
        
        return radii, profiles

    def stack_on_array(self, array, filterType='cumulative', minRadius=0.1, maxRadius=4.5, numRadii=25,
                       projection='xy', radDistance=1000.0, radDistanceUnits='kpc/h', use_subhalos=False,
                       halo_mass_avg=10**(13.22), halo_mass_upper=5*10**(14), halo_abundance_target=5e-4,
                       z=None, pixelSize=0.5):
        """Abstract stacking function that works on any 2D array.

        Args:
            array (np.ndarray): 2D array to stack on. Requires shape (nPixels, nPixels) such that the array is square.
            filterType (str, optional): Stacked Filter Types. One of ['cumulative', 'CAP', 'DSigma', 'upsilon', 'ringring']. Defaults to 'cumulative'.
            minRadius (float, optional): Minimum radius for stacking. Defaults to 0.1.
            maxRadius (float, optional): Maximum radius for stacking. Defaults to 4.5.
            numRadii (int, optional): Number of radial bins for stacking. Defaults to 25.
            projection (str, optional): Direction projection used. Defaults to 'xy'.
            radDistance (float, optional): Radial distance units for stacking. Defaults to 1000.
            radDistanceUnits (str, optional): Units for radDistance. Either 'kpc/h' or 'arcmin'. Defaults to 'kpc/h'.
            halo_mass_avg (float, optional): Average halo mass for selecting halos ('massive' method). Defaults to 10**(13.22).
            halo_mass_upper (float, optional): Upper mass bound for selecting halos ('massive' method). Defaults to 5*10**(14).
            halo_abundance_target (float, optional): Target number density in (cMpc/h)^-3 for subhalo abundance matching ('abundance' method). Defaults to 5e-4.
            z (float, optional): Redshift for angular distance calculation (required if radDistanceUnits='arcmin'). Defaults to None.
            pixelSize (float, optional): Pixel size in arcminutes (required if radDistanceUnits='arcmin'). Defaults to 0.5.

        Returns:
            tuple: (radii, profiles) - 1D radii array and 2D profiles array.
        """
        
        nPixels = array.shape[0]
        assert array.shape == (nPixels, nPixels), f"Array must be square, got shape: {array.shape}"

        # Load the halo catalog and select halos
        if use_subhalos:
            subhalos = self.loadSubHalos()
            haloMass = subhalos['SubhaloMass']
            haloPos = subhalos['SubhaloPos']
        else:
            haloes = self.loadHalos()
            haloMass = haloes['GroupMass']
            haloPos = haloes['GroupPos']

        if halo_mass_avg is None:
            # Use legacy selection method for backward compatibility
            warnings.warn("halo_mass_avg is None, using legacy halo selection method. This may lead to unexpected results. Please specify halo_mass_avg explicitly for consistent behavior.",
                          DeprecationWarning, stacklevel=2)
            halo_mask = select_halos(haloMass, 'binned', ind=2)
        elif use_subhalos:
            halo_mask = select_halos(haloMass, 'abundance',
                                     target_number=halo_abundance_target,
                                     Lbox=self.header['BoxSize'])
        else:
            halo_mask = select_halos(haloMass, 'massive',
                                     target_average_mass=halo_mass_avg,
                                     upper_mass_bound=halo_mass_upper)
        
        print(f'Number of halos selected: {halo_mask.shape[0]}')
        
        # Convert radDistance to pixels based on units
        if radDistanceUnits == 'kpc/h':
            kpcPerPixel = self.header['BoxSize'] / nPixels
            RadPixel = radDistance / kpcPerPixel
            pixelSize_true = kpcPerPixel
        elif radDistanceUnits == 'arcmin':
            if z is None:
                z = self.z
            # Calculate arcmin per pixel
            cosmo = FlatLambdaCDM(H0=100 * self.header['HubbleParam'], Om0=self.header['Omega0'], Tcmb0=2.7255 * u.K)
            # dA = cosmo.angular_diameter_distance(z).to(u.kpc).value
            # dA *= self.header['HubbleParam']  # Convert to kpc/h
            # theta_arcmin = np.degrees(self.header['BoxSize'] / dA) * 60
            theta_arcmin = comoving_to_arcmin(self.header['BoxSize'], z, cosmo=cosmo)
            arcminPerPixel = theta_arcmin / nPixels
            RadPixel = radDistance / arcminPerPixel
            pixelSize_true = arcminPerPixel
        else:
            raise ValueError(f"radDistanceUnits must be 'kpc/h' or 'arcmin', got: {radDistanceUnits}")
        
        # Set up filter function
        if filterType == 'cumulative':
            filterFunc = total_mass
        elif filterType == 'CAP':
            filterFunc = CAP
        elif filterType == 'ringring':
            filterFunc = CAP_ringring
        elif filterType == 'DSigma':
            filterFunc = delta_sigma_kernel
            # filterFunc = delta_sigma_ring
        elif filterType == 'upsilon':
            filterFunc = upsilon
        elif filterType == 'DSigma_mccarthy':
            filterFunc = delta_sigma_mccarthy
            if radDistanceUnits != 'arcmin':
                raise ValueError('DSigma_mccarthy filter currently requires radDistanceUnits to be arcmin')
            if z is None:
                raise ValueError('DSigma_mccarthy filter requires a redshift z to be specified')
        else:
            raise NotImplementedError('Filter Type not implemented: ' + filterType)

        # Set up radial bins and cutout size
        radii = np.linspace(minRadius, maxRadius, numRadii)
        if filterType == 'CAP':
            n_vir = int(np.ceil(np.sqrt(2) * maxRadius)) + 1
        else:
            n_vir = int(radii.max() + 1)  # number of virial radii to cutout

        # Do stacking
        profiles = []
        for j, haloID in enumerate(halo_mask):
            # Get halo position for the specified projection
            if projection == 'xy':
                haloPos_2D = haloPos[haloID, :2]
            elif projection == 'xz':
                haloPos_2D = haloPos[haloID, [0, 2]]
            elif projection == 'yz':
                haloPos_2D = haloPos[haloID, 1:]
            else:
                raise NotImplementedError('Projection type not implemented: ' + projection)
            
            # Convert halo position to pixel coordinates
            if radDistanceUnits == 'kpc/h':
                haloLoc = np.round(haloPos_2D / (self.header['BoxSize'] / nPixels)).astype(int)
            else:  # arcmin units
                haloLoc = np.round(haloPos_2D / (self.header['BoxSize'] / nPixels)).astype(int)
            
            # Create cutout and radial distance grid
            cutout = SimulationStacker.cutout_2d_periodic(array, haloLoc, n_vir*RadPixel)
            rr = SimulationStacker.radial_distance_grid(cutout, (-n_vir, n_vir))
            
            
            if filterType == 'DSigma_mccarthy':
                # Use the Delta Sigma filter from Ian McCarthy et al. 2024
                # pass
                if radDistanceUnits != 'arcmin':
                    raise ValueError('DSigma_mccarthy filter currently requires radDistanceUnits to be arcmin')
                radii, profile, _ = delta_sigma_mccarthy(cutout, rr, pixel_scale_arcmin=pixelSize, z=z, # type: ignore
                                                         cosmo=cosmo, rmin_theta=minRadius, rmax_theta=maxRadius, n_rbins=numRadii)
            elif filterType == 'DSigma':
                # TODO: This does not work with stackField for some reason. (I think fixed?)
                if radDistanceUnits == 'arcmin':
                    dr = 0.5 # 0.5 arcmin in pixels
                else:
                    # dr = 0.2 # 0.2 kpc/h in pixels
                    dr = 3 / RadPixel
                
                profile = []
                for rad in radii:
                    # TODO: pixel_size unit conversions!! Important
                    # filt_result = filterFunc(cutout, rr, rad, pixel_size=1.)  # type: ignore
                    filt_result = filterFunc(cutout, rr, rad, dr=dr, pixel_size=pixelSize_true)  # type: ignore
                    profile.append(filt_result)
                
                profile = np.array(profile)
            else:
                
                # Apply filters at each radius
                profile = []
                for rad in radii:
                    # TODO: pixel_size here is placeholder. Carefully check units!
                    filt_result = filterFunc(cutout, rr, rad, pixel_size=1.) # type: ignore
                    profile.append(filt_result)
                
                profile = np.array(profile)
            
            profiles.append(profile)
            
        profiles = np.array(profiles).T

        # if filterType == 'CAP':
        #     # Post-process CAP profiles to convert to physical units
        #     radii_CAP = np.linspace(minRadius, maxRadius, 25)
        #     cap_profiles = CAP_from_mass(radii_CAP, radii, profiles.mean(axis=1))
        #     return radii_CAP, cap_profiles
        # if filterType == 'DSigma':
        #     # Post-process DSigma profiles to convert to physical units
        #     # radii_DSigma = np.linspace(minRadius, maxRadius, 9)
        #     dsigma_profiles = DSigma_from_mass(radii, radii, profiles)
        #     return radii, dsigma_profiles
        
        return radii, profiles

    # Other util functions:
    
    @staticmethod
    def cutout_2d_periodic(array, center, length):
        """Extract a square cutout from a 2D array with periodic boundary conditions.

        Args:
            array (np.ndarray): 2D input array.
            center (tuple): (x, y) center pixel coordinates for the cutout.
            length (float or int): Half-width of the cutout in pixels.
                Rounded to the nearest integer.

        Returns:
            np.ndarray: Square cutout, shape (2*length+1, 2*length+1).
        """
        length = int(round(length))
        x, y = center
        size = 2 * length + 1
    
        # Generate index ranges with wrapping
        row_indices = [(x + i) % array.shape[0] for i in range(-length, length + 1)]
        col_indices = [(y + j) % array.shape[1] for j in range(-length, length + 1)]
    
        # Use np.ix_ to create a 2D index grid
        cutout = array[np.ix_(row_indices, col_indices)]
    
        return cutout
    
    @staticmethod
    def radial_distance_grid(array, bounds):
        """Create a 2D radial distance grid centred at (0, 0) in physical coordinates.

        Args:
            array (np.ndarray): 2D array whose shape (rows, cols) defines the
                grid resolution (only shape is used).
            bounds (tuple): (xy_min, xy_max) physical coordinate range applied
                equally to both axes (e.g. in kpc/h or virial radii).

        Returns:
            np.ndarray: 2D array of radial distances from (0, 0), same shape
                as array.
        """
        rows, cols = array.shape
        xy_min, xy_max = bounds
    
        # Generate coordinate values for each axis
        x_coords = np.linspace(xy_min, xy_max, cols)
        y_coords = np.linspace(xy_min, xy_max, rows)
    
        # Create meshgrid of coordinates
        X, Y = np.meshgrid(x_coords, y_coords)
    
        # Calculate distances from the center (0,0)
        radial_distances = np.sqrt(X**2 + Y**2)
        
        return radial_distances
    

    
    # Some tools for file handling and loading:

    def snapPath(self, chunkNum=0, pathOnly=False):
        """Get the snapshot file path for the current simulation.

        Args:
            chunkNum (int, optional): Chunk number for multi-file snapshots
                (IllustrisTNG only). Defaults to 0.
            pathOnly (bool, optional): If True, return only the directory path.
                Defaults to False.

        Returns:
            str: Full path to the snapshot HDF5 file, or directory if pathOnly.
        """
        return snap_path(self.simPath, self.snapshot, self.simType, 
                        sim_name=self.sim, feedback=self.feedback, 
                        chunk_num=chunkNum, path_only=pathOnly)

    def loadHalos(self):
        """Load the FoF halo catalogue for the current simulation and snapshot.

        Returns:
            dict: Halo catalogue with keys 'GroupMass' (M_sun/h), 'GroupPos'
                (kpc/h, shape (nHalos, 3)), and 'GroupRad' (kpc/h).
        """
        return load_halos(self.simPath, self.snapshot, self.simType, 
                         sim_name=self.sim, header=self.header)
    
    def loadSubHalos(self):
        """Load the subhalo catalogue for the current simulation and snapshot.

        Returns:
            dict: Subhalo catalogue with keys 'SubhaloMass' (M_sun/h),
                'SubhaloPos' (kpc/h, shape (nSubhalos, 3)), 'SubhaloGrNr',
                and 'SubhaloMStar' (M_sun/h).
        """
        return load_subhalos(self.simPath, self.snapshot, self.simType, 
                            sim_name=self.sim, header=self.header)

    def loadSubsets(self, pType, keys=None):
        """Load particle data for the specified type from the full snapshot.

        Args:
            pType (str): Particle type, e.g. 'gas', 'DM', 'Stars', 'BH'.
            keys (list, optional): Specific fields to load, e.g.
                ['Coordinates', 'Masses']. Defaults to None (uses defaults
                from loadIO based on pType).

        Returns:
            dict: Particle data arrays keyed by field name.
        """
        return load_subsets(self.simPath, self.snapshot, self.simType, pType,
                           sim_name=self.sim, feedback=self.feedback, header=self.header, keys=keys)

    def loadSubset(self, pType, snapPath, keys=None):
        """Load particle data for the specified type from a specific snapshot file.

        Args:
            pType (str): Particle type, e.g. 'gas', 'DM', 'Stars', 'BH'.
            snapPath (str): Full path to the snapshot HDF5 file.
            keys (list, optional): Specific fields to load. Defaults to None
                (uses ['Coordinates', 'Masses']).

        Returns:
            dict: Particle data arrays keyed by field name.
        """
        return load_subset(self.simPath, self.snapshot, self.simType, pType, snapPath,
                          header=self.header, keys=keys, sim_name=self.sim)

    def loadData(self, pType, nPixels=None, projection='xy', type='field',
                 mask=False, maskRad=3.0, base_path=None, dim='2D'):
        """Load a cached field or map from file.

        Args:
            pType (str): Particle type identifier.
            nPixels (int, optional): Number of pixels per side. Defaults to
                self.nPixels.
            projection (str, optional): Projection direction, one of
                ['xy', 'xz', 'yz']. Defaults to 'xy'.
            type (str, optional): Data type, one of ['field', 'map'].
                Defaults to 'field'.
            mask (bool, optional): If True, load the masked version.
                Defaults to False.
            maskRad (float, optional): Mask radius scale used in filename.
                Defaults to 3.0.
            base_path (str, optional): Base directory for cached files.
                Defaults to None (uses /pscratch/ default).
            dim (str, optional): Dimensionality, one of ['2D', '3D'].
                Defaults to '2D'.

        Returns:
            np.ndarray: Cached field or map.

        Raises:
            ValueError: If the file does not exist.
        """
        if nPixels is None:
            nPixels = self.nPixels
        return load_data(self.simType, self.sim, self.snapshot, 
                         self.feedback, pType, nPixels, projection, type, 
                         mask=mask, maskRad=maskRad, base_path=base_path, dim=dim)



if __name__ == "__main__":
    pass