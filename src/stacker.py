# !pip install illustris_python
import sys

import numpy as np
import matplotlib.pyplot as plt
import h5py

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

from tools import numba_tsc_3D, hist2d_numba_seq
from utils import fft_smoothed_map
from halos import select_massive_halos, halo_ind
from filters import total_mass, delta_sigma, CAP, CAP_from_mass, DSigma_from_mass
from loadIO import snap_path, load_halos, load_subsets, load_subset, load_data, save_data


class SimulationStacker(object):

    def __init__(self, 
                 sim: str, 
                 snapshot: int, 
                 nPixels=2000, 
                 simType='IllustrisTNG', 
                 feedback=None, # Only for SIMBA
                 z=0.0):
        """_summary_

        Args:
            sim (str): Simulation Instance, One of ['TNG300-1', 'TNG300-2', 'TNG100-1', 'TNG100-2', 'm50n512', 'm100n1024']
            snapshot (int): _description_
            nPixels (int, optional): Pixel size of the output 2D field, i.e. the number of pixels in each direction.
            simType (str, optional): Simulation type, one of ['IllustrisTNG', 'SIMBA']. Defaults to 'IllustrisTNG'.
            feedback (str, optional): feedback types for SIMBA. Defaults to None. One of 
                ['s50', 's50nox', 's50noagn', 's50nofb', 's50nojet'].
            z (float, optional): Redshift of the snapshot. Defaults to 0.0.
            
        TODO:
            Add support for tSZ and kSZ maps!
            Automatic selection of closest snapshot from redshift specification!
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
        with h5py.File(self.snapPath(self.simType), 'r') as f:
        
            self.header = dict(f['Header'].attrs.items())
        
        # self.Lbox = self.header['BoxSize'] # kpc/h
        self.h = self.header['HubbleParam'] # Hubble parameter
        
        # self.kpcPerPixel = self.Lbox / self.nPixels # technically kpc/h per pixel
        self.fields = {}
        self.maps = {}

    def makeMap(self, pType, z=None, projection='xy', beamsize=1.6, save=False, load=True, pixelSize=0.5):
        """Make a 2D map convolved with a beam for a given particle type. 
        This is more realistic than makeField

        Args:
            pType (str): Particle Type. One of 'gas', 'DM', 'Stars', or 'BH'
            z (float, optional): Redshift of the snapshot. Defaults to None, in which case self.z is used.
            # nPixels (int, optional): Number of pixels in each direction of the 2D map. Defaults to self.nPixels.
            projection (str, optional): Direction of the map projection. Currently only 'xy' is implemented. Defaults to 'xy'.
            beamsize (float, optional): Size of the beam in arcminutes. Defaults to 1.6.
            save (bool, optional): If True, saves the map to a file. Defaults to False.
            load (bool, optional): If True, loads the map from a file if it exists and returns the map. Defaults to True.
            pixelSize (float, optional): Size of each pixel in arcminutes. Defaults to 0.5.

        Returns:
            np.ndarray: 2D numpy array of the map for the given particle type.
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
        
        # If we don't have the map pre-saved, we then make the map. 
        # Since this is before doing beam convolution, this step is fine to do using makeField.
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

    def convolveMap(self, map_, fwhm_arcmin, pixel_size_arcmin):
        """
        DEPRECIATED: Check below for new Convolution code
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
    
    # def get_smooth_density(D, fwhm, Lbox, pizel_size_arcmin, N_dim):
    #     """
    #     Smooth density map D ((0, Lbox] and N_dim^2 cells) with Gaussian beam of FWHM
    #     """
    #   
    #     kstep = 2*np.pi/(N_dim*np.pi/180*pizel_size_arcmin)
    #     #karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim)) # physical (not correct)
    #     karr = np.fft.fftfreq(N_dim, d=Lboxdeg*np.pi/180./(2*np.pi*N_dim)) # angular
    #     print("kstep = ", kstep, karr[1]-karr[0]) # N_dim/d gives kstep
    #
    #     # fourier transform the map and apply gaussian beam
    #     D = D.astype(np.float32)
    #     dfour = scipy.fft.fftn(D, workers=-1)
    #     dksmo = np.zeros((N_dim, N_dim), dtype=np.complex64)
    #     ksq = np.zeros((N_dim, N_dim), dtype=np.complex64)
    #     ksq[:, :] = karr[None, :]**2+karr[:,None]**2
    #     dksmo[:, :] = SimulationStacker.gauss_beam(ksq, fwhm)*dfour
    #     drsmo = np.real(scipy.fft.ifftn(dksmo, workers=-1))
    #
    #     return drsmo

    def makeField(self, pType, nPixels=None, projection='xy', save=False, load=True):
        """Used a histogram binning to make projected 2D fields of a given particle type from the simulation.

        Args:
            pType (str): Particle Type. One of 'gas', 'DM', 'Stars', or 'BH'
            nPixels (int, optional): Number of pixels in each direction of the 2D Field. Defaults to self.nPixels.
            projection (str, optional): Direction of the field projection. Currently only 'xy' is implemented. Defaults to 'xy'.
            save (bool, optional): If True, saves the field to a file. Defaults to False.
            load (bool, optional): If True, loads the field from a file if it exists and returns the field. Defaults to True.

        Raises:
            NotImplementedError: If field is not one of the ones listed above.

        Returns:
            np.ndarry: 2D numpy array of the field for the given particle type.
            
        TODO:
            Add directionality to the fields (i.e. x, y, and z projected 2D fields.)
        """
        if nPixels is None:
            nPixels = self.nPixels

        if load:
            try:
                return self.loadData(pType, nPixels=nPixels, projection=projection, type='field')
            except ValueError as e:
                print(e)
                print("Computing the field instead...")
               
        Lbox = self.header['BoxSize'] # kpc/h
        
        # Get all particle snap chunks:

        folderPath = self.snapPath(self.simType, pathOnly=True)
        if self.simType == 'IllustrisTNG':
            snaps = glob.glob(folderPath + 'snap_*.hdf5') # TODO: fix this for SIMBA (done I think)
        elif self.simType == 'SIMBA':
            snaps = glob.glob(folderPath)
            print('Snaps:', snaps)
        # The code below does the statistic by chunk rather than by the whole dataset
        
        # Initialize empty maps
        gridSize = [nPixels, nPixels]
        minMax = [0, self.header['BoxSize']]
        field_total = np.zeros(gridSize)
        
        t0 = time.time()
        for i, snap in enumerate(snaps):
            particles = self.loadSubset(pType, snapPath=snap)
            coordinates = particles['Coordinates'] # kpc/h
            masses = particles['Masses']  * 1e10 / self.header['HubbleParam'] # Msun/h
            
            if projection == 'xy':
                coordinates = coordinates[:, :2]  # Take x and y coordinates
            elif projection == 'xz':
                coordinates = coordinates[:, [0, 2]] # Take x and z coordinates
            elif projection == 'yz':
                coordinates = coordinates[:, 1:] # Take y and z coordinates
            else:
                raise NotImplementedError('Projection type not implemented: ' + projection)
            
            # Convert coordinates to pixel coordinates        

            result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], masses, 
                                        'sum', bins=gridSize, range=[minMax, minMax]) # type: ignore
            field = result.statistic
            
            field_total += field

            if i % 10 == 0:
                print(f'Processed {i} snapshots, time elapsed: {time.time() - t0:.2f} seconds')

        print('Binned statistic time:', time.time() - t0)


        # particles = self.loadSubsets(pType)
        # coordinates = particles['Coordinates'] # kpc/h
        # masses = particles['Masses'] # Msun/h
        
                
        # if projection == 'xy':
        #     coordinates = coordinates[:, :2]  # Take x and y coordinates
        # elif projection == 'xz':
        #     coordinates = coordinates[:, [0, 2]] # Take x and z coordinates
        # elif projection == 'yz':
        #     coordinates = coordinates[:, 1:] # Take y and z coordinates
        # else:
        #     raise NotImplementedError('Projection type not implemented: ' + projection)
        
        # # Convert coordinates to pixel coordinates        
        # gridSize = [nPixels, nPixels]
        # minMax = [0, self.header['BoxSize']]

        # t0 = time.time()
        # result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], masses, 
        #                              'sum', bins=gridSize, range=[minMax, minMax]) # type: ignore
        # field = result.statistic
        
        # print('Binned statistic time:', time.time() - t0)
        
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
                 halo_mass_avg=10**(13.22), halo_mass_upper=None):
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
        
        fieldKey = (pType, z, projection, pixelSize)
        if not (fieldKey in self.maps and self.maps[fieldKey] is not None):
            self.maps[fieldKey] = self.makeMap(pType, z=z, projection=projection,
                                               save=save, load=load, pixelSize=pixelSize)
        
        nPixels = self.maps[fieldKey].shape[0]  # Get the number of pixels from the loaded map
        assert self.maps[fieldKey].shape == (nPixels, nPixels), f"Map shape mismatch: {self.maps[fieldKey].shape} != {(nPixels, nPixels)}"

        # Get the true arcmin per pixel:
        cosmo = FlatLambdaCDM(H0=100 * self.header['HubbleParam'], Om0=self.header['Omega0'], Tcmb0=2.7255 * u.K)
        # Get distance to the snapshot redshift
        dA = cosmo.angular_diameter_distance(z).to(u.kpc).value
        dA *= self.header['HubbleParam']  # Convert to kpc/h
        # Get the box size in angular units.
        theta_arcmin = np.degrees(self.header['BoxSize'] / dA) * 60  # Convert to arcminutes
        arcminPerPixel = theta_arcmin / nPixels  # Arcminutes per pixel, this is the true pixelSize after rounding.


        # Load the halo catalog
        haloes = self.loadHalos(self.simType)
        haloMass = haloes['GroupMass']
        haloPos = haloes['GroupPos']
        halo_mask = select_massive_halos(haloMass, halo_mass_avg, halo_mass_upper)
        print('Number of halos selected:', halo_mask.shape[0])
        
        RadPixel = radDistance / arcminPerPixel # Convert to pixels
        
        # Fix the stacking function:
        # filterFunc = SimulationStacker.total_mass
        if filterType == 'cumulative':
            filterFunc = total_mass
        elif filterType == 'CAP':
            filterFunc = CAP
        elif filterType == 'DSigma':
            filterFunc = delta_sigma
        #TODO: DSigma Filter Function.


        # Now do stacking - same code below as stackField, but using the map instead of the field.
        i = 0
        profiles = []
        # if filterType == 'CAP':
        #     dx = (maxRadius - minRadius) / (numRadii - 1)
        #     new_max = np.sqrt(2) * maxRadius
        #     n_new = int(np.ceil((new_max - minRadius) / dx)) + 1

        #     radii = np.linspace(minRadius, minRadius + dx * (n_new - 1), n_new)
        #     # TODO: Check that this part makes sense!!!
        # else:
        #     radii = np.linspace(minRadius, maxRadius, numRadii)
        radii = np.linspace(minRadius, maxRadius, numRadii)
        if filterType == 'CAP':
            n_vir = int(np.ceil(np.sqrt(2) * maxRadius)) + 1
        else:
            n_vir = radii.max() + 1 # number of virial radii to cutout

        profiles = []

        for j, haloID in enumerate(halo_mask):
            # Load the snapshot for gas and DM around that halo:
            if projection == 'xy':
                haloPos_2D = haloPos[haloID, :2]
            elif projection == 'xz':
                haloPos_2D = haloPos[haloID, [0, 2]]
            elif projection == 'yz':
                haloPos_2D = haloPos[haloID, 1:]
            
            # Convert halo position to pixel coordinates in arcminutes
            
            haloLoc = np.round(haloPos_2D / self.header['BoxSize'] * nPixels).astype(int)  # Convert to arcminutes
            # haloLoc = np.round(haloPos_2D / kpcPerPixel).astype(int) # Halo location in pixels
            # fieldKey = (pType, nPixels, projection, pixelSize)
            cutout = SimulationStacker.cutout_2d_periodic(self.maps[fieldKey], haloLoc, n_vir*RadPixel)

            rr = SimulationStacker.radial_distance_grid(cutout, (-n_vir, n_vir))
            
            profile = []
    
            for rad in radii:
                filt_result = filterFunc(cutout, rr, rad)
                profile.append(filt_result)
        
        
            profile = np.array(profile)
            profiles.append(profile)
            # print(i)
            i += 1
            
        profiles = np.array(profiles).T
        
        return radii, profiles
        # if filterType == 'cumulative':
        #     return radii, profiles
        # elif filterType == 'CAP':
        #     radii_CAP = np.linspace(minRadius, maxRadius, numRadii)
        #     cap_profiles = SimulationStacker.CAP_from_mass(radii_CAP, radii, profiles.mean(axis=1))
        #     return radii_CAP, cap_profiles
        # else:
        #     raise NotImplementedError('Filter Type not implemented: ' + filterType)
        # return 
        
    def stackField(self, pType, filterType='cumulative', minRadius=0.1, maxRadius=4.5, numRadii=25,
                   projection='xy', nPixels=None, save=False, load=True, radDistance=1000):
        """Do stacking on the computed field.

        Args:
            pType (str): Particle Type. One of 'gas', 'DM', or 'Stars'
            filterType (str, optional): Stacked Filter Types. One of ['cumulative', 'CAP', 'DSigma']. Defaults to 'cumulative'.
            minRadius (float, optional): Minimum radius in kpc/h for the stacking. Defaults to 0.1.
            maxRadius (float, optional): Maximum radius in kpc/h for the stacking. Defaults to 4.5.
            numRadii (int, optional): Number of radial bins for the stacking. Defaults to 25.
            projection (str, optional): Direction of the field projection. Currently only 'xy' is implemented. Defaults to 'xy'.
            nPixels (int, optional): Number of pixels in each direction of the 2D Field. Defaults to self.nPixels.
            save (bool, optional): If True, saves the stacked field to a file. Defaults to True.
            load (bool, optional): If True, loads the stacked field from a file if it exists. Defaults to True.
            radDistance (float, optional): Radial distance units for stacking. Defaults to 1000 kpc/h (so converts to 1 Mpc/h).
                If None, uses the mean halo radius from the halo catalog.

        Raises:
            NotImplementedError: If pType is not one of the ones listed above.

        Returns:
            radii, profiles : 1D radii and 2D profiles for the stacked field.
        """

        
        if nPixels is None:
            nPixels = self.nPixels
            kpcPerPixel = self.header['BoxSize'] / nPixels # kpc/h per pixel

        # if not self.fields.get(pType):
        fieldKey = (pType, nPixels, projection)
        # fieldKey = pType + '_' + str(self.nPixels) + '_xy'
        if not (fieldKey in self.fields and self.fields[fieldKey] is not None):
            self.fields[fieldKey] = self.makeField(pType, nPixels=self.nPixels, projection=projection,
                                                   save=save, load=load)
        else:
            assert self.fields[fieldKey].shape == (nPixels, nPixels), \
                f"Field shape mismatch: {self.fields[fieldKey].shape} != {(nPixels, nPixels)}"

        # Load the halo catalog
        # Note: This is a bit of a hack, but it works for now. - this is according to copilot lmao
        haloes = self.loadHalos(self.simType)
        haloMass = haloes['GroupMass']
        haloPos = haloes['GroupPos']
        
        mass_min, mass_max, _ = halo_ind(2)
        
        halo_mask = np.where(np.logical_and((haloMass > mass_min), (haloMass < mass_max)))[0]
        print(halo_mask.shape)
        
        # if self.R200 is None:
        #     R200 = haloes['GroupRad'][halo_mask].mean()
        # else:
        #     R200 = self.R200 # kpc/h, from the input
        # R200_Pixel = R200 / kpcPerPixel
        
        #
        if radDistance is None:
            radDistance = haloes['GroupRad'][halo_mask].mean() # kpc/h, from the input, 
        
        RadPixel = radDistance / kpcPerPixel # Convert to pixels    
        
        # Fix the stacking function:
        # filterFunc = SimulationStacker.total_mass
        if filterType == 'cumulative':
            filterFunc = total_mass
        elif filterType == 'CAP':
            filterFunc = CAP
        elif filterType == 'DSigma':
            filterFunc = delta_sigma
        #TODO: DSigma Filter Function.


        # Do stacking
        i = 0
        profiles = []
        # if filterType == 'CAP':
        #     dx = (maxRadius - minRadius) / (numRadii - 1)
        #     new_max = np.sqrt(2) * maxRadius
        #     n_new = int(np.ceil((new_max - minRadius) / dx)) + 1

        #     radii = np.linspace(minRadius, minRadius + dx * (n_new - 1), n_new)
        #     # TODO: Check that this part makes sense!!!
        # else:
        #     radii = np.linspace(minRadius, maxRadius, numRadii)
        radii = np.linspace(minRadius, maxRadius, numRadii)
        if filterType == 'CAP':
            n_vir = int(np.ceil(np.sqrt(2) * maxRadius)) + 1
        else:
            n_vir = radii.max() + 1 # number of virial radii to cutout
            
        profiles = []

        for j, haloID in enumerate(halo_mask):
        
            # Load the snapshot for gas and DM around that halo:
            if projection == 'xy':
                haloPos_2D = haloPos[haloID, :2]
            elif projection == 'xz':
                haloPos_2D = haloPos[haloID, [0, 2]]
            elif projection == 'yz':
                haloPos_2D = haloPos[haloID, 1:]

            haloLoc = np.round(haloPos_2D / kpcPerPixel).astype(int) # Halo location in pixels # type: ignore
            fieldKey = (pType, nPixels, projection)
            cutout = SimulationStacker.cutout_2d_periodic(self.fields[fieldKey], haloLoc, n_vir*RadPixel)

            rr = SimulationStacker.radial_distance_grid(cutout, (-n_vir, n_vir))
            
            profile = []
    
            for rad in radii:
                filt_result = filterFunc(cutout, rr, rad)
                profile.append(filt_result)
        
        
            profile = np.array(profile)
            profiles.append(profile)
            # print(i)
            i += 1
            
        profiles = np.array(profiles).T
        
        if filterType == 'cumulative':
            return radii, profiles
        elif filterType == 'CAP':
            radii_CAP = np.linspace(minRadius, maxRadius, 25)
            cap_profiles = CAP_from_mass(radii_CAP, radii, profiles.mean(axis=1))
            return radii_CAP, cap_profiles
        else:
            raise NotImplementedError('Filter Type not implemented: ' + filterType)


    # Other util functions:
    
    @staticmethod
    def cutout_2d_periodic(array, center, length):
        """
        Returns a square cutout from a 2D array with periodic boundary conditions.
    
        Parameters:
        - array: 2D numpy array
        - center: tuple (x, y) center index
        - length: float or int, half-width of the cutout (will be rounded)
    
        Returns:
        - 2D numpy array cutout of shape (2*length+1, 2*length+1)
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
        """
        array: 2D numpy array (only shape is used)
        bounds: tuple ((x_min, x_max), (y_min, y_max)) representing physical bounds
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

    def snapPath(self, simType, chunkNum=0, pathOnly=False):
        """Get the snapshot path for the given simulation type."""
        return snap_path(self.simPath, self.snapshot, simType, 
                        sim_name=self.sim, feedback=self.feedback, 
                        chunk_num=chunkNum, path_only=pathOnly)

    def loadHalos(self, simType):
        """Load halo data for the specified simulation type."""
        return load_halos(self.simPath, self.snapshot, simType, 
                         sim_name=self.sim, header=self.header)

    def loadSubsets(self, pType, keys=None):
        """Load particle subsets for the specified particle type."""
        return load_subsets(self.simPath, self.snapshot, self.simType, pType,
                           sim_name=self.sim, feedback=self.feedback, header=self.header, keys=keys)

    def loadSubset(self, pType, snapPath, keys=None):
        """Load a subset of particles from the snapshot."""
        return load_subset(self.simPath, self.snapshot, self.simType, pType, snapPath,
                          header=self.header, keys=keys, sim_name=self.sim)

    def loadData(self, pType, nPixels=None, projection='xy', type='field'):
        """Load a precomputed field or map from file."""
        if nPixels is None:
            nPixels = self.nPixels
        return load_data(self.simPath, self.simType, self.sim, self.snapshot, 
                        self.feedback, pType, nPixels, projection, type)

    # def snapPath(self, simType, chunkNum=0, pathOnly=False):
    #     """Get the snapshot path for the given simulation type.

    #     Args:
    #         simType (str): The type of simulation (e.g., 'IllustrisTNG', 'SIMBA').
    #         chunkNum (int): The chunk number for the simulation (Only used in the case of IllustrisTNG currently)

    #     Returns:
    #         str: The path to the snapshot file.
    #     """
    #     if simType == 'IllustrisTNG':
    #         folderPath = self.simPath + '/snapdir_' + str(self.snapshot).zfill(3) + '/'
    #         snapPath = il.snapshot.snapPath(self.simPath, self.snapshot, chunkNum=chunkNum)
    #     elif simType == 'SIMBA':
    #         folderPath = self.simPath + 'snapshots/'
    #         snapPath = folderPath + 'snap_' + self.sim + '_' + str(self.snapshot) + '.hdf5'
    #         folderPath = snapPath # This is hacky, we do this because SIMBA has a different file structure. TODO: Make this less hacky.
    #     if pathOnly:
    #         return folderPath
    #     else:
    #         return snapPath

    # def loadHalos(self, simType):
    #     """Load halo data for the specified simulation type.

    #     Args:
    #         simType (str): The type of simulation (e.g., 'IllustrisTNG', 'SIMBA').

    #     Returns:
    #         dict: A dictionary containing halo properties (e.g., mass, position, radius).
    #     """
        
    #     if simType == 'IllustrisTNG':
    #         haloes = {}
    #         haloes_cat = il.groupcat.loadHalos(self.simPath, self.snapshot)
    #         # haloes['GroupMass'] = haloes_cat['GroupMass'] * 1e10 * self.header['HubbleParam'] # Convert to solar masses
    #         haloes['GroupMass'] = haloes_cat['GroupMass'] * 1e10 # Convert to Msun/h
    #         haloes['GroupPos'] = haloes_cat['GroupPos']
    #         haloes['GroupRad'] = haloes_cat['Group_R_TopHat200']
            
    #     elif simType == 'SIMBA':
    #         haloPath = self.simPath + 'catalogs/' +  self.sim + '_' + str(self.snapshot) + '.hdf5'
    #         haloes = {}
    #         with h5py.File(haloPath, 'r') as f:
    #             # Print all top-level groups/datasets
    #             # print("Keys:")
    #             # print(f['halo_data']['dicts'].keys())
    #             haloes['GroupPos'] = f['halo_data']['pos'][:] * self.header['HubbleParam'] # kpc/h # type: ignore
    #             # haloes['GroupMass'] = f['halo_data']['dicts']['masses.total'][:] # SIMBA already in solar masses (I think)
    #             haloes['GroupMass'] = f['halo_data']['dicts']['masses.total'][:] * self.header['HubbleParam'] # Convert to Msun/h # type: ignore
    #             # haloes['GroupMass'] = f['halo_data']['dicts']['virial_quantities.m200c'][:]
    #             haloes['GroupRad'] = f['halo_data']['dicts']['virial_quantities.r200c'][:] * self.header['HubbleParam'] # kpc/h # type: ignore
    #     return haloes
    
    # def loadSubsets(self, pType):
    #     """Load particle subsets for the specified particle type.

    #     Args:
    #         pType (str): The type of particles to load (e.g., 'gas', 'DM', 'Stars').

    #     Raises:
    #         NotImplementedError: If the particle type is not implemented.

    #     Returns:
    #         dict: A dictionary containing the particle properties.
    #     """
        
    #     if self.simType == 'IllustrisTNG':
    #         if pType =='gas':
    #             particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['Masses','Coordinates'])
    #         elif pType == 'DM':
    #             particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['ParticleIDs','Coordinates'])
    #             particles['Masses'] = self.header['MassTable'][1] * np.ones_like(particles['ParticleIDs'])  # DM mass
    #         elif pType == 'Stars':
    #             particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['Masses','Coordinates'])
    #         elif pType == 'BH':
    #             # TODO: Check to make sure that the masses here are correct.
    #             particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['Masses','Coordinates'])
    #         else:
    #             raise NotImplementedError('Particle Type not implemented')
                                        
    #     elif self.simType == 'SIMBA':
    #         if pType == 'gas':
    #             pTypeval = 'PartType0'
    #         elif pType == 'DM':
    #             pTypeval = 'PartType1'
    #         elif pType == 'Stars':
    #             pTypeval = 'PartType4'
    #         elif pType == 'BH':
    #             # TODO: Check that the masses here make sense.
    #             pTypeval = 'PartType5'
    #         else:
    #             raise NotImplementedError('Particle Type not implemented')
            
    #         keys = ['Coordinates', 'Masses']
    #         snapPath = self.simPath + 'snapshots/snap_' + self.sim + '_' + str(self.snapshot) + '.hdf5'
    #         particles = {}
    #         with h5py.File(snapPath, 'r') as f:
    #             # Print all top-level groups/datasets
    #             # print("Keys:")
    #             # print(list(f.keys()))
    #             # particles = f['PartType0']
    #             header = dict(f['Header'].attrs.items())
    #             for key in keys:
    #                 particles[key] = f[pTypeval][key][:] # type: ignore
            
    #     particles['Masses'] = particles['Masses'] * 1e10 / self.header['HubbleParam'] # Convert masses to Msun/h
    #     return particles

    # def loadSubset(self, pType, snapPath, keys=None):
    #     """Load a subset of particles from the snapshot.

    #     Args:
    #         pType (str): The type of particles to load (e.g., 'gas', 'DM', 'Stars').
    #         snapPath (str): The path to the snapshot file.
    #         keys (list, optional): The keys to load from the snapshot. Defaults to ['Coordinates', 'Masses'].

    #     Raises:
    #         NotImplementedError: If the particle type is not implemented.

    #     Returns:
    #         dict: A dictionary containing the particle properties.
    #     """
    #      # Avoid mutable default arg; build a fresh list each call
    #     if keys is None:
    #         keys = ['Coordinates', 'Masses']
    #     read_keys = list(keys)  # copy so we can mutate safely
        
    #     addMass = False # This is to handle the case for IllustrisTNG sims not having 'Masses' as a category in sims.
    #     if pType == 'gas':
    #         pTypeval = 'PartType0'
    #     elif pType == 'DM':
    #         pTypeval = 'PartType1'
            
    #         if 'Masses' in read_keys and self.simType == 'IllustrisTNG':
    #              read_keys[read_keys.index('Masses')] = 'ParticleIDs'
    #              addMass = True
    #     elif pType == 'Stars':
    #         pTypeval = 'PartType4'
    #     elif pType == 'BH':
    #         # TODO: Check that the masses here make sense.
    #         pTypeval = 'PartType5'
    #     else:
    #         raise NotImplementedError('Particle Type not implemented')

    #     particles = {}
    #     with h5py.File(snapPath, 'r') as f:
    #         # Print all top-level groups/datasets
    #         # print("Keys:")
    #         # print(list(f.keys()))
    #         # particles = f['PartType0']
    #         header = dict(f['Header'].attrs.items())
    #         for key in read_keys:
    #             particles[key] = f[pTypeval][key][:] # type: ignore

    #     if addMass:
    #         particles['Masses'] = self.header['MassTable'][1] * np.ones_like(particles['ParticleIDs'])  # DM mass
    #         del particles['ParticleIDs']  # Remove ParticleIDs if we added Masses
                
    #     particles['Masses'] = particles['Masses'] * 1e10 / self.header['HubbleParam'] # Convert masses to Msun/h
    #     return particles



if __name__ == "__main__":
    pass