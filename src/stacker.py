# !pip install illustris_python
import sys

import numpy as np
# import matplotlib.pyplot as plt
import h5py

from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

# from abacusnbody.analysis.tsc import tsc_parallel
import time
# import json
# import pprint 

sys.path.append('../../illustrisPython/')
import illustris_python as il # type: ignore

# print('Import Parameters from JSON')
# JSON Parameters:
# param_Dict = json.loads(sys.argv[1])
# locals().update(param_Dict)

# print('Parameters:')
# pprint.pprint(param_Dict)

'''
JSON Parameters
sim = 'TNG300-1' # 'TNG300' or 'TNG100' for boxsize, '-1', '-2' for resolution
pType = 'gas' # particle type; 'gas' or 'DM' or 'Stars'
snapshot = 99 # Redshift snapshot; currently only 99 (z=0) or 67 (z=0.5)
nPixels = 10000 # size of the 2D output box
'''

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
            pType (str): Particle Type. One of 'gas', 'DM', or 'Stars'
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
            map_ = self.convolveMap(map_, beamsize, pixel_size_arcmin=arcminPerPixel)

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
        """Convolve the map with a Gaussian beam.

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
    
    def makeField(self, pType, nPixels=None, projection='xy', save=False, load=True):
        """Used a histogram binning to make projected 2D fields of a given particle type from the simulation.

        Args:
            pType (str): Particle Type. One of 'gas', 'DM', or 'Stars'
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
        
        # Get all particles
        particles = self.loadSubsets(pType)
        coordinates = particles['Coordinates'] # kpc/h
        masses = particles['Masses'] # Msun/h
        
        # if pType =='gas':
        #     particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['Masses','Coordinates'])
        # elif pType == 'DM':
        #     particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['ParticleIDs','Coordinates'])
        # elif pType == 'Stars':
        #     particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['Masses','Coordinates'])
        
        # if pType == 'gas':
        #     coordinates = particles['Coordinates']
        #     masses= particles['Masses']
        # elif pType == 'DM':
        #     coordinates = particles['Coordinates']
        #     IDs = particles['ParticleIDs']
        #     DM_mass = header['MassTable'][1]
        # elif pType == 'Stars':
        #     coordinates = particles['Coordinates']
        #     masses= particles['Masses']
        # else:
        #     raise NotImplementedError('Particle Type not implemented')
        
        if projection == 'xy':
            coordinates = coordinates[:, :2]  # Take x and y coordinates
        elif projection == 'xz':
            coordinates = coordinates[:, [0, 2]] # Take x and z coordinates
        elif projection == 'yz':
            coordinates = coordinates[:, 1:] # Take y and z coordinates
        else:
            raise NotImplementedError('Projection type not implemented: ' + projection)
        # xx = coordinates[:,0]
        # yy = coordinates[:,1]
        
        gridSize = [nPixels, nPixels]
        minMax = [0, self.header['BoxSize']]
        # # Debug
        # print(f"Making field for {pType} at z={self.z}, projection={projection}, "
        #       f"nPixels={nPixels}, gridSize={gridSize}, minMax={minMax}")
        # print("Coordinates shape:", coordinates.shape)
        t0 = time.time()
        result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], masses, 
                                     'sum', bins=gridSize, range=[minMax, minMax]) # type: ignore
        field = result.statistic
        
        print('Binned statistic time:', time.time() - t0)
        if save:
            if self.simType == 'IllustrisTNG':
                saveName = (self.sim + '_' + str(self.snapshot) + '_' + 
                            pType + '_' + str(nPixels) + '_' + projection)
                np.save(f'/pscratch/sd/r/rhliu/simulations/{self.simType}/products/2D/{saveName}.npy', field)
            elif self.simType == 'SIMBA':
                saveName = (self.sim + '_' + self.feedback + '_' + str(self.snapshot) + '_' +  # type: ignore
                            pType + '_' + str(nPixels) + '_' + projection)
                np.save(f'/pscratch/sd/r/rhliu/simulations/{self.simType}/products/2D/{saveName}.npy', field)
        # if pType == 'gas':
        #     result = binned_statistic_2d(xx, yy, masses, 'sum', bins=gridSize, range=[minMax, minMax])
        #     field = result.statistic
        # elif pType == 'DM':
        #     result = binned_statistic_2d(xx, yy, IDs, 'count', bins=gridSize, range=[minMax, minMax])
        #     field = result.statistic * DM_mass
        # elif pType == 'Stars':
        #     result = binned_statistic_2d(xx, yy, masses, 'sum', bins=gridSize, range=[minMax, minMax])
        #     field = result.statistic

        return field
    
    def stackMap(self, pType, filterType='cumulative', minRadius=0.5, maxRadius=6, numRadii=11,
                 z=None, projection='xy', save=True, load=True, radDistance=1.0, pixelSize=0.5, 
                 halo_mass_avg=10**(13.22), halo_mass_upper=None):
        """Stack the map of a given particle type.

        Args:
            pType (str): Particle type to stack.
            filterType (str, optional): Type of filter to apply. Defaults to 'cumulative'.
            minRadius (float, optional): Minimum radius for stacking. Defaults to 0.2.
            maxRadius (int, optional): Maximum radius for stacking. Defaults to 9.
            numRadii: 
            z (float, optional): Redshift of the snapshot. Defaults to None, in which case self.z is used.
            projection (str, optional): Direction of the field projection. Currently only 'xy' is implemented. Defaults to 'xy'.
            nPixels (int, optional): Number of pixels in each direction of the 2D Field. Defaults to None.
            save (bool, optional): If True, saves the stacked map to a file. Defaults to True.
            load (bool, optional): If True, loads the stacked map from a file if it exists. Defaults to True.
            radDistance (float, optional): Radial distance units for stacking. Defaults to 1 arcmin. 
                Note there is no None option here as in stackField.
            pixelSize (float, optional): Size of each pixel in arcminutes. Defaults to 0.5.
            halo_mass_avg (float, optional): Average halo mass for selecting halos. Defaults to 10**(13.22).
            halo_mass_upper (float, optional): Upper mass bound for selecting halos. Defaults to None.
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
        halo_mask = SimulationStacker.select_massive_halos(haloMass, halo_mass_avg, halo_mass_upper)
        print('Number of halos selected:', halo_mask.shape[0])
        
        RadPixel = radDistance / arcminPerPixel # Convert to pixels
        
        # Fix the stacking function:
        # filterFunc = SimulationStacker.total_mass
        if filterType == 'cumulative':
            filterFunc = SimulationStacker.total_mass
        elif filterType == 'CAP':
            filterFunc = SimulationStacker.CAP
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

        
        return 
        
    def stackField(self, pType, filterType='cumulative', minRadius=0.1, maxRadius=4.5, numRadii=25,
                   projection='xy', nPixels=None, save=True, load=True, radDistance=1000):
        """Do stacking on the computed field.

        Args:
            pType (str): Particle Type. One of 'gas', 'DM', or 'Stars'
            filterType (str, optional): Stacked Filter Types. One of ['cumulative', 'CAP', 'DSigma']. Defaults to 'CumulativeMass'.
            minRadius (float, optional): Minimum radius in kpc/h for the stacking. Defaults to 0.2.
            maxRadius (float, optional): Maximum radius in kpc/h for the stacking. Defaults to 9.
            projection (str, optional): Direction of the field projection. Currently only 'xy' is implemented. Defaults to 'xy'.
            nPixel (int, optional): Number of pixels in each direction of the 2D Field. Defaults to self.nPixels.
            save (bool, optional): If True, saves the stacked field to a file. Defaults to True.
            load (bool, optional): If True, loads the stacked field from a file if it exists. Defaults to True.
            radDistance (float, optional): Radial distance units for stacking. Defaults to 1000 kpc/h (so converts to 1 Mpc/h). 
                If None, uses the mean halo radius from the halo catalog.
            

        Returns:
            _type_: _description_
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
        
        mass_min, mass_max, _ = self.halo_ind(2)
        
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
        filterFunc = SimulationStacker.total_mass
        # if filterType == 'cumulative':
        #     filterFunc = SimulationStacker.total_mass
        # elif filterType == 'CAP':
        #     filterFunc = SimulationStacker.CAP
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
            cap_profiles = SimulationStacker.CAP_from_mass(radii_CAP, radii, profiles.mean(axis=1))
            return radii_CAP, cap_profiles
        else:
            raise NotImplementedError('Filter Type not implemented: ' + filterType)

    # def loadField(self, pType, nPixels=None, projection='xy'):
    #     """Load a precomputed field from file for a given particle type.

    #     Args:
    #         pType (str): Particle Type. One of 'gas', 'DM', or 'Stars'.
    #         nPixels (int, optional): Number of pixels in each direction of the 2D Field. Defaults to self.nPixels.
    #         projection (str, optional): Direction of the field projection. Defaults to 'xy'.

    #     Returns:
    #         np.ndarray: 2D numpy array of the field for the given particle type.
    #     """
    #     if nPixels is None:
    #         nPixels = self.nPixels
        
    #     try:    
    #         if self.simType == 'IllustrisTNG':
    #             saveName = self.sim + '_' + str(self.snapshot) + '_' + \
    #                 pType + '_' + str(nPixels) + '_' + projection
    #             field = np.load(f'/pscratch/sd/r/rhliu/simulations/{self.simType}/products/2D/{saveName}.npy')
    #         elif self.simType == 'SIMBA':
    #             saveName = self.sim + '_' + self.feedback + '_' + str(self.snapshot) + '_' + \
    #                 pType + '_' + str(nPixels) + '_' + projection
    #             field = np.load(f'/pscratch/sd/r/rhliu/simulations/{self.simType}/products/2D/{saveName}.npy')
    #     except FileNotFoundError:
    #         raise ValueError(f"Field for file '{saveName}' not found. Please compute it first.")
        
    #     return field
    
    def loadData(self, pType, nPixels=None, projection='xy', type='field'):
        """Load a precomputed field or map from file for a given particle type.

        Args:
            pType (str): Particle Type. One of 'gas', 'DM', or 'Stars'.
            nPixels (int, optional): Number of pixels in each direction of the 2D Field. Defaults to self.nPixels.
            projection (str, optional): Direction of the field projection. Defaults to 'xy'.
            type (str, optional): Type of data to load ('field' or 'map'). Defaults to 'field'.

        Returns:
            np.ndarray: 2D numpy array of the field or map for the given particle type.
        """
        if nPixels is None:
            nPixels = self.nPixels

        suffix = '_map' if type == 'map' else ''
        try:
            if self.simType == 'IllustrisTNG':
                saveName = self.sim + '_' + str(self.snapshot) + '_' + \
                    pType + '_' + str(nPixels) + '_' + projection + suffix
                data = np.load(f'/pscratch/sd/r/rhliu/simulations/{self.simType}/products/2D/{saveName}.npy')
            elif self.simType == 'SIMBA':
                saveName = (self.sim + '_' + self.feedback + '_' + str(self.snapshot) + '_' +  # type: ignore
                            pType + '_' + str(nPixels) + '_' + projection + suffix )
                data = np.load(f'/pscratch/sd/r/rhliu/simulations/{self.simType}/products/2D/{saveName}.npy')
        except FileNotFoundError:
            raise ValueError(f"Data for file '{saveName}' not found. Please compute it first.")

        return data

    # Filter Functions:

    @staticmethod
    def total_mass(mass_grid, r_grid, r):
        """ Cumulative Mass Filter

        Args:
            mass_grid (np.ndarray): 2D array of mass values in the grid.
            r_grid (np.ndarray): 2D array of radial distances from the centre of the mass grid. Same shape as mass_grid.
            r (float): radius at which to compute the cumulative mass.

        Returns:
            float: cumulative mass
        """
       
        mass_tot = np.sum(mass_grid[r_grid<r])
        return mass_tot        

    @staticmethod
    def CAP(mass_grid, r_grid, r):
        """Compensated Aperture Photometry (CAP) Filter, see papers on kSZ/tSZ stacking


        Args:
            mass_grid (np.ndarray): 2D array of mass values in the grid.
            r_grid (np.ndarray): 2D array of radial distances from the centre of the mass grid. Same shape as mass_grid.
            r (float): radius at which to compute the cumulative mass.

        Returns:
            float: cumulative mass
        """

        r1 = r * np.sqrt(2.)
        inDisk = 1.*(r_grid <= r)
        inRing = 1.*(r_grid > r)*(r_grid <= r1)
        inRing *= np.sum(inDisk) / np.sum(inRing) # Normalize the ring
        filterW = inDisk - inRing

        filtMap = np.sum(filterW * mass_grid)
        return filtMap

    @staticmethod
    def delta_sigma(mass_grid, r_grid, r, dr=0.1):
        """Delta Sigma Filter, note that the amplitude of this filter is not necessarily
        correct

        Args:
            mass_grid (np.ndarray): 2D array of mass values in the grid.
            r_grid (np.ndarray): 2D array of radial distances from the centre of the mass grid. Same shape as mass_grid.
            r (float): radius at which to compute the Delta Sigma profile.
            dr (float, optional): area at which to aveage the surface mass density. Defaults to 0.1.

        Returns:
            np.ndarray: The computed Delta Sigma profile.
        """

        mean_sigma = np.sum(mass_grid[r_grid<r]) / (np.pi*r**2)
        # mean_sigma = np.mean(mass_grid[r_grid<r]) 

        r_mask = np.logical_and((r_grid >= r), (r_grid < r+dr))
        sigma_value = np.sum(mass_grid[r_mask]) / (2*np.pi*r*dr)
        # sigma_value = np.mean(mass_grid[r_mask])

        return mean_sigma - sigma_value
    
    @staticmethod
    def CAP_from_mass(r, radii_2D, M_2D, k=3):
        """Compute the analytic variant Compensated Aperture Photometry (CAP) profile from the mass distribution.
        CAP(r) = 2 * M(r) - M(sqrt(2) * r)

        Args:
            r (float): The radius at which to compute the profile.
            radii_2D (np.ndarray): 1D array of radial distances. (This is the radial distance of the interpolated mass profile.)
            M_2D (np.ndarray): 1D or 2D array of mass values.
            k (int, optional): The order of the spline interpolation. Defaults to 3.

        Raises:
            ValueError: If the input arrays are not compatible.

        Returns:
            np.ndarray: The computed CAP profile. 1D or 2D array depending on the input M_2D.
        """
        
        r = np.atleast_1d(r)
        
        if M_2D.ndim == 1:
            M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D, k=k)
            return 2 * M_interp(r) - M_interp(np.sqrt(2) * r) # type: ignore

        elif M_2D.ndim == 2:
            result = []
            for i in range(M_2D.shape[1]):
                M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D[:, i], k=k)
                cap = 2 * M_interp(r) - M_interp(np.sqrt(2) * r) # type: ignore
                result.append(cap)
            return np.stack(result, axis=-1)  # shape (n, l)

        else:
            raise ValueError("M_2D must be either a 1D or 2D array.")

    @staticmethod
    def DSigma_from_mass(r, radii_2D, M_2D, k=3):
        """Compute the analytic variant of the Delta Sigma profile from the mass distribution.
        DSigma(r) = M(r)/(pi*r^2) - dM/dr/(2*pi*r)

        Args:
            r (float): The radius at which to compute the profile.
            radii_2D (np.ndarray): 1D array of radial distances. 
                (This is the radial distance of the interpolated mass profile.)
            M_2D (np.ndarray): 1D or 2D array of mass values.
            k (int, optional): The order of the spline interpolation. Defaults to 3.

        Raises:
            ValueError: If the input arrays are not compatible.

        Returns:
            np.ndarray: The computed Delta Sigma profile.
        """
        
        r = np.atleast_1d(r)

        if M_2D.ndim == 1:
            M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D, k=k)
            dM_dr_interp = M_interp.derivative()
            return M_interp(r)/(np.pi*r**2) - dM_dr_interp(r)/(2*np.pi*r)

        elif M_2D.ndim == 2:
            result = []
            for i in range(M_2D.shape[1]):
                M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D[:, i], k=k)
                dM_dr_interp = M_interp.derivative()
                dsigma = M_interp(r)/(np.pi*r**2) - dM_dr_interp(r)/(2*np.pi*r)
                result.append(dsigma)
            return np.stack(result, axis=-1)  # shape (n, l)

        else:
            raise ValueError("M_2D must be either a 1D or 2D array.")


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
    
    def halo_ind(self, ind):
        """Simple wrapper that masks haloes into mass bins.

        Args:
            ind (int): Index of the mass bin.

        Returns:
            tuple: (mass_min, mass_max, title_str) for the given mass bin.
        """
        if ind == 0:
            mass_min = 5e11 # solar masses
            mass_max = 1e12 # solar masses
            title_str = r'$5\times 10^{11} M_\odot < M_{\rm halo} < 10^{12} M_\odot$, '
        elif ind == 1:
            mass_min = 1e12 # solar masses
            mass_max = 1e13 # solar masses
            title_str = r'$1\times 10^{12} M_\odot < M_{\rm halo} < 10^{13} M_\odot$, '
        elif ind == 2:
            mass_min = 1e13 # solar masses
            # mass_max = 1e14 # solar masses
            # title_str = r'$1\times 10^{13} M_\odot < M_{\rm halo} < 10^{14} M_\odot$, '
            mass_max = 1e19 # solar masses /h
            title_str = r'$1\times 10^{13} M_\odot < M_{\rm halo} < 10^{19} M_\odot$, '
        # elif ind == 3:
        #     mass_min = 1e14 # solar masses
        #     mass_max = 1e19 # solar masses
        #     title_str = r'$M_{\rm halo} > 10^{14} M_\odot$, '
        else:
            print('Wrong ind')
        return mass_min, mass_max, title_str
    
    import numpy as np

    @staticmethod
    def select_massive_halos(halo_masses, target_average_mass, upper_mass_bound=None):
        """
        Returns a boolean mask selecting the most massive halos such that the average mass
        of the selected halos is at least the target average mass.

        Parameters
        ----------
        halo_masses : np.ndarray
            Array of halo masses (can be in any consistent units).
        target_average_mass : float
            The minimum average mass desired for the selected halos.
        upper_mass_bound : float, optional
            If provided, only consider halos with mass ≤ upper_mass_bound.

        Returns
        -------
        mask : np.ndarray
            Boolean mask array of the same shape as halo_masses, selecting halos
            that meet the criteria.
        """

        halo_masses = np.asarray(halo_masses)

        # Apply upper mass bound if given
        if upper_mass_bound is not None:
            valid_mask = halo_masses <= upper_mass_bound
            filtered_masses = halo_masses[valid_mask]
        else:
            valid_mask = np.ones_like(halo_masses, dtype=bool)
            filtered_masses = halo_masses

        # Sort halo masses in descending order
        sorted_indices = np.argsort(filtered_masses)[::-1]
        sorted_masses = filtered_masses[sorted_indices]

        # Cumulative average
        cumulative_sum = np.cumsum(sorted_masses)
        counts = np.arange(1, len(sorted_masses) + 1)
        cumulative_avg = cumulative_sum / counts

        # Find the cutoff index where the average drops below the target
        idx = np.searchsorted(cumulative_avg[::-1], target_average_mass, side='right')
        if idx == 0:
            # No subset meets the target average
            # return np.zeros_like(halo_masses, dtype=bool)
            raise ValueError("No subset of halos meets the target average mass.")

        cutoff = len(sorted_masses) - idx
        selected_indices = sorted_indices[:cutoff]

        # Build final mask
        # final_mask = np.zeros_like(halo_masses, dtype=bool)
        # final_mask[np.where(valid_mask)[0][selected_indices]] = True
        final_mask = np.where(valid_mask)[0][selected_indices]

        return final_mask

    @staticmethod
    def format_string_sci(num):
        """
        Converts a number to scientific notation with up to 2 decimal places,
        removing trailing zeroes after the decimal point.

        Parameters
        ----------
        num : float or int
            The number to convert.

        Returns
        -------
        str
            Scientific notation string (e.g., '5e12', '5.2e13', '5.12e12').
        """
        base, exponent = f"{num:.2e}".split('e')
        base = base.rstrip('0').rstrip('.')  # Remove trailing zeros and decimal if needed
        return f"{base}e{int(exponent)}"

    def snapPath(self, simType):
        """Get the snapshot path for the given simulation type.

        Args:
            simType (str): The type of simulation (e.g., 'IllustrisTNG', 'SIMBA').

        Returns:
            str: The path to the snapshot file.
        """
        if simType == 'IllustrisTNG':
            snapPath = il.snapshot.snapPath(self.simPath, self.snapshot)
        elif simType == 'SIMBA':
            snapPath = self.simPath + 'snapshots/snap_' + self.sim + '_' + str(self.snapshot) + '.hdf5'
        return snapPath

    def loadHalos(self, simType):
        """Load halo data for the specified simulation type.

        Args:
            simType (str): The type of simulation (e.g., 'IllustrisTNG', 'SIMBA').

        Returns:
            dict: A dictionary containing halo properties (e.g., mass, position, radius).
        """
        
        if simType == 'IllustrisTNG':
            haloes = {}
            haloes_cat = il.groupcat.loadHalos(self.simPath, self.snapshot)
            # haloes['GroupMass'] = haloes_cat['GroupMass'] * 1e10 * self.header['HubbleParam'] # Convert to solar masses
            haloes['GroupMass'] = haloes_cat['GroupMass'] * 1e10 # Convert to Msun/h
            haloes['GroupPos'] = haloes_cat['GroupPos']
            haloes['GroupRad'] = haloes_cat['Group_R_TopHat200']
            
        elif simType == 'SIMBA':
            haloPath = self.simPath + 'catalogs/' +  self.sim + '_' + str(self.snapshot) + '.hdf5'
            haloes = {}
            with h5py.File(haloPath, 'r') as f:
                # Print all top-level groups/datasets
                # print("Keys:")
                # print(f['halo_data']['dicts'].keys())
                haloes['GroupPos'] = f['halo_data']['pos'][:] * self.header['HubbleParam'] # kpc/h # type: ignore
                # haloes['GroupMass'] = f['halo_data']['dicts']['masses.total'][:] # SIMBA already in solar masses (I think)
                haloes['GroupMass'] = f['halo_data']['dicts']['masses.total'][:] * self.header['HubbleParam'] # Convert to Msun/h # type: ignore
                # haloes['GroupMass'] = f['halo_data']['dicts']['virial_quantities.m200c'][:]
                haloes['GroupRad'] = f['halo_data']['dicts']['virial_quantities.r200c'][:] * self.header['HubbleParam'] # kpc/h # type: ignore
        return haloes
    
    def loadSubsets(self, pType):
        """Load particle subsets for the specified particle type.

        Args:
            pType (str): The type of particles to load (e.g., 'gas', 'DM', 'Stars').

        Raises:
            NotImplementedError: If the particle type is not implemented.

        Returns:
            dict: A dictionary containing the particle properties.
        """
        
        if self.simType == 'IllustrisTNG':
            if pType =='gas':
                particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['Masses','Coordinates'])
            elif pType == 'DM':
                particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['ParticleIDs','Coordinates'])
                particles['Masses'] = self.header['MassTable'][1] * np.ones_like(particles['ParticleIDs'])  # DM mass
            elif pType == 'Stars':
                particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['Masses','Coordinates'])
            else:
                raise NotImplementedError('Particle Type not implemented')
                                        
        elif self.simType == 'SIMBA':
            if pType == 'gas':
                pTypeval = 'PartType0'
            elif pType == 'DM':
                pTypeval = 'PartType1'
            elif pType == 'Stars':
                pTypeval = 'PartType4'
            else:
                raise NotImplementedError('Particle Type not implemented')
            
            keys = ['Coordinates', 'Masses']
            snapPath = self.simPath + 'snapshots/snap_' + self.sim + '_' + str(self.snapshot) + '.hdf5'
            particles = {}
            with h5py.File(snapPath, 'r') as f:
                # Print all top-level groups/datasets
                # print("Keys:")
                # print(list(f.keys()))
                # particles = f['PartType0']
                header = dict(f['Header'].attrs.items())
                for key in keys:
                    particles[key] = f[pTypeval][key][:] # type: ignore
            
        particles['Masses'] = particles['Masses'] * 1e10 / self.header['HubbleParam'] # Convert masses to Msun/h
        return particles
            

            


if __name__ == "__main__":
    pass