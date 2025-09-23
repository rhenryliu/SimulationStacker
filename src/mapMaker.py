# sys.path.append('../../illustrisPython/')
import illustris_python as il 

from tools import numba_tsc_3D, hist2d_numba_seq
# from stacker import SimulationStacker
from utils import fft_smoothed_map
from halos import select_massive_halos, halo_ind
from filters import total_mass, delta_sigma, CAP, CAP_from_mass, DSigma_from_mass
from loadIO import snap_path, load_halos, load_subsets, load_subset, load_data, save_data
from astropy.cosmology import FlatLambdaCDM, Planck18
import astropy.units as u
import numpy as np
import glob
from scipy.stats import binned_statistic_2d
import time

def compute_cosmological_parameters(header, z, cosmology=Planck18):
    """Compute cosmological parameters for angular distance calculations.

    Args:
        header (dict): Header information from the simulation snapshot.
        z (float): Redshift of the snapshot.

    Returns:
        cosmo (FlatLambdaCDM): Cosmology object for angular distance calculations.
        dA (float): Angular diameter distance at redshift z. Units: kpc/h
        theta_arcmin (float): Angular size of the simulation box in arcminutes.
    """
    
    # cosmo = FlatLambdaCDM(H0=100 * header['HubbleParam'], Om0=header['Omega0'], Tcmb0=2.7255 * u.K)
    dA = cosmology.angular_diameter_distance(z).to(u.kpc).value
    dA *= header['HubbleParam']  # Convert to kpc/h
    theta_arcmin = np.degrees(header['BoxSize'] / dA) * 60
    return cosmology, dA, theta_arcmin

def calculate_pixel_parameters(theta_arcmin, pixelSize):
    """Calculate pixel grid parameters. This is used for creating maps with pixels of a given size.

    Args:
        theta_arcmin (float): Angular size of the simulation box in arcminutes.
        pixelSize (float): Size of each pixel in arcminutes.

    Returns:
        nPixels (int): Number of pixels in each direction.
        arcminPerPixel (float): Angular size of each pixel in arcminutes.
    """
    
    nPixels = np.ceil(theta_arcmin / pixelSize).astype(int)
    arcminPerPixel = theta_arcmin / nPixels
    return nPixels, arcminPerPixel

def create_basic_field(stacker, pType, nPixels, projection, save, load, 
                      weight_calculator=None, required_keys=None):
    """Create a field for any particle type with customizable weighting.
    
    Args:
        stacker: The stacker object
        pType: Particle type
        nPixels: Number of pixels
        projection: Projection direction
        save: Whether to save the field
        load: Whether to load existing field
        weight_calculator: Function to calculate weights from particles. 
                          If None, uses masses.
        required_keys: List of keys to load from particles. 
                      If None, uses ['Coordinates', 'Masses'].
    """
    if load:
        try:
            return stacker.loadData(pType, nPixels=nPixels, projection=projection, type='field')
        except ValueError as e:
            print(e)
            print("Computing the field instead...")
    
    # Set default keys if not provided
    if required_keys is None:
        required_keys = ['Coordinates', 'Masses']
    
    # Common field creation logic
    folderPath = stacker.snapPath(stacker.simType, pathOnly=True)
    if stacker.simType == 'IllustrisTNG':
        snaps = glob.glob(folderPath + 'snap_*.hdf5')
    elif stacker.simType == 'SIMBA':
        snaps = glob.glob(folderPath)
    
    gridSize = [nPixels, nPixels]
    minMax = [0, stacker.header['BoxSize']]
    field_total = np.zeros(gridSize)
    
    t0 = time.time()
    for i, snap in enumerate(snaps):
        particles = stacker.loadSubset(pType, snapPath=snap, keys=required_keys)
        coordinates = particles['Coordinates']
        
        # Calculate weights using provided function or default to masses
        if weight_calculator is not None:
            weights = weight_calculator(particles, stacker)
        else:
            weights = particles['Masses'] * 1e10 / stacker.header['HubbleParam']
        
        # Handle projection
        coordinates = get_projected_coordinates(coordinates, projection)
        
        # Bin the data
        result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], weights, 
                                    'sum', bins=gridSize, range=[minMax, minMax]) # type: ignore
        field_total += result.statistic
        
        if i % 10 == 0:
            print(f'Processed {i} snapshots, time elapsed: {time.time() - t0:.2f} seconds')
    
    if save:
        save_field_data(stacker, pType, nPixels, projection, field_total)
    
    return field_total

def create_sz_field(stacker, pType, nPixels, projection, save, load):
    """Create SZ-specific fields (tSZ, kSZ, tau)

    Args:
        stacker (SZStacker): SZStacker object for loading and saving data.
        pType (str): Particle type to create the field for. Either 'tSZ', 'kSZ', or 'tau'.
        nPixels (int): Number of pixels of the map in each direction. The created field will be square.
        projection (str): Projection direction (One of 'xy', 'xz', 'yz').
        save (bool): Whether to save the field data.
        load (bool): Whether to load existing field data.

    Returns:
        2D np.ndarray: The created SZ field. 2D array of shape (nPixels, nPixels).
    """
    
    # Define the required keys for SZ calculations
    sz_keys = ['Coordinates', 'Masses', 'ElectronAbundance', 
               'InternalEnergy', 'Density', 'Velocities']
    
    # Create the weight calculator function for SZ fields
    def sz_weight_calculator(particles, stacker_obj):
        # SZ field constants
        sz_constants = get_sz_constants()
        
        # Get simulation parameters
        h = stacker_obj.header['HubbleParam']
        unit_mass = 1.e10 * (sz_constants['solar_mass'] / h)
        unit_dens = 1.e10 * (sz_constants['solar_mass'] / h) / (sz_constants['kpc_to_cm'] / h)**3
        unit_vol = (sz_constants['kpc_to_cm'] / h)**3
        
        z = stacker_obj.z
        a = 1./(1 + z)
        
        # Calculate SZ quantities
        sz_quantities = calculate_sz_quantities(particles, sz_constants, unit_dens, unit_vol, a)
        
        # Return appropriate weights based on pType
        return get_sz_weights(pType, sz_quantities)
    
    # Use the generalized create_basic_field function
    return create_basic_field(stacker, pType, nPixels, projection, save, load,
                             weight_calculator=sz_weight_calculator,
                             required_keys=sz_keys)

def get_projected_coordinates(coordinates, projection):
    """Helper function for getting 2D coordinates for specified projection.

    Args:
        coordinates (np.ndarray): 3D coordinates of particles.
        projection (str): Projection direction ('xy', 'xz', or 'yz').

    Raises:
        ValueError: If projection type is not one of 'xy', 'xz' or 'yz'.

    Returns:
        np.ndarray: 2D projected coordinates.
    """
    
    if projection == 'xy':
        return coordinates[:, :2]
    elif projection == 'xz':
        return coordinates[:, [0, 2]]
    elif projection == 'yz':
        return coordinates[:, 1:]
    else:
        raise ValueError("Projection type not one of 'xy', 'xz' or 'yz': " + projection)

def get_sz_constants():
    """Return dictionary of SZ-related physical constants.

    Returns:
        dict: Dictionary of SZ-related physical constants.
    """
    # Physical constants in cgs units
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
    
    return {
        'gamma': gamma, # Adiabatic index
        'k_B': k_B, # Boltzmann constant in cgs
        'm_p': m_p, # Proton mass in grams
        'unit_c': unit_c, # Velocity unit in cm/s
        'X_H': X_H, # Hydrogen mass fraction
        'sigma_T': sigma_T, # Thomson cross-section in cm^2
        'm_e': m_e, # Electron mass in grams
        'c': c, # Speed of light in cm/s
        'kpc_to_cm': kpc_to_cm, # kpc to cm conversion
        'solar_mass': solar_mass # Solar mass in grams
    }

def calculate_sz_quantities(particles, constants, unit_dens, unit_vol, a):
    """Calculate SZ-related quantities from particle properties.

    Args:
        particles (dict): Dictionary containing particle properties.
        constants (dict): Dictionary containing physical constants.
        unit_dens (float): Density unit conversion factor.
        unit_vol (float): Volume unit conversion factor.
        a (float): Scale factor.

    Returns:
        dict: Dictionary containing calculated SZ-related quantities.
    """

    # Extract particle properties
    Co = particles['Coordinates']
    EA = particles['ElectronAbundance']
    IE = particles['InternalEnergy']
    D = particles['Density']
    M = particles['Masses']
    V = particles['Velocities']
    
    # Calculate derived quantities
    dV = M/D
    D *= unit_dens
    
    # Obtain electron temperature, electron density and velocity.
    Te = ((constants['gamma'] - 1.) * IE / constants['k_B'] * 
          4 * constants['m_p'] / (1 + 3 * constants['X_H'] + 4 * constants['X_H'] * EA) * 
          constants['unit_c']) # K
    ne = EA * constants['X_H'] * D / constants['m_p'] # ccm^-3 # True for TNG and mixed for MTNG because of unit difference
    Ve = V * np.sqrt(a) # km/s
    
    # Calculate SZ signals
    const = constants['k_B'] * constants['sigma_T'] / (constants['m_e'] * constants['c']**2) # const for y compton computation.
    
    return {
        'dY': const * (ne * Te * dV) * unit_vol,
        'b': constants['sigma_T'] * (ne[:, None] * (Ve/constants['c']) * dV[:, None]) * unit_vol,
        'tau': constants['sigma_T'] * (ne * dV) * unit_vol
    }

def get_sz_weights(pType, sz_quantities):
    """Get appropriate weights for different SZ field types.

    Args:
        pType (str): The type of SZ field ('tSZ', 'kSZ', or 'tau').
        sz_quantities (dict): Dictionary containing SZ-related quantities.

    Raises:
        ValueError: If the particle type is not recognized.

    Returns:
        np.ndarray: The weights for the specified SZ field type.
    """
    if pType == 'tSZ':
        return sz_quantities['dY']
    elif pType == 'kSZ':
        # For kSZ, we need to handle the velocity component properly
        # Assuming we want the line-of-sight component (z-direction for 'xy' projection)
        return sz_quantities['b'][:, 2]  # Take z-component for xy projection
    elif pType == 'tau':
        return sz_quantities['tau']
    else:
        raise ValueError('Particle type not recognized: ' + pType)

def save_field_data(stacker, pType, nPixels, projection, field_data):
    """Save field data to appropriate location.

    Args:
        stacker (SimulationStacker): The stacker instance.
        pType (str): The type of particle ('tSZ', 'kSZ', or 'tau').
        nPixels (int): The number of pixels in the map.
        projection (str): The projection direction ('xy', 'yz', or 'xz').
        field_data (np.ndarray): The field data to save.
    """
    
    if stacker.simType == 'IllustrisTNG':
        saveName = (stacker.sim + '_' + str(stacker.snapshot) + '_' + 
                    pType + '_' + str(nPixels) + '_' + projection)
        np.save(f'/pscratch/sd/r/rhliu/simulations/{stacker.simType}/products/2D/{saveName}.npy', field_data)
    elif stacker.simType == 'SIMBA':
        saveName = (stacker.sim + '_' + stacker.feedback + '_' + str(stacker.snapshot) + '_' +
                    pType + '_' + str(nPixels) + '_' + projection)
        np.save(f'/pscratch/sd/r/rhliu/simulations/{stacker.simType}/products/2D/{saveName}.npy', field_data)

def create_field(stacker, pType, nPixels, projection, save, load):
    """Wrapper function to create the appropriate field type.

    Args:
        stacker (SimulationStacker): The stacker instance.
        pType (str): The type of particle ('tSZ', 'kSZ', or 'tau').
        nPixels (int): The number of pixels in the map.
        projection (str): The projection direction ('xy', 'yz', or 'xz').
        save (bool): Whether to save the field data.
        load (bool): Whether to load the field data.

    Returns:
        np.ndarray: The created field data.
    """
    
    sz_types = ['tSZ', 'kSZ', 'tau']
    
    if pType in sz_types:
        return create_sz_field(stacker, pType, nPixels, projection, save, load)
    else:
        return create_basic_field(stacker, pType, nPixels, projection, save, load)


