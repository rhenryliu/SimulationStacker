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
# from stacker import SimulationStacker

# Filter Functions:

def fft_smoothed_map(D, fwhm_arcmin, pixel_size_arcmin):
    """
    Smooth a 2D map D with a Gaussian beam of FWHM (arcmin) using FFTs.
        Periodic boundary conditions are assumed (same as gaussian_filter(mode='wrap')).


    Args:
        D (np.ndarray): 2D map to smooth.
        fwhm_arcmin (float): Full width at half maximum of the Gaussian beam (in arcminutes).
        pixel_size_arcmin (float): Size of each pixel (in arcminutes).

    Returns:
        np.ndarray: Smoothed 2D map.

    TODO: 
        Test to make sure that this works as expected when compared to gaussian_smoothed_map
    """
    
    D = np.asarray(D, dtype=np.float32)
    n0, n1 = D.shape
    assert n0 == n1, "Map must be square for this implementation."
    n = n0

    # Build flat-sky multipole grid l_x, l_y (radian^-1)
    #TODO: this should be the physical rather than angular variant, incorrect
    theta_pix_rad = np.deg2rad(pixel_size_arcmin / 60.0) 
    lx = np.fft.fftfreq(n, d=theta_pix_rad) * 2.0 * np.pi
    ly = lx
    LX, LY = np.meshgrid(lx, ly, indexing='xy')
    ellsq = LX**2 + LY**2

    # Beam transfer function B_ell
    B_ell = gauss_beam(ellsq, fwhm_arcmin)  # uses radians internally

    # FFT, multiply, inverse FFT
    dfour = scipy.fft.fftn(D, workers=-1)
    dksmo = dfour * B_ell
    drsmo = np.real(scipy.fft.ifftn(dksmo, workers=-1))
    return drsmo
    
def gauss_beam(ellsq, fwhm):
    """
    Gaussian beam of size fwhm
    
    Args:
        ellsq: squared angular multipole term used in the Gaussian beam function's exponent. 
            In flat-sky this is defined as (l_x^2 + l_y^2), where l_x and l_y are the Fourier space frequencies in rad^-1.
        fwhm: Gaussian FWHM of the beam (in arcminutes)
        
    Returns:
        np.ndarray: The Gaussian beam transfer function.
    """
    tht_fwhm = np.deg2rad(fwhm/60.)
    return np.exp(-0.5*(tht_fwhm**2.)*(ellsq)/(8.*np.log(2.)))

def gaussian_smoothed_map(D, fwhm_arcmin, pixel_size_arcmin):
    """
    DEPRECIATED: Check below for new Convolution code
    Convolve the map with a Gaussian beam.

    Args:
        D (np.ndarray): 2D numpy array of the field for the given particle type.
        fwhm_arcmin (float): Full width at half maximum of the Gaussian beam in arcminutes.
        pixel_size_arcmin (float): Size of the pixel in arcminutes.

    Returns:
        np.ndarray: Convolved 2D numpy array.
    """
    
    sigma_pixels = fwhm_arcmin / (2.355 * pixel_size_arcmin)

    # Apply Gaussian filter
    convolved_map = gaussian_filter(D, sigma=sigma_pixels, mode='wrap')
    
    return convolved_map

def format_string_sci(num):
    """
    Converts a number to scientific notation with up to 2 decimal places,
    removing trailing zeroes after the decimal point.

    Args:
        num (float or int): The number to convert.

    Returns:
        str: Scientific notation string (e.g., '5e12', '5.2e13', '5.12e12').
    """
    base, exponent = f"{num:.2e}".split('e')
    base = base.rstrip('0').rstrip('.')  # Remove trailing zeros and decimal if needed
    return f"{base}e{int(exponent)}"