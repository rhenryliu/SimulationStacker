import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def total_mass(mass_grid, r_grid, r):
    """Calculate the total mass within a given radius.

    Args:
        mass_grid (np.ndarray): The mass distribution array.
        r_grid (np.ndarray): The radial grid corresponding to the mass distribution.
        r (float): The radius at which to calculate the total mass.

    Returns:
        float: The total mass within the given radius.
    """
    
    return float(np.sum(mass_grid[r_grid < r]))

def CAP(mass_grid, r_grid, r):
    """Calculate the Circular Averages Profile (CAP) for a given radius.

    Args:
        mass_grid (np.ndarray): The mass distribution array.
        r_grid (np.ndarray): The radial grid corresponding to the mass distribution.
        r (float): The radius at which to calculate the CAP.

    Returns:
        float: The value of the CAP at the given radius.
    """
    
    r1 = r * np.sqrt(2.0)
    inDisk = 1.0 * (r_grid <= r)
    inRing = 1.0 * (r_grid > r) * (r_grid <= r1)
    inRing *= np.sum(inDisk) / np.sum(inRing)
    return float(np.sum((inDisk - inRing) * mass_grid))

def delta_sigma(mass_grid, r_grid, r, dr=0.1):
    """Calculate the excess surface mass density (ΔΣ) for a given radius.

    Args:
        mass_grid (np.ndarray): The mass distribution array.
        r_grid (np.ndarray): The radial grid corresponding to the mass distribution.
        r (float): The radius at which to calculate ΔΣ.
        dr (float, optional): The width of the annulus. Defaults to 0.1.

    Returns:
        float: The value of ΔΣ at the given radius.
    """
        
    mean_sigma = np.sum(mass_grid[r_grid < r]) / (np.pi * r**2)
    r_mask = np.logical_and((r_grid >= r), (r_grid < r + dr))
    sigma_value = np.sum(mass_grid[r_mask]) / (2 * np.pi * r * dr)
    return float(mean_sigma - sigma_value)

def CAP_from_mass(r, radii_2D, M_2D, k=3):
    """Calculate the Circular Averages Profile (CAP) from 1D mass distribution.

    Args:
        r (float): The radius at which to calculate the CAP.
        radii_2D (np.ndarray): The radial grid corresponding to the 2D mass distribution.
        M_2D (np.ndarray): The 2D mass distribution array.
        k (int, optional): The degree of the spline interpolation. Defaults to 3.

    Raises:
        ValueError: If the dimensions of the input arrays are not compatible.

    Returns:
        float: The value of the CAP at the given radius.
    """
    
    r = np.atleast_1d(r)
    if M_2D.ndim == 1:
        M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D, k=k)
        return 2 * M_interp(r) - M_interp(np.sqrt(2) * r) # type: ignore
    elif M_2D.ndim == 2:
        result = []
        for i in range(M_2D.shape[1]):
            M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D[:, i], k=k)
            result.append(2 * M_interp(r) - M_interp(np.sqrt(2) * r)) # type: ignore
        return np.stack(result, axis=-1)
    else:
        raise ValueError("M_2D must be either a 1D or 2D array.")

def DSigma_from_mass(r, radii_2D, M_2D, k=3):
    """Calculate the excess surface mass density (ΔΣ) from 1D mass distribution.

    Args:
        r (float): The radius at which to calculate ΔΣ.
        radii_2D (np.ndarray): The radial grid corresponding to the 2D mass distribution.
        M_2D (np.ndarray): The 2D mass distribution array.
        k (int, optional): The degree of the spline interpolation. Defaults to 3.

    Raises:
        ValueError: If the dimensions of the input arrays are not compatible.

    Returns:
        float: The value of ΔΣ at the given radius.
    """
    
    r = np.atleast_1d(r)
    if M_2D.ndim == 1:
        M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D, k=k)
        dM_dr_interp = M_interp.derivative()
        return M_interp(r) / (np.pi * r**2) - dM_dr_interp(r) / (2 * np.pi * r)
    elif M_2D.ndim == 2:
        result = []
        for i in range(M_2D.shape[1]):
            M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D[:, i], k=k)
            dM_dr_interp = M_interp.derivative()
            result.append(M_interp(r) / (np.pi * r**2) - dM_dr_interp(r) / (2 * np.pi * r))
        return np.stack(result, axis=-1)
    else:
        raise ValueError("M_2D must be either a 1D or 2D array.")