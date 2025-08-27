import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def total_mass(mass_grid, r_grid, r):
    return float(np.sum(mass_grid[r_grid < r]))

def CAP(mass_grid, r_grid, r):
    r1 = r * np.sqrt(2.0)
    inDisk = 1.0 * (r_grid <= r)
    inRing = 1.0 * (r_grid > r) * (r_grid <= r1)
    inRing *= np.sum(inDisk) / np.sum(inRing)
    return float(np.sum((inDisk - inRing) * mass_grid))

def delta_sigma(mass_grid, r_grid, r, dr=0.1):
    mean_sigma = np.sum(mass_grid[r_grid < r]) / (np.pi * r**2)
    r_mask = np.logical_and((r_grid >= r), (r_grid < r + dr))
    sigma_value = np.sum(mass_grid[r_mask]) / (2 * np.pi * r * dr)
    return float(mean_sigma - sigma_value)

def CAP_from_mass(r, radii_2D, M_2D, k=3):
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