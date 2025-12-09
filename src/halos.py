import numpy as np

def halo_ind(ind):
    if ind == 0:
        return 5e11, 1e12, r'$5\times 10^{11} M_\odot < M_{\rm halo} < 10^{12} M_\odot$, '
    elif ind == 1:
        return 1e12, 1e13, r'$1\times 10^{12} M_\odot < M_{\rm halo} < 10^{13} M_\odot$, '
    elif ind == 2:
        return 1e13, 1e19, r'$1\times 10^{13} M_\odot < M_{\rm halo} < 10^{19} M_\odot$, '
    else:
        raise ValueError("Wrong ind")

def select_massive_halos(halo_masses, target_average_mass, upper_mass_bound=None):
    """Select haloes such that the average mass of the selected halos meets the target average mass.

    Args:
        halo_masses (array-like): Array of halo masses.
        target_average_mass (float): Target average mass for the selected halos.
        upper_mass_bound (float, optional): Upper bound on halo mass. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    halo_masses = np.asarray(halo_masses)
    if upper_mass_bound is not None:
        valid_mask = halo_masses <= upper_mass_bound
        filtered = halo_masses[valid_mask]
    else:
        valid_mask = np.ones_like(halo_masses, dtype=bool)
        filtered = halo_masses

    order = np.argsort(filtered)[::-1]
    sorted_m = filtered[order]
    cum_sum = np.cumsum(sorted_m)
    counts = np.arange(1, len(sorted_m) + 1)
    cum_avg = cum_sum / counts

    idx = np.searchsorted(cum_avg[::-1], target_average_mass, side='right')
    if idx == 0:
        raise ValueError("No subset of halos meets the target average mass.")
    cutoff = len(sorted_m) - idx
    selected = order[:cutoff]
    return np.where(valid_mask)[0][selected]

def select_abundance_subhalos(halo_masses, target_number, Lbox):
    """Select haloes such that the average mass of the selected halos meets the target average mass.

    Args:
        halo_masses (array-like): Array of halo masses.
        target_number (float): Target number density for the selected halos. Units: (cMpc/h)^-3
        Lbox (float): Length of the simulation box in ckpc/h

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    box_volume = (Lbox / 1e3) ** 3  # Convert ckpc/h to cMpc/h
    Ngal = int(target_number * box_volume) # Total number of galaxies desired
    
    
    idx_selected = np.argsort(halo_masses)[::-1][:Ngal]
    
    # sorted_m = halo_masses[order]
    # cum_counts = np.arange(1, len(sorted_m) + 1)
    # cum_number_density = cum_counts / box_volume

    # idx = np.searchsorted(cum_number_density, target_number, side='right')
    # if idx == 0:
    #     raise ValueError("No subset of halos meets the target number density.")
    # cutoff = idx
    # selected = order[:cutoff]
    return idx_selected