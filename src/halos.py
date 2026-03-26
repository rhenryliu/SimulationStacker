import numpy as np

def halo_ind(ind):
    """Return mass bin boundaries and label string for a given bin index.

    Args:
        ind (int): Mass bin index. Must be 0, 1, or 2.

    Returns:
        tuple: A 3-tuple (mass_min, mass_max, label) where mass_min and
            mass_max are the lower and upper halo mass bounds in M☉, and
            label is a LaTeX-formatted string describing the bin.

    Raises:
        ValueError: If ind is not 0, 1, or 2.
    """
    if ind == 0:
        return 5e11, 1e12, r'$5\times 10^{11} M_\odot < M_{\rm halo} < 10^{12} M_\odot$, '
    elif ind == 1:
        return 1e12, 1e13, r'$10^{12} M_\odot < M_{\rm halo} < 10^{13} M_\odot$, '
    elif ind == 2:
        return 1e13, 1e19, r'$10^{13} M_\odot < M_{\rm halo} < 10^{19} M_\odot$, '
    elif ind == 3:
        return 1e14, 1e19, r'$10^{14} M_\odot < M_{\rm halo} < 10^{19} M_\odot$, '
    else:
        raise ValueError("Wrong ind")

def select_binned_halos(halo_masses, ind):
    """Select halos within a specified mass bin.

    Args:
        halo_masses (array-like): Array of halo masses.
        ind (int): Mass bin index. Must be 0, 1, 2, or 3.

    Returns:
        np.ndarray: Integer indices into the original halo_masses array
            identifying the selected halos, shape (N_selected,).
    """
    mass_min, mass_max, _ = halo_ind(ind)
    mask = (halo_masses > mass_min) & (halo_masses < mass_max)
    return np.where(mask)[0]

def select_massive_halos(halo_masses, target_average_mass, upper_mass_bound=None):
    """Select haloes such that the average mass of the selected halos meets the target average mass.

    Args:
        halo_masses (array-like): Array of halo masses.
        target_average_mass (float): Target average mass for the selected halos.
        upper_mass_bound (float, optional): Upper bound on halo mass. Defaults to None.

    Raises:
        ValueError: If no subset of halos satisfies the target average mass
            (e.g. if target_average_mass exceeds the highest individual mass).

    Returns:
        np.ndarray: Integer indices into the original halo_masses array
            identifying the selected halos, shape (N_selected,).
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
    """Select subhalos by abundance matching to a target number density.

    Sorts subhalos by the provided mass proxy in descending order and
    selects the top N such that N / box_volume matches the target number
    density.  The function is agnostic to the choice of mass proxy: any
    monotonic observable can be passed, e.g. total bound mass, stellar
    mass, or a DM-inclusive aperture mass.

    In practice, stellar mass (SubhaloMStar) is preferred because it is
    directly observable and avoids the tidal-stripping bias that affects
    current total subhalo masses for satellites (Reddick et al. 2013).
    For SIMBA, where CAESAR does not compute a bound DM mass per galaxy,
    stellar mass is also the most consistently defined quantity across
    both central and satellite galaxies.

    Args:
        halo_masses (array-like): Mass proxy array for all subhalos.
            Can be total bound mass, stellar mass, or any other monotonic
            observable suitable for abundance matching.  Units must be
            consistent within the array; the ranking is purely ordinal.
        target_number (float): Target number density in (cMpc/h)^-3.
        Lbox (float): Simulation box side length in ckpc/h.

    Returns:
        np.ndarray: Integer indices into halo_masses for the selected
            subhalos, sorted by decreasing mass proxy, shape (N_selected,).
    """
    box_volume = (Lbox / 1e3) ** 3  # Convert ckpc/h to cMpc/h
    Ngal = int(target_number * box_volume)  # Total number of galaxies desired
    return np.argsort(halo_masses)[::-1][:Ngal]

def select_halos(halo_masses, method, **kwargs):
    """Unified halo selection dispatcher.

    Routes to one of three selection strategies based on the method argument.

    Args:
        halo_masses (array-like): Array of halo masses.
        method (str): Selection method. One of:
            - ``'binned'``: Select halos within a fixed mass bin via
              :func:`select_binned_halos`. Required kwarg: ``ind`` (int).
            - ``'massive'``: Select the most massive halos whose cumulative
              average meets a target mass via :func:`select_massive_halos`.
              Required kwarg: ``target_average_mass`` (float).
              Optional kwarg: ``upper_mass_bound`` (float).
            - ``'abundance'``: Select halos by number density matching via
              :func:`select_abundance_subhalos`. Required kwargs:
              ``target_number`` (float, in (cMpc/h)^-3) and ``Lbox`` (float,
              in ckpc/h).
        **kwargs: Method-specific parameters as described above.

    Returns:
        np.ndarray: Integer indices into ``halo_masses`` for the selected
            halos, shape (N_selected,).

    Raises:
        ValueError: If ``method`` is not one of the recognised options.
    """
    if method == 'binned':
        return select_binned_halos(halo_masses, kwargs['ind'])
    elif method == 'massive':
        return select_massive_halos(
            halo_masses,
            kwargs['target_average_mass'],
            kwargs.get('upper_mass_bound'),
        )
    elif method == 'abundance':
        return select_abundance_subhalos(
            halo_masses,
            kwargs['target_number'],
            kwargs['Lbox'],
        )
    else:
        raise ValueError(
            f"Unknown halo selection method: '{method}'. "
            "Must be one of 'binned', 'massive', 'abundance'."
        )