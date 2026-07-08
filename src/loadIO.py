import h5py
import numpy as np
import glob
import illustris_python as il
from pathlib import Path


def snap_path(sim_path, snapshot, sim_type, sim_name=None, feedback=None, chunk_num=0, path_only=False):
    """Get the snapshot path for the given simulation type.

    Args:
        sim_path (str): Base path to the simulation.
        snapshot (int): Snapshot number.
        sim_type (str): The type of simulation (e.g., 'IllustrisTNG', 'SIMBA', 'FLAMINGO').
        sim_name (str, optional): Name of the simulation (for SIMBA).
        feedback (str, optional): Feedback type (for SIMBA).
        chunk_num (int): The chunk number for the simulation (Only used for IllustrisTNG.
            For FLAMINGO, chunks are found by globbing the folder path instead).
        path_only (bool): If True, returns only the folder path.

    Returns:
        str: The path to the snapshot file or folder. For FLAMINGO the file path is
        the *virtual* snapshot file (which stitches all chunk files plus the halo
        membership files), while the folder path is the directory holding the raw
        chunk files ``flamingo_NNNN.{i}.hdf5`` for per-chunk iteration.
    """
    if sim_type == 'IllustrisTNG':
        folder_path = sim_path + '/snapdir_' + str(snapshot).zfill(3) + '/'
        snap_path = il.snapshot.snapPath(sim_path, snapshot, chunkNum=chunk_num)
    elif sim_type == 'SIMBA':
        folder_path = sim_path + 'snapshots/'
        snap_path = folder_path + 'snap_' + sim_name + '_' + str(snapshot) + '.hdf5'
        folder_path = snap_path  # SIMBA has different file structure
    elif sim_type == 'FLAMINGO':
        snap_str = str(snapshot).zfill(4)
        base = sim_path + 'snapshots/flamingo_' + snap_str + '/'
        folder_path = base + 'swift_snapshot_' + snap_str + '/'
        snap_path = base + 'flamingo_' + snap_str + '.hdf5'  # virtual file

    if path_only:
        return folder_path
    else:
        return snap_path


def load_flamingo_header(snap_file):
    """Load a FLAMINGO virtual snapshot header, normalized to TNG-style conventions.

    FLAMINGO (SWIFT) uses comoving Mpc and 1e10 Msun with NO factors of h,
    and stores cosmology in a separate 'Cosmology' group. This function
    synthesizes the Gadget-style header dict the rest of the pipeline expects
    (BoxSize in ckpc/h, scalar attributes, 'HubbleParam'/'Omega0' keys).

    Note: the virtual file's 'NumPart_Total' overflows 32 bits, so the true
    counts (from 'NumPart_ThisFile', which covers the whole box in the virtual
    file) are stored under 'NumPart_Total' here.

    Args:
        snap_file (str): Path to the FLAMINGO virtual snapshot HDF5 file.

    Returns:
        dict: TNG-style header with keys 'BoxSize' (ckpc/h), 'HubbleParam',
        'Omega0', 'OmegaBaryon', 'OmegaLambda', 'Redshift', 'Time' (scale
        factor), 'MassTable' (all zeros; every FLAMINGO ptype has a
        per-particle mass field), 'NumPart_Total', and 'NumFilesPerSnapshot'.
    """
    with h5py.File(snap_file, 'r') as f:
        raw = dict(f['Header'].attrs.items())
        cosmo = dict(f['Cosmology'].attrs.items())

    h = float(cosmo['h'][0])
    header = {
        'BoxSize': float(raw['BoxSize'][0]) * 1000.0 * h,  # cMpc -> ckpc/h
        'HubbleParam': h,
        'Omega0': float(cosmo['Omega_m'][0]),
        'OmegaBaryon': float(cosmo['Omega_b'][0]),
        'OmegaLambda': float(cosmo['Omega_lambda'][0]),
        'Redshift': float(raw['Redshift'][0]),
        'Time': float(raw['Scale-factor'][0]),
        'MassTable': np.asarray(raw['MassTable'], dtype=np.float64),
        'NumPart_Total': raw['NumPart_ThisFile'].astype(np.int64),
        'NumFilesPerSnapshot': 1,
    }
    return header


def load_halos(sim_path, snapshot, sim_type, sim_name=None, header=None):
    """Load halo data for the specified simulation type.

    Args:
        sim_path (str): Base path to the simulation.
        snapshot (int): Snapshot number.
        sim_type (str): The type of simulation (e.g., 'IllustrisTNG', 'SIMBA').
        sim_name (str, optional): Name of the simulation (for SIMBA).
        header (dict, optional): Simulation header (required for SIMBA).

    Returns:
        dict: A dictionary containing halo properties (e.g., mass, position, radius).
    """
    if sim_type == 'IllustrisTNG':
        haloes = {}
        haloes_cat = il.groupcat.loadHalos(sim_path, snapshot)
        haloes['GroupPos'] = haloes_cat['GroupPos']
        haloes['GroupMass'] = haloes_cat['GroupMass'] * 1e10  # Convert to Msun/h
        # haloes['GroupRad'] = haloes_cat['Group_R_TopHat200']
        haloes['GroupRad'] = haloes_cat['Group_R_Mean200']
        del haloes_cat  # free memory

    elif sim_type == 'SIMBA':
        if header is None:
            raise ValueError("Header is required for SIMBA simulations")

        halo_path = sim_path + 'catalogs/' + sim_name + '_' + str(snapshot) + '.hdf5'
        haloes = {}
        # Load entire halo catalog - this is doable since catalogs are small
        haloes_cat = load_as_dict(halo_path, 'halo_data')
        haloes['GroupPos'] = haloes_cat['pos'] * header['HubbleParam']  # kpc/h
        haloes['GroupMass'] = haloes_cat['dicts']['masses.total'] * header['HubbleParam']  # Msun/h
        # GroupRad here intentionally uses r200 (mean-overdensity radius) rather than r200c (critical-overdensity radius)
        haloes['GroupRad'] = haloes_cat['dicts']['virial_quantities.r200'] * header['HubbleParam']  # kpc/h
        del haloes_cat  # free memory

    elif sim_type == 'FLAMINGO':
        if header is None:
            raise ValueError("Header is required for FLAMINGO simulations")

        h = header['HubbleParam']
        soap_path = sim_path + 'SOAP-HBT/halo_properties_' + str(snapshot).zfill(4) + '.hdf5'
        haloes = {}
        # SOAP native units: comoving Mpc and 1e10 Msun, with NO factors of h.
        # Only centrals are kept, matching the FoF-halo semantics of the TNG
        # group catalog (SOAP lists every subhalo, with spherical-overdensity
        # properties zeroed for satellites).
        # GroupRad/GroupMass use SO/200_mean to match TNG's Group_R_Mean200 convention.
        with h5py.File(soap_path, 'r') as f:
            is_central = f['InputHalos/IsCentral'][:].astype(bool)
            haloes['GroupPos'] = f['InputHalos/HaloCentre'][:][is_central] * 1000.0 * h  # ckpc/h
            haloes['GroupMass'] = (f['SO/200_mean/TotalMass'][:][is_central].astype(np.float64)
                                   * 1e10 * h)  # Msun/h
            haloes['GroupRad'] = (f['SO/200_mean/SORadius'][:][is_central].astype(np.float64)
                                  * 1000.0 * h)  # ckpc/h
        del is_central

    return haloes

def load_subhalos(sim_path, snapshot, sim_type, sim_name=None, header=None):
    """Load subhalo data for the specified simulation type.

    Args:
        sim_path (str): Base path to the simulation.
        snapshot (int): Snapshot number.
        sim_type (str): The type of simulation (e.g., 'IllustrisTNG', 'SIMBA').
        sim_name (str, optional): Name of the simulation (for SIMBA).
        header (dict, optional): Simulation header (required for SIMBA).

    Returns:
        dict: A dictionary containing subhalo properties (e.g., mass, position).
    """
    if sim_type == 'IllustrisTNG':
        subhaloes = {}
        subhaloes_cat = il.groupcat.loadSubhalos(sim_path, snapshot)
        
        subhaloes['SubhaloPos'] = subhaloes_cat['SubhaloPos']
        subhaloes['SubhaloMass'] = subhaloes_cat['SubhaloMass'] * 1e10  # Convert to Msun/h
        subhaloes['SubhaloGrNr'] = subhaloes_cat['SubhaloGrNr']
        # subhaloes['SubhaloID'] = subhaloes_cat['SubhaloID']
        subhaloes['SubhaloMStar'] = subhaloes_cat['SubhaloMassType'][:, 4] * 1e10  # Stellar mass in Msun/h
        
    elif sim_type == 'SIMBA':
        if header is None:
            raise ValueError("Header is required for SIMBA simulations")
        
        subhalo_path = sim_path + 'catalogs/' + sim_name + '_' + str(snapshot) + '.hdf5'
        subhaloes = {}
        # CAESAR identifies galaxies with a 6D Friends-of-Friends (6DFOF)
        # algorithm that runs only on baryonic particles (gas + stars + BH).
        # DM is excluded from the galaxy finder by design, so
        # galaxy_data['dicts']['masses.total'] is a purely baryonic mass
        # (~10^10–10^11 Msun) — NOT analogous to TNG's SubhaloMass, which
        # includes all gravitationally bound particles (DM + baryons).
        subhaloes_cat = load_as_dict(subhalo_path, 'galaxy_data')

        # parent_halo_index: confirmed in CAESAR docs and source as the index
        # into halo_data for each galaxy's parent FoF halo.
        parent_idx = subhaloes_cat['parent_halo_index']

        subhaloes['SubhaloPos']   = subhaloes_cat['pos'] * header['HubbleParam']  # kpc/h

        # SubhaloMass proxy for SHAM: baryonic mass + DM within a 30 kpc aperture.
        #
        # There is no CAESAR equivalent to TNG's SubhaloMass (SUBFIND total
        # bound mass including DM).  Three options were considered:
        #
        #   Option 1 — baryonic mass only (masses.total, ~10^11 Msun):
        #     Unique per galaxy; allows satellite selection; but 100× lower
        #     than the mass-cut selection scale, so SHAM selects a different
        #     population entirely.
        #
        #   Option 2 — baryonic + 30 kpc aperture DM (masses.total + masses.dm_30kpc):
        #     Still unique per galaxy; includes a local DM contribution that
        #     partially captures the galaxy's DM environment; physically
        #     closer to SUBFIND's satellite-subhalo mass than baryons alone.
        #     Currently in use (chosen as the best available approximation), while
        #     SHAM selection in stack_on_array uses SubhaloMStar (stellar mass)
        #
        #   Option 3 — parent FoF halo mass:
        #     Right scale for centrals (~10^13 Msun) but degenerate — all
        #     galaxies in the same halo share the same mass, so SHAM selects
        #     all galaxies from the few most massive halos rather than a
        #     representative mix of centrals and satellites.
        #
        # NOTE: dm_30kpc is a spherical aperture sum, not a bound-mass
        # calculation.  For deeply embedded satellites the aperture may
        # include DM belonging to the host rather than the satellite itself.
        # This is a known limitation with no better alternative in CAESAR.
        subhaloes['SubhaloMass']  = (
            subhaloes_cat['dicts']['masses.dm_30kpc'] +
            subhaloes_cat['dicts']['masses.total']
        ) * header['HubbleParam']                                                   # Msun/h

        subhaloes['SubhaloGrNr']  = parent_idx                                     # confirmed key
        subhaloes['SubhaloID']    = subhaloes_cat['GroupID']
        subhaloes['SubhaloMStar'] = subhaloes_cat['dicts']['masses.stellar'] * header['HubbleParam']  # Msun/h

    elif sim_type == 'FLAMINGO':
        if header is None:
            raise ValueError("Header is required for FLAMINGO simulations")

        h = header['HubbleParam']
        soap_path = sim_path + 'SOAP-HBT/halo_properties_' + str(snapshot).zfill(4) + '.hdf5'
        subhaloes = {}
        # All SOAP rows are kept (centrals + satellites), matching TNG's SUBFIND
        # subhalo catalog semantics. SOAP native units: comoving Mpc, 1e10 Msun, no h.
        # SubhaloMass/SubhaloMStar use BoundSubhalo (HBT gravitationally-bound
        # particles), the direct analog of TNG's SubhaloMass/SubhaloMassType[:,4].
        with h5py.File(soap_path, 'r') as f:
            is_central = f['InputHalos/IsCentral'][:].astype(bool)
            host_idx = f['SOAP/HostHaloIndex'][:]  # SOAP row of top-level parent; -1 for centrals
            subhaloes['SubhaloPos'] = f['InputHalos/HaloCentre'][:] * 1000.0 * h  # ckpc/h
            subhaloes['SubhaloMass'] = (f['BoundSubhalo/TotalMass'][:].astype(np.float64)
                                        * 1e10 * h)  # Msun/h
            subhaloes['SubhaloMStar'] = (f['BoundSubhalo/StellarMass'][:].astype(np.float64)
                                         * 1e10 * h)  # Msun/h
            subhaloes['SubhaloID'] = f['InputHalos/HaloCatalogueIndex'][:]

        # SubhaloGrNr must index into the centrals-only catalog returned by
        # load_halos (see stack_on_array's parent-mass lookup). Map each
        # subhalo's top-level parent SOAP row (itself for centrals) to its rank
        # among centrals.
        central_rank = np.cumsum(is_central) - 1  # SOAP row -> centrals-only row
        parent_row = np.where(host_idx >= 0, host_idx, np.arange(len(is_central)))
        subhaloes['SubhaloGrNr'] = central_rank[parent_row]

    return subhaloes

def _convert_flamingo_particles(particles, header):
    """Convert FLAMINGO particle fields in place to pipeline conventions.

    FLAMINGO (SWIFT) native units are comoving Mpc and 1e10 Msun with NO
    factors of h. The pipeline expects Coordinates in ckpc/h and Masses in
    1e10 Msun/h (mapMaker applies the final *1e10 for all sim types), so:
    Coordinates *= 1000*h and Masses *= h.

    Velocities (if present) are left in FLAMINGO native units: PECULIAR km/s.
    Note this differs from the Gadget convention (km*sqrt(a)/s) used by
    TNG/SIMBA — FLAMINGO-specific code must NOT apply the sqrt(a) factor.

    Args:
        particles (dict): Particle fields as read from file (modified in place).
        header (dict): Normalized simulation header (for 'HubbleParam').

    Returns:
        dict: The same dict, with converted fields.
    """
    h = header['HubbleParam']
    if 'Coordinates' in particles:
        particles['Coordinates'] = particles['Coordinates'] * (1000.0 * h)  # cMpc -> ckpc/h
    if 'Masses' in particles:
        particles['Masses'] = particles['Masses'] * h  # 1e10 Msun -> 1e10 Msun/h
    return particles


def load_subsets(sim_path, snapshot, sim_type, p_type, sim_name=None, feedback=None, header=None, keys=None):
    """Load particle subsets for the specified particle type.

    Args:
        sim_path (str): Base path to the simulation.
        snapshot (int): Snapshot number.
        sim_type (str): The type of simulation.
        p_type (str): The type of particles to load (e.g., 'gas', 'DM', 'Stars', 'tSZ', 'kSZ', 'tau').
        sim_name (str, optional): Name of the simulation (for SIMBA).
        feedback (str, optional): Feedback type (for SIMBA).
        header (dict, optional): Simulation header (required for mass conversion).
        keys (list, optional): Specific keys to load. If None, defaults are used based on particle type.

    Returns:
        dict: A dictionary containing the particle properties.
    """
    if header is None:
        raise ValueError("Header is required for mass conversion")
    
    # Handle SZ particle types by mapping to gas with appropriate fields
    if p_type in ['tSZ', 'kSZ', 'tau']:
        actual_p_type = 'gas'
        if keys is None:
            keys = ['Coordinates', 'Masses', 'ElectronAbundance', 'InternalEnergy', 'Density', 'Velocities']
    else:
        actual_p_type = p_type
        if keys is None:
            keys = ['Coordinates', 'Masses']
        
    if sim_type == 'IllustrisTNG':
        if actual_p_type == 'gas':
            if keys is None:
                particles = il.snapshot.loadSubset(sim_path, snapshot, actual_p_type, fields=['Masses', 'Coordinates'])
            else:
                particles = il.snapshot.loadSubset(sim_path, snapshot, actual_p_type, fields=keys)
        elif actual_p_type == 'DM':
            # Handle DM case where we need ParticleIDs for mass
            dm_keys = list(keys) if keys else ['Coordinates', 'Masses']
            if 'Masses' in dm_keys:
                dm_keys[dm_keys.index('Masses')] = 'ParticleIDs'
                particles = il.snapshot.loadSubset(sim_path, snapshot, actual_p_type, fields=dm_keys)
                particles['Masses'] = header['MassTable'][1] * np.ones_like(particles['ParticleIDs'])
                del particles['ParticleIDs']
            else:
                particles = il.snapshot.loadSubset(sim_path, snapshot, actual_p_type, fields=dm_keys)
        elif actual_p_type in ['Stars', 'BH']:
            particles = il.snapshot.loadSubset(sim_path, snapshot, actual_p_type, fields=keys)
        else:
            raise NotImplementedError('Particle Type not implemented')
                                    
    elif sim_type == 'SIMBA':
        if actual_p_type == 'gas':
            p_type_val = 'PartType0'
        elif actual_p_type == 'DM':
            p_type_val = 'PartType1'
        elif actual_p_type == 'Stars':
            p_type_val = 'PartType4'
        elif actual_p_type == 'BH':
            p_type_val = 'PartType5'
        else:
            raise NotImplementedError('Particle Type not implemented')
        
        snap_path = sim_path + 'snapshots/snap_' + sim_name + '_' + str(snapshot) + '.hdf5'
        particles = {}
        with h5py.File(snap_path, 'r') as f:
            for key in keys:
                particles[key] = f[p_type_val][key][:] # type: ignore

    elif sim_type == 'FLAMINGO':
        flamingo_p_type_map = {'gas': 'PartType0', 'DM': 'PartType1',
                               'Stars': 'PartType4', 'BH': 'PartType5'}
        if actual_p_type not in flamingo_p_type_map:
            raise NotImplementedError('Particle Type not implemented: ' + actual_p_type)
        p_type_val = flamingo_p_type_map[actual_p_type]

        # Read through the virtual snapshot file (stitches all 64 chunk files).
        # WARNING: full-box reads are huge (5.4e9 gas particles for L1_m9);
        # prefer per-chunk iteration via load_subset for field making.
        virtual_file = sim_path + ('snapshots/flamingo_' + str(snapshot).zfill(4)
                                   + '/flamingo_' + str(snapshot).zfill(4) + '.hdf5')
        particles = {}
        with h5py.File(virtual_file, 'r') as f:
            grp = f[p_type_val]
            for key in keys:
                # FLAMINGO black holes have no 'Masses' dataset; DynamicalMasses
                # (the gravitating mass) is the analog used for mass fields.
                read_key = 'DynamicalMasses' if (key == 'Masses' and p_type_val == 'PartType5') else key
                if read_key not in grp: # type: ignore
                    raise KeyError(f"Dataset '{read_key}' not found in {p_type_val} for FLAMINGO. "
                                   f"Note FLAMINGO has no ElectronAbundance/InternalEnergy; SZ fields "
                                   f"use ComptonYParameters/ElectronNumberDensities instead.")
                particles[key] = grp[read_key][:] # type: ignore
        _convert_flamingo_particles(particles, header)

    # particles['Masses'] = particles['Masses'] * 1e10 / header['HubbleParam']  # Convert masses to Msun/h
    return particles


def load_subset(sim_path, snapshot, sim_type, p_type, snap_path, header=None, keys=None, sim_name=None):
    """Load a subset of particles from a specific snapshot file.

    Args:
        sim_path (str): Base path to the simulation.
        snapshot (int): Snapshot number.
        sim_type (str): The type of simulation.
        p_type (str): The type of particles to load (e.g., 'gas', 'DM', 'Stars', 'tSZ', 'kSZ', 'tau').
        snap_path (str): The path to the snapshot file.
        header (dict, optional): Simulation header (required for mass conversion).
        keys (list, optional): The keys to load from the snapshot.
        sim_name (str, optional): Name of the simulation (for SIMBA).

    Returns:
        dict: A dictionary containing the particle properties.
    """
    if keys is None:
        keys = ['Coordinates', 'Masses']
    read_keys = list(keys)  # copy so we can mutate safely
    
    if header is None:
        raise ValueError("Header is required for mass conversion")
    
    # Handle SZ particle types by mapping to gas
    if p_type in ['tSZ', 'kSZ', 'tau']:
        actual_p_type = 'gas'
    else:
        actual_p_type = p_type
    
    add_mass = False  # Handle IllustrisTNG DM case without Masses field
    if actual_p_type == 'gas':
        p_type_val = 'PartType0'
    elif actual_p_type == 'DM':
        p_type_val = 'PartType1'
        if 'Masses' in read_keys and sim_type == 'IllustrisTNG':
            read_keys[read_keys.index('Masses')] = 'ParticleIDs'
            add_mass = True
    elif actual_p_type == 'Stars':
        p_type_val = 'PartType4'
    elif actual_p_type == 'BH':
        p_type_val = 'PartType5'
    else:
        raise NotImplementedError(f'Particle Type not implemented: {actual_p_type}')

    particles = {}
    with h5py.File(snap_path, 'r') as f:
        file_header = dict(f['Header'].attrs.items())
        for key in read_keys:
            # FLAMINGO black holes have no 'Masses' dataset; use DynamicalMasses
            # (the gravitating mass) as the analog for mass fields.
            if key == 'Masses' and p_type_val == 'PartType5' and sim_type == 'FLAMINGO':
                particles[key] = f[p_type_val]['DynamicalMasses'][:] # type: ignore
            else:
                particles[key] = f[p_type_val][key][:] # type: ignore

    if add_mass:
        particles['Masses'] = header['MassTable'][1] * np.ones_like(particles['ParticleIDs'])  # DM mass

    if (not 'ParticleIDs' in keys) and ('ParticleIDs' in particles):
        del particles['ParticleIDs']  # Remove ParticleIDs if we added Masses

    if sim_type == 'FLAMINGO':
        # cMpc -> ckpc/h, 1e10 Msun -> 1e10 Msun/h (see _convert_flamingo_particles)
        _convert_flamingo_particles(particles, header)

    # particles['Masses'] = particles['Masses'] * 1e10 / header['HubbleParam']  # Convert masses to Msun/h
    return particles


def _get_data_filepath(sim_type, sim_name, snapshot, feedback, p_type, n_pixels,
                       projection='xy', data_type='field', dim='2D',
                       mask=False, maskRad=2.0, base_path=None):
    """Generate the file path for saving/loading data.

    Args:
        sim_type (str): The type of simulation.
        sim_name (str): Name of the simulation.
        snapshot (int): Snapshot number.
        feedback (str): Feedback type (for SIMBA).
        p_type (str): Particle type.
        n_pixels (int): Number of pixels.
        projection (str): Projection direction.
        data_type (str): Type of data ('field' or 'map').
        dim (str): Dimension of the data ('2D' or '3D').
        mask (bool): Whether to apply halo masking.
        maskRad (float): Radius for masking in units of R200c.
        base_path (str): Base path to the directory.

    Returns:
        Path: Full file path for the data file.
    """
    if base_path is None:
        base_path = '/pscratch/sd/r/rhliu/simulations/'
    
    if dim == '3D' and data_type == 'map':
        raise ValueError("3D maps are not supported. Please use 'field' for 3D data.")
    
    # Build suffix
    suffix = '_map' if data_type == 'map' else ''
    if mask:
        suffix += f'_masked{maskRad}R200c'
    
    # Build filename
    if sim_type == 'IllustrisTNG':
        if dim == '2D':
            filename = f'{sim_name}_{snapshot}_{p_type}_{n_pixels}_{projection}{suffix}.npy'
        else:  # 3D
            filename = f'{sim_name}_{snapshot}_{p_type}_{n_pixels}{suffix}.npy'
    elif sim_type in ('SIMBA', 'FLAMINGO'):
        # Both suites have feedback variants, included in the filename
        if dim == '2D':
            filename = f'{sim_name}_{feedback}_{snapshot}_{p_type}_{n_pixels}_{projection}{suffix}.npy'
        else:  # 3D
            filename = f'{sim_name}_{feedback}_{snapshot}_{p_type}_{n_pixels}{suffix}.npy'
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")
    
    # Build directory path
    dir_path = Path(f'{base_path}/{sim_type}/products/{dim}/')
    if mask:
        dir_path = dir_path / 'masked'
    
    return dir_path / filename


def load_data(sim_type, sim_name, snapshot, feedback, p_type, n_pixels,
              projection='xy', data_type='field', dim='2D',
              mask=False, maskRad=2.0,
              base_path=None):
    """Load a precomputed field or map from file.

    Args:
        sim_type (str): The type of simulation.
        sim_name (str): Name of the simulation.
        snapshot (int): Snapshot number.
        feedback (str): Feedback type (for SIMBA).
        p_type (str): Particle type.
        n_pixels (int): Number of pixels.
        projection (str): Projection direction.
        data_type (str): Type of data to load ('field' or 'map').
        dim (str): Dimension of the data ('2D' or '3D'). For 3D, only 'field' is supported.
        mask (bool): Whether to apply halo masking.
        maskRad (float): Radius for masking in units of R200c.
        base_path (str): Base path to the directory containing data.

    Returns:
        np.ndarray: 2D numpy array of the field or map.
    """
    filepath = _get_data_filepath(sim_type, sim_name, snapshot, feedback, p_type, n_pixels,
                                   projection, data_type, dim, mask, maskRad, base_path)
    
    try:
        data = np.load(filepath)
    except FileNotFoundError:
        raise ValueError(f"Data file '{filepath}' not found. Please compute it first.")

    return data


def save_data(data, sim_type, sim_name, snapshot, feedback, p_type, n_pixels, 
              projection='xy', data_type='field', dim='2D',
              mask=False, maskRad=2.0,
              base_path=None, mkdir=True):
    """Save a field or map to file.

    Args:
        data (np.ndarray): 2D array to save.
        sim_type (str): The type of simulation.
        sim_name (str): Name of the simulation.
        snapshot (int): Snapshot number.
        feedback (str): Feedback type (for SIMBA).
        p_type (str): Particle type.
        n_pixels (int): Number of pixels.
        projection (str): Projection direction.
        data_type (str): Type of data ('field' or 'map').
        dim (str): Dimension of the data ('2D' or '3D'). For 3D, only 'field' is supported.
        mask (bool): Whether to apply halo masking.
        maskRad (float): Radius for masking in units of R200c.
        base_path (str): Base path to the directory for saving data.
        mkdir (bool): Whether to create the directory if it doesn't exist.
    Returns:
        None
    """
    filepath = _get_data_filepath(sim_type, sim_name, snapshot, feedback, p_type, n_pixels,
                                   projection, data_type, dim, mask, maskRad, base_path)
    
    if mkdir:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    print('Saving data to:', filepath)
    np.save(filepath, data)


def load_group_to_dict(group):
    """Recursively load an HDF5 group into nested dicts of NumPy arrays.

    Args:
        group (h5py.Group): The HDF5 group to load, e.g. f['galaxy_data'].

    Returns:
        dict: Nested dict with datasets as NumPy arrays and subgroups as dicts.
    """
    out = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            out[key] = item[:]  # fully load dataset into memory
        elif isinstance(item, h5py.Group):
            out[key] = load_group_to_dict(item)
        # you can add an `else` branch if you care about other HDF5 object types
    return out

def load_as_dict(file_path, group_name):
    """Load an HDF5 file group into nested dicts of NumPy arrays.

    Args:
        file_path (str): Path to the HDF5 file.
        group_name (str): Name of the group to load.

    Returns:
        dict: Nested dict with datasets as NumPy arrays and subgroups as dicts.
    """
    with h5py.File(file_path, 'r') as f:
        group = f[group_name]
        return load_group_to_dict(group)