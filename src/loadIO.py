import h5py
import numpy as np
import glob
import illustris_python as il


def snap_path(sim_path, snapshot, sim_type, sim_name=None, feedback=None, chunk_num=0, path_only=False):
    """Get the snapshot path for the given simulation type.

    Args:
        sim_path (str): Base path to the simulation.
        snapshot (int): Snapshot number.
        sim_type (str): The type of simulation (e.g., 'IllustrisTNG', 'SIMBA').
        sim_name (str, optional): Name of the simulation (for SIMBA).
        feedback (str, optional): Feedback type (for SIMBA).
        chunk_num (int): The chunk number for the simulation (Only used for IllustrisTNG).
        path_only (bool): If True, returns only the folder path.

    Returns:
        str: The path to the snapshot file or folder.
    """
    if sim_type == 'IllustrisTNG':
        folder_path = sim_path + '/snapdir_' + str(snapshot).zfill(3) + '/'
        snap_path = il.snapshot.snapPath(sim_path, snapshot, chunkNum=chunk_num)
    elif sim_type == 'SIMBA':
        folder_path = sim_path + 'snapshots/'
        snap_path = folder_path + 'snap_' + sim_name + '_' + str(snapshot) + '.hdf5'
        folder_path = snap_path  # SIMBA has different file structure
    
    if path_only:
        return folder_path
    else:
        return snap_path


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
        haloes['GroupMass'] = haloes_cat['GroupMass'] * 1e10  # Convert to Msun/h
        haloes['GroupPos'] = haloes_cat['GroupPos']
        haloes['GroupRad'] = haloes_cat['Group_R_TopHat200']
        
    elif sim_type == 'SIMBA':
        if header is None:
            raise ValueError("Header is required for SIMBA simulations")
        
        halo_path = sim_path + 'catalogs/' + sim_name + '_' + str(snapshot) + '.hdf5'
        haloes = {}
        with h5py.File(halo_path, 'r') as f:
            haloes['GroupPos'] = f['halo_data']['pos'][:] * header['HubbleParam']  # kpc/h # type: ignore
            haloes['GroupMass'] = f['halo_data']['dicts']['masses.total'][:] * header['HubbleParam']  # Msun/h # type: ignore
            haloes['GroupRad'] = f['halo_data']['dicts']['virial_quantities.r200c'][:] * header['HubbleParam']  # kpc/h # type: ignore

    return haloes


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
        raise NotImplementedError('Particle Type not implemented')

    particles = {}
    with h5py.File(snap_path, 'r') as f:
        file_header = dict(f['Header'].attrs.items())
        for key in read_keys:
            particles[key] = f[p_type_val][key][:] # type: ignore

    if add_mass:
        particles['Masses'] = header['MassTable'][1] * np.ones_like(particles['ParticleIDs'])  # DM mass
        del particles['ParticleIDs']  # Remove ParticleIDs if we added Masses
            
    # particles['Masses'] = particles['Masses'] * 1e10 / header['HubbleParam']  # Convert masses to Msun/h
    return particles


def load_data(sim_path, sim_type, sim_name, snapshot, feedback, p_type, n_pixels, projection='xy', data_type='field'):
    """Load a precomputed field or map from file.

    Args:
        sim_path (str): Base path to the simulation.
        sim_type (str): The type of simulation.
        sim_name (str): Name of the simulation.
        snapshot (int): Snapshot number.
        feedback (str): Feedback type (for SIMBA).
        p_type (str): Particle type.
        n_pixels (int): Number of pixels.
        projection (str): Projection direction.
        data_type (str): Type of data to load ('field' or 'map').

    Returns:
        np.ndarray: 2D numpy array of the field or map.
    """
    suffix = '_map' if data_type == 'map' else ''
    
    try:
        if sim_type == 'IllustrisTNG':
            save_name = (sim_name + '_' + str(snapshot) + '_' + 
                        p_type + '_' + str(n_pixels) + '_' + projection + suffix)
            data = np.load(f'/pscratch/sd/r/rhliu/simulations/{sim_type}/products/2D/{save_name}.npy')
        elif sim_type == 'SIMBA':
            save_name = (sim_name + '_' + feedback + '_' + str(snapshot) + '_' +
                        p_type + '_' + str(n_pixels) + '_' + projection + suffix)
            data = np.load(f'/pscratch/sd/r/rhliu/simulations/{sim_type}/products/2D/{save_name}.npy')
    except FileNotFoundError:
        raise ValueError(f"Data for file '{save_name}' not found. Please compute it first.")

    return data


def save_data(data, sim_path, sim_type, sim_name, snapshot, feedback, p_type, n_pixels, projection='xy', data_type='field'):
    """Save a field or map to file.

    Args:
        data (np.ndarray): 2D array to save.
        sim_path (str): Base path to the simulation.
        sim_type (str): The type of simulation.
        sim_name (str): Name of the simulation.
        snapshot (int): Snapshot number.
        feedback (str): Feedback type (for SIMBA).
        p_type (str): Particle type.
        n_pixels (int): Number of pixels.
        projection (str): Projection direction.
        data_type (str): Type of data ('field' or 'map').
    """
    suffix = '_map' if data_type == 'map' else ''
    
    if sim_type == 'IllustrisTNG':
        save_name = (sim_name + '_' + str(snapshot) + '_' + 
                    p_type + '_' + str(n_pixels) + '_' + projection + suffix)
        np.save(f'/pscratch/sd/r/rhliu/simulations/{sim_type}/products/2D/{save_name}.npy', data)
    elif sim_type == 'SIMBA':
        save_name = (sim_name + '_' + feedback + '_' + str(snapshot) + '_' +
                    p_type + '_' + str(n_pixels) + '_' + projection + suffix)
        np.save(f'/pscratch/sd/r/rhliu/simulations/{sim_type}/products/2D/{save_name}.npy', data)

