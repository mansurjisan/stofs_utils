"""
Data extraction module for STOFS3D

Contains functions to extract data from SCHISM model output files,
including slabs, vertical profiles, and time series at specific points.
"""
import os
import numpy as np
from datetime import datetime
from netCDF4 import Dataset
import warnings
from ..core.coordinate_utils import get_zcor_interp_coefficient
from ..utils.helpers import split_quads, ensure_directory
from ..io.netcdf import add_variable
from ..io.ascii_io import read_station_file 

# Constants
DEFAULT_FILL_VALUE = -99999.0
DEFAULT_DEPTH_LEVEL = -4.5


def extract_2d_slab(ds, var_name, k1, coeff):
    """
    Extract a 2D slab from 3D data at interpolated levels.
    
    Parameters
    ----------
    ds : netCDF4.Dataset
        Dataset containing the variable
    var_name : str
        Name of the variable to extract
    k1 : numpy.ndarray
        Lower k-level indices
    coeff : numpy.ndarray
        Interpolation coefficients
        
    Returns
    -------
    numpy.ndarray
        Interpolated values at the specified level
    """
    var_data = np.squeeze(ds[var_name][:])
    NP = var_data.shape[1]
    ntimes = var_data.shape[0]
    
    # Initialize output array
    result = np.full((ntimes, NP), np.nan)
    
    # Interpolate at each time step
    for it in range(ntimes):
        values = var_data[it]
        tmp = np.array(values[np.arange(NP), k1] * (1-coeff) + values[np.arange(NP), k1+1] * coeff)
        result[it, :] = np.squeeze(tmp)
    
    return result


def write_slab_netcdf(output_file, date, x, y, depth, tris, times, elev2d, 
                     temp_sur, temp_bot, salt_sur, salt_bot,
                     uvel_sur, uvel_bot, vvel_sur, vvel_bot,
                     uvel_inter, vvel_inter, depth_level=-4.5, fill_value=-99999.0):
    """
    Write slab data to NetCDF file.
    
    Parameters
    ----------
    output_file : str
        Output file path
    date : datetime
        Reference date
    x, y : numpy.ndarray
        Node coordinates
    depth : numpy.ndarray
        Bathymetry
    tris : numpy.ndarray
        Triangulation connectivity
    times : numpy.ndarray
        Time values
    elev2d, temp_sur, temp_bot, salt_sur, salt_bot, uvel_sur, uvel_bot, 
    vvel_sur, vvel_bot, uvel_inter, vvel_inter : numpy.ndarray
        Data arrays
    depth_level : float, optional
        Depth level for interpolated values
    fill_value : float, optional
        Fill value for masked data
        
    Returns
    -------
    str
        Path to output file
    """
    NP = len(x)
    NE = len(tris)
    NV = 3
    
    with Dataset(output_file, "w", format="NETCDF4") as fout:
        # Create dimensions
        fout.createDimension('time', None)
        fout.createDimension('nSCHISM_hgrid_node', NP)
        fout.createDimension('nSCHISM_hgrid_face', NE)
        fout.createDimension('nMaxSCHISM_hgrid_face_nodes', NV)
        
        # Create time variable
        time_var = fout.createVariable('time', 'f', ('time',))
        time_var.long_name = "Time"
        time_var.units = f'seconds since {date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:00:00 UTC'
        time_var.base_date = f'{date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:00:00 UTC'
        time_var.standard_name = "time"
        time_var[:] = times
        
        # Create coordinate variables
        x_var = fout.createVariable('SCHISM_hgrid_node_x', 'f8', ('nSCHISM_hgrid_node',))
        x_var.long_name = "node x-coordinate"
        x_var.standard_name = "longitude"
        x_var.units = "degrees_east"
        x_var.mesh = "SCHISM_hgrid"
        x_var[:] = x
        
        y_var = fout.createVariable('SCHISM_hgrid_node_y', 'f8', ('nSCHISM_hgrid_node',))
        y_var.long_name = "node y-coordinate"
        y_var.standard_name = "latitude"
        y_var.units = "degrees_north"
        y_var.mesh = "SCHISM_hgrid"
        y_var[:] = y
        
        # Create element connectivity
        ele_var = fout.createVariable('SCHISM_hgrid_face_nodes', 'i', ('nSCHISM_hgrid_face', 'nMaxSCHISM_hgrid_face_nodes',))
        ele_var.long_name = "element"
        ele_var.standard_name = "face_node_connectivity"
        ele_var.start_index = 1
        ele_var.units = "nondimensional"
        ele_var[:] = np.array(tris)
        
        # Create depth variable
        depth_var = fout.createVariable('depth', 'f', ('nSCHISM_hgrid_node',))
        depth_var.long_name = "bathymetry"
        depth_var.units = "m"
        depth_var.mesh = "SCHISM_hgrid"
        depth_var[:] = depth
        
        # Create elevation variable
        elev_var = fout.createVariable('elev', 'f8', ('time', 'nSCHISM_hgrid_node',), fill_value=fill_value)
        elev_var.long_name = "water elevation"
        elev_var.units = "m"
        elev_var.mesh = "SCHISM_hgrid"
        elev_var[:, :] = elev2d
        
        # Create temperature variables
        temp_sur_var = fout.createVariable('temp_surface', 'f8', ('time', 'nSCHISM_hgrid_node',), fill_value=fill_value)
        temp_sur_var.long_name = "sea surface temperature"
        temp_sur_var.units = "deg C"
        temp_sur_var[:, :] = temp_sur
        
        temp_bot_var = fout.createVariable('temp_bottom', 'f8', ('time', 'nSCHISM_hgrid_node',), fill_value=fill_value)
        temp_bot_var.long_name = "Bottom temperature"
        temp_bot_var.units = "deg C"
        temp_bot_var[:, :] = temp_bot
        
        # Create salinity variables
        salt_sur_var = fout.createVariable('salt_surface', 'f8', ('time', 'nSCHISM_hgrid_node',), fill_value=fill_value)
        salt_sur_var.long_name = "sea surface salinity"
        salt_sur_var.units = "psu"
        salt_sur_var[:, :] = salt_sur
        
        salt_bot_var = fout.createVariable('salt_bottom', 'f8', ('time', 'nSCHISM_hgrid_node',), fill_value=fill_value)
        salt_bot_var.long_name = "Bottom salinity"
        salt_bot_var.units = "psu"
        salt_bot_var[:, :] = salt_bot
        
        # Create velocity variables
        uvel_sur_var = fout.createVariable('uvel_surface', 'f8', ('time', 'nSCHISM_hgrid_node',), fill_value=fill_value)
        uvel_sur_var.long_name = "U-component at the surface"
        uvel_sur_var.units = "m/s"
        uvel_sur_var[:, :] = uvel_sur
        
        vvel_sur_var = fout.createVariable('vvel_surface', 'f8', ('time', 'nSCHISM_hgrid_node',), fill_value=fill_value)
        vvel_sur_var.long_name = "V-component at the surface"
        vvel_sur_var.units = "m/s"
        vvel_sur_var[:, :] = vvel_sur
        
        uvel_bot_var = fout.createVariable('uvel_bottom', 'f8', ('time', 'nSCHISM_hgrid_node',), fill_value=fill_value)
        uvel_bot_var.long_name = "U-component at the bottom"
        uvel_bot_var.units = "m/s"
        uvel_bot_var[:, :] = uvel_bot
        
        vvel_bot_var = fout.createVariable('vvel_bottom', 'f8', ('time', 'nSCHISM_hgrid_node',), fill_value=fill_value)
        vvel_bot_var.long_name = "V-component at the bottom"
        vvel_bot_var.units = "m/s"
        vvel_bot_var[:, :] = vvel_bot
        
        # Create interpolated velocity variables
        uvel_interp_var = fout.createVariable(f'uvel{-depth_level}', 'f8', ('time', 'nSCHISM_hgrid_node',), fill_value=fill_value)
        uvel_interp_var.long_name = f"U-component at {-depth_level}m below free surface"
        uvel_interp_var.units = "m/s"
        uvel_interp_var[:, :] = uvel_inter
        
        vvel_interp_var = fout.createVariable(f'vvel{-depth_level}', 'f8', ('time', 'nSCHISM_hgrid_node',), fill_value=fill_value)
        vvel_interp_var.long_name = f"V-component at {-depth_level}m below free surface"
        vvel_interp_var.units = "m/s"
        vvel_interp_var[:, :] = vvel_inter
        
        # Add global attributes
        fout.title = 'SCHISM Model output'
        fout.source = 'SCHISM model output version v10'
        fout.references = 'http://ccrm.vims.edu/schismweb/'
        
    return output_file


def extract_slab_forecast(input_dir, stack, date, results_dir="./results", 
                        depth_level=DEFAULT_DEPTH_LEVEL, fill_value=DEFAULT_FILL_VALUE):
    """
    Extract surface and depth-specific data from SCHISM forecast output NetCDF files.
    
    Parameters
    ----------
    input_dir : str
        Path to SCHISM outputs directory
    stack : int
        Stack number
    date : datetime
        Forecast date
    results_dir : str, optional
        Output directory (default: "./results")
    depth_level : float, optional
        Depth level to extract (negative for below surface, default: -4.5m)
    fill_value : float, optional
        Fill value for masked data (default: -99999.0)
        
    Returns
    -------
    str
        Path to output file
    """
    # Make sure output directory exists
    ensure_directory(results_dir)
    
    # Read NetCDF files
    ds_2d = Dataset(f"{input_dir}/outputs/out2d_{stack}.nc")
    ds_u = Dataset(f"{input_dir}/outputs/horizontalVelX_{stack}.nc")
    ds_v = Dataset(f"{input_dir}/outputs/horizontalVelY_{stack}.nc")
    ds_s = Dataset(f"{input_dir}/outputs/salinity_{stack}.nc")
    ds_t = Dataset(f"{input_dir}/outputs/temperature_{stack}.nc")
    
    # Get kbp and sigma from vgrid.in
    with open(f'{input_dir}/vgrid.in', 'r') as fid:
        lines = fid.readlines()
    
    ivcor = int(lines[0].strip().split()[0])
    nvrt = int(lines[1].strip().split()[0])
    lines = lines[2:]
    sline = np.array(lines[0].split()).astype('float')
    
    if sline.min() < 0:
        kbp = np.array([int(i.split()[1])-1 for i in lines])
        NP = len(kbp)
        sigma = -np.ones([NP, nvrt])
        for i, line in enumerate(lines):
            sigma[i, kbp[i]:] = np.array(line.strip().split()[2:]).astype('float')
    else:
        sline = sline.astype('int')
        kbp = sline - 1
        NP = len(sline)
        sigma = np.array([line.split()[1:] for line in lines[1:]]).T.astype('float')
        fpm = sigma < -1
        sigma[fpm] = -1
    
    # Get coordinates
    x = ds_2d['SCHISM_hgrid_node_x'][:]
    y = ds_2d['SCHISM_hgrid_node_y'][:]
    depth = ds_2d['depth'][:]
    
    # Get elements and split quads into tris
    elements = ds_2d['SCHISM_hgrid_face_nodes'][:, :]
    tris = split_quads(elements)
    NE = len(tris)
    NV = 3
    
    # Get times
    times = ds_2d['time'][:]
    ntimes = len(times)
    
    # Get wetdry nodes
    elev2d = ds_2d['elevation'][:, :]
    maxelev = np.max(elev2d, axis=0)
    idry = np.where(maxelev + depth <= 1e-6)
    elev2d[:, idry] = fill_value
    
    # Process surface data
    temp_sur = ds_t['temperature'][:, :, -1]
    salt_sur = ds_s['salinity'][:, :, -1]
    uvel_sur = np.squeeze(ds_u['horizontalVelX'][:, :, -1])
    vvel_sur = np.squeeze(ds_v['horizontalVelY'][:, :, -1])
    
    # Process and compute interpolation coefficients
    k1_all = []
    coeff_all = []
    
    for it in range(ntimes):
        elev = ds_2d['elevation'][it, :]
        zcor = (depth[:, None] + elev[:, None]) * sigma
        zinter = np.ones(NP) * depth_level + elev
        k1, coeff = get_zcor_interp_coefficient(zcor, zinter, kbp, nvrt)
        k1_all.append(k1)
        coeff_all.append(coeff)
    
    # Process bottom data and prepare for interpolation
    temp_bot = np.zeros((ntimes, NP))
    salt_bot = np.zeros((ntimes, NP))
    uvel_bot = np.zeros((ntimes, NP))
    vvel_bot = np.zeros((ntimes, NP))
    
    # Extract interpolated data using helper functions
    uvel_inter = extract_2d_slab(ds_u, 'horizontalVelX', k1_all[0], coeff_all[0])
    vvel_inter = extract_2d_slab(ds_v, 'horizontalVelY', k1_all[0], coeff_all[0])
    
    # Process bottom data for each time step
    for it in range(ntimes):
        temp_tmp = np.squeeze(ds_t['temperature'][it, :, :])
        salt_tmp = np.squeeze(ds_s['salinity'][it, :, :])
        uvel = np.squeeze(ds_u['horizontalVelX'][it, :, :])
        vvel = np.squeeze(ds_v['horizontalVelY'][it, :, :])
        
        temp_bot[it, :] = temp_tmp[np.arange(NP), kbp]
        salt_bot[it, :] = salt_tmp[np.arange(NP), kbp]
        uvel_bot[it, :] = uvel[np.arange(NP), kbp-1]
        vvel_bot[it, :] = vvel[np.arange(NP), kbp-1]
    
    # Mask dry nodes
    temp_sur[:, idry] = fill_value
    salt_sur[:, idry] = fill_value
    uvel_sur[:, idry] = fill_value
    vvel_sur[:, idry] = fill_value
    temp_bot[:, idry] = fill_value
    salt_bot[:, idry] = fill_value
    uvel_bot[:, idry] = fill_value
    vvel_bot[:, idry] = fill_value
    uvel_inter[:, idry] = fill_value
    vvel_inter[:, idry] = fill_value
    
    # Change fill values
    elev2d[np.where(elev2d > 10000)] = fill_value
    temp_sur[np.where(temp_sur > 10000)] = fill_value
    salt_sur[np.where(salt_sur > 10000)] = fill_value
    uvel_sur[np.where(uvel_sur > 10000)] = fill_value
    vvel_sur[np.where(vvel_sur > 10000)] = fill_value
    temp_bot[np.where(temp_bot > 10000)] = fill_value
    salt_bot[np.where(salt_bot > 10000)] = fill_value
    uvel_bot[np.where(uvel_bot > 10000)] = fill_value
    vvel_bot[np.where(vvel_bot > 10000)] = fill_value
    uvel_inter[np.where(uvel_inter > 10000)] = fill_value
    vvel_inter[np.where(vvel_inter > 10000)] = fill_value
    
    # Write to NetCDF file
    output_file = f"{results_dir}/schout_2d_{stack}.nc"
    write_slab_netcdf(
        output_file, date, x, y, depth, tris, times, elev2d,
        temp_sur, temp_bot, salt_sur, salt_bot,
        uvel_sur, uvel_bot, vvel_sur, vvel_bot,
        uvel_inter, vvel_inter, depth_level=DEFAULT_DEPTH_LEVEL, fill_value=DEFAULT_FILL_VALUE
    )
    
    # Close input files
    ds_2d.close()
    ds_u.close()
    ds_v.close()
    ds_s.close()
    ds_t.close()
    
    print(f"Extracted data saved to: {output_file}")
    return output_file


def extract_point_timeseries(input_dir, stack_range, point_coords, point_names=None, 
                           output_file=None, variables=None, vertical_layers=None):
    """
    Extract time series at specific points from SCHISM output.
    
    Parameters
    ----------
    input_dir : str
        Directory containing SCHISM output
    stack_range : list or tuple
        Range of stacks to process (e.g., [1, 2])
    point_coords : numpy.ndarray
        Point coordinates [npoints, 2] (lon, lat)
    point_names : list, optional
        Names for each point
    output_file : str, optional
        Output file path
    variables : list, optional
        Variables to extract (default: ['elevation', 'temperature', 'salinity', 'horizontalVelX', 'horizontalVelY'])
    vertical_layers : list, optional
        Vertical layers to extract for 3D variables (default: [0, -1] - bottom and surface)
        
    Returns
    -------
    dict
        Dictionary with extracted data
    """
    from ..core.grid import SchismGrid
    from scipy.interpolate import griddata
    
    if variables is None:
        variables = ['elevation', 'temperature', 'salinity', 'horizontalVelX', 'horizontalVelY']
    
    if vertical_layers is None:
        vertical_layers = [0, -1]  # Bottom and surface
    
    # Load grid
    grid_file = f"{input_dir}/hgrid.gr3"
    if not os.path.exists(grid_file):
        raise FileNotFoundError(f"Grid file not found: {grid_file}")
    
    grid = SchismGrid(grid_file)
    
    # Identify 2D and 3D variables
    var_2d = ['elevation']
    var_3d = [v for v in variables if v not in var_2d]
    
    # Get point indices using area coordinates
    ie, ip, acor = grid.compute_acor(point_coords)
    
    # Check if any points are outside the domain
    if np.any(ie == -1):
        outside_pts = np.where(ie == -1)[0]
        warnings.warn(f"Points {outside_pts} are outside the model domain")
    
    # Initialize output data structure
    data = {
        'time': [],
        'points': {
            'coords': point_coords,
            'names': point_names if point_names is not None else [f"Point_{i}" for i in range(len(point_coords))]
        },
        'variables': {}
    }
    
    for var in variables:
        if var in var_2d:
            data['variables'][var] = []
        else:
            data['variables'][var] = {layer: [] for layer in vertical_layers}
    
    # Process each stack
    for stack in range(stack_range[0], stack_range[1] + 1):
        print(f"Processing stack {stack}")
        
        # Process 2D variables
        for var in var_2d:
            try:
                ds = Dataset(f"{input_dir}/outputs/out2d_{stack}.nc")
                times = ds['time'][:]
                
                if len(data['time']) == 0:
                    data['time'] = list(times)
                else:
                    data['time'].extend(times)
                
                var_data = ds[var][:]
                
                # Interpolate to points
                point_data = []
                for i in range(var_data.shape[0]):
                    values = var_data[i]
                    # Use area coordinates for interpolation
                    interp_values = np.sum(values[ip] * acor, axis=1)
                    point_data.append(interp_values)
                
                data['variables'][var].extend(point_data)
                ds.close()
                
            except Exception as e:
                warnings.warn(f"Error processing {var} from stack {stack}: {e}")
        
        # Process 3D variables
        for var in var_3d:
            try:
                ds = Dataset(f"{input_dir}/outputs/{var}_{stack}.nc")
                var_data = ds[var][:]
                
                for layer in vertical_layers:
                    layer_data = []
                    for i in range(var_data.shape[0]):
                        values = var_data[i, :, layer]
                        # Use area coordinates for interpolation
                        interp_values = np.sum(values[ip] * acor, axis=1)
                        layer_data.append(interp_values)
                    
                    data['variables'][var][layer].extend(layer_data)
                
                ds.close()
                
            except Exception as e:
                warnings.warn(f"Error processing {var} from stack {stack}: {e}")
    
    # Convert lists to numpy arrays
    data['time'] = np.array(data['time'])
    
    for var in var_2d:
        data['variables'][var] = np.array(data['variables'][var])
    
    for var in var_3d:
        for layer in vertical_layers:
            data['variables'][var][layer] = np.array(data['variables'][var][layer])
    
    # Save to file if requested
    if output_file is not None:
        with Dataset(output_file, 'w', format='NETCDF4') as ds:
            # Create dimensions
            ds.createDimension('time', None)
            ds.createDimension('point', len(point_coords))
            
            # Create variables
            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = 'seconds since reference'
            time_var[:] = data['time']
            
            # Store point coordinates
            x_var = ds.createVariable('x', 'f8', ('point',))
            x_var.units = 'degrees_east'
            x_var.long_name = 'longitude'
            x_var[:] = point_coords[:, 0]
            
            y_var = ds.createVariable('y', 'f8', ('point',))
            y_var.units = 'degrees_north'
            y_var.long_name = 'latitude'
            y_var[:] = point_coords[:, 1]
            
            # Store point names if available
            if point_names is not None:
                name_var = ds.createVariable('name', 'str', ('point',))
                name_var.long_name = 'point name'
                name_var[:] = np.array(point_names)
            
            # Store 2D variables
            for var in var_2d:
                var_out = ds.createVariable(var, 'f8', ('time', 'point'), fill_value=-99999.0)
                var_out[:] = data['variables'][var]
            
            # Store 3D variables
            for var in var_3d:
                for layer in vertical_layers:
                    layer_name = 'bottom' if layer == 0 else 'surface' if layer == -1 else f'layer_{layer}'
                    var_name = f"{var}_{layer_name}"
                    var_out = ds.createVariable(var_name, 'f8', ('time', 'point'), fill_value=-99999.0)
                    var_out[:] = data['variables'][var][layer]
    
    return data


def extract_vertical_profile(input_dir, stack, locations, variables=None, output_file=None):
    """
    Extract vertical profiles at specified locations.
    
    Parameters
    ----------
    input_dir : str
        Directory containing SCHISM output
    stack : int
        Stack number to process
    locations : numpy.ndarray
        Location coordinates [nlocations, 2] (lon, lat)
    variables : list, optional
        Variables to extract (default: ['temperature', 'salinity', 'horizontalVelX', 'horizontalVelY'])
    output_file : str, optional
        Output file path
        
    Returns
    -------
    dict
        Dictionary with extracted data
    """
    from ..core.grid import SchismGrid
    from ..core.vertical_grid import SchismVGrid
    
    if variables is None:
        variables = ['temperature', 'salinity', 'horizontalVelX', 'horizontalVelY']
    
    # Load grid
    grid = SchismGrid(f"{input_dir}/hgrid.gr3")
    grid.compute_side(fmt=2)
    
    # Load vertical grid
    vgrid = SchismVGrid()
    vgrid.read_vgrid(f"{input_dir}/vgrid.in")
    
    # Get location indices using area coordinates
    ie, ip, acor = grid.compute_acor(locations)
    
    # Check if any locations are outside the domain
    if np.any(ie == -1):
        outside_pts = np.where(ie == -1)[0]
        warnings.warn(f"Locations {outside_pts} are outside the model domain")
    
    # Load elevation
    ds_2d = Dataset(f"{input_dir}/outputs/out2d_{stack}.nc")
    times = ds_2d['time'][:]
    
    # Initialize output data
    profiles = {
        'locations': locations,
        'variables': {},
        'time': times,
        'depths': {}
    }
    
    # Process each time step
    for time_idx, time_val in enumerate(times):
        print(f"Processing time step {time_idx+1}/{len(times)}")
        
        # Get elevation
        elev = ds_2d['elevation'][time_idx]
        
        # Compute z-coordinates
        zcor = vgrid.compute_zcor(grid.dp, eta=elev)
        
        # Get interpolated z-coordinates at locations
        z_interp = np.zeros((len(locations), vgrid.nvrt))
        for i in range(len(locations)):
            z_interp[i] = np.sum(zcor[ip[i]] * acor[i, :, None], axis=0)
        
        # Store depths
        if time_idx == 0:
            for i in range(len(locations)):
                profiles['depths'][i] = z_interp[i]
        
        # Process each variable
        for var in variables:
            if time_idx == 0:
                profiles['variables'][var] = np.zeros((len(times), len(locations), vgrid.nvrt))
            
            try:
                # Read data
                ds_var = Dataset(f"{input_dir}/outputs/{var}_{stack}.nc")
                var_data = ds_var[var][time_idx]
                
                # Interpolate to locations
                for i in range(len(locations)):
                    if ie[i] >= 0:  # Only for valid points
                        var_at_point = np.sum(var_data[ip[i]] * acor[i, :, None], axis=0)
                        profiles['variables'][var][time_idx, i] = var_at_point
                
                ds_var.close()
                
            except Exception as e:
                warnings.warn(f"Error processing {var}: {e}")
    
    # Close 2D dataset
    ds_2d.close()
    
    # Save to file if requested
    if output_file is not None:
        with Dataset(output_file, 'w', format='NETCDF4') as ds:
            # Create dimensions
            ds.createDimension('time', len(times))
            ds.createDimension('location', len(locations))
            ds.createDimension('level', vgrid.nvrt)
            
            # Create variables
            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = 'seconds since reference'
            time_var[:] = times
            
            # Store location coordinates
            x_var = ds.createVariable('longitude', 'f8', ('location',))
            x_var.units = 'degrees_east'
            x_var[:] = locations[:, 0]
            
            y_var = ds.createVariable('latitude', 'f8', ('location',))
            y_var.units = 'degrees_north'
            y_var[:] = locations[:, 1]
            
            # Store
            # Store depths
            z_var = ds.createVariable('depth', 'f8', ('location', 'level'))
            z_var.units = 'm'
            z_var.positive = 'up'
            for i in range(len(locations)):
                z_var[i, :] = profiles['depths'][i]
            
            # Store variables
            for var in variables:
                var_out = ds.createVariable(var, 'f8', ('time', 'location', 'level'), fill_value=-99999.0)
                var_out[:] = profiles['variables'][var]
    
    return profiles


def extract_transect(input_dir, stack, start_point, end_point, num_points=100, 
                   variables=None, output_file=None, time_step=0):
    """
    Extract data along a transect.
    
    Parameters
    ----------
    input_dir : str
        Directory containing SCHISM output
    stack : int
        Stack number to process
    start_point : list or tuple
        Start point coordinates [lon, lat]
    end_point : list or tuple
        End point coordinates [lon, lat]
    num_points : int, optional
        Number of points along transect (default: 100)
    variables : list, optional
        Variables to extract (default: ['elevation', 'temperature', 'salinity', 'horizontalVelX', 'horizontalVelY'])
    output_file : str, optional
        Output file path
    time_step : int, optional
        Time step to extract (default: 0)
        
    Returns
    -------
    dict
        Dictionary with extracted data
    """
    from ..core.grid import SchismGrid
    from ..core.vertical_grid import SchismVGrid
    
    if variables is None:
        variables = ['elevation', 'temperature', 'salinity', 'horizontalVelX', 'horizontalVelY']
    
    # Generate points along transect
    lons = np.linspace(start_point[0], end_point[0], num_points)
    lats = np.linspace(start_point[1], end_point[1], num_points)
    transect_points = np.column_stack((lons, lats))
    
    # Load grid
    grid = SchismGrid(f"{input_dir}/hgrid.gr3")
    grid.compute_side(fmt=2)
    
    # Load vertical grid
    vgrid = SchismVGrid()
    vgrid.read_vgrid(f"{input_dir}/vgrid.in")
    
    # Get point indices using area coordinates
    ie, ip, acor = grid.compute_acor(transect_points)
    
    # Check if any points are outside the domain
    outside_pts = np.where(ie == -1)[0]
    if len(outside_pts) > 0:
        warnings.warn(f"{len(outside_pts)} points along transect are outside model domain")
        # Filter valid points
        valid_mask = ie != -1
        transect_points = transect_points[valid_mask]
        ie = ie[valid_mask]
        ip = ip[valid_mask]
        acor = acor[valid_mask]
    
    # Calculate distances along transect
    distances = np.zeros(len(transect_points))
    for i in range(1, len(transect_points)):
        # Simple Euclidean distance (could be replaced with haversine for geographic coordinates)
        dx = transect_points[i, 0] - transect_points[i-1, 0]
        dy = transect_points[i, 1] - transect_points[i-1, 1]
        distances[i] = distances[i-1] + np.sqrt(dx*dx + dy*dy)
    
    # Load elevation
    ds_2d = Dataset(f"{input_dir}/outputs/out2d_{stack}.nc")
    times = ds_2d['time'][:]
    
    if time_step >= len(times):
        raise ValueError(f"Time step {time_step} out of range (max: {len(times)-1})")
    
    # Get elevation for requested time step
    elev = ds_2d['elevation'][time_step]
    
    # Compute z-coordinates
    zcor = vgrid.compute_zcor(grid.dp, eta=elev)
    
    # Get interpolated z-coordinates at transect points
    z_interp = np.zeros((len(transect_points), vgrid.nvrt))
    for i in range(len(transect_points)):
        z_interp[i] = np.sum(zcor[ip[i]] * acor[i, :, None], axis=0)
    
    # Initialize output data
    transect_data = {
        'points': transect_points,
        'distances': distances,
        'z_levels': z_interp,
        'variables': {},
        'time': times[time_step]
    }
    
    # Process each variable
    for var in variables:
        try:
            # Determine if 2D or 3D variable
            if var == 'elevation':
                # 2D variable
                var_data = elev
                # Interpolate to transect points
                var_at_points = np.zeros(len(transect_points))
                for i in range(len(transect_points)):
                    var_at_points[i] = np.sum(var_data[ip[i]] * acor[i, :], axis=0)
                
                transect_data['variables'][var] = var_at_points
            else:
                # 3D variable
                ds_var = Dataset(f"{input_dir}/outputs/{var}_{stack}.nc")
                var_data = ds_var[var][time_step]
                
                # Interpolate to transect points
                var_at_points = np.zeros((len(transect_points), vgrid.nvrt))
                for i in range(len(transect_points)):
                    var_at_points[i] = np.sum(var_data[ip[i]] * acor[i, :, None], axis=0)
                
                transect_data['variables'][var] = var_at_points
                ds_var.close()
                
        except Exception as e:
            warnings.warn(f"Error processing {var}: {e}")
    
    # Close 2D dataset
    ds_2d.close()
    
    # Save to file if requested
    if output_file is not None:
        with Dataset(output_file, 'w', format='NETCDF4') as ds:
            # Create dimensions
            ds.createDimension('point', len(transect_points))
            ds.createDimension('level', vgrid.nvrt)
            
            # Create variables
            time_var = ds.createVariable('time', 'f8', ())
            time_var.units = 'seconds since reference'
            time_var[:] = times[time_step]
            
            # Store transect coordinates
            x_var = ds.createVariable('longitude', 'f8', ('point',))
            x_var.units = 'degrees_east'
            x_var[:] = transect_points[:, 0]
            
            y_var = ds.createVariable('latitude', 'f8', ('point',))
            y_var.units = 'degrees_north'
            y_var[:] = transect_points[:, 1]
            
            # Store distances
            dist_var = ds.createVariable('distance', 'f8', ('point',))
            dist_var.units = 'degrees'
            dist_var.long_name = 'distance along transect'
            dist_var[:] = distances
            
            # Store depths
            z_var = ds.createVariable('depth', 'f8', ('point', 'level'))
            z_var.units = 'm'
            z_var.positive = 'up'
            z_var[:] = z_interp
            
            # Store variables
            for var in transect_data['variables']:
                if var == 'elevation':
                    var_out = ds.createVariable(var, 'f8', ('point',), fill_value=-99999.0)
                    var_out[:] = transect_data['variables'][var]
                else:
                    var_out = ds.createVariable(var, 'f8', ('point', 'level'), fill_value=-99999.0)
                    var_out[:] = transect_data['variables'][var]
    
    return transect_data


def extract_station_timeseries(input_dir, output_dir, station_file='station.in', date=None):
    """
    Extract station timeseries from staout_* files.
    
    Parameters
    ----------
    input_dir : str
        Directory containing SCHISM outputs (staout_* files)
    output_dir : str
        Directory to save output
    station_file : str, optional
        Station file with station coordinates and names (default: 'station.in')
    date : datetime, optional
        Reference date for time values
        
    Returns
    -------
    str
        Path to output file
    """
    import glob
    from scipy import interpolate
    from netCDF4 import stringtochar
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Get list of staout files
    staout_files = glob.glob(f"{input_dir}/staout_*")
    if not staout_files:
        raise FileNotFoundError(f"No staout_* files found in {input_dir}")
    
    # Read station information
    lon, lat, station_names = read_station_file(f"{input_dir}/{station_file}")
    nstation = len(station_names)
    
    # Define variable mapping based on staout files
    var_mapping = {
        'staout_1': {'name': 'elevation', 'long_name': 'water surface elevation', 'standard_name': 'sea_surface_height_above_datum', 'units': 'm'},
        'staout_2': {'name': 'air_pressure', 'long_name': 'air pressure', 'standard_name': 'air_pressure', 'units': 'Pa'},
        'staout_3': {'name': 'windx', 'long_name': 'eastward wind', 'standard_name': 'eastward_wind', 'units': 'm s-1'},
        'staout_4': {'name': 'windy', 'long_name': 'northward wind', 'standard_name': 'northward_wind', 'units': 'm s-1'},
        'staout_5': {'name': 'temp', 'long_name': 'sea water temperature', 'standard_name': 'sea_water_temperature', 'units': 'C'},
        'staout_6': {'name': 'salt', 'long_name': 'sea water salinity', 'standard_name': 'sea_water_salinity', 'units': 'psu'},
        'staout_7': {'name': 'u', 'long_name': 'eastward sea water velocity', 'standard_name': 'eastward_sea_water_velocity', 'units': 'm s-1'},
        'staout_8': {'name': 'v', 'long_name': 'northward sea water velocity', 'standard_name': 'northward_sea_water_velocity', 'units': 'm s-1'}
    }
    
    # Output file path
    if date is not None:
        output_file = f"{output_dir}/staout_timeseries_{date.strftime('%Y-%m-%d-%H')}.nc"
    else:
        output_file = f"{output_dir}/staout_timeseries.nc"
    
    # Create NetCDF file
    with Dataset(output_file, "w", format="NETCDF4") as fout:
        # Dimensions
        fout.createDimension('station', nstation)
        fout.createDimension('namelen', 50)  # For station names
        fout.createDimension('time', None)
        
        # Process each staout file
        for ivar, staout_file in enumerate(sorted(staout_files)):
            base_name = os.path.basename(staout_file)
            
            # Skip if not in mapping
            if base_name not in var_mapping:
                continue
            
            print(f"Processing {staout_file}")
            
            try:
                # Read data
                data = np.loadtxt(staout_file)
                time = data[:, 0]
                nt = len(time)
                
                # Create model data array
                model = np.ndarray(shape=(nt, nstation), dtype=float)
                model[:, :] = data[:, 1:]
                
                # Interpolate to regular time steps
                out_dt = 360
                t_interp = np.arange(out_dt, time[-1] + out_dt/2, out_dt)
                
                f_interp = interpolate.interp1d(time, model, axis=0, fill_value='extrapolate')
                model = f_interp(t_interp)
                
                # Add common variables on first variable
                if ivar == 0:
                    # Time variable
                    time_var = fout.createVariable('time', 'f8', ('time',))
                    time_var.long_name = "Time"
                    if date is not None:
                        time_var.units = f'seconds since {date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:00:00 UTC'
                        time_var.base_date = f'{date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:00:00 UTC'
                    else:
                        time_var.units = 'seconds'
                    time_var.standard_name = "time"
                    time_var[:] = t_interp
                    
                    # Station name variable
                    station_var = fout.createVariable('station_name', 'c', ('station', 'namelen',))
                    station_var.long_name = "station name"
                    
                    # Create string array for station names
                    names = np.empty((nstation,), 'S50')
                    for i in range(nstation):
                        names[i] = str(station_names[i])
                    namesc = stringtochar(names)
                    station_var[:] = namesc
                    
                    # Coordinate variables
                    x_var = fout.createVariable('x', 'f8', ('station',))
                    x_var.long_name = "longitude"
                    x_var.standard_name = "longitude"
                    x_var.units = "degrees_east"
                    x_var.positive = "east"
                    x_var[:] = lon
                    
                    y_var = fout.createVariable('y', 'f8', ('station',))
                    y_var.long_name = "latitude"
                    y_var.standard_name = "latitude"
                    y_var.units = "degrees_north"
                    y_var.positive = "north"
                    y_var[:] = lat
                    
                    # Global attributes
                    fout.title = 'SCHISM Model station output'
                    fout.source = 'SCHISM model'
                    fout.references = 'http://ccrm.vims.edu/schismweb/'
                
                # Create variable for current field
                if base_name in var_mapping:
                    var_info = var_mapping[base_name]
                    var_out = fout.createVariable(var_info['name'], 'f8', ('time', 'station',), fill_value=-99999.)
                    var_out.long_name = var_info['long_name']
                    var_out.standard_name = var_info['standard_name']
                    var_out.units = var_info['units']
                    var_out[:, :] = model
            
            except Exception as e:
                warnings.warn(f"Error processing {staout_file}: {e}")
    
    return output_file


def extract_maxele(input_dir, stack_range, output_file=None):
    """
    Extract maximum elevation from SCHISM output.
    
    Parameters
    ----------
    input_dir : str
        Directory containing SCHISM outputs
    stack_range : list or tuple
        Range of stacks to process [start, end]
    output_file : str, optional
        Output file path
        
    Returns
    -------
    tuple
        (maxele, time_of_maxele, x, y, depth)
    """
    # Initialize with large negative values
    maxele = None
    time_of_maxele = None
    x = None
    y = None
    depth = None
    
    # Process each stack
    for stack in range(stack_range[0], stack_range[1] + 1):
        print(f"Processing stack {stack}")
        
        try:
            ds = Dataset(f"{input_dir}/outputs/out2d_{stack}.nc")
            
            # Get coordinates and bathymetry on first stack
            if maxele is None:
                x = ds['SCHISM_hgrid_node_x'][:]
                y = ds['SCHISM_hgrid_node_y'][:]
                depth = ds['depth'][:]
                
                # Initialize with large negative values
                maxele = np.full_like(x, -1e6)
                time_of_maxele = np.zeros_like(x)
            
            # Get times
            times = ds['time'][:]
            
            # Get elevation
            elev = ds['elevation'][:, :]
            
            # Update maximum elevation
            for i in range(elev.shape[0]):
                higher = elev[i, :] > maxele
                maxele[higher] = elev[i, higher]
                time_of_maxele[higher] = times[i]
            
            ds.close()
            
        except Exception as e:
            warnings.warn(f"Error processing stack {stack}: {e}")
    
    # Replace remaining large negative values with NaN
    maxele[maxele < -1e5] = np.nan
    
    # Save to file if requested
    if output_file is not None:
        with Dataset(output_file, 'w', format='NETCDF4') as ds:
            # Create dimensions
            ds.createDimension('node', len(x))
            
            # Create variables
            x_var = ds.createVariable('x', 'f8', ('node',))
            x_var.long_name = "longitude"
            x_var.standard_name = "longitude"
            x_var.units = "degrees_east"
            x_var[:] = x
            
            y_var = ds.createVariable('y', 'f8', ('node',))
            y_var.long_name = "latitude"
            y_var.standard_name = "latitude"
            y_var.units = "degrees_north"
            y_var[:] = y
            
            depth_var = ds.createVariable('depth', 'f8', ('node',))
            depth_var.long_name = "bathymetry"
            depth_var.units = "m"
            depth_var[:] = depth
            
            maxele_var = ds.createVariable('maxele', 'f8', ('node',), fill_value=-99999.0)
            maxele_var.long_name = "maximum elevation"
            maxele_var.units = "m"
            maxele_var[:] = maxele
            
            time_var = ds.createVariable('time_of_maxele', 'f8', ('node',), fill_value=-99999.0)
            time_var.long_name = "time of maximum elevation"
            time_var.units = "seconds"
            time_var[:] = time_of_maxele
            
            # Global attributes
            ds.title = 'SCHISM Model maximum elevation'
            ds.source = 'SCHISM model'
            ds.references = 'http://ccrm.vims.edu/schismweb/'
    
    return maxele, time_of_maxele, x, y, depth


def extract_cli():
    """
    Command-line interface for extraction utilities.
    
    This function handles command-line arguments for the extraction utilities.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract data from SCHISM outputs')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')
    
    # Parser for extracting slabs
    parser_slab = subparsers.add_parser('slab', help='Extract surface and depth levels')
    parser_slab.add_argument('--input_dir', type=str, required=True, help='Input directory with SCHISM outputs')
    parser_slab.add_argument('--stack', type=int, required=True, help='Stack number')
    parser_slab.add_argument('--date', type=lambda s: datetime.strptime(s, '%Y-%m-%d-%H'),
                           required=True, help='Date (YYYY-MM-DD-HH)')
    parser_slab.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser_slab.add_argument('--depth_level', type=float, default=-4.5, help='Depth level to extract')
    
    # Parser for extracting time series at points
    parser_ts = subparsers.add_parser('timeseries', help='Extract time series at specific points')
    parser_ts.add_argument('--input_dir', type=str, required=True, help='Input directory with SCHISM outputs')
    parser_ts.add_argument('--stack_start', type=int, required=True, help='Starting stack number')
    parser_ts.add_argument('--stack_end', type=int, required=True, help='Ending stack number')
    parser_ts.add_argument('--points', type=str, required=True, help='Points file (CSV with lon,lat columns)')
    parser_ts.add_argument('--output_file', type=str, required=True, help='Output file')
    parser_ts.add_argument('--variables', type=str, default='elevation,temperature,salinity,horizontalVelX,horizontalVelY',
                         help='Comma-separated list of variables')
    
    # Parser for extracting station time series
    parser_station = subparsers.add_parser('station', help='Extract station time series')
    parser_station.add_argument('--input_dir', type=str, required=True, help='Input directory with SCHISM outputs')
    parser_station.add_argument('--station_file', type=str, default='station.in', help='Station file')
    parser_station.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser_station.add_argument('--date', type=lambda s: datetime.strptime(s, '%Y-%m-%d-%H'),
                              help='Date (YYYY-MM-DD-HH)')
    
    # Parser for extracting maximum elevation
    parser_maxele = subparsers.add_parser('maxele', help='Extract maximum elevation')
    parser_maxele.add_argument('--input_dir', type=str, required=True, help='Input directory with SCHISM outputs')
    parser_maxele.add_argument('--stack_start', type=int, required=True, help='Starting stack number')
    parser_maxele.add_argument('--stack_end', type=int, required=True, help='Ending stack number')
    parser_maxele.add_argument('--output_file', type=str, required=True, help='Output file')
    
    args = parser.parse_args()
    
    if args.command == 'slab':
        output_file = extract_slab_forecast(
            input_dir=args.input_dir,
            stack=args.stack,
            date=args.date,
            results_dir=args.output_dir,
            depth_level=args.depth_level
        )
        print(f"Slab data extracted to: {output_file}")
        
    elif args.command == 'timeseries':
        # Read points from CSV
        points_data = np.loadtxt(args.points, delimiter=',')
        if points_data.ndim == 1:
            points_data = points_data.reshape(1, -1)
        
        output_data = extract_point_timeseries(
            input_dir=args.input_dir,
            stack_range=[args.stack_start, args.stack_end],
            point_coords=points_data[:, :2],
            variables=args.variables.split(','),
            output_file=args.output_file
        )
        print(f"Time series data extracted to: {args.output_file}")
        
    elif args.command == 'station':
        output_file = extract_station_timeseries(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            station_file=args.station_file,
            date=args.date
        )
        print(f"Station time series extracted to: {output_file}")
        
    elif args.command == 'maxele':
        maxele, time_of_maxele, x, y, depth = extract_maxele(
            input_dir=args.input_dir,
            stack_range=[args.stack_start, args.stack_end],
            output_file=args.output_file
        )
        print(f"Maximum elevation data extracted to: {args.output_file}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    extract_cli()