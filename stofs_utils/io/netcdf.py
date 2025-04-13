"""
NetCDF file handling utilities for STOFS3D

Contains functions for reading, writing, and manipulating NetCDF files
used in STOFS3D operational forecasting.
"""
import numpy as np
from netCDF4 import Dataset
import os
import datetime


def read_netcdf(fname, fmt=0, order=0):
    """
    Read NetCDF files and return their contents.
    
    Parameters
    ----------
    fname : str
        Path to NetCDF file
    fmt : int, optional
        Output format:
        - 0: Return reorganized Dataset as custom object (default)
        - 1: Return netCDF4.Dataset directly
        - 2: Return reorganized Dataset, ignore attributes
    order : int, optional
        Variable dimension order:
        - 0: Keep original order (default)
        - 1: Reverse dimension order (matlab/fortran format)
        
    Returns
    -------
    object or netCDF4.Dataset
        NetCDF dataset in requested format
    """
    # Open the file
    nc = Dataset(fname)
    
    if fmt == 1:
        return nc
    elif fmt in [0, 2]:
        # Create a data container
        class DataContainer:
            pass
        
        F = DataContainer()
        F.file_format = nc.file_format
        F.VINFO = []
        
        # Read dimensions
        nc_dims = [i for i in nc.dimensions]
        F.dimname = nc_dims
        F.dims = []
        F.dim_unlimited = []
        
        for i in nc_dims:
            F.dims.append(nc.dimensions[i].size)
            F.dim_unlimited.append(nc.dimensions[i].isunlimited())
            
        F.VINFO.append(f'dimname: {F.dimname}')
        F.VINFO.append(f'dim: {F.dims}')
        
        # Read attributes
        nc_attrs = nc.ncattrs()
        F.attrs = nc_attrs
        
        for i in nc_attrs:
            setattr(F, i, nc.getncattr(i))
        
        # Read variables
        nc_vars = list(nc.variables)
        F.vars = np.array(nc_vars)
        
        for i in nc_vars:
            if fmt == 0:
                # Create container for variable
                vi = DataContainer()
                
                # Get dimension info
                dimi = nc.variables[i].dimensions
                vi.dimname = dimi
                vi.dims = [nc.dimensions[j].size for j in dimi]
                
                # Get value
                vi.val = nc.variables[i][:]
                
                # Get attributes
                vi.attrs = nc.variables[i].ncattrs()
                for j in nc.variables[i].ncattrs():
                    setattr(vi, j, nc.variables[i].getncattr(j))
                
                # Handle dimension order
                if order == 1:
                    vi.dimname = list(np.flipud(vi.dimname))
                    vi.dims = list(np.flipud(vi.dims))
                    nm = np.flipud(np.arange(np.ndim(vi.val)))
                    vi.val = vi.val.transpose(nm)
                    
                vinfo = f'{i}:{vi.val.shape}'
                
            elif fmt == 2:
                vi = np.array(nc.variables[i][:])
                
                if order == 1:
                    vi = vi.transpose(np.flip(np.arange(vi.ndim)))
                    
                vinfo = f'{i}:{vi.shape}'
                
            F.VINFO.append(vinfo)
            setattr(F, i, vi)
        
        nc.close()
        return F
    else:
        raise ValueError(f"Invalid format: {fmt}")


def write_netcdf(fname, data, fmt=0, order=0):
    """
    Write data to a NetCDF file.
    
    Parameters
    ----------
    fname : str
        Output file path
    data : object
        Data to write to file
    fmt : int, optional
        Data format:
        - 0: Data has custom object format (default)
        - 1: Data has netCDF4.Dataset format
    order : int, optional
        Variable dimension order:
        - 0: Keep original order (default)
        - 1: Reverse dimension order (matlab/fortran format)
    """
    if fmt == 1:
        # Write from netCDF4.Dataset
        fid = Dataset(fname, 'w', format=data.file_format)
        fid.setncattr('file_format', data.file_format)
        
        # Set attributes
        nc_attrs = data.ncattrs()
        for i in nc_attrs:
            fid.setncattr(i, getattr(data, i))
        
        # Set dimensions
        nc_dims = [i for i in data.dimensions]
        for i in nc_dims:
            if data.dimensions[i].isunlimited() is True:
                fid.createDimension(i, None)
            else:
                fid.createDimension(i, data.dimensions[i].size)
        
        # Set variables
        nc_vars = [i for i in data.variables]
        
        if order == 0:
            for var_name in nc_vars:
                var_obj = data.variables[var_name]
                vid = fid.createVariable(var_name, var_obj.dtype, var_obj.dimensions)
                
                # Set variable attributes
                for attr_name in var_obj.ncattrs():
                    vid.setncattr(attr_name, var_obj.getncattr(attr_name))
                    
                fid.variables[var_name][:] = var_obj[:]
                
        elif order == 1:
            for var_name in nc_vars:
                var_obj = data.variables[var_name]
                vid = fid.createVariable(var_name, var_obj.dtype, 
                                        np.flipud(var_obj.dimensions))
                
                # Set variable attributes
                for attr_name in var_obj.ncattrs():
                    vid.setncattr(attr_name, var_obj.getncattr(attr_name))
                
                # Transpose dimensions
                nm = np.flipud(np.arange(np.ndim(var_obj[:])))
                fid.variables[var_name][:] = var_obj[:].transpose(nm)
                
        fid.close()
        
    elif fmt == 0:
        # Write from custom object
        fid = Dataset(fname, 'w', format=data.file_format)
        
        # Set attributes
        fid.setncattr('file_format', data.file_format)
        if hasattr(data, 'attrs'):
            for i in data.attrs:
                fid.setncattr(i, getattr(data, i))
        
        # Set dimensions
        for i in range(len(data.dims)):
            dim_unlimited = False
            if hasattr(data, 'dim_unlimited'):
                dim_unlimited = data.dim_unlimited[i]
                
            if dim_unlimited is True:
                fid.createDimension(data.dimname[i], None)
            else:
                fid.createDimension(data.dimname[i], data.dims[i])
        
        # Set variables
        if order == 0:
            for var_name in data.vars:
                vi = getattr(data, var_name)
                vid = fid.createVariable(var_name, vi.val.dtype, vi.dimname)
                
                if hasattr(vi, 'attrs'):
                    for j in vi.attrs:
                        attr_val = getattr(vi, j)
                        vid.setncattr(j, attr_val)
                        
                fid.variables[var_name][:] = vi.val
                
        elif order == 1:
            for var_name in data.vars:
                vi = getattr(data, var_name)
                vid = fid.createVariable(var_name, vi.val.dtype, 
                                        np.flipud(vi.dimname))
                
                if hasattr(vi, 'attrs'):
                    for j in vi.attrs:
                        attr_val = getattr(vi, j)
                        vid.setncattr(j, attr_val)
                
                if np.ndim(vi.val) >= 2:
                    nm = np.flipud(np.arange(np.ndim(vi.val)))
                    fid.variables[var_name][:] = vi.val.transpose(nm)
                else:
                    fid.variables[var_name][:] = vi.val
                    
        fid.close()
    else:
        raise ValueError(f"Invalid format: {fmt}")


def create_netcdf_dataset(fname, file_format="NETCDF4"):
    """
    Create a new NetCDF dataset.
    
    Parameters
    ----------
    fname : str
        Output file path
    file_format : str, optional
        NetCDF file format (default: "NETCDF4")
        
    Returns
    -------
    netCDF4.Dataset
        New NetCDF dataset
    """
    return Dataset(fname, 'w', format=file_format)


def add_dimension(nc, name, size=None):
    """
    Add a dimension to a NetCDF dataset.
    
    Parameters
    ----------
    nc : netCDF4.Dataset
        NetCDF dataset
    name : str
        Dimension name
    size : int or None, optional
        Dimension size (None for unlimited dimension)
    """
    nc.createDimension(name, size)


def add_variable(nc, name, datatype, dimensions, data=None, **attrs):
    """
    Add a variable to a NetCDF dataset.
    
    Parameters
    ----------
    nc : netCDF4.Dataset
        NetCDF dataset
    name : str
        Variable name
    datatype : str or numpy.dtype
        Data type (e.g., 'f4', 'i4')
    dimensions : tuple or list
        Dimension names
    data : array_like, optional
        Variable data
    **attrs : dict
        Variable attributes
        
    Returns
    -------
    netCDF4.Variable
        Created variable
    """
    var = nc.createVariable(name, datatype, dimensions)
    
    # Set attributes
    for key, value in attrs.items():
        var.setncattr(key, value)
    
    # Set data if provided
    if data is not None:
        var[:] = data
        
    return var


def extract_slab_forecast_netcdf(input_filename, stack, date, results_dir="./results"):
    """
    Extract surface and depth-specific data from SCHISM forecast output NetCDF files.
    
    Derived from extract_slab_fcst_netcdf4.py
    
    Parameters
    ----------
    input_filename : str
        Path to input NetCDF file
    stack : int
        Stack number
    date : datetime
        Forecast date
    results_dir : str, optional
        Output directory (default: "./results")
        
    Returns
    -------
    str
        Path to output file
    """
    import numpy as np
    import os
    from netCDF4 import Dataset
    from ..core.coordinate_utils import get_zcor_interp_coefficient
    
    # Make sure output directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Read NetCDF files
    ds_2d = Dataset(f"{input_filename}/outputs/out2d_{stack}.nc")
    ds_u = Dataset(f"{input_filename}/outputs/horizontalVelX_{stack}.nc")
    ds_v = Dataset(f"{input_filename}/outputs/horizontalVelY_{stack}.nc")
    ds_s = Dataset(f"{input_filename}/outputs/salinity_{stack}.nc")
    ds_t = Dataset(f"{input_filename}/outputs/temperature_{stack}.nc")
    
    # Get kbp and sigma from vgrid.in
    with open(f'{input_filename}/vgrid.in', 'r') as fid:
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
    
    # Get wetdry nodes
    elev2d = ds_2d['elevation'][:, :]
    maxelev = np.max(elev2d, axis=0)
    idry = np.where(maxelev + depth <= 1e-6)
    elev2d[:, idry] = -99999
    
    # Get elements and split quads into tris
    elements = ds_2d['SCHISM_hgrid_face_nodes'][:, :]
    tris = []
    for ele in elements:
        ele = np.ma.masked_values(ele, -1)
        ele = ele[~ele.mask]
        if len(ele) == 3:
            tris.append([ele[0], ele[1], ele[2]])
        elif len(ele) == 4:
            tris.append([ele[0], ele[1], ele[3]])
            tris.append([ele[1], ele[2], ele[3]])
    NE = len(tris)
    NV = 3
    
    # Get times
    times = ds_2d['time'][:]
    ntimes = len(times)
    
    # Initialize variables for surface and bottom data
    temp_sur = np.full((ntimes, NP), np.nan)
    salt_sur = np.full((ntimes, NP), np.nan)
    uvel_sur = np.full((ntimes, NP), np.nan)
    vvel_sur = np.full((ntimes, NP), np.nan)
    
    uvel_inter = np.full((ntimes, NP), np.nan)
    vvel_inter = np.full((ntimes, NP), np.nan)
    
    temp_bot = np.full((ntimes, NP), np.nan)
    salt_bot = np.full((ntimes, NP), np.nan)
    uvel_bot = np.full((ntimes, NP), np.nan)
    vvel_bot = np.full((ntimes, NP), np.nan)
    
    # Set interpolation level (depth below surface)
    level = [-4.5]
    
    # Process each time step
    for it in np.arange(ntimes):
        print(f"Processing time step {it+1}/{ntimes}")
        elev = ds_2d['elevation'][it, :]
        
        # Surface values
        temp_sur[it, :] = ds_t['temperature'][it, :, -1]
        salt_sur[it, :] = ds_s['salinity'][it, :, -1]
        uvel_sur[it, :] = np.squeeze(ds_u['horizontalVelX'][it, :, -1])
        vvel_sur[it, :] = np.squeeze(ds_v['horizontalVelY'][it, :, -1])
        
        # All levels
        salt_tmp = np.squeeze(ds_s['salinity'][it, :, :])
        temp_tmp = np.squeeze(ds_t['temperature'][it, :, :])
        
        uvel = np.squeeze(ds_u['horizontalVelX'][it, :, :])
        vvel = np.squeeze(ds_v['horizontalVelY'][it, :, :])
        
        # Compute zcor
        zcor = (depth[:, None] + elev[:, None]) * sigma
        
        # Initialize for interpolation
        k1 = np.full((NP), np.nan)
        coeff = np.full((NP), np.nan)
        zinter = np.ones(NP) * level + elev
        
        # Get interpolation coefficients
        k1, coeff = get_zcor_interp_coefficient(zcor, zinter, kbp, nvrt)
        
        # Bottom salt/temp/u/v
        temp_bot[it, :] = temp_tmp[np.arange(NP), kbp]
        salt_bot[it, :] = salt_tmp[np.arange(NP), kbp]
        
        uvel_bot[it, :] = uvel[np.arange(NP), kbp-1]
        vvel_bot[it, :] = vvel[np.arange(NP), kbp-1]
        
        # Interpolate at level
        tmp = np.array(uvel[np.arange(NP), k1] * (1-coeff) + uvel[np.arange(NP), k1+1] * coeff)
        uvel_inter[it, :] = np.squeeze(tmp)
        
        tmp = np.array(vvel[np.arange(NP), k1] * (1-coeff) + vvel[np.arange(NP), k1+1] * coeff)
        vvel_inter[it, :] = np.squeeze(tmp)
    
    # Mask dry nodes
    temp_sur[:, idry] = -99999
    salt_sur[:, idry] = -99999
    uvel_sur[:, idry] = -99999
    vvel_sur[:, idry] = -99999
    temp_bot[:, idry] = -99999
    salt_bot[:, idry] = -99999
    uvel_bot[:, idry] = -99999
    vvel_bot[:, idry] = -99999
    
    # u/v at 4.5m
    uvel_inter[:, idry] = -99999
    vvel_inter[:, idry] = -99999
    
    # Change fill_values
    elev2d[np.where(elev2d > 10000)] = -99999
    temp_sur[np.where(temp_sur > 10000)] = -99999
    salt_sur[np.where(salt_sur > 10000)] = -99999
    uvel_sur[np.where(uvel_sur > 10000)] = -99999
    vvel_sur[np.where(vvel_sur > 10000)] = -99999
    
    temp_bot[np.where(temp_bot > 10000)] = -99999
    salt_bot[np.where(salt_bot > 10000)] = -99999
    uvel_bot[np.where(uvel_bot > 10000)] = -99999
    vvel_bot[np.where(vvel_bot > 10000)] = -99999
    
    uvel_inter[np.where(uvel_inter > 10000)] = -99999
    vvel_inter[np.where(vvel_inter > 10000)] = -99999
    
    # Create output file
    output_file = f"{results_dir}/schout_2d_{stack}.nc"
    with Dataset(output_file, "w", format="NETCDF4") as fout:
        # Create dimensions
        fout.createDimension('time', None)
        fout.createDimension('nSCHISM_hgrid_node', NP)
        fout.createDimension('nSCHISM_hgrid_face', NE)
        fout.createDimension('nMaxSCHISM_hgrid_face_nodes', NV)
        
        # Create variables
        add_variable(fout, 'time', 'f', ('time',),
                    long_name="Time",
                    units=f'seconds since {date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:00:00 UTC',
                    base_date=f'{date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:00:00 UTC',
                    standard_name="time",
                    data=times)
        
        add_variable(fout, 'SCHISM_hgrid_node_x', 'f8', ('nSCHISM_hgrid_node',),
                    long_name="node x-coordinate",
                    standard_name="longitude",
                    units="degrees_east",
                    mesh="SCHISM_hgrid",
                    data=x)
        
        add_variable(fout, 'SCHISM_hgrid_node_y', 'f8', ('nSCHISM_hgrid_node',),
                    long_name="node y-coordinate",
                    standard_name="latitude",
                    units="degrees_north",
                    mesh="SCHISM_hgrid",
                    data=y)
        
        add_variable(fout, 'SCHISM_hgrid_face_nodes', 'i', ('nSCHISM_hgrid_face', 'nMaxSCHISM_hgrid_face_nodes',),
                    long_name="element",
                    standard_name="face_node_connectivity",
                    start_index=1,
                    units="nondimensional",
                    data=np.array(tris))
        
        add_variable(fout, 'depth', 'f', ('nSCHISM_hgrid_node',),
                    long_name="bathymetry",
                    units="m",
                    mesh="SCHISM_hgrid",
                    data=depth)
        
        add_variable(fout, 'elev', 'f8', ('time', 'nSCHISM_hgrid_node',),
                    long_name="water elevation",
                    units="m",
                    mesh="SCHISM_hgrid",
                    fill_value=-99999,
                    data=elev2d)
        
        add_variable(fout, 'temp_surface', 'f8', ('time', 'nSCHISM_hgrid_node',),
                    long_name="sea surface temperature",
                    units="deg C",
                    fill_value=-99999,
                    data=temp_sur)
        
        add_variable(fout, 'temp_bottom', 'f8', ('time', 'nSCHISM_hgrid_node',),
                    long_name="Bottom temperature",
                    units="deg C",
                    fill_value=-99999,
                    data=temp_bot)
        
        add_variable(fout, 'salt_surface', 'f8', ('time', 'nSCHISM_hgrid_node',),
                    long_name="sea surface salinity",
                    units="psu",
                    fill_value=-99999,
                    data=salt_sur)
        
        add_variable(fout, 'salt_bottom', 'f8', ('time', 'nSCHISM_hgrid_node',),
                    long_name="Bottom salinity",
                    units="psu",
                    fill_value=-99999,
                    data=salt_bot)
        
        add_variable(fout, 'uvel_surface', 'f8', ('time', 'nSCHISM_hgrid_node',),
                    long_name="U-component at the surface",
                    units="m/s",
                    fill_value=-99999,
                    data=uvel_sur)
        
        add_variable(fout, 'vvel_surface', 'f8', ('time', 'nSCHISM_hgrid_node',),
                    long_name="V-component at the surface",
                    units="m/s",
                    fill_value=-99999,
                    data=vvel_sur)
        
        add_variable(fout, 'uvel_bottom', 'f8', ('time', 'nSCHISM_hgrid_node',),
                    long_name="U-component at the bottom",
                    units="m/s",
                    fill_value=-99999,
                    data=uvel_bot)
        
        add_variable(fout, 'vvel_bottom', 'f8', ('time', 'nSCHISM_hgrid_node',),
                    long_name="V-component at the bottom",
                    units="m/s",
                    fill_value=-99999,
                    data=vvel_bot)
        
        add_variable(fout, 'uvel4.5', 'f8', ('time', 'nSCHISM_hgrid_node',),
                    long_name="U-component at 4.5m below free surface",
                    units="m/s",
                    fill_value=-99999,
                    data=uvel_inter)
        
        add_variable(fout, 'vvel4.5', 'f8', ('time', 'nSCHISM_hgrid_node',),
                    long_name="V-component at 4.5m below free surface",
                    units="m/s",
                    fill_value=-99999,
                    data=vvel_inter)
        
        # Add global attributes
        fout.title = 'SCHISM Model output'
        fout.source = 'SCHISM model output version v10'
        fout.references = 'http://ccrm.vims.edu/schismweb/'
        
    # Close all input files
    ds_2d.close()
    ds_u.close()
    ds_v.close()
    ds_s.close()
    ds_t.close()
    
    return output_file


def generate_station_timeseries(date, input_dir, output_dir, station_info_file='stofs_3d_atl_staout_nc.csv', 
                              json_file='stofs_3d_atl_staout_nc.json'):
    """
    Generate timeseries at observation locations from SCHISM staout files.
    
    Derived from generate_station_timeseries.py
    
    Parameters
    ----------
    date : datetime
        Start date for timeseries
    input_dir : str
        Directory containing staout files
    output_dir : str 
        Directory to save timeseries
    station_info_file : str, optional
        CSV file with station information
    json_file : str, optional
        JSON file with variable information
        
    Returns
    -------
    str
        Path to output file
    """
    import pandas as pd
    import numpy as np
    import json
    from scipy import interpolate
    from netCDF4 import Dataset, stringtochar
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read station information
    df = pd.read_csv(station_info_file, index_col=[0], sep=';')
    station_info = df['station_info']
    lon = df['lon']
    lat = df['lat']
    nstation = len(station_info)
    namelen = 50
    
    # Read variable definitions
    with open(json_file) as d:
        var_dict = json.load(d)
    
    # Output file path
    output_file = f"{output_dir}/staout_timeseries_{date.strftime('%Y-%m-%d-%H')}.nc"
    
    # Create NetCDF file
    with Dataset(output_file, "w", format="NETCDF4") as fout:
        # Process each variable
        for ivar, var in enumerate(var_dict):
            # Read model output
            staout_fname = var_dict[var]['staout_fname']
            data = np.loadtxt(f"{input_dir}/{staout_fname}")
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
            
            # Add dimensions and common variables on first variable
            if ivar == 0:
                # Create dimensions
                fout.createDimension('station', nstation)
                fout.createDimension('namelen', namelen)
                fout.createDimension('time', None)
                
                # Create time variable
                add_variable(fout, 'time', 'f8', ('time',),
                            long_name="Time",
                            units=f'seconds since {date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:00:00 UTC',
                            base_date=f'{date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:00:00 UTC',
                            standard_name="time",
                            data=t_interp)
                
                # Create station_name variable
                names = np.empty((nstation,), 'S' + repr(namelen))
                for i in range(nstation):
                    names[i] = str(station_info[i])
                namesc = stringtochar(names)
                
                var_obj = fout.createVariable('station_name', 'c', ('station', 'namelen',))
                var_obj.long_name = "station name"
                var_obj[:] = namesc
                
                # Create coordinate variables
                add_variable(fout, 'x', 'f8', ('station',),
                            long_name="longitude",
                            standard_name="longitude",
                            units="degrees_east",
                            positive="east",
                            data=lon)
                
                add_variable(fout, 'y', 'f8', ('station',),
                            long_name="latitude",
                            standard_name="latitude",
                            units="degrees_north",
                            positive="north",
                            data=lat)
                
                # Add global attributes
                fout.title = 'SCHISM Model output'
                fout.source = 'SCHISM model output version v10'
                fout.references = 'http://ccrm.vims.edu/schismweb/'
            
            # Create variable for current field
            # Create variable for current field
            out_var = var_dict[var]['name']
            add_variable(fout, out_var, 'f8', ('time', 'station',),
                        long_name=var_dict[var]['long_name'],
                        standard_name=var_dict[var]['stardard_name'],
                        units=var_dict[var]['units'],
                        fill_value=-99999.,
                        data=model)
    
    return output_file


def extract_variable(ds, var_name, indices=None):
    """
    Extract a variable from a NetCDF dataset with optional indexing.
    
    Parameters
    ----------
    ds : netCDF4.Dataset or object
        Dataset to extract from
    var_name : str
        Variable name to extract
    indices : tuple or None, optional
        Indices for slicing the variable (e.g., (slice(0,10), 5, slice(None)))
        
    Returns
    -------
    np.ndarray
        Extracted variable data
    """
    if isinstance(ds, Dataset):
        var = ds.variables[var_name]
        if indices is not None:
            return var[indices]
        else:
            return var[:]
    else:
        # Handle custom object format
        if hasattr(ds, var_name):
            var = getattr(ds, var_name)
            if hasattr(var, 'val'):
                data = var.val
            else:
                data = var
                
            if indices is not None:
                return data[indices]
            else:
                return data
        else:
            raise ValueError(f"Variable {var_name} not found in dataset")


def get_netcdf_dims(ds):
    """
    Get dimension information from a NetCDF dataset.
    
    Parameters
    ----------
    ds : netCDF4.Dataset or object
        Dataset to query
        
    Returns
    -------
    dict
        Dictionary of dimension names and sizes
    """
    if isinstance(ds, Dataset):
        return {dim: ds.dimensions[dim].size for dim in ds.dimensions}
    else:
        # Handle custom object format
        if hasattr(ds, 'dimname') and hasattr(ds, 'dims'):
            return dict(zip(ds.dimname, ds.dims))
        else:
            raise ValueError("Cannot determine dimensions from dataset")


def create_combined_netcdf(output_file, input_files, var_names=None):
    """
    Combine multiple NetCDF files into a single file.
    
    Parameters
    ----------
    output_file : str
        Path to output file
    input_files : list
        List of input file paths
    var_names : list or None, optional
        Variables to include (None for all variables)
        
    Returns
    -------
    str
        Path to output file
    """
    # Open first file to get structure
    ds0 = Dataset(input_files[0])
    
    # Determine variables to include
    if var_names is None:
        var_names = list(ds0.variables.keys())
    
    # Create output file
    with Dataset(output_file, 'w', format=ds0.file_format) as ds_out:
        # Copy global attributes
        for attr in ds0.ncattrs():
            ds_out.setncattr(attr, ds0.getncattr(attr))
            
        # Create dimensions (assume time is unlimited)
        for dim_name, dim in ds0.dimensions.items():
            if dim.isunlimited():
                ds_out.createDimension(dim_name, None)
            else:
                ds_out.createDimension(dim_name, dim.size)
        
        # Create variables
        for var_name in var_names:
            if var_name in ds0.variables:
                var_in = ds0.variables[var_name]
                var_out = ds_out.createVariable(
                    var_name, 
                    var_in.dtype, 
                    var_in.dimensions,
                    zlib=True,
                    complevel=1
                )
                
                # Copy variable attributes
                for attr in var_in.ncattrs():
                    if attr != '_FillValue':  # Skip _FillValue as it's handled in createVariable
                        var_out.setncattr(attr, var_in.getncattr(attr))
        
        # Find time dimension and variable
        time_dim = None
        time_var = None
        for dim_name in ds0.dimensions:
            if 'time' in dim_name.lower() and ds0.dimensions[dim_name].isunlimited():
                time_dim = dim_name
                
        for var_name in ds0.variables:
            if 'time' in var_name.lower() and len(ds0.variables[var_name].dimensions) == 1:
                time_var = var_name
                break
        
        if time_dim is None or time_var is None:
            raise ValueError("Could not identify time dimension and variable")
            
        # Process each file
        time_index = 0
        for i, file_path in enumerate(input_files):
            print(f"Processing file {i+1}/{len(input_files)}: {file_path}")
            
            with Dataset(file_path) as ds_in:
                # Determine time size in this file
                time_size = ds_in.dimensions[time_dim].size
                
                # Copy each variable
                for var_name in var_names:
                    if var_name in ds_in.variables:
                        var_in = ds_in.variables[var_name]
                        var_out = ds_out.variables[var_name]
                        
                        # Handle based on dimensions
                        if time_dim in var_in.dimensions:
                            # Time-dependent variable
                            dim_index = var_in.dimensions.index(time_dim)
                            
                            # Create slices for each dimension
                            slices = [slice(None)] * len(var_in.dimensions)
                            out_slices = [slice(None)] * len(var_in.dimensions)
                            
                            # Set time slice
                            out_slices[dim_index] = slice(time_index, time_index + time_size)
                            
                            # Copy data
                            var_out[tuple(out_slices)] = var_in[tuple(slices)]
                        else:
                            # Non-time variable, only copy from first file
                            if i == 0:
                                var_out[:] = var_in[:]
                
                # Update time index
                time_index += time_size
    
    # Close input dataset
    ds0.close()
    
    return output_file


def get_netcdf_info(fname):
    """
    Get information about a NetCDF file without loading all the data.
    
    Parameters
    ----------
    fname : str
        Path to NetCDF file
        
    Returns
    -------
    dict
        Dictionary with file information
    """
    with Dataset(fname) as ds:
        # Create info dictionary
        info = {
            'format': ds.file_format,
            'dimensions': {dim: {
                'size': ds.dimensions[dim].size,
                'unlimited': ds.dimensions[dim].isunlimited()
            } for dim in ds.dimensions},
            'variables': {},
            'global_attrs': {attr: ds.getncattr(attr) for attr in ds.ncattrs()}
        }
        
        # Get variable info
        for var_name in ds.variables:
            var = ds.variables[var_name]
            info['variables'][var_name] = {
                'dimensions': var.dimensions,
                'shape': var.shape,
                'dtype': str(var.dtype),
                'attributes': {attr: var.getncattr(attr) for attr in var.ncattrs()
                              if attr != '_FillValue'}
            }
            
            # Add _FillValue separately if it exists
            if hasattr(var, '_FillValue'):
                info['variables'][var_name]['attributes']['_FillValue'] = var._FillValue
    
    return info


def compare_netcdf_files(file1, file2, rtol=1e-5, atol=1e-8):
    """
    Compare two NetCDF files for equality.
    
    Parameters
    ----------
    file1 : str
        Path to first NetCDF file
    file2 : str
        Path to second NetCDF file
    rtol : float, optional
        Relative tolerance for numerical comparison
    atol : float, optional
        Absolute tolerance for numerical comparison
        
    Returns
    -------
    bool
        True if files are equivalent, False otherwise
    tuple
        Differences details if files differ
    """
    with Dataset(file1) as ds1, Dataset(file2) as ds2:
        # Compare dimensions
        dims1 = set(ds1.dimensions.keys())
        dims2 = set(ds2.dimensions.keys())
        
        if dims1 != dims2:
            return False, f"Dimension mismatch: {dims1.symmetric_difference(dims2)}"
        
        for dim in dims1:
            if ds1.dimensions[dim].size != ds2.dimensions[dim].size:
                return False, f"Dimension size mismatch for {dim}: {ds1.dimensions[dim].size} vs {ds2.dimensions[dim].size}"
        
        # Compare variables
        vars1 = set(ds1.variables.keys())
        vars2 = set(ds2.variables.keys())
        
        if vars1 != vars2:
            return False, f"Variable mismatch: {vars1.symmetric_difference(vars2)}"
        
        for var_name in vars1:
            var1 = ds1.variables[var_name]
            var2 = ds2.variables[var_name]
            
            # Compare dimensions
            if var1.dimensions != var2.dimensions:
                return False, f"Variable dimension mismatch for {var_name}: {var1.dimensions} vs {var2.dimensions}"
            
            # Compare data (accounting for NaN values)
            data1 = var1[:]
            data2 = var2[:]
            
            if data1.shape != data2.shape:
                return False, f"Variable shape mismatch for {var_name}: {data1.shape} vs {data2.shape}"
            
            # Compare non-NaN values
            if hasattr(data1, 'mask') and hasattr(data2, 'mask'):
                # For masked arrays
                mask = data1.mask | data2.mask
                data1_valid = data1.data[~mask]
                data2_valid = data2.data[~mask]
            else:
                # For regular arrays
                nan_mask = np.isnan(data1) | np.isnan(data2)
                if np.any(nan_mask):
                    data1_valid = data1[~nan_mask]
                    data2_valid = data2[~nan_mask]
                else:
                    data1_valid = data1
                    data2_valid = data2
            
            if not np.allclose(data1_valid, data2_valid, rtol=rtol, atol=atol):
                diff = np.abs(data1_valid - data2_valid)
                max_diff = np.max(diff)
                return False, f"Variable data mismatch for {var_name}: max diff = {max_diff}"
    
    return True, "Files are equivalent"
