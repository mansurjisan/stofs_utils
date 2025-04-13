"""
Station data processing module for STOFS3D

Contains functions for reading, processing, and generating station outputs
from SCHISM model results.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from netCDF4 import Dataset, stringtochar
from scipy import interpolate
import json
import time

from ..core.grid import SchismGrid
from ..core.vertical_grid import SchismVGrid, compute_zcor
from ..utils.helpers import ensure_directory


def read_station_file(station_file):
    """
    Read a SCHISM station file.
    
    Parameters
    ----------
    station_file : str
        Path to station file
        
    Returns
    -------
    tuple
        (lon, lat, station_names) - Coordinates and station names
    """
    with open(station_file) as f:
        f.readline()  # Skip first line
        f.readline()  # Skip second line
        
        station_name = []
        lon = []
        lat = []
        
        for line in f.read().splitlines():
            if '!' in line:
                station_name.append(line.split('!')[-1])
                lon.append(line.split(' ')[1])
                lat.append(line.split(' ')[2])
    
    return np.array(lon).astype('float'), np.array(lat).astype('float'), np.array(station_name)


def generate_station_timeseries(date, input_dir, output_dir, 
                              station_info_file='stofs_3d_atl_staout_nc.csv',
                              json_file='stofs_3d_atl_staout_nc.json'):
    """
    Generate timeseries at observation locations from SCHISM staout files.
    
    Parameters
    ----------
    date : datetime
        Start date for timeseries
    input_dir : str
        Directory containing staout files
    output_dir : str 
        Directory to save output netCDF
    station_info_file : str, optional
        CSV file with station information
    json_file : str, optional
        JSON file with variable information
        
    Returns
    -------
    str
        Path to output file
    """
    # Make sure output directory exists
    ensure_directory(output_dir)
    
    # Read station information
    df = pd.read_csv(station_info_file, index_col=[0], sep=';')
    station_info = df['station_info']
    lon = df['lon']
    lat = df['lat']
    nstation = len(station_info)
    namelen = 50
    
    # Read variable definitions from JSON
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
                # Dimensions
                fout.createDimension('station', nstation)
                fout.createDimension('namelen', namelen)
                fout.createDimension('time', None)
                
                # Time variable
                time_var = fout.createVariable('time', 'f8', ('time',))
                time_var.long_name = "Time"
                time_var.units = f'seconds since {date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:00:00 UTC'
                time_var.base_date = f'{date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:00:00 UTC'
                time_var.standard_name = "time"
                time_var[:] = t_interp
                
                # Station name variable
                station_var = fout.createVariable('station_name', 'c', ('station','namelen',))
                station_var.long_name = "station name"
                
                # Create string array for station names
                names = np.empty((nstation,), 'S'+repr(namelen))
                for i in range(nstation):
                    names[i] = str(station_info[i])
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
                fout.title = 'SCHISM Model output'
                fout.source = 'SCHISM model output version v10'
                fout.references = 'http://ccrm.vims.edu/schismweb/'
            
            # Create variable for current field
            out_var = var_dict[var]['name']
            var_out = fout.createVariable(out_var, 'f8', ('time', 'station',), fill_value=-99999.)
            var_out.long_name = var_dict[var]['long_name']
            var_out.standard_name = var_dict[var]['stardard_name']
            var_out.units = var_dict[var]['units']
            var_out[:, :] = model
    
    return output_file


def get_stations_profile(date, stack_start, stack_end, output_dir, 
                        run_dir='.', station_file='station.in'):
    """
    Extract vertical profiles at station locations from SCHISM output.
    
    Parameters
    ----------
    date : datetime
        Start date
    stack_start : int
        Starting stack number
    stack_end : int
        Ending stack number
    output_dir : str
        Output directory
    run_dir : str, optional
        SCHISM run directory (default: current directory)
    station_file : str, optional
        Station file name (default: 'station.in')
        
    Returns
    -------
    str
        Path to output file
    """
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Load grid information
    grid_npz = f'{run_dir}/grid.npz'
    if os.path.exists(grid_npz):
        from ..io.grid_io import load_grid
        data = load_grid(grid_npz)
        hgrid = data.hgrid
        vgrid = data.vgrid
    else:
        # Create grid from scratch
        hgrid = SchismGrid(f'{run_dir}/hgrid.gr3')
        vgrid = SchismVGrid()
        vgrid.read_vgrid(f'{run_dir}/vgrid.in')
        
        # Save grid for future use
        from ..io.grid_io import save_grid
        save_grid(hgrid, vgrid, fname='grid', path=run_dir)
    
    # Set outputs directory
    outputs_dir = f'{run_dir}/outputs'
    
    # Generate stacks list
    stacks = list(range(stack_start, stack_end + 1))
    print(f'Processing stacks: {stacks}')
    
    # Initialize variables
    times = []
    zeta = []
    uwind = []
    vwind = []
    salt = []
    temp = []
    uvel = []
    vvel = []
    zcor = []
    
    svars2d = {'elevation': zeta, 'windSpeedX': uwind, 'windSpeedY': vwind}
    svars3d = {'salinity': salt, 'temperature': temp, 'horizontalVelX': uvel, 
               'horizontalVelY': vvel, 'zCoordinates': zcor}
    
    # Read station information
    lon, lat, station_names = read_station_file(f'{run_dir}/{station_file}')
    nstation = len(station_names)
    
    # Compute area coordinate for stations
    station_points = np.c_[lon, lat]
    ie, ip, acor = hgrid.compute_acor(station_points)
    
    # Get station depths
    station_depths = hgrid.dp[ip]
    station_depths0 = np.sum(station_depths * acor, axis=1)
    
    # Get sigma information
    if vgrid.ivcor == 1:
        station_sigma = vgrid.sigma[ip]
        station_kbp = vgrid.kbp[ip]
    
    # Check if any points are outside grid
    pts_outside_grid = np.nonzero(ie == -1)[0]
    if len(pts_outside_grid) != 0:
        raise ValueError(f'Points outside domain: {station_points[pts_outside_grid]}')
    
    # Process each stack
    for i, istack in enumerate(stacks):
        print(f'Processing stack {istack}')
        
        # Get elevation
        ds2d = Dataset(f'{outputs_dir}/out2d_{istack}.nc')
        
        # Get times
        times2 = ds2d['time'][:]
        ntimes = len(times2)
        times.extend(times2)
        
        # Process 2D variables
        for it in range(ntimes):
            print(f'Processing time {it+1}/{ntimes} from stack {istack} for 2D variables')
            for var, value in svars2d.items():
                # Get values at station points
                trii = ds2d.variables[var][it][ip]
                tri = np.sum(trii * acor, axis=1)
                value.append(tri)
        
        ds2d.close()
        
        # Process 3D variables
        for var, value in svars3d.items():
            ds3d = Dataset(f'{outputs_dir}/{var}_{istack}.nc')
            ndims = ds3d.variables[var].ndim
            dimname = ds3d.variables[var].dimensions
            
            for it in range(ntimes):
                print(f'Processing time {it+1}/{ntimes} from stack {istack} for 3D variable {var}')
                
                # Get values at station points
                if 'nSCHISM_hgrid_node' in dimname:
                    trii = ds3d[var][it][ip]
                elif 'nSCHISM_hgrid_face' in dimname:
                    trii = ds3d[var][it][ie]
                else:
                    raise ValueError(f'Unknown variable format: {var}')
                
                # Horizontal interpolation
                if 'nSCHISM_hgrid_node' in dimname:
                    if ndims == 2:
                        tri = np.sum(trii * acor, axis=1)
                    elif ndims == 3:
                        tri = np.sum(trii * acor[..., None], axis=1)
                    elif ndims == 4:
                        tri = np.sum(trii * acor[..., None, None], axis=1)
                else:
                    tri = trii
                
                value.append(tri)
            
            ds3d.close()
    
    # Create output file
    output_file = f'{output_dir}/stofs_stations_forecast.nc'
    with Dataset(output_file, "w", format="NETCDF3_CLASSIC") as fout:
        # Create dimensions
        fout.createDimension('station', nstation)
        fout.createDimension('namelen', 100)
        fout.createDimension('siglay', vgrid.nvrt)
        fout.createDimension('time', None)
        
        # Create variables
        time_var = fout.createVariable('time', 'f4', ('time',))
        time_var.long_name = "Time"
        time_var.units = f'seconds since {date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:00:00 UTC'
        time_var.base_date = f'{date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:00:00 UTC'
        time_var.standard_name = "time"
        time_var[:] = times
        
        # Station names
        name_var = fout.createVariable('station_name', 'c', ('station', 'namelen',))
        name_var.long_name = "station name"
        
        names = np.empty((nstation,), 'S100')
        for i in range(nstation):
            names[i] = str(station_names[i])
        namesc = stringtochar(names)
        name_var[:] = namesc
        
        # Coordinates
        lon_var = fout.createVariable('lon', 'f4', ('station',))
        lon_var.long_name = "longitude"
        lon_var.standard_name = "longitude"
        lon_var.units = "degrees_east"
        lon_var.positive = "east"
        lon_var[:] = lon
        
        lat_var = fout.createVariable('lat', 'f4', ('station',))
        lat_var.long_name = "latitude"
        lat_var.standard_name = "latitude"
        lat_var.units = "degrees_north"
        lat_var.positive = "north"
        lat_var[:] = lat
        
        # Depth
        depth_var = fout.createVariable('depth', 'f4', ('station',))
        depth_var.long_name = "Bathymetry"
        depth_var.standard_name = "depth"
        depth_var.units = "meters"
        depth_var[:] = station_depths0
        
        # Water elevation
        zeta_var = fout.createVariable('zeta', 'f4', ('time', 'station',), fill_value=-99999.)
        zeta_var.long_name = "water surface elevation above navd88"
        zeta_var.standard_name = "sea_surface_height_above_navd88"
        zeta_var.units = "m"
        zeta_var[:, :] = np.array(zeta)
        
        # Z coordinates
        zcor_var = fout.createVariable('zCoordinates', 'f4', ('time', 'station', 'siglay',), fill_value=-99999.)
        zcor_var.long_name = "vertical coordinate, positive upward"
        zcor_var.standard_name = "vertical coordinate"
        zcor_var.units = "m"
        zcor_var[:, :, :] = np.array(zcor)
        
        # Salinity
        salt_var = fout.createVariable('salinity', 'f4', ('time', 'station', 'siglay',), fill_value=-99999.)
        salt_var.long_name = "salinity"
        salt_var.standard_name = "sea_water_salinity"
        salt_var.units = "psu"
        salt_var[:, :, :] = np.array(salt)
        
        # Temperature
        temp_var = fout.createVariable('temperature', 'f4', ('time', 'station', 'siglay',), fill_value=-99999.)
        temp_var.long_name = "temperature"
        temp_var.standard_name = "sea_water_temperature"
        temp_var.units = "degree_C"
        temp_var[:, :, :] = np.array(temp)
        
        # U velocity
        u_var = fout.createVariable('u', 'f4', ('time', 'station', 'siglay',), fill_value=-99999.)
        u_var.long_name = "Eastward Water Velocity"
        u_var.standard_name = "eastward_sea_water_velocity"
        u_var.units = "meters s-1"
        u_var[:, :, :] = np.array(uvel)
        
        # V velocity
        v_var = fout.createVariable('v', 'f4', ('time', 'station', 'siglay',), fill_value=-99999.)
        v_var.long_name = "Northward Water Velocity"
        v_var.standard_name = "northward_sea_water_velocity"
        v_var.units = "meters s-1"
        v_var[:, :, :] = np.array(vvel)
        
        # Wind
        uwind_var = fout.createVariable('uwind_speed', 'f4', ('time', 'station',), fill_value=-99999.)
        uwind_var.long_name = "Eastward Wind Velocity"
        uwind_var.standard_name = "eastward_wind"
        uwind_var.units = "meters s-1"
        uwind_var[:, :] = np.array(uwind)
        
        vwind_var = fout.createVariable('vwind_speed', 'f4', ('time', 'station',), fill_value=-99999.)
        vwind_var.long_name = "Northward Wind Velocity"
        vwind_var.standard_name = "northward_wind"
        vwind_var.units = "meters s-1"
        vwind_var[:, :] = np.array(vwind)
        
        # Global attributes
        fout.title = 'SCHISM Model output'
        fout.references = 'http://ccrm.vims.edu/schismweb/'
    
    print(f'Station profile data saved to: {output_file}')
    return output_file


def stations_cli():
    """
    Command-line interface for the station data processing.
    
    This function handles command-line arguments for the station data processing utilities.
    Run from command line with: python -m stofs_utils.processing.stations [arguments]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Process SCHISM station output data')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')
    
    # Parser for generating timeseries
    parser_ts = subparsers.add_parser('timeseries', help='Generate station timeseries')
    parser_ts.add_argument('--date', type=lambda s: datetime.strptime(s, '%Y-%m-%d-%H'),
                          required=True, help='Start date (YYYY-MM-DD-HH)')
    parser_ts.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser_ts.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser_ts.add_argument('--station_info', type=str, default='stofs_3d_atl_staout_nc.csv',
                         help='Station info CSV file (default: stofs_3d_atl_staout_nc.csv)')
    parser_ts.add_argument('--json_file', type=str, default='stofs_3d_atl_staout_nc.json',
                         help='Variable info JSON file (default: stofs_3d_atl_staout_nc.json)')
    
    # Parser for generating profiles
    parser_prof = subparsers.add_parser('profile', help='Generate station profiles')
    parser_prof.add_argument('--date', type=lambda s: datetime.strptime(s, '%Y-%m-%d-%H'),
                          required=True, help='Start date (YYYY-MM-DD-HH)')
    parser_prof.add_argument('--stack_start', type=int, required=True, help='Starting stack number')
    parser_prof.add_argument('--stack_end', type=int, required=True, help='Ending stack number')
    parser_prof.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser_prof.add_argument('--run_dir', type=str, default='.', help='SCHISM run directory')
    parser_prof.add_argument('--station_file', type=str, default='station.in',
                          help='Station file name (default: station.in)')
    
    args = parser.parse_args()
    
    if args.command == 'timeseries':
        output_file = generate_station_timeseries(
            date=args.date,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            station_info_file=args.station_info,
            json_file=args.json_file
        )
        print(f"Station timeseries file created: {output_file}")
        
    elif args.command == 'profile':
        output_file = get_stations_profile(
            date=args.date,
            stack_start=args.stack_start,
            stack_end=args.stack_end,
            output_dir=args.output_dir,
            run_dir=args.run_dir,
            station_file=args.station_file
        )
        print(f"Station profile file created: {output_file}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    stations_cli()