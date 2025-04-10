"""
River processing utilities for STOFS3D

Contains functions for processing river data, including discharge calculations,
flux timeseries generation, and temperature modeling for rivers like St. Lawrence.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz


def get_river_discharge(fname, datevectors, datevectors2=None):
    """
    Extract river discharge data from CSV file.
    
    Parameters
    ----------
    fname : str
        Path to CSV file with river discharge data
    datevectors : pd.DatetimeIndex
        Date range for which data is required
    datevectors2 : pd.DatetimeIndex, optional
        Extended date range for extrapolation
        
    Returns
    -------
    list
        Discharge values for each date in datevectors and datevectors2
    """
    # Read CSV file
    df = pd.read_csv(fname, sep=',', na_values='')
    
    # Clean up columns
    df.drop(df.columns[[0, 2, 3, 4, 5, 7, 8, 9]], axis=1, inplace=True)
    df.rename(columns={df.columns[0]: 'date_local', df.columns[1]: 'flow'}, inplace=True)
    
    # Parse dates and convert to UTC
    ts = pd.to_datetime(pd.Series(df['date_local'].values))
    ts3 = ts.dt.tz_convert('UTC')
    
    # Add UTC dates to dataframe
    df.insert(1, 'date_utc', ts3)
    df.set_index('date_utc', inplace=True)
    
    # Extract data for each date
    data = []
    for i, dt in enumerate(datevectors):
        print(f'Getting data for day {i+1}:')
        try:
            data.append(round(float(df.loc[dt]['flow']), 3))
        except KeyError:
            if i == 0:
                raise KeyError(f'No discharge data for hindcast {dt}, use old flux.th!')
            else:
                print(f"No discharge data for {dt}, use the previous day's data!")
                data.append(data[-1])
    
    # Extend for forecast period if datevectors2 provided
    if datevectors2 is not None:
        for dt in datevectors2[len(datevectors):]:
            data.append(data[-1])
    
    return data


def generate_flux_timeseries(date, rivers, river_files, output_file='flux.th'):
    """
    Generate flux.th time history file with river discharges.
    
    Parameters
    ----------
    date : datetime
        Start date for the time series
    rivers : list
        List of river names
    river_files : dict
        Dictionary mapping river names to discharge data files
    output_file : str, optional
        Output filename (default: 'flux.th')
        
    Returns
    -------
    str
        Path to output file
    """
    # Generate date ranges
    enddate = date + timedelta(days=1)
    datevectors = pd.date_range(
        start=date.strftime('%Y-%m-%d %H:00:00'), 
        end=enddate.strftime('%Y-%m-%d %H:00:00'), 
        tz='UTC'
    )
    
    enddate2 = date + timedelta(days=6)
    datevectors2 = pd.date_range(
        start=date.strftime('%Y-%m-%d %H:00:00'), 
        end=enddate2.strftime('%Y-%m-%d %H:00:00'), 
        tz='UTC'
    )
    
    # Get discharge for each river
    flow = {}
    for river in rivers:
        if river in river_files:
            flow[river] = get_river_discharge(
                river_files[river], 
                datevectors, 
                datevectors2
            )
        else:
            raise ValueError(f"No discharge file defined for {river}")
    
    # Create output data
    data = []
    for i, dt in enumerate(datevectors2):
        line = []
        time_seconds = (dt - datevectors[0]).total_seconds()
        print(f'time = {time_seconds}')
        line.append(time_seconds)
        
        for river in rivers:
            line.append(-flow[river][i])  # Negative for source
        
        data.append(line)
    
    # Write file
    np.savetxt(output_file, np.array(data), fmt='%.3f')
    
    return output_file


def get_river_temperature(date, air_temp_file, rivers, output_file='TEM_1.th'):
    """
    Generate temperature time history file for rivers.
    
    For St. Lawrence River, uses a linear regression with air temperature.
    
    Parameters
    ----------
    date : datetime
        Start date for the time series
    air_temp_file : str
        Path to netCDF file with air temperature data
    rivers : list
        List of river names
    output_file : str, optional
        Output filename (default: 'TEM_1.th')
        
    Returns
    -------
    str
        Path to output file
    """
    from netCDF4 import Dataset
    
    # Generate date ranges
    enddate = date + timedelta(days=1)
    datevectors = pd.date_range(
        start=date.strftime('%Y-%m-%d %H:00:00'), 
        end=enddate.strftime('%Y-%m-%d %H:00:00'), 
        tz='UTC'
    )
    
    enddate2 = date + timedelta(days=5)
    datevectors2 = pd.date_range(
        start=date.strftime('%Y-%m-%d %H:00:00'), 
        end=enddate2.strftime('%Y-%m-%d %H:00:00'), 
        tz='UTC'
    )
    
    # Initialize temperature dictionary
    temp = {}
    
    # St Lawrence river - linear regression with airT (y=0.83x+2.817)
    point = (45.415, -73.623056)  # Reference location
    
    # Read air temperature data
    ds = Dataset(air_temp_file)
    
    # Extract coordinates
    lon = ds['lon'][0, :]
    lat = ds['lat'][:, 0]
    
    # Find nearby grid point
    idxs = ((lat - point[0]) > 0) & ((lat - point[0]) < 0.2)
    lat_idx = np.where(idxs)[0]
    
    idxs = ((lon - point[1]) > 0) & ((lon - point[1]) < 0.2)
    lon_idx = np.where(idxs)[0]
    
    # Get time information
    times = ds['time'][:]
    sflux_startdate = pd.Timestamp(ds['time'].units.split('since ')[-1], tz='UTC')
    
    # Convert time to timestamps
    timestamps = [sflux_startdate + timedelta(seconds=round(dt*86400)) for dt in times]
    
    # Extract air temperature and convert to water temperature
    airT = np.squeeze(ds['stmp'][:, lat_idx, lon_idx] - 273.15)  # Convert K to C
    waterT = 0.83 * airT + 2.817
    
    # Set minimum water temperature to zero
    idxs = waterT < 0
    waterT[idxs] = 0
    
    # Create dataframe and resample to hourly
    df = pd.DataFrame(waterT, index=timestamps)
    df2 = df.resample('h').mean()
    df2.fillna(method='bfill', inplace=True)
    
    # Get values for specified dates
    timestamps = df2.index
    indices = np.where(np.isin(timestamps, datevectors2))[0]
    
    # Store temperature values
    for river in rivers:
        temp[river] = df2.iloc[np.array(indices)][0].values
    
    # Create output data
    data = []
    for i, dt in enumerate(datevectors2):
        line = []
        time_seconds = (dt - datevectors[0]).total_seconds()
        print(f'time = {time_seconds}')
        
        line.append(time_seconds)
        for river in rivers:
            line.append(temp[river][i])
        
        data.append(line)
    
    # Write file
    np.savetxt(output_file, np.array(data), fmt='%.3f')
    
    # Close dataset
    ds.close()
    
    return output_file


def generate_river_input(date, config, output_dir='.'):
    """
    Generate all river input files (flux.th, TEM_1.th) for SCHISM.
    
    Parameters
    ----------
    date : datetime
        Start date
    config : dict
        Configuration dictionary with:
        - rivers: List of river names
        - discharge_files: Dict of river discharge files
        - air_temp_file: Path to air temperature file
    output_dir : str, optional
        Output directory (default: '.')
        
    Returns
    -------
    dict
        Paths to generated files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate flux.th
    flux_file = os.path.join(output_dir, 'flux.th')
    generate_flux_timeseries(
        date, 
        config['rivers'], 
        config['discharge_files'], 
        output_file=flux_file
    )
    
    # Generate TEM_1.th
    temp_file = os.path.join(output_dir, 'TEM_1.th')
    get_river_temperature(
        date, 
        config['air_temp_file'], 
        config['rivers'], 
        output_file=temp_file
    )
    
    return {
        'flux': flux_file,
        'temperature': temp_file
    }


def add_pump_to_sink(sinks, pump):
    """
    Add pump flows to sink data.
    
    Parameters
    ----------
    sinks : list
        List of sink flow data
    pump : numpy.ndarray
        Pump flow data
        
    Returns
    -------
    list
        Updated sink flow data
    """
    sinks_all = []
    for row in sinks:
        row.extend(pump.tolist())
        sinks_all.append(row)
        
    return sinks_all


def write_th_file(dataset, timeinterval, fname, issource=True):
    """
    Write time history file.
    
    Parameters
    ----------
    dataset : list or numpy.ndarray
        Flow values
    timeinterval : list or numpy.ndarray
        Time values in seconds
    fname : str
        Output filename
    issource : bool, optional
        True for source, False for sink (default: True)
        
    Returns
    -------
    str
        Path to output file
    """
    data = []
    for values, interval in zip(dataset, timeinterval):
        if issource:
            data.append(" ".join([f"{interval:G}", *[f'{x: .4f}' for x in values], '\n']))
        else:
            data.append(" ".join([f"{interval:G}", *[f'{-x: .4f}' for x in values], '\n']))
            
    with open(fname, 'w+') as fid:
        fid.writelines(data)
        
    return fname
