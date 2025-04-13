"""
ASCII file I/O utilities for STOFS3D

Contains functions for reading and writing various ASCII file formats
used in STOFS3D, including time-history (.th) files, property files,
and other text-based formats.
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime


def read_th_file(filename, header=False):
    """
    Read a SCHISM time-history (.th) file.
    
    Parameters
    ----------
    filename : str
        Path to .th file
    header : bool, optional
        Whether the file has a header (default: False)
        
    Returns
    -------
    tuple
        (times, values) - Time values and data values
    """
    # Determine number of header lines
    skip_rows = 1 if header else 0
    
    # Read the file
    data = np.loadtxt(filename, skiprows=skip_rows)
    
    # Extract times and values
    times = data[:, 0]
    values = data[:, 1:]
    
    return times, values


def write_th_file(times, values, filename, header=None, fmt=None):
    """
    Write a SCHISM time-history (.th) file.
    
    Parameters
    ----------
    times : numpy.ndarray
        Time values
    values : numpy.ndarray
        Data values
    filename : str
        Output file path
    header : str, optional
        Header string to include (default: None)
    fmt : str, optional
        Format string for values (default: '%.4f')
        
    Returns
    -------
    str
        Path to output file
    """
    if fmt is None:
        fmt = '%.4f'
    
    # Combine times and values
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    
    data = np.column_stack((times, values))
    
    # Write file
    if header is not None:
        with open(filename, 'w') as f:
            f.write(f"{header}\n")
            np.savetxt(f, data, fmt=fmt)
    else:
        np.savetxt(filename, data, fmt=fmt)
    
    return filename


def read_prop_file(filename):
    """
    Read a SCHISM property file.
    
    Parameters
    ----------
    filename : str
        Path to property file
        
    Returns
    -------
    numpy.ndarray
        Property values
    """
    data = np.loadtxt(filename)
    
    if data.ndim == 2:
        return data[:, 1]
    else:
        return np.array([data[1]])


def write_prop_file(values, filename, fmt=None):
    """
    Write a SCHISM property file.
    
    Parameters
    ----------
    values : numpy.ndarray
        Property values
    filename : str
        Output file path
    fmt : str, optional
        Format string for values (default: '%8.5f')
        
    Returns
    -------
    str
        Path to output file
    """
    if fmt is None:
        fmt = '%8.5f'
    
    # Create index and value pairs
    indices = np.arange(1, len(values) + 1)
    data = np.column_stack((indices, values))
    
    # Write file
    np.savetxt(filename, data, fmt=["%d", fmt])
    
    return filename


def read_station_file(filename):
    """
    Read a SCHISM station file.
    
    Parameters
    ----------
    filename : str
        Path to station file
        
    Returns
    -------
    tuple
        (x, y, station_names) - Coordinates and station names
    """
    with open(filename) as f:
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


def write_station_file(x, y, z, station_names, filename):
    """
    Write a SCHISM station file.
    
    Parameters
    ----------
    x : numpy.ndarray
        X coordinates
    y : numpy.ndarray
        Y coordinates
    z : numpy.ndarray
        Z coordinates
    station_names : list
        Station names
    filename : str
        Output file path
        
    Returns
    -------
    str
        Path to output file
    """
    with open(filename, 'w') as f:
        f.write('ACE/gredit: station locations\n')
        f.write(f'{len(x)}\n')
        
        for i in range(len(x)):
            f.write(f'{i+1} {x[i]:.8f} {y[i]:.8f} {z[i]:.8f} !{station_names[i]}\n')
    
    return filename


def read_param_file(filename, format=None):
    """
    Read a SCHISM parameter file.
    
    Parameters
    ----------
    filename : str
        Path to parameter file
    format : int, optional
        Output format:
        - None: Determine automatically (default)
        - 0: Return values as strings
        - 1: Convert values to float or int when possible
        
    Returns
    -------
    dict
        Parameter name-value pairs
    """
    # Read all lines
    with open(filename, 'r') as f:
        lines = [i.strip() for i in f.readlines()]
    
    # Parse each line with assignment
    lines = [i for i in lines if ('=' in i) and (i != '') and (i[0] != '!') and (i[0] != '&')]
    
    # Create parameter dictionary
    param = {}
    for line in lines:
        # Remove comments
        if '!' in line:
            line = line[:line.find('!')]
        
        # Split and clean
        key, val = line.split('=')
        key = key.strip()
        val = val.strip()
        
        # Convert values if requested
        if format == 1 and val.lstrip('-').replace('.', '', 1).isdigit():
            val = float(val) if ('.' in val) else int(val)
        
        param[key] = val
    
    return param


def write_param_file(params, filename):
    """
    Write a SCHISM parameter file.
    
    Parameters
    ----------
    params : dict
        Parameter name-value pairs
    filename : str
        Output file path
        
    Returns
    -------
    str
        Path to output file
    """
    with open(filename, 'w+') as f:
        for key in sorted(params.keys()):
            f.write(f'{key:10}= {params[key]}\n')
    
    return filename


def read_featureID_file(filename):
    """
    Read NWM feature IDs from a file.
    
    Parameters
    ----------
    filename : str
        Path to feature ID file
        
    Returns
    -------
    list
        List of feature IDs
    """
    with open(filename) as f:
        lines = f.readlines()
        feature_ids = []
        for line in lines:
            feature_ids.append(line.split('\n')[0])
    
    return feature_ids


def write_featureID_file(feature_ids, filename):
    """
    Write NWM feature IDs to a file.
    
    Parameters
    ----------
    feature_ids : list
        List of feature IDs
    filename : str
        Output file path
        
    Returns
    -------
    str
        Path to output file
    """
    with open(filename, 'w') as f:
        for fid in feature_ids:
            f.write(f"{fid}\n")
    
    return filename


def read_flux_th(filename):
    """
    Read a SCHISM flux.th file.
    
    Parameters
    ----------
    filename : str
        Path to flux.th file
        
    Returns
    -------
    tuple
        (times, flows) - Times and flow values
    """
    data = np.loadtxt(filename)
    times = data[:, 0]
    flows = data[:, 1:]
    
    return times, flows


def write_flux_th(times, flows, filename):
    """
    Write a SCHISM flux.th file.
    
    Parameters
    ----------
    times : numpy.ndarray
        Time values in seconds
    flows : numpy.ndarray
        Flow values
    filename : str
        Output file path
        
    Returns
    -------
    str
        Path to output file
    """
    data = np.column_stack((times, flows))
    np.savetxt(filename, data, fmt='%.3f')
    
    return filename


def read_msource_th(filename):
    """
    Read a SCHISM msource.th file.
    
    Parameters
    ----------
    filename : str
        Path to msource.th file
        
    Returns
    -------
    tuple
        (times, concentrations, temperatures) - Times, concentration and temperature values
    """
    data = np.loadtxt(filename, dtype=int)
    times = data[:, 0]
    n_sources = (data.shape[1] - 1) // 2
    
    concentrations = data[:, 1:n_sources+1]
    temperatures = data[:, n_sources+1:]
    
    return times, concentrations, temperatures


def write_msource_th(times, concentrations, temperatures, filename):
    """
    Write a SCHISM msource.th file.
    
    Parameters
    ----------
    times : numpy.ndarray
        Time values in seconds
    concentrations : numpy.ndarray
        Concentration values
    temperatures : numpy.ndarray
        Temperature values
    filename : str
        Output file path
        
    Returns
    -------
    str
        Path to output file
    """
    data = np.column_stack((times, concentrations, temperatures))
    np.savetxt(filename, data, fmt='%d', delimiter=' ')
    
    return filename


def read_hgrid_ll(filename):
    """
    Read a SCHISM hgrid.ll file.
    
    Parameters
    ----------
    filename : str
        Path to hgrid.ll file
        
    Returns
    -------
    tuple
        (lon, lat, dp, elnode, i34) - Coordinates, depths, element connectivity, element types
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Read header
    ne, np_points = map(int, lines[1].split()[0:2])
    
    # Read node information
    lon = np.zeros(np_points)
    lat = np.zeros(np_points)
    dp = np.zeros(np_points)
    
    for i in range(np_points):
        parts = lines[2+i].split()
        lon[i] = float(parts[1])
        lat[i] = float(parts[2])
        dp[i] = float(parts[3])
    
    # Read element connectivity
    elnode = np.zeros((ne, 4), dtype=int) - 1
    i34 = np.zeros(ne, dtype=int)
    
    for i in range(ne):
        parts = lines[2+np_points+i].split()
        i34[i] = int(parts[1])
        
        for j in range(i34[i]):
            elnode[i, j] = int(parts[2+j]) - 1  # Convert to 0-based indexing
    
    return lon, lat, dp, elnode, i34


def write_hgrid_ll(lon, lat, dp, elnode, i34, filename, info=None):
    """
    Write a SCHISM hgrid.ll file.
    
    Parameters
    ----------
    lon : numpy.ndarray
        Longitude coordinates
    lat : numpy.ndarray
        Latitude coordinates
    dp : numpy.ndarray
        Depths
    elnode : numpy.ndarray
        Element connectivity
    i34 : numpy.ndarray
        Element types
    filename : str
        Output file path
    info : str, optional
        Grid information text (default: None)
        
    Returns
    -------
    str
        Path to output file
    """
    ne = len(i34)
    np_points = len(lon)
    
    with open(filename, 'w+') as f:
        # Write header
        if info is None:
            f.write('!grd info\n')
        else:
            f.write(f'!grd info:{info}\n')
            
        f.write(f'{ne} {np_points}\n')
        
        # Write node information
        for i in range(np_points):
            f.write(f'{i+1:<d} {lon[i]:<.8f} {lat[i]:<.8f} {dp[i]:<.8f}\n')
        
        # Write element connectivity
        for i in range(ne):
            if i34[i] == 3:
                f.write(f'{i+1:<d} {i34[i]:<d} {elnode[i,0]+1:<d} {elnode[i,1]+1:<d} {elnode[i,2]+1:<d}\n')
            elif i34[i] == 4:
                f.write(f'{i+1:<d} {i34[i]:<d} {elnode[i,0]+1:<d} {elnode[i,1]+1:<d} {elnode[i,2]+1:<d} {elnode[i,3]+1:<d}\n')
    
    return filename


def read_vgrid(filename):
    """
    Read a SCHISM vgrid.in file.
    
    Parameters
    ----------
    filename : str
        Path to vgrid.in file
        
    Returns
    -------
    dict
        Vertical grid information
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    vgrid = {}
    vgrid['ivcor'] = int(lines[0].strip().split()[0])
    vgrid['nvrt'] = int(lines[1].strip().split()[0])
    
    if vgrid['ivcor'] == 1:
        # Sigma grid
        lines = lines[2:]
        sline = np.array(lines[0].split()).astype('float')
        
        if sline.min() < 0:  # Old format
            vgrid['kbp'] = np.array([int(i.split()[1])-1 for i in lines])
            vgrid['np'] = len(vgrid['kbp'])
            vgrid['sigma'] = -np.ones([vgrid['np'], vgrid['nvrt']])
            
            for i, line in enumerate(lines):
                vgrid['sigma'][i, vgrid['kbp'][i]:] = np.array(line.strip().split()[2:]).astype('float')
        else:  # New format
            sline = sline.astype('int')
            vgrid['kbp'] = sline - 1
            vgrid['np'] = len(sline)
            vgrid['sigma'] = np.array([i.split()[1:] for i in lines[1:]]).T.astype('float')
            fpm = vgrid['sigma'] < -1
            vgrid['sigma'][fpm] = -1
    
    elif vgrid['ivcor'] == 2:
        # SZ grid
        vgrid['kz'], vgrid['h_s'] = lines[1].strip().split()[1:3]
        vgrid['kz'] = int(vgrid['kz'])
        vgrid['h_s'] = float(vgrid['h_s'])
        
        # Read z grid
        vgrid['ztot'] = []
        irec = 2
        for i in range(vgrid['kz']):
            irec = irec + 1
            vgrid['ztot'].append(lines[irec].strip().split()[1])
        vgrid['ztot'] = np.array(vgrid['ztot']).astype('float')
        
        # Read s grid
        vgrid['sigma'] = []
        irec = irec + 2
        vgrid['nsig'] = vgrid['nvrt'] - vgrid['kz'] + 1
        vgrid['h_c'], vgrid['theta_b'], vgrid['theta_f'] = np.array(lines[irec].strip().split()[:3]).astype('float')
        
        for i in range(vgrid['nsig']):
            irec = irec + 1
            vgrid['sigma'].append(lines[irec].strip().split()[1])
        vgrid['sigma'] = np.array(vgrid['sigma']).astype('float')
    
    return vgrid


def write_vgrid(vgrid, filename):
    """
    Write a SCHISM vgrid.in file.
    
    Parameters
    ----------
    vgrid : dict
        Vertical grid information
    filename : str
        Output file path
        
    Returns
    -------
    str
        Path to output file
    """
    with open(filename, 'w+') as f:
        f.write(f"{vgrid['ivcor']}    !ivcor\n")
        f.write(f"{vgrid['nvrt']}  \n")
        
        if vgrid['ivcor'] == 1:
            # Sigma grid
            for i in range(vgrid['np']):
                vgrid['sigma'][i, :vgrid['kbp'][i]] = -9
            
            fstr = '    ' + ' {:10d}' * vgrid['np'] + '\n'
            kbp = vgrid['kbp'] + 1
            f.write(fstr.format(*kbp))
            
            fstr = '{:8d}' + ' {:10.6f}' * vgrid['np'] + '\n'
            sigma = vgrid['sigma'].T
            
            for i, k in enumerate(sigma):
                f.write(fstr.format(i + 1, *k))
            
        elif vgrid['ivcor'] == 2:
            # SZ grid
            f.write(f"{vgrid['nvrt']} {vgrid['kz']} {vgrid['h_s']} !nvrt, kz, h_s \nZ levels\n")
            
            for k, zlevel in enumerate(vgrid['ztot']):
                f.write(f"{k + 1} {zlevel}\n")
                
            f.write(f"S levels\n{vgrid['h_c']} {vgrid['theta_b']} {vgrid['theta_f']} !h_c, theta_b, theta_f\n")
            
            for k, slevel in enumerate(vgrid['sigma']):
                f.write(f"{k + 1} {slevel:9.6f}\n")
    
    return filename


def read_formatted_csv(filename, sep=',', na_values='', date_parser=None, **kwargs):
    """
    Read a formatted CSV file with advanced options.
    
    Parameters
    ----------
    filename : str
        Path to CSV file
    sep : str, optional
        Column separator (default: ',')
    na_values : str, optional
        Value to treat as NaN (default: '')
    date_parser : callable, optional
        Function to parse dates
    **kwargs : dict
        Additional arguments for pandas.read_csv
        
    Returns
    -------
    pandas.DataFrame
        Loaded data
    """
    return pd.read_csv(filename, sep=sep, na_values=na_values, 
                     date_parser=date_parser, **kwargs)


def write_formatted_csv(df, filename, sep=',', index=False, date_format=None, **kwargs):
    """
    Write a formatted CSV file with advanced options.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to write
    filename : str
        Output file path
    sep : str, optional
        Column separator (default: ',')
    index : bool, optional
        Whether to write row index (default: False)
    date_format : str, optional
        Format for date columns
    **kwargs : dict
        Additional arguments for pandas.to_csv
        
    Returns
    -------
    str
        Path to output file
    """
    df.to_csv(filename, sep=sep, index=index, date_format=date_format, **kwargs)
    return filename
