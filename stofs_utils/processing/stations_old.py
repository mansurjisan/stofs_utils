"""
Station-based output extraction for STOFS-3D

Includes tools for reading station locations, extracting time series or vertical
profiles from SCHISM NetCDF output, and writing results to NetCDF format.
"""

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from netCDF4 import Dataset


def read_station_file(station_file_path):
    """
    Read SCHISM station.in file.

    Parameters
    ----------
    station_file_path : str
        Path to station.in file.

    Returns
    -------
    list of dict
        Each dict has keys: 'name', 'lon', 'lat'
    """
    stations = []
    with open(station_file_path) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                name, lon, lat = parts
                stations.append({
                    'name': name,
                    'lon': float(lon),
                    'lat': float(lat)
                })
    return stations


def extract_station_timeseries(ds, stations, variables):
    """
    Extracts variables at station locations using nearest-neighbor interpolation.

    Parameters
    ----------
    ds : xarray.Dataset
        SCHISM dataset (e.g., out2d_x.nc)
    stations : list of dict
        List of stations with 'lon' and 'lat'
    variables : list of str
        Variable names to extract

    Returns
    -------
    dict
        Dictionary of station data keyed by station name
    """
    x = ds['SCHISM_hgrid_node_x'].values
    y = ds['SCHISM_hgrid_node_y'].values
    coords = np.column_stack((x, y))
    tree = cKDTree(coords)

    results = {}
    for station in stations:
        dist, idx = tree.query([station['lon'], station['lat']])
        station_data = {}
        for var in variables:
            data = ds[var].values  # shape: (time, node)
            station_data[var] = data[:, idx]
        results[station['name']] = station_data
    return results


def write_station_netcdf(output_path, data, stations, variables, times):
    """
    Write extracted station time series to NetCDF.

    Parameters
    ----------
    output_path : str
        Output file path
    data : dict
        Output from extract_station_timeseries()
    stations : list of dict
        Station metadata
    variables : list of str
        Variables written
    times : numpy.ndarray or list
        Time array (1D)
    """
    with Dataset(output_path, 'w') as nc:
        nc.createDimension('time', len(times))
        nc.createDimension('station', len(stations))

        time_var = nc.createVariable('time', 'f8', ('time',))
        time_var[:] = times
        time_var.units = 'seconds since 1970-01-01 00:00:00'

        name_var = nc.createVariable('station_name', str, ('station',))
        lon_var = nc.createVariable('lon', 'f4', ('station',))
        lat_var = nc.createVariable('lat', 'f4', ('station',))

        for i, s in enumerate(stations):
            name_var[i] = s['name']
            lon_var[i] = s['lon']
            lat_var[i] = s['lat']

        for var in variables:
            v = nc.createVariable(var, 'f4', ('time', 'station'))
            for j, s in enumerate(stations):
                v[:, j] = data[s['name']][var]


# Future additions:
# def extract_station_profiles(...):
#     # Interpolation logic for 3D variables (e.g., salinity/temperature)
#     pass

# def convert_staout_to_netcdf(...):
#     # Optional: read raw staout and convert to nc
#     pass
