"""
Time utility functions for STOFS3D

Contains functions for time conversions, time range generation and
other time-related processing for the STOFS3D forecasting system.
"""
import numpy as np
import datetime
from dateutil import parser


def date_to_seconds(date_str, reference_date=None):
    """
    Convert date string to seconds since reference date.
    
    Parameters:
    -----------
    date_str : str
        Date string in format 'YYYY-MM-DD HH:MM:SS'
    reference_date : str, optional
        Reference date string in format 'YYYY-MM-DD HH:MM:SS'
        If None, uses '1970-01-01 00:00:00'
        
    Returns:
    --------
    float
        Seconds since reference date
    """
    if reference_date is None:
        reference_date = '1970-01-01 00:00:00'
    
    date_obj = parser.parse(date_str)
    ref_obj = parser.parse(reference_date)
    
    delta = date_obj - ref_obj
    return delta.total_seconds()


def get_time_range(start_date, end_date, interval_hours=1):
    """
    Generate time range for forecasts.
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD HH:MM:SS'
    end_date : str
        End date in format 'YYYY-MM-DD HH:MM:SS'
    interval_hours : float, optional
        Time interval in hours, default is 1
        
    Returns:
    --------
    list
        List of datetime objects from start_date to end_date with interval_hours spacing
    """
    start = parser.parse(start_date)
    end = parser.parse(end_date)
    
    delta = datetime.timedelta(hours=interval_hours)
    
    time_range = []
    current = start
    
    while current <= end:
        time_range.append(current)
        current += delta
        
    return time_range


def seconds_to_date(seconds, reference_date=None):
    """
    Convert seconds since reference date to datetime.
    
    Parameters:
    -----------
    seconds : float
        Seconds since reference date
    reference_date : str, optional
        Reference date string in format 'YYYY-MM-DD HH:MM:SS'
        If None, uses '1970-01-01 00:00:00'
        
    Returns:
    --------
    datetime
        Datetime object
    """
    if reference_date is None:
        reference_date = '1970-01-01 00:00:00'
    
    ref_obj = parser.parse(reference_date)
    date_obj = ref_obj + datetime.timedelta(seconds=seconds)
    
    return date_obj


def generate_forecast_times(start_date, forecast_hours, interval_hours=1):
    """
    Generate time range for a forecast period.
    
    Parameters:
    -----------
    start_date : datetime.datetime
        Start date of the forecast
    forecast_hours : int
        Number of hours to forecast
    interval_hours : float, optional
        Time interval in hours (default: 1)
        
    Returns:
    --------
    list
        List of datetime objects for the forecast period
    """
    end_date = start_date + datetime.timedelta(hours=forecast_hours)
    return get_time_range(start_date.strftime('%Y-%m-%d %H:%M:%S'),
                          end_date.strftime('%Y-%m-%d %H:%M:%S'),
                          interval_hours)


def date_range_to_seconds(dates, reference_date=None):
    """
    Convert a list of dates to seconds since reference date.
    
    Parameters:
    -----------
    dates : list
        List of datetime objects
    reference_date : datetime, optional
        Reference date (default: first date in the list)
        
    Returns:
    --------
    numpy.ndarray
        Array of seconds since reference date
    """
    if reference_date is None and len(dates) > 0:
        reference_date = dates[0]
    
    seconds = []
    for date in dates:
        delta = date - reference_date
        seconds.append(delta.total_seconds())
    
    return np.array(seconds)


def format_date_str(date, fmt='%Y-%m-%d %H:%M:%S'):
    """
    Format a datetime object as a string.
    
    Parameters:
    -----------
    date : datetime
        Datetime object to format
    fmt : str, optional
        Format string (default: '%Y-%m-%d %H:%M:%S')
        
    Returns:
    --------
    str
        Formatted date string
    """
    return date.strftime(fmt)


def parse_date_str(date_str, fmt=None):
    """
    Parse a date string to a datetime object.
    
    Parameters:
    -----------
    date_str : str
        Date string to parse
    fmt : str, optional
        Format string (default: None, uses dateutil.parser)
        
    Returns:
    --------
    datetime
        Parsed datetime object
    """
    if fmt is None:
        return parser.parse(date_str)
    else:
        return datetime.datetime.strptime(date_str, fmt)


def compute_time_steps(start_date, end_date, time_step=3600):
    """
    Compute time steps between two dates.
    
    Parameters:
    -----------
    start_date : datetime
        Start date
    end_date : datetime
        End date
    time_step : float, optional
        Time step in seconds (default: 3600, 1 hour)
        
    Returns:
    --------
    int
        Number of time steps
    """
    delta = end_date - start_date
    return int(delta.total_seconds() / time_step) + 1


def create_netcdf_time_variable(ds, times, units=None, calendar='gregorian'):
    """
    Create a time variable in a NetCDF dataset.
    
    Parameters:
    -----------
    ds : netCDF4.Dataset
        NetCDF dataset
    times : list or numpy.ndarray
        Time values (seconds since reference)
    units : str, optional
        Time units (default: 'seconds since 1970-01-01 00:00:00')
    calendar : str, optional
        Calendar type (default: 'gregorian')
        
    Returns:
    --------
    netCDF4.Variable
        Time variable
    """
    if units is None:
        units = 'seconds since 1970-01-01 00:00:00'
    
    # Create time dimension if it doesn't exist
    if 'time' not in ds.dimensions:
        ds.createDimension('time', None)
    
    # Create time variable
    time_var = ds.createVariable('time', 'f8', ('time',))
    time_var.units = units
    time_var.calendar = calendar
    time_var.standard_name = 'time'
    time_var.long_name = 'Time'
    time_var[:] = times
    
    return time_var


def is_leap_year(year):
    """
    Check if a year is a leap year.
    
    Parameters:
    -----------
    year : int
        Year to check
        
    Returns:
    --------
    bool
        True if leap year, False otherwise
    """
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False


def day_of_year(date):
    """
    Get the day of year for a date.
    
    Parameters:
    -----------
    date : datetime
        Date
        
    Returns:
    --------
    int
        Day of year (1-366)
    """
    return date.timetuple().tm_yday