"""
Helper utility functions for STOFS3D

Contains general utility functions used throughout the STOFS3D package.
"""
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import matplotlib.tri as mtri


def split_quads(elements=None):
    """
    Split quad elements to triangles.
    
    Parameters
    ----------
    elements : numpy.ndarray
        Element connectivity array [ne, 4]
        
    Returns
    -------
    numpy.ndarray
        Array of triangles
    """
    if elements is None:
        raise ValueError('elements should be a numpy array of (np, 4)')
    
    # Mask -1 values (for triangles in quad array)
    elements = np.ma.masked_values(elements, -1)
    
    tris = []
    for ele in elements:
        # Extract non-masked values
        ele = ele[~ele.mask]
        if len(ele) == 3:
            tris.append([ele[0], ele[1], ele[2]])
        elif len(ele) == 4:
            tris.append([ele[0], ele[1], ele[3]])
            tris.append([ele[1], ele[2], ele[3]])
    
    return np.array(tris).astype('int')


def triangulation(lon, lat, tris):
    """
    Create a matplotlib triangulation from points and connectivity.
    
    Parameters
    ----------
    lon : numpy.ndarray
        Longitude/X coordinates
    lat : numpy.ndarray
        Latitude/Y coordinates
    tris : numpy.ndarray
        Triangle connectivity
        
    Returns
    -------
    matplotlib.tri.Triangulation
        Triangulation object
    """
    # Check if 1-based indexing and convert to 0-based
    if tris.max() >= len(lon):
        tris = tris - 1
    
    return mtri.Triangulation(lon, lat, tris)


def date_to_seconds(date, reference_date=None):
    """
    Convert a date to seconds since a reference date.
    
    Parameters
    ----------
    date : datetime
        Date to convert
    reference_date : datetime, optional
        Reference date (default: date with time set to 00:00:00)
        
    Returns
    -------
    float
        Seconds since reference date
    """
    if reference_date is None:
        reference_date = datetime(date.year, date.month, date.day)
    
    delta = date - reference_date
    return delta.total_seconds()


def seconds_to_date(seconds, reference_date):
    """
    Convert seconds since a reference date to a datetime.
    
    Parameters
    ----------
    seconds : float
        Seconds since reference date
    reference_date : datetime
        Reference date
        
    Returns
    -------
    datetime
        Resulting date
    """
    return reference_date + timedelta(seconds=seconds)


def get_time_range(start_date, end_date, delta=None, inclusive=True):
    """
    Generate a range of dates.
    
    Parameters
    ----------
    start_date : datetime
        Start date
    end_date : datetime
        End date
    delta : timedelta, optional
        Time step (default: 1 hour)
    inclusive : bool, optional
        Whether to include end_date (default: True)
        
    Returns
    -------
    list
        List of datetime objects
    """
    if delta is None:
        delta = timedelta(hours=1)
    
    dates = []
    current = start_date
    
    if inclusive:
        end_date = end_date + delta
    
    while current < end_date:
        dates.append(current)
        current += delta
    
    return dates


def ensure_directory(directory):
    """
    Ensure a directory exists, creating it if necessary.
    
    Parameters
    ----------
    directory : str
        Directory path
        
    Returns
    -------
    str
        Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return directory


def get_file_list(directory, pattern=None, sort=True):
    """
    Get a list of files in a directory.
    
    Parameters
    ----------
    directory : str
        Directory path
    pattern : str, optional
        File pattern (e.g., '*.nc')
    sort : bool, optional
        Whether to sort files (default: True)
        
    Returns
    -------
    list
        List of file paths
    """
    import glob
    
    if pattern is None:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files = [os.path.join(directory, f) for f in files]
    else:
        files = glob.glob(os.path.join(directory, pattern))
    
    if sort:
        files.sort()
    
    return files


def find_nearest_point(x, y, points):
    """
    Find the index of the nearest point.
    
    Parameters
    ----------
    x : float
        X coordinate
    y : float
        Y coordinate
    points : numpy.ndarray
        Array of points [n, 2]
        
    Returns
    -------
    int
        Index of nearest point
    """
    dist = (points[:, 0] - x)**2 + (points[:, 1] - y)**2
    return np.argmin(dist)


def find_nearest_time(time, times):
    """
    Find the index of the nearest time.
    
    Parameters
    ----------
    time : float or datetime
        Time to find
    times : numpy.ndarray or list
        Array of times
        
    Returns
    -------
    int
        Index of nearest time
    """
    if isinstance(time, datetime) and isinstance(times[0], datetime):
        # Convert to numerical values for comparison
        time_val = time.timestamp()
        time_vals = np.array([t.timestamp() for t in times])
    else:
        time_val = time
        time_vals = times
    
    return np.argmin(np.abs(time_vals - time_val))


def interpolate_in_time(data, times, query_times, axis=0):
    """
    Interpolate data in time.
    
    Parameters
    ----------
    data : numpy.ndarray
        Data array with time axis
    times : numpy.ndarray
        Time values
    query_times : numpy.ndarray
        Times to interpolate to
    axis : int, optional
        Time axis (default: 0)
        
    Returns
    -------
    numpy.ndarray
        Interpolated data
    """
    from scipy.interpolate import interp1d
    
    # Create interpolator
    f = interp1d(times, data, axis=axis, bounds_error=False, fill_value="extrapolate")
    
    # Interpolate
    return f(query_times)


def interpolate_in_vertical(data, z_in, z_out, mask_invalid=True):
    """
    Interpolate data in vertical dimension.
    
    Parameters
    ----------
    data : numpy.ndarray
        Data array
    z_in : numpy.ndarray
        Input z coordinates
    z_out : numpy.ndarray
        Output z coordinates
    mask_invalid : bool, optional
        Whether to mask invalid values (default: True)
        
    Returns
    -------
    numpy.ndarray
        Interpolated data
    """
    from scipy.interpolate import interp1d
    
    # Create interpolator
    f = interp1d(z_in, data, axis=1, bounds_error=False, fill_value="extrapolate")
    
    # Interpolate
    data_interp = f(z_out)
    
    # Mask invalid values
    if mask_invalid:
        data_interp[np.isnan(data_interp)] = 0
    
    return data_interp


def load_config(config_file):
    """
    Load configuration from file.
    
    Parameters
    ----------
    config_file : str
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    import json
    import yaml
    
    if config_file.endswith('.json'):
        with open(config_file, 'r') as f:
            config = json.load(f)
    elif config_file.endswith(('.yaml', '.yml')):
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML files")
    else:
        raise ValueError(f"Unsupported config file format: {config_file}")
    
    return config


def timer(func):
    """
    Timer decorator for functions.
    
    Parameters
    ----------
    func : callable
        Function to time
        
    Returns
    -------
    callable
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        import time
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    
    return wrapper


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points.
    
    Parameters
    ----------
    lon1 : float
        Longitude of point 1
    lat1 : float
        Latitude of point 1
    lon2 : float
        Longitude of point 2
    lat2 : float
        Latitude of point 2
        
    Returns
    -------
    float
        Distance in kilometers
    """
    from math import radians, sin, cos, sqrt, atan2
    
    # Convert coordinates to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Earth radius in kilometers
    r = 6371.0
    
    return r * c


def compute_gradient(z, dx, dy):
    """
    Compute gradient of a 2D field.
    
    Parameters
    ----------
    z : numpy.ndarray
        2D field [ny, nx]
    dx : float or numpy.ndarray
        Grid spacing in x
    dy : float or numpy.ndarray
        Grid spacing in y
        
    Returns
    -------
    tuple
        (dz_dx, dz_dy) - Gradient components
    """
    # Compute gradient using central differences
    dz_dx = np.zeros_like(z)
    dz_dy = np.zeros_like(z)
    
    # Interior points
    dz_dx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2 * dx)
    dz_dy[1:-1, :] = (z[2:, :] - z[:-2, :]) / (2 * dy)
    
    # Boundaries
    dz_dx[:, 0] = (z[:, 1] - z[:, 0]) / dx
    dz_dx[:, -1] = (z[:, -1] - z[:, -2]) / dx
    dz_dy[0, :] = (z[1, :] - z[0, :]) / dy
    dz_dy[-1, :] = (z[-1, :] - z[-2, :]) / dy
    
    return dz_dx, dz_dy


def parallel_process(func, args_list, n_jobs=None):
    """
    Process tasks in parallel.
    
    Parameters
    ----------
    func : callable
        Function to call
    args_list : list
        List of argument tuples for func
    n_jobs : int, optional
        Number of jobs (default: number of CPUs)
        
    Returns
    -------
    list
        Results
    """
    try:
        from multiprocessing import Pool, cpu_count
        
        if n_jobs is None:
            n_jobs = cpu_count()
        
        with Pool(n_jobs) as pool:
            results = pool.starmap(func, args_list)
        
        return results
    
    except ImportError:
        # Fallback to serial processing
        return [func(*args) for args in args_list]


def color_map_from_values(values, cmap_name='jet', levels=None):
    """
    Create a colormap from values.
    
    Parameters
    ----------
    values : numpy.ndarray
        Values to map
    cmap_name : str, optional
        Colormap name (default: 'jet')
    levels : int or numpy.ndarray, optional
        Levels for colormap
        
    Returns
    -------
    tuple
        (cmap, norm) - Colormap and normalization
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, Normalize
    
    # Get colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Determine levels
    if levels is None:
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        if isinstance(levels, int):
            vmin = np.nanmin(values)
            vmax = np.nanmax(values)
            levels = np.linspace(vmin, vmax, levels)
        
        # Create normalization
        norm = BoundaryNorm(levels, cmap.N)
    
    return cmap, norm


def remove_small_regions(data, min_size=100):
    """
    Remove small disconnected regions in binary data.
    
    Parameters
    ----------
    data : numpy.ndarray
        Binary mask
    min_size : int, optional
        Minimum region size (default: 100)
        
    Returns
    -------
    numpy.ndarray
        Cleaned mask
    """
    try:
        from scipy import ndimage
        
        # Label connected regions
        labeled_array, num_features = ndimage.label(data)
        
        # Get sizes of each region
        sizes = np.bincount(labeled_array.ravel())
        
        # Set small regions to 0
        mask_sizes = sizes < min_size
        mask_sizes[0] = False  # Keep background
        remove_regions = mask_sizes[labeled_array]
        labeled_array[remove_regions] = 0
        
        # Convert back to binary
        return labeled_array > 0
    
    except ImportError:
        return data


def extract_time_series(data, times, x_index, y_index):
    """
    Extract time series from a dataset.
    
    Parameters
    ----------
    data : numpy.ndarray
        Data array [time, y, x]
    times : numpy.ndarray
        Time values
    x_index : int
        X index
    y_index : int
        Y index
        
    Returns
    -------
    tuple
        (times, values) - Time series
    """
    return times, data[:, y_index, x_index]


def smooth_timeseries(data, window_size=3):
    """
    Apply smoothing to a time series.
    
    Parameters
    ----------
    data : numpy.ndarray
        Time series data
    window_size : int, optional
        Smoothing window size (default: 3)
        
    Returns
    -------
    numpy.ndarray
        Smoothed data
    """
    # Make window size odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Calculate half window size
    hw = window_size // 2
    
    # Create output array
    result = np.zeros_like(data)
    
    # Apply moving average
    for i in range(len(data)):
        start = max(0, i - hw)
        end = min(len(data), i + hw + 1)
        window_data = data[start:end]
        result[i] = np.mean(window_data)
    
    return result


def merge_dictionaries(dict1, dict2):
    """
    Merge two dictionaries recursively.
    
    Parameters
    ----------
    dict1 : dict
        First dictionary
    dict2 : dict
        Second dictionary
        
    Returns
    -------
    dict
        Merged dictionary
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dictionaries(result[key], value)
        else:
            result[key] = value
    
    return result


def generate_unique_filename(base_name, extension, directory='.'):
    """
    Generate a unique filename.
    
    Parameters
    ----------
    base_name : str
        Base filename
    extension : str
        File extension
    directory : str, optional
        Directory path (default: '.')
        
    Returns
    -------
    str
        Unique filename
    """
    counter = 1
    
    # Ensure extension starts with a dot
    if not extension.startswith('.'):
        extension = '.' + extension
    
    # Try to find a unique name
    filename = f"{base_name}{extension}"
    full_path = os.path.join(directory, filename)
    
    while os.path.exists(full_path):
        filename = f"{base_name}_{counter}{extension}"
        full_path = os.path.join(directory, filename)
        counter += 1
    
    return filename


def bytes_to_human_readable(size_bytes):
    """
    Convert bytes to human-readable format.
    
    Parameters
    ----------
    size_bytes : int
        Size in bytes
        
    Returns
    -------
    str
        Human-readable size
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def is_windows():
    """
    Check if running on Windows.
    
    Returns
    -------
    bool
        True if running on Windows
    """
    return os.name == 'nt'


def get_available_memory():
    """
    Get available system memory.
    
    Returns
    -------
    int
        Available memory in bytes
    """
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return None


def format_time_label(seconds, reference_time=None):
    """
    Format time label from seconds.
    
    Parameters
    ----------
    seconds : float
        Seconds since reference time
    reference_time : datetime, optional
        Reference time (default: None)
        
    Returns
    -------
    str
        Formatted time label
    """
    if reference_time is None:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours:02d}:{minutes:02d}"
    else:
        time = reference_time + timedelta(seconds=seconds)
        return time.strftime("%Y-%m-%d %H:%M")
