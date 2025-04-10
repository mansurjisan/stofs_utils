"""
Grid I/O utilities for STOFS3D

Contains functions for reading, writing, and converting SCHISM grid files.
"""
import os
import numpy as np
import pickle
from netCDF4 import Dataset
import warnings


def load_grid(fname):
    """
    Load a SCHISM grid from a serialized file.
    
    Parameters
    ----------
    fname : str
        Path to grid file (.npz or .pkl)
        
    Returns
    -------
    object
        Container with grid objects
    """
    if fname.endswith('.npz'):
        return _load_npz(fname)
    elif fname.endswith('.pkl'):
        return _load_pkl(fname)
    else:
        raise ValueError(f"Unsupported format: {fname}")


def save_grid(grid_obj, vgrid_obj=None, fname='grid', path='.'):
    """
    Save SCHISM grid objects to a serialized file.
    
    Parameters
    ----------
    grid_obj : SchismGrid or None
        Horizontal grid object
    vgrid_obj : SchismVGrid or None
        Vertical grid object
    fname : str, optional
        Base filename (default: 'grid')
    path : str, optional
        Directory to save the file (default: '.')
        
    Returns
    -------
    str
        Path to saved file
    """
    # Create container object
    class DataContainer:
        pass
    
    S = DataContainer()
    
    if grid_obj is not None:
        S.hgrid = grid_obj
    
    if vgrid_obj is not None:
        S.vgrid = vgrid_obj
    
    # Determine file extension
    if not (fname.endswith('.npz') or fname.endswith('.pkl')):
        fname = f"{fname}.npz"
    
    # Full file path
    full_path = os.path.join(path, fname)
    
    # Save based on extension
    if full_path.endswith('.npz'):
        _save_npz(full_path, S)
    elif full_path.endswith('.pkl'):
        _save_pkl(full_path, S)
    else:
        raise ValueError(f"Unsupported format: {full_path}")
    
    return full_path


def _save_npz(fname, data):
    """
    Save data as NPZ format
    
    Parameters
    ----------
    fname : str
        Output filename
    data : object
        Data to save
    """
    # Get all attributes
    svars = list(data.__dict__.keys())
    
    # Check for functions and handle them
    rvars = []
    for vari in svars:
        if hasattr(data.__dict__[vari], '__call__'):
            import cloudpickle
            try:
                data.__dict__[vari] = cloudpickle.dumps(data.__dict__[vari])
            except:
                print(f'Function {vari} not saved')
                rvars.append(vari)
    
    svars = np.setdiff1d(svars, rvars)
    
    # Construct save string
    save_str = f'np.savez_compressed("{fname}"'
    for vari in svars:
        save_str = f'{save_str}, {vari}=data.{vari}'
    save_str = f'{save_str})'
    
    # Execute save command
    exec(save_str)


def _save_pkl(fname, data):
    """
    Save data as pickle format
    
    Parameters
    ----------
    fname : str
        Output filename
    data : object
        Data to save
    """
    with open(fname, 'wb') as fid:
        pickle.dump(data, fid, pickle.HIGHEST_PROTOCOL)


def _load_npz(fname):
    """
    Load data from NPZ format
    
    Parameters
    ----------
    fname : str
        Input filename
        
    Returns
    -------
    object
        Loaded data
    """
    # Create data container
    class DataContainer:
        pass
    
    # Load data
    data0 = np.load(fname, allow_pickle=True)
    keys0 = data0.keys()
    
    # Extract data
    vdata = DataContainer()
    for keyi in keys0:
        datai = data0[keyi]
        # Handle object arrays
        if datai.dtype == np.dtype('O'):
            datai = datai[()]
        
        # Handle functions
        if 'cloudpickle.cloudpickle' in str(datai):
            import pickle
            try:
                datai = pickle.loads(datai)
            except:
                continue
        
        # Set attribute
        setattr(vdata, keyi, datai)
    
    return vdata


def _load_pkl(fname):
    """
    Load data from pickle format
    
    Parameters
    ----------
    fname : str
        Input filename
        
    Returns
    -------
    object
        Loaded data
    """
    class DataContainer:
        pass
    
    with open(fname, 'rb') as fid:
        data = pickle.load(fid)
    
    vdata = DataContainer()
    vdata.__dict__ = data.__dict__.copy()
    
    return vdata


def read_schism_hgrid(fname):
    """
    Read a SCHISM horizontal grid file.
    
    Parameters
    ----------
    fname : str
        Path to grid file (.gr3 or .ll)
        
    Returns
    -------
    SchismGrid
        Loaded grid object
    """
    from ..core.grid import SchismGrid
    grid = SchismGrid()
    grid.read_hgrid(fname)
    return grid


def read_schism_vgrid(fname):
    """
    Read a SCHISM vertical grid file.
    
    Parameters
    ----------
    fname : str
        Path to vgrid.in file
        
    Returns
    -------
    SchismVGrid
        Loaded vertical grid object
    """
    from ..core.vertical_grid import SchismVGrid
    vgrid = SchismVGrid()
    vgrid.read_vgrid(fname)
    return vgrid


def read_schism_bpfile(fname, fmt=0):
    """
    Read a SCHISM bpfile (station file).
    
    Parameters
    ----------
    fname : str
        Path to bpfile
    fmt : int, optional
        Format type:
        - 0: Standard bp file (default)
        - 1: ACE/gredit *.reg file
        
    Returns
    -------
    object
        Loaded station data
    """
    # Create data container
    class BpFile:
        def __init__(self):
            self.nsta = 0
            self.x = np.array([])
            self.y = np.array([])
            self.z = np.array([])
            self.station = []
    
    bp = BpFile()
    
    # Read file content
    with open(fname, 'r') as f:
        lines = [i.strip().split() for i in f.readlines()]
    
    stations = [i.strip().split('!')[-1] for i in open(fname, 'r').readlines()[2:] if ('!' in i)]
    
    if fmt == 0:
        # Standard bp file
        bp.nsta = int(lines[1][0])
        if bp.nsta == 0:
            return bp
        
        # Process lines with coordinates and values
        fc = lambda x: x if len(x) == 4 else [*x[:4], x[4][1:]]
        data = np.array([fc(line) for line in lines[2:(2+bp.nsta)]])
        
        bp.x = data[:, 1].astype(float)
        bp.y = data[:, 2].astype(float)
        bp.z = data[:, 3].astype(float)
        
    elif fmt == 1:
        # ACE/gredit *.reg file
        bp.nsta = int(lines[2][0])
        if bp.nsta == 0:
            return bp
        
        data = np.squeeze(np.array([lines[3:]])).astype('float')
        bp.x = data[:, 0]
        bp.y = data[:, 1]
        bp.z = np.zeros(bp.nsta)
        
    else:
        raise ValueError(f"Unknown format: {fmt}")
    
    # Set station names if available
    if len(stations) == bp.nsta:
        bp.station = np.array(stations)
    else:
        bp.station = np.array([f"{i}" for i in range(bp.nsta)])
    
    return bp


def write_schism_bpfile(bp, fname, fmt=0):
    """
    Write a SCHISM bpfile (station file).
    
    Parameters
    ----------
    bp : object
        Bpfile object with station data
    fname : str
        Output file path
    fmt : int, optional
        Format type:
        - 0: Standard bp file (default)
        - 1: ACE/gredit *.reg file
    """
    with open(fname, 'w+') as fid:
        # Write header
        if hasattr(bp, 'note'):
            fid.write(f'ACE/gredit: {bp.note}')
        
        if fmt == 0:
            fid.write('bpfile in ACE/gredit format\n')
            fid.write(f'{bp.nsta}\n')
        elif fmt == 1:
            fid.write('Region in ACE/gredit format\n1\n')
            fid.write(f'{bp.nsta} 1\n')
        
        # Get station names
        stations = [i+1 for i in range(bp.nsta)]
        if hasattr(bp, 'station') and len(bp.station) == bp.nsta:
            stations = bp.station
        
        # Write points
        for i in range(bp.nsta):
            if fmt == 0:
                fid.write(f'{i+1} {bp.x[i]:<.8f} {bp.y[i]:<.8f} {bp.z[i]:<.8f} !{stations[i]}\n')
            elif fmt == 1:
                fid.write(f'{bp.x[i]:<.8f} {bp.y[i]:<.8f}\n')


def save_schism_grid(fname='grid', path='.', fmt=0):
    """
    Read and save SCHISM grid files from a directory.
    
    Parameters
    ----------
    fname : str, optional
        Base filename for saving (default: 'grid')
    path : str, optional
        Directory where grid files exist (default: '.')
    fmt : int, optional
        Format option:
        - 0: Don't save grid's full geometry (default)
        - 1: Save full geometry
        
    Returns
    -------
    object
        Container with saved grid objects
    """
    from ..core.grid import SchismGrid
    from ..core.vertical_grid import SchismVGrid
    
    gname = f'{path}/hgrid.gr3'
    gname_ll = f'{path}/hgrid.ll'
    vname = f'{path}/vgrid.in'
    
    # Create a container for the data
    class DataContainer:
        pass
    
    S = DataContainer()
    
    # Load horizontal grid if exists
    if os.path.exists(gname):
        gd = SchismGrid(gname)
        
        # Add lon/lat if available
        if os.path.exists(gname_ll):
            gdl = SchismGrid(gname_ll)
            gd.lon, gd.lat = gdl.x, gdl.y
        
        # Compute additional properties if requested
        if fmt == 1:
            # These methods need to be available in SchismGrid
            gd.compute_all()
            gd.compute_bnd()
        
        S.hgrid = gd
    
    # Load vertical grid if exists
    if os.path.exists(vname):
        S.vgrid = SchismVGrid()
        S.vgrid.read_vgrid(vname)
    
    # Check if any grids were loaded
    if not (hasattr(S, 'hgrid') or hasattr(S, 'vgrid')):
        raise FileNotFoundError(f"Not found: {gname}, {vname}")
    
    # Save the container
    save_file = os.path.join('.', fname)
    if not (save_file.endswith('.npz') or save_file.endswith('.pkl')):
        save_file = f"{save_file}.npz"
    
    _save_npz(save_file, S)
    
    return S


def sms2grd(sms_file, grd_file=None):
    """
    Read SMS *.2dm grid and convert to SCHISM format.
    
    Parameters
    ----------
    sms_file : str
        Path to SMS *.2dm file
    grd_file : str or None, optional
        Path to output SCHISM *.gr3 file (if None, no file is written)
        
    Returns
    -------
    SchismGrid
        Converted grid object
    """
    from ..core.grid import SchismGrid
    
    # Read 2dm file
    with open(sms_file, 'r') as fid:
        lines = fid.readlines()
    
    # Process triangle elements
    E3 = np.array([[*i.strip().split()[1:-1], '-1'] 
                  for i in lines if i.startswith('E3T')]).astype('int')
    
    # Process quad elements
    E4 = np.array([i.strip().split()[1:-1] 
                  for i in lines if i.startswith('E4Q')]).astype('int')
    
    # Combine and sort elements
    E34 = np.r_[E3, E4]
    sind = np.argsort(E34[:, 0])
    E34 = E34[sind]
    
    # Process nodes
    ND = np.array([i.strip().split()[1:] 
                  for i in lines if i.startswith('ND')]).astype('float')
    sind = np.argsort(ND[:, 0])
    ND = ND[sind]
    
    # Create grid object
    gd = SchismGrid()
    gd.ne = E34.shape[0]
    gd.np = ND.shape[0]
    gd.elnode = E34[:, 1:] - 1  # Convert to 0-based indexing
    gd.x, gd.y, gd.dp = ND[:, 1:].T
    
    # Set element types
    gd.i34 = np.ones(gd.ne, dtype=int) * 4
    gd.i34[E34[:, -1] == -1] = 3
    
    # Write grid if output file specified
    if grd_file is not None:
        gd.write_hgrid(grd_file)
    
    return gd


def grd2sms(grd, sms_file):
    """
    Convert SCHISM grid to SMS *.2dm format.
    
    Parameters
    ----------
    grd : SchismGrid or str
        SCHISM grid object or path to grid file
    sms_file : str
        Path to output SMS *.2dm file
    """
    from ..core.grid import SchismGrid
    
    # Read grid if string is provided
    if isinstance(grd, str):
        gd = SchismGrid(grd)
    elif isinstance(grd, SchismGrid):
        gd = grd
    else:
        raise ValueError(f"Unsupported grid type: {type(grd)}")
    
    # Write 2dm file
    with open(sms_file, 'w+') as fid:
        # Header
        fid.write('MESH2D\n')
        
        # Write elements
        for i in range(gd.ne):
            if gd.i34[i] == 3:
                fid.write(f'E3T {i+1} {gd.elnode[i,0]+1} {gd.elnode[i,1]+1} {gd.elnode[i,2]+1} 1\n')
            elif gd.i34[i] == 4:
                fid.write(f'E4Q {i+1} {gd.elnode[i,0]+1} {gd.elnode[i,1]+1} {gd.elnode[i,2]+1} {gd.elnode[i,3]+1} 1\n')
        
        # Write nodes
        for i in range(gd.np):
            fid.write(f'ND {i+1} {gd.x[i]:.8f} {gd.y[i]:.8f} {gd.dp[i]:.8f}\n')


def scatter_to_schism_grid(xyz, angle_min=None, area_max=None, side_min=None, side_max=None):
    """
    Create SCHISM grid from scatter points.
    
    Parameters
    ----------
    xyz : numpy.ndarray
        Points with format c_[x, y] or c_[x, y, z]
    angle_min : float or None, optional
        Minimum allowed internal angle (degrees)
    area_max : float or None, optional
        Maximum allowed element area
    side_min : float or None, optional
        Minimum allowed side length
    side_max : float or None, optional
        Maximum allowed side length
        
    Returns
    -------
    SchismGrid
        Created grid object
    """
    from ..core.grid import SchismGrid
    import matplotlib.tri as mtri
    
    # Extract points
    x, y = xyz.T[:2]
    np_points = len(x)
    z = xyz[:, 2] if xyz.shape[1] >= 3 else np.zeros(np_points)
    
    # Triangulate points
    tri = mtri.Triangulation(x, y)
    
    # Create grid object
    gd = SchismGrid()
    gd.np = np_points
    gd.ne = len(tri.triangles)
    gd.x, gd.y, gd.dp = x, y, z
    
    # Set connectivity
    gd.elnode = np.c_[tri.triangles, -np.ones((gd.ne, 1), dtype=int)]
    gd.i34 = np.ones(gd.ne, dtype=int) * 3
    
    # Filter elements if criteria provided
    if any(param is not None for param in [angle_min, area_max, side_min, side_max]):
        _clean_grid(gd, angle_min, area_max, side_min, side_max)
    
    return gd


def _clean_grid(gd, angle_min=5, area_max=None, side_min=None, side_max=None):
    """
    Clean a grid by removing elements that don't meet criteria.
    
    Parameters
    ----------
    gd : SchismGrid
        Grid to clean
    angle_min : float or None, optional
        Minimum allowed internal angle (degrees)
    area_max : float or None, optional
        Maximum allowed element area
    side_min : float or None, optional
        Minimum allowed side length
    side_max : float or None, optional
        Maximum allowed side length
    """
    # Compute necessary grid properties
    if not hasattr(gd, 'area'):
        gd.compute_area()
    if not hasattr(gd, 'xctr'):
        gd.compute_ctr()
    
    # Find elements with small angles
    angles = []
    fp3 = gd.i34 == 3
    fp4 = gd.i34 == 4
    id1, id2, id3 = np.ones([3, gd.ne], dtype=int)
    sid = np.arange(gd.ne)
    
    for i in range(4):
        id1[fp3] = i % 3
        id2[fp3] = (i + 1) % 3
        id3[fp3] = (i + 2) % 3
        
        id1[fp4] = i % 4
        id2[fp4] = (i + 1) % 4
        id3[fp4] = (i + 2) % 4
        
        x1 = gd.x[gd.elnode[sid, id1]]
        y1 = gd.y[gd.elnode[sid, id1]]
        x2 = gd.x[gd.elnode[sid, id2]]
        y2 = gd.y[gd.elnode[sid, id2]]
        x3 = gd.x[gd.elnode[sid, id3]]
        y3 = gd.y[gd.elnode[sid, id3]]
        
        # Compute angle
        a1 = np.arctan2(y2 - y1, x2 - x1)
        a2 = np.arctan2(y3 - y2, x3 - x2)
        ai = np.abs(a1 - a2) * 180 / np.pi
        ai = np.minimum(ai, 360 - ai)
        
        angles.append(ai)
    
    angles = np.array(angles).T
    min_angles = np.min(angles, axis=1)
    
    # Calculate side lengths
    sides = []
    for i in range(4):
        id1 = i % 4
        id2 = (i + 1) % 4
        
        # Only process relevant elements
        if i < 3:
            # All elements for sides 0, 1, 2
            nodes1 = gd.elnode[:, id1]
            nodes2 = gd.elnode[:, id2]
        else:
            # Only quads for side 3
            nodes1 = gd.elnode[fp4, id1]
            nodes2 = gd.elnode[fp4, id2]
        
        # Skip masked nodes
        valid = (nodes1 >= 0) & (nodes2 >= 0)
        if np.any(valid):
            x1 = gd.x[nodes1[valid]]
            y1 = gd.y[nodes1[valid]]
            x2 = gd.x[nodes2[valid]]
            y2 = gd.y[nodes2[valid]]
            
            side_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            sides.append(side_length)
    
    sides = np.array(sides)
    min_sides = np.min(sides, axis=0)
    max_sides = np.max(sides, axis=0)
    
    # Identify elements to remove
    to_remove = np.zeros(gd.ne, dtype=bool)
    
    if angle_min is not None:
        to_remove |= (min_angles < angle_min)
    
    if area_max is not None:
        to_remove |= (gd.area > area_max)
    
    if side_min is not None:
        to_remove |= (min_sides < side_min)
    
    if side_max is not None:
        to_remove |= (max_sides > side_max)
    
    # Keep elements that meet criteria
    keep_elems = ~to_remove
    if np.any(keep_elems):
        # Update grid
        gd.ne = np.sum(keep_elems)
        gd.i34 = gd.i34[keep_elems]
        gd.elnode = gd.elnode[keep_elems]
        gd.area = gd.area[keep_elems]
        gd.xctr = gd.xctr[keep_elems]
        gd.yctr = gd.yctr[keep_elems]
        gd.dpe = gd.dpe[keep_elems]
    else:
        warnings.warn("No elements remain after filtering")


def convert_dem_format(fname, sname, fmt=0):
    """
    Convert a DEM file from one format to another.
    
    Parameters
    ----------
    fname : str
        Input DEM file
    sname : str
        Output file name
    fmt : int, optional
        Format type:
        - 0: Convert DEM file in *.asc format to *.npz format (default)
    """
    if fmt == 0:
        if not fname.endswith('.asc'):
            fname = fname + '.asc'
        
        # Read file
        with open(fname, 'r') as fid:
            ncols = int(fid.readline().strip().split()[1])
            nrows = int(fid.readline().strip().split()[1])
            xn, xll = fid.readline().strip().split()
            xll = float(xll)
            yn, yll = fid.readline().strip().split()
            yll = float(yll)
            dxy = float(fid.readline().strip().split()[1])
            nodata = float(fid.readline().strip().split()[1])
        
        # Read elevation data
        elev = np.loadtxt(fname, skiprows=6)
        
        # Adjust corner coordinates if needed
        if xn.lower() == 'xllcorner' and yn.lower() == 'yllcorner':
            xll = xll + dxy/2
            yll = yll + dxy/2
        
        # Create output container
        class DataContainer:
            pass
        
        S = DataContainer()
        S.lon = xll + dxy * np.arange(ncols)
        S.lat = yll - dxy * np.arange(nrows) + (nrows-1) * dxy
        S.elev = elev.astype('float32')
        S.nodata = nodata
        
        # Save as npz
        if not sname.endswith('.npz'):
            sname = sname + '.npz'
        
        _save_npz(sname, S)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def read_schism_prop(fname):
    """
    Read a SCHISM property file (element-based).
    
    Parameters
    ----------
    fname : str
        Path to property file
        
    Returns
    -------
    numpy.ndarray
        Property values
    """
    pdata = np.loadtxt(fname)
    if pdata.ndim == 2:
        pvalue = pdata[:, 1]
    else:
        pvalue = pdata[None, :][0, 1]
    
    return pvalue


def write_schism_prop(gd, fname='schism.prop', value=None, fmt='{:8.5f}'):
    """
    Write a SCHISM property file.
    
    Parameters
    ----------
    gd : SchismGrid
        Grid object
    fname : str, optional
        Output filename (default: 'schism.prop')
    value : numpy.ndarray or float or None, optional
        Property values (default: uses gd.dpe)
    fmt : str, optional
        Output format (default: '{:8.5f}')
    """
    # Get property values
    if value is None:
        if not hasattr(gd, 'dpe'):
            gd.compute_ctr()
        pvi = gd.dpe.copy()
    else:
        if hasattr(value, "__len__"):
            pvi = value
        else:
            pvi = np.ones(gd.ne) * value
    
    # Convert to int if needed
    if 'd' in fmt:
        pvi = pvi.astype('int')
    
    # Write file
    with open(fname, 'w+') as fid:
        for i in range(gd.ne):
            fid.write(f"{i+1} {fmt.format(pvi[i])}\n")
