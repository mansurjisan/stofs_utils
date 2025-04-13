"""
ADCIRC format conversion module for STOFS3D

Contains functions to convert SCHISM model output files to ADCIRC format,
which can be useful for compatibility with ADCIRC-based visualization and analysis tools.
"""
import os
import time
import numpy as np
from datetime import datetime
import argparse
from netCDF4 import Dataset
import shapefile

from ..utils.helpers import split_quads
from ..core.coordinate_utils import inside_polygon
from ..io.netcdf import add_variable


def find_points_in_polyshp(pt_xy, shapefile_names):
    """
    Find points within shapefile polygons.
    
    Parameters
    ----------
    pt_xy : numpy.ndarray
        Array of points [n, 2]
    shapefile_names : list of str
        List of shapefile paths
        
    Returns
    -------
    numpy.ndarray
        Boolean mask of points inside polygons
    """
    ind = np.zeros(pt_xy[:, 0].shape)
    for shapefile_name in shapefile_names:
        sf = shapefile.Reader(shapefile_name)
        shapes = sf.shapes()

        for i, shp in enumerate(shapes):
            poly_xy = np.array(shp.points).T
            print(f'shape {i+1} of {len(shapes)}, {poly_xy[:, 0]}')
            ind += inside_polygon(pt_xy, poly_xy[0], poly_xy[1])  # 1: true; 0: false

    ind = ind.astype('bool')
    return ind


def convert_to_adcirc(
    input_filename, 
    output_dir='.', 
    input_city_identifier_file=None, 
    static_city_mask=True,
    fill_value=-99999.0
):
    """
    Convert SCHISM output to ADCIRC netCDF format.
    
    Parameters
    ----------
    input_filename : str
        Path to SCHISM netCDF output file
    output_dir : str, optional
        Directory for output file (default: current directory)
    input_city_identifier_file : str, optional
        Path to file defining urban regions
        If None, defaults based on static_city_mask
    static_city_mask : bool, optional
        Whether to use static mask (True) or search in polygons (False)
    fill_value : float, optional
        Fill value for dry nodes and masked areas
        
    Returns
    -------
    str
        Path to output ADCIRC file
    """
    # Start timing
    t0 = time.time()
    
    # Determine appropriate filenames
    input_fileindex = os.path.basename(input_filename).replace("_", ".").split(".")[1]
    output_filename = f"{output_dir}/schout_adcirc_{input_fileindex}.nc"
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine city identifier file
    if input_city_identifier_file is None:
        if static_city_mask:
            input_city_identifier_file = './Shapefiles/city_poly.node_id.txt'
        else:
            input_city_identifier_file = './Shapefiles/city_poly.shp'

    if not os.path.exists(input_city_identifier_file):
        raise FileNotFoundError(f"City identifier file not found: {input_city_identifier_file}")
    
    print(f"Using {input_city_identifier_file} to identify urban area")
    
    # Create path for temporary node ID file if needed
    if not static_city_mask:
        output_nodeId_fname = os.path.splitext(input_city_identifier_file)[0] + '.node_id.txt'
    
    # Read input file
    ds = Dataset(input_filename)
    units = ds['time'].units
    base_date = ds['time'].base_date

    # Get coordinates/bathymetry
    x = ds['SCHISM_hgrid_node_x'][:]
    y = ds['SCHISM_hgrid_node_y'][:]
    depth = ds['depth'][:]
    NP = depth.shape[0]

    # Get elements and split quads into tris
    elements = ds['SCHISM_hgrid_face_nodes'][:, :]
    tris = split_quads(elements)
    NE = len(tris)
    print(f'NE is {NE}')
    NV = 3

    # Get times
    times = ds['time'][:]
    ntimes = len(times)

    # Process elevation data
    elev = ds['elevation'][:, :]
    idxs = np.where(elev > 100000)
    elev[idxs] = fill_value

    # Calculate maximum elevation and time of maximum
    maxelev = np.max(elev, axis=0)
    idxs = np.argmax(elev, axis=0)
    time_maxelev = times[idxs]

    # Calculate disturbance (deviation from initial condition)
    maxdist = maxelev.copy()
    land_node_idx = depth < 0
    maxdist[land_node_idx] = np.maximum(0, maxelev[land_node_idx] + depth[land_node_idx])

    # Find city nodes
    if static_city_mask:
        city_node_idx = np.loadtxt(input_city_identifier_file).astype(bool)
    else:
        city_node_idx = find_points_in_polyshp(np.c_[x, y], shapefile_names=[input_city_identifier_file])
        np.savetxt(output_nodeId_fname, city_node_idx)

    # Set mask for dry nodes
    idry = np.zeros(NP)
    idxs = np.where(maxelev + depth <= 1e-6)
    maxelev[idxs] = fill_value

    # Set mask for small disturbance on land and cities
    small_dist_idx = maxdist < 0.3
    filled_idx = small_dist_idx * (land_node_idx + city_node_idx)
    maxdist[filled_idx] = fill_value

    # Process wind data
    uwind = ds['windSpeedX'][:, :]
    vwind = ds['windSpeedY'][:, :]
    idxs = np.where(uwind > 100000)
    uwind[idxs] = fill_value
    vwind[idxs] = fill_value

    ds.close()

    # Create output file
    with Dataset(output_filename, "w", format="NETCDF4") as fout:
        # Create dimensions
        fout.createDimension('time', None)
        fout.createDimension('node', NP)
        fout.createDimension('nele', NE)
        fout.createDimension('nvertex', NV)

        # Add all variables using the add_variable utility
        add_variable(fout, 'time', 'f8', ('time',),
                    long_name="Time",
                    base_date=base_date,
                    standard_name="time",
                    units=units,
                    data=times)

        add_variable(fout, 'x', 'f8', ('node',),
                    long_name="node x-coordinate",
                    standard_name="longitude",
                    units="degrees_east",
                    positive="east",
                    data=x)

        add_variable(fout, 'y', 'f8', ('node',),
                    long_name="node y-coordinate",
                    standard_name="latitude",
                    units="degrees_north",
                    positive="north",
                    data=y)

        add_variable(fout, 'element', 'i', ('nele', 'nvertex',),
                    long_name="element",
                    standard_name="face_node_connectivity",
                    start_index=1,
                    units="nondimensional",
                    data=np.array(tris))

        add_variable(fout, 'depth', 'f8', ('node',),
                    long_name="distance below XGEOID20B",
                    standard_name="depth below XGEOID20B",
                    coordinates="time y x",
                    location="node",
                    units="m",
                    data=depth)

        add_variable(fout, 'zeta_max', 'f8', ('node',),
                    standard_name="maximum_sea_surface_height_above_xgeoid20b",
                    coordinates="y x",
                    location="node",
                    units="m",
                    fill_value=fill_value,
                    data=maxelev)

        add_variable(fout, 'time_of_zeta_max', 'f8', ('node',),
                    standard_name="time_of_maximum_sea_surface_height_above_xgeoid20b",
                    coordinates="y x",
                    location="node",
                    units="sec",
                    fill_value=fill_value,
                    data=time_maxelev)

        add_variable(fout, 'disturbance_max', 'f8', ('node',),
                    standard_name="maximum_depature_from_initial_condition",
                    coordinates="y x",
                    location="node",
                    units="m",
                    fill_value=fill_value,
                    data=maxdist)

        add_variable(fout, 'zeta', 'f8', ('time', 'node',),
                    standard_name="sea_surface_height_above_xgeoid20b",
                    coordinates="time y x",
                    location="node",
                    units="m",
                    fill_value=fill_value,
                    data=elev)

        add_variable(fout, 'uwind', 'f8', ('time', 'node',),
                    long_name="10m_above_ground/UGRD",
                    standard_name="eastward_wind",
                    coordinates="time y x",
                    location="node",
                    units="ms-1",
                    fill_value=fill_value,
                    data=uwind)

        add_variable(fout, 'vwind', 'f8', ('time', 'node',),
                    long_name="10m_above_ground/VGRD",
                    standard_name="northward_wind",
                    coordinates="time y x",
                    location="node",
                    units="ms-1",
                    fill_value=fill_value,
                    data=vwind)

        # Add global attributes
        fout.title = 'SCHISM Model output'
        fout.source = 'SCHISM model output version v10'
        fout.references = 'http://ccrm.vims.edu/schismweb/'

    print(f'ADCIRC conversion took {time.time()-t0:.2f} seconds')
    return output_filename


def adcirc_cli():
    """
    Command-line interface for the ADCIRC converter.
    
    This function handles command-line arguments for the ADCIRC conversion utility.
    Run from command line with: python -m stofs_utils.processing.adcirc [arguments]
    """
    parser = argparse.ArgumentParser(description='Convert SCHISM output to ADCIRC format')
    parser.add_argument('--input_filename', required=True, help='Input file in SCHISM format')
    parser.add_argument('--input_city_identifier_file', help='Input shapefile defining urban region')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    args = parser.parse_args()

    # Determine hostname to configure system-specific settings
    import socket
    myhost = socket.gethostname()
    static_city_mask = True  # Default to static mask
    
    # Different settings based on system
    if "sciclone" in myhost or "frontera" in myhost:
        static_city_mask = False  # Use polygon-based mask on these systems
    
    print(f"Running on {myhost}")

    # Perform conversion
    output_file = convert_to_adcirc(
        input_filename=args.input_filename,
        output_dir=args.output_dir,
        input_city_identifier_file=args.input_city_identifier_file,
        static_city_mask=static_city_mask
    )
    
    print(f"ADCIRC file created: {output_file}")


if __name__ == "__main__":
    adcirc_cli()
