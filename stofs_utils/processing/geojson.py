"""
GeoJSON generation module for STOFS3D

Contains functions to generate GeoJSON representations of SCHISM model outputs,
particularly for visualizing water disturbance contours.
"""
import os
import time
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from geopandas import GeoDataFrame
import multiprocessing as mp

from ..utils.helpers import split_quads, triangulation


def contour_to_gdf(disturbance, levels, triangulation):
    """
    Convert disturbance data to GeoDataFrame with contours.
    
    Parameters
    ----------
    disturbance : numpy.ndarray
        Disturbance values at nodes
    levels : list
        Contour levels
    triangulation : matplotlib.tri.Triangulation
        Triangulation of the grid
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing contours
    """
    # Set min/max values
    MinVal = levels[0]
    MaxVal = levels[-1]

    # Create level ranges for labeling
    MinMax = [(disturbance.min(), levels[0])]
    for i in np.arange(len(levels)-1):
        MinMax.append((levels[i], levels[i+1]))

    fig = plt.figure()
    ax = fig.add_subplot()

    my_cmap = plt.cm.jet
    contour = ax.tricontourf(triangulation, disturbance, vmin=MinVal, vmax=MaxVal,
        levels=levels, cmap=my_cmap, extend='min')

    # Transform a `matplotlib.contour.QuadContourSet` to a GeoDataFrame
    polygons, colors = [], []
    data = []
    for i, polygon in enumerate(contour.collections):
        mpoly = []
        for path in polygon.get_paths():
            try:
                path.should_simplify = False
                poly = path.to_polygons()
                # Each polygon should contain an exterior ring + maybe hole(s):
                exterior, holes = [], []
                if len(poly) > 0 and len(poly[0]) > 3:
                    # The first of the list is the exterior ring:
                    exterior = poly[0]
                    # Other(s) are hole(s):
                    if len(poly) > 1:
                        holes = [h for h in poly[1:] if len(h) > 3]
                mpoly.append(make_valid(Polygon(exterior, holes)))
            except:
                print('Warning: Geometry error when making polygon #{}'.format(i))

        if len(mpoly) > 1:
            mpoly = MultiPolygon(mpoly)
            polygons.append(mpoly)
            colors.append(polygon.get_facecolor().tolist()[0])
            data.append({'id': i+1, 'minWaterLevel': MinMax[i][0], 'maxWaterLevel': MinMax[i][1], 
                    'verticalDatum': 'XGEOID20B', 'units': 'meters', 'geometry': mpoly})
        elif len(mpoly) == 1:
            polygons.append(mpoly[0])
            colors.append(polygon.get_facecolor().tolist()[0])
            data.append({'id': i+1, 'minWaterLevel': MinMax[i][0], 'maxWaterLevel': MinMax[i][1], 
                    'verticalDatum': 'XGEOID20B', 'units': 'meters', 'geometry': mpoly[0]})
    plt.close('all')

    gdf = GeoDataFrame(data)

    # Get color in Hex
    colors_elev = []
    my_cmap = plt.cm.jet

    for i in range(len(gdf)):
        color = my_cmap(i/len(gdf))
        colors_elev.append(mpl.colors.to_hex(color))

    gdf['rgba'] = colors_elev

    # Set crs
    gdf = gdf.set_crs(4326)

    return gdf


def get_disturbance(elevation, depth, levels, fill_value, out_filename, triangulation):
    """
    Calculate water disturbance and save to GeoJSON/GPKG file.
    
    Parameters
    ----------
    elevation : numpy.ndarray
        Water surface elevation
    depth : numpy.ndarray
        Bathymetry (depth values)
    levels : list
        Contour levels
    fill_value : float
        Fill value for dry nodes
    out_filename : str
        Output filename
    triangulation : matplotlib.tri.Triangulation
        Triangulation of the grid
    """
    # Calculate disturbance
    disturbance = copy.deepcopy(elevation)
    idxs_land_node = depth < 0
    disturbance[idxs_land_node] = np.maximum(0, elevation[idxs_land_node] + depth[idxs_land_node])

    # Set mask on dry nodes
    idxs_dry = np.where(elevation + depth <= 1.e-6)
    disturbance[idxs_dry] = fill_value

    # Set mask on nodes with small max disturbance (<0.3 m) on land
    idxs_small = disturbance < 0.3
    idxs_mask_maxdist = idxs_small * idxs_land_node
    disturbance[idxs_mask_maxdist] = fill_value

    # Get mask for triangulation
    imask = disturbance < -90000
    mask = np.any(np.where(imask[triangulation.triangles], True, False), axis=1)
    triangulation.set_mask(mask)
    
    # Convert to GeoDataFrame
    gdf = contour_to_gdf(disturbance, levels, triangulation)
    
    # Determine file format and layer name
    if out_filename.endswith('.gpkg'):
        gdf.to_file(out_filename, driver="GPKG", layer='disturbance')
    else:
        gdf.to_file(out_filename, driver="GeoJSON")


def generate_disturbance_contours(
    input_filename,
    output_dir=".",
    output_prefix="stofs_3d_atl",
    levels=None,
    fill_value=-99999.0,
    use_multiprocessing=True
):
    """
    Generate disturbance contours from SCHISM output.
    
    Parameters
    ----------
    input_filename : str
        Path to SCHISM netCDF output file
    output_dir : str, optional
        Directory for output files (default: current directory)
    output_prefix : str, optional
        Prefix for output filenames
    levels : list, optional
        Contour levels (default: 0.3 to 2.0 by 0.1)
    fill_value : float, optional
        Fill value for dry nodes
    use_multiprocessing : bool, optional
        Whether to use multiprocessing (default: True)
        
    Returns
    -------
    list
        List of output filenames
    """
    # Start timing
    t0 = time.time()
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Set default levels if not provided
    if levels is None:
        levels = [round(i, 1) for i in np.arange(0.3, 2.1, 0.1)]
    
    # Reading netcdf dataset
    ds = Dataset(input_filename)
    
    # Get coordinates/bathymetry
    x = ds['SCHISM_hgrid_node_x'][:]
    y = ds['SCHISM_hgrid_node_y'][:]
    depth = ds['depth'][:]
    
    # Get elements and split quads into tris
    elements = ds['SCHISM_hgrid_face_nodes'][:, :]
    t_split = time.time()
    tris = split_quads(elements)
    print(f'Splitting quads took {time.time() - t_split:.2f} seconds')
    
    # Get triangulation for contour plot
    tri = triangulation(x, y, tris)
    
    # Get time and elevation data
    times = ds['time'][:]
    try:
        dates = ds.num2date(times, ds['time'].units)
    except:
        # Fallback to netCDF4 num2date if ds.num2date not available
        from netCDF4 import num2date
        dates = num2date(times, ds['time'].units)
    
    # Get elevation
    elev = ds['elevation'][:, :]
    idxs = np.where(elev > 100000)
    elev[idxs] = fill_value
    
    # Calculate max elevation for this stack
    maxelev = np.max(elev, axis=0)
    idxs = np.argmax(elev, axis=0)
    time_maxelev = times[idxs]
    
    # Generate filenames for all time steps
    filenames = []
    
    # For hindcast (n-files)
    for i in range(24, 0, -1):
        filenames.append(f"{output_prefix}.t12z.disturbance.n{i:03d}.gpkg")
    
    # For forecast (f-files)
    for i in range(96):
        filenames.append(f"{output_prefix}.t12z.disturbance.f{i+1:03d}.gpkg")
    
    # Slice to the actual number of times available
    filenames = filenames[:len(times)]
    
    # Add output directory path
    filenames = [os.path.join(output_dir, f) for f in filenames]
    
    # Process disturbance contours
    t_disturbance = time.time()
    
    if use_multiprocessing:
        # Determine number of processes
        npool = min(len(times), mp.cpu_count() - 1)
        if npool < 1:
            npool = 1
            
        print(f"Using {npool} processes for parallel processing")
        
        # Create and start process pool
        pool = mp.Pool(npool)
        args_list = [(np.squeeze(elev[i, :]), depth, levels, fill_value, filenames[i], tri) 
                     for i in range(len(times))]
        pool.starmap(get_disturbance, args_list)
        
        # Clean up
        pool.close()
        del pool
    else:
        # Sequential processing
        for i in range(len(times)):
            get_disturbance(np.squeeze(elev[i, :]), depth, levels, fill_value, filenames[i], tri)
    
    print(f'Calculating and masking disturbance for all times took {time.time() - t_disturbance:.2f} seconds')
    print(f'Total processing time: {time.time() - t0:.2f} seconds')
    
    ds.close()
    return filenames


def geojson_cli():
    """
    Command-line interface for the GeoJSON generator.
    
    This function handles command-line arguments for the GeoJSON generation utility.
    Run from command line with: python -m stofs_utils.processing.geojson [arguments]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate GeoJSON disturbance contours from SCHISM output')
    parser.add_argument('--input_filename', required=True, help='Input file in SCHISM format')
    parser.add_argument('--output_dir', default='.', help='Output directory')
    parser.add_argument('--output_prefix', default='stofs_3d_atl', help='Prefix for output filenames')
    parser.add_argument('--min_level', type=float, default=0.3, help='Minimum contour level')
    parser.add_argument('--max_level', type=float, default=2.0, help='Maximum contour level')
    parser.add_argument('--level_step', type=float, default=0.1, help='Contour level step')
    parser.add_argument('--single_process', action='store_true', help='Disable multiprocessing')
    
    args = parser.parse_args()
    
    # Generate levels from arguments
    levels = [round(level, 1) for level in np.arange(args.min_level, args.max_level + args.level_step, args.level_step)]
    
    # Generate contours
    output_files = generate_disturbance_contours(
        input_filename=args.input_filename,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        levels=levels,
        use_multiprocessing=not args.single_process
    )
    
    print(f"GeoJSON files created: {len(output_files)}")
    print(f"First file: {output_files[0]}")
    print(f"Last file: {output_files[-1]}")


if __name__ == "__main__":
    geojson_cli()
