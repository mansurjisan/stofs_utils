"""
GeoJSON generation module for STOFS3D

Contains functions to generate GeoJSON representations of SCHISM model outputs,
particularly for visualizing water disturbance contours.
"""
import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from geopandas import GeoDataFrame
import multiprocessing as mp

from ..utils.helpers import split_quads, triangulation


def contour_to_gdf(disturbance, levels, tri, cmap_name='jet', min_max_values=None):
    """
    Convert matplotlib contour data to GeoDataFrame.
    
    Parameters
    ----------
    disturbance : numpy.ndarray
        Disturbance values at nodes
    levels : list
        Contour levels
    tri : matplotlib.tri.Triangulation
        Triangulation of the grid
    cmap_name : str, optional
        Matplotlib colormap name (default: 'jet')
    min_max_values : tuple, optional
        (min_value, max_value) for color mapping
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing contours
    """
    # Set min/max values
    if min_max_values is None:
        MinVal = levels[0]
        MaxVal = levels[-1]
    else:
        MinVal, MaxVal = min_max_values
    
    # Create level ranges for labeling
    MinMax = [(disturbance.min(), levels[0])]
    for i in np.arange(len(levels)-1):
        MinMax.append((levels[i], levels[i+1]))
    
    # Create contours
    fig = plt.figure()
    ax = fig.add_subplot()
    
    my_cmap = plt.cm.get_cmap(cmap_name)
    contour = ax.tricontourf(tri, disturbance, vmin=MinVal, vmax=MaxVal,
                            levels=levels, cmap=my_cmap, extend='min')
    
    # Transform contour collections to polygons
    polygons, colors = [], []
    data = []
    
    for i, polygon in enumerate(contour.collections):
        mpoly = []
        for path in polygon.get_paths():
            try:
                path.should_simplify = False
                poly = path.to_polygons()
                
                # Each polygon should contain an exterior ring + maybe hole(s)
                exterior, holes = [], []
                if len(poly) > 0 and len(poly[0]) > 3:
                    # The first ring is the exterior
                    exterior = poly[0]
                    # Other rings are holes
                    if len(poly) > 1:
                        holes = [h for h in poly[1:] if len(h) > 3]
                
                mpoly.append(make_valid(Polygon(exterior, holes)))
            except Exception as e:
                print(f'Warning: Geometry error when making polygon #{i}: {str(e)}')
        
        if len(mpoly) > 1:
            mpoly = MultiPolygon(mpoly)
            polygons.append(mpoly)
            colors.append(polygon.get_facecolor().tolist()[0])
            data.append({
                'id': i+1, 
                'minWaterLevel': MinMax[i][0], 
                'maxWaterLevel': MinMax[i][1],
                'verticalDatum': 'XGEOID20B', 
                'units': 'meters', 
                'geometry': mpoly
            })
        elif len(mpoly) == 1:
            polygons.append(mpoly[0])
            colors.append(polygon.get_facecolor().tolist()[0])
            data.append({
                'id': i+1, 
                'minWaterLevel': MinMax[i][0], 
                'maxWaterLevel': MinMax[i][1],
                'verticalDatum': 'XGEOID20B', 
                'units': 'meters', 
                'geometry': mpoly[0]
            })
    
    plt.close('all')
    
    # Create GeoDataFrame
    gdf = GeoDataFrame(data)
    
    # Create color values
    colors_elev = []
    my_cmap = plt.cm.get_cmap(cmap_name)
    
    for i in range(len(gdf)):
        color = my_cmap(i/len(gdf))
        colors_elev.append(mpl.colors.to_hex(color))
    
    gdf['rgba'] = colors_elev
    
    # Set CRS
    gdf = gdf.set_crs(4326)
    
    return gdf

def get_disturbance(elevation, depth, levels, fill_value, out_filename, tri, 
                   dry_node_threshold=1.e-6, min_disturbance_threshold=0.3,
                   layer_name='disturbance', output_format='GPKG'):
    """
    Calculate water disturbance and save to GeoJSON file.
    
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
    tri : matplotlib.tri.Triangulation
        Triangulation of the grid
    dry_node_threshold : float, optional
        Threshold for identifying dry nodes (elevation + depth <= threshold)
    min_disturbance_threshold : float, optional
        Minimum disturbance value to include (default: 0.3m)
    layer_name : str, optional
        Name of the layer in the output file (default: 'disturbance')
    output_format : str, optional
        Output file format (default: 'GPKG', options: 'GPKG', 'GeoJSON', 'ESRI Shapefile')
    """
    # Calculate disturbance
    disturbance = np.copy(elevation)
    idxs_land_node = depth < 0
    disturbance[idxs_land_node] = np.maximum(0, elevation[idxs_land_node] + depth[idxs_land_node])
    
    # Set mask on dry nodes
    idxs_dry = np.where(elevation + depth <= dry_node_threshold)
    disturbance[idxs_dry] = fill_value
    
    # Set mask on nodes with small max disturbance on land
    idxs_small = disturbance < min_disturbance_threshold
    idxs_mask_maxdist = idxs_small * idxs_land_node
    disturbance[idxs_mask_maxdist] = fill_value
    
    # [rest of function remains the same]
    
    # Convert to GeoDataFrame
    gdf = contour_to_gdf(disturbance, levels, tri_copy)
    
    # Map format strings to driver names
    format_drivers = {
        'GPKG': 'GPKG',
        'GeoJSON': 'GeoJSON',
        'SHP': 'ESRI Shapefile',
        'ESRI Shapefile': 'ESRI Shapefile'
    }
    
    # Get appropriate driver
    driver = format_drivers.get(output_format.upper(), 'GPKG')
    
    # Save to file
    gdf.to_file(out_filename, driver=driver, layer=layer_name)
    
    return gdf

def generate_disturbance_contours(
    input_filename, 
    output_dir=".", 
    output_prefix="stofs_3d_atl",
    levels=None, 
    fill_value=-99999.0,
    use_multiprocessing=True,
    output_format='GPKG',
    layer_name='disturbance',
    dry_node_threshold=1.e-6,
    min_disturbance_threshold=0.3
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
    
    # Read netcdf dataset
    ds = Dataset(input_filename)
    
    # Get coordinates/bathymetry
    x = ds['SCHISM_hgrid_node_x'][:]
    y = ds['SCHISM_hgrid_node_y'][:]
    depth = ds['depth'][:]
    
    # Get elements and split quads into tris
    elements = ds['SCHISM_hgrid_face_nodes'][:, :]
    split_start = time.time()
    tris = split_quads(elements)
    print(f'Splitting quads took {time.time() - split_start:.2f} seconds')
    
    # Get triangulation for contour plot
    tri = triangulation(x, y, tris)
    
    # Get time and elevation data
    times = ds['time'][:]
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
    
    # For hindcast
    for i in range(24, 0, -1):
        filenames.append(f"{output_prefix}.t12z.disturbance.n{i:03d}.gpkg")
    
    # For forecast
    for i in range(96):
        filenames.append(f"{output_prefix}.t12z.disturbance.f{i+1:03d}.gpkg")
    
    # Slice to the actual available times
    filenames = filenames[:len(times)]
    
    # Add output directory path
    filenames = [os.path.join(output_dir, f) for f in filenames]
    
    # Process disturbance contours
    disturbance_start = time.time()
    
    if use_multiprocessing:
        # Set up multiprocessing pool
        npool = min(len(times), mp.cpu_count() - 1)
        
        # Create argument list for parallel processing
#        args_list = [(np.squeeze(elev[i, :]), depth, levels, fill_value, filenames[i], tri) 
#                    for i in range(len(times))]
        args_list = [(np.squeeze(elev[i, :]), depth, levels, fill_value, filenames[i], tri,
                     dry_node_threshold, min_disturbance_threshold, layer_name, output_format) 
                    for i in range(len(times))]

        # Run parallel processing
        with mp.Pool(npool) as pool:
            pool.starmap(get_disturbance, args_list)
    else:
        # Process sequentially
        for i in range(len(times)):
            get_disturbance(np.squeeze(elev[i, :]), depth, levels, fill_value, filenames[i], tri)
    
    print(f'Calculating and masking disturbance for all times took {time.time() - disturbance_start:.2f} seconds')
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
    parser.add_argument('--output_format', default='GPKG', choices=['GPKG', 'GeoJSON', 'SHP'], 
                       help='Output file format')
    parser.add_argument('--layer_name', default='disturbance', help='Layer name in output file')
    parser.add_argument('--dry_threshold', type=float, default=1.e-6, 
                       help='Threshold for identifying dry nodes')
    parser.add_argument('--min_disturbance', type=float, default=0.3, 
                       help='Minimum disturbance threshold')
   
    args = parser.parse_args()
    
    # Generate levels from arguments
    levels = [round(level, 1) for level in np.arange(args.min_level, args.max_level + args.level_step, args.level_step)]
    
    # Generate contours
    output_files = generate_disturbance_contours(
        input_filename=args.input_filename,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        levels=levels,
        use_multiprocessing=not args.single_process,
        output_format=args.output_format,
        layer_name=args.layer_name,
        dry_node_threshold=args.dry_threshold,
        min_disturbance_threshold=args.min_disturbance
    )
    
    print(f"GeoJSON files created: {len(output_files)}")
    print(f"First file: {output_files[0]}")
    print(f"Last file: {output_files[-1]}")


if __name__ == "__main__":
    geojson_cli()
