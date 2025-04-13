"""
Shapefile I/O utilities for STOFS3D

Provides functions to read/write shapefiles and extract attributes or geometry for model preprocessing.
"""

import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon


def read_shapefile(file_path):
    """
    Read a shapefile and return a GeoDataFrame.

    Parameters
    ----------
    file_path : str
        Path to the shapefile (.shp)

    Returns
    -------
    geopandas.GeoDataFrame
        Shapefile content as a GeoDataFrame
    """
    return gpd.read_file(file_path)


def filter_by_attribute(gdf, column, value):
    """
    Filter a GeoDataFrame by attribute value.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    column : str
        Column name to filter by
    value : any
        Value to match

    Returns
    -------
    GeoDataFrame
        Filtered GeoDataFrame
    """
    return gdf[gdf[column] == value]


def extract_coordinates(geometry):
    """
    Extract coordinates from shapely geometry.

    Parameters
    ----------
    geometry : shapely.geometry
        A Point, LineString, or Polygon geometry

    Returns
    -------
    list of tuple
        List of (lon, lat) or (x, y) coordinates
    """
    if isinstance(geometry, Point):
        return [(geometry.x, geometry.y)]
    elif isinstance(geometry, (LineString, Polygon)):
        return list(geometry.exterior.coords if isinstance(geometry, Polygon) else geometry.coords)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")


def write_shapefile(gdf, output_path, driver="ESRI Shapefile"):
    """
    Write a GeoDataFrame to a shapefile.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame to write
    output_path : str
        Output path (should end in .shp)
    driver : str, optional
        Shapefile driver (default is 'ESRI Shapefile')
    """
    gdf.to_file(output_path, driver=driver)
