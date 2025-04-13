"""
Shapefile I/O utilities for STOFS3D

Contains functions for reading, writing, and manipulating shapefiles used
in STOFS3D oceanographic modeling.
"""
import os
import numpy as np
import warnings


def read_shapefile(filename, return_attributes=True):
    """
    Read a shapefile and return its geometry and attributes.
    
    Parameters
    ----------
    filename : str
        Path to shapefile (.shp)
    return_attributes : bool, optional
        Whether to return attributes (default: True)
        
    Returns
    -------
    dict
        Dictionary containing geometries and attributes
    """
    try:
        import shapefile
    except ImportError:
        raise ImportError("The pyshp package is required. Please install it with: pip install pyshp")
    
    # Create output container
    output = {
        'type': None,
        'geometries': [],
        'attributes': [] if return_attributes else None
    }
    
    # Read shapefile
    try:
        sf = shapefile.Reader(filename)
        
        # Get shape type
        shape_types = {
            1: 'POINT',
            3: 'POLYLINE',
            5: 'POLYGON',
            8: 'MULTIPOINT',
            11: 'POINTZ',
            13: 'POLYLINEZ',
            15: 'POLYGONZ',
            18: 'MULTIPOINTZ',
            21: 'POINTM',
            23: 'POLYLINEM',
            25: 'POLYGONM',
            28: 'MULTIPOINTM'
        }
        output['type'] = shape_types.get(sf.shapeType, f'UNKNOWN({sf.shapeType})')
        
        # Process shapes
        shapes = sf.shapes()
        for shape in shapes:
            geometry = {
                'points': np.array(shape.points),
                'parts': shape.parts
            }
            
            # Add Z values if present
            if hasattr(shape, 'z') and shape.z:
                geometry['z'] = np.array(shape.z)
            
            output['geometries'].append(geometry)
        
        # Process attributes if requested
        if return_attributes:
            fields = [field[0] for field in sf.fields[1:]]  # Skip DeletionFlag
            for record in sf.records():
                attributes = {field: value for field, value in zip(fields, record)}
                output['attributes'].append(attributes)
        
        return output
    
    except Exception as e:
        raise IOError(f"Error reading shapefile {filename}: {str(e)}")


def write_shapefile(filename, geometries, attributes=None, shape_type=None):
    """
    Write geometries and attributes to a shapefile.
    
    Parameters
    ----------
    filename : str
        Output filename (without extension)
    geometries : list
        List of geometry dictionaries with 'points' and 'parts'
    attributes : list, optional
        List of attribute dictionaries
    shape_type : str, optional
        Shape type (POINT, POLYLINE, POLYGON, etc.)
        If None, determined from first geometry
        
    Returns
    -------
    str
        Path to created shapefile
    """
    try:
        import shapefile
    except ImportError:
        raise ImportError("The pyshp package is required. Please install it with: pip install pyshp")
    
    # Ensure directory exists
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Remove file extension if present
    if filename.lower().endswith('.shp'):
        filename = filename[:-4]
    
    # Determine shape type if not provided
    if shape_type is None and geometries:
        if 'points' in geometries[0] and len(geometries[0]['points']) > 0:
            if len(geometries[0]['parts']) <= 1:
                if len(geometries[0]['points']) == 1:
                    shape_type = 'POINT'
                else:
                    # Check if first and last points are the same
                    pts = geometries[0]['points']
                    if np.array_equal(pts[0], pts[-1]):
                        shape_type = 'POLYGON'
                    else:
                        shape_type = 'POLYLINE'
            else:
                # Multi-part geometry
                shape_type = 'POLYGON'
    
    # Map shape type to shapefile type
    shape_type_map = {
        'POINT': shapefile.POINT,
        'POLYLINE': shapefile.POLYLINE,
        'POLYGON': shapefile.POLYGON,
        'MULTIPOINT': shapefile.MULTIPOINT,
        'POINTZ': shapefile.POINTZ,
        'POLYLINEZ': shapefile.POLYLINEZ,
        'POLYGONZ': shapefile.POLYGONZ,
        'MULTIPOINTZ': shapefile.MULTIPOINTZ
    }
    
    if shape_type is None:
        raise ValueError("Unable to determine shape type and none provided")
    
    shp_type = shape_type_map.get(shape_type.upper())
    if shp_type is None:
        raise ValueError(f"Unsupported shape type: {shape_type}")
    
    # Create writer
    w = shapefile.Writer(filename, shapeType=shp_type)
    
    # Add fields if attributes provided
    if attributes and len(attributes) > 0:
        field_types = {}
        
        # Determine field types from first attribute dictionary
        for field, value in attributes[0].items():
            if isinstance(value, int):
                field_types[field] = 'N'  # Number
            elif isinstance(value, float):
                field_types[field] = 'F'  # Float
            elif isinstance(value, bool):
                field_types[field] = 'L'  # Logical
            else:
                field_types[field] = 'C'  # Character
        
        # Add fields
        for field, field_type in field_types.items():
            w.field(field, field_type, 50 if field_type == 'C' else 10, 
                   10 if field_type == 'F' else 0)
    
    # Write geometries and attributes
    for i, geometry in enumerate(geometries):
        if shape_type == 'POINT':
            # Point shapefile
            w.point(geometry['points'][0][0], geometry['points'][0][1])
        elif shape_type == 'POINTZ':
            # PointZ shapefile
            w.pointz(geometry['points'][0][0], geometry['points'][0][1], geometry['z'][0])
        elif shape_type == 'MULTIPOINT':
            # Multipoint shapefile
            w.multipoint(geometry['points'])
        elif shape_type == 'POLYLINE':
            # Polyline shapefile
            parts = []
            if 'parts' in geometry and len(geometry['parts']) > 0:
                # Use defined parts
                for j in range(len(geometry['parts'])):
                    start = geometry['parts'][j]
                    end = geometry['parts'][j+1] if j < len(geometry['parts'])-1 else len(geometry['points'])
                    parts.append(geometry['points'][start:end])
            else:
                # Single part
                parts = [geometry['points']]
            w.line(parts)
        elif shape_type == 'POLYGON':
            # Polygon shapefile
            parts = []
            if 'parts' in geometry and len(geometry['parts']) > 0:
                # Use defined parts
                for j in range(len(geometry['parts'])):
                    start = geometry['parts'][j]
                    end = geometry['parts'][j+1] if j < len(geometry['parts'])-1 else len(geometry['points'])
                    part = geometry['points'][start:end]
                    # Ensure polygon is closed
                    if not np.array_equal(part[0], part[-1]):
                        part = np.vstack([part, part[0]])
                    parts.append(part)
            else:
                # Single part
                part = geometry['points']
                # Ensure polygon is closed
                if not np.array_equal(part[0], part[-1]):
                    part = np.vstack([part, part[0]])
                parts = [part]
            w.poly(parts)
        
        # Add attributes if available
        if attributes and i < len(attributes):
            w.record(**attributes[i])
        else:
            w.record()
    
    # Write projection file
    write_projection_file(filename, 'WGS84')
    
    return filename + '.shp'


def write_projection_file(shapefile_path, projection_name):
    """
    Write a projection file (.prj) for a shapefile.
    
    Parameters
    ----------
    shapefile_path : str
        Path to shapefile (without extension)
    projection_name : str
        Projection name or WKT string
        
    Returns
    -------
    str
        Path to projection file
    """
    # Remove extension if present
    if shapefile_path.lower().endswith('.shp'):
        shapefile_path = shapefile_path[:-4]
    
    # Common projection definitions
    projections = {
        'WGS84': 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]',
        'NAD83': 'GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]',
        'UTM_ZONE17N': 'PROJCS["NAD_1983_UTM_Zone_17N",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-81.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]'
    }
    
    # Get projection string
    if projection_name in projections:
        prj_string = projections[projection_name]
    else:
        # Assume it's a WKT string
        prj_string = projection_name
    
    # Write projection file
    with open(f"{shapefile_path}.prj", 'w') as f:
        f.write(prj_string)
    
    return f"{shapefile_path}.prj"


def shapefile_to_geojson(shapefile_path, geojson_path=None):
    """
    Convert a shapefile to GeoJSON format.
    
    Parameters
    ----------
    shapefile_path : str
        Path to input shapefile
    geojson_path : str, optional
        Path to output GeoJSON file
        If None, derived from shapefile_path
        
    Returns
    -------
    str
        Path to GeoJSON file
    """
    try:
        import json
        from shapely.geometry import Point, LineString, Polygon, mapping
        import shapefile
    except ImportError:
        missing = []
        try:
            import json
        except ImportError:
            missing.append("json")
        try:
            from shapely.geometry import Point, LineString, Polygon, mapping
        except ImportError:
            missing.append("shapely")
        try:
            import shapefile
        except ImportError:
            missing.append("pyshp")
            
        raise ImportError(f"Missing required packages: {', '.join(missing)}. "
                          f"Please install them with: pip install {' '.join(missing)}")
    
    # Default output path
    if geojson_path is None:
        base = os.path.splitext(shapefile_path)[0]
        geojson_path = f"{base}.geojson"
    
    # Read shapefile
    sf = shapefile.Reader(shapefile_path)
    fields = [field[0] for field in sf.fields if field[0] != 'DeletionFlag']
    
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Process each shape
    for shape_rec in sf.shapeRecords():
        geometry = shape_rec.shape
        record = shape_rec.record
        
        # Create attributes dictionary
        attributes = {field: val for field, val in zip(fields, record)}
        
        # Create feature
        feature = {
            "type": "Feature",
            "properties": attributes,
            "geometry": {}
        }
        
        # Process geometry based on type
        if geometry.shapeType == shapefile.POINT:
            feature["geometry"] = {
                "type": "Point",
                "coordinates": geometry.points[0]
            }
        elif geometry.shapeType == shapefile.POLYLINE:
            feature["geometry"] = {
                "type": "MultiLineString",
                "coordinates": []
            }
            
            # Handle parts
            parts = geometry.parts
            parts.append(len(geometry.points))
            for i in range(len(parts)-1):
                feature["geometry"]["coordinates"].append(
                    geometry.points[parts[i]:parts[i+1]]
                )
        elif geometry.shapeType == shapefile.POLYGON:
            feature["geometry"] = {
                "type": "Polygon",
                "coordinates": []
            }
            
            # Handle parts
            parts = geometry.parts
            parts.append(len(geometry.points))
            for i in range(len(parts)-1):
                ring = geometry.points[parts[i]:parts[i+1]]
                # Ensure ring is closed
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                feature["geometry"]["coordinates"].append(ring)
        
        # Add feature to collection
        geojson["features"].append(feature)
    
    # Write GeoJSON
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f)
    
    return geojson_path


def geojson_to_shapefile(geojson_path, shapefile_path=None):
    """
    Convert a GeoJSON file to shapefile format.
    
    Parameters
    ----------
    geojson_path : str
        Path to input GeoJSON file
    shapefile_path : str, optional
        Path to output shapefile (without extension)
        If None, derived from geojson_path
        
    Returns
    -------
    str
        Path to shapefile (.shp)
    """
    try:
        import json
        import shapefile
    except ImportError:
        missing = []
        try:
            import json
        except ImportError:
            missing.append("json")
        try:
            import shapefile
        except ImportError:
            missing.append("pyshp")
            
        raise ImportError(f"Missing required packages: {', '.join(missing)}. "
                          f"Please install them with: pip install {' '.join(missing)}")
    
    # Default output path
    if shapefile_path is None:
        base = os.path.splitext(geojson_path)[0]
        shapefile_path = base
    
    # Read GeoJSON
    with open(geojson_path, 'r') as f:
        geojson = json.load(f)
    
    # Determine geometry type
    if "features" not in geojson:
        raise ValueError("Invalid GeoJSON: missing 'features' key")
    
    features = geojson["features"]
    if not features:
        raise ValueError("No features found in GeoJSON")
    
    # Determine shape type from first feature
    geom_type = features[0]["geometry"]["type"]
    
    # Create shapefile writer based on geometry type
    if geom_type == "Point":
        w = shapefile.Writer(shapefile_path, shapefile.POINT)
    elif geom_type in ["LineString", "MultiLineString"]:
        w = shapefile.Writer(shapefile_path, shapefile.POLYLINE)
    elif geom_type in ["Polygon", "MultiPolygon"]:
        w = shapefile.Writer(shapefile_path, shapefile.POLYGON)
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")
    
    # Add fields from properties of first feature
    if "properties" in features[0] and features[0]["properties"]:
        props = features[0]["properties"]
        for field, value in props.items():
            if isinstance(value, int):
                w.field(field, 'N', 10, 0)
            elif isinstance(value, float):
                w.field(field, 'F', 10, 10)
            elif isinstance(value, bool):
                w.field(field, 'L', 1, 0)
            else:
                w.field(field, 'C', 50, 0)
    
    # Process each feature
    for feature in features:
        geometry = feature["geometry"]
        properties = feature.get("properties", {})
        
        # Process geometry
        if geometry["type"] == "Point":
            w.point(geometry["coordinates"][0], geometry["coordinates"][1])
        elif geometry["type"] == "LineString":
            w.line([geometry["coordinates"]])
        elif geometry["type"] == "MultiLineString":
            w.line(geometry["coordinates"])
        elif geometry["type"] == "Polygon":
            w.poly(geometry["coordinates"])
        elif geometry["type"] == "MultiPolygon":
            w.poly(geometry["coordinates"][0])  # Use first polygon
        
        # Add record
        if properties:
            w.record(**properties)
        else:
            w.record()
    
    # Write file
    w.close()
    
    # Write projection file - assume WGS84
    write_projection_file(shapefile_path, 'WGS84')
    
    return shapefile_path + ".shp"


def extract_shapefile_boundary(shapefile_path, output_path=None):
    """
    Extract boundary from a polygon shapefile.
    
    Parameters
    ----------
    shapefile_path : str
        Path to input shapefile
    output_path : str, optional
        Path to output shapefile
        If None, derived from shapefile_path
        
    Returns
    -------
    str
        Path to output shapefile
    """
    try:
        import shapefile
    except ImportError:
        raise ImportError("The pyshp package is required. Please install it with: pip install pyshp")
    
    # Default output path
    if output_path is None:
        base = os.path.splitext(shapefile_path)[0]
        output_path = f"{base}_boundary"
    
    # Read shapefile
    sf = shapefile.Reader(shapefile_path)
    
    # Check shape type
    if sf.shapeType != shapefile.POLYGON:
        raise ValueError("Input shapefile must be a polygon shapefile")
    
    # Create output shapefile
    w = shapefile.Writer(output_path, shapefile.POLYLINE)
    
    # Copy fields
    for field in sf.fields[1:]:  # Skip DeletionFlag
        w.field(field[0], field[1], field[2], field[3])
    
    # Process each shape
    for shape_rec in sf.shapeRecords():
        geometry = shape_rec.shape
        record = shape_rec.record
        
        # Get parts
        parts = geometry.parts
        parts.append(len(geometry.points))
        
        # Extract boundary lines
        lines = []
        for i in range(len(parts)-1):
            lines.append(geometry.points[parts[i]:parts[i+1]])
        
        # Write polyline
        w.line(lines)
        
        # Write record
        w.record(*record)
    
    # Close shapefile
    w.close()
    
    # Write projection file
    write_projection_file(output_path, 'WGS84')
    
    return output_path + ".shp"


def merge_shapefiles(shapefile_paths, output_path):
    """
    Merge multiple shapefiles into one.
    
    Parameters
    ----------
    shapefile_paths : list
        List of paths to shapefiles
    output_path : str
        Path to output shapefile
        
    Returns
    -------
    str
        Path to output shapefile
    """
    try:
        import shapefile
    except ImportError:
        raise ImportError("The pyshp package is required. Please install it with: pip install pyshp")
    
    if not shapefile_paths:
        raise ValueError("No shapefiles provided")
    
    # Remove file extension if present
    if output_path.lower().endswith('.shp'):
        output_path = output_path[:-4]
    
    # Read first shapefile to determine type and fields
    sf1 = shapefile.Reader(shapefile_paths[0])
    shape_type = sf1.shapeType
    
    # Create output shapefile
    w = shapefile.Writer(output_path, shape_type)
    
    # Copy fields from first shapefile
    fields = []
    for field in sf1.fields[1:]:  # Skip DeletionFlag
        w.field(field[0], field[1], field[2], field[3])
        fields.append(field[0])
    
    # Process each shapefile
    for shp_path in shapefile_paths:
        sf = shapefile.Reader(shp_path)
        
        # Check compatible shape type
        if sf.shapeType != shape_type:
            warnings.warn(f"Shapefile {shp_path} has different shape type than first shapefile. Skipping.")
            continue
        
        # Process each shape
        for shape_rec in sf.shapeRecords():
            geometry = shape_rec.shape
            record = shape_rec.record
            
            # Write geometry based on type
            if shape_type == shapefile.POINT:
                w.point(geometry.points[0][0], geometry.points[0][1])
            elif shape_type == shapefile.POLYLINE:
                parts = []
                part_indices = list(geometry.parts) + [len(geometry.points)]
                for i in range(len(geometry.parts)):
                    start = part_indices[i]
                    end = part_indices[i+1]
                    parts.append(geometry.points[start:end])
                w.line(parts)
            elif shape_type == shapefile.POLYGON:
                parts = []
                part_indices = list(geometry.parts) + [len(geometry.points)]
                for i in range(len(geometry.parts)):
                    start = part_indices[i]
                    end = part_indices[i+1]
                    parts.append(geometry.points[start:end])
                w.poly(parts)
            
            # Write record
            # Create a complete record to match fields
            record_dict = {}
            sf_fields = [f[0] for f in sf.fields[1:]]
            for i, field_name in enumerate(sf_fields):
                if field_name in fields:
                    record_dict[field_name] = record[i]
            
            # Fill missing fields with None
            for field_name in fields:
                if field_name not in record_dict:
                    record_dict[field_name] = None
            
            # Write record
            w.record(**record_dict)
    
    # Close shapefile
    w.close()
    
    # Write projection file
    write_projection_file(output_path, 'WGS84')
    
    return output_path + ".shp"


def clip_shapefile_by_bounds(shapefile_path, bounds, output_path=None):
    """
    Clip a shapefile by a bounding box.
    
    Parameters
    ----------
    shapefile_path : str
        Path to input shapefile
    bounds : tuple
        Bounding box (minx, miny, maxx, maxy)
    output_path : str, optional
        Path to output shapefile
        If None, derived from shapefile_path
        
    Returns
    -------
    str
        Path to output shapefile
    """
    try:
        import shapefile
        from shapely.geometry import Polygon, LineString, Point, box
        from shapely.ops import unary_union
    except ImportError:
        missing = []
        try:
            import shapefile
        except ImportError:
            missing.append("pyshp")
        try:
            from shapely.geometry import Polygon, LineString, Point, box
            from shapely.ops import unary_union
        except ImportError:
            missing.append("shapely")
            
        raise ImportError(f"Missing required packages: {', '.join(missing)}. "
                          f"Please install them with: pip install {' '.join(missing)}")
    
    # Default output path
    if output_path is None:
        base = os.path.splitext(shapefile_path)[0]
        output_path = f"{base}_clipped"
    
    # Remove file extension if present
    if output_path.lower().endswith('.shp'):
        output_path = output_path[:-4]
    
    # Create bounding box
    minx, miny, maxx, maxy = bounds
    bbox = box(minx, miny, maxx, maxy)
    
    # Read shapefile
    sf = shapefile.Reader(shapefile_path)
    
    # Create output shapefile
    w = shapefile.Writer(output_path, sf.shapeType)
    
    # Copy fields
    for field in sf.fields[1:]:  # Skip DeletionFlag
        w.field(field[0], field[1], field[2], field[3])
    
    # Process each shape
    for shape_rec in sf.shapeRecords():
        geometry = shape_rec.shape
        record = shape_rec.record
        
        # Convert shapefile geometry to shapely geometry
        if sf.shapeType == shapefile.POINT:
            shapely_geom = Point(geometry.points[0])
        elif sf.shapeType == shapefile.POLYLINE:
            parts = []
            part_indices = list(geometry.parts) + [len(geometry.points)]
            for i in range(len(geometry.parts)):
                start = part_indices[i]
                end = part_indices[i+1]
                parts.append(LineString(geometry.points[start:end]))
            shapely_geom = unary_union(parts)
        elif sf.shapeType == shapefile.POLYGON:
            parts = []
            part_indices = list(geometry.parts) + [len(geometry.points)]
            for i in range(len(geometry.parts)):
                start = part_indices[i]
                end = part_indices[i+1]
                parts.append(Polygon(geometry.points[start:end]))
            shapely_geom = unary_union(parts)
        else:
            warnings.warn(f"Unsupported shape type: {sf.shapeType}. Skipping.")
            continue
        
        # Clip geometry
        clipped_geom = shapely_geom.intersection(bbox)
        
        # Skip if geometry is empty after clipping
        if clipped_geom.is_empty:
            continue
        
        # Convert clipped shapely geometry back to shapefile geometry
        if sf.shapeType == shapefile.POINT:
            if clipped_geom.geom_type == 'Point':
                w.point(clipped_geom.x, clipped_geom.y)
                w.record(*record)
        elif sf.shapeType == shapefile.POLYLINE:
            if clipped_geom.geom_type == 'LineString':
                w.line([[list(coord) for coord in clipped_geom.coords]])
                w.record(*record)
            elif clipped_geom.geom_type == 'MultiLineString':
                w.line([[list(coord) for coord in part.coords] for part in clipped_geom.geoms])
                w.record(*record)
        elif sf.shapeType == shapefile.POLYGON:
            if clipped_geom.geom_type == 'Polygon':
                exterior = [list(coord) for coord in clipped_geom.exterior.coords]
                interiors = [[list(coord) for coord in interior.coords] for interior in clipped_geom.interiors]
                w.poly([exterior] + interiors)
                w.record(*record)
            elif clipped_geom.geom_type == 'MultiPolygon':
                for polygon in clipped_geom.geoms:
                    exterior = [list(coord) for coord in polygon.exterior.coords]
                    interiors = [[list(coord) for coord in interior.coords] for interior in polygon.interiors]
                    w.poly([exterior] + interiors)
                    w.record(*record)
    
    # Close shapefile
    w.close()
    
    # Write projection file
    write_projection_file(output_path, 'WGS84')
    
    return output_path + ".shp"


def shapefile_to_points(shapefile_path, output_path=None, sample_distance=None):
    """
    Convert polygon or polyline shapefile to points.
    
    Parameters
    ----------
    shapefile_path : str
        Path to input shapefile
    output_path : str, optional
        Path to output shapefile
        If None, derived from shapefile_path
    sample_distance : float, optional
        Distance between points for line/polygon
        If None, use vertices
        
    Returns
    -------
    str
        Path to output shapefile
    """
    try:
        import shapefile
        from shapely.geometry import Polygon, LineString, Point, box
    except ImportError:
        missing = []
        try:
            import shapefile
        except ImportError:
            missing.append("pyshp")
        try:
            from shapely.geometry import Polygon, LineString, Point, box
        except ImportError:
            missing.append("shapely")
            
        raise ImportError(f"Missing required packages: {', '.join(missing)}. "
                          f"Please install them with: pip install {' '.join(missing)}")
    
    # Default output path
    if output_path is None:
        base = os.path.splitext(shapefile_path)[0]
        output_path = f"{base}_points"
    
    # Remove file extension if present
    if output_path.lower().endswith('.shp'):
        output_path = output_path[:-4]
    
    # Read shapefile
    sf = shapefile.Reader(shapefile_path)
    
    # Create output shapefile
    w = shapefile.Writer(output_path, shapefile.POINT)
    
    # Copy fields and add identifier field
    for field in sf.fields[1:]:  # Skip DeletionFlag
        w.field(field[0], field[1], field[2], field[3])
    w.field('POINT_ID', 'N', 10, 0)
    
    # Process each shape
    point_id = 1
    for shape_rec in sf.shapeRecords():
        geometry = shape_rec.shape
        record = shape_rec.record
        
        # Extract points based on shape type
        if sf.shapeType == shapefile.POINT:
            # Already a point
            w.point(geometry.points[0][0], geometry.points[0][1])
            
            # Create record with original attributes plus point ID
            record_dict = {field: value for field, value in zip([f[0] for f in sf.fields[1:]], record)}
            record_dict['POINT_ID'] = point_id
            w.record(**record_dict)
            
            point_id += 1
            
        elif sf.shapeType in [shapefile.POLYLINE, shapefile.POLYGON]:
            # Convert to shapely geometry for resampling if needed
            parts = []
            part_indices = list(geometry.parts) + [len(geometry.points)]
            
            for i in range(len(geometry.parts)):
                start = part_indices[i]
                end = part_indices[i+1]
                part_points = geometry.points[start:end]
                
                if sf.shapeType == shapefile.POLYLINE:
                    shapely_geom = LineString(part_points)
                else:
                    shapely_geom = Polygon(part_points)
                
                # Sample points along geometry
                if sample_distance is not None and sample_distance > 0:
                    if sf.shapeType == shapefile.POLYLINE:
                        # Sample along line
                        line_length = shapely_geom.length
                        num_points = max(2, int(line_length / sample_distance))
                        sample_points = [shapely_geom.interpolate(i * line_length / (num_points - 1)) 
                                         for i in range(num_points)]
                    else:
                        # Sample along polygon boundary
                        boundary = shapely_geom.exterior
                        boundary_length = boundary.length
                        num_points = max(2, int(boundary_length / sample_distance))
                        sample_points = [boundary.interpolate(i * boundary_length / (num_points - 1)) 
                                         for i in range(num_points)]
                else:
                    # Use original vertices
                    if sf.shapeType == shapefile.POLYLINE:
                        sample_points = [Point(p) for p in part_points]
                    else:
                        sample_points = [Point(p) for p in shapely_geom.exterior.coords]
                
                # Add points to output
                for point in sample_points:
                    w.point(point.x, point.y)
                    
                    # Create record with original attributes plus point ID
                    record_dict = {field: value for field, value in zip([f[0] for f in sf.fields[1:]], record)}
                    record_dict['POINT_ID'] = point_id
                    w.record(**record_dict)
                    
                    point_id += 1
    
    # Close shapefile
    w.close()
    
    # Write projection file
    write_projection_file(output_path, 'WGS84')
    
    return output_path + ".shp"


def get_shapefile_bounds(shapefile_path):
    """
    Get bounding box of a shapefile.
    
    Parameters
    ----------
    shapefile_path : str
        Path to shapefile
        
    Returns
    -------
    tuple
        (minx, miny, maxx, maxy)
    """
    try:
        import shapefile
    except ImportError:
        raise ImportError("The pyshp package is required. Please install it with: pip install pyshp")
    
    # Read shapefile
    sf = shapefile.Reader(shapefile_path)
    
    # Get bounding box
    return sf.bbox


def shapefile_to_grid(shapefile_path, resolution, output_path=None):
    """
    Convert a polygon shapefile to a regular grid.
    
    Parameters
    ----------
    shapefile_path : str
        Path to polygon shapefile
    resolution : float or tuple
        Grid cell resolution in units of the shapefile
        If float, same resolution in x and y
        If tuple, (x_res, y_res)
    output_path : str, optional
        Path to output file
        If None, derived from shapefile_path
        
    Returns
    -------
    tuple
        (x, y, grid) - Grid coordinates and boolean mask
    """
    try:
        import shapefile
        from shapely.geometry import Polygon, Point
        import numpy as np
    except ImportError:
        missing = []
        try:
            import shapefile
        except ImportError:
            missing.append("pyshp")
        try:
            from shapely.geometry import Polygon, Point
        except ImportError:
            missing.append("shapely")
            
        raise ImportError(f"Missing required packages: {', '.join(missing)}. "
                          f"Please install them with: pip install {' '.join(missing)}")
    
    # Read shapefile
    sf = shapefile.Reader(shapefile_path)
    
    # Check shape type
    if sf.shapeType != shapefile.POLYGON:
        raise ValueError("Input shapefile must be a polygon shapefile")
    
    # Get bounding box
    minx, miny, maxx, maxy = sf.bbox
    
    # Parse resolution
    if isinstance(resolution, (int, float)):
        x_res = y_res = resolution
    else:
        x_res, y_res = resolution
    
    # Create grid
    x = np.arange(minx, maxx + x_res, x_res)
    y = np.arange(miny, maxy + y_res, y_res)
    xx, yy = np.meshgrid(x, y)
    
    # Create combined shapely polygon
    polygons = []
    for shape in sf.shapes():
        parts = []
        part_indices = list(shape.parts) + [len(shape.points)]
        
        for i in range(len(shape.parts)):
            start = part_indices[i]
            end = part_indices[i+1]
            parts.append(Polygon(shape.points[start:end]))
        
        polygons.extend(parts)
    
    # Check each grid point against polygons
    grid = np.zeros((len(y), len(x)), dtype=bool)
    for i in range(len(y)):
        for j in range(len(x)):
            point = Point(x[j], y[i])
            for polygon in polygons:
                if polygon.contains(point):
                    grid[i, j] = True
                    break
    
    # Save grid if output path provided
    if output_path is not None:
        np.savez(output_path, x=x, y=y, grid=grid)
    
    return x, y, grid


def extract_contours_from_shapefile(shapefile_path, field_name, levels=None, output_path=None):
    """
    Extract contour lines from a polygon shapefile with attribute data.
    
    Parameters
    ----------
    shapefile_path : str
        Path to polygon shapefile
    field_name : str
        Field name for contouring
    levels : list, optional
        Contour levels
        If None, 10 evenly spaced levels
    output_path : str, optional
        Path to output shapefile
        If None, derived from shapefile_path
        
    Returns
    -------
    str
        Path to contour shapefile
    """
    try:
        import shapefile
        from shapely.geometry import Polygon, LineString
    except ImportError:
        missing = []
        try:
            import shapefile
        except ImportError:
            missing.append("pyshp")
        try:
            from shapely.geometry import Polygon, LineString
        except ImportError:
            missing.append("shapely")
            
        raise ImportError(f"Missing required packages: {', '.join(missing)}. "
                          f"Please install them with: pip install {' '.join(missing)}")
    
    # Default output path
    if output_path is None:
        base = os.path.splitext(shapefile_path)[0]
        output_path = f"{base}_contours"
    
    # Remove file extension if present
    if output_path.lower().endswith('.shp'):
        output_path = output_path[:-4]
    
    # Read shapefile
    sf = shapefile.Reader(shapefile_path)
    
    # Check shape type
    if sf.shapeType != shapefile.POLYGON:
        raise ValueError("Input shapefile must be a polygon shapefile")
    
    # Get field values
    fields = [field[0] for field in sf.fields[1:]]  # Skip DeletionFlag
    if field_name not in fields:
        raise ValueError(f"Field '{field_name}' not found in shapefile")
    
    field_index = fields.index(field_name)
    values = [rec[field_index] for rec in sf.records()]
    
    # Determine contour levels
    if levels is None:
        min_val = min(values)
        max_val = max(values)
        levels = np.linspace(min_val, max_val, 10)
    
    # Create output shapefile
    w = shapefile.Writer(output_path, shapefile.POLYLINE)
    w.field('LEVEL', 'F', 10, 5)
    
    # Process each contour level
    for level in levels:
        # Find polygons above and below the level
        above_polygons = []
        below_polygons = []
        
        for i, (shape, value) in enumerate(zip(sf.shapes(), values)):
            if value >= level:
                # Convert to shapely polygon
                parts = []
                part_indices = list(shape.parts) + [len(shape.points)]
                
                for j in range(len(shape.parts)):
                    start = part_indices[j]
                    end = part_indices[j+1]
                    parts.append(Polygon(shape.points[start:end]))
                
                above_polygons.extend(parts)
            else:
                # Convert to shapely polygon
                parts = []
                part_indices = list(shape.parts) + [len(shape.points)]
                
                for j in range(len(shape.parts)):
                    start = part_indices[j]
                    end = part_indices[j+1]
                    parts.append(Polygon(shape.points[start:end]))
                
                below_polygons.extend(parts)
        
        # Create contour lines at the boundaries
        contour_lines = []
        
        # Process each above polygon against each below polygon
        for above_poly in above_polygons:
            for below_poly in below_polygons:
                # Check if they share a boundary
                if above_poly.touches(below_poly):
                    # Get the intersection (should be a line)
                    intersection = above_poly.intersection(below_poly)
                    
                    if intersection.geom_type == 'LineString':
                        contour_lines.append(intersection)
                    elif intersection.geom_type == 'MultiLineString':
                        contour_lines.extend(list(intersection.geoms))
        
        # Add contour lines to shapefile
        for line in contour_lines:
            w.line([[list(coord) for coord in line.coords]])
            w.record(LEVEL=level)
    
    # Close shapefile
    w.close()
    
    # Write projection file
    write_projection_file(output_path, 'WGS84')
    
    return output_path + ".shp"


def reproject_shapefile(input_path, output_path, target_projection):
    """
    Reproject a shapefile to a new coordinate system.
    
    Parameters
    ----------
    input_path : str
        Path to input shapefile
    output_path : str
        Path to output shapefile
    target_projection : str
        Target projection name or WKT string
        
    Returns
    -------
    str
        Path to reprojected shapefile
    """
    try:
        import shapefile
        from pyproj import Transformer, CRS
        import shapely.geometry as sg
        from shapely.ops import transform
        import functools
    except ImportError:
        missing = []
        try:
            import shapefile
        except ImportError:
            missing.append("pyshp")
        try:
            from pyproj import Transformer, CRS
        except ImportError:
            missing.append("pyproj")
        try:
            import shapely.geometry as sg
            from shapely.ops import transform
        except ImportError:
            missing.append("shapely")
            
        raise ImportError(f"Missing required packages: {', '.join(missing)}. "
                          f"Please install them with: pip install {' '.join(missing)}")
    
    # Remove file extension if present
    if output_path.lower().endswith('.shp'):
        output_path = output_path[:-4]
    
    # Read input shapefile
    sf = shapefile.Reader(input_path)
    
    # Read projection from .prj file
    prj_path = os.path.splitext(input_path)[0] + '.prj'
    if os.path.exists(prj_path):
        with open(prj_path, 'r') as f:
            source_projection = f.read()
    else:
        raise ValueError("Source projection (.prj) file not found")
    
    # Set up projections
    source_crs = CRS.from_wkt(source_projection)
    
    # Determine target CRS
    if target_projection in ['WGS84', 'NAD83', 'UTM_ZONE17N']:
        # Get from common projections
        projections = {
            'WGS84': 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]',
            'NAD83': 'GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]',
            'UTM_ZONE17N': 'PROJCS["NAD_1983_UTM_Zone_17N",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-81.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]'
        }
        target_crs = CRS.from_wkt(projections[target_projection])
    else:
        # Try parsing as EPSG code or WKT
        target_crs = CRS.from_user_input(target_projection)
    
    # Create transformer
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    
    # Create transformation function
    project = functools.partial(
        transformer.transform,
    )
    
    # Create output shapefile with same shape type
    w = shapefile.Writer(output_path, sf.shapeType)
    
    # Copy fields
    for field in sf.fields[1:]:  # Skip DeletionFlag
        w.field(field[0], field[1], field[2], field[3])
    
    # Process shapes
    for shape_rec in sf.shapeRecords():
        geometry = shape_rec.shape
        record = shape_rec.record
        
        # Process based on shape type
        if sf.shapeType == shapefile.POINT:
            # Convert to shapely Point and transform
            point = sg.Point(geometry.points[0])
            transformed_point = transform(project, point)
            
            # Write transformed point
            w.point(transformed_point.x, transformed_point.y)
            w.record(*record)
            
        elif sf.shapeType == shapefile.POLYLINE:
            # Convert to shapely LineString or MultiLineString and transform
            parts = []
            part_indices = list(geometry.parts) + [len(geometry.points)]
            
            for i in range(len(geometry.parts)):
                start = part_indices[i]
                end = part_indices[i+1]
                line = sg.LineString(geometry.points[start:end])
                transformed_line = transform(project, line)
                parts.append(list(transformed_line.coords))
            
            # Write transformed polyline
            w.line(parts)
            w.record(*record)
            
        elif sf.shapeType == shapefile.POLYGON:
            # Convert to shapely Polygon or MultiPolygon and transform
            parts = []
            part_indices = list(geometry.parts) + [len(geometry.points)]
            
            for i in range(len(geometry.parts)):
                start = part_indices[i]
                end = part_indices[i+1]
                poly = sg.Polygon(geometry.points[start:end])
                transformed_poly = transform(project, poly)
                
                if transformed_poly.geom_type == 'Polygon':
                    exterior = list(transformed_poly.exterior.coords)
                    interiors = [list(interior.coords) for interior in transformed_poly.interiors]
                    parts.append([exterior] + interiors)
                else:
                    # Handle case where transformation creates MultiPolygon
                    for geom in transformed_poly.geoms:
                        exterior = list(geom.exterior.coords)
                        interiors = [list(interior.coords) for interior in geom.interiors]
                        parts.append([exterior] + interiors)
            
            # Write transformed polygon
            for part in parts:
                w.poly(part)
                w.record(*record)
    
    # Close shapefile
    w.close()
    
    # Write projection file
    write_projection_file(output_path, target_projection)
    
    return output_path + ".shp"


def shapefile_to_schism_boundary(shapefile_path, output_path=None):
    """
    Convert a polyline shapefile to SCHISM boundary format.
    
    Parameters
    ----------
    shapefile_path : str
        Path to polyline shapefile
    output_path : str, optional
        Path to output file
        If None, derived from shapefile_path
        
    Returns
    -------
    str
        Path to output file
    """
    try:
        import shapefile
    except ImportError:
        raise ImportError("The pyshp package is required. Please install it with: pip install pyshp")
    
    # Default output path
    if output_path is None:
        base = os.path.splitext(shapefile_path)[0]
        output_path = f"{base}_bnd.in"
    
    # Read shapefile
    sf = shapefile.Reader(shapefile_path)
    
    # Check shape type
    if sf.shapeType != shapefile.POLYLINE:
        raise ValueError("Input shapefile must be a polyline shapefile")
    
    # Get attributes to identify open vs. land boundaries
    open_boundaries = []
    land_boundaries = []
    
    # Check if shapefile has a TYPE field
    fields = [field[0] for field in sf.fields[1:]]  # Skip DeletionFlag
    has_type_field = 'TYPE' in fields
    type_index = fields.index('TYPE') if has_type_field else -1
    
    # Process each shape
    for i, shape_rec in enumerate(sf.shapeRecords()):
        geometry = shape_rec.shape
        record = shape_rec.record
        
        # Determine boundary type
        boundary_type = 0  # Default: land boundary
        if has_type_field:
            type_val = record[type_index]
            if isinstance(type_val, str):
                if type_val.lower() in ['open', 'o']:
                    boundary_type = 1
            elif isinstance(type_val, (int, float)):
                if type_val == 1:
                    boundary_type = 1
        
        # Extract coordinates
        coords = []
        for j in range(len(geometry.points)):
            coords.append((geometry.points[j][0], geometry.points[j][1]))
        
        # Add to appropriate list
        if boundary_type == 1:
            open_boundaries.append(coords)
        else:
            land_boundaries.append(coords)
    
    # Write SCHISM boundary file
    with open(output_path, 'w') as f:
        # Write open boundaries
        f.write(f'{len(open_boundaries)} = Number of open boundaries\n')
        total_open_nodes = sum(len(boundary) for boundary in open_boundaries)
        f.write(f'{total_open_nodes} = Total number of open boundary nodes\n')
        
        for i, boundary in enumerate(open_boundaries):
            f.write(f'{len(boundary)} = Number of nodes for open boundary {i+1}\n')
            for j, (x, y) in enumerate(boundary):
                f.write(f'{j+1}\n')  # Node index (placeholder)
        
        # Write land boundaries
        f.write(f'{len(land_boundaries)} = Number of land boundaries\n')
        total_land_nodes = sum(len(boundary) for boundary in land_boundaries)
        f.write(f'{total_land_nodes} = Total number of land boundary nodes\n')
        
        for i, boundary in enumerate(land_boundaries):
            f.write(f'{len(boundary)} 0 = Number of nodes for land boundary {i+1}\n')
            for j, (x, y) in enumerate(boundary):
                f.write(f'{j+1}\n')  # Node index (placeholder)
    
    return output_path