"""
JSON utility functions for STOFS3D

Contains functions for loading, saving, and manipulating JSON data
used in STOFS3D operational forecasting.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import argparse


def load_json(filename):
    """
    Load data from a JSON file.
    
    Parameters
    ----------
    filename : str
        Path to JSON file
        
    Returns
    -------
    dict
        Loaded JSON data
    
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    json.JSONDecodeError
        If the file contains invalid JSON
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, filename, indent=4, ensure_dir=True):
    """
    Save data to a JSON file.
    
    Parameters
    ----------
    data : dict
        Data to save
    filename : str
        Output path
    indent : int, optional
        JSON indentation (default: 4)
    ensure_dir : bool, optional
        Create directory if it doesn't exist (default: True)
        
    Returns
    -------
    str
        Path to saved file
    """
    if ensure_dir:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=indent)
    
    return filename


def load_station_mapping(filename):
    """
    Load station mapping from JSON file.
    
    Parameters
    ----------
    filename : str
        Path to station mapping JSON file
        
    Returns
    -------
    dict
        Station mapping data
    """
    mapping = load_json(filename)
    return mapping


def numpy_to_json_serializable(obj):
    """
    Convert numpy types to JSON-serializable types.
    
    Parameters
    ----------
    obj : object
        Object to convert
        
    Returns
    -------
    object
        JSON-serializable object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    
    def default(self, obj):
        """
        Convert numpy types to JSON-serializable types.
        
        Parameters
        ----------
        obj : object
            Object to convert
            
        Returns
        -------
        object
            JSON-serializable object
        """
        return numpy_to_json_serializable(obj)


def save_json_with_numpy(data, filename, indent=4, ensure_dir=True):
    """
    Save data containing NumPy types to a JSON file.
    
    Parameters
    ----------
    data : dict
        Data to save
    filename : str
        Output path
    indent : int, optional
        JSON indentation (default: 4)
    ensure_dir : bool, optional
        Create directory if it doesn't exist (default: True)
        
    Returns
    -------
    str
        Path to saved file
    """
    if ensure_dir:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=indent, cls=NumpyJSONEncoder)
    
    return filename


def merge_json_files(file_list, output_file=None):
    """
    Merge multiple JSON files into a single dictionary.
    
    Parameters
    ----------
    file_list : list
        List of JSON file paths
    output_file : str, optional
        Path to save merged JSON (default: None, don't save)
        
    Returns
    -------
    dict
        Merged JSON data
    """
    merged_data = {}
    
    for file_path in file_list:
        with open(file_path, 'r') as f:
            data = json.load(f)
            merged_data.update(data)
    
    if output_file:
        save_json(merged_data, output_file)
    
    return merged_data


def json_to_dataframe(json_data, normalize=False):
    """
    Convert JSON data to pandas DataFrame.
    
    Parameters
    ----------
    json_data : dict or str
        JSON data or path to JSON file
    normalize : bool, optional
        Whether to normalize semi-structured JSON data (default: False)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame representation of JSON data
    """
    # Load data if string is provided
    if isinstance(json_data, str):
        json_data = load_json(json_data)
    
    # Convert to DataFrame
    if normalize:
        return pd.json_normalize(json_data)
    else:
        return pd.DataFrame(json_data)


def dataframe_to_json(df, orient='records'):
    """
    Convert pandas DataFrame to JSON.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to convert
    orient : str, optional
        DataFrame to JSON orientation (default: 'records')
        
    Returns
    -------
    dict or list
        JSON data
    """
    return json.loads(df.to_json(orient=orient))


def create_station_json(stations, filename=None):
    """
    Create a station information JSON file for STOFS3D.
    
    Parameters
    ----------
    stations : pandas.DataFrame or dict
        Station information with columns/keys:
        'station_id', 'name', 'lon', 'lat', 'type'
    filename : str, optional
        Output filename (default: None, don't save)
        
    Returns
    -------
    dict
        Station information in JSON format
    """
    # Convert DataFrame to dict if needed
    if isinstance(stations, pd.DataFrame):
        stations_dict = {}
        for _, row in stations.iterrows():
            station_id = row['station_id']
            stations_dict[station_id] = {
                'name': row['name'],
                'lon': float(row['lon']),
                'lat': float(row['lat']),
                'type': row['type']
            }
    else:
        stations_dict = stations
    
    # Save to file if filename provided
    if filename:
        save_json(stations_dict, filename)
    
    return stations_dict


def create_variable_json(variables, filename=None):
    """
    Create a variable information JSON file for STOFS3D.
    
    Parameters
    ----------
    variables : pandas.DataFrame or dict
        Variable information with columns/keys:
        'name', 'long_name', 'standard_name', 'units', 'file_pattern'
    filename : str, optional
        Output filename (default: None, don't save)
        
    Returns
    -------
    dict
        Variable information in JSON format
    """
    # Convert DataFrame to dict if needed
    if isinstance(variables, pd.DataFrame):
        variables_dict = {}
        for _, row in variables.iterrows():
            var_name = row['name']
            variables_dict[var_name] = {
                'long_name': row['long_name'],
                'standard_name': row['standard_name'],
                'units': row['units'],
                'file_pattern': row['file_pattern']
            }
    else:
        variables_dict = variables
    
    # Save to file if filename provided
    if filename:
        save_json(variables_dict, filename)
    
    return variables_dict


def validate_json_schema(json_data, schema):
    """
    Validate JSON data against a schema.
    
    Parameters
    ----------
    json_data : dict
        JSON data to validate
    schema : dict
        JSON schema
        
    Returns
    -------
    bool
        True if valid, raises exception if invalid
        
    Raises
    ------
    jsonschema.exceptions.ValidationError
        If validation fails
    """
    try:
        import jsonschema
        jsonschema.validate(instance=json_data, schema=schema)
        return True
    except ImportError:
        import warnings
        warnings.warn("jsonschema package not installed, skipping validation")
        return True


def find_json_files(directory, pattern="*.json", recursive=False):
    """
    Find JSON files in a directory.
    
    Parameters
    ----------
    directory : str
        Directory to search
    pattern : str, optional
        File pattern (default: "*.json")
    recursive : bool, optional
        Whether to search recursively (default: False)
        
    Returns
    -------
    list
        List of JSON file paths
    """
    import glob
    
    if recursive:
        pattern = os.path.join(directory, "**", pattern)
        files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(directory, pattern)
        files = glob.glob(pattern)
    
    return sorted(files)


def parse_json_dates(json_data, date_fields=None):
    """
    Parse date strings in JSON data to datetime objects.
    
    Parameters
    ----------
    json_data : dict
        JSON data
    date_fields : list or dict, optional
        Fields to parse as dates or mapping of fields to date formats
        
    Returns
    -------
    dict
        JSON data with parsed dates
    """
    if date_fields is None:
        return json_data
    
    result = json_data.copy()
    
    # Handle both list and dict formats for date_fields
    if isinstance(date_fields, list):
        for field in date_fields:
            if field in result:
                result[field] = datetime.fromisoformat(result[field])
    else:  # dict mapping field names to formats
        for field, fmt in date_fields.items():
            if field in result:
                result[field] = datetime.strptime(result[field], fmt)
    
    return result


def process_staout_json(filename, output_filename=None):
    """
    Process a STOFS3D staout configuration JSON file.
    
    Parameters
    ----------
    filename : str
        Input JSON file
    output_filename : str, optional
        Output JSON file (default: None, don't save)
        
    Returns
    -------
    dict
        Processed staout configuration
    """
    # Load the original JSON
    staout_data = load_json(filename)
    
    # Build a standardized staout configuration
    staout_config = {}
    
    for var_name, var_info in staout_data.items():
        staout_config[var_name] = {
            'name': var_info.get('name', var_name),
            'long_name': var_info.get('long_name', var_name),
            'standard_name': var_info.get('standard_name', var_name),
            'units': var_info.get('units', ''),
            'staout_fname': var_info.get('staout_fname', f'staout_{var_name}')
        }
    
    # Save if output filename provided
    if output_filename:
        save_json(staout_config, output_filename)
    
    return staout_config


def flatten_json(json_data, separator='_'):
    """
    Flatten a nested JSON structure.
    
    Parameters
    ----------
    json_data : dict
        Nested JSON data
    separator : str, optional
        Key separator for flattened dictionary (default: '_')
        
    Returns
    -------
    dict
        Flattened JSON data
    """
    def _flatten(current, key='', result=None):
        if result is None:
            result = {}
        
        if isinstance(current, dict):
            for k in current:
                new_key = f"{key}{separator}{k}" if key else k
                _flatten(current[k], new_key, result)
        elif isinstance(current, list):
            for i, item in enumerate(current):
                new_key = f"{key}{separator}{i}" if key else str(i)
                _flatten(item, new_key, result)
        else:
            result[key] = current
        
        return result
    
    return _flatten(json_data)


def unflatten_json(json_data, separator='_'):
    """
    Unflatten a flattened JSON structure.
    
    Parameters
    ----------
    json_data : dict
        Flattened JSON data
    separator : str, optional
        Key separator used in flattened dictionary (default: '_')
        
    Returns
    -------
    dict
        Nested JSON data
    """
    result = {}
    
    for key, value in json_data.items():
        parts = key.split(separator)
        current = result
        
        for i, part in enumerate(parts):
            if i == len(parts) - 1:  # Last part
                current[part] = value
            else:
                if part not in current:
                    # Try to convert to int if it's a digit (for list indices)
                    next_part = parts[i + 1]
                    if next_part.isdigit():
                        current[part] = []
                    else:
                        current[part] = {}
                
                # Handle list indices
                if isinstance(current[part], list):
                    next_part = int(parts[i + 1])
                    while len(current[part]) <= next_part:
                        current[part].append({})
                    current = current[part][next_part]
                else:
                    current = current[part]
    
    return result


def print_json(data, indent=4):
    """
    Pretty print JSON data.
    
    Parameters
    ----------
    data : dict
        JSON data to print
    indent : int, optional
        Indentation level (default: 4)
    """
    print(json.dumps(data, indent=indent, cls=NumpyJSONEncoder))


def compare_json_files(file1, file2, ignore_keys=None):
    """
    Compare two JSON files for structural and content equality.
    
    Parameters
    ----------
    file1 : str
        Path to first JSON file
    file2 : str
        Path to second JSON file
    ignore_keys : list, optional
        Keys to ignore in comparison (default: None)
        
    Returns
    -------
    bool
        True if files are equal (considering ignored keys)
    dict
        Differences between files (empty if equal)
    """
    data1 = load_json(file1)
    data2 = load_json(file2)
    
    differences = {}
    
    def _compare_dict(d1, d2, path=""):
        if ignore_keys and path.split('.')[-1] in ignore_keys:
            return
        
        # Check keys in d1
        for k in d1:
            if ignore_keys and k in ignore_keys:
                continue
                
            new_path = f"{path}.{k}" if path else k
            
            if k not in d2:
                differences[new_path] = (f"Key '{k}' in first file but not in second", d1[k], None)
            elif type(d1[k]) != type(d2[k]):
                differences[new_path] = (f"Type mismatch: {type(d1[k])} vs {type(d2[k])}", d1[k], d2[k])
            elif isinstance(d1[k], dict):
                _compare_dict(d1[k], d2[k], new_path)
            elif isinstance(d1[k], list):
                if len(d1[k]) != len(d2[k]):
                    differences[new_path] = (f"List length mismatch: {len(d1[k])} vs {len(d2[k])}", d1[k], d2[k])
                else:
                    for i, (item1, item2) in enumerate(zip(d1[k], d2[k])):
                        list_path = f"{new_path}[{i}]"
                        if isinstance(item1, dict) and isinstance(item2, dict):
                            _compare_dict(item1, item2, list_path)
                        elif item1 != item2:
                            differences[list_path] = ("List item mismatch", item1, item2)
            elif d1[k] != d2[k]:
                differences[new_path] = ("Value mismatch", d1[k], d2[k])
        
        # Check for keys in d2 that aren't in d1
        for k in d2:
            if ignore_keys and k in ignore_keys:
                continue
                
            if k not in d1:
                new_path = f"{path}.{k}" if path else k
                differences[new_path] = (f"Key '{k}' in second file but not in first", None, d2[k])
    
    _compare_dict(data1, data2)
    return len(differences) == 0, differences


def extract_json_path(json_data, path, separator='.'):
    """
    Extract a value from JSON data using a path string.
    
    Parameters
    ----------
    json_data : dict
        JSON data
    path : str
        Path to the value (e.g., "stations.0.name")
    separator : str, optional
        Path separator (default: '.')
        
    Returns
    -------
    object
        Value at the specified path or None if not found
    """
    parts = path.split(separator)
    current = json_data
    
    for part in parts:
        # Handle array indices
        if part.isdigit() and isinstance(current, list):
            index = int(part)
            if index < len(current):
                current = current[index]
            else:
                return None
        # Handle dictionary keys
        elif part in current:
            current = current[part]
        else:
            return None
    
    return current


def json_utils_cli():
    """Command-line interface for JSON utilities."""
    parser = argparse.ArgumentParser(description='STOFS3D JSON utility tools')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # View JSON command
    view_parser = subparsers.add_parser('view', help='View JSON file contents')
    view_parser.add_argument('file', help='JSON file to view')
    view_parser.add_argument('--path', help='Extract specific path (e.g., stations.0.name)')
    view_parser.add_argument('--indent', type=int, default=4, help='Indentation level')
    
    # Merge JSON command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple JSON files')
    merge_parser.add_argument('files', nargs='+', help='JSON files to merge')
    merge_parser.add_argument('--output', '-o', required=True, help='Output file')
    
    # Validate JSON command
    validate_parser = subparsers.add_parser('validate', help='Validate JSON against schema')
    validate_parser.add_argument('file', help='JSON file to validate')
    validate_parser.add_argument('schema', help='Schema file to validate against')
    
    # Convert JSON to CSV command
    to_csv_parser = subparsers.add_parser('to-csv', help='Convert JSON file to CSV')
    to_csv_parser.add_argument('file', help='JSON file to convert')
    to_csv_parser.add_argument('--output', '-o', required=True, help='Output CSV file')
    to_csv_parser.add_argument('--normalize', '-n', action='store_true', help='Normalize nested JSON')
    
    # Convert CSV to JSON command
    from_csv_parser = subparsers.add_parser('from-csv', help='Convert CSV file to JSON')
    from_csv_parser.add_argument('file', help='CSV file to convert')
    from_csv_parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    from_csv_parser.add_argument('--orient', default='records', 
                              choices=['records', 'columns', 'index', 'split', 'tight'],
                              help='DataFrame to JSON orientation')
    
    # Flatten JSON command
    flatten_parser = subparsers.add_parser('flatten', help='Flatten nested JSON structure')
    flatten_parser.add_argument('file', help='JSON file to flatten')
    flatten_parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    flatten_parser.add_argument('--separator', default='_', help='Key separator for flattened dictionary')
    
    # Unflatten JSON command
    unflatten_parser = subparsers.add_parser('unflatten', help='Unflatten a flattened JSON structure')
    unflatten_parser.add_argument('file', help='JSON file to unflatten')
    unflatten_parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    unflatten_parser.add_argument('--separator', default='_', help='Key separator used in flattened dictionary')
    
    # Compare JSON files command
    compare_parser = subparsers.add_parser('compare', help='Compare two JSON files')
    compare_parser.add_argument('file1', help='First JSON file')
    compare_parser.add_argument('file2', help='Second JSON file')
    compare_parser.add_argument('--ignore-keys', nargs='+', help='Keys to ignore in comparison')
    
    # Process staout JSON command
    staout_parser = subparsers.add_parser('process-staout', help='Process a STOFS3D staout configuration')
    staout_parser.add_argument('file', help='Input staout JSON file')
    staout_parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    
    # Find JSON files command
    find_parser = subparsers.add_parser('find', help='Find JSON files in directory')
    find_parser.add_argument('directory', help='Directory to search')
    find_parser.add_argument('--pattern', default='*.json', help='File pattern')
    find_parser.add_argument('--recursive', '-r', action='store_true', help='Search recursively')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'view':
        data = load_json(args.file)
        if args.path:
            data = extract_json_path(data, args.path)
            if data is None:
                print(f"Path '{args.path}' not found in JSON file")
                return 1
        print_json(data, args.indent)
    
    elif args.command == 'merge':
        merged = merge_json_files(args.files, args.output)
        print(f"Merged {len(args.files)} files into {args.output}")
    
    elif args.command == 'validate':
        try:
            schema = load_json(args.schema)
            data = load_json(args.file)
            valid = validate_json_schema(data, schema)
            if valid:
                print(f"Validation successful: {args.file} is valid according to the schema")
            return 0
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return 1
    
    elif args.command == 'to-csv':
        try:
            df = json_to_dataframe(args.file, normalize=args.normalize)
            df.to_csv(args.output, index=False)
            print(f"Converted {args.file} to CSV: {args.output}")
        except Exception as e:
            print(f"Error converting to CSV: {str(e)}")
            return 1
    
    elif args.command == 'from-csv':
        try:
            df = pd.read_csv(args.file)
            data = dataframe_to_json(df, orient=args.orient)
            save_json(data, args.output)
            print(f"Converted {args.file} to JSON: {args.output}")
        except Exception as e:
            print(f"Error converting from CSV: {str(e)}")
            return 1
    
    elif args.command == 'flatten':
        try:
            data = load_json(args.file)
            flat_data = flatten_json(data, separator=args.separator)
            save_json(flat_data, args.output)
            print(f"Flattened {args.file} to {args.output}")
        except Exception as e:
            print(f"Error flattening JSON: {str(e)}")
            return 1
    
    elif args.command == 'unflatten':
        try:
            data = load_json(args.file)
            nested_data = unflatten_json(data, separator=args.separator)
            save_json(nested_data, args.output)
            print(f"Unflattened {args.file} to {args.output}")
        except Exception as e:
            print(f"Error unflattening JSON: {str(e)}")
            return 1
    
    elif args.command == 'compare':
        equal, differences = compare_json_files(args.file1, args.file2, args.ignore_keys)
        if equal:
            print(f"Files {args.file1} and {args.file2} are identical")
            return 0
        else:
            print(f"Files {args.file1} and {args.file2} have differences:")
            for path, (reason, val1, val2) in differences.items():
                print(f"  {path}: {reason}")
                print(f"    File 1: {val1}")
                print(f"    File 2: {val2}")
            return 1
    
    elif args.command == 'process-staout':
        config = process_staout_json(args.file, args.output)
        print(f"Processed staout configuration {args.file} to {args.output}")
    
    elif args.command == 'find':
        files = find_json_files(args.directory, args.pattern, args.recursive)
        print(f"Found {len(files)} JSON files in {args.directory}:")
        for file in files:
            print(f"  {file}")
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    # Execute CLI when run as a script
    sys.exit(json_utils_cli())