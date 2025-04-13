#!/usr/bin/env python3
"""
Script to generate temperature time history file for the St. Lawrence River.

This script uses the stofs_utils package to generate river temperature data
based on air temperature from a NetCDF file using a linear regression model.
"""
import argparse
from datetime import datetime

from stofs_utils.processing.river import get_river_temperature
from stofs_utils.utils.helpers import ensure_directory


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate temperature time history file for rivers"
    )
    parser.add_argument(
        'date',
        type=lambda d: datetime.strptime(d, '%Y-%m-%d-%H'),
        help="Start date in format 'YYYY-MM-DD-HH'",
    )
    parser.add_argument(
        '--air_temp_file',
        type=str,
        default='stofs_3d_atl.t12z.gfs.rad.nc',
        help="NetCDF file with air temperature data"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='TEM_1.th',
        help="Output temperature time history file"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help="Output directory"
    )
    
    return parser.parse_args()


def main():
    """Main function to generate river temperature file."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Ensure output directory exists
    ensure_directory(args.output_dir)
    
    # Set rivers list
    rivers = ['St_lawrence']
    
    # Full output path
    output_path = f"{args.output_dir}/{args.output_file}"
    
    # Generate temperature file
    output_file = get_river_temperature(
        date=args.date,
        air_temp_file=args.air_temp_file,
        rivers=rivers,
        output_file=output_path
    )
    
    print(f"Temperature file generated: {output_file}")


if __name__ == '__main__':
    main()
