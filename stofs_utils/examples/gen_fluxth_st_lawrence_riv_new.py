#!/usr/bin/env python3
"""
Script to generate flux.th time history file for the St. Lawrence River.
Uses stofs_utils package functions for river discharge processing.
"""
import argparse
from datetime import datetime

from stofs_utils.processing.river import generate_flux_timeseries


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate flux.th file for river discharge"
    )
    parser.add_argument(
        'date',
        type=lambda d: datetime.strptime(d, '%Y-%m-%d-%H'),
        help="Start date in format 'YYYY-MM-DD-HH'",
    )
    parser.add_argument(
        '--river_file',
        type=str,
        default='river_st_law_obs.csv',
        help="CSV file with river discharge data"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='flux.th',
        help="Output flux time history file"
    )
    
    return parser.parse_args()


def main():
    """Main function to generate river flux file."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set rivers list and files dictionary
    rivers = ['St_lawrence']
    river_files = {'St_lawrence': args.river_file}
    
    # Generate flux time history file
    output_file = generate_flux_timeseries(
        date=args.date,
        rivers=rivers,
        river_files=river_files,
        output_file=args.output_file
    )
    
    print(f"Flux file generated: {output_file}")


if __name__ == '__main__':
    main()
