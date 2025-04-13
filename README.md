# STOFS-utils: Storm Surge and Tide Operational Forecast System Utilities

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mansurjisan/stofs_utils/python-package.yml?branch=main)](https://github.com/mansurjisan/stofs_utils/actions/workflows/python-package.yml)

A Python package providing utilities for working with STOFS3D (Storm Surge and Tide Operational Forecast System) data, including grid handling, coordinate transformations, NetCDF operations, and data processing.

## Features

- **Grid Handling**: Read, write and manipulate SCHISM horizontal and vertical grids
- **Coordinate Transformations**: Convert between different coordinate systems and projections
- **NetCDF Operations**: Read, write, and process NetCDF files used in STOFS3D
- **Data Processing**: Extract sections, process time series, generate visualizations
- **River and Source/Sink Processing**: Handle river inputs and other source/sink data
- **Hotstart File Generation**: Create and manipulate SCHISM hotstart files

## Installation

```bash
# Clone the repository
git clone https://github.com/mansurjisan/stofs_utils.git
cd stofs-utils

# Install the package
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"
```

## Dependencies

Basic dependencies:
- numpy
- scipy
- pandas
- netCDF4
- matplotlib
- pyproj
- xarray

Additional dependencies for specific functionalities:
- shapely
- geopandas
- pyshp

## Package Structure

```
stofs_utils/
├── core/           # Core functionality
│   ├── coordinate_utils.py
│   ├── grid.py
│   ├── time_utils.py
│   └── vertical_grid.py
├── io/             # Input/output utilities
│   ├── ascii_io.py
│   ├── grid_io.py
│   ├── json_utils.py
│   ├── netcdf.py
│   └── shapefile_io.py
├── processing/     # Data processing modules
│   ├── adcirc.py
│   ├── extract.py
│   ├── geojson.py
│   ├── hotstart.py
│   ├── river.py
│   ├── source_sink.py
│   └── stations.py
├── utils/          # General utilities
│   └── helpers.py
└── visualizations/ # Visualization utilities
```

## Usage Examples

### Grid Handling

```python
from stofs_utils.core.grid import SchismGrid
from stofs_utils.core.vertical_grid import SchismVGrid

# Read a horizontal grid
grid = SchismGrid('hgrid.gr3')

# Read a vertical grid
vgrid = SchismVGrid()
vgrid.read_vgrid('vgrid.in')

# Plot the grid
grid.plot_grid(fmt=1)
```

### Coordinate Transformations

```python
from stofs_utils.core.coordinate_utils import proj_pts

# Convert points from lat/lon to UTM
x, y = [longitude], [latitude]
px, py = proj_pts(x, y, prj1='epsg:4326', prj2='epsg:26918')
```

### NetCDF Operations

```python
from stofs_utils.io.netcdf import read_netcdf, write_netcdf

# Read a NetCDF file
data = read_netcdf('output.nc')

# Write a NetCDF file
write_netcdf('processed.nc', data)
```

### Extracting Data

```python
from stofs_utils.processing.extract import extract_slab_forecast

# Extract a 2D slab forecast
output_file = extract_slab_forecast(
    input_dir='model_outputs',
    stack=1,
    date=datetime(2023, 1, 1),
    results_dir='results'
)
```

### Time Series Processing

```python
from stofs_utils.processing.stations import generate_station_timeseries

# Generate station time series
output_file = generate_station_timeseries(
    date=datetime(2023, 1, 1),
    input_dir='model_outputs',
    output_dir='time_series'
)
```

## Command-line Interface

Several modules provide command-line interfaces for direct use:

```bash
# Extract data slabs
python -m stofs_utils.processing.extract slab --input_dir=/path/to/data --stack=1 --date=2023-01-01-00

# Process station data
python -m stofs_utils.processing.stations timeseries --date=2023-01-01-00 --input_dir=/path/to/data --output_dir=results

# Convert SCHISM output to ADCIRC format
python -m stofs_utils.processing.adcirc --input_filename=/path/to/file.nc --output_dir=results
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_extract.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NOAA National Ocean Service
- Virginia Institute of Marine Science
- SCHISM development team