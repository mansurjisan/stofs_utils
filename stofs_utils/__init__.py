"""
STOFS-utils: Utilities for STOFS3D (Storm Tide Operational Forecast System)

This package provides tools for working with STOFS3D data, including grid handling,
coordinate transformations, NetCDF operations, and data processing.
"""

__version__ = "1.0.0"

# Import key modules for easier access
from . import core
from . import io
from . import processing
from . import utils
from . import visualizations