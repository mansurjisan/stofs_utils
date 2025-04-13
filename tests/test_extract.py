import numpy as np
import xarray as xr
from tempfile import NamedTemporaryFile
from netCDF4 import Dataset
from stofs_utils.processing.extract import extract_2d_slab, write_slab_netcdf

def test_extract_2d_slab():
    # Create synthetic data
    time, nvrt, np = 3, 5, 4
    data = np.random.rand(time, nvrt, np)

    # Fake weights (interpolating between level 2 and 3)
    k0 = np.full((time, np), 2)
    alpha = np.random.rand(time, np)

    # Create a mock xarray Dataset
    ds = xr.Dataset({
        'temp': (('time', 'nvrt', 'node'), data)
    })

    # Run extraction
    slab = extract_2d_slab(ds, 'temp', k0, alpha)

    # Check dimensions
    assert slab.shape == (time, np)
    assert not np.any(np.isnan(slab))


def test_write_slab_netcdf():
    # Dummy data
    nt, np = 3, 4
    time = np.arange(nt)
    x = np.linspace(-70, -69, np)
    y = np.linspace(40, 41, np)
    slab = np.random.rand(nt, np)
    varname = "temperature"
    attrs = {
        "long_name": "Sea temperature",
        "units": "degC",
        "time_units": "seconds since 2000-01-01"
    }

    # Create a temporary file
    with NamedTemporaryFile(suffix=".nc") as tmp:
        write_slab_netcdf(tmp.name, time, x, y, slab, varname, attrs)

        # Reopen and verify
        with Dataset(tmp.name) as ds:
            assert varname in ds.variables
            assert ds.variables[varname].shape == (nt, np)
            assert np.allclose(ds.variables['time'][:], time)
