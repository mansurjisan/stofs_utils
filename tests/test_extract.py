import numpy as np
import xarray as xr
from tempfile import NamedTemporaryFile
from netCDF4 import Dataset
from stofs_utils.processing.extract import extract_2d_slab, write_slab_netcdf

def test_extract_2d_slab():
    # Create synthetic data
    # time, nvrt, np = 3, 5, 4
    # data = np.random.rand(time, nvrt, np)
    time_steps, nvrt, n_nodes = 3, 5, 4

    # Fake weights (interpolating between level 2 and 3)
    data = np.random.rand(time_steps, nvrt, n_nodes)
    k0 = np.full((time_steps, n_nodes), 2)
    alpha = np.random.rand(time_steps, n_nodes)

    # Create a mock xarray Dataset
    ds = xr.Dataset({
        'temp': (('time', 'nvrt', 'node'), data)
    })

    # Run extraction
    slab = extract_2d_slab(ds, 'temp', k0, alpha)

    # Check dimensions
    # assert slab.shape == (time, np)
    assert slab.shape == (time_steps, n_nodes) 
    assert not np.any(np.isnan(slab))


def test_write_slab_netcdf():
    # Dummy data
    nt, n_nodes = 3, 4 # Use n_nodes instead of np
    time_vals = np.arange(nt) # Use np (NumPy) correctly here
    x = np.linspace(-70, -69, n_nodes)
    y = np.linspace(40, 41, n_nodes)
    slab = np.random.rand(nt, n_nodes)
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
            assert ds.variables[varname].shape == (nt, n_nodes) 
            assert np.allclose(ds.variables['time'][:], time_vals)
