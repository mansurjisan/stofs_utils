# source_sink_simple.py
"""
River source/sink generation module for STOFS3D using NWM data.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from ..utils.helpers import ensure_directory


def read_river_mapping(mapping_file):
    """
    Read the river node-to-segment mapping file.

    Parameters
    ----------
    mapping_file : str
        Path to mapping CSV or TXT file

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: node, COMID, [scale, type]
    """
    return pd.read_csv(mapping_file, delim_whitespace=True, comment='#')


def extract_nwm_streamflow(nwm_files, comid_list):
    """
    Extract time series from NWM output for given COMIDs.

    Parameters
    ----------
    nwm_files : list of str
        List of NWM NetCDF files
    comid_list : list of int
        COMIDs to extract

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions (time, comid)
    """
    all_data = []
    for f in nwm_files:
        ds = xr.open_dataset(f)
        sel = ds.sel(feature_id=comid_list)
        all_data.append(sel['streamflow'])
    return xr.concat(all_data, dim='time')


def scale_streamflow(streamflow_ds, mapping_df):
    """
    Apply user-defined scaling factors to streamflow.

    Parameters
    ----------
    streamflow_ds : xarray.Dataset
        NWM streamflow Dataset with dimensions (time, comid)
    mapping_df : pandas.DataFrame
        Mapping with 'COMID' and 'scale' columns

    Returns
    -------
    xarray.DataArray
        Scaled streamflow
    """
    scale = mapping_df.set_index('COMID')['scale'].reindex(streamflow_ds.feature_id.values).fillna(1.0)
    return streamflow_ds * scale.values


def write_nwm_th_files(out_dir, time_array, node_ids, stream_data):
    """
    Write SCHISM river forcing files (vsource.th, vsink.th, msource.th).

    Parameters
    ----------
    out_dir : str
        Output directory
    time_array : ndarray
        1D array of model times (seconds since reference)
    node_ids : list of int
        SCHISM node IDs
    stream_data : ndarray
        2D array of shape (ntime, nnode)
    """
    ensure_directory(out_dir)

    # Determine sinks vs. sources
    types = np.where(stream_data.mean(axis=0) < 0, 'sink', 'source')

    # Write msource.th
    with open(os.path.join(out_dir, 'msource.th'), 'w') as f:
        for i in range(len(node_ids)):
            f.write(f"{i+1} {node_ids[i]}\n")

    # Write vsource.th and vsink.th
    ntime = len(time_array)
    with open(os.path.join(out_dir, 'vsource.th'), 'w') as fs, \
         open(os.path.join(out_dir, 'vsink.th'), 'w') as fk:

        for t in range(ntime):
            fs.write(f"{time_array[t]:.1f} " + " ".join([
                f"{stream_data[t, i]:.3f}" if types[i] == 'source' else "0.000"
                for i in range(len(node_ids))]) + "\n")

            fk.write(f"{time_array[t]:.1f} " + " ".join([
                f"{abs(stream_data[t, i]):.3f}" if types[i] == 'sink' else "0.000"
                for i in range(len(node_ids))]) + "\n")
def source_sink_cli():
    """
    Command-line interface to generate SCHISM source/sink files from NWM streamflow.
    """
    parser = argparse.ArgumentParser(description="Generate SCHISM source/sink files from NWM streamflow data.")
    parser.add_argument('--nwm_files', nargs='+', required=True, help='List of NWM NetCDF files')
    parser.add_argument('--mapping_file', required=True, help='Path to mapping file with COMID and SCHISM node')
    parser.add_argument('--output_dir', required=True, help='Directory to save SCHISM .th files')
    parser.add_argument('--ref_date', type=lambda s: datetime.strptime(s, '%Y-%m-%d-%H'), required=True, help='Reference date (e.g., 2023-08-29-00)')

    args = parser.parse_args()

    # Load mapping
    mapping = read_river_mapping(args.mapping_file)
    comids = mapping['COMID'].values
    node_ids = mapping['node'].values

    # Extract and scale streamflow
    streamflow = extract_nwm_streamflow(args.nwm_files, comids)
    scaled_flow = scale_streamflow(streamflow, mapping)

    # Create time array (seconds since ref_date)
    times = streamflow['time'].values
    ref_time = np.datetime64(args.ref_date)
    seconds_since = (times - ref_time) / np.timedelta64(1, 's')

    # Write SCHISM .th files
    write_nwm_th_files(args.output_dir, seconds_since, node_ids, scaled_flow.values)


if __name__ == '__main__':
    source_sink_cli()
