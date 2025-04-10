"""
Source/sink processing utilities for STOFS3D

Contains functions for generating and processing source/sink data for SCHISM,
including source relocation and streamflow lookup.
"""
import os
import numpy as np
import pandas as pd
import json
from netCDF4 import Dataset


class SourceSinkIn:
    """
    Class for handling SCHISM source_sink.in files.
    
    The source_sink.in file defines source and sink elements in SCHISM.
    """
    
    def __init__(self, filename=None, number_of_groups=2, ele_groups=None):
        """
        Initialize a SourceSinkIn instance.
        
        Parameters
        ----------
        filename : str, optional
            Path to source_sink.in file
        number_of_groups : int, optional
            Number of groups (2 for source and sink) (default: 2)
        ele_groups : list, optional
            List of element groups (if not loading from file)
        """
        if ele_groups is None:
            ele_groups = []
            
        self.n_group = number_of_groups
        
        if filename is not None:
            # Read file and initialize from its content
            self.source_file = filename
            self.np_group = []
            self.ip_group = []
            
            with open(self.source_file) as fin:
                for k in range(0, self.n_group):
                    # Read number of points in this group
                    self.np_group.append(int(fin.readline().split()[0]))
                    print(f"Points in Group {k+1}: {self.np_group[k]}")
                    
                    # Read element indices
                    self.ip_group.append(np.empty((self.np_group[k]), dtype=int))
                    for i in range(0, self.np_group[k]):
                        self.ip_group[k][i] = int(fin.readline())
                    
                    # Empty line between groups
                    fin.readline()
                    
                    if self.np_group[k] > 0:
                        print(f"p first: {self.ip_group[k][0]}")
                        print(f"p last: {self.ip_group[k][-1]}")
        else:
            # Initialize from provided element groups
            self.np_group = [len(x) for x in ele_groups]
            self.ip_group = [np.array(x) for x in ele_groups]
    
    def writer(self, filename=None):
        """
        Write source_sink.in file.
        
        Parameters
        ----------
        filename : str, optional
            Output filename (default: self.source_file)
            
        Returns
        -------
        str
            Path to output file
        """
        if filename is None:
            filename = self.source_file
        
        with open(filename, 'w') as fout:
            for k in range(0, self.n_group):
                print(f"Points in Group {k+1}: {self.np_group[k]}")
                fout.write(f"{self.np_group[k]}\n")
                
                for i in range(0, self.np_group[k]):
                    fout.write(f"{self.ip_group[k][i]}\n")
                
                fout.write("\n")  # Empty line
        
        return filename


def relocate_sources(old_source_sink_in, old_vsource, times, outdir=None, relocate_map=None, output_vsource=False):
    """
    Relocate sources based on mapping.
    
    Parameters
    ----------
    old_source_sink_in : SourceSinkIn
        Original source_sink.in object
    old_vsource : numpy.ndarray
        Original vsource data
    times : numpy.ndarray
        Time values
    outdir : str, optional
        Output directory
    relocate_map : numpy.ndarray, optional
        Relocation mapping array
    output_vsource : bool, optional
        Whether to write vsource.th file (default: False)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with relocated vsource data
    """
    if outdir is None:
        outdir = '.'
        
    # Make a dataframe from vsource to facilitate column operations
    df = pd.DataFrame(data=old_vsource, columns=[str(x) for x in old_source_sink_in.ip_group[0]])
    df['time'] = times
    
    # Check relocate_map
    if relocate_map is not None:
        eleids = relocate_map[:, 0]
        new2old_sources = relocate_map[:, 1]
    else:
        raise ValueError('relocate_map is not provided')
    
    # Assemble new source_sink.in
    source_sink_in = SourceSinkIn(
        filename=None, 
        number_of_groups=2, 
        ele_groups=[eleids.tolist(), []]
    )
    source_sink_in.writer(f'{outdir}/source.in')
    
    # Rearrange vsource.th
    map_dict = {
        str(k): str(v) for k, v in np.c_[old_source_sink_in.ip_group[0][new2old_sources], eleids]
    }
    map_dict = {**{'time': 'time'}, **map_dict}
    
    # Check if all columns in mapping exist in the dataframe
    if not set(map_dict.keys()).issubset(set(df.columns)):
        raise ValueError("Some columns in the mapping don't exist in the DataFrame")
    
    # Subset and rename the columns based on the mapping
    df_subset = df[list(map_dict.keys())].rename(columns=map_dict)
    
    # Save the subset dataframe to a new CSV file
    if output_vsource:
        df_subset.to_csv(f'{outdir}/vsource.th', index=False, header=False, sep=' ')
    
    # Create msource.th
    msource = np.c_[
        np.r_[df['time'].iloc[0], df['time'].iloc[-1]],
        np.ones((2, len(eleids)), dtype=int) * -9999,
        np.zeros((2, len(eleids)), dtype=int)
    ]
    np.savetxt(f'{outdir}/msource.th', msource, fmt='%d', delimiter=' ')
    
    return df_subset


def read_featureID_file(filename):
    """
    Read feature IDs from a file.
    
    Parameters
    ----------
    filename : str
        Path to feature ID file
        
    Returns
    -------
    list
        List of feature IDs
    """
    with open(filename) as f:
        lines = f.readlines()
        feature_ids = []
        for line in lines:
            feature_ids.append(line.split('\n')[0])
    
    return feature_ids


def get_aggregated_features(nc_feature_id, features):
    """
    Get aggregated feature indices.
    
    Parameters
    ----------
    nc_feature_id : numpy.ndarray
        Feature IDs from NetCDF file
    features : list
        List of feature lists
        
    Returns
    -------
    list
        Indices of aggregated features
    """
    aggregated_features = []
    for source_feats in features:
        aggregated_features.extend(list(source_feats))
    
    in_file = []
    for feature in aggregated_features:
        idx = np.where(nc_feature_id == int(feature))[0]
        in_file.append(idx.item())
    
    in_file_2 = []
    sidx = 0
    for source_feats in features:
        eidx = sidx + len(source_feats)
        in_file_2.append(in_file[sidx:eidx])
        sidx = eidx
    
    return in_file_2


def streamflow_lookup(file, indexes, threshold=-1e-5):
    """
    Look up streamflow values for specified indices.
    
    Parameters
    ----------
    file : str
        Path to NetCDF file
    indexes : list
        List of indices to look up
    threshold : float, optional
        Minimum threshold for streamflow (default: -1e-5)
        
    Returns
    -------
    list
        Streamflow values
    """
    nc = Dataset(file)
    streamflow = nc["streamflow"][:]
    
    # Apply threshold
    streamflow[np.where(streamflow < threshold)] = 0.0
    
    # Change masked values to zero
    streamflow[np.where(streamflow.mask)] = 0.0
    
    # Get data for each index
    data = []
    for indxs in indexes:
        # Note: Dataset already considers scale factor and offset
        data.append(np.sum(streamflow[indxs]))
    
    nc.close()
    return data


def generate_source_sink(date, basepath='.', layers=None, output_dir=None):
    """
    Generate source/sink files for SCHISM.
    
    Parameters
    ----------
    date : datetime
        Start date for sources
    basepath : str, optional
        Base directory path (default: '.')
    layers : list, optional
        List of layer names (default: ['conus'])
    output_dir : str, optional
        Output directory (default: same as basepath)
        
    Returns
    -------
    dict
        Paths to generated files
    """
    if layers is None:
        layers = ['conus']
    
    if output_dir is None:
        output_dir = basepath
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Old source_sink.in file
    old_source_sink_in = f'{basepath}/source_sink.in.before_relocate'
    
    # Relocate map file
    relocate_map = np.loadtxt(f'{basepath}/relocate_map.txt', dtype=int)
    
    # Source scaling file
    fname_scale = f'{basepath}/source_scale.txt'
    
    # Directory with NWM files
    fdir = basepath
    
    # Initialize arrays
    sources_all = {}
    sinks_all = {}
    eid_sources = []
    eid_sinks = []
    times = []
    dates = []
    
    # Process each layer
    for layer in layers:
        print(f'Layer is {layer}')
        fname_source = f'{basepath}/sources_{layer}.json'
        fname_sink = f'{basepath}/sinks_{layer}.json'
        
        # Load source and sink definitions
        sources_fid = json.load(open(fname_source))
        sinks_fid = json.load(open(fname_sink))
        
        # Add to the final list
        eid_sources.extend(list(sources_fid.keys()))
        eid_sinks.extend(list(sinks_fid.keys()))
        
        # Read NC files
        print(f'{fdir}/nwm*.{layer}.nc')
        files = glob.glob(f'{fdir}/nwm*.{layer}.nc')
        files.sort()
        
        print(f'File 0 is {files[0]}')
        nc_fid0 = Dataset(files[0])["feature_id"][:]
        src_idxs = get_aggregated_features(nc_fid0, sources_fid.values())
        snk_idxs = get_aggregated_features(nc_fid0, sinks_fid.values())
        
        sources = []
        sinks = []
        
        for fname in files:
            ds = Dataset(fname)
            ncfeatureid = ds['feature_id'][:]
            
            # Check if feature IDs changed
            if not np.all(ncfeatureid == nc_fid0):
                print(f'Indexes of feature_id are changed in {fname}')
                src_idxs = get_aggregated_features(ncfeatureid, sources_fid.values())
                snk_idxs = get_aggregated_features(ncfeatureid, sinks_fid.values())
                nc_fid0 = ncfeatureid
            
            # Get streamflow values
            sources.append(streamflow_lookup(fname, src_idxs))
            sinks.append(streamflow_lookup(fname, snk_idxs))
            
            # Get model time
            model_time = datetime.strptime(ds.model_output_valid_time, "%Y-%m-%d_%H:%M:%S")
            if layer == 'conus':
                dates.append(str(model_time))
                times.append((model_time - date).total_seconds())
            
            ds.close()
        
        sources_all[layer] = np.array(sources)
        sinks_all[layer] = np.array(sinks)
    
    # Combine source data
    sources = sources_all['conus']
    sinks = sinks_all['conus']
    
    nsource = np.array(eid_sources).shape[0]
    nsink = np.array(eid_sinks).shape[0]
    print(f'nsource is {nsource}')
    print(f'nsink is {nsink}')
    
    # Write source_sink.in.before_relocate if it doesn't exist
    if not os.path.exists('source_sink.in.before_relocate'):
        print('Writing source_sink.in.before_relocate file...')
        with open('source_sink.in.before_relocate', 'w+') as f:
            f.write('{:<d} \n'.format(nsource))
            for eid in eid_sources:
                f.write('{:<d} \n'.format(int(eid)))
            f.write('\n')
            
            f.write('{:<d} \n'.format(nsink))
            for eid in eid_sinks:
                f.write('{:<d} \n'.format(int(eid)))
    
    # Relocate sources
    source_sink_obj = SourceSinkIn(filename=old_source_sink_in)
    df_vsources = relocate_sources(
        old_source_sink_in=source_sink_obj,
        old_vsource=sources,
        times=np.array(times),
        outdir=output_dir,
        relocate_map=relocate_map
    )
    
    # Apply source scaling if file exists
    if os.path.exists(fname_scale):
        with open(fname_scale) as f:
            total = f.readline().split(' ')[0]
            print(f'Total sources need to be scaled: {total}!')
            
            for line in f.read().splitlines():
                scale_idx = line.split(',')[-2].strip()
                scale_value = float(line.split(',')[-1])
                print(scale_idx)
                print(scale_value)
                
                print(f'Pre-scaling is {df_vsources[scale_idx]}')
                df_vsources[scale_idx] = df_vsources[scale_idx] * scale_value
                print(f'Post-scaling is {df_vsources[scale_idx]}')
    
    # Write vsource.th
    vsource_file = f'{output_dir}/vsource.th'
    df_vsources.to_csv(vsource_file, index=False, header=False, sep=' ', float_format='%.2f')
    
    return {
        'source_in': f'{output_dir}/source.in',
        'vsource': vsource_file,
        'msource': f'{output_dir}/msource.th'
    }


def process_nwm_files(date, input_dir, output_dir, sourcefile='featureID_source.idx', 
                     sinkfile='featureID_sink.idx', pumpfile='pump_sinks.txt'):
    """
    Process NWM files to extract source/sink data for SCHISM.
    
    Parameters
    ----------
    date : datetime
        Start date
    input_dir : str
        Directory with NWM files
    output_dir : str
        Output directory
    sourcefile : str, optional
        File with source feature IDs (default: 'featureID_source.idx')
    sinkfile : str, optional
        File with sink feature IDs (default: 'featureID_sink.idx')
    pumpfile : str, optional
        File with pump data (default: 'pump_sinks.txt')
        
    Returns
    -------
    dict
        Paths to generated files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read feature IDs
    sources_fid = read_featureID_file(os.path.join(input_dir, sourcefile))
    sinks_fid = read_featureID_file(os.path.join(input_dir, sinkfile))
    
    # Find NWM files
    files = glob.glob(os.path.join(input_dir, 'nwm*.nc'))
    files.sort()
    
    # Process files
    sources = []
    sinks = []
    times = []
    
    # Get initial feature IDs
    nc_fid0 = Dataset(files[0])["feature_id"][:]
    src_idxs = get_aggregated_features(nc_fid0, [sources_fid])
    snk_idxs = get_aggregated_features(nc_fid0, [sinks_fid])
    
    for fname in files:
        ds = Dataset(fname)
        ncfeatureid = ds['feature_id'][:]
        
        # Check if feature IDs changed
        if not np.all(ncfeatureid == nc_fid0):
            print(f'Indexes of feature_id are changed in {fname}')
            src_idxs = get_aggregated_features(ncfeatureid, [sources_fid])
            snk_idxs = get_aggregated_features(ncfeatureid, [sinks_fid])
            nc_fid0 = ncfeatureid
        
        # Get streamflow values
        sources.append(streamflow_lookup(fname, src_idxs))
        sinks.append(streamflow_lookup(fname, snk_idxs))
        
        # Get model time
        model_time = datetime.strptime(ds.model_output_valid_time, "%Y-%m-%d_%H:%M:%S")
        times.append((model_time - date).total_seconds())
        
        ds.close()
    
    # Convert to arrays
    sources = np.array(sources)
    sinks = np.array(sinks)
    
    # Apply source scaling
    scale_file = os.path.join(input_dir, 'source_scale.txt')
    if os.path.exists(scale_file):
        with open(scale_file) as f:
            total = f.readline().split(' ')[0]
            for line in f.read().splitlines():
                scale_idx = int(line.split(',')[-2])
                scale_value = float(line.split(',')[-1])
                
                print(f'Pre-scaling is {sources[:, scale_idx-1]}')
                sources[:, scale_idx-1] = sources[:, scale_idx-1] * scale_value
                print(f'Post-scaling is {sources[:, scale_idx-1]}')
    
    # Add pump to sinks if file exists
    if os.path.exists(os.path.join(input_dir, pumpfile)):
        data = np.loadtxt(os.path.join(input_dir, pumpfile))
        sinks = add_pump_to_sink(sinks, -data[:, 1])
    
    # Write output files
    vsource_file = os.path.join(output_dir, 'vsource.th')
    vsink_file = os.path.join(output_dir, 'vsink.th')
    
    write_th_file(sources, times, vsource_file, issource=True)
    write_th_file(sinks, times, vsink_file, issource=False)
    
    return {
        'vsource': vsource_file,
        'vsink': vsink_file
    }


def add_pump_to_sink(sinks, pump):
    """
    Add pump flows to sink data.
    
    Parameters
    ----------
    sinks : list or numpy.ndarray
        Sink flow data
    pump : list or numpy.ndarray
        Pump flow data
        
    Returns
    -------
    list
        Updated sink flow data
    """
    sinks_all = []
    for row in sinks:
        new_row = row.copy() if isinstance(row, list) else row.tolist()
        new_row.extend(pump if isinstance(pump, list) else pump.tolist())
        sinks_all.append(new_row)
    
    return sinks_all


def write_th_file(dataset, timeinterval, fname, issource=True):
    """
    Write time history file.
    
    Parameters
    ----------
    dataset : list or numpy.ndarray
        Flow values
    timeinterval : list or numpy.ndarray
        Time values in seconds
    fname : str
        Output filename
    issource : bool, optional
        True for source, False for sink (default: True)
        
    Returns
    -------
    str
        Path to output file
    """
    data = []
    for values, interval in zip(dataset, timeinterval):
        if issource:
            data.append(" ".join([f"{interval:G}", *[f'{x: .4f}' for x in values], '\n']))
        else:
            data.append(" ".join([f"{interval:G}", *[f'{-x: .4f}' for x in values], '\n']))
            
    with open(fname, 'w+') as fid:
        fid.writelines(data)
        
    return fname
