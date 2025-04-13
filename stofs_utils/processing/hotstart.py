"""
Hotstart file processing for STOFS3D

Contains classes and functions for handling SCHISM hotstart files,
including reading, writing, interpolation, and variable manipulation.
"""
import os
import time
import numpy as np
import xarray as xr
from scipy.sparse import csc_matrix
from scipy.interpolate import griddata
import warnings
import matplotlib.pyplot as plt 
import shapefile

# Local imports
from ..io.grid_io import read_schism_hgrid, read_schism_vgrid
from ..io.netcdf import create_netcdf_dataset, add_dimension, add_variable
from ..core.coordinate_utils import near_pts, inside_polygon 

# Constants
MIN_WATER_DEPTH_H0 = 1e-5
PLOT_LAYER_INDEX = -1 

class Hotstart:
    """
    Class for SCHISM's hotstart.nc.
    
    Different ways to instantiate a Hotstart instance:
        (1) Hotstart(grid_info=some_dir):
            Where some_dir contains hgrid.gr3 and vgrid.in; hotstart variables
            are initialized to 0 mostly.
        
        (2) Hotstart(grid_info=some_dir, hot_file=some_file):
            Same as (1), but with hotstart variable values from some_file.
        
        (3) Hotstart(grid_info={'ne': 2, 'np': 4, 'ns': 5, 'nvrt': 4}):
            Initialize with given dimensions, variables initialized to 0 mostly.
    """
    
    def __init__(self, grid_info, ntracers=2, hot_file=None):
        """
        Initialize a Hotstart instance.
        
        Parameters
        ----------
        grid_info : str or dict
            Either a directory containing grid files or a dict with dimensions
        ntracers : int, optional
            Number of tracers (default: 2)
        hot_file : str, optional
            Path to existing hotstart file to initialize from
        """
        self.file_format = 'NETCDF4'
        self.dimname = ['node', 'elem', 'side', 'nVert', 'ntracers', 'one']
        
        # Set up dimensions
        if isinstance(grid_info, dict):
            self.dims = [grid_info['np'], grid_info['ne'], grid_info['ns'], 
                         grid_info['nvrt'], ntracers, 1]
        elif isinstance(grid_info, str):
            self.source_dir = grid_info
            self.grid = self._create_data_container()
            
            # Load grid information
            
            self.grid.hgrid = read_schism_hgrid(f'{self.source_dir}/hgrid.gr3')
            self.grid.hgrid.compute_side(fmt=1)
            self.grid.vgrid = read_schism_vgrid(f'{self.source_dir}/vgrid.in')
            
            self.dims = [
                self.grid.hgrid.np, 
                self.grid.hgrid.ne, 
                self.grid.hgrid.ns, 
                self.grid.vgrid.nvrt,
                ntracers, 
                1
            ]
            
            # Check hot_file dimensions
            if hot_file is not None:
                try:
                    my_hot = xr.open_dataset(hot_file)
                    if my_hot.dims['elem'] != self.dims[1]:
                        raise Exception(f'Inconsistent geometry: {self.source_dir}')
                    if ntracers != my_hot.dims['ntracers']:
                        ntracers = my_hot.dims['ntracers']
                        print(f'Warning: inconsistent ntracers, setting ntracers={ntracers} based on the value in the hotstart file')
                except Exception as e:
                    print(f"Error opening hotstart file: {e}")
                    hot_file = None
        
        # Define the variables in hotstart.nc
        self.vars = ['time', 'iths', 'ifile', 'idry_e', 'idry_s', 'idry', 'eta2', 'we', 'tr_el',
                     'tr_nd', 'tr_nd0', 'su2', 'sv2', 'q2', 'xl', 'dfv', 'dfh', 'dfq1', 'dfq2',
                     'nsteps_from_cold', 'cumsum_eta']
                     
        # Categorize variables by dimension type
        self.vars_0d = ['time', 'iths', 'ifile', 'nsteps_from_cold']
        self.vars_1d_node_based = ['cumsum_eta']
        self.vars_2d_node_based = ['q2', 'xl', 'dfv', 'dfh', 'dfq1', 'dfq2']
        self.vars_3d_node_based = ['tr_nd', 'tr_nd0']
        self.var_wild = None
        
        # Initialize variables
        if hot_file is None:
            # Initialize with 0
            self.set_var('time', 0.0)
            self.set_var('iths', 0)
            self.set_var('ifile', 1)
            
            self.set_var('idry_e', np.zeros(self.dims[1]))  # all wet
            self.set_var('idry_s', np.zeros(self.dims[2]))  # all wet
            self.set_var('idry', np.zeros(self.dims[0]))  # all wet
            
            self.set_var('eta2', np.zeros(self.dims[0]))
            self.set_var('we', np.zeros((self.dims[1], self.dims[3])))
            self.set_var('tr_el', np.zeros((self.dims[1], self.dims[3], ntracers)))
            self.set_var('tr_nd', np.zeros((self.dims[0], self.dims[3], ntracers)))
            self.set_var('tr_nd0', np.zeros((self.dims[0], self.dims[3], ntracers)))
            self.set_var('su2', np.zeros((self.dims[2], self.dims[3])))
            self.set_var('sv2', np.zeros((self.dims[2], self.dims[3])))
            
            self.set_var('q2', np.zeros((self.dims[0], self.dims[3])))
            self.set_var('xl', np.zeros((self.dims[0], self.dims[3])))
            self.set_var('dfv', np.zeros((self.dims[0], self.dims[3])))
            self.set_var('dfh', np.zeros((self.dims[0], self.dims[3])))
            self.set_var('dfq1', np.zeros((self.dims[0], self.dims[3])))
            self.set_var('dfq2', np.zeros((self.dims[0], self.dims[3])))
            
            self.set_var('nsteps_from_cold', 0)
            self.set_var('cumsum_eta', np.zeros(self.dims[0]))
        else:
            # Read from existing hotstart.nc
            try:
                my_hot = xr.open_dataset(hot_file)
                for var_str in self.vars:
                    try:
                        self.set_var(var_str, my_hot[var_str].data)
                    except KeyError:
                        if var_str in ['nsteps_from_cold', 'cumsum_eta']:
                            self.set_var(var_str, 0)
                        else:
                            print(f"Warning: Variable {var_str} not found in {hot_file}")
            except Exception as e:
                print(f"Error reading hotstart file: {e}")
                # Proceed with default initialization
                self.__init__(grid_info, ntracers)
                
        # Create a dictionary of variables for easier access
        self.var_dict = {}
        for var_str in self.vars:
            if hasattr(self, var_str):
                self.var_dict[var_str] = getattr(self, var_str)
    
    def _create_data_container(self):
        """Create a simple data container class"""
        class DataContainer:
            pass
        return DataContainer()
        
    def set_var(self, var_str="", val=None):
        """
        Set a variable in the hotstart object.
        
        Parameters
        ----------
        var_str : str
            Variable name
        val : array_like
            Variable value
        """
        vi = self._create_data_container()
        
        # Determine variable dimensions
        if var_str in self.vars_0d:
            vi.dimname = ('one',)
        elif var_str == 'idry_e':
            vi.dimname = ('elem',)
        elif var_str == 'idry_s':
            vi.dimname = ('side',)
        elif var_str in ['idry', 'eta2', 'cumsum_eta']:
            vi.dimname = ('node',)
        elif var_str == 'we':
            vi.dimname = ('elem', 'nVert')
        elif var_str in ['su2', 'sv2']:
            vi.dimname = ('side', 'nVert')
        elif var_str in self.vars_2d_node_based:
            vi.dimname = ('node', 'nVert')
        elif var_str == 'tr_el':
            vi.dimname = ('elem', 'nVert', 'ntracers')
        elif var_str in self.vars_3d_node_based:
            vi.dimname = ('node', 'nVert', 'ntracers')
        else:
            raise ValueError(f'{var_str} is not a variable in hotstart.nc')
        
        # Set appropriate data type
        if var_str == 'time':
            vi.val = np.array(val).astype('float64')
        elif var_str in ['iths', 'ifile', 'nsteps_from_cold', 'idry_e', 'idry_s', 'idry']:
            vi.val = np.array(val).astype('int32')
        else:
            vi.val = np.array(val).astype('float64')
        
        # Set the attribute
        setattr(self, var_str, vi)
    
    def interp_from_existing_hotstart(self, hot_in, iplot=False, i_vert_interp=True):
        """
        Interpolate from an existing hotstart file.
        
        This method trades some accuracy for efficiency by using the nearest
        neighbor method in the horizontal dimension.
        
        Parameters
        ----------
        hot_in : Hotstart
            Background hotstart object
        iplot : bool, optional
            Save diagnostic plots if True (default: False)
        i_vert_interp : bool, optional
            Perform vertical interpolation if True (default: True)
            
        Returns
        -------
        None
            Updates the current Hotstart object in-place
        """
        print(f'Using {int(os.getenv("SLURM_CPUS_PER_TASK", 1))} processors')
        print(f'Using {int(os.getenv("OMP_NUM_THREADS", 1))} threads')
        print(f'Using {int(os.getenv("SLURM_NTASKS", 1))} tasks')
        
        h0 = 1e-5  # minimum water depth
        plot_layer = -1  # surface layer for plots
        t = np.time.time()
        t0 = t
        
        in_dir = hot_in.source_dir
        out_dir = self.source_dir
        
        # Read base hotstart
        eta2_in = np.array(hot_in.eta2.val)
        
        # Read and prepare grids
        hot_in.grid = gather_grid_info(hot_in.grid, eta=eta2_in, dir=in_dir)
        grid_in = hot_in.grid
        
        grid_out = self.grid
        
        # Interpolate elevation
        eta2 = griddata(
            np.c_[grid_in.hgrid.x, grid_in.hgrid.y], 
            hot_in.eta2.val, 
            (grid_out.hgrid.x, grid_out.hgrid.y), 
            method='linear'
        )
        
        # Use nearest neighbor for NaN values
        eta2_tmp = griddata(
            np.c_[grid_in.hgrid.x, grid_in.hgrid.y], 
            hot_in.eta2.val, 
            (self.grid.hgrid.x, self.grid.hgrid.y), 
            method='nearest'
        )
        eta2[np.isnan(eta2)] = eta2_tmp[np.isnan(eta2)]
        
        # Update grid with new elevation
        grid_out = gather_grid_info(grid_out, eta=eta2, dir=out_dir)
        
        print(f'Reading inputs and calculating geometry took {np.time.time()-t} seconds')
        t = np.time.time()
                
        neighbor = near_pts(
            np.c_[grid_out.hgrid.x, grid_out.hgrid.y], 
            np.c_[grid_in.hgrid.x, grid_in.hgrid.y]
        )
        
        neighbor_e = near_pts(
            np.c_[grid_out.hgrid.xctr, grid_out.hgrid.yctr], 
            np.c_[grid_in.hgrid.xctr, grid_in.hgrid.yctr]
        )
        
        neighbor_s = near_pts(
            np.c_[grid_out.hgrid.side_x, grid_out.hgrid.side_y], 
            np.c_[grid_in.hgrid.side_x, grid_in.hgrid.side_y]
        )
        
        print(f'Finding nearest neighbors for nodes/elements/sides took {np.time.time()-t} seconds')
        t = np.time.time()
        
        # Generate diagnostic plots
        if iplot:
            
            plt.figure()
            plt.scatter(grid_in.hgrid.side_x, grid_in.hgrid.side_y, s=5, 
                       c=hot_in.su2.val[:, PLOT_LAYER_INDEX], cmap='jet', vmin=0, vmax=2)
            plt.savefig(f'{out_dir}/su2_in2.png', dpi=700)
            
            plt.figure()
            plt.scatter(grid_in.hgrid.x, grid_in.hgrid.y, s=5, 
                      c=hot_in.tr_nd.val[:, plot_layer, 0], cmap='jet', vmin=0, vmax=33)
            plt.savefig(f'{out_dir}/trnd_in2.png', dpi=700)
            
            plt.figure()
            plt.scatter(grid_in.hgrid.xctr, grid_in.hgrid.yctr, s=5, 
                      c=hot_in.tr_el.val[:, plot_layer, 0], cmap='jet', vmin=0, vmax=33)
            plt.savefig(f'{out_dir}/trel_in2.png', dpi=700)
            
            print(f'Generating diagnostic outputs for hot_in took {np.time.time()-t} seconds')
            t = np.time.time()
        
        # Calculate vertical interpolation weights
        if i_vert_interp:
            if not hasattr(grid_out.vgrid, "z_weight_lower"):
                [
                    grid_out.vgrid.z_weight_lower, 
                    grid_out.vgrid.z_idx_lower, 
                    grid_out.vgrid.z_idx_upper
                ] = get_vertical_weight(
                    grid_in.vgrid.zcor, 
                    grid_in.vgrid.kbp, 
                    grid_out.vgrid.zcor, 
                    neighbors=neighbor
                )
            
            print(f'Calculating node-based vertical interpolation weights took {np.time.time()-t} seconds')
            t = np.time.time()
            
            if not hasattr(grid_out.vgrid, "ze_weight_lower"):
                [
                    grid_out.vgrid.ze_weight_lower, 
                    grid_out.vgrid.ze_idx_lower, 
                    grid_out.vgrid.ze_idx_upper
                ] = get_vertical_weight(
                    grid_in.vgrid.zcor_e, 
                    grid_in.vgrid.kbp_e, 
                    grid_out.vgrid.zcor_e, 
                    neighbors=neighbor_e
                )
            
            print(f'Calculating element-based vertical interpolation weights took {np.time.time()-t} seconds')
            t = np.time.time()
            
            if not hasattr(grid_out.vgrid, "zs_weight_lower"):
                [
                    grid_out.vgrid.zs_weight_lower, 
                    grid_out.vgrid.zs_idx_lower, 
                    grid_out.vgrid.zs_idx_upper
                ] = get_vertical_weight(
                    grid_in.vgrid.zcor_s, 
                    grid_in.vgrid.kbp_s, 
                    grid_out.vgrid.zcor_s, 
                    neighbors=neighbor_s
                )
            
            print(f'Calculating side-based vertical interpolation weights took {np.time.time()-t} seconds')
            t = np.time.time()
        
        # Set dry nodes and elements based on eta2
        self.idry.val = (eta2 < -grid_out.hgrid.dp + MIN_WATER_DEPTH_H0).astype('int32')
        
        # An element is wet if and only if depths at all nodes > h0
        self.idry_e.val = np.ones(grid_out.hgrid.ne).astype('int32')
        
        for i34 in [3, 4]:
            idx = (grid_out.hgrid.i34 == i34)
            self.idry_e.val[idx] = np.amax(
                self.idry.val[grid_out.hgrid.elnode[idx, 0:i34]], 
                axis=1
            ).astype(int)
        
        # A side is wet if and only if both nodes are wet
        self.idry_s.val = np.amax(
            self.idry.val[grid_out.hgrid.isidenode], 
            axis=1
        ).astype('int32')
        
        # Set surface elevation
        self.eta2.val = eta2
        
        print(f'Setting dry indicators took {np.time.time()-t} seconds')
        t = np.time.time()
        
        # Process variables
        for var_str in self.vars:
            if var_str in ['eta2', 'idry', 'idry_e', 'idry_s']:
                continue  # already set
            elif var_str in self.vars_0d:  # single number
                self.var_dict[var_str].val = hot_in.var_dict[var_str].val
            elif var_str in self.vars_1d_node_based:
                self.var_dict[var_str].val = hot_in.var_dict[var_str].val[neighbor]
            elif var_str in self.vars_2d_node_based + self.vars_3d_node_based:
                if i_vert_interp:
                    continue  # to be handled later
                else:
                    self.var_dict[var_str].val = hot_in.var_dict[var_str].val[neighbor]
            elif var_str in ['we', 'tr_el']:
                if i_vert_interp:
                    continue  # to be handled later
                else:
                    self.var_dict[var_str].val = hot_in.var_dict[var_str].val[neighbor_e]
            elif var_str in ['su2', 'sv2']:
                if i_vert_interp:
                    continue  # to be handled later
                else:
                    self.var_dict[var_str].val = hot_in.var_dict[var_str].val[neighbor_s]
            else:
                raise ValueError(f'{var_str} not in list')
            
            print(f'Processing {var_str} took {np.time.time()-t} seconds')
            t = np.time.time()
        
        if i_vert_interp:
            # Process we and tr_el
            we_tmp = hot_in.we.val[neighbor_e]
            trel_tmp = hot_in.tr_el.val[neighbor_e]
            
            print(f'Reading we and tr_el took {np.time.time()-t} seconds')
            t = np.time.time()
            
            row = np.r_[np.array(range(self.dims[1])), np.array(range(self.dims[1]))]
            
            for k in range(self.dims[3]):
                col = np.r_[
                    grid_out.vgrid.ze_idx_lower[:, k], 
                    grid_out.vgrid.ze_idx_upper[:, k]
                ]
                
                data = np.r_[
                    grid_out.vgrid.ze_weight_lower[:, k], 
                    1.0 - grid_out.vgrid.ze_weight_lower[:, k]
                ]
                
                weights = csc_matrix(
                    (data, (row, col)), 
                    shape=(self.dims[1], hot_in.dims[3])
                ).toarray()
                
                self.we.val[:, k] = np.sum(we_tmp * weights, axis=1)
                
                for j in range(self.dims[4]):
                    self.tr_el.val[:, k, j] = np.sum(trel_tmp[:, :, j] * weights, axis=1)
                
                print(f'Processing Layer {k+1} of {self.dims[3]} for we and tr_el took {np.time.time()-t} seconds')
                t = np.time.time()
            
            if iplot:
                plt.figure()
                plt.scatter(grid_out.hgrid.xctr, grid_out.hgrid.yctr, s=5, 
                          c=self.tr_el.val[:, plot_layer, 0], cmap='jet', vmin=0, vmax=33)
                plt.savefig(f'{out_dir}/trel_out2.png', dpi=700)
                
                print(f'Generating diagnostic outputs for tr_el took {np.time.time()-t} seconds')
                t = np.time.time()
            
            # Process su2, sv2
            row = np.r_[np.array(range(self.dims[2])), np.array(range(self.dims[2]))]
            
            for k in range(self.dims[3]):
                col = np.r_[
                    grid_out.vgrid.zs_idx_lower[:, k], 
                    grid_out.vgrid.zs_idx_upper[:, k]
                ]
                
                data = np.r_[
                    grid_out.vgrid.zs_weight_lower[:, k], 
                    1.0 - grid_out.vgrid.zs_weight_lower[:, k]
                ]
                
                weights = csc_matrix(
                    (data, (row, col)), 
                    shape=(self.dims[2], hot_in.dims[3])
                ).toarray()
                
                self.su2.val[:, k] = np.sum(
                    hot_in.var_dict['su2'].val[neighbor_s] * weights, 
                    axis=1
                )
                
                self.sv2.val[:, k] = np.sum(
                    hot_in.var_dict['sv2'].val[neighbor_s] * weights, 
                    axis=1
                )
                
                print(f'Processing Layer {k+1} of {self.dims[3]} for su2 and sv2 took {np.time.time()-t} seconds')
                t = np.time.time()
            
            if iplot:
                plt.figure()
                plt.scatter(grid_out.hgrid.side_x, grid_out.hgrid.side_y, s=5, 
                          c=self.su2.val[:, plot_layer], cmap='jet', vmin=0, vmax=2)
                plt.savefig(f'{out_dir}/su2_out2.png', dpi=700)
                
                print(f'Generating diagnostic outputs for su2 took {np.time.time()-t} seconds')
                t = np.time.time()
            
            # Process node-based variables
            trnd_tmp = hot_in.tr_nd.val[neighbor]
            row = np.r_[np.array(range(self.dims[0])), np.array(range(self.dims[0]))]
            
            for k in range(self.dims[3]):
                col = np.r_[
                    grid_out.vgrid.z_idx_lower[:, k], 
                    grid_out.vgrid.z_idx_upper[:, k]
                ]
                
                data = np.r_[
                    grid_out.vgrid.z_weight_lower[:, k], 
                    1.0 - grid_out.vgrid.z_weight_lower[:, k]
                ]
                
                weights = csc_matrix(
                    (data, (row, col)), 
                    shape=(self.dims[0], hot_in.dims[3])
                ).toarray()
                
                for var_str in self.vars_2d_node_based:
                    self.var_dict[var_str].val[:, k] = np.sum(
                        hot_in.var_dict[var_str].val[neighbor] * weights, 
                        axis=1
                    )
                
                for j in range(self.dims[4]):  # loop ntracers
                    self.tr_nd.val[:, k, j] = np.sum(trnd_tmp[:, :, j] * weights, axis=1)
                
                print(f'Processing Layer {k+1} of {self.dims[3]} for all N-dimensional node-based variables took {np.time.time()-t} seconds')
                t = np.time.time()
            
            self.tr_nd0.val = self.tr_nd.val[:]
            
            if iplot:
                plt.figure()
                plt.scatter(grid_out.hgrid.x, grid_out.hgrid.y, s=5, 
                          c=self.tr_nd.val[:, plot_layer, 0], cmap='jet', vmin=0, vmax=33)
                plt.savefig(f'{out_dir}/trnd_out2.png', dpi=700)
                
                print(f'Generating diagnostic outputs for trnd took {np.time.time()-t} seconds')
                t = np.time.time()
            
            print(f'Total time for interpolation: {np.time.time()-t0} seconds')
    
    def trnd_propogate(self):
        """
        Propagate tracer data from nodes to elements.
        """
        for i in range(self.dims[4]):  # For each tracer
            for j in range(self.dims[3]):  # For each layer
                tmp_ele_vals = self._interp_node_to_elem(self.tr_nd.val[:, j, i])
                self.tr_el.val[:, j, i] = tmp_ele_vals
        
        # Copy tr_nd to tr_nd0
        self.tr_nd0.val[:] = self.tr_nd.val[:]
    
    def _interp_node_to_elem(self, values):
        """
        Interpolate values from nodes to elements.
        
        Parameters
        ----------
        values : numpy.ndarray
            Values at nodes
            
        Returns
        -------
        numpy.ndarray
            Values at elements
        """
        # Get triangles and quads
        fp3 = self.grid.hgrid.i34 == 3
        fp4 = ~fp3
        
        # Initialize output array
        ele_vals = np.zeros(self.grid.hgrid.ne)
        
        # Interpolate for triangles (average of 3 nodes)
        ele_vals[fp3] = values[self.grid.hgrid.elnode[fp3, :3]].mean(axis=1)
        
        # Interpolate for quads (average of 4 nodes)
        ele_vals[fp4] = values[self.grid.hgrid.elnode[fp4]].mean(axis=1)
        
        return ele_vals
    
    def writer(self, fname):
        """
        Write hotstart data to a NetCDF file.
        
        Parameters
        ----------
        fname : str
            Output file path
        """
        
        # Create NetCDF file
        ds = create_netcdf_dataset(fname, file_format=self.file_format)
        
        # Create dimensions
        for i, dim_name in enumerate(self.dimname):
            add_dimension(ds, dim_name, self.dims[i])
        
        # Write variables
        for var_str in self.vars:
            var_obj = getattr(self, var_str)
            
            if var_str in self.vars_0d:
                add_variable(ds, var_str, var_obj.val.dtype, var_obj.dimname, var_obj.val)
            else:
                add_variable(ds, var_str, var_obj.val.dtype, var_obj.dimname, var_obj.val)
        
        ds.close()
    
    def replace_vars(self, var_dict=None, shapefile_name=None):
        """
        Replace variables within specific regions.
        
        Parameters
        ----------
        var_dict : dict, optional
            Dictionary of variables to replace
        shapefile_name : str, optional
            Path to shapefile defining regions
        """
        if var_dict is None:
            var_dict = {}
            
        if not hasattr(self.grid, 'hgrid'):
            raise Exception('Missing hgrid, initialize Hotstart instance with hgrid and try again.')
            
        if shapefile_name is not None:
            ele_idx_list, node_idx_list = find_ele_node_in_shpfile(
                shapefile_name=shapefile_name,
                grid=self.grid.hgrid,
            )
        else:
            return
            
        for var in var_dict.keys():
            if self.var_dict[var].val.shape != var_dict[var].shape:
                raise Exception(f'Inconsistent dimensions for {var}')
                
            if self.var_dict[var].dimname[0] == 'elem':
                for ind in ele_idx_list:
                    self.var_dict[var].val[ind] = var_dict[var][ind]
            elif self.var_dict[var].dimname[0] == 'node':
                for ind in node_idx_list:
                    self.var_dict[var].val[ind] = var_dict[var][ind]
            else:
                raise ValueError(f'Operation not implemented for dimension {self.var_dict[var].dimname}')


def gather_grid_info(grid_in, eta, dir):
    """
    Gather and compute necessary grid information.
    
    Parameters
    ----------
    grid_in : DataContainer
        Grid container object
    eta : numpy.ndarray
        Surface elevation at nodes
    dir : str
        Directory containing grid files
        
    Returns
    -------
    DataContainer
        Updated grid container with additional information
    """
    # Load horizontal grid if needed
    if not hasattr(grid_in, 'hgrid'):
        grid_in.hgrid = read_schism_hgrid(f'{dir}/hgrid.gr3')
    
    # Compute element centers
    grid_in.hgrid.compute_ctr()
    
    # Compute side information
    grid_in.hgrid.compute_side(fmt=1)
    
    # Calculate side centers
    grid_in.hgrid.side_x = (grid_in.hgrid.x[grid_in.hgrid.isidenode[:, 0]] + 
                            grid_in.hgrid.x[grid_in.hgrid.isidenode[:, 1]]) / 2.0
    grid_in.hgrid.side_y = (grid_in.hgrid.y[grid_in.hgrid.isidenode[:, 0]] + 
                            grid_in.hgrid.y[grid_in.hgrid.isidenode[:, 1]]) / 2.0
    
    # Load vertical grid if needed
    if not hasattr(grid_in, 'vgrid'):
        grid_in.vgrid = read_schism_vgrid(f'{dir}/vgrid.in')
    
    # Compute z-coordinates
    grid_in.vgrid.zcor = grid_in.vgrid.compute_zcor(grid_in.hgrid.dp, eta=eta)
    
    # Compute side z-coordinates
    grid_in.vgrid.zcor_s = (grid_in.vgrid.zcor[grid_in.hgrid.isidenode[:, 0]] + 
                            grid_in.vgrid.zcor[grid_in.hgrid.isidenode[:, 1]]) / 2.0
    grid_in.vgrid.kbp_s = np.min(grid_in.vgrid.kbp[grid_in.hgrid.isidenode[:, :]], axis=1)
    
    # Compute element z-coordinates
    grid_in.vgrid.zcor_e = np.zeros((grid_in.hgrid.ne, grid_in.vgrid.nvrt))
    grid_in.vgrid.kbp_e = np.zeros((grid_in.hgrid.ne)).astype(int)
    
    # Handle different element types
    for i in [3, 4]:
        # Find elements of type i (triangles or quads)
        II = (grid_in.hgrid.i34 == i)
        
        # Calculate element z-coordinates as average of node z-coordinates
        for j in range(i):
            grid_in.vgrid.zcor_e[II, :] += grid_in.vgrid.zcor[grid_in.hgrid.elnode[II, j], :] / i
        
        # Set bottom layer for elements
        grid_in.vgrid.kbp_e[II] = np.min(grid_in.vgrid.kbp[grid_in.hgrid.elnode[II, :i]], axis=1)
    
    return grid_in


def get_vertical_weight(zcor_in, kbp_in, zcor_out, neighbors):
    """
    Calculate vertical interpolation weights.
    
    Parameters
    ----------
    zcor_in : numpy.ndarray
        Source z-coordinates
    kbp_in : numpy.ndarray
        Source bottom layer indices
    zcor_out : numpy.ndarray
        Target z-coordinates
    neighbors : numpy.ndarray
        Horizontal neighbor indices
        
    Returns
    -------
    tuple
        (z_weight_lower, z_idx_lower, z_idx_upper) - Weights and indices
    """
    # Initialize weight and index arrays
    z_weight_lower = 0.0 * zcor_out
    z_idx_lower = (0.0 * zcor_out).astype(int)
    z_idx_upper = (0.0 * zcor_out).astype(int)
    
    # Get z-coordinates at neighbor locations
    zcor_tmp = zcor_in[neighbors]
    
    # Get dimensions
    n_points = zcor_out.shape[0]
    nvrt_in = zcor_in.shape[1]
    
    # Calculate layer thickness
    dz = zcor_tmp[:, 1:] - zcor_tmp[:, :-1]
    
    # Loop through all points
    for i in range(n_points):
        # Find lower index for each z-level
        l_idx = np.searchsorted(zcor_tmp[i], zcor_out[i]) - 1
        
        # Handle special cases
        below = l_idx == -1  # Point is below bottom
        interior = (l_idx >= 0) & (l_idx < nvrt_in - 1)  # Point is between layers
        above = (l_idx == (nvrt_in - 1))  # Point is above top layer
        
        # Calculate weights for interior points
        z_weight_lower[i, interior] = (
            (zcor_tmp[i, l_idx[interior] + 1] - zcor_out[i, interior]) / 
            dz[i, l_idx[interior]]
        )
        z_idx_lower[i, interior] = l_idx[interior]
        z_idx_upper[i, interior] = l_idx[interior] + 1
        
        # Handle points below bottom
        z_weight_lower[i, below] = 0.0
        z_idx_lower[i, below] = kbp_in[neighbors[i]]
        z_idx_upper[i, below] = kbp_in[neighbors[i]]
        
        # Handle points above top
        z_weight_lower[i, above] = 1.0
        z_idx_lower[i, above] = l_idx[above]
        z_idx_upper[i, above] = l_idx[above]
    
    return [z_weight_lower, z_idx_lower, z_idx_upper]


def find_ele_node_in_shpfile(shapefile_name, grid):
    """
    Find element/node indices within polygons defined in a shapefile.
    
    Parameters
    ----------
    shapefile_name : str
        Path to shapefile containing one or more polygons
    grid : SchismGrid
        SCHISM grid object
        
    Returns
    -------
    tuple
        (ele_ind_list, node_ind_list) - Lists of element and node indices
    """
    
    # Make sure element centers are computed
    if not hasattr(grid, 'xctr'):
        grid.compute_ctr()
    
    # Read shapefile
    sf = shapefile.Reader(shapefile_name)
    shapes = sf.shapes()
    
    # Find elements in each polygon
    ele_ind_list = []
    for shp in shapes:
        poly_xy = np.array(shp.points).T
        ind = inside_polygon(np.c_[grid.xctr, grid.yctr], poly_xy[0], poly_xy[1])
        ind = ind.astype('bool')
        ele_ind_list.append(ind)
    
    # Find nodes in each polygon
    node_ind_list = []
    for shp in shapes:
        poly_xy = np.array(shp.points).T
        ind = inside_polygon(np.c_[grid.x, grid.y], poly_xy[0], poly_xy[1])
        ind = ind.astype('bool')
        node_ind_list.append(ind)
    
    return [ele_ind_list, node_ind_list]


def replace_hot_vars(infile, outfile, grid, vars_list=None, shapefile_name=None):
    """
    Replace variables in a hotstart file within specified regions.
    
    Parameters
    ----------
    infile : str
        Path to input hotstart file
    outfile : str
        Path to output hotstart file
    grid : SchismGrid
        SCHISM grid object
    vars_list : list, optional
        List of variables to replace
    shapefile_name : str, optional
        Path to shapefile defining regions
        
    Returns
    -------
    str
        Path to new hotstart file
    """
    if vars_list is None:
        vars_list = []
        
    # Open input and output files
    hot_in = xr.open_dataset(infile)
    hot_out = xr.open_dataset(outfile)
    
    # Check dimensions
    for dim_name in ['elem', 'side', 'node', 'nVert', 'ntracers']:
        if hot_in.dims[dim_name] != hot_out.dims[dim_name]:
            raise ValueError(f'{dim_name} dimensions not equal')
    
    # Check grid consistency
    if grid.ne != hot_in.dims['elem']:
        raise ValueError('Grid not consistent with hotstart.nc')
    
    # Find elements and nodes in shapefile
    if shapefile_name is not None:
        ele_idx_list, node_idx_list = find_ele_node_in_shpfile(
            shapefile_name=shapefile_name,
            grid=grid,
        )
    else:
        raise ValueError("shapefile_name must be provided")
    
    # Replace variables
    for var in vars_list:
        grid.n_points = hot_out[var].data.shape[0]
        
        if grid.n_points == hot_in.dims['elem']:
            for ind in ele_idx_list:
                hot_out[var].data[ind] = hot_in[var].data[ind]
        elif grid.n_points == hot_in.dims['node']:
            for ind in node_idx_list:
                hot_out[var].data[ind] = hot_in[var].data[ind]
        else:
            raise ValueError(f'Unknown dimension {grid.n_points}')
    
    # Save new file
    new_file = f"{outfile}.new"
    hot_out.to_netcdf(new_file)
    
    # Close datasets
    hot_in.close()
    hot_out.close()
    
    return new_file


def clone_hotstart(hot_file, grid_info, output_file=None):
    """
    Create a new hotstart file based on an existing one but with different grid.
    
    Parameters
    ----------
    hot_file : str
        Path to existing hotstart file
    grid_info : str
        Path to directory containing grid files
    output_file : str, optional
        Path to output file (default: './hotstart.nc')
        
    Returns
    -------
    Hotstart
        New hotstart object
    """
    # Load existing hotstart file
    src_hot = xr.open_dataset(hot_file)
    
    # Get number of tracers
    ntracers = src_hot.dims['ntracers']
    
    # Create new hotstart object
    new_hot = Hotstart(grid_info=grid_info, ntracers=ntracers)
    
    # Copy values from source hot that don't depend on grid dimensions
    for var in new_hot.vars_0d:
        if var in src_hot:
            new_hot.set_var(var, src_hot[var].values)
    
    # If output file specified, write it
    if output_file is not None:
        new_hot.writer(output_file)
    
    return new_hot
