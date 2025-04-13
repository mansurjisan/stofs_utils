"""
Vertical grid handling functionality for STOFS3D
Based on schism_vgrid class from schism_file.py
"""
import numpy as np
from numpy import zeros, ones, array, nonzero, nan, isnan, sinh, tanh, pi, mod

class SchismVGrid:
    """
    SCHISM vertical grid implementation
    
    Handles both traditional sigma layers (ivcor=1) and SZ hybrid coordinates (ivcor=2)
    """
    
    def __init__(self):
        """Initialize empty vertical grid"""
        self.ivcor = None
        self.nvrt = None
        self.sigma = None
        self.kbp = None
        
    def read_vgrid(self, fname):
        """
        Read SCHISM vertical grid file
        
        Parameters
        ----------
        fname : str
            Path to vgrid.in file
            
        Returns
        -------
        np.ndarray
            Sigma coordinates
        """
        with open(fname, 'r') as fid:
            lines = fid.readlines()
        
        self.ivcor = int(lines[0].strip().split()[0])
        self.nvrt = int(lines[1].strip().split()[0])
        
        if self.ivcor == 1:
            # Read vgrid info (sigma coordinates)
            lines = lines[2:]
            sline = array(lines[0].split()).astype('float')
            
            if sline.min() < 0:  # Old format
                self.kbp = array([int(i.split()[1])-1 for i in lines])
                self.np = len(self.kbp)
                self.sigma = -ones([self.np, self.nvrt])
                
                for i, line in enumerate(lines):
                    self.sigma[i, self.kbp[i]:] = array(line.strip().split()[2:]).astype('float')
            else:  # New format
                sline = sline.astype('int')
                self.kbp = sline - 1
                self.np = len(sline)
                self.sigma = array([i.split()[1:] for i in lines[1:]]).T.astype('float')
                fpm = self.sigma < -1
                self.sigma[fpm] = -1
                
        elif self.ivcor == 2:
            # Read SZ hybrid grid
            self.kz, self.h_s = lines[1].strip().split()[1:3]
            self.kz = int(self.kz)
            self.h_s = float(self.h_s)
            
            # Read z grid
            self.ztot = []
            irec = 2
            for i in range(self.kz):
                irec = irec + 1
                self.ztot.append(lines[irec].strip().split()[1])
            self.ztot = array(self.ztot).astype('float')
            
            # Read s grid
            self.sigma = []
            irec = irec + 2
            self.nsig = self.nvrt - self.kz + 1
            self.h_c, self.theta_b, self.theta_f = array(lines[irec].strip().split()[:3]).astype('float')
            
            for i in range(self.nsig):
                irec = irec + 1
                self.sigma.append(lines[irec].strip().split()[1])
            self.sigma = array(self.sigma).astype('float')
            
        return self.sigma
    
    def compute_zcor(self, dp, eta=0, fmt=0, method=0, sigma=None, kbp=None, ifix=0):
        """
        Compute z-coordinates
        
        Parameters
        ----------
        dp : float or np.ndarray
            Depth at nodes
        eta : float or np.ndarray, optional
            Surface elevation, default=0
        fmt : int, optional
            Output format of zcor
            - 0: Bottom depths beyond kbp are extended (default)
            - 1: Bottom depths beyond kbp are nan
        method : int, optional
            - 0: Use object's sigma/kbp (default)
            - 1: Use provided sigma/kbp (for computing subset of nodes)
        sigma : np.ndarray, optional
            Sigma coordinates to use if method=1
        kbp : np.ndarray, optional
            Bottom layer indices to use if method=1
        ifix : int, optional
            - 0: Raise error if elevation problems with ivcor=2 (default)
            - 1: Use traditional sigma in shallow areas if problems
            
        Returns
        -------
        np.ndarray or tuple
            Z-coordinates or (Z-coordinates, kbp) if method=1 and ivcor=2
        """
        if self.ivcor == 1:
            if method == 0:
                return self._compute_zcor_sigma(self.sigma, dp, eta, fmt, self.kbp)
            if method == 1:
                return self._compute_zcor_sigma(sigma, dp, eta, fmt, kbp)
                
        elif self.ivcor == 2:
            zcor, kbp_out = self._compute_zcor_sz(dp, eta, fmt, ifix)
            if method == 0:
                return zcor
            if method == 1:
                return [zcor, kbp_out]
    
    def _compute_zcor_sigma(self, sigma, dp, eta=0, fmt=0, kbp=None):
        """Helper method to compute z-coordinates for sigma grid (ivcor=1)"""
        np_size = sigma.shape[0]
        
        if not hasattr(dp, '__len__'):
            dp = ones(np_size) * dp
        if not hasattr(eta, '__len__'):
            eta = ones(np_size) * eta
        
        # Get kbp if not provided
        if kbp is None:
            kbp = array([nonzero(abs(i + 1) < 1e-10)[0][-1] for i in sigma])
        
        # Thickness of water column
        hw = dp + eta
        
        # Add elevation
        zcor = hw[:, None] * sigma + eta[:, None]
        fpz = hw < 0
        zcor[fpz] = -dp[fpz][:, None]
        
        # Change format
        if fmt == 1:
            for i in range(np_size):
                zcor[i, :kbp[i]] = nan
                
        return zcor
    
    def _compute_zcor_sz(self, dp, eta=0, fmt=0, ifix=0):
        """Helper method to compute z-coordinates for SZ grid (ivcor=2)"""
        # Get dimension of pts
        if not hasattr(dp, '__len__'):
            np_size = 1
            dp = array([dp])
        else:
            np_size = len(dp)
            
        if not hasattr(eta, '__len__'):
            eta = ones(np_size) * eta
            
        zcor = ones([self.nvrt, np_size]) * nan
        
        cs = (1 - self.theta_b) * sinh(self.theta_f * self.sigma) / sinh(self.theta_f) + \
             self.theta_b * (tanh(self.theta_f * (self.sigma + 0.5)) - tanh(self.theta_f * 0.5)) / 2 / tanh(self.theta_f * 0.5)
        
        # For sigma layer: depth <= h_c
        hmod = dp.copy()
        fp = hmod > self.h_s
        hmod[fp] = self.h_s
        fps = hmod <= self.h_c
        zcor[(self.kz - 1):, fps] = self.sigma[:, None] * (hmod[fps][None, :] + eta[fps][None, :]) + eta[fps][None, :]
        
        # depth > h_c
        fpc = eta <= (-self.h_c - (hmod - self.h_c) * self.theta_f / sinh(self.theta_f))
        
        if np.sum(fpc) > 0:
            if ifix == 0:
                raise ValueError(f'Please choose a larger h_c: {self.h_c}')
            if ifix == 1:
                zcor[(self.kz - 1):, ~fps] = eta[~fps][None, :] + (eta[~fps][None, :] + hmod[~fps][None, :]) * self.sigma[:, None]
        else:
            zcor[(self.kz - 1):, ~fps] = eta[~fps][None, :] * (1 + self.sigma[:, None]) + self.h_c * self.sigma[:, None] + cs[:, None] * (hmod[~fps] - self.h_c)
        
        # For z layer
        kbp = -ones(np_size).astype('int')
        kbp[dp <= self.h_s] = self.kz - 1
        fpz = dp > self.h_s
        sind = nonzero(fpz)[0]
        
        for i in sind:
            for k in range(0, self.kz - 1):
                if (-dp[i] >= self.ztot[k]) * (-dp[i] <= self.ztot[k + 1]):
                    kbp[i] = k
                    break
                    
            # Check
            if kbp[i] == -1:
                raise ValueError('Cannot find a bottom level for node')
            elif kbp[i] < 0 or kbp[i] >= (self.kz - 1):
                raise ValueError(f'Impossible kbp,kz: {kbp[i]}, {self.kz}')
            
            # Assign values
            zcor[kbp[i], i] = -dp[i]
            for k in range(kbp[i] + 1, self.kz - 1):
                zcor[k, i] = self.ztot[k]
                
        zcor = zcor.T
        self.kbp = kbp
        
        # Change format
        if fmt == 0:
            for i in range(np_size):
                zcor[i, :kbp[i]] = zcor[i, kbp[i]]
                
        return zcor, kbp
    
    def write_vgrid(self, fname='vgrid.in', fmt=0):
        """
        Write SCHISM vertical grid
        
        Parameters
        ----------
        fname : str, optional
            Output filename, default='vgrid.in'
        fmt : int, optional
            Format for ivcor=1 grid
            - 0: Latest format (one line per level)
            - 1: Old format (one line per node)
        """
        if self.ivcor == 1:
            nvrt, np_size, kbp, sigma = self.nvrt, self.np, self.kbp.copy(), self.sigma.copy()
            
            with open(fname, 'w+') as fid:
                fid.write('1    !average # of layers={}\n{}  \n'.format(np.mean(nvrt - kbp), nvrt))
                
                if fmt == 0:
                    for i in range(np_size):
                        sigma[i, :kbp[i]] = -9
                    
                    fstr = '    ' + ' {:10d}' * np_size + '\n'
                    kbp = kbp + 1
                    fid.write(fstr.format(*kbp))
                    
                    fstr = '{:8d}' + ' {:10.6f}' * np_size + '\n'
                    sigma = sigma.T
                    
                    for i, k in enumerate(sigma):
                        fid.write(fstr.format(i + 1, *k))
                        
                elif fmt == 1:
                    for i, (k, sig) in enumerate(zip(kbp, sigma)):
                        fstr = '{:9d} {:3d}' + ' {:11.6f}' * (nvrt - k) + '\n'
                        fid.write(fstr.format(i + 1, k + 1, *sig[k:]))
                        
        elif self.ivcor == 2:
            with open(fname, 'w+') as fid:
                fid.write('2  !ivcor\n')
                fid.write('{} {} {} !nvrt, kz, h_s \nZ levels\n'.format(self.nvrt, self.kz, self.h_s))
                
                for k, zlevel in enumerate(self.ztot):
                    fid.write('{} {}\n'.format(k + 1, zlevel))
                    
                fid.write('S levels\n{} {} {} !h_c, theta_b, theta_f\n'.format(self.h_c, self.theta_b, self.theta_f))
                
                for k, slevel in enumerate(self.sigma):
                    fid.write('{} {:9.6f}\n'.format(k + 1, slevel))
                    
        else:
            raise ValueError(f'Unknown ivcor={self.ivcor}')

def create_schism_vgrid(fname='vgrid.in', ivcor=2, nvrt=10, zlevels=-1.e6, h_c=10, theta_b=0.5, theta_f=1.0):
    """
    Create SCHISM vertical grid
    
    Parameters
    ----------
    fname : str, optional
        Output filename, default='vgrid.in'
    ivcor : int, optional
        Vertical coordinate type
        - 1: LCS^2 grid
        - 2: SZ hybrid grid (default)
    nvrt : int, optional
        Number of vertical layers, default=10
    zlevels : float or np.ndarray, optional
        For ivcor=2: Z levels or single number for h_s, default=-1.e6
    h_c : float, optional
        For ivcor=2: Thickness of sigma layer, default=10
    theta_b : float, optional
        For ivcor=2: Bottom theta parameter, default=0.5
    theta_f : float, optional
        For ivcor=2: Surface theta parameter, default=1.0
        
    Returns
    -------
    SchismVGrid
        Created vertical grid
    """
    vd = SchismVGrid()
    vd.ivcor, vd.nvrt = ivcor, nvrt
    
    if ivcor == 2:
        if hasattr(zlevels, '__len__'):
            vd.kz, vd.ztot, vd.h_s = len(zlevels), zlevels, -zlevels[-1]
        else:
            vd.kz, vd.ztot, vd.h_s = 1, [zlevels], -zlevels
            
        vd.h_c, vd.theta_b, vd.theta_f = h_c, theta_b, theta_f
        vd.sigma = np.linspace(-1, 0, nvrt + 1 - vd.kz)
        vd.write_vgrid(fname)
        
    else:
        raise ValueError('ivcor=1 option not available yet')
        
    return vd

def read_schism_vgrid(fname):
    """
    Read SCHISM vertical grid
    
    Parameters
    ----------
    fname : str
        Path to vgrid.in file
        
    Returns
    -------
    SchismVGrid
        Loaded vertical grid
    """
    vd = SchismVGrid()
    vd.read_vgrid(fname)
    return vd

def compute_zcor(sigma, dp, eta=0, fmt=0, kbp=None, ivcor=1, vd=None, method=0, ifix=0):
    """
    Compute z-coordinates (standalone function version)
    
    Parameters
    ----------
    sigma : np.ndarray
        Sigma coordinates
    dp : float or np.ndarray
        Depth at nodes
    eta : float or np.ndarray, optional
        Surface elevation, default=0
    fmt : int, optional
        Output format, default=0
    kbp : np.ndarray, optional
        Bottom layer indices
    ivcor : int, optional
        Vertical coordinate type, default=1
    vd : SchismVGrid, optional
        Vertical grid object (needed for ivcor=2)
    method : int, optional
        Method to use, default=0
    ifix : int, optional
        Fix option for ivcor=2, default=0
        
    Returns
    -------
    np.ndarray or tuple
        Z-coordinates or (Z-coordinates, kbp)
    """
    if ivcor == 1:
        np_size = sigma.shape[0]
        
        if not hasattr(dp, '__len__'):
            dp = ones(np_size) * dp
        if not hasattr(eta, '__len__'):
            eta = ones(np_size) * eta
        
        # Get kbp
        if kbp is None:
            kbp = array([nonzero(abs(i + 1) < 1e-10)[0][-1] for i in sigma])
        
        # Thickness of water column
        hw = dp + eta
        
        # Add elevation
        zcor = hw[:, None] * sigma + eta[:, None]
        fpz = hw < 0
        zcor[fpz] = -dp[fpz][:, None]
        
        # Change format
        if fmt == 1:
            for i in range(np_size):
                zcor[i, :kbp[i]] = nan
                
        return zcor
        
    elif ivcor == 2:
        if vd is None:
            raise ValueError("vd (SchismVGrid) must be provided for ivcor=2")
            
        zcor, kbp_out = vd._compute_zcor_sz(dp, eta, fmt, ifix)
        
        if method == 0:
            return zcor
        if method == 1:
            return [zcor, kbp_out]
