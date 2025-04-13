"""
Grid handling functionality for STOFS3D
Based on schism_grid class from schism_file.py
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib as mpl
from .coordinate_utils import inside_polygon, signa
import os

class SchismGrid:
    """SCHISM horizontal grid"""
    
    def __init__(self, fname=None):
        """
        Initialize SCHISM grid instance
        
        Parameters
        ----------
        fname : str, optional
            Path to grid file. If provided, will read the grid
        """
        self.source_file = fname
        if fname is None:
            pass
        elif fname.endswith('gr3') or fname.endswith('.ll'):
            self.read_hgrid(fname)
        elif fname.endswith('.pkl') or fname.endswith('.npz'):
            # Load from pickled or numpy file
            from ..io.grid_io import load_grid
            temp_grid = load_grid(fname)
            self.__dict__ = temp_grid.hgrid.__dict__.copy()
        else:
            raise Exception(f'Grid file format {fname} not recognized')
    
    def read_hgrid(self, fname):
        """
        Read SCHISM horizontal grid file
        
        Parameters
        ----------
        fname : str
            Path to grid file (typically .gr3)
        """
        self.source_file = fname
        
        with open(fname, 'r') as fid:
            lines = fid.readlines()
        
        # Read grid size info
        self.ne, self.np = np.array(lines[1].split()[0:2]).astype('int')
        
        # Read node coordinates and depths
        self.x, self.y, self.dp = np.array([i.split()[1:4] for i in lines[2:(2+self.np)]]).astype('float').T
        
        if len(lines) < (2+self.np+self.ne):
            return
        
        # Read element connectivity
        fdata = [i.strip().split() for i in lines[(2+self.np):(2+self.np+self.ne)]]
        fdata = np.array([i if len(i)==6 else [*i,'-1'] for i in fdata]).astype('int')
        self.i34 = fdata[:, 1]
        self.elnode = fdata[:, 2:] - 1
        
        # Compute side info
        self.compute_side()
        
        if len(lines) < (4+self.np+self.ne):
            return
        
        # Read boundary info
        self._read_boundary_info(lines)
    
    def _read_boundary_info(self, lines):
        """Read boundary information from grid file"""
        # Read open boundary info
        n = 2 + self.np + self.ne
        self.nob = int(lines[n].strip().split()[0])
        n = n + 2
        self.nobn = []
        self.iobn = []
        
        for i in range(self.nob):
            self.nobn.append(int(lines[n].strip().split()[0]))
            self.iobn.append(np.array([int(lines[n+1+k].strip().split()[0])-1 for k in range(self.nobn[-1])]))
            n = n + 1 + self.nobn[-1]
        
        self.nobn = np.array(self.nobn)
        self.iobn = np.array(self.iobn, dtype='O')
        if len(self.iobn) == 1:
            self.iobn = self.iobn.astype('int')
        
        # Read land boundary info
        self.nlb = int(lines[n].strip().split()[0])
        n = n + 2
        self.nlbn = []
        self.ilbn = []
        self.island = []
        
        for i in range(self.nlb):
            sline = lines[n].split('=')[0].split()
            self.nlbn.append(int(sline[0]))
            ibtype = 0
            self.ilbn.append(np.array([int(lines[n+1+k].strip().split()[0])-1 for k in range(self.nlbn[-1])]))
            n = n + 1 + self.nlbn[-1]
            
            # Add boundary type info
            if len(sline) == 2:
                ibtype = int(sline[1])
            if self.ilbn[-1][0] == self.ilbn[-1][-1]:
                ibtype = 1
            self.island.append(ibtype)
        
        self.island = np.array(self.island)
        self.nlbn = np.array(self.nlbn)
        self.ilbn = np.array(self.ilbn, dtype='O')
        if len(self.ilbn) == 1:
            self.ilbn = self.ilbn.astype('int')
    
    def compute_side(self, fmt=0):
        """
        Compute side information of SCHISM grid
        
        Parameters
        ----------
        fmt : int
            0: compute ns (# of sides) only
            1: compute (ns, isidenode, isdel)
            2: compute (ns, isidenode, isdel), and (xcj, ycj, dps, distj)
            
        Returns
        -------
        int or tuple
            ns or (ns, isidenode, isdel) depending on fmt
        """
        # Collect sides
        fp3 = self.i34 == 3
        self.elnode[fp3, -1] = self.elnode[fp3, 0]
        sis = []
        sie = []
        
        for i in range(4):
            sis.append(np.c_[self.elnode[:, (i+1)%4], self.elnode[:, (i+2)%4]])
            sie.append(np.arange(self.ne))
        
        sie = np.array(sie).T.ravel()
        sis = np.array(sis).transpose([1, 0, 2]).reshape([len(sie), 2])
        fpn = np.diff(sis, axis=1)[:, 0] != 0
        sis = sis[fpn]
        sie = sie[fpn]
        self.elnode[fp3, -1] = -2
        
        # Sort sides
        usis = np.sort(sis, axis=1).T
        usis, sind, sindr = np.unique(usis[0] + 1j * usis[1], return_index=True, return_inverse=True)
        self.ns = len(sind)
        
        if fmt == 0:
            return self.ns
        elif fmt in [1, 2]:
            # Build isidenode
            sinda = np.argsort(sind)
            sinds = sind[sinda]
            self.isidenode = sis[sinds]
            
            # Build isdel
            se1 = sie[sinds]
            se2 = -np.ones(self.ns).astype('int')
            sindl = np.setdiff1d(np.arange(len(sie)), sind)
            se2[sindr[sindl]] = sie[sindl]
            se2 = se2[sinda]
            self.isdel = np.c_[se1, se2]
            fps = (se1 > se2) * (se2 != -1)
            self.isdel[fps] = np.fliplr(self.isdel[fps])
            
            # Compute xcj, ycj and dps
            if fmt == 2:
                self.xcj, self.ycj, self.dps = np.c_[self.x, self.y, self.dp][self.isidenode].mean(axis=1).T
                self.distj = np.abs(np.diff(self.x[self.isidenode], axis=1) + 1j * np.diff(self.y[self.isidenode], axis=1))[:, 0]
            
            return self.ns, self.isidenode, self.isdel
        
    def compute_ctr(self):
        """
        Compute element center information: (xctr, yctr, dpe)
        
        Returns
        -------
        np.ndarray
            Element depth values (dpe)
        """
        if not hasattr(self, 'xctr'):
            fp3 = self.i34 == 3
            fp4 = ~fp3
            self.xctr, self.yctr, self.dpe = np.zeros([3, self.ne])
            
            self.xctr[fp3] = self.x[self.elnode[fp3, :3]].mean(axis=1)
            self.xctr[fp4] = self.x[self.elnode[fp4, :]].mean(axis=1)
            
            self.yctr[fp3] = self.y[self.elnode[fp3, :3]].mean(axis=1)
            self.yctr[fp4] = self.y[self.elnode[fp4, :]].mean(axis=1)
            
            self.dpe[fp3] = self.dp[self.elnode[fp3, :3]].mean(axis=1)
            self.dpe[fp4] = self.dp[self.elnode[fp4, :]].mean(axis=1)
        
        return self.dpe
    
    def compute_area(self):
        """
        Compute element areas
        
        Returns
        -------
        np.ndarray
            Element areas
        """
        fp = self.elnode[:, -1] < 0
        x1 = self.x[self.elnode[:, 0]]
        y1 = self.y[self.elnode[:, 0]]
        x2 = self.x[self.elnode[:, 1]]
        y2 = self.y[self.elnode[:, 1]]
        x3 = self.x[self.elnode[:, 2]]
        y3 = self.y[self.elnode[:, 2]]
        x4 = self.x[self.elnode[:, 3]]
        y4 = self.y[self.elnode[:, 3]]
        x4[fp] = x1[fp]
        y4[fp] = y1[fp]
        
        self.area = ((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)+(x3-x1)*(y4-y1)-(x4-x1)*(y3-y1))/2
        return self.area
    
    def compute_nne(self):
        """
        Compute nodal ball information: nne, mnei, indel, ine
        
        Returns
        -------
        np.ndarray
            nne (number of elements in nodal ball)
        """
        # Get index of all node and elements
        elem = np.tile(np.arange(self.ne), [4, 1]).T.ravel()
        node = self.elnode.ravel()
        fpn = node != -2
        elem, node = elem[fpn], node[fpn]
        fpn = np.argsort(node)
        elem, node = elem[fpn], node[fpn]
        
        # Compute nne, ine, indel
        unode, sind, self.nne = np.unique(node, return_index=True, return_counts=True)
        self.mnei = self.nne.max()
        self.ine = -np.ones([self.np, self.mnei]).astype('int')
        
        for i in range(self.mnei):
            fpe = self.nne > i
            sinde = sind[fpe] + i
            self.ine[fpe, i] = elem[sinde]
        
        self.indel = np.array([np.array(i[:k]) for i, k in zip(self.ine, self.nne)], dtype='O')
        return self.nne
    
    def compute_acor(self, pxy, fmt=0):
        """
        Compute areal coordinates for points pxy
        
        Parameters
        ----------
        pxy : np.ndarray
            Points [npt, 2] for which to compute coordinates
        fmt : int
            0: faster method using neighbors
            1: slower method using point-wise comparison
            
        Returns
        -------
        tuple
            (ie, ip, acor) - element indices, node indices, and area coordinates
        """
        npt = len(pxy)
        pip = -np.ones([npt, 3]).astype('int')
        pacor = np.zeros([npt, 3])
        
        if fmt == 0:
            from ..utils.helpers import near_pts
            
            pie = -np.ones(npt).astype('int')
            sindp = np.arange(npt)
            
            # Search element centers
            if not hasattr(self, 'xctr'):
                self.compute_ctr()
            
            sinde = near_pts(pxy[sindp], np.c_[self.xctr, self.yctr])
            fps, sip, sacor = self.inside_elem(pxy[sindp], sinde)
            
            if len(fps) != 0:
                pie[sindp[fps]] = sinde[fps]
                pip[sindp[fps]] = sip
                pacor[sindp[fps]] = sacor
            
            # Search direct neighbors
            fp = pie[sindp] == -1
            sindp, sinde = sindp[fp], sinde[fp]
            
            if len(sindp) != 0:
                if not hasattr(self, 'ic3'):
                    self.compute_ic3()
                
                for i in range(self.i34.max()):
                    ie = self.ic3[sinde, i]
                    fps, sip, sacor = self.inside_elem(pxy[sindp], ie)
                    
                    if len(fps) != 0:
                        pie[sindp[fps]] = ie[fps]
                        pip[sindp[fps]] = sip
                        pacor[sindp[fps]] = sacor
                    
                    # Update sindp
                    fp = pie[sindp] == -1
                    sindp, sinde = sindp[fp], sinde[fp]
                    if len(sindp) == 0:
                        break
            
            # Search elements inside node ball
            if len(sindp) != 0:
                if not hasattr(self, 'ine'):
                    self.compute_nne()
                
                sindn = near_pts(pxy[sindp], np.c_[self.x, self.y])
                pip[sindp] = sindn[:, None]
                pacor[sindp, 0] = 1
                
                for i in range(self.mnei):
                    ie = self.ine[sindn, i]
                    fps, sip, sacor = self.inside_elem(pxy[sindp], ie)
                    
                    if len(fps) != 0:
                        pie[sindp[fps]] = ie[fps]
                        pip[sindp[fps]] = sip
                        pacor[sindp[fps]] = sacor
                    
                    # Update sindp
                    if i < (self.mnei - 1):
                        fp = (pie[sindp] == -1) * (self.ine[sindn, i+1] != -1)
                        sindp, sindn = sindp[fp], sindn[fp]
                    
                    if len(sindp) == 0:
                        break
            
            # Use point-wise method for remaining points
            sindp = np.nonzero(pie == -1)[0]
            sindp = sindp[self.inside_grid(pxy[sindp]) == 1]
            
            if len(sindp) != 0:
                pie[sindp], pip[sindp], pacor[sindp] = self.compute_acor(pxy[sindp], fmt=1)
        
        elif fmt == 1:
            # Check 1st triangle
            sindn = self.elnode.T[:3]
            pie = inside_polygon(pxy, self.x[sindn], self.y[sindn], fmt=1)
            fps = pie != -1
            pip[fps] = sindn.T[pie[fps]]
            
            # Check 2nd triangle
            sind4 = np.nonzero(self.i34 == 4)[0]
            sind2 = np.nonzero(~fps)[0]
            
            if len(sind2) != 0 and len(sind4) != 0:
                sindn = self.elnode[sind4].T[np.array([0, 2, 3])]
                pie2 = inside_polygon(pxy[sind2], self.x[sindn], self.y[sindn], fmt=1)
                fps = pie2 != -1
                pie[sind2[fps]] = sind4[pie2[fps]]
                pip[sind2[fps]] = sindn.T[pie2[fps]]
            
            # Compute acor
            fpn = pie != -1
            if np.sum(fpn) != 0:
                x1, x2, x3 = self.x[pip[fpn]].T
                y1, y2, y3 = self.y[pip[fpn]].T
                x, y = pxy[fpn].T
                
                A1 = signa(np.c_[x, x2, x3], np.c_[y, y2, y3])
                A2 = signa(np.c_[x1, x, x3], np.c_[y1, y, y3])
                A = signa(np.c_[x1, x2, x3], np.c_[y1, y2, y3])
                
                pacor[fpn] = np.c_[A1/A, A2/A, 1-(A1+A2)/A]
            
            if np.sum(~fpn) != 0:
                from ..utils.helpers import near_pts
                sindn = near_pts(pxy[~fpn], np.c_[self.x, self.y])
                pip[~fpn] = sindn[:, None]
                pacor[~fpn, 0] = 1
        
        return pie, pip, pacor
    
    def inside_elem(self, pxy, ie):
        """
        Check whether points are inside elements, then compute area coordinates
        
        Parameters
        ----------
        pxy : np.ndarray
            Points [npt, 2]
        ie : np.ndarray
            Element indices corresponding to each point
            
        Returns
        -------
        tuple
            (sind, pip, pacor) - indices, nodes, and area coordinates
        """
        sind = []
        pip = []
        pacor = []
        fps = None
        
        for i in range(self.i34.max() - 2):
            # Get points and element info
            if i == 0:
                ip = self.elnode[ie, :3]
                x1, x2, x3 = self.x[ip].T
                y1, y2, y3 = self.y[ip].T
                xi, yi = pxy.T
            
            if i == 1:
                fpr = (~fps) * (self.i34[ie] == 4)
                sindr = np.nonzero(fpr)[0]
                ip = self.elnode[ie[fpr]][:, np.array([0, 2, 3])]
                x1, x2, x3 = self.x[ip].T
                y1, y2, y3 = self.y[ip].T
                xi, yi = pxy[fpr].T
            
            # Compute area coordinates
            A0 = signa(np.c_[x1, x2, x3], np.c_[y1, y2, y3])
            A1 = signa(np.c_[xi, x2, x3], np.c_[yi, y2, y3])
            A2 = signa(np.c_[x1, xi, x3], np.c_[y1, yi, y3])
            A3 = signa(np.c_[x1, x2, xi], np.c_[y1, y2, yi])
            
            fps = (A1 >= 0) * (A2 >= 0) * (A3 >= 0)
            ac1 = A1[fps] / A0[fps]
            ac2 = A2[fps] / A0[fps]
            
            if not isinstance(fps, np.ndarray):
                fps = np.array([fps])
            
            # Get index of points
            if i == 0:
                sind.extend(np.nonzero(fps)[0])
            if i == 1:
                sind.extend(sindr[fps])
            
            pip.extend(ip[fps])
            pacor.extend(np.c_[ac1, ac2, 1-ac1-ac2])
        
        return np.array(sind), np.array(pip), np.array(pacor)
    
    def inside_grid(self, pxy):
        """
        Check whether points are inside grid
        
        Parameters
        ----------
        pxy : np.ndarray
            Points [npt, 2]
            
        Returns
        -------
        np.ndarray
            Mask array (0: outside, 1: inside)
        """
        npt = len(pxy)
        sindp = np.arange(npt)
        
        if not hasattr(self, 'bndinfo'):
            self.compute_bnd()
        
        if not hasattr(self.bndinfo, 'nb'):
            self.compute_bnd()
        
        for i in range(self.bndinfo.nb):
            fpb = self.bndinfo.ibn[i]
            fp = inside_polygon(pxy[sindp], self.x[fpb], self.y[fpb]) == 1
            
            if self.bndinfo.island[i] == 0:
                sindp = sindp[fp]
            else:
                sindp = sindp[~fp]
            
            if len(sindp) == 0:
                break
        
        sind = np.zeros(npt).astype('int')
        sind[sindp] = 1
        return sind
    
    def compute_ic3(self):
        """
        Compute element-to-side table
        
        Returns
        -------
        tuple
            (ic3, elside) - element-to-element and element-to-side tables
        """
        # Get index for all elements and sides
        if not hasattr(self, 'isdel'):
            self.compute_side(fmt=1)
        
        side = np.tile(np.arange(self.ns), [2, 1]).T.ravel()
        elem = self.isdel.ravel()
        fpn = elem != -1
        side, elem = side[fpn], elem[fpn]
        fpn = np.argsort(elem)
        side, elem = side[fpn], elem[fpn]
        
        # Build elside
        uelem, sind = np.unique(elem, return_index=True)
        self.elside = -np.ones([self.ne, 4]).astype('int')
        m34 = self.i34.max()
        
        for i in range(m34):
            fps = np.nonzero(self.i34 > i)[0]
            i34 = self.i34[fps]
            sinds = sind[fps] + i
            sd = side[sinds]
            n1, n2 = self.isidenode[sd].T
            
            for k in range(m34):  # Sort order of sides
                id1, id2 = (k+1) % i34, (k+2) % i34
                fpk = ((self.elnode[fps, id1] == n1) * (self.elnode[fps, id2] == n2)) | ((self.elnode[fps, id1] == n2) * (self.elnode[fps, id2] == n1))
                self.elside[fps[fpk], k] = sd[fpk]
        
        self.elside[self.i34 == 3, -1] = -1
        
        # Build ic3
        self.ic3 = -np.ones([self.ne, 4]).astype('int')
        ie = np.arange(self.ne)
        
        for i in range(m34):
            es = self.isdel[self.elside[:, i]]
            fp = es[:, 0] == ie
            self.ic3[fp, i] = es[fp, 1]
            self.ic3[~fp, i] = es[~fp, 0]
        
        self.ic3[self.elside == -1] = -1
        return self.ic3, self.elside
    
    def compute_bnd(self):
        """Compute boundary information"""
        print('Computing grid boundaries')
        
        if not hasattr(self, 'isdel') or not hasattr(self, 'isidenode'):
            self.compute_side(fmt=1)
        
        # Find boundary side and element
        fpn = self.isdel[:, -1] == -1
        isn = self.isidenode[fpn]
        be = self.isdel[fpn][:, 0]
        nbs = len(be)
        
        # Sort isn
        i2 = np.ones(nbs).astype('int')
        fp3 = np.nonzero(self.i34[be] == 3)[0]
        fp4 = np.nonzero(self.i34[be] == 4)[0]
        
        for i in range(4):
            if i == 3:
                i1 = self.elnode[be[fp4], 3]
                i2 = self.elnode[be[fp4], 0]
                fp = (isn[fp4, 0] == i2) * (isn[fp4, 1] == i1)
                isn[fp4[fp]] = np.fliplr(isn[fp4[fp]])
            else:
                i1 = self.elnode[be, i]
                i2[fp3] = self.elnode[be[fp3], (i+1) % 3]
                i2[fp4] = self.elnode[be[fp4], i+1]
                fp = (isn[:, 0] == i2) * (isn[:, 1] == i1)
                isn[fp] = np.fliplr(isn[fp])
        
        # Compute all boundaries
        sinds = dict(zip(isn[:, 0], np.arange(nbs)))  # Dict for sides
        ifb = np.ones(nbs).astype('int')
        nb = 0
        nbn = []
        ibn = []
        
        while np.sum(ifb) != 0:
            # Start points
            id0 = isn[np.nonzero(ifb == 1)[0][0], 0]
            id = isn[sinds[id0], 1]
            ibni = [id0, id]
            ifb[sinds[id0]] = 0
            ifb[sinds[id]] = 0
            
            while True:
                id = isn[sinds[id], 1]
                ifb[sinds[id]] = 0
                if id == id0:
                    break
                ibni.append(id)
            
            nb = nb + 1
            nbn.append(len(ibni))
            ibn.append(np.array(ibni))
        
        # Sort bnd
        nbn = np.array(nbn)
        ibn = np.array(ibn, dtype='O')
        fps = np.flipud(np.argsort(nbn))
        nbn, ibn = nbn[fps], ibn[fps]
        
        # Save boundary information
        if not hasattr(self, 'bndinfo'):
            self.bndinfo = type('zdata', (), {})()
        
        ip = []
        sind = []
        S = self.bndinfo
        
        for m, ibni in enumerate(ibn):
            ip.extend(ibni)
            sind.extend(np.tile(m, len(ibni)))
        
        ip = np.array(ip)
        sind = np.array(sind)
        S.sind = sind
        S.ip = ip
        S.island = np.ones(nb).astype('int')
        S.nb = nb
        S.nbn = nbn
        S.ibn = ibn
        S.x = self.x[ip]
        S.y = self.y[ip]
        
        # Find the outline
        for i in range(nb):
            px = self.x[S.ibn[i]]
            i0 = np.nonzero(px == px.min())[0][0]
            sid = S.ibn[i][np.array([(i0-1) % S.nbn[i], i0, (i0+1) % S.nbn[i]])]
            
            if signa(self.x[sid], self.y[sid]) > 0:
                S.island[i] = 0
                break
        
        # Add to grid bnd info
        if not hasattr(self, 'nob'):
            self.nob = 0
            self.nobn = np.array([])
            self.iobn = np.array([[]])
            sind = np.argsort(S.island)
            self.nlb = S.nb
            self.nlbn = S.nbn[sind]
            self.ilbn = S.ibn[sind]
            self.island = S.island[sind]

    def interp_node_to_elem(self, value=None):
        """
        Interpolate node values to element values
        
        Parameters
        ----------
        value : np.ndarray, optional
            Node values to interpolate. If None, uses self.dp
            
        Returns
        -------
        np.ndarray
            Element values
        """
        # Interpolate
        dp = self.dp if (value is None) else value
        fp3 = self.i34 == 3
        fp4 = ~fp3
        dpe = np.zeros(self.ne)
        dpe[fp3] = dp[self.elnode[fp3, :3]].mean(axis=1)
        dpe[fp4] = dp[self.elnode[fp4]].mean(axis=1)
        return dpe

    def interp_elem_to_node(self, value=None, fmt=0, p=1):
            """
            Interpolate element values to nodes
            
            Parameters
            ----------
            value : np.ndarray, optional
                Element values to interpolate. If None, uses self.dpe
            fmt : int
                0: simple average
                1: inverse distance (power=p)
                2: maximum of surrounding nodal values
                3: minimum of surrounding nodal values
            p : float
                Power parameter for inverse distance weighting (only used if fmt=1)
                
            Returns
            -------
            np.ndarray
                Node values
            """
            # Element values
            if not hasattr(self, 'nne'):
                self.compute_nne()
            if (value is None) and (not hasattr(self, 'dpe')):
                self.compute_ctr()
            v0 = self.dpe if (value is None) else value

            # Interpolation
            vs = v0[self.ine]
            if fmt == 0:
                w = self.ine != -1
                tw = w.sum(axis=1)
                if np.sum(np.isnan(value)) != 0:
                    vs[~w] = 0
                    v = vs.sum(axis=1)/tw
                else:
                    v = (w*vs).sum(axis=1)/tw
            if fmt == 2:
                vs[self.ine == -1] = v0.min()-1
                v = vs.max(axis=1)
            if fmt == 3:
                vs[self.ine == -1] = v0.max()+1
                v = vs.min(axis=1)
            if fmt == 1:
                dist = np.abs((self.xctr[self.ine]+1j*self.yctr[self.ine])-(self.x+1j*self.y)[:,None])
                w = 1/(dist**p)
                w[self.ine == -1] = 0
                tw = w.sum(axis=1)
                v = (w*vs).sum(axis=1)/tw
            return v

    def compute_gradient(self, fmt=0):
            """
            Compute gradient of grid depth on each element first,
            then transfer to nodes with selected interpolation method
            
            Parameters
            ----------
            fmt : int
                Format for node interpolation method (see interp_elem_to_node)
                
            Returns
            -------
            tuple
                (dpdx, dpdy, dpdxy) - x gradient, y gradient, magnitude
            """
            if not hasattr(self, 'area'):
                self.compute_area()
            if not hasattr(self, 'dpe'):
                self.compute_ctr()
                
            # Get pts
            fp = self.elnode[:, -1] < 0
            fpn = ~fp
            x1 = self.x[self.elnode[:, 0]]
            y1 = self.y[self.elnode[:, 0]]
            v1 = self.dp[self.elnode[:, 0]]
            x2 = self.x[self.elnode[:, 1]]
            y2 = self.y[self.elnode[:, 1]]
            v2 = self.dp[self.elnode[:, 1]]
            x3 = self.x[self.elnode[:, 2]]
            y3 = self.y[self.elnode[:, 2]]
            v3 = self.dp[self.elnode[:, 2]]
            x4 = self.x[self.elnode[:, 3]]
            y4 = self.y[self.elnode[:, 3]]
            v4 = self.dp[self.elnode[:, 3]]
            x4[fp] = x1[fp]
            y4[fp] = y1[fp]
            v4[fp] = v1[fp]
            a1 = ((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))/2
            a2 = ((x3-x1)*(y4-y1)-(x4-x1)*(y3-y1))/2

            # Compute gradients
            self.dpedx = (v1*(y2-y3)+v2*(y3-y1)+v3*(y1-y2))/(2*a1)
            self.dpedy = ((x3-x2)*v1+(x1-x3)*v2+(x2-x1)*v3)/(2*a1)
            self.dpedxy = np.sqrt(self.dpedx**2+self.dpedy**2)

            # Modify quads
            dpedx2 = (v1[fpn]*(y3[fpn]-y4[fpn])+v3[fpn]*(y4[fpn]-y1[fpn])+v4[fpn]*(y1[fpn]-y3[fpn]))/(2*a2[fpn])
            dpedy2 = ((x4[fpn]-x3[fpn])*v1[fpn]+(x1[fpn]-x4[fpn])*v3[fpn]+(x3[fpn]-x1[fpn])*v4[fpn])/(2*a2[fpn])
            dpedxy2 = np.sqrt(dpedx2**2+dpedy2**2)

            self.dpedx[fpn] = (self.dpedx[fpn]+dpedx2)/2
            self.dpedy[fpn] = (self.dpedy[fpn]+dpedy2)/2
            self.dpedxy[fpn] = (self.dpedxy[fpn]+dpedxy2)/2

            # Get node value
            self.dpdx = self.interp_elem_to_node(value=self.dpedx, fmt=fmt)
            self.dpdy = self.interp_elem_to_node(value=self.dpedy, fmt=fmt)
            self.dpdxy = self.interp_elem_to_node(value=self.dpedxy, fmt=fmt)

            return self.dpdx, self.dpdy, self.dpdxy

    def interp(self, pxy, value=None, fmt=0):
            """
            Interpolate to get value at points
            
            Parameters
            ----------
            pxy : np.ndarray
                Points [npt, 2] at which to interpolate
            value : np.ndarray, optional
                Node values to interpolate from. If None, uses self.dp
            fmt : int
                Format for compute_acor method
                
            Returns
            -------
            np.ndarray
                Interpolated values at points
            """
            # Get node value
            vi = self.dp if value is None else value
            if len(vi) == self.ne:
                vi = self.interp_elem_to_node(value=vi)

            # Interp
            pip, pacor = self.compute_acor(pxy, fmt=fmt)[1:]
            return (vi[pip]*pacor).sum(axis=1)

    def plot_grid(self, ax=None, method=0, fmt=0, value=None, mask=None, ec=None, fc=None,
                     lw=0.1, levels=None, ticks=None, xlim=None, ylim=None, clim=None, 
                     extend='both', cb=True, **args):
            """
            Plot grid with default color value (grid depth)
            
            Parameters
            ----------
            ax : matplotlib.axes.Axes, optional
                Axes to plot on
            method : int
                0: using tricontourf; 1: using PolyCollection
            fmt : int
                0: plot grid only; 1: plot filled contours; 2: plot contour lines
            value : np.ndarray, optional
                Color values (size np or ne)
            mask : np.ndarray, optional
                Element mask (size ne, True to include)
            ec : str, optional
                Grid line color
            fc : str, optional
                Element fill color
            lw : float
                Grid line width
            levels : int or array-like
                Number of contour levels or specific levels
            ticks : int or array-like
                Number of colorbar ticks or specific ticks
            xlim, ylim : tuple, optional
                Plot axis limits
            clim : tuple, optional
                Color value range [vmin, vmax]
            extend : str
                Colorbar extend mode
            cb : bool
                Whether to add colorbar
            **args : dict
                Additional arguments to pass to plotting functions
                
            Returns
            -------
            matplotlib.collections.Collection
                The plotted collection
            """
            if ec is None:
                ec = 'None'
            if fc is None:
                fc = 'None'
            if ax is None:
                ax = plt.gca()

            if method == 0:
                fp3 = self.i34 == 3
                fp4 = self.i34 == 4
                
                if (fmt == 0) or (ec != 'None'):  # Compute lines of grid
                    # Triangles
                    tri = self.elnode[fp3, :3]
                    tri = np.c_[tri, tri[:, 0]]
                    x3 = self.x[tri]
                    y3 = self.y[tri]
                    x3 = np.c_[x3, np.ones([np.sum(fp3), 1])*np.nan]
                    x3 = np.reshape(x3, x3.size)
                    y3 = np.c_[y3, np.ones([np.sum(fp3), 1])*np.nan]
                    y3 = np.reshape(y3, y3.size)
                    
                    # Quads
                    quad = self.elnode[fp4, :]
                    quad = np.c_[quad, quad[:, 0]]
                    x4 = self.x[quad]
                    y4 = self.y[quad]
                    x4 = np.c_[x4, np.ones([np.sum(fp4), 1])*np.nan]
                    x4 = np.reshape(x4, x4.size)
                    y4 = np.c_[y4, np.ones([np.sum(fp4), 1])*np.nan]
                    y4 = np.reshape(y4, y4.size)

                if fmt == 0:
                    if ec == 'None':
                        ec = 'k'
                    hg = ax.plot(np.r_[x3, x4], np.r_[y3, y4], lw=lw, color=ec)[0]
                elif fmt in [1, 2]:
                    tri = np.r_[self.elnode[(fp3|fp4), :3], 
                               np.c_[self.elnode[fp4, 0], self.elnode[fp4, 2:]]]
                    
                    # Determine value
                    if value is None:
                        value = self.dp
                    else:
                        if len(value) == self.ne:
                            value = self.interp_elem_to_node(value=value)
                        elif len(value) != self.np:
                            raise ValueError(f'value has wrong size: {value.shape}')

                    # Determine clim
                    if clim is None:
                        fpn = ~np.isnan(value)
                        vmin, vmax = np.min(value[fpn]), np.max(value[fpn])
                        if vmin == vmax:
                            vmax = vmax + 1e-6
                    else:
                        vmin, vmax = clim

                    # Determine levels
                    if levels is None:
                        levels = 51
                    if not hasattr(levels, '__len__'):
                        levels = np.linspace(vmin, vmax, int(levels))

                    # Set mask
                    if np.sum(np.isnan(value)) != 0:
                        tri = tri[~np.isnan(value[tri].sum(axis=1))]

                    if (vmax-vmin)/(abs(vmax)+abs(vmin)) < 1e-10:
                        if fmt == 1:
                            hg = ax.tricontourf(self.x, self.y, tri, value,
                                              vmin=vmin, vmax=vmax, extend=extend, **args)
                        if fmt == 2:
                            hg = ax.tricontour(self.x, self.y, tri, value,
                                             vmin=vmin, vmax=vmax, extend=extend, **args)
                    else:
                        if fmt == 1:
                            hg = ax.tricontourf(self.x, self.y, tri, value, levels=levels,
                                              vmin=vmin, vmax=vmax, extend=extend, **args)
                        if fmt == 2:
                            hg = ax.tricontour(self.x, self.y, tri, value, levels=levels,
                                             vmin=vmin, vmax=vmax, extend=extend, **args)

                    # Set colormap limits
                    plt.sci(hg)
                    plt.clim(vmin=vmin, vmax=vmax)
                    
                    # Add colorbar
                    if cb and fmt == 1:
                        hc = plt.colorbar(hg)
                        self.hc = hc
                        if ticks is not None:
                            if not hasattr(ticks, '__len__'):
                                hc.set_ticks(np.linspace(vmin, vmax, int(ticks)))
                            else:
                                hc.set_ticks(ticks)

                    # Plot grid
                    if ec != 'None':
                        hg_lines = ax.plot(np.r_[x3, x4], np.r_[y3, y4], lw=lw, color=ec)[0]

                self.hg = hg
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                return hg
                
            elif method == 1:
                # Create polygon
                xy4 = self.x[self.elnode], self.y[self.elnode]
                xy4 = np.array([s[0:-1, :] if (i34 == 3 and len(s) == 4) else s 
                              for s, i34 in zip(xy4, self.i34)])

                # Element value
                if value is None:
                    if not hasattr(self, 'dpe'):
                        self.compute_ctr()
                    value = self.dpe
                else:
                    if len(value) == self.np:
                        value = self.interp_node_to_elem(value=value)
                    elif len(value) != self.ne:
                        raise ValueError(f'value has wrong size: {value.shape}')

                # Apply mask
                if mask is not None:
                    xy4 = xy4[mask]
                    value = value[mask]

                # Get clim
                if clim is None:
                    clim = [np.min(value), np.max(value)]

                # Plot
                if fmt == 0:
                    hg = mpl.collections.PolyCollection(xy4, lw=lw, edgecolor=ec, 
                                                      facecolor=fc, antialiased=False, **args)
                else:
                    hg = mpl.collections.PolyCollection(xy4, lw=lw, edgecolor=ec, array=value,
                                                      clim=clim, antialiased=False, **args)

                    # Add colorbar
                    if cb:
                        hc = plt.colorbar(hg)
                        self.hc = hc
                        if ticks is not None:
                            hc.set_ticks(ticks)
                        hc.set_clim(clim)

                # Add to figure
                ax.add_collection(hg)
                ax.autoscale_view()
                self.hg = hg
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                return hg
        
    def plot(self, **args):
            """
            Alias for plot_grid()
            
            Parameters
            ----------
            **args : dict
                Arguments passed to plot_grid
                
            Returns
            -------
            matplotlib.collections.Collection
                The plotted collection
            """
            return self.plot_grid(**args)

    def plot_bnd(self, c='k', lw=0.5, ax=None, **args):
            """
            Plot SCHISM grid boundary
            
            Parameters
            ----------
            c : str or list
                Colors for boundaries. If len(c)==1, same color for all.
                If len(c)==2, c[0] for open boundaries, c[1] for land boundaries
            lw : float
                Line width
            ax : matplotlib.axes.Axes, optional
                Axes to plot on
            **args : dict
                Additional arguments to pass to plotting functions
                
            Returns
            -------
            list
                Handles to plotted lines [hb1, hb2]
            """
            if ax is not None:
                plt.sca(ax)
            if len(c) == 1:
                c = c*2
            if not hasattr(self, 'nob'):
                self.compute_bnd()

            # Get indices for boundaries
            sindo = []
            for i in range(self.nob):
                sindo = np.r_[sindo, -1, self.iobn[i]]
            sindo = np.array(sindo).astype('int')
            fpn = sindo == -1
            bx1 = self.x[sindo]
            by1 = self.y[sindo]
            bx1[fpn] = np.nan
            by1[fpn] = np.nan

            sindl = []
            for i in range(self.nlb):
                if self.island[i] == 0:
                    sindl = np.r_[sindl, -1, self.ilbn[i]]
                else:
                    sindl = np.r_[sindl, -1, self.ilbn[i], self.ilbn[i][0]]
            sindl = np.array(sindl).astype('int')
            fpn = sindl == -1
            bx2 = self.x[sindl]
            by2 = self.y[sindl]
            bx2[fpn] = np.nan
            by2[fpn] = np.nan

            hb1 = plt.plot(bx1, by1, c[0], lw=lw, **args)[0]
            hb2 = plt.plot(bx2, by2, c[-1], lw=lw, **args)[0]
            self.hb = [hb1, hb2]
            return self.hb

    def write_hgrid(self, fname, value=None, fmt=0, elnode=1, bndfile=None, Info=None):
            """
            Write *.gr3 file
            
            Parameters
            ----------
            fname : str
                File name to write
            value : np.ndarray or float, optional
                Depth values to output. If None, uses self.dp
            fmt : int
                0: don't output grid boundary info; 1: output grid boundary info
            elnode : int
                1: output grid connectivity; 0: don't output grid connectivity
            bndfile : str, optional
                If provided, appends this file's content to the output
            Info : str, optional
                Annotation for the gr3 file
            """
            # Get depth value
            if value is None:
                dp = self.dp
            else:
                if hasattr(value, "__len__"):
                    dp = value
                else:
                    dp = np.ones(self.np) * value
            if fmt == 1:
                elnode = 1

            # Write *gr3
            with open(fname, 'w+') as fid:
                fid.write('!grd info:{}\n'.format(Info if Info else ""))
                fid.write('{} {}\n'.format(self.ne, self.np))
                for i in range(self.np):
                    fid.write('{:<d} {:<.8f} {:<.8f} {:<.8f}\n'.format(i+1, self.x[i], self.y[i], dp[i]))
                if elnode != 0:
                    for i in range(self.ne):
                        if self.i34[i] == 3:
                            fid.write('{:<d} {:d} {:d} {:d} {:d}\n'.format(i+1, self.i34[i], *self.elnode[i, :3]+1))
                        if self.i34[i] == 4:
                            fid.write('{:<d} {:d} {:d} {:d} {:d} {:d}\n'.format(i+1, self.i34[i], *self.elnode[i, :]+1))

                # Write boundary information
                if fmt == 1 and bndfile is None:
                    self.write_bnd(fid=fid)
                if bndfile is not None:
                    with open(bndfile, 'r') as bf:
                        fid.writelines(bf.readlines())

    def write_bnd(self, fname='grd.bnd', fid=None):
            """
            Write grid's boundary information
            
            Parameters
            ----------
            fname : str, optional
                File name for boundary information (only used if fid is None)
            fid : file-like, optional
                File handle to write to
            """
            if not hasattr(self, 'nob'):
                self.compute_bnd()
            
            # Open file if needed
            if fid is None:
                bid = open(fname, 'w+')
            else:
                bid = fid

            # Write open boundary
            bid.write('{} = Number of open boundaries\n'.format(self.nob))
            bid.write('{} = Total number of open boundary nodes\n'.format(int(np.sum(self.nobn))))
            for i in range(self.nob):
                bid.write('{} = Number of nodes for open boundary {}\n'.format(self.nobn[i], i+1))
                bid.writelines(['{}\n'.format(k+1) for k in self.iobn[i]])

            # Write land boundary
            bid.write('{} = number of land boundaries\n'.format(self.nlb))
            bid.write('{} = Total number of land boundary nodes\n'.format(int(np.sum(self.nlbn))))
            nln = int(np.sum(self.island == 0))
            for i in range(self.nlb):
                if self.island[i] == 0:
                    bid.write('{} {} = Number of nodes for land boundary {}\n'.format(self.nlbn[i], self.island[i], i+1))
                else:
                    bid.write('{} {} = Number of nodes for island boundary {}\n'.format(self.nlbn[i], self.island[i], i+1-nln))
                bid.writelines(['{}\n'.format(k+1) for k in self.ilbn[i]])
            
            # Close file if we opened it
            if fid is None:
                bid.close()

    def write_prop(self, fname='schism.prop', value=None, fmt='{:8.5f}'):
            """
            Write SCHISM prop file
            
            Parameters
            ----------
            fname : str
                File name
            value : np.ndarray or float, optional
                Property values. If None, uses self.dpe
            fmt : str
                Output format for property values
            """
            # Get prop value
            if value is None:
                if not hasattr(self, 'dpe'):
                    self.compute_ctr()
                pvi = self.dpe.copy()
            else:
                if hasattr(value, "__len__"):
                    pvi = value
                else:
                    pvi = np.ones(self.ne) * value
            
            if 'd' in fmt:
                pvi = pvi.astype('int')

            # Prepare values
            fstr = ('{:d} ' + fmt + ' \n') * self.ne
            fval = np.array([list(range(1, self.ne+1)), pvi], dtype='O').T

            # Write prop value
            with open(fname, 'w+') as fid:
                fid.writelines(fstr.format(*fval.ravel()))

    def save(self, fname=None, **args):
            """
            Save grid to file
            
            Parameters
            ----------
            fname : str, optional
                File name. Extension determines format:
                *.npz, *.pkl: save as binary file
                *.gr3, *.ll, *.ic: save as SCHISM grid file
                If None, uses source_file with .npz extension
            **args : dict
                Additional arguments for write_hgrid if saving as grid
            """
            if fname is None:
                fname = f"{os.path.splitext(self.source_file)[0]}.npz"
                
            if fname.endswith('.gr3') or fname.endswith('.ll') or fname.endswith('.ic'):
                self.write_hgrid(fname, **args)
            else:
                from ..io.grid_io import save_grid
                save_grid(fname, self)

    def check_skew_elems(self, angle_min=5, fname='skew_element.bp', fmt=0):
            """
            Check SCHISM grid's skewness (elements with internal angles < angle_min)
            
            Parameters
            ----------
            angle_min : float
                Minimum angle threshold in degrees
            fname : str, optional
                File name to save skew element locations (None to skip saving)
            fmt : int
                0: default behavior; 1: return indices of skew elements
                
            Returns
            -------
            np.ndarray, optional
                Indices of skew elements if fmt=1
            """
            if not hasattr(self, 'dpe'):
                self.compute_ctr()

            # For triangles
            fp = self.i34 == 3
            x = self.x[self.elnode[fp, :3]]
            y = self.y[self.elnode[fp, :3]]
            xctr = self.xctr[fp]
            yctr = self.yctr[fp]
            zctr = self.dpe[fp]
            sind = []
            
            for i in range(3):
                id1 = i
                id2 = (i+1) % 3
                id3 = (i+2) % 3
                x1 = x[:, id1]
                x2 = x[:, id2]
                x3 = x[:, id3]
                y1 = y[:, id1]
                y2 = y[:, id2]
                y3 = y[:, id3]
                ai = np.abs(np.angle((x1-x2)+1j*(y1-y2))-np.angle((x3-x2)+1j*(y3-y2)))*180/np.pi
                sindi = np.nonzero(ai <= angle_min)[0]
                if len(sindi) != 0:
                    sind.extend(sindi)
                    
            sind = np.array(sind)
            if len(sind) != 0:
                XS3 = xctr[sind]
                YS3 = yctr[sind]
                ZS3 = zctr[sind]
                sind3 = sind.copy()
            else:
                XS3 = np.array([])
                YS3 = np.array([])
                ZS3 = np.array([])
                sind3 = np.array([])

            # For quads
            fp = self.i34 == 4
            x = self.x[self.elnode[fp, :]]
            y = self.y[self.elnode[fp, :]]
            xctr = self.xctr[fp]
            yctr = self.yctr[fp]
            zctr = self.dpe[fp]
            sind = []
            
            for i in range(4):
                id1 = i
                id2 = (i+1) % 4
                id3 = (i+2) % 4
                x1 = x[:, id1]
                x2 = x[:, id2]
                x3 = x[:, id3]
                y1 = y[:, id1]
                y2 = y[:, id2]
                y3 = y[:, id3]
                ai = np.abs(np.angle((x1-x2)+1j*(y1-y2))-np.angle((x3-x2)+1j*(y3-y2)))*180/np.pi
                sindi = np.nonzero(ai <= angle_min)[0]
                if len(sindi) != 0:
                    sind.extend(sindi)
                    
            sind = np.array(sind)
            if len(sind) != 0:
                XS4 = xctr[sind]
                YS4 = yctr[sind]
                ZS4 = zctr[sind]
                sind4 = sind
            else:
                XS4 = np.array([])
                YS4 = np.array([])
                ZS4 = np.array([])
                sind4 = np.array([])

            # Combine and save
            if fname is not None:
                from ..io.ascii_io import write_bpfile
                self.xskew = np.r_[XS3, XS4]
                self.yskew = np.r_[YS3, YS4]
                zskew = np.r_[ZS3, ZS4]
                write_bpfile(fname, self.xskew, self.yskew, zskew)
                
            if fmt == 1:
                return np.array([*sind3, *sind4]).astype('int')
                
    def write_shapefile_bnd(self, fname, prj='epsg:4326'):
                """
                Write grid boundaries to a shapefile
                
                Parameters
                ----------
                fname : str
                    Output shapefile name (without extension)
                prj : str
                    Projection identifier
                """
                from ..io.shapefile_io import write_shapefile_data
                
                self.shp_bnd = type('zdata', (), {})()
                self.shp_bnd.type = 'POLYLINE'
                xy = np.array([[], []]).T
                
                for i in range(self.nob):
                    ind = self.iobn[i]
                    xyi = np.c_[self.x[ind], self.y[ind]]
                    xyi = np.insert(xyi, 0, np.nan, axis=0)
                    xy = np.r_[xy, xyi]
                    
                for i in range(self.nlb):
                    ind = self.ilbn[i]
                    xyi = np.c_[self.x[ind], self.y[ind]]
                    if self.island[i] == 1:
                        # Close the loop if it's an island
                        if xyi[0, 0] != xyi[-1, 0] or xyi[0, 1] != xyi[-1, 1]:
                            xyi = np.r_[xyi, xyi[0:1, :]]
                    xyi = np.insert(xyi, 0, np.nan, axis=0)
                    xy = np.r_[xy, xyi]
                    
                self.shp_bnd.xy = xy
                self.shp_bnd.prj = prj
                write_shapefile_data(fname, self.shp_bnd)

    def write_shapefile_node(self, fname, prj='epsg:4326'):
                """
                Write grid nodes to a shapefile
                
                Parameters
                ----------
                fname : str
                    Output shapefile name (without extension)
                prj : str
                    Projection identifier
                """
                from ..io.shapefile_io import write_shapefile_data
                
                self.shp_node = type('zdata', (), {})()
                self.shp_node.type = 'POINT'
                self.shp_node.xy = np.c_[self.x, self.y]
                self.shp_node.attname = ['id_node']
                self.shp_node.attvalue = np.arange(self.np) + 1
                self.shp_node.prj = prj
                write_shapefile_data(fname, self.shp_node)

    def write_shapefile_element(self, fname, prj='epsg:4326'):
                """
                Write grid elements to a shapefile
                
                Parameters
                ----------
                fname : str
                    Output shapefile name (without extension)
                prj : str
                    Projection identifier
                """
                from ..io.shapefile_io import write_shapefile_data
                
                self.shp_elem = type('zdata', (), {})()
                self.shp_elem.type = 'POLYGON'
                elnode = self.elnode.copy()
                fp = elnode[:, -1] < 0
                elnode[fp, -1] = elnode[fp, 0]
                elnode = np.fliplr(elnode)
                
                xy = np.zeros((self.ne,), dtype=object)
                for i in range(self.ne):
                    # Get coordinates for this element's nodes
                    nodes = elnode[i]
                    xy_coords = np.c_[self.x[nodes], self.y[nodes]]
                    # Close the polygon if needed
                    if xy_coords[0, 0] != xy_coords[-1, 0] or xy_coords[0, 1] != xy_coords[-1, 1]:
                        xy_coords = np.r_[xy_coords, xy_coords[0:1, :]]
                    xy[i] = xy_coords
                    
                self.shp_elem.xy = xy
                self.shp_elem.attname = ['id_elem']
                self.shp_elem.attvalue = np.arange(self.ne) + 1
                self.shp_elem.prj = prj
                write_shapefile_data(fname, self.shp_elem)

    def compute_all(self, fmt=0):
                """
                Compute all geometry information of the grid
                
                Parameters
                ----------
                fmt : int
                    0: skip if attributes already exist
                    1: recompute all attributes
                """
                if (not hasattr(self, 'dpe')) or fmt == 1:
                    self.compute_ctr()
                if (not hasattr(self, 'area')) or fmt == 1:
                    self.compute_area()
                if (not hasattr(self, 'dps')) or fmt == 1:
                    self.compute_side(fmt=2)
                if (not hasattr(self, 'ine')) or fmt == 1:
                    self.compute_nne()
                if (not hasattr(self, 'ic3')) or fmt == 1:
                    self.compute_ic3()

    def split_quads(self, angle_min=60, angle_max=120, fname='new.gr3'):
                """
                Split quads that have internal angles outside the specified range
                
                Parameters
                ----------
                angle_min : float
                    Minimum acceptable angle
                angle_max : float
                    Maximum acceptable angle
                fname : str
                    Output file name for new grid
                """
                if not hasattr(self, 'index_bad_quad'):
                    self.check_quads(angle_min, angle_max)

                # Compute (angle_max-angle_min) in split triangles
                qind = self.index_bad_quad
                x = self.x[self.elnode[qind, :]]
                y = self.y[self.elnode[qind, :]]

                # Compute difference between internal angles
                A = []
                for i in range(4):
                    id1 = (i-1+4) % 4
                    id2 = i
                    id3 = (i+1) % 4
                    x1 = x[:, id1]
                    x2 = x[:, id2]
                    x3 = x[:, id3]
                    y1 = y[:, id1]
                    y2 = y[:, id2]
                    y3 = y[:, id3]

                    a1 = np.angle((x1-x2)+1j*(y1-y2))-np.angle((x3-x2)+1j*(y3-y2))
                    a2 = np.angle((x2-x3)+1j*(y2-y3))-np.angle((x1-x3)+1j*(y1-y3))
                    a3 = np.angle((x3-x1)+1j*(y3-y1))-np.angle((x2-x1)+1j*(y2-y1))
                    a1 = np.mod(a1*180/np.pi+360, 360)
                    a2 = np.mod(a2*180/np.pi+360, 360)
                    a3 = np.mod(a3*180/np.pi+360, 360)

                    # Compute amax-amin
                    a = np.c_[a1, a2, a3]
                    Ai = a.max(axis=1) - a.min(axis=1)
                    if i == 0:
                        A = Ai
                    else:
                        A = np.c_[A, Ai]

                # Split quads
                flag = np.sign(A[:, 0] + A[:, 2] - A[:, 1] - A[:, 3])

                ne = self.ne
                nea = len(self.index_bad_quad)
                self.elnode = np.r_[self.elnode, np.ones([nea, 4])-3].astype('int')
                for i in range(nea):
                    ind = self.index_bad_quad[i]
                    nds = self.elnode[ind, :].copy()
                    if flag[i] >= 0:
                        self.elnode[ind, :] = np.r_[nds[[0, 1, 2]], -2]
                        self.i34[ind] = 3
                        self.elnode[ne+i, :] = np.r_[nds[[2, 3, 0]], -2]
                    else:
                        self.elnode[ind, :] = np.r_[nds[[1, 2, 3]], -2]
                        self.i34[ind] = 3
                        self.elnode[ne+i, :] = np.r_[nds[[3, 0, 1]], -2]

                self.ne = ne + nea
                self.i34 = np.r_[self.i34, np.ones(nea)*3].astype('int')
                self.elnode = self.elnode.astype('int')

                # Write new grid
                self.write_hgrid(fname)

    def check_quads(self, angle_min=60, angle_max=120, fname='bad_quad.bp'):
                """
                Check the quality of quads
                
                Parameters
                ----------
                angle_min : float
                    Minimum acceptable internal angle
                angle_max : float
                    Maximum acceptable internal angle
                fname : str
                    Output file name to save locations of bad quads
                """
                qind = np.nonzero(self.i34 == 4)[0]
                x = self.x[self.elnode[qind, :]]
                y = self.y[self.elnode[qind, :]]

                # Compute internal angle
                a = []
                for i in range(4):
                    id1 = (i-1+4) % 4
                    id2 = i
                    id3 = (i+1) % 4
                    x1 = x[:, id1]
                    x2 = x[:, id2]
                    x3 = x[:, id3]
                    y1 = y[:, id1]
                    y2 = y[:, id2]
                    y3 = y[:, id3]

                    ai = np.angle((x1-x2)+1j*(y1-y2))-np.angle((x3-x2)+1j*(y3-y2))
                    a.append(ai*180/np.pi)
                a = np.array(a).T
                a = np.mod(a+360, 360)

                # Check violation
                fp = np.zeros_like(a[:, 0], dtype=bool)
                for i in range(4):
                    fp = fp | (a[:, i] <= angle_min) | (a[:, i] >= angle_max)

                self.index_bad_quad = qind[np.nonzero(fp)[0]]

                # Output bad_quad location as bp file
                if not hasattr(self, 'xctr'):
                    self.compute_ctr()
                    
                if len(self.index_bad_quad) > 0:
                    qxi = self.xctr[self.index_bad_quad]
                    qyi = self.yctr[self.index_bad_quad]
                    from ..io.ascii_io import write_bpfile
                    write_bpfile(fname, qxi, qyi, np.zeros(len(qxi)))

    def plot_bad_quads(self, color='r', ms=12, *args, **kwargs):
                """
                Plot grid with bad quads highlighted
                
                Parameters
                ----------
                color : str
                    Color for bad quad markers
                ms : float
                    Marker size for bad quads
                *args, **kwargs :
                    Additional arguments passed to plt.plot
                """
                if not hasattr(self, 'index_bad_quad'):
                    self.check_quads()
                if not hasattr(self, 'xctr'):
                    self.compute_ctr()

                qxi = self.xctr[self.index_bad_quad]
                qyi = self.yctr[self.index_bad_quad]
                self.plot_grid()
                plt.plot(qxi, qyi, '.', color=color, ms=ms, *args, **kwargs)
