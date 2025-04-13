# stofs3d/core/grid.py
"""
Grid handling functionality for STOFS3D
Based on schism_grid class from schism_file.py
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib as mpl
from .coordinate_utils import inside_polygon, signa

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
            self
