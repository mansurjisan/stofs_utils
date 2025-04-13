"""
Coordinate utility functions for STOFS3D

Contains functions for coordinate transformations, projections,
area coordinate calculations, and vertical coordinate interpolation.
"""
import numpy as np
from pyproj import Transformer
import warnings


def get_zcor_interp_coefficient(zcor, zinter, kbp, nvrt=None):
    """
    Calculate vertical interpolation coefficients for z-coordinates.
    
    Parameters
    ----------
    zcor : np.ndarray
        Z-coordinates array of shape [np, nvrt]
    zinter : np.ndarray
        Target depths where values will be interpolated, shape [np]
    kbp : np.ndarray
        Bottom layer indices, shape [np]
    nvrt : int, optional
        Number of vertical layers (if None, inferred from zcor)
        
    Returns
    -------
    tuple
        (k1, coeff) - layer indices and interpolation coefficients
        k1 : np.ndarray - k-level at each node
        coeff : np.ndarray - interpolation coefficient
    """
    if nvrt is None:
        nvrt = zcor.shape[1]
    
    # Initialize output arrays
    k1 = np.zeros(len(zinter), dtype=int)
    coeff = np.zeros(len(zinter), dtype=float)
    
    # Surface
    idxs = zinter >= zcor[:, -1]
    k1[idxs] = nvrt - 2
    coeff[idxs] = 1.0
    
    # Bottom
    idxs = zinter < zcor[:, 0]
    k1[idxs] = kbp[idxs]
    coeff[idxs] = 0.0
    
    # Intermediate layers
    for k in np.arange(nvrt - 1):
        idxs = (zinter >= zcor[:, k]) * (zinter < zcor[:, k + 1])
        k1[idxs] = k
        coeff[idxs] = (zinter[idxs] - zcor[idxs, k]) / (zcor[idxs, k+1] - zcor[idxs, k])
    
    # Check for issues
    if np.any(np.isnan(np.r_[k1, coeff])):
        raise ValueError("Error in vertical interpolation - NaN values detected")
        
    return np.array(k1).astype('int'), np.array(coeff)


def signa(x, y):
    """
    Compute signed area for triangles.
    
    Parameters
    ----------
    x : np.ndarray
        x-coordinates, last dimension must have 3 elements
    y : np.ndarray
        y-coordinates, last dimension must have 3 elements
        
    Returns
    -------
    np.ndarray
        Signed areas
    """
    if x.ndim == 1:
        area = ((x[0] - x[2]) * (y[1] - y[2]) - (x[1] - x[2]) * (y[0] - y[2])) / 2
    elif x.ndim == 2:
        area = ((x[..., 0] - x[..., 2]) * (y[..., 1] - y[..., 2]) - 
                (x[..., 1] - x[..., 2]) * (y[..., 0] - y[..., 2])) / 2
    
    area = np.squeeze(area)
    return area


def inside_polygon(pts, px, py, fmt=0, method=0):
    """
    Check whether points are inside polygons.
    
    Parameters
    ----------
    pts : np.ndarray
        Points to check, shape [npt, 2]
    px : np.ndarray
        x-coordinates of polygon vertices
    py : np.ndarray
        y-coordinates of polygon vertices
    fmt : int, optional
        Output format
        - 0: Return boolean mask [npt, nploy] (default)
        - 1: Return polygon indices [npt] (-1 if outside all polygons)
    method : int, optional
        Algorithm method
        - 0: Use matplotlib.path.Path (default)
        - 1: Use explicit ray method
        
    Returns
    -------
    np.ndarray
        Boolean mask or indices depending on fmt
    """
    import matplotlib.path as mpath
    
    # Check dimension
    if px.ndim == 1:
        px = px[:, None]
        py = py[:, None]
    
    # Get dimensions
    npt = pts.shape[0]
    nv, npy = px.shape
    
    if nv == 3 and fmt == 1:
        # For triangles with fmt=1
        px1 = px.min(axis=0)
        px2 = px.max(axis=0)
        py1 = py.min(axis=0)
        py2 = py.max(axis=0)
        
        sind = []
        for i in range(npt):
            pxi = pts[i, 0]
            pyi = pts[i, 1]
            sindp = np.nonzero((pxi >= px1) * (pxi <= px2) * (pyi >= py1) * (pyi <= py2))[0]
            npy_local = len(sindp)
            
            if npy_local == 0:
                sind.append(-1)
            else:
                isum = np.ones(npy_local)
                for m in range(nv):
                    xi = np.c_[np.ones(npy_local) * pxi, px[m, sindp], px[np.mod(m+1, nv), sindp]]
                    yi = np.c_[np.ones(npy_local) * pyi, py[m, sindp], py[np.mod(m+1, nv), sindp]]
                    area = signa(xi, yi)
                    fp = area < 0
                    isum[fp] = 0
                    
                sindi = np.nonzero(isum != 0)[0]
                
                if len(sindi) == 0:
                    sind.append(-1)
                else:
                    sind.append(sindp[sindi[0]])
                    
        sind = np.array(sind)
    else:
        if method == 0:
            sind = []
            for m in range(npy):
                path = mpath.Path(np.c_[px[:, m], py[:, m]])
                sindi = path.contains_points(pts)
                sind.append(sindi)
            sind = np.array(sind).T + 0  # Convert logical to int
            
        elif method == 1:
            # Ray method
            sind = np.ones([npt, npy])
            x1 = pts[:, 0][:, None]
            y1 = pts[:, 1][:, None]
            
            for m in range(nv):
                x2 = px[m, :][None, :]
                y2 = py[m, :][None, :]
                isum = np.zeros([npt, npy])
                
                for n in range(1, nv-1):
                    x3 = px[(n+m) % nv, :][None, :]
                    y3 = py[(n+m) % nv, :][None, :]
                    x4 = px[(n+m+1) % nv, :][None, :]
                    y4 = py[(n+m+1) % nv, :][None, :]
                    
                    # Ray intersection tests
                    fp1 = ((y1-y3)*(x2-x1)+(x3-x1)*(y2-y1))*((y1-y4)*(x2-x1)+(x4-x1)*(y2-y1)) <= 0
                    fp2 = ((y2-y1)*(x4-x3)-(y4-y3)*(x2-x1))*((y4-y3)*(x1-x3)+(y3-y1)*(x4-x3)) <= 0
                    
                    fp12 = fp1 * fp2
                    isum[fp12] = isum[fp12] + 1
                    
                fp = ((isum % 2) == 0) | ((x1 == x2) * (y1 == y2))
                sind[fp] = 0
        
        # Change format if needed
        if fmt == 1:
            sindm = np.argmax(sind, axis=1)
            sindm[sind[np.arange(npt), sindm] == 0] = -1
            sind = sindm
        elif fmt == 0 and npy == 1:
            sind = sind[:, 0]
            
    return sind


def proj(fname0=None, fmt0=None, prj0=None, fname1=None, fmt1=None, prj1=None, 
         x=None, y=None, lon0=None, lat0=None, order0=0, order1=0):
    """
    Transform projection of coordinates or files.
    
    Parameters
    ----------
    fname0 : str, optional
        Input filename (if transforming a file)
    fmt0 : int, optional
        Input file format
    prj0 : str, optional
        Input projection identifier (e.g., 'epsg:26918')
    fname1 : str, optional
        Output filename (if saving to a file)
    fmt1 : int, optional
        Output file format
    prj1 : str, optional
        Output projection identifier (e.g., 'epsg:4326')
    x : np.ndarray, optional
        X coordinates to transform
    y : np.ndarray, optional
        Y coordinates to transform
    lon0 : float, optional
        Reference longitude for CPP projection
    lat0 : float, optional
        Reference latitude for CPP projection
    order0 : int, optional
        Input coordinate order (0: projected, 1: lat/lon)
    order1 : int, optional
        Output coordinate order (0: projected, 1: lat/lon)
        
    Returns
    -------
    tuple or None
        Transformed coordinates [x, y] if not writing to a file
    """
    # Check projections
    prj0 = prj0.lower() if prj0 else None
    prj1 = prj1.lower() if prj1 else None
    
    # Handle direct coordinate conversion
    if x is not None and y is not None:
        if None in (prj0, prj1):
            raise ValueError("Both source and target projections must be specified")
            
        icpp = 0  # Custom CPP projection flag
        if prj0 == 'epsg:4326':
            order0 = 1
        if prj1 == 'epsg:4326':
            order1 = 1
        if 'cpp' in (prj0, prj1):
            icpp = 1
            rearth = 6378206.4
        
        # CPP projections
        if icpp == 1 and prj0 == 'cpp':
            if prj1 != 'epsg:4326':
                raise ValueError(f"Unsupported projection combination: {prj0} -> {prj1}")
            if None in (lon0, lat0):
                raise ValueError("lon0 and lat0 are required for cpp=>ll transform")
                
            x1 = lon0 + x * 180 / (np.pi * rearth * np.cos(lat0 * np.pi / 180))
            y1 = y * 180 / (np.pi * rearth)
            
        elif icpp == 1 and prj1 == 'cpp':
            if prj0 != 'epsg:4326':
                raise ValueError(f"Unsupported projection combination: {prj0} -> {prj1}")
                
            if lon0 is None:
                lon0 = np.mean(x)
            if lat0 is None:
                lat0 = np.mean(y)
                
            x1 = rearth * (x - lon0) * (np.pi / 180) * np.cos(lat0 * np.pi / 180)
            y1 = rearth * y * np.pi / 180
            
        else:
            # Standard projections using pyproj
            if order0 == 1:
                x, y = y, x  # Swap for lat/lon
                
            fpn = ~(np.isnan(x) | np.isnan(y))
            x1 = np.full_like(x, np.nan)
            y1 = np.full_like(y, np.nan)
            
            transformer = Transformer.from_crs(prj0, prj1)
            x1[fpn], y1[fpn] = transformer.transform(x[fpn], y[fpn])
            
            if order1 == 1:
                x1, y1 = y1, x1  # Swap for lat/lon output
                
            if np.any(np.isnan(x1[fpn])) or np.any(np.isnan(y1[fpn])):
                raise ValueError("NaN values found in transformation")
                
        return [x1, y1]
    
    # If no coordinates provided, this is file-based transformation
    # This would need to be implemented based on your file handling code
    warnings.warn("File-based projection transformation not implemented in this function")
    return None


def proj_pts(x, y, prj1='epsg:4326', prj2='epsg:26918'):
    """
    Convert projection of points from prj1 to prj2.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    prj1 : str, optional
        Source projection (default: 'epsg:4326')
    prj2 : str, optional
        Target projection (default: 'epsg:26918')
        
    Returns
    -------
    tuple
        Transformed coordinates [px, py]
    """
    px, py = proj(prj0=prj1, prj1=prj2, x=x, y=y)
    return [px, py]


def near_pts(points_a, points_b, method=0, N=100):
    """
    Find indices of nearest points in points_b for each point in points_a.
    
    Parameters
    ----------
    points_a : np.ndarray
        Query points, shape [n, 2]
    points_b : np.ndarray
        Reference points, shape [m, 2]
    method : int, optional
        Method to use:
        - 0: scipy.spatial.cKDTree (default, fastest)
        - 1: Quick method by subgroups
        - 2: Slower direct calculation
    N : int, optional
        Subgroup size for method=1, default=100
        
    Returns
    -------
    np.ndarray
        Indices of nearest points in points_b
    """
    if method == 0:
        # Fast method using KD Tree
        from scipy import spatial
        tree = spatial.cKDTree(points_b)
        return tree.query(points_a)[1]
        
    elif method == 1:
        # Subgroup method
        p = points_a[:, 0] + (1j) * points_a[:, 1]
        p0 = points_b[:, 0] + (1j) * points_b[:, 1]
        
        # Limit N to array size
        N = min(N, len(p))
        
        # Divide points into subgroups based on distance
        ps0 = []
        ps = []
        ds = []
        inds = []
        inum = np.arange(len(p))
        
        while True:
            if len(inum) == 0:
                break
                
            dist = abs(p[inum] - p[inum[0]])
            sind = np.argsort(dist)
            inum = inum[sind]
            dist = dist[sind]
            sN = min(N, len(inum))
            
            ps0.append(p[inum[0]])
            ps.append(p[inum[:sN]])
            ds.append(dist[sN-1])
            inds.append(inum[:sN])
            inum = inum[sN:]
        
        # Find nearest points for each subgroup
        inds0 = []
        for m in range(len(ps)):
            dist = abs(p0 - ps0[m])
            dsi = ds[m]
            psi = ps[m]
            
            # Find radius around ps0[m]
            dsm = dsi
            while True:
                fp = dist <= dsm
                if np.sum(fp) > 0:
                    break
                else:
                    dsm = dsm + dsi
            
            # Subgroup points of p0
            fp = dist <= (dsm + 2 * dsi)
            ind0 = np.nonzero(fp)[0]
            p0i = p0[ind0]
            
            psii = psi[:, None]
            p0ii = p0i[None, :]
            dist = abs(psii - p0ii)
            indi = dist.argmin(axis=1)
            inds0.append(ind0[indi])
        
        # Arrange index
        pind = np.array([]).astype('int')
        pind0 = np.array([]).astype('int')
        
        for m in range(len(inds)):
            pind = np.r_[pind, inds[m]]
            pind0 = np.r_[pind0, inds0[m]]
        
        ind = np.argsort(pind)
        sind = pind0[ind]
        
    elif method == 2:
        # Direct calculation
        n = points_a.shape[0]
        n0 = points_b.shape[0]
        N = max(min(1e7 // n0, n), 1e2)
        print(f'Total points: {n}')
        
        i0 = int(0)
        i1 = int(N)
        ind = np.array([])
        
        while True:
            print(f'Processing points: {i0}-{i1}')
            x = points_a[i0:i1, 0]
            y = points_a[i0:i1, 1]
            x0 = points_b[:, 0]
            y0 = points_b[:, 1]
            
            dist = (x[None, :] - x0[:, None])**2 + (y[None, :] - y0[:, None])**2
            
            indi = []
            for i in range(x.shape[0]):
                disti = dist[:, i]
                indi.append(np.nonzero(disti == min(disti))[0][0])
            
            if i0 == 0:
                ind = np.array(indi)
            else:
                ind = np.r_[ind, np.squeeze(np.array(indi))]
            
            # Next step
            i0 = int(i0 + N)
            i1 = int(min(i1 + N, n))
            if i0 >= n:
                break
                
        sind = ind
        
    return sind
