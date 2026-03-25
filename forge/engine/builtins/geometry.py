"""Geometry Toolbox for Forge.

Provides 15 computational geometry functions compatible with Octave/MATLAB
geometry conventions. Uses 1-based indexing for returned vertex/simplex indices.

Backend: scipy.spatial + numpy.
"""

from __future__ import annotations

import numpy as np

from forge.engine.types import ForgeArray, _unwrap


# ── Helpers ──────────────────────────────────────────────────────

def _wrap(value):
    """Wrap a numpy array as a ForgeArray."""
    if isinstance(value, np.ndarray):
        return ForgeArray(value)
    return value


def _scalar(value):
    """Extract scalar from 0-d or single-element array."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return value.item()
    return value


# ── Toolbox function registry ───────────────────────────────────
GEOMETRY_REGISTRY: dict[str, callable] = {}


def _tb(name: str | None = None):
    """Local decorator to register a toolbox function."""
    def decorator(func):
        fn_name = name or func.__name__
        GEOMETRY_REGISTRY[fn_name] = func
        return func
    return decorator


# =====================================================================
# Convex Hull
# =====================================================================

@_tb("convhull")
def forge_convhull(x, y=None):
    """Compute convex hull of 2-D or 3-D points.

    k = convhull(x, y)
    k = convhull(x, y, z)
    [k, vol] = convhull(...)

    For 2-D: x, y are coordinate vectors.
    If y is None, x is assumed to be an Nx2 or Nx3 matrix.

    Returns vertex indices (1-based) of the convex hull boundary.
    """
    from scipy.spatial import ConvexHull

    if y is not None:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        pts = np.column_stack([x_arr, y_arr])
    else:
        pts = np.asarray(x, dtype=np.float64)

    hull = ConvexHull(pts)
    # Return 1-based indices; for 2-D, return ordered boundary vertices
    if pts.shape[1] == 2:
        verts = hull.vertices + 1
        # Close the polygon (MATLAB convention)
        verts = np.append(verts, verts[0])
        return (_wrap(verts), hull.volume)
    else:
        return (_wrap(hull.simplices + 1), hull.volume)


# =====================================================================
# Delaunay Triangulation
# =====================================================================

@_tb("delaunay")
def forge_delaunay(x, y=None):
    """Delaunay triangulation of 2-D points.

    tri = delaunay(x, y)

    Returns simplex connectivity list with 1-based vertex indices.
    Each row of tri defines one triangle (3 vertex indices).
    """
    from scipy.spatial import Delaunay as _Delaunay

    if y is not None:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        pts = np.column_stack([x_arr, y_arr])
    else:
        pts = np.asarray(x, dtype=np.float64)

    tri = _Delaunay(pts)
    return _wrap(tri.simplices + 1)


@_tb("delaunayn")
def forge_delaunayn(P):
    """N-dimensional Delaunay triangulation.

    T = delaunayn(P)

    P: (npoints, ndim) array of point coordinates.
    Returns simplex indices (1-based).
    """
    from scipy.spatial import Delaunay as _Delaunay

    P = np.asarray(P, dtype=np.float64)
    tri = _Delaunay(P)
    return _wrap(tri.simplices + 1)


# =====================================================================
# Voronoi Diagram
# =====================================================================

@_tb("voronoi")
def forge_voronoi(x, y=None):
    """Compute Voronoi diagram for 2-D points.

    [vx, vy] = voronoi(x, y)

    Returns the Voronoi edge coordinates suitable for plotting.
    vx and vy are 2xN arrays where each column is a line segment.
    """
    from scipy.spatial import Voronoi as _Voronoi

    if y is not None:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        pts = np.column_stack([x_arr, y_arr])
    else:
        pts = np.asarray(x, dtype=np.float64)

    vor = _Voronoi(pts)

    # Collect finite Voronoi edges
    vx_list = []
    vy_list = []
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:
            v0 = vor.vertices[ridge[0]]
            v1 = vor.vertices[ridge[1]]
            vx_list.append([v0[0], v1[0]])
            vy_list.append([v0[1], v1[1]])

    if vx_list:
        vx = np.array(vx_list).T  # 2 x N
        vy = np.array(vy_list).T
    else:
        vx = np.array([]).reshape(2, 0)
        vy = np.array([]).reshape(2, 0)

    return (_wrap(vx), _wrap(vy))


@_tb("voronoin")
def forge_voronoin(P):
    """N-dimensional Voronoi diagram.

    [V, C] = voronoin(P)

    V: Voronoi vertices array.
    C: Cell array of vertex indices for each point's Voronoi cell
       (1-based; -1 denotes vertex at infinity).
    """
    from scipy.spatial import Voronoi as _Voronoi

    P = np.asarray(P, dtype=np.float64)
    vor = _Voronoi(P)
    vertices = _wrap(vor.vertices)

    # Convert region indices: -1 stays -1, others become 1-based
    regions = []
    for reg_idx in vor.point_region:
        region = vor.regions[reg_idx]
        regions.append([v + 1 if v >= 0 else -1 for v in region])

    return (vertices, regions)


# =====================================================================
# Point Location / Search
# =====================================================================

@_tb("tsearchn")
def forge_tsearchn(X, T, XI):
    """Find enclosing simplex for query points.

    [idx, P] = tsearchn(X, T, XI)

    X:  triangulation vertices (npoints, ndim)
    T:  simplex connectivity (nsimplices, ndim+1), 1-based indices
    XI: query points (nquery, ndim)

    Returns simplex indices (1-based) containing each query point.
    Points outside the triangulation receive NaN.
    """
    from scipy.spatial import Delaunay as _Delaunay

    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64) - 1  # convert to 0-based
    XI = np.asarray(XI, dtype=np.float64)

    # Build Delaunay from vertices
    tri = _Delaunay(X)
    # Use scipy's find_simplex
    simplex_idx = tri.find_simplex(XI)

    # Convert to 1-based, NaN for outside
    result = np.where(simplex_idx >= 0, simplex_idx + 1, np.nan)
    return _wrap(result)


@_tb("dsearchn")
def forge_dsearchn(P, PQ):
    """Nearest-point search using KD-tree.

    [k, d] = dsearchn(P, PQ)

    For each query point in PQ, returns the index (1-based) of the
    closest point in P and the distance.
    """
    from scipy.spatial import KDTree

    P = np.asarray(P, dtype=np.float64)
    PQ = np.asarray(PQ, dtype=np.float64)

    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if PQ.ndim == 1:
        PQ = PQ.reshape(-1, 1)

    tree = KDTree(P)
    dist, idx = tree.query(PQ)

    return (_wrap(np.asarray(idx) + 1), _wrap(np.asarray(dist)))


# =====================================================================
# Interpolation on Scattered Data
# =====================================================================

@_tb("griddata")
def forge_griddata(x, y, v, xq, yq, method="linear"):
    """Interpolate scattered data onto grid.

    vq = griddata(x, y, v, xq, yq)
    vq = griddata(x, y, v, xq, yq, method)

    x, y: scattered data point coordinates
    v: values at scattered points
    xq, yq: query grid coordinates
    method: 'linear' (default), 'nearest', 'cubic'

    Returns interpolated values at query points.
    """
    from scipy.interpolate import griddata as _griddata

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    xq = np.asarray(xq, dtype=np.float64)
    yq = np.asarray(yq, dtype=np.float64)

    points = np.column_stack([x, y])
    xi = np.column_stack([xq.ravel(), yq.ravel()])

    method = str(method).lower()
    vq = _griddata(points, v, xi, method=method)
    return _wrap(vq.reshape(xq.shape))


@_tb("griddatan")
def forge_griddatan(X, v, XI, method="linear"):
    """N-dimensional scattered data interpolation.

    vq = griddatan(X, v, XI)
    vq = griddatan(X, v, XI, method)

    X: (npoints, ndim) data point coordinates
    v: values at data points
    XI: (nquery, ndim) query points
    method: 'linear' (default), 'nearest'
    """
    from scipy.interpolate import griddata as _griddata

    X = np.asarray(X, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).ravel()
    XI = np.asarray(XI, dtype=np.float64)

    method = str(method).lower()
    vq = _griddata(X, v, XI, method=method)
    return _wrap(vq)


# =====================================================================
# Point-in-Polygon
# =====================================================================

@_tb("inpolygon")
def forge_inpolygon(xq, yq, xv, yv):
    """Test if points are inside polygon.

    [in, on] = inpolygon(xq, yq, xv, yv)

    xq, yq: query point coordinates
    xv, yv: polygon vertex coordinates

    Returns:
      in_val: logical array, true if point is inside or on boundary
      on_val: logical array, true if point is on the boundary
    """
    from matplotlib.path import Path

    xq = np.asarray(xq, dtype=np.float64)
    yq = np.asarray(yq, dtype=np.float64)
    xv = np.asarray(xv, dtype=np.float64).ravel()
    yv = np.asarray(yv, dtype=np.float64).ravel()

    shape = xq.shape
    xq_flat = xq.ravel()
    yq_flat = yq.ravel()

    # Build polygon path
    poly_verts = np.column_stack([xv, yv])
    path = Path(poly_verts)

    query_pts = np.column_stack([xq_flat, yq_flat])

    # Points inside (including boundary)
    in_val = path.contains_points(query_pts)
    # Points on boundary
    on_val = path.contains_points(query_pts, radius=0.0) & ~path.contains_points(query_pts, radius=-1e-10)

    in_result = in_val.reshape(shape)
    on_result = on_val.reshape(shape)

    return (in_result, on_result)


# =====================================================================
# Rectangle Intersection
# =====================================================================

@_tb("rectint")
def forge_rectint(A, B):
    """Compute area of intersection of rectangles.

    area = rectint(A, B)

    A: Mx4 matrix, each row [x, y, width, height]
    B: Nx4 matrix, each row [x, y, width, height]

    Returns MxN matrix of intersection areas.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)

    m = A.shape[0]
    n = B.shape[0]
    result = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        ax, ay, aw, ah = A[i, 0], A[i, 1], A[i, 2], A[i, 3]
        for j in range(n):
            bx, by, bw, bh = B[j, 0], B[j, 1], B[j, 2], B[j, 3]

            # Intersection
            x_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            y_overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
            result[i, j] = x_overlap * y_overlap

    return _scalar(_wrap(result))


# =====================================================================
# 3-D Rotation Matrices
# =====================================================================

@_tb("rotx")
def forge_rotx(angle_deg):
    """3x3 rotation matrix about the X-axis.

    R = rotx(angle)

    angle: rotation angle in degrees.

    R = [1     0          0    ]
        [0  cos(a)   -sin(a)  ]
        [0  sin(a)    cos(a)  ]
    """
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    return _wrap(np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c,  -s ],
        [0.0,  s,   c ],
    ], dtype=np.float64))


@_tb("roty")
def forge_roty(angle_deg):
    """3x3 rotation matrix about the Y-axis.

    R = roty(angle)

    angle: rotation angle in degrees.

    R = [ cos(a)  0  sin(a)]
        [   0     1    0   ]
        [-sin(a)  0  cos(a)]
    """
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    return _wrap(np.array([
        [ c,  0.0,  s ],
        [0.0, 1.0, 0.0],
        [-s,  0.0,  c ],
    ], dtype=np.float64))


@_tb("rotz")
def forge_rotz(angle_deg):
    """3x3 rotation matrix about the Z-axis.

    R = rotz(angle)

    angle: rotation angle in degrees.

    R = [cos(a)  -sin(a)  0]
        [sin(a)   cos(a)  0]
        [  0        0     1]
    """
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    return _wrap(np.array([
        [ c,  -s,  0.0],
        [ s,   c,  0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64))
