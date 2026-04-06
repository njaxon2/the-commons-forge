"""Round-19 polish: interpolation, computational geometry, coordinate transforms.

V&V Traceability (backfill):
    R-POL19-01 .. R-POL19-03 (parent requirements)
    R-POL19-01-nn .. R-POL19-03-nn (unit sub-requirements)

SRS trace: SRS-FUNC-001, SRS-VAL-001, SRS-COMPAT-001
"""
import pytest, math
from forge.engine.session import ForgeSession

@pytest.fixture(scope="module")
def s():
    return ForgeSession()

def _ws(s, name):
    """Get raw workspace variable at full precision."""
    return float(s.get_workspace_dict()[name])

# ── Interpolation ───────────────────────────────────────────────────


class _InterpolationGroup:
    """R-POL19-01: Forge SHALL provide interp1, interp2, and griddata functions
    that interpolate data using linear, spline, nearest, and extrapolation
    methods, producing results consistent with MATLAB/Octave reference values.

    Model-user argument: An engineer migrating signal processing or curve-fitting
    scripts from Octave relies on interp1/interp2 for resampling sensor data,
    reconstructing waveforms, and filling measurement gaps. Incorrect interpolation
    silently corrupts downstream calculations such as FFTs or integration.

    Decomposition:
        R-POL19-01-01: interp1 linear interpolation at midpoint
        R-POL19-01-02: interp1 spline interpolation at midpoint
        R-POL19-01-03: interp1 nearest-neighbor interpolation
        R-POL19-01-04: interp1 linear extrapolation beyond data range
        R-POL19-01-05: interp2 on meshgrid surface
        R-POL19-01-06: griddata scattered-point interpolation

    Consistency: Sub-requirements cover all four interp1 methods (linear, spline,
    nearest, extrap), plus 2-D grid (interp2) and scattered (griddata)
    interpolation. Together they verify the full interpolation API surface.
    """
    pass


def test_interp1_linear(s):
    """R-POL19-01-01: interp1 linear at midpoint yields 25.0."""
    r = s.eval("interp1([1 2 3 4], [10 20 30 40], 2.5)")
    assert float(r) == pytest.approx(25.0)

def test_interp1_spline(s):
    """R-POL19-01-02: interp1 spline at midpoint yields approx 25.0."""
    r = s.eval("interp1([1 2 3 4], [10 20 30 40], 2.5, 'spline')")
    assert float(r) == pytest.approx(25.0, abs=1.0)

def test_interp1_nearest(s):
    """R-POL19-01-03: interp1 nearest at 2.3 yields 20.0."""
    r = s.eval("interp1([1 2 3 4], [10 20 30 40], 2.3, 'nearest')")
    assert float(r) == pytest.approx(20.0)

def test_interp1_extrap(s):
    """R-POL19-01-04: interp1 linear extrap beyond range yields 50.0."""
    r = s.eval("interp1([1 2 3 4], [10 20 30 40], 5, 'linear', 'extrap')")
    assert float(r) == pytest.approx(50.0)

def test_interp2_meshgrid(s):
    """R-POL19-01-05: interp2 on X+Y surface at (1.5,1.5) yields 3.0."""
    r = s.eval("[X,Y] = meshgrid(1:3, 1:3); V = X + Y; interp2(X, Y, V, 1.5, 1.5)")
    assert float(r) == pytest.approx(3.0)

def test_griddata_basic(s):
    """R-POL19-01-06: griddata on unit square corners interpolates center."""
    r = s.eval("griddata([0 1 1 0], [0 0 1 1], [1 2 3 4], 0.5, 0.5)")
    assert float(r) == pytest.approx(3.0, abs=1.0)

# ── Computational geometry ──────────────────────────────────────────


class _ComputationalGeometryGroup:
    """R-POL19-02: Forge SHALL provide convhull, delaunay, inpolygon, and
    polyarea functions that compute convex hulls, Delaunay triangulations,
    point-in-polygon tests, and polygon areas matching MATLAB/Octave behavior.

    Model-user argument: A scientist porting geospatial or mesh-generation code
    from Octave uses convhull for boundary detection, delaunay for FEM mesh
    preprocessing, inpolygon for region classification, and polyarea for area
    measurements. Each must produce correct geometric results to avoid invalid
    meshes or misclassified spatial data.

    Decomposition:
        R-POL19-02-01: convhull excludes interior points
        R-POL19-02-02: delaunay of unit square yields 2 triangles
        R-POL19-02-03: inpolygon returns 1 for interior point
        R-POL19-02-04: inpolygon returns 0 for exterior point
        R-POL19-02-05: inpolygon returns 1 for edge point
        R-POL19-02-06: polyarea of unit square equals 1.0
        R-POL19-02-07: polyarea of right triangle equals 6.0

    Consistency: The sub-requirements cover hull computation (01), triangulation
    (02), all three in/out/edge polygon cases (03-05), and area for two distinct
    shapes (06-07). This validates the full computational geometry API.
    """
    pass


def test_convhull_square(s):
    """R-POL19-02-01: Interior point (0.5,0.5) excluded from hull."""
    r = s.eval("convhull([0 1 1 0 0.5], [0 0 1 1 0.5])")
    # Result is a formatted string like "    1    2    3    4    1"
    nums = [int(x) for x in str(r).split()]
    assert len(nums) == 5
    # vertex 5 (the interior point) should NOT appear
    assert 5 not in nums

def test_delaunay_square(s):
    """R-POL19-02-02: 4-point square yields 2 triangles."""
    r = s.eval("delaunay([0 1 1 0], [0 0 1 1])")
    # Result is multi-line string; each line is a triangle
    lines = [l.strip() for l in str(r).strip().split('\n') if l.strip()]
    assert len(lines) == 2
    for line in lines:
        verts = [int(x) for x in line.split()]
        assert len(verts) == 3

def test_inpolygon_inside(s):
    """R-POL19-02-03: inpolygon returns 1 for interior point."""
    r = s.eval("inpolygon(0.5, 0.5, [0 1 1 0], [0 0 1 1])")
    assert float(r) == 1

def test_inpolygon_outside(s):
    """R-POL19-02-04: inpolygon returns 0 for exterior point."""
    r = s.eval("inpolygon(2, 2, [0 1 1 0], [0 0 1 1])")
    assert float(r) == 0

def test_inpolygon_edge(s):
    """R-POL19-02-05: inpolygon returns 1 for edge point."""
    r = s.eval("inpolygon(0, 0.5, [0 1 1 0], [0 0 1 1])")
    assert float(r) == 1

def test_polyarea_unit_square(s):
    """R-POL19-02-06: polyarea of unit square equals 1.0."""
    r = s.eval("polyarea([0 1 1 0], [0 0 1 1])")
    assert float(r) == pytest.approx(1.0)

# ── Coordinate transforms ──────────────────────────────────────────


class _CoordinateTransformGroup:
    """R-POL19-03: Forge SHALL provide cart2pol, pol2cart, cart2sph, and sph2cart
    coordinate transform functions that produce results matching MATLAB/Octave
    reference values within machine precision.

    Model-user argument: Engineers working in robotics, antenna design, and
    signal processing routinely convert between Cartesian, polar, and spherical
    coordinate systems. Incorrect transforms propagate errors into beam patterns,
    trajectory calculations, and spatial filtering, making numerical agreement
    with the reference implementation essential.

    Decomposition:
        R-POL19-03-01: cart2pol(3,4) yields r=5, theta=atan2(4,3)
        R-POL19-03-02: pol2cart(pi/4,1) yields correct x,y
        R-POL19-03-03: cart2sph(1,1,1) yields correct az,el,r
        R-POL19-03-04: sph2cart(pi/4,0,1) yields correct x,y,z

    Consistency: Each of the four coordinate transform functions is tested with
    known analytic values. Together they verify all four conversion directions
    (2-D and 3-D, forward and inverse).
    """
    pass


def test_cart2pol(s):
    """R-POL19-03-01: cart2pol(3,4) yields r=5, theta=atan2(4,3)."""
    s.eval("[th,r] = cart2pol(3,4)")
    th = _ws(s, "th")
    r = _ws(s, "r")
    assert r == pytest.approx(5.0)
    assert th == pytest.approx(math.atan2(4, 3))

def test_pol2cart(s):
    """R-POL19-03-02: pol2cart(pi/4,1) yields cos/sin(pi/4)."""
    s.eval("[x,y] = pol2cart(pi/4, 1)")
    x = _ws(s, "x")
    y = _ws(s, "y")
    assert x == pytest.approx(math.cos(math.pi / 4))
    assert y == pytest.approx(math.sin(math.pi / 4))

def test_cart2sph(s):
    """R-POL19-03-03: cart2sph(1,1,1) yields correct az, el, r."""
    s.eval("[az,el,r] = cart2sph(1,1,1)")
    az = _ws(s, "az")
    el = _ws(s, "el")
    r = _ws(s, "r")
    assert r == pytest.approx(math.sqrt(3))
    assert az == pytest.approx(math.pi / 4)
    assert el == pytest.approx(math.atan2(1, math.sqrt(2)))

def test_sph2cart(s):
    """R-POL19-03-04: sph2cart(pi/4,0,1) yields correct x,y,z."""
    s.eval("[x,y,z] = sph2cart(pi/4, 0, 1)")
    x = _ws(s, "x")
    y = _ws(s, "y")
    z = _ws(s, "z")
    assert x == pytest.approx(math.cos(math.pi / 4))
    assert y == pytest.approx(math.sin(math.pi / 4))
    assert z == pytest.approx(0.0, abs=1e-10)

def test_polyarea_triangle(s):
    """R-POL19-02-07: Area of right triangle with legs 3,4 = 6."""
    r = s.eval("polyarea([0 3 0], [0 0 4])")
    assert float(r) == pytest.approx(6.0)
