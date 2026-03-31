"""Round-19 polish: interpolation, computational geometry, coordinate transforms."""
import pytest, math
from forge.engine.session import ForgeSession

@pytest.fixture(scope="module")
def s():
    return ForgeSession()

def _ws(s, name):
    """Get raw workspace variable at full precision."""
    return float(s.get_workspace_dict()[name])

# ── Interpolation ───────────────────────────────────────────────────

def test_interp1_linear(s):
    r = s.eval("interp1([1 2 3 4], [10 20 30 40], 2.5)")
    assert float(r) == pytest.approx(25.0)

def test_interp1_spline(s):
    r = s.eval("interp1([1 2 3 4], [10 20 30 40], 2.5, 'spline')")
    assert float(r) == pytest.approx(25.0, abs=1.0)

def test_interp1_nearest(s):
    r = s.eval("interp1([1 2 3 4], [10 20 30 40], 2.3, 'nearest')")
    assert float(r) == pytest.approx(20.0)

def test_interp1_extrap(s):
    r = s.eval("interp1([1 2 3 4], [10 20 30 40], 5, 'linear', 'extrap')")
    assert float(r) == pytest.approx(50.0)

def test_interp2_meshgrid(s):
    r = s.eval("[X,Y] = meshgrid(1:3, 1:3); V = X + Y; interp2(X, Y, V, 1.5, 1.5)")
    assert float(r) == pytest.approx(3.0)

def test_griddata_basic(s):
    r = s.eval("griddata([0 1 1 0], [0 0 1 1], [1 2 3 4], 0.5, 0.5)")
    assert float(r) == pytest.approx(3.0, abs=1.0)

# ── Computational geometry ──────────────────────────────────────────

def test_convhull_square(s):
    """Interior point (0.5,0.5) excluded from hull."""
    r = s.eval("convhull([0 1 1 0 0.5], [0 0 1 1 0.5])")
    # Result is a formatted string like "    1    2    3    4    1"
    nums = [int(x) for x in str(r).split()]
    assert len(nums) == 5
    # vertex 5 (the interior point) should NOT appear
    assert 5 not in nums

def test_delaunay_square(s):
    """4-point square yields 2 triangles."""
    r = s.eval("delaunay([0 1 1 0], [0 0 1 1])")
    # Result is multi-line string; each line is a triangle
    lines = [l.strip() for l in str(r).strip().split('\n') if l.strip()]
    assert len(lines) == 2
    for line in lines:
        verts = [int(x) for x in line.split()]
        assert len(verts) == 3

def test_inpolygon_inside(s):
    r = s.eval("inpolygon(0.5, 0.5, [0 1 1 0], [0 0 1 1])")
    assert float(r) == 1

def test_inpolygon_outside(s):
    r = s.eval("inpolygon(2, 2, [0 1 1 0], [0 0 1 1])")
    assert float(r) == 0

def test_inpolygon_edge(s):
    r = s.eval("inpolygon(0, 0.5, [0 1 1 0], [0 0 1 1])")
    assert float(r) == 1

def test_polyarea_unit_square(s):
    r = s.eval("polyarea([0 1 1 0], [0 0 1 1])")
    assert float(r) == pytest.approx(1.0)

# ── Coordinate transforms ──────────────────────────────────────────

def test_cart2pol(s):
    s.eval("[th,r] = cart2pol(3,4)")
    th = _ws(s, "th")
    r = _ws(s, "r")
    assert r == pytest.approx(5.0)
    assert th == pytest.approx(math.atan2(4, 3))

def test_pol2cart(s):
    s.eval("[x,y] = pol2cart(pi/4, 1)")
    x = _ws(s, "x")
    y = _ws(s, "y")
    assert x == pytest.approx(math.cos(math.pi / 4))
    assert y == pytest.approx(math.sin(math.pi / 4))

def test_cart2sph(s):
    s.eval("[az,el,r] = cart2sph(1,1,1)")
    az = _ws(s, "az")
    el = _ws(s, "el")
    r = _ws(s, "r")
    assert r == pytest.approx(math.sqrt(3))
    assert az == pytest.approx(math.pi / 4)
    assert el == pytest.approx(math.atan2(1, math.sqrt(2)))

def test_sph2cart(s):
    s.eval("[x,y,z] = sph2cart(pi/4, 0, 1)")
    x = _ws(s, "x")
    y = _ws(s, "y")
    z = _ws(s, "z")
    assert x == pytest.approx(math.cos(math.pi / 4))
    assert y == pytest.approx(math.sin(math.pi / 4))
    assert z == pytest.approx(0.0, abs=1e-10)

def test_polyarea_triangle(s):
    """Area of right triangle with legs 3,4 = 6."""
    r = s.eval("polyarea([0 3 0], [0 0 4])")
    assert float(r) == pytest.approx(6.0)
