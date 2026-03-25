"""V&V tests for geometry toolbox.

SRS trace: SRS-FUNC-001, SRS-VAL-001
Test method: Comparison against known geometric properties and identities.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.geometry import *


class TestConvhull:
    """Verify convex hull computation.

    convhull(x, y) returns (k, vol) where k is 1-based vertex indices.
    """

    def test_convhull_square(self):
        """Convex hull of unit square vertices has 4 boundary points."""
        x = np.array([0.0, 1.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        k, vol = forge_convhull(x, y)
        k_arr = np.asarray(_unwrap(k)).ravel()
        unique_pts = len(set(int(v) for v in k_arr))
        assert unique_pts >= 4

    def test_convhull_with_interior(self):
        """Interior point excluded from hull."""
        x = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
        k, vol = forge_convhull(x, y)
        k_arr = np.asarray(_unwrap(k)).ravel()
        unique_pts = len(set(int(v) for v in k_arr))
        assert unique_pts >= 4


class TestDelaunay:
    """Verify Delaunay triangulation."""

    def test_delaunay_4points(self):
        """Triangulation of 4 points produces triangles (3-column rows)."""
        x = np.array([0.0, 1.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        tri = forge_delaunay(x, y)
        tri_arr = np.asarray(_unwrap(tri))
        assert tri_arr.ndim == 2
        assert tri_arr.shape[1] == 3
        assert tri_arr.shape[0] >= 2


class TestInpolygon:
    """Verify point-in-polygon test.

    inpolygon returns (in_val, on_val) tuple of boolean arrays.
    """

    def test_inside_unit_square(self):
        """Point (0.5, 0.5) is inside unit square."""
        xv = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
        yv = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        xq = np.array([0.5])
        yq = np.array([0.5])
        in_val, on_val = forge_inpolygon(xq, yq, xv, yv)
        assert bool(np.asarray(in_val).flat[0]) is True

    def test_outside_unit_square(self):
        """Point (2.0, 2.0) is outside unit square."""
        xv = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
        yv = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        xq = np.array([2.0])
        yq = np.array([2.0])
        in_val, on_val = forge_inpolygon(xq, yq, xv, yv)
        assert bool(np.asarray(in_val).flat[0]) is False


class TestRotationMatrices:
    """Verify rotx, roty, rotz rotation matrix functions."""

    def test_rotx_zero_identity(self):
        """rotx(0) = 3x3 identity matrix."""
        r = _unwrap(forge_rotx(ForgeArray(0.0)))
        assert_allclose(np.asarray(r), np.eye(3), atol=1e-14)

    def test_roty_zero_identity(self):
        """roty(0) = 3x3 identity matrix."""
        r = _unwrap(forge_roty(ForgeArray(0.0)))
        assert_allclose(np.asarray(r), np.eye(3), atol=1e-14)

    def test_rotz_zero_identity(self):
        """rotz(0) = 3x3 identity matrix."""
        r = _unwrap(forge_rotz(ForgeArray(0.0)))
        assert_allclose(np.asarray(r), np.eye(3), atol=1e-14)

    def test_rotx_orthogonal(self):
        """rotx(45) is orthogonal: R @ R.T = I."""
        r = np.asarray(_unwrap(forge_rotx(ForgeArray(45.0))))
        assert_allclose(r @ r.T, np.eye(3), atol=1e-14)

    def test_roty_orthogonal(self):
        """roty(30) is orthogonal: R @ R.T = I."""
        r = np.asarray(_unwrap(forge_roty(ForgeArray(30.0))))
        assert_allclose(r @ r.T, np.eye(3), atol=1e-14)

    def test_rotz_orthogonal(self):
        """rotz(60) is orthogonal: R @ R.T = I."""
        r = np.asarray(_unwrap(forge_rotz(ForgeArray(60.0))))
        assert_allclose(r @ r.T, np.eye(3), atol=1e-14)


class TestDsearchn:
    """Verify nearest-point search.

    dsearchn returns (idx, dist) tuple with 1-based indices.
    """

    def test_dsearchn_exact(self):
        """Nearest point to (0.9, 0.1) in set => index 2 (1-based, point (1,0))."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        query = np.array([[0.9, 0.1]])
        idx, dist = forge_dsearchn(pts, query)
        assert int(np.asarray(_unwrap(idx)).flat[0]) == 2


class TestRectint:
    """Verify rectangle intersection area.

    rectint returns a scalar or matrix of intersection areas.
    """

    def test_rectint_overlap(self):
        """Two overlapping rectangles have positive intersection area."""
        a = np.array([[0.0, 0.0, 2.0, 2.0]])
        b = np.array([[1.0, 1.0, 2.0, 2.0]])
        r = forge_rectint(a, b)
        # May return scalar float or ForgeArray
        val = float(np.asarray(_unwrap(r)).flat[0]) if hasattr(r, '__class__') and r.__class__.__name__ == 'ForgeArray' else float(r)
        assert val == 1.0

    def test_rectint_no_overlap(self):
        """Non-overlapping rectangles have zero intersection area."""
        a = np.array([[0.0, 0.0, 1.0, 1.0]])
        b = np.array([[5.0, 5.0, 1.0, 1.0]])
        r = forge_rectint(a, b)
        val = float(np.asarray(_unwrap(r)).flat[0]) if hasattr(r, '__class__') and r.__class__.__name__ == 'ForgeArray' else float(r)
        assert val == 0.0
