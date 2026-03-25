"""V&V tests for polynomial toolbox.

SRS trace: SRS-FUNC-001, SRS-VAL-001
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.polynomial import *


class TestRootsAndPoly:

    def test_roots_quadratic(self):
        """x^2 - 5x + 6 = 0 => roots at 2, 3."""
        p = ForgeArray(np.array([1.0, -5.0, 6.0]))
        r = np.sort(_unwrap(forge_roots(p)).ravel().real)
        np.testing.assert_allclose(r, [2, 3], atol=1e-10)

    def test_poly_from_roots(self):
        """poly([2, 3]) => [1, -5, 6]."""
        x = ForgeArray(np.array([2.0, 3.0]))
        p = _unwrap(forge_poly(x)).ravel()
        np.testing.assert_allclose(p, [1, -5, 6], atol=1e-10)

    def test_roots_poly_roundtrip(self):
        """poly(roots(p)) == p (up to scaling)."""
        p = ForgeArray(np.array([1.0, -6.0, 11.0, -6.0]))
        r = forge_roots(p)
        p2 = _unwrap(forge_poly(r)).ravel().real
        np.testing.assert_allclose(p2 / p2[0], _unwrap(p).ravel(), atol=1e-10)


class TestPolyEval:

    def test_polyval_simple(self):
        """2x^2 + 3x + 1 at x=2 => 15."""
        p = ForgeArray(np.array([2.0, 3.0, 1.0]))
        r = float(_unwrap(forge_polyval(p, ForgeArray(2.0))).flat[0])
        assert abs(r - 15.0) < 1e-10

    def test_polyval_vectorized(self):
        p = ForgeArray(np.array([1.0, 0.0, -1.0]))  # x^2 - 1
        x = ForgeArray(np.array([0.0, 1.0, 2.0]))
        r = _unwrap(forge_polyval(p, x)).ravel()
        np.testing.assert_allclose(r, [-1, 0, 3], atol=1e-14)

    def test_polyfit_linear(self):
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        y = ForgeArray(np.array([2.0, 4.0, 6.0, 8.0, 10.0]))
        p = _unwrap(forge_polyfit(x, y, ForgeArray(1.0))).ravel()
        assert abs(p[0] - 2.0) < 1e-10  # slope
        assert abs(p[1]) < 1e-10  # intercept


class TestPolyCalculus:

    def test_polyder(self):
        """d/dx(x^3 + 2x) = 3x^2 + 2."""
        p = ForgeArray(np.array([1.0, 0.0, 2.0, 0.0]))
        dp = _unwrap(forge_polyder(p)).ravel()
        np.testing.assert_allclose(dp, [3, 0, 2], atol=1e-14)

    def test_polyint(self):
        """integral(3x^2 + 2) = x^3 + 2x + C."""
        p = ForgeArray(np.array([3.0, 0.0, 2.0]))
        ip = _unwrap(forge_polyint(p)).ravel()
        assert abs(ip[0] - 1.0) < 1e-14
        assert abs(ip[2] - 2.0) < 1e-14

    def test_polyder_polyint_roundtrip(self):
        p = ForgeArray(np.array([1.0, 2.0, 3.0]))
        ip = forge_polyint(p)
        dp = _unwrap(forge_polyder(ip)).ravel()
        np.testing.assert_allclose(dp, [1, 2, 3], atol=1e-14)


class TestConvolution:

    def test_conv(self):
        """(x+1)(x+2) = x^2 + 3x + 2."""
        a = ForgeArray(np.array([1.0, 1.0]))
        b = ForgeArray(np.array([1.0, 2.0]))
        r = _unwrap(forge_conv(a, b)).ravel()
        np.testing.assert_allclose(r, [1, 3, 2], atol=1e-14)

    def test_deconv(self):
        """(x^2 + 3x + 2) / (x + 1) = (x + 2)."""
        b = ForgeArray(np.array([1.0, 3.0, 2.0]))
        a = ForgeArray(np.array([1.0, 1.0]))
        q, rem = forge_deconv(b, a)
        np.testing.assert_allclose(_unwrap(q).ravel(), [1, 2], atol=1e-14)


class TestInterpolation:

    def test_pchip_identity(self):
        x = ForgeArray(np.array([0.0, 1.0, 2.0, 3.0]))
        y = ForgeArray(np.array([0.0, 1.0, 4.0, 9.0]))
        xq = ForgeArray(np.array([0.0, 1.0, 2.0, 3.0]))
        r = _unwrap(forge_pchip(x, y, xq)).ravel()
        np.testing.assert_allclose(r, [0, 1, 4, 9], atol=1e-10)

    def test_spline_interpolation(self):
        x = ForgeArray(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        y = ForgeArray(np.array([0.0, 1.0, 0.0, 1.0, 0.0]))
        xq = ForgeArray(np.array([0.5, 1.5, 2.5, 3.5]))
        r = _unwrap(forge_spline(x, y, xq)).ravel()
        # Just verify it returns sensible values
        assert all(abs(v) < 2.0 for v in r)


class TestPiecewisePolynomial:

    def test_mkpp_ppval(self):
        """Simple linear pp: f(x) = x on [0,1], f(x) = 2-x on [1,2]."""
        breaks = ForgeArray(np.array([0.0, 1.0, 2.0]))
        coefs = ForgeArray(np.array([[1.0, 0.0], [-1.0, 1.0]]))
        pp = forge_mkpp(breaks, coefs)
        x = ForgeArray(np.array([0.5, 1.0, 1.5]))
        r = _unwrap(forge_ppval(pp, x)).ravel()
        np.testing.assert_allclose(r, [0.5, 1.0, 0.5], atol=1e-14)

    def test_unmkpp(self):
        breaks = ForgeArray(np.array([0.0, 1.0, 2.0]))
        coefs = ForgeArray(np.array([[1.0, 0.0], [-1.0, 1.0]]))
        pp = forge_mkpp(breaks, coefs)
        b, c = forge_unmkpp(pp)
        np.testing.assert_array_equal(_unwrap(b).ravel(), [0, 1, 2])


class TestMisc:

    def test_compan(self):
        """Companion matrix of x^2 - 3x + 2."""
        p = ForgeArray(np.array([1.0, -3.0, 2.0]))
        C = _unwrap(forge_compan(p))
        # Eigenvalues of companion = roots of polynomial
        eigs = np.sort(np.linalg.eigvals(C).real)
        np.testing.assert_allclose(eigs, [1, 2], atol=1e-10)

    def test_polyreduce(self):
        p = ForgeArray(np.array([0.0, 0.0, 1.0, 2.0, 3.0]))
        r = _unwrap(forge_polyreduce(p)).ravel()
        np.testing.assert_allclose(r, [1, 2, 3], atol=1e-14)
