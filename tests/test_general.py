"""V&V tests for general toolbox (31 functions).

SRS trace: SRS-FUNC-001, SRS-VAL-001
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar
from forge.engine.builtins.general import *


class TestCoordinateTransforms:
    """Verify coordinate conversions are round-trip consistent."""

    def test_cart2pol_roundtrip(self):
        x, y = ForgeArray(3.0), ForgeArray(4.0)
        theta, r = forge_cart2pol(x, y)
        x2, y2 = forge_pol2cart(theta, r)
        assert abs(_unwrap(x2) - 3.0) < 1e-14
        assert abs(_unwrap(y2) - 4.0) < 1e-14

    def test_cart2pol_known(self):
        theta, r = forge_cart2pol(ForgeArray(1.0), ForgeArray(1.0))
        assert abs(_unwrap(r) - np.sqrt(2)) < 1e-14
        assert abs(_unwrap(theta) - np.pi/4) < 1e-14

    def test_cart2sph_roundtrip(self):
        x, y, z = ForgeArray(1.0), ForgeArray(2.0), ForgeArray(3.0)
        az, el, r = forge_cart2sph(x, y, z)
        x2, y2, z2 = forge_sph2cart(az, el, r)
        assert abs(_unwrap(x2) - 1.0) < 1e-14
        assert abs(_unwrap(y2) - 2.0) < 1e-14
        assert abs(_unwrap(z2) - 3.0) < 1e-14

    def test_pol2cart_vectorized(self):
        theta = ForgeArray(np.array([0, np.pi/2, np.pi]))
        r = ForgeArray(np.array([1.0, 1.0, 1.0]))
        x, y = forge_pol2cart(theta, r)
        np.testing.assert_allclose(_unwrap(x).ravel(), [1, 0, -1], atol=1e-14)
        np.testing.assert_allclose(_unwrap(y).ravel(), [0, 1, 0], atol=1e-14)


class TestArrayManipulation:

    def test_circshift_right(self):
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r = _unwrap(forge_circshift(x, ForgeArray(2.0)))
        np.testing.assert_array_equal(r.ravel(), [4, 5, 1, 2, 3])

    def test_circshift_left(self):
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r = _unwrap(forge_circshift(x, ForgeArray(-1.0)))
        np.testing.assert_array_equal(r.ravel(), [2, 3, 4, 5, 1])

    def test_flip_vector(self):
        x = ForgeArray(np.array([1.0, 2.0, 3.0]))
        r = _unwrap(forge_flip(x))
        np.testing.assert_array_equal(r.ravel(), [3, 2, 1])

    def test_flip_dim(self):
        x = ForgeArray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        r = _unwrap(forge_flip(x, ForgeArray(2.0)))
        np.testing.assert_array_equal(r, np.array([[2, 1], [4, 3]]))

    def test_sortrows(self):
        x = ForgeArray(np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]]))
        r = _unwrap(forge_sortrows(x))
        assert r[0, 0] == 1.0
        assert r[2, 0] == 3.0

    def test_repelem_scalar(self):
        x = ForgeArray(np.array([1.0, 2.0, 3.0]))
        r = _unwrap(forge_repelem(x, ForgeArray(2.0)))
        np.testing.assert_array_equal(r.ravel(), [1, 1, 2, 2, 3, 3])

    def test_postpad(self):
        x = ForgeArray(np.array([1.0, 2.0]))
        r = _unwrap(forge_postpad(x, ForgeArray(5.0)))
        np.testing.assert_array_equal(r.ravel(), [1, 2, 0, 0, 0])

    def test_prepad(self):
        x = ForgeArray(np.array([1.0, 2.0]))
        r = _unwrap(forge_prepad(x, ForgeArray(5.0)))
        np.testing.assert_array_equal(r.ravel(), [0, 0, 0, 1, 2])

    def test_postpad_truncate(self):
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r = _unwrap(forge_postpad(x, ForgeArray(3.0)))
        np.testing.assert_array_equal(r.ravel(), [1, 2, 3])

    def test_rescale_default(self):
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r = _unwrap(forge_rescale(x))
        assert abs(r.flat[0]) < 1e-15  # min -> 0
        assert abs(r.flat[-1] - 1.0) < 1e-15  # max -> 1

    def test_rescale_custom_range(self):
        x = ForgeArray(np.array([0.0, 5.0, 10.0]))
        r = _unwrap(forge_rescale(x, ForgeArray(-1.0), ForgeArray(1.0))).ravel()
        np.testing.assert_allclose(r, [-1, 0, 1], atol=1e-14)


class TestNumericalCalculus:

    def test_trapz_uniform(self):
        """Integral of x from 0 to 1 = 0.5."""
        x = ForgeArray(np.linspace(0, 1, 1001))
        y = x  # f(x) = x
        r = _unwrap(forge_trapz(y, x))
        assert abs(r - 0.5) < 1e-6

    def test_cumtrapz_length(self):
        y = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0]))
        r = _unwrap(forge_cumtrapz(y))
        assert r.size == 3

    def test_gradient_linear(self):
        """Gradient of linear function is constant."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r = _unwrap(forge_gradient(x))
        np.testing.assert_allclose(r.ravel(), 1.0, atol=1e-14)

    def test_gradient_with_spacing(self):
        x = ForgeArray(np.array([0.0, 1.0, 4.0, 9.0, 16.0]))
        r = _unwrap(forge_gradient(x, ForgeArray(1.0)))
        # Central differences of x^2 at x=0,1,2,3,4
        assert abs(r.flat[2] - 4.0) < 1e-10  # 2*2


class TestInterpolation:

    def test_interp1_linear(self):
        x = ForgeArray(np.array([0.0, 1.0, 2.0, 3.0]))
        y = ForgeArray(np.array([0.0, 1.0, 4.0, 9.0]))
        xq = ForgeArray(np.array([0.5, 1.5, 2.5]))
        r = _unwrap(forge_interp1(x, y, xq))
        assert abs(r.flat[0] - 0.5) < 1e-14  # Linear interp
        assert abs(r.flat[1] - 2.5) < 1e-14


class TestMathUtilities:

    def test_deg2rad(self):
        r = _unwrap(forge_deg2rad(ForgeArray(180.0)))
        assert abs(r - np.pi) < 1e-14

    def test_rad2deg(self):
        r = _unwrap(forge_rad2deg(ForgeArray(np.pi)))
        assert abs(r - 180.0) < 1e-12

    def test_nextpow2(self):
        r = _unwrap(forge_nextpow2(ForgeArray(100.0)))
        assert int(r.flat[0]) == 7  # 2^7 = 128

    def test_nextpow2_exact(self):
        r = _unwrap(forge_nextpow2(ForgeArray(64.0)))
        assert int(r.flat[0]) == 6

    def test_bincoeff(self):
        r = _unwrap(forge_bincoeff(ForgeArray(10.0), ForgeArray(3.0)))
        assert abs(r - 120.0) < 1e-10

    def test_idivide_fix(self):
        r = _unwrap(forge_idivide(ForgeArray(7.0), ForgeArray(2.0)))
        assert int(r.flat[0]) == 3

    def test_idivide_floor(self):
        r = _unwrap(forge_idivide(ForgeArray(-7.0), ForgeArray(2.0), ForgeChar("floor")))
        assert int(r.flat[0]) == -4

    def test_xor(self):
        a = ForgeArray(np.array([True, True, False, False]))
        b = ForgeArray(np.array([True, False, True, False]))
        r = _unwrap(forge_xor(a, b)).ravel()
        np.testing.assert_array_equal(r, [False, True, True, False])


class TestComparison:

    def test_isequal_same(self):
        a = ForgeArray(np.array([1.0, 2.0, 3.0]))
        assert _unwrap(forge_isequal(a, a)) == True

    def test_isequal_different(self):
        a = ForgeArray(np.array([1.0, 2.0]))
        b = ForgeArray(np.array([1.0, 3.0]))
        assert _unwrap(forge_isequal(a, b)) == False

    def test_isequaln_nan(self):
        a = ForgeArray(np.array([1.0, np.nan, 3.0]))
        b = ForgeArray(np.array([1.0, np.nan, 3.0]))
        assert _unwrap(forge_isequaln(a, b)) == True

    def test_isequaln_nan_different(self):
        a = ForgeArray(np.array([1.0, np.nan]))
        b = ForgeArray(np.array([1.0, 2.0]))
        assert _unwrap(forge_isequaln(a, b)) == False


class TestMisc:

    def test_polyarea_square(self):
        """Unit square area = 1."""
        x = ForgeArray(np.array([0.0, 1.0, 1.0, 0.0]))
        y = ForgeArray(np.array([0.0, 0.0, 1.0, 1.0]))
        r = _unwrap(forge_polyarea(x, y))
        assert abs(r - 1.0) < 1e-14

    def test_polyarea_triangle(self):
        x = ForgeArray(np.array([0.0, 1.0, 0.0]))
        y = ForgeArray(np.array([0.0, 0.0, 1.0]))
        r = _unwrap(forge_polyarea(x, y))
        assert abs(r - 0.5) < 1e-14

    def test_accumarray(self):
        subs = ForgeArray(np.array([1, 1, 2, 3, 3]))
        vals = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r = _unwrap(forge_accumarray(subs, vals)).ravel()
        np.testing.assert_array_equal(r, [3, 3, 9])

    def test_logspace(self):
        r = _unwrap(forge_logspace(ForgeArray(0.0), ForgeArray(2.0), ForgeArray(3.0))).ravel()
        np.testing.assert_allclose(r, [1, 10, 100], rtol=1e-14)

    def test_deal_single(self):
        r = forge_deal(ForgeArray(5.0))
        assert _unwrap(r) == 5.0

    def test_int2str(self):
        r = forge_int2str(ForgeArray(42.0))
        assert r.to_str() == "42"
