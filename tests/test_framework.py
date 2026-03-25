"""Tests for V&V framework."""
import numpy as np
import pytest
from forge.validation.framework import (
    arrays_close, assert_close, assert_exact, assert_identity,
    assert_property, ToleranceSpec, DEFAULT_TOL, LOOSE_TOL, EXACT_TOL,
    VVReport,
)


class TestArraysClose:
    def test_identical(self):
        ok, msg = arrays_close([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert ok

    def test_within_abs_tolerance(self):
        ok, _ = arrays_close([1.0], [1.0 + 1e-15])
        assert ok

    def test_outside_tolerance(self):
        ok, _ = arrays_close([1.0], [2.0])
        assert not ok

    def test_nan_match(self):
        ok, _ = arrays_close([float("nan")], [float("nan")])
        assert ok

    def test_nan_mismatch(self):
        ok, _ = arrays_close([float("nan")], [1.0])
        assert not ok

    def test_inf_match(self):
        ok, _ = arrays_close([float("inf")], [float("inf")])
        assert ok

    def test_inf_sign_mismatch(self):
        ok, _ = arrays_close([float("inf")], [float("-inf")])
        assert not ok

    def test_shape_mismatch(self):
        ok, msg = arrays_close([1.0, 2.0], [1.0])
        assert not ok
        assert "Shape" in msg

    def test_relative_tolerance(self):
        tol = ToleranceSpec(abs_tol=0, rel_tol=0.01)
        ok, _ = arrays_close([100.0], [100.5], tol)
        assert ok
        ok, _ = arrays_close([100.0], [102.0], tol)
        assert not ok

    def test_ulp_tolerance(self):
        tol = ToleranceSpec(ulp_tol=2)
        a = 1.0
        b = a + 2 * np.spacing(a)
        ok, _ = arrays_close([a], [b], tol)
        assert ok

    def test_mixed_nan_inf_finite(self):
        a = [1.0, float("nan"), float("inf"), -float("inf"), 0.0]
        b = [1.0, float("nan"), float("inf"), -float("inf"), 0.0]
        ok, _ = arrays_close(a, b)
        assert ok

    def test_zero_vs_zero(self):
        ok, _ = arrays_close([0.0], [0.0])
        assert ok

    def test_loose_tolerance(self):
        ok, _ = arrays_close([1.0], [1.00005], LOOSE_TOL)
        assert ok


class TestAssertClose:
    def test_passes(self):
        assert_close([1.0, 2.0], [1.0, 2.0])

    def test_fails_with_message(self):
        with pytest.raises(AssertionError, match="custom msg"):
            assert_close([1.0], [2.0], msg="custom msg")


class TestAssertExact:
    def test_integer_exact(self):
        assert_exact([1, 2, 3], [1, 2, 3])

    def test_fails_on_diff(self):
        with pytest.raises(AssertionError):
            assert_exact([1, 2], [1, 3])

    def test_shape_mismatch(self):
        with pytest.raises(AssertionError, match="Shape"):
            assert_exact([1, 2], [1, 2, 3])


class TestAssertIdentity:
    def test_sin_asin(self):
        x = np.array([0.0, 0.5, -0.5])
        assert_identity(np.sin, np.arcsin, x, LOOSE_TOL)

    def test_exp_log(self):
        x = np.array([1.0, 2.0, 10.0])
        assert_identity(np.exp, np.log, x, LOOSE_TOL)


class TestAssertProperty:
    def test_passes(self):
        assert_property(True, "should pass")

    def test_fails(self):
        with pytest.raises(AssertionError, match="nope"):
            assert_property(False, "nope")


class TestVVReport:
    def test_empty_report(self):
        r = VVReport("test_tb")
        assert r.total == 0
        assert r.passed == 0

    def test_record_and_summary(self):
        r = VVReport("elfun")
        r.record("sind", "sind(30)=0.5", True)
        r.record("sind", "sind(90)=1", True)
        r.record("cosd", "cosd(90)=0", True)
        r.record("cosd", "cosd(NaN)", False, "returned 0 instead of NaN")
        s = r.summary()
        assert s["total"] == 4
        assert s["passed"] == 3
        assert s["failed"] == 1
        assert len(s["failures"]) == 1
        assert s["failures"][0]["function"] == "cosd"
