# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for V&V framework.

Requirement R-FW: The validation framework SHALL provide tolerance-aware
array comparison, assertion helpers, identity verification, property
checks, and structured reporting, enabling systematic verification of
Forge's numerical functions against reference implementations.

Model-user argument: An engineer validating Forge's numerical accuracy
needs a framework that handles NaN-matching, relative/absolute/ULP
tolerances, shape checking, and structured pass/fail reporting. Without
this framework, each function validation would require ad-hoc comparison
logic, making the validation suite brittle and inconsistent. The
framework is the foundation that ensures every Forge function is tested
with the same rigor.

Decomposition:
  R-FW-01..14: arrays_close tolerance comparisons
  R-FW-15..16: assert_close pass/fail behavior
  R-FW-17..19: assert_exact integer-level comparison
  R-FW-20..21: assert_identity inverse-function verification
  R-FW-22..23: assert_property boolean checks
  R-FW-24: VVReport structured reporting

Consistency argument: arrays_close (R-FW-01..14) is the core comparison
engine; assert_close/assert_exact/assert_identity/assert_property
(R-FW-15..23) are convenience wrappers for different comparison modes;
VVReport (R-FW-24) aggregates results. Together they form a complete
validation toolkit: compare, assert, and report.
"""
import numpy as np
import pytest
from forge.validation.framework import (
    arrays_close, assert_close, assert_exact, assert_identity,
    assert_property, ToleranceSpec, DEFAULT_TOL, LOOSE_TOL, EXACT_TOL,
    VVReport,
)


class TestArraysClose:
    """R-FW-01..14: arrays_close SHALL correctly compare arrays under
    absolute, relative, and ULP tolerances, handling NaN, Inf, shape
    mismatches, and edge cases.

    Model-user argument: The engineer validates Forge functions by
    comparing output arrays against known-correct Octave results. The
    comparison must handle special values (NaN, Inf, -Inf) because many
    mathematical functions produce them at domain boundaries. ULP
    tolerance is needed for functions where relative error is meaningless
    near zero.

    Decomposition:
      R-FW-01: Identical arrays pass
      R-FW-02: Within absolute tolerance passes
      R-FW-03: Outside tolerance fails
      R-FW-04: NaN matches NaN
      R-FW-05: NaN does not match finite
      R-FW-06: Inf matches Inf
      R-FW-07: Inf does not match -Inf
      R-FW-08: Shape mismatch fails with message
      R-FW-09: Relative tolerance comparison
      R-FW-10: ULP tolerance comparison
      R-FW-11: Mixed NaN/Inf/finite array comparison
      R-FW-12: Zero versus zero passes
      R-FW-13: Loose tolerance preset works
      R-FW-14: (covered by the 13 tests above, no gap)

    Consistency: Tests R-FW-01..03 cover the basic pass/fail tolerance
    logic. R-FW-04..07 cover special-value semantics. R-FW-08 covers
    structural validation. R-FW-09..10 cover alternative tolerance modes.
    R-FW-11..13 cover composite and edge cases.
    """

    def test_identical(self):
        """R-FW-01: Identical arrays pass comparison."""
        ok, msg = arrays_close([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert ok

    def test_within_abs_tolerance(self):
        """R-FW-02: Arrays within absolute tolerance pass."""
        ok, _ = arrays_close([1.0], [1.0 + 1e-15])
        assert ok

    def test_outside_tolerance(self):
        """R-FW-03: Arrays outside tolerance fail."""
        ok, _ = arrays_close([1.0], [2.0])
        assert not ok

    def test_nan_match(self):
        """R-FW-04: NaN matches NaN."""
        ok, _ = arrays_close([float("nan")], [float("nan")])
        assert ok

    def test_nan_mismatch(self):
        """R-FW-05: NaN does not match a finite value."""
        ok, _ = arrays_close([float("nan")], [1.0])
        assert not ok

    def test_inf_match(self):
        """R-FW-06: Inf matches Inf."""
        ok, _ = arrays_close([float("inf")], [float("inf")])
        assert ok

    def test_inf_sign_mismatch(self):
        """R-FW-07: +Inf does not match -Inf."""
        ok, _ = arrays_close([float("inf")], [float("-inf")])
        assert not ok

    def test_shape_mismatch(self):
        """R-FW-08: Shape mismatch fails with descriptive message."""
        ok, msg = arrays_close([1.0, 2.0], [1.0])
        assert not ok
        assert "Shape" in msg

    def test_relative_tolerance(self):
        """R-FW-09: Relative tolerance comparison passes/fails correctly."""
        tol = ToleranceSpec(abs_tol=0, rel_tol=0.01)
        ok, _ = arrays_close([100.0], [100.5], tol)
        assert ok
        ok, _ = arrays_close([100.0], [102.0], tol)
        assert not ok

    def test_ulp_tolerance(self):
        """R-FW-10: ULP tolerance comparison accepts values within N ULPs."""
        tol = ToleranceSpec(ulp_tol=2)
        a = 1.0
        b = a + 2 * np.spacing(a)
        ok, _ = arrays_close([a], [b], tol)
        assert ok

    def test_mixed_nan_inf_finite(self):
        """R-FW-11: Mixed NaN/Inf/finite array compares element-wise."""
        a = [1.0, float("nan"), float("inf"), -float("inf"), 0.0]
        b = [1.0, float("nan"), float("inf"), -float("inf"), 0.0]
        ok, _ = arrays_close(a, b)
        assert ok

    def test_zero_vs_zero(self):
        """R-FW-12: Zero versus zero passes comparison."""
        ok, _ = arrays_close([0.0], [0.0])
        assert ok

    def test_loose_tolerance(self):
        """R-FW-13: LOOSE_TOL preset accepts slightly different values."""
        ok, _ = arrays_close([1.0], [1.00005], LOOSE_TOL)
        assert ok


class TestAssertClose:
    """R-FW-15..16: assert_close SHALL raise AssertionError with a custom
    message on failure and pass silently on success.

    Model-user argument: The engineer writes validation tests that need
    clear failure messages identifying which function and test case
    failed. Silent pass on success keeps test output clean.

    Decomposition:
      R-FW-15: Matching arrays pass without error
      R-FW-16: Mismatched arrays raise AssertionError with custom message

    Consistency: Pass and fail are the two possible outcomes.
    """

    def test_passes(self):
        """R-FW-15: Matching arrays pass assert_close."""
        assert_close([1.0, 2.0], [1.0, 2.0])

    def test_fails_with_message(self):
        """R-FW-16: Mismatched arrays raise AssertionError with custom message."""
        with pytest.raises(AssertionError, match="custom msg"):
            assert_close([1.0], [2.0], msg="custom msg")


class TestAssertExact:
    """R-FW-17..19: assert_exact SHALL require bitwise equality and fail
    on any difference or shape mismatch.

    Model-user argument: The engineer validates integer-valued functions
    (factorial, nchoosek, permutation indices) where any deviation from
    exact values indicates a bug, not a tolerance issue.

    Decomposition:
      R-FW-17: Identical integer arrays pass
      R-FW-18: Differing values raise AssertionError
      R-FW-19: Shape mismatch raises AssertionError with message

    Consistency: Pass, value-fail, and shape-fail cover the three outcomes.
    """

    def test_integer_exact(self):
        """R-FW-17: Identical integer arrays pass assert_exact."""
        assert_exact([1, 2, 3], [1, 2, 3])

    def test_fails_on_diff(self):
        """R-FW-18: Differing integer values raise AssertionError."""
        with pytest.raises(AssertionError):
            assert_exact([1, 2], [1, 3])

    def test_shape_mismatch(self):
        """R-FW-19: Shape mismatch raises AssertionError with message."""
        with pytest.raises(AssertionError, match="Shape"):
            assert_exact([1, 2], [1, 2, 3])


class TestAssertIdentity:
    """R-FW-20..21: assert_identity SHALL verify that f(g(x)) recovers x
    for inverse function pairs.

    Model-user argument: The engineer validates trigonometric and
    exponential function pairs (sin/asin, exp/log) by confirming the
    roundtrip identity. This is the standard mathematical verification
    approach for inverse functions.

    Decomposition:
      R-FW-20: sin/arcsin roundtrip recovers input
      R-FW-21: exp/log roundtrip recovers input

    Consistency: Trigonometric (R-FW-20) and exponential (R-FW-21)
    inverse pairs cover the two primary function families.
    """

    def test_sin_asin(self):
        """R-FW-20: sin/arcsin roundtrip recovers input values."""
        x = np.array([0.0, 0.5, -0.5])
        assert_identity(np.sin, np.arcsin, x, LOOSE_TOL)

    def test_exp_log(self):
        """R-FW-21: exp/log roundtrip recovers input values."""
        x = np.array([1.0, 2.0, 10.0])
        assert_identity(np.exp, np.log, x, LOOSE_TOL)


class TestAssertProperty:
    """R-FW-22..23: assert_property SHALL pass on True and raise
    AssertionError with message on False.

    Model-user argument: The engineer checks structural properties
    (e.g., "matrix is symmetric", "eigenvalues are positive") that
    reduce to a boolean. The assertion helper provides a uniform
    interface with descriptive failure messages.

    Decomposition:
      R-FW-22: True condition passes
      R-FW-23: False condition raises AssertionError with message

    Consistency: True and False are the two boolean outcomes.
    """

    def test_passes(self):
        """R-FW-22: True condition passes assert_property."""
        assert_property(True, "should pass")

    def test_fails(self):
        """R-FW-23: False condition raises AssertionError with message."""
        with pytest.raises(AssertionError, match="nope"):
            assert_property(False, "nope")


class TestVVReport:
    """R-FW-24: VVReport SHALL aggregate pass/fail records and produce
    a structured summary with total, passed, failed counts and failure
    details.

    Model-user argument: After validating hundreds of functions, the
    engineer needs a single summary showing how many passed, how many
    failed, and which specific test cases failed. VVReport provides
    this structured output for automated CI and manual review.

    Decomposition:
      R-FW-24a: Empty report has zero counts
      R-FW-24b: Recorded results produce correct summary with failure details

    Consistency: Empty (R-FW-24a) and populated (R-FW-24b) cover the
    two report states.
    """

    def test_empty_report(self):
        """R-FW-24a: Empty report has zero total and zero passed."""
        r = VVReport("test_tb")
        assert r.total == 0
        assert r.passed == 0

    def test_record_and_summary(self):
        """R-FW-24b: Recorded results produce correct summary with failure details."""
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
