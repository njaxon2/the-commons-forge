# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V test framework with tolerance comparison and reference testing."""
import numpy as np


class ToleranceSpec:
    """Specify comparison tolerances for V&V."""
    def __init__(self, abs_tol=0.0, rel_tol=0.0, ulp_tol=0):
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.ulp_tol = ulp_tol

    def __repr__(self):
        return f"ToleranceSpec(abs={self.abs_tol}, rel={self.rel_tol}, ulp={self.ulp_tol})"


DEFAULT_TOL = ToleranceSpec(abs_tol=1e-12, rel_tol=1e-10)
LOOSE_TOL = ToleranceSpec(abs_tol=1e-6, rel_tol=1e-4)
EXACT_TOL = ToleranceSpec(abs_tol=0.0, rel_tol=0.0)


def arrays_close(a, b, tol=None):
    """Compare arrays with specified tolerance, handling NaN and Inf.

    Returns (ok, message) tuple.
    """
    if tol is None:
        tol = DEFAULT_TOL
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        return False, f"Shape mismatch: {a.shape} vs {b.shape}"

    # NaN positions must match
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return False, "NaN position mismatch"

    # Inf positions and signs must match
    inf_a, inf_b = np.isinf(a), np.isinf(b)
    if not np.array_equal(inf_a, inf_b):
        return False, "Inf position mismatch"
    if inf_a.any():
        if not np.array_equal(np.sign(a[inf_a]), np.sign(b[inf_b])):
            return False, "Inf sign mismatch"

    # Compare finite values
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return True, "OK (all non-finite, matched)"

    af, bf = a[finite], b[finite]
    diff = np.abs(af - bf)

    # ULP comparison
    if tol.ulp_tol > 0:
        spacing = np.maximum(np.spacing(af), np.spacing(bf))
        ulp_ok = diff <= tol.ulp_tol * spacing
        if ulp_ok.all():
            return True, "OK (within ULP tolerance)"

    # Absolute and relative tolerance
    abs_ok = diff <= tol.abs_tol
    scale = np.maximum(np.abs(af), np.abs(bf))
    # Avoid division issues when both values are zero
    rel_ok = np.where(scale > 0, diff <= tol.rel_tol * scale, True)

    combined = abs_ok | rel_ok
    if combined.all():
        return True, "OK"

    # Find worst failure for diagnostic
    failures = ~combined
    fail_diffs = diff[failures]
    worst_idx = np.argmax(fail_diffs)
    fail_indices = np.where(failures)[0]
    return False, (
        f"Max diff {fail_diffs[worst_idx]:.2e} at flat index {fail_indices[worst_idx]} "
        f"(values: {af[failures][worst_idx]:.6e} vs {bf[failures][worst_idx]:.6e}, "
        f"abs_tol={tol.abs_tol}, rel_tol={tol.rel_tol})"
    )


def assert_close(a, b, tol=None, msg=""):
    """Assert arrays are close within tolerance."""
    ok, detail = arrays_close(a, b, tol)
    if not ok:
        raise AssertionError(f"{msg}: {detail}" if msg else detail)


def assert_exact(a, b, msg=""):
    """Assert arrays are exactly equal (bitwise for floats, value for ints)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise AssertionError(f"{msg}: Shape mismatch {a.shape} vs {b.shape}")
    if not np.array_equal(a, b):
        diff_idx = np.argmax(a.ravel() != b.ravel())
        raise AssertionError(
            f"{msg}: First diff at index {diff_idx}: {a.ravel()[diff_idx]} vs {b.ravel()[diff_idx]}"
        )


def assert_identity(func, inverse_func, x, tol=None, msg=""):
    """Assert func(inverse_func(x)) == x (round-trip identity)."""
    result = func(inverse_func(x))
    assert_close(result, x, tol, msg or f"Identity failed for {func.__name__}({inverse_func.__name__}(x))")


def assert_property(condition, msg="Property not satisfied"):
    """Assert a mathematical property holds."""
    if not condition:
        raise AssertionError(msg)


class VVReport:
    """Collect V&V test results for reporting."""

    def __init__(self, toolbox_name):
        self.toolbox = toolbox_name
        self.results = []

    def record(self, function_name, test_name, passed, detail=""):
        self.results.append({
            "function": function_name,
            "test": test_name,
            "passed": passed,
            "detail": detail,
        })

    @property
    def total(self):
        return len(self.results)

    @property
    def passed(self):
        return sum(1 for r in self.results if r["passed"])

    @property
    def failed(self):
        return self.total - self.passed

    def summary(self):
        return {
            "toolbox": self.toolbox,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.passed / self.total if self.total > 0 else 0,
            "failures": [r for r in self.results if not r["passed"]],
        }
