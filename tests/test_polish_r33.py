# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish round 33 -- error message quality and edge case hardening.

SRS trace: SRS-FUNC-001 (Octave-compatible function library)
           SRS-ERR-001 (Clear, actionable error messages)
"""
import math
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture(scope="module")
def S():
    return ForgeSession()


def _val(session, expr):
    """Evaluate and return numpy array from workspace ans."""
    r = session.eval(expr)
    ws = session.workspace
    if isinstance(r, str):
        ans = ws.get("ans") if ws.has("ans") else None
        if ans is not None:
            return np.asarray(_unwrap(ans)).ravel()
        return r
    return np.asarray(_unwrap(r)).ravel() if isinstance(r, ForgeArray) else r


def _scalar(session, expr):
    """Evaluate and return a scalar float."""
    r = session.eval(expr)
    ws = session.workspace
    if isinstance(r, str):
        ans = ws.get("ans") if ws.has("ans") else None
        if ans is not None:
            return float(np.asarray(_unwrap(ans)).ravel()[0])
        return float(r.strip())
    return float(np.asarray(_unwrap(r)).ravel()[0])


def _raw(session, varname, expr):
    """Evaluate expr assigning to varname, return raw workspace object."""
    session.eval(expr)
    ws = session.workspace
    return ws.get(varname) if ws.has(varname) else None


def _err(session, expr):
    """Evaluate and return the error string, or empty if no error."""
    r = session.eval(expr)
    if isinstance(r, str) and r.startswith("error:"):
        return r
    return ""


# -- 1. Graceful error messages -----------------------------------------------

class TestErrorMessages:
    """Verify that common user mistakes produce clear, actionable error messages."""

    def test_matrix_row_dimension_mismatch(self, S):
        """[1 2; 3 4 5] should report dimension mismatch."""
        r = _err(S, "A_t1 = [1 2; 3 4 5]")
        assert r, "Expected an error for mismatched matrix rows"
        assert "dimension" in r.lower() and "mismatch" in r.lower(), f"Unclear error: {r}"

    def test_inner_matrix_dimensions(self, S):
        """A * B with incompatible inner dimensions should say so clearly."""
        S.eval("M1_t = [1 2; 3 4]")
        r = _err(S, "M1_t * [1 2 3]")
        assert r, "Expected an error for inner dimension mismatch"
        assert "inner matrix dimensions" in r.lower(), f"Unclear error: {r}"

    def test_cell_indexing_on_non_cell(self, S):
        """Using {} on a numeric array should mention cell indexing."""
        S.eval("NC_t = [1 2 3]")
        r = _err(S, "NC_t{1}")
        assert r, "Expected an error for cell indexing on non-cell"
        assert "cell" in r.lower(), f"Unclear error: {r}"

    def test_undefined_function(self, S):
        """Calling an undefined function should report it clearly."""
        r = _err(S, "xyzzy_nope(1,2,3)")
        assert r, "Expected error for undefined function"
        assert "undefined" in r.lower(), f"Unclear error: {r}"

    def test_too_many_indices(self, S):
        """Indexing a scalar with too many subscripts should give clear error."""
        S.eval("S5_t = 5")
        r = _err(S, "S5_t(1,2,3,4)")
        assert r, "Expected error for too many indices"
        assert "index" in r.lower(), f"Unclear error: {r}"


# -- 2. Edge cases that must not crash ----------------------------------------

class TestEdgeCases:
    """Verify that empty/degenerate inputs do not crash."""

    def test_empty_matrix_no_crash(self, S):
        """[] should produce result without crashing."""
        r = S.eval("[]")
        assert "error" not in r.lower()

    def test_empty_cell_no_crash(self, S):
        """{} should produce result without crashing."""
        r = S.eval("{}")
        assert "error" not in r.lower()

    def test_empty_string_no_crash(self, S):
        """Empty char should not crash."""
        r = S.eval("x_es = char([])")
        assert "error" not in str(r).lower()

    def test_zeros_empty_no_crash(self, S):
        """zeros(0,3) should produce empty result without crashing."""
        r = S.eval("zeros(0,3)")
        assert "error" not in str(r).lower()

    def test_size_empty(self, S):
        """size([]) should return [0 0]."""
        v = _val(S, "size([])")
        np.testing.assert_array_equal(v[:2], [0, 0])

    def test_length_empty(self, S):
        """length([]) should return 0."""
        assert _scalar(S, "length([])") == 0

    def test_isempty_true(self, S):
        """isempty([]) should return 1 (numeric)."""
        r = _scalar(S, "isempty([])")
        assert r == 1.0

    def test_isempty_false(self, S):
        """isempty([1]) should return 0 (numeric)."""
        r = _scalar(S, "isempty([1])")
        assert r == 0.0

    def test_sum_empty(self, S):
        """sum([]) should return 0."""
        assert _scalar(S, "sum([])") == 0

    def test_prod_empty(self, S):
        """prod([]) should return 1."""
        assert _scalar(S, "prod([])") == 1

    def test_max_empty_no_crash(self, S):
        """max([]) should return empty, not crash."""
        r = S.eval("max([])")
        assert "error" not in str(r).lower()


# -- 3. Numeric edge cases ----------------------------------------------------

class TestNumericEdgeCases:
    """Verify IEEE 754 behavior for division by zero, NaN, Inf."""

    def test_div_by_zero_positive(self, S):
        """1/0 should produce Inf."""
        assert _scalar(S, "1/0") == float("inf")

    def test_div_by_zero_negative(self, S):
        """-1/0 should produce -Inf."""
        assert _scalar(S, "-1/0") == float("-inf")

    def test_zero_div_zero(self, S):
        """0/0 should produce NaN."""
        assert math.isnan(_scalar(S, "0/0"))

    def test_isnan_of_nan(self, S):
        """isnan(NaN) should return 1."""
        assert _scalar(S, "isnan(NaN)") == 1.0

    def test_isinf_of_inf(self, S):
        """isinf(Inf) should return 1."""
        assert _scalar(S, "isinf(Inf)") == 1.0

    def test_isfinite_of_number(self, S):
        """isfinite(42) should return 1."""
        assert _scalar(S, "isfinite(42)") == 1.0

    def test_isfinite_of_inf(self, S):
        """isfinite(Inf) should return 0."""
        assert _scalar(S, "isfinite(Inf)") == 0.0

    def test_isnan_returns_numeric(self, S):
        """isnan should return numeric (double), not boolean."""
        r = _raw(S, "nr_t", "nr_t = isnan(NaN)")
        arr = np.asarray(_unwrap(r))
        assert arr.dtype != np.bool_, "isnan should return numeric, not bool"

    def test_eps_value(self, S):
        """eps should be machine epsilon (~2.2204e-16)."""
        val = _scalar(S, "eps")
        assert abs(val - 2.220446049250313e-16) < 1e-30


# -- 4. Complex number support ------------------------------------------------

class TestComplexNumbers:
    """Verify basic complex number operations."""

    def test_complex_literal(self, S):
        """3 + 4i should create a complex number."""
        r = _raw(S, "zzc", "zzc = 3 + 4i")
        arr = np.asarray(_unwrap(r))
        assert np.issubdtype(arr.dtype, np.complexfloating)
        assert arr.ravel()[0] == 3 + 4j

    def test_abs_complex(self, S):
        """abs(3+4i) should return 5."""
        S.eval("zzc = 3 + 4i")
        assert _scalar(S, "abs(zzc)") == pytest.approx(5.0)

    def test_real_part(self, S):
        """real(3+4i) should return 3."""
        assert _scalar(S, "real(zzc)") == pytest.approx(3.0)

    def test_imag_part(self, S):
        """imag(3+4i) should return 4."""
        assert _scalar(S, "imag(zzc)") == pytest.approx(4.0)

    def test_conjugate(self, S):
        """conj(3+4i) should return 3-4i."""
        r = _raw(S, "cc_t", "cc_t = conj(zzc)")
        arr = np.asarray(_unwrap(r))
        assert arr.ravel()[0] == pytest.approx(3 - 4j)

    def test_angle(self, S):
        """angle(3+4i) should return atan2(4,3)."""
        val = _scalar(S, "angle(zzc)")
        assert val == pytest.approx(math.atan2(4, 3))
