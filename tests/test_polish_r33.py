# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish round 33 -- error message quality and edge case hardening.

SRS trace: SRS-FUNC-001 (Octave-compatible function library)
           SRS-ERR-001 (Clear, actionable error messages)

V&V Traceability (backfill)
===========================
R-POL33-01: Common user mistakes SHALL produce clear, actionable error
            messages that name the problem and suggest a fix.

    Model-user argument: When an engineer accidentally writes [1 2; 3 4 5],
    the error message must say "dimension mismatch," not a cryptic Python
    traceback. Clear errors save debugging time and build trust in the tool.

    Decomposition:
      R-POL33-01a: Mismatched matrix rows report "dimension mismatch."
      R-POL33-01b: Incompatible multiplication reports "inner matrix dimensions."
      R-POL33-01c: Cell indexing on non-cell mentions "cell."
      R-POL33-01d: Undefined function reports "undefined."
      R-POL33-01e: Too many subscript indices reports "index."

    Consistency: Five common mistake categories (01a-e) cover the primary
    error-message quality surface.

R-POL33-02: Empty and degenerate inputs SHALL not crash the engine.

    Model-user argument: Scientists exploring data interactively often
    create empty matrices, cells, or strings by accident. These must
    produce sensible results (empty output, zeros, etc.) instead of
    unhandled exceptions that terminate the session.

    Decomposition:
      R-POL33-02a: [] does not crash.
      R-POL33-02b: {} does not crash.
      R-POL33-02c: char([]) does not crash.
      R-POL33-02d: zeros(0,3) does not crash.
      R-POL33-02e: size([]) returns [0 0].
      R-POL33-02f: length([]) returns 0.
      R-POL33-02g: isempty([]) returns 1.
      R-POL33-02h: isempty([1]) returns 0.
      R-POL33-02i: sum([]) returns 0.
      R-POL33-02j: prod([]) returns 1.
      R-POL33-02k: max([]) does not crash.

    Consistency: Eleven edge cases (02a-k) cover empty matrix, cell, string,
    zeros, and aggregate functions on empty inputs.

R-POL33-03: IEEE 754 special values (Inf, NaN, eps) SHALL be handled
            correctly in arithmetic and type-checking functions.

    Model-user argument: Numerical algorithms produce Inf and NaN during
    convergence failures or singular operations. An engineer uses isnan(),
    isinf(), isfinite() to detect and handle these cases. Wrong IEEE 754
    behavior causes algorithms to loop forever or produce garbage.

    Decomposition:
      R-POL33-03a: 1/0 produces Inf.
      R-POL33-03b: -1/0 produces -Inf.
      R-POL33-03c: 0/0 produces NaN.
      R-POL33-03d: isnan(NaN) returns 1.
      R-POL33-03e: isinf(Inf) returns 1.
      R-POL33-03f: isfinite(42) returns 1.
      R-POL33-03g: isfinite(Inf) returns 0.
      R-POL33-03h: isnan returns numeric (not boolean).
      R-POL33-03i: eps equals machine epsilon (~2.22e-16).

    Consistency: Division edge cases (03a-c), type-check functions (03d-g),
    return type (03h), and constant value (03i) cover IEEE 754 support.

R-POL33-04: Complex number literals and functions (abs, real, imag, conj,
            angle) SHALL produce correct results.

    Model-user argument: Signal-processing engineers work with complex
    phasors daily. If 3+4i does not create a complex number or abs(3+4i)
    does not return 5, FFT-based workflows produce wrong magnitude and
    phase results.

    Decomposition:
      R-POL33-04a: 3 + 4i creates a complex number.
      R-POL33-04b: abs(3+4i) returns 5.
      R-POL33-04c: real(3+4i) returns 3.
      R-POL33-04d: imag(3+4i) returns 4.
      R-POL33-04e: conj(3+4i) returns 3-4i.
      R-POL33-04f: angle(3+4i) returns atan2(4,3).

    Consistency: Creation (04a) and five accessor/transform functions
    (04b-f) cover the complex number API.
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
    """R-POL33-01: Common user mistakes SHALL produce clear, actionable error
    messages that name the problem and suggest a fix.

    Model-user argument: When an engineer accidentally writes [1 2; 3 4 5],
    the error message must say "dimension mismatch," not a cryptic Python
    traceback. Clear errors save debugging time and build trust in the tool.

    Decomposition:
      R-POL33-01a: Mismatched matrix rows report "dimension mismatch."
      R-POL33-01b: Incompatible multiplication reports "inner matrix dimensions."
      R-POL33-01c: Cell indexing on non-cell mentions "cell."
      R-POL33-01d: Undefined function reports "undefined."
      R-POL33-01e: Too many subscript indices reports "index."

    Consistency: Five common mistake categories (01a-e) cover the primary
    error-message quality surface.
    """

    def test_matrix_row_dimension_mismatch(self, S):
        """R-POL33-01a: [1 2; 3 4 5] SHALL report dimension mismatch."""
        r = _err(S, "A_t1 = [1 2; 3 4 5]")
        assert r, "Expected an error for mismatched matrix rows"
        assert "dimension" in r.lower() and "mismatch" in r.lower(), f"Unclear error: {r}"

    def test_inner_matrix_dimensions(self, S):
        """R-POL33-01b: Incompatible A * B SHALL report 'inner matrix dimensions'."""
        S.eval("M1_t = [1 2; 3 4]")
        r = _err(S, "M1_t * [1 2 3]")
        assert r, "Expected an error for inner dimension mismatch"
        assert "inner matrix dimensions" in r.lower(), f"Unclear error: {r}"

    def test_cell_indexing_on_non_cell(self, S):
        """R-POL33-01c: Using {} on numeric array SHALL mention 'cell'."""
        S.eval("NC_t = [1 2 3]")
        r = _err(S, "NC_t{1}")
        assert r, "Expected an error for cell indexing on non-cell"
        assert "cell" in r.lower(), f"Unclear error: {r}"

    def test_undefined_function(self, S):
        """R-POL33-01d: Calling undefined function SHALL report 'undefined'."""
        r = _err(S, "xyzzy_nope(1,2,3)")
        assert r, "Expected error for undefined function"
        assert "undefined" in r.lower(), f"Unclear error: {r}"

    def test_too_many_indices(self, S):
        """R-POL33-01e: Too many subscript indices SHALL report 'index'."""
        S.eval("S5_t = 5")
        r = _err(S, "S5_t(1,2,3,4)")
        assert r, "Expected error for too many indices"
        assert "index" in r.lower(), f"Unclear error: {r}"


# -- 2. Edge cases that must not crash ----------------------------------------

class TestEdgeCases:
    """R-POL33-02: Empty and degenerate inputs SHALL not crash the engine.

    Model-user argument: Scientists exploring data interactively often
    create empty matrices, cells, or strings by accident. These must
    produce sensible results (empty output, zeros, etc.) instead of
    unhandled exceptions that terminate the session.

    Decomposition:
      R-POL33-02a: [] does not crash.
      R-POL33-02b: {} does not crash.
      R-POL33-02c: char([]) does not crash.
      R-POL33-02d: zeros(0,3) does not crash.
      R-POL33-02e: size([]) returns [0 0].
      R-POL33-02f: length([]) returns 0.
      R-POL33-02g: isempty([]) returns 1.
      R-POL33-02h: isempty([1]) returns 0.
      R-POL33-02i: sum([]) returns 0.
      R-POL33-02j: prod([]) returns 1.
      R-POL33-02k: max([]) does not crash.

    Consistency: Eleven edge cases (02a-k) cover empty matrix, cell, string,
    zeros, and aggregate functions on empty inputs.
    """

    def test_empty_matrix_no_crash(self, S):
        """R-POL33-02a: [] SHALL not crash."""
        r = S.eval("[]")
        assert "error" not in r.lower()

    def test_empty_cell_no_crash(self, S):
        """R-POL33-02b: {} SHALL not crash."""
        r = S.eval("{}")
        assert "error" not in r.lower()

    def test_empty_string_no_crash(self, S):
        """R-POL33-02c: char([]) SHALL not crash."""
        r = S.eval("x_es = char([])")
        assert "error" not in str(r).lower()

    def test_zeros_empty_no_crash(self, S):
        """R-POL33-02d: zeros(0,3) SHALL not crash."""
        r = S.eval("zeros(0,3)")
        assert "error" not in str(r).lower()

    def test_size_empty(self, S):
        """R-POL33-02e: size([]) SHALL return [0 0]."""
        v = _val(S, "size([])")
        np.testing.assert_array_equal(v[:2], [0, 0])

    def test_length_empty(self, S):
        """R-POL33-02f: length([]) SHALL return 0."""
        assert _scalar(S, "length([])") == 0

    def test_isempty_true(self, S):
        """R-POL33-02g: isempty([]) SHALL return 1."""
        r = _scalar(S, "isempty([])")
        assert r == 1.0

    def test_isempty_false(self, S):
        """R-POL33-02h: isempty([1]) SHALL return 0."""
        r = _scalar(S, "isempty([1])")
        assert r == 0.0

    def test_sum_empty(self, S):
        """R-POL33-02i: sum([]) SHALL return 0."""
        assert _scalar(S, "sum([])") == 0

    def test_prod_empty(self, S):
        """R-POL33-02j: prod([]) SHALL return 1."""
        assert _scalar(S, "prod([])") == 1

    def test_max_empty_no_crash(self, S):
        """R-POL33-02k: max([]) SHALL not crash."""
        r = S.eval("max([])")
        assert "error" not in str(r).lower()


# -- 3. Numeric edge cases ----------------------------------------------------

class TestNumericEdgeCases:
    """R-POL33-03: IEEE 754 special values (Inf, NaN, eps) SHALL be handled
    correctly in arithmetic and type-checking functions.

    Model-user argument: Numerical algorithms produce Inf and NaN during
    convergence failures or singular operations. An engineer uses isnan(),
    isinf(), isfinite() to detect and handle these cases. Wrong IEEE 754
    behavior causes algorithms to loop forever or produce garbage.

    Decomposition:
      R-POL33-03a: 1/0 produces Inf.
      R-POL33-03b: -1/0 produces -Inf.
      R-POL33-03c: 0/0 produces NaN.
      R-POL33-03d: isnan(NaN) returns 1.
      R-POL33-03e: isinf(Inf) returns 1.
      R-POL33-03f: isfinite(42) returns 1.
      R-POL33-03g: isfinite(Inf) returns 0.
      R-POL33-03h: isnan returns numeric (not boolean).
      R-POL33-03i: eps equals machine epsilon (~2.22e-16).

    Consistency: Division edge cases (03a-c), type-check functions (03d-g),
    return type (03h), and constant value (03i) cover IEEE 754 support.
    """

    def test_div_by_zero_positive(self, S):
        """R-POL33-03a: 1/0 SHALL produce Inf."""
        assert _scalar(S, "1/0") == float("inf")

    def test_div_by_zero_negative(self, S):
        """R-POL33-03b: -1/0 SHALL produce -Inf."""
        assert _scalar(S, "-1/0") == float("-inf")

    def test_zero_div_zero(self, S):
        """R-POL33-03c: 0/0 SHALL produce NaN."""
        assert math.isnan(_scalar(S, "0/0"))

    def test_isnan_of_nan(self, S):
        """R-POL33-03d: isnan(NaN) SHALL return 1."""
        assert _scalar(S, "isnan(NaN)") == 1.0

    def test_isinf_of_inf(self, S):
        """R-POL33-03e: isinf(Inf) SHALL return 1."""
        assert _scalar(S, "isinf(Inf)") == 1.0

    def test_isfinite_of_number(self, S):
        """R-POL33-03f: isfinite(42) SHALL return 1."""
        assert _scalar(S, "isfinite(42)") == 1.0

    def test_isfinite_of_inf(self, S):
        """R-POL33-03g: isfinite(Inf) SHALL return 0."""
        assert _scalar(S, "isfinite(Inf)") == 0.0

    def test_isnan_returns_numeric(self, S):
        """R-POL33-03h: isnan SHALL return numeric (double), not boolean."""
        r = _raw(S, "nr_t", "nr_t = isnan(NaN)")
        arr = np.asarray(_unwrap(r))
        assert arr.dtype != np.bool_, "isnan should return numeric, not bool"

    def test_eps_value(self, S):
        """R-POL33-03i: eps SHALL equal machine epsilon (~2.22e-16)."""
        val = _scalar(S, "eps")
        assert abs(val - 2.220446049250313e-16) < 1e-30


# -- 4. Complex number support ------------------------------------------------

class TestComplexNumbers:
    """R-POL33-04: Complex number literals and functions (abs, real, imag,
    conj, angle) SHALL produce correct results.

    Model-user argument: Signal-processing engineers work with complex
    phasors daily. If 3+4i does not create a complex number or abs(3+4i)
    does not return 5, FFT-based workflows produce wrong magnitude and
    phase results.

    Decomposition:
      R-POL33-04a: 3 + 4i creates a complex number.
      R-POL33-04b: abs(3+4i) returns 5.
      R-POL33-04c: real(3+4i) returns 3.
      R-POL33-04d: imag(3+4i) returns 4.
      R-POL33-04e: conj(3+4i) returns 3-4i.
      R-POL33-04f: angle(3+4i) returns atan2(4,3).

    Consistency: Creation (04a) and five accessor/transform functions
    (04b-f) cover the complex number API.
    """

    def test_complex_literal(self, S):
        """R-POL33-04a: 3 + 4i SHALL create a complex number."""
        r = _raw(S, "zzc", "zzc = 3 + 4i")
        arr = np.asarray(_unwrap(r))
        assert np.issubdtype(arr.dtype, np.complexfloating)
        assert arr.ravel()[0] == 3 + 4j

    def test_abs_complex(self, S):
        """R-POL33-04b: abs(3+4i) SHALL return 5."""
        S.eval("zzc = 3 + 4i")
        assert _scalar(S, "abs(zzc)") == pytest.approx(5.0)

    def test_real_part(self, S):
        """R-POL33-04c: real(3+4i) SHALL return 3."""
        assert _scalar(S, "real(zzc)") == pytest.approx(3.0)

    def test_imag_part(self, S):
        """R-POL33-04d: imag(3+4i) SHALL return 4."""
        assert _scalar(S, "imag(zzc)") == pytest.approx(4.0)

    def test_conjugate(self, S):
        """R-POL33-04e: conj(3+4i) SHALL return 3-4i."""
        r = _raw(S, "cc_t", "cc_t = conj(zzc)")
        arr = np.asarray(_unwrap(r))
        assert arr.ravel()[0] == pytest.approx(3 - 4j)

    def test_angle(self, S):
        """R-POL33-04f: angle(3+4i) SHALL return atan2(4,3)."""
        val = _scalar(S, "angle(zzc)")
        assert val == pytest.approx(math.atan2(4, 3))
