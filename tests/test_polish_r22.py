# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""V&V polish round 22: string ops, matrix indexing, logical operations.

V&V Traceability (backfill):
    R-POL22-01 .. R-POL22-03 (parent requirements)
    R-POL22-01-nn .. R-POL22-03-nn (unit sub-requirements)

SRS trace: SRS-FUNC-001, SRS-VAL-001, SRS-COMPAT-001
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar
from forge.engine.session import ForgeSession


@pytest.fixture(scope="module")
def s():
    return ForgeSession()


def _get(s, name):
    """Get workspace variable, unwrap ForgeArray to numpy."""
    v = s.workspace.get(name)
    if isinstance(v, ForgeArray):
        return _unwrap(v)
    return v


def _get_str(s, name):
    """Get workspace variable as string (for ForgeChar)."""
    v = s.workspace.get(name)
    if isinstance(v, ForgeChar):
        return str(v)
    if isinstance(v, ForgeArray):
        arr = _unwrap(v)
        return "".join(chr(int(c)) for c in arr.flatten())
    return str(v)


# ── String operations ──────────────────────────────────────────────


class TestStringOperations:
    """R-POL22-01: Forge SHALL support character array operations including
    element-wise equality, horizontal concatenation, single and range indexing,
    char() construction from ASCII codes, and inequality detection, all matching
    MATLAB/Octave behavior.

    Model-user argument: An engineer porting text-processing scripts from Octave
    relies on character array indexing and comparison for parsing instrument
    output, building formatted strings, and validating user input. Incorrect
    element-wise comparison or indexing would silently produce wrong parse
    results and corrupt downstream data processing.

    Decomposition:
        R-POL22-01-01: 'hello' == 'hello' returns array of 1s
        R-POL22-01-02: ['hello' ' ' 'world'] concatenates to 'hello world'
        R-POL22-01-03: s(2) indexes single character
        R-POL22-01-04: s(2:3) indexes character range
        R-POL22-01-05: char(65:90) produces uppercase alphabet
        R-POL22-01-06: 'abc' == 'abd' returns [1 1 0]

    Consistency: Sub-requirements cover equality (01, 06), concatenation (02),
    single index (03), range index (04), and construction (05). Together they
    span the full character array operation set.
    """

    def test_string_equality(self, s):
        """R-POL22-01-01: 'hello' == 'hello' returns array of ones."""
        s.eval("str_eq = 'hello' == 'hello'")
        r = _get(s, "str_eq")
        expected = np.ones(5)
        np.testing.assert_array_equal(np.array(r).flatten(), expected)

    def test_string_hcat(self, s):
        """R-POL22-01-02: horizontal concatenation yields 'hello world'."""
        s.eval("str_cat = ['hello' ' ' 'world']")
        r = _get_str(s, "str_cat")
        assert r == "hello world"

    def test_char_indexing_single(self, s):
        """R-POL22-01-03: s(2) of 'test' returns 'e'."""
        s.eval("cs = 'test'")
        s.eval("cs2 = cs(2)")
        r = _get_str(s, "cs2")
        assert r == "e"

    def test_char_indexing_range(self, s):
        """R-POL22-01-04: s(2:3) of 'test' returns 'es'."""
        s.eval("cs3 = 'test'")
        s.eval("cs4 = cs3(2:3)")
        r = _get_str(s, "cs4")
        assert r == "es"

    def test_char_from_codes(self, s):
        """R-POL22-01-05: char(65:90) produces 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'."""
        s.eval("alpha = char(65:90)")
        r = _get_str(s, "alpha")
        assert r == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def test_string_inequality(self, s):
        """R-POL22-01-06: 'abc' == 'abd' returns [1 1 0]."""
        s.eval("str_neq = 'abc' == 'abd'")
        r = _get(s, "str_neq")
        arr = np.array(r).flatten()
        np.testing.assert_array_equal(arr, [1, 1, 0])


# ── Matrix indexing operations ─────────────────────────────────────


class TestMatrixIndexing:
    """R-POL22-02: Forge SHALL support matrix indexing operations including
    colon linearization, the end keyword (with arithmetic), submatrix
    extraction, and transpose-multiply, all using 1-based column-major
    semantics matching MATLAB/Octave.

    Model-user argument: A scientist porting linear algebra or image processing
    code from Octave uses A(:) for vectorization, A(end,:) for boundary rows,
    submatrix slicing for ROI extraction, and transpose-multiply for projection.
    Incorrect indexing or column-major ordering would silently produce wrong
    matrix results.

    Decomposition:
        R-POL22-02-01: A(:) linearizes in column-major order
        R-POL22-02-02: A(end,:) returns last row
        R-POL22-02-03: A(:,end) returns last column
        R-POL22-02-04: A(1:2,1:2) extracts top-left submatrix
        R-POL22-02-05: A' * A produces correct result
        R-POL22-02-06: v(end-1) returns second-to-last element

    Consistency: Sub-requirements cover linearization (01), end keyword in rows
    and columns (02-03), range slicing (04), transpose-multiply (05), and end
    arithmetic (06). Together they verify the full indexing API.
    """

    def test_colon_linearize(self, s):
        """R-POL22-02-01: magic(3)(:) yields column-major 9-element vector."""
        s.eval("M3 = magic(3)")
        s.eval("Mcol = M3(:)")
        r = _get(s, "Mcol")
        arr = np.array(r).flatten()
        assert arr.size == 9
        # magic(3) = [8 1 6; 3 5 7; 4 9 2], column-major: [8,3,4,1,5,9,6,7,2]
        np.testing.assert_array_equal(arr, [8, 3, 4, 1, 5, 9, 6, 7, 2])

    def test_last_row(self, s):
        """R-POL22-02-02: A(end,:) returns [4 9 2] for magic(3)."""
        s.eval("M3b = magic(3)")
        s.eval("Mlr = M3b(end, :)")
        r = _get(s, "Mlr")
        np.testing.assert_array_equal(np.array(r).flatten(), [4, 9, 2])

    def test_last_col(self, s):
        """R-POL22-02-03: A(:,end) returns [6; 7; 2] for magic(3)."""
        s.eval("M3c = magic(3)")
        s.eval("Mlc = M3c(:, end)")
        r = _get(s, "Mlc")
        np.testing.assert_array_equal(np.array(r).flatten(), [6, 7, 2])

    def test_submatrix(self, s):
        """R-POL22-02-04: A(1:2,1:2) returns top-left 2x2 of magic(3)."""
        s.eval("M3d = magic(3)")
        s.eval("Msub = M3d(1:2, 1:2)")
        r = _get(s, "Msub")
        arr = np.array(r)
        assert arr.shape == (2, 2)
        np.testing.assert_array_equal(arr, [[8, 1], [3, 5]])

    def test_transpose_multiply(self, s):
        """R-POL22-02-05: A' * A produces correct 3x3 result."""
        s.eval("M3e = magic(3)")
        s.eval("Mtp = M3e' * M3e")
        r = _get(s, "Mtp")
        arr = np.array(r)
        expected = np.array([[89, 59, 77], [59, 107, 59], [77, 59, 89]])
        np.testing.assert_array_equal(arr, expected)

    def test_end_minus_one(self, s):
        """R-POL22-02-06: v(end-1) returns 40 from [10 20 30 40 50]."""
        s.eval("vend = [10 20 30 40 50]")
        s.eval("vend2 = vend(end-1)")
        r = _get(s, "vend2")
        assert float(np.array(r).flat[0]) == 40


# ── Logical operations ─────────────────────────────────────────────


class TestLogicalOperations:
    """R-POL22-03: Forge SHALL provide logical operations (all, any, xor,
    logical not) that handle vectors and empty arrays with results matching
    MATLAB/Octave conventions (including vacuous truth for empty arrays).

    Model-user argument: A scientist porting conditional logic and data
    validation code from Octave uses all/any for convergence checks, xor for
    toggle logic, and logical negation for mask inversion. The MATLAB convention
    that all([]) returns 1 (vacuously true) is a known compatibility trap that
    must be matched to avoid divergent control flow.

    Decomposition:
        R-POL22-03-01: all([1 1 1]) = 1
        R-POL22-03-02: all([1 0 1]) = 0
        R-POL22-03-03: any([0 0 1]) = 1
        R-POL22-03-04: any([0 0 0]) = 0
        R-POL22-03-05: xor(1,0) = 1
        R-POL22-03-06: xor(1,1) = 0
        R-POL22-03-07: ~[1 0 1] = [0 1 0]
        R-POL22-03-08: all([]) = 1 (vacuously true)

    Consistency: Sub-requirements cover all with true/false/empty (01-02, 08),
    any with true/false (03-04), xor both outcomes (05-06), and vector negation
    (07). Together they validate the full logical operation set including the
    critical empty-array edge case.
    """

    def test_all_true(self, s):
        """R-POL22-03-01: all([1 1 1]) returns 1."""
        s.eval("at = all([1 1 1])")
        r = _get(s, "at")
        assert float(np.array(r).flat[0]) == 1

    def test_all_false(self, s):
        """R-POL22-03-02: all([1 0 1]) returns 0."""
        s.eval("af = all([1 0 1])")
        r = _get(s, "af")
        assert float(np.array(r).flat[0]) == 0

    def test_any_true(self, s):
        """R-POL22-03-03: any([0 0 1]) returns 1."""
        s.eval("ayt = any([0 0 1])")
        r = _get(s, "ayt")
        assert float(np.array(r).flat[0]) == 1

    def test_any_false(self, s):
        """R-POL22-03-04: any([0 0 0]) returns 0."""
        s.eval("ayf = any([0 0 0])")
        r = _get(s, "ayf")
        assert float(np.array(r).flat[0]) == 0

    def test_xor_true(self, s):
        """R-POL22-03-05: xor(1,0) returns 1."""
        s.eval("xt = xor(1, 0)")
        r = _get(s, "xt")
        assert float(np.array(r).flat[0]) == 1

    def test_xor_false(self, s):
        """R-POL22-03-06: xor(1,1) returns 0."""
        s.eval("xf = xor(1, 1)")
        r = _get(s, "xf")
        assert float(np.array(r).flat[0]) == 0

    def test_logical_not_vector(self, s):
        """R-POL22-03-07: ~[1 0 1] returns [0 1 0]."""
        s.eval("ln = ~[1 0 1]")
        r = _get(s, "ln")
        np.testing.assert_array_equal(np.array(r).flatten(), [0, 1, 0])

    def test_all_empty(self, s):
        """R-POL22-03-08: all([]) returns 1 (MATLAB vacuously true convention)."""
        s.eval("ae = all([])")
        r = _get(s, "ae")
        assert float(np.array(r).flat[0]) == 1
