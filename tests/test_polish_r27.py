# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for command-style syntax, short-circuit evaluation, matrix comparisons,
colon expressions, and matrix concatenation edge cases (R27 polish).

V&V Traceability (backfill)
===========================
R-POL27-01: Command-style syntax SHALL work for cd, format, clear, who, and
            whos without parentheses.

    Model-user argument: Octave and MATLAB users type commands like
    ``cd /tmp`` or ``format long`` at the prompt without parentheses. An
    engineer migrating scripts that use these commands will expect them to
    work identically. Requiring parentheses would break muscle memory and
    existing scripts.

    Decomposition:
      R-POL27-01a: cd /tmp changes the working directory.
      R-POL27-01b: format long switches display format to long.
      R-POL27-01c: format short resets display format to short.
      R-POL27-01d: format short e sets scientific notation format.
      R-POL27-01e: clear x y z removes named variables from workspace.
      R-POL27-01f: who lists workspace variable names.
      R-POL27-01g: whos lists variables with size/type details.

    Consistency: Sub-requirements span directory navigation (01a), display
    format variants (01b-d), variable cleanup (01e), and introspection
    (01f-g), covering all tested command-style invocations.

R-POL27-02: Short-circuit operators || and && SHALL not evaluate the second
            operand when the result is determined by the first.

    Model-user argument: Guard patterns like ``exist('x') && x > 0`` are
    common in Octave scripts. If the second operand is evaluated when it
    should not be, the script crashes on an undefined variable. Correct
    short-circuit behavior is essential for safe migration.

    Decomposition:
      R-POL27-02a: true || error(...) does not raise.
      R-POL27-02b: false && error(...) does not raise.
      R-POL27-02c: 1 && 0 returns 0.
      R-POL27-02d: 0 || 1 returns 1.
      R-POL27-02e: 1 && 1 returns 1.
      R-POL27-02f: 0 || 0 returns 0.

    Consistency: Sub-requirements cover both skip cases (02a-b) and all
    four truth-table outcomes (02c-f), fully verifying short-circuit logic.

R-POL27-03: Element-wise comparison operators SHALL work on matrices and
            vectors, returning logical arrays.

    Model-user argument: Vectorized comparisons like ``A > threshold`` are
    fundamental to Octave workflows for filtering data. An engineer expects
    element-wise logical arrays from >, <, >=, <=, ==, and ~= operators on
    conformable arrays.

    Decomposition:
      R-POL27-03a: Vector > scalar returns correct logical vector.
      R-POL27-03b: Vector == vector returns element-wise equality.
      R-POL27-03c: Vector ~= vector returns element-wise inequality.
      R-POL27-03d: Matrix >= matrix returns correct logical matrix.
      R-POL27-03e: Vector < scalar returns correct logical vector.
      R-POL27-03f: Vector <= scalar returns correct logical vector.

    Consistency: All six comparison operators are covered across scalar
    broadcast (03a, 03e, 03f) and conformable array (03b-d) cases.

R-POL27-04: Colon expressions SHALL generate ranges with start:stop and
            start:step:stop syntax, including descending and fractional steps.

    Model-user argument: Range generation via colon notation is the most
    common way to create index vectors and iteration bounds in Octave.
    Incorrect range semantics (wrong length, wrong direction, off-by-one)
    would silently corrupt loop bounds and indexing in migrated scripts.

    Decomposition:
      R-POL27-04a: 1:5 produces [1 2 3 4 5].
      R-POL27-04b: 1:2:10 produces [1 3 5 7 9].
      R-POL27-04c: 5:-1:1 produces [5 4 3 2 1].
      R-POL27-04d: 0:0.1:1 produces 11 elements from 0.0 to 1.0.
      R-POL27-04e: 5:1 (empty ascending) produces empty array.
      R-POL27-04f: 3:3 produces [3].

    Consistency: Sub-requirements cover ascending (04a), stepped (04b),
    descending (04c), fractional (04d), empty (04e), and single-element
    (04f) ranges, spanning all colon-expression edge cases.

R-POL27-05: Matrix concatenation SHALL support row and column concatenation
            including nested brackets, function results, and empty matrices.

    Model-user argument: Building matrices by concatenation (semicolons for
    vertical, spaces for horizontal) is core Octave syntax. An engineer
    constructing composite matrices from sub-blocks needs correct shape
    inference and empty-matrix handling.

    Decomposition:
      R-POL27-05a: [1 2 3; 4 5 6] produces a 2x3 matrix.
      R-POL27-05b: [[1 2]; [3 4]; [5 6]] produces a 3x2 matrix.
      R-POL27-05c: [zeros(2,3); ones(1,3)] produces a 3x3 matrix.
      R-POL27-05d: [] produces an empty matrix.
      R-POL27-05e: [[1 2] [3 4]] produces a 1x4 row.
      R-POL27-05f: [1; 2; 3] produces a 3x1 column vector.

    Consistency: Sub-requirements cover semicolon rows (05a), nested
    brackets (05b), function-result concatenation (05c), empty (05d),
    horizontal (05e), and column-vector (05f) cases.
"""
import os
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def sess():
    return ForgeSession()


@pytest.fixture
def s(sess):
    return sess._engine


def _val(s, name):
    """Get a workspace variable as a numpy array."""
    v = s.workspace.get(name)
    return np.asarray(_unwrap(v))


def _scalar(s, name):
    return float(_val(s, name).flat[0])


# ---------------------------------------------------------------------------
# 1. Command-style syntax
# ---------------------------------------------------------------------------

class TestCommandStyleSyntax:
    """R-POL27-01: Command-style syntax SHALL work for cd, format, clear, who,
    and whos without parentheses.

    Model-user argument: Octave and MATLAB users type commands like
    ``cd /tmp`` or ``format long`` at the prompt without parentheses. An
    engineer migrating scripts that use these commands will expect them to
    work identically. Requiring parentheses would break muscle memory and
    existing scripts.

    Decomposition:
      R-POL27-01a: cd /tmp changes the working directory.
      R-POL27-01b: format long switches display format to long.
      R-POL27-01c: format short resets display format to short.
      R-POL27-01d: format short e sets scientific notation format.
      R-POL27-01e: clear x y z removes named variables from workspace.
      R-POL27-01f: who lists workspace variable names.
      R-POL27-01g: whos lists variables with size/type details.

    Consistency: Sub-requirements span directory navigation (01a), display
    format variants (01b-d), variable cleanup (01e), and introspection
    (01f-g), covering all tested command-style invocations.
    """

    def test_cd_changes_directory(self, sess):
        """R-POL27-01a: cd /tmp SHALL change the process working directory."""
        old = os.getcwd()
        try:
            sess.eval("cd /tmp")
            assert os.getcwd() == "/tmp"
        finally:
            os.chdir(old)

    def test_format_long(self, sess):
        """R-POL27-01b: format long SHALL switch the display format."""
        sess.eval("format long")
        fmt = getattr(sess, "_format", getattr(sess, "format", "short"))
        assert fmt == "long"

    def test_format_short(self, sess):
        """R-POL27-01c: format short SHALL reset to default."""
        sess.eval("format long")
        sess.eval("format short")
        fmt = getattr(sess, "_format", getattr(sess, "format", "short"))
        assert fmt == "short"

    def test_format_short_e(self, sess):
        """R-POL27-01d: format short e SHALL set scientific notation format."""
        sess.eval("format short e")
        fmt = getattr(sess, "_format", getattr(sess, "format", "short"))
        assert "short" in fmt and "e" in fmt

    def test_clear_variables(self, sess, s):
        """R-POL27-01e: clear x y z SHALL remove named variables from workspace."""
        sess.eval("x = 1")
        sess.eval("y = 2")
        sess.eval("z = 3")
        assert s.workspace.has("x")
        assert s.workspace.has("y")
        assert s.workspace.has("z")
        sess.eval("clear x y z")
        assert not s.workspace.has("x")
        assert not s.workspace.has("y")
        assert not s.workspace.has("z")

    def test_who_lists_variables(self, sess):
        """R-POL27-01f: who SHALL list workspace variable names."""
        sess.eval("clear")
        sess.eval("alpha = 1")
        sess.eval("beta = 2")
        result = sess.eval("who")
        assert "alpha" in result
        assert "beta" in result

    def test_whos_lists_with_details(self, sess):
        """R-POL27-01g: whos SHALL list variables with size/type details."""
        sess.eval("clear")
        sess.eval("myvar = [1 2 3]")
        result = sess.eval("whos")
        assert "myvar" in result
        assert "double" in result


# ---------------------------------------------------------------------------
# 2. Short-circuit evaluation
# ---------------------------------------------------------------------------

class TestShortCircuitEvaluation:
    """R-POL27-02: Short-circuit operators || and && SHALL not evaluate the
    second operand when the result is determined by the first.

    Model-user argument: Guard patterns like ``exist('x') && x > 0`` are
    common in Octave scripts. If the second operand is evaluated when it
    should not be, the script crashes on an undefined variable. Correct
    short-circuit behavior is essential for safe migration.

    Decomposition:
      R-POL27-02a: true || error(...) does not raise.
      R-POL27-02b: false && error(...) does not raise.
      R-POL27-02c: 1 && 0 returns 0.
      R-POL27-02d: 0 || 1 returns 1.
      R-POL27-02e: 1 && 1 returns 1.
      R-POL27-02f: 0 || 0 returns 0.

    Consistency: Sub-requirements cover both skip cases (02a-b) and all
    four truth-table outcomes (02c-f), fully verifying short-circuit logic.
    """

    def test_true_or_short_circuits(self, sess):
        """R-POL27-02a: true || error(...) SHALL NOT raise."""
        result = sess.eval("true || error('should not reach')")
        assert "should not reach" not in result

    def test_false_and_short_circuits(self, sess):
        """R-POL27-02b: false && error(...) SHALL NOT raise."""
        result = sess.eval("false && error('should not reach')")
        assert "should not reach" not in result

    def test_and_returns_zero(self, sess, s):
        """R-POL27-02c: 1 && 0 SHALL return 0 (logical false)."""
        sess.eval("r = 1 && 0")
        assert _scalar(s, "r") == 0

    def test_or_returns_one(self, sess, s):
        """R-POL27-02d: 0 || 1 SHALL return 1 (logical true)."""
        sess.eval("r = 0 || 1")
        assert _scalar(s, "r") == 1

    def test_and_both_true(self, sess, s):
        """R-POL27-02e: 1 && 1 SHALL return 1."""
        sess.eval("r = 1 && 1")
        assert _scalar(s, "r") == 1

    def test_or_both_false(self, sess, s):
        """R-POL27-02f: 0 || 0 SHALL return 0."""
        sess.eval("r = 0 || 0")
        assert _scalar(s, "r") == 0


# ---------------------------------------------------------------------------
# 3. Comparison operators on matrices
# ---------------------------------------------------------------------------

class TestMatrixComparisons:
    """R-POL27-03: Element-wise comparison operators SHALL work on matrices
    and vectors, returning logical arrays.

    Model-user argument: Vectorized comparisons like ``A > threshold`` are
    fundamental to Octave workflows for filtering data. An engineer expects
    element-wise logical arrays from >, <, >=, <=, ==, and ~= operators on
    conformable arrays.

    Decomposition:
      R-POL27-03a: Vector > scalar returns correct logical vector.
      R-POL27-03b: Vector == vector returns element-wise equality.
      R-POL27-03c: Vector ~= vector returns element-wise inequality.
      R-POL27-03d: Matrix >= matrix returns correct logical matrix.
      R-POL27-03e: Vector < scalar returns correct logical vector.
      R-POL27-03f: Vector <= scalar returns correct logical vector.

    Consistency: All six comparison operators are covered across scalar
    broadcast (03a, 03e, 03f) and conformable array (03b-d) cases.
    """

    def test_vector_greater_than_scalar(self, sess, s):
        """R-POL27-03a: [1 2 3] > 2 SHALL give [0 0 1]."""
        sess.eval("c = [1 2 3] > 2")
        arr = _val(s, "c").flatten()
        np.testing.assert_array_equal(arr, [False, False, True])

    def test_vector_equality(self, sess, s):
        """R-POL27-03b: [1 2 3] == [1 0 3] SHALL give [1 0 1]."""
        sess.eval("c = [1 2 3] == [1 0 3]")
        arr = _val(s, "c").flatten()
        np.testing.assert_array_equal(arr, [True, False, True])

    def test_vector_not_equal(self, sess, s):
        """R-POL27-03c: [1 2 3] ~= [1 0 3] SHALL give [0 1 0]."""
        sess.eval("c = [1 2 3] ~= [1 0 3]")
        arr = _val(s, "c").flatten()
        np.testing.assert_array_equal(arr, [False, True, False])

    def test_matrix_greater_equal(self, sess, s):
        """R-POL27-03d: [1 2; 3 4] >= [1 3; 2 4] SHALL give [1 0; 1 1]."""
        sess.eval("c = [1 2; 3 4] >= [1 3; 2 4]")
        arr = _val(s, "c")
        np.testing.assert_array_equal(arr, [[True, False], [True, True]])

    def test_vector_less_than(self, sess, s):
        """R-POL27-03e: [5 3 1] < 3 SHALL give [0 0 1]."""
        sess.eval("c = [5 3 1] < 3")
        arr = _val(s, "c").flatten()
        np.testing.assert_array_equal(arr, [False, False, True])

    def test_vector_less_equal(self, sess, s):
        """R-POL27-03f: [1 2 3] <= 2 SHALL give [1 1 0]."""
        sess.eval("c = [1 2 3] <= 2")
        arr = _val(s, "c").flatten()
        np.testing.assert_array_equal(arr, [True, True, False])


# ---------------------------------------------------------------------------
# 4. Colon expressions
# ---------------------------------------------------------------------------

class TestColonExpressions:
    """R-POL27-04: Colon expressions SHALL generate ranges with start:stop and
    start:step:stop syntax, including descending and fractional steps.

    Model-user argument: Range generation via colon notation is the most
    common way to create index vectors and iteration bounds in Octave.
    Incorrect range semantics (wrong length, wrong direction, off-by-one)
    would silently corrupt loop bounds and indexing in migrated scripts.

    Decomposition:
      R-POL27-04a: 1:5 produces [1 2 3 4 5].
      R-POL27-04b: 1:2:10 produces [1 3 5 7 9].
      R-POL27-04c: 5:-1:1 produces [5 4 3 2 1].
      R-POL27-04d: 0:0.1:1 produces 11 elements from 0.0 to 1.0.
      R-POL27-04e: 5:1 (empty ascending) produces empty array.
      R-POL27-04f: 3:3 produces [3].

    Consistency: Sub-requirements cover ascending (04a), stepped (04b),
    descending (04c), fractional (04d), empty (04e), and single-element
    (04f) ranges, spanning all colon-expression edge cases.
    """

    def test_simple_range(self, sess, s):
        """R-POL27-04a: 1:5 SHALL give [1 2 3 4 5]."""
        sess.eval("d = 1:5")
        np.testing.assert_array_almost_equal(_val(s, "d").flatten(), [1, 2, 3, 4, 5])

    def test_step_range(self, sess, s):
        """R-POL27-04b: 1:2:10 SHALL give [1 3 5 7 9]."""
        sess.eval("d = 1:2:10")
        np.testing.assert_array_almost_equal(_val(s, "d").flatten(), [1, 3, 5, 7, 9])

    def test_descending_range(self, sess, s):
        """R-POL27-04c: 5:-1:1 SHALL give [5 4 3 2 1]."""
        sess.eval("d = 5:-1:1")
        np.testing.assert_array_almost_equal(_val(s, "d").flatten(), [5, 4, 3, 2, 1])

    def test_fractional_step(self, sess, s):
        """R-POL27-04d: 0:0.1:1 SHALL produce 11 elements from 0.0 to 1.0."""
        sess.eval("d = 0:0.1:1")
        expected = np.arange(0, 1.0 + 0.05, 0.1)
        arr = _val(s, "d").flatten()
        assert len(arr) == 11
        np.testing.assert_array_almost_equal(arr, expected)

    def test_empty_range(self, sess, s):
        """R-POL27-04e: 5:1 SHALL produce an empty array."""
        sess.eval("d = 5:1")
        arr = _val(s, "d").flatten()
        assert len(arr) == 0

    def test_single_element_range(self, sess, s):
        """R-POL27-04f: 3:3 SHALL produce [3]."""
        sess.eval("d = 3:3")
        np.testing.assert_array_almost_equal(_val(s, "d").flatten(), [3])


# ---------------------------------------------------------------------------
# 5. Matrix concatenation edge cases
# ---------------------------------------------------------------------------

class TestMatrixConcatenation:
    """R-POL27-05: Matrix concatenation SHALL support row and column
    concatenation including nested brackets, function results, and empty
    matrices.

    Model-user argument: Building matrices by concatenation (semicolons for
    vertical, spaces for horizontal) is core Octave syntax. An engineer
    constructing composite matrices from sub-blocks needs correct shape
    inference and empty-matrix handling.

    Decomposition:
      R-POL27-05a: [1 2 3; 4 5 6] produces a 2x3 matrix.
      R-POL27-05b: [[1 2]; [3 4]; [5 6]] produces a 3x2 matrix.
      R-POL27-05c: [zeros(2,3); ones(1,3)] produces a 3x3 matrix.
      R-POL27-05d: [] produces an empty matrix.
      R-POL27-05e: [[1 2] [3 4]] produces a 1x4 row.
      R-POL27-05f: [1; 2; 3] produces a 3x1 column vector.

    Consistency: Sub-requirements cover semicolon rows (05a), nested
    brackets (05b), function-result concatenation (05c), empty (05d),
    horizontal (05e), and column-vector (05f) cases.
    """

    def test_semicolon_row_concat(self, sess, s):
        """R-POL27-05a: [1 2 3; 4 5 6] SHALL produce a 2x3 matrix."""
        sess.eval("m = [1 2 3; 4 5 6]")
        arr = _val(s, "m")
        assert arr.shape == (2, 3)
        np.testing.assert_array_equal(arr, [[1, 2, 3], [4, 5, 6]])

    def test_nested_bracket_vertical(self, sess, s):
        """R-POL27-05b: [[1 2]; [3 4]; [5 6]] SHALL produce a 3x2 matrix."""
        sess.eval("m = [[1 2]; [3 4]; [5 6]]")
        arr = _val(s, "m")
        assert arr.shape == (3, 2)
        np.testing.assert_array_equal(arr, [[1, 2], [3, 4], [5, 6]])

    def test_function_concat_vertical(self, sess, s):
        """R-POL27-05c: [zeros(2,3); ones(1,3)] SHALL produce a 3x3 matrix."""
        sess.eval("m = [zeros(2,3); ones(1,3)]")
        arr = _val(s, "m")
        assert arr.shape == (3, 3)
        np.testing.assert_array_equal(arr[:2, :], np.zeros((2, 3)))
        np.testing.assert_array_equal(arr[2, :], np.ones(3))

    def test_empty_matrix(self, sess, s):
        """R-POL27-05d: [] SHALL produce an empty matrix."""
        sess.eval("m = []")
        arr = _val(s, "m")
        assert arr.size == 0

    def test_horizontal_concat(self, sess, s):
        """R-POL27-05e: [[1 2] [3 4]] SHALL produce a 1x4 row."""
        sess.eval("m = [[1 2] [3 4]]")
        arr = _val(s, "m").flatten()
        np.testing.assert_array_equal(arr, [1, 2, 3, 4])

    def test_column_vector(self, sess, s):
        """R-POL27-05f: [1; 2; 3] SHALL be a 3x1 column vector."""
        sess.eval("m = [1; 2; 3]")
        arr = _val(s, "m")
        assert arr.shape == (3, 1)
        np.testing.assert_array_equal(arr.flatten(), [1, 2, 3])
