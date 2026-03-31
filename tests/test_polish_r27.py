# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for command-style syntax, short-circuit evaluation, matrix comparisons,
colon expressions, and matrix concatenation edge cases (R27 polish)."""
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
    """Verify command-style (no parens) invocations for cd, format, clear, who, whos."""

    def test_cd_changes_directory(self, sess):
        """cd /tmp should change the process working directory."""
        old = os.getcwd()
        try:
            sess.eval("cd /tmp")
            assert os.getcwd() == "/tmp"
        finally:
            os.chdir(old)

    def test_format_long(self, sess):
        """format long should switch the display format."""
        sess.eval("format long")
        fmt = getattr(sess, "_format", getattr(sess, "format", "short"))
        assert fmt == "long"

    def test_format_short(self, sess):
        """format short should reset to default."""
        sess.eval("format long")
        sess.eval("format short")
        fmt = getattr(sess, "_format", getattr(sess, "format", "short"))
        assert fmt == "short"

    def test_format_short_e(self, sess):
        """format short e should set scientific notation format."""
        sess.eval("format short e")
        fmt = getattr(sess, "_format", getattr(sess, "format", "short"))
        assert "short" in fmt and "e" in fmt

    def test_clear_variables(self, sess, s):
        """clear x y z should remove named variables from workspace."""
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
        """who should list workspace variable names."""
        sess.eval("clear")
        sess.eval("alpha = 1")
        sess.eval("beta = 2")
        result = sess.eval("who")
        assert "alpha" in result
        assert "beta" in result

    def test_whos_lists_with_details(self, sess):
        """whos should list variables with size/type details."""
        sess.eval("clear")
        sess.eval("myvar = [1 2 3]")
        result = sess.eval("whos")
        assert "myvar" in result
        assert "double" in result


# ---------------------------------------------------------------------------
# 2. Short-circuit evaluation
# ---------------------------------------------------------------------------

class TestShortCircuitEvaluation:
    """Verify || and && short-circuit (do not evaluate second operand when unnecessary)."""

    def test_true_or_short_circuits(self, sess):
        """true || error(...) should NOT raise because LHS is true."""
        result = sess.eval("true || error('should not reach')")
        assert "should not reach" not in result

    def test_false_and_short_circuits(self, sess):
        """false && error(...) should NOT raise because LHS is false."""
        result = sess.eval("false && error('should not reach')")
        assert "should not reach" not in result

    def test_and_returns_zero(self, sess, s):
        """1 && 0 should return 0 (logical false)."""
        sess.eval("r = 1 && 0")
        assert _scalar(s, "r") == 0

    def test_or_returns_one(self, sess, s):
        """0 || 1 should return 1 (logical true)."""
        sess.eval("r = 0 || 1")
        assert _scalar(s, "r") == 1

    def test_and_both_true(self, sess, s):
        """1 && 1 should return 1."""
        sess.eval("r = 1 && 1")
        assert _scalar(s, "r") == 1

    def test_or_both_false(self, sess, s):
        """0 || 0 should return 0."""
        sess.eval("r = 0 || 0")
        assert _scalar(s, "r") == 0


# ---------------------------------------------------------------------------
# 3. Comparison operators on matrices
# ---------------------------------------------------------------------------

class TestMatrixComparisons:
    """Element-wise comparison operators on matrices/vectors."""

    def test_vector_greater_than_scalar(self, sess, s):
        """[1 2 3] > 2 should give [0 0 1]."""
        sess.eval("c = [1 2 3] > 2")
        arr = _val(s, "c").flatten()
        np.testing.assert_array_equal(arr, [False, False, True])

    def test_vector_equality(self, sess, s):
        """[1 2 3] == [1 0 3] should give [1 0 1]."""
        sess.eval("c = [1 2 3] == [1 0 3]")
        arr = _val(s, "c").flatten()
        np.testing.assert_array_equal(arr, [True, False, True])

    def test_vector_not_equal(self, sess, s):
        """[1 2 3] ~= [1 0 3] should give [0 1 0]."""
        sess.eval("c = [1 2 3] ~= [1 0 3]")
        arr = _val(s, "c").flatten()
        np.testing.assert_array_equal(arr, [False, True, False])

    def test_matrix_greater_equal(self, sess, s):
        """[1 2; 3 4] >= [1 3; 2 4] should give [1 0; 1 1]."""
        sess.eval("c = [1 2; 3 4] >= [1 3; 2 4]")
        arr = _val(s, "c")
        np.testing.assert_array_equal(arr, [[True, False], [True, True]])

    def test_vector_less_than(self, sess, s):
        """[5 3 1] < 3 should give [0 0 1]."""
        sess.eval("c = [5 3 1] < 3")
        arr = _val(s, "c").flatten()
        np.testing.assert_array_equal(arr, [False, False, True])

    def test_vector_less_equal(self, sess, s):
        """[1 2 3] <= 2 should give [1 1 0]."""
        sess.eval("c = [1 2 3] <= 2")
        arr = _val(s, "c").flatten()
        np.testing.assert_array_equal(arr, [True, True, False])


# ---------------------------------------------------------------------------
# 4. Colon expressions
# ---------------------------------------------------------------------------

class TestColonExpressions:
    """Colon range generation: start:stop, start:step:stop."""

    def test_simple_range(self, sess, s):
        """1:5 should give [1 2 3 4 5]."""
        sess.eval("d = 1:5")
        np.testing.assert_array_almost_equal(_val(s, "d").flatten(), [1, 2, 3, 4, 5])

    def test_step_range(self, sess, s):
        """1:2:10 should give [1 3 5 7 9]."""
        sess.eval("d = 1:2:10")
        np.testing.assert_array_almost_equal(_val(s, "d").flatten(), [1, 3, 5, 7, 9])

    def test_descending_range(self, sess, s):
        """5:-1:1 should give [5 4 3 2 1]."""
        sess.eval("d = 5:-1:1")
        np.testing.assert_array_almost_equal(_val(s, "d").flatten(), [5, 4, 3, 2, 1])

    def test_fractional_step(self, sess, s):
        """0:0.1:1 should give [0 0.1 0.2 ... 1.0]."""
        sess.eval("d = 0:0.1:1")
        expected = np.arange(0, 1.0 + 0.05, 0.1)
        arr = _val(s, "d").flatten()
        assert len(arr) == 11
        np.testing.assert_array_almost_equal(arr, expected)

    def test_empty_range(self, sess, s):
        """5:1 (ascending default step with stop < start) should give empty."""
        sess.eval("d = 5:1")
        arr = _val(s, "d").flatten()
        assert len(arr) == 0

    def test_single_element_range(self, sess, s):
        """3:3 should give [3]."""
        sess.eval("d = 3:3")
        np.testing.assert_array_almost_equal(_val(s, "d").flatten(), [3])


# ---------------------------------------------------------------------------
# 5. Matrix concatenation edge cases
# ---------------------------------------------------------------------------

class TestMatrixConcatenation:
    """Row/column concatenation and edge cases."""

    def test_semicolon_row_concat(self, sess, s):
        """[1 2 3; 4 5 6] should produce a 2x3 matrix."""
        sess.eval("m = [1 2 3; 4 5 6]")
        arr = _val(s, "m")
        assert arr.shape == (2, 3)
        np.testing.assert_array_equal(arr, [[1, 2, 3], [4, 5, 6]])

    def test_nested_bracket_vertical(self, sess, s):
        """[[1 2]; [3 4]; [5 6]] should produce a 3x2 matrix."""
        sess.eval("m = [[1 2]; [3 4]; [5 6]]")
        arr = _val(s, "m")
        assert arr.shape == (3, 2)
        np.testing.assert_array_equal(arr, [[1, 2], [3, 4], [5, 6]])

    def test_function_concat_vertical(self, sess, s):
        """[zeros(2,3); ones(1,3)] should produce a 3x3 matrix."""
        sess.eval("m = [zeros(2,3); ones(1,3)]")
        arr = _val(s, "m")
        assert arr.shape == (3, 3)
        np.testing.assert_array_equal(arr[:2, :], np.zeros((2, 3)))
        np.testing.assert_array_equal(arr[2, :], np.ones(3))

    def test_empty_matrix(self, sess, s):
        """[] should produce an empty matrix."""
        sess.eval("m = []")
        arr = _val(s, "m")
        assert arr.size == 0

    def test_horizontal_concat(self, sess, s):
        """[[1 2] [3 4]] should produce a 1x4 row."""
        sess.eval("m = [[1 2] [3 4]]")
        arr = _val(s, "m").flatten()
        np.testing.assert_array_equal(arr, [1, 2, 3, 4])

    def test_column_vector(self, sess, s):
        """[1; 2; 3] should be a 3x1 column vector."""
        sess.eval("m = [1; 2; 3]")
        arr = _val(s, "m")
        assert arr.shape == (3, 1)
        np.testing.assert_array_equal(arr.flatten(), [1, 2, 3])
