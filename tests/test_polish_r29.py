# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for matrix deletion, value semantics, logical indexing assignment,
end keyword, tilde multi-assign, operator precedence, string functions
(R29 polish)."""
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar


@pytest.fixture
def sess():
    return ForgeSession()


@pytest.fixture
def s(sess):
    return sess._engine


# ── 1. Matrix row/element deletion via = [] ─────────────────────────
class TestMatrixDeletion:
    def test_delete_row(self, s):
        """A(2,:) = [] deletes the second row."""
        s.eval("A = [1 2; 3 4]; A(2,:) = []")
        r = _unwrap(s.eval("A"))
        np.testing.assert_array_equal(r, [[1, 2]])

    def test_delete_elements_by_index(self, s):
        """A([2 4]) = [] removes elements at positions 2 and 4."""
        s.eval("A = [1 2 3 4 5]; A([2 4]) = []")
        r = _unwrap(s.eval("A"))
        np.testing.assert_array_equal(r.ravel(), [1, 3, 5])

    def test_delete_column(self, s):
        """A(:,2) = [] deletes the second column."""
        s.eval("A = [1 2 3; 4 5 6]; A(:,2) = []")
        r = _unwrap(s.eval("A"))
        np.testing.assert_array_equal(r, [[1, 3], [4, 6]])

    def test_delete_all_elements(self, s):
        """A(:) = [] produces an empty matrix."""
        s.eval("A = [1 2 3]; A(:) = []")
        r = _unwrap(s.eval("A"))
        assert r.size == 0


# ── 1c. Value semantics (copy-on-assign) ────────────────────────────
class TestValueSemantics:
    def test_assign_copies_array(self, s):
        """B = A makes an independent copy; modifying B leaves A unchanged."""
        s.eval("A = [1 2; 3 4]; B = A; B(1,1) = 99")
        a = _unwrap(s.eval("A"))
        b = _unwrap(s.eval("B"))
        np.testing.assert_array_equal(a, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(b, [[99, 2], [3, 4]])


# ── 2. Logical indexing assignment ───────────────────────────────────
class TestLogicalIndexingAssign:
    def test_logical_mask_zero_out(self, s):
        """A(A > 25) = 0 zeros elements greater than 25."""
        s.eval("A = [10 20 30 40 50]; A(A > 25) = 0")
        r = _unwrap(s.eval("A"))
        np.testing.assert_array_equal(r.ravel(), [10, 20, 0, 0, 0])

    def test_logical_mask_negative_replace(self, s):
        """A(A < 3) = -1 replaces elements less than 3 with -1."""
        s.eval("A = [1 2 3; 4 5 6]; A(A < 3) = -1")
        r = _unwrap(s.eval("A"))
        np.testing.assert_array_equal(r, [[-1, -1, 3], [4, 5, 6]])


# ── 3. End keyword in expressions ───────────────────────────────────
class TestEndKeyword:
    def test_end_minus_1_colon_end(self, s):
        """A(end-1:end) returns the last two elements."""
        r = s.eval("A = [10 20 30 40 50]; A(end-1:end)")
        np.testing.assert_array_equal(_unwrap(r).ravel(), [40, 50])

    def test_end_2d(self, s):
        """A(end, end) returns bottom-right element of a matrix."""
        r = s.eval("A = [1 2 3; 4 5 6]; A(end, end)")
        assert float(_unwrap(r).ravel()[0]) == 6.0


# ── 4. Multiple assignment with tilde ────────────────────────────────
class TestTildeMultiAssign:
    def test_sort_ignore_first(self, s):
        """[~, idx] = sort([3 1 2]) discards sorted values, keeps indices."""
        s.eval("[~, idx] = sort([3 1 2])")
        r = _unwrap(s.eval("idx"))
        np.testing.assert_array_equal(r.ravel(), [2, 3, 1])

    def test_find_ignore_first_two(self, s):
        """[~, ~, v] = find([0 3 0 4]) discards row/col, keeps values."""
        s.eval("[~, ~, v] = find([0 3 0 4])")
        r = _unwrap(s.eval("v"))
        np.testing.assert_array_equal(r.ravel(), [3, 4])


# ── 5. Operator precedence ──────────────────────────────────────────
class TestOperatorPrecedence:
    def test_power_left_associative(self, s):
        """2^3^2 = (2^3)^2 = 64 (left-associative, matches Octave)."""
        r = s.eval("2^3^2")
        assert float(_unwrap(r).ravel()[0]) == 64.0

    def test_unary_minus_after_power(self, s):
        """-2^2 = -(2^2) = -4."""
        r = s.eval("-2^2")
        assert float(_unwrap(r).ravel()[0]) == -4.0

    def test_mul_add_precedence(self, s):
        """3*4 + 5*6 = 12 + 30 = 42."""
        r = s.eval("3 * 4 + 5 * 6")
        assert float(_unwrap(r).ravel()[0]) == 42.0

    def test_elementwise_power_broadcast(self, s):
        """2 .^ [1 2 3] = [2 4 8]."""
        r = s.eval("2 .^ [1 2 3]")
        np.testing.assert_array_equal(_unwrap(r).ravel(), [2, 4, 8])


# ── 6. String functions ──────────────────────────────────────────────
class TestStringFunctions:
    def test_string_indexing_with_end(self, s):
        """s(7:end) on 'hello world' gives 'world'."""
        r = s.eval("s = 'hello world'; s(7:end)")
        assert isinstance(r, ForgeChar)
        assert r.to_str() == "world"

    def test_toupper(self, s):
        """toupper('hello') returns 'HELLO'."""
        r = s.eval("toupper('hello')")
        assert isinstance(r, ForgeChar)
        assert r.to_str() == "HELLO"

    def test_tolower(self, s):
        """tolower('HELLO') returns 'hello'."""
        r = s.eval("tolower('HELLO')")
        assert isinstance(r, ForgeChar)
        assert r.to_str() == "hello"
