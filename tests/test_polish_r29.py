# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for matrix deletion, value semantics, logical indexing assignment,
end keyword, tilde multi-assign, operator precedence, string functions
(R29 polish).

V&V Traceability (backfill)
===========================
R-POL29-01: Matrix row/column/element deletion via assignment to [] SHALL
            remove the specified elements and reshape the result correctly.

    Model-user argument: Octave engineers routinely delete matrix rows with
    A(2,:) = [] or filter elements with A([2 4]) = []. If deletion semantics
    differ, data-cleaning scripts produce arrays of wrong shape, leading to
    dimension-mismatch errors downstream.

    Decomposition:
      R-POL29-01a: A(2,:) = [] deletes the second row.
      R-POL29-01b: A([2 4]) = [] removes elements at those positions.
      R-POL29-01c: A(:,2) = [] deletes the second column.
      R-POL29-01d: A(:) = [] produces an empty matrix.

    Consistency: Row (01a), indexed element (01b), column (01c), and
    full-clear (01d) deletions are all covered.

R-POL29-02: Assignment SHALL use value semantics (copy-on-assign) so that
            modifying B does not alter A after B = A.

    Model-user argument: Octave uses value semantics; arrays are independent
    copies after assignment. An engineer who writes B = A and modifies B
    expects A to remain unchanged. Reference semantics would introduce
    aliasing bugs that are invisible and extremely hard to diagnose.

    Decomposition:
      R-POL29-02a: B = A makes an independent copy; modifying B leaves A unchanged.

    Consistency: Single sub-requirement fully tests the copy-on-assign contract.

R-POL29-03: Logical indexing assignment SHALL zero out or replace elements
            matching a boolean mask.

    Model-user argument: Thresholding operations like A(A > 25) = 0 are the
    standard way to clamp or filter data in Octave. If logical masks do not
    work as selectors on the left-hand side, engineers must write explicit
    loops, defeating the purpose of vectorized code.

    Decomposition:
      R-POL29-03a: A(A > 25) = 0 zeros elements greater than 25.
      R-POL29-03b: A(A < 3) = -1 replaces elements less than 3 with -1.

    Consistency: Greater-than (03a) and less-than (03b) masks cover both
    comparison directions.

R-POL29-04: The end keyword SHALL resolve to the last valid index in
            subscript expressions, including arithmetic like end-1.

    Model-user argument: Expressions like A(end-1:end) to get the last two
    elements are idiomatic Octave. If end does not resolve correctly inside
    subscripts, indexing code silently reads the wrong elements.

    Decomposition:
      R-POL29-04a: A(end-1:end) returns the last two elements.
      R-POL29-04b: A(end, end) returns the bottom-right element of a 2D matrix.

    Consistency: 1D range with end (04a) and 2D subscript with end (04b)
    cover the primary use cases.

R-POL29-05: Tilde (~) in multi-output assignment SHALL discard the
            corresponding output without error.

    Model-user argument: Octave users write [~, idx] = sort(v) to discard
    the sorted values and keep only the indices. If tilde is not supported,
    engineers must create throwaway variables, cluttering the workspace.

    Decomposition:
      R-POL29-05a: [~, idx] = sort(...) discards sorted values, keeps indices.
      R-POL29-05b: [~, ~, v] = find(...) discards first two outputs.

    Consistency: Single tilde (05a) and double tilde (05b) cover the pattern.

R-POL29-06: Operator precedence SHALL follow Octave rules for power,
            unary minus, multiplication/addition, and element-wise power.

    Model-user argument: An engineer typing 2^3^2 expects 64 (left-
    associative, as in Octave), not 512 (right-associative). Incorrect
    precedence silently produces wrong numerical results in formulas.

    Decomposition:
      R-POL29-06a: 2^3^2 = 64 (left-associative).
      R-POL29-06b: -2^2 = -4 (power before unary minus).
      R-POL29-06c: 3*4 + 5*6 = 42 (multiplication before addition).
      R-POL29-06d: 2 .^ [1 2 3] = [2 4 8] (element-wise broadcast).

    Consistency: Associativity (06a), unary-vs-power (06b), mul-vs-add
    (06c), and element-wise broadcast (06d) cover the critical precedence
    rules.

R-POL29-07: String indexing and case-conversion functions SHALL work
            correctly on ForgeChar values.

    Model-user argument: Extracting substrings via s(7:end) and converting
    case with toupper/tolower are basic string operations that every Octave
    script uses. Incorrect behavior breaks text processing workflows.

    Decomposition:
      R-POL29-07a: s(7:end) on 'hello world' gives 'world'.
      R-POL29-07b: toupper('hello') returns 'HELLO'.
      R-POL29-07c: tolower('HELLO') returns 'hello'.

    Consistency: Substring extraction (07a) and both case-conversion
    directions (07b-c) are covered.
"""
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
    """R-POL29-01: Matrix row/column/element deletion via assignment to []
    SHALL remove the specified elements and reshape the result correctly.

    Model-user argument: Octave engineers routinely delete matrix rows with
    A(2,:) = [] or filter elements with A([2 4]) = []. If deletion semantics
    differ, data-cleaning scripts produce arrays of wrong shape, leading to
    dimension-mismatch errors downstream.

    Decomposition:
      R-POL29-01a: A(2,:) = [] deletes the second row.
      R-POL29-01b: A([2 4]) = [] removes elements at those positions.
      R-POL29-01c: A(:,2) = [] deletes the second column.
      R-POL29-01d: A(:) = [] produces an empty matrix.

    Consistency: Row (01a), indexed element (01b), column (01c), and
    full-clear (01d) deletions are all covered.
    """

    def test_delete_row(self, s):
        """R-POL29-01a: A(2,:) = [] SHALL delete the second row."""
        s.eval("A = [1 2; 3 4]; A(2,:) = []")
        r = _unwrap(s.eval("A"))
        np.testing.assert_array_equal(r, [[1, 2]])

    def test_delete_elements_by_index(self, s):
        """R-POL29-01b: A([2 4]) = [] SHALL remove elements at positions 2 and 4."""
        s.eval("A = [1 2 3 4 5]; A([2 4]) = []")
        r = _unwrap(s.eval("A"))
        np.testing.assert_array_equal(r.ravel(), [1, 3, 5])

    def test_delete_column(self, s):
        """R-POL29-01c: A(:,2) = [] SHALL delete the second column."""
        s.eval("A = [1 2 3; 4 5 6]; A(:,2) = []")
        r = _unwrap(s.eval("A"))
        np.testing.assert_array_equal(r, [[1, 3], [4, 6]])

    def test_delete_all_elements(self, s):
        """R-POL29-01d: A(:) = [] SHALL produce an empty matrix."""
        s.eval("A = [1 2 3]; A(:) = []")
        r = _unwrap(s.eval("A"))
        assert r.size == 0


# ── 1c. Value semantics (copy-on-assign) ────────────────────────────
class TestValueSemantics:
    """R-POL29-02: Assignment SHALL use value semantics (copy-on-assign) so
    that modifying B does not alter A after B = A.

    Model-user argument: Octave uses value semantics; arrays are independent
    copies after assignment. An engineer who writes B = A and modifies B
    expects A to remain unchanged. Reference semantics would introduce
    aliasing bugs that are invisible and extremely hard to diagnose.

    Decomposition:
      R-POL29-02a: B = A makes an independent copy.

    Consistency: Single sub-requirement fully tests the copy-on-assign contract.
    """

    def test_assign_copies_array(self, s):
        """R-POL29-02a: B = A SHALL make an independent copy; modifying B leaves A unchanged."""
        s.eval("A = [1 2; 3 4]; B = A; B(1,1) = 99")
        a = _unwrap(s.eval("A"))
        b = _unwrap(s.eval("B"))
        np.testing.assert_array_equal(a, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(b, [[99, 2], [3, 4]])


# ── 2. Logical indexing assignment ───────────────────────────────────
class TestLogicalIndexingAssign:
    """R-POL29-03: Logical indexing assignment SHALL zero out or replace
    elements matching a boolean mask.

    Model-user argument: Thresholding operations like A(A > 25) = 0 are the
    standard way to clamp or filter data in Octave. If logical masks do not
    work as selectors on the left-hand side, engineers must write explicit
    loops, defeating the purpose of vectorized code.

    Decomposition:
      R-POL29-03a: A(A > 25) = 0 zeros elements greater than 25.
      R-POL29-03b: A(A < 3) = -1 replaces elements less than 3 with -1.

    Consistency: Greater-than (03a) and less-than (03b) masks cover both
    comparison directions.
    """

    def test_logical_mask_zero_out(self, s):
        """R-POL29-03a: A(A > 25) = 0 SHALL zero elements greater than 25."""
        s.eval("A = [10 20 30 40 50]; A(A > 25) = 0")
        r = _unwrap(s.eval("A"))
        np.testing.assert_array_equal(r.ravel(), [10, 20, 0, 0, 0])

    def test_logical_mask_negative_replace(self, s):
        """R-POL29-03b: A(A < 3) = -1 SHALL replace elements less than 3 with -1."""
        s.eval("A = [1 2 3; 4 5 6]; A(A < 3) = -1")
        r = _unwrap(s.eval("A"))
        np.testing.assert_array_equal(r, [[-1, -1, 3], [4, 5, 6]])


# ── 3. End keyword in expressions ───────────────────────────────────
class TestEndKeyword:
    """R-POL29-04: The end keyword SHALL resolve to the last valid index in
    subscript expressions, including arithmetic like end-1.

    Model-user argument: Expressions like A(end-1:end) to get the last two
    elements are idiomatic Octave. If end does not resolve correctly inside
    subscripts, indexing code silently reads the wrong elements.

    Decomposition:
      R-POL29-04a: A(end-1:end) returns the last two elements.
      R-POL29-04b: A(end, end) returns the bottom-right element.

    Consistency: 1D range with end (04a) and 2D subscript with end (04b)
    cover the primary use cases.
    """

    def test_end_minus_1_colon_end(self, s):
        """R-POL29-04a: A(end-1:end) SHALL return the last two elements."""
        r = s.eval("A = [10 20 30 40 50]; A(end-1:end)")
        np.testing.assert_array_equal(_unwrap(r).ravel(), [40, 50])

    def test_end_2d(self, s):
        """R-POL29-04b: A(end, end) SHALL return bottom-right element of a matrix."""
        r = s.eval("A = [1 2 3; 4 5 6]; A(end, end)")
        assert float(_unwrap(r).ravel()[0]) == 6.0


# ── 4. Multiple assignment with tilde ────────────────────────────────
class TestTildeMultiAssign:
    """R-POL29-05: Tilde (~) in multi-output assignment SHALL discard the
    corresponding output without error.

    Model-user argument: Octave users write [~, idx] = sort(v) to discard
    the sorted values and keep only the indices. If tilde is not supported,
    engineers must create throwaway variables, cluttering the workspace.

    Decomposition:
      R-POL29-05a: [~, idx] = sort(...) discards sorted values, keeps indices.
      R-POL29-05b: [~, ~, v] = find(...) discards first two outputs.

    Consistency: Single tilde (05a) and double tilde (05b) cover the pattern.
    """

    def test_sort_ignore_first(self, s):
        """R-POL29-05a: [~, idx] = sort([3 1 2]) SHALL discard sorted values."""
        s.eval("[~, idx] = sort([3 1 2])")
        r = _unwrap(s.eval("idx"))
        np.testing.assert_array_equal(r.ravel(), [2, 3, 1])

    def test_find_ignore_first_two(self, s):
        """R-POL29-05b: [~, ~, v] = find([0 3 0 4]) SHALL discard row/col."""
        s.eval("[~, ~, v] = find([0 3 0 4])")
        r = _unwrap(s.eval("v"))
        np.testing.assert_array_equal(r.ravel(), [3, 4])


# ── 5. Operator precedence ──────────────────────────────────────────
class TestOperatorPrecedence:
    """R-POL29-06: Operator precedence SHALL follow Octave rules for power,
    unary minus, multiplication/addition, and element-wise power.

    Model-user argument: An engineer typing 2^3^2 expects 64 (left-
    associative, as in Octave), not 512 (right-associative). Incorrect
    precedence silently produces wrong numerical results in formulas.

    Decomposition:
      R-POL29-06a: 2^3^2 = 64 (left-associative).
      R-POL29-06b: -2^2 = -4 (power before unary minus).
      R-POL29-06c: 3*4 + 5*6 = 42 (multiplication before addition).
      R-POL29-06d: 2 .^ [1 2 3] = [2 4 8] (element-wise broadcast).

    Consistency: Associativity (06a), unary-vs-power (06b), mul-vs-add
    (06c), and element-wise broadcast (06d) cover the critical precedence
    rules.
    """

    def test_power_left_associative(self, s):
        """R-POL29-06a: 2^3^2 SHALL equal 64 (left-associative)."""
        r = s.eval("2^3^2")
        assert float(_unwrap(r).ravel()[0]) == 64.0

    def test_unary_minus_after_power(self, s):
        """R-POL29-06b: -2^2 SHALL equal -4 (power before unary minus)."""
        r = s.eval("-2^2")
        assert float(_unwrap(r).ravel()[0]) == -4.0

    def test_mul_add_precedence(self, s):
        """R-POL29-06c: 3*4 + 5*6 SHALL equal 42."""
        r = s.eval("3 * 4 + 5 * 6")
        assert float(_unwrap(r).ravel()[0]) == 42.0

    def test_elementwise_power_broadcast(self, s):
        """R-POL29-06d: 2 .^ [1 2 3] SHALL equal [2 4 8]."""
        r = s.eval("2 .^ [1 2 3]")
        np.testing.assert_array_equal(_unwrap(r).ravel(), [2, 4, 8])


# ── 6. String functions ──────────────────────────────────────────────
class TestStringFunctions:
    """R-POL29-07: String indexing and case-conversion functions SHALL work
    correctly on ForgeChar values.

    Model-user argument: Extracting substrings via s(7:end) and converting
    case with toupper/tolower are basic string operations that every Octave
    script uses. Incorrect behavior breaks text processing workflows.

    Decomposition:
      R-POL29-07a: s(7:end) on 'hello world' gives 'world'.
      R-POL29-07b: toupper('hello') returns 'HELLO'.
      R-POL29-07c: tolower('HELLO') returns 'hello'.

    Consistency: Substring extraction (07a) and both case-conversion
    directions (07b-c) are covered.
    """

    def test_string_indexing_with_end(self, s):
        """R-POL29-07a: s(7:end) on 'hello world' SHALL give 'world'."""
        r = s.eval("s = 'hello world'; s(7:end)")
        assert isinstance(r, ForgeChar)
        assert r.to_str() == "world"

    def test_toupper(self, s):
        """R-POL29-07b: toupper('hello') SHALL return 'HELLO'."""
        r = s.eval("toupper('hello')")
        assert isinstance(r, ForgeChar)
        assert r.to_str() == "HELLO"

    def test_tolower(self, s):
        """R-POL29-07c: tolower('HELLO') SHALL return 'hello'."""
        r = s.eval("tolower('HELLO')")
        assert isinstance(r, ForgeChar)
        assert r.to_str() == "hello"
