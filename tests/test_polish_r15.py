# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Polish R15 -- evaluator feature tests.

Covers:
  a. try/catch with error object (.message, .identifier)
  b. Nested function calls
  c. Logical indexing
  d. String comparison in switch (cell array of cases)
  e. Augmented assignment (row slice)
"""
import numpy as np
import pytest
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


def ws(s, name):
    return s.workspace.get(name)


def to_np(val):
    """Convert ForgeArray to flat numpy array."""
    return np.array(val._data).flatten()


# -- a. try/catch with error object --

class TestTryCatchErrorObject:
    def test_catch_message(self, s):
        s.eval("try; error('myid:foo', 'oops'); catch e; r = e.message; end")
        assert ws(s, "r").to_str() == "oops"

    def test_catch_identifier(self, s):
        s.eval("try; error('myid:foo', 'oops'); catch e; r = e.identifier; end")
        assert ws(s, "r").to_str() == "myid:foo"

    def test_catch_no_error(self, s):
        s.eval("try; x = 5; catch e; x = -1; end")
        assert float(ws(s, "x")) == 5.0

    def test_catch_generic_error(self, s):
        s.eval("try; error('something broke'); catch e; r = e.message; end")
        assert "something broke" in ws(s, "r").to_str()


# -- b. Nested function calls --

class TestNestedFunctionCalls:
    def test_max_min(self, s):
        s.eval("x = max(min([3 1 2]), 5)")
        assert float(ws(s, "x")) == 5.0

    def test_sum_abs(self, s):
        s.eval("x = sum(abs([-1 -2 3]))")
        assert float(ws(s, "x")) == 6.0

    def test_length_of_zeros(self, s):
        s.eval("x = length(zeros(1, 7))")
        assert float(ws(s, "x")) == 7.0


# -- c. Logical indexing --

class TestLogicalIndexing:
    def test_basic_logical(self, s):
        s.eval("A = [10 20 30 40 50]; B = A(A > 25)")
        np.testing.assert_array_equal(to_np(ws(s, "B")), [30, 40, 50])

    def test_logical_equal(self, s):
        s.eval("v = [1 2 3 2 1]; w = v(v == 2)")
        np.testing.assert_array_equal(to_np(ws(s, "w")), [2, 2])

    def test_logical_combined(self, s):
        s.eval("A = [5 15 25 35]; B = A(A >= 10 & A <= 30)")
        np.testing.assert_array_equal(to_np(ws(s, "B")), [15, 25])


# -- d. String comparison in switch --

class TestSwitchCellString:
    def test_cell_case_match_first(self, s):
        s.eval("switch 'hello'; case {'hello','world'}; r=1; otherwise; r=0; end")
        assert float(ws(s, "r")) == 1.0

    def test_cell_case_match_second(self, s):
        s.eval("switch 'world'; case {'hello','world'}; r=1; otherwise; r=0; end")
        assert float(ws(s, "r")) == 1.0

    def test_cell_case_no_match(self, s):
        s.eval("switch 'nope'; case {'hello','world'}; r=1; otherwise; r=0; end")
        assert float(ws(s, "r")) == 0.0

    def test_numeric_cell_case(self, s):
        s.eval("switch 3; case {1,2,3}; r=1; otherwise; r=0; end")
        assert float(ws(s, "r")) == 1.0

    def test_simple_string_case(self, s):
        s.eval("switch 'abc'; case 'abc'; r=1; otherwise; r=0; end")
        assert float(ws(s, "r")) == 1.0


# -- e. Augmented assignment (row/col slice) --

class TestAugmentedAssignment:
    def test_row_assign(self, s):
        s.eval("A = [1 2; 3 4]; A(1,:) = [10 20]")
        A = np.array(ws(s, "A")._data)
        np.testing.assert_array_equal(A[0], [10, 20])
        np.testing.assert_array_equal(A[1], [3, 4])

    def test_col_assign(self, s):
        s.eval("A = [1 2; 3 4]; A(:,2) = [50; 60]")
        A = np.array(ws(s, "A")._data)
        np.testing.assert_array_equal(A[:, 1], [50, 60])

    def test_scalar_assign_into_matrix(self, s):
        s.eval("A = zeros(3,3); A(2,2) = 99")
        A = np.array(ws(s, "A")._data)
        assert A[1, 1] == 99.0

    def test_add_assign_slice(self, s):
        s.eval("K = zeros(4,4); ke = [2 -1; -1 2]; K(1:2,1:2) = K(1:2,1:2) + ke")
        K = np.array(ws(s, "K")._data)
        assert K[0, 0] == 2.0
        assert K[0, 1] == -1.0
