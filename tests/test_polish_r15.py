# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Polish R15 -- evaluator feature tests.

Covers:
  a. try/catch with error object (.message, .identifier)
  b. Nested function calls
  c. Logical indexing
  d. String comparison in switch (cell array of cases)
  e. Augmented assignment (row slice)

Requirement R-POL15-01:
    The try/catch construct SHALL capture error objects with .message and
    .identifier fields, and SHALL skip the catch block when no error occurs.

    Model-user argument:
    An engineer migrating from MATLAB/Octave wraps risky I/O and solver
    calls in try/catch blocks, inspecting e.identifier to decide recovery
    strategy (retry, fallback, abort). If the error object is missing
    .identifier or .message, their fault-tolerant scripts lose the ability
    to distinguish error types and degrade to generic catch-all handling.

    Decomposition:
    R-POL15-01a: catch e captures e.message from error('id','msg').
    R-POL15-01b: catch e captures e.identifier from error('id','msg').
    R-POL15-01c: try block with no error skips catch body.
    R-POL15-01d: catch e captures message from error('msg') (no id).

    Consistency argument:
    01a-01b verify structured error fields. 01c verifies the no-error path.
    01d verifies the single-argument error() form. Together they cover
    the full try/catch error-object contract.

Requirement R-POL15-02:
    Nested function calls SHALL evaluate inner calls first and pass results
    to outer calls correctly.

    Model-user argument:
    An engineer writes compact expressions like max(min(x),threshold) to
    clamp values in one line. If nested calls mis-order evaluation or
    silently drop intermediate results, the engineer must refactor concise
    idioms into verbose multi-line temporaries, losing the expressiveness
    that MATLAB/Octave is valued for.

    Decomposition:
    R-POL15-02a: max(min([3 1 2]), 5) evaluates to 5.
    R-POL15-02b: sum(abs([-1 -2 3])) evaluates to 6.
    R-POL15-02c: length(zeros(1,7)) evaluates to 7.

    Consistency argument:
    Each sub-requirement nests two function calls with a different
    combination of aggregation and element-wise functions, covering the
    three most common nesting patterns.

Requirement R-POL15-03:
    Logical indexing SHALL select elements where the condition is true,
    including compound conditions with & (and).

    Model-user argument:
    An engineer filters measurement vectors with expressions like
    data(data > threshold) dozens of times per script. If logical indexing
    silently returns wrong elements or fails on compound conditions, every
    filtering operation in every analysis script is suspect.

    Decomposition:
    R-POL15-03a: A(A > 25) selects elements greater than 25.
    R-POL15-03b: v(v == 2) selects all elements equal to 2.
    R-POL15-03c: A(A >= 10 & A <= 30) selects elements in range.

    Consistency argument:
    01a tests greater-than, 01b tests equality, 01c tests compound
    range condition. Together they cover the three fundamental logical
    indexing patterns.

Requirement R-POL15-04:
    Switch statements SHALL match string values against cell arrays of
    candidate strings and against simple string literals.

    Model-user argument:
    An engineer uses switch/case with cell arrays of strings to dispatch
    on file extensions, sensor types, or command keywords. If cell-case
    matching fails, the otherwise branch executes for valid inputs, causing
    silent misconfiguration in processing pipelines.

    Decomposition:
    R-POL15-04a: switch 'hello' matches case {'hello','world'}.
    R-POL15-04b: switch 'world' matches case {'hello','world'}.
    R-POL15-04c: switch 'nope' falls through to otherwise.
    R-POL15-04d: switch 3 matches case {1,2,3} (numeric cell).
    R-POL15-04e: switch 'abc' matches case 'abc' (simple string).

    Consistency argument:
    01a-01b test matching first and second cell entries. 01c tests the
    otherwise fallback. 01d tests numeric cell matching. 01e tests simple
    string matching. Together they cover all switch/case matching modes.

Requirement R-POL15-05:
    Augmented assignment to matrix slices (row, column, scalar, and
    computed sub-matrices) SHALL update only the targeted elements.

    Model-user argument:
    An engineer assembles global stiffness matrices by accumulating
    element stiffness matrices into sub-blocks: K(dofs, dofs) = K(dofs,
    dofs) + ke. If slice assignment corrupts adjacent entries or ignores
    the addition, finite element assemblies produce wrong results and
    structural analyses become meaningless.

    Decomposition:
    R-POL15-05a: A(1,:) = [10 20] replaces only the first row.
    R-POL15-05b: A(:,2) = [50;60] replaces only the second column.
    R-POL15-05c: A(2,2) = 99 sets a single element in a zero matrix.
    R-POL15-05d: K(1:2,1:2) = K(1:2,1:2) + ke accumulates a sub-block.

    Consistency argument:
    01a tests row slice, 01b tests column slice, 01c tests scalar
    element, 01d tests read-modify-write on a sub-block. Together they
    cover the four assignment patterns used in matrix assembly.
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


# -- a. try/catch with error object (R-POL15-01) --

class TestTryCatchErrorObject:
    """R-POL15-01: try/catch with structured error objects."""

    def test_catch_message(self, s):
        """R-POL15-01a: catch e captures e.message."""
        s.eval("try; error('myid:foo', 'oops'); catch e; r = e.message; end")
        assert ws(s, "r").to_str() == "oops"

    def test_catch_identifier(self, s):
        """R-POL15-01b: catch e captures e.identifier."""
        s.eval("try; error('myid:foo', 'oops'); catch e; r = e.identifier; end")
        assert ws(s, "r").to_str() == "myid:foo"

    def test_catch_no_error(self, s):
        """R-POL15-01c: try with no error skips catch body."""
        s.eval("try; x = 5; catch e; x = -1; end")
        assert float(ws(s, "x")) == 5.0

    def test_catch_generic_error(self, s):
        """R-POL15-01d: catch e captures message from single-arg error()."""
        s.eval("try; error('something broke'); catch e; r = e.message; end")
        assert "something broke" in ws(s, "r").to_str()


# -- b. Nested function calls (R-POL15-02) --

class TestNestedFunctionCalls:
    """R-POL15-02: Nested function call evaluation order."""

    def test_max_min(self, s):
        """R-POL15-02a: max(min([3 1 2]), 5) equals 5."""
        s.eval("x = max(min([3 1 2]), 5)")
        assert float(ws(s, "x")) == 5.0

    def test_sum_abs(self, s):
        """R-POL15-02b: sum(abs([-1 -2 3])) equals 6."""
        s.eval("x = sum(abs([-1 -2 3]))")
        assert float(ws(s, "x")) == 6.0

    def test_length_of_zeros(self, s):
        """R-POL15-02c: length(zeros(1,7)) equals 7."""
        s.eval("x = length(zeros(1, 7))")
        assert float(ws(s, "x")) == 7.0


# -- c. Logical indexing (R-POL15-03) --

class TestLogicalIndexing:
    """R-POL15-03: Logical indexing on numeric arrays."""

    def test_basic_logical(self, s):
        """R-POL15-03a: A(A > 25) selects elements greater than 25."""
        s.eval("A = [10 20 30 40 50]; B = A(A > 25)")
        np.testing.assert_array_equal(to_np(ws(s, "B")), [30, 40, 50])

    def test_logical_equal(self, s):
        """R-POL15-03b: v(v == 2) selects all elements equal to 2."""
        s.eval("v = [1 2 3 2 1]; w = v(v == 2)")
        np.testing.assert_array_equal(to_np(ws(s, "w")), [2, 2])

    def test_logical_combined(self, s):
        """R-POL15-03c: A(A >= 10 & A <= 30) selects range."""
        s.eval("A = [5 15 25 35]; B = A(A >= 10 & A <= 30)")
        np.testing.assert_array_equal(to_np(ws(s, "B")), [15, 25])


# -- d. String comparison in switch (R-POL15-04) --

class TestSwitchCellString:
    """R-POL15-04: Switch/case with cell arrays and string literals."""

    def test_cell_case_match_first(self, s):
        """R-POL15-04a: switch 'hello' matches first cell entry."""
        s.eval("switch 'hello'; case {'hello','world'}; r=1; otherwise; r=0; end")
        assert float(ws(s, "r")) == 1.0

    def test_cell_case_match_second(self, s):
        """R-POL15-04b: switch 'world' matches second cell entry."""
        s.eval("switch 'world'; case {'hello','world'}; r=1; otherwise; r=0; end")
        assert float(ws(s, "r")) == 1.0

    def test_cell_case_no_match(self, s):
        """R-POL15-04c: switch 'nope' falls through to otherwise."""
        s.eval("switch 'nope'; case {'hello','world'}; r=1; otherwise; r=0; end")
        assert float(ws(s, "r")) == 0.0

    def test_numeric_cell_case(self, s):
        """R-POL15-04d: switch 3 matches numeric cell {1,2,3}."""
        s.eval("switch 3; case {1,2,3}; r=1; otherwise; r=0; end")
        assert float(ws(s, "r")) == 1.0

    def test_simple_string_case(self, s):
        """R-POL15-04e: switch 'abc' matches simple string case."""
        s.eval("switch 'abc'; case 'abc'; r=1; otherwise; r=0; end")
        assert float(ws(s, "r")) == 1.0


# -- e. Augmented assignment (row/col slice) (R-POL15-05) --

class TestAugmentedAssignment:
    """R-POL15-05: Slice assignment into matrices."""

    def test_row_assign(self, s):
        """R-POL15-05a: A(1,:) = [10 20] replaces only the first row."""
        s.eval("A = [1 2; 3 4]; A(1,:) = [10 20]")
        A = np.array(ws(s, "A")._data)
        np.testing.assert_array_equal(A[0], [10, 20])
        np.testing.assert_array_equal(A[1], [3, 4])

    def test_col_assign(self, s):
        """R-POL15-05b: A(:,2) = [50;60] replaces only the second column."""
        s.eval("A = [1 2; 3 4]; A(:,2) = [50; 60]")
        A = np.array(ws(s, "A")._data)
        np.testing.assert_array_equal(A[:, 1], [50, 60])

    def test_scalar_assign_into_matrix(self, s):
        """R-POL15-05c: A(2,2) = 99 sets a single element."""
        s.eval("A = zeros(3,3); A(2,2) = 99")
        A = np.array(ws(s, "A")._data)
        assert A[1, 1] == 99.0

    def test_add_assign_slice(self, s):
        """R-POL15-05d: K(1:2,1:2) += ke accumulates a sub-block."""
        s.eval("K = zeros(4,4); ke = [2 -1; -1 2]; K(1:2,1:2) = K(1:2,1:2) + ke")
        K = np.array(ws(s, "K")._data)
        assert K[0, 0] == 2.0
        assert K[0, 1] == -1.0
