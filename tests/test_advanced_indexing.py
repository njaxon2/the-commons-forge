# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for improved indexing, colon handling, and format.

Requirement R-IDX: The engine SHALL support advanced matrix indexing
operations including bare colon flattening, vectorized subscripting,
logical indexing, short-circuit operators, and display formatting,
matching Octave/MATLAB behavior for all standard subscripting patterns.

Model-user argument: An engineer migrating from MATLAB relies on concise
indexing idioms (A(:), A(:,2), x(x>3), x([1 3 5])) to extract and
reshape subsets of experimental data. If any of these patterns silently
produce wrong results or raise unexpected errors, the engineer will
distrust Forge output and revert to the incumbent tool.

Decomposition:
  R-IDX-01..03: Bare colon indexing (flatten, row select, column select)
  R-IDX-04..05: Vectorized and logical indexing
  R-IDX-06..10: Short-circuit and element-wise logical operators
  R-IDX-11..12: Format/display commands

Consistency argument: Bare colon tests (R-IDX-01..03) cover column-major
flattening and single-dimension selection. Vectorized and logical tests
(R-IDX-04..05) cover multi-element subscript arrays and boolean masks.
Logical operator tests (R-IDX-06..10) cover scalar short-circuit, array
element-wise, and error cases. Format tests (R-IDX-11..12) cover display
reset and numeric rendering. Together these span the indexing and display
surface the engineer encounters in interactive data exploration.
"""
import pytest
import numpy as np
from forge.engine.evaluator import Session, ForgeError
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def s():
    return Session()


class TestBareColonIndexing:
    """R-IDX-01..03: Bare colon indexing SHALL flatten or select entire
    rows/columns in column-major order, matching Octave semantics.

    Model-user argument: The engineer uses A(:) to linearize a matrix for
    plotting or statistical analysis, and A(:,k) or A(k,:) to pull
    individual rows or columns. These are among the most common indexing
    patterns in MATLAB scripts.

    Decomposition:
      R-IDX-01: A(:) flattens to column-major vector
      R-IDX-02: A(:, k) selects all rows of column k
      R-IDX-03: A(k, :) selects all columns of row k

    Consistency: These three sub-requirements cover the three bare-colon
    usage patterns (full flatten, column select, row select) that
    exhaust the colon-as-subscript grammar.
    """

    def test_flatten_with_colon(self, s):
        """R-IDX-01: A(:) flattens to column-major vector."""
        s.eval("A = [1 2; 3 4]")
        s.eval("v = A(:)")
        v = _unwrap(s.workspace.get("v"))
        # Column-major order: [1; 3; 2; 4]
        assert v.ravel().tolist() == [1.0, 3.0, 2.0, 4.0]

    def test_colon_row_select(self, s):
        """R-IDX-02: A(:, k) selects all rows of column k."""
        s.eval("A = [1 2 3; 4 5 6; 7 8 9]")
        s.eval("v = A(:, 2)")
        v = _unwrap(s.workspace.get("v"))
        np.testing.assert_allclose(v.ravel(), [2, 5, 8])

    def test_colon_col_select(self, s):
        """R-IDX-03: A(k, :) selects all columns of row k."""
        s.eval("A = [1 2 3; 4 5 6; 7 8 9]")
        s.eval("v = A(2, :)")
        v = _unwrap(s.workspace.get("v"))
        np.testing.assert_allclose(v.ravel(), [4, 5, 6])


class TestVectorizedIndexing:
    """R-IDX-04..05: Vectorized and logical indexing SHALL return the
    correct subset of elements.

    Model-user argument: The engineer frequently uses integer-array
    subscripts (x([1 3 5])) to cherry-pick samples, and logical masks
    (x(x>3)) to filter data by threshold. Both patterns must produce
    results identical to MATLAB/Octave.

    Decomposition:
      R-IDX-04: Integer-array subscript returns elements at listed positions
      R-IDX-05: Logical mask subscript returns elements satisfying condition

    Consistency: Integer-array and logical-mask indexing are the two
    non-scalar, non-colon subscripting modes in the M-language grammar.
    """

    def test_array_index(self, s):
        """R-IDX-04: x([1 3 5]) returns elements at those 1-based positions."""
        s.eval("x = [10 20 30 40 50]")
        s.eval("y = x([1 3 5])")
        y = _unwrap(s.workspace.get("y"))
        np.testing.assert_allclose(y.ravel(), [10, 30, 50])

    def test_logical_index(self, s):
        """R-IDX-05: x(x > 3) returns elements satisfying the condition."""
        s.eval("x = [1 2 3 4 5]")
        s.eval("y = x(x > 3)")
        y = _unwrap(s.workspace.get("y"))
        np.testing.assert_allclose(y.ravel(), [4, 5])


class TestLogicalShortCircuit:
    """R-IDX-06..10: Short-circuit (&&, ||) and element-wise (&) logical
    operators SHALL behave per Octave rules: short-circuit requires scalar
    operands; element-wise works on arrays.

    Model-user argument: The engineer uses && and || in if-guards for
    scalar conditions (e.g., if n > 0 && x(n) < tol). Accidentally using
    && on arrays must raise an error, not silently broadcast, because
    silent broadcasting would mask logical bugs in control flow.

    Decomposition:
      R-IDX-06: Scalar && with both true yields true
      R-IDX-07: Scalar && with false RHS yields false
      R-IDX-08: Scalar || short-circuits on true LHS
      R-IDX-09: Array && raises error (non-scalar operand)
      R-IDX-10: Element-wise & on arrays produces element-wise result

    Consistency: R-IDX-06..08 cover the truth-table branches for scalar
    short-circuit operators. R-IDX-09 tests the error guard for non-scalar
    operands. R-IDX-10 confirms the array alternative (&) works correctly.
    """

    def test_scalar_and_true(self, s):
        """R-IDX-06: Scalar 1 && 1 yields true."""
        s.eval("r = 1 && 1")
        r = _unwrap(s.workspace.get("r"))
        assert bool(r)

    def test_scalar_and_false(self, s):
        """R-IDX-07: Scalar 1 && 0 yields false."""
        s.eval("r = 1 && 0")
        r = _unwrap(s.workspace.get("r"))
        assert not bool(r)

    def test_scalar_or_shortcircuit(self, s):
        """R-IDX-08: Scalar 1 || 0 short-circuits to true."""
        s.eval("r = 1 || 0")
        r = _unwrap(s.workspace.get("r"))
        assert bool(r)

    def test_array_and_errors(self, s):
        """R-IDX-09: Array && array raises error for non-scalar operands."""
        with pytest.raises((ForgeError, Exception)):
            s.eval("[1 2] && [3 4]")

    def test_elementwise_and_on_arrays(self, s):
        """R-IDX-10: Element-wise & on arrays produces per-element result."""
        s.eval("r = [1 0] & [1 1]")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [1, 0])


class TestFormatCommand:
    """R-IDX-11..12: The format command SHALL reset display precision and
    disp() SHALL render numeric values in the active format.

    Model-user argument: The engineer switches between format short and
    format long while inspecting results in the command window. After
    calling format() (reset), pi must display as 3.1416, not full
    precision, so the output matches their MATLAB muscle memory.

    Decomposition:
      R-IDX-11: format() resets to short and pi displays as 3.1416
      R-IDX-12: disp(x) renders x in the current format setting

    Consistency: These two tests cover the format reset behavior and the
    disp rendering path, which are the two user-facing display mechanisms.
    """

    def test_format_default(self):
        """R-IDX-11: format() resets to short; pi displays as 3.1416."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval("format()")
        # format() resets to short; verify by checking pi display
        r = s.eval("pi")
        assert "3.1416" in str(r)

    def test_format_display(self):
        """R-IDX-12: disp(x) renders x in the active format setting."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval("x = 3.14159265358979")
        r1 = s.eval("disp(x)")
        # Default short format
        assert "3.14159" in r1 or "3.1416" in r1
