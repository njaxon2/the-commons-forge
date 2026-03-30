# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for improved indexing, colon handling, and format."""
import pytest
import numpy as np
from forge.engine.evaluator import Session, ForgeError
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def s():
    return Session()


class TestBareColonIndexing:
    def test_flatten_with_colon(self, s):
        """A(:) should flatten to column vector."""
        s.eval("A = [1 2; 3 4]")
        s.eval("v = A(:)")
        v = _unwrap(s.workspace.get("v"))
        # Column-major order: [1; 3; 2; 4]
        assert v.ravel().tolist() == [1.0, 3.0, 2.0, 4.0]

    def test_colon_row_select(self, s):
        """A(:, 2) should select all rows of column 2."""
        s.eval("A = [1 2 3; 4 5 6; 7 8 9]")
        s.eval("v = A(:, 2)")
        v = _unwrap(s.workspace.get("v"))
        np.testing.assert_allclose(v.ravel(), [2, 5, 8])

    def test_colon_col_select(self, s):
        """A(2, :) should select all columns of row 2."""
        s.eval("A = [1 2 3; 4 5 6; 7 8 9]")
        s.eval("v = A(2, :)")
        v = _unwrap(s.workspace.get("v"))
        np.testing.assert_allclose(v.ravel(), [4, 5, 6])


class TestVectorizedIndexing:
    def test_array_index(self, s):
        """x([1 3 5]) should return elements at those positions."""
        s.eval("x = [10 20 30 40 50]")
        s.eval("y = x([1 3 5])")
        y = _unwrap(s.workspace.get("y"))
        np.testing.assert_allclose(y.ravel(), [10, 30, 50])

    def test_logical_index(self, s):
        """x(x > 3) should return elements > 3."""
        s.eval("x = [1 2 3 4 5]")
        s.eval("y = x(x > 3)")
        y = _unwrap(s.workspace.get("y"))
        np.testing.assert_allclose(y.ravel(), [4, 5])


class TestLogicalShortCircuit:
    def test_scalar_and_true(self, s):
        s.eval("r = 1 && 1")
        r = _unwrap(s.workspace.get("r"))
        assert bool(r)

    def test_scalar_and_false(self, s):
        s.eval("r = 1 && 0")
        r = _unwrap(s.workspace.get("r"))
        assert not bool(r)

    def test_scalar_or_shortcircuit(self, s):
        s.eval("r = 1 || 0")
        r = _unwrap(s.workspace.get("r"))
        assert bool(r)

    def test_array_and_errors(self, s):
        """[1 2] && [3 4] should raise error (non-scalar operand)."""
        with pytest.raises((ForgeError, Exception)):
            s.eval("[1 2] && [3 4]")

    def test_elementwise_and_on_arrays(self, s):
        """[1 0] & [1 1] should work element-wise."""
        s.eval("r = [1 0] & [1 1]")
        r = _unwrap(s.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [1, 0])


class TestFormatCommand:
    def test_format_default(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval("format()")
        assert "short" in result

    def test_format_display(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval("x = 3.14159265358979")
        r1 = s.eval("disp(x)")
        # Default short format
        assert "3.14159" in r1 or "3.1416" in r1
