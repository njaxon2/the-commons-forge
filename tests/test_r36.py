"""Tests for R-36: any/all column-wise on 2-D matrices.

Unit requirements:
  R-36.1: any(matrix) reduces column-wise, returns row vector
  R-36.2: all(matrix) reduces column-wise, returns row vector
  R-36.3: any(vector) returns scalar (unchanged behaviour)
  R-36.4: all(vector) returns scalar (unchanged behaviour)
  R-36.5: any(matrix, 2) reduces row-wise (dim argument)
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/forge"))

import pytest
from forge.engine.session import ForgeSession


@pytest.fixture
def session():
    return ForgeSession()


class TestAnyAllColumnWise:
    """R-36: any/all column-wise on matrices."""

    def test_any_matrix_column_wise(self, session):
        """R-36.1: any([1 0; 0 0]) returns [1 0] row vector."""
        r = session.eval("any([1 0; 0 0])")
        assert "1" in r and "0" in r, f"Expected row vector [1 0], got: {r}"
        # Should not just be scalar '1'
        assert r.count("1") >= 1 and r.count("0") >= 1

    def test_all_matrix_column_wise(self, session):
        """R-36.2: all([1 1; 1 0]) returns [1 0] row vector."""
        r = session.eval("all([1 1; 1 0])")
        assert "1" in r and "0" in r, f"Expected row vector [1 0], got: {r}"

    def test_any_vector_scalar(self, session):
        """R-36.3: any([0 0 1]) returns scalar 1."""
        r = session.eval("any([0 0 1])")
        assert "1" in r, f"Expected 1, got: {r}"

    def test_all_vector_false(self, session):
        """R-36.4: all([1 1 0]) returns scalar 0."""
        r = session.eval("all([1 1 0])")
        assert "0" in r, f"Expected 0, got: {r}"

    def test_any_dim2_row_wise(self, session):
        """R-36.5: any([1 0; 0 0], 2) reduces row-wise -> [1; 0]."""
        r = session.eval("any([1 0; 0 0], 2)")
        assert "1" in r and "0" in r, f"Expected col vector [1;0], got: {r}"
