"""Tests for R-38: struct array correct size, numel, length, and class.

Unit requirements:
  R-38.1: size() of N-element struct array returns [1, N]
  R-38.2: numel() of N-element struct array returns N
  R-38.3: length() of N-element struct array returns N
  R-38.4: class() of struct array returns 'struct'
  R-38.5: whos displays struct array as 'struct' type
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/forge"))

import pytest
from forge.engine.session import ForgeSession


@pytest.fixture
def session():
    return ForgeSession()


class TestStructArrayDimensions:
    """R-38: struct array size reporting."""

    def test_size_two_element_struct_array(self, session):
        """R-38.1: size() returns [1, 2] for 2-element struct array."""
        r = session.eval("s(1).x=10; s(2).x=20; size(s)")
        assert "1" in r and "2" in r, f"Expected 1x2, got: {r}"

    def test_numel_three_element_struct_array(self, session):
        """R-38.2: numel() returns 3 for 3-element struct array."""
        r = session.eval("s(1).x=1; s(2).x=2; s(3).x=3; numel(s)")
        assert "3" in r, f"Expected 3, got: {r}"

    def test_length_four_element_struct_array(self, session):
        """R-38.3: length() returns 4 for 4-element struct array."""
        r = session.eval("s(1).v=1; s(2).v=2; s(3).v=3; s(4).v=4; length(s)")
        assert "4" in r, f"Expected 4, got: {r}"

    def test_class_struct_array_is_struct(self, session):
        """R-38.4: class() returns 'struct' for struct arrays."""
        r = session.eval("s(1).x=1; s(2).x=2; class(s)")
        assert "struct" in r, f"Expected 'struct', got: {r}"

    def test_single_struct_size_unchanged(self, session):
        """R-38.1 regression: single struct via dot-assign is still 1x1."""
        r = session.eval("s.x=1; s.y=2; size(s)")
        assert "1" in r, f"Expected 1x1, got: {r}"

    def test_struct_array_field_access_correct(self, session):
        """R-38: field access at index 2 returns correct value."""
        r = session.eval("s(1).x=99; s(2).x=42; s(2).x")
        assert "42" in r, f"Expected 42, got: {r}"

    def test_whos_shows_struct_for_struct_array(self, session):
        """R-38.5: whos shows 'struct' type for struct arrays."""
        session.eval("s(1).x=1; s(2).x=2")
        r = session.eval("whos s")
        assert "struct" in r, f"Expected struct in whos output, got: {r}"
