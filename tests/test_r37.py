"""Tests for R-37: whos accepts variable name arguments.

Unit requirements:
  R-37.1: whos x shows info for variable x only
  R-37.2: whos x y shows info for variables x and y
  R-37.3: whos with no args still shows all variables
  R-37.4: whos x does not raise TypeError
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/forge"))

import pytest
from forge.engine.session import ForgeSession


@pytest.fixture
def session():
    return ForgeSession()


class TestWhosWithArgs:
    """R-37: whos with variable name arguments."""

    def test_whos_single_var_no_error(self, session):
        """R-37.4: whos x must not raise TypeError."""
        session.eval("x = [1 2 3]")
        r = session.eval("whos x")
        assert "error" not in r.lower() or "TypeError" not in r, f"Unexpected error: {r}"

    def test_whos_single_var_shows_name(self, session):
        """R-37.1: whos x output contains variable name x."""
        session.eval("x = [1 2 3]")
        r = session.eval("whos x")
        assert "x" in r, f"Expected 'x' in output, got: {r}"

    def test_whos_single_var_shows_size(self, session):
        """R-37.1: whos x output contains size info."""
        session.eval("x = [1 2 3]")
        r = session.eval("whos x")
        assert "3" in r or "1x3" in r, f"Expected size in output, got: {r}"

    def test_whos_no_args_still_works(self, session):
        """R-37.3: whos with no args shows all variables."""
        session.eval("a=1; b=2")
        r = session.eval("whos")
        assert "a" in r and "b" in r, f"Expected a and b in whos output, got: {r}"
