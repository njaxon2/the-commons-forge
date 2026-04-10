"""Tests for R-35: Comma as statement separator.

In Octave, commas act as non-suppressing statement terminators at the
statement level. These tests verify that Forge now accepts comma-separated
statements at top level and inside control flow headers and bodies.
"""
import pytest
from forge.engine.session import ForgeSession


# ---------------------------------------------------------------------------
# R-35a: Comma between top-level statements
# ---------------------------------------------------------------------------

def test_comma_top_level_two_statements():
    """R-35a: a=1, b=2 evaluates both assignments without ParseError."""
    s = ForgeSession()
    result = s.eval("a=1, b=2")
    assert "ParseError" not in result
    # a and b should be set
    val_a = s.eval("a")
    val_b = s.eval("b")
    assert "1" in str(val_a)
    assert "2" in str(val_b)


def test_comma_top_level_output_not_suppressed():
    """R-35d: comma does not suppress output (unlike semicolon)."""
    s = ForgeSession()
    result = s.eval("x=42, y=99")
    # With comma, values should be printed
    assert "42" in result
    assert "99" in result


def test_semicolon_still_suppresses():
    """R-35d: semicolon still suppresses output (regression guard)."""
    s = ForgeSession()
    result = s.eval("x=42; y=99;")
    assert "42" not in result
    assert "99" not in result


# ---------------------------------------------------------------------------
# R-35b: Comma after control flow header
# ---------------------------------------------------------------------------

def test_for_comma_header():
    """R-35b: for i=1:3, disp(i), end accepted as valid syntax."""
    s = ForgeSession()
    result = s.eval("for i=1:3, disp(i), end")
    assert "ParseError" not in result
    assert "1" in result
    assert "2" in result
    assert "3" in result


def test_if_comma_header():
    """R-35b: if true, disp(1), end accepted as valid syntax."""
    s = ForgeSession()
    result = s.eval("if true, disp(42), end")
    assert "ParseError" not in result
    assert "42" in result


def test_while_comma_header():
    """R-35b: while x>0, x=x-1, end accepted as valid syntax."""
    s = ForgeSession()
    result = s.eval("x=3; while x>0, x=x-1; end; x")
    assert "ParseError" not in result
    assert "0" in result


def test_try_comma_header():
    """R-35b: try, error(...), catch e, disp(...), end is accepted."""
    s = ForgeSession()
    result = s.eval("try, error('caught me'), catch e, disp(e.message), end")
    assert "ParseError" not in result
    assert "caught me" in result


# ---------------------------------------------------------------------------
# R-35c: Comma between statements inside body
# ---------------------------------------------------------------------------

def test_for_body_comma_separated():
    """R-35c: for loop body with comma-separated statements."""
    s = ForgeSession()
    result = s.eval("s=0; for i=1:5, s=s+i, end")
    assert "ParseError" not in result
    # s should equal 15 after the loop
    val = s.eval("s")
    assert "15" in str(val)


# ---------------------------------------------------------------------------
# R-35e: Regression - commas inside expressions unaffected
# ---------------------------------------------------------------------------

def test_matrix_commas_unaffected():
    """R-35e: commas inside matrix literals still work."""
    s = ForgeSession()
    result = s.eval("v = [1, 2, 3]")
    assert "ParseError" not in result
    assert "1" in result and "2" in result and "3" in result


def test_function_call_commas_unaffected():
    """R-35e: commas inside function call arguments still work."""
    s = ForgeSession()
    result = s.eval("v = max([3, 1, 2])")
    assert "ParseError" not in result
    assert "3" in result


def test_multi_return_commas_unaffected():
    """R-35e: commas in multi-return assignment [a,b]=f() still work."""
    s = ForgeSession()
    result = s.eval("[mn, idx] = min([5, 2, 8])")
    assert "ParseError" not in result
    # mn should be 2
    mn_val = s.eval("mn")
    assert "2" in str(mn_val)
