"""test_polish_r17.py - Error handling, rethrow, MException, assert, narginchk/nargoutchk.

Covers:
  - try/catch with rethrow
  - narginchk / nargoutchk argument validation
  - assert (single, comparison, tolerance, array)
  - error/warning with printf-style formatting
  - MException class creation and field access
"""

import pytest
from forge.engine.session import ForgeSession


@pytest.fixture
def S():
    return ForgeSession()


# ---- rethrow ----------------------------------------------------------------

def test_rethrow_propagates_error(S):
    """rethrow(e) should re-raise the caught error."""
    r = S.eval("try; error('myid:sub', 'bad %d', 42); catch e; rethrow(e); end")
    assert "error" in r.lower() and "bad 42" in r


def test_rethrow_preserves_identifier(S):
    """rethrow should preserve the original identifier."""
    r = S.eval(
        "try; error('pkg:func', 'fail'); catch e; "
        "try; rethrow(e); catch e2; disp(e2.identifier); end; end"
    )
    assert "pkg:func" in r


# ---- narginchk / nargoutchk ------------------------------------------------

def test_narginchk_pass(S):
    """narginchk should pass when arg count is within range."""
    S.eval("function f(x, y); narginchk(1, 3); end")
    r = S.eval("f(1, 2)")
    assert "error" not in str(r).lower() or r is None or r == ""


def test_narginchk_within_range(S):
    """narginchk(1,3) should pass when called with 2 args."""
    S.eval("function g2(x, y); narginchk(1, 3); disp(x + y); end")
    r = S.eval("g2(3, 4)")
    assert "7" in str(r)


def test_nargoutchk_pass(S):
    """nargoutchk basic call should not error."""
    S.eval("function [a, b] = h(x); nargoutchk(0, 2); a = x; b = x+1; end")
    r = S.eval("[p, q] = h(5); disp(p)")
    assert "5" in str(r)


# ---- assert -----------------------------------------------------------------

def test_assert_true(S):
    """assert(true) should pass silently."""
    r = S.eval("assert(true)")
    assert "error" not in str(r).lower() or r is None or r == ""


def test_assert_false_errors(S):
    """assert(false) should raise an error."""
    r = S.eval("assert(false)")
    assert "error" in str(r).lower()


def test_assert_comparison_equal(S):
    """assert(obs, exp) with equal arrays should pass."""
    r = S.eval("assert([1 2 3], [1 2 3])")
    assert "error" not in str(r).lower() or r is None or r == ""


def test_assert_tolerance_pass(S):
    """assert(obs, exp, tol) within tolerance should pass."""
    r = S.eval("assert(1.0001, 1.0, 0.01)")
    assert "error" not in str(r).lower() or r is None or r == ""


def test_assert_tolerance_fail(S):
    """assert(obs, exp, tol) outside tolerance should error."""
    r = S.eval("assert(1.5, 1.0, 0.01)")
    assert "error" in str(r).lower()


# ---- error / warning with printf format -------------------------------------

def test_error_printf_format(S):
    """error('fmt', args...) should produce formatted message."""
    r = S.eval("try; error('Value is %d not %d', 5, 10); catch e; disp(e.message); end")
    assert "Value is 5 not 10" in r


def test_warning_printf_format(S):
    """warning('fmt', args...) should not crash and format the string."""
    r = S.eval("warning('Score is %d', 99)")
    assert "error" not in str(r).lower() or r is None or r == ""


# ---- MException --------------------------------------------------------------

def test_MException_create(S):
    """MException(id, msg) should create struct with identifier and message."""
    S.eval("e = MException('test:err', 'something broke')")
    r_id = S.eval("disp(e.identifier)")
    r_msg = S.eval("disp(e.message)")
    assert "test:err" in r_id
    assert "something broke" in r_msg


def test_MException_printf(S):
    """MException(id, fmt, args...) should format the message."""
    S.eval("e = MException('calc:overflow', 'val=%d max=%d', 999, 100)")
    r = S.eval("disp(e.message)")
    assert "val=999" in r and "max=100" in r


def test_MException_rethrow(S):
    """Create MException, throw via error(), catch, rethrow, catch again."""
    code = (
        "me = MException('io:file', 'not found'); "
        "try; error(me.message); catch e1; "
        "  try; rethrow(e1); catch e2; disp(e2.message); end; "
        "end"
    )
    r = S.eval(code)
    assert "not found" in r
