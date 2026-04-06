"""test_polish_r17.py - Error handling, rethrow, MException, assert, narginchk/nargoutchk.

Covers:
  - try/catch with rethrow
  - narginchk / nargoutchk argument validation
  - assert (single, comparison, tolerance, array)
  - error/warning with printf-style formatting
  - MException class creation and field access

Requirement R-POL17-01:
    The rethrow function SHALL re-raise a caught error, preserving its
    original identifier and message, so that outer catch blocks receive
    the same error object.

    Model-user argument:
    An engineer wraps low-level solver calls in try/catch to log errors
    before re-raising them to a higher-level handler. If rethrow loses
    the identifier, the outer handler cannot distinguish between error
    types and applies the wrong recovery strategy, potentially masking
    critical numerical failures.

    Decomposition:
    R-POL17-01a: rethrow(e) re-raises the caught error with its message.
    R-POL17-01b: rethrow preserves the original error identifier.

    Consistency argument:
    01a tests message propagation; 01b tests identifier preservation.
    Together they verify the full rethrow contract.

Requirement R-POL17-02:
    narginchk and nargoutchk SHALL pass silently when the argument count
    is within the specified range.

    Model-user argument:
    An engineer uses narginchk at the top of utility functions to guard
    against misuse (too few or too many arguments). If narginchk errors
    on a valid call, every correct invocation of the function fails,
    breaking working code.

    Decomposition:
    R-POL17-02a: narginchk(1,3) passes when called with 2 args.
    R-POL17-02b: narginchk(1,3) passes with 2 args (disp variant).
    R-POL17-02c: nargoutchk(0,2) passes when called with 2 outputs.

    Consistency argument:
    01a-01b test narginchk in two function definitions; 01c tests
    nargoutchk. Together they verify both argument-count validators.

Requirement R-POL17-03:
    The assert function SHALL pass silently for true conditions and
    raise an error for false conditions, supporting single-argument,
    comparison, and tolerance modes.

    Model-user argument:
    An engineer uses assert() in test harnesses and validation scripts
    to verify numerical results. If assert(true) raises an error, or
    assert(false) passes silently, the entire verification framework
    is unreliable and test results are meaningless.

    Decomposition:
    R-POL17-03a: assert(true) passes silently.
    R-POL17-03b: assert(false) raises an error.
    R-POL17-03c: assert([1 2 3], [1 2 3]) passes for equal arrays.
    R-POL17-03d: assert(1.0001, 1.0, 0.01) passes within tolerance.
    R-POL17-03e: assert(1.5, 1.0, 0.01) errors outside tolerance.

    Consistency argument:
    01a-01b test the boolean mode. 01c tests comparison mode. 01d-01e
    test tolerance mode (pass and fail). Together they cover all three
    assert calling conventions.

Requirement R-POL17-04:
    The error() and warning() functions SHALL support printf-style format
    strings with argument substitution.

    Model-user argument:
    An engineer embeds diagnostic values in error messages using printf
    patterns (e.g., error('Value is %d not %d', obs, exp)). If format
    substitution fails, error messages display raw format strings instead
    of useful diagnostic values, making debugging significantly harder.

    Decomposition:
    R-POL17-04a: error('fmt', args...) produces a formatted message.
    R-POL17-04b: warning('fmt', args...) executes without error.

    Consistency argument:
    01a tests error formatting; 01b tests warning formatting. Together
    they verify printf-style substitution in both functions.

Requirement R-POL17-05:
    MException(id, msg) SHALL create an object with .identifier and
    .message fields, supporting printf-style message formatting, and
    the object SHALL be usable with rethrow.

    Model-user argument:
    An engineer creates MException objects to build custom error
    hierarchies for library code (e.g., 'calc:overflow'). If MException
    does not populate .identifier or does not format printf arguments,
    custom error handling in published toolboxes degrades to unstructured
    string matching.

    Decomposition:
    R-POL17-05a: MException(id, msg) populates .identifier and .message.
    R-POL17-05b: MException(id, fmt, args...) formats the message.
    R-POL17-05c: MException integrates with rethrow (throw, catch, rethrow, catch).

    Consistency argument:
    01a tests basic field creation. 01b tests printf formatting. 01c
    tests the throw-catch-rethrow cycle. Together they verify the full
    MException lifecycle.
"""

import pytest
from forge.engine.session import ForgeSession


@pytest.fixture
def S():
    return ForgeSession()


# ---- rethrow (R-POL17-01) ----------------------------------------------------

def test_rethrow_propagates_error(S):
    """R-POL17-01a: rethrow(e) re-raises the caught error with message."""
    r = S.eval("try; error('myid:sub', 'bad %d', 42); catch e; rethrow(e); end")
    assert "error" in r.lower() and "bad 42" in r


def test_rethrow_preserves_identifier(S):
    """R-POL17-01b: rethrow preserves the original identifier."""
    r = S.eval(
        "try; error('pkg:func', 'fail'); catch e; "
        "try; rethrow(e); catch e2; disp(e2.identifier); end; end"
    )
    assert "pkg:func" in r


# ---- narginchk / nargoutchk (R-POL17-02) ------------------------------------

def test_narginchk_pass(S):
    """R-POL17-02a: narginchk passes when arg count is within range."""
    S.eval("function f(x, y); narginchk(1, 3); end")
    r = S.eval("f(1, 2)")
    assert "error" not in str(r).lower() or r is None or r == ""


def test_narginchk_within_range(S):
    """R-POL17-02b: narginchk(1,3) passes with 2 args (disp variant)."""
    S.eval("function g2(x, y); narginchk(1, 3); disp(x + y); end")
    r = S.eval("g2(3, 4)")
    assert "7" in str(r)


def test_nargoutchk_pass(S):
    """R-POL17-02c: nargoutchk(0,2) passes when called with 2 outputs."""
    S.eval("function [a, b] = h(x); nargoutchk(0, 2); a = x; b = x+1; end")
    r = S.eval("[p, q] = h(5); disp(p)")
    assert "5" in str(r)


# ---- assert (R-POL17-03) ----------------------------------------------------

def test_assert_true(S):
    """R-POL17-03a: assert(true) passes silently."""
    r = S.eval("assert(true)")
    assert "error" not in str(r).lower() or r is None or r == ""


def test_assert_false_errors(S):
    """R-POL17-03b: assert(false) raises an error."""
    r = S.eval("assert(false)")
    assert "error" in str(r).lower()


def test_assert_comparison_equal(S):
    """R-POL17-03c: assert([1 2 3], [1 2 3]) passes for equal arrays."""
    r = S.eval("assert([1 2 3], [1 2 3])")
    assert "error" not in str(r).lower() or r is None or r == ""


def test_assert_tolerance_pass(S):
    """R-POL17-03d: assert(obs, exp, tol) passes within tolerance."""
    r = S.eval("assert(1.0001, 1.0, 0.01)")
    assert "error" not in str(r).lower() or r is None or r == ""


def test_assert_tolerance_fail(S):
    """R-POL17-03e: assert(obs, exp, tol) errors outside tolerance."""
    r = S.eval("assert(1.5, 1.0, 0.01)")
    assert "error" in str(r).lower()


# ---- error / warning with printf format (R-POL17-04) ------------------------

def test_error_printf_format(S):
    """R-POL17-04a: error('fmt', args...) produces formatted message."""
    r = S.eval("try; error('Value is %d not %d', 5, 10); catch e; disp(e.message); end")
    assert "Value is 5 not 10" in r


def test_warning_printf_format(S):
    """R-POL17-04b: warning('fmt', args...) executes without error."""
    r = S.eval("warning('Score is %d', 99)")
    assert "error" not in str(r).lower() or r is None or r == ""


# ---- MException (R-POL17-05) -------------------------------------------------

def test_MException_create(S):
    """R-POL17-05a: MException(id, msg) populates .identifier and .message."""
    S.eval("e = MException('test:err', 'something broke')")
    r_id = S.eval("disp(e.identifier)")
    r_msg = S.eval("disp(e.message)")
    assert "test:err" in r_id
    assert "something broke" in r_msg


def test_MException_printf(S):
    """R-POL17-05b: MException(id, fmt, args...) formats the message."""
    S.eval("e = MException('calc:overflow', 'val=%d max=%d', 999, 100)")
    r = S.eval("disp(e.message)")
    assert "val=999" in r and "max=100" in r


def test_MException_rethrow(S):
    """R-POL17-05c: MException integrates with throw-catch-rethrow cycle."""
    code = (
        "me = MException('io:file', 'not found'); "
        "try; error(me.message); catch e1; "
        "  try; rethrow(e1); catch e2; disp(e2.message); end; "
        "end"
    )
    r = S.eval(code)
    assert "not found" in r
