"""
tests/test_polish_r16.py -- Polish round 16: core control flow & scoping.

Covers: do...until, unwind_protect, while+break, for+continue,
        parfor, global variables, persistent variables, multi-return functions.

Requirement R-POL16-01:
    The do...until loop SHALL execute the body at least once and repeat
    until the condition becomes true.

    Model-user argument:
    An engineer migrating from MATLAB/Octave uses do...until for iterative
    convergence checks where the loop body must run at least once (e.g.,
    Newton-Raphson iterations). If the body is skipped when the condition
    is already true, convergence loops silently return the initial guess
    instead of a refined solution.

    Decomposition:
    R-POL16-01a: do x=x*2 until x>100 produces 128 (correct doubling).
    R-POL16-01b: Body executes at least once even when condition is
                 immediately true.

    Consistency argument:
    01a tests normal iteration; 01b tests the "at least once" guarantee.
    Together they verify the defining behavior of do...until.

Requirement R-POL16-02:
    unwind_protect SHALL execute the cleanup block regardless of whether
    the protected block raises an error, and SHALL support newline-
    separated syntax.

    Model-user argument:
    An engineer wraps file-handle and hardware-connection code in
    unwind_protect to guarantee cleanup (closing files, releasing locks)
    even on error. If the cleanup block is skipped after an error, file
    handles leak and instrument connections hang until the session is
    killed.

    Decomposition:
    R-POL16-02a: Cleanup block runs after an error in the protected block.
    R-POL16-02b: Cleanup block runs when no error occurs.
    R-POL16-02c: Newline-separated unwind_protect syntax works.

    Consistency argument:
    01a tests error path, 01b tests success path, 01c tests alternate
    syntax. Together they cover the full unwind_protect contract.

Requirement R-POL16-03:
    while loops SHALL support break to exit early, and SHALL not execute
    the body when the condition is initially false.

    Model-user argument:
    An engineer uses while(true) with break for search loops that
    terminate on a found condition. If break does not exit the loop,
    the script hangs indefinitely. If a false initial condition still
    enters the body, guard logic is violated.

    Decomposition:
    R-POL16-03a: break exits a while(true) loop at the correct iteration.
    R-POL16-03b: while(false) never executes the body.

    Consistency argument:
    01a tests break; 01b tests false-initial-condition. Together they
    cover the two key while-loop control paths.

Requirement R-POL16-04:
    for loops SHALL support continue to skip the remainder of the current
    iteration body.

    Model-user argument:
    An engineer uses continue to skip invalid data points or boundary
    conditions within a processing loop. If continue does not skip
    correctly, invalid data contaminates the accumulated result.

    Decomposition:
    R-POL16-04a: Summing odd numbers 1:10 with continue on even yields 25.
    R-POL16-04b: Counter incremented 4 out of 5 times when i==3 is skipped.

    Consistency argument:
    01a tests continue with accumulation; 01b tests continue with a
    counter. Together they verify continue skips correctly.

Requirement R-POL16-05:
    parfor SHALL execute loop bodies and produce the same results as a
    regular for loop (serial fallback).

    Model-user argument:
    An engineer uses parfor for embarrassingly parallel computations
    (parameter sweeps, Monte Carlo). Even without actual parallelism,
    the serial fallback must produce correct indexed results. If parfor
    silently drops iterations or mis-indexes, sweep results are incomplete.

    Decomposition:
    R-POL16-05a: parfor i=1:4 with x(i)=i^2 produces [1,4,9,16].
    R-POL16-05b: parfor with preallocated array produces [10,20,30,40].

    Consistency argument:
    01a tests dynamic array growth; 01b tests preallocated assignment.
    Together they cover the two common parfor usage patterns.

Requirement R-POL16-06:
    Global variables SHALL be shared between the base workspace and
    function workspaces, and modifications in functions SHALL be visible
    in the base workspace.

    Model-user argument:
    An engineer uses global variables for configuration constants and
    shared state between callback functions. If a function modifies a
    global but the change is not visible in the base workspace, callback-
    driven simulations silently use stale configuration values.

    Decomposition:
    R-POL16-06a: Function reads global variable set in base workspace.
    R-POL16-06b: Function modifies global, change visible in base.

    Consistency argument:
    01a tests read direction; 01b tests write direction. Together they
    verify bidirectional global variable sharing.

Requirement R-POL16-07:
    Persistent variables SHALL retain their value between successive
    calls to the same function.

    Model-user argument:
    An engineer uses persistent counters and accumulators in callback
    functions (e.g., counting solver iterations, accumulating running
    statistics). If persistent variables reset on each call, running
    totals restart from zero and iteration counts are always 1.

    Decomposition:
    R-POL16-07a: Persistent counter increments across three calls (1,2,3).
    R-POL16-07b: Persistent accumulator retains running total (10,30,35).

    Consistency argument:
    01a tests increment pattern; 01b tests accumulation pattern.
    Together they verify persistent state retention.

Requirement R-POL16-08:
    Multi-return functions SHALL return all declared output variables via
    [a,b] = func(x) syntax.

    Model-user argument:
    An engineer defines utility functions that return multiple values
    (e.g., [mu, sigma] = stats(x)). If multi-return silently drops the
    second output, the engineer gets only the mean and must call a
    separate function for standard deviation, breaking established
    MATLAB/Octave idioms.

    Decomposition:
    R-POL16-08a: [mn, mx] = minmax(x) returns both min and max.
    R-POL16-08b: [mu, sigma] = stats(x) returns both mean and std.

    Consistency argument:
    01a and 01b each test a two-output function with different return
    types. Together they verify the multi-return mechanism.
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _val(ws, name):
    """Extract scalar float from workspace variable."""
    v = ws.get(name)
    if hasattr(v, "_data"):
        return float(np.asarray(v._data).flat[0])
    if isinstance(v, np.ndarray):
        return float(v.flat[0])
    return v


def _arr(ws, name):
    """Extract 1-D numpy array from workspace variable."""
    v = ws.get(name)
    if hasattr(v, "_data"):
        return np.asarray(v._data).flatten()
    return np.asarray(v).flatten()


# ---------------------------------------------------------------------------
# do...until (R-POL16-01)
# ---------------------------------------------------------------------------
class TestDoUntil:
    """R-POL16-01: do...until loop semantics."""

    def test_basic_doubling(self):
        """R-POL16-01a: do x=x*2 until x>100 produces 128."""
        s = ForgeSession()
        s.eval("x=1; do; x=x*2; until (x>100)")
        assert _val(s.workspace, "x") == 128

    def test_runs_at_least_once(self):
        """R-POL16-01b: Body executes at least once even if condition is immediately true."""
        s = ForgeSession()
        s.eval("x=999; do; x=x+1; until (true)")
        assert _val(s.workspace, "x") == 1000


# ---------------------------------------------------------------------------
# unwind_protect (R-POL16-02)
# ---------------------------------------------------------------------------
class TestUnwindProtect:
    """R-POL16-02: unwind_protect cleanup guarantee."""

    def test_cleanup_after_error(self):
        """R-POL16-02a: Cleanup runs after error in protected block."""
        s = ForgeSession()
        s.eval('unwind_protect; error("oops"); unwind_protect_cleanup; x = "cleaned"; end')
        assert "cleaned" in str(s.workspace.get("x"))

    def test_cleanup_runs_without_error(self):
        """R-POL16-02b: Cleanup runs when no error occurs."""
        s = ForgeSession()
        s.eval("unwind_protect; y=10; unwind_protect_cleanup; z=20; end")
        assert _val(s.workspace, "y") == 10
        assert _val(s.workspace, "z") == 20

    def test_newline_separated(self):
        """R-POL16-02c: Newline-separated unwind_protect syntax works."""
        s = ForgeSession()
        code = "unwind_protect\n  a=1\nunwind_protect_cleanup\n  b=2\nend"
        s.eval(code)
        assert _val(s.workspace, "a") == 1
        assert _val(s.workspace, "b") == 2


# ---------------------------------------------------------------------------
# while + break (R-POL16-03)
# ---------------------------------------------------------------------------
class TestWhileBreak:
    """R-POL16-03: while loop with break and false-initial-condition."""

    def test_break_at_6(self):
        """R-POL16-03a: break exits while(true) at x=6."""
        s = ForgeSession()
        s.eval("x=0; while true; x=x+1; if x>5; break; end; end")
        assert _val(s.workspace, "x") == 6

    def test_while_condition_false(self):
        """R-POL16-03b: while(false) never executes the body."""
        s = ForgeSession()
        s.eval("x=0; while (x > 10); x=x+1; end")
        assert _val(s.workspace, "x") == 0


# ---------------------------------------------------------------------------
# for + continue (R-POL16-04)
# ---------------------------------------------------------------------------
class TestForContinue:
    """R-POL16-04: for loop with continue."""

    def test_sum_odd_numbers(self):
        """R-POL16-04a: Sum of odd numbers 1:10 using continue equals 25."""
        s = ForgeSession()
        s.eval("s=0; for i=1:10; if mod(i,2)==0; continue; end; s=s+i; end")
        assert _val(s.workspace, "s") == 25  # 1+3+5+7+9

    def test_continue_skips_body(self):
        """R-POL16-04b: continue skips body for i==3, counter is 4."""
        s = ForgeSession()
        s.eval("c=0; for i=1:5; if i==3; continue; end; c=c+1; end")
        assert _val(s.workspace, "c") == 4  # skipped i==3


# ---------------------------------------------------------------------------
# parfor (serial fallback) (R-POL16-05)
# ---------------------------------------------------------------------------
class TestParfor:
    """R-POL16-05: parfor serial fallback produces correct results."""

    def test_squares(self):
        """R-POL16-05a: parfor i=1:4 with x(i)=i^2 produces [1,4,9,16]."""
        s = ForgeSession()
        s.eval("parfor i=1:4; x(i)=i^2; end")
        arr = _arr(s.workspace, "x")
        np.testing.assert_array_equal(arr, [1, 4, 9, 16])

    def test_parfor_with_preallocated(self):
        """R-POL16-05b: parfor with preallocated array produces [10,20,30,40]."""
        s = ForgeSession()
        s.eval("y=zeros(1,4); parfor i=1:4; y(i)=i*10; end")
        arr = _arr(s.workspace, "y")
        np.testing.assert_array_equal(arr, [10, 20, 30, 40])


# ---------------------------------------------------------------------------
# global variables (R-POL16-06)
# ---------------------------------------------------------------------------
class TestGlobal:
    """R-POL16-06: Global variable sharing between base and function scopes."""

    def test_global_in_function(self):
        """R-POL16-06a: Function reads global set in base workspace."""
        s = ForgeSession()
        s.eval("global g; g = 42")
        s.eval("function r = getg(); global g; r = g; end")
        s.eval("r = getg()")
        assert _val(s.workspace, "r") == 42

    def test_global_modified_by_function(self):
        """R-POL16-06b: Function modifies global, visible in base."""
        s = ForgeSession()
        s.eval("global g; g = 10")
        s.eval("function setg(v); global g; g = v; end")
        s.eval("setg(99)")
        assert _val(s.workspace, "g") == 99


# ---------------------------------------------------------------------------
# persistent variables (R-POL16-07)
# ---------------------------------------------------------------------------
class TestPersistent:
    """R-POL16-07: Persistent variable retention across function calls."""

    def test_persistent_counter(self):
        """R-POL16-07a: Persistent counter increments across three calls."""
        s = ForgeSession()
        s.eval("function count_p(); persistent n; if isempty(n); n=0; end; n=n+1; disp(n); end")
        r1 = s.eval("count_p()")
        r2 = s.eval("count_p()")
        r3 = s.eval("count_p()")
        assert str(r1).strip() == "1"
        assert str(r2).strip() == "2"
        assert str(r3).strip() == "3"

    def test_persistent_across_calls(self):
        """R-POL16-07b: Persistent accumulator retains running total."""
        s = ForgeSession()
        s.eval("function r = accum(v); persistent total; if isempty(total); total=0; end; total=total+v; r=total; end")
        s.eval("a = accum(10)")
        s.eval("b = accum(20)")
        s.eval("c = accum(5)")
        assert _val(s.workspace, "a") == 10
        assert _val(s.workspace, "b") == 30
        assert _val(s.workspace, "c") == 35


# ---------------------------------------------------------------------------
# multi-return functions (R-POL16-08)
# ---------------------------------------------------------------------------
class TestMultiReturn:
    """R-POL16-08: Multi-return function output capture."""

    def test_minmax(self):
        """R-POL16-08a: [mn, mx] = minmax(x) returns min and max."""
        s = ForgeSession()
        s.eval("function [mn, mx] = minmax(x); mn=min(x); mx=max(x); end")
        s.eval("[a,b]=minmax([3 1 4 1 5])")
        assert _val(s.workspace, "a") == 1
        assert _val(s.workspace, "b") == 5

    def test_stats_function(self):
        """R-POL16-08b: [mu, sigma] = stats(x) returns mean and std."""
        s = ForgeSession()
        s.eval("function [m, s] = stats(x); m=mean(x); s=std(x); end")
        s.eval("[mu, sigma]=stats([2 4 4 4 5 5 7 9])")
        assert abs(_val(s.workspace, "mu") - 5.0) < 1e-10
        # std uses N-1 by default
        assert _val(s.workspace, "sigma") > 0
