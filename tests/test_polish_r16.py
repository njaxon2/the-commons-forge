"""
tests/test_polish_r16.py -- Polish round 16: core control flow & scoping.

Covers: do...until, unwind_protect, while+break, for+continue,
        parfor, global variables, persistent variables, multi-return functions.
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
# do...until
# ---------------------------------------------------------------------------
class TestDoUntil:
    def test_basic_doubling(self):
        s = ForgeSession()
        s.eval("x=1; do; x=x*2; until (x>100)")
        assert _val(s.workspace, "x") == 128

    def test_runs_at_least_once(self):
        """Body must execute at least once even if condition is immediately true."""
        s = ForgeSession()
        s.eval("x=999; do; x=x+1; until (true)")
        assert _val(s.workspace, "x") == 1000


# ---------------------------------------------------------------------------
# unwind_protect
# ---------------------------------------------------------------------------
class TestUnwindProtect:
    def test_cleanup_after_error(self):
        s = ForgeSession()
        s.eval('unwind_protect; error("oops"); unwind_protect_cleanup; x = "cleaned"; end')
        assert "cleaned" in str(s.workspace.get("x"))

    def test_cleanup_runs_without_error(self):
        s = ForgeSession()
        s.eval("unwind_protect; y=10; unwind_protect_cleanup; z=20; end")
        assert _val(s.workspace, "y") == 10
        assert _val(s.workspace, "z") == 20

    def test_newline_separated(self):
        """unwind_protect with newline-separated statements."""
        s = ForgeSession()
        code = "unwind_protect\n  a=1\nunwind_protect_cleanup\n  b=2\nend"
        s.eval(code)
        assert _val(s.workspace, "a") == 1
        assert _val(s.workspace, "b") == 2


# ---------------------------------------------------------------------------
# while + break
# ---------------------------------------------------------------------------
class TestWhileBreak:
    def test_break_at_6(self):
        s = ForgeSession()
        s.eval("x=0; while true; x=x+1; if x>5; break; end; end")
        assert _val(s.workspace, "x") == 6

    def test_while_condition_false(self):
        s = ForgeSession()
        s.eval("x=0; while (x > 10); x=x+1; end")
        assert _val(s.workspace, "x") == 0


# ---------------------------------------------------------------------------
# for + continue
# ---------------------------------------------------------------------------
class TestForContinue:
    def test_sum_odd_numbers(self):
        s = ForgeSession()
        s.eval("s=0; for i=1:10; if mod(i,2)==0; continue; end; s=s+i; end")
        assert _val(s.workspace, "s") == 25  # 1+3+5+7+9

    def test_continue_skips_body(self):
        s = ForgeSession()
        s.eval("c=0; for i=1:5; if i==3; continue; end; c=c+1; end")
        assert _val(s.workspace, "c") == 4  # skipped i==3


# ---------------------------------------------------------------------------
# parfor (serial fallback)
# ---------------------------------------------------------------------------
class TestParfor:
    def test_squares(self):
        s = ForgeSession()
        s.eval("parfor i=1:4; x(i)=i^2; end")
        arr = _arr(s.workspace, "x")
        np.testing.assert_array_equal(arr, [1, 4, 9, 16])

    def test_parfor_with_preallocated(self):
        s = ForgeSession()
        s.eval("y=zeros(1,4); parfor i=1:4; y(i)=i*10; end")
        arr = _arr(s.workspace, "y")
        np.testing.assert_array_equal(arr, [10, 20, 30, 40])


# ---------------------------------------------------------------------------
# global variables
# ---------------------------------------------------------------------------
class TestGlobal:
    def test_global_in_function(self):
        s = ForgeSession()
        s.eval("global g; g = 42")
        s.eval("function r = getg(); global g; r = g; end")
        s.eval("r = getg()")
        assert _val(s.workspace, "r") == 42

    def test_global_modified_by_function(self):
        s = ForgeSession()
        s.eval("global g; g = 10")
        s.eval("function setg(v); global g; g = v; end")
        s.eval("setg(99)")
        assert _val(s.workspace, "g") == 99


# ---------------------------------------------------------------------------
# persistent variables
# ---------------------------------------------------------------------------
class TestPersistent:
    def test_persistent_counter(self):
        s = ForgeSession()
        s.eval("function count_p(); persistent n; if isempty(n); n=0; end; n=n+1; disp(n); end")
        r1 = s.eval("count_p()")
        r2 = s.eval("count_p()")
        r3 = s.eval("count_p()")
        assert str(r1).strip() == "1"
        assert str(r2).strip() == "2"
        assert str(r3).strip() == "3"

    def test_persistent_across_calls(self):
        """Persistent var retains value between calls."""
        s = ForgeSession()
        s.eval("function r = accum(v); persistent total; if isempty(total); total=0; end; total=total+v; r=total; end")
        s.eval("a = accum(10)")
        s.eval("b = accum(20)")
        s.eval("c = accum(5)")
        assert _val(s.workspace, "a") == 10
        assert _val(s.workspace, "b") == 30
        assert _val(s.workspace, "c") == 35


# ---------------------------------------------------------------------------
# multi-return functions
# ---------------------------------------------------------------------------
class TestMultiReturn:
    def test_minmax(self):
        s = ForgeSession()
        s.eval("function [mn, mx] = minmax(x); mn=min(x); mx=max(x); end")
        s.eval("[a,b]=minmax([3 1 4 1 5])")
        assert _val(s.workspace, "a") == 1
        assert _val(s.workspace, "b") == 5

    def test_stats_function(self):
        s = ForgeSession()
        s.eval("function [m, s] = stats(x); m=mean(x); s=std(x); end")
        s.eval("[mu, sigma]=stats([2 4 4 4 5 5 7 9])")
        assert abs(_val(s.workspace, "mu") - 5.0) < 1e-10
        # std uses N-1 by default
        assert _val(s.workspace, "sigma") > 0
