# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish round 34 -- control flow edge cases and advanced evaluator features.

SRS trace: SRS-FUNC-001 (Octave-compatible function library)
           SRS-CTRL-001 (Control flow: switch/while/for/break/continue/return)
           SRS-FUNC-002 (varargin/varargout, anonymous functions, closures)
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture(scope="module")
def S():
    return ForgeSession()


def _scalar(session, var):
    """Get a scalar float from workspace variable."""
    return float(np.asarray(_unwrap(session.workspace.get(var))).ravel()[0])


def _str(session, var):
    """Get a Python string from a ForgeChar workspace variable."""
    v = session.workspace.get(var)
    return str(v)


# -- 3a: switch with cell-array case ------------------------------------
class TestSwitchEdgeCases:

    def test_switch_cell_array_match(self):
        """switch 3; case {1,2,3} -> found"""
        s = ForgeSession()
        s.eval("switch 3; case {1,2,3}; r='found'; otherwise; r='nope'; end")
        assert _str(s, "r") == "found"

    def test_switch_cell_array_no_match(self):
        """switch 5; case {1,2,3} -> otherwise"""
        s = ForgeSession()
        s.eval("switch 5; case {1,2,3}; r='found'; otherwise; r='nope'; end")
        assert _str(s, "r") == "nope"

    def test_switch_string_no_otherwise(self):
        """switch 'test'; case 'test'; r=1; end -> 1"""
        s = ForgeSession()
        s.eval("switch 'test'\ncase 'test'\n  r=1;\nend")
        assert _scalar(s, "r") == 1.0

    def test_switch_no_match_no_otherwise(self):
        """switch with no matching case and no otherwise leaves var unchanged."""
        s = ForgeSession()
        s.eval("r = 99;\nswitch 5\ncase 1\n  r=1;\ncase 2\n  r=2;\nend")
        assert _scalar(s, "r") == 99.0

    def test_nested_switch(self):
        """Nested switch inside another switch case."""
        s = ForgeSession()
        code = (
            "switch 2\n"
            "case 1\n"
            "  r = 'one';\n"
            "case 2\n"
            "  switch 'inner'\n"
            "    case 'inner'\n"
            "      r = 'nested_found';\n"
            "    otherwise\n"
            "      r = 'nested_nope';\n"
            "  end\n"
            "otherwise\n"
            "  r = 'nope';\n"
            "end"
        )
        s.eval(code)
        assert _str(s, "r") == "nested_found"

    def test_switch_numeric_cases(self):
        """switch selects correct numeric case among several."""
        s = ForgeSession()
        code = "switch 3\ncase 1\n  r=10;\ncase 2\n  r=20;\ncase 3\n  r=30;\nend"
        s.eval(code)
        assert _scalar(s, "r") == 30.0


# -- 4: while with complex conditions -----------------------------------
class TestWhileComplex:

    def test_while_logical_and(self):
        """while x<10 && mod(x,3)~=0 -> x=3"""
        s = ForgeSession()
        s.eval("x=1; while x<10 && mod(x,3)~=0; x=x+1; end")
        assert _scalar(s, "x") == 3.0

    def test_while_with_break(self):
        """while true with break at 5."""
        s = ForgeSession()
        s.eval("x=0; while true; x=x+1; if x>=5; break; end; end")
        assert _scalar(s, "x") == 5.0

    def test_while_with_continue(self):
        """while with continue to sum only odd numbers 1..9 -> 25."""
        s = ForgeSession()
        s.eval("s=0; i=0; while i<10; i=i+1; if mod(i,2)==0; continue; end; s=s+i; end")
        assert _scalar(s, "s") == 25.0

    def test_do_until(self):
        """do-until loop (Octave extension)."""
        s = ForgeSession()
        s.eval("x=0; do; x=x+1; until x>=5;")
        assert _scalar(s, "x") == 5.0


# -- 5: nested function definitions -------------------------------------
class TestNestedFunctions:

    def test_script_local_functions(self):
        """Function defined in script calls another function from same script."""
        s = ForgeSession()
        code = (
            "function y = double_it(x)\n"
            "  y = x * 2;\n"
            "end\n"
            "function y = add_and_double(a, b)\n"
            "  y = double_it(a + b);\n"
            "end\n"
            "r = add_and_double(3, 4);"
        )
        s.eval(code)
        assert _scalar(s, "r") == 14.0

    def test_function_multi_output(self):
        """Function returning two outputs."""
        s = ForgeSession()
        code = (
            "function [mn, mx] = minmax(v)\n"
            "  mn = min(v);\n"
            "  mx = max(v);\n"
            "end\n"
            "[lo, hi] = minmax([3 1 4 1 5 9]);"
        )
        s.eval(code)
        assert _scalar(s, "lo") == 1.0
        assert _scalar(s, "hi") == 9.0


# -- 6: varargin / varargout --------------------------------------------
class TestVararginVarargout:

    def test_varargin_varargout_basic(self):
        """[a,b,c] = multi(1,2,3) with varargin/varargout -> 2,4,6"""
        s = ForgeSession()
        code = (
            "function [varargout] = multi(varargin)\n"
            "  for i=1:length(varargin)\n"
            "    varargout{i} = varargin{i} * 2;\n"
            "  end\n"
            "end\n"
            "[a,b,c] = multi(1,2,3);"
        )
        s.eval(code)
        assert _scalar(s, "a") == 2.0
        assert _scalar(s, "b") == 4.0
        assert _scalar(s, "c") == 6.0

    def test_varargin_length(self):
        """length(varargin) returns correct count."""
        s = ForgeSession()
        code = (
            "function n = count_args(varargin)\n"
            "  n = length(varargin);\n"
            "end\n"
            "r = count_args(10, 20, 30, 40);"
        )
        s.eval(code)
        assert _scalar(s, "r") == 4.0


# -- 7: anonymous functions with captures --------------------------------
class TestAnonymousFunctions:

    def test_arrayfun_with_capture(self):
        """arrayfun(@(x) x*scale, [1 2 3]) with scale=10 -> [10 20 30]"""
        s = ForgeSession()
        s.eval("scale = 10; r = arrayfun(@(x) x*scale, [1 2 3]);")
        v = np.asarray(_unwrap(s.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(v, [10, 20, 30])

    def test_cellfun_with_anonymous(self):
        """cellfun(@(x) x^2, {1,2,3,4}) -> [1 4 9 16]"""
        s = ForgeSession()
        s.eval("r = cellfun(@(x) x^2, {1, 2, 3, 4});")
        v = np.asarray(_unwrap(s.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(v, [1, 4, 9, 16])

    def test_nested_anonymous_closure(self):
        """f = @(a) @(b) a+b; g = f(10); r = g(5) -> 15"""
        s = ForgeSession()
        s.eval("f = @(a) @(b) a+b; g = f(10); r = g(5);")
        assert _scalar(s, "r") == 15.0

    def test_arrayfun_inline_operation(self):
        """arrayfun with inline math on vector."""
        s = ForgeSession()
        s.eval("r = arrayfun(@(x) x^2 + 1, [1 2 3]);")
        v = np.asarray(_unwrap(s.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(v, [2, 5, 10])


# -- 8: return statement -------------------------------------------------
class TestReturnStatement:

    def test_early_return_positive(self):
        """Function with early return for positive input."""
        s = ForgeSession()
        code = (
            "function r = check(x)\n"
            "  if x > 0\n"
            "    r = 'positive';\n"
            "    return;\n"
            "  end\n"
            "  r = 'non-positive';\n"
            "end\n"
            "r = check(5);"
        )
        s.eval(code)
        assert _str(s, "r") == "positive"

    def test_early_return_negative(self):
        """Function falls through when return not triggered."""
        s = ForgeSession()
        code = (
            "function r = check(x)\n"
            "  if x > 0\n"
            "    r = 'positive';\n"
            "    return;\n"
            "  end\n"
            "  r = 'non-positive';\n"
            "end\n"
            "r = check(-3);"
        )
        s.eval(code)
        assert _str(s, "r") == "non-positive"

    def test_for_break_early_exit(self):
        """for with break exits at correct iteration."""
        s = ForgeSession()
        s.eval("r=-1; for i=1:100; if i==7; r=i; break; end; end")
        assert _scalar(s, "r") == 7.0
