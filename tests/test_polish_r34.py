# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish round 34 -- control flow edge cases and advanced evaluator features.

SRS trace: SRS-FUNC-001 (Octave-compatible function library)
           SRS-CTRL-001 (Control flow: switch/while/for/break/continue/return)
           SRS-FUNC-002 (varargin/varargout, anonymous functions, closures)

V&V Traceability (backfill)
===========================
R-POL34-01: switch statements SHALL support cell-array case lists, string
            matching, nested switches, and correct fallthrough/otherwise
            behavior.

    Model-user argument: Engineers use switch/case for dispatching on file
    types, command names, or error codes. Cell-array case lists like
    ``case {1,2,3}`` are common for grouping equivalent options. If switch
    does not match correctly, dispatch logic silently takes the wrong branch.

    Decomposition:
      R-POL34-01a: switch 3; case {1,2,3} matches.
      R-POL34-01b: switch 5; case {1,2,3} falls to otherwise.
      R-POL34-01c: switch 'test'; case 'test' matches without otherwise.
      R-POL34-01d: No matching case and no otherwise leaves variable unchanged.
      R-POL34-01e: Nested switch inside a case body works correctly.
      R-POL34-01f: switch selects correct numeric case among several.

    Consistency: Cell-array match/miss (01a-b), string match (01c), no-match
    no-otherwise (01d), nesting (01e), and multi-case selection (01f) cover
    the switch API.

R-POL34-02: while loops SHALL support compound conditions, break, continue,
            and do-until (Octave extension).

    Model-user argument: Data-acquisition loops with complex termination
    conditions (``while x<N && valid``) are standard in instrument control.
    break and continue are needed for early exit and skip logic. do-until
    is an Octave extension that engineers may have in existing scripts.

    Decomposition:
      R-POL34-02a: while with && compound condition terminates correctly.
      R-POL34-02b: while true with break exits at correct iteration.
      R-POL34-02c: while with continue sums only odd numbers.
      R-POL34-02d: do-until loop executes body at least once.

    Consistency: Compound condition (02a), break (02b), continue (02c), and
    do-until (02d) cover while-loop control flow.

R-POL34-03: Script-local function definitions SHALL support mutual calls
            and multi-output returns.

    Model-user argument: Scientists write helper functions within a script
    file. These must be callable from other functions in the same file, and
    must support multiple return values for decomposition results (e.g.,
    [min, max] = minmax(v)).

    Decomposition:
      R-POL34-03a: Function defined in script calls another from same script.
      R-POL34-03b: Function returns two outputs correctly.

    Consistency: Mutual calls (03a) and multi-output (03b) cover script-local
    function capabilities.

R-POL34-04: varargin and varargout SHALL allow variable-argument-count
            functions to accept and return arbitrary numbers of values.

    Model-user argument: Library functions with flexible signatures (e.g.,
    plot(x), plot(x,y), plot(x,y,'--')) use varargin. varargout enables
    functions that return different numbers of outputs depending on context.
    Both are essential for Octave compatibility.

    Decomposition:
      R-POL34-04a: varargout with varargin doubles each input.
      R-POL34-04b: length(varargin) returns correct argument count.

    Consistency: Round-trip varargin/varargout (04a) and count query (04b)
    cover the variable-argument API.

R-POL34-05: Anonymous functions SHALL capture variables from the enclosing
            scope (closures) and support nesting.

    Model-user argument: Engineers create anonymous functions for arrayfun
    and cellfun callbacks, often capturing a scale factor or threshold from
    the workspace. Nested closures like @(a) @(b) a+b are used for partial
    application. If capture fails, callbacks silently use wrong values.

    Decomposition:
      R-POL34-05a: arrayfun with captured scale variable works.
      R-POL34-05b: cellfun with anonymous squaring function works.
      R-POL34-05c: Nested anonymous closure captures outer variable.
      R-POL34-05d: arrayfun with inline math expression works.

    Consistency: Variable capture (05a), cellfun dispatch (05b), nested
    closures (05c), and inline expressions (05d) cover anonymous functions.

R-POL34-06: return and break statements SHALL exit functions and loops at
            the correct point.

    Model-user argument: Early return is used for input validation and guard
    clauses. break exits search loops when a target is found. If either
    statement does not transfer control correctly, functions run past their
    intended exit point and produce wrong results.

    Decomposition:
      R-POL34-06a: Early return exits function at the correct point (positive).
      R-POL34-06b: Function falls through when return is not triggered.
      R-POL34-06c: for with break exits at the correct iteration.

    Consistency: Return triggered (06a), return not triggered (06b), and
    break in for-loop (06c) cover control-flow exit statements.
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
    """R-POL34-01: switch statements SHALL support cell-array case lists,
    string matching, nested switches, and correct fallthrough/otherwise
    behavior.

    Model-user argument: Engineers use switch/case for dispatching on file
    types, command names, or error codes. Cell-array case lists like
    ``case {1,2,3}`` are common for grouping equivalent options. If switch
    does not match correctly, dispatch logic silently takes the wrong branch.

    Decomposition:
      R-POL34-01a: switch 3; case {1,2,3} matches.
      R-POL34-01b: switch 5; case {1,2,3} falls to otherwise.
      R-POL34-01c: switch 'test'; case 'test' matches without otherwise.
      R-POL34-01d: No matching case and no otherwise leaves variable unchanged.
      R-POL34-01e: Nested switch inside a case body works correctly.
      R-POL34-01f: switch selects correct numeric case among several.

    Consistency: Cell-array match/miss (01a-b), string match (01c), no-match
    no-otherwise (01d), nesting (01e), and multi-case selection (01f) cover
    the switch API.
    """

    def test_switch_cell_array_match(self):
        """R-POL34-01a: switch 3; case {1,2,3} SHALL match."""
        s = ForgeSession()
        s.eval("switch 3; case {1,2,3}; r='found'; otherwise; r='nope'; end")
        assert _str(s, "r") == "found"

    def test_switch_cell_array_no_match(self):
        """R-POL34-01b: switch 5; case {1,2,3} SHALL fall to otherwise."""
        s = ForgeSession()
        s.eval("switch 5; case {1,2,3}; r='found'; otherwise; r='nope'; end")
        assert _str(s, "r") == "nope"

    def test_switch_string_no_otherwise(self):
        """R-POL34-01c: switch 'test'; case 'test' SHALL match without otherwise."""
        s = ForgeSession()
        s.eval("switch 'test'\ncase 'test'\n  r=1;\nend")
        assert _scalar(s, "r") == 1.0

    def test_switch_no_match_no_otherwise(self):
        """R-POL34-01d: No matching case and no otherwise SHALL leave variable unchanged."""
        s = ForgeSession()
        s.eval("r = 99;\nswitch 5\ncase 1\n  r=1;\ncase 2\n  r=2;\nend")
        assert _scalar(s, "r") == 99.0

    def test_nested_switch(self):
        """R-POL34-01e: Nested switch inside a case body SHALL work correctly."""
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
        """R-POL34-01f: switch SHALL select correct numeric case among several."""
        s = ForgeSession()
        code = "switch 3\ncase 1\n  r=10;\ncase 2\n  r=20;\ncase 3\n  r=30;\nend"
        s.eval(code)
        assert _scalar(s, "r") == 30.0


# -- 4: while with complex conditions -----------------------------------
class TestWhileComplex:
    """R-POL34-02: while loops SHALL support compound conditions, break,
    continue, and do-until (Octave extension).

    Model-user argument: Data-acquisition loops with complex termination
    conditions are standard in instrument control. break and continue are
    needed for early exit and skip logic. do-until is an Octave extension
    that engineers may have in existing scripts.

    Decomposition:
      R-POL34-02a: while with && compound condition terminates correctly.
      R-POL34-02b: while true with break exits at correct iteration.
      R-POL34-02c: while with continue sums only odd numbers.
      R-POL34-02d: do-until loop executes body at least once.

    Consistency: Compound condition (02a), break (02b), continue (02c), and
    do-until (02d) cover while-loop control flow.
    """

    def test_while_logical_and(self):
        """R-POL34-02a: while x<10 && mod(x,3)~=0 SHALL terminate at x=3."""
        s = ForgeSession()
        s.eval("x=1; while x<10 && mod(x,3)~=0; x=x+1; end")
        assert _scalar(s, "x") == 3.0

    def test_while_with_break(self):
        """R-POL34-02b: while true with break at 5 SHALL exit at x=5."""
        s = ForgeSession()
        s.eval("x=0; while true; x=x+1; if x>=5; break; end; end")
        assert _scalar(s, "x") == 5.0

    def test_while_with_continue(self):
        """R-POL34-02c: while with continue SHALL sum only odd numbers 1..9 = 25."""
        s = ForgeSession()
        s.eval("s=0; i=0; while i<10; i=i+1; if mod(i,2)==0; continue; end; s=s+i; end")
        assert _scalar(s, "s") == 25.0

    def test_do_until(self):
        """R-POL34-02d: do-until loop SHALL execute body at least once."""
        s = ForgeSession()
        s.eval("x=0; do; x=x+1; until x>=5;")
        assert _scalar(s, "x") == 5.0


# -- 5: nested function definitions -------------------------------------
class TestNestedFunctions:
    """R-POL34-03: Script-local function definitions SHALL support mutual
    calls and multi-output returns.

    Model-user argument: Scientists write helper functions within a script
    file. These must be callable from other functions in the same file, and
    must support multiple return values.

    Decomposition:
      R-POL34-03a: Function defined in script calls another from same script.
      R-POL34-03b: Function returns two outputs correctly.

    Consistency: Mutual calls (03a) and multi-output (03b) cover script-local
    function capabilities.
    """

    def test_script_local_functions(self):
        """R-POL34-03a: Function defined in script SHALL call another from same script."""
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
        """R-POL34-03b: Function SHALL return two outputs correctly."""
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
    """R-POL34-04: varargin and varargout SHALL allow variable-argument-count
    functions to accept and return arbitrary numbers of values.

    Model-user argument: Library functions with flexible signatures use
    varargin. varargout enables functions that return different numbers of
    outputs depending on context. Both are essential for Octave compatibility.

    Decomposition:
      R-POL34-04a: varargout with varargin doubles each input.
      R-POL34-04b: length(varargin) returns correct argument count.

    Consistency: Round-trip varargin/varargout (04a) and count query (04b)
    cover the variable-argument API.
    """

    def test_varargin_varargout_basic(self):
        """R-POL34-04a: varargout with varargin SHALL double each input."""
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
        """R-POL34-04b: length(varargin) SHALL return correct argument count."""
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
    """R-POL34-05: Anonymous functions SHALL capture variables from the
    enclosing scope (closures) and support nesting.

    Model-user argument: Engineers create anonymous functions for arrayfun
    and cellfun callbacks, often capturing a scale factor or threshold from
    the workspace. Nested closures are used for partial application. If
    capture fails, callbacks silently use wrong values.

    Decomposition:
      R-POL34-05a: arrayfun with captured scale variable works.
      R-POL34-05b: cellfun with anonymous squaring function works.
      R-POL34-05c: Nested anonymous closure captures outer variable.
      R-POL34-05d: arrayfun with inline math expression works.

    Consistency: Variable capture (05a), cellfun dispatch (05b), nested
    closures (05c), and inline expressions (05d) cover anonymous functions.
    """

    def test_arrayfun_with_capture(self):
        """R-POL34-05a: arrayfun with captured scale variable SHALL work."""
        s = ForgeSession()
        s.eval("scale = 10; r = arrayfun(@(x) x*scale, [1 2 3]);")
        v = np.asarray(_unwrap(s.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(v, [10, 20, 30])

    def test_cellfun_with_anonymous(self):
        """R-POL34-05b: cellfun with anonymous squaring function SHALL work."""
        s = ForgeSession()
        s.eval("r = cellfun(@(x) x^2, {1, 2, 3, 4});")
        v = np.asarray(_unwrap(s.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(v, [1, 4, 9, 16])

    def test_nested_anonymous_closure(self):
        """R-POL34-05c: Nested anonymous closure SHALL capture outer variable."""
        s = ForgeSession()
        s.eval("f = @(a) @(b) a+b; g = f(10); r = g(5);")
        assert _scalar(s, "r") == 15.0

    def test_arrayfun_inline_operation(self):
        """R-POL34-05d: arrayfun with inline math expression SHALL work."""
        s = ForgeSession()
        s.eval("r = arrayfun(@(x) x^2 + 1, [1 2 3]);")
        v = np.asarray(_unwrap(s.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(v, [2, 5, 10])


# -- 8: return statement -------------------------------------------------
class TestReturnStatement:
    """R-POL34-06: return and break statements SHALL exit functions and loops
    at the correct point.

    Model-user argument: Early return is used for input validation and guard
    clauses. break exits search loops when a target is found. If either
    statement does not transfer control correctly, functions run past their
    intended exit point.

    Decomposition:
      R-POL34-06a: Early return exits function at the correct point.
      R-POL34-06b: Function falls through when return is not triggered.
      R-POL34-06c: for with break exits at the correct iteration.

    Consistency: Return triggered (06a), return not triggered (06b), and
    break in for-loop (06c) cover control-flow exit statements.
    """

    def test_early_return_positive(self):
        """R-POL34-06a: Early return SHALL exit function at the correct point."""
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
        """R-POL34-06b: Function SHALL fall through when return is not triggered."""
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
        """R-POL34-06c: for with break SHALL exit at the correct iteration."""
        s = ForgeSession()
        s.eval("r=-1; for i=1:100; if i==7; r=i; break; end; end")
        assert _scalar(s, "r") == 7.0
