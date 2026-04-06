# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for function handle and anonymous function edge cases (R25).

V&V Traceability (backfill):
    R-POL25-01 .. R-POL25-06 (parent requirements)
    R-POL25-01-nn .. R-POL25-06-nn (unit sub-requirements)

SRS trace: SRS-FUNC-001, SRS-VAL-001, SRS-COMPAT-001
"""
import pytest
import numpy as np
from forge.engine.evaluator import Session
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def s():
    return Session()


def _scalar(v):
    """Extract a Python float from a workspace value."""
    arr = _unwrap(v)
    return float(np.asarray(arr).flat[0])


def _array(v):
    """Extract a flat numpy array from a workspace value."""
    return np.asarray(_unwrap(v)).flatten()


# -- Anonymous function captures -------------------------------------------


class TestAnonFunctionCaptures:
    """R-POL25-01: Forge SHALL support anonymous function definitions that
    capture variables from the enclosing scope (closures), accept multiple
    arguments, and return anonymous functions (higher-order functions),
    matching MATLAB/Octave behavior.

    Model-user argument: An engineer porting optimization or callback-based code
    from Octave creates anonymous functions that capture parameters (e.g.,
    @(x) x + offset), pass multi-argument cost functions to solvers, and build
    function factories via nested anonymous functions. Broken closures or
    higher-order returns would make entire classes of functional-style Octave
    code non-portable.

    Decomposition:
        R-POL25-01-01: Closure captures variable from enclosing scope
        R-POL25-01-02: Multi-argument anonymous function evaluates correctly
        R-POL25-01-03: Anonymous function returning anonymous function chains

    Consistency: Sub-requirements cover single-variable capture (01),
    multi-argument evaluation (02), and higher-order function return (03).
    Together they validate the closure and lambda semantics.
    """

    def test_closure_captures_variable(self, s):
        """R-POL25-01-01: a=5; f=@(x) x+a; f(3) yields 8."""
        s.eval("a = 5; f = @(x) x + a;")
        r = s.eval("f(3)")
        assert _scalar(r) == 8.0

    def test_multi_arg_anon(self, s):
        """R-POL25-01-02: @(x,y) x.^2+y.^2 at (3,4) yields 25."""
        r = s.eval("f = @(x,y) x.^2 + y.^2; f(3,4)")
        assert _scalar(r) == 25.0

    def test_chained_anon_returning_anon(self, s):
        """R-POL25-01-03: @(x) @(y) x+y chains: f(10)(5) yields 15."""
        s.eval("f = @(x) @(y) x + y;")
        s.eval("g = f(10);")
        r = s.eval("g(5)")
        assert _scalar(r) == 15.0


# -- Function handles to builtins -----------------------------------------


class TestFunctionHandleBuiltins:
    """R-POL25-02: Forge SHALL support creating function handles to built-in
    functions via @name syntax, and passing them to higher-order functions
    like cellfun, matching MATLAB/Octave behavior.

    Model-user argument: An engineer porting data-processing pipelines from
    Octave uses @sin, @max, and similar handles as arguments to cellfun,
    arrayfun, and custom iterators. If handle creation or dispatch fails,
    functional-style processing patterns that are ubiquitous in Octave code
    become non-portable.

    Decomposition:
        R-POL25-02-01: @sin handle evaluates sin(pi/2) = 1
        R-POL25-02-02: @max handle evaluates max([3 1 4]) = 4
        R-POL25-02-03: cellfun(@length, ...) maps over cell array

    Consistency: Sub-requirements cover handle creation for math (01) and
    reduction (02) builtins, plus higher-order dispatch via cellfun (03).
    Together they validate the @name handle mechanism.
    """

    def test_handle_sin(self, s):
        """R-POL25-02-01: @sin handle evaluates sin(pi/2) = 1."""
        r = s.eval("fh = @sin; fh(pi/2)")
        assert abs(_scalar(r) - 1.0) < 1e-10

    def test_handle_max(self, s):
        """R-POL25-02-02: @max handle evaluates max([3 1 4]) = 4."""
        r = s.eval("fh = @max; fh([3 1 4])")
        assert _scalar(r) == 4.0

    def test_cellfun_with_handle(self, s):
        """R-POL25-02-03: cellfun(@length, {'hello','hi','hey'}) yields [5 2 3]."""
        r = s.eval("cellfun(@length, {'hello', 'hi', 'hey'})")
        np.testing.assert_array_equal(_array(r), [5, 2, 3])


# -- feval -----------------------------------------------------------------


class TestFeval:
    """R-POL25-03: Forge SHALL support feval() with function handles, string
    function names, and anonymous function handles, matching MATLAB/Octave
    dispatch behavior.

    Model-user argument: An engineer porting dynamic dispatch code from Octave
    uses feval('funcname', ...) for runtime function selection (e.g., choosing
    solvers by name from config files) and feval(@handle, ...) for callback
    invocation. Both paths must work for plugin-style architectures to port.

    Decomposition:
        R-POL25-03-01: feval(@sin, pi/2) returns 1
        R-POL25-03-02: feval('sin', pi/2) returns 1 (string name)
        R-POL25-03-03: feval(@(x) x^2, 5) returns 25 (anonymous handle)

    Consistency: Sub-requirements cover handle dispatch (01), string dispatch
    (02), and anonymous handle dispatch (03). Together they verify all three
    feval calling conventions.
    """

    def test_feval_function_handle(self, s):
        """R-POL25-03-01: feval(@sin, pi/2) returns 1."""
        r = s.eval("feval(@sin, pi/2)")
        assert abs(_scalar(r) - 1.0) < 1e-10

    def test_feval_string_name(self, s):
        """R-POL25-03-02: feval('sin', pi/2) returns 1."""
        r = s.eval("feval('sin', pi/2)")
        assert abs(_scalar(r) - 1.0) < 1e-10

    def test_feval_anon_handle(self, s):
        """R-POL25-03-03: feval(@(x) x^2, 5) returns 25."""
        r = s.eval("feval(@(x) x^2, 5)")
        assert _scalar(r) == 25.0


# -- arrayfun with anonymous functions -------------------------------------


class TestArrayfunAnon:
    """R-POL25-04: Forge SHALL support arrayfun() with anonymous functions,
    including UniformOutput=false mode that returns cell arrays, matching
    MATLAB/Octave behavior.

    Model-user argument: A scientist porting vectorized processing code from
    Octave uses arrayfun(@(x) expr, v) to apply element-wise transformations
    without explicit loops, and 'UniformOutput', false when outputs are
    non-uniform. Both modes are essential for idiomatic Octave code.

    Decomposition:
        R-POL25-04-01: arrayfun(@(x) x^2, [1 2 3 4 5]) yields [1 4 9 16 25]
        R-POL25-04-02: arrayfun with UniformOutput=false returns ForgeCell

    Consistency: Sub-requirements cover uniform output (01) and non-uniform
    output (02) modes. Together they verify both arrayfun return paths.
    """

    def test_arrayfun_square(self, s):
        """R-POL25-04-01: arrayfun(@(x) x^2, [1..5]) yields [1 4 9 16 25]."""
        r = s.eval("arrayfun(@(x) x^2, [1 2 3 4 5])")
        np.testing.assert_array_equal(_array(r), [1, 4, 9, 16, 25])

    def test_arrayfun_uniform_output_false(self, s):
        """R-POL25-04-02: arrayfun with UniformOutput=false returns cell."""
        r = s.eval("arrayfun(@(x) x^2, [1 2 3], 'UniformOutput', false)")
        # Result should be a cell
        from forge.engine.containers import ForgeCell
        assert isinstance(r, ForgeCell)


# -- Nested function calls with handles ------------------------------------


class TestNestedHandles:
    """R-POL25-05: Forge SHALL support nested anonymous function calls where
    one anonymous function references another, and function composition
    patterns, matching MATLAB/Octave behavior.

    Model-user argument: An engineer building composable processing pipelines
    in Octave creates functions like g=@(x) f(x)+1 and compose=@(f,g) @(x)
    f(g(x)). These patterns are fundamental to functional-style signal
    processing chains and must work for pipeline architectures to port.

    Decomposition:
        R-POL25-05-01: g=@(x) f(x)+1 correctly references f
        R-POL25-05-02: compose=@(f,g) @(x) f(g(x)) composes correctly

    Consistency: Sub-requirements cover direct nesting (01) and generic
    composition (02). Together they validate nested handle resolution.
    """

    def test_nested_handle_reference(self, s):
        """R-POL25-05-01: g=@(x) f(x)+1 where f=@(x) 2*x; g(5) yields 11."""
        s.eval("f = @(x) 2*x; g = @(x) f(x) + 1;")
        r = s.eval("g(5)")
        assert _scalar(r) == 11.0

    def test_compose_handles(self, s):
        """R-POL25-05-02: compose(@sin, @(x) x*pi); h(0.5) yields 1."""
        s.eval("compose = @(f,g) @(x) f(g(x));")
        s.eval("h = compose(@sin, @(x) x*pi);")
        r = s.eval("h(0.5)")
        assert abs(_scalar(r) - 1.0) < 1e-10


# -- str2func --------------------------------------------------------------


class TestStr2func:
    """R-POL25-06: Forge SHALL provide str2func and func2str for converting
    between string function names and function handles, supporting round-trip
    conversion matching MATLAB/Octave behavior.

    Model-user argument: An engineer porting configuration-driven code from
    Octave uses str2func to convert function names read from config files into
    callable handles, and func2str for logging and display. Both must work for
    dynamic dispatch architectures.

    Decomposition:
        R-POL25-06-01: str2func('sin') creates callable handle, f(0)=0
        R-POL25-06-02: str2func('max') creates callable handle, f([3 1 4])=4
        R-POL25-06-03: func2str(@sin) returns 'sin'
        R-POL25-06-04: str2func then func2str round-trip preserves name

    Consistency: Sub-requirements cover str2func for different builtins (01-02),
    func2str for named handles (03), and round-trip (04). Together they validate
    the string/handle conversion API.
    """

    def test_str2func_sin(self, s):
        """R-POL25-06-01: str2func('sin'); f(0) yields 0."""
        s.eval("f = str2func('sin');")
        r = s.eval("f(0)")
        assert abs(_scalar(r)) < 1e-10

    def test_str2func_max(self, s):
        """R-POL25-06-02: str2func('max'); f([3 1 4]) yields 4."""
        s.eval("f = str2func('max');")
        r = s.eval("f([3 1 4])")
        assert _scalar(r) == 4.0


# -- func2str --------------------------------------------------------------


class TestFunc2str:

    def test_func2str_named_handle(self, s):
        """R-POL25-06-03: func2str(@sin) returns 'sin'."""
        r = s.eval("func2str(@sin)")
        assert str(r) == "sin"

    def test_func2str_after_str2func(self, s):
        """R-POL25-06-04: str2func then func2str round-trip preserves 'cos'."""
        s.eval("f = str2func('cos');")
        r = s.eval("func2str(f)")
        assert str(r) == "cos"
