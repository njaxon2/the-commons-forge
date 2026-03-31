# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for function handle and anonymous function edge cases (R25)."""
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
    def test_closure_captures_variable(self, s):
        """a = 5; f = @(x) x + a; f(3) -> 8"""
        s.eval("a = 5; f = @(x) x + a;")
        r = s.eval("f(3)")
        assert _scalar(r) == 8.0

    def test_multi_arg_anon(self, s):
        """f = @(x,y) x.^2 + y.^2; f(3,4) -> 25"""
        r = s.eval("f = @(x,y) x.^2 + y.^2; f(3,4)")
        assert _scalar(r) == 25.0

    def test_chained_anon_returning_anon(self, s):
        """f = @(x) @(y) x + y; g = f(10); g(5) -> 15"""
        s.eval("f = @(x) @(y) x + y;")
        s.eval("g = f(10);")
        r = s.eval("g(5)")
        assert _scalar(r) == 15.0


# -- Function handles to builtins -----------------------------------------


class TestFunctionHandleBuiltins:
    def test_handle_sin(self, s):
        """fh = @sin; fh(pi/2) -> 1"""
        r = s.eval("fh = @sin; fh(pi/2)")
        assert abs(_scalar(r) - 1.0) < 1e-10

    def test_handle_max(self, s):
        """fh = @max; fh([3 1 4]) -> 4"""
        r = s.eval("fh = @max; fh([3 1 4])")
        assert _scalar(r) == 4.0

    def test_cellfun_with_handle(self, s):
        """cellfun(@length, {'hello', 'hi', 'hey'}) -> [5 2 3]"""
        r = s.eval("cellfun(@length, {'hello', 'hi', 'hey'})")
        np.testing.assert_array_equal(_array(r), [5, 2, 3])


# -- feval -----------------------------------------------------------------


class TestFeval:
    def test_feval_function_handle(self, s):
        """feval(@sin, pi/2) -> 1"""
        r = s.eval("feval(@sin, pi/2)")
        assert abs(_scalar(r) - 1.0) < 1e-10

    def test_feval_string_name(self, s):
        """feval('sin', pi/2) -> 1"""
        r = s.eval("feval('sin', pi/2)")
        assert abs(_scalar(r) - 1.0) < 1e-10

    def test_feval_anon_handle(self, s):
        """feval(@(x) x^2, 5) -> 25"""
        r = s.eval("feval(@(x) x^2, 5)")
        assert _scalar(r) == 25.0


# -- arrayfun with anonymous functions -------------------------------------


class TestArrayfunAnon:
    def test_arrayfun_square(self, s):
        """arrayfun(@(x) x^2, [1 2 3 4 5]) -> [1 4 9 16 25]"""
        r = s.eval("arrayfun(@(x) x^2, [1 2 3 4 5])")
        np.testing.assert_array_equal(_array(r), [1, 4, 9, 16, 25])

    def test_arrayfun_uniform_output_false(self, s):
        """arrayfun with UniformOutput=false returns cell array."""
        r = s.eval("arrayfun(@(x) x^2, [1 2 3], 'UniformOutput', false)")
        # Result should be a cell
        from forge.engine.containers import ForgeCell
        assert isinstance(r, ForgeCell)


# -- Nested function calls with handles ------------------------------------


class TestNestedHandles:
    def test_nested_handle_reference(self, s):
        """f = @(x) 2*x; g = @(x) f(x) + 1; g(5) -> 11"""
        s.eval("f = @(x) 2*x; g = @(x) f(x) + 1;")
        r = s.eval("g(5)")
        assert _scalar(r) == 11.0

    def test_compose_handles(self, s):
        """compose = @(f,g) @(x) f(g(x)); h = compose(@sin, @(x) x*pi); h(0.5) -> 1"""
        s.eval("compose = @(f,g) @(x) f(g(x));")
        s.eval("h = compose(@sin, @(x) x*pi);")
        r = s.eval("h(0.5)")
        assert abs(_scalar(r) - 1.0) < 1e-10


# -- str2func --------------------------------------------------------------


class TestStr2func:
    def test_str2func_sin(self, s):
        """str2func('sin')(0) -> 0"""
        s.eval("f = str2func('sin');")
        r = s.eval("f(0)")
        assert abs(_scalar(r)) < 1e-10

    def test_str2func_max(self, s):
        """str2func('max')([3 1 4]) -> 4"""
        s.eval("f = str2func('max');")
        r = s.eval("f([3 1 4])")
        assert _scalar(r) == 4.0


# -- func2str --------------------------------------------------------------


class TestFunc2str:
    def test_func2str_named_handle(self, s):
        """func2str(@sin) -> 'sin'"""
        r = s.eval("func2str(@sin)")
        assert str(r) == "sin"

    def test_func2str_after_str2func(self, s):
        """Round-trip: str2func then func2str."""
        s.eval("f = str2func('cos');")
        r = s.eval("func2str(f)")
        assert str(r) == "cos"
