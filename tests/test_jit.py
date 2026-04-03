# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Tests for Numba JIT compilation of numeric for-loops."""
import pytest
import numpy as np
import time
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


def get_val(s, varname):
    """Get raw numpy data from workspace variable."""
    v = s._engine.workspace.get(varname)
    if hasattr(v, '_data'):
        return v._data
    return v


class TestJITCorrectness:
    """Verify JIT-compiled loops produce identical results to interpreted."""

    def test_simple_sin_cos_loop(self, s):
        """for i=1:100, result(i) = sin(i) * cos(i)"""
        s.eval("for i = 1:100; result(i) = sin(i) * cos(i); end")
        result = get_val(s, "result")
        expected = np.array([np.sin(i) * np.cos(i) for i in range(1, 101)])
        np.testing.assert_allclose(result.ravel(), expected, rtol=1e-12)

    def test_arithmetic_loop(self, s):
        """for i=1:50, r(i) = i^2 + 3*i - 1"""
        s.eval("for i = 1:50; r(i) = i^2 + 3*i - 1; end")
        result = get_val(s, "r")
        expected = np.array([i**2 + 3*i - 1 for i in range(1, 51)])
        np.testing.assert_allclose(result.ravel(), expected, rtol=1e-12)

    def test_exp_log_loop(self, s):
        """for i=1:20, r(i) = exp(-i/10) * log(i+1)"""
        s.eval("for i = 1:20; r(i) = exp(-i/10) * log(i+1); end")
        result = get_val(s, "r")
        expected = np.array([np.exp(-i/10) * np.log(i+1) for i in range(1, 21)])
        np.testing.assert_allclose(result.ravel(), expected, rtol=1e-12)

    def test_sqrt_abs_loop(self, s):
        """for i=1:30, r(i) = sqrt(abs(sin(i)))"""
        s.eval("for i = 1:30; r(i) = sqrt(abs(sin(i))); end")
        result = get_val(s, "r")
        expected = np.array([np.sqrt(np.abs(np.sin(i))) for i in range(1, 31)])
        np.testing.assert_allclose(result.ravel(), expected, rtol=1e-12)

    def test_step_range(self, s):
        """for i=1:2:20, r(i) = i*2"""
        # Note: with step=2, i takes values 1,3,5,...,19 and writes to r(i)
        s.eval("r = zeros(1, 20); for i = 1:2:19; r(i) = i * 2; end")
        result = get_val(s, "r")
        expected = np.zeros(20)
        for i in range(1, 20, 2):
            expected[i-1] = i * 2
        np.testing.assert_allclose(result.ravel(), expected, rtol=1e-12)

    def test_accumulator_pattern(self, s):
        """for i=1:100, s = s + i"""
        s.eval("s = 0; for i = 1:100; s = s + i; end")
        result = get_val(s, "s")
        np.testing.assert_allclose(float(np.asarray(result).ravel()[0]), 5050.0, rtol=1e-12)

    def test_accumulator_with_math(self, s):
        """for i=1:50, s = s + sin(i)"""
        s.eval("s = 0; for i = 1:50; s = s + sin(i); end")
        result = get_val(s, "s")
        expected = sum(np.sin(i) for i in range(1, 51))
        np.testing.assert_allclose(float(np.asarray(result).ravel()[0]), expected, rtol=1e-10)

    def test_multiple_assignments(self, s):
        """Loop with multiple array assignments per iteration."""
        s.eval("for i = 1:20; a(i) = sin(i); b(i) = cos(i); end")
        a = get_val(s, "a")
        b = get_val(s, "b")
        expected_a = np.array([np.sin(i) for i in range(1, 21)])
        expected_b = np.array([np.cos(i) for i in range(1, 21)])
        np.testing.assert_allclose(a.ravel(), expected_a, rtol=1e-12)
        np.testing.assert_allclose(b.ravel(), expected_b, rtol=1e-12)

    def test_pi_constant(self, s):
        """Using pi inside JIT loop."""
        s.eval("for i = 1:10; r(i) = sin(i * pi / 10); end")
        result = get_val(s, "r")
        expected = np.array([np.sin(i * np.pi / 10) for i in range(1, 11)])
        np.testing.assert_allclose(result.ravel(), expected, rtol=1e-12)

    def test_read_external_scalar(self, s):
        """Loop reads a scalar from workspace."""
        s.eval("k = 2.5; for i = 1:10; r(i) = k * sin(i); end")
        result = get_val(s, "r")
        expected = np.array([2.5 * np.sin(i) for i in range(1, 11)])
        np.testing.assert_allclose(result.ravel(), expected, rtol=1e-12)

    def test_read_external_array(self, s):
        """Loop reads from an existing array."""
        s.eval("x = [10 20 30 40 50]; for i = 1:5; r(i) = x(i) * 2; end")
        result = get_val(s, "r")
        expected = np.array([20, 40, 60, 80, 100], dtype=float)
        np.testing.assert_allclose(result.ravel(), expected, rtol=1e-12)


class TestJITFallback:
    """Verify graceful fallback for non-eligible loops."""

    def test_string_body_falls_back(self, s):
        """Loop with string operations should still work (interpreted)."""
        s.eval("r = {}; for i = 1:3; r{i} = 'hello'; end")
        # Should complete without error
        result = get_val(s, "r")
        assert result is not None

    def test_cell_iteration_works(self, s):
        """for i = {1, 'a', 2} should work interpreted."""
        s.eval("s = 0; for i = {1, 2, 3}; s = s + i; end")
        result = get_val(s, "s")
        np.testing.assert_allclose(float(np.asarray(result).ravel()[0]), 6.0)

    def test_matrix_iteration_works(self, s):
        """for col = [1 2; 3 4] iterates columns."""
        s.eval("s = 0; for col = [1 2; 3 4]; s = s + col(1); end")
        result = get_val(s, "s")
        np.testing.assert_allclose(float(np.asarray(result).ravel()[0]), 3.0)

    def test_nested_if_in_loop(self, s):
        """Loop with if-statement in body is NOT JIT-eligible but works."""
        s.eval("s = 0; for i = 1:10; if mod(i, 2) == 0; s = s + i; end; end")
        result = get_val(s, "s")
        # Sum of even numbers 2+4+6+8+10 = 30
        np.testing.assert_allclose(float(np.asarray(result).ravel()[0]), 30.0)

    def test_function_call_in_loop(self, s):
        """Loop calling non-math functions falls back gracefully."""
        s.eval("for i = 1:5; r(i) = length([1 2 3]); end")
        result = get_val(s, "r")
        np.testing.assert_allclose(result.ravel(), [3, 3, 3, 3, 3])

    def test_break_in_loop(self, s):
        """Loop with break falls back to interpreter."""
        s.eval("s = 0; for i = 1:100; if i > 5; break; end; s = s + i; end")
        result = get_val(s, "s")
        np.testing.assert_allclose(float(np.asarray(result).ravel()[0]), 15.0)


class TestJITModuleAPI:
    """Test the JIT module API directly."""

    def test_numba_availability(self):
        """is_numba_available should reflect actual import."""
        from forge.engine.jit_compiler import is_numba_available
        # Since we installed numba, it should be True
        assert is_numba_available() is True

    def test_can_jit_simple(self):
        """can_jit_loop on a simple eligible loop."""
        from forge.engine.jit_compiler import can_jit_loop
        from forge.engine.parser import (
            ForStatement, Assignment, Index, Identifier,
            NumberLiteral, ColonExpr, ExpressionStatement,
        )
        # Build: for i = 1:10; r(i) = i * 2; end
        loop = ForStatement(
            var='i',
            iter_expr=ColonExpr(NumberLiteral('1'), NumberLiteral('10')),
            body=[ExpressionStatement(Assignment(
                targets=Index(Identifier('r'), [Identifier('i')]),
                value=BinaryOp('*', Identifier('i'), NumberLiteral('2'))
            ))]
        )
        assert can_jit_loop(loop) is True

    def test_cannot_jit_with_if(self):
        """can_jit_loop returns False for loops with control flow."""
        from forge.engine.jit_compiler import can_jit_loop
        from forge.engine.parser import (
            ForStatement, IfStatement, NumberLiteral, ColonExpr,
        )
        loop = ForStatement(
            var='i',
            iter_expr=ColonExpr(NumberLiteral('1'), NumberLiteral('10')),
            body=[IfStatement(
                condition=NumberLiteral('1'),
                body=[],
                elseifs=[],
            )]
        )
        assert can_jit_loop(loop) is False

    def test_cache_reuse(self, s):
        """Running the same loop pattern twice should use cache."""
        from forge.engine.jit_compiler import _jit_cache
        before = len(_jit_cache)
        s.eval("for i = 1:10; aa(i) = sin(i); end")
        after1 = len(_jit_cache)
        s.eval("for i = 1:10; aa(i) = sin(i); end")
        after2 = len(_jit_cache)
        # Cache should have grown by at most 1 (first run compiles, second reuses)
        assert after2 == after1


class TestJITBenchmark:
    """Benchmark JIT vs interpreted (informational, not strict pass/fail)."""

    def test_million_iteration_benchmark(self, s):
        """Time a large loop - JIT should be significantly faster."""
        code = "for i = 1:1000000; result(i) = sin(i) * cos(i); end"

        # Run once to warm up JIT cache
        s.eval(code)
        result1 = get_val(s, "result").ravel().copy()

        # Time second run (uses cached JIT)
        start = time.time()
        s.eval(code)
        elapsed = time.time() - start
        result2 = get_val(s, "result").ravel()

        # Verify correctness
        np.testing.assert_allclose(result1, result2, rtol=1e-12)

        # Just print timing - the test passes either way
        print(f"\n  1M iteration sin*cos loop: {elapsed:.3f}s")
        expected = np.array([np.sin(i) * np.cos(i) for i in range(1, 1000001)])
        np.testing.assert_allclose(result2, expected, rtol=1e-10)


# Need to import BinaryOp for test_can_jit_simple
from forge.engine.parser import BinaryOp
