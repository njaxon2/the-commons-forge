# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Tests for expression fusion / in-place ufunc optimization.

Requirement R-PERF-01: The expression fusion system SHALL combine element-wise
binary operations on large arrays into single-pass buffer-cached ufuncs,
producing results numerically identical to unfused evaluation while avoiding
unnecessary memory allocation.

Model-user argument: An engineer processing large signal or simulation data
(10,000+ element arrays) expects element-wise operations like "a .* b + c ./ d"
to execute efficiently. In MATLAB/Octave, vectorized operations are the
idiomatic performance pattern. Expression fusion makes these operations faster
by reducing temporary array allocations and improving cache locality, without
the user needing to change their code or even know the optimization exists.

Decomposition:
    R-PERF-01  Fused element-wise operations on large arrays produce correct
               results (correctness through the session evaluator).
    R-PERF-02  The expr_fusion module API correctly identifies fusible vs.
               non-fusible operations and produces correct direct results.
    R-PERF-03  Fused operations perform correctly under repeated/stress
               execution (performance-like scenarios).

Consistency argument: R-PERF-01 tests correctness end-to-end through the
session evaluator for all supported operators, chained expressions, scalar
broadcast, small-array bypass, and transcendental functions. R-PERF-02 tests
the fusion module API directly: eligibility checks (can_fuse), direct binop
execution (fused_binop), aliasing safety, and cache management. R-PERF-03
verifies correctness is maintained under repeated execution in loops, which
exercises the buffer cache reuse path. Together these ensure the optimization
is invisible to the user while being correct and safe.
"""
import pytest
import numpy as np
import time
from forge.engine.evaluator import Session
from forge.engine.types import ForgeArray, _unwrap
from forge.engine import expr_fusion


@pytest.fixture
def s():
    return Session()


class TestFusionCorrectness:
    """R-PERF-01: Fused element-wise operations SHALL produce results
    numerically identical to standard (unfused) evaluation for all supported
    operators and expression patterns.

    Model-user argument: The engineer writes expressions like "z = x + y" or
    "d = a .* b + c ./ d" and expects correct results regardless of whether
    fusion is active. Any discrepancy between fused and unfused results would
    produce silent numerical errors in the engineer's analysis.

    Decomposition:
        R-PERF-01.1   Element-wise addition on large arrays.
        R-PERF-01.2   Element-wise subtraction on large arrays.
        R-PERF-01.3   Element-wise dot-multiply on large arrays.
        R-PERF-01.4   Element-wise dot-divide on large arrays.
        R-PERF-01.5   Element-wise dot-power on large arrays.
        R-PERF-01.6   Chained expression (a .* b + c).
        R-PERF-01.7   Four-operator chain with no buffer aliasing.
        R-PERF-01.8   Scalar broadcast (x + 1) on large array.
        R-PERF-01.9   Small arrays bypass fusion and still produce correct results.
        R-PERF-01.10  Transcendental functions (sin, cos) through fusion.
        R-PERF-01.11  Repeated evaluation produces correct results each time.

    Consistency: Covers all five supported element-wise operators individually,
    then chained and mixed expressions, scalar broadcast, the small-array
    bypass path, transcendentals, and loop stability. This exhaustively tests
    correctness across the fusion decision boundary and all operator types.
    """

    def test_elementwise_add_large(self, s):
        """R-PERF-01.1: Element-wise addition on 1000-element arrays."""
        s.eval("x = rand(1000, 1); y = rand(1000, 1)")
        s.eval("z = x + y")
        x = _unwrap(s.workspace.get("x"))
        y = _unwrap(s.workspace.get("y"))
        z = _unwrap(s.workspace.get("z"))
        np.testing.assert_allclose(z, x + y)

    def test_elementwise_sub_large(self, s):
        """R-PERF-01.2: Element-wise subtraction on 1000-element arrays."""
        s.eval("x = rand(500, 2); y = rand(500, 2)")
        s.eval("z = x - y")
        x = _unwrap(s.workspace.get("x"))
        y = _unwrap(s.workspace.get("y"))
        z = _unwrap(s.workspace.get("z"))
        np.testing.assert_allclose(z, x - y)

    def test_dotmul_large(self, s):
        """R-PERF-01.3: Element-wise dot-multiply on 200x200 arrays."""
        s.eval("a = rand(200, 200); b = rand(200, 200)")
        s.eval("c = a .* b")
        a = _unwrap(s.workspace.get("a"))
        b = _unwrap(s.workspace.get("b"))
        c = _unwrap(s.workspace.get("c"))
        np.testing.assert_allclose(c, a * b)

    def test_dotdiv_large(self, s):
        """R-PERF-01.4: Element-wise dot-divide on 200x200 arrays."""
        s.eval("a = rand(200, 200) + 0.1; b = rand(200, 200) + 0.1")
        s.eval("c = a ./ b")
        a = _unwrap(s.workspace.get("a"))
        b = _unwrap(s.workspace.get("b"))
        c = _unwrap(s.workspace.get("c"))
        np.testing.assert_allclose(c, a / b)

    def test_dotpow_large(self, s):
        """R-PERF-01.5: Element-wise dot-power on 100x100 arrays."""
        s.eval("a = rand(100, 100) + 0.5; b = rand(100, 100) + 0.5")
        s.eval("c = a .^ b")
        a = _unwrap(s.workspace.get("a"))
        b = _unwrap(s.workspace.get("b"))
        c = _unwrap(s.workspace.get("c"))
        np.testing.assert_allclose(c, a ** b)

    def test_chained_expression(self, s):
        """R-PERF-01.6: Chained a .* b + c produces correct result."""
        s.eval("a = rand(500, 1); b = rand(500, 1); c = rand(500, 1)")
        s.eval("d = a .* b + c")
        a = _unwrap(s.workspace.get("a"))
        b = _unwrap(s.workspace.get("b"))
        c = _unwrap(s.workspace.get("c"))
        d = _unwrap(s.workspace.get("d"))
        np.testing.assert_allclose(d, a * b + c)

    def test_chained_four_ops(self, s):
        """R-PERF-01.7: Four-operator chain with no buffer aliasing."""
        s.eval("a = rand(5000, 1); b = rand(5000, 1)")
        s.eval("c = rand(5000, 1); d = rand(5000, 1) + 0.1")
        s.eval("z = a .* b + c ./ d")
        a = _unwrap(s.workspace.get("a"))
        b = _unwrap(s.workspace.get("b"))
        c = _unwrap(s.workspace.get("c"))
        d = _unwrap(s.workspace.get("d"))
        z = _unwrap(s.workspace.get("z"))
        np.testing.assert_allclose(z, a * b + c / d, atol=1e-12)

    def test_broadcast_scalar(self, s):
        """R-PERF-01.8: Scalar broadcast (x + 1) on large array."""
        s.eval("x = rand(1000, 1)")
        s.eval("y = x + 1")
        x = _unwrap(s.workspace.get("x"))
        y = _unwrap(s.workspace.get("y"))
        np.testing.assert_allclose(y, x + 1.0)

    def test_small_array_no_fusion(self, s):
        """R-PERF-01.9: Small arrays bypass fusion and still produce correct results."""
        s.eval("a = [1 2 3]; b = [4 5 6]")
        s.eval("c = a + b")
        c = _unwrap(s.workspace.get("c"))
        np.testing.assert_allclose(c.ravel(), [5, 7, 9])

    def test_sin_cos_expression(self, s):
        """R-PERF-01.10: sin(x) + cos(x) produces correct results through fusion."""
        s.eval("x = rand(10000, 1)")
        s.eval("y = sin(x) + cos(x)")
        x = _unwrap(s.workspace.get("x"))
        y = _unwrap(s.workspace.get("y"))
        np.testing.assert_allclose(y, np.sin(x) + np.cos(x), atol=1e-12)

    def test_repeated_eval_in_loop(self, s):
        """R-PERF-01.11: Repeated evaluation produces correct results each iteration."""
        s.eval("x = rand(1000, 1); y = rand(1000, 1)")
        for i in range(5):
            s.eval("z = x + y")
            x = _unwrap(s.workspace.get("x"))
            y = _unwrap(s.workspace.get("y"))
            z = _unwrap(s.workspace.get("z"))
            np.testing.assert_allclose(z, x + y)


class TestFusionModule:
    """R-PERF-02: The expr_fusion module SHALL correctly distinguish fusible
    from non-fusible operations, produce correct results via fused_binop, and
    manage output buffers without aliasing.

    Model-user argument: The fusion module is the internal engine that decides
    which operations to optimize. It must never fuse matrix multiply ("*") or
    matrix divide ("/") because those have different semantics than element-wise
    operations. It must also avoid fusing small arrays (where the overhead
    exceeds the benefit) and non-numeric types (where ufuncs are not applicable).

    Decomposition:
        R-PERF-02.1   can_fuse returns True for all element-wise operators.
        R-PERF-02.2   can_fuse returns False for matrix operators.
        R-PERF-02.3   can_fuse returns False for small arrays.
        R-PERF-02.4   can_fuse returns False for non-numeric arrays.
        R-PERF-02.5   fused_binop addition produces correct result.
        R-PERF-02.6   fused_binop subtraction produces correct result.
        R-PERF-02.7   fused_binop dot-multiply produces correct result.
        R-PERF-02.8   fused_binop dot-divide produces correct result.
        R-PERF-02.9   fused_binop dot-power produces correct result.
        R-PERF-02.10  Consecutive fused_binop calls do not alias output arrays.
        R-PERF-02.11  clear_caches empties the shape cache.

    Consistency: R-PERF-02.1 through R-PERF-02.4 exhaustively test the
    eligibility gate (operator type, array size, element type). R-PERF-02.5
    through R-PERF-02.9 verify each operator through the fused path. R-PERF-02.10
    confirms memory safety. R-PERF-02.11 confirms cache management. Together
    these ensure the module correctly guards and executes all fusion paths.
    """

    def test_can_fuse_elementwise(self):
        """R-PERF-02.1: can_fuse returns True for all element-wise operators."""
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        assert expr_fusion.can_fuse("+", a, b)
        assert expr_fusion.can_fuse("-", a, b)
        assert expr_fusion.can_fuse(".*", a, b)
        assert expr_fusion.can_fuse("./", a, b)
        assert expr_fusion.can_fuse(".^", a, b)

    def test_cannot_fuse_matrix_ops(self):
        """R-PERF-02.2: can_fuse returns False for matrix operators."""
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        assert not expr_fusion.can_fuse("*", a, b)
        assert not expr_fusion.can_fuse("/", a, b)
        assert not expr_fusion.can_fuse("^", a, b)

    def test_cannot_fuse_small_arrays(self):
        """R-PERF-02.3: can_fuse returns False for below-threshold arrays."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert not expr_fusion.can_fuse("+", a, b)

    def test_cannot_fuse_non_numeric(self):
        """R-PERF-02.4: can_fuse returns False for boolean arrays."""
        a = np.array([True, False, True] * 50)
        b = np.array([False, True, False] * 50)
        assert not expr_fusion.can_fuse("+", a, b)

    def test_fused_binop_add(self):
        """R-PERF-02.5: fused_binop addition matches numpy."""
        a = np.random.rand(200, 200)
        b = np.random.rand(200, 200)
        result = expr_fusion.fused_binop("+", a, b)
        np.testing.assert_allclose(result, a + b)

    def test_fused_binop_sub(self):
        """R-PERF-02.6: fused_binop subtraction matches numpy."""
        a = np.random.rand(200, 200)
        b = np.random.rand(200, 200)
        result = expr_fusion.fused_binop("-", a, b)
        np.testing.assert_allclose(result, a - b)

    def test_fused_binop_dotmul(self):
        """R-PERF-02.7: fused_binop dot-multiply matches numpy."""
        a = np.random.rand(200, 200)
        b = np.random.rand(200, 200)
        result = expr_fusion.fused_binop(".*", a, b)
        np.testing.assert_allclose(result, a * b)

    def test_fused_binop_dotdiv(self):
        """R-PERF-02.8: fused_binop dot-divide matches numpy."""
        a = np.random.rand(200, 200) + 0.1
        b = np.random.rand(200, 200) + 0.1
        result = expr_fusion.fused_binop("./", a, b)
        np.testing.assert_allclose(result, a / b)

    def test_fused_binop_dotpow(self):
        """R-PERF-02.9: fused_binop dot-power matches numpy."""
        a = np.random.rand(200, 200) + 0.5
        b = np.random.rand(200, 200) + 0.5
        result = expr_fusion.fused_binop(".^", a, b)
        np.testing.assert_allclose(result, a ** b)

    def test_no_aliasing_between_calls(self):
        """R-PERF-02.10: Consecutive fused_binop calls do not alias output arrays."""
        a = np.random.rand(200, 200)
        b = np.random.rand(200, 200)
        c = np.random.rand(200, 200)
        r1 = expr_fusion.fused_binop("+", a, b)
        r2 = expr_fusion.fused_binop("+", a, c)
        np.testing.assert_allclose(r1, a + b)
        np.testing.assert_allclose(r2, a + c)
        assert not np.shares_memory(r1, r2)

    def test_clear_caches(self):
        """R-PERF-02.11: clear_caches empties the shape cache."""
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        expr_fusion.fused_binop("+", a, b)
        expr_fusion.clear_caches()
        assert len(expr_fusion._shape_cache._cache) == 0


class TestFusionPerformance:
    """R-PERF-03: Fused operations SHALL produce correct results under repeated
    execution in performance-like scenarios (stress testing the buffer cache).

    Model-user argument: The engineer running iterative algorithms (e.g.
    gradient descent, time-stepping simulations) evaluates the same expression
    pattern thousands of times. The fusion buffer cache must not corrupt results
    across iterations.

    Decomposition:
        R-PERF-03.1  sin(x)+cos(x) correct after 50 repeated evaluations.
        R-PERF-03.2  a.*b+c./d correct after 50 repeated evaluations.

    Consistency: These two cases cover single-function chains (transcendentals)
    and multi-operator chains (arithmetic) under repeated execution, exercising
    both the ufunc path and the buffer reuse path.
    """

    def test_sin_cos_loop(self, s):
        """R-PERF-03.1: sin(x)+cos(x) correct after 50 iterations on 10k array."""
        s.eval("x = rand(10000, 1)")
        start = time.perf_counter()
        for _ in range(50):
            s.eval("y = sin(x) + cos(x)")
        elapsed = time.perf_counter() - start
        x = _unwrap(s.workspace.get("x"))
        y = _unwrap(s.workspace.get("y"))
        np.testing.assert_allclose(y, np.sin(x) + np.cos(x), atol=1e-12)
        print(f"\n  [perf] 50x sin(x)+cos(x) on 10k: {elapsed:.3f}s")

    def test_chained_ops_loop(self, s):
        """R-PERF-03.2: a.*b+c./d correct after 50 iterations on 5k arrays."""
        s.eval("a = rand(5000, 1); b = rand(5000, 1)")
        s.eval("c = rand(5000, 1); d = rand(5000, 1) + 0.1")
        start = time.perf_counter()
        for _ in range(50):
            s.eval("z = a .* b + c ./ d")
        elapsed = time.perf_counter() - start
        a = _unwrap(s.workspace.get("a"))
        b = _unwrap(s.workspace.get("b"))
        c = _unwrap(s.workspace.get("c"))
        d = _unwrap(s.workspace.get("d"))
        z = _unwrap(s.workspace.get("z"))
        np.testing.assert_allclose(z, a * b + c / d, atol=1e-12)
        print(f"\n  [perf] 50x a.*b+c./d on 5k: {elapsed:.3f}s")
