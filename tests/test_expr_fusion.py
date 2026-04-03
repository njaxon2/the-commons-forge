# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Tests for expression fusion / in-place ufunc optimization."""
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
    """Fused expressions must produce identical results to unfused."""

    def test_elementwise_add_large(self, s):
        s.eval("x = rand(1000, 1); y = rand(1000, 1)")
        s.eval("z = x + y")
        x = _unwrap(s.workspace.get("x"))
        y = _unwrap(s.workspace.get("y"))
        z = _unwrap(s.workspace.get("z"))
        np.testing.assert_allclose(z, x + y)

    def test_elementwise_sub_large(self, s):
        s.eval("x = rand(500, 2); y = rand(500, 2)")
        s.eval("z = x - y")
        x = _unwrap(s.workspace.get("x"))
        y = _unwrap(s.workspace.get("y"))
        z = _unwrap(s.workspace.get("z"))
        np.testing.assert_allclose(z, x - y)

    def test_dotmul_large(self, s):
        s.eval("a = rand(200, 200); b = rand(200, 200)")
        s.eval("c = a .* b")
        a = _unwrap(s.workspace.get("a"))
        b = _unwrap(s.workspace.get("b"))
        c = _unwrap(s.workspace.get("c"))
        np.testing.assert_allclose(c, a * b)

    def test_dotdiv_large(self, s):
        s.eval("a = rand(200, 200) + 0.1; b = rand(200, 200) + 0.1")
        s.eval("c = a ./ b")
        a = _unwrap(s.workspace.get("a"))
        b = _unwrap(s.workspace.get("b"))
        c = _unwrap(s.workspace.get("c"))
        np.testing.assert_allclose(c, a / b)

    def test_dotpow_large(self, s):
        s.eval("a = rand(100, 100) + 0.5; b = rand(100, 100) + 0.5")
        s.eval("c = a .^ b")
        a = _unwrap(s.workspace.get("a"))
        b = _unwrap(s.workspace.get("b"))
        c = _unwrap(s.workspace.get("c"))
        np.testing.assert_allclose(c, a ** b)

    def test_chained_expression(self, s):
        """Chained element-wise: a .* b + c produces correct result."""
        s.eval("a = rand(500, 1); b = rand(500, 1); c = rand(500, 1)")
        s.eval("d = a .* b + c")
        a = _unwrap(s.workspace.get("a"))
        b = _unwrap(s.workspace.get("b"))
        c = _unwrap(s.workspace.get("c"))
        d = _unwrap(s.workspace.get("d"))
        np.testing.assert_allclose(d, a * b + c)

    def test_chained_four_ops(self, s):
        """a .* b + c ./ d must produce correct result (no buffer aliasing)."""
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
        """Fusion with scalar broadcast: x + 1 on large array."""
        s.eval("x = rand(1000, 1)")
        s.eval("y = x + 1")
        x = _unwrap(s.workspace.get("x"))
        y = _unwrap(s.workspace.get("y"))
        np.testing.assert_allclose(y, x + 1.0)

    def test_small_array_no_fusion(self, s):
        """Small arrays bypass fusion (below threshold) and still work."""
        s.eval("a = [1 2 3]; b = [4 5 6]")
        s.eval("c = a + b")
        c = _unwrap(s.workspace.get("c"))
        np.testing.assert_allclose(c.ravel(), [5, 7, 9])

    def test_sin_cos_expression(self, s):
        """sin(x) + cos(x) produces correct results through fusion."""
        s.eval("x = rand(10000, 1)")
        s.eval("y = sin(x) + cos(x)")
        x = _unwrap(s.workspace.get("x"))
        y = _unwrap(s.workspace.get("y"))
        np.testing.assert_allclose(y, np.sin(x) + np.cos(x), atol=1e-12)

    def test_repeated_eval_in_loop(self, s):
        """Repeated evaluation produces correct results each time."""
        s.eval("x = rand(1000, 1); y = rand(1000, 1)")
        for i in range(5):
            s.eval("z = x + y")
            x = _unwrap(s.workspace.get("x"))
            y = _unwrap(s.workspace.get("y"))
            z = _unwrap(s.workspace.get("z"))
            np.testing.assert_allclose(z, x + y)


class TestFusionModule:
    """Direct tests of the expr_fusion module."""

    def test_can_fuse_elementwise(self):
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        assert expr_fusion.can_fuse("+", a, b)
        assert expr_fusion.can_fuse("-", a, b)
        assert expr_fusion.can_fuse(".*", a, b)
        assert expr_fusion.can_fuse("./", a, b)
        assert expr_fusion.can_fuse(".^", a, b)

    def test_cannot_fuse_matrix_ops(self):
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        assert not expr_fusion.can_fuse("*", a, b)
        assert not expr_fusion.can_fuse("/", a, b)
        assert not expr_fusion.can_fuse("^", a, b)

    def test_cannot_fuse_small_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert not expr_fusion.can_fuse("+", a, b)

    def test_cannot_fuse_non_numeric(self):
        a = np.array([True, False, True] * 50)
        b = np.array([False, True, False] * 50)
        assert not expr_fusion.can_fuse("+", a, b)

    def test_fused_binop_add(self):
        a = np.random.rand(200, 200)
        b = np.random.rand(200, 200)
        result = expr_fusion.fused_binop("+", a, b)
        np.testing.assert_allclose(result, a + b)

    def test_fused_binop_sub(self):
        a = np.random.rand(200, 200)
        b = np.random.rand(200, 200)
        result = expr_fusion.fused_binop("-", a, b)
        np.testing.assert_allclose(result, a - b)

    def test_fused_binop_dotmul(self):
        a = np.random.rand(200, 200)
        b = np.random.rand(200, 200)
        result = expr_fusion.fused_binop(".*", a, b)
        np.testing.assert_allclose(result, a * b)

    def test_fused_binop_dotdiv(self):
        a = np.random.rand(200, 200) + 0.1
        b = np.random.rand(200, 200) + 0.1
        result = expr_fusion.fused_binop("./", a, b)
        np.testing.assert_allclose(result, a / b)

    def test_fused_binop_dotpow(self):
        a = np.random.rand(200, 200) + 0.5
        b = np.random.rand(200, 200) + 0.5
        result = expr_fusion.fused_binop(".^", a, b)
        np.testing.assert_allclose(result, a ** b)

    def test_no_aliasing_between_calls(self):
        """Two consecutive fused_binop calls must not alias output arrays."""
        a = np.random.rand(200, 200)
        b = np.random.rand(200, 200)
        c = np.random.rand(200, 200)
        r1 = expr_fusion.fused_binop("+", a, b)
        r2 = expr_fusion.fused_binop("+", a, c)
        np.testing.assert_allclose(r1, a + b)
        np.testing.assert_allclose(r2, a + c)
        assert not np.shares_memory(r1, r2)

    def test_clear_caches(self):
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        expr_fusion.fused_binop("+", a, b)
        expr_fusion.clear_caches()
        assert len(expr_fusion._shape_cache._cache) == 0


class TestFusionPerformance:
    """Verify fusion runs correctly in performance-like scenarios."""

    def test_sin_cos_loop(self, s):
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
