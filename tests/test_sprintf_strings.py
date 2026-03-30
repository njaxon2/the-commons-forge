# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for sprintf/fprintf with string arguments and other format fixes."""
import pytest
import sys
sys.path.insert(0, ".")
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


class TestSprintfStrings:
    """sprintf must handle ForgeChar (%s) before ForgeArray branch."""

    def test_sprintf_string_arg(self, s):
        r = s.eval("sprintf(\"%s\", \"hello\")")
        assert "hello" in str(r)

    def test_sprintf_string_in_format(self, s):
        r = s.eval("sprintf(\"Name: %s\", \"Alice\")")
        assert "Alice" in str(r)

    def test_sprintf_mixed_types(self, s):
        r = s.eval("sprintf(\"%s is %d years old\", \"Bob\", 30)")
        assert "Bob" in str(r)
        assert "30" in str(r)

    def test_sprintf_multiple_strings(self, s):
        r = s.eval("sprintf(\"%s %s\", \"hello\", \"world\")")
        assert "hello world" in str(r)

    def test_sprintf_float(self, s):
        r = s.eval("sprintf(\"%.2f\", 3.14159)")
        assert "3.14" in str(r)

    def test_sprintf_int(self, s):
        r = s.eval("sprintf(\"%d\", 42)")
        assert "42" in str(r)

    def test_sprintf_multiple_formats(self, s):
        r = s.eval("sprintf(\"%d + %d = %d\", 2, 3, 5)")
        assert "2 + 3 = 5" in str(r)


class TestCtranspose:
    """ctranspose (conjugate transpose) for complex matrices."""

    def test_ctranspose_real(self, s):
        r = s.eval("ctranspose([1 2; 3 4])")
        assert "1" in str(r) and "3" in str(r)

    def test_ctranspose_complex(self, s):
        # Conjugate transpose negates imaginary part
        r = s.eval("x = ctranspose([1+2i])")
        assert "1 - 2i" in str(r) or "1-2i" in str(r) or "1.-2.j" in str(r).lower()


class TestRealisticScripts:
    """Multi-line scripts that real users would write."""

    def test_fibonacci_recursive(self, s):
        script = """
function result = fib(n)
  if n <= 1
    result = n;
  else
    result = fib(n-1) + fib(n-2);
  end
end
result = fib(10);
"""
        s.eval(script)
        r = s.workspace.get("result")
        from forge.engine.types import ForgeArray
        val = float(r.data.flat[0]) if isinstance(r, ForgeArray) else float(r)
        assert val == 55.0

    def test_gauss_seidel(self, s):
        script = """
A = [4 -1 0; -1 4 -1; 0 -1 4];
b = [1; 2; 3];
x = zeros(3, 1);
for iter = 1:50
  for i = 1:3
    sigma = 0;
    for j = 1:3
      if j ~= i
        sigma = sigma + A(i,j) * x(j);
      end
    end
    x(i) = (b(i) - sigma) / A(i,i);
  end
end
residual = norm(A * x - b);
"""
        s.eval(script)
        r = s.workspace.get("residual")
        from forge.engine.types import ForgeArray
        val = float(r.data.flat[0]) if isinstance(r, ForgeArray) else float(r)
        assert val < 1e-10

    def test_higher_order_function(self, s):
        script = """
function y = apply(f, x)
  y = f(x);
end
result = apply(@(x) x.^2 + 1, [1 2 3 4 5]);
"""
        s.eval(script)
        r = s.workspace.get("result")
        import numpy as np
        from forge.engine.types import ForgeArray
        data = r.data if isinstance(r, ForgeArray) else np.asarray(r)
        assert list(data.flat) == [2.0, 5.0, 10.0, 17.0, 26.0]

    def test_signal_fft(self, s):
        script = """
t = linspace(0, 1, 256);
signal = sin(2*pi*10*t);
Y = fft(signal);
n = length(Y);
"""
        s.eval(script)
        r = s.workspace.get("n")
        from forge.engine.types import ForgeArray
        val = float(r.data.flat[0]) if isinstance(r, ForgeArray) else float(r)
        assert val == 256.0


class TestEdgeCases:
    """Edge cases in Octave syntax."""

    def test_empty_for(self, s):
        r = s.eval("x = 0; for i = []; x = x + 1; end; x")
        assert "0" in str(r)

    def test_nested_struct(self, s):
        r = s.eval("st.inner.x = 42; st.inner.x")
        assert "42" in str(r)

    def test_line_continuation(self, s):
        r = s.eval("x = 1 + ...\n2 + ...\n3")
        assert "6" in str(r)

    def test_multiple_semicolons(self, s):
        r = s.eval(";;;a = 1;;; a")
        assert "1" in str(r)

    def test_power_precedence(self, s):
        r = s.eval("2^3^2")
        # In Octave, ^ is right-associative: 2^(3^2) = 2^9 = 512
        assert "512" in str(r)

    def test_unary_minus_power(self, s):
        r = s.eval("-2^2")
        # -2^2 = -(2^2) = -4 (not (-2)^2 = 4)
        assert "-4" in str(r)
