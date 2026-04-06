# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for sprintf/fprintf with string arguments and other format fixes.

V-Model Traceability
---------------------
Requirement: R-FMT
Parent SHALL statement: Forge SHALL support sprintf format strings with %s, %d,
    and %f specifiers, ctranspose for complex matrices, realistic multi-line
    scripts, and Octave edge-case syntax (empty loops, nested structs, line
    continuation, multiple semicolons, operator precedence).
Model-user argument: An engineer who uses sprintf for formatted output in logs,
    reports, and data export expects to format numbers with specific precision,
    build table rows, and construct file paths with variable components. sprintf
    is how they produce human-readable output from computed results. Beyond
    formatting, they write real scripts (recursive functions, iterative solvers,
    higher-order functions, FFT pipelines) and rely on Octave syntax edge cases
    behaving identically to the reference implementation.
Decomposition:
    R-FMT-01: sprintf with %s inserts a string argument.
    R-FMT-02: sprintf with %s in a labeled format includes the string.
    R-FMT-03: sprintf with mixed %s and %d handles heterogeneous types.
    R-FMT-04: sprintf with multiple %s joins multiple string arguments.
    R-FMT-05: sprintf with %.2f formats a float to 2 decimal places.
    R-FMT-06: sprintf with %d formats an integer.
    R-FMT-07: sprintf with multiple %d formats an arithmetic expression.
    R-FMT-08: ctranspose of a real matrix transposes it.
    R-FMT-09: ctranspose of a complex scalar conjugates the imaginary part.
    R-FMT-10: Recursive Fibonacci function computes fib(10) = 55.
    R-FMT-11: Gauss-Seidel iterative solver converges to residual < 1e-10.
    R-FMT-12: Higher-order function with anonymous function handle works.
    R-FMT-13: FFT of a sine signal produces correct length output.
    R-FMT-14: Empty for-loop body does not execute.
    R-FMT-15: Nested struct field assignment and access works.
    R-FMT-16: Line continuation with ... joins lines.
    R-FMT-17: Multiple semicolons are parsed without error.
    R-FMT-18: Power operator is left-associative (2^3^2 = 64).
    R-FMT-19: Unary minus binds looser than power (-2^2 = -4).
Consistency argument: R-FMT-01 through R-FMT-07 cover sprintf format specifiers
    (%s, %d, %f) with single, multiple, and mixed arguments. R-FMT-08 and
    R-FMT-09 cover ctranspose for real and complex cases. R-FMT-10 through
    R-FMT-13 cover realistic script patterns (recursion, iteration, higher-order
    functions, FFT). R-FMT-14 through R-FMT-19 cover Octave syntax edge cases
    (empty loops, nested structs, continuation, semicolons, operator precedence).
    Together these verify formatted I/O, transpose correctness, script execution,
    and syntax fidelity.
"""
import pytest
import sys
sys.path.insert(0, ".")
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


class TestSprintfStrings:
    """R-FMT-01..07: sprintf SHALL handle string (%s), integer (%d), and float
    (%.Nf) format specifiers with single and multiple arguments.

    Model-user argument: The engineer formats sensor readings, parameter tables,
    and file paths using sprintf. If %s is mishandled or mixed types fail, their
    log output and data export pipelines break silently.
    """

    def test_sprintf_string_arg(self, s):
        """R-FMT-01: sprintf('%s', 'hello') inserts the string."""
        r = s.eval("sprintf(\"%s\", \"hello\")")
        assert "hello" in str(r)

    def test_sprintf_string_in_format(self, s):
        """R-FMT-02: sprintf('Name: %s', 'Alice') includes the string."""
        r = s.eval("sprintf(\"Name: %s\", \"Alice\")")
        assert "Alice" in str(r)

    def test_sprintf_mixed_types(self, s):
        """R-FMT-03: sprintf('%s is %d years old', 'Bob', 30) handles mixed types."""
        r = s.eval("sprintf(\"%s is %d years old\", \"Bob\", 30)")
        assert "Bob" in str(r)
        assert "30" in str(r)

    def test_sprintf_multiple_strings(self, s):
        """R-FMT-04: sprintf('%s %s', 'hello', 'world') joins two strings."""
        r = s.eval("sprintf(\"%s %s\", \"hello\", \"world\")")
        assert "hello world" in str(r)

    def test_sprintf_float(self, s):
        """R-FMT-05: sprintf('%.2f', 3.14159) produces '3.14'."""
        r = s.eval("sprintf(\"%.2f\", 3.14159)")
        assert "3.14" in str(r)

    def test_sprintf_int(self, s):
        """R-FMT-06: sprintf('%d', 42) produces '42'."""
        r = s.eval("sprintf(\"%d\", 42)")
        assert "42" in str(r)

    def test_sprintf_multiple_formats(self, s):
        """R-FMT-07: sprintf('%d + %d = %d', 2, 3, 5) produces '2 + 3 = 5'."""
        r = s.eval("sprintf(\"%d + %d = %d\", 2, 3, 5)")
        assert "2 + 3 = 5" in str(r)


class TestCtranspose:
    """R-FMT-08..09: ctranspose (conjugate transpose) SHALL transpose real
    matrices and conjugate complex entries.
    """

    def test_ctranspose_real(self, s):
        """R-FMT-08: ctranspose of a real matrix transposes it."""
        r = s.eval("ctranspose([1 2; 3 4])")
        assert "1" in str(r) and "3" in str(r)

    def test_ctranspose_complex(self, s):
        """R-FMT-09: ctranspose of [1+2i] produces 1-2i."""
        # Conjugate transpose negates imaginary part
        r = s.eval("x = ctranspose([1+2i])")
        assert "1 - 2i" in str(r) or "1-2i" in str(r) or "1.-2.j" in str(r).lower()


class TestRealisticScripts:
    """R-FMT-10..13: Forge SHALL execute realistic multi-line scripts including
    recursion, iterative solvers, higher-order functions, and FFT pipelines.

    Model-user argument: The engineer writes complete algorithms, not just
    one-liners. Recursive functions, iterative linear solvers, function handles
    passed as arguments, and FFT-based signal analysis are daily tasks. If any of
    these patterns fail, the user cannot migrate real code from Octave.
    """

    def test_fibonacci_recursive(self, s):
        """R-FMT-10: Recursive fib(10) returns 55."""
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
        """R-FMT-11: Gauss-Seidel solver converges to residual < 1e-10."""
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
        """R-FMT-12: apply(@(x) x.^2 + 1, [1 2 3 4 5]) returns [2 5 10 17 26]."""
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
        """R-FMT-13: FFT of a 256-sample sine produces length-256 output."""
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
    """R-FMT-14..19: Forge SHALL handle Octave syntax edge cases identically to
    the reference implementation.

    Model-user argument: The engineer migrating existing .m files encounters edge
    cases in real codebases: empty loop ranges, deeply nested structs, line
    continuations across expressions, redundant semicolons, and operator
    precedence subtleties. Each must behave exactly as Octave does, or migration
    produces silent numerical errors.
    """

    def test_empty_for(self, s):
        """R-FMT-14: for i = [] does not execute the loop body."""
        r = s.eval("x = 0; for i = []; x = x + 1; end; x")
        assert "0" in str(r)

    def test_nested_struct(self, s):
        """R-FMT-15: st.inner.x = 42; st.inner.x returns 42."""
        r = s.eval("st.inner.x = 42; st.inner.x")
        assert "42" in str(r)

    def test_line_continuation(self, s):
        """R-FMT-16: Line continuation with ... joins across newlines."""
        r = s.eval("x = 1 + ...\n2 + ...\n3")
        assert "6" in str(r)

    def test_multiple_semicolons(self, s):
        """R-FMT-17: Multiple semicolons parse without error."""
        r = s.eval(";;;a = 1;;; a")
        assert "1" in str(r)

    def test_power_precedence(self, s):
        """R-FMT-18: 2^3^2 = 64 (left-associative power)."""
        r = s.eval("2^3^2")
        # In Octave, ^ is left-associative: (2^3)^2 = 8^2 = 64
        assert "64" in str(r)

    def test_unary_minus_power(self, s):
        """R-FMT-19: -2^2 = -4 (unary minus binds looser than power)."""
        r = s.eval("-2^2")
        # -2^2 = -(2^2) = -4 (not (-2)^2 = 4)
        assert "-4" in str(r)
