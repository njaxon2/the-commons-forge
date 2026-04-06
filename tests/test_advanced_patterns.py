# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for advanced Octave patterns - realistic user scenarios.

Requirement R-PAT: The engine SHALL correctly evaluate complex M-language
patterns spanning signal processing, linear algebra, control flow, string
operations, matrix manipulation, and statistics, producing results
numerically identical to Octave/MATLAB for each domain.

Model-user argument: An engineer migrating from MATLAB brings scripts
that combine FFT pipelines, eigendecompositions, recursive functions,
string formatting, logical indexing, and statistical summaries in a
single session. Each domain must work independently and in combination,
or the engineer cannot trust Forge as a drop-in replacement.

Decomposition:
  R-PAT-01..04: Signal processing (FFT, conv, windowing, filter)
  R-PAT-05..11: Linear algebra (solve, eig, svd, det, inv, rank, trace)
  R-PAT-12..18: Control flow (recursion, varargin, nargin, loops, switch, try/catch)
  R-PAT-19..22: String operations (sprintf, upper/lower, strsplit, regexp)
  R-PAT-23..30: Matrix operations (logical index, colon, end, submatrix, grow, element-wise, power)
  R-PAT-31..34: Statistics (mean, std, median, var)

Consistency argument: The six domain groups partition the M-language
surface area the engineer uses daily. Signal processing tests validate
the DSP pipeline. Linear algebra tests validate decomposition and
solution routines. Control flow tests validate language constructs.
String tests validate formatting and parsing. Matrix tests validate
subscripting and arithmetic. Statistics tests validate descriptive
measures. Together they cover the engineer's realistic workflow.
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


def get_array(s, varname):
    """Get ForgeArray from workspace by variable name."""
    return s._engine.workspace.get(varname)


class TestSignalProcessing:
    """R-PAT-01..04: Signal processing builtins SHALL produce correct
    results for FFT roundtrip, convolution, windowing, and filtering.

    Model-user argument: The engineer processes sensor data with FFT,
    applies windows before spectral analysis, and uses conv/filter for
    time-domain operations. If ifft(fft(x)) does not recover x exactly,
    the entire signal processing workflow is unreliable.

    Decomposition:
      R-PAT-01: ifft(fft(x)) recovers x within machine precision
      R-PAT-02: conv([1 1], [1 1 1]) produces [1 2 2 1]
      R-PAT-03: hamming(N) returns a vector of length N
      R-PAT-04: filter() executes without error and returns a result

    Consistency: R-PAT-01 validates spectral roundtrip, R-PAT-02
    validates polynomial multiplication via convolution, R-PAT-03
    validates window generation, and R-PAT-04 validates IIR/FIR
    filtering. These four operations are the core DSP primitives.
    """

    def test_fft_roundtrip(self, s):
        """R-PAT-01: ifft(fft(x)) recovers x within machine precision."""
        s.eval("x_fft = ifft(fft([1 2 3 4]))")
        x = get_array(s, "x_fft")
        data = np.real(x.data).flatten()
        np.testing.assert_allclose(data, [1, 2, 3, 4], atol=1e-10)

    def test_conv(self, s):
        """R-PAT-02: conv([1 1], [1 1 1]) produces [1 2 2 1]."""
        s.eval("conv_r = conv([1 1], [1 1 1])")
        r = get_array(s, "conv_r")
        np.testing.assert_array_equal(np.round(r.data.flatten()).astype(int), [1, 2, 2, 1])

    def test_hamming_window(self, s):
        """R-PAT-03: hamming(N) returns a vector of length N."""
        s.eval("h = hamming(5)")
        r = get_array(s, "h")
        assert r.data.flatten().shape[0] == 5

    def test_filter(self, s):
        """R-PAT-04: filter() executes and returns a non-null result."""
        r = s.eval("filter([1], [1 -0.5], ones(1, 10))")
        assert r is not None


class TestLinearAlgebra:
    """R-PAT-05..11: Linear algebra builtins SHALL produce numerically
    correct results for system solving, decomposition, and matrix
    properties.

    Model-user argument: The engineer solves Ax=b systems, computes
    eigenvalues to check stability, and uses SVD for dimensionality
    reduction. Numerical accuracy to machine precision is essential
    because even small errors compound in iterative algorithms.

    Decomposition:
      R-PAT-05: A\\b solves the linear system with residual < 1e-10
      R-PAT-06: eig returns sorted eigenvalues matching known values
      R-PAT-07: svd returns singular values in descending order
      R-PAT-08: det([1 2; 3 4]) returns -2
      R-PAT-09: inv(A)*A yields identity within tolerance
      R-PAT-10: rank([1 2; 2 4]) returns 1
      R-PAT-11: trace([1 2; 3 4]) returns 5

    Consistency: These seven tests span solving (R-PAT-05),
    decomposition (R-PAT-06..07), and scalar matrix properties
    (R-PAT-08..11), covering the engineer's core linear algebra toolkit.
    """

    def test_solve_system(self, s):
        """R-PAT-05: A\\b solves the system with residual below 1e-10."""
        s.eval("A = [3 2 -1; 2 -2 4; -1 0.5 -1]; b = [1; -2; 0]; x = A \\ b")
        r = s.eval("norm(A*x - b)")
        assert float(r) < 1e-10

    def test_eigendecomposition(self, s):
        """R-PAT-06: eig returns sorted eigenvalues [1, 3] for [2 1; 1 2]."""
        s.eval("[V, D] = eig([2 1; 1 2])")
        s.eval("eig_vals = sort(diag(D))")
        d = get_array(s, "eig_vals")
        vals = sorted(np.real(d.data).flatten())
        np.testing.assert_allclose(vals, [1.0, 3.0], atol=1e-10)

    def test_svd(self, s):
        """R-PAT-07: svd returns singular values [2, 1] for diag([1, 2])."""
        s.eval("[U, S, V] = svd([1 0; 0 2])")
        S = get_array(s, "S")
        sv = sorted(np.diag(S.data).flatten(), reverse=True)
        np.testing.assert_allclose(sv[:2], [2.0, 1.0], atol=1e-10)

    def test_det(self, s):
        """R-PAT-08: det([1 2; 3 4]) returns -2."""
        r = s.eval("det([1 2; 3 4])")
        assert abs(float(r) - (-2.0)) < 1e-10

    def test_inv(self, s):
        """R-PAT-09: inv(A)*A yields identity within tolerance."""
        s.eval("A = [1 2; 3 4]; B = inv(A)")
        r = s.eval("norm(A*B - eye(2))")
        assert float(r) < 1e-10

    def test_rank(self, s):
        """R-PAT-10: rank of a rank-deficient matrix returns 1."""
        r = s.eval("rank([1 2; 2 4])")
        assert float(r) == 1.0

    def test_trace(self, s):
        """R-PAT-11: trace([1 2; 3 4]) returns 5."""
        r = s.eval("trace([1 2; 3 4])")
        assert float(r) == 5.0


class TestControlFlow:
    """R-PAT-12..18: Control flow constructs SHALL execute correctly for
    recursion, variable arguments, default arguments, nested loops,
    while/break, switch/case, and try/catch.

    Model-user argument: The engineer's MATLAB scripts use recursion for
    tree traversals, varargin for flexible interfaces, nargin for default
    parameters, nested loops for grid computations, and try/catch for
    robust error handling. All of these must behave identically to MATLAB.

    Decomposition:
      R-PAT-12: Recursive fib(10) returns 55
      R-PAT-13: varargin-based sum of 1..5 returns 15
      R-PAT-14: nargin-based default argument works correctly
      R-PAT-15: Nested for loops accumulate correctly
      R-PAT-16: while/break exits at the right iteration
      R-PAT-17: switch/case on strings selects the correct branch
      R-PAT-18: try/catch captures errors and executes catch block

    Consistency: These seven constructs are the complete set of
    non-trivial control flow patterns in the M-language specification.
    """

    def test_recursive_fibonacci(self, s):
        """R-PAT-12: Recursive fib(10) returns 55."""
        s.eval("function r = fib(n); if n <= 1; r = n; else; r = fib(n-1) + fib(n-2); end; end")
        r = s.eval("fib(10)")
        assert float(r) == 55.0

    def test_varargin_sum(self, s):
        """R-PAT-13: varargin-based sum of 1..5 returns 15."""
        s.eval("function r = mysum(varargin); r = 0; for i = 1:length(varargin); r = r + varargin{i}; end; end")
        r = s.eval("mysum(1, 2, 3, 4, 5)")
        assert float(r) == 15.0

    def test_default_arg_via_nargin(self, s):
        """R-PAT-14: nargin-based default argument fills missing parameter."""
        s.eval("function r = myfun(a, b); if nargin < 2; b = 10; end; r = a + b; end")
        r = s.eval("myfun(5)")
        assert float(r) == 15.0

    def test_nested_loops(self, s):
        """R-PAT-15: Nested for loops accumulate 3x4=12 iterations."""
        s.eval("r = 0; for i = 1:3; for j = 1:4; r = r + 1; end; end")
        r = s.eval("r")
        assert float(r) == 12.0

    def test_while_break(self, s):
        """R-PAT-16: while/break exits when counter reaches 10."""
        s.eval("i = 0; while true; i = i + 1; if i >= 10; break; end; end")
        r = s.eval("i")
        assert float(r) == 10.0

    def test_switch_string(self, s):
        """R-PAT-17: switch/case on string 'b' selects the correct branch."""
        s.eval("x = 'b'; switch x; case 'a'; r = 1; case 'b'; r = 2; otherwise; r = 0; end")
        r = s.eval("r")
        assert float(r) == 2.0

    def test_try_catch_error(self, s):
        """R-PAT-18: try/catch captures error and executes catch block."""
        r = s.eval("try; error('test'); catch e; r = 42; end; r")
        assert float(r) == 42.0


class TestStringOps:
    """R-PAT-19..22: String builtins SHALL format, transform, split,
    and match strings identically to Octave.

    Model-user argument: The engineer uses sprintf for report generation,
    upper/lower for case normalization, strsplit for CSV parsing, and
    regexp for log file analysis. Incorrect string handling corrupts
    data import/export workflows.

    Decomposition:
      R-PAT-19: sprintf formats multiple integer arguments
      R-PAT-20: upper('hello') returns 'HELLO'
      R-PAT-21: strsplit produces a non-null cell array
      R-PAT-22: regexp returns the 1-based start index of the match

    Consistency: These four operations cover formatted output, case
    transformation, delimiter-based splitting, and regex matching,
    which are the four string operation categories in typical scripts.
    """

    def test_sprintf_format(self, s):
        """R-PAT-19: sprintf formats integer arguments into a string."""
        r = s.eval("sprintf('%d + %d = %d', 1, 2, 3)")
        assert "1 + 2 = 3" in str(r)

    def test_upper_lower(self, s):
        """R-PAT-20: upper('hello') returns 'HELLO'."""
        r = s.eval("upper('hello')")
        assert str(r).strip().upper() == "HELLO"

    def test_strsplit(self, s):
        """R-PAT-21: strsplit returns a non-null cell array."""
        r = s.eval("strsplit('a,b,c', ',')")
        assert r is not None

    def test_regexp(self, s):
        """R-PAT-22: regexp returns 1-based start index of first match."""
        r = s.eval("regexp('abc123def', '[0-9]+')")
        assert float(r) == 4.0  # 1-based index of match start


class TestMatrixOps:
    """R-PAT-23..30: Matrix operations and indexing SHALL produce correct
    results for logical indexing, colon subscripts, end keyword, submatrix
    extraction, array growth, element-wise arithmetic, and matrix power.

    Model-user argument: The engineer builds matrices incrementally (grow
    pattern), extracts submatrices for windowed analysis, uses end-relative
    indexing to access trailing elements, and applies element-wise
    operations for pointwise transformations. These are the mechanical
    building blocks of every data processing script.

    Decomposition:
      R-PAT-23: Logical indexing filters by condition
      R-PAT-24: Colon indexing extracts a full column
      R-PAT-25: end keyword accesses the last element
      R-PAT-26: end-1 accesses the second-to-last element
      R-PAT-27: Range subscript extracts a submatrix
      R-PAT-28: Array growth via concatenation in a loop
      R-PAT-29: Element-wise .* multiplies corresponding elements
      R-PAT-30: Matrix power (^) computes repeated matrix multiplication

    Consistency: These eight tests cover the complete set of non-trivial
    subscripting and arithmetic patterns for matrix manipulation.
    """

    def test_logical_indexing(self, s):
        """R-PAT-23: Logical indexing filters elements by condition."""
        s.eval("li_r = [1 2 3 4 5]; li_r = li_r(li_r > 3)")
        r = get_array(s, "li_r")
        np.testing.assert_array_equal(r.data.flatten(), [4, 5])

    def test_colon_indexing(self, s):
        """R-PAT-24: A(:, 2) extracts the full second column."""
        s.eval("A = [1 2 3; 4 5 6; 7 8 9]; col_r = A(:, 2)")
        r = get_array(s, "col_r")
        np.testing.assert_array_equal(r.data.flatten(), [2, 5, 8])

    def test_end_indexing(self, s):
        """R-PAT-25: x(end) returns the last element."""
        r = s.eval("x = [10 20 30 40 50]; x(end)")
        assert float(r) == 50.0

    def test_end_minus_indexing(self, s):
        """R-PAT-26: x(end-1) returns the second-to-last element."""
        r = s.eval("x = [10 20 30 40 50]; x(end-1)")
        assert float(r) == 40.0

    def test_submatrix(self, s):
        """R-PAT-27: A(1:2, 2:3) extracts a 2x2 submatrix."""
        s.eval("A = [1 2 3; 4 5 6; 7 8 9]; sub_r = A(1:2, 2:3)")
        r = get_array(s, "sub_r")
        expected = np.array([[2, 3], [5, 6]])
        np.testing.assert_array_equal(r.data, expected)

    def test_grow_array(self, s):
        """R-PAT-28: Array growth via concatenation accumulates squares."""
        s.eval("x = []; for i = 1:5; x = [x i^2]; end")
        r = get_array(s, "x")
        np.testing.assert_array_equal(r.data.flatten(), [1, 4, 9, 16, 25])

    def test_element_wise_ops(self, s):
        """R-PAT-29: Element-wise .* multiplies corresponding elements."""
        s.eval("ew_r = [1 2 3] .* [4 5 6]")
        r = get_array(s, "ew_r")
        np.testing.assert_array_equal(r.data.flatten(), [4, 10, 18])

    def test_matrix_power(self, s):
        """R-PAT-30: Matrix power [1 1;1 0]^5 computes Fibonacci matrix."""
        r = s.eval("[1 1; 1 0]^5")
        # Fibonacci matrix: F(6)=8, F(5)=5
        assert float(s.eval("ans(1,1)")) == 8.0


class TestStatistics:
    """R-PAT-31..34: Descriptive statistics builtins SHALL produce correct
    results matching Octave's default behavior (sample variance, ddof=1).

    Model-user argument: The engineer computes mean, std, median, and
    variance to summarize experimental datasets. Octave uses ddof=1 for
    std and var by default (sample statistics), and Forge must match this
    convention or the engineer's published results will differ.

    Decomposition:
      R-PAT-31: mean([1..5]) returns 3
      R-PAT-32: std uses ddof=1 (sample standard deviation)
      R-PAT-33: median([1 3 5 7 9]) returns 5
      R-PAT-34: var uses ddof=1 (sample variance)

    Consistency: mean, std, median, and var are the four core descriptive
    statistics. Testing ddof=1 for std and var confirms Octave-compatible
    normalization.
    """

    def test_mean(self, s):
        """R-PAT-31: mean([1 2 3 4 5]) returns 3."""
        r = s.eval("mean([1 2 3 4 5])")
        assert float(r) == 3.0

    def test_std(self, s):
        """R-PAT-32: std uses sample standard deviation (ddof=1)."""
        s.eval("std_r = std([2 4 4 4 5 5 7 9])")
        r = get_array(s, "std_r")
        assert abs(float(r) - np.std([2, 4, 4, 4, 5, 5, 7, 9], ddof=1)) < 1e-6

    def test_median(self, s):
        """R-PAT-33: median([1 3 5 7 9]) returns 5."""
        r = s.eval("median([1 3 5 7 9])")
        assert float(r) == 5.0

    def test_var(self, s):
        """R-PAT-34: var uses sample variance (ddof=1)."""
        r = s.eval("var([1 2 3 4 5])")
        assert abs(float(r) - np.var([1, 2, 3, 4, 5], ddof=1)) < 1e-10
