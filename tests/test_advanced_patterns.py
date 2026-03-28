"""Tests for advanced Octave patterns - realistic user scenarios."""
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
    """Signal processing workflow tests."""

    def test_fft_roundtrip(self, s):
        s.eval("x_fft = ifft(fft([1 2 3 4]))")
        x = get_array(s, "x_fft")
        data = np.real(x.data).flatten()
        np.testing.assert_allclose(data, [1, 2, 3, 4], atol=1e-10)

    def test_conv(self, s):
        s.eval("conv_r = conv([1 1], [1 1 1])")
        r = get_array(s, "conv_r")
        np.testing.assert_array_equal(np.round(r.data.flatten()).astype(int), [1, 2, 2, 1])

    def test_hamming_window(self, s):
        s.eval("h = hamming(5)")
        r = get_array(s, "h")
        assert r.data.flatten().shape[0] == 5

    def test_filter(self, s):
        r = s.eval("filter([1], [1 -0.5], ones(1, 10))")
        assert r is not None


class TestLinearAlgebra:
    """Linear algebra workflow tests."""

    def test_solve_system(self, s):
        s.eval("A = [3 2 -1; 2 -2 4; -1 0.5 -1]; b = [1; -2; 0]; x = A \ b")
        r = s.eval("norm(A*x - b)")
        assert float(r) < 1e-10

    def test_eigendecomposition(self, s):
        s.eval("[V, D] = eig([2 1; 1 2])")
        s.eval("eig_vals = sort(diag(D))")
        d = get_array(s, "eig_vals")
        vals = sorted(np.real(d.data).flatten())
        np.testing.assert_allclose(vals, [1.0, 3.0], atol=1e-10)

    def test_svd(self, s):
        s.eval("[U, S, V] = svd([1 0; 0 2])")
        S = get_array(s, "S")
        sv = sorted(np.diag(S.data).flatten(), reverse=True)
        np.testing.assert_allclose(sv[:2], [2.0, 1.0], atol=1e-10)

    def test_det(self, s):
        r = s.eval("det([1 2; 3 4])")
        assert abs(float(r) - (-2.0)) < 1e-10

    def test_inv(self, s):
        s.eval("A = [1 2; 3 4]; B = inv(A)")
        r = s.eval("norm(A*B - eye(2))")
        assert float(r) < 1e-10

    def test_rank(self, s):
        r = s.eval("rank([1 2; 2 4])")
        assert float(r) == 1.0

    def test_trace(self, s):
        r = s.eval("trace([1 2; 3 4])")
        assert float(r) == 5.0


class TestControlFlow:
    """Control flow and function tests."""

    def test_recursive_fibonacci(self, s):
        s.eval("function r = fib(n); if n <= 1; r = n; else; r = fib(n-1) + fib(n-2); end; end")
        r = s.eval("fib(10)")
        assert float(r) == 55.0

    def test_varargin_sum(self, s):
        s.eval("function r = mysum(varargin); r = 0; for i = 1:length(varargin); r = r + varargin{i}; end; end")
        r = s.eval("mysum(1, 2, 3, 4, 5)")
        assert float(r) == 15.0

    def test_default_arg_via_nargin(self, s):
        s.eval("function r = myfun(a, b); if nargin < 2; b = 10; end; r = a + b; end")
        r = s.eval("myfun(5)")
        assert float(r) == 15.0

    def test_nested_loops(self, s):
        s.eval("r = 0; for i = 1:3; for j = 1:4; r = r + 1; end; end")
        r = s.eval("r")
        assert float(r) == 12.0

    def test_while_break(self, s):
        s.eval("i = 0; while true; i = i + 1; if i >= 10; break; end; end")
        r = s.eval("i")
        assert float(r) == 10.0

    def test_switch_string(self, s):
        s.eval("x = 'b'; switch x; case 'a'; r = 1; case 'b'; r = 2; otherwise; r = 0; end")
        r = s.eval("r")
        assert float(r) == 2.0

    def test_try_catch_error(self, s):
        r = s.eval("try; error('test'); catch e; r = 42; end; r")
        assert float(r) == 42.0


class TestStringOps:
    """String operation tests."""

    def test_sprintf_format(self, s):
        r = s.eval("sprintf('%d + %d = %d', 1, 2, 3)")
        assert "1 + 2 = 3" in str(r)

    def test_upper_lower(self, s):
        r = s.eval("upper('hello')")
        assert str(r).strip().upper() == "HELLO"

    def test_strsplit(self, s):
        r = s.eval("strsplit('a,b,c', ',')")
        assert r is not None

    def test_regexp(self, s):
        r = s.eval("regexp('abc123def', '[0-9]+')")
        assert float(r) == 4.0  # 1-based index of match start


class TestMatrixOps:
    """Matrix operation and indexing tests."""

    def test_logical_indexing(self, s):
        s.eval("li_r = [1 2 3 4 5]; li_r = li_r(li_r > 3)")
        r = get_array(s, "li_r")
        np.testing.assert_array_equal(r.data.flatten(), [4, 5])

    def test_colon_indexing(self, s):
        s.eval("A = [1 2 3; 4 5 6; 7 8 9]; col_r = A(:, 2)")
        r = get_array(s, "col_r")
        np.testing.assert_array_equal(r.data.flatten(), [2, 5, 8])

    def test_end_indexing(self, s):
        r = s.eval("x = [10 20 30 40 50]; x(end)")
        assert float(r) == 50.0

    def test_end_minus_indexing(self, s):
        r = s.eval("x = [10 20 30 40 50]; x(end-1)")
        assert float(r) == 40.0

    def test_submatrix(self, s):
        s.eval("A = [1 2 3; 4 5 6; 7 8 9]; sub_r = A(1:2, 2:3)")
        r = get_array(s, "sub_r")
        expected = np.array([[2, 3], [5, 6]])
        np.testing.assert_array_equal(r.data, expected)

    def test_grow_array(self, s):
        s.eval("x = []; for i = 1:5; x = [x i^2]; end")
        r = get_array(s, "x")
        np.testing.assert_array_equal(r.data.flatten(), [1, 4, 9, 16, 25])

    def test_element_wise_ops(self, s):
        s.eval("ew_r = [1 2 3] .* [4 5 6]")
        r = get_array(s, "ew_r")
        np.testing.assert_array_equal(r.data.flatten(), [4, 10, 18])

    def test_matrix_power(self, s):
        r = s.eval("[1 1; 1 0]^5")
        # Fibonacci matrix: F(6)=8, F(5)=5
        assert float(s.eval("ans(1,1)")) == 8.0


class TestStatistics:
    """Statistics function tests."""

    def test_mean(self, s):
        r = s.eval("mean([1 2 3 4 5])")
        assert float(r) == 3.0

    def test_std(self, s):
        s.eval("std_r = std([2 4 4 4 5 5 7 9])")
        r = get_array(s, "std_r")
        assert abs(float(r) - np.std([2, 4, 4, 4, 5, 5, 7, 9], ddof=1)) < 1e-6

    def test_median(self, s):
        r = s.eval("median([1 3 5 7 9])")
        assert float(r) == 5.0

    def test_var(self, s):
        r = s.eval("var([1 2 3 4 5])")
        assert abs(float(r) - np.var([1, 2, 3, 4, 5], ddof=1)) < 1e-10
