"""Tests for complex multi-function scripts and realistic workflows."""
import pytest
import numpy as np
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


class TestMultiFunctionScripts:
    """Test scripts with multiple function definitions."""

    def test_stats_function(self, s):
        code = "function [mu, sigma] = stats(x)\n  mu = mean(x);\n  sigma = std(x);\nend\n[m, sig] = stats([1 2 3 4 5 6 7 8 9 10]);"
        s.eval(code)
        assert abs(float(s.eval("m")) - 5.5) < 0.01

    def test_function_calling_function(self, s):
        code = "function r = square(x)\n  r = x .^ 2;\nend\nfunction r = sum_of_squares(v)\n  r = sum(square(v));\nend\nresult = sum_of_squares([1 2 3 4]);"
        s.eval(code)
        assert float(s.eval("result")) == 30.0

    def test_varargin_accumulator(self, s):
        code = "function r = flex_add(varargin)\n  r = 0;\n  for k = 1:nargin\n    r = r + varargin{k};\n  end\nend\ntotal = flex_add(1, 2, 3, 4, 5);"
        s.eval(code)
        assert float(s.eval("total")) == 15.0

    def test_positive_definite_eigenvalues(self, s):
        code = "n = 5;\nA = rand(n);\nA = A + A';\nA = A + n * eye(n);\n[V, D] = eig(A);\neigenvalues = diag(D);\nall_pos = all(eigenvalues > 0);"
        s.eval(code)
        assert float(s.eval("all_pos")) == 1.0


class TestMultiOutput:
    """Test multi-output function calls."""

    def test_svd_3out(self, s):
        s.eval("[U, S, V] = svd([1 2; 3 4])")
        U = s._engine.workspace.get("U")
        assert U.data.shape == (2, 2)

    def test_lu_3out(self, s):
        s.eval("[L, U, P] = lu([1 2; 3 4])")
        L = s._engine.workspace.get("L")
        assert L.data.shape == (2, 2)

    def test_qr_2out(self, s):
        s.eval("[Q, R] = qr([1 2; 3 4])")
        Q = s._engine.workspace.get("Q")
        assert Q.data.shape == (2, 2)

    def test_unique_2out(self, s):
        s.eval("[vals, idx] = unique([3 1 2 1 3])")
        vals = s._engine.workspace.get("vals")
        np.testing.assert_array_equal(vals.data.flatten(), [1, 2, 3])

    def test_sort_2out(self, s):
        s.eval("[sorted_v, idx] = sort([3 1 4 1 5])")
        sv = s._engine.workspace.get("sorted_v")
        np.testing.assert_array_equal(sv.data.flatten(), [1, 1, 3, 4, 5])

    def test_size_2out(self, s):
        s.eval("[m, n] = size([1 2 3; 4 5 6])")
        assert float(s.eval("m")) == 2.0
        assert float(s.eval("n")) == 3.0

    def test_meshgrid_2out(self, s):
        s.eval("[X, Y] = meshgrid(1:3, 1:2)")
        X = s._engine.workspace.get("X")
        assert X.data.shape == (2, 3)


class TestParfor:
    """Test parfor loop execution."""

    def test_parfor_basic(self, s):
        s.eval("r = 0; parfor i = 1:5; r = r + i; end")
        assert float(s.eval("r")) == 15.0

    def test_parfor_array(self, s):
        s.eval("x = zeros(1, 10); parfor i = 1:10; x(i) = i^2; end")
        x = s._engine.workspace.get("x")
        np.testing.assert_array_equal(x.data.flatten(), [i**2 for i in range(1, 11)])
