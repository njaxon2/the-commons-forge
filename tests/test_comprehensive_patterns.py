"""Comprehensive tests for Octave-compatible patterns verified manually."""
import pytest
import sys
import numpy as np
sys.path.insert(0, ".")
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray


def get_val(s, name):
    """Get numeric value from workspace."""
    v = s.workspace.get(name)
    if isinstance(v, ForgeArray):
        return float(v.data.flat[0])
    return float(v)


def get_array(s, name):
    """Get array data from workspace."""
    v = s.workspace.get(name)
    if isinstance(v, ForgeArray):
        return v.data
    return np.asarray(v)


@pytest.fixture
def s():
    return ForgeSession()


class TestSprintfVectorized:
    def test_vector_int(self, s):
        r = s.eval('sprintf("%d ", [1 2 3 4 5])')
        assert "1 2 3 4 5" in str(r)

    def test_vector_float(self, s):
        r = s.eval('sprintf("%.1f ", [1.5 2.5 3.5])')
        assert "1.5" in str(r) and "3.5" in str(r)

    def test_paired_vectors(self, s):
        r = s.eval('sprintf("(%d,%d) ", [1 2 3], [4 5 6])')
        assert "(1,4)" in str(r) and "(3,6)" in str(r)

    def test_string_arg(self, s):
        r = s.eval('sprintf("Hello %s!", "world")')
        assert "Hello world!" in str(r)


class TestFprintfVectorized:
    def test_vector_to_stdout(self, s):
        r = s.eval('fprintf(1, "%d ", [10 20 30])')
        assert "10" in str(r) and "30" in str(r)


class TestNestedStructs:
    def test_two_level(self, s):
        r = s.eval("st.inner.x = 42; st.inner.x")
        assert "42" in str(r)

    def test_three_level(self, s):
        r = s.eval("a.b.c.d = 99; a.b.c.d")
        assert "99" in str(r)

    def test_multiple_fields(self, s):
        script = (
            'config.db.host = "localhost";\n'
            'config.db.port = 5432;\n'
            'config.app.name = "forge";\n'
        )
        s.eval(script)
        r = s.eval("config.db.port")
        assert "5432" in str(r)


class TestFunctionPatterns:
    def test_multiple_functions(self, s):
        script = (
            "function y = double_it(x)\n"
            "  y = 2 * x;\n"
            "end\n"
            "function y = triple_it(x)\n"
            "  y = 3 * x;\n"
            "end\n"
            "result = double_it(5) + triple_it(5);\n"
        )
        s.eval(script)
        assert get_val(s, "result") == 25.0

    def test_function_calling_function(self, s):
        script = (
            "function y = sq(x)\n"
            "  y = x^2;\n"
            "end\n"
            "function y = sum_sq(a, b)\n"
            "  y = sq(a) + sq(b);\n"
            "end\n"
            "result = sum_sq(3, 4);\n"
        )
        s.eval(script)
        assert get_val(s, "result") == 25.0

    def test_three_output_function(self, s):
        script = (
            "function [mn, mx, rng] = stats3(x)\n"
            "  mn = min(x);\n"
            "  mx = max(x);\n"
            "  rng = mx - mn;\n"
            "end\n"
            "[a, b, c] = stats3([3 1 4 1 5 9 2 6]);\n"
        )
        s.eval(script)
        assert get_val(s, "a") == 1.0
        assert get_val(s, "b") == 9.0
        assert get_val(s, "c") == 8.0

    def test_default_arg_via_nargin(self, s):
        script = (
            "function y = myfun(x, scale)\n"
            "  if nargin < 2\n"
            "    scale = 1;\n"
            "  end\n"
            "  y = x * scale;\n"
            "end\n"
            "r1 = myfun(5);\n"
            "r2 = myfun(5, 3);\n"
        )
        s.eval(script)
        assert get_val(s, "r1") == 5.0
        assert get_val(s, "r2") == 15.0


class TestMatrixOperations:
    def test_broadcast(self, s):
        s.eval("A = [1 2 3]; B = [10; 20; 30]; C = A + B")
        data = get_array(s, "C")
        assert data.shape == (3, 3)

    def test_logical_assignment(self, s):
        s.eval("A = [1 2 3 4 5]; A(A > 3) = 0")
        data = get_array(s, "A")
        assert list(data.flat) == [1.0, 2.0, 3.0, 0.0, 0.0]

    def test_grow_array(self, s):
        s.eval("A = []; for i = 1:5; A = [A, i^2]; end")
        data = get_array(s, "A")
        assert list(data.flat) == [1.0, 4.0, 9.0, 16.0, 25.0]

    def test_submatrix(self, s):
        s.eval("A = [1 2 3; 4 5 6; 7 8 9]; B = A(2:3, 1:2)")
        data = get_array(s, "B")
        assert data.shape == (2, 2)
        assert float(data[0, 0]) == 4.0

    def test_row_assignment(self, s):
        s.eval("A = zeros(3); A(2,:) = [4 5 6]")
        data = get_array(s, "A")
        assert float(data[1, 0]) == 4.0
        assert float(data[1, 2]) == 6.0


class TestCurveFitting:
    def test_polyfit_polyval(self, s):
        script = (
            "t = linspace(0, 10, 50);\n"
            "y_true = 0.5*t.^2 - 3*t + 7;\n"
            "p = polyfit(t, y_true, 2);\n"
            "y_fit = polyval(p, t);\n"
            "err = max(abs(y_true - y_fit));\n"
        )
        s.eval(script)
        assert get_val(s, "err") < 1e-10

    def test_linear_system(self, s):
        script = "A = [3 1; 1 2]; b = [9; 8]; x = A \\ b; residual = norm(A * x - b);"
        s.eval(script)
        assert get_val(s, "residual") < 1e-10


class TestEigenAnalysis:
    def test_symmetric_eigenvalues(self, s):
        script = (
            "A = [2 1; 1 2];\n"
            "[V, D] = eig(A);\n"
            'reconstruction_error = norm(A*V - V*D, "fro");\n'
        )
        s.eval(script)
        assert get_val(s, "reconstruction_error") < 1e-12

    def test_trace_equals_eigensum(self, s):
        script = (
            "A = [4 1 0; 1 3 1; 0 1 2];\n"
            "[V, D] = eig(A);\n"
            "eigenvalues = diag(D);\n"
            "trace_err = abs(trace(A) - sum(eigenvalues));\n"
        )
        s.eval(script)
        assert get_val(s, "trace_err") < 1e-10


class TestStringProcessing:
    def test_strsplit_join(self, s):
        r = s.eval('strjoin(strsplit("hello world", " "), "-")')
        assert "hello-world" in str(r)

    def test_upper_lower(self, s):
        r1 = s.eval('upper("hello")')
        assert "HELLO" in str(r1)
        r2 = s.eval('lower("HELLO")')
        assert "hello" in str(r2)

    def test_strrep(self, s):
        r = s.eval('strrep("hello world", "world", "there")')
        assert "hello there" in str(r)


class TestCellArrays:
    def test_cell_of_handles(self, s):
        script = (
            "ops = {@(a,b) a+b, @(a,b) a-b, @(a,b) a.*b};\n"
            "result = ops{2}(10, 3);\n"
        )
        s.eval(script)
        assert get_val(s, "result") == 7.0

    def test_map_reduce(self, s):
        script = (
            "data = {[1 2 3], [4 5], [6 7 8 9]};\n"
            "lengths = cellfun(@length, data);\n"
            "total = sum(lengths);\n"
        )
        s.eval(script)
        assert get_val(s, "total") == 9.0


class TestControlFlow:
    def test_while_break(self, s):
        script = (
            "n = 0;\n"
            "while true\n"
            "  n = n + 1;\n"
            "  if n >= 10\n"
            "    break;\n"
            "  end\n"
            "end\n"
        )
        s.eval(script)
        assert get_val(s, "n") == 10.0

    def test_do_until(self, s):
        s.eval("x = 0; do; x = x + 1; until (x >= 5)")
        assert get_val(s, "x") == 5.0


class TestFileIO:
    def test_csv_roundtrip(self, s):
        script = (
            "data = [1 2 3; 4 5 6];\n"
            'csvwrite("/tmp/forge_ctest.csv", data);\n'
            'result = csvread("/tmp/forge_ctest.csv");\n'
        )
        s.eval(script)
        data = get_array(s, "result")
        assert data.shape[0] == 2 and data.shape[1] == 3

    def test_isdir(self, s):
        r = s.eval('isdir("/tmp")')
        assert "1" in str(r)


class TestDynamicField:
    def test_dynamic_field_read(self, s):
        script = (
            'st = struct("x", 1, "y", 2, "z", 3);\n'
            'fn = "y";\n'
            "val = st.(fn);\n"
        )
        s.eval(script)
        assert get_val(s, "val") == 2.0

    def test_struct_iteration(self, s):
        script = (
            'st = struct("a", 10, "b", 20, "c", 30);\n'
            "fields = fieldnames(st);\n"
            "total = 0;\n"
            "for i = 1:length(fields)\n"
            "  total = total + st.(fields{i});\n"
            "end\n"
        )
        s.eval(script)
        assert get_val(s, "total") == 60.0
