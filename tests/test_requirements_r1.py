"""Tests for Forge Requirements R01-R16."""
import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def s():
    return ForgeSession()


# ============================================================
# R01 - Semicolon Output Suppression
# ============================================================

class TestR01_SemicolonSuppression:
    def test_semicolon_suppresses(self, s):
        r = s.eval("a = 5;")
        assert r == ""

    def test_no_semicolon_displays(self, s):
        r = s.eval("a = 5")
        assert r.strip() != ""
        assert "5" in r

    def test_semicolon_matrix(self, s):
        r = s.eval("A = [1 2; 3 4];")
        assert r == ""

    def test_no_semicolon_matrix(self, s):
        r = s.eval("A = [1 2; 3 4]")
        assert "1" in r and "4" in r


# ============================================================
# R02 - MATLAB-Style Output Formatting
# ============================================================

class TestR02_OutputFormat:
    def test_scalar_integer_display(self, s):
        r = s.eval("x = 5")
        # Should not show numpy brackets
        assert "[[" not in r

    def test_matrix_no_numpy_brackets(self, s):
        r = s.eval("A = [1 2; 3 4]")
        assert "[[" not in r

    def test_float_format_short(self, s):
        r = s.eval("x = 3.14159265")
        assert "3.1416" in r

    def test_column_alignment(self, s):
        r = s.eval("A = [1 20 300; 4 50 600]")
        lines = [l for l in r.strip().split("\n") if l.strip() and any(c.isdigit() for c in l)]
        assert len(lines) >= 2


# ============================================================
# R03 - Character Array Display
# ============================================================

class TestR03_CharDisplay:
    def test_char_displays_as_text(self, s):
        r = s.eval("s = 'hello'")
        assert "hello" in r
        assert "104" not in r

    def test_class_of_char(self, s):
        s.eval("s = 'hello';")
        r = s.eval("class(s)")
        assert "char" in r


# ============================================================
# R04 - Slash in Single-Quoted Strings
# ============================================================

class TestR04_SlashInStrings:
    def test_single_quote_with_slash(self, s):
        # Should not raise ParseError about RDIVIDE
        try:
            s.eval("p = '/tmp/test'")
        except Exception as e:
            if "ParseError" in str(type(e).__name__) or "RDIVIDE" in str(e):
                pytest.fail(f"Slash in string caused parse error: {e}")

    def test_addpath_with_slash(self, s):
        try:
            s.eval("addpath('/tmp')")
        except Exception as e:
            if "ParseError" in str(type(e).__name__) or "RDIVIDE" in str(e):
                pytest.fail(f"addpath with slash failed: {e}")


# ============================================================
# R05 - Command-Style Syntax
# ============================================================

class TestR05_CommandStyle:
    def test_who_command(self, s):
        s.eval("x = 5;")
        r = s.eval("who")
        assert "function" not in r.lower()

    def test_whos_command(self, s):
        s.eval("x = 5;")
        r = s.eval("whos")
        assert "function" not in r.lower()

    def test_hold_on_command(self, s):
        try:
            s.eval("hold on")
        except NameError:
            pytest.fail("hold on raised NameError")

    def test_axis_equal_command(self, s):
        try:
            s.eval("axis equal")
        except NameError:
            pytest.fail("axis equal raised NameError")


# ============================================================
# R06 - Float-to-Int Coercion
# ============================================================

class TestR06_FloatToInt:
    def test_reshape_with_floats(self, s):
        r = s.eval("reshape(1:6, 2, 3)")
        assert "1" in r and "6" in r

    def test_zeros_with_float_args(self, s):
        s.eval("n = 3;")
        r = s.eval("zeros(n, n)")
        assert "0" in r

    def test_ones_with_expression(self, s):
        r = s.eval("ones(2+1, 2)")
        assert "1" in r


# ============================================================
# R07 - Struct Auto-Creation
# ============================================================

class TestR07_StructAutoCreate:
    def test_create_struct_by_field(self, s):
        s.eval("msh.nodes = [0 0; 1 0];")
        r = s.eval("msh.nodes")
        assert "0" in r and "1" in r

    def test_multiple_fields(self, s):
        s.eval("msh.x = 5;")
        s.eval("msh.y = 10;")
        r = s.eval("msh.x")
        assert "5" in r
        r = s.eval("msh.y")
        assert "10" in r

    def test_no_collision_with_builtins(self, s):
        s.eval("mesh_data.nodes = [1 2 3];")
        r = s.eval("mesh_data.nodes")
        assert "1" in r


# ============================================================
# R08 - Sparse Matrix Construction
# ============================================================

class TestR08_Sparse:
    def test_sparse_zeros(self, s):
        s.eval("S = sparse(3, 3);")
        r = s.eval("nnz(S)")
        assert "0" in r

    def test_sparse_triplet(self, s):
        s.eval("I = [1 1 2 3]; J = [1 2 2 3]; V = [4 1 5 6];")
        s.eval("K = sparse(I, J, V, 3, 3);")
        r = s.eval("full(K)")
        assert "4" in r and "5" in r


# ============================================================
# R09 - Indexed Assignment with RHS Expressions
# ============================================================

class TestR09_IndexedAssignment:
    def test_indexed_assign_with_addition(self, s):
        s.eval("K = zeros(4, 4);")
        s.eval("ke = [2 -1; -1 2];")
        s.eval("K(1:2, 1:2) = K(1:2, 1:2) + ke;")
        r = s.eval("K(1,1)")
        assert "2" in r

    def test_indexed_assign_scalar(self, s):
        s.eval("A = zeros(3,3);")
        s.eval("A(2,2) = 99;")
        r = s.eval("A(2,2)")
        assert "99" in r


# ============================================================
# R10 - eig() Return Order
# ============================================================

class TestR10_EigOrder:
    def test_eig_order(self, s):
        s.eval("A = [2 1; 0 3];")
        s.eval("[V, D] = eig(A);")
        # D should be diagonal eigenvalue matrix with 2 and 3
        r_d = s.eval("D")
        assert "2" in r_d and "3" in r_d
        # D should be diagonal => D(1,2) == 0
        r_offdiag = s.eval("D(1,2)")
        assert "0" in r_offdiag


# ============================================================
# R11 - Nested Function Calls
# ============================================================

class TestR11_NestedCalls:
    def test_max_abs(self, s):
        r = s.eval("max(abs([-3 5 -7 2]))")
        assert "7" in r

    def test_sum_abs(self, s):
        r = s.eval("sum(abs([-1 -2 3]))")
        assert "6" in r

    def test_length_find(self, s):
        s.eval("v = [0 1 0 1 1];")
        r = s.eval("length(find(v))")
        assert "3" in r


# ============================================================
# R12 - Multi-Output Functions
# ============================================================

class TestR12_MultiOutput:
    def test_max_with_index(self, s):
        s.eval("[m, idx] = max([10 30 20]);")
        r_m = s.eval("m")
        r_idx = s.eval("idx")
        assert "30" in r_m
        assert "2" in r_idx

    def test_find_row_col(self, s):
        s.eval("A = [0 1; 2 0];")
        s.eval("[r, c] = find(A);")
        r_r = s.eval("r")
        r_c = s.eval("c")
        assert "1" in r_r and "2" in r_r

    def test_size_two_outputs(self, s):
        s.eval("A = ones(3, 5);")
        s.eval("[m, n] = size(A);")
        r_m = s.eval("m")
        r_n = s.eval("n")
        assert "3" in r_m
        assert "5" in r_n


# ============================================================
# R13 - .m File Auto-Discovery
# ============================================================

class TestR13_MFileDiscovery:
    def test_function_from_path(self, s, tmp_path):
        mfile = tmp_path / "myadd.m"
        mfile.write_text("function r = myadd(a, b)\n  r = a + b;\nend\n")
        s.eval('addpath("' + str(tmp_path) + '")')
        r = s.eval("myadd(3, 7)")
        assert "10" in r

    def test_function_not_found(self, s):
        with pytest.raises(Exception):
            s.eval("nonexistent_function_xyz(1)")


# ============================================================
# R14 - Script Execution
# ============================================================

class TestR14_ScriptExecution:
    def test_run_script(self, s, tmp_path):
        script = tmp_path / "myscript.m"
        script.write_text("a = 42;\nb = a * 2;\n")
        s.eval('run("' + str(tmp_path / "myscript.m") + '")')
        r = s.eval("b")
        assert "84" in r


# ============================================================
# R15 - Missing Core Functions
# ============================================================

class TestR15_MissingFunctions:
    def test_dot_product(self, s):
        r = s.eval("dot([1 2 3], [4 5 6])")
        assert "32" in r

    def test_tic_toc(self, s):
        s.eval("tic;")
        s.eval("x = ones(10,10);")
        r = s.eval("toc")
        # Should print elapsed time
        assert r.strip() != ""

    def test_sub2ind(self, s):
        r = s.eval("sub2ind([3 4], 2, 3)")
        assert "8" in r

    def test_ind2sub(self, s):
        s.eval("[r, c] = ind2sub([3 4], 8);")
        r_r = s.eval("r")
        r_c = s.eval("c")
        assert "2" in r_r
        assert "3" in r_c

    def test_setdiff(self, s):
        r = s.eval("setdiff([1 2 3 4], [2 4])")
        assert "1" in r and "3" in r

    def test_intersect(self, s):
        r = s.eval("intersect([1 2 3], [2 3 4])")
        assert "2" in r and "3" in r

    def test_union(self, s):
        r = s.eval("union([1 2], [2 3])")
        assert "1" in r and "3" in r


# ============================================================
# R16 - Plot Integration (smoke tests)
# ============================================================

class TestR16_Plotting:
    def test_plot_no_crash(self, s):
        s.eval("x = linspace(0, 6.28, 50);")
        s.eval("y = sin(x);")
        s.eval("plot(x, y)")

    def test_figure_not_function_ref(self, s):
        r = s.eval("figure")
        assert "function" not in str(r).lower() or r.strip() == ""
