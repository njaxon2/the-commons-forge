# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for M-language evaluator (Stages 2.5-2.11)."""
import numpy as np
import pytest
from forge.engine.evaluator import Session, ForgeError, Workspace
from forge.engine.types import ForgeArray
from forge.engine.containers import ForgeChar, ForgeCell, ForgeStruct


@pytest.fixture
def s():
    return Session()


# ============================================================
# Stage 2.5: Expression Evaluation
# ============================================================

class TestArithmeticEval:
    def test_number(self, s):
        r = s.eval("42")
        assert float(r) == 42.0

    def test_add(self, s):
        r = s.eval("2 + 3")
        assert float(r) == 5.0

    def test_subtract(self, s):
        r = s.eval("10 - 4")
        assert float(r) == 6.0

    def test_multiply(self, s):
        r = s.eval("3 * 7")
        assert float(r) == 21.0

    def test_divide(self, s):
        r = s.eval("15 / 3")
        assert float(r) == 5.0

    def test_power(self, s):
        r = s.eval("2 .^ 10")
        assert float(r) == 1024.0

    def test_negate(self, s):
        r = s.eval("-5")
        assert float(r) == -5.0

    def test_complex_expr(self, s):
        r = s.eval("(2 + 3) * 4 - 1")
        assert float(r) == 19.0

    def test_modulo(self, s):
        s.eval("x = mod(17, 5)")
        assert float(s.workspace.get("x")) == 2.0

    def test_hex_literal(self, s):
        r = s.eval("0xFF")
        assert float(r) == 255.0

    def test_imaginary(self, s):
        r = s.eval("3i")
        assert complex(r.data.flat[0]) == 3j


class TestComparisonEval:
    def test_eq_true(self, s):
        r = s.eval("5 == 5")
        assert bool(r.data.flat[0])

    def test_eq_false(self, s):
        r = s.eval("5 == 3")
        assert not bool(r.data.flat[0])

    def test_ne(self, s):
        r = s.eval("5 ~= 3")
        assert bool(r.data.flat[0])

    def test_lt(self, s):
        r = s.eval("3 < 5")
        assert bool(r.data.flat[0])

    def test_ge(self, s):
        r = s.eval("5 >= 5")
        assert bool(r.data.flat[0])


class TestLogicalEval:
    def test_short_circuit_and(self, s):
        r = s.eval("1 && 1")
        assert bool(r.data.flat[0])

    def test_short_circuit_and_false(self, s):
        r = s.eval("0 && 1")
        assert not bool(r.data.flat[0])

    def test_short_circuit_or(self, s):
        r = s.eval("0 || 1")
        assert bool(r.data.flat[0])

    def test_bitwise_and(self, s):
        s.eval("x = [1 0 1] & [1 1 0]")
        np.testing.assert_array_equal(s.workspace.get("x").data.ravel(), [True, False, False])

    def test_not(self, s):
        r = s.eval("~0")
        assert bool(r.data.flat[0])


class TestTransposeEval:
    def test_conjugate_transpose(self, s):
        s.eval("A = [1 2; 3 4]")
        r = s.eval("A'")
        np.testing.assert_array_equal(r.data, [[1, 3], [2, 4]])

    def test_dot_transpose(self, s):
        s.eval("A = [1 2; 3 4]")
        r = s.eval("A.'")
        np.testing.assert_array_equal(r.data, [[1, 3], [2, 4]])


class TestColonEval:
    def test_simple_range(self, s):
        r = s.eval("1:5")
        np.testing.assert_array_equal(r.data.ravel(), [1, 2, 3, 4, 5])

    def test_stepped_range(self, s):
        r = s.eval("0:2:10")
        np.testing.assert_array_equal(r.data.ravel(), [0, 2, 4, 6, 8, 10])


class TestMatrixEval:
    def test_row_vector(self, s):
        r = s.eval("[1, 2, 3]")
        np.testing.assert_array_equal(r.data.ravel(), [1, 2, 3])

    def test_matrix(self, s):
        r = s.eval("[1 2; 3 4]")
        np.testing.assert_array_equal(r.data, [[1, 2], [3, 4]])

    def test_empty_matrix(self, s):
        r = s.eval("[]")
        assert r.isempty()

    def test_matrix_multiply(self, s):
        s.eval("A = [1 2; 3 4]")
        s.eval("B = [5 6; 7 8]")
        r = s.eval("A * B")
        np.testing.assert_array_equal(r.data, [[19, 22], [43, 50]])


class TestIndexingEval:
    def test_array_index(self, s):
        s.eval("x = [10, 20, 30]")
        r = s.eval("x(2)")
        assert float(r) == 20.0

    def test_matrix_index(self, s):
        s.eval("A = [1 2; 3 4]")
        r = s.eval("A(2, 1)")
        assert float(r) == 3.0

    def test_assign_index(self, s):
        s.eval("x = [1, 2, 3]")
        s.eval("x(2) = 99")
        assert s.workspace.get("x")[2] == 99

    def test_colon_index(self, s):
        s.eval("x = [10 20 30 40 50]")
        r = s.eval("x(2:4)")
        np.testing.assert_array_equal(r.data.ravel(), [20, 30, 40])


class TestFunctionCallEval:
    def test_builtin_sqrt(self, s):
        r = s.eval("sqrt(25)")
        assert abs(float(r) - 5.0) < 1e-10

    def test_builtin_sin(self, s):
        r = s.eval("sin(0)")
        assert abs(float(r)) < 1e-10

    def test_builtin_abs(self, s):
        r = s.eval("abs(-5)")
        assert float(r) == 5.0

    def test_builtin_size(self, s):
        s.eval("A = [1 2 3; 4 5 6]")
        r = s.eval("size(A)")
        np.testing.assert_array_equal(r.data.ravel(), [2, 3])

    def test_builtin_length(self, s):
        s.eval("x = [1 2 3 4 5]")
        r = s.eval("length(x)")
        assert float(r) == 5

    def test_builtin_zeros(self, s):
        r = s.eval("zeros(2, 3)")
        assert r.shape == (2, 3)
        assert np.all(r.data == 0)

    def test_builtin_ones(self, s):
        r = s.eval("ones(3)")
        assert r.shape == (3, 3)

    def test_builtin_eye(self, s):
        r = s.eval("eye(3)")
        np.testing.assert_array_equal(r.data, np.eye(3))

    def test_builtin_disp(self, s):
        s.eval("disp('hello')")
        assert "hello" in s.output_buffer.getvalue()

    def test_builtin_sum(self, s):
        r = s.eval("sum([1 2 3 4])")
        assert float(r) == 10.0

    def test_builtin_max(self, s):
        r = s.eval("max([3 1 4 1 5])")
        assert float(r) == 5.0

    def test_builtin_find(self, s):
        r = s.eval("find([0 1 0 1 1])")
        np.testing.assert_array_equal(r.data.ravel(), [2, 4, 5])  # 1-based


class TestAnonymousFunction:
    def test_simple(self, s):
        s.eval("f = @(x) x.^2")
        r = s.eval("f(5)")
        assert float(r) == 25.0

    def test_multi_arg(self, s):
        s.eval("f = @(x, y) x + y")
        r = s.eval("f(3, 4)")
        assert float(r) == 7.0

    def test_closure(self, s):
        s.eval("a = 10")
        s.eval("f = @(x) x + a")
        r = s.eval("f(5)")
        assert float(r) == 15.0


class TestFunctionHandle:
    def test_handle(self, s):
        s.eval("f = @sin")
        r = s.eval("f(0)")
        assert abs(float(r)) < 1e-10


class TestFieldAccessEval:
    def test_struct_field(self, s):
        s.eval("s = struct('x', 1, 'y', 2)")
        r = s.eval("s.x")
        assert r == 1

    def test_struct_assign_field(self, s):
        s.eval("s = struct('x', 1)")
        s.eval("s.y = 42")
        assert s.workspace.get("s")._fields["y"] == 42


class TestCellEval:
    def test_cell_literal(self, s):
        r = s.eval("{1, 'hello', [1 2 3]}")
        assert isinstance(r, ForgeCell)
        assert r.numel() == 3

    def test_cell_index(self, s):
        s.eval("c = {10, 20, 30}")
        r = s.eval("c{2}")
        assert r == 20


# ============================================================
# Stage 2.6: Control Flow
# ============================================================

class TestIfEval:
    def test_if_true(self, s):
        s.eval("x = 5")
        s.eval("if x > 0\n  y = 1;\nend")
        assert float(s.workspace.get("y")) == 1.0

    def test_if_false(self, s):
        s.eval("x = -1")
        s.eval("if x > 0\n  y = 1;\nelse\n  y = -1;\nend")
        assert float(s.workspace.get("y")) == -1.0

    def test_elseif(self, s):
        s.eval("x = 0")
        s.eval("if x > 0\n  y = 1;\nelseif x == 0\n  y = 0;\nelse\n  y = -1;\nend")
        assert float(s.workspace.get("y")) == 0.0


class TestForEval:
    def test_simple_for(self, s):
        s.eval("s = 0")
        s.eval("for i = 1:5\n  s = s + i;\nend")
        assert float(s.workspace.get("s")) == 15.0

    def test_for_break(self, s):
        s.eval("s = 0")
        s.eval("for i = 1:100\n  if i > 5\n    break;\n  end\n  s = s + i;\nend")
        assert float(s.workspace.get("s")) == 15.0

    def test_for_continue(self, s):
        s.eval("s = 0")
        s.eval("for i = 1:10\n  if mod(i, 2) == 0\n    continue;\n  end\n  s = s + i;\nend")
        assert float(s.workspace.get("s")) == 25.0  # 1+3+5+7+9


class TestWhileEval:
    def test_simple_while(self, s):
        s.eval("x = 10")
        s.eval("while x > 0\n  x = x - 3;\nend")
        assert float(s.workspace.get("x")) == -2.0

    def test_while_break(self, s):
        s.eval("x = 0")
        s.eval("while 1\n  x = x + 1;\n  if x >= 5\n    break;\n  end\nend")
        assert float(s.workspace.get("x")) == 5.0


class TestSwitchEval:
    def test_switch(self, s):
        s.eval("x = 2")
        s.eval("switch x\n  case 1\n    y = 'a';\n  case 2\n    y = 'b';\n  otherwise\n    y = 'c';\nend")
        assert s.workspace.get("y").to_str() == "b"

    def test_switch_otherwise(self, s):
        s.eval("x = 99")
        s.eval("switch x\n  case 1\n    y = 'a';\n  otherwise\n    y = 'z';\nend")
        assert s.workspace.get("y").to_str() == "z"


class TestTryCatchEval:
    def test_no_error(self, s):
        s.eval("try\n  x = 5;\ncatch\n  x = -1;\nend")
        assert float(s.workspace.get("x")) == 5.0

    def test_catch_error(self, s):
        s.eval("try\n  error('test error');\ncatch err\n  x = 1;\nend")
        assert float(s.workspace.get("x")) == 1.0

    def test_catch_var(self, s):
        s.eval("try\n  error('myid', 'bad stuff');\ncatch err\n  msg = err.message;\nend")
        assert "bad stuff" in str(s.workspace.get("msg"))


# ============================================================
# Stage 2.7: User Functions
# ============================================================

class TestUserFunctions:
    def test_simple_function(self, s):
        s.eval("function y = square(x)\n  y = x .^ 2;\nend")
        r = s.eval("square(5)")
        assert float(r) == 25.0

    def test_multi_return(self, s):
        s.eval("function [mn, mx] = bounds(x)\n  mn = min(x);\n  mx = max(x);\nend")
        s.eval("[a, b] = bounds([3 1 4 1 5])")
        assert float(s.workspace.get("a")) == 1.0
        assert float(s.workspace.get("b")) == 5.0

    def test_function_no_return(self, s):
        s.eval("function greet()\n  disp('hi');\nend")
        s.eval("greet()")
        assert "hi" in s.output_buffer.getvalue()

    def test_recursive_function(self, s):
        s.eval("function n = factorial(x)\n  if x <= 1\n    n = 1;\n  else\n    n = x * factorial(x - 1);\n  end\nend")
        r = s.eval("factorial(5)")
        assert float(r) == 120.0

    def test_function_with_loop(self, s):
        s.eval("function s = mysum(x)\n  s = 0;\n  for i = 1:length(x)\n    s = s + x(i);\n  end\nend")
        r = s.eval("mysum([1 2 3 4 5])")
        assert float(r) == 15.0


# ============================================================
# Stage 2.8-2.9: I/O and Errors
# ============================================================

class TestIO:
    def test_disp_string(self, s):
        s.eval("disp('hello world')")
        assert "hello world" in s.output_buffer.getvalue()

    def test_disp_number(self, s):
        s.eval("disp(42)")
        assert "42" in s.output_buffer.getvalue()

    def test_sprintf(self, s):
        r = s.eval("sprintf('x = %d', 42)")
        assert r.to_str() == "x = 42"

    def test_error_basic(self, s):
        with pytest.raises(ForgeError, match="something broke"):
            s.eval("error('something broke')")

    def test_error_with_id(self, s):
        with pytest.raises(ForgeError) as exc_info:
            s.eval("error('mypackage:badInput', 'invalid value')")
        assert exc_info.value.identifier == "mypackage:badInput"


# ============================================================
# Stage 2.10-2.11: Constants and Integration
# ============================================================

class TestConstants:
    def test_pi(self, s):
        r = s.eval("pi")
        assert abs(float(r) - 3.14159265) < 1e-6

    def test_true_false(self, s):
        r = s.eval("true")
        assert bool(r.data.flat[0])
        r = s.eval("false")
        assert not bool(r.data.flat[0])


class TestIntegration:
    def test_multi_statement(self, s):
        s.eval("x = 5; y = 10; z = x + y")
        assert float(s.workspace.get("z")) == 15.0

    def test_fibonacci(self, s):
        s.eval("""
a = 1; b = 1
for i = 1:8
  c = a + b;
  a = b;
  b = c;
end
""")
        assert float(s.workspace.get("b")) == 55.0  # fib(10)

    def test_matrix_operations(self, s):
        s.eval("A = [1 2; 3 4]")
        s.eval("B = A' * A")
        np.testing.assert_array_equal(s.workspace.get("B").data, [[10, 14], [14, 20]])

    def test_newton_sqrt(self, s):
        s.eval("""
function y = mysqrt(x)
  y = x / 2;
  for i = 1:20
    y = (y + x / y) / 2;
  end
end
""")
        r = s.eval("mysqrt(2)")
        assert abs(float(r) - 1.41421356) < 1e-6

    def test_workspace_isolation(self, s):
        s.eval("function y = f(x)\n  z = x * 2;\n  y = z;\nend")
        s.eval("result = f(5)")
        assert float(s.workspace.get("result")) == 10.0
        assert not s.workspace.has("z")  # z should not leak
