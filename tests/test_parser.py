"""Tests for M-language parser (Stages 2.2-2.4)."""
import pytest
from forge.engine.parser import (
    parse, Parser, ParseError,
    NumberLiteral, StringLiteral, Identifier, UnaryOp, BinaryOp, CompareOp, LogicalOp,
    TransposeOp, ColonExpr, Index, CellIndex, FieldAccess, DynamicFieldAccess,
    MatrixLiteral, CellLiteral, FunctionHandle, AnonFunction, EndKeyword,
    Assignment, IfStatement, ForStatement, WhileStatement, DoUntilStatement,
    SwitchStatement, TryCatchStatement, ReturnStatement, BreakStatement, ContinueStatement,
    FunctionDef, ExpressionStatement, GlobalStatement, PersistentStatement,
)


def expr(src):
    """Parse single expression from source."""
    stmts = parse(src)
    assert len(stmts) == 1
    if isinstance(stmts[0], ExpressionStatement):
        return stmts[0].expr
    return stmts[0]


# ============================================================
# Stage 2.2: Expression Parser
# ============================================================

class TestNumberParsing:
    def test_integer(self):
        e = expr("42")
        assert isinstance(e, NumberLiteral)
        assert e.value == "42"

    def test_float(self):
        e = expr("3.14")
        assert e.value == "3.14"

    def test_scientific(self):
        e = expr("1e-5")
        assert e.value == "1e-5"

    def test_imaginary(self):
        e = expr("3i")
        assert e.value == "3i"


class TestStringParsing:
    def test_double_quoted(self):
        e = expr('"hello"')
        assert isinstance(e, StringLiteral)
        assert e.value == "hello"
        assert not e.is_char

    def test_single_quoted(self):
        e = expr("'world'")
        assert isinstance(e, StringLiteral)
        assert e.value == "world"
        assert e.is_char


class TestIdentifier:
    def test_simple(self):
        e = expr("myVar")
        assert isinstance(e, Identifier)
        assert e.name == "myVar"


class TestUnaryOps:
    def test_negate(self):
        e = expr("-x")
        assert isinstance(e, UnaryOp)
        assert e.op == "-"
        assert isinstance(e.operand, Identifier)

    def test_not(self):
        e = expr("~x")
        assert isinstance(e, UnaryOp)
        assert e.op == "~"

    def test_positive(self):
        e = expr("+x")
        # Unary + is identity, returns the operand directly
        assert isinstance(e, Identifier)


class TestBinaryOps:
    def test_add(self):
        e = expr("a + b")
        assert isinstance(e, BinaryOp)
        assert e.op == "+"
        assert isinstance(e.left, Identifier)
        assert isinstance(e.right, Identifier)

    def test_subtract(self):
        e = expr("a - b")
        assert isinstance(e, BinaryOp) and e.op == "-"

    def test_multiply(self):
        e = expr("a * b")
        assert isinstance(e, BinaryOp) and e.op == "*"

    def test_divide(self):
        e = expr("a / b")
        assert isinstance(e, BinaryOp) and e.op == "/"

    def test_power(self):
        e = expr("a ^ b")
        assert isinstance(e, BinaryOp) and e.op == "^"

    def test_element_wise_multiply(self):
        e = expr("a .* b")
        assert isinstance(e, BinaryOp) and e.op == ".*"

    def test_element_wise_divide(self):
        e = expr("a ./ b")
        assert isinstance(e, BinaryOp) and e.op == "./"

    def test_element_wise_power(self):
        e = expr("a .^ b")
        assert isinstance(e, BinaryOp) and e.op == ".^"

    def test_backslash(self):
        e = expr("A \\ b")
        assert isinstance(e, BinaryOp) and e.op == "\\"


class TestPrecedence:
    def test_mul_before_add(self):
        e = expr("a + b * c")
        assert isinstance(e, BinaryOp) and e.op == "+"
        assert isinstance(e.right, BinaryOp) and e.right.op == "*"

    def test_power_before_mul(self):
        e = expr("a * b ^ c")
        assert isinstance(e, BinaryOp) and e.op == "*"
        assert isinstance(e.right, BinaryOp) and e.right.op == "^"

    def test_parentheses_override(self):
        e = expr("(a + b) * c")
        assert isinstance(e, BinaryOp) and e.op == "*"
        assert isinstance(e.left, BinaryOp) and e.left.op == "+"

    def test_power_right_associative(self):
        e = expr("a ^ b ^ c")
        assert isinstance(e, BinaryOp) and e.op == "^"
        assert isinstance(e.right, BinaryOp) and e.right.op == "^"

    def test_comparison_lower_than_arithmetic(self):
        e = expr("a + b == c * d")
        assert isinstance(e, CompareOp)
        assert isinstance(e.left, BinaryOp) and e.left.op == "+"
        assert isinstance(e.right, BinaryOp) and e.right.op == "*"

    def test_logical_lower_than_compare(self):
        e = expr("a > 0 && b < 10")
        assert isinstance(e, LogicalOp) and e.op == "&&"
        assert isinstance(e.left, CompareOp)
        assert isinstance(e.right, CompareOp)

    def test_or_lower_than_and(self):
        e = expr("a && b || c")
        assert isinstance(e, LogicalOp) and e.op == "||"
        assert isinstance(e.left, LogicalOp) and e.left.op == "&&"

    def test_unary_minus_precedence(self):
        e = expr("-a ^ b")
        # Should be -(a^b), not (-a)^b
        assert isinstance(e, UnaryOp)
        assert isinstance(e.operand, BinaryOp) and e.operand.op == "^"


class TestComparison:
    def test_eq(self):
        e = expr("a == b")
        assert isinstance(e, CompareOp) and e.op == "=="

    def test_ne(self):
        e = expr("a ~= b")
        assert isinstance(e, CompareOp) and e.op == "~="

    def test_lt_gt_le_ge(self):
        for op in ["<", ">", "<=", ">="]:
            e = expr(f"a {op} b")
            assert isinstance(e, CompareOp) and e.op == op


class TestTranspose:
    def test_simple(self):
        e = expr("A'")
        assert isinstance(e, TransposeOp) and e.conjugate

    def test_dot_transpose(self):
        e = expr("A.'")
        assert isinstance(e, TransposeOp) and not e.conjugate

    def test_transpose_in_expr(self):
        e = expr("A' * B")
        assert isinstance(e, BinaryOp) and e.op == "*"
        assert isinstance(e.left, TransposeOp)


class TestColonExpr:
    def test_simple_range(self):
        e = expr("1:10")
        assert isinstance(e, ColonExpr)
        assert e.step is None

    def test_stepped_range(self):
        e = expr("1:2:10")
        assert isinstance(e, ColonExpr)
        assert e.step is not None


class TestIndexing:
    def test_function_call(self):
        e = expr("f(x, y)")
        assert isinstance(e, Index)
        assert isinstance(e.target, Identifier) and e.target.name == "f"
        assert len(e.args) == 2

    def test_array_indexing(self):
        e = expr("A(1, 2)")
        assert isinstance(e, Index)
        assert len(e.args) == 2

    def test_cell_indexing(self):
        e = expr("C{1}")
        assert isinstance(e, CellIndex)
        assert len(e.args) == 1

    def test_chained_indexing(self):
        e = expr("A(1).field")
        assert isinstance(e, FieldAccess)
        assert isinstance(e.target, Index)

    def test_nested_indexing(self):
        e = expr("A(B(1))")
        assert isinstance(e, Index)
        assert isinstance(e.args[0], Index)


class TestFieldAccess:
    def test_simple(self):
        e = expr("s.field")
        assert isinstance(e, FieldAccess)
        assert e.field == "field"

    def test_chained(self):
        e = expr("a.b.c")
        assert isinstance(e, FieldAccess) and e.field == "c"
        assert isinstance(e.target, FieldAccess) and e.target.field == "b"

    def test_dynamic_field(self):
        e = expr("s.(name)")
        assert isinstance(e, DynamicFieldAccess)


class TestMatrixLiteral:
    def test_row_vector(self):
        e = expr("[1, 2, 3]")
        assert isinstance(e, MatrixLiteral)
        assert len(e.rows) == 1 and len(e.rows[0]) == 3

    def test_matrix(self):
        e = expr("[1 2; 3 4]")
        assert isinstance(e, MatrixLiteral)
        assert len(e.rows) == 2

    def test_empty(self):
        e = expr("[]")
        assert isinstance(e, MatrixLiteral)
        assert len(e.rows) == 0


class TestCellLiteral:
    def test_simple(self):
        e = expr("{1, 'a', [2 3]}")
        assert isinstance(e, CellLiteral)
        assert len(e.rows) == 1
        assert len(e.rows[0]) == 3

    def test_multi_row(self):
        e = expr("{1, 2; 3, 4}")
        assert isinstance(e, CellLiteral)
        assert len(e.rows) == 2


class TestFunctionHandle:
    def test_named(self):
        e = expr("@sin")
        assert isinstance(e, FunctionHandle)
        assert e.name == "sin"

    def test_anonymous(self):
        e = expr("@(x) x.^2")
        assert isinstance(e, AnonFunction)
        assert e.args == ["x"]
        assert isinstance(e.body, BinaryOp) and e.body.op == ".^"

    def test_multi_arg_anon(self):
        e = expr("@(x, y) x + y")
        assert isinstance(e, AnonFunction)
        assert e.args == ["x", "y"]


class TestEndKeyword:
    def test_end_in_indexing(self):
        e = expr("A(end)")
        assert isinstance(e, Index)
        assert isinstance(e.args[0], EndKeyword)

    def test_end_in_range(self):
        e = expr("A(1:end)")
        assert isinstance(e, Index)
        arg = e.args[0]
        assert isinstance(arg, ColonExpr)
        assert isinstance(arg.stop, EndKeyword)


# ============================================================
# Stage 2.3: Statement Parser
# ============================================================

class TestAssignment:
    def test_simple(self):
        s = parse("x = 5")[0]
        assert isinstance(s, Assignment)
        assert isinstance(s.targets, Identifier) and s.targets.name == "x"

    def test_indexed(self):
        s = parse("A(1, 2) = 5")[0]
        assert isinstance(s, Assignment)
        assert isinstance(s.targets, Index)

    def test_field(self):
        s = parse("s.x = 10")[0]
        assert isinstance(s, Assignment)
        assert isinstance(s.targets, FieldAccess)

    def test_multi_return(self):
        s = parse("[a, b] = func(x)")[0]
        assert isinstance(s, Assignment)
        assert isinstance(s.targets, list) and len(s.targets) == 2


class TestExprStatement:
    def test_bare_expression(self):
        s = parse("disp(x)")[0]
        assert isinstance(s, ExpressionStatement)
        assert s.print_result is True

    def test_semicolon_suppresses(self):
        s = parse("x = 5;")[0]
        assert isinstance(s, Assignment)


class TestIfStatement:
    def test_simple_if(self):
        s = parse("if x > 0\n  y = 1;\nend")[0]
        assert isinstance(s, IfStatement)
        assert isinstance(s.condition, CompareOp)
        assert len(s.body) == 1

    def test_if_else(self):
        s = parse("if x > 0\n  y = 1;\nelse\n  y = -1;\nend")[0]
        assert isinstance(s, IfStatement)
        assert s.else_body is not None
        assert len(s.else_body) == 1

    def test_if_elseif_else(self):
        src = "if x > 0\n  y = 1;\nelseif x == 0\n  y = 0;\nelse\n  y = -1;\nend"
        s = parse(src)[0]
        assert isinstance(s, IfStatement)
        assert len(s.elseifs) == 1
        assert s.else_body is not None

    def test_endif(self):
        s = parse("if true\n  x = 1;\nendif")[0]
        assert isinstance(s, IfStatement)


class TestForLoop:
    def test_simple(self):
        s = parse("for i = 1:10\n  x(i) = i;\nend")[0]
        assert isinstance(s, ForStatement)
        assert s.var == "i"
        assert isinstance(s.iter_expr, ColonExpr)

    def test_endfor(self):
        s = parse("for i = 1:5\n  x = i;\nendfor")[0]
        assert isinstance(s, ForStatement)


class TestWhileLoop:
    def test_simple(self):
        s = parse("while x > 0\n  x = x - 1;\nend")[0]
        assert isinstance(s, WhileStatement)
        assert isinstance(s.condition, CompareOp)


class TestDoUntil:
    def test_simple(self):
        s = parse("do\n  x = x + 1;\nuntil x > 10")[0]
        assert isinstance(s, DoUntilStatement)
        assert isinstance(s.condition, CompareOp)


class TestSwitch:
    def test_simple(self):
        src = "switch x\n  case 1\n    y = 'a';\n  case 2\n    y = 'b';\n  otherwise\n    y = 'c';\nend"
        s = parse(src)[0]
        assert isinstance(s, SwitchStatement)
        assert len(s.cases) == 2
        assert s.otherwise_body is not None


class TestTryCatch:
    def test_simple(self):
        s = parse("try\n  x = risky();\ncatch err\n  disp(err);\nend")[0]
        assert isinstance(s, TryCatchStatement)
        assert s.catch_var == "err"
        assert len(s.try_body) == 1
        assert len(s.catch_body) == 1

    def test_try_without_catch_var(self):
        s = parse("try\n  x = 1;\ncatch\n  x = 0;\nend")[0]
        assert isinstance(s, TryCatchStatement)
        assert s.catch_var is None


class TestControlFlow:
    def test_return(self):
        s = parse("return")[0]
        assert isinstance(s, ReturnStatement)

    def test_break(self):
        s = parse("break")[0]
        assert isinstance(s, BreakStatement)

    def test_continue(self):
        s = parse("continue")[0]
        assert isinstance(s, ContinueStatement)

    def test_global(self):
        s = parse("global x y z")[0]
        assert isinstance(s, GlobalStatement)
        assert s.names == ["x", "y", "z"]

    def test_persistent(self):
        s = parse("persistent count")[0]
        assert isinstance(s, PersistentStatement)
        assert s.names == ["count"]


# ============================================================
# Stage 2.4: Function Definitions
# ============================================================

class TestFunctionDef:
    def test_simple(self):
        src = "function y = square(x)\n  y = x.^2;\nend"
        s = parse(src)[0]
        assert isinstance(s, FunctionDef)
        assert s.name == "square"
        assert s.params == ["x"]
        assert s.returns == ["y"]
        assert len(s.body) == 1

    def test_no_return(self):
        src = "function greet(name)\n  disp(name);\nend"
        s = parse(src)[0]
        assert isinstance(s, FunctionDef)
        assert s.returns == []
        assert s.params == ["name"]

    def test_multi_return(self):
        src = "function [mn, mx] = bounds(x)\n  mn = min(x);\n  mx = max(x);\nend"
        s = parse(src)[0]
        assert isinstance(s, FunctionDef)
        assert s.returns == ["mn", "mx"]
        assert s.params == ["x"]
        assert len(s.body) == 2

    def test_no_args(self):
        src = "function x = get_value()\n  x = 42;\nend"
        s = parse(src)[0]
        assert isinstance(s, FunctionDef)
        assert s.params == []
        assert s.returns == ["x"]

    def test_endfunction(self):
        src = "function y = f(x)\n  y = x;\nendfunction"
        s = parse(src)[0]
        assert isinstance(s, FunctionDef)


class TestNestedStatements:
    def test_for_with_if(self):
        src = "for i = 1:10\n  if i > 5\n    break;\n  end\nend"
        s = parse(src)[0]
        assert isinstance(s, ForStatement)
        assert isinstance(s.body[0], IfStatement)

    def test_function_with_for(self):
        src = "function s = mysum(x)\n  s = 0;\n  for i = 1:length(x)\n    s = s + x(i);\n  end\nend"
        s = parse(src)[0]
        assert isinstance(s, FunctionDef)
        assert any(isinstance(stmt, ForStatement) for stmt in s.body)


class TestMultipleStatements:
    def test_sequence(self):
        src = "x = 1\ny = 2\nz = x + y"
        stmts = parse(src)
        assert len(stmts) == 3
        assert all(isinstance(s, Assignment) for s in stmts)

    def test_semicolon_separated(self):
        src = "x = 1; y = 2; z = 3"
        stmts = parse(src)
        assert len(stmts) == 3
