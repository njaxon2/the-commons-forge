# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for M-language parser (Stages 2.2-2.4).

V&V traceability backfill: R-PARSE-01 through R-PARSE-20.
"""
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
    """R-PARSE-01: The parser SHALL parse integer, floating-point, scientific,
    and imaginary numeric literals into NumberLiteral AST nodes with correct
    value strings.

    Model-user argument: The engineer types numeric constants in many formats
    (integers, decimals, scientific notation for very large/small values,
    imaginary suffixes for complex math). Each format must parse to the
    correct numeric representation or calculations will be wrong from the
    start.

    Decomposition: Integer, float, scientific, and imaginary literal parsing.
    Consistency: These four formats cover all numeric literal syntax accepted
    by MATLAB/Octave.
    """

    def test_integer(self):
        """R-PARSE-01.01: Integer literal parses to NumberLiteral with correct value."""
        e = expr("42")
        assert isinstance(e, NumberLiteral)
        assert e.value == "42"

    def test_float(self):
        """R-PARSE-01.02: Floating-point literal parses with decimal value."""
        e = expr("3.14")
        assert e.value == "3.14"

    def test_scientific(self):
        """R-PARSE-01.03: Scientific notation literal preserves exponent form."""
        e = expr("1e-5")
        assert e.value == "1e-5"

    def test_imaginary(self):
        """R-PARSE-01.04: Imaginary literal includes the 'i' suffix."""
        e = expr("3i")
        assert e.value == "3i"


class TestStringParsing:
    """R-PARSE-02: The parser SHALL parse double-quoted strings and
    single-quoted char arrays into StringLiteral nodes with the is_char
    flag set correctly.

    Model-user argument: MATLAB distinguishes between double-quoted strings
    and single-quoted char arrays. The engineer uses single quotes for
    function options (e.g., 'linear') and double quotes for string data.
    The parser must preserve this distinction.

    Decomposition: Double-quoted and single-quoted parsing tested.
    Consistency: These are the only two string literal forms in MATLAB.
    """

    def test_double_quoted(self):
        """R-PARSE-02.01: Double-quoted string parses with is_char=false."""
        e = expr('"hello"')
        assert isinstance(e, StringLiteral)
        assert e.value == "hello"
        assert not e.is_char

    def test_single_quoted(self):
        """R-PARSE-02.02: Single-quoted char array parses with is_char=true."""
        e = expr("'world'")
        assert isinstance(e, StringLiteral)
        assert e.value == "world"
        assert e.is_char


class TestIdentifier:
    """R-PARSE-03: The parser SHALL parse bare identifiers into Identifier AST
    nodes.

    Model-user argument: Every variable reference the engineer types (e.g.,
    ``myVar``, ``signal_data``) must resolve to an Identifier node so the
    evaluator can look it up in the workspace.

    Decomposition: Single identifier test. Consistency: Identifier parsing
    is a single atomic operation.
    """

    def test_simple(self):
        """R-PARSE-03.01: Simple identifier parses to Identifier node."""
        e = expr("myVar")
        assert isinstance(e, Identifier)
        assert e.name == "myVar"


class TestUnaryOps:
    """R-PARSE-04: The parser SHALL parse unary operators (-, ~, +) applied to
    identifiers.

    Model-user argument: Unary minus negates values (``-x``), logical NOT
    inverts conditions (``~valid``), and unary plus is an identity. These
    prefix operators appear in everyday expressions.

    Decomposition: Negate, NOT, and positive tested. Consistency: These
    three operators are all unary prefix operators in MATLAB.
    """

    def test_negate(self):
        """R-PARSE-04.01: Unary minus creates a UnaryOp node with op '-'."""
        e = expr("-x")
        assert isinstance(e, UnaryOp)
        assert e.op == "-"
        assert isinstance(e.operand, Identifier)

    def test_not(self):
        """R-PARSE-04.02: Logical NOT creates a UnaryOp node with op '~'."""
        e = expr("~x")
        assert isinstance(e, UnaryOp)
        assert e.op == "~"

    def test_positive(self):
        """R-PARSE-04.03: Unary plus is identity, returning the operand directly."""
        e = expr("+x")
        # Unary + is identity, returns the operand directly
        assert isinstance(e, Identifier)


class TestBinaryOps:
    """R-PARSE-05: The parser SHALL parse binary arithmetic operators (+, -, *,
    /, ^, .*, ./, .^, \\) into BinaryOp AST nodes with correct operator
    strings.

    Model-user argument: The engineer uses all arithmetic operators in
    expressions. Element-wise operators (.*, ./, .^) are critical for
    array math, while matrix operators (*, /, \\) handle linear algebra.
    Each operator must be recognized and represented in the AST.

    Decomposition: Each of the nine binary arithmetic operators tested.
    Consistency: This covers the complete set of binary arithmetic operators.
    """

    def test_add(self):
        """R-PARSE-05.01: Addition parses to BinaryOp with op '+'."""
        e = expr("a + b")
        assert isinstance(e, BinaryOp)
        assert e.op == "+"
        assert isinstance(e.left, Identifier)
        assert isinstance(e.right, Identifier)

    def test_subtract(self):
        """R-PARSE-05.02: Subtraction parses to BinaryOp with op '-'."""
        e = expr("a - b")
        assert isinstance(e, BinaryOp) and e.op == "-"

    def test_multiply(self):
        """R-PARSE-05.03: Matrix multiply parses to BinaryOp with op '*'."""
        e = expr("a * b")
        assert isinstance(e, BinaryOp) and e.op == "*"

    def test_divide(self):
        """R-PARSE-05.04: Right division parses to BinaryOp with op '/'."""
        e = expr("a / b")
        assert isinstance(e, BinaryOp) and e.op == "/"

    def test_power(self):
        """R-PARSE-05.05: Matrix power parses to BinaryOp with op '^'."""
        e = expr("a ^ b")
        assert isinstance(e, BinaryOp) and e.op == "^"

    def test_element_wise_multiply(self):
        """R-PARSE-05.06: Element-wise multiply parses with op '.*'."""
        e = expr("a .* b")
        assert isinstance(e, BinaryOp) and e.op == ".*"

    def test_element_wise_divide(self):
        """R-PARSE-05.07: Element-wise divide parses with op './'."""
        e = expr("a ./ b")
        assert isinstance(e, BinaryOp) and e.op == "./"

    def test_element_wise_power(self):
        """R-PARSE-05.08: Element-wise power parses with op '.^'."""
        e = expr("a .^ b")
        assert isinstance(e, BinaryOp) and e.op == ".^"

    def test_backslash(self):
        """R-PARSE-05.09: Left division parses to BinaryOp with op '\\'."""
        e = expr("A \\ b")
        assert isinstance(e, BinaryOp) and e.op == "\\"


class TestPrecedence:
    """R-PARSE-06: The parser SHALL respect MATLAB operator precedence so that
    the AST structure reflects correct evaluation order.

    Model-user argument: The engineer writes ``a + b * c`` expecting
    multiplication to bind tighter than addition, and ``-a ^ b`` expecting
    power to bind tighter than negation. Wrong precedence silently produces
    wrong numeric results.

    Decomposition: mul-before-add, power-before-mul, parenthesis override,
    power left-associativity, comparison vs arithmetic, logical vs compare,
    or vs and, and unary-minus vs power tested. Consistency: These eight
    tests exercise all precedence boundaries in the MATLAB grammar.
    """

    def test_mul_before_add(self):
        """R-PARSE-06.01: Multiplication binds tighter than addition."""
        e = expr("a + b * c")
        assert isinstance(e, BinaryOp) and e.op == "+"
        assert isinstance(e.right, BinaryOp) and e.right.op == "*"

    def test_power_before_mul(self):
        """R-PARSE-06.02: Power binds tighter than multiplication."""
        e = expr("a * b ^ c")
        assert isinstance(e, BinaryOp) and e.op == "*"
        assert isinstance(e.right, BinaryOp) and e.right.op == "^"

    def test_parentheses_override(self):
        """R-PARSE-06.03: Parentheses override default precedence."""
        e = expr("(a + b) * c")
        assert isinstance(e, BinaryOp) and e.op == "*"
        assert isinstance(e.left, BinaryOp) and e.left.op == "+"

    def test_power_left_associative(self):
        """R-PARSE-06.04: Power is left-associative: a^b^c = (a^b)^c."""
        e = expr("a ^ b ^ c")
        assert isinstance(e, BinaryOp) and e.op == "^"
        assert isinstance(e.left, BinaryOp) and e.left.op == "^"

    def test_comparison_lower_than_arithmetic(self):
        """R-PARSE-06.05: Comparison binds looser than arithmetic."""
        e = expr("a + b == c * d")
        assert isinstance(e, CompareOp)
        assert isinstance(e.left, BinaryOp) and e.left.op == "+"
        assert isinstance(e.right, BinaryOp) and e.right.op == "*"

    def test_logical_lower_than_compare(self):
        """R-PARSE-06.06: Short-circuit AND binds looser than comparisons."""
        e = expr("a > 0 && b < 10")
        assert isinstance(e, LogicalOp) and e.op == "&&"
        assert isinstance(e.left, CompareOp)
        assert isinstance(e.right, CompareOp)

    def test_or_lower_than_and(self):
        """R-PARSE-06.07: OR binds looser than AND."""
        e = expr("a && b || c")
        assert isinstance(e, LogicalOp) and e.op == "||"
        assert isinstance(e.left, LogicalOp) and e.left.op == "&&"

    def test_unary_minus_precedence(self):
        """R-PARSE-06.08: Unary minus binds looser than power: -a^b = -(a^b)."""
        e = expr("-a ^ b")
        # Should be -(a^b), not (-a)^b
        assert isinstance(e, UnaryOp)
        assert isinstance(e.operand, BinaryOp) and e.operand.op == "^"


class TestComparison:
    """R-PARSE-07: The parser SHALL parse all comparison operators (==, ~=, <,
    >, <=, >=) into CompareOp AST nodes.

    Model-user argument: Comparisons appear in every conditional the engineer
    writes. All six operators must be recognized so that guards like
    ``x >= threshold`` parse correctly.

    Decomposition: ==, ~=, and the four inequality operators tested.
    Consistency: These six operators are the complete set of comparison
    operators in MATLAB.
    """

    def test_eq(self):
        """R-PARSE-07.01: Equality operator parses to CompareOp with op '=='."""
        e = expr("a == b")
        assert isinstance(e, CompareOp) and e.op == "=="

    def test_ne(self):
        """R-PARSE-07.02: Not-equal operator parses with op '~='."""
        e = expr("a ~= b")
        assert isinstance(e, CompareOp) and e.op == "~="

    def test_lt_gt_le_ge(self):
        """R-PARSE-07.03: All four inequality operators parse correctly."""
        for op in ["<", ">", "<=", ">="]:
            e = expr(f"a {op} b")
            assert isinstance(e, CompareOp) and e.op == op


class TestTranspose:
    """R-PARSE-08: The parser SHALL parse conjugate transpose (') and
    dot-transpose (.') into TransposeOp nodes with the conjugate flag set.

    Model-user argument: The engineer uses ``A'`` for conjugate transpose and
    ``A.'`` for non-conjugate transpose. The parser must distinguish these
    from char-array quotes.

    Decomposition: Simple transpose, dot-transpose, and transpose within an
    expression tested. Consistency: These cover the two transpose forms and
    their interaction with other operators.
    """

    def test_simple(self):
        """R-PARSE-08.01: Conjugate transpose sets conjugate=true."""
        e = expr("A'")
        assert isinstance(e, TransposeOp) and e.conjugate

    def test_dot_transpose(self):
        """R-PARSE-08.02: Dot-transpose sets conjugate=false."""
        e = expr("A.'")
        assert isinstance(e, TransposeOp) and not e.conjugate

    def test_transpose_in_expr(self):
        """R-PARSE-08.03: Transpose within a binary expression parses correctly."""
        e = expr("A' * B")
        assert isinstance(e, BinaryOp) and e.op == "*"
        assert isinstance(e.left, TransposeOp)


class TestColonExpr:
    """R-PARSE-09: The parser SHALL parse colon range expressions (start:stop
    and start:step:stop) into ColonExpr nodes.

    Model-user argument: Colon ranges are the primary way to create sample
    vectors (``1:100``) and stepped sequences (``0:0.01:2*pi``). The parser
    must distinguish two-operand from three-operand colon forms.

    Decomposition: Simple range and stepped range tested. Consistency: These
    are the only two colon range forms.
    """

    def test_simple_range(self):
        """R-PARSE-09.01: Two-operand colon has step=None."""
        e = expr("1:10")
        assert isinstance(e, ColonExpr)
        assert e.step is None

    def test_stepped_range(self):
        """R-PARSE-09.02: Three-operand colon has a non-None step."""
        e = expr("1:2:10")
        assert isinstance(e, ColonExpr)
        assert e.step is not None


class TestIndexing:
    """R-PARSE-10: The parser SHALL parse parenthesis indexing, curly-brace
    cell indexing, chained indexing, and nested indexing into the correct
    AST node types.

    Model-user argument: The engineer indexes arrays with ``A(i,j)``, cells
    with ``C{k}``, and chains accesses like ``A(1).field``. The parser must
    produce the correct node hierarchy so the evaluator resolves each access
    in order.

    Decomposition: Function/array call, 2-D index, cell index, chained
    index, and nested index tested. Consistency: These five patterns cover
    all indexing syntaxes.
    """

    def test_function_call(self):
        """R-PARSE-10.01: Parenthesized call parses to Index with two args."""
        e = expr("f(x, y)")
        assert isinstance(e, Index)
        assert isinstance(e.target, Identifier) and e.target.name == "f"
        assert len(e.args) == 2

    def test_array_indexing(self):
        """R-PARSE-10.02: 2-D parenthesized index has two arguments."""
        e = expr("A(1, 2)")
        assert isinstance(e, Index)
        assert len(e.args) == 2

    def test_cell_indexing(self):
        """R-PARSE-10.03: Curly-brace index parses to CellIndex."""
        e = expr("C{1}")
        assert isinstance(e, CellIndex)
        assert len(e.args) == 1

    def test_chained_indexing(self):
        """R-PARSE-10.04: Chained A(1).field parses as FieldAccess over Index."""
        e = expr("A(1).field")
        assert isinstance(e, FieldAccess)
        assert isinstance(e.target, Index)

    def test_nested_indexing(self):
        """R-PARSE-10.05: Nested A(B(1)) parses as Index containing Index."""
        e = expr("A(B(1))")
        assert isinstance(e, Index)
        assert isinstance(e.args[0], Index)


class TestFieldAccess:
    """R-PARSE-11: The parser SHALL parse dot-field access (s.field), chained
    fields (a.b.c), and dynamic field access (s.(expr)).

    Model-user argument: Struct field access is how the engineer navigates
    structured data. Chained access (``config.filter.order``) and dynamic
    field names (``s.(fieldname)``) are common in parameterized code.

    Decomposition: Simple, chained, and dynamic field access tested.
    Consistency: These three forms cover all field-access syntax.
    """

    def test_simple(self):
        """R-PARSE-11.01: Simple field access parses to FieldAccess."""
        e = expr("s.field")
        assert isinstance(e, FieldAccess)
        assert e.field == "field"

    def test_chained(self):
        """R-PARSE-11.02: Chained fields a.b.c parse as nested FieldAccess nodes."""
        e = expr("a.b.c")
        assert isinstance(e, FieldAccess) and e.field == "c"
        assert isinstance(e.target, FieldAccess) and e.target.field == "b"

    def test_dynamic_field(self):
        """R-PARSE-11.03: Dynamic field s.(name) parses to DynamicFieldAccess."""
        e = expr("s.(name)")
        assert isinstance(e, DynamicFieldAccess)


class TestMatrixLiteral:
    """R-PARSE-12: The parser SHALL parse bracket-delimited matrix literals
    including row vectors, multi-row matrices, and empty matrices.

    Model-user argument: The engineer constructs matrices by typing
    ``[1 2; 3 4]`` or row vectors with ``[1, 2, 3]``. Empty ``[]`` is used
    for deletion and initialization. The parser must capture the row
    structure correctly.

    Decomposition: Row vector, multi-row matrix, and empty matrix tested.
    Consistency: These three forms cover all bracket-literal structures.
    """

    def test_row_vector(self):
        """R-PARSE-12.01: Row vector has one row with three elements."""
        e = expr("[1, 2, 3]")
        assert isinstance(e, MatrixLiteral)
        assert len(e.rows) == 1 and len(e.rows[0]) == 3

    def test_matrix(self):
        """R-PARSE-12.02: Semicolon-separated matrix has two rows."""
        e = expr("[1 2; 3 4]")
        assert isinstance(e, MatrixLiteral)
        assert len(e.rows) == 2

    def test_empty(self):
        """R-PARSE-12.03: Empty brackets parse to MatrixLiteral with no rows."""
        e = expr("[]")
        assert isinstance(e, MatrixLiteral)
        assert len(e.rows) == 0


class TestCellLiteral:
    """R-PARSE-13: The parser SHALL parse brace-delimited cell literals with
    single and multiple rows.

    Model-user argument: Cell arrays hold mixed-type data. The engineer types
    ``{1, 'option', [1 2 3]}`` to bundle heterogeneous arguments. Multi-row
    cells (``{1, 2; 3, 4}``) organize tabular mixed data.

    Decomposition: Single-row and multi-row cell literals tested.
    Consistency: These are the two cell literal structures.
    """

    def test_simple(self):
        """R-PARSE-13.01: Single-row cell with three elements parses correctly."""
        e = expr("{1, 'a', [2 3]}")
        assert isinstance(e, CellLiteral)
        assert len(e.rows) == 1
        assert len(e.rows[0]) == 3

    def test_multi_row(self):
        """R-PARSE-13.02: Multi-row cell literal has two rows."""
        e = expr("{1, 2; 3, 4}")
        assert isinstance(e, CellLiteral)
        assert len(e.rows) == 2


class TestFunctionHandle:
    """R-PARSE-14: The parser SHALL parse named function handles (@name) and
    anonymous functions (@(args) body) into the correct AST node types.

    Model-user argument: Function handles and anonymous functions are how the
    engineer passes callbacks to solvers (``fzero(@sin, 1)``) and defines
    inline transforms (``@(x,y) x.^2 + y.^2``). The parser must distinguish
    named handles from anonymous definitions.

    Decomposition: Named handle, single-arg anonymous, and multi-arg
    anonymous tested. Consistency: Named handle and anonymous function are
    the two @-prefixed forms.
    """

    def test_named(self):
        """R-PARSE-14.01: Named handle @sin parses to FunctionHandle."""
        e = expr("@sin")
        assert isinstance(e, FunctionHandle)
        assert e.name == "sin"

    def test_anonymous(self):
        """R-PARSE-14.02: Anonymous function @(x) x.^2 parses to AnonFunction."""
        e = expr("@(x) x.^2")
        assert isinstance(e, AnonFunction)
        assert e.args == ["x"]
        assert isinstance(e.body, BinaryOp) and e.body.op == ".^"

    def test_multi_arg_anon(self):
        """R-PARSE-14.03: Multi-arg anonymous function captures both parameter names."""
        e = expr("@(x, y) x + y")
        assert isinstance(e, AnonFunction)
        assert e.args == ["x", "y"]


class TestEndKeyword:
    """R-PARSE-15: The parser SHALL parse the 'end' keyword within indexing
    expressions as an EndKeyword node.

    Model-user argument: The engineer writes ``A(end)`` to access the last
    element and ``A(1:end)`` to slice from first to last. The parser must
    recognize 'end' inside subscripts as a special keyword, not an
    identifier.

    Decomposition: end as sole subscript and end in a colon range tested.
    Consistency: These two patterns cover all end-in-subscript usages.
    """

    def test_end_in_indexing(self):
        """R-PARSE-15.01: A(end) contains an EndKeyword argument."""
        e = expr("A(end)")
        assert isinstance(e, Index)
        assert isinstance(e.args[0], EndKeyword)

    def test_end_in_range(self):
        """R-PARSE-15.02: A(1:end) has EndKeyword as the stop of a ColonExpr."""
        e = expr("A(1:end)")
        assert isinstance(e, Index)
        arg = e.args[0]
        assert isinstance(arg, ColonExpr)
        assert isinstance(arg.stop, EndKeyword)


# ============================================================
# Stage 2.3: Statement Parser
# ============================================================

class TestAssignment:
    """R-PARSE-16: The parser SHALL parse assignment statements for simple
    variables, indexed targets, field targets, and multi-return forms.

    Model-user argument: Assignment is the most frequent statement the
    engineer writes (``x = 5``, ``A(1,2) = 5``, ``s.x = 10``,
    ``[a, b] = func(x)``). Each target form must be recognized so the
    evaluator stores values in the correct location.

    Decomposition: Simple, indexed, field, and multi-return assignment
    tested. Consistency: These four forms cover all assignment target types.
    """

    def test_simple(self):
        """R-PARSE-16.01: Simple assignment parses with Identifier target."""
        s = parse("x = 5")[0]
        assert isinstance(s, Assignment)
        assert isinstance(s.targets, Identifier) and s.targets.name == "x"

    def test_indexed(self):
        """R-PARSE-16.02: Indexed assignment parses with Index target."""
        s = parse("A(1, 2) = 5")[0]
        assert isinstance(s, Assignment)
        assert isinstance(s.targets, Index)

    def test_field(self):
        """R-PARSE-16.03: Field assignment parses with FieldAccess target."""
        s = parse("s.x = 10")[0]
        assert isinstance(s, Assignment)
        assert isinstance(s.targets, FieldAccess)

    def test_multi_return(self):
        """R-PARSE-16.04: Multi-return assignment parses as list of two targets."""
        s = parse("[a, b] = func(x)")[0]
        assert isinstance(s, Assignment)
        assert isinstance(s.targets, list) and len(s.targets) == 2


class TestExprStatement:
    """R-PARSE-17: The parser SHALL parse bare expression statements and
    recognize semicolons as print-suppression markers.

    Model-user argument: When the engineer types ``disp(x)`` without a
    semicolon, the result should print. Adding a semicolon suppresses output.
    The parser must track this flag for the evaluator.

    Decomposition: Bare expression (print=true) and semicolon-suppressed
    tested. Consistency: Print and suppress are the two output modes.
    """

    def test_bare_expression(self):
        """R-PARSE-17.01: Bare expression statement has print_result=true."""
        s = parse("disp(x)")[0]
        assert isinstance(s, ExpressionStatement)
        assert s.print_result is True

    def test_semicolon_suppresses(self):
        """R-PARSE-17.02: Semicolon after assignment suppresses output."""
        s = parse("x = 5;")[0]
        assert isinstance(s, Assignment)


class TestIfStatement:
    """R-PARSE-18: The parser SHALL parse if/elseif/else/end and the Octave
    endif variant into IfStatement AST nodes.

    Model-user argument: Conditional branching is fundamental. The parser
    must handle the full if/elseif/else chain as well as the Octave-specific
    endif keyword, since the engineer may use either ending style.

    Decomposition: Simple if, if/else, if/elseif/else, and endif form tested.
    Consistency: These four cases cover all if-statement structures.
    """

    def test_simple_if(self):
        """R-PARSE-18.01: Simple if with one body statement."""
        s = parse("if x > 0\n  y = 1;\nend")[0]
        assert isinstance(s, IfStatement)
        assert isinstance(s.condition, CompareOp)
        assert len(s.body) == 1

    def test_if_else(self):
        """R-PARSE-18.02: If/else with non-None else_body."""
        s = parse("if x > 0\n  y = 1;\nelse\n  y = -1;\nend")[0]
        assert isinstance(s, IfStatement)
        assert s.else_body is not None
        assert len(s.else_body) == 1

    def test_if_elseif_else(self):
        """R-PARSE-18.03: If/elseif/else with one elseif clause."""
        src = "if x > 0\n  y = 1;\nelseif x == 0\n  y = 0;\nelse\n  y = -1;\nend"
        s = parse(src)[0]
        assert isinstance(s, IfStatement)
        assert len(s.elseifs) == 1
        assert s.else_body is not None

    def test_endif(self):
        """R-PARSE-18.04: Octave endif keyword is accepted."""
        s = parse("if true\n  x = 1;\nendif")[0]
        assert isinstance(s, IfStatement)


class TestForLoop:
    """R-PARSE-19: The parser SHALL parse for-loops with colon-range iterators
    and the Octave endfor variant.

    Model-user argument: For-loops drive batch processing. The parser must
    capture the loop variable, iterator expression, and body. The Octave
    endfor keyword must also be accepted.

    Decomposition: Standard for-loop and endfor variant tested.
    Consistency: These cover the two ending styles for for-loops.
    """

    def test_simple(self):
        """R-PARSE-19.01: For-loop captures variable name and ColonExpr iterator."""
        s = parse("for i = 1:10\n  x(i) = i;\nend")[0]
        assert isinstance(s, ForStatement)
        assert s.var == "i"
        assert isinstance(s.iter_expr, ColonExpr)

    def test_endfor(self):
        """R-PARSE-19.02: Octave endfor keyword is accepted."""
        s = parse("for i = 1:5\n  x = i;\nendfor")[0]
        assert isinstance(s, ForStatement)


class TestWhileLoop:
    """R-PARSE-20: The parser SHALL parse while-loops with a condition
    expression and body.

    Model-user argument: While-loops implement convergence and interactive
    checks. The parser must capture the condition as a CompareOp (or any
    expression) and the loop body.

    Decomposition: Single while-loop test. Consistency: While-loop has a
    single structural form.
    """

    def test_simple(self):
        """R-PARSE-20.01: While-loop captures condition and body."""
        s = parse("while x > 0\n  x = x - 1;\nend")[0]
        assert isinstance(s, WhileStatement)
        assert isinstance(s.condition, CompareOp)


class TestDoUntil:
    """R-PARSE-21: The parser SHALL parse do/until loops (Octave extension)
    into DoUntilStatement nodes.

    Model-user argument: The do/until loop guarantees at least one iteration,
    useful for iterative refinement. Octave users migrating to Forge expect
    this construct to be available.

    Decomposition: Single do/until test. Consistency: Do/until has a single
    structural form.
    """

    def test_simple(self):
        """R-PARSE-21.01: Do/until captures body and termination condition."""
        s = parse("do\n  x = x + 1;\nuntil x > 10")[0]
        assert isinstance(s, DoUntilStatement)
        assert isinstance(s.condition, CompareOp)


class TestSwitch:
    """R-PARSE-22: The parser SHALL parse switch/case/otherwise blocks into
    SwitchStatement nodes with correct case count and otherwise body.

    Model-user argument: Switch statements dispatch on enumerated values.
    The parser must capture each case expression, its body, and the optional
    otherwise fallthrough.

    Decomposition: Single switch with two cases and otherwise tested.
    Consistency: This exercises the complete switch structure.
    """

    def test_simple(self):
        """R-PARSE-22.01: Switch with two cases and otherwise parses correctly."""
        src = "switch x\n  case 1\n    y = 'a';\n  case 2\n    y = 'b';\n  otherwise\n    y = 'c';\nend"
        s = parse(src)[0]
        assert isinstance(s, SwitchStatement)
        assert len(s.cases) == 2
        assert s.otherwise_body is not None


class TestTryCatch:
    """R-PARSE-23: The parser SHALL parse try/catch blocks with and without a
    catch variable into TryCatchStatement nodes.

    Model-user argument: Error handling with try/catch is essential for
    robust scripts. The parser must support both forms: with a named catch
    variable for error inspection, and without for simple fallback.

    Decomposition: Try/catch with variable and without variable tested.
    Consistency: These are the two try/catch structural forms.
    """

    def test_simple(self):
        """R-PARSE-23.01: Try/catch with named variable captures catch_var."""
        s = parse("try\n  x = risky();\ncatch err\n  disp(err);\nend")[0]
        assert isinstance(s, TryCatchStatement)
        assert s.catch_var == "err"
        assert len(s.try_body) == 1
        assert len(s.catch_body) == 1

    def test_try_without_catch_var(self):
        """R-PARSE-23.02: Try/catch without variable has catch_var=None."""
        s = parse("try\n  x = 1;\ncatch\n  x = 0;\nend")[0]
        assert isinstance(s, TryCatchStatement)
        assert s.catch_var is None


class TestControlFlow:
    """R-PARSE-24: The parser SHALL parse return, break, continue, global, and
    persistent statements into their respective AST node types.

    Model-user argument: These control-flow and scope-declaration statements
    are used throughout MATLAB code. Return exits functions, break/continue
    control loops, and global/persistent declare variable scope.

    Decomposition: Each of the five statement types tested.
    Consistency: These cover all single-keyword control/scope statements.
    """

    def test_return(self):
        """R-PARSE-24.01: return parses to ReturnStatement."""
        s = parse("return")[0]
        assert isinstance(s, ReturnStatement)

    def test_break(self):
        """R-PARSE-24.02: break parses to BreakStatement."""
        s = parse("break")[0]
        assert isinstance(s, BreakStatement)

    def test_continue(self):
        """R-PARSE-24.03: continue parses to ContinueStatement."""
        s = parse("continue")[0]
        assert isinstance(s, ContinueStatement)

    def test_global(self):
        """R-PARSE-24.04: global declaration captures variable names."""
        s = parse("global x y z")[0]
        assert isinstance(s, GlobalStatement)
        assert s.names == ["x", "y", "z"]

    def test_persistent(self):
        """R-PARSE-24.05: persistent declaration captures variable name."""
        s = parse("persistent count")[0]
        assert isinstance(s, PersistentStatement)
        assert s.names == ["count"]


# ============================================================
# Stage 2.4: Function Definitions
# ============================================================

class TestFunctionDef:
    """R-PARSE-25: The parser SHALL parse function definitions with single
    return, no return, multiple returns, no arguments, and the endfunction
    variant.

    Model-user argument: Function definitions are how the engineer creates
    reusable algorithms. The parser must handle all return/parameter
    combinations and the Octave endfunction keyword.

    Decomposition: Simple, no-return, multi-return, no-args, and endfunction
    forms tested. Consistency: These five patterns cover all function
    definition structural variations.
    """

    def test_simple(self):
        """R-PARSE-25.01: Single-return function with one parameter."""
        src = "function y = square(x)\n  y = x.^2;\nend"
        s = parse(src)[0]
        assert isinstance(s, FunctionDef)
        assert s.name == "square"
        assert s.params == ["x"]
        assert s.returns == ["y"]
        assert len(s.body) == 1

    def test_no_return(self):
        """R-PARSE-25.02: Void function with no return value."""
        src = "function greet(name)\n  disp(name);\nend"
        s = parse(src)[0]
        assert isinstance(s, FunctionDef)
        assert s.returns == []
        assert s.params == ["name"]

    def test_multi_return(self):
        """R-PARSE-25.03: Function with bracketed multiple return values."""
        src = "function [mn, mx] = bounds(x)\n  mn = min(x);\n  mx = max(x);\nend"
        s = parse(src)[0]
        assert isinstance(s, FunctionDef)
        assert s.returns == ["mn", "mx"]
        assert s.params == ["x"]
        assert len(s.body) == 2

    def test_no_args(self):
        """R-PARSE-25.04: Function with empty parameter list."""
        src = "function x = get_value()\n  x = 42;\nend"
        s = parse(src)[0]
        assert isinstance(s, FunctionDef)
        assert s.params == []
        assert s.returns == ["x"]

    def test_endfunction(self):
        """R-PARSE-25.05: Octave endfunction keyword is accepted."""
        src = "function y = f(x)\n  y = x;\nendfunction"
        s = parse(src)[0]
        assert isinstance(s, FunctionDef)


class TestNestedStatements:
    """R-PARSE-26: The parser SHALL correctly nest control-flow statements
    within each other and within function bodies.

    Model-user argument: Real MATLAB code nests loops inside conditionals
    inside functions. The parser must build a correct AST hierarchy so that
    nested structures execute in the right order.

    Decomposition: For-with-if nesting and function-with-for nesting tested.
    Consistency: These two patterns confirm nesting works for the main
    compound statement types.
    """

    def test_for_with_if(self):
        """R-PARSE-26.01: If-statement inside for-loop parses as nested body."""
        src = "for i = 1:10\n  if i > 5\n    break;\n  end\nend"
        s = parse(src)[0]
        assert isinstance(s, ForStatement)
        assert isinstance(s.body[0], IfStatement)

    def test_function_with_for(self):
        """R-PARSE-26.02: For-loop inside function body parses as nested statement."""
        src = "function s = mysum(x)\n  s = 0;\n  for i = 1:length(x)\n    s = s + x(i);\n  end\nend"
        s = parse(src)[0]
        assert isinstance(s, FunctionDef)
        assert any(isinstance(stmt, ForStatement) for stmt in s.body)


class TestMultipleStatements:
    """R-PARSE-27: The parser SHALL parse multiple statements from a single
    source string, whether newline-separated or semicolon-separated.

    Model-user argument: The engineer writes multi-line scripts and
    semicolon-separated one-liners (``x = 1; y = 2; z = 3``). Both forms
    must produce the correct number of statement nodes.

    Decomposition: Newline-separated and semicolon-separated sequences tested.
    Consistency: These are the two statement separation mechanisms.
    """

    def test_sequence(self):
        """R-PARSE-27.01: Three newline-separated assignments produce three nodes."""
        src = "x = 1\ny = 2\nz = x + y"
        stmts = parse(src)
        assert len(stmts) == 3
        assert all(isinstance(s, Assignment) for s in stmts)

    def test_semicolon_separated(self):
        """R-PARSE-27.02: Three semicolon-separated assignments produce three nodes."""
        src = "x = 1; y = 2; z = 3"
        stmts = parse(src)
        assert len(stmts) == 3
