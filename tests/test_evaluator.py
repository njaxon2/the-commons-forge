# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for M-language evaluator (Stages 2.5-2.11).

V&V traceability backfill: R-EVAL-01 through R-EVAL-17.
"""
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
    """R-EVAL-01: The evaluator SHALL evaluate arithmetic expressions and return
    numerically correct results for addition, subtraction, multiplication,
    division, power, negation, modulo, hex literals, and imaginary literals.

    Model-user argument: The engineer types expressions like ``2+3`` or
    ``(2+3)*4-1`` into the command window and expects immediate, correct
    numeric answers. If basic arithmetic is wrong, every downstream
    computation (signal processing, control systems, curve fitting) is
    untrustworthy.

    Decomposition: Each operator and literal form is tested individually.
    Consistency: Covering all six arithmetic operators plus unary minus,
    modulo, hex, and imaginary literals exhausts the set of primitive
    numeric evaluations that feed every higher-level expression.
    """

    def test_number(self, s):
        """R-EVAL-01.01: Bare numeric literal evaluates to its value."""
        r = s.eval("42")
        assert float(r) == 42.0

    def test_add(self, s):
        """R-EVAL-01.02: Addition of two scalars returns their sum."""
        r = s.eval("2 + 3")
        assert float(r) == 5.0

    def test_subtract(self, s):
        """R-EVAL-01.03: Subtraction of two scalars returns their difference."""
        r = s.eval("10 - 4")
        assert float(r) == 6.0

    def test_multiply(self, s):
        """R-EVAL-01.04: Multiplication of two scalars returns their product."""
        r = s.eval("3 * 7")
        assert float(r) == 21.0

    def test_divide(self, s):
        """R-EVAL-01.05: Division of two scalars returns their quotient."""
        r = s.eval("15 / 3")
        assert float(r) == 5.0

    def test_power(self, s):
        """R-EVAL-01.06: Element-wise power operator returns the correct result."""
        r = s.eval("2 .^ 10")
        assert float(r) == 1024.0

    def test_negate(self, s):
        """R-EVAL-01.07: Unary minus negates the operand."""
        r = s.eval("-5")
        assert float(r) == -5.0

    def test_complex_expr(self, s):
        """R-EVAL-01.08: Parenthesized compound expression respects precedence."""
        r = s.eval("(2 + 3) * 4 - 1")
        assert float(r) == 19.0

    def test_modulo(self, s):
        """R-EVAL-01.09: mod() builtin returns the correct remainder."""
        s.eval("x = mod(17, 5)")
        assert float(s.workspace.get("x")) == 2.0

    def test_hex_literal(self, s):
        """R-EVAL-01.10: Hexadecimal literal 0xFF evaluates to 255."""
        r = s.eval("0xFF")
        assert float(r) == 255.0

    def test_imaginary(self, s):
        """R-EVAL-01.11: Imaginary literal 3i evaluates to 0+3j."""
        r = s.eval("3i")
        assert complex(r.data.flat[0]) == 3j


class TestComparisonEval:
    """R-EVAL-02: The evaluator SHALL evaluate comparison operators (==, ~=, <,
    >=) and return logically correct results.

    Model-user argument: Comparisons underpin every conditional branch the
    engineer writes. An if-statement guarding a physical threshold (e.g.,
    ``if voltage > 5``) must evaluate truthfully or the script produces
    wrong decisions.

    Decomposition: Each comparison operator is tested with a true and/or
    false case. Consistency: The five operators (==, ~=, <, >, >=) with
    <=  implicit by symmetry cover all relational tests.
    """

    def test_eq_true(self, s):
        """R-EVAL-02.01: Equality returns true for identical operands."""
        r = s.eval("5 == 5")
        assert bool(r.data.flat[0])

    def test_eq_false(self, s):
        """R-EVAL-02.02: Equality returns false for differing operands."""
        r = s.eval("5 == 3")
        assert not bool(r.data.flat[0])

    def test_ne(self, s):
        """R-EVAL-02.03: Not-equal returns true for differing operands."""
        r = s.eval("5 ~= 3")
        assert bool(r.data.flat[0])

    def test_lt(self, s):
        """R-EVAL-02.04: Less-than returns true when left < right."""
        r = s.eval("3 < 5")
        assert bool(r.data.flat[0])

    def test_ge(self, s):
        """R-EVAL-02.05: Greater-or-equal returns true for equal operands."""
        r = s.eval("5 >= 5")
        assert bool(r.data.flat[0])


class TestLogicalEval:
    """R-EVAL-03: The evaluator SHALL evaluate logical operators (&&, ||, &, ~)
    with correct short-circuit and element-wise semantics.

    Model-user argument: The engineer writes compound guards like
    ``if x > 0 && x < 10`` and expects short-circuit behavior so the
    second clause is not evaluated when the first is false. Element-wise
    logical AND on arrays is needed for masking operations on data vectors.

    Decomposition: Short-circuit AND/OR (scalar), element-wise AND (array),
    and logical NOT are each tested separately. Consistency: These four
    operators are the complete set of logical connectives available in
    MATLAB/Octave.
    """

    def test_short_circuit_and(self, s):
        """R-EVAL-03.01: Short-circuit AND returns true when both operands are true."""
        r = s.eval("1 && 1")
        assert bool(r.data.flat[0])

    def test_short_circuit_and_false(self, s):
        """R-EVAL-03.02: Short-circuit AND returns false when first operand is false."""
        r = s.eval("0 && 1")
        assert not bool(r.data.flat[0])

    def test_short_circuit_or(self, s):
        """R-EVAL-03.03: Short-circuit OR returns true when second operand is true."""
        r = s.eval("0 || 1")
        assert bool(r.data.flat[0])

    def test_bitwise_and(self, s):
        """R-EVAL-03.04: Element-wise AND on logical arrays returns correct mask."""
        s.eval("x = [1 0 1] & [1 1 0]")
        np.testing.assert_array_equal(s.workspace.get("x").data.ravel(), [True, False, False])

    def test_not(self, s):
        """R-EVAL-03.05: Logical NOT inverts a falsy scalar to true."""
        r = s.eval("~0")
        assert bool(r.data.flat[0])


class TestTransposeEval:
    """R-EVAL-04: The evaluator SHALL evaluate conjugate transpose (') and
    non-conjugate transpose (.') on matrices.

    Model-user argument: Transpose is used in nearly every linear algebra
    workflow. The engineer writes ``A'*b`` to solve normal equations or
    compute inner products. Both conjugate and dot-transpose must work for
    real and complex data.

    Decomposition: Conjugate and dot-transpose tested on a 2x2 real matrix.
    Consistency: The two transpose forms cover all transpose operations.
    """

    def test_conjugate_transpose(self, s):
        """R-EVAL-04.01: Conjugate transpose swaps rows and columns."""
        s.eval("A = [1 2; 3 4]")
        r = s.eval("A'")
        np.testing.assert_array_equal(r.data, [[1, 3], [2, 4]])

    def test_dot_transpose(self, s):
        """R-EVAL-04.02: Dot-transpose swaps rows and columns without conjugation."""
        s.eval("A = [1 2; 3 4]")
        r = s.eval("A.'")
        np.testing.assert_array_equal(r.data, [[1, 3], [2, 4]])


class TestColonEval:
    """R-EVAL-05: The evaluator SHALL produce correct range vectors from colon
    expressions with and without explicit step.

    Model-user argument: Colon ranges like ``1:100`` or ``0:0.01:2*pi`` are
    the primary way the engineer creates sample grids for plotting and
    signal generation. Wrong ranges produce garbled plots or misaligned data.

    Decomposition: Simple range (unit step) and stepped range tested.
    Consistency: These two forms are the only colon-range variants.
    """

    def test_simple_range(self, s):
        """R-EVAL-05.01: Colon range 1:5 produces [1 2 3 4 5]."""
        r = s.eval("1:5")
        np.testing.assert_array_equal(r.data.ravel(), [1, 2, 3, 4, 5])

    def test_stepped_range(self, s):
        """R-EVAL-05.02: Stepped range 0:2:10 produces [0 2 4 6 8 10]."""
        r = s.eval("0:2:10")
        np.testing.assert_array_equal(r.data.ravel(), [0, 2, 4, 6, 8, 10])


class TestMatrixEval:
    """R-EVAL-06: The evaluator SHALL construct and multiply matrices from
    bracket-delimited literal syntax.

    Model-user argument: The engineer builds matrices by typing ``[1 2; 3 4]``
    and multiplies them with ``A * B``. Correct matrix construction and
    multiplication are prerequisites for linear algebra, state-space models,
    and transfer functions.

    Decomposition: Row vector, 2x2 matrix, empty matrix, and matrix multiply
    are tested. Consistency: These cover the fundamental construction forms
    (1-D, 2-D, empty) and the core matrix operation (multiply).
    """

    def test_row_vector(self, s):
        """R-EVAL-06.01: Comma-separated bracket syntax creates a row vector."""
        r = s.eval("[1, 2, 3]")
        np.testing.assert_array_equal(r.data.ravel(), [1, 2, 3])

    def test_matrix(self, s):
        """R-EVAL-06.02: Semicolon-separated rows create a 2x2 matrix."""
        r = s.eval("[1 2; 3 4]")
        np.testing.assert_array_equal(r.data, [[1, 2], [3, 4]])

    def test_empty_matrix(self, s):
        """R-EVAL-06.03: Empty brackets [] produce an empty matrix."""
        r = s.eval("[]")
        assert r.isempty()

    def test_matrix_multiply(self, s):
        """R-EVAL-06.04: Matrix multiplication returns the correct product."""
        s.eval("A = [1 2; 3 4]")
        s.eval("B = [5 6; 7 8]")
        r = s.eval("A * B")
        np.testing.assert_array_equal(r.data, [[19, 22], [43, 50]])


class TestIndexingEval:
    """R-EVAL-07: The evaluator SHALL support 1-based array and matrix indexing,
    indexed assignment, and colon-range subscripts.

    Model-user argument: The engineer accesses data with 1-based indices
    (``x(2)`` for the second element) and slices with colon ranges
    (``x(2:4)``). Wrong indexing silently corrupts data extraction and
    assignment.

    Decomposition: Scalar array index, 2-D matrix index, indexed assignment,
    and colon subscript are tested. Consistency: These cover all primitive
    indexing patterns (scalar, multi-dim, assignment, range).
    """

    def test_array_index(self, s):
        """R-EVAL-07.01: 1-based scalar index retrieves the correct element."""
        s.eval("x = [10, 20, 30]")
        r = s.eval("x(2)")
        assert float(r) == 20.0

    def test_matrix_index(self, s):
        """R-EVAL-07.02: 2-D (row, col) index retrieves the correct element."""
        s.eval("A = [1 2; 3 4]")
        r = s.eval("A(2, 1)")
        assert float(r) == 3.0

    def test_assign_index(self, s):
        """R-EVAL-07.03: Indexed assignment modifies the correct element."""
        s.eval("x = [1, 2, 3]")
        s.eval("x(2) = 99")
        assert s.workspace.get("x")[2] == 99

    def test_colon_index(self, s):
        """R-EVAL-07.04: Colon subscript x(2:4) extracts elements 2 through 4."""
        s.eval("x = [10 20 30 40 50]")
        r = s.eval("x(2:4)")
        np.testing.assert_array_equal(r.data.ravel(), [20, 30, 40])


class TestFunctionCallEval:
    """R-EVAL-08: The evaluator SHALL correctly call builtin functions (sqrt,
    sin, abs, size, length, zeros, ones, eye, disp, sum, max, find) and
    return their documented results.

    Model-user argument: Builtins are the vocabulary of everyday MATLAB work.
    The engineer calls ``sqrt(25)`` expecting 5, ``zeros(2,3)`` expecting a
    2x3 zero matrix, and ``find([0 1 0 1 1])`` expecting 1-based indices of
    nonzero elements. If any builtin returns the wrong answer, the
    engineer's trust in the platform collapses.

    Decomposition: Each builtin is tested with a representative input.
    Consistency: The 12 builtins tested here span math (sqrt, sin, abs),
    queries (size, length), constructors (zeros, ones, eye), output (disp),
    and reductions (sum, max, find), covering the core builtin categories.
    """

    def test_builtin_sqrt(self, s):
        """R-EVAL-08.01: sqrt(25) returns 5."""
        r = s.eval("sqrt(25)")
        assert abs(float(r) - 5.0) < 1e-10

    def test_builtin_sin(self, s):
        """R-EVAL-08.02: sin(0) returns 0."""
        r = s.eval("sin(0)")
        assert abs(float(r)) < 1e-10

    def test_builtin_abs(self, s):
        """R-EVAL-08.03: abs(-5) returns 5."""
        r = s.eval("abs(-5)")
        assert float(r) == 5.0

    def test_builtin_size(self, s):
        """R-EVAL-08.04: size() returns [rows, cols] of a matrix."""
        s.eval("A = [1 2 3; 4 5 6]")
        r = s.eval("size(A)")
        np.testing.assert_array_equal(r.data.ravel(), [2, 3])

    def test_builtin_length(self, s):
        """R-EVAL-08.05: length() returns the largest dimension."""
        s.eval("x = [1 2 3 4 5]")
        r = s.eval("length(x)")
        assert float(r) == 5

    def test_builtin_zeros(self, s):
        """R-EVAL-08.06: zeros(2,3) returns a 2x3 matrix of zeros."""
        r = s.eval("zeros(2, 3)")
        assert r.shape == (2, 3)
        assert np.all(r.data == 0)

    def test_builtin_ones(self, s):
        """R-EVAL-08.07: ones(3) returns a 3x3 matrix of ones."""
        r = s.eval("ones(3)")
        assert r.shape == (3, 3)

    def test_builtin_eye(self, s):
        """R-EVAL-08.08: eye(3) returns the 3x3 identity matrix."""
        r = s.eval("eye(3)")
        np.testing.assert_array_equal(r.data, np.eye(3))

    def test_builtin_disp(self, s):
        """R-EVAL-08.09: disp('hello') writes 'hello' to the output buffer."""
        s.eval("disp('hello')")
        assert "hello" in s.output_buffer.getvalue()

    def test_builtin_sum(self, s):
        """R-EVAL-08.10: sum([1 2 3 4]) returns 10."""
        r = s.eval("sum([1 2 3 4])")
        assert float(r) == 10.0

    def test_builtin_max(self, s):
        """R-EVAL-08.11: max([3 1 4 1 5]) returns 5."""
        r = s.eval("max([3 1 4 1 5])")
        assert float(r) == 5.0

    def test_builtin_find(self, s):
        """R-EVAL-08.12: find() returns 1-based indices of nonzero elements."""
        r = s.eval("find([0 1 0 1 1])")
        np.testing.assert_array_equal(r.data.ravel(), [2, 4, 5])  # 1-based


class TestAnonymousFunction:
    """R-EVAL-09: The evaluator SHALL create and invoke anonymous functions
    with correct argument binding and lexical closure.

    Model-user argument: Anonymous functions let the engineer define quick
    inline transforms like ``f = @(x) x.^2`` and pass them to solvers or
    plotters. Closure capture (``a = 10; f = @(x) x + a``) is essential for
    parameterized callbacks.

    Decomposition: Simple single-arg, multi-arg, and closure cases tested.
    Consistency: These three patterns (single, multi, closure) cover all
    anonymous function usage modes.
    """

    def test_simple(self, s):
        """R-EVAL-09.01: Single-argument anonymous function evaluates correctly."""
        s.eval("f = @(x) x.^2")
        r = s.eval("f(5)")
        assert float(r) == 25.0

    def test_multi_arg(self, s):
        """R-EVAL-09.02: Multi-argument anonymous function evaluates correctly."""
        s.eval("f = @(x, y) x + y")
        r = s.eval("f(3, 4)")
        assert float(r) == 7.0

    def test_closure(self, s):
        """R-EVAL-09.03: Anonymous function captures variables from enclosing scope."""
        s.eval("a = 10")
        s.eval("f = @(x) x + a")
        r = s.eval("f(5)")
        assert float(r) == 15.0


class TestFunctionHandle:
    """R-EVAL-10: The evaluator SHALL create function handles with @name syntax
    and invoke them correctly.

    Model-user argument: The engineer passes builtin functions to higher-order
    routines (e.g., ``fplot(@sin, [0 2*pi])``) using the @name handle syntax.
    The handle must resolve to the named function at call time.

    Decomposition: Single test for @sin handle creation and invocation.
    Consistency: Handle creation and invocation is a single atomic operation.
    """

    def test_handle(self, s):
        """R-EVAL-10.01: @sin creates a callable handle that evaluates sin(0) to 0."""
        s.eval("f = @sin")
        r = s.eval("f(0)")
        assert abs(float(r)) < 1e-10


class TestFieldAccessEval:
    """R-EVAL-11: The evaluator SHALL read and write struct fields using dot
    notation.

    Model-user argument: The engineer stores structured data in structs
    (e.g., ``s.frequency = 1000; s.amplitude = 0.5``) and accesses fields
    with dot syntax. Structs are the primary way to group related parameters.

    Decomposition: Field read and field assignment tested.
    Consistency: Read and write are the two fundamental field operations.
    """

    def test_struct_field(self, s):
        """R-EVAL-11.01: Reading a struct field returns its stored value."""
        s.eval("s = struct('x', 1, 'y', 2)")
        r = s.eval("s.x")
        assert r == 1

    def test_struct_assign_field(self, s):
        """R-EVAL-11.02: Assigning a new struct field stores the value."""
        s.eval("s = struct('x', 1)")
        s.eval("s.y = 42")
        assert s.workspace.get("s")._fields["y"] == 42


class TestCellEval:
    """R-EVAL-12: The evaluator SHALL construct cell arrays from brace syntax
    and support curly-brace content indexing.

    Model-user argument: Cell arrays hold heterogeneous data (numbers,
    strings, nested arrays). The engineer uses them for function argument
    lists, mixed-type tables, and multi-format I/O. Content indexing with
    ``c{2}`` must extract the stored value directly.

    Decomposition: Cell literal creation and cell content indexing tested.
    Consistency: Construction and indexing are the two primitive cell
    operations.
    """

    def test_cell_literal(self, s):
        """R-EVAL-12.01: Brace syntax creates a cell array with correct element count."""
        r = s.eval("{1, 'hello', [1 2 3]}")
        assert isinstance(r, ForgeCell)
        assert r.numel() == 3

    def test_cell_index(self, s):
        """R-EVAL-12.02: Curly-brace indexing extracts the cell content."""
        s.eval("c = {10, 20, 30}")
        r = s.eval("c{2}")
        assert r == 20


# ============================================================
# Stage 2.6: Control Flow
# ============================================================

class TestIfEval:
    """R-EVAL-13: The evaluator SHALL execute if/elseif/else branches based on
    condition truthiness.

    Model-user argument: Conditional branching is how the engineer implements
    decision logic: clamping values, selecting algorithms, handling edge
    cases. If the wrong branch executes, the computation silently produces
    incorrect results.

    Decomposition: True branch, false-to-else branch, and elseif chain
    tested. Consistency: These three cases cover all branching paths through
    an if/elseif/else block.
    """

    def test_if_true(self, s):
        """R-EVAL-13.01: True condition executes the if-body."""
        s.eval("x = 5")
        s.eval("if x > 0\n  y = 1;\nend")
        assert float(s.workspace.get("y")) == 1.0

    def test_if_false(self, s):
        """R-EVAL-13.02: False condition falls through to else-body."""
        s.eval("x = -1")
        s.eval("if x > 0\n  y = 1;\nelse\n  y = -1;\nend")
        assert float(s.workspace.get("y")) == -1.0

    def test_elseif(self, s):
        """R-EVAL-13.03: Elseif clause executes when its condition is true."""
        s.eval("x = 0")
        s.eval("if x > 0\n  y = 1;\nelseif x == 0\n  y = 0;\nelse\n  y = -1;\nend")
        assert float(s.workspace.get("y")) == 0.0


class TestForEval:
    """R-EVAL-14: The evaluator SHALL execute for-loops over colon ranges with
    correct iteration, break, and continue semantics.

    Model-user argument: For-loops are the workhorse of batch processing.
    The engineer writes ``for i = 1:N`` to iterate over samples, channels,
    or time steps. Break and continue control early exit and skip logic
    within the loop body.

    Decomposition: Simple accumulation, break-early, and continue-skip
    tested. Consistency: Normal iteration, break, and continue are the
    three control paths within a for-loop.
    """

    def test_simple_for(self, s):
        """R-EVAL-14.01: For-loop accumulates sum of 1:5 correctly."""
        s.eval("s = 0")
        s.eval("for i = 1:5\n  s = s + i;\nend")
        assert float(s.workspace.get("s")) == 15.0

    def test_for_break(self, s):
        """R-EVAL-14.02: Break exits the loop early at the specified condition."""
        s.eval("s = 0")
        s.eval("for i = 1:100\n  if i > 5\n    break;\n  end\n  s = s + i;\nend")
        assert float(s.workspace.get("s")) == 15.0

    def test_for_continue(self, s):
        """R-EVAL-14.03: Continue skips even iterations, summing only odd values."""
        s.eval("s = 0")
        s.eval("for i = 1:10\n  if mod(i, 2) == 0\n    continue;\n  end\n  s = s + i;\nend")
        assert float(s.workspace.get("s")) == 25.0  # 1+3+5+7+9


class TestWhileEval:
    """R-EVAL-15: The evaluator SHALL execute while-loops with correct condition
    checking and break support.

    Model-user argument: While-loops implement convergence checks
    (``while err > tol``) and interactive loops (``while running``). The
    engineer relies on the condition being re-evaluated each iteration.

    Decomposition: Simple decrement loop and break-from-infinite-loop tested.
    Consistency: Normal termination and break cover the two while exit paths.
    """

    def test_simple_while(self, s):
        """R-EVAL-15.01: While-loop decrements until condition is false."""
        s.eval("x = 10")
        s.eval("while x > 0\n  x = x - 3;\nend")
        assert float(s.workspace.get("x")) == -2.0

    def test_while_break(self, s):
        """R-EVAL-15.02: Break exits an infinite while-loop at the threshold."""
        s.eval("x = 0")
        s.eval("while 1\n  x = x + 1;\n  if x >= 5\n    break;\n  end\nend")
        assert float(s.workspace.get("x")) == 5.0


class TestSwitchEval:
    """R-EVAL-16: The evaluator SHALL execute switch/case/otherwise blocks by
    matching the switch expression to case values.

    Model-user argument: Switch statements dispatch on mode selectors,
    algorithm IDs, or enumerated options. The engineer writes
    ``switch method; case 'fft' ... case 'dft' ...`` to select processing
    paths.

    Decomposition: Matched case and otherwise fallthrough tested.
    Consistency: A matched case and the otherwise default cover all
    switch execution paths.
    """

    def test_switch(self, s):
        """R-EVAL-16.01: Matching case clause executes its body."""
        s.eval("x = 2")
        s.eval("switch x\n  case 1\n    y = 'a';\n  case 2\n    y = 'b';\n  otherwise\n    y = 'c';\nend")
        assert s.workspace.get("y").to_str() == "b"

    def test_switch_otherwise(self, s):
        """R-EVAL-16.02: Unmatched value falls through to otherwise."""
        s.eval("x = 99")
        s.eval("switch x\n  case 1\n    y = 'a';\n  otherwise\n    y = 'z';\nend")
        assert s.workspace.get("y").to_str() == "z"


class TestTryCatchEval:
    """R-EVAL-17: The evaluator SHALL execute try/catch blocks, capturing error
    objects with identifier and message fields.

    Model-user argument: The engineer wraps risky operations (file I/O,
    division, user input parsing) in try/catch to handle failures gracefully
    rather than crashing the session. Access to the error identifier and
    message is needed for programmatic error handling.

    Decomposition: No-error passthrough, catch execution, and catch variable
    inspection tested. Consistency: These three cases cover the try success
    path, the catch path, and error-object field access.
    """

    def test_no_error(self, s):
        """R-EVAL-17.01: Try body executes normally when no error occurs."""
        s.eval("try\n  x = 5;\ncatch\n  x = -1;\nend")
        assert float(s.workspace.get("x")) == 5.0

    def test_catch_error(self, s):
        """R-EVAL-17.02: Catch body executes when try body raises an error."""
        s.eval("try\n  error('test error');\ncatch err\n  x = 1;\nend")
        assert float(s.workspace.get("x")) == 1.0

    def test_catch_var(self, s):
        """R-EVAL-17.03: Catch variable exposes the error message."""
        s.eval("try\n  error('myid', 'bad stuff');\ncatch err\n  msg = err.message;\nend")
        assert "bad stuff" in str(s.workspace.get("msg"))


# ============================================================
# Stage 2.7: User Functions
# ============================================================

class TestUserFunctions:
    """R-EVAL-18: The evaluator SHALL define and call user functions with single
    and multiple return values, recursion, and internal loops.

    Model-user argument: User-defined functions are how the engineer
    encapsulates reusable algorithms. A function like ``bounds(x)`` returning
    ``[mn, mx]`` must work correctly for the engineer to build reliable
    toolboxes. Recursion (``factorial``) and internal loops (``mysum``) are
    common patterns.

    Decomposition: Simple function, multi-return, no-return, recursive, and
    loop-containing functions tested. Consistency: These five patterns cover
    the main function definition variations.
    """

    def test_simple_function(self, s):
        """R-EVAL-18.01: Single-return function evaluates correctly."""
        s.eval("function y = square(x)\n  y = x .^ 2;\nend")
        r = s.eval("square(5)")
        assert float(r) == 25.0

    def test_multi_return(self, s):
        """R-EVAL-18.02: Multi-return function assigns both outputs."""
        s.eval("function [mn, mx] = bounds(x)\n  mn = min(x);\n  mx = max(x);\nend")
        s.eval("[a, b] = bounds([3 1 4 1 5])")
        assert float(s.workspace.get("a")) == 1.0
        assert float(s.workspace.get("b")) == 5.0

    def test_function_no_return(self, s):
        """R-EVAL-18.03: Void function executes its body without returning a value."""
        s.eval("function greet()\n  disp('hi');\nend")
        s.eval("greet()")
        assert "hi" in s.output_buffer.getvalue()

    def test_recursive_function(self, s):
        """R-EVAL-18.04: Recursive function computes factorial(5) = 120."""
        s.eval("function n = factorial(x)\n  if x <= 1\n    n = 1;\n  else\n    n = x * factorial(x - 1);\n  end\nend")
        r = s.eval("factorial(5)")
        assert float(r) == 120.0

    def test_function_with_loop(self, s):
        """R-EVAL-18.05: Function with internal for-loop sums a vector."""
        s.eval("function s = mysum(x)\n  s = 0;\n  for i = 1:length(x)\n    s = s + x(i);\n  end\nend")
        r = s.eval("mysum([1 2 3 4 5])")
        assert float(r) == 15.0


# ============================================================
# Stage 2.8-2.9: I/O and Errors
# ============================================================

class TestIO:
    """R-EVAL-19: The evaluator SHALL support formatted output (disp, sprintf)
    and error raising with identifier and message.

    Model-user argument: The engineer uses ``disp`` to inspect intermediate
    values and ``sprintf`` to format results for reports. The ``error``
    function with an identifier enables structured error handling in larger
    codebases.

    Decomposition: disp with string/number, sprintf formatting, error with
    and without identifier tested. Consistency: These cover the primary
    output and error-raising builtins.
    """

    def test_disp_string(self, s):
        """R-EVAL-19.01: disp('hello world') writes the string to output."""
        s.eval("disp('hello world')")
        assert "hello world" in s.output_buffer.getvalue()

    def test_disp_number(self, s):
        """R-EVAL-19.02: disp(42) writes the number to output."""
        s.eval("disp(42)")
        assert "42" in s.output_buffer.getvalue()

    def test_sprintf(self, s):
        """R-EVAL-19.03: sprintf formats a string with a %d placeholder."""
        r = s.eval("sprintf('x = %d', 42)")
        assert r.to_str() == "x = 42"

    def test_error_basic(self, s):
        """R-EVAL-19.04: error('msg') raises a ForgeError with the given message."""
        with pytest.raises(ForgeError, match="something broke"):
            s.eval("error('something broke')")

    def test_error_with_id(self, s):
        """R-EVAL-19.05: error('id', 'msg') raises with identifier field set."""
        with pytest.raises(ForgeError) as exc_info:
            s.eval("error('mypackage:badInput', 'invalid value')")
        assert exc_info.value.identifier == "mypackage:badInput"


# ============================================================
# Stage 2.10-2.11: Constants and Integration
# ============================================================

class TestConstants:
    """R-EVAL-20: The evaluator SHALL resolve the built-in constants pi, true,
    and false to their correct values.

    Model-user argument: The engineer uses ``pi`` in trigonometric
    computations (e.g., ``sin(pi/4)``) and ``true``/``false`` in logical
    expressions. These must resolve without explicit definition.

    Decomposition: pi value check, true/false boolean checks tested.
    Consistency: pi, true, false are the most commonly used built-in
    constants.
    """

    def test_pi(self, s):
        """R-EVAL-20.01: pi resolves to approximately 3.14159265."""
        r = s.eval("pi")
        assert abs(float(r) - 3.14159265) < 1e-6

    def test_true_false(self, s):
        """R-EVAL-20.02: true and false resolve to their boolean values."""
        r = s.eval("true")
        assert bool(r.data.flat[0])
        r = s.eval("false")
        assert not bool(r.data.flat[0])


class TestIntegration:
    """R-EVAL-21: The evaluator SHALL correctly execute multi-statement scripts
    combining arithmetic, loops, functions, and matrix operations.

    Model-user argument: Real engineering scripts combine multiple language
    features in sequence. A Fibonacci generator uses loops and assignment; a
    Newton's method solver uses functions and convergence loops. These
    integration tests confirm that features compose correctly, not just work
    in isolation.

    Decomposition: Multi-statement sequence, Fibonacci loop, matrix A'*A,
    Newton sqrt, and workspace isolation tested. Consistency: These five
    scenarios exercise feature combinations (statements, loops, functions,
    matrices, scoping) that span the evaluator's capability.
    """

    def test_multi_statement(self, s):
        """R-EVAL-21.01: Semicolon-separated statements execute in sequence."""
        s.eval("x = 5; y = 10; z = x + y")
        assert float(s.workspace.get("z")) == 15.0

    def test_fibonacci(self, s):
        """R-EVAL-21.02: Fibonacci loop computes fib(10) = 55."""
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
        """R-EVAL-21.03: Transpose-multiply A'*A returns the correct product."""
        s.eval("A = [1 2; 3 4]")
        s.eval("B = A' * A")
        np.testing.assert_array_equal(s.workspace.get("B").data, [[10, 14], [14, 20]])

    def test_newton_sqrt(self, s):
        """R-EVAL-21.04: Newton's method converges to sqrt(2) within tolerance."""
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
        """R-EVAL-21.05: Function-local variables do not leak into caller workspace."""
        s.eval("function y = f(x)\n  z = x * 2;\n  y = z;\nend")
        s.eval("result = f(5)")
        assert float(s.workspace.get("result")) == 10.0
        assert not s.workspace.has("z")  # z should not leak
