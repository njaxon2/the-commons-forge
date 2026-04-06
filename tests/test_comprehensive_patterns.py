# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Octave-compatible patterns verified manually.

Requirement R-CPAT: The engine SHALL correctly evaluate comprehensive,
end-to-end MATLAB/Octave patterns combining formatted output, nested
structs, multi-function scripts, matrix operations, curve fitting,
eigenanalysis, string processing, cell arrays, control flow, file I/O,
and dynamic field access in realistic sequences.

Model-user argument: An engineer migrating from MATLAB writes scripts
that combine multiple language features in a single file: defining helper
functions, building nested configuration structs, performing curve fits,
checking eigenvalue properties, and reading/writing CSV data. Each
pattern must produce identical results to Octave, because the engineer
validates Forge output against known MATLAB results before trusting it
for production calculations.

Decomposition:
  R-CPAT-01..05: Formatted output (sprintf/fprintf with vectors, strings)
  R-CPAT-06..08: Nested struct field access at 2, 3, and multi-field depth
  R-CPAT-09..12: Function patterns (multi-function, composition, 3-output, nargin)
  R-CPAT-13..17: Matrix operations (broadcast, logical assignment, grow, submatrix, row assign)
  R-CPAT-18..19: Curve fitting (polyfit/polyval roundtrip, linear system)
  R-CPAT-20..21: Eigenanalysis (reconstruction, trace/eigensum identity)
  R-CPAT-22..24: String processing (split/join, case, replace)
  R-CPAT-25..26: Cell arrays (function handles in cells, cellfun/map-reduce)
  R-CPAT-27..28: Control flow (while/break, do/until)
  R-CPAT-29..30: File I/O (CSV roundtrip, isdir)
  R-CPAT-31..32: Dynamic field access (read by variable, iteration over fields)

Consistency argument: These 32 tests span 11 distinct feature categories
that together cover the engineer's realistic script-writing patterns.
Each category is independently testable, and the categories collectively
cover data formatting, data structures, function definition, matrix
arithmetic, numerical methods, text processing, container types, control
flow, file operations, and reflection/metaprogramming.
"""
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
    """R-CPAT-01..04: sprintf SHALL apply format strings to vectors by
    recycling the format specifier across all elements, and handle
    string arguments correctly.

    Model-user argument: The engineer uses sprintf to generate formatted
    log lines from measurement vectors (e.g., sensor readings, timestamps).
    Vectorized formatting is essential because it avoids explicit loops
    and matches the concise MATLAB idiom the engineer already knows.

    Decomposition:
      R-CPAT-01: Integer vector formatting recycles %d across elements
      R-CPAT-02: Float vector formatting applies %.1f to each element
      R-CPAT-03: Paired vectors interleave into (x,y) format
      R-CPAT-04: String argument substitution with %s

    Consistency: These four tests cover scalar-format-on-vector (R-CPAT-01..02),
    multi-vector interleaving (R-CPAT-03), and string substitution (R-CPAT-04),
    spanning the sprintf usage patterns in typical scripts.
    """

    def test_vector_int(self, s):
        """R-CPAT-01: sprintf recycles %d across integer vector elements."""
        r = s.eval('sprintf("%d ", [1 2 3 4 5])')
        assert "1 2 3 4 5" in str(r)

    def test_vector_float(self, s):
        """R-CPAT-02: sprintf applies %.1f to each float vector element."""
        r = s.eval('sprintf("%.1f ", [1.5 2.5 3.5])')
        assert "1.5" in str(r) and "3.5" in str(r)

    def test_paired_vectors(self, s):
        """R-CPAT-03: sprintf interleaves paired vectors into (x,y) format."""
        r = s.eval('sprintf("(%d,%d) ", [1 2 3], [4 5 6])')
        assert "(1,4)" in str(r) and "(3,6)" in str(r)

    def test_string_arg(self, s):
        """R-CPAT-04: sprintf substitutes a string argument via %s."""
        r = s.eval('sprintf("Hello %s!", "world")')
        assert "Hello world!" in str(r)


class TestFprintfVectorized:
    """R-CPAT-05: fprintf SHALL format vector output to stdout.

    Model-user argument: The engineer uses fprintf(1, ...) to write
    formatted progress messages during long-running computations. The
    vector form avoids per-element loops.

    Decomposition:
      R-CPAT-05: fprintf(1, "%d ", vector) outputs formatted elements

    Consistency: Single test covers the stdout fprintf vector path.
    """

    def test_vector_to_stdout(self, s):
        """R-CPAT-05: fprintf formats vector elements to stdout."""
        r = s.eval('fprintf(1, "%d ", [10 20 30])')
        assert "10" in str(r) and "30" in str(r)


class TestNestedStructs:
    """R-CPAT-06..08: Nested struct field access SHALL support arbitrary
    depth levels and multiple fields per level.

    Model-user argument: The engineer builds configuration structs with
    hierarchical fields (config.db.host, config.app.name) to organize
    parameters for multi-component simulations. Dot-chain access at any
    depth must work without intermediate variable declarations.

    Decomposition:
      R-CPAT-06: Two-level struct access (st.inner.x)
      R-CPAT-07: Three-level struct access (a.b.c.d)
      R-CPAT-08: Multiple fields at same level (config.db.port)

    Consistency: Tests at depth 2, 3, and multi-field cover the struct
    nesting patterns used in realistic configuration management.
    """

    def test_two_level(self, s):
        """R-CPAT-06: Two-level nested struct field access works."""
        r = s.eval("st.inner.x = 42; st.inner.x")
        assert "42" in str(r)

    def test_three_level(self, s):
        """R-CPAT-07: Three-level nested struct field access works."""
        r = s.eval("a.b.c.d = 99; a.b.c.d")
        assert "99" in str(r)

    def test_multiple_fields(self, s):
        """R-CPAT-08: Multiple fields at same struct level are independent."""
        script = (
            'config.db.host = "localhost";\n'
            'config.db.port = 5432;\n'
            'config.app.name = "forge";\n'
        )
        s.eval(script)
        r = s.eval("config.db.port")
        assert "5432" in str(r)


class TestFunctionPatterns:
    """R-CPAT-09..12: Function definition patterns SHALL support multiple
    functions in one script, cross-function calls, multi-output returns,
    and nargin-based default arguments.

    Model-user argument: The engineer defines multiple helper functions at
    the top of a script, calls them from each other, and expects multi-output
    returns (e.g., [mn, mx, rng] = stats3(x)). nargin-based defaults let
    the engineer write flexible interfaces without overloading.

    Decomposition:
      R-CPAT-09: Multiple function definitions in one script, called from script body
      R-CPAT-10: Function calling another function defined in the same script
      R-CPAT-11: Three-output function returns all values correctly
      R-CPAT-12: nargin-based default argument fills missing parameter

    Consistency: These four tests cover the function definition and calling
    patterns that appear in every non-trivial MATLAB script.
    """

    def test_multiple_functions(self, s):
        """R-CPAT-09: Multiple functions defined and called in one script."""
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
        """R-CPAT-10: Function calls another function from the same script."""
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
        """R-CPAT-11: Three-output function returns all values correctly."""
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
        """R-CPAT-12: nargin-based default fills missing second argument."""
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
    """R-CPAT-13..17: Matrix operations SHALL correctly handle broadcasting,
    logical assignment, incremental growth, submatrix extraction, and row
    assignment.

    Model-user argument: The engineer builds data matrices incrementally in
    loops, extracts sub-regions for windowed analysis, and uses logical
    assignment to clamp outliers. These operations are performed hundreds of
    times in a typical analysis script.

    Decomposition:
      R-CPAT-13: Row + column vector broadcasting produces correct shape
      R-CPAT-14: Logical assignment zeros elements matching a condition
      R-CPAT-15: Array growth via horizontal concatenation in a loop
      R-CPAT-16: Submatrix extraction returns correct shape and values
      R-CPAT-17: Row assignment into a pre-allocated matrix

    Consistency: These five tests cover the matrix construction (grow, assign),
    extraction (submatrix), transformation (logical assignment), and
    arithmetic (broadcast) patterns that appear in data processing workflows.
    """

    def test_broadcast(self, s):
        """R-CPAT-13: Row + column vector broadcasting produces 3x3 matrix."""
        s.eval("A = [1 2 3]; B = [10; 20; 30]; C = A + B")
        data = get_array(s, "C")
        assert data.shape == (3, 3)

    def test_logical_assignment(self, s):
        """R-CPAT-14: A(A > 3) = 0 zeros elements exceeding threshold."""
        s.eval("A = [1 2 3 4 5]; A(A > 3) = 0")
        data = get_array(s, "A")
        assert list(data.flat) == [1.0, 2.0, 3.0, 0.0, 0.0]

    def test_grow_array(self, s):
        """R-CPAT-15: Array growth via concatenation accumulates squares."""
        s.eval("A = []; for i = 1:5; A = [A, i^2]; end")
        data = get_array(s, "A")
        assert list(data.flat) == [1.0, 4.0, 9.0, 16.0, 25.0]

    def test_submatrix(self, s):
        """R-CPAT-16: A(2:3, 1:2) extracts correct 2x2 submatrix."""
        s.eval("A = [1 2 3; 4 5 6; 7 8 9]; B = A(2:3, 1:2)")
        data = get_array(s, "B")
        assert data.shape == (2, 2)
        assert float(data[0, 0]) == 4.0

    def test_row_assignment(self, s):
        """R-CPAT-17: Row assignment into pre-allocated zeros matrix."""
        s.eval("A = zeros(3); A(2,:) = [4 5 6]")
        data = get_array(s, "A")
        assert float(data[1, 0]) == 4.0
        assert float(data[1, 2]) == 6.0


class TestCurveFitting:
    """R-CPAT-18..19: Curve fitting and linear system solving SHALL
    produce residuals below machine precision.

    Model-user argument: The engineer fits polynomial models to
    experimental data and solves linear systems for calibration
    parameters. If polyfit/polyval or A\\b produce even small residuals,
    the calibration becomes unreliable.

    Decomposition:
      R-CPAT-18: polyfit/polyval roundtrip on noise-free quadratic
      R-CPAT-19: A\\b linear system residual below 1e-10

    Consistency: Polynomial fitting (R-CPAT-18) and direct linear
    solving (R-CPAT-19) are the two primary curve fitting approaches.
    """

    def test_polyfit_polyval(self, s):
        """R-CPAT-18: polyfit/polyval roundtrip on noise-free data has zero error."""
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
        """R-CPAT-19: A\\b residual below machine precision."""
        script = "A = [3 1; 1 2]; b = [9; 8]; x = A \\ b; residual = norm(A * x - b);"
        s.eval(script)
        assert get_val(s, "residual") < 1e-10


class TestEigenAnalysis:
    """R-CPAT-20..21: Eigenanalysis SHALL satisfy algebraic identities
    (A*V = V*D, trace(A) = sum(eigenvalues)).

    Model-user argument: The engineer uses eigenvalue decomposition for
    stability analysis and modal analysis. The reconstruction identity
    A*V = V*D is how the engineer verifies correctness. The trace/eigensum
    identity is a quick sanity check for symmetric matrices.

    Decomposition:
      R-CPAT-20: Reconstruction error norm(A*V - V*D) below 1e-12
      R-CPAT-21: abs(trace(A) - sum(eigenvalues)) below 1e-10

    Consistency: Reconstruction (R-CPAT-20) tests the eigenvector/eigenvalue
    pair, and trace identity (R-CPAT-21) tests eigenvalue accuracy
    independently, covering both verification approaches.
    """

    def test_symmetric_eigenvalues(self, s):
        """R-CPAT-20: Eigenvector reconstruction error below 1e-12."""
        script = (
            "A = [2 1; 1 2];\n"
            "[V, D] = eig(A);\n"
            'reconstruction_error = norm(A*V - V*D, "fro");\n'
        )
        s.eval(script)
        assert get_val(s, "reconstruction_error") < 1e-12

    def test_trace_equals_eigensum(self, s):
        """R-CPAT-21: Trace equals sum of eigenvalues within 1e-10."""
        script = (
            "A = [4 1 0; 1 3 1; 0 1 2];\n"
            "[V, D] = eig(A);\n"
            "eigenvalues = diag(D);\n"
            "trace_err = abs(trace(A) - sum(eigenvalues));\n"
        )
        s.eval(script)
        assert get_val(s, "trace_err") < 1e-10


class TestStringProcessing:
    """R-CPAT-22..24: String processing builtins SHALL split/join, change
    case, and perform substitutions correctly.

    Model-user argument: The engineer parses delimited sensor log files
    with strsplit, normalizes identifiers with upper/lower, and cleans
    data labels with strrep. Incorrect string handling corrupts the
    data pipeline before any numerical analysis begins.

    Decomposition:
      R-CPAT-22: strsplit then strjoin roundtrip with different delimiter
      R-CPAT-23: upper and lower case transformations
      R-CPAT-24: strrep substring replacement

    Consistency: Split/join (R-CPAT-22), case (R-CPAT-23), and replace
    (R-CPAT-24) cover the three string transformation categories.
    """

    def test_strsplit_join(self, s):
        """R-CPAT-22: strsplit then strjoin with different delimiter."""
        r = s.eval('strjoin(strsplit("hello world", " "), "-")')
        assert "hello-world" in str(r)

    def test_upper_lower(self, s):
        """R-CPAT-23: upper and lower case transformations."""
        r1 = s.eval('upper("hello")')
        assert "HELLO" in str(r1)
        r2 = s.eval('lower("HELLO")')
        assert "hello" in str(r2)

    def test_strrep(self, s):
        """R-CPAT-24: strrep replaces substring correctly."""
        r = s.eval('strrep("hello world", "world", "there")')
        assert "hello there" in str(r)


class TestCellArrays:
    """R-CPAT-25..26: Cell arrays SHALL support function handle storage
    and cellfun-based map-reduce patterns.

    Model-user argument: The engineer stores operation pipelines as cell
    arrays of function handles (e.g., ops = {@plus, @minus, @times})
    and uses cellfun to apply transformations uniformly across data
    partitions. Both patterns are idiomatic MATLAB for batch processing.

    Decomposition:
      R-CPAT-25: Cell array of function handles, invoked via ops{k}(a,b)
      R-CPAT-26: cellfun(@length, data) computes lengths, sum aggregates

    Consistency: Handle invocation (R-CPAT-25) and cellfun map-reduce
    (R-CPAT-26) are the two cell-array usage patterns for computation.
    """

    def test_cell_of_handles(self, s):
        """R-CPAT-25: Function handle stored in cell array is callable."""
        script = (
            "ops = {@(a,b) a+b, @(a,b) a-b, @(a,b) a.*b};\n"
            "result = ops{2}(10, 3);\n"
        )
        s.eval(script)
        assert get_val(s, "result") == 7.0

    def test_map_reduce(self, s):
        """R-CPAT-26: cellfun(@length, data) computes element lengths."""
        script = (
            "data = {[1 2 3], [4 5], [6 7 8 9]};\n"
            "lengths = cellfun(@length, data);\n"
            "total = sum(lengths);\n"
        )
        s.eval(script)
        assert get_val(s, "total") == 9.0


class TestControlFlow:
    """R-CPAT-27..28: Control flow constructs SHALL support while/break
    and do/until loops.

    Model-user argument: The engineer uses while/break for convergence
    loops (iterate until residual is small enough) and do/until for
    loops that must execute at least once (e.g., user input validation
    in batch scripts).

    Decomposition:
      R-CPAT-27: while true with break exits at n=10
      R-CPAT-28: do/until executes body then checks condition

    Consistency: while/break (R-CPAT-27) and do/until (R-CPAT-28) are
    the two conditional-exit loop constructs in the M-language.
    """

    def test_while_break(self, s):
        """R-CPAT-27: while/break exits when counter reaches 10."""
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
        """R-CPAT-28: do/until executes body then checks exit condition."""
        s.eval("x = 0; do; x = x + 1; until (x >= 5)")
        assert get_val(s, "x") == 5.0


class TestFileIO:
    """R-CPAT-29..30: File I/O builtins SHALL read and write CSV files
    and query filesystem properties.

    Model-user argument: The engineer exports computation results to CSV
    for sharing with colleagues who use Excel, and imports CSV data files
    from measurement instruments. isdir checks are used to validate paths
    before batch processing.

    Decomposition:
      R-CPAT-29: csvwrite then csvread roundtrip preserves shape and values
      R-CPAT-30: isdir("/tmp") returns true on a known directory

    Consistency: CSV roundtrip (R-CPAT-29) covers data persistence, and
    isdir (R-CPAT-30) covers filesystem query, the two file I/O categories.
    """

    def test_csv_roundtrip(self, s):
        """R-CPAT-29: csvwrite/csvread roundtrip preserves matrix shape."""
        script = (
            "data = [1 2 3; 4 5 6];\n"
            'csvwrite("/tmp/forge_ctest.csv", data);\n'
            'result = csvread("/tmp/forge_ctest.csv");\n'
        )
        s.eval(script)
        data = get_array(s, "result")
        assert data.shape[0] == 2 and data.shape[1] == 3

    def test_isdir(self, s):
        """R-CPAT-30: isdir returns true for an existing directory."""
        r = s.eval('isdir("/tmp")')
        assert "1" in str(r)


class TestDynamicField:
    """R-CPAT-31..32: Dynamic field access via st.(varname) SHALL read
    struct fields by variable name and support iteration over fieldnames.

    Model-user argument: The engineer iterates over struct fields to
    aggregate sensor channels without hard-coding field names (e.g.,
    for i = 1:length(fields); total = total + st.(fields{i}); end).
    This reflection pattern is essential for generic data processing
    functions that work on any struct layout.

    Decomposition:
      R-CPAT-31: st.(fn) reads the field named by variable fn
      R-CPAT-32: Iteration over fieldnames with dynamic access sums all fields

    Consistency: Single-field dynamic read (R-CPAT-31) and loop-based
    iteration (R-CPAT-32) cover the two dynamic field access patterns.
    """

    def test_dynamic_field_read(self, s):
        """R-CPAT-31: st.(fn) reads the field named by variable fn."""
        script = (
            'st = struct("x", 1, "y", 2, "z", 3);\n'
            'fn = "y";\n'
            "val = st.(fn);\n"
        )
        s.eval(script)
        assert get_val(s, "val") == 2.0

    def test_struct_iteration(self, s):
        """R-CPAT-32: Iteration over fieldnames with dynamic access sums all fields."""
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
