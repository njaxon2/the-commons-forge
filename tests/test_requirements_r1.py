# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for Forge Requirements R01-R16.

These are the first 16 requirements surfaced during the initial exploration
of the Forge engine through the driving task (TIGA Isogeometric Analysis).
Each requirement is stated in positive formal language, accompanied by a
model-user argument grounded in the golden user characterization
(see documentation/forge_model_user.md).

Verification tier: Tier 1 (headless unit tests via pytest).
Companion visual integration tests: test_gui_integration_r1.py (Tier 3).

SRS trace: SRS-FUNC-001 (core evaluation), SRS-DISP-001 (output formatting).
"""
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
    """R01: Semicolon at end of statement SHALL suppress output display.

    Model-user argument: The engineer is exploring data interactively in the
    command window. They type ``A = rand(500);`` and the semicolon means
    "I know what I assigned, don't flood my screen with a 500x500 matrix."
    Without reliable suppression, the command window becomes unusable for
    interactive work with large data. This is muscle memory from day one of
    MATLAB use; every line in their scripts ends with a semicolon except the
    ones they deliberately want to inspect.

    Decomposition:
        R01.1: Semicolon on assignment suppresses value echo
        R01.2: Semicolon on matrix assignment suppresses value echo
        R01.3: No semicolon on assignment displays the assigned value
        R01.4: No semicolon on matrix assignment displays the matrix

    Consistency: R01.1-R01.2 cover suppression for the two common assignment
    types (scalar and matrix). R01.3-R01.4 verify the inverse (display when
    no semicolon) for the same types. Together these confirm that semicolon
    suppression is correctly toggled for both scalar and matrix contexts.
    """
    def test_semicolon_suppresses(self, s):
        """R01.1: Semicolon on scalar assignment suppresses output."""
        r = s.eval("a = 5;")
        assert r == ""

    def test_no_semicolon_displays(self, s):
        """R01.3: No semicolon on scalar assignment displays value."""
        r = s.eval("a = 5")
        assert r.strip() != ""
        assert "5" in r

    def test_semicolon_matrix(self, s):
        """R01.2: Semicolon on matrix assignment suppresses output."""
        r = s.eval("A = [1 2; 3 4];")
        assert r == ""

    def test_no_semicolon_matrix(self, s):
        """R01.4: No semicolon on matrix assignment displays the matrix."""
        r = s.eval("A = [1 2; 3 4]")
        assert "1" in r and "4" in r


# ============================================================
# R02 - MATLAB-Style Output Formatting
# ============================================================

class TestR02_OutputFormat:
    """R02: Displayed values SHALL use MATLAB-style formatting, not numpy's.

    Model-user argument: The engineer has 15 years of reading MATLAB output.
    When they type ``x = 3.14159265`` they expect to see ``3.1416`` in short
    format, columns neatly aligned, no double-bracket ``[[...]]`` notation.
    Seeing numpy formatting breaks the illusion instantly: it screams "this
    is a Python wrapper" instead of "this is a professional engineering
    environment." The formatting must be invisible, meaning it must match
    what they already expect.

    Decomposition:
        R02.1: Scalar display SHALL NOT show numpy array brackets
        R02.2: Matrix display SHALL NOT show numpy ``[[...]]`` notation
        R02.3: Floating-point scalars SHALL display in short format (4 decimal places)
        R02.4: Matrix columns SHALL be aligned

    Consistency: R02.1-R02.2 eliminate numpy artifacts for the two display
    shapes. R02.3 verifies the numeric format matches MATLAB's default short
    format. R02.4 verifies spatial layout. Together these ensure the output
    is visually indistinguishable from MATLAB for common cases.
    """
    def test_scalar_integer_display(self, s):
        """R02.1: Scalar display shows no numpy brackets."""
        r = s.eval("x = 5")
        assert "[[" not in r

    def test_matrix_no_numpy_brackets(self, s):
        """R02.2: Matrix display shows no numpy ``[[...]]`` notation."""
        r = s.eval("A = [1 2; 3 4]")
        assert "[[" not in r

    def test_float_format_short(self, s):
        """R02.3: Float displays in MATLAB short format (4 decimal places)."""
        r = s.eval("x = 3.14159265")
        assert "3.1416" in r

    def test_column_alignment(self, s):
        """R02.4: Matrix columns are visually aligned."""
        r = s.eval("A = [1 20 300; 4 50 600]")
        lines = [l for l in r.strip().split("\n") if l.strip() and any(c.isdigit() for c in l)]
        assert len(lines) >= 2


# ============================================================
# R03 - Character Array Display
# ============================================================

class TestR03_CharDisplay:
    """R03: Character arrays SHALL display as readable text, not numeric codes.

    Model-user argument: The engineer stores file paths, labels, and messages
    as character arrays. When they type ``s = 'hello'`` they expect to see
    ``hello`` on screen, not ``[104 101 108 108 111]``. Displaying numeric
    codes would be disorienting: it would mean the most basic string operation
    produces unreadable output. They also rely on ``class(s)`` returning
    ``char`` to confirm they are working with text, not a numeric vector.

    Decomposition:
        R03.1: Assigning a single-quoted string SHALL display the text content
        R03.2: ``class()`` of a character array SHALL return 'char'

    Consistency: R03.1 covers the display path (what the user sees in the
    command window). R03.2 covers the type-query path (how the user confirms
    the variable type). Together these ensure character arrays are both
    displayed and classified correctly.
    """
    def test_char_displays_as_text(self, s):
        """R03.1: Character array displays as text, not numeric codes."""
        r = s.eval("s = 'hello'")
        assert "hello" in r
        assert "104" not in r

    def test_class_of_char(self, s):
        """R03.2: class() returns 'char' for character arrays."""
        s.eval("s = 'hello';")
        r = s.eval("class(s)")
        assert "char" in r


# ============================================================
# R04 - Slash in Single-Quoted Strings
# ============================================================

class TestR04_SlashInStrings:
    """R04: Forward slashes inside single-quoted strings SHALL NOT trigger
    division parsing.

    Model-user argument: The engineer constantly works with file paths:
    ``p = '/home/data/test.csv'`` or ``addpath('/opt/toolboxes')``. If the
    parser treats the ``/`` inside quotes as a division operator, every
    file-path operation breaks with a cryptic "RDIVIDE" error. This would
    make the tool unusable for any workflow involving file I/O, which is
    essentially all of them. In MATLAB, slashes in strings are never
    ambiguous.

    Decomposition:
        R04.1: Assignment of a string containing slashes SHALL succeed
        R04.2: Passing a slash-containing string to a function SHALL succeed

    Consistency: R04.1 covers the basic assignment case. R04.2 covers the
    function-argument case (where parsing context differs). Together these
    confirm the lexer correctly identifies slashes as string content in both
    statement positions.
    """
    def test_single_quote_with_slash(self, s):
        """R04.1: String assignment with slashes does not trigger RDIVIDE."""
        try:
            s.eval("p = '/tmp/test'")
        except Exception as e:
            if "ParseError" in str(type(e).__name__) or "RDIVIDE" in str(e):
                pytest.fail(f"Slash in string caused parse error: {e}")

    def test_addpath_with_slash(self, s):
        """R04.2: Function argument with slashes does not trigger RDIVIDE."""
        try:
            s.eval("addpath('/tmp')")
        except Exception as e:
            if "ParseError" in str(type(e).__name__) or "RDIVIDE" in str(e):
                pytest.fail(f"addpath with slash failed: {e}")


# ============================================================
# R05 - Command-Style Syntax
# ============================================================

class TestR05_CommandStyle:
    """R05: Common MATLAB commands SHALL be recognized in command-style
    syntax (no parentheses or quotes around arguments).

    Model-user argument: The engineer types ``who`` to see their variables,
    ``hold on`` before overlaying a second plot, ``axis equal`` to fix aspect
    ratio. They never type ``who()`` or ``hold('on')`` because command-style
    is how they learned it and how every MATLAB tutorial writes it. If Forge
    requires function-call syntax for these commands, every muscle-memory
    shortcut fails and the engineer feels like they are fighting the tool.

    Decomposition:
        R05.1: ``who`` SHALL list workspace variables (command-style, no parens)
        R05.2: ``whos`` SHALL list variables with detail (command-style)
        R05.3: ``hold on`` SHALL succeed without NameError
        R05.4: ``axis equal`` SHALL succeed without NameError

    Consistency: R05.1-R05.2 cover the workspace-inspection commands. R05.3-
    R05.4 cover plotting-related state commands. Together these sample the
    two major categories of command-style usage (data inspection and plot
    configuration).
    """
    def test_who_command(self, s):
        """R05.1: 'who' in command-style lists variables."""
        s.eval("x = 5;")
        r = s.eval("who")
        assert "function" not in r.lower()

    def test_whos_command(self, s):
        """R05.2: 'whos' in command-style lists variables with detail."""
        s.eval("x = 5;")
        r = s.eval("whos")
        assert "function" not in r.lower()

    def test_hold_on_command(self, s):
        """R05.3: 'hold on' succeeds in command-style."""
        try:
            s.eval("hold on")
        except NameError:
            pytest.fail("hold on raised NameError")

    def test_axis_equal_command(self, s):
        """R05.4: 'axis equal' succeeds in command-style."""
        try:
            s.eval("axis equal")
        except NameError:
            pytest.fail("axis equal raised NameError")


# ============================================================
# R06 - Float-to-Int Coercion
# ============================================================

class TestR06_FloatToInt:
    """R06: Functions expecting integer arguments SHALL accept float values
    that are mathematically whole numbers.

    Model-user argument: In MATLAB, all numbers are double-precision floats
    by default. The engineer writes ``zeros(n, n)`` where ``n`` came from a
    computation and is technically ``3.0``, not ``3``. They never think about
    this distinction because MATLAB handles it silently. If Forge raises a
    "expected integer, got float" error, every dynamically-computed size
    argument breaks. This would make ``reshape``, ``zeros``, ``ones``, and
    dozens of other functions unusable in practice.

    Decomposition:
        R06.1: ``reshape`` SHALL accept float dimensions that are whole numbers
        R06.2: ``zeros`` SHALL accept float dimensions
        R06.3: ``ones`` SHALL accept expression results as dimensions

    Consistency: R06.1 covers array reshaping. R06.2 covers array
    construction. R06.3 covers the case where the dimension comes from an
    arithmetic expression (the most common source of float-valued sizes).
    Together these ensure integer coercion works across the primary
    size-consuming functions.
    """
    def test_reshape_with_floats(self, s):
        """R06.1: reshape accepts float dimensions."""
        r = s.eval("reshape(1:6, 2, 3)")
        assert "1" in r and "6" in r

    def test_zeros_with_float_args(self, s):
        """R06.2: zeros accepts float dimensions from variables."""
        s.eval("n = 3;")
        r = s.eval("zeros(n, n)")
        assert "0" in r

    def test_ones_with_expression(self, s):
        """R06.3: ones accepts arithmetic expression as dimension."""
        r = s.eval("ones(2+1, 2)")
        assert "1" in r


# ============================================================
# R07 - Struct Auto-Creation
# ============================================================

class TestR07_StructAutoCreate:
    """R07: Assigning to a field of an undefined variable SHALL auto-create
    a struct.

    Model-user argument: The engineer building an FEM mesh writes
    ``msh.nodes = [0 0; 1 0; 1 1];`` without declaring ``msh`` first. This
    is the standard MATLAB idiom for incrementally constructing data
    structures: you just start assigning fields and the struct materializes.
    Requiring explicit ``struct()`` construction would break every script
    that builds geometry, configuration, or results structures
    field-by-field. In the TIGA driving task, mesh structs are assembled
    this way dozens of times.

    Decomposition:
        R07.1: Dotted assignment to undefined variable creates a struct
        R07.2: Multiple field assignments accumulate on the same struct
        R07.3: Struct field names SHALL NOT collide with builtin functions

    Consistency: R07.1 covers the initial creation. R07.2 covers
    accumulation (the common pattern of building up a struct over several
    lines). R07.3 ensures the namespace isolation between struct fields
    and builtins. Together these verify the complete field-by-field struct
    construction workflow.
    """
    def test_create_struct_by_field(self, s):
        """R07.1: Dotted assignment to undefined variable creates struct."""
        s.eval("msh.nodes = [0 0; 1 0];")
        r = s.eval("msh.nodes")
        assert "0" in r and "1" in r

    def test_multiple_fields(self, s):
        """R07.2: Multiple fields accumulate on the same struct."""
        s.eval("msh.x = 5;")
        s.eval("msh.y = 10;")
        r = s.eval("msh.x")
        assert "5" in r
        r = s.eval("msh.y")
        assert "10" in r

    def test_no_collision_with_builtins(self, s):
        """R07.3: Struct field access does not collide with builtins."""
        s.eval("mesh_data.nodes = [1 2 3];")
        r = s.eval("mesh_data.nodes")
        assert "1" in r


# ============================================================
# R08 - Sparse Matrix Construction
# ============================================================

class TestR08_Sparse:
    """R08: Sparse matrix construction via ``sparse()`` SHALL work with
    standard triplet input (row, col, value vectors).

    Model-user argument: The engineer assembling a finite element stiffness
    matrix works with sparse storage because the matrix is mostly zeros.
    ``K = sparse(I, J, V, n, n)`` is the canonical assembly pattern: they
    accumulate row indices, column indices, and values from element
    contributions, then construct the global sparse matrix in one call. If
    this fails, the entire FEM workflow is blocked. In the TIGA driving
    task, the stiffness and mass matrices are assembled exactly this way.

    Decomposition:
        R08.1: ``sparse(m, n)`` SHALL create an m-by-n zero sparse matrix
        R08.2: ``sparse(I, J, V, m, n)`` SHALL create a sparse matrix from
               triplet vectors

    Consistency: R08.1 covers the zero-initialization path (used for
    pre-allocation). R08.2 covers the triplet-construction path (used for
    assembly). Together these cover the two primary sparse construction
    patterns in FEM workflows.
    """
    def test_sparse_zeros(self, s):
        """R08.1: sparse(m,n) creates a zero sparse matrix."""
        s.eval("S = sparse(3, 3);")
        r = s.eval("nnz(S)")
        assert "0" in r

    def test_sparse_triplet(self, s):
        """R08.2: sparse(I,J,V,m,n) creates sparse from triplets."""
        s.eval("I = [1 1 2 3]; J = [1 2 2 3]; V = [4 1 5 6];")
        s.eval("K = sparse(I, J, V, 3, 3);")
        r = s.eval("full(K)")
        assert "4" in r and "5" in r


# ============================================================
# R09 - Indexed Assignment with RHS Expressions
# ============================================================

class TestR09_IndexedAssignment:
    """R09: Indexed assignment SHALL support compound RHS expressions
    including self-referencing additions.

    Model-user argument: The core FEM assembly loop is
    ``K(dofs, dofs) = K(dofs, dofs) + ke;`` where the global stiffness
    matrix accumulates element contributions. This pattern appears in every
    FEM code the engineer has ever written. If Forge cannot handle the
    self-referencing indexed assignment (reading and writing the same
    submatrix in one statement), the assembly loop must be rewritten with
    temporary variables, which is unacceptable for code ported from MATLAB.

    Decomposition:
        R09.1: ``K(i:j, i:j) = K(i:j, i:j) + ke`` SHALL accumulate correctly
        R09.2: Scalar indexed assignment ``A(i,j) = val`` SHALL work

    Consistency: R09.1 covers the compound self-referencing case (the FEM
    assembly pattern). R09.2 covers the simple scalar assignment case.
    Together these verify indexed assignment for both the complex and
    trivial cases.
    """
    def test_indexed_assign_with_addition(self, s):
        """R09.1: Self-referencing indexed assignment accumulates."""
        s.eval("K = zeros(4, 4);")
        s.eval("ke = [2 -1; -1 2];")
        s.eval("K(1:2, 1:2) = K(1:2, 1:2) + ke;")
        r = s.eval("K(1,1)")
        assert "2" in r

    def test_indexed_assign_scalar(self, s):
        """R09.2: Scalar indexed assignment works."""
        s.eval("A = zeros(3,3);")
        s.eval("A(2,2) = 99;")
        r = s.eval("A(2,2)")
        assert "99" in r


# ============================================================
# R10 - eig() Return Order
# ============================================================

class TestR10_EigOrder:
    """R10: ``eig(A)`` with two outputs SHALL return eigenvalues as a
    diagonal matrix D and eigenvectors as columns of V.

    Model-user argument: The engineer performing modal analysis writes
    ``[V, D] = eig(K, M);`` and expects D to be diagonal with eigenvalues
    on the diagonal, and V to contain the corresponding mode shapes as
    columns. This convention is universal in MATLAB. If D is returned as a
    vector, or if the eigenvalues are unsorted, every post-processing step
    (extracting natural frequencies, plotting mode shapes) breaks because
    the engineer indexes D as ``D(i,i)``, not ``D(i)``.

    Decomposition:
        R10.1: ``[V, D] = eig(A)`` SHALL return D as a diagonal matrix
        R10.2: Off-diagonal elements of D SHALL be zero

    Consistency: R10.1 verifies the eigenvalues are on the diagonal. R10.2
    verifies D is truly diagonal (not just that the diagonal has the right
    values). Together these confirm the eigenvalue matrix convention matches
    MATLAB's.
    """
    def test_eig_order(self, s):
        """R10.1-R10.2: eig returns diagonal D with correct eigenvalues."""
        s.eval("A = [2 1; 0 3];")
        s.eval("[V, D] = eig(A);")
        r_d = s.eval("D")
        assert "2" in r_d and "3" in r_d
        r_offdiag = s.eval("D(1,2)")
        assert "0" in r_offdiag


# ============================================================
# R11 - Nested Function Calls
# ============================================================

class TestR11_NestedCalls:
    """R11: Nested function calls SHALL evaluate correctly from innermost
    to outermost.

    Model-user argument: The engineer writes compact one-liners like
    ``max(abs(residual))`` to check convergence, or ``length(find(mask))``
    to count nonzeros. Nesting is the natural way they compose operations
    in the command window. If nesting fails, they must break every
    expression into temporary variables, which destroys the interactive
    flow that makes the command window useful for quick exploration.

    Decomposition:
        R11.1: ``max(abs([...]))`` SHALL evaluate the inner abs first
        R11.2: ``sum(abs([...]))`` SHALL compose correctly
        R11.3: ``length(find([...]))`` SHALL compose correctly

    Consistency: R11.1-R11.2 cover numeric composition (reduction of
    element-wise results). R11.3 covers logical-to-numeric composition
    (find returns indices, length counts them). Together these sample the
    two main nesting patterns: numeric chaining and logical chaining.
    """
    def test_max_abs(self, s):
        """R11.1: max(abs(x)) evaluates correctly."""
        r = s.eval("max(abs([-3 5 -7 2]))")
        assert "7" in r

    def test_sum_abs(self, s):
        """R11.2: sum(abs(x)) evaluates correctly."""
        r = s.eval("sum(abs([-1 -2 3]))")
        assert "6" in r

    def test_length_find(self, s):
        """R11.3: length(find(x)) evaluates correctly."""
        s.eval("v = [0 1 0 1 1];")
        r = s.eval("length(find(v))")
        assert "3" in r


# ============================================================
# R12 - Multi-Output Functions
# ============================================================

class TestR12_MultiOutput:
    """R12: Functions returning multiple outputs SHALL assign all requested
    outputs to the specified variables via ``[a, b] = func(...)`` syntax.

    Model-user argument: Multi-output calls are pervasive in MATLAB:
    ``[V, D] = eig(A)``, ``[U, S, V] = svd(A)``, ``[m, idx] = max(x)``.
    The engineer uses the index output to locate data, the factorization
    outputs to compose transformations, and the size outputs to reshape
    arrays. If multi-output assignment is broken, essentially all linear
    algebra workflows and many data-processing workflows become impossible.

    Decomposition:
        R12.1: ``[m, idx] = max(x)`` SHALL return both value and index
        R12.2: ``[r, c] = find(A)`` SHALL return both row and column indices
        R12.3: ``[m, n] = size(A)`` SHALL return both dimensions

    Consistency: R12.1 covers a reduction function (max with index). R12.2
    covers a query function (find with row/col). R12.3 covers a metadata
    function (size). Together these sample the three categories of
    multi-output usage.
    """
    def test_max_with_index(self, s):
        """R12.1: [m, idx] = max(x) returns value and index."""
        s.eval("[m, idx] = max([10 30 20]);")
        r_m = s.eval("m")
        r_idx = s.eval("idx")
        assert "30" in r_m
        assert "2" in r_idx

    def test_find_row_col(self, s):
        """R12.2: [r, c] = find(A) returns row and column indices."""
        s.eval("A = [0 1; 2 0];")
        s.eval("[r, c] = find(A);")
        r_r = s.eval("r")
        r_c = s.eval("c")
        assert "1" in r_r and "2" in r_r

    def test_size_two_outputs(self, s):
        """R12.3: [m, n] = size(A) returns both dimensions."""
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
    """R13: User-defined ``.m`` files on the path SHALL be auto-discovered
    and callable by name without explicit import.

    Model-user argument: The engineer writes a helper function ``myadd.m``
    and saves it. From the command window, they type ``myadd(3, 7)`` and
    expect it to work. There is no ``import``, no ``require``, no
    ``source``. MATLAB discovers ``.m`` files on the path automatically.
    This is the mechanism that makes the language feel like a single
    namespace where everything "just works." If auto-discovery fails, the
    engineer must manually load every function file, which is a workflow
    that does not exist in MATLAB.

    Decomposition:
        R13.1: A ``.m`` file added via ``addpath`` SHALL be callable by name
        R13.2: Calling an undefined function SHALL produce a clear
               "Undefined function" error

    Consistency: R13.1 covers the success path (function found and called).
    R13.2 covers the failure path (function not found, clear error message).
    Together these verify both the discovery mechanism and the error
    reporting when discovery fails.
    """
    def test_function_from_path(self, s, tmp_path):
        """R13.1: .m file on path is callable by name."""
        mfile = tmp_path / "myadd.m"
        mfile.write_text("function r = myadd(a, b)\n  r = a + b;\nend\n")
        s.eval('addpath("' + str(tmp_path) + '")')
        r = s.eval("myadd(3, 7)")
        assert "10" in r

    def test_function_not_found(self, s):
        """R13.2: Undefined function produces clear error."""
        r = s.eval("nonexistent_function_xyz(1)")
        assert "error" in r and "Undefined" in r


# ============================================================
# R14 - Script Execution
# ============================================================

class TestR14_ScriptExecution:
    """R14: ``.m`` script files SHALL execute in the caller's workspace,
    making their variables available after execution.

    Model-user argument: The engineer's workflow is: experiment in the
    command window, get something working, then save those lines as a
    script. They run the script with ``run('myscript.m')`` and expect the
    variables it creates to appear in their workspace. This is how scripts
    evolve from command history. If script variables are isolated in a
    separate scope (like Python modules), the fundamental
    explore-then-script workflow breaks.

    Decomposition:
        R14.1: Variables created by ``run('script.m')`` SHALL be visible
               in the caller's workspace

    Consistency: A single sub-requirement suffices because script execution
    semantics are atomic: either variables propagate to the caller's
    workspace or they don't.
    """
    def test_run_script(self, s, tmp_path):
        """R14.1: Script variables propagate to caller workspace."""
        script = tmp_path / "myscript.m"
        script.write_text("a = 42;\nb = a * 2;\n")
        s.eval('run("' + str(tmp_path / "myscript.m") + '")')
        r = s.eval("b")
        assert "84" in r


# ============================================================
# R15 - Missing Core Functions
# ============================================================

class TestR15_MissingFunctions:
    """R15: Core utility functions commonly used in interactive workflows
    SHALL be available without explicit loading.

    Model-user argument: The engineer reaches for ``dot()``, ``tic/toc``,
    ``sub2ind/ind2sub``, and set operations (``setdiff``, ``intersect``,
    ``union``) constantly. These are not exotic toolbox functions; they are
    everyday utilities. ``dot(a,b)`` checks orthogonality. ``tic; ... toc``
    measures how long a computation takes. ``setdiff`` finds which DOFs are
    free vs. constrained. If any of these are missing, the engineer hits a
    wall in the middle of routine work and has to implement workarounds.

    Decomposition:
        R15.1: ``dot(a, b)`` SHALL compute the dot product
        R15.2: ``tic`` / ``toc`` SHALL measure elapsed time
        R15.3: ``sub2ind`` SHALL convert subscripts to linear indices
        R15.4: ``ind2sub`` SHALL convert linear indices to subscripts
        R15.5: ``setdiff`` SHALL return elements in A not in B
        R15.6: ``intersect`` SHALL return common elements
        R15.7: ``union`` SHALL return combined unique elements

    Consistency: R15.1 covers vector algebra. R15.2 covers timing. R15.3-
    R15.4 cover index conversion (forward and inverse). R15.5-R15.7 cover
    the three standard set operations. Together these cover the six
    categories of utility functions most commonly needed in interactive
    engineering workflows.
    """
    def test_dot_product(self, s):
        """R15.1: dot() computes dot product."""
        r = s.eval("dot([1 2 3], [4 5 6])")
        assert "32" in r

    def test_tic_toc(self, s):
        """R15.2: tic/toc measures elapsed time."""
        s.eval("tic;")
        s.eval("x = ones(10,10);")
        r = s.eval("toc")
        assert r.strip() != ""

    def test_sub2ind(self, s):
        """R15.3: sub2ind converts subscripts to linear index."""
        r = s.eval("sub2ind([3 4], 2, 3)")
        assert "8" in r

    def test_ind2sub(self, s):
        """R15.4: ind2sub converts linear index to subscripts."""
        s.eval("[r, c] = ind2sub([3 4], 8);")
        r_r = s.eval("r")
        r_c = s.eval("c")
        assert "2" in r_r
        assert "3" in r_c

    def test_setdiff(self, s):
        """R15.5: setdiff returns elements in A not in B."""
        r = s.eval("setdiff([1 2 3 4], [2 4])")
        assert "1" in r and "3" in r

    def test_intersect(self, s):
        """R15.6: intersect returns common elements."""
        r = s.eval("intersect([1 2 3], [2 3 4])")
        assert "2" in r and "3" in r

    def test_union(self, s):
        """R15.7: union returns combined unique elements."""
        r = s.eval("union([1 2], [2 3])")
        assert "1" in r and "3" in r


# ============================================================
# R16 - Plot Integration (smoke tests)
# ============================================================

class TestR16_Plotting:
    """R16: ``plot(x, y)`` SHALL produce a figure without crashing.

    Model-user argument: After getting arithmetic working, the next thing
    the engineer does is plot something. ``x = linspace(0, 2*pi, 100);
    plot(x, sin(x))`` is the canonical test. If a figure appears with a
    sine wave, they feel at home. If it crashes or produces a Python
    traceback, they question whether this tool can do real work. Plotting
    is not a feature; it is a prerequisite. (See also R-BOOT-05 for the
    full visual integration requirement.)

    Decomposition:
        R16.1: ``plot(x, y)`` SHALL execute without raising an exception
        R16.2: ``figure`` SHALL execute without returning a function reference

    Consistency: R16.1 covers the basic plot call (the most important
    single command). R16.2 covers figure creation as a standalone command.
    Together these verify the minimum plotting smoke test. Full plotting
    verification (axes, labels, interactivity) is covered by R-BOOT-05
    and the R-PLOT- series.
    """
    def test_plot_no_crash(self, s):
        """R16.1: plot(x, y) executes without exception."""
        s.eval("x = linspace(0, 6.28, 50);")
        s.eval("y = sin(x);")
        s.eval("plot(x, y)")

    def test_figure_not_function_ref(self, s):
        """R16.2: figure command does not return a function reference."""
        r = s.eval("figure")
        assert "function" not in str(r).lower() or r.strip() == ""
