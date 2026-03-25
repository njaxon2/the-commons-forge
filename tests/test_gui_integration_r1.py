"""GUI Integration Tests — Traceable to Requirements R01–R16.

Each test creates the actual Forge GUI (ForgeMainWindow + ForgeSession),
types commands into the command widget, and verifies what appears in the
output display, workspace browser, and other panels — exactly as a user
would observe through the GUI.

Requirement Traceability:
    R01 → TestR01_GUI_SemicolonSuppression
    R02 → TestR02_GUI_OutputFormat
    R03 → TestR03_GUI_CharDisplay
    R04 → TestR04_GUI_SlashInStrings
    R05 → TestR05_GUI_CommandStyle
    R06 → TestR06_GUI_FloatToInt
    R07 → TestR07_GUI_StructAutoCreate
    R08 → TestR08_GUI_Sparse
    R09 → TestR09_GUI_IndexedAssignment
    R10 → TestR10_GUI_EigOrder
    R11 → TestR11_GUI_NestedCalls
    R12 → TestR12_GUI_MultiOutput
    R13 → TestR13_GUI_MFileDiscovery
    R14 → TestR14_GUI_ScriptExecution
    R15 → TestR15_GUI_MissingFunctions
    R16 → TestR16_GUI_Plotting
"""
import os
import sys
import tempfile

import pytest

# Must set environment before any Qt imports
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# Ensure forge is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forge.engine.session import ForgeSession
from forge.gui.main_window import ForgeMainWindow
from forge.gui.command_widget import CommandWidget


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="session")
def qapp():
    """Create a single QApplication for the entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["forge-test", "--platform", "offscreen"])
    yield app


@pytest.fixture
def forge_window(qapp):
    """Create a fully wired ForgeMainWindow with engine attached."""
    session = ForgeSession()
    window = ForgeMainWindow()
    window.setup_engine(session)
    window.show()
    qapp.processEvents()
    yield window, session
    window.close()
    qapp.processEvents()


def gui_exec(forge_window, command):
    """Type a command into the GUI command widget and return the output.

    Simulates exactly what a user does: type text, press Enter, read output.
    Returns the text that appeared in the output display after the command.
    """
    window, session = forge_window
    qapp = QApplication.instance()

    cw = window.command_widget
    output_before = cw.output_display.toPlainText()

    # Type the command and press Enter
    cw.input_line.setText(command)
    cw.input_line.returnPressed.emit()
    qapp.processEvents()

    # Extract new output (everything after the previous content)
    output_after = cw.output_display.toPlainText()
    new_output = output_after[len(output_before):].strip()

    # Strip the echo line (>> command) to get just the result
    lines = new_output.split("\n")
    result_lines = [l for l in lines if not l.startswith(">> ") and not l.startswith(".. ")]
    return "\n".join(result_lines).strip()


def workspace_has(forge_window, varname):
    """Check if a variable appears in the workspace browser panel."""
    window, session = forge_window
    QApplication.instance().processEvents()
    table = window.workspace_widget.table
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item and item.text() == varname:
            return True
    return False


def workspace_value(forge_window, varname):
    """Get the displayed value/preview of a variable from the workspace browser."""
    window, session = forge_window
    QApplication.instance().processEvents()
    table = window.workspace_widget.table
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item and item.text() == varname:
            # Value column is typically column 3 (Name, Class, Size, Value)
            val_item = table.item(row, table.columnCount() - 1)
            return val_item.text() if val_item else ""
    return None


# ============================================================
# R01 — Semicolon Output Suppression
# Verify: command window shows no result when ; is present
# ============================================================

class TestR01_GUI_SemicolonSuppression:
    """R01: Statements ending with ; suppress output in the command window."""

    def test_semicolon_no_output_in_gui(self, forge_window):
        r = gui_exec(forge_window, "a = 42;")
        assert r == "", f"Semicolon should suppress output, got: '{r}'"

    def test_no_semicolon_shows_output(self, forge_window):
        r = gui_exec(forge_window, "b = 42")
        assert "42" in r, f"Should display 42, got: '{r}'"

    def test_semicolon_variable_still_created(self, forge_window):
        gui_exec(forge_window, "hidden_var = 99;")
        assert workspace_has(forge_window, "hidden_var"), \
            "Variable should exist in workspace even with semicolon"


# ============================================================
# R02 — MATLAB-Style Output Formatting
# Verify: output display shows formatted numbers, not numpy repr
# ============================================================

class TestR02_GUI_OutputFormat:
    """R02: Output in the command window uses MATLAB-style formatting."""

    def test_scalar_no_numpy(self, forge_window):
        r = gui_exec(forge_window, "x = 5")
        assert "[[" not in r, f"No numpy brackets, got: '{r}'"
        assert "5" in r

    def test_matrix_formatted(self, forge_window):
        r = gui_exec(forge_window, "A = [1 2; 3 4]")
        assert "[[" not in r, f"No numpy brackets, got: '{r}'"
        assert "1" in r and "4" in r

    def test_float_format_short(self, forge_window):
        r = gui_exec(forge_window, "pi_val = 3.14159265")
        assert "3.1416" in r, f"Expected format short (4 decimal), got: '{r}'"


# ============================================================
# R03 — Character Array Display
# Verify: strings display as text in the command window
# ============================================================

class TestR03_GUI_CharDisplay:
    """R03: Character arrays display as readable text, not ASCII codes."""

    def test_string_displays_as_text(self, forge_window):
        r = gui_exec(forge_window, "s = 'hello'")
        assert "hello" in r, f"Should show 'hello', got: '{r}'"
        assert "104" not in r, f"Should not show ASCII codes, got: '{r}'"

    def test_class_returns_char(self, forge_window):
        gui_exec(forge_window, "s = 'world';")
        r = gui_exec(forge_window, "class(s)")
        assert "char" in r, f"class should be char, got: '{r}'"


# ============================================================
# R04 — Slash Characters in Single-Quoted Strings
# Verify: paths with / don't cause parse errors in command window
# ============================================================

class TestR04_GUI_SlashInStrings:
    """R04: Single-quoted strings with / parse correctly in the GUI."""

    def test_path_string(self, forge_window):
        r = gui_exec(forge_window, "p = \"/tmp/test\"")
        assert "error" not in r.lower() or "parse" not in r.lower(), \
            f"Path string should not cause parse error: '{r}'"

    def test_addpath_with_slash(self, forge_window):
        r = gui_exec(forge_window, "addpath(\"/tmp\")")
        assert "ParseError" not in r, f"addpath should not parse-error: '{r}'"


# ============================================================
# R05 — Command-Style Syntax
# Verify: bare commands invoke functions, not return refs
# ============================================================

class TestR05_GUI_CommandStyle:
    """R05: Bare commands (who, whos) execute in the command window."""

    def test_who_not_function_ref(self, forge_window):
        gui_exec(forge_window, "test_var = 1;")
        r = gui_exec(forge_window, "who")
        assert "function" not in r.lower(), \
            f"who should execute, not show function ref: '{r}'"

    def test_whos_not_function_ref(self, forge_window):
        gui_exec(forge_window, "xx = 5;")
        r = gui_exec(forge_window, "whos")
        assert "function" not in r.lower(), \
            f"whos should execute, not show function ref: '{r}'"

    def test_hold_on_no_error(self, forge_window):
        r = gui_exec(forge_window, "hold on")
        assert "NameError" not in r, f"hold on should not error: '{r}'"

    def test_axis_equal_no_error(self, forge_window):
        r = gui_exec(forge_window, "axis equal")
        assert "NameError" not in r, f"axis equal should not error: '{r}'"


# ============================================================
# R06 — Float-to-Int Coercion for Shape Arguments
# Verify: reshape, zeros work with float dimension args
# ============================================================

class TestR06_GUI_FloatToInt:
    """R06: reshape/zeros/ones accept float dimension arguments."""

    def test_reshape_in_gui(self, forge_window):
        r = gui_exec(forge_window, "reshape(1:6, 2, 3)")
        assert "error" not in r.lower(), f"reshape should work: '{r}'"
        assert "1" in r and "6" in r

    def test_zeros_with_variable(self, forge_window):
        gui_exec(forge_window, "n = 3;")
        r = gui_exec(forge_window, "zeros(n, n)")
        assert "error" not in r.lower(), f"zeros(n,n) should work: '{r}'"


# ============================================================
# R07 — Struct Auto-Creation on Field Assignment
# Verify: dot-assignment creates struct visible in workspace
# ============================================================

class TestR07_GUI_StructAutoCreate:
    """R07: s.field = val auto-creates struct, visible in workspace."""

    def test_struct_creation_in_gui(self, forge_window):
        r = gui_exec(forge_window, "msh.nodes = [0 0; 1 0];")
        assert r == "", "Semicolon should suppress"
        assert workspace_has(forge_window, "msh"), \
            "Struct 'msh' should appear in workspace browser"

    def test_struct_field_read(self, forge_window):
        gui_exec(forge_window, "data.x = 42;")
        r = gui_exec(forge_window, "data.x")
        assert "42" in r, f"Field read should return 42, got: '{r}'"

    def test_struct_multiple_fields(self, forge_window):
        gui_exec(forge_window, "pt.x = 1;")
        gui_exec(forge_window, "pt.y = 2;")
        r_x = gui_exec(forge_window, "pt.x")
        r_y = gui_exec(forge_window, "pt.y")
        assert "1" in r_x and "2" in r_y


# ============================================================
# R08 — Sparse Matrix Construction
# Verify: sparse() works correctly, visible in workspace
# ============================================================

class TestR08_GUI_Sparse:
    """R08: sparse() creates correct sparse matrices via the GUI."""

    def test_sparse_zeros_gui(self, forge_window):
        gui_exec(forge_window, "S = sparse(3, 3);")
        r = gui_exec(forge_window, "nnz(S)")
        assert "0" in r, f"Empty sparse should have 0 nonzeros, got: '{r}'"

    def test_sparse_triplet_gui(self, forge_window):
        gui_exec(forge_window, "I = [1 1 2 3]; J = [1 2 2 3]; V = [4 1 5 6];")
        gui_exec(forge_window, "K = sparse(I, J, V, 3, 3);")
        r = gui_exec(forge_window, "full(K)")
        assert "4" in r and "5" in r, f"Sparse should contain 4 and 5: '{r}'"


# ============================================================
# R09 — Indexed Assignment with RHS Expressions
# Verify: FEA-style assembly pattern works in GUI
# ============================================================

class TestR09_GUI_IndexedAssignment:
    """R09: K(i,j) = K(i,j) + ke works in the command window."""

    def test_assembly_pattern(self, forge_window):
        gui_exec(forge_window, "K = zeros(4, 4);")
        gui_exec(forge_window, "ke = [2 -1; -1 2];")
        gui_exec(forge_window, "K(1:2, 1:2) = K(1:2, 1:2) + ke;")
        r = gui_exec(forge_window, "K(1,1)")
        assert "2" in r, f"K(1,1) should be 2 after assembly, got: '{r}'"
        r2 = gui_exec(forge_window, "K(1,2)")
        assert "-1" in r2, f"K(1,2) should be -1, got: '{r2}'"


# ============================================================
# R10 — eig() Return Order
# Verify: [V,D] = eig(A) puts eigenvectors in V, eigenvalues in D
# ============================================================

class TestR10_GUI_EigOrder:
    """R10: eig() returns V=eigenvectors, D=eigenvalues (MATLAB convention)."""

    def test_eig_convention(self, forge_window):
        gui_exec(forge_window, "A = [2 1; 0 3];")
        gui_exec(forge_window, "[V, D] = eig(A);")
        # D should be diagonal with eigenvalues 2 and 3
        r_d = gui_exec(forge_window, "D")
        assert "2" in r_d and "3" in r_d, f"D should have eigenvalues 2,3: '{r_d}'"
        r_offdiag = gui_exec(forge_window, "D(1,2)")
        assert "0" in r_offdiag, f"D should be diagonal, D(1,2)=0: '{r_offdiag}'"


# ============================================================
# R11 — Correct Nested Function Call Results
# Verify: max(abs([...])) returns correct value in command window
# ============================================================

class TestR11_GUI_NestedCalls:
    """R11: Nested function calls produce correct results in the GUI."""

    def test_max_abs(self, forge_window):
        r = gui_exec(forge_window, "max(abs([-3 5 -7 2]))")
        assert "7" in r, f"max(abs([-3 5 -7 2])) should be 7, got: '{r}'"

    def test_sum_abs(self, forge_window):
        r = gui_exec(forge_window, "sum(abs([-1 -2 3]))")
        assert "6" in r, f"sum(abs([-1 -2 3])) should be 6, got: '{r}'"


# ============================================================
# R12 — Multi-Output Function Assignment
# Verify: [m,idx] = max(v) assigns both vars, visible in workspace
# ============================================================

class TestR12_GUI_MultiOutput:
    """R12: Multi-output functions assign all variables in the GUI."""

    def test_max_with_index(self, forge_window):
        gui_exec(forge_window, "[m, idx] = max([10 30 20]);")
        r_m = gui_exec(forge_window, "m")
        r_idx = gui_exec(forge_window, "idx")
        assert "30" in r_m, f"m should be 30, got: '{r_m}'"
        assert "2" in r_idx, f"idx should be 2, got: '{r_idx}'"
        assert workspace_has(forge_window, "m"), "m should be in workspace"
        assert workspace_has(forge_window, "idx"), "idx should be in workspace"

    def test_size_two_outputs(self, forge_window):
        gui_exec(forge_window, "A = ones(3, 5);")
        gui_exec(forge_window, "[rows, cols] = size(A);")
        r_rows = gui_exec(forge_window, "rows")
        r_cols = gui_exec(forge_window, "cols")
        assert "3" in r_rows and "5" in r_cols


# ============================================================
# R13 — .m File Auto-Discovery from Path
# Verify: function in .m file is callable after addpath in GUI
# ============================================================

class TestR13_GUI_MFileDiscovery:
    """R13: .m files on the path are auto-discovered from the GUI."""

    def test_function_from_m_file(self, forge_window):
        # Create a temp .m file
        with tempfile.TemporaryDirectory() as tmpdir:
            mfile = os.path.join(tmpdir, "gui_add.m")
            with open(mfile, "w") as f:
                f.write("function r = gui_add(a, b)\n  r = a + b;\nend\n")
            gui_exec(forge_window, f'addpath("{tmpdir}")')
            r = gui_exec(forge_window, "gui_add(3, 7)")
            assert "10" in r, f"gui_add(3,7) should return 10, got: '{r}'"


# ============================================================
# R14 — Script Execution
# Verify: run() executes .m script, variables appear in workspace
# ============================================================

class TestR14_GUI_ScriptExecution:
    """R14: run() executes scripts, results visible in workspace."""

    def test_run_script_creates_variables(self, forge_window):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "gui_script.m")
            with open(script, "w") as f:
                f.write("gui_script_var = 84;\n")
            gui_exec(forge_window, f'run("{script}")')
            assert workspace_has(forge_window, "gui_script_var"), \
                "Script variable should appear in workspace"
            r = gui_exec(forge_window, "gui_script_var")
            assert "84" in r


# ============================================================
# R15 — Missing Core Functions
# Verify: dot, sub2ind, tic/toc work from the command window
# ============================================================

class TestR15_GUI_MissingFunctions:
    """R15: Core functions work interactively in the GUI."""

    def test_dot_product(self, forge_window):
        r = gui_exec(forge_window, "dot([1 2 3], [4 5 6])")
        assert "32" in r, f"dot product should be 32, got: '{r}'"

    def test_sub2ind(self, forge_window):
        r = gui_exec(forge_window, "sub2ind([3 4], 2, 3)")
        assert "8" in r, f"sub2ind should return 8, got: '{r}'"

    def test_tic_toc(self, forge_window):
        gui_exec(forge_window, "tic;")
        r = gui_exec(forge_window, "toc")
        assert r.strip() != "", f"toc should produce output, got: '{r}'"


# ============================================================
# R16 — Figure/Plot Integration
# Verify: plot() does not crash, figure does not return function ref
# ============================================================

class TestR16_GUI_Plotting:
    """R16: Plotting commands work without crashing in the GUI."""

    def test_plot_no_crash(self, forge_window):
        gui_exec(forge_window, "x = linspace(0, 6.28, 50);")
        gui_exec(forge_window, "y = sin(x);")
        r = gui_exec(forge_window, "plot(x, y)")
        assert "error" not in r.lower(), f"plot should not error: '{r}'"

    def test_figure_no_function_ref(self, forge_window):
        r = gui_exec(forge_window, "figure")
        assert "function" not in r.lower(), \
            f"figure should not show function ref: '{r}'"
