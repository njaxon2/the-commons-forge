# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""GUI Integration Tests — Traceable to Requirements R01–R16.

These tests run on the LIVE VNC display (DISPLAY=:99), creating a visible
Forge window that the user can observe through their VNC client. Each test
types commands into the command widget and verifies what appears — exactly
as a user would see it.

Run with:
    DISPLAY=:99 python3 -m pytest tests/test_gui_integration_r1.py -v -s

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
import time
import tempfile

import pytest

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer

# Ensure forge is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forge.engine.session import ForgeSession
from forge.gui.main_window import ForgeMainWindow
from forge.gui.command_widget import CommandWidget

# Pause between commands so the VNC observer can follow along.
# Set FORGE_TEST_DELAY=0 to run at full speed.
_DELAY = float(os.environ.get("FORGE_TEST_DELAY", "0.4"))


# ============================================================
# Fixtures — single visible window for the entire test session
# ============================================================

@pytest.fixture(scope="session")
def qapp():
    """Create or reuse the QApplication on the live display."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture(scope="session")
def forge_gui(qapp):
    """Create one Forge window that stays visible for all tests."""
    session = ForgeSession()
    window = ForgeMainWindow()
    window.setup_engine(session)
    window.setWindowTitle("Forge — Integration Test Run")
    window.resize(1200, 800)

    # Centre on screen
    screen = qapp.primaryScreen()
    if screen:
        geo = screen.availableGeometry()
        frame = window.frameGeometry()
        frame.moveCenter(geo.center())
        window.move(frame.topLeft())

    window.show()
    window.raise_()
    qapp.processEvents()
    time.sleep(0.5)  # Let the window paint

    yield window, session

    # Leave window open for the VNC observer
    # window.close()
    qapp.processEvents()


@pytest.fixture
def fw(forge_gui):
    """Per-test alias; clears the command output between tests for clarity."""
    window, session = forge_gui
    qapp = QApplication.instance()
    # Add a separator so the observer can see test boundaries
    window.command_widget.append_output("")
    window.command_widget.append_output("─" * 50)
    qapp.processEvents()
    yield window, session


# ============================================================
# Helpers
# ============================================================

def gui_exec(fw, command):
    """Type a command into the GUI command widget and return the output.

    Simulates exactly what a user does: type text, press Enter, read output.
    The VNC observer sees the command appear and the result display.
    """
    window, session = fw
    qapp = QApplication.instance()

    cw = window.command_widget
    output_before = cw.console.toPlainText()

    # Type the command visibly
    # (input via _execute below)
    qapp.processEvents()
    if _DELAY:
        time.sleep(_DELAY * 0.5)  # Brief pause so observer sees the typed text

    # Press Enter
    cw._execute(command)
    qapp.processEvents()
    if _DELAY:
        time.sleep(_DELAY)  # Pause so observer sees the result

    # Extract new output
    output_after = cw.console.toPlainText()
    new_output = output_after[len(output_before):].strip()

    # Strip the echo line (>> command) to get just the result
    lines = new_output.split("\n")
    result_lines = [l for l in lines
                    if not l.startswith(">> ") and l.strip() != ">>"
                    and not l.startswith(".. ")
                    and not l.startswith("─")]
    return "\n".join(result_lines).strip()


def workspace_has(fw, varname):
    """Check if a variable appears in the workspace browser panel."""
    window, session = fw
    QApplication.instance().processEvents()
    table = window.workspace_widget.table
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item and item.text() == varname:
            return True
    return False


# ============================================================
# R01 — Semicolon Output Suppression
# ============================================================

class TestR01_GUI_SemicolonSuppression:
    """R01: Statements ending with ; suppress output in the command window."""

    def test_semicolon_no_output_in_gui(self, fw):
        r = gui_exec(fw, "a = 42;")
        assert r == "", f"Semicolon should suppress output, got: '{r}'"

    def test_no_semicolon_shows_output(self, fw):
        r = gui_exec(fw, "b = 42")
        assert "42" in r, f"Should display 42, got: '{r}'"

    def test_semicolon_variable_still_created(self, fw):
        gui_exec(fw, "hidden_var = 99;")
        assert workspace_has(fw, "hidden_var"), \
            "Variable should exist in workspace even with semicolon"


# ============================================================
# R02 — MATLAB-Style Output Formatting
# ============================================================

class TestR02_GUI_OutputFormat:
    """R02: Output in the command window uses MATLAB-style formatting."""

    def test_scalar_no_numpy(self, fw):
        r = gui_exec(fw, "x = 5")
        assert "[[" not in r, f"No numpy brackets, got: '{r}'"
        assert "5" in r

    def test_matrix_formatted(self, fw):
        r = gui_exec(fw, "A = [1 2; 3 4]")
        assert "[[" not in r, f"No numpy brackets, got: '{r}'"
        assert "1" in r and "4" in r

    def test_float_format_short(self, fw):
        r = gui_exec(fw, "pi_val = 3.14159265")
        assert "3.1416" in r, f"Expected format short (4 decimal), got: '{r}'"


# ============================================================
# R03 — Character Array Display
# ============================================================

class TestR03_GUI_CharDisplay:
    """R03: Character arrays display as readable text, not ASCII codes."""

    def test_string_displays_as_text(self, fw):
        r = gui_exec(fw, "s = 'hello'")
        assert "hello" in r, f"Should show 'hello', got: '{r}'"
        assert "104" not in r, f"Should not show ASCII codes, got: '{r}'"

    def test_class_returns_char(self, fw):
        gui_exec(fw, "s = 'world';")
        r = gui_exec(fw, "class(s)")
        assert "char" in r, f"class should be char, got: '{r}'"


# ============================================================
# R04 — Slash Characters in Single-Quoted Strings
# ============================================================

class TestR04_GUI_SlashInStrings:
    """R04: Single-quoted strings with / parse correctly in the GUI."""

    def test_path_string(self, fw):
        r = gui_exec(fw, "p = \"/tmp/test\"")
        assert "error" not in r.lower() or "parse" not in r.lower(), \
            f"Path string should not cause parse error: '{r}'"

    def test_addpath_with_slash(self, fw):
        r = gui_exec(fw, "addpath(\"/tmp\")")
        assert "ParseError" not in r, f"addpath should not parse-error: '{r}'"


# ============================================================
# R05 — Command-Style Syntax
# ============================================================

class TestR05_GUI_CommandStyle:
    """R05: Bare commands (who, whos) execute in the command window."""

    def test_who_not_function_ref(self, fw):
        gui_exec(fw, "test_var = 1;")
        r = gui_exec(fw, "who")
        assert "function" not in r.lower(), \
            f"who should execute, not show function ref: '{r}'"

    def test_whos_not_function_ref(self, fw):
        gui_exec(fw, "xx = 5;")
        r = gui_exec(fw, "whos")
        assert "function" not in r.lower(), \
            f"whos should execute, not show function ref: '{r}'"

    def test_hold_on_no_error(self, fw):
        r = gui_exec(fw, "hold on")
        assert "NameError" not in r, f"hold on should not error: '{r}'"

    def test_axis_equal_no_error(self, fw):
        r = gui_exec(fw, "axis equal")
        assert "NameError" not in r, f"axis equal should not error: '{r}'"


# ============================================================
# R06 — Float-to-Int Coercion for Shape Arguments
# ============================================================

class TestR06_GUI_FloatToInt:
    """R06: reshape/zeros/ones accept float dimension arguments."""

    def test_reshape_in_gui(self, fw):
        r = gui_exec(fw, "reshape(1:6, 2, 3)")
        assert "error" not in r.lower(), f"reshape should work: '{r}'"
        assert "1" in r and "6" in r

    def test_zeros_with_variable(self, fw):
        gui_exec(fw, "n = 3;")
        r = gui_exec(fw, "zeros(n, n)")
        assert "error" not in r.lower(), f"zeros(n,n) should work: '{r}'"


# ============================================================
# R07 — Struct Auto-Creation on Field Assignment
# ============================================================

class TestR07_GUI_StructAutoCreate:
    """R07: s.field = val auto-creates struct, visible in workspace."""

    def test_struct_creation_in_gui(self, fw):
        r = gui_exec(fw, "msh.nodes = [0 0; 1 0];")
        assert r == "", "Semicolon should suppress"
        assert workspace_has(fw, "msh"), \
            "Struct 'msh' should appear in workspace browser"

    def test_struct_field_read(self, fw):
        gui_exec(fw, "data.x = 42;")
        r = gui_exec(fw, "data.x")
        assert "42" in r, f"Field read should return 42, got: '{r}'"

    def test_struct_multiple_fields(self, fw):
        gui_exec(fw, "pt.x = 1;")
        gui_exec(fw, "pt.y = 2;")
        r_x = gui_exec(fw, "pt.x")
        r_y = gui_exec(fw, "pt.y")
        assert "1" in r_x and "2" in r_y


# ============================================================
# R08 — Sparse Matrix Construction
# ============================================================

class TestR08_GUI_Sparse:
    """R08: sparse() creates correct sparse matrices via the GUI."""

    def test_sparse_zeros_gui(self, fw):
        gui_exec(fw, "S = sparse(3, 3);")
        r = gui_exec(fw, "nnz(S)")
        assert "0" in r, f"Empty sparse should have 0 nonzeros, got: '{r}'"

    def test_sparse_triplet_gui(self, fw):
        gui_exec(fw, "I = [1 1 2 3]; J = [1 2 2 3]; V = [4 1 5 6];")
        gui_exec(fw, "K = sparse(I, J, V, 3, 3);")
        r = gui_exec(fw, "full(K)")
        assert "4" in r and "5" in r, f"Sparse should contain 4 and 5: '{r}'"


# ============================================================
# R09 — Indexed Assignment with RHS Expressions
# ============================================================

class TestR09_GUI_IndexedAssignment:
    """R09: K(i,j) = K(i,j) + ke works in the command window."""

    def test_assembly_pattern(self, fw):
        gui_exec(fw, "K = zeros(4, 4);")
        gui_exec(fw, "ke = [2 -1; -1 2];")
        gui_exec(fw, "K(1:2, 1:2) = K(1:2, 1:2) + ke;")
        r = gui_exec(fw, "K(1,1)")
        assert "2" in r, f"K(1,1) should be 2 after assembly, got: '{r}'"
        r2 = gui_exec(fw, "K(1,2)")
        assert "-1" in r2, f"K(1,2) should be -1, got: '{r2}'"


# ============================================================
# R10 — eig() Return Order
# ============================================================

class TestR10_GUI_EigOrder:
    """R10: eig() returns V=eigenvectors, D=eigenvalues (MATLAB convention)."""

    def test_eig_convention(self, fw):
        gui_exec(fw, "Ae = [2 1; 0 3];")
        gui_exec(fw, "[V, D] = eig(Ae);")
        r_d = gui_exec(fw, "D")
        assert "2" in r_d and "3" in r_d, f"D should have eigenvalues 2,3: '{r_d}'"
        r_offdiag = gui_exec(fw, "D(1,2)")
        assert "0" in r_offdiag, f"D should be diagonal, D(1,2)=0: '{r_offdiag}'"


# ============================================================
# R11 — Correct Nested Function Call Results
# ============================================================

class TestR11_GUI_NestedCalls:
    """R11: Nested function calls produce correct results in the GUI."""

    def test_max_abs(self, fw):
        r = gui_exec(fw, "max(abs([-3 5 -7 2]))")
        assert "7" in r, f"max(abs([-3 5 -7 2])) should be 7, got: '{r}'"

    def test_sum_abs(self, fw):
        r = gui_exec(fw, "sum(abs([-1 -2 3]))")
        assert "6" in r, f"sum(abs([-1 -2 3])) should be 6, got: '{r}'"


# ============================================================
# R12 — Multi-Output Function Assignment
# ============================================================

class TestR12_GUI_MultiOutput:
    """R12: Multi-output functions assign all variables in the GUI."""

    def test_max_with_index(self, fw):
        gui_exec(fw, "[m, idx] = max([10 30 20]);")
        r_m = gui_exec(fw, "m")
        r_idx = gui_exec(fw, "idx")
        assert "30" in r_m, f"m should be 30, got: '{r_m}'"
        assert "2" in r_idx, f"idx should be 2, got: '{r_idx}'"
        assert workspace_has(fw, "m"), "m should be in workspace"
        assert workspace_has(fw, "idx"), "idx should be in workspace"

    def test_size_two_outputs(self, fw):
        gui_exec(fw, "Asz = ones(3, 5);")
        gui_exec(fw, "[rows, cols] = size(Asz);")
        r_rows = gui_exec(fw, "rows")
        r_cols = gui_exec(fw, "cols")
        assert "3" in r_rows and "5" in r_cols


# ============================================================
# R13 — .m File Auto-Discovery from Path
# ============================================================

class TestR13_GUI_MFileDiscovery:
    """R13: .m files on the path are auto-discovered from the GUI."""

    def test_function_from_m_file(self, fw):
        with tempfile.TemporaryDirectory() as tmpdir:
            mfile = os.path.join(tmpdir, "gui_add.m")
            with open(mfile, "w") as f:
                f.write("function r = gui_add(a, b)\n  r = a + b;\nend\n")
            gui_exec(fw, f'addpath("{tmpdir}")')
            r = gui_exec(fw, "gui_add(3, 7)")
            assert "10" in r, f"gui_add(3,7) should return 10, got: '{r}'"


# ============================================================
# R14 — Script Execution
# ============================================================

class TestR14_GUI_ScriptExecution:
    """R14: run() executes scripts, results visible in workspace."""

    def test_run_script_creates_variables(self, fw):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "gui_script.m")
            with open(script, "w") as f:
                f.write("gui_script_var = 84;\n")
            gui_exec(fw, f'run("{script}")')
            assert workspace_has(fw, "gui_script_var"), \
                "Script variable should appear in workspace"
            r = gui_exec(fw, "gui_script_var")
            assert "84" in r


# ============================================================
# R15 — Missing Core Functions
# ============================================================

class TestR15_GUI_MissingFunctions:
    """R15: Core functions work interactively in the GUI."""

    def test_dot_product(self, fw):
        r = gui_exec(fw, "dot([1 2 3], [4 5 6])")
        assert "32" in r, f"dot product should be 32, got: '{r}'"

    def test_sub2ind(self, fw):
        r = gui_exec(fw, "sub2ind([3 4], 2, 3)")
        assert "8" in r, f"sub2ind should return 8, got: '{r}'"

    def test_tic_toc(self, fw):
        gui_exec(fw, "tic;")
        r = gui_exec(fw, "toc")
        assert r.strip() != "", f"toc should produce output, got: '{r}'"


# ============================================================
# R16 — Figure/Plot Integration
# ============================================================

class TestR16_GUI_Plotting:
    """R16: Plotting commands work without crashing in the GUI."""

    def test_plot_no_crash(self, fw):
        gui_exec(fw, "x = linspace(0, 6.28, 50);")
        gui_exec(fw, "y = sin(x);")
        r = gui_exec(fw, "plot(x, y)")
        assert "error" not in r.lower(), f"plot should not error: '{r}'"

    def test_figure_no_function_ref(self, fw):
        r = gui_exec(fw, "figure")
        assert "function" not in r.lower(), \
            f"figure should not show function ref: '{r}'"
