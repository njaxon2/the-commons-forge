"""
tests/test_polish_r12.py -- File I/O, formatting, and plotting polish tests.

Round 12 polish: csvread/csvwrite, dlmread/dlmwrite, save/load,
sprintf/fprintf, plot/bar/scatter/subplot.

Requirement R-POL12-01:
    The file I/O functions (csvwrite/csvread, dlmwrite/dlmread, save/load)
    SHALL round-trip numeric data and workspace variables through disk files
    without loss of values, and SHALL write clean numeric formatting (no
    unwanted scientific notation).

    Model-user argument:
    An engineer migrating from MATLAB/Octave constantly exports simulation
    results to CSV for colleagues and imports measurement data from
    tab-delimited instruments. If csvwrite silently drops precision or
    dlmread misparses a tab-separated file, experimental data is corrupted.
    The save/load cycle is equally critical: losing workspace state between
    sessions means losing hours of computation.

    Decomposition:
    R-POL12-01a: csvwrite then csvread recovers the original matrix.
    R-POL12-01b: csvwrite creates the file on disk.
    R-POL12-01c: csvwrite writes clean numeric format (no scientific notation).
    R-POL12-01d: dlmwrite/dlmread round-trips with tab delimiter.
    R-POL12-01e: dlmwrite/dlmread round-trips with comma delimiter.
    R-POL12-01f: dlmwrite writes clean numeric format.
    R-POL12-01g: save then load restores workspace variables.
    R-POL12-01h: save creates a .mat file on disk.
    R-POL12-01i: load works in command style (no parentheses).

    Consistency argument:
    Sub-requirements 01a-01c cover csvwrite/csvread value fidelity, file
    creation, and format cleanliness. 01d-01f cover dlmwrite/dlmread with
    two delimiter types and format. 01g-01i cover save/load value fidelity,
    file creation, and command-style syntax. Together they fully verify all
    six file I/O functions.

Requirement R-POL12-02:
    The string formatting functions (sprintf, fprintf) SHALL format strings,
    integers, and floats with printf-style specifiers, process escape
    sequences, and write to file handles.

    Model-user argument:
    An engineer generates formatted log files and console output using
    sprintf/fprintf patterns memorized over years of MATLAB use. If \\n
    does not produce an actual newline, or %d truncates instead of rounding,
    their automated reporting scripts produce garbled output that they must
    hand-edit.

    Decomposition:
    R-POL12-02a: sprintf formats string and integer substitutions.
    R-POL12-02b: sprintf formats floating-point with precision specifier.
    R-POL12-02c: sprintf processes \\n and \\t escape sequences.
    R-POL12-02d: fprintf writes formatted text to a file handle.
    R-POL12-02e: fprintf processes \\n as actual newline in file output.

    Consistency argument:
    01a-01c cover sprintf with three format-specifier categories. 01d-01e
    cover fprintf file output with value formatting and escape processing.
    Together they verify both formatting functions.

Requirement R-POL12-03:
    The plotting functions (plot, bar, scatter, subplot) SHALL execute
    without error in headless (Agg) mode.

    Model-user argument:
    An engineer runs batch scripts on headless servers to generate plots
    for reports. If plot() crashes when no display is available, their
    entire batch pipeline fails. The functions must at minimum execute
    cleanly; visual correctness is verified separately in GUI integration
    tests.

    Decomposition:
    R-POL12-03a: plot executes without error.
    R-POL12-03b: bar executes without error.
    R-POL12-03c: scatter executes without error.
    R-POL12-03d: subplot with mixed plot types executes without error.

    Consistency argument:
    Sub-requirements 03a-03d each test one plotting function (or combination)
    for crash-free execution. Together they cover the four most common
    plotting entry points.
"""
import os, sys, tempfile, pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray
from forge.engine.containers import ForgeChar


@pytest.fixture
def s():
    """Fresh ForgeSession for each test."""
    return ForgeSession()


# -- csvwrite / csvread (R-POL12-01a through 01c) -----------------------------

def test_csvwrite_csvread_roundtrip(s):
    """R-POL12-01a: csvwrite then csvread recovers the original matrix."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_csv.csv")
    s.eval(f"A = [1 2 3; 4 5 6]; csvwrite('{path}', A)")
    s.eval(f"B = csvread('{path}')")
    B = s._engine.workspace.get("B")
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    np.testing.assert_array_almost_equal(B.data, expected)


def test_csvwrite_creates_file(s):
    """R-POL12-01b: csvwrite creates a file on disk."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_csv_exist.csv")
    if os.path.exists(path):
        os.remove(path)
    s.eval(f"csvwrite('{path}', [10 20 30])")
    assert os.path.exists(path)


def test_csvwrite_format_clean(s):
    """R-POL12-01c: csvwrite writes clean numbers, not scientific notation."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_csv_fmt.csv")
    s.eval(f"csvwrite('{path}', [1 2 3])")
    with open(path) as f:
        content = f.read()
    # Should not contain 'e+' from scientific notation
    assert "e+" not in content.lower()
    assert "1" in content


# -- dlmwrite / dlmread (R-POL12-01d through 01f) -----------------------------

def test_dlmwrite_dlmread_tab(s):
    """R-POL12-01d: dlmwrite/dlmread round-trip with tab delimiter."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_dlm_tab.txt")
    s.eval(f"M = [10 20; 30 40]; dlmwrite('{path}', M, '\t')")
    s.eval(f"N = dlmread('{path}', '\t')")
    N = s._engine.workspace.get("N")
    expected = np.array([[10, 20], [30, 40]], dtype=float)
    np.testing.assert_array_almost_equal(N.data, expected)


def test_dlmwrite_dlmread_comma(s):
    """R-POL12-01e: dlmwrite/dlmread round-trip with comma delimiter."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_dlm_comma.txt")
    s.eval(f"M = [5 6 7; 8 9 10]; dlmwrite('{path}', M, ',')")
    s.eval(f"N = dlmread('{path}', ',')")
    N = s._engine.workspace.get("N")
    expected = np.array([[5, 6, 7], [8, 9, 10]], dtype=float)
    np.testing.assert_array_almost_equal(N.data, expected)


def test_dlmwrite_format_clean(s):
    """R-POL12-01f: dlmwrite writes clean numbers, not scientific notation."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_dlm_fmt.txt")
    s.eval(f"dlmwrite('{path}', [1 2 3], ',')")
    with open(path) as f:
        content = f.read()
    # %.6g should produce "1,2,3" not "1.000000000000000000e+00,..."
    assert "e+" not in content.lower()


# -- save / load (R-POL12-01g through 01i) ------------------------------------

def test_save_load_roundtrip(s):
    """R-POL12-01g: save then load restores workspace variables."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_save.mat")
    s.eval("x = 42; y = [1 2 3]")
    s.eval(f"save('{path}', 'x', 'y')")
    s.eval("clear")
    s.eval(f"load('{path}')")
    x = s._engine.workspace.get("x")
    y = s._engine.workspace.get("y")
    assert float(x.data.flat[0]) == 42.0
    np.testing.assert_array_almost_equal(y.data.ravel(), [1, 2, 3])


def test_save_creates_file(s):
    """R-POL12-01h: save creates a .mat file on disk."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_save_exist.mat")
    if os.path.exists(path):
        os.remove(path)
    s.eval("z = 99")
    s.eval(f"save('{path}', 'z')")
    assert os.path.exists(path)


def test_load_command_style(s):
    """R-POL12-01i: load works in command style without parentheses."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_cmdload.mat")
    s.eval("abc = 77")
    s.eval(f"save('{path}', 'abc')")
    s.eval("clear")
    s.eval(f"load {path}")
    abc = s._engine.workspace.get("abc")
    assert float(abc.data.flat[0]) == 77.0


# -- sprintf (R-POL12-02a through 02c) ----------------------------------------

def test_sprintf_string_and_int(s):
    """R-POL12-02a: sprintf formats string and integer substitutions."""
    s.eval("s = sprintf('Hello %s, age %d', 'World', 25)")
    val = s._engine.workspace.get("s")
    assert isinstance(val, ForgeChar)
    assert val.to_str() == "Hello World, age 25"


def test_sprintf_float(s):
    """R-POL12-02b: sprintf formats floating-point with precision specifier."""
    s.eval("s = sprintf('pi=%.4f', 3.14159)")
    val = s._engine.workspace.get("s")
    assert "3.1416" in val.to_str()


def test_sprintf_escape_sequences(s):
    """R-POL12-02c: sprintf processes \\n and \\t escape sequences."""
    s.eval("s = sprintf('a\\tb\\nc')")
    val = s._engine.workspace.get("s")
    text = val.to_str()
    assert "\t" in text
    assert "\n" in text


# -- fprintf (R-POL12-02d through 02e) ----------------------------------------

def test_fprintf_to_file(s):
    """R-POL12-02d: fprintf writes formatted text to a file handle."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_fprintf.txt")
    s.eval(f"fid = fopen('{path}', 'w')")
    s.eval("fprintf(fid, 'value=%d', 42)")
    s.eval("fclose(fid)")
    with open(path) as f:
        content = f.read()
    assert content == "value=42"


def test_fprintf_newline_escape(s):
    """R-POL12-02e: fprintf processes \\n as actual newline in file output."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_fprintf_nl.txt")
    s.eval(f"fid = fopen('{path}', 'w')")
    s.eval("fprintf(fid, 'line1\\nline2\\n')")
    s.eval("fclose(fid)")
    with open(path) as f:
        content = f.read()
    assert content == "line1\nline2\n"


# -- plot functions (R-POL12-03a through 03d) ----------------------------------

def test_plot_no_crash(s):
    """R-POL12-03a: plot() executes without error in headless mode."""
    import matplotlib
    matplotlib.use("Agg")
    r = s.eval("figure; plot([1 2 3], [4 5 6]); title('Test')")
    assert "error" not in r.lower() if r else True


def test_bar_no_crash(s):
    """R-POL12-03b: bar() executes without error in headless mode."""
    import matplotlib
    matplotlib.use("Agg")
    r = s.eval("figure; bar([1 2 3 4])")
    assert "error" not in r.lower() if r else True


def test_scatter_no_crash(s):
    """R-POL12-03c: scatter() executes without error in headless mode."""
    import matplotlib
    matplotlib.use("Agg")
    r = s.eval("figure; scatter([1 2 3], [4 5 6])")
    assert "error" not in r.lower() if r else True


def test_subplot_no_crash(s):
    """R-POL12-03d: subplot with mixed plot types executes without error."""
    import matplotlib
    matplotlib.use("Agg")
    r = s.eval("figure; subplot(2,1,1); plot([1 2 3]); subplot(2,1,2); bar([4 5 6])")
    assert "error" not in r.lower() if r else True
