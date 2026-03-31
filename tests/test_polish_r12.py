"""
tests/test_polish_r12.py – File I/O, formatting, and plotting polish tests.

Round 12 polish: csvread/csvwrite, dlmread/dlmwrite, save/load,
sprintf/fprintf, plot/bar/scatter/subplot.
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


# ── csvwrite / csvread ────────────────────────────────────────────────

def test_csvwrite_csvread_roundtrip(s):
    """csvwrite then csvread should recover the original matrix."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_csv.csv")
    s.eval(f"A = [1 2 3; 4 5 6]; csvwrite('{path}', A)")
    s.eval(f"B = csvread('{path}')")
    B = s._engine.workspace.get("B")
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    np.testing.assert_array_almost_equal(B.data, expected)


def test_csvwrite_creates_file(s):
    """csvwrite should actually create the file on disk."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_csv_exist.csv")
    if os.path.exists(path):
        os.remove(path)
    s.eval(f"csvwrite('{path}', [10 20 30])")
    assert os.path.exists(path)


def test_csvwrite_format_clean(s):
    """csvwrite should write clean numbers, not scientific notation."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_csv_fmt.csv")
    s.eval(f"csvwrite('{path}', [1 2 3])")
    with open(path) as f:
        content = f.read()
    # Should not contain 'e+' from scientific notation
    assert "e+" not in content.lower()
    assert "1" in content


# ── dlmwrite / dlmread ────────────────────────────────────────────────

def test_dlmwrite_dlmread_tab(s):
    """dlmwrite/dlmread round-trip with tab delimiter."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_dlm_tab.txt")
    s.eval(f"M = [10 20; 30 40]; dlmwrite('{path}', M, '\t')")
    s.eval(f"N = dlmread('{path}', '\t')")
    N = s._engine.workspace.get("N")
    expected = np.array([[10, 20], [30, 40]], dtype=float)
    np.testing.assert_array_almost_equal(N.data, expected)


def test_dlmwrite_dlmread_comma(s):
    """dlmwrite/dlmread round-trip with comma delimiter."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_dlm_comma.txt")
    s.eval(f"M = [5 6 7; 8 9 10]; dlmwrite('{path}', M, ',')")
    s.eval(f"N = dlmread('{path}', ',')")
    N = s._engine.workspace.get("N")
    expected = np.array([[5, 6, 7], [8, 9, 10]], dtype=float)
    np.testing.assert_array_almost_equal(N.data, expected)


def test_dlmwrite_format_clean(s):
    """dlmwrite should write clean numbers, not 18-digit scientific notation."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_dlm_fmt.txt")
    s.eval(f"dlmwrite('{path}', [1 2 3], ',')")
    with open(path) as f:
        content = f.read()
    # %.6g should produce "1,2,3" not "1.000000000000000000e+00,..."
    assert "e+" not in content.lower()


# ── save / load ───────────────────────────────────────────────────────

def test_save_load_roundtrip(s):
    """save then load should restore workspace variables."""
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
    """save should create a .mat file on disk."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_save_exist.mat")
    if os.path.exists(path):
        os.remove(path)
    s.eval("z = 99")
    s.eval(f"save('{path}', 'z')")
    assert os.path.exists(path)


def test_load_command_style(s):
    """load should work in command style: load /path/to/file.mat"""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_cmdload.mat")
    s.eval("abc = 77")
    s.eval(f"save('{path}', 'abc')")
    s.eval("clear")
    s.eval(f"load {path}")
    abc = s._engine.workspace.get("abc")
    assert float(abc.data.flat[0]) == 77.0


# ── sprintf ───────────────────────────────────────────────────────────

def test_sprintf_string_and_int(s):
    """sprintf should format strings and integers."""
    s.eval("s = sprintf('Hello %s, age %d', 'World', 25)")
    val = s._engine.workspace.get("s")
    assert isinstance(val, ForgeChar)
    assert val.to_str() == "Hello World, age 25"


def test_sprintf_float(s):
    """sprintf should format floating-point numbers."""
    s.eval("s = sprintf('pi=%.4f', 3.14159)")
    val = s._engine.workspace.get("s")
    assert "3.1416" in val.to_str()


def test_sprintf_escape_sequences(s):
    """sprintf should process \\n and \\t escape sequences."""
    s.eval("s = sprintf('a\\tb\\nc')")
    val = s._engine.workspace.get("s")
    text = val.to_str()
    assert "\t" in text
    assert "\n" in text


# ── fprintf ───────────────────────────────────────────────────────────

def test_fprintf_to_file(s):
    """fprintf should write formatted text to a file."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_fprintf.txt")
    s.eval(f"fid = fopen('{path}', 'w')")
    s.eval("fprintf(fid, 'value=%d', 42)")
    s.eval("fclose(fid)")
    with open(path) as f:
        content = f.read()
    assert content == "value=42"


def test_fprintf_newline_escape(s):
    """fprintf should process \\n as actual newline."""
    path = os.path.join(tempfile.gettempdir(), "forge_r12_fprintf_nl.txt")
    s.eval(f"fid = fopen('{path}', 'w')")
    s.eval("fprintf(fid, 'line1\\nline2\\n')")
    s.eval("fclose(fid)")
    with open(path) as f:
        content = f.read()
    assert content == "line1\nline2\n"


# ── plot functions (no display, no crash) ─────────────────────────────

def test_plot_no_crash(s):
    """plot() should execute without error."""
    import matplotlib
    matplotlib.use("Agg")
    r = s.eval("figure; plot([1 2 3], [4 5 6]); title('Test')")
    assert "error" not in r.lower() if r else True


def test_bar_no_crash(s):
    """bar() should execute without error."""
    import matplotlib
    matplotlib.use("Agg")
    r = s.eval("figure; bar([1 2 3 4])")
    assert "error" not in r.lower() if r else True


def test_scatter_no_crash(s):
    """scatter() should execute without error."""
    import matplotlib
    matplotlib.use("Agg")
    r = s.eval("figure; scatter([1 2 3], [4 5 6])")
    assert "error" not in r.lower() if r else True


def test_subplot_no_crash(s):
    """subplot() with mixed plot types should not crash."""
    import matplotlib
    matplotlib.use("Agg")
    r = s.eval("figure; subplot(2,1,1); plot([1 2 3]); subplot(2,1,2); bar([4 5 6])")
    assert "error" not in r.lower() if r else True
