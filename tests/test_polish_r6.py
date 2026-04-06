"""Tests for polish round 6: arrayfun, cellfun, structfun, num2str, str2num, str2double.

V-model traceability backfill: R-POL6-01 through R-POL6-06.
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.containers import ForgeCell, ForgeChar
from forge.engine.types import ForgeArray


@pytest.fixture
def s():
    return ForgeSession()


def _get_ans(s):
    """Get the 'ans' variable from workspace."""
    return s.workspace.get('ans')


# ── arrayfun ─────────────────────────────────────────────────

class TestArrayfun:
    """R-POL6-01: arrayfun SHALL apply a function element-wise to one or
    more arrays, returning a numeric array by default or a ForgeCell
    when UniformOutput is false.

    Model-user argument: Engineers use arrayfun as the vectorized apply
    pattern for element-wise transformations that cannot be expressed as
    simple arithmetic. It is the array counterpart to cellfun and is
    used extensively in data processing pipelines migrated from Octave.

    Decomposition:
      R-POL6-01a: arrayfun with single input applies function element-wise.
      R-POL6-01b: arrayfun with two inputs applies binary function.
      R-POL6-01c: arrayfun with UniformOutput=false returns ForgeCell.

    Consistency: Single-input, multi-input, and non-uniform output modes
    cover the three primary calling conventions of arrayfun.
    """

    def test_basic_anonymous(self, s):
        """R-POL6-01a: arrayfun(@(x) x^2, [1 2 3]) -> [1 4 9]."""
        s.eval('r = arrayfun(@(x) x^2, [1 2 3]);')
        r = s.workspace.get('r')
        np.testing.assert_array_equal(r.data.ravel(), [1, 4, 9])

    def test_multi_input(self, s):
        """R-POL6-01b: arrayfun(@(x,y) x+y, [1 2], [3 4]) -> [4 6]."""
        s.eval('r = arrayfun(@(x,y) x+y, [1 2], [3 4]);')
        r = s.workspace.get('r')
        np.testing.assert_array_equal(r.data.ravel(), [4, 6])

    def test_uniform_output_false(self, s):
        """R-POL6-01c: arrayfun with UniformOutput=false returns ForgeCell."""
        s.eval('r = arrayfun(@(x) x^2, [1 2 3], "UniformOutput", false);')
        r = s.workspace.get('r')
        assert isinstance(r, ForgeCell), f"Expected ForgeCell, got {type(r)}"
        assert len(r._data) == 3


# ── cellfun ──────────────────────────────────────────────────

class TestCellfun:
    """R-POL6-02: cellfun SHALL apply a function to each element of a
    cell array, supporting both function handles and string-form
    specifiers, with optional non-uniform output.

    Model-user argument: Engineers processing heterogeneous data stored
    in cell arrays use cellfun for type-checking (ischar), length
    computation, and string conversion. The 'isclass' string form is
    an Octave-specific calling convention that must work for migrated
    code.

    Decomposition:
      R-POL6-02a: cellfun(@ischar, ...) returns logical array.
      R-POL6-02b: cellfun('isclass', ..., 'char') uses string-form spec.
      R-POL6-02c: cellfun with UniformOutput=false returns ForgeCell.

    Consistency: Handle-based, string-form, and non-uniform output modes
    cover the three distinct calling conventions.
    """

    def test_ischar(self, s):
        """R-POL6-02a: cellfun(@ischar, ...) returns [1 0 1]."""
        s.eval('r = cellfun(@ischar, {"a", 1, "b"});')
        r = s.workspace.get('r')
        np.testing.assert_array_equal(r.data.ravel(), [1, 0, 1])

    def test_isclass_string_form(self, s):
        """R-POL6-02b: cellfun('isclass', ..., 'char') returns [1 0]."""
        s.eval('r = cellfun("isclass", {"a", 1}, "char");')
        r = s.workspace.get('r')
        np.testing.assert_array_equal(r.data.ravel(), [1, 0])

    def test_uniform_output_false(self, s):
        """R-POL6-02c: cellfun with UniformOutput=false returns ForgeCell."""
        s.eval('r = cellfun(@num2str, {1, 2, 3}, "UniformOutput", false);')
        r = s.workspace.get('r')
        assert isinstance(r, ForgeCell), f"Expected ForgeCell, got {type(r)}"
        assert len(r._data) == 3


# ── structfun ────────────────────────────────────────────────

class TestStructfun:
    """R-POL6-03: structfun SHALL apply a function to each field value
    of a struct, returning a numeric array by default or a ForgeCell
    when UniformOutput is false.

    Model-user argument: Engineers with struct-organized data (e.g.,
    sensor channels stored as struct fields) use structfun to apply
    uniform processing across all fields. This avoids manual fieldnames
    iteration and mirrors the Octave idiom.

    Decomposition:
      R-POL6-03a: structfun(@(x) x*2, st) doubles each field value.
      R-POL6-03b: structfun with UniformOutput=false returns ForgeCell.

    Consistency: Numeric output and cell output cover both output modes.
    """

    def test_basic(self, s):
        """R-POL6-03a: structfun(@(x) x*2, st) doubles field values."""
        s.eval('st.a = 1; st.b = 2; st.c = 3;')
        s.eval('r = structfun(@(x) x*2, st);')
        r = s.workspace.get('r')
        np.testing.assert_array_equal(r.data.ravel(), [2, 4, 6])

    def test_uniform_output_false(self, s):
        """R-POL6-03b: structfun with UniformOutput=false returns ForgeCell."""
        s.eval('st3.x = [1 2]; st3.y = [3 4 5];')
        s.eval('r = structfun(@(v) length(v), st3, "UniformOutput", false);')
        r = s.workspace.get('r')
        assert isinstance(r, ForgeCell), f"Expected ForgeCell, got {type(r)}"
        assert len(r._data) == 2


# ── num2str ──────────────────────────────────────────────────

class TestNum2str:
    """R-POL6-04: num2str SHALL convert numeric values to their string
    representation, supporting scalars, vectors, and format specifiers.

    Model-user argument: Engineers building display strings, file names,
    or log messages use num2str to embed numeric values in text. The
    function must handle scalars, vectors (space-separated), and
    printf-style format strings, matching Octave output exactly.

    Decomposition:
      R-POL6-04a: Scalar conversion produces the decimal string.
      R-POL6-04b: Vector conversion produces space-separated values.
      R-POL6-04c: Format-string conversion applies the format.

    Consistency: Scalar, vector, and formatted modes cover the three
    calling conventions of num2str.
    """

    def test_scalar(self, s):
        """R-POL6-04a: num2str(3.14) -> '3.14'."""
        s.eval('r = num2str(3.14);')
        r = s.workspace.get('r')
        text = r.to_str() if hasattr(r, 'to_str') else str(r)
        assert text == '3.14', f"Expected '3.14', got {text!r}"

    def test_vector(self, s):
        """R-POL6-04b: num2str([1 2 3]) -> '1  2  3'."""
        s.eval('r = num2str([1 2 3]);')
        r = s.workspace.get('r')
        text = r.to_str() if hasattr(r, 'to_str') else str(r)
        assert text == '1  2  3', f"Expected '1  2  3', got {text!r}"

    def test_format_string(self, s):
        """R-POL6-04c: num2str(pi, '%10.5f') applies format."""
        s.eval('r = num2str(pi, "%10.5f");')
        r = s.workspace.get('r')
        text = r.to_str() if hasattr(r, 'to_str') else str(r)
        assert text.strip() == '3.14159', f"Expected '3.14159' (stripped), got {text!r}"


# ── str2num ──────────────────────────────────────────────────

class TestStr2num:
    """R-POL6-05: str2num SHALL parse a numeric string into a scalar or
    matrix, supporting both simple scalars and bracket-delimited matrix
    notation.

    Model-user argument: Engineers reading numeric data from text files
    or user input use str2num to convert strings back to numbers. The
    bracket-matrix syntax (e.g., '[1 2; 3 4]') is used for inline
    matrix specification in configuration files.

    Decomposition:
      R-POL6-05a: Scalar string parses to a numeric value.
      R-POL6-05b: Matrix string parses to the correct 2x2 matrix.

    Consistency: Scalar and matrix parsing cover the two primary input
    formats.
    """

    def test_scalar(self, s):
        """R-POL6-05a: str2num('3.14') -> 3.14."""
        s.eval('r = str2num("3.14");')
        r = s.workspace.get('r')
        val = float(r.data.flat[0]) if hasattr(r, 'data') else float(r)
        assert abs(val - 3.14) < 1e-10

    def test_matrix(self, s):
        """R-POL6-05b: str2num('[1 2; 3 4]') -> 2x2 matrix."""
        s.eval('r = str2num("[1 2; 3 4]");')
        r = s.workspace.get('r')
        arr = r.data if hasattr(r, 'data') else np.array(r)
        assert arr.shape == (2, 2), f"Expected (2,2), got {arr.shape}"
        np.testing.assert_array_equal(arr, [[1, 2], [3, 4]])


# ── str2double ───────────────────────────────────────────────

class TestStr2double:
    """R-POL6-06: str2double SHALL parse a numeric string to a double,
    returning NaN for non-numeric input.

    Model-user argument: Engineers use str2double for strict numeric
    parsing where non-numeric input should produce NaN rather than an
    error. This is the standard pattern for robust data import where
    missing or malformed values are common.

    Decomposition:
      R-POL6-06a: Valid numeric string parses to correct double.
      R-POL6-06b: Non-numeric string returns NaN.

    Consistency: Valid and invalid inputs cover both branches of the
    function's contract.
    """

    def test_valid(self, s):
        """R-POL6-06a: str2double('3.14') -> 3.14."""
        s.eval('r = str2double("3.14");')
        r = s.workspace.get('r')
        val = float(r.data.flat[0]) if hasattr(r, 'data') else float(r)
        assert abs(val - 3.14) < 1e-10

    def test_invalid_returns_nan(self, s):
        """R-POL6-06b: str2double('abc') -> NaN."""
        s.eval('r = str2double("abc");')
        r = s.workspace.get('r')
        val = float(r.data.flat[0]) if hasattr(r, 'data') else float(r)
        assert np.isnan(val), f"Expected NaN, got {val}"
