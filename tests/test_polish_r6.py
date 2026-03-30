"""Tests for polish round 6: arrayfun, cellfun, structfun, num2str, str2num, str2double."""
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
    def test_basic_anonymous(self, s):
        """arrayfun(@(x) x^2, [1 2 3]) -> [1 4 9]"""
        s.eval('r = arrayfun(@(x) x^2, [1 2 3]);')
        r = s.workspace.get('r')
        np.testing.assert_array_equal(r.data.ravel(), [1, 4, 9])

    def test_multi_input(self, s):
        """arrayfun(@(x,y) x+y, [1 2], [3 4]) -> [4 6]"""
        s.eval('r = arrayfun(@(x,y) x+y, [1 2], [3 4]);')
        r = s.workspace.get('r')
        np.testing.assert_array_equal(r.data.ravel(), [4, 6])

    def test_uniform_output_false(self, s):
        """arrayfun with UniformOutput=false returns ForgeCell."""
        s.eval('r = arrayfun(@(x) x^2, [1 2 3], "UniformOutput", false);')
        r = s.workspace.get('r')
        assert isinstance(r, ForgeCell), f"Expected ForgeCell, got {type(r)}"
        assert len(r._data) == 3


# ── cellfun ──────────────────────────────────────────────────

class TestCellfun:
    def test_ischar(self, s):
        """cellfun(@ischar, {'a', 1, 'b'}) -> [1 0 1]"""
        s.eval('r = cellfun(@ischar, {"a", 1, "b"});')
        r = s.workspace.get('r')
        np.testing.assert_array_equal(r.data.ravel(), [1, 0, 1])

    def test_isclass_string_form(self, s):
        """cellfun('isclass', {'a', 1}, 'char') -> [1 0]"""
        s.eval('r = cellfun("isclass", {"a", 1}, "char");')
        r = s.workspace.get('r')
        np.testing.assert_array_equal(r.data.ravel(), [1, 0])

    def test_uniform_output_false(self, s):
        """cellfun with UniformOutput=false returns ForgeCell."""
        s.eval('r = cellfun(@num2str, {1, 2, 3}, "UniformOutput", false);')
        r = s.workspace.get('r')
        assert isinstance(r, ForgeCell), f"Expected ForgeCell, got {type(r)}"
        assert len(r._data) == 3


# ── structfun ────────────────────────────────────────────────

class TestStructfun:
    def test_basic(self, s):
        """structfun(@(x) x*2, struct) -> doubled values."""
        s.eval('st.a = 1; st.b = 2; st.c = 3;')
        s.eval('r = structfun(@(x) x*2, st);')
        r = s.workspace.get('r')
        np.testing.assert_array_equal(r.data.ravel(), [2, 4, 6])

    def test_uniform_output_false(self, s):
        """structfun with UniformOutput=false returns ForgeCell."""
        s.eval('st3.x = [1 2]; st3.y = [3 4 5];')
        s.eval('r = structfun(@(v) length(v), st3, "UniformOutput", false);')
        r = s.workspace.get('r')
        assert isinstance(r, ForgeCell), f"Expected ForgeCell, got {type(r)}"
        assert len(r._data) == 2


# ── num2str ──────────────────────────────────────────────────

class TestNum2str:
    def test_scalar(self, s):
        """num2str(3.14) -> '3.14'"""
        s.eval('r = num2str(3.14);')
        r = s.workspace.get('r')
        text = r.to_str() if hasattr(r, 'to_str') else str(r)
        assert text == '3.14', f"Expected '3.14', got {text!r}"

    def test_vector(self, s):
        """num2str([1 2 3]) -> '1  2  3'"""
        s.eval('r = num2str([1 2 3]);')
        r = s.workspace.get('r')
        text = r.to_str() if hasattr(r, 'to_str') else str(r)
        assert text == '1  2  3', f"Expected '1  2  3', got {text!r}"

    def test_format_string(self, s):
        """num2str(pi, '%10.5f') -> '   3.14159'"""
        s.eval('r = num2str(pi, "%10.5f");')
        r = s.workspace.get('r')
        text = r.to_str() if hasattr(r, 'to_str') else str(r)
        assert text.strip() == '3.14159', f"Expected '3.14159' (stripped), got {text!r}"


# ── str2num ──────────────────────────────────────────────────

class TestStr2num:
    def test_scalar(self, s):
        """str2num('3.14') -> 3.14"""
        s.eval('r = str2num("3.14");')
        r = s.workspace.get('r')
        val = float(r.data.flat[0]) if hasattr(r, 'data') else float(r)
        assert abs(val - 3.14) < 1e-10

    def test_matrix(self, s):
        """str2num('[1 2; 3 4]') -> 2x2 matrix"""
        s.eval('r = str2num("[1 2; 3 4]");')
        r = s.workspace.get('r')
        arr = r.data if hasattr(r, 'data') else np.array(r)
        assert arr.shape == (2, 2), f"Expected (2,2), got {arr.shape}"
        np.testing.assert_array_equal(arr, [[1, 2], [3, 4]])


# ── str2double ───────────────────────────────────────────────

class TestStr2double:
    def test_valid(self, s):
        """str2double('3.14') -> 3.14"""
        s.eval('r = str2double("3.14");')
        r = s.workspace.get('r')
        val = float(r.data.flat[0]) if hasattr(r, 'data') else float(r)
        assert abs(val - 3.14) < 1e-10

    def test_invalid_returns_nan(self, s):
        """str2double('abc') -> NaN"""
        s.eval('r = str2double("abc");')
        r = s.workspace.get('r')
        val = float(r.data.flat[0]) if hasattr(r, 'data') else float(r)
        assert np.isnan(val), f"Expected NaN, got {val}"
