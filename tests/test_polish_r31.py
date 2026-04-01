# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish R31 -- advanced string, cell, regexp, type-checking, and matrix ops."""
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray
from forge.engine.containers import ForgeChar, ForgeCell


@pytest.fixture
def s():
    return ForgeSession()


# ---------- 1. sprintf edge cases ----------
class TestSprintfEdgeCases:
    def test_zero_padded_int(self, s):
        r = s.eval("sprintf(\"%05d\", 42)")
        assert "00042" in r

    def test_left_aligned_string(self, s):
        r = s.eval("sprintf(\"%-10s|\", \"left\")")
        assert "left      |" in r

    def test_sign_prefix_float(self, s):
        r = s.eval("sprintf(\"%+.2f\", 3.14)")
        assert "+3.14" in r

    def test_scientific_notation(self, s):
        r = s.eval("sprintf(\"%e\", 123456789)")
        assert "1.234568e+08" in r


# ---------- 2. cell array operations ----------
class TestCellArrayOps:
    def test_cell_creation(self, s):
        r = s.eval("c = cell(3,2)")
        # Should create a 3x2 cell
        c = s._engine.workspace.get("c")
        assert isinstance(c, ForgeCell)
        assert c.numel() == 6

    def test_cell_assignment(self, s):
        s.eval("c = cell(2,2); c{1,1} = \"hello\"; c{1,2} = 42")
        c = s._engine.workspace.get("c")
        assert isinstance(c.content_get(1, 1), ForgeChar)
        v = c.content_get(1, 2)
        assert float(v.data.flat[0]) == 42.0

    def test_cell_expansion_horzcat(self, s):
        r = s.eval("x = {1,2,3}; [x{:}]")
        assert "1" in r and "2" in r and "3" in r

    def test_iscell_true(self, s):
        r = s.eval("iscell(cell(2,2))")
        assert "1" in r

    def test_iscell_false(self, s):
        r = s.eval("iscell(42)")
        assert "0" in r


# ---------- 3. cellfun ----------
class TestCellfun:
    def test_cellfun_class(self, s):
        r = s.eval("cellfun(@class, {1, true}, \"UniformOutput\", false)")
        assert "double" in r
        assert "logical" in r

    def test_cellfun_ischar(self, s):
        r = s.eval("cellfun(@ischar, {\"a\", 1, \"b\"})")
        assert "1" in r and "0" in r

    def test_cellfun_numel(self, s):
        r = s.eval("cellfun(@numel, {\"abc\", \"de\", \"f\"})")
        assert "3" in r and "2" in r and "1" in r


# ---------- 4. iscellstr ----------
class TestIscellstr:
    def test_iscellstr_true(self, s):
        r = s.eval("iscellstr({\"a\",\"b\",\"c\"})")
        assert "1" in r

    def test_iscellstr_mixed(self, s):
        r = s.eval("iscellstr({\"a\", 1, \"b\"})")
        assert "0" in r

    def test_iscellstr_numeric(self, s):
        r = s.eval("iscellstr(42)")
        assert "0" in r


# ---------- 5. regexp ----------
class TestRegexp:
    def test_regexp_match_digits(self, s):
        r = s.eval("regexp(\"abc123def456\", \"\\d+\", \"match\")")
        assert "123" in r and "456" in r

    def test_regexp_tokens(self, s):
        r = s.eval("regexp(\"2024-01-15\", \"(\\d{4})-(\\d{2})-(\\d{2})\", \"tokens\")")
        assert "cell" in r.lower() or "2024" in r


# ---------- 6. regexprep ----------
class TestRegexprep:
    def test_regexprep_simple(self, s):
        r = s.eval("regexprep(\"Hello World\", \"o\", \"0\")")
        assert "Hell0 W0rld" in r

    def test_regexprep_digit_replace(self, s):
        r = s.eval("regexprep(\"foo123bar456\", \"\\d+\", \"#\")")
        assert "foo#bar#" in r

    def test_regexprep_backreference(self, s):
        r = s.eval("regexprep(\"CamelCase\", \"([a-z])([A-Z])\", \"$1_$2\")")
        assert "Camel_Case" in r


# ---------- 7. type checking ----------
class TestTypeChecking:
    def test_isnumeric_true(self, s):
        r = s.eval("isnumeric(42)")
        assert "1" in r

    def test_isnumeric_false(self, s):
        r = s.eval("isnumeric(\"hello\")")
        assert "0" in r

    def test_ischar_true(self, s):
        r = s.eval("ischar(\"hello\")")
        assert "1" in r

    def test_ischar_false(self, s):
        r = s.eval("ischar(42)")
        assert "0" in r

    def test_islogical_true(self, s):
        r = s.eval("islogical(true)")
        assert "1" in r

    def test_isfloat_true(self, s):
        r = s.eval("isfloat(3.14)")
        assert "1" in r

    def test_isinteger_true(self, s):
        r = s.eval("isinteger(int32(5))")
        assert "1" in r


# ---------- 8. triu / tril ----------
class TestTriuTril:
    def test_triu(self, s):
        s.eval("m = triu([1 2 3; 4 5 6; 7 8 9])")
        m = s._engine.workspace.get("m")
        expected = np.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]])
        np.testing.assert_array_equal(m.data, expected)

    def test_tril(self, s):
        s.eval("m = tril([1 2 3; 4 5 6; 7 8 9])")
        m = s._engine.workspace.get("m")
        expected = np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]])
        np.testing.assert_array_equal(m.data, expected)

    def test_triu_with_k(self, s):
        s.eval("m = triu([1 2 3; 4 5 6; 7 8 9], 1)")
        m = s._engine.workspace.get("m")
        expected = np.array([[0, 2, 3], [0, 0, 6], [0, 0, 0]])
        np.testing.assert_array_equal(m.data, expected)

    def test_tril_with_k(self, s):
        s.eval("m = tril([1 2 3; 4 5 6; 7 8 9], -1)")
        m = s._engine.workspace.get("m")
        expected = np.array([[0, 0, 0], [4, 0, 0], [7, 8, 0]])
        np.testing.assert_array_equal(m.data, expected)


# ---------- 9. diag ----------
class TestDiag:
    def test_diag_create(self, s):
        s.eval("m = diag([1 2 3])")
        m = s._engine.workspace.get("m")
        expected = np.diag([1, 2, 3])
        np.testing.assert_array_equal(m.data, expected)

    def test_diag_extract(self, s):
        s.eval("d = diag([1 2 3; 4 5 6; 7 8 9])")
        d = s._engine.workspace.get("d")
        np.testing.assert_array_equal(d.data.ravel(), [1, 5, 9])

    def test_diag_create_superdiag(self, s):
        s.eval("m = diag([1 2 3], 1)")
        m = s._engine.workspace.get("m")
        expected = np.diag([1, 2, 3], 1)
        np.testing.assert_array_equal(m.data, expected)

    def test_diag_create_subdiag(self, s):
        s.eval("m = diag([1 2 3], -1)")
        m = s._engine.workspace.get("m")
        expected = np.diag([1, 2, 3], -1)
        np.testing.assert_array_equal(m.data, expected)

    def test_diag_extract_superdiag(self, s):
        s.eval("d = diag([1 2 3; 4 5 6; 7 8 9], 1)")
        d = s._engine.workspace.get("d")
        np.testing.assert_array_equal(d.data.ravel(), [2, 6])
