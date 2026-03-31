# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""V&V polish round 22: string ops, matrix indexing, logical operations.

SRS trace: SRS-FUNC-001, SRS-VAL-001, SRS-COMPAT-001
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar
from forge.engine.session import ForgeSession


@pytest.fixture(scope="module")
def s():
    return ForgeSession()


def _get(s, name):
    """Get workspace variable, unwrap ForgeArray to numpy."""
    v = s.workspace.get(name)
    if isinstance(v, ForgeArray):
        return _unwrap(v)
    return v


def _get_str(s, name):
    """Get workspace variable as string (for ForgeChar)."""
    v = s.workspace.get(name)
    if isinstance(v, ForgeChar):
        return str(v)
    if isinstance(v, ForgeArray):
        arr = _unwrap(v)
        return "".join(chr(int(c)) for c in arr.flatten())
    return str(v)


# ── String operations ──────────────────────────────────────────────


class TestStringOperations:
    """String equality, concatenation, indexing, char()."""

    def test_string_equality(self, s):
        """'hello' == 'hello' -> array of 1s (element-wise)"""
        s.eval("str_eq = 'hello' == 'hello'")
        r = _get(s, "str_eq")
        expected = np.ones(5)
        np.testing.assert_array_equal(np.array(r).flatten(), expected)

    def test_string_hcat(self, s):
        """['hello' ' ' 'world'] -> 'hello world'"""
        s.eval("str_cat = ['hello' ' ' 'world']")
        r = _get_str(s, "str_cat")
        assert r == "hello world"

    def test_char_indexing_single(self, s):
        """s = 'test'; s(2) -> 'e'"""
        s.eval("cs = 'test'")
        s.eval("cs2 = cs(2)")
        r = _get_str(s, "cs2")
        assert r == "e"

    def test_char_indexing_range(self, s):
        """s = 'test'; s(2:3) -> 'es'"""
        s.eval("cs3 = 'test'")
        s.eval("cs4 = cs3(2:3)")
        r = _get_str(s, "cs4")
        assert r == "es"

    def test_char_from_codes(self, s):
        """char(65:90) -> 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'"""
        s.eval("alpha = char(65:90)")
        r = _get_str(s, "alpha")
        assert r == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def test_string_inequality(self, s):
        """'abc' == 'abd' -> [1 1 0]"""
        s.eval("str_neq = 'abc' == 'abd'")
        r = _get(s, "str_neq")
        arr = np.array(r).flatten()
        np.testing.assert_array_equal(arr, [1, 1, 0])


# ── Matrix indexing operations ─────────────────────────────────────


class TestMatrixIndexing:
    """Colon linearization, end keyword, submatrix extraction."""

    def test_colon_linearize(self, s):
        """A = magic(3); A(:) -> 9-element column vector"""
        s.eval("M3 = magic(3)")
        s.eval("Mcol = M3(:)")
        r = _get(s, "Mcol")
        arr = np.array(r).flatten()
        assert arr.size == 9
        # magic(3) = [8 1 6; 3 5 7; 4 9 2], column-major: [8,3,4,1,5,9,6,7,2]
        np.testing.assert_array_equal(arr, [8, 3, 4, 1, 5, 9, 6, 7, 2])

    def test_last_row(self, s):
        """A(end, :) -> last row of magic(3) = [4 9 2]"""
        s.eval("M3b = magic(3)")
        s.eval("Mlr = M3b(end, :)")
        r = _get(s, "Mlr")
        np.testing.assert_array_equal(np.array(r).flatten(), [4, 9, 2])

    def test_last_col(self, s):
        """A(:, end) -> last column of magic(3) = [6; 7; 2]"""
        s.eval("M3c = magic(3)")
        s.eval("Mlc = M3c(:, end)")
        r = _get(s, "Mlc")
        np.testing.assert_array_equal(np.array(r).flatten(), [6, 7, 2])

    def test_submatrix(self, s):
        """A(1:2, 1:2) -> top-left 2x2 of magic(3)"""
        s.eval("M3d = magic(3)")
        s.eval("Msub = M3d(1:2, 1:2)")
        r = _get(s, "Msub")
        arr = np.array(r)
        assert arr.shape == (2, 2)
        np.testing.assert_array_equal(arr, [[8, 1], [3, 5]])

    def test_transpose_multiply(self, s):
        """A' * A should produce correct result"""
        s.eval("M3e = magic(3)")
        s.eval("Mtp = M3e' * M3e")
        r = _get(s, "Mtp")
        arr = np.array(r)
        expected = np.array([[89, 59, 77], [59, 107, 59], [77, 59, 89]])
        np.testing.assert_array_equal(arr, expected)

    def test_end_minus_one(self, s):
        """v(end-1) -> second to last element"""
        s.eval("vend = [10 20 30 40 50]")
        s.eval("vend2 = vend(end-1)")
        r = _get(s, "vend2")
        assert float(np.array(r).flat[0]) == 40


# ── Logical operations ─────────────────────────────────────────────


class TestLogicalOperations:
    """all, any, xor, logical not."""

    def test_all_true(self, s):
        """all([1 1 1]) -> 1"""
        s.eval("at = all([1 1 1])")
        r = _get(s, "at")
        assert float(np.array(r).flat[0]) == 1

    def test_all_false(self, s):
        """all([1 0 1]) -> 0"""
        s.eval("af = all([1 0 1])")
        r = _get(s, "af")
        assert float(np.array(r).flat[0]) == 0

    def test_any_true(self, s):
        """any([0 0 1]) -> 1"""
        s.eval("ayt = any([0 0 1])")
        r = _get(s, "ayt")
        assert float(np.array(r).flat[0]) == 1

    def test_any_false(self, s):
        """any([0 0 0]) -> 0"""
        s.eval("ayf = any([0 0 0])")
        r = _get(s, "ayf")
        assert float(np.array(r).flat[0]) == 0

    def test_xor_true(self, s):
        """xor(1, 0) -> 1"""
        s.eval("xt = xor(1, 0)")
        r = _get(s, "xt")
        assert float(np.array(r).flat[0]) == 1

    def test_xor_false(self, s):
        """xor(1, 1) -> 0"""
        s.eval("xf = xor(1, 1)")
        r = _get(s, "xf")
        assert float(np.array(r).flat[0]) == 0

    def test_logical_not_vector(self, s):
        """~[1 0 1] -> [0 1 0]"""
        s.eval("ln = ~[1 0 1]")
        r = _get(s, "ln")
        np.testing.assert_array_equal(np.array(r).flatten(), [0, 1, 0])

    def test_all_empty(self, s):
        """all([]) -> 1 (MATLAB convention: vacuously true)"""
        s.eval("ae = all([])")
        r = _get(s, "ae")
        assert float(np.array(r).flat[0]) == 1
