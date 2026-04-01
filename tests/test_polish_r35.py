# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""V&V polish round 35 — strtrim, textscan, sscanf, struct, num2cell, setdiff, sortrows.

SRS trace: SRS-FUNC-001 (Octave-compatible function library)
"""

import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeCell, ForgeChar


@pytest.fixture(scope="module")
def S():
    return ForgeSession()


def _var(session, name):
    """Get raw workspace variable."""
    return session.workspace.get(name)


def _scalar(session, name):
    """Get scalar float from workspace variable."""
    return float(np.asarray(_unwrap(session.workspace.get(name))).ravel()[0])


# ── strtrim ──────────────────────────────────────────────────────

class TestStrtrim:
    def test_strtrim_basic(self, S):
        S.eval('r35_st1 = strtrim("  hello  ");')
        v = _var(S, "r35_st1")
        assert isinstance(v, ForgeChar)
        assert v.to_str() == "hello"

    def test_strtrim_cell(self, S):
        S.eval('r35_st2 = strtrim({"  a  ", "  b  "});')
        v = _var(S, "r35_st2")
        assert isinstance(v, ForgeCell)
        assert v._data[0].to_str() == "a"
        assert v._data[1].to_str() == "b"

    def test_strtrim_no_whitespace(self, S):
        S.eval('r35_st3 = strtrim("abc");')
        v = _var(S, "r35_st3")
        assert isinstance(v, ForgeChar)
        assert v.to_str() == "abc"


# ── textscan ─────────────────────────────────────────────────────

class TestTextscan:
    def test_textscan_csv_delimiter(self, S):
        S.eval(r'r35_ts1 = textscan("1,2,3\n4,5,6", "%f%f%f", "Delimiter", ",");')
        v = _var(S, "r35_ts1")
        assert isinstance(v, ForgeCell)
        assert len(v._data) == 3
        np.testing.assert_array_almost_equal(v._data[0].data.flatten(), [1, 4])
        np.testing.assert_array_almost_equal(v._data[1].data.flatten(), [2, 5])
        np.testing.assert_array_almost_equal(v._data[2].data.flatten(), [3, 6])

    def test_textscan_named_format(self, S):
        S.eval('r35_ts2 = textscan("name:John age:30", "name:%s age:%d");')
        v = _var(S, "r35_ts2")
        assert isinstance(v, ForgeCell)
        assert len(v._data) == 2
        # First column: "John"
        col0 = v._data[0]
        if isinstance(col0, ForgeCell):
            assert col0._data[0].to_str() == "John"
        # Second column: 30
        col1 = v._data[1]
        assert isinstance(col1, ForgeArray)
        assert float(col1.data.flat[0]) == 30.0

    def test_textscan_whitespace_delim(self, S):
        S.eval('r35_ts3 = textscan("10 20 30", "%d %d %d");')
        v = _var(S, "r35_ts3")
        assert isinstance(v, ForgeCell)
        assert len(v._data) == 3
        assert float(v._data[0].data.flat[0]) == 10.0
        assert float(v._data[1].data.flat[0]) == 20.0
        assert float(v._data[2].data.flat[0]) == 30.0


# ── sscanf ───────────────────────────────────────────────────────

class TestSscanf:
    def test_sscanf_float(self, S):
        S.eval('r35_sf1 = sscanf("3.14", "%f");')
        v = _var(S, "r35_sf1")
        assert isinstance(v, ForgeArray)
        np.testing.assert_almost_equal(float(v.data.flat[0]), 3.14)

    def test_sscanf_int(self, S):
        S.eval('r35_sf2 = sscanf("42", "%d");')
        v = _var(S, "r35_sf2")
        assert isinstance(v, ForgeArray)
        assert float(v.data.flat[0]) == 42.0

    def test_sscanf_mixed(self, S):
        S.eval('r35_sf3 = sscanf("42 hello", "%d %s");')
        v = _var(S, "r35_sf3")
        assert isinstance(v, ForgeCell)
        item0 = v._data[0]
        assert isinstance(item0, ForgeArray)
        assert float(item0.data.flat[0]) == 42.0
        item1 = v._data[1]
        assert isinstance(item1, ForgeChar)
        assert item1.to_str() == "hello"


# ── struct ───────────────────────────────────────────────────────

class TestStruct:
    def test_struct_basic(self, S):
        S.eval('r35_s1 = struct("x", 1, "y", 2);')
        S.eval("r35_s1x = r35_s1.x;")
        S.eval("r35_s1y = r35_s1.y;")
        assert _scalar(S, "r35_s1x") == 1.0
        assert _scalar(S, "r35_s1y") == 2.0

    def test_struct_array(self, S):
        S.eval('r35_sa = struct("name", {"a","b"}, "val", {1,2});')
        S.eval("r35_sa1n = r35_sa(1).name;")
        v1 = _var(S, "r35_sa1n")
        assert isinstance(v1, ForgeChar)
        assert v1.to_str() == "a"
        S.eval("r35_sa2v = r35_sa(2).val;")
        assert _scalar(S, "r35_sa2v") == 2.0

    def test_struct_empty(self, S):
        S.eval("r35_se = struct();")
        S.eval("r35_sef = fieldnames(r35_se);")
        v = _var(S, "r35_sef")
        assert isinstance(v, ForgeCell)
        assert v.numel() == 0


# ── num2cell ─────────────────────────────────────────────────────

class TestNum2cell:
    def test_num2cell_1d(self, S):
        S.eval("r35_nc1 = num2cell([1 2 3]);")
        v = _var(S, "r35_nc1")
        assert isinstance(v, ForgeCell)
        assert v.numel() == 3
        for i, expected in enumerate([1, 2, 3]):
            assert float(v._data[i].data.flat[0]) == expected

    def test_num2cell_2d(self, S):
        S.eval("r35_nc2 = num2cell([1 2; 3 4]);")
        v = _var(S, "r35_nc2")
        assert isinstance(v, ForgeCell)
        assert v.numel() == 4

    def test_num2cell_dim1(self, S):
        """num2cell(A, 1) splits into column vectors."""
        S.eval("r35_nc3 = num2cell([1 2; 3 4], 1);")
        v = _var(S, "r35_nc3")
        assert isinstance(v, ForgeCell)
        assert v.numel() == 2
        np.testing.assert_array_equal(v._data[0].data.flatten(), [1, 3])
        np.testing.assert_array_equal(v._data[1].data.flatten(), [2, 4])

    def test_num2cell_dim2(self, S):
        """num2cell(A, 2) splits into row vectors."""
        S.eval("r35_nc4 = num2cell([1 2; 3 4], 2);")
        v = _var(S, "r35_nc4")
        assert isinstance(v, ForgeCell)
        assert v.numel() == 2
        np.testing.assert_array_equal(v._data[0].data.flatten(), [1, 2])
        np.testing.assert_array_equal(v._data[1].data.flatten(), [3, 4])


# ── setdiff ──────────────────────────────────────────────────────

class TestSetdiffRows:
    def test_setdiff_rows(self, S):
        S.eval('r35_sd = setdiff([1 2; 3 4; 1 2], [1 2], "rows");')
        v = _var(S, "r35_sd")
        assert isinstance(v, ForgeArray)
        np.testing.assert_array_equal(v.data.flatten(), [3, 4])


# ── sortrows ─────────────────────────────────────────────────────

class TestSortrows:
    def test_sortrows_col_spec(self, S):
        S.eval("r35_sr1 = sortrows([3 1; 1 3; 2 2], 2);")
        v = _var(S, "r35_sr1")
        assert isinstance(v, ForgeArray)
        expected = np.array([[3, 1], [2, 2], [1, 3]])
        np.testing.assert_array_equal(v.data, expected)

    def test_sortrows_default(self, S):
        S.eval("r35_sr2 = sortrows([3 1; 1 3; 2 2]);")
        v = _var(S, "r35_sr2")
        assert isinstance(v, ForgeArray)
        expected = np.array([[1, 3], [2, 2], [3, 1]])
        np.testing.assert_array_equal(v.data, expected)
