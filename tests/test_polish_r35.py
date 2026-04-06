# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""V&V polish round 35 -- strtrim, textscan, sscanf, struct, num2cell, setdiff, sortrows.

SRS trace: SRS-FUNC-001 (Octave-compatible function library)

V&V Traceability (backfill)
===========================
R-POL35-01: strtrim SHALL strip leading and trailing whitespace from strings
            and cell arrays of strings.

    Model-user argument: Data read from CSV files or user input often has
    extraneous whitespace. An engineer uses strtrim to clean field values
    before comparison or storage. If whitespace is not stripped, string
    comparisons fail silently.

    Decomposition:
      R-POL35-01a: strtrim("  hello  ") returns "hello".
      R-POL35-01b: strtrim({"  a  ", "  b  "}) trims each cell element.
      R-POL35-01c: strtrim("abc") returns "abc" unchanged.

    Consistency: Single string (01a), cell array (01b), and no-op (01c)
    cover the strtrim API.

R-POL35-02: textscan SHALL parse formatted text strings into cell arrays of
            typed columns, supporting custom and whitespace delimiters.

    Model-user argument: Scientists parsing CSV or space-delimited instrument
    output use textscan with format strings like "%f%f%f". If column parsing
    is wrong, data columns are misaligned and analysis produces nonsense.

    Decomposition:
      R-POL35-02a: textscan with comma delimiter parses three float columns.
      R-POL35-02b: textscan with named format parses string and integer.
      R-POL35-02c: textscan with whitespace delimiter parses three integers.

    Consistency: Custom delimiter (02a), mixed-type format (02b), and
    whitespace delimiter (02c) cover the primary textscan patterns.

R-POL35-03: sscanf SHALL parse formatted strings into numeric arrays or
            mixed cell results.

    Model-user argument: Engineers use sscanf to extract numeric values from
    headers or metadata strings. Both pure-numeric and mixed (number + string)
    formats must work for flexible parsing.

    Decomposition:
      R-POL35-03a: sscanf("3.14", "%f") returns 3.14.
      R-POL35-03b: sscanf("42", "%d") returns 42.
      R-POL35-03c: sscanf("42 hello", "%d %s") returns mixed cell result.

    Consistency: Float (03a), integer (03b), and mixed (03c) formats cover
    the sscanf API.

R-POL35-04: struct() SHALL create structures with named fields, support
            struct arrays via cell-valued fields, and support empty structs.

    Model-user argument: Structs are the primary named-field container in
    Octave, used for configuration, results, and record-oriented data.
    Struct arrays (one struct per record) are essential for tabular data
    workflows.

    Decomposition:
      R-POL35-04a: struct("x", 1, "y", 2) creates a struct with fields x and y.
      R-POL35-04b: struct("name", {"a","b"}, "val", {1,2}) creates a struct array.
      R-POL35-04c: struct() creates an empty struct with no fields.

    Consistency: Scalar struct (04a), struct array (04b), and empty struct
    (04c) cover the struct creation API.

R-POL35-05: num2cell SHALL convert numeric arrays to cell arrays, optionally
            splitting along a specified dimension.

    Model-user argument: Engineers converting matrices to cell arrays for
    cellfun processing use num2cell. The dimension argument controls whether
    the result contains column vectors (dim=1) or row vectors (dim=2).

    Decomposition:
      R-POL35-05a: num2cell([1 2 3]) produces a 1x3 cell of scalars.
      R-POL35-05b: num2cell([1 2; 3 4]) produces a 2x2 cell of scalars.
      R-POL35-05c: num2cell(A, 1) splits into column vectors.
      R-POL35-05d: num2cell(A, 2) splits into row vectors.

    Consistency: 1D (05a), 2D element-wise (05b), column split (05c), and
    row split (05d) cover the num2cell API.

R-POL35-06: setdiff with 'rows' option SHALL return rows in the first matrix
            that are not in the second.

    Model-user argument: Scientists comparing experiment batches use setdiff
    with 'rows' to find new data points. If row comparison is wrong,
    duplicate or missing records go undetected.

    Decomposition:
      R-POL35-06a: setdiff(A, B, 'rows') returns rows in A not in B.

    Consistency: Single sub-requirement covers the row-wise setdiff.

R-POL35-07: sortrows SHALL sort matrix rows by specified columns, defaulting
            to column 1.

    Model-user argument: Engineers sort tabular data by a key column (e.g.,
    sort sensor readings by timestamp). If the column specifier is ignored
    or sorting is wrong, data ordering is corrupted.

    Decomposition:
      R-POL35-07a: sortrows(A, 2) sorts by column 2.
      R-POL35-07b: sortrows(A) sorts by column 1 by default.

    Consistency: Explicit column (07a) and default column (07b) cover
    sortrows.
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
    """R-POL35-01: strtrim SHALL strip leading and trailing whitespace from
    strings and cell arrays of strings.

    Model-user argument: Data read from CSV files or user input often has
    extraneous whitespace. An engineer uses strtrim to clean field values
    before comparison or storage. If whitespace is not stripped, string
    comparisons fail silently.

    Decomposition:
      R-POL35-01a: strtrim("  hello  ") returns "hello".
      R-POL35-01b: strtrim({"  a  ", "  b  "}) trims each cell element.
      R-POL35-01c: strtrim("abc") returns "abc" unchanged.

    Consistency: Single string (01a), cell array (01b), and no-op (01c)
    cover the strtrim API.
    """

    def test_strtrim_basic(self, S):
        """R-POL35-01a: strtrim('  hello  ') SHALL return 'hello'."""
        S.eval('r35_st1 = strtrim("  hello  ");')
        v = _var(S, "r35_st1")
        assert isinstance(v, ForgeChar)
        assert v.to_str() == "hello"

    def test_strtrim_cell(self, S):
        """R-POL35-01b: strtrim({'  a  ', '  b  '}) SHALL trim each cell element."""
        S.eval('r35_st2 = strtrim({"  a  ", "  b  "});')
        v = _var(S, "r35_st2")
        assert isinstance(v, ForgeCell)
        assert v._data[0].to_str() == "a"
        assert v._data[1].to_str() == "b"

    def test_strtrim_no_whitespace(self, S):
        """R-POL35-01c: strtrim('abc') SHALL return 'abc' unchanged."""
        S.eval('r35_st3 = strtrim("abc");')
        v = _var(S, "r35_st3")
        assert isinstance(v, ForgeChar)
        assert v.to_str() == "abc"


# ── textscan ─────────────────────────────────────────────────────

class TestTextscan:
    """R-POL35-02: textscan SHALL parse formatted text strings into cell
    arrays of typed columns.

    Model-user argument: Scientists parsing CSV or space-delimited instrument
    output use textscan with format strings like "%f%f%f". If column parsing
    is wrong, data columns are misaligned and analysis produces nonsense.

    Decomposition:
      R-POL35-02a: textscan with comma delimiter parses three float columns.
      R-POL35-02b: textscan with named format parses string and integer.
      R-POL35-02c: textscan with whitespace delimiter parses three integers.

    Consistency: Custom delimiter (02a), mixed-type format (02b), and
    whitespace delimiter (02c) cover the primary textscan patterns.
    """

    def test_textscan_csv_delimiter(self, S):
        """R-POL35-02a: textscan with comma delimiter SHALL parse three float columns."""
        S.eval(r'r35_ts1 = textscan("1,2,3\n4,5,6", "%f%f%f", "Delimiter", ",");')
        v = _var(S, "r35_ts1")
        assert isinstance(v, ForgeCell)
        assert len(v._data) == 3
        np.testing.assert_array_almost_equal(v._data[0].data.flatten(), [1, 4])
        np.testing.assert_array_almost_equal(v._data[1].data.flatten(), [2, 5])
        np.testing.assert_array_almost_equal(v._data[2].data.flatten(), [3, 6])

    def test_textscan_named_format(self, S):
        """R-POL35-02b: textscan with named format SHALL parse string and integer."""
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
        """R-POL35-02c: textscan with whitespace delimiter SHALL parse three integers."""
        S.eval('r35_ts3 = textscan("10 20 30", "%d %d %d");')
        v = _var(S, "r35_ts3")
        assert isinstance(v, ForgeCell)
        assert len(v._data) == 3
        assert float(v._data[0].data.flat[0]) == 10.0
        assert float(v._data[1].data.flat[0]) == 20.0
        assert float(v._data[2].data.flat[0]) == 30.0


# ── sscanf ───────────────────────────────────────────────────────

class TestSscanf:
    """R-POL35-03: sscanf SHALL parse formatted strings into numeric arrays
    or mixed cell results.

    Model-user argument: Engineers use sscanf to extract numeric values from
    headers or metadata strings. Both pure-numeric and mixed formats must
    work for flexible parsing.

    Decomposition:
      R-POL35-03a: sscanf("3.14", "%f") returns 3.14.
      R-POL35-03b: sscanf("42", "%d") returns 42.
      R-POL35-03c: sscanf("42 hello", "%d %s") returns mixed cell result.

    Consistency: Float (03a), integer (03b), and mixed (03c) formats cover
    the sscanf API.
    """

    def test_sscanf_float(self, S):
        """R-POL35-03a: sscanf('3.14', '%f') SHALL return 3.14."""
        S.eval('r35_sf1 = sscanf("3.14", "%f");')
        v = _var(S, "r35_sf1")
        assert isinstance(v, ForgeArray)
        np.testing.assert_almost_equal(float(v.data.flat[0]), 3.14)

    def test_sscanf_int(self, S):
        """R-POL35-03b: sscanf('42', '%d') SHALL return 42."""
        S.eval('r35_sf2 = sscanf("42", "%d");')
        v = _var(S, "r35_sf2")
        assert isinstance(v, ForgeArray)
        assert float(v.data.flat[0]) == 42.0

    def test_sscanf_mixed(self, S):
        """R-POL35-03c: sscanf('42 hello', '%d %s') SHALL return mixed cell result."""
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
    """R-POL35-04: struct() SHALL create structures with named fields,
    support struct arrays, and support empty structs.

    Model-user argument: Structs are the primary named-field container in
    Octave, used for configuration, results, and record-oriented data.
    Struct arrays are essential for tabular data workflows.

    Decomposition:
      R-POL35-04a: struct("x", 1, "y", 2) creates a struct with fields x and y.
      R-POL35-04b: struct("name", {"a","b"}, "val", {1,2}) creates a struct array.
      R-POL35-04c: struct() creates an empty struct with no fields.

    Consistency: Scalar struct (04a), struct array (04b), and empty struct
    (04c) cover the struct creation API.
    """

    def test_struct_basic(self, S):
        """R-POL35-04a: struct('x', 1, 'y', 2) SHALL create a struct with fields x and y."""
        S.eval('r35_s1 = struct("x", 1, "y", 2);')
        S.eval("r35_s1x = r35_s1.x;")
        S.eval("r35_s1y = r35_s1.y;")
        assert _scalar(S, "r35_s1x") == 1.0
        assert _scalar(S, "r35_s1y") == 2.0

    def test_struct_array(self, S):
        """R-POL35-04b: struct with cell-valued fields SHALL create a struct array."""
        S.eval('r35_sa = struct("name", {"a","b"}, "val", {1,2});')
        S.eval("r35_sa1n = r35_sa(1).name;")
        v1 = _var(S, "r35_sa1n")
        assert isinstance(v1, ForgeChar)
        assert v1.to_str() == "a"
        S.eval("r35_sa2v = r35_sa(2).val;")
        assert _scalar(S, "r35_sa2v") == 2.0

    def test_struct_empty(self, S):
        """R-POL35-04c: struct() SHALL create an empty struct with no fields."""
        S.eval("r35_se = struct();")
        S.eval("r35_sef = fieldnames(r35_se);")
        v = _var(S, "r35_sef")
        assert isinstance(v, ForgeCell)
        assert v.numel() == 0


# ── num2cell ─────────────────────────────────────────────────────

class TestNum2cell:
    """R-POL35-05: num2cell SHALL convert numeric arrays to cell arrays,
    optionally splitting along a specified dimension.

    Model-user argument: Engineers converting matrices to cell arrays for
    cellfun processing use num2cell. The dimension argument controls whether
    the result contains column vectors or row vectors.

    Decomposition:
      R-POL35-05a: num2cell([1 2 3]) produces a 1x3 cell of scalars.
      R-POL35-05b: num2cell([1 2; 3 4]) produces a 2x2 cell of scalars.
      R-POL35-05c: num2cell(A, 1) splits into column vectors.
      R-POL35-05d: num2cell(A, 2) splits into row vectors.

    Consistency: 1D (05a), 2D element-wise (05b), column split (05c), and
    row split (05d) cover the num2cell API.
    """

    def test_num2cell_1d(self, S):
        """R-POL35-05a: num2cell([1 2 3]) SHALL produce a 1x3 cell of scalars."""
        S.eval("r35_nc1 = num2cell([1 2 3]);")
        v = _var(S, "r35_nc1")
        assert isinstance(v, ForgeCell)
        assert v.numel() == 3
        for i, expected in enumerate([1, 2, 3]):
            assert float(v._data[i].data.flat[0]) == expected

    def test_num2cell_2d(self, S):
        """R-POL35-05b: num2cell([1 2; 3 4]) SHALL produce a 2x2 cell of scalars."""
        S.eval("r35_nc2 = num2cell([1 2; 3 4]);")
        v = _var(S, "r35_nc2")
        assert isinstance(v, ForgeCell)
        assert v.numel() == 4

    def test_num2cell_dim1(self, S):
        """R-POL35-05c: num2cell(A, 1) SHALL split into column vectors."""
        S.eval("r35_nc3 = num2cell([1 2; 3 4], 1);")
        v = _var(S, "r35_nc3")
        assert isinstance(v, ForgeCell)
        assert v.numel() == 2
        np.testing.assert_array_equal(v._data[0].data.flatten(), [1, 3])
        np.testing.assert_array_equal(v._data[1].data.flatten(), [2, 4])

    def test_num2cell_dim2(self, S):
        """R-POL35-05d: num2cell(A, 2) SHALL split into row vectors."""
        S.eval("r35_nc4 = num2cell([1 2; 3 4], 2);")
        v = _var(S, "r35_nc4")
        assert isinstance(v, ForgeCell)
        assert v.numel() == 2
        np.testing.assert_array_equal(v._data[0].data.flatten(), [1, 2])
        np.testing.assert_array_equal(v._data[1].data.flatten(), [3, 4])


# ── setdiff ──────────────────────────────────────────────────────

class TestSetdiffRows:
    """R-POL35-06: setdiff with 'rows' option SHALL return rows in the first
    matrix that are not in the second.

    Model-user argument: Scientists comparing experiment batches use setdiff
    with 'rows' to find new data points.

    Decomposition:
      R-POL35-06a: setdiff(A, B, 'rows') returns rows in A not in B.

    Consistency: Single sub-requirement covers the row-wise setdiff.
    """

    def test_setdiff_rows(self, S):
        """R-POL35-06a: setdiff(A, B, 'rows') SHALL return rows in A not in B."""
        S.eval('r35_sd = setdiff([1 2; 3 4; 1 2], [1 2], "rows");')
        v = _var(S, "r35_sd")
        assert isinstance(v, ForgeArray)
        np.testing.assert_array_equal(v.data.flatten(), [3, 4])


# ── sortrows ─────────────────────────────────────────────────────

class TestSortrows:
    """R-POL35-07: sortrows SHALL sort matrix rows by specified columns,
    defaulting to column 1.

    Model-user argument: Engineers sort tabular data by a key column. If
    the column specifier is ignored or sorting is wrong, data ordering
    is corrupted.

    Decomposition:
      R-POL35-07a: sortrows(A, 2) sorts by column 2.
      R-POL35-07b: sortrows(A) sorts by column 1 by default.

    Consistency: Explicit column (07a) and default column (07b) cover
    sortrows.
    """

    def test_sortrows_col_spec(self, S):
        """R-POL35-07a: sortrows(A, 2) SHALL sort by column 2."""
        S.eval("r35_sr1 = sortrows([3 1; 1 3; 2 2], 2);")
        v = _var(S, "r35_sr1")
        assert isinstance(v, ForgeArray)
        expected = np.array([[3, 1], [2, 2], [1, 3]])
        np.testing.assert_array_equal(v.data, expected)

    def test_sortrows_default(self, S):
        """R-POL35-07b: sortrows(A) SHALL sort by column 1 by default."""
        S.eval("r35_sr2 = sortrows([3 1; 1 3; 2 2]);")
        v = _var(S, "r35_sr2")
        assert isinstance(v, ForgeArray)
        expected = np.array([[1, 3], [2, 2], [3, 1]])
        np.testing.assert_array_equal(v.data, expected)
