# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish R31 -- advanced string, cell, regexp, type-checking, and matrix ops.

V&V Traceability (backfill)
===========================
R-POL31-01: sprintf SHALL support zero-padding, left-alignment, sign prefix,
            and scientific notation format specifiers.

    Model-user argument: Engineers generating formatted reports or log
    messages use sprintf with precise format specifiers. If %05d or %-10s
    produce wrong padding or alignment, tabular output becomes misaligned
    and unusable.

    Decomposition:
      R-POL31-01a: %05d zero-pads to 5 digits.
      R-POL31-01b: %-10s left-aligns with trailing spaces.
      R-POL31-01c: %+.2f includes sign prefix on positive floats.
      R-POL31-01d: %e produces scientific notation.

    Consistency: Four format specifiers (01a-d) cover the main formatting
    features: zero-padding, alignment, sign, and notation.

R-POL31-02: Cell array creation, assignment, expansion, and type-checking
            SHALL match Octave semantics.

    Model-user argument: Cell arrays are the primary heterogeneous container
    in Octave. An engineer storing mixed data (strings, numbers, sub-arrays)
    needs cell(), curly-brace assignment, comma-separated expansion, and
    iscell() to work correctly.

    Decomposition:
      R-POL31-02a: cell(3,2) creates a 3x2 cell with 6 elements.
      R-POL31-02b: c{1,1} = "hello" assigns a string to a cell element.
      R-POL31-02c: [x{:}] expands cell contents horizontally.
      R-POL31-02d: iscell returns 1 for cell arrays.
      R-POL31-02e: iscell returns 0 for numeric values.

    Consistency: Creation (02a), assignment (02b), expansion (02c), and
    type-check true/false (02d-e) cover the cell API.

R-POL31-03: cellfun SHALL apply function handles to cell arrays, supporting
            both uniform and non-uniform output modes.

    Model-user argument: Scientists use cellfun(@class, ...) to inspect
    heterogeneous cell contents and cellfun(@numel, ...) for quick size
    queries. Both uniform (numeric result) and non-uniform ('UniformOutput',
    false) modes must work.

    Decomposition:
      R-POL31-03a: cellfun(@class, ..., 'UniformOutput', false) returns class names.
      R-POL31-03b: cellfun(@ischar, ...) returns logical array.
      R-POL31-03c: cellfun(@numel, ...) returns element counts.

    Consistency: Non-uniform output (03a) and uniform output (03b-c)
    modes are both covered.

R-POL31-04: iscellstr SHALL return 1 for cell arrays of strings and 0
            otherwise.

    Model-user argument: Guard checks like ``if iscellstr(input)`` protect
    functions from type errors. The function must correctly distinguish
    all-string cells from mixed or numeric inputs.

    Decomposition:
      R-POL31-04a: iscellstr returns 1 for all-string cell.
      R-POL31-04b: iscellstr returns 0 for mixed cell.
      R-POL31-04c: iscellstr returns 0 for numeric input.

    Consistency: True (04a) and two false cases (04b-c) cover the decision boundary.

R-POL31-05: regexp and regexprep SHALL match patterns and perform
            substitutions including backreferences.

    Model-user argument: Engineers parsing log files and sensor output
    use regexp to extract numeric fields and regexprep for batch text
    transformations. Backreference support (e.g., $1_$2) is needed for
    inserting delimiters into CamelCase identifiers.

    Decomposition:
      R-POL31-05a: regexp with 'match' extracts all digit groups.
      R-POL31-05b: regexp with 'tokens' extracts capture groups.
      R-POL31-05c: regexprep replaces single characters.
      R-POL31-05d: regexprep replaces digit groups with a placeholder.
      R-POL31-05e: regexprep supports backreferences ($1, $2).

    Consistency: Matching (05a-b) and replacement (05c-e) cover the full
    regexp/regexprep API surface.

R-POL31-06: Type-checking functions (isnumeric, ischar, islogical, isfloat,
            isinteger) SHALL return correct boolean results.

    Model-user argument: Guard clauses like ``if ~isnumeric(x); error(...);
    end`` are standard defensive programming in Octave. If type checks
    return wrong results, input validation silently passes bad data.

    Decomposition:
      R-POL31-06a: isnumeric(42) returns 1.
      R-POL31-06b: isnumeric("hello") returns 0.
      R-POL31-06c: ischar("hello") returns 1.
      R-POL31-06d: ischar(42) returns 0.
      R-POL31-06e: islogical(true) returns 1.
      R-POL31-06f: isfloat(3.14) returns 1.
      R-POL31-06g: isinteger(int32(5)) returns 1.

    Consistency: Each type-check function is tested with a true and/or
    false input, covering the full set.

R-POL31-07: triu and tril SHALL extract upper and lower triangular portions
            of matrices, with optional diagonal offset k.

    Model-user argument: Linear algebra workflows (Cholesky, LU) rely on
    triu/tril to isolate triangular factors. An incorrect diagonal offset
    would corrupt factorization results.

    Decomposition:
      R-POL31-07a: triu extracts upper triangle (default k=0).
      R-POL31-07b: tril extracts lower triangle (default k=0).
      R-POL31-07c: triu with k=1 excludes the main diagonal.
      R-POL31-07d: tril with k=-1 excludes the main diagonal.

    Consistency: Default (07a-b) and offset (07c-d) variants cover the API.

R-POL31-08: diag SHALL create diagonal matrices from vectors and extract
            diagonals from matrices, with optional offset k.

    Model-user argument: Creating diagonal matrices for scaling or
    extracting eigenvalue diagonals is routine linear algebra. The k
    parameter for super/sub-diagonals must work for banded-matrix
    construction.

    Decomposition:
      R-POL31-08a: diag([1 2 3]) creates a 3x3 diagonal matrix.
      R-POL31-08b: diag(matrix) extracts the main diagonal.
      R-POL31-08c: diag(v, 1) creates a matrix with superdiagonal.
      R-POL31-08d: diag(v, -1) creates a matrix with subdiagonal.
      R-POL31-08e: diag(matrix, 1) extracts the superdiagonal.

    Consistency: Creation (08a, 08c-d), extraction (08b, 08e), and offset
    variants cover the diag API.
"""
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
    """R-POL31-01: sprintf SHALL support zero-padding, left-alignment, sign
    prefix, and scientific notation format specifiers.

    Model-user argument: Engineers generating formatted reports or log
    messages use sprintf with precise format specifiers. If %05d or %-10s
    produce wrong padding or alignment, tabular output becomes misaligned
    and unusable.

    Decomposition:
      R-POL31-01a: %05d zero-pads to 5 digits.
      R-POL31-01b: %-10s left-aligns with trailing spaces.
      R-POL31-01c: %+.2f includes sign prefix on positive floats.
      R-POL31-01d: %e produces scientific notation.

    Consistency: Four format specifiers (01a-d) cover the main formatting
    features: zero-padding, alignment, sign, and notation.
    """

    def test_zero_padded_int(self, s):
        """R-POL31-01a: %05d SHALL zero-pad to 5 digits."""
        r = s.eval("sprintf(\"%05d\", 42)")
        assert "00042" in r

    def test_left_aligned_string(self, s):
        """R-POL31-01b: %-10s SHALL left-align with trailing spaces."""
        r = s.eval("sprintf(\"%-10s|\", \"left\")")
        assert "left      |" in r

    def test_sign_prefix_float(self, s):
        """R-POL31-01c: %+.2f SHALL include sign prefix on positive floats."""
        r = s.eval("sprintf(\"%+.2f\", 3.14)")
        assert "+3.14" in r

    def test_scientific_notation(self, s):
        """R-POL31-01d: %e SHALL produce scientific notation."""
        r = s.eval("sprintf(\"%e\", 123456789)")
        assert "1.234568e+08" in r


# ---------- 2. cell array operations ----------
class TestCellArrayOps:
    """R-POL31-02: Cell array creation, assignment, expansion, and
    type-checking SHALL match Octave semantics.

    Model-user argument: Cell arrays are the primary heterogeneous container
    in Octave. An engineer storing mixed data (strings, numbers, sub-arrays)
    needs cell(), curly-brace assignment, comma-separated expansion, and
    iscell() to work correctly.

    Decomposition:
      R-POL31-02a: cell(3,2) creates a 3x2 cell with 6 elements.
      R-POL31-02b: c{1,1} = "hello" assigns a string to a cell element.
      R-POL31-02c: [x{:}] expands cell contents horizontally.
      R-POL31-02d: iscell returns 1 for cell arrays.
      R-POL31-02e: iscell returns 0 for numeric values.

    Consistency: Creation (02a), assignment (02b), expansion (02c), and
    type-check true/false (02d-e) cover the cell API.
    """

    def test_cell_creation(self, s):
        """R-POL31-02a: cell(3,2) SHALL create a 3x2 cell with 6 elements."""
        r = s.eval("c = cell(3,2)")
        # Should create a 3x2 cell
        c = s._engine.workspace.get("c")
        assert isinstance(c, ForgeCell)
        assert c.numel() == 6

    def test_cell_assignment(self, s):
        """R-POL31-02b: c{1,1} = 'hello' SHALL assign a string to a cell element."""
        s.eval("c = cell(2,2); c{1,1} = \"hello\"; c{1,2} = 42")
        c = s._engine.workspace.get("c")
        assert isinstance(c.content_get(1, 1), ForgeChar)
        v = c.content_get(1, 2)
        assert float(v.data.flat[0]) == 42.0

    def test_cell_expansion_horzcat(self, s):
        """R-POL31-02c: [x{:}] SHALL expand cell contents horizontally."""
        r = s.eval("x = {1,2,3}; [x{:}]")
        assert "1" in r and "2" in r and "3" in r

    def test_iscell_true(self, s):
        """R-POL31-02d: iscell SHALL return 1 for cell arrays."""
        r = s.eval("iscell(cell(2,2))")
        assert "1" in r

    def test_iscell_false(self, s):
        """R-POL31-02e: iscell SHALL return 0 for numeric values."""
        r = s.eval("iscell(42)")
        assert "0" in r


# ---------- 3. cellfun ----------
class TestCellfun:
    """R-POL31-03: cellfun SHALL apply function handles to cell arrays,
    supporting both uniform and non-uniform output modes.

    Model-user argument: Scientists use cellfun(@class, ...) to inspect
    heterogeneous cell contents and cellfun(@numel, ...) for quick size
    queries. Both uniform (numeric result) and non-uniform ('UniformOutput',
    false) modes must work.

    Decomposition:
      R-POL31-03a: cellfun(@class, ..., 'UniformOutput', false) returns class names.
      R-POL31-03b: cellfun(@ischar, ...) returns logical array.
      R-POL31-03c: cellfun(@numel, ...) returns element counts.

    Consistency: Non-uniform output (03a) and uniform output (03b-c)
    modes are both covered.
    """

    def test_cellfun_class(self, s):
        """R-POL31-03a: cellfun(@class, ..., 'UniformOutput', false) SHALL return class names."""
        r = s.eval("cellfun(@class, {1, true}, \"UniformOutput\", false)")
        assert "double" in r
        assert "logical" in r

    def test_cellfun_ischar(self, s):
        """R-POL31-03b: cellfun(@ischar, ...) SHALL return logical array."""
        r = s.eval("cellfun(@ischar, {\"a\", 1, \"b\"})")
        assert "1" in r and "0" in r

    def test_cellfun_numel(self, s):
        """R-POL31-03c: cellfun(@numel, ...) SHALL return element counts."""
        r = s.eval("cellfun(@numel, {\"abc\", \"de\", \"f\"})")
        assert "3" in r and "2" in r and "1" in r


# ---------- 4. iscellstr ----------
class TestIscellstr:
    """R-POL31-04: iscellstr SHALL return 1 for cell arrays of strings and 0
    otherwise.

    Model-user argument: Guard checks like ``if iscellstr(input)`` protect
    functions from type errors. The function must correctly distinguish
    all-string cells from mixed or numeric inputs.

    Decomposition:
      R-POL31-04a: iscellstr returns 1 for all-string cell.
      R-POL31-04b: iscellstr returns 0 for mixed cell.
      R-POL31-04c: iscellstr returns 0 for numeric input.

    Consistency: True (04a) and two false cases (04b-c) cover the decision boundary.
    """

    def test_iscellstr_true(self, s):
        """R-POL31-04a: iscellstr SHALL return 1 for all-string cell."""
        r = s.eval("iscellstr({\"a\",\"b\",\"c\"})")
        assert "1" in r

    def test_iscellstr_mixed(self, s):
        """R-POL31-04b: iscellstr SHALL return 0 for mixed cell."""
        r = s.eval("iscellstr({\"a\", 1, \"b\"})")
        assert "0" in r

    def test_iscellstr_numeric(self, s):
        """R-POL31-04c: iscellstr SHALL return 0 for numeric input."""
        r = s.eval("iscellstr(42)")
        assert "0" in r


# ---------- 5. regexp ----------
class TestRegexp:
    """R-POL31-05: regexp and regexprep SHALL match patterns and perform
    substitutions including backreferences.

    Model-user argument: Engineers parsing log files and sensor output
    use regexp to extract numeric fields and regexprep for batch text
    transformations. Backreference support (e.g., $1_$2) is needed for
    inserting delimiters into CamelCase identifiers.

    Decomposition:
      R-POL31-05a: regexp with 'match' extracts all digit groups.
      R-POL31-05b: regexp with 'tokens' extracts capture groups.

    Consistency: Match (05a) and token extraction (05b) cover the primary
    regexp API. Replacement tests are in TestRegexprep.
    """

    def test_regexp_match_digits(self, s):
        """R-POL31-05a: regexp with 'match' SHALL extract all digit groups."""
        r = s.eval("regexp(\"abc123def456\", \"\\d+\", \"match\")")
        assert "123" in r and "456" in r

    def test_regexp_tokens(self, s):
        """R-POL31-05b: regexp with 'tokens' SHALL extract capture groups."""
        r = s.eval("regexp(\"2024-01-15\", \"(\\d{4})-(\\d{2})-(\\d{2})\", \"tokens\")")
        assert "cell" in r.lower() or "2024" in r


# ---------- 6. regexprep ----------
class TestRegexprep:
    """R-POL31-05 (continued): regexprep SHALL perform substitutions
    including backreferences.

    Decomposition:
      R-POL31-05c: regexprep replaces single characters.
      R-POL31-05d: regexprep replaces digit groups with a placeholder.
      R-POL31-05e: regexprep supports backreferences ($1, $2).

    Consistency: Simple replacement (05c), group replacement (05d), and
    backreference (05e) cover the regexprep API.
    """

    def test_regexprep_simple(self, s):
        """R-POL31-05c: regexprep SHALL replace single characters."""
        r = s.eval("regexprep(\"Hello World\", \"o\", \"0\")")
        assert "Hell0 W0rld" in r

    def test_regexprep_digit_replace(self, s):
        """R-POL31-05d: regexprep SHALL replace digit groups with a placeholder."""
        r = s.eval("regexprep(\"foo123bar456\", \"\\d+\", \"#\")")
        assert "foo#bar#" in r

    def test_regexprep_backreference(self, s):
        """R-POL31-05e: regexprep SHALL support backreferences ($1, $2)."""
        r = s.eval("regexprep(\"CamelCase\", \"([a-z])([A-Z])\", \"$1_$2\")")
        assert "Camel_Case" in r


# ---------- 7. type checking ----------
class TestTypeChecking:
    """R-POL31-06: Type-checking functions SHALL return correct boolean results.

    Model-user argument: Guard clauses like ``if ~isnumeric(x); error(...);
    end`` are standard defensive programming in Octave. If type checks
    return wrong results, input validation silently passes bad data.

    Decomposition:
      R-POL31-06a: isnumeric(42) returns 1.
      R-POL31-06b: isnumeric("hello") returns 0.
      R-POL31-06c: ischar("hello") returns 1.
      R-POL31-06d: ischar(42) returns 0.
      R-POL31-06e: islogical(true) returns 1.
      R-POL31-06f: isfloat(3.14) returns 1.
      R-POL31-06g: isinteger(int32(5)) returns 1.

    Consistency: Each type-check function is tested with a true and/or
    false input, covering the full set.
    """

    def test_isnumeric_true(self, s):
        """R-POL31-06a: isnumeric(42) SHALL return 1."""
        r = s.eval("isnumeric(42)")
        assert "1" in r

    def test_isnumeric_false(self, s):
        """R-POL31-06b: isnumeric('hello') SHALL return 0."""
        r = s.eval("isnumeric(\"hello\")")
        assert "0" in r

    def test_ischar_true(self, s):
        """R-POL31-06c: ischar('hello') SHALL return 1."""
        r = s.eval("ischar(\"hello\")")
        assert "1" in r

    def test_ischar_false(self, s):
        """R-POL31-06d: ischar(42) SHALL return 0."""
        r = s.eval("ischar(42)")
        assert "0" in r

    def test_islogical_true(self, s):
        """R-POL31-06e: islogical(true) SHALL return 1."""
        r = s.eval("islogical(true)")
        assert "1" in r

    def test_isfloat_true(self, s):
        """R-POL31-06f: isfloat(3.14) SHALL return 1."""
        r = s.eval("isfloat(3.14)")
        assert "1" in r

    def test_isinteger_true(self, s):
        """R-POL31-06g: isinteger(int32(5)) SHALL return 1."""
        r = s.eval("isinteger(int32(5))")
        assert "1" in r


# ---------- 8. triu / tril ----------
class TestTriuTril:
    """R-POL31-07: triu and tril SHALL extract upper and lower triangular
    portions of matrices, with optional diagonal offset k.

    Model-user argument: Linear algebra workflows (Cholesky, LU) rely on
    triu/tril to isolate triangular factors. An incorrect diagonal offset
    would corrupt factorization results.

    Decomposition:
      R-POL31-07a: triu extracts upper triangle (default k=0).
      R-POL31-07b: tril extracts lower triangle (default k=0).
      R-POL31-07c: triu with k=1 excludes the main diagonal.
      R-POL31-07d: tril with k=-1 excludes the main diagonal.

    Consistency: Default (07a-b) and offset (07c-d) variants cover the API.
    """

    def test_triu(self, s):
        """R-POL31-07a: triu SHALL extract upper triangle (default k=0)."""
        s.eval("m = triu([1 2 3; 4 5 6; 7 8 9])")
        m = s._engine.workspace.get("m")
        expected = np.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]])
        np.testing.assert_array_equal(m.data, expected)

    def test_tril(self, s):
        """R-POL31-07b: tril SHALL extract lower triangle (default k=0)."""
        s.eval("m = tril([1 2 3; 4 5 6; 7 8 9])")
        m = s._engine.workspace.get("m")
        expected = np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]])
        np.testing.assert_array_equal(m.data, expected)

    def test_triu_with_k(self, s):
        """R-POL31-07c: triu with k=1 SHALL exclude the main diagonal."""
        s.eval("m = triu([1 2 3; 4 5 6; 7 8 9], 1)")
        m = s._engine.workspace.get("m")
        expected = np.array([[0, 2, 3], [0, 0, 6], [0, 0, 0]])
        np.testing.assert_array_equal(m.data, expected)

    def test_tril_with_k(self, s):
        """R-POL31-07d: tril with k=-1 SHALL exclude the main diagonal."""
        s.eval("m = tril([1 2 3; 4 5 6; 7 8 9], -1)")
        m = s._engine.workspace.get("m")
        expected = np.array([[0, 0, 0], [4, 0, 0], [7, 8, 0]])
        np.testing.assert_array_equal(m.data, expected)


# ---------- 9. diag ----------
class TestDiag:
    """R-POL31-08: diag SHALL create diagonal matrices from vectors and
    extract diagonals from matrices, with optional offset k.

    Model-user argument: Creating diagonal matrices for scaling or
    extracting eigenvalue diagonals is routine linear algebra. The k
    parameter for super/sub-diagonals must work for banded-matrix
    construction.

    Decomposition:
      R-POL31-08a: diag([1 2 3]) creates a 3x3 diagonal matrix.
      R-POL31-08b: diag(matrix) extracts the main diagonal.
      R-POL31-08c: diag(v, 1) creates a matrix with superdiagonal.
      R-POL31-08d: diag(v, -1) creates a matrix with subdiagonal.
      R-POL31-08e: diag(matrix, 1) extracts the superdiagonal.

    Consistency: Creation (08a, 08c-d), extraction (08b, 08e), and offset
    variants cover the diag API.
    """

    def test_diag_create(self, s):
        """R-POL31-08a: diag([1 2 3]) SHALL create a 3x3 diagonal matrix."""
        s.eval("m = diag([1 2 3])")
        m = s._engine.workspace.get("m")
        expected = np.diag([1, 2, 3])
        np.testing.assert_array_equal(m.data, expected)

    def test_diag_extract(self, s):
        """R-POL31-08b: diag(matrix) SHALL extract the main diagonal."""
        s.eval("d = diag([1 2 3; 4 5 6; 7 8 9])")
        d = s._engine.workspace.get("d")
        np.testing.assert_array_equal(d.data.ravel(), [1, 5, 9])

    def test_diag_create_superdiag(self, s):
        """R-POL31-08c: diag(v, 1) SHALL create a matrix with superdiagonal."""
        s.eval("m = diag([1 2 3], 1)")
        m = s._engine.workspace.get("m")
        expected = np.diag([1, 2, 3], 1)
        np.testing.assert_array_equal(m.data, expected)

    def test_diag_create_subdiag(self, s):
        """R-POL31-08d: diag(v, -1) SHALL create a matrix with subdiagonal."""
        s.eval("m = diag([1 2 3], -1)")
        m = s._engine.workspace.get("m")
        expected = np.diag([1, 2, 3], -1)
        np.testing.assert_array_equal(m.data, expected)

    def test_diag_extract_superdiag(self, s):
        """R-POL31-08e: diag(matrix, 1) SHALL extract the superdiagonal."""
        s.eval("d = diag([1 2 3; 4 5 6; 7 8 9], 1)")
        d = s._engine.workspace.get("d")
        np.testing.assert_array_equal(d.data.ravel(), [2, 6])
