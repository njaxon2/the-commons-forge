# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish round 32 -- bitwise, set ops, accumarray, reshape/permute/squeeze, repmat.

SRS trace: SRS-FUNC-001 (Octave-compatible function library)

V&V Traceability (backfill)
===========================
R-POL32-01: Bitwise operations (bitand, bitor, bitxor, bitshift, bitget,
            bitset) SHALL produce correct results on integer operands.

    Model-user argument: Engineers working with hardware registers, protocol
    flags, or binary file formats use bitwise operations daily. If bitand or
    bitshift return wrong values, bit-level data extraction produces
    corrupted readings.

    Decomposition:
      R-POL32-01a: bitand(12, 10) returns 8.
      R-POL32-01b: bitor(12, 10) returns 14.
      R-POL32-01c: bitxor(12, 10) returns 6.
      R-POL32-01d: bitshift(1, 4) returns 16 (left shift).
      R-POL32-01e: bitshift(16, -2) returns 4 (right shift).
      R-POL32-01f: bitget(13, 1) returns 1 (LSB).
      R-POL32-01g: bitset(0, 3) returns 4.

    Consistency: All seven bitwise functions are tested with representative
    operands, covering AND, OR, XOR, shift, get, and set operations.

R-POL32-02: Set operations (ismember) SHALL work for both numeric arrays
            and string-in-cell lookups, with optional multi-output [tf, loc].

    Model-user argument: Scientists checking whether sample IDs exist in a
    reference set use ismember. Both the logical result (tf) and the location
    index (loc) are needed for data alignment between experiment tables.

    Decomposition:
      R-POL32-02a: ismember("hello", {"hello","world","test"}) returns 1.
      R-POL32-02b: ismember("foo", {"hello","world","test"}) returns 0.
      R-POL32-02c: [tf, loc] = ismember([2 4 6], [1 2 3 4 5]) returns correct tf.
      R-POL32-02d: [tf, loc] returns correct loc indices.

    Consistency: String lookup (02a-b) and numeric multi-output (02c-d)
    cover the ismember API.

R-POL32-03: accumarray SHALL accumulate values by subscript using default
            sum and custom function handles (@mean, @max).

    Model-user argument: Grouping sensor readings by station ID via
    accumarray is a standard data-reduction step. Both the default sum
    and custom aggregation functions must work for flexible analysis.

    Decomposition:
      R-POL32-03a: accumarray with default sum aggregates correctly.
      R-POL32-03b: accumarray with @mean computes group means.
      R-POL32-03c: accumarray with @max computes group maxima.

    Consistency: Default (03a) and two custom functions (03b-c) cover the
    accumarray dispatch mechanism.

R-POL32-04: reshape, permute, and squeeze SHALL transform array dimensions
            correctly, including auto-computed dimensions.

    Model-user argument: Reshaping data for plotting or algorithm input
    (e.g., reshape(data, [], nChannels)) is routine. If auto-dimension
    inference or permute order is wrong, arrays silently have wrong layout
    and downstream computations produce nonsense.

    Decomposition:
      R-POL32-04a: reshape to 2x3 produces correct shape.
      R-POL32-04b: reshape with [] auto-computes row count.
      R-POL32-04c: permute([3 1 2]) reorders dimensions correctly.
      R-POL32-04d: squeeze removes singleton dimensions.

    Consistency: Explicit shape (04a), auto-dimension (04b), permute (04c),
    and squeeze (04d) cover the dimension-manipulation API.

R-POL32-05: repmat SHALL tile a matrix to the specified row and column
            repetition counts.

    Model-user argument: Engineers use repmat to expand template patterns
    (e.g., repmat(kernel, nTiles, mTiles)) for convolution or block-matrix
    construction. Incorrect tiling produces wrong-shaped output.

    Decomposition:
      R-POL32-05a: repmat([1 2; 3 4], 2, 3) produces a 4x6 matrix.
      R-POL32-05b: Tiled values match the original block pattern.

    Consistency: Shape (05a) and content (05b) verification cover repmat.
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture(scope="module")
def S():
    return ForgeSession()


def _val(session, expr):
    """Evaluate and return numpy array."""
    r = session.eval(expr)
    if isinstance(r, str):
        # Try to get from workspace
        return r
    return np.asarray(_unwrap(r)).ravel() if isinstance(r, ForgeArray) else r


def _scalar(session, expr):
    """Evaluate and return a scalar float."""
    r = session.eval(expr)
    ws = session.workspace
    # If result is display string, re-read from ans
    if isinstance(r, str):
        ans = ws.get("ans") if ws.has("ans") else None
        if ans is not None:
            return float(np.asarray(_unwrap(ans)).ravel()[0])
        return float(r.strip())
    return float(np.asarray(_unwrap(r)).ravel()[0])


# ── Bitwise operations ────────────────────────────────────────────
class TestBitwise:
    """R-POL32-01: Bitwise operations SHALL produce correct results on
    integer operands.

    Model-user argument: Engineers working with hardware registers, protocol
    flags, or binary file formats use bitwise operations daily. If bitand or
    bitshift return wrong values, bit-level data extraction produces
    corrupted readings.

    Decomposition:
      R-POL32-01a: bitand(12, 10) returns 8.
      R-POL32-01b: bitor(12, 10) returns 14.
      R-POL32-01c: bitxor(12, 10) returns 6.
      R-POL32-01d: bitshift(1, 4) returns 16.
      R-POL32-01e: bitshift(16, -2) returns 4.
      R-POL32-01f: bitget(13, 1) returns 1.
      R-POL32-01g: bitset(0, 3) returns 4.

    Consistency: All seven bitwise functions are tested.
    """

    def test_bitand(self, S):
        """R-POL32-01a: bitand(12, 10) SHALL return 8."""
        assert _scalar(S, "bitand(12, 10)") == 8

    def test_bitor(self, S):
        """R-POL32-01b: bitor(12, 10) SHALL return 14."""
        assert _scalar(S, "bitor(12, 10)") == 14

    def test_bitxor(self, S):
        """R-POL32-01c: bitxor(12, 10) SHALL return 6."""
        assert _scalar(S, "bitxor(12, 10)") == 6

    def test_bitshift_left(self, S):
        """R-POL32-01d: bitshift(1, 4) SHALL return 16."""
        assert _scalar(S, "bitshift(1, 4)") == 16

    def test_bitshift_right(self, S):
        """R-POL32-01e: bitshift(16, -2) SHALL return 4."""
        assert _scalar(S, "bitshift(16, -2)") == 4

    def test_bitget_lsb(self, S):
        """R-POL32-01f: bitget(13, 1) SHALL return 1 (LSB)."""
        assert _scalar(S, "bitget(13, 1)") == 1

    def test_bitset(self, S):
        """R-POL32-01g: bitset(0, 3) SHALL return 4."""
        assert _scalar(S, "bitset(0, 3)") == 4


# ── Set operations edge cases ─────────────────────────────────────
class TestSetOps:
    """R-POL32-02: Set operations (ismember) SHALL work for both numeric
    arrays and string-in-cell lookups, with optional multi-output [tf, loc].

    Model-user argument: Scientists checking whether sample IDs exist in a
    reference set use ismember. Both the logical result (tf) and the location
    index (loc) are needed for data alignment between experiment tables.

    Decomposition:
      R-POL32-02a: ismember("hello", {"hello","world","test"}) returns 1.
      R-POL32-02b: ismember("foo", {"hello","world","test"}) returns 0.
      R-POL32-02c: [tf, loc] returns correct tf.
      R-POL32-02d: [tf, loc] returns correct loc indices.

    Consistency: String lookup (02a-b) and numeric multi-output (02c-d)
    cover the ismember API.
    """

    def test_ismember_string_in_cell(self, S):
        """R-POL32-02a: ismember('hello', {'hello','world','test'}) SHALL return 1."""
        v = _scalar(S, 'ismember("hello", {"hello", "world", "test"})')
        assert v == 1

    def test_ismember_string_not_in_cell(self, S):
        """R-POL32-02b: ismember('foo', {'hello','world','test'}) SHALL return 0."""
        v = _scalar(S, 'ismember("foo", {"hello", "world", "test"})')
        assert v == 0

    def test_ismember_multi_output_tf(self, S):
        """R-POL32-02c: [tf, loc] = ismember(...) SHALL return correct tf."""
        S.eval("[tf, loc] = ismember([2 4 6], [1 2 3 4 5]);")
        tf = np.asarray(_unwrap(S.workspace.get("tf"))).ravel()
        np.testing.assert_array_equal(tf, [1, 1, 0])

    def test_ismember_multi_output_loc(self, S):
        """R-POL32-02d: [tf, loc] = ismember(...) SHALL return correct loc indices."""
        S.eval("[tf, loc] = ismember([2 4 6], [1 2 3 4 5]);")
        loc = np.asarray(_unwrap(S.workspace.get("loc"))).ravel()
        np.testing.assert_array_equal(loc, [2, 4, 0])


# ── accumarray with function handles ──────────────────────────────
class TestAccumarray:
    """R-POL32-03: accumarray SHALL accumulate values by subscript using
    default sum and custom function handles.

    Model-user argument: Grouping sensor readings by station ID via
    accumarray is a standard data-reduction step. Both the default sum
    and custom aggregation functions must work for flexible analysis.

    Decomposition:
      R-POL32-03a: accumarray with default sum aggregates correctly.
      R-POL32-03b: accumarray with @mean computes group means.
      R-POL32-03c: accumarray with @max computes group maxima.

    Consistency: Default (03a) and two custom functions (03b-c) cover the
    accumarray dispatch mechanism.
    """

    def test_accumarray_default_sum(self, S):
        """R-POL32-03a: accumarray with default sum SHALL aggregate correctly."""
        S.eval("r = accumarray([1;1;2;2;3], [10;20;30;40;50]);")
        r = np.asarray(_unwrap(S.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(r, [30, 70, 50])

    def test_accumarray_mean(self, S):
        """R-POL32-03b: accumarray with @mean SHALL compute group means."""
        S.eval("r = accumarray([1;1;2;2;3], [10;20;30;40;50], [], @mean);")
        r = np.asarray(_unwrap(S.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(r, [15, 35, 50])

    def test_accumarray_max(self, S):
        """R-POL32-03c: accumarray with @max SHALL compute group maxima."""
        S.eval("r = accumarray([1;1;2;2;3], [10;20;30;40;50], [], @max);")
        r = np.asarray(_unwrap(S.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(r, [20, 40, 50])


# ── reshape / permute / squeeze ───────────────────────────────────
class TestReshapePermuteSqueze:
    """R-POL32-04: reshape, permute, and squeeze SHALL transform array
    dimensions correctly, including auto-computed dimensions.

    Model-user argument: Reshaping data for plotting or algorithm input
    is routine. If auto-dimension inference or permute order is wrong,
    arrays silently have wrong layout.

    Decomposition:
      R-POL32-04a: reshape to 2x3 produces correct shape.
      R-POL32-04b: reshape with [] auto-computes row count.
      R-POL32-04c: permute([3 1 2]) reorders dimensions correctly.
      R-POL32-04d: squeeze removes singleton dimensions.

    Consistency: Explicit shape (04a), auto-dimension (04b), permute (04c),
    and squeeze (04d) cover the dimension-manipulation API.
    """

    def test_reshape_2x3(self, S):
        """R-POL32-04a: reshape to 2x3 SHALL produce correct shape."""
        S.eval("r = reshape([1 2 3 4 5 6], 2, 3);")
        r = np.asarray(_unwrap(S.workspace.get("r")))
        assert r.shape == (2, 3)

    def test_reshape_auto_rows(self, S):
        """R-POL32-04b: reshape with [] SHALL auto-compute row count."""
        S.eval("r = reshape([1 2 3 4 5 6], [], 2);")
        r = np.asarray(_unwrap(S.workspace.get("r")))
        assert r.shape == (3, 2)

    def test_permute_reorder(self, S):
        """R-POL32-04c: permute([3 1 2]) SHALL reorder dimensions correctly."""
        S.eval("r = size(permute(rand(2,3,4), [3 1 2]));")
        r = np.asarray(_unwrap(S.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(r, [4, 2, 3])

    def test_squeeze_removes_singletons(self, S):
        """R-POL32-04d: squeeze SHALL remove singleton dimensions."""
        S.eval("r = size(squeeze(rand(1,3,1,4)));")
        r = np.asarray(_unwrap(S.workspace.get("r"))).ravel()
        np.testing.assert_array_equal(r, [3, 4])


# ── repmat ─────────────────────────────────────────────────────────
class TestRepmat:
    """R-POL32-05: repmat SHALL tile a matrix to the specified row and column
    repetition counts.

    Model-user argument: Engineers use repmat to expand template patterns
    for convolution or block-matrix construction. Incorrect tiling produces
    wrong-shaped output.

    Decomposition:
      R-POL32-05a: repmat([1 2; 3 4], 2, 3) produces a 4x6 matrix.
      R-POL32-05b: Tiled values match the original block pattern.

    Consistency: Shape (05a) and content (05b) verification cover repmat.
    """

    def test_repmat_2x3(self, S):
        """R-POL32-05a: repmat([1 2; 3 4], 2, 3) SHALL produce a 4x6 matrix."""
        S.eval("r = repmat([1 2; 3 4], 2, 3);")
        r = np.asarray(_unwrap(S.workspace.get("r")))
        assert r.shape == (4, 6)

    def test_repmat_values(self, S):
        """R-POL32-05b: Tiled values SHALL match the original block pattern."""
        S.eval("r = repmat([1 2; 3 4], 2, 3);")
        r = np.asarray(_unwrap(S.workspace.get("r")))
        # Top-left 2x2 block should be original
        np.testing.assert_array_equal(r[:2, :2], [[1, 2], [3, 4]])
        # Second block row should repeat
        np.testing.assert_array_equal(r[2:4, :2], [[1, 2], [3, 4]])
