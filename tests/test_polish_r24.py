# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for multi-output nargout, find, meshgrid, sub2ind/ind2sub (R24).

V&V Traceability (backfill):
    R-POL24-01 .. R-POL24-04 (parent requirements)
    R-POL24-01-nn .. R-POL24-04-nn (unit sub-requirements)

SRS trace: SRS-FUNC-001, SRS-VAL-001, SRS-COMPAT-001
"""
import pytest
import numpy as np
from forge.engine.evaluator import Session
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def s():
    return Session()


def _scalar(v):
    """Extract a Python float from a workspace value."""
    arr = _unwrap(v)
    return float(np.asarray(arr).flat[0])


# -- nargout / multi-output context ----------------------------------------


class TestNargoutMultiOutput:
    """R-POL24-01: Forge SHALL support multi-output assignment syntax
    [a, b, ...] = func(...) for size, min, max, sort, and unique, returning
    the correct number of outputs with 1-based indices matching MATLAB/Octave.

    Model-user argument: An engineer porting numerical code from Octave uses
    multi-output calls constantly: [m,n]=size(A) for dimension queries,
    [val,idx]=min(v) for locating extrema, [sv,si]=sort(v) for ranked access,
    and [u,i,j]=unique(v) for deduplication with index mapping. If the
    multi-output dispatch returns wrong counts or swapped outputs, every
    downstream index-based operation silently produces wrong results.

    Decomposition:
        R-POL24-01-01: [a] = size(M) returns full size vector
        R-POL24-01-02: [m,n] = size(M) returns rows and cols separately
        R-POL24-01-03: length(v) returns element count
        R-POL24-01-04: [mn,idx] = min(v) returns value and 1-based index
        R-POL24-01-05: [mx,idx] = max(v) returns value and 1-based index
        R-POL24-01-06: [sv,si] = sort(v) returns sorted values and permutation
        R-POL24-01-07: [u,i,j] = unique(v) returns unique vals, first indices, inverse

    Consistency: Sub-requirements cover single-output (01, 03), two-output
    (02, 04, 05, 06), and three-output (07) dispatch paths. Together they
    verify that nargout-aware functions return the correct number and order of
    outputs.
    """

    def test_size_single_output(self, s):
        """R-POL24-01-01: [a] = size(M) returns full size vector."""
        s.eval("[a] = size([1 2; 3 4])")
        a = _unwrap(s.workspace.get("a"))
        np.testing.assert_array_equal(np.asarray(a).flatten(), [2, 2])

    def test_size_two_outputs(self, s):
        """R-POL24-01-02: [m,n] = size(M) returns rows and cols separately."""
        s.eval("[m, n] = size([1 2; 3 4])")
        assert _scalar(s.workspace.get("m")) == 2
        assert _scalar(s.workspace.get("n")) == 2

    def test_length_scalar(self, s):
        """R-POL24-01-03: length([1 2 3 4 5]) returns 5."""
        s.eval("n = length([1 2 3 4 5])")
        assert _scalar(s.workspace.get("n")) == 5

    def test_min_two_outputs(self, s):
        """R-POL24-01-04: [mn,idx] = min(v) returns value and 1-based index."""
        s.eval("[mn, idx] = min([3 1 4 1 5])")
        assert _scalar(s.workspace.get("mn")) == 1
        assert _scalar(s.workspace.get("idx")) == 2

    def test_max_two_outputs(self, s):
        """R-POL24-01-05: [mx,idx] = max(v) returns value and 1-based index."""
        s.eval("[mx, idx] = max([3 1 4 1 5])")
        assert _scalar(s.workspace.get("mx")) == 5
        assert _scalar(s.workspace.get("idx")) == 5

    def test_sort_two_outputs(self, s):
        """R-POL24-01-06: [sv,si] = sort(v) returns sorted values and permutation."""
        s.eval("[sv, si] = sort([3 1 4 1 5])")
        sv = _unwrap(s.workspace.get("sv")).flatten()
        si = _unwrap(s.workspace.get("si")).flatten()
        np.testing.assert_array_equal(sv, [1, 1, 3, 4, 5])
        np.testing.assert_array_equal(si, [2, 4, 1, 3, 5])

    def test_unique_three_outputs(self, s):
        """R-POL24-01-07: [u,i,j] = unique(v) returns unique, first idx, inverse."""
        s.eval("[u, ui, uj] = unique([3 1 2 1 3])")
        u = _unwrap(s.workspace.get("u")).flatten()
        ui = _unwrap(s.workspace.get("ui")).flatten()
        uj = _unwrap(s.workspace.get("uj")).flatten()
        np.testing.assert_array_equal(u, [1, 2, 3])
        np.testing.assert_array_equal(ui, [2, 3, 1])
        np.testing.assert_array_equal(uj, [3, 1, 2, 1, 3])


# -- find multi-output ------------------------------------------------------


class TestFindMultiOutput:
    """R-POL24-02: Forge SHALL support find() with 1, 2, and 3 output arguments,
    returning linear indices, row/column indices, and nonzero values
    respectively, using 1-based indexing matching MATLAB/Octave.

    Model-user argument: A scientist porting sparse matrix or masking code from
    Octave uses find() with multiple outputs to locate nonzero elements by
    position and value. Incorrect indices would cause wrong element access in
    sparse assembly or logical masking operations.

    Decomposition:
        R-POL24-02-01: find(v) returns 1-based linear indices
        R-POL24-02-02: [r,c] = find(M) returns row and column indices
        R-POL24-02-03: [r,c,v] = find(M) also returns nonzero values
        R-POL24-02-04: find on all-zero returns empty

    Consistency: Sub-requirements cover single-output (01), two-output (02),
    three-output (03), and empty-result (04) paths. Together they verify all
    find() output modes.
    """

    def test_find_single_output(self, s):
        """R-POL24-02-01: find([0 3 0 4 5]) returns [2, 4, 5]."""
        s.eval("[fi] = find([0 3 0 4 5])")
        fi = _unwrap(s.workspace.get("fi")).flatten()
        np.testing.assert_array_equal(fi, [2, 4, 5])

    def test_find_two_outputs_matrix(self, s):
        """R-POL24-02-02: [r,c] = find([1 0; 0 2]) returns rows and cols."""
        s.eval("[fr, fc] = find([1 0; 0 2])")
        fr = _unwrap(s.workspace.get("fr")).flatten()
        fc = _unwrap(s.workspace.get("fc")).flatten()
        np.testing.assert_array_equal(fr, [1, 2])
        np.testing.assert_array_equal(fc, [1, 2])

    def test_find_three_outputs_matrix(self, s):
        """R-POL24-02-03: [r,c,v] = find(M) returns nonzero values."""
        s.eval("[fr, fc, fv] = find([1 0; 0 2])")
        fv = _unwrap(s.workspace.get("fv")).flatten()
        np.testing.assert_array_equal(fv, [1, 2])

    def test_find_empty(self, s):
        """R-POL24-02-04: find([0 0 0]) returns empty."""
        s.eval("fi = find([0 0 0])")
        fi = _unwrap(s.workspace.get("fi"))
        assert np.asarray(fi).size == 0


# -- meshgrid ---------------------------------------------------------------


class TestMeshgrid:
    """R-POL24-03: Forge SHALL provide meshgrid() that generates 2-D and 3-D
    coordinate grids with correct shapes and value patterns matching
    MATLAB/Octave output.

    Model-user argument: An engineer porting surface plots, PDE solvers, or
    interpolation code from Octave relies on meshgrid to set up evaluation
    grids. Wrong grid shapes or transposed axes would produce distorted surfaces
    and incorrect numerical solutions.

    Decomposition:
        R-POL24-03-01: meshgrid(1:3, 1:2) produces 2x3 grids
        R-POL24-03-02: X repeats x along rows, Y repeats y along columns
        R-POL24-03-03: meshgrid with 3 ranges produces 3-D grids
        R-POL24-03-04: meshgrid(v) with single arg uses v for both x and y

    Consistency: Sub-requirements cover 2-D shape (01), 2-D values (02), 3-D
    extension (03), and single-argument shorthand (04). Together they verify
    meshgrid for all supported call signatures.
    """

    def test_meshgrid_2d_shape(self, s):
        """R-POL24-03-01: meshgrid(1:3, 1:2) produces 2x3 grids."""
        s.eval("[X, Y] = meshgrid(1:3, 1:2)")
        X = _unwrap(s.workspace.get("X"))
        Y = _unwrap(s.workspace.get("Y"))
        assert X.shape == (2, 3)
        assert Y.shape == (2, 3)

    def test_meshgrid_2d_values(self, s):
        """R-POL24-03-02: X repeats x along rows, Y repeats y along cols."""
        s.eval("[X, Y] = meshgrid(1:3, 1:2)")
        X = _unwrap(s.workspace.get("X"))
        Y = _unwrap(s.workspace.get("Y"))
        np.testing.assert_array_equal(X[0, :], [1, 2, 3])
        np.testing.assert_array_equal(X[1, :], [1, 2, 3])
        np.testing.assert_array_equal(Y[:, 0], [1, 2])

    def test_meshgrid_3d(self, s):
        """R-POL24-03-03: meshgrid with 3 ranges produces (2,2,2) grids."""
        s.eval("[X3, Y3, Z3] = meshgrid(1:2, 1:2, 1:2)")
        X3 = _unwrap(s.workspace.get("X3"))
        Y3 = _unwrap(s.workspace.get("Y3"))
        Z3 = _unwrap(s.workspace.get("Z3"))
        assert X3.shape == (2, 2, 2)
        assert Y3.shape == (2, 2, 2)
        assert Z3.shape == (2, 2, 2)

    def test_meshgrid_single_arg(self, s):
        """R-POL24-03-04: meshgrid(v) with single arg produces 3x3 grid."""
        s.eval("[X, Y] = meshgrid(1:3)")
        X = _unwrap(s.workspace.get("X"))
        assert X.shape == (3, 3)


# -- sub2ind / ind2sub -------------------------------------------------------


class TestSubIndConversion:
    """R-POL24-04: Forge SHALL provide sub2ind and ind2sub functions that convert
    between subscript indices and linear indices using 1-based column-major
    ordering matching MATLAB/Octave.

    Model-user argument: An engineer porting matrix manipulation or sparse
    assembly code from Octave uses sub2ind/ind2sub for converting between row/
    column subscripts and linear storage positions. Incorrect conversion would
    silently access wrong matrix elements, producing corrupt stiffness matrices
    or image pixel lookups.

    Decomposition:
        R-POL24-04-01: sub2ind([3 3], 2, 3) = 8
        R-POL24-04-02: ind2sub([3 3], 8) returns (2, 3)
        R-POL24-04-03: sub2ind then ind2sub round-trip preserves indices
        R-POL24-04-04: sub2ind([3 3], 1, 1) = 1 (first element)
        R-POL24-04-05: ind2sub([3 3], 9) returns (3, 3) (last element)

    Consistency: Sub-requirements cover forward (01, 04), reverse (02, 05), and
    round-trip (03) conversions, including boundary elements. Together they
    verify correct column-major index mapping.
    """

    def test_sub2ind_basic(self, s):
        """R-POL24-04-01: sub2ind([3 3], 2, 3) = 8."""
        s.eval("si = sub2ind([3 3], 2, 3)")
        assert _scalar(s.workspace.get("si")) == 8

    def test_ind2sub_basic(self, s):
        """R-POL24-04-02: ind2sub([3 3], 8) returns r=2, c=3."""
        s.eval("[ir, ic] = ind2sub([3 3], 8)")
        assert _scalar(s.workspace.get("ir")) == 2
        assert _scalar(s.workspace.get("ic")) == 3

    def test_sub2ind_ind2sub_roundtrip(self, s):
        """R-POL24-04-03: sub2ind then ind2sub round-trip preserves (3,4)."""
        s.eval("idx = sub2ind([4 5], 3, 4)")
        s.eval("[r, c] = ind2sub([4 5], idx)")
        assert _scalar(s.workspace.get("r")) == 3
        assert _scalar(s.workspace.get("c")) == 4

    def test_sub2ind_first_element(self, s):
        """R-POL24-04-04: sub2ind([3 3], 1, 1) = 1."""
        s.eval("si = sub2ind([3 3], 1, 1)")
        assert _scalar(s.workspace.get("si")) == 1

    def test_ind2sub_last_element(self, s):
        """R-POL24-04-05: ind2sub([3 3], 9) returns r=3, c=3."""
        s.eval("[ir, ic] = ind2sub([3 3], 9)")
        assert _scalar(s.workspace.get("ir")) == 3
        assert _scalar(s.workspace.get("ic")) == 3
