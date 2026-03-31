# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for multi-output nargout, find, meshgrid, sub2ind/ind2sub (R24)."""
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
    def test_size_single_output(self, s):
        """[a] = size(M) returns full size vector."""
        s.eval("[a] = size([1 2; 3 4])")
        a = _unwrap(s.workspace.get("a"))
        np.testing.assert_array_equal(np.asarray(a).flatten(), [2, 2])

    def test_size_two_outputs(self, s):
        """[m, n] = size(M) returns rows and cols separately."""
        s.eval("[m, n] = size([1 2; 3 4])")
        assert _scalar(s.workspace.get("m")) == 2
        assert _scalar(s.workspace.get("n")) == 2

    def test_length_scalar(self, s):
        """length([1 2 3 4 5]) returns 5."""
        s.eval("n = length([1 2 3 4 5])")
        assert _scalar(s.workspace.get("n")) == 5

    def test_min_two_outputs(self, s):
        """[mn, idx] = min(v) returns value and 1-based index."""
        s.eval("[mn, idx] = min([3 1 4 1 5])")
        assert _scalar(s.workspace.get("mn")) == 1
        assert _scalar(s.workspace.get("idx")) == 2

    def test_max_two_outputs(self, s):
        """[mx, idx] = max(v) returns value and 1-based index."""
        s.eval("[mx, idx] = max([3 1 4 1 5])")
        assert _scalar(s.workspace.get("mx")) == 5
        assert _scalar(s.workspace.get("idx")) == 5

    def test_sort_two_outputs(self, s):
        """[s, i] = sort(v) returns sorted values and permutation indices."""
        s.eval("[sv, si] = sort([3 1 4 1 5])")
        sv = _unwrap(s.workspace.get("sv")).flatten()
        si = _unwrap(s.workspace.get("si")).flatten()
        np.testing.assert_array_equal(sv, [1, 1, 3, 4, 5])
        np.testing.assert_array_equal(si, [2, 4, 1, 3, 5])

    def test_unique_three_outputs(self, s):
        """[u, i, j] = unique(v) returns unique vals, first indices, inverse."""
        s.eval("[u, ui, uj] = unique([3 1 2 1 3])")
        u = _unwrap(s.workspace.get("u")).flatten()
        ui = _unwrap(s.workspace.get("ui")).flatten()
        uj = _unwrap(s.workspace.get("uj")).flatten()
        np.testing.assert_array_equal(u, [1, 2, 3])
        np.testing.assert_array_equal(ui, [2, 3, 1])
        np.testing.assert_array_equal(uj, [3, 1, 2, 1, 3])


# -- find multi-output ------------------------------------------------------


class TestFindMultiOutput:
    def test_find_single_output(self, s):
        """[i] = find(v) returns linear indices of nonzero elements."""
        s.eval("[fi] = find([0 3 0 4 5])")
        fi = _unwrap(s.workspace.get("fi")).flatten()
        np.testing.assert_array_equal(fi, [2, 4, 5])

    def test_find_two_outputs_matrix(self, s):
        """[r, c] = find(M) returns row and column indices."""
        s.eval("[fr, fc] = find([1 0; 0 2])")
        fr = _unwrap(s.workspace.get("fr")).flatten()
        fc = _unwrap(s.workspace.get("fc")).flatten()
        np.testing.assert_array_equal(fr, [1, 2])
        np.testing.assert_array_equal(fc, [1, 2])

    def test_find_three_outputs_matrix(self, s):
        """[r, c, v] = find(M) also returns nonzero values."""
        s.eval("[fr, fc, fv] = find([1 0; 0 2])")
        fv = _unwrap(s.workspace.get("fv")).flatten()
        np.testing.assert_array_equal(fv, [1, 2])

    def test_find_empty(self, s):
        """find on all-zero returns empty."""
        s.eval("fi = find([0 0 0])")
        fi = _unwrap(s.workspace.get("fi"))
        assert np.asarray(fi).size == 0


# -- meshgrid ---------------------------------------------------------------


class TestMeshgrid:
    def test_meshgrid_2d_shape(self, s):
        """[X, Y] = meshgrid(1:3, 1:2) produces 2x3 grids."""
        s.eval("[X, Y] = meshgrid(1:3, 1:2)")
        X = _unwrap(s.workspace.get("X"))
        Y = _unwrap(s.workspace.get("Y"))
        assert X.shape == (2, 3)
        assert Y.shape == (2, 3)

    def test_meshgrid_2d_values(self, s):
        """meshgrid X repeats x along rows, Y repeats y along cols."""
        s.eval("[X, Y] = meshgrid(1:3, 1:2)")
        X = _unwrap(s.workspace.get("X"))
        Y = _unwrap(s.workspace.get("Y"))
        np.testing.assert_array_equal(X[0, :], [1, 2, 3])
        np.testing.assert_array_equal(X[1, :], [1, 2, 3])
        np.testing.assert_array_equal(Y[:, 0], [1, 2])

    def test_meshgrid_3d(self, s):
        """[X, Y, Z] = meshgrid(1:2, 1:2, 1:2) produces 3D grids."""
        s.eval("[X3, Y3, Z3] = meshgrid(1:2, 1:2, 1:2)")
        X3 = _unwrap(s.workspace.get("X3"))
        Y3 = _unwrap(s.workspace.get("Y3"))
        Z3 = _unwrap(s.workspace.get("Z3"))
        assert X3.shape == (2, 2, 2)
        assert Y3.shape == (2, 2, 2)
        assert Z3.shape == (2, 2, 2)

    def test_meshgrid_single_arg(self, s):
        """meshgrid(v) with one arg uses v for both x and y."""
        s.eval("[X, Y] = meshgrid(1:3)")
        X = _unwrap(s.workspace.get("X"))
        assert X.shape == (3, 3)


# -- sub2ind / ind2sub -------------------------------------------------------


class TestSubIndConversion:
    def test_sub2ind_basic(self, s):
        """sub2ind([3 3], 2, 3) == 8 (column-major)."""
        s.eval("si = sub2ind([3 3], 2, 3)")
        assert _scalar(s.workspace.get("si")) == 8

    def test_ind2sub_basic(self, s):
        """[r, c] = ind2sub([3 3], 8) returns r=2, c=3."""
        s.eval("[ir, ic] = ind2sub([3 3], 8)")
        assert _scalar(s.workspace.get("ir")) == 2
        assert _scalar(s.workspace.get("ic")) == 3

    def test_sub2ind_ind2sub_roundtrip(self, s):
        """sub2ind -> ind2sub roundtrip preserves indices."""
        s.eval("idx = sub2ind([4 5], 3, 4)")
        s.eval("[r, c] = ind2sub([4 5], idx)")
        assert _scalar(s.workspace.get("r")) == 3
        assert _scalar(s.workspace.get("c")) == 4

    def test_sub2ind_first_element(self, s):
        """sub2ind([3 3], 1, 1) == 1."""
        s.eval("si = sub2ind([3 3], 1, 1)")
        assert _scalar(s.workspace.get("si")) == 1

    def test_ind2sub_last_element(self, s):
        """ind2sub([3 3], 9) returns r=3, c=3."""
        s.eval("[ir, ic] = ind2sub([3 3], 9)")
        assert _scalar(s.workspace.get("ir")) == 3
        assert _scalar(s.workspace.get("ic")) == 3
