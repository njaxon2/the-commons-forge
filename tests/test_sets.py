# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for sets toolbox (9 functions).

SRS trace: SRS-FUNC-001, SRS-VAL-001
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.sets import *


class TestIntersect:

    def test_intersect_known(self):
        a = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0]))
        b = ForgeArray(np.array([3.0, 4.0, 5.0, 6.0]))
        r = _unwrap(forge_intersect(a, b)).ravel()
        np.testing.assert_array_equal(r, [3, 4])

    def test_intersect_no_overlap(self):
        a = ForgeArray(np.array([1.0, 2.0]))
        b = ForgeArray(np.array([3.0, 4.0]))
        r = _unwrap(forge_intersect(a, b)).ravel()
        assert len(r) == 0


class TestUnion:

    def test_union_known(self):
        a = ForgeArray(np.array([1.0, 2.0, 3.0]))
        b = ForgeArray(np.array([3.0, 4.0, 5.0]))
        r = _unwrap(forge_union(a, b)).ravel()
        np.testing.assert_array_equal(r, [1, 2, 3, 4, 5])

    def test_union_duplicates(self):
        a = ForgeArray(np.array([1.0, 1.0, 2.0]))
        b = ForgeArray(np.array([2.0, 2.0, 3.0]))
        r = _unwrap(forge_union(a, b)).ravel()
        np.testing.assert_array_equal(r, [1, 2, 3])


class TestSetdiff:

    def test_setdiff_known(self):
        a = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        b = ForgeArray(np.array([2.0, 4.0]))
        r = _unwrap(forge_setdiff(a, b)).ravel()
        np.testing.assert_array_equal(r, [1, 3, 5])

    def test_setdiff_all_removed(self):
        a = ForgeArray(np.array([1.0, 2.0]))
        b = ForgeArray(np.array([1.0, 2.0, 3.0]))
        r = _unwrap(forge_setdiff(a, b)).ravel()
        assert len(r) == 0


class TestSetxor:

    def test_setxor_known(self):
        a = ForgeArray(np.array([1.0, 2.0, 3.0]))
        b = ForgeArray(np.array([2.0, 3.0, 4.0]))
        r = _unwrap(forge_setxor(a, b)).ravel()
        np.testing.assert_array_equal(r, [1, 4])


class TestUnique:

    def test_unique_sorted(self):
        x = ForgeArray(np.array([3.0, 1.0, 2.0, 1.0, 3.0]))
        r = _unwrap(forge_unique(x)).ravel()
        np.testing.assert_array_equal(r, [1, 2, 3])

    def test_unique_already_unique(self):
        x = ForgeArray(np.array([5.0, 4.0, 3.0, 2.0, 1.0]))
        r = _unwrap(forge_unique(x)).ravel()
        np.testing.assert_array_equal(r, [1, 2, 3, 4, 5])


class TestIsmember:

    def test_ismember_known(self):
        a = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0]))
        b = ForgeArray(np.array([2.0, 4.0, 6.0]))
        tf, loc = forge_ismember(a, b)
        r = _unwrap(tf).ravel()
        np.testing.assert_array_equal(r, [0, 1, 0, 1])


class TestUniquetol:

    def test_uniquetol_within_tol(self):
        x = ForgeArray(np.array([1.0, 1.0 + 1e-14, 2.0, 2.0 + 1e-14, 3.0]))
        r = _unwrap(forge_uniquetol(x, 1e-12)).ravel()
        np.testing.assert_array_equal(r, [1, 2, 3])


class TestEmptySets:

    def test_intersect_empty(self):
        a = ForgeArray(np.array([1.0, 2.0]))
        b = ForgeArray(np.array([], dtype=float))
        r = _unwrap(forge_intersect(a, b)).ravel()
        assert len(r) == 0

    def test_union_empty(self):
        a = ForgeArray(np.array([1.0, 2.0]))
        b = ForgeArray(np.array([], dtype=float))
        r = _unwrap(forge_union(a, b)).ravel()
        np.testing.assert_array_equal(r, [1, 2])


class TestSortedOutput:

    def test_intersect_sorted(self):
        a = ForgeArray(np.array([5.0, 3.0, 1.0]))
        b = ForgeArray(np.array([3.0, 1.0, 7.0]))
        r = _unwrap(forge_intersect(a, b)).ravel()
        # Output should be sorted
        np.testing.assert_array_equal(r, sorted(r))
        np.testing.assert_array_equal(r, [1, 3])
