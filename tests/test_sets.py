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
    """R-SET-01: intersect SHALL return the sorted vector of elements common to
    both input vectors.

    Model-user argument: When a structural engineer builds a FEM model with
    multiple mesh regions, they use intersect to find shared boundary nodes
    between adjacent regions. If intersect silently drops common nodes or
    returns them unsorted, the engineer will assemble coupling equations on
    the wrong DOFs, producing a singular stiffness matrix with no obvious
    cause.

    Decomposition:
      R-SET-01.1 — Known overlap returns correct common elements.
      R-SET-01.2 — Disjoint inputs return an empty result.

    Consistency: R-SET-01.1 proves common elements are identified and sorted.
    R-SET-01.2 proves the empty-overlap edge case is handled without error.
    Together they cover both the positive and null intersection cases.
    """

    def test_intersect_known(self):
        """R-SET-01.1: Overlapping vectors return sorted common elements."""
        a = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0]))
        b = ForgeArray(np.array([3.0, 4.0, 5.0, 6.0]))
        r = _unwrap(forge_intersect(a, b)).ravel()
        np.testing.assert_array_equal(r, [3, 4])

    def test_intersect_no_overlap(self):
        """R-SET-01.2: Disjoint vectors return an empty result."""
        a = ForgeArray(np.array([1.0, 2.0]))
        b = ForgeArray(np.array([3.0, 4.0]))
        r = _unwrap(forge_intersect(a, b)).ravel()
        assert len(r) == 0


class TestUnion:
    """R-SET-02: union SHALL return the sorted vector of all unique elements
    present in either input vector.

    Model-user argument: Engineers merge boundary node sets from separate mesh
    regions into a single constraint list using union. If duplicates survive
    the merge, downstream assembly will double-count constraint equations,
    producing an over-constrained system that either fails to solve or gives
    silently wrong displacements.

    Decomposition:
      R-SET-02.1 — Known inputs produce the correct merged, sorted result.
      R-SET-02.2 — Duplicate elements within and across inputs are removed.

    Consistency: R-SET-02.1 verifies correct merging and sorting. R-SET-02.2
    verifies duplicate elimination. Together they confirm union produces a
    unique sorted superset.
    """

    def test_union_known(self):
        """R-SET-02.1: Overlapping vectors produce a sorted unique merge."""
        a = ForgeArray(np.array([1.0, 2.0, 3.0]))
        b = ForgeArray(np.array([3.0, 4.0, 5.0]))
        r = _unwrap(forge_union(a, b)).ravel()
        np.testing.assert_array_equal(r, [1, 2, 3, 4, 5])

    def test_union_duplicates(self):
        """R-SET-02.2: Intra- and inter-vector duplicates are eliminated."""
        a = ForgeArray(np.array([1.0, 1.0, 2.0]))
        b = ForgeArray(np.array([2.0, 2.0, 3.0]))
        r = _unwrap(forge_union(a, b)).ravel()
        np.testing.assert_array_equal(r, [1, 2, 3])


class TestSetdiff:
    """R-SET-03: setdiff SHALL return the sorted vector of elements in the
    first input that are not in the second input.

    Model-user argument: The canonical FEM workflow for extracting free DOFs
    is setdiff(all_dofs, fixed_dofs). If setdiff fails to remove constrained
    DOFs or accidentally removes unconstrained ones, the reduced stiffness
    matrix will be the wrong size and the solver will either crash or return
    physically meaningless results.

    Decomposition:
      R-SET-03.1 — Known subtraction returns only the remaining elements.
      R-SET-03.2 — When all elements are removed, the result is empty.

    Consistency: R-SET-03.1 proves correct element subtraction. R-SET-03.2
    proves the fully-constrained edge case (all DOFs fixed) returns empty
    without error. Together they cover partial and total removal.
    """

    def test_setdiff_known(self):
        """R-SET-03.1: Elements present in b are removed from a."""
        a = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        b = ForgeArray(np.array([2.0, 4.0]))
        r = _unwrap(forge_setdiff(a, b)).ravel()
        np.testing.assert_array_equal(r, [1, 3, 5])

    def test_setdiff_all_removed(self):
        """R-SET-03.2: Complete overlap yields an empty result."""
        a = ForgeArray(np.array([1.0, 2.0]))
        b = ForgeArray(np.array([1.0, 2.0, 3.0]))
        r = _unwrap(forge_setdiff(a, b)).ravel()
        assert len(r) == 0


class TestSetxor:
    """R-SET-04: setxor SHALL return the sorted vector of elements belonging
    to exactly one of the two input vectors (symmetric difference).

    Model-user argument: An engineer comparing two versions of a boundary
    condition set uses setxor to find DOFs that changed between revisions.
    If setxor incorrectly includes shared DOFs or omits changed ones, the
    engineer will miss constraint changes and the updated model will silently
    carry stale boundary conditions.

    Decomposition:
      R-SET-04.1 — Known inputs return only the non-shared elements, sorted.

    Consistency: R-SET-04.1 verifies that shared elements (2, 3) are excluded
    and unique-to-each elements (1, 4) are returned sorted. A single test
    suffices because symmetric difference is fully exercised by one case with
    overlap.
    """

    def test_setxor_known(self):
        """R-SET-04.1: Symmetric difference excludes shared elements."""
        a = ForgeArray(np.array([1.0, 2.0, 3.0]))
        b = ForgeArray(np.array([2.0, 3.0, 4.0]))
        r = _unwrap(forge_setxor(a, b)).ravel()
        np.testing.assert_array_equal(r, [1, 4])


class TestUnique:
    """R-SET-05: unique SHALL return the sorted vector of distinct elements
    from the input.

    Model-user argument: After concatenating node index vectors from multiple
    load cases, the engineer calls unique to produce a clean, non-redundant
    index list. If unique leaves duplicates or missorts, downstream indexing
    into the global stiffness matrix will access wrong rows, corrupting the
    assembled system.

    Decomposition:
      R-SET-05.1 — Duplicates are removed and result is sorted ascending.
      R-SET-05.2 — Already-unique input is returned sorted (no false removal).

    Consistency: R-SET-05.1 proves duplicate removal and sorting. R-SET-05.2
    proves that unique values are never dropped. Together they confirm both
    deduplication and preservation of distinct elements.
    """

    def test_unique_sorted(self):
        """R-SET-05.1: Duplicates removed, result sorted ascending."""
        x = ForgeArray(np.array([3.0, 1.0, 2.0, 1.0, 3.0]))
        r = _unwrap(forge_unique(x)).ravel()
        np.testing.assert_array_equal(r, [1, 2, 3])

    def test_unique_already_unique(self):
        """R-SET-05.2: All-distinct input returned sorted without loss."""
        x = ForgeArray(np.array([5.0, 4.0, 3.0, 2.0, 1.0]))
        r = _unwrap(forge_unique(x)).ravel()
        np.testing.assert_array_equal(r, [1, 2, 3, 4, 5])


class TestIsmember:
    """R-SET-06: ismember SHALL return a logical vector indicating which
    elements of the first input appear in the second input.

    Model-user argument: Before applying loads to a node subset, the engineer
    uses ismember to verify which requested nodes actually exist in the mesh.
    If ismember returns false positives, loads will be applied to nonexistent
    nodes (silent data corruption). If it returns false negatives, loads will
    be silently omitted.

    Decomposition:
      R-SET-06.1 — Known membership returns correct logical flags.

    Consistency: R-SET-06.1 tests a mixed case (two members, two non-members)
    which exercises both true and false branches of the membership check in a
    single assertion.
    """

    def test_ismember_known(self):
        """R-SET-06.1: Membership flags match expected pattern."""
        a = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0]))
        b = ForgeArray(np.array([2.0, 4.0, 6.0]))
        tf, loc = forge_ismember(a, b)
        r = _unwrap(tf).ravel()
        np.testing.assert_array_equal(r, [0, 1, 0, 1])


class TestUniquetol:
    """R-SET-07: uniquetol SHALL return the sorted vector of distinct elements
    where values within the specified tolerance are treated as equal.

    Model-user argument: Node coordinates from mesh generators carry floating-
    point noise at machine epsilon. An engineer merging nodes from two mesh
    regions needs uniquetol to recognize that 1.0 and 1.0+1e-14 are the same
    physical point. Without tolerance-aware deduplication, the model will
    contain near-duplicate nodes that produce zero-length elements and a
    singular stiffness matrix.

    Decomposition:
      R-SET-07.1 — Values within tolerance collapse to representative values.

    Consistency: R-SET-07.1 uses pairs separated by 1e-14 with a tolerance of
    1e-12, confirming that sub-tolerance differences are merged while values
    separated by 1.0 remain distinct.
    """

    def test_uniquetol_within_tol(self):
        """R-SET-07.1: Near-duplicate values collapse within tolerance."""
        x = ForgeArray(np.array([1.0, 1.0 + 1e-14, 2.0, 2.0 + 1e-14, 3.0]))
        r = _unwrap(forge_uniquetol(x, 1e-12)).ravel()
        np.testing.assert_array_equal(r, [1, 2, 3])


class TestEmptySets:
    """R-SET-08: Set operations SHALL handle empty input vectors without error,
    returning mathematically correct results.

    Model-user argument: In parametric FEM scripts, boundary condition sets
    may be conditionally empty (e.g., a free-free beam has no fixed DOFs).
    If intersect or union crashes on an empty input, the parametric sweep
    halts entirely. The engineer expects empty-in to produce the mathematically
    correct empty-or-passthrough result, not an exception.

    Decomposition:
      R-SET-08.1 — intersect with an empty vector returns empty.
      R-SET-08.2 — union with an empty vector returns the non-empty input.

    Consistency: R-SET-08.1 confirms the identity intersect(A, {}) = {}.
    R-SET-08.2 confirms the identity union(A, {}) = A. Together they verify
    that empty inputs are propagated according to set-theoretic identities.
    """

    def test_intersect_empty(self):
        """R-SET-08.1: Intersect with empty vector returns empty."""
        a = ForgeArray(np.array([1.0, 2.0]))
        b = ForgeArray(np.array([], dtype=float))
        r = _unwrap(forge_intersect(a, b)).ravel()
        assert len(r) == 0

    def test_union_empty(self):
        """R-SET-08.2: Union with empty vector returns the non-empty input."""
        a = ForgeArray(np.array([1.0, 2.0]))
        b = ForgeArray(np.array([], dtype=float))
        r = _unwrap(forge_union(a, b)).ravel()
        np.testing.assert_array_equal(r, [1, 2])


class TestSortedOutput:
    """R-SET-09: All set operations SHALL return results in ascending sorted
    order regardless of input ordering.

    Model-user argument: Octave guarantees sorted output from set functions.
    Engineers rely on this for binary search, direct indexing, and visual
    inspection of DOF lists. If Forge returns unsorted results, code that
    assumes sorted order (e.g., using the result as sparse matrix indices)
    will silently produce wrong assemblies or trigger CHOLMOD errors.

    Decomposition:
      R-SET-09.1 — intersect on reverse-sorted inputs returns ascending order.

    Consistency: R-SET-09.1 feeds descending inputs and asserts both correct
    content and ascending order. Since sorting is a shared post-processing
    step across all set functions, verifying it on intersect with adversarial
    input order is representative.
    """

    def test_intersect_sorted(self):
        """R-SET-09.1: Reverse-sorted inputs still produce ascending output."""
        a = ForgeArray(np.array([5.0, 3.0, 1.0]))
        b = ForgeArray(np.array([3.0, 1.0, 7.0]))
        r = _unwrap(forge_intersect(a, b)).ravel()
        np.testing.assert_array_equal(r, sorted(r))
        np.testing.assert_array_equal(r, [1, 3])
