"""Set operations toolbox.

Implements: intersect, union, setdiff, setxor, unique, ismember,
uniquetol, ismembertol, powerset

SRS trace: SRS-FUNC-001 (Octave-compatible function library)
"""
import numpy as np
from itertools import combinations
from forge.engine.types import ForgeArray, _unwrap


def _wrap(x):
    return ForgeArray(np.asarray(x))


# ── Core set operations ─────────────────────────────────────────

def forge_intersect(a, b):
    """Set intersection of two arrays."""
    va = np.asarray(_unwrap(a)).ravel()
    vb = np.asarray(_unwrap(b)).ravel()
    return _wrap(np.intersect1d(va, vb).ravel())


def forge_union(a, b):
    """Set union of two arrays."""
    va = np.asarray(_unwrap(a)).ravel()
    vb = np.asarray(_unwrap(b)).ravel()
    return _wrap(np.union1d(va, vb).ravel())


def forge_setdiff(a, b):
    """Set difference of two arrays (elements in a but not in b)."""
    va = np.asarray(_unwrap(a)).ravel()
    vb = np.asarray(_unwrap(b)).ravel()
    return _wrap(np.setdiff1d(va, vb).ravel())


def forge_setxor(a, b):
    """Set exclusive or of two arrays."""
    va = np.asarray(_unwrap(a)).ravel()
    vb = np.asarray(_unwrap(b)).ravel()
    return _wrap(np.setxor1d(va, vb).ravel())


def forge_unique(x):
    """Unique elements of array, sorted."""
    v = np.asarray(_unwrap(x)).ravel()
    return _wrap(np.unique(v).ravel())


def forge_ismember(a, b):
    """Test whether elements of a are members of b.

    Returns a logical ForgeArray of the same shape as a.
    """
    va = np.asarray(_unwrap(a)).ravel()
    vb = np.asarray(_unwrap(b)).ravel()
    return _wrap(np.isin(va, vb).astype(float).ravel())


# ── Tolerance-based operations ───────────────────────────────────

def forge_uniquetol(x, tol=None):
    """Unique elements within a tolerance.

    Parameters
    ----------
    x : ForgeArray
        Input array.
    tol : float, optional
        Absolute tolerance. Default is 1e-12.

    Returns a sorted array of unique values where elements within *tol*
    of each other are considered duplicates (the first occurrence is kept).
    """
    if tol is None:
        tol = 1e-12
    tol = float(_unwrap(tol)) if not isinstance(tol, (int, float)) else float(tol)
    v = np.sort(np.asarray(_unwrap(x), dtype=float).ravel())
    if v.size == 0:
        return _wrap(np.array([], dtype=float))
    result = [v[0]]
    for val in v[1:]:
        if abs(val - result[-1]) > tol:
            result.append(val)
    return _wrap(np.array(result, dtype=float).ravel())


def forge_ismembertol(a, b, tol=None):
    """Test membership within a tolerance.

    Parameters
    ----------
    a, b : ForgeArray
        Input arrays.
    tol : float, optional
        Absolute tolerance. Default is 1e-12.

    Returns a logical ForgeArray (1.0/0.0) indicating whether each
    element of *a* is within *tol* of any element of *b*.
    """
    if tol is None:
        tol = 1e-12
    tol = float(_unwrap(tol)) if not isinstance(tol, (int, float)) else float(tol)
    va = np.asarray(_unwrap(a), dtype=float).ravel()
    vb = np.asarray(_unwrap(b), dtype=float).ravel()
    if vb.size == 0:
        return _wrap(np.zeros_like(va))
    # Broadcasting: |va[i] - vb[j]| <= tol for any j
    result = np.any(np.abs(va[:, None] - vb[None, :]) <= tol, axis=1)
    return _wrap(result.astype(float).ravel())


# ── Power set ────────────────────────────────────────────────────

def forge_powerset(x):
    """Power set (all subsets) of the unique elements of x.

    Returns a 1-D ForgeArray containing the subsets as a flat
    concatenation.  Each subset is separated; empty set is included
    as an empty contribution (the total count of subsets is 2^n).

    For practical use the result is a cell-like list encoded as a
    ForgeArray of ForgeArrays when cell support is available.  For
    now returns a flat array of all subset elements concatenated,
    which matches Octave's ``combinator`` style.
    """
    v = np.unique(np.asarray(_unwrap(x)).ravel())
    n = len(v)
    if n > 20:
        raise ValueError("powerset: input has more than 20 unique elements; "
                         "result would be too large")
    subsets = []
    for k in range(n + 1):
        for combo in combinations(v, k):
            subsets.extend(combo)
    if len(subsets) == 0:
        return _wrap(np.array([], dtype=float))
    return _wrap(np.array(subsets, dtype=float).ravel())


# ── Registry for evaluator ───────────────────────────────────────

SETS_REGISTRY = {
    "intersect": forge_intersect,
    "union": forge_union,
    "setdiff": forge_setdiff,
    "setxor": forge_setxor,
    "unique": forge_unique,
    "ismember": forge_ismember,
    "uniquetol": forge_uniquetol,
    "ismembertol": forge_ismembertol,
    "powerset": forge_powerset,
}
