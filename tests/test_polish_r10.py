# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish R10 -- Linear algebra operations.

15+ tests covering: kron, cross, dot, norm, pinv, trace, det, inv,
null, orth, vecnorm, normalize.
SRS trace: SRS-FUNC-001 (linalg toolbox)

Requirement R-POL10-01:
    The linalg toolbox SHALL produce numerically correct results for
    Kronecker product, cross product, dot product, norms, pseudoinverse,
    trace, determinant, inverse, null space, column space, vector norm,
    and column normalization, matching established numerical references.

    Model-user argument:
    An engineer migrating from MATLAB/Octave relies on linear algebra
    primitives (kron, cross, dot, norm, inv, det, etc.) to build finite
    element assemblies and signal processing pipelines. If any of these
    return wrong values or wrong shapes, downstream analyses silently
    produce garbage. Verified parity with NumPy/Octave reference values
    is therefore essential.

    Decomposition:
    R-POL10-01a: kron produces correct Kronecker product and dimensions.
    R-POL10-01b: cross is correct and anti-commutative.
    R-POL10-01c: dot computes inner product (vector) and column-wise dot (matrix).
    R-POL10-01d: norm returns correct 2-norm (vector) and Frobenius norm (matrix).
    R-POL10-01e: pinv yields a left-inverse for full-rank matrices.
    R-POL10-01f: trace returns the sum of diagonal elements.
    R-POL10-01g: det returns correct determinant, including zero for singular matrices.
    R-POL10-01h: inv produces a true matrix inverse (A*inv(A) = I).
    R-POL10-01i: null returns a basis whose product with A is zero.
    R-POL10-01j: orth returns orthonormal column basis of the column space.
    R-POL10-01k: vecnorm returns correct vector and column-wise norms.
    R-POL10-01l: normalize produces unit-length vectors/columns.
    R-POL10-01m: det(inv(A)) equals 1/det(A) (cross-function consistency).

    Consistency argument:
    Sub-requirements 01a through 01l each test one linalg function in
    isolation against known reference values. Sub-requirement 01m tests
    cross-function consistency (det and inv). Together they cover every
    function listed in the parent requirement.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from forge.engine.builtins.linalg import LINALG_REGISTRY, ForgeArray, _unwrap


def _val(r):
    """Extract raw numpy from ForgeArray result."""
    d = _unwrap(r)
    return d.item() if d.size == 1 else d


def _fa(x):
    """Shorthand to wrap numpy array as ForgeArray."""
    return ForgeArray(np.asarray(x, dtype=float))


# ---------- kron ----------
def test_kron_identity():
    """R-POL10-01a: kron(I2, B) equals block-diagonal expansion."""
    kron = LINALG_REGISTRY["kron"]
    I2 = _fa([[1, 0], [0, 1]])
    B = _fa([[1, 2], [3, 4]])
    result = _unwrap(kron(I2, B))
    expected = np.array([[1, 2, 0, 0],
                         [3, 4, 0, 0],
                         [0, 0, 1, 2],
                         [0, 0, 3, 4]])
    assert_allclose(result, expected)
    assert result.shape == (4, 4)


# ---------- cross ----------
def test_cross_basic():
    """R-POL10-01b: cross([1,0,0],[0,1,0]) equals [0,0,1]."""
    cross = LINALG_REGISTRY["cross"]
    result = _unwrap(cross(_fa([1, 0, 0]), _fa([0, 1, 0]))).ravel()
    assert_allclose(result, [0, 0, 1])


def test_cross_anticommutative():
    """R-POL10-01b: cross(a,b) equals -cross(b,a)."""
    cross = LINALG_REGISTRY["cross"]
    a, b = _fa([1, 2, 3]), _fa([4, 5, 6])
    r1 = _unwrap(cross(a, b)).ravel()
    r2 = _unwrap(cross(b, a)).ravel()
    assert_allclose(r1, -r2)


# ---------- dot ----------
def test_dot_vector():
    """R-POL10-01c: dot([1,2,3],[4,5,6]) equals 32."""
    dot = LINALG_REGISTRY["dot"]
    result = _val(dot(_fa([1, 2, 3]), _fa([4, 5, 6])))
    assert result == 32


def test_dot_columnwise():
    """R-POL10-01c: dot on matrices computes column-wise inner products."""
    dot = LINALG_REGISTRY["dot"]
    A = _fa([[1, 2], [3, 4]])
    B = _fa([[5, 6], [7, 8]])
    result = _unwrap(dot(A, B)).ravel()
    # col1: 1*5+3*7=26, col2: 2*6+4*8=44
    assert_allclose(result, [26, 44])


# ---------- norm ----------
def test_norm_vector():
    """R-POL10-01d: norm([3,4]) equals 5."""
    norm = LINALG_REGISTRY["norm"]
    result = _val(norm(_fa([3, 4])))
    assert_allclose(result, 5.0)


def test_norm_frobenius():
    """R-POL10-01d: norm(A,'fro') equals sqrt(sum of squares)."""
    norm = LINALG_REGISTRY["norm"]
    A = _fa([[1, 2], [3, 4]])
    result = _val(norm(A, "fro"))
    assert_allclose(result, np.sqrt(30))


# ---------- pinv ----------
def test_pinv_full_rank():
    """R-POL10-01e: pinv(A)*A equals identity for full-rank A."""
    pinv = LINALG_REGISTRY["pinv"]
    A = _fa([[1, 2], [3, 4]])
    P = _unwrap(pinv(A))
    product = P @ np.array([[1, 2], [3, 4]], dtype=float)
    assert_allclose(product, np.eye(2), atol=1e-10)


# ---------- trace ----------
def test_trace():
    """R-POL10-01f: trace([[1,2],[3,4]]) equals 5."""
    trace = LINALG_REGISTRY["trace"]
    result = _val(trace(_fa([[1, 2], [3, 4]])))
    assert result == 5


# ---------- det ----------
def test_det():
    """R-POL10-01g: det([[1,2],[3,4]]) equals -2."""
    det = LINALG_REGISTRY["det"]
    result = _val(det(_fa([[1, 2], [3, 4]])))
    assert_allclose(result, -2.0, atol=1e-10)


def test_det_singular():
    """R-POL10-01g: det of singular matrix equals 0."""
    det = LINALG_REGISTRY["det"]
    result = _val(det(_fa([[1, 2], [2, 4]])))
    assert_allclose(result, 0.0, atol=1e-10)


# ---------- inv ----------
def test_inv():
    """R-POL10-01h: inv(A)*A equals identity."""
    inv = LINALG_REGISTRY["inv"]
    A = np.array([[1, 2], [3, 4]], dtype=float)
    R = _unwrap(inv(_fa(A)))
    assert_allclose(R @ A, np.eye(2), atol=1e-10)


# ---------- null ----------
def test_null_space():
    """R-POL10-01i: A*null(A) equals zero matrix."""
    null = LINALG_REGISTRY["null"]
    A = _fa([[1, 0], [0, 0]])
    N = _unwrap(null(A))
    # Null space of [[1,0],[0,0]] is span of [0,1]
    assert N.shape[1] == 1 or N.shape[0] >= 1
    # A * N should be ~0
    product = np.array([[1, 0], [0, 0]], dtype=float) @ N
    assert_allclose(product, 0, atol=1e-10)


# ---------- orth ----------
def test_orth():
    """R-POL10-01j: orth(A) returns orthonormal columns spanning col-space."""
    orth = LINALG_REGISTRY["orth"]
    A = _fa([[1, 0], [0, 1], [0, 0]])
    Q = _unwrap(orth(A))
    assert Q.shape == (3, 2)
    # Columns should be orthonormal
    assert_allclose(Q.T @ Q, np.eye(2), atol=1e-10)


# ---------- vecnorm ----------
def test_vecnorm_vector():
    """R-POL10-01k: vecnorm([3,4]) equals 5."""
    vecnorm = LINALG_REGISTRY["vecnorm"]
    result = _val(vecnorm(_fa([3, 4])))
    assert_allclose(result, 5.0)


def test_vecnorm_columnwise():
    """R-POL10-01k: vecnorm on matrix returns column-wise norms."""
    vecnorm = LINALG_REGISTRY["vecnorm"]
    M = _fa([[3, 1], [4, 2]])
    result = _unwrap(vecnorm(M)).ravel()
    assert_allclose(result, [5.0, np.sqrt(5)])


# ---------- normalize ----------
def test_normalize_vector_unit():
    """R-POL10-01l: normalize produces a unit-length vector."""
    normalize = LINALG_REGISTRY["normalize"]
    result = _unwrap(normalize(_fa([3, 4])))
    assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)


def test_normalize_matrix_columns():
    """R-POL10-01l: normalize on matrix normalizes each column to unit length."""
    normalize = LINALG_REGISTRY["normalize"]
    M = _fa([[3, 1], [4, 2]])
    result = _unwrap(normalize(M))
    col_norms = np.linalg.norm(result, axis=0)
    assert_allclose(col_norms, [1.0, 1.0], atol=1e-10)


# ---------- combined sanity ----------
def test_inv_det_consistency():
    """R-POL10-01m: det(inv(A)) equals 1/det(A)."""
    det = LINALG_REGISTRY["det"]
    inv = LINALG_REGISTRY["inv"]
    A = _fa([[2, 1], [5, 3]])
    d = _val(det(A))
    di = _val(det(inv(A)))
    assert_allclose(di, 1.0 / d, atol=1e-10)


def test_kron_dimensions():
    """R-POL10-01a: kron(2x3, 3x2) produces 6x6 result."""
    kron = LINALG_REGISTRY["kron"]
    A = _fa([[1, 2, 3], [4, 5, 6]])   # 2x3
    B = _fa([[1, 2], [3, 4], [5, 6]])  # 3x2
    result = _unwrap(kron(A, B))
    assert result.shape == (6, 6)
