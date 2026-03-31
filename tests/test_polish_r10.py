# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish R10 – Linear algebra operations.

15+ tests covering: kron, cross, dot, norm, pinv, trace, det, inv,
null, orth, vecnorm, normalize.
SRS trace: SRS-FUNC-001 (linalg toolbox)
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
    cross = LINALG_REGISTRY["cross"]
    result = _unwrap(cross(_fa([1, 0, 0]), _fa([0, 1, 0]))).ravel()
    assert_allclose(result, [0, 0, 1])


def test_cross_anticommutative():
    cross = LINALG_REGISTRY["cross"]
    a, b = _fa([1, 2, 3]), _fa([4, 5, 6])
    r1 = _unwrap(cross(a, b)).ravel()
    r2 = _unwrap(cross(b, a)).ravel()
    assert_allclose(r1, -r2)


# ---------- dot ----------
def test_dot_vector():
    dot = LINALG_REGISTRY["dot"]
    result = _val(dot(_fa([1, 2, 3]), _fa([4, 5, 6])))
    assert result == 32


def test_dot_columnwise():
    dot = LINALG_REGISTRY["dot"]
    A = _fa([[1, 2], [3, 4]])
    B = _fa([[5, 6], [7, 8]])
    result = _unwrap(dot(A, B)).ravel()
    # col1: 1*5+3*7=26, col2: 2*6+4*8=44
    assert_allclose(result, [26, 44])


# ---------- norm ----------
def test_norm_vector():
    norm = LINALG_REGISTRY["norm"]
    result = _val(norm(_fa([3, 4])))
    assert_allclose(result, 5.0)


def test_norm_frobenius():
    norm = LINALG_REGISTRY["norm"]
    A = _fa([[1, 2], [3, 4]])
    result = _val(norm(A, "fro"))
    assert_allclose(result, np.sqrt(30))


# ---------- pinv ----------
def test_pinv_full_rank():
    pinv = LINALG_REGISTRY["pinv"]
    A = _fa([[1, 2], [3, 4]])
    P = _unwrap(pinv(A))
    product = P @ np.array([[1, 2], [3, 4]], dtype=float)
    assert_allclose(product, np.eye(2), atol=1e-10)


# ---------- trace ----------
def test_trace():
    trace = LINALG_REGISTRY["trace"]
    result = _val(trace(_fa([[1, 2], [3, 4]])))
    assert result == 5


# ---------- det ----------
def test_det():
    det = LINALG_REGISTRY["det"]
    result = _val(det(_fa([[1, 2], [3, 4]])))
    assert_allclose(result, -2.0, atol=1e-10)


def test_det_singular():
    det = LINALG_REGISTRY["det"]
    result = _val(det(_fa([[1, 2], [2, 4]])))
    assert_allclose(result, 0.0, atol=1e-10)


# ---------- inv ----------
def test_inv():
    inv = LINALG_REGISTRY["inv"]
    A = np.array([[1, 2], [3, 4]], dtype=float)
    R = _unwrap(inv(_fa(A)))
    assert_allclose(R @ A, np.eye(2), atol=1e-10)


# ---------- null ----------
def test_null_space():
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
    orth = LINALG_REGISTRY["orth"]
    A = _fa([[1, 0], [0, 1], [0, 0]])
    Q = _unwrap(orth(A))
    assert Q.shape == (3, 2)
    # Columns should be orthonormal
    assert_allclose(Q.T @ Q, np.eye(2), atol=1e-10)


# ---------- vecnorm ----------
def test_vecnorm_vector():
    vecnorm = LINALG_REGISTRY["vecnorm"]
    result = _val(vecnorm(_fa([3, 4])))
    assert_allclose(result, 5.0)


def test_vecnorm_columnwise():
    vecnorm = LINALG_REGISTRY["vecnorm"]
    M = _fa([[3, 1], [4, 2]])
    result = _unwrap(vecnorm(M)).ravel()
    assert_allclose(result, [5.0, np.sqrt(5)])


# ---------- normalize ----------
def test_normalize_vector_unit():
    normalize = LINALG_REGISTRY["normalize"]
    result = _unwrap(normalize(_fa([3, 4])))
    assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)


def test_normalize_matrix_columns():
    normalize = LINALG_REGISTRY["normalize"]
    M = _fa([[3, 1], [4, 2]])
    result = _unwrap(normalize(M))
    col_norms = np.linalg.norm(result, axis=0)
    assert_allclose(col_norms, [1.0, 1.0], atol=1e-10)


# ---------- combined sanity ----------
def test_inv_det_consistency():
    """det(inv(A)) == 1/det(A)"""
    det = LINALG_REGISTRY["det"]
    inv = LINALG_REGISTRY["inv"]
    A = _fa([[2, 1], [5, 3]])
    d = _val(det(A))
    di = _val(det(inv(A)))
    assert_allclose(di, 1.0 / d, atol=1e-10)


def test_kron_dimensions():
    """kron(m x n, p x q) should be (m*p) x (n*q)."""
    kron = LINALG_REGISTRY["kron"]
    A = _fa([[1, 2, 3], [4, 5, 6]])   # 2x3
    B = _fa([[1, 2], [3, 4], [5, 6]])  # 3x2
    result = _unwrap(kron(A, B))
    assert result.shape == (6, 6)
