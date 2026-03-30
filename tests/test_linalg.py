# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for linear algebra toolbox.

SRS trace: SRS-FUNC-001, SRS-VAL-001
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.linalg import *


class TestMatrixProperties:

    def test_rank_full(self):
        A = ForgeArray(np.eye(3))
        assert int(_unwrap(forge_rank(A)).flat[0]) == 3

    def test_rank_deficient(self):
        A = ForgeArray(np.array([[1.0, 2.0], [2.0, 4.0]]))
        assert int(_unwrap(forge_rank(A)).flat[0]) == 1

    def test_cond_identity(self):
        A = ForgeArray(np.eye(3))
        r = float(_unwrap(forge_cond(A)).flat[0])
        assert abs(r - 1.0) < 1e-10

    def test_trace_identity(self):
        A = ForgeArray(np.eye(4))
        assert abs(float(_unwrap(forge_trace(A)).flat[0]) - 4.0) < 1e-14

    def test_trace_known(self):
        A = ForgeArray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert abs(float(_unwrap(forge_trace(A)).flat[0]) - 5.0) < 1e-14


class TestMatrixStructure:

    def test_isdiag_true(self):
        A = ForgeArray(np.diag([1.0, 2.0, 3.0]))
        assert _unwrap(forge_isdiag(A)).flat[0] == True

    def test_isdiag_false(self):
        A = ForgeArray(np.array([[1.0, 1.0], [0.0, 1.0]]))
        assert _unwrap(forge_isdiag(A)).flat[0] == False

    def test_istril_true(self):
        A = ForgeArray(np.tril(np.ones((3, 3))))
        assert _unwrap(forge_istril(A)).flat[0] == True

    def test_istriu_true(self):
        A = ForgeArray(np.triu(np.ones((3, 3))))
        assert _unwrap(forge_istriu(A)).flat[0] == True

    def test_issymmetric(self):
        A = ForgeArray(np.array([[1.0, 2.0], [2.0, 3.0]]))
        assert _unwrap(forge_issymmetric(A)).flat[0] == True

    def test_issymmetric_false(self):
        A = ForgeArray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert _unwrap(forge_issymmetric(A)).flat[0] == False

    def test_ishermitian_real(self):
        A = ForgeArray(np.array([[1.0, 2.0], [2.0, 3.0]]))
        assert _unwrap(forge_ishermitian(A)).flat[0] == True

    def test_isdefinite_positive(self):
        A = ForgeArray(np.eye(3))
        assert int(_unwrap(forge_isdefinite(A)).flat[0]) == 1

    def test_bandwidth_diagonal(self):
        A = ForgeArray(np.diag([1.0, 2.0, 3.0]))
        lo, up = forge_bandwidth(A)
        assert int(_unwrap(lo).flat[0]) == 0
        assert int(_unwrap(up).flat[0]) == 0

    def test_bandwidth_tridiag(self):
        A = ForgeArray(np.array([[1, 1, 0], [1, 2, 1], [0, 1, 3.0]]))
        lo, up = forge_bandwidth(A)
        assert int(_unwrap(lo).flat[0]) == 1
        assert int(_unwrap(up).flat[0]) == 1

    def test_isbanded(self):
        A = ForgeArray(np.diag([1.0, 2.0, 3.0]))
        assert _unwrap(forge_isbanded(A, ForgeArray(0), ForgeArray(0))).flat[0] == True


class TestSubspaces:

    def test_null_identity(self):
        """Null space of identity is empty."""
        A = ForgeArray(np.eye(3))
        N = _unwrap(forge_null(A))
        assert N.shape[1] == 0 or N.size == 0

    def test_null_rank_deficient(self):
        A = ForgeArray(np.array([[1.0, 2.0], [2.0, 4.0]]))
        N = _unwrap(forge_null(A))
        # Null space should be 1D
        assert N.shape[1] >= 1

    def test_orth_full_rank(self):
        A = ForgeArray(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
        Q = _unwrap(forge_orth(A))
        assert Q.shape[1] == 2

    def test_cross_product(self):
        a = ForgeArray(np.array([1.0, 0.0, 0.0]))
        b = ForgeArray(np.array([0.0, 1.0, 0.0]))
        r = _unwrap(forge_cross(a, b)).ravel()
        np.testing.assert_allclose(r, [0, 0, 1], atol=1e-14)


class TestMatrixFunctions:

    def test_expm_zero(self):
        """expm(0) = I."""
        A = ForgeArray(np.zeros((3, 3)))
        r = _unwrap(forge_expm(A))
        np.testing.assert_allclose(r, np.eye(3), atol=1e-14)

    def test_logm_identity(self):
        """logm(I) = 0."""
        A = ForgeArray(np.eye(3))
        r = _unwrap(forge_logm(A))
        np.testing.assert_allclose(r, np.zeros((3, 3)), atol=1e-14)

    def test_expm_logm_roundtrip(self):
        A = ForgeArray(np.array([[1.0, 0.5], [0.0, 2.0]]))
        r = _unwrap(forge_logm(ForgeArray(_unwrap(forge_expm(A)))))
        np.testing.assert_allclose(r, _unwrap(A), atol=1e-10)


class TestSolvers:

    def test_linsolve_simple(self):
        A = ForgeArray(np.array([[2.0, 1.0], [1.0, 3.0]]))
        b = ForgeArray(np.array([[5.0], [7.0]]))
        x = _unwrap(forge_linsolve(A, b)).ravel()
        np.testing.assert_allclose(x, [1.6, 1.8], atol=1e-10)

    def test_rref_identity(self):
        A = ForgeArray(np.eye(3))
        r = _unwrap(forge_rref(A))
        np.testing.assert_allclose(r, np.eye(3), atol=1e-14)

    def test_rref_augmented(self):
        A = ForgeArray(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        r = _unwrap(forge_rref(A))
        # Should be in row echelon form
        assert abs(r[0, 0] - 1.0) < 1e-12
        assert abs(r[1, 1] - 1.0) < 1e-12

    def test_lscov(self):
        A = ForgeArray(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]))
        b = ForgeArray(np.array([2.0, 3.0, 4.0]))
        x = _unwrap(forge_lscov(A, b)).ravel()
        assert abs(x[1] - 1.0) < 1e-10  # slope should be ~1

    def test_ols(self):
        X = ForgeArray(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]))
        y = ForgeArray(np.array([2.0, 4.0, 6.0]))
        beta = _unwrap(forge_ols(y, X)).ravel()
        assert abs(beta[1] - 2.0) < 1e-10


class TestMisc:

    def test_vech(self):
        A = ForgeArray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        r = _unwrap(forge_vech(A)).ravel()
        np.testing.assert_array_equal(r, [1, 3, 4])

    def test_vecnorm(self):
        x = ForgeArray(np.array([3.0, 4.0]))
        r = float(_unwrap(forge_vecnorm(x)).flat[0])
        assert abs(r - 5.0) < 1e-14

    def test_planerot(self):
        x = ForgeArray(np.array([3.0, 4.0]))
        G, y = forge_planerot(x)
        assert abs(_unwrap(y).ravel()[0] - 5.0) < 1e-14
        assert abs(_unwrap(y).ravel()[1]) < 1e-14
