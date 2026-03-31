# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for advanced linear algebra: decompositions, matrix functions, properties.

SRS trace: SRS-FUNC-001, SRS-VAL-001
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.linalg import (
    _forge_svd, _forge_lu, _forge_qr, _forge_chol,
    forge_expm, forge_logm, forge_funm,
    forge_rank, forge_cond,
    forge_issymmetric, forge_ishermitian,
    forge_isdiag, forge_istriu, forge_istril,
)
from forge.engine.session import ForgeSession


@pytest.fixture(scope="module")
def s():
    return ForgeSession()


def _get(s, name):
    """Get workspace variable as numpy array."""
    v = s.workspace.get(name)
    return _unwrap(v) if isinstance(v, ForgeArray) else np.atleast_2d(v)


# ---------------------------------------------------------------------------
# 1. Matrix Decompositions
# ---------------------------------------------------------------------------

class TestEig:

    def test_eig_eigenvalues(self, s):
        """eig([2 1; 1 3]) eigenvalues ~ 1.382, 3.618"""
        s.eval("[Ve,De] = eig([2 1; 1 3])")
        D = _get(s, "De")
        vals = sorted([D[0, 0], D[1, 1]])
        assert abs(vals[0] - (5 - np.sqrt(5)) / 2) < 1e-10
        assert abs(vals[1] - (5 + np.sqrt(5)) / 2) < 1e-10

    def test_eig_reconstruction(self, s):
        """V*D*inv(V) ~ A"""
        s.eval("Ae = [2 1; 1 3]")
        s.eval("[Ve2,De2] = eig(Ae)")
        s.eval("eig_err = max(max(abs(Ve2*De2*inv(Ve2) - Ae)))")
        err = float(_get(s, "eig_err").flat[0])
        assert err < 1e-10


class TestSvd:

    def test_svd_reconstruction_nonsquare(self):
        """U*S*V_transpose ~ A for 2x3 matrix"""
        A = ForgeArray(np.array([[3., 2., 2.], [2., 3., -2.]]))
        U, S, V = _forge_svd(A)
        recon = _unwrap(U) @ _unwrap(S) @ _unwrap(V).T
        assert np.allclose(recon, _unwrap(A), atol=1e-10)

    def test_svd_S_dimensions(self):
        """S should be same size as A"""
        A = ForgeArray(np.array([[3., 2., 2.], [2., 3., -2.]]))
        U, S, V = _forge_svd(A)
        assert _unwrap(S).shape == (2, 3)

    def test_svd_square(self):
        """SVD of square matrix"""
        A = ForgeArray(np.array([[1., 2.], [3., 4.]]))
        U, S, V = _forge_svd(A)
        recon = _unwrap(U) @ _unwrap(S) @ _unwrap(V).T
        assert np.allclose(recon, _unwrap(A), atol=1e-10)


class TestLu:

    def test_lu_reconstruction(self):
        """P*A = L*U"""
        A = ForgeArray(np.array([[1., 2.], [3., 4.]]))
        L, U, P = _forge_lu(A)
        assert np.allclose(_unwrap(P) @ _unwrap(A), _unwrap(L) @ _unwrap(U), atol=1e-10)


class TestQr:

    def test_qr_reconstruction(self):
        """Q*R ~ A"""
        A = ForgeArray(np.array([[1., 2.], [3., 4.]]))
        Q, R = _forge_qr(A)
        assert np.allclose(_unwrap(Q) @ _unwrap(R), _unwrap(A), atol=1e-10)

    def test_qr_orthogonal(self):
        """Q_transpose * Q ~ I"""
        A = ForgeArray(np.array([[1., 2.], [3., 4.]]))
        Q, R = _forge_qr(A)
        Qd = _unwrap(Q)
        assert np.allclose(Qd.T @ Qd, np.eye(2), atol=1e-10)


class TestChol:

    def test_chol_upper_triangular(self):
        """chol returns upper triangular"""
        A = ForgeArray(np.array([[4., 2.], [2., 3.]]))
        R = _forge_chol(A)
        Rd = _unwrap(R)
        assert np.allclose(Rd, np.triu(Rd))

    def test_chol_reconstruction(self):
        """R_transpose * R = A"""
        A = ForgeArray(np.array([[4., 2.], [2., 3.]]))
        R = _forge_chol(A)
        Rd = _unwrap(R)
        assert np.allclose(Rd.T @ Rd, _unwrap(A), atol=1e-10)

    def test_chol_3x3(self):
        """chol of 3x3 positive definite"""
        M = np.array([[4., 2., 1.], [2., 5., 3.], [1., 3., 6.]])
        A = ForgeArray(M)
        R = _forge_chol(A)
        Rd = _unwrap(R)
        assert np.allclose(Rd.T @ Rd, M, atol=1e-10)
        assert np.allclose(Rd, np.triu(Rd))


class TestSchur:

    def test_schur_reconstruction(self, s):
        """U*T*U_tr ~ A"""
        s.eval("Aschur = [1 2; 3 4]")
        s.eval("[Usch,Tsch] = schur(Aschur)")
        U = _get(s, "Usch")
        T = _get(s, "Tsch")
        A = _get(s, "Aschur")
        assert np.allclose(U @ T @ U.T, A, atol=1e-10)

    def test_schur_unitary(self, s):
        """U should be unitary"""
        s.eval("[Usch2,Tsch2] = schur([1 2; 3 4])")
        U = _get(s, "Usch2")
        assert np.allclose(U.T @ U, np.eye(2), atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Matrix Functions
# ---------------------------------------------------------------------------

class TestExpm:

    def test_expm_nilpotent(self):
        """expm([0 1; 0 0]) = [[1 1],[0 1]]"""
        A = ForgeArray(np.array([[0., 1.], [0., 0.]]))
        r = _unwrap(forge_expm(A))
        assert np.allclose(r, np.array([[1, 1], [0, 1]]), atol=1e-10)


class TestLogm:

    def test_logm_inverse_of_expm(self):
        """logm(expm(A)) = A for diagonal A"""
        A = ForgeArray(np.array([[1., 0.], [0., 2.]]))
        r = _unwrap(forge_logm(forge_expm(A)))
        assert np.allclose(r, _unwrap(A), atol=1e-10)


class TestSqrtm:

    def test_sqrtm_diagonal(self, s):
        """sqrtm([4 0; 0 9]) = [[2 0],[0 3]]"""
        s.eval("sqm = sqrtm([4 0; 0 9])")
        r = _get(s, "sqm")
        assert np.allclose(r, np.array([[2, 0], [0, 3]]), atol=1e-10)


class TestFunm:

    def test_funm_exp_diagonal(self, s):
        """funm(A, @exp) for diagonal matrix"""
        s.eval("fm = funm([1 0; 0 2], @exp)")
        r = _get(s, "fm")
        assert abs(r[0, 0] - np.exp(1)) < 1e-4
        assert abs(r[1, 1] - np.exp(2)) < 1e-4
        assert abs(r[0, 1]) < 1e-4
        assert abs(r[1, 0]) < 1e-4


# ---------------------------------------------------------------------------
# 3. Condition and Rank
# ---------------------------------------------------------------------------

class TestRank:

    def test_rank_deficient(self):
        A = ForgeArray(np.array([[1., 0.], [0., 0.]]))
        assert int(_unwrap(forge_rank(A)).flat[0]) == 1

    def test_rank_full(self):
        assert int(_unwrap(forge_rank(ForgeArray(np.eye(3)))).flat[0]) == 3


class TestCond:

    def test_cond_diagonal(self):
        A = ForgeArray(np.array([[1., 0.], [0., 2.]]))
        assert abs(float(_unwrap(forge_cond(A)).flat[0]) - 2.0) < 1e-10


class TestRcond:

    def test_rcond_diagonal(self, s):
        """rcond([1 0; 0 2]) = 0.5"""
        s.eval("rc = rcond([1 0; 0 2])")
        r = float(_get(s, "rc").flat[0])
        assert abs(r - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# 4. Matrix Properties
# ---------------------------------------------------------------------------

class TestIssymmetric:

    def test_symmetric_true(self):
        A = ForgeArray(np.array([[1., 2.], [2., 3.]]))
        assert _unwrap(forge_issymmetric(A)).flat[0] == True

    def test_symmetric_false(self):
        A = ForgeArray(np.array([[1., 2.], [3., 4.]]))
        assert _unwrap(forge_issymmetric(A)).flat[0] == False


class TestIshermitian:

    def test_hermitian_real_symmetric(self):
        A = ForgeArray(np.array([[1., 2.], [2., 3.]]))
        assert _unwrap(forge_ishermitian(A)).flat[0] == True


class TestIsdiag:

    def test_diag_true(self):
        assert _unwrap(forge_isdiag(ForgeArray(np.diag([1., 2.])))).flat[0] == True

    def test_diag_false(self):
        assert _unwrap(forge_isdiag(ForgeArray(np.array([[1., 1.], [0., 2.]])))).flat[0] == False


class TestIstriu:

    def test_upper_true(self):
        A = ForgeArray(np.array([[1., 2.], [0., 3.]]))
        assert _unwrap(forge_istriu(A)).flat[0] == True

    def test_upper_false(self):
        A = ForgeArray(np.array([[1., 2.], [1., 3.]]))
        assert _unwrap(forge_istriu(A)).flat[0] == False


class TestIstril:

    def test_lower_true(self):
        A = ForgeArray(np.array([[1., 0.], [2., 3.]]))
        assert _unwrap(forge_istril(A)).flat[0] == True

    def test_lower_false(self):
        A = ForgeArray(np.array([[1., 1.], [2., 3.]]))
        assert _unwrap(forge_istril(A)).flat[0] == False
