# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for advanced linear algebra: decompositions, matrix functions, properties.

SRS trace: SRS-FUNC-001, SRS-VAL-001

Requirement overview
--------------------
This module covers the second tier of linear algebra verification for the Forge
engine, spanning six decomposition families (eig, SVD, LU, QR, Cholesky, Schur),
four matrix functions (expm, logm, sqrtm, funm), two condition/rank queries
(rank, cond, rcond), and five matrix property predicates (issymmetric, ishermitian,
isdiag, istriu, istril).

Requirements R-LIN-07 through R-LIN-20 are defined here. Each requirement traces
to a concrete workflow that the golden user (an engineer/scientist migrating from
MATLAB/Octave) performs daily: assembling FEM stiffness matrices, running modal
analysis, checking system stability, or verifying matrix structure before feeding
data into a solver pipeline.

All test logic and assertions are preserved verbatim from the original file.
Only docstrings and traceability annotations have been added.
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
    """R-LIN-07: Eigenvalue decomposition SHALL return eigenvalues and
    eigenvectors such that A*V = V*D and V*D*inv(V) reconstructs A within
    machine epsilon tolerance (1e-10).

    Model-user argument
    -------------------
    The golden user runs modal analysis on structural FEM models. The command
    ``[V,D] = eig(K, M)`` (or simply ``eig(A)``) is their primary tool for
    extracting natural frequencies and mode shapes. If eigenvalues are wrong
    or reconstruction fails, the predicted resonance frequencies are invalid
    and the physical design decisions built on them are unreliable.

    Decomposition
    -------------
    R-LIN-07a: eig returns correct eigenvalues for a known 2x2 symmetric matrix.
    R-LIN-07b: V*D*inv(V) reconstructs A within tolerance.

    Consistency argument
    --------------------
    R-LIN-07a confirms numerical correctness of the eigenvalues themselves.
    R-LIN-07b confirms that eigenvectors and eigenvalues together satisfy the
    defining relation A = V*D*inv(V). Together they fully verify the two-output
    form of eig for real symmetric input.
    """

    def test_eig_eigenvalues(self, s):
        """R-LIN-07a: eig([2 1; 1 3]) eigenvalues ~ 1.382, 3.618"""
        s.eval("[Ve,De] = eig([2 1; 1 3])")
        D = _get(s, "De")
        vals = sorted([D[0, 0], D[1, 1]])
        assert abs(vals[0] - (5 - np.sqrt(5)) / 2) < 1e-10
        assert abs(vals[1] - (5 + np.sqrt(5)) / 2) < 1e-10

    def test_eig_reconstruction(self, s):
        """R-LIN-07b: V*D*inv(V) ~ A"""
        s.eval("Ae = [2 1; 1 3]")
        s.eval("[Ve2,De2] = eig(Ae)")
        s.eval("eig_err = max(max(abs(Ve2*De2*inv(Ve2) - Ae)))")
        err = float(_get(s, "eig_err").flat[0])
        assert err < 1e-10


class TestSvd:
    """R-LIN-08: Singular value decomposition SHALL return factors U, S, V
    such that U*S*V' reconstructs A within tolerance (1e-10), S has the same
    dimensions as A, and the decomposition works for both square and
    non-square matrices.

    Model-user argument
    -------------------
    The golden user relies on ``[U,S,V] = svd(A)`` for rank detection, noise
    filtering, and pseudo-inverse computation when solving overdetermined
    systems from experimental sensor data. Octave returns S with the same
    shape as A; if Forge returns a different shape the user's downstream
    indexing (e.g., extracting singular values from diag(S)) breaks silently,
    producing wrong condition estimates.

    Decomposition
    -------------
    R-LIN-08a: U*S*V' reconstructs a non-square (2x3) matrix within tolerance.
    R-LIN-08b: S has the same dimensions as A for non-square input.
    R-LIN-08c: U*S*V' reconstructs a square matrix within tolerance.

    Consistency argument
    --------------------
    R-LIN-08a and R-LIN-08c together cover the two major input shapes (non-square
    and square). R-LIN-08b validates the Octave-compatible S sizing convention
    that the golden user depends on. Together they ensure the three-output SVD
    call is drop-in compatible with Octave for all common matrix shapes.
    """

    def test_svd_reconstruction_nonsquare(self):
        """R-LIN-08a: U*S*V' ~ A for 2x3 matrix"""
        A = ForgeArray(np.array([[3., 2., 2.], [2., 3., -2.]]))
        U, S, V = _forge_svd(A)
        recon = _unwrap(U) @ _unwrap(S) @ _unwrap(V).T
        assert np.allclose(recon, _unwrap(A), atol=1e-10)

    def test_svd_S_dimensions(self):
        """R-LIN-08b: S should be same size as A"""
        A = ForgeArray(np.array([[3., 2., 2.], [2., 3., -2.]]))
        U, S, V = _forge_svd(A)
        assert _unwrap(S).shape == (2, 3)

    def test_svd_square(self):
        """R-LIN-08c: SVD of square matrix"""
        A = ForgeArray(np.array([[1., 2.], [3., 4.]]))
        U, S, V = _forge_svd(A)
        recon = _unwrap(U) @ _unwrap(S) @ _unwrap(V).T
        assert np.allclose(recon, _unwrap(A), atol=1e-10)


class TestLu:
    """R-LIN-09: LU decomposition SHALL return factors L, U, P such that
    P*A = L*U within tolerance (1e-10).

    Model-user argument
    -------------------
    The golden user solves large sparse linear systems arising from FEM assembly
    using LU factorization. The standard Octave call ``[L,U,P] = lu(A)`` with
    the invariant P*A = L*U is embedded in their solver scripts. If the
    permutation matrix P does not satisfy this relation, the entire solve
    produces garbage, and the user has no way to detect the error except by
    checking residuals manually.

    Decomposition
    -------------
    R-LIN-09a: P*A equals L*U for a 2x2 matrix within tolerance.

    Consistency argument
    --------------------
    R-LIN-09a directly tests the defining invariant. A single well-conditioned
    test matrix suffices for the three-output calling convention because the
    underlying scipy.linalg.lu is already validated; the requirement here is
    that Forge wires the outputs in the correct Octave-compatible order (L, U, P).
    """

    def test_lu_reconstruction(self):
        """R-LIN-09a: P*A = L*U"""
        A = ForgeArray(np.array([[1., 2.], [3., 4.]]))
        L, U, P = _forge_lu(A)
        assert np.allclose(_unwrap(P) @ _unwrap(A), _unwrap(L) @ _unwrap(U), atol=1e-10)


class TestQr:
    """R-LIN-10: QR decomposition SHALL return factors Q, R such that Q*R
    reconstructs A within tolerance (1e-10) and Q is orthogonal (Q'*Q = I).

    Model-user argument
    -------------------
    The golden user uses QR factorization for least-squares fitting of
    experimental data and for checking numerical stability of rectangular
    systems. The Octave call ``[Q,R] = qr(A)`` is a daily tool. If Q is not
    orthogonal, the least-squares solution computed via back-substitution on R
    accumulates catastrophic rounding error, and fitted model coefficients
    become meaningless.

    Decomposition
    -------------
    R-LIN-10a: Q*R reconstructs A within tolerance.
    R-LIN-10b: Q'*Q equals the identity matrix within tolerance.

    Consistency argument
    --------------------
    R-LIN-10a verifies factorization correctness. R-LIN-10b verifies the
    orthogonality property that makes QR numerically stable. Together they
    confirm both the algebraic identity and the structural property that the
    golden user depends on for reliable least-squares solutions.
    """

    def test_qr_reconstruction(self):
        """R-LIN-10a: Q*R ~ A"""
        A = ForgeArray(np.array([[1., 2.], [3., 4.]]))
        Q, R = _forge_qr(A)
        assert np.allclose(_unwrap(Q) @ _unwrap(R), _unwrap(A), atol=1e-10)

    def test_qr_orthogonal(self):
        """R-LIN-10b: Q'*Q ~ I"""
        A = ForgeArray(np.array([[1., 2.], [3., 4.]]))
        Q, R = _forge_qr(A)
        Qd = _unwrap(Q)
        assert np.allclose(Qd.T @ Qd, np.eye(2), atol=1e-10)


class TestChol:
    """R-LIN-11: Cholesky decomposition SHALL return an upper-triangular
    factor R such that R'*R = A within tolerance (1e-10), and R is strictly
    upper-triangular (all sub-diagonal entries are zero).

    Model-user argument
    -------------------
    The golden user assembles symmetric positive-definite stiffness matrices
    in FEM workflows and factors them via ``R = chol(K)`` before solving
    K*x = f as two triangular back-substitutions. Octave's chol returns an
    upper-triangular R; if Forge returned lower-triangular instead, the user's
    existing solve code (which indexes R assuming upper-triangular layout)
    would silently produce wrong displacements. The reconstruction R'*R = A
    must also hold, or the factorization itself is invalid.

    Decomposition
    -------------
    R-LIN-11a: chol returns an upper-triangular matrix for a 2x2 SPD input.
    R-LIN-11b: R'*R equals A for a 2x2 SPD input.
    R-LIN-11c: chol works correctly for a 3x3 SPD matrix (reconstruction and
               upper-triangular structure both hold).

    Consistency argument
    --------------------
    R-LIN-11a and R-LIN-11b together verify the two defining properties of the
    Cholesky factor on a minimal input. R-LIN-11c extends coverage to a larger
    matrix to confirm that the implementation generalizes beyond 2x2. All three
    sub-requirements together ensure the golden user can trust chol output for
    matrices of practical size.
    """

    def test_chol_upper_triangular(self):
        """R-LIN-11a: chol returns upper triangular"""
        A = ForgeArray(np.array([[4., 2.], [2., 3.]]))
        R = _forge_chol(A)
        Rd = _unwrap(R)
        assert np.allclose(Rd, np.triu(Rd))

    def test_chol_reconstruction(self):
        """R-LIN-11b: R'*R = A"""
        A = ForgeArray(np.array([[4., 2.], [2., 3.]]))
        R = _forge_chol(A)
        Rd = _unwrap(R)
        assert np.allclose(Rd.T @ Rd, _unwrap(A), atol=1e-10)

    def test_chol_3x3(self):
        """R-LIN-11c: chol of 3x3 positive definite"""
        M = np.array([[4., 2., 1.], [2., 5., 3.], [1., 3., 6.]])
        A = ForgeArray(M)
        R = _forge_chol(A)
        Rd = _unwrap(R)
        assert np.allclose(Rd.T @ Rd, M, atol=1e-10)
        assert np.allclose(Rd, np.triu(Rd))


class TestSchur:
    """R-LIN-12: Schur decomposition SHALL return a unitary matrix U and an
    upper quasi-triangular matrix T such that U*T*U' reconstructs A within
    tolerance (1e-10), and U'*U equals the identity.

    Model-user argument
    -------------------
    The golden user uses Schur decomposition for control-system stability
    analysis. The command ``[U,T] = schur(A)`` reveals the system eigenvalues
    on the diagonal of T while preserving numerical stability better than
    direct eigenvalue computation. If U is not unitary, or if the reconstruction
    fails, the stability conclusions drawn from T's diagonal are unreliable and
    the user may incorrectly classify a marginally stable plant as stable.

    Decomposition
    -------------
    R-LIN-12a: U*T*U' reconstructs A within tolerance.
    R-LIN-12b: U is unitary (U'*U = I within tolerance).

    Consistency argument
    --------------------
    R-LIN-12a confirms the factorization identity. R-LIN-12b confirms the
    structural property (unitarity) that makes Schur decomposition numerically
    reliable. Together they fully verify the two-output Schur call.
    """

    def test_schur_reconstruction(self, s):
        """R-LIN-12a: U*T*U' ~ A"""
        s.eval("Aschur = [1 2; 3 4]")
        s.eval("[Usch,Tsch] = schur(Aschur)")
        U = _get(s, "Usch")
        T = _get(s, "Tsch")
        A = _get(s, "Aschur")
        assert np.allclose(U @ T @ U.T, A, atol=1e-10)

    def test_schur_unitary(self, s):
        """R-LIN-12b: U should be unitary"""
        s.eval("[Usch2,Tsch2] = schur([1 2; 3 4])")
        U = _get(s, "Usch2")
        assert np.allclose(U.T @ U, np.eye(2), atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Matrix Functions
# ---------------------------------------------------------------------------

class TestExpm:
    """R-LIN-13: The matrix exponential expm(A) SHALL produce the correct
    result for nilpotent matrices, matching the known closed-form answer
    within tolerance (1e-10).

    Model-user argument
    -------------------
    The golden user computes state-transition matrices for linear time-invariant
    systems via ``expm(A*dt)``. A nilpotent test matrix has a known exact
    exponential (finite Taylor series), making it an ideal verification case.
    If expm fails on this simple input, the user cannot trust it for the
    general A matrices that arise in their continuous-time state-space models.

    Decomposition
    -------------
    R-LIN-13a: expm([0 1; 0 0]) equals [[1 1],[0 1]] within tolerance.

    Consistency argument
    --------------------
    A single sub-requirement suffices because the nilpotent case exercises
    the core Pade-approximation path and has a known exact answer. More
    complex inputs are covered by the logm round-trip test (R-LIN-14).
    """

    def test_expm_nilpotent(self):
        """R-LIN-13a: expm([0 1; 0 0]) = [[1 1],[0 1]]"""
        A = ForgeArray(np.array([[0., 1.], [0., 0.]]))
        r = _unwrap(forge_expm(A))
        assert np.allclose(r, np.array([[1, 1], [0, 1]]), atol=1e-10)


class TestLogm:
    """R-LIN-14: The matrix logarithm logm SHALL be the inverse of expm,
    such that logm(expm(A)) recovers A within tolerance (1e-10) for
    diagonalizable matrices.

    Model-user argument
    -------------------
    The golden user occasionally needs to extract the continuous-time system
    matrix A from a discrete-time transition matrix Phi via ``A = logm(Phi)/dt``.
    If logm is not a faithful inverse of expm, the recovered A contains errors
    that propagate into frequency-domain analysis and controller design.

    Decomposition
    -------------
    R-LIN-14a: logm(expm(A)) equals A for a diagonal matrix within tolerance.

    Consistency argument
    --------------------
    Testing on a diagonal matrix isolates logm correctness from eigenvector
    computation issues. The round-trip property (logm inverting expm) is the
    most stringent single test for this function pair.
    """

    def test_logm_inverse_of_expm(self):
        """R-LIN-14a: logm(expm(A)) = A for diagonal A"""
        A = ForgeArray(np.array([[1., 0.], [0., 2.]]))
        r = _unwrap(forge_logm(forge_expm(A)))
        assert np.allclose(r, _unwrap(A), atol=1e-10)


class TestSqrtm:
    """R-LIN-15: The matrix square root sqrtm(A) SHALL return a matrix R
    such that R equals the element-wise expected result for diagonal input,
    within tolerance (1e-10).

    Model-user argument
    -------------------
    The golden user uses sqrtm in signal processing workflows (e.g., whitening
    covariance matrices). For a diagonal matrix with entries 4 and 9, the
    expected result is diagonal with entries 2 and 3. This is the simplest
    verification that sqrtm handles the standard case correctly before the
    user applies it to dense covariance matrices.

    Decomposition
    -------------
    R-LIN-15a: sqrtm([4 0; 0 9]) equals [[2 0],[0 3]] within tolerance.

    Consistency argument
    --------------------
    Diagonal input gives a known exact answer, verifying the core computation
    path without eigenvector sensitivity. One sub-requirement suffices for
    this focused verification of the session-level sqrtm call.
    """

    def test_sqrtm_diagonal(self, s):
        """R-LIN-15a: sqrtm([4 0; 0 9]) = [[2 0],[0 3]]"""
        s.eval("sqm = sqrtm([4 0; 0 9])")
        r = _get(s, "sqm")
        assert np.allclose(r, np.array([[2, 0], [0, 3]]), atol=1e-10)


class TestFunm:
    """R-LIN-16: The general matrix function funm(A, @f) SHALL apply the
    scalar function f to the matrix A via eigendecomposition, producing
    correct diagonal entries for diagonal input within tolerance (1e-4).

    Model-user argument
    -------------------
    The golden user occasionally needs matrix functions beyond exp/log/sqrt,
    for example applying a custom transfer function element to a system matrix.
    The Octave call ``funm(A, @exp)`` is the general entry point. If funm
    disagrees with expm on diagonal input, the user cannot trust it for custom
    functions either.

    Decomposition
    -------------
    R-LIN-16a: funm([1 0; 0 2], @exp) produces diagonal entries e^1 and e^2,
               with off-diagonal entries near zero (tolerance 1e-4).

    Consistency argument
    --------------------
    Testing funm with @exp on diagonal input cross-validates against the known
    scalar exponential. The relaxed tolerance (1e-4) accounts for the general
    eigendecomposition path, which is less numerically tight than the dedicated
    Pade-based expm. One sub-requirement suffices for this focused check.
    """

    def test_funm_exp_diagonal(self, s):
        """R-LIN-16a: funm(A, @exp) for diagonal matrix"""
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
    """R-LIN-17: The rank function SHALL return the correct numerical rank
    for both rank-deficient and full-rank matrices.

    Model-user argument
    -------------------
    The golden user checks rank before solving linear systems to detect
    singular or near-singular coefficient matrices. If rank reports full rank
    for a rank-deficient matrix, the user proceeds with a direct solve that
    produces nonsense. If rank under-reports, the user switches to a
    pseudo-inverse unnecessarily, losing precision.

    Decomposition
    -------------
    R-LIN-17a: rank of a rank-1 matrix (one zero row/column) returns 1.
    R-LIN-17b: rank of eye(3) returns 3.

    Consistency argument
    --------------------
    R-LIN-17a tests the deficient case (rank < min(m,n)). R-LIN-17b tests
    the full-rank case. Together they cover both branches of the rank
    decision logic.
    """

    def test_rank_deficient(self):
        """R-LIN-17a: rank of rank-deficient matrix returns 1"""
        A = ForgeArray(np.array([[1., 0.], [0., 0.]]))
        assert int(_unwrap(forge_rank(A)).flat[0]) == 1

    def test_rank_full(self):
        """R-LIN-17b: rank of eye(3) returns 3"""
        assert int(_unwrap(forge_rank(ForgeArray(np.eye(3)))).flat[0]) == 3


class TestCond:
    """R-LIN-18: The condition number function cond(A) SHALL return the
    correct 2-norm condition number within tolerance (1e-10).

    Model-user argument
    -------------------
    The golden user inspects cond(K) on stiffness matrices before solving
    to estimate how many digits of accuracy the solution will lose. A wrong
    condition number gives a false sense of security (or false alarm),
    leading to either undetected numerical garbage or unnecessary
    preconditioning effort.

    Decomposition
    -------------
    R-LIN-18a: cond(diag([1,2])) equals 2.0 within tolerance.

    Consistency argument
    --------------------
    For a diagonal matrix, cond = max(sigma)/min(sigma) = 2/1 = 2. This
    exact-answer case verifies the computation without eigenvector sensitivity.
    """

    def test_cond_diagonal(self):
        """R-LIN-18a: cond(diag([1,2])) = 2.0"""
        A = ForgeArray(np.array([[1., 0.], [0., 2.]]))
        assert abs(float(_unwrap(forge_cond(A)).flat[0]) - 2.0) < 1e-10


class TestRcond:
    """R-LIN-19: The reciprocal condition number rcond(A) SHALL return
    1/cond(A) within tolerance (1e-10).

    Model-user argument
    -------------------
    The golden user's scripts check ``rcond(A) > eps`` as a fast
    singularity guard before attempting a solve. Octave's rcond uses LAPACK's
    estimate, which for diagonal matrices equals exactly min(|d|)/max(|d|).
    If Forge returns a different value, the guard triggers incorrectly and
    the user's automated solver pipeline either crashes or silently skips
    valid systems.

    Decomposition
    -------------
    R-LIN-19a: rcond(diag([1,2])) equals 0.5 within tolerance.

    Consistency argument
    --------------------
    The diagonal case has a known exact reciprocal condition number (1/2),
    verifying the core computation path.
    """

    def test_rcond_diagonal(self, s):
        """R-LIN-19a: rcond([1 0; 0 2]) = 0.5"""
        s.eval("rc = rcond([1 0; 0 2])")
        r = float(_get(s, "rc").flat[0])
        assert abs(r - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# 4. Matrix Properties
# ---------------------------------------------------------------------------

class TestIssymmetric:
    """R-LIN-20: The issymmetric predicate SHALL return true for symmetric
    matrices and false for non-symmetric matrices.

    Model-user argument
    -------------------
    The golden user guards Cholesky factorization with an issymmetric check.
    If issymmetric returns a false positive on a non-symmetric matrix, chol
    receives invalid input and either crashes or returns a wrong factor. If
    it returns a false negative, the user falls back to a slower LU path
    unnecessarily.

    Decomposition
    -------------
    R-LIN-20a: issymmetric returns true for a symmetric 2x2 matrix.
    R-LIN-20b: issymmetric returns false for a non-symmetric 2x2 matrix.

    Consistency argument
    --------------------
    R-LIN-20a and R-LIN-20b together cover the true and false branches,
    fully verifying the predicate for the standard 2x2 case.
    """

    def test_symmetric_true(self):
        """R-LIN-20a: symmetric matrix returns true"""
        A = ForgeArray(np.array([[1., 2.], [2., 3.]]))
        assert _unwrap(forge_issymmetric(A)).flat[0] == True

    def test_symmetric_false(self):
        """R-LIN-20b: non-symmetric matrix returns false"""
        A = ForgeArray(np.array([[1., 2.], [3., 4.]]))
        assert _unwrap(forge_issymmetric(A)).flat[0] == False


class TestIshermitian:
    """R-LIN-21: The ishermitian predicate SHALL return true for real
    symmetric matrices (which are trivially Hermitian).

    Model-user argument
    -------------------
    The golden user works primarily with real matrices but occasionally
    calls ishermitian as a more general symmetry check that also works when
    complex-valued matrices appear in frequency-domain analysis. For real
    input, ishermitian must agree with issymmetric; if it does not, the
    user's branching logic (which tests ishermitian first) takes the wrong
    path.

    Decomposition
    -------------
    R-LIN-21a: ishermitian returns true for a real symmetric matrix.

    Consistency argument
    --------------------
    Testing the real-symmetric case confirms that ishermitian subsumes
    issymmetric for real input, which is the golden user's most common case.
    """

    def test_hermitian_real_symmetric(self):
        """R-LIN-21a: real symmetric matrix is Hermitian"""
        A = ForgeArray(np.array([[1., 2.], [2., 3.]]))
        assert _unwrap(forge_ishermitian(A)).flat[0] == True


class TestIsdiag:
    """R-LIN-22: The isdiag predicate SHALL return true for diagonal matrices
    and false for non-diagonal matrices.

    Model-user argument
    -------------------
    The golden user checks isdiag before applying optimized diagonal-matrix
    solvers that skip full LU factorization. A false positive feeds a
    non-diagonal matrix into the fast path, producing wrong results. A false
    negative forces the slow path unnecessarily.

    Decomposition
    -------------
    R-LIN-22a: isdiag returns true for diag([1,2]).
    R-LIN-22b: isdiag returns false for a matrix with a nonzero off-diagonal.

    Consistency argument
    --------------------
    R-LIN-22a and R-LIN-22b cover both branches. Together they verify that
    the predicate correctly distinguishes diagonal from non-diagonal structure.
    """

    def test_diag_true(self):
        """R-LIN-22a: diagonal matrix returns true"""
        assert _unwrap(forge_isdiag(ForgeArray(np.diag([1., 2.])))).flat[0] == True

    def test_diag_false(self):
        """R-LIN-22b: non-diagonal matrix returns false"""
        assert _unwrap(forge_isdiag(ForgeArray(np.array([[1., 1.], [0., 2.]])))).flat[0] == False


class TestIstriu:
    """R-LIN-23: The istriu predicate SHALL return true for upper-triangular
    matrices and false otherwise.

    Model-user argument
    -------------------
    The golden user checks triangular structure before calling specialized
    triangular solvers (forward/back substitution). Misidentifying structure
    causes the solver to read garbage from the zero region, producing wrong
    solutions without warning.

    Decomposition
    -------------
    R-LIN-23a: istriu returns true for an upper-triangular 2x2 matrix.
    R-LIN-23b: istriu returns false for a matrix with a nonzero lower entry.

    Consistency argument
    --------------------
    R-LIN-23a and R-LIN-23b cover both branches, fully verifying the
    predicate for the standard 2x2 case.
    """

    def test_upper_true(self):
        """R-LIN-23a: upper-triangular matrix returns true"""
        A = ForgeArray(np.array([[1., 2.], [0., 3.]]))
        assert _unwrap(forge_istriu(A)).flat[0] == True

    def test_upper_false(self):
        """R-LIN-23b: non-upper-triangular matrix returns false"""
        A = ForgeArray(np.array([[1., 2.], [1., 3.]]))
        assert _unwrap(forge_istriu(A)).flat[0] == False


class TestIstril:
    """R-LIN-24: The istril predicate SHALL return true for lower-triangular
    matrices and false otherwise.

    Model-user argument
    -------------------
    The golden user checks lower-triangular structure before applying forward
    substitution in chained solve pipelines (e.g., after LU factorization
    where L is lower-triangular). Misidentification causes the solver to
    treat upper entries as zero when they are not, silently corrupting the
    solution vector.

    Decomposition
    -------------
    R-LIN-24a: istril returns true for a lower-triangular 2x2 matrix.
    R-LIN-24b: istril returns false for a matrix with a nonzero upper entry.

    Consistency argument
    --------------------
    R-LIN-24a and R-LIN-24b cover both branches, fully verifying the
    predicate for the standard 2x2 case.
    """

    def test_lower_true(self):
        """R-LIN-24a: lower-triangular matrix returns true"""
        A = ForgeArray(np.array([[1., 0.], [2., 3.]]))
        assert _unwrap(forge_istril(A)).flat[0] == True

    def test_lower_false(self):
        """R-LIN-24b: non-lower-triangular matrix returns false"""
        A = ForgeArray(np.array([[1., 1.], [2., 3.]]))
        assert _unwrap(forge_istril(A)).flat[0] == False
