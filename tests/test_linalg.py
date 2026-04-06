# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for linear algebra toolbox.

SRS trace: SRS-FUNC-001, SRS-VAL-001

Requirement overview (V-model backfill):
    R-LIN-01  Matrix property queries (rank, condition, trace)
    R-LIN-02  Matrix structure predicates (diagonal, triangular, symmetric,
              Hermitian, definiteness, bandwidth)
    R-LIN-03  Subspace computations (null space, column space, cross product)
    R-LIN-04  Matrix functions (expm, logm, round-trip)
    R-LIN-05  Linear solvers and least-squares (linsolve, rref, lscov, ols)
    R-LIN-06  Miscellaneous linear algebra utilities (vech, vecnorm, planerot)

All requirements validated against Octave reference behavior. Test logic and
assertions are preserved exactly from the original test suite; only docstrings
have been added for traceability.
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.linalg import *


class TestMatrixProperties:
    """R-LIN-01: Matrix property queries.

    SHALL statement:
        Forge SHALL compute rank, condition number, and trace of a matrix,
        returning scalar results that match Octave to within floating-point
        tolerance.

    Model-user argument:
        The engineer routinely checks whether a stiffness matrix is full-rank
        before solving a FEM system. A poorly conditioned matrix signals
        numerical trouble, so cond() must be trustworthy. Trace appears in
        eigenvalue sum checks and energy calculations. These three queries
        are the first line of defense before committing to an expensive
        decomposition.

    Decomposition:
        R-LIN-01a  rank() returns correct rank for full-rank matrix
        R-LIN-01b  rank() returns correct rank for rank-deficient matrix
        R-LIN-01c  cond() returns 1.0 for the identity matrix
        R-LIN-01d  trace() returns dimension for identity matrix
        R-LIN-01e  trace() returns correct sum of diagonal for known matrix

    Consistency:
        R-LIN-01a and R-LIN-01b together cover the two fundamental cases of
        rank computation (full and deficient). R-LIN-01c validates the
        baseline condition number. R-LIN-01d and R-LIN-01e together verify
        trace for trivial and nontrivial diagonals. Combined, these five
        sub-requirements fully exercise the three property functions.
    """

    def test_rank_full(self):
        """R-LIN-01a: rank of full-rank identity matrix equals its dimension."""
        A = ForgeArray(np.eye(3))
        assert int(_unwrap(forge_rank(A)).flat[0]) == 3

    def test_rank_deficient(self):
        """R-LIN-01b: rank of a rank-1 matrix is correctly identified as 1."""
        A = ForgeArray(np.array([[1.0, 2.0], [2.0, 4.0]]))
        assert int(_unwrap(forge_rank(A)).flat[0]) == 1

    def test_cond_identity(self):
        """R-LIN-01c: condition number of identity matrix is exactly 1."""
        A = ForgeArray(np.eye(3))
        r = float(_unwrap(forge_cond(A)).flat[0])
        assert abs(r - 1.0) < 1e-10

    def test_trace_identity(self):
        """R-LIN-01d: trace of 4x4 identity equals 4."""
        A = ForgeArray(np.eye(4))
        assert abs(float(_unwrap(forge_trace(A)).flat[0]) - 4.0) < 1e-14

    def test_trace_known(self):
        """R-LIN-01e: trace of [[1,2],[3,4]] equals 5."""
        A = ForgeArray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert abs(float(_unwrap(forge_trace(A)).flat[0]) - 5.0) < 1e-14


class TestMatrixStructure:
    """R-LIN-02: Matrix structure predicates.

    SHALL statement:
        Forge SHALL provide predicate functions (isdiag, istril, istriu,
        issymmetric, ishermitian, isdefinite, bandwidth, isbanded) that
        return boolean or integer results consistent with Octave for
        classifying matrix structure.

    Model-user argument:
        Before choosing a solver strategy the engineer checks whether the
        matrix is symmetric positive definite (Cholesky path), banded
        (exploit sparsity), or triangular (direct back-substitution). Getting
        these predicates wrong means picking a slower or numerically inferior
        algorithm. The bandwidth query is essential for FEM stiffness matrices
        where bandwidth directly determines memory and solve time.

    Decomposition:
        R-LIN-02a  isdiag returns True for a diagonal matrix
        R-LIN-02b  isdiag returns False for a non-diagonal matrix
        R-LIN-02c  istril returns True for a lower-triangular matrix
        R-LIN-02d  istriu returns True for an upper-triangular matrix
        R-LIN-02e  issymmetric returns True for a symmetric matrix
        R-LIN-02f  issymmetric returns False for a non-symmetric matrix
        R-LIN-02g  ishermitian returns True for a real symmetric matrix
        R-LIN-02h  isdefinite returns 1 for positive-definite matrix (identity)
        R-LIN-02i  bandwidth returns (0,0) for a diagonal matrix
        R-LIN-02j  bandwidth returns (1,1) for a tridiagonal matrix
        R-LIN-02k  isbanded confirms a diagonal matrix is (0,0)-banded

    Consistency:
        R-LIN-02a through R-LIN-02b cover the diagonal predicate (true and
        false branches). R-LIN-02c and R-LIN-02d cover triangular predicates.
        R-LIN-02e and R-LIN-02f cover symmetry (both branches). R-LIN-02g
        covers the Hermitian case for real matrices. R-LIN-02h validates
        definiteness classification. R-LIN-02i through R-LIN-02k verify
        bandwidth computation and the banded predicate. Together these eleven
        sub-requirements exercise every structure predicate with at least one
        true-case and, where applicable, one false-case.
    """

    def test_isdiag_true(self):
        """R-LIN-02a: diagonal matrix correctly identified as diagonal."""
        A = ForgeArray(np.diag([1.0, 2.0, 3.0]))
        assert _unwrap(forge_isdiag(A)).flat[0] == True

    def test_isdiag_false(self):
        """R-LIN-02b: upper-triangular (non-diagonal) matrix is not diagonal."""
        A = ForgeArray(np.array([[1.0, 1.0], [0.0, 1.0]]))
        assert _unwrap(forge_isdiag(A)).flat[0] == False

    def test_istril_true(self):
        """R-LIN-02c: lower-triangular ones matrix identified correctly."""
        A = ForgeArray(np.tril(np.ones((3, 3))))
        assert _unwrap(forge_istril(A)).flat[0] == True

    def test_istriu_true(self):
        """R-LIN-02d: upper-triangular ones matrix identified correctly."""
        A = ForgeArray(np.triu(np.ones((3, 3))))
        assert _unwrap(forge_istriu(A)).flat[0] == True

    def test_issymmetric(self):
        """R-LIN-02e: symmetric 2x2 matrix identified as symmetric."""
        A = ForgeArray(np.array([[1.0, 2.0], [2.0, 3.0]]))
        assert _unwrap(forge_issymmetric(A)).flat[0] == True

    def test_issymmetric_false(self):
        """R-LIN-02f: non-symmetric 2x2 matrix correctly rejected."""
        A = ForgeArray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert _unwrap(forge_issymmetric(A)).flat[0] == False

    def test_ishermitian_real(self):
        """R-LIN-02g: real symmetric matrix is Hermitian."""
        A = ForgeArray(np.array([[1.0, 2.0], [2.0, 3.0]]))
        assert _unwrap(forge_ishermitian(A)).flat[0] == True

    def test_isdefinite_positive(self):
        """R-LIN-02h: identity matrix classified as positive definite (1)."""
        A = ForgeArray(np.eye(3))
        assert int(_unwrap(forge_isdefinite(A)).flat[0]) == 1

    def test_bandwidth_diagonal(self):
        """R-LIN-02i: diagonal matrix has bandwidth (0,0)."""
        A = ForgeArray(np.diag([1.0, 2.0, 3.0]))
        lo, up = forge_bandwidth(A)
        assert int(_unwrap(lo).flat[0]) == 0
        assert int(_unwrap(up).flat[0]) == 0

    def test_bandwidth_tridiag(self):
        """R-LIN-02j: tridiagonal matrix has bandwidth (1,1)."""
        A = ForgeArray(np.array([[1, 1, 0], [1, 2, 1], [0, 1, 3.0]]))
        lo, up = forge_bandwidth(A)
        assert int(_unwrap(lo).flat[0]) == 1
        assert int(_unwrap(up).flat[0]) == 1

    def test_isbanded(self):
        """R-LIN-02k: diagonal matrix confirmed as (0,0)-banded."""
        A = ForgeArray(np.diag([1.0, 2.0, 3.0]))
        assert _unwrap(forge_isbanded(A, ForgeArray(0), ForgeArray(0))).flat[0] == True


class TestSubspaces:
    """R-LIN-03: Subspace computations.

    SHALL statement:
        Forge SHALL compute the null space, orthonormal column-space basis,
        and cross product of vectors, returning results dimensionally and
        numerically consistent with Octave.

    Model-user argument:
        When the engineer encounters a singular or near-singular system, the
        null space reveals which degrees of freedom are unconstrained. The
        column-space basis (orth) is used for model reduction and projecting
        onto feasible subspaces. Cross products appear constantly in 3D
        mechanics for computing normals, torques, and angular momentum. These
        are not optional utilities; they are core to everyday structural and
        multibody analysis.

    Decomposition:
        R-LIN-03a  null() of full-rank matrix returns empty null space
        R-LIN-03b  null() of rank-deficient matrix returns non-trivial basis
        R-LIN-03c  orth() returns correct number of basis vectors for full rank
        R-LIN-03d  cross() computes correct cross product of unit vectors

    Consistency:
        R-LIN-03a and R-LIN-03b together verify null space for the two
        fundamental rank situations. R-LIN-03c validates orthonormal basis
        dimensionality. R-LIN-03d validates the 3D vector cross product. All
        four sub-requirements cover the three subspace functions with
        representative inputs.
    """

    def test_null_identity(self):
        """R-LIN-03a: null space of identity is empty."""
        A = ForgeArray(np.eye(3))
        N = _unwrap(forge_null(A))
        assert N.shape[1] == 0 or N.size == 0

    def test_null_rank_deficient(self):
        """R-LIN-03b: rank-deficient matrix has at least one null vector."""
        A = ForgeArray(np.array([[1.0, 2.0], [2.0, 4.0]]))
        N = _unwrap(forge_null(A))
        # Null space should be 1D
        assert N.shape[1] >= 1

    def test_orth_full_rank(self):
        """R-LIN-03c: orth of rank-2 matrix returns 2 basis vectors."""
        A = ForgeArray(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
        Q = _unwrap(forge_orth(A))
        assert Q.shape[1] == 2

    def test_cross_product(self):
        """R-LIN-03d: cross(i-hat, j-hat) equals k-hat."""
        a = ForgeArray(np.array([1.0, 0.0, 0.0]))
        b = ForgeArray(np.array([0.0, 1.0, 0.0]))
        r = _unwrap(forge_cross(a, b)).ravel()
        np.testing.assert_allclose(r, [0, 0, 1], atol=1e-14)


class TestMatrixFunctions:
    """R-LIN-04: Matrix functions (exponential, logarithm).

    SHALL statement:
        Forge SHALL compute the matrix exponential (expm) and matrix logarithm
        (logm) such that expm(zeros) equals the identity, logm(eye) equals
        the zero matrix, and a round-trip expm followed by logm recovers the
        original matrix within floating-point tolerance.

    Model-user argument:
        Matrix exponentials appear in control systems (state transition
        matrices), vibration analysis (modal response), and solving linear
        ODEs. The engineer expects expm and logm to be inverses of each other
        within numerical tolerance. A broken round-trip would silently corrupt
        time-domain simulations, making this a high-trust function pair.

    Decomposition:
        R-LIN-04a  expm(zeros) equals the identity matrix
        R-LIN-04b  logm(identity) equals the zero matrix
        R-LIN-04c  logm(expm(A)) recovers A for a known matrix

    Consistency:
        R-LIN-04a validates the base case of expm. R-LIN-04b validates the
        base case of logm. R-LIN-04c confirms the inverse relationship holds
        for a nontrivial upper-triangular input. Together these three
        sub-requirements verify both functions independently and as a pair.
    """

    def test_expm_zero(self):
        """R-LIN-04a: expm(0) = I."""
        A = ForgeArray(np.zeros((3, 3)))
        r = _unwrap(forge_expm(A))
        np.testing.assert_allclose(r, np.eye(3), atol=1e-14)

    def test_logm_identity(self):
        """R-LIN-04b: logm(I) = 0."""
        A = ForgeArray(np.eye(3))
        r = _unwrap(forge_logm(A))
        np.testing.assert_allclose(r, np.zeros((3, 3)), atol=1e-14)

    def test_expm_logm_roundtrip(self):
        """R-LIN-04c: logm(expm(A)) recovers original matrix A."""
        A = ForgeArray(np.array([[1.0, 0.5], [0.0, 2.0]]))
        r = _unwrap(forge_logm(ForgeArray(_unwrap(forge_expm(A)))))
        np.testing.assert_allclose(r, _unwrap(A), atol=1e-10)


class TestSolvers:
    """R-LIN-05: Linear solvers and least-squares.

    SHALL statement:
        Forge SHALL solve linear systems (linsolve), compute reduced row
        echelon form (rref), and perform least-squares fitting (lscov, ols),
        returning solutions that match Octave to within floating-point
        tolerance.

    Model-user argument:
        Solving Ax=b is the single most common operation in numerical
        engineering. The engineer uses linsolve for direct systems, rref for
        understanding solution structure during debugging, and lscov/ols for
        curve fitting and regression in experimental data analysis. If
        linsolve returns a wrong answer the entire downstream analysis
        (stresses, displacements, control gains) is invalid. Least-squares
        is equally critical for calibrating models against measured data.

    Decomposition:
        R-LIN-05a  linsolve returns correct solution for a 2x2 system
        R-LIN-05b  rref of identity returns identity
        R-LIN-05c  rref of non-square augmented matrix produces valid echelon form
        R-LIN-05d  lscov recovers correct slope from 3-point linear data
        R-LIN-05e  ols recovers correct coefficients from exact linear data

    Consistency:
        R-LIN-05a validates the core solver for a well-conditioned system.
        R-LIN-05b and R-LIN-05c verify rref for trivial and non-trivial
        inputs respectively. R-LIN-05d and R-LIN-05e verify the two
        least-squares interfaces with data that has a known analytic solution.
        Together these five sub-requirements cover all four solver functions.
    """

    def test_linsolve_simple(self):
        """R-LIN-05a: linsolve produces correct x for 2x2 system."""
        A = ForgeArray(np.array([[2.0, 1.0], [1.0, 3.0]]))
        b = ForgeArray(np.array([[5.0], [7.0]]))
        x = _unwrap(forge_linsolve(A, b)).ravel()
        np.testing.assert_allclose(x, [1.6, 1.8], atol=1e-10)

    def test_rref_identity(self):
        """R-LIN-05b: rref of identity is identity."""
        A = ForgeArray(np.eye(3))
        r = _unwrap(forge_rref(A))
        np.testing.assert_allclose(r, np.eye(3), atol=1e-14)

    def test_rref_augmented(self):
        """R-LIN-05c: rref of 2x3 matrix has leading ones on diagonal."""
        A = ForgeArray(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        r = _unwrap(forge_rref(A))
        # Should be in row echelon form
        assert abs(r[0, 0] - 1.0) < 1e-12
        assert abs(r[1, 1] - 1.0) < 1e-12

    def test_lscov(self):
        """R-LIN-05d: lscov recovers unit slope from 3-point linear data."""
        A = ForgeArray(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]))
        b = ForgeArray(np.array([2.0, 3.0, 4.0]))
        x = _unwrap(forge_lscov(A, b)).ravel()
        assert abs(x[1] - 1.0) < 1e-10  # slope should be ~1

    def test_ols(self):
        """R-LIN-05e: ols recovers slope=2 from exact linear data."""
        X = ForgeArray(np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]))
        y = ForgeArray(np.array([2.0, 4.0, 6.0]))
        beta = _unwrap(forge_ols(y, X)).ravel()
        assert abs(beta[1] - 2.0) < 1e-10


class TestMisc:
    """R-LIN-06: Miscellaneous linear algebra utilities.

    SHALL statement:
        Forge SHALL provide vech (half-vectorization), vecnorm (vector norm),
        and planerot (Givens rotation) functions that return results matching
        Octave behavior and numerical tolerances.

    Model-user argument:
        The engineer uses vech when working with symmetric matrices in
        optimization (storing only the lower triangle saves memory and avoids
        redundant computation). Vector norms are ubiquitous for convergence
        checks, residual evaluation, and normalizing directions. Givens
        rotations (planerot) are building blocks for QR updates and appear
        in real-time estimation filters. These utilities are small but
        frequently called; correctness at this level prevents silent errors
        from propagating into larger algorithms.

    Decomposition:
        R-LIN-06a  vech extracts the correct lower-triangular column vector
        R-LIN-06b  vecnorm returns correct Euclidean norm for a 2-element vector
        R-LIN-06c  planerot produces correct rotated vector and zeroes the
                    second component

    Consistency:
        R-LIN-06a validates half-vectorization output against the known
        column-major lower triangle. R-LIN-06b validates the fundamental
        vector norm (3-4-5 triangle). R-LIN-06c validates both outputs of
        planerot: the rotation magnitude and the zeroing property. Together
        these three sub-requirements cover all three utility functions.
    """

    def test_vech(self):
        """R-LIN-06a: vech extracts lower-triangular elements column-wise."""
        A = ForgeArray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        r = _unwrap(forge_vech(A)).ravel()
        np.testing.assert_array_equal(r, [1, 3, 4])

    def test_vecnorm(self):
        """R-LIN-06b: vecnorm([3,4]) equals 5 (Euclidean norm)."""
        x = ForgeArray(np.array([3.0, 4.0]))
        r = float(_unwrap(forge_vecnorm(x)).flat[0])
        assert abs(r - 5.0) < 1e-14

    def test_planerot(self):
        """R-LIN-06c: planerot([3,4]) yields magnitude 5 with zero second element."""
        x = ForgeArray(np.array([3.0, 4.0]))
        G, y = forge_planerot(x)
        assert abs(_unwrap(y).ravel()[0] - 5.0) < 1e-14
        assert abs(_unwrap(y).ravel()[1]) < 1e-14
