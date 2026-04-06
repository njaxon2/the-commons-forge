# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for critical engine fixes.

Requirement R-FIX: The engine SHALL not regress on previously fixed bugs.
Each test in this file corresponds to a specific defect that was
discovered, diagnosed, and corrected. The test prevents that defect from
recurring in future development.

Model-user argument: An engineer migrating from MATLAB expects that
matrix multiplication, division, indexing with 'end', colon ranges,
linear algebra decompositions, and session output all behave correctly.
A regression in any of these areas would silently corrupt the engineer's
calculations, producing wrong numerical results that might not be
caught until downstream analysis reveals inconsistencies.

Decomposition:
  R-FIX-01..05: Matrix multiplication (mat*vec, vec*mat, mat*mat, .*, scalar*)
  R-FIX-06..07: Matrix division (backslash solve, element-wise .\\)
  R-FIX-08..10: End keyword in indexing (linear, range, end-1)
  R-FIX-11..16: Colon range generation (integer, fractional, no-overshoot, descending, empty, pi-step)
  R-FIX-17..25: Core linear algebra (inv, det, norm, eig, svd, qr, fft/ifft, pinv, kron)
  R-FIX-26..28: Session output buffer (disp, multiple disp, no output leak)

Consistency argument: The six groups cover arithmetic operators (R-FIX-01..07),
indexing mechanics (R-FIX-08..10), range generation (R-FIX-11..16),
decomposition/transform functions (R-FIX-17..25), and output plumbing
(R-FIX-26..28). Each group addresses a distinct subsystem where bugs
were historically found, and together they guard the engine's most
critical numerical and display pathways.
"""
import pytest
import numpy as np
from forge.engine.evaluator import Session
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def session():
    return Session()


class TestMatrixMultiply:
    """R-FIX-01..05: Matrix multiplication SHALL dispatch correctly for
    all operand shape combinations (mat*vec, vec*mat, mat*mat, .*, scalar*).

    Model-user argument: The engineer computes A*v for linear transforms,
    v*A for row-vector projections, A*B for chained transforms, A.*B for
    element-wise scaling, and c*A for uniform scaling. Each dispatch path
    was previously a source of bugs due to shape inference ambiguity.

    Decomposition:
      R-FIX-01: Matrix times column vector
      R-FIX-02: Row vector times matrix
      R-FIX-03: Matrix times matrix
      R-FIX-04: Element-wise multiply (.*)
      R-FIX-05: Scalar times matrix

    Consistency: These five tests cover every shape combination for the
    * and .* operators: (2D,1D), (1D,2D), (2D,2D), element-wise, and
    scalar broadcast.
    """

    def test_matrix_times_vector(self, session):
        """R-FIX-01: Matrix times column vector produces correct result."""
        session.eval("A = [1 2; 3 4]")
        session.eval("v = [5; 6]")
        session.eval("r = A * v")
        r = _unwrap(session.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [17, 39])

    def test_vector_times_matrix(self, session):
        """R-FIX-02: Row vector times matrix produces correct result."""
        session.eval("v = [1 2]")
        session.eval("A = [3 4; 5 6]")
        session.eval("r = v * A")
        r = _unwrap(session.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [13, 16])

    def test_matrix_times_matrix(self, session):
        """R-FIX-03: Matrix times matrix produces correct result."""
        session.eval("A = [1 2; 3 4]")
        session.eval("B = [5 6; 7 8]")
        session.eval("C = A * B")
        r = _unwrap(session.workspace.get("C"))
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_allclose(r, expected)

    def test_element_wise_multiply(self, session):
        """R-FIX-04: Element-wise .* multiplies corresponding elements."""
        session.eval("A = [1 2; 3 4]")
        session.eval("B = [5 6; 7 8]")
        session.eval("C = A .* B")
        r = _unwrap(session.workspace.get("C"))
        expected = np.array([[5, 12], [21, 32]])
        np.testing.assert_allclose(r, expected)

    def test_scalar_times_matrix(self, session):
        """R-FIX-05: Scalar times matrix broadcasts correctly."""
        session.eval("A = [1 2; 3 4]")
        session.eval("r = 2 * A")
        r = _unwrap(session.workspace.get("r"))
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_allclose(r, expected)


class TestMatrixDivision:
    """R-FIX-06..07: Matrix division SHALL solve linear systems via
    backslash and perform element-wise division via .\\

    Model-user argument: The engineer uses A\\b to solve linear systems
    (e.g., least squares, calibration) and ./ or .\\ for element-wise
    normalization. Incorrect backslash dispatch was a critical bug that
    produced silently wrong solutions.

    Decomposition:
      R-FIX-06: A\\b solves system and A*x recovers b
      R-FIX-07: a.\\ b performs element-wise left division

    Consistency: R-FIX-06 covers matrix backslash (linear solve) and
    R-FIX-07 covers element-wise left division, the two division modes.
    """

    def test_backslash_solve(self, session):
        """R-FIX-06: A\\b solves the system; A*x recovers b."""
        session.eval("A = [2 1; 1 3]")
        session.eval("b = [5; 10]")
        session.eval("x = A \\ b")
        x = _unwrap(session.workspace.get("x"))
        session.eval("check = A * x")
        check = _unwrap(session.workspace.get("check"))
        np.testing.assert_allclose(check.ravel(), [5, 10], atol=1e-10)

    def test_element_wise_backslash(self, session):
        """R-FIX-07: a.\\ b performs element-wise left division."""
        session.eval("a = [2, 4]")
        session.eval("b = [10, 20]")
        session.eval("r = a .\\ b")
        r = _unwrap(session.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [5, 5])


class TestEndKeyword:
    """R-FIX-08..10: The 'end' keyword in subscript expressions SHALL
    resolve to the last valid index in the relevant dimension.

    Model-user argument: The engineer uses x(end), x(3:end), and x(end-1)
    constantly to access trailing elements without knowing array length.
    A bug in end-resolution caused off-by-one errors or crashes when
    the array size changed dynamically.

    Decomposition:
      R-FIX-08: x(end) returns the last element
      R-FIX-09: x(3:end) returns elements from index 3 to the end
      R-FIX-10: x(end-1) returns the second-to-last element

    Consistency: These three patterns (end alone, end in range, end with
    arithmetic) cover all end-keyword usage forms in subscript expressions.
    """

    def test_end_linear(self, session):
        """R-FIX-08: x(end) returns the last element."""
        session.eval("x = [10 20 30 40 50]")
        session.eval("y = x(end)")
        y = _unwrap(session.workspace.get("y"))
        assert y.item() == 50.0

    def test_end_in_range(self, session):
        """R-FIX-09: x(3:end) returns elements from index 3 to the end."""
        session.eval("x = [10 20 30 40 50]")
        session.eval("y = x(3:end)")
        y = _unwrap(session.workspace.get("y"))
        np.testing.assert_allclose(y.ravel(), [30, 40, 50])

    def test_end_minus(self, session):
        """R-FIX-10: x(end-1) returns the second-to-last element."""
        session.eval("x = [10 20 30 40 50]")
        session.eval("y = x(end-1)")
        y = _unwrap(session.workspace.get("y"))
        assert y.item() == 40.0


class TestColonFunction:
    """R-FIX-11..16: The colon operator (start:step:stop) SHALL generate
    ranges that match Octave behavior for integer steps, fractional steps,
    descending ranges, empty ranges, and irrational steps.

    Model-user argument: The engineer generates sample grids (0:0.01:10),
    countdown sequences (10:-1:1), and angular sweeps (0:pi:2*pi). A bug
    in colon caused overshooting the endpoint or generating wrong step
    counts, which corrupted loop bounds and plot axes.

    Decomposition:
      R-FIX-11: Integer colon 1:5 generates [1 2 3 4 5]
      R-FIX-12: Fractional step 0:0.5:2 generates [0 0.5 1 1.5 2]
      R-FIX-13: No overshoot on 0.5:0.5:2.5 (exactly 5 elements)
      R-FIX-14: Descending 5:-1:1 generates [5 4 3 2 1]
      R-FIX-15: Empty range 5:1 (no step, ascending default) is empty
      R-FIX-16: Irrational step 0:pi:10 does not exceed 10

    Consistency: These six tests partition the colon operator behavior:
    unit step (R-FIX-11), sub-unit step (R-FIX-12..13), negative step
    (R-FIX-14), degenerate/empty (R-FIX-15), and irrational step (R-FIX-16).
    """

    def test_integer_colon(self, session):
        """R-FIX-11: Integer colon 1:5 generates [1 2 3 4 5]."""
        session.eval("x = 1:5")
        x = _unwrap(session.workspace.get("x"))
        np.testing.assert_allclose(x.ravel(), [1, 2, 3, 4, 5])

    def test_fractional_step(self, session):
        """R-FIX-12: Fractional step 0:0.5:2 generates correct sequence."""
        session.eval("x = 0:0.5:2")
        x = _unwrap(session.workspace.get("x"))
        np.testing.assert_allclose(x.ravel(), [0, 0.5, 1, 1.5, 2])

    def test_no_overshoot(self, session):
        """R-FIX-13: 0.5:0.5:2.5 produces exactly 5 elements without overshoot."""
        session.eval("x = 0.5:0.5:2.5")
        x = _unwrap(session.workspace.get("x"))
        np.testing.assert_allclose(x.ravel(), [0.5, 1.0, 1.5, 2.0, 2.5])
        assert len(x.ravel()) == 5

    def test_descending(self, session):
        """R-FIX-14: Descending 5:-1:1 generates [5 4 3 2 1]."""
        session.eval("x = 5:-1:1")
        x = _unwrap(session.workspace.get("x"))
        np.testing.assert_allclose(x.ravel(), [5, 4, 3, 2, 1])

    def test_empty_colon(self, session):
        """R-FIX-15: Empty range 5:1 produces an empty array."""
        session.eval("x = 5:1")
        x = _unwrap(session.workspace.get("x"))
        assert x.size == 0

    def test_pi_step(self, session):
        """R-FIX-16: Irrational step 0:pi:10 does not exceed endpoint."""
        session.eval("x = 0:pi:10")
        x = _unwrap(session.workspace.get("x"))
        assert x.ravel()[-1] <= 10.0 + 1e-10
        assert len(x.ravel()) == 4


class TestCoreLAFunctions:
    """R-FIX-17..25: Core linear algebra and transform builtins SHALL
    produce numerically correct results matching Octave.

    Model-user argument: The engineer relies on inv, det, norm, eig, svd,
    qr, fft/ifft, pinv, and kron as building blocks for control system
    design, signal processing, and structural analysis. A bug in any of
    these functions corrupts all downstream computations that depend on
    them.

    Decomposition:
      R-FIX-17: inv(A)*A yields identity
      R-FIX-18: det([1 2;3 4]) returns -2
      R-FIX-19: norm([3,4]) returns 5
      R-FIX-20: eig of diagonal matrix returns diagonal entries
      R-FIX-21: svd of identity returns ones on diagonal
      R-FIX-22: Q*R from qr(A) reconstructs A
      R-FIX-23: ifft(fft(x)) recovers x
      R-FIX-24: pinv of identity returns identity
      R-FIX-25: kron(I2, I2) produces I4

    Consistency: These nine tests cover matrix inversion (R-FIX-17),
    scalar properties (R-FIX-18..19), eigendecomposition (R-FIX-20),
    singular value decomposition (R-FIX-21), QR factorization (R-FIX-22),
    spectral transforms (R-FIX-23), pseudoinverse (R-FIX-24), and
    Kronecker product (R-FIX-25), spanning the full LA builtin surface.
    """

    def test_inv(self, session):
        """R-FIX-17: inv(A)*A yields identity matrix."""
        session.eval("A = [1 2; 3 4]")
        session.eval("B = inv(A)")
        session.eval("I = A * B")
        I = _unwrap(session.workspace.get("I"))
        np.testing.assert_allclose(I, np.eye(2), atol=1e-10)

    def test_det(self, session):
        """R-FIX-18: det([1 2;3 4]) returns -2."""
        session.eval("d = det([1 2; 3 4])")
        d = _unwrap(session.workspace.get("d"))
        np.testing.assert_allclose(d.item(), -2.0, atol=1e-10)

    def test_norm(self, session):
        """R-FIX-19: norm([3,4]) returns 5 (Euclidean norm)."""
        session.eval("n = norm([3, 4])")
        n = _unwrap(session.workspace.get("n"))
        np.testing.assert_allclose(n.item(), 5.0, atol=1e-10)

    def test_eig(self, session):
        """R-FIX-20: eig of diagonal matrix returns diagonal entries."""
        session.eval("[V, D] = eig([2 0; 0 3])")
        D = _unwrap(session.workspace.get("D"))
        evals = sorted(np.diag(D).real)
        np.testing.assert_allclose(evals, [2.0, 3.0], atol=1e-10)

    def test_svd(self, session):
        """R-FIX-21: svd of identity returns ones on the diagonal."""
        session.eval("[U, S, V] = svd(eye(3))")
        S = _unwrap(session.workspace.get("S"))
        np.testing.assert_allclose(np.diag(S), [1, 1, 1], atol=1e-10)

    def test_qr(self, session):
        """R-FIX-22: Q*R from qr(A) reconstructs A."""
        session.eval("A = [1 2; 3 4]")
        session.eval("[Q, R] = qr(A)")
        Q = _unwrap(session.workspace.get("Q"))
        R = _unwrap(session.workspace.get("R"))
        np.testing.assert_allclose(Q @ R, np.array([[1, 2], [3, 4]]), atol=1e-10)

    def test_fft_ifft_roundtrip(self, session):
        """R-FIX-23: ifft(fft(x)) recovers x within machine precision."""
        session.eval("x = [1 2 3 4]")
        session.eval("y = ifft(fft(x))")
        y = _unwrap(session.workspace.get("y"))
        np.testing.assert_allclose(y.real.ravel(), [1, 2, 3, 4], atol=1e-10)

    def test_pinv(self, session):
        """R-FIX-24: pinv of identity returns identity."""
        session.eval("P = pinv(eye(3))")
        P = _unwrap(session.workspace.get("P"))
        np.testing.assert_allclose(P, np.eye(3), atol=1e-10)

    def test_kron(self, session):
        """R-FIX-25: kron(I2, I2) produces I4."""
        session.eval("K = kron(eye(2), eye(2))")
        K = _unwrap(session.workspace.get("K"))
        np.testing.assert_allclose(K, np.eye(4), atol=1e-10)


class TestSessionOutputBuffer:
    """R-FIX-26..28: The session output buffer SHALL correctly capture
    disp() output and not leak output between evaluations.

    Model-user argument: The engineer uses disp() to inspect intermediate
    results during debugging. If disp output leaks into subsequent
    evaluations or disappears, the engineer cannot trust the command
    window for interactive exploration.

    Decomposition:
      R-FIX-26: disp(42) produces output containing "42"
      R-FIX-27: Consecutive disp calls each produce their own output
      R-FIX-28: Output from one eval does not leak into the next

    Consistency: R-FIX-26 tests basic capture, R-FIX-27 tests repeated
    capture, and R-FIX-28 tests isolation between evaluations.
    """

    def test_disp_output(self):
        """R-FIX-26: disp(42) produces output containing '42'."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval("disp(42)")
        assert "42" in result

    def test_multiple_disp(self):
        """R-FIX-27: Consecutive disp calls each produce correct output."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval("x = 1")
        r1 = s.eval("disp(x)")
        r2 = s.eval("disp(x)")
        assert "1" in r1
        assert "1" in r2

    def test_output_doesnt_leak(self):
        """R-FIX-28: Output from one eval does not leak into the next."""
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval("disp(42)")
        result = s.eval("x = 1")
        assert "42" not in result
