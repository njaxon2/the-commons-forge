"""Regression tests for critical engine fixes."""
import pytest
import numpy as np
from forge.engine.evaluator import Session
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def session():
    return Session()


class TestMatrixMultiply:
    def test_matrix_times_vector(self, session):
        session.eval("A = [1 2; 3 4]")
        session.eval("v = [5; 6]")
        session.eval("r = A * v")
        r = _unwrap(session.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [17, 39])

    def test_vector_times_matrix(self, session):
        session.eval("v = [1 2]")
        session.eval("A = [3 4; 5 6]")
        session.eval("r = v * A")
        r = _unwrap(session.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [13, 16])

    def test_matrix_times_matrix(self, session):
        session.eval("A = [1 2; 3 4]")
        session.eval("B = [5 6; 7 8]")
        session.eval("C = A * B")
        r = _unwrap(session.workspace.get("C"))
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_allclose(r, expected)

    def test_element_wise_multiply(self, session):
        session.eval("A = [1 2; 3 4]")
        session.eval("B = [5 6; 7 8]")
        session.eval("C = A .* B")
        r = _unwrap(session.workspace.get("C"))
        expected = np.array([[5, 12], [21, 32]])
        np.testing.assert_allclose(r, expected)

    def test_scalar_times_matrix(self, session):
        session.eval("A = [1 2; 3 4]")
        session.eval("r = 2 * A")
        r = _unwrap(session.workspace.get("r"))
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_allclose(r, expected)


class TestMatrixDivision:
    def test_backslash_solve(self, session):
        session.eval("A = [2 1; 1 3]")
        session.eval("b = [5; 10]")
        session.eval("x = A \\ b")
        x = _unwrap(session.workspace.get("x"))
        session.eval("check = A * x")
        check = _unwrap(session.workspace.get("check"))
        np.testing.assert_allclose(check.ravel(), [5, 10], atol=1e-10)

    def test_element_wise_backslash(self, session):
        session.eval("a = [2, 4]")
        session.eval("b = [10, 20]")
        session.eval("r = a .\\ b")
        r = _unwrap(session.workspace.get("r"))
        np.testing.assert_allclose(r.ravel(), [5, 5])


class TestEndKeyword:
    def test_end_linear(self, session):
        session.eval("x = [10 20 30 40 50]")
        session.eval("y = x(end)")
        y = _unwrap(session.workspace.get("y"))
        assert y.item() == 50.0

    def test_end_in_range(self, session):
        session.eval("x = [10 20 30 40 50]")
        session.eval("y = x(3:end)")
        y = _unwrap(session.workspace.get("y"))
        np.testing.assert_allclose(y.ravel(), [30, 40, 50])

    def test_end_minus(self, session):
        session.eval("x = [10 20 30 40 50]")
        session.eval("y = x(end-1)")
        y = _unwrap(session.workspace.get("y"))
        assert y.item() == 40.0


class TestColonFunction:
    def test_integer_colon(self, session):
        session.eval("x = 1:5")
        x = _unwrap(session.workspace.get("x"))
        np.testing.assert_allclose(x.ravel(), [1, 2, 3, 4, 5])

    def test_fractional_step(self, session):
        session.eval("x = 0:0.5:2")
        x = _unwrap(session.workspace.get("x"))
        np.testing.assert_allclose(x.ravel(), [0, 0.5, 1, 1.5, 2])

    def test_no_overshoot(self, session):
        session.eval("x = 0.5:0.5:2.5")
        x = _unwrap(session.workspace.get("x"))
        np.testing.assert_allclose(x.ravel(), [0.5, 1.0, 1.5, 2.0, 2.5])
        assert len(x.ravel()) == 5

    def test_descending(self, session):
        session.eval("x = 5:-1:1")
        x = _unwrap(session.workspace.get("x"))
        np.testing.assert_allclose(x.ravel(), [5, 4, 3, 2, 1])

    def test_empty_colon(self, session):
        session.eval("x = 5:1")
        x = _unwrap(session.workspace.get("x"))
        assert x.size == 0

    def test_pi_step(self, session):
        session.eval("x = 0:pi:10")
        x = _unwrap(session.workspace.get("x"))
        assert x.ravel()[-1] <= 10.0 + 1e-10
        assert len(x.ravel()) == 4


class TestCoreLAFunctions:
    def test_inv(self, session):
        session.eval("A = [1 2; 3 4]")
        session.eval("B = inv(A)")
        session.eval("I = A * B")
        I = _unwrap(session.workspace.get("I"))
        np.testing.assert_allclose(I, np.eye(2), atol=1e-10)

    def test_det(self, session):
        session.eval("d = det([1 2; 3 4])")
        d = _unwrap(session.workspace.get("d"))
        np.testing.assert_allclose(d.item(), -2.0, atol=1e-10)

    def test_norm(self, session):
        session.eval("n = norm([3, 4])")
        n = _unwrap(session.workspace.get("n"))
        np.testing.assert_allclose(n.item(), 5.0, atol=1e-10)

    def test_eig(self, session):
        session.eval("[V, D] = eig([2 0; 0 3])")
        D = _unwrap(session.workspace.get("D"))
        evals = sorted(np.diag(D).real)
        np.testing.assert_allclose(evals, [2.0, 3.0], atol=1e-10)

    def test_svd(self, session):
        session.eval("[U, S, V] = svd(eye(3))")
        S = _unwrap(session.workspace.get("S"))
        np.testing.assert_allclose(np.diag(S), [1, 1, 1], atol=1e-10)

    def test_qr(self, session):
        session.eval("A = [1 2; 3 4]")
        session.eval("[Q, R] = qr(A)")
        Q = _unwrap(session.workspace.get("Q"))
        R = _unwrap(session.workspace.get("R"))
        np.testing.assert_allclose(Q @ R, np.array([[1, 2], [3, 4]]), atol=1e-10)

    def test_fft_ifft_roundtrip(self, session):
        session.eval("x = [1 2 3 4]")
        session.eval("y = ifft(fft(x))")
        y = _unwrap(session.workspace.get("y"))
        np.testing.assert_allclose(y.real.ravel(), [1, 2, 3, 4], atol=1e-10)

    def test_pinv(self, session):
        session.eval("P = pinv(eye(3))")
        P = _unwrap(session.workspace.get("P"))
        np.testing.assert_allclose(P, np.eye(3), atol=1e-10)

    def test_kron(self, session):
        session.eval("K = kron(eye(2), eye(2))")
        K = _unwrap(session.workspace.get("K"))
        np.testing.assert_allclose(K, np.eye(4), atol=1e-10)


class TestSessionOutputBuffer:
    def test_disp_output(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        result = s.eval("disp(42)")
        assert "42" in result

    def test_multiple_disp(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval("x = 1")
        r1 = s.eval("disp(x)")
        r2 = s.eval("disp(x)")
        assert "1" in r1
        assert "1" in r2

    def test_output_doesnt_leak(self):
        from forge.engine.session import ForgeSession
        s = ForgeSession()
        s.eval("disp(42)")
        result = s.eval("x = 1")
        assert "42" not in result
