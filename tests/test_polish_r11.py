"""Tests for ODE solvers and optimization functions (polish round 11)."""
import numpy as np
import pytest
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


def _val(s, name):
    """Get workspace variable as raw numpy array."""
    v = s._engine.workspace.get(name)
    return v.data if hasattr(v, "data") else np.array(v)


# ── ODE solvers ──────────────────────────────────────────────────────────

class TestODE45:
    def test_exponential_decay(self, s):
        s.eval("[t,y] = ode45(@(t,y) -y, [0 5], 1)")
        y = _val(s, "y")
        assert abs(float(y[0, 0]) - 1.0) < 1e-10
        assert abs(float(y[-1, 0]) - np.exp(-5)) < 1e-4

    def test_linear_growth(self, s):
        s.eval("[t,y] = ode45(@(t,y) 1, [0 3], 0)")
        t = _val(s, "t")
        y = _val(s, "y")
        # y should be approximately t
        assert abs(float(y[-1, 0]) - float(t[-1, 0])) < 1e-4

    def test_returns_column_vectors(self, s):
        s.eval("[t,y] = ode45(@(t,y) -y, [0 1], 1)")
        t = _val(s, "t")
        y = _val(s, "y")
        assert t.ndim == 2 and t.shape[1] == 1
        assert y.ndim == 2 and y.shape[1] == 1


class TestODE23:
    def test_exponential_decay(self, s):
        s.eval("[t,y] = ode23(@(t,y) -y, [0 5], 1)")
        y = _val(s, "y")
        assert abs(float(y[-1, 0]) - np.exp(-5)) < 1e-3


class TestODE15s:
    def test_stiff_decay(self, s):
        s.eval("[t,y] = ode15s(@(t,y) -1000*y, [0 1], 1)")
        y = _val(s, "y")
        assert abs(float(y[-1, 0])) < 1e-10


class TestODE113:
    def test_exponential_decay(self, s):
        s.eval("[t,y] = ode113(@(t,y) -y, [0 5], 1)")
        y = _val(s, "y")
        assert abs(float(y[-1, 0])) < 1e-2


# ── Optimization ─────────────────────────────────────────────────────────

class TestFzero:
    def test_quadratic_root(self, s):
        s.eval("x = fzero(@(x) x^2 - 4, 1)")
        x = float(np.asarray(_val(s, "x")).flat[0])
        assert abs(x - 2.0) < 1e-8

    def test_sin_root(self, s):
        s.eval("x = fzero(@sin, 3)")
        x = float(np.asarray(_val(s, "x")).flat[0])
        assert abs(x - np.pi) < 1e-8

    def test_bracket(self, s):
        s.eval("x = fzero(@(x) x^3 - 1, [0 2])")
        x = float(np.asarray(_val(s, "x")).flat[0])
        assert abs(x - 1.0) < 1e-8


class TestFminbnd:
    def test_quadratic_min(self, s):
        s.eval("x = fminbnd(@(x) (x-3)^2, 0, 5)")
        x = float(np.asarray(_val(s, "x")).flat[0])
        assert abs(x - 3.0) < 1e-6

    def test_sin_min(self, s):
        s.eval("x = fminbnd(@sin, 0, 2*pi)")
        x = float(np.asarray(_val(s, "x")).flat[0])
        assert abs(x - 3 * np.pi / 2) < 1e-4


class TestFminsearch:
    def test_rosenbrock_origin(self, s):
        s.eval("x = fminsearch(@(x) (x(1)-1)^2 + (x(2)-2)^2, [0, 0])")
        x = np.asarray(_val(s, "x")).flatten()
        assert abs(x[0] - 1.0) < 1e-3
        assert abs(x[1] - 2.0) < 1e-3

    def test_scalar(self, s):
        s.eval("x = fminsearch(@(x) (x-5)^2, 0)")
        x = float(np.asarray(_val(s, "x")).flat[0])
        assert abs(x - 5.0) < 1e-3
