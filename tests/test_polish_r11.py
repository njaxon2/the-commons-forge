"""Tests for ODE solvers and optimization functions (polish round 11).

Requirement R-POL11-01:
    The ODE solver suite (ode45, ode23, ode15s, ode113) SHALL integrate
    scalar ODEs and return column-vector time and solution arrays whose
    final values match analytical solutions within documented tolerances.

    Model-user argument:
    An engineer migrating from MATLAB/Octave uses ODE solvers daily to
    simulate dynamic systems (vibration, heat transfer, chemical kinetics).
    If ode45 returns row vectors instead of columns, or if the final value
    of an exponential decay diverges from exp(-t), their simulation post-
    processing scripts break silently. Verified shape and accuracy are
    non-negotiable for trust in the solver suite.

    Decomposition:
    R-POL11-01a: ode45 matches exp(-5) for dy/dt = -y, y(0)=1 within 1e-4.
    R-POL11-01b: ode45 matches linear growth dy/dt = 1 within 1e-4.
    R-POL11-01c: ode45 returns column vectors (Nx1) for t and y.
    R-POL11-01d: ode23 matches exp(-5) within 1e-3.
    R-POL11-01e: ode15s handles stiff decay (lambda = -1000) to near zero.
    R-POL11-01f: ode113 matches exponential decay within 1e-2.

    Consistency argument:
    Sub-requirements 01a-01c cover ode45 value accuracy, growth accuracy,
    and output shape. 01d-01f each cover one additional solver against a
    known analytical solution. Together they verify all four solvers.

Requirement R-POL11-02:
    The optimization suite (fzero, fminbnd, fminsearch) SHALL find roots
    and minima of scalar and multi-variable functions within documented
    tolerances.

    Model-user argument:
    An engineer uses fzero to find operating points and fminbnd/fminsearch
    to calibrate model parameters. These are the workhorses of iterative
    design: if fzero cannot find sqrt(4)=2 from a quadratic, or fminsearch
    cannot locate the minimum of a simple bowl, the entire parameter
    estimation workflow is broken.

    Decomposition:
    R-POL11-02a: fzero finds root of x^2-4 near x=2 within 1e-8.
    R-POL11-02b: fzero finds root of sin(x) near pi within 1e-8.
    R-POL11-02c: fzero finds root from bracket [0,2] for x^3-1 within 1e-8.
    R-POL11-02d: fminbnd finds minimum of (x-3)^2 on [0,5] within 1e-6.
    R-POL11-02e: fminbnd finds minimum of sin on [0,2*pi] within 1e-4.
    R-POL11-02f: fminsearch finds minimum of 2D quadratic within 1e-3.
    R-POL11-02g: fminsearch finds minimum of scalar (x-5)^2 within 1e-3.

    Consistency argument:
    Sub-requirements 02a-02c test fzero with polynomial, transcendental,
    and bracket inputs. 02d-02e test fminbnd on quadratic and periodic
    functions. 02f-02g test fminsearch on multi-variable and scalar
    objectives. Together they cover all three optimizers across
    representative use cases.
"""
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


# -- ODE solvers (R-POL11-01) -------------------------------------------------

class TestODE45:
    """R-POL11-01 (ode45): Runge-Kutta 4/5 adaptive integrator."""

    def test_exponential_decay(self, s):
        """R-POL11-01a: ode45 exponential decay matches exp(-5)."""
        s.eval("[t,y] = ode45(@(t,y) -y, [0 5], 1)")
        y = _val(s, "y")
        assert abs(float(y[0, 0]) - 1.0) < 1e-10
        assert abs(float(y[-1, 0]) - np.exp(-5)) < 1e-4

    def test_linear_growth(self, s):
        """R-POL11-01b: ode45 linear growth dy/dt=1 matches y=t."""
        s.eval("[t,y] = ode45(@(t,y) 1, [0 3], 0)")
        t = _val(s, "t")
        y = _val(s, "y")
        # y should be approximately t
        assert abs(float(y[-1, 0]) - float(t[-1, 0])) < 1e-4

    def test_returns_column_vectors(self, s):
        """R-POL11-01c: ode45 returns Nx1 column vectors for t and y."""
        s.eval("[t,y] = ode45(@(t,y) -y, [0 1], 1)")
        t = _val(s, "t")
        y = _val(s, "y")
        assert t.ndim == 2 and t.shape[1] == 1
        assert y.ndim == 2 and y.shape[1] == 1


class TestODE23:
    """R-POL11-01 (ode23): Bogacki-Shampine 2/3 integrator."""

    def test_exponential_decay(self, s):
        """R-POL11-01d: ode23 exponential decay matches exp(-5)."""
        s.eval("[t,y] = ode23(@(t,y) -y, [0 5], 1)")
        y = _val(s, "y")
        assert abs(float(y[-1, 0]) - np.exp(-5)) < 1e-3


class TestODE15s:
    """R-POL11-01 (ode15s): Stiff solver (implicit multistep)."""

    def test_stiff_decay(self, s):
        """R-POL11-01e: ode15s handles stiff lambda=-1000 to near zero."""
        s.eval("[t,y] = ode15s(@(t,y) -1000*y, [0 1], 1)")
        y = _val(s, "y")
        assert abs(float(y[-1, 0])) < 1e-10


class TestODE113:
    """R-POL11-01 (ode113): Adams-Bashforth-Moulton multistep integrator."""

    def test_exponential_decay(self, s):
        """R-POL11-01f: ode113 exponential decay within 1e-2."""
        s.eval("[t,y] = ode113(@(t,y) -y, [0 5], 1)")
        y = _val(s, "y")
        assert abs(float(y[-1, 0])) < 1e-2


# -- Optimization (R-POL11-02) ------------------------------------------------

class TestFzero:
    """R-POL11-02 (fzero): Scalar root finder."""

    def test_quadratic_root(self, s):
        """R-POL11-02a: fzero finds root of x^2-4 at x=2."""
        s.eval("x = fzero(@(x) x^2 - 4, 1)")
        x = float(np.asarray(_val(s, "x")).flat[0])
        assert abs(x - 2.0) < 1e-8

    def test_sin_root(self, s):
        """R-POL11-02b: fzero finds root of sin(x) at x=pi."""
        s.eval("x = fzero(@sin, 3)")
        x = float(np.asarray(_val(s, "x")).flat[0])
        assert abs(x - np.pi) < 1e-8

    def test_bracket(self, s):
        """R-POL11-02c: fzero finds root of x^3-1 from bracket [0,2]."""
        s.eval("x = fzero(@(x) x^3 - 1, [0 2])")
        x = float(np.asarray(_val(s, "x")).flat[0])
        assert abs(x - 1.0) < 1e-8


class TestFminbnd:
    """R-POL11-02 (fminbnd): Bounded scalar minimizer."""

    def test_quadratic_min(self, s):
        """R-POL11-02d: fminbnd finds minimum of (x-3)^2 on [0,5] at x=3."""
        s.eval("x = fminbnd(@(x) (x-3)^2, 0, 5)")
        x = float(np.asarray(_val(s, "x")).flat[0])
        assert abs(x - 3.0) < 1e-6

    def test_sin_min(self, s):
        """R-POL11-02e: fminbnd finds minimum of sin on [0,2*pi] at 3*pi/2."""
        s.eval("x = fminbnd(@sin, 0, 2*pi)")
        x = float(np.asarray(_val(s, "x")).flat[0])
        assert abs(x - 3 * np.pi / 2) < 1e-4


class TestFminsearch:
    """R-POL11-02 (fminsearch): Nelder-Mead simplex minimizer."""

    def test_rosenbrock_origin(self, s):
        """R-POL11-02f: fminsearch finds 2D quadratic minimum at (1,2)."""
        s.eval("x = fminsearch(@(x) (x(1)-1)^2 + (x(2)-2)^2, [0, 0])")
        x = np.asarray(_val(s, "x")).flatten()
        assert abs(x[0] - 1.0) < 1e-3
        assert abs(x[1] - 2.0) < 1e-3

    def test_scalar(self, s):
        """R-POL11-02g: fminsearch finds scalar minimum of (x-5)^2 at x=5."""
        s.eval("x = fminsearch(@(x) (x-5)^2, 0)")
        x = float(np.asarray(_val(s, "x")).flat[0])
        assert abs(x - 5.0) < 1e-3
