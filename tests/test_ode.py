"""V&V tests for ODE toolbox (9 functions).

SRS trace: SRS-FUNC-001, SRS-VAL-001
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.ode import *


# ── Helper: extract solution arrays ──────────────────────────────

def _get_ty(result):
    """Extract (t, y) numpy arrays from ODE solver result tuple."""
    t_out, y_out = result
    t = np.asarray(_unwrap(t_out)).ravel()
    y = np.asarray(_unwrap(y_out))
    return t, y


# ══════════════════════════════════════════════════════════════════
# ODE45 Tests
# ══════════════════════════════════════════════════════════════════

class TestOde45:
    """R-ODE-01: ode45 SHALL integrate non-stiff ODE systems using an explicit
    Runge-Kutta (4,5) method, accepting (function_handle, tspan, y0) and
    returning (t, y) arrays whose trajectory error is within default tolerances.

    Model-user argument: The migrating engineer uses ode45 as a daily workhorse
    for non-stiff dynamics such as spring-mass systems, orbit propagation, and
    chemical kinetics. The (function_handle, tspan, y0) calling convention is
    muscle memory from years of Octave/MATLAB use. If ode45 rejects that
    signature or produces visibly wrong trajectories, the engineer will not
    trust Forge for any simulation work.

    Decomposition:
        R-ODE-01.1  Scalar exponential decay dy/dt = -y tracks analytical solution.
        R-ODE-01.2  Endpoint accuracy of exponential decay within 1e-3.
        R-ODE-01.3  2-D harmonic oscillator position component matches cos(t).
        R-ODE-01.4  2-D harmonic oscillator velocity component matches -sin(t).
        R-ODE-01.5  Constant-rate growth dy/dt = 1 yields y = t exactly.

    Consistency: R-ODE-01.1 and R-ODE-01.2 cover scalar first-order accuracy
    (trajectory and endpoint). R-ODE-01.3 and R-ODE-01.4 cover both state
    variables of a coupled second-order system, confirming multi-dimensional
    integration. R-ODE-01.5 tests the degenerate constant-RHS case, verifying
    no spurious drift. Together these span scalar, vector, trivial, and
    oscillatory dynamics, which fully exercise the RK45 adaptive stepper.
    """

    def test_exponential_decay(self):
        """R-ODE-01.1: Scalar exponential decay trajectory within rtol 1e-2."""
        def dydt(t, y):
            return -y
        tspan = [0.0, 3.0]
        y0 = [1.0]
        t, y = _get_ty(forge_ode45(dydt, tspan, y0))
        y_exact = np.exp(-t)
        assert_allclose(y.ravel(), y_exact, rtol=1e-2)

    def test_exponential_decay_endpoint(self):
        """R-ODE-01.2: Endpoint y(3) within 1e-3 of e^{-3}."""
        def dydt(t, y):
            return -y
        tspan = [0.0, 3.0]
        y0 = [1.0]
        t, y = _get_ty(forge_ode45(dydt, tspan, y0))
        assert abs(y.ravel()[-1] - np.exp(-3.0)) < 1e-3

    def test_harmonic_oscillator(self):
        """R-ODE-01.3: Harmonic oscillator position y1(t) matches cos(t) within atol 2e-3.

        System: dy1/dt = y2, dy2/dt = -y1.
        y(0) = [1, 0] => y1(t) = cos(t).
        """
        def dydt(t, y):
            return np.array([y[1], -y[0]])
        tspan = [0.0, 2 * np.pi]
        y0 = [1.0, 0.0]
        t, y = _get_ty(forge_ode45(dydt, tspan, y0))
        y1 = y[:, 0]
        y1_exact = np.cos(t)
        assert_allclose(y1, y1_exact, atol=2e-3)

    def test_harmonic_oscillator_velocity(self):
        """R-ODE-01.4: Harmonic oscillator velocity y2(t) matches -sin(t) within atol 2e-3."""
        def dydt(t, y):
            return np.array([y[1], -y[0]])
        tspan = [0.0, 2 * np.pi]
        y0 = [1.0, 0.0]
        t, y = _get_ty(forge_ode45(dydt, tspan, y0))
        y2 = y[:, 1]
        y2_exact = -np.sin(t)
        assert_allclose(y2, y2_exact, atol=2e-3)

    def test_linear_growth(self):
        """R-ODE-01.5: Constant RHS dy/dt = 1 yields y = t within atol 1e-8."""
        def dydt(t, y):
            return np.array([1.0])
        tspan = [0.0, 5.0]
        y0 = [0.0]
        t, y = _get_ty(forge_ode45(dydt, tspan, y0))
        assert_allclose(y.ravel(), t, atol=1e-8)


# ══════════════════════════════════════════════════════════════════
# ODE23 Tests
# ══════════════════════════════════════════════════════════════════

class TestOde23:
    """R-ODE-02: ode23 SHALL integrate non-stiff ODE systems using an explicit
    Runge-Kutta (2,3) method, accepting the same (function_handle, tspan, y0)
    signature as ode45 and returning (t, y) arrays with lower-order but
    acceptable accuracy.

    Model-user argument: The engineer reaches for ode23 when a quick, coarse
    estimate is acceptable, or when the problem is smooth enough that the
    cheaper stepper saves wall-clock time without sacrificing needed precision.
    Interchangeability with ode45 (same calling convention, same output shape)
    lets the engineer swap solvers in one line. If ode23 diverges from the
    expected calling pattern or produces grossly wrong results, the engineer
    loses a standard tool from their workflow.

    Decomposition:
        R-ODE-02.1  Scalar exponential decay trajectory within rtol 1e-2.
        R-ODE-02.2  Harmonic oscillator position within atol 1e-2.
        R-ODE-02.3  Quadratic growth dy/dt = 2t yields y = t^2 within atol 1e-3.

    Consistency: R-ODE-02.1 repeats the canonical scalar test at the same
    tolerance as ode45 to confirm interchangeability. R-ODE-02.2 checks
    multi-dimensional integration. R-ODE-02.3 adds a time-dependent RHS
    (polynomial growth), which is a distinct class not tested in the ode45
    suite. Together these confirm that the lower-order method handles the same
    problem classes as ode45 within its documented accuracy.
    """

    def test_exponential_decay(self):
        """R-ODE-02.1: Scalar exponential decay trajectory within rtol 1e-2."""
        def dydt(t, y):
            return -y
        tspan = [0.0, 3.0]
        y0 = [1.0]
        t, y = _get_ty(forge_ode23(dydt, tspan, y0))
        y_exact = np.exp(-t)
        assert_allclose(y.ravel(), y_exact, rtol=1e-2)

    def test_harmonic_oscillator(self):
        """R-ODE-02.2: Harmonic oscillator position within atol 1e-2."""
        def dydt(t, y):
            return np.array([y[1], -y[0]])
        tspan = [0.0, 2 * np.pi]
        y0 = [1.0, 0.0]
        t, y = _get_ty(forge_ode23(dydt, tspan, y0))
        y1 = y[:, 0]
        y1_exact = np.cos(t)
        assert_allclose(y1, y1_exact, atol=1e-2)

    def test_quadratic_growth(self):
        """R-ODE-02.3: Quadratic growth dy/dt = 2t yields y = t^2 within atol 1e-3."""
        def dydt(t, y):
            return np.array([2.0 * t])
        tspan = [0.0, 4.0]
        y0 = [0.0]
        t, y = _get_ty(forge_ode23(dydt, tspan, y0))
        assert_allclose(y.ravel(), t ** 2, atol=1e-3)


# ══════════════════════════════════════════════════════════════════
# ODE15s (Stiff Solver) Tests
# ══════════════════════════════════════════════════════════════════

class TestOde15s:
    """R-ODE-03: ode15s SHALL integrate stiff ODE systems using an implicit
    method, accepting (function_handle, tspan, y0) and producing (t, y) arrays
    that remain stable and accurate for problems with widely separated time
    scales.

    Model-user argument: The engineer encounters stiff systems in circuit
    transient simulation, fast chemical reaction channels, and thermal problems
    with mixed conduction/radiation time constants. Attempting these with ode45
    causes catastrophic step-size collapse. ode15s is the expected escape hatch;
    if it fails to converge or mishandles the stiff decay, the engineer has no
    viable path for an entire class of real-world problems.

    Decomposition:
        R-ODE-03.1  Stiff exponential decay (lambda = -1000) trajectory within rtol 1e-2.
        R-ODE-03.2  Endpoint y(0.01) within 1e-3 of e^{-10}.

    Consistency: R-ODE-03.1 confirms the implicit stepper tracks a rapidly
    decaying solution across its full trajectory (many e-folding times).
    R-ODE-03.2 verifies the final value independently, catching cases where
    trajectory interpolation is correct but endpoint handling diverges. Together
    they prove stability and accuracy for the canonical stiff test problem.
    """

    def test_stiff_exponential_decay(self):
        """R-ODE-03.1: Stiff decay (lambda = -1000) trajectory within rtol 1e-2."""
        def dydt(t, y):
            return -1000.0 * y
        tspan = [0.0, 0.01]
        y0 = [1.0]
        t, y = _get_ty(forge_ode15s(dydt, tspan, y0))
        y_exact = np.exp(-1000.0 * t)
        assert_allclose(y.ravel(), y_exact, rtol=1e-2)

    def test_stiff_decay_endpoint(self):
        """R-ODE-03.2: Stiff decay endpoint y(0.01) within 1e-3 of e^{-10}."""
        def dydt(t, y):
            return -1000.0 * y
        tspan = [0.0, 0.01]
        y0 = [1.0]
        t, y = _get_ty(forge_ode15s(dydt, tspan, y0))
        assert abs(y.ravel()[-1] - np.exp(-10.0)) < 1e-3


# ══════════════════════════════════════════════════════════════════
# ODE Options Tests
# ══════════════════════════════════════════════════════════════════

class TestOdeOptions:
    """R-ODE-04: odeset SHALL construct an options structure from key-value
    pairs, and odeget SHALL retrieve individual fields from that structure,
    following the Octave/MATLAB calling convention. Default tolerances SHALL be
    RelTol = 1e-3 and AbsTol = 1e-6.

    Model-user argument: The engineer routinely tightens tolerances for
    validation runs (e.g., odeset('RelTol', 1e-8)) and loosens them for quick
    parameter sweeps. The odeset/odeget pair is the standard mechanism; any
    deviation from expected defaults or key-value semantics will silently
    corrupt integration accuracy. If odeget returns the wrong value or odeset
    ignores a key, the engineer will spend hours debugging phantom numerical
    errors.

    Decomposition:
        R-ODE-04.1  Default RelTol equals 1e-3.
        R-ODE-04.2  Default AbsTol equals 1e-6.
        R-ODE-04.3  Key-value pairs set both RelTol and AbsTol correctly.
        R-ODE-04.4  odeget retrieves a previously set field value.
        R-ODE-04.5  odeget on a nonexistent field returns None (not an error).

    Consistency: R-ODE-04.1 and R-ODE-04.2 verify the two critical default
    tolerances independently. R-ODE-04.3 confirms the key-value constructor
    overwrites defaults for multiple fields in a single call. R-ODE-04.4
    verifies the round-trip (set then get). R-ODE-04.5 covers the edge case
    of querying an absent field, ensuring graceful fallback. Together these
    prove that the options pipeline (construction, storage, retrieval, defaults,
    missing-field handling) is complete and correct.
    """

    def test_odeset_default_reltol(self):
        """R-ODE-04.1: Default RelTol equals 1e-3."""
        opts = forge_odeset()
        assert opts.RelTol == pytest.approx(1e-3)

    def test_odeset_default_abstol(self):
        """R-ODE-04.2: Default AbsTol equals 1e-6."""
        opts = forge_odeset()
        assert opts.AbsTol == pytest.approx(1e-6)

    def test_odeset_key_value(self):
        """R-ODE-04.3: Key-value pairs correctly set RelTol and AbsTol."""
        opts = forge_odeset('RelTol', 1e-6, 'AbsTol', 1e-9)
        assert opts.RelTol == pytest.approx(1e-6)
        assert opts.AbsTol == pytest.approx(1e-9)

    def test_odeget_field(self):
        """R-ODE-04.4: odeget retrieves the value previously set by odeset."""
        opts = forge_odeset('RelTol', 1e-8)
        val = forge_odeget(opts, 'RelTol')
        assert val == pytest.approx(1e-8)

    def test_odeget_missing_field(self):
        """R-ODE-04.5: odeget on a nonexistent field returns None."""
        opts = forge_odeset()
        val = forge_odeget(opts, 'NonExistent')
        assert val is None
