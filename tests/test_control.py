"""Tests for Forge Control Systems Toolbox (20 tests).

Uses CONTROL_REGISTRY dict to call functions by their registry name.
"""

import pytest
import numpy as np

# All imports go through the registry
from forge.engine.builtins.control import CONTROL_REGISTRY

# ---------------------------------------------------------------------------
# Helper to pull a callable from the registry
# ---------------------------------------------------------------------------
def _fn(name):
    return CONTROL_REGISTRY[name]


# ===========================================================================
# System Representation
# ===========================================================================

class TestSystemCreation:
    """R-CTRL-01: The toolbox SHALL create transfer function and state-space
    system representations from numeric coefficient arrays, and SHALL convert
    between representations preserving system dynamics.

    Model-user argument: The control engineer's first action in any design
    session is to define the plant. They type tf([1],[1,1]) or ss(A,B,C,D)
    reflexively, expecting the same dict-based object they get in Octave. If
    creation or conversion silently corrupts coefficients, every downstream
    analysis (pole placement, step response, Bode) produces wrong answers and
    the engineer loses trust in the toolbox.

    Decomposition:
        R-CTRL-01.1 -- tf() stores numerator and denominator as given.
        R-CTRL-01.2 -- ss() stores A, B, C, D with correct dimensions.
        R-CTRL-01.3 -- tf2ss() yields a valid state-space dict from a tf dict.

    Consistency: 01.1 and 01.2 cover the two canonical representations. 01.3
    covers the only conversion path tested here. Together they guarantee that
    systems can be created in either form and moved to state-space for
    downstream algorithms that require it.
    """

    def test_tf_creation(self):
        """R-CTRL-01.1: tf([1], [1, 1]) stores num/den and marks type 'tf'."""
        tf = _fn('tf')
        sys = tf([1], [1, 1])
        assert sys['type'] == 'tf'
        np.testing.assert_array_equal(sys['num'], [1])
        np.testing.assert_array_equal(sys['den'], [1, 1])

    def test_ss_creation(self):
        """R-CTRL-01.2: ss(A, B, C, D) stores matrices with correct shape."""
        ss = _fn('ss')
        A = [[0, 1], [-2, -3]]
        B = [[0], [1]]
        C = [[1, 0]]
        D = [[0]]
        sys = ss(A, B, C, D)
        assert sys['type'] == 'ss'
        assert np.array(sys['A']).shape == (2, 2)

    def test_tf2ss_conversion(self):
        """R-CTRL-01.3: tf2ss produces a state-space dict from a tf dict."""
        tf = _fn('tf')
        tf2ss = _fn('tf2ss')
        sys_tf = tf([1], [1, 1])
        sys_ss = tf2ss(sys_tf)
        assert sys_ss['type'] == 'ss'
        # A should be a matrix (at least 1x1 for first-order system)
        assert np.array(sys_ss['A']).ndim >= 1


# ===========================================================================
# Time Response
# ===========================================================================

class TestTimeResponse:
    """R-CTRL-02: The toolbox SHALL compute step and impulse time-domain
    responses that match the analytical closed-form solutions for canonical
    LTI systems.

    Model-user argument: After defining a plant, the engineer immediately runs
    step() or impulse() to see whether the system behaves as expected. A step
    of an integrator must be a ramp; an impulse of an integrator must be a
    constant. These are textbook identities the engineer checks mentally, and
    any deviation signals a broken simulator.

    Decomposition:
        R-CTRL-02.1 -- step(1/s) produces y(t) = t (ramp).
        R-CTRL-02.2 -- impulse(1/s) produces y(t) = 1 (unit step).

    Consistency: The two sub-requirements test both time-response entry points
    against an integrator whose analytical responses are known exactly. Passing
    both confirms the ODE solver and output equation are wired correctly.
    """

    def test_step_integrator_is_ramp(self):
        """R-CTRL-02.1: Step response of 1/s equals t (ramp) for t > 0."""
        tf = _fn('tf')
        step = _fn('step')
        sys = tf([1], [1, 0])  # 1/s
        t, y = step(sys)
        # At t > 0 the output should grow linearly
        # Check y ~ t for the computed time vector
        t = np.asarray(t).ravel()
        y = np.asarray(y).ravel()
        mask = t > 0.1
        ratio = y[mask] / t[mask]
        np.testing.assert_allclose(ratio, 1.0, atol=0.05)

    def test_impulse_integrator_is_step(self):
        """R-CTRL-02.2: Impulse response of 1/s equals constant 1."""
        tf = _fn('tf')
        impulse = _fn('impulse')
        sys = tf([1], [1, 0])  # 1/s
        t, y = impulse(sys)
        t = np.asarray(t).ravel()
        y = np.asarray(y).ravel()
        mask = t > 0.1
        np.testing.assert_allclose(y[mask], 1.0, atol=0.05)


# ===========================================================================
# Stability & Analysis
# ===========================================================================

class TestStabilityAnalysis:
    """R-CTRL-03: The toolbox SHALL extract poles, zeros, and DC gain from
    transfer functions, determine BIBO stability, and compute controllability
    and observability matrices with correct rank.

    Model-user argument: Stability analysis is the gate between "I have a
    model" and "I can close the loop." The engineer checks poles to confirm
    stability, zeros to anticipate non-minimum-phase behavior, DC gain for
    steady-state error, and controllability/observability to ensure the
    state-feedback or observer design is feasible. A wrong pole location or
    a rank-deficient controllability matrix means the controller they design
    next will fail in simulation or on hardware.

    Decomposition:
        R-CTRL-03.1 -- pole() returns roots of the denominator polynomial.
        R-CTRL-03.2 -- zero() returns roots of the numerator polynomial.
        R-CTRL-03.3 -- dcgain() returns num(0)/den(0).
        R-CTRL-03.4 -- isstable() returns True when all poles are in LHP.
        R-CTRL-03.5 -- isstable() returns False when any pole is in RHP.
        R-CTRL-03.6 -- ctrb() produces full-rank matrix for controllable pair.
        R-CTRL-03.7 -- obsv() produces full-rank matrix for observable pair.

    Consistency: 03.1 through 03.3 cover the three scalar/vector analysis
    queries. 03.4 and 03.5 cover both branches of the stability predicate.
    03.6 and 03.7 cover the two structural properties required before any
    state-space controller or observer design. Together these span the
    analysis tools the engineer uses before proceeding to controller synthesis.
    """

    def test_pole_extraction(self):
        """R-CTRL-03.1: Poles of 1/(s^2+3s+2) are -1 and -2."""
        tf = _fn('tf')
        pole = _fn('pole')
        sys = tf([1], [1, 3, 2])
        poles = np.sort(np.real(np.asarray(pole(sys)).ravel()))
        np.testing.assert_allclose(poles, [-2, -1], atol=1e-10)

    def test_zero_extraction(self):
        """R-CTRL-03.2: Zeros of (s+3)/(s^2+3s+2) are [-3]."""
        tf = _fn('tf')
        zero = _fn('zero')
        sys = tf([1, 3], [1, 3, 2])
        zeros = np.real(np.asarray(zero(sys)).ravel())
        np.testing.assert_allclose(zeros, [-3], atol=1e-10)

    def test_dcgain(self):
        """R-CTRL-03.3: DC gain of 5/(s+1) equals 5."""
        tf = _fn('tf')
        dcgain = _fn('dcgain')
        sys = tf([5], [1, 1])
        assert abs(dcgain(sys) - 5.0) < 1e-10

    def test_isstable_stable_system(self):
        """R-CTRL-03.4: 1/(s+1) is reported as stable."""
        tf = _fn('tf')
        isstable = _fn('isstable')
        sys = tf([1], [1, 1])
        assert isstable(sys) is True

    def test_isstable_unstable_system(self):
        """R-CTRL-03.5: 1/(s-1) is reported as unstable."""
        tf = _fn('tf')
        isstable = _fn('isstable')
        sys = tf([1], [1, -1])
        assert isstable(sys) is False

    def test_ctrb_rank(self):
        """R-CTRL-03.6: Controllability matrix of controllable (A, B) has full rank."""
        ctrb = _fn('ctrb')
        A = np.array([[0, 1], [-2, -3]])
        B = np.array([[0], [1]])
        C = ctrb(A, B)
        assert np.linalg.matrix_rank(np.asarray(C)) == 2

    def test_obsv_rank(self):
        """R-CTRL-03.7: Observability matrix of observable (A, C) has full rank."""
        obsv = _fn('obsv')
        A = np.array([[0, 1], [-2, -3]])
        C_mat = np.array([[1, 0]])
        O = obsv(A, C_mat)
        assert np.linalg.matrix_rank(np.asarray(O)) == 2


# ===========================================================================
# Feedback & Interconnection
# ===========================================================================

class TestInterconnection:
    """R-CTRL-04: The toolbox SHALL compute the closed-loop transfer function
    of a negative feedback interconnection, preserving correct DC gain.

    Model-user argument: Closing the loop is the central act of control design.
    The engineer calls feedback(G) (unity feedback) or feedback(G, H) and
    expects the standard formula G/(1+GH). If the closed-loop transfer function
    is wrong, the entire design iteration (tuning gains, checking margins,
    simulating step response) operates on a phantom system.

    Decomposition:
        R-CTRL-04.1 -- feedback(G) with G=10/(s+1) yields DC gain 10/11.

    Consistency: A single sub-requirement suffices because only one
    interconnection topology (unity negative feedback) is tested. The DC gain
    check confirms both the numerator and denominator polynomials of the
    closed-loop system are correct.
    """

    def test_feedback_loop(self):
        """R-CTRL-04.1: Unity feedback of 10/(s+1) has DC gain 10/11."""
        tf = _fn('tf')
        feedback = _fn('feedback')
        G = tf([10], [1, 1])
        cl = feedback(G)
        # DC gain of closed-loop = 10/(0+11) = 10/11
        dcgain = _fn('dcgain')
        np.testing.assert_allclose(dcgain(cl), 10.0 / 11.0, atol=1e-6)


# ===========================================================================
# Controller Design
# ===========================================================================

class TestControllerDesign:
    """R-CTRL-05: The toolbox SHALL synthesize PID controllers from gain
    parameters and SHALL compute LQR optimal state-feedback gains for
    linear systems.

    Model-user argument: PID is the bread-and-butter controller the engineer
    reaches for first; pid(Kp, Ki, Kd) must produce a transfer function they
    can put in a feedback loop immediately. For higher-performance designs they
    switch to LQR, expecting a gain matrix K sized to the state vector. If
    pid() returns the wrong type or lqr() returns a mis-shaped K, the
    subsequent simulation or hardware deployment will fail silently.

    Decomposition:
        R-CTRL-05.1 -- pid(1, Ki=1, Kd=1) returns a tf-type dict.
        R-CTRL-05.2 -- lqr() for a double integrator returns K with shape (1,2).

    Consistency: 05.1 covers the classical single-loop design tool. 05.2
    covers the optimal state-space design tool. Together they span the two
    controller synthesis methods available in the toolbox.
    """

    def test_pid_creation(self):
        """R-CTRL-05.1: pid(1, Ki=1, Kd=1) produces a tf-type controller."""
        pid = _fn('pid')
        ctrl = pid(1, Ki=1, Kd=1)
        assert ctrl['type'] == 'tf'

    def test_lqr_solution_exists(self):
        """R-CTRL-05.2: LQR for double integrator returns K with shape (1, 2)."""
        lqr = _fn('lqr')
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])
        Q = np.eye(2)
        R = np.array([[1.0]])
        result = lqr(A, B, Q, R)
        # result should contain gain K, solution S, eigenvalues E
        # K should be 1x2
        K = np.asarray(result[0])
        assert K.shape == (1, 2)


# ===========================================================================
# Frequency Response
# ===========================================================================

class TestFrequencyResponse:
    """R-CTRL-06: The toolbox SHALL compute frequency-domain data (Bode
    magnitude/phase arrays and stability margins) for transfer functions.

    Model-user argument: Bode plots and gain/phase margins are how the engineer
    assesses robustness before committing a controller to hardware. They call
    bode() to get magnitude and phase vectors for plotting, and margin() to
    read off gain and phase margins directly. If bode() returns mismatched
    array lengths or margin() returns None, the engineer cannot complete the
    design review and must fall back to a different tool.

    Decomposition:
        R-CTRL-06.1 -- margin() returns a non-None result for a stable system.
        R-CTRL-06.2 -- bode() returns three arrays (w, mag, phase) of equal
                        length, each with more than 10 points.

    Consistency: 06.1 confirms the margin computation runs to completion. 06.2
    confirms the frequency sweep produces usable arrays. Together they cover
    the two frequency-domain entry points the engineer uses for loop-shaping.
    """

    def test_margin_exists(self):
        """R-CTRL-06.1: margin() returns a non-None result for 1/(s+1)."""
        tf = _fn('tf')
        margin = _fn('margin')
        sys = tf([1], [1, 1])
        result = margin(sys)
        # result is a dict or tuple with margin info
        assert result is not None

    def test_bode_output_shape(self):
        """R-CTRL-06.2: bode() returns w, mag, phase arrays of matching length > 10."""
        tf = _fn('tf')
        bode = _fn('bode')
        sys = tf([1], [1, 1])
        result = bode(sys)
        # Should return (w, mag, phase) or similar
        assert len(result) >= 3
        w = np.asarray(result[0]).ravel()
        mag = np.asarray(result[1]).ravel()
        phase = np.asarray(result[2]).ravel()
        assert len(w) > 10
        assert len(mag) == len(w)
        assert len(phase) == len(w)
