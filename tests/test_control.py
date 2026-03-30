# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
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
    def test_tf_creation(self):
        """tf([1], [1, 1]) should create a valid transfer function dict."""
        tf = _fn('tf')
        sys = tf([1], [1, 1])
        assert sys['type'] == 'tf'
        np.testing.assert_array_equal(sys['num'], [1])
        np.testing.assert_array_equal(sys['den'], [1, 1])

    def test_ss_creation(self):
        """ss(A, B, C, D) should create a valid state-space dict."""
        ss = _fn('ss')
        A = [[0, 1], [-2, -3]]
        B = [[0], [1]]
        C = [[1, 0]]
        D = [[0]]
        sys = ss(A, B, C, D)
        assert sys['type'] == 'ss'
        assert np.array(sys['A']).shape == (2, 2)

    def test_tf2ss_conversion(self):
        """tf2ss should produce a valid state-space representation."""
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
    def test_step_integrator_is_ramp(self):
        """Step response of 1/s (integrator) should be a ramp: y(t) = t."""
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
        """Impulse response of 1/s should be a unit step (constant 1)."""
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
    def test_pole_extraction(self):
        """Poles of 1/(s+1)(s+2) = 1/(s^2+3s+2) should be -1 and -2."""
        tf = _fn('tf')
        pole = _fn('pole')
        sys = tf([1], [1, 3, 2])
        poles = np.sort(np.real(np.asarray(pole(sys)).ravel()))
        np.testing.assert_allclose(poles, [-2, -1], atol=1e-10)

    def test_zero_extraction(self):
        """Zeros of (s+3)/(s^2+3s+2) should be [-3]."""
        tf = _fn('tf')
        zero = _fn('zero')
        sys = tf([1, 3], [1, 3, 2])
        zeros = np.real(np.asarray(zero(sys)).ravel())
        np.testing.assert_allclose(zeros, [-3], atol=1e-10)

    def test_dcgain(self):
        """DC gain of 5/(s+1) should be 5."""
        tf = _fn('tf')
        dcgain = _fn('dcgain')
        sys = tf([5], [1, 1])
        assert abs(dcgain(sys) - 5.0) < 1e-10

    def test_isstable_stable_system(self):
        """1/(s+1) is stable."""
        tf = _fn('tf')
        isstable = _fn('isstable')
        sys = tf([1], [1, 1])
        assert isstable(sys) is True

    def test_isstable_unstable_system(self):
        """1/(s-1) is unstable."""
        tf = _fn('tf')
        isstable = _fn('isstable')
        sys = tf([1], [1, -1])
        assert isstable(sys) is False

    def test_ctrb_rank(self):
        """Controllability matrix of a controllable 2x2 system has full rank."""
        ctrb = _fn('ctrb')
        A = np.array([[0, 1], [-2, -3]])
        B = np.array([[0], [1]])
        C = ctrb(A, B)
        assert np.linalg.matrix_rank(np.asarray(C)) == 2

    def test_obsv_rank(self):
        """Observability matrix of an observable 2x2 system has full rank."""
        obsv = _fn('obsv')
        A = np.array([[0, 1], [-2, -3]])
        C_mat = np.array([[1, 0]])
        O = obsv(A, C_mat)
        assert np.linalg.matrix_rank(np.asarray(O)) == 2


# ===========================================================================
# Feedback & Interconnection
# ===========================================================================

class TestInterconnection:
    def test_feedback_loop(self):
        """Negative feedback of G=10/(s+1) with H=1 should give 10/(s+11)."""
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
    def test_pid_creation(self):
        """pid(1, 1, 1) should create a PID controller transfer function."""
        pid = _fn('pid')
        ctrl = pid(1, Ki=1, Kd=1)
        assert ctrl['type'] == 'tf'

    def test_lqr_solution_exists(self):
        """LQR for double integrator should return gain matrix K."""
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
    def test_margin_exists(self):
        """margin() should return gain margin and phase margin values."""
        tf = _fn('tf')
        margin = _fn('margin')
        sys = tf([1], [1, 1])
        result = margin(sys)
        # result is a dict or tuple with margin info
        assert result is not None

    def test_bode_output_shape(self):
        """bode() should return frequency, magnitude, and phase arrays."""
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
