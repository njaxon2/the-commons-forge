# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
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

    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 => y(t) = e^{-t}."""
        def dydt(t, y):
            return -y
        tspan = [0.0, 3.0]
        y0 = [1.0]
        t, y = _get_ty(forge_ode45(dydt, tspan, y0))
        y_exact = np.exp(-t)
        assert_allclose(y.ravel(), y_exact, rtol=1e-2)

    def test_exponential_decay_endpoint(self):
        """Check y(3) ~ e^{-3}."""
        def dydt(t, y):
            return -y
        tspan = [0.0, 3.0]
        y0 = [1.0]
        t, y = _get_ty(forge_ode45(dydt, tspan, y0))
        assert abs(y.ravel()[-1] - np.exp(-3.0)) < 1e-3

    def test_harmonic_oscillator(self):
        """x'' + x = 0 => x(t) = cos(t), x'(t) = -sin(t).

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
        """Check velocity component: y2(t) = -sin(t)."""
        def dydt(t, y):
            return np.array([y[1], -y[0]])
        tspan = [0.0, 2 * np.pi]
        y0 = [1.0, 0.0]
        t, y = _get_ty(forge_ode45(dydt, tspan, y0))
        y2 = y[:, 1]
        y2_exact = -np.sin(t)
        assert_allclose(y2, y2_exact, atol=2e-3)

    def test_linear_growth(self):
        """dy/dt = 1, y(0) = 0 => y(t) = t."""
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

    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 => y(t) = e^{-t}."""
        def dydt(t, y):
            return -y
        tspan = [0.0, 3.0]
        y0 = [1.0]
        t, y = _get_ty(forge_ode23(dydt, tspan, y0))
        y_exact = np.exp(-t)
        assert_allclose(y.ravel(), y_exact, rtol=1e-2)

    def test_harmonic_oscillator(self):
        """Same harmonic oscillator as ode45 test but with ode23."""
        def dydt(t, y):
            return np.array([y[1], -y[0]])
        tspan = [0.0, 2 * np.pi]
        y0 = [1.0, 0.0]
        t, y = _get_ty(forge_ode23(dydt, tspan, y0))
        y1 = y[:, 0]
        y1_exact = np.cos(t)
        assert_allclose(y1, y1_exact, atol=1e-2)

    def test_quadratic_growth(self):
        """dy/dt = 2*t, y(0) = 0 => y(t) = t^2."""
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

    def test_stiff_exponential_decay(self):
        """Stiff problem: dy/dt = -1000*y, y(0) = 1 => y(t) = e^{-1000t}."""
        def dydt(t, y):
            return -1000.0 * y
        tspan = [0.0, 0.01]
        y0 = [1.0]
        t, y = _get_ty(forge_ode15s(dydt, tspan, y0))
        y_exact = np.exp(-1000.0 * t)
        assert_allclose(y.ravel(), y_exact, rtol=1e-2)

    def test_stiff_decay_endpoint(self):
        """Check endpoint: y(0.01) ~ e^{-10} ~ 4.54e-5."""
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

    def test_odeset_default_reltol(self):
        """odeset() should have default RelTol = 1e-3."""
        opts = forge_odeset()
        assert opts.RelTol == pytest.approx(1e-3)

    def test_odeset_default_abstol(self):
        opts = forge_odeset()
        assert opts.AbsTol == pytest.approx(1e-6)

    def test_odeset_key_value(self):
        """odeset('RelTol', 1e-6) should set RelTol."""
        opts = forge_odeset('RelTol', 1e-6, 'AbsTol', 1e-9)
        assert opts.RelTol == pytest.approx(1e-6)
        assert opts.AbsTol == pytest.approx(1e-9)

    def test_odeget_field(self):
        """odeget(opts, 'RelTol') should return the set value."""
        opts = forge_odeset('RelTol', 1e-8)
        val = forge_odeget(opts, 'RelTol')
        assert val == pytest.approx(1e-8)

    def test_odeget_missing_field(self):
        """odeget on a missing field should return default (None)."""
        opts = forge_odeset()
        val = forge_odeget(opts, 'NonExistent')
        assert val is None
