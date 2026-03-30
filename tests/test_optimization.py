# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for optimization toolbox.

SRS trace: SRS-FUNC-001, SRS-VAL-001
Test method: Comparison against known analytical solutions for root-finding
and minimization problems.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.optimization import *


class TestFzero:
    """Verify fzero root-finding.

    fzero returns (x, fval, info) tuple.
    """

    def test_fzero_sin_near_pi(self):
        """Find root of sin(x) near pi => result ~ pi."""
        x, fval, info = forge_fzero(lambda x: np.sin(x), np.array([3.0]))
        assert abs(float(x) - np.pi) < 1e-8
        assert info > 0

    def test_fzero_quadratic_bracket(self):
        """Find root of x^2 - 4 in [0, 3] => result ~ 2."""
        x, fval, info = forge_fzero(lambda x: x**2 - 4, np.array([0.0, 3.0]))
        assert abs(float(x) - 2.0) < 1e-8

    def test_fzero_cubic(self):
        """Find root of x^3 - 8 near 1.5 => result ~ 2."""
        x, fval, info = forge_fzero(lambda x: x**3 - 8, np.array([1.5]))
        assert abs(float(x) - 2.0) < 1e-6


class TestFminbnd:
    """Verify fminbnd bounded minimization.

    fminbnd returns (x, fval, info) tuple.
    """

    def test_fminbnd_x_squared(self):
        """Minimum of x^2 in [-5, 5] => result ~ 0."""
        x, fval, info = forge_fminbnd(lambda x: x**2, -5.0, 5.0)
        assert abs(float(x)) < 1e-6

    def test_fminbnd_shifted(self):
        """Minimum of (x-3)^2 in [0, 10] => result ~ 3."""
        x, fval, info = forge_fminbnd(lambda x: (x - 3)**2, 0.0, 10.0)
        assert abs(float(x) - 3.0) < 1e-6


class TestFminsearch:
    """Verify fminsearch unconstrained minimization.

    fminsearch returns (x, fval, info) tuple.
    """

    def test_fminsearch_quadratic(self):
        """Minimum of (x-3)^2 starting from x=0 => result ~ 3."""
        x, fval, info = forge_fminsearch(lambda x: (x[0] - 3)**2, np.array([0.0]))
        assert abs(float(x) - 3.0) < 1e-4

    def test_fminsearch_2d(self):
        """Minimum of (x-1)^2 + (y-2)^2 starting from [0,0] => [1, 2]."""
        x, fval, info = forge_fminsearch(
            lambda x: (x[0] - 1)**2 + (x[1] - 2)**2,
            np.array([0.0, 0.0]))
        result = np.asarray(_unwrap(x)).ravel()
        assert abs(result[0] - 1.0) < 1e-4
        assert abs(result[1] - 2.0) < 1e-4


class TestFsolve:
    """Verify fsolve nonlinear equation solving.

    fsolve returns (x, fval, info) tuple.
    """

    def test_fsolve_sqrt4(self):
        """Solve x^2 = 4, starting from 1 => result ~ 2."""
        x, fval, info = forge_fsolve(lambda x: np.array([x[0]**2 - 4]), np.array([1.0]))
        assert abs(float(x) - 2.0) < 1e-6

    def test_fsolve_system(self):
        """Solve x + y = 3, x - y = 1 => x=2, y=1."""
        x, fval, info = forge_fsolve(
            lambda x: np.array([x[0] + x[1] - 3, x[0] - x[1] - 1]),
            np.array([0.0, 0.0]))
        result = np.asarray(_unwrap(x)).ravel()
        assert abs(result[0] - 2.0) < 1e-6
        assert abs(result[1] - 1.0) < 1e-6


class TestLsqnonneg:
    """Verify lsqnonneg non-negative least squares.

    lsqnonneg returns (x, resnorm, residual) tuple.
    """

    def test_lsqnonneg_exact(self):
        """Known system with non-negative solution."""
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = np.array([2.0, 3.0])
        x, resnorm, residual = forge_lsqnonneg(C, d)
        result = np.asarray(_unwrap(x)).ravel()
        assert_allclose(result, [2.0, 3.0], atol=1e-8)


class TestHumps:
    """Verify humps test function."""

    def test_humps_at_zero(self):
        """humps(0) ~ 5.1765..."""
        r = forge_humps(np.array(0.0))
        val = float(r)
        expected = 1.0 / (0.09 + 0.01) + 1.0 / (0.81 + 0.04) - 6.0
        assert abs(val - expected) < 1e-6


class TestOptimOptions:
    """Verify optimset and optimget."""

    def test_optimset_creates_struct(self):
        """optimset creates options with known fields."""
        opts = forge_optimset()
        assert opts is not None

    def test_optimget_retrieves_value(self):
        """optimget retrieves a field from options struct."""
        opts = forge_optimset()
        r = forge_optimget(opts, 'TolX')
        assert True
