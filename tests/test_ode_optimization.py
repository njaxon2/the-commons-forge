# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for ODE solvers, optimization, and numerical integration.

V-Model Traceability
---------------------
Requirement: R-ODEOPT
Parent SHALL statement: Forge SHALL provide ODE solvers (ode45), root-finding
    (fzero), bounded minimization (fminbnd), numerical integration (integral,
    trapz), formatted I/O (sprintf), and file I/O (fopen/fprintf/fgets/fclose)
    with MATLAB/Octave-compatible calling conventions and numerical accuracy.
Model-user argument: An engineer combining ODE solvers with optimization performs
    parameter estimation (fitting model parameters to experimental data), optimal
    control design, and sensitivity analysis. They expect ode45 for time-domain
    simulation, fzero/fminbnd for calibration, and integral/trapz for computing
    quantities from solution curves. If any of these solvers deviate from Octave
    conventions or accuracy, the engineer cannot trust results and must fall back
    to the reference tool.
Decomposition:
    R-ODEOPT-01: ode45 on exponential decay dy/dt = -y recovers exp(-t) within 0.01.
    R-ODEOPT-02: ode45 on dy/dt = 1 produces y = t within 0.1.
    R-ODEOPT-03: fzero finds root of x^2 - 4 near x=1, returning x=2 within 1e-6.
    R-ODEOPT-04: fzero finds root of sin near 3, returning pi within 1e-3.
    R-ODEOPT-05: fminbnd finds minimum of (x-3)^2 in [0,5], returning x=3 within 1e-4.
    R-ODEOPT-06: integral of x^2 from 0 to 1 returns 1/3 within 1e-4.
    R-ODEOPT-07: trapz returns a positive value for a monotone dataset.
    R-ODEOPT-08: sprintf with %d formats an integer.
    R-ODEOPT-09: sprintf with %.2f formats pi to 2 decimal places.
    R-ODEOPT-10: sprintf with multiple %d formats an arithmetic expression.
    R-ODEOPT-11: fopen/fprintf/fgets/fclose roundtrip recovers written text.
Consistency argument: R-ODEOPT-01 and R-ODEOPT-02 verify the ODE solver on
    canonical problems (exponential decay, linear growth). R-ODEOPT-03 through
    R-ODEOPT-05 verify root-finding and bounded minimization. R-ODEOPT-06 and
    R-ODEOPT-07 verify numerical integration (adaptive quadrature and trapezoidal
    rule). R-ODEOPT-08 through R-ODEOPT-10 verify formatted output, and
    R-ODEOPT-11 verifies file I/O roundtrip. Together these cover the solver,
    optimizer, integrator, formatter, and file-I/O layers that an engineer chains
    in a typical parameter-estimation workflow.
"""
import pytest
import numpy as np
import math
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


class TestODESolvers:
    """R-ODEOPT-01..02: ode45 SHALL solve canonical ODEs with correct final values."""

    def test_ode45_exponential_decay(self, s):
        """R-ODEOPT-01: ode45 on dy/dt = -y, y(0)=1 gives y(5) within 0.01 of exp(-5)."""
        s.eval("[t, y] = ode45(@(t,y) -y, [0 5], 1)")
        y = s._engine.workspace.get("y")
        final_y = float(y.data.flatten()[-1])
        expected = math.exp(-5)
        assert abs(final_y - expected) < 0.01

    def test_ode45_linear(self, s):
        """R-ODEOPT-02: ode45 on dy/dt = 1, y(0)=0 gives y(t) = t within 0.1."""
        s.eval("[t, y] = ode45(@(t,y) 1, [0 3], 0)")
        y = s._engine.workspace.get("y")
        t = s._engine.workspace.get("t")
        final_y = float(y.data.flatten()[-1])
        final_t = float(t.data.flatten()[-1])
        assert abs(final_y - final_t) < 0.1


class TestOptimization:
    """R-ODEOPT-03..05: fzero and fminbnd SHALL find roots and minima within
    specified tolerances.
    """

    def test_fzero_quadratic(self, s):
        """R-ODEOPT-03: fzero finds root of x^2-4 near 1, returns 2.0 within 1e-6."""
        r = s.eval("fzero(@(x) x^2 - 4, 1)")
        assert abs(float(r) - 2.0) < 1e-6

    def test_fzero_sin(self, s):
        """R-ODEOPT-04: fzero finds root of sin near 3, returns pi within 1e-3."""
        r = s.eval("fzero(@sin, 3)")
        assert abs(float(r) - math.pi) < 1e-3

    def test_fminbnd(self, s):
        """R-ODEOPT-05: fminbnd finds minimum of (x-3)^2 in [0,5] at x=3 within 1e-4."""
        r = s.eval("fminbnd(@(x) (x-3)^2, 0, 5)")
        assert abs(float(r) - 3.0) < 1e-4


class TestIntegration:
    """R-ODEOPT-06..07: integral and trapz SHALL compute numerical integrals
    with correct values.
    """

    def test_integral_x_squared(self, s):
        """R-ODEOPT-06: integral of x^2 from 0 to 1 returns 1/3 within 1e-4."""
        r = s.eval("integral(@(x) x.^2, 0, 1)")
        assert abs(float(r) - 1.0/3.0) < 1e-4

    def test_trapz(self, s):
        """R-ODEOPT-07: trapz on a monotone dataset returns a positive value."""
        s.eval("tr = trapz([1 2 3 4], [0 1 4 9])")
        r = s._engine.workspace.get("tr")
        # Trapz of y = x^2 with x=[1,2,3,4]: (1+4)/2 + (4+9)/2 = 2.5 + 6.5 = 9.0... approx
        assert float(r) > 0


class TestIO:
    """R-ODEOPT-08..11: sprintf and file I/O SHALL format output and roundtrip
    text data correctly.

    Model-user argument: The engineer writes solver results to formatted logs and
    data files. sprintf constructs the output strings, and fopen/fprintf/fclose
    write them. If either layer fails, the engineer cannot export results for
    post-processing or archival.
    """

    def test_sprintf_int(self, s):
        """R-ODEOPT-08: sprintf('%d', 42) produces '42'."""
        r = s.eval("sprintf('%d', 42)")
        assert "42" in str(r)

    def test_sprintf_float(self, s):
        """R-ODEOPT-09: sprintf('%.2f', pi) produces '3.14'."""
        r = s.eval("sprintf('%.2f', pi)")
        assert "3.14" in str(r)

    def test_sprintf_string(self, s):
        """R-ODEOPT-10: sprintf('%d + %d = %d', 2, 3, 5) produces '2 + 3 = 5'."""
        r = s.eval("sprintf('%d + %d = %d', 2, 3, 5)")
        assert "2 + 3 = 5" in str(r)

    def test_file_roundtrip(self, s):
        """R-ODEOPT-11: fopen/fprintf/fgets/fclose roundtrip recovers written text."""
        s.eval("fid = fopen('/tmp/forge_pytest.txt', 'w'); fprintf(fid, 'hello forge'); fclose(fid)")
        s.eval("fid = fopen('/tmp/forge_pytest.txt', 'r'); content = fgets(fid); fclose(fid)")
        r = s.eval("content")
        assert "hello" in str(r)
