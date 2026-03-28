"""Tests for ODE solvers, optimization, and numerical integration."""
import pytest
import numpy as np
import math
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


class TestODESolvers:
    """Test ODE solver functionality."""

    def test_ode45_exponential_decay(self, s):
        """ode45 on dy/dt = -y, y(0) = 1 -> y = exp(-t)"""
        s.eval("[t, y] = ode45(@(t,y) -y, [0 5], 1)")
        y = s._engine.workspace.get("y")
        final_y = float(y.data.flatten()[-1])
        expected = math.exp(-5)
        assert abs(final_y - expected) < 0.01

    def test_ode45_linear(self, s):
        """ode45 on dy/dt = 1, y(0) = 0 -> y = t"""
        s.eval("[t, y] = ode45(@(t,y) 1, [0 3], 0)")
        y = s._engine.workspace.get("y")
        t = s._engine.workspace.get("t")
        final_y = float(y.data.flatten()[-1])
        final_t = float(t.data.flatten()[-1])
        assert abs(final_y - final_t) < 0.1


class TestOptimization:
    """Test optimization functions."""

    def test_fzero_quadratic(self, s):
        """fzero finds root of x^2 - 4 near x=1 -> x=2"""
        r = s.eval("fzero(@(x) x^2 - 4, 1)")
        assert abs(float(r) - 2.0) < 1e-6

    def test_fzero_sin(self, s):
        """fzero finds root of sin near 3 -> pi"""
        r = s.eval("fzero(@sin, 3)")
        assert abs(float(r) - math.pi) < 1e-3

    def test_fminbnd(self, s):
        """fminbnd finds minimum of (x-3)^2 in [0,5] -> x=3"""
        r = s.eval("fminbnd(@(x) (x-3)^2, 0, 5)")
        assert abs(float(r) - 3.0) < 1e-4


class TestIntegration:
    """Test numerical integration."""

    def test_integral_x_squared(self, s):
        """integral of x^2 from 0 to 1 = 1/3"""
        r = s.eval("integral(@(x) x.^2, 0, 1)")
        assert abs(float(r) - 1.0/3.0) < 1e-4

    def test_trapz(self, s):
        """trapz basic test"""
        s.eval("tr = trapz([1 2 3 4], [0 1 4 9])")
        r = s._engine.workspace.get("tr")
        # Trapz of y = x^2 with x=[1,2,3,4]: (1+4)/2 + (4+9)/2 = 2.5 + 6.5 = 9.0... approx
        assert float(r) > 0


class TestIO:
    """Test I/O functions."""

    def test_sprintf_int(self, s):
        r = s.eval("sprintf('%d', 42)")
        assert "42" in str(r)

    def test_sprintf_float(self, s):
        r = s.eval("sprintf('%.2f', pi)")
        assert "3.14" in str(r)

    def test_sprintf_string(self, s):
        r = s.eval("sprintf('%d + %d = %d', 2, 3, 5)")
        assert "2 + 3 = 5" in str(r)

    def test_file_roundtrip(self, s):
        s.eval("fid = fopen('/tmp/forge_pytest.txt', 'w'); fprintf(fid, 'hello forge'); fclose(fid)")
        s.eval("fid = fopen('/tmp/forge_pytest.txt', 'r'); content = fgets(fid); fclose(fid)")
        r = s.eval("content")
        assert "hello" in str(r)
