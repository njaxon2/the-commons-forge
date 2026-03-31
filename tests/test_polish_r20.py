# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for polynomial, special, and number-theory functions (polish round 20).

SRS trace: SRS-FUNC-001, SRS-VAL-001
Test method: Comparison against known mathematical values and numpy/scipy reference.
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.polynomial import (
    forge_polyval, forge_roots, forge_polyfit, forge_polyder,
    forge_polyint, forge_conv, forge_deconv,
)
from forge.engine.builtins.specfun import (
    forge_gamma, forge_factorial, forge_nchoosek, forge_beta,
    forge_besselj, forge_erf, forge_erfc, forge_legendre,
    forge_isprime, forge_primes, forge_factor, forge_gcd, forge_lcm,
)


def _val(result):
    """Unwrap a ForgeArray to a plain numpy array."""
    return _unwrap(result)


def _sval(result):
    """Unwrap a ForgeArray to a scalar float."""
    return float(_unwrap(result).flat[0])


# ── Polynomial functions ──────────────────────────────────────────────


class TestPolyval:
    def test_quadratic(self):
        # 1*x^2 - 3*x + 2 at x=5 => 25-15+2 = 12
        p = ForgeArray(np.array([1.0, -3.0, 2.0]))
        r = _sval(forge_polyval(p, ForgeArray(5.0)))
        assert abs(r - 12.0) < 1e-12

    def test_constant(self):
        p = ForgeArray(np.array([7.0]))
        r = _sval(forge_polyval(p, ForgeArray(99.0)))
        assert abs(r - 7.0) < 1e-12


class TestRoots:
    def test_quadratic_roots(self):
        # x^2 - 3x + 2 = (x-1)(x-2) => roots 2, 1
        p = ForgeArray(np.array([1.0, -3.0, 2.0]))
        r = np.sort(_val(forge_roots(p)).ravel())
        np.testing.assert_allclose(r, [1.0, 2.0], atol=1e-12)


class TestPolyfit:
    def test_perfect_quadratic(self):
        # fit y = x^2 exactly
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0]))
        y = ForgeArray(np.array([1.0, 4.0, 9.0, 16.0]))
        r = _val(forge_polyfit(x, y, ForgeArray(2.0))).ravel()
        np.testing.assert_allclose(r, [1.0, 0.0, 0.0], atol=1e-10)


class TestPolyder:
    def test_derivative(self):
        # d/dx (3x^2 + 2x + 1) = 6x + 2
        p = ForgeArray(np.array([3.0, 2.0, 1.0]))
        r = _val(forge_polyder(p)).ravel()
        np.testing.assert_allclose(r, [6.0, 2.0], atol=1e-12)


class TestPolyint:
    def test_integrate_no_constant(self):
        # integral of 6x + 2 = 3x^2 + 2x + 0
        p = ForgeArray(np.array([6.0, 2.0]))
        r = _val(forge_polyint(p)).ravel()
        np.testing.assert_allclose(r, [3.0, 2.0, 0.0], atol=1e-12)

    def test_integrate_with_constant(self):
        # integral of 6x + 2 with k=1 => 3x^2 + 2x + 1
        p = ForgeArray(np.array([6.0, 2.0]))
        r = _val(forge_polyint(p, ForgeArray(1.0))).ravel()
        np.testing.assert_allclose(r, [3.0, 2.0, 1.0], atol=1e-12)


class TestConv:
    def test_multiply_binomials(self):
        # (x+1)(x+1) = x^2 + 2x + 1
        a = ForgeArray(np.array([1.0, 1.0]))
        b = ForgeArray(np.array([1.0, 1.0]))
        r = _val(forge_conv(a, b)).ravel()
        np.testing.assert_allclose(r, [1.0, 2.0, 1.0], atol=1e-12)


class TestDeconv:
    def test_divide_poly(self):
        # (x^2 + 2x + 1) / (x + 1) = (x + 1), remainder 0
        b = ForgeArray(np.array([1.0, 2.0, 1.0]))
        a = ForgeArray(np.array([1.0, 1.0]))
        q, rem = forge_deconv(b, a)
        np.testing.assert_allclose(_val(q).ravel(), [1.0, 1.0], atol=1e-12)
        # remainder may be reduced; check all entries near zero
        assert np.all(np.abs(_val(rem).ravel()) < 1e-12)


# ── Special functions ─────────────────────────────────────────────────


class TestGamma:
    def test_gamma_5(self):
        # gamma(5) = 4! = 24
        assert abs(_sval(forge_gamma(ForgeArray(5.0))) - 24.0) < 1e-10

    def test_gamma_half(self):
        # gamma(0.5) = sqrt(pi)
        assert abs(_sval(forge_gamma(ForgeArray(0.5))) - np.sqrt(np.pi)) < 1e-10


class TestFactorial:
    def test_factorial_10(self):
        assert abs(_sval(forge_factorial(ForgeArray(10.0))) - 3628800.0) < 1e-6


class TestNchoosek:
    def test_10_choose_3(self):
        assert abs(_sval(forge_nchoosek(ForgeArray(10.0), ForgeArray(3.0))) - 120.0) < 1e-10


class TestBeta:
    def test_beta_2_3(self):
        # beta(2,3) = 1/12
        assert abs(_sval(forge_beta(ForgeArray(2.0), ForgeArray(3.0))) - 1.0/12.0) < 1e-10


class TestBesselj:
    def test_besselj_0_0(self):
        # J_0(0) = 1
        assert abs(_sval(forge_besselj(ForgeArray(0.0), ForgeArray(0.0))) - 1.0) < 1e-12


class TestErf:
    def test_erf_zero(self):
        assert abs(_sval(forge_erf(ForgeArray(0.0)))) < 1e-15

    def test_erf_inf(self):
        assert abs(_sval(forge_erf(ForgeArray(float('inf')))) - 1.0) < 1e-15

    def test_erfc_zero(self):
        assert abs(_sval(forge_erfc(ForgeArray(0.0))) - 1.0) < 1e-15


class TestLegendre:
    def test_legendre_2_half(self):
        # P_2(0.5) = (3*0.25 - 1)/2 = -0.125
        r = _val(forge_legendre(ForgeArray(2.0), ForgeArray(0.5)))
        assert abs(r[0, 0] - (-0.125)) < 1e-10


# ── Number theory ─────────────────────────────────────────────────────


class TestIsprime:
    def test_prime_7(self):
        r = _val(forge_isprime(ForgeArray(7.0))).ravel()
        assert bool(r[0]) is True

    def test_composite_4(self):
        r = _val(forge_isprime(ForgeArray(4.0))).ravel()
        assert bool(r[0]) is False


class TestPrimes:
    def test_primes_20(self):
        r = _val(forge_primes(ForgeArray(20.0))).ravel()
        np.testing.assert_array_equal(r, [2, 3, 5, 7, 11, 13, 17, 19])


class TestFactor:
    def test_factor_60(self):
        r = _val(forge_factor(ForgeArray(60.0))).ravel()
        np.testing.assert_array_equal(r, [2, 2, 3, 5])


class TestGcd:
    def test_gcd_12_8(self):
        assert abs(_sval(forge_gcd(ForgeArray(12.0), ForgeArray(8.0))) - 4.0) < 1e-12


class TestLcm:
    def test_lcm_4_6(self):
        r = _sval(forge_lcm(ForgeArray(4.0), ForgeArray(6.0)))
        assert abs(r - 12.0) < 1e-12
