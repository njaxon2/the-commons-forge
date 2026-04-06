# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for polynomial, special, and number-theory functions (polish round 20).

V&V Traceability (backfill):
    R-POL20-01 .. R-POL20-03 (parent requirements)
    R-POL20-01-nn .. R-POL20-03-nn (unit sub-requirements)

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
    """R-POL20-01: Forge SHALL provide polynomial functions (polyval, roots,
    polyfit, polyder, polyint, conv, deconv) that evaluate, find roots, fit,
    differentiate, integrate, multiply, and divide polynomials with results
    matching MATLAB/Octave reference values.

    Model-user argument: An engineer porting control systems or signal processing
    code from Octave uses polynomial operations for transfer function
    manipulation, filter design, and curve fitting. Numerical agreement with the
    reference implementation is critical because polynomial coefficients feed
    directly into stability analysis and frequency response calculations.

    Decomposition:
        R-POL20-01-01: polyval evaluates quadratic correctly
        R-POL20-01-02: polyval evaluates constant polynomial
        R-POL20-01-03: roots finds quadratic roots
        R-POL20-01-04: polyfit recovers quadratic coefficients
        R-POL20-01-05: polyder differentiates polynomial
        R-POL20-01-06: polyint integrates without constant
        R-POL20-01-07: polyint integrates with constant
        R-POL20-01-08: conv multiplies binomials
        R-POL20-01-09: deconv divides polynomial with zero remainder

    Consistency: Sub-requirements cover evaluation (01-02), root-finding (03),
    fitting (04), calculus (05-07), and arithmetic (08-09). Together they span
    the full polynomial API surface.
    """

    def test_quadratic(self):
        """R-POL20-01-01: polyval of x^2-3x+2 at x=5 yields 12."""
        # 1*x^2 - 3*x + 2 at x=5 => 25-15+2 = 12
        p = ForgeArray(np.array([1.0, -3.0, 2.0]))
        r = _sval(forge_polyval(p, ForgeArray(5.0)))
        assert abs(r - 12.0) < 1e-12

    def test_constant(self):
        """R-POL20-01-02: polyval of constant polynomial returns that constant."""
        p = ForgeArray(np.array([7.0]))
        r = _sval(forge_polyval(p, ForgeArray(99.0)))
        assert abs(r - 7.0) < 1e-12


class TestRoots:

    def test_quadratic_roots(self):
        """R-POL20-01-03: roots of x^2-3x+2 are [1, 2]."""
        # x^2 - 3x + 2 = (x-1)(x-2) => roots 2, 1
        p = ForgeArray(np.array([1.0, -3.0, 2.0]))
        r = np.sort(_val(forge_roots(p)).ravel())
        np.testing.assert_allclose(r, [1.0, 2.0], atol=1e-12)


class TestPolyfit:

    def test_perfect_quadratic(self):
        """R-POL20-01-04: polyfit recovers [1 0 0] from y=x^2 data."""
        # fit y = x^2 exactly
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0]))
        y = ForgeArray(np.array([1.0, 4.0, 9.0, 16.0]))
        r = _val(forge_polyfit(x, y, ForgeArray(2.0))).ravel()
        np.testing.assert_allclose(r, [1.0, 0.0, 0.0], atol=1e-10)


class TestPolyder:

    def test_derivative(self):
        """R-POL20-01-05: d/dx(3x^2+2x+1) = [6, 2]."""
        # d/dx (3x^2 + 2x + 1) = 6x + 2
        p = ForgeArray(np.array([3.0, 2.0, 1.0]))
        r = _val(forge_polyder(p)).ravel()
        np.testing.assert_allclose(r, [6.0, 2.0], atol=1e-12)


class TestPolyint:

    def test_integrate_no_constant(self):
        """R-POL20-01-06: integral of [6,2] yields [3,2,0]."""
        # integral of 6x + 2 = 3x^2 + 2x + 0
        p = ForgeArray(np.array([6.0, 2.0]))
        r = _val(forge_polyint(p)).ravel()
        np.testing.assert_allclose(r, [3.0, 2.0, 0.0], atol=1e-12)

    def test_integrate_with_constant(self):
        """R-POL20-01-07: integral of [6,2] with k=1 yields [3,2,1]."""
        # integral of 6x + 2 with k=1 => 3x^2 + 2x + 1
        p = ForgeArray(np.array([6.0, 2.0]))
        r = _val(forge_polyint(p, ForgeArray(1.0))).ravel()
        np.testing.assert_allclose(r, [3.0, 2.0, 1.0], atol=1e-12)


class TestConv:

    def test_multiply_binomials(self):
        """R-POL20-01-08: conv([1,1],[1,1]) yields [1,2,1]."""
        # (x+1)(x+1) = x^2 + 2x + 1
        a = ForgeArray(np.array([1.0, 1.0]))
        b = ForgeArray(np.array([1.0, 1.0]))
        r = _val(forge_conv(a, b)).ravel()
        np.testing.assert_allclose(r, [1.0, 2.0, 1.0], atol=1e-12)


class TestDeconv:

    def test_divide_poly(self):
        """R-POL20-01-09: deconv([1,2,1],[1,1]) yields [1,1] remainder 0."""
        # (x^2 + 2x + 1) / (x + 1) = (x + 1), remainder 0
        b = ForgeArray(np.array([1.0, 2.0, 1.0]))
        a = ForgeArray(np.array([1.0, 1.0]))
        q, rem = forge_deconv(b, a)
        np.testing.assert_allclose(_val(q).ravel(), [1.0, 1.0], atol=1e-12)
        # remainder may be reduced; check all entries near zero
        assert np.all(np.abs(_val(rem).ravel()) < 1e-12)


# ── Special functions ─────────────────────────────────────────────────


class TestGamma:
    """R-POL20-02: Forge SHALL provide special mathematical functions (gamma,
    factorial, nchoosek, beta, besselj, erf, erfc, legendre) that return
    results matching known mathematical reference values.

    Model-user argument: A scientist porting statistical or physics simulation
    code from Octave uses gamma, beta, Bessel, and error functions in
    probability distributions, heat transfer models, and wave propagation.
    Even small deviations from reference values propagate into incorrect
    confidence intervals or physical predictions.

    Decomposition:
        R-POL20-02-01: gamma(5) = 24
        R-POL20-02-02: gamma(0.5) = sqrt(pi)
        R-POL20-02-03: factorial(10) = 3628800
        R-POL20-02-04: nchoosek(10,3) = 120
        R-POL20-02-05: beta(2,3) = 1/12
        R-POL20-02-06: besselj(0,0) = 1
        R-POL20-02-07: erf(0) = 0
        R-POL20-02-08: erf(inf) = 1
        R-POL20-02-09: erfc(0) = 1
        R-POL20-02-10: legendre(2, 0.5) yields P_2(0.5) = -0.125

    Consistency: Sub-requirements cover each special function with at least one
    known-value test point. Together they validate the entire special-function
    API against established mathematical identities.
    """

    def test_gamma_5(self):
        """R-POL20-02-01: gamma(5) = 4! = 24."""
        # gamma(5) = 4! = 24
        assert abs(_sval(forge_gamma(ForgeArray(5.0))) - 24.0) < 1e-10

    def test_gamma_half(self):
        """R-POL20-02-02: gamma(0.5) = sqrt(pi)."""
        # gamma(0.5) = sqrt(pi)
        assert abs(_sval(forge_gamma(ForgeArray(0.5))) - np.sqrt(np.pi)) < 1e-10


class TestFactorial:

    def test_factorial_10(self):
        """R-POL20-02-03: factorial(10) = 3628800."""
        assert abs(_sval(forge_factorial(ForgeArray(10.0))) - 3628800.0) < 1e-6


class TestNchoosek:

    def test_10_choose_3(self):
        """R-POL20-02-04: nchoosek(10,3) = 120."""
        assert abs(_sval(forge_nchoosek(ForgeArray(10.0), ForgeArray(3.0))) - 120.0) < 1e-10


class TestBeta:

    def test_beta_2_3(self):
        """R-POL20-02-05: beta(2,3) = 1/12."""
        # beta(2,3) = 1/12
        assert abs(_sval(forge_beta(ForgeArray(2.0), ForgeArray(3.0))) - 1.0/12.0) < 1e-10


class TestBesselj:

    def test_besselj_0_0(self):
        """R-POL20-02-06: J_0(0) = 1."""
        # J_0(0) = 1
        assert abs(_sval(forge_besselj(ForgeArray(0.0), ForgeArray(0.0))) - 1.0) < 1e-12


class TestErf:

    def test_erf_zero(self):
        """R-POL20-02-07: erf(0) = 0."""
        assert abs(_sval(forge_erf(ForgeArray(0.0)))) < 1e-15

    def test_erf_inf(self):
        """R-POL20-02-08: erf(inf) = 1."""
        assert abs(_sval(forge_erf(ForgeArray(float('inf')))) - 1.0) < 1e-15

    def test_erfc_zero(self):
        """R-POL20-02-09: erfc(0) = 1."""
        assert abs(_sval(forge_erfc(ForgeArray(0.0))) - 1.0) < 1e-15


class TestLegendre:

    def test_legendre_2_half(self):
        """R-POL20-02-10: P_2(0.5) = -0.125."""
        # P_2(0.5) = (3*0.25 - 1)/2 = -0.125
        r = _val(forge_legendre(ForgeArray(2.0), ForgeArray(0.5)))
        assert abs(r[0, 0] - (-0.125)) < 1e-10


# ── Number theory ─────────────────────────────────────────────────────


class TestIsprime:
    """R-POL20-03: Forge SHALL provide number-theory functions (isprime, primes,
    factor, gcd, lcm) that return correct results for standard test values.

    Model-user argument: An engineer using Octave for cryptographic prototyping
    or combinatorial algorithm development depends on isprime, factor, gcd, and
    lcm for primality testing, factorization, and modular arithmetic. Wrong
    results break security assumptions and algorithmic correctness.

    Decomposition:
        R-POL20-03-01: isprime(7) returns true
        R-POL20-03-02: isprime(4) returns false
        R-POL20-03-03: primes(20) returns [2,3,5,7,11,13,17,19]
        R-POL20-03-04: factor(60) returns [2,2,3,5]
        R-POL20-03-05: gcd(12,8) = 4
        R-POL20-03-06: lcm(4,6) = 12

    Consistency: Sub-requirements cover primality (01-02), enumeration (03),
    factorization (04), and GCD/LCM (05-06). Together they validate the full
    number-theory API.
    """

    def test_prime_7(self):
        """R-POL20-03-01: isprime(7) returns true."""
        r = _val(forge_isprime(ForgeArray(7.0))).ravel()
        assert bool(r[0]) is True

    def test_composite_4(self):
        """R-POL20-03-02: isprime(4) returns false."""
        r = _val(forge_isprime(ForgeArray(4.0))).ravel()
        assert bool(r[0]) is False


class TestPrimes:

    def test_primes_20(self):
        """R-POL20-03-03: primes(20) returns [2,3,5,7,11,13,17,19]."""
        r = _val(forge_primes(ForgeArray(20.0))).ravel()
        np.testing.assert_array_equal(r, [2, 3, 5, 7, 11, 13, 17, 19])


class TestFactor:

    def test_factor_60(self):
        """R-POL20-03-04: factor(60) returns [2,2,3,5]."""
        r = _val(forge_factor(ForgeArray(60.0))).ravel()
        np.testing.assert_array_equal(r, [2, 2, 3, 5])


class TestGcd:

    def test_gcd_12_8(self):
        """R-POL20-03-05: gcd(12,8) = 4."""
        assert abs(_sval(forge_gcd(ForgeArray(12.0), ForgeArray(8.0))) - 4.0) < 1e-12


class TestLcm:

    def test_lcm_4_6(self):
        """R-POL20-03-06: lcm(4,6) = 12."""
        r = _sval(forge_lcm(ForgeArray(4.0), ForgeArray(6.0)))
        assert abs(r - 12.0) < 1e-12
