# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for specfun toolbox (21 functions).

SRS trace: SRS-FUNC-001, SRS-VAL-001
Test method: Cross-reference with scipy.special and known identities.
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.specfun import *


class TestBetaFunctions:

    def test_beta_identity(self):
        """beta(a,b) = gamma(a)*gamma(b)/gamma(a+b)."""
        from scipy.special import gamma
        a, b = 3.0, 4.0
        expected = gamma(a) * gamma(b) / gamma(a + b)
        r = _unwrap(forge_beta(ForgeArray(a), ForgeArray(b)))
        assert abs(r - expected) < 1e-12

    def test_beta_symmetric(self):
        """beta(a,b) == beta(b,a)."""
        r1 = _unwrap(forge_beta(ForgeArray(2.0), ForgeArray(5.0)))
        r2 = _unwrap(forge_beta(ForgeArray(5.0), ForgeArray(2.0)))
        assert abs(r1 - r2) < 1e-15

    def test_betaln(self):
        """betaln = log(beta)."""
        a, b = ForgeArray(3.0), ForgeArray(4.0)
        r = _unwrap(forge_betaln(a, b))
        expected = np.log(_unwrap(forge_beta(a, b)))
        assert abs(r - expected) < 1e-12

    def test_betainc_bounds(self):
        """betainc(0,a,b) = 0, betainc(1,a,b) = 1."""
        a, b = ForgeArray(2.0), ForgeArray(3.0)
        assert abs(_unwrap(forge_betainc(ForgeArray(0.0), a, b))) < 1e-15
        assert abs(_unwrap(forge_betainc(ForgeArray(1.0), a, b)) - 1.0) < 1e-15

    def test_betaincinv_roundtrip(self):
        a, b = ForgeArray(2.0), ForgeArray(3.0)
        x = ForgeArray(0.4)
        y = forge_betainc(x, a, b)
        x2 = forge_betaincinv(y, a, b)
        assert abs(_unwrap(x2) - 0.4) < 1e-10


class TestFactorialAndCombinatorics:

    def test_factorial_5(self):
        assert abs(_unwrap(forge_factorial(ForgeArray(5.0))) - 120.0) < 1e-10

    def test_factorial_0(self):
        assert abs(_unwrap(forge_factorial(ForgeArray(0.0))) - 1.0) < 1e-10

    def test_nchoosek_10_3(self):
        assert abs(_unwrap(forge_nchoosek(ForgeArray(10.0), ForgeArray(3.0))) - 120.0) < 1e-10

    def test_nchoosek_symmetry(self):
        r1 = _unwrap(forge_nchoosek(ForgeArray(10.0), ForgeArray(3.0)))
        r2 = _unwrap(forge_nchoosek(ForgeArray(10.0), ForgeArray(7.0)))
        assert abs(r1 - r2) < 1e-10


class TestPrimality:

    def test_isprime_small(self):
        r = _unwrap(forge_isprime(ForgeArray(np.array([1, 2, 3, 4, 5, 6, 7])))).ravel()
        np.testing.assert_array_equal(r, [False, True, True, False, True, False, True])

    def test_isprime_large(self):
        assert _unwrap(forge_isprime(ForgeArray(997)))[()]  == True

    def test_primes_10(self):
        r = _unwrap(forge_primes(ForgeArray(10.0))).ravel()
        np.testing.assert_array_equal(r, [2, 3, 5, 7])

    def test_primes_2(self):
        r = _unwrap(forge_primes(ForgeArray(2.0))).ravel()
        np.testing.assert_array_equal(r, [2])

    def test_factor_12(self):
        r = _unwrap(forge_factor(ForgeArray(12.0))).ravel()
        np.testing.assert_array_equal(r, [2, 2, 3])

    def test_factor_prime(self):
        r = _unwrap(forge_factor(ForgeArray(13.0))).ravel()
        np.testing.assert_array_equal(r, [13])

    def test_lcm(self):
        r = _unwrap(forge_lcm(ForgeArray(12.0), ForgeArray(18.0)))
        assert int(r.flat[0]) == 36


class TestSpecialIntegrals:

    def test_sinint_0(self):
        assert abs(_unwrap(forge_sinint(ForgeArray(0.0)))) < 1e-15

    def test_cosint_known(self):
        """Ci(1) is a known constant."""
        from scipy.special import sici
        expected = sici(1.0)[1]
        r = _unwrap(forge_cosint(ForgeArray(1.0)))
        assert abs(r - expected) < 1e-12

    def test_ellipke_0(self):
        """K(0) = pi/2, E(0) = pi/2."""
        K, E = forge_ellipke(ForgeArray(0.0))
        assert abs(_unwrap(K) - np.pi/2) < 1e-14
        assert abs(_unwrap(E) - np.pi/2) < 1e-14


class TestRealFunctions:

    def test_nthroot_cube(self):
        """nthroot(-8, 3) = -2."""
        r = _unwrap(forge_nthroot(ForgeArray(-8.0), ForgeArray(3.0)))
        assert abs(r - (-2.0)) < 1e-14

    def test_nthroot_positive(self):
        r = _unwrap(forge_nthroot(ForgeArray(27.0), ForgeArray(3.0)))
        assert abs(r - 3.0) < 1e-14

    def test_reallog_positive(self):
        r = _unwrap(forge_reallog(ForgeArray(np.e)))
        assert abs(r - 1.0) < 1e-14

    def test_reallog_negative_raises(self):
        with pytest.raises(ValueError):
            forge_reallog(ForgeArray(-1.0))

    def test_realsqrt_positive(self):
        r = _unwrap(forge_realsqrt(ForgeArray(4.0)))
        assert abs(r - 2.0) < 1e-14

    def test_realsqrt_negative_raises(self):
        with pytest.raises(ValueError):
            forge_realsqrt(ForgeArray(-1.0))


class TestGammaFunctions:

    def test_gammainc_bounds(self):
        """gammainc(0, a) = 0."""
        r = _unwrap(forge_gammainc(ForgeArray(0.0), ForgeArray(1.0)))
        assert abs(r) < 1e-15

    def test_gammaincinv_roundtrip(self):
        a = ForgeArray(2.0)
        x = ForgeArray(1.5)
        y = forge_gammainc(x, a)
        x2 = forge_gammaincinv(y, a)
        assert abs(_unwrap(x2) - 1.5) < 1e-8


class TestLegendre:

    def test_legendre_0_1(self):
        """P_0(x) = 1 for all x."""
        r = _unwrap(forge_legendre(ForgeArray(0.0), ForgeArray(0.5)))
        assert abs(r[0, 0] - 1.0) < 1e-14

    def test_legendre_1(self):
        """P_1(x) = x."""
        r = _unwrap(forge_legendre(ForgeArray(1.0), ForgeArray(0.7)))
        assert abs(r[0, 0] - 0.7) < 1e-14
