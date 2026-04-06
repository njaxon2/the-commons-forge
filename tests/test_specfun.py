"""V&V tests for specfun toolbox (21 functions).

SRS trace: SRS-FUNC-001, SRS-VAL-001
Test method: Cross-reference with scipy.special and known identities.
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.specfun import *


class TestBetaFunctions:
    """R-SPEC-01: Beta function family SHALL return values consistent with
    the identity beta(a,b) = gamma(a)*gamma(b)/gamma(a+b), with symmetry
    beta(a,b) == beta(b,a), and the regularized incomplete beta function
    SHALL satisfy boundary conditions betainc(0,a,b)=0, betainc(1,a,b)=1,
    with betaincinv roundtripping to the original argument within 1e-10.

    Model-user argument: A scientist building Bayesian inference pipelines
    relies on beta functions for conjugate prior calculations. Incorrect beta
    values propagate silently into posterior distributions, producing
    misleading credible intervals. Symmetry and boundary correctness are
    non-negotiable for any statistical computing environment.

    Decomposition:
        R-SPEC-01.1: beta(a,b) matches gamma-based identity within 1e-12
        R-SPEC-01.2: beta(a,b) == beta(b,a) within 1e-15
        R-SPEC-01.3: betaln(a,b) == log(beta(a,b)) within 1e-12
        R-SPEC-01.4: betainc boundary values at x=0 and x=1
        R-SPEC-01.5: betaincinv(betainc(x,a,b),a,b) recovers x within 1e-10

    Consistency: R-SPEC-01.1 validates the core computation against the
    canonical identity. R-SPEC-01.2 confirms the algebraic symmetry property.
    R-SPEC-01.3 verifies the log-domain variant is consistent with the
    linear-domain function. R-SPEC-01.4 pins the regularized incomplete beta
    at its two boundary values. R-SPEC-01.5 closes the loop by verifying that
    the inverse function recovers the original argument. Together these five
    sub-requirements cover value accuracy, symmetry, log-domain consistency,
    boundary behavior, and inverse roundtrip for the full beta family.
    """

    def test_beta_identity(self):
        """R-SPEC-01.1: beta(a,b) matches gamma identity within 1e-12."""
        from scipy.special import gamma
        a, b = 3.0, 4.0
        expected = gamma(a) * gamma(b) / gamma(a + b)
        r = _unwrap(forge_beta(ForgeArray(a), ForgeArray(b)))
        assert abs(r - expected) < 1e-12

    def test_beta_symmetric(self):
        """R-SPEC-01.2: beta(a,b) == beta(b,a) within 1e-15."""
        r1 = _unwrap(forge_beta(ForgeArray(2.0), ForgeArray(5.0)))
        r2 = _unwrap(forge_beta(ForgeArray(5.0), ForgeArray(2.0)))
        assert abs(r1 - r2) < 1e-15

    def test_betaln(self):
        """R-SPEC-01.3: betaln(a,b) == log(beta(a,b)) within 1e-12."""
        a, b = ForgeArray(3.0), ForgeArray(4.0)
        r = _unwrap(forge_betaln(a, b))
        expected = np.log(_unwrap(forge_beta(a, b)))
        assert abs(r - expected) < 1e-12

    def test_betainc_bounds(self):
        """R-SPEC-01.4: betainc(0,a,b)=0 and betainc(1,a,b)=1."""
        a, b = ForgeArray(2.0), ForgeArray(3.0)
        assert abs(_unwrap(forge_betainc(ForgeArray(0.0), a, b))) < 1e-15
        assert abs(_unwrap(forge_betainc(ForgeArray(1.0), a, b)) - 1.0) < 1e-15

    def test_betaincinv_roundtrip(self):
        """R-SPEC-01.5: betaincinv(betainc(x,a,b),a,b) recovers x within 1e-10."""
        a, b = ForgeArray(2.0), ForgeArray(3.0)
        x = ForgeArray(0.4)
        y = forge_betainc(x, a, b)
        x2 = forge_betaincinv(y, a, b)
        assert abs(_unwrap(x2) - 0.4) < 1e-10


class TestFactorialAndCombinatorics:
    """R-SPEC-02: factorial SHALL return n! exactly for non-negative integers,
    and nchoosek SHALL return binomial coefficients satisfying C(n,k) == C(n,n-k).

    Model-user argument: An engineer designing combinatorial experiments (e.g.,
    antenna array configurations or DOE layouts) uses factorial and nchoosek to
    enumerate feasible designs. Off-by-one errors in these functions would yield
    incorrect design counts, potentially omitting valid configurations or
    over-allocating resources.

    Decomposition:
        R-SPEC-02.1: factorial(5) == 120
        R-SPEC-02.2: factorial(0) == 1
        R-SPEC-02.3: nchoosek(10,3) == 120
        R-SPEC-02.4: nchoosek(n,k) == nchoosek(n,n-k) (symmetry)

    Consistency: R-SPEC-02.1 validates the general recursive case. R-SPEC-02.2
    confirms the base case 0! = 1. R-SPEC-02.3 checks a known binomial
    coefficient. R-SPEC-02.4 verifies the symmetry identity C(n,k) = C(n,n-k).
    These four sub-requirements cover both functions across base cases, general
    cases, and algebraic identities.
    """

    def test_factorial_5(self):
        """R-SPEC-02.1: factorial(5) returns 120."""
        assert abs(_unwrap(forge_factorial(ForgeArray(5.0))) - 120.0) < 1e-10

    def test_factorial_0(self):
        """R-SPEC-02.2: factorial(0) returns 1."""
        assert abs(_unwrap(forge_factorial(ForgeArray(0.0))) - 1.0) < 1e-10

    def test_nchoosek_10_3(self):
        """R-SPEC-02.3: nchoosek(10,3) returns 120."""
        assert abs(_unwrap(forge_nchoosek(ForgeArray(10.0), ForgeArray(3.0))) - 120.0) < 1e-10

    def test_nchoosek_symmetry(self):
        """R-SPEC-02.4: nchoosek(10,3) == nchoosek(10,7)."""
        r1 = _unwrap(forge_nchoosek(ForgeArray(10.0), ForgeArray(3.0)))
        r2 = _unwrap(forge_nchoosek(ForgeArray(10.0), ForgeArray(7.0)))
        assert abs(r1 - r2) < 1e-10


class TestPrimality:
    """R-SPEC-03: isprime SHALL correctly classify integers as prime or
    composite, primes SHALL return all primes up to a given limit, factor
    SHALL return the complete prime factorization, and lcm SHALL return the
    least common multiple of two integers.

    Model-user argument: A researcher in coding theory or lightweight
    cryptography prototypes uses primality testing and factorization to
    validate field sizes for error-correcting codes or to verify that
    chosen moduli have the expected prime structure. Incorrect classification
    or incomplete factorizations would produce invalid algebraic structures,
    breaking the mathematical guarantees the code relies on.

    Decomposition:
        R-SPEC-03.1: isprime classifies [1..7] correctly
        R-SPEC-03.2: isprime identifies 997 as prime
        R-SPEC-03.3: primes(10) returns [2, 3, 5, 7]
        R-SPEC-03.4: primes(2) returns [2]
        R-SPEC-03.5: factor(12) returns [2, 2, 3]
        R-SPEC-03.6: factor(13) returns [13]
        R-SPEC-03.7: lcm(12, 18) returns 36

    Consistency: R-SPEC-03.1 and R-SPEC-03.2 cover isprime on a small vector
    and a larger known prime respectively. R-SPEC-03.3 and R-SPEC-03.4 verify
    the primes sieve at a general limit and the minimal edge case. R-SPEC-03.5
    and R-SPEC-03.6 test factor on a composite and a prime. R-SPEC-03.7 checks
    lcm. Together these cover all four primality/factorization functions across
    both typical and edge-case inputs.
    """

    def test_isprime_small(self):
        """R-SPEC-03.1: isprime classifies [1..7] as [F,T,T,F,T,F,T]."""
        r = _unwrap(forge_isprime(ForgeArray(np.array([1, 2, 3, 4, 5, 6, 7])))).ravel()
        np.testing.assert_array_equal(r, [False, True, True, False, True, False, True])

    def test_isprime_large(self):
        """R-SPEC-03.2: isprime(997) returns True."""
        assert _unwrap(forge_isprime(ForgeArray(997)))[()]  == True

    def test_primes_10(self):
        """R-SPEC-03.3: primes(10) returns [2, 3, 5, 7]."""
        r = _unwrap(forge_primes(ForgeArray(10.0))).ravel()
        np.testing.assert_array_equal(r, [2, 3, 5, 7])

    def test_primes_2(self):
        """R-SPEC-03.4: primes(2) returns [2]."""
        r = _unwrap(forge_primes(ForgeArray(2.0))).ravel()
        np.testing.assert_array_equal(r, [2])

    def test_factor_12(self):
        """R-SPEC-03.5: factor(12) returns [2, 2, 3]."""
        r = _unwrap(forge_factor(ForgeArray(12.0))).ravel()
        np.testing.assert_array_equal(r, [2, 2, 3])

    def test_factor_prime(self):
        """R-SPEC-03.6: factor(13) returns [13]."""
        r = _unwrap(forge_factor(ForgeArray(13.0))).ravel()
        np.testing.assert_array_equal(r, [13])

    def test_lcm(self):
        """R-SPEC-03.7: lcm(12, 18) returns 36."""
        r = _unwrap(forge_lcm(ForgeArray(12.0), ForgeArray(18.0)))
        assert int(r.flat[0]) == 36


class TestSpecialIntegrals:
    """R-SPEC-04: sinint SHALL return zero at x=0, cosint SHALL match
    scipy.special.sici reference values, and ellipke SHALL return K(0)=pi/2
    and E(0)=pi/2.

    Model-user argument: An electrical engineer computing electromagnetic
    field distributions uses sine/cosine integrals for antenna radiation
    patterns and elliptic integrals for inductance calculations in coil
    geometries. Errors at boundary values (x=0, m=0) are especially dangerous
    because these represent physically meaningful limiting cases (e.g., zero
    eccentricity, zero argument) that are often used as sanity checks.

    Decomposition:
        R-SPEC-04.1: sinint(0) == 0
        R-SPEC-04.2: cosint(1) matches scipy reference
        R-SPEC-04.3: ellipke(0) returns K=pi/2, E=pi/2

    Consistency: R-SPEC-04.1 pins sinint at its zero crossing. R-SPEC-04.2
    validates cosint against an independent reference implementation at a
    non-trivial argument. R-SPEC-04.3 verifies both outputs of ellipke at the
    known boundary m=0. Together these three sub-requirements cover all three
    special integral functions at their most critical reference points.
    """

    def test_sinint_0(self):
        """R-SPEC-04.1: sinint(0) returns 0."""
        assert abs(_unwrap(forge_sinint(ForgeArray(0.0)))) < 1e-15

    def test_cosint_known(self):
        """R-SPEC-04.2: cosint(1) matches scipy.special.sici reference."""
        from scipy.special import sici
        expected = sici(1.0)[1]
        r = _unwrap(forge_cosint(ForgeArray(1.0)))
        assert abs(r - expected) < 1e-12

    def test_ellipke_0(self):
        """R-SPEC-04.3: ellipke(0) returns K=pi/2, E=pi/2."""
        K, E = forge_ellipke(ForgeArray(0.0))
        assert abs(_unwrap(K) - np.pi/2) < 1e-14
        assert abs(_unwrap(E) - np.pi/2) < 1e-14


class TestRealFunctions:
    """R-SPEC-05: nthroot SHALL return real n-th roots (including negative
    bases with odd roots), reallog SHALL return the natural log for positive
    arguments and raise ValueError for negative arguments, and realsqrt SHALL
    return the square root for non-negative arguments and raise ValueError
    for negative arguments.

    Model-user argument: A physicist or mechanical engineer computing
    real-valued quantities (temperatures, pressures, stresses) needs
    guaranteed real results. The standard complex-valued cube root of -8
    returns a complex principal root, which is physically meaningless for a
    quantity like thermal conductivity. These functions enforce real-domain
    semantics so that invalid inputs fail loudly rather than silently
    producing complex artifacts.

    Decomposition:
        R-SPEC-05.1: nthroot(-8, 3) == -2
        R-SPEC-05.2: nthroot(27, 3) == 3
        R-SPEC-05.3: reallog(e) == 1
        R-SPEC-05.4: reallog(-1) raises ValueError
        R-SPEC-05.5: realsqrt(4) == 2
        R-SPEC-05.6: realsqrt(-1) raises ValueError

    Consistency: R-SPEC-05.1 tests the critical negative-base odd-root case
    that distinguishes nthroot from the power operator. R-SPEC-05.2 confirms
    the positive-base case. R-SPEC-05.3 and R-SPEC-05.4 verify reallog for
    a valid and an invalid input respectively. R-SPEC-05.5 and R-SPEC-05.6
    do the same for realsqrt. All six sub-requirements together confirm that
    each real-domain function produces correct values for valid inputs and
    raises errors for invalid inputs.
    """

    def test_nthroot_cube(self):
        """R-SPEC-05.1: nthroot(-8, 3) returns -2."""
        r = _unwrap(forge_nthroot(ForgeArray(-8.0), ForgeArray(3.0)))
        assert abs(r - (-2.0)) < 1e-14

    def test_nthroot_positive(self):
        """R-SPEC-05.2: nthroot(27, 3) returns 3."""
        r = _unwrap(forge_nthroot(ForgeArray(27.0), ForgeArray(3.0)))
        assert abs(r - 3.0) < 1e-14

    def test_reallog_positive(self):
        """R-SPEC-05.3: reallog(e) returns 1."""
        r = _unwrap(forge_reallog(ForgeArray(np.e)))
        assert abs(r - 1.0) < 1e-14

    def test_reallog_negative_raises(self):
        """R-SPEC-05.4: reallog(-1) raises ValueError."""
        with pytest.raises(ValueError):
            forge_reallog(ForgeArray(-1.0))

    def test_realsqrt_positive(self):
        """R-SPEC-05.5: realsqrt(4) returns 2."""
        r = _unwrap(forge_realsqrt(ForgeArray(4.0)))
        assert abs(r - 2.0) < 1e-14

    def test_realsqrt_negative_raises(self):
        """R-SPEC-05.6: realsqrt(-1) raises ValueError."""
        with pytest.raises(ValueError):
            forge_realsqrt(ForgeArray(-1.0))


class TestGammaFunctions:
    """R-SPEC-06: The regularized lower incomplete gamma function gammainc
    SHALL return 0 at x=0 for any positive a, and gammaincinv SHALL invert
    gammainc so that gammaincinv(gammainc(x,a),a) recovers x within 1e-8.

    Model-user argument: A statistician fitting chi-squared or Poisson
    models evaluates the regularized gamma function as the CDF of the gamma
    distribution. If gammainc(0,a) is not exactly zero, normalization of
    probability distributions breaks down. The inverse function is essential
    for computing quantiles (percentile thresholds) used in hypothesis testing.

    Decomposition:
        R-SPEC-06.1: gammainc(0, a) == 0
        R-SPEC-06.2: gammaincinv(gammainc(x, a), a) recovers x within 1e-8

    Consistency: R-SPEC-06.1 pins the lower incomplete gamma at its zero
    boundary. R-SPEC-06.2 verifies the inverse roundtrip. Together these
    two sub-requirements confirm both the forward and inverse functions are
    correct and mutually consistent.
    """

    def test_gammainc_bounds(self):
        """R-SPEC-06.1: gammainc(0, a) returns 0."""
        r = _unwrap(forge_gammainc(ForgeArray(0.0), ForgeArray(1.0)))
        assert abs(r) < 1e-15

    def test_gammaincinv_roundtrip(self):
        """R-SPEC-06.2: gammaincinv(gammainc(x, a), a) recovers x within 1e-8."""
        a = ForgeArray(2.0)
        x = ForgeArray(1.5)
        y = forge_gammainc(x, a)
        x2 = forge_gammaincinv(y, a)
        assert abs(_unwrap(x2) - 1.5) < 1e-8


class TestLegendre:
    """R-SPEC-07: The Legendre polynomial function SHALL return P_0(x) = 1
    for all x (degree 0) and P_1(x) = x (degree 1).

    Model-user argument: A geophysicist performing spherical harmonic analysis
    of gravitational or magnetic field data expands measurements into Legendre
    polynomial series. P_0 and P_1 are the DC offset and linear tilt terms
    respectively. If these base cases are wrong, every higher-degree coefficient
    computed by recurrence will inherit the error, corrupting the entire
    harmonic decomposition.

    Decomposition:
        R-SPEC-07.1: legendre(0, x) returns 1 for any x
        R-SPEC-07.2: legendre(1, x) returns x

    Consistency: R-SPEC-07.1 verifies the degree-0 base case. R-SPEC-07.2
    verifies the degree-1 base case. Since all higher-degree Legendre
    polynomials are built by three-term recurrence from these two, correctness
    of both base cases is sufficient to establish the foundation of the
    recurrence.
    """

    def test_legendre_0_1(self):
        """R-SPEC-07.1: P_0(0.5) returns 1."""
        r = _unwrap(forge_legendre(ForgeArray(0.0), ForgeArray(0.5)))
        assert abs(r[0, 0] - 1.0) < 1e-14

    def test_legendre_1(self):
        """R-SPEC-07.2: P_1(0.7) returns 0.7."""
        r = _unwrap(forge_legendre(ForgeArray(1.0), ForgeArray(0.7)))
        assert abs(r[0, 0] - 0.7) < 1e-14
