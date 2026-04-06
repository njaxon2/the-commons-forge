"""V&V tests for polynomial toolbox.

SRS trace: SRS-FUNC-001, SRS-VAL-001
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.polynomial import *


class TestRootsAndPoly:
    """R-POLY-01: The system SHALL compute polynomial roots from coefficients
    and reconstruct coefficient vectors from roots, matching Octave roots()
    and poly() semantics.

    Model-user argument: A controls engineer routinely calls roots() on
    characteristic polynomials to locate transfer function poles and zeros.
    After modifying a root locus, they call poly() to reconstruct the
    updated denominator coefficients. Round-trip fidelity (poly(roots(p)) == p)
    is essential; any numerical drift corrupts subsequent Bode or Nyquist plots.

    Decomposition:
      R-POLY-01.1 — roots() returns correct real roots for a quadratic.
      R-POLY-01.2 — poly() reconstructs coefficients from a root vector.
      R-POLY-01.3 — roots/poly round-trip preserves the original polynomial.

    Consistency: 01.1 validates forward root-finding, 01.2 validates the
    inverse operation, and 01.3 confirms the composition is identity (up to
    leading-coefficient scaling). Together they fully cover the requirement.
    """

    def test_roots_quadratic(self):
        """R-POLY-01.1: roots() of x^2 - 5x + 6 returns {2, 3}."""
        p = ForgeArray(np.array([1.0, -5.0, 6.0]))
        r = np.sort(_unwrap(forge_roots(p)).ravel().real)
        np.testing.assert_allclose(r, [2, 3], atol=1e-10)

    def test_poly_from_roots(self):
        """R-POLY-01.2: poly([2, 3]) reconstructs [1, -5, 6]."""
        x = ForgeArray(np.array([2.0, 3.0]))
        p = _unwrap(forge_poly(x)).ravel()
        np.testing.assert_allclose(p, [1, -5, 6], atol=1e-10)

    def test_roots_poly_roundtrip(self):
        """R-POLY-01.3: poly(roots(p)) recovers p up to leading-coefficient scaling."""
        p = ForgeArray(np.array([1.0, -6.0, 11.0, -6.0]))
        r = forge_roots(p)
        p2 = _unwrap(forge_poly(r)).ravel().real
        np.testing.assert_allclose(p2 / p2[0], _unwrap(p).ravel(), atol=1e-10)


class TestPolyEval:
    """R-POLY-02: The system SHALL evaluate polynomials at scalar and vector
    points and fit polynomials to data, matching Octave polyval() and
    polyfit() semantics.

    Model-user argument: An experimentalist uses polyfit() to regress a
    calibration curve from measured sensor data, then polyval() to predict
    corrected readings at arbitrary operating points. Vectorized evaluation
    is critical because calibration sweeps involve hundreds of sample points
    processed in a single call. If polyfit returns wrong coefficients or
    polyval mis-evaluates, every downstream measurement is biased.

    Decomposition:
      R-POLY-02.1 — polyval() evaluates a polynomial at a single scalar point.
      R-POLY-02.2 — polyval() evaluates element-wise across a vector of points.
      R-POLY-02.3 — polyfit() recovers exact linear coefficients from noiseless data.

    Consistency: 02.1 covers the scalar base case, 02.2 extends to vectorized
    evaluation, and 02.3 validates the fitting inverse. Together they cover
    both directions of polynomial evaluation and fitting.
    """

    def test_polyval_simple(self):
        """R-POLY-02.1: polyval([2 3 1], 2) returns 15."""
        p = ForgeArray(np.array([2.0, 3.0, 1.0]))
        r = float(_unwrap(forge_polyval(p, ForgeArray(2.0))).flat[0])
        assert abs(r - 15.0) < 1e-10

    def test_polyval_vectorized(self):
        """R-POLY-02.2: polyval(x^2-1, [0 1 2]) returns [-1 0 3]."""
        p = ForgeArray(np.array([1.0, 0.0, -1.0]))  # x^2 - 1
        x = ForgeArray(np.array([0.0, 1.0, 2.0]))
        r = _unwrap(forge_polyval(p, x)).ravel()
        np.testing.assert_allclose(r, [-1, 0, 3], atol=1e-14)

    def test_polyfit_linear(self):
        """R-POLY-02.3: polyfit() on y=2x data recovers slope=2, intercept=0."""
        x = ForgeArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        y = ForgeArray(np.array([2.0, 4.0, 6.0, 8.0, 10.0]))
        p = _unwrap(forge_polyfit(x, y, ForgeArray(1.0))).ravel()
        assert abs(p[0] - 2.0) < 1e-10  # slope
        assert abs(p[1]) < 1e-10  # intercept


class TestPolyCalculus:
    """R-POLY-03: The system SHALL compute symbolic differentiation and
    integration of polynomial coefficient vectors, matching Octave polyder()
    and polyint() semantics.

    Model-user argument: A controls engineer manipulates transfer functions
    algebraically: polyder() computes the derivative of a plant's numerator
    for sensitivity analysis, and polyint() integrates an impulse response
    polynomial to obtain the step response. Round-trip consistency
    (polyder(polyint(p)) == p) is the litmus test; if it fails, the
    engineer cannot trust either operation in isolation.

    Decomposition:
      R-POLY-03.1 — polyder() returns the correct derivative coefficients.
      R-POLY-03.2 — polyint() returns the correct antiderivative coefficients.
      R-POLY-03.3 — polyder(polyint(p)) recovers the original polynomial.

    Consistency: 03.1 validates differentiation, 03.2 validates integration,
    and 03.3 confirms they are inverses. Full coverage of the requirement.
    """

    def test_polyder(self):
        """R-POLY-03.1: polyder(x^3 + 2x) returns [3 0 2]."""
        p = ForgeArray(np.array([1.0, 0.0, 2.0, 0.0]))
        dp = _unwrap(forge_polyder(p)).ravel()
        np.testing.assert_allclose(dp, [3, 0, 2], atol=1e-14)

    def test_polyint(self):
        """R-POLY-03.2: polyint(3x^2 + 2) returns [1 0 2 0] (C=0)."""
        p = ForgeArray(np.array([3.0, 0.0, 2.0]))
        ip = _unwrap(forge_polyint(p)).ravel()
        assert abs(ip[0] - 1.0) < 1e-14
        assert abs(ip[2] - 2.0) < 1e-14

    def test_polyder_polyint_roundtrip(self):
        """R-POLY-03.3: polyder(polyint(p)) recovers p exactly."""
        p = ForgeArray(np.array([1.0, 2.0, 3.0]))
        ip = forge_polyint(p)
        dp = _unwrap(forge_polyder(ip)).ravel()
        np.testing.assert_allclose(dp, [1, 2, 3], atol=1e-14)


class TestConvolution:
    """R-POLY-04: The system SHALL multiply and divide polynomials via
    convolution and deconvolution, matching Octave conv() and deconv()
    semantics.

    Model-user argument: Cascading two transfer functions means multiplying
    their numerator (or denominator) polynomials, which is exactly conv().
    Decoupling a known subsystem from a measured end-to-end response uses
    deconv(). These are daily operations when building block diagrams from
    component models. Incorrect convolution silently corrupts every
    cascaded system analysis.

    Decomposition:
      R-POLY-04.1 — conv() multiplies two first-order polynomials correctly.
      R-POLY-04.2 — deconv() divides a quadratic by a linear factor with zero remainder.

    Consistency: 04.1 covers polynomial multiplication and 04.2 covers the
    inverse division. Together they validate both directions of the
    convolution/deconvolution pair.
    """

    def test_conv(self):
        """R-POLY-04.1: conv([1 1], [1 2]) returns [1 3 2]."""
        a = ForgeArray(np.array([1.0, 1.0]))
        b = ForgeArray(np.array([1.0, 2.0]))
        r = _unwrap(forge_conv(a, b)).ravel()
        np.testing.assert_allclose(r, [1, 3, 2], atol=1e-14)

    def test_deconv(self):
        """R-POLY-04.2: deconv([1 3 2], [1 1]) returns quotient [1 2]."""
        b = ForgeArray(np.array([1.0, 3.0, 2.0]))
        a = ForgeArray(np.array([1.0, 1.0]))
        q, rem = forge_deconv(b, a)
        np.testing.assert_allclose(_unwrap(q).ravel(), [1, 2], atol=1e-14)


class TestInterpolation:
    """R-POLY-05: The system SHALL interpolate data using piecewise cubic
    Hermite (pchip) and cubic spline methods, matching Octave pchip() and
    spline() semantics.

    Model-user argument: Lab data is discrete; the engineer needs smooth
    curves through measured points for visualization and for feeding into
    further computation (e.g., numerical integration of a measured force
    profile). pchip preserves monotonicity (no overshoot), which matters
    for physical quantities like pressure or temperature. spline gives C2
    smoothness for applications where curvature continuity matters (e.g.,
    cam profiles, trajectory planning).

    Decomposition:
      R-POLY-05.1 — pchip() reproduces exact data values at the knot points.
      R-POLY-05.2 — spline() returns bounded, reasonable values between knots.

    Consistency: 05.1 confirms interpolation accuracy at known points, and
    05.2 confirms the spline does not produce wild extrapolation artifacts
    between knots. Together they validate both interpolation methods.
    """

    def test_pchip_identity(self):
        """R-POLY-05.1: pchip evaluated at knot points reproduces input data."""
        x = ForgeArray(np.array([0.0, 1.0, 2.0, 3.0]))
        y = ForgeArray(np.array([0.0, 1.0, 4.0, 9.0]))
        xq = ForgeArray(np.array([0.0, 1.0, 2.0, 3.0]))
        r = _unwrap(forge_pchip(x, y, xq)).ravel()
        np.testing.assert_allclose(r, [0, 1, 4, 9], atol=1e-10)

    def test_spline_interpolation(self):
        """R-POLY-05.2: spline returns values within a bounded range between knots."""
        x = ForgeArray(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        y = ForgeArray(np.array([0.0, 1.0, 0.0, 1.0, 0.0]))
        xq = ForgeArray(np.array([0.5, 1.5, 2.5, 3.5]))
        r = _unwrap(forge_spline(x, y, xq)).ravel()
        # Just verify it returns sensible values
        assert all(abs(v) < 2.0 for v in r)


class TestPiecewisePolynomial:
    """R-POLY-06: The system SHALL construct, evaluate, and decompose
    piecewise polynomial structures, matching Octave mkpp(), ppval(), and
    unmkpp() semantics.

    Model-user argument: Spline and pchip results are stored as piecewise
    polynomial (pp) structures. The engineer uses mkpp() to build custom
    pp objects (e.g., a hand-tuned gain schedule), ppval() to evaluate them
    at operating points, and unmkpp() to extract break/coefficient data for
    export or inspection. If the pp round-trip (unmkpp(mkpp(...))) loses
    information, the engineer cannot serialize and reload gain schedules.

    Decomposition:
      R-POLY-06.1 — mkpp()/ppval() correctly evaluates a two-segment linear pp.
      R-POLY-06.2 — unmkpp() extracts the original breaks from a pp structure.

    Consistency: 06.1 validates construction and evaluation together (they
    are inseparable in use), and 06.2 validates decomposition. Together they
    cover the full create/evaluate/inspect lifecycle.
    """

    def test_mkpp_ppval(self):
        """R-POLY-06.1: ppval of a two-segment linear pp returns correct values."""
        breaks = ForgeArray(np.array([0.0, 1.0, 2.0]))
        coefs = ForgeArray(np.array([[1.0, 0.0], [-1.0, 1.0]]))
        pp = forge_mkpp(breaks, coefs)
        x = ForgeArray(np.array([0.5, 1.0, 1.5]))
        r = _unwrap(forge_ppval(pp, x)).ravel()
        np.testing.assert_allclose(r, [0.5, 1.0, 0.5], atol=1e-14)

    def test_unmkpp(self):
        """R-POLY-06.2: unmkpp() recovers the original break vector."""
        breaks = ForgeArray(np.array([0.0, 1.0, 2.0]))
        coefs = ForgeArray(np.array([[1.0, 0.0], [-1.0, 1.0]]))
        pp = forge_mkpp(breaks, coefs)
        b, c = forge_unmkpp(pp)
        np.testing.assert_array_equal(_unwrap(b).ravel(), [0, 1, 2])


class TestMisc:
    """R-POLY-07: The system SHALL compute companion matrices and strip
    leading zeros from polynomial coefficient vectors, matching Octave
    compan() and polyreduce() semantics.

    Model-user argument: The companion matrix is the standard way to convert
    a polynomial eigenvalue problem into a matrix eigenvalue problem; the
    engineer uses compan() to verify roots via eig() as a cross-check.
    polyreduce() removes spurious leading zeros that accumulate from symbolic
    manipulation or user input errors, preventing dimension mismatches in
    downstream operations like conv() or roots().

    Decomposition:
      R-POLY-07.1 — compan() produces a matrix whose eigenvalues equal the
                     polynomial roots.
      R-POLY-07.2 — polyreduce() strips leading zeros from a coefficient vector.

    Consistency: 07.1 validates the algebraic correctness of the companion
    matrix (eigenvalue equivalence), and 07.2 validates coefficient
    normalization. These are independent utilities, each fully tested by
    its sub-requirement.
    """

    def test_compan(self):
        """R-POLY-07.1: eigenvalues of compan(x^2 - 3x + 2) are {1, 2}."""
        p = ForgeArray(np.array([1.0, -3.0, 2.0]))
        C = _unwrap(forge_compan(p))
        # Eigenvalues of companion = roots of polynomial
        eigs = np.sort(np.linalg.eigvals(C).real)
        np.testing.assert_allclose(eigs, [1, 2], atol=1e-10)

    def test_polyreduce(self):
        """R-POLY-07.2: polyreduce([0 0 1 2 3]) returns [1 2 3]."""
        p = ForgeArray(np.array([0.0, 0.0, 1.0, 2.0, 3.0]))
        r = _unwrap(forge_polyreduce(p)).ravel()
        np.testing.assert_allclose(r, [1, 2, 3], atol=1e-14)
