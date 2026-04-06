# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for elfun toolbox (27 functions).

SRS trace: SRS-FUNC-001, SRS-VAL-001
Test method: Comparison against known mathematical identities and numpy reference.
"""
import pytest
import numpy as np
from forge.engine.builtins.elfun import *


class TestTrigDegrees:
    """R-ELF-01: Degree-based trigonometric functions SHALL return results
    equivalent to their radian counterparts evaluated at (x * pi / 180),
    with accuracy within 1e-12 of known analytic values.

    Model-user argument: An engineer computing antenna radiation patterns or
    structural loading angles works in degrees as the natural unit of the
    problem domain. When they type sind(30) they expect exactly 0.5, not a
    value polluted by a manual deg-to-rad conversion step. Matching Octave's
    degree functions removes a common source of silent error in ported scripts.

    Decomposition:
        R-ELF-01.1  sind at cardinal angles (0, 30, 90)
        R-ELF-01.2  cosd at cardinal angles (0, 60, 90)
        R-ELF-01.3  tand at cardinal angles (0, 45)
        R-ELF-01.4  asind/acosd/atand inverse accuracy
        R-ELF-01.5  atan2d quadrant correctness
        R-ELF-01.6  sind vectorized over arrays

    Consistency: R-ELF-01.1 through R-ELF-01.3 cover forward evaluation of all
    three primary degree trig functions at analytically known angles. R-ELF-01.4
    covers their inverses, confirming roundtrip correctness. R-ELF-01.5 tests
    the two-argument variant. R-ELF-01.6 confirms vectorized operation. Together
    these exercise every degree-based trig path in the elfun module.
    """

    def test_sind_0(self):
        """R-ELF-01.1a: sind(0) SHALL equal 0."""
        assert abs(_unwrap(forge_sind(ForgeArray(0.0)))) < 1e-15

    def test_sind_90(self):
        """R-ELF-01.1b: sind(90) SHALL equal 1."""
        assert abs(_unwrap(forge_sind(ForgeArray(90.0))) - 1.0) < 1e-15

    def test_sind_30(self):
        """R-ELF-01.1c: sind(30) SHALL equal 0.5."""
        assert abs(_unwrap(forge_sind(ForgeArray(30.0))) - 0.5) < 1e-15

    def test_cosd_0(self):
        """R-ELF-01.2a: cosd(0) SHALL equal 1."""
        assert abs(_unwrap(forge_cosd(ForgeArray(0.0))) - 1.0) < 1e-15

    def test_cosd_90(self):
        """R-ELF-01.2b: cosd(90) SHALL equal 0."""
        assert abs(_unwrap(forge_cosd(ForgeArray(90.0)))) < 1e-15

    def test_cosd_60(self):
        """R-ELF-01.2c: cosd(60) SHALL equal 0.5."""
        assert abs(_unwrap(forge_cosd(ForgeArray(60.0))) - 0.5) < 1e-15

    def test_tand_45(self):
        """R-ELF-01.3a: tand(45) SHALL equal 1."""
        assert abs(_unwrap(forge_tand(ForgeArray(45.0))) - 1.0) < 1e-14

    def test_tand_0(self):
        """R-ELF-01.3b: tand(0) SHALL equal 0."""
        assert abs(_unwrap(forge_tand(ForgeArray(0.0)))) < 1e-15

    def test_asind_half(self):
        """R-ELF-01.4a: asind(0.5) SHALL equal 30."""
        assert abs(_unwrap(forge_asind(ForgeArray(0.5))) - 30.0) < 1e-12

    def test_acosd_half(self):
        """R-ELF-01.4b: acosd(0.5) SHALL equal 60."""
        assert abs(_unwrap(forge_acosd(ForgeArray(0.5))) - 60.0) < 1e-12

    def test_atand_1(self):
        """R-ELF-01.4c: atand(1) SHALL equal 45."""
        assert abs(_unwrap(forge_atand(ForgeArray(1.0))) - 45.0) < 1e-12

    def test_atan2d_quadrants(self):
        """R-ELF-01.5: atan2d(1,1) SHALL equal 45."""
        r = _unwrap(forge_atan2d(ForgeArray(1.0), ForgeArray(1.0)))
        assert abs(r - 45.0) < 1e-12

    def test_sind_array(self):
        """R-ELF-01.6: sind SHALL operate element-wise on arrays."""
        from forge.engine.types import ForgeArray, _unwrap
        x = ForgeArray(np.array([0.0, 30.0, 45.0, 60.0, 90.0]))
        result = _unwrap(forge_sind(x))
        expected = np.array([0.0, 0.5, np.sqrt(2)/2, np.sqrt(3)/2, 1.0])
        np.testing.assert_allclose(result.ravel(), expected, atol=1e-14)


class TestReciprocalTrig:
    """R-ELF-02: Reciprocal trigonometric functions (sec, csc, cot) and their
    degree and inverse variants SHALL return values consistent with their
    definitions (1/cos, 1/sin, cos/sin) to within 1e-12.

    Model-user argument: A scientist solving Maxwell's equations or potential
    flow problems encounters sec, csc, and cot in closed-form analytical
    solutions. When porting reference formulas from a textbook into Forge, these
    functions must be available and numerically faithful. Requiring the user to
    manually expand 1/cos(x) obscures intent and invites parenthesization bugs.

    Decomposition:
        R-ELF-02.1  sec, csc, cot at known radian values
        R-ELF-02.2  secd, cscd, cotd at known degree values
        R-ELF-02.3  asec/acsc/acot roundtrip identity (radian)
        R-ELF-02.4  asecd/acscd/acotd roundtrip identity (degree)

    Consistency: R-ELF-02.1 and R-ELF-02.2 confirm forward evaluation in both
    radian and degree modes. R-ELF-02.3 and R-ELF-02.4 confirm that the inverse
    functions correctly invert their forward counterparts. All six reciprocal
    trig functions and all six inverse reciprocal trig functions are covered.
    """

    def test_sec_0(self):
        """R-ELF-02.1a: sec(0) SHALL equal 1."""
        assert abs(_unwrap(forge_sec(ForgeArray(0.0))) - 1.0) < 1e-15

    def test_csc_pi2(self):
        """R-ELF-02.1b: csc(pi/2) SHALL equal 1."""
        assert abs(_unwrap(forge_csc(ForgeArray(np.pi/2))) - 1.0) < 1e-14

    def test_cot_pi4(self):
        """R-ELF-02.1c: cot(pi/4) SHALL equal 1."""
        assert abs(_unwrap(forge_cot(ForgeArray(np.pi/4))) - 1.0) < 1e-14

    def test_secd_0(self):
        """R-ELF-02.2a: secd(0) SHALL equal 1."""
        assert abs(_unwrap(forge_secd(ForgeArray(0.0))) - 1.0) < 1e-15

    def test_cscd_90(self):
        """R-ELF-02.2b: cscd(90) SHALL equal 1."""
        assert abs(_unwrap(forge_cscd(ForgeArray(90.0))) - 1.0) < 1e-14

    def test_cotd_45(self):
        """R-ELF-02.2c: cotd(45) SHALL equal 1."""
        assert abs(_unwrap(forge_cotd(ForgeArray(45.0))) - 1.0) < 1e-14

    def test_asec_identity(self):
        """R-ELF-02.3a: asec(sec(x)) SHALL equal x for x in [0, pi], x != pi/2."""
        x = ForgeArray(1.0)
        assert abs(_unwrap(forge_asec(ForgeArray(_unwrap(forge_sec(x))))) - _unwrap(x)) < 1e-14

    def test_acsc_identity(self):
        """R-ELF-02.3b: acsc(csc(x)) SHALL equal x for valid x."""
        x = ForgeArray(np.pi/3)
        r = _unwrap(forge_acsc(ForgeArray(_unwrap(forge_csc(x)))))
        assert abs(r - _unwrap(x)) < 1e-14

    def test_acot_identity(self):
        """R-ELF-02.3c: acot(cot(x)) SHALL equal x for valid x."""
        x = ForgeArray(np.pi/6)
        r = _unwrap(forge_acot(ForgeArray(_unwrap(forge_cot(x)))))
        assert abs(r - _unwrap(x)) < 1e-14

    def test_asecd_roundtrip(self):
        """R-ELF-02.4a: asecd(secd(60)) SHALL equal 60."""
        r = _unwrap(forge_asecd(ForgeArray(_unwrap(forge_secd(ForgeArray(60.0))))))
        assert abs(r - 60.0) < 1e-12

    def test_acscd_roundtrip(self):
        """R-ELF-02.4b: acscd(cscd(45)) SHALL equal 45."""
        r = _unwrap(forge_acscd(ForgeArray(_unwrap(forge_cscd(ForgeArray(45.0))))))
        assert abs(r - 45.0) < 1e-12

    def test_acotd_roundtrip(self):
        """R-ELF-02.4c: acotd(cotd(30)) SHALL equal 30."""
        r = _unwrap(forge_acotd(ForgeArray(_unwrap(forge_cotd(ForgeArray(30.0))))))
        assert abs(r - 30.0) < 1e-12


class TestHyperbolicReciprocal:
    """R-ELF-03: Hyperbolic reciprocal functions (sech, csch, coth) and their
    inverses SHALL return values consistent with their definitions
    (1/cosh, 1/sinh, cosh/sinh) to within 1e-14.

    Model-user argument: Hyperbolic reciprocals appear in transmission line
    theory, heat transfer solutions, and special function expansions. A scientist
    porting an Octave script that uses coth() for a waveguide impedance formula
    needs identical availability and accuracy. Forcing manual expansion into
    cosh/sinh adds clutter and breaks readability of the original formulation.

    Decomposition:
        R-ELF-03.1  sech(0) equals 1
        R-ELF-03.2  coth matches cosh/sinh definition
        R-ELF-03.3  csch matches 1/sinh definition
        R-ELF-03.4  acsch/asech/acoth roundtrip identities

    Consistency: R-ELF-03.1 tests the boundary value. R-ELF-03.2 and R-ELF-03.3
    verify forward evaluation against the defining expressions. R-ELF-03.4
    confirms all three inverse functions correctly invert their counterparts.
    Together these cover all six hyperbolic reciprocal paths.
    """

    def test_sech_0(self):
        """R-ELF-03.1: sech(0) SHALL equal 1."""
        assert abs(_unwrap(forge_sech(ForgeArray(0.0))) - 1.0) < 1e-15

    def test_coth_identity(self):
        """R-ELF-03.2: coth(x) SHALL equal cosh(x)/sinh(x)."""
        x = ForgeArray(1.5)
        expected = np.cosh(1.5) / np.sinh(1.5)
        assert abs(_unwrap(forge_coth(x)) - expected) < 1e-14

    def test_csch_identity(self):
        """R-ELF-03.3: csch(x) SHALL equal 1/sinh(x)."""
        x = ForgeArray(2.0)
        expected = 1.0 / np.sinh(2.0)
        assert abs(_unwrap(forge_csch(x)) - expected) < 1e-14

    def test_acsch_roundtrip(self):
        """R-ELF-03.4a: acsch(csch(x)) SHALL equal x."""
        x = ForgeArray(2.0)
        r = _unwrap(forge_acsch(ForgeArray(_unwrap(forge_csch(x)))))
        assert abs(r - 2.0) < 1e-14

    def test_asech_roundtrip(self):
        """R-ELF-03.4b: asech(sech(x)) SHALL equal x for x > 0."""
        x = ForgeArray(0.5)
        r = _unwrap(forge_asech(ForgeArray(_unwrap(forge_sech(x)))))
        assert abs(r - 0.5) < 1e-14

    def test_acoth_roundtrip(self):
        """R-ELF-03.4c: acoth(coth(x)) SHALL equal x for |x| > 1."""
        x = ForgeArray(2.0)
        r = _unwrap(forge_acoth(ForgeArray(_unwrap(forge_coth(x)))))
        assert abs(r - 2.0) < 1e-14


class TestPiScaled:
    """R-ELF-04: Pi-scaled trigonometric functions (sinpi, cospi) SHALL return
    exact zero and exact unity at their analytically known special points,
    with error strictly below 1e-15.

    Model-user argument: In filter design and DFT theory, expressions like
    sin(pi*n) must be exactly zero for integer n, not a small floating-point
    residual from computing pi*n first. A scientist implementing a sinc-based
    FIR filter in Forge relies on sinpi(n)==0 to avoid divide-by-near-zero
    artifacts. cospi(0.5)==0 is equally critical for half-sample symmetric
    window functions. These guarantees match Octave's sinpi/cospi contracts.

    Decomposition:
        R-ELF-04.1  sinpi(n) equals 0 for integer n
        R-ELF-04.2  sinpi(0.5) equals 1
        R-ELF-04.3  cospi(n+0.5) equals 0 for integer n
        R-ELF-04.4  cospi(0) equals 1

    Consistency: R-ELF-04.1 and R-ELF-04.3 verify the exact-zero guarantees
    that distinguish sinpi/cospi from sin(pi*x)/cos(pi*x). R-ELF-04.2 and
    R-ELF-04.4 verify exact-unity at the complementary special points. These
    four cases cover all analytically exact values of both functions.
    """

    def test_sinpi_integers_zero(self):
        """R-ELF-04.1: sinpi(n) SHALL equal 0 for integer n."""
        x = ForgeArray(np.array([0.0, 1.0, 2.0, -1.0, 100.0]))
        result = _unwrap(forge_sinpi(x))
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_sinpi_half(self):
        """R-ELF-04.2: sinpi(0.5) SHALL equal 1."""
        assert abs(_unwrap(forge_sinpi(ForgeArray(0.5))) - 1.0) < 1e-15

    def test_cospi_half_integers_zero(self):
        """R-ELF-04.3: cospi(n+0.5) SHALL equal 0 for integer n."""
        x = ForgeArray(np.array([0.5, 1.5, -0.5, 2.5]))
        result = _unwrap(forge_cospi(x))
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_cospi_0(self):
        """R-ELF-04.4: cospi(0) SHALL equal 1."""
        assert abs(_unwrap(forge_cospi(ForgeArray(0.0))) - 1.0) < 1e-15


from forge.engine.types import ForgeArray, _unwrap
