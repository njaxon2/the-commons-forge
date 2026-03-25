"""V&V tests for elfun toolbox (27 functions).

SRS trace: SRS-FUNC-001, SRS-VAL-001
Test method: Comparison against known mathematical identities and numpy reference.
"""
import pytest
import numpy as np
from forge.engine.builtins.elfun import *


class TestTrigDegrees:
    """Verify degree-based trig functions against radian equivalents."""

    def test_sind_0(self):
        assert abs(_unwrap(forge_sind(ForgeArray(0.0)))) < 1e-15

    def test_sind_90(self):
        assert abs(_unwrap(forge_sind(ForgeArray(90.0))) - 1.0) < 1e-15

    def test_sind_30(self):
        assert abs(_unwrap(forge_sind(ForgeArray(30.0))) - 0.5) < 1e-15

    def test_cosd_0(self):
        assert abs(_unwrap(forge_cosd(ForgeArray(0.0))) - 1.0) < 1e-15

    def test_cosd_90(self):
        assert abs(_unwrap(forge_cosd(ForgeArray(90.0)))) < 1e-15

    def test_cosd_60(self):
        assert abs(_unwrap(forge_cosd(ForgeArray(60.0))) - 0.5) < 1e-15

    def test_tand_45(self):
        assert abs(_unwrap(forge_tand(ForgeArray(45.0))) - 1.0) < 1e-14

    def test_tand_0(self):
        assert abs(_unwrap(forge_tand(ForgeArray(0.0)))) < 1e-15

    def test_asind_half(self):
        assert abs(_unwrap(forge_asind(ForgeArray(0.5))) - 30.0) < 1e-12

    def test_acosd_half(self):
        assert abs(_unwrap(forge_acosd(ForgeArray(0.5))) - 60.0) < 1e-12

    def test_atand_1(self):
        assert abs(_unwrap(forge_atand(ForgeArray(1.0))) - 45.0) < 1e-12

    def test_atan2d_quadrants(self):
        r = _unwrap(forge_atan2d(ForgeArray(1.0), ForgeArray(1.0)))
        assert abs(r - 45.0) < 1e-12

    def test_sind_array(self):
        """Vectorized operation."""
        from forge.engine.types import ForgeArray, _unwrap
        x = ForgeArray(np.array([0.0, 30.0, 45.0, 60.0, 90.0]))
        result = _unwrap(forge_sind(x))
        expected = np.array([0.0, 0.5, np.sqrt(2)/2, np.sqrt(3)/2, 1.0])
        np.testing.assert_allclose(result.ravel(), expected, atol=1e-14)


class TestReciprocalTrig:
    """Verify sec, csc, cot and their degree/inverse variants."""

    def test_sec_0(self):
        assert abs(_unwrap(forge_sec(ForgeArray(0.0))) - 1.0) < 1e-15

    def test_csc_pi2(self):
        assert abs(_unwrap(forge_csc(ForgeArray(np.pi/2))) - 1.0) < 1e-14

    def test_cot_pi4(self):
        assert abs(_unwrap(forge_cot(ForgeArray(np.pi/4))) - 1.0) < 1e-14

    def test_secd_0(self):
        assert abs(_unwrap(forge_secd(ForgeArray(0.0))) - 1.0) < 1e-15

    def test_cscd_90(self):
        assert abs(_unwrap(forge_cscd(ForgeArray(90.0))) - 1.0) < 1e-14

    def test_cotd_45(self):
        assert abs(_unwrap(forge_cotd(ForgeArray(45.0))) - 1.0) < 1e-14

    def test_asec_identity(self):
        """asec(sec(x)) == x for x in [0, pi], x != pi/2."""
        x = ForgeArray(1.0)
        assert abs(_unwrap(forge_asec(ForgeArray(_unwrap(forge_sec(x))))) - _unwrap(x)) < 1e-14

    def test_acsc_identity(self):
        x = ForgeArray(np.pi/3)
        r = _unwrap(forge_acsc(ForgeArray(_unwrap(forge_csc(x)))))
        assert abs(r - _unwrap(x)) < 1e-14

    def test_acot_identity(self):
        x = ForgeArray(np.pi/6)
        r = _unwrap(forge_acot(ForgeArray(_unwrap(forge_cot(x)))))
        assert abs(r - _unwrap(x)) < 1e-14

    def test_asecd_roundtrip(self):
        r = _unwrap(forge_asecd(ForgeArray(_unwrap(forge_secd(ForgeArray(60.0))))))
        assert abs(r - 60.0) < 1e-12

    def test_acscd_roundtrip(self):
        r = _unwrap(forge_acscd(ForgeArray(_unwrap(forge_cscd(ForgeArray(45.0))))))
        assert abs(r - 45.0) < 1e-12

    def test_acotd_roundtrip(self):
        r = _unwrap(forge_acotd(ForgeArray(_unwrap(forge_cotd(ForgeArray(30.0))))))
        assert abs(r - 30.0) < 1e-12


class TestHyperbolicReciprocal:
    """Verify hyperbolic reciprocal functions."""

    def test_sech_0(self):
        assert abs(_unwrap(forge_sech(ForgeArray(0.0))) - 1.0) < 1e-15

    def test_coth_identity(self):
        """coth(x) = cosh(x)/sinh(x)."""
        x = ForgeArray(1.5)
        expected = np.cosh(1.5) / np.sinh(1.5)
        assert abs(_unwrap(forge_coth(x)) - expected) < 1e-14

    def test_csch_identity(self):
        x = ForgeArray(2.0)
        expected = 1.0 / np.sinh(2.0)
        assert abs(_unwrap(forge_csch(x)) - expected) < 1e-14

    def test_acsch_roundtrip(self):
        x = ForgeArray(2.0)
        r = _unwrap(forge_acsch(ForgeArray(_unwrap(forge_csch(x)))))
        assert abs(r - 2.0) < 1e-14

    def test_asech_roundtrip(self):
        x = ForgeArray(0.5)
        r = _unwrap(forge_asech(ForgeArray(_unwrap(forge_sech(x)))))
        assert abs(r - 0.5) < 1e-14

    def test_acoth_roundtrip(self):
        x = ForgeArray(2.0)
        r = _unwrap(forge_acoth(ForgeArray(_unwrap(forge_coth(x)))))
        assert abs(r - 2.0) < 1e-14


class TestPiScaled:
    """Verify sinpi/cospi exactness at special values."""

    def test_sinpi_integers_zero(self):
        """sinpi(n) == 0 for integer n."""
        x = ForgeArray(np.array([0.0, 1.0, 2.0, -1.0, 100.0]))
        result = _unwrap(forge_sinpi(x))
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_sinpi_half(self):
        assert abs(_unwrap(forge_sinpi(ForgeArray(0.5))) - 1.0) < 1e-15

    def test_cospi_half_integers_zero(self):
        """cospi(n+0.5) == 0 for integer n."""
        x = ForgeArray(np.array([0.5, 1.5, -0.5, 2.5]))
        result = _unwrap(forge_cospi(x))
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_cospi_0(self):
        assert abs(_unwrap(forge_cospi(ForgeArray(0.0))) - 1.0) < 1e-15


from forge.engine.types import ForgeArray, _unwrap
