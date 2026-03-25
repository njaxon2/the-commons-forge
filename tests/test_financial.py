"""Tests for Forge Financial Toolbox (15 tests).

Uses FINANCIAL_REGISTRY dict to call functions by their registry name.
"""

import pytest
import numpy as np

from forge.engine.builtins.financial import FINANCIAL_REGISTRY


def _fn(name):
    return FINANCIAL_REGISTRY[name]


# ===========================================================================
# Time Value of Money
# ===========================================================================

class TestTimeValueOfMoney:
    def test_pvfix_1000_at_5pct_10yr(self):
        """PV of $1000 FV at 5% for 10 years ~ -613.91."""
        pvfix = _fn('pvfix')
        result = pvfix(0.05, 10, fv=1000)
        np.testing.assert_allclose(result, -613.913, atol=0.01)

    def test_fvfix_1000_pv_at_5pct_10yr(self):
        """FV of $1000 PV at 5% for 10 years ~ -1628.89."""
        fvfix = _fn('fvfix')
        result = fvfix(0.05, 10, pv=1000)
        np.testing.assert_allclose(result, -1628.895, atol=0.01)

    def test_npv_known(self):
        """NPV of [-1000, 300, 400, 500] at 10% ~ -21.04."""
        npv = _fn('npv')
        # npv in this implementation discounts starting from t=1
        # So initial investment must be added separately:
        # NPV = -1000 + npv(0.10, [300, 400, 500])
        pv = npv(0.10, [300, 400, 500])
        total = -1000 + pv
        np.testing.assert_allclose(total, -21.04, atol=1.0)

    def test_irr_known(self):
        """IRR of [-1000, 400, 400, 400] ~ 9.7%."""
        irr = _fn('irr')
        result = irr([-1000, 400, 400, 400])
        np.testing.assert_allclose(result, 0.0966, atol=0.01)

    def test_rate_calculation(self):
        """rate() should recover approximately 5% from PV/FV pair."""
        rate = _fn('rate')
        # $1000 PV grows to $1628.89 in 10 years at 5%
        r = rate(10, 0, -1000, 1628.89)
        np.testing.assert_allclose(r, 0.05, atol=0.001)

    def test_pmt_calculation(self):
        """pmt for $100k loan at 0.5%/mo for 360 months ~ -599.55."""
        pmt = _fn('pmt')
        result = pmt(0.005, 360, 100000)
        np.testing.assert_allclose(result, -599.55, atol=1.0)

    def test_nper_calculation(self):
        """nper to pay off $100k at 0.5%/mo with -600/mo payment."""
        nper = _fn('nper')
        result = nper(0.005, -600, 100000)
        assert result > 300  # should take ~360 months
        assert result < 400


# ===========================================================================
# Black-Scholes
# ===========================================================================

class TestBlackScholes:
    def test_blsprice_known(self):
        """BS call at S=100, K=100, T=1, r=0.05, sigma=0.2 ~ 10.45."""
        blsprice = _fn('blsprice')
        call, put = blsprice(100, 100, 0.05, 1, 0.2)
        np.testing.assert_allclose(call, 10.4506, atol=0.05)
        # Put-call parity check
        parity_diff = call - put - 100 + 100 * np.exp(-0.05)
        np.testing.assert_allclose(parity_diff, 0.0, atol=0.01)

    def test_blsdelta_call_range(self):
        """Call delta should be in [0, 1]."""
        blsdelta = _fn('blsdelta')
        call_delta, put_delta = blsdelta(100, 100, 0.05, 1, 0.2)
        assert 0 <= call_delta <= 1
        assert -1 <= put_delta <= 0

    def test_blsimpv_roundtrip(self):
        """Implied vol from BS price should recover original sigma."""
        blsprice = _fn('blsprice')
        blsimpv = _fn('blsimpv')
        sigma_orig = 0.25
        call, _ = blsprice(100, 100, 0.05, 1, sigma_orig)
        sigma_recovered = blsimpv(100, 100, 0.05, 1, call, call=True)
        np.testing.assert_allclose(sigma_recovered, sigma_orig, atol=0.001)

    def test_blsprice_put_positive(self):
        """OTM put (S=120, K=100) should still be positive."""
        blsprice = _fn('blsprice')
        _, put = blsprice(120, 100, 0.05, 1, 0.2)
        assert put > 0

    def test_blsprice_itm_call_greater_than_intrinsic(self):
        """ITM call (S=110, K=100) >= intrinsic value 10."""
        blsprice = _fn('blsprice')
        call, _ = blsprice(110, 100, 0.05, 1, 0.2)
        assert call >= 10.0


# ===========================================================================
# Consistency Checks
# ===========================================================================

class TestConsistency:
    def test_pvfix_fvfix_inverse(self):
        """pvfix and fvfix should be inverses."""
        pvfix = _fn('pvfix')
        fvfix = _fn('fvfix')
        pv = pvfix(0.06, 5, fv=1000)
        fv = fvfix(0.06, 5, pv=pv)
        # fv should recover ~-1000 (sign convention)
        np.testing.assert_allclose(fv, 1000, atol=0.01)

    def test_pmt_times_nper_covers_principal(self):
        """Total payments from pmt * nper should exceed loan principal."""
        pmt = _fn('pmt')
        payment = pmt(0.005, 360, 100000)
        total = abs(payment) * 360
        assert total > 100000  # must pay back more than borrowed
