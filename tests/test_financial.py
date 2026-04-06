# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
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
    """R-FIN-01: Forge SHALL provide time-value-of-money functions (pvfix,
    fvfix, npv, irr, rate, pmt, nper) that return results matching
    standard financial formulas to within published reference tolerances.

    Model-user argument: An engineer evaluating capital equipment purchases
    uses NPV and IRR to compare project alternatives, pmt to model lease
    payments, and rate to back-solve financing terms. These calculations
    previously ran in Octave's financial toolbox. Numerical agreement with
    textbook formulas is essential; a 1% error in IRR can flip a go/no-go
    decision on a multi-year facility investment.

    Decomposition:
      R-FIN-01a: pvfix of $1000 FV at 5% for 10 years returns ~-613.91.
      R-FIN-01b: fvfix of $1000 PV at 5% for 10 years returns ~-1628.89.
      R-FIN-01c: NPV of [-1000, 300, 400, 500] at 10% returns ~-21.04.
      R-FIN-01d: IRR of [-1000, 400, 400, 400] returns ~9.7%.
      R-FIN-01e: rate() recovers ~5% from a known PV/FV pair.
      R-FIN-01f: pmt for $100k at 0.5%/mo for 360 months returns ~-599.55.
      R-FIN-01g: nper to repay $100k at 0.5%/mo with $600/mo is 300-400.

    Consistency argument: R-FIN-01a and R-FIN-01b test the two fundamental
    present/future value primitives. R-FIN-01c and R-FIN-01d test
    multi-cashflow analysis (NPV, IRR). R-FIN-01e tests the inverse
    rate-solving path. R-FIN-01f and R-FIN-01g test the loan amortization
    pair. Together these cover every time-value-of-money function in the
    toolbox against known reference values.
    """

    def test_pvfix_1000_at_5pct_10yr(self):
        """R-FIN-01a: PV of $1000 FV at 5% for 10 years is ~-613.91."""
        pvfix = _fn('pvfix')
        result = pvfix(0.05, 10, fv=1000)
        np.testing.assert_allclose(result, -613.913, atol=0.01)

    def test_fvfix_1000_pv_at_5pct_10yr(self):
        """R-FIN-01b: FV of $1000 PV at 5% for 10 years is ~-1628.89."""
        fvfix = _fn('fvfix')
        result = fvfix(0.05, 10, pv=1000)
        np.testing.assert_allclose(result, -1628.895, atol=0.01)

    def test_npv_known(self):
        """R-FIN-01c: NPV of [-1000, 300, 400, 500] at 10% is ~-21.04."""
        npv = _fn('npv')
        # npv in this implementation discounts starting from t=1
        # So initial investment must be added separately:
        # NPV = -1000 + npv(0.10, [300, 400, 500])
        pv = npv(0.10, [300, 400, 500])
        total = -1000 + pv
        np.testing.assert_allclose(total, -21.04, atol=1.0)

    def test_irr_known(self):
        """R-FIN-01d: IRR of [-1000, 400, 400, 400] is ~9.7%."""
        irr = _fn('irr')
        result = irr([-1000, 400, 400, 400])
        np.testing.assert_allclose(result, 0.0966, atol=0.01)

    def test_rate_calculation(self):
        """R-FIN-01e: rate() recovers ~5% from a known PV/FV pair."""
        rate = _fn('rate')
        # $1000 PV grows to $1628.89 in 10 years at 5%
        r = rate(10, 0, -1000, 1628.89)
        np.testing.assert_allclose(r, 0.05, atol=0.001)

    def test_pmt_calculation(self):
        """R-FIN-01f: pmt for $100k at 0.5%/mo for 360 months is ~-599.55."""
        pmt = _fn('pmt')
        result = pmt(0.005, 360, 100000)
        np.testing.assert_allclose(result, -599.55, atol=1.0)

    def test_nper_calculation(self):
        """R-FIN-01g: nper for $100k at 0.5%/mo with $600/mo payment."""
        nper = _fn('nper')
        result = nper(0.005, -600, 100000)
        assert result > 300  # should take ~360 months
        assert result < 400


# ===========================================================================
# Black-Scholes
# ===========================================================================

class TestBlackScholes:
    """R-FIN-02: Forge SHALL provide Black-Scholes option pricing functions
    (blsprice, blsdelta, blsimpv) that satisfy put-call parity and recover
    input parameters through inverse operations.

    Model-user argument: The engineer uses Black-Scholes to price real
    options on engineering projects (e.g., option to expand a facility,
    option to abandon). In Octave, blsprice/blsdelta/blsimpv form a
    standard analysis set. Put-call parity is the fundamental consistency
    check; if it fails, all derived Greeks and implied vol calculations
    are suspect.

    Decomposition:
      R-FIN-02a: blsprice at S=K=100, T=1, r=0.05, sigma=0.2 returns
                 call ~10.45 and satisfies put-call parity.
      R-FIN-02b: blsdelta call delta is in [0,1], put delta in [-1,0].
      R-FIN-02c: blsimpv recovers original sigma from a BS price.
      R-FIN-02d: OTM put (S=120, K=100) has positive price.
      R-FIN-02e: ITM call (S=110, K=100) exceeds intrinsic value of 10.

    Consistency argument: R-FIN-02a validates the core pricing formula and
    put-call parity. R-FIN-02b validates the first-order Greek. R-FIN-02c
    validates the inverse (implied vol) path. R-FIN-02d and R-FIN-02e
    check boundary behavior for OTM and ITM options. Together they cover
    pricing, Greeks, inverse recovery, and boundary conditions.
    """

    def test_blsprice_known(self):
        """R-FIN-02a: BS call ~10.45 and satisfies put-call parity."""
        blsprice = _fn('blsprice')
        call, put = blsprice(100, 100, 0.05, 1, 0.2)
        np.testing.assert_allclose(call, 10.4506, atol=0.05)
        # Put-call parity check
        parity_diff = call - put - 100 + 100 * np.exp(-0.05)
        np.testing.assert_allclose(parity_diff, 0.0, atol=0.01)

    def test_blsdelta_call_range(self):
        """R-FIN-02b: Call delta in [0,1], put delta in [-1,0]."""
        blsdelta = _fn('blsdelta')
        call_delta, put_delta = blsdelta(100, 100, 0.05, 1, 0.2)
        assert 0 <= call_delta <= 1
        assert -1 <= put_delta <= 0

    def test_blsimpv_roundtrip(self):
        """R-FIN-02c: Implied vol recovers original sigma."""
        blsprice = _fn('blsprice')
        blsimpv = _fn('blsimpv')
        sigma_orig = 0.25
        call, _ = blsprice(100, 100, 0.05, 1, sigma_orig)
        sigma_recovered = blsimpv(100, 100, 0.05, 1, call, call=True)
        np.testing.assert_allclose(sigma_recovered, sigma_orig, atol=0.001)

    def test_blsprice_put_positive(self):
        """R-FIN-02d: OTM put (S=120, K=100) has positive price."""
        blsprice = _fn('blsprice')
        _, put = blsprice(120, 100, 0.05, 1, 0.2)
        assert put > 0

    def test_blsprice_itm_call_greater_than_intrinsic(self):
        """R-FIN-02e: ITM call (S=110, K=100) exceeds intrinsic value 10."""
        blsprice = _fn('blsprice')
        call, _ = blsprice(110, 100, 0.05, 1, 0.2)
        assert call >= 10.0


# ===========================================================================
# Consistency Checks
# ===========================================================================

class TestConsistency:
    """R-FIN-03: Forge SHALL ensure internal consistency between financial
    functions such that inverse operations and accounting identities hold.

    Model-user argument: When the engineer builds a financial model that
    chains multiple functions (e.g., compute PV, then verify with FV, then
    compute total payments), internal inconsistencies would produce silent
    errors in project evaluations. In Octave, pvfix and fvfix are exact
    inverses. The total-payments-exceed-principal identity is a basic
    sanity check for any amortization model.

    Decomposition:
      R-FIN-03a: pvfix and fvfix are inverses (round-trip recovers FV).
      R-FIN-03b: Total payments (pmt * nper) exceed the loan principal.

    Consistency argument: R-FIN-03a tests algebraic invertibility of the
    core PV/FV pair. R-FIN-03b tests the economic invariant that interest
    makes total payments exceed principal. Together they confirm that
    chained financial calculations remain self-consistent.
    """

    def test_pvfix_fvfix_inverse(self):
        """R-FIN-03a: pvfix and fvfix are inverses."""
        pvfix = _fn('pvfix')
        fvfix = _fn('fvfix')
        pv = pvfix(0.06, 5, fv=1000)
        fv = fvfix(0.06, 5, pv=pv)
        # fv should recover ~-1000 (sign convention)
        np.testing.assert_allclose(fv, 1000, atol=0.01)

    def test_pmt_times_nper_covers_principal(self):
        """R-FIN-03b: Total payments exceed loan principal."""
        pmt = _fn('pmt')
        payment = pmt(0.005, 360, 100000)
        total = abs(payment) * 360
        assert total > 100000  # must pay back more than borrowed
