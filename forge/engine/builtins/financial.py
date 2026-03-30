# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""
Forge Financial Toolbox
========================
Time value of money, bond analytics, Black-Scholes option pricing,
and basic portfolio optimization.

Backend: numpy, scipy.stats
"""

import numpy as np
from numpy import ndarray
from typing import Union, Optional, Dict, Any, Tuple

Numeric = Union[int, float, np.number]


# ---------------------------------------------------------------------------
# Time Value of Money
# ---------------------------------------------------------------------------

def pvfix(rate: Numeric, nper: Numeric, pmt: Numeric = 0,
          fv: Numeric = 0, pmt_type: int = 0) -> float:
    """Present value of fixed periodic payments.

    Parameters
    ----------
    rate : scalar – interest rate per period
    nper : scalar – number of periods
    pmt  : scalar – payment per period (default 0)
    fv   : scalar – future value (default 0)
    pmt_type : 0=end-of-period, 1=beginning-of-period
    """
    r = float(rate)
    n = float(nper)
    p = float(pmt)
    f = float(fv)
    if r == 0:
        return -(f + p * n)
    q = (1 + r) ** n
    when = 1 + r * pmt_type
    return -(f / q + p * when * (q - 1) / (r * q))


def fvfix(rate: Numeric, nper: Numeric, pmt: Numeric = 0,
          pv: Numeric = 0, pmt_type: int = 0) -> float:
    """Future value of fixed periodic payments."""
    r = float(rate)
    n = float(nper)
    p = float(pmt)
    v = float(pv)
    if r == 0:
        return -(v + p * n)
    q = (1 + r) ** n
    when = 1 + r * pmt_type
    return -(v * q + p * when * (q - 1) / r)


def npv(rate: Numeric, cashflows) -> float:
    """Net present value of a cash-flow series."""
    r = float(rate)
    cf = np.asarray(cashflows, dtype=float).ravel()
    t = np.arange(1, len(cf) + 1)
    return float(np.sum(cf / (1 + r) ** t))


def irr(cashflows) -> float:
    """Internal rate of return (Newton-Raphson)."""
    cf = np.asarray(cashflows, dtype=float).ravel()
    # initial guess
    guess = 0.1
    for _ in range(200):
        t = np.arange(len(cf))
        npv_val = np.sum(cf / (1 + guess) ** t)
        dnpv = np.sum(-t * cf / (1 + guess) ** (t + 1))
        if abs(dnpv) < 1e-14:
            break
        new_guess = guess - npv_val / dnpv
        if abs(new_guess - guess) < 1e-10:
            guess = new_guess
            break
        guess = new_guess
    return float(guess)


def rate(nper: Numeric, pmt: Numeric, pv: Numeric,
         fv: Numeric = 0, pmt_type: int = 0,
         guess: float = 0.1) -> float:
    """Solve for interest rate per period (Newton-Raphson)."""
    n = float(nper)
    p = float(pmt)
    v = float(pv)
    f = float(fv)
    r = guess
    for _ in range(200):
        if abs(r) < 1e-14:
            diff = v + p * n + f
            ddiff = 0.0
        else:
            q = (1 + r) ** n
            when = 1 + r * pmt_type
            diff = v * q + p * when * (q - 1) / r + f
            dq = n * (1 + r) ** (n - 1)
            ddiff = (v * dq +
                     p * pmt_type * (q - 1) / r +
                     p * when * (dq * r - (q - 1)) / r ** 2)
        if abs(ddiff) < 1e-14:
            break
        r_new = r - diff / ddiff
        if abs(r_new - r) < 1e-10:
            r = r_new
            break
        r = r_new
    return float(r)


def nper(rate_val: Numeric, pmt: Numeric, pv: Numeric,
         fv: Numeric = 0, pmt_type: int = 0) -> float:
    """Number of periods."""
    r = float(rate_val)
    p = float(pmt)
    v = float(pv)
    f = float(fv)
    if r == 0:
        if p == 0:
            return float('inf')
        return -(v + f) / p
    when = 1 + r * pmt_type
    z = p * when / r
    return float(np.log((-f + z) / (v + z)) / np.log(1 + r))


def pmt(rate_val: Numeric, nper_val: Numeric, pv: Numeric,
        fv: Numeric = 0, pmt_type: int = 0) -> float:
    """Payment per period."""
    r = float(rate_val)
    n = float(nper_val)
    v = float(pv)
    f = float(fv)
    if r == 0:
        return -(v + f) / n
    q = (1 + r) ** n
    when = 1 + r * pmt_type
    return -(v * q + f) * r / (when * (q - 1))


# ---------------------------------------------------------------------------
# Bond Analytics
# ---------------------------------------------------------------------------

def bndprice(yld: Numeric, coupon_rate: Numeric, nper_val: Numeric,
             face: Numeric = 100, freq: int = 2) -> float:
    """Clean price of a bond.

    Parameters
    ----------
    yld         : yield to maturity per period (annual / freq)
    coupon_rate : annual coupon rate
    nper_val    : number of coupon periods remaining
    face        : face value (default 100)
    freq        : coupons per year (default 2)
    """
    y = float(yld) / freq
    c = float(coupon_rate) / freq * float(face)
    n = int(nper_val)
    f = float(face)
    if y == 0:
        return c * n + f
    pv_coupons = c * (1 - (1 + y) ** (-n)) / y
    pv_face = f / (1 + y) ** n
    return float(pv_coupons + pv_face)


def bndyield(price: Numeric, coupon_rate: Numeric, nper_val: Numeric,
             face: Numeric = 100, freq: int = 2,
             guess: float = 0.05) -> float:
    """Yield to maturity given bond price (Newton-Raphson)."""
    p = float(price)
    cr = float(coupon_rate)
    n = int(nper_val)
    f = float(face)
    yld = guess
    for _ in range(200):
        bp = bndprice(yld, cr, n, f, freq)
        # numerical derivative
        dy = 1e-6
        bp2 = bndprice(yld + dy, cr, n, f, freq)
        deriv = (bp2 - bp) / dy
        if abs(deriv) < 1e-14:
            break
        yld_new = yld - (bp - p) / deriv
        if abs(yld_new - yld) < 1e-10:
            yld = yld_new
            break
        yld = yld_new
    return float(yld)


def cfamounts(coupon_rate: Numeric, nper_val: int,
              face: Numeric = 100, freq: int = 2) -> ndarray:
    """Cash-flow amounts for a bond (array of coupon + principal at end)."""
    c = float(coupon_rate) / freq * float(face)
    n = int(nper_val)
    cf = np.full(n, c)
    cf[-1] += float(face)
    return cf


def accrfrac(settle_days: Numeric, coupon_days: Numeric) -> float:
    """Accrued interest fraction = settle_days / coupon_days."""
    return float(settle_days) / float(coupon_days)


# ---------------------------------------------------------------------------
# Black-Scholes Option Pricing
# ---------------------------------------------------------------------------

def _bs_d1d2(S, K, r, T, sigma):
    from scipy.stats import norm as _norm  # noqa: F811
    s = float(S); k = float(K); rr = float(r)
    t = float(T); sig = float(sigma)
    d1 = (np.log(s / k) + (rr + 0.5 * sig ** 2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    return d1, d2, s, k, rr, t, sig


def blsprice(S: Numeric, K: Numeric, r: Numeric,
             T: Numeric, sigma: Numeric) -> Tuple[float, float]:
    """Black-Scholes European call and put price.

    Returns (call_price, put_price).
    """
    from scipy.stats import norm
    d1, d2, s, k, rr, t, sig = _bs_d1d2(S, K, r, T, sigma)
    call = s * norm.cdf(d1) - k * np.exp(-rr * t) * norm.cdf(d2)
    put = k * np.exp(-rr * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
    return float(call), float(put)


def blsimpv(S: Numeric, K: Numeric, r: Numeric,
            T: Numeric, price: Numeric, call: bool = True,
            tol: float = 1e-8) -> float:
    """Implied volatility via bisection."""
    lo, hi = 1e-6, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        c, p = blsprice(S, K, r, T, mid)
        model_price = c if call else p
        if model_price > float(price):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def blsdelta(S: Numeric, K: Numeric, r: Numeric,
             T: Numeric, sigma: Numeric) -> Tuple[float, float]:
    """Black-Scholes delta (call, put)."""
    from scipy.stats import norm
    d1, *_ = _bs_d1d2(S, K, r, T, sigma)
    return float(norm.cdf(d1)), float(norm.cdf(d1) - 1)


def blsgamma(S: Numeric, K: Numeric, r: Numeric,
             T: Numeric, sigma: Numeric) -> float:
    """Black-Scholes gamma (same for call and put)."""
    from scipy.stats import norm
    d1, d2, s, k, rr, t, sig = _bs_d1d2(S, K, r, T, sigma)
    return float(norm.pdf(d1) / (s * sig * np.sqrt(t)))


def blsvega(S: Numeric, K: Numeric, r: Numeric,
            T: Numeric, sigma: Numeric) -> float:
    """Black-Scholes vega."""
    from scipy.stats import norm
    d1, d2, s, k, rr, t, sig = _bs_d1d2(S, K, r, T, sigma)
    return float(s * norm.pdf(d1) * np.sqrt(t))


def blstheta(S: Numeric, K: Numeric, r: Numeric,
             T: Numeric, sigma: Numeric) -> Tuple[float, float]:
    """Black-Scholes theta (call, put)."""
    from scipy.stats import norm
    d1, d2, s, k, rr, t, sig = _bs_d1d2(S, K, r, T, sigma)
    common = -(s * norm.pdf(d1) * sig) / (2 * np.sqrt(t))
    call_theta = common - rr * k * np.exp(-rr * t) * norm.cdf(d2)
    put_theta = common + rr * k * np.exp(-rr * t) * norm.cdf(-d2)
    return float(call_theta), float(put_theta)


def blsrho(S: Numeric, K: Numeric, r: Numeric,
           T: Numeric, sigma: Numeric) -> Tuple[float, float]:
    """Black-Scholes rho (call, put)."""
    from scipy.stats import norm
    d1, d2, s, k, rr, t, sig = _bs_d1d2(S, K, r, T, sigma)
    call_rho = k * t * np.exp(-rr * t) * norm.cdf(d2)
    put_rho = -k * t * np.exp(-rr * t) * norm.cdf(-d2)
    return float(call_rho), float(put_rho)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def frontcon(returns: ndarray, cov_matrix: ndarray,
             n_points: int = 20) -> Tuple[ndarray, ndarray, ndarray]:
    """Efficient frontier via quadratic optimisation (analytic two-fund).

    Returns (risk, ret, weights) – each row of weights is a portfolio.
    """
    mu = np.asarray(returns, dtype=float).ravel()
    C = np.asarray(cov_matrix, dtype=float)
    n = len(mu)
    ones = np.ones(n)
    Cinv = np.linalg.inv(C)

    a = float(ones @ Cinv @ ones)
    b = float(ones @ Cinv @ mu)
    c = float(mu @ Cinv @ mu)
    det = a * c - b * b

    mu_min = b / a
    mu_max = max(mu) * 1.2
    target_rets = np.linspace(mu_min, mu_max, n_points)

    risks = np.empty(n_points)
    all_weights = np.empty((n_points, n))

    for i, tr in enumerate(target_rets):
        lam = (c - b * tr) / det
        gam = (a * tr - b) / det
        w = Cinv @ (lam * ones + gam * mu)
        all_weights[i] = w
        risks[i] = np.sqrt(float(w @ C @ w))

    return risks, target_rets, all_weights


def portopt(returns: ndarray, cov_matrix: ndarray,
            n_portfolios: int = 10) -> Dict[str, ndarray]:
    """Basic mean-variance portfolio optimisation.

    Returns dict with keys: 'risk', 'return', 'weights'.
    """
    risks, rets, weights = frontcon(returns, cov_matrix, n_portfolios)
    return {'risk': risks, 'return': rets, 'weights': weights}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def disc2cont(disc_rate: Numeric, freq: int = 1) -> float:
    """Discrete to continuous compounding rate."""
    return float(freq * np.log(1 + float(disc_rate) / freq))


def cont2disc(cont_rate: Numeric, freq: int = 1) -> float:
    """Continuous to discrete compounding rate."""
    return float(freq * (np.exp(float(cont_rate) / freq) - 1))


def ret2tick(returns) -> ndarray:
    """Convert return series to price series (starting at 1)."""
    r = np.asarray(returns, dtype=float).ravel()
    prices = np.empty(len(r) + 1)
    prices[0] = 1.0
    prices[1:] = np.cumprod(1 + r)
    return prices


def tick2ret(prices) -> ndarray:
    """Convert price series to return series."""
    p = np.asarray(prices, dtype=float).ravel()
    return (p[1:] - p[:-1]) / p[:-1]


def portsim(returns: ndarray, cov_matrix: ndarray,
            n_steps: int = 252, n_sims: int = 1000,
            weights=None) -> ndarray:
    """Simulate portfolio paths via geometric Brownian motion.

    Returns array of shape (n_steps+1, n_sims) with portfolio values.
    """
    mu = np.asarray(returns, dtype=float).ravel()
    C = np.asarray(cov_matrix, dtype=float)
    n = len(mu)
    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.asarray(weights, dtype=float).ravel()

    port_mu = float(w @ mu)
    port_var = float(w @ C @ w)
    port_sigma = np.sqrt(port_var)

    dt = 1.0 / 252
    paths = np.ones((n_steps + 1, n_sims))
    rng = np.random.default_rng()
    for t in range(1, n_steps + 1):
        z = rng.standard_normal(n_sims)
        paths[t] = paths[t - 1] * np.exp(
            (port_mu - 0.5 * port_var) * dt + port_sigma * np.sqrt(dt) * z
        )
    return paths


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FINANCIAL_REGISTRY: Dict[str, Any] = {
    'pvfix': pvfix,
    'fvfix': fvfix,
    'npv': npv,
    'irr': irr,
    'rate': rate,
    'nper': nper,
    'pmt': pmt,
    'bndprice': bndprice,
    'bndyield': bndyield,
    'cfamounts': cfamounts,
    'accrfrac': accrfrac,
    'blsprice': blsprice,
    'blsimpv': blsimpv,
    'blsdelta': blsdelta,
    'blsgamma': blsgamma,
    'blsvega': blsvega,
    'blstheta': blstheta,
    'blsrho': blsrho,
    'frontcon': frontcon,
    'portopt': portopt,
    'disc2cont': disc2cont,
    'cont2disc': cont2disc,
    'ret2tick': ret2tick,
    'tick2ret': tick2ret,
    'portsim': portsim,
}
