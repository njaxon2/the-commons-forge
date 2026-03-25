"""Special functions (specfun toolbox).

Implements: beta, betainc, betaincinv, betaln, cosint, ellipke, expint,
factor, factorial, gammainc, gammaincinv, isprime, lcm, legendre,
nchoosek, nthroot, primes, reallog, realpow, realsqrt, sinint

SRS trace: SRS-FUNC-001
"""
import numpy as np
from scipy import special
from forge.engine.types import ForgeArray, _unwrap


def _wrap(x):
    return ForgeArray(np.asarray(x))

def _scalar(x):
    if isinstance(x, ForgeArray):
        return x.data.flat[0].item()
    return x


def forge_beta(a, b):
    return _wrap(special.beta(_unwrap(a), _unwrap(b)))

def forge_betaln(a, b):
    return _wrap(special.betaln(_unwrap(a), _unwrap(b)))

def forge_betainc(x, a, b):
    return _wrap(special.betainc(_unwrap(a), _unwrap(b), _unwrap(x)))

def forge_betaincinv(y, a, b):
    return _wrap(special.betaincinv(_unwrap(a), _unwrap(b), _unwrap(y)))

def forge_factorial(n):
    return _wrap(special.factorial(_unwrap(n), exact=False))

def forge_nchoosek(n, k):
    return _wrap(special.comb(_unwrap(n), _unwrap(k), exact=False))

def forge_isprime(n):
    arr = np.atleast_1d(_unwrap(n)).astype(int)
    result = np.zeros_like(arr, dtype=bool)
    for idx, v in np.ndenumerate(arr):
        if v < 2:
            result[idx] = False
        elif v < 4:
            result[idx] = True
        elif v % 2 == 0:
            result[idx] = False
        else:
            is_p = True
            for d in range(3, int(np.sqrt(v)) + 1, 2):
                if v % d == 0:
                    is_p = False
                    break
            result[idx] = is_p
    return ForgeArray(result.ravel())

def forge_primes(n):
    n_val = int(_scalar(n))
    if n_val < 2:
        return ForgeArray(np.array([], dtype=int))
    sieve = np.ones(n_val + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(n_val)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return ForgeArray(np.where(sieve)[0])

def forge_factor(n):
    n_val = int(_scalar(n))
    if n_val <= 1:
        return ForgeArray(np.array([n_val]))
    factors = []
    d = 2
    while d * d <= n_val:
        while n_val % d == 0:
            factors.append(d)
            n_val //= d
        d += 1
    if n_val > 1:
        factors.append(n_val)
    return ForgeArray(np.array(factors))

def forge_lcm(a, b):
    ad, bd = np.atleast_1d(_unwrap(a)).astype(int), np.atleast_1d(_unwrap(b)).astype(int)
    r = np.lcm(ad, bd)
    return ForgeArray(r.ravel()) if np.ndim(r) > 0 else ForgeArray(r)

def forge_expint(x):
    return _wrap(special.exp1(_unwrap(x)))

def forge_sinint(x):
    return _wrap(special.sici(_unwrap(x))[0])

def forge_cosint(x):
    return _wrap(special.sici(_unwrap(x))[1])

def forge_ellipke(m):
    md = _unwrap(m)
    K = special.ellipk(md)
    E = special.ellipe(md)
    return ForgeArray(K), ForgeArray(E)

def forge_legendre(n, x):
    n_val = int(_scalar(n))
    xd = _unwrap(x).ravel()
    # Returns (n+1) x len(x) matrix of associated Legendre functions
    result = np.zeros((n_val + 1, len(xd)))
    for m in range(n_val + 1):
        result[m, :] = special.lpmv(m, n_val, xd)
    return ForgeArray(result)

def forge_nthroot(x, n):
    xd, nd = _unwrap(x), _unwrap(n)
    return _wrap(np.sign(xd) * np.abs(xd) ** (1.0 / nd))

def forge_reallog(x):
    xd = _unwrap(x)
    if np.any(xd <= 0):
        raise ValueError("reallog: argument must be positive")
    return _wrap(np.log(xd))

def forge_realpow(x, y):
    xd, yd = _unwrap(x), _unwrap(y)
    result = xd ** yd
    if np.any(np.iscomplex(result)):
        raise ValueError("realpow: result is complex")
    return _wrap(np.real(result))

def forge_realsqrt(x):
    xd = _unwrap(x)
    if np.any(xd < 0):
        raise ValueError("realsqrt: argument must be non-negative")
    return _wrap(np.sqrt(xd))

def forge_gammainc(x, a):
    return _wrap(special.gammainc(_unwrap(a), _unwrap(x)))

def forge_gammaincinv(y, a):
    return _wrap(special.gammaincinv(_unwrap(a), _unwrap(y)))


SPECFUN_REGISTRY = {
    "beta": forge_beta, "betaln": forge_betaln,
    "betainc": forge_betainc, "betaincinv": forge_betaincinv,
    "factorial": forge_factorial, "nchoosek": forge_nchoosek,
    "isprime": forge_isprime, "primes": forge_primes,
    "factor": forge_factor, "lcm": forge_lcm,
    "expint": forge_expint, "sinint": forge_sinint,
    "cosint": forge_cosint, "ellipke": forge_ellipke,
    "legendre": forge_legendre, "nthroot": forge_nthroot,
    "reallog": forge_reallog, "realpow": forge_realpow,
    "realsqrt": forge_realsqrt,
    "gammainc": forge_gammainc, "gammaincinv": forge_gammaincinv,
}
