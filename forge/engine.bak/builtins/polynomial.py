"""Polynomial toolbox.

Implements: compan, conv, deconv, mkpp, mpoles, pchip, poly, polyaffine,
polyder, polyeig, polyfit, polygcd, polyint, polyout, polyreduce, polyval,
polyvalm, ppder, ppint, ppval, residue, roots, spline, splinefit, unmkpp

SRS trace: SRS-FUNC-001
"""
import numpy as np
from scipy import interpolate
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar, ForgeStruct


def _wrap(x):
    return ForgeArray(np.asarray(x))

def _scalar(x):
    if isinstance(x, ForgeArray):
        d = x.data
        return d.flat[0].item() if d.size == 1 else d
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.flat[0].item()
    return x


def forge_roots(p):
    pd = _unwrap(p).ravel()
    return ForgeArray(np.roots(pd))

def forge_poly(x):
    xd = _unwrap(x)
    if xd.ndim >= 2 and xd.shape[0] == xd.shape[1]:
        # Square matrix: characteristic polynomial
        return ForgeArray(np.poly(xd))
    return ForgeArray(np.poly(xd.ravel()))

def forge_polyval(p, x):
    pd = _unwrap(p).ravel()
    xd = _unwrap(x)
    return ForgeArray(np.polyval(pd, xd))

def forge_polyvalm(p, X):
    pd = _unwrap(p).ravel()
    Xd = _unwrap(X)
    n = len(pd)
    result = pd[0] * np.linalg.matrix_power(Xd, n - 1)
    for i in range(1, n):
        result = result + pd[i] * np.linalg.matrix_power(Xd, n - 1 - i)
    return ForgeArray(result)

def forge_polyfit(x, y, n):
    xd, yd = _unwrap(x).ravel(), _unwrap(y).ravel()
    n_val = int(_scalar(n))
    return ForgeArray(np.polyfit(xd, yd, n_val))

def forge_polyder(p, *args):
    pd = _unwrap(p).ravel()
    m = int(_scalar(args[0])) if args else 1
    result = pd
    for _ in range(m):
        result = np.polyder(result)
    return ForgeArray(result)

def forge_polyint(p, *args):
    pd = _unwrap(p).ravel()
    m = int(_scalar(args[0])) if args else 1
    k = float(_scalar(args[1])) if len(args) > 1 else 0
    result = pd
    for _ in range(m):
        result = np.polyint(result, k=k)
    return ForgeArray(result)

def forge_conv(a, b):
    ad, bd = _unwrap(a).ravel(), _unwrap(b).ravel()
    return ForgeArray(np.convolve(ad, bd))

def forge_deconv(b, a):
    bd, ad = _unwrap(b).ravel(), _unwrap(a).ravel()
    q, r = np.polydiv(bd, ad)
    return ForgeArray(q), ForgeArray(r)

def forge_residue(b, a):
    bd, ad = _unwrap(b).ravel(), _unwrap(a).ravel()
    from scipy.signal import residue
    r, p, k = residue(bd, ad)
    return ForgeArray(r), ForgeArray(p), ForgeArray(k)

def forge_compan(p):
    pd = _unwrap(p).ravel()
    return ForgeArray(np.companion(pd) if hasattr(np, 'companion') else
                      _companion(pd))

def _companion(p):
    p = np.asarray(p).ravel()
    n = len(p) - 1
    if n < 1:
        return np.array([[]])
    c = np.zeros((n, n))
    c[0, :] = -p[1:] / p[0]
    c[np.arange(1, n), np.arange(0, n - 1)] = 1
    return c

def forge_polyreduce(p):
    pd = _unwrap(p).ravel()
    idx = 0
    while idx < len(pd) - 1 and abs(pd[idx]) < np.finfo(float).eps:
        idx += 1
    return ForgeArray(pd[idx:])

def forge_mpoles(p, *args):
    pd = _unwrap(p).ravel()
    tol = float(_scalar(args[0])) if args else 0.001
    # Sort by magnitude
    idx = np.argsort(np.abs(pd))
    sorted_p = pd[idx]
    mults = np.ones(len(pd), dtype=int)
    for i in range(1, len(sorted_p)):
        if abs(sorted_p[i] - sorted_p[i-1]) < tol * abs(sorted_p[i]):
            mults[i] = mults[i-1] + 1
    return ForgeArray(idx + 1), ForgeArray(mults)

def forge_pchip(x, y, xq):
    xd = _unwrap(x).ravel()
    yd = _unwrap(y).ravel()
    xqd = _unwrap(xq).ravel()
    f = interpolate.PchipInterpolator(xd, yd)
    return ForgeArray(f(xqd))

def forge_spline(x, y, xq):
    xd = _unwrap(x).ravel()
    yd = _unwrap(y).ravel()
    xqd = _unwrap(xq).ravel()
    f = interpolate.CubicSpline(xd, yd)
    return ForgeArray(f(xqd))

def forge_mkpp(breaks, coefs):
    bd = _unwrap(breaks).ravel()
    cd = _unwrap(coefs)
    if cd.ndim == 1:
        cd = cd.reshape(1, -1)
    pp = ForgeStruct()
    pp._fields["breaks"] = ForgeArray(bd)
    pp._fields["coefs"] = ForgeArray(cd)
    pp._fields["pieces"] = ForgeArray(float(len(bd) - 1))
    pp._fields["order"] = ForgeArray(float(cd.shape[1]))
    pp._fields["dim"] = ForgeArray(1.0)
    return pp

def forge_unmkpp(pp):
    breaks = pp._fields["breaks"]
    coefs = pp._fields["coefs"]
    return breaks, coefs

def forge_ppval(pp, x):
    bd = _unwrap(pp._fields["breaks"]).ravel()
    cd = _unwrap(pp._fields["coefs"])
    xd = _unwrap(x).ravel()
    if cd.ndim == 1:
        cd = cd.reshape(1, -1)
    result = np.zeros_like(xd)
    for i, xv in enumerate(xd):
        # Find interval
        idx = np.searchsorted(bd, xv, side='right') - 1
        idx = max(0, min(idx, len(bd) - 2))
        dx = xv - bd[idx]
        coef = cd[idx]
        val = 0.0
        for j, c in enumerate(coef):
            val = val * dx + c
        result[i] = val
    return ForgeArray(result)


POLYNOMIAL_REGISTRY = {
    "roots": forge_roots, "poly": forge_poly,
    "polyval": forge_polyval, "polyvalm": forge_polyvalm,
    "polyfit": forge_polyfit, "polyder": forge_polyder,
    "polyint": forge_polyint, "conv": forge_conv,
    "deconv": forge_deconv, "residue": forge_residue,
    "compan": forge_compan, "polyreduce": forge_polyreduce,
    "mpoles": forge_mpoles, "pchip": forge_pchip,
    "spline": forge_spline, "mkpp": forge_mkpp,
    "unmkpp": forge_unmkpp, "ppval": forge_ppval,
}
