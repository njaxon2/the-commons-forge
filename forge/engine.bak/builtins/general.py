# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""General toolbox functions.

SRS trace: SRS-FUNC-001
"""
import numpy as np
from scipy import integrate, interpolate
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar


def _wrap(x):
    return ForgeArray(np.asarray(x))

def _scalar(x):
    if isinstance(x, ForgeArray):
        d = x.data
        if d.ndim == 0:
            return d.item()
        if d.size == 1:
            return d.flat[0].item()
        return d
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        if x.size == 1:
            return x.flat[0].item()
        return x
    return x

def forge_cart2pol(*args):
    x, y = _unwrap(args[0]), _unwrap(args[1])
    theta = np.arctan2(y, x)
    r = np.hypot(x, y)
    if len(args) > 2:
        return ForgeArray(theta), ForgeArray(r), args[2]
    return ForgeArray(theta), ForgeArray(r)

def forge_pol2cart(*args):
    theta, r = _unwrap(args[0]), _unwrap(args[1])
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    if len(args) > 2:
        return ForgeArray(x), ForgeArray(y), args[2]
    return ForgeArray(x), ForgeArray(y)

def forge_cart2sph(*args):
    x, y, z = _unwrap(args[0]), _unwrap(args[1]), _unwrap(args[2])
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.hypot(x, y))
    return ForgeArray(az), ForgeArray(el), ForgeArray(r)

def forge_sph2cart(*args):
    az, el, r = _unwrap(args[0]), _unwrap(args[1]), _unwrap(args[2])
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return ForgeArray(x), ForgeArray(y), ForgeArray(z)

def forge_circshift(x, k, *args):
    data = _unwrap(x)
    k_val = int(_scalar(k))
    if args:
        dim = int(_scalar(args[0])) - 1
    else:
        dim = -1 if data.ndim == 2 and data.shape[0] == 1 else 0
    return ForgeArray(np.roll(data, k_val, axis=dim))

def forge_flip(x, *args):
    data = _unwrap(x)
    if args:
        dim = int(_scalar(args[0])) - 1
        return ForgeArray(np.flip(data, axis=dim))
    return ForgeArray(np.flip(data))

def forge_shiftdim(x, n=None):
    data = _unwrap(x)
    if n is not None:
        n_val = int(_scalar(n))
    else:
        n_val = None
    if n_val is None:
        shape = data.shape
        idx = 0
        while idx < len(shape) and shape[idx] == 1:
            idx += 1
        return ForgeArray(data.reshape(shape[idx:]))
    if n_val > 0:
        axes = list(range(n_val, data.ndim)) + list(range(n_val))
        return ForgeArray(np.transpose(data, axes))
    else:
        new_shape = (1,) * (-n_val) + data.shape
        return ForgeArray(data.reshape(new_shape))

def forge_sortrows(x, *args):
    data = _unwrap(x)
    if data.ndim < 2:
        data = data.reshape(-1, 1)
    col = int(_scalar(args[0])) - 1 if args else 0
    idx = np.argsort(data[:, col], kind='stable')
    return ForgeArray(data[idx])

def forge_repelem(x, *args):
    data = _unwrap(x)
    if len(args) == 1:
        r = int(_scalar(args[0]))
        return ForgeArray(np.repeat(data.ravel(), r))
    r, c = int(_scalar(args[0])), int(_scalar(args[1]))
    return ForgeArray(np.repeat(np.repeat(data, r, axis=0), c, axis=1))

def forge_postpad(x, l, *args):
    data = _unwrap(x).ravel()
    l_val = int(_scalar(l))
    c = float(_scalar(args[0])) if args else 0.0
    if len(data) >= l_val:
        return ForgeArray(data[:l_val])
    return ForgeArray(np.concatenate([data, np.full(l_val - len(data), c)]))

def forge_prepad(x, l, *args):
    data = _unwrap(x).ravel()
    l_val = int(_scalar(l))
    c = float(_scalar(args[0])) if args else 0.0
    if len(data) >= l_val:
        return ForgeArray(data[-l_val:])
    return ForgeArray(np.concatenate([np.full(l_val - len(data), c), data]))

def forge_rescale(x, *args):
    data = _unwrap(x).astype(float)
    lo, hi = np.min(data), np.max(data)
    if lo == hi:
        return ForgeArray(np.full_like(data, 0.5))
    norm = (data - lo) / (hi - lo)
    if args and len(args) >= 2:
        a, b = float(_scalar(args[0])), float(_scalar(args[1]))
        return ForgeArray(norm * (b - a) + a)
    return ForgeArray(norm)

def forge_trapz(y, *args):
    yd = _unwrap(y)
    if args:
        xd = _unwrap(args[0])
        return ForgeArray(np.trapezoid(yd, xd))
    return ForgeArray(np.trapezoid(yd))

def forge_cumtrapz(y, *args):
    yd = _unwrap(y).ravel()
    if args:
        xd = _unwrap(args[0]).ravel()
        dx = np.diff(xd)
        return ForgeArray(np.cumsum(0.5 * (yd[:-1] + yd[1:]) * dx))
    return ForgeArray(np.cumsum(0.5 * (yd[:-1] + yd[1:])))

def forge_gradient(f, *args):
    fd = _unwrap(f)
    if fd.ndim == 2 and fd.shape[0] == 1:
        fd = fd.ravel()
    if args:
        h = float(_scalar(args[0]))
        return ForgeArray(np.gradient(fd, h))
    return ForgeArray(np.gradient(fd))

def forge_integral(fun, a, b):
    a_val, b_val = float(_scalar(a)), float(_scalar(b))
    result, _ = integrate.quad(fun, a_val, b_val)
    return ForgeArray(result)

def forge_interp1(x, y, xq, *args):
    xd, yd, xqd = _unwrap(x).ravel(), _unwrap(y).ravel(), _unwrap(xq).ravel()
    method = "linear"
    if args:
        m = args[0]
        if isinstance(m, ForgeChar):
            method = m.to_str()
        elif isinstance(m, str):
            method = m
    method_map = {"linear": "linear", "nearest": "nearest",
                  "spline": "cubic", "cubic": "cubic", "pchip": "cubic"}
    scipy_kind = method_map.get(method, "linear")
    f = interpolate.interp1d(xd, yd, kind=scipy_kind, fill_value="extrapolate")
    return ForgeArray(f(xqd))

def forge_deg2rad(x):
    return ForgeArray(np.deg2rad(_unwrap(x)))

def forge_rad2deg(x):
    return ForgeArray(np.rad2deg(_unwrap(x)))

def forge_nextpow2(x):
    v = np.abs(_unwrap(x))
    result = np.ceil(np.log2(np.maximum(v, 1)))
    if result.ndim == 0:
        return ForgeArray(float(result))
    return ForgeArray(result)

def forge_bincoeff(n, k):
    from scipy.special import comb
    return ForgeArray(comb(_unwrap(n), _unwrap(k), exact=False))

def forge_idivide(a, b, *args):
    ad, bd = _unwrap(a), _unwrap(b)
    mode = "fix"
    if args:
        m = args[0]
        mode = m.to_str() if isinstance(m, ForgeChar) else str(m)
    if mode == "floor":
        return ForgeArray(np.floor(ad / bd).astype(int))
    elif mode == "ceil":
        return ForgeArray(np.ceil(ad / bd).astype(int))
    elif mode == "round":
        return ForgeArray(np.round(ad / bd).astype(int))
    else:
        r = np.fix(ad / bd)
    if np.ndim(r) == 0:
        return ForgeArray(float(int(r)))
    return ForgeArray(r.astype(int))

def forge_xor(a, b):
    return ForgeArray(np.logical_xor(_unwrap(a), _unwrap(b)))

def forge_isequal(*args):
    if len(args) < 2:
        return ForgeArray(np.array(True))
    first = _unwrap(args[0])
    for a in args[1:]:
        if not np.array_equal(first, _unwrap(a)):
            return ForgeArray(np.array(False))
    return ForgeArray(np.array(True))

def forge_isequaln(*args):
    if len(args) < 2:
        return ForgeArray(np.array(True))
    first = _unwrap(args[0])
    for a in args[1:]:
        other = _unwrap(a)
        if first.shape != other.shape:
            return ForgeArray(np.array(False))
        mask = np.isnan(first) & np.isnan(other)
        if not np.all((first == other) | mask):
            return ForgeArray(np.array(False))
    return ForgeArray(np.array(True))

def forge_int2str(x):
    v = int(_scalar(x))
    return ForgeChar(str(v))

def forge_polyarea(x, y):
    xd, yd = _unwrap(x).ravel(), _unwrap(y).ravel()
    return ForgeArray(0.5 * np.abs(np.dot(xd, np.roll(yd, 1)) - np.dot(yd, np.roll(xd, 1))))

def forge_rat(x, *args):
    from fractions import Fraction
    v = float(_scalar(x))
    tol = float(_scalar(args[0])) if args else 1e-6
    f = Fraction(v).limit_denominator(int(1/tol))
    return ForgeArray(f.numerator), ForgeArray(f.denominator)

def forge_deal(*args):
    if len(args) == 1:
        return args[0]
    return args

def forge_accumarray(subs, val, *args):
    subs_d = _unwrap(subs).ravel().astype(int)
    val_d = _unwrap(val).ravel()
    sz = int(np.max(subs_d))
    if args:
        sz = max(sz, int(_scalar(args[0])))
    result = np.zeros(sz)
    for i, v in zip(subs_d, val_d):
        result[i - 1] += v
    return ForgeArray(result.ravel())

def forge_logspace(a, b, *args):
    a_val, b_val = float(_scalar(a)), float(_scalar(b))
    n = int(_scalar(args[0])) if args else 50
    return ForgeArray(np.logspace(a_val, b_val, n))

def forge_cplxpair(x, *args):
    data = _unwrap(x).ravel()
    tol = float(_scalar(args[0])) if args else 100 * np.finfo(float).eps
    reals = data[np.abs(data.imag) <= tol * np.abs(data)]
    cpx = data[np.abs(data.imag) > tol * np.abs(data)]
    pairs = []
    used = np.zeros(len(cpx), dtype=bool)
    for i in range(len(cpx)):
        if used[i]:
            continue
        for j in range(i+1, len(cpx)):
            if used[j]:
                continue
            if np.abs(cpx[i] - np.conj(cpx[j])) <= tol * np.abs(cpx[i]):
                neg = cpx[i] if cpx[i].imag < 0 else cpx[j]
                pos = cpx[j] if cpx[i].imag < 0 else cpx[i]
                pairs.extend([neg, pos])
                used[i] = used[j] = True
                break
    result = np.concatenate([np.array(pairs), np.sort(reals.real)]) if pairs else np.sort(reals.real)
    return ForgeArray(result)


GENERAL_REGISTRY = {
    "cart2pol": forge_cart2pol, "pol2cart": forge_pol2cart,
    "cart2sph": forge_cart2sph, "sph2cart": forge_sph2cart,
    "circshift": forge_circshift, "flip": forge_flip,
    "shiftdim": forge_shiftdim, "sortrows": forge_sortrows,
    "repelem": forge_repelem, "postpad": forge_postpad,
    "prepad": forge_prepad, "rescale": forge_rescale,
    "trapz": forge_trapz, "cumtrapz": forge_cumtrapz,
    "gradient": forge_gradient, "integral": forge_integral,
    "interp1": forge_interp1,
    "deg2rad": forge_deg2rad, "rad2deg": forge_rad2deg,
    "nextpow2": forge_nextpow2, "bincoeff": forge_bincoeff,
    "idivide": forge_idivide, "xor": forge_xor,
    "isequal": forge_isequal, "isequaln": forge_isequaln,
    "int2str": forge_int2str, "polyarea": forge_polyarea,
    "rat": forge_rat, "deal": forge_deal,
    "accumarray": forge_accumarray, "logspace": forge_logspace,
    "cplxpair": forge_cplxpair,
}
