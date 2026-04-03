# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Statistics Toolbox for Forge — Octave-compatible functions.

Implements 49 Octave statistics functions: descriptive statistics,
correlation, moving-window functions, distribution helpers, and ranking.

Backend: NumPy + SciPy (scipy.stats).

SRS trace: SRS-FUNC-STATS
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

# ── ForgeArray interop ───────────────────────────────────────────
try:
    from forge.engine.types import ForgeArray, _unwrap
except ImportError:
    ForgeArray = np.ndarray

    def _unwrap(x):
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x, dtype=np.float64)

try:
    from forge.engine.containers import ForgeChar
except ImportError:
    ForgeChar = str


def _fa(x):
    """Wrap result as ForgeArray."""
    if type(x) is np.ndarray and x.ndim >= 2:
        return ForgeArray._from_ndarray(x)
    arr = np.asarray(x)
    if ForgeArray is np.ndarray:
        return arr
    return ForgeArray(arr)


def _ensure_float(x):
    data = x._data if isinstance(x, ForgeArray) else _unwrap(x)
    if type(data) is np.ndarray and data.dtype == np.float64:
        return data  # Already float64 ndarray, no copy needed
    return np.asarray(data, dtype=np.float64)


def _scalar(x):
    if isinstance(x, ForgeArray):
        d = x.data if hasattr(x, 'data') else np.asarray(x)
        return d.flat[0].item() if d.size == 1 else d
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.flat[0].item()
    return x


def _first_nonsingleton(data):
    """Return first non-singleton axis (0-based), or 0 if all singleton."""
    for i, s in enumerate(data.shape):
        if s > 1:
            return i
    return 0


# ═══════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════

def bounds(x):
    """Return [min, max] of data.

    B = bounds(X)
    """
    data = _ensure_float(x)
    return _fa(np.array([data.min(), data.max()]))


def center(x, *args):
    """Center data by subtracting the mean.

    Y = center(X)
    Y = center(X, DIM)
    """
    data = _ensure_float(x)
    dim = int(_scalar(args[0])) - 1 if args else _first_nonsingleton(data)
    m = np.mean(data, axis=dim, keepdims=True)
    return _fa(data - m)


def iqr(x, *args):
    """Interquartile range.

    R = iqr(X)
    R = iqr(X, DIM)
    """
    data = _ensure_float(x)
    dim = int(_scalar(args[0])) - 1 if args else None
    q75 = np.percentile(data, 75, axis=dim)
    q25 = np.percentile(data, 25, axis=dim)
    return _fa(q75 - q25)


def kurtosis(x, *args):
    """Kurtosis of data.

    K = kurtosis(X)
    K = kurtosis(X, FLAG)
    K = kurtosis(X, FLAG, DIM)

    FLAG=1 (default): sample kurtosis (bias-corrected).
    FLAG=0: population kurtosis.
    """
    data = _ensure_float(x)
    flag = int(_scalar(args[0])) if len(args) > 0 else 1
    dim = int(_scalar(args[1])) - 1 if len(args) > 1 else _first_nonsingleton(data)
    bias = (flag == 0)
    return _fa(sp_stats.kurtosis(data, axis=dim, fisher=True, bias=bias))


def mad(x, *args):
    """Median absolute deviation.

    M = mad(X)
    M = mad(X, FLAG)
    M = mad(X, FLAG, DIM)

    FLAG=0 (default): MAD from median.
    FLAG=1: MAD from mean (mean absolute deviation).
    """
    data = _ensure_float(x)
    flag = int(_scalar(args[0])) if len(args) > 0 else 0
    dim = int(_scalar(args[1])) - 1 if len(args) > 1 else _first_nonsingleton(data)
    if flag == 1:
        # Mean absolute deviation
        m = np.mean(data, axis=dim, keepdims=True)
        return _fa(np.mean(np.abs(data - m), axis=dim))
    else:
        # Median absolute deviation
        med = np.median(data, axis=dim, keepdims=True)
        return _fa(np.median(np.abs(data - med), axis=dim))


def mape(y_true, y_pred):
    """Mean absolute percentage error.

    E = mape(Y_TRUE, Y_PRED)
    """
    yt = _ensure_float(y_true)
    yp = _ensure_float(y_pred)
    mask = yt != 0
    return _fa(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def mean(x, *args):
    """Arithmetic mean.

    M = mean(X)
    M = mean(X, DIM)
    M = mean(X, 'all')
    """
    data = _ensure_float(x)
    if args and isinstance(args[0], str) and args[0].lower() == 'all':
        return _fa(np.mean(data))
    dim = int(_scalar(args[0])) - 1 if args else _first_nonsingleton(data)
    return _fa(np.mean(data, axis=dim))


def meansq(x, *args):
    """Mean squared value.

    M = meansq(X)
    M = meansq(X, DIM)
    """
    data = _ensure_float(x)
    dim = int(_scalar(args[0])) - 1 if args else _first_nonsingleton(data)
    return _fa(np.mean(data ** 2, axis=dim))


def median(x, *args):
    """Median value.

    M = median(X)
    M = median(X, DIM)
    """
    data = _ensure_float(x)
    dim = int(_scalar(args[0])) - 1 if args else _first_nonsingleton(data)
    return _fa(np.median(data, axis=dim))


def mode(x, *args):
    """Most frequent value.

    M = mode(X)
    M = mode(X, DIM)
    """
    data = _ensure_float(x)
    dim = int(_scalar(args[0])) - 1 if args else _first_nonsingleton(data)
    result = sp_stats.mode(data, axis=dim, keepdims=False)
    return _fa(result.mode)


def moment(x, p, *args):
    """Central moment of specified order.

    M = moment(X, P)
    M = moment(X, P, DIM)
    """
    data = _ensure_float(x)
    p = int(_scalar(p))
    dim = int(_scalar(args[0])) - 1 if args else _first_nonsingleton(data)
    return _fa(sp_stats.moment(data, moment=p, axis=dim))


def normalize(x, *args):
    """Normalize data (z-score by default).

    Y = normalize(X)
    Y = normalize(X, 'center')
    Y = normalize(X, 'zscore')
    Y = normalize(X, 'range')
    Y = normalize(X, METHOD, DIM)
    """
    data = _ensure_float(x)
    method = 'zscore'
    dim = _first_nonsingleton(data)

    if len(args) > 0 and isinstance(args[0], str):
        method = args[0].lower()
    if len(args) > 1:
        dim = int(_scalar(args[1])) - 1

    if method == 'zscore':
        m = np.mean(data, axis=dim, keepdims=True)
        s = np.std(data, axis=dim, keepdims=True, ddof=1)
        s[s == 0] = 1
        return _fa((data - m) / s)
    elif method == 'center':
        m = np.mean(data, axis=dim, keepdims=True)
        return _fa(data - m)
    elif method == 'range':
        lo = np.min(data, axis=dim, keepdims=True)
        hi = np.max(data, axis=dim, keepdims=True)
        rng = hi - lo
        rng[rng == 0] = 1
        return _fa((data - lo) / rng)
    else:
        raise ValueError(f"normalize: unknown method '{method}'")


def prctile(x, p, *args):
    """Percentiles of data.

    Y = prctile(X, P)
    Y = prctile(X, P, DIM)
    """
    data = _ensure_float(x)
    p_val = _ensure_float(p)
    dim = int(_scalar(args[0])) - 1 if args else _first_nonsingleton(data)
    return _fa(np.percentile(data, p_val, axis=dim))


def quantile(x, p, *args):
    """Quantiles of data.

    Y = quantile(X, P)
    Y = quantile(X, P, DIM)
    P is in [0,1].
    """
    data = _ensure_float(x)
    p_val = _ensure_float(p)
    dim = int(_scalar(args[0])) - 1 if args else _first_nonsingleton(data)
    return _fa(np.quantile(data, p_val, axis=dim))


def range_stat(x, *args):
    """Range of data (max - min).

    R = range(X)
    R = range(X, DIM)
    """
    data = _ensure_float(x)
    dim = int(_scalar(args[0])) - 1 if args else _first_nonsingleton(data)
    return _fa(np.max(data, axis=dim) - np.min(data, axis=dim))


def ranks(x, *args):
    """Rank of each element.

    R = ranks(X)
    R = ranks(X, DIM)
    Ties receive average rank.
    """
    data = _ensure_float(x)
    dim = int(_scalar(args[0])) - 1 if args else _first_nonsingleton(data)
    result = np.apply_along_axis(sp_stats.rankdata, dim, data)
    return _fa(result)


def rms(x, *args):
    """Root mean square.

    R = rms(X)
    R = rms(X, DIM)
    """
    data = _ensure_float(x)
    dim = int(_scalar(args[0])) - 1 if args else _first_nonsingleton(data)
    return _fa(np.sqrt(np.mean(data ** 2, axis=dim)))


def rmse(y_true, y_pred, *args):
    """Root mean squared error.

    E = rmse(Y_TRUE, Y_PRED)
    """
    yt = _ensure_float(y_true)
    yp = _ensure_float(y_pred)
    return _fa(np.sqrt(np.mean((yt - yp) ** 2)))


def run_count(x, *args):
    """Count runs of identical values.

    [COUNTS, VALUES] = run_count(X)
    """
    data = _ensure_float(x).ravel()
    if data.size == 0:
        return _fa(np.array([])), _fa(np.array([]))
    changes = np.concatenate(([0], np.where(np.diff(data) != 0)[0] + 1, [len(data)]))
    counts = np.diff(changes)
    values = data[changes[:-1]]
    return _fa(counts.astype(np.float64)), _fa(values)


def runlength(x):
    """Run-length encoding.

    [VALUES, LENGTHS, POSITIONS] = runlength(X)
    """
    data = _ensure_float(x).ravel()
    if data.size == 0:
        return _fa(np.array([])), _fa(np.array([])), _fa(np.array([]))
    changes = np.concatenate(([0], np.where(np.diff(data) != 0)[0] + 1, [len(data)]))
    lengths = np.diff(changes)
    values = data[changes[:-1]]
    positions = changes[:-1] + 1  # 1-based
    return _fa(values), _fa(lengths.astype(np.float64)), _fa(positions.astype(np.float64))


def skewness(x, *args):
    """Skewness of data.

    S = skewness(X)
    S = skewness(X, FLAG)
    S = skewness(X, FLAG, DIM)
    """
    data = _ensure_float(x)
    flag = int(_scalar(args[0])) if len(args) > 0 else 1
    dim = int(_scalar(args[1])) - 1 if len(args) > 1 else _first_nonsingleton(data)
    bias = (flag == 0)
    return _fa(sp_stats.skew(data, axis=dim, bias=bias))


def statistics(x):
    """Compute a vector of common statistics.

    S = statistics(X)
    Returns: [min, 1st quartile, median, 3rd quartile, max, mean, std, skew, kurtosis]
    """
    data = _ensure_float(x).ravel()
    result = np.array([
        np.min(data),
        np.percentile(data, 25),
        np.median(data),
        np.percentile(data, 75),
        np.max(data),
        np.mean(data),
        np.std(data, ddof=1),
        float(sp_stats.skew(data)),
        float(sp_stats.kurtosis(data, fisher=True)),
    ])
    return _fa(result)


def std(x, *args):
    """Standard deviation.

    S = std(X)
    S = std(X, W)
    S = std(X, W, DIM)

    W=0 (default): normalize by N-1 (sample).
    W=1: normalize by N (population).
    """
    data = _ensure_float(x)
    w = int(_scalar(args[0])) if len(args) > 0 else 0
    dim = int(_scalar(args[1])) - 1 if len(args) > 1 else _first_nonsingleton(data)
    ddof = 0 if w == 1 else 1
    return _fa(np.std(data, axis=dim, ddof=ddof))


def var(x, *args):
    """Variance.

    V = var(X)
    V = var(X, W)
    V = var(X, W, DIM)

    W=0 (default): normalize by N-1.
    W=1: normalize by N.
    """
    data = _ensure_float(x)
    w = int(_scalar(args[0])) if len(args) > 0 else 0
    dim = int(_scalar(args[1])) - 1 if len(args) > 1 else _first_nonsingleton(data)
    ddof = 0 if w == 1 else 1
    return _fa(np.var(data, axis=dim, ddof=ddof))


def zscore(x, *args):
    """Standardized z-scores.

    Z = zscore(X)
    Z = zscore(X, FLAG)
    Z = zscore(X, FLAG, DIM)
    """
    data = _ensure_float(x)
    flag = int(_scalar(args[0])) if len(args) > 0 else 0
    dim = int(_scalar(args[1])) - 1 if len(args) > 1 else _first_nonsingleton(data)
    ddof = 0 if flag == 1 else 1
    m = np.mean(data, axis=dim, keepdims=True)
    s = np.std(data, axis=dim, keepdims=True, ddof=ddof)
    s[s == 0] = 1
    return _fa((data - m) / s)


# ═══════════════════════════════════════════════════════════════════
# 2. CORRELATION & COVARIANCE
# ═══════════════════════════════════════════════════════════════════

def corr(x, y=None):
    """Correlation coefficients.

    R = corr(X)
    R = corr(X, Y)
    """
    data_x = _ensure_float(x)
    if y is not None:
        data_y = _ensure_float(y)
        if data_x.ndim == 1 and data_y.ndim == 1:
            combined = np.column_stack([data_x, data_y])
            return _fa(np.corrcoef(combined, rowvar=False))
        combined = np.column_stack([data_x, data_y]) if data_x.ndim == 1 else data_x
        return _fa(np.corrcoef(combined, rowvar=False))
    if data_x.ndim == 1:
        return _fa(np.array([[1.0]]))
    return _fa(np.corrcoef(data_x, rowvar=False))


def corrcoef(x, y=None):
    """Correlation coefficient matrix (alias for corr).

    R = corrcoef(X)
    R = corrcoef(X, Y)
    """
    data_x = _ensure_float(x)
    if y is not None:
        data_y = _ensure_float(y)
        return _fa(np.corrcoef(data_x, data_y))
    return _fa(np.corrcoef(data_x))


def corrcov(C):
    """Convert covariance matrix to correlation matrix.

    R = corrcov(C)
    """
    C = _ensure_float(C)
    d = np.sqrt(np.diag(C))
    d[d == 0] = 1
    D_inv = np.diag(1.0 / d)
    return _fa(D_inv @ C @ D_inv)


def cov(x, *args):
    """Covariance matrix.

    C = cov(X)
    C = cov(X, Y)
    C = cov(X, NORM)   where NORM is 0 (N-1) or 1 (N)
    """
    data_x = _ensure_float(x)
    # For a vector input, treat as column of observations (Octave convention)
    if data_x.ndim <= 2 and min(data_x.shape) <= 1:
        x_flat = data_x.ravel()
    else:
        x_flat = None

    if len(args) == 0:
        if x_flat is not None:
            return _fa(np.cov(x_flat, ddof=1))
        return _fa(np.cov(data_x, rowvar=False))
    arg = args[0]
    # Check if arg is a scalar 0 or 1 (normalization flag)
    try:
        sv = _scalar(arg)
        if isinstance(sv, (int, float)) and sv in (0, 1):
            ddof = int(1 - sv)
            if x_flat is not None:
                return _fa(np.cov(x_flat, ddof=ddof))
            return _fa(np.cov(data_x, rowvar=False, ddof=ddof))
    except Exception:
        pass
    # Otherwise treat as Y
    data_y = _ensure_float(arg)
    # When both are vectors, flatten so np.cov treats them as two variables
    x_v = x_flat if x_flat is not None else data_x.ravel()
    if data_y.ndim <= 2 and min(data_y.shape) <= 1:
        return _fa(np.cov(x_v, data_y.ravel()))
    return _fa(np.cov(data_x, data_y, rowvar=False))


def kendall(x, y):
    """Kendall rank correlation coefficient.

    TAU = kendall(X, Y)
    """
    xd = _ensure_float(x).ravel()
    yd = _ensure_float(y).ravel()
    tau, _ = sp_stats.kendalltau(xd, yd)
    return _fa(np.float64(tau))


def spearman(x, y=None):
    """Spearman rank correlation coefficient.

    RHO = spearman(X, Y)
    RHO = spearman(X)       (correlation matrix of columns)
    """
    data_x = _ensure_float(x)
    if y is not None:
        data_y = _ensure_float(y)
        if data_x.ndim == 1 and data_y.ndim == 1:
            rho, _ = sp_stats.spearmanr(data_x, data_y)
            return _fa(np.float64(rho))
        combined = np.column_stack([data_x, data_y])
        rho, _ = sp_stats.spearmanr(combined)
        return _fa(rho)
    if data_x.ndim == 1:
        return _fa(np.array([[1.0]]))
    rho, _ = sp_stats.spearmanr(data_x)
    return _fa(rho)


# ═══════════════════════════════════════════════════════════════════
# 3. DISCRETE DISTRIBUTION HELPERS
# ═══════════════════════════════════════════════════════════════════

def discrete_cdf(x, values, probs):
    """CDF of a discrete distribution.

    P = discrete_cdf(X, VALUES, PROBS)
    """
    xd = _ensure_float(x).ravel()
    vals = _ensure_float(values).ravel()
    prbs = _ensure_float(probs).ravel()
    # Sort by value
    order = np.argsort(vals)
    vals = vals[order]
    prbs = prbs[order]
    cumprobs = np.cumsum(prbs)
    result = np.zeros_like(xd)
    for i, xi in enumerate(xd):
        idx = np.searchsorted(vals, xi, side='right')
        result[i] = cumprobs[idx - 1] if idx > 0 else 0.0
    return _fa(result)


def discrete_inv(p, values, probs):
    """Inverse CDF (quantile function) of a discrete distribution.

    X = discrete_inv(P, VALUES, PROBS)
    """
    pd = _ensure_float(p).ravel()
    vals = _ensure_float(values).ravel()
    prbs = _ensure_float(probs).ravel()
    order = np.argsort(vals)
    vals = vals[order]
    prbs = prbs[order]
    cumprobs = np.cumsum(prbs)
    result = np.zeros_like(pd)
    for i, pi in enumerate(pd):
        idx = np.searchsorted(cumprobs, pi)
        idx = min(idx, len(vals) - 1)
        result[i] = vals[idx]
    return _fa(result)


def discrete_pdf(x, values, probs):
    """PDF (PMF) of a discrete distribution.

    P = discrete_pdf(X, VALUES, PROBS)
    """
    xd = _ensure_float(x).ravel()
    vals = _ensure_float(values).ravel()
    prbs = _ensure_float(probs).ravel()
    result = np.zeros_like(xd)
    for i, xi in enumerate(xd):
        matches = np.where(np.abs(vals - xi) < 1e-12)[0]
        if len(matches) > 0:
            result[i] = prbs[matches[0]]
    return _fa(result)


def discrete_rnd(values, probs, *args):
    """Random samples from a discrete distribution.

    X = discrete_rnd(VALUES, PROBS, N)
    X = discrete_rnd(VALUES, PROBS, R, C)
    """
    vals = _ensure_float(values).ravel()
    prbs = _ensure_float(probs).ravel()
    prbs = prbs / prbs.sum()  # normalize

    if len(args) == 0:
        shape = (1,)
    elif len(args) == 1:
        shape = (int(_scalar(args[0])),)
    else:
        shape = tuple(int(_scalar(a)) for a in args)

    idx = np.random.choice(len(vals), size=shape, p=prbs)
    return _fa(vals[idx])


# ═══════════════════════════════════════════════════════════════════
# 4. EMPIRICAL DISTRIBUTION HELPERS
# ═══════════════════════════════════════════════════════════════════

def empirical_cdf(x, data):
    """Empirical CDF.

    P = empirical_cdf(X, DATA)
    """
    xd = _ensure_float(x).ravel()
    dd = _ensure_float(data).ravel()
    dd_sorted = np.sort(dd)
    n = len(dd_sorted)
    result = np.searchsorted(dd_sorted, xd, side='right').astype(np.float64) / n
    return _fa(result)


def empirical_inv(p, data):
    """Empirical inverse CDF (quantile).

    X = empirical_inv(P, DATA)
    """
    pd = _ensure_float(p).ravel()
    dd = np.sort(_ensure_float(data).ravel())
    n = len(dd)
    idx = np.clip(np.ceil(pd * n).astype(int) - 1, 0, n - 1)
    return _fa(dd[idx])


def empirical_pdf(x, data):
    """Empirical PDF (kernel density estimate).

    P = empirical_pdf(X, DATA)
    Uses Gaussian KDE.
    """
    xd = _ensure_float(x).ravel()
    dd = _ensure_float(data).ravel()
    if len(dd) < 2:
        return _fa(np.zeros_like(xd))
    kde = sp_stats.gaussian_kde(dd)
    return _fa(kde(xd))


def empirical_rnd(data, *args):
    """Random samples from empirical distribution.

    X = empirical_rnd(DATA, N)
    X = empirical_rnd(DATA, R, C)
    """
    dd = _ensure_float(data).ravel()
    if len(args) == 0:
        shape = (1,)
    elif len(args) == 1:
        shape = (int(_scalar(args[0])),)
    else:
        shape = tuple(int(_scalar(a)) for a in args)
    idx = np.random.randint(0, len(dd), size=shape)
    return _fa(dd[idx])


# ═══════════════════════════════════════════════════════════════════
# 5. HISTOGRAM
# ═══════════════════════════════════════════════════════════════════

def hist(x, *args):
    """Histogram bin counts (like Octave hist).

    N = hist(X)
    N = hist(X, NBINS)
    N = hist(X, X_CENTERS)
    [N, X_CENTERS] = hist(X, ...)

    With no output arguments, Octave would plot; here we just return counts.
    Default number of bins is 10.
    """
    data = _ensure_float(x).ravel()
    nbins = 10
    edges_given = False

    if len(args) >= 1:
        arg1 = _ensure_float(args[0])
        if arg1.size == 1:
            nbins = int(arg1.flat[0])
        else:
            # arg1 is vector of bin centers
            centers = arg1.ravel()
            # Convert centers to edges
            edges_arr = np.empty(len(centers) + 1)
            edges_arr[0] = -np.inf
            for i in range(len(centers) - 1):
                edges_arr[i + 1] = (centers[i] + centers[i + 1]) / 2.0
            edges_arr[-1] = np.inf
            counts, _ = np.histogram(data, bins=edges_arr)
            return _fa(counts.astype(np.float64)), _fa(centers)

    counts, bin_edges = np.histogram(data, bins=nbins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return _fa(counts.astype(np.float64)), _fa(centers)


def histc(x, edges):
    """Histogram bin counts (like Octave histc).

    N = histc(X, EDGES)
    [N, IDX] = histc(X, EDGES)

    Counts how many elements of X fall in each bin defined by EDGES.
    The last bin includes the right edge.
    """
    data = _ensure_float(x).ravel()
    edges = _ensure_float(edges).ravel()
    n_bins = len(edges) - 1
    counts = np.zeros(len(edges), dtype=np.float64)
    idx = np.zeros(len(data), dtype=np.float64)

    for i, val in enumerate(data):
        if val < edges[0]:
            idx[i] = 0
            continue
        if val >= edges[-1]:
            if val == edges[-1]:
                counts[-1] += 1
                idx[i] = len(edges)
            else:
                idx[i] = 0
            continue
        bin_idx = np.searchsorted(edges, val, side='right') - 1
        bin_idx = max(0, min(bin_idx, len(edges) - 1))
        counts[bin_idx] += 1
        idx[i] = bin_idx + 1  # 1-based

    return _fa(counts), _fa(idx)


# ═══════════════════════════════════════════════════════════════════
# 6. MOVING-WINDOW FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _movfun_core(x, k, func, dim=None):
    """Core moving-window function."""
    data = _ensure_float(x)
    k = int(_scalar(k))
    if dim is None:
        dim = _first_nonsingleton(data)

    n = data.shape[dim]
    half = k // 2
    result = np.empty_like(data)

    # Move along the given axis
    slices_in = [slice(None)] * data.ndim
    slices_out = [slice(None)] * data.ndim

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i - half + k)
        slices_in[dim] = slice(lo, hi)
        slices_out[dim] = i
        result[tuple(slices_out)] = func(data[tuple(slices_in)], axis=dim)

    return _fa(result)


def movmad(x, k, *args):
    """Moving median absolute deviation.

    Y = movmad(X, K)
    """
    def _mad_func(chunk, axis):
        med = np.median(chunk, axis=axis, keepdims=True)
        return np.median(np.abs(chunk - med), axis=axis)
    return _movfun_core(x, k, _mad_func)


def movmax(x, k, *args):
    """Moving maximum.

    Y = movmax(X, K)
    """
    return _movfun_core(x, k, np.max)


def movmean(x, k, *args):
    """Moving mean.

    Y = movmean(X, K)
    """
    return _movfun_core(x, k, np.mean)


def movmedian(x, k, *args):
    """Moving median.

    Y = movmedian(X, K)
    """
    return _movfun_core(x, k, np.median)


def movmin(x, k, *args):
    """Moving minimum.

    Y = movmin(X, K)
    """
    return _movfun_core(x, k, np.min)


def movprod(x, k, *args):
    """Moving product.

    Y = movprod(X, K)
    """
    return _movfun_core(x, k, np.prod)


def movstd(x, k, *args):
    """Moving standard deviation.

    Y = movstd(X, K)
    """
    def _std_func(chunk, axis):
        return np.std(chunk, axis=axis, ddof=1)
    return _movfun_core(x, k, _std_func)


def movsum(x, k, *args):
    """Moving sum.

    Y = movsum(X, K)
    """
    return _movfun_core(x, k, np.sum)


def movvar(x, k, *args):
    """Moving variance.

    Y = movvar(X, K)
    """
    def _var_func(chunk, axis):
        return np.var(chunk, axis=axis, ddof=1)
    return _movfun_core(x, k, _var_func)


# ═══════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════


# ── Poisson distribution ──────────────────────────────────────────

def poisspdf(x, lambda_param):
    """Poisson probability mass function.

    Y = poisspdf(X, LAMBDA)
    """
    xd = _ensure_float(x)
    lam = _ensure_float(lambda_param)
    return _fa(sp_stats.poisson.pmf(xd, lam))


# ── Binomial distribution ────────────────────────────────────────

def binopdf(x, n, p):
    """Binomial probability mass function.

    Y = binopdf(X, N, P)
    """
    xd = _ensure_float(x)
    nd = _ensure_float(n)
    pd = _ensure_float(p)
    return _fa(sp_stats.binom.pmf(xd, nd, pd))


STATISTICS_REGISTRY = {
    # ── Descriptive statistics ────────────────────────────────────
    'bounds':           bounds,
    'center':           center,
    'iqr':              iqr,
    'kurtosis':         kurtosis,
    'mad':              mad,
    'mape':             mape,
    'mean':             mean,
    'meansq':           meansq,
    'median':           median,
    'mode':             mode,
    'moment':           moment,
    'normalize':        normalize,
    'prctile':          prctile,
    'quantile':         quantile,
    'range':            range_stat,
    'ranks':            ranks,
    'rms':              rms,
    'rmse':             rmse,
    'run_count':        run_count,
    'runlength':        runlength,
    'skewness':         skewness,
    'statistics':       statistics,
    'std':              std,
    'var':              var,
    'zscore':           zscore,
    # ── Correlation & covariance ──────────────────────────────────
    'corr':             corr,
    'corrcoef':         corrcoef,
    'corrcov':          corrcov,
    'cov':              cov,
    'kendall':          kendall,
    'spearman':         spearman,
    # ── Discrete distribution ─────────────────────────────────────
    'discrete_cdf':     discrete_cdf,
    'discrete_inv':     discrete_inv,
    'discrete_pdf':     discrete_pdf,
    'discrete_rnd':     discrete_rnd,
    # ── Empirical distribution ────────────────────────────────────
    'empirical_cdf':    empirical_cdf,
    'empirical_inv':    empirical_inv,
    'empirical_pdf':    empirical_pdf,
    'empirical_rnd':    empirical_rnd,
    # ── Histogram ─────────────────────────────────────────────────
    'hist':             hist,
    'histc':            histc,
    # ── Moving-window functions ───────────────────────────────────
    'movmad':           movmad,
    'movmax':           movmax,
    'movmean':          movmean,
    'movmedian':        movmedian,
    'movmin':           movmin,
    'movprod':          movprod,
    'movstd':           movstd,
    'movsum':           movsum,
    'movvar':           movvar,
    # ── Poisson distribution ───────────────────────────────────────
    'poisspdf':         poisspdf,
    # ── Binomial distribution ─────────────────────────────────────
    'binopdf':          binopdf,
}

