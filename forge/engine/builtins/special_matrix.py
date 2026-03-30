# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Special matrices toolbox.

Implements: gallery, hadamard, hankel, hilb, invhilb, magic, pascal,
rosser, toeplitz, vander, wilkinson

SRS trace: SRS-FUNC-001 (Octave-compatible function library)
"""
import numpy as np
import scipy.linalg
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar


def _scalar(x):
    """Extract Python scalar from ForgeArray."""
    if hasattr(x, 'data'):
        d = x.data
        return d.flat[0].item() if d.size >= 1 else d
    if hasattr(x, 'flat'):
        return x.flat[0].item() if x.size >= 1 else x
    return x

def _wrap(x):
    return ForgeArray(np.asarray(x, dtype=float))


# ── Special matrices ─────────────────────────────────────────────

def forge_hadamard(n):
    """Hadamard matrix of order n (must be a power of 2)."""
    n = int(_scalar(n))
    return _wrap(scipy.linalg.hadamard(n).astype(float))


def forge_hankel(c, r=None):
    """Hankel matrix.

    Parameters
    ----------
    c : ForgeArray
        First column of the matrix.
    r : ForgeArray, optional
        Last row of the matrix.  If omitted, zeros are used below the
        anti-diagonal.
    """
    vc = np.asarray(_unwrap(c), dtype=float).ravel()
    vr = None if r is None else np.asarray(_unwrap(r), dtype=float).ravel()
    if vr is None:
        nr = len(vc)
        vr = np.zeros(nr, dtype=float)
        vr[0] = vc[-1]
    # Build Hankel: H[i,j] = c[i+j] for i+j < len(c), else r[i+j-len(c)+1]
    nc = len(vc)
    nrr = len(vr)
    mat = np.zeros((nc, nrr), dtype=float)
    for i in range(nc):
        for j in range(nrr):
            idx = i + j
            if idx < nc:
                mat[i, j] = vc[idx]
            else:
                mat[i, j] = vr[idx - nc + 1] if (idx - nc + 1) < nrr else 0.0
    return _wrap(mat)


def forge_hilb(n):
    """Hilbert matrix of order n.

    H[i,j] = 1 / (i + j + 1)  (0-indexed).
    """
    n = int(_scalar(n))
    return _wrap(scipy.linalg.hilbert(n))


def forge_invhilb(n):
    """Inverse of the Hilbert matrix of order n (exact integer entries)."""
    n = int(_scalar(n))
    return _wrap(scipy.linalg.invhilbert(n))


def forge_magic(n):
    """Magic square of order n (n >= 3).

    Every row, column, and diagonal sums to the same value.
    """
    n = int(_scalar(n))
    if n < 3:
        raise ValueError("magic: N must be at least 3")

    if n % 2 == 1:
        # Odd order — Siamese method
        M = np.zeros((n, n), dtype=float)
        i, j = 0, n // 2
        for num in range(1, n * n + 1):
            M[i, j] = num
            ni, nj = (i - 1) % n, (j + 1) % n
            if M[ni, nj] != 0:
                ni = (i + 1) % n
                nj = j
            i, j = ni, nj
        return _wrap(M)

    elif n % 4 == 0:
        # Doubly even order
        M = np.arange(1, n * n + 1, dtype=float).reshape(n, n)
        # Mark positions to flip
        for i in range(n):
            for j in range(n):
                ii = i % 4
                jj = j % 4
                if (ii == 0 or ii == 3) and (jj == 0 or jj == 3):
                    M[i, j] = n * n + 1 - M[i, j]
                elif (ii == 1 or ii == 2) and (jj == 1 or jj == 2):
                    M[i, j] = n * n + 1 - M[i, j]
        return _wrap(M)

    else:
        # Singly even order (n = 4k+2, k >= 1)
        k = n // 2
        # Build odd-order magic square of size k
        A = np.asarray(_unwrap(forge_magic(ForgeArray(np.array(k, dtype=float)))))
        M = np.zeros((n, n), dtype=float)
        M[:k, :k] = A
        M[k:, k:] = A + k * k
        M[:k, k:] = A + 2 * k * k
        M[k:, :k] = A + 3 * k * k

        # Fix columns to balance sums
        m = (n - 2) // 4
        for i in range(k):
            for j in range(m):
                if j == 0 and i == m:
                    continue
                col = j if j > 0 else 0
                if j == 0:
                    col = m if i != m else 0
                    M[i, col], M[k + i, col] = M[k + i, col], M[i, col]
                else:
                    M[i, j], M[k + i, j] = M[k + i, j], M[i, j]

        # Swap rightmost columns
        for i in range(k):
            for j in range(n - m + 1, n):
                if j < n:
                    M[i, j], M[k + i, j] = M[k + i, j], M[i, j]

        return _wrap(M)


def forge_pascal(n):
    """Pascal matrix of order n."""
    n = int(_scalar(n))
    return _wrap(scipy.linalg.pascal(n).astype(float))


def forge_rosser():
    """The classic 8x8 Rosser test matrix.

    Eigenvalues: 0 (repeated), 10*sqrt(10405), ±sqrt(10405),
    510±100*sqrt(26), 1000.
    """
    M = np.array([
        [611,  196, -192,  407,   -8,  -52,  -49,   29],
        [196,  899,  113, -192,  -71,  -43,   -8,  -44],
        [-192, 113,  899,  196,   61,   49,    8,   52],
        [407, -192,  196,  611,    8,   44,   59,  -23],
        [-8,   -71,   61,    8,  411, -599,  208,  208],
        [-52,  -43,   49,   44, -599,  411,  208,  208],
        [-49,   -8,    8,   59,  208,  208,   99, -911],
        [29,   -44,   52,  -23,  208,  208, -911,   99],
    ], dtype=float)
    return _wrap(M)


def forge_toeplitz(c, r=None):
    """Toeplitz matrix.

    Parameters
    ----------
    c : ForgeArray
        First column.
    r : ForgeArray, optional
        First row.  If omitted, uses conjugate of c.
    """
    vc = np.asarray(_unwrap(c), dtype=float).ravel()
    if r is None:
        return _wrap(scipy.linalg.toeplitz(vc))
    vr = np.asarray(_unwrap(r), dtype=float).ravel()
    return _wrap(scipy.linalg.toeplitz(vc, vr))


def forge_vander(x, n=None):
    """Vandermonde matrix.

    Parameters
    ----------
    x : ForgeArray
        Column vector of points.
    n : int, optional
        Number of columns.  Default is len(x).

    Returns V where V[i,j] = x[i]^(n-1-j), matching Octave convention.
    """
    vx = np.asarray(_unwrap(x), dtype=float).ravel()
    if n is None:
        ncols = len(vx)
    else:
        ncols = int(_scalar(n)) if not isinstance(n, (int, float)) else int(n)
    return _wrap(np.vander(vx, N=ncols))


def forge_wilkinson(n):
    """Wilkinson tridiagonal test matrix of order n.

    A symmetric tridiagonal matrix with diagonal
    |floor(n/2)|, |floor(n/2)-1|, ..., 1, 0, 1, ..., |floor(n/2)|
    and ones on the super/sub-diagonals.
    """
    n = int(_scalar(n))
    if n < 1:
        raise ValueError("wilkinson: N must be at least 1")
    m = n // 2
    diag = np.abs(np.arange(m, -(m + 1 if n % 2 == 1 else m), -1), dtype=float)
    # Ensure correct length
    diag = np.concatenate([np.arange(m, -1, -1), np.arange(1, m + 1)])
    if n % 2 == 0:
        diag = np.concatenate([np.arange(m, 0, -1), np.arange(0, m)])
    else:
        diag = np.concatenate([np.arange(m, -1, -1), np.arange(1, m + 1)])
    diag = diag.astype(float)
    off = np.ones(n - 1, dtype=float)
    M = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return _wrap(M)


# ── Gallery dispatcher ───────────────────────────────────────────

_GALLERY_MAP = {
    "hadamard": forge_hadamard,
    "hankel": forge_hankel,
    "hilb": forge_hilb,
    "invhilb": forge_invhilb,
    "magic": forge_magic,
    "pascal": forge_pascal,
    "rosser": forge_rosser,
    "toeplitz": forge_toeplitz,
    "vander": forge_vander,
    "wilkinson": forge_wilkinson,
}


def forge_gallery(name, n=None):
    """Generate a named test matrix.

    Parameters
    ----------
    name : ForgeChar or str
        Matrix name (e.g. 'hilb', 'magic', 'rosser').
    n : int, optional
        Matrix size (not required for fixed-size matrices like rosser).
    """
    if isinstance(name, ForgeChar):
        sname = name.to_str().lower().strip()
    else:
        sname = str(_unwrap(name)).lower().strip()

    fn = _GALLERY_MAP.get(sname)
    if fn is None:
        raise ValueError(f"gallery: unknown matrix '{sname}'. "
                         f"Available: {', '.join(sorted(_GALLERY_MAP))}")

    if sname == "rosser":
        return fn()
    if n is None:
        raise ValueError(f"gallery: matrix '{sname}' requires a size argument")
    return fn(n)


# ── Registry for evaluator ───────────────────────────────────────

SPECIAL_MATRIX_REGISTRY = {
    "gallery": forge_gallery,
    "hadamard": forge_hadamard,
    "hankel": forge_hankel,
    "hilb": forge_hilb,
    "invhilb": forge_invhilb,
    "magic": forge_magic,
    "pascal": forge_pascal,
    "rosser": forge_rosser,
    "toeplitz": forge_toeplitz,
    "vander": forge_vander,
    "wilkinson": forge_wilkinson,
}
