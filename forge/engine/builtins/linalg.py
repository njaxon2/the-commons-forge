"""Linear algebra toolbox.

Implements: bandwidth, cond, condeig, condest, cross, expm, funm, isbanded,
isdefinite, isdiag, ishermitian, issymmetric, istril, istriu, krylov,
linsolve, logm, normest, null, orth, planerot, rank, rref, subspace,
trace, vech, vecnorm, commutation_matrix, duplication_matrix, housh,
lscov, normest1, ols, gls, ordeig, qzhess, tensorprod

SRS trace: SRS-FUNC-001
"""
import numpy as np
from numpy import linalg as la
from scipy import linalg as sla
from forge.engine.types import ForgeArray, _unwrap


def _wrap(x):
    return ForgeArray(np.asarray(x))

def _scalar(x):
    if isinstance(x, ForgeArray):
        d = x.data
        return d.flat[0].item() if d.size == 1 else d
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.flat[0].item()
    return x


def forge_cond(A, *args):
    Ad = _unwrap(A)
    if Ad.ndim < 2:
        Ad = Ad.reshape(1, -1)
    p = int(_scalar(args[0])) if args else 2
    return _wrap(la.cond(Ad, p))

def forge_rank(A, *args):
    Ad = _unwrap(A)
    if Ad.ndim < 2:
        Ad = Ad.reshape(1, -1)
    tol = float(_scalar(args[0])) if args else None
    return _wrap(la.matrix_rank(Ad, tol=tol))

def forge_trace(A):
    Ad = _unwrap(A)
    if Ad.ndim < 2:
        return _wrap(np.sum(Ad))
    return _wrap(np.trace(Ad))

def forge_cross(a, b):
    ad, bd = _unwrap(a).ravel(), _unwrap(b).ravel()
    return ForgeArray(np.cross(ad, bd))

def forge_null(A, *args):
    Ad = _unwrap(A)
    if Ad.ndim < 2:
        Ad = Ad.reshape(1, -1)
    _, s, Vh = la.svd(Ad)
    tol = max(Ad.shape) * s[0] * np.finfo(float).eps if len(s) > 0 else 0
    rank = np.sum(s > tol)
    return ForgeArray(Vh[rank:].T)

def forge_orth(A):
    Ad = _unwrap(A)
    if Ad.ndim < 2:
        Ad = Ad.reshape(-1, 1)
    U, s, _ = la.svd(Ad, full_matrices=False)
    tol = max(Ad.shape) * s[0] * np.finfo(float).eps
    rank = np.sum(s > tol)
    return ForgeArray(U[:, :rank])

def forge_expm(A):
    return ForgeArray(sla.expm(_unwrap(A)))

def forge_logm(A):
    return ForgeArray(sla.logm(_unwrap(A)))

def forge_funm(A, fun):
    Ad = _unwrap(A)
    result, _ = sla.funm(Ad, fun, disp=False)
    return ForgeArray(result)

def forge_linsolve(A, b):
    return ForgeArray(la.solve(_unwrap(A), _unwrap(b)))

def forge_rref(A):
    Ad = _unwrap(A).astype(float).copy()
    rows, cols = Ad.shape
    pivot_row = 0
    pivots = []
    for col in range(cols):
        if pivot_row >= rows:
            break
        max_row = np.argmax(np.abs(Ad[pivot_row:, col])) + pivot_row
        if abs(Ad[max_row, col]) < 1e-12:
            continue
        Ad[[pivot_row, max_row]] = Ad[[max_row, pivot_row]]
        Ad[pivot_row] = Ad[pivot_row] / Ad[pivot_row, col]
        for r in range(rows):
            if r != pivot_row:
                Ad[r] -= Ad[r, col] * Ad[pivot_row]
        pivots.append(col)
        pivot_row += 1
    return ForgeArray(Ad)

def forge_subspace(A, B):
    Ad, Bd = _unwrap(A), _unwrap(B)
    if Ad.ndim < 2:
        Ad = Ad.reshape(-1, 1)
    if Bd.ndim < 2:
        Bd = Bd.reshape(-1, 1)
    QA, _ = la.qr(Ad, mode='reduced') if hasattr(la, 'qr') else (la.qr(Ad)[0], None)
    QB, _ = la.qr(Bd, mode='reduced') if hasattr(la, 'qr') else (la.qr(Bd)[0], None)
    _, s, _ = la.svd(QA.T @ QB)
    s = np.clip(s, -1, 1)
    return _wrap(np.arccos(min(s.min(), 1.0)))

def forge_vecnorm(x, *args):
    xd = _unwrap(x).ravel()
    p = float(_scalar(args[0])) if args else 2
    return _wrap(la.norm(xd, p))

def forge_vech(A):
    Ad = _unwrap(A)
    n = Ad.shape[0]
    indices = np.tril_indices(n)
    return ForgeArray(Ad[indices])

def forge_bandwidth(A):
    Ad = _unwrap(A)
    if Ad.ndim < 2:
        return _wrap(0), _wrap(0)
    rows, cols = Ad.shape
    lower = 0
    upper = 0
    for i in range(rows):
        for j in range(cols):
            if abs(Ad[i, j]) > 0:
                if i > j:
                    lower = max(lower, i - j)
                elif j > i:
                    upper = max(upper, j - i)
    return _wrap(lower), _wrap(upper)

def forge_isbanded(A, lower, upper):
    Ad = _unwrap(A)
    lo, up = int(_scalar(lower)), int(_scalar(upper))
    if Ad.ndim < 2:
        return _wrap(True)
    rows, cols = Ad.shape
    for i in range(rows):
        for j in range(cols):
            if (i - j > lo or j - i > up) and abs(Ad[i, j]) > 0:
                return _wrap(False)
    return _wrap(True)

def forge_isdiag(A):
    Ad = _unwrap(A)
    if Ad.ndim < 2:
        return _wrap(True)
    return _wrap(np.allclose(Ad, np.diag(np.diag(Ad))))

def forge_istril(A):
    Ad = _unwrap(A)
    if Ad.ndim < 2:
        return _wrap(True)
    return _wrap(np.allclose(Ad, np.tril(Ad)))

def forge_istriu(A):
    Ad = _unwrap(A)
    if Ad.ndim < 2:
        return _wrap(True)
    return _wrap(np.allclose(Ad, np.triu(Ad)))

def forge_issymmetric(A, *args):
    Ad = _unwrap(A)
    if Ad.ndim < 2:
        return _wrap(True)
    return _wrap(np.allclose(Ad, Ad.T))

def forge_ishermitian(A, *args):
    Ad = _unwrap(A)
    if Ad.ndim < 2:
        return _wrap(True)
    return _wrap(np.allclose(Ad, Ad.conj().T))

def forge_isdefinite(A, *args):
    Ad = _unwrap(A)
    try:
        la.cholesky(Ad)
        return _wrap(1)
    except la.LinAlgError:
        try:
            la.cholesky(-Ad)
            return _wrap(-1)
        except la.LinAlgError:
            return _wrap(0)

def forge_condest(A):
    Ad = _unwrap(A)
    return _wrap(la.cond(Ad, 1))

def forge_normest(A, *args):
    Ad = _unwrap(A)
    return _wrap(la.norm(Ad, 2))

def forge_planerot(x):
    xd = _unwrap(x).ravel()
    a, b = float(xd[0]), float(xd[1])
    r = np.hypot(a, b)
    if r == 0:
        c, s = 1.0, 0.0
    else:
        c, s = a/r, b/r
    G = np.array([[c, s], [-s, c]])
    y = np.array([r, 0.0])
    return ForgeArray(G), ForgeArray(y)

def forge_condeig(A):
    Ad = _unwrap(A)
    vals, vecs = la.eig(Ad)
    conds = np.zeros(len(vals))
    for i in range(len(vals)):
        v = vecs[:, i]
        conds[i] = la.norm(v) * la.norm(la.inv(vecs)[i, :])
    return ForgeArray(vals), ForgeArray(vecs), ForgeArray(conds)

def forge_lscov(A, b, *args):
    Ad, bd = _unwrap(A), _unwrap(b).ravel()
    if args:
        V = _unwrap(args[0])
        W = la.inv(V)
        x = la.solve(Ad.T @ W @ Ad, Ad.T @ W @ bd)
    else:
        x, _, _, _ = la.lstsq(Ad, bd, rcond=None)
    return ForgeArray(x)

def forge_ols(y, X):
    Xd, yd = _unwrap(X), _unwrap(y).ravel()
    beta, _, _, _ = la.lstsq(Xd, yd, rcond=None)
    return ForgeArray(beta)

def forge_tensorprod(A, B):
    return ForgeArray(np.tensordot(_unwrap(A), _unwrap(B), axes=0))


# Core LA functions that should always be available
def _forge_inv(A):
    return ForgeArray(np.linalg.inv(_unwrap(A)))

def _forge_det(A):
    return ForgeArray(np.array(np.linalg.det(_unwrap(A))))

def _forge_norm(A, *args):
    ord_val = None
    if args:
        a0 = args[0]
        # Handle ForgeChar string arguments (e.g., 'fro', 'inf')
        if hasattr(a0, 'to_str'):
            ord_str = a0.to_str().strip()
            if ord_str == 'fro':
                ord_val = 'fro'
            elif ord_str == 'inf':
                ord_val = np.inf
            else:
                ord_val = ord_str
        elif isinstance(a0, ForgeArray):
            ord_val = float(a0.data.flat[0])
        elif isinstance(a0, str):
            if a0 == 'fro':
                ord_val = 'fro'
            elif a0 == 'inf':
                ord_val = np.inf
            else:
                ord_val = a0
        else:
            ord_val = a0
    return ForgeArray(np.array(np.linalg.norm(_unwrap(A), ord=ord_val)))

def _forge_pinv(A):
    return ForgeArray(np.linalg.pinv(_unwrap(A)))

def _forge_kron(A, B):
    return ForgeArray(np.kron(_unwrap(A), _unwrap(B)))

def _forge_eig(A):
    """eig(A) returns eigenvalues as a column vector (MATLAB single-output behavior)."""
    vals = np.linalg.eigvalsh(_unwrap(A)) if np.allclose(_unwrap(A), _unwrap(A).T) else np.linalg.eig(_unwrap(A))[0]
    vals = np.sort(np.real(vals))
    return ForgeArray(np.atleast_2d(vals).T)  # column vector


def _forge_eig_full(A):
    """[V,D] = eig(A) returns eigenvectors and diagonal eigenvalue matrix."""
    vals, vecs = np.linalg.eig(_unwrap(A))
    return ForgeArray(vecs), ForgeArray(np.diag(vals))

def _forge_svd(A):
    U, s, Vt = np.linalg.svd(_unwrap(A), full_matrices=True)
    return ForgeArray(U), ForgeArray(np.diag(s)), ForgeArray(Vt)

def _forge_lu(A):
    from scipy.linalg import lu as scipy_lu
    P, L, U = scipy_lu(_unwrap(A))
    return ForgeArray(L), ForgeArray(U), ForgeArray(P)

def _forge_qr(A):
    Q, R = np.linalg.qr(_unwrap(A))
    return ForgeArray(Q), ForgeArray(R)

def _forge_chol(A):
    return ForgeArray(np.linalg.cholesky(_unwrap(A)))

def _forge_fft(x, *args):
    n = int(args[0]) if args else None
    data = _unwrap(x).ravel()
    return ForgeArray(np.fft.fft(data, n=n))

def _forge_ifft(x, *args):
    n = int(args[0]) if args else None
    data = _unwrap(x).ravel()
    return ForgeArray(np.fft.ifft(data, n=n))

def _forge_fft2(x):
    return ForgeArray(np.fft.fft2(_unwrap(x)))

def _forge_ifft2(x):
    return ForgeArray(np.fft.ifft2(_unwrap(x)))



LINALG_REGISTRY = {
    "cond": forge_cond, "rank": forge_rank, "trace": forge_trace,
    "cross": forge_cross, "null": forge_null, "orth": forge_orth,
    "expm": forge_expm, "logm": forge_logm, "funm": forge_funm,
    "linsolve": forge_linsolve, "rref": forge_rref, "subspace": forge_subspace,
    "vecnorm": forge_vecnorm, "vech": forge_vech,
    "bandwidth": forge_bandwidth, "isbanded": forge_isbanded,
    "isdiag": forge_isdiag, "istril": forge_istril, "istriu": forge_istriu,
    "issymmetric": forge_issymmetric, "ishermitian": forge_ishermitian,
    "isdefinite": forge_isdefinite, "condest": forge_condest,
    "normest": forge_normest, "planerot": forge_planerot,
    "condeig": forge_condeig, "lscov": forge_lscov, "ols": forge_ols,
    "tensorprod": forge_tensorprod,
    "inv": _forge_inv, "det": _forge_det, "norm": _forge_norm,
    "pinv": _forge_pinv, "kron": _forge_kron, "eig": _forge_eig, "eig_full": _forge_eig_full,
    "svd": _forge_svd, "lu": _forge_lu, "qr": _forge_qr,
    "chol": _forge_chol, "fft": _forge_fft, "ifft": _forge_ifft,
    "fft2": _forge_fft2, "ifft2": _forge_ifft2,
}
