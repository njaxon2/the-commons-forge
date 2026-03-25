"""Forge sparse matrix toolbox — 32+ functions wrapping scipy.sparse."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from forge.engine.types import ForgeArray, _unwrap


# ---------------------------------------------------------------------------
# Construction / conversion
# ---------------------------------------------------------------------------

def forge_sparse(i_or_m, j=None, v=None, m=None, n=None):
    i_val = _unwrap(i_or_m) if isinstance(i_or_m, ForgeArray) else np.asarray(i_or_m)
    if j is not None and v is None:
        j_val = _unwrap(j) if isinstance(j, ForgeArray) else np.asarray(j)
        rows = int(np.asarray(i_val).flat[0])
        cols = int(np.asarray(j_val).flat[0])
        return sp.csc_matrix((rows, cols))
    if j is None:
        return sp.csc_matrix(np.asarray(i_val))
    j_val = _unwrap(j) if isinstance(j, ForgeArray) else np.asarray(j)
    v_val = _unwrap(v) if isinstance(v, ForgeArray) else np.asarray(v)
    i_arr = np.asarray(i_val, dtype=float).ravel().astype(int) - 1
    j_arr = np.asarray(j_val, dtype=float).ravel().astype(int) - 1
    v_arr = np.asarray(v_val, dtype=float).ravel()
    if m is not None:
        m_v = _unwrap(m) if isinstance(m, ForgeArray) else m
        m = int(np.asarray(m_v).flat[0])
    else:
        m = int(i_arr.max()) + 1
    if n is not None:
        n_v = _unwrap(n) if isinstance(n, ForgeArray) else n
        n = int(np.asarray(n_v).flat[0])
    else:
        n = int(j_arr.max()) + 1
    return sp.coo_matrix((v_arr, (i_arr, j_arr)), shape=(m, n)).tocsc()



def forge_full(S):
    """Convert sparse matrix to dense ForgeArray."""
    return ForgeArray(S.toarray())


def forge_issparse(x):
    """Return True if *x* is a scipy sparse matrix."""
    return sp.issparse(x)


def forge_nnz(S):
    """Number of stored (explicit) nonzeros."""
    return S.nnz


def forge_nzmax(S):
    """Maximum number of nonzero entries (same as nnz for scipy)."""
    return S.nnz


def forge_spconvert(T):
    """Build sparse from an Nx3 triplet matrix [i j v]."""
    T = np.asarray(_unwrap(T), dtype=float)
    i = T[:, 0].astype(int) - 1
    j = T[:, 1].astype(int) - 1
    v = T[:, 2]
    m = int(i.max()) + 1
    n = int(j.max()) + 1
    return sp.coo_matrix((v, (i, j)), shape=(m, n)).tocsc()


def forge_spdiags(B, d, m, n):
    """Sparse matrix from diagonals."""
    B = np.asarray(_unwrap(B), dtype=float)
    d = np.asarray(_unwrap(d), dtype=int).ravel()
    diags = [B[k, :] for k in range(B.shape[0])]
    return sp.diags(diags, d.tolist(), shape=(m, n), format="csc")


def forge_speye(m, n=None):
    """Sparse identity matrix."""
    if n is None:
        n = m
    return sp.eye(m, n, format="csc")


def forge_sprand(m, n, density):
    """Sparse random matrix (uniform distribution)."""
    return sp.random(m, n, density=density, format="csc")


def forge_sprandn(m, n, density):
    """Sparse random matrix (normal distribution)."""
    rng = np.random.default_rng()
    return sp.random(m, n, density=density, format="csc",
                     data_rvs=rng.standard_normal)


def forge_sprandsym(n, density):
    """Sparse random symmetric matrix."""
    R = forge_sprandn(n, n, density)
    return (R + R.T) / 2


def forge_spones(S):
    """Replace nonzero entries with ones."""
    out = S.copy()
    out.data[:] = 1.0
    return out


def forge_spfun(fun, S):
    """Apply *fun* to every nonzero element."""
    out = S.copy()
    out.data = np.array([fun(v) for v in out.data])
    return out


def forge_spstats(S):
    """Column-wise stats: (count, mean, variance)."""
    S_csc = sp.csc_matrix(S)
    count = np.diff(S_csc.indptr)
    col_sum = np.array(S_csc.sum(axis=0)).ravel()
    col_sum_sq = np.array(S_csc.multiply(S_csc).sum(axis=0)).ravel()
    mean = np.zeros(S_csc.shape[1])
    var = np.zeros(S_csc.shape[1])
    nz = count > 0
    mean[nz] = col_sum[nz] / count[nz]
    var[nz] = col_sum_sq[nz] / count[nz] - mean[nz] ** 2
    return ForgeArray(count), ForgeArray(mean), ForgeArray(var)


def forge_spy(S):
    """Return (row_indices, col_indices) of nonzero entries."""
    coo = sp.coo_matrix(S)
    return ForgeArray(coo.row), ForgeArray(coo.col)


def forge_nonzeros(S):
    """Return nonzero values as a ForgeArray column."""
    csc = sp.csc_matrix(S)
    return ForgeArray(csc.data.copy())


def forge_colperm(S):
    """Column ordering to reduce fill-in (approximate minimum degree)."""
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    S_csc = sp.csc_matrix(S)
    sym = S_csc + S_csc.T
    perm = reverse_cuthill_mckee(sym)
    return ForgeArray(perm + 1)  # 1-based


def forge_spaugment(S, c=1.0):
    """Build augmented matrix [c*I, S; S', 0]."""
    S_csc = sp.csc_matrix(S)
    m, n = S_csc.shape
    top_left = c * sp.eye(m, format="csc")
    bottom_right = sp.csc_matrix((n, n))
    top = sp.hstack([top_left, S_csc], format="csc")
    bottom = sp.hstack([S_csc.T, bottom_right], format="csc")
    return sp.vstack([top, bottom], format="csc")


# ---------------------------------------------------------------------------
# Iterative solvers — all return (x, info) where info==0 means converged
# ---------------------------------------------------------------------------

def _solver_wrapper(solver_fn, A, b, tol=1e-6, maxit=None, M=None):
    """Common wrapper for scipy iterative solvers."""
    b_arr = np.asarray(_unwrap(b), dtype=float).ravel()
    kwargs = {"rtol": tol}
    if maxit is not None:
        kwargs["maxiter"] = maxit
    if M is not None:
        kwargs["M"] = M
    x, info = solver_fn(A, b_arr, **kwargs)
    return ForgeArray(x), int(info)


def forge_bicg(A, b, tol=1e-6, maxit=None, M=None):
    """BiConjugate Gradient solver."""
    return _solver_wrapper(spla.bicg, A, b, tol, maxit, M)


def forge_bicgstab(A, b, tol=1e-6, maxit=None, M=None):
    """BiConjugate Gradient Stabilized solver."""
    return _solver_wrapper(spla.bicgstab, A, b, tol, maxit, M)


def forge_cgs(A, b, tol=1e-6, maxit=None, M=None):
    """Conjugate Gradient Squared solver."""
    return _solver_wrapper(spla.cgs, A, b, tol, maxit, M)


def forge_gmres(A, b, tol=1e-6, maxit=None, M=None):
    """GMRES solver."""
    return _solver_wrapper(spla.gmres, A, b, tol, maxit, M)


def forge_pcg(A, b, tol=1e-6, maxit=None, M=None):
    """Preconditioned Conjugate Gradient (wraps scipy CG)."""
    return _solver_wrapper(spla.cg, A, b, tol, maxit, M)


def forge_qmr(A, b, tol=1e-6, maxit=None, M=None):
    """Quasi-Minimal Residual solver."""
    return _solver_wrapper(spla.qmr, A, b, tol, maxit, M)


def forge_tfqmr(A, b, tol=1e-6, maxit=None, M=None):
    """Transpose-Free Quasi-Minimal Residual solver."""
    return _solver_wrapper(spla.tfqmr, A, b, tol, maxit, M)


def forge_pcr(A, b, tol=1e-6, maxit=None, M=None):
    """Preconditioned Conjugate Residual (alias for CG)."""
    return _solver_wrapper(spla.cg, A, b, tol, maxit, M)


# ---------------------------------------------------------------------------
# Preconditioners
# ---------------------------------------------------------------------------

def forge_ichol(A):
    """Incomplete Cholesky (via ILU with low drop tolerance)."""
    A_csc = sp.csc_matrix(A, dtype=float)
    ilu = spla.spilu(A_csc, drop_tol=1e-4)
    n = A_csc.shape[0]
    return spla.LinearOperator((n, n), matvec=ilu.solve)


def forge_ilu(A):
    """Incomplete LU factorisation."""
    A_csc = sp.csc_matrix(A, dtype=float)
    ilu = spla.spilu(A_csc)
    n = A_csc.shape[0]
    return spla.LinearOperator((n, n), matvec=ilu.solve)


# ---------------------------------------------------------------------------
# Eigenvalues / SVD
# ---------------------------------------------------------------------------

def forge_eigs(A, k=6):
    """Compute *k* largest-magnitude eigenvalues/vectors of sparse A."""
    A_sp = sp.csc_matrix(A, dtype=float)
    vals, vecs = spla.eigs(A_sp, k=k)
    return ForgeArray(vals), ForgeArray(vecs)


def forge_svds(A, k=6):
    """Compute *k* largest singular values/vectors of sparse A."""
    A_sp = sp.csc_matrix(A, dtype=float)
    U, s, Vt = spla.svds(A_sp, k=k)
    return ForgeArray(U), ForgeArray(s), ForgeArray(Vt)


# ---------------------------------------------------------------------------
# Graph stubs
# ---------------------------------------------------------------------------

def forge_etreeplot(*_args, **_kwargs):
    """Stub — elimination-tree plot (not yet implemented)."""
    return None


def forge_gplot(*_args, **_kwargs):
    """Stub — graph plot (not yet implemented)."""
    return None


def forge_treelayout(*_args, **_kwargs):
    """Stub — tree layout (not yet implemented)."""
    return None


def forge_treeplot(*_args, **_kwargs):
    """Stub — tree plot (not yet implemented)."""
    return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SPARSE_REGISTRY = {
    "sparse":       forge_sparse,
    "full":         forge_full,
    "issparse":     forge_issparse,
    "nnz":          forge_nnz,
    "nzmax":        forge_nzmax,
    "spconvert":    forge_spconvert,
    "spdiags":      forge_spdiags,
    "speye":        forge_speye,
    "sprand":       forge_sprand,
    "sprandn":      forge_sprandn,
    "sprandsym":    forge_sprandsym,
    "spones":       forge_spones,
    "spfun":        forge_spfun,
    "spstats":      forge_spstats,
    "spy":          forge_spy,
    "nonzeros":     forge_nonzeros,
    "colperm":      forge_colperm,
    "spaugment":    forge_spaugment,
    "bicg":         forge_bicg,
    "bicgstab":     forge_bicgstab,
    "cgs":          forge_cgs,
    "gmres":        forge_gmres,
    "pcg":          forge_pcg,
    "qmr":          forge_qmr,
    "tfqmr":        forge_tfqmr,
    "pcr":          forge_pcr,
    "ichol":        forge_ichol,
    "ilu":          forge_ilu,
    "eigs":         forge_eigs,
    "svds":         forge_svds,
    "etreeplot":    forge_etreeplot,
    "gplot":        forge_gplot,
    "treelayout":   forge_treelayout,
    "treeplot":     forge_treeplot,
}
