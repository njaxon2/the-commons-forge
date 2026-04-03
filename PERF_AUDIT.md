# Forge Performance Audit
**Date:** 2026-04-03
**Scope:** Systematic performance review of all 782+ registered functions, engine dispatch,
and comparison with MATLAB R2019a as the competitive target.

---

## 1. Architecture Overview

### Function Count by Toolbox (782 in BUILTIN_REGISTRY + ~90 evaluator inline)

| Toolbox             | Count | Backend          |
|---------------------|-------|------------------|
| Plotting            | 74    | matplotlib       |
| Signal Processing   | 71    | scipy.signal     |
| Image Processing    | 52    | scipy/numpy      |
| Statistics          | 52    | scipy.stats      |
| Control Systems     | 50    | scipy.linalg     |
| Strings             | 50    | Python str       |
| Linear Algebra      | 47    | numpy/scipy.linalg |
| File I/O            | 33    | Python stdlib    |
| Sparse Matrices     | 34    | scipy.sparse     |
| Special Functions   | 32    | scipy.special    |
| General             | 32    | numpy            |
| Elementary Functions| 27    | numpy            |
| Financial           | 25    | numpy            |
| Fuzzy Logic         | 20    | numpy            |
| Communications      | 19    | numpy            |
| Symbolic Math       | 19    | sympy            |
| Polynomials         | 18    | numpy            |
| Neural Networks     | 18    | numpy            |
| Geometry            | 14    | scipy.spatial    |
| Instrument Control  | 14    | stubs            |
| Time                | 14    | Python datetime  |
| Optimization        | 13    | scipy.optimize   |
| Special Matrices    | 11    | numpy            |
| ODE Solvers         | 10    | scipy.integrate  |
| Web                 | 10    | httpx            |
| Sets                | 9     | numpy            |
| Parallel Computing  | 9     | multiprocessing  |
| Database            | 8     | sqlite3          |
| Audio               | 7     | sounddevice      |
| **Evaluator inline**| ~90   | numpy            |
| **Session builtins**| ~30   | Python           |

### Dispatch Path (hot path)
```
eval(code) -> parse(code) -> _exec(stmt, ws) -> _eval_expr(node, ws) -> dispatch table
```
- Statement dispatch: O(1) dict lookup (9 node types)
- Expression dispatch: O(1) dict lookup (19 node types)
- Function call: self.functions[name](*args) -- O(1) dict lookup
- Parse cache: LRU per source string
- Number cache: LRU for NumberLiteral nodes

### Wrapping/Unwrapping Overhead
Every operation goes through:
1. _unwrap(x) to get numpy array -- 1 isinstance check, 1 attribute access
2. numpy/scipy operation
3. ForgeArray(result) to rewrap -- type checks, ndim promotion to 2D

The ForgeArray.__init__ has a fast path for type(data) is np.ndarray and dtype is None
but still does an ndim check and possible reshape. _from_ndarray is a faster classmethod
that skips all checks.

---

## 2. Performance Findings by Category

### 2.1 LINEAR ALGEBRA (Critical -- 47 functions)

#### Current routing:
- inv, det, svd, qr, chol, eig, pinv, norm -> **numpy.linalg** (good default)
- lu -> **scipy.linalg.lu** (needs scipy)
- expm, logm, funm -> **scipy.linalg** (appropriate)
- linsolve -> **numpy.linalg.solve** (misnamed: should use structure detection)
- backslash operator -> **_smart_mldivide** with structure detection (good)

#### BOTTLENECK 1 (HIGH): lu always uses scipy
```python
def _forge_lu(A):
    from scipy.linalg import lu as scipy_lu  # <-- import inside function!
    P, L, U = scipy_lu(_unwrap(A))
```
**Problem:** On Windows with reference BLAS, scipy.linalg is 30-93x slower than numpy.linalg.
The inline import adds overhead on every call.
**Fix:** Move import to module level. Add numpy fallback for when scipy is slow. For the
Pro/MKL build, scipy.linalg.lu is fine. For pip/open-source, implement via numpy QR or
detect BLAS backend and route accordingly.

#### BOTTLENECK 2 (HIGH): linsolve ignores matrix structure
```python
def forge_linsolve(A, b):
    return ForgeArray(la.solve(_unwrap(A), _unwrap(b)))
```
**Problem:** MATLAB linsolve accepts an opts struct telling it the matrix is triangular,
symmetric, positive-definite, etc. This skips decomposition entirely.
**Fix:** Add opts parameter, dispatch to scipy.linalg.solve_triangular, cho_solve, etc.

#### BOTTLENECK 3 (MEDIUM): bandwidth and isbanded use Python loops
```python
def forge_bandwidth(A):
    for i in range(rows):
        for j in range(cols):      # O(n^2) Python loop!
            if abs(Ad[i, j]) > 0:
```
**Fix:** Use np.nonzero + vectorized index arithmetic: O(nnz) numpy ops.

#### BOTTLENECK 4 (MEDIUM): vecnorm uses Python list comprehension
```python
return ForgeArray(np.array([la.norm(xd[:, j], p) for j in range(xd.shape[1])]))
```
**Fix:** Use np.linalg.norm(xd, ord=p, axis=0) -- single vectorized call.

#### BOTTLENECK 5 (MEDIUM): condeig computes full inverse inside a loop
```python
conds[i] = la.norm(v) * la.norm(la.inv(vecs)[i, :])  # inv computed INSIDE loop
```
**Fix:** Compute la.inv(vecs) once outside the loop.

#### BOTTLENECK 6 (LOW): FFT not using scipy.fft
```python
def _forge_fft(x, *args):
    return ForgeArray(np.fft.fft(data, n=n))  # numpy FFT
```
**Fix:** scipy.fft is significantly faster (pocketfft backend, threading). Replace with
scipy.fft.fft/ifft/fft2/ifft2 for 2-5x faster FFT operations.

#### MISSING FUNCTIONS (MATLAB users expect these):
- schur -- Schur decomposition (scipy.linalg.schur)
- hess -- Hessenberg form (scipy.linalg.hessenberg)
- balance -- Diagonal scaling for eigenvalue problems
- sqrtm -- Matrix square root
- lyap / sylvester -- Matrix equations (have lyap in control, not in linalg)
- mldivide / mrdivide as named functions (only operators \ and /)
- cholupdate, qrupdate -- Rank-1 updates
- ldl -- LDL factorization
- rcond -- Reciprocal condition number (faster than cond for singularity check)
- lsqminnorm -- Minimum-norm least-squares

---

### 2.2 ELEMENT-WISE MATH (25+ functions in evaluator)

#### Current routing:
- sin, cos, tan, exp, log, sqrt, abs, etc. -> **numpy ufuncs** (optimal)
- Expression fusion: can_fuse triggers for arrays >= 100 elements

#### BOTTLENECK 7 (HIGH): ForgeArray wrapping overhead on scalar/small operations
Every math operation does:
```python
ForgeArray(f(x._data))  # involves isinstance check, ndim check, possible reshape
```
For scalar operations in tight loops (e.g., x = sin(x) + cos(x) in a for-loop),
the wrapping/unwrapping overhead dominates.
**Fix:** Short-circuit scalar path: if x._data.shape == (1,1), return
ForgeArray._from_ndarray(f(x._data)) which skips all checks.

#### BOTTLENECK 8 (MEDIUM): Expression fusion threshold too high
can_fuse requires >= 100 elements. For arrays of 10-99 elements (common in
signal processing), every +, -, .* allocates a temporary.
**Fix:** Lower threshold to 64 (matching cache line heuristics), or better:
benchmark to find the crossover point.

#### BOTTLENECK 9 (MEDIUM): _make_math creates generic closure per function
The *a varargs pattern in the math closure adds overhead. Since these are always
single-argument functions (except atan2), specialize the fast path to accept
exactly one argument and use ForgeArray._from_ndarray.

#### MISSING FUNCTIONS:
- exp2 -- 2^x (faster than 2.^x)
- expm1, log1p -- Accurate for small values (numpy has these)
- cbrt -- Cube root

---

### 2.3 MATRIX OPERATIONS (evaluator inline)

#### BOTTLENECK 10 (HIGH): _is_matrix_op called on EVERY *, /, \, ^
```python
def _is_matrix_op(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)  # <-- unnecessary conversion!
```
**Problem:** np.asarray is called even when inputs are already numpy arrays.
**Fix:** Check ndim and size directly via hasattr, avoid np.asarray.

#### BOTTLENECK 11 (HIGH): * operator creates ForgeArray via slow path
```python
if op == "*":
    return ForgeArray(l @ r if _is_matrix_op(l, r) else l * r)
```
**Fix:** Use ForgeArray._from_ndarray for 2D results to skip ndim checks.
Already done for + and - but NOT for *, /, \, ^.

#### BOTTLENECK 12 (MEDIUM): Binary op dispatch uses if/elif chain
The _eval_binop method uses sequential if op == "+", if op == "-", etc.
For .* (6th check) or .^ (10th check), this wastes cycles.
**Fix:** Dict-based dispatch for binary ops (same pattern as statement/expression dispatch).

---

### 2.4 SIGNAL PROCESSING (71 functions)

#### Current routing: All backed by scipy.signal (good)

#### BOTTLENECK 13 (MEDIUM): conv uses numpy.convolve (no FFT for large arrays)
numpy.convolve is fine for 1-D but does not use FFT for large arrays.
**Fix:** For large arrays (n > 500), use scipy.signal.fftconvolve.

#### BOTTLENECK 14 (MEDIUM): filter function not registered
MATLAB filter(b, a, x) is one of the most-used DSP functions. Currently only
lfilter is registered (scipy name). Users expect "filter".
**Fix:** Add "filter": lfilter to SIGNAL_REGISTRY.

#### MISSING FUNCTIONS (commonly used in MATLAB):
- filter (alias for lfilter) -- HIGH priority
- fir1 (alias for firwin with MATLAB conventions)
- pwelch -- Welch PSD (most common PSD method)
- tfestimate -- Transfer function estimation
- mscohere -- Magnitude-squared coherence
- cpsd -- Cross power spectral density
- goertzel -- Efficient single-frequency DFT
- bandpower -- Band power
- envelope -- Signal envelope
- istft -- Inverse STFT (have stft and synthesis, but istft name expected)

---

### 2.5 STATISTICS (52 functions)

#### BOTTLENECK 15 (MEDIUM): Moving window functions may use Python loops
**Fix:** Use scipy.ndimage.uniform_filter1d for movmean, numpy stride tricks for others.

#### BOTTLENECK 16 (LOW): mean, std, var wrap numpy redundantly
Each call goes through _ensure_float which may copy data unnecessarily.
**Fix:** Check dtype first; skip copy if already float64.

#### MISSING FUNCTIONS (largest gap vs MATLAB):
**Distribution functions (pdf/cdf/inv/rnd for ~15 distributions):**
- normpdf, normcdf, norminv, normrnd -- Normal distribution
- chi2pdf, chi2cdf, chi2inv -- Chi-squared
- tpdf, tcdf, tinv -- Student t
- fpdf, fcdf, finv -- F-distribution
- exppdf, expcdf, expinv -- Exponential
- unifpdf, unifcdf, unifinv -- Uniform
- betapdf, betacdf, betainv -- Beta
- gammpdf, gammcdf, gamminv -- Gamma
- lognpdf, logncdf, logninv -- Lognormal
- wblpdf, wblcdf, wblinv -- Weibull
- poisspdf exists, but most others missing

**Hypothesis tests:**
- ttest, ttest2, chi2gof, kstest, anova1, anova2

**Regression/ML:**
- regress, robustfit, pca, pcacov, kmeans, pdist, squareform, fitdist

This is the **largest functional gap** vs MATLAB. scipy.stats has all backends.

---

### 2.6 SPARSE MATRICES (34 functions)

#### BOTTLENECK 17 (MEDIUM): Sparse matrices bypass ForgeArray entirely
Sparse results are returned as raw scipy.sparse objects. This means no consistent
interface for GUI display and workspace inspection.

#### BOTTLENECK 18 (LOW): spfun uses Python list comprehension
**Fix:** Apply numpy ufunc directly to .data array.

#### MISSING FUNCTIONS:
- spalloc, sparse 6-arg form, chol/lu for sparse, amd/colamd/symamd ordering

---

### 2.7 INTERPRETER OVERHEAD (Engine-level)

#### BOTTLENECK 19 (CRITICAL): For-loop overhead
Each iteration: AST walk + dict lookups + type dispatch + ForgeArray wrap/unwrap.
**Estimated:** ~50-200 microseconds per iteration (vs ~1 microsecond in MATLAB JIT).

**Current mitigations:**
- Numba JIT for eligible numeric loops (good, but narrow eligibility)
- Expression fusion for element-wise chains >= 100 elements

**Additional fixes:**
1. Expand JIT eligibility: if/else in loop body, more array patterns
2. Vectorization hints in documentation
3. Future: bytecode compiler to eliminate recursive _eval_expr overhead

#### BOTTLENECK 20 (HIGH): Parse overhead on repeated eval
Parse cache mitigates per-string, but .m files parse line-by-line.
**Fix:** Parse entire files at once (already done for run command).

#### BOTTLENECK 21 (MEDIUM): Workspace variable access overhead
Each read/write goes through method dispatch.
**Fix:** For JIT loops, variables are already extracted. For interpreted loops,
consider pre-extracting loop variables into a local dict.

---

## 3. BLAS/LAPACK Routing Strategy

### Current situation:
- **numpy.linalg:** Uses whatever BLAS numpy was compiled with (MKL on conda, OpenBLAS on pip)
- **scipy.linalg:** Uses whatever BLAS scipy was compiled with (may differ from numpy!)
- **Windows pip installs:** scipy ships reference BLAS (30-93x slower than MKL)

### Recommended strategy:

| Operation | Open-source (pip) | Pro (MKL conda) |
|-----------|-------------------|------------------|
| A \ b (solve) | numpy.linalg.solve | scipy.linalg.solve |
| inv(A) | numpy.linalg.inv | scipy.linalg.inv |
| eig(A) | numpy.linalg.eig/eigh | scipy.linalg.eig/eigh |
| svd(A) | numpy.linalg.svd | scipy.linalg.svd |
| chol(A) | numpy.linalg.cholesky | scipy.linalg.cholesky |
| lu(A) | numpy QR fallback | scipy.linalg.lu |
| qr(A) | numpy.linalg.qr | scipy.linalg.qr |
| fft(x) | scipy.fft | scipy.fft |
| expm(A) | scipy.linalg.expm | scipy.linalg.expm |

**Implementation:** Add a _BLAS_BACKEND detection at import time:
```python
import numpy as np
try:
    _blas_info = np.__config__.blas_opt_info
    _MKL = 'mkl' in str(_blas_info.get('libraries', []))
except:
    _MKL = False
```
Route functions accordingly. One-time check, zero runtime cost.

---

## 4. Missing Core Functions (Not in any toolbox)

### Matrix Operations:
- sub2ind, ind2sub -- index conversion (commonly needed)
- bsxfun -- deprecated in modern MATLAB but still used in legacy code
- accumarray improvements -- current impl may need optimization for large data

### Data Types:
- table -- ForgeTable exists but may not be fully functional
- categorical, datetime, duration -- modern MATLAB types

### I/O:
- load / save for .mat files (v5 format) -- critical for data exchange
- xlsread / xlswrite / readtable -- Excel I/O
- textscan -- formatted text reading
- fgets / fgetl -- line-by-line file reading
- sscanf -- formatted string scanning

### Programming:
- inputname, mfilename -- introspection
- dbstop, dbcont, dbstack -- debugging
- try/catch identifier access (ME.identifier, ME.message)

---

## 5. Priority Action Items

### P0 -- Critical (do first, biggest impact)

1. **BLAS routing layer:** Detect MKL vs reference BLAS at import. Route lu through
   numpy on pip installs. (~1 hour, eliminates 30-93x penalty on Windows pip)

2. **Switch FFT to scipy.fft:** Replace np.fft with scipy.fft for all FFT operations.
   (~30 min, 2-5x improvement on FFT operations)

3. **Distribution functions (pdf/cdf/inv/rnd):** Add wrappers for the 15 most common
   distributions using scipy.stats. (~2 hours, closes the biggest functional gap)

### P1 -- High (next sprint)

4. **Fix _is_matrix_op np.asarray waste:** Remove unnecessary np.asarray calls.
   (~15 min, speeds up every *, /, \ operation)

5. **Use _from_ndarray in all binary ops:** Currently only +, - use fast path.
   Extend to *, /, \, ^, .*, etc. (~30 min)

6. **Vectorize bandwidth, isbanded:** Replace Python loops with np.nonzero.
   (~30 min)

7. **Fix condeig repeated inverse:** Compute inverse once. (~10 min)

8. **Add linsolve opts parameter:** Support triangular, symmetric, posdef flags.
   (~1 hour, matches MATLAB API)

9. **Register filter as alias for lfilter:** (~5 min, high user-facing impact)

10. **Specialize scalar math path:** Skip ForgeArray overhead for 1x1 arrays.
    (~1 hour, major loop performance improvement)

### P2 -- Medium (following sprints)

11. Binary op dict dispatch: Replace if/elif chain with dict lookup. (~30 min)
12. Expand JIT loop eligibility: Support if/else, more array patterns. (~4 hours)
13. Moving window functions: Use numpy stride tricks or scipy. (~2 hours)
14. Add sub2ind, ind2sub. (~30 min)
15. Add schur, hess, balance, sqrtm, ldl, rcond. (~2 hours)
16. Add pwelch, cpsd, mscohere, tfestimate. (~2 hours)
17. Hypothesis tests (ttest, kstest, anova). (~3 hours)
18. Lower expression fusion threshold: Benchmark and tune. (~1 hour)
19. ForgeSparse wrapper or robust sparse display. (~2 hours)

### P3 -- Low (backlog)

20. Bytecode VM for interpreter core: Major project, ~2-4 weeks.
21. CHOLMOD/SuperLU for sparse direct solvers.
22. Parallel FFT via scipy.fft workers.
23. GPU offload via CuPy for large matrix ops.

---

## 6. Benchmark Gaps

Current benchmarks (18 files) cover:
matmul, solve, det_inv, lu, svd, eig, fft, elementwise, cumsum, find,
sort, unique, reshape, conv, stats, loop, funcall, index

### Missing benchmarks needed:
- bench_fft_large.m -- FFT on 10^6+ points (scipy.fft vs numpy.fft)
- bench_filter.m -- IIR/FIR filtering (butter+filter pipeline)
- bench_sparse_solve.m -- Sparse Ax=b
- bench_chol.m -- Cholesky factorization
- bench_qr.m -- QR factorization
- bench_interp.m -- Interpolation
- bench_distribution.m -- pdf/cdf evaluation
- bench_scalar_loop.m -- Scalar operations in a loop (JIT test)
- bench_mixed_loop.m -- Array operations in a loop (interpreter overhead)

---

## 7. Quick Wins Summary (< 2 hours total, measurable impact)

| Fix | Time | Impact |
|-----|------|--------|
| BLAS detection + numpy routing for lu | 1h | 30-93x on Windows pip |
| scipy.fft for all FFT ops | 30m | 2-5x FFT speedup |
| _is_matrix_op remove np.asarray | 15m | 5-10% on all matrix ops |
| _from_ndarray in * / \ ^ ops | 30m | ~10% on all matrix ops |
| Register filter alias | 5m | Unblocks DSP users |
| Fix condeig inverse | 10m | 100x on condeig |
| Vectorize bandwidth/isbanded | 30m | 100x+ for large matrices |
