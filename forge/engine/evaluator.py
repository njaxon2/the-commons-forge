"""M-language evaluator: AST → execution.

Walks the AST produced by the parser, evaluating expressions and executing statements
against a workspace (variable scope) with access to built-in functions.
"""
import numpy as np
import sys
import time
import os
import io
from typing import Any, Dict, List, Optional, Callable
from forge.engine.types import (
    ForgeArray, _unwrap, forge_eye, forge_ones, forge_zeros, forge_rand, forge_randn,
    forge_randi, forge_true, forge_false, forge_diag, forge_linspace, forge_colon,
    forge_repmat, forge_double, forge_single, forge_int8, forge_int16, forge_int32,
    forge_int64, forge_uint8, forge_uint16, forge_uint32, forge_uint64, forge_logical,
    forge_complex, forge_isnan, forge_isinf, forge_isfinite, forge_isnumeric,
    forge_islogical, forge_isfloat, forge_isinteger, forge_ischar,
    FORGE_PI, FORGE_E, FORGE_INF, FORGE_NAN, FORGE_EPS, FORGE_I,
)
from forge.engine.containers import (
    ForgeChar, ForgeCell, ForgeStruct, ForgeMap, forge_char,
    forge_strcmp, forge_strcmpi, forge_cell, forge_iscell, forge_fieldnames,
    forge_isstruct, forge_struct,
)
from forge.engine.builtins import BUILTIN_REGISTRY
from forge.engine.parser import (
    NumberLiteral, StringLiteral, Identifier, UnaryOp, BinaryOp, CompareOp,
    LogicalOp, TransposeOp, ColonExpr, Index, CellIndex, FieldAccess,
    DynamicFieldAccess, MatrixLiteral, CellLiteral, FunctionHandle, AnonFunction,
    EndKeyword, BareColon, Assignment, IfStatement, ForStatement, WhileStatement,
    DoUntilStatement, SwitchStatement, TryCatchStatement, ReturnStatement,
    BreakStatement, ContinueStatement, FunctionDef, ExpressionStatement,
    GlobalStatement, PersistentStatement, ClassDef, UnwindProtect, parse,
)


class BreakSignal(Exception):
    pass

class ContinueSignal(Exception):
    pass

class ReturnSignal(Exception):
    def __init__(self, value=None):
        self.value = value


class ForgeError(Exception):
    def __init__(self, identifier, message):
        super().__init__(message)
        self.identifier = identifier
        self.message = message


class UndefinedFunctionError(Exception):
    """Raised when a function call fails because the function is not defined."""
    pass


class Workspace:
    """Variable scope with global/persistent support."""

    def __init__(self, parent=None):
        self._vars: Dict[str, Any] = {}
        self._parent = parent
        self._globals: set = set()

    def get(self, name: str) -> Any:
        if name in self._vars:
            return self._vars[name]
        if self._parent:
            return self._parent.get(name)
        raise NameError(f"Undefined variable: {name}")

    def set(self, name: str, value: Any):
        self._vars[name] = value

    def has(self, name: str) -> bool:
        if name in self._vars:
            return True
        if self._parent:
            return self._parent.has(name)
        return False

    def delete(self, name: str):
        self._vars.pop(name, None)

    def names(self) -> List[str]:
        return list(self._vars.keys())

    def items(self):
        return self._vars.items()



# --- Container utility builtins (R90) ---
import numpy as np

def _cellfun_builtin(func, cell_arg, *extra_args, **kwargs):
    """Apply function to each element of a cell array."""
    from forge.engine.containers import ForgeCell, ForgeChar
    from forge.engine.types import ForgeArray
    if isinstance(cell_arg, ForgeCell):
        items = cell_arg._data
    elif isinstance(cell_arg, (list, tuple)):
        items = list(cell_arg)
    else:
        items = [cell_arg]

    results = []
    for item in items:
        r = func(item)
        results.append(r)

    # Check if all results are scalars -> return array
    all_scalar = all(
        isinstance(r, (int, float, np.integer, np.floating)) or
        (isinstance(r, ForgeArray) and r.data.size == 1) or
        (isinstance(r, np.ndarray) and r.size == 1)
        for r in results
    )
    if all_scalar:
        vals = []
        for r in results:
            if isinstance(r, ForgeArray):
                vals.append(float(r.data.flat[0]))
            elif isinstance(r, np.ndarray):
                vals.append(float(r.flat[0]))
            else:
                vals.append(float(r))
        return ForgeArray(np.array(vals, dtype=np.float64))
    return ForgeCell(results)


def _arrayfun_builtin(func, arr, *extra_arrs, **kwargs):
    """Apply function element-wise to array(s)."""
    from forge.engine.types import ForgeArray
    if isinstance(arr, ForgeArray):
        data = arr.data
    elif isinstance(arr, np.ndarray):
        data = arr
    else:
        data = np.atleast_1d(np.asarray(arr, dtype=np.float64))

    extra_data = []
    for ea in extra_arrs:
        if isinstance(ea, ForgeArray):
            extra_data.append(ea.data)
        elif isinstance(ea, np.ndarray):
            extra_data.append(ea)
        else:
            extra_data.append(np.atleast_1d(np.asarray(ea, dtype=np.float64)))

    results = []
    flat = data.ravel()
    extra_flat = [e.ravel() for e in extra_data]
    for i in range(len(flat)):
        call_args = [flat[i]] + [ef[i] if i < len(ef) else ef[-1] for ef in extra_flat]
        r = func(*call_args)
        if isinstance(r, ForgeArray):
            results.append(float(r.data.flat[0]))
        elif isinstance(r, np.ndarray):
            results.append(float(r.flat[0]))
        else:
            results.append(float(r))

    result = np.array(results, dtype=np.float64).reshape(data.shape)
    return ForgeArray(result)


def _num2cell_builtin(arr, *args):
    """Convert array to cell array."""
    from forge.engine.containers import ForgeCell
    from forge.engine.types import ForgeArray
    if isinstance(arr, ForgeArray):
        data = arr.data
    elif isinstance(arr, np.ndarray):
        data = arr
    else:
        return ForgeCell([arr])

    return ForgeCell([ForgeArray(np.atleast_1d(np.array(x))) for x in data.ravel()])


def _cell2mat_builtin(c):
    """Convert cell array of matrices to a single matrix."""
    from forge.engine.containers import ForgeCell
    from forge.engine.types import ForgeArray
    if not isinstance(c, ForgeCell):
        return c

    items = c._data
    vals = []
    for item in items:
        if isinstance(item, ForgeArray):
            vals.append(item.data)
        elif isinstance(item, np.ndarray):
            vals.append(item)
        elif isinstance(item, (int, float)):
            vals.append(np.array(item))
        else:
            vals.append(np.array(float(item)))

    shape = c.shape
    if len(shape) == 2 and shape[0] > 1:
        rows = []
        idx = 0
        for i in range(shape[0]):
            row_blocks = []
            for j in range(shape[1]):
                row_blocks.append(np.atleast_2d(vals[idx]))
                idx += 1
            rows.append(np.hstack(row_blocks))
        result = np.vstack(rows)
    else:
        result = np.concatenate([np.atleast_1d(v) for v in vals])

    return ForgeArray(result)


def _rmfield_builtin(s, field):
    """Remove field from struct."""
    from forge.engine.containers import ForgeStruct, ForgeChar
    if isinstance(field, ForgeChar):
        field = field.to_str()
    field = str(field)
    if not isinstance(s, ForgeStruct):
        raise ValueError("First argument must be a struct")
    new_s = ForgeStruct()
    for k, v in s._fields.items():
        if k != field:
            new_s._fields[k] = v
    return new_s


def _cat_builtin(dim, *arrays):
    """Concatenate arrays along dimension dim."""
    from forge.engine.types import ForgeArray
    dim = int(dim) - 1  # Convert to 0-based
    np_arrays = []
    for a in arrays:
        if isinstance(a, ForgeArray):
            np_arrays.append(a.data)
        elif isinstance(a, np.ndarray):
            np_arrays.append(a)
        else:
            np_arrays.append(np.atleast_1d(np.asarray(a, dtype=np.float64)))

    if dim == 0:
        processed = []
        for a in np_arrays:
            if a.ndim == 1:
                processed.append(a.reshape(1, -1))
            else:
                processed.append(a)
        result = np.vstack(processed)
    elif dim == 1:
        processed = []
        for a in np_arrays:
            if a.ndim == 1:
                processed.append(a.reshape(1, -1))
            else:
                processed.append(a)
        result = np.hstack(processed)
    else:
        result = np.concatenate(np_arrays, axis=dim)

    return ForgeArray(result)



def _deal_builtin(*args):
    """Deal inputs to outputs. [a,b,c] = deal(x,y,z) or [a,b] = deal(x)."""
    if len(args) == 1:
        # Replicate single input for all outputs
        return args[0]
    return args


def _structfun_builtin(func, s):
    """Apply function to each field of a struct."""
    from forge.engine.containers import ForgeStruct, ForgeCell, ForgeChar
    from forge.engine.types import ForgeArray
    import numpy as np
    if not isinstance(s, ForgeStruct):
        raise ValueError("Second argument must be a struct")
    results = []
    for name, val in s._fields.items():
        r = func(val)
        results.append(r)
    # Check if all scalar
    all_scalar = all(
        isinstance(r, (int, float, np.integer, np.floating)) or
        (isinstance(r, ForgeArray) and r.data.size == 1)
        for r in results
    )
    if all_scalar:
        vals = []
        for r in results:
            if isinstance(r, ForgeArray):
                vals.append(float(r.data.flat[0]))
            else:
                vals.append(float(r))
        return ForgeArray(np.array(vals, dtype=np.float64))
    return ForgeCell(results)

class Session:
    """Forge execution session — holds workspace, output, and function registry."""

    def __init__(self):
        self.workspace = Workspace()
        self.functions: Dict[str, Any] = {}  # name → FunctionDef or callable
        self.output_buffer = io.StringIO()
        self.ans = None
        self._index_sizes = []  # Stack of (size,) for end keyword resolution
        self._setup_builtins()
        self._setup_constants()
        self._session_ref = None  # Set by ForgeSession to enable .m file discovery

    def _setup_constants(self):
        self.workspace.set("pi", ForgeArray(FORGE_PI))
        self.workspace.set("e", ForgeArray(FORGE_E))
        self.workspace.set("Inf", ForgeArray(FORGE_INF))
        self.workspace.set("inf", ForgeArray(FORGE_INF))
        self.workspace.set("NaN", ForgeArray(FORGE_NAN))
        self.workspace.set("nan", ForgeArray(FORGE_NAN))
        self.workspace.set("eps", ForgeArray(FORGE_EPS))
        self.workspace.set("realmin", ForgeArray(np.atleast_2d(np.finfo(float).tiny)))
        self.workspace.set("realmax", ForgeArray(np.atleast_2d(np.finfo(float).max)))
        self.workspace.set("i", ForgeArray(FORGE_I))
        self.workspace.set("j", ForgeArray(FORGE_I))
        self.workspace.set("true", ForgeArray(np.array(True)))
        self.workspace.set("false", ForgeArray(np.array(False)))

    def _setup_builtins(self):
        """Register built-in functions."""
        b = self.functions

        # Matrix construction
        b["eye"] = lambda *a: forge_eye(*[int(v) if isinstance(v, float) and v == int(v) else v for v in [_to_py(x) for x in a]])
        b["ones"] = lambda *a: forge_ones(*[int(v) if isinstance(v, float) and v == int(v) else v for v in [_to_py(x) for x in a]])
        b["zeros"] = lambda *a: forge_zeros(*[int(v) if isinstance(v, float) and v == int(v) else v for v in [_to_py(x) for x in a]])
        b["rand"] = lambda *a: forge_rand(*[int(v) if isinstance(v, float) and v == int(v) else v for v in [_to_py(x) for x in a]])
        b["randn"] = lambda *a: forge_randn(*[int(v) if isinstance(v, float) and v == int(v) else v for v in [_to_py(x) for x in a]])
        b["randi"] = lambda *a: forge_randi(*[int(v) if isinstance(v, float) and v == int(v) else v for v in [_to_py(x) for x in a]])
        b["true"] = lambda *a: forge_true(*[_to_py(x) for x in a])
        b["false"] = lambda *a: forge_false(*[_to_py(x) for x in a])
        b["diag"] = lambda *a: forge_diag(*a)
        b["linspace"] = lambda *a: forge_linspace(*[_to_py(x) for x in a])
        b["repmat"] = lambda *a: forge_repmat(*a)

        def _meshgrid(*args):
            from forge.engine.types import ForgeArray
            import numpy as np
            arrays = [_unwrap(a) if isinstance(a, ForgeArray) else np.asarray(a) for a in args]
            arrays = [a.flatten() for a in arrays]
            if len(arrays) == 2:
                X, Y = np.meshgrid(arrays[0], arrays[1])
                return ForgeArray(X), ForgeArray(Y)
            elif len(arrays) == 3:
                X, Y, Z = np.meshgrid(arrays[0], arrays[1], arrays[2], indexing="xy")
                return ForgeArray(X), ForgeArray(Y), ForgeArray(Z)
            elif len(arrays) == 1:
                X, Y = np.meshgrid(arrays[0], arrays[0])
                return ForgeArray(X), ForgeArray(Y)
            else:
                raise ValueError("meshgrid requires 1-3 arguments")

        def _ndgrid(*args):
            from forge.engine.types import ForgeArray
            import numpy as np
            arrays = [_unwrap(a).flatten() if isinstance(a, ForgeArray) else np.asarray(a).flatten() for a in args]
            grids = np.meshgrid(*arrays, indexing="ij")
            return tuple(ForgeArray(g) for g in grids)

        b["meshgrid"] = _meshgrid
        b["ndgrid"] = _ndgrid

        # Type conversion
        b["double"] = forge_double
        b["single"] = forge_single
        b["int8"] = forge_int8
        b["int16"] = forge_int16
        b["int32"] = forge_int32
        b["int64"] = forge_int64
        b["uint8"] = forge_uint8
        b["uint16"] = forge_uint16
        b["uint32"] = forge_uint32
        b["uint64"] = forge_uint64
        b["logical"] = forge_logical
        b["complex"] = forge_complex
        b["char"] = forge_char

        # Type checking
        b["isnumeric"] = lambda x: ForgeArray(np.array(forge_isnumeric(x)))
        b["islogical"] = lambda x: ForgeArray(np.array(forge_islogical(x)))
        b["isfloat"] = lambda x: ForgeArray(np.array(forge_isfloat(x)))
        b["isinteger"] = lambda x: ForgeArray(np.array(forge_isinteger(x)))
        b["iscell"] = lambda x: ForgeArray(np.array(forge_iscell(x)))
        b["isstruct"] = lambda x: ForgeArray(np.array(forge_isstruct(x)))
        b["ischar"] = lambda x: ForgeArray(np.array(forge_ischar(x)))
        b["isnan"] = forge_isnan
        b["isinf"] = forge_isinf
        b["isfinite"] = forge_isfinite
        b["isempty"] = lambda x: ForgeArray(np.array(x.isempty() if isinstance(x, ForgeArray) else len(x) == 0))
        def _isfield(s, f):
            from forge.engine.containers import ForgeStruct, ForgeChar
            if isinstance(f, ForgeChar):
                f = f.to_str()
            if isinstance(s, ForgeStruct):
                return ForgeArray(np.float64(1.0 if f in s._fields else 0.0))
            return ForgeArray(np.float64(0.0))
        b["isfield"] = _isfield
        b["isscalar"] = lambda x: ForgeArray(np.float64(1.0 if (isinstance(x, ForgeArray) and x.isscalar()) or np.isscalar(x) else 0.0))
        b["isvector"] = lambda x: ForgeArray(np.float64(1.0 if (isinstance(x, ForgeArray) and (x.data.ndim <= 1 or min(x.data.shape) == 1)) else 0.0))
        b["ismatrix"] = lambda x: ForgeArray(np.float64(1.0 if isinstance(x, ForgeArray) and x.data.ndim == 2 else 0.0))
        b["issquare"] = lambda x: ForgeArray(np.float64(1.0 if isinstance(x, ForgeArray) and x.data.ndim == 2 and x.data.shape[0] == x.data.shape[1] else 0.0))

        # Math
        for name in ['abs','sqrt','exp','log','log2','log10','sin','cos','tan',
                      'asin','acos','atan','sinh','cosh','tanh','ceil','floor',
                      'round','fix','sign','real','imag','conj','angle']:
            fn = getattr(np, name)
            b[name] = lambda *a, f=fn: ForgeArray(f(_unwrap(a[0])))

        # Add docstrings to math functions
        _docs = {'abs': 'Absolute value. Usage: abs(X)', 'sqrt': 'Square root. Usage: sqrt(X)',
            'exp': 'Exponential (e^X). Usage: exp(X)', 'log': 'Natural logarithm. Usage: log(X)',
            'log2': 'Base-2 logarithm. Usage: log2(X)', 'log10': 'Base-10 logarithm. Usage: log10(X)',
            'sin': 'Sine (radians). See also: cos, tan, asin', 'cos': 'Cosine (radians). See also: sin, tan, acos',
            'tan': 'Tangent (radians). See also: sin, cos, atan', 'asin': 'Inverse sine (radians).',
            'acos': 'Inverse cosine (radians).', 'atan': 'Inverse tangent (radians).',
            'sinh': 'Hyperbolic sine.', 'cosh': 'Hyperbolic cosine.', 'tanh': 'Hyperbolic tangent.',
            'ceil': 'Round toward +infinity.', 'floor': 'Round toward -infinity.',
            'round': 'Round to nearest integer.', 'fix': 'Round toward zero.',
            'sign': 'Signum function (-1, 0, or 1).', 'real': 'Real part of complex number.',
            'imag': 'Imaginary part of complex number.', 'conj': 'Complex conjugate.',
            'angle': 'Phase angle (radians).'}
        for _n, _d in _docs.items():
            if _n in b:
                b[_n].__doc__ = _d

        b["mod"] = lambda a, m: ForgeArray(np.mod(_unwrap(a), _unwrap(m)))
        b["rem"] = lambda a, m: ForgeArray(np.remainder(_unwrap(a), _unwrap(m)))
        b["atan2"] = lambda y, x: ForgeArray(np.arctan2(_unwrap(y), _unwrap(x)))
        b["hypot"] = lambda x, y: ForgeArray(np.hypot(_unwrap(x), _unwrap(y)))
        b["power"] = lambda x, y: ForgeArray(np.power(_unwrap(x), _unwrap(y)))

        # Array ops
        b["size"] = self._builtin_size
        b["length"] = lambda x: ForgeArray(np.array(max(x._shape) if (hasattr(x, "_shape") and len(x._data) > 0) else (0 if hasattr(x, "_data") and len(x._data) == 0 else (x.length() if isinstance(x, ForgeArray) else len(x)))))
        b["numel"] = lambda x: ForgeArray(np.array(x._shape[0]*x._shape[1] if hasattr(x, "_shape") else (x.numel() if isinstance(x, ForgeArray) else len(x))))
        b["ndims"] = lambda x: ForgeArray(np.array(x.ndim if isinstance(x, ForgeArray) else 0))
        b["reshape"] = lambda x, *a: ForgeArray(_unwrap(x).reshape(*[int(_to_py(v)) for v in a]))
        b["squeeze"] = lambda x: ForgeArray(np.squeeze(_unwrap(x)))
        b["permute"] = lambda x, order: ForgeArray(np.transpose(_unwrap(x), [int(_to_py(o))-1 for o in _unwrap(order).flatten()]))
        b["ndims"] = lambda x: ForgeArray(np.float64(max(2, _unwrap(x).ndim)))
        b["transpose"] = lambda x: x.T if isinstance(x, ForgeArray) else ForgeArray(np.asarray(x).T)
        b["sum"] = lambda x, *a: ForgeArray(np.sum(_unwrap(x), axis=_to_py(a[0])-1 if a else None))
        b["prod"] = lambda x, *a: ForgeArray(np.prod(_unwrap(x), axis=_to_py(a[0])-1 if a else None))
        b["min"] = lambda x, *a: ForgeArray(np.min(_unwrap(x))) if not a else ForgeArray(np.minimum(_unwrap(x), _unwrap(a[0])))
        b["max"] = lambda x, *a: ForgeArray(np.max(_unwrap(x))) if not a else ForgeArray(np.maximum(_unwrap(x), _unwrap(a[0])))
        b["sort"] = lambda x, *a: ForgeArray(np.sort(_unwrap(x), axis=-1))
        b["find"] = lambda x: ForgeArray(np.flatnonzero(_unwrap(x)) + 1)  # 1-based
        b["any"] = lambda x, *a: ForgeArray(np.array(np.any(_unwrap(x))))
        b["all"] = lambda x, *a: ForgeArray(np.array(np.all(_unwrap(x))))
        b["cumsum"] = lambda x, *a: ForgeArray(np.cumsum(_unwrap(x), axis=_to_py(a[0])-1 if a else None))
        b["cumprod"] = lambda x, *a: ForgeArray(np.cumprod(_unwrap(x), axis=_to_py(a[0])-1 if a else None))
        b["diff"] = lambda x, *a: ForgeArray(np.diff(_unwrap(x), n=_to_py(a[0]) if a else 1, axis=-1))
        b["cat"] = lambda dim, *arrays: ForgeArray(np.concatenate([_unwrap(a) for a in arrays], axis=_to_py(dim)-1))
        b["horzcat"] = lambda *a: ForgeArray(np.concatenate([_unwrap(x) for x in a if np.asarray(_unwrap(x)).size > 0], axis=1)) if any(np.asarray(_unwrap(x)).size > 0 for x in a) else ForgeArray(np.array([]).reshape(0, 0))
        b["vertcat"] = lambda *a: ForgeArray(np.concatenate([_unwrap(x) for x in a if np.asarray(_unwrap(x)).size > 0], axis=0)) if any(np.asarray(_unwrap(x)).size > 0 for x in a) else ForgeArray(np.array([]).reshape(0, 0))
        b["fliplr"] = lambda x: ForgeArray(np.fliplr(_unwrap(x)))
        b["flipud"] = lambda x: ForgeArray(np.flipud(_unwrap(x)))
        b["rot90"] = lambda x, *a: ForgeArray(np.rot90(_unwrap(x), k=_to_py(a[0]) if a else 1))
        b["triu"] = lambda x, *a: ForgeArray(np.triu(_unwrap(x), k=_to_py(a[0]) if a else 0))
        b["tril"] = lambda x, *a: ForgeArray(np.tril(_unwrap(x), k=_to_py(a[0]) if a else 0))

        # String ops
        b["strcmp"] = lambda a, c: ForgeArray(np.array(forge_strcmp(a, c)))
        b["strcmpi"] = lambda a, c: ForgeArray(np.array(forge_strcmpi(a, c)))
        b["num2str"] = lambda x: ForgeChar(str(_to_py(x)))
        b["str2num"] = lambda s: ForgeArray(float(s.to_str() if isinstance(s, ForgeChar) else s))

        # I/O
        b["disp"] = self._builtin_disp
        b["fprintf"] = self._builtin_fprintf
        b["sprintf"] = self._builtin_sprintf
        b["error"] = self._builtin_error
        b["warning"] = self._builtin_warning
        b["assert"] = self._builtin_assert
        b["clc"] = lambda: None  # Clear command window (no-op in engine)

        # Propagate docstrings for common lambda-wrapped functions
        _fn_docs = {
            'linspace': 'Linearly-spaced vector. Usage: linspace(a, b) or linspace(a, b, n)',
            'logspace': 'Logarithmically-spaced vector. Usage: logspace(a, b, n)',
            'zeros': 'Create array of zeros. Usage: zeros(n) or zeros(m, n)',
            'ones': 'Create array of ones. Usage: ones(n) or ones(m, n)',
            'eye': 'Identity matrix. Usage: eye(n) or eye(m, n)',
            'rand': 'Uniform random numbers in [0,1). Usage: rand(n) or rand(m, n)',
            'randn': 'Standard normal random numbers. Usage: randn(n) or randn(m, n)',
            'diag': 'Create diagonal matrix or extract diagonal. Usage: diag(v) or diag(A, k)',
            'det': 'Matrix determinant. Usage: det(A)',
            'inv': 'Matrix inverse. Usage: inv(A)',
            'eig': 'Eigenvalues and eigenvectors. Usage: [V, D] = eig(A)',
            'svd': 'Singular value decomposition. Usage: [U, S, V] = svd(A)',
            'lu': 'LU decomposition. Usage: [L, U, P] = lu(A)',
            'qr': 'QR decomposition. Usage: [Q, R] = qr(A)',
            'chol': 'Cholesky factorization. Usage: R = chol(A)',
            'norm': 'Vector or matrix norm. Usage: norm(X) or norm(X, p)',
            'pinv': 'Pseudoinverse. Usage: pinv(A)',
            'kron': 'Kronecker tensor product. Usage: kron(A, B)',
            'cross': 'Cross product. Usage: cross(a, b)',
            'dot': 'Dot product. Usage: dot(a, b)',
            'sort': 'Sort elements. Usage: sort(X) or [Y, I] = sort(X)',
            'find': 'Find nonzero elements. Usage: I = find(X)',
            'max': 'Maximum value. Usage: max(X) or max(a, b)',
            'min': 'Minimum value. Usage: min(X) or min(a, b)',
            'sum': 'Sum of elements. Usage: sum(X) or sum(X, dim)',
            'prod': 'Product of elements. Usage: prod(X)',
            'cumsum': 'Cumulative sum. Usage: cumsum(X)',
            'cumprod': 'Cumulative product. Usage: cumprod(X)',
            'diff': 'Differences between adjacent elements. Usage: diff(X) or diff(X, n)',
            'reshape': 'Reshape array. Usage: reshape(A, m, n)',
            'repmat': 'Replicate and tile. Usage: repmat(A, m, n)',
            'cat': 'Concatenate arrays. Usage: cat(dim, A1, A2)',
            'horzcat': 'Horizontal concatenation. Usage: horzcat(A, B)',
            'vertcat': 'Vertical concatenation. Usage: vertcat(A, B)',
            'fft': 'Fast Fourier transform. Usage: fft(X) or fft(X, n)',
            'ifft': 'Inverse FFT. Usage: ifft(X)',
            'plot': '2-D line plot. Usage: plot(Y) or plot(X, Y) or plot(X, Y, fmt)',
            'figure': 'Create or select figure window. Usage: figure or figure(n)',
            'disp': 'Display value of expression. Usage: disp(X)',
            'fprintf': 'Write formatted data. Usage: fprintf(fmt, ...) or fprintf(fid, fmt, ...)',
            'sprintf': 'Format data into string. Usage: sprintf(fmt, ...)',
            'strcmp': 'Compare strings. Usage: strcmp(s1, s2). Returns 1 if equal.',
            'strcmpi': 'Case-insensitive string compare. Usage: strcmpi(s1, s2)',
            'size': 'Array dimensions. Usage: size(A) or [m, n] = size(A)',
            'length': 'Length of largest dimension. Usage: length(X)',
            'numel': 'Number of elements. Usage: numel(X)',
            'transpose': 'Matrix transpose. Usage: transpose(A) or A.\' ',
            'mod': 'Modulus after division. Usage: mod(a, m)',
            'rem': 'Remainder after division. Usage: rem(a, b)',
            'atan2': 'Four-quadrant inverse tangent. Usage: atan2(y, x)',
            'hypot': 'Hypotenuse. Usage: hypot(x, y) = sqrt(x^2+y^2)',
            'any': 'True if any element is nonzero. Usage: any(X)',
            'all': 'True if all elements are nonzero. Usage: all(X)',
            'isnan': 'Test for NaN. Usage: isnan(X)',
            'isinf': 'Test for Inf. Usage: isinf(X)',
            'isfinite': 'Test for finite values. Usage: isfinite(X)',
            'isempty': 'Test if array is empty. Usage: isempty(X)',
            'isnumeric': 'Test if numeric. Usage: isnumeric(X)',
            'islogical': 'Test if logical. Usage: islogical(X)',
            'ischar': 'Test if character array. Usage: ischar(X)',
            'class': 'Class of object. Usage: class(X)',
            'typecast': 'Convert without changing bits. Usage: typecast(X, type)',
        }
        for _fn_name, _fn_doc in _fn_docs.items():
            if _fn_name in b and not getattr(b[_fn_name], '__doc__', None):
                try:
                    b[_fn_name].__doc__ = _fn_doc
                except AttributeError:
                    pass

        def _help_func(*args):
            if not args:
                self.output_buffer.write("Forge help system. Use: help <function_name>" + chr(10))
                return
            name = args[0]
            if hasattr(name, "to_str"):
                func_name = name.to_str()
            else:
                func_name = str(name)
            func = self.functions.get(func_name)
            if func and hasattr(func, "__doc__") and func.__doc__:
                self.output_buffer.write("  " + func_name + ": " + func.__doc__ + chr(10))
            else:
                self.output_buffer.write("  " + func_name + ": No help available." + chr(10))
        b["help"] = _help_func

        def _which_func(*args):
            if not args:
                return
            name = args[0]
            if hasattr(name, "to_str"):
                name = name.to_str()
            name = str(name)
            if name in self.functions:
                self.output_buffer.write("  " + name + " is a built-in function" + chr(10))
            else:
                self.output_buffer.write("  " + name + " is undefined" + chr(10))
        b["which"] = _which_func

        # Struct/cell
        b["struct"] = lambda *a: forge_struct(*a)
        b["fieldnames"] = forge_fieldnames
        b["cell"] = lambda *a: forge_cell(*[int(v) if isinstance(v, float) and v == int(v) else v for v in [_to_py(x) for x in a]])

        # R15: dot product
        b["dot"] = lambda a, b: ForgeArray(np.dot(_unwrap(a).flatten(), _unwrap(b).flatten()))

        # R15: sub2ind / ind2sub
        def _forge_sub2ind(sz, *args):
            sz_arr = _unwrap(sz).ravel().astype(int)
            if len(args) == 2:
                r = _unwrap(args[0]).ravel().astype(int)
                c = _unwrap(args[1]).ravel().astype(int)
                m = sz_arr[0]
                idx = r + (c - 1) * m
                return ForgeArray(idx.astype(float))
            raise ValueError("sub2ind requires size and subscript arguments")
        b["sub2ind"] = _forge_sub2ind

        def _forge_ind2sub(sz, idx):
            sz_arr = _unwrap(sz).ravel().astype(int)
            idx_arr = _unwrap(idx).ravel().astype(int)
            m = sz_arr[0]
            r = ((idx_arr - 1) % m) + 1
            c = ((idx_arr - 1) // m) + 1
            if idx_arr.size == 1:
                return (ForgeArray(np.array(float(r[0]))), ForgeArray(np.array(float(c[0]))))
            return (ForgeArray(r.astype(float)), ForgeArray(c.astype(float)))
        b["ind2sub"] = _forge_ind2sub

        # R15: tic/toc
        _tic_stack = []
        def _forge_tic():
            _tic_stack.append(time.time())
        def _forge_toc():
            if _tic_stack:
                elapsed = time.time() - _tic_stack.pop()
                self.output_buffer.write(f"Elapsed time is {elapsed:.6f} seconds.\n")
                return ForgeArray(np.array(elapsed))
            self.output_buffer.write("No tic was called.\n")
            return ForgeArray(np.array(0.0))
        b["tic"] = _forge_tic
        b["toc"] = _forge_toc

        # Misc
        def _class_name(x):
            from forge.engine.containers import ForgeChar as FC, ForgeCell, ForgeStruct
            if isinstance(x, ForgeArray):
                # Map numpy dtype to MATLAB class name
                _dtype_map = {
                    "float64": "double", "float32": "single",
                    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
                    "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
                    "bool": "logical", "complex128": "double", "complex64": "single",
                }
                dtype_name = str(x.data.dtype)
                return FC(_dtype_map.get(dtype_name, dtype_name))
            if isinstance(x, FC):
                return FC("char")
            if isinstance(x, ForgeCell):
                return FC("cell")
            if isinstance(x, ForgeStruct):
                return FC("struct")
            if isinstance(x, bool):
                return FC("logical")
            if isinstance(x, (int, float)):
                return FC("double")
            if isinstance(x, str):
                return FC("char")
            return FC(type(x).__name__)
        b["class"] = _class_name
        b["typecast"] = lambda x, t: x.astype(t.to_str() if isinstance(t, ForgeChar) else str(t))

        # narginchk / nargoutchk
        def _narginchk(minargs, maxargs):
            # These are checked at call time; standalone just validates
            pass
        def _nargoutchk(minargs, maxargs):
            pass
        b["narginchk"] = _narginchk
        b["nargoutchk"] = _nargoutchk

        # exist(name, type) - check if name exists
        def _exist(name, *args):
            name = name.to_str() if isinstance(name, ForgeChar) else str(name)
            kind = args[0].to_str() if args and isinstance(args[0], ForgeChar) else (str(args[0]) if args else "")
            if kind == "var":
                return ForgeArray(np.array(1.0 if self.workspace.has(name) else 0.0))
            if kind == "file":
                return ForgeArray(np.array(2.0 if os.path.isfile(name) else 0.0))
            if kind == "dir":
                return ForgeArray(np.array(7.0 if os.path.isdir(name) else 0.0))
            if kind == "builtin":
                return ForgeArray(np.array(5.0 if name in b else 0.0))
            # Default: check var, then builtin, then file
            if self.workspace.has(name):
                return ForgeArray(np.array(1.0))
            if name in b:
                return ForgeArray(np.array(5.0))
            if os.path.isfile(name):
                return ForgeArray(np.array(2.0))
            if os.path.isdir(name):
                return ForgeArray(np.array(7.0))
            return ForgeArray(np.array(0.0))
        b["exist"] = _exist

        # feval(funcname, args...) - call function by name
        def _feval(name, *args):
            name = name.to_str() if isinstance(name, ForgeChar) else str(name)
            if name in self.builtins:
                return self.builtins[name](*args)
            raise NameError(f"undefined function '{name}'")
        b["feval"] = _feval

        # mfilename - returns empty (we're not in an m-file in REPL)
        b["mfilename"] = lambda *a: ForgeChar("")

        # inputname(n) - returns empty in REPL context
        b["inputname"] = lambda n: ForgeChar("")

        # nargs / nargout as functions (for querying builtins)
        def _nargout_query(*args):
            if args:
                name = args[0].to_str() if isinstance(args[0], ForgeChar) else str(args[0])
                if name in b and hasattr(b[name], '__code__'):
                    return ForgeArray(np.array(-1.0))  # -1 = variable
                return ForgeArray(np.array(-1.0))
            return ForgeArray(np.array(0.0))
        b["nargout"] = _nargout_query

        def _nargin_query(*args):
            if args:
                name = args[0].to_str() if isinstance(args[0], ForgeChar) else str(args[0])
                if name in b:
                    func = b[name]
                    if hasattr(func, '__code__'):
                        import inspect
                        try:
                            sig = inspect.signature(func)
                            # Count non-var params
                            count = sum(1 for p in sig.parameters.values()
                                       if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
                            has_var = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
                            return ForgeArray(np.array(float(-count if has_var else count)))
                        except (ValueError, TypeError):
                            pass
                return ForgeArray(np.array(-1.0))
            return ForgeArray(np.array(0.0))
        b["nargin"] = _nargin_query

        # Merge toolbox registries (elfun, general, specfun, ...)
        b.update(BUILTIN_REGISTRY)

    def _builtin_size(self, x, *args):
        from forge.engine.containers import ForgeCell
        if isinstance(x, ForgeCell):
            shape = x._shape
            if args:
                dim = int(_to_py(args[0]))
                return ForgeArray(np.array(shape[dim-1] if dim <= len(shape) else 1))
            return ForgeArray(np.array(list(shape)))
        if not isinstance(x, ForgeArray):
            return ForgeArray(np.array([1, 1]))
        if args:
            dim = _to_py(args[0])
            dim = int(dim); return ForgeArray(np.array(x.shape[dim-1] if dim <= x.ndim else 1))
        return ForgeArray(np.array(x.shape))

    def _builtin_disp(self, *args):
        for a in args:
            if isinstance(a, ForgeChar):
                self.output_buffer.write(a.to_str() + "\n")
            elif isinstance(a, ForgeArray):
                self.output_buffer.write(str(a.data) + "\n")
            else:
                self.output_buffer.write(str(a) + "\n")

    def _builtin_fprintf(self, *args):
        fmt = args[0]
        if isinstance(fmt, ForgeChar):
            fmt = fmt.to_str()
        fmt = _process_c_escapes(fmt)
        # Handle array arguments: in MATLAB, fprintf repeats the format
        # for each element when given array args
        # Convert ForgeChar to string for %s format specifiers
        raw_vals = []
        for a in args[1:]:
            if isinstance(a, ForgeChar):
                raw_vals.append(a.to_str())
            else:
                raw_vals.append(_to_py(a))
        # Check if any arg is an array with multiple elements
        import numpy as _np
        has_array = any(hasattr(v, '__len__') and _np.asarray(v).size > 1 for v in raw_vals)
        if has_array and len(raw_vals) == 1:
            # Single array arg: repeat format for each element
            arr = _np.asarray(raw_vals[0]).ravel()
            out = ''
            for val in arr:
                out += fmt % (float(val),)
            self.output_buffer.write(out)
        elif not has_array:
            # All scalar args — handle complex by taking real part for real formats
            def _to_real_if_needed(v):
                if isinstance(v, (_np.complexfloating, complex)):
                    return float(v.real)
                if isinstance(v, (_np.floating, _np.integer)):
                    return float(v)
                return v
            vals = tuple(_to_real_if_needed(v) for v in raw_vals)
            self.output_buffer.write(fmt % vals)
        else:
            # Multiple args, some arrays: try element-wise
            arrays = [_np.asarray(v).ravel() for v in raw_vals]
            max_len = max(len(a) for a in arrays)
            out = ''
            for i in range(max_len):
                vals = tuple(float(a[min(i, len(a)-1)]) for a in arrays)
                out += fmt % vals
            self.output_buffer.write(out)

    def _builtin_sprintf(self, *args):
        fmt = args[0]
        if isinstance(fmt, ForgeChar):
            fmt = fmt.to_str()
        fmt = _process_c_escapes(fmt)
        vals = []
        for a in args[1:]:
            if isinstance(a, ForgeChar):
                vals.append(a.to_str())
            else:
                vals.append(_to_py(a))
        return ForgeChar(fmt % tuple(vals))

    def _builtin_error(self, *args):
        if len(args) >= 2:
            ident = args[0]
            msg = args[1]
            if isinstance(ident, ForgeChar):
                ident = ident.to_str()
            if isinstance(msg, ForgeChar):
                msg = msg.to_str()
            raise ForgeError(ident, msg)
        msg = args[0]
        if isinstance(msg, ForgeChar):
            msg = msg.to_str()
        raise ForgeError("", str(msg))

    def _builtin_assert(self, *args):
        """MATLAB-style assert: assert(cond), assert(obs, exp), assert(obs, exp, tol)."""
        if len(args) == 1:
            val = args[0]
            if isinstance(val, ForgeArray):
                if not np.all(val.data != 0):
                    raise ForgeError("assert failed")
            elif not val:
                raise ForgeError("assert failed")
        elif len(args) == 2:
            obs = _unwrap(args[0]) if isinstance(args[0], ForgeArray) else np.asarray(args[0])
            exp = _unwrap(args[1]) if isinstance(args[1], ForgeArray) else np.asarray(args[1])
            if not np.allclose(obs, exp):
                raise ForgeError(f"assert failed: observed differs from expected")
        elif len(args) == 3:
            obs = _unwrap(args[0]) if isinstance(args[0], ForgeArray) else np.asarray(args[0])
            exp = _unwrap(args[1]) if isinstance(args[1], ForgeArray) else np.asarray(args[1])
            tol = float(_to_py(args[2]))
            if not np.allclose(obs, exp, atol=tol, rtol=0):
                raise ForgeError(f"assert failed: max diff = {np.max(np.abs(obs - exp)):.4e}, tol = {tol:.4e}")

    def _builtin_warning(self, *args):
        msg = args[0]
        if isinstance(msg, ForgeChar):
            msg = msg.to_str()
        self.output_buffer.write(f"warning: {msg}\n")

    def _resolve_m_file(self, name):
        """Search CWD then session path for {name}.m, parse and register it."""
        if self._session_ref is None:
            return None
        # Search current working directory first (MATLAB/Octave behavior)
        search_dirs = [os.getcwd()] + list(self._session_ref.path)
        for directory in search_dirs:
            mfile = os.path.join(directory, name + ".m")
            if os.path.isfile(mfile):
                with open(mfile, "r") as f:
                    source = f.read()
                stmts = parse(source)
                if stmts and isinstance(stmts[0], FunctionDef):
                    # It defines a function - register it
                    self._exec(stmts[0], self.workspace)
                    return self.functions.get(name)
                else:
                    # It is a script - execute all statements
                    for stmt in stmts:
                        self._exec(stmt, self.workspace)
                    return "__script__"
        return None

    def eval(self, source: str) -> Any:
        """Parse and evaluate M-language source code."""
        stmts = parse(source)
        return self._exec_stmts(stmts, self.workspace)

    def _exec_stmts(self, stmts, ws: Workspace) -> Any:
        result = None
        for stmt in stmts:
            result = self._exec(stmt, ws)
        return result

    def _exec(self, node, ws: Workspace) -> Any:
        """Execute a single AST node."""
        if isinstance(node, ExpressionStatement):
            val = self._eval_expr(node.expr, ws)
            # R05: If result is a bare callable (command-style: who, whos, etc.)
            # and the expression is a bare Identifier (not Index/call), auto-invoke
            if callable(val) and not isinstance(val, ForgeArray):
                if isinstance(node.expr, Identifier):
                    val = val()
            if val is not None:
                ws.set("ans", val)
                self.ans = val
            if node.print_result:
                return val
            return None

        if isinstance(node, Assignment):
            return self._exec_assign(node, ws)

        if isinstance(node, IfStatement):
            return self._exec_if(node, ws)

        if isinstance(node, ForStatement):
            return self._exec_for(node, ws)

        if isinstance(node, WhileStatement):
            return self._exec_while(node, ws)

        if isinstance(node, DoUntilStatement):
            return self._exec_do_until(node, ws)

        if isinstance(node, SwitchStatement):
            return self._exec_switch(node, ws)

        if isinstance(node, TryCatchStatement):
            return self._exec_try(node, ws)

        if isinstance(node, ClassDef):
            return self._exec_classdef(node, ws)

        if isinstance(node, UnwindProtect):
            return self._exec_unwind_protect(node, ws)

        if isinstance(node, ReturnStatement):
            raise ReturnSignal()

        if isinstance(node, BreakStatement):
            raise BreakSignal()

        if isinstance(node, ContinueStatement):
            raise ContinueSignal()

        if isinstance(node, FunctionDef):
            self.functions[node.name] = node
            return None

        if isinstance(node, GlobalStatement):
            for name in node.names:
                ws._globals.add(name)
            return None

        if isinstance(node, PersistentStatement):
            return None

        raise RuntimeError(f"Unknown AST node: {type(node).__name__}")

    # ============================================================
    # Expression evaluation
    # ============================================================

    def _eval_expr(self, node, ws: Workspace) -> Any:
        if isinstance(node, NumberLiteral):
            return self._eval_number(node)

        if isinstance(node, StringLiteral):
            if node.is_char:
                return ForgeChar(node.value)
            return ForgeChar(node.value)  # Both become char for now

        if isinstance(node, Identifier):
            name = node.name
            if ws.has(name):
                return ws.get(name)
            if name in self.functions:
                return self.functions[name]
            # R13: Try .m file auto-discovery
            resolved = self._resolve_m_file(name)
            if resolved is not None and resolved != "__script__":
                return self.functions[name] if name in self.functions else resolved
            if resolved == "__script__":
                return ws.get(name) if ws.has(name) else None
            raise NameError(f"Undefined variable or function: {name}")

        if isinstance(node, UnaryOp):
            val = self._eval_expr(node.operand, ws)
            if node.op == "-":
                return -val if isinstance(val, ForgeArray) else ForgeArray(-np.asarray(val))
            if node.op == "~":
                return ForgeArray(~np.asarray(_unwrap(val), dtype=bool))

        if isinstance(node, BinaryOp):
            return self._eval_binop(node, ws)

        if isinstance(node, CompareOp):
            return self._eval_compare(node, ws)

        if isinstance(node, LogicalOp):
            return self._eval_logical(node, ws)

        if isinstance(node, TransposeOp):
            val = self._eval_expr(node.operand, ws)
            if isinstance(val, ForgeArray):
                if node.conjugate:
                    return ForgeArray(np.conj(val.data).T)
                return val.T
            return ForgeArray(np.asarray(val).T)

        if isinstance(node, BareColon):
            return None

        if isinstance(node, ColonExpr):
            start = _to_py(self._eval_expr(node.start, ws))
            stop = _to_py(self._eval_expr(node.stop, ws))
            if node.step is not None:
                step = _to_py(self._eval_expr(node.step, ws))
                return forge_colon(start, step, stop)
            return forge_colon(start, stop)

        if isinstance(node, Index):
            return self._eval_index(node, ws)

        if isinstance(node, CellIndex):
            target = self._eval_expr(node.target, ws)
            if isinstance(target, ForgeCell):
                args = [_to_int(self._eval_expr(a, ws)) for a in node.args]
                return target.content_get(*args)
            raise TypeError("Cell indexing on non-cell")

        if isinstance(node, FieldAccess):
            target = self._eval_expr(node.target, ws)
            if isinstance(target, ForgeStruct):
                return target._fields[node.field]
            # Support ForgeObject (classdef instances)
            from forge.engine.classdef import ForgeObject
            if isinstance(target, ForgeObject):
                return getattr(target, node.field)
            raise TypeError(f"Field access on {type(target).__name__}")

        if isinstance(node, DynamicFieldAccess):
            target = self._eval_expr(node.target, ws)
            field = self._eval_expr(node.field_expr, ws)
            if isinstance(field, ForgeChar):
                field = field.to_str()
            from forge.engine.classdef import ForgeObject as _FO
            if isinstance(target, _FO):
                return getattr(target, str(field))
            return target._fields[str(field)]

        if isinstance(node, MatrixLiteral):
            return self._eval_matrix(node, ws)

        if isinstance(node, CellLiteral):
            return self._eval_cell_literal(node, ws)

        if isinstance(node, FunctionHandle):
            if node.name in self.functions:
                fn = self.functions[node.name]
                if isinstance(fn, FunctionDef):
                    return lambda *a: self._call_function(fn, list(a), ws)
                return fn
            raise NameError(f"Undefined function: {node.name}")

        if isinstance(node, AnonFunction):
            params = node.args
            body = node.body
            captured_ws = ws
            def anon(*args):
                local_ws = Workspace(captured_ws)
                for p, a in zip(params, args):
                    local_ws.set(p, a)
                return self._eval_expr(body, local_ws)
            return anon

        if isinstance(node, EndKeyword):
            if self._index_sizes:
                return ForgeArray(np.array(float(self._index_sizes[-1])))
            raise RuntimeError("'end' used outside of indexing context")

        raise RuntimeError(f"Cannot evaluate: {type(node).__name__}")

    def _eval_number(self, node: NumberLiteral) -> ForgeArray:
        v = node.value
        if v == "true":
            return ForgeArray(np.array(True))
        if v == "false":
            return ForgeArray(np.array(False))
        if v.startswith("0x") or v.startswith("0X"):
            return ForgeArray(int(v, 16))
        if v.startswith("0b") or v.startswith("0B"):
            return ForgeArray(int(v, 2))
        if v.endswith("i") or v.endswith("j"):
            return ForgeArray(complex(0, float(v[:-1])))
        if "." in v or "e" in v.lower():
            return ForgeArray(float(v))
        return ForgeArray(float(v))  # Octave: all numbers are double by default

    def _eval_binop(self, node: BinaryOp, ws: Workspace) -> ForgeArray:
        left = self._eval_expr(node.left, ws)
        right = self._eval_expr(node.right, ws)
        # Auto-call zero-arg callables used as values (e.g., toc * 1000)
        if callable(left) and not isinstance(left, (ForgeArray, type)):
            left = left()
        if callable(right) and not isinstance(right, (ForgeArray, type)):
            right = right()
        l, r = _unwrap(left), _unwrap(right)
        op = node.op
        if op == "+": return ForgeArray(l + r)
        if op == "-": return ForgeArray(l - r)
        if op == "*": return ForgeArray(l @ r if _is_matrix_op(l, r) else l * r)
        if op == "/": return ForgeArray(np.linalg.solve(r.T, l.T).T if _is_matrix_op(l, r) else l / r)
        if op == "\\": return ForgeArray(np.linalg.solve(l, r) if _is_matrix_op(l, r) else r / l)
        if op == "^": return ForgeArray(np.linalg.matrix_power(l, int(r.flat[0])) if (l.ndim == 2 and l.size > 1) else l ** r)
        if op == ".*": return ForgeArray(l * r)
        if op == "./": return ForgeArray(l / r)
        if op == ".\\": return ForgeArray(r / l)
        if op == ".^": return ForgeArray(l ** r)
        raise RuntimeError(f"Unknown binary op: {op}")

    def _eval_compare(self, node: CompareOp, ws: Workspace) -> ForgeArray:
        left = _unwrap(self._eval_expr(node.left, ws))
        right = _unwrap(self._eval_expr(node.right, ws))
        op = node.op
        if op == "==": return ForgeArray(left == right)
        if op == "~=" or op == "!=": return ForgeArray(left != right)
        if op == "<": return ForgeArray(left < right)
        if op == ">": return ForgeArray(left > right)
        if op == "<=": return ForgeArray(left <= right)
        if op == ">=": return ForgeArray(left >= right)
        raise RuntimeError(f"Unknown comparison: {op}")

    def _eval_logical(self, node: LogicalOp, ws: Workspace) -> ForgeArray:
        left = _unwrap(self._eval_expr(node.left, ws))
        op = node.op
        if op == "&&":
            if left.size != 1:
                raise ForgeError("Octave:nonconformant-args",
                    "binary operator '&&' with a 'matrix' operand must be converted to a scalar")
            if not bool(left.flat[0]):
                return ForgeArray(np.array(False))
            right = _unwrap(self._eval_expr(node.right, ws))
            if right.size != 1:
                raise ForgeError("Octave:nonconformant-args",
                    "binary operator '&&' with a 'matrix' operand must be converted to a scalar")
            return ForgeArray(np.array(bool(right.flat[0])))
        if op == "||":
            if left.size != 1:
                raise ForgeError("Octave:nonconformant-args",
                    "binary operator '||' with a 'matrix' operand must be converted to a scalar")
            if bool(left.flat[0]):
                return ForgeArray(np.array(True))
            right = _unwrap(self._eval_expr(node.right, ws))
            if right.size != 1:
                raise ForgeError("Octave:nonconformant-args",
                    "binary operator '||' with a 'matrix' operand must be converted to a scalar")
            return ForgeArray(np.array(bool(right.flat[0])))
        right = _unwrap(self._eval_expr(node.right, ws))
        if op == "&": return ForgeArray(np.logical_and(left, right))
        if op == "|": return ForgeArray(np.logical_or(left, right))
        raise RuntimeError(f"Unknown logical op: {op}")

    def _eval_index(self, node: Index, ws: Workspace) -> Any:
        # Step 1: Resolve the target
        name = node.target.name if isinstance(node.target, Identifier) else None
        try:
            target = self._eval_expr(node.target, ws)
        except (NameError, ForgeError) as e:
            # R13: If the name is not found, try .m file discovery before failing
            if name:
                resolved = self._resolve_m_file(name)
                if resolved is not None and resolved != "__script__":
                    target = self.functions[name] if name in self.functions else resolved
                else:
                    raise UndefinedFunctionError(f"Undefined function: {name}")
            else:
                raise

        # Step 2: If target is callable (function, lambda, etc.), call it
        if callable(target) and not isinstance(target, ForgeArray):
            # Command-style functions: convert bare identifiers to strings
            _CMD_STYLE = {"hold", "axis", "format", "grid", "box",
                          "legend", "title", "xlabel", "ylabel", "zlabel",
                          "colormap", "shading", "view", "cd"}
            if name and name in _CMD_STYLE:
                args = []
                for a in node.args:
                    if isinstance(a, Identifier) and not ws.has(a.name) and a.name not in self.functions:
                        args.append(ForgeChar(a.name))
                    else:
                        args.append(self._eval_expr(a, ws))
            else:
                args = [self._eval_expr(a, ws) for a in node.args]
            result = target(*args)
            return result

        # Step 3: If target is a FunctionDef AST node, call it
        if isinstance(target, FunctionDef):
            args = [self._eval_expr(a, ws) for a in node.args]
            return self._call_function(target, args, ws)

        # Step 3.5: If it's a cell array, () indexing returns element (struct array behavior)
        if isinstance(target, ForgeCell):
            args = [self._eval_expr(a, ws) for a in node.args]
            if len(args) == 1:
                idx = int(_to_py(args[0]))
                return target._data[idx - 1]  # 1-based
            raise TypeError("Cannot index ForgeCell with multiple indices via ()")

        # Step 4: If it's an array, do indexing (with end support)
        if isinstance(target, ForgeArray):
            n_args = len(node.args)
            args = []
            for dim_idx, a in enumerate(node.args):
                if isinstance(a, BareColon):
                    # Bare colon: no end resolution needed
                    args.append(None)
                    continue
                if n_args == 1:
                    self._index_sizes.append(target.numel())
                else:
                    sz = target.shape[dim_idx] if dim_idx < target.ndim else 1
                    self._index_sizes.append(sz)
                try:
                    args.append(self._eval_expr(a, ws))
                finally:
                    self._index_sizes.pop()
            return self._do_index(target, args)

        # Step 5: Check if name is a registered function
        if name and name in self.functions:
            fn = self.functions[name]
            args = [self._eval_expr(a, ws) for a in node.args]
            if callable(fn):
                return fn(*args)
            if isinstance(fn, FunctionDef):
                return self._call_function(fn, args, ws)

        # Fall back to indexing
        args = [self._eval_expr(a, ws) for a in node.args]
        return self._do_index(target, args)

    def _do_index(self, target, args):
        """Perform array indexing with evaluated args."""
        from forge.engine.containers import ForgeChar
        # ForgeChar indexing returns a char substring
        if isinstance(target, ForgeChar):
            s = target.to_str()
            if len(args) == 1:
                idx = args[0]
                if idx is None:
                    return ForgeChar(s)
                if isinstance(idx, ForgeArray):
                    raw_idx = _unwrap(idx)
                    if raw_idx.dtype == np.bool_:
                        chars = [s[i] for i in range(len(s)) if raw_idx.ravel()[i]]
                        return ForgeChar(''.join(chars))
                    else:
                        indices = raw_idx.astype(int).ravel() - 1
                        chars = [s[i] for i in indices if 0 <= i < len(s)]
                        return ForgeChar(''.join(chars))
                idx_int = int(_to_py(idx)) - 1
                if 0 <= idx_int < len(s):
                    return ForgeChar(s[idx_int])
                raise IndexError(f"String index {idx_int+1} out of range (length {len(s)})")
            raise IndexError("Too many indices for string")
        if not isinstance(target, ForgeArray):
            raise TypeError(f"Cannot index {type(target).__name__}")
        data = _unwrap(target)

        if len(args) == 1:
            idx = args[0]
            if idx is None:
                # A(:) - flatten to column vector
                return ForgeArray(data.ravel(order='F').reshape(-1, 1))
            if isinstance(idx, ForgeArray):
                raw_idx = _unwrap(idx)
                if raw_idx.dtype == np.bool_:
                    # Logical indexing
                    return ForgeArray(data.ravel(order='F')[raw_idx.ravel()])
                else:
                    # Numeric indexing (1-based) - linear indexing into column-major
                    flat = data.ravel(order='F')
                    int_idx = raw_idx.astype(int).ravel() - 1  # Convert to 0-based
                    result = flat[int_idx]
                    # Preserve shape: if source is column vector, result is column
                    # If source is row vector, result is row
                    if data.ndim == 2 and data.shape[1] == 1:
                        result = result.reshape(-1, 1)  # Column vector
                    elif data.ndim == 2 and data.shape[0] == 1:
                        result = result.reshape(1, -1)  # Row vector
                    return ForgeArray(result)
            # Single scalar index
            idx_int = int(_to_py(idx))
            flat = data.ravel(order='F')
            return ForgeArray(np.array(flat[idx_int - 1]))  # 1-based

        # Multi-dimensional indexing
        slices = []
        for dim, arg in enumerate(args):
            if arg is None:
                # : means select all along this dimension
                slices.append(slice(None))
            elif isinstance(arg, ForgeArray):
                raw = _unwrap(arg)
                if raw.dtype == np.bool_:
                    slices.append(raw.ravel())
                else:
                    slices.append(raw.astype(int).ravel() - 1)  # 1-based to 0-based
            else:
                idx_int = int(_to_py(arg))
                slices.append(idx_int - 1)  # 1-based to 0-based

        result = data[np.ix_(*[s if isinstance(s, np.ndarray) else (np.arange(data.shape[i]) if isinstance(s, slice) else np.array([s])) for i, s in enumerate(slices)])]
        # Squeeze singleton dimensions from scalar indices
        squeeze_axes = [i for i, s in enumerate(slices) if isinstance(s, (int, np.integer))]
        for ax in sorted(squeeze_axes, reverse=True):
            result = np.squeeze(result, axis=ax)
        return ForgeArray(result)

    def _eval_matrix(self, node: MatrixLiteral, ws: Workspace) -> ForgeArray:
        if not node.rows:
            return ForgeArray(np.array([]).reshape(0, 0))
        # Check if all elements are ForgeChar (string/char concatenation)
        raw_vals = []
        all_char = True
        for row in node.rows:
            row_raw = [self._eval_expr(e, ws) for e in row]
            raw_vals.append(row_raw)
            for v in row_raw:
                if not isinstance(v, ForgeChar):
                    all_char = False
        if all_char and raw_vals:
            # Concatenate as char arrays (MATLAB: ['hello', ' ', 'world'] => 'hello world')
            result_str = ""
            for row_raw in raw_vals:
                for v in row_raw:
                    result_str += v.to_str()
            return ForgeChar(result_str)
        # Standard numeric concatenation
        rows = []
        for row_raw in raw_vals:
            vals = [_unwrap(v) for v in row_raw]
            vals = [np.atleast_2d(v) if np.asarray(v).ndim < 2 else v for v in vals]
            # Filter out empty arrays (MATLAB skips [] in concatenation)
            vals = [v for v in vals if v.size > 0]
            if not vals:
                continue
            rows.append(np.concatenate(vals, axis=1))
        if not rows:
            return ForgeArray(np.array([]).reshape(0, 0))
        return ForgeArray(np.concatenate(rows, axis=0))

    def _eval_cell_literal(self, node: CellLiteral, ws: Workspace) -> ForgeCell:
        elements = []
        for row in node.rows:
            for e in row:
                elements.append(self._eval_expr(e, ws))
        return ForgeCell(elements)

    # ============================================================
    # Statement execution
    # ============================================================

    def _exec_assign(self, node: Assignment, ws: Workspace) -> Any:
        value = self._eval_expr(node.value, ws)
        # Auto-call bare callable identifiers on RHS (e.g., t = toc)
        if callable(value) and not isinstance(value, ForgeArray):
            if isinstance(node.value, Identifier):
                value = value()
        target = node.targets

        if isinstance(target, list):
            nargout = len(target)
            if not isinstance(value, tuple) and nargout > 1:
                val_tuple = self._eval_multi_output(node.value, ws, nargout)
                if val_tuple is not None and isinstance(val_tuple, tuple):
                    value = val_tuple
            if isinstance(value, tuple):
                for t, v in zip(target, value):
                    if t.name != "~":  # ~ = discard output
                        ws.set(t.name, v)
            else:
                if target[0].name != "~":
                    ws.set(target[0].name, value)
            return value

        if isinstance(target, Identifier):
            ws.set(target.name, value)
            return value

        if isinstance(target, Index):
            target_name = target.target.name if isinstance(target.target, Identifier) else None
            arr = self._eval_expr(target.target, ws)
            n_args = len(target.args)
            args = []
            for dim_idx, a in enumerate(target.args):
                if isinstance(a, BareColon):
                    args.append(None)
                    continue
                if isinstance(arr, ForgeArray):
                    if n_args == 1:
                        self._index_sizes.append(arr.numel())
                    else:
                        sz = arr.shape[dim_idx] if dim_idx < arr.ndim else 1
                        self._index_sizes.append(sz)
                try:
                    args.append(self._eval_expr(a, ws))
                finally:
                    if isinstance(arr, ForgeArray) and not isinstance(a, BareColon):
                        self._index_sizes.pop()
            assign_val = _unwrap(value) if isinstance(value, ForgeArray) else value
            if isinstance(arr, ForgeArray):
                data = arr.data
                if len(args) == 1:
                    idx = args[0]
                    if idx is None:
                        data[:] = assign_val
                    elif isinstance(idx, ForgeArray):
                        raw_idx = _unwrap(idx)
                        if raw_idx.dtype == np.bool_:
                            # Squeeze assign_val for boolean indexing
                            av = assign_val
                            if isinstance(av, np.ndarray) and av.ndim > 1:
                                if av.size == 1:
                                    av = av.flat[0]
                                else:
                                    av = av.ravel()
                            # Use direct boolean mask (reshape to match data)
                            if raw_idx.shape == data.shape:
                                data[raw_idx] = av
                            else:
                                # Shapes differ: broadcast mask
                                mask_bc = np.broadcast_to(raw_idx, data.shape).copy()
                                data[mask_bc] = av
                        else:
                            flat = data.ravel(order='F')
                            int_idx = raw_idx.astype(int).ravel() - 1
                            if np.isscalar(assign_val):
                                flat[int_idx] = assign_val
                            else:
                                flat[int_idx] = np.asarray(assign_val).ravel()
                            data[:] = flat.reshape(data.shape, order='F')
                    else:
                        idx_int = int(_to_py(idx)) - 1
                        data.ravel(order='F')[idx_int] = assign_val
                elif len(args) >= 2:
                    slices = []
                    for dim, arg in enumerate(args):
                        if arg is None:
                            slices.append(slice(None))
                        elif isinstance(arg, ForgeArray):
                            raw = _unwrap(arg)
                            if raw.dtype == np.bool_:
                                slices.append(raw.ravel())
                            else:
                                slices.append(raw.astype(int).ravel() - 1)
                        else:
                            slices.append(int(_to_py(arg)) - 1)
                    idx_arrays = []
                    for i, s in enumerate(slices):
                        if isinstance(s, slice):
                            idx_arrays.append(np.arange(data.shape[i] if i < len(data.shape) else 1))
                        elif isinstance(s, np.ndarray):
                            idx_arrays.append(s)
                        else:
                            idx_arrays.append(np.array([s]))
                    ix = np.ix_(*idx_arrays)
                    data[ix] = assign_val
            return value

        if isinstance(target, FieldAccess):
            # Handle struct array: results(c).name = value
            if isinstance(target.target, Index) and isinstance(target.target.target, Identifier):
                struct_name = target.target.target.name
                idx_args = [self._eval_expr(a, ws) for a in target.target.args]
                idx = int(_to_py(idx_args[0])) if len(idx_args) == 1 else None

                if not ws.has(struct_name):
                    # Create new struct array
                    # ForgeCell imported at top
                    struct_arr = ForgeCell([None] * max(idx, 1))
                    for i in range(len(struct_arr._data)):
                        struct_arr._data[i] = ForgeStruct()
                    ws.set(struct_name, struct_arr)

                container = ws.get(struct_name)

                if isinstance(container, ForgeCell) and idx is not None:
                    # Grow if needed
                    while len(container._data) < idx:
                        container._data.append(ForgeStruct())
                    entry = container._data[idx - 1]
                    if not isinstance(entry, ForgeStruct):
                        entry = ForgeStruct()
                        container._data[idx - 1] = entry
                    entry._fields[target.field] = value
                elif isinstance(container, ForgeStruct):
                    # Single struct being indexed — convert to struct array
                    # ForgeCell imported at top
                    struct_arr = ForgeCell([container])
                    while len(struct_arr._data) < idx:
                        struct_arr._data.append(ForgeStruct())
                    struct_arr._data[idx - 1]._fields[target.field] = value
                    ws.set(struct_name, struct_arr)
                return value

            if isinstance(target.target, Identifier) and not ws.has(target.target.name):
                new_struct = ForgeStruct()
                new_struct._fields[target.field] = value
                ws.set(target.target.name, new_struct)
                return value
            obj = self._eval_expr(target.target, ws)
            if isinstance(obj, ForgeStruct):
                obj._fields[target.field] = value
            else:
                from forge.engine.classdef import ForgeObject
                if isinstance(obj, ForgeObject):
                    obj.set(target.field, value)
            return value

        if isinstance(target, CellIndex):
            obj = self._eval_expr(target.target, ws)
            if isinstance(obj, ForgeCell):
                args = [_to_py(self._eval_expr(a, ws)) for a in target.args]
                obj.content_set(value, *args)
            return value

        raise RuntimeError(f"Cannot assign to {type(target).__name__}")

    def _exec_if(self, node: IfStatement, ws: Workspace) -> Any:
        if _is_truthy(self._eval_expr(node.condition, ws)):
            return self._exec_stmts(node.body, ws)
        for cond, body in node.elseifs:
            if _is_truthy(self._eval_expr(cond, ws)):
                return self._exec_stmts(body, ws)
        if node.else_body:
            return self._exec_stmts(node.else_body, ws)
        return None

    def _exec_for(self, node: ForStatement, ws: Workspace) -> Any:
        iter_val = self._eval_expr(node.iter_expr, ws)
        data = _unwrap(iter_val)
        if data.ndim >= 2:
            # Iterate over columns
            for col_idx in range(data.shape[1]):
                ws.set(node.var, ForgeArray(data[:, col_idx:col_idx+1]))
                try:
                    self._exec_stmts(node.body, ws)
                except BreakSignal:
                    break
                except ContinueSignal:
                    continue
        else:
            for val in data.ravel():
                ws.set(node.var, ForgeArray(val))
                try:
                    self._exec_stmts(node.body, ws)
                except BreakSignal:
                    break
                except ContinueSignal:
                    continue
        return None

    def _exec_while(self, node: WhileStatement, ws: Workspace) -> Any:
        while _is_truthy(self._eval_expr(node.condition, ws)):
            try:
                self._exec_stmts(node.body, ws)
            except BreakSignal:
                break
            except ContinueSignal:
                continue
        return None

    def _exec_do_until(self, node: DoUntilStatement, ws: Workspace) -> Any:
        while True:
            try:
                self._exec_stmts(node.body, ws)
            except BreakSignal:
                break
            except ContinueSignal:
                pass
            if _is_truthy(self._eval_expr(node.condition, ws)):
                break
        return None

    def _exec_switch(self, node: SwitchStatement, ws: Workspace) -> Any:
        val = self._eval_expr(node.expr, ws)
        for case_expr, body in node.cases:
            case_val = self._eval_expr(case_expr, ws)
            if _values_equal(val, case_val):
                return self._exec_stmts(body, ws)
        if node.otherwise_body:
            return self._exec_stmts(node.otherwise_body, ws)
        return None

    def _exec_try(self, node: TryCatchStatement, ws: Workspace) -> Any:
        try:
            return self._exec_stmts(node.try_body, ws)
        except (ForgeError, Exception) as e:
            if node.catch_body is not None:
                if node.catch_var:
                    err_struct = ForgeStruct()
                    err_struct.identifier = getattr(e, "identifier", "")
                    err_struct.message = str(e)
                    ws.set(node.catch_var, err_struct)
                return self._exec_stmts(node.catch_body, ws)
            return None

    def _exec_classdef(self, node: ClassDef, ws: Workspace):
        """Register a classdef as a constructable class."""
        from forge.engine.classdef import (
            ForgeClass, Property, Method, register_class, ForgeObject
        )
        # Build properties
        props = {}
        for name, default_expr in node.properties.items():
            default_val = self._eval_expr(default_expr, ws) if default_expr is not None else None
            props[name] = Property(name=name, default_value=default_val)
        # Build methods
        methods = {}
        for func_def in node.methods:
            def make_method(fd):
                def method_impl(obj, *args):
                    # Create local workspace with obj as first parameter
                    local_ws = Workspace()
                    if fd.params:
                        local_ws.set(fd.params[0], obj)
                    for i, param in enumerate(fd.params[1:], 1):
                        if i - 1 < len(args):
                            local_ws.set(param, args[i - 1])
                    local_ws.set("nargin", ForgeArray(np.array(float(len(args) + 1))))
                    try:
                        self._exec_stmts(fd.body, local_ws)
                    except ReturnSignal:
                        pass
                    if fd.returns:
                        return local_ws.get(fd.returns[0])
                    return obj
                return method_impl
            methods[func_def.name] = Method(
                name=func_def.name,
                function_def=make_method(func_def),
            )
        # Create and register class
        forge_cls = ForgeClass(
            name=node.name,
            superclasses=node.superclasses,
            properties=props,
            methods=methods,
            is_handle=node.is_handle,
        )
        register_class(forge_cls)
        # Register constructor as a callable in functions dict
        self.functions[node.name] = lambda *args: forge_cls.construct(*args)
        return None

    def _exec_unwind_protect(self, node: UnwindProtect, ws: Workspace):
        """Execute unwind_protect: always run cleanup."""
        try:
            result = self._exec_stmts(node.try_body, ws)
        except Exception:
            result = None
        finally:
            self._exec_stmts(node.cleanup_body, ws)
        return result

    def _call_function(self, funcdef: FunctionDef, args: list, caller_ws: Workspace, nargout=None) -> Any:
        """Call a user-defined function with varargin/varargout support."""
        local_ws = Workspace()
        # Copy constants from base workspace into local workspace
        _CONSTANTS = {"pi", "e", "eps", "Inf", "inf", "NaN", "nan",
                      "realmin", "realmax", "i", "j", "true", "false"}
        for const in _CONSTANTS:
            if self.workspace.has(const):
                local_ws.set(const, self.workspace.get(const))
        # varargin/varargout detection
        has_varargin = (len(funcdef.params) > 0 and funcdef.params[-1] == "varargin")
        has_varargout = (len(funcdef.returns) > 0 and funcdef.returns[-1] == "varargout")
        fixed_params = funcdef.params[:-1] if has_varargin else funcdef.params
        # Set nargin/nargout
        local_ws.set("nargin", ForgeArray(np.array(float(len(args)))))
        _nout = nargout if nargout is not None else len(funcdef.returns)
        local_ws.set("nargout", ForgeArray(np.array(float(_nout))))
        # Bind fixed parameters
        for i, param in enumerate(fixed_params):
            if i < len(args):
                local_ws.set(param, args[i])
        # Bind varargin: collect extra args into a cell array
        if has_varargin:
            extra = args[len(fixed_params):]
            local_ws.set("varargin", ForgeCell(extra))
        # Initialize varargout as empty cell
        if has_varargout:
            local_ws.set("varargout", ForgeCell([]))
        # Initialize return variables
        for ret in funcdef.returns:
            if not local_ws.has(ret):
                local_ws.set(ret, ForgeArray(0.0))
        # Execute body
        try:
            self._exec_stmts(funcdef.body, local_ws)
        except ReturnSignal:
            pass
        # Collect return values (expand varargout)
        returns = funcdef.returns
        if len(returns) == 0:
            return None
        results = []
        for ret in returns:
            if ret == "varargout" and has_varargout:
                va = local_ws.get("varargout")
                if isinstance(va, ForgeCell):
                    results.extend(va._data if hasattr(va, "_data") else [])
                else:
                    results.append(va)
            else:
                results.append(local_ws.get(ret))
        if len(results) == 1:
            return results[0]
        return tuple(results)


    def _eval_multi_output(self, expr, ws, nargout):
        if not isinstance(expr, Index):
            return None
        name = expr.target.name if isinstance(expr.target, Identifier) else None
        if name is None:
            return None
        args = [self._eval_expr(a, ws) for a in expr.args]
        if name == 'max':
            x = _unwrap(args[0])
            idx_flat = int(np.argmax(x.ravel()))
            return (ForgeArray(np.max(x)), ForgeArray(np.array(float(idx_flat + 1))))
        if name == 'min':
            x = _unwrap(args[0])
            idx_flat = int(np.argmin(x.ravel()))
            return (ForgeArray(np.min(x)), ForgeArray(np.array(float(idx_flat + 1))))
        if name == 'find':
            x = _unwrap(args[0])
            if nargout >= 2:
                if x.ndim < 2:
                    x = x.reshape(1, -1)
                rows, cols = np.nonzero(x)
                return (ForgeArray(rows + 1), ForgeArray(cols + 1))
            return None
        if name == 'size':
            x = args[0]
            if isinstance(x, ForgeArray):
                shape = x.shape
                if nargout >= 2:
                    nrows = shape[0] if len(shape) > 0 else 1
                    ncols = shape[1] if len(shape) > 1 else 1
                    return (ForgeArray(np.array(float(nrows))), ForgeArray(np.array(float(ncols))))
            return None
        if name == 'ind2sub':
            if len(args) >= 2:
                sz = _unwrap(args[0]).ravel().astype(int)
                ind = _unwrap(args[1]).ravel().astype(int) - 1
                if len(sz) >= 2:
                    rows = (ind % sz[0]) + 1
                    cols = (ind // sz[0]) + 1
                    return (ForgeArray(rows.astype(float)), ForgeArray(cols.astype(float)))
            return None
        if name == 'sort':
            x = _unwrap(args[0])
            if x.ndim == 2 and (x.shape[0] == 1 or x.shape[1] == 1):
                flat = x.ravel()
                idx = np.argsort(flat)
                sorted_vals = flat[idx]
                if x.shape[0] == 1:
                    return (ForgeArray(sorted_vals.reshape(1, -1)), ForgeArray((idx + 1).astype(float).reshape(1, -1)))
                else:
                    return (ForgeArray(sorted_vals.reshape(-1, 1)), ForgeArray((idx + 1).astype(float).reshape(-1, 1)))
            else:
                idx = np.argsort(x, axis=0)
                sorted_vals = np.take_along_axis(x, idx, axis=0)
                return (ForgeArray(sorted_vals), ForgeArray((idx + 1).astype(float)))
        if name == 'eig':
            x = _unwrap(args[0])
            if len(args) >= 2:
                b = _unwrap(args[1])
                from scipy.linalg import eig as scipy_eig
                vals, vecs = scipy_eig(x, b)
            else:
                vals, vecs = np.linalg.eig(x)
            return (ForgeArray(vecs), ForgeArray(np.diag(vals)))
        return None


def _process_c_escapes(s):
    """Process C-style escape sequences in format strings (MATLAB fprintf behavior)."""
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            c = s[i + 1]
            if c == 'n': result.append('\n'); i += 2
            elif c == 't': result.append('\t'); i += 2
            elif c == 'r': result.append('\r'); i += 2
            elif c == '\\': result.append('\\'); i += 2
            elif c == '0': result.append('\0'); i += 2
            elif c == 'a': result.append('\a'); i += 2
            elif c == 'b': result.append('\b'); i += 2
            elif c == 'f': result.append('\f'); i += 2
            elif c == 'v': result.append('\v'); i += 2
            elif c == "'": result.append("'"); i += 2
            elif c == '"': result.append('"'); i += 2
            else: result.append(s[i]); i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)

def _to_py(val):
    """Convert ForgeArray scalar to Python number."""
    if isinstance(val, ForgeArray):
        if val.isscalar():
            return val.data.flat[0].item()
        return val.data
    if isinstance(val, ForgeChar):
        return val.to_str()
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    return val


def _to_int(val):
    """Convert to Python int (for indexing)."""
    if isinstance(val, ForgeArray):
        return int(val.data.flat[0])
    if isinstance(val, (float, np.floating)):
        return int(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    return int(val)


def _is_truthy(val) -> bool:
    """Octave truthiness: all elements nonzero."""
    if isinstance(val, ForgeArray):
        return bool(np.all(val.data != 0))
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    return bool(val)


def _is_matrix_op(a, b) -> bool:
    """Check if matrix (not element-wise) operation applies.
    In Octave, * is always matrix multiply for 2D arrays.
    Element-wise is .* operator. So * should use @ whenever
    both operands are 2D (even if one is a vector)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2:
        return False
    if a.size == 1 or b.size == 1:
        return False
    return True


def _values_equal(a, b) -> bool:
    """Compare values for switch/case."""
    a_raw = _unwrap(a) if isinstance(a, ForgeArray) else a
    b_raw = _unwrap(b) if isinstance(b, ForgeArray) else b
    if isinstance(a, ForgeChar) and isinstance(b, ForgeChar):
        return a.to_str() == b.to_str()
    try:
        return np.array_equal(a_raw, b_raw)
    except (TypeError, ValueError):
        return a_raw == b_raw
