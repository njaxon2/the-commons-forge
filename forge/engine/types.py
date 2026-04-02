# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Forge data types: ForgeArray with 1-based indexing, type conversion, and MATLAB/Octave semantics."""
import numpy as np
from enum import Enum


# Octave type name → numpy dtype mapping
DTYPE_MAP = {
    "double": np.float64,
    "single": np.float32,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "logical": np.bool_,
    "char": np.uint8,  # char arrays stored as uint8 internally
}

DTYPE_TO_NAME = {v: k for k, v in DTYPE_MAP.items()}
# Handle numpy dtype aliases
DTYPE_TO_NAME[np.float64] = "double"
DTYPE_TO_NAME[np.float32] = "single"
DTYPE_TO_NAME[np.bool_] = "logical"


class ForgeArray:
    """N-dimensional array with 1-based indexing and Octave type semantics.

    Wraps a numpy ndarray but provides:
    - 1-based indexing: A(1) is the first element
    - end keyword support
    - Colon indexing: A(2:5), A(:), A(end-2:end)
    - MATLAB/Octave size semantics: scalar is 1x1, vector is 1xN or Nx1
    - Type-preserving arithmetic
    """

    __slots__ = ("_data", "_is_char")

    def __init__(self, data, dtype=None):
        if isinstance(data, ForgeArray):
            self._data = data._data.copy() if dtype is None else data._data.astype(DTYPE_MAP.get(dtype, dtype))
            if dtype is None:
                self._is_char = data._is_char  # Preserve char flag on copy; will be overwritten below if dtype specified
        elif isinstance(data, np.ndarray):
            self._data = data if dtype is None else data.astype(DTYPE_MAP.get(dtype, dtype))
        elif isinstance(data, (list, tuple)):
            if dtype and dtype in DTYPE_MAP and np.issubdtype(DTYPE_MAP[dtype], np.integer):
                # For integer types, first create array then cast (avoids OverflowError on out-of-range values)
                self._data = np.array(data).astype(DTYPE_MAP[dtype])
            else:
                self._data = np.array(data, dtype=DTYPE_MAP.get(dtype, dtype) if dtype else None)
        elif isinstance(data, (int, float, complex, np.integer, np.floating, np.complexfloating)):
            self._data = np.array(data, dtype=DTYPE_MAP.get(dtype, dtype) if dtype else None)
            if self._data.ndim == 0:
                self._data = self._data.reshape(1, 1)  # Scalars are 1x1 in Octave
        elif isinstance(data, bool) or isinstance(data, np.bool_):
            self._data = np.array(data, dtype=np.bool_)
            if self._data.ndim == 0:
                self._data = self._data.reshape(1, 1)
        else:
            import scipy.sparse as _sp
            if _sp.issparse(data):
                self._data = data  # Preserve sparse matrices as-is
            else:
                self._data = np.asarray(data)
        self._is_char = (dtype == "char")
        # Ensure at least 2D for MATLAB semantics (scalars and vectors)
        # Skip reshape for sparse matrices (they are always 2D)
        import scipy.sparse as _sp2
        if _sp2.issparse(self._data):
            pass  # scipy sparse is always 2D
        elif self._data.ndim == 0:
            self._data = self._data.reshape(1, 1)
        elif self._data.ndim == 1:
            self._data = self._data.reshape(1, -1)  # Row vector by default

    @property
    def data(self):
        """Access underlying numpy array."""
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    @property
    def T(self):
        return ForgeArray(self._data.T)

    def type_name(self):
        """Return Octave-style type name."""
        if self._is_char:
            return "char"
        dt = self._data.dtype.type
        if dt == np.uint8:
            return "uint8"
        return DTYPE_TO_NAME.get(dt, str(self._data.dtype))

    # --- Size functions (Octave semantics) ---
    def numel(self):
        return self._data.size

    def length(self):
        """Length = max dimension size (Octave semantics)."""
        return max(self._data.shape) if self._data.size > 0 else 0

    def rows(self):
        return self._data.shape[0]

    def columns(self):
        """columns(A) — return number of columns in matrix A."""
        return self._data.shape[1] if self._data.ndim >= 2 else 1

    def isempty(self):
        return self._data.size == 0

    def isscalar(self):
        """isscalar(x) — true if x is a 1x1 scalar."""
        return self._data.size == 1

    def isvector(self):
        """isvector(x) — true if x is a row or column vector."""
        if self._data.size == 0:
            return False
        non_singleton = sum(1 for s in self._data.shape if s > 1)
        return non_singleton <= 1

    def ismatrix(self):
        """ismatrix(x) — true if x is a 2-D matrix."""
        return self._data.ndim == 2

    def issquare(self):
        """issquare(x) — true if x is a square matrix."""
        return self._data.ndim == 2 and self._data.shape[0] == self._data.shape[1]

    # --- 1-based indexing ---
    def _convert_index(self, idx):
        """Convert 1-based index to 0-based."""
        if isinstance(idx, (int, np.integer)):
            if idx < 1:
                raise IndexError(f"Index {idx} out of range (1-based indexing, minimum is 1)")
            return int(idx) - 1
        elif isinstance(idx, slice):
            start = (idx.start - 1) if idx.start is not None else None
            # Octave slices are inclusive on both ends; Python slices are exclusive on stop
            stop = idx.stop if idx.stop is not None else None
            step = idx.step
            return slice(start, stop, step)  # stop is already 1-based, which maps to exclusive 0-based
        elif isinstance(idx, ForgeArray):
            # Logical or numeric indexing
            if idx.dtype == np.bool_:
                return idx._data
            else:
                return (idx._data - 1).astype(int)
        elif isinstance(idx, np.ndarray):
            if idx.dtype == np.bool_:
                return idx
            return (idx - 1).astype(int)
        return idx

    def __getitem__(self, key):
        if isinstance(key, tuple):
            converted = tuple(self._convert_index(k) for k in key)
            result = self._data[converted]
        elif isinstance(key, (int, np.integer)):
            # Single integer on N-D array: linear (column-major) indexing
            idx = self._convert_index(key)
            result = self._data.ravel(order='F')[idx]
        elif isinstance(key, slice):
            converted = self._convert_index(key)
            result = self._data.ravel(order='F')[converted]
        else:
            converted = self._convert_index(key)
            result = self._data[converted]
        if isinstance(result, np.ndarray):
            return ForgeArray(result)
        # Return native Python scalar
        if isinstance(result, (np.integer, np.floating, np.complexfloating, np.bool_)):
            return result.item()
        return result

    def __setitem__(self, key, value):
        if isinstance(value, ForgeArray):
            value = value._data
        if isinstance(key, tuple):
            converted = tuple(self._convert_index(k) for k in key)
            self._data[converted] = value
        elif isinstance(key, (int, np.integer)):
            idx = self._convert_index(key)
            flat = self._data.ravel(order='F')
            flat[idx] = value
            self._data = flat.reshape(self._data.shape, order='F')
        else:
            converted = self._convert_index(key)
            self._data[converted] = value

    # --- Arithmetic (element-wise, returning ForgeArray) ---
    def __add__(self, other):
        return ForgeArray(self._data + _unwrap(other))

    def __radd__(self, other):
        return ForgeArray(_unwrap(other) + self._data)

    def __sub__(self, other):
        return ForgeArray(self._data - _unwrap(other))

    def __rsub__(self, other):
        return ForgeArray(_unwrap(other) - self._data)

    def __mul__(self, other):
        return ForgeArray(self._data * _unwrap(other))

    def __rmul__(self, other):
        return ForgeArray(_unwrap(other) * self._data)

    def __truediv__(self, other):
        return ForgeArray(self._data / _unwrap(other))

    def __rtruediv__(self, other):
        return ForgeArray(_unwrap(other) / self._data)

    def __pow__(self, other):
        return ForgeArray(self._data ** _unwrap(other))

    def __rpow__(self, other):
        return ForgeArray(_unwrap(other) ** self._data)

    def __neg__(self):
        return ForgeArray(-self._data)

    def __pos__(self):
        return ForgeArray(+self._data)

    def __abs__(self):
        return ForgeArray(np.abs(self._data))

    def __mod__(self, other):
        return ForgeArray(self._data % _unwrap(other))

    # --- Comparison (return logical ForgeArray) ---
    def __eq__(self, other):
        return ForgeArray(self._data == _unwrap(other))

    def __ne__(self, other):
        return ForgeArray(self._data != _unwrap(other))

    def __lt__(self, other):
        return ForgeArray(self._data < _unwrap(other))

    def __le__(self, other):
        return ForgeArray(self._data <= _unwrap(other))

    def __gt__(self, other):
        return ForgeArray(self._data > _unwrap(other))

    def __ge__(self, other):
        return ForgeArray(self._data >= _unwrap(other))

    # --- Matrix operations ---
    def __matmul__(self, other):
        return ForgeArray(self._data @ _unwrap(other))

    # --- Representation ---
    def __repr__(self):
        tn = self.type_name()
        if self.isscalar():
            return f"ForgeArray({self._data.flat[0]}, {tn})"
        return f"ForgeArray({self.shape}, {tn})"

    def __str__(self):
        return str(self._data)

    def __len__(self):
        return self._data.shape[0]

    def __float__(self):
        if self.isscalar():
            return float(self._data.flat[0])
        raise TypeError("Cannot convert non-scalar ForgeArray to float")

    def __int__(self):
        if self.isscalar():
            return int(self._data.flat[0])
        raise TypeError("Cannot convert non-scalar ForgeArray to int")

    def __bool__(self):
        if self.isscalar():
            return bool(self._data.flat[0])
        raise ValueError("Truth value of non-scalar ForgeArray is ambiguous")

    # --- Type conversion ---
    def astype(self, dtype_name):
        """Convert to specified type (Octave name or numpy dtype)."""
        dt = DTYPE_MAP.get(dtype_name, dtype_name)
        return ForgeArray(self._data.astype(dt))

    def copy(self):
        return ForgeArray(self._data.copy())


def _unwrap(x):
    """Extract numpy array from ForgeArray, or pass through."""
    if isinstance(x, ForgeArray):
        return x._data
    return x


# --- Type conversion functions (Octave built-ins) ---

def forge_double(x):
    """Convert to double (float64)."""
    return ForgeArray(x, dtype="double")

def forge_single(x):
    """Convert to single (float32)."""
    return ForgeArray(x, dtype="single")

def _octave_int_cast(x, dtype_name):
    """Octave-compatible integer cast: round then saturate."""
    target_dtype = DTYPE_MAP[dtype_name]
    info = np.iinfo(target_dtype)
    arr = _unwrap(x) if isinstance(x, ForgeArray) else np.asarray(x, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    # Round to nearest (Octave rounds, not truncates)
    arr = np.round(arr)
    # Saturate to type range (Octave saturates, not wraps)
    arr = np.clip(arr, info.min, info.max)
    return ForgeArray(arr.astype(target_dtype))

def forge_int8(x):
    """int8(x) — convert to signed 8-bit integer."""
    return _octave_int_cast(x, "int8")

def forge_int16(x):
    """int16(x) — convert to signed 16-bit integer."""
    return _octave_int_cast(x, "int16")

def forge_int32(x):
    """int32(x) — convert to signed 32-bit integer."""
    return _octave_int_cast(x, "int32")

def forge_int64(x):
    """int64(x) — convert to signed 64-bit integer."""
    return _octave_int_cast(x, "int64")

def forge_uint8(x):
    """uint8(x) — convert to unsigned 8-bit integer."""
    return _octave_int_cast(x, "uint8")

def forge_uint16(x):
    """uint16(x) — convert to unsigned 16-bit integer."""
    return _octave_int_cast(x, "uint16")

def forge_uint32(x):
    """uint32(x) — convert to unsigned 32-bit integer."""
    return _octave_int_cast(x, "uint32")

def forge_uint64(x):
    """uint64(x) — convert to unsigned 64-bit integer."""
    return _octave_int_cast(x, "uint64")

def forge_logical(x):
    """Convert to logical (boolean)."""
    return ForgeArray(x, dtype="logical")

def forge_complex(re, im=None):
    """Create complex array."""
    if im is None:
        return ForgeArray(np.asarray(_unwrap(re), dtype=np.complex128))
    return ForgeArray(np.asarray(_unwrap(re), dtype=np.float64) + 1j * np.asarray(_unwrap(im), dtype=np.float64))


# --- Type checking functions ---

def forge_isa(x, classname):
    """Test if x is of given class."""
    if not isinstance(x, ForgeArray):
        return False
    return x.type_name() == classname

def forge_isnumeric(x):
    if not isinstance(x, ForgeArray):
        return False
    return np.issubdtype(x.dtype, np.number)

def forge_islogical(x):
    if not isinstance(x, ForgeArray):
        return False
    return x.dtype == np.bool_

def forge_isfloat(x):
    """isfloat(x) — true if x is a floating-point type."""
    if not isinstance(x, ForgeArray):
        return False
    return np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating)

def forge_isinteger(x):
    if not isinstance(x, ForgeArray):
        return False
    return np.issubdtype(x.dtype, np.integer)

def forge_ischar(x):
    if not isinstance(x, ForgeArray):
        return isinstance(x, str)
    return x.type_name() == "char"

def forge_isnan(x):
    return ForgeArray(np.float64(np.isnan(_unwrap(x))))

def forge_isinf(x):
    return ForgeArray(np.float64(np.isinf(_unwrap(x))))

def forge_isfinite(x):
    return ForgeArray(np.float64(np.isfinite(_unwrap(x))))


# --- Special values ---

FORGE_PI = np.float64(np.pi)
FORGE_E = np.float64(np.e)
FORGE_INF = np.float64(np.inf)
FORGE_NAN = np.float64(np.nan)
FORGE_EPS = np.float64(np.finfo(np.float64).eps)
FORGE_I = np.complex128(1j)
FORGE_J = np.complex128(1j)
FORGE_TRUE = ForgeArray(np.array(True))
FORGE_FALSE = ForgeArray(np.array(False))


# --- Matrix construction ---

def forge_eye(m, n=None, dtype="double"):
    """Identity matrix."""
    if n is None:
        n = m
    return ForgeArray(np.eye(int(m), int(n), dtype=DTYPE_MAP.get(dtype, dtype)))

def forge_ones(*args, dtype="double"):
    """Matrix of ones."""
    shape = _parse_size_args(args)
    return ForgeArray(np.ones(shape, dtype=DTYPE_MAP.get(dtype, dtype)))

def forge_zeros(*args, dtype="double"):
    """Matrix of zeros."""
    shape = _parse_size_args(args)
    return ForgeArray(np.zeros(shape, dtype=DTYPE_MAP.get(dtype, dtype)))

def forge_rand(*args):
    """Uniform random matrix [0,1)."""
    shape = _parse_size_args(args)
    return ForgeArray(np.random.rand(*shape))

def forge_randn(*args):
    """Standard normal random matrix."""
    shape = _parse_size_args(args)
    return ForgeArray(np.random.randn(*shape))

def forge_randi(imax, *args):
    """Random integers from 1 to imax."""
    shape = _parse_size_args(args) if args else (1, 1)
    return ForgeArray(np.random.randint(1, int(imax) + 1, size=shape))

def forge_true(*args):
    """Logical true array."""
    shape = _parse_size_args(args) if args else (1, 1)
    return ForgeArray(np.ones(shape, dtype=np.bool_))

def forge_false(*args):
    """Logical false array."""
    shape = _parse_size_args(args) if args else (1, 1)
    return ForgeArray(np.zeros(shape, dtype=np.bool_))

def forge_diag(v, k=0):
    """Create diagonal matrix or extract diagonal."""
    v_data = _unwrap(v)
    # Unwrap k from ForgeArray if needed
    if isinstance(k, ForgeArray):
        k = int(k.data.flat[0])
    else:
        k = int(k)
    if v_data.ndim <= 1 or (v_data.ndim == 2 and min(v_data.shape) == 1):
        # Vector input: create diagonal matrix
        return ForgeArray(np.diag(v_data.ravel(), k))
    else:
        # Matrix input: extract diagonal
        return ForgeArray(np.diag(v_data, k))

def forge_linspace(a, b, n=100):
    """Linearly-spaced vector."""
    return ForgeArray(np.linspace(float(a), float(b), int(n)))

def forge_colon(start, stop_or_step=None, stop=None):
    """Colon operator: start:stop or start:step:stop.
    Matches Octave semantics: generates values from start to stop (inclusive)
    with given step. Uses careful rounding to avoid off-by-one errors."""
    if stop is None:
        # start:stop with step=1
        a, b = float(start), float(stop_or_step)
        if b < a:
            return ForgeArray(np.array([]).reshape(1, 0))
        n = int(np.floor(b - a + 1.5))  # number of elements for integer step
        return ForgeArray(np.arange(a, a + n, 1.0))
    else:
        # start:step:stop
        a = float(start)
        step = float(stop_or_step)
        b = float(stop)
        if step == 0:
            return ForgeArray(np.array([]).reshape(1, 0))
        if (step > 0 and b < a) or (step < 0 and b > a):
            return ForgeArray(np.array([]).reshape(1, 0))
        # Octave counts: n = max(0, floor((b - a) / step) + 1)
        n = max(0, int(np.floor((b - a) / step + 1 + np.finfo(float).eps * 128)))
        # Verify last element does not overshoot
        last = a + (n - 1) * step
        if step > 0 and last > b + abs(step) * 1e-10:
            n -= 1
        elif step < 0 and last < b - abs(step) * 1e-10:
            n -= 1
        if n <= 0:
            return ForgeArray(np.array([]).reshape(1, 0))
        result = a + np.arange(n) * step
        return ForgeArray(result)

def forge_repmat(a, m, n=None):
    """Tile array."""
    if n is None:
        n = m
    return ForgeArray(np.tile(_unwrap(a), (int(m), int(n))))


def _parse_size_args(args):
    """Parse Octave size arguments: (n), (m,n), (m,n,p,...), ([m,n])."""
    if len(args) == 0:
        return (1, 1)
    if len(args) == 1:
        a = args[0]
        if isinstance(a, (list, tuple)):
            return tuple(int(x) for x in a)
        if isinstance(a, ForgeArray):
            return tuple(int(x) for x in a._data.ravel())
        if isinstance(a, np.ndarray):
            return tuple(int(x) for x in a.ravel())
        n = int(a)
        return (n, n)  # zeros(3) = 3x3 in Octave
    return tuple(int(a) for a in args)
