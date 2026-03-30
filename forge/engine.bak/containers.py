# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Forge container types: char arrays, cell arrays, structs, and containers.Map."""
import numpy as np
from collections import OrderedDict
from forge.engine.types import ForgeArray, _unwrap


# ============================================================
# Char Arrays
# ============================================================

class ForgeChar(ForgeArray):
    """Character array — stores text as uint8 codes with string semantics."""

    def __init__(self, data):
        if isinstance(data, str):
            codes = np.array([ord(c) for c in data], dtype=np.uint8)
            super().__init__(codes, dtype="char")
        elif isinstance(data, ForgeChar):
            super().__init__(data._data.copy(), dtype="char")
        elif isinstance(data, (list, tuple)):
            if all(isinstance(x, str) for x in data):
                # Cell array of strings → 2D char array (padded)
                max_len = max(len(s) for s in data) if data else 0
                rows = []
                for s in data:
                    row = [ord(c) for c in s] + [32] * (max_len - len(s))  # pad with spaces
                    rows.append(row)
                super().__init__(np.array(rows, dtype=np.uint8), dtype="char")
            else:
                super().__init__(np.array(data, dtype=np.uint8), dtype="char")
        else:
            super().__init__(data, dtype="char")
        self._is_char = True

    def to_str(self):
        """Convert to Python string."""
        if self._data.ndim <= 1 or (self._data.ndim == 2 and self._data.shape[0] == 1):
            return "".join(chr(c) for c in self._data.ravel())
        # Multi-row: return list of strings
        return ["".join(chr(c) for c in row) for row in self._data]

    def __repr__(self):
        s = self.to_str()
        if isinstance(s, list):
            return f"ForgeChar({s})"
        return f"ForgeChar('{s}')"

    def __str__(self):
        s = self.to_str()
        if isinstance(s, list):
            return "\n".join(s)
        return s

    def __eq__(self, other):
        if isinstance(other, str):
            other = ForgeChar(other)
        if isinstance(other, ForgeChar):
            return ForgeArray(self._data == other._data)
        return ForgeArray(self._data == _unwrap(other))

    def __add__(self, other):
        """Concatenate char arrays (horzcat for strings)."""
        if isinstance(other, str):
            other = ForgeChar(other)
        if isinstance(other, ForgeChar):
            return ForgeChar(np.concatenate([self._data.ravel(), other._data.ravel()]))
        return super().__add__(other)


def forge_char(*args):
    """Create char array from string, codes, or cell of strings."""
    if len(args) == 1:
        return ForgeChar(args[0])
    # Multiple args: concatenate
    result = ForgeChar(args[0])
    for a in args[1:]:
        result = result + ForgeChar(a) if isinstance(a, str) else result + a
    return result


def forge_strcmp(s1, s2):
    """Compare strings (case-sensitive)."""
    a = s1.to_str() if isinstance(s1, ForgeChar) else s1
    b = s2.to_str() if isinstance(s2, ForgeChar) else s2
    return a == b


def forge_strcmpi(s1, s2):
    """Compare strings (case-insensitive)."""
    a = (s1.to_str() if isinstance(s1, ForgeChar) else s1).lower()
    b = (s2.to_str() if isinstance(s2, ForgeChar) else s2).lower()
    return a == b


def forge_strncmp(s1, s2, n):
    """Compare first n characters (case-sensitive)."""
    a = s1.to_str() if isinstance(s1, ForgeChar) else s1
    b = s2.to_str() if isinstance(s2, ForgeChar) else s2
    return a[:n] == b[:n]


def forge_strncmpi(s1, s2, n):
    """Compare first n characters (case-insensitive)."""
    a = (s1.to_str() if isinstance(s1, ForgeChar) else s1).lower()
    b = (s2.to_str() if isinstance(s2, ForgeChar) else s2).lower()
    return a[:n] == b[:n]


# ============================================================
# Cell Arrays
# ============================================================

class ForgeCell:
    """Cell array — heterogeneous container with () and {} indexing."""

    __slots__ = ("_data", "_shape")

    def __init__(self, *args):
        if len(args) == 0:
            self._data = []
            self._shape = (0, 0)
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            items = list(args[0])
            self._data = items
            self._shape = (1, len(items))
        elif len(args) == 1 and isinstance(args[0], ForgeCell):
            self._data = list(args[0]._data)
            self._shape = args[0]._shape
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
            # cell(m, n) — create m×n empty cell
            m, n = args
            self._data = [ForgeArray(np.array([])) for _ in range(m * n)]
            self._shape = (m, n)
        else:
            self._data = list(args)
            self._shape = (1, len(args))

    @property
    def shape(self):
        return self._shape

    def numel(self):
        return len(self._data)

    def length(self):
        return max(self._shape) if self.numel() > 0 else 0

    def isempty(self):
        return self.numel() == 0

    def _linear_index(self, idx):
        """Convert 1-based index to 0-based linear index."""
        if isinstance(idx, int):
            if idx < 1:
                raise IndexError(f"Cell index {idx} out of range (1-based)")
            return idx - 1
        raise TypeError(f"Cell index must be int, got {type(idx)}")

    def content_get(self, *indices):
        """Get cell content: C{i} or C{i,j}."""
        if len(indices) == 1:
            return self._data[self._linear_index(indices[0])]
        elif len(indices) == 2:
            r, c = indices
            idx = (r - 1) * self._shape[1] + (c - 1)
            return self._data[idx]
        raise IndexError("Too many indices for cell")

    def content_set(self, value, *indices):
        """Set cell content: C{i} = v or C{i,j} = v."""
        if len(indices) == 1:
            idx = self._linear_index(indices[0])
            while idx >= len(self._data):
                self._data.append(ForgeArray(np.array([])))
                self._shape = (1, len(self._data))
            self._data[idx] = value
        elif len(indices) == 2:
            r, c = indices
            idx = (r - 1) * self._shape[1] + (c - 1)
            self._data[idx] = value

    def cell_get(self, *indices):
        """Get sub-cell: C(i) returns a 1-element cell."""
        content = self.content_get(*indices)
        return ForgeCell([content])

    def __repr__(self):
        return f"ForgeCell({self._shape}, {len(self._data)} elements)"

    def __len__(self):
        return self._shape[0]


def forge_cell(*args):
    """Create cell array. cell(m,n) or cell(n)."""
    if len(args) == 1 and isinstance(args[0], int):
        n = args[0]
        return ForgeCell(n, n)
    if len(args) == 2:
        return ForgeCell(int(args[0]), int(args[1]))
    return ForgeCell(*args)


def forge_iscell(x):
    return isinstance(x, ForgeCell)


def forge_cell2mat(c):
    """Convert cell of matrices to single matrix."""
    if not isinstance(c, ForgeCell):
        raise TypeError("Input must be a cell array")
    arrays = [_unwrap(x) if isinstance(x, ForgeArray) else np.asarray(x) for x in c._data]
    if c._shape[0] == 1:
        return ForgeArray(np.concatenate(arrays, axis=-1))
    # 2D cell: block matrix
    rows_list = []
    nc = c._shape[1]
    for r in range(c._shape[0]):
        row_arrays = arrays[r * nc:(r + 1) * nc]
        rows_list.append(np.concatenate(row_arrays, axis=-1))
    return ForgeArray(np.concatenate(rows_list, axis=0))


def forge_num2cell(a):
    """Convert array to cell (one element per cell)."""
    data = _unwrap(a)
    items = [ForgeArray(x) for x in data.ravel()]
    c = ForgeCell(items)
    c._shape = data.shape
    return c


def forge_cellfun(func, c, uniform_output=True):
    """Apply function to each cell element."""
    if not isinstance(c, ForgeCell):
        raise TypeError("Second argument must be a cell array")
    results = [func(item) for item in c._data]
    if uniform_output:
        return ForgeArray(np.array([float(r) if isinstance(r, (int, float)) else r for r in results]))
    return ForgeCell(results)


# ============================================================
# Structs
# ============================================================

class ForgeStruct:
    """Struct — ordered dict with field access via dot notation."""

    __slots__ = ("_fields",)

    def __init__(self, *args, **kwargs):
        self._fields = OrderedDict()
        if len(args) == 0:
            self._fields.update(kwargs)
        elif len(args) >= 2:
            # struct('field1', val1, 'field2', val2, ...)
            it = iter(args)
            for key in it:
                val = next(it)
                self._fields[str(key)] = val
        elif len(args) == 1 and isinstance(args[0], dict):
            self._fields.update(args[0])
        elif len(args) == 1 and isinstance(args[0], ForgeStruct):
            self._fields = OrderedDict(args[0]._fields)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(f"No field '{name}' in struct")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._fields[name] = value

    def __contains__(self, key):
        return key in self._fields

    def __repr__(self):
        fields = ", ".join(self._fields.keys())
        return f"ForgeStruct({fields})"

    def __eq__(self, other):
        if not isinstance(other, ForgeStruct):
            return False
        return self._fields == other._fields


def forge_struct(*args, **kwargs):
    """Create struct."""
    return ForgeStruct(*args, **kwargs)


def forge_isstruct(x):
    return isinstance(x, ForgeStruct)


def forge_fieldnames(s):
    """Return cell of field names."""
    if not isinstance(s, ForgeStruct):
        raise TypeError("Input must be a struct")
    return ForgeCell(list(s._fields.keys()))


def forge_rmfield(s, field):
    """Remove field from struct."""
    if not isinstance(s, ForgeStruct):
        raise TypeError("Input must be a struct")
    new_s = ForgeStruct(s)
    if isinstance(field, str):
        del new_s._fields[field]
    elif isinstance(field, ForgeChar):
        del new_s._fields[field.to_str()]
    return new_s


def forge_setfield(s, field, value):
    """Set field on struct (returns new struct)."""
    new_s = ForgeStruct(s)
    new_s._fields[str(field)] = value
    return new_s


def forge_getfield(s, field):
    """Get field from struct."""
    return s._fields[str(field)]


def forge_orderfields(s):
    """Return struct with alphabetically sorted fields."""
    new_s = ForgeStruct()
    for k in sorted(s._fields.keys()):
        new_s._fields[k] = s._fields[k]
    return new_s


def forge_structfun(func, s):
    """Apply function to each field value."""
    if not isinstance(s, ForgeStruct):
        raise TypeError("Input must be a struct")
    result = ForgeStruct()
    for k, v in s._fields.items():
        result._fields[k] = func(v)
    return result


# ============================================================
# Containers.Map
# ============================================================

class ForgeMap:
    """containers.Map — key-value dictionary with typed keys."""

    __slots__ = ("_data", "_key_type", "_value_type")

    def __init__(self, keys=None, values=None, key_type="char", value_type="any"):
        self._data = OrderedDict()
        self._key_type = key_type
        self._value_type = value_type
        if keys is not None and values is not None:
            if isinstance(keys, ForgeCell):
                keys = keys._data
            if isinstance(values, ForgeCell):
                values = values._data
            if isinstance(keys, (list, tuple)):
                for k, v in zip(keys, values):
                    self._data[k] = v

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._data)

    def keys(self):
        return ForgeCell(list(self._data.keys()))

    def values(self):
        return ForgeCell(list(self._data.values()))

    def isKey(self, key):
        return key in self._data

    def remove(self, key):
        if key in self._data:
            del self._data[key]

    def length(self):
        return len(self._data)

    def __repr__(self):
        return f"ForgeMap({len(self._data)} entries)"


# ============================================================
# Sparse Matrices (thin wrapper for Stage 1.9)
# ============================================================

try:
    import scipy.sparse as sp

    class ForgeSparse:
        """Sparse matrix wrapping scipy.sparse."""

        __slots__ = ("_data",)

        def __init__(self, data):
            if isinstance(data, ForgeSparse):
                self._data = data._data.copy()
            elif isinstance(data, sp.spmatrix):
                self._data = data.tocsc()
            elif isinstance(data, ForgeArray):
                self._data = sp.csc_matrix(data.data)
            elif isinstance(data, np.ndarray):
                self._data = sp.csc_matrix(data)
            else:
                self._data = sp.csc_matrix(data)

        @staticmethod
        def from_triplets(i, j, v, m=None, n=None):
            """Create sparse from (row, col, val) triplets (1-based indices)."""
            i0 = np.asarray(_unwrap(i)).ravel() - 1  # Convert to 0-based
            j0 = np.asarray(_unwrap(j)).ravel() - 1
            v0 = np.asarray(_unwrap(v)).ravel().astype(float)
            if m is None:
                m = int(i0.max()) + 1
            if n is None:
                n = int(j0.max()) + 1
            return ForgeSparse(sp.csc_matrix((v0, (i0, j0)), shape=(m, n)))

        @property
        def shape(self):
            return self._data.shape

        @property
        def dtype(self):
            return self._data.dtype

        def nnz(self):
            return self._data.nnz

        def full(self):
            """Convert to dense ForgeArray."""
            return ForgeArray(self._data.toarray())

        def __repr__(self):
            return f"ForgeSparse({self.shape}, nnz={self.nnz()})"

        def __add__(self, other):
            if isinstance(other, ForgeSparse):
                return ForgeSparse(self._data + other._data)
            return ForgeSparse(self._data + _unwrap(other))

        def __mul__(self, other):
            if isinstance(other, (int, float, np.integer, np.floating)):
                return ForgeSparse(self._data * other)
            if isinstance(other, ForgeSparse):
                return ForgeSparse(self._data.multiply(other._data))
            return ForgeSparse(self._data.multiply(_unwrap(other)))

        def __matmul__(self, other):
            if isinstance(other, ForgeSparse):
                return ForgeSparse(self._data @ other._data)
            if isinstance(other, ForgeArray):
                return ForgeArray(self._data @ other.data)
            return ForgeArray(self._data @ _unwrap(other))

    def forge_sparse(i_or_A, j=None, v=None, m=None, n=None):
        """Create sparse matrix."""
        if j is None:
            return ForgeSparse(i_or_A)
        return ForgeSparse.from_triplets(i_or_A, j, v, m, n)

    def forge_issparse(x):
        return isinstance(x, ForgeSparse)

    def forge_full(x):
        if isinstance(x, ForgeSparse):
            return x.full()
        return ForgeArray(x)

    HAS_SPARSE = True

except ImportError:
    HAS_SPARSE = False
    def forge_sparse(*args, **kwargs):
        raise ImportError("scipy.sparse not available")
    def forge_issparse(x):
        return False
    def forge_full(x):
        return ForgeArray(x)
