"""Expression fusion: detect and optimize element-wise operation chains.

Conservative optimization that:
1. Uses numpy ufunc out= parameter to reduce intermediate allocations
2. Pre-computes output shape/dtype to skip numpy broadcast analysis on repeat calls
3. Falls back to normal evaluation if any condition is not met

The main win is in tight loops where the same expression shape is evaluated
thousands of times: the ufunc out= path lets numpy skip its internal
temporary allocation.
"""
import numpy as np
from collections import OrderedDict

# Element-wise operations safe for fusion (no matrix semantics)
_ELEMENTWISE_OPS = frozenset({"+", "-", ".*", "./", ".\\", ".^"})

# Map Forge ops to numpy ufuncs for out= parameter support
_OP_TO_UFUNC = {
    "+": np.add,
    "-": np.subtract,
    ".*": np.multiply,
    "./": np.true_divide,
    ".^": np.power,
}


class _ShapeCache:
    """LRU cache of (output_shape, output_dtype) keyed by input signatures.

    Avoids repeated calls to np.broadcast_shapes and np.result_type
    in tight loops.
    """

    __slots__ = ("_cache", "_max")

    def __init__(self, max_entries: int = 64):
        self._cache: OrderedDict = OrderedDict()
        self._max = max_entries

    def get(self, key):
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    def put(self, key, value):
        self._cache[key] = value
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()


# Module-level singleton
_shape_cache = _ShapeCache()


def is_elementwise_op(op: str) -> bool:
    """Return True if *op* is a pure element-wise binary operator."""
    return op in _ELEMENTWISE_OPS


def fused_binop(op: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Evaluate an element-wise binary operation using ufunc with out= parameter.

    Parameters
    ----------
    op : str
        One of the element-wise operators (+, -, .*, ./, .\\, .^).
    left, right : np.ndarray
        Unwrapped operand arrays.

    Returns
    -------
    np.ndarray
        Fresh result array (safe to store -- no aliasing).
    """
    # Look up or compute output metadata
    cache_key = (op, left.shape, left.dtype, right.shape, right.dtype)
    cached = _shape_cache.get(cache_key)
    if cached is not None:
        out_shape, out_dtype = cached
    else:
        try:
            out_shape = np.broadcast_shapes(left.shape, right.shape)
        except ValueError:
            raise
        out_dtype = np.result_type(left.dtype, right.dtype)
        if op == ".^" and np.issubdtype(out_dtype, np.integer):
            out_dtype = np.float64
        _shape_cache.put(cache_key, (out_shape, out_dtype))

    # Allocate output buffer
    out = np.empty(out_shape, dtype=out_dtype)

    # Execute ufunc in-place
    if op == ".\\":
        np.true_divide(right, left, out=out)
    else:
        ufunc = _OP_TO_UFUNC[op]
        ufunc(left, right, out=out)

    return out


def can_fuse(op: str, left, right) -> bool:
    """Check whether a binop can be safely fused.

    Conditions:
    - op must be element-wise
    - both operands must be dense numpy arrays (not sparse, not scalar Python)
    - both must be numeric
    - at least one operand must have >= 100 elements (avoid overhead on tiny arrays)
    """
    if op not in _ELEMENTWISE_OPS:
        return False
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
        return False
    # Skip tiny arrays where overhead exceeds benefit
    total = max(left.size, right.size)
    if total < 100:
        return False
    # Skip non-numeric dtypes
    if not (np.issubdtype(left.dtype, np.number) and
            np.issubdtype(right.dtype, np.number)):
        return False
    return True


def clear_caches():
    """Release all cached metadata.  Called on engine reset / workspace clear."""
    _shape_cache.clear()
