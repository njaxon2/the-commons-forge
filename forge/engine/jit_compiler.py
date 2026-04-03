# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""JIT compilation for numeric for-loops using Numba (optional).

When Numba is available, detects simple numeric for-loops and compiles them
to native code for 10-100x speedup. Gracefully degrades to interpreted
execution when Numba is not installed.
"""
import hashlib
import numpy as np
from typing import Any, Dict, Optional

from forge.engine.parser import (
    ForStatement, Assignment, Index, Identifier, BinaryOp, UnaryOp,
    CompareOp, NumberLiteral, ColonExpr, ExpressionStatement,
    TransposeOp, LogicalOp,
)

# ---------------------------------------------------------------------------
# Numba availability detection
# ---------------------------------------------------------------------------
_NUMBA_AVAILABLE = False
_njit = None

try:
    from numba import njit as _njit_real
    _NUMBA_AVAILABLE = True
    _njit = _njit_real
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Recognized math functions that map to numpy equivalents in Numba
# ---------------------------------------------------------------------------
_JIT_MATH_FUNCS = {
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'sinh', 'cosh', 'tanh',
    'exp', 'log', 'log2', 'log10',
    'sqrt', 'abs', 'floor', 'ceil', 'round', 'fix',
    'mod', 'rem', 'power',
    'sign',
}

# Map Forge/Octave function names to numpy equivalents for code generation
_FUNC_TO_NUMPY = {
    'sin': 'np.sin', 'cos': 'np.cos', 'tan': 'np.tan',
    'asin': 'np.arcsin', 'acos': 'np.arccos', 'atan': 'np.arctan',
    'atan2': 'np.arctan2',
    'sinh': 'np.sinh', 'cosh': 'np.cosh', 'tanh': 'np.tanh',
    'exp': 'np.exp', 'log': 'np.log', 'log2': 'np.log2', 'log10': 'np.log10',
    'sqrt': 'np.sqrt', 'abs': 'np.abs',
    'floor': 'np.floor', 'ceil': 'np.ceil', 'round': 'np.round_',
    'fix': 'np.fix', 'mod': 'np.mod', 'rem': 'np.remainder',
    'power': 'np.power', 'sign': 'np.sign',
}

# Binary operator map
_BINOP_MAP = {
    '+': '+', '-': '-', '*': '*', '/': '/',
    '.*': '*', './': '/', '.^': '**', '^': '**',
}

# ---------------------------------------------------------------------------
# Compiled function cache: AST hash -> compiled function
# ---------------------------------------------------------------------------
_jit_cache: Dict[str, Any] = {}


def is_numba_available() -> bool:
    """Return True if Numba is importable."""
    return _NUMBA_AVAILABLE


def can_jit_loop(for_node: ForStatement) -> bool:
    """Check if a for-loop is JIT-compilable.

    Requirements:
    - Loop variable iterates over a numeric range (ColonExpr with numeric bounds)
    - Body contains only assignments of the form: result(i) = <numeric expr>
    - Numeric expressions may use: arithmetic, recognized math functions, literals
    - No string operations, cell arrays, structs, control flow, or break/continue
    - No nested loops, function calls beyond recognized math
    """
    if not _NUMBA_AVAILABLE:
        return False

    # Iterator must be a ColonExpr (1:N or 1:step:N) with numeric endpoints
    if not isinstance(for_node.iter_expr, ColonExpr):
        return False
    if not _is_numeric_expr(for_node.iter_expr.start):
        return False
    if not _is_numeric_expr(for_node.iter_expr.stop):
        return False
    if for_node.iter_expr.step is not None and not _is_numeric_expr(for_node.iter_expr.step):
        return False

    # Body must be non-empty
    if not for_node.body:
        return False

    loop_var = for_node.var

    for stmt in for_node.body:
        # Unwrap ExpressionStatement wrapper
        actual = stmt
        if isinstance(actual, ExpressionStatement):
            actual = actual.expr

        if not isinstance(actual, Assignment):
            return False

        # Target must be array element assignment: arr(idx) or scalar accumulator
        target = actual.targets
        if isinstance(target, Index):
            if not isinstance(target.target, Identifier):
                return False
            # Only 1D indexing for now: result(i)
            if len(target.args) != 1:
                return False
            if not _is_jit_expr(target.args[0], loop_var):
                return False
        elif isinstance(target, Identifier):
            # Simple scalar assignment like: s = s + expr (accumulator)
            pass
        else:
            return False

        # Value must be a JIT-compatible expression
        if not _is_jit_expr(actual.value, loop_var):
            return False

    return True


def _is_numeric_expr(node) -> bool:
    """Check if an expression resolves to a numeric value."""
    if isinstance(node, NumberLiteral):
        return True
    if isinstance(node, Identifier):
        return True  # Will be checked at runtime
    if isinstance(node, UnaryOp) and node.op in ('+', '-'):
        return _is_numeric_expr(node.operand)
    if isinstance(node, BinaryOp) and node.op in _BINOP_MAP:
        return _is_numeric_expr(node.left) and _is_numeric_expr(node.right)
    return False


def _is_jit_expr(node, loop_var: str) -> bool:
    """Check if an expression is JIT-compilable."""
    if isinstance(node, NumberLiteral):
        return True
    if isinstance(node, Identifier):
        return True
    if isinstance(node, UnaryOp):
        if node.op in ('+', '-'):
            return _is_jit_expr(node.operand, loop_var)
        return False
    if isinstance(node, BinaryOp):
        if node.op in _BINOP_MAP:
            return _is_jit_expr(node.left, loop_var) and _is_jit_expr(node.right, loop_var)
        return False
    if isinstance(node, CompareOp):
        return _is_jit_expr(node.left, loop_var) and _is_jit_expr(node.right, loop_var)
    if isinstance(node, LogicalOp):
        return _is_jit_expr(node.left, loop_var) and _is_jit_expr(node.right, loop_var)
    if isinstance(node, Index):
        if isinstance(node.target, Identifier):
            fname = node.target.name
            if fname in _JIT_MATH_FUNCS:
                return all(_is_jit_expr(a, loop_var) for a in node.args)
            # Array read: arr(expr)
            if len(node.args) == 1:
                return _is_jit_expr(node.args[0], loop_var)
        return False
    if isinstance(node, TransposeOp):
        return _is_jit_expr(node.operand, loop_var)
    return False


def _ast_hash(for_node: ForStatement) -> str:
    """Generate a hash key for caching compiled loops based on AST structure."""
    sig = _ast_signature(for_node)
    return hashlib.md5(sig.encode()).hexdigest()


def _ast_signature(node) -> str:
    """Build a structural signature string from an AST node."""
    if isinstance(node, ForStatement):
        body_sig = ';'.join(_ast_signature(s) for s in node.body)
        return f"FOR({_ast_signature(node.iter_expr)},{body_sig})"
    if isinstance(node, ExpressionStatement):
        return _ast_signature(node.expr)
    if isinstance(node, Assignment):
        return f"ASSIGN({_ast_signature(node.targets)},{_ast_signature(node.value)})"
    if isinstance(node, Index):
        args_sig = ','.join(_ast_signature(a) for a in node.args)
        return f"IDX({_ast_signature(node.target)},{args_sig})"
    if isinstance(node, Identifier):
        return f"ID({node.name})"
    if isinstance(node, NumberLiteral):
        return f"NUM({node.value})"
    if isinstance(node, BinaryOp):
        return f"BIN({node.op},{_ast_signature(node.left)},{_ast_signature(node.right)})"
    if isinstance(node, UnaryOp):
        return f"UN({node.op},{_ast_signature(node.operand)})"
    if isinstance(node, CompareOp):
        return f"CMP({node.op},{_ast_signature(node.left)},{_ast_signature(node.right)})"
    if isinstance(node, LogicalOp):
        return f"LOG({node.op},{_ast_signature(node.left)},{_ast_signature(node.right)})"
    if isinstance(node, ColonExpr):
        step_sig = _ast_signature(node.step) if node.step else "NONE"
        return f"COLON({_ast_signature(node.start)},{step_sig},{_ast_signature(node.stop)})"
    if isinstance(node, TransposeOp):
        return f"TRANS({_ast_signature(node.operand)},{node.conjugate})"
    return f"UNKNOWN({type(node).__name__})"


def compile_and_run_loop(for_node: ForStatement, ws_get, ws_set) -> bool:
    """Compile and execute a JIT-eligible for-loop.

    Args:
        for_node: The ForStatement AST node.
        ws_get: Callable(name) -> value, to read workspace variables.
        ws_set: Callable(name, value), to write workspace variables.

    Returns:
        True if JIT execution succeeded, False if it should fall back.
    """
    if not _NUMBA_AVAILABLE:
        return False

    try:
        return _compile_and_run_impl(for_node, ws_get, ws_set)
    except Exception:
        # Any failure -> fall back to interpreter silently
        return False


def _compile_and_run_impl(for_node: ForStatement, ws_get, ws_set) -> bool:
    """Internal implementation of JIT compilation and execution."""
    loop_var = for_node.var
    colon = for_node.iter_expr

    # Resolve range bounds from workspace
    range_start = _resolve_scalar(colon.start, ws_get)
    range_stop = _resolve_scalar(colon.stop, ws_get)
    range_step = _resolve_scalar(colon.step, ws_get) if colon.step else 1.0

    if range_start is None or range_stop is None or range_step is None:
        return False
    if range_step == 0:
        return False

    # Analyze body: collect written arrays, read arrays, scalar accumulators
    written_arrays = {}
    scalar_accums = {}
    read_arrays = set()
    read_scalars = set()

    for stmt in for_node.body:
        actual = stmt
        if isinstance(actual, ExpressionStatement):
            actual = actual.expr
        if not isinstance(actual, Assignment):
            return False

        target = actual.targets
        if isinstance(target, Index) and isinstance(target.target, Identifier):
            written_arrays[target.target.name] = True
        elif isinstance(target, Identifier):
            scalar_accums[target.name] = True
        else:
            return False

        _collect_reads(actual.value, loop_var, read_arrays, read_scalars,
                       set(written_arrays.keys()) | set(scalar_accums.keys()))

    # Resolve arrays from workspace
    array_data = {}
    n_iters = int(np.floor((range_stop - range_start) / range_step)) + 1
    for name in set(list(written_arrays.keys()) + list(read_arrays)):
        val = ws_get(name)
        if val is None:
            if name in written_arrays:
                array_data[name] = np.zeros(max(n_iters, 1), dtype=np.float64)
            else:
                return False
        else:
            arr = val
            if hasattr(arr, '_data'):
                arr = arr._data
            if isinstance(arr, np.ndarray):
                out = arr.ravel().astype(np.float64, copy=True)
                # Ensure array is large enough for writes
                if name in written_arrays and out.size < n_iters:
                    out = np.concatenate([out, np.zeros(n_iters - out.size, dtype=np.float64)])
                array_data[name] = out
            else:
                try:
                    v = float(arr)
                    if name in written_arrays:
                        out = np.zeros(max(n_iters, 1), dtype=np.float64)
                        out[0] = v
                        array_data[name] = out
                    else:
                        array_data[name] = np.array([v], dtype=np.float64)
                except (TypeError, ValueError):
                    return False

    # Resolve scalar accumulators
    scalar_data = {}
    for name in scalar_accums:
        val = ws_get(name)
        if val is not None:
            try:
                if hasattr(val, '_data'):
                    val = val._data
                scalar_data[name] = float(np.asarray(val).ravel()[0])
            except (TypeError, ValueError):
                return False
        else:
            scalar_data[name] = 0.0

    # Resolve read-only scalars
    readonly_scalars = {}
    for name in read_scalars:
        if name == loop_var or name in array_data or name in scalar_accums:
            continue
        val = ws_get(name)
        if val is None:
            return False
        try:
            if hasattr(val, '_data'):
                val = val._data
            readonly_scalars[name] = float(np.asarray(val).ravel()[0])
        except (TypeError, ValueError):
            return False

    # Generate and compile the function
    cache_key = _ast_hash(for_node)
    if cache_key not in _jit_cache:
        func_src = _generate_jit_function(for_node, loop_var,
                                           written_arrays, scalar_accums,
                                           read_arrays, readonly_scalars)
        if func_src is None:
            return False

        local_ns = {'np': np}
        try:
            exec(func_src, local_ns)  # nosec - generated from validated AST
            raw_func = local_ns['_jit_loop_body']
            jit_func = _njit(cache=False)(raw_func)
            _jit_cache[cache_key] = jit_func
        except Exception:
            return False

    jit_func = _jit_cache[cache_key]

    # Build argument list matching the generated function signature
    args = [range_start, range_stop, range_step]
    for name in sorted(array_data.keys()):
        args.append(array_data[name])
    for name in sorted(scalar_accums.keys()):
        args.append(scalar_data[name])
    for name in sorted(readonly_scalars.keys()):
        args.append(readonly_scalars[name])

    # Execute
    try:
        result = jit_func(*args)
    except Exception:
        return False

    # Unpack results back to workspace
    from forge.engine.types import ForgeArray

    sorted_arrays = sorted(array_data.keys())
    sorted_accums = sorted(scalar_accums.keys())
    total_returns = len(sorted_arrays) + len(sorted_accums)

    if total_returns == 1:
        # Single return value, not a tuple
        if sorted_arrays:
            ws_set(sorted_arrays[0], ForgeArray(np.asarray(result, dtype=np.float64)))
        else:
            ws_set(sorted_accums[0], ForgeArray(np.float64(result)))
    else:
        idx = 0
        for name in sorted_arrays:
            ws_set(name, ForgeArray(np.asarray(result[idx], dtype=np.float64)))
            idx += 1
        for name in sorted_accums:
            ws_set(name, ForgeArray(np.float64(result[idx])))
            idx += 1

    return True


def _resolve_scalar(node, ws_get) -> Optional[float]:
    """Resolve an AST node to a scalar float."""
    if isinstance(node, NumberLiteral):
        try:
            return float(node.value)
        except ValueError:
            return None
    if isinstance(node, Identifier):
        val = ws_get(node.name)
        if val is None:
            return None
        try:
            if hasattr(val, '_data'):
                val = val._data
            return float(np.asarray(val).ravel()[0])
        except (TypeError, ValueError):
            return None
    if isinstance(node, UnaryOp):
        inner = _resolve_scalar(node.operand, ws_get)
        if inner is None:
            return None
        if node.op == '-':
            return -inner
        if node.op == '+':
            return inner
        return None
    if isinstance(node, BinaryOp):
        left = _resolve_scalar(node.left, ws_get)
        right = _resolve_scalar(node.right, ws_get)
        if left is None or right is None:
            return None
        if node.op in ('+',):
            return left + right
        if node.op in ('-',):
            return left - right
        if node.op in ('*', '.*'):
            return left * right
        if node.op in ('/', './'):
            return left / right if right != 0 else None
        if node.op in ('^', '.^'):
            return left ** right
        return None
    return None


def _collect_reads(node, loop_var, arrays, scalars, known_targets):
    """Collect all variable reads from an expression."""
    if isinstance(node, Identifier):
        name = node.name
        if name != loop_var and name not in known_targets:
            scalars.add(name)
    elif isinstance(node, Index):
        if isinstance(node.target, Identifier):
            name = node.target.name
            if name in _JIT_MATH_FUNCS:
                for arg in node.args:
                    _collect_reads(arg, loop_var, arrays, scalars, known_targets)
            else:
                arrays.add(name)
                for arg in node.args:
                    _collect_reads(arg, loop_var, arrays, scalars, known_targets)
        else:
            for arg in node.args:
                _collect_reads(arg, loop_var, arrays, scalars, known_targets)
    elif isinstance(node, (BinaryOp, CompareOp, LogicalOp)):
        _collect_reads(node.left, loop_var, arrays, scalars, known_targets)
        _collect_reads(node.right, loop_var, arrays, scalars, known_targets)
    elif isinstance(node, UnaryOp):
        _collect_reads(node.operand, loop_var, arrays, scalars, known_targets)
    elif isinstance(node, TransposeOp):
        _collect_reads(node.operand, loop_var, arrays, scalars, known_targets)


def _generate_jit_function(for_node, loop_var, written_arrays,
                            scalar_accums, read_arrays, readonly_scalars) -> Optional[str]:
    """Generate Python source for a Numba-compilable function."""
    sorted_arrays = sorted(set(list(written_arrays.keys()) + list(read_arrays)))
    sorted_accums = sorted(scalar_accums.keys())
    sorted_ro = sorted(readonly_scalars.keys())

    # Build parameter list
    params = ['_range_start', '_range_stop', '_range_step']
    for name in sorted_arrays:
        params.append('_arr_' + name)
    for name in sorted_accums:
        params.append('_acc_' + name)
    for name in sorted_ro:
        params.append('_ro_' + name)

    # Build variable name mapping for code generation
    var_map = {}
    var_map[loop_var] = '_i'
    for name in sorted_arrays:
        var_map[name] = '_arr_' + name
    for name in sorted_accums:
        var_map[name] = '_acc_' + name
    for name in sorted_ro:
        var_map[name] = '_ro_' + name

    # Generate loop body
    body_lines = []
    for stmt in for_node.body:
        actual = stmt
        if isinstance(actual, ExpressionStatement):
            actual = actual.expr
        if not isinstance(actual, Assignment):
            return None

        target = actual.targets
        value_code = _expr_to_code(actual.value, var_map, loop_var)
        if value_code is None:
            return None

        if isinstance(target, Index) and isinstance(target.target, Identifier):
            arr_name = var_map.get(target.target.name)
            idx_code = _expr_to_code(target.args[0], var_map, loop_var)
            if arr_name is None or idx_code is None:
                return None
            # Convert 1-based to 0-based indexing
            body_lines.append('        ' + arr_name + '[int(' + idx_code + ') - 1] = ' + value_code)
        elif isinstance(target, Identifier):
            acc_name = var_map.get(target.name)
            if acc_name is None:
                return None
            body_lines.append('        ' + acc_name + ' = ' + value_code)
        else:
            return None

    if not body_lines:
        return None

    body_str = '\n'.join(body_lines)

    # Build return tuple
    returns = []
    for name in sorted_arrays:
        returns.append('_arr_' + name)
    for name in sorted_accums:
        returns.append('_acc_' + name)

    if len(returns) == 1:
        ret_str = returns[0]
    else:
        ret_str = '(' + ', '.join(returns) + ')'

    func_src = (
        'def _jit_loop_body(' + ', '.join(params) + '):\n'
        '    _n = int(np.floor((_range_stop - _range_start) / _range_step)) + 1\n'
        '    for _idx in range(_n):\n'
        '        _i = _range_start + _idx * _range_step\n'
        + body_str + '\n'
        '    return ' + ret_str + '\n'
    )
    return func_src


def _expr_to_code(node, var_map, loop_var) -> Optional[str]:
    """Convert an AST expression node to Python source code string."""
    if isinstance(node, NumberLiteral):
        return str(float(node.value))
    if isinstance(node, Identifier):
        name = node.name
        if name in var_map:
            return var_map[name]
        if name == 'pi':
            return 'np.pi'
        if name == 'e':
            return 'np.e'
        if name in ('inf', 'Inf'):
            return 'np.inf'
        if name in ('nan', 'NaN'):
            return 'np.nan'
        return None
    if isinstance(node, UnaryOp):
        inner = _expr_to_code(node.operand, var_map, loop_var)
        if inner is None:
            return None
        if node.op == '-':
            return '(-' + inner + ')'
        if node.op == '+':
            return inner
        return None
    if isinstance(node, BinaryOp):
        left = _expr_to_code(node.left, var_map, loop_var)
        right = _expr_to_code(node.right, var_map, loop_var)
        if left is None or right is None:
            return None
        op = _BINOP_MAP.get(node.op)
        if op is None:
            return None
        return '(' + left + ' ' + op + ' ' + right + ')'
    if isinstance(node, CompareOp):
        left = _expr_to_code(node.left, var_map, loop_var)
        right = _expr_to_code(node.right, var_map, loop_var)
        if left is None or right is None:
            return None
        op = node.op
        if op == '~=':
            op = '!='
        return '(' + left + ' ' + op + ' ' + right + ')'
    if isinstance(node, LogicalOp):
        left = _expr_to_code(node.left, var_map, loop_var)
        right = _expr_to_code(node.right, var_map, loop_var)
        if left is None or right is None:
            return None
        op_map = {'&': '&', '|': '|', '&&': 'and', '||': 'or'}
        op = op_map.get(node.op)
        if op is None:
            return None
        return '(' + left + ' ' + op + ' ' + right + ')'
    if isinstance(node, Index):
        if isinstance(node.target, Identifier):
            fname = node.target.name
            if fname in _JIT_MATH_FUNCS:
                np_name = _FUNC_TO_NUMPY.get(fname)
                if np_name is None:
                    return None
                arg_codes = []
                for arg in node.args:
                    c = _expr_to_code(arg, var_map, loop_var)
                    if c is None:
                        return None
                    arg_codes.append(c)
                return np_name + '(' + ', '.join(arg_codes) + ')'
            else:
                arr_name = var_map.get(fname)
                if arr_name is None:
                    return None
                if len(node.args) != 1:
                    return None
                idx_code = _expr_to_code(node.args[0], var_map, loop_var)
                if idx_code is None:
                    return None
                return arr_name + '[int(' + idx_code + ') - 1]'
        return None
    if isinstance(node, TransposeOp):
        inner = _expr_to_code(node.operand, var_map, loop_var)
        return inner
    return None
