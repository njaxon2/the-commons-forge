# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""M-language evaluator: AST → execution.

Walks the AST produced by the parser, evaluating expressions and executing statements
against a workspace (variable scope) with access to built-in functions.
"""
import numpy as np
import sys
import time
import os
from scipy.linalg import solve as _scipy_solve, solve_triangular as _solve_tri, cho_factor as _cho_factor, cho_solve as _cho_solve
import scipy.sparse as _sp

# Pre-seeded RNG for sampled symmetry check (deterministic, no perf cost)
_MLDIVIDE_RNG = np.random.default_rng(42)

def _smart_mldivide(A, b):
    """Fast-dispatch linear solve with structure detection."""
    m, n = A.shape
    if m != n:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    # Fast triangular check: O(n) via triu/tril nonzero scan
    if not np.any(np.triu(A, 1)):  # lower triangular
        return _solve_tri(A, b, lower=True, check_finite=False)
    if not np.any(np.tril(A, -1)):  # upper triangular
        return _solve_tri(A, b, lower=False, check_finite=False)
    # Sampled symmetry check for large matrices -> Cholesky if SPD
    if n >= 64:
        k = min(100, n)
        idx = _MLDIVIDE_RNG.integers(0, n, size=k)
        jdx = _MLDIVIDE_RNG.integers(0, n, size=k)
        if np.allclose(A[idx, jdx], A[jdx, idx]):
            try:
                cf = _cho_factor(A, check_finite=False)
                return _cho_solve(cf, b, check_finite=False)
            except np.linalg.LinAlgError:
                pass
    # General dense: scipy with skip finite check
    return _scipy_solve(A, b, overwrite_a=False, overwrite_b=False, check_finite=False)

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
    ForgeTable,
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
from forge.engine.classdef import (
    ForgeObject, ForgeClass, Property, Method, register_class,
    get_class, class_exists,
)
from forge.engine.builtins.sets import forge_ismember
from forge.engine.expr_fusion import can_fuse, fused_binop
from forge.engine.jit_compiler import can_jit_loop, compile_and_run_loop, is_numba_available


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
        self._global_store: Dict[str, Any] = None  # ref to Session._global_store
        self._persistent_vars: set = set()

    def get(self, name: str) -> Any:
        # Check globals: if declared global and exists in global store
        if name in self._globals and self._global_store is not None and name in self._global_store:
            return self._global_store[name]
        if name in self._vars:
            return self._vars[name]
        if self._parent:
            return self._parent.get(name)
        raise NameError(f"Undefined variable: {name}")

    def set(self, name: str, value: Any):
        # If variable is declared global, store in global store
        if name in self._globals and self._global_store is not None:
            self._global_store[name] = value
        self._vars[name] = value

    def has(self, name: str) -> bool:
        if name in self._globals and self._global_store is not None and name in self._global_store:
            return True
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
    """Apply function to each element of a cell array.

    Supports:
      cellfun(@func, C)
      cellfun(@func, C1, C2, ...)           -- multiple cell inputs
      cellfun(@func, C, 'UniformOutput', false)
      cellfun('isclass', C, 'char')         -- string-form builtins
    """

    # --- Handle string-form func names ---
    uniform_output = True
    positional_extras = list(extra_args)

    if isinstance(func, (str, ForgeChar)):
        func_name = func.to_str() if isinstance(func, ForgeChar) else func
        if func_name == 'isclass':
            if positional_extras:
                class_name = positional_extras.pop(0)
                if isinstance(class_name, ForgeChar):
                    class_name = class_name.to_str()
                elif isinstance(class_name, str):
                    pass
                else:
                    class_name = str(class_name)
                type_map = {
                    'char': (str, ForgeChar),
                    'double': (float, int, np.floating, np.integer, ForgeArray),
                    'cell': (ForgeCell, list),
                    'logical': (bool, np.bool_),
                }
                target = type_map.get(class_name, ())
                func = lambda item, _t=target: bool(isinstance(item, _t))
            else:
                func = lambda item: False
        elif func_name in ('ischar', 'isstring'):
            func = lambda item: bool(isinstance(item, (str, ForgeChar)))
        elif func_name == 'isnumeric':
            func = lambda item: bool(isinstance(item, (int, float, np.number, ForgeArray)))
        elif func_name == 'isempty':
            func = lambda item: bool(
                (isinstance(item, ForgeArray) and item.data.size == 0) or
                (isinstance(item, (str, ForgeChar)) and len(item.to_str() if isinstance(item, ForgeChar) else item) == 0)
            )
        else:
            raise ValueError(f"Unknown cellfun string function: {func_name}")

    # --- Parse name-value pairs from positional_extras ---
    clean_extras = []
    i = 0
    while i < len(positional_extras):
        arg = positional_extras[i]
        arg_str = arg.to_str() if isinstance(arg, ForgeChar) else (arg if isinstance(arg, str) else None)
        if arg_str and arg_str.lower() == 'uniformoutput' and i + 1 < len(positional_extras):
            val = positional_extras[i + 1]
            if isinstance(val, ForgeArray):
                uniform_output = bool(val.data.flat[0])
            elif isinstance(val, (bool, np.bool_)):
                uniform_output = bool(val)
            elif isinstance(val, (int, float)):
                uniform_output = bool(val)
            else:
                uniform_output = bool(val)
            i += 2
            continue
        clean_extras.append(arg)
        i += 1

    # --- Resolve cell items ---
    if isinstance(cell_arg, ForgeCell):
        items = cell_arg._data
    elif isinstance(cell_arg, (list, tuple)):
        items = list(cell_arg)
    else:
        items = [cell_arg]

    # --- Extra cell arrays for multi-input ---
    extra_cells = []
    for ea in clean_extras:
        if isinstance(ea, ForgeCell):
            extra_cells.append(ea._data)
        elif isinstance(ea, (list, tuple)):
            extra_cells.append(list(ea))
        else:
            extra_cells.append([ea])

    results = []
    for idx, item in enumerate(items):
        if extra_cells:
            call_args = [item] + [ec[idx] if idx < len(ec) else ec[-1] for ec in extra_cells]
            r = func(*call_args)
        else:
            r = func(item)
        results.append(r)

    if not uniform_output:
        return ForgeCell(results)

    # Check if all results are scalars -> return array
    all_scalar = all(
        isinstance(r, (int, float, bool, np.integer, np.floating, np.bool_)) or
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
            elif isinstance(r, bool):
                vals.append(1.0 if r else 0.0)
            else:
                vals.append(float(r))
        return ForgeArray(np.array(vals, dtype=np.float64))
    return ForgeCell(results)


def _arrayfun_builtin(func, arr, *extra_arrs, **kwargs):
    """Apply function element-wise to array(s).

    Supports:
      arrayfun(@func, A)
      arrayfun(@func, A, B, ...)            -- multiple array inputs
      arrayfun(@func, A, 'UniformOutput', false)
    """

    # --- Parse name-value pairs from extra_arrs ---
    uniform_output = True
    clean_extras = []
    args_list = list(extra_arrs)
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        arg_str = arg.to_str() if isinstance(arg, ForgeChar) else (arg if isinstance(arg, str) else None)
        if arg_str and arg_str.lower() == 'uniformoutput' and i + 1 < len(args_list):
            val = args_list[i + 1]
            if isinstance(val, ForgeArray):
                uniform_output = bool(val.data.flat[0])
            elif isinstance(val, (bool, np.bool_)):
                uniform_output = bool(val)
            elif isinstance(val, (int, float)):
                uniform_output = bool(val)
            else:
                uniform_output = bool(val)
            i += 2
            continue
        clean_extras.append(arg)
        i += 1

    if isinstance(arr, ForgeArray):
        data = arr.data
    elif isinstance(arr, np.ndarray):
        data = arr
    else:
        data = np.atleast_1d(np.asarray(arr, dtype=np.float64))

    extra_data = []
    for ea in clean_extras:
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
        results.append(r)

    if not uniform_output:
        return ForgeCell(results)

    # Uniform output: convert to numeric array
    vals = []
    for r in results:
        if isinstance(r, ForgeArray):
            vals.append(float(r.data.flat[0]))
        elif isinstance(r, np.ndarray):
            vals.append(float(r.flat[0]))
        else:
            vals.append(float(r))

    result = np.array(vals, dtype=np.float64).reshape(data.shape)
    return ForgeArray(result)


def _num2cell_builtin(arr, *args):
    """Convert array to cell array, optionally along a dimension."""
    if isinstance(arr, ForgeArray):
        data = arr.data
    elif isinstance(arr, np.ndarray):
        data = arr
    else:
        return ForgeCell([arr])

    data = np.atleast_2d(data)

    # Check for dim argument
    dim = None
    if args:
        d = args[0]
        if isinstance(d, ForgeArray):
            dim = int(d.data.flat[0])
        elif isinstance(d, (int, float)):
            dim = int(d)

    if dim is not None:
        # dim=1: split along columns (each column becomes a cell element)
        # dim=2: split along rows (each row becomes a cell element)
        if dim == 1:
            # Each column becomes a cell element
            result = []
            for j in range(data.shape[1]):
                result.append(ForgeArray(data[:, j:j+1].copy()))
            cell = ForgeCell(result)
            cell._shape = (1, data.shape[1])
            return cell
        elif dim == 2:
            # Each row becomes a cell element
            result = []
            for i in range(data.shape[0]):
                result.append(ForgeArray(data[i:i+1, :].copy()))
            cell = ForgeCell(result)
            cell._shape = (data.shape[0], 1)
            return cell
        else:
            # Higher dims: fall through to element-wise
            pass

    # No dim: convert each element to a cell, preserving 2d shape
    result = [ForgeArray(np.atleast_1d(np.array(x))) for x in data.ravel()]
    cell = ForgeCell(result)
    if data.ndim >= 2:
        cell._shape = data.shape
    return cell


def _cell2mat_builtin(c):
    """Convert cell array of matrices to a single matrix."""
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
        # Expand arrays to have enough dimensions for the target axis
        expanded = []
        for a in np_arrays:
            while a.ndim <= dim:
                a = np.expand_dims(a, axis=-1)
            expanded.append(a)
        result = np.concatenate(expanded, axis=dim)

    return ForgeArray(result)



def _deal_builtin(*args):
    """Deal inputs to outputs. [a,b,c] = deal(x,y,z) or [a,b] = deal(x)."""
    if len(args) == 1:
        # Replicate single input for all outputs
        return args[0]
    return args


def _structfun_builtin(func, s, *extra_args):
    """Apply function to each field of a struct.

    Supports:
      structfun(@func, S)
      structfun(@func, S, 'UniformOutput', false)
    """
    import numpy as np
    if not isinstance(s, ForgeStruct):
        raise ValueError("Second argument must be a struct")

    # Parse name-value pairs
    uniform_output = True
    args_list = list(extra_args)
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        arg_str = arg.to_str() if isinstance(arg, ForgeChar) else (arg if isinstance(arg, str) else None)
        if arg_str and arg_str.lower() == 'uniformoutput' and i + 1 < len(args_list):
            val = args_list[i + 1]
            if isinstance(val, ForgeArray):
                uniform_output = bool(val.data.flat[0])
            elif isinstance(val, (bool, np.bool_)):
                uniform_output = bool(val)
            else:
                uniform_output = bool(val)
            i += 2
            continue
        i += 1

    results = []
    for name, val in s._fields.items():
        r = func(val)
        results.append(r)

    if not uniform_output:
        return ForgeCell(results)

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
        self._global_store: Dict[str, Any] = {}  # shared global variable storage
        self._persistent_store: Dict[str, Dict[str, Any]] = {}  # func_name -> {var: val}
        self._current_function: str = ""  # track current executing function name
        self._setup_builtins()
        self._setup_constants()
        self._wire_global_store()
        self._session_ref = None  # Set by ForgeSession to enable .m file discovery
        # -- Dispatch tables for O(1) node-type routing (perf #28) --
        self._init_dispatch_tables()


    def _init_dispatch_tables(self):
        """Build dict-based dispatch tables for statement and expression evaluation.

        Replaces isinstance chains with O(1) dict lookup for ~20 node types each.
        """
        # -- Statement dispatch --
        self._stmt_dispatch = {
            Assignment: self._exec_assign,
            IfStatement: self._exec_if,
            ForStatement: self._exec_for,
            WhileStatement: self._exec_while,
            DoUntilStatement: self._exec_do_until,
            SwitchStatement: self._exec_switch,
            TryCatchStatement: self._exec_try,
            ClassDef: self._exec_classdef,
            UnwindProtect: self._exec_unwind_protect,
        }

        # -- Expression dispatch --
        self._expr_dispatch = {
            NumberLiteral: self._eval_number,
            StringLiteral: self._eval_string_literal,
            Identifier: self._eval_identifier,
            UnaryOp: self._eval_unary,
            BinaryOp: self._eval_binop,
            CompareOp: self._eval_compare,
            LogicalOp: self._eval_logical,
            TransposeOp: self._eval_transpose,
            BareColon: self._eval_bare_colon,
            ColonExpr: self._eval_colon_expr,
            Index: self._eval_index,
            CellIndex: self._eval_cell_index,
            FieldAccess: self._eval_field_access,
            DynamicFieldAccess: self._eval_dynamic_field,
            MatrixLiteral: self._eval_matrix,
            CellLiteral: self._eval_cell_literal,
            FunctionHandle: self._eval_function_handle,
            AnonFunction: self._eval_anon_function,
            EndKeyword: self._eval_end_keyword,
        }

        # -- LRU cache for NumberLiteral results (perf: avoid re-parsing constants) --
        self._number_cache = {}

    def _wire_global_store(self):
        """Connect workspace to shared global store."""
        self.workspace._global_store = self._global_store

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
        b["isempty"] = lambda x: ForgeArray(np.float64(x.isempty() if isinstance(x, ForgeArray) else len(x) == 0))
        def _isfield(s, f):
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

        # Math — map Octave names to numpy (arcsin etc. for older numpy versions)
        _np_name_map = {
            'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan',
            'asinh': 'arcsinh', 'acosh': 'arccosh', 'atanh': 'arctanh',
        }
        for name in ['abs','sqrt','exp','log','log2','log10','sin','cos','tan',
                      'asin','acos','atan','sinh','cosh','tanh','ceil','floor',
                      'round','fix','sign','real','imag','conj','angle']:
            np_name = _np_name_map.get(name, name)
            fn = getattr(np, np_name) if np_name != name else getattr(np, name)
            def _make_math(f, nm):
                def _fn(*a):
                    if not a: raise ValueError(f"{nm} requires at least 1 argument")
                    x = a[0]
                    if isinstance(x, ForgeArray):
                        result = f(x._data)
                        if type(result) is np.ndarray and result.ndim >= 2:
                            return ForgeArray._from_ndarray(result)
                    else:
                        result = f(_unwrap(x))
                    return ForgeArray(result)
                return _fn
            b[name] = _make_math(fn, name)

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
        def _forge_reshape(x, *a):
            dims = []
            for v in a:
                pv = _to_py(v)
                if isinstance(pv, np.ndarray) and pv.size == 0:
                    dims.append(-1)  # [] means auto-compute
                else:
                    dims.append(int(pv))
            data = _unwrap(x)
            # Octave reshape uses Fortran order
            return ForgeArray(data.reshape(dims, order='F'))
        b["reshape"] = _forge_reshape
        b["squeeze"] = lambda x: ForgeArray(np.squeeze(_unwrap(x)))
        b["permute"] = lambda x, order: ForgeArray(np.transpose(_unwrap(x), [int(_to_py(o))-1 for o in _unwrap(order).flatten()]))
        b["ndims"] = lambda x: ForgeArray(np.float64(max(2, _unwrap(x).ndim)))
        b["rows"] = lambda x: ForgeArray(np.float64(_unwrap(x).shape[0] if _unwrap(x).ndim >= 1 else 1))
        b["columns"] = lambda x: ForgeArray(np.float64(_unwrap(x).shape[1] if _unwrap(x).ndim >= 2 else 1))
        b["transpose"] = lambda x: x.T if isinstance(x, ForgeArray) else ForgeArray(np.asarray(x).T)
        b["ctranspose"] = lambda x: ForgeArray(np.conj(np.asarray(x.data if isinstance(x, ForgeArray) else x).T))
        def _fast_sum(x, *a):
            data = x._data if isinstance(x, ForgeArray) else _unwrap(x)
            result = np.sum(data, axis=int(_to_py(a[0]))-1 if a else None)
            return ForgeArray._from_ndarray(result) if type(result) is np.ndarray and result.ndim >= 2 else ForgeArray(result)
        b["sum"] = _fast_sum
        def _fast_prod(x, *a):
            data = x._data if isinstance(x, ForgeArray) else _unwrap(x)
            result = np.prod(data, axis=int(_to_py(a[0]))-1 if a else None)
            return ForgeArray._from_ndarray(result) if type(result) is np.ndarray and result.ndim >= 2 else ForgeArray(result)
        b["prod"] = _fast_prod
        def _forge_min(x, *a):
            data = _unwrap(x)
            if not a:
                if np.asarray(data).size == 0:
                    return ForgeArray(np.array([]).reshape(0, 0))
                return ForgeArray(np.min(data))
            return ForgeArray(np.minimum(data, _unwrap(a[0])))
        b["min"] = _forge_min
        def _forge_max(x, *a):
            data = _unwrap(x)
            if not a:
                if np.asarray(data).size == 0:
                    return ForgeArray(np.array([]).reshape(0, 0))
                return ForgeArray(np.max(data))
            return ForgeArray(np.maximum(data, _unwrap(a[0])))
        b["max"] = _forge_max
        b["sort"] = lambda x, *a: ForgeArray(np.sort(_unwrap(x), axis=-1))
        def _forge_find_builtin(x):
            data = _unwrap(x) if isinstance(x, ForgeArray) else x
            if _sp.issparse(data):
                coo = _sp.coo_matrix(data)
                m = coo.shape[0]
                idx = coo.col * m + coo.row + 1  # column-major 1-based
                return ForgeArray(np.sort(idx).astype(float))
            return ForgeArray(np.flatnonzero(data) + 1)
        b["find"] = _forge_find_builtin  # 1-based
        b["any"] = lambda x, *a: ForgeArray(np.array(np.any(_unwrap(x))))
        b["all"] = lambda x, *a: ForgeArray(np.array(np.all(_unwrap(x))))
        def _fast_cumsum(x, *a):
            data = x._data if isinstance(x, ForgeArray) else _unwrap(x)
            if a:
                axis = int(_to_py(a[0])) - 1
            elif data.ndim >= 2 and data.shape[0] == 1:
                axis = 1  # Row vector: cumsum along columns (preserves shape)
            elif data.ndim >= 2 and data.shape[1] == 1:
                axis = 0  # Column vector: cumsum along rows
            else:
                axis = None
            result = np.cumsum(data, axis=axis)
            if type(result) is np.ndarray and result.ndim >= 2:
                return ForgeArray._from_ndarray(result)
            return ForgeArray(result)
        b["cumsum"] = _fast_cumsum
        def _fast_cumprod(x, *a):
            data = x._data if isinstance(x, ForgeArray) else _unwrap(x)
            if a:
                axis = int(_to_py(a[0])) - 1
            elif data.ndim >= 2 and data.shape[0] == 1:
                axis = 1
            elif data.ndim >= 2 and data.shape[1] == 1:
                axis = 0
            else:
                axis = None
            result = np.cumprod(data, axis=axis)
            if type(result) is np.ndarray and result.ndim >= 2:
                return ForgeArray._from_ndarray(result)
            return ForgeArray(result)
        b["cumprod"] = _fast_cumprod
        b["diff"] = lambda x, *a: ForgeArray(np.diff(_unwrap(x), n=_to_py(a[0]) if a else 1, axis=-1))
        b["cat"] = lambda dim, *arrays: ForgeArray(np.concatenate([_unwrap(a) for a in arrays], axis=_to_py(dim)-1))
        def _forge_horzcat(*a):
            non_empty = [_unwrap(x) for x in a if np.asarray(_unwrap(x)).size > 0]
            if not non_empty:
                return ForgeArray(np.array([]).reshape(0, 0))
            try:
                return ForgeArray(np.concatenate(non_empty, axis=1))
            except ValueError:
                raise ValueError("horizontal dimensions mismatch")
        b["horzcat"] = _forge_horzcat
        def _forge_vertcat(*a):
            non_empty = [_unwrap(x) for x in a if np.asarray(_unwrap(x)).size > 0]
            if not non_empty:
                return ForgeArray(np.array([]).reshape(0, 0))
            try:
                return ForgeArray(np.concatenate(non_empty, axis=0))
            except ValueError:
                raise ValueError("vertical dimensions mismatch")
        b["vertcat"] = _forge_vertcat
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
        b["cellfun"] = _cellfun_builtin
        b["arrayfun"] = _arrayfun_builtin
        b["num2cell"] = _num2cell_builtin
        b["cell2mat"] = _cell2mat_builtin
        b["rmfield"] = _rmfield_builtin
        b["cat"] = _cat_builtin

        def _getfield_builtin(s, *fields):
            cur = s
            for f in fields:
                fname = f.to_str() if isinstance(f, ForgeChar) else str(f)
                if isinstance(cur, ForgeStruct):
                    cur = cur._fields[fname]
                else:
                    raise ValueError(f"Cannot get field '{fname}' from non-struct")
            return cur

        def _setfield_builtin(s, *args):
            if len(args) < 2:
                raise ValueError("setfield requires at least 3 arguments")
            *fields, value = args
            if not isinstance(s, ForgeStruct):
                raise ValueError("First argument must be a struct")
            new_s = ForgeStruct()
            for k, v in s._fields.items():
                new_s._fields[k] = v
            fname = fields[0].to_str() if isinstance(fields[0], ForgeChar) else str(fields[0])
            new_s._fields[fname] = value
            return new_s

        b["getfield"] = _getfield_builtin
        b["setfield"] = _setfield_builtin
        b["orderfields"] = lambda s: __import__('forge.engine.containers', fromlist=['forge_orderfields']).forge_orderfields(s)

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
            if isinstance(x, ForgeArray):
                # Map numpy dtype to MATLAB class name
                _dtype_map = {
                    "float64": "double", "float32": "single",
                    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
                    "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
                    "bool": "logical", "complex128": "double", "complex64": "single",
                }
                dtype_name = str(x.data.dtype)
                return ForgeChar(_dtype_map.get(dtype_name, dtype_name))
            if isinstance(x, ForgeChar):
                return ForgeChar("char")
            if isinstance(x, ForgeCell):
                return ForgeChar("cell")
            if isinstance(x, ForgeStruct):
                return ForgeChar("struct")
            if isinstance(x, bool):
                return ForgeChar("logical")
            if isinstance(x, (int, float)):
                return ForgeChar("double")
            if isinstance(x, str):
                return ForgeChar("char")
            if isinstance(x, ForgeObject):
                return ForgeChar(x.class_name)
            return ForgeChar(type(x).__name__)
        b["class"] = _class_name
        b["typecast"] = lambda x, t: x.astype(t.to_str() if isinstance(t, ForgeChar) else str(t))

        # narginchk / nargoutchk
        def _narginchk(minargs, maxargs):
            _min = int(_unwrap(minargs).flat[0]) if isinstance(minargs, ForgeArray) else int(minargs)
            _max = int(_unwrap(maxargs).flat[0]) if isinstance(maxargs, ForgeArray) else int(maxargs)
            # Read nargin from the current workspace (set by function call)
            _ws = self._current_workspace if hasattr(self, "_current_workspace") else self.workspace
            _ni = _ws.get("nargin") if _ws.has("nargin") else None
            if _ni is not None:
                _n = int(_unwrap(_ni).flat[0]) if isinstance(_ni, ForgeArray) else int(_ni)
                if _n < _min:
                    raise ForgeError("Forge:narginchk", f"narginchk: incorrect number of input arguments (got {_n}, need at least {_min})")
                if _n > _max:
                    raise ForgeError("Forge:narginchk", f"narginchk: too many input arguments (got {_n}, max is {_max})")
        def _nargoutchk(minargs, maxargs):
            _min = int(_unwrap(minargs).flat[0]) if isinstance(minargs, ForgeArray) else int(minargs)
            _max = int(_unwrap(maxargs).flat[0]) if isinstance(maxargs, ForgeArray) else int(maxargs)
            _ws = self._current_workspace if hasattr(self, "_current_workspace") else self.workspace
            _no = _ws.get("nargout") if _ws.has("nargout") else None
            if _no is not None:
                _n = int(_unwrap(_no).flat[0]) if isinstance(_no, ForgeArray) else int(_no)
                if _n < _min:
                    raise ForgeError("Forge:nargoutchk", f"nargoutchk: incorrect number of output arguments (got {_n}, need at least {_min})")
                if _n > _max:
                    raise ForgeError("Forge:nargoutchk", f"nargoutchk: too many output arguments (got {_n}, max is {_max})")
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

        # feval(funcname, args...) - call function by name or handle
        def _feval(name, *args):
            # If name is already a callable (function handle), call it directly
            if callable(name) and not isinstance(name, (ForgeChar, ForgeArray)):
                return name(*args)
            name = name.to_str() if isinstance(name, ForgeChar) else str(name)
            if name in self.functions:
                return self.functions[name](*args)
            raise NameError(f"undefined function '{name}'")
        b["feval"] = _feval

        # str2func(name) - convert string to function handle
        def _str2func(name):
            name = name.to_str() if isinstance(name, ForgeChar) else str(name)
            # Handle anonymous function strings like '@(x) x^2'
            if name.startswith('@'):
                return self.eval(name)
            if name in self.functions:
                fn = self.functions[name]
                if isinstance(fn, FunctionDef):
                    result = lambda *a: self._call_function(fn, list(a), self.workspace)
                    result._forge_name = name
                    return result
                if not hasattr(fn, '_forge_name'):
                    fn._forge_name = name
                return fn
            raise NameError(f"undefined function '{name}'")
        b["str2func"] = _str2func

        # func2str(handle) - convert function handle to string
        def _func2str(handle):
            if hasattr(handle, '_forge_name'):
                return ForgeChar(handle._forge_name)
            if hasattr(handle, '__name__'):
                name = handle.__name__
                # Strip lambda wrapper prefixes
                if name != '<lambda>':
                    return ForgeChar(name)
            # For anonymous functions, return generic representation
            return ForgeChar('@<anonymous>')
        b["func2str"] = _func2str

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
        """size(A) — array dimensions. [m, n] = size(A)."""
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
        """disp(x) — display value of variable."""
        for a in args:
            if isinstance(a, ForgeChar):
                self.output_buffer.write(a.to_str() + "\n")
            elif isinstance(a, ForgeArray):
                self.output_buffer.write(str(a.data) + "\n")
            else:
                self.output_buffer.write(str(a) + "\n")

    def _builtin_fprintf(self, *args):
        """fprintf(fid, fmt, ...) — write formatted data to file."""
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
        """Execute a single AST node (dict-dispatch, perf #28)."""
        # Fast path: ExpressionStatement is the most common node
        node_type = type(node)
        if node_type is ExpressionStatement:
            return self._exec_expression_stmt(node, ws)

        # Dict dispatch for nodes with dedicated handler methods
        handler = self._stmt_dispatch.get(node_type)
        if handler is not None:
            return handler(node, ws)

        # Signal nodes (raise, no return)
        if node_type is ReturnStatement:
            raise ReturnSignal()
        if node_type is BreakStatement:
            raise BreakSignal()
        if node_type is ContinueStatement:
            raise ContinueSignal()

        # Inline handlers for less-frequent nodes
        if node_type is FunctionDef:
            self.functions[node.name] = node
            return None
        if node_type is GlobalStatement:
            return self._exec_global(node, ws)
        if node_type is PersistentStatement:
            return self._exec_persistent(node, ws)

        raise RuntimeError(f"Unknown AST node: {type(node).__name__}")

    def _exec_expression_stmt(self, node, ws: Workspace) -> Any:
        """Handle ExpressionStatement nodes."""
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

    def _exec_global(self, node, ws: Workspace) -> None:
        """Handle GlobalStatement nodes."""
        for name in node.names:
            ws._globals.add(name)
            # Sync: if global store has a value, load it into workspace
            if name in self._global_store:
                ws.set(name, self._global_store[name])
        return None

    def _exec_persistent(self, node, ws: Workspace) -> None:
        """Handle PersistentStatement nodes."""
        func_name = self._current_function
        if func_name:
            store = self._persistent_store.setdefault(func_name, {})
            for name in node.names:
                if name in store:
                    ws.set(name, store[name])
                else:
                    # First call: initialize to empty matrix (Octave behavior)
                    ws.set(name, ForgeArray(np.empty((0, 0))))
                # Mark variable as persistent in workspace
                if not hasattr(ws, '_persistent_vars'):
                    ws._persistent_vars = set()
                ws._persistent_vars.add(name)
        return None

    # ============================================================
    # Expression evaluation
    # ============================================================

    def _eval_expr(self, node, ws: Workspace) -> Any:
        """Evaluate an expression AST node (dict-dispatch, perf #28)."""
        handler = self._expr_dispatch.get(type(node))
        if handler is not None:
            return handler(node, ws)
        raise RuntimeError(f"Cannot evaluate: {type(node).__name__}")

    # -- Extracted expression handlers (uniform signature: self, node, ws) --

    def _eval_string_literal(self, node, ws: Workspace):
        """Handle StringLiteral nodes."""
        if node.is_char:
            return ForgeChar(node.value)
        return ForgeChar(node.value)  # Both become char for now

    def _eval_identifier(self, node, ws: Workspace):
        """Handle Identifier nodes."""
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

    def _eval_unary(self, node, ws: Workspace):
        """Handle UnaryOp nodes."""
        val = self._eval_expr(node.operand, ws)
        if node.op == "-":
            return -val if isinstance(val, ForgeArray) else ForgeArray(-np.asarray(val))
        if node.op == "~":
            return ForgeArray(~np.asarray(_unwrap(val), dtype=bool))

    def _eval_transpose(self, node, ws: Workspace):
        """Handle TransposeOp nodes."""
        val = self._eval_expr(node.operand, ws)
        if isinstance(val, ForgeArray):
            if node.conjugate:
                return ForgeArray(np.conj(val.data).T)
            return val.T
        return ForgeArray(np.asarray(val).T)

    def _eval_bare_colon(self, node, ws: Workspace):
        """Handle BareColon nodes."""
        return None

    def _eval_colon_expr(self, node, ws: Workspace):
        """Handle ColonExpr nodes."""
        start = _to_py(self._eval_expr(node.start, ws))
        stop = _to_py(self._eval_expr(node.stop, ws))
        if node.step is not None:
            step = _to_py(self._eval_expr(node.step, ws))
            return forge_colon(start, step, stop)
        return forge_colon(start, stop)

    def _eval_cell_index(self, node, ws: Workspace):
        """Handle CellIndex nodes."""
        target = self._eval_expr(node.target, ws)
        if isinstance(target, ForgeCell):
            self._index_sizes.append(target.numel())
            try:
                raw_args = [self._eval_expr(a, ws) for a in node.args]
            finally:
                self._index_sizes.pop()
            # BareColon expands to all elements (e.g. c{:})
            if len(raw_args) == 1 and raw_args[0] is None:
                # Return tuple of all elements for expansion
                if target.numel() == 1:
                    return target.content_get(1)
                return tuple(target._data)
            args = [_to_int(a) for a in raw_args]
            return target.content_get(*args)
        raise TypeError("Cell indexing on non-cell")

    def _eval_field_access(self, node, ws: Workspace):
        """Handle FieldAccess nodes."""
        target = self._eval_expr(node.target, ws)
        if isinstance(target, ForgeStruct):
            return target._fields[node.field]
        # Support ForgeMap dot access (Count, KeyType, ValueType, keys, values, etc.)
        if isinstance(target, ForgeMap):
            field = node.field
            if field == "Count":
                return target.Count
            if field == "KeyType":
                return target.KeyType
            if field == "ValueType":
                return target.ValueType
            # Method-like access: keys(), values(), isKey(), remove(), length()
            method = getattr(target, field, None)
            if method is not None and callable(method):
                return method
            raise TypeError(f"containers.Map has no property '{field}'")
        # Support ForgeTable dot access (column names, Properties)
        if isinstance(target, ForgeTable):
            field = node.field
            if field == "Properties":
                # Return a struct with VariableNames
                props = ForgeStruct()
                props._fields["VariableNames"] = ForgeCell(
                    [ForgeChar(n) for n in target._var_names])
                return props
            return target.get_column(field)
        # Support ForgeObject (classdef instances)
        if isinstance(target, ForgeObject):
            return getattr(target, node.field)
        # Generic Python object dot access (e.g. inputParser methods/properties)
        if hasattr(target, node.field):
            val = getattr(target, node.field)
            # Wrap plain dict as ForgeStruct for further dot access
            if isinstance(val, dict):
                s = ForgeStruct()
                for k, v in val.items():
                    s._fields[k] = v
                return s
            return val
        raise TypeError(f"Field access on {type(target).__name__}")

    def _eval_dynamic_field(self, node, ws: Workspace):
        """Handle DynamicFieldAccess nodes."""
        target = self._eval_expr(node.target, ws)
        field = self._eval_expr(node.field_expr, ws)
        if isinstance(field, ForgeChar):
            field = field.to_str()
        if isinstance(target, ForgeObject):
            return getattr(target, str(field))
        return target._fields[str(field)]

    def _eval_function_handle(self, node, ws: Workspace):
        """Handle FunctionHandle nodes."""
        if node.name in self.functions:
            fn = self.functions[node.name]
            if isinstance(fn, FunctionDef):
                result = lambda *a: self._call_function(fn, list(a), ws)
                result._forge_name = node.name
                return result
            if not hasattr(fn, '_forge_name'):
                fn._forge_name = node.name
            return fn
        raise NameError(f"Undefined function: {node.name}")

    def _eval_anon_function(self, node, ws: Workspace):
        """Handle AnonFunction nodes."""
        params = node.args
        body = node.body
        captured_ws = ws
        def anon(*args):
            local_ws = Workspace(captured_ws)
            for p, a in zip(params, args):
                local_ws.set(p, a)
            return self._eval_expr(body, local_ws)
        return anon

    def _eval_end_keyword(self, node, ws: Workspace):
        """Handle EndKeyword nodes."""
        if self._index_sizes:
            return ForgeArray(np.array(float(self._index_sizes[-1])))
        raise RuntimeError("'end' used outside of indexing context")

    def _eval_number(self, node, ws: Workspace = None) -> ForgeArray:
        """Evaluate NumberLiteral. Uses cache for common constants."""
        v = node.value
        cached = self._number_cache.get(v)
        if cached is not None:
            return cached
        result = self._parse_number(v)
        # Cache small integer-like constants (very common in loops/indexing)
        if len(self._number_cache) < 256:
            self._number_cache[v] = result
        return result

    @staticmethod
    def _parse_number(v: str) -> ForgeArray:
        """Parse a number string into a ForgeArray (no caching)."""
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
        _any_sparse = _sp.issparse(l) or _sp.issparse(r)
        def _wrap_sparse(result):
            if _sp.issparse(result):
                return result  # keep sparse as raw scipy
            return ForgeArray(result)
        if op == "+":
            if _any_sparse:
                return _wrap_sparse(l + r)
            if can_fuse(op, l, r):
                return ForgeArray._from_ndarray(fused_binop(op, l, r))
            result = l + r
            return ForgeArray._from_ndarray(result) if result.ndim >= 2 else ForgeArray(result)
        if op == "-":
            if _any_sparse:
                return _wrap_sparse(l - r)
            if can_fuse(op, l, r):
                return ForgeArray._from_ndarray(fused_binop(op, l, r))
            result = l - r
            return ForgeArray._from_ndarray(result) if result.ndim >= 2 else ForgeArray(result)
        if op == "*":
            if _any_sparse:
                if _is_matrix_op(l, r):
                    return _wrap_sparse(l @ r)
                else:
                    if _sp.issparse(l):
                        return _wrap_sparse(l.multiply(r))
                    else:
                        return _wrap_sparse(r.multiply(l))
            try:
                return ForgeArray(l @ r if _is_matrix_op(l, r) else l * r)
            except ValueError:
                if _is_matrix_op(l, r):
                    raise ValueError(f"operator *: inner matrix dimensions must agree ({l.shape[0]}x{l.shape[1]} * {r.shape[0]}x{r.shape[1]})")
                raise
        if op == "/":
            if _any_sparse and _is_matrix_op(l, r):
                from scipy.sparse.linalg import spsolve
                rt = r.T.tocsc() if _sp.issparse(r) else _sp.csc_matrix(r.T)
                lt = l.T.toarray().ravel() if _sp.issparse(l) else l.T.ravel() if l.T.ndim > 1 and l.T.shape[1] == 1 else l.T
                x = spsolve(rt, lt)
                return ForgeArray(np.atleast_2d(x).T if x.ndim == 1 else x.T)
            return ForgeArray(_smart_mldivide(r.T, l.T).T if _is_matrix_op(l, r) else l / r)
        if op == "\\":
            if _any_sparse and _is_matrix_op(l, r):
                from scipy.sparse.linalg import spsolve
                l_sp = l.tocsc() if _sp.issparse(l) else _sp.csc_matrix(l)
                r_arr = r.toarray() if _sp.issparse(r) else r
                if r_arr.ndim == 2 and r_arr.shape[1] == 1:
                    x = spsolve(l_sp, r_arr.ravel())
                    return ForgeArray(np.atleast_2d(x).T)
                x = spsolve(l_sp, r_arr)
                return ForgeArray(np.atleast_2d(x).T if x.ndim == 1 else x)
            return ForgeArray(_smart_mldivide(l, r) if _is_matrix_op(l, r) else r / l)
        if op == "^":
            return ForgeArray(np.linalg.matrix_power(l, int(r.flat[0])) if (l.ndim == 2 and l.size > 1) else l ** r)
        if op == ".*":
            if _any_sparse:
                if _sp.issparse(l):
                    return _wrap_sparse(l.multiply(r))
                else:
                    return _wrap_sparse(r.multiply(l))
            if can_fuse(op, l, r):
                return ForgeArray._from_ndarray(fused_binop(op, l, r))
            result = l * r
            return ForgeArray._from_ndarray(result) if type(result) is np.ndarray and result.ndim >= 2 else ForgeArray(result)
        if op == "./":
            if can_fuse(op, l, r):
                return ForgeArray._from_ndarray(fused_binop(op, l, r))
            result = l / r
            return ForgeArray._from_ndarray(result) if type(result) is np.ndarray and result.ndim >= 2 else ForgeArray(result)
        if op == ".\\":
            if can_fuse(op, l, r):
                return ForgeArray._from_ndarray(fused_binop(op, l, r))
            result = r / l
            return ForgeArray._from_ndarray(result) if type(result) is np.ndarray and result.ndim >= 2 else ForgeArray(result)
        if op == ".^":
            if can_fuse(op, l, r):
                return ForgeArray._from_ndarray(fused_binop(op, l, r))
            result = l ** r
            return ForgeArray._from_ndarray(result) if type(result) is np.ndarray and result.ndim >= 2 else ForgeArray(result)
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
            # Auto-call zero-arg callables used as function arguments
            # (e.g., datestr(now) -- now resolves to callable, should be called)
            args = [v() if (callable(v) and not isinstance(v, (ForgeArray, type))
                           and isinstance(a, Identifier) and not ws.has(a.name))
                    else v
                    for v, a in zip(args, node.args)]
            result = target(*args)
            # Single-output context: if function returns a tuple, take first element
            # (multi-output is handled by _eval_multi_output for [a,b]=func() syntax)
            if isinstance(result, tuple) and len(result) > 0:
                result = result[0]
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
                _check_1based_index(idx, len(target._data))
                return target._data[idx - 1]  # 1-based
            raise TypeError("Cannot index ForgeCell with multiple indices via ()")

        # Step 3.6: If it's a Map, do key-based indexing
        if isinstance(target, ForgeMap):
            args = [self._eval_expr(a, ws) for a in node.args]
            if len(args) == 1:
                key = args[0]
                if isinstance(key, ForgeChar):
                    key = key.to_str()
                elif isinstance(key, ForgeArray):
                    key = str(_to_py(key))
                return target[key]
            raise TypeError("containers.Map supports single-key indexing only")

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
                        if np.any(raw_idx.astype(int).ravel() <= 0):
                            bad = int(raw_idx.astype(int).ravel()[raw_idx.astype(int).ravel() <= 0][0])
                            _check_1based_index(bad, len(s))
                        chars = [s[i] for i in indices if 0 <= i < len(s)]
                        return ForgeChar(''.join(chars))
                _check_1based_index(int(_to_py(idx)), len(s))
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
                    if np.any(raw_idx.astype(int).ravel() <= 0):
                        bad = int(raw_idx.astype(int).ravel()[raw_idx.astype(int).ravel() <= 0][0])
                        _check_1based_index(bad, len(flat))
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
            _check_1based_index(idx_int, len(data.ravel(order=F)))
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
                    if np.any(raw.astype(int).ravel() <= 0):
                        bad = int(raw.astype(int).ravel()[raw.astype(int).ravel() <= 0][0])
                        _check_1based_index(bad)
                    slices.append(raw.astype(int).ravel() - 1)  # 1-based to 0-based
            else:
                idx_int = int(_to_py(arg))
                _check_1based_index(idx_int, data.shape[dim] if dim < len(data.shape) else 1)
                slices.append(idx_int - 1)  # 1-based to 0-based

        try:
            result = data[np.ix_(*[s if isinstance(s, np.ndarray) else (np.arange(data.shape[i]) if isinstance(s, slice) else np.array([s])) for i, s in enumerate(slices)])]
        except IndexError as e:
            if "too many indices" in str(e):
                raise IndexError(f"index exceeds array dimensions: array is {data.ndim}-dimensional, but {len(slices)} indices were given")
            raise
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
            row_raw = []
            for e in row:
                val = self._eval_expr(e, ws)
                # Expand tuples from cell {:} expansion
                if isinstance(val, tuple):
                    row_raw.extend(val)
                else:
                    row_raw.append(val)
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
            try:
                rows.append(np.concatenate(vals, axis=1))
            except ValueError:
                raise ValueError("horizontal dimensions mismatch")
        if not rows:
            return ForgeArray(np.array([]).reshape(0, 0))
        try:
            return ForgeArray(np.concatenate(rows, axis=0))
        except ValueError:
            sizes = " vs ".join(str(r.shape[1]) for r in rows)
            raise ValueError(f"vertical dimensions mismatch ({sizes} columns)")

    def _eval_cell_literal(self, node: CellLiteral, ws: Workspace) -> ForgeCell:
        elements = []
        for row in node.rows:
            for e in row:
                elements.append(self._eval_expr(e, ws))
        cell = ForgeCell(elements)
        # Preserve 2-D shape from parsed rows
        nrows = len(node.rows)
        if nrows > 1:
            ncols = len(node.rows[0])
            cell._shape = (nrows, ncols)
        return cell

    # ============================================================
    # Statement execution
    # ============================================================

    def _exec_assign(self, node: Assignment, ws: Workspace) -> Any:
        target = node.targets

        if isinstance(target, list):
            nargout = len(target)
            # Try nargout-aware multi-output path FIRST to avoid computing
            # unneeded outputs (e.g. unique with nargout=1 skips indices).
            value = self._eval_multi_output(node.value, ws, nargout)
            if value is None:
                value = self._eval_expr(node.value, ws)
                if callable(value) and not isinstance(value, ForgeArray):
                    if isinstance(node.value, Identifier):
                        value = value()
            if isinstance(value, tuple):
                for t, v in zip(target, value):
                    if t.name != "~":  # ~ = discard output
                        ws.set(t.name, v)
            else:
                if target[0].name != "~":
                    ws.set(target[0].name, value)
            return value

        value = self._eval_expr(node.value, ws)
        # Auto-call bare callable identifiers on RHS (e.g., t = toc)
        if callable(value) and not isinstance(value, ForgeArray):
            if isinstance(node.value, Identifier):
                value = value()

        if isinstance(target, Identifier):
            # Value semantics: copy ForgeArray to prevent aliasing (B = A)
            # Skip copy if RHS is an expression (function call, binop, etc.)
            # — those always produce new allocations. Only copy for bare identifiers.
            if isinstance(value, ForgeArray) and not isinstance(value, ForgeChar):
                if isinstance(node.value, Identifier):
                    value = value.copy()
            ws.set(target.name, value)
            return value

        if isinstance(target, Index):
            target_name = target.target.name if isinstance(target.target, Identifier) else None
            try:
                arr = self._eval_expr(target.target, ws)
            except (NameError, KeyError):
                # Octave auto-creates arrays on indexed assignment to undefined vars
                if target_name is not None:
                    arr = ForgeArray(np.zeros(0))
                    ws.set(target_name, arr)
                else:
                    raise
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
                # --- Deletion: A(idx) = [] or A(row,:) = [] ---
                _is_empty_rhs = (isinstance(assign_val, np.ndarray) and assign_val.size == 0)
                if _is_empty_rhs:
                    data = arr.data
                    if len(args) == 1:
                        # A([2 4]) = [] -> delete elements, result is row vector
                        idx = args[0]
                        if isinstance(idx, ForgeArray):
                            raw_idx = _unwrap(idx).astype(int).ravel() - 1
                        elif idx is None:
                            # A(:) = [] -> delete all
                            raw_idx = np.arange(data.size)
                        else:
                            raw_idx = np.array([int(_to_py(idx)) - 1])
                        flat = data.ravel(order='F')
                        new_flat = np.delete(flat, raw_idx)
                        if new_flat.size == 0:
                            arr._data = np.zeros((0, 0))
                        else:
                            arr._data = new_flat.reshape(1, -1)
                        if target_name:
                            ws.set(target_name, arr)
                        return value
                    elif len(args) == 2:
                        # A(2,:) = [] -> delete row(s); A(:,2) = [] -> delete col(s)
                        row_arg, col_arg = args
                        if col_arg is None and row_arg is not None:
                            # A(idx,:) = [] -> delete rows
                            if isinstance(row_arg, ForgeArray):
                                rows_to_del = _unwrap(row_arg).astype(int).ravel() - 1
                            else:
                                rows_to_del = np.array([int(_to_py(row_arg)) - 1])
                            arr._data = np.delete(data, rows_to_del, axis=0)
                            if target_name:
                                ws.set(target_name, arr)
                            return value
                        elif row_arg is None and col_arg is not None:
                            # A(:,idx) = [] -> delete columns
                            if isinstance(col_arg, ForgeArray):
                                cols_to_del = _unwrap(col_arg).astype(int).ravel() - 1
                            else:
                                cols_to_del = np.array([int(_to_py(col_arg)) - 1])
                            arr._data = np.delete(data, cols_to_del, axis=1)
                            if target_name:
                                ws.set(target_name, arr)
                            return value
                        else:
                            raise ValueError("A null assignment can have only one non-colon index")
                # --- End deletion ---
                data = arr.data
                if len(args) == 1:
                    idx = args[0]
                    # --- Auto-grow: expand array if index exceeds current size ---
                    def _max_index(idx_val):
                        """Get the maximum 1-based index from an index expression."""
                        if isinstance(idx_val, ForgeArray):
                            raw = _unwrap(idx_val)
                            if raw.dtype == np.bool_:
                                return None  # boolean indexing doesn't grow
                            return int(np.max(raw))
                        elif idx_val is None:
                            return None  # colon indexing
                        else:
                            return int(_to_py(idx_val))
                    max_idx = _max_index(idx)
                    # Check for zero/negative indices before auto-grow
                    if max_idx is not None and max_idx <= 0:
                        _check_1based_index(max_idx, data.size)
                    if isinstance(idx, ForgeArray):
                        raw_check = _unwrap(idx)
                        if raw_check.dtype != np.bool_ and raw_check.size > 0:
                            min_val = int(np.min(raw_check))
                            if min_val <= 0:
                                _check_1based_index(min_val, data.size)
                    if max_idx is not None and max_idx > data.size:
                        # Octave auto-extends with zeros
                        new_data = np.zeros(max_idx, dtype=data.dtype)
                        new_data[:data.size] = data.ravel(order='F')
                        data = new_data.reshape(1, -1) if data.ndim <= 1 or data.shape[0] <= 1 else new_data.reshape(-1, 1)
                        arr._data = data
                        if target_name:
                            ws.set(target_name, arr)
                    # --- end auto-grow ---
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
                            if np.any(raw_idx.astype(int).ravel() <= 0):
                                bad = int(raw_idx.astype(int).ravel()[raw_idx.astype(int).ravel() <= 0][0])
                                _check_1based_index(bad)
                            int_idx = raw_idx.astype(int).ravel() - 1
                            if np.isscalar(assign_val):
                                flat[int_idx] = assign_val
                            else:
                                flat[int_idx] = np.asarray(assign_val).ravel()
                            data[:] = flat.reshape(data.shape, order='F')
                    else:
                        _check_1based_index(int(_to_py(idx)), len(data.ravel(order=F)))
                        idx_int = int(_to_py(idx)) - 1
                        flat = data.ravel(order='F')
                        flat[idx_int] = assign_val
                        data[:] = flat.reshape(data.shape, order='F')
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
                            _check_1based_index(int(_to_py(arg)))
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
            elif isinstance(arr, ForgeMap):
                # containers.Map indexed assignment: m("key") = value
                if len(args) == 1:
                    key = args[0]
                    if isinstance(key, ForgeChar):
                        key = key.to_str()
                    elif isinstance(key, ForgeArray):
                        key = str(_to_py(key))
                    arr[key] = value
                else:
                    raise TypeError("containers.Map supports single-key indexing only")
            elif isinstance(arr, ForgeCell):
                # Cell () assignment: c(i) = value (wraps in cell)
                if len(args) == 1:
                    idx = int(_to_py(args[0]))
                    arr.content_set(value, idx)
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
            # Octave compat: s.x = 1 when s is a non-struct replaces s with a struct
            if isinstance(target.target, Identifier) and ws.has(target.target.name):
                existing = ws.get(target.target.name)
                if not isinstance(existing, ForgeStruct):
                    if not isinstance(existing, ForgeObject):
                        new_struct = ForgeStruct()
                        new_struct._fields[target.field] = value
                        ws.set(target.target.name, new_struct)
                        return value
            # Auto-create nested structs: s.a.b.c = val
            if isinstance(target.target, FieldAccess):
                self._ensure_nested_struct(target.target, ws)
            obj = self._eval_expr(target.target, ws)
            if isinstance(obj, ForgeStruct):
                obj._fields[target.field] = value
                return value
            if isinstance(obj, ForgeTable):
                obj.set_column(target.field, value)
                return value
            if isinstance(obj, ForgeStruct):
                obj._fields[target.field] = value
            else:
                if isinstance(obj, ForgeObject):
                    obj.set(target.field, value)
            return value

        if isinstance(target, CellIndex):
            obj = self._eval_expr(target.target, ws)
            if isinstance(obj, ForgeCell):
                self._index_sizes.append(obj.numel())
                try:
                    args = [_to_py(self._eval_expr(a, ws)) for a in target.args]
                finally:
                    self._index_sizes.pop()
                obj.content_set(value, *args)
            return value

        raise RuntimeError(f"Cannot assign to {type(target).__name__}")

    def _ensure_nested_struct(self, node, ws):
        """Ensure intermediate structs exist for nested field assignment."""
        if isinstance(node, FieldAccess):
            if isinstance(node.target, Identifier):
                if not ws.has(node.target.name):
                    ws.set(node.target.name, ForgeStruct())
                obj = ws.get(node.target.name)
                if isinstance(obj, ForgeStruct) and node.field not in obj._fields:
                    obj._fields[node.field] = ForgeStruct()
            elif isinstance(node.target, FieldAccess):
                self._ensure_nested_struct(node.target, ws)
                parent = self._eval_expr(node.target, ws)
                if isinstance(parent, ForgeStruct) and node.field not in parent._fields:
                    parent._fields[node.field] = ForgeStruct()

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
        # JIT acceleration: compile eligible numeric for-loops via Numba
        if is_numba_available() and can_jit_loop(node):
            def _ws_get(name):
                return ws.get(name) if ws.has(name) else None
            def _ws_set(name, val):
                ws.set(name, val)
            if compile_and_run_loop(node, _ws_get, _ws_set):
                return None
        iter_val = self._eval_expr(node.iter_expr, ws)
        # R09: Handle cell array iteration: for i = {1,2,3}, each element
        if isinstance(iter_val, ForgeCell):
            for idx in range(iter_val.numel()):
                ws.set(node.var, iter_val._data[idx])
                try:
                    self._exec_stmts(node.body, ws)
                except BreakSignal:
                    break
                except ContinueSignal:
                    continue
            return None
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
            # Cell array case: match if ANY element equals the switch value
            if isinstance(case_val, ForgeCell):
                matched = False
                for elem in case_val._data:
                    if _values_equal(val, elem):
                        matched = True
                        break
                if matched:
                    return self._exec_stmts(body, ws)
            elif _values_equal(val, case_val):
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
                    err_struct._fields["identifier"] = ForgeChar(getattr(e, "identifier", ""))
                    err_struct._fields["message"] = ForgeChar(str(e))
                    ws.set(node.catch_var, err_struct)
                return self._exec_stmts(node.catch_body, ws)
            return None

    def _exec_classdef(self, node: ClassDef, ws: Workspace):
        """Register a classdef as a constructable class."""
        # Build properties
        props = {}
        for name, default_expr in node.properties.items():
            default_val = self._eval_expr(default_expr, ws) if default_expr is not None else None
            props[name] = Property(name=name, default_value=default_val)
        # Build methods
        methods = {}
        _cls_name = node.name
        for func_def in node.methods:
            def make_method(fd, _cn=_cls_name):
                def method_impl(obj, *args):
                    # Create local workspace
                    local_ws = Workspace()
                    local_ws._global_store = self._global_store
                    # Copy constants from base workspace
                    _CONSTS = {"pi", "e", "eps", "Inf", "inf", "NaN", "nan",
                               "realmin", "realmax", "i", "j", "true", "false"}
                    for _cname in _CONSTS:
                        if self.workspace.has(_cname):
                            local_ws.set(_cname, self.workspace.get(_cname))
                    # Set caller context for access control
                    if isinstance(obj, ForgeObject):
                        obj._caller_context = 'internal'
                    is_ctor = (fd.name == _cn)
                    if is_ctor:
                        # Constructor: bind return var to ForgeObject
                        ret_name = fd.returns[0] if fd.returns else None
                        obj_in_params = ret_name and ret_name in fd.params
                        if ret_name:
                            local_ws.set(ret_name, obj)
                        if obj_in_params:
                            param_list = [p for p in fd.params if p != ret_name]
                            for i, param in enumerate(param_list):
                                if i < len(args):
                                    local_ws.set(param, args[i])
                        else:
                            for i, param in enumerate(fd.params):
                                if i < len(args):
                                    local_ws.set(param, args[i])
                    else:
                        # Regular method: first param is self/obj
                        # Value semantics: copy obj so original is not mutated
                        if not obj._class.is_handle:
                            obj = obj.copy()
                        if fd.params:
                            local_ws.set(fd.params[0], obj)
                        for i, param in enumerate(fd.params[1:]):
                            if i < len(args):
                                local_ws.set(param, args[i])
                    # nargin: for constructors, count user args only (not obj); for methods, count obj + args
                    _nargin = len(args) if is_ctor else len(args) + 1
                    local_ws.set("nargin", ForgeArray(np.array(float(_nargin))))
                    try:
                        self._exec_stmts(fd.body, local_ws)
                    except ReturnSignal:
                        pass
                    finally:
                        if isinstance(obj, ForgeObject):
                            obj._caller_context = None
                    if fd.returns:
                        ret_val = local_ws.get(fd.returns[0])
                        # For constructors: if return var was overwritten with non-object, use original
                        if is_ctor and isinstance(obj, ForgeObject) and not isinstance(ret_val, ForgeObject):
                            return obj
                        return ret_val
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
        local_ws._global_store = self._global_store  # share global store
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
        prev_func = self._current_function
        prev_ws = getattr(self, "_current_workspace", None)
        self._current_function = funcdef.name
        self._current_workspace = local_ws
        try:
            self._exec_stmts(funcdef.body, local_ws)
        except ReturnSignal:
            pass
        finally:
            # Save persistent variables back to store
            if hasattr(local_ws, '_persistent_vars') and local_ws._persistent_vars:
                store = self._persistent_store.setdefault(funcdef.name, {})
                for pvar in local_ws._persistent_vars:
                    if pvar in local_ws._vars:
                        store[pvar] = local_ws._vars[pvar]
            self._current_function = prev_func
            self._current_workspace = prev_ws
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


    _MULTI_OUTPUT_FUNCS = frozenset({
        "max", "min", "find", "size", "ind2sub", "sort", "eig", "svd",
        "lu", "qr", "chol", "unique", "meshgrid", "fileparts", "ismember",
        "deal", "cellfun", "arrayfun",
    })

    def _eval_multi_output(self, expr, ws, nargout):
        if not isinstance(expr, Index):
            return None
        name = expr.target.name if isinstance(expr.target, Identifier) else None
        if name is None:
            return None
        # Fast bail-out for functions not known to have multi-output paths
        if name not in self._MULTI_OUTPUT_FUNCS and name not in self.functions:
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
            x = _unwrap(args[0]) if isinstance(args[0], ForgeArray) else args[0]
            if nargout >= 2:
                if _sp.issparse(x):
                    coo = _sp.coo_matrix(x)
                    rows = coo.row.astype(float) + 1
                    cols = coo.col.astype(float) + 1
                    vals = coo.data.astype(float)
                    order = np.lexsort((coo.row, coo.col))
                    rows, cols, vals = rows[order], cols[order], vals[order]
                    if nargout >= 3:
                        return (ForgeArray(rows), ForgeArray(cols), ForgeArray(vals))
                    return (ForgeArray(rows), ForgeArray(cols))
                else:
                    if hasattr(x, 'ndim') and x.ndim < 2:
                        x = x.reshape(1, -1)
                    rows, cols = np.nonzero(x)
                    if nargout >= 3:
                        vals = x[rows, cols].astype(float)
                        return (ForgeArray(rows + 1), ForgeArray(cols + 1), ForgeArray(vals))
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
                if nargout <= 1:
                    vals = scipy_eig(x, b, right=False)
                    return ForgeArray(np.diag(vals))
                vals, vecs = scipy_eig(x, b)
            else:
                if nargout <= 1:
                    vals = np.linalg.eigvals(x)
                    return ForgeArray(np.diag(vals))
                vals, vecs = np.linalg.eig(x)
            return (ForgeArray(vecs), ForgeArray(np.diag(vals)))
        if name == 'svd':
            x = _unwrap(args[0])
            if nargout <= 1:
                s = np.linalg.svd(x, compute_uv=False)
                return ForgeArray(s)
            U, s, Vt = np.linalg.svd(x, full_matrices=True)
            m, n = x.shape
            k = min(m, n)
            S = np.zeros((m, n))
            S[:k, :k] = np.diag(s)
            if nargout >= 3:
                return (ForgeArray(U), ForgeArray(S), ForgeArray(Vt.T))
            return (ForgeArray(U), ForgeArray(S))
        if name == 'lu':
            x = _unwrap(args[0])
            from scipy.linalg import lu as scipy_lu
            P, L, U = scipy_lu(x.copy(), overwrite_a=True, check_finite=False)
            if nargout >= 3:
                return (ForgeArray(L), ForgeArray(U), ForgeArray(P))
            return (ForgeArray(L), ForgeArray(U))
        if name == 'qr':
            x = _unwrap(args[0])
            Q, R = np.linalg.qr(x)
            return (ForgeArray(Q), ForgeArray(R))
        if name == 'chol':
            x = _unwrap(args[0])
            R = np.linalg.cholesky(x).T  # Octave returns upper triangular
            if nargout >= 2:
                # [R, p] = chol(A) returns 0 if successful
                return (ForgeArray(R), ForgeArray(np.array(0.0)))
            return ForgeArray(R)
        if name == 'unique':
            raw = _unwrap(args[0])
            x = raw.ravel() if raw.ndim != 1 else raw
            if nargout >= 3:
                vals, i_first, i_inverse = np.unique(x, return_index=True, return_inverse=True)
                return (ForgeArray(vals), ForgeArray((i_first + 1).astype(float)), ForgeArray((i_inverse + 1).astype(float)))
            if nargout == 2:
                vals, i_first = np.unique(x, return_index=True)
                return (ForgeArray(vals), ForgeArray((i_first + 1).astype(float)))
            # nargout==1: smart dispatch for best performance
            # Boolean mask is 2x faster for bounded integer data with low cardinality
            if np.issubdtype(x.dtype, np.integer) or (np.issubdtype(x.dtype, np.floating) and x.size > 10000):
                xmin, xmax = x.min(), x.max()
                range_size = xmax - xmin + 1
                if range_size > 0 and range_size < x.size * 0.5 and range_size < 10_000_000:
                    # Boolean mask approach: O(n + range) vs O(n log n)
                    ints = (x - xmin).astype(np.int64)
                    seen = np.zeros(int(range_size), dtype=np.bool_)
                    seen[ints] = True
                    return ForgeArray(np.nonzero(seen)[0].astype(float) + xmin)
            return ForgeArray(np.unique(x))
        if name == 'meshgrid':
            if len(args) >= 2:
                X, Y = np.meshgrid(_unwrap(args[0]).ravel(), _unwrap(args[1]).ravel())
                if nargout >= 3 and len(args) >= 3:
                    X, Y, Z = np.meshgrid(_unwrap(args[0]).ravel(), _unwrap(args[1]).ravel(), _unwrap(args[2]).ravel())
                    return (ForgeArray(X), ForgeArray(Y), ForgeArray(Z))
                return (ForgeArray(X), ForgeArray(Y))
            return None
        if name == 'fileparts':
            import os
            fname = args[0].to_str() if isinstance(args[0], ForgeChar) else str(args[0])
            d = os.path.dirname(fname)
            base = os.path.basename(fname)
            name_only, ext = os.path.splitext(base)
            return (ForgeChar(d), ForgeChar(name_only), ForgeChar(ext))
        if name == 'ismember':
            result = forge_ismember(*args)
            if isinstance(result, tuple) and nargout >= 2:
                return result[:nargout]
            if isinstance(result, tuple):
                return result[0]
            return result
        if name == 'deal':
            if len(args) == 1:
                return tuple(args[0] for _ in range(nargout))
            else:
                return tuple(args[:nargout])
        # Generic: if function returns a tuple, use it directly
        if name in self.functions:
            func = self.functions[name]
            if isinstance(func, FunctionDef):
                return self._call_function(func, args, ws, nargout=nargout)
            if callable(func):
                result = func(*args)
                if isinstance(result, tuple):
                    return result
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
        d = val._data
        if d.size == 1:
            return d.flat[0].item()
        return d
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, ForgeChar):
        return val.to_str()
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    return val



def _check_1based_index(idx_val, dim_size=None):
    """Validate that a 1-based index is positive (Octave/MATLAB convention)."""
    if idx_val <= 0:
        if dim_size is not None:
            raise IndexError(
                f"index ({idx_val}): out of bound 0; value must be in the range [1, {dim_size}]"
            )
        raise IndexError(
            f"index ({idx_val}): out of bound 0; value must be positive"
        )


def _to_int(val):
    """Convert to Python int (for indexing)."""
    if isinstance(val, int):
        return val
    if isinstance(val, ForgeArray):
        return int(val._data.flat[0])
    if isinstance(val, float):
        return int(val)
    if isinstance(val, (np.integer, np.floating)):
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
    if _sp.issparse(a) or _sp.issparse(b):
        sa = a.shape if hasattr(a, "shape") else np.asarray(a).shape
        sb = b.shape if hasattr(b, "shape") else np.asarray(b).shape
        if len(sa) != 2 or len(sb) != 2:
            return False
        if (sa[0] * sa[1]) == 1 or (sb[0] * sb[1]) == 1:
            return False
        return True
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
