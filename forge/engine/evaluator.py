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
    forge_islogical, forge_isfloat, forge_isinteger,
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
    GlobalStatement, PersistentStatement, parse,
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
        b["isnan"] = forge_isnan
        b["isinf"] = forge_isinf
        b["isfinite"] = forge_isfinite
        b["isempty"] = lambda x: ForgeArray(np.array(x.isempty() if isinstance(x, ForgeArray) else len(x) == 0))

        # Math
        for name in ['abs','sqrt','exp','log','log2','log10','sin','cos','tan',
                      'asin','acos','atan','sinh','cosh','tanh','ceil','floor',
                      'round','fix','sign','real','imag','conj','angle']:
            fn = getattr(np, name)
            b[name] = lambda *a, f=fn: ForgeArray(f(_unwrap(a[0])))

        b["mod"] = lambda a, m: ForgeArray(np.mod(_unwrap(a), _unwrap(m)))
        b["rem"] = lambda a, m: ForgeArray(np.remainder(_unwrap(a), _unwrap(m)))
        b["atan2"] = lambda y, x: ForgeArray(np.arctan2(_unwrap(y), _unwrap(x)))
        b["hypot"] = lambda x, y: ForgeArray(np.hypot(_unwrap(x), _unwrap(y)))
        b["power"] = lambda x, y: ForgeArray(np.power(_unwrap(x), _unwrap(y)))

        # Array ops
        b["size"] = self._builtin_size
        b["length"] = lambda x: ForgeArray(np.array(x.length() if isinstance(x, ForgeArray) else len(x)))
        b["numel"] = lambda x: ForgeArray(np.array(x.numel() if isinstance(x, ForgeArray) else len(x)))
        b["ndims"] = lambda x: ForgeArray(np.array(x.ndim if isinstance(x, ForgeArray) else 0))
        b["reshape"] = lambda x, *a: ForgeArray(_unwrap(x).reshape(*[int(_to_py(v)) for v in a]))
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
        b["clc"] = lambda: None  # Clear command window (no-op in engine)

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
        b["class"] = lambda x: ForgeChar(x.type_name() if isinstance(x, ForgeArray) else type(x).__name__)
        b["typecast"] = lambda x, t: x.astype(t.to_str() if isinstance(t, ForgeChar) else str(t))

        # Merge toolbox registries (elfun, general, specfun, ...)
        b.update(BUILTIN_REGISTRY)

    def _builtin_size(self, x, *args):
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
        raw_vals = [_to_py(a) for a in args[1:]]
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
            # All scalar args
            vals = tuple(float(v) if isinstance(v, (_np.floating, _np.integer)) else v for v in raw_vals)
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
        vals = tuple(_to_py(a) for a in args[1:])
        return ForgeChar(fmt % vals)

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

    def _builtin_warning(self, *args):
        msg = args[0]
        if isinstance(msg, ForgeChar):
            msg = msg.to_str()
        self.output_buffer.write(f"warning: {msg}\n")

    def _resolve_m_file(self, name):
        """Search session path for {name}.m, parse and register it."""
        if self._session_ref is None:
            return None
        for directory in self._session_ref.path:
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
            raise TypeError(f"Field access on {type(target).__name__}")

        if isinstance(node, DynamicFieldAccess):
            target = self._eval_expr(node.target, ws)
            field = self._eval_expr(node.field_expr, ws)
            if isinstance(field, ForgeChar):
                field = field.to_str()
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
        rows = []
        for row in node.rows:
            vals = [_unwrap(self._eval_expr(e, ws)) for e in row]
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
        target = node.targets

        if isinstance(target, list):
            nargout = len(target)
            if not isinstance(value, tuple) and nargout > 1:
                val_tuple = self._eval_multi_output(node.value, ws, nargout)
                if val_tuple is not None and isinstance(val_tuple, tuple):
                    value = val_tuple
            if isinstance(value, tuple):
                for t, v in zip(target, value):
                    ws.set(t.name, v)
            else:
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
                            data.ravel(order='F')[raw_idx.ravel()] = assign_val
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
                elif len(args) == 2:
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
                            idx_arrays.append(np.arange(data.shape[i]))
                        elif isinstance(s, np.ndarray):
                            idx_arrays.append(s)
                        else:
                            idx_arrays.append(np.array([s]))
                    ix = np.ix_(*idx_arrays)
                    data[ix] = assign_val
            return value

        if isinstance(target, FieldAccess):
            if isinstance(target.target, Identifier) and not ws.has(target.target.name):
                new_struct = ForgeStruct()
                new_struct._fields[target.field] = value
                ws.set(target.target.name, new_struct)
                return value
            obj = self._eval_expr(target.target, ws)
            if isinstance(obj, ForgeStruct):
                obj._fields[target.field] = value
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

    def _call_function(self, funcdef: FunctionDef, args: list, caller_ws: Workspace) -> Any:
        """Call a user-defined function."""
        local_ws = Workspace()
        # Set nargin/nargout
        local_ws.set("nargin", ForgeArray(np.array(float(len(args)))))
        local_ws.set("nargout", ForgeArray(np.array(float(len(funcdef.returns)))))
        # Bind parameters
        for i, param in enumerate(funcdef.params):
            if i < len(args):
                local_ws.set(param, args[i])
        # Initialize return variables
        for ret in funcdef.returns:
            if not local_ws.has(ret):
                local_ws.set(ret, ForgeArray(0.0))
        # Execute body
        try:
            self._exec_stmts(funcdef.body, local_ws)
        except ReturnSignal:
            pass
        # Collect return values
        if len(funcdef.returns) == 0:
            return None
        if len(funcdef.returns) == 1:
            return local_ws.get(funcdef.returns[0])
        return tuple(local_ws.get(r) for r in funcdef.returns)


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
