# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""M-language evaluator: AST → execution.

Walks the AST produced by the parser, evaluating expressions and executing statements
against a workspace (variable scope) with access to built-in functions.
"""
import numpy as np
import sys
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
    EndKeyword, Assignment, IfStatement, ForStatement, WhileStatement,
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
        self._setup_builtins()
        self._setup_constants()

    def _setup_constants(self):
        self.workspace.set("pi", ForgeArray(FORGE_PI))
        self.workspace.set("e", ForgeArray(FORGE_E))
        self.workspace.set("Inf", ForgeArray(FORGE_INF))
        self.workspace.set("inf", ForgeArray(FORGE_INF))
        self.workspace.set("NaN", ForgeArray(FORGE_NAN))
        self.workspace.set("nan", ForgeArray(FORGE_NAN))
        self.workspace.set("eps", ForgeArray(FORGE_EPS))
        self.workspace.set("i", ForgeArray(FORGE_I))
        self.workspace.set("j", ForgeArray(FORGE_I))
        self.workspace.set("true", ForgeArray(np.array(True)))
        self.workspace.set("false", ForgeArray(np.array(False)))

    def _setup_builtins(self):
        """Register built-in functions."""
        b = self.functions

        # Matrix construction
        b["eye"] = lambda *a: forge_eye(*[_to_py(x) for x in a])
        b["ones"] = lambda *a: forge_ones(*[_to_py(x) for x in a])
        b["zeros"] = lambda *a: forge_zeros(*[_to_py(x) for x in a])
        b["rand"] = lambda *a: forge_rand(*[_to_py(x) for x in a])
        b["randn"] = lambda *a: forge_randn(*[_to_py(x) for x in a])
        b["randi"] = lambda *a: forge_randi(*[_to_py(x) for x in a])
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
        b["reshape"] = lambda x, *a: ForgeArray(_unwrap(x).reshape(*[_to_py(v) for v in a]))
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
        b["horzcat"] = lambda *a: ForgeArray(np.concatenate([_unwrap(x) for x in a], axis=1))
        b["vertcat"] = lambda *a: ForgeArray(np.concatenate([_unwrap(x) for x in a], axis=0))
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

        # Struct/cell
        b["struct"] = lambda *a: forge_struct(*a)
        b["fieldnames"] = forge_fieldnames
        b["cell"] = lambda *a: forge_cell(*[_to_py(x) for x in a])

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
            return ForgeArray(np.array(x.shape[dim-1] if dim <= x.ndim else 1))
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
        vals = tuple(_to_py(a) for a in args[1:])
        self.output_buffer.write(fmt % vals)

    def _builtin_sprintf(self, *args):
        fmt = args[0]
        if isinstance(fmt, ForgeChar):
            fmt = fmt.to_str()
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
            if node.print_result and val is not None:
                ws.set("ans", val)
                self.ans = val
            return val

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
                # Return the function/callable — Index node handles actual calling
                return self.functions[name]
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
            return None  # Handled specially in indexing context

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
        if op == "\\": return ForgeArray(np.linalg.solve(l, r) if _is_matrix_op(l, r) else l / r)
        if op == "^": return ForgeArray(np.linalg.matrix_power(l, int(r)) if _is_matrix_op(l, r) else l ** r)
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
            if not np.all(left):
                return ForgeArray(np.array(False))
            right = _unwrap(self._eval_expr(node.right, ws))
            return ForgeArray(np.array(bool(np.all(right))))
        if op == "||":
            if np.all(left):
                return ForgeArray(np.array(True))
            right = _unwrap(self._eval_expr(node.right, ws))
            return ForgeArray(np.array(bool(np.all(right))))
        right = _unwrap(self._eval_expr(node.right, ws))
        if op == "&": return ForgeArray(np.logical_and(left, right))
        if op == "|": return ForgeArray(np.logical_or(left, right))
        raise RuntimeError(f"Unknown logical op: {op}")

    def _eval_index(self, node: Index, ws: Workspace) -> Any:
        target = self._eval_expr(node.target, ws)
        # Check if it's a function call
        if isinstance(target, ForgeArray) and not callable(target):
            # Array indexing
            args = [self._eval_expr(a, ws) for a in node.args]
            return self._do_index(target, args)
        if callable(target):
            args = [self._eval_expr(a, ws) for a in node.args]
            return target(*args)
        if isinstance(node.target, Identifier):
            name = node.target.name
            if name in self.functions:
                fn = self.functions[name]
                args = [self._eval_expr(a, ws) for a in node.args]
                if callable(fn):
                    return fn(*args)
                return self._call_function(fn, args, ws)
        # Fall back to indexing
        args = [self._eval_expr(a, ws) for a in node.args]
        return self._do_index(target, args)

    def _do_index(self, target, args):
        """Perform array indexing with evaluated args."""
        if not isinstance(target, ForgeArray):
            raise TypeError(f"Cannot index {type(target).__name__}")
        if len(args) == 1:
            idx = args[0]
            if isinstance(idx, ForgeArray):
                if idx.dtype == np.bool_:
                    return ForgeArray(target.data[idx.data.ravel()])
                return ForgeArray(np.array([target[int(i)] for i in idx.data.ravel()]))
            return target[_to_int(idx)]
        elif len(args) == 2:
            r, c = args
            return target[_to_int(r), _to_int(c)]
        raise IndexError(f"Too many index dimensions: {len(args)}")

    def _eval_matrix(self, node: MatrixLiteral, ws: Workspace) -> ForgeArray:
        if not node.rows:
            return ForgeArray(np.array([]).reshape(0, 0))
        rows = []
        for row in node.rows:
            vals = [_unwrap(self._eval_expr(e, ws)) for e in row]
            vals = [np.atleast_2d(v) if np.asarray(v).ndim < 2 else v for v in vals]
            rows.append(np.concatenate(vals, axis=1))
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
            # Multi-return: [a, b] = func(x)
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
            arr = self._eval_expr(target.target, ws)
            args = [self._eval_expr(a, ws) for a in target.args]
            assign_val = _to_py(value) if isinstance(value, ForgeArray) and value.isscalar() else _unwrap(value)
            if isinstance(arr, ForgeArray):
                if len(args) == 1:
                    arr[_to_int(args[0])] = assign_val
                elif len(args) == 2:
                    arr[_to_int(args[0]), _to_int(args[1])] = assign_val
            return value

        if isinstance(target, FieldAccess):
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
    """Check if both operands are 2D matrices (not scalar/vector element-wise)."""
    a, b = np.asarray(a), np.asarray(b)
    return a.ndim == 2 and b.ndim == 2 and a.size > 1 and b.size > 1


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
