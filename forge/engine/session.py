"""Forge engine session — wraps the evaluator with builtins and session management."""
import os
import sys
import io
import numpy as np
from forge.engine.evaluator import Session as _EvalSession, Workspace, ForgeError, UndefinedFunctionError
from forge.engine.builtins import BUILTIN_REGISTRY
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar
from forge.engine.parser import parse


class ForgeSession:
    """A complete Forge/Octave execution session with workspace and path."""

    def __init__(self):
        self._engine = _EvalSession()
        self.path = [os.getcwd()]
        self.format = 'short'
        self.history = []
        self.last_error = None

        # Register all toolbox builtins into the engine's function table
        for name, func in BUILTIN_REGISTRY.items():
            self._engine.functions[name] = func

        # Register session-level builtins (cd, pwd, who, clear, etc.)
        self._register_session_builtins()

        # Give engine a back-reference for .m file discovery (R13)
        self._engine._session_ref = self

    # -- public API --------------------------------------------------------

    @property
    def workspace(self):
        return self._engine.workspace

    def eval(self, code):
        """Evaluate M-language code and return display string."""
        try:
            self.history.append(code)
            # Reset output buffer
            self._engine.output_buffer = io.StringIO()
            result = self._engine.eval(code)
            # Read captured output
            output = self._engine.output_buffer.getvalue()
            if output:
                return output.rstrip(chr(10))
            # R01: Check if statement was suppressed (semicolon)
            # The evaluator returns None for suppressed ExpressionStatements.
            # For assignments, check the source code for trailing semicolon.
            if result is not None:
                # Check if the code ends with semicolon (strip whitespace)
                stripped = code.rstrip()
                if stripped.endswith(';'):
                    return ''
                return self._format_result(result)
            return ''
        except ForgeError as e:
            self.last_error = e
            return f'error: {e.identifier}: {e.message}'
        except UndefinedFunctionError:
            raise
        except Exception as e:
            self.last_error = e
            return f'error: {type(e).__name__}: {e}'

    def get_workspace_dict(self):
        """Return workspace variables as plain dict for GUI display."""
        ws = self._engine.workspace
        return {n: ws.get(n) for n in ws.names()}

    # -- formatting --------------------------------------------------------

    def _format_result(self, value):
        from forge.engine.containers import ForgeCell, ForgeStruct
        # R03: ForgeChar displays as text
        if isinstance(value, ForgeChar):
            return value.to_str()
        if isinstance(value, ForgeCell):
            return self._format_cell(value)
        if isinstance(value, ForgeStruct):
            return self._format_struct(value)
        if isinstance(value, ForgeArray):
            data = _unwrap(value)
            if data.size == 0:
                return '  [](0x0)'
            # R02: MATLAB-style formatting
            if data.size == 1:
                v = data.flat[0]
                return '  ' + self._format_scalar(v)
            # Vector or matrix
            if data.ndim <= 2 and data.size <= 200:
                return self._format_matrix(data)
            shape = 'x'.join(str(s) for s in data.shape)
            return f'  [{shape} {data.dtype}]'
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return '  1' if value else '  0'
        if isinstance(value, (int, float)):
            return '  ' + self._format_scalar(value)
        return str(value)

    def _format_scalar(self, v):
        import numpy as _np
        if isinstance(v, (bool, _np.bool_)):
            return '1' if v else '0'
        if isinstance(v, (complex, _np.complexfloating)):
            return str(v)
        if isinstance(v, (int, _np.integer)):
            return str(int(v))
        # float
        if self.format == 'long':
            return f'{v:.15g}'
        elif self.format == 'short e':
            return f'{v:.4e}'
        elif self.format == 'long e':
            return f'{v:.15e}'
        else:  # short
            if _np.isfinite(v) and v == int(v) and abs(v) < 1e15:
                return str(int(v))
            return f'{v:.4f}'

    def _format_matrix(self, data):
        import numpy as _np
        if data.ndim == 1:
            data = data.reshape(1, -1)
        formatted = []
        for r in range(data.shape[0]):
            row = []
            for c in range(data.shape[1]):
                row.append(self._format_scalar(data[r, c]))
            formatted.append(row)
        ncols = data.shape[1]
        col_widths = []
        for c in range(ncols):
            w = max(len(formatted[r][c]) for r in range(data.shape[0]))
            col_widths.append(max(w, 1))
        lines = []
        for r in range(data.shape[0]):
            parts = []
            for c in range(ncols):
                parts.append(formatted[r][c].rjust(col_widths[c] + 4))
            lines.append(''.join(parts))
        return '\n'.join(lines)

    def _format_cell(self, cell):
        """Format a cell array for display, MATLAB style."""
        from forge.engine.containers import ForgeChar, ForgeCell, ForgeStruct
        items = cell._data
        shape = cell.shape
        lines = []
        lines.append('{')
        for i, item in enumerate(items):
            if isinstance(item, ForgeChar):
                val_str = f"  [{i+1}] '{item.to_str()}'"
            elif isinstance(item, ForgeArray):
                data = _unwrap(item)
                if data.size == 0:
                    val_str = f"  [{i+1}] []"
                elif data.size == 1:
                    val_str = f"  [{i+1}] {self._format_scalar(data.flat[0])}"
                elif data.size <= 6:
                    vals = " ".join(self._format_scalar(x) for x in data.flat)
                    val_str = f"  [{i+1}] [{vals}]"
                else:
                    sh = "x".join(str(s) for s in data.shape)
                    val_str = f"  [{i+1}] [{sh} double]"
            elif isinstance(item, ForgeCell):
                n = len(item._data)
                val_str = f"  [{i+1}] {{{n}x1 cell}}"
            elif isinstance(item, ForgeStruct):
                fields = list(item._fields.keys()) if hasattr(item, '_fields') else []
                val_str = f"  [{i+1}] [1x1 struct]"
            else:
                val_str = f"  [{i+1}] {item}"
            lines.append(val_str)
            if i >= 19:
                lines.append(f"  ... ({len(items) - 20} more)")
                break
        lines.append('}')
        return '\n'.join(lines)

    def _format_struct(self, s):
        """Format a struct for display, MATLAB style."""
        from forge.engine.containers import ForgeChar, ForgeCell, ForgeStruct
        if not hasattr(s, '_fields') or not s._fields:
            return "  struct with no fields."
        lines = []
        for name, val in s._fields.items():
            if isinstance(val, ForgeChar):
                val_str = f"'{val.to_str()}'"
            elif isinstance(val, ForgeArray):
                data = _unwrap(val)
                if data.size == 0:
                    val_str = "[]"
                elif data.size == 1:
                    val_str = self._format_scalar(data.flat[0])
                elif data.size <= 6:
                    vals = " ".join(self._format_scalar(x) for x in data.flat)
                    val_str = f"[{vals}]"
                else:
                    sh = "x".join(str(s2) for s2 in data.shape)
                    val_str = f"[{sh} double]"
            elif isinstance(val, ForgeCell):
                n = len(val._data)
                val_str = f"{{{n}x1 cell}}"
            elif isinstance(val, ForgeStruct):
                val_str = "[1x1 struct]"
            else:
                val_str = str(val)
            lines.append(f"    {name}: {val_str}")
        return '\n'.join(lines)

    # -- session builtins --------------------------------------------------

    def _register_session_builtins(self):
        session = self

        def forge_cd(*args):
            if not args:
                return os.getcwd()
            os.chdir(str(args[0]))
            session.path[0] = os.getcwd()
            return os.getcwd()

        def forge_pwd():
            return os.getcwd()

        def forge_who():
            return '\n'.join(sorted(session._engine.workspace.names()))

        def forge_whos():
            _dtype_map = {
                "float64": "double", "float32": "single",
                "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
                "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
                "bool": "logical", "complex128": "double complex", "complex64": "single complex",
            }
            lines = []
            ws = session._engine.workspace
            for name in sorted(ws.names()):
                val = ws.get(name)
                if isinstance(val, ForgeChar):
                    s_val = val.to_str()
                    shape = f'1x{len(s_val)}'
                    lines.append(f'  {name:20s} {shape:10s} {"char":10s}')
                elif isinstance(val, ForgeArray):
                    d = _unwrap(val)
                    shape = 'x'.join(str(s) for s in d.shape)
                    cls = _dtype_map.get(str(d.dtype), str(d.dtype))
                    lines.append(f'  {name:20s} {shape:10s} {cls:10s}')
                else:
                    from forge.engine.containers import ForgeCell, ForgeStruct
                    if isinstance(val, ForgeCell):
                        shape = 'x'.join(str(s) for s in val.shape)
                        lines.append(f'  {name:20s} {shape:10s} {"cell":10s}')
                    elif isinstance(val, ForgeStruct):
                        n_fields = len(val._fields) if hasattr(val, '_fields') else 0
                        lines.append(f'  {name:20s} {"1x1":10s} {"struct":10s}')
                    else:
                        lines.append(f'  {name:20s} {"1x1":10s} {type(val).__name__:10s}')
            return '\n'.join(lines)

        def forge_clear(*args):
            ws = session._engine.workspace
            _PROTECTED = {"pi", "e", "eps", "Inf", "inf", "NaN", "nan",
                          "true", "false", "i", "j", "realmin", "realmax"}
            if not args:
                for n in list(ws.names()):
                    if n not in _PROTECTED:
                        ws.delete(n)
            else:
                for a in args:
                    name = str(a)
                    if name not in _PROTECTED:
                        ws.delete(name)

        def forge_addpath(*args):
            for p in args:
                path = str(p)
                if path not in session.path:
                    session.path.insert(0, path)

        def forge_rmpath(*args):
            for p in args:
                path = str(p)
                if path in session.path:
                    session.path.remove(path)

        def forge_path_cmd():
            return '\n'.join(session.path)

        def forge_disp(x):
            text = session._format_result(x)
            if text:
                session._engine.output_buffer.write(text + chr(10))

        def forge_exist(name):
            n = str(name)
            ws = session._engine.workspace
            if ws.has(n):
                return ForgeArray(np.array(1.0))
            if n in session._engine.functions:
                return ForgeArray(np.array(5.0))
            return ForgeArray(np.array(0.0))

        def forge_class(x):
            from forge.engine.containers import ForgeCell, ForgeStruct
            _dtype_map = {
                "float64": "double", "float32": "single",
                "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
                "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
                "bool": "logical", "complex128": "double", "complex64": "single",
            }
            if isinstance(x, ForgeChar):
                return 'char'
            if isinstance(x, ForgeArray):
                dtype_name = str(_unwrap(x).dtype)
                return _dtype_map.get(dtype_name, dtype_name)
            if isinstance(x, ForgeCell):
                return 'cell'
            if isinstance(x, ForgeStruct):
                return 'struct'
            if isinstance(x, bool):
                return 'logical'
            if isinstance(x, (int, float)):
                return 'double'
            return type(x).__name__

        def forge_format(*args):
            if not args:
                return session.format
            fmt_str = str(args[0])
            if len(args) > 1:
                fmt_str += ' ' + str(args[1])
            session.format = fmt_str

        def forge_run(path_arg):
            """Execute a .m script file."""
            from forge.engine.parser import parse as _parse
            p = str(path_arg)
            if isinstance(path_arg, ForgeChar):
                p = path_arg.to_str()
            with open(p, 'r') as f:
                source = f.read()
            stmts = _parse(source)
            for stmt in stmts:
                session._engine._exec(stmt, session._engine.workspace)


        def forge_clc():
            """Clear the command window output."""
            # In GUI mode, this would clear the output display
            # In engine mode, this is a no-op
            pass

        for name, fn in [
            ('cd', forge_cd), ('pwd', forge_pwd), ('who', forge_who),
            ('whos', forge_whos), ('clear', forge_clear),
            ('addpath', forge_addpath), ('rmpath', forge_rmpath),
            ('path', forge_path_cmd), ('disp', forge_disp),
            ('exist', forge_exist), ('class', forge_class),
            ('format', forge_format),
            ('run', forge_run),
            ('clc', forge_clc),
            ('source', forge_run),
        ]:
            self._engine.functions[name] = fn

        # Container utilities (R90)
        from forge.engine.evaluator import (
            _cellfun_builtin, _arrayfun_builtin, _num2cell_builtin,
            _cell2mat_builtin, _rmfield_builtin, _cat_builtin
        )
        self._engine.functions["cellfun"] = _cellfun_builtin
        self._engine.functions["arrayfun"] = _arrayfun_builtin
        self._engine.functions["num2cell"] = _num2cell_builtin
        self._engine.functions["cell2mat"] = _cell2mat_builtin
        self._engine.functions["rmfield"] = _rmfield_builtin
        self._engine.functions["cat"] = _cat_builtin

        # File I/O system (R94)
        session._file_handles = {}  # fid -> file object
        session._next_fid = 3  # 0=stdin, 1=stdout, 2=stderr

        def forge_fopen(*args):
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import numpy as np
            if len(args) < 1:
                raise ValueError("fopen requires at least a filename")
            fname = args[0]
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            fname = str(fname)
            mode = "r"
            if len(args) >= 2:
                m = args[1]
                if isinstance(m, ForgeChar): m = m.to_str()
                mode = str(m)
            mode_map = {"r": "r", "w": "w", "a": "a", "r+": "r+", "w+": "w+", "a+": "a+",
                        "rb": "rb", "wb": "wb", "ab": "ab"}
            py_mode = mode_map.get(mode, mode)
            try:
                fh = open(fname, py_mode)
                fid = session._next_fid
                session._next_fid += 1
                session._file_handles[fid] = fh
                return ForgeArray(np.float64(fid))
            except:
                return ForgeArray(np.float64(-1))

        def forge_fclose(fid=None):
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import numpy as np
            if fid is None or (isinstance(fid, ForgeChar) and fid.to_str() == "all"):
                for f in session._file_handles.values():
                    try: f.close()
                    except: pass
                session._file_handles.clear()
                return ForgeArray(np.float64(0))
            fid_val = int(float(fid.data.flat[0]) if isinstance(fid, ForgeArray) else float(fid))
            if fid_val in session._file_handles:
                session._file_handles[fid_val].close()
                del session._file_handles[fid_val]
                return ForgeArray(np.float64(0))
            return ForgeArray(np.float64(-1))

        def forge_fprintf_file(fid, fmt, *args):
            """fprintf to file handle."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import numpy as np
            fid_val = int(float(fid.data.flat[0]) if isinstance(fid, ForgeArray) else float(fid))
            if isinstance(fmt, ForgeChar): fmt = fmt.to_str()
            fmt = str(fmt)
            # Convert args
            py_args = []
            for a in args:
                if isinstance(a, ForgeChar): py_args.append(a.to_str())
                elif isinstance(a, ForgeArray): py_args.append(a.data.flat[0])
                else: py_args.append(a)
            try:
                text = fmt % tuple(py_args) if py_args else fmt
            except:
                text = fmt
            if fid_val == 1:
                return text  # stdout
            elif fid_val == 2:
                import sys; sys.stderr.write(text)
                return ''
            elif fid_val in session._file_handles:
                session._file_handles[fid_val].write(text)
                return ''
            return ''

        def forge_fgets(fid, nchar=None):
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            import numpy as np
            fid_val = int(float(fid.data.flat[0]) if isinstance(fid, ForgeArray) else float(fid))
            if fid_val not in session._file_handles:
                return ForgeArray(np.float64(-1))
            fh = session._file_handles[fid_val]
            if nchar is not None:
                nchar = int(float(nchar.data.flat[0]) if isinstance(nchar, ForgeArray) else float(nchar))
                line = fh.read(nchar)
            else:
                line = fh.readline()
            if not line:
                return ForgeArray(np.float64(-1))
            return ForgeChar(line)

        def forge_fgetl(fid):
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            import numpy as np
            fid_val = int(float(fid.data.flat[0]) if isinstance(fid, ForgeArray) else float(fid))
            if fid_val not in session._file_handles:
                return ForgeArray(np.float64(-1))
            line = session._file_handles[fid_val].readline()
            if not line:
                return ForgeArray(np.float64(-1))
            return ForgeChar(line.rstrip('\n').rstrip('\r'))

        def forge_feof(fid):
            from forge.engine.types import ForgeArray
            import numpy as np
            fid_val = int(float(fid.data.flat[0]) if isinstance(fid, ForgeArray) else float(fid))
            if fid_val not in session._file_handles:
                return ForgeArray(np.float64(1))
            fh = session._file_handles[fid_val]
            pos = fh.tell()
            ch = fh.read(1)
            if not ch:
                return ForgeArray(np.float64(1))
            fh.seek(pos)
            return ForgeArray(np.float64(0))

        def forge_ftell(fid):
            from forge.engine.types import ForgeArray
            import numpy as np
            fid_val = int(float(fid.data.flat[0]) if isinstance(fid, ForgeArray) else float(fid))
            if fid_val not in session._file_handles:
                return ForgeArray(np.float64(-1))
            return ForgeArray(np.float64(session._file_handles[fid_val].tell()))

        def forge_fseek(fid, offset, origin=None):
            from forge.engine.types import ForgeArray
            import numpy as np
            fid_val = int(float(fid.data.flat[0]) if isinstance(fid, ForgeArray) else float(fid))
            offset = int(float(offset.data.flat[0]) if isinstance(offset, ForgeArray) else float(offset))
            whence = 0
            if origin is not None:
                ov = int(float(origin.data.flat[0]) if isinstance(origin, ForgeArray) else float(origin))
                whence = {-1: 0, 0: 1, 1: 2}.get(ov, ov)
            if fid_val in session._file_handles:
                session._file_handles[fid_val].seek(offset, whence)
                return ForgeArray(np.float64(0))
            return ForgeArray(np.float64(-1))

        def forge_frewind(fid):
            from forge.engine.types import ForgeArray
            import numpy as np
            return forge_fseek(fid, ForgeArray(np.float64(0)))

        def forge_fileread(fname):
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            with open(str(fname), 'r') as f:
                return ForgeChar(f.read())

        def forge_dlmread(fname, delim=None, *args):
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import numpy as np
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            if isinstance(delim, ForgeChar): delim = delim.to_str()
            if delim is None or delim == "": delim = None
            data = np.loadtxt(str(fname), delimiter=delim)
            return ForgeArray(data)

        def forge_dlmwrite(fname, data, delim=None):
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import numpy as np
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            if isinstance(delim, ForgeChar): delim = delim.to_str()
            if isinstance(data, ForgeArray): data = data.data
            if delim is None: delim = ","
            np.savetxt(str(fname), np.atleast_2d(data), delimiter=delim, fmt='%.6g')
            return ''

        def forge_csvread(fname, *args):
            from forge.engine.containers import ForgeChar
            return forge_dlmread(fname, ForgeChar(","), *args)

        def forge_csvwrite(fname, data):
            from forge.engine.containers import ForgeChar
            return forge_dlmwrite(fname, data, ForgeChar(","))

        def forge_tempname():
            import tempfile
            from forge.engine.containers import ForgeChar
            return ForgeChar(tempfile.mktemp())

        def forge_tempdir():
            import tempfile
            from forge.engine.containers import ForgeChar
            return ForgeChar(tempfile.gettempdir())

        for _name, _func in [
            ("fopen", forge_fopen), ("fclose", forge_fclose),
            ("fgets", forge_fgets), ("fgetl", forge_fgetl),
            ("feof", forge_feof), ("ftell", forge_ftell),
            ("fseek", forge_fseek), ("frewind", forge_frewind),
            ("fileread", forge_fileread),
            ("dlmread", forge_dlmread), ("dlmwrite", forge_dlmwrite),
            ("csvread", forge_csvread), ("csvwrite", forge_csvwrite),
            ("tempname", forge_tempname), ("tempdir", forge_tempdir),
        ]:
            self._engine.functions[_name] = _func

        # Override fprintf to handle file handles
        _orig_fprintf = self._engine.functions.get("fprintf")
        def forge_fprintf_dispatch(*args):
            from forge.engine.types import ForgeArray
            import numpy as np
            if args and isinstance(args[0], ForgeArray):
                v = args[0].data.flat[0]
                if isinstance(v, (int, float, np.integer, np.floating)) and float(v) == int(float(v)):
                    fid_val = int(float(v))
                    if fid_val in session._file_handles or fid_val in (1, 2):
                        return forge_fprintf_file(*args)
            # Fall through to original fprintf (stdout)
            if _orig_fprintf:
                return _orig_fprintf(*args)
            raise ValueError("fprintf requires arguments")
        self._engine.functions["fprintf"] = forge_fprintf_dispatch

        # Misc builtins (R96)
        from forge.engine.evaluator import _deal_builtin, _structfun_builtin
        self._engine.functions["deal"] = _deal_builtin
        self._engine.functions["structfun"] = _structfun_builtin

        # ODE Solvers (R97)
        def forge_ode45(func, tspan, y0, *args):
            """Solve ODE using Runge-Kutta 4(5) method (Dormand-Prince)."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeCell
            import numpy as np
            from scipy.integrate import solve_ivp

            # Extract tspan
            if isinstance(tspan, ForgeArray):
                tspan_arr = tspan.data.ravel()
            else:
                tspan_arr = np.asarray(tspan).ravel()

            # Extract y0
            if isinstance(y0, ForgeArray):
                y0_arr = y0.data.ravel()
            else:
                y0_arr = np.atleast_1d(np.asarray(y0, dtype=np.float64))

            t_eval = None
            if len(tspan_arr) > 2:
                t_eval = tspan_arr
                t_span = (tspan_arr[0], tspan_arr[-1])
            else:
                t_span = (float(tspan_arr[0]), float(tspan_arr[-1]))

            # Wrap the Forge function for scipy
            def rhs(t, y):
                t_fa = ForgeArray(np.float64(t))
                y_fa = ForgeArray(np.array(y, dtype=np.float64).reshape(-1, 1))
                result = func(t_fa, y_fa)
                if isinstance(result, ForgeArray):
                    return result.data.ravel()
                return np.asarray(result, dtype=np.float64).ravel()

            sol = solve_ivp(rhs, t_span, y0_arr.astype(np.float64),
                           method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-9)

            t_out = ForgeArray(sol.t.reshape(1, -1))
            y_out = ForgeArray(sol.y)  # n_vars x n_points
            return (t_out, y_out)

        def forge_ode23(func, tspan, y0, *args):
            """Solve ODE using Bogacki-Shampine method."""
            from forge.engine.types import ForgeArray
            import numpy as np
            from scipy.integrate import solve_ivp

            if isinstance(tspan, ForgeArray):
                tspan_arr = tspan.data.ravel()
            else:
                tspan_arr = np.asarray(tspan).ravel()

            if isinstance(y0, ForgeArray):
                y0_arr = y0.data.ravel()
            else:
                y0_arr = np.atleast_1d(np.asarray(y0, dtype=np.float64))

            t_eval = tspan_arr if len(tspan_arr) > 2 else None
            t_span = (float(tspan_arr[0]), float(tspan_arr[-1]))

            def rhs(t, y):
                t_fa = ForgeArray(np.float64(t))
                y_fa = ForgeArray(np.array(y, dtype=np.float64).reshape(-1, 1))
                result = func(t_fa, y_fa)
                if isinstance(result, ForgeArray):
                    return result.data.ravel()
                return np.asarray(result, dtype=np.float64).ravel()

            sol = solve_ivp(rhs, t_span, y0_arr.astype(np.float64),
                           method='RK23', t_eval=t_eval, rtol=1e-3, atol=1e-6)

            t_out = ForgeArray(sol.t.reshape(1, -1))
            y_out = ForgeArray(sol.y)
            return (t_out, y_out)

        def forge_ode15s(func, tspan, y0, *args):
            """Solve stiff ODE using BDF method."""
            from forge.engine.types import ForgeArray
            import numpy as np
            from scipy.integrate import solve_ivp

            if isinstance(tspan, ForgeArray):
                tspan_arr = tspan.data.ravel()
            else:
                tspan_arr = np.asarray(tspan).ravel()

            if isinstance(y0, ForgeArray):
                y0_arr = y0.data.ravel()
            else:
                y0_arr = np.atleast_1d(np.asarray(y0, dtype=np.float64))

            t_eval = tspan_arr if len(tspan_arr) > 2 else None
            t_span = (float(tspan_arr[0]), float(tspan_arr[-1]))

            def rhs(t, y):
                t_fa = ForgeArray(np.float64(t))
                y_fa = ForgeArray(np.array(y, dtype=np.float64).reshape(-1, 1))
                result = func(t_fa, y_fa)
                if isinstance(result, ForgeArray):
                    return result.data.ravel()
                return np.asarray(result, dtype=np.float64).ravel()

            sol = solve_ivp(rhs, t_span, y0_arr.astype(np.float64),
                           method='BDF', t_eval=t_eval, rtol=1e-6, atol=1e-9)

            t_out = ForgeArray(sol.t.reshape(1, -1))
            y_out = ForgeArray(sol.y)
            return (t_out, y_out)

        def forge_odeset(*args):
            """Create ODE options struct (stub)."""
            from forge.engine.containers import ForgeStruct, ForgeChar
            from forge.engine.types import ForgeArray
            opts = ForgeStruct()
            i = 0
            while i < len(args) - 1:
                key = args[i]
                val = args[i+1]
                if isinstance(key, ForgeChar):
                    key = key.to_str()
                opts._fields[str(key)] = val
                i += 2
            return opts

        self._engine.functions["ode45"] = forge_ode45
        self._engine.functions["ode23"] = forge_ode23
        self._engine.functions["ode15s"] = forge_ode15s
        self._engine.functions["odeset"] = forge_odeset

        # Optimization functions (R98)
        def forge_fzero(func, x0, *args):
            """Find zero of a function. x = fzero(@f, x0) or fzero(@f, [a b])."""
            from forge.engine.types import ForgeArray
            import numpy as np
            from scipy.optimize import brentq, fsolve as _fsolve

            if isinstance(x0, ForgeArray):
                x0_arr = x0.data.ravel()
            else:
                x0_arr = np.atleast_1d(np.asarray(x0, dtype=np.float64))

            def f_scalar(x):
                r = func(ForgeArray(np.float64(x)))
                if isinstance(r, ForgeArray):
                    return float(r.data.flat[0])
                return float(r)

            if len(x0_arr) == 2:
                # Bracket method
                root = brentq(f_scalar, float(x0_arr[0]), float(x0_arr[1]))
            else:
                # Initial guess method
                root = _fsolve(f_scalar, float(x0_arr[0]))[0]
            return ForgeArray(np.float64(root))

        def forge_fminbnd(func, a, b, *args):
            """Find minimum of function on interval [a, b]."""
            from forge.engine.types import ForgeArray
            import numpy as np
            from scipy.optimize import minimize_scalar

            if isinstance(a, ForgeArray): a = float(a.data.flat[0])
            if isinstance(b, ForgeArray): b = float(b.data.flat[0])

            def f_scalar(x):
                r = func(ForgeArray(np.float64(x)))
                if isinstance(r, ForgeArray):
                    return float(r.data.flat[0])
                return float(r)

            result = minimize_scalar(f_scalar, bounds=(float(a), float(b)), method='bounded')
            return ForgeArray(np.float64(result.x))

        def forge_fminsearch(func, x0, *args):
            """Find minimum of unconstrained multivariable function (Nelder-Mead)."""
            from forge.engine.types import ForgeArray
            import numpy as np
            from scipy.optimize import minimize

            if isinstance(x0, ForgeArray):
                x0_arr = x0.data.ravel().astype(np.float64)
            else:
                x0_arr = np.atleast_1d(np.asarray(x0, dtype=np.float64))

            def f_vec(x):
                x_fa = ForgeArray(np.array(x, dtype=np.float64))
                r = func(x_fa)
                if isinstance(r, ForgeArray):
                    return float(r.data.flat[0])
                return float(r)

            result = minimize(f_vec, x0_arr, method='Nelder-Mead')
            return ForgeArray(result.x)

        def forge_fsolve(func, x0, *args):
            """Solve system of nonlinear equations."""
            from forge.engine.types import ForgeArray
            import numpy as np
            from scipy.optimize import fsolve as _fsolve

            if isinstance(x0, ForgeArray):
                x0_arr = x0.data.ravel().astype(np.float64)
            else:
                x0_arr = np.atleast_1d(np.asarray(x0, dtype=np.float64))

            def f_vec(x):
                x_fa = ForgeArray(np.array(x, dtype=np.float64))
                r = func(x_fa)
                if isinstance(r, ForgeArray):
                    return r.data.ravel()
                return np.asarray(r, dtype=np.float64).ravel()

            result = _fsolve(f_vec, x0_arr)
            return ForgeArray(result)

        def forge_integral(func, a, b, *args):
            """Numerical integration. q = integral(@f, a, b)."""
            from forge.engine.types import ForgeArray
            import numpy as np
            from scipy.integrate import quad

            if isinstance(a, ForgeArray): a = float(a.data.flat[0])
            if isinstance(b, ForgeArray): b = float(b.data.flat[0])

            def f_scalar(x):
                r = func(ForgeArray(np.float64(x)))
                if isinstance(r, ForgeArray):
                    return float(r.data.flat[0])
                return float(r)

            val, err = quad(f_scalar, float(a), float(b))
            return ForgeArray(np.float64(val))

        self._engine.functions["fzero"] = forge_fzero
        self._engine.functions["fminbnd"] = forge_fminbnd
        self._engine.functions["fminsearch"] = forge_fminsearch
        self._engine.functions["fsolve"] = forge_fsolve
        self._engine.functions["integral"] = forge_integral

        # Fill gaps (R99)
        import numpy as np
        from forge.engine.types import ForgeArray
        from forge.engine.containers import ForgeChar

        def forge_cbrt(x):
            if isinstance(x, ForgeArray): return ForgeArray(np.cbrt(x.data))
            return ForgeArray(np.cbrt(np.float64(x)))

        def forge_pow2(x):
            if isinstance(x, ForgeArray): return ForgeArray(np.ldexp(1.0, x.data.astype(int)))
            return ForgeArray(np.float64(2**int(float(x))))

        def forge_asinh(x):
            if isinstance(x, ForgeArray): return ForgeArray(np.arcsinh(x.data))
            return ForgeArray(np.arcsinh(np.float64(x)))

        def forge_acosh(x):
            if isinstance(x, ForgeArray): return ForgeArray(np.arccosh(x.data))
            return ForgeArray(np.arccosh(np.float64(x)))

        def forge_atanh(x):
            if isinstance(x, ForgeArray): return ForgeArray(np.arctanh(x.data))
            return ForgeArray(np.arctanh(np.float64(x)))

        def forge_isa(x, typename):
            if isinstance(typename, ForgeChar): typename = typename.to_str()
            cls = session._engine.functions.get("class")
            if cls:
                actual = cls(x)
                if isinstance(actual, ForgeChar): actual = actual.to_str()
                return ForgeArray(np.float64(1.0 if str(actual) == str(typename) else 0.0))
            return ForgeArray(np.float64(0.0))

        def forge_isreal(x):
            if isinstance(x, ForgeArray):
                return ForgeArray(np.float64(1.0 if not np.iscomplexobj(x.data) else 0.0))
            return ForgeArray(np.float64(1.0 if not isinstance(x, complex) else 0.0))

        def forge_display(x):
            return session._format_result(x)

        def forge_interp2(X, Y, V, Xq, Yq, *args):
            from scipy.interpolate import RegularGridInterpolator
            if isinstance(X, ForgeArray): X = X.data
            if isinstance(Y, ForgeArray): Y = Y.data
            if isinstance(V, ForgeArray): V = V.data
            if isinstance(Xq, ForgeArray): Xq = Xq.data
            if isinstance(Yq, ForgeArray): Yq = Yq.data
            # Build interpolator
            x_vec = np.unique(X.ravel())
            y_vec = np.unique(Y.ravel())
            interp = RegularGridInterpolator((y_vec, x_vec), V, method='linear', bounds_error=False, fill_value=np.nan)
            points = np.column_stack([Yq.ravel(), Xq.ravel()])
            result = interp(points).reshape(Xq.shape)
            return ForgeArray(result)

        def forge_save(fname, *varnames):
            """Save workspace variables to .mat-like file."""
            import json
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            ws = session._engine.workspace
            data = {}
            names = [v.to_str() if isinstance(v, ForgeChar) else str(v) for v in varnames] if varnames else list(ws.names())
            for name in names:
                val = ws.get(name)
                if isinstance(val, ForgeArray):
                    data[name] = {"type": "array", "data": val.data.tolist(), "shape": list(val.data.shape)}
                elif isinstance(val, ForgeChar):
                    data[name] = {"type": "char", "data": val.to_str()}
                elif isinstance(val, (int, float)):
                    data[name] = {"type": "scalar", "data": float(val)}
            with open(str(fname), 'w') as f:
                json.dump(data, f)
            return ''

        def forge_load(fname, *varnames):
            """Load workspace variables from saved file."""
            import json
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            with open(str(fname), 'r') as f:
                data = json.load(f)
            ws = session._engine.workspace
            for name, val_dict in data.items():
                if varnames and name not in [v.to_str() if isinstance(v, ForgeChar) else str(v) for v in varnames]:
                    continue
                if val_dict["type"] == "array":
                    arr = np.array(val_dict["data"], dtype=np.float64).reshape(val_dict["shape"])
                    ws.set(name, ForgeArray(arr))
                elif val_dict["type"] == "char":
                    ws.set(name, ForgeChar(val_dict["data"]))
                elif val_dict["type"] == "scalar":
                    ws.set(name, ForgeArray(np.float64(val_dict["data"])))
            return ''

        for _n, _f in [
            ("cbrt", forge_cbrt), ("pow2", forge_pow2),
            ("asinh", forge_asinh), ("acosh", forge_acosh), ("atanh", forge_atanh),
            ("isa", forge_isa), ("isreal", forge_isreal),
            ("display", forge_display), ("interp2", forge_interp2),
            ("save", forge_save), ("load", forge_load),
        ]:
            self._engine.functions[_n] = _f

        # Small gap fixes (R102)
        def forge_sqrtm(A):
            from forge.engine.types import ForgeArray
            import numpy as np
            from scipy.linalg import sqrtm as _sqrtm
            if isinstance(A, ForgeArray): A = A.data
            result = _sqrtm(np.atleast_2d(A))
            if np.isrealobj(A) and np.allclose(result.imag, 0):
                result = result.real
            return ForgeArray(result)

        def forge_rcond(A):
            from forge.engine.types import ForgeArray
            import numpy as np
            if isinstance(A, ForgeArray): A = A.data
            return ForgeArray(np.float64(1.0 / np.linalg.cond(np.atleast_2d(A))))

        def forge_blkdiag(*args):
            from forge.engine.types import ForgeArray
            import numpy as np
            from scipy.linalg import block_diag
            mats = []
            for a in args:
                if isinstance(a, ForgeArray):
                    mats.append(np.atleast_2d(a.data))
                elif isinstance(a, np.ndarray):
                    mats.append(np.atleast_2d(a))
                else:
                    mats.append(np.atleast_2d(np.array([[float(a)]])))
            return ForgeArray(block_diag(*mats))

        self._engine.functions["sqrtm"] = forge_sqrtm
        self._engine.functions["rcond"] = forge_rcond
        self._engine.functions["blkdiag"] = forge_blkdiag

        # R105: eval, feval, nargin, nargout, strjoin
        def forge_eval_str(code_str):
            """eval(str) - evaluate string as code."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(code_str, ForgeChar):
                code_str = code_str.to_str()
            elif isinstance(code_str, str):
                pass
            else:
                code_str = str(code_str)
            result = session._engine.eval(code_str)
            return result

        def forge_feval(fname, *args):
            """feval(fname, args...) - call function by name."""
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar):
                fname = fname.to_str()
            if fname in session._engine.functions:
                return session._engine.functions[fname](*args)
            raise NameError(f"Undefined function: {fname}")

        def forge_nargin_func(fname):
            """nargin(fname) - number of input arguments."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import inspect
            if isinstance(fname, ForgeChar):
                fname = fname.to_str()
            if fname in session._engine.functions:
                func = session._engine.functions[fname]
                try:
                    sig = inspect.signature(func)
                    # Count params without defaults that aren't *args/**kwargs
                    count = 0
                    has_var = False
                    for p in sig.parameters.values():
                        if p.kind == p.VAR_POSITIONAL:
                            has_var = True
                        elif p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                            count += 1
                    if has_var:
                        return ForgeArray(np.float64(-1))  # variable args
                    return ForgeArray(np.float64(count))
                except (ValueError, TypeError):
                    return ForgeArray(np.float64(-1))
            return ForgeArray(np.float64(0))

        def forge_nargout_func(fname):
            """nargout(fname) - number of output arguments."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar):
                fname = fname.to_str()
            # Most functions return 1 output; some special cases
            multi_out = {"size": 2, "eig": 2, "svd": 3, "lu": 3, "qr": 2,
                         "min": 2, "max": 2, "sort": 2, "find": 3,
                         "meshgrid": 2, "cart2pol": 2, "pol2cart": 2}
            if fname in multi_out:
                return ForgeArray(np.float64(multi_out[fname]))
            return ForgeArray(np.float64(1))

        def forge_strjoin(cell_arr, delim=None):
            """strjoin(C, delim) - join cell array of strings."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            from forge.engine.containers import ForgeCell
            if delim is None:
                delim = " "
            elif isinstance(delim, ForgeChar):
                delim = delim.to_str()
            elif isinstance(delim, str):
                pass
            parts = []
            if isinstance(cell_arr, ForgeCell):
                data = cell_arr._data
                items = data if isinstance(data, list) else data.flat
                for item in items:
                    if isinstance(item, ForgeChar):
                        parts.append(item.to_str())
                    elif isinstance(item, str):
                        parts.append(item)
                    elif item is not None:
                        parts.append(str(item))
            result = delim.join(parts)
            return ForgeChar(result)

        session._engine.functions["eval"] = forge_eval_str
        session._engine.functions["feval"] = forge_feval
        session._engine.functions["nargin"] = forge_nargin_func
        session._engine.functions["nargout"] = forge_nargout_func
        session._engine.functions["strjoin"] = forge_strjoin

        # R108: sprintf, error, warning, assert, inputname, class
        def forge_sprintf(fmt, *args):
            """sprintf(fmt, args...) - format string."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            fmt_str = fmt.to_str() if isinstance(fmt, ForgeChar) else (str(fmt) if not isinstance(fmt, str) else fmt)
            # Convert ForgeArray args to Python scalars
            conv_args = []
            for a in args:
                if isinstance(a, ForgeArray):
                    v = a.data.flat[0]
                    if isinstance(v, (np.integer,)):
                        conv_args.append(int(v))
                    elif isinstance(v, (np.floating,)):
                        conv_args.append(float(v))
                    else:
                        conv_args.append(v)
                elif isinstance(a, ForgeChar):
                    conv_args.append(a.to_str())
                else:
                    conv_args.append(a)
            # Process escape sequences
            fmt_str = fmt_str.replace("\\n", "\n").replace("\\t", "\t")
            try:
                result = fmt_str % tuple(conv_args)
            except TypeError:
                result = fmt_str
            return ForgeChar(result)

        def forge_error(msg, *args):
            """error(msg, args...) - throw error."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(msg, ForgeChar):
                msg = msg.to_str()
            if args:
                conv_args = []
                for a in args:
                    if isinstance(a, ForgeArray):
                        conv_args.append(a.data.flat[0])
                    elif isinstance(a, ForgeChar):
                        conv_args.append(a.to_str())
                    else:
                        conv_args.append(a)
                msg = msg.replace("\\n", "\n") % tuple(conv_args)
            raise RuntimeError(msg)

        def forge_warning(msg, *args):
            """warning(msg) - print warning."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            import warnings
            if isinstance(msg, ForgeChar):
                msg = msg.to_str()
            if args:
                conv_args = []
                for a in args:
                    if isinstance(a, ForgeArray):
                        conv_args.append(a.data.flat[0])
                    elif isinstance(a, ForgeChar):
                        conv_args.append(a.to_str())
                    else:
                        conv_args.append(a)
                msg = msg % tuple(conv_args)
            import sys
            print(f"warning: {msg}", file=sys.stderr)
            return ForgeArray(np.array(0.0))

        def forge_assert(*args):
            """assert(cond) or assert(obs, exp) or assert(obs, exp, tol)."""
            from forge.engine.types import ForgeArray
            if len(args) == 1:
                cond = args[0]
                if isinstance(cond, ForgeArray):
                    if not np.all(cond.data):
                        raise AssertionError("assertion failed")
                elif not cond:
                    raise AssertionError("assertion failed")
            elif len(args) == 2:
                obs, exp = args
                if isinstance(obs, ForgeArray): obs = obs.data
                if isinstance(exp, ForgeArray): exp = exp.data
                if not np.allclose(obs, exp):
                    raise AssertionError(f"assert ({obs}) != ({exp})")
            elif len(args) >= 3:
                obs, exp, tol = args[0], args[1], args[2]
                if isinstance(obs, ForgeArray): obs = obs.data
                if isinstance(exp, ForgeArray): exp = exp.data
                if isinstance(tol, ForgeArray): tol = float(tol.data.flat[0])
                if not np.allclose(obs, exp, atol=tol, rtol=0):
                    raise AssertionError(f"assert ({obs}) != ({exp}) within tol={tol}")
            return ForgeArray(np.array(1.0))

        def forge_inputname(n):
            """inputname(n) - name of nth input argument (placeholder)."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray):
                n = int(n.data.flat[0])
            return ForgeChar(f"arg{n}")

        def forge_class(obj):
            """class(obj) - return class name."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar, ForgeCell, ForgeStruct
            if isinstance(obj, ForgeChar):
                return ForgeChar("char")
            if isinstance(obj, ForgeCell):
                return ForgeChar("cell")
            if isinstance(obj, ForgeStruct):
                return ForgeChar("struct")
            if isinstance(obj, ForgeArray):
                dt = obj.data.dtype
                if dt == np.float64: return ForgeChar("double")
                if dt == np.float32: return ForgeChar("single")
                if dt == np.int32: return ForgeChar("int32")
                if dt == np.int64: return ForgeChar("int64")
                if dt == np.bool_: return ForgeChar("logical")
                if dt == np.complex128: return ForgeChar("double")
                return ForgeChar(str(dt))
            if isinstance(obj, (int, float)):
                return ForgeChar("double")
            if isinstance(obj, str):
                return ForgeChar("char")
            if isinstance(obj, bool):
                return ForgeChar("logical")
            return ForgeChar(type(obj).__name__)

        def forge_typecast_val(x, typename):
            """typecast to named type."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(typename, ForgeChar):
                typename = typename.to_str()
            if isinstance(x, ForgeArray):
                data = x.data
            else:
                data = np.atleast_1d(x)
            type_map = {
                "double": np.float64, "single": np.float32,
                "int8": np.int8, "int16": np.int16, "int32": np.int32, "int64": np.int64,
                "uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32, "uint64": np.uint64,
                "logical": np.bool_,
            }
            if typename in type_map:
                return ForgeArray(data.astype(type_map[typename]))
            return ForgeArray(data)

        session._engine.functions["sprintf"] = forge_sprintf
        session._engine.functions["error"] = forge_error
        session._engine.functions["warning"] = forge_warning
        session._engine.functions["assert"] = forge_assert
        session._engine.functions["inputname"] = forge_inputname
        session._engine.functions["class"] = forge_class
        session._engine.functions["typecast"] = forge_typecast_val

        # R110: Fast TIGA assembly built-ins
        def _findspan(n, p, u, U):
            """Find knot span (0-based)."""
            if u >= U[n+1]: return n
            if u <= U[p]: return p
            lo, hi = p, n + 1
            mid = (lo + hi) // 2
            while u < U[mid] or u >= U[mid + 1]:
                if u < U[mid]: hi = mid
                else: lo = mid
                mid = (lo + hi) // 2
            return mid

        def _basisfun(i, u, p, U):
            """B-spline basis functions (p+1 values)."""
            N = np.zeros(p + 1)
            N[0] = 1.0
            left = np.zeros(p + 1)
            right = np.zeros(p + 1)
            for j in range(1, p + 1):
                left[j] = u - U[i + 1 - j]
                right[j] = U[i + j] - u
                saved = 0.0
                for r in range(j):
                    temp = N[r] / (right[r + 1] + left[j - r])
                    N[r] = saved + right[r + 1] * temp
                    saved = left[j - r] * temp
                N[j] = saved
            return N

        def _derbasisfun(i, u, p, nd, U):
            """Basis function derivatives."""
            ndu = np.zeros((p + 1, p + 1))
            ndu[0, 0] = 1.0
            left = np.zeros(p + 1)
            right = np.zeros(p + 1)
            for j in range(1, p + 1):
                left[j] = u - U[i + 1 - j]
                right[j] = U[i + j] - u
                saved = 0.0
                for r in range(j):
                    ndu[j, r] = right[r + 1] + left[j - r]
                    temp = ndu[r, j - 1] / ndu[j, r]
                    ndu[r, j] = saved + right[r + 1] * temp
                    saved = left[j - r] * temp
                ndu[j, j] = saved
            ders = np.zeros((nd + 1, p + 1))
            for j in range(p + 1):
                ders[0, j] = ndu[j, p]
            a = np.zeros((2, p + 1))
            for r in range(p + 1):
                s1, s2 = 0, 1
                a[0, 0] = 1.0
                for k in range(1, nd + 1):
                    d = 0.0
                    rk, pk = r - k, p - k
                    if r >= k:
                        a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                        d = a[s2, 0] * ndu[rk, pk]
                    j1 = max(0, -(rk)) + 1 if rk < 0 else 1
                    j2 = min(k - 1, pk - r) if (r - 1) <= pk else k - 1
                    for j in range(j1, j2 + 1):
                        a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                        d += a[s2, j] * ndu[rk + j, pk]
                    if r <= pk:
                        a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                        d += a[s2, k] * ndu[r, pk]
                    ders[k, r] = d
                    s1, s2 = s2, s1
            r = p
            for k in range(1, nd + 1):
                for j in range(p + 1):
                    ders[k, j] *= r
                r *= (p - k)
            return ders

        def forge_tiga_assemble_2d(p_arr, Xi_r_arr, Xi_t_arr, CPx_arr, CPy_arr, Cw_arr, bc_str=None):
            """Fast 2D NURBS Laplacian assembly.

            [K, F, free_dofs] = tiga_assemble_2d(p, Xi_r, Xi_t, CPx, CPy, Cw)
            Returns stiffness matrix K, force vector F, and free DOF indices.
            """
            from forge.engine.types import ForgeArray
            p = int(p_arr.data.flat[0]) if isinstance(p_arr, ForgeArray) else int(p_arr)
            Xi_r = Xi_r_arr.data.flatten() if isinstance(Xi_r_arr, ForgeArray) else np.asarray(Xi_r_arr).flatten()
            Xi_t = Xi_t_arr.data.flatten() if isinstance(Xi_t_arr, ForgeArray) else np.asarray(Xi_t_arr).flatten()
            CPx = CPx_arr.data if isinstance(CPx_arr, ForgeArray) else np.asarray(CPx_arr)
            CPy = CPy_arr.data if isinstance(CPy_arr, ForgeArray) else np.asarray(CPy_arr)
            Cw = Cw_arr.data if isinstance(Cw_arr, ForgeArray) else np.asarray(Cw_arr)

            n_r = len(Xi_r) - p - 1
            n_t = len(Xi_t) - p - 1
            n_2d = n_r * n_t

            nqp = p + 2
            gp, gw = np.polynomial.legendre.leggauss(nqp)
            knots_r = np.unique(Xi_r)
            knots_t = np.unique(Xi_t)

            K = np.zeros((n_2d, n_2d))

            for er in range(len(knots_r) - 1):
                xi_a, xi_b = knots_r[er], knots_r[er + 1]
                if xi_b - xi_a < 1e-14: continue
                Jr = (xi_b - xi_a) / 2.0
                for et in range(len(knots_t) - 1):
                    eta_a, eta_b = knots_t[et], knots_t[et + 1]
                    if eta_b - eta_a < 1e-14: continue
                    Jt = (eta_b - eta_a) / 2.0
                    for qr in range(nqp):
                        xi = (xi_a + xi_b) / 2 + Jr * gp[qr]
                        span_r = _findspan(n_r - 1, p, xi, Xi_r)
                        ders_r = _derbasisfun(span_r, xi, p, 1, Xi_r)
                        Nr, dNr = ders_r[0], ders_r[1]
                        for qt in range(nqp):
                            eta = (eta_a + eta_b) / 2 + Jt * gp[qt]
                            span_t = _findspan(n_t - 1, p, eta, Xi_t)
                            ders_t = _derbasisfun(span_t, eta, p, 1, Xi_t)
                            Nt, dNt = ders_t[0], ders_t[1]
                            wt_q = gw[qr] * Jr * gw[qt] * Jt

                            W, dW_dxi, dW_deta = 0.0, 0.0, 0.0
                            for a in range(p + 1):
                                ir = span_r - p + a
                                for b in range(p + 1):
                                    it = span_t - p + b
                                    ww = Cw[ir, it]
                                    W += Nr[a] * Nt[b] * ww
                                    dW_dxi += dNr[a] * Nt[b] * ww
                                    dW_deta += Nr[a] * dNt[b] * ww

                            dx_dxi, dx_deta = 0.0, 0.0
                            dy_dxi, dy_deta = 0.0, 0.0
                            for a in range(p + 1):
                                ir = span_r - p + a
                                for b in range(p + 1):
                                    it = span_t - p + b
                                    ww = Cw[ir, it]
                                    dR_dxi = (dNr[a] * Nt[b] * ww * W - Nr[a] * Nt[b] * ww * dW_dxi) / W**2
                                    dR_deta = (Nr[a] * dNt[b] * ww * W - Nr[a] * Nt[b] * ww * dW_deta) / W**2
                                    dx_dxi += dR_dxi * CPx[ir, it]
                                    dx_deta += dR_deta * CPx[ir, it]
                                    dy_dxi += dR_dxi * CPy[ir, it]
                                    dy_deta += dR_deta * CPy[ir, it]

                            detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
                            if abs(detJ) < 1e-15: continue
                            inv_J = np.array([[dy_deta, -dy_dxi], [-dx_deta, dx_dxi]]) / detJ

                            dR_dx = np.zeros((p + 1, p + 1))
                            dR_dy = np.zeros((p + 1, p + 1))
                            glob = np.zeros((p + 1, p + 1), dtype=int)
                            for a in range(p + 1):
                                ir = span_r - p + a
                                for b in range(p + 1):
                                    it = span_t - p + b
                                    ww = Cw[ir, it]
                                    dr_dxi = (dNr[a] * Nt[b] * ww * W - Nr[a] * Nt[b] * ww * dW_dxi) / W**2
                                    dr_deta = (Nr[a] * dNt[b] * ww * W - Nr[a] * Nt[b] * ww * dW_deta) / W**2
                                    dR_dx[a, b] = inv_J[0, 0] * dr_dxi + inv_J[0, 1] * dr_deta
                                    dR_dy[a, b] = inv_J[1, 0] * dr_dxi + inv_J[1, 1] * dr_deta
                                    glob[a, b] = it * n_r + ir

                            for a in range(p + 1):
                                for b in range(p + 1):
                                    gA = glob[a, b]
                                    for c in range(p + 1):
                                        for d in range(p + 1):
                                            gB = glob[c, d]
                                            K[gA, gB] += (dR_dx[a, b] * dR_dx[c, d] + dR_dy[a, b] * dR_dy[c, d]) * abs(detJ) * wt_q

            # Apply BCs
            bc_dofs = set()
            for j in range(n_t):
                bc_dofs.add(j * n_r)
                bc_dofs.add(j * n_r + n_r - 1)
            free_dofs = sorted(set(range(n_2d)) - bc_dofs)

            return ForgeArray(K), ForgeArray(np.zeros(n_2d)), ForgeArray(np.array(free_dofs, dtype=float) + 1)  # 1-based

        session._engine.functions["tiga_assemble_2d"] = forge_tiga_assemble_2d

        # R113: Utility functions
        def forge_fieldnames(s):
            """fieldnames(struct) - return cell array of field names."""
            from forge.engine.containers import ForgeStruct, ForgeCell, ForgeChar
            if isinstance(s, ForgeStruct):
                names = list(s._fields.keys())
                return ForgeCell([ForgeChar(n) for n in names])
            return ForgeCell([])

        def forge_isfield(s, fname):
            """isfield(struct, name) - check if field exists."""
            from forge.engine.containers import ForgeStruct, ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(fname, ForgeChar):
                fname = fname.to_str()
            if isinstance(s, ForgeStruct):
                return ForgeArray(np.float64(1.0 if fname in s._fields else 0.0))
            return ForgeArray(np.float64(0.0))

        def forge_struct_func(*args):
            """struct(field1, val1, field2, val2, ...) - create struct."""
            from forge.engine.containers import ForgeStruct, ForgeChar
            s = ForgeStruct()
            i = 0
            while i + 1 < len(args):
                name = args[i]
                val = args[i + 1]
                if isinstance(name, ForgeChar):
                    name = name.to_str()
                s._fields[name] = val
                i += 2
            return s

        def forge_isstruct(x):
            """isstruct(x) - check if x is a struct."""
            from forge.engine.containers import ForgeStruct
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(1.0 if isinstance(x, ForgeStruct) else 0.0))

        def forge_iscell(x):
            """iscell(x) - check if x is a cell array."""
            from forge.engine.containers import ForgeCell
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(1.0 if isinstance(x, ForgeCell) else 0.0))

        def forge_ischar(x):
            """ischar(x) - check if x is a char array."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(1.0 if isinstance(x, ForgeChar) else 0.0))

        def forge_isnumeric(x):
            """isnumeric(x) - check if x is numeric."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar, ForgeCell, ForgeStruct
            if isinstance(x, ForgeChar) or isinstance(x, ForgeCell) or isinstance(x, ForgeStruct):
                return ForgeArray(np.float64(0.0))
            if isinstance(x, ForgeArray):
                return ForgeArray(np.float64(1.0 if x.data.dtype.kind in ('f', 'i', 'u', 'c') else 0.0))
            if isinstance(x, (int, float, complex)):
                return ForgeArray(np.float64(1.0))
            return ForgeArray(np.float64(0.0))

        def forge_islogical(x):
            """islogical(x) - check if x is logical."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                return ForgeArray(np.float64(1.0 if x.data.dtype == np.bool_ else 0.0))
            if isinstance(x, bool):
                return ForgeArray(np.float64(1.0))
            return ForgeArray(np.float64(0.0))

        def forge_exist(name, typ=None):
            """exist(name) - check if variable/function exists."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(name, ForgeChar):
                name = name.to_str()
            # Check functions
            if name in session._engine.functions:
                return ForgeArray(np.float64(5.0))  # 5 = built-in function
            # Check workspace
            if name in session._engine.workspace._vars:
                return ForgeArray(np.float64(1.0))  # 1 = variable
            return ForgeArray(np.float64(0.0))

        def forge_which(name):
            """which(name) - locate function."""
            from forge.engine.containers import ForgeChar
            if isinstance(name, ForgeChar):
                name = name.to_str()
            if name in session._engine.functions:
                return ForgeChar(f"built-in function: {name}")
            return ForgeChar("")

        def forge_methods(obj_or_name):
            """methods(name) - list methods of an object or class."""
            from forge.engine.containers import ForgeChar, ForgeCell
            return ForgeCell([])

        session._engine.functions["fieldnames"] = forge_fieldnames
        session._engine.functions["isfield"] = forge_isfield
        session._engine.functions["struct"] = forge_struct_func
        session._engine.functions["isstruct"] = forge_isstruct
        session._engine.functions["iscell"] = forge_iscell
        session._engine.functions["ischar"] = forge_ischar
        session._engine.functions["isnumeric"] = forge_isnumeric
        session._engine.functions["islogical"] = forge_islogical
        session._engine.functions["exist"] = forge_exist
        session._engine.functions["which"] = forge_which
        session._engine.functions["methods"] = forge_methods

        # R115: Linear algebra decompositions + conv2
        def forge_schur(A):
            """[U, T] = schur(A) - Schur decomposition."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import schur as _schur
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            T, U = _schur(data)
            return ForgeArray(T), ForgeArray(U)

        def forge_hess(A):
            """[P, H] = hess(A) - Hessenberg decomposition."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import hessenberg
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            H, P = hessenberg(data, calc_q=True)
            return ForgeArray(P), ForgeArray(H)

        def forge_balance(A):
            """[D, B] = balance(A) - diagonal scaling for eigenvalue computation."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import matrix_balance
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            B, T = matrix_balance(data)
            return ForgeArray(T), ForgeArray(B)

        def forge_conv2(A, B, *args):
            """C = conv2(A, B) - 2D convolution."""
            from forge.engine.types import ForgeArray
            from scipy.signal import convolve2d
            a_data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            b_data = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            mode = "full"
            from forge.engine.containers import ForgeChar
            for arg in args:
                if isinstance(arg, ForgeChar):
                    mode = arg.to_str()
                elif isinstance(arg, str):
                    mode = arg
            result = convolve2d(a_data, b_data, mode=mode)
            return ForgeArray(result)

        def forge_eigs_fixed(A, k=None, *args):
            """eigs(A, k) - compute k largest eigenvalues."""
            from forge.engine.types import ForgeArray
            from scipy.sparse.linalg import eigs as _eigs
            from scipy.sparse import issparse, csc_matrix
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            if k is None:
                k = min(6, data.shape[0] - 2)
            elif isinstance(k, ForgeArray):
                k = int(k.data.flat[0])
            k = min(k, data.shape[0] - 2)
            if k < 1:
                k = 1
            try:
                if not issparse(data):
                    data = csc_matrix(data)
                vals, vecs = _eigs(data, k=k)
                return ForgeArray(np.sort(np.real(vals))[::-1])
            except Exception as e:
                # Fallback to dense eigenvalues
                if issparse(data):
                    data = data.toarray()
                ev = np.linalg.eigvals(data)
                ev = np.sort(np.real(ev))[::-1]
                return ForgeArray(ev[:k])

        def forge_expm(A):
            """expm(A) - matrix exponential."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import expm as _expm
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(_expm(data))

        def forge_logm(A):
            """logm(A) - matrix logarithm."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import logm as _logm
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            result = _logm(data)
            if np.isrealobj(data) and np.allclose(result.imag, 0):
                result = result.real
            return ForgeArray(result)

        def forge_funm(A, func):
            """funm(A, @f) - general matrix function."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import funm as _funm
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            if callable(func):
                result = _funm(data, func)
                return ForgeArray(result)
            return ForgeArray(data)

        def forge_condest(A):
            """condest(A) - 1-norm condition number estimate."""
            from forge.engine.types import ForgeArray
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.float64(np.linalg.cond(data, 1)))

        session._engine.functions["schur"] = forge_schur
        session._engine.functions["hess"] = forge_hess
        session._engine.functions["balance"] = forge_balance
        session._engine.functions["conv2"] = forge_conv2
        session._engine.functions["eigs"] = forge_eigs_fixed
        session._engine.functions["expm"] = forge_expm
        session._engine.functions["logm"] = forge_logm
        session._engine.functions["funm"] = forge_funm
        session._engine.functions["condest"] = forge_condest





