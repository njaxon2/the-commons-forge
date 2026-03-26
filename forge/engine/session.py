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





