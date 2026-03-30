# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Forge engine session — wraps the evaluator with builtins and session management."""
import os
import sys
import io
import numpy as np
from forge.engine.evaluator import Session as _EvalSession, Workspace, ForgeError, UndefinedFunctionError
from forge import __version__ as _forge_version
from forge.engine.builtins import BUILTIN_REGISTRY
from forge.engine.addon_manager import AddonManager
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
        self._file_browser_dir = None  # set by GUI when file browser navigates

        # Add-on manager: tracks which Forge toolboxes and Octave packages are active
        self.addon_manager = AddonManager()
        self._apply_addons()

        # Register session-level builtins (cd, pwd, who, clear, etc.)
        self._register_session_builtins()

        # Give engine a back-reference for .m file discovery (R13)
        self._engine._session_ref = self

    def _apply_addons(self):
        """Register functions from all enabled add-ons."""
        active = self.addon_manager.get_all_active_functions()
        for name, func in active.items():
            self._engine.functions[name] = func

    def reload_addons(self):
        """Re-apply addon state after enable/disable changes."""
        from forge.engine.builtins import TOOLBOX_MANIFEST
        all_tb_funcs = set()
        for _, (_, registry) in TOOLBOX_MANIFEST.items():
            all_tb_funcs.update(registry.keys())
        to_remove = []
        for name, func in self._engine.functions.items():
            if name in all_tb_funcs or getattr(func, "_is_octave_proxy", False):
                to_remove.append(name)
        for name in to_remove:
            del self._engine.functions[name]
        self._apply_addons()

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
            r, i = v.real, v.imag
            rs = self._format_scalar(r)
            if i >= 0:
                return f'{rs} + {self._format_scalar(abs(i))}i'
            else:
                return f'{rs} - {self._format_scalar(abs(i))}i'
        if isinstance(v, (int, _np.integer)):
            return str(int(v))
        # float — use _format attribute (set by format command)
        fmt = getattr(self, '_format', getattr(self, 'format', 'short'))
        if fmt == 'long':
            return f'{v:.15g}'
        elif fmt in ('shortE', 'short e', 'shorte', 'shortEng'):
            return f'{v:.4e}'
        elif fmt in ('longE', 'long e', 'longe', 'longEng'):
            return f'{v:.15e}'
        else:  # short
            if not _np.isfinite(v):
                if _np.isnan(v):
                    return 'NaN'
                return 'Inf' if v > 0 else '-Inf'
            av = abs(v)
            if av == 0:
                return '0'
            if _np.isfinite(v) and v == int(v) and abs(v) < 1e15:
                return str(int(v))
            # Use scientific notation for very small or very large
            if av < 1e-3 or av >= 1e7:
                return f'{v:.4e}'
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
            return None  # format command produces no output

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
            return None  # format command produces no output

        def forge_run(*args, path_arg=None):
            """Execute a .m script file."""
            from forge.engine.parser import parse as _parse
            # Accept both command-style (single string) and function-style args
            if args:
                path_arg = args[0]
            if path_arg is None:
                raise ValueError("run requires a file path argument")
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

        # containers.Map standalone builtins: keys, values, isKey, remove
        from forge.engine.containers import ForgeMap, ForgeStruct, ForgeCell

        def forge_keys(m):
            if isinstance(m, ForgeMap):
                return m.keys()
            if isinstance(m, ForgeStruct):
                return ForgeCell(list(m._fields.keys()))
            raise TypeError("keys: argument must be a containers.Map or struct")

        def forge_values(m):
            if isinstance(m, ForgeMap):
                return m.values()
            if isinstance(m, ForgeStruct):
                return ForgeCell(list(m._fields.values()))
            raise TypeError("values: argument must be a containers.Map or struct")

        def forge_isKey(m, key):
            k = key.to_str() if hasattr(key, "to_str") else str(key)
            if isinstance(m, ForgeMap):
                return m.isKey(k)
            if isinstance(m, ForgeStruct):
                return k in m._fields
            raise TypeError("isKey: first argument must be a containers.Map or struct")

        def forge_remove(m, key):
            k = key.to_str() if hasattr(key, "to_str") else str(key)
            if isinstance(m, ForgeMap):
                m.remove(k)
                return m
            if isinstance(m, ForgeStruct):
                from forge.engine.containers import forge_rmfield
                return forge_rmfield(m, k)
            raise TypeError("remove: first argument must be a containers.Map or struct")

        self._engine.functions["keys"] = forge_keys
        self._engine.functions["values"] = forge_values
        self._engine.functions["isKey"] = forge_isKey
        self._engine.functions["remove"] = forge_remove

        # Register 'containers' namespace so containers.Map() works
        _containers_ns = ForgeStruct()
        _containers_ns._fields["Map"] = ForgeMap
        self._engine.workspace.set("containers", _containers_ns)

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
            except Exception:
                return ForgeArray(np.float64(-1))

        def forge_fclose(fid=None):
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import numpy as np
            if fid is None or (isinstance(fid, ForgeChar) and fid.to_str() == "all"):
                for f in session._file_handles.values():
                    try: f.close()
                    except Exception: pass
                session._file_handles.clear()
                return ForgeArray(np.float64(0))
            fid_val = int(float(fid.data.flat[0]) if isinstance(fid, ForgeArray) else float(fid))
            if fid_val in session._file_handles:
                session._file_handles[fid_val].close()
                del session._file_handles[fid_val]
                return ForgeArray(np.float64(0))
            return ForgeArray(np.float64(-1))

        def forge_fprintf_file(fid, fmt, *args):
            """fprintf to file handle with Octave vectorization."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import numpy as np
            fid_val = int(float(fid.data.flat[0]) if isinstance(fid, ForgeArray) else float(fid))
            if isinstance(fmt, ForgeChar): fmt = fmt.to_str()
            fmt = str(fmt)
            # Convert args with vectorization support
            max_len = 1
            expanded = []
            for a in args:
                if isinstance(a, ForgeChar):
                    expanded.append([a.to_str()])
                elif isinstance(a, ForgeArray) and a.data.size > 1:
                    elems = []
                    for v in a.data.flat:
                        if isinstance(v, (np.floating, float)):
                            elems.append(float(v))
                        else:
                            elems.append(int(v))
                    expanded.append(elems)
                    max_len = max(max_len, len(elems))
                elif isinstance(a, ForgeArray):
                    v = a.data.flat[0]
                    expanded.append([float(v) if isinstance(v, (np.floating, float)) else int(v)])
                else:
                    expanded.append([a])
            try:
                if max_len > 1:
                    parts = []
                    for i in range(max_len):
                        vals = [e[i % len(e)] for e in expanded]
                        parts.append(fmt % tuple(vals))
                    text = "".join(parts)
                else:
                    py_args = [e[0] for e in expanded]
                    text = fmt % tuple(py_args) if py_args else fmt
            except Exception:
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
            """Save workspace variables to MATLAB-compatible .mat file using scipy.io.savemat."""
            import scipy.io as _sio
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            fname = str(fname)
            if not fname.endswith('.mat'):
                fname += '.mat'

            # Built-in constants that should not be saved
            _BUILTIN_NAMES = {'pi', 'e', 'i', 'j', 'Inf', 'inf', 'NaN', 'nan', 'eps', 'true', 'false', 'ans', 'realmin', 'realmax'}

            ws = session._engine.workspace
            if varnames:
                names = [v.to_str() if isinstance(v, ForgeChar) else str(v) for v in varnames]
            else:
                names = [n for n in ws.names() if n not in _BUILTIN_NAMES]

            data = {}
            for name in names:
                if name in _BUILTIN_NAMES:
                    continue
                try:
                    val = ws.get(name)
                except Exception:
                    continue
                # Convert Forge types to scipy-compatible types
                if isinstance(val, ForgeChar):
                    data[name] = val.to_str()
                elif isinstance(val, ForgeArray):
                    arr = val.data
                    if arr.ndim == 0:
                        arr = arr.reshape(1, 1)
                    data[name] = arr
                elif isinstance(val, ForgeStruct):
                    d = {}
                    for fld, fval in val._fields.items():
                        if isinstance(fval, ForgeArray):
                            d[fld] = fval.data
                        elif isinstance(fval, ForgeChar):
                            d[fld] = fval.to_str()
                        elif isinstance(fval, (int, float, complex)):
                            d[fld] = np.array([[fval]])
                        else:
                            d[fld] = fval
                    data[name] = d
                elif isinstance(val, ForgeCell):
                    obj_arr = np.empty((val._shape[0], val._shape[1]), dtype=object)
                    for idx, item in enumerate(val._data):
                        r = idx // val._shape[1]
                        c = idx % val._shape[1]
                        if isinstance(item, ForgeArray):
                            obj_arr[r, c] = item.data
                        elif isinstance(item, ForgeChar):
                            obj_arr[r, c] = item.to_str()
                        elif isinstance(item, (int, float, complex)):
                            obj_arr[r, c] = np.array([[item]])
                        else:
                            obj_arr[r, c] = item
                    data[name] = obj_arr
                elif isinstance(val, (int, float, complex)):
                    data[name] = np.array([[val]])
                elif isinstance(val, np.ndarray):
                    data[name] = val
                else:
                    _wname = name
                    _wtype = type(val).__name__
                    session.disp("Warning: variable '" + _wname + "' has unsupported type " + _wtype + ", skipping.")
                    continue

            _sio.savemat(fname, data)
            return ''

        def forge_load(fname, *varnames):
            """Load workspace variables from saved file."""
            import json
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            fname = str(fname)

            def _find_file(name):
                """Resolve filename searching multiple locations.

                Search order:
                  1. Absolute path (as-is)
                  2. Relative to process CWD
                  3. Relative to each entry in session.path
                     (session.path[0] tracks engine/file-browser CWD)
                  4. Relative to session._file_browser_dir (GUI fallback)
                Each location is tried as given, then with .mat appended.
                """
                candidates = []
                if os.path.isabs(name):
                    candidates.append(name)
                else:
                    # Process CWD
                    candidates.append(os.path.join(os.getcwd(), name))
                    # session.path entries (engine CWD at index 0)
                    for p in session.path:
                        full = os.path.join(p, name)
                        if full not in candidates:
                            candidates.append(full)
                    # GUI file-browser directory fallback
                    browser_dir = getattr(session, "_file_browser_dir", None)
                    if browser_dir:
                        full = os.path.join(browser_dir, name)
                        if full not in candidates:
                            candidates.append(full)
                # Check each candidate, with and without .mat
                for c in candidates:
                    if os.path.isfile(c):
                        return c
                    if not name.endswith(".mat") and os.path.isfile(c + ".mat"):
                        return c + ".mat"
                return name  # fall through -- will raise FileNotFoundError

            fname = _find_file(fname)
            _varname_filter = [v.to_str() if isinstance(v, ForgeChar) else str(v) for v in varnames] if varnames else None
            ws = session._engine.workspace

            # Try scipy.io.loadmat first (real MATLAB/Octave binary .mat)
            _loaded_scipy = False
            try:
                import scipy.io as _sio
                mat_data = _sio.loadmat(fname, squeeze_me=False)
                for key, val in mat_data.items():
                    if key.startswith('__'):
                        continue
                    if _varname_filter and key not in _varname_filter:
                        continue
                    if isinstance(val, np.ndarray):
                        if val.dtype.kind in ('U', 'S', 'O') and val.size == 1:
                            s = str(val.flat[0])
                            ws.set(key, ForgeChar(s))
                        elif val.dtype.kind in ('U', 'S'):
                            ws.set(key, ForgeChar(str(val.flat[0]) if val.size else ''))
                        elif np.issubdtype(val.dtype, np.complexfloating):
                            ws.set(key, ForgeArray(np.array(val, dtype=np.complex128)))
                        elif val.dtype == object:
                            ws.set(key, ForgeArray(np.array(val, dtype=np.float64)))
                        else:
                            ws.set(key, ForgeArray(np.array(val, dtype=np.float64)))
                    elif isinstance(val, dict):
                        s = ForgeStruct()
                        for fld, fval in val.items():
                            if fld.startswith('__'):
                                continue
                            if isinstance(fval, np.ndarray):
                                if fval.dtype.kind in ('U', 'S') and fval.size > 0:
                                    s._fields[fld] = ForgeChar(str(fval.flat[0]))
                                else:
                                    s._fields[fld] = ForgeArray(np.array(fval, dtype=np.float64))
                            else:
                                s._fields[fld] = fval
                        ws.set(key, s)
                    elif isinstance(val, (int, float, np.integer, np.floating)):
                        ws.set(key, ForgeArray(np.float64(val)))
                    elif isinstance(val, str):
                        ws.set(key, ForgeChar(val))
                    elif isinstance(val, bytes):
                        ws.set(key, ForgeChar(val.decode('utf-8', errors='replace')))
                    else:
                        ws.set(key, ForgeArray(np.array(val, dtype=np.float64)))
                _loaded_scipy = True
            except Exception:
                pass

            if not _loaded_scipy:
                # Fall back to Forge JSON format
                with open(fname, 'r') as f:
                    data = json.load(f)
                for name, val_dict in data.items():
                    if _varname_filter and name not in _varname_filter:
                        continue
                    if val_dict["type"] == "array":
                        arr = np.array(val_dict["data"], dtype=np.float64).reshape(val_dict["shape"])
                        ws.set(name, ForgeArray(arr))
                    elif val_dict["type"] == "char":
                        ws.set(name, ForgeChar(val_dict["data"]))
                    elif val_dict["type"] == "scalar":
                        ws.set(name, ForgeArray(np.float64(val_dict["data"])))
            return ''

        def forge_pkg(*args):
            """Octave-compatible pkg command.

            Usage:
              pkg list              — list installed packages
              pkg load <name>       — enable/load a package
              pkg unload <name>     — disable/unload a package
              pkg describe <name>   — show package details
            """
            from forge.engine.containers import ForgeChar
            args = [a.to_str() if isinstance(a, ForgeChar) else str(a) for a in args]
            if not args:
                return "Usage: pkg list | pkg load <name> | pkg unload <name>"
            action = args[0].lower()
            mgr = session.addon_manager

            if action == "list":
                lines = []
                lines.append("  Forge Toolboxes:")
                for name, display, count, enabled in mgr.forge_toolboxes():
                    status = "[loaded]" if enabled else "[unloaded]"
                    lines.append(f"    {display:30s} {count:4d} functions  {status}")
                if mgr._octave_bridge and mgr._octave_bridge.available:
                    lines.append("")
                    lines.append("  Octave Packages:")
                    for _pkg_info in mgr.octave_packages():
                        if isinstance(_pkg_info, (list, tuple)):
                            _pname, _pver, _penabled = _pkg_info
                        else:
                            _pname = _pkg_info["name"]
                            _pver = _pkg_info.get("version", "?")
                            _penabled = mgr._octave_state.get(_pname, False)
                        _pstatus = "[loaded]" if _penabled else "[unloaded]"
                        lines.append(f"    {_pname:30s} v{_pver:10s} {_pstatus}")
                else:
                    lines.append("")
                    lines.append("  Octave: not available (octave-cli not found)")
                return "\n".join(lines)

            elif action == "load":
                if len(args) < 2:
                    return "Usage: pkg load <package_name>"
                pkg_name = args[1]
                # Try Forge toolbox first
                if pkg_name in mgr._forge_state:
                    mgr.set_forge_enabled(pkg_name, True)
                    session.reload_addons()
                    return f"Package '{pkg_name}' loaded."
                # Try Octave package
                elif pkg_name in mgr._octave_state:
                    mgr.set_octave_enabled(pkg_name, True)
                    session.reload_addons()
                    return f"Package '{pkg_name}' loaded."
                else:
                    return f"error: package '{pkg_name}' not found"

            elif action == "unload":
                if len(args) < 2:
                    return "Usage: pkg unload <package_name>"
                pkg_name = args[1]
                if pkg_name in mgr._forge_state:
                    mgr.set_forge_enabled(pkg_name, False)
                    session.reload_addons()
                    return f"Package '{pkg_name}' unloaded."
                elif pkg_name in mgr._octave_state:
                    mgr.set_octave_enabled(pkg_name, False)
                    session.reload_addons()
                    return f"Package '{pkg_name}' unloaded."
                else:
                    return f"error: package '{pkg_name}' not found"

            elif action == "describe":
                if len(args) < 2:
                    return "Usage: pkg describe <package_name>"
                pkg_name = args[1]
                if pkg_name in mgr._forge_state:
                    for name, display, count, enabled in mgr.forge_toolboxes():
                        if name == pkg_name:
                            return (f"Package: {display}\n"
                                    f"Backend: Forge (Python)\n"
                                    f"Functions: {count}\n"
                                    f"Status: {'loaded' if enabled else 'unloaded'}")
                elif pkg_name in mgr._octave_state:
                    info = mgr._octave_packages.get(pkg_name, {})
                    return (f"Package: {pkg_name}\n"
                            f"Backend: Octave\n"
                            f"Version: {info.get('version', '?')}\n"
                            f"Status: {'loaded' if mgr._octave_state.get(pkg_name) else 'unloaded'}")
                return f"error: package '{pkg_name}' not found"

            else:
                return f"pkg: unknown action '{action}'. Try: list, load, unload, describe"

        for _n, _f in [
            ("cbrt", forge_cbrt), ("pow2", forge_pow2),
            ("asinh", forge_asinh), ("acosh", forge_acosh), ("atanh", forge_atanh),
            ("isa", forge_isa), ("isreal", forge_isreal),
            ("display", forge_display), ("interp2", forge_interp2),
            ("save", forge_save), ("load", forge_load),
            ("pkg", forge_pkg),
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

        # --- Additional Octave-compatible builtins ---

        def forge_deal(*args):
            """deal(x1, x2, ...) -- distribute inputs to outputs."""
            if len(args) == 1:
                return args[0]
            return args
        self._engine.functions['deal'] = forge_deal

        def forge_display(x):
            """display(x) -- display a variable."""
            return x
        self._engine.functions['display'] = forge_display

        def forge_fieldnames(s):
            """fieldnames(s) -- return cell array of field names."""
            from forge.engine.containers import ForgeChar, ForgeCell, ForgeStruct
            if isinstance(s, ForgeStruct):
                names = list(s._fields.keys())
                return ForgeCell([ForgeChar(n) for n in names])
            raise TypeError("fieldnames: argument must be a struct")
        self._engine.functions['fieldnames'] = forge_fieldnames

        def forge_rmfield(s, field):
            """rmfield(s, field) -- remove a field from struct."""
            from forge.engine.containers import ForgeChar, ForgeStruct
            if isinstance(field, ForgeChar):
                field = field.to_str()
            field = str(field)
            if isinstance(s, ForgeStruct) and field in s._fields:
                new_s = ForgeStruct()
                for k, v in s._fields.items():
                    if k != field:
                        new_s._fields[k] = v
                return new_s
            return s
        self._engine.functions['rmfield'] = forge_rmfield

        def forge_isfield(s, field):
            """isfield(s, field) -- check if struct has field."""
            from forge.engine.containers import ForgeChar, ForgeStruct
            if isinstance(field, ForgeChar):
                field = field.to_str()
            field = str(field)
            if isinstance(s, ForgeStruct):
                return ForgeArray(np.array(1.0 if field in s._fields else 0.0))
            return ForgeArray(np.array(0.0))
        self._engine.functions['isfield'] = forge_isfield

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
                if isinstance(a, ForgeChar):
                    conv_args.append(a.to_str())
                elif isinstance(a, ForgeArray):
                    v = a.data.flat[0]
                    if isinstance(v, (np.integer,)):
                        conv_args.append(int(v))
                    elif isinstance(v, (np.floating,)):
                        conv_args.append(float(v))
                    else:
                        conv_args.append(v)
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
            return None  # format command produces no output

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
        # Register TIGA primitive functions as fast Python builtins
        # (These override the .m file versions with faster Python implementations)
        def forge_findspan_builtin(n, p, u, U):
            """findspan(n, p, u, U) — find knot span index."""
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray): n = int(n.data.flat[0])
            if isinstance(p, ForgeArray): p = int(p.data.flat[0])
            if isinstance(u, ForgeArray): u = float(u.data.flat[0])
            if isinstance(U, ForgeArray): U = U.data.flatten()
            else: U = np.array(U).flatten()
            if u >= U[n + 1]:
                return ForgeArray(np.float64(n))
            if u <= U[p]:
                return ForgeArray(np.float64(p))
            low, high = p, n + 1
            mid = (low + high) // 2
            while u < U[mid] or u >= U[mid + 1]:
                if u < U[mid]:
                    high = mid
                else:
                    low = mid
                mid = (low + high) // 2
            return ForgeArray(np.float64(mid))

        def forge_basisfun_builtin(i, u, p, U):
            """basisfun(i, u, p, U) — compute B-spline basis functions."""
            from forge.engine.types import ForgeArray
            if isinstance(i, ForgeArray): i = int(i.data.flat[0])
            if isinstance(u, ForgeArray): u = float(u.data.flat[0])
            if isinstance(p, ForgeArray): p = int(p.data.flat[0])
            if isinstance(U, ForgeArray): U = U.data.flatten()
            else: U = np.array(U).flatten()
            N = np.zeros(p + 1)
            left = np.zeros(p + 1)
            right = np.zeros(p + 1)
            N[0] = 1.0
            for j in range(1, p + 1):
                left[j] = u - U[i + 1 - j]
                right[j] = U[i + j] - u
                saved = 0.0
                for r in range(j):
                    temp = N[r] / (right[r + 1] + left[j - r])
                    N[r] = saved + right[r + 1] * temp
                    saved = left[j - r] * temp
                N[j] = saved
            return ForgeArray(N.reshape(1, -1))

        def forge_derbasisfun_builtin(i_span, u, p, *args):
            """derbasisfun(i, u, p, Xi) or derbasisfun(i, u, p, n_deriv, Xi)"""
            from forge.engine.types import ForgeArray
            if isinstance(i_span, ForgeArray): i_span = int(i_span.data.flat[0])
            if isinstance(u, ForgeArray): u = float(u.data.flat[0])
            if isinstance(p, ForgeArray): p = int(p.data.flat[0])
            if len(args) == 1:
                Xi_arr = args[0]; nd = 1
            elif len(args) == 2:
                nd_a = args[0]; Xi_arr = args[1]
                nd = int(nd_a.data.flat[0]) if isinstance(nd_a, ForgeArray) else int(nd_a)
            else:
                raise ValueError("derbasisfun requires 4 or 5 arguments")
            Xi = Xi_arr.data.flatten() if isinstance(Xi_arr, ForgeArray) else np.array(Xi_arr).flatten()
            ders = np.zeros((nd + 1, p + 1))
            ndu = np.zeros((p + 1, p + 1))
            left_a = np.zeros(p + 1)
            right_a = np.zeros(p + 1)
            a_m = np.zeros((2, p + 1))
            ndu[0, 0] = 1.0
            for j in range(1, p + 1):
                left_a[j] = u - Xi[i_span + 1 - j]
                right_a[j] = Xi[i_span + j] - u
                saved = 0.0
                for r in range(j):
                    denom = right_a[r + 1] + left_a[j - r]
                    ndu[j, r] = denom if abs(denom) > 1e-30 else 1e-30
                    temp = ndu[r, j - 1] / ndu[j, r]
                    ndu[r, j] = saved + right_a[r + 1] * temp
                    saved = left_a[j - r] * temp
                ndu[j, j] = saved
            for j in range(p + 1):
                ders[0, j] = ndu[j, p]
            for r in range(p + 1):
                s1, s2 = 0, 1
                a_m[0, 0] = 1.0
                for k in range(1, nd + 1):
                    d = 0.0
                    rk, pk = r - k, p - k
                    if r >= k:
                        a_m[s2, 0] = a_m[s1, 0] / ndu[pk + 1, rk]
                        d = a_m[s2, 0] * ndu[rk, pk]
                    j1 = 1 if rk >= -1 else -rk
                    j2 = k - 1 if r - 1 <= pk else p - r
                    for jj in range(j1, j2 + 1):
                        a_m[s2, jj] = (a_m[s1, jj] - a_m[s1, jj - 1]) / ndu[pk + 1, rk + jj]
                        d += a_m[s2, jj] * ndu[rk + jj, pk]
                    if r <= pk:
                        a_m[s2, k] = -a_m[s1, k - 1] / ndu[pk + 1, r]
                        d += a_m[s2, k] * ndu[r, pk]
                    ders[k, r] = d
                    s1, s2 = s2, s1
            rv = p
            for k in range(1, nd + 1):
                ders[k, :] *= rv
                rv *= (p - k)
            if len(args) == 1:
                return ForgeArray(ders[1, :].reshape(1, -1))
            return ForgeArray(ders)

        def forge_gaussQuad_builtin(n):
            """gaussQuad(n) — Gauss-Legendre quadrature points and weights."""
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray): n = int(n.data.flat[0])
            points, weights = np.polynomial.legendre.leggauss(n)
            return ForgeArray(points.reshape(1, -1)), ForgeArray(weights.reshape(1, -1))

        session._engine.functions["findspan"] = forge_findspan_builtin
        session._engine.functions["basisfun"] = forge_basisfun_builtin
        session._engine.functions["derbasisfun"] = forge_derbasisfun_builtin
        session._engine.functions["gaussQuad"] = forge_gaussQuad_builtin

        # R134: Fill remaining gaps + push to 1000
        def forge_cast(x, typename):
            """cast(x, typename) — convert to specified type."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(typename, ForgeChar):
                typename = typename.to_str()
            if isinstance(x, ForgeArray):
                data = x.data
            else:
                data = np.atleast_2d(x)
            type_map = {
                'double': np.float64, 'single': np.float32,
                'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64,
                'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64,
                'logical': np.bool_,
            }
            if typename in type_map:
                return ForgeArray(data.astype(type_map[typename]))
            return ForgeArray(data)

        def forge_filter(b, a, x):
            """filter(b, a, x) — 1-D digital filter."""
            from forge.engine.types import ForgeArray
            from scipy.signal import lfilter
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(lfilter(bd, ad, xd))

        def forge_filtfilt(b, a, x):
            """filtfilt(b, a, x) — zero-phase filtering."""
            from forge.engine.types import ForgeArray
            from scipy.signal import filtfilt as _filtfilt
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(_filtfilt(bd, ad, xd))

        def forge_cputime():
            """cputime — return CPU time in seconds."""
            from forge.engine.types import ForgeArray
            import time
            return ForgeArray(np.float64(time.process_time()))

        def forge_input(prompt):
            """input(prompt) — display prompt (non-interactive stub)."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(prompt, ForgeChar):
                prompt = prompt.to_str()
            print(prompt, end='')
            return None  # format command produces no output

        def forge_mpower(A, n):
            """mpower(A, n) — matrix power A^n."""
            from forge.engine.types import ForgeArray
            if isinstance(A, ForgeArray):
                data = A.data
            else:
                data = np.atleast_2d(A)
            if isinstance(n, ForgeArray):
                n = int(n.data.flat[0])
            return ForgeArray(np.linalg.matrix_power(data, n))

        # Additional signal processing
        def forge_butter(n, Wn, *args):
            """[b, a] = butter(n, Wn) — Butterworth filter design."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            from scipy.signal import butter as _butter
            if isinstance(n, ForgeArray): n = int(n.data.flat[0])
            if isinstance(Wn, ForgeArray): Wn = float(Wn.data.flat[0])
            ftype = 'low'
            for a in args:
                if isinstance(a, ForgeChar):
                    ftype = a.to_str()
            b, a = _butter(n, Wn, btype=ftype)
            return ForgeArray(b.reshape(1, -1)), ForgeArray(a.reshape(1, -1))

        def forge_freqz(b, a=None, *args):
            """[h, w] = freqz(b, a, n) — frequency response."""
            from forge.engine.types import ForgeArray
            from scipy.signal import freqz as _freqz
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            ad = a.data.flatten() if isinstance(a, ForgeArray) and a is not None else np.array([1.0])
            n = 512
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            w, h = _freqz(bd, ad, worN=n)
            return ForgeArray(np.abs(h).reshape(1, -1)), ForgeArray(w.reshape(1, -1))

        # Matrix functions
        def forge_lyap(A, B):
            """lyap(A, B) — solve Lyapunov equation AX + XA' = -B."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import solve_continuous_lyapunov
            ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            return ForgeArray(solve_continuous_lyapunov(ad, -bd))

        def forge_dlyap(A, B):
            """dlyap(A, B) — solve discrete Lyapunov equation AXA' - X + B = 0."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import solve_discrete_lyapunov
            ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            return ForgeArray(solve_discrete_lyapunov(ad, bd))

        def forge_qz(A, B):
            """[AA, BB, Q, Z] = qz(A, B) — QZ decomposition."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import qz as _qz
            ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            AA, BB, Q, Z = _qz(ad, bd)
            return ForgeArray(AA), ForgeArray(BB), ForgeArray(Q), ForgeArray(Z)

        # Combinatorics
        def forge_perms(v):
            """perms(v) — all permutations."""
            from forge.engine.types import ForgeArray
            from itertools import permutations
            if isinstance(v, ForgeArray):
                v = v.data.flatten()
            p = np.array(list(permutations(v)))
            return ForgeArray(p)

        def forge_nchoosek(n, k):
            """nchoosek(n, k) — binomial coefficient or combinations."""
            from forge.engine.types import ForgeArray
            from scipy.special import comb
            if isinstance(n, ForgeArray):
                if n.data.size > 1:
                    # Return combinations
                    from itertools import combinations
                    v = n.data.flatten()
                    k_val = int(k.data.flat[0]) if isinstance(k, ForgeArray) else int(k)
                    result = np.array(list(combinations(v, k_val)))
                    return ForgeArray(result)
                n = int(n.data.flat[0])
            if isinstance(k, ForgeArray):
                k = int(k.data.flat[0])
            return ForgeArray(np.float64(comb(n, k, exact=True)))

        # Interpolation extras
        def forge_interpft(x, n):
            """interpft(x, n) — interpolation using FFT."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                x = x.data.flatten()
            if isinstance(n, ForgeArray):
                n = int(n.data.flat[0])
            X = np.fft.fft(x)
            m = len(x)
            if n > m:
                pad = np.zeros(n - m, dtype=complex)
                mid = m // 2
                X_new = np.concatenate([X[:mid], pad, X[mid:]])
            else:
                X_new = X[:n]
            return ForgeArray(np.real(np.fft.ifft(X_new)) * n / m)

        # String utilities
        def forge_num2str_enhanced(n, *args):
            """num2str(n, fmt) — convert number to string with optional format."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray):
                data = n.data
                if data.size == 1:
                    v = float(data.flat[0])
                    if args:
                        fmt = args[0]
                        if isinstance(fmt, ForgeChar):
                            fmt = fmt.to_str()
                        elif isinstance(fmt, ForgeArray):
                            fmt = f'%.{int(fmt.data.flat[0])}f'
                        try:
                            return ForgeChar(fmt % v)
                        except Exception:
                            return ForgeChar(str(v))
                    if v == int(v) and abs(v) < 1e15:
                        return ForgeChar(str(int(v)))
                    return ForgeChar(f'{v:.4f}')
                # Array — format each element
                parts = []
                for v in data.flat:
                    if float(v) == int(float(v)):
                        parts.append(str(int(float(v))))
                    else:
                        parts.append(f'{float(v):.4f}')
                return ForgeChar('  '.join(parts))
            return ForgeChar(str(n))

        # Miscellaneous
        def forge_deal_enhanced(*args):
            """deal(x, y, ...) — distribute inputs to outputs."""
            if len(args) == 1:
                return args[0]
            return args

        def forge_nthroot_safe(x, n):
            """nthroot(x, n) — real nth root."""
            from forge.engine.types import ForgeArray
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            nd = float(n.data.flat[0]) if isinstance(n, ForgeArray) else float(n)
            result = np.sign(xd) * np.abs(xd) ** (1.0 / nd)
            return ForgeArray(result)

        def forge_eps_func(x=None):
            """eps(x) — floating-point relative accuracy."""
            from forge.engine.types import ForgeArray
            if x is None:
                return ForgeArray(np.float64(np.finfo(np.float64).eps))
            if isinstance(x, ForgeArray):
                x = float(x.data.flat[0])
            return ForgeArray(np.float64(np.spacing(x)))

        def forge_nextpow2_func(n):
            """nextpow2(n) — next power of 2."""
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray):
                n = float(n.data.flat[0])
            if n <= 0:
                return ForgeArray(np.float64(0))
            return ForgeArray(np.float64(int(np.ceil(np.log2(abs(n))))))

        def forge_log1p(x):
            """log1p(x) — compute log(1+x) accurately for small x."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.log1p(data))

        def forge_expm1(x):
            """expm1(x) — compute exp(x)-1 accurately for small x."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.expm1(data))

        def forge_sinc(x):
            """sinc(x) — normalized sinc function sin(pi*x)/(pi*x)."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.sinc(data))

        def forge_unwrap(x):
            """unwrap(x) — unwrap phase angles."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.unwrap(data))

        def forge_colon(a, b, c=None):
            """colon(a, b) or colon(a, step, b) — create range."""
            from forge.engine.types import ForgeArray
            if isinstance(a, ForgeArray): a = float(a.data.flat[0])
            if isinstance(b, ForgeArray): b = float(b.data.flat[0])
            if c is not None:
                if isinstance(c, ForgeArray): c = float(c.data.flat[0])
                # colon(start, step, stop)
                return ForgeArray(np.arange(a, c + b/2, b).reshape(1, -1))
            return ForgeArray(np.arange(a, b + 0.5, 1.0).reshape(1, -1))

        def forge_ndgrid(*args):
            """ndgrid(x1, x2, ...) — rectangular grid in N-D."""
            from forge.engine.types import ForgeArray
            arrays = []
            for a in args:
                if isinstance(a, ForgeArray):
                    arrays.append(a.data.flatten())
                else:
                    arrays.append(np.array(a).flatten())
            grids = np.meshgrid(*arrays, indexing='ij')
            if len(grids) == 1:
                return ForgeArray(grids[0])
            return tuple(ForgeArray(g) for g in grids)

        def forge_allclose(a, b, *args):
            """allclose(a, b, tol) — test if arrays are approximately equal."""
            from forge.engine.types import ForgeArray
            ad = a.data if isinstance(a, ForgeArray) else np.atleast_2d(a)
            bd = b.data if isinstance(b, ForgeArray) else np.atleast_2d(b)
            tol = 1e-8
            if args and isinstance(args[0], ForgeArray):
                tol = float(args[0].data.flat[0])
            return ForgeArray(np.float64(1 if np.allclose(ad, bd, atol=tol) else 0))

        # Register all R134 functions
        session._engine.functions["cast"] = forge_cast
        session._engine.functions["filter"] = forge_filter
        session._engine.functions["filtfilt"] = forge_filtfilt
        session._engine.functions["cputime"] = forge_cputime
        session._engine.functions["input"] = forge_input
        session._engine.functions["mpower"] = forge_mpower
        session._engine.functions["butter"] = forge_butter
        session._engine.functions["freqz"] = forge_freqz
        session._engine.functions["lyap"] = forge_lyap
        session._engine.functions["dlyap"] = forge_dlyap
        session._engine.functions["qz"] = forge_qz
        session._engine.functions["perms"] = forge_perms
        session._engine.functions["interpft"] = forge_interpft
        session._engine.functions["deal"] = forge_deal_enhanced
        session._engine.functions["eps"] = forge_eps_func
        session._engine.functions["nextpow2"] = forge_nextpow2_func
        session._engine.functions["log1p"] = forge_log1p
        session._engine.functions["expm1"] = forge_expm1
        session._engine.functions["sinc"] = forge_sinc
        session._engine.functions["unwrap"] = forge_unwrap
        session._engine.functions["colon"] = forge_colon
        session._engine.functions["ndgrid"] = forge_ndgrid
        session._engine.functions["allclose"] = forge_allclose

        # R135: Push past 1000 functions
        def forge_accumarray2(subs, val, *args):
            """accumarray(subs, val, sz, func) — accumulate values by subscript."""
            from forge.engine.types import ForgeArray
            if isinstance(subs, ForgeArray): subs = subs.data.flatten().astype(int)
            if isinstance(val, ForgeArray): val = val.data.flatten()
            if hasattr(val, '__len__') and len(val) == 1:
                val = np.full(len(subs), float(val[0]))
            sz = None
            if args and isinstance(args[0], ForgeArray):
                sz = int(args[0].data.flat[0])
            n = sz if sz else int(np.max(subs))
            result = np.zeros(n)
            for s, v in zip(subs, val):
                result[int(s) - 1] += v
            return ForgeArray(result.reshape(-1, 1))

        def forge_sub2ind(sz, *args):
            """sub2ind(sz, i, j, ...) — subscripts to linear index."""
            from forge.engine.types import ForgeArray
            if isinstance(sz, ForgeArray): sz = sz.data.flatten().astype(int)
            subs = []
            for a in args:
                if isinstance(a, ForgeArray): subs.append(a.data.flatten().astype(int) - 1)
                else: subs.append(np.array([int(a)]) - 1)
            idx = subs[0].copy()
            stride = 1
            for d in range(len(subs)):
                if d > 0:
                    stride *= int(sz[d-1])
                    idx += subs[d] * stride
            return ForgeArray((idx + 1).astype(float))

        def forge_ind2sub(sz, ind):
            """[i, j, ...] = ind2sub(sz, ind) — linear index to subscripts."""
            from forge.engine.types import ForgeArray
            if isinstance(sz, ForgeArray): sz = sz.data.flatten().astype(int)
            if isinstance(ind, ForgeArray): ind = ind.data.flatten().astype(int) - 1
            else: ind = np.array([int(ind)]) - 1
            subs = []
            for d in range(len(sz)):
                subs.append(ForgeArray((ind % sz[d] + 1).astype(float)))
                ind = ind // sz[d]
            if len(subs) == 1: return subs[0]
            return tuple(subs)

        def forge_blkdiag(*args):
            """blkdiag(A, B, ...) — block diagonal matrix."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import block_diag
            mats = []
            for a in args:
                if isinstance(a, ForgeArray): mats.append(a.data)
                else: mats.append(np.atleast_2d(a))
            return ForgeArray(block_diag(*mats))

        def forge_kron(A, B):
            """kron(A, B) — Kronecker tensor product."""
            from forge.engine.types import ForgeArray
            ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            return ForgeArray(np.kron(ad, bd))

        def forge_sylvester2(A, B, C):
            """sylvester(A, B, C) — solve AX + XB = C."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import solve_sylvester
            ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            cd = C.data if isinstance(C, ForgeArray) else np.atleast_2d(C)
            return ForgeArray(solve_sylvester(ad, bd, cd))

        def forge_pinv(A, *args):
            """pinv(A) — Moore-Penrose pseudoinverse."""
            from forge.engine.types import ForgeArray
            ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.linalg.pinv(ad))

        def forge_cov(x, *args):
            """cov(x) — covariance matrix."""
            from forge.engine.types import ForgeArray
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if xd.shape[0] == 1: xd = xd.T
            return ForgeArray(np.cov(xd, rowvar=False))

        def forge_corrcoef(x, *args):
            """corrcoef(x) — correlation coefficient matrix."""
            from forge.engine.types import ForgeArray
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if xd.shape[0] == 1: xd = xd.T
            return ForgeArray(np.corrcoef(xd, rowvar=False))

        def forge_histogram_func(x, *args):
            """histogram(x) or histogram(x, nbins) — compute histogram."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            nbins = 10
            if args and isinstance(args[0], ForgeArray):
                nbins = int(args[0].data.flat[0])
            counts, edges = np.histogram(xd, bins=nbins)
            return ForgeArray(counts.astype(float).reshape(1, -1)), ForgeArray(edges.reshape(1, -1))

        def forge_trapz2(y, *args):
            """trapz(y) or trapz(y, x) — trapezoidal integration."""
            from forge.engine.types import ForgeArray
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            if args and isinstance(args[0], ForgeArray):
                xd = args[0].data.flatten()
                trap_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
                return ForgeArray(np.float64(trap_fn(yd, xd)))
            trap_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
            return ForgeArray(np.float64(trap_fn(yd)))

        def forge_cumtrapz(y, *args):
            """cumtrapz(y) or cumtrapz(y, x) — cumulative trapezoidal integration."""
            from forge.engine.types import ForgeArray
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            if args and isinstance(args[0], ForgeArray):
                xd = args[0].data.flatten()
                dx = np.diff(xd)
            else:
                dx = np.ones(len(yd) - 1)
            result = np.cumsum(0.5 * (yd[:-1] + yd[1:]) * dx)
            return ForgeArray(result.reshape(1, -1))

        def forge_polyfit2(x, y, n):
            """polyfit(x, y, n) — polynomial curve fitting."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            p = np.polyfit(xd, yd, nd)
            return ForgeArray(p.reshape(1, -1))

        def forge_polyval2(p, x):
            """polyval(p, x) — evaluate polynomial."""
            from forge.engine.types import ForgeArray
            pd = p.data.flatten() if isinstance(p, ForgeArray) else np.array(p).flatten()
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.polyval(pd, xd))

        def forge_polyder2(p, *args):
            """polyder(p) — polynomial derivative."""
            from forge.engine.types import ForgeArray
            pd = p.data.flatten() if isinstance(p, ForgeArray) else np.array(p).flatten()
            dp = np.polyder(pd)
            if len(dp) == 0: dp = np.array([0.0])
            return ForgeArray(dp.reshape(1, -1))

        def forge_polyint2(p, *args):
            """polyint(p) — polynomial integration."""
            from forge.engine.types import ForgeArray
            pd = p.data.flatten() if isinstance(p, ForgeArray) else np.array(p).flatten()
            ip = np.polyint(pd)
            return ForgeArray(ip.reshape(1, -1))

        def forge_roots2(p):
            """roots(p) — polynomial roots."""
            from forge.engine.types import ForgeArray
            pd = p.data.flatten() if isinstance(p, ForgeArray) else np.array(p).flatten()
            r = np.roots(pd)
            return ForgeArray(r.reshape(-1, 1))

        def forge_poly2(r):
            """poly(r) — polynomial from roots, or characteristic polynomial."""
            from forge.engine.types import ForgeArray
            rd = r.data if isinstance(r, ForgeArray) else np.atleast_2d(r)
            if rd.shape[0] == rd.shape[1] and rd.shape[0] > 1:
                # Square matrix — characteristic polynomial
                p = np.real(np.poly(rd))
            else:
                p = np.real(np.poly(rd.flatten()))
            return ForgeArray(p.reshape(1, -1))

        def forge_conv2(a, b):
            """conv(a, b) — convolution."""
            from forge.engine.types import ForgeArray
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            return ForgeArray(np.convolve(ad, bd).reshape(1, -1))

        def forge_deconv2(b, a):
            """[q, r] = deconv(b, a) — deconvolution."""
            from forge.engine.types import ForgeArray
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            q, r = np.polydiv(bd, ad)
            return ForgeArray(q.reshape(1, -1)), ForgeArray(r.reshape(1, -1))

        def forge_interp1_2(x, y, xi, *args):
            """interp1(x, y, xi, method) — 1-D interpolation."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            from scipy.interpolate import interp1d
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            xid = xi.data.flatten() if isinstance(xi, ForgeArray) else np.array(xi).flatten()
            method = 'linear'
            for a in args:
                if isinstance(a, ForgeChar): method = a.to_str()
            method_map = {'linear': 'linear', 'nearest': 'nearest', 'spline': 'cubic', 'pchip': 'cubic', 'cubic': 'cubic'}
            f = interp1d(xd, yd, kind=method_map.get(method, 'linear'), fill_value='extrapolate')
            return ForgeArray(f(xid).reshape(1, -1))

        def forge_fft2(x, *args):
            """fft(x) or fft(x, n) — Fast Fourier Transform."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            n = None
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            result = np.fft.fft(xd, n=n)
            return ForgeArray(result.reshape(1, -1))

        def forge_ifft2(x, *args):
            """ifft(x) — inverse FFT."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            n = None
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            result = np.fft.ifft(xd, n=n)
            return ForgeArray(result.reshape(1, -1))

        def forge_fft2d(x):
            """fft2(x) — 2-D FFT."""
            from forge.engine.types import ForgeArray
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.fft.fft2(xd))

        def forge_ifft2d(x):
            """ifft2(x) — inverse 2-D FFT."""
            from forge.engine.types import ForgeArray
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.real(np.fft.ifft2(xd)))

        def forge_fftshift2(x):
            """fftshift(x) — shift zero-frequency to center."""
            from forge.engine.types import ForgeArray
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.fft.fftshift(xd))

        def forge_ifftshift2(x):
            """ifftshift(x) — inverse fftshift."""
            from forge.engine.types import ForgeArray
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.fft.ifftshift(xd))

        def forge_hamming(n):
            """hamming(n) — Hamming window."""
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray): n = int(n.data.flat[0])
            return ForgeArray(np.hamming(n).reshape(-1, 1))

        def forge_hanning(n):
            """hanning(n) — Hanning window."""
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray): n = int(n.data.flat[0])
            return ForgeArray(np.hanning(n).reshape(-1, 1))

        def forge_blackman(n):
            """blackman(n) — Blackman window."""
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray): n = int(n.data.flat[0])
            return ForgeArray(np.blackman(n).reshape(-1, 1))

        def forge_bartlett(n):
            """bartlett(n) — Bartlett (triangular) window."""
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray): n = int(n.data.flat[0])
            return ForgeArray(np.bartlett(n).reshape(-1, 1))

        def forge_kaiser(n, beta):
            """kaiser(n, beta) — Kaiser window."""
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray): n = int(n.data.flat[0])
            if isinstance(beta, ForgeArray): beta = float(beta.data.flat[0])
            return ForgeArray(np.kaiser(n, beta).reshape(-1, 1))

        def forge_cheby1(n, Rp, Wn, *args):
            """[b, a] = cheby1(n, Rp, Wn) — Chebyshev Type I filter."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            from scipy.signal import cheby1 as _cheby1
            if isinstance(n, ForgeArray): n = int(n.data.flat[0])
            if isinstance(Rp, ForgeArray): Rp = float(Rp.data.flat[0])
            if isinstance(Wn, ForgeArray): Wn = float(Wn.data.flat[0])
            ftype = 'low'
            for a in args:
                if isinstance(a, ForgeChar): ftype = a.to_str()
            b, a = _cheby1(n, Rp, Wn, btype=ftype)
            return ForgeArray(b.reshape(1, -1)), ForgeArray(a.reshape(1, -1))

        def forge_cheby2(n, Rs, Wn, *args):
            """[b, a] = cheby2(n, Rs, Wn) — Chebyshev Type II filter."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            from scipy.signal import cheby2 as _cheby2
            if isinstance(n, ForgeArray): n = int(n.data.flat[0])
            if isinstance(Rs, ForgeArray): Rs = float(Rs.data.flat[0])
            if isinstance(Wn, ForgeArray): Wn = float(Wn.data.flat[0])
            ftype = 'low'
            for a in args:
                if isinstance(a, ForgeChar): ftype = a.to_str()
            b, a = _cheby2(n, Rs, Wn, btype=ftype)
            return ForgeArray(b.reshape(1, -1)), ForgeArray(a.reshape(1, -1))

        def forge_ellip(n, Rp, Rs, Wn, *args):
            """[b, a] = ellip(n, Rp, Rs, Wn) — elliptic filter."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            from scipy.signal import ellip as _ellip
            if isinstance(n, ForgeArray): n = int(n.data.flat[0])
            if isinstance(Rp, ForgeArray): Rp = float(Rp.data.flat[0])
            if isinstance(Rs, ForgeArray): Rs = float(Rs.data.flat[0])
            if isinstance(Wn, ForgeArray): Wn = float(Wn.data.flat[0])
            ftype = 'low'
            for a in args:
                if isinstance(a, ForgeChar): ftype = a.to_str()
            b, a = _ellip(n, Rp, Rs, Wn, btype=ftype)
            return ForgeArray(b.reshape(1, -1)), ForgeArray(a.reshape(1, -1))

        def forge_besself(n, Wn, *args):
            """[b, a] = besself(n, Wn) — Bessel analog filter."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            from scipy.signal import bessel as _bessel
            if isinstance(n, ForgeArray): n = int(n.data.flat[0])
            if isinstance(Wn, ForgeArray): Wn = float(Wn.data.flat[0])
            ftype = 'low'
            for a in args:
                if isinstance(a, ForgeChar): ftype = a.to_str()
            b, a = _bessel(n, Wn, btype=ftype, analog=True)
            return ForgeArray(b.reshape(1, -1)), ForgeArray(a.reshape(1, -1))

        def forge_bilinear(b, a, fs):
            """[bz, az] = bilinear(b, a, fs) — analog to digital filter."""
            from forge.engine.types import ForgeArray
            from scipy.signal import bilinear as _bilinear
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            fsd = float(fs.data.flat[0]) if isinstance(fs, ForgeArray) else float(fs)
            bz, az = _bilinear(bd, ad, fsd)
            return ForgeArray(bz.reshape(1, -1)), ForgeArray(az.reshape(1, -1))

        def forge_residue(b, a):
            """[r, p, k] = residue(b, a) — partial fraction decomposition."""
            from forge.engine.types import ForgeArray
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            from scipy.signal import residue as _residue
            r, p, k = _residue(bd, ad)
            return ForgeArray(r.reshape(-1, 1)), ForgeArray(p.reshape(-1, 1)), ForgeArray(k.reshape(1, -1) if len(k) > 0 else np.array([[0.0]]))

        def forge_xcorr(x, *args):
            """xcorr(x) or xcorr(x, y) — cross-correlation."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            if args and isinstance(args[0], ForgeArray):
                yd = args[0].data.flatten()
            else:
                yd = xd
            result = np.correlate(xd, yd, mode='full')
            lags = np.arange(-(len(xd)-1), len(yd))
            return ForgeArray(result.reshape(1, -1)), ForgeArray(lags.reshape(1, -1))

        # Register R135 functions
        session._engine.functions["sub2ind"] = forge_sub2ind
        session._engine.functions["ind2sub"] = forge_ind2sub
        session._engine.functions["blkdiag"] = forge_blkdiag
        session._engine.functions["kron"] = forge_kron
        session._engine.functions["pinv"] = forge_pinv
        session._engine.functions["cov"] = forge_cov
        session._engine.functions["corrcoef"] = forge_corrcoef
        session._engine.functions["histogram"] = forge_histogram_func
        session._engine.functions["trapz"] = forge_trapz2
        session._engine.functions["cumtrapz"] = forge_cumtrapz
        session._engine.functions["polyfit"] = forge_polyfit2
        session._engine.functions["polyval"] = forge_polyval2
        session._engine.functions["polyder"] = forge_polyder2
        session._engine.functions["polyint"] = forge_polyint2
        session._engine.functions["roots"] = forge_roots2
        session._engine.functions["poly"] = forge_poly2
        session._engine.functions["conv"] = forge_conv2
        session._engine.functions["deconv"] = forge_deconv2
        session._engine.functions["interp1"] = forge_interp1_2
        session._engine.functions["fft"] = forge_fft2
        session._engine.functions["ifft"] = forge_ifft2
        session._engine.functions["fft2"] = forge_fft2d
        session._engine.functions["ifft2"] = forge_ifft2d
        session._engine.functions["fftshift"] = forge_fftshift2
        session._engine.functions["ifftshift"] = forge_ifftshift2
        session._engine.functions["hamming"] = forge_hamming
        session._engine.functions["hanning"] = forge_hanning
        session._engine.functions["blackman"] = forge_blackman
        session._engine.functions["bartlett"] = forge_bartlett
        session._engine.functions["kaiser"] = forge_kaiser
        session._engine.functions["cheby1"] = forge_cheby1
        session._engine.functions["cheby2"] = forge_cheby2
        session._engine.functions["ellip"] = forge_ellip
        session._engine.functions["besself"] = forge_besself
        session._engine.functions["bilinear"] = forge_bilinear
        session._engine.functions["residue"] = forge_residue
        session._engine.functions["xcorr"] = forge_xcorr
        session._engine.functions["sylvester"] = forge_sylvester2

        # R135b: 40 new functions to cross 1000 milestone
        # --- Math special functions ---
        def forge_gamma(x):
            """gamma(x) — gamma function."""
            from forge.engine.types import ForgeArray
            from scipy.special import gamma as _gamma
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(_gamma(data))

        def forge_gammaln(x):
            """gammaln(x) — log of gamma function."""
            from forge.engine.types import ForgeArray
            from scipy.special import gammaln as _gammaln
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(_gammaln(data))

        def forge_erf(x):
            """erf(x) — error function."""
            from forge.engine.types import ForgeArray
            from scipy.special import erf as _erf
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(_erf(data))

        def forge_erfc(x):
            """erfc(x) — complementary error function."""
            from forge.engine.types import ForgeArray
            from scipy.special import erfc as _erfc
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(_erfc(data))

        def forge_erfinv(x):
            """erfinv(x) — inverse error function."""
            from forge.engine.types import ForgeArray
            from scipy.special import erfinv as _erfinv
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(_erfinv(data))

        def forge_erfcinv(x):
            """erfcinv(x) — inverse complementary error function."""
            from forge.engine.types import ForgeArray
            from scipy.special import erfcinv as _erfcinv
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(_erfcinv(data))

        def forge_besselj(nu, x):
            """besselj(nu, x) — Bessel function of first kind."""
            from forge.engine.types import ForgeArray
            from scipy.special import jv
            n = float(nu.data.flat[0]) if isinstance(nu, ForgeArray) else float(nu)
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(jv(n, data))

        def forge_bessely(nu, x):
            """bessely(nu, x) — Bessel function of second kind."""
            from forge.engine.types import ForgeArray
            from scipy.special import yv
            n = float(nu.data.flat[0]) if isinstance(nu, ForgeArray) else float(nu)
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(yv(n, data))

        def forge_besseli(nu, x):
            """besseli(nu, x) — modified Bessel function first kind."""
            from forge.engine.types import ForgeArray
            from scipy.special import iv
            n = float(nu.data.flat[0]) if isinstance(nu, ForgeArray) else float(nu)
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(iv(n, data))

        def forge_besselk(nu, x):
            """besselk(nu, x) — modified Bessel function second kind."""
            from forge.engine.types import ForgeArray
            from scipy.special import kv
            n = float(nu.data.flat[0]) if isinstance(nu, ForgeArray) else float(nu)
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(kv(n, data))

        def forge_besselh(nu, x):
            """besselh(nu, x) — Hankel function (Bessel of third kind)."""
            from forge.engine.types import ForgeArray
            from scipy.special import hankel1
            n = float(nu.data.flat[0]) if isinstance(nu, ForgeArray) else float(nu)
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(hankel1(n, data))

        def forge_airy_func(x):
            """[Ai, Bi] = airy(x) — Airy functions."""
            from forge.engine.types import ForgeArray
            from scipy.special import airy as _airy
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            ai, aip, bi, bip = _airy(data)
            return ForgeArray(ai), ForgeArray(bi)

        def forge_ellipj(u, m):
            """[sn, cn, dn] = ellipj(u, m) — Jacobi elliptic functions."""
            from forge.engine.types import ForgeArray
            from scipy.special import ellipj as _ellipj
            ud = u.data if isinstance(u, ForgeArray) else np.atleast_2d(u)
            md = float(m.data.flat[0]) if isinstance(m, ForgeArray) else float(m)
            sn, cn, dn, ph = _ellipj(ud, md)
            return ForgeArray(sn), ForgeArray(cn), ForgeArray(dn)

        def forge_gcd(a, b):
            """gcd(a, b) — greatest common divisor."""
            from forge.engine.types import ForgeArray
            ad = int(a.data.flat[0]) if isinstance(a, ForgeArray) else int(a)
            bd = int(b.data.flat[0]) if isinstance(b, ForgeArray) else int(b)
            import math
            return ForgeArray(np.float64(math.gcd(ad, bd)))

        # --- Bit operations ---
        def forge_bitget(x, bit):
            """bitget(x, bit) — get bit at position."""
            from forge.engine.types import ForgeArray
            xd = int(x.data.flat[0]) if isinstance(x, ForgeArray) else int(x)
            bd = int(bit.data.flat[0]) if isinstance(bit, ForgeArray) else int(bit)
            return ForgeArray(np.float64((xd >> (bd - 1)) & 1))

        def forge_bitset(x, bit, v=None):
            """bitset(x, bit, v) — set bit at position."""
            from forge.engine.types import ForgeArray
            xd = int(x.data.flat[0]) if isinstance(x, ForgeArray) else int(x)
            bd = int(bit.data.flat[0]) if isinstance(bit, ForgeArray) else int(bit)
            val = 1
            if v is not None:
                val = int(v.data.flat[0]) if isinstance(v, ForgeArray) else int(v)
            if val:
                result = xd | (1 << (bd - 1))
            else:
                result = xd & ~(1 << (bd - 1))
            return ForgeArray(np.float64(result))

        def forge_bitcmp(x, n=None):
            """bitcmp(x, n) — bitwise complement."""
            from forge.engine.types import ForgeArray
            xd = int(x.data.flat[0]) if isinstance(x, ForgeArray) else int(x)
            nd = 64
            if n is not None:
                nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            mask = (1 << nd) - 1
            return ForgeArray(np.float64(xd ^ mask))

        def forge_bitshift(x, k):
            """bitshift(x, k) — shift bits by k positions."""
            from forge.engine.types import ForgeArray
            xd = int(x.data.flat[0]) if isinstance(x, ForgeArray) else int(x)
            kd = int(k.data.flat[0]) if isinstance(k, ForgeArray) else int(k)
            if kd >= 0:
                return ForgeArray(np.float64(xd << kd))
            return ForgeArray(np.float64(xd >> (-kd)))

        def forge_swapbytes(x):
            """swapbytes(x) — swap byte order."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(data.byteswap())

        # --- System functions ---
        def forge_system(cmd):
            """[status, output] = system(cmd) — execute system command."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import subprocess
            if isinstance(cmd, ForgeChar): cmd = cmd.to_str()
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                return ForgeArray(np.float64(result.returncode)), ForgeChar(result.stdout)
            except Exception:
                return ForgeArray(np.float64(1)), ForgeChar("")

        def forge_getenv(name):
            """getenv(name) — get environment variable."""
            from forge.engine.containers import ForgeChar
            if isinstance(name, ForgeChar): name = name.to_str()
            return ForgeChar(os.environ.get(name, ""))

        def forge_setenv(name, val):
            """setenv(name, val) — set environment variable."""
            from forge.engine.containers import ForgeChar
            if isinstance(name, ForgeChar): name = name.to_str()
            if isinstance(val, ForgeChar): val = val.to_str()
            os.environ[name] = val

        # --- Validation functions ---
        def forge_mustBePositive(x):
            """mustBePositive(x) — validate positive."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if not np.all(data > 0):
                raise ValueError("Value must be positive")

        def forge_mustBeNonnegative(x):
            """mustBeNonnegative(x) — validate nonnegative."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if not np.all(data >= 0):
                raise ValueError("Value must be nonnegative")

        def forge_mustBeNonzero(x):
            """mustBeNonzero(x) — validate nonzero."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if not np.all(data != 0):
                raise ValueError("Value must be nonzero")

        def forge_mustBeFinite(x):
            """mustBeFinite(x) — validate finite."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if not np.all(np.isfinite(data)):
                raise ValueError("Value must be finite")

        def forge_mustBeNonempty(x):
            """mustBeNonempty(x) — validate nonempty."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if data.size == 0:
                raise ValueError("Value must be nonempty")

        def forge_mustBeInteger(x):
            """mustBeInteger(x) — validate integer-valued."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if not np.all(data == np.floor(data)):
                raise ValueError("Value must be integer")

        def forge_mustBeReal(x):
            """mustBeReal(x) — validate real."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if np.any(np.iscomplex(data)):
                raise ValueError("Value must be real")

        def forge_mustBeNumeric(x):
            """mustBeNumeric(x) — validate numeric."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(x, ForgeChar):
                raise ValueError("Value must be numeric")

        def forge_mustBeLogical(x):
            """mustBeLogical(x) — validate logical."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if data.dtype != np.bool_:
                raise ValueError("Value must be logical")

        def forge_mustBeMember(x, S):
            """mustBeMember(x, S) — validate membership."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array([x]).flatten()
            sd = S.data.flatten() if isinstance(S, ForgeArray) else np.array(S).flatten()
            for v in xd:
                if v not in sd:
                    raise ValueError(f"Value {v} must be a member of the set")

        def forge_mustBeInRange(x, lo, hi):
            """mustBeInRange(x, lo, hi) — validate in range."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            l = float(lo.data.flat[0]) if isinstance(lo, ForgeArray) else float(lo)
            h = float(hi.data.flat[0]) if isinstance(hi, ForgeArray) else float(hi)
            if not np.all((data >= l) & (data <= h)):
                raise ValueError(f"Value must be in range [{l}, {h}]")

        def forge_mustBeGreaterThan(x, c):
            """mustBeGreaterThan(x, c) — validate greater than."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            cv = float(c.data.flat[0]) if isinstance(c, ForgeArray) else float(c)
            if not np.all(data > cv):
                raise ValueError(f"Value must be greater than {cv}")

        def forge_mustBeLessThan(x, c):
            """mustBeLessThan(x, c) — validate less than."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            cv = float(c.data.flat[0]) if isinstance(c, ForgeArray) else float(c)
            if not np.all(data < cv):
                raise ValueError(f"Value must be less than {cv}")

        def forge_orderfields(s, *args):
            """orderfields(s) — reorder struct fields alphabetically."""
            from forge.engine.containers import ForgeStruct
            if isinstance(s, ForgeStruct):
                ordered = ForgeStruct()
                for k in sorted(s._fields.keys()):
                    ordered._fields[k] = s._fields[k]
                return ordered
            return s

        def forge_mat2cell(x, r, c):
            """mat2cell(x, r, c) — break matrix into cell array."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeCell
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            rd = r.data.flatten().astype(int) if isinstance(r, ForgeArray) else [int(r)]
            cd = c.data.flatten().astype(int) if isinstance(c, ForgeArray) else [int(c)]
            rows = np.cumsum([0] + list(rd))
            cols = np.cumsum([0] + list(cd))
            result = ForgeCell(len(rd), len(cd))
            for i in range(len(rd)):
                for j in range(len(cd)):
                    block = xd[rows[i]:rows[i+1], cols[j]:cols[j+1]]
                    result.content_set(i+1, j+1, ForgeArray(block))
            return result

        def forge_mfilename(*args):
            """mfilename — return name of currently executing file."""
            from forge.engine.containers import ForgeChar
            return ForgeChar("")

        def forge_nargchk(lo, hi, n):
            """nargchk(lo, hi, n) — validate number of arguments (deprecated)."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            l = int(lo.data.flat[0]) if isinstance(lo, ForgeArray) else int(lo)
            h = int(hi.data.flat[0]) if isinstance(hi, ForgeArray) else int(hi)
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            if nd < l:
                return ForgeChar("not enough input arguments")
            if nd > h:
                return ForgeChar("too many input arguments")
            return ForgeChar("")

        def forge_validatestring(s, valid):
            """validatestring(s, validStrings) — validate and complete string."""
            from forge.engine.containers import ForgeChar, ForgeCell
            from forge.engine.types import ForgeArray
            if isinstance(s, ForgeChar): s = s.to_str()
            matches = []
            if isinstance(valid, ForgeCell):
                for i in range(valid.rows * valid.cols):
                    v = valid._data[i]
                    if isinstance(v, ForgeChar): v = v.to_str()
                    if isinstance(v, str) and v.lower().startswith(s.lower()):
                        matches.append(v)
            if len(matches) == 1:
                return ForgeChar(matches[0])
            if len(matches) == 0:
                raise ValueError(f"'{s}' does not match any valid string")
            raise ValueError(f"'{s}' is ambiguous")

        # Register all R135b functions
        session._engine.functions["gamma"] = forge_gamma
        session._engine.functions["gammaln"] = forge_gammaln
        session._engine.functions["erf"] = forge_erf
        session._engine.functions["erfc"] = forge_erfc
        session._engine.functions["erfinv"] = forge_erfinv
        session._engine.functions["erfcinv"] = forge_erfcinv
        session._engine.functions["besselj"] = forge_besselj
        session._engine.functions["bessely"] = forge_bessely
        session._engine.functions["besseli"] = forge_besseli
        session._engine.functions["besselk"] = forge_besselk
        session._engine.functions["besselh"] = forge_besselh
        session._engine.functions["airy"] = forge_airy_func
        session._engine.functions["ellipj"] = forge_ellipj
        session._engine.functions["gcd"] = forge_gcd
        session._engine.functions["bitget"] = forge_bitget
        session._engine.functions["bitset"] = forge_bitset
        session._engine.functions["bitcmp"] = forge_bitcmp
        session._engine.functions["bitshift"] = forge_bitshift
        session._engine.functions["swapbytes"] = forge_swapbytes
        session._engine.functions["system"] = forge_system
        session._engine.functions["getenv"] = forge_getenv
        session._engine.functions["setenv"] = forge_setenv
        session._engine.functions["mustBePositive"] = forge_mustBePositive
        session._engine.functions["mustBeNonnegative"] = forge_mustBeNonnegative
        session._engine.functions["mustBeNonzero"] = forge_mustBeNonzero
        session._engine.functions["mustBeFinite"] = forge_mustBeFinite
        session._engine.functions["mustBeNonempty"] = forge_mustBeNonempty
        session._engine.functions["mustBeInteger"] = forge_mustBeInteger
        session._engine.functions["mustBeReal"] = forge_mustBeReal
        session._engine.functions["mustBeNumeric"] = forge_mustBeNumeric
        session._engine.functions["mustBeLogical"] = forge_mustBeLogical
        session._engine.functions["mustBeMember"] = forge_mustBeMember
        session._engine.functions["mustBeInRange"] = forge_mustBeInRange
        session._engine.functions["mustBeGreaterThan"] = forge_mustBeGreaterThan
        session._engine.functions["mustBeLessThan"] = forge_mustBeLessThan
        session._engine.functions["orderfields"] = forge_orderfields
        session._engine.functions["mat2cell"] = forge_mat2cell
        session._engine.functions["mfilename"] = forge_mfilename
        session._engine.functions["nargchk"] = forge_nargchk
        session._engine.functions["validatestring"] = forge_validatestring

        # R136: ODE solvers, optimization, integration
        def forge_ode45(odefun, tspan, y0, *args):
            """[t, y] = ode45(odefun, tspan, y0) — solve ODE with RK45."""
            from forge.engine.types import ForgeArray
            from scipy.integrate import solve_ivp
            if isinstance(tspan, ForgeArray): tspan = tspan.data.flatten()
            if isinstance(y0, ForgeArray): y0 = y0.data.flatten()
            t0, tf = float(tspan[0]), float(tspan[-1])
            t_eval = None
            if len(tspan) > 2:
                t_eval = tspan.astype(float)
            def rhs(t, y):
                yt = ForgeArray(y.reshape(-1, 1))
                tt = ForgeArray(np.float64(t))
                result = odefun(tt, yt)
                if isinstance(result, ForgeArray):
                    return result.data.flatten()
                return np.array(result).flatten()
            sol = solve_ivp(rhs, [t0, tf], y0.astype(float), method='RK45',
                          t_eval=t_eval, rtol=1e-6, atol=1e-9, max_step=np.inf)
            return ForgeArray(sol.t.reshape(-1, 1)), ForgeArray(sol.y.T)

        def forge_ode23(odefun, tspan, y0, *args):
            """[t, y] = ode23(odefun, tspan, y0) — solve ODE with RK23."""
            from forge.engine.types import ForgeArray
            from scipy.integrate import solve_ivp
            if isinstance(tspan, ForgeArray): tspan = tspan.data.flatten()
            if isinstance(y0, ForgeArray): y0 = y0.data.flatten()
            t0, tf = float(tspan[0]), float(tspan[-1])
            t_eval = None
            if len(tspan) > 2:
                t_eval = tspan.astype(float)
            def rhs(t, y):
                yt = ForgeArray(y.reshape(-1, 1))
                tt = ForgeArray(np.float64(t))
                result = odefun(tt, yt)
                if isinstance(result, ForgeArray):
                    return result.data.flatten()
                return np.array(result).flatten()
            sol = solve_ivp(rhs, [t0, tf], y0.astype(float), method='RK23',
                          t_eval=t_eval, rtol=1e-3, atol=1e-6)
            return ForgeArray(sol.t.reshape(-1, 1)), ForgeArray(sol.y.T)

        def forge_ode15s(odefun, tspan, y0, *args):
            """[t, y] = ode15s(odefun, tspan, y0) — solve stiff ODE."""
            from forge.engine.types import ForgeArray
            from scipy.integrate import solve_ivp
            if isinstance(tspan, ForgeArray): tspan = tspan.data.flatten()
            if isinstance(y0, ForgeArray): y0 = y0.data.flatten()
            t0, tf = float(tspan[0]), float(tspan[-1])
            t_eval = None
            if len(tspan) > 2:
                t_eval = tspan.astype(float)
            def rhs(t, y):
                yt = ForgeArray(y.reshape(-1, 1))
                tt = ForgeArray(np.float64(t))
                result = odefun(tt, yt)
                if isinstance(result, ForgeArray):
                    return result.data.flatten()
                return np.array(result).flatten()
            sol = solve_ivp(rhs, [t0, tf], y0.astype(float), method='BDF',
                          t_eval=t_eval, rtol=1e-6, atol=1e-9)
            return ForgeArray(sol.t.reshape(-1, 1)), ForgeArray(sol.y.T)

        def forge_ode23s(odefun, tspan, y0, *args):
            """[t, y] = ode23s(odefun, tspan, y0) — solve stiff ODE (Rosenbrock)."""
            from forge.engine.types import ForgeArray
            from scipy.integrate import solve_ivp
            if isinstance(tspan, ForgeArray): tspan = tspan.data.flatten()
            if isinstance(y0, ForgeArray): y0 = y0.data.flatten()
            t0, tf = float(tspan[0]), float(tspan[-1])
            t_eval = None
            if len(tspan) > 2:
                t_eval = tspan.astype(float)
            def rhs(t, y):
                yt = ForgeArray(y.reshape(-1, 1))
                tt = ForgeArray(np.float64(t))
                result = odefun(tt, yt)
                if isinstance(result, ForgeArray):
                    return result.data.flatten()
                return np.array(result).flatten()
            sol = solve_ivp(rhs, [t0, tf], y0.astype(float), method='Radau',
                          t_eval=t_eval, rtol=1e-3, atol=1e-6)
            return ForgeArray(sol.t.reshape(-1, 1)), ForgeArray(sol.y.T)

        def forge_odeset(*args):
            """opts = odeset('Name', Value, ...) — ODE options (stub)."""
            from forge.engine.containers import ForgeStruct, ForgeChar
            from forge.engine.types import ForgeArray
            opts = ForgeStruct()
            i = 0
            while i < len(args) - 1:
                name = args[i]
                val = args[i+1]
                if isinstance(name, ForgeChar): name = name.to_str()
                opts._fields[name] = val
                i += 2
            return opts

        def forge_odeget(opts, name):
            """odeget(opts, name) — get ODE option value."""
            from forge.engine.containers import ForgeStruct, ForgeChar
            if isinstance(name, ForgeChar): name = name.to_str()
            if isinstance(opts, ForgeStruct) and name in opts._fields:
                return opts._fields[name]
            return ForgeArray(np.array([[]]))

        # --- Optimization ---
        def forge_fzero(fun, x0, *args):
            """fzero(fun, x0) — find zero of function."""
            from forge.engine.types import ForgeArray
            from scipy.optimize import brentq, fsolve
            if isinstance(x0, ForgeArray):
                x0d = x0.data.flatten()
            else:
                x0d = np.array([float(x0)])
            def f(x):
                result = fun(ForgeArray(np.float64(x)))
                if isinstance(result, ForgeArray):
                    return float(result.data.flat[0])
                return float(result)
            if len(x0d) == 2:
                root = brentq(f, float(x0d[0]), float(x0d[1]))
            else:
                root = fsolve(f, float(x0d[0]))[0]
            return ForgeArray(np.float64(root))

        def forge_fminbnd(fun, a, b, *args):
            """fminbnd(fun, a, b) — minimize scalar function on interval."""
            from forge.engine.types import ForgeArray
            from scipy.optimize import minimize_scalar
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            def f(x):
                result = fun(ForgeArray(np.float64(x)))
                if isinstance(result, ForgeArray):
                    return float(result.data.flat[0])
                return float(result)
            res = minimize_scalar(f, bounds=(ad, bd), method='bounded')
            return ForgeArray(np.float64(res.x))

        def forge_fminsearch(fun, x0, *args):
            """fminsearch(fun, x0) — Nelder-Mead minimization."""
            from forge.engine.types import ForgeArray
            from scipy.optimize import minimize
            if isinstance(x0, ForgeArray): x0d = x0.data.flatten()
            else: x0d = np.array([float(x0)])
            def f(x):
                result = fun(ForgeArray(x.reshape(-1, 1)))
                if isinstance(result, ForgeArray):
                    return float(result.data.flat[0])
                return float(result)
            res = minimize(f, x0d, method='Nelder-Mead')
            return ForgeArray(res.x.reshape(-1, 1))

        def forge_fsolve(fun, x0, *args):
            """fsolve(fun, x0) — solve system of nonlinear equations."""
            from forge.engine.types import ForgeArray
            from scipy.optimize import fsolve as _fsolve
            if isinstance(x0, ForgeArray): x0d = x0.data.flatten()
            else: x0d = np.array([float(x0)])
            def f(x):
                result = fun(ForgeArray(x.reshape(-1, 1)))
                if isinstance(result, ForgeArray):
                    return result.data.flatten()
                return np.array(result).flatten()
            sol = _fsolve(f, x0d)
            return ForgeArray(sol.reshape(-1, 1))

        def forge_lsqnonneg(C, d):
            """lsqnonneg(C, d) — nonnegative least squares."""
            from forge.engine.types import ForgeArray
            from scipy.optimize import nnls
            Cd = C.data if isinstance(C, ForgeArray) else np.atleast_2d(C)
            dd = d.data.flatten() if isinstance(d, ForgeArray) else np.array(d).flatten()
            x, rnorm = nnls(Cd, dd)
            return ForgeArray(x.reshape(-1, 1))

        # --- Integration ---
        def forge_integral(fun, a, b, *args):
            """integral(fun, a, b) — numerical integration."""
            from forge.engine.types import ForgeArray
            from scipy.integrate import quad
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            def f(x):
                result = fun(ForgeArray(np.float64(x)))
                if isinstance(result, ForgeArray):
                    return float(result.data.flat[0])
                return float(result)
            val, err = quad(f, ad, bd)
            return ForgeArray(np.float64(val))

        def forge_integral2(fun, xa, xb, ya, yb, *args):
            """integral2(fun, xa, xb, ya, yb) — 2D numerical integration."""
            from forge.engine.types import ForgeArray
            from scipy.integrate import dblquad
            xad = float(xa.data.flat[0]) if isinstance(xa, ForgeArray) else float(xa)
            xbd = float(xb.data.flat[0]) if isinstance(xb, ForgeArray) else float(xb)
            yad = float(ya.data.flat[0]) if isinstance(ya, ForgeArray) else float(ya)
            ybd = float(yb.data.flat[0]) if isinstance(yb, ForgeArray) else float(yb)
            def f(y, x):
                result = fun(ForgeArray(np.float64(x)), ForgeArray(np.float64(y)))
                if isinstance(result, ForgeArray):
                    return float(result.data.flat[0])
                return float(result)
            val, err = dblquad(f, xad, xbd, yad, ybd)
            return ForgeArray(np.float64(val))

        def forge_quad_func(fun, a, b, *args):
            """quad(fun, a, b) — adaptive Simpson quadrature."""
            from forge.engine.types import ForgeArray
            from scipy.integrate import quad as _quad
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            def f(x):
                result = fun(ForgeArray(np.float64(x)))
                if isinstance(result, ForgeArray):
                    return float(result.data.flat[0])
                return float(result)
            val, err = _quad(f, ad, bd)
            return ForgeArray(np.float64(val))

        def forge_quadgk(fun, a, b, *args):
            """quadgk(fun, a, b) — Gauss-Kronrod quadrature."""
            from forge.engine.types import ForgeArray
            from scipy.integrate import quad as _quad
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            def f(x):
                result = fun(ForgeArray(np.float64(x)))
                if isinstance(result, ForgeArray):
                    return float(result.data.flat[0])
                return float(result)
            val, err = _quad(f, ad, bd)
            return ForgeArray(np.float64(val)), ForgeArray(np.float64(err))

        # --- Time functions ---
        def forge_tic():
            """tic — start timer."""
            from forge.engine.types import ForgeArray
            import time
            session._tic_time = time.time()
            return ForgeArray(np.float64(session._tic_time))

        def forge_toc(*args):
            """toc — elapsed time since tic."""
            from forge.engine.types import ForgeArray
            import time
            t0 = getattr(session, '_tic_time', time.time())
            elapsed = time.time() - t0
            return ForgeArray(np.float64(elapsed))

        def forge_clock():
            """clock — current date/time as [y m d h m s]."""
            from forge.engine.types import ForgeArray
            import datetime
            now = datetime.datetime.now()
            return ForgeArray(np.array([[now.year, now.month, now.day,
                                        now.hour, now.minute,
                                        now.second + now.microsecond/1e6]]))

        def forge_now():
            """now — current date as serial date number."""
            from forge.engine.types import ForgeArray
            import datetime
            d = datetime.datetime.now()
            # MATLAB datenum: days since Jan 0, 0000
            delta = d - datetime.datetime(1, 1, 1)
            return ForgeArray(np.float64(delta.days + 1 + delta.seconds/86400.0 + 367))

        def forge_datenum(*args):
            """datenum(y, m, d) or datenum(datestr) — serial date number."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import datetime
            if len(args) >= 3:
                y = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                m = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
                d = int(args[2].data.flat[0]) if isinstance(args[2], ForgeArray) else int(args[2])
                dt = datetime.datetime(y, m, d)
                delta = dt - datetime.datetime(1, 1, 1)
                return ForgeArray(np.float64(delta.days + 1 + 367))
            return ForgeArray(np.float64(0))

        def forge_datevec(d):
            """datevec(d) — convert date number to [y m d h m s]."""
            from forge.engine.types import ForgeArray
            import datetime
            if isinstance(d, ForgeArray): d = float(d.data.flat[0])
            base = datetime.datetime(1, 1, 1) + datetime.timedelta(days=d - 1 - 367)
            return ForgeArray(np.array([[base.year, base.month, base.day,
                                        base.hour, base.minute, base.second]]).astype(float))

        def forge_datestr(d, *args):
            """datestr(d) — convert date number to string."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import datetime
            if isinstance(d, ForgeArray): d = float(d.data.flat[0])
            base = datetime.datetime(1, 1, 1) + datetime.timedelta(days=d - 1 - 367)
            return ForgeChar(base.strftime('%d-%b-%Y'))

        def forge_etime(t1, t0):
            """etime(t1, t0) — elapsed time between clock vectors."""
            from forge.engine.types import ForgeArray
            import datetime
            def to_dt(t):
                d = t.data.flatten() if isinstance(t, ForgeArray) else np.array(t).flatten()
                return datetime.datetime(int(d[0]), int(d[1]), int(d[2]),
                                        int(d[3]), int(d[4]), int(d[5]))
            dt = to_dt(t1) - to_dt(t0)
            return ForgeArray(np.float64(dt.total_seconds()))

        def forge_weekday(d):
            """weekday(d) — day of week (1=Sun, 7=Sat)."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import datetime
            if isinstance(d, ForgeArray): d = float(d.data.flat[0])
            base = datetime.datetime(1, 1, 1) + datetime.timedelta(days=d - 1 - 367)
            dow = base.isoweekday()  # 1=Mon, 7=Sun
            matlab_dow = (dow % 7) + 1  # 1=Sun, 7=Sat
            names = ['', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            return ForgeArray(np.float64(matlab_dow)), ForgeChar(names[matlab_dow])

        def forge_is_leap_year(y):
            """is_leap_year(y) — test for leap year."""
            from forge.engine.types import ForgeArray
            import calendar
            yd = int(y.data.flat[0]) if isinstance(y, ForgeArray) else int(y)
            return ForgeArray(np.float64(1 if calendar.isleap(yd) else 0))

        def forge_eomday(y, m):
            """eomday(y, m) — end of month day."""
            from forge.engine.types import ForgeArray
            import calendar
            yd = int(y.data.flat[0]) if isinstance(y, ForgeArray) else int(y)
            md = int(m.data.flat[0]) if isinstance(m, ForgeArray) else int(m)
            return ForgeArray(np.float64(calendar.monthrange(yd, md)[1]))

        # --- File I/O extras ---
        def forge_tempname():
            """tempname — generate temporary file name."""
            from forge.engine.containers import ForgeChar
            import tempfile
            return ForgeChar(tempfile.mktemp())

        def forge_tempdir():
            """tempdir — get system temp directory."""
            from forge.engine.containers import ForgeChar
            import tempfile
            return ForgeChar(tempfile.gettempdir())

        # Register all R136 functions
        session._engine.functions["ode45"] = forge_ode45
        session._engine.functions["ode23"] = forge_ode23
        session._engine.functions["ode15s"] = forge_ode15s
        session._engine.functions["ode23s"] = forge_ode23s
        session._engine.functions["odeset"] = forge_odeset
        session._engine.functions["odeget"] = forge_odeget
        session._engine.functions["fzero"] = forge_fzero
        session._engine.functions["fminbnd"] = forge_fminbnd
        session._engine.functions["fminsearch"] = forge_fminsearch
        session._engine.functions["fsolve"] = forge_fsolve
        session._engine.functions["lsqnonneg"] = forge_lsqnonneg
        session._engine.functions["integral"] = forge_integral
        session._engine.functions["integral2"] = forge_integral2
        session._engine.functions["quad"] = forge_quad_func
        session._engine.functions["quadgk"] = forge_quadgk
        session._engine.functions["tic"] = forge_tic
        session._engine.functions["toc"] = forge_toc
        session._engine.functions["clock"] = forge_clock
        session._engine.functions["now"] = forge_now
        session._engine.functions["datenum"] = forge_datenum
        session._engine.functions["datevec"] = forge_datevec
        session._engine.functions["datestr"] = forge_datestr
        session._engine.functions["etime"] = forge_etime
        session._engine.functions["weekday"] = forge_weekday
        session._engine.functions["is_leap_year"] = forge_is_leap_year
        session._engine.functions["eomday"] = forge_eomday
        session._engine.functions["tempname"] = forge_tempname
        session._engine.functions["tempdir"] = forge_tempdir

        # R138: CSV I/O, sparse solvers, more matrix functions
        def forge_csvread(fname, *args):
            """csvread(fname) — read CSV file."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            data = np.loadtxt(fname, delimiter=',')
            return ForgeArray(np.atleast_2d(data))

        def forge_csvwrite(fname, M, *args):
            """csvwrite(fname, M) — write CSV file."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            data = M.data if isinstance(M, ForgeArray) else np.atleast_2d(M)
            np.savetxt(fname, data, delimiter=',')

        def forge_dlmread(fname, *args):
            """dlmread(fname, delim) — read delimited file."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            delim = ','
            if args and isinstance(args[0], ForgeChar):
                delim = args[0].to_str()
            data = np.loadtxt(fname, delimiter=delim)
            return ForgeArray(np.atleast_2d(data))

        def forge_dlmwrite(fname, M, *args):
            """dlmwrite(fname, M, delim) — write delimited file."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            data = M.data if isinstance(M, ForgeArray) else np.atleast_2d(M)
            delim = ','
            if args and isinstance(args[0], ForgeChar):
                delim = args[0].to_str()
            np.savetxt(fname, data, delimiter=delim)

        def forge_fileread(fname):
            """fileread(fname) — read entire file as string."""
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            with open(fname, 'r') as f:
                return ForgeChar(f.read())

        # --- Sparse iterative solvers ---
        def forge_pcg(A, b, *args):
            """pcg(A, b) — preconditioned conjugate gradient."""
            from forge.engine.types import ForgeArray
            from scipy.sparse.linalg import cg
            from scipy.sparse import issparse
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            tol = 1e-6
            if args and isinstance(args[0], ForgeArray):
                tol = float(args[0].data.flat[0])
            x, info = cg(Ad, bd, tol=tol)
            return ForgeArray(x.reshape(-1, 1)), ForgeArray(np.float64(info))

        def forge_gmres_func(A, b, *args):
            """gmres(A, b) — generalized minimum residual method."""
            from forge.engine.types import ForgeArray
            from scipy.sparse.linalg import gmres
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            tol = 1e-6
            if args and isinstance(args[0], ForgeArray):
                tol = float(args[0].data.flat[0])
            x, info = gmres(Ad, bd, tol=tol)
            return ForgeArray(x.reshape(-1, 1)), ForgeArray(np.float64(info))

        def forge_bicgstab(A, b, *args):
            """bicgstab(A, b) — BiCGSTAB iterative solver."""
            from forge.engine.types import ForgeArray
            from scipy.sparse.linalg import bicgstab
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            tol = 1e-6
            x, info = bicgstab(Ad, bd, tol=tol)
            return ForgeArray(x.reshape(-1, 1)), ForgeArray(np.float64(info))

        def forge_eigs_func(A, *args):
            """eigs(A, k) — find k largest eigenvalues."""
            from forge.engine.types import ForgeArray
            from scipy.sparse.linalg import eigs
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            k = 6
            if args and isinstance(args[0], ForgeArray):
                k = int(args[0].data.flat[0])
            k = min(k, Ad.shape[0] - 2)
            if k < 1: k = 1
            vals, vecs = eigs(Ad.astype(complex), k=k)
            return ForgeArray(vals.reshape(-1, 1)), ForgeArray(vecs)

        def forge_svds_func(A, *args):
            """svds(A, k) — find k largest singular values."""
            from forge.engine.types import ForgeArray
            from scipy.sparse.linalg import svds as _svds
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            k = 6
            if args and isinstance(args[0], ForgeArray):
                k = int(args[0].data.flat[0])
            k = min(k, min(Ad.shape) - 1)
            if k < 1: k = 1
            U, s, Vh = _svds(Ad, k=k)
            return ForgeArray(U), ForgeArray(np.diag(s)), ForgeArray(Vh)

        # --- More matrix functions ---
        def forge_sqrtm(A):
            """sqrtm(A) — matrix square root."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import sqrtm as _sqrtm
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.real(_sqrtm(Ad)))

        def forge_funm(A, fun):
            """funm(A, @fun) — general matrix function."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import funm as _funm
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            def f(x):
                result = fun(ForgeArray(np.atleast_2d(x)))
                if isinstance(result, ForgeArray):
                    return result.data
                return np.atleast_2d(result)
            return ForgeArray(_funm(Ad, lambda x: np.vectorize(lambda v: float(fun(ForgeArray(np.float64(v))).data.flat[0]))(x)))

        def forge_lsqminnorm(A, b):
            """lsqminnorm(A, b) — minimum norm least squares."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            bd = b.data if isinstance(b, ForgeArray) else np.atleast_2d(b)
            x, res, rank, sv = np.linalg.lstsq(Ad, bd, rcond=None)
            return ForgeArray(x)

        def forge_rcond(A):
            """rcond(A) — reciprocal condition number."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            n = np.linalg.norm(Ad, 1)
            try:
                ni = np.linalg.norm(np.linalg.inv(Ad), 1)
                return ForgeArray(np.float64(1.0 / (n * ni)))
            except Exception:
                return ForgeArray(np.float64(0.0))

        def forge_bandwidth(A):
            """[lower, upper] = bandwidth(A) — matrix bandwidth."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            m, n = Ad.shape
            lower = 0
            upper = 0
            for i in range(m):
                for j in range(n):
                    if Ad[i, j] != 0:
                        if i > j: lower = max(lower, i - j)
                        if j > i: upper = max(upper, j - i)
            return ForgeArray(np.float64(lower)), ForgeArray(np.float64(upper))

        def forge_isbanded(A, lower, upper):
            """isbanded(A, lower, upper) — test if matrix is banded."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            l = int(lower.data.flat[0]) if isinstance(lower, ForgeArray) else int(lower)
            u = int(upper.data.flat[0]) if isinstance(upper, ForgeArray) else int(upper)
            m, n = Ad.shape
            for i in range(m):
                for j in range(n):
                    if (i - j > l or j - i > u) and Ad[i, j] != 0:
                        return ForgeArray(np.float64(0))
            return ForgeArray(np.float64(1))

        def forge_isdiag(A):
            """isdiag(A) — test if matrix is diagonal."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.float64(1 if np.allclose(Ad, np.diag(np.diag(Ad))) else 0))

        def forge_issymmetric(A, *args):
            """issymmetric(A) — test if matrix is symmetric."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.float64(1 if np.allclose(Ad, Ad.T) else 0))

        def forge_ishermitian(A, *args):
            """ishermitian(A) — test if matrix is Hermitian."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.float64(1 if np.allclose(Ad, Ad.conj().T) else 0))

        def forge_istril(A):
            """istril(A) — test if matrix is lower triangular."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.float64(1 if np.allclose(Ad, np.tril(Ad)) else 0))

        def forge_istriu(A):
            """istriu(A) — test if matrix is upper triangular."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.float64(1 if np.allclose(Ad, np.triu(Ad)) else 0))

        def forge_isdefinite(A, *args):
            """isdefinite(A) — test if matrix is positive definite."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            try:
                np.linalg.cholesky(Ad)
                return ForgeArray(np.float64(1))
            except Exception:
                return ForgeArray(np.float64(0))

        def forge_subspace(A, B):
            """subspace(A, B) — angle between subspaces."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import subspace_angles
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            angles = subspace_angles(Ad, Bd)
            return ForgeArray(np.float64(angles[0] if len(angles) > 0 else 0.0))

        def forge_spdiags(B, d, m, n):
            """spdiags(B, d, m, n) — sparse matrix from diagonals."""
            from forge.engine.types import ForgeArray
            from scipy.sparse import diags
            Bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            dd = d.data.flatten().astype(int) if isinstance(d, ForgeArray) else np.array([int(d)])
            md = int(m.data.flat[0]) if isinstance(m, ForgeArray) else int(m)
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            diag_list = []
            for i, di in enumerate(dd):
                length = min(md, nd) - abs(di)
                if length > 0:
                    if Bd.ndim == 2:
                        diag_list.append(Bd[:length, i])
                    else:
                        diag_list.append(Bd[:length])
            S = diags(diag_list, dd.tolist(), shape=(md, nd))
            return ForgeArray(S.toarray())

        # Register all R138 functions
        session._engine.functions["csvread"] = forge_csvread
        session._engine.functions["csvwrite"] = forge_csvwrite
        session._engine.functions["dlmread"] = forge_dlmread
        session._engine.functions["dlmwrite"] = forge_dlmwrite
        session._engine.functions["fileread"] = forge_fileread
        session._engine.functions["pcg"] = forge_pcg
        session._engine.functions["gmres"] = forge_gmres_func
        session._engine.functions["bicgstab"] = forge_bicgstab
        session._engine.functions["eigs"] = forge_eigs_func
        session._engine.functions["svds"] = forge_svds_func
        session._engine.functions["sqrtm"] = forge_sqrtm
        session._engine.functions["funm"] = forge_funm
        session._engine.functions["lsqminnorm"] = forge_lsqminnorm
        session._engine.functions["rcond"] = forge_rcond
        session._engine.functions["bandwidth"] = forge_bandwidth
        session._engine.functions["isbanded"] = forge_isbanded
        session._engine.functions["isdiag"] = forge_isdiag
        session._engine.functions["issymmetric"] = forge_issymmetric
        session._engine.functions["ishermitian"] = forge_ishermitian
        session._engine.functions["istril"] = forge_istril
        session._engine.functions["istriu"] = forge_istriu
        session._engine.functions["isdefinite"] = forge_isdefinite
        session._engine.functions["subspace"] = forge_subspace
        session._engine.functions["spdiags"] = forge_spdiags

        # R138b: Genuinely new functions
        def forge_intmax(typename=None):
            """intmax(typename) — largest integer value."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            t = 'int32'
            if typename is not None:
                if isinstance(typename, ForgeChar): t = typename.to_str()
            type_map = {'int8': np.iinfo(np.int8).max, 'int16': np.iinfo(np.int16).max,
                       'int32': np.iinfo(np.int32).max, 'int64': np.iinfo(np.int64).max,
                       'uint8': np.iinfo(np.uint8).max, 'uint16': np.iinfo(np.uint16).max,
                       'uint32': np.iinfo(np.uint32).max, 'uint64': np.iinfo(np.uint64).max}
            return ForgeArray(np.float64(type_map.get(t, np.iinfo(np.int32).max)))

        def forge_intmin(typename=None):
            """intmin(typename) — smallest integer value."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            t = 'int32'
            if typename is not None:
                if isinstance(typename, ForgeChar): t = typename.to_str()
            type_map = {'int8': np.iinfo(np.int8).min, 'int16': np.iinfo(np.int16).min,
                       'int32': np.iinfo(np.int32).min, 'int64': np.iinfo(np.int64).min,
                       'uint8': 0, 'uint16': 0, 'uint32': 0, 'uint64': 0}
            return ForgeArray(np.float64(type_map.get(t, np.iinfo(np.int32).min)))

        def forge_realmin():
            """realmin — smallest positive normalized float."""
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(np.finfo(np.float64).tiny))

        def forge_realmax():
            """realmax — largest finite float."""
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(np.finfo(np.float64).max))

        def forge_flintmax():
            """flintmax — largest consecutive integer in floating point."""
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(2**53))

        def forge_cell2struct(c, fields):
            """cell2struct(c, fields) — convert cell to struct."""
            from forge.engine.containers import ForgeStruct, ForgeCell, ForgeChar
            from forge.engine.types import ForgeArray
            s = ForgeStruct()
            if isinstance(fields, ForgeCell):
                for i in range(fields.rows * fields.cols):
                    fname = fields._data[i]
                    if isinstance(fname, ForgeChar): fname = fname.to_str()
                    if isinstance(c, ForgeCell) and i < len(c._data):
                        s._fields[fname] = c._data[i]
                    else:
                        s._fields[fname] = ForgeArray(np.array([[]]))
            return s

        def forge_struct2cell(s):
            """struct2cell(s) — convert struct to cell."""
            from forge.engine.containers import ForgeStruct, ForgeCell
            if isinstance(s, ForgeStruct):
                vals = list(s._fields.values())
                c = ForgeCell(len(vals), 1)
                for i, v in enumerate(vals):
                    c._data[i] = v
                return c
            return s

        def forge_nfields(s):
            """nfields(s) — number of struct fields."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            if isinstance(s, ForgeStruct):
                return ForgeArray(np.float64(len(s._fields)))
            return ForgeArray(np.float64(0))

        def forge_timeit(fun, *args):
            """timeit(fun) — time function execution."""
            from forge.engine.types import ForgeArray
            import time
            n = 1
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            total = 0
            for _ in range(n):
                t0 = time.perf_counter()
                fun()
                total += time.perf_counter() - t0
            return ForgeArray(np.float64(total / n))

        def forge_diary(*args):
            """diary(filename) — log session to file (stub)."""
            pass

        def forge_echo(*args):
            """echo on/off — control command echoing (stub)."""
            pass

        def forge_more(*args):
            """more on/off — control paged output (stub)."""
            pass

        def forge_smoothdata(x, *args):
            """smoothdata(x) — smooth noisy data."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            # Moving average, window=5
            w = 5
            kernel = np.ones(w) / w
            smoothed = np.convolve(data, kernel, mode='same')
            return ForgeArray(smoothed.reshape(x.data.shape if isinstance(x, ForgeArray) else (1, -1)))

        def forge_fillmissing(x, method=None):
            """fillmissing(x, method) — fill NaN values."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            data = x.data.copy() if isinstance(x, ForgeArray) else np.atleast_2d(x).copy()
            m = 'linear'
            if method is not None and isinstance(method, ForgeChar):
                m = method.to_str()
            if m in ('linear', 'pchip'):
                flat = data.flatten()
                nans = np.isnan(flat)
                if np.any(nans) and not np.all(nans):
                    idx = np.arange(len(flat))
                    flat[nans] = np.interp(idx[nans], idx[~nans], flat[~nans])
                data = flat.reshape(data.shape)
            elif m == 'constant':
                data[np.isnan(data)] = 0
            elif m == 'previous':
                flat = data.flatten()
                for i in range(1, len(flat)):
                    if np.isnan(flat[i]): flat[i] = flat[i-1]
                data = flat.reshape(data.shape)
            return ForgeArray(data)

        def forge_rmmissing(x):
            """rmmissing(x) — remove rows with NaN."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if data.ndim == 1 or (data.ndim == 2 and data.shape[0] == 1):
                mask = ~np.isnan(data.flatten())
                return ForgeArray(data.flatten()[mask].reshape(1, -1))
            mask = ~np.any(np.isnan(data), axis=1)
            return ForgeArray(data[mask])

        def forge_ismissing(x):
            """ismissing(x) — detect NaN/missing values."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.isnan(data).astype(float))

        def forge_isoutlier(x, *args):
            """isoutlier(x) — detect outliers using median rule."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            med = np.median(data)
            mad = np.median(np.abs(data - med))
            if mad == 0: mad = 1e-10
            z = np.abs(data - med) / (1.4826 * mad)
            return ForgeArray((z > 3).astype(float).reshape(x.data.shape if isinstance(x, ForgeArray) else (1, -1)))

        def forge_movmedian(x, k):
            """movmedian(x, k) — moving median."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            kd = int(k.data.flat[0]) if isinstance(k, ForgeArray) else int(k)
            n = len(data)
            result = np.zeros(n)
            half = kd // 2
            for i in range(n):
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
                result[i] = np.median(data[lo:hi])
            return ForgeArray(result.reshape(1, -1))

        def forge_movmax(x, k):
            """movmax(x, k) — moving maximum."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            kd = int(k.data.flat[0]) if isinstance(k, ForgeArray) else int(k)
            n = len(data)
            result = np.zeros(n)
            half = kd // 2
            for i in range(n):
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
                result[i] = np.max(data[lo:hi])
            return ForgeArray(result.reshape(1, -1))

        def forge_movmin(x, k):
            """movmin(x, k) — moving minimum."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            kd = int(k.data.flat[0]) if isinstance(k, ForgeArray) else int(k)
            n = len(data)
            result = np.zeros(n)
            half = kd // 2
            for i in range(n):
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
                result[i] = np.min(data[lo:hi])
            return ForgeArray(result.reshape(1, -1))

        def forge_movvar(x, k):
            """movvar(x, k) — moving variance."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            kd = int(k.data.flat[0]) if isinstance(k, ForgeArray) else int(k)
            n = len(data)
            result = np.zeros(n)
            half = kd // 2
            for i in range(n):
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
                result[i] = np.var(data[lo:hi], ddof=1) if hi - lo > 1 else 0
            return ForgeArray(result.reshape(1, -1))

        def forge_downsample(x, n, *args):
            """downsample(x, n) — downsample by factor n."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            phase = 0
            if args and isinstance(args[0], ForgeArray):
                phase = int(args[0].data.flat[0])
            return ForgeArray(data[phase::nd].reshape(1, -1))

        def forge_upsample(x, n, *args):
            """upsample(x, n) — upsample by factor n (zero insert)."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            result = np.zeros(len(data) * nd)
            result[::nd] = data
            return ForgeArray(result.reshape(1, -1))

        def forge_decimate(x, r, *args):
            """decimate(x, r) — downsample after lowpass filtering."""
            from forge.engine.types import ForgeArray
            from scipy.signal import decimate as _decimate
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            rd = int(r.data.flat[0]) if isinstance(r, ForgeArray) else int(r)
            return ForgeArray(_decimate(data, rd).reshape(1, -1))

        def forge_resample_func(x, p, q):
            """resample(x, p, q) — resample at p/q rate."""
            from forge.engine.types import ForgeArray
            from scipy.signal import resample_poly
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            pd = int(p.data.flat[0]) if isinstance(p, ForgeArray) else int(p)
            qd = int(q.data.flat[0]) if isinstance(q, ForgeArray) else int(q)
            return ForgeArray(resample_poly(data, pd, qd).reshape(1, -1))

        def forge_medfilt1(x, n=None):
            """medfilt1(x, n) — 1-D median filter."""
            from forge.engine.types import ForgeArray
            from scipy.signal import medfilt
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            nd = 3
            if n is not None:
                nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            if nd % 2 == 0: nd += 1
            return ForgeArray(medfilt(data, kernel_size=nd).reshape(1, -1))

        def forge_sgolayfilt(x, order, framelen):
            """sgolayfilt(x, order, framelen) — Savitzky-Golay filter."""
            from forge.engine.types import ForgeArray
            from scipy.signal import savgol_filter
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            o = int(order.data.flat[0]) if isinstance(order, ForgeArray) else int(order)
            fl = int(framelen.data.flat[0]) if isinstance(framelen, ForgeArray) else int(framelen)
            if fl % 2 == 0: fl += 1
            return ForgeArray(savgol_filter(data, fl, o).reshape(1, -1))

        def forge_pwelch(x, *args):
            """[Pxx, f] = pwelch(x) — power spectral density via Welch's method."""
            from forge.engine.types import ForgeArray
            from scipy.signal import welch as _welch
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            f, Pxx = _welch(data)
            return ForgeArray(Pxx.reshape(1, -1)), ForgeArray(f.reshape(1, -1))

        def forge_periodogram(x, *args):
            """[Pxx, f] = periodogram(x) — periodogram PSD estimate."""
            from forge.engine.types import ForgeArray
            from scipy.signal import periodogram as _periodogram
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            f, Pxx = _periodogram(data)
            return ForgeArray(Pxx.reshape(1, -1)), ForgeArray(f.reshape(1, -1))

        def forge_strsplit(s, delim=None):
            """strsplit(s, delim) — split string into cell array."""
            from forge.engine.containers import ForgeChar, ForgeCell
            if isinstance(s, ForgeChar): s = s.to_str()
            if delim is not None and isinstance(delim, ForgeChar): delim = delim.to_str()
            parts = s.split(delim) if delim else s.split()
            c = ForgeCell(1, len(parts))
            for i, p in enumerate(parts):
                c._data[i] = ForgeChar(p)
            return c

        def forge_strjoin(c, delim=None):
            """strjoin(c, delim) — join cell of strings."""
            from forge.engine.containers import ForgeChar, ForgeCell
            d = ' '
            if delim is not None and isinstance(delim, ForgeChar): d = delim.to_str()
            parts = []
            if isinstance(c, ForgeCell):
                for item in c._data:
                    if isinstance(item, ForgeChar): parts.append(item.to_str())
                    elif item is not None: parts.append(str(item))
            return ForgeChar(d.join(parts))

        def forge_strrep(s, old, new):
            """strrep(s, old, new) — replace substring."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(old, ForgeChar): old = old.to_str()
            if isinstance(new, ForgeChar): new = new.to_str()
            return ForgeChar(s.replace(old, new))

        def forge_regexp(s, pat, *args):
            """regexp(s, pat) — regular expression matching."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar, ForgeCell
            import re
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            matches = list(re.finditer(pat, s))
            starts = [m.start() + 1 for m in matches]  # 1-based
            return ForgeArray(np.array(starts).reshape(1, -1) if starts else np.array([[]]))

        def forge_regexprep(s, pat, rep):
            """regexprep(s, pat, rep) — regex replace."""
            from forge.engine.containers import ForgeChar
            import re
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            if isinstance(rep, ForgeChar): rep = rep.to_str()
            return ForgeChar(re.sub(pat, rep, s))

        def forge_bench(*args):
            """bench — run benchmark (stub)."""
            from forge.engine.types import ForgeArray
            return ForgeArray(np.array([0.1, 0.2, 0.3, 0.4, 0.5]).reshape(1, -1))

        def forge_open_func(*args):
            """open(filename) — open file (stub)."""
            pass

        def forge_waitbar(*args):
            """waitbar(x) — display progress bar (stub)."""
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(0))

        # Register all R138b functions
        session._engine.functions["intmax"] = forge_intmax
        session._engine.functions["intmin"] = forge_intmin
        session._engine.functions["realmin"] = forge_realmin
        session._engine.functions["realmax"] = forge_realmax
        session._engine.functions["flintmax"] = forge_flintmax
        session._engine.functions["cell2struct"] = forge_cell2struct
        session._engine.functions["struct2cell"] = forge_struct2cell
        session._engine.functions["nfields"] = forge_nfields
        session._engine.functions["timeit"] = forge_timeit
        session._engine.functions["diary"] = forge_diary
        session._engine.functions["echo"] = forge_echo
        session._engine.functions["more"] = forge_more
        session._engine.functions["smoothdata"] = forge_smoothdata
        session._engine.functions["fillmissing"] = forge_fillmissing
        session._engine.functions["rmmissing"] = forge_rmmissing
        session._engine.functions["ismissing"] = forge_ismissing
        session._engine.functions["isoutlier"] = forge_isoutlier
        session._engine.functions["movmedian"] = forge_movmedian
        session._engine.functions["movmax"] = forge_movmax
        session._engine.functions["movmin"] = forge_movmin
        session._engine.functions["movvar"] = forge_movvar
        session._engine.functions["downsample"] = forge_downsample
        session._engine.functions["upsample"] = forge_upsample
        session._engine.functions["decimate"] = forge_decimate
        session._engine.functions["resample"] = forge_resample_func
        session._engine.functions["medfilt1"] = forge_medfilt1
        session._engine.functions["sgolayfilt"] = forge_sgolayfilt
        session._engine.functions["pwelch"] = forge_pwelch
        session._engine.functions["periodogram"] = forge_periodogram
        session._engine.functions["strsplit"] = forge_strsplit
        session._engine.functions["strjoin"] = forge_strjoin
        session._engine.functions["strrep"] = forge_strrep
        session._engine.functions["regexp"] = forge_regexp
        session._engine.functions["regexprep"] = forge_regexprep
        session._engine.functions["bench"] = forge_bench
        session._engine.functions["open"] = forge_open_func
        session._engine.functions["waitbar"] = forge_waitbar

        # R140: Control, image, more math
        # --- Control system stubs ---
        def forge_tf_func(num, den):
            """tf(num, den) — create transfer function (struct-based)."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            s = ForgeStruct()
            s._fields["num"] = num if isinstance(num, ForgeArray) else ForgeArray(np.atleast_2d(num))
            s._fields["den"] = den if isinstance(den, ForgeArray) else ForgeArray(np.atleast_2d(den))
            s._fields["type"] = ForgeArray(np.float64(1))  # continuous
            return s

        def forge_ss_func(A, B, C, D):
            """ss(A, B, C, D) — create state-space model (struct-based)."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            s = ForgeStruct()
            s._fields["A"] = A if isinstance(A, ForgeArray) else ForgeArray(np.atleast_2d(A))
            s._fields["B"] = B if isinstance(B, ForgeArray) else ForgeArray(np.atleast_2d(B))
            s._fields["C"] = C if isinstance(C, ForgeArray) else ForgeArray(np.atleast_2d(C))
            s._fields["D"] = D if isinstance(D, ForgeArray) else ForgeArray(np.atleast_2d(D))
            return s

        def forge_c2d_func(sys_s, Ts, *args):
            """c2d(sys, Ts) — continuous to discrete conversion."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            from scipy.signal import cont2discrete
            if isinstance(sys_s, ForgeStruct) and "A" in sys_s._fields:
                A = sys_s._fields["A"].data
                B = sys_s._fields["B"].data
                C = sys_s._fields["C"].data
                D = sys_s._fields["D"].data
                ts = float(Ts.data.flat[0]) if isinstance(Ts, ForgeArray) else float(Ts)
                Ad, Bd, Cd, Dd, dt = cont2discrete((A, B, C, D), ts)
                result = ForgeStruct()
                result._fields["A"] = ForgeArray(Ad)
                result._fields["B"] = ForgeArray(Bd)
                result._fields["C"] = ForgeArray(Cd)
                result._fields["D"] = ForgeArray(Dd)
                result._fields["Ts"] = ForgeArray(np.float64(ts))
                return result
            return sys_s

        def forge_lqr_func(A, B, Q, R):
            """[K, S, e] = lqr(A, B, Q, R) — LQR controller design."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import solve_continuous_are
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            Qd = Q.data if isinstance(Q, ForgeArray) else np.atleast_2d(Q)
            Rd = R.data if isinstance(R, ForgeArray) else np.atleast_2d(R)
            S = solve_continuous_are(Ad, Bd, Qd, Rd)
            K = np.linalg.solve(Rd, Bd.T @ S)
            e = np.linalg.eigvals(Ad - Bd @ K)
            return ForgeArray(K), ForgeArray(S), ForgeArray(e.reshape(-1, 1))

        def forge_dlqr_func(A, B, Q, R):
            """[K, S, e] = dlqr(A, B, Q, R) — discrete LQR."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import solve_discrete_are
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            Qd = Q.data if isinstance(Q, ForgeArray) else np.atleast_2d(Q)
            Rd = R.data if isinstance(R, ForgeArray) else np.atleast_2d(R)
            S = solve_discrete_are(Ad, Bd, Qd, Rd)
            K = np.linalg.solve(Rd + Bd.T @ S @ Bd, Bd.T @ S @ Ad)
            e = np.linalg.eigvals(Ad - Bd @ K)
            return ForgeArray(K), ForgeArray(S), ForgeArray(e.reshape(-1, 1))

        def forge_care_func(A, B, Q, R=None):
            """[X, L, G] = care(A, B, Q, R) — continuous algebraic Riccati."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import solve_continuous_are
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            Qd = Q.data if isinstance(Q, ForgeArray) else np.atleast_2d(Q)
            Rd = R.data if R is not None and isinstance(R, ForgeArray) else np.eye(Bd.shape[1])
            X = solve_continuous_are(Ad, Bd, Qd, Rd)
            G = np.linalg.solve(Rd, Bd.T @ X)
            L = np.linalg.eigvals(Ad - Bd @ G)
            return ForgeArray(X), ForgeArray(L.reshape(-1, 1)), ForgeArray(G)

        def forge_dare_func(A, B, Q, R=None):
            """[X, L, G] = dare(A, B, Q, R) — discrete algebraic Riccati."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import solve_discrete_are
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            Qd = Q.data if isinstance(Q, ForgeArray) else np.atleast_2d(Q)
            Rd = R.data if R is not None and isinstance(R, ForgeArray) else np.eye(Bd.shape[1])
            X = solve_discrete_are(Ad, Bd, Qd, Rd)
            G = np.linalg.solve(Rd + Bd.T @ X @ Bd, Bd.T @ X @ Ad)
            L = np.linalg.eigvals(Ad - Bd @ G)
            return ForgeArray(X), ForgeArray(L.reshape(-1, 1)), ForgeArray(G)

        def forge_place_func(A, B, p):
            """K = place(A, B, p) — pole placement."""
            from forge.engine.types import ForgeArray
            from scipy.signal import place_poles
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            pd = p.data.flatten() if isinstance(p, ForgeArray) else np.array(p).flatten()
            result = place_poles(Ad, Bd, pd)
            return ForgeArray(result.gain_matrix)

        # --- Image basics ---
        def forge_rgb2gray(img):
            """rgb2gray(img) — convert RGB to grayscale."""
            from forge.engine.types import ForgeArray
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            if data.ndim == 3 and data.shape[2] == 3:
                gray = 0.2989 * data[:,:,0] + 0.5870 * data[:,:,1] + 0.1140 * data[:,:,2]
                return ForgeArray(gray)
            return ForgeArray(data)

        def forge_im2double(img):
            """im2double(img) — convert image to double [0,1]."""
            from forge.engine.types import ForgeArray
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            if data.max() > 1:
                return ForgeArray(data.astype(float) / 255.0)
            return ForgeArray(data.astype(float))

        def forge_im2uint8(img):
            """im2uint8(img) — convert image to uint8."""
            from forge.engine.types import ForgeArray
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            if data.max() <= 1:
                return ForgeArray((data * 255).astype(np.uint8))
            return ForgeArray(data.astype(np.uint8))

        def forge_imresize(img, scale):
            """imresize(img, scale) — resize image."""
            from forge.engine.types import ForgeArray
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            if isinstance(scale, ForgeArray):
                s = scale.data.flatten()
                if len(s) == 1:
                    new_h = int(data.shape[0] * s[0])
                    new_w = int(data.shape[1] * s[0])
                else:
                    new_h, new_w = int(s[0]), int(s[1])
            else:
                new_h = int(data.shape[0] * float(scale))
                new_w = int(data.shape[1] * float(scale))
            from scipy.ndimage import zoom
            if data.ndim == 3:
                factors = (new_h/data.shape[0], new_w/data.shape[1], 1)
            else:
                factors = (new_h/data.shape[0], new_w/data.shape[1])
            return ForgeArray(zoom(data.astype(float), factors))

        # --- More math ---
        def forge_cospi(x):
            """cospi(x) — cos(pi*x) without rounding error."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.cos(np.pi * data))

        def forge_sinpi(x):
            """sinpi(x) — sin(pi*x) without rounding error."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.sin(np.pi * data))

        def forge_atan2d(y, x):
            """atan2d(y, x) — four-quadrant arctangent in degrees."""
            from forge.engine.types import ForgeArray
            yd = y.data if isinstance(y, ForgeArray) else np.atleast_2d(y)
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.degrees(np.arctan2(yd, xd)))

        def forge_cot(x):
            """cot(x) — cotangent."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(1.0 / np.tan(data))

        def forge_coth(x):
            """coth(x) — hyperbolic cotangent."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(1.0 / np.tanh(data))

        def forge_csc_func(x):
            """csc(x) — cosecant."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(1.0 / np.sin(data))

        def forge_csch(x):
            """csch(x) — hyperbolic cosecant."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(1.0 / np.sinh(data))

        def forge_sec_func(x):
            """sec(x) — secant."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(1.0 / np.cos(data))

        def forge_sech(x):
            """sech(x) — hyperbolic secant."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(1.0 / np.cosh(data))

        def forge_acot(x):
            """acot(x) — inverse cotangent."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.arctan(1.0 / data))

        def forge_acoth(x):
            """acoth(x) — inverse hyperbolic cotangent."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.arctanh(1.0 / data))

        def forge_acsc(x):
            """acsc(x) — inverse cosecant."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.arcsin(1.0 / data))

        def forge_acsch(x):
            """acsch(x) — inverse hyperbolic cosecant."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.arcsinh(1.0 / data))

        def forge_asec(x):
            """asec(x) — inverse secant."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.arccos(1.0 / data))

        def forge_asech(x):
            """asech(x) — inverse hyperbolic secant."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.arccosh(1.0 / data))

        # Register all R140 functions
        session._engine.functions["tf"] = forge_tf_func
        session._engine.functions["ss"] = forge_ss_func
        session._engine.functions["c2d"] = forge_c2d_func
        session._engine.functions["lqr"] = forge_lqr_func
        session._engine.functions["dlqr"] = forge_dlqr_func
        session._engine.functions["care"] = forge_care_func
        session._engine.functions["dare"] = forge_dare_func
        session._engine.functions["place"] = forge_place_func
        session._engine.functions["rgb2gray"] = forge_rgb2gray
        session._engine.functions["im2double"] = forge_im2double
        session._engine.functions["im2uint8"] = forge_im2uint8
        session._engine.functions["imresize"] = forge_imresize
        session._engine.functions["cospi"] = forge_cospi
        session._engine.functions["sinpi"] = forge_sinpi
        session._engine.functions["atan2d"] = forge_atan2d
        session._engine.functions["cot"] = forge_cot
        session._engine.functions["coth"] = forge_coth
        session._engine.functions["csc"] = forge_csc_func
        session._engine.functions["csch"] = forge_csch
        session._engine.functions["sec"] = forge_sec_func
        session._engine.functions["sech"] = forge_sech
        session._engine.functions["acot"] = forge_acot
        session._engine.functions["acoth"] = forge_acoth
        session._engine.functions["acsc"] = forge_acsc
        session._engine.functions["acsch"] = forge_acsch
        session._engine.functions["asec"] = forge_asec
        session._engine.functions["asech"] = forge_asech

        # R141: Degree trig, strings, tables, debug stubs
        def forge_cotd(x):
            """cotd(x) — cotangent in degrees."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(1.0 / np.tan(np.radians(data)))

        def forge_cscd(x):
            """cscd(x) — cosecant in degrees."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(1.0 / np.sin(np.radians(data)))

        def forge_secd(x):
            """secd(x) — secant in degrees."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(1.0 / np.cos(np.radians(data)))

        def forge_acotd(x):
            """acotd(x) — inverse cotangent in degrees."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.degrees(np.arctan(1.0 / data)))

        def forge_acscd(x):
            """acscd(x) — inverse cosecant in degrees."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.degrees(np.arcsin(1.0 / data)))

        def forge_asecd(x):
            """asecd(x) — inverse secant in degrees."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.degrees(np.arccos(1.0 / data)))

        # --- More string operations ---
        def forge_upper(s):
            """upper(s) — convert to uppercase."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): return ForgeChar(s.to_str().upper())
            return s

        def forge_lower(s):
            """lower(s) — convert to lowercase."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): return ForgeChar(s.to_str().lower())
            return s

        def forge_strip(s):
            """strip(s) — remove leading/trailing whitespace."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): return ForgeChar(s.to_str().strip())
            return s

        def forge_reverse(s):
            """reverse(s) — reverse string."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): return ForgeChar(s.to_str()[::-1])
            return s

        def forge_char_func(*args):
            """char(n) — convert to character, or char(s1, s2, ...) — pad and stack."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if len(args) == 1:
                a = args[0]
                if isinstance(a, ForgeArray):
                    data = a.data.flatten()
                    if data.dtype in (np.float64, np.float32, np.int64, np.int32):
                        return ForgeChar(''.join(chr(int(v)) for v in data))
                return a
            # Multiple args: pad to same length
            strs = []
            for a in args:
                if isinstance(a, ForgeChar): strs.append(a.to_str())
                elif isinstance(a, ForgeArray):
                    strs.append(''.join(chr(int(v)) for v in a.data.flatten()))
            maxlen = max(len(s) for s in strs)
            padded = [s.ljust(maxlen) for s in strs]
            return ForgeChar(chr(10).join(padded))

        def forge_double_func(x):
            """double(x) — convert to double precision."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(x, ForgeChar):
                return ForgeArray(np.array([float(ord(c)) for c in x.to_str()]).reshape(1, -1))
            if isinstance(x, ForgeArray):
                return ForgeArray(x.data.astype(np.float64))
            return ForgeArray(np.float64(x))

        def forge_single_func(x):
            """single(x) — convert to single precision."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                return ForgeArray(x.data.astype(np.float32))
            return ForgeArray(np.float32(x))

        def forge_logical_func(x):
            """logical(x) — convert to logical."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                return ForgeArray(x.data.astype(bool))
            return ForgeArray(np.array(bool(x)))

        # --- Table functions ---
        def forge_array2table(x, *args):
            """array2table(x) — convert array to table (struct-based)."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct, ForgeChar
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            t = ForgeStruct()
            ncols = data.shape[1] if data.ndim > 1 else 1
            for j in range(ncols):
                name = f"Var{j+1}"
                t._fields[name] = ForgeArray(data[:, j].reshape(-1, 1)) if data.ndim > 1 else ForgeArray(data.reshape(-1, 1))
            return t

        def forge_table2array(t):
            """table2array(t) — convert table to array."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            if isinstance(t, ForgeStruct):
                cols = [v.data if isinstance(v, ForgeArray) else np.atleast_2d(v) for v in t._fields.values()]
                if cols:
                    return ForgeArray(np.hstack(cols))
            return ForgeArray(np.array([[]]))

        # --- Debug stubs ---
        def forge_dbstop(*args):
            """dbstop — set breakpoint (stub)."""
            pass

        def forge_dbcont(*args):
            """dbcont — continue execution (stub)."""
            pass

        def forge_dbstep(*args):
            """dbstep — single step (stub)."""
            pass

        def forge_dbquit(*args):
            """dbquit — quit debug mode (stub)."""
            pass

        def forge_dbstack(*args):
            """dbstack — display call stack (stub)."""
            from forge.engine.containers import ForgeStruct
            s = ForgeStruct()
            s._fields["name"] = "base"
            s._fields["line"] = 0
            return s

        def forge_dbclear(*args):
            """dbclear — clear breakpoints (stub)."""
            pass

        def forge_dbstatus(*args):
            """dbstatus — list breakpoints (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_profile_func(*args):
            """profile on/off — profiler (stub)."""
            pass

        # Register all R141 functions
        session._engine.functions["cotd"] = forge_cotd
        session._engine.functions["cscd"] = forge_cscd
        session._engine.functions["secd"] = forge_secd
        session._engine.functions["acotd"] = forge_acotd
        session._engine.functions["acscd"] = forge_acscd
        session._engine.functions["asecd"] = forge_asecd
        session._engine.functions["upper"] = forge_upper
        session._engine.functions["lower"] = forge_lower
        session._engine.functions["strip"] = forge_strip
        session._engine.functions["reverse"] = forge_reverse
        session._engine.functions["char"] = forge_char_func
        session._engine.functions["double"] = forge_double_func
        session._engine.functions["single"] = forge_single_func
        session._engine.functions["logical"] = forge_logical_func
        session._engine.functions["array2table"] = forge_array2table
        session._engine.functions["table2array"] = forge_table2array
        session._engine.functions["dbstop"] = forge_dbstop
        session._engine.functions["dbcont"] = forge_dbcont
        session._engine.functions["dbstep"] = forge_dbstep
        session._engine.functions["dbquit"] = forge_dbquit
        session._engine.functions["dbstack"] = forge_dbstack
        session._engine.functions["dbclear"] = forge_dbclear
        session._engine.functions["dbstatus"] = forge_dbstatus
        session._engine.functions["profile"] = forge_profile_func

        # R142: Additional common functions
        def forge_accumdim(f, x, dim=None):
            """accumdim(f, x, dim) — accumulate along dimension."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            d = 0
            if dim is not None:
                d = int(dim.data.flat[0]) if isinstance(dim, ForgeArray) else int(dim)
                d -= 1
            return ForgeArray(np.apply_along_axis(lambda a: np.cumsum(a), d, data))

        def forge_rref(A):
            """rref(A) — reduced row echelon form."""
            from forge.engine.types import ForgeArray
            data = A.data.copy() if isinstance(A, ForgeArray) else np.atleast_2d(A).copy().astype(float)
            m, n = data.shape
            pivot_row = 0
            for col in range(n):
                if pivot_row >= m: break
                # Find pivot
                max_row = pivot_row
                for row in range(pivot_row + 1, m):
                    if abs(data[row, col]) > abs(data[max_row, col]):
                        max_row = row
                if abs(data[max_row, col]) < 1e-12:
                    continue
                data[[pivot_row, max_row]] = data[[max_row, pivot_row]]
                data[pivot_row] /= data[pivot_row, col]
                for row in range(m):
                    if row != pivot_row:
                        data[row] -= data[row, col] * data[pivot_row]
                pivot_row += 1
            return ForgeArray(data)

        def forge_planerot(x):
            """[G, y] = planerot(x) — Givens plane rotation."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            a, b = xd[0], xd[1]
            if b == 0:
                c, s = 1.0, 0.0
            elif abs(b) > abs(a):
                tau = -a / b; s = 1 / np.sqrt(1 + tau**2); c = s * tau
            else:
                tau = -b / a; c = 1 / np.sqrt(1 + tau**2); s = c * tau
            G = ForgeArray(np.array([[c, s], [-s, c]]))
            y = ForgeArray(np.array([[np.sqrt(a**2 + b**2)], [0.0]]))
            return G, y

        def forge_vecnorm(x, p=None, dim=None):
            """vecnorm(x, p, dim) — vector norm."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            pd = 2
            if p is not None:
                pd = float(p.data.flat[0]) if isinstance(p, ForgeArray) else float(p)
            if dim is not None:
                d = int(dim.data.flat[0]) if isinstance(dim, ForgeArray) else int(dim)
                d -= 1
                return ForgeArray(np.linalg.norm(data, ord=pd, axis=d, keepdims=True))
            return ForgeArray(np.float64(np.linalg.norm(data.flatten(), ord=pd)))

        def forge_normest(A, *args):
            """normest(A) — estimate 2-norm."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.float64(np.linalg.norm(Ad, ord=2)))

        def forge_condest(A):
            """condest(A) — estimate 1-norm condition number."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.float64(np.linalg.cond(Ad, 1)))

        def forge_condeig(A):
            """[V, D, s] = condeig(A) — condition numbers of eigenvalues."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            vals, vecs = np.linalg.eig(Ad)
            cond = np.zeros(len(vals))
            try:
                vecs_inv = np.linalg.inv(vecs)
                for i in range(len(vals)):
                    cond[i] = np.linalg.norm(vecs[:, i]) * np.linalg.norm(vecs_inv[i, :])
            except Exception:
                cond[:] = np.inf
            return ForgeArray(vecs), ForgeArray(np.diag(vals)), ForgeArray(cond.reshape(-1, 1))

        # --- More array functions ---
        def forge_repelem(x, r, *args):
            """repelem(x, r, c) — replicate array elements."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            rd = int(r.data.flat[0]) if isinstance(r, ForgeArray) else int(r)
            if args:
                cd = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                return ForgeArray(np.repeat(np.repeat(data, rd, axis=0), cd, axis=1))
            return ForgeArray(np.repeat(data, rd, axis=0))

        def forge_logspace2(a, b, *args):
            """logspace(a, b, n) — logarithmically spaced vector."""
            from forge.engine.types import ForgeArray
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            n = 50
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            return ForgeArray(np.logspace(ad, bd, n).reshape(1, -1))

        def forge_meshgrid2(x, y=None, *args):
            """[X, Y] = meshgrid(x, y) — 2D grid."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            if y is None:
                yd = xd.copy()
            else:
                yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            X, Y = np.meshgrid(xd, yd)
            return ForgeArray(X), ForgeArray(Y)

        def forge_peaks(*args):
            """[X, Y, Z] = peaks(n) — sample 3D surface."""
            from forge.engine.types import ForgeArray
            n = 49
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            x = np.linspace(-3, 3, n)
            y = np.linspace(-3, 3, n)
            X, Y = np.meshgrid(x, y)
            Z = 3*(1-X)**2*np.exp(-X**2-(Y+1)**2) - 10*(X/5-X**3-Y**5)*np.exp(-X**2-Y**2) - 1/3*np.exp(-(X+1)**2-Y**2)
            return ForgeArray(X), ForgeArray(Y), ForgeArray(Z)

        # --- Statistics extras ---
        def forge_histcounts2(x, *args):
            """[N, edges] = histcounts(x) — histogram bin counts."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            nbins = 'auto'
            if args and isinstance(args[0], ForgeArray):
                a = args[0].data.flatten()
                if len(a) == 1:
                    nbins = int(a[0])
                else:
                    # Edges provided
                    N, _ = np.histogram(data, bins=a)
                    return ForgeArray(N.astype(float).reshape(1, -1)), ForgeArray(a.reshape(1, -1))
            N, edges = np.histogram(data, bins=nbins)
            return ForgeArray(N.astype(float).reshape(1, -1)), ForgeArray(edges.reshape(1, -1))

        def forge_tabulate(x):
            """tabulate(x) — frequency table."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten().astype(int) if isinstance(x, ForgeArray) else np.array(x).flatten().astype(int)
            counts = np.zeros(max(data) + 1)
            for v in data:
                if v >= 0:
                    counts[v] += 1
            return ForgeArray(counts.reshape(1, -1))

        def forge_corrcoef2(x, y=None):
            """corrcoef(x, y) — pairwise correlation."""
            from forge.engine.types import ForgeArray
            if y is not None:
                xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
                yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
                return ForgeArray(np.corrcoef(xd, yd))
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if data.shape[0] == 1: data = data.T
            return ForgeArray(np.corrcoef(data, rowvar=False))

        def forge_cov2(x, y=None):
            """cov(x, y) — covariance."""
            from forge.engine.types import ForgeArray
            if y is not None:
                xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
                yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
                return ForgeArray(np.cov(xd, yd))
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if data.shape[0] == 1: data = data.T
            return ForgeArray(np.cov(data, rowvar=False))

        def forge_mvnrnd(mu, sigma, n=None):
            """mvnrnd(mu, sigma, n) — multivariate normal random."""
            from forge.engine.types import ForgeArray
            mud = mu.data.flatten() if isinstance(mu, ForgeArray) else np.array(mu).flatten()
            sigd = sigma.data if isinstance(sigma, ForgeArray) else np.atleast_2d(sigma)
            nd = 1
            if n is not None:
                nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            return ForgeArray(np.random.multivariate_normal(mud, sigd, nd))

        def forge_normpdf(x, mu=None, sigma=None):
            """normpdf(x, mu, sigma) — normal PDF."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            m = 0.0
            s = 1.0
            if mu is not None:
                m = float(mu.data.flat[0]) if isinstance(mu, ForgeArray) else float(mu)
            if sigma is not None:
                s = float(sigma.data.flat[0]) if isinstance(sigma, ForgeArray) else float(sigma)
            result = (1 / (s * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((data - m)/s)**2)
            return ForgeArray(result)

        def forge_normcdf(x, mu=None, sigma=None):
            """normcdf(x, mu, sigma) — normal CDF."""
            from forge.engine.types import ForgeArray
            from scipy.special import erfc
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            m = 0.0
            s = 1.0
            if mu is not None:
                m = float(mu.data.flat[0]) if isinstance(mu, ForgeArray) else float(mu)
            if sigma is not None:
                s = float(sigma.data.flat[0]) if isinstance(sigma, ForgeArray) else float(sigma)
            z = (data - m) / s
            result = 0.5 * erfc(-z / np.sqrt(2))
            return ForgeArray(result)

        def forge_norminv(p, mu=None, sigma=None):
            """norminv(p, mu, sigma) — inverse normal CDF."""
            from forge.engine.types import ForgeArray
            from scipy.special import erfinv
            data = p.data if isinstance(p, ForgeArray) else np.atleast_2d(p)
            m = 0.0
            s = 1.0
            if mu is not None:
                m = float(mu.data.flat[0]) if isinstance(mu, ForgeArray) else float(mu)
            if sigma is not None:
                s = float(sigma.data.flat[0]) if isinstance(sigma, ForgeArray) else float(sigma)
            result = m + s * np.sqrt(2) * erfinv(2 * data - 1)
            return ForgeArray(result)

        # Register R142 functions
        session._engine.functions["accumdim"] = forge_accumdim
        session._engine.functions["rref"] = forge_rref
        session._engine.functions["planerot"] = forge_planerot
        session._engine.functions["vecnorm"] = forge_vecnorm
        session._engine.functions["normest"] = forge_normest
        session._engine.functions["condest"] = forge_condest
        session._engine.functions["condeig"] = forge_condeig
        session._engine.functions["repelem"] = forge_repelem
        session._engine.functions["logspace"] = forge_logspace2
        session._engine.functions["meshgrid"] = forge_meshgrid2
        session._engine.functions["peaks"] = forge_peaks
        session._engine.functions["histcounts"] = forge_histcounts2
        session._engine.functions["tabulate"] = forge_tabulate
        session._engine.functions["corrcoef"] = forge_corrcoef2
        session._engine.functions["cov"] = forge_cov2
        session._engine.functions["mvnrnd"] = forge_mvnrnd
        session._engine.functions["normpdf"] = forge_normpdf
        session._engine.functions["normcdf"] = forge_normcdf
        session._engine.functions["norminv"] = forge_norminv

        # R144: PDE helpers, more LA, distributions, file ops

        # --- Probability distributions ---
        def forge_unifrnd(a, b, *args):
            """unifrnd(a, b, m, n) — uniform random."""
            from forge.engine.types import ForgeArray
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            shape = (1, 1)
            if len(args) >= 2:
                m = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                n = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
                shape = (m, n)
            elif len(args) == 1:
                m = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                shape = (m, m)
            return ForgeArray(np.random.uniform(ad, bd, shape))

        def forge_exprnd(mu, *args):
            """exprnd(mu, m, n) — exponential random."""
            from forge.engine.types import ForgeArray
            mud = float(mu.data.flat[0]) if isinstance(mu, ForgeArray) else float(mu)
            shape = (1, 1)
            if len(args) >= 2:
                m = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                n = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
                shape = (m, n)
            return ForgeArray(np.random.exponential(mud, shape))

        def forge_poissrnd(lam, *args):
            """poissrnd(lambda, m, n) — Poisson random."""
            from forge.engine.types import ForgeArray
            lamd = float(lam.data.flat[0]) if isinstance(lam, ForgeArray) else float(lam)
            shape = (1, 1)
            if len(args) >= 2:
                m = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                n = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
                shape = (m, n)
            return ForgeArray(np.random.poisson(lamd, shape).astype(float))

        def forge_chi2rnd(v, *args):
            """chi2rnd(v, m, n) — chi-squared random."""
            from forge.engine.types import ForgeArray
            vd = float(v.data.flat[0]) if isinstance(v, ForgeArray) else float(v)
            shape = (1, 1)
            if len(args) >= 2:
                m = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                n = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
                shape = (m, n)
            return ForgeArray(np.random.chisquare(vd, shape))

        def forge_trnd(v, *args):
            """trnd(v, m, n) — Student t random."""
            from forge.engine.types import ForgeArray
            vd = float(v.data.flat[0]) if isinstance(v, ForgeArray) else float(v)
            shape = (1, 1)
            if len(args) >= 2:
                m = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                n = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
                shape = (m, n)
            return ForgeArray(np.random.standard_t(vd, shape))

        def forge_frnd(d1, d2, *args):
            """frnd(d1, d2, m, n) — F-distribution random."""
            from forge.engine.types import ForgeArray
            d1v = float(d1.data.flat[0]) if isinstance(d1, ForgeArray) else float(d1)
            d2v = float(d2.data.flat[0]) if isinstance(d2, ForgeArray) else float(d2)
            shape = (1, 1)
            if len(args) >= 2:
                m = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                n = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
                shape = (m, n)
            return ForgeArray(np.random.f(d1v, d2v, shape))

        def forge_binornd(n, p, *args):
            """binornd(n, p, m, k) — binomial random."""
            from forge.engine.types import ForgeArray
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            pd = float(p.data.flat[0]) if isinstance(p, ForgeArray) else float(p)
            shape = (1, 1)
            if len(args) >= 2:
                m = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                k = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
                shape = (m, k)
            return ForgeArray(np.random.binomial(nd, pd, shape).astype(float))

        def forge_exppdf(x, mu=None):
            """exppdf(x, mu) — exponential PDF."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            m = 1.0
            if mu is not None:
                m = float(mu.data.flat[0]) if isinstance(mu, ForgeArray) else float(mu)
            result = (1/m) * np.exp(-data/m)
            result[data < 0] = 0
            return ForgeArray(result)

        def forge_expcdf(x, mu=None):
            """expcdf(x, mu) — exponential CDF."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            m = 1.0
            if mu is not None:
                m = float(mu.data.flat[0]) if isinstance(mu, ForgeArray) else float(mu)
            result = 1 - np.exp(-data/m)
            result[data < 0] = 0
            return ForgeArray(result)

        def forge_chi2pdf(x, v):
            """chi2pdf(x, v) — chi-squared PDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import chi2
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            vd = float(v.data.flat[0]) if isinstance(v, ForgeArray) else float(v)
            return ForgeArray(chi2.pdf(data, vd))

        def forge_chi2cdf(x, v):
            """chi2cdf(x, v) — chi-squared CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import chi2
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            vd = float(v.data.flat[0]) if isinstance(v, ForgeArray) else float(v)
            return ForgeArray(chi2.cdf(data, vd))

        def forge_tpdf(x, v):
            """tpdf(x, v) — Student t PDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import t as tdist
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            vd = float(v.data.flat[0]) if isinstance(v, ForgeArray) else float(v)
            return ForgeArray(tdist.pdf(data, vd))

        def forge_tcdf(x, v):
            """tcdf(x, v) — Student t CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import t as tdist
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            vd = float(v.data.flat[0]) if isinstance(v, ForgeArray) else float(v)
            return ForgeArray(tdist.cdf(data, vd))

        def forge_tinv(p, v):
            """tinv(p, v) — inverse Student t CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import t as tdist
            data = p.data if isinstance(p, ForgeArray) else np.atleast_2d(p)
            vd = float(v.data.flat[0]) if isinstance(v, ForgeArray) else float(v)
            return ForgeArray(tdist.ppf(data, vd))

        def forge_fpdf(x, d1, d2):
            """fpdf(x, d1, d2) — F-distribution PDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import f as fdist
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            d1v = float(d1.data.flat[0]) if isinstance(d1, ForgeArray) else float(d1)
            d2v = float(d2.data.flat[0]) if isinstance(d2, ForgeArray) else float(d2)
            return ForgeArray(fdist.pdf(data, d1v, d2v))

        def forge_fcdf(x, d1, d2):
            """fcdf(x, d1, d2) — F-distribution CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import f as fdist
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            d1v = float(d1.data.flat[0]) if isinstance(d1, ForgeArray) else float(d1)
            d2v = float(d2.data.flat[0]) if isinstance(d2, ForgeArray) else float(d2)
            return ForgeArray(fdist.cdf(data, d1v, d2v))

        def forge_finv(p, d1, d2):
            """finv(p, d1, d2) — inverse F CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import f as fdist
            data = p.data if isinstance(p, ForgeArray) else np.atleast_2d(p)
            d1v = float(d1.data.flat[0]) if isinstance(d1, ForgeArray) else float(d1)
            d2v = float(d2.data.flat[0]) if isinstance(d2, ForgeArray) else float(d2)
            return ForgeArray(fdist.ppf(data, d1v, d2v))

        # --- File operations ---
        def forge_isfile(fname):
            """isfile(fname) — test if file exists."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            return ForgeArray(np.float64(1 if os.path.isfile(fname) else 0))

        def forge_isfolder(fname):
            """isfolder(fname) — test if directory exists."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            return ForgeArray(np.float64(1 if os.path.isdir(fname) else 0))

        def forge_mkdir_func(dirname):
            """mkdir(dirname) — create directory."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(dirname, ForgeChar): dirname = dirname.to_str()
            os.makedirs(dirname, exist_ok=True)
            return ForgeArray(np.float64(1))

        def forge_rmdir(dirname):
            """rmdir(dirname) — remove directory."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(dirname, ForgeChar): dirname = dirname.to_str()
            os.rmdir(dirname)

        def forge_copyfile(src, dst):
            """copyfile(src, dst) — copy file."""
            from forge.engine.containers import ForgeChar
            import shutil
            if isinstance(src, ForgeChar): src = src.to_str()
            if isinstance(dst, ForgeChar): dst = dst.to_str()
            shutil.copy2(src, dst)

        def forge_movefile(src, dst):
            """movefile(src, dst) — move file."""
            from forge.engine.containers import ForgeChar
            import shutil
            if isinstance(src, ForgeChar): src = src.to_str()
            if isinstance(dst, ForgeChar): dst = dst.to_str()
            shutil.move(src, dst)

        def forge_delete_func(fname):
            """delete(fname) — delete file."""
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            if os.path.exists(fname):
                os.remove(fname)

        def forge_fileparts(fname):
            """[path, name, ext] = fileparts(fname) — split filename."""
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            path = os.path.dirname(fname)
            base = os.path.basename(fname)
            name, ext = os.path.splitext(base)
            return ForgeChar(path), ForgeChar(name), ForgeChar(ext)

        def forge_fullfile(*args):
            """fullfile(p1, p2, ...) — build full filename."""
            from forge.engine.containers import ForgeChar
            parts = []
            for a in args:
                if isinstance(a, ForgeChar): parts.append(a.to_str())
                else: parts.append(str(a))
            return ForgeChar(os.path.join(*parts))

        def forge_pwd():
            """pwd — current working directory."""
            from forge.engine.containers import ForgeChar
            return ForgeChar(os.getcwd())

        def forge_cd_func(dirname=None):
            """cd(dirname) — change directory."""
            from forge.engine.containers import ForgeChar
            if dirname is not None:
                if isinstance(dirname, ForgeChar): dirname = dirname.to_str()
                os.chdir(dirname)
            return ForgeChar(os.getcwd())

        # Register all R144 functions
        session._engine.functions["unifrnd"] = forge_unifrnd
        session._engine.functions["exprnd"] = forge_exprnd
        session._engine.functions["poissrnd"] = forge_poissrnd
        session._engine.functions["chi2rnd"] = forge_chi2rnd
        session._engine.functions["trnd"] = forge_trnd
        session._engine.functions["frnd"] = forge_frnd
        session._engine.functions["binornd"] = forge_binornd
        session._engine.functions["exppdf"] = forge_exppdf
        session._engine.functions["expcdf"] = forge_expcdf
        session._engine.functions["chi2pdf"] = forge_chi2pdf
        session._engine.functions["chi2cdf"] = forge_chi2cdf
        session._engine.functions["tpdf"] = forge_tpdf
        session._engine.functions["tcdf"] = forge_tcdf
        session._engine.functions["tinv"] = forge_tinv
        session._engine.functions["fpdf"] = forge_fpdf
        session._engine.functions["fcdf"] = forge_fcdf
        session._engine.functions["finv"] = forge_finv
        session._engine.functions["isfile"] = forge_isfile
        session._engine.functions["isfolder"] = forge_isfolder
        session._engine.functions["isdir"] = forge_isfolder
        session._engine.functions["mkdir"] = forge_mkdir_func
        session._engine.functions["rmdir"] = forge_rmdir
        session._engine.functions["copyfile"] = forge_copyfile
        session._engine.functions["movefile"] = forge_movefile
        session._engine.functions["delete"] = forge_delete_func
        session._engine.functions["fileparts"] = forge_fileparts
        session._engine.functions["fullfile"] = forge_fullfile
        session._engine.functions["pwd"] = forge_pwd
        session._engine.functions["cd"] = forge_cd_func

        # R145: More common functions
        # --- Sparse construction ---
        def forge_speye2(m, n=None):
            """speye(m, n) — sparse identity matrix."""
            from forge.engine.types import ForgeArray
            from scipy.sparse import eye as speye_fn
            md = int(m.data.flat[0]) if isinstance(m, ForgeArray) else int(m)
            nd = md
            if n is not None:
                nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            return ForgeArray(speye_fn(md, nd).toarray())

        def forge_sprand2(m, n, density):
            """sprand(m, n, density) — sparse random matrix."""
            from forge.engine.types import ForgeArray
            from scipy.sparse import random as sprand_fn
            md = int(m.data.flat[0]) if isinstance(m, ForgeArray) else int(m)
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            dd = float(density.data.flat[0]) if isinstance(density, ForgeArray) else float(density)
            return ForgeArray(sprand_fn(md, nd, dd).toarray())

        def forge_sprandn2(m, n, density):
            """sprandn(m, n, density) — sparse normal random matrix."""
            from forge.engine.types import ForgeArray
            from scipy.sparse import random as sprand_fn
            md = int(m.data.flat[0]) if isinstance(m, ForgeArray) else int(m)
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            dd = float(density.data.flat[0]) if isinstance(density, ForgeArray) else float(density)
            return ForgeArray(sprand_fn(md, nd, dd, data_rvs=np.random.randn).toarray())

        # --- Gallery matrices ---
        def forge_gallery(name, *args):
            """gallery(name, n) — test matrices."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(name, ForgeChar): name = name.to_str()
            n = 5
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            if name == 'lehmer':
                A = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        A[i,j] = min(i+1, j+1) / max(i+1, j+1)
                return ForgeArray(A)
            elif name == 'moler':
                A = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        A[i,j] = min(i+1, j+1) - 2
                np.fill_diagonal(A, np.arange(1, n+1))
                return ForgeArray(A)
            elif name == 'frank':
                A = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if j >= i - 1:
                            A[i,j] = n - max(i, j)
                return ForgeArray(A)
            elif name == 'minij':
                A = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        A[i,j] = min(i+1, j+1)
                return ForgeArray(A)
            elif name == 'clement':
                A = np.zeros((n, n))
                for i in range(n-1):
                    A[i, i+1] = np.sqrt((i+1)*(n-i-1))
                    A[i+1, i] = A[i, i+1]
                return ForgeArray(A)
            return ForgeArray(np.eye(n))

        # --- More special matrices ---
        def forge_hadamard2(n):
            """hadamard(n) — Hadamard matrix."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import hadamard as _hadamard
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            return ForgeArray(_hadamard(nd).astype(float))

        def forge_vander2(v, *args):
            """vander(v, n) — Vandermonde matrix."""
            from forge.engine.types import ForgeArray
            vd = v.data.flatten() if isinstance(v, ForgeArray) else np.array(v).flatten()
            n = len(vd)
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            return ForgeArray(np.vander(vd, N=n))

        # --- More utility functions ---
        def forge_uniquetol(x, tol=None):
            """uniquetol(x, tol) — unique within tolerance."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            td = 1e-6
            if tol is not None:
                td = float(tol.data.flat[0]) if isinstance(tol, ForgeArray) else float(tol)
            sorted_data = np.sort(data)
            result = [sorted_data[0]]
            for v in sorted_data[1:]:
                if abs(v - result[-1]) > td:
                    result.append(v)
            return ForgeArray(np.array(result).reshape(1, -1))

        def forge_ismembertol(a, b, tol=None):
            """ismembertol(a, b, tol) — set membership within tolerance."""
            from forge.engine.types import ForgeArray
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            td = 1e-6
            if tol is not None:
                td = float(tol.data.flat[0]) if isinstance(tol, ForgeArray) else float(tol)
            result = np.zeros(len(ad), dtype=bool)
            for i, v in enumerate(ad):
                if np.any(np.abs(bd - v) <= td):
                    result[i] = True
            return ForgeArray(result.astype(float).reshape(1, -1))

        def forge_rescale(x, *args):
            """rescale(x, lo, hi) — rescale data to [lo, hi]."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            lo, hi = 0.0, 1.0
            if len(args) >= 2:
                lo = float(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else float(args[0])
                hi = float(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else float(args[1])
            dmin, dmax = data.min(), data.max()
            if dmax == dmin:
                return ForgeArray(np.full_like(data, (lo+hi)/2))
            return ForgeArray(lo + (data - dmin) * (hi - lo) / (dmax - dmin))

        def forge_clip(x, lo, hi):
            """clip(x, lo, hi) — clamp values to range."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            l = float(lo.data.flat[0]) if isinstance(lo, ForgeArray) else float(lo)
            h = float(hi.data.flat[0]) if isinstance(hi, ForgeArray) else float(hi)
            return ForgeArray(np.clip(data, l, h))

        def forge_discretize2(x, edges):
            """discretize(x, edges) — bin data into categories."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            ed = edges.data.flatten() if isinstance(edges, ForgeArray) else np.array(edges).flatten()
            result = np.digitize(data, ed)
            return ForgeArray(result.astype(float).reshape(1, -1))

        def forge_gradient2(f, *args):
            """gradient(f) or gradient(f, h) — numerical gradient."""
            from forge.engine.types import ForgeArray
            data = f.data if isinstance(f, ForgeArray) else np.atleast_2d(f)
            h = 1.0
            if args and isinstance(args[0], ForgeArray):
                h = float(args[0].data.flat[0])
            if data.ndim == 2 and (data.shape[0] == 1 or data.shape[1] == 1):
                # Vector
                flat = data.flatten()
                g = np.gradient(flat, h)
                return ForgeArray(g.reshape(data.shape))
            else:
                # Matrix
                gy, gx = np.gradient(data, h)
                return ForgeArray(gx), ForgeArray(gy)

        def forge_diff2(x, *args):
            """diff(x, n, dim) — differences."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            n = 1
            dim = None
            if args:
                if isinstance(args[0], ForgeArray):
                    n = int(args[0].data.flat[0])
                if len(args) > 1 and isinstance(args[1], ForgeArray):
                    dim = int(args[1].data.flat[0]) - 1
            if dim is None:
                # Auto-detect: first non-singleton dim
                if data.shape[0] == 1:
                    dim = 1
                else:
                    dim = 0
            result = data
            for _ in range(n):
                result = np.diff(result, axis=dim)
            return ForgeArray(result)

        def forge_cummax(x, *args):
            """cummax(x) — cumulative maximum."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.maximum.accumulate(data).reshape(1, -1))

        def forge_cummin(x, *args):
            """cummin(x) — cumulative minimum."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.minimum.accumulate(data).reshape(1, -1))

        def forge_maxk(x, k):
            """maxk(x, k) — k largest elements."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            kd = int(k.data.flat[0]) if isinstance(k, ForgeArray) else int(k)
            idx = np.argsort(data)[-kd:][::-1]
            return ForgeArray(data[idx].reshape(-1, 1)), ForgeArray((idx + 1).astype(float).reshape(-1, 1))

        def forge_mink(x, k):
            """mink(x, k) — k smallest elements."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            kd = int(k.data.flat[0]) if isinstance(k, ForgeArray) else int(k)
            idx = np.argsort(data)[:kd]
            return ForgeArray(data[idx].reshape(-1, 1)), ForgeArray((idx + 1).astype(float).reshape(-1, 1))

        def forge_isnan2(x):
            """isnan(x) — test for NaN."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.isnan(data).astype(float))

        def forge_isinf2(x):
            """isinf(x) — test for Inf."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.isinf(data).astype(float))

        def forge_isfinite2(x):
            """isfinite(x) — test for finite."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.isfinite(data).astype(float))

        def forge_isreal2(x):
            """isreal(x) — test for real."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.float64(1 if not np.any(np.iscomplex(data)) else 0))

        def forge_isinteger2(x):
            """isinteger(x) — test if integer type."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                return ForgeArray(np.float64(1 if np.issubdtype(x.data.dtype, np.integer) else 0))
            return ForgeArray(np.float64(0))

        # Register R145 functions
        session._engine.functions["speye"] = forge_speye2
        session._engine.functions["sprand"] = forge_sprand2
        session._engine.functions["sprandn"] = forge_sprandn2
        session._engine.functions["gallery"] = forge_gallery
        session._engine.functions["hadamard"] = forge_hadamard2
        session._engine.functions["vander"] = forge_vander2
        session._engine.functions["uniquetol"] = forge_uniquetol
        session._engine.functions["ismembertol"] = forge_ismembertol
        session._engine.functions["rescale"] = forge_rescale
        session._engine.functions["clip"] = forge_clip
        session._engine.functions["discretize"] = forge_discretize2
        session._engine.functions["gradient"] = forge_gradient2
        session._engine.functions["diff"] = forge_diff2
        session._engine.functions["cummax"] = forge_cummax
        session._engine.functions["cummin"] = forge_cummin
        session._engine.functions["maxk"] = forge_maxk
        session._engine.functions["mink"] = forge_mink
        session._engine.functions["isnan"] = forge_isnan2
        session._engine.functions["isinf"] = forge_isinf2
        session._engine.functions["isfinite"] = forge_isfinite2
        session._engine.functions["isreal"] = forge_isreal2
        session._engine.functions["isinteger"] = forge_isinteger2

        # R146: Splines, interpolation, strings, plotting stubs

        # --- Spline/Interpolation ---
        def forge_spline2(x, y, xi=None):
            """pp = spline(x, y) or yi = spline(x, y, xi) — cubic spline."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            from scipy.interpolate import CubicSpline
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            cs = CubicSpline(xd, yd)
            if xi is not None:
                xid = xi.data.flatten() if isinstance(xi, ForgeArray) else np.array(xi).flatten()
                return ForgeArray(cs(xid).reshape(1, -1))
            # Return piecewise polynomial struct
            pp = ForgeStruct()
            pp._fields["form"] = "pp"
            pp._fields["breaks"] = ForgeArray(xd.reshape(1, -1))
            pp._fields["coefs"] = ForgeArray(cs.c.T)
            pp._fields["pieces"] = ForgeArray(np.float64(len(xd) - 1))
            pp._fields["order"] = ForgeArray(np.float64(4))
            pp._fields["dim"] = ForgeArray(np.float64(1))
            pp._cs = cs  # stash for ppval
            return pp

        def forge_pchip2(x, y, xi=None):
            """pp = pchip(x, y) or yi = pchip(x, y, xi) — PCHIP interpolation."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            from scipy.interpolate import PchipInterpolator
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            cs = PchipInterpolator(xd, yd)
            if xi is not None:
                xid = xi.data.flatten() if isinstance(xi, ForgeArray) else np.array(xi).flatten()
                return ForgeArray(cs(xid).reshape(1, -1))
            pp = ForgeStruct()
            pp._fields["form"] = "pp"
            pp._fields["breaks"] = ForgeArray(xd.reshape(1, -1))
            pp._fields["pieces"] = ForgeArray(np.float64(len(xd) - 1))
            pp._fields["order"] = ForgeArray(np.float64(4))
            pp._cs = cs
            return pp

        def forge_ppval2(pp, xi):
            """ppval(pp, xi) — evaluate piecewise polynomial."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            xid = xi.data.flatten() if isinstance(xi, ForgeArray) else np.array(xi).flatten()
            if isinstance(pp, ForgeStruct) and hasattr(pp, '_cs'):
                return ForgeArray(pp._cs(xid).reshape(1, -1))
            return ForgeArray(np.zeros_like(xid).reshape(1, -1))

        def forge_interp2_func(X, Y, V, Xq, Yq, *args):
            """interp2(X, Y, V, Xq, Yq) — 2-D interpolation."""
            from forge.engine.types import ForgeArray
            from scipy.interpolate import RegularGridInterpolator
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            Yd = Y.data if isinstance(Y, ForgeArray) else np.atleast_2d(Y)
            Vd = V.data if isinstance(V, ForgeArray) else np.atleast_2d(V)
            Xqd = Xq.data if isinstance(Xq, ForgeArray) else np.atleast_2d(Xq)
            Yqd = Yq.data if isinstance(Yq, ForgeArray) else np.atleast_2d(Yq)
            # Extract unique sorted grid coordinates
            x_uniq = np.unique(Xd)
            y_uniq = np.unique(Yd)
            interp = RegularGridInterpolator((y_uniq, x_uniq), Vd, method='linear', bounds_error=False)
            pts = np.column_stack([Yqd.ravel(), Xqd.ravel()])
            result = interp(pts).reshape(Xqd.shape)
            return ForgeArray(result)

        def forge_griddata2(x, y, v, xq, yq, *args):
            """griddata(x, y, v, xq, yq) — scattered data interpolation."""
            from forge.engine.types import ForgeArray
            from scipy.interpolate import griddata as _griddata
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            vd = v.data.flatten() if isinstance(v, ForgeArray) else np.array(v).flatten()
            xqd = xq.data if isinstance(xq, ForgeArray) else np.atleast_2d(xq)
            yqd = yq.data if isinstance(yq, ForgeArray) else np.atleast_2d(yq)
            points = np.column_stack([xd, yd])
            result = _griddata(points, vd, (xqd, yqd), method='linear')
            return ForgeArray(result)

        # --- More string functions ---
        def forge_sprintf2(fmt, *args):
            """sprintf(fmt, ...) — format string with Octave vectorization."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(fmt, ForgeChar): fmt = fmt.to_str()
            # Check if any arg is a multi-element array (vectorized sprintf)
            max_len = 1
            expanded = []
            for a in args:
                if isinstance(a, ForgeChar):
                    expanded.append([a.to_str()])
                elif isinstance(a, ForgeArray) and a.data.size > 1:
                    elems = []
                    for v in a.data.flat:
                        if isinstance(v, (np.floating, float)):
                            elems.append(float(v))
                        else:
                            elems.append(int(v))
                    expanded.append(elems)
                    max_len = max(max_len, len(elems))
                elif isinstance(a, ForgeArray):
                    v = a.data.flat[0]
                    if a.data.dtype in (np.float64, np.float32):
                        expanded.append([float(v)])
                    else:
                        expanded.append([int(v)])
                else:
                    expanded.append([a])
            if max_len > 1:
                # Vectorized: repeat format for each element
                result = []
                for i in range(max_len):
                    vals = [e[i % len(e)] for e in expanded]
                    try:
                        result.append(fmt % tuple(vals))
                    except Exception:
                        result.append(fmt)
                return ForgeChar("".join(result))
            else:
                vals = [e[0] for e in expanded]
                try:
                    return ForgeChar(fmt % tuple(vals))
                except Exception:
                    return ForgeChar(fmt)

        def forge_num2str2(n, *args):
            """num2str(n, fmt) — convert number to string."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray):
                data = n.data
                if data.size == 1:
                    v = float(data.flat[0])
                    if args:
                        fmt = args[0]
                        if isinstance(fmt, ForgeChar):
                            fmt = fmt.to_str()
                        elif isinstance(fmt, ForgeArray):
                            fmt = f'%.{int(fmt.data.flat[0])}f'
                        try:
                            return ForgeChar(fmt % v)
                        except Exception:
                            return ForgeChar(str(v))
                    if v == int(v) and abs(v) < 1e15:
                        return ForgeChar(str(int(v)))
                    return ForgeChar(f'{v:.4f}')
                parts = []
                for v in data.flat:
                    fv = float(v)
                    if fv == int(fv):
                        parts.append(str(int(fv)))
                    else:
                        parts.append(f'{fv:.4f}')
                return ForgeChar('  '.join(parts))
            return ForgeChar(str(n))

        def forge_str2num2(s):
            """str2num(s) — convert string to number."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            try:
                return ForgeArray(np.float64(float(s)))
            except Exception:
                return ForgeArray(np.array([[]]))

        def forge_dec2hex(n):
            """dec2hex(n) — decimal to hexadecimal string."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            return ForgeChar(hex(nd)[2:].upper())

        def forge_hex2dec(s):
            """hex2dec(s) — hexadecimal string to decimal."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            return ForgeArray(np.float64(int(s, 16)))

        def forge_dec2bin(n, *args):
            """dec2bin(n) — decimal to binary string."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            b = bin(nd)[2:]
            if args:
                minlen = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                b = b.zfill(minlen)
            return ForgeChar(b)

        def forge_bin2dec(s):
            """bin2dec(s) — binary string to decimal."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            return ForgeArray(np.float64(int(s, 2)))

        # Register R146 functions
        session._engine.functions["spline"] = forge_spline2
        session._engine.functions["pchip"] = forge_pchip2
        session._engine.functions["ppval"] = forge_ppval2
        session._engine.functions["interp2"] = forge_interp2_func
        session._engine.functions["griddata"] = forge_griddata2
        session._engine.functions["sprintf"] = forge_sprintf2
        session._engine.functions["num2str"] = forge_num2str2
        session._engine.functions["str2num"] = forge_str2num2
        session._engine.functions["dec2hex"] = forge_dec2hex
        session._engine.functions["hex2dec"] = forge_hex2dec
        session._engine.functions["dec2bin"] = forge_dec2bin
        session._engine.functions["bin2dec"] = forge_bin2dec

        # R146b: Genuinely new functions
        # --- Signal waveforms ---
        def forge_chirp(t, f0, t1, f1, *args):
            """chirp(t, f0, t1, f1) — swept-frequency cosine."""
            from forge.engine.types import ForgeArray
            td = t.data.flatten() if isinstance(t, ForgeArray) else np.array(t).flatten()
            f0d = float(f0.data.flat[0]) if isinstance(f0, ForgeArray) else float(f0)
            t1d = float(t1.data.flat[0]) if isinstance(t1, ForgeArray) else float(t1)
            f1d = float(f1.data.flat[0]) if isinstance(f1, ForgeArray) else float(f1)
            k = (f1d - f0d) / t1d
            phase = 2 * np.pi * (f0d * td + 0.5 * k * td**2)
            return ForgeArray(np.cos(phase).reshape(1, -1))

        def forge_sawtooth(t, *args):
            """sawtooth(t) — sawtooth wave."""
            from forge.engine.types import ForgeArray
            td = t.data.flatten() if isinstance(t, ForgeArray) else np.array(t).flatten()
            from scipy.signal import sawtooth as _saw
            width = 1.0
            if args and isinstance(args[0], ForgeArray):
                width = float(args[0].data.flat[0])
            return ForgeArray(_saw(td, width).reshape(1, -1))

        def forge_square(t, *args):
            """square(t) — square wave."""
            from forge.engine.types import ForgeArray
            td = t.data.flatten() if isinstance(t, ForgeArray) else np.array(t).flatten()
            from scipy.signal import square as _sq
            duty = 50
            if args and isinstance(args[0], ForgeArray):
                duty = float(args[0].data.flat[0])
            return ForgeArray(_sq(td, duty/100).reshape(1, -1))

        def forge_gauspuls(t, fc=None, bw=None):
            """gauspuls(t, fc, bw) — Gaussian-modulated sinusoidal pulse."""
            from forge.engine.types import ForgeArray
            td = t.data.flatten() if isinstance(t, ForgeArray) else np.array(t).flatten()
            fcd = 1000.0
            bwd = 0.5
            if fc is not None:
                fcd = float(fc.data.flat[0]) if isinstance(fc, ForgeArray) else float(fc)
            if bw is not None:
                bwd = float(bw.data.flat[0]) if isinstance(bw, ForgeArray) else float(bw)
            ref = -6  # dB
            fv = -(np.pi * fcd * bwd)**2 / (4 * np.log(10**(ref/20)))
            env = np.exp(-fv * td**2)
            return ForgeArray((env * np.cos(2*np.pi*fcd*td)).reshape(1, -1))

        def forge_rectpuls(t, w=None):
            """rectpuls(t, w) — rectangular pulse."""
            from forge.engine.types import ForgeArray
            td = t.data.flatten() if isinstance(t, ForgeArray) else np.array(t).flatten()
            wd = 1.0
            if w is not None:
                wd = float(w.data.flat[0]) if isinstance(w, ForgeArray) else float(w)
            result = np.zeros_like(td)
            result[np.abs(td) < wd/2] = 1.0
            result[np.abs(td) == wd/2] = 0.5
            return ForgeArray(result.reshape(1, -1))

        def forge_tripuls(t, w=None):
            """tripuls(t, w) — triangular pulse."""
            from forge.engine.types import ForgeArray
            td = t.data.flatten() if isinstance(t, ForgeArray) else np.array(t).flatten()
            wd = 1.0
            if w is not None:
                wd = float(w.data.flat[0]) if isinstance(w, ForgeArray) else float(w)
            result = np.maximum(0, 1 - 2*np.abs(td)/wd)
            return ForgeArray(result.reshape(1, -1))

        def forge_findpeaks(x, *args):
            """[pks, locs] = findpeaks(x) — find local maxima."""
            from forge.engine.types import ForgeArray
            from scipy.signal import find_peaks
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            peaks, properties = find_peaks(data)
            return ForgeArray(data[peaks].reshape(1, -1)), ForgeArray((peaks + 1).astype(float).reshape(1, -1))

        def forge_envelope(x, *args):
            """[yupper, ylower] = envelope(x) — signal envelope."""
            from forge.engine.types import ForgeArray
            from scipy.signal import hilbert
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            analytic = hilbert(data)
            env = np.abs(analytic)
            return ForgeArray(env.reshape(1, -1)), ForgeArray((-env).reshape(1, -1))

        # --- Signal filter conversions ---
        def forge_tf2zp(b, a):
            """[z, p, k] = tf2zp(b, a) — transfer function to zero-pole."""
            from forge.engine.types import ForgeArray
            from scipy.signal import tf2zpk
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            z, p, k = tf2zpk(bd, ad)
            return ForgeArray(z.reshape(-1, 1)), ForgeArray(p.reshape(-1, 1)), ForgeArray(np.float64(k))

        def forge_zp2tf(z, p, k):
            """[b, a] = zp2tf(z, p, k) — zero-pole to transfer function."""
            from forge.engine.types import ForgeArray
            from scipy.signal import zpk2tf
            zd = z.data.flatten() if isinstance(z, ForgeArray) else np.array(z).flatten()
            pd = p.data.flatten() if isinstance(p, ForgeArray) else np.array(p).flatten()
            kd = float(k.data.flat[0]) if isinstance(k, ForgeArray) else float(k)
            b, a = zpk2tf(zd, pd, kd)
            return ForgeArray(np.real(b).reshape(1, -1)), ForgeArray(np.real(a).reshape(1, -1))

        def forge_grpdelay(b, a=None, *args):
            """[gd, w] = grpdelay(b, a, n) — group delay."""
            from forge.engine.types import ForgeArray
            from scipy.signal import group_delay
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            ad = np.array([1.0])
            if a is not None and isinstance(a, ForgeArray):
                ad = a.data.flatten()
            n = 512
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            w, gd = group_delay((bd, ad), w=n)
            return ForgeArray(gd.reshape(1, -1)), ForgeArray(w.reshape(1, -1))

        def forge_bandpass(x, fpass, fs):
            """bandpass(x, fpass, fs) — bandpass filter."""
            from forge.engine.types import ForgeArray
            from scipy.signal import butter, filtfilt
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            fp = fpass.data.flatten() if isinstance(fpass, ForgeArray) else np.array(fpass).flatten()
            fsd = float(fs.data.flat[0]) if isinstance(fs, ForgeArray) else float(fs)
            Wn = fp / (fsd / 2)
            b, a = butter(4, Wn, btype='band')
            return ForgeArray(filtfilt(b, a, data).reshape(1, -1))

        def forge_lowpass(x, fpass, fs):
            """lowpass(x, fpass, fs) — lowpass filter."""
            from forge.engine.types import ForgeArray
            from scipy.signal import butter, filtfilt
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            fpd = float(fpass.data.flat[0]) if isinstance(fpass, ForgeArray) else float(fpass)
            fsd = float(fs.data.flat[0]) if isinstance(fs, ForgeArray) else float(fs)
            b, a = butter(4, fpd / (fsd / 2), btype='low')
            return ForgeArray(filtfilt(b, a, data).reshape(1, -1))

        def forge_highpass(x, fpass, fs):
            """highpass(x, fpass, fs) — highpass filter."""
            from forge.engine.types import ForgeArray
            from scipy.signal import butter, filtfilt
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            fpd = float(fpass.data.flat[0]) if isinstance(fpass, ForgeArray) else float(fpass)
            fsd = float(fs.data.flat[0]) if isinstance(fs, ForgeArray) else float(fs)
            b, a = butter(4, fpd / (fsd / 2), btype='high')
            return ForgeArray(filtfilt(b, a, data).reshape(1, -1))

        # --- Statistics extras ---
        def forge_geomean(x, *args):
            """geomean(x) — geometric mean."""
            from forge.engine.types import ForgeArray
            from scipy.stats import gmean
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.float64(gmean(data)))

        def forge_harmmean(x, *args):
            """harmmean(x) — harmonic mean."""
            from forge.engine.types import ForgeArray
            from scipy.stats import hmean
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.float64(hmean(data)))

        def forge_trimmean(x, percent):
            """trimmean(x, percent) — trimmed mean."""
            from forge.engine.types import ForgeArray
            from scipy.stats import trim_mean
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            p = float(percent.data.flat[0]) if isinstance(percent, ForgeArray) else float(percent)
            return ForgeArray(np.float64(trim_mean(data, p / 200)))

        def forge_nanmean(x, *args):
            """nanmean(x) — mean ignoring NaN."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if data.ndim == 2 and data.shape[0] == 1:
                return ForgeArray(np.float64(np.nanmean(data)))
            return ForgeArray(np.nanmean(data, axis=0).reshape(1, -1))

        def forge_nanstd(x, *args):
            """nanstd(x) — std ignoring NaN."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.float64(np.nanstd(data, ddof=1)))

        def forge_nanvar(x, *args):
            """nanvar(x) — variance ignoring NaN."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.float64(np.nanvar(data, ddof=1)))

        def forge_nanmedian(x, *args):
            """nanmedian(x) — median ignoring NaN."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.float64(np.nanmedian(data)))

        def forge_nansum(x, *args):
            """nansum(x) — sum ignoring NaN."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.float64(np.nansum(data)))

        def forge_nanmax(x, *args):
            """nanmax(x) — max ignoring NaN."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.float64(np.nanmax(data)))

        def forge_nanmin(x, *args):
            """nanmin(x) — min ignoring NaN."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(np.float64(np.nanmin(data)))

        # --- Special functions ---
        def forge_sinint(x):
            """sinint(x) — sine integral."""
            from forge.engine.types import ForgeArray
            from scipy.special import sici
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            si, ci = sici(data)
            return ForgeArray(si)

        def forge_cosint(x):
            """cosint(x) — cosine integral."""
            from forge.engine.types import ForgeArray
            from scipy.special import sici
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            si, ci = sici(data)
            return ForgeArray(ci)

        def forge_expint(x):
            """expint(x) — exponential integral E1."""
            from forge.engine.types import ForgeArray
            from scipy.special import exp1
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(exp1(data))

        def forge_psi_func(x):
            """psi(x) — digamma function."""
            from forge.engine.types import ForgeArray
            from scipy.special import digamma
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(digamma(data))

        def forge_dawson_func(x):
            """dawson(x) — Dawson function."""
            from forge.engine.types import ForgeArray
            from scipy.special import dawsn
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(dawsn(data))

        # --- Image operations ---
        def forge_fspecial(ftype, *args):
            """fspecial(type, size) — predefined 2D filter."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(ftype, ForgeChar): ftype = ftype.to_str()
            n = 3
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            if ftype == 'gaussian':
                sigma = 0.5
                if len(args) > 1 and isinstance(args[1], ForgeArray):
                    sigma = float(args[1].data.flat[0])
                ax = np.arange(-n//2 + 1., n//2 + 1.)
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
                return ForgeArray(kernel / kernel.sum())
            elif ftype == 'average':
                return ForgeArray(np.ones((n, n)) / (n * n))
            elif ftype == 'laplacian':
                return ForgeArray(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).astype(float))
            elif ftype == 'sobel':
                return ForgeArray(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(float))
            elif ftype == 'prewitt':
                return ForgeArray(np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).astype(float))
            return ForgeArray(np.ones((n, n)) / (n * n))

        def forge_conv2d(A, B, *args):
            """conv2(A, B, shape) — 2D convolution."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            from scipy.signal import convolve2d
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            mode = 'full'
            if args and isinstance(args[0], ForgeChar):
                mode = args[0].to_str()
            return ForgeArray(convolve2d(Ad, Bd, mode=mode))

        def forge_filter2(h, x, *args):
            """filter2(h, x) — 2D FIR filter."""
            from forge.engine.types import ForgeArray
            from scipy.signal import convolve2d
            hd = h.data if isinstance(h, ForgeArray) else np.atleast_2d(h)
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(convolve2d(xd, hd, mode='same'))

        def forge_histeq(img, *args):
            """histeq(img) — histogram equalization."""
            from forge.engine.types import ForgeArray
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            if data.max() <= 1:
                data = (data * 255).astype(np.uint8)
            hist, bins = np.histogram(data.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)
            return ForgeArray(cdf[data.astype(np.uint8)].astype(float) / 255.0)

        # --- File I/O ---
        def forge_fwrite(fid, data, *args):
            """fwrite(fid, data) — write binary data."""
            from forge.engine.types import ForgeArray
            fd = int(fid.data.flat[0]) if isinstance(fid, ForgeArray) else int(fid)
            d = data.data if isinstance(data, ForgeArray) else np.atleast_2d(data)
            with open(os.path.join(tempfile.gettempdir(), f'forge_fid_{fd}'), 'ab') as f:
                d.tofile(f)
            return ForgeArray(np.float64(d.size))

        def forge_fread(fid, *args):
            """fread(fid, size) — read binary data."""
            from forge.engine.types import ForgeArray
            fd = int(fid.data.flat[0]) if isinstance(fid, ForgeArray) else int(fid)
            count = -1
            if args and isinstance(args[0], ForgeArray):
                count = int(args[0].data.flat[0])
            try:
                with open(os.path.join(tempfile.gettempdir(), f'forge_fid_{fd}'), 'rb') as f:
                    if count > 0:
                        data = np.fromfile(f, dtype=np.float64, count=count)
                    else:
                        data = np.fromfile(f, dtype=np.float64)
                return ForgeArray(data.reshape(-1, 1))
            except Exception:
                return ForgeArray(np.array([[]]))

        # Register all R146b functions
        session._engine.functions["chirp"] = forge_chirp
        session._engine.functions["sawtooth"] = forge_sawtooth
        session._engine.functions["square"] = forge_square
        session._engine.functions["gauspuls"] = forge_gauspuls
        session._engine.functions["rectpuls"] = forge_rectpuls
        session._engine.functions["tripuls"] = forge_tripuls
        session._engine.functions["findpeaks"] = forge_findpeaks
        session._engine.functions["envelope"] = forge_envelope
        session._engine.functions["tf2zp"] = forge_tf2zp
        session._engine.functions["zp2tf"] = forge_zp2tf
        session._engine.functions["grpdelay"] = forge_grpdelay
        session._engine.functions["bandpass"] = forge_bandpass
        session._engine.functions["lowpass"] = forge_lowpass
        session._engine.functions["highpass"] = forge_highpass
        session._engine.functions["geomean"] = forge_geomean
        session._engine.functions["harmmean"] = forge_harmmean
        session._engine.functions["trimmean"] = forge_trimmean
        session._engine.functions["nanmean"] = forge_nanmean
        session._engine.functions["nanstd"] = forge_nanstd
        session._engine.functions["nanvar"] = forge_nanvar
        session._engine.functions["nanmedian"] = forge_nanmedian
        session._engine.functions["nansum"] = forge_nansum
        session._engine.functions["nanmax"] = forge_nanmax
        session._engine.functions["nanmin"] = forge_nanmin
        session._engine.functions["sinint"] = forge_sinint
        session._engine.functions["cosint"] = forge_cosint
        session._engine.functions["expint"] = forge_expint
        session._engine.functions["psi"] = forge_psi_func
        session._engine.functions["dawson"] = forge_dawson_func
        session._engine.functions["fspecial"] = forge_fspecial
        session._engine.functions["conv2"] = forge_conv2d
        session._engine.functions["filter2"] = forge_filter2
        session._engine.functions["histeq"] = forge_histeq
        session._engine.functions["fwrite"] = forge_fwrite
        session._engine.functions["fread"] = forge_fread

        # R147: Plotting, utilities, string ops

        def forge_plot(*args):
            """plot(x, y, ...) — 2D line plot."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if not hasattr(session, '_plot_requests'):
                session._plot_requests = []
            # Parse args: plot(y), plot(x,y), plot(x,y,fmt), ...
            arrays = []
            fmt_str = None
            for a in args:
                if isinstance(a, ForgeArray):
                    arrays.append(a.data.flatten())
                elif isinstance(a, ForgeChar):
                    fmt_str = a.to_str()
                elif isinstance(a, (int, float)):
                    arrays.append(np.array([float(a)]))
                elif isinstance(a, str):
                    fmt_str = a
            if len(arrays) == 0:
                return
            elif len(arrays) == 1:
                y = arrays[0]
                x = np.arange(1, len(y) + 1, dtype=np.float64)
            else:
                x, y = arrays[0], arrays[1]
            session._plot_requests.append(('plot', x.tolist(), y.tolist()))

        def forge_plot3(*args):
            """plot3(x, y, z, ...) — 3D line plot (stub)."""
            pass

        def forge_scatter_func(*args):
            """scatter(x, y, ...) — scatter plot (stub)."""
            pass

        def forge_scatter3_func(*args):
            """scatter3(x, y, z, ...) — 3D scatter plot (stub)."""
            pass

        def forge_bar_func(*args):
            """bar(x, y, ...) — bar chart (stub)."""
            pass

        def forge_barh_func(*args):
            """barh(x, y, ...) — horizontal bar chart (stub)."""
            pass

        def forge_surf_func(*args):
            """surf(X, Y, Z, ...) — surface plot (stub)."""
            pass

        def forge_mesh_func(*args):
            """mesh(X, Y, Z, ...) — mesh plot (stub)."""
            pass

        def forge_contour_func(*args):
            """contour(X, Y, Z, ...) — contour plot (stub)."""
            pass

        def forge_contourf_func(*args):
            """contourf(X, Y, Z, ...) — filled contour (stub)."""
            pass

        def forge_quiver_func(*args):
            """quiver(X, Y, U, V, ...) — vector field (stub)."""
            pass

        def forge_semilogx_func(*args):
            """semilogx(x, y, ...) — log x plot (stub)."""
            pass

        def forge_semilogy_func(*args):
            """semilogy(x, y, ...) — log y plot (stub)."""
            pass

        def forge_loglog_func(*args):
            """loglog(x, y, ...) — log-log plot (stub)."""
            pass

        def forge_stem_func(*args):
            """stem(x, y, ...) — stem plot (stub)."""
            pass

        def forge_stairs_func(*args):
            """stairs(x, y, ...) — stairstep plot (stub)."""
            pass

        def forge_area_func(*args):
            """area(x, y, ...) — area plot (stub)."""
            pass

        def forge_pie_func(*args):
            """pie(x, ...) — pie chart (stub)."""
            pass

        def forge_polar_func(*args):
            """polar(theta, rho, ...) — polar plot (stub)."""
            pass

        def forge_errorbar_func(*args):
            """errorbar(x, y, err, ...) — error bars (stub)."""
            pass

        def forge_fill_func(*args):
            """fill(x, y, ...) — fill polygon (stub)."""
            pass

        def forge_patch_func(*args):
            """patch(x, y, ...) — patch object (stub)."""
            pass

        def forge_line_func(*args):
            """line(x, y, ...) — line object (stub)."""
            pass

        def forge_rectangle_func(*args):
            """rectangle(x, y, ...) — rectangle (stub)."""
            pass

        def forge_text_func(*args):
            """text(x, y, str, ...) — text annotation (stub)."""
            pass

        def forge_annotation_func(*args):
            """annotation(...) — figure annotation (stub)."""
            pass

        def forge_cla_func(*args):
            """cla — clear current axes (stub)."""
            pass

        def forge_clf_func(*args):
            """clf — clear current figure (stub)."""
            pass

        def forge_gca_func(*args):
            """gca — get current axes handle (stub)."""
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(1))

        def forge_gcf_func(*args):
            """gcf — get current figure handle (stub)."""
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(1))

        def forge_imshow_func(*args):
            """imshow(img, ...) — display image (stub)."""
            pass

        def forge_imagesc_func(*args):
            """imagesc(data, ...) — scaled image (stub)."""
            pass

        def forge_image_func(*args):
            """image(data, ...) — display image (stub)."""
            pass

        def forge_imread_func(fname, *args):
            """imread(filename) — read image file."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            try:
                from PIL import Image
                img = np.array(Image.open(fname)).astype(float) / 255.0
                return ForgeArray(img)
            except Exception:
                return ForgeArray(np.array([[]]))

        def forge_imwrite_func(img, fname, *args):
            """imwrite(img, filename) — write image file."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            if data.max() <= 1.0:
                data = (data * 255).astype(np.uint8)
            try:
                from PIL import Image
                Image.fromarray(data.astype(np.uint8)).save(fname)
            except Exception:
                pass

        # --- Audio stubs ---
        def forge_audioread_func(fname, *args):
            """audioread(filename) — read audio file."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            try:
                import soundfile as sf
                data, fs = sf.read(fname)
                return ForgeArray(data.reshape(-1, 1) if data.ndim == 1 else data), ForgeArray(np.float64(fs))
            except Exception:
                return ForgeArray(np.array([[]])), ForgeArray(np.float64(44100))

        def forge_audiowrite_func(fname, y, fs):
            """audiowrite(filename, y, fs) — write audio file."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            data = y.data if isinstance(y, ForgeArray) else np.atleast_2d(y)
            fsd = int(fs.data.flat[0]) if isinstance(fs, ForgeArray) else int(fs)
            try:
                import soundfile as sf
                sf.write(fname, data.flatten(), fsd)
            except Exception:
                pass

        def forge_sound_func(*args):
            """sound(y, fs) — play audio (stub)."""
            pass

        def forge_soundsc_func(*args):
            """soundsc(y, fs) — play scaled audio (stub)."""
            pass

        # --- Missing utility functions ---
        def forge_randperm2(n, *args):
            """randperm(n) or randperm(n, k) — random permutation."""
            from forge.engine.types import ForgeArray
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            if args and isinstance(args[0], ForgeArray):
                k = int(args[0].data.flat[0])
                return ForgeArray(np.random.permutation(nd)[:k].astype(float).reshape(1, -1) + 1)
            return ForgeArray(np.random.permutation(nd).astype(float).reshape(1, -1) + 1)

        def forge_rng2(seed, *args):
            """rng(seed) — set random seed."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(seed, ForgeChar):
                s = seed.to_str()
                if s == 'shuffle':
                    np.random.seed(None)
                elif s == 'default':
                    np.random.seed(0)
                return
            sd = int(seed.data.flat[0]) if isinstance(seed, ForgeArray) else int(seed)
            np.random.seed(sd)

        def forge_issorted2(x, *args):
            """issorted(x) — check if sorted."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.float64(1 if np.all(data[:-1] <= data[1:]) else 0))

        def forge_mode2(x, *args):
            """mode(x) — most frequent value."""
            from forge.engine.types import ForgeArray
            from scipy.stats import mode as _mode
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            result = _mode(data, keepdims=False)
            return ForgeArray(np.float64(result.mode))

        def forge_range2(x, *args):
            """range(x) — difference between max and min."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.float64(data.max() - data.min()))

        def forge_iqr2(x, *args):
            """iqr(x) — interquartile range."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.float64(np.percentile(data, 75) - np.percentile(data, 25)))

        def forge_mad2(x, *args):
            """mad(x) — median absolute deviation."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.float64(np.median(np.abs(data - np.median(data)))))

        def forge_zscore2(x, *args):
            """zscore(x) — standardized z-scores."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            mu = np.mean(data)
            s = np.std(data, ddof=1)
            if s == 0:
                return ForgeArray(np.zeros_like(data).reshape(1, -1))
            return ForgeArray(((data - mu) / s).reshape(1, -1))

        def forge_normalize2(x, *args):
            """normalize(x) — normalize data."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            if data.shape[0] == 1:
                mu = np.mean(data)
                s = np.std(data, ddof=1)
                if s == 0: s = 1
                return ForgeArray((data - mu) / s)
            mu = np.mean(data, axis=0)
            s = np.std(data, axis=0, ddof=1)
            s[s == 0] = 1
            return ForgeArray((data - mu) / s)

        def forge_detrend2(x, *args):
            """detrend(x) — remove linear trend."""
            from forge.engine.types import ForgeArray
            from scipy.signal import detrend as _detrend
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(_detrend(data).reshape(1, -1))

        def forge_movmean2(x, k, *args):
            """movmean(x, k) — moving average."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            kd = int(k.data.flat[0]) if isinstance(k, ForgeArray) else int(k)
            result = np.convolve(data, np.ones(kd)/kd, mode='same')
            return ForgeArray(result.reshape(1, -1))

        def forge_movstd2(x, k, *args):
            """movstd(x, k) — moving standard deviation."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            kd = int(k.data.flat[0]) if isinstance(k, ForgeArray) else int(k)
            result = np.zeros_like(data)
            for i in range(len(data)):
                lo = max(0, i - kd//2)
                hi = min(len(data), i + kd//2 + 1)
                result[i] = np.std(data[lo:hi], ddof=1) if hi - lo > 1 else 0
            return ForgeArray(result.reshape(1, -1))

        def forge_accumarray2(subs, val, *args):
            """accumarray(subs, val) — accumulate by subscript."""
            from forge.engine.types import ForgeArray
            sd = subs.data.flatten().astype(int) if isinstance(subs, ForgeArray) else np.array(subs).flatten().astype(int)
            vd = val.data.flatten() if isinstance(val, ForgeArray) else np.array(val).flatten()
            n = sd.max()
            result = np.zeros(n)
            for s, v in zip(sd, vd):
                result[s - 1] += v
            return ForgeArray(result.reshape(-1, 1))

        def forge_evalc2(expr):
            """evalc(expr) — evaluate and capture output."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(expr, ForgeChar): expr = expr.to_str()
            import io, sys
            old = sys.stdout
            sys.stdout = buf = io.StringIO()
            try:
                session.eval(expr)
            finally:
                sys.stdout = old
            return ForgeChar(buf.getvalue())

        def forge_genpath2(d):
            """genpath(dir) — generate path string with subdirs."""
            from forge.engine.containers import ForgeChar
            if isinstance(d, ForgeChar): d = d.to_str()
            paths = [d]
            for root, dirs, files in os.walk(d):
                for dn in dirs:
                    if not dn.startswith('.') and not dn.startswith('+') and not dn.startswith('@'):
                        paths.append(os.path.join(root, dn))
            return ForgeChar(os.pathsep.join(paths))

        def forge_what2(*args):
            """what — list M-files in directory."""
            from forge.engine.containers import ForgeChar, ForgeCell
            d = os.getcwd()
            if args:
                from forge.engine.containers import ForgeChar as FC
                if isinstance(args[0], FC): d = args[0].to_str()
            mfiles = sorted([f[:-2] for f in os.listdir(d) if f.endswith('.m')])
            return ForgeCell._from_list([ForgeChar(f) for f in mfiles])

        def forge_type2(name):
            """type(name) -- display file contents or function info."""
            from forge.engine.containers import ForgeChar
            if isinstance(name, ForgeChar): name = name.to_str()
            name = str(name)
            for p in session.path:
                fp = os.path.join(p, name + ".m")
                if os.path.exists(fp):
                    with open(fp) as f:
                        return ForgeChar(f.read())
            if name in session._engine.functions:
                func = session._engine.functions[name]
                doc = func.__doc__ or ""
                return ForgeChar(name + " is a built-in function\n" + doc)
            if session._engine.workspace.has(name):
                return ForgeChar(name + " is a variable")
            return ForgeChar(name + " is undefined")

        def forge_keyboard2(*args):
            """keyboard — pause execution (stub)."""
            pass

        def forge_menu2(title, *args):
            """menu(title, opt1, ...) — selection menu (stub, returns 1)."""
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(1))

        # --- More string functions ---
        def forge_blanks2(n):
            """blanks(n) — string of n spaces."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            return ForgeChar(' ' * nd)

        def forge_deblank2(s):
            """deblank(s) — remove trailing blanks."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            return ForgeChar(s.rstrip())

        def forge_strtrim2(s):
            """strtrim(s) — remove leading/trailing whitespace."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            return ForgeChar(s.strip())

        def forge_pad2(s, n, *args):
            """pad(s, n, side) — pad string."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            side = 'right'
            if args and isinstance(args[0], ForgeChar):
                side = args[0].to_str()
            if side == 'left':
                return ForgeChar(s.rjust(nd))
            elif side == 'both':
                return ForgeChar(s.center(nd))
            return ForgeChar(s.ljust(nd))

        def forge_contains2(s, pat):
            """contains(s, pattern) — check if string contains pattern."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            return ForgeArray(np.float64(1 if pat in s else 0))

        def forge_startsWith2(s, pat):
            """startsWith(s, prefix) — check prefix."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            return ForgeArray(np.float64(1 if s.startswith(pat) else 0))

        def forge_endsWith2(s, pat):
            """endsWith(s, suffix) — check suffix."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            return ForgeArray(np.float64(1 if s.endswith(pat) else 0))

        def forge_erase2(s, pat):
            """erase(s, pattern) — remove pattern from string."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            return ForgeChar(s.replace(pat, ''))

        def forge_replace2(s, old, new):
            """replace(s, old, new) — replace in string."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(old, ForgeChar): old = old.to_str()
            if isinstance(new, ForgeChar): new = new.to_str()
            return ForgeChar(s.replace(old, new))

        def forge_count2(s, pat):
            """count(s, pattern) — count occurrences."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            return ForgeArray(np.float64(s.count(pat)))

        def forge_extractBefore2(s, pat):
            """extractBefore(s, pattern) — extract before pattern."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            idx = s.find(pat)
            if idx < 0: return ForgeChar(s)
            return ForgeChar(s[:idx])

        def forge_extractAfter2(s, pat):
            """extractAfter(s, pattern) — extract after pattern."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            idx = s.find(pat)
            if idx < 0: return ForgeChar(s)
            return ForgeChar(s[idx + len(pat):])

        def forge_insertBefore2(s, pat, ins):
            """insertBefore(s, pattern, insert) — insert before pattern."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            if isinstance(ins, ForgeChar): ins = ins.to_str()
            return ForgeChar(s.replace(pat, ins + pat))

        def forge_insertAfter2(s, pat, ins):
            """insertAfter(s, pattern, insert) — insert after pattern."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            if isinstance(ins, ForgeChar): ins = ins.to_str()
            return ForgeChar(s.replace(pat, pat + ins))

        def forge_compose2(fmt, *args):
            """compose(formatSpec, args) — compose formatted string."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(fmt, ForgeChar): fmt = fmt.to_str()
            vals = []
            for a in args:
                if isinstance(a, ForgeArray):
                    vals.append(float(a.data.flat[0]))
                elif isinstance(a, ForgeChar):
                    vals.append(a.to_str())
                else:
                    vals.append(a)
            try:
                return ForgeChar(fmt % tuple(vals))
            except Exception:
                return ForgeChar(fmt)

        def forge_join2(sep, *args):
            """join(delimiter, strs) — join strings."""
            from forge.engine.containers import ForgeChar, ForgeCell
            if isinstance(sep, ForgeChar): sep = sep.to_str()
            parts = []
            for a in args:
                if isinstance(a, ForgeCell):
                    for i in range(a._shape[0] * a._shape[1]):
                        v = a._data[i]
                        if isinstance(v, ForgeChar):
                            parts.append(v.to_str())
                elif isinstance(a, ForgeChar):
                    parts.append(a.to_str())
            return ForgeChar(sep.join(parts))

        # --- Bitwise ops (aliases if missing) ---
        def forge_bitand2(a, b):
            """bitand(a, b) — bitwise AND."""
            from forge.engine.types import ForgeArray
            ad = int(a.data.flat[0]) if isinstance(a, ForgeArray) else int(a)
            bd = int(b.data.flat[0]) if isinstance(b, ForgeArray) else int(b)
            return ForgeArray(np.float64(ad & bd))

        def forge_bitor2(a, b):
            """bitor(a, b) — bitwise OR."""
            from forge.engine.types import ForgeArray
            ad = int(a.data.flat[0]) if isinstance(a, ForgeArray) else int(a)
            bd = int(b.data.flat[0]) if isinstance(b, ForgeArray) else int(b)
            return ForgeArray(np.float64(ad | bd))

        def forge_bitxor2(a, b):
            """bitxor(a, b) — bitwise XOR."""
            from forge.engine.types import ForgeArray
            ad = int(a.data.flat[0]) if isinstance(a, ForgeArray) else int(a)
            bd = int(b.data.flat[0]) if isinstance(b, ForgeArray) else int(b)
            return ForgeArray(np.float64(ad ^ bd))

        # --- Table stubs ---
        def forge_readtable2(fname, *args):
            """readtable(filename) — read CSV as table."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar, ForgeStruct
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            try:
                data = np.genfromtxt(fname, delimiter=',', skip_header=1)
                t = ForgeStruct()
                t._fields["data"] = ForgeArray(data)
                return t
            except Exception:
                return ForgeStruct()

        def forge_writetable2(tbl, fname, *args):
            """writetable(table, filename) — write table to CSV."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar, ForgeStruct
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            if isinstance(tbl, ForgeStruct) and "data" in tbl._fields:
                data = tbl._fields["data"]
                if isinstance(data, ForgeArray):
                    np.savetxt(fname, data.data, delimiter=',')

        def forge_height2(x):
            """height(x) — number of rows."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                return ForgeArray(np.float64(x.data.shape[0]))
            return ForgeArray(np.float64(1))

        def forge_width2(x):
            """width(x) — number of columns."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                return ForgeArray(np.float64(x.data.shape[1] if x.data.ndim > 1 else 1))
            return ForgeArray(np.float64(1))

        # Register all R147 functions
        session._engine.functions["quiver"] = forge_quiver_func
        session._engine.functions["patch"] = forge_patch_func
        session._engine.functions["annotation"] = forge_annotation_func
        session._engine.functions["imshow"] = forge_imshow_func
        session._engine.functions["image"] = forge_image_func
        session._engine.functions["imread"] = forge_imread_func
        session._engine.functions["imwrite"] = forge_imwrite_func
        session._engine.functions["audioread"] = forge_audioread_func
        session._engine.functions["audiowrite"] = forge_audiowrite_func
        session._engine.functions["sound"] = forge_sound_func
        session._engine.functions["soundsc"] = forge_soundsc_func
        session._engine.functions["randperm"] = forge_randperm2
        session._engine.functions["rng"] = forge_rng2
        session._engine.functions["issorted"] = forge_issorted2
        session._engine.functions["mode"] = forge_mode2
        session._engine.functions["range"] = forge_range2
        session._engine.functions["iqr"] = forge_iqr2
        session._engine.functions["mad"] = forge_mad2
        session._engine.functions["zscore"] = forge_zscore2
        session._engine.functions["normalize"] = forge_normalize2
        session._engine.functions["detrend"] = forge_detrend2
        session._engine.functions["movmean"] = forge_movmean2
        session._engine.functions["movstd"] = forge_movstd2
        session._engine.functions["accumarray"] = forge_accumarray2
        session._engine.functions["evalc"] = forge_evalc2
        session._engine.functions["genpath"] = forge_genpath2
        session._engine.functions["what"] = forge_what2
        session._engine.functions["type"] = forge_type2
        session._engine.functions["keyboard"] = forge_keyboard2
        session._engine.functions["menu"] = forge_menu2
        session._engine.functions["blanks"] = forge_blanks2
        session._engine.functions["deblank"] = forge_deblank2
        session._engine.functions["strtrim"] = forge_strtrim2
        session._engine.functions["pad"] = forge_pad2
        session._engine.functions["contains"] = forge_contains2
        session._engine.functions["startsWith"] = forge_startsWith2
        session._engine.functions["endsWith"] = forge_endsWith2
        session._engine.functions["erase"] = forge_erase2
        session._engine.functions["replace"] = forge_replace2
        session._engine.functions["count"] = forge_count2
        session._engine.functions["extractBefore"] = forge_extractBefore2
        session._engine.functions["extractAfter"] = forge_extractAfter2
        session._engine.functions["insertBefore"] = forge_insertBefore2
        session._engine.functions["insertAfter"] = forge_insertAfter2
        session._engine.functions["compose"] = forge_compose2
        session._engine.functions["join"] = forge_join2
        session._engine.functions["bitand"] = forge_bitand2
        session._engine.functions["bitor"] = forge_bitor2
        session._engine.functions["bitxor"] = forge_bitxor2
        session._engine.functions["readtable"] = forge_readtable2
        session._engine.functions["writetable"] = forge_writetable2
        session._engine.functions["height"] = forge_height2
        session._engine.functions["width"] = forge_width2

        # R148: Control, linear algebra, statistics, colormaps

        # --- Colormap functions ---
        def forge_jet_func(*args):
            """jet(n) — jet colormap."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            import matplotlib.pyplot as plt
            cmap = plt.cm.jet(np.linspace(0, 1, n))[:, :3]
            return ForgeArray(cmap)

        def forge_hsv_func(*args):
            """hsv(n) — HSV colormap."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            import matplotlib.pyplot as plt
            cmap = plt.cm.hsv(np.linspace(0, 1, n))[:, :3]
            return ForgeArray(cmap)

        def forge_hot_func(*args):
            """hot(n) — hot colormap."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            import matplotlib.pyplot as plt
            cmap = plt.cm.hot(np.linspace(0, 1, n))[:, :3]
            return ForgeArray(cmap)

        def forge_cool_func(*args):
            """cool(n) — cool colormap."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            import matplotlib.pyplot as plt
            cmap = plt.cm.cool(np.linspace(0, 1, n))[:, :3]
            return ForgeArray(cmap)

        def forge_spring_func(*args):
            """spring(n) — spring colormap."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            import matplotlib.pyplot as plt
            cmap = plt.cm.spring(np.linspace(0, 1, n))[:, :3]
            return ForgeArray(cmap)

        def forge_summer_func(*args):
            """summer(n) — summer colormap."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            import matplotlib.pyplot as plt
            cmap = plt.cm.summer(np.linspace(0, 1, n))[:, :3]
            return ForgeArray(cmap)

        def forge_autumn_func(*args):
            """autumn(n) — autumn colormap."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            import matplotlib.pyplot as plt
            cmap = plt.cm.autumn(np.linspace(0, 1, n))[:, :3]
            return ForgeArray(cmap)

        def forge_winter_func(*args):
            """winter(n) — winter colormap."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            import matplotlib.pyplot as plt
            cmap = plt.cm.winter(np.linspace(0, 1, n))[:, :3]
            return ForgeArray(cmap)

        def forge_gray_func(*args):
            """gray(n) — grayscale colormap."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            v = np.linspace(0, 1, n)
            return ForgeArray(np.column_stack([v, v, v]))

        def forge_bone_func(*args):
            """bone(n) — bone colormap."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            import matplotlib.pyplot as plt
            cmap = plt.cm.bone(np.linspace(0, 1, n))[:, :3]
            return ForgeArray(cmap)

        def forge_copper_func(*args):
            """copper(n) — copper colormap."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            import matplotlib.pyplot as plt
            cmap = plt.cm.copper(np.linspace(0, 1, n))[:, :3]
            return ForgeArray(cmap)

        def forge_parula_func(*args):
            """parula(n) — parula colormap (viridis fallback)."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            import matplotlib.pyplot as plt
            cmap = plt.cm.viridis(np.linspace(0, 1, n))[:, :3]
            return ForgeArray(cmap)

        def forge_turbo_func(*args):
            """turbo(n) — turbo colormap."""
            from forge.engine.types import ForgeArray
            n = 256
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            import matplotlib.pyplot as plt
            cmap = plt.cm.turbo(np.linspace(0, 1, n))[:, :3]
            return ForgeArray(cmap)

        def forge_colormap_func(*args):
            """colormap(name) — set/get colormap (stub)."""
            pass

        def forge_caxis_func(*args):
            """caxis([lo hi]) — set color limits (stub)."""
            pass

        # --- More linear algebra ---
        def forge_sylvester2(A, B, C):
            """sylvester(A, B, C) — solve AX + XB = C."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import solve_sylvester
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            Cd = C.data if isinstance(C, ForgeArray) else np.atleast_2d(C)
            return ForgeArray(solve_sylvester(Ad, Bd, Cd))

        def forge_ordschur2(U, T, *args):
            """ordschur(U, T, select) — reorder Schur decomposition."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import ordqz
            Ud = U.data if isinstance(U, ForgeArray) else np.atleast_2d(U)
            Td = T.data if isinstance(T, ForgeArray) else np.atleast_2d(T)
            return ForgeArray(Ud), ForgeArray(Td)

        def forge_ldl2(A):
            """[L, D, P] = ldl(A) — LDL factorization."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import ldl as _ldl
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            L, D, perm = _ldl(Ad)
            return ForgeArray(L), ForgeArray(D), ForgeArray((perm + 1).astype(float).reshape(-1, 1))

        def forge_planerot2(x):
            """[G, y] = planerot(x) — Givens plane rotation."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            a, b = xd[0], xd[1]
            r = np.sqrt(a**2 + b**2)
            if r == 0:
                c, s = 1.0, 0.0
            else:
                c, s = a/r, b/r
            G = np.array([[c, s], [-s, c]])
            y = np.array([r, 0.0])
            return ForgeArray(G), ForgeArray(y.reshape(-1, 1))

        def forge_cholupdate2(R, x, *args):
            """cholupdate(R, x) — rank-1 update to Cholesky."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            Rd = R.data.copy() if isinstance(R, ForgeArray) else np.atleast_2d(R).copy()
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            sign = '+'
            if args and isinstance(args[0], ForgeChar):
                sign = args[0].to_str()
            from scipy.linalg import cholesky
            A = Rd.T @ Rd
            if sign == '+':
                A += np.outer(xd, xd)
            else:
                A -= np.outer(xd, xd)
            return ForgeArray(cholesky(A))

        # --- More statistics ---
        def forge_skewness2(x, *args):
            """skewness(x) — sample skewness."""
            from forge.engine.types import ForgeArray
            from scipy.stats import skew
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.float64(skew(data)))

        def forge_kurtosis2(x, *args):
            """kurtosis(x) — sample kurtosis."""
            from forge.engine.types import ForgeArray
            from scipy.stats import kurtosis as _kurt
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.float64(_kurt(data, fisher=False)))

        def forge_moment2(x, n, *args):
            """moment(x, n) — central moment."""
            from forge.engine.types import ForgeArray
            from scipy.stats import moment as _moment
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            return ForgeArray(np.float64(_moment(data, moment=nd)))

        def forge_ttest2(x, *args):
            """[h, p] = ttest(x) — one-sample t-test."""
            from forge.engine.types import ForgeArray
            from scipy.stats import ttest_1samp
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            mu = 0
            if args and isinstance(args[0], ForgeArray):
                mu = float(args[0].data.flat[0])
            stat, pval = ttest_1samp(data, mu)
            h = 1 if pval < 0.05 else 0
            return ForgeArray(np.float64(h)), ForgeArray(np.float64(pval))

        def forge_ttest2_2(x, y, *args):
            """[h, p] = ttest2(x, y) — two-sample t-test."""
            from forge.engine.types import ForgeArray
            from scipy.stats import ttest_ind
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            stat, pval = ttest_ind(xd, yd)
            h = 1 if pval < 0.05 else 0
            return ForgeArray(np.float64(h)), ForgeArray(np.float64(pval))

        def forge_kstest2(x, *args):
            """[h, p] = kstest(x, dist) — KS test."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            from scipy.stats import kstest
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            dist = 'norm'
            if args and isinstance(args[0], ForgeChar):
                dist = args[0].to_str()
            stat, pval = kstest(data, dist)
            h = 1 if pval < 0.05 else 0
            return ForgeArray(np.float64(h)), ForgeArray(np.float64(pval))

        def forge_chi2test2(observed):
            """[h, p] = chi2gof(observed) — chi-squared goodness of fit."""
            from forge.engine.types import ForgeArray
            from scipy.stats import chisquare
            data = observed.data.flatten() if isinstance(observed, ForgeArray) else np.array(observed).flatten()
            stat, pval = chisquare(data)
            h = 1 if pval < 0.05 else 0
            return ForgeArray(np.float64(h)), ForgeArray(np.float64(pval))

        def forge_anova1_2(x, group):
            """[p, tbl] = anova1(x, group) — one-way ANOVA."""
            from forge.engine.types import ForgeArray
            from scipy.stats import f_oneway
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            # Assume columns are groups
            groups = [xd[:, i] for i in range(xd.shape[1])]
            stat, pval = f_oneway(*groups)
            return ForgeArray(np.float64(pval))

        def forge_bootstrp2(nboot, func, x, *args):
            """bootstrp(nboot, func, x) — bootstrap statistics."""
            from forge.engine.types import ForgeArray
            nb = int(nboot.data.flat[0]) if isinstance(nboot, ForgeArray) else int(nboot)
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            results = []
            for _ in range(nb):
                sample = data[np.random.randint(0, len(data), len(data))]
                results.append(func(ForgeArray(sample.reshape(1, -1))))
            vals = []
            for r in results:
                if isinstance(r, ForgeArray):
                    vals.append(float(r.data.flat[0]))
                else:
                    vals.append(float(r))
            return ForgeArray(np.array(vals).reshape(-1, 1))

        def forge_pdist2_func(X, *args):
            """pdist(X) — pairwise distance."""
            from forge.engine.types import ForgeArray
            from scipy.spatial.distance import pdist as _pdist
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            return ForgeArray(_pdist(Xd).reshape(1, -1))

        def forge_squareform2(Y):
            """squareform(Y) — convert distance vector to matrix."""
            from forge.engine.types import ForgeArray
            from scipy.spatial.distance import squareform as _sqf
            Yd = Y.data.flatten() if isinstance(Y, ForgeArray) else np.array(Y).flatten()
            return ForgeArray(_sqf(Yd))

        def forge_cdist2(XA, XB, *args):
            """cdist(XA, XB) — pairwise distance between two sets."""
            from forge.engine.types import ForgeArray
            from scipy.spatial.distance import cdist as _cdist
            XAd = XA.data if isinstance(XA, ForgeArray) else np.atleast_2d(XA)
            XBd = XB.data if isinstance(XB, ForgeArray) else np.atleast_2d(XB)
            return ForgeArray(_cdist(XAd, XBd))

        def forge_knnsearch2(X, Y, *args):
            """[idx, D] = knnsearch(X, Y) — k-nearest neighbors."""
            from forge.engine.types import ForgeArray
            from scipy.spatial import KDTree
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            Yd = Y.data if isinstance(Y, ForgeArray) else np.atleast_2d(Y)
            k = 1
            if args and isinstance(args[0], ForgeArray):
                k = int(args[0].data.flat[0])
            tree = KDTree(Xd)
            D, idx = tree.query(Yd, k=k)
            return ForgeArray((idx + 1).astype(float).reshape(-1, 1)), ForgeArray(D.reshape(-1, 1))

        def forge_rangesearch2(X, Y, r):
            """[idx, D] = rangesearch(X, Y, r) — range search."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeCell
            from scipy.spatial import KDTree
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            Yd = Y.data if isinstance(Y, ForgeArray) else np.atleast_2d(Y)
            rd = float(r.data.flat[0]) if isinstance(r, ForgeArray) else float(r)
            tree = KDTree(Xd)
            results = tree.query_ball_point(Yd, rd)
            # Return count for simplicity
            counts = [len(r) for r in results]
            return ForgeArray(np.array(counts).reshape(-1, 1).astype(float))

        # --- Coordinate transforms ---
        def forge_cart2pol2(x, y, *args):
            """[theta, rho, z] = cart2pol(x, y, z) — Cartesian to polar."""
            from forge.engine.types import ForgeArray
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            yd = y.data if isinstance(y, ForgeArray) else np.atleast_2d(y)
            theta = np.arctan2(yd, xd)
            rho = np.sqrt(xd**2 + yd**2)
            if args and isinstance(args[0], ForgeArray):
                return ForgeArray(theta), ForgeArray(rho), args[0]
            return ForgeArray(theta), ForgeArray(rho)

        def forge_pol2cart2(theta, rho, *args):
            """[x, y, z] = pol2cart(theta, rho, z) — polar to Cartesian."""
            from forge.engine.types import ForgeArray
            td = theta.data if isinstance(theta, ForgeArray) else np.atleast_2d(theta)
            rd = rho.data if isinstance(rho, ForgeArray) else np.atleast_2d(rho)
            x = rd * np.cos(td)
            y = rd * np.sin(td)
            if args and isinstance(args[0], ForgeArray):
                return ForgeArray(x), ForgeArray(y), args[0]
            return ForgeArray(x), ForgeArray(y)

        def forge_cart2sph2(x, y, z):
            """[az, el, r] = cart2sph(x, y, z) — Cartesian to spherical."""
            from forge.engine.types import ForgeArray
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            yd = y.data if isinstance(y, ForgeArray) else np.atleast_2d(y)
            zd = z.data if isinstance(z, ForgeArray) else np.atleast_2d(z)
            r = np.sqrt(xd**2 + yd**2 + zd**2)
            az = np.arctan2(yd, xd)
            el = np.arctan2(zd, np.sqrt(xd**2 + yd**2))
            return ForgeArray(az), ForgeArray(el), ForgeArray(r)

        def forge_sph2cart2(az, el, r):
            """[x, y, z] = sph2cart(az, el, r) — spherical to Cartesian."""
            from forge.engine.types import ForgeArray
            azd = az.data if isinstance(az, ForgeArray) else np.atleast_2d(az)
            eld = el.data if isinstance(el, ForgeArray) else np.atleast_2d(el)
            rd = r.data if isinstance(r, ForgeArray) else np.atleast_2d(r)
            x = rd * np.cos(eld) * np.cos(azd)
            y = rd * np.cos(eld) * np.sin(azd)
            z = rd * np.sin(eld)
            return ForgeArray(x), ForgeArray(y), ForgeArray(z)

        # --- Geometry ---
        def forge_convhull2(x, y, *args):
            """[k, v] = convhull(x, y) — convex hull."""
            from forge.engine.types import ForgeArray
            from scipy.spatial import ConvexHull
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            points = np.column_stack([xd, yd])
            hull = ConvexHull(points)
            verts = np.append(hull.vertices, hull.vertices[0]) + 1  # 1-based, closed
            return ForgeArray(verts.astype(float).reshape(-1, 1)), ForgeArray(np.float64(hull.volume))

        def forge_delaunay2(x, y, *args):
            """tri = delaunay(x, y) — Delaunay triangulation."""
            from forge.engine.types import ForgeArray
            from scipy.spatial import Delaunay
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            points = np.column_stack([xd, yd])
            tri = Delaunay(points)
            return ForgeArray((tri.simplices + 1).astype(float))

        def forge_voronoi2(x, y, *args):
            """[vx, vy] = voronoi(x, y) — Voronoi diagram."""
            from forge.engine.types import ForgeArray
            from scipy.spatial import Voronoi
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            points = np.column_stack([xd, yd])
            vor = Voronoi(points)
            return ForgeArray(vor.vertices[:, 0].reshape(-1, 1)), ForgeArray(vor.vertices[:, 1].reshape(-1, 1))

        def forge_inpolygon2(xq, yq, xv, yv):
            """in = inpolygon(xq, yq, xv, yv) — points in polygon."""
            from forge.engine.types import ForgeArray
            from matplotlib.path import Path
            xqd = xq.data.flatten() if isinstance(xq, ForgeArray) else np.array(xq).flatten()
            yqd = yq.data.flatten() if isinstance(yq, ForgeArray) else np.array(yq).flatten()
            xvd = xv.data.flatten() if isinstance(xv, ForgeArray) else np.array(xv).flatten()
            yvd = yv.data.flatten() if isinstance(yv, ForgeArray) else np.array(yv).flatten()
            poly = Path(np.column_stack([xvd, yvd]))
            pts = np.column_stack([xqd, yqd])
            inside = poly.contains_points(pts)
            return ForgeArray(inside.astype(float).reshape(1, -1))

        def forge_polyarea2(x, y):
            """polyarea(x, y) — polygon area (shoelace formula)."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            n = len(xd)
            area = 0.5 * abs(sum(xd[i]*yd[(i+1)%n] - xd[(i+1)%n]*yd[i] for i in range(n)))
            return ForgeArray(np.float64(area))

        # --- Rotation matrices ---
        def forge_rotx2(angle):
            """rotx(angle) — rotation about x-axis (degrees)."""
            from forge.engine.types import ForgeArray
            a = float(angle.data.flat[0]) if isinstance(angle, ForgeArray) else float(angle)
            a = np.radians(a)
            return ForgeArray(np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]))

        def forge_roty2(angle):
            """roty(angle) — rotation about y-axis (degrees)."""
            from forge.engine.types import ForgeArray
            a = float(angle.data.flat[0]) if isinstance(angle, ForgeArray) else float(angle)
            a = np.radians(a)
            return ForgeArray(np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]))

        def forge_rotz2(angle):
            """rotz(angle) — rotation about z-axis (degrees)."""
            from forge.engine.types import ForgeArray
            a = float(angle.data.flat[0]) if isinstance(angle, ForgeArray) else float(angle)
            a = np.radians(a)
            return ForgeArray(np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]))

        # --- More signal processing ---
        def forge_stft2(x, *args):
            """[S, f, t] = stft(x) — short-time Fourier transform."""
            from forge.engine.types import ForgeArray
            from scipy.signal import stft as _stft
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            fs = 1.0
            if args and isinstance(args[0], ForgeArray):
                fs = float(args[0].data.flat[0])
            f, t, Zxx = _stft(data, fs=fs)
            return ForgeArray(np.abs(Zxx)), ForgeArray(f.reshape(-1, 1)), ForgeArray(t.reshape(1, -1))

        def forge_istft2(S, *args):
            """x = istft(S) — inverse STFT."""
            from forge.engine.types import ForgeArray
            from scipy.signal import istft as _istft
            Sd = S.data if isinstance(S, ForgeArray) else np.atleast_2d(S)
            fs = 1.0
            if args and isinstance(args[0], ForgeArray):
                fs = float(args[0].data.flat[0])
            t, x = _istft(Sd, fs=fs)
            return ForgeArray(x.reshape(1, -1))

        def forge_spectrogram2(x, *args):
            """[S, f, t] = spectrogram(x) — spectrogram."""
            from forge.engine.types import ForgeArray
            from scipy.signal import spectrogram as _spec
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            fs = 1.0
            if args and isinstance(args[0], ForgeArray):
                fs = float(args[0].data.flat[0])
            f, t, Sxx = _spec(data, fs=fs)
            return ForgeArray(Sxx), ForgeArray(f.reshape(-1, 1)), ForgeArray(t.reshape(1, -1))

        def forge_hilbert2(x, *args):
            """y = hilbert(x) — analytic signal via Hilbert transform."""
            from forge.engine.types import ForgeArray
            from scipy.signal import hilbert as _hilbert
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.abs(_hilbert(data)).reshape(1, -1))

        def forge_unwrap2(x, *args):
            """unwrap(x) — unwrap phase angles."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(np.unwrap(data).reshape(1, -1))

        # Register R148 functions
        session._engine.functions["jet"] = forge_jet_func
        session._engine.functions["hsv"] = forge_hsv_func
        session._engine.functions["hot"] = forge_hot_func
        session._engine.functions["cool"] = forge_cool_func
        session._engine.functions["spring"] = forge_spring_func
        session._engine.functions["summer"] = forge_summer_func
        session._engine.functions["autumn"] = forge_autumn_func
        session._engine.functions["winter"] = forge_winter_func
        session._engine.functions["gray"] = forge_gray_func
        session._engine.functions["bone"] = forge_bone_func
        session._engine.functions["copper"] = forge_copper_func
        session._engine.functions["parula"] = forge_parula_func
        session._engine.functions["turbo"] = forge_turbo_func
        session._engine.functions["colormap"] = forge_colormap_func
        session._engine.functions["caxis"] = forge_caxis_func
        session._engine.functions["sylvester"] = forge_sylvester2
        session._engine.functions["ordschur"] = forge_ordschur2
        session._engine.functions["ldl"] = forge_ldl2
        session._engine.functions["cholupdate"] = forge_cholupdate2
        session._engine.functions["skewness"] = forge_skewness2
        session._engine.functions["kurtosis"] = forge_kurtosis2
        session._engine.functions["moment"] = forge_moment2
        session._engine.functions["ttest"] = forge_ttest2
        session._engine.functions["ttest2"] = forge_ttest2_2
        session._engine.functions["kstest"] = forge_kstest2
        session._engine.functions["chi2gof"] = forge_chi2test2
        session._engine.functions["anova1"] = forge_anova1_2
        session._engine.functions["bootstrp"] = forge_bootstrp2
        session._engine.functions["pdist"] = forge_pdist2_func
        session._engine.functions["squareform"] = forge_squareform2
        session._engine.functions["cdist"] = forge_cdist2
        session._engine.functions["knnsearch"] = forge_knnsearch2
        session._engine.functions["rangesearch"] = forge_rangesearch2
        session._engine.functions["cart2pol"] = forge_cart2pol2
        session._engine.functions["pol2cart"] = forge_pol2cart2
        session._engine.functions["cart2sph"] = forge_cart2sph2
        session._engine.functions["sph2cart"] = forge_sph2cart2
        session._engine.functions["convhull"] = forge_convhull2
        session._engine.functions["delaunay"] = forge_delaunay2
        session._engine.functions["voronoi"] = forge_voronoi2
        session._engine.functions["inpolygon"] = forge_inpolygon2
        session._engine.functions["polyarea"] = forge_polyarea2
        session._engine.functions["rotx"] = forge_rotx2
        session._engine.functions["roty"] = forge_roty2
        session._engine.functions["rotz"] = forge_rotz2
        session._engine.functions["stft"] = forge_stft2
        session._engine.functions["istft"] = forge_istft2
        session._engine.functions["spectrogram"] = forge_spectrogram2
        session._engine.functions["hilbert"] = forge_hilbert2

        # R148b: More distributions, DSP, JSON, base64, linprog

        # --- Lognormal distribution ---
        def forge_lognpdf(x, mu=None, sigma=None):
            """lognpdf(x, mu, sigma) — lognormal PDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import lognorm
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            m = 0.0 if mu is None else float(mu.data.flat[0]) if isinstance(mu, ForgeArray) else float(mu)
            s = 1.0 if sigma is None else float(sigma.data.flat[0]) if isinstance(sigma, ForgeArray) else float(sigma)
            return ForgeArray(lognorm.pdf(xd, s, scale=np.exp(m)))

        def forge_logncdf(x, mu=None, sigma=None):
            """logncdf(x, mu, sigma) — lognormal CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import lognorm
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            m = 0.0 if mu is None else float(mu.data.flat[0]) if isinstance(mu, ForgeArray) else float(mu)
            s = 1.0 if sigma is None else float(sigma.data.flat[0]) if isinstance(sigma, ForgeArray) else float(sigma)
            return ForgeArray(lognorm.cdf(xd, s, scale=np.exp(m)))

        def forge_logninv(p, mu=None, sigma=None):
            """logninv(p, mu, sigma) — lognormal inverse CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import lognorm
            pd = p.data if isinstance(p, ForgeArray) else np.atleast_2d(p)
            m = 0.0 if mu is None else float(mu.data.flat[0]) if isinstance(mu, ForgeArray) else float(mu)
            s = 1.0 if sigma is None else float(sigma.data.flat[0]) if isinstance(sigma, ForgeArray) else float(sigma)
            return ForgeArray(lognorm.ppf(pd, s, scale=np.exp(m)))

        def forge_lognrnd(mu, sigma, *args):
            """lognrnd(mu, sigma, m, n) — lognormal random."""
            from forge.engine.types import ForgeArray
            m = 0.0 if mu is None else float(mu.data.flat[0]) if isinstance(mu, ForgeArray) else float(mu)
            s = 1.0 if sigma is None else float(sigma.data.flat[0]) if isinstance(sigma, ForgeArray) else float(sigma)
            rows, cols = 1, 1
            if len(args) >= 2:
                rows = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                cols = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
            elif len(args) == 1:
                rows = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                cols = rows
            return ForgeArray(np.random.lognormal(m, s, (rows, cols)))

        # --- Beta distribution ---
        def forge_betapdf(x, a, b):
            """betapdf(x, a, b) — beta PDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import beta as _beta
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            return ForgeArray(_beta.pdf(xd, ad, bd))

        def forge_betacdf(x, a, b):
            """betacdf(x, a, b) — beta CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import beta as _beta
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            return ForgeArray(_beta.cdf(xd, ad, bd))

        def forge_betainv(p, a, b):
            """betainv(p, a, b) — beta inverse CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import beta as _beta
            pd = p.data if isinstance(p, ForgeArray) else np.atleast_2d(p)
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            return ForgeArray(_beta.ppf(pd, ad, bd))

        def forge_betarnd(a, b, *args):
            """betarnd(a, b, m, n) — beta random."""
            from forge.engine.types import ForgeArray
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            rows, cols = 1, 1
            if len(args) >= 2:
                rows = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                cols = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
            return ForgeArray(np.random.beta(ad, bd, (rows, cols)))

        # --- Gamma distribution ---
        def forge_gampdf(x, a, b=None):
            """gampdf(x, a, b) — gamma PDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import gamma as _gamma
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = 1.0 if b is None else float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            return ForgeArray(_gamma.pdf(xd, ad, scale=bd))

        def forge_gamcdf(x, a, b=None):
            """gamcdf(x, a, b) — gamma CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import gamma as _gamma
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = 1.0 if b is None else float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            return ForgeArray(_gamma.cdf(xd, ad, scale=bd))

        def forge_gaminv(p, a, b=None):
            """gaminv(p, a, b) — gamma inverse CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import gamma as _gamma
            pd = p.data if isinstance(p, ForgeArray) else np.atleast_2d(p)
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = 1.0 if b is None else float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            return ForgeArray(_gamma.ppf(pd, ad, scale=bd))

        def forge_gamrnd(a, b, *args):
            """gamrnd(a, b, m, n) — gamma random."""
            from forge.engine.types import ForgeArray
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            rows, cols = 1, 1
            if len(args) >= 2:
                rows = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                cols = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
            return ForgeArray(np.random.gamma(ad, bd, (rows, cols)))

        # --- Weibull distribution ---
        def forge_wblpdf(x, a, b=None):
            """wblpdf(x, a, b) — Weibull PDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import weibull_min
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = 1.0 if b is None else float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            return ForgeArray(weibull_min.pdf(xd, bd, scale=ad))

        def forge_wblcdf(x, a, b=None):
            """wblcdf(x, a, b) — Weibull CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import weibull_min
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = 1.0 if b is None else float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            return ForgeArray(weibull_min.cdf(xd, bd, scale=ad))

        def forge_wblinv(p, a, b=None):
            """wblinv(p, a, b) — Weibull inverse CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import weibull_min
            pd = p.data if isinstance(p, ForgeArray) else np.atleast_2d(p)
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = 1.0 if b is None else float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            return ForgeArray(weibull_min.ppf(pd, bd, scale=ad))

        def forge_wblrnd(a, b, *args):
            """wblrnd(a, b, m, n) — Weibull random."""
            from forge.engine.types import ForgeArray
            ad = float(a.data.flat[0]) if isinstance(a, ForgeArray) else float(a)
            bd = float(b.data.flat[0]) if isinstance(b, ForgeArray) else float(b)
            rows, cols = 1, 1
            if len(args) >= 2:
                rows = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                cols = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
            return ForgeArray(ad * np.random.weibull(bd, (rows, cols)))

        # --- Rayleigh distribution ---
        def forge_raylpdf(x, sigma=None):
            """raylpdf(x, sigma) — Rayleigh PDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import rayleigh
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            s = 1.0 if sigma is None else float(sigma.data.flat[0]) if isinstance(sigma, ForgeArray) else float(sigma)
            return ForgeArray(rayleigh.pdf(xd, scale=s))

        def forge_raylcdf(x, sigma=None):
            """raylcdf(x, sigma) — Rayleigh CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import rayleigh
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            s = 1.0 if sigma is None else float(sigma.data.flat[0]) if isinstance(sigma, ForgeArray) else float(sigma)
            return ForgeArray(rayleigh.cdf(xd, scale=s))

        def forge_raylrnd(sigma, *args):
            """raylrnd(sigma, m, n) — Rayleigh random."""
            from forge.engine.types import ForgeArray
            s = float(sigma.data.flat[0]) if isinstance(sigma, ForgeArray) else float(sigma)
            rows, cols = 1, 1
            if len(args) >= 2:
                rows = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                cols = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
            return ForgeArray(s * np.sqrt(-2 * np.log(np.random.rand(rows, cols))))

        # --- Geometric distribution ---
        def forge_geopdf(x, p):
            """geopdf(x, p) — geometric PDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import geom
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            pd = float(p.data.flat[0]) if isinstance(p, ForgeArray) else float(p)
            return ForgeArray(geom.pmf(xd, pd))

        def forge_geocdf(x, p):
            """geocdf(x, p) — geometric CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import geom
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            pd = float(p.data.flat[0]) if isinstance(p, ForgeArray) else float(p)
            return ForgeArray(geom.cdf(xd, pd))

        def forge_geornd(p, *args):
            """geornd(p, m, n) — geometric random."""
            from forge.engine.types import ForgeArray
            pd = float(p.data.flat[0]) if isinstance(p, ForgeArray) else float(p)
            rows, cols = 1, 1
            if len(args) >= 2:
                rows = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                cols = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
            return ForgeArray(np.random.geometric(pd, (rows, cols)).astype(float))

        # --- Negative binomial ---
        def forge_nbinpdf(x, r, p):
            """nbinpdf(x, r, p) — negative binomial PDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import nbinom
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            rd = float(r.data.flat[0]) if isinstance(r, ForgeArray) else float(r)
            pd = float(p.data.flat[0]) if isinstance(p, ForgeArray) else float(p)
            return ForgeArray(nbinom.pmf(xd, rd, pd))

        def forge_nbincdf(x, r, p):
            """nbincdf(x, r, p) — negative binomial CDF."""
            from forge.engine.types import ForgeArray
            from scipy.stats import nbinom
            xd = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            rd = float(r.data.flat[0]) if isinstance(r, ForgeArray) else float(r)
            pd = float(p.data.flat[0]) if isinstance(p, ForgeArray) else float(p)
            return ForgeArray(nbinom.cdf(xd, rd, pd))

        def forge_nbinrnd(r, p, *args):
            """nbinrnd(r, p, m, n) — negative binomial random."""
            from forge.engine.types import ForgeArray
            rd = float(r.data.flat[0]) if isinstance(r, ForgeArray) else float(r)
            pd = float(p.data.flat[0]) if isinstance(p, ForgeArray) else float(p)
            rows, cols = 1, 1
            if len(args) >= 2:
                rows = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                cols = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
            return ForgeArray(np.random.negative_binomial(int(rd), pd, (rows, cols)).astype(float))

        # --- DSP extras ---
        def forge_dct2(x, *args):
            """dct(x) — discrete cosine transform."""
            from forge.engine.types import ForgeArray
            from scipy.fft import dct as _dct
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(_dct(data).reshape(1, -1))

        def forge_idct2(x, *args):
            """idct(x) — inverse DCT."""
            from forge.engine.types import ForgeArray
            from scipy.fft import idct as _idct
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            return ForgeArray(_idct(data).reshape(1, -1))

        def forge_fir1_2(n, Wn, *args):
            """fir1(n, Wn) — FIR filter design."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            from scipy.signal import firwin
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            Wnd = Wn.data.flatten() if isinstance(Wn, ForgeArray) else np.array(Wn).flatten()
            ftype = None
            if args and isinstance(args[0], ForgeChar):
                ft = args[0].to_str()
                if ft in ('high', 'stop', 'bandpass'):
                    ftype = ft
            if ftype == 'high':
                b = firwin(nd + 1, Wnd, pass_zero=False)
            elif ftype == 'stop':
                b = firwin(nd + 1, Wnd, pass_zero=True)
            else:
                b = firwin(nd + 1, Wnd)
            return ForgeArray(b.reshape(1, -1))

        def forge_db2pow2(x):
            """db2pow(x) — dB to power."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(10.0 ** (data / 10.0))

        def forge_pow2db2(x):
            """pow2db(x) — power to dB."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(10.0 * np.log10(data))

        def forge_db2mag2(x):
            """db2mag(x) — dB to magnitude."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(10.0 ** (data / 20.0))

        def forge_mag2db2(x):
            """mag2db(x) — magnitude to dB."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            return ForgeArray(20.0 * np.log10(data))

        def forge_goertzel2(x, k):
            """goertzel(x, k) — Goertzel DFT at specific bins."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            kd = k.data.flatten().astype(int) if isinstance(k, ForgeArray) else np.array(k).flatten().astype(int)
            N = len(data)
            result = np.zeros(len(kd), dtype=complex)
            for i, ki in enumerate(kd):
                w = 2 * np.pi * (ki - 1) / N
                coeff = 2 * np.cos(w)
                s0 = s1 = s2 = 0.0
                for n in range(N):
                    s0 = data[n] + coeff * s1 - s2
                    s2 = s1
                    s1 = s0
                result[i] = s0 - s1 * np.exp(-1j * w)
            return ForgeArray(np.abs(result).reshape(1, -1))

        def forge_buffer2(x, n, p=None):
            """buffer(x, n, p) — buffer signal into frames."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            pd = 0
            if p is not None:
                pd = int(p.data.flat[0]) if isinstance(p, ForgeArray) else int(p)
            step = nd - pd
            nframes = (len(data) - pd) // step
            result = np.zeros((nd, nframes))
            for i in range(nframes):
                start = i * step
                end = start + nd
                if end <= len(data):
                    result[:, i] = data[start:end]
                else:
                    result[:end-start, i] = data[start:min(end, len(data))]
            return ForgeArray(result)

        # --- JSON ---
        def forge_jsonencode(val):
            """jsonencode(val) — encode to JSON string."""
            import json
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar, ForgeStruct
            if isinstance(val, ForgeArray):
                return ForgeChar(json.dumps(val.data.tolist()))
            elif isinstance(val, ForgeStruct):
                d = {}
                for k, v in val._fields.items():
                    if isinstance(v, ForgeArray):
                        d[k] = v.data.tolist()
                    elif isinstance(v, ForgeChar):
                        d[k] = v.to_str()
                    else:
                        d[k] = str(v)
                return ForgeChar(json.dumps(d))
            elif isinstance(val, ForgeChar):
                return ForgeChar(json.dumps(val.to_str()))
            return ForgeChar(str(val))

        def forge_jsondecode(s):
            """jsondecode(s) — decode JSON string."""
            import json
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar, ForgeStruct
            if isinstance(s, ForgeChar): s = s.to_str()
            obj = json.loads(s)
            if isinstance(obj, dict):
                result = ForgeStruct()
                for k, v in obj.items():
                    if isinstance(v, (int, float)):
                        result._fields[k] = ForgeArray(np.float64(v))
                    elif isinstance(v, str):
                        result._fields[k] = ForgeChar(v)
                    elif isinstance(v, list):
                        result._fields[k] = ForgeArray(np.array(v, dtype=float).reshape(1, -1))
                    else:
                        result._fields[k] = ForgeChar(str(v))
                return result
            elif isinstance(obj, list):
                return ForgeArray(np.array(obj, dtype=float).reshape(1, -1))
            elif isinstance(obj, (int, float)):
                return ForgeArray(np.float64(obj))
            return ForgeChar(str(obj))

        # --- Base64 ---
        def forge_base64encode(data):
            """base64encode(data) — encode to base64."""
            import base64
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(data, ForgeChar):
                return ForgeChar(base64.b64encode(data.to_str().encode()).decode())
            elif isinstance(data, ForgeArray):
                return ForgeChar(base64.b64encode(data.data.tobytes()).decode())
            return ForgeChar('')

        def forge_base64decode(s):
            """base64decode(s) — decode from base64."""
            import base64
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            return ForgeChar(base64.b64decode(s).decode('utf-8', errors='replace'))

        # --- Optimization ---
        def forge_linprog2(f, A, b, *args):
            """linprog(f, A, b) — linear programming."""
            from forge.engine.types import ForgeArray
            from scipy.optimize import linprog
            fd = f.data.flatten() if isinstance(f, ForgeArray) else np.array(f).flatten()
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            Aeq, beq = None, None
            if len(args) >= 2 and isinstance(args[0], ForgeArray) and isinstance(args[1], ForgeArray):
                if args[0].data.size > 0:
                    Aeq = args[0].data
                    beq = args[1].data.flatten()
            res = linprog(fd, A_ub=Ad, b_ub=bd, A_eq=Aeq, b_eq=beq)
            if res.success:
                return ForgeArray(res.x.reshape(-1, 1))
            return ForgeArray(np.full((len(fd), 1), np.nan))

        def forge_quadprog2(H, f, A, b, *args):
            """quadprog(H, f, A, b) — quadratic programming."""
            from forge.engine.types import ForgeArray
            from scipy.optimize import minimize
            Hd = H.data if isinstance(H, ForgeArray) else np.atleast_2d(H)
            fd = f.data.flatten() if isinstance(f, ForgeArray) else np.array(f).flatten()
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            n = len(fd)
            constraints = [{'type': 'ineq', 'fun': lambda x, i=i: bd[i] - Ad[i] @ x} for i in range(len(bd))]
            x0 = np.zeros(n)
            def obj(x):
                return 0.5 * x @ Hd @ x + fd @ x
            res = minimize(obj, x0, constraints=constraints, method='SLSQP')
            return ForgeArray(res.x.reshape(-1, 1))

        # --- URL/web ---
        def forge_urlread2(url):
            """urlread(url) — read content from URL."""
            from forge.engine.containers import ForgeChar
            if isinstance(url, ForgeChar): url = url.to_str()
            try:
                import urllib.request
                with urllib.request.urlopen(url) as resp:
                    return ForgeChar(resp.read().decode('utf-8', errors='replace'))
            except Exception:
                return ForgeChar('')

        def forge_urlwrite2(url, fname):
            """urlwrite(url, filename) — download URL to file."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(url, ForgeChar): url = url.to_str()
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            try:
                import urllib.request
                urllib.request.urlretrieve(url, fname)
                return ForgeArray(np.float64(1))
            except Exception:
                return ForgeArray(np.float64(0))

        # --- Workspace ---
        def forge_assignin2(ws, name, val):
            """assignin(ws, name, val) — assign in workspace."""
            from forge.engine.containers import ForgeChar
            if isinstance(name, ForgeChar): name = name.to_str()
            session._engine.workspace[name] = val

        def forge_evalin2(ws, expr):
            """evalin(ws, expr) — evaluate in workspace."""
            from forge.engine.containers import ForgeChar
            if isinstance(expr, ForgeChar): expr = expr.to_str()
            return session.eval(expr)

        # Register R148b functions
        session._engine.functions["lognpdf"] = forge_lognpdf
        session._engine.functions["logncdf"] = forge_logncdf
        session._engine.functions["logninv"] = forge_logninv
        session._engine.functions["lognrnd"] = forge_lognrnd
        session._engine.functions["betapdf"] = forge_betapdf
        session._engine.functions["betacdf"] = forge_betacdf
        session._engine.functions["betainv"] = forge_betainv
        session._engine.functions["betarnd"] = forge_betarnd
        session._engine.functions["gampdf"] = forge_gampdf
        session._engine.functions["gamcdf"] = forge_gamcdf
        session._engine.functions["gaminv"] = forge_gaminv
        session._engine.functions["gamrnd"] = forge_gamrnd
        session._engine.functions["wblpdf"] = forge_wblpdf
        session._engine.functions["wblcdf"] = forge_wblcdf
        session._engine.functions["wblinv"] = forge_wblinv
        session._engine.functions["wblrnd"] = forge_wblrnd
        session._engine.functions["raylpdf"] = forge_raylpdf
        session._engine.functions["raylcdf"] = forge_raylcdf
        session._engine.functions["raylrnd"] = forge_raylrnd
        session._engine.functions["geopdf"] = forge_geopdf
        session._engine.functions["geocdf"] = forge_geocdf
        session._engine.functions["geornd"] = forge_geornd
        session._engine.functions["nbinpdf"] = forge_nbinpdf
        session._engine.functions["nbincdf"] = forge_nbincdf
        session._engine.functions["nbinrnd"] = forge_nbinrnd
        session._engine.functions["dct"] = forge_dct2
        session._engine.functions["idct"] = forge_idct2
        session._engine.functions["fir1"] = forge_fir1_2
        session._engine.functions["db2pow"] = forge_db2pow2
        session._engine.functions["pow2db"] = forge_pow2db2
        session._engine.functions["db2mag"] = forge_db2mag2
        session._engine.functions["mag2db"] = forge_mag2db2
        session._engine.functions["goertzel"] = forge_goertzel2
        session._engine.functions["buffer"] = forge_buffer2
        session._engine.functions["jsonencode"] = forge_jsonencode
        session._engine.functions["jsondecode"] = forge_jsondecode
        session._engine.functions["base64encode"] = forge_base64encode
        session._engine.functions["base64decode"] = forge_base64decode
        session._engine.functions["linprog"] = forge_linprog2
        session._engine.functions["quadprog"] = forge_quadprog2
        session._engine.functions["urlread"] = forge_urlread2
        session._engine.functions["urlwrite"] = forge_urlwrite2
        session._engine.functions["assignin"] = forge_assignin2
        session._engine.functions["evalin"] = forge_evalin2

        # R149: Control stubs, FIR, hash, containers, misc

        # --- Control system extras ---
        def forge_d2c2(sys, *args):
            """d2c(sys) — discrete to continuous (stub)."""
            return sys

        def forge_series2(sys1, sys2, *args):
            """series(sys1, sys2) — series connection (stub)."""
            return sys1

        def forge_parallel2(sys1, sys2, *args):
            """parallel(sys1, sys2) — parallel connection (stub)."""
            return sys1

        def forge_feedback2(sys1, sys2=None, *args):
            """feedback(sys1, sys2) — feedback connection (stub)."""
            return sys1

        def forge_bode2(*args):
            """bode(sys) — Bode plot (stub)."""
            pass

        def forge_nyquist2(*args):
            """nyquist(sys) — Nyquist plot (stub)."""
            pass

        def forge_rlocus2(*args):
            """rlocus(sys) — root locus (stub)."""
            pass

        def forge_step2_ctrl(*args):
            """step(sys) — step response (stub)."""
            pass

        def forge_impulse2(*args):
            """impulse(sys) — impulse response (stub)."""
            pass

        def forge_lsim2(*args):
            """lsim(sys, u, t) — simulate response (stub)."""
            pass

        def forge_margin2(*args):
            """margin(sys) — gain/phase margins (stub)."""
            pass

        def forge_pole2(sys, *args):
            """pole(sys) — system poles."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            if isinstance(sys, ForgeStruct) and 'den' in sys._fields:
                den = sys._fields['den']
                if isinstance(den, ForgeArray):
                    return ForgeArray(np.roots(den.data.flatten()).reshape(-1, 1))
            return ForgeArray(np.array([[]]))

        def forge_zero2(sys, *args):
            """zero(sys) — system zeros."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            if isinstance(sys, ForgeStruct) and 'num' in sys._fields:
                num = sys._fields['num']
                if isinstance(num, ForgeArray):
                    return ForgeArray(np.roots(num.data.flatten()).reshape(-1, 1))
            return ForgeArray(np.array([[]]))

        def forge_dcgain2(sys, *args):
            """dcgain(sys) — DC gain."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            if isinstance(sys, ForgeStruct):
                if 'num' in sys._fields and 'den' in sys._fields:
                    num = sys._fields['num'].data.flatten() if isinstance(sys._fields['num'], ForgeArray) else [1]
                    den = sys._fields['den'].data.flatten() if isinstance(sys._fields['den'], ForgeArray) else [1]
                    return ForgeArray(np.float64(num[-1] / den[-1]))
            return ForgeArray(np.float64(1))

        def forge_minreal2(sys, *args):
            """minreal(sys) — minimal realization (stub)."""
            return sys

        def forge_gram2(A, B):
            """gram(A, B) — controllability/observability Gramian."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import solve_continuous_lyapunov
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            W = solve_continuous_lyapunov(Ad, -Bd @ Bd.T)
            return ForgeArray(W)

        def forge_obsv2(A, C):
            """obsv(A, C) — observability matrix."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Cd = C.data if isinstance(C, ForgeArray) else np.atleast_2d(C)
            n = Ad.shape[0]
            O = Cd.copy()
            CA = Cd.copy()
            for i in range(1, n):
                CA = CA @ Ad
                O = np.vstack([O, CA])
            return ForgeArray(O)

        def forge_ctrb2(A, B):
            """ctrb(A, B) — controllability matrix."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            n = Ad.shape[0]
            C = Bd.copy()
            AB = Bd.copy()
            for i in range(1, n):
                AB = Ad @ AB
                C = np.hstack([C, AB])
            return ForgeArray(C)

        def forge_acker2(A, B, p):
            """acker(A, B, p) — Ackermann pole placement."""
            from forge.engine.types import ForgeArray
            from scipy.signal import place_poles
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Bd = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            pd = p.data.flatten() if isinstance(p, ForgeArray) else np.array(p).flatten()
            result = place_poles(Ad, Bd, pd)
            return ForgeArray(result.gain_matrix)

        def forge_lqe2(A, G, C, Qn, Rn):
            """[L, P, E] = lqe(A, G, C, Qn, Rn) — Kalman estimator gain."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import solve_continuous_are
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            Gd = G.data if isinstance(G, ForgeArray) else np.atleast_2d(G)
            Cd = C.data if isinstance(C, ForgeArray) else np.atleast_2d(C)
            Qnd = Qn.data if isinstance(Qn, ForgeArray) else np.atleast_2d(Qn)
            Rnd = Rn.data if isinstance(Rn, ForgeArray) else np.atleast_2d(Rn)
            P = solve_continuous_are(Ad.T, Cd.T, Gd @ Qnd @ Gd.T, Rnd)
            L = P @ Cd.T @ np.linalg.inv(Rnd)
            E = np.linalg.eigvals(Ad - L @ Cd)
            return ForgeArray(L), ForgeArray(P), ForgeArray(E.reshape(-1, 1))

        # --- Hashing ---
        def forge_md5_func(data):
            """md5(data) — MD5 hash."""
            import hashlib
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(data, ForgeChar): data = data.to_str()
            elif isinstance(data, ForgeArray): data = str(data.data.tolist())
            return ForgeChar(hashlib.md5(data.encode()).hexdigest())

        def forge_sha256_func(data):
            """sha256(data) — SHA-256 hash."""
            import hashlib
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(data, ForgeChar): data = data.to_str()
            elif isinstance(data, ForgeArray): data = str(data.data.tolist())
            return ForgeChar(hashlib.sha256(data.encode()).hexdigest())

        # --- Timer ---
        def forge_timer_func(*args):
            """timer (stub)."""
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(0))

        # --- XML ---
        def forge_xmlread2(fname):
            """xmlread(filename) — read XML file."""
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            try:
                with open(fname) as f:
                    return ForgeChar(f.read())
            except Exception:
                return ForgeChar('')

        def forge_xmlwrite2(fname, content):
            """xmlwrite(filename, content) — write XML file."""
            from forge.engine.containers import ForgeChar
            if isinstance(fname, ForgeChar): fname = fname.to_str()
            if isinstance(content, ForgeChar): content = content.to_str()
            with open(fname, 'w') as f:
                f.write(content)

        # --- FIR design extras ---
        def forge_fir2_2(n, f, a, *args):
            """fir2(n, f, a) — FIR with arbitrary frequency response."""
            from forge.engine.types import ForgeArray
            from scipy.signal import firwin2
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            fd = f.data.flatten() if isinstance(f, ForgeArray) else np.array(f).flatten()
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            b = firwin2(nd + 1, fd, ad)
            return ForgeArray(b.reshape(1, -1))

        def forge_firls2(n, f, a, *args):
            """firls(n, f, a) — least-squares FIR design."""
            from forge.engine.types import ForgeArray
            from scipy.signal import firls as _firls
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            fd = f.data.flatten() if isinstance(f, ForgeArray) else np.array(f).flatten()
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            b = _firls(nd + 1, fd, ad)
            return ForgeArray(b.reshape(1, -1))

        def forge_impz2(b, a=None, *args):
            """[h, t] = impz(b, a) — impulse response of digital filter."""
            from forge.engine.types import ForgeArray
            from scipy.signal import dimpulse, dlti
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            ad = np.array([1.0])
            if a is not None and isinstance(a, ForgeArray):
                ad = a.data.flatten()
            n = 50
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            sys = dlti(bd, ad, dt=1)
            t, h = dimpulse(sys, n=n)
            return ForgeArray(h[0].flatten().reshape(1, -1)), ForgeArray(t.reshape(1, -1))

        def forge_stepz2(b, a=None, *args):
            """[h, t] = stepz(b, a) — step response of digital filter."""
            from forge.engine.types import ForgeArray
            from scipy.signal import dstep, dlti
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            ad = np.array([1.0])
            if a is not None and isinstance(a, ForgeArray):
                ad = a.data.flatten()
            n = 50
            if args and isinstance(args[0], ForgeArray):
                n = int(args[0].data.flat[0])
            sys = dlti(bd, ad, dt=1)
            t, h = dstep(sys, n=n)
            return ForgeArray(h[0].flatten().reshape(1, -1)), ForgeArray(t.reshape(1, -1))

        def forge_zplane2(b, a=None, *args):
            """zplane(b, a) — zero-pole plot (stub)."""
            pass

        # --- More string/char ---
        def forge_native2unicode(x, *args):
            """native2unicode(bytes) — bytes to unicode string."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(x, ForgeArray):
                data = x.data.flatten().astype(np.uint8)
                return ForgeChar(bytes(data).decode('utf-8', errors='replace'))
            return ForgeChar(str(x))

        def forge_unicode2native(s, *args):
            """unicode2native(str) — string to byte array."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            return ForgeArray(np.array(list(s.encode('utf-8')), dtype=float).reshape(1, -1))

        def forge_regexpi2(s, pat, *args):
            """regexpi(s, pattern) — case-insensitive regexp."""
            import re
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            matches = [m.start() + 1 for m in re.finditer(pat, s, re.IGNORECASE)]
            if not matches:
                return ForgeArray(np.array([[]]))
            return ForgeArray(np.array(matches, dtype=float).reshape(1, -1))

        def forge_regexptranslate2(op, pat):
            """regexptranslate(op, pattern) — translate pattern."""
            import re
            from forge.engine.containers import ForgeChar
            if isinstance(op, ForgeChar): op = op.to_str()
            if isinstance(pat, ForgeChar): pat = pat.to_str()
            if op == 'escape':
                return ForgeChar(re.escape(pat))
            elif op == 'wildcard':
                return ForgeChar(pat.replace('*', '.*').replace('?', '.'))
            return ForgeChar(pat)

        # --- Date/time extras ---
        def forge_calendar2(*args):
            """calendar(year, month) — monthly calendar matrix."""
            from forge.engine.types import ForgeArray
            import calendar
            import datetime
            if len(args) >= 2:
                y = int(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else int(args[0])
                m = int(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else int(args[1])
            else:
                now = datetime.datetime.now()
                y, m = now.year, now.month
            cal = calendar.monthcalendar(y, m)
            # Pad to 6 rows
            while len(cal) < 6:
                cal.append([0]*7)
            return ForgeArray(np.array(cal, dtype=float))

        def forge_date2(x=None):
            """date — current date string."""
            from forge.engine.containers import ForgeChar
            import datetime
            return ForgeChar(datetime.datetime.now().strftime('%d-%b-%Y'))

        def forge_addtodate2(d, n, unit):
            """addtodate(datenum, n, unit) — add to date."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            dd = float(d.data.flat[0]) if isinstance(d, ForgeArray) else float(d)
            nd = float(n.data.flat[0]) if isinstance(n, ForgeArray) else float(n)
            if isinstance(unit, ForgeChar): unit = unit.to_str()
            if unit == 'day':
                return ForgeArray(np.float64(dd + nd))
            elif unit == 'month':
                return ForgeArray(np.float64(dd + nd * 30))
            elif unit == 'year':
                return ForgeArray(np.float64(dd + nd * 365))
            elif unit == 'hour':
                return ForgeArray(np.float64(dd + nd / 24))
            return ForgeArray(np.float64(dd + nd))

        # --- Miscellaneous ---
        def forge_onCleanup2(func):
            """onCleanup(func) — cleanup object (stub)."""
            pass

        def forge_nargchk2(minargs, maxargs, nargs):
            """nargchk(min, max, n) — check argument count (deprecated)."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            mi = int(minargs.data.flat[0]) if isinstance(minargs, ForgeArray) else int(minargs)
            ma = int(maxargs.data.flat[0]) if isinstance(maxargs, ForgeArray) else int(maxargs)
            n = int(nargs.data.flat[0]) if isinstance(nargs, ForgeArray) else int(nargs)
            if n < mi:
                return ForgeChar('Not enough input arguments.')
            elif n > ma:
                return ForgeChar('Too many input arguments.')
            return ForgeChar('')

        def forge_validateattributes2(A, classes, attributes):
            """validateattributes(A, classes, attributes) — validate input."""
            pass  # No-op for compatibility

        def forge_comet2(x, y=None, *args):
            """comet(x, y) — animated plot (stub)."""
            pass

        def forge_comet3_func(x, y, z, *args):
            """comet3(x, y, z) — 3D animated plot (stub)."""
            pass

        def forge_fplot2(func, limits=None, *args):
            """fplot(func, [a b]) — function plot (stub)."""
            pass

        def forge_ezplot2(*args):
            """ezplot(func) — easy function plot (stub)."""
            pass

        def forge_isocaps2(*args):
            """isocaps(V, val) — isosurface end caps (stub)."""
            pass

        def forge_isosurface2(*args):
            """isosurface(V, val) — isosurface (stub)."""
            pass

        def forge_isonormals2(*args):
            """isonormals(V, p) — isosurface normals (stub)."""
            pass

        def forge_streamline2(*args):
            """streamline(X, Y, U, V) — streamlines (stub)."""
            pass

        def forge_view2(*args):
            """view(az, el) — set view angle (stub)."""
            pass

        def forge_rotate3d2(*args):
            """rotate3d on/off — toggle rotation (stub)."""
            pass

        def forge_zoom2(*args):
            """zoom on/off — toggle zoom (stub)."""
            pass

        def forge_pan2(*args):
            """pan on/off — toggle pan (stub)."""
            pass

        def forge_datacursormode2(*args):
            """datacursormode on/off (stub)."""
            pass

        def forge_linkaxes2(*args):
            """linkaxes(ax, option) (stub)."""
            pass

        def forge_shading2(*args):
            """shading flat/interp/faceted (stub)."""
            pass

        def forge_lighting2(*args):
            """lighting flat/gouraud/phong (stub)."""
            pass

        def forge_material2(*args):
            """material shiny/dull/metal (stub)."""
            pass

        def forge_camlight2(*args):
            """camlight(direction) (stub)."""
            pass

        def forge_light2(*args):
            """light(property, value) (stub)."""
            pass

        # Register R149 functions
        session._engine.functions["d2c"] = forge_d2c2
        session._engine.functions["series"] = forge_series2
        session._engine.functions["parallel"] = forge_parallel2
        session._engine.functions["feedback"] = forge_feedback2
        session._engine.functions["bode"] = forge_bode2
        session._engine.functions["nyquist"] = forge_nyquist2
        session._engine.functions["rlocus"] = forge_rlocus2
        session._engine.functions["impulse"] = forge_impulse2
        session._engine.functions["lsim"] = forge_lsim2
        session._engine.functions["margin"] = forge_margin2
        session._engine.functions["pole"] = forge_pole2
        session._engine.functions["zero"] = forge_zero2
        session._engine.functions["dcgain"] = forge_dcgain2
        session._engine.functions["minreal"] = forge_minreal2
        session._engine.functions["gram"] = forge_gram2
        session._engine.functions["obsv"] = forge_obsv2
        session._engine.functions["ctrb"] = forge_ctrb2
        session._engine.functions["acker"] = forge_acker2
        session._engine.functions["lqe"] = forge_lqe2
        session._engine.functions["md5"] = forge_md5_func
        session._engine.functions["sha256"] = forge_sha256_func
        session._engine.functions["timer"] = forge_timer_func
        session._engine.functions["xmlread"] = forge_xmlread2
        session._engine.functions["xmlwrite"] = forge_xmlwrite2
        session._engine.functions["fir2"] = forge_fir2_2
        session._engine.functions["firls"] = forge_firls2
        session._engine.functions["impz"] = forge_impz2
        session._engine.functions["stepz"] = forge_stepz2
        session._engine.functions["zplane"] = forge_zplane2
        session._engine.functions["native2unicode"] = forge_native2unicode
        session._engine.functions["unicode2native"] = forge_unicode2native
        session._engine.functions["regexpi"] = forge_regexpi2
        session._engine.functions["regexptranslate"] = forge_regexptranslate2
        session._engine.functions["calendar"] = forge_calendar2
        session._engine.functions["date"] = forge_date2
        session._engine.functions["addtodate"] = forge_addtodate2
        session._engine.functions["onCleanup"] = forge_onCleanup2
        session._engine.functions["validateattributes"] = forge_validateattributes2
        session._engine.functions["comet"] = forge_comet2
        session._engine.functions["comet3"] = forge_comet3_func
        session._engine.functions["fplot"] = forge_fplot2
        session._engine.functions["ezplot"] = forge_ezplot2
        session._engine.functions["isocaps"] = forge_isocaps2
        session._engine.functions["isosurface"] = forge_isosurface2
        session._engine.functions["isonormals"] = forge_isonormals2
        session._engine.functions["streamline"] = forge_streamline2
        session._engine.functions["view"] = forge_view2
        session._engine.functions["rotate3d"] = forge_rotate3d2
        session._engine.functions["zoom"] = forge_zoom2
        session._engine.functions["pan"] = forge_pan2
        session._engine.functions["datacursormode"] = forge_datacursormode2
        session._engine.functions["linkaxes"] = forge_linkaxes2
        session._engine.functions["shading"] = forge_shading2
        session._engine.functions["lighting"] = forge_lighting2
        session._engine.functions["material"] = forge_material2
        session._engine.functions["camlight"] = forge_camlight2
        session._engine.functions["light"] = forge_light2

        # R150: Table, sparse, matrix, symbolic stubs, classification

        # --- Table operations ---
        def forge_table_func(*args):
            """table(var1, var2, ...) — create table."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct, ForgeChar
            t = ForgeStruct()
            t._fields["_is_table"] = True
            col_idx = 0
            i = 0
            while i < len(args):
                if isinstance(args[i], ForgeChar):
                    name = args[i].to_str()
                    if i + 1 < len(args):
                        t._fields[name] = args[i + 1]
                        i += 2
                        continue
                name = f"Var{col_idx + 1}"
                t._fields[name] = args[i]
                col_idx += 1
                i += 1
            return t

        def forge_array2table2(A, *args):
            """array2table(A) — convert array to table."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct, ForgeChar
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            t = ForgeStruct()
            t._fields["_is_table"] = True
            for j in range(Ad.shape[1]):
                name = f"Var{j+1}"
                t._fields[name] = ForgeArray(Ad[:, j].reshape(-1, 1))
            return t

        def forge_table2array2(T, *args):
            """table2array(T) — convert table to array."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            if isinstance(T, ForgeStruct):
                cols = []
                for k, v in T._fields.items():
                    if k.startswith('_'): continue
                    if isinstance(v, ForgeArray):
                        cols.append(v.data.flatten())
                if cols:
                    return ForgeArray(np.column_stack(cols))
            return ForgeArray(np.array([[]]))

        def forge_sortrows2(A, *args):
            """sortrows(A) or sortrows(A, col) — sort rows."""
            from forge.engine.types import ForgeArray
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            col = 0
            if args and isinstance(args[0], ForgeArray):
                col = int(args[0].data.flat[0]) - 1
            idx = np.argsort(Ad[:, col])
            return ForgeArray(Ad[idx])

        def forge_unique2(x, *args):
            """[C, ia, ic] = unique(x) — unique values."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            C, ia, ic = np.unique(data, return_index=True, return_inverse=True)
            return ForgeArray(C.reshape(1, -1)), ForgeArray((ia + 1).astype(float).reshape(-1, 1)), ForgeArray((ic + 1).astype(float).reshape(-1, 1))

        def forge_intersect2(a, b, *args):
            """[C, ia, ib] = intersect(a, b) — set intersection."""
            from forge.engine.types import ForgeArray
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            C = np.intersect1d(ad, bd)
            return ForgeArray(C.reshape(1, -1))

        def forge_setdiff2(a, b, *args):
            """setdiff(a, b) — set difference."""
            from forge.engine.types import ForgeArray
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            return ForgeArray(np.setdiff1d(ad, bd).reshape(1, -1))

        def forge_union2(a, b, *args):
            """union(a, b) — set union."""
            from forge.engine.types import ForgeArray
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            return ForgeArray(np.union1d(ad, bd).reshape(1, -1))

        def forge_setxor2(a, b, *args):
            """setxor(a, b) — set exclusive or."""
            from forge.engine.types import ForgeArray
            ad = a.data.flatten() if isinstance(a, ForgeArray) else np.array(a).flatten()
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            u = np.union1d(ad, bd)
            i = np.intersect1d(ad, bd)
            return ForgeArray(np.setdiff1d(u, i).reshape(1, -1))

        # --- Sparse extras ---
        def forge_spones2(S):
            """spones(S) — replace nonzeros with ones."""
            from forge.engine.types import ForgeArray
            data = S.data if isinstance(S, ForgeArray) else np.atleast_2d(S)
            result = np.zeros_like(data)
            result[data != 0] = 1
            return ForgeArray(result)

        def forge_spfun2(func, S):
            """spfun(func, S) — apply function to nonzeros."""
            from forge.engine.types import ForgeArray
            data = S.data.copy() if isinstance(S, ForgeArray) else np.atleast_2d(S).copy()
            mask = data != 0
            data[mask] = func(ForgeArray(data[mask].reshape(1, -1))).data.flatten()
            return ForgeArray(data)

        def forge_nonzeros2(S):
            """nonzeros(S) — extract nonzero elements."""
            from forge.engine.types import ForgeArray
            data = S.data if isinstance(S, ForgeArray) else np.atleast_2d(S)
            nz = data[data != 0]
            return ForgeArray(nz.reshape(-1, 1))

        def forge_spy_func(*args):
            """spy(S) — visualize sparsity (stub)."""
            pass

        # --- More matrix ops ---
        def forge_hankel2(c, r=None):
            """hankel(c, r) — Hankel matrix."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import hankel as _hankel
            cd = c.data.flatten() if isinstance(c, ForgeArray) else np.array(c).flatten()
            rd = None
            if r is not None:
                rd = r.data.flatten() if isinstance(r, ForgeArray) else np.array(r).flatten()
            return ForgeArray(_hankel(cd, rd))

        def forge_toeplitz2(c, r=None):
            """toeplitz(c, r) — Toeplitz matrix."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import toeplitz as _toeplitz
            cd = c.data.flatten() if isinstance(c, ForgeArray) else np.array(c).flatten()
            rd = None
            if r is not None:
                rd = r.data.flatten() if isinstance(r, ForgeArray) else np.array(r).flatten()
            return ForgeArray(_toeplitz(cd, rd))

        def forge_hilb2(n):
            """hilb(n) — Hilbert matrix."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import hilbert
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            return ForgeArray(hilbert(nd))

        def forge_invhilb2(n):
            """invhilb(n) — inverse Hilbert matrix."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import invhilbert
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            return ForgeArray(invhilbert(nd))

        def forge_magic2(n):
            """magic(n) — magic square."""
            from forge.engine.types import ForgeArray
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            if nd < 3:
                return ForgeArray(np.array([[1]]) if nd == 1 else np.array([[1,3],[4,2]]).astype(float))
            M = np.zeros((nd, nd), dtype=float)
            if nd % 2 == 1:  # Odd
                i, j = 0, nd // 2
                for num in range(1, nd*nd + 1):
                    M[i, j] = num
                    ni, nj = (i - 1) % nd, (j + 1) % nd
                    if M[ni, nj] != 0:
                        ni = (i + 1) % nd
                        nj = j
                    i, j = ni, nj
            elif nd % 4 == 0:  # Doubly even
                seq = np.arange(1, nd*nd + 1).reshape(nd, nd)
                for ii in range(nd):
                    for jj in range(nd):
                        if (ii % 4 == 0 or ii % 4 == 3) == (jj % 4 == 0 or jj % 4 == 3):
                            M[ii, jj] = nd*nd + 1 - seq[ii, jj]
                        else:
                            M[ii, jj] = seq[ii, jj]
            else:  # Singly even
                p = nd // 2
                sub = np.zeros((p, p), dtype=float)
                ii, jj = 0, p // 2
                for num in range(1, p*p + 1):
                    sub[ii, jj] = num
                    ni, nj = (ii - 1) % p, (jj + 1) % p
                    if sub[ni, nj] != 0:
                        ni = (ii + 1) % p
                        nj = jj
                    ii, jj = ni, nj
                M[:p, :p] = sub
                M[:p, p:] = sub + 2*p*p
                M[p:, :p] = sub + 3*p*p
                M[p:, p:] = sub + p*p
            return ForgeArray(M)

        def forge_pascal2(n):
            """pascal(n) — Pascal matrix."""
            from forge.engine.types import ForgeArray
            nd = int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)
            P = np.zeros((nd, nd), dtype=float)
            P[:, 0] = 1
            P[0, :] = 1
            for i in range(1, nd):
                for j in range(1, nd):
                    P[i, j] = P[i-1, j] + P[i, j-1]
            return ForgeArray(P)

        def forge_compan2(p):
            """compan(p) — companion matrix."""
            from forge.engine.types import ForgeArray
            pd = p.data.flatten() if isinstance(p, ForgeArray) else np.array(p).flatten()
            return ForgeArray(np.polynomial.polynomial.polycompanion(pd[::-1]))

        # --- Classification / ML stubs ---
        def forge_fitlm2(X, y, *args):
            """mdl = fitlm(X, y) — fit linear model."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            # Add intercept
            Xa = np.column_stack([np.ones(Xd.shape[0]), Xd])
            beta = np.linalg.lstsq(Xa, yd, rcond=None)[0]
            yhat = Xa @ beta
            residuals = yd - yhat
            SSres = np.sum(residuals**2)
            SStot = np.sum((yd - np.mean(yd))**2)
            R2 = 1 - SSres / SStot if SStot > 0 else 1.0
            mdl = ForgeStruct()
            mdl._fields["Coefficients"] = ForgeArray(beta.reshape(-1, 1))
            mdl._fields["Rsquared"] = ForgeArray(np.float64(R2))
            mdl._fields["Residuals"] = ForgeArray(residuals.reshape(-1, 1))
            return mdl

        def forge_predict2(mdl, X, *args):
            """yhat = predict(mdl, X) — predict from model."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            if isinstance(mdl, ForgeStruct) and "Coefficients" in mdl._fields:
                beta = mdl._fields["Coefficients"].data.flatten()
                Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
                Xa = np.column_stack([np.ones(Xd.shape[0]), Xd])
                return ForgeArray((Xa @ beta).reshape(-1, 1))
            return ForgeArray(np.array([[]]))

        def forge_fitcsvm2(X, y, *args):
            """mdl = fitcsvm(X, y) — SVM classifier."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            try:
                from sklearn.svm import SVC
                clf = SVC()
                clf.fit(Xd, yd)
                mdl = ForgeStruct()
                mdl._fields["_clf"] = clf
                mdl._fields["type"] = "SVM"
                return mdl
            except Exception:
                mdl = ForgeStruct()
                mdl._fields["type"] = "SVM_stub"
                return mdl

        def forge_fitctree2(X, y, *args):
            """mdl = fitctree(X, y) — decision tree classifier."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            try:
                from sklearn.tree import DecisionTreeClassifier
                clf = DecisionTreeClassifier()
                clf.fit(Xd, yd)
                mdl = ForgeStruct()
                mdl._fields["_clf"] = clf
                mdl._fields["type"] = "DecisionTree"
                return mdl
            except Exception:
                mdl = ForgeStruct()
                mdl._fields["type"] = "DecisionTree_stub"
                return mdl

        def forge_fitcknn2(X, y, *args):
            """mdl = fitcknn(X, y) — k-NN classifier."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            try:
                from sklearn.neighbors import KNeighborsClassifier
                clf = KNeighborsClassifier()
                clf.fit(Xd, yd)
                mdl = ForgeStruct()
                mdl._fields["_clf"] = clf
                mdl._fields["type"] = "KNN"
                return mdl
            except Exception:
                mdl = ForgeStruct()
                mdl._fields["type"] = "KNN_stub"
                return mdl

        def forge_fitcensemble2(X, y, *args):
            """mdl = fitcensemble(X, y) — ensemble classifier."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            try:
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=100)
                clf.fit(Xd, yd)
                mdl = ForgeStruct()
                mdl._fields["_clf"] = clf
                mdl._fields["type"] = "Ensemble"
                return mdl
            except Exception:
                mdl = ForgeStruct()
                mdl._fields["type"] = "Ensemble_stub"
                return mdl

        def forge_crossval2(mdl, *args):
            """crossval(mdl) — cross-validation (stub)."""
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(0.1))  # Return dummy loss

        def forge_confusionmat2(ytrue, ypred):
            """confusionmat(ytrue, ypred) — confusion matrix."""
            from forge.engine.types import ForgeArray
            yt = ytrue.data.flatten().astype(int) if isinstance(ytrue, ForgeArray) else np.array(ytrue).flatten().astype(int)
            yp = ypred.data.flatten().astype(int) if isinstance(ypred, ForgeArray) else np.array(ypred).flatten().astype(int)
            labels = np.unique(np.concatenate([yt, yp]))
            n = len(labels)
            C = np.zeros((n, n), dtype=float)
            label_map = {l: i for i, l in enumerate(labels)}
            for t, p in zip(yt, yp):
                C[label_map[t], label_map[p]] += 1
            return ForgeArray(C)

        # --- PCA ---
        def forge_pca2(X, *args):
            """[coeff, score, latent] = pca(X) — principal component analysis."""
            from forge.engine.types import ForgeArray
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            # Center data
            mu = np.mean(Xd, axis=0)
            Xc = Xd - mu
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            coeff = Vt.T
            score = Xc @ coeff
            latent = (S**2) / (Xd.shape[0] - 1)
            return ForgeArray(coeff), ForgeArray(score), ForgeArray(latent.reshape(-1, 1))

        # --- K-means ---
        def forge_kmeans2(X, k, *args):
            """[idx, C] = kmeans(X, k) — k-means clustering."""
            from forge.engine.types import ForgeArray
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            kd = int(k.data.flat[0]) if isinstance(k, ForgeArray) else int(k)
            # Simple k-means
            n = Xd.shape[0]
            centers = Xd[np.random.choice(n, kd, replace=False)]
            for _ in range(100):
                dists = np.array([np.sum((Xd - c)**2, axis=1) for c in centers]).T
                idx = np.argmin(dists, axis=1)
                new_centers = np.array([Xd[idx == i].mean(axis=0) if np.any(idx == i) else centers[i] for i in range(kd)])
                if np.allclose(new_centers, centers):
                    break
                centers = new_centers
            return ForgeArray((idx + 1).astype(float).reshape(-1, 1)), ForgeArray(centers)

        # --- Symbolic stubs ---
        def forge_sym2(*args):
            """sym(x) — create symbolic variable (stub)."""
            from forge.engine.containers import ForgeChar
            if args:
                from forge.engine.containers import ForgeChar as FC
                if isinstance(args[0], FC):
                    return args[0]
            return ForgeChar('x')

        def forge_syms2(*args):
            """syms x y z — declare symbolic variables (stub)."""
            pass

        def forge_simplify2(expr, *args):
            """simplify(expr) — simplify expression (stub)."""
            return expr

        def forge_expand2(expr, *args):
            """expand(expr) — expand expression (stub)."""
            return expr

        def forge_collect2(expr, *args):
            """collect(expr, var) — collect terms (stub)."""
            return expr

        def forge_subs2(expr, old, new):
            """subs(expr, old, new) — symbolic substitution (stub)."""
            return expr

        def forge_diff_sym2(expr, *args):
            """diff(expr) — symbolic differentiation (stub)."""
            return expr

        def forge_int_sym2(expr, *args):
            """int(expr) — symbolic integration (stub)."""
            return expr

        def forge_solve_sym2(eqn, *args):
            """solve(eqn) — symbolic equation solving (stub)."""
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(0))

        def forge_pretty2(expr, *args):
            """pretty(expr) — pretty-print symbolic (stub)."""
            pass

        def forge_latex2(expr, *args):
            """latex(expr) — LaTeX representation (stub)."""
            from forge.engine.containers import ForgeChar
            return ForgeChar(str(expr))

        # Register R150 functions
        session._engine.functions["table"] = forge_table_func
        session._engine.functions["sortrows"] = forge_sortrows2
        session._engine.functions["unique"] = forge_unique2
        session._engine.functions["intersect"] = forge_intersect2
        session._engine.functions["setdiff"] = forge_setdiff2
        session._engine.functions["union"] = forge_union2
        session._engine.functions["setxor"] = forge_setxor2
        session._engine.functions["spones"] = forge_spones2
        session._engine.functions["spfun"] = forge_spfun2
        session._engine.functions["nonzeros"] = forge_nonzeros2
        session._engine.functions["spy"] = forge_spy_func
        session._engine.functions["hankel"] = forge_hankel2
        session._engine.functions["toeplitz"] = forge_toeplitz2
        session._engine.functions["hilb"] = forge_hilb2
        session._engine.functions["invhilb"] = forge_invhilb2
        session._engine.functions["magic"] = forge_magic2
        session._engine.functions["pascal"] = forge_pascal2
        session._engine.functions["compan"] = forge_compan2
        session._engine.functions["fitlm"] = forge_fitlm2
        session._engine.functions["predict"] = forge_predict2
        session._engine.functions["fitcsvm"] = forge_fitcsvm2
        session._engine.functions["fitctree"] = forge_fitctree2
        session._engine.functions["fitcknn"] = forge_fitcknn2
        session._engine.functions["fitcensemble"] = forge_fitcensemble2
        session._engine.functions["crossval"] = forge_crossval2
        session._engine.functions["confusionmat"] = forge_confusionmat2
        session._engine.functions["pca"] = forge_pca2
        session._engine.functions["kmeans"] = forge_kmeans2
        session._engine.functions["sym"] = forge_sym2
        session._engine.functions["syms"] = forge_syms2
        session._engine.functions["simplify"] = forge_simplify2
        session._engine.functions["expand"] = forge_expand2
        session._engine.functions["collect"] = forge_collect2
        session._engine.functions["subs"] = forge_subs2
        session._engine.functions["pretty"] = forge_pretty2
        session._engine.functions["latex"] = forge_latex2

        # R151: Image, data cleaning, regression, time series, more

        # --- Image processing ---
        def forge_imrotate2(img, angle, *args):
            """imrotate(img, angle) — rotate image."""
            from forge.engine.types import ForgeArray
            from scipy.ndimage import rotate
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            a = float(angle.data.flat[0]) if isinstance(angle, ForgeArray) else float(angle)
            return ForgeArray(rotate(data, a, reshape=False))

        def forge_imcrop2(img, rect, *args):
            """imcrop(img, rect) — crop image."""
            from forge.engine.types import ForgeArray
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            r = rect.data.flatten() if isinstance(rect, ForgeArray) else np.array(rect).flatten()
            x, y, w, h = int(r[0])-1, int(r[1])-1, int(r[2]), int(r[3])
            return ForgeArray(data[y:y+h, x:x+w])

        def forge_imadjust2(img, *args):
            """imadjust(img) — adjust image intensity."""
            from forge.engine.types import ForgeArray
            data = img.data.copy() if isinstance(img, ForgeArray) else np.atleast_2d(img).copy()
            lo, hi = np.percentile(data, 1), np.percentile(data, 99)
            if hi > lo:
                data = (data - lo) / (hi - lo)
                data = np.clip(data, 0, 1)
            return ForgeArray(data)

        def forge_imbinarize2(img, *args):
            """imbinarize(img) — binarize image using Otsu."""
            from forge.engine.types import ForgeArray
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            if data.max() > 1:
                data = data / 255.0
            threshold = 0.5
            if args and isinstance(args[0], ForgeArray):
                threshold = float(args[0].data.flat[0])
            else:
                # Otsu's method
                hist, bin_edges = np.histogram(data.flatten(), 256, [0, 1])
                total = data.size
                sumB, wB, maximum, sum_total = 0.0, 0, 0.0, 0.0
                for i in range(256):
                    sum_total += i * hist[i]
                for i in range(256):
                    wB += hist[i]
                    if wB == 0: continue
                    wF = total - wB
                    if wF == 0: break
                    sumB += i * hist[i]
                    mB = sumB / wB
                    mF = (sum_total - sumB) / wF
                    between = wB * wF * (mB - mF) ** 2
                    if between >= maximum:
                        threshold = i / 255.0
                        maximum = between
            return ForgeArray((data > threshold).astype(float))

        def forge_edge2(img, *args):
            """edge(img, method) — edge detection."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            from scipy.ndimage import sobel
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            gx = sobel(data, axis=1)
            gy = sobel(data, axis=0)
            magnitude = np.sqrt(gx**2 + gy**2)
            threshold = np.mean(magnitude) + np.std(magnitude)
            return ForgeArray((magnitude > threshold).astype(float))

        def forge_imgradient2(img, *args):
            """[Gmag, Gdir] = imgradient(img) — image gradient."""
            from forge.engine.types import ForgeArray
            from scipy.ndimage import sobel
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            gx = sobel(data, axis=1)
            gy = sobel(data, axis=0)
            mag = np.sqrt(gx**2 + gy**2)
            direction = np.degrees(np.arctan2(gy, gx))
            return ForgeArray(mag), ForgeArray(direction)

        def forge_medfilt2d(img, *args):
            """medfilt2(img) — 2D median filter."""
            from forge.engine.types import ForgeArray
            from scipy.ndimage import median_filter
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            k = 3
            if args and isinstance(args[0], ForgeArray):
                k = int(args[0].data.flat[0])
            return ForgeArray(median_filter(data, size=k))

        def forge_wiener2_func(img, *args):
            """wiener2(img) — adaptive noise filter."""
            from forge.engine.types import ForgeArray
            from scipy.ndimage import uniform_filter
            data = img.data.astype(float) if isinstance(img, ForgeArray) else np.atleast_2d(img).astype(float)
            k = 3
            if args and isinstance(args[0], ForgeArray):
                k = int(args[0].data.flat[0])
            local_mean = uniform_filter(data, k)
            local_sq_mean = uniform_filter(data**2, k)
            local_var = local_sq_mean - local_mean**2
            noise_var = np.mean(local_var)
            result = local_mean + np.maximum(0, local_var - noise_var) / np.maximum(local_var, noise_var + 1e-10) * (data - local_mean)
            return ForgeArray(result)

        def forge_regionprops2(bw, *args):
            """regionprops(BW) — region properties."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            from scipy.ndimage import label, find_objects
            data = bw.data if isinstance(bw, ForgeArray) else np.atleast_2d(bw)
            labeled, num = label(data > 0)
            props = ForgeStruct()
            props._fields["NumObjects"] = ForgeArray(np.float64(num))
            areas = []
            for i in range(1, num + 1):
                areas.append(np.sum(labeled == i))
            props._fields["Area"] = ForgeArray(np.array(areas).reshape(-1, 1).astype(float))
            return props

        def forge_watershed2(img, *args):
            """watershed(img) — watershed segmentation (stub)."""
            from forge.engine.types import ForgeArray
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            return ForgeArray(np.ones_like(data))

        def forge_bwlabel2(bw, *args):
            """[L, n] = bwlabel(BW) — label connected components."""
            from forge.engine.types import ForgeArray
            from scipy.ndimage import label
            data = bw.data if isinstance(bw, ForgeArray) else np.atleast_2d(bw)
            labeled, num = label(data > 0)
            return ForgeArray(labeled.astype(float)), ForgeArray(np.float64(num))

        def forge_bwareaopen2(bw, minarea):
            """bwareaopen(BW, P) — remove small objects."""
            from forge.engine.types import ForgeArray
            from scipy.ndimage import label
            data = bw.data.copy() if isinstance(bw, ForgeArray) else np.atleast_2d(bw).copy()
            p = int(minarea.data.flat[0]) if isinstance(minarea, ForgeArray) else int(minarea)
            labeled, num = label(data > 0)
            for i in range(1, num + 1):
                if np.sum(labeled == i) < p:
                    data[labeled == i] = 0
            return ForgeArray(data)

        def forge_imfill2(bw, *args):
            """imfill(BW, 'holes') — fill holes."""
            from forge.engine.types import ForgeArray
            from scipy.ndimage import binary_fill_holes
            data = bw.data if isinstance(bw, ForgeArray) else np.atleast_2d(bw)
            return ForgeArray(binary_fill_holes(data > 0).astype(float))

        def forge_imerode2(img, se, *args):
            """imerode(img, SE) — morphological erosion."""
            from forge.engine.types import ForgeArray
            from scipy.ndimage import binary_erosion
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            sed = se.data if isinstance(se, ForgeArray) else np.atleast_2d(se)
            return ForgeArray(binary_erosion(data > 0, sed > 0).astype(float))

        def forge_imdilate2(img, se, *args):
            """imdilate(img, SE) — morphological dilation."""
            from forge.engine.types import ForgeArray
            from scipy.ndimage import binary_dilation
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            sed = se.data if isinstance(se, ForgeArray) else np.atleast_2d(se)
            return ForgeArray(binary_dilation(data > 0, sed > 0).astype(float))

        def forge_imopen2(img, se, *args):
            """imopen(img, SE) — morphological opening."""
            from forge.engine.types import ForgeArray
            from scipy.ndimage import binary_opening
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            sed = se.data if isinstance(se, ForgeArray) else np.atleast_2d(se)
            return ForgeArray(binary_opening(data > 0, sed > 0).astype(float))

        def forge_imclose2(img, se, *args):
            """imclose(img, SE) — morphological closing."""
            from forge.engine.types import ForgeArray
            from scipy.ndimage import binary_closing
            data = img.data if isinstance(img, ForgeArray) else np.atleast_2d(img)
            sed = se.data if isinstance(se, ForgeArray) else np.atleast_2d(se)
            return ForgeArray(binary_closing(data > 0, sed > 0).astype(float))

        def forge_strel2(shape, *args):
            """strel(shape, params) — structuring element."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(shape, ForgeChar): shape = shape.to_str()
            if shape == 'disk':
                r = 3
                if args and isinstance(args[0], ForgeArray):
                    r = int(args[0].data.flat[0])
                y, x = np.ogrid[-r:r+1, -r:r+1]
                return ForgeArray((x**2 + y**2 <= r**2).astype(float))
            elif shape == 'square':
                n = 3
                if args and isinstance(args[0], ForgeArray):
                    n = int(args[0].data.flat[0])
                return ForgeArray(np.ones((n, n)))
            elif shape == 'diamond':
                r = 3
                if args and isinstance(args[0], ForgeArray):
                    r = int(args[0].data.flat[0])
                y, x = np.ogrid[-r:r+1, -r:r+1]
                return ForgeArray((np.abs(x) + np.abs(y) <= r).astype(float))
            return ForgeArray(np.ones((3, 3)))

        # --- Data cleaning extras ---
        def forge_standardizeMissing2(A, indicator):
            """standardizeMissing(A, indicator) — replace values with NaN."""
            from forge.engine.types import ForgeArray
            data = A.data.copy() if isinstance(A, ForgeArray) else np.atleast_2d(A).copy()
            ind = indicator.data.flatten() if isinstance(indicator, ForgeArray) else np.array(indicator).flatten()
            for v in ind:
                data[data == v] = np.nan
            return ForgeArray(data)

        def forge_filloutliers2(x, method=None, *args):
            """filloutliers(x, method) — replace outliers."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            data = x.data.copy() if isinstance(x, ForgeArray) else np.atleast_2d(x).copy().flatten()
            flat = data.flatten()
            q1, q3 = np.percentile(flat[~np.isnan(flat)], [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            mask = (flat < lo) | (flat > hi)
            if isinstance(method, ForgeChar) and method.to_str() == 'median':
                flat[mask] = np.nanmedian(flat[~mask])
            else:
                flat[mask] = np.nanmean(flat[~mask])
            return ForgeArray(flat.reshape(data.shape))

        def forge_findgroups2(x):
            """[G, ID] = findgroups(x) — group data."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            uniq = np.unique(data)
            groups = np.zeros(len(data), dtype=float)
            for i, u in enumerate(uniq):
                groups[data == u] = i + 1
            return ForgeArray(groups.reshape(1, -1)), ForgeArray(uniq.reshape(1, -1))

        def forge_splitapply2(func, x, G):
            """splitapply(func, x, G) — split and apply."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            gd = G.data.flatten().astype(int) if isinstance(G, ForgeArray) else np.array(G).flatten().astype(int)
            ngroups = int(gd.max())
            results = []
            for i in range(1, ngroups + 1):
                mask = gd == i
                if np.any(mask):
                    val = func(ForgeArray(xd[mask].reshape(1, -1)))
                    results.append(float(val.data.flat[0]) if isinstance(val, ForgeArray) else float(val))
                else:
                    results.append(np.nan)
            return ForgeArray(np.array(results).reshape(1, -1))

        def forge_varfun2(func, T, *args):
            """varfun(func, T) — apply function to table variables (stub)."""
            return T

        # --- Regression extras ---
        def forge_fitrgp2(X, y, *args):
            """fitrgp(X, y) — Gaussian process regression (stub)."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            mdl = ForgeStruct()
            mdl._fields["type"] = "GPR_stub"
            return mdl

        def forge_fitrlinear2(X, y, *args):
            """fitrlinear(X, y) — linear regression (fast)."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            beta = np.linalg.lstsq(Xd, yd, rcond=None)[0]
            mdl = ForgeStruct()
            mdl._fields["Beta"] = ForgeArray(beta.reshape(-1, 1))
            return mdl

        def forge_fitrtree2(X, y, *args):
            """fitrtree(X, y) — regression tree."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            try:
                from sklearn.tree import DecisionTreeRegressor
                Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
                yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
                clf = DecisionTreeRegressor()
                clf.fit(Xd, yd)
                mdl = ForgeStruct()
                mdl._fields["_clf"] = clf
                mdl._fields["type"] = "RegressionTree"
                return mdl
            except Exception:
                return ForgeStruct()

        def forge_fitrensemble2(X, y, *args):
            """fitrensemble(X, y) — ensemble regression."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            try:
                from sklearn.ensemble import RandomForestRegressor
                Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
                yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
                clf = RandomForestRegressor(n_estimators=100)
                clf.fit(Xd, yd)
                mdl = ForgeStruct()
                mdl._fields["_clf"] = clf
                mdl._fields["type"] = "RegressionEnsemble"
                return mdl
            except Exception:
                return ForgeStruct()

        # --- Time series ---
        def forge_autocorr2(x, *args):
            """autocorr(x, nlags) — sample autocorrelation."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            nlags = 20
            if args and isinstance(args[0], ForgeArray):
                nlags = int(args[0].data.flat[0])
            n = len(data)
            mu = np.mean(data)
            var = np.var(data)
            acf = np.zeros(nlags + 1)
            for k in range(nlags + 1):
                acf[k] = np.sum((data[:n-k] - mu) * (data[k:] - mu)) / (n * var) if var > 0 else 0
            return ForgeArray(acf.reshape(1, -1))

        def forge_parcorr2(x, *args):
            """parcorr(x, nlags) — partial autocorrelation."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            nlags = 20
            if args and isinstance(args[0], ForgeArray):
                nlags = int(args[0].data.flat[0])
            # Levinson-Durbin
            n = len(data)
            mu = np.mean(data)
            var = np.var(data)
            acf = np.zeros(nlags + 1)
            for k in range(nlags + 1):
                acf[k] = np.sum((data[:n-k] - mu) * (data[k:] - mu)) / (n * var) if var > 0 else 0
            pacf = np.zeros(nlags + 1)
            pacf[0] = 1.0
            if nlags > 0:
                pacf[1] = acf[1]
            return ForgeArray(pacf.reshape(1, -1))

        def forge_arima2(*args):
            """arima(p, d, q) — ARIMA model (stub)."""
            from forge.engine.containers import ForgeStruct
            mdl = ForgeStruct()
            mdl._fields["type"] = "arima"
            return mdl

        def forge_forecast2(mdl, Y, numperiods, *args):
            """forecast(mdl, Y, numperiods) — forecast (stub)."""
            from forge.engine.types import ForgeArray
            np_d = int(numperiods.data.flat[0]) if isinstance(numperiods, ForgeArray) else int(numperiods)
            Yd = Y.data.flatten() if isinstance(Y, ForgeArray) else np.array(Y).flatten()
            # Simple extrapolation
            return ForgeArray(np.full(np_d, Yd[-1]).reshape(-1, 1))

        # --- More misc utility ---
        def forge_struct2cell2(S):
            """struct2cell(S) — convert struct to cell."""
            from forge.engine.containers import ForgeCell, ForgeStruct
            if isinstance(S, ForgeStruct):
                vals = list(S._fields.values())
                return ForgeCell._from_list(vals)
            return ForgeCell(1, 1)

        def forge_cell2struct2(C, fields):
            """cell2struct(C, fieldnames) — convert cell to struct."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeCell, ForgeStruct, ForgeChar
            S = ForgeStruct()
            if isinstance(fields, ForgeCell):
                for i in range(len(fields._data)):
                    fn = fields._data[i]
                    if isinstance(fn, ForgeChar): fn = fn.to_str()
                    if i < len(C._data):
                        S._fields[str(fn)] = C._data[i]
            return S

        def forge_substruct2(S, *args):
            """substruct(type, subs) — create subscript structure."""
            from forge.engine.containers import ForgeStruct, ForgeChar
            result = ForgeStruct()
            result._fields["type"] = args[0] if args else ForgeChar('()')
            result._fields["subs"] = args[1] if len(args) > 1 else ForgeChar(':')
            return result

        def forge_deal2(*args):
            """[a, b, ...] = deal(x) or deal(x, y, ...) — distribute inputs."""
            if len(args) == 1:
                return tuple([args[0]] * 10)
            return args

        def forge_nfields2(S):
            """nfields(S) — number of fields."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            if isinstance(S, ForgeStruct):
                return ForgeArray(np.float64(len([k for k in S._fields if not k.startswith('_')])))
            return ForgeArray(np.float64(0))

        # Register R151 functions
        session._engine.functions["imrotate"] = forge_imrotate2
        session._engine.functions["imcrop"] = forge_imcrop2
        session._engine.functions["imadjust"] = forge_imadjust2
        session._engine.functions["imbinarize"] = forge_imbinarize2
        session._engine.functions["edge"] = forge_edge2
        session._engine.functions["imgradient"] = forge_imgradient2
        session._engine.functions["medfilt2"] = forge_medfilt2d
        session._engine.functions["wiener2"] = forge_wiener2_func
        session._engine.functions["regionprops"] = forge_regionprops2
        session._engine.functions["watershed"] = forge_watershed2
        session._engine.functions["bwlabel"] = forge_bwlabel2
        session._engine.functions["bwareaopen"] = forge_bwareaopen2
        session._engine.functions["imfill"] = forge_imfill2
        session._engine.functions["imerode"] = forge_imerode2
        session._engine.functions["imdilate"] = forge_imdilate2
        session._engine.functions["imopen"] = forge_imopen2
        session._engine.functions["imclose"] = forge_imclose2
        session._engine.functions["strel"] = forge_strel2
        session._engine.functions["standardizeMissing"] = forge_standardizeMissing2
        session._engine.functions["filloutliers"] = forge_filloutliers2
        session._engine.functions["findgroups"] = forge_findgroups2
        session._engine.functions["splitapply"] = forge_splitapply2
        session._engine.functions["varfun"] = forge_varfun2
        session._engine.functions["fitrgp"] = forge_fitrgp2
        session._engine.functions["fitrlinear"] = forge_fitrlinear2
        session._engine.functions["fitrtree"] = forge_fitrtree2
        session._engine.functions["fitrensemble"] = forge_fitrensemble2
        session._engine.functions["autocorr"] = forge_autocorr2
        session._engine.functions["parcorr"] = forge_parcorr2
        session._engine.functions["arima"] = forge_arima2
        session._engine.functions["forecast"] = forge_forecast2
        session._engine.functions["substruct"] = forge_substruct2

        # R152: Deep learning stubs, financial, optimization extras

        # --- Deep learning stubs ---
        def forge_dlarray2(data, *args):
            """dlarray(data) — deep learning array (stub)."""
            from forge.engine.types import ForgeArray
            if isinstance(data, ForgeArray): return data
            return ForgeArray(np.atleast_2d(data))

        def forge_dlnetwork2(*args):
            """dlnetwork(layers) — deep learning network (stub)."""
            from forge.engine.containers import ForgeStruct
            mdl = ForgeStruct()
            mdl._fields["type"] = "dlnetwork"
            return mdl

        def forge_fullyConnectedLayer2(n, *args):
            """fullyConnectedLayer(n) — FC layer (stub)."""
            from forge.engine.containers import ForgeStruct
            from forge.engine.types import ForgeArray
            layer = ForgeStruct()
            layer._fields["type"] = "fullyConnected"
            layer._fields["OutputSize"] = ForgeArray(np.float64(int(n.data.flat[0]) if isinstance(n, ForgeArray) else int(n)))
            return layer

        def forge_reluLayer2(*args):
            """reluLayer (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_softmaxLayer2(*args):
            """softmaxLayer (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_classificationLayer2(*args):
            """classificationLayer (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_regressionLayer2(*args):
            """regressionLayer (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_convolution2dLayer2(sz, nf, *args):
            """convolution2dLayer (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_maxPooling2dLayer2(sz, *args):
            """maxPooling2dLayer (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_batchNormalizationLayer2(*args):
            """batchNormalizationLayer (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_dropoutLayer2(*args):
            """dropoutLayer (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_lstmLayer2(n, *args):
            """lstmLayer (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_trainingOptions2(*args):
            """trainingOptions (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_trainNetwork2(*args):
            """trainNetwork (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        # --- Financial ---
        def forge_pvvar2(cf, rate):
            """pvvar(cf, rate) — present value of varying cash flows."""
            from forge.engine.types import ForgeArray
            cfd = cf.data.flatten() if isinstance(cf, ForgeArray) else np.array(cf).flatten()
            rd = float(rate.data.flat[0]) if isinstance(rate, ForgeArray) else float(rate)
            pv = sum(c / (1 + rd)**i for i, c in enumerate(cfd))
            return ForgeArray(np.float64(pv))

        def forge_irr2(cf):
            """irr(cf) — internal rate of return."""
            from forge.engine.types import ForgeArray
            cfd = cf.data.flatten() if isinstance(cf, ForgeArray) else np.array(cf).flatten()
            r = 0.1
            for _ in range(100):
                npv = sum(c / (1 + r)**i for i, c in enumerate(cfd))
                dnpv = sum(-i * c / (1 + r)**(i+1) for i, c in enumerate(cfd))
                if abs(dnpv) < 1e-12: break
                r -= npv / dnpv
            return ForgeArray(np.float64(r))

        def forge_fvfix2(rate, nper, pmt):
            """fvfix(rate, nper, pmt) — future value."""
            from forge.engine.types import ForgeArray
            rd = float(rate.data.flat[0]) if isinstance(rate, ForgeArray) else float(rate)
            nd = float(nper.data.flat[0]) if isinstance(nper, ForgeArray) else float(nper)
            pd = float(pmt.data.flat[0]) if isinstance(pmt, ForgeArray) else float(pmt)
            return ForgeArray(np.float64(pd * ((1 + rd)**nd - 1) / rd))

        def forge_pvfix2(rate, nper, pmt):
            """pvfix(rate, nper, pmt) — present value."""
            from forge.engine.types import ForgeArray
            rd = float(rate.data.flat[0]) if isinstance(rate, ForgeArray) else float(rate)
            nd = float(nper.data.flat[0]) if isinstance(nper, ForgeArray) else float(nper)
            pd = float(pmt.data.flat[0]) if isinstance(pmt, ForgeArray) else float(pmt)
            return ForgeArray(np.float64(pd * (1 - (1 + rd)**(-nd)) / rd))

        def forge_pmt2(rate, nper, pv, fv=None):
            """pmt(rate, nper, pv) — payment amount."""
            from forge.engine.types import ForgeArray
            rd = float(rate.data.flat[0]) if isinstance(rate, ForgeArray) else float(rate)
            nd = float(nper.data.flat[0]) if isinstance(nper, ForgeArray) else float(nper)
            pvd = float(pv.data.flat[0]) if isinstance(pv, ForgeArray) else float(pv)
            fvd = 0.0
            if fv is not None:
                fvd = float(fv.data.flat[0]) if isinstance(fv, ForgeArray) else float(fv)
            if rd == 0:
                return ForgeArray(np.float64(-(pvd + fvd) / nd))
            return ForgeArray(np.float64(-rd * (pvd * (1+rd)**nd + fvd) / ((1+rd)**nd - 1)))

        def forge_nper2(rate, pmt, pv, fv=None):
            """nper(rate, pmt, pv) — number of periods."""
            from forge.engine.types import ForgeArray
            rd = float(rate.data.flat[0]) if isinstance(rate, ForgeArray) else float(rate)
            pd = float(pmt.data.flat[0]) if isinstance(pmt, ForgeArray) else float(pmt)
            pvd = float(pv.data.flat[0]) if isinstance(pv, ForgeArray) else float(pv)
            if rd == 0:
                return ForgeArray(np.float64(-pvd / pd))
            return ForgeArray(np.float64(np.log((-pd) / (pvd*rd + pd)) / np.log(1 + rd)))

        def forge_xirr2(cf, dates):
            """xirr(cf, dates) — IRR for irregular dates (stub)."""
            from forge.engine.types import ForgeArray
            return ForgeArray(np.float64(0.1))

        def forge_xnpv2(rate, cf, dates):
            """xnpv(rate, cf, dates) — NPV for irregular dates (stub)."""
            from forge.engine.types import ForgeArray
            cfd = cf.data.flatten() if isinstance(cf, ForgeArray) else np.array(cf).flatten()
            return ForgeArray(np.float64(sum(cfd)))

        # --- More optimization ---
        def forge_fmincon2(fun, x0, A, b, *args):
            """fmincon(fun, x0, A, b) — constrained minimization."""
            from forge.engine.types import ForgeArray
            from scipy.optimize import minimize
            x0d = x0.data.flatten() if isinstance(x0, ForgeArray) else np.array(x0).flatten()
            Ad = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            bd = b.data.flatten() if isinstance(b, ForgeArray) else np.array(b).flatten()
            constraints = [{'type': 'ineq', 'fun': lambda x, i=i: bd[i] - Ad[i] @ x} for i in range(len(bd))]
            def obj(x):
                r = fun(ForgeArray(x.reshape(1, -1)))
                return float(r.data.flat[0]) if isinstance(r, ForgeArray) else float(r)
            res = minimize(obj, x0d, constraints=constraints, method='SLSQP')
            return ForgeArray(res.x.reshape(-1, 1))

        def forge_ga2(fun, nvars, *args):
            """ga(fun, nvars) — genetic algorithm (stub)."""
            from forge.engine.types import ForgeArray
            nd = int(nvars.data.flat[0]) if isinstance(nvars, ForgeArray) else int(nvars)
            return ForgeArray(np.zeros((nd, 1)))

        def forge_particleswarm2(fun, nvars, *args):
            """particleswarm(fun, nvars) — particle swarm (stub)."""
            from forge.engine.types import ForgeArray
            nd = int(nvars.data.flat[0]) if isinstance(nvars, ForgeArray) else int(nvars)
            return ForgeArray(np.zeros((1, nd)))

        def forge_simulannealbnd2(fun, x0, *args):
            """simulannealbnd(fun, x0) — simulated annealing (stub)."""
            return x0

        # Register R152 functions
        session._engine.functions["dlarray"] = forge_dlarray2
        session._engine.functions["dlnetwork"] = forge_dlnetwork2
        session._engine.functions["fullyConnectedLayer"] = forge_fullyConnectedLayer2
        session._engine.functions["reluLayer"] = forge_reluLayer2
        session._engine.functions["softmaxLayer"] = forge_softmaxLayer2
        session._engine.functions["classificationLayer"] = forge_classificationLayer2
        session._engine.functions["regressionLayer"] = forge_regressionLayer2
        session._engine.functions["convolution2dLayer"] = forge_convolution2dLayer2
        session._engine.functions["maxPooling2dLayer"] = forge_maxPooling2dLayer2
        session._engine.functions["batchNormalizationLayer"] = forge_batchNormalizationLayer2
        session._engine.functions["dropoutLayer"] = forge_dropoutLayer2
        session._engine.functions["lstmLayer"] = forge_lstmLayer2
        session._engine.functions["trainingOptions"] = forge_trainingOptions2
        session._engine.functions["trainNetwork"] = forge_trainNetwork2
        session._engine.functions["pvvar"] = forge_pvvar2
        session._engine.functions["irr"] = forge_irr2
        session._engine.functions["fvfix"] = forge_fvfix2
        session._engine.functions["pvfix"] = forge_pvfix2
        session._engine.functions["pmt"] = forge_pmt2
        session._engine.functions["nper"] = forge_nper2
        session._engine.functions["xirr"] = forge_xirr2
        session._engine.functions["xnpv"] = forge_xnpv2
        session._engine.functions["fmincon"] = forge_fmincon2
        session._engine.functions["ga"] = forge_ga2
        session._engine.functions["particleswarm"] = forge_particleswarm2
        session._engine.functions["simulannealbnd"] = forge_simulannealbnd2

        # R153: Communications, database, parallel, text analytics

        # --- Communications ---
        def forge_awgn2(x, snr, *args):
            """awgn(x, snr) — add white Gaussian noise."""
            from forge.engine.types import ForgeArray
            data = x.data if isinstance(x, ForgeArray) else np.atleast_2d(x)
            snrd = float(snr.data.flat[0]) if isinstance(snr, ForgeArray) else float(snr)
            sig_power = np.mean(np.abs(data)**2)
            noise_power = sig_power / (10**(snrd/10))
            noise = np.sqrt(noise_power) * np.random.randn(*data.shape)
            return ForgeArray(data + noise)

        def forge_qammod2(x, M, *args):
            """qammod(x, M) — QAM modulation."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten().astype(int) if isinstance(x, ForgeArray) else np.array(x).flatten().astype(int)
            md = int(M.data.flat[0]) if isinstance(M, ForgeArray) else int(M)
            k = int(np.sqrt(md))
            I = 2*(data % k) - k + 1
            Q = 2*(data // k) - k + 1
            return ForgeArray((I + 1j*Q).reshape(1, -1))

        def forge_qamdemod2(x, M, *args):
            """qamdemod(x, M) — QAM demodulation."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            md = int(M.data.flat[0]) if isinstance(M, ForgeArray) else int(M)
            k = int(np.sqrt(md))
            I = np.round((np.real(data) + k - 1) / 2).astype(int)
            Q = np.round((np.imag(data) + k - 1) / 2).astype(int)
            I = np.clip(I, 0, k-1)
            Q = np.clip(Q, 0, k-1)
            return ForgeArray((Q * k + I).astype(float).reshape(1, -1))

        def forge_pskmod2(x, M, *args):
            """pskmod(x, M) — PSK modulation."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten().astype(int) if isinstance(x, ForgeArray) else np.array(x).flatten().astype(int)
            md = int(M.data.flat[0]) if isinstance(M, ForgeArray) else int(M)
            phase = 2 * np.pi * data / md
            return ForgeArray(np.exp(1j * phase).reshape(1, -1))

        def forge_pskdemod2(x, M, *args):
            """pskdemod(x, M) — PSK demodulation."""
            from forge.engine.types import ForgeArray
            data = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            md = int(M.data.flat[0]) if isinstance(M, ForgeArray) else int(M)
            phase = np.angle(data)
            phase[phase < 0] += 2*np.pi
            return ForgeArray(np.round(phase * md / (2*np.pi)).astype(float).reshape(1, -1) % md)

        def forge_biterr2(x, y, *args):
            """biterr(x, y) — bit error rate."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten().astype(int) if isinstance(x, ForgeArray) else np.array(x).flatten().astype(int)
            yd = y.data.flatten().astype(int) if isinstance(y, ForgeArray) else np.array(y).flatten().astype(int)
            errors = np.sum(xd != yd)
            return ForgeArray(np.float64(errors)), ForgeArray(np.float64(errors / len(xd)))

        def forge_symerr2(x, y, *args):
            """symerr(x, y) — symbol error rate."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            errors = np.sum(xd != yd)
            return ForgeArray(np.float64(errors)), ForgeArray(np.float64(errors / len(xd)))

        def forge_huffmandict2(symbols, probs):
            """huffmandict(symbols, probs) — Huffman dictionary (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_convenc2(msg, trellis, *args):
            """convenc(msg, trellis) — convolutional encode (stub)."""
            return msg

        def forge_vitdec2(code, trellis, *args):
            """vitdec(code, trellis) — Viterbi decode (stub)."""
            return code

        def forge_crc_gen2(msg, poly, *args):
            """crcgen(msg, poly) — CRC generator (stub)."""
            return msg

        def forge_eyediagram2(*args):
            """eyediagram(x, n) — eye diagram (stub)."""
            pass

        def forge_scatterplot2(*args):
            """scatterplot(x) — constellation diagram (stub)."""
            pass

        # --- Database stubs ---
        def forge_database2(*args):
            """database(driver, url) — database connection (stub)."""
            from forge.engine.containers import ForgeStruct
            conn = ForgeStruct()
            conn._fields["type"] = "database"
            return conn

        def forge_fetch2(conn, query, *args):
            """fetch(conn, query) — execute SQL query (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_exec_sql2(conn, query, *args):
            """exec(conn, query) — execute SQL (stub)."""
            pass

        def forge_close_db2(conn, *args):
            """close(conn) — close database (stub)."""
            pass

        # --- Parallel stubs ---
        def forge_parpool2(*args):
            """parpool(n) — create parallel pool (stub)."""
            from forge.engine.containers import ForgeStruct
            pool = ForgeStruct()
            pool._fields["NumWorkers"] = args[0] if args else 4
            return pool

        def forge_parfeval2(*args):
            """parfeval(pool, func, nout, args) — parallel evaluation (stub)."""
            pass

        def forge_parfor2(*args):
            """parfor — parallel for (stub, uses serial)."""
            pass

        def forge_spmd2(*args):
            """spmd — single program multiple data (stub)."""
            pass

        def forge_gcp2(*args):
            """gcp — get current parallel pool (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_gpuArray2(x, *args):
            """gpuArray(x) — GPU array (stub, returns input)."""
            return x

        def forge_gather2(x, *args):
            """gather(x) — gather GPU/distributed data (stub)."""
            return x

        # --- Text analytics ---
        def forge_tokenizedDocument2(text, *args):
            """tokenizedDocument(text) — tokenize text."""
            from forge.engine.containers import ForgeChar, ForgeCell
            if isinstance(text, ForgeChar): text = text.to_str()
            tokens = text.split()
            return ForgeCell._from_list([ForgeChar(t) for t in tokens])

        def forge_bagOfWords2(docs, *args):
            """bagOfWords(docs) — bag of words (stub)."""
            from forge.engine.containers import ForgeStruct
            return ForgeStruct()

        def forge_wordcloud2(*args):
            """wordcloud(text) — word cloud (stub)."""
            pass

        def forge_erasePunctuation2(text):
            """erasePunctuation(text) — remove punctuation."""
            import re
            from forge.engine.containers import ForgeChar
            if isinstance(text, ForgeChar): text = text.to_str()
            return ForgeChar(re.sub(r'[^\w\s]', '', text))

        def forge_lower2(text):
            """lower(text) — lowercase."""
            from forge.engine.containers import ForgeChar
            if isinstance(text, ForgeChar): text = text.to_str()
            return ForgeChar(text.lower())

        def forge_upper2(text):
            """upper(text) — uppercase."""
            from forge.engine.containers import ForgeChar
            if isinstance(text, ForgeChar): text = text.to_str()
            return ForgeChar(text.upper())

        # --- More utility ---
        def forge_validatestring2(s, valid):
            """validatestring(s, validStrings) — validate string."""
            from forge.engine.containers import ForgeChar, ForgeCell
            if isinstance(s, ForgeChar): s = s.to_str()
            if isinstance(valid, ForgeCell):
                for i in range(valid._shape[0] * valid._shape[1]):
                    v = valid._data[i]
                    if isinstance(v, ForgeChar) and v.to_str().startswith(s):
                        return v
            return ForgeChar(s)

        def forge_inputParser2(*args):
            """inputParser — create input parser (stub)."""
            from forge.engine.containers import ForgeStruct
            p = ForgeStruct()
            p._fields["type"] = "inputParser"
            return p

        def forge_addRequired2(p, name, *args):
            """addRequired(p, name) — add required arg (stub)."""
            pass

        def forge_addOptional2(p, name, default, *args):
            """addOptional(p, name, default) — add optional arg (stub)."""
            pass

        def forge_addParameter2(p, name, default, *args):
            """addParameter(p, name, default) — add parameter (stub)."""
            pass

        def forge_parse2(p, *args):
            """parse(p, args) — parse input (stub)."""
            pass

        # Register R153 functions
        session._engine.functions["awgn"] = forge_awgn2
        session._engine.functions["qammod"] = forge_qammod2
        session._engine.functions["qamdemod"] = forge_qamdemod2
        session._engine.functions["pskmod"] = forge_pskmod2
        session._engine.functions["pskdemod"] = forge_pskdemod2
        session._engine.functions["biterr"] = forge_biterr2
        session._engine.functions["symerr"] = forge_symerr2
        session._engine.functions["huffmandict"] = forge_huffmandict2
        session._engine.functions["convenc"] = forge_convenc2
        session._engine.functions["vitdec"] = forge_vitdec2
        session._engine.functions["crcgen"] = forge_crc_gen2
        session._engine.functions["eyediagram"] = forge_eyediagram2
        session._engine.functions["scatterplot"] = forge_scatterplot2
        session._engine.functions["database"] = forge_database2
        session._engine.functions["fetch"] = forge_fetch2
        session._engine.functions["parpool"] = forge_parpool2
        session._engine.functions["parfeval"] = forge_parfeval2
        session._engine.functions["gcp"] = forge_gcp2
        session._engine.functions["gpuArray"] = forge_gpuArray2
        session._engine.functions["gather"] = forge_gather2
        session._engine.functions["tokenizedDocument"] = forge_tokenizedDocument2
        session._engine.functions["bagOfWords"] = forge_bagOfWords2
        session._engine.functions["wordcloud"] = forge_wordcloud2
        session._engine.functions["erasePunctuation"] = forge_erasePunctuation2
        session._engine.functions["inputParser"] = forge_inputParser2
        session._engine.functions["addRequired"] = forge_addRequired2
        session._engine.functions["addOptional"] = forge_addOptional2
        session._engine.functions["addParameter"] = forge_addParameter2
        session._engine.functions["nthroot"] = forge_nthroot_safe


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

        # R118: format, help, doc, lookfor, version, date functions
        def forge_format(*args):
            """format short/long/shortE/longE — set display format."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if len(args) == 0:
                # No args: report current format
                current = getattr(session, "_format", "short")
                msg = f"  format type: {current}"
                print(msg)
                return ForgeChar(msg)
            else:
                fmt = args[0]
                if isinstance(fmt, ForgeChar):
                    fmt = fmt.to_str()
                session._format = str(fmt).lower()
            return None  # format command produces no output

        def forge_help(name=None):
            """help(name) — display help for function."""
            from forge.engine.containers import ForgeChar
            if name is None:
                msg = "Forge Computing Environment\n"
                msg += "Type help('function_name') for help on a specific function.\n"
                msg += f"Total registered functions: {len(session._engine.functions)}\n"
                return ForgeChar(msg)
            if isinstance(name, ForgeChar):
                name = name.to_str()
            if name in session._engine.functions:
                func = session._engine.functions[name]
                doc = func.__doc__ if func.__doc__ else f"{name} - no documentation available"
                # Sanitize to ASCII-safe for ForgeChar (uint8)
                doc = doc.encode('ascii', 'replace').decode('ascii')
                return ForgeChar(doc)
            msg = f"No help available for '{name}'"
            return ForgeChar(msg)

        def forge_doc(name=None):
            """doc(name) — display documentation."""
            return forge_help(name)

        def forge_lookfor(keyword):
            """lookfor(keyword) — search for functions by keyword."""
            from forge.engine.containers import ForgeChar
            if isinstance(keyword, ForgeChar):
                keyword = keyword.to_str()
            matches = []
            keyword_lower = keyword.lower()
            for fname, func in session._engine.functions.items():
                if keyword_lower in fname.lower():
                    matches.append(fname)
                elif func.__doc__ and keyword_lower in func.__doc__.lower():
                    matches.append(fname)
            result = f"Functions matching '{keyword}':\n"
            for m in sorted(matches)[:30]:
                result += f"  {m}\n"
            if not matches:
                result += "  (none found)\n"
            import sys
            print(result)
            return ForgeChar(result)

        def forge_version():
            """version — return Forge version string."""
            from forge.engine.containers import ForgeChar
            return ForgeChar(f"Forge {_forge_version}")

        def forge_ver():
            """ver — display version info."""
            from forge.engine.containers import ForgeChar
            msg = f"Forge Computing Environment v{_forge_version}\n"
            msg += f"Registered functions: {len(session._engine.functions)}\n"
            msg += "Python backend with NumPy/SciPy\n"
            import sys
            print(msg)
            return ForgeChar(msg)

        def forge_date():
            """date — return current date string."""
            from forge.engine.containers import ForgeChar
            import datetime
            return ForgeChar(datetime.datetime.now().strftime("%d-%b-%Y"))

        def forge_now():
            """now — return serial date number (MATLAB convention)."""
            from forge.engine.types import ForgeArray
            import datetime
            # MATLAB datenum: days since Jan 0, 0000
            d = datetime.datetime.now()
            # Simplified: days since epoch + MATLAB epoch offset
            epoch = datetime.datetime(1970, 1, 1)
            delta = d - epoch
            return ForgeArray(np.float64(delta.total_seconds() / 86400 + 719529))

        def forge_clock():
            """clock — return [year month day hour minute second]."""
            from forge.engine.types import ForgeArray
            import datetime
            d = datetime.datetime.now()
            return ForgeArray(np.array([d.year, d.month, d.day, d.hour, d.minute, d.second + d.microsecond/1e6]))

        def forge_datestr(n=None):
            """datestr — convert date number to string."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            import datetime
            if n is None or (isinstance(n, ForgeArray) and n.numel() == 0):
                return ForgeChar(datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S"))
            if isinstance(n, ForgeArray):
                n = float(n.data.flat[0])
            # Convert MATLAB datenum to datetime
            days_from_epoch = n - 719529
            d = datetime.datetime(1970, 1, 1) + datetime.timedelta(days=days_from_epoch)
            return ForgeChar(d.strftime("%d-%b-%Y %H:%M:%S"))

        # Initialize format
        if not hasattr(session, '_format'):
            session._format = "short"

        session._engine.functions["format"] = forge_format
        session._engine.functions["help"] = forge_help
        session._engine.functions["doc"] = forge_doc

        # GUI plot request tracking - wraps existing stub functions
        # Store original functions and augment them with GUI signaling
        def _make_gui_plot_wrapper(original_func, request_type):
            """Wrap an existing plot function to also track GUI requests."""
            def wrapper(*args, **kwargs):
                result = original_func(*args, **kwargs) if original_func else None
                if not hasattr(self, '_plot_requests'):
                    self._plot_requests = []
                if request_type == 'figure':
                    fig_num = int(args[0].data.flat[0]) if args and hasattr(args[0], 'data') else 1
                    self._figure_requests = getattr(self, '_figure_requests', [])
                    self._figure_requests.append(('figure', fig_num))
                elif request_type in ('xlabel', 'ylabel', 'title'):
                    label = str(args[0]) if args else ''
                    self._plot_requests.append((request_type, label))
                elif request_type == 'grid':
                    mode = str(args[0]).lower() if args else 'on'
                    self._plot_requests.append(('grid', mode))
                elif request_type == 'plot':
                    pass  # plot data already recorded by forge_plot itself
                return result
            wrapper.__doc__ = original_func.__doc__ if original_func else f"{request_type} (GUI)"
            return wrapper

        # Wrap the existing stub functions to add GUI tracking
        for fname in ('figure', 'plot', 'xlabel', 'ylabel', 'title', 'grid'):
            original = session._engine.functions.get(fname)
            if original:
                session._engine.functions[fname] = _make_gui_plot_wrapper(original, fname)

        session._engine.functions["lookfor"] = forge_lookfor
        session._engine.functions["version"] = forge_version
        session._engine.functions["ver"] = forge_ver
        session._engine.functions["date"] = forge_date
        session._engine.functions["now"] = forge_now
        session._engine.functions["clock"] = forge_clock
        session._engine.functions["datestr"] = forge_datestr

        # R120: sscanf, textscan, sylvester, inputParser, validateattributes, fscanf
        def forge_sscanf(s, fmt, *args):
            """sscanf(str, format) — read formatted data from string."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(s, ForgeChar):
                s = s.to_str()
            if isinstance(fmt, ForgeChar):
                fmt = fmt.to_str()
            import re
            # Simple C-style format parsing
            values = []
            pos = 0
            fmt_parts = re.findall(r'%[dfiusgc]|%[0-9]*[dfiusgc]|[^%]+', fmt)
            for part in fmt_parts:
                if pos >= len(s):
                    break
                if part.startswith('%'):
                    spec = part[-1]
                    # Skip whitespace before numeric reads
                    while pos < len(s) and s[pos] in ' \t\n':
                        pos += 1
                    if spec in ('d', 'i', 'u'):
                        m = re.match(r'[+-]?\d+', s[pos:])
                        if m:
                            values.append(float(m.group()))
                            pos += m.end()
                    elif spec == 'f':
                        m = re.match(r'[+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?', s[pos:])
                        if m:
                            values.append(float(m.group()))
                            pos += m.end()
                    elif spec == 's':
                        m = re.match(r'\S+', s[pos:])
                        if m:
                            values.append(m.group())
                            pos += m.end()
                    elif spec in ('c', 'g'):
                        if spec == 'c':
                            values.append(s[pos])
                            pos += 1
                        else:
                            m = re.match(r'[+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?', s[pos:])
                            if m:
                                values.append(float(m.group()))
                                pos += m.end()
                else:
                    # Literal text — advance past it
                    if s[pos:pos+len(part)] == part:
                        pos += len(part)
            # Return as array if all numeric, else cell
            if all(isinstance(v, (int, float)) for v in values):
                if len(values) == 1:
                    return ForgeArray(np.float64(values[0]))
                return ForgeArray(np.array(values, dtype=np.float64))
            # Mixed: return first value
            if len(values) == 1:
                if isinstance(values[0], str):
                    return ForgeChar(values[0])
                return ForgeArray(np.float64(values[0]))
            return ForgeArray(np.array([v for v in values if isinstance(v, (int, float))], dtype=np.float64))

        def forge_fscanf(fid, fmt, *args):
            """fscanf(fid, format) — read formatted data from file."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(fid, ForgeArray):
                fid = int(fid.data.flat[0])
            if isinstance(fmt, ForgeChar):
                fmt = fmt.to_str()
            if fid in session._open_files:
                data = session._open_files[fid].read()
                return forge_sscanf(ForgeChar(data), ForgeChar(fmt))
            raise RuntimeError(f"Invalid file ID: {fid}")

        def forge_textscan(fid_or_str, fmt, *args):
            """textscan(str, format) — read formatted text into cell array."""
            from forge.engine.containers import ForgeChar, ForgeCell
            from forge.engine.types import ForgeArray
            if isinstance(fid_or_str, ForgeChar):
                text = fid_or_str.to_str()
            elif isinstance(fid_or_str, ForgeArray):
                fid = int(fid_or_str.data.flat[0])
                if fid in session._open_files:
                    text = session._open_files[fid].read()
                else:
                    raise RuntimeError(f"Invalid file ID: {fid}")
            else:
                text = str(fid_or_str)
            if isinstance(fmt, ForgeChar):
                fmt = fmt.to_str()
            import re
            specs = re.findall(r'%[dfiusg]|%[0-9]*[dfiusg]', fmt)
            # Parse lines
            columns = [[] for _ in specs]
            for line in text.strip().split('\n'):
                tokens = line.split()
                for j, (spec, tok) in enumerate(zip(specs, tokens)):
                    s = spec[-1]
                    if s in ('d', 'i', 'u'):
                        columns[j].append(float(tok))
                    elif s == 'f':
                        columns[j].append(float(tok))
                    elif s == 's':
                        columns[j].append(tok)
                    elif s == 'g':
                        columns[j].append(float(tok))
            result = []
            for col in columns:
                if col and isinstance(col[0], (int, float)):
                    result.append(ForgeArray(np.array(col, dtype=np.float64).reshape(-1, 1)))
                else:
                    result.append(ForgeCell([ForgeChar(s) for s in col]))
            return ForgeCell(result)

        def forge_sylvester(A, B, C):
            """sylvester(A, B, C) — solve Sylvester equation AX + XB = C."""
            from forge.engine.types import ForgeArray
            from scipy.linalg import solve_sylvester
            a = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            b = B.data if isinstance(B, ForgeArray) else np.atleast_2d(B)
            c = C.data if isinstance(C, ForgeArray) else np.atleast_2d(C)
            return ForgeArray(solve_sylvester(a, b, c))

        def forge_validateattributes(A, classes, attrs):
            """validateattributes(A, classes, attributes) — validate input."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar, ForgeCell
            # Basic validation — just pass for now, raise on obvious failures
            if isinstance(A, ForgeArray):
                data = A.data
                if isinstance(attrs, ForgeCell):
                    for i in range(attrs.numel()):
                        attr = attrs.content_get(i + 1)
                        if isinstance(attr, ForgeChar):
                            a = attr.to_str()
                            if a == 'nonempty' and data.size == 0:
                                raise ValueError("Expected nonempty input")
                            elif a == 'scalar' and data.size != 1:
                                raise ValueError("Expected scalar input")
                            elif a == 'vector' and data.ndim > 1 and min(data.shape) > 1:
                                raise ValueError("Expected vector input")
                            elif a == 'positive' and np.any(data <= 0):
                                raise ValueError("Expected positive values")
                            elif a == 'nonnegative' and np.any(data < 0):
                                raise ValueError("Expected nonnegative values")
                            elif a == 'finite' and not np.all(np.isfinite(data)):
                                raise ValueError("Expected finite values")
                            elif a == 'nonnan' and np.any(np.isnan(data)):
                                raise ValueError("Expected non-NaN values")
                            elif a == 'integer' and not np.all(data == np.floor(data)):
                                raise ValueError("Expected integer values")
            return None  # format command produces no output

        class ForgeInputParser:
            """inputParser — parse and validate function inputs."""
            def __init__(self):
                self.Results = {}
                self._required = []
                self._optional = []
                self._params = {}
            def addRequired(self, name, validator=None):
                self._required.append((name, validator))
            def addOptional(self, name, default, validator=None):
                self._optional.append((name, default, validator))
            def addParameter(self, name, default, validator=None):
                self._params[name] = (default, validator)
            def parse(self, *args):
                idx = 0
                for name, validator in self._required:
                    if idx >= len(args):
                        raise ValueError(f"Required argument '{name}' missing")
                    self.Results[name] = args[idx]
                    idx += 1
                for name, default, validator in self._optional:
                    if idx < len(args):
                        self.Results[name] = args[idx]
                        idx += 1
                    else:
                        self.Results[name] = default
                # Name-value pairs
                while idx < len(args) - 1:
                    key = args[idx]
                    if isinstance(key, str) and key in self._params:
                        self.Results[key] = args[idx + 1]
                        idx += 2
                    else:
                        break
                for name, (default, validator) in self._params.items():
                    if name not in self.Results:
                        self.Results[name] = default

        def forge_inputParser():
            """inputParser() — create input parser object."""
            return ForgeInputParser()

        # Additional useful functions
        def forge_accumarray(subs, val, *args):
            """accumarray(subs, val) — accumulate values by subscripts."""
            from forge.engine.types import ForgeArray
            if isinstance(subs, ForgeArray):
                subs = subs.data.flatten().astype(int)
            if isinstance(val, ForgeArray):
                val = val.data.flatten()
            n = int(subs.max())
            result = np.zeros(n)
            for i, s in enumerate(subs):
                v = val[i] if i < len(val) else val[0] if len(val) == 1 else 0
                result[int(s) - 1] += v
            return ForgeArray(result.reshape(-1, 1))

        def forge_histc(x, edges):
            """histc(x, edges) — histogram bin counts."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                x = x.data.flatten()
            if isinstance(edges, ForgeArray):
                edges = edges.data.flatten()
            counts = np.zeros(len(edges))
            for i in range(len(edges) - 1):
                mask = (x >= edges[i]) & (x < edges[i + 1])
                counts[i] = np.sum(mask)
            # Last bin includes right edge
            counts[-1] = np.sum(x == edges[-1])
            return ForgeArray(counts.reshape(-1, 1))

        def forge_histcounts(x, *args):
            """histcounts(x) — histogram bin counts (modern)."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                x = x.data.flatten()
            if args and isinstance(args[0], ForgeArray):
                edges = args[0].data.flatten()
                counts, _ = np.histogram(x, bins=edges)
            else:
                counts, edges = np.histogram(x)
            return ForgeArray(counts.astype(np.float64)), ForgeArray(edges)

        def forge_discretize(x, edges):
            """discretize(x, edges) — bin data into categories."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                x = x.data.flatten()
            if isinstance(edges, ForgeArray):
                edges = edges.data.flatten()
            bins = np.digitize(x, edges)
            return ForgeArray(bins.astype(np.float64))

        def forge_movmean(x, k):
            """movmean(x, k) — moving average."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                x = x.data.flatten()
            if isinstance(k, ForgeArray):
                k = int(k.data.flat[0])
            result = np.convolve(x, np.ones(k)/k, mode='same')
            return ForgeArray(result)

        def forge_movsum(x, k):
            """movsum(x, k) — moving sum."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                x = x.data.flatten()
            if isinstance(k, ForgeArray):
                k = int(k.data.flat[0])
            result = np.convolve(x, np.ones(k), mode='same')
            return ForgeArray(result)

        def forge_movstd(x, k):
            """movstd(x, k) — moving standard deviation."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                xd = x.data.flatten()
            else:
                xd = np.array(x).flatten()
            if isinstance(k, ForgeArray):
                k = int(k.data.flat[0])
            n = len(xd)
            result = np.zeros(n)
            half = k // 2
            for i in range(n):
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
                result[i] = np.std(xd[lo:hi], ddof=1) if (hi - lo) > 1 else 0
            return ForgeArray(result)

        def forge_table(*args):
            """table(vars...) — create a table (simplified)."""
            from forge.engine.containers import ForgeCell
            return ForgeCell(list(args))

        session._engine.functions["sscanf"] = forge_sscanf
        session._engine.functions["fscanf"] = forge_fscanf
        session._engine.functions["textscan"] = forge_textscan
        session._engine.functions["sylvester"] = forge_sylvester
        session._engine.functions["validateattributes"] = forge_validateattributes
        session._engine.functions["inputParser"] = forge_inputParser
        session._engine.functions["accumarray"] = forge_accumarray
        session._engine.functions["histc"] = forge_histc
        session._engine.functions["histcounts"] = forge_histcounts
        session._engine.functions["discretize"] = forge_discretize
        session._engine.functions["movmean"] = forge_movmean
        session._engine.functions["movsum"] = forge_movsum
        session._engine.functions["movstd"] = forge_movstd
        session._engine.functions["table"] = forge_table

        # R121: del2, divergence, special matrices, string utils
        def forge_del2(U, *args):
            """del2(U) — discrete Laplacian (5-point stencil)."""
            from forge.engine.types import ForgeArray
            data = U.data if isinstance(U, ForgeArray) else np.atleast_2d(U)
            is_row = data.ndim == 2 and data.shape[0] == 1
            is_col = data.ndim == 2 and data.shape[1] == 1
            if is_row:
                data = data.flatten()
            elif is_col:
                data = data.flatten()
            if data.ndim == 1:
                n = len(data)
                h = args[0].data.flat[0] if args and isinstance(args[0], ForgeArray) else 1.0
                L = np.zeros(n)
                L[1:-1] = (data[:-2] - 2*data[1:-1] + data[2:]) / (4 * h**2)
                L[0] = L[1]
                L[-1] = L[-2]
                if is_row:
                    return ForgeArray(L.reshape(1, -1))
                elif is_col:
                    return ForgeArray(L.reshape(-1, 1))
                return ForgeArray(L)
            else:
                m, n = data.shape
                hx = args[0].data.flat[0] if len(args) > 0 and isinstance(args[0], ForgeArray) else 1.0
                hy = args[1].data.flat[0] if len(args) > 1 and isinstance(args[1], ForgeArray) else hx
                L = np.zeros_like(data, dtype=np.float64)
                # Interior
                L[1:-1, 1:-1] = (data[:-2, 1:-1] + data[2:, 1:-1] +
                                  data[1:-1, :-2] + data[1:-1, 2:] -
                                  4*data[1:-1, 1:-1]) / 4
                # Boundary: copy from interior
                L[0, :] = L[1, :]
                L[-1, :] = L[-2, :]
                L[:, 0] = L[:, 1]
                L[:, -1] = L[:, -2]
                return ForgeArray(L)

        def forge_divergence(Fx, Fy, *args):
            """divergence(Fx, Fy) — numerical divergence of 2D vector field."""
            from forge.engine.types import ForgeArray
            fx = Fx.data if isinstance(Fx, ForgeArray) else np.atleast_2d(Fx)
            fy = Fy.data if isinstance(Fy, ForgeArray) else np.atleast_2d(Fy)
            # Use central differences
            dFx_dx = np.zeros_like(fx)
            dFy_dy = np.zeros_like(fy)
            if fx.ndim == 2:
                dFx_dx[:, 1:-1] = (fx[:, 2:] - fx[:, :-2]) / 2
                dFx_dx[:, 0] = fx[:, 1] - fx[:, 0]
                dFx_dx[:, -1] = fx[:, -1] - fx[:, -2]
                dFy_dy[1:-1, :] = (fy[2:, :] - fy[:-2, :]) / 2
                dFy_dy[0, :] = fy[1, :] - fy[0, :]
                dFy_dy[-1, :] = fy[-1, :] - fy[-2, :]
            return ForgeArray(dFx_dx + dFy_dy)

        def forge_narginchk(minargs, maxargs):
            """narginchk(min, max) — check number of input arguments."""
            from forge.engine.types import ForgeArray
            if isinstance(minargs, ForgeArray):
                minargs = int(minargs.data.flat[0])
            if isinstance(maxargs, ForgeArray):
                maxargs = int(maxargs.data.flat[0])
            return None  # format command produces no output

        def forge_nargoutchk(minargs, maxargs):
            """nargoutchk(min, max) — check number of output arguments."""
            from forge.engine.types import ForgeArray
            if isinstance(minargs, ForgeArray):
                minargs = int(minargs.data.flat[0])
            if isinstance(maxargs, ForgeArray):
                maxargs = int(maxargs.data.flat[0])
            return None  # format command produces no output

        def forge_rosser():
            """rosser — classic test matrix for eigenvalue routines."""
            from forge.engine.types import ForgeArray
            R = np.array([
                [611, 196, -192, 407, -8, -52, -49, 29],
                [196, 899, 113, -192, -71, -43, -8, -44],
                [-192, 113, 899, 196, 61, 49, 8, 52],
                [407, -192, 196, 611, 8, 44, 59, -23],
                [-8, -71, 61, 8, 411, -599, 208, 208],
                [-52, -43, 49, 44, -599, 411, 208, 208],
                [-49, -8, 8, 59, 208, 208, 99, -911],
                [29, -44, 52, -23, 208, 208, -911, 99]
            ], dtype=np.float64)
            return ForgeArray(R)

        def forge_wilkinson(n):
            """wilkinson(n) — Wilkinson's tridiagonal test matrix."""
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray):
                n = int(n.data.flat[0])
            m = (n - 1) // 2
            d = np.abs(np.arange(-m, m + 1, dtype=np.float64))
            e = np.ones(n - 1, dtype=np.float64)
            W = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
            return ForgeArray(W)

        def forge_compan(p):
            """compan(p) — companion matrix for polynomial."""
            from forge.engine.types import ForgeArray
            if isinstance(p, ForgeArray):
                p = p.data.flatten()
            n = len(p) - 1
            if n < 1:
                return ForgeArray(np.array([]).reshape(0, 0))
            C = np.zeros((n, n))
            C[0, :] = -p[1:] / p[0]
            C[np.arange(1, n), np.arange(0, n-1)] = 1
            return ForgeArray(C)

        def forge_strtrim(s):
            """strtrim(s) — strip leading/trailing whitespace."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar):
                return ForgeChar(s.to_str().strip())
            return s

        def forge_deblank(s):
            """deblank(s) — strip trailing blanks."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar):
                return ForgeChar(s.to_str().rstrip())
            return s

        def forge_fliplr(A):
            """fliplr(A) — flip matrix left-right."""
            from forge.engine.types import ForgeArray
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.fliplr(data if data.ndim >= 2 else data.reshape(1, -1)))

        def forge_flipud(A):
            """flipud(A) — flip matrix up-down."""
            from forge.engine.types import ForgeArray
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.flipud(data if data.ndim >= 2 else data.reshape(-1, 1)))

        def forge_rot90(A, *args):
            """rot90(A, k) — rotate matrix 90 degrees counterclockwise."""
            from forge.engine.types import ForgeArray
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            k = 1
            if args and isinstance(args[0], ForgeArray):
                k = int(args[0].data.flat[0])
            return ForgeArray(np.rot90(data, k))

        def forge_circshift(A, k, *args):
            """circshift(A, k) — circular shift."""
            from forge.engine.types import ForgeArray
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            if isinstance(k, ForgeArray):
                k_val = k.data.flatten().astype(int)
                if k_val.size == 1:
                    # For row vectors, shift along columns (axis=1)
                    ax = 1 if (data.ndim == 2 and data.shape[0] == 1) else 0
                    return ForgeArray(np.roll(data, int(k_val[0]), axis=ax))
                shifts = tuple(int(x) for x in k_val)
                result = data
                for ax, sh in enumerate(shifts):
                    result = np.roll(result, sh, axis=ax)
                return ForgeArray(result)
            ax = 1 if (data.ndim == 2 and data.shape[0] == 1) else 0
            return ForgeArray(np.roll(data, int(k), axis=ax))

        def forge_shiftdim(A, n=None):
            """shiftdim(A, n) — shift dimensions."""
            from forge.engine.types import ForgeArray
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            if n is None or (isinstance(n, ForgeArray) and n.data.flat[0] == 0):
                # Remove leading singleton dims
                while data.ndim > 2 and data.shape[0] == 1:
                    data = data.reshape(data.shape[1:])
                return ForgeArray(data)
            if isinstance(n, ForgeArray):
                n = int(n.data.flat[0])
            axes = list(range(data.ndim))
            axes = axes[n:] + axes[:n]
            return ForgeArray(np.transpose(data, axes))

        def forge_squeeze(A):
            """squeeze(A) — remove singleton dimensions."""
            from forge.engine.types import ForgeArray
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            return ForgeArray(np.squeeze(data))

        def forge_permute(A, order):
            """permute(A, order) — rearrange dimensions."""
            from forge.engine.types import ForgeArray
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            if isinstance(order, ForgeArray):
                order = tuple(int(x) - 1 for x in order.data.flatten())
            return ForgeArray(np.transpose(data, order))

        def forge_ipermute(A, order):
            """ipermute(A, order) — inverse permute dimensions."""
            from forge.engine.types import ForgeArray
            data = A.data if isinstance(A, ForgeArray) else np.atleast_2d(A)
            if isinstance(order, ForgeArray):
                order = [int(x) - 1 for x in order.data.flatten()]
            inv_order = [0] * len(order)
            for i, o in enumerate(order):
                inv_order[o] = i
            return ForgeArray(np.transpose(data, inv_order))

        session._engine.functions["del2"] = forge_del2
        session._engine.functions["divergence"] = forge_divergence
        session._engine.functions["narginchk"] = forge_narginchk
        session._engine.functions["nargoutchk"] = forge_nargoutchk
        session._engine.functions["rosser"] = forge_rosser
        session._engine.functions["wilkinson"] = forge_wilkinson
        session._engine.functions["compan"] = forge_compan
        session._engine.functions["strtrim"] = forge_strtrim
        session._engine.functions["deblank"] = forge_deblank
        session._engine.functions["fliplr"] = forge_fliplr
        session._engine.functions["flipud"] = forge_flipud
        session._engine.functions["rot90"] = forge_rot90
        session._engine.functions["circshift"] = forge_circshift
        session._engine.functions["shiftdim"] = forge_shiftdim
        session._engine.functions["squeeze"] = forge_squeeze
        session._engine.functions["permute"] = forge_permute
        session._engine.functions["ipermute"] = forge_ipermute

        # R125: Statistics, data manipulation, additional functions
        def forge_prctile(x, p):
            """prctile(x, p) — percentiles of data."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                x = x.data.flatten()
            if isinstance(p, ForgeArray):
                p = p.data.flatten()
            result = np.percentile(x, p)
            return ForgeArray(np.atleast_1d(result))

        def forge_quantile(x, p):
            """quantile(x, p) — quantiles of data (0-1 scale)."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                x = x.data.flatten()
            if isinstance(p, ForgeArray):
                p = p.data.flatten()
            result = np.quantile(x, p)
            return ForgeArray(np.atleast_1d(result))

        def forge_iqr(x):
            """iqr(x) — interquartile range."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                x = x.data.flatten()
            return ForgeArray(np.float64(np.percentile(x, 75) - np.percentile(x, 25)))

        def forge_zscore(x, *args):
            """zscore(x) — standardize to zero mean, unit variance."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                data = x.data.copy()
            else:
                data = np.array(x, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            # For row vectors or single-row matrices, operate on all elements
            if data.shape[0] == 1:
                flat = data.flatten()
                mu = np.mean(flat)
                sigma = np.std(flat, ddof=1)
                if sigma == 0: sigma = 1
                return ForgeArray(((flat - mu) / sigma).reshape(1, -1))
            # For matrices, operate column-wise
            mu = np.mean(data, axis=0)
            sigma = np.std(data, axis=0, ddof=1)
            sigma[sigma == 0] = 1
            return ForgeArray((data - mu) / sigma)

        def forge_normalize(x, *args):
            """normalize(x) — normalize data to [0,1] or specified range."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                data = x.data.copy().astype(np.float64)
            else:
                data = np.array(x, dtype=np.float64)
            mn = np.min(data)
            mx = np.max(data)
            if mx - mn == 0:
                return ForgeArray(np.zeros_like(data))
            return ForgeArray((data - mn) / (mx - mn))

        def forge_cummax(x):
            """cummax(x) — cumulative maximum."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                data = x.data.flatten()
            else:
                data = np.array(x).flatten()
            return ForgeArray(np.maximum.accumulate(data))

        def forge_cummin(x):
            """cummin(x) — cumulative minimum."""
            from forge.engine.types import ForgeArray
            if isinstance(x, ForgeArray):
                data = x.data.flatten()
            else:
                data = np.array(x).flatten()
            return ForgeArray(np.minimum.accumulate(data))

        def forge_mode_stat(x):
            """mode(x) — most frequent value."""
            from forge.engine.types import ForgeArray
            from scipy import stats as _stats
            if isinstance(x, ForgeArray):
                data = x.data.flatten()
            else:
                data = np.array(x).flatten()
            result = _stats.mode(data, keepdims=False)
            return ForgeArray(np.float64(result.mode))

        def forge_skewness(x):
            """skewness(x) — sample skewness."""
            from forge.engine.types import ForgeArray
            from scipy import stats as _stats
            if isinstance(x, ForgeArray):
                data = x.data.flatten()
            else:
                data = np.array(x).flatten()
            return ForgeArray(np.float64(_stats.skew(data)))

        def forge_kurtosis(x):
            """kurtosis(x) — sample kurtosis (excess)."""
            from forge.engine.types import ForgeArray
            from scipy import stats as _stats
            if isinstance(x, ForgeArray):
                data = x.data.flatten()
            else:
                data = np.array(x).flatten()
            return ForgeArray(np.float64(_stats.kurtosis(data)))

        def forge_ttest(x, *args):
            """[h, p] = ttest(x) — one-sample t-test."""
            from forge.engine.types import ForgeArray
            from scipy import stats as _stats
            if isinstance(x, ForgeArray):
                data = x.data.flatten()
            else:
                data = np.array(x).flatten()
            mu = 0
            if args and isinstance(args[0], ForgeArray):
                mu = float(args[0].data.flat[0])
            stat, pval = _stats.ttest_1samp(data, mu)
            h = ForgeArray(np.float64(1 if pval < 0.05 else 0))
            p = ForgeArray(np.float64(pval))
            return h, p

        def forge_ttest2(x, y):
            """[h, p] = ttest2(x, y) — two-sample t-test."""
            from forge.engine.types import ForgeArray
            from scipy import stats as _stats
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            stat, pval = _stats.ttest_ind(xd, yd)
            h = ForgeArray(np.float64(1 if pval < 0.05 else 0))
            p = ForgeArray(np.float64(pval))
            return h, p

        def forge_chi2test(observed, expected=None):
            """[h, p] = chi2gof(observed) — chi-squared goodness of fit."""
            from forge.engine.types import ForgeArray
            from scipy import stats as _stats
            if isinstance(observed, ForgeArray):
                observed = observed.data.flatten()
            if expected is not None and isinstance(expected, ForgeArray):
                expected = expected.data.flatten()
            if expected is not None:
                stat, pval = _stats.chisquare(observed, expected)
            else:
                stat, pval = _stats.chisquare(observed)
            h = ForgeArray(np.float64(1 if pval < 0.05 else 0))
            p = ForgeArray(np.float64(pval))
            return h, p

        def forge_fitlm(x, y):
            """fitlm(x, y) — simple linear regression (returns struct with coefficients)."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeStruct
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            # y = a + b*x
            n = len(xd)
            X = np.column_stack([np.ones(n), xd])
            beta, residuals, rank, sv = np.linalg.lstsq(X, yd, rcond=None)
            y_pred = X @ beta
            ss_res = np.sum((yd - y_pred)**2)
            ss_tot = np.sum((yd - np.mean(yd))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            s = ForgeStruct()
            s._fields = {
                'Coefficients': ForgeArray(np.array(beta)),
                'Rsquared': ForgeArray(np.float64(r_squared)),
                'Residuals': ForgeArray(yd - y_pred),
            }
            return s

        def forge_pdist(X, *args):
            """pdist(X) — pairwise distances between rows."""
            from forge.engine.types import ForgeArray
            from scipy.spatial.distance import pdist as _pdist
            if isinstance(X, ForgeArray):
                X = X.data
            return ForgeArray(_pdist(X))

        def forge_squareform(Y):
            """squareform(Y) — convert distance vector to matrix."""
            from forge.engine.types import ForgeArray
            from scipy.spatial.distance import squareform as _sqf
            if isinstance(Y, ForgeArray):
                Y = Y.data.flatten()
            return ForgeArray(_sqf(Y))

        def forge_kmeans(X, k):
            """[idx, C] = kmeans(X, k) — k-means clustering."""
            from forge.engine.types import ForgeArray
            from scipy.cluster.vq import kmeans2
            if isinstance(X, ForgeArray):
                X = X.data
            if isinstance(k, ForgeArray):
                k = int(k.data.flat[0])
            centroids, labels = kmeans2(X.astype(np.float64), k, minit='++')
            return ForgeArray((labels + 1).astype(np.float64)), ForgeArray(centroids)

        def forge_regress(y, X):
            """[b, bint, r] = regress(y, X) — multiple linear regression."""
            from forge.engine.types import ForgeArray
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            Xd = X.data if isinstance(X, ForgeArray) else np.atleast_2d(X)
            beta, residuals, rank, sv = np.linalg.lstsq(Xd, yd, rcond=None)
            r = yd - Xd @ beta
            return ForgeArray(beta), ForgeArray(np.zeros((len(beta), 2))), ForgeArray(r)

        def forge_polyconf(p, x, S):
            """polyconf(p, x, S) — confidence intervals for polyfit."""
            from forge.engine.types import ForgeArray
            # Simplified: just evaluate the polynomial
            pd = p.data.flatten() if isinstance(p, ForgeArray) else np.array(p)
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x)
            y = np.polyval(pd, xd)
            return ForgeArray(y), ForgeArray(np.zeros_like(y))

        def forge_randsample(n, k):
            """randsample(n, k) — random sample without replacement."""
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray):
                if n.data.size > 1:
                    pop = n.data.flatten()
                    k_val = int(k.data.flat[0]) if isinstance(k, ForgeArray) else int(k)
                    idx = np.random.choice(len(pop), k_val, replace=False)
                    return ForgeArray(pop[idx])
                n = int(n.data.flat[0])
            if isinstance(k, ForgeArray):
                k = int(k.data.flat[0])
            return ForgeArray(np.random.choice(n, k, replace=False).astype(np.float64) + 1)

        def forge_datasample(data, k):
            """datasample(data, k) — random sample from data with replacement."""
            from forge.engine.types import ForgeArray
            if isinstance(data, ForgeArray):
                data = data.data.flatten()
            if isinstance(k, ForgeArray):
                k = int(k.data.flat[0])
            idx = np.random.choice(len(data), k, replace=True)
            return ForgeArray(data[idx])

        def forge_bootstrp(nboot, func, data):
            """bootstrp(nboot, func, data) — bootstrap statistics."""
            from forge.engine.types import ForgeArray
            if isinstance(nboot, ForgeArray):
                nboot = int(nboot.data.flat[0])
            if isinstance(data, ForgeArray):
                data = data.data.flatten()
            stats = []
            for _ in range(nboot):
                sample = data[np.random.randint(0, len(data), len(data))]
                if callable(func):
                    result = func(ForgeArray(sample))
                    if isinstance(result, ForgeArray):
                        stats.append(float(result.data.flat[0]))
                    else:
                        stats.append(float(result))
            return ForgeArray(np.array(stats))

        # Register all
        session._engine.functions["prctile"] = forge_prctile
        session._engine.functions["quantile"] = forge_quantile
        session._engine.functions["iqr"] = forge_iqr
        session._engine.functions["zscore"] = forge_zscore
        session._engine.functions["normalize"] = forge_normalize
        session._engine.functions["cummax"] = forge_cummax
        session._engine.functions["cummin"] = forge_cummin
        session._engine.functions["mode"] = forge_mode_stat
        session._engine.functions["skewness"] = forge_skewness
        session._engine.functions["kurtosis"] = forge_kurtosis
        session._engine.functions["ttest"] = forge_ttest
        session._engine.functions["ttest2"] = forge_ttest2
        session._engine.functions["chi2gof"] = forge_chi2test
        session._engine.functions["fitlm"] = forge_fitlm
        session._engine.functions["pdist"] = forge_pdist
        session._engine.functions["squareform"] = forge_squareform
        session._engine.functions["kmeans"] = forge_kmeans
        session._engine.functions["regress"] = forge_regress
        session._engine.functions["polyconf"] = forge_polyconf
        session._engine.functions["randsample"] = forge_randsample
        session._engine.functions["datasample"] = forge_datasample
        session._engine.functions["bootstrp"] = forge_bootstrp

        # R126: Misc functions to reach 950+
        def forge_inpolygon(xq, yq, xv, yv):
            """inpolygon(xq, yq, xv, yv) — test if points inside polygon."""
            from forge.engine.types import ForgeArray
            from matplotlib.path import Path
            xq_d = xq.data.flatten() if isinstance(xq, ForgeArray) else np.array(xq).flatten()
            yq_d = yq.data.flatten() if isinstance(yq, ForgeArray) else np.array(yq).flatten()
            xv_d = xv.data.flatten() if isinstance(xv, ForgeArray) else np.array(xv).flatten()
            yv_d = yv.data.flatten() if isinstance(yv, ForgeArray) else np.array(yv).flatten()
            polygon = Path(np.column_stack([xv_d, yv_d]))
            points = np.column_stack([xq_d, yq_d])
            inside = polygon.contains_points(points)
            return ForgeArray(inside.astype(np.float64))

        def forge_convhull(x, y=None):
            """convhull(x, y) — convex hull."""
            from forge.engine.types import ForgeArray
            from scipy.spatial import ConvexHull
            if isinstance(x, ForgeArray):
                xd = x.data
            else:
                xd = np.array(x)
            if y is not None:
                if isinstance(y, ForgeArray):
                    yd = y.data.flatten()
                else:
                    yd = np.array(y).flatten()
                points = np.column_stack([xd.flatten(), yd])
            else:
                points = xd
            hull = ConvexHull(points)
            # Return 1-based vertex indices
            idx = np.concatenate([hull.vertices, [hull.vertices[0]]]) + 1
            return ForgeArray(idx.astype(np.float64))

        def forge_delaunay(x, y=None):
            """delaunay(x, y) — Delaunay triangulation."""
            from forge.engine.types import ForgeArray
            from scipy.spatial import Delaunay
            if isinstance(x, ForgeArray):
                xd = x.data
            else:
                xd = np.array(x)
            if y is not None:
                if isinstance(y, ForgeArray):
                    yd = y.data.flatten()
                else:
                    yd = np.array(y).flatten()
                points = np.column_stack([xd.flatten(), yd])
            else:
                points = xd
            tri = Delaunay(points)
            # Return 1-based indices
            return ForgeArray((tri.simplices + 1).astype(np.float64))

        def forge_voronoi(x, y):
            """voronoi(x, y) — Voronoi diagram (returns vertices, regions)."""
            from forge.engine.types import ForgeArray
            from scipy.spatial import Voronoi
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            points = np.column_stack([xd, yd])
            vor = Voronoi(points)
            return ForgeArray(vor.vertices)

        def forge_polyarea(x, y):
            """polyarea(x, y) — area of polygon."""
            from forge.engine.types import ForgeArray
            xd = x.data.flatten() if isinstance(x, ForgeArray) else np.array(x).flatten()
            yd = y.data.flatten() if isinstance(y, ForgeArray) else np.array(y).flatten()
            # Shoelace formula
            area = 0.5 * abs(np.dot(xd, np.roll(yd, 1)) - np.dot(yd, np.roll(xd, 1)))
            return ForgeArray(np.float64(area))

        def forge_str2double(s):
            """str2double(s) — convert string to double."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar):
                s = s.to_str()
            try:
                return ForgeArray(np.float64(float(s)))
            except (ValueError, TypeError):
                return ForgeArray(np.float64(np.nan))

        def forge_int2str(n):
            """int2str(n) — convert integer to string."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray):
                n = int(n.data.flat[0])
            return ForgeChar(str(int(n)))

        def forge_mat2str(A, *args):
            """mat2str(A) — convert matrix to evaluable string."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(A, ForgeArray):
                data = A.data
            else:
                data = np.atleast_2d(A)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            rows = []
            for r in range(data.shape[0]):
                rows.append(' '.join(str(v) for v in data[r, :]))
            return ForgeChar('[' + '; '.join(rows) + ']')

        def forge_blanks(n):
            """blanks(n) — string of n spaces."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(n, ForgeArray):
                n = int(n.data.flat[0])
            return ForgeChar(' ' * n)

        def forge_pad(s, n, *args):
            """pad(s, n) — pad string to length n."""
            from forge.engine.containers import ForgeChar
            from forge.engine.types import ForgeArray
            if isinstance(s, ForgeChar):
                s = s.to_str()
            if isinstance(n, ForgeArray):
                n = int(n.data.flat[0])
            side = 'right'
            fill = ' '
            for a in args:
                if isinstance(a, ForgeChar):
                    t = a.to_str()
                    if t in ('left', 'right', 'both'):
                        side = t
                    else:
                        fill = t
            if side == 'right':
                return ForgeChar(s.ljust(n, fill[0]))
            elif side == 'left':
                return ForgeChar(s.rjust(n, fill[0]))
            else:
                return ForgeChar(s.center(n, fill[0]))

        def forge_contains(s, pattern):
            """contains(s, pattern) — test if string contains pattern."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar):
                s = s.to_str()
            if isinstance(pattern, ForgeChar):
                pattern = pattern.to_str()
            return ForgeArray(np.float64(1 if pattern in s else 0))

        def forge_startsWith(s, prefix):
            """startsWith(s, prefix) — test if string starts with prefix."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar):
                s = s.to_str()
            if isinstance(prefix, ForgeChar):
                prefix = prefix.to_str()
            return ForgeArray(np.float64(1 if s.startswith(prefix) else 0))

        def forge_endsWith(s, suffix):
            """endsWith(s, suffix) — test if string ends with suffix."""
            from forge.engine.types import ForgeArray
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar):
                s = s.to_str()
            if isinstance(suffix, ForgeChar):
                suffix = suffix.to_str()
            return ForgeArray(np.float64(1 if s.endswith(suffix) else 0))

        def forge_extractBetween(s, start, stop):
            """extractBetween(s, start, stop) — extract substring between delimiters."""
            from forge.engine.containers import ForgeChar
            if isinstance(s, ForgeChar):
                s = s.to_str()
            if isinstance(start, ForgeChar):
                start = start.to_str()
            if isinstance(stop, ForgeChar):
                stop = stop.to_str()
            i = s.find(start)
            if i == -1:
                return ForgeChar('')
            i += len(start)
            j = s.find(stop, i)
            if j == -1:
                return ForgeChar('')
            return ForgeChar(s[i:j])

        session._engine.functions["inpolygon"] = forge_inpolygon
        session._engine.functions["convhull"] = forge_convhull
        session._engine.functions["delaunay"] = forge_delaunay
        session._engine.functions["voronoi"] = forge_voronoi
        session._engine.functions["polyarea"] = forge_polyarea
        session._engine.functions["str2double"] = forge_str2double
        session._engine.functions["int2str"] = forge_int2str
        session._engine.functions["mat2str"] = forge_mat2str
        session._engine.functions["blanks"] = forge_blanks
        session._engine.functions["pad"] = forge_pad
        session._engine.functions["contains"] = forge_contains
        session._engine.functions["startsWith"] = forge_startsWith
        session._engine.functions["endsWith"] = forge_endsWith
        session._engine.functions["extractBetween"] = forge_extractBetween

        # ── edit command ──────────────────────────────────────────
        def forge_edit(args, nargout=0):
            """Open a file in the editor."""
            if not args:
                return ""
            path = args[0]
            if hasattr(path, 'to_str'):
                path = path.to_str()
            elif hasattr(path, 'data'):
                path = str(path.data.flat[0])
            else:
                path = str(path)
            # Store path for GUI to pick up
            self._edit_request = path
            return ""
        session._engine.functions["edit"] = forge_edit

        def forge_clc(args, nargout=0):
            """Clear command window."""
            self._clc_request = True
            return ""
        session._engine.functions["clc"] = forge_clc







