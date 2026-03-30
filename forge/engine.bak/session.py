# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Forge engine session — wraps the evaluator with builtins and session management."""
import os
import sys
import numpy as np
from forge.engine.evaluator import Session as _EvalSession, Workspace, ForgeError
from forge.engine.builtins import BUILTIN_REGISTRY
from forge.engine.types import ForgeArray, _unwrap


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

    # -- public API --------------------------------------------------------

    @property
    def workspace(self):
        return self._engine.workspace

    def eval(self, code):
        """Evaluate M-language code and return display string."""
        try:
            self.history.append(code)
            result = self._engine.eval(code)
            # Capture any printed output
            output = self._engine.output_buffer
            if hasattr(self._engine, 'output_buffer'):
                self._engine.output_buffer = ''
            if output:
                return output.rstrip('\n')
            if result is not None:
                return self._format_result(result)
            return ''
        except ForgeError as e:
            self.last_error = e
            return f'error: {e.identifier}: {e.message}'
        except Exception as e:
            self.last_error = e
            return f'error: {type(e).__name__}: {e}'

    def get_workspace_dict(self):
        """Return workspace variables as plain dict for GUI display."""
        ws = self._engine.workspace
        return {n: ws.get(n) for n in ws.names()}

    # -- formatting --------------------------------------------------------

    def _format_result(self, value):
        if isinstance(value, ForgeArray):
            data = _unwrap(value)
            if data.size == 0:
                return '  [](0x0)'
            if data.size == 1:
                return f'  {data.flat[0]}'
            if data.ndim <= 2 and data.size <= 200:
                return str(data)
            shape = 'x'.join(str(s) for s in data.shape)
            return f'  [{shape} {data.dtype}]'
        if isinstance(value, str):
            return value
        return str(value)

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
            lines = []
            ws = session._engine.workspace
            for name in sorted(ws.names()):
                val = ws.get(name)
                if isinstance(val, ForgeArray):
                    d = _unwrap(val)
                    shape = 'x'.join(str(s) for s in d.shape)
                    lines.append(f'  {name:20s} {shape:10s} {str(d.dtype):10s}')
                else:
                    lines.append(f'  {name:20s} {"1x1":10s} {type(val).__name__:10s}')
            return '\n'.join(lines)

        def forge_clear(*args):
            ws = session._engine.workspace
            if not args:
                for n in list(ws.names()):
                    ws.delete(n)
            else:
                for a in args:
                    ws.delete(str(a))

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
            return session._format_result(x)

        def forge_exist(name):
            n = str(name)
            ws = session._engine.workspace
            if ws.has(n):
                return ForgeArray(np.array(1.0))
            if n in session._engine.functions:
                return ForgeArray(np.array(5.0))
            return ForgeArray(np.array(0.0))

        def forge_class(x):
            if isinstance(x, ForgeArray):
                return str(_unwrap(x).dtype)
            return type(x).__name__

        for name, fn in [
            ('cd', forge_cd), ('pwd', forge_pwd), ('who', forge_who),
            ('whos', forge_whos), ('clear', forge_clear),
            ('addpath', forge_addpath), ('rmpath', forge_rmpath),
            ('path', forge_path_cmd), ('disp', forge_disp),
            ('exist', forge_exist), ('class', forge_class),
        ]:
            self._engine.functions[name] = fn
