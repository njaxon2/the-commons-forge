# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Octave subprocess bridge for Forge.

Manages a persistent octave-cli process and marshals data between
the Forge engine (Python/NumPy) and GNU Octave.
"""

import shutil
import subprocess
import threading
import re
import numpy as np

from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar


class OctaveBridge:
    """Communicate with a persistent octave-cli subprocess."""

    _SENTINEL = "__FORGE_SENTINEL_89a7c__"

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()
        self._loaded_packages = set()
        self._octave_path = shutil.which("octave-cli")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def available(self):
        return self._octave_path is not None

    def start(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.available:
            raise RuntimeError("octave-cli not found on PATH")
        self._proc = subprocess.Popen(
            [self._octave_path, "--no-gui", "--interactive", "--norc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Read the startup banner
        self._raw_eval("format long;")

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write("exit\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None
        self._loaded_packages.clear()

    def ensure_running(self):
        if self._proc is None or self._proc.poll() is not None:
            self.start()

    # ------------------------------------------------------------------
    # Package management
    # ------------------------------------------------------------------

    def load_package(self, name):
        if name in self._loaded_packages:
            return
        self.ensure_running()
        self._raw_eval(f"pkg load {name};")
        self._loaded_packages.add(name)

    def unload_package(self, name):
        if name not in self._loaded_packages:
            return
        self.ensure_running()
        self._raw_eval(f"pkg unload {name};")
        self._loaded_packages.discard(name)

    def list_packages(self):
        """Return list of installed Octave packages as dicts."""
        if not self.available:
            return []
        try:
            r = subprocess.run(
                [self._octave_path, "--no-gui", "--eval", "pkg list"],
                capture_output=True, text=True, timeout=15,
            )
            return self._parse_pkg_list(r.stdout)
        except Exception:
            return []

    def list_package_functions(self, pkg_name):
        """Return list of function names provided by a package."""
        if not self.available:
            return []
        try:
            r = subprocess.run(
                [self._octave_path, "--no-gui", "--eval",
                 f"pkg load {pkg_name}; "
                 f"d = pkg('describe', '{pkg_name}'); "
                 f"for i = 1:numel(d{{1}}.provides), "
                 f"  disp(d{{1}}.provides{{i}}.function); "
                 f"end"],
                capture_output=True, text=True, timeout=15,
            )
            funcs = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
            return funcs
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Function calls
    # ------------------------------------------------------------------

    def call_function(self, name, *args, nargout=1):
        """Call an Octave function and return the result."""
        self.ensure_running()
        # Serialize arguments
        setup_lines = []
        arg_names = []
        for i, arg in enumerate(args):
            vname = f"__fa{i}__"
            arg_names.append(vname)
            setup_lines.append(f"{vname} = {self._to_octave(arg)};")

        arg_str = ", ".join(arg_names)
        if nargout == 0:
            call_expr = f"{name}({arg_str});"
        elif nargout == 1:
            call_expr = f"__fout__ = {name}({arg_str});"
        else:
            out_vars = [f"__fout{j}__" for j in range(nargout)]
            call_expr = f"[{', '.join(out_vars)}] = {name}({arg_str});"

        code = "\n".join(setup_lines) + "\n" + call_expr
        raw = self._raw_eval(code)

        if nargout == 0:
            return raw.strip() if raw.strip() else None
        elif nargout == 1:
            result_str = self._raw_eval("disp(__fout__);")
            return self._from_octave(result_str)
        else:
            results = []
            for vname in out_vars:
                result_str = self._raw_eval(f"disp({vname});")
                results.append(self._from_octave(result_str))
            return tuple(results)

    def eval_code(self, code):
        """Evaluate raw M-code in Octave and return text output."""
        self.ensure_running()
        return self._raw_eval(code)

    # ------------------------------------------------------------------
    # Internal communication
    # ------------------------------------------------------------------

    def _raw_eval(self, code):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self.start()
            p = self._proc
            # Send code followed by sentinel
            p.stdin.write(code + "\n")
            p.stdin.write(f'fprintf("{self._SENTINEL}\\n");\n')
            p.stdin.flush()

            lines = []
            while True:
                line = p.stdout.readline()
                if not line:
                    break
                if self._SENTINEL in line:
                    break
                lines.append(line)
            return "".join(lines)

    # ------------------------------------------------------------------
    # Data marshalling
    # ------------------------------------------------------------------

    def _to_octave(self, val):
        """Convert a Python/Forge value to Octave literal string."""
        if val is None:
            return "[]"
        v = _unwrap(val) if hasattr(val, '_data') else val
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                return str(v.item())
            if v.ndim == 1:
                return "[" + " ".join(str(x) for x in v.flat) + "]"
            # 2-D
            rows = []
            for r in range(v.shape[0]):
                rows.append(" ".join(str(v[r, c]) for c in range(v.shape[1])))
            return "[" + "; ".join(rows) + "]"
        if isinstance(v, (int, float, np.integer, np.floating)):
            return str(v)
        if isinstance(v, complex):
            return f"({v.real} + {v.imag}i)"
        if isinstance(v, str):
            escaped = v.replace("'", "''")
            return f"'{escaped}'"
        if isinstance(v, ForgeChar):
            escaped = str(v).replace("'", "''")
            return f"'{escaped}'"
        return str(v)

    def _from_octave(self, text):
        """Parse Octave text output back into a ForgeArray or scalar."""
        text = text.strip()
        if not text:
            return ForgeArray(np.array([]))

        # Try to parse as numeric matrix
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # Filter out variable name headers like "ans ="
        lines = [l for l in lines if not re.match(r'^\w+\s*=$', l)]

        try:
            rows = []
            for line in lines:
                # Split on whitespace, parse as floats
                vals = [float(x) for x in line.split()]
                if vals:
                    rows.append(vals)
            if rows:
                arr = np.array(rows)
                if arr.size == 1:
                    return ForgeArray(arr.ravel())
                return ForgeArray(arr.squeeze())
        except (ValueError, TypeError):
            pass

        # Fall back to string
        return ForgeChar(text)

    @staticmethod
    def _parse_pkg_list(output):
        """Parse 'pkg list' tabular output."""
        packages = []
        for line in output.split("\n"):
            line = line.strip()
            if not line or line.startswith("Package") or line.startswith("-"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                packages.append({
                    "name": parts[0],
                    "version": parts[1],
                    "path": parts[2],
                })
        return packages
