# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Forge engine profiler — Octave-compatible profiling interface.

Usage (from M-language):
    profile('on')        — start profiling
    profile('off')       — stop profiling
    profile('resume')    — resume without clearing data
    profile('clear')     — clear accumulated data
    profile('status')    — return current status string
    profshow()           — display top functions by cumulative time
    profexport()         — export profile data as a dict
"""
import cProfile
import pstats
import io
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from forge.engine.types import ForgeArray


# ---------------------------------------------------------------------------
# Internal profiler state
# ---------------------------------------------------------------------------

@dataclass
class _ProfileEntry:
    """Accumulated stats for a single callable."""
    name: str
    ncalls: int = 0
    tottime: float = 0.0   # exclusive time
    cumtime: float = 0.0   # inclusive time


class ForgeProfiler:
    """Wraps cProfile to provide an Octave-style profiling API."""

    def __init__(self):
        self._profiler: Optional[cProfile.Profile] = None
        self._active: bool = False
        self._entries: Dict[str, _ProfileEntry] = {}
        self._start_wall: Optional[float] = None
        self._total_wall: float = 0.0

    # -- public API ---------------------------------------------------------

    def on(self):
        """Start (or restart) profiling, clearing previous data."""
        self.clear()
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        self._active = True
        self._start_wall = time.perf_counter()

    def off(self):
        """Stop profiling and collect stats."""
        if self._profiler is not None and self._active:
            self._profiler.disable()
            self._active = False
            if self._start_wall is not None:
                self._total_wall += time.perf_counter() - self._start_wall
                self._start_wall = None
            self._collect()

    def resume(self):
        """Resume profiling without clearing data."""
        if self._profiler is None:
            self._profiler = cProfile.Profile()
        self._profiler.enable()
        self._active = True
        self._start_wall = time.perf_counter()

    def clear(self):
        """Clear all accumulated profile data."""
        self._profiler = None
        self._active = False
        self._entries.clear()
        self._total_wall = 0.0
        self._start_wall = None

    def status(self) -> str:
        """Return a human-readable status string."""
        state = 'on' if self._active else 'off'
        n = len(self._entries)
        return f'profiler is {state}, {n} entries, {self._total_wall:.3f}s wall'

    def show(self, n: int = 20) -> str:
        """Return a formatted table of the top *n* functions by cumtime."""
        if not self._entries:
            return 'No profile data collected.'
        sorted_entries = sorted(
            self._entries.values(), key=lambda e: e.cumtime, reverse=True
        )[:n]
        lines = [
            f'  {"ncalls":>8s}  {"tottime":>10s}  {"cumtime":>10s}  {"function"}'
        ]
        lines.append('  ' + '-' * 60)
        for e in sorted_entries:
            lines.append(
                f'  {e.ncalls:8d}  {e.tottime:10.6f}  {e.cumtime:10.6f}  {e.name}'
            )
        return '\n'.join(lines)

    def export(self) -> dict:
        """Export profile data as a plain dict (struct in M-language)."""
        names = []
        ncalls = []
        tottime = []
        cumtime = []
        for e in sorted(self._entries.values(), key=lambda x: x.cumtime,
                        reverse=True):
            names.append(e.name)
            ncalls.append(e.ncalls)
            tottime.append(e.tottime)
            cumtime.append(e.cumtime)
        return {
            'FunctionName': names,
            'NumCalls': ForgeArray(np.array(ncalls, dtype=np.float64)),
            'TotalTime': ForgeArray(np.array(tottime, dtype=np.float64)),
            'CumulativeTime': ForgeArray(np.array(cumtime, dtype=np.float64)),
            'WallTime': self._total_wall,
        }

    # -- internals ----------------------------------------------------------

    def _collect(self):
        """Harvest stats from the cProfile.Profile object."""
        if self._profiler is None:
            return
        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        # stats.stats is a dict: (file, line, name) -> (ncalls, totcalls, tottime, cumtime, callers)
        for key, val in stats.stats.items():
            fname = key[2]  # function name
            nc = val[1]     # total calls
            tt = val[2]     # tottime
            ct = val[3]     # cumtime
            if fname in self._entries:
                entry = self._entries[fname]
                entry.ncalls += nc
                entry.tottime += tt
                entry.cumtime += ct
            else:
                self._entries[fname] = _ProfileEntry(
                    name=fname, ncalls=nc, tottime=tt, cumtime=ct
                )


# ---------------------------------------------------------------------------
# Singleton + registry
# ---------------------------------------------------------------------------

_PROFILER = ForgeProfiler()


def forge_profile(*args):
    """Octave-compatible ``profile`` command dispatcher."""
    if not args:
        return _PROFILER.status()
    cmd = str(args[0]).lower()
    if cmd == 'on':
        _PROFILER.on()
    elif cmd == 'off':
        _PROFILER.off()
    elif cmd == 'resume':
        _PROFILER.resume()
    elif cmd == 'clear':
        _PROFILER.clear()
    elif cmd == 'status':
        return _PROFILER.status()
    else:
        raise ValueError(f"profile: unknown command '{cmd}'")


def forge_profshow(*args):
    """Display top profiled functions."""
    n = 20
    if args:
        n = int(args[0])
    return _PROFILER.show(n)


def forge_profexport():
    """Export profile data as a struct/dict."""
    return _PROFILER.export()


# Registry for bulk import into the session
PROFILER_REGISTRY: Dict[str, callable] = {
    'profile': forge_profile,
    'profshow': forge_profshow,
    'profexport': forge_profexport,
}
