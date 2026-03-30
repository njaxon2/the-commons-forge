"""Add-on manager for Forge.

Tracks which Forge toolboxes and Octave packages are enabled,
and provides the merged function table to the session.
"""

import json
import os

from forge.engine.builtins import TOOLBOX_MANIFEST


class AddonManager:
    """Central coordinator for Forge and Octave add-ons."""

    PREFS_FILE = os.path.join(os.path.expanduser("~"), ".forge", "addons.json")

    def __init__(self):
        self._octave_bridge = None
        self._octave_packages = {}       # name -> {"version", "path"}
        self._octave_pkg_functions = {}   # name -> [func_names]

        # Forge toolbox state: name -> enabled (bool)
        self._forge_state = {}
        # Octave package state: name -> enabled (bool)
        self._octave_state = {}

        # Callbacks for state changes
        self._on_change_callbacks = []

        self._init_forge_toolboxes()
        self._init_octave()
        self._load_prefs()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_forge_toolboxes(self):
        """Initialize Forge toolbox state — all enabled by default."""
        for name in TOOLBOX_MANIFEST:
            self._forge_state.setdefault(name, True)

    def _init_octave(self):
        """Detect Octave and enumerate installed packages."""
        from forge.engine.octave_bridge import OctaveBridge
        self._octave_bridge = OctaveBridge()
        if self._octave_bridge.available:
            for pkg in self._octave_bridge.list_packages():
                pname = pkg["name"]
                self._octave_packages[pname] = pkg
                self._octave_state.setdefault(pname, False)

    def _load_prefs(self):
        """Load saved enable/disable state."""
        if not os.path.exists(self.PREFS_FILE):
            return
        try:
            with open(self.PREFS_FILE, "r") as f:
                data = json.load(f)
            for name, enabled in data.get("forge", {}).items():
                if name in self._forge_state:
                    self._forge_state[name] = bool(enabled)
            for name, enabled in data.get("octave", {}).items():
                if name in self._octave_state:
                    self._octave_state[name] = bool(enabled)
        except Exception:
            pass

    def _save_prefs(self):
        """Persist current state."""
        os.makedirs(os.path.dirname(self.PREFS_FILE), exist_ok=True)
        data = {"forge": self._forge_state, "octave": self._octave_state}
        try:
            with open(self.PREFS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def octave_available(self):
        return self._octave_bridge is not None and self._octave_bridge.available

    @property
    def octave_bridge(self):
        return self._octave_bridge

    def on_change(self, callback):
        """Register a callback for addon state changes."""
        self._on_change_callbacks.append(callback)

    # -- Forge toolboxes --

    def forge_toolboxes(self):
        """Return list of (name, display_name, func_count, enabled)."""
        result = []
        for name, (display, registry) in TOOLBOX_MANIFEST.items():
            result.append((name, display, len(registry), self._forge_state.get(name, True)))
        return result

    def is_forge_enabled(self, name):
        return self._forge_state.get(name, True)

    def set_forge_enabled(self, name, enabled):
        if name not in self._forge_state:
            return
        self._forge_state[name] = enabled
        self._save_prefs()
        self._notify()

    # -- Octave packages --

    def octave_packages(self):
        """Return list of (name, version, enabled)."""
        result = []
        for name, info in self._octave_packages.items():
            result.append((name, info.get("version", "?"), self._octave_state.get(name, False)))
        return result

    def is_octave_enabled(self, name):
        return self._octave_state.get(name, False)

    def set_octave_enabled(self, name, enabled):
        if name not in self._octave_state:
            return
        self._octave_state[name] = enabled
        if enabled:
            self._ensure_octave_functions(name)
            self._octave_bridge.load_package(name)
        else:
            self._octave_bridge.unload_package(name)
        self._save_prefs()
        self._notify()

    # -- Function table --

    def get_forge_functions(self):
        """Return dict of {name: callable} for all enabled Forge toolboxes."""
        funcs = {}
        for tb_name, (_, registry) in TOOLBOX_MANIFEST.items():
            if self._forge_state.get(tb_name, True):
                funcs.update(registry)
        return funcs

    def get_octave_functions(self):
        """Return dict of {name: callable_proxy} for all enabled Octave packages."""
        funcs = {}
        for pkg_name in self._octave_state:
            if not self._octave_state[pkg_name]:
                continue
            self._ensure_octave_functions(pkg_name)
            for func_name in self._octave_pkg_functions.get(pkg_name, []):
                # Skip if Forge already provides it (Forge takes priority)
                funcs[func_name] = self._make_octave_proxy(func_name)
        return funcs

    def get_all_active_functions(self):
        """Return merged function dict. Forge functions take priority."""
        octave_funcs = self.get_octave_functions()
        forge_funcs = self.get_forge_functions()
        # Octave first, then Forge overwrites — Forge wins on conflicts
        merged = {}
        merged.update(octave_funcs)
        merged.update(forge_funcs)
        return merged

    def get_conflicts(self):
        """Return set of function names provided by both Forge and Octave."""
        forge_names = set()
        for tb_name, (_, registry) in TOOLBOX_MANIFEST.items():
            if self._forge_state.get(tb_name, True):
                forge_names.update(registry.keys())
        octave_names = set()
        for pkg_name in self._octave_state:
            if self._octave_state[pkg_name]:
                octave_names.update(self._octave_pkg_functions.get(pkg_name, []))
        return forge_names & octave_names

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_octave_functions(self, pkg_name):
        """Lazily discover functions provided by an Octave package."""
        if pkg_name in self._octave_pkg_functions:
            return
        if self._octave_bridge is not None:
            funcs = self._octave_bridge.list_package_functions(pkg_name)
            self._octave_pkg_functions[pkg_name] = funcs

    def _make_octave_proxy(self, func_name):
        """Create a callable proxy that routes to octave-cli."""
        bridge = self._octave_bridge

        def proxy(*args):
            return bridge.call_function(func_name, *args)
        proxy.__name__ = func_name
        proxy.__doc__ = f"[Octave] {func_name} — executed via octave-cli"
        proxy._is_octave_proxy = True
        return proxy

    def _notify(self):
        for cb in self._on_change_callbacks:
            try:
                cb()
            except Exception:
                pass

    def shutdown(self):
        """Clean up Octave subprocess."""
        if self._octave_bridge is not None:
            self._octave_bridge.stop()
