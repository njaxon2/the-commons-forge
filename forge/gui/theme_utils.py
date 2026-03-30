"""Shared theme detection utilities for Forge GUI widgets."""


def detect_palette():
    """Detect the current theme palette from the running QApplication.

    Returns a dict of color keys. Falls back to saved preference, then dark.
    """
    try:
        from forge.gui.themes import get_theme_palette, get_available_themes
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            for w in app.topLevelWidgets():
                if hasattr(w, "_current_theme"):
                    try:
                        return get_theme_palette(w._current_theme)
                    except Exception:
                        pass
            qss = app.styleSheet()
            if qss:
                for name in get_available_themes():
                    try:
                        p = get_theme_palette(name)
                        bg0 = p.get("bg0", "")
                        bg1 = p.get("bg1", "")
                        if bg0 and bg1 and bg0 in qss and bg1 in qss:
                            return p
                    except Exception:
                        pass
        return _palette_from_prefs()
    except Exception:
        return _palette_from_prefs()


def _palette_from_prefs():
    """Load palette from saved preference file, defaulting to dark."""
    try:
        import json, os
        from forge.gui.themes import get_theme_palette
        prefs_file = os.path.join(os.path.expanduser("~"), ".forge", "theme_prefs.json")
        if os.path.exists(prefs_file):
            with open(prefs_file, "r") as fh:
                prefs = json.load(fh)
            theme = prefs.get("default_theme", "dark")
            return get_theme_palette(theme)
    except Exception:
        pass
    try:
        from forge.gui.themes import get_theme_palette
        return get_theme_palette("dark")
    except Exception:
        return {
            "bg0": "#1e1e2e", "bg1": "#252536", "bg2": "#2a2a3c",
            "bg3": "#313145", "bg4": "#3a3a50", "bg5": "#44445a",
            "fg0": "#cdd6f4", "fg1": "#bac2de", "fg2": "#a6adc8",
            "fg3": "#6c7086",
            "border0": "#313145", "border1": "#44445a",
            "accent": "#00BCD4", "accent_h": "#18FFFF",
            "accent_p": "#0097A7", "accent_dim": "#006064",
            "accent_bg": "#002830",
            "error": "#f38ba8", "warning": "#fab387",
            "success": "#a6e3a1", "info": "#89b4fa",
            "selection": "#264f78", "cur_line": "#2a2a3c",
            "alt_row": "#282840", "tab_active": "#2a2a3c",
            "shadow": "rgba(0,0,0,0.25)",
        }


def is_light_theme():
    """Quick check: is the current theme light?"""
    p = detect_palette()
    bg0 = p.get("bg0", "#1e1e2e")
    try:
        r = int(bg0[1:3], 16)
        return r > 180
    except Exception:
        return False
