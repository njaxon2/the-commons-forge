# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Forge plotting function library — Octave-compatible wrappers around
matplotlib (forge/engine/builtins/plotting.py)."""

import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as _Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

from forge.engine.types import ForgeArray, _unwrap


def _enhance_current():
    # Enhance the current figure window with Forge tools (Probe, etc.).
    try:
        from forge_ide.gui.figure_enhancer import enhance_figure
        enhance_figure(plt.gcf())
    except Exception:
        pass


# Enable interactive mode so figures appear without blocking
plt.ion()

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_hold_state: bool = False


def _cur_fig():
    return plt.gcf()


def _cur_ax():
    return plt.gca()


def _maybe_clear():
    if not _hold_state:
        _cur_ax().cla()


def _to_np(x):
    arr = np.asarray(_unwrap(x), dtype=float)
    # Flatten row/column vectors to 1D for matplotlib compatibility
    # (matplotlib interprets 2D arrays as column-per-series)
    if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
        arr = arr.ravel()
    return arr


# ---------------------------------------------------------------------------
# Octave line-spec parser  (e.g. "r--o")
# ---------------------------------------------------------------------------

_COLOUR_MAP = {
    "r": "red", "g": "green", "b": "blue", "k": "black",
    "w": "white", "c": "cyan", "m": "magenta", "y": "yellow",
}
_LINE_MAP = {"-": "-", "--": "--", ":": ":", "-.": "-."}
_MARKER_MAP = {
    "o": "o", "+": "+", "*": "*", ".": ".", "x": "x",
    "s": "s", "d": "d", "^": "^", "v": "v", ">": ">", "<": "<",
    "p": "p", "h": "h",
}


def _parse_linespec(fmt: str) -> dict:
    """Parse an Octave-style linespec string into matplotlib kwargs."""
    kwargs: dict = {}
    if not fmt:
        return kwargs
    # colour
    for ch, name in _COLOUR_MAP.items():
        if ch in fmt:
            kwargs["color"] = name
            fmt = fmt.replace(ch, "", 1)
            break
    # line style (try longest first)
    for ls in ("--", "-.", ":", "-"):
        if ls in fmt:
            kwargs["linestyle"] = ls
            fmt = fmt.replace(ls, "", 1)
            break
    # marker
    for mk in _MARKER_MAP:
        if mk in fmt:
            kwargs["marker"] = mk
            break
    return kwargs


# ===================================================================
# 2-D plots
# ===================================================================

def _is_property_name(s):
    """Check if a string is a MATLAB plot property name (not a linespec)."""
    props = {'linewidth', 'markersize', 'color', 'marker', 'linestyle',
             'markeredgecolor', 'markerfacecolor', 'displayname'}
    return isinstance(s, str) and s.lower() in props

def _extract_plot_kwargs(a, start):
    """Extract key-value property pairs from args starting at index start."""
    kwargs = {}
    i = start
    while i + 1 < len(a):
        if not isinstance(a[i], str) or not _is_property_name(a[i]):
            break
        key = a[i].lower()
        val = a[i + 1]
        if key == 'linewidth':
            kwargs['linewidth'] = float(_to_np(val).flat[0]) if hasattr(val, '__array__') else float(val)
        elif key == 'markersize':
            kwargs['markersize'] = float(_to_np(val).flat[0]) if hasattr(val, '__array__') else float(val)
        elif key == 'color':
            kwargs['color'] = val
        elif key == 'displayname':
            kwargs['label'] = val if isinstance(val, str) else str(val)
        elif key == 'linestyle':
            kwargs['linestyle'] = val if isinstance(val, str) else str(val)
        elif key == 'marker':
            kwargs['marker'] = val if isinstance(val, str) else str(val)
        i += 2
    return kwargs, i

def forge_plot(*args):
    """plot(y), plot(x,y), plot(x,y,fmt), plot(x,y,fmt,'Prop',val,...), etc."""
    _maybe_clear()
    ax = _cur_ax()
    i = 0
    # Convert args, but preserve ForgeChar as strings (format specs like 'b-', 'r--')
    from forge.engine.containers import ForgeChar
    a = []
    for arg in args:
        if isinstance(arg, ForgeChar):
            a.append(arg.to_str())  # Convert to Python string
        elif hasattr(arg, '__array__') or isinstance(arg, ForgeArray):
            a.append(_unwrap(arg))
        else:
            a.append(arg)
    while i < len(a):
        # Determine x, y, fmt for this data group
        x_data = None
        y_data = None
        fmt_kwargs = {}

        if i < len(a) and not isinstance(a[i], str):
            # First arg is data
            if i + 1 < len(a) and not isinstance(a[i + 1], str):
                # plot(x, y, ...)
                x_data = _to_np(a[i])
                y_data = _to_np(a[i + 1])
                i += 2
            else:
                # plot(y, ...) or plot(y, fmt, ...)
                y_data = _to_np(a[i])
                i += 1
        else:
            break

        # Check for format string (short: 'b-', 'r--o', etc.)
        if i < len(a) and isinstance(a[i], str) and len(a[i]) <= 4 and not _is_property_name(a[i]):
            fmt_kwargs = _parse_linespec(a[i])
            i += 1

        # Check for property key-value pairs
        extra_kwargs, i = _extract_plot_kwargs(a, i)
        fmt_kwargs.update(extra_kwargs)

        # Plot
        if x_data is not None:
            ax.plot(x_data, y_data, **fmt_kwargs)
        else:
            ax.plot(y_data, **fmt_kwargs)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_semilogx(*args):
    """semilogx(x, y) — semi-log plot (x-axis logarithmic)."""
    _maybe_clear()
    forge_plot(*args)
    _cur_ax().set_xscale("log")
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_semilogy(*args):
    """semilogy(x, y) — semi-log plot (y-axis logarithmic)."""
    _maybe_clear()
    forge_plot(*args)
    _cur_ax().set_yscale("log")
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_loglog(*args):
    """loglog(x, y) — log-log scale plot."""
    _maybe_clear()
    forge_plot(*args)
    ax = _cur_ax()
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_bar(*args, **kwargs):
    """bar(y), bar(x,y), bar(x,y,width), bar(...,color), bar(...,'prop',val)."""
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray
    _maybe_clear()
    # Parse positional args
    pos = []
    kw = {}
    i = 0
    str_args = list(args)
    while i < len(str_args):
        a = str_args[i]
        if isinstance(a, ForgeChar) or (isinstance(a, str) and len(a) <= 2 and not a.replace('.','').isdigit()):
            # Color shorthand like "r", "b", "g"
            s = a.to_str() if isinstance(a, ForgeChar) else a
            if len(s) <= 2 and s.isalpha():
                kw["color"] = s
                i += 1
                continue
        # Check for keyword pair
        if isinstance(a, (str, ForgeChar)):
            s = a.to_str() if isinstance(a, ForgeChar) else a
            if s.lower() in ("facecolor", "edgecolor", "linewidth", "barwidth"):
                if i + 1 < len(str_args):
                    val = str_args[i+1]
                    if isinstance(val, ForgeChar):
                        val = val.to_str()
                    elif isinstance(val, ForgeArray):
                        val = float(val.data.flat[0])
                    kw[s.lower()] = val
                    i += 2
                    continue
        pos.append(a)
        i += 1
    # Now interpret positional args
    if len(pos) == 1:
        y = _to_np(pos[0]).ravel()
        x = np.arange(1, len(y) + 1)
        width = 0.8
    elif len(pos) == 2:
        x = _to_np(pos[0]).ravel()
        y = _to_np(pos[1]).ravel()
        width = 0.8
    elif len(pos) >= 3:
        x = _to_np(pos[0]).ravel()
        y = _to_np(pos[1]).ravel()
        w = pos[2]
        if isinstance(w, ForgeArray):
            width = float(w.data.flat[0])
        else:
            width = float(w)
        # 4th positional could be color
        if len(pos) >= 4:
            c = pos[3]
            if isinstance(c, ForgeChar):
                kw["color"] = c.to_str()
            elif isinstance(c, str):
                kw["color"] = c
    else:
        return
    bar_kw = {"width": width}
    if "color" in kw:
        bar_kw["color"] = kw["color"]
    if "edgecolor" in kw:
        bar_kw["edgecolor"] = kw["edgecolor"]
    _cur_ax().bar(x, y, **bar_kw)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_barh(x, y=None, height=0.8):
    """barh(y, data) — horizontal bar chart."""
    _maybe_clear()
    if y is None:
        y = _to_np(x)
        x = np.arange(1, len(y) + 1)
    else:
        x, y = _to_np(x), _to_np(y)
    _cur_ax().barh(x, y, height=height)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_scatter(x, y, s=None, c=None, **kwargs):
    """scatter(x, y) — scatter plot."""
    _maybe_clear()
    _cur_ax().scatter(_to_np(x), _to_np(y), s=s, c=c, **kwargs)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_stem(x, y=None):
    """stem(x, y) — discrete sequence (stem) plot."""
    _maybe_clear()
    if y is None:
        _cur_ax().stem(_to_np(x))
    else:
        _cur_ax().stem(_to_np(x), _to_np(y))
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_stairs(x, y=None):
    """stairs(x, y) — stairstep plot."""
    _maybe_clear()
    if y is None:
        _cur_ax().step(np.arange(len(_to_np(x))), _to_np(x), where="post")
    else:
        _cur_ax().step(_to_np(x), _to_np(y), where="post")
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_area(x, y=None):
    """area(x, y) — filled area plot."""
    _maybe_clear()
    if y is None:
        y = _to_np(x)
        x = np.arange(len(y))
    else:
        x, y = _to_np(x), _to_np(y)
    _cur_ax().fill_between(x, y)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_errorbar(x, y, err, **kwargs):
    """errorbar(x, y, err) — plot with error bars."""
    _maybe_clear()
    _cur_ax().errorbar(_to_np(x), _to_np(y), yerr=_to_np(err), **kwargs)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_histogram(data, bins=10):
    _maybe_clear()
    counts, edges = np.histogram(_to_np(data), bins=bins)
    _cur_ax().bar(edges[:-1], counts, width=np.diff(edges), align="edge")
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_pie(x, **kwargs):
    """pie(x) — pie chart."""
    _maybe_clear()
    _cur_ax().pie(_to_np(x), **kwargs)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_polar(theta, r):
    """polar(theta, rho) — polar coordinate plot."""
    fig = _cur_fig()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(_to_np(theta), _to_np(r))
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_fill(x, y, *args, **kwargs):
    """fill(x, y, c) — filled polygon."""
    _maybe_clear()
    _cur_ax().fill(_to_np(x), _to_np(y), *args, **kwargs)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_line(x, y, **kwargs):
    """line(x, y) — create line graphics object."""
    _maybe_clear()
    _cur_ax().plot(_to_np(x), _to_np(y), **kwargs)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_rectangle(pos, width, height, **kwargs):
    """Draw rectangle at (x, y) with given width and height."""
    _maybe_clear()
    x, y = float(pos[0]), float(pos[1])
    _cur_ax().add_patch(_Rectangle((x, y), width, height, **kwargs))
    _cur_ax().autoscale_view()
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


# ===================================================================
# 3-D plots
# ===================================================================

def _get_3d_ax():
    fig = _cur_fig()
    ax = fig.gca()
    if not hasattr(ax, "plot3D"):
        # Get current subplot geometry if any
        geo = ax.get_geometry() if hasattr(ax, "get_geometry") else None
        ss = ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else None
        ax.remove()
        if ss is not None:
            ax = fig.add_subplot(ss, projection="3d")
        elif geo is not None:
            ax = fig.add_subplot(geo[0], geo[1], geo[2], projection="3d")
        else:
            ax = fig.add_subplot(111, projection="3d")
        plt.sca(ax)
    return ax


def forge_plot3(*args, **kwargs):
    """plot3(x,y,z), plot3(x,y,z,fmt), plot3(x,y,z,fmt,'Prop',val,...)"""
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray
    ax = _get_3d_ax()
    # Convert args
    a = []
    for arg in args:
        if isinstance(arg, ForgeChar):
            a.append(arg.to_str())
        elif hasattr(arg, '__array__') or isinstance(arg, ForgeArray):
            a.append(_to_np(arg))
        else:
            a.append(arg)
    # Parse: x,y,z required, then optional fmt, then key-value pairs
    if len(a) >= 3:
        x, y, z = a[0], a[1], a[2]
        rest = a[3:]
        fmt_kwargs = {}
        # Check for format string
        if rest and isinstance(rest[0], str) and len(rest[0]) <= 4:
            fmt_kwargs = _parse_linespec(rest[0])
            rest = rest[1:]
        # Parse key-value pairs (e.g. 'LineWidth', 2)
        i = 0
        while i + 1 < len(rest):
            key = rest[i]
            val = rest[i + 1]
            if isinstance(key, str):
                kl = key.lower()
                if kl == 'linewidth':
                    fmt_kwargs['linewidth'] = float(_to_np(val).flat[0]) if hasattr(val, '__array__') else float(val)
                elif kl == 'markersize':
                    fmt_kwargs['markersize'] = float(_to_np(val).flat[0]) if hasattr(val, '__array__') else float(val)
                elif kl == 'color':
                    fmt_kwargs['color'] = val
            i += 2
        fmt_kwargs.update(kwargs)
        ax.plot(x, y, z, **fmt_kwargs)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_scatter3(x, y, z, **kwargs):
    """scatter3(x, y, z) — 3-D scatter plot."""
    ax = _get_3d_ax()
    ax.scatter(_to_np(x), _to_np(y), _to_np(z), **kwargs)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def _prepare_surf_args(*args):
    """Parse surf/mesh args: surf(Z), surf(X,Y,Z), surf(...,C), surf(...,prop,val)"""
    from forge.engine.types import ForgeArray
    from forge.engine.containers import ForgeChar
    converted = []
    props = {}
    i = 0
    raw = list(args)
    # Unpack tuple/list first arg (e.g. surf(peaks(30)) passes a tuple)
    if len(raw) == 1 and isinstance(raw[0], (tuple, list)):
        raw = list(raw[0])
    while i < len(raw):
        a = raw[i]
        if isinstance(a, ForgeChar):
            # Property name - next arg is value
            key = a.to_str().lower()
            if i + 1 < len(raw):
                val = raw[i + 1]
                if isinstance(val, (ForgeArray,)):
                    val = _to_np(val)
                    if val.size == 1:
                        val = float(val.flat[0])
                elif isinstance(val, ForgeChar):
                    val = val.to_str()
                props[key] = val
                i += 2
                continue
            i += 1
            continue
        converted.append(_to_np(a) if hasattr(a, "__array__") or isinstance(a, ForgeArray) else a)
        i += 1

    if len(converted) == 1:
        Z = converted[0]
        m, n = Z.shape
        X, Y = np.meshgrid(np.arange(1, n + 1, dtype=float), np.arange(1, m + 1, dtype=float))
        C = None
    elif len(converted) == 2:
        Z, C = converted[0], converted[1]
        m, n = Z.shape
        X, Y = np.meshgrid(np.arange(1, n + 1, dtype=float), np.arange(1, m + 1, dtype=float))
    elif len(converted) == 3:
        X, Y, Z = converted[0], converted[1], converted[2]
        # Auto-meshgrid if X,Y are 1D vectors
        if X.ndim == 1 and Y.ndim == 1:
            X, Y = np.meshgrid(X, Y)
        C = None
    elif len(converted) >= 4:
        X, Y, Z, C = converted[0], converted[1], converted[2], converted[3]
        if X.ndim == 1 and Y.ndim == 1:
            X, Y = np.meshgrid(X, Y)
    else:
        raise ValueError("surf requires at least 1 argument (Z)")
    return X, Y, Z, C, props


def forge_surf(*args, **kwargs):
    """surf(X, Y, Z) — 3-D surface plot."""
    ax = _get_3d_ax()
    X, Y, Z, C, props = _prepare_surf_args(*args)
    kw = {"cmap": "viridis", "edgecolor": "none"}
    if C is not None:
        kw["facecolors"] = plt.cm.viridis((C - C.min()) / max(C.ptp(), 1e-15))
    kw.update(props)
    kw.update(kwargs)
    surf_obj = ax.plot_surface(X, Y, Z, **kw)
    fig = plt.gcf()
    fig._forge_last_mappable = surf_obj
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_surfc(*args, **kwargs):
    """surfc(X, Y, Z) — surface plot with contour underneath."""
    ax = _get_3d_ax()
    X, Y, Z, C, props = _prepare_surf_args(*args)
    kw = {"alpha": 0.7, "cmap": "viridis"}
    kw.update(props)
    kw.update(kwargs)
    surf_obj = ax.plot_surface(X, Y, Z, **kw)
    ax.contour(X, Y, Z, zdir="z", offset=Z.min())
    fig = plt.gcf()
    fig._forge_last_mappable = surf_obj
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_surfl(*args, **kwargs):
    """surfl(X, Y, Z) — surface plot with lighting."""
    ax = _get_3d_ax()
    X, Y, Z, C, props = _prepare_surf_args(*args)
    kw = {"rstride": 1, "cstride": 1, "shade": True, "cmap": "copper"}
    kw.update(props)
    kw.update(kwargs)
    ax.plot_surface(X, Y, Z, **kw)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_mesh(*args, **kwargs):
    """mesh(X, Y, Z) — 3-D mesh surface plot."""
    ax = _get_3d_ax()
    X, Y, Z, C, props = _prepare_surf_args(*args)
    kw = {}
    kw.update(props)
    kw.update(kwargs)
    ax.plot_wireframe(X, Y, Z, **kw)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_meshc(*args, **kwargs):
    """meshc(X, Y, Z) — mesh plot with contour underneath."""
    ax = _get_3d_ax()
    X, Y, Z, C, props = _prepare_surf_args(*args)
    kw = {}
    kw.update(props)
    kw.update(kwargs)
    ax.plot_wireframe(X, Y, Z, **kw)
    ax.contour(X, Y, Z, zdir="z", offset=Z.min())
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_meshz(*args, **kwargs):
    """meshz(X, Y, Z) — mesh plot with curtain."""
    ax = _get_3d_ax()
    X, Y, Z, C, props = _prepare_surf_args(*args)
    kw = {}
    kw.update(props)
    kw.update(kwargs)
    ax.plot_wireframe(X, Y, Z, **kw)
    ax.plot_surface(X, Y, np.zeros_like(Z), alpha=0.1)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_contour(*args, **kwargs):
    """contour(Z), contour(X, Y, Z), contour(..., N), contour(..., levels)"""
    from forge.engine.types import ForgeArray
    _maybe_clear()
    # Unpack tuple/list first arg (e.g. contour(peaks(30)) passes a tuple)
    flat_args = list(args)
    if len(flat_args) == 1 and isinstance(flat_args[0], (tuple, list)):
        flat_args = list(flat_args[0])
    arrays = [_to_np(a) for a in flat_args if hasattr(a, "__array__") or isinstance(a, ForgeArray)]
    extra = [a for a in flat_args if not (hasattr(a, "__array__") or isinstance(a, ForgeArray))]
    if len(arrays) == 1:
        Z = arrays[0]
        cs = _cur_ax().contour(Z, *extra, **kwargs)
    elif len(arrays) >= 3:
        X, Y, Z = arrays[0], arrays[1], arrays[2]
        cs = _cur_ax().contour(_to_np(X), _to_np(Y), _to_np(Z), *extra, **kwargs)
    else:
        raise ValueError("contour requires 1 or 3 array arguments")
    _cur_fig()._forge_last_mappable = cs
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_contourf(X, Y, Z, *args, **kwargs):
    """contourf(X, Y, Z) — filled contour plot."""
    _maybe_clear()
    # Convert ForgeArray args (e.g., levels)
    converted_args = []
    for a in args:
        if hasattr(a, '__array__') or isinstance(a, ForgeArray):
            val = _to_np(a)
            if val.size == 1:
                converted_args.append(int(val.flat[0]))  # scalar -> int (num levels)
            else:
                converted_args.append(val)
        else:
            converted_args.append(a)
    Xn, Yn, Zn = _to_np(X), _to_np(Y), _to_np(Z)
    # Handle constant data (contourf needs variation)
    if np.ptp(Zn) < 1e-15:
        Zn = Zn + np.random.randn(*Zn.shape) * 1e-15
    cs = _cur_ax().contourf(Xn, Yn, Zn, *converted_args, **kwargs)
    # Store for colorbar
    _cur_fig()._forge_last_mappable = cs
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_contour3(X, Y, Z, *args, **kwargs):
    """contour3(X, Y, Z) — 3-D contour plot."""
    ax = _get_3d_ax()
    ax.contour(_to_np(X), _to_np(Y), _to_np(Z), *args, **kwargs)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_waterfall(X, Y, Z, **kwargs):
    """waterfall(X, Y, Z) — waterfall plot."""
    ax = _get_3d_ax()
    Xn, Yn, Zn = _to_np(X), _to_np(Y), _to_np(Z)
    for i in range(Xn.shape[0]):
        ax.plot(Xn[i, :], Yn[i, :], Zn[i, :], color="blue", **kwargs)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_pcolor(*args, **kwargs):
    """pcolor(Z), pcolor(X,Y,Z) - pseudocolor plot"""
    from forge.engine.types import ForgeArray
    _maybe_clear()
    arrays = [_to_np(a) for a in args if hasattr(a, "__array__") or isinstance(a, ForgeArray)]
    if len(arrays) == 1:
        Z = arrays[0]
        pc = _cur_ax().pcolormesh(Z, shading="auto", **kwargs)
    elif len(arrays) >= 3:
        X, Y, Z = arrays[0], arrays[1], arrays[2]
        if X.ndim == 1 and Y.ndim == 1:
            X, Y = np.meshgrid(X, Y)
        pc = _cur_ax().pcolormesh(X, Y, Z, shading="auto", **kwargs)
    else:
        raise ValueError("pcolor requires 1 or 3 array arguments")
    _cur_fig()._forge_last_mappable = pc
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_imagesc(*args, **kwargs):
    """imagesc(Z), imagesc(x,y,Z) - scaled image display"""
    from forge.engine.types import ForgeArray
    _maybe_clear()
    arrays = [_to_np(a) for a in args if hasattr(a, "__array__") or isinstance(a, ForgeArray)]
    if len(arrays) == 1:
        Z = arrays[0]
        im = _cur_ax().imshow(Z, aspect="auto", origin="upper", **kwargs)
    elif len(arrays) >= 3:
        x, y, Z = arrays[0].flatten(), arrays[1].flatten(), arrays[2]
        extent = [x[0], x[-1], y[-1], y[0]]
        im = _cur_ax().imshow(Z, aspect="auto", origin="upper", extent=extent, **kwargs)
    else:
        raise ValueError("imagesc requires 1 or 3 arguments")
    _cur_fig()._forge_last_mappable = im
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_shading(*args):
    """shading flat/interp/faceted - change surface shading"""
    from forge.engine.containers import ForgeChar
    if not args:
        return
    mode = args[0]
    if isinstance(mode, ForgeChar):
        mode = mode.to_str()
    mode = str(mode).lower()
    ax = _cur_ax()
    for coll in ax.collections:
        if hasattr(coll, "set_edgecolor"):
            if mode == "flat":
                coll.set_edgecolor("face")
            elif mode == "interp":
                coll.set_edgecolor("none")
            elif mode == "faceted":
                coll.set_edgecolor("black")
                coll.set_linewidth(0.5)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_peaks(*args):
    """[X,Y,Z] = peaks(n) - MATLAB test surface"""
    from forge.engine.types import ForgeArray
    n = 49
    if args:
        n = int(_to_np(args[0]).flat[0])
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)
    Z = 3 * (1 - X)**2 * np.exp(-X**2 - (Y + 1)**2)         - 10 * (X / 5 - X**3 - Y**5) * np.exp(-X**2 - Y**2)         - 1/3 * np.exp(-(X + 1)**2 - Y**2)
    return ForgeArray(X), ForgeArray(Y), ForgeArray(Z)


def forge_sombrero(*args):
    """[X,Y,Z] = sombrero(n) - sinc hat function"""
    from forge.engine.types import ForgeArray
    n = 41
    if args:
        n = int(_to_np(args[0]).flat[0])
    x = np.linspace(-8, 8, n)
    y = np.linspace(-8, 8, n)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2) + np.finfo(float).eps
    Z = np.sin(R) / R
    return ForgeArray(X), ForgeArray(Y), ForgeArray(Z)


# ===================================================================
# Formatting
# ===================================================================


def _parse_text_kwargs(args):
    """Extract MATLAB-style keyword pairs from args list.
    Returns (text_string, matplotlib_kwargs)."""
    from forge.engine.containers import ForgeChar
    from forge.engine.types import ForgeArray

    text = None
    kw = {}
    i = 0
    while i < len(args):
        a = args[i]
        # First non-keyword arg is the text
        if text is None:
            if isinstance(a, ForgeChar):
                text = a.to_str()
            elif hasattr(a, 'to_str'):
                text = a.to_str()
            else:
                text = str(a)
            i += 1
            continue
        # Remaining: check for keyword pairs
        if isinstance(a, (str, ForgeChar)):
            key = a.to_str() if isinstance(a, ForgeChar) else a
            if key.lower() in ("fontsize", "fontweight", "fontname", "color",
                               "horizontalalignment", "verticalalignment",
                               "interpreter", "rotation") and i + 1 < len(args):
                val = args[i + 1]
                if isinstance(val, ForgeChar):
                    val = val.to_str()
                elif hasattr(val, 'to_str'):
                    val = val.to_str()
                elif isinstance(val, ForgeArray):
                    val = float(val.data.flat[0])
                kw[key.lower()] = val
                i += 2
                continue
        i += 1
    return text or "", kw

def forge_title(*args):
    """title(str) or title(str, 'FontSize', 14, ...)."""
    text, kw = _parse_text_kwargs(args)
    # Map to matplotlib kwargs
    mpl_kw = {}
    if "fontsize" in kw:
        mpl_kw["fontsize"] = kw["fontsize"]
    if "fontweight" in kw:
        mpl_kw["fontweight"] = kw["fontweight"]
    if "color" in kw:
        mpl_kw["color"] = kw["color"]
    _cur_ax().set_title(text, **mpl_kw)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_xlabel(*args):
    """xlabel(str) or xlabel(str, 'FontSize', 12, ...)."""
    text, kw = _parse_text_kwargs(args)
    mpl_kw = {}
    if "fontsize" in kw:
        mpl_kw["fontsize"] = kw["fontsize"]
    if "fontweight" in kw:
        mpl_kw["fontweight"] = kw["fontweight"]
    if "color" in kw:
        mpl_kw["color"] = kw["color"]
    _cur_ax().set_xlabel(text, **mpl_kw)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_ylabel(*args):
    """ylabel(str) or ylabel(str, 'FontSize', 12, ...)."""
    text, kw = _parse_text_kwargs(args)
    mpl_kw = {}
    if "fontsize" in kw:
        mpl_kw["fontsize"] = kw["fontsize"]
    if "fontweight" in kw:
        mpl_kw["fontweight"] = kw["fontweight"]
    if "color" in kw:
        mpl_kw["color"] = kw["color"]
    _cur_ax().set_ylabel(text, **mpl_kw)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_zlabel(*args):
    """zlabel(str) or zlabel(str, 'FontSize', 12, ...)."""
    text, kw = _parse_text_kwargs(args)
    ax = _cur_ax()
    if hasattr(ax, "set_zlabel"):
        mpl_kw = {}
        if "fontsize" in kw:
            mpl_kw["fontsize"] = kw["fontsize"]
        ax.set_zlabel(text, **mpl_kw)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_legend(*args, **kwargs):
    """legend(str1, str2, ...) — add legend to plot."""
    # Convert ForgeChar args to Python strings, extracting MATLAB-style keyword pairs
    from forge.engine.containers import ForgeChar
    # MATLAB keyword args that can appear in positional args
    _LEGEND_KWARGS = {"location", "fontsize", "orientation", "numcolumns",
                      "interpreter", "box", "color", "textcolor", "edgecolor"}
    str_args = []
    for a in args:
        if isinstance(a, ForgeChar):
            str_args.append(a.to_str())
        elif hasattr(a, 'to_str'):
            str_args.append(a.to_str())
        elif isinstance(a, str):
            str_args.append(a)
        else:
            str_args.append(a)  # keep non-string as-is
    # Extract keyword pairs from positional args
    labels = []
    _loc_map = {"north": "upper center", "south": "lower center",
                "east": "center right", "west": "center left",
                "northeast": "upper right", "northwest": "upper left",
                "southeast": "lower right", "southwest": "lower left",
                "best": "best", "none": "best"}
    i = 0
    while i < len(str_args):
        if isinstance(str_args[i], str) and str_args[i].lower() in _LEGEND_KWARGS and i + 1 < len(str_args):
            key = str_args[i].lower()
            val = str_args[i + 1]
            if isinstance(val, ForgeChar):
                val = val.to_str()
            elif hasattr(val, 'to_str'):
                val = val.to_str()
            if key == "location":
                loc = val.lower() if isinstance(val, str) else str(val)
                kwargs["loc"] = _loc_map.get(loc, loc)
            elif key == "fontsize":
                kwargs["fontsize"] = float(val) if not isinstance(val, str) else val
            elif key == "numcolumns":
                kwargs["ncol"] = int(float(val)) if not isinstance(val, str) else val
            i += 2
        else:
            if isinstance(str_args[i], str):
                labels.append(str_args[i])
            else:
                labels.append(str(str_args[i]))
            i += 1
    ax = _cur_ax()
    if labels:
        # Collect all plottable handles in creation order
        from matplotlib.container import BarContainer
        from matplotlib.lines import Line2D
        from matplotlib.collections import PathCollection
        handles = []
        # First: bar containers (in order)
        for c in ax.containers:
            if isinstance(c, BarContainer):
                handles.append(c)
        # Then: lines and collections
        for artist in ax.get_children():
            if isinstance(artist, Line2D) and not artist.get_label().startswith('_'):
                handles.append(artist)
            elif isinstance(artist, PathCollection):
                handles.append(artist)
        # Also check for unlabeled lines (created by plot())
        for line in ax.lines:
            if line not in handles:
                handles.append(line)
        if len(labels) <= len(handles):
            ax.legend(handles[:len(labels)], labels, **kwargs)
        else:
            ax.legend(labels, **kwargs)
    else:
        ax.legend(**kwargs)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_grid(on=None):
    """grid on/off — toggle grid lines on axes."""
    if on is None:
        # toggle
        ax = _cur_ax()
        ax.grid(not ax.xaxis.get_gridlines()[0].get_visible())
    else:
        # Handle ForgeChar string args from command-style (grid on, grid off)
        if hasattr(on, 'to_str'):
            on = on.to_str()
        if isinstance(on, str):
            _cur_ax().grid(on.lower() in ('on', 'true', '1'))
        else:
            _cur_ax().grid(bool(on))
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_axis(spec=None):
    """axis([xmin xmax ymin ymax]) or axis('equal'), axis('tight'), etc."""
    ax = _cur_ax()
    if spec is None:
        return list(ax.axis())
    if hasattr(spec, 'to_str'):
        spec = spec.to_str()
    if isinstance(spec, str):
        ax.axis(spec)
    else:
        lims = _to_np(spec).ravel()
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_xlim(lo=None, hi=None):
    """xlim([lo hi]) — set x-axis limits."""
    if lo is None:
        return _cur_ax().get_xlim()
    if hi is None:
        arr = _to_np(lo).ravel()
        _cur_ax().set_xlim(float(arr[0]), float(arr[1]))
    else:
        _cur_ax().set_xlim(float(lo), float(hi))
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_ylim(lo=None, hi=None):
    """ylim([lo hi]) — set y-axis limits."""
    if lo is None:
        return _cur_ax().get_ylim()
    if hi is None:
        # Array form: ylim([lo hi])
        arr = _to_np(lo).ravel()
        _cur_ax().set_ylim(float(arr[0]), float(arr[1]))
    else:
        _cur_ax().set_ylim(float(lo), float(hi))
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_zlim(lo=None, hi=None):
    """zlim([lo hi]) — set z-axis limits."""
    ax = _cur_ax()
    if not hasattr(ax, "set_zlim"):
        return
    if lo is None:
        return ax.get_zlim()
    ax.set_zlim(float(lo), float(hi))
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_hold(state=None):
    """Toggle or set hold state. Accepts 'on', 'off', or boolean."""
    global _hold_state
    if state is None:
        _hold_state = not _hold_state
    else:
        # Handle string args from command-style: hold on, hold off
        s = str(state)
        if hasattr(state, 'to_str'):
            s = state.to_str()
        if s.lower() == 'on':
            _hold_state = True
        elif s.lower() == 'off':
            _hold_state = False
        else:
            _hold_state = bool(state)


def forge_colorbar(**kwargs):
    """colorbar — add colorbar to current axes."""
    fig = _cur_fig()
    ax = _cur_ax()
    mappable = getattr(fig, '_forge_last_mappable', None)
    if mappable is None:
        # Try to find a mappable from the current axes images/collections
        images = ax.get_images()
        if images:
            mappable = images[-1]
        else:
            for c in ax.collections:
                if hasattr(c, 'get_array') and c.get_array() is not None:
                    mappable = c
                    break
    if mappable is not None:
        plt.colorbar(mappable, ax=ax, **kwargs)
    else:
        plt.colorbar(**kwargs)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_colormap(name: str):
    """Set the default colormap by name."""
    plt.set_cmap(name)


# ===================================================================
# Figure management
# ===================================================================

def forge_figure(n=None):
    """Create or switch to figure *n*."""
    if n is None:
        fig = plt.figure()
    else:
        fig = plt.figure(int(n))
    # Raise the figure window to the front
    try:
        manager = fig.canvas.manager
        if hasattr(manager, 'window'):
            manager.window.raise_()
            manager.window.activateWindow()
    except Exception:
        pass
    plt.draw()
    plt.pause(0.01)
    _enhance_current()
    return fig


def forge_subplot(m, n, p):
    """subplot(m, n, p) — create axes in tiled positions."""
    m_int = int(_to_np(m).flat[0]) if hasattr(m, '__array__') else int(m)
    n_int = int(_to_np(n).flat[0]) if hasattr(n, '__array__') else int(n)
    p_int = int(_to_np(p).flat[0]) if hasattr(p, '__array__') else int(p)
    fig = _cur_fig()
    ax = fig.add_subplot(m_int, n_int, p_int)
    plt.sca(ax)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()
    return ax


def forge_gca():
    """gca — get current axes handle."""
    return _cur_ax()


def forge_gcf():
    """gcf — get current figure handle."""
    return _cur_fig()


def forge_cla():
    """cla — clear current axes."""
    _cur_ax().cla()
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_clf():
    """clf — clear current figure."""
    _cur_fig().clf()
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


# Module-level flag for IDE to detect close-all
_close_all_requested = False

def forge_close(n=None):
    """close — close current figure window. close all closes all figures."""
    global _close_all_requested
    if n is None:
        plt.close()
    elif n == "all":
        plt.close("all")
        _close_all_requested = True
    else:
        plt.close(int(n))


def _resolve_figure(h):
    """Resolve a figure handle argument to a matplotlib Figure object."""
    import matplotlib.pyplot as _plt
    from matplotlib.figure import Figure as _Figure
    # If h is a callable (e.g. gcf passed without parens), call it
    if callable(h):
        h = h()
    # If h is already a matplotlib Figure, use it directly
    if isinstance(h, _Figure):
        return h
    # If h is a ForgeArray, extract the number
    if hasattr(h, 'array'):
        h = int(h.array.flat[0])
    elif hasattr(h, 'data'):
        h = int(h.data.flat[0])
    # Now h should be an int figure number
    return _plt.figure(int(h))


def forge_saveas(*args):
    """Save figure: saveas(handle, filename) or saveas(handle, filename, format) or saveas(filename)."""
    # Parse args: saveas(h, file) or saveas(h, file, fmt) or saveas(file)
    if len(args) >= 2:
        fig = _resolve_figure(args[0])
        filename = args[1]
        if hasattr(filename, 'to_str'):
            filename = filename.to_str()
        elif hasattr(filename, 'array'):
            filename = str(filename)
        fmt = args[2] if len(args) > 2 else None
        if hasattr(fmt, 'to_str'):
            fmt = fmt.to_str()
    else:
        filename = args[0]
        if hasattr(filename, 'to_str'):
            filename = filename.to_str()
        fig = _cur_fig()
        fmt = None
    if fmt:
        fig.savefig(str(filename), format=str(fmt), dpi=150, bbox_inches="tight")
    else:
        fig.savefig(str(filename), dpi=150, bbox_inches="tight")


def forge_print_fig(*args):
    """Octave-compatible print command: print(filename), print(h, filename), print('-dpng', filename)."""
    from matplotlib.figure import Figure as _Figure
    dpi = 150
    fig = _cur_fig()
    filename = None
    fmt = None
    for a in args:
        # Handle callable (e.g. gcf without parens)
        if callable(a):
            a = a()
        # Handle matplotlib Figure directly
        if isinstance(a, _Figure):
            fig = a
            continue
        if hasattr(a, 'to_str'):
            a = a.to_str()
        elif hasattr(a, 'array'):
            import numpy as np
            val = a.array.flat[0]
            if np.issubdtype(type(val), np.floating) or np.issubdtype(type(val), np.integer):
                fig = plt.figure(int(val))
                continue
        if isinstance(a, str):
            if a.startswith("-r"):
                dpi = int(a[2:])
            elif a.startswith("-d"):
                fmt = a[2:]
            else:
                filename = a
    if filename:
        kwargs = {"dpi": dpi, "bbox_inches": "tight"}
        if fmt:
            kwargs["format"] = fmt
        fig.savefig(filename, **kwargs)


def forge_suptitle(*args, **kwargs):
    """suptitle(str) or suptitle(str, 'FontSize', n, ...) -- add super title to figure."""
    if not args:
        return
    title_str = args[0]
    if hasattr(title_str, 'to_str'):
        title_str = title_str.to_str()
    mpl_kwargs = {}
    i = 1
    while i < len(args):
        key = args[i]
        if hasattr(key, 'to_str'):
            key = key.to_str()
        if isinstance(key, str) and i + 1 < len(args):
            val = args[i + 1]
            if hasattr(val, 'to_str'):
                val = val.to_str()
            elif hasattr(val, 'array'):
                val = float(val.array.flat[0])
            prop = key.lower()
            if prop == 'fontsize':
                mpl_kwargs['fontsize'] = val
            elif prop == 'fontweight':
                mpl_kwargs['fontweight'] = val
            elif prop == 'color':
                mpl_kwargs['color'] = val
            i += 2
        else:
            i += 1
    fig = _cur_fig()
    fig.suptitle(str(title_str), **mpl_kwargs)
    plt.draw()


# ===================================================================
# Registry
# ===================================================================

def forge_drawnow():
    """Force figure update."""
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_pause(t=None):
    """Pause execution. With arg, pause for t seconds."""
    if t is not None:
        from forge.engine.types import ForgeArray
        if isinstance(t, ForgeArray):
            t = float(t.data.flat[0])
        import time
        time.sleep(float(t))
    else:
        plt.pause(0.1)



def forge_view(*args):
    """Set 3D view angle. view(az, el) or view(2) or view(3)."""
    from forge.engine.types import ForgeArray
    ax = _cur_ax()
    if len(args) == 1:
        v = args[0]
        if isinstance(v, ForgeArray):
            v = float(v.data.flat[0])
        v = int(v)
        if v == 2:
            ax.view_init(elev=90, azim=-90)
        elif v == 3:
            ax.view_init(elev=30, azim=-37.5)
    elif len(args) == 2:
        az = args[0]
        el = args[1]
        if isinstance(az, ForgeArray):
            az = float(az.data.flat[0])
        if isinstance(el, ForgeArray):
            el = float(el.data.flat[0])
        ax.view_init(elev=float(el), azim=float(az))
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_clim(*args):
    """Set color axis limits. clim([lo hi]) or clim(lo, hi)."""
    from forge.engine.types import ForgeArray
    ax = _cur_ax()
    if len(args) == 1:
        v = args[0]
        if isinstance(v, ForgeArray):
            v = v.data.flatten()
        lo, hi = float(v[0]), float(v[1])
    elif len(args) == 2:
        lo = float(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else float(args[0])
        hi = float(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else float(args[1])
    else:
        return
    for im in ax.get_images() + ax.collections:
        if hasattr(im, "set_clim"):
            im.set_clim(lo, hi)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_rotate3d(state=None):
    """Enable/disable 3D rotation."""
    pass


def forge_set_prop(*args):
    """Set graphics object property. set(h, prop, val, ...)."""
    from forge.engine.types import ForgeArray
    from forge.engine.containers import ForgeChar
    if len(args) >= 3:
        for i in range(1, len(args)-1, 2):
            prop = args[i]
            val = args[i+1]
            if isinstance(prop, ForgeChar):
                prop = prop.to_str()
            if isinstance(val, ForgeChar):
                val = val.to_str()
            if isinstance(val, ForgeArray):
                val = float(val.data.flat[0])
            prop = str(prop).lower()
            ax = _cur_ax()
            if prop == "fontsize":
                ax.title.set_fontsize(val)
                ax.xaxis.label.set_fontsize(val)
                ax.yaxis.label.set_fontsize(val)
                ax.tick_params(axis='both', labelsize=val)
            elif prop == "linewidth":
                for spine in ax.spines.values():
                    spine.set_linewidth(val)
            elif prop == "xlim":
                arr = _to_np(val).ravel() if not isinstance(val, (int, float)) else None
                if arr is not None:
                    ax.set_xlim(float(arr[0]), float(arr[1]))
            elif prop == "ylim":
                arr = _to_np(val).ravel() if not isinstance(val, (int, float)) else None
                if arr is not None:
                    ax.set_ylim(float(arr[0]), float(arr[1]))
            elif prop == "xtick":
                ax.set_xticks(_to_np(val).ravel())
            elif prop == "ytick":
                ax.set_yticks(_to_np(val).ravel())
            elif prop == "xticklabel":
                if isinstance(val, list):
                    ax.set_xticklabels([str(v) for v in val])
            elif prop == "yticklabel":
                if isinstance(val, list):
                    ax.set_yticklabels([str(v) for v in val])
        plt.draw()
        plt.pause(0.01)


def forge_get_prop(*args):
    """Get graphics object property."""
    return None


def forge_text_func(*args):
    """Add text annotation. text(x, y, str, 'prop', val, ...)."""
    from forge.engine.types import ForgeArray
    from forge.engine.containers import ForgeChar
    ax = _cur_ax()
    if len(args) >= 3:
        x = float(args[0].data.flat[0]) if isinstance(args[0], ForgeArray) else float(args[0])
        y = float(args[1].data.flat[0]) if isinstance(args[1], ForgeArray) else float(args[1])
        s = args[2]
        if isinstance(s, ForgeChar):
            s = s.to_str()
        # Extract keyword pairs from remaining args
        kw = {}
        i = 3
        while i + 1 < len(args):
            key = args[i]
            val = args[i + 1]
            if isinstance(key, ForgeChar):
                key = key.to_str()
            elif hasattr(key, 'to_str'):
                key = key.to_str()
            if isinstance(val, ForgeChar):
                val = val.to_str()
            elif hasattr(val, 'to_str'):
                val = val.to_str()
            elif isinstance(val, ForgeArray):
                val = float(val.data.flat[0])
            key = str(key).lower()
            if key == "fontsize":
                kw["fontsize"] = val
            elif key == "fontweight":
                kw["fontweight"] = val
            elif key == "horizontalalignment":
                kw["ha"] = val
            elif key == "verticalalignment":
                kw["va"] = val
            elif key == "color":
                kw["color"] = val
            elif key == "rotation":
                kw["rotation"] = float(val)
            i += 2
        ax.text(x, y, str(s), **kw)
    plt.draw()
    plt.pause(0.01)
    _enhance_current()


def forge_xline(x, *args):
    """xline(x) - vertical line at x."""
    import matplotlib.pyplot as plt
    xv = float(_to_np(x).flat[0]) if hasattr(x, 'data') else float(x)
    style_kw = {}
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, str):
            if a == "LineWidth" and i + 1 < len(args):
                style_kw['linewidth'] = float(_to_np(args[i+1]).flat[0]) if hasattr(args[i+1], 'data') else float(args[i+1])
                i += 2; continue
            elif len(a) <= 4 and not a[0].isupper():
                style_kw['linestyle'] = a[:2] if len(a) > 1 and a[0] == '-' else '-'
                if a[-1].isalpha(): style_kw['color'] = a[-1]
                i += 1; continue
        i += 1
    _cur_ax().axvline(x=xv, **style_kw)
    return _FA(np.array(xv))

def forge_yline(y, *args):
    """yline(y) - horizontal line at y."""
    import matplotlib.pyplot as plt
    yv = float(_to_np(y).flat[0]) if hasattr(y, 'data') else float(y)
    style_kw = {}
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, str):
            if a == "LineWidth" and i + 1 < len(args):
                style_kw['linewidth'] = float(_to_np(args[i+1]).flat[0]) if hasattr(args[i+1], 'data') else float(args[i+1])
                i += 2; continue
            elif len(a) <= 4 and not a[0].isupper():
                style_kw['linestyle'] = a[:2] if len(a) > 1 and a[0] == '-' else '-'
                if a[-1].isalpha(): style_kw['color'] = a[-1]
                i += 1; continue
        i += 1
    _cur_ax().axhline(y=yv, **style_kw)
    return _FA(np.array(yv))

def forge_meshgrid(x, y=None, z=None):
    """[X,Y] = meshgrid(x,y) or [X,Y,Z] = meshgrid(x,y,z) - 2D/3D grid."""
    xv = _to_np(x).flatten()
    yv = _to_np(y).flatten() if y is not None else xv.copy()
    if z is not None:
        zv = _to_np(z).flatten()
        X, Y, Z = np.meshgrid(xv, yv, zv, indexing="xy")
        return ForgeArray(X), ForgeArray(Y), ForgeArray(Z)
    X, Y = np.meshgrid(xv, yv)
    return ForgeArray(X), ForgeArray(Y)



# ---------------------------------------------------------------------------
# bar3 -- 3D bar chart
# ---------------------------------------------------------------------------

def forge_bar3(*args, **kwargs):
    """bar3(Z) or bar3(Y, Z) -- 3D bar chart."""
    from mpl_toolkits.mplot3d import Axes3D
    from forge.engine.types import ForgeArray, _unwrap
    if len(args) == 0:
        raise ValueError("bar3 requires at least one argument")
    Z = _unwrap(args[0]) if isinstance(args[0], ForgeArray) else np.asarray(args[0])
    if Z.ndim == 1:
        Z = Z.reshape(1, -1)
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    rows, cols = Z.shape
    xpos, ypos = np.meshgrid(np.arange(cols), np.arange(rows))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos, dtype=float)
    dx = dy = 0.6
    dz = Z.flatten().astype(float)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ForgeArray(np.array(0.0))


# ---------------------------------------------------------------------------
# boxplot -- Box-and-whisker plot
# ---------------------------------------------------------------------------

def forge_boxplot(*args, **kwargs):
    """boxplot(X) -- Box-and-whisker plot."""
    from forge.engine.types import ForgeArray, _unwrap
    if len(args) == 0:
        raise ValueError("boxplot requires at least one argument")
    X = _unwrap(args[0]) if isinstance(args[0], ForgeArray) else np.asarray(args[0])
    if X.ndim == 1:
        data = [X]
    else:
        data = [X[:, j] for j in range(X.shape[1])]
    ax = plt.gca()
    bp = ax.boxplot(data, patch_artist=True)
    for box in bp['boxes']:
        box.set_facecolor('#89b4fa')
        box.set_alpha(0.7)
    return ForgeArray(np.array(0.0))


# ---------------------------------------------------------------------------
# heatmap -- 2D heatmap visualization
# ---------------------------------------------------------------------------

def forge_heatmap(*args, **kwargs):
    """heatmap(Z) -- Display matrix as a color-coded heatmap."""
    from forge.engine.types import ForgeArray, _unwrap
    if len(args) == 0:
        raise ValueError("heatmap requires at least one argument")
    Z = _unwrap(args[0]) if isinstance(args[0], ForgeArray) else np.asarray(args[0])
    cmap = 'viridis'
    if len(args) > 1 and isinstance(args[1], str):
        cmap = args[1]
    ax = plt.gca()
    im = ax.imshow(Z, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax)
    return ForgeArray(np.array(0.0))


# ---------------------------------------------------------------------------
# ginput -- Graphical input from mouse
# ---------------------------------------------------------------------------

def forge_ginput(*args, **kwargs):
    """ginput(n) -- Get n points from the current axes using the mouse.

    Note: In headless mode, returns empty arrays.
    """
    from forge.engine.types import ForgeArray
    n = 1
    if len(args) > 0:
        n = int(args[0])
    try:
        pts = plt.ginput(n, timeout=30)
        if pts:
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            return ForgeArray(x), ForgeArray(y)
    except Exception:
        pass
    return ForgeArray(np.array([])), ForgeArray(np.array([]))


PLOTTING_REGISTRY = {
    # 2-D
    "plot":         forge_plot,
    "semilogx":     forge_semilogx,
    "semilogy":     forge_semilogy,
    "loglog":       forge_loglog,
    "bar":          forge_bar,
    "barh":         forge_barh,
    "scatter":      forge_scatter,
    "stem":         forge_stem,
    "stairs":       forge_stairs,
    "area":         forge_area,
    "errorbar":     forge_errorbar,
    "histogram":    forge_histogram,
    "pie":          forge_pie,
    "polar":        forge_polar,
    "polarplot":    forge_polar,
    "fill":         forge_fill,
    "line":         forge_line,
    "rectangle":    forge_rectangle,
    # 3-D
    "plot3":        forge_plot3,
    "scatter3":     forge_scatter3,
    "surf":         forge_surf,
    "surfc":        forge_surfc,
    "surfl":        forge_surfl,
    "mesh":         forge_mesh,
    "meshc":        forge_meshc,
    "meshz":        forge_meshz,
    "contour":      forge_contour,
    "contourf":     forge_contourf,
    "contour3":     forge_contour3,
    "waterfall":    forge_waterfall,
    "pcolor":       forge_pcolor,
    "imagesc":      forge_imagesc,
    "shading":      forge_shading,
    "peaks":        forge_peaks,
    "sombrero":     forge_sombrero,
    # Formatting
    "title":        forge_title,
    "xlabel":       forge_xlabel,
    "ylabel":       forge_ylabel,
    "zlabel":       forge_zlabel,
    "legend":       forge_legend,
    "grid":         forge_grid,
    "axis":         forge_axis,
    "xlim":         forge_xlim,
    "ylim":         forge_ylim,
    "zlim":         forge_zlim,
    "hold":         forge_hold,
    "colorbar":     forge_colorbar,
    "colormap":     forge_colormap,
    # Figure management
    "figure":       forge_figure,
    "subplot":      forge_subplot,
    "gca":          forge_gca,
    "gcf":          forge_gcf,
    "cla":          forge_cla,
    "clf":          forge_clf,
    "close":        forge_close,
    "saveas":       forge_saveas,
    "print":        forge_print_fig,
    "suptitle":     forge_suptitle,
    "drawnow":      forge_drawnow,
    "pause":        forge_pause,
    # View and display
    "view":          forge_view,
    "clim":          forge_clim,
    "caxis":         forge_clim,
    "rotate3d":      forge_rotate3d,
    "set":           forge_set_prop,
    "get":           forge_get_prop,
    "text":          forge_text_func,
    "xline":         forge_xline,
    "yline":         forge_yline,
    "meshgrid":      forge_meshgrid,
    "bar3":         forge_bar3,
    "boxplot":      forge_boxplot,
    "heatmap":      forge_heatmap,
    "ginput":       forge_ginput,
}
