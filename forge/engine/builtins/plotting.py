"""Forge plotting function library — Octave-compatible wrappers around
matplotlib (forge/engine/builtins/plotting.py)."""

import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as _Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

from forge.engine.types import ForgeArray, _unwrap

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


def forge_semilogx(*args):
    _maybe_clear()
    forge_plot(*args)
    _cur_ax().set_xscale("log")
    plt.draw()
    plt.pause(0.01)


def forge_semilogy(*args):
    _maybe_clear()
    forge_plot(*args)
    _cur_ax().set_yscale("log")
    plt.draw()
    plt.pause(0.01)


def forge_loglog(*args):
    _maybe_clear()
    forge_plot(*args)
    ax = _cur_ax()
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.draw()
    plt.pause(0.01)


def forge_bar(x, y=None, width=0.8):
    _maybe_clear()
    if y is None:
        y = _to_np(x)
        x = np.arange(1, len(y) + 1)
    else:
        x, y = _to_np(x), _to_np(y)
    _cur_ax().bar(x, y, width=width)
    plt.draw()
    plt.pause(0.01)


def forge_barh(x, y=None, height=0.8):
    _maybe_clear()
    if y is None:
        y = _to_np(x)
        x = np.arange(1, len(y) + 1)
    else:
        x, y = _to_np(x), _to_np(y)
    _cur_ax().barh(x, y, height=height)
    plt.draw()
    plt.pause(0.01)


def forge_scatter(x, y, s=None, c=None, **kwargs):
    _maybe_clear()
    _cur_ax().scatter(_to_np(x), _to_np(y), s=s, c=c, **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_stem(x, y=None):
    _maybe_clear()
    if y is None:
        _cur_ax().stem(_to_np(x))
    else:
        _cur_ax().stem(_to_np(x), _to_np(y))
    plt.draw()
    plt.pause(0.01)


def forge_stairs(x, y=None):
    _maybe_clear()
    if y is None:
        _cur_ax().step(np.arange(len(_to_np(x))), _to_np(x), where="post")
    else:
        _cur_ax().step(_to_np(x), _to_np(y), where="post")
    plt.draw()
    plt.pause(0.01)


def forge_area(x, y=None):
    _maybe_clear()
    if y is None:
        y = _to_np(x)
        x = np.arange(len(y))
    else:
        x, y = _to_np(x), _to_np(y)
    _cur_ax().fill_between(x, y)
    plt.draw()
    plt.pause(0.01)


def forge_errorbar(x, y, err, **kwargs):
    _maybe_clear()
    _cur_ax().errorbar(_to_np(x), _to_np(y), yerr=_to_np(err), **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_histogram(data, bins=10):
    _maybe_clear()
    counts, edges = np.histogram(_to_np(data), bins=bins)
    _cur_ax().bar(edges[:-1], counts, width=np.diff(edges), align="edge")
    plt.draw()
    plt.pause(0.01)


def forge_pie(x, **kwargs):
    _maybe_clear()
    _cur_ax().pie(_to_np(x), **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_polar(theta, r):
    fig = _cur_fig()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(_to_np(theta), _to_np(r))
    plt.draw()
    plt.pause(0.01)


def forge_fill(x, y, *args, **kwargs):
    _maybe_clear()
    _cur_ax().fill(_to_np(x), _to_np(y), *args, **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_line(x, y, **kwargs):
    _maybe_clear()
    _cur_ax().plot(_to_np(x), _to_np(y), **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_rectangle(pos, width, height, **kwargs):
    """Draw rectangle at (x, y) with given width and height."""
    _maybe_clear()
    x, y = float(pos[0]), float(pos[1])
    _cur_ax().add_patch(_Rectangle((x, y), width, height, **kwargs))
    _cur_ax().autoscale_view()
    plt.draw()
    plt.pause(0.01)


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


def forge_scatter3(x, y, z, **kwargs):
    ax = _get_3d_ax()
    ax.scatter(_to_np(x), _to_np(y), _to_np(z), **kwargs)
    plt.draw()
    plt.pause(0.01)


def _prepare_surf_args(*args):
    """Parse surf/mesh args: surf(Z), surf(X,Y,Z), surf(...,C), surf(...,prop,val)"""
    from forge.engine.types import ForgeArray
    from forge.engine.containers import ForgeChar
    converted = []
    props = {}
    i = 0
    raw = list(args)
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


def forge_surfc(*args, **kwargs):
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


def forge_surfl(*args, **kwargs):
    ax = _get_3d_ax()
    X, Y, Z, C, props = _prepare_surf_args(*args)
    kw = {"rstride": 1, "cstride": 1, "shade": True, "cmap": "copper"}
    kw.update(props)
    kw.update(kwargs)
    ax.plot_surface(X, Y, Z, **kw)
    plt.draw()
    plt.pause(0.01)


def forge_mesh(*args, **kwargs):
    ax = _get_3d_ax()
    X, Y, Z, C, props = _prepare_surf_args(*args)
    kw = {}
    kw.update(props)
    kw.update(kwargs)
    ax.plot_wireframe(X, Y, Z, **kw)
    plt.draw()
    plt.pause(0.01)


def forge_meshc(*args, **kwargs):
    ax = _get_3d_ax()
    X, Y, Z, C, props = _prepare_surf_args(*args)
    kw = {}
    kw.update(props)
    kw.update(kwargs)
    ax.plot_wireframe(X, Y, Z, **kw)
    ax.contour(X, Y, Z, zdir="z", offset=Z.min())
    plt.draw()
    plt.pause(0.01)


def forge_meshz(*args, **kwargs):
    ax = _get_3d_ax()
    X, Y, Z, C, props = _prepare_surf_args(*args)
    kw = {}
    kw.update(props)
    kw.update(kwargs)
    ax.plot_wireframe(X, Y, Z, **kw)
    ax.plot_surface(X, Y, np.zeros_like(Z), alpha=0.1)
    plt.draw()
    plt.pause(0.01)


def forge_contour(X, Y, Z, *args, **kwargs):
    _maybe_clear()
    _cur_ax().contour(_to_np(X), _to_np(Y), _to_np(Z), *args, **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_contourf(X, Y, Z, *args, **kwargs):
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


def forge_contour3(X, Y, Z, *args, **kwargs):
    ax = _get_3d_ax()
    ax.contour(_to_np(X), _to_np(Y), _to_np(Z), *args, **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_waterfall(X, Y, Z, **kwargs):
    ax = _get_3d_ax()
    Xn, Yn, Zn = _to_np(X), _to_np(Y), _to_np(Z)
    for i in range(Xn.shape[0]):
        ax.plot(Xn[i, :], Yn[i, :], Zn[i, :], color="blue", **kwargs)
    plt.draw()
    plt.pause(0.01)


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

def forge_title(t: str):
    _cur_ax().set_title(str(t))
    plt.draw()
    plt.pause(0.01)


def forge_xlabel(label: str):
    _cur_ax().set_xlabel(str(label))
    plt.draw()
    plt.pause(0.01)


def forge_ylabel(label: str):
    _cur_ax().set_ylabel(str(label))
    plt.draw()
    plt.pause(0.01)


def forge_zlabel(label: str):
    ax = _cur_ax()
    if hasattr(ax, "set_zlabel"):
        ax.set_zlabel(str(label))
    plt.draw()
    plt.pause(0.01)


def forge_legend(*args, **kwargs):
    # Convert ForgeChar args to Python strings
    from forge.engine.containers import ForgeChar
    labels = []
    for a in args:
        if isinstance(a, ForgeChar):
            labels.append(a.to_str())
        elif hasattr(a, 'to_str'):
            labels.append(a.to_str())
        elif isinstance(a, str):
            labels.append(a)
        else:
            labels.append(str(a))
    if labels:
        _cur_ax().legend(labels, **kwargs)
    else:
        _cur_ax().legend(**kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_grid(on=None):
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


def forge_xlim(lo=None, hi=None):
    if lo is None:
        return _cur_ax().get_xlim()
    _cur_ax().set_xlim(float(lo), float(hi))
    plt.draw()
    plt.pause(0.01)


def forge_ylim(lo=None, hi=None):
    if lo is None:
        return _cur_ax().get_ylim()
    _cur_ax().set_ylim(float(lo), float(hi))
    plt.draw()
    plt.pause(0.01)


def forge_zlim(lo=None, hi=None):
    ax = _cur_ax()
    if not hasattr(ax, "set_zlim"):
        return
    if lo is None:
        return ax.get_zlim()
    ax.set_zlim(float(lo), float(hi))
    plt.draw()
    plt.pause(0.01)


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
    fig = _cur_fig()
    mappable = getattr(fig, '_forge_last_mappable', None)
    if mappable is not None:
        plt.colorbar(mappable, ax=_cur_ax(), **kwargs)
    else:
        plt.colorbar(**kwargs)
    plt.draw()
    plt.pause(0.01)
    plt.draw()
    plt.pause(0.01)


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
    return fig


def forge_subplot(m, n, p):
    m_int = int(_to_np(m).flat[0]) if hasattr(m, '__array__') else int(m)
    n_int = int(_to_np(n).flat[0]) if hasattr(n, '__array__') else int(n)
    p_int = int(_to_np(p).flat[0]) if hasattr(p, '__array__') else int(p)
    fig = _cur_fig()
    ax = fig.add_subplot(m_int, n_int, p_int)
    plt.sca(ax)
    plt.draw()
    plt.pause(0.01)
    return ax


def forge_gca():
    return _cur_ax()


def forge_gcf():
    return _cur_fig()


def forge_cla():
    _cur_ax().cla()
    plt.draw()
    plt.pause(0.01)


def forge_clf():
    _cur_fig().clf()
    plt.draw()
    plt.pause(0.01)


def forge_close(n=None):
    if n is None:
        plt.close()
    elif n == "all":
        plt.close("all")
    else:
        plt.close(int(n))


def forge_saveas(filename, fmt=None):
    """Save current figure to *filename*."""
    if fmt:
        _cur_fig().savefig(filename, format=fmt)
    else:
        _cur_fig().savefig(filename)


def forge_print_fig(filename, *args):
    """Octave-compatible 'print' command (saves to file)."""
    dpi = 150
    for a in args:
        if isinstance(a, str) and a.startswith("-r"):
            dpi = int(a[2:])
    _cur_fig().savefig(filename, dpi=dpi)


# ===================================================================
# Registry
# ===================================================================

def forge_drawnow():
    """Force figure update."""
    plt.draw()
    plt.pause(0.01)


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
    "drawnow":      forge_drawnow,
    "pause":        forge_pause,
}
