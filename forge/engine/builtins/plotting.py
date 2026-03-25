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
    return np.asarray(_unwrap(x), dtype=float)


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

def forge_plot(*args):
    """plot(y), plot(x,y), plot(x,y,fmt), plot(x1,y1,fmt1, x2,y2,fmt2,...)."""
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
        if i + 1 < len(a) and isinstance(a[i + 1], str):
            ax.plot(_to_np(a[i]), **_parse_linespec(a[i + 1]))
            i += 2
        elif i + 2 < len(a) and isinstance(a[i + 2], str):
            ax.plot(_to_np(a[i]), _to_np(a[i + 1]), **_parse_linespec(a[i + 2]))
            i += 3
        elif i + 1 < len(a) and not isinstance(a[i + 1], str):
            ax.plot(_to_np(a[i]), _to_np(a[i + 1]))
            i += 2
        else:
            ax.plot(_to_np(a[i]))
            i += 1
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
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")
    return ax


def forge_plot3(x, y, z, **kwargs):
    ax = _get_3d_ax()
    ax.plot(_to_np(x), _to_np(y), _to_np(z), **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_scatter3(x, y, z, **kwargs):
    ax = _get_3d_ax()
    ax.scatter(_to_np(x), _to_np(y), _to_np(z), **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_surf(X, Y, Z, **kwargs):
    ax = _get_3d_ax()
    ax.plot_surface(_to_np(X), _to_np(Y), _to_np(Z), **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_surfc(X, Y, Z, **kwargs):
    ax = _get_3d_ax()
    ax.plot_surface(_to_np(X), _to_np(Y), _to_np(Z), alpha=0.7, **kwargs)
    ax.contour(_to_np(X), _to_np(Y), _to_np(Z), zdir="z",
               offset=_to_np(Z).min())
    plt.draw()
    plt.pause(0.01)


def forge_surfl(X, Y, Z, **kwargs):
    ax = _get_3d_ax()
    ax.plot_surface(_to_np(X), _to_np(Y), _to_np(Z),
                    rstride=1, cstride=1, shade=True, **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_mesh(X, Y, Z, **kwargs):
    ax = _get_3d_ax()
    ax.plot_wireframe(_to_np(X), _to_np(Y), _to_np(Z), **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_meshc(X, Y, Z, **kwargs):
    ax = _get_3d_ax()
    ax.plot_wireframe(_to_np(X), _to_np(Y), _to_np(Z), **kwargs)
    ax.contour(_to_np(X), _to_np(Y), _to_np(Z), zdir="z",
               offset=_to_np(Z).min())
    plt.draw()
    plt.pause(0.01)


def forge_meshz(X, Y, Z, **kwargs):
    ax = _get_3d_ax()
    Zn = _to_np(Z)
    ax.plot_wireframe(_to_np(X), _to_np(Y), Zn, **kwargs)
    # "curtain" along edges
    ax.plot_surface(_to_np(X), _to_np(Y), np.zeros_like(Zn), alpha=0.1)
    plt.draw()
    plt.pause(0.01)


def forge_contour(X, Y, Z, *args, **kwargs):
    _maybe_clear()
    _cur_ax().contour(_to_np(X), _to_np(Y), _to_np(Z), *args, **kwargs)
    plt.draw()
    plt.pause(0.01)


def forge_contourf(X, Y, Z, *args, **kwargs):
    _maybe_clear()
    _cur_ax().contourf(_to_np(X), _to_np(Y), _to_np(Z), *args, **kwargs)
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
    _cur_ax().legend(*args, **kwargs)
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
    plt.colorbar(**kwargs)
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
        return plt.figure()
    return plt.figure(int(n))


def forge_subplot(m, n, p):
    return _cur_fig().add_subplot(int(m), int(n), int(p))


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
