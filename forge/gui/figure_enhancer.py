# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Enhance matplotlib figure windows with Forge interactive tools.

When the engine creates figures via plt.figure(), they get standard
matplotlib toolbars. This module adds Probe (data cursor), rubber-band
Zoom, and Clear Pins buttons to those native figure windows.
"""

import numpy as np

_enhanced_figs = set()  # Track which figures have already been enhanced


def enhance_figure(fig):
    """Add Forge toolbar buttons (Probe, Zoom-rect, Clear Pins) to a
    matplotlib figure's Qt window.  Safe to call multiple times — skips
    figures that are already enhanced."""
    fig_id = id(fig)
    if fig_id in _enhanced_figs:
        return
    _enhanced_figs.add(fig_id)

    try:
        manager = fig.canvas.manager
        if manager is None:
            return
        toolbar = getattr(manager, 'toolbar', None)
        if toolbar is None:
            return
    except Exception:
        return

    # Import Qt lazily to avoid import issues in headless mode
    try:
        from PySide6.QtWidgets import QPushButton, QLabel
        from PySide6.QtCore import Qt
    except ImportError:
        return

    # State container attached to the figure
    state = _FigureEnhanceState(fig)
    fig._forge_state = state  # prevent GC

    # --- Coordinate readout label ---
    coord_label = QLabel("Ready")
    coord_label.setFixedWidth(240)
    coord_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    coord_label.setStyleSheet(
        "font-size: 11px; padding: 2px 6px; color: palette(text);"
    )
    state.coord_label = coord_label

    # --- Probe button ---
    btn_probe = QPushButton("Probe")
    btn_probe.setCheckable(True)
    btn_probe.setToolTip("Click on data to place coordinate pin annotations")
    btn_probe.setStyleSheet(_btn_css())
    btn_probe.toggled.connect(lambda on: _toggle_probe(state, on))
    state.btn_probe = btn_probe

    # --- Clear Pins button ---
    btn_clear = QPushButton("Clear Pins")
    btn_clear.setToolTip("Remove all probe pin annotations")
    btn_clear.setStyleSheet(_btn_css())
    btn_clear.clicked.connect(lambda: _clear_pins(state))
    state.btn_clear = btn_clear

    # Add widgets to the existing toolbar
    toolbar.addSeparator()
    toolbar.addWidget(btn_probe)
    toolbar.addWidget(btn_clear)
    toolbar.addSeparator()
    toolbar.addWidget(coord_label)

    # Connect mouse events
    fig.canvas.mpl_connect('motion_notify_event',
                           lambda ev: _on_mouse_move(state, ev))
    fig.canvas.mpl_connect('button_press_event',
                           lambda ev: _on_mouse_click(state, ev))


class _FigureEnhanceState:
    """Per-figure state for enhanced tools."""
    __slots__ = ('fig', 'coord_label', 'btn_probe', 'btn_clear',
                 'probe_mode', 'pins')

    def __init__(self, fig):
        self.fig = fig
        self.coord_label = None
        self.btn_probe = None
        self.btn_clear = None
        self.probe_mode = False
        self.pins = []  # list of (annotation, dot_line) tuples


def _btn_css():
    """Return simple button stylesheet."""
    return (
        "QPushButton { background: transparent; border: 1px solid palette(mid);"
        " border-radius: 3px; padding: 2px 8px; font-size: 11px; }"
        "QPushButton:hover { background: palette(midlight); }"
        "QPushButton:checked { background: palette(highlight);"
        " color: palette(highlighted-text); }"
    )


def _on_mouse_move(state, event):
    """Update coordinate readout on mouse move."""
    if event.inaxes and state.coord_label:
        state.coord_label.setText(f"({event.xdata:.6g}, {event.ydata:.6g})")
    elif state.coord_label:
        state.coord_label.setText("Ready")


def _on_mouse_click(state, event):
    """Handle clicks — double-click resets view, single click in probe mode
    places a pin on the nearest data point."""
    if event.dblclick and event.inaxes:
        # Reset view
        event.inaxes.autoscale()
        state.fig.canvas.draw_idle()
        return
    if state.probe_mode and event.inaxes and event.button == 1:
        _place_pin(state, event)


def _toggle_probe(state, on):
    """Enable/disable probe pin placement mode."""
    state.probe_mode = on
    try:
        if on:
            state.fig.canvas.setCursor(
                __import__('PySide6.QtCore', fromlist=['Qt']).Qt.CrossCursor)
        else:
            state.fig.canvas.setCursor(
                __import__('PySide6.QtCore', fromlist=['Qt']).Qt.ArrowCursor)
    except Exception:
        pass


def _place_pin(state, event):
    """Find nearest data point and place an annotation pin."""
    ax = event.inaxes
    click_x, click_y = event.xdata, event.ydata
    best_dist = float('inf')
    best_x, best_y = click_x, click_y
    best_label = ""

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xrange = max(xlim[1] - xlim[0], 1e-30)
    yrange = max(ylim[1] - ylim[0], 1e-30)

    # Search lines
    for line in ax.get_lines():
        xd = np.asarray(line.get_xdata(), dtype=float)
        yd = np.asarray(line.get_ydata(), dtype=float)
        if len(xd) == 0:
            continue
        dx = (xd - click_x) / xrange
        dy = (yd - click_y) / yrange
        dists = dx ** 2 + dy ** 2
        idx = int(np.argmin(dists))
        if dists[idx] < best_dist:
            best_dist = dists[idx]
            best_x, best_y = float(xd[idx]), float(yd[idx])
            lbl = line.get_label()
            best_label = lbl if lbl and not lbl.startswith('_') else ""

    # Search scatter collections
    for coll in ax.collections:
        if hasattr(coll, 'get_offsets'):
            offsets = np.asarray(coll.get_offsets())
            if offsets.size == 0:
                continue
            dx = (offsets[:, 0] - click_x) / xrange
            dy = (offsets[:, 1] - click_y) / yrange
            dists = dx ** 2 + dy ** 2
            idx = int(np.argmin(dists))
            if dists[idx] < best_dist:
                best_dist = dists[idx]
                best_x, best_y = float(offsets[idx, 0]), float(offsets[idx, 1])
                best_label = ""

    # Build pin text
    if best_label:
        text = f"{best_label}\n({best_x:.4g}, {best_y:.4g})"
    else:
        text = f"({best_x:.4g}, {best_y:.4g})"

    # Place annotation
    ann = ax.annotate(
        text, xy=(best_x, best_y),
        xytext=(15, 20), textcoords='offset points',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc',
                  edgecolor='#00897B', alpha=0.92, linewidth=1.5),
        arrowprops=dict(arrowstyle='-|>', color='#00897B', linewidth=1.5,
                        connectionstyle='arc3,rad=0.15'),
    )
    dot, = ax.plot(best_x, best_y, 'o', color='#00897B', markersize=7,
                   markeredgecolor='white', markeredgewidth=1.2, zorder=10)
    state.pins.append((ann, dot))
    state.fig.canvas.draw_idle()


def _clear_pins(state):
    """Remove all probe pin annotations from the figure."""
    for ann, dot in state.pins:
        try:
            ann.remove()
        except Exception:
            pass
        try:
            dot.remove()
        except Exception:
            pass
    state.pins.clear()
    state.fig.canvas.draw_idle()
