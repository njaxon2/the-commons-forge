"""Forge GUI Theme System – Premium Edition.

Target location: forge/gui/themes.py

Provides three polished QSS themes (dark, light, midnight), theme
application, syntax-highlight colour palettes for the code editor,
and a user-configurable preferences overlay system.

Accent family: Teal / Cyan (#00BCD4)
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional


# =====================================================================
# Colour Palettes  (centralised so every theme is self-consistent)
# =====================================================================

_DARK = {
    "bg0":       "#1e1e2e",   # deepest background
    "bg1":       "#252536",   # panels / docks
    "bg2":       "#2a2a3c",   # editor background
    "bg3":       "#313145",   # toolbar / menubar gradient start
    "bg4":       "#3a3a50",   # toolbar gradient end / hover bg
    "bg5":       "#44445a",   # active/pressed bg
    "fg0":       "#cdd6f4",   # primary text
    "fg1":       "#bac2de",   # secondary text
    "fg2":       "#a6adc8",   # muted text
    "fg3":       "#6c7086",   # disabled / placeholder
    "border0":   "#313145",   # subtle border
    "border1":   "#44445a",   # stronger border
    "accent":    "#00BCD4",   # primary accent (teal)
    "accent_h":  "#00E5FF",   # accent hover (brighter cyan)
    "accent_p":  "#0097A7",   # accent pressed (deeper teal)
    "accent_dim":"#00838F",   # accent muted
    "accent_bg": "#003d4d",   # accent background tint
    "error":     "#f38ba8",
    "warning":   "#fab387",
    "success":   "#a6e3a1",
    "info":      "#89b4fa",
    "selection":  "#264f78",
    "cur_line":   "#2a2a3c",
    "scrollbar":  "rgba(166,173,200,0.18)",
    "scrollbar_h":"rgba(166,173,200,0.35)",
    "scrollbar_p":"rgba(166,173,200,0.50)",
    "shadow":     "rgba(0,0,0,0.28)",
    "alt_row":    "#282840",
    "tab_active": "#1e1e2e",
    "tab_inactive":"#252536",
    "tooltip_bg": "#363650",
    "tooltip_bd": "#44445a",
}

_LIGHT = {
    "bg0":       "#f8f9fc",
    "bg1":       "#eef0f5",
    "bg2":       "#ffffff",
    "bg3":       "#e8eaf0",
    "bg4":       "#dce0e8",
    "bg5":       "#ccd0da",
    "fg0":       "#1e1e2e",
    "fg1":       "#4c4f69",
    "fg2":       "#6c6f85",
    "fg3":       "#9ca0b0",
    "border0":   "#dce0e8",
    "border1":   "#bcc0cc",
    "accent":    "#00897B",
    "accent_h":  "#00BCD4",
    "accent_p":  "#00695C",
    "accent_dim":"#4DB6AC",
    "accent_bg": "#e0f7fa",
    "error":     "#d32f2f",
    "warning":   "#e65100",
    "success":   "#2e7d32",
    "info":      "#1565c0",
    "selection":  "#b2ebf2",
    "cur_line":   "#f1f3f8",
    "scrollbar":  "rgba(76,79,105,0.15)",
    "scrollbar_h":"rgba(76,79,105,0.30)",
    "scrollbar_p":"rgba(76,79,105,0.45)",
    "shadow":     "rgba(0,0,0,0.10)",
    "alt_row":    "#f2f4f8",
    "tab_active": "#ffffff",
    "tab_inactive":"#eef0f5",
    "tooltip_bg": "#ffffff",
    "tooltip_bd": "#bcc0cc",
}

_MIDNIGHT = {
    "bg0":       "#0d0d18",
    "bg1":       "#121222",
    "bg2":       "#16162a",
    "bg3":       "#1a1a30",
    "bg4":       "#222240",
    "bg5":       "#2a2a4a",
    "fg0":       "#c0c8e0",
    "fg1":       "#a0a8c0",
    "fg2":       "#8088a8",
    "fg3":       "#505878",
    "border0":   "#1a1a30",
    "border1":   "#2a2a4a",
    "accent":    "#00BCD4",
    "accent_h":  "#18FFFF",
    "accent_p":  "#0097A7",
    "accent_dim":"#006064",
    "accent_bg": "#002830",
    "error":     "#ff6b6b",
    "warning":   "#ffa94d",
    "success":   "#69db7c",
    "info":      "#74c0fc",
    "selection":  "#1a3a5c",
    "cur_line":   "#16162a",
    "scrollbar":  "rgba(160,168,192,0.12)",
    "scrollbar_h":"rgba(160,168,192,0.25)",
    "scrollbar_p":"rgba(160,168,192,0.40)",
    "shadow":     "rgba(0,0,0,0.45)",
    "alt_row":    "#141428",
    "tab_active": "#0d0d18",
    "tab_inactive":"#121222",
    "tooltip_bg": "#1e1e38",
    "tooltip_bd": "#2a2a4a",
}


# =====================================================================
# QSS Template  (shared by all three themes, interpolated per-palette)
# =====================================================================

def _build_qss(c: dict) -> str:
    """Build a complete QSS stylesheet from a colour palette dict."""
    return f"""
/* ================================================================
   Forge IDE – Premium Theme
   Generated from palette. Do not edit directly.
   ================================================================ */

/* ── Global Reset & Defaults ─────────────────────────────────── */
* {{
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue",
                 "Ubuntu", "Noto Sans", Arial, sans-serif;
    font-size: 13px;
    outline: none;
}}

QMainWindow {{
    background-color: {c["bg0"]};
    color: {c["fg0"]};
}}

QWidget {{
    background-color: transparent;
    color: {c["fg0"]};
}}

QDialog {{
    background-color: {c["bg1"]};
    color: {c["fg0"]};
    border-radius: 10px;
}}

/* ── Menu Bar ────────────────────────────────────────────────── */
QMenuBar {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 {c["bg3"]}, stop:1 {c["bg1"]});
    color: {c["fg1"]};
    border-bottom: 1px solid {c["border0"]};
    padding: 2px 0px;
    spacing: 1px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 6px 12px;
    border-radius: 6px;
    margin: 2px 1px;
}}

QMenuBar::item:selected {{
    background-color: {c["accent_bg"]};
    color: {c["accent_h"]};
}}

QMenuBar::item:pressed {{
    background-color: {c["accent"]};
    color: #ffffff;
}}

/* ── Menus (Dropdown) ────────────────────────────────────────── */
QMenu {{
    background-color: {c["bg1"]};
    color: {c["fg0"]};
    border: 1px solid {c["border1"]};
    border-radius: 8px;
    padding: 6px 0px;
}}

QMenu::item {{
    padding: 8px 32px 8px 20px;
    border-radius: 4px;
    margin: 2px 6px;
}}

QMenu::item:selected {{
    background-color: {c["accent_bg"]};
    color: {c["accent_h"]};
}}

QMenu::item:disabled {{
    color: {c["fg3"]};
}}

QMenu::separator {{
    height: 1px;
    background-color: {c["border0"]};
    margin: 6px 12px;
}}

QMenu::indicator {{
    width: 16px;
    height: 16px;
    margin-left: 6px;
}}

QMenu::icon {{
    margin-left: 8px;
}}

QMenu::right-arrow {{
    margin-right: 8px;
}}

/* ── Toolbars ────────────────────────────────────────────────── */
QToolBar {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 {c["bg3"]}, stop:1 {c["bg1"]});
    border-bottom: 1px solid {c["border0"]};
    spacing: 2px;
    padding: 3px 6px;
}}

QToolBar::separator {{
    width: 1px;
    background-color: {c["border0"]};
    margin: 4px 6px;
}}

QToolBar::handle {{
    background: {c["border1"]};
    width: 2px;
    margin: 4px 2px;
    border-radius: 1px;
}}

QToolButton {{
    background: transparent;
    color: {c["fg1"]};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 5px 8px;
    margin: 1px;
}}

QToolButton:hover {{
    background-color: {c["accent_bg"]};
    border: 1px solid {c["accent_dim"]};
    color: {c["accent_h"]};
}}

QToolButton:pressed {{
    background-color: {c["accent"]};
    color: #ffffff;
}}

QToolButton:checked {{
    background-color: {c["accent_bg"]};
    border: 1px solid {c["accent"]};
    color: {c["accent"]};
}}

QToolButton:disabled {{
    color: {c["fg3"]};
}}

QToolButton[popupMode="1"] {{
    padding-right: 18px;
}}

QToolButton::menu-button {{
    border: none;
    border-left: 1px solid {c["border0"]};
    width: 16px;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}}

/* ── Push Buttons ────────────────────────────────────────────── */
QPushButton {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 {c["accent"]}, stop:1 {c["accent_p"]});
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 7px 20px;
    font-weight: 600;
    min-height: 18px;
}}

QPushButton:hover {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 {c["accent_h"]}, stop:1 {c["accent"]});
}}

QPushButton:pressed {{
    background-color: {c["accent_p"]};
}}

QPushButton:disabled {{
    background-color: {c["bg4"]};
    color: {c["fg3"]};
}}

QPushButton:flat {{
    background: transparent;
    color: {c["accent"]};
    border: none;
}}

QPushButton:flat:hover {{
    background-color: {c["accent_bg"]};
    color: {c["accent_h"]};
}}

QPushButton#secondaryButton,
QPushButton[secondary="true"] {{
    background: transparent;
    color: {c["fg1"]};
    border: 1px solid {c["border1"]};
}}

QPushButton#secondaryButton:hover,
QPushButton[secondary="true"]:hover {{
    background-color: {c["bg4"]};
    border-color: {c["accent"]};
    color: {c["accent"]};
}}

/* ── Line Edit / Text Edit / Plain Text ──────────────────────── */
QLineEdit {{
    background-color: {c["bg2"]};
    color: {c["fg0"]};
    border: 1px solid {c["border1"]};
    border-radius: 8px;
    padding: 6px 12px;
    selection-background-color: {c["selection"]};
    selection-color: {c["fg0"]};
}}

QLineEdit:hover {{
    border-color: {c["accent_dim"]};
}}

QLineEdit:focus {{
    border: 2px solid {c["accent"]};
    padding: 5px 11px;
}}

QLineEdit:disabled {{
    background-color: {c["bg3"]};
    color: {c["fg3"]};
}}

QLineEdit:read-only {{
    background-color: {c["bg3"]};
}}

QTextEdit, QPlainTextEdit {{
    background-color: {c["bg2"]};
    color: {c["fg0"]};
    border: 1px solid {c["border0"]};
    border-radius: 8px;
    padding: 4px;
    selection-background-color: {c["selection"]};
    selection-color: {c["fg0"]};
}}

QTextEdit:focus, QPlainTextEdit:focus {{
    border: 2px solid {c["accent"]};
    padding: 3px;
}}

/* ── Tab Widget ──────────────────────────────────────────────── */
QTabWidget {{
    background: transparent;
}}

QTabWidget::pane {{
    background-color: {c["bg0"]};
    border: 1px solid {c["border0"]};
    border-top: none;
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
}}

QTabBar {{
    background: transparent;
    qproperty-drawBase: 0;
}}

QTabBar::tab {{
    background-color: {c["tab_inactive"]};
    color: {c["fg2"]};
    border: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border-bottom: 3px solid transparent;
    padding: 8px 18px;
    margin-right: 2px;
    min-width: 90px;
    font-weight: 500;
}}

QTabBar::tab:hover {{
    background-color: {c["bg4"]};
    color: {c["fg0"]};
    border-bottom: 3px solid {c["accent_dim"]};
}}

QTabBar::tab:selected {{
    background-color: {c["tab_active"]};
    color: {c["accent"]};
    border-bottom: 3px solid {c["accent"]};
    font-weight: 600;
}}

QTabBar::close-button {{
    image: none;
    subcontrol-position: right;
    border-radius: 8px;
    padding: 2px;
    padding: 2px;
    margin-right: 4px;
}}

QTabBar::close-button:hover {{
    background-color: {c["error"]};
}}

QTabBar::scroller {{
    width: 28px;
}}

QTabBar QToolButton {{
    background-color: {c["bg3"]};
    border: none;
    border-radius: 4px;
    margin: 2px;
}}

QTabBar QToolButton:hover {{
    background-color: {c["bg5"]};
}}

/* ── Dock Widgets ────────────────────────────────────────────── */
QDockWidget {{
    color: {c["fg1"]};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 {c["bg3"]}, stop:1 {c["bg1"]});
    color: {c["fg1"]};
    padding: 8px 12px;
    border-bottom: 1px solid {c["border0"]};
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
}}

QDockWidget::close-button,
QDockWidget::float-button {{
    background: transparent;
    border: none;
    padding: 3px;
    border-radius: 4px;
    icon-size: 14px;
}}

QDockWidget::close-button:hover {{
    background-color: {c["error"]};
}}

QDockWidget::float-button:hover {{
    background-color: {c["bg5"]};
}}

/* ── Scroll Bars ─────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {c["scrollbar"]};
    border-radius: 4px;
    min-height: 24px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {c["scrollbar_h"]};
}}

QScrollBar::handle:vertical:pressed {{
    background-color: {c["scrollbar_p"]};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
    border: none;
    background: transparent;
}}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 0;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background-color: {c["scrollbar"]};
    border-radius: 4px;
    min-width: 24px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {c["scrollbar_h"]};
}}

QScrollBar::handle:horizontal:pressed {{
    background-color: {c["scrollbar_p"]};
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
    border: none;
    background: transparent;
}}

QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* ── Scroll Area ─────────────────────────────────────────────── */
QScrollArea {{
    border: none;
    background: transparent;
}}

/* ── Tree View / List View / Table View ──────────────────────── */
QTreeView, QListView, QTableView {{
    background-color: {c["bg2"]};
    color: {c["fg0"]};
    border: 1px solid {c["border0"]};
    border-radius: 8px;
    padding: 2px;
    alternate-background-color: {c["alt_row"]};
    selection-background-color: {c["selection"]};
    selection-color: {c["fg0"]};
    outline: none;
    show-decoration-selected: 1;
}}

QTreeView::item, QListView::item, QTableView::item {{
    padding: 4px 8px;
    border-radius: 4px;
    margin: 1px 2px;
}}

QTreeView::item:hover, QListView::item:hover {{
    background-color: {c["accent_bg"]};
}}

QTreeView::item:selected, QListView::item:selected,
QTableView::item:selected {{
    background-color: {c["selection"]};
    color: {c["fg0"]};
}}

QTreeView::branch {{
    background: transparent;
}}

QTreeView::branch:has-siblings:!adjoins-item {{
    border-image: none;
}}

QTreeView::branch:has-children:!has-siblings:closed,
QTreeView::branch:closed:has-children:has-siblings {{
    border-image: none;
    image: none;
}}

QTreeView::branch:open:has-children:!has-siblings,
QTreeView::branch:open:has-children:has-siblings {{
    border-image: none;
    image: none;
}}

/* ── Header View (Table / Tree headers) ──────────────────────── */
QHeaderView {{
    background-color: {c["bg1"]};
    border: none;
}}

QHeaderView::section {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 {c["bg3"]}, stop:1 {c["bg1"]});
    color: {c["fg1"]};
    border: none;
    border-right: 1px solid {c["border0"]};
    border-bottom: 1px solid {c["border0"]};
    padding: 6px 10px;
    font-weight: 600;
    font-size: 12px;
}}

QHeaderView::section:hover {{
    background-color: {c["bg4"]};
    color: {c["accent"]};
}}

QHeaderView::section:checked {{
    background-color: {c["accent_bg"]};
    color: {c["accent"]};
}}

QHeaderView::down-arrow {{
    subcontrol-position: center right;
    right: 8px;
}}

QHeaderView::up-arrow {{
    subcontrol-position: center right;
    right: 8px;
}}

/* ── Combo Box ───────────────────────────────────────────────── */
QComboBox {{
    background-color: {c["bg2"]};
    color: {c["fg0"]};
    border: 1px solid {c["border1"]};
    border-radius: 8px;
    padding: 6px 12px;
    min-height: 18px;
    min-width: 80px;
}}

QComboBox:hover {{
    border-color: {c["accent_dim"]};
}}

QComboBox:focus {{
    border: 2px solid {c["accent"]};
    padding: 5px 11px;
}}

QComboBox:on {{
    border-color: {c["accent"]};
    border-bottom-left-radius: 0px;
    border-bottom-right-radius: 0px;
}}

QComboBox::drop-down {{
    border: none;
    width: 28px;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}}

QComboBox::down-arrow {{
    width: 12px;
    height: 12px;
}}

QComboBox QAbstractItemView {{
    background-color: {c["bg1"]};
    color: {c["fg0"]};
    border: 1px solid {c["border1"]};
    border-radius: 0 0 8px 8px;
    selection-background-color: {c["accent_bg"]};
    selection-color: {c["accent_h"]};
    padding: 4px;
    outline: none;
}}

/* ── Spin Box ────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background-color: {c["bg2"]};
    color: {c["fg0"]};
    border: 1px solid {c["border1"]};
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 18px;
}}

QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {c["accent_dim"]};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 2px solid {c["accent"]};
    padding: 5px 9px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border: none;
    border-left: 1px solid {c["border0"]};
    border-top-right-radius: 8px;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border: none;
    border-left: 1px solid {c["border0"]};
    border-bottom-right-radius: 8px;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {c["bg4"]};
}}

/* ── Check Box ───────────────────────────────────────────────── */
QCheckBox {{
    spacing: 8px;
    color: {c["fg0"]};
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {c["border1"]};
    border-radius: 4px;
    background-color: {c["bg2"]};
}}

QCheckBox::indicator:hover {{
    border-color: {c["accent"]};
}}

QCheckBox::indicator:checked {{
    background-color: {c["accent"]};
    border-color: {c["accent"]};
}}

QCheckBox::indicator:disabled {{
    background-color: {c["bg3"]};
    border-color: {c["border0"]};
}}

/* ── Radio Button ────────────────────────────────────────────── */
QRadioButton {{
    spacing: 8px;
    color: {c["fg0"]};
}}

QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {c["border1"]};
    border-radius: 11px;
    background-color: {c["bg2"]};
}}

QRadioButton::indicator:hover {{
    border-color: {c["accent"]};
}}

QRadioButton::indicator:checked {{
    background-color: {c["accent"]};
    border-color: {c["accent"]};
}}

/* ── Slider ──────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 6px;
    background-color: {c["bg4"]};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {c["accent"]};
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 9px;
    border: 2px solid {c["bg0"]};
}}

QSlider::handle:horizontal:hover {{
    background-color: {c["accent_h"]};
    border: 2px solid {c["accent_bg"]};
}}

QSlider::sub-page:horizontal {{
    background-color: {c["accent"]};
    border-radius: 3px;
}}

QSlider::groove:vertical {{
    width: 6px;
    background-color: {c["bg4"]};
    border-radius: 3px;
}}

QSlider::handle:vertical {{
    background-color: {c["accent"]};
    width: 18px;
    height: 18px;
    margin: 0 -7px;
    border-radius: 9px;
    border: 2px solid {c["bg0"]};
}}

QSlider::handle:vertical:hover {{
    background-color: {c["accent_h"]};
}}

/* ── Progress Bar ────────────────────────────────────────────── */
QProgressBar {{
    background-color: {c["bg4"]};
    border: none;
    border-radius: 6px;
    text-align: center;
    color: {c["fg0"]};
    font-weight: 600;
    font-size: 11px;
    min-height: 12px;
    max-height: 12px;
}}

QProgressBar::chunk {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {c["accent_p"]}, stop:1 {c["accent"]});
    border-radius: 6px;
}}

/* ── Group Box ───────────────────────────────────────────────── */
QGroupBox {{
    background-color: {c["bg1"]};
    border: 1px solid {c["border0"]};
    border-radius: 10px;
    margin-top: 14px;
    padding: 16px 10px 10px 10px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 2px 10px;
    color: {c["accent"]};
    background-color: {c["bg1"]};
    border-radius: 4px;
}}

/* ── Status Bar ──────────────────────────────────────────────── */
QStatusBar {{
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 {c["bg1"]}, stop:1 {c["bg0"]});
    color: {c["fg2"]};
    border-top: 1px solid {c["border0"]};
    padding: 0;
    min-height: 24px;
}}

QStatusBar::item {{
    border: none;
}}

QStatusBar QLabel {{
    color: {c["fg2"]};
    padding: 2px 10px;
    font-size: 12px;
}}

QStatusBar QPushButton {{
    background: transparent;
    color: {c["fg2"]};
    border: none;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    font-weight: normal;
    min-height: 0;
}}

QStatusBar QPushButton:hover {{
    background-color: {c["accent_bg"]};
    color: {c["accent"]};
}}

/* ── Tooltip ─────────────────────────────────────────────────── */
QDialog {{
    background: {c["bg0"]};
    border-radius: 8px;
}}

QToolTip {{
    background-color: {c["tooltip_bg"]};
    color: {c["fg0"]};
    border: 1px solid {c["tooltip_bd"]};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* ── Splitter ────────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {c["border0"]};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

QSplitter::handle:hover {{
    background-color: {c["accent"]};
}}

/* ── Frame ───────────────────────────────────────────────────── */
QFrame {{
    border: none;
}}

QFrame[frameShape="4"],
QFrame[frameShape="5"] {{
    background-color: {c["border0"]};
    max-height: 1px;
}}

/* ── Stacked Widget ──────────────────────────────────────────── */
QStackedWidget {{
    background: transparent;
}}

/* ── Label ───────────────────────────────────────────────────── */
QLabel {{
    background: transparent;
    color: {c["fg0"]};
    border: none;
}}

QLabel[heading="true"] {{
    font-size: 18px;
    font-weight: 700;
    color: {c["fg0"]};
}}

QLabel[subheading="true"] {{
    font-size: 14px;
    font-weight: 600;
    color: {c["fg1"]};
}}

QLabel[muted="true"] {{
    color: {c["fg2"]};
    font-size: 12px;
}}

QLabel:disabled {{
    color: {c["fg3"]};
}}

/* ── Dialog Button Box ───────────────────────────────────────── */
QDialogButtonBox {{
    dialogbuttonbox-buttons-have-icons: 0;
}}

/* ── Input Dialog ────────────────────────────────────────────── */
QInputDialog {{
    background-color: {c["bg1"]};
}}

/* ── File Dialog ─────────────────────────────────────────────── */
QFileDialog {{
    background-color: {c["bg1"]};
}}

QFileDialog QListView,
QFileDialog QTreeView {{
    background-color: {c["bg2"]};
}}

/* ── Message Box ─────────────────────────────────────────────── */
QMessageBox {{
    background-color: {c["bg1"]};
}}

QMessageBox QLabel {{
    color: {c["fg0"]};
    font-size: 13px;
}}

/* ── Wizard ──────────────────────────────────────────────────── */
QWizard {{
    background-color: {c["bg1"]};
}}

/* ── Calendar Widget ─────────────────────────────────────────── */
QCalendarWidget {{
    background-color: {c["bg1"]};
}}

QCalendarWidget QToolButton {{
    color: {c["fg0"]};
    background: transparent;
}}

QCalendarWidget QMenu {{
    background-color: {c["bg1"]};
}}

/* ── Text Browser ────────────────────────────────────────────── */
QTextBrowser {{
    background-color: {c["bg2"]};
    color: {c["fg0"]};
    border: 1px solid {c["border0"]};
    border-radius: 8px;
}}

/* ── Abstract Scroll Area ────────────────────────────────────── */
QAbstractScrollArea {{
    background-color: {c["bg2"]};
    border: 1px solid {c["border0"]};
    border-radius: 8px;
}}

QAbstractScrollArea::corner {{
    background: transparent;
    border: none;
}}

/* ── Size Grip ───────────────────────────────────────────────── */
QSizeGrip {{
    background: transparent;
    width: 16px;
    height: 16px;
}}

/* ── Focus Glow  (via border) ────────────────────────────────── */
QLineEdit:focus,
QTextEdit:focus,
QPlainTextEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus,
QTreeView:focus,
QListView:focus,
QTableView:focus {{
    border: 2px solid {c["accent"]};
}}

/* ── Disabled State (global) ─────────────────────────────────── */
*:disabled {{
    color: {c["fg3"]};
}}
"""


# =====================================================================
# Build theme strings
# =====================================================================

DARK_THEME = _build_qss(_DARK)
LIGHT_THEME = _build_qss(_LIGHT)
MIDNIGHT_THEME = _build_qss(_MIDNIGHT)


# =====================================================================
# User-configurable preferences overlay
# =====================================================================

_PREFS_FILE = os.path.join(
    os.path.expanduser("~"), ".forge", "theme_prefs.json"
)

_DEFAULT_PREFERENCES: Dict[str, Any] = {
    "default_theme":   "dark",
    "font_family":     "",       # empty = use theme default
    "font_size":       0,        # 0 = use theme default (13)
    "accent_color":    "",       # empty = use theme default
    "custom_css":      "",       # appended after the theme QSS
}


def _load_preferences() -> Dict[str, Any]:
    prefs = deepcopy(_DEFAULT_PREFERENCES)
    try:
        if os.path.exists(_PREFS_FILE):
            with open(_PREFS_FILE, "r") as fh:
                user = json.load(fh)
            if isinstance(user, dict):
                for k in _DEFAULT_PREFERENCES:
                    if k in user:
                        prefs[k] = user[k]
    except Exception:
        pass
    return prefs


def save_preferences(prefs: Dict[str, Any]) -> None:
    """Persist user theme preferences to disk."""
    os.makedirs(os.path.dirname(_PREFS_FILE), exist_ok=True)
    with open(_PREFS_FILE, "w") as fh:
        json.dump(prefs, fh, indent=2)


def get_preferences() -> Dict[str, Any]:
    """Return current user theme preferences (merged with defaults)."""
    return _load_preferences()


def _apply_preference_overlay(qss: str, prefs: Dict[str, Any]) -> str:
    """Apply user preference overrides on top of a theme QSS."""
    patches: list[str] = []

    ff = prefs.get("font_family", "")
    fs = prefs.get("font_size", 0)
    if ff or fs:
        rule = "* {"
        if ff:
            rule += f' font-family: "{ff}";'
        if fs and fs > 0:
            rule += f" font-size: {fs}px;"
        rule += " }"
        patches.append(rule)

    ac = prefs.get("accent_color", "")
    if ac:
        patches.append(
            f"QPushButton {{ background-color: {ac}; }}\n"
            f"QTabBar::tab:selected {{ border-bottom-color: {ac}; color: {ac}; }}\n"
            f"QProgressBar::chunk {{ background-color: {ac}; }}\n"
            f"QSlider::handle:horizontal {{ background-color: {ac}; }}\n"
            f"QSlider::sub-page:horizontal {{ background-color: {ac}; }}\n"
        )

    custom = prefs.get("custom_css", "")
    if custom:
        patches.append(custom)

    if patches:
        qss = qss + "\n/* ── User Preference Overrides ──── */\n" + "\n".join(patches)
    return qss


# =====================================================================
# Theme registry
# =====================================================================

THEMES: Dict[str, str] = {
    "dark":     DARK_THEME,
    "light":    LIGHT_THEME,
    "midnight": MIDNIGHT_THEME,
}
_THEMES = THEMES


# =====================================================================
# Syntax highlight colour palettes
# =====================================================================

_EDITOR_COLORS: Dict[str, Dict[str, str]] = {
    "dark": {
        "keyword":      "#00BCD4",   # teal accent
        "builtin":      "#ffb74d",   # amber
        "string":       "#a6e3a1",   # green
        "number":       "#fab387",   # peach
        "comment":      "#6c7086",   # overlay0
        "operator":     "#cdd6f4",   # text
        "function":     "#cba6f7",   # mauve
        "class":        "#94e2d5",   # teal light
        "decorator":    "#f5c2e7",   # pink
        "self":         "#89b4fa",   # blue
        "error":        "#f38ba8",   # red
        "line_number":  "#44445a",   # surface1
        "current_line": "#2a2a3c",   # surface0
        "brace_match":  "#264f78",   # selection
        "selection":    "#264f78",
        "background":   "#1e1e2e",   # base
        "foreground":   "#cdd6f4",   # text
    },
    "light": {
        "keyword":      "#00897B",   # teal
        "builtin":      "#e65100",   # deep orange
        "string":       "#2e7d32",   # green
        "number":       "#d84315",   # deep orange
        "comment":      "#9ca0b0",   # overlay
        "operator":     "#1e1e2e",   # dark
        "function":     "#7b1fa2",   # purple
        "class":        "#00695c",   # dark teal
        "decorator":    "#ad1457",   # pink
        "self":         "#1565c0",   # blue
        "error":        "#d32f2f",   # red
        "line_number":  "#bcc0cc",   # surface2
        "current_line": "#f1f3f8",   # mantle
        "brace_match":  "#b2ebf2",   # selection
        "selection":    "#b2ebf2",
        "background":   "#ffffff",
        "foreground":   "#1e1e2e",
    },
    "midnight": {
        "keyword":      "#18FFFF",   # bright cyan
        "builtin":      "#ffa94d",   # amber
        "string":       "#69db7c",   # green
        "number":       "#ffa94d",   # orange
        "comment":      "#505878",   # muted
        "operator":     "#c0c8e0",   # text
        "function":     "#b197fc",   # violet
        "class":        "#63e6be",   # teal
        "decorator":    "#ffa8c9",   # pink
        "self":         "#74c0fc",   # blue
        "error":        "#ff6b6b",   # red
        "line_number":  "#2a2a4a",   # dark
        "current_line": "#16162a",   # surface
        "brace_match":  "#1a3a5c",   # selection
        "selection":    "#1a3a5c",
        "background":   "#0d0d18",   # deepest
        "foreground":   "#c0c8e0",   # text
    },
}


# =====================================================================
# Public API
# =====================================================================

def get_available_themes() -> List[str]:
    """Return list of available theme names.

    Returns
    -------
    list of str
        Currently ``['dark', 'light', 'midnight']``.
    """
    return list(THEMES.keys())


def apply_theme(app: Any, theme_name: str) -> None:
    """Apply a named theme to a QApplication instance.

    Loads user preferences from ``~/.forge/theme_prefs.json`` and
    applies any overrides (custom font, accent colour, extra CSS)
    on top of the base theme.

    Parameters
    ----------
    app : QApplication
        The running Qt application instance.
    theme_name : str
        One of the names returned by :func:`get_available_themes`.

    Raises
    ------
    ValueError
        If *theme_name* is not recognised.
    """
    theme_name = theme_name.lower()
    if theme_name not in THEMES:
        raise ValueError(
            f"Unknown theme '{theme_name}'. "
            f"Available: {get_available_themes()}"
        )
    qss = THEMES[theme_name]
    prefs = _load_preferences()
    qss = _apply_preference_overlay(qss, prefs)
    app.setStyleSheet(qss)


def get_editor_colors(theme: str) -> Dict[str, str]:
    """Return syntax-highlight colour palette for a theme.

    Parameters
    ----------
    theme : str
        Theme name (``'dark'``, ``'light'``, or ``'midnight'``).

    Returns
    -------
    dict
        Mapping of token type -> hex colour string.

    Raises
    ------
    ValueError
        If *theme* is not recognised.
    """
    theme = theme.lower()
    if theme not in _EDITOR_COLORS:
        raise ValueError(
            f"Unknown theme '{theme}'. "
            f"Available: {get_available_themes()}"
        )
    return dict(_EDITOR_COLORS[theme])


def get_theme_palette(theme: str) -> Dict[str, str]:
    """Return the raw colour palette dict for a theme.

    Useful for programmatic access to theme colours outside of QSS.

    Parameters
    ----------
    theme : str
        Theme name.

    Returns
    -------
    dict
        Full colour palette with keys like ``bg0``, ``accent``, etc.
    """
    _palettes = {"dark": _DARK, "light": _LIGHT, "midnight": _MIDNIGHT}
    theme = theme.lower()
    if theme not in _palettes:
        raise ValueError(
            f"Unknown theme '{theme}'. Available: {list(_palettes.keys())}"
        )
    return dict(_palettes[theme])
