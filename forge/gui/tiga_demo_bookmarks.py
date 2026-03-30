# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""TIGA demo bookmarks — opens key files and places bookmarks for a guided tour."""

import os

TIGA_DIR = os.path.expanduser("~/forge/ForgeHome/tiga")

# (filename, line_number) pairs tracing the IGA pipeline
BOOKMARKS = [
    ("findspan.m", 1),  # Knot span search (Algorithm A2.1, The NURBS Book)
    ("findspan.m", 14),  # Binary search loop
    ("basisfun.m", 1),  # Basis function evaluation (Algorithm A2.2)
    ("basisfun.m", 14),  # Cox-de Boor recursion core
    ("gaussQuad.m", 1),  # Gauss-Legendre quadrature points and weights
    ("tiga_poisson1d.m", 10),  # Problem setup: knot vector, degree, DOFs
    ("tiga_poisson1d.m", 29),  # Call findspan + basisfun for partition of unity check
    ("tiga_poisson1d.m", 71),  # Assemble stiffness matrix K and load vector F
    ("tiga_poisson1d.m", 98),  # Inner loop: findspan + derbasisfun for element assembly
    ("tiga_poisson1d.m", 120),  # Apply Dirichlet boundary conditions
    ("tiga_poisson1d.m", 131),  # Solve K*d = F
    ("tiga_2d_poisson.m", 1),  # 2D Poisson: tensor-product B-splines
    ("tiga_2d_poisson.m", 30),  # Assemble 1D stiffness and mass matrices
    ("tiga_2d_poisson.m", 59),  # Kronecker product: K_2D = K1⊗M1 + M1⊗K1
    ("tiga_2d_poisson.m", 110),  # Apply 2D boundary conditions
    ("tiga_2d_poisson.m", 138),  # Solve 2D system
    ("tiga_2d_poisson.m", 219),  # Surface plot of IGA vs exact solution
    ("tiga_convergence.m", 1),  # h-refinement convergence study
    ("tiga_convergence.m", 14),  # Loop over degrees p=1,2,3
]

# Files to open (in tab order)
FILES_TO_OPEN = [
    "findspan.m",
    "basisfun.m",
    "gaussQuad.m",
    "tiga_poisson1d.m",
    "tiga_2d_poisson.m",
    "tiga_convergence.m",
]


def setup_demo(main_window):
    """Open TIGA files, place bookmarks, and refresh the bookmarks panel."""
    ew = main_window.editor_widget

    # Open each file
    for fname in FILES_TO_OPEN:
        fpath = os.path.join(TIGA_DIR, fname)
        if os.path.exists(fpath):
            ew.open_file(fpath)

    # Place bookmarks
    for fname, line in BOOKMARKS:
        fpath = os.path.join(TIGA_DIR, fname)
        # Find the tab with this file
        for idx in range(ew.tabs.count()):
            editor = ew.tabs.widget(idx)
            if getattr(editor, 'file_path', None) and getattr(editor, 'file_path', None) == fpath:
                if not hasattr(editor, '_bookmarks'):
                    editor._bookmarks = set()
                editor._bookmarks.add(line)
                editor.viewport().update()
                break

    # Refresh bookmarks panel and show it
    if hasattr(main_window, '_bookmarks_panel'):
        main_window._bookmarks_panel.refresh()
    if hasattr(main_window, '_bookmarks_dock'):
        main_window._bookmarks_dock.setVisible(True)
        main_window._bookmarks_dock.raise_()

    # Switch to the first solver file
    for idx in range(ew.tabs.count()):
        editor = ew.tabs.widget(idx)
        if getattr(editor, 'file_path', '') and getattr(editor, 'file_path', '').endswith('tiga_poisson1d.m'):
            ew.tabs.setCurrentIndex(idx)
            break
