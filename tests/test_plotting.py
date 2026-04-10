# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for plotting functions."""
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def cleanup_plots():
    plt.close('all')
    yield
    plt.close('all')


class TestPlot2D:
    """R-PLOT-01: The system SHALL render 2D plot types (line, scatter, bar,
    histogram) from numeric array inputs using plot, scatter, bar, and hist
    commands.

    Model-user argument: The migrating engineer expects plot(x,y) to produce
    a line chart instantly, scatter(x,y) to show point clouds, bar(x) to
    display categorical comparisons, and hist(data) to visualize distributions.
    These four verbs cover the vast majority of day-to-day visualization work
    and must each produce a visible figure from raw numeric vectors.

    Decomposition:
      R-PLOT-01a: plot(x, y) creates a figure with at least one axes object
                  containing the line data.
      R-PLOT-01b: scatter(x, y) creates a scatter plot without error.
      R-PLOT-01c: bar(x) creates a bar chart without error.
      R-PLOT-01d: hist(data) creates a histogram without error.

    Consistency argument: The four sub-requirements cover each of the four 2D
    plot types named in the parent requirement. Each sub-requirement exercises
    a distinct plotting function on numeric array input, and together they
    exhaust the set of plot types specified.
    """

    def test_plot_basic(self):
        """R-PLOT-01a: plot(x, y) creates a figure with populated axes."""
        from forge.engine.builtins.plotting import forge_figure, forge_plot, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_plot(np.array([1, 2, 3]), np.array([4, 5, 6]))
        fig = plt.gcf()
        assert len(fig.axes) > 0
        forge_close()

    def test_scatter(self):
        """R-PLOT-01b: scatter(x, y) creates a scatter plot without error."""
        from forge.engine.builtins.plotting import forge_figure, forge_scatter, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_scatter(np.array([1, 2, 3]), np.array([4, 5, 6]))
        forge_close()

    def test_bar(self):
        """R-PLOT-01c: bar(x) creates a bar chart without error."""
        from forge.engine.builtins.plotting import forge_figure, forge_bar, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_bar(np.array([1, 2, 3]))
        forge_close()

    def test_histogram(self):
        """R-PLOT-01d: hist(data) creates a histogram without error."""
        from forge.engine.builtins.plotting import forge_figure, forge_histogram, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_histogram(np.random.randn(100))
        forge_close()


class TestPlotFormatting:
    """R-PLOT-02: The system SHALL allow the user to annotate and configure
    plot appearance using title, xlabel, ylabel, grid, xlim, and ylim commands.

    Model-user argument: After generating a plot, the engineer immediately
    labels axes, sets a title, toggles the grid, and adjusts axis limits to
    frame the data for a report or presentation. These formatting commands are
    typed interactively right after the plot call. If any of them silently
    fails, the exported figure is unusable and the user loses trust in the tool.

    Decomposition:
      R-PLOT-02a: title(str) sets the axes title to the given string.
      R-PLOT-02b: xlabel(str) and ylabel(str) set the respective axis labels.
      R-PLOT-02c: grid() toggles the grid without error.
      R-PLOT-02d: xlim(a, b) and ylim(a, b) set the axis limits correctly.

    Consistency argument: The sub-requirements cover every formatting command
    listed in the parent requirement. R-PLOT-02a handles title, R-PLOT-02b
    handles both axis labels, R-PLOT-02c handles grid, and R-PLOT-02d handles
    both axis limit commands. Together they exhaust the specified set.
    """

    def test_title(self):
        """R-PLOT-02a: title(str) sets the axes title string."""
        from forge.engine.builtins.plotting import forge_figure, forge_title, forge_close
        forge_figure()
        forge_title('Test Title')
        ax = plt.gca()
        assert ax.get_title() == 'Test Title'
        forge_close()

    def test_xlabel_ylabel(self):
        """R-PLOT-02b: xlabel and ylabel set the respective axis labels."""
        from forge.engine.builtins.plotting import forge_figure, forge_xlabel, forge_ylabel, forge_close
        forge_figure()
        forge_xlabel('X')
        forge_ylabel('Y')
        ax = plt.gca()
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'
        forge_close()

    def test_grid(self):
        """R-PLOT-02c: grid() toggles the grid without error."""
        from forge.engine.builtins.plotting import forge_figure, forge_grid, forge_close
        forge_figure()
        forge_grid()
        forge_close()

    def test_xlim_ylim(self):
        """R-PLOT-02d: xlim and ylim set the axis limits correctly."""
        from forge.engine.builtins.plotting import forge_figure, forge_xlim, forge_ylim, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_xlim(0, 10)
        forge_ylim(-1, 1)
        ax = plt.gca()
        assert ax.get_xlim() == (0.0, 10.0)
        forge_close()


class TestFigureManagement:
    """R-PLOT-03: The system SHALL support figure lifecycle management through
    figure, subplot, clf, and saveas commands.

    Model-user argument: The engineer creates a figure, arranges multiple
    subplots for side-by-side comparison, clears stale content with clf before
    re-plotting, and exports the final result with saveas for inclusion in
    reports. These four operations define the complete lifecycle of a plotting
    session. If figure creation, layout, clearing, or export breaks, the
    entire visualization workflow is blocked.

    Decomposition:
      R-PLOT-03a: figure() creates a new figure object.
      R-PLOT-03b: subplot(m, n, k) creates the specified subplot layout.
      R-PLOT-03c: clf() clears all axes from the current figure.
      R-PLOT-03d: saveas(path) writes the figure to a file on disk.

    Consistency argument: The four sub-requirements map one-to-one to the four
    lifecycle commands in the parent requirement. R-PLOT-03a covers creation,
    R-PLOT-03b covers layout, R-PLOT-03c covers clearing, and R-PLOT-03d
    covers export. Together they span the full figure lifecycle.
    """

    def test_figure_create(self):
        """R-PLOT-03a: figure() creates a non-null figure object."""
        from forge.engine.builtins.plotting import forge_figure, forge_close
        forge_figure()
        assert plt.gcf() is not None
        forge_close()

    def test_subplot(self):
        """R-PLOT-03b: subplot(m, n, k) produces the correct number of axes."""
        from forge.engine.builtins.plotting import forge_figure, forge_subplot, forge_close
        forge_figure()
        forge_subplot(2, 1, 1)
        forge_subplot(2, 1, 2)
        fig = plt.gcf()
        assert len(fig.axes) == 2
        forge_close()

    def test_clf(self):
        """R-PLOT-03c: clf() removes all axes from the current figure."""
        from forge.engine.builtins.plotting import forge_figure, forge_plot, forge_clf, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_plot(np.array([1, 2]), np.array([3, 4]))
        forge_clf()
        assert len(plt.gcf().axes) == 0
        forge_close()

    def test_saveas(self, tmp_path):
        """R-PLOT-03d: saveas(path) writes the figure to a file on disk."""
        from forge.engine.builtins.plotting import forge_figure, forge_plot, forge_saveas, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_plot(np.array([1, 2, 3]), np.array([1, 4, 9]))
        outpath = str(tmp_path / 'test.png')
        forge_saveas(outpath)
        import os
        assert os.path.exists(outpath)
        forge_close()


class TestCloseAll:
    """R-POLISH-26: The close command shall successfully handle ForgeChar
    arguments, so that close all closes all figure windows without raising
    a ValueError.

    Model-user argument: The golden user opens multiple figures during an
    interactive session, then types close all to clear the visual workspace
    before starting a new analysis. Currently this crashes with
    ValueError: Truth value of non-scalar ForgeArray is ambiguous because
    the all argument is passed as a ForgeChar and the equality check
    n == "all" returns a non-scalar ForgeArray. This breaks the most
    common multi-figure cleanup workflow.

    Decomposition:
      R-POLISH-26-A: forge_close(ForgeChar("all")) closes all matplotlib
                     figures and sets _close_all_requested without error.
      R-POLISH-26-B: forge_close() with no argument closes the current
                     figure without error (existing behavior preserved).

    Consistency argument: R-POLISH-26-A covers the ForgeChar "all" path
    which is the specific crashing case. R-POLISH-26-B verifies the
    no-argument path is not regressed. Together they cover the two most
    common close() usages.
    """

    def test_close_all_with_forgechar_does_not_raise(self):
        """R-POLISH-26-A: forge_close(ForgeChar("all")) shall not raise ValueError."""
        from forge.engine.builtins.plotting import forge_figure, forge_close
        from forge.engine.containers import ForgeChar
        import forge.engine.builtins.plotting as _pm
        forge_figure()
        forge_figure()
        _pm._close_all_requested = False
        forge_close(ForgeChar("all"))
        assert _pm._close_all_requested is True

    def test_close_all_with_forgechar_closes_all_figures(self):
        """R-POLISH-26-A: forge_close(ForgeChar("all")) closes all matplotlib figures."""
        from forge.engine.builtins.plotting import forge_figure, forge_close
        from forge.engine.containers import ForgeChar
        forge_figure()
        forge_figure()
        assert len(plt.get_fignums()) >= 2
        forge_close(ForgeChar("all"))
        assert len(plt.get_fignums()) == 0

    def test_close_no_arg_not_regressed(self):
        """R-POLISH-26-B: forge_close() with no argument still closes current figure."""
        from forge.engine.builtins.plotting import forge_figure, forge_close
        forge_figure()
        count_before = len(plt.get_fignums())
        forge_close()
        assert len(plt.get_fignums()) == count_before - 1
