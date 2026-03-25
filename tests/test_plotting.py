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
    def test_plot_basic(self):
        from forge.engine.builtins.plotting import forge_figure, forge_plot, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_plot(np.array([1, 2, 3]), np.array([4, 5, 6]))
        fig = plt.gcf()
        assert len(fig.axes) > 0
        forge_close()

    def test_scatter(self):
        from forge.engine.builtins.plotting import forge_figure, forge_scatter, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_scatter(np.array([1, 2, 3]), np.array([4, 5, 6]))
        forge_close()

    def test_bar(self):
        from forge.engine.builtins.plotting import forge_figure, forge_bar, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_bar(np.array([1, 2, 3]))
        forge_close()

    def test_histogram(self):
        from forge.engine.builtins.plotting import forge_figure, forge_histogram, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_histogram(np.random.randn(100))
        forge_close()


class TestPlotFormatting:
    def test_title(self):
        from forge.engine.builtins.plotting import forge_figure, forge_title, forge_close
        forge_figure()
        forge_title('Test Title')
        ax = plt.gca()
        assert ax.get_title() == 'Test Title'
        forge_close()

    def test_xlabel_ylabel(self):
        from forge.engine.builtins.plotting import forge_figure, forge_xlabel, forge_ylabel, forge_close
        forge_figure()
        forge_xlabel('X')
        forge_ylabel('Y')
        ax = plt.gca()
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'
        forge_close()

    def test_grid(self):
        from forge.engine.builtins.plotting import forge_figure, forge_grid, forge_close
        forge_figure()
        forge_grid()
        forge_close()

    def test_xlim_ylim(self):
        from forge.engine.builtins.plotting import forge_figure, forge_xlim, forge_ylim, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_xlim(0, 10)
        forge_ylim(-1, 1)
        ax = plt.gca()
        assert ax.get_xlim() == (0.0, 10.0)
        forge_close()


class TestFigureManagement:
    def test_figure_create(self):
        from forge.engine.builtins.plotting import forge_figure, forge_close
        forge_figure()
        assert plt.gcf() is not None
        forge_close()

    def test_subplot(self):
        from forge.engine.builtins.plotting import forge_figure, forge_subplot, forge_close
        forge_figure()
        forge_subplot(2, 1, 1)
        forge_subplot(2, 1, 2)
        fig = plt.gcf()
        assert len(fig.axes) == 2
        forge_close()

    def test_clf(self):
        from forge.engine.builtins.plotting import forge_figure, forge_plot, forge_clf, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_plot(np.array([1, 2]), np.array([3, 4]))
        forge_clf()
        assert len(plt.gcf().axes) == 0
        forge_close()

    def test_saveas(self, tmp_path):
        from forge.engine.builtins.plotting import forge_figure, forge_plot, forge_saveas, forge_close
        from forge.engine.types import ForgeArray, _unwrap
        forge_figure()
        forge_plot(np.array([1, 2, 3]), np.array([1, 4, 9]))
        outpath = str(tmp_path / 'test.png')
        forge_saveas(outpath)
        import os
        assert os.path.exists(outpath)
        forge_close()
