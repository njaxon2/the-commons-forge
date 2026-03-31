"""
tests/test_plotting_r2.py - Round 2 plotting function smoke tests.

Verifies that all major plotting functions execute without errors
under the Agg (non-display) backend.
"""
import pytest
import matplotlib
matplotlib.use("Agg")

from forge.engine.session import ForgeSession


@pytest.fixture(scope="module")
def sess():
    """Single session shared across all tests in this module."""
    return ForgeSession()


# ── 2-D basic ────────────────────────────────────────────────

def test_plot_basic(sess):
    r = sess.eval('figure; plot([1 2 3], [4 5 6], "r--"); title("Test"); xlabel("X"); ylabel("Y")')
    assert "error" not in str(r).lower()


def test_plot_single_vector(sess):
    r = sess.eval("figure; plot([10 20 30 40])")
    assert "error" not in str(r).lower()


def test_subplot_grid(sess):
    cmds = [
        "figure",
        "subplot(2,2,1); plot([1 2 3])",
        "subplot(2,2,2); bar([1 2 3])",
        "subplot(2,2,3); scatter([1 2],[3 4])",
        "subplot(2,2,4); stem([1 2 3 4])",
    ]
    for cmd in cmds:
        r = sess.eval(cmd)
        assert "error" not in str(r).lower(), f"Failed on: {cmd}"


def test_hold_on_multiplot(sess):
    r = sess.eval('figure; plot([1 2 3]); hold on; plot([3 2 1], "r"); hold off')
    assert "error" not in str(r).lower()


# ── Log-scale plots ─────────────────────────────────────────

def test_semilogx(sess):
    r = sess.eval("figure; semilogx([1 10 100 1000], [0 1 2 3])")
    assert "error" not in str(r).lower()


def test_semilogy(sess):
    r = sess.eval("figure; semilogy([1 2 3 4], [1 10 100 1000])")
    assert "error" not in str(r).lower()


def test_loglog(sess):
    r = sess.eval("figure; loglog([1 10 100], [1 100 10000])")
    assert "error" not in str(r).lower()


# ── Specialized 2-D ─────────────────────────────────────────

def test_polar(sess):
    r = sess.eval("figure; polar(0:0.1:2*pi, sin(0:0.1:2*pi))")
    assert "error" not in str(r).lower()


def test_polarplot_alias(sess):
    r = sess.eval("figure; polarplot(0:0.1:2*pi, cos(0:0.1:2*pi))")
    assert "error" not in str(r).lower()


def test_errorbar(sess):
    r = sess.eval("figure; errorbar([1 2 3], [4 5 6], [0.5 0.3 0.7])")
    assert "error" not in str(r).lower()


def test_histogram(sess):
    r = sess.eval("figure; histogram(randn(1000,1), 30)")
    assert "error" not in str(r).lower()


def test_bar(sess):
    r = sess.eval("figure; bar([5 3 8 1 6])")
    assert "error" not in str(r).lower()


def test_scatter(sess):
    r = sess.eval("figure; scatter([1 2 3 4], [4 3 2 1])")
    assert "error" not in str(r).lower()


def test_stem(sess):
    r = sess.eval("figure; stem([1 2 3 4 5])")
    assert "error" not in str(r).lower()


# ── Contour / Image ─────────────────────────────────────────

def test_contour_xyz(sess):
    r = sess.eval("figure; contour(peaks(30))")
    assert "error" not in str(r).lower()


def test_contour_z_only(sess):
    r = sess.eval("figure; contour(rand(20,20))")
    assert "error" not in str(r).lower()


def test_imagesc_colorbar(sess):
    r = sess.eval("figure; imagesc(rand(10,10)); colorbar")
    assert "error" not in str(r).lower()


# ── 3-D surface / mesh ──────────────────────────────────────

def test_surf(sess):
    r = sess.eval("figure; surf(peaks(30))")
    assert "error" not in str(r).lower()


def test_mesh(sess):
    r = sess.eval("figure; mesh(peaks(30))")
    assert "error" not in str(r).lower()


# ── Formatting helpers ───────────────────────────────────────

def test_legend(sess):
    r = sess.eval('figure; plot([1 2 3]); hold on; plot([3 2 1]); legend("up", "down")')
    assert "error" not in str(r).lower()


def test_grid_axis(sess):
    r = sess.eval("figure; plot([1 2 3]); grid on; axis([0 4 0 4])")
    assert "error" not in str(r).lower()
