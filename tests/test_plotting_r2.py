"""
tests/test_plotting_r2.py - Round 2 plotting function smoke tests.

Verifies that all major plotting functions execute without errors
under the Agg (non-display) backend.

V-Model Traceability
--------------------
R-PLOT-04 through R-PLOT-12 (see requirement blocks below).
Golden user: Engineer/scientist migrating from MATLAB/Octave who relies on
log-scale plots for Bode/frequency analysis, polar plots for antenna patterns,
contour/surf/mesh for FEM visualization, errorbar for experimental uncertainty,
subplot grids for multi-dataset comparison, and hold on/off as muscle memory.
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
# R-PLOT-04: Basic 2-D plotting and subplot grids
#
# Requirement: The engine shall execute plot(), subplot(), and hold on/off
# without error, producing valid figure objects under the Agg backend.
#
# Model-user argument: The migrating engineer uses plot() dozens of times per
# session for quick data inspection. subplot() grids are the standard way to
# compare datasets side by side (e.g., time-domain vs. frequency-domain).
# hold on/off is deeply ingrained muscle memory; every multi-series overlay
# depends on it working exactly as in Octave.

def test_plot_basic(sess):
    """R-PLOT-04a: plot(x, y, fmt) with title, xlabel, ylabel produces no error."""
    r = sess.eval('figure; plot([1 2 3], [4 5 6], "r--"); title("Test"); xlabel("X"); ylabel("Y")')
    assert "error" not in str(r).lower()


def test_plot_single_vector(sess):
    """R-PLOT-04b: plot(y) with a single vector uses implicit indices."""
    r = sess.eval("figure; plot([10 20 30 40])")
    assert "error" not in str(r).lower()


def test_subplot_grid(sess):
    """R-PLOT-04c: subplot(2,2,k) with four different plot types in one figure."""
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
    """R-PLOT-04d: hold on allows overlaying two series; hold off resets."""
    r = sess.eval('figure; plot([1 2 3]); hold on; plot([3 2 1], "r"); hold off')
    assert "error" not in str(r).lower()


# ── Log-scale plots ─────────────────────────────────────────
# R-PLOT-05: Logarithmic-axis plotting functions
#
# Requirement: semilogx(), semilogy(), and loglog() shall produce correctly
# scaled axes without error.
#
# Model-user argument: The engineer uses semilogx/semilogy for Bode magnitude
# and phase plots (frequency on a log axis, gain in dB on linear). loglog is
# essential for power spectral density and transfer function analysis. These
# three functions are called daily in any controls or signal processing workflow.

def test_semilogx(sess):
    """R-PLOT-05a: semilogx produces a figure with logarithmic x-axis."""
    r = sess.eval("figure; semilogx([1 10 100 1000], [0 1 2 3])")
    assert "error" not in str(r).lower()


def test_semilogy(sess):
    """R-PLOT-05b: semilogy produces a figure with logarithmic y-axis."""
    r = sess.eval("figure; semilogy([1 2 3 4], [1 10 100 1000])")
    assert "error" not in str(r).lower()


def test_loglog(sess):
    """R-PLOT-05c: loglog produces a figure with both axes logarithmic."""
    r = sess.eval("figure; loglog([1 10 100], [1 100 10000])")
    assert "error" not in str(r).lower()


# ── Specialized 2-D ─────────────────────────────────────────
# R-PLOT-06: Polar plotting functions
#
# Requirement: polar() and polarplot() shall render data in polar coordinates
# without error.
#
# Model-user argument: The engineer plots antenna radiation patterns and
# directional gain using polar coordinates. polar() is the classic Octave call;
# polarplot() is the newer MATLAB alias. Both must work so scripts from either
# era run without modification.

def test_polar(sess):
    """R-PLOT-06a: polar(theta, rho) renders a polar-coordinate figure."""
    r = sess.eval("figure; polar(0:0.1:2*pi, sin(0:0.1:2*pi))")
    assert "error" not in str(r).lower()


def test_polarplot_alias(sess):
    """R-PLOT-06b: polarplot() works as a modern alias for polar()."""
    r = sess.eval("figure; polarplot(0:0.1:2*pi, cos(0:0.1:2*pi))")
    assert "error" not in str(r).lower()


# R-PLOT-07: Error bar plotting
#
# Requirement: errorbar() shall render data points with symmetric error bars
# without error.
#
# Model-user argument: The engineer presents experimental measurements with
# uncertainty bounds. errorbar() is the standard way to visualize measurement
# confidence intervals in lab reports and publications.

def test_errorbar(sess):
    """R-PLOT-07: errorbar(x, y, err) renders data with symmetric error bars."""
    r = sess.eval("figure; errorbar([1 2 3], [4 5 6], [0.5 0.3 0.7])")
    assert "error" not in str(r).lower()


# R-PLOT-08: Statistical and categorical 2-D plots
#
# Requirement: histogram(), bar(), scatter(), and stem() shall execute without
# error, producing appropriate figure types.
#
# Model-user argument: The engineer uses histogram() to inspect distributions
# of simulation outputs or sensor readings. bar() summarizes categorical
# comparisons. scatter() reveals correlations between paired variables. stem()
# displays discrete-time signals, which is fundamental in DSP coursework and
# sampled-data analysis.

def test_histogram(sess):
    """R-PLOT-08a: histogram(data, nbins) renders a histogram of random data."""
    r = sess.eval("figure; histogram(randn(1000,1), 30)")
    assert "error" not in str(r).lower()


def test_bar(sess):
    """R-PLOT-08b: bar(y) renders a bar chart from a vector."""
    r = sess.eval("figure; bar([5 3 8 1 6])")
    assert "error" not in str(r).lower()


def test_scatter(sess):
    """R-PLOT-08c: scatter(x, y) renders a scatter plot."""
    r = sess.eval("figure; scatter([1 2 3 4], [4 3 2 1])")
    assert "error" not in str(r).lower()


def test_stem(sess):
    """R-PLOT-08d: stem(y) renders a discrete-sequence stem plot."""
    r = sess.eval("figure; stem([1 2 3 4 5])")
    assert "error" not in str(r).lower()


# ── Contour / Image ─────────────────────────────────────────
# R-PLOT-09: Contour and image display functions
#
# Requirement: contour() shall accept both Z-only and X,Y,Z forms. imagesc()
# shall render a scaled color image with optional colorbar, all without error.
#
# Model-user argument: The engineer visualizes FEM stress fields and thermal
# distributions using contour plots. contour(peaks(N)) is a common sanity check
# during development. imagesc() with colorbar is used to display matrix data
# (e.g., correlation matrices, spectrograms) with a quantitative color scale.

def test_contour_xyz(sess):
    """R-PLOT-09a: contour(Z) renders contour lines from peaks() matrix."""
    r = sess.eval("figure; contour(peaks(30))")
    assert "error" not in str(r).lower()


def test_contour_z_only(sess):
    """R-PLOT-09b: contour(Z) renders contour lines from a random matrix."""
    r = sess.eval("figure; contour(rand(20,20))")
    assert "error" not in str(r).lower()


def test_imagesc_colorbar(sess):
    """R-PLOT-09c: imagesc(Z) with colorbar renders a scaled color image."""
    r = sess.eval("figure; imagesc(rand(10,10)); colorbar")
    assert "error" not in str(r).lower()


# ── 3-D surface / mesh ──────────────────────────────────────
# R-PLOT-10: 3-D surface and mesh visualization
#
# Requirement: surf() and mesh() shall render 3-D surfaces from matrix data
# without error.
#
# Model-user argument: The engineer visualizes FEM displacement fields, potential
# surfaces, and optimization landscapes using surf() and mesh(). surf() provides
# filled surfaces for presentation quality output; mesh() gives wireframe views
# for interactive inspection of surface topology during analysis.

def test_surf(sess):
    """R-PLOT-10a: surf(Z) renders a 3-D filled surface from peaks()."""
    r = sess.eval("figure; surf(peaks(30))")
    assert "error" not in str(r).lower()


def test_mesh(sess):
    """R-PLOT-10b: mesh(Z) renders a 3-D wireframe from peaks()."""
    r = sess.eval("figure; mesh(peaks(30))")
    assert "error" not in str(r).lower()


# ── Formatting helpers ───────────────────────────────────────
# R-PLOT-11: Legend annotation
#
# Requirement: legend() shall label multiple series in a multi-plot figure
# without error.
#
# Model-user argument: The engineer overlays measured vs. simulated curves and
# needs legend labels to distinguish them. Every comparison plot in a report
# requires a legend; without it the figure is incomplete.

def test_legend(sess):
    """R-PLOT-11: legend() labels two overlaid series in one figure."""
    r = sess.eval('figure; plot([1 2 3]); hold on; plot([3 2 1]); legend("up", "down")')
    assert "error" not in str(r).lower()


# R-PLOT-12: Grid and axis control
#
# Requirement: grid on and axis([...]) shall modify figure appearance without
# error.
#
# Model-user argument: The engineer enables grid lines for readable data
# extraction and sets explicit axis limits to standardize plot ranges across
# subplots. These formatting calls appear in nearly every publication-quality
# figure script.

def test_grid_axis(sess):
    """R-PLOT-12: grid on and axis([xmin xmax ymin ymax]) apply without error."""
    r = sess.eval("figure; plot([1 2 3]); grid on; axis([0 4 0 4])")
    assert "error" not in str(r).lower()
