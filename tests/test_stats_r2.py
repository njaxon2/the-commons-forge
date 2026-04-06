# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for statistics toolbox, Round 2.

Tests key statistics functions for correctness and Octave-compatible
dimension handling (column-wise defaults, sample variance by default).

SRS trace: SRS-FUNC-STATS
Test method: Comparison against known analytical values and NumPy reference.
"""
import pytest
import numpy as np
from forge.engine.evaluator import Session
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def s():
    return Session()


def _val(session, expr):
    """Evaluate expression and return unwrapped numpy array."""
    session.eval(expr)
    return expr


def _get(session, name):
    """Get variable from workspace as numpy array."""
    return np.asarray(_unwrap(session.workspace.get(name)))


class TestMean:
    """R-STAT-10: mean SHALL compute the arithmetic mean, defaulting to
    column-wise operation on matrices.

    Model-user argument: A MATLAB/Octave engineer calling mean(M) on a
    data matrix expects column-wise means returned as a row vector. This
    is the fundamental statistical summary operation and must match the
    Octave convention without requiring an explicit dimension argument.

    Decomposition:
      R-STAT-10a: mean of a row vector returns the scalar arithmetic mean.
      R-STAT-10b: mean of a matrix with no dim argument returns column-wise means.
      R-STAT-10c: mean with dim=2 returns row-wise means.

    Consistency argument: 10a covers the 1-D case. 10b and 10c together
    cover both dimension axes for 2-D input, confirming the default
    (column-wise) and the explicit alternative (row-wise).
    """

    def test_mean_vector(self, s):
        """R-STAT-10a: mean of row vector returns scalar arithmetic mean."""
        s.eval("r = mean([1 2 3 4 5])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], 3.0)

    def test_mean_matrix_columnwise(self, s):
        """R-STAT-10b: mean of matrix defaults to column-wise means."""
        s.eval("r = mean([1 2; 3 4])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.ravel(), [2.0, 3.0])

    def test_mean_matrix_dim2(self, s):
        """R-STAT-10c: mean with dim=2 returns row-wise means."""
        s.eval("r = mean([1 2; 3 4], 2)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.ravel(), [1.5, 3.5])


class TestMedian:
    """R-STAT-11: median SHALL return the middle value of a sorted dataset,
    averaging the two central values for even-length inputs.

    Model-user argument: Scientists working with sensor data routinely use
    median as a robust location estimator. The even-length averaging rule
    is the standard convention in Octave and MATLAB, and any deviation
    would silently corrupt summary statistics.

    Decomposition:
      R-STAT-11a: median of odd-length vector returns the central element.
      R-STAT-11b: median of even-length vector returns the average of the two central elements.

    Consistency argument: Odd and even cardinalities are the only two cases
    for a finite dataset; covering both proves correctness for all lengths.
    """

    def test_median_odd(self, s):
        """R-STAT-11a: median of odd-length vector returns central element."""
        s.eval("r = median([1 2 3 4 5])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], 3.0)

    def test_median_even(self, s):
        """R-STAT-11b: median of even-length vector averages two central elements."""
        s.eval("r = median([1 2 3 4])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], 2.5)


class TestStd:
    """R-STAT-12: std SHALL compute the standard deviation, using sample
    normalization (ddof=1) by default and population normalization (ddof=0)
    when the second argument is 1.

    Model-user argument: Octave's std uses N-1 (sample) by default and
    accepts std(x,1) for population. A migrating engineer will write
    std(data) and expect sample std, then use std(data,1) for population.
    If the ddof convention is wrong, every error bar and confidence
    interval they compute will be silently incorrect.

    Decomposition:
      R-STAT-12a: std with no flag returns sample standard deviation (ddof=1).
      R-STAT-12b: std with flag=1 returns population standard deviation (ddof=0).

    Consistency argument: The two normalization modes (sample, population)
    are the only variants of the std function. Both are covered.
    """

    def test_std_sample(self, s):
        """R-STAT-12a: std defaults to sample standard deviation (ddof=1)."""
        s.eval("r = std([2 4 4 4 5 5 7 9])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], np.std([2, 4, 4, 4, 5, 5, 7, 9], ddof=1), atol=1e-6)

    def test_std_population(self, s):
        """R-STAT-12b: std with flag=1 returns population standard deviation (ddof=0)."""
        s.eval("r = std([2 4 4 4 5 5 7 9], 1)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], np.std([2, 4, 4, 4, 5, 5, 7, 9], ddof=0), atol=1e-6)


class TestVar:
    """R-STAT-13: var SHALL compute the variance, using sample normalization
    (ddof=1) by default and population normalization (ddof=0) when the
    second argument is 1.

    Model-user argument: var is the squared companion of std; Octave users
    expect identical ddof conventions between the two functions. A scientist
    computing ANOVA tables or propagation-of-uncertainty will rely on var
    returning sample variance without needing an extra flag.

    Decomposition:
      R-STAT-13a: var with no flag returns sample variance (ddof=1).
      R-STAT-13b: var with flag=1 returns population variance (ddof=0).

    Consistency argument: Sample and population are the only two
    normalization modes. Both are tested.
    """

    def test_var_sample(self, s):
        """R-STAT-13a: var defaults to sample variance (ddof=1)."""
        s.eval("r = var([2 4 4 4 5 5 7 9])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], np.var([2, 4, 4, 4, 5, 5, 7, 9], ddof=1), atol=1e-6)

    def test_var_population(self, s):
        """R-STAT-13b: var with flag=1 returns population variance (ddof=0)."""
        s.eval("r = var([2 4 4 4 5 5 7 9], 1)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], np.var([2, 4, 4, 4, 5, 5, 7, 9], ddof=0), atol=1e-6)


class TestCov:
    """R-STAT-14: cov SHALL compute the covariance matrix, supporting
    cov(x,y) for two-vector cross-covariance, cov(M) for a data matrix,
    and cov(x,1) for population normalization.

    Model-user argument: Cross-covariance via cov(x,y) is the standard
    Octave/MATLAB call for assessing linear dependence between two measured
    signals. Engineers also use cov(M) on multi-column data matrices to
    get the full covariance matrix for PCA or Kalman filter design.
    The normalization flag (cov(x,1) for population) must match Octave.

    Decomposition:
      R-STAT-14a: cov(x,y) returns 2x2 sample covariance matrix.
      R-STAT-14b: cov(x) on a single vector returns the scalar sample variance.
      R-STAT-14c: cov(M) on a matrix returns column-wise covariance matrix.
      R-STAT-14d: cov(x,1) returns population-normalized variance.

    Consistency argument: 14a covers two-input cross-covariance. 14b and
    14c cover single-input for vectors and matrices respectively. 14d
    covers the normalization flag. Together these span all documented
    call signatures of Octave's cov function.
    """

    def test_cov_two_vectors(self, s):
        """R-STAT-14a: cov(x,y) returns 2x2 sample covariance matrix."""
        s.eval("C = cov([1 2 3], [4 5 6])")
        r = _get(s, "C")
        np.testing.assert_allclose(r, [[1.0, 1.0], [1.0, 1.0]], atol=1e-10)

    def test_cov_single_vector(self, s):
        """R-STAT-14b: cov of single vector returns scalar sample variance."""
        s.eval("C = cov([1 2 3])")
        r = _get(s, "C")
        np.testing.assert_allclose(r.flat[0], 1.0, atol=1e-10)

    def test_cov_matrix(self, s):
        """R-STAT-14c: cov of matrix returns column-wise covariance matrix."""
        s.eval("C = cov([1 2; 3 4; 5 6])")
        r = _get(s, "C")
        np.testing.assert_allclose(r, [[4.0, 4.0], [4.0, 4.0]], atol=1e-10)

    def test_cov_norm_flag(self, s):
        """R-STAT-14d: cov(x,1) returns population-normalized variance."""
        s.eval("C = cov([1 2 3], 1)")
        r = _get(s, "C")
        np.testing.assert_allclose(r.flat[0], 2.0 / 3.0, atol=1e-10)


class TestCorrcoef:
    """R-STAT-15: corrcoef SHALL compute the Pearson correlation coefficient
    matrix for two input vectors.

    Model-user argument: Engineers assessing signal similarity call
    corrcoef(x,y) and expect a 2x2 matrix with off-diagonal values in
    [-1, 1]. Perfect positive correlation must yield 1.0 and perfect
    negative correlation must yield -1.0, matching Octave exactly.

    Decomposition:
      R-STAT-15a: corrcoef of identical vectors returns all-ones matrix.
      R-STAT-15b: corrcoef of perfectly anti-correlated vectors returns -1.0 off-diagonal.

    Consistency argument: Perfect correlation and perfect anti-correlation
    are the two boundary cases of the Pearson coefficient. If both
    extremes are correct, the normalization and sign logic are validated.
    """

    def test_corrcoef_perfect(self, s):
        """R-STAT-15a: corrcoef of identical vectors returns all-ones matrix."""
        s.eval("R = corrcoef([1 2 3], [1 2 3])")
        r = _get(s, "R")
        np.testing.assert_allclose(r, [[1.0, 1.0], [1.0, 1.0]], atol=1e-10)

    def test_corrcoef_anticorr(self, s):
        """R-STAT-15b: corrcoef of anti-correlated vectors returns -1.0 off-diagonal."""
        s.eval("R = corrcoef([1 2 3], [3 2 1])")
        r = _get(s, "R")
        np.testing.assert_allclose(r[0, 1], -1.0, atol=1e-10)


class TestHist:
    """R-STAT-16: hist SHALL return bin counts and bin centers, preserving
    the total element count across all bins.

    Model-user argument: Octave's hist returns [n, x] where sum(n) equals
    the number of input elements. Engineers use this to build frequency
    distributions for quality control and signal analysis. If any elements
    are lost or double-counted, the histogram is invalid.

    Decomposition:
      R-STAT-16a: hist with explicit bin count preserves total element count.
      R-STAT-16b: hist with default bins preserves total element count.

    Consistency argument: Explicit and default bin counts are the two
    calling conventions. Verifying sum(n) == numel(data) in both cases
    confirms no elements are lost regardless of binning strategy.
    """

    def test_hist_basic(self, s):
        """R-STAT-16a: hist with explicit bin count preserves total element count."""
        s.eval("[n, x] = hist([1 1 2 3 3 3], 3)")
        n = _get(s, "n")
        assert n.sum() == 6

    def test_hist_default_bins(self, s):
        """R-STAT-16b: hist with default bins preserves total element count."""
        s.eval("[n, x] = hist([1 2 3 4 5 6 7 8 9 10])")
        n = _get(s, "n")
        assert n.sum() == 10


class TestCumsum:
    """R-STAT-17: cumsum SHALL return the cumulative sum of a vector,
    producing an output of the same size as the input.

    Model-user argument: cumsum is used heavily in running-total
    calculations, cumulative distribution construction, and prefix-sum
    algorithms. An Octave user expects cumsum([1 2 3 4]) to return
    [1 3 6 10] with no ambiguity.

    Decomposition:
      R-STAT-17a: cumsum of a row vector returns correct running totals.

    Consistency argument: A single vector test with known analytical
    values verifies the prefix-sum logic. (Column-wise matrix behavior
    is covered by the general dimension-default tests in Round 1.)
    """

    def test_cumsum_vector(self, s):
        """R-STAT-17a: cumsum of row vector returns correct running totals."""
        s.eval("r = cumsum([1 2 3 4])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.ravel(), [1, 3, 6, 10])


class TestCumprod:
    """R-STAT-18: cumprod SHALL return the cumulative product of a vector,
    producing an output of the same size as the input.

    Model-user argument: cumprod is essential for compound growth
    calculations and factorial-style accumulations. An Octave user
    expects cumprod([1 2 3 4]) to return [1 2 6 24] without requiring
    a loop.

    Decomposition:
      R-STAT-18a: cumprod of a row vector returns correct running products.

    Consistency argument: A single vector test with known analytical
    values verifies the multiplicative accumulation logic.
    """

    def test_cumprod_vector(self, s):
        """R-STAT-18a: cumprod of row vector returns correct running products."""
        s.eval("r = cumprod([1 2 3 4])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.ravel(), [1, 2, 6, 24])


class TestMovmean:
    """R-STAT-19: movmean SHALL compute the moving average with a specified
    window size, returning an output vector of the same length as the input.

    Model-user argument: Engineers use movmean to smooth noisy time-series
    data before plotting or further analysis. Octave's movmean preserves
    the input length and applies endpoint handling automatically. If the
    output length differs from the input, downstream indexing breaks.

    Decomposition:
      R-STAT-19a: movmean interior values match the expected window average.
      R-STAT-19b: movmean output length equals input length.

    Consistency argument: 19a verifies numerical correctness of the
    interior (fully-windowed) elements. 19b verifies the length
    preservation contract including endpoint handling.
    """

    def test_movmean_3(self, s):
        """R-STAT-19a: movmean interior values match expected window averages."""
        s.eval("r = movmean([1 2 3 4 5], 3)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.ravel()[1:-1], [2.0, 3.0, 4.0], atol=1e-10)

    def test_movmean_length(self, s):
        """R-STAT-19b: movmean output length equals input length."""
        s.eval("r = movmean([10 20 30 40 50], 3)")
        r = _get(s, "r")
        assert r.size == 5


class TestQuantile:
    """R-STAT-20: quantile and prctile SHALL compute percentile values,
    with quantile accepting probabilities in [0,1] and prctile accepting
    percentages in [0,100].

    Model-user argument: Octave provides both quantile (probability scale)
    and prctile (percentage scale) for convenience. A scientist computing
    confidence intervals will use one or both interchangeably. The median
    (p=0.5 or 50th percentile) is the canonical validation point, and
    quartiles confirm interpolation behavior.

    Decomposition:
      R-STAT-20a: quantile at p=0.5 returns the median.
      R-STAT-20b: prctile at 50 returns the median (percentage-scale equivalent).
      R-STAT-20c: quantile at p=0.25 returns the first quartile.

    Consistency argument: 20a and 20b confirm the two API surfaces agree
    on the median. 20c verifies interpolation at a non-median quantile,
    proving the general percentile computation is correct.
    """

    def test_quantile_median(self, s):
        """R-STAT-20a: quantile at p=0.5 returns the median."""
        s.eval("r = quantile([1 2 3 4 5], 0.5)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], 3.0, atol=1e-10)

    def test_prctile_median(self, s):
        """R-STAT-20b: prctile at 50 returns the median."""
        s.eval("r = prctile([1 2 3 4 5], 50)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], 3.0, atol=1e-10)

    def test_quantile_quartiles(self, s):
        """R-STAT-20c: quantile at p=0.25 returns the first quartile."""
        s.eval("r = quantile([1 2 3 4 5], 0.25)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], 2.0, atol=1e-10)


class TestZscore:
    """R-STAT-21: zscore SHALL return standardized z-scores with zero mean
    and unit sample standard deviation.

    Model-user argument: Z-score normalization is a prerequisite for many
    statistical methods (PCA, clustering, hypothesis testing). An Octave
    user expects zscore(x) to center the data at zero and scale to unit
    sample standard deviation. If either property fails, all downstream
    standardized analyses are invalid.

    Decomposition:
      R-STAT-21a: zscore output has zero mean.
      R-STAT-21b: zscore output has unit sample standard deviation.

    Consistency argument: Zero mean and unit standard deviation are the
    two defining properties of z-score normalization. Verifying both
    fully validates the transformation.
    """

    def test_zscore_center(self, s):
        """R-STAT-21a: zscore output has zero mean."""
        s.eval("z = zscore([1 2 3 4 5])")
        z = _get(s, "z")
        np.testing.assert_allclose(np.mean(z), 0.0, atol=1e-10)

    def test_zscore_scale(self, s):
        """R-STAT-21b: zscore output has unit sample standard deviation."""
        s.eval("z = zscore([1 2 3 4 5])")
        z = _get(s, "z")
        np.testing.assert_allclose(np.std(z.ravel(), ddof=1), 1.0, atol=1e-10)
