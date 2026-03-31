# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for statistics toolbox — Round 2.

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
    """mean: arithmetic mean, column-wise default."""

    def test_mean_vector(self, s):
        s.eval("r = mean([1 2 3 4 5])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], 3.0)

    def test_mean_matrix_columnwise(self, s):
        # Octave: mean([1 2; 3 4]) => [2 3]
        s.eval("r = mean([1 2; 3 4])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.ravel(), [2.0, 3.0])

    def test_mean_matrix_dim2(self, s):
        # mean([1 2; 3 4], 2) => [1.5; 3.5]
        s.eval("r = mean([1 2; 3 4], 2)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.ravel(), [1.5, 3.5])


class TestMedian:
    """median: middle value."""

    def test_median_odd(self, s):
        s.eval("r = median([1 2 3 4 5])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], 3.0)

    def test_median_even(self, s):
        s.eval("r = median([1 2 3 4])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], 2.5)


class TestStd:
    """std: standard deviation (sample by default, ddof=1)."""

    def test_std_sample(self, s):
        # Sample std of [2 4 4 4 5 5 7 9] = 2.13809
        s.eval("r = std([2 4 4 4 5 5 7 9])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], np.std([2, 4, 4, 4, 5, 5, 7, 9], ddof=1), atol=1e-6)

    def test_std_population(self, s):
        # std(x, 1) uses N normalization
        s.eval("r = std([2 4 4 4 5 5 7 9], 1)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], np.std([2, 4, 4, 4, 5, 5, 7, 9], ddof=0), atol=1e-6)


class TestVar:
    """var: variance (sample by default)."""

    def test_var_sample(self, s):
        s.eval("r = var([2 4 4 4 5 5 7 9])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], np.var([2, 4, 4, 4, 5, 5, 7, 9], ddof=1), atol=1e-6)

    def test_var_population(self, s):
        s.eval("r = var([2 4 4 4 5 5 7 9], 1)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], np.var([2, 4, 4, 4, 5, 5, 7, 9], ddof=0), atol=1e-6)


class TestCov:
    """cov: covariance matrix."""

    def test_cov_two_vectors(self, s):
        # cov([1 2 3], [4 5 6]) => [[1 1;1 1]]
        s.eval("C = cov([1 2 3], [4 5 6])")
        r = _get(s, "C")
        np.testing.assert_allclose(r, [[1.0, 1.0], [1.0, 1.0]], atol=1e-10)

    def test_cov_single_vector(self, s):
        # cov([1 2 3]) = var([1 2 3]) = 1
        s.eval("C = cov([1 2 3])")
        r = _get(s, "C")
        np.testing.assert_allclose(r.flat[0], 1.0, atol=1e-10)

    def test_cov_matrix(self, s):
        # cov of 3x2 matrix
        s.eval("C = cov([1 2; 3 4; 5 6])")
        r = _get(s, "C")
        np.testing.assert_allclose(r, [[4.0, 4.0], [4.0, 4.0]], atol=1e-10)

    def test_cov_norm_flag(self, s):
        # cov([1 2 3], 1) => normalize by N => 2/3
        s.eval("C = cov([1 2 3], 1)")
        r = _get(s, "C")
        np.testing.assert_allclose(r.flat[0], 2.0 / 3.0, atol=1e-10)


class TestCorrcoef:
    """corrcoef: correlation coefficient matrix."""

    def test_corrcoef_perfect(self, s):
        s.eval("R = corrcoef([1 2 3], [1 2 3])")
        r = _get(s, "R")
        np.testing.assert_allclose(r, [[1.0, 1.0], [1.0, 1.0]], atol=1e-10)

    def test_corrcoef_anticorr(self, s):
        s.eval("R = corrcoef([1 2 3], [3 2 1])")
        r = _get(s, "R")
        np.testing.assert_allclose(r[0, 1], -1.0, atol=1e-10)


class TestHist:
    """hist: histogram bin counts."""

    def test_hist_basic(self, s):
        s.eval("[n, x] = hist([1 1 2 3 3 3], 3)")
        n = _get(s, "n")
        assert n.sum() == 6  # all elements accounted for

    def test_hist_default_bins(self, s):
        s.eval("[n, x] = hist([1 2 3 4 5 6 7 8 9 10])")
        n = _get(s, "n")
        assert n.sum() == 10


class TestCumsum:
    """cumsum: cumulative sum."""

    def test_cumsum_vector(self, s):
        s.eval("r = cumsum([1 2 3 4])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.ravel(), [1, 3, 6, 10])


class TestCumprod:
    """cumprod: cumulative product."""

    def test_cumprod_vector(self, s):
        s.eval("r = cumprod([1 2 3 4])")
        r = _get(s, "r")
        np.testing.assert_allclose(r.ravel(), [1, 2, 6, 24])


class TestMovmean:
    """movmean: moving average."""

    def test_movmean_3(self, s):
        s.eval("r = movmean([1 2 3 4 5], 3)")
        r = _get(s, "r")
        # Interior elements: [2, 3, 4]; edges: partial window
        np.testing.assert_allclose(r.ravel()[1:-1], [2.0, 3.0, 4.0], atol=1e-10)

    def test_movmean_length(self, s):
        s.eval("r = movmean([10 20 30 40 50], 3)")
        r = _get(s, "r")
        assert r.size == 5


class TestQuantile:
    """quantile/prctile: percentile calculations."""

    def test_quantile_median(self, s):
        s.eval("r = quantile([1 2 3 4 5], 0.5)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], 3.0, atol=1e-10)

    def test_prctile_median(self, s):
        s.eval("r = prctile([1 2 3 4 5], 50)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], 3.0, atol=1e-10)

    def test_quantile_quartiles(self, s):
        s.eval("r = quantile([1 2 3 4 5], 0.25)")
        r = _get(s, "r")
        np.testing.assert_allclose(r.flat[0], 2.0, atol=1e-10)


class TestZscore:
    """zscore: standardized z-scores."""

    def test_zscore_center(self, s):
        s.eval("z = zscore([1 2 3 4 5])")
        z = _get(s, "z")
        # Mean of z-scores should be 0
        np.testing.assert_allclose(np.mean(z), 0.0, atol=1e-10)

    def test_zscore_scale(self, s):
        s.eval("z = zscore([1 2 3 4 5])")
        z = _get(s, "z")
        # Std of z-scores should be ~1 (with ddof=1)
        np.testing.assert_allclose(np.std(z.ravel(), ddof=1), 1.0, atol=1e-10)
