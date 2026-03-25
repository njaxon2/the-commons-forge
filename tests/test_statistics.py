import pytest
import numpy as np


class TestBasicStats:
    def test_mean(self):
        from forge.engine.builtins.statistics import mean
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _unwrap(mean(x)).ravel()
        assert abs(result[0] - 3.0) < 1e-10

    def test_median(self):
        from forge.engine.builtins.statistics import median
        from forge.engine.types import _unwrap
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = _unwrap(median(x)).ravel()
        assert abs(result[0] - 3.0) < 1e-10

    def test_std(self):
        from forge.engine.builtins.statistics import std
        from forge.engine.types import _unwrap
        x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = _unwrap(std(x)).ravel()
        # Octave std uses N-1 (sample std)
        expected = np.std(x, ddof=1)
        assert abs(result[0] - expected) < 1e-10

    def test_var(self):
        from forge.engine.builtins.statistics import var
        from forge.engine.types import _unwrap
        x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = _unwrap(var(x)).ravel()
        expected = np.var(x, ddof=1)
        assert abs(result[0] - expected) < 1e-10


class TestCorrelation:
    def test_corr_perfect(self):
        from forge.engine.builtins.statistics import corr
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = _unwrap(corr(x, y)).ravel()
        assert abs(result[0] - 1.0) < 1e-10

    def test_cov_diagonal(self):
        from forge.engine.builtins.statistics import cov
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _unwrap(cov(x))
        # cov of single vector is scalar variance
        expected = np.var(x, ddof=1)
        assert abs(result.ravel()[0] - expected) < 1e-10


class TestHigherMoments:
    def test_kurtosis_normal(self):
        from forge.engine.builtins.statistics import kurtosis
        from forge.engine.types import _unwrap
        np.random.seed(42)
        x = np.random.randn(10000)
        result = _unwrap(kurtosis(x)).ravel()
        # Octave kurtosis of normal ~ 3 (non-excess) or ~0 (excess)
        assert abs(result[0] - 3.0) < 0.3 or abs(result[0]) < 0.3

    def test_skewness_symmetric(self):
        from forge.engine.builtins.statistics import skewness
        from forge.engine.types import _unwrap
        np.random.seed(42)
        x = np.random.randn(10000)
        result = _unwrap(skewness(x)).ravel()
        assert abs(result[0]) < 0.1


class TestQuantiles:
    def test_prctile(self):
        from forge.engine.builtins.statistics import prctile
        from forge.engine.types import _unwrap
        x = np.arange(1.0, 101.0)  # 1..100
        result = _unwrap(prctile(x, 50)).ravel()
        assert abs(result[0] - 50.5) < 1.0

    def test_quantile(self):
        from forge.engine.builtins.statistics import quantile
        from forge.engine.types import _unwrap
        x = np.arange(1.0, 101.0)
        result = _unwrap(quantile(x, 0.25)).ravel()
        assert result[0] > 20 and result[0] < 30


class TestRobust:
    def test_zscore(self):
        from forge.engine.builtins.statistics import zscore
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = _unwrap(zscore(x)).ravel()
        # z-scores should have mean ~0 and std ~1
        assert abs(np.mean(z)) < 1e-10
        assert abs(np.std(z, ddof=1) - 1.0) < 1e-10

    def test_iqr(self):
        from forge.engine.builtins.statistics import iqr
        from forge.engine.types import _unwrap
        x = np.arange(1.0, 101.0)
        result = _unwrap(iqr(x)).ravel()
        # IQR of 1..100 should be ~50
        assert result[0] > 40 and result[0] < 60

    def test_mad(self):
        from forge.engine.builtins.statistics import mad
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _unwrap(mad(x)).ravel()
        # MAD = median(|x - median(x)|)
        assert result[0] > 0


class TestDistribution:
    def test_mode(self):
        from forge.engine.builtins.statistics import mode
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0])
        result = _unwrap(mode(x)).ravel()
        assert abs(result[0] - 3.0) < 1e-10

    def test_histc(self):
        from forge.engine.builtins.statistics import histc
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        edges = np.array([0.0, 2.5, 5.0])
        n, idx = histc(x, edges)
        arr = _unwrap(n).ravel()
        # Bins: [0,2.5) has {1,2}, [2.5,5.0] has {3,4,5}
        assert arr[0] == 2
        assert arr[1] >= 2  # at least 3,4; 5 may land in last bin


class TestMoving:
    def test_movmean(self):
        from forge.engine.builtins.statistics import movmean
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = _unwrap(movmean(x, 3)).ravel()
        assert len(y) == 5
        assert abs(y[1] - 2.0) < 1e-10  # mean of [1,2,3]
        assert abs(y[2] - 3.0) < 1e-10  # mean of [2,3,4]
        assert abs(y[3] - 4.0) < 1e-10  # mean of [3,4,5]


class TestMisc:
    def test_ranks(self):
        from forge.engine.builtins.statistics import ranks
        from forge.engine.types import _unwrap
        x = np.array([40.0, 10.0, 30.0, 20.0])
        r = _unwrap(ranks(x)).ravel()
        # 10->1, 20->2, 30->3, 40->4
        np.testing.assert_array_equal(r, [4, 1, 3, 2])

    def test_normalize(self):
        from forge.engine.builtins.statistics import normalize
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = _unwrap(normalize(x)).ravel()
        assert abs(np.mean(z)) < 1e-10
        assert abs(np.std(z, ddof=0) - 1.0) < 0.5

    def test_bounds(self):
        from forge.engine.builtins.statistics import bounds
        from forge.engine.types import _unwrap
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        result = _unwrap(bounds(x)).ravel()
        assert result[0] == 1.0
        assert result[1] == 9.0

    def test_range_stat(self):
        from forge.engine.builtins.statistics import range_stat as forge_range
        from forge.engine.types import _unwrap
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        result = _unwrap(forge_range(x)).ravel()
        assert abs(result[0] - 8.0) < 1e-10
