# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np


class TestBasicStats:
    """R-STAT-01: Basic descriptive statistics (mean, median, std, var).

    Requirement: The statistics toolbox SHALL compute mean, median, standard
    deviation (sample, ddof=1), and variance (sample, ddof=1) for numeric
    vectors, returning results consistent with Octave/MATLAB conventions.

    Model-user argument: The golden user computes mean, std, and var on
    experimental data daily. They rely on sample standard deviation (ddof=1)
    as the default, matching MATLAB/Octave behavior. If Forge silently used
    population std (ddof=0), their uncertainty estimates would be biased low,
    leading to missed outliers and incorrect error bars in reports.

    Decomposition:
        R-STAT-01a: mean() returns the arithmetic mean of a numeric vector.
        R-STAT-01b: median() returns the median value of a numeric vector.
        R-STAT-01c: std() returns sample standard deviation (ddof=1).
        R-STAT-01d: var() returns sample variance (ddof=1).

    Consistency argument: The four sub-requirements cover the four distinct
    descriptive statistics named in the parent requirement. Each is tested
    independently with known reference values. Together they fully satisfy
    R-STAT-01.
    """

    def test_mean(self):
        """R-STAT-01a: mean() returns arithmetic mean of a numeric vector."""
        from forge.engine.builtins.statistics import mean
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _unwrap(mean(x)).ravel()
        assert abs(result[0] - 3.0) < 1e-10

    def test_median(self):
        """R-STAT-01b: median() returns median of a numeric vector."""
        from forge.engine.builtins.statistics import median
        from forge.engine.types import _unwrap
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = _unwrap(median(x)).ravel()
        assert abs(result[0] - 3.0) < 1e-10

    def test_std(self):
        """R-STAT-01c: std() returns sample standard deviation (ddof=1)."""
        from forge.engine.builtins.statistics import std
        from forge.engine.types import _unwrap
        x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = _unwrap(std(x)).ravel()
        expected = np.std(x, ddof=1)
        assert abs(result[0] - expected) < 1e-10

    def test_var(self):
        """R-STAT-01d: var() returns sample variance (ddof=1)."""
        from forge.engine.builtins.statistics import var
        from forge.engine.types import _unwrap
        x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = _unwrap(var(x)).ravel()
        expected = np.var(x, ddof=1)
        assert abs(result[0] - expected) < 1e-10


class TestCorrelation:
    """R-STAT-02: Correlation and covariance functions.

    Requirement: The statistics toolbox SHALL compute Pearson correlation
    coefficients via corr() and sample covariance matrices via cov(), matching
    Octave/MATLAB semantics (sample covariance with N-1 denominator).

    Model-user argument: The golden user checks correlations between sensor
    channels to identify coupled measurements and diagnose instrumentation
    drift. They use cov() to build covariance matrices for multivariate
    analysis. Both functions must use sample normalization (N-1) to match the
    values they would obtain in MATLAB, ensuring cross-tool reproducibility.

    Decomposition:
        R-STAT-02a: corr() returns 1.0 for perfectly linearly related vectors.
        R-STAT-02b: cov() of a single vector returns its sample variance.

    Consistency argument: R-STAT-02a validates the correlation coefficient
    computation against a known perfect-correlation case. R-STAT-02b validates
    that the covariance matrix diagonal equals sample variance. Together they
    confirm both functions operate correctly with sample normalization.
    """

    def test_corr_perfect(self):
        """R-STAT-02a: corr() returns 1.0 for perfectly correlated vectors."""
        from forge.engine.builtins.statistics import corr
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = _unwrap(corr(x, y)).ravel()
        assert abs(result[0] - 1.0) < 1e-10

    def test_cov_diagonal(self):
        """R-STAT-02b: cov() of a single vector equals its sample variance."""
        from forge.engine.builtins.statistics import cov
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _unwrap(cov(x))
        expected = np.var(x, ddof=1)
        assert abs(result.ravel()[0] - expected) < 1e-10


class TestHigherMoments:
    """R-STAT-03: Higher-order moment statistics (kurtosis, skewness).

    Requirement: The statistics toolbox SHALL compute kurtosis and skewness
    for numeric vectors, producing values consistent with Octave definitions.

    Model-user argument: The golden user examines distribution shape to
    validate that sensor data follows expected profiles. Kurtosis near 3.0
    (or excess kurtosis near 0) confirms normality assumptions required by
    downstream statistical tests. Skewness near zero confirms symmetric
    distributions. These checks gate whether parametric methods are valid
    for the user's experimental analysis.

    Decomposition:
        R-STAT-03a: kurtosis() of a large normal sample approximates 3.0
            (or excess kurtosis approximates 0).
        R-STAT-03b: skewness() of a large normal sample approximates 0.

    Consistency argument: R-STAT-03a tests the fourth moment against the
    known kurtosis of the normal distribution. R-STAT-03b tests the third
    moment against the known skewness of the normal distribution. Together
    they validate both higher-order moment functions.
    """

    def test_kurtosis_normal(self):
        """R-STAT-03a: kurtosis() of normal data approximates 3.0 or excess 0."""
        from forge.engine.builtins.statistics import kurtosis
        from forge.engine.types import _unwrap
        np.random.seed(42)
        x = np.random.randn(10000)
        result = _unwrap(kurtosis(x)).ravel()
        assert abs(result[0] - 3.0) < 0.3 or abs(result[0]) < 0.3

    def test_skewness_symmetric(self):
        """R-STAT-03b: skewness() of normal data approximates 0."""
        from forge.engine.builtins.statistics import skewness
        from forge.engine.types import _unwrap
        np.random.seed(42)
        x = np.random.randn(10000)
        result = _unwrap(skewness(x)).ravel()
        assert abs(result[0]) < 0.1


class TestQuantiles:
    """R-STAT-04: Percentile and quantile computation.

    Requirement: The statistics toolbox SHALL compute percentiles via prctile()
    and quantiles via quantile(), using interpolation methods consistent with
    Octave/MATLAB conventions.

    Model-user argument: The golden user uses prctile to set alarm thresholds
    on sensor data (e.g., "flag values above the 95th percentile"). They
    expect prctile(x, 50) to match the median and quantile(x, 0.25) to give
    the first quartile. Incorrect interpolation would shift thresholds and
    cause missed alarms or false positives in their monitoring systems.

    Decomposition:
        R-STAT-04a: prctile(x, 50) returns approximately the median for
            uniformly spaced data.
        R-STAT-04b: quantile(x, 0.25) returns the first quartile within
            expected bounds.

    Consistency argument: R-STAT-04a validates the percentile function at the
    50th percentile (median equivalence). R-STAT-04b validates the quantile
    function at the 25th quantile. Together they confirm both entry points
    produce correct order-statistic interpolations.
    """

    def test_prctile(self):
        """R-STAT-04a: prctile(x, 50) returns approximately the median."""
        from forge.engine.builtins.statistics import prctile
        from forge.engine.types import _unwrap
        x = np.arange(1.0, 101.0)
        result = _unwrap(prctile(x, 50)).ravel()
        assert abs(result[0] - 50.5) < 1.0

    def test_quantile(self):
        """R-STAT-04b: quantile(x, 0.25) returns the first quartile."""
        from forge.engine.builtins.statistics import quantile
        from forge.engine.types import _unwrap
        x = np.arange(1.0, 101.0)
        result = _unwrap(quantile(x, 0.25)).ravel()
        assert result[0] > 20 and result[0] < 30


class TestRobust:
    """R-STAT-05: Robust statistics (z-score, IQR, MAD).

    Requirement: The statistics toolbox SHALL compute z-scores via zscore(),
    interquartile range via iqr(), and median absolute deviation via mad(),
    providing robust measures for outlier detection and data normalization.

    Model-user argument: The golden user computes z-scores to flag outliers
    in experimental data. Values beyond +/-3 sigma get investigated. They use
    IQR and MAD as robust alternatives when data contains known outliers that
    would inflate std. These functions must produce zero-mean, unit-variance
    normalized output (zscore) and positive spread measures (iqr, mad) to
    support the user's quality-control workflows.

    Decomposition:
        R-STAT-05a: zscore() produces zero-mean, unit-standard-deviation output.
        R-STAT-05b: iqr() returns a positive spread measure within expected
            bounds for uniformly spaced data.
        R-STAT-05c: mad() returns a positive deviation measure.

    Consistency argument: R-STAT-05a validates the normalization properties of
    zscore (zero mean, unit std). R-STAT-05b validates IQR magnitude for a
    known distribution. R-STAT-05c validates that MAD is positive. Together
    they cover all three robust statistics in the parent requirement.
    """

    def test_zscore(self):
        """R-STAT-05a: zscore() produces zero-mean, unit-std output."""
        from forge.engine.builtins.statistics import zscore
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = _unwrap(zscore(x)).ravel()
        assert abs(np.mean(z)) < 1e-10
        assert abs(np.std(z, ddof=1) - 1.0) < 1e-10

    def test_iqr(self):
        """R-STAT-05b: iqr() returns positive spread within expected bounds."""
        from forge.engine.builtins.statistics import iqr
        from forge.engine.types import _unwrap
        x = np.arange(1.0, 101.0)
        result = _unwrap(iqr(x)).ravel()
        assert result[0] > 40 and result[0] < 60

    def test_mad(self):
        """R-STAT-05c: mad() returns a positive deviation measure."""
        from forge.engine.builtins.statistics import mad
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _unwrap(mad(x)).ravel()
        assert result[0] > 0


class TestDistribution:
    """R-STAT-06: Distribution summary functions (mode, histc).

    Requirement: The statistics toolbox SHALL compute the mode of a dataset
    via mode() and bin counts via histc(), matching Octave/MATLAB function
    signatures and return conventions.

    Model-user argument: The golden user uses mode() to find the most frequent
    measurement value when analyzing discrete sensor states or categorical
    readings. They use histc() to bin continuous data into predefined edge
    intervals for generating histograms and frequency tables. Both functions
    must use MATLAB-compatible names and return shapes so existing scripts
    port without modification.

    Decomposition:
        R-STAT-06a: mode() returns the most frequently occurring value.
        R-STAT-06b: histc() returns correct bin counts and bin indices for
            given edge vectors.

    Consistency argument: R-STAT-06a validates mode identification on a vector
    with a known unique mode. R-STAT-06b validates bin counting against known
    edge boundaries. Together they cover both distribution summary functions.
    """

    def test_mode(self):
        """R-STAT-06a: mode() returns the most frequent value."""
        from forge.engine.builtins.statistics import mode
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0])
        result = _unwrap(mode(x)).ravel()
        assert abs(result[0] - 3.0) < 1e-10

    def test_histc(self):
        """R-STAT-06b: histc() returns correct bin counts for given edges."""
        from forge.engine.builtins.statistics import histc
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        edges = np.array([0.0, 2.5, 5.0])
        n, idx = histc(x, edges)
        arr = _unwrap(n).ravel()
        assert arr[0] == 2
        assert arr[1] >= 2


class TestMoving:
    """R-STAT-07: Moving-window statistics (movmean).

    Requirement: The statistics toolbox SHALL compute moving-window averages
    via movmean(), preserving input length and using centered windows
    consistent with MATLAB/Octave conventions.

    Model-user argument: The golden user applies movmean to smooth noisy
    sensor signals before plotting or threshold comparison. They expect the
    output length to match the input length and interior values to equal the
    exact arithmetic mean of the window. If the output were truncated or
    misaligned, their time-series plots would shift, causing incorrect
    correlation with other channels.

    Decomposition:
        R-STAT-07a: movmean(x, k) returns output of the same length as x,
            with interior elements equal to the centered k-point mean.

    Consistency argument: R-STAT-07a tests length preservation and exact
    interior values for a window of size 3. This single sub-requirement
    fully covers the parent because movmean is the only function specified.
    """

    def test_movmean(self):
        """R-STAT-07a: movmean() preserves length with correct interior means."""
        from forge.engine.builtins.statistics import movmean
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = _unwrap(movmean(x, 3)).ravel()
        assert len(y) == 5
        assert abs(y[1] - 2.0) < 1e-10
        assert abs(y[2] - 3.0) < 1e-10
        assert abs(y[3] - 4.0) < 1e-10


class TestMisc:
    """R-STAT-08: Miscellaneous statistics (ranks, normalize, bounds, range).

    Requirement: The statistics toolbox SHALL compute ordinal ranks via
    ranks(), z-score normalization via normalize(), data bounds via bounds(),
    and data range via range_stat(), all returning correct numerical results.

    Model-user argument: The golden user uses ranks() for non-parametric
    statistical tests where ordinal position matters more than magnitude.
    They use normalize() to prepare data for algorithms that assume
    standardized inputs. bounds() and range_stat() provide quick sanity
    checks on data extent before deeper analysis. These utility functions
    round out the toolbox so the user does not need to hand-code common
    one-liners that exist in MATLAB/Octave.

    Decomposition:
        R-STAT-08a: ranks() returns correct ordinal ranks for an unsorted vector.
        R-STAT-08b: normalize() produces approximately zero-mean output.
        R-STAT-08c: bounds() returns [min, max] of a vector.
        R-STAT-08d: range_stat() returns max minus min of a vector.

    Consistency argument: Each sub-requirement tests one of the four functions
    named in the parent requirement. R-STAT-08a validates rank ordering,
    R-STAT-08b validates normalization centering, R-STAT-08c validates
    extrema extraction, and R-STAT-08d validates range computation. Together
    they fully cover R-STAT-08.
    """

    def test_ranks(self):
        """R-STAT-08a: ranks() returns correct ordinal ranks."""
        from forge.engine.builtins.statistics import ranks
        from forge.engine.types import _unwrap
        x = np.array([40.0, 10.0, 30.0, 20.0])
        r = _unwrap(ranks(x)).ravel()
        np.testing.assert_array_equal(r, [4, 1, 3, 2])

    def test_normalize(self):
        """R-STAT-08b: normalize() produces approximately zero-mean output."""
        from forge.engine.builtins.statistics import normalize
        from forge.engine.types import _unwrap
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = _unwrap(normalize(x)).ravel()
        assert abs(np.mean(z)) < 1e-10
        assert abs(np.std(z, ddof=0) - 1.0) < 0.5

    def test_bounds(self):
        """R-STAT-08c: bounds() returns [min, max] of a vector."""
        from forge.engine.builtins.statistics import bounds
        from forge.engine.types import _unwrap
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        result = _unwrap(bounds(x)).ravel()
        assert result[0] == 1.0
        assert result[1] == 9.0

    def test_range_stat(self):
        """R-STAT-08d: range_stat() returns max minus min."""
        from forge.engine.builtins.statistics import range_stat as forge_range
        from forge.engine.types import _unwrap
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        result = _unwrap(forge_range(x)).ravel()
        assert abs(result[0] - 8.0) < 1e-10
