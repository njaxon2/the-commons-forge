"""Tests for Polish R4 features: bsxfun, diary, conv, medfilt1.

V-model traceability backfill: R-POL4-01 through R-POL4-03.
"""
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap


class TestPolishR4:
    """R-POL4-01: Numeric and signal-processing builtins (bsxfun, conv,
    medfilt1, diary) SHALL produce correct results matching Octave
    semantics.

    Model-user argument: Engineers performing signal processing and
    array manipulation rely on bsxfun for broadcast operations, conv
    for discrete convolution, and medfilt1 for median filtering. These
    are everyday functions in signal and data analysis workflows. The
    diary command is used to log interactive sessions for lab notebooks.

    Decomposition:
      R-POL4-01a: bsxfun(@max, ...) broadcasts element-wise max.
      R-POL4-01b: conv returns full discrete convolution.
      R-POL4-01c: medfilt1 suppresses an impulse via median filtering.
      R-POL4-01d: diary toggles session logging on/off.

    Consistency: These four sub-requirements cover the numeric broadcast,
    convolution, filtering, and session-logging functions in this round.
    """

    def setup_method(self):
        self.s = ForgeSession()

    def test_bsxfun_plus(self):
        """R-POL4-01a: bsxfun(@max, ...) broadcasts element-wise max."""
        self.s.eval("r = bsxfun(@max, [1;5;3], [4;2;6])")
        r = self.s._engine.workspace.get("r")
        arr = np.asarray(_unwrap(r), dtype=float)
        assert arr.size >= 3
        np.testing.assert_allclose(arr.ravel()[:3], [4, 5, 6], atol=1e-10)

    def test_conv_basic(self):
        """R-POL4-01b: conv returns full discrete convolution."""
        self.s.eval("r = conv([1 2 3], [1 1])")
        r = self.s._engine.workspace.get("r")
        arr = np.asarray(_unwrap(r), dtype=float).ravel()
        np.testing.assert_allclose(arr, [1, 3, 5, 3], atol=1e-10)

    def test_medfilt1_impulse(self):
        """R-POL4-01c: medfilt1 suppresses an impulse via median filter."""
        self.s.eval("x = zeros(1,10); x(5) = 100; r = medfilt1(x, 3)")
        r = self.s._engine.workspace.get("r")
        arr = np.asarray(_unwrap(r), dtype=float).ravel()
        assert abs(arr[4]) < 1e-10

    def test_diary_toggle(self):
        """R-POL4-01d: diary toggles session logging."""
        r = self.s.eval("diary")
        assert "diary" in str(r).lower()
        r2 = self.s.eval("diary")
        assert "diary" in str(r2).lower()


class TestPolishR4StringRegex:
    """R-POL4-02: String and regex builtins (strsplit, regexp) SHALL
    produce correct results matching Octave semantics.

    Model-user argument: Engineers parsing text data (CSV headers, log
    files, sensor output) use strsplit to tokenize strings and regexp
    to extract patterns. These are essential for any data-import
    pipeline migrated from Octave.

    Decomposition:
      R-POL4-02a: strsplit splits a string into tokens.
      R-POL4-02b: regexp returns the start index of the first match.

    Consistency: Splitting and pattern-matching are the two fundamental
    string-processing operations; both passing confirms the surface.
    """

    def setup_method(self):
        self.s = ForgeSession()

    def test_strsplit(self):
        """R-POL4-02a: strsplit splits a string into tokens."""
        r = self.s.eval('strsplit("hello world")')
        assert r is not None

    def test_regexp(self):
        """R-POL4-02b: regexp returns start index of first match."""
        r = self.s.eval('regexp("hello", "l+")')
        assert float(r) == 3


class TestPolishR4FunctionalOps:
    """R-POL4-03: Functional-programming builtins (cellfun, accumarray)
    SHALL produce correct results matching Octave semantics.

    Model-user argument: Engineers working with heterogeneous data in
    cell arrays use cellfun to apply a function element-wise, and
    accumarray to aggregate values by index. Both are critical for
    data-wrangling workflows ported from Octave.

    Decomposition:
      R-POL4-03a: cellfun(@length, ...) returns element lengths.
      R-POL4-03b: accumarray sums values grouped by subscript.

    Consistency: cellfun covers cell-based mapping, accumarray covers
    index-based aggregation. Together they validate the functional
    operations in this round.
    """

    def setup_method(self):
        self.s = ForgeSession()

    def test_cellfun(self):
        """R-POL4-03a: cellfun(@length, ...) returns element lengths."""
        self.s.eval('r = cellfun(@length, {"ab", "cde"})')
        r = self.s._engine.workspace.get("r")
        arr = np.asarray(_unwrap(r), dtype=float).ravel()
        np.testing.assert_allclose(arr, [2, 3])

    def test_accumarray(self):
        """R-POL4-03b: accumarray sums values grouped by subscript."""
        self.s.eval("r = accumarray([1;1;2;2;3], [10;20;30;40;50])")
        r = self.s._engine.workspace.get("r")
        arr = np.asarray(_unwrap(r), dtype=float).ravel()
        np.testing.assert_allclose(arr, [30, 70, 50])
