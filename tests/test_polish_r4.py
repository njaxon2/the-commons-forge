"""Tests for Polish R4 features: bsxfun, diary, conv, medfilt1."""
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap


class TestPolishR4:
    def setup_method(self):
        self.s = ForgeSession()

    def test_bsxfun_plus(self):
        self.s.eval("r = bsxfun(@max, [1;5;3], [4;2;6])")
        r = self.s._engine.workspace.get("r")
        arr = np.asarray(_unwrap(r), dtype=float)
        assert arr.size >= 3
        np.testing.assert_allclose(arr.ravel()[:3], [4, 5, 6], atol=1e-10)

    def test_conv_basic(self):
        self.s.eval("r = conv([1 2 3], [1 1])")
        r = self.s._engine.workspace.get("r")
        arr = np.asarray(_unwrap(r), dtype=float).ravel()
        np.testing.assert_allclose(arr, [1, 3, 5, 3], atol=1e-10)

    def test_medfilt1_impulse(self):
        self.s.eval("x = zeros(1,10); x(5) = 100; r = medfilt1(x, 3)")
        r = self.s._engine.workspace.get("r")
        arr = np.asarray(_unwrap(r), dtype=float).ravel()
        assert abs(arr[4]) < 1e-10

    def test_diary_toggle(self):
        r = self.s.eval("diary")
        assert "diary" in str(r).lower()
        r2 = self.s.eval("diary")
        assert "diary" in str(r2).lower()

    def test_strsplit(self):
        r = self.s.eval('strsplit("hello world")')
        assert r is not None

    def test_regexp(self):
        r = self.s.eval('regexp("hello", "l+")')
        assert float(r) == 3

    def test_cellfun(self):
        self.s.eval('r = cellfun(@length, {"ab", "cde"})')
        r = self.s._engine.workspace.get("r")
        arr = np.asarray(_unwrap(r), dtype=float).ravel()
        np.testing.assert_allclose(arr, [2, 3])

    def test_accumarray(self):
        self.s.eval("r = accumarray([1;1;2;2;3], [10;20;30;40;50])")
        r = self.s._engine.workspace.get("r")
        arr = np.asarray(_unwrap(r), dtype=float).ravel()
        np.testing.assert_allclose(arr, [30, 70, 50])
