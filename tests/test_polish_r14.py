# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Polish round 14: regex functions and type conversions."""
import pytest
import numpy as np


class TestRegexp:
    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    def _val(self, name):
        from forge.engine.types import _unwrap
        ws = self.s.get_workspace_dict()
        return _unwrap(ws[name])

    def _raw(self, name):
        ws = self.s.get_workspace_dict()
        return ws[name]

    # --- regexp: start indices ---
    def test_regexp_indices_basic(self):
        self.s.eval(r'r = regexp("hello world", "\w+")')
        v = self._val("r")
        np.testing.assert_array_equal(v.flatten(), [1, 7])

    def test_regexp_indices_no_match(self):
        self.s.eval(r'r = regexp("hello", "\d+")')
        v = self._val("r")
        assert v.size == 0

    def test_regexp_single_match(self):
        self.s.eval(r'r = regexp("abc123", "\d+")')
        v = self._val("r")
        np.testing.assert_array_equal(v.flatten(), [4])

    # --- regexp: match option ---
    def test_regexp_match_option(self):
        self.s.eval(r'r = regexp("hello world", "\w+", "match")')
        raw = self._raw("r")
        from forge.engine.containers import ForgeCell
        assert isinstance(raw, ForgeCell)
        strs = [str(x) for x in raw._data]
        assert strs == ["hello", "world"]

    def test_regexp_match_no_match(self):
        self.s.eval(r'r = regexp("hello", "\d+", "match")')
        raw = self._raw("r")
        from forge.engine.containers import ForgeCell
        assert isinstance(raw, ForgeCell)
        assert len(raw._data) == 0

    # --- regexpi: case-insensitive ---
    def test_regexpi_basic(self):
        self.s.eval('r = regexpi("Hello World", "hello")')
        v = self._val("r")
        np.testing.assert_array_equal(v.flatten(), [1])

    def test_regexpi_match_option(self):
        self.s.eval('r = regexpi("Hello WORLD", "[a-z]+", "match")')
        raw = self._raw("r")
        from forge.engine.containers import ForgeCell
        assert isinstance(raw, ForgeCell)
        # case-insensitive: should match both words
        strs = [str(x) for x in raw._data]
        assert strs == ["Hello", "WORLD"]

    # --- regexprep ---
    def test_regexprep_basic(self):
        self.s.eval('r = regexprep("hello world", "world", "earth")')
        assert str(self._raw("r")) == "hello earth"

    def test_regexprep_pattern(self):
        self.s.eval(r'r = regexprep("abc123def456", "\d+", "NUM")')
        assert str(self._raw("r")) == "abcNUMdefNUM"

    # --- int32: round (not truncate) ---
    def test_int32_rounds(self):
        self.s.eval("a = int32(3.7)")
        v = self._val("a")
        assert v.flatten()[0] == 4

    def test_int32_rounds_down(self):
        self.s.eval("a = int32(3.2)")
        v = self._val("a")
        assert v.flatten()[0] == 3

    def test_int32_negative(self):
        self.s.eval("a = int32(-2.6)")
        v = self._val("a")
        assert v.flatten()[0] == -3

    # --- uint8: saturate (not wrap) ---
    def test_uint8_saturates_high(self):
        self.s.eval("a = uint8(300)")
        v = self._val("a")
        assert v.flatten()[0] == 255

    def test_uint8_saturates_low(self):
        self.s.eval("a = uint8(-5)")
        v = self._val("a")
        assert v.flatten()[0] == 0

    def test_uint8_normal(self):
        self.s.eval("a = uint8(100)")
        v = self._val("a")
        assert v.flatten()[0] == 100

    # --- single precision ---
    def test_single_precision(self):
        self.s.eval("a = single(pi)")
        v = self._val("a")
        assert v.dtype == np.float32
        np.testing.assert_allclose(v.flatten()[0], np.float32(np.pi))

    # --- logical ---
    def test_logical_conversion(self):
        self.s.eval("a = logical([0 1 2 0])")
        v = self._val("a")
        np.testing.assert_array_equal(v.flatten(), [False, True, True, False])

    def test_logical_dtype(self):
        self.s.eval("a = logical([0 1])")
        v = self._val("a")
        assert v.dtype == np.bool_

    # --- int16 saturate ---
    def test_int16_saturates(self):
        self.s.eval("a = int16(40000)")
        v = self._val("a")
        assert v.flatten()[0] == 32767

    # --- uint16 round ---
    def test_uint16_rounds(self):
        self.s.eval("a = uint16(3.5)")
        v = self._val("a")
        assert v.flatten()[0] == 4
