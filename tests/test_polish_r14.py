# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Polish round 14: regex functions and type conversions.

Requirement R-POL14-01:
    The regular expression functions (regexp, regexpi, regexprep) SHALL
    return correct start indices, matched substrings, and substituted
    strings, with regexpi providing case-insensitive matching.

    Model-user argument:
    An engineer migrating from MATLAB/Octave parses instrument log files,
    serial-port data, and configuration files using regexp. If regexp
    returns 0-based indices instead of 1-based, or if regexpi fails to
    match case-insensitively, text-processing scripts that have worked
    for years in MATLAB silently produce wrong extractions. Correct
    1-based indexing and option handling are essential for porting.

    Decomposition:
    R-POL14-01a: regexp returns 1-based start indices for all matches.
    R-POL14-01b: regexp returns empty array when no match exists.
    R-POL14-01c: regexp returns correct index for a single match.
    R-POL14-01d: regexp with 'match' option returns cell of matched strings.
    R-POL14-01e: regexp 'match' returns empty cell on no match.
    R-POL14-01f: regexpi matches case-insensitively (start indices).
    R-POL14-01g: regexpi with 'match' returns original-case substrings.
    R-POL14-01h: regexprep substitutes matched text with replacement.
    R-POL14-01i: regexprep replaces all occurrences in the string.

    Consistency argument:
    Sub-requirements 01a-01e cover regexp with default and 'match' modes,
    including edge cases. 01f-01g cover regexpi. 01h-01i cover regexprep.
    Together they verify all three regex functions.

Requirement R-POL14-02:
    The integer and floating-point type conversion functions (int32, uint8,
    int16, uint16, single, logical) SHALL round (not truncate), saturate
    (not wrap), and preserve correct dtype semantics matching MATLAB/Octave
    behavior.

    Model-user argument:
    An engineer processing sensor data and images relies on integer type
    casts behaving identically to MATLAB. If uint8(300) wraps to 44
    instead of saturating to 255, image processing pipelines produce
    corrupt pixel values. If int32(3.7) truncates to 3 instead of
    rounding to 4, quantization steps in signal processing accumulate
    systematic bias.

    Decomposition:
    R-POL14-02a: int32 rounds 3.7 to 4 (not truncate to 3).
    R-POL14-02b: int32 rounds 3.2 down to 3.
    R-POL14-02c: int32 rounds -2.6 to -3.
    R-POL14-02d: uint8 saturates 300 to 255 (not wrap).
    R-POL14-02e: uint8 saturates -5 to 0.
    R-POL14-02f: uint8 passes 100 unchanged.
    R-POL14-02g: single produces float32 dtype with correct value.
    R-POL14-02h: logical converts nonzero to true, zero to false.
    R-POL14-02i: logical produces bool_ dtype.
    R-POL14-02j: int16 saturates 40000 to 32767.
    R-POL14-02k: uint16 rounds 3.5 to 4.

    Consistency argument:
    Sub-requirements 02a-02c test int32 rounding in three cases. 02d-02f
    test uint8 saturation and pass-through. 02g tests single precision.
    02h-02i test logical conversion and dtype. 02j-02k test int16 and
    uint16. Together they cover every type conversion function in the
    parent requirement.
"""
import pytest
import numpy as np


class TestRegexp:
    """R-POL14-01: Regular expression matching, case-insensitive matching,
    and substitution.
    """

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
        """R-POL14-01a: regexp returns 1-based start indices for all matches."""
        self.s.eval(r'r = regexp("hello world", "\w+")')
        v = self._val("r")
        np.testing.assert_array_equal(v.flatten(), [1, 7])

    def test_regexp_indices_no_match(self):
        """R-POL14-01b: regexp returns empty array when no match exists."""
        self.s.eval(r'r = regexp("hello", "\d+")')
        v = self._val("r")
        assert v.size == 0

    def test_regexp_single_match(self):
        """R-POL14-01c: regexp returns correct index for a single match."""
        self.s.eval(r'r = regexp("abc123", "\d+")')
        v = self._val("r")
        np.testing.assert_array_equal(v.flatten(), [4])

    # --- regexp: match option ---
    def test_regexp_match_option(self):
        """R-POL14-01d: regexp with 'match' returns cell of matched strings."""
        self.s.eval(r'r = regexp("hello world", "\w+", "match")')
        raw = self._raw("r")
        from forge.engine.containers import ForgeCell
        assert isinstance(raw, ForgeCell)
        strs = [str(x) for x in raw._data]
        assert strs == ["hello", "world"]

    def test_regexp_match_no_match(self):
        """R-POL14-01e: regexp 'match' returns empty cell on no match."""
        self.s.eval(r'r = regexp("hello", "\d+", "match")')
        raw = self._raw("r")
        from forge.engine.containers import ForgeCell
        assert isinstance(raw, ForgeCell)
        assert len(raw._data) == 0

    # --- regexpi: case-insensitive ---
    def test_regexpi_basic(self):
        """R-POL14-01f: regexpi matches case-insensitively."""
        self.s.eval('r = regexpi("Hello World", "hello")')
        v = self._val("r")
        np.testing.assert_array_equal(v.flatten(), [1])

    def test_regexpi_match_option(self):
        """R-POL14-01g: regexpi 'match' returns original-case substrings."""
        self.s.eval('r = regexpi("Hello WORLD", "[a-z]+", "match")')
        raw = self._raw("r")
        from forge.engine.containers import ForgeCell
        assert isinstance(raw, ForgeCell)
        # case-insensitive: should match both words
        strs = [str(x) for x in raw._data]
        assert strs == ["Hello", "WORLD"]

    # --- regexprep ---
    def test_regexprep_basic(self):
        """R-POL14-01h: regexprep substitutes matched text."""
        self.s.eval('r = regexprep("hello world", "world", "earth")')
        assert str(self._raw("r")) == "hello earth"

    def test_regexprep_pattern(self):
        """R-POL14-01i: regexprep replaces all occurrences."""
        self.s.eval(r'r = regexprep("abc123def456", "\d+", "NUM")')
        assert str(self._raw("r")) == "abcNUMdefNUM"

    # --- int32: round (not truncate) ---
    def test_int32_rounds(self):
        """R-POL14-02a: int32(3.7) rounds to 4."""
        self.s.eval("a = int32(3.7)")
        v = self._val("a")
        assert v.flatten()[0] == 4

    def test_int32_rounds_down(self):
        """R-POL14-02b: int32(3.2) rounds to 3."""
        self.s.eval("a = int32(3.2)")
        v = self._val("a")
        assert v.flatten()[0] == 3

    def test_int32_negative(self):
        """R-POL14-02c: int32(-2.6) rounds to -3."""
        self.s.eval("a = int32(-2.6)")
        v = self._val("a")
        assert v.flatten()[0] == -3

    # --- uint8: saturate (not wrap) ---
    def test_uint8_saturates_high(self):
        """R-POL14-02d: uint8(300) saturates to 255."""
        self.s.eval("a = uint8(300)")
        v = self._val("a")
        assert v.flatten()[0] == 255

    def test_uint8_saturates_low(self):
        """R-POL14-02e: uint8(-5) saturates to 0."""
        self.s.eval("a = uint8(-5)")
        v = self._val("a")
        assert v.flatten()[0] == 0

    def test_uint8_normal(self):
        """R-POL14-02f: uint8(100) passes through unchanged."""
        self.s.eval("a = uint8(100)")
        v = self._val("a")
        assert v.flatten()[0] == 100

    # --- single precision ---
    def test_single_precision(self):
        """R-POL14-02g: single(pi) produces float32 with correct value."""
        self.s.eval("a = single(pi)")
        v = self._val("a")
        assert v.dtype == np.float32
        np.testing.assert_allclose(v.flatten()[0], np.float32(np.pi))

    # --- logical ---
    def test_logical_conversion(self):
        """R-POL14-02h: logical converts nonzero to true, zero to false."""
        self.s.eval("a = logical([0 1 2 0])")
        v = self._val("a")
        np.testing.assert_array_equal(v.flatten(), [False, True, True, False])

    def test_logical_dtype(self):
        """R-POL14-02i: logical produces bool_ dtype."""
        self.s.eval("a = logical([0 1])")
        v = self._val("a")
        assert v.dtype == np.bool_

    # --- int16 saturate ---
    def test_int16_saturates(self):
        """R-POL14-02j: int16(40000) saturates to 32767."""
        self.s.eval("a = int16(40000)")
        v = self._val("a")
        assert v.flatten()[0] == 32767

    # --- uint16 round ---
    def test_uint16_rounds(self):
        """R-POL14-02k: uint16(3.5) rounds to 4."""
        self.s.eval("a = uint16(3.5)")
        v = self._val("a")
        assert v.flatten()[0] == 4
