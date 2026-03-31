# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish R8 -- String & I/O builtins (strtrim, deblank, fliplr, blanks,
native2unicode, unicode2native, dec2hex, hex2dec, dec2bin, bin2dec,
dec2base, base2dec, char, double)."""

import pytest, numpy as np


def _val(session, name):
    """Extract scalar Python value from workspace variable."""
    from forge.engine.types import _unwrap
    ws = session.get_workspace_dict()
    v = _unwrap(ws[name])
    if isinstance(v, np.ndarray):
        return v.item() if v.size == 1 else v
    return v


def _str(session, name):
    """Extract string from workspace ForgeChar variable."""
    ws = session.get_workspace_dict()
    v = ws[name]
    return str(v)


class TestStringBuiltins:
    """Tests for strtrim, deblank, blanks, fliplr, char, double."""

    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    # -- strtrim --
    def test_strtrim_basic(self):
        self.s.eval("x = strtrim('  hello  ')")
        assert _str(self.s, "x") == "hello"

    def test_strtrim_tabs(self):
        self.s.eval("x = strtrim(char([9 72 105 9]))")
        assert _str(self.s, "x") == "Hi"

    # -- deblank --
    def test_deblank_trailing(self):
        self.s.eval("x = deblank('hello   ')")
        assert _str(self.s, "x") == "hello"

    def test_deblank_preserves_leading(self):
        self.s.eval("x = deblank('  hello  ')")
        assert _str(self.s, "x") == "  hello"

    # -- blanks --
    def test_blanks(self):
        self.s.eval("x = blanks(5)")
        assert _str(self.s, "x") == "     "

    # -- fliplr on char --
    def test_fliplr_string(self):
        self.s.eval("x = fliplr('abcde')")
        assert _str(self.s, "x") == "edcba"

    def test_fliplr_matrix(self):
        self.s.eval("x = fliplr([1 2 3; 4 5 6])")
        ws = self.s.get_workspace_dict()
        arr = ws["x"].data
        assert arr[0, 0] == 3
        assert arr[1, 2] == 4

    # -- char(n) --
    def test_char_scalar(self):
        self.s.eval("x = char(65)")
        assert _str(self.s, "x") == "A"

    def test_char_vector(self):
        self.s.eval("x = char([72 101 108])")
        assert _str(self.s, "x") == "Hel"

    # -- double('A') --
    def test_double_char_scalar(self):
        self.s.eval("x = double('A')")
        assert _val(self.s, "x") == 65.0

    def test_double_char_string(self):
        self.s.eval("x = double('Hi')")
        ws = self.s.get_workspace_dict()
        arr = ws["x"].data.flatten()
        np.testing.assert_array_equal(arr, [72.0, 105.0])


class TestBaseConversion:
    """Tests for dec2hex, hex2dec, dec2bin, bin2dec, dec2base, base2dec."""

    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    # -- dec2hex / hex2dec --
    def test_dec2hex(self):
        self.s.eval("x = dec2hex(255)")
        assert _str(self.s, "x") == "FF"

    def test_hex2dec(self):
        self.s.eval("x = hex2dec('FF')")
        assert _val(self.s, "x") == 255.0

    def test_dec2hex_minlen(self):
        self.s.eval("x = dec2hex(10, 4)")
        assert _str(self.s, "x") == "000A"

    # -- dec2bin / bin2dec --
    def test_dec2bin(self):
        self.s.eval("x = dec2bin(10)")
        assert _str(self.s, "x") == "1010"

    def test_bin2dec(self):
        self.s.eval("x = bin2dec('1010')")
        assert _val(self.s, "x") == 10.0

    def test_dec2bin_minlen(self):
        self.s.eval("x = dec2bin(5, 8)")
        assert _str(self.s, "x") == "00000101"

    # -- dec2base / base2dec --
    def test_dec2base_octal(self):
        self.s.eval("x = dec2base(8, 8)")
        assert _str(self.s, "x") == "10"

    def test_dec2base_hex(self):
        self.s.eval("x = dec2base(255, 16)")
        assert _str(self.s, "x") == "FF"

    def test_base2dec_octal(self):
        self.s.eval("x = base2dec('17', 8)")
        assert _val(self.s, "x") == 15.0

    def test_dec2base_minlen(self):
        self.s.eval("x = dec2base(3, 2, 8)")
        assert _str(self.s, "x") == "00000011"

    def test_base2dec_base36(self):
        self.s.eval("x = base2dec('Z', 36)")
        assert _val(self.s, "x") == 35.0


class TestEncodingConversion:
    """Tests for native2unicode and unicode2native."""

    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    def test_native2unicode(self):
        self.s.eval("x = native2unicode([72 101 108 108 111])")
        assert "Hello" in _str(self.s, "x")

    def test_unicode2native(self):
        self.s.eval("x = unicode2native('A')")
        ws = self.s.get_workspace_dict()
        assert ws["x"].data.flat[0] == 65
