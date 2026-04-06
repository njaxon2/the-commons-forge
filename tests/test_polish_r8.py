# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish R8: String & I/O builtins (strtrim, deblank, fliplr, blanks,
native2unicode, unicode2native, dec2hex, hex2dec, dec2bin, bin2dec,
dec2base, base2dec, char, double).

V-model traceability backfill: R-POL8-01 through R-POL8-03.
"""

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
    """R-POL8-01: String manipulation builtins (strtrim, deblank, blanks,
    fliplr, char, double) SHALL produce correct results matching Octave
    semantics.

    Model-user argument: Engineers processing text data (sensor labels,
    file headers, instrument output) rely on strtrim to clean whitespace,
    deblank to strip trailing spaces, fliplr to reverse strings, and
    char/double for ASCII code-point conversion. These are basic string
    operations that appear in virtually every data-import script.

    Decomposition:
      R-POL8-01a: strtrim removes leading and trailing whitespace.
      R-POL8-01b: strtrim removes leading and trailing tab characters.
      R-POL8-01c: deblank removes trailing whitespace only.
      R-POL8-01d: deblank preserves leading whitespace.
      R-POL8-01e: blanks(n) returns n space characters.
      R-POL8-01f: fliplr reverses a character string.
      R-POL8-01g: fliplr reverses columns of a numeric matrix.
      R-POL8-01h: char(n) converts ASCII code to character.
      R-POL8-01i: char(vector) converts ASCII code vector to string.
      R-POL8-01j: double('A') returns ASCII code 65.
      R-POL8-01k: double('Hi') returns ASCII code vector.

    Consistency: These eleven sub-requirements cover every string builtin
    listed in the parent, including both string and numeric inputs for
    fliplr, char, and double.
    """

    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    # -- strtrim --
    def test_strtrim_basic(self):
        """R-POL8-01a: strtrim removes leading and trailing whitespace."""
        self.s.eval("x = strtrim('  hello  ')")
        assert _str(self.s, "x") == "hello"

    def test_strtrim_tabs(self):
        """R-POL8-01b: strtrim removes leading and trailing tabs."""
        self.s.eval("x = strtrim(char([9 72 105 9]))")
        assert _str(self.s, "x") == "Hi"

    # -- deblank --
    def test_deblank_trailing(self):
        """R-POL8-01c: deblank removes trailing whitespace."""
        self.s.eval("x = deblank('hello   ')")
        assert _str(self.s, "x") == "hello"

    def test_deblank_preserves_leading(self):
        """R-POL8-01d: deblank preserves leading whitespace."""
        self.s.eval("x = deblank('  hello  ')")
        assert _str(self.s, "x") == "  hello"

    # -- blanks --
    def test_blanks(self):
        """R-POL8-01e: blanks(5) returns 5 space characters."""
        self.s.eval("x = blanks(5)")
        assert _str(self.s, "x") == "     "

    # -- fliplr on char --
    def test_fliplr_string(self):
        """R-POL8-01f: fliplr reverses a character string."""
        self.s.eval("x = fliplr('abcde')")
        assert _str(self.s, "x") == "edcba"

    def test_fliplr_matrix(self):
        """R-POL8-01g: fliplr reverses columns of a numeric matrix."""
        self.s.eval("x = fliplr([1 2 3; 4 5 6])")
        ws = self.s.get_workspace_dict()
        arr = ws["x"].data
        assert arr[0, 0] == 3
        assert arr[1, 2] == 4

    # -- char(n) --
    def test_char_scalar(self):
        """R-POL8-01h: char(65) returns 'A'."""
        self.s.eval("x = char(65)")
        assert _str(self.s, "x") == "A"

    def test_char_vector(self):
        """R-POL8-01i: char([72 101 108]) returns 'Hel'."""
        self.s.eval("x = char([72 101 108])")
        assert _str(self.s, "x") == "Hel"

    # -- double('A') --
    def test_double_char_scalar(self):
        """R-POL8-01j: double('A') returns 65."""
        self.s.eval("x = double('A')")
        assert _val(self.s, "x") == 65.0

    def test_double_char_string(self):
        """R-POL8-01k: double('Hi') returns [72, 105]."""
        self.s.eval("x = double('Hi')")
        ws = self.s.get_workspace_dict()
        arr = ws["x"].data.flatten()
        np.testing.assert_array_equal(arr, [72.0, 105.0])


class TestBaseConversion:
    """R-POL8-02: Base-conversion builtins (dec2hex, hex2dec, dec2bin,
    bin2dec, dec2base, base2dec) SHALL convert between numeric values
    and their string representations in arbitrary bases.

    Model-user argument: Engineers working with embedded systems,
    communication protocols, and binary file formats constantly convert
    between decimal, hex, binary, and octal representations. These
    functions are used in register manipulation, bitfield parsing,
    and protocol decoding scripts migrated from Octave.

    Decomposition:
      R-POL8-02a: dec2hex(255) returns 'FF'.
      R-POL8-02b: hex2dec('FF') returns 255.
      R-POL8-02c: dec2hex with minimum length zero-pads.
      R-POL8-02d: dec2bin(10) returns '1010'.
      R-POL8-02e: bin2dec('1010') returns 10.
      R-POL8-02f: dec2bin with minimum length zero-pads.
      R-POL8-02g: dec2base(8, 8) returns '10' (octal).
      R-POL8-02h: dec2base(255, 16) returns 'FF' (hex via base).
      R-POL8-02i: base2dec('17', 8) returns 15 (octal to decimal).
      R-POL8-02j: dec2base with minimum length zero-pads.
      R-POL8-02k: base2dec('Z', 36) returns 35 (base-36).

    Consistency: Hex, binary, and arbitrary-base conversions with both
    directions and zero-padding cover the complete API surface.
    """

    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    # -- dec2hex / hex2dec --
    def test_dec2hex(self):
        """R-POL8-02a: dec2hex(255) returns 'FF'."""
        self.s.eval("x = dec2hex(255)")
        assert _str(self.s, "x") == "FF"

    def test_hex2dec(self):
        """R-POL8-02b: hex2dec('FF') returns 255."""
        self.s.eval("x = hex2dec('FF')")
        assert _val(self.s, "x") == 255.0

    def test_dec2hex_minlen(self):
        """R-POL8-02c: dec2hex(10, 4) zero-pads to '000A'."""
        self.s.eval("x = dec2hex(10, 4)")
        assert _str(self.s, "x") == "000A"

    # -- dec2bin / bin2dec --
    def test_dec2bin(self):
        """R-POL8-02d: dec2bin(10) returns '1010'."""
        self.s.eval("x = dec2bin(10)")
        assert _str(self.s, "x") == "1010"

    def test_bin2dec(self):
        """R-POL8-02e: bin2dec('1010') returns 10."""
        self.s.eval("x = bin2dec('1010')")
        assert _val(self.s, "x") == 10.0

    def test_dec2bin_minlen(self):
        """R-POL8-02f: dec2bin(5, 8) zero-pads to '00000101'."""
        self.s.eval("x = dec2bin(5, 8)")
        assert _str(self.s, "x") == "00000101"

    # -- dec2base / base2dec --
    def test_dec2base_octal(self):
        """R-POL8-02g: dec2base(8, 8) returns '10' (octal)."""
        self.s.eval("x = dec2base(8, 8)")
        assert _str(self.s, "x") == "10"

    def test_dec2base_hex(self):
        """R-POL8-02h: dec2base(255, 16) returns 'FF'."""
        self.s.eval("x = dec2base(255, 16)")
        assert _str(self.s, "x") == "FF"

    def test_base2dec_octal(self):
        """R-POL8-02i: base2dec('17', 8) returns 15."""
        self.s.eval("x = base2dec('17', 8)")
        assert _val(self.s, "x") == 15.0

    def test_dec2base_minlen(self):
        """R-POL8-02j: dec2base(3, 2, 8) zero-pads to '00000011'."""
        self.s.eval("x = dec2base(3, 2, 8)")
        assert _str(self.s, "x") == "00000011"

    def test_base2dec_base36(self):
        """R-POL8-02k: base2dec('Z', 36) returns 35."""
        self.s.eval("x = base2dec('Z', 36)")
        assert _val(self.s, "x") == 35.0


class TestEncodingConversion:
    """R-POL8-03: Encoding conversion builtins (native2unicode,
    unicode2native) SHALL convert between byte arrays and Unicode
    strings.

    Model-user argument: Engineers processing data from legacy
    instruments or international text files use native2unicode and
    unicode2native for character encoding conversion. These functions
    bridge between raw byte values and displayable strings.

    Decomposition:
      R-POL8-03a: native2unicode converts byte array to string.
      R-POL8-03b: unicode2native converts character to byte value.

    Consistency: Bytes-to-string and string-to-bytes cover both
    directions of the encoding conversion.
    """

    def setup_method(self):
        from forge.engine.session import ForgeSession
        self.s = ForgeSession()

    def test_native2unicode(self):
        """R-POL8-03a: native2unicode([72 101 108 108 111]) -> 'Hello'."""
        self.s.eval("x = native2unicode([72 101 108 108 111])")
        assert "Hello" in _str(self.s, "x")

    def test_unicode2native(self):
        """R-POL8-03b: unicode2native('A') -> 65."""
        self.s.eval("x = unicode2native('A')")
        ws = self.s.get_workspace_dict()
        assert ws["x"].data.flat[0] == 65
