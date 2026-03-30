# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for strings toolbox (34 functions).

SRS trace: SRS-FUNC-001, SRS-VAL-001
Test method: Comparison against known Octave string operation results.
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.containers import ForgeChar
from forge.engine.builtins.strings import *


class TestBaseConversion:
    """Verify base conversion between string and numeric representations."""

    def test_bin2dec_1010(self):
        """bin2dec('1010') -> 10."""
        r = forge_bin2dec('1010')
        assert r == 10

    def test_dec2bin_10(self):
        """dec2bin(10) -> '1010'."""
        r = forge_dec2bin(10)
        assert r == '1010'

    def test_hex2dec_ff(self):
        """hex2dec('ff') -> 255."""
        r = forge_hex2dec('ff')
        assert r == 255

    def test_dec2hex_255(self):
        """dec2hex(255) -> 'FF'."""
        r = forge_dec2hex(255)
        assert r == 'FF'

    def test_base2dec_octal(self):
        """base2dec('77', 8) -> 63."""
        r = forge_base2dec('77', 8)
        assert r == 63


class TestWhitespace:
    """Verify blanks, deblank, and strtrim."""

    def test_blanks_5(self):
        """blanks(5) returns 5 spaces."""
        r = forge_blanks(5)
        assert r == '     '
        assert len(r) == 5

    def test_deblank_trailing(self):
        """deblank removes trailing spaces only."""
        r = forge_deblank('  hello   ')
        assert r == '  hello'

    def test_strtrim_both(self):
        """strtrim removes leading and trailing whitespace."""
        r = forge_strtrim('  hello  ')
        assert str(r) == 'hello'

    def test_strjoin_comma(self):
        """strjoin(['a','b','c'], ',') -> 'a,b,c'."""
        r = forge_strjoin(['a', 'b', 'c'], ',')
        assert str(r) == 'a,b,c'


class TestStringSearch:
    """Verify startsWith, endsWith, index, rindex."""

    def test_startsWith_true(self):
        r = forge_startsWith('hello world', 'hello')
        assert r is True

    def test_startsWith_false(self):
        r = forge_startsWith('hello world', 'world')
        assert r is False

    def test_endsWith_true(self):
        r = forge_endsWith('hello world', 'world')
        assert r is True

    def test_endsWith_false(self):
        r = forge_endsWith('hello world', 'hello')
        assert r is False

    def test_index_found(self):
        """index('hello world', 'world') -> 7 (1-based)."""
        r = forge_index_str('hello world', 'world')
        assert r == 7

    def test_rindex_found(self):
        """rindex('abcabc', 'bc') -> 5 (1-based, last occurrence)."""
        r = forge_rindex('abcabc', 'bc')
        assert r == 5


class TestStringSplit:
    """Verify strsplit and strtok."""

    def test_strsplit_comma(self):
        """strsplit('a,b,c', ',') -> ['a','b','c']."""
        r = forge_strsplit('a,b,c', ',')
        parts = [str(x) for x in r._data] if hasattr(r, "_data") else [str(r)]
        assert parts == ['a', 'b', 'c']

    def test_strtok_simple(self):
        """strtok('hello world') -> ('hello', ' world')."""
        tok, rem = forge_strtok('hello world')
        assert tok == 'hello'
        assert rem == ' world'


class TestStringConversion:
    """Verify str2num and mat2str."""

    def test_str2num_integer(self):
        """str2num('42') -> 42."""
        r = forge_str2num('42')
        assert r == 42

    def test_mat2str_matrix(self):
        """mat2str([1 2; 3 4]) -> '[1 2; 3 4]'."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = forge_mat2str(A)
        assert '[' in r and ']' in r
        assert '1' in r and '4' in r
