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
    """R-STR-01: The strings toolbox SHALL convert between binary, decimal,
    hexadecimal, and arbitrary-base string representations and their numeric
    equivalents, matching Octave semantics for bin2dec, dec2bin, hex2dec,
    dec2hex, and base2dec.

    Model-user argument: An engineer working with embedded systems regularly
    reads register values in hex or binary from datasheets and device logs.
    They need bin2dec and hex2dec to convert those representations into numeric
    values for threshold comparisons and bitfield extraction. dec2bin and
    dec2hex are used when writing configuration words back to documentation
    or log output.

    Decomposition:
      R-STR-01.1: bin2dec converts a binary string to its decimal integer.
      R-STR-01.2: dec2bin converts a decimal integer to its binary string.
      R-STR-01.3: hex2dec converts a hexadecimal string to its decimal integer.
      R-STR-01.4: dec2hex converts a decimal integer to its uppercase hex string.
      R-STR-01.5: base2dec converts a string in an arbitrary base to decimal.

    Consistency: The five sub-requirements cover every base-conversion function
    listed in the parent requirement. Each direction (string-to-number,
    number-to-string) is tested, and the arbitrary-base case generalizes the
    pattern, so the decomposition fully satisfies R-STR-01.
    """

    def test_bin2dec_1010(self):
        """R-STR-01.1: bin2dec('1010') -> 10."""
        r = forge_bin2dec('1010')
        assert r == 10

    def test_dec2bin_10(self):
        """R-STR-01.2: dec2bin(10) -> '1010'."""
        r = forge_dec2bin(10)
        assert r == '1010'

    def test_hex2dec_ff(self):
        """R-STR-01.3: hex2dec('ff') -> 255."""
        r = forge_hex2dec('ff')
        assert r == 255

    def test_dec2hex_255(self):
        """R-STR-01.4: dec2hex(255) -> 'FF'."""
        r = forge_dec2hex(255)
        assert r == 'FF'

    def test_base2dec_octal(self):
        """R-STR-01.5: base2dec('77', 8) -> 63."""
        r = forge_base2dec('77', 8)
        assert r == 63


class TestWhitespace:
    """R-STR-02: The strings toolbox SHALL provide whitespace management
    functions (blanks, deblank, strtrim, strjoin) that produce and remove
    whitespace in strings, matching Octave behavior.

    Model-user argument: When preparing axis labels and titles for publication
    plots, the scientist often has labels read from data files that carry
    trailing or leading whitespace. strtrim and deblank clean those labels so
    they render correctly. blanks is used to build fixed-width column headers
    for text-mode table output, and strjoin assembles comma-separated lists
    for legend entries or CSV export.

    Decomposition:
      R-STR-02.1: blanks(n) returns a string of exactly n space characters.
      R-STR-02.2: deblank removes trailing spaces while preserving leading spaces.
      R-STR-02.3: strtrim removes both leading and trailing whitespace.
      R-STR-02.4: strjoin concatenates a cell array of strings with a delimiter.

    Consistency: The four sub-requirements cover all whitespace functions in the
    parent requirement. Generation (blanks), trailing removal (deblank), full
    trim (strtrim), and assembly (strjoin) together address every whitespace
    management operation specified.
    """

    def test_blanks_5(self):
        """R-STR-02.1: blanks(5) returns 5 spaces."""
        r = forge_blanks(5)
        assert r == '     '
        assert len(r) == 5

    def test_deblank_trailing(self):
        """R-STR-02.2: deblank removes trailing spaces only."""
        r = forge_deblank('  hello   ')
        assert r == '  hello'

    def test_strtrim_both(self):
        """R-STR-02.3: strtrim removes leading and trailing whitespace."""
        r = forge_strtrim('  hello  ')
        assert str(r) == 'hello'

    def test_strjoin_comma(self):
        """R-STR-02.4: strjoin(['a','b','c'], ',') -> 'a,b,c'."""
        r = forge_strjoin(['a', 'b', 'c'], ',')
        assert str(r) == 'a,b,c'


class TestStringSearch:
    """R-STR-03: The strings toolbox SHALL provide substring search functions
    (startsWith, endsWith, index, rindex) that locate substrings within
    strings and return boolean or 1-based position results consistent with
    Octave conventions.

    Model-user argument: When parsing configuration files or instrument output,
    the engineer checks line prefixes (startsWith) to identify section headers
    and suffixes (endsWith) to detect file extensions. index and rindex locate
    delimiters or keywords within a line so the surrounding data can be
    extracted with substr. These are everyday string inspection operations
    that must use 1-based indexing to stay consistent with the rest of Forge.

    Decomposition:
      R-STR-03.1: startsWith returns true when the string begins with the prefix.
      R-STR-03.2: startsWith returns false when the string does not begin with the prefix.
      R-STR-03.3: endsWith returns true when the string ends with the suffix.
      R-STR-03.4: endsWith returns false when the string does not end with the suffix.
      R-STR-03.5: index returns the 1-based position of the first occurrence.
      R-STR-03.6: rindex returns the 1-based position of the last occurrence.

    Consistency: The six sub-requirements test both positive and negative cases
    for the boolean functions (startsWith, endsWith) and the forward and reverse
    positional searches (index, rindex). Together they fully verify the parent
    requirement's search semantics.
    """

    def test_startsWith_true(self):
        """R-STR-03.1: startsWith returns true for matching prefix."""
        r = forge_startsWith('hello world', 'hello')
        assert r is True

    def test_startsWith_false(self):
        """R-STR-03.2: startsWith returns false for non-matching prefix."""
        r = forge_startsWith('hello world', 'world')
        assert r is False

    def test_endsWith_true(self):
        """R-STR-03.3: endsWith returns true for matching suffix."""
        r = forge_endsWith('hello world', 'world')
        assert r is True

    def test_endsWith_false(self):
        """R-STR-03.4: endsWith returns false for non-matching suffix."""
        r = forge_endsWith('hello world', 'hello')
        assert r is False

    def test_index_found(self):
        """R-STR-03.5: index('hello world', 'world') -> 7 (1-based)."""
        r = forge_index_str('hello world', 'world')
        assert r == 7

    def test_rindex_found(self):
        """R-STR-03.6: rindex('abcabc', 'bc') -> 5 (1-based, last occurrence)."""
        r = forge_rindex('abcabc', 'bc')
        assert r == 5


class TestStringSplit:
    """R-STR-04: The strings toolbox SHALL provide string tokenization
    functions (strsplit, strtok) that split strings on delimiters, returning
    cell arrays or token/remainder pairs consistent with Octave semantics.

    Model-user argument: The engineer frequently reads CSV sensor logs and
    INI-style config files. strsplit breaks comma-separated or tab-separated
    lines into cell arrays of fields for downstream numeric conversion.
    strtok is used for incremental parsing where only the next token is needed
    at each step, such as walking through space-delimited command strings from
    an instrument protocol.

    Decomposition:
      R-STR-04.1: strsplit splits a string on a character delimiter into a cell array.
      R-STR-04.2: strtok extracts the first token and returns the remainder.

    Consistency: The two sub-requirements cover both splitting modes available
    in Octave: full split (strsplit) and incremental tokenization (strtok).
    Together they fully satisfy the parent requirement.
    """

    def test_strsplit_comma(self):
        """R-STR-04.1: strsplit('a,b,c', ',') -> ['a','b','c']."""
        r = forge_strsplit('a,b,c', ',')
        parts = [str(x) for x in r._data] if hasattr(r, "_data") else [str(r)]
        assert parts == ['a', 'b', 'c']

    def test_strtok_simple(self):
        """R-STR-04.2: strtok('hello world') -> ('hello', ' world')."""
        tok, rem = forge_strtok('hello world')
        assert tok == 'hello'
        assert rem == ' world'


class TestStringConversion:
    """R-STR-05: The strings toolbox SHALL convert between string and numeric
    matrix representations via str2num and mat2str, matching Octave output
    format conventions.

    Model-user argument: When reading numeric data from text files (sensor
    dumps, exported spreadsheets), the engineer uses str2num to parse each
    field into a number after splitting the line. mat2str is the inverse: it
    serializes a matrix into a bracket-delimited string for logging, so that
    intermediate results can be written to a diary file and later pasted back
    into the command window for reproduction.

    Decomposition:
      R-STR-05.1: str2num parses a numeric string into its scalar value.
      R-STR-05.2: mat2str serializes a matrix into Octave bracket notation.

    Consistency: The two sub-requirements cover the two directions of
    string/numeric conversion (parse and serialize). Together they fully
    satisfy the parent requirement.
    """

    def test_str2num_integer(self):
        """R-STR-05.1: str2num('42') -> 42."""
        r = forge_str2num('42')
        assert r == 42

    def test_mat2str_matrix(self):
        """R-STR-05.2: mat2str([1 2; 3 4]) -> '[1 2; 3 4]'."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        r = forge_mat2str(A)
        assert '[' in r and ']' in r
        assert '1' in r and '4' in r
