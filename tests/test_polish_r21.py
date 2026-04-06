# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Polish Round 21: date/time functions, string operations, textscan.

V&V Traceability (backfill):
    R-POL21-01 .. R-POL21-03 (parent requirements)
    R-POL21-01-nn .. R-POL21-03-nn (unit sub-requirements)

SRS trace: SRS-FUNC-001, SRS-VAL-001, SRS-COMPAT-001

Validates:
  - now, datestr, datenum, datevec, clock, tic/toc
  - upper, lower, strsplit, strjoin, strrep, num2str, mat2str, sprintf, fliplr
  - textscan basic parsing
  - datenum epoch correctness (MATLAB-compatible)
  - auto-call of zero-arg functions as arguments (e.g., datestr(now))
"""

import math
import time
import numpy as np
import pytest

from forge.engine.session import ForgeSession


@pytest.fixture
def S():
    return ForgeSession()


# ── Date / Time ──────────────────────────────────────────────────

class TestDateTime:
    """R-POL21-01: Forge SHALL provide date/time functions (now, datenum,
    datevec, datestr, clock, tic/toc) that return MATLAB-compatible datenum
    epoch values, support round-trip conversions, and auto-call zero-argument
    functions when passed as arguments.

    Model-user argument: An engineer migrating time-series analysis scripts from
    Octave uses datenum for timestamp arithmetic, datestr for human-readable
    labels, and tic/toc for performance profiling. Incorrect epoch values would
    silently shift all time-series alignment, and broken auto-call of now()
    would require manual parentheses insertion across hundreds of scripts.

    Decomposition:
        R-POL21-01-01: now returns a reasonable datenum
        R-POL21-01-02: datenum(2024,1,15) = 739266 (MATLAB reference)
        R-POL21-01-03: datevec(datenum(2024,1,15)) round-trips correctly
        R-POL21-01-04: datestr from datenum produces readable date string
        R-POL21-01-05: datestr(now) auto-calls now()
        R-POL21-01-06: datevec(now) auto-calls now()
        R-POL21-01-07: clock returns 6-element vector [Y M D H M S]
        R-POL21-01-08: tic/toc measures elapsed time
        R-POL21-01-09: datenum/datevec consistency across dates

    Consistency: Sub-requirements cover epoch correctness (01-02), round-trip
    (03), formatting (04), auto-call behavior (05-06), clock introspection (07),
    timing (08), and cross-date consistency (09). Together they validate the
    full date/time API.
    """

    def test_now_returns_number(self, S):
        """R-POL21-01-01: now returns a datenum greater than 700000."""
        val = S.eval("now")
        assert isinstance(float(str(val).strip()), float)
        assert float(str(val).strip()) > 700000  # reasonable datenum

    def test_datenum_2024_01_15(self, S):
        """R-POL21-01-02: datenum(2024,1,15) = 739266 (MATLAB reference)."""
        val = S.eval("datenum(2024, 1, 15)")
        assert int(str(val).strip()) == 739266

    def test_datevec_roundtrip(self, S):
        """R-POL21-01-03: datevec(datenum(2024,1,15)) returns [2024 1 15 0 0 0]."""
        raw = S.eval("datevec(datenum(2024, 1, 15))")
        # Parse the vector output
        nums = [float(x) for x in str(raw).split()]
        assert nums[:4] == [2024, 1, 15, 0]

    def test_datestr_from_datenum(self, S):
        """R-POL21-01-04: datestr(datenum(2024,1,15)) contains '15', 'Jan', '2024'."""
        result = str(S.eval("datestr(datenum(2024, 1, 15))")).strip()
        assert "15" in result and "Jan" in result and "2024" in result

    def test_datestr_now_autocall(self, S):
        """R-POL21-01-05: datestr(now) auto-calls now() and returns current year."""
        result = str(S.eval("datestr(now)")).strip()
        assert "2026" in result or "202" in result  # current year

    def test_datevec_now_autocall(self, S):
        """R-POL21-01-06: datevec(now) auto-calls now() and returns current year."""
        raw = S.eval("datevec(now)")
        nums = [float(x) for x in str(raw).split()]
        assert nums[0] >= 2026  # current year

    def test_clock_returns_6_elements(self, S):
        """R-POL21-01-07: clock returns [year month day hour min sec]."""
        raw = S.eval("clock")
        nums = [float(x) for x in str(raw).split()]
        assert len(nums) == 6
        assert nums[0] >= 2026  # year
        assert 1 <= nums[1] <= 12  # month

    def test_tic_toc_timing(self, S):
        """R-POL21-01-08: tic/toc measures elapsed time within reasonable bounds."""
        S.eval("tic;")
        time.sleep(0.05)
        raw = S.eval("t = toc")
        val = float(str(raw).strip())
        assert 0.01 < val < 2.0  # reasonable elapsed time

    def test_datenum_epoch_consistency(self, S):
        """R-POL21-01-09: datenum and datevec are consistent for 2000-06-15."""
        S.eval("dn = datenum(2000, 6, 15);")
        raw = S.eval("datevec(dn)")
        nums = [float(x) for x in str(raw).split()]
        assert nums[0] == 2000
        assert nums[1] == 6
        assert nums[2] == 15


# ── String Operations ────────────────────────────────────────────

class TestStringOps:
    """R-POL21-02: Forge SHALL provide string manipulation functions (upper,
    lower, strsplit, strjoin, strrep, num2str, mat2str, sprintf, fliplr on
    strings) that produce results identical to MATLAB/Octave behavior.

    Model-user argument: A scientist porting data-processing scripts uses string
    functions to parse filenames, format output labels, assemble CSV headers,
    and generate reports. Differences in capitalization, splitting, joining, or
    formatting would cause silent data corruption in file I/O and display code.

    Decomposition:
        R-POL21-02-01: upper('hello') = 'HELLO'
        R-POL21-02-02: lower('HELLO') = 'hello'
        R-POL21-02-03: strsplit('a,b,c', ',') yields {a, b, c}
        R-POL21-02-04: strjoin({'a','b','c'}, '-') = 'a-b-c'
        R-POL21-02-05: strrep('hello world', 'world', 'earth') = 'hello earth'
        R-POL21-02-06: num2str(pi) starts with '3.14'
        R-POL21-02-07: mat2str([1 2; 3 4]) = '[1 2; 3 4]'
        R-POL21-02-08: sprintf('%06.2f', 3.14) = '003.14'
        R-POL21-02-09: fliplr('hello') = 'olleh'
        R-POL21-02-10: num2str(42) = '42'

    Consistency: Sub-requirements cover case conversion (01-02), splitting and
    joining (03-04), replacement (05), numeric-to-string formatting (06-08, 10),
    and string reversal (09). Together they validate the full string API.
    """

    def test_upper(self, S):
        """R-POL21-02-01: upper('hello') = 'HELLO'."""
        assert str(S.eval('upper("hello")')).strip() == "HELLO"

    def test_lower(self, S):
        """R-POL21-02-02: lower('HELLO') = 'hello'."""
        assert str(S.eval('lower("HELLO")')).strip() == "hello"

    def test_strsplit(self, S):
        """R-POL21-02-03: strsplit('a,b,c', ',') contains a, b, c."""
        raw = str(S.eval('strsplit("a,b,c", ",")'))
        assert "a" in raw and "b" in raw and "c" in raw

    def test_strjoin(self, S):
        """R-POL21-02-04: strjoin({'a','b','c'}, '-') = 'a-b-c'."""
        result = str(S.eval('strjoin({"a","b","c"}, "-")')).strip()
        assert result == "a-b-c"

    def test_strrep(self, S):
        """R-POL21-02-05: strrep replaces substring correctly."""
        result = str(S.eval('strrep("hello world", "world", "earth")')).strip()
        assert result == "hello earth"

    def test_num2str_pi(self, S):
        """R-POL21-02-06: num2str(pi) starts with '3.14'."""
        result = str(S.eval("num2str(pi)")).strip()
        assert result.startswith("3.14")

    def test_mat2str(self, S):
        """R-POL21-02-07: mat2str([1 2; 3 4]) = '[1 2; 3 4]'."""
        result = str(S.eval("mat2str([1 2; 3 4])")).strip()
        assert result == "[1 2; 3 4]"

    def test_sprintf_format(self, S):
        """R-POL21-02-08: sprintf('%06.2f', 3.14) = '003.14'."""
        result = str(S.eval('sprintf("%06.2f", 3.14)')).strip()
        assert result == "003.14"

    def test_fliplr_string(self, S):
        """R-POL21-02-09: fliplr('hello') = 'olleh'."""
        result = str(S.eval('fliplr("hello")')).strip()
        assert result == "olleh"

    def test_num2str_integer(self, S):
        """R-POL21-02-10: num2str(42) = '42'."""
        result = str(S.eval("num2str(42)")).strip()
        assert result == "42"


# ── textscan ─────────────────────────────────────────────────────

class TestTextscan:
    """R-POL21-03: textscan SHALL parse formatted string data into numeric and
    string tokens matching MATLAB/Octave behavior.

    Model-user argument: An engineer importing instrument data exported as
    space-delimited text relies on textscan for column parsing. If textscan
    fails to extract numeric tokens correctly, the entire data import pipeline
    breaks and requires manual reformatting.

    Decomposition:
        R-POL21-03-01: textscan parses '1 2 3' with '%f %f %f'

    Consistency: This single sub-requirement verifies the core parsing path.
    Additional format specifiers (%d, %s, delimiters) are covered in
    dedicated textscan test files.
    """

    def test_textscan_basic(self, S):
        """R-POL21-03-01: textscan parses space-delimited floats."""
        S.eval('s = "1 2 3";')
        raw = str(S.eval('textscan(s, "%f %f %f")'))
        assert "1" in raw and "2" in raw and "3" in raw
