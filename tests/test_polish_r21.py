# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Polish Round 21 — date/time functions, string operations, textscan.

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
    def test_now_returns_number(self, S):
        val = S.eval("now")
        assert isinstance(float(str(val).strip()), float)
        assert float(str(val).strip()) > 700000  # reasonable datenum

    def test_datenum_2024_01_15(self, S):
        """MATLAB reference: datenum(2024,1,15) = 739266."""
        val = S.eval("datenum(2024, 1, 15)")
        assert int(str(val).strip()) == 739266

    def test_datevec_roundtrip(self, S):
        """datevec(datenum(2024,1,15)) should return [2024 1 15 0 0 0]."""
        raw = S.eval("datevec(datenum(2024, 1, 15))")
        # Parse the vector output
        nums = [float(x) for x in str(raw).split()]
        assert nums[:4] == [2024, 1, 15, 0]

    def test_datestr_from_datenum(self, S):
        result = str(S.eval("datestr(datenum(2024, 1, 15))")).strip()
        assert "15" in result and "Jan" in result and "2024" in result

    def test_datestr_now_autocall(self, S):
        """datestr(now) should auto-call now() and return a date string."""
        result = str(S.eval("datestr(now)")).strip()
        assert "2026" in result or "202" in result  # current year

    def test_datevec_now_autocall(self, S):
        """datevec(now) should auto-call now()."""
        raw = S.eval("datevec(now)")
        nums = [float(x) for x in str(raw).split()]
        assert nums[0] >= 2026  # current year

    def test_clock_returns_6_elements(self, S):
        raw = S.eval("clock")
        nums = [float(x) for x in str(raw).split()]
        assert len(nums) == 6
        assert nums[0] >= 2026  # year
        assert 1 <= nums[1] <= 12  # month

    def test_tic_toc_timing(self, S):
        S.eval("tic;")
        time.sleep(0.05)
        raw = S.eval("t = toc")
        val = float(str(raw).strip())
        assert 0.01 < val < 2.0  # reasonable elapsed time

    def test_datenum_epoch_consistency(self, S):
        """datenum and datevec should be consistent across dates."""
        S.eval("dn = datenum(2000, 6, 15);")
        raw = S.eval("datevec(dn)")
        nums = [float(x) for x in str(raw).split()]
        assert nums[0] == 2000
        assert nums[1] == 6
        assert nums[2] == 15


# ── String Operations ────────────────────────────────────────────

class TestStringOps:
    def test_upper(self, S):
        assert str(S.eval('upper("hello")')).strip() == "HELLO"

    def test_lower(self, S):
        assert str(S.eval('lower("HELLO")')).strip() == "hello"

    def test_strsplit(self, S):
        raw = str(S.eval('strsplit("a,b,c", ",")'))
        assert "a" in raw and "b" in raw and "c" in raw

    def test_strjoin(self, S):
        result = str(S.eval('strjoin({"a","b","c"}, "-")')).strip()
        assert result == "a-b-c"

    def test_strrep(self, S):
        result = str(S.eval('strrep("hello world", "world", "earth")')).strip()
        assert result == "hello earth"

    def test_num2str_pi(self, S):
        result = str(S.eval("num2str(pi)")).strip()
        assert result.startswith("3.14")

    def test_mat2str(self, S):
        result = str(S.eval("mat2str([1 2; 3 4])")).strip()
        assert result == "[1 2; 3 4]"

    def test_sprintf_format(self, S):
        result = str(S.eval('sprintf("%06.2f", 3.14)')).strip()
        assert result == "003.14"

    def test_fliplr_string(self, S):
        result = str(S.eval('fliplr("hello")')).strip()
        assert result == "olleh"

    def test_num2str_integer(self, S):
        result = str(S.eval("num2str(42)")).strip()
        assert result == "42"


# ── textscan ─────────────────────────────────────────────────────

class TestTextscan:
    def test_textscan_basic(self, S):
        S.eval('s = "1 2 3";')
        raw = str(S.eval('textscan(s, "%f %f %f")'))
        assert "1" in raw and "2" in raw and "3" in raw
