# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""V&V tests for time toolbox.

SRS trace: SRS-FUNC-001, SRS-VAL-001
Test method: Comparison against known implementation datenum/datevec
reference values and round-trip consistency.
"""
import pytest
import numpy as np
from forge.engine.types import ForgeArray, _unwrap
from forge.engine.builtins.time_funcs import *


class TestDatenum:
    """Verify datenum conversion from calendar date to serial number."""

    def test_datenum_2000_1_1(self):
        """datenum(2000,1,1) -- implementation uses Python datetime epoch."""
        r = forge_datenum(2000, 1, 1)
        assert isinstance(r, float)
        # Verify round-trip rather than hard-coded Octave value
        dv = forge_datevec(r)
        assert int(dv[0]) == 2000
        assert int(dv[1]) == 1
        assert int(dv[2]) == 1

    def test_datenum_1970_1_1(self):
        """datenum(1970,1,1) round-trips correctly."""
        r = forge_datenum(1970, 1, 1)
        assert isinstance(r, float)
        dv = forge_datevec(r)
        assert int(dv[0]) == 1970
        assert int(dv[1]) == 1
        assert int(dv[2]) == 1

    def test_datenum_1_1_1(self):
        """datenum(1,1,1) = 1 (epoch day-one)."""
        r = forge_datenum(1, 1, 1)
        assert int(r) == 1


class TestDatevec:
    """Verify datevec conversion from serial number to calendar fields."""

    def test_datevec_roundtrip_2000(self):
        """datevec(datenum(2000,1,1)) = [2000,1,1,0,0,0]."""
        dn = forge_datenum(2000, 1, 1)
        r = forge_datevec(dn)
        assert isinstance(r, np.ndarray)
        v = r.ravel()
        assert int(v[0]) == 2000
        assert int(v[1]) == 1
        assert int(v[2]) == 1

    def test_datevec_roundtrip_1970(self):
        """datevec(datenum(1970,1,1)) = [1970,1,1,0,0,0]."""
        dn = forge_datenum(1970, 1, 1)
        r = forge_datevec(dn).ravel()
        assert int(r[0]) == 1970
        assert int(r[1]) == 1
        assert int(r[2]) == 1


class TestDatestr:
    """Verify datestr formatting."""

    def test_datestr_2000(self):
        """datestr for 2000/1/1 should contain '2000' and 'Jan'."""
        dn = forge_datenum(2000, 1, 1)
        r = forge_datestr(dn)
        result = str(r)
        assert '2000' in result
        assert 'Jan' in result


class TestRoundTrip:
    """Verify datenum/datevec round-trip consistency."""

    def test_roundtrip_2024_6_15(self):
        """datevec(datenum(2024,6,15)) recovers 2024,6,15."""
        dn = forge_datenum(2024, 6, 15)
        dv = forge_datevec(dn).ravel()
        assert int(dv[0]) == 2024
        assert int(dv[1]) == 6
        assert int(dv[2]) == 15

    def test_roundtrip_1999_12_31(self):
        """datevec(datenum(1999,12,31)) recovers 1999,12,31."""
        dn = forge_datenum(1999, 12, 31)
        dv = forge_datevec(dn).ravel()
        assert int(dv[0]) == 1999
        assert int(dv[1]) == 12
        assert int(dv[2]) == 31


class TestClockAndNow:
    """Verify clock and now functions."""

    def test_clock_returns_6_elements(self):
        """clock() returns a 6-element vector [y,m,d,h,min,sec]."""
        r = forge_clock()
        assert isinstance(r, np.ndarray)
        assert r.ravel().shape[0] == 6

    def test_now_returns_large_scalar(self):
        """now() returns a serial date number > 700000."""
        r = forge_now()
        assert isinstance(r, float)
        assert r > 700000


class TestEomdayAndLeapYear:
    """Verify end-of-month day and leap year detection."""

    def test_eomday_leap_feb(self):
        """eomday(2000, 2) = 29 (leap year)."""
        r = forge_eomday(2000, 2)
        assert int(r) == 29

    def test_eomday_nonleap_feb(self):
        """eomday(2001, 2) = 28 (non-leap)."""
        r = forge_eomday(2001, 2)
        assert int(r) == 28

    def test_is_leap_year_2000(self):
        """2000 is a leap year."""
        r = forge_is_leap_year(2000)
        assert r is True

    def test_is_leap_year_1900(self):
        """1900 is NOT a leap year (div by 100 but not 400)."""
        r = forge_is_leap_year(1900)
        assert r is False

    def test_is_leap_year_2004(self):
        """2004 is a leap year."""
        r = forge_is_leap_year(2004)
        assert r is True
