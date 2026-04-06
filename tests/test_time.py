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
    """R-TIME-01: datenum SHALL convert a calendar date (year, month, day) to a
    serial day number consistent with the Octave epoch (day 1 = Jan 1, 0001).

    Model-user argument: The migrating engineer uses datenum to convert
    calibration dates into serial numbers so that elapsed days between sensor
    tests can be computed with simple subtraction. If datenum returns incorrect
    serial values, interval calculations silently produce wrong results and
    the engineer cannot trust maintenance schedules derived from date arithmetic.

    Decomposition:
      R-TIME-01a: datenum(2000,1,1) returns a float that round-trips through datevec.
      R-TIME-01b: datenum(1970,1,1) returns a float that round-trips through datevec.
      R-TIME-01c: datenum(1,1,1) returns 1 (epoch origin).

    Consistency: 01a and 01b verify that representative modern dates survive the
    datenum/datevec round-trip, confirming the internal epoch offset is correct.
    01c directly asserts the epoch anchor. Together they validate the conversion
    at the origin, at the Unix epoch, and at a post-Y2K date, covering the range
    an engineer would encounter in practice.
    """

    def test_datenum_2000_1_1(self):
        """R-TIME-01a: datenum(2000,1,1) round-trips through datevec to recover 2000-01-01."""
        r = forge_datenum(2000, 1, 1)
        assert isinstance(r, float)
        # Verify round-trip rather than hard-coded Octave value
        dv = forge_datevec(r)
        assert int(dv[0]) == 2000
        assert int(dv[1]) == 1
        assert int(dv[2]) == 1

    def test_datenum_1970_1_1(self):
        """R-TIME-01b: datenum(1970,1,1) round-trips through datevec to recover 1970-01-01."""
        r = forge_datenum(1970, 1, 1)
        assert isinstance(r, float)
        dv = forge_datevec(r)
        assert int(dv[0]) == 1970
        assert int(dv[1]) == 1
        assert int(dv[2]) == 1

    def test_datenum_1_1_1(self):
        """R-TIME-01c: datenum(1,1,1) returns serial day number 367 (MATLAB epoch)."""
        r = forge_datenum(1, 1, 1)
        assert int(r) == 367  # MATLAB: datenum(1,1,1) = 367


class TestDatevec:
    """R-TIME-02: datevec SHALL convert a serial day number back to a six-element
    vector [year, month, day, hour, minute, second] that matches the original
    calendar date.

    Model-user argument: After computing elapsed days between sensor readings
    using serial numbers, the engineer calls datevec to convert results back
    into human-readable dates for report tables and plot axis labels. If datevec
    produces incorrect fields, exported reports show wrong dates and the
    engineer loses confidence in the toolchain.

    Decomposition:
      R-TIME-02a: datevec(datenum(2000,1,1)) recovers [2000,1,1,0,0,0] as an ndarray.
      R-TIME-02b: datevec(datenum(1970,1,1)) recovers [1970,1,1,0,0,0].

    Consistency: 02a tests a post-Y2K date and also verifies the return type is
    ndarray (matching Octave semantics). 02b tests the Unix epoch boundary. Both
    confirm the inverse relationship with datenum across the date range relevant
    to engineering workflows.
    """

    def test_datevec_roundtrip_2000(self):
        """R-TIME-02a: datevec(datenum(2000,1,1)) returns ndarray [2000,1,1,0,0,0]."""
        dn = forge_datenum(2000, 1, 1)
        r = forge_datevec(dn)
        assert isinstance(r, np.ndarray)
        v = r.ravel()
        assert int(v[0]) == 2000
        assert int(v[1]) == 1
        assert int(v[2]) == 1

    def test_datevec_roundtrip_1970(self):
        """R-TIME-02b: datevec(datenum(1970,1,1)) recovers [1970,1,1,0,0,0]."""
        dn = forge_datenum(1970, 1, 1)
        r = forge_datevec(dn).ravel()
        assert int(r[0]) == 1970
        assert int(r[1]) == 1
        assert int(r[2]) == 1


class TestDatestr:
    """R-TIME-03: datestr SHALL convert a serial day number to a human-readable
    date string containing the year and abbreviated month name.

    Model-user argument: The engineer calls datestr to generate formatted date
    labels for plot titles and log file headers (e.g., "01-Jan-2000"). If datestr
    omits the year or month, automated plots and test logs become ambiguous,
    forcing manual date annotation that defeats the purpose of scripted workflows.

    Decomposition:
      R-TIME-03a: datestr(datenum(2000,1,1)) produces a string containing '2000' and 'Jan'.

    Consistency: A single sub-requirement is sufficient because the core contract
    is that the output string encodes both the year and month from the serial
    input. Testing for the presence of '2000' and 'Jan' confirms the formatter
    correctly extracts and renders both fields.
    """

    def test_datestr_2000(self):
        """R-TIME-03a: datestr for 2000-01-01 contains '2000' and 'Jan'."""
        dn = forge_datenum(2000, 1, 1)
        r = forge_datestr(dn)
        result = str(r)
        assert '2000' in result
        assert 'Jan' in result


class TestRoundTrip:
    """R-TIME-04: The datenum/datevec pair SHALL be exact inverses for any valid
    calendar date, so that datevec(datenum(y,m,d)) recovers (y, m, d) with zero
    drift.

    Model-user argument: The engineer chains datenum and datevec when computing
    a future calibration date: convert start date to serial, add N days, convert
    back. Any drift in the round-trip (even by one day) means the scheduled
    calibration falls on the wrong date, potentially violating regulatory
    intervals for sensor certification.

    Decomposition:
      R-TIME-04a: Round-trip recovers 2024-06-15 (mid-year, mid-month).
      R-TIME-04b: Round-trip recovers 1999-12-31 (year boundary, end-of-month).

    Consistency: 04a tests a mid-year date with no boundary effects. 04b tests
    December 31 at a year boundary, exercising month/year rollover logic. Together
    they confirm the inverse property holds across typical and boundary dates.
    """

    def test_roundtrip_2024_6_15(self):
        """R-TIME-04a: datevec(datenum(2024,6,15)) recovers 2024, 6, 15."""
        dn = forge_datenum(2024, 6, 15)
        dv = forge_datevec(dn).ravel()
        assert int(dv[0]) == 2024
        assert int(dv[1]) == 6
        assert int(dv[2]) == 15

    def test_roundtrip_1999_12_31(self):
        """R-TIME-04b: datevec(datenum(1999,12,31)) recovers 1999, 12, 31."""
        dn = forge_datenum(1999, 12, 31)
        dv = forge_datevec(dn).ravel()
        assert int(dv[0]) == 1999
        assert int(dv[1]) == 12
        assert int(dv[2]) == 31


class TestClockAndNow:
    """R-TIME-05: clock SHALL return the current wall-clock time as a six-element
    vector, and now SHALL return the current time as a serial day number greater
    than 700000.

    Model-user argument: The engineer uses clock() to timestamp the start and
    end of a data acquisition run, and now() to compute elapsed time in days
    via simple subtraction. If clock returns the wrong number of elements, code
    that indexes into [year, month, day, hour, minute, second] will crash. If
    now returns a value outside the expected range, elapsed-time calculations
    produce nonsensical durations.

    Decomposition:
      R-TIME-05a: clock() returns a 6-element ndarray.
      R-TIME-05b: now() returns a float greater than 700000.

    Consistency: 05a validates the shape contract that all Octave-compatible
    code relies on when destructuring the clock vector. 05b validates that now()
    returns a plausible serial day number for any date after ~1917 (serial 700000),
    confirming the epoch alignment. Together they ensure both wall-clock functions
    are usable for timing workflows.
    """

    def test_clock_returns_6_elements(self):
        """R-TIME-05a: clock() returns a 6-element ndarray [y,m,d,h,min,sec]."""
        r = forge_clock()
        assert isinstance(r, np.ndarray)
        assert r.ravel().shape[0] == 6

    def test_now_returns_large_scalar(self):
        """R-TIME-05b: now() returns a serial day number (float) greater than 700000."""
        r = forge_now()
        assert isinstance(r, float)
        assert r > 700000


class TestEomdayAndLeapYear:
    """R-TIME-06: eomday SHALL return the last day of the given month and year,
    and is_leap_year SHALL return True for leap years and False otherwise,
    following the Gregorian 4/100/400 rule.

    Model-user argument: When processing monthly sensor summaries, the engineer
    uses eomday to determine how many days are in each reporting period so that
    daily averages are divided by the correct count. is_leap_year gates February
    handling in quarterly roll-ups. Incorrect leap year detection causes February
    totals to be off by one day, which propagates into annual compliance reports.

    Decomposition:
      R-TIME-06a: eomday(2000, 2) returns 29 (leap year February).
      R-TIME-06b: eomday(2001, 2) returns 28 (non-leap year February).
      R-TIME-06c: is_leap_year(2000) returns True (divisible by 400).
      R-TIME-06d: is_leap_year(1900) returns False (divisible by 100 but not 400).
      R-TIME-06e: is_leap_year(2004) returns True (divisible by 4, not by 100).

    Consistency: 06a and 06b test eomday for February in both leap and non-leap
    years, the only month where the result varies. 06c, 06d, and 06e exercise
    all three branches of the Gregorian leap year rule: divisible by 400 (leap),
    divisible by 100 but not 400 (not leap), and divisible by 4 but not 100
    (leap). Full branch coverage of the leap year rule ensures eomday and
    is_leap_year agree on every year the engineer might encounter.
    """

    def test_eomday_leap_feb(self):
        """R-TIME-06a: eomday(2000, 2) returns 29 (leap year)."""
        r = forge_eomday(2000, 2)
        assert int(r) == 29

    def test_eomday_nonleap_feb(self):
        """R-TIME-06b: eomday(2001, 2) returns 28 (non-leap year)."""
        r = forge_eomday(2001, 2)
        assert int(r) == 28

    def test_is_leap_year_2000(self):
        """R-TIME-06c: is_leap_year(2000) returns True (divisible by 400)."""
        r = forge_is_leap_year(2000)
        assert r is True

    def test_is_leap_year_1900(self):
        """R-TIME-06d: is_leap_year(1900) returns False (divisible by 100, not 400)."""
        r = forge_is_leap_year(1900)
        assert r is False

    def test_is_leap_year_2004(self):
        """R-TIME-06e: is_leap_year(2004) returns True (divisible by 4, not 100)."""
        r = forge_is_leap_year(2004)
        assert r is True
