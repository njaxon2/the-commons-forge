"""Time Toolbox for Forge.

Provides 14 date/time functions compatible with Octave/MATLAB
time operations. Uses MATLAB datenum convention (days since Jan 0, year 0).

Backend: Python datetime module.
"""

from __future__ import annotations

import calendar as _calendar
import datetime as _dt
import time as _time

import numpy as np

from forge.engine.types import ForgeArray, _unwrap


# ── Helpers ──────────────────────────────────────────────────────

def _wrap(value):
    """Wrap a numpy array as a ForgeArray."""
    if isinstance(value, np.ndarray):
        return ForgeArray(value)
    return value


def _scalar(value):
    """Extract scalar from 0-d or single-element array."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return value.item()
    return value


# MATLAB datenum epoch: day 1 = Jan 1, year 0001.
# The MATLAB convention counts Jan 1, 0000 as day 0 (proleptic Gregorian),
# but effectively datenum(1,1,1) = 1.  Python's datetime starts at year 1.
_DATENUM_EPOCH = _dt.datetime(1, 1, 1)  # represents datenum = 1


def _datetime_to_datenum(dt_obj):
    """Convert a Python datetime to MATLAB datenum."""
    delta = dt_obj - _DATENUM_EPOCH
    return delta.total_seconds() / 86400.0 + 1.0


def _datenum_to_datetime(dn):
    """Convert a MATLAB datenum to Python datetime."""
    return _DATENUM_EPOCH + _dt.timedelta(days=float(dn) - 1.0)


# ── Toolbox function registry ───────────────────────────────────
TIME_REGISTRY: dict[str, callable] = {}


def _tb(name: str | None = None):
    """Local decorator to register a toolbox function."""
    def decorator(func):
        fn_name = name or func.__name__
        TIME_REGISTRY[fn_name] = func
        return func
    return decorator


# =====================================================================
# Date / Time Query
# =====================================================================

@_tb("clock")
def forge_clock():
    """Return current date and time as [year month day hour minute second].

    clock() -> [2024 3 15 10 30 45.123]
    """
    now = _dt.datetime.now()
    return np.array([
        now.year, now.month, now.day,
        now.hour, now.minute,
        now.second + now.microsecond / 1e6
    ], dtype=np.float64)


@_tb("now")
def forge_now():
    """Return current date/time as a datenum serial date number.

    now() -> 739321.456  (example)
    """
    return _datetime_to_datenum(_dt.datetime.now())


@_tb("date")
def forge_date():
    """Return current date as a string in 'DD-Mon-YYYY' format.

    date() -> '15-Mar-2024'
    """
    return _dt.datetime.now().strftime("%d-%b-%Y")


@_tb("ctime")
def forge_ctime(t=None):
    """Convert Unix timestamp to date string, or return current time string.

    ctime()          -> 'Fri Mar 15 10:30:45 2024'
    ctime(1710500000) -> time string for that epoch
    """
    if t is None:
        return _time.ctime()
    return _time.ctime(float(t))


@_tb("asctime")
def forge_asctime(time_struct=None):
    """Return date string from time structure or current time.

    asctime() -> 'Fri Mar 15 10:30:45 2024'
    """
    if time_struct is None:
        return _time.asctime()
    if isinstance(time_struct, (list, np.ndarray)):
        # Interpret as clock vector [Y M D H Mi S]
        v = np.asarray(time_struct, dtype=np.float64).ravel()
        dt = _dt.datetime(int(v[0]), int(v[1]), int(v[2]),
                          int(v[3]), int(v[4]), int(v[5]))
        return dt.strftime("%a %b %d %H:%M:%S %Y")
    return _time.asctime(time_struct)


# =====================================================================
# Date Number Conversion
# =====================================================================

@_tb("datenum")
def forge_datenum(*args):
    """Convert date to serial date number (MATLAB compatible).

    datenum(Y, M, D)           -> serial date number
    datenum(Y, M, D, H, Mi, S) -> serial date number with time
    datenum('15-Mar-2024')     -> serial date number from string
    datenum(datevec)           -> serial date number from date vector

    MATLAB datenum: day 1 = January 1, year 0001.
    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, str):
            # Parse date string
            for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y",
                        "%d-%b-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = _dt.datetime.strptime(str(arg), fmt)
                    return _datetime_to_datenum(dt)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse date string: '{arg}'")
        elif isinstance(arg, (list, np.ndarray)):
            # Date vector [Y M D] or [Y M D H Mi S]
            v = np.asarray(arg, dtype=np.float64).ravel()
            if len(v) >= 6:
                dt = _dt.datetime(int(v[0]), int(v[1]), int(v[2]),
                                  int(v[3]), int(v[4]), int(v[5]))
            elif len(v) >= 3:
                dt = _dt.datetime(int(v[0]), int(v[1]), int(v[2]))
            else:
                raise ValueError("Date vector must have at least 3 elements")
            return _datetime_to_datenum(dt)
        elif isinstance(arg, _dt.datetime):
            return _datetime_to_datenum(arg)
        else:
            # Assume already a datenum
            return float(arg)
    elif len(args) == 3:
        y, m, d = int(args[0]), int(args[1]), int(args[2])
        dt = _dt.datetime(y, m, d)
        return _datetime_to_datenum(dt)
    elif len(args) == 6:
        y, m, d = int(args[0]), int(args[1]), int(args[2])
        h, mi, s = int(args[3]), int(args[4]), int(args[5])
        dt = _dt.datetime(y, m, d, h, mi, s)
        return _datetime_to_datenum(dt)
    else:
        raise ValueError("datenum requires 1, 3, or 6 arguments")


@_tb("datestr")
def forge_datestr(datenum_val, fmt=None):
    """Convert serial date number to date string.

    datestr(739321)         -> '15-Mar-2024'
    datestr(739321, 'yyyy-mm-dd') -> '2024-03-15'

    Common format codes (MATLAB-style):
      0  -> 'dd-mmm-yyyy HH:MM:SS'
      1  -> 'dd-mmm-yyyy'
      2  -> 'mm/dd/yy'
     23  -> 'mm/dd/yyyy'
     31  -> 'yyyy-mm-dd'
    """
    dt = _datenum_to_datetime(datenum_val)

    if fmt is None:
        return dt.strftime("%d-%b-%Y")

    # Handle numeric format codes
    if isinstance(fmt, (int, float)):
        fmt = int(fmt)
        fmt_map = {
            0: "%d-%b-%Y %H:%M:%S",
            1: "%d-%b-%Y",
            2: "%m/%d/%y",
            23: "%m/%d/%Y",
            29: "%Y-%m-%d",
            31: "%Y-%m-%d",
        }
        py_fmt = fmt_map.get(fmt, "%d-%b-%Y")
        return dt.strftime(py_fmt)

    # Handle string format (MATLAB -> Python strftime)
    fmt_str = str(fmt)
    fmt_str = fmt_str.replace("yyyy", "%Y")
    fmt_str = fmt_str.replace("yy", "%y")
    fmt_str = fmt_str.replace("mmm", "%b")
    fmt_str = fmt_str.replace("mm", "%m")
    fmt_str = fmt_str.replace("dd", "%d")
    fmt_str = fmt_str.replace("HH", "%H")
    fmt_str = fmt_str.replace("MM", "%M")
    fmt_str = fmt_str.replace("SS", "%S")
    return dt.strftime(fmt_str)


@_tb("datevec")
def forge_datevec(datenum_val):
    """Convert serial date number to date vector.

    datevec(739321) -> [2024 3 15 0 0 0]

    Returns [year, month, day, hour, minute, second] as a numpy array.
    """
    if isinstance(datenum_val, str):
        datenum_val = forge_datenum(datenum_val)
    dt = _datenum_to_datetime(datenum_val)
    return np.array([
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute,
        dt.second + dt.microsecond / 1e6
    ], dtype=np.float64)


# =====================================================================
# Calendar Functions
# =====================================================================

@_tb("calendar")
def forge_calendar(y=None, m=None):
    """Return a calendar matrix for a given month.

    calendar()      -> current month
    calendar(2024, 3) -> March 2024

    Returns a 6x7 matrix where rows are weeks and columns are days
    (Sunday-first). Zeros represent days outside the month.
    """
    if y is None or m is None:
        t = _dt.date.today()
        y = t.year
        m = t.month
    y, m = int(y), int(m)

    cal = _calendar.monthcalendar(y, m)
    # Pad to exactly 6 rows
    while len(cal) < 6:
        cal.append([0, 0, 0, 0, 0, 0, 0])

    mat = np.array(cal, dtype=np.float64)
    # monthcalendar: Mon(0)..Sun(6) -> MATLAB: Sun(0)..Sat(6)
    mat = mat[:, [6, 0, 1, 2, 3, 4, 5]]
    return _wrap(mat)


@_tb("weekday")
def forge_weekday(datenum_val):
    """Return day of week from datenum.

    [d, name] = weekday(datenum)
    d: 1=Sunday, 2=Monday, ..., 7=Saturday
    name: 'Sunday', 'Monday', etc.

    Returns (day_number, day_name).
    """
    if isinstance(datenum_val, str):
        datenum_val = forge_datenum(datenum_val)
    dt = _datenum_to_datetime(datenum_val)
    # isoweekday: Mon=1..Sun=7 -> MATLAB: Sun=1..Sat=7
    iso = dt.isoweekday()
    matlab_day = (iso % 7) + 1
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday"]
    return (matlab_day, day_names[matlab_day - 1])


@_tb("eomday")
def forge_eomday(y, m):
    """Return last day of month for given year and month.

    eomday(2024, 2) -> 29  (leap year)
    eomday(2023, 2) -> 28
    """
    y = np.asarray(y, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    if y.ndim == 0 and m.ndim == 0:
        return _calendar.monthrange(int(y), int(m))[1]
    # Vectorized
    y_flat = y.ravel()
    m_flat = m.ravel()
    result = np.array([_calendar.monthrange(int(yi), int(mi))[1]
                       for yi, mi in zip(y_flat, m_flat)], dtype=np.float64)
    return _wrap(result.reshape(y.shape) if y.ndim > 0 else result)


@_tb("is_leap_year")
def forge_is_leap_year(y=None):
    """Check if year is a leap year.

    is_leap_year(2024) -> true
    is_leap_year(2023) -> false
    is_leap_year()     -> check current year
    """
    if y is None:
        y = _dt.date.today().year
    y = int(y)
    return _calendar.isleap(y)


# =====================================================================
# Elapsed Time
# =====================================================================

@_tb("etime")
def forge_etime(t1, t0):
    """Elapsed time between two clock vectors (in seconds).

    t0 = clock(); ... ; t1 = clock();
    elapsed = etime(t1, t0)

    Each clock vector: [Y M D H Mi S].
    """
    t1 = np.asarray(t1, dtype=np.float64).ravel()
    t0 = np.asarray(t0, dtype=np.float64).ravel()
    dt1 = _dt.datetime(int(t1[0]), int(t1[1]), int(t1[2]),
                       int(t1[3]), int(t1[4]), int(t1[5]))
    dt0 = _dt.datetime(int(t0[0]), int(t0[1]), int(t0[2]),
                       int(t0[3]), int(t0[4]), int(t0[5]))
    return (dt1 - dt0).total_seconds()


@_tb("addtodate")
def forge_addtodate(dn, qty, units):
    """Add a quantity of time units to a datenum.

    addtodate(datenum, 1, 'month')  -> datenum + 1 month
    addtodate(datenum, 5, 'day')    -> datenum + 5 days
    addtodate(datenum, 2, 'year')   -> datenum + 2 years
    addtodate(datenum, 3, 'hour')   -> datenum + 3 hours
    addtodate(datenum, 30, 'minute') -> datenum + 30 minutes
    addtodate(datenum, 10, 'second') -> datenum + 10 seconds
    """
    dt = _datenum_to_datetime(dn)
    qty = int(qty)
    units = str(units).lower()

    if units in ("day", "days"):
        dt += _dt.timedelta(days=qty)
    elif units in ("hour", "hours"):
        dt += _dt.timedelta(hours=qty)
    elif units in ("minute", "minutes"):
        dt += _dt.timedelta(minutes=qty)
    elif units in ("second", "seconds"):
        dt += _dt.timedelta(seconds=qty)
    elif units in ("month", "months"):
        new_month = dt.month + qty
        new_year = dt.year + (new_month - 1) // 12
        new_month = (new_month - 1) % 12 + 1
        max_day = _calendar.monthrange(new_year, new_month)[1]
        new_day = min(dt.day, max_day)
        dt = dt.replace(year=new_year, month=new_month, day=new_day)
    elif units in ("year", "years"):
        new_year = dt.year + qty
        max_day = _calendar.monthrange(new_year, dt.month)[1]
        new_day = min(dt.day, max_day)
        dt = dt.replace(year=new_year, day=new_day)
    else:
        raise ValueError(f"Unknown time unit: '{units}'")

    return _datetime_to_datenum(dt)
