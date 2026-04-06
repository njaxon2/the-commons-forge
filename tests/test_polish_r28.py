# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for file I/O, strsplit, regexp named tokens, cell {end+1},
fseek string origins, fread/fwrite binary (R28 polish).

V&V Traceability (backfill)
===========================
R-POL28-01: File I/O functions (fopen, fprintf, fclose, fgets, fgetl, feof,
            ftell, fseek, fwrite, fread) SHALL behave identically to Octave
            for text and binary file operations.

    Model-user argument: An engineer porting data-acquisition scripts from
    Octave relies on fopen/fprintf/fclose for logging, fgets/fgetl for
    reading configuration files line by line, and fwrite/fread for binary
    sensor data. Any deviation causes silent data corruption or I/O failures
    in production pipelines.

    Decomposition:
      R-POL28-01a: fopen('w') + fprintf + fclose writes text correctly.
      R-POL28-01b: fgets returns the line including trailing newline.
      R-POL28-01c: fgetl returns the line without trailing newline.
      R-POL28-01d: while ~feof loop with fgetl and {end+1} reads all lines.
      R-POL28-01e: ftell returns current file position after reading.
      R-POL28-01f: fseek with 'bof' string origin resets to beginning.
      R-POL28-01g: fwrite/fread round-trip binary doubles through file handles.
      R-POL28-01h: fopen on nonexistent file in read mode returns -1.

    Consistency: Sub-requirements cover write (01a), line reading with and
    without newline (01b-c), loop-based full-file read (01d), position
    tracking (01e-f), binary I/O (01g), and error handling (01h).

R-POL28-02: tempname and tempdir SHALL return valid temporary path strings.

    Model-user argument: Scripts that generate intermediate files use
    tempname() and tempdir() for safe temporary paths. These must return
    valid, non-empty strings so file operations do not fail.

    Decomposition:
      R-POL28-02a: tempname() returns a non-empty string.
      R-POL28-02b: tempdir() returns a path containing "tmp".

    Consistency: Both temporary-path functions are covered.

R-POL28-03: fileread SHALL return the entire contents of a text file as a
            single string.

    Model-user argument: Scientists use fileread() as a quick way to slurp
    an entire configuration or data file into a variable. It must return the
    complete file contents without truncation.

    Decomposition:
      R-POL28-03a: fileread returns full file contents as a string.

    Consistency: Single function, single sub-requirement.

R-POL28-04: strsplit SHALL correctly split strings on custom and default
            delimiters.

    Model-user argument: Parsing CSV lines or whitespace-delimited data is
    a daily task for engineers processing instrument output. strsplit must
    handle both explicit delimiters and default whitespace splitting.

    Decomposition:
      R-POL28-04a: strsplit with '::' delimiter splits correctly.
      R-POL28-04b: strsplit with default whitespace splitting trims and splits.

    Consistency: Custom delimiter (04a) and default whitespace (04b) cover
    the two main usage patterns.

R-POL28-05: regexp with 'names' option SHALL return a struct with named
            capture group fields.

    Model-user argument: Engineers parsing structured text (log files, sensor
    headers) use named capture groups to extract fields by name. The result
    must be a struct with field names matching the group names, not a raw
    cell array.

    Decomposition:
      R-POL28-05a: regexp 'names' returns a ForgeStruct.
      R-POL28-05b: Named group values are accessible as struct fields.
      R-POL28-05c: Single named group works correctly.

    Consistency: Type check (05a), multi-group access (05b), and
    single-group edge case (05c) cover the named-token API.

R-POL28-06: Cell arrays SHALL support {end+1} assignment to grow the array,
            including from empty.

    Model-user argument: Accumulating results in a loop via c{end+1} = val
    is idiomatic Octave. If this pattern fails, engineers must rewrite
    collection loops, which is a common migration blocker.

    Decomposition:
      R-POL28-06a: c{end+1} grows an existing cell by one element.
      R-POL28-06b: c{end+1} works from an empty cell.

    Consistency: Non-empty (06a) and empty (06b) starting points are both
    covered.
"""
import os
import pytest
import numpy as np
from forge.engine.session import ForgeSession
from forge.engine.types import ForgeArray, _unwrap


@pytest.fixture
def sess():
    return ForgeSession()


@pytest.fixture
def s(sess):
    return sess._engine


# ── File I/O: fopen / fprintf / fclose ──────────────────────────────
class TestFileIO:
    """R-POL28-01: File I/O functions SHALL behave identically to Octave for
    text and binary file operations.

    Model-user argument: An engineer porting data-acquisition scripts from
    Octave relies on fopen/fprintf/fclose for logging, fgets/fgetl for
    reading configuration files line by line, and fwrite/fread for binary
    sensor data. Any deviation causes silent data corruption or I/O failures
    in production pipelines.

    Decomposition:
      R-POL28-01a: fopen('w') + fprintf + fclose writes text correctly.
      R-POL28-01b: fgets returns the line including trailing newline.
      R-POL28-01c: fgetl returns the line without trailing newline.
      R-POL28-01d: while ~feof loop with fgetl and {end+1} reads all lines.
      R-POL28-01e: ftell returns current file position after reading.
      R-POL28-01f: fseek with 'bof' string origin resets to beginning.
      R-POL28-01g: fwrite/fread round-trip binary doubles through file handles.
      R-POL28-01h: fopen on nonexistent file in read mode returns -1.

    Consistency: Sub-requirements cover write (01a), line reading with and
    without newline (01b-c), loop-based full-file read (01d), position
    tracking (01e-f), binary I/O (01g), and error handling (01h).
    """

    def test_fopen_fprintf_fclose(self, s):
        """R-POL28-01a: fopen('w') + fprintf + fclose SHALL write text correctly."""
        s.eval("fid = fopen('/tmp/forge_r28_test.txt', 'w')")
        s.eval(r"fprintf(fid, 'alpha\nbeta\ngamma\n')")
        s.eval("fclose(fid)")
        with open("/tmp/forge_r28_test.txt") as f:
            assert f.read() == "alpha\nbeta\ngamma\n"

    def test_fgets_returns_line_with_newline(self, s):
        """R-POL28-01b: fgets SHALL return the line including trailing newline."""
        with open("/tmp/forge_r28_fgets.txt", "w") as f:
            f.write("hello\nworld\n")
        s.eval("fid = fopen('/tmp/forge_r28_fgets.txt', 'r')")
        s.eval("line = fgets(fid)")
        s.eval("fclose(fid)")
        val = s.eval("line")
        assert str(val) == "hello\n"

    def test_fgetl_strips_newline(self, s):
        """R-POL28-01c: fgetl SHALL return line without trailing newline."""
        with open("/tmp/forge_r28_fgetl.txt", "w") as f:
            f.write("hello\nworld\n")
        s.eval("fid = fopen('/tmp/forge_r28_fgetl.txt', 'r')")
        s.eval("line = fgetl(fid)")
        s.eval("fclose(fid)")
        val = s.eval("line")
        assert str(val) == "hello"

    def test_feof_loop_reads_all_lines(self, s):
        """R-POL28-01d: while ~feof loop with fgetl and {end+1} SHALL read all lines."""
        with open("/tmp/forge_r28_feof.txt", "w") as f:
            f.write("line1\nline2\nline3\n")
        s.eval("fid = fopen('/tmp/forge_r28_feof.txt', 'r')")
        s.eval("collected = {}")
        s.eval("while ~feof(fid); ln = fgetl(fid); collected{end+1} = ln; end")
        s.eval("fclose(fid)")
        val = s.eval("collected")
        assert hasattr(val, '_data')
        texts = [str(d) for d in val._data]
        assert texts == ["line1", "line2", "line3"]

    def test_ftell_position(self, s):
        """R-POL28-01e: ftell SHALL return current position after reading."""
        with open("/tmp/forge_r28_ftell.txt", "w") as f:
            f.write("abcdef\n")
        s.eval("fid = fopen('/tmp/forge_r28_ftell.txt', 'r')")
        s.eval("fgets(fid)")
        pos = _unwrap(s.eval("ftell(fid)"))
        s.eval("fclose(fid)")
        assert pos == 7  # 6 chars + newline

    def test_fseek_bof_string(self, s):
        """R-POL28-01f: fseek with 'bof' string origin SHALL reset to beginning."""
        with open("/tmp/forge_r28_fseek.txt", "w") as f:
            f.write("abcdef\n")
        s.eval("fid = fopen('/tmp/forge_r28_fseek.txt', 'r')")
        s.eval("fgets(fid)")
        s.eval("fseek(fid, 0, 'bof')")
        pos = _unwrap(s.eval("ftell(fid)"))
        s.eval("fclose(fid)")
        assert pos == 0

    def test_fwrite_fread_binary(self, s):
        """R-POL28-01g: fwrite/fread SHALL round-trip binary doubles through file handles."""
        s.eval("fid = fopen('/tmp/forge_r28_bin.dat', 'wb')")
        s.eval("fwrite(fid, [10 20 30], 'double')")
        s.eval("fclose(fid)")
        s.eval("fid = fopen('/tmp/forge_r28_bin.dat', 'rb')")
        s.eval("data = fread(fid, 3, 'double')")
        s.eval("fclose(fid)")
        val = s.eval("data")
        np.testing.assert_array_equal(val.data.ravel(), [10, 20, 30])

    def test_fopen_invalid_returns_minus1(self, s):
        """R-POL28-01h: fopen on nonexistent file in read mode SHALL return -1."""
        fid = _unwrap(s.eval("fopen('/tmp/forge_r28_nonexistent_xyz', 'r')"))
        assert fid == -1


# ── tempname / tempdir ──────────────────────────────────────────────
class TestTempFiles:
    """R-POL28-02: tempname and tempdir SHALL return valid temporary path strings.

    Model-user argument: Scripts that generate intermediate files use
    tempname() and tempdir() for safe temporary paths. These must return
    valid, non-empty strings so file operations do not fail.

    Decomposition:
      R-POL28-02a: tempname() returns a non-empty string.
      R-POL28-02b: tempdir() returns a path containing "tmp".

    Consistency: Both temporary-path functions are covered.
    """

    def test_tempname_returns_string(self, s):
        """R-POL28-02a: tempname() SHALL return a non-empty string."""
        val = s.eval("tempname()")
        assert isinstance(str(val), str)
        assert len(str(val)) > 0

    def test_tempdir_returns_tmp(self, s):
        """R-POL28-02b: tempdir() SHALL return a path containing 'tmp'."""
        val = s.eval("tempdir()")
        assert "/tmp" in str(val) or "tmp" in str(val).lower()


# ── fileread ────────────────────────────────────────────────────────
class TestFileread:
    """R-POL28-03: fileread SHALL return the entire contents of a text file
    as a single string.

    Model-user argument: Scientists use fileread() as a quick way to slurp
    an entire configuration or data file into a variable. It must return the
    complete file contents without truncation.

    Decomposition:
      R-POL28-03a: fileread returns full file contents as a string.

    Consistency: Single function, single sub-requirement.
    """

    def test_fileread_entire_file(self, s):
        """R-POL28-03a: fileread SHALL return full file contents as a string."""
        with open("/tmp/forge_r28_fileread.txt", "w") as f:
            f.write("one\ntwo\nthree\n")
        val = s.eval("fileread('/tmp/forge_r28_fileread.txt')")
        assert str(val) == "one\ntwo\nthree\n"


# ── strsplit edge cases ─────────────────────────────────────────────
class TestStrsplit:
    """R-POL28-04: strsplit SHALL correctly split strings on custom and
    default delimiters.

    Model-user argument: Parsing CSV lines or whitespace-delimited data is
    a daily task for engineers processing instrument output. strsplit must
    handle both explicit delimiters and default whitespace splitting.

    Decomposition:
      R-POL28-04a: strsplit with '::' delimiter splits correctly.
      R-POL28-04b: strsplit with default whitespace splitting trims and splits.

    Consistency: Custom delimiter (04a) and default whitespace (04b) cover
    the two main usage patterns.
    """

    def test_strsplit_double_colon(self, s):
        """R-POL28-04a: strsplit with '::' delimiter SHALL split correctly."""
        s.eval("r = strsplit('a::b::c', '::')")
        val = s.eval("r")
        texts = [str(d) for d in val._data]
        assert texts == ["a", "b", "c"]

    def test_strsplit_whitespace_default(self, s):
        """R-POL28-04b: strsplit with default whitespace SHALL trim and split."""
        s.eval("r = strsplit('  hello  world  ')")
        val = s.eval("r")
        texts = [str(d) for d in val._data]
        assert texts == ["hello", "world"]


# ── regexp named tokens ─────────────────────────────────────────────
class TestRegexpNamedTokens:
    """R-POL28-05: regexp with 'names' option SHALL return a struct with
    named capture group fields.

    Model-user argument: Engineers parsing structured text (log files, sensor
    headers) use named capture groups to extract fields by name. The result
    must be a struct with field names matching the group names, not a raw
    cell array.

    Decomposition:
      R-POL28-05a: regexp 'names' returns a ForgeStruct.
      R-POL28-05b: Named group values are accessible as struct fields.
      R-POL28-05c: Single named group works correctly.

    Consistency: Type check (05a), multi-group access (05b), and
    single-group edge case (05c) cover the named-token API.
    """

    def test_regexp_names_returns_struct(self, s):
        """R-POL28-05a: regexp with 'names' SHALL return a ForgeStruct."""
        code = r'r = regexp("John Smith 42", "(?<name>\w+ \w+) (?<age>\d+)", "names")'
        s.eval(code)
        val = s.eval("r")
        from forge.engine.containers import ForgeStruct
        assert isinstance(val, ForgeStruct)

    def test_regexp_names_field_values(self, s):
        """R-POL28-05b: Named group values SHALL be accessible as struct fields."""
        code = r'r = regexp("John Smith 42", "(?<name>\w+ \w+) (?<age>\d+)", "names")'
        s.eval(code)
        name = s.eval("r.name")
        age = s.eval("r.age")
        assert str(name) == "John Smith"
        assert str(age) == "42"

    def test_regexp_names_single_group(self, s):
        """R-POL28-05c: regexp named token with single group SHALL work correctly."""
        code = r'r = regexp("age:42", "age:(?<val>\d+)", "names")'
        s.eval(code)
        val = s.eval("r.val")
        assert str(val) == "42"


# ── Cell {end+1} assignment ─────────────────────────────────────────
class TestCellEndPlus1:
    """R-POL28-06: Cell arrays SHALL support {end+1} assignment to grow the
    array, including from empty.

    Model-user argument: Accumulating results in a loop via c{end+1} = val
    is idiomatic Octave. If this pattern fails, engineers must rewrite
    collection loops, which is a common migration blocker.

    Decomposition:
      R-POL28-06a: c{end+1} grows an existing cell by one element.
      R-POL28-06b: c{end+1} works from an empty cell.

    Consistency: Non-empty (06a) and empty (06b) starting points are both
    covered.
    """

    def test_cell_end_plus1_grows(self, s):
        """R-POL28-06a: c{end+1} = val SHALL grow the cell by one element."""
        s.eval("c = {'a', 'b'}")
        s.eval("c{end+1} = 'c'")
        val = s.eval("c")
        assert len(val._data) == 3
        assert str(val._data[2]) == "c"

    def test_cell_end_plus1_from_empty(self, s):
        """R-POL28-06b: c{end+1} SHALL work from an empty cell."""
        s.eval("c = {}")
        s.eval("c{end+1} = 'first'")
        val = s.eval("c")
        assert len(val._data) == 1
        assert str(val._data[0]) == "first"
