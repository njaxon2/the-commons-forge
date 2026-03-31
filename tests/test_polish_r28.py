# Copyright 2026 The Commons (TM)
# SPDX-License-Identifier: Apache-2.0
"""Tests for file I/O, strsplit, regexp named tokens, cell {end+1},
fseek string origins, fread/fwrite binary (R28 polish)."""
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
    def test_fopen_fprintf_fclose(self, s):
        """fopen write mode, fprintf to file, fclose."""
        s.eval("fid = fopen('/tmp/forge_r28_test.txt', 'w')")
        s.eval(r"fprintf(fid, 'alpha\nbeta\ngamma\n')")
        s.eval("fclose(fid)")
        with open("/tmp/forge_r28_test.txt") as f:
            assert f.read() == "alpha\nbeta\ngamma\n"

    def test_fgets_returns_line_with_newline(self, s):
        """fgets returns the line including the trailing newline."""
        with open("/tmp/forge_r28_fgets.txt", "w") as f:
            f.write("hello\nworld\n")
        s.eval("fid = fopen('/tmp/forge_r28_fgets.txt', 'r')")
        s.eval("line = fgets(fid)")
        s.eval("fclose(fid)")
        val = s.eval("line")
        assert str(val) == "hello\n"

    def test_fgetl_strips_newline(self, s):
        """fgetl returns line without newline."""
        with open("/tmp/forge_r28_fgetl.txt", "w") as f:
            f.write("hello\nworld\n")
        s.eval("fid = fopen('/tmp/forge_r28_fgetl.txt', 'r')")
        s.eval("line = fgetl(fid)")
        s.eval("fclose(fid)")
        val = s.eval("line")
        assert str(val) == "hello"

    def test_feof_loop_reads_all_lines(self, s):
        """while ~feof loop with fgetl and {end+1} collects all lines."""
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
        """ftell returns current position after reading."""
        with open("/tmp/forge_r28_ftell.txt", "w") as f:
            f.write("abcdef\n")
        s.eval("fid = fopen('/tmp/forge_r28_ftell.txt', 'r')")
        s.eval("fgets(fid)")
        pos = _unwrap(s.eval("ftell(fid)"))
        s.eval("fclose(fid)")
        assert pos == 7  # 6 chars + newline

    def test_fseek_bof_string(self, s):
        """fseek with 'bof' string origin resets to beginning."""
        with open("/tmp/forge_r28_fseek.txt", "w") as f:
            f.write("abcdef\n")
        s.eval("fid = fopen('/tmp/forge_r28_fseek.txt', 'r')")
        s.eval("fgets(fid)")
        s.eval("fseek(fid, 0, 'bof')")
        pos = _unwrap(s.eval("ftell(fid)"))
        s.eval("fclose(fid)")
        assert pos == 0

    def test_fwrite_fread_binary(self, s):
        """fwrite/fread round-trip binary doubles through file handles."""
        s.eval("fid = fopen('/tmp/forge_r28_bin.dat', 'wb')")
        s.eval("fwrite(fid, [10 20 30], 'double')")
        s.eval("fclose(fid)")
        s.eval("fid = fopen('/tmp/forge_r28_bin.dat', 'rb')")
        s.eval("data = fread(fid, 3, 'double')")
        s.eval("fclose(fid)")
        val = s.eval("data")
        np.testing.assert_array_equal(val.data.ravel(), [10, 20, 30])

    def test_fopen_invalid_returns_minus1(self, s):
        """fopen on nonexistent file in read mode returns -1."""
        fid = _unwrap(s.eval("fopen('/tmp/forge_r28_nonexistent_xyz', 'r')"))
        assert fid == -1


# ── tempname / tempdir ──────────────────────────────────────────────
class TestTempFiles:
    def test_tempname_returns_string(self, s):
        val = s.eval("tempname()")
        assert isinstance(str(val), str)
        assert len(str(val)) > 0

    def test_tempdir_returns_tmp(self, s):
        val = s.eval("tempdir()")
        assert "/tmp" in str(val) or "tmp" in str(val).lower()


# ── fileread ────────────────────────────────────────────────────────
class TestFileread:
    def test_fileread_entire_file(self, s):
        with open("/tmp/forge_r28_fileread.txt", "w") as f:
            f.write("one\ntwo\nthree\n")
        val = s.eval("fileread('/tmp/forge_r28_fileread.txt')")
        assert str(val) == "one\ntwo\nthree\n"


# ── strsplit edge cases ─────────────────────────────────────────────
class TestStrsplit:
    def test_strsplit_double_colon(self, s):
        """strsplit with '::' delimiter."""
        s.eval("r = strsplit('a::b::c', '::')")
        val = s.eval("r")
        texts = [str(d) for d in val._data]
        assert texts == ["a", "b", "c"]

    def test_strsplit_whitespace_default(self, s):
        """strsplit with default whitespace splitting."""
        s.eval("r = strsplit('  hello  world  ')")
        val = s.eval("r")
        texts = [str(d) for d in val._data]
        assert texts == ["hello", "world"]


# ── regexp named tokens ─────────────────────────────────────────────
class TestRegexpNamedTokens:
    def test_regexp_names_returns_struct(self, s):
        """regexp with 'names' returns a struct with named group fields."""
        code = r'r = regexp("John Smith 42", "(?<name>\w+ \w+) (?<age>\d+)", "names")'
        s.eval(code)
        val = s.eval("r")
        from forge.engine.containers import ForgeStruct
        assert isinstance(val, ForgeStruct)

    def test_regexp_names_field_values(self, s):
        """Named group values are accessible as struct fields."""
        code = r'r = regexp("John Smith 42", "(?<name>\w+ \w+) (?<age>\d+)", "names")'
        s.eval(code)
        name = s.eval("r.name")
        age = s.eval("r.age")
        assert str(name) == "John Smith"
        assert str(age) == "42"

    def test_regexp_names_single_group(self, s):
        """regexp named token with single group."""
        code = r'r = regexp("age:42", "age:(?<val>\d+)", "names")'
        s.eval(code)
        val = s.eval("r.val")
        assert str(val) == "42"


# ── Cell {end+1} assignment ─────────────────────────────────────────
class TestCellEndPlus1:
    def test_cell_end_plus1_grows(self, s):
        """c{end+1} = val grows the cell."""
        s.eval("c = {'a', 'b'}")
        s.eval("c{end+1} = 'c'")
        val = s.eval("c")
        assert len(val._data) == 3
        assert str(val._data[2]) == "c"

    def test_cell_end_plus1_from_empty(self, s):
        """c{end+1} works from empty cell."""
        s.eval("c = {}")
        s.eval("c{end+1} = 'first'")
        val = s.eval("c")
        assert len(val._data) == 1
        assert str(val._data[0]) == "first"
