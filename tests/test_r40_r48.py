"""Tests for R-40 through R-48."""
import os, sys
os.environ["MPLBACKEND"] = "Agg"
import pytest
sys.path.insert(0, os.path.expanduser("~/forge"))
from forge.engine.session import ForgeSession

@pytest.fixture
def s():
    return ForgeSession()

# --- R-40: sort with descend mode ---

def test_sort_descend_vector(s):
    """sort(v,'descend') returns descending order."""
    r = s.eval("sort([3 1 4 1 5],'descend')")
    assert "5" in r and r.index("5") < r.index("4"), repr(r)

def test_sort_descend_indices(s):
    """[sv,i]=sort(v,'descend') returns correct descending indices."""
    r = s.eval("[sv,i]=sort([3 1 2],'descend'); i")
    assert "1" in r  # index of 3 (max) should be first
    # [3 1 2] descend -> [3 2 1], indices [1 3 2]
    assert r.strip().startswith("1") or "1    3    2" in r, repr(r)

def test_sort_ascend_unchanged(s):
    """sort(v,'ascend') still works."""
    r = s.eval("sort([3 1 2],'ascend')")
    assert "1    2    3" in r, repr(r)

# --- R-40: rem semantics ---

def test_rem_negative_dividend(s):
    """rem(-3, 4) returns -3 (sign follows dividend)."""
    r = s.eval("rem(-3, 4)")
    assert "-3" in r, repr(r)

def test_rem_positive_both(s):
    """rem(7, 3) returns 1 (same as mod for positives)."""
    r = s.eval("rem(7, 3)")
    assert "1" in r, repr(r)

def test_rem_negative_divisor(s):
    """rem(3, -4) returns 3 (sign follows dividend)."""
    r = s.eval("rem(3, -4)")
    assert "3" in r, repr(r)

# --- R-42: round half-away-from-zero ---

def test_round_half_up(s):
    """round(2.5) returns 3 (MATLAB rounds half away from zero)."""
    r = s.eval("round(2.5)")
    assert "3" in r, repr(r)

def test_round_half_negative(s):
    """round(-2.5) returns -3."""
    r = s.eval("round(-2.5)")
    assert "-3" in r, repr(r)

def test_round_normal(s):
    """round(2.3) returns 2 (unchanged)."""
    r = s.eval("round(2.3)")
    assert "2" in r and "3" not in r, repr(r)

# --- R-43: cast rounds and clamps ---

def test_cast_rounds(s):
    """cast(3.7,'int32') returns 4 (rounds, not truncates)."""
    r = s.eval("cast(3.7,'int32')")
    assert "4" in r, repr(r)

def test_cast_clamps_max(s):
    """cast(200,'int8') returns 127 (saturates at max)."""
    r = s.eval("cast(200,'int8')")
    assert "127" in r, repr(r)

def test_cast_clamps_min(s):
    """cast(-200,'int8') returns -128 (saturates at min)."""
    r = s.eval("cast(-200,'int8')")
    assert "-128" in r, repr(r)

def test_cast_negative_rounds(s):
    """cast(-3.7,'int32') returns -4."""
    r = s.eval("cast(-3.7,'int32')")
    assert "-4" in r, repr(r)

# --- R-44: zeros/ones with typename ---

def test_zeros_typename(s):
    """zeros(2,'uint8') returns uint8 array without error."""
    r = s.eval("class(zeros(2,'uint8'))")
    assert "uint8" in r, repr(r)

def test_ones_typename_3arg(s):
    """ones(2,3,'int32') returns int32 array."""
    r = s.eval("class(ones(2,3,'int32'))")
    assert "int32" in r, repr(r)

def test_zeros_logical_typename(s):
    """zeros(2,'logical') works."""
    r = s.eval("class(zeros(2,'logical'))")
    assert "logical" in r, repr(r)

# --- R-45: isempty for cell ---

def test_isempty_empty_cell(s):
    """isempty({}) returns 1."""
    r = s.eval("isempty({})")
    assert "1" in r, repr(r)

def test_isempty_nonempty_cell(s):
    """isempty({1,2}) returns 0."""
    r = s.eval("isempty({1,2})")
    assert "0" in r, repr(r)

# --- R-46: strcmp with cell array ---

def test_strcmp_cell_vs_string(s):
    """strcmp({'a','b','c'},'b') returns logical row vector [0 1 0]."""
    r = s.eval("strcmp({'a','b','c'},'b')")
    assert "0" in r and "1" in r, repr(r)
    # Check it's a vector: should have 3 elements
    nums = [x.strip() for x in r.split() if x.strip() in ("0","1")]
    assert len(nums) == 3, repr(r)

def test_strcmp_string_vs_cell(s):
    """strcmp('b',{'a','b','c'}) also returns [0 1 0]."""
    r = s.eval("strcmp('b',{'a','b','c'})")
    nums = [x.strip() for x in r.split() if x.strip() in ("0","1")]
    assert len(nums) == 3, repr(r)

# --- R-47: strncmp ---

def test_strncmp_match(s):
    """strncmp('abcde','abcxy',3) returns 1 (first 3 chars match)."""
    r = s.eval("strncmp('abcde','abcxy',3)")
    assert "1" in r, repr(r)

def test_strncmp_no_match(s):
    """strncmp('abcde','abcxy',4) returns 0 (4th char differs)."""
    r = s.eval("strncmp('abcde','abcxy',4)")
    assert "0" in r, repr(r)

def test_strncmpi_case_insensitive(s):
    """strncmpi('ABCde','abcxy',3) returns 1."""
    r = s.eval("strncmpi('ABCde','abcxy',3)")
    assert "1" in r, repr(r)

# --- R-48: inputParser.parse ---

def test_inputparser_parse_required(s):
    """parse(p, val) with addRequired populates p.Results.x."""
    r = s.eval("p=inputParser(); p.addRequired('x'); parse(p, 42); p.Results.x")
    assert "42" in r, repr(r)

def test_inputparser_parse_optional_default(s):
    """parse(p) with addOptional uses default value."""
    r = s.eval("p=inputParser(); p.addOptional('x', 99); parse(p); p.Results.x")
    assert "99" in r, repr(r)

def test_inputparser_parse_optional_supplied(s):
    """parse(p, val) with addOptional uses supplied value."""
    r = s.eval("p=inputParser(); p.addOptional('x', 0); parse(p, 7); p.Results.x")
    assert "7" in r, repr(r)
