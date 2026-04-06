# Copyright 2026 The Commons
# SPDX-License-Identifier: Apache-2.0
"""Polish R9: evaluator edge-case tests.

V-model traceability backfill: R-POL9-01 through R-POL9-07.
"""
import pytest
from forge.engine.session import ForgeSession


@pytest.fixture
def s():
    return ForgeSession()


# ---------- 1. switch with string cases ----------
class TestSwitchStringCases:
    """R-POL9-01: switch/case SHALL match string values, selecting the
    correct branch or falling through to otherwise when no case matches.

    Model-user argument: Engineers writing command dispatchers and
    option parsers use switch/case with string values constantly. This
    is the standard pattern for routing user input or configuration
    options in Octave scripts. String comparison must be exact and
    the otherwise clause must serve as the default fallback.

    Decomposition:
      R-POL9-01a: First string case matches correctly.
      R-POL9-01b: Second string case matches correctly.
      R-POL9-01c: No matching case leaves variable unchanged.
      R-POL9-01d: otherwise clause executes when no case matches.

    Consistency: First-match, second-match, no-match, and otherwise
    cover all possible branch outcomes of switch/case.
    """

    def test_switch_matches_first_string(self, s):
        """R-POL9-01a: First string case matches."""
        s.eval('switch "hello"; case "hello"; x=1; case "world"; x=2; end')
        x = s._engine.workspace.get("x")
        assert float(x.data.flat[0]) == 1.0

    def test_switch_matches_second_string(self, s):
        """R-POL9-01b: Second string case matches."""
        s.eval('switch "world"; case "hello"; x=1; case "world"; x=2; end')
        x = s._engine.workspace.get("x")
        assert float(x.data.flat[0]) == 2.0

    def test_switch_no_match(self, s):
        """R-POL9-01c: No matching case leaves variable unchanged."""
        s.eval('x=0; switch "other"; case "hello"; x=1; case "world"; x=2; end')
        x = s._engine.workspace.get("x")
        assert float(x.data.flat[0]) == 0.0

    def test_switch_otherwise(self, s):
        """R-POL9-01d: otherwise clause executes on no match."""
        s.eval('switch "other"; case "hello"; x=1; otherwise; x=99; end')
        x = s._engine.workspace.get("x")
        assert float(x.data.flat[0]) == 99.0


# ---------- 2. for loop over cell array ----------
class TestForCellArray:
    """R-POL9-02: for loops SHALL iterate over cell array elements,
    assigning each element to the loop variable in sequence.

    Model-user argument: Engineers processing heterogeneous data stored
    in cell arrays use for-over-cell to iterate without manual indexing.
    This pattern is common when processing lists of file names, mixed
    numeric/string parameters, or variable-length records.

    Decomposition:
      R-POL9-02a: for over numeric cell sums all elements.
      R-POL9-02b: for over cell counts iterations correctly.
      R-POL9-02c: for over mixed-type cell completes without error.

    Consistency: Numeric accumulation, counting, and mixed-type iteration
    cover the common use patterns and the type-safety edge case.
    """

    def test_for_cell_sum(self, s):
        """R-POL9-02a: for over {1,2,3} sums to 6."""
        r = s.eval("x=0; for i={1,2,3}; x=x+i; end; x")
        assert "6" in r

    def test_for_cell_count(self, s):
        """R-POL9-02b: for over 4-element cell counts 4 iterations."""
        r = s.eval("n=0; for i={10,20,30,40}; n=n+1; end; n")
        assert "4" in r

    def test_for_cell_mixed_types(self, s):
        """R-POL9-02c: for over mixed-type cell completes without error."""
        # Should iterate without error even with mixed types
        s.eval('for i={"a", 1, "b"}; end')


# ---------- 3. string concatenation via horzcat ----------
class TestStringConcat:
    """R-POL9-03: Double-quoted string arrays inside brackets SHALL
    concatenate horizontally, matching Octave's horzcat behavior.

    Model-user argument: Engineers building display strings and file
    paths use ["prefix" " " "suffix"] syntax for string assembly. This
    bracket-based concatenation is the idiomatic Octave way to build
    strings without calling strcat.

    Decomposition:
      R-POL9-03a: Three-string horzcat produces joined result.
      R-POL9-03b: Two-string horzcat produces joined result.

    Consistency: Three-element and two-element cases confirm the
    concatenation works for varying numbers of operands.
    """

    def test_horzcat_strings(self, s):
        """R-POL9-03a: ["hello" " " "world"] -> "hello world"."""
        s.eval('s = ["hello" " " "world"]')
        val = s._engine.workspace.get("s")
        assert val.to_str() == "hello world"

    def test_horzcat_two_strings(self, s):
        """R-POL9-03b: ["foo" "bar"] -> "foobar"."""
        s.eval('s = ["foo" "bar"]')
        val = s._engine.workspace.get("s")
        assert val.to_str() == "foobar"


# ---------- 4. end keyword in indexing ----------
class TestEndKeyword:
    """R-POL9-04: The 'end' keyword in array indexing SHALL resolve to
    the last index along the relevant dimension, supporting arithmetic
    expressions like end-1.

    Model-user argument: Engineers access the last element of arrays
    using A(end) and nearby elements using A(end-1), A(end-2), etc.
    This is one of the most common Octave indexing idioms and must
    work correctly for migrated code.

    Decomposition:
      R-POL9-04a: A(end) returns the last element.
      R-POL9-04b: A(end-1) returns the second-to-last element.
      R-POL9-04c: A(end-2) returns the third-to-last element.

    Consistency: end, end-1, and end-2 confirm that arithmetic on the
    end keyword evaluates correctly at different offsets.
    """

    def test_end_last_element(self, s):
        """R-POL9-04a: A(end) returns last element."""
        r = s.eval("A=[1 2 3 4 5]; A(end)")
        assert "5" in r

    def test_end_minus_one(self, s):
        """R-POL9-04b: A(end-1) returns second-to-last element."""
        r = s.eval("A=[1 2 3 4 5]; A(end-1)")
        assert "4" in r

    def test_end_minus_two(self, s):
        """R-POL9-04c: A(end-2) returns third-to-last element."""
        r = s.eval("A=[10 20 30 40 50]; A(end-2)")
        assert "30" in r


# ---------- 5. isfield ----------
class TestIsfield:
    """R-POL9-05: isfield SHALL return 1 when the named field exists in
    a struct, and 0 when it does not.

    Model-user argument: Engineers use isfield as a guard before
    accessing struct fields to avoid runtime errors. This pattern is
    standard in Octave code that processes structs with optional or
    variable fields.

    Decomposition:
      R-POL9-05a: isfield returns 1 for existing field.
      R-POL9-05b: isfield returns 0 for nonexistent field.

    Consistency: True and false cases cover both branches.
    """

    def test_isfield_true(self, s):
        """R-POL9-05a: isfield returns 1 for existing field."""
        r = s.eval('st.x=1; st.y=2; isfield(st, "x")')
        assert "1" in r

    def test_isfield_false(self, s):
        """R-POL9-05b: isfield returns 0 for nonexistent field."""
        r = s.eval('st.x=1; isfield(st, "z")')
        assert "0" in r


# ---------- 6. error ID handling ----------
class TestErrorId:
    """R-POL9-06: error() SHALL support the (identifier, format, args)
    calling convention, populating both .identifier and .message on the
    caught exception, and the simple (message) form.

    Model-user argument: Engineers classify errors by ID for
    programmatic error handling (e.g., catching specific error types
    while letting others propagate). The format-string variant with
    sprintf-style arguments is used to include diagnostic values in
    error messages.

    Decomposition:
      R-POL9-06a: error(id, fmt, args) populates .identifier and .message.
      R-POL9-06b: error(msg) populates .message with the plain string.

    Consistency: ID-based and simple forms cover the two calling
    conventions.
    """

    def test_error_id_and_message(self, s):
        """R-POL9-06a: error(id, fmt, args) populates identifier and message."""
        r = s.eval('try; error("myid:myerr", "msg %d", 42); catch e; disp(e.identifier); disp(e.message); end')
        assert "myid:myerr" in r
        assert "msg 42" in r

    def test_error_simple(self, s):
        """R-POL9-06b: error(msg) populates .message."""
        r = s.eval('try; error("something broke"); catch e; disp(e.message); end')
        assert "something broke" in r


# ---------- 7. warning function ----------
class TestWarning:
    """R-POL9-07: warning() SHALL display a warning message with the
    'warning:' prefix, supporting plain strings, format strings with
    arguments, and the (id, format, args) form.

    Model-user argument: Engineers use warning() to emit non-fatal
    diagnostics during computation (e.g., convergence issues, deprecated
    usage). The output must include the 'warning:' prefix so it is
    visually distinguishable from regular output, matching Octave
    behavior.

    Decomposition:
      R-POL9-07a: warning(msg) displays with 'warning:' prefix.
      R-POL9-07b: warning(fmt, args) formats the message.
      R-POL9-07c: warning(id, fmt, args) formats and displays the message.
      R-POL9-07d: warning(id, fmt, numeric_arg) formats numeric values.

    Consistency: Plain, formatted, ID+formatted, and numeric-format
    variants cover all four calling conventions of warning().
    """

    def test_warning_simple(self, s):
        """R-POL9-07a: warning('hello') displays 'warning: hello'."""
        r = s.eval('warning("hello")')
        assert "warning: hello" in r

    def test_warning_format(self, s):
        """R-POL9-07b: warning('hello %s', 'world') formats correctly."""
        r = s.eval('warning("hello %s", "world")')
        assert "warning: hello world" in r

    def test_warning_id_format(self, s):
        """R-POL9-07c: warning(id, fmt, args) formats the message."""
        r = s.eval('warning("test:warn", "hello %s", "world")')
        assert "warning: hello world" in r

    def test_warning_id_numeric_format(self, s):
        """R-POL9-07d: warning(id, fmt, num) formats numeric value."""
        r = s.eval('warning("test:warn", "value is %d", 42)')
        assert "warning: value is 42" in r
