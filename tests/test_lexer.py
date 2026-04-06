# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for M-language lexer. Stage 2.1 from V&V plan.

V&V traceability backfill: R-LEX-01 through R-LEX-16.
"""
import pytest
from forge.engine.lexer import tokenize, TokenType, LexerError, KEYWORDS


def tok_types(src):
    """Return list of (type, value) tuples, excluding EOF."""
    return [(t.type, t.value) for t in tokenize(src) if t.type != TokenType.EOF]


def tok_values(src):
    """Return list of values, excluding EOF and NEWLINE."""
    return [t.value for t in tokenize(src) if t.type not in (TokenType.EOF, TokenType.NEWLINE)]


class TestNumbers:
    """R-LEX-01: The lexer SHALL tokenize numeric literals in integer,
    floating-point, scientific, hexadecimal, binary, and imaginary formats
    as NUMBER tokens with correct value strings.

    Model-user argument: The engineer enters numbers in many formats:
    plain integers for indices, decimals for measurements, scientific
    notation for physical constants (e.g., 1e-5 for tolerances), hex for
    bit masks, and imaginary suffixes for complex impedance calculations.
    Each format must tokenize correctly or the parser receives garbage.

    Decomposition: Each numeric format tested individually plus a
    multi-number sequence. Consistency: The ten sub-tests cover all numeric
    literal formats recognized by MATLAB/Octave.
    """

    def test_integer(self):
        """R-LEX-01.01: Integer literal tokenizes as NUMBER."""
        ts = tok_types("42")
        assert ts == [(TokenType.NUMBER, "42")]

    def test_float(self):
        """R-LEX-01.02: Floating-point literal tokenizes with decimal."""
        ts = tok_types("3.14")
        assert ts == [(TokenType.NUMBER, "3.14")]

    def test_scientific(self):
        """R-LEX-01.03: Scientific notation with negative exponent."""
        ts = tok_types("1e-5")
        assert ts == [(TokenType.NUMBER, "1e-5")]

    def test_scientific_positive(self):
        """R-LEX-01.04: Scientific notation with positive exponent and uppercase E."""
        ts = tok_types("2.5E+3")
        assert ts == [(TokenType.NUMBER, "2.5E+3")]

    def test_hex(self):
        """R-LEX-01.05: Hexadecimal literal tokenizes as NUMBER."""
        ts = tok_types("0xFF")
        assert ts == [(TokenType.NUMBER, "0xFF")]

    def test_binary(self):
        """R-LEX-01.06: Binary literal tokenizes as NUMBER."""
        ts = tok_types("0b1010")
        assert ts == [(TokenType.NUMBER, "0b1010")]

    def test_imaginary(self):
        """R-LEX-01.07: Imaginary literal with 'i' suffix."""
        ts = tok_types("3.5i")
        assert ts == [(TokenType.NUMBER, "3.5i")]

    def test_imaginary_j(self):
        """R-LEX-01.08: Imaginary literal with 'j' suffix."""
        ts = tok_types("2j")
        assert ts == [(TokenType.NUMBER, "2j")]

    def test_leading_dot(self):
        """R-LEX-01.09: Number with leading dot (.5) tokenizes correctly."""
        ts = tok_types(".5")
        assert ts == [(TokenType.NUMBER, ".5")]

    def test_multiple_numbers(self):
        """R-LEX-01.10: Multiple space-separated numbers tokenize as separate values."""
        vals = tok_values("1 2.0 3e4")
        assert vals == ["1", "2.0", "3e4"]


class TestStrings:
    """R-LEX-02: The lexer SHALL tokenize double-quoted strings as STRING
    tokens, processing escape sequences (\\n, \\t, \"\") correctly.

    Model-user argument: The engineer uses double-quoted strings for
    formatted output (e.g., sprintf format strings with \\n and \\t). Escape
    sequences must be interpreted at the lexer level so the parser receives
    the actual characters.

    Decomposition: Plain string, escaped newline, escaped tab, escaped
    quote, and empty string tested. Consistency: These cover the main
    escape sequences and edge cases.
    """

    def test_double_quoted(self):
        """R-LEX-02.01: Plain double-quoted string tokenizes as STRING."""
        ts = tok_types('"hello"')
        assert ts == [(TokenType.STRING, "hello")]

    def test_escaped_newline(self):
        """R-LEX-02.02: \\n escape produces actual newline character."""
        ts = tok_types(r'"line1\nline2"')
        assert ts[0] == (TokenType.STRING, "line1\nline2")

    def test_escaped_tab(self):
        """R-LEX-02.03: \\t escape produces actual tab character."""
        ts = tok_types(r'"col1\tcol2"')
        assert ts[0] == (TokenType.STRING, "col1\tcol2")

    def test_escaped_quote(self):
        """R-LEX-02.04: Doubled quotes inside string produce literal quote."""
        ts = tok_types('"say ""hi"""')
        assert ts[0] == (TokenType.STRING, 'say "hi"')

    def test_empty_string(self):
        """R-LEX-02.05: Empty double-quoted string tokenizes correctly."""
        ts = tok_types('""')
        assert ts == [(TokenType.STRING, "")]


class TestCharArrays:
    """R-LEX-03: The lexer SHALL tokenize single-quoted char arrays as CHAR
    tokens, handling escaped single quotes.

    Model-user argument: The engineer uses single-quoted char arrays for
    option strings (e.g., 'linear', 'spline') and text data. Apostrophes
    within char arrays use the doubled-quote escape (''it''''s'').

    Decomposition: Plain char, escaped quote, and empty char tested.
    Consistency: These cover the main char array patterns.
    """

    def test_single_quoted(self):
        """R-LEX-03.01: Single-quoted text tokenizes as CHAR."""
        ts = tok_types("'hello'")
        assert ts == [(TokenType.CHAR, "hello")]

    def test_escaped_single_quote(self):
        """R-LEX-03.02: Doubled single quote produces literal apostrophe."""
        ts = tok_types("'it''s'")
        assert ts == [(TokenType.CHAR, "it's")]

    def test_empty_char(self):
        """R-LEX-03.03: Empty single-quoted char array tokenizes correctly."""
        ts = tok_types("''")
        assert ts == [(TokenType.CHAR, "")]


class TestTransposeVsChar:
    """R-LEX-04: The lexer SHALL disambiguate the single-quote character as
    TRANSPOSE after identifiers, closing brackets, and numbers versus
    CHAR at expression-start or after operators.

    Model-user argument: MATLAB reuses the single quote for both transpose
    (``A'``) and char arrays (``'hello'``). The lexer must use context to
    decide: after an identifier or closing bracket it is transpose; after
    an operator or at start of expression it begins a char array. Getting
    this wrong breaks either transpose or string handling.

    Decomposition: Transpose after rparen, ident, rbracket, and number;
    char at start, after operator, and after comma tested. Consistency:
    These seven contexts cover all disambiguation cases.
    """

    def test_transpose_after_rparen(self):
        """R-LEX-04.01: Quote after closing paren is TRANSPOSE."""
        ts = tok_types("A(1)'")
        types = [t[0] for t in ts]
        assert TokenType.TRANSPOSE in types

    def test_transpose_after_ident(self):
        """R-LEX-04.02: Quote after identifier is TRANSPOSE."""
        ts = tok_types("A'")
        assert ts == [(TokenType.IDENT, "A"), (TokenType.TRANSPOSE, "'")]

    def test_char_at_start(self):
        """R-LEX-04.03: Quote after assignment operator starts a CHAR."""
        ts = tok_types("x = 'hello'")
        types = [t[0] for t in ts]
        assert TokenType.CHAR in types
        assert TokenType.TRANSPOSE not in types

    def test_transpose_after_rbracket(self):
        """R-LEX-04.04: Quote after closing bracket is TRANSPOSE."""
        ts = tok_types("[1 2]'")
        types = [t[0] for t in ts]
        assert TokenType.TRANSPOSE in types

    def test_char_after_operator(self):
        """R-LEX-04.05: Quote after binary operator starts a CHAR."""
        ts = tok_types("x + 'abc'")
        types = [t[0] for t in ts]
        assert TokenType.CHAR in types

    def test_transpose_after_number(self):
        """R-LEX-04.06: Quote after number is TRANSPOSE."""
        ts = tok_types("5'")
        assert ts[-1] == (TokenType.TRANSPOSE, "'")

    def test_char_after_comma(self):
        """R-LEX-04.07: Quote after comma in argument list starts a CHAR."""
        ts = tok_types("f(x, 'opt')")
        found_char = any(t[0] == TokenType.CHAR for t in ts)
        assert found_char


class TestIdentifiersKeywords:
    """R-LEX-05: The lexer SHALL tokenize identifiers as IDENT and reserved
    keywords as KEYWORD, distinguishing them by whole-word matching.

    Model-user argument: Variable names like ``iffy`` must not be confused
    with the keyword ``if``. The engineer expects arbitrary valid identifiers
    to be usable without colliding with language keywords.

    Decomposition: Simple identifier, underscore-prefixed identifier,
    keyword, all-keywords sweep, non-keyword prefix, and keyword in context
    tested. Consistency: These cover identifier recognition, keyword
    recognition, and the boundary between them.
    """

    def test_identifier(self):
        """R-LEX-05.01: Simple identifier tokenizes as IDENT."""
        ts = tok_types("myVar")
        assert ts == [(TokenType.IDENT, "myVar")]

    def test_underscore_ident(self):
        """R-LEX-05.02: Underscore-prefixed identifier tokenizes as IDENT."""
        ts = tok_types("_private")
        assert ts == [(TokenType.IDENT, "_private")]

    def test_keyword_if(self):
        """R-LEX-05.03: Reserved word 'if' tokenizes as KEYWORD."""
        ts = tok_types("if")
        assert ts == [(TokenType.KEYWORD, "if")]

    def test_all_keywords(self):
        """R-LEX-05.04: Every KEYWORDS entry tokenizes as KEYWORD."""
        for kw in KEYWORDS:
            ts = tok_types(kw)
            assert ts[0] == (TokenType.KEYWORD, kw), f"Failed for keyword: {kw}"

    def test_ident_not_keyword(self):
        """R-LEX-05.05: Identifier starting with keyword prefix is IDENT."""
        ts = tok_types("iffy")
        assert ts == [(TokenType.IDENT, "iffy")]

    def test_keyword_in_context(self):
        """R-LEX-05.06: Keyword followed by identifiers tokenizes correctly."""
        vals = tok_values("if x > 0")
        assert vals == ["if", "x", ">", "0"]


class TestOperators:
    """R-LEX-06: The lexer SHALL tokenize all arithmetic, element-wise,
    comparison, logical, and special operators with their correct token types.

    Model-user argument: The engineer uses a wide variety of operators:
    arithmetic (+, -, *, /), element-wise (.*, ./, .^), comparison (==, ~=,
    <, >, <=, >=), logical (&, |, &&, ||, ~, !), backslash for left-divide,
    power (^), colon (:), and at-sign (@). Each must map to the correct
    token type or expressions misbehave.

    Decomposition: Arithmetic, element-wise, dot-transpose, comparison,
    logical, backslash, power, colon, and at-sign operator groups tested.
    Consistency: These nine sub-tests cover the complete operator set.
    """

    def test_arithmetic(self):
        """R-LEX-06.01: Arithmetic operators tokenize with correct values."""
        vals = tok_values("a + b - c * d / e")
        assert vals == ["a", "+", "b", "-", "c", "*", "d", "/", "e"]

    def test_element_wise(self):
        """R-LEX-06.02: Element-wise operators .* ./ .^ produce correct types."""
        ts = tok_types("a .* b ./ c .^ d")
        ops = [t for t in ts if t[0] in (TokenType.DOT_TIMES, TokenType.DOT_RDIVIDE, TokenType.DOT_POWER)]
        assert len(ops) == 3

    def test_dot_transpose(self):
        """R-LEX-06.03: Dot-transpose .'' produces DOT_TRANSPOSE token."""
        ts = tok_types("A.'")
        assert ts == [(TokenType.IDENT, "A"), (TokenType.DOT_TRANSPOSE, ".'")]

    def test_comparison(self):
        """R-LEX-06.04: All comparison operators map to correct token types."""
        for op, tt in [("==", TokenType.EQ), ("~=", TokenType.NE), ("!=", TokenType.NE),
                       ("<", TokenType.LT), (">", TokenType.GT),
                       ("<=", TokenType.LE), (">=", TokenType.GE)]:
            ts = tok_types(f"a {op} b")
            assert any(t[0] == tt for t in ts), f"Failed for {op}"

    def test_logical(self):
        """R-LEX-06.05: All logical operators map to correct token types."""
        for op, tt in [("&", TokenType.AND), ("|", TokenType.OR),
                       ("&&", TokenType.SHORT_AND), ("||", TokenType.SHORT_OR),
                       ("~", TokenType.NOT), ("!", TokenType.NOT)]:
            ts = tok_types(f"{op}x" if op in ("~", "!") else f"a {op} b")
            assert any(t[0] == tt for t in ts), f"Failed for {op}"

    def test_backslash(self):
        """R-LEX-06.06: Backslash operator produces LDIVIDE token."""
        ts = tok_types("A \\ b")
        assert any(t[0] == TokenType.LDIVIDE for t in ts)

    def test_power(self):
        """R-LEX-06.07: Caret operator produces POWER token."""
        ts = tok_types("x ^ 2")
        assert any(t[0] == TokenType.POWER for t in ts)

    def test_colon(self):
        """R-LEX-06.08: Colon operator produces COLON token."""
        ts = tok_types("1:10")
        assert (TokenType.COLON, ":") in ts

    def test_at_sign(self):
        """R-LEX-06.09: At-sign produces AT token."""
        ts = tok_types("@sin")
        assert ts[0] == (TokenType.AT, "@")


class TestDelimiters:
    """R-LEX-07: The lexer SHALL tokenize delimiter characters (parentheses,
    brackets, braces, dots, commas, semicolons) with correct token types.

    Model-user argument: Delimiters structure every expression: parentheses
    for grouping and calls, brackets for matrices, braces for cells, dots
    for field access, commas for element separation, semicolons for row
    breaks. Each must be correctly identified.

    Decomposition: Parentheses, brackets with semicolons, braces with
    commas, and dot-field delimiter tested. Consistency: These cover all
    delimiter characters.
    """

    def test_parens(self):
        """R-LEX-07.01: Parentheses produce LPAREN and RPAREN tokens."""
        ts = tok_types("f(x)")
        types = [t[0] for t in ts]
        assert TokenType.LPAREN in types
        assert TokenType.RPAREN in types

    def test_brackets(self):
        """R-LEX-07.02: Brackets produce LBRACKET, RBRACKET, and SEMICOLON."""
        ts = tok_types("[1 2; 3 4]")
        types = [t[0] for t in ts]
        assert TokenType.LBRACKET in types
        assert TokenType.RBRACKET in types
        assert TokenType.SEMICOLON in types

    def test_braces(self):
        """R-LEX-07.03: Braces produce LBRACE, RBRACE, and COMMA."""
        ts = tok_types("{1, 'a'}")
        types = [t[0] for t in ts]
        assert TokenType.LBRACE in types
        assert TokenType.RBRACE in types
        assert TokenType.COMMA in types

    def test_dot_field(self):
        """R-LEX-07.04: Dot between identifiers produces DOT token."""
        ts = tok_types("s.field")
        assert ts == [(TokenType.IDENT, "s"), (TokenType.DOT, "."), (TokenType.IDENT, "field")]


class TestComments:
    """R-LEX-08: The lexer SHALL tokenize percent comments, hash comments
    (Octave), and block comments as COMMENT tokens.

    Model-user argument: The engineer documents code with ``% comments`` and
    may use Octave-style ``# comments``. Block comments (%{ ... %}) wrap
    multi-line explanations. Comments must be recognized so the parser can
    skip them.

    Decomposition: Percent comment, hash comment, block comment, and
    comment-only line tested. Consistency: These cover all comment syntax
    forms in MATLAB/Octave.
    """

    def test_percent_comment(self):
        """R-LEX-08.01: Percent comment is captured as COMMENT token."""
        ts = tok_types("x = 1 % this is a comment\n")
        comments = [t for t in ts if t[0] == TokenType.COMMENT]
        assert len(comments) == 1
        assert "this is a comment" in comments[0][1]

    def test_hash_comment(self):
        """R-LEX-08.02: Octave hash comment is captured as COMMENT token."""
        ts = tok_types("x = 1 # Octave comment\n")
        comments = [t for t in ts if t[0] == TokenType.COMMENT]
        assert len(comments) == 1

    def test_block_comment(self):
        """R-LEX-08.03: Block comment %{ ... %} is captured as single COMMENT."""
        src = "%{\nThis is a\nblock comment\n%}"
        ts = tok_types(src)
        comments = [t for t in ts if t[0] == TokenType.COMMENT]
        assert len(comments) == 1
        assert "block comment" in comments[0][1]

    def test_comment_only_line(self):
        """R-LEX-08.04: Line with only a comment produces one COMMENT token."""
        ts = tok_types("% just a comment")
        non_eof = [t for t in ts if t[0] != TokenType.EOF]
        assert len(non_eof) == 1
        assert non_eof[0][0] == TokenType.COMMENT


class TestLineContinuation:
    """R-LEX-09: The lexer SHALL consume ellipsis line continuations (...) and
    join the continued lines transparently.

    Model-user argument: The engineer breaks long expressions across lines
    with ``...`` for readability. The lexer must consume the ellipsis and
    newline so the parser sees a single continuous expression.

    Decomposition: Ellipsis consumption and token preservation tested.
    Consistency: These two tests confirm both that the ellipsis is consumed
    and that surrounding tokens survive.
    """

    def test_ellipsis_continuation(self):
        """R-LEX-09.01: Ellipsis and newline are consumed from token stream."""
        src = "x = 1 + ...\n  2"
        ts = tok_types(src)
        # The ... and newline should be consumed, giving: x = 1 + 2
        vals = [t[1] for t in ts if t[0] not in (TokenType.NEWLINE, TokenType.COMMENT)]
        assert "..." not in vals
        assert "2" in vals

    def test_ellipsis_preserves_tokens(self):
        """R-LEX-09.02: Tokens on either side of ellipsis are preserved."""
        src = "a + ...\nb"
        vals = tok_values(src)
        assert vals == ["a", "+", "b"]


class TestNewlines:
    """R-LEX-10: The lexer SHALL emit NEWLINE tokens for line breaks, handling
    both LF and CRLF line endings.

    Model-user argument: MATLAB scripts may originate on Windows (CRLF) or
    Unix (LF). The lexer must normalize both into single NEWLINE tokens so
    the parser sees consistent statement boundaries regardless of platform.

    Decomposition: LF newline and CRLF newline tested. Consistency: These
    are the two line-ending forms.
    """

    def test_newline_emitted(self):
        """R-LEX-10.01: LF produces a NEWLINE token."""
        ts = tok_types("a\nb")
        types = [t[0] for t in ts]
        assert TokenType.NEWLINE in types

    def test_cr_lf(self):
        """R-LEX-10.02: CRLF produces exactly one NEWLINE token."""
        ts = tok_types("a\r\nb")
        newlines = [t for t in ts if t[0] == TokenType.NEWLINE]
        assert len(newlines) == 1


class TestAssignment:
    """R-LEX-11: The lexer SHALL tokenize the assignment operator (=) as
    ASSIGN, distinct from comparison equality (==).

    Model-user argument: The single = is assignment; == is comparison. The
    lexer must emit ASSIGN for ``x = 5`` so the parser builds an assignment
    statement rather than a comparison.

    Decomposition: Single assignment statement tested. Consistency: One test
    suffices; the == case is covered in TestOperators.
    """

    def test_simple_assign(self):
        """R-LEX-11.01: Assignment operator tokenizes as ASSIGN."""
        ts = tok_types("x = 5")
        assert ts == [(TokenType.IDENT, "x"), (TokenType.ASSIGN, "="), (TokenType.NUMBER, "5")]


class TestLineNumbers:
    """R-LEX-12: The lexer SHALL track line numbers so that tokens on
    successive lines report incrementing line values.

    Model-user argument: Error messages reference line numbers. The lexer
    must track lines accurately so that parse errors point the engineer to
    the correct source location.

    Decomposition: Three tokens on three lines tested. Consistency: This
    verifies monotonic line tracking.
    """

    def test_line_tracking(self):
        """R-LEX-12.01: Tokens on lines 1, 2, 3 report matching line numbers."""
        src = "a\nb\nc"
        tokens = tokenize(src)
        a = [t for t in tokens if t.value == "a"][0]
        b = [t for t in tokens if t.value == "b"][0]
        c = [t for t in tokens if t.value == "c"][0]
        assert a.line == 1
        assert b.line == 2
        assert c.line == 3


class TestErrors:
    """R-LEX-13: The lexer SHALL raise LexerError for unterminated string and
    char array literals.

    Model-user argument: If the engineer forgets a closing quote, the lexer
    should raise a clear error rather than consuming the rest of the file as
    string content. Immediate error reporting prevents confusing downstream
    parse failures.

    Decomposition: Unterminated double-quoted string and unterminated
    single-quoted char tested. Consistency: These are the two quote types
    that can be unterminated.
    """

    def test_unterminated_string(self):
        """R-LEX-13.01: Unterminated double-quoted string raises LexerError."""
        with pytest.raises(LexerError):
            tokenize('"hello')

    def test_unterminated_char(self):
        """R-LEX-13.02: Unterminated single-quoted char raises LexerError."""
        with pytest.raises(LexerError):
            tokenize("'hello")


class TestComplexExpressions:
    """R-LEX-14: The lexer SHALL correctly tokenize complete MATLAB expressions
    including matrix literals, function calls, cell literals, anonymous
    functions, function definitions, if statements, for loops, and mixed
    operator expressions.

    Model-user argument: Real MATLAB code combines many token types in a
    single expression. The lexer must handle these compound expressions
    without misidentifying any token, since the parser depends on a correct
    token stream.

    Decomposition: Eight representative compound expressions tested.
    Consistency: These expressions collectively use every token type and
    exercise the interactions between them (e.g., transpose vs char in
    ``A' * (B ./ C)``).
    """

    def test_matrix_literal(self):
        """R-LEX-14.01: Matrix literal tokenizes with brackets and semicolons."""
        vals = tok_values("[1 2; 3 4]")
        assert vals == ["[", "1", "2", ";", "3", "4", "]"]

    def test_function_call(self):
        """R-LEX-14.02: Function call tokenizes with parens and comma."""
        vals = tok_values("f(x, y)")
        assert vals == ["f", "(", "x", ",", "y", ")"]

    def test_cell_literal(self):
        """R-LEX-14.03: Cell literal tokenizes with braces."""
        vals = tok_values("{1, 'a', [2 3]}")
        assert "1" in vals and "{" in vals and "}" in vals

    def test_anonymous_function(self):
        """R-LEX-14.04: Anonymous function tokenizes @, parens, and .^ operator."""
        vals = tok_values("@(x) x.^2")
        assert vals == ["@", "(", "x", ")", "x", ".^", "2"]

    def test_function_def(self):
        """R-LEX-14.05: Function definition header tokenizes all parts."""
        src = "function [a, b] = myFunc(x, y)"
        vals = tok_values(src)
        assert "function" in vals
        assert "myFunc" in vals
        assert "[" in vals

    def test_if_statement(self):
        """R-LEX-14.06: If statement tokenizes keyword and end keyword."""
        src = "if x > 0\n  y = 1;\nend"
        vals = tok_values(src)
        assert "if" in vals and "end" in vals

    def test_for_loop(self):
        """R-LEX-14.07: For loop tokenizes keyword, colon, and range bounds."""
        src = "for i = 1:10\n  x(i) = i^2;\nend"
        vals = tok_values(src)
        assert "for" in vals and "1" in vals and ":" in vals and "10" in vals

    def test_mixed_expression(self):
        """R-LEX-14.08: Mixed expression has TRANSPOSE and DOT_RDIVIDE tokens."""
        src = "result = A' * (B ./ C) + sin(x)"
        toks = tokenize(src)
        types = [t.type for t in toks]
        assert TokenType.TRANSPOSE in types
        assert TokenType.DOT_RDIVIDE in types
