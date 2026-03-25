"""Tests for M-language lexer. Stage 2.1 from V&V plan."""
import pytest
from forge.engine.lexer import tokenize, TokenType, LexerError, KEYWORDS


def tok_types(src):
    """Return list of (type, value) tuples, excluding EOF."""
    return [(t.type, t.value) for t in tokenize(src) if t.type != TokenType.EOF]


def tok_values(src):
    """Return list of values, excluding EOF and NEWLINE."""
    return [t.value for t in tokenize(src) if t.type not in (TokenType.EOF, TokenType.NEWLINE)]


class TestNumbers:
    def test_integer(self):
        ts = tok_types("42")
        assert ts == [(TokenType.NUMBER, "42")]

    def test_float(self):
        ts = tok_types("3.14")
        assert ts == [(TokenType.NUMBER, "3.14")]

    def test_scientific(self):
        ts = tok_types("1e-5")
        assert ts == [(TokenType.NUMBER, "1e-5")]

    def test_scientific_positive(self):
        ts = tok_types("2.5E+3")
        assert ts == [(TokenType.NUMBER, "2.5E+3")]

    def test_hex(self):
        ts = tok_types("0xFF")
        assert ts == [(TokenType.NUMBER, "0xFF")]

    def test_binary(self):
        ts = tok_types("0b1010")
        assert ts == [(TokenType.NUMBER, "0b1010")]

    def test_imaginary(self):
        ts = tok_types("3.5i")
        assert ts == [(TokenType.NUMBER, "3.5i")]

    def test_imaginary_j(self):
        ts = tok_types("2j")
        assert ts == [(TokenType.NUMBER, "2j")]

    def test_leading_dot(self):
        ts = tok_types(".5")
        assert ts == [(TokenType.NUMBER, ".5")]

    def test_multiple_numbers(self):
        vals = tok_values("1 2.0 3e4")
        assert vals == ["1", "2.0", "3e4"]


class TestStrings:
    def test_double_quoted(self):
        ts = tok_types('"hello"')
        assert ts == [(TokenType.STRING, "hello")]

    def test_escaped_newline(self):
        ts = tok_types(r'"line1\nline2"')
        assert ts[0] == (TokenType.STRING, "line1\nline2")

    def test_escaped_tab(self):
        ts = tok_types(r'"col1\tcol2"')
        assert ts[0] == (TokenType.STRING, "col1\tcol2")

    def test_escaped_quote(self):
        ts = tok_types('"say ""hi"""')
        assert ts[0] == (TokenType.STRING, 'say "hi"')

    def test_empty_string(self):
        ts = tok_types('""')
        assert ts == [(TokenType.STRING, "")]


class TestCharArrays:
    def test_single_quoted(self):
        ts = tok_types("'hello'")
        assert ts == [(TokenType.CHAR, "hello")]

    def test_escaped_single_quote(self):
        ts = tok_types("'it''s'")
        assert ts == [(TokenType.CHAR, "it's")]

    def test_empty_char(self):
        ts = tok_types("''")
        assert ts == [(TokenType.CHAR, "")]


class TestTransposeVsChar:
    def test_transpose_after_rparen(self):
        ts = tok_types("A(1)'")
        types = [t[0] for t in ts]
        assert TokenType.TRANSPOSE in types

    def test_transpose_after_ident(self):
        ts = tok_types("A'")
        assert ts == [(TokenType.IDENT, "A"), (TokenType.TRANSPOSE, "'")]

    def test_char_at_start(self):
        ts = tok_types("x = 'hello'")
        types = [t[0] for t in ts]
        assert TokenType.CHAR in types
        assert TokenType.TRANSPOSE not in types

    def test_transpose_after_rbracket(self):
        ts = tok_types("[1 2]'")
        types = [t[0] for t in ts]
        assert TokenType.TRANSPOSE in types

    def test_char_after_operator(self):
        ts = tok_types("x + 'abc'")
        types = [t[0] for t in ts]
        assert TokenType.CHAR in types

    def test_transpose_after_number(self):
        ts = tok_types("5'")
        assert ts[-1] == (TokenType.TRANSPOSE, "'")

    def test_char_after_comma(self):
        ts = tok_types("f(x, 'opt')")
        found_char = any(t[0] == TokenType.CHAR for t in ts)
        assert found_char


class TestIdentifiersKeywords:
    def test_identifier(self):
        ts = tok_types("myVar")
        assert ts == [(TokenType.IDENT, "myVar")]

    def test_underscore_ident(self):
        ts = tok_types("_private")
        assert ts == [(TokenType.IDENT, "_private")]

    def test_keyword_if(self):
        ts = tok_types("if")
        assert ts == [(TokenType.KEYWORD, "if")]

    def test_all_keywords(self):
        for kw in KEYWORDS:
            ts = tok_types(kw)
            assert ts[0] == (TokenType.KEYWORD, kw), f"Failed for keyword: {kw}"

    def test_ident_not_keyword(self):
        ts = tok_types("iffy")
        assert ts == [(TokenType.IDENT, "iffy")]

    def test_keyword_in_context(self):
        vals = tok_values("if x > 0")
        assert vals == ["if", "x", ">", "0"]


class TestOperators:
    def test_arithmetic(self):
        vals = tok_values("a + b - c * d / e")
        assert vals == ["a", "+", "b", "-", "c", "*", "d", "/", "e"]

    def test_element_wise(self):
        ts = tok_types("a .* b ./ c .^ d")
        ops = [t for t in ts if t[0] in (TokenType.DOT_TIMES, TokenType.DOT_RDIVIDE, TokenType.DOT_POWER)]
        assert len(ops) == 3

    def test_dot_transpose(self):
        ts = tok_types("A.'")
        assert ts == [(TokenType.IDENT, "A"), (TokenType.DOT_TRANSPOSE, ".'")]

    def test_comparison(self):
        for op, tt in [("==", TokenType.EQ), ("~=", TokenType.NE), ("!=", TokenType.NE),
                       ("<", TokenType.LT), (">", TokenType.GT),
                       ("<=", TokenType.LE), (">=", TokenType.GE)]:
            ts = tok_types(f"a {op} b")
            assert any(t[0] == tt for t in ts), f"Failed for {op}"

    def test_logical(self):
        for op, tt in [("&", TokenType.AND), ("|", TokenType.OR),
                       ("&&", TokenType.SHORT_AND), ("||", TokenType.SHORT_OR),
                       ("~", TokenType.NOT), ("!", TokenType.NOT)]:
            ts = tok_types(f"{op}x" if op in ("~", "!") else f"a {op} b")
            assert any(t[0] == tt for t in ts), f"Failed for {op}"

    def test_backslash(self):
        ts = tok_types("A \\ b")
        assert any(t[0] == TokenType.LDIVIDE for t in ts)

    def test_power(self):
        ts = tok_types("x ^ 2")
        assert any(t[0] == TokenType.POWER for t in ts)

    def test_colon(self):
        ts = tok_types("1:10")
        assert (TokenType.COLON, ":") in ts

    def test_at_sign(self):
        ts = tok_types("@sin")
        assert ts[0] == (TokenType.AT, "@")


class TestDelimiters:
    def test_parens(self):
        ts = tok_types("f(x)")
        types = [t[0] for t in ts]
        assert TokenType.LPAREN in types
        assert TokenType.RPAREN in types

    def test_brackets(self):
        ts = tok_types("[1 2; 3 4]")
        types = [t[0] for t in ts]
        assert TokenType.LBRACKET in types
        assert TokenType.RBRACKET in types
        assert TokenType.SEMICOLON in types

    def test_braces(self):
        ts = tok_types("{1, 'a'}")
        types = [t[0] for t in ts]
        assert TokenType.LBRACE in types
        assert TokenType.RBRACE in types
        assert TokenType.COMMA in types

    def test_dot_field(self):
        ts = tok_types("s.field")
        assert ts == [(TokenType.IDENT, "s"), (TokenType.DOT, "."), (TokenType.IDENT, "field")]


class TestComments:
    def test_percent_comment(self):
        ts = tok_types("x = 1 % this is a comment\n")
        comments = [t for t in ts if t[0] == TokenType.COMMENT]
        assert len(comments) == 1
        assert "this is a comment" in comments[0][1]

    def test_hash_comment(self):
        ts = tok_types("x = 1 # Octave comment\n")
        comments = [t for t in ts if t[0] == TokenType.COMMENT]
        assert len(comments) == 1

    def test_block_comment(self):
        src = "%{\nThis is a\nblock comment\n%}"
        ts = tok_types(src)
        comments = [t for t in ts if t[0] == TokenType.COMMENT]
        assert len(comments) == 1
        assert "block comment" in comments[0][1]

    def test_comment_only_line(self):
        ts = tok_types("% just a comment")
        non_eof = [t for t in ts if t[0] != TokenType.EOF]
        assert len(non_eof) == 1
        assert non_eof[0][0] == TokenType.COMMENT


class TestLineContinuation:
    def test_ellipsis_continuation(self):
        src = "x = 1 + ...\n  2"
        ts = tok_types(src)
        # The ... and newline should be consumed, giving: x = 1 + 2
        vals = [t[1] for t in ts if t[0] not in (TokenType.NEWLINE, TokenType.COMMENT)]
        assert "..." not in vals
        assert "2" in vals

    def test_ellipsis_preserves_tokens(self):
        src = "a + ...\nb"
        vals = tok_values(src)
        assert vals == ["a", "+", "b"]


class TestNewlines:
    def test_newline_emitted(self):
        ts = tok_types("a\nb")
        types = [t[0] for t in ts]
        assert TokenType.NEWLINE in types

    def test_cr_lf(self):
        ts = tok_types("a\r\nb")
        newlines = [t for t in ts if t[0] == TokenType.NEWLINE]
        assert len(newlines) == 1


class TestAssignment:
    def test_simple_assign(self):
        ts = tok_types("x = 5")
        assert ts == [(TokenType.IDENT, "x"), (TokenType.ASSIGN, "="), (TokenType.NUMBER, "5")]


class TestLineNumbers:
    def test_line_tracking(self):
        src = "a\nb\nc"
        tokens = tokenize(src)
        a = [t for t in tokens if t.value == "a"][0]
        b = [t for t in tokens if t.value == "b"][0]
        c = [t for t in tokens if t.value == "c"][0]
        assert a.line == 1
        assert b.line == 2
        assert c.line == 3


class TestErrors:
    def test_unterminated_string(self):
        with pytest.raises(LexerError):
            tokenize('"hello')

    def test_unterminated_char(self):
        with pytest.raises(LexerError):
            tokenize("'hello")


class TestComplexExpressions:
    def test_matrix_literal(self):
        vals = tok_values("[1 2; 3 4]")
        assert vals == ["[", "1", "2", ";", "3", "4", "]"]

    def test_function_call(self):
        vals = tok_values("f(x, y)")
        assert vals == ["f", "(", "x", ",", "y", ")"]

    def test_cell_literal(self):
        vals = tok_values("{1, 'a', [2 3]}")
        assert "1" in vals and "{" in vals and "}" in vals

    def test_anonymous_function(self):
        vals = tok_values("@(x) x.^2")
        assert vals == ["@", "(", "x", ")", "x", ".^", "2"]

    def test_function_def(self):
        src = "function [a, b] = myFunc(x, y)"
        vals = tok_values(src)
        assert "function" in vals
        assert "myFunc" in vals
        assert "[" in vals

    def test_if_statement(self):
        src = "if x > 0\n  y = 1;\nend"
        vals = tok_values(src)
        assert "if" in vals and "end" in vals

    def test_for_loop(self):
        src = "for i = 1:10\n  x(i) = i^2;\nend"
        vals = tok_values(src)
        assert "for" in vals and "1" in vals and ":" in vals and "10" in vals

    def test_mixed_expression(self):
        src = "result = A' * (B ./ C) + sin(x)"
        toks = tokenize(src)
        types = [t.type for t in toks]
        assert TokenType.TRANSPOSE in types
        assert TokenType.DOT_RDIVIDE in types
