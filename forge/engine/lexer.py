"""M-language lexer/tokenizer for Octave/MATLAB syntax."""
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional
import re


class TokenType(Enum):
    # Literals
    NUMBER = auto()       # 42, 3.14, 1e-5, 0xFF, 0b101
    STRING = auto()       # "hello" (string object)
    CHAR = auto()         # 'hello' (char array)

    # Identifiers and keywords
    IDENT = auto()        # variable/function names
    KEYWORD = auto()      # if, for, while, etc.

    # Operators
    PLUS = auto()         # +
    MINUS = auto()        # -
    TIMES = auto()        # *
    RDIVIDE = auto()      # /
    LDIVIDE = auto()      # \
    POWER = auto()        # ^
    DOT_TIMES = auto()    # .*
    DOT_RDIVIDE = auto()  # ./
    DOT_LDIVIDE = auto()  # .\
    DOT_POWER = auto()    # .^
    TRANSPOSE = auto()    # '  (postfix)
    DOT_TRANSPOSE = auto()  # .'

    # Comparison
    EQ = auto()           # ==
    NE = auto()           # ~=  or !=
    LT = auto()           # <
    GT = auto()           # >
    LE = auto()           # <=
    GE = auto()           # >=

    # Logical
    AND = auto()          # &
    OR = auto()           # |
    SHORT_AND = auto()    # &&
    SHORT_OR = auto()     # ||
    NOT = auto()          # ~ or !

    # Assignment
    ASSIGN = auto()       # =

    # Delimiters
    LPAREN = auto()       # (
    RPAREN = auto()       # )
    LBRACKET = auto()     # [
    RBRACKET = auto()     # ]
    LBRACE = auto()       # {
    RBRACE = auto()       # }
    COMMA = auto()        # ,
    SEMICOLON = auto()    # ;
    COLON = auto()        # :
    DOT = auto()          # . (field access)
    AT = auto()           # @
    ELLIPSIS = auto()     # ... (line continuation)

    # Special
    NEWLINE = auto()
    COMMENT = auto()
    EOF = auto()


KEYWORDS = {
    'if', 'else', 'elseif', 'end', 'endif', 'endfor', 'endwhile', 'endfunction',
    'endswitch', 'end_try_catch', 'endclassdef',
    'for', 'while', 'do', 'until',
    'switch', 'case', 'otherwise',
    'try', 'catch',
    'function', 'return', 'break', 'continue',
    'global', 'persistent',
    'classdef', 'properties', 'methods', 'events', 'enumeration',
    'parfor', 'spmd',
    'true', 'false',
}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.col})"


class LexerError(Exception):
    def __init__(self, msg, line, col):
        super().__init__(f"Line {line}, Col {col}: {msg}")
        self.line = line
        self.col = col


def _can_be_transpose(prev_token):
    """Determine if ' should be transpose (postfix) vs char string start.
    Transpose follows: ), ], }, identifier, number, .'
    Char literal follows everything else (start of line, operator, comma, etc).
    Keywords that expect an expression after them (case, return, if, etc.)
    should NOT trigger transpose — the ' starts a string.
    """
    if prev_token is None:
        return False
    if prev_token.type == TokenType.KEYWORD:
        # These keywords expect an expression — ' after them is a string
        _EXPR_KEYWORDS = {'case', 'return', 'if', 'elseif', 'while', 'until',
                          'switch', 'global', 'persistent', 'otherwise'}
        if prev_token.value in _EXPR_KEYWORDS:
            return False
        # 'end' can be transpose target (e.g., A(end)')
        return True
    return prev_token.type in (
        TokenType.RPAREN, TokenType.RBRACKET, TokenType.RBRACE,
        TokenType.IDENT, TokenType.NUMBER,
        TokenType.TRANSPOSE, TokenType.DOT_TRANSPOSE,
    )


class Lexer:
    """Tokenize M-language source code."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []
        self._prev_token: Optional[Token] = None

    def _peek(self, offset=0):
        p = self.pos + offset
        return self.source[p] if p < len(self.source) else '\0'

    def _advance(self):
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _match(self, expected):
        if self._peek() == expected:
            self._advance()
            return True
        return False

    def _emit(self, ttype, value):
        tok = Token(ttype, value, self.line, self.col - len(value))
        self.tokens.append(tok)
        self._prev_token = tok
        return tok

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source, return list of tokens."""
        self.tokens = []
        self._prev_token = None

        while self.pos < len(self.source):
            ch = self._peek()

            # Skip spaces and tabs (not newlines)
            if ch in (' ', '\t'):
                self._advance()
                continue

            # Line continuation ...
            if ch == '.' and self._peek(1) == '.' and self._peek(2) == '.':
                self._advance(); self._advance(); self._advance()
                # Skip to end of line
                while self.pos < len(self.source) and self._peek() != '\n':
                    self._advance()
                if self.pos < len(self.source):
                    self._advance()  # Skip newline
                continue

            # Newline
            if ch == '\n':
                self._advance()
                self._emit(TokenType.NEWLINE, '\n')
                continue

            if ch == '\r':
                self._advance()
                if self._peek() == '\n':
                    self._advance()
                self._emit(TokenType.NEWLINE, '\n')
                continue

            # Comments
            if ch == '%' or (ch == '#' and self._peek(1) != '!'):
                # Block comment %{ ... %}
                if ch == '%' and self._peek(1) == '{':
                    start_line = self.line
                    self._advance(); self._advance()  # skip %{
                    comment = ''
                    depth = 1
                    while self.pos < len(self.source) and depth > 0:
                        if self._peek() == '%' and self._peek(1) == '{':
                            depth += 1
                            comment += self._advance()
                            comment += self._advance()
                        elif self._peek() == '%' and self._peek(1) == '}':
                            depth -= 1
                            if depth > 0:
                                comment += self._advance()
                                comment += self._advance()
                            else:
                                self._advance(); self._advance()
                        else:
                            comment += self._advance()
                    self._emit(TokenType.COMMENT, comment)
                    continue
                # Line comment
                comment = ''
                self._advance()  # skip % or #
                while self.pos < len(self.source) and self._peek() != '\n':
                    comment += self._advance()
                self._emit(TokenType.COMMENT, comment.strip())
                continue

            # Numbers
            if ch.isdigit() or (ch == '.' and self._peek(1).isdigit()):
                self._read_number()
                continue

            # Hex/binary literals
            if ch == '0' and self._peek(1) in ('x', 'X', 'b', 'B'):
                self._read_number()
                continue

            # Strings (double-quoted)
            if ch == '"':
                self._read_string()
                continue

            # Char array or transpose
            if ch == "'":
                if _can_be_transpose(self._prev_token):
                    self._advance()
                    self._emit(TokenType.TRANSPOSE, "'")
                else:
                    self._read_char()
                continue

            # Identifiers and keywords
            if ch.isalpha() or ch == '_':
                self._read_ident()
                continue

            # Two-character operators
            if ch == '.' and self._peek(1) in ('*', '/', '\\', '^', "'"):
                self._advance()
                op = self._advance()
                op_map = {'*': TokenType.DOT_TIMES, '/': TokenType.DOT_RDIVIDE,
                          '\\': TokenType.DOT_LDIVIDE, '^': TokenType.DOT_POWER,
                          "'": TokenType.DOT_TRANSPOSE}
                self._emit(op_map[op], '.' + op)
                continue

            if ch == '=' and self._peek(1) == '=':
                self._advance(); self._advance()
                self._emit(TokenType.EQ, '==')
                continue

            if ch == '~' and self._peek(1) == '=':
                self._advance(); self._advance()
                self._emit(TokenType.NE, '~=')
                continue

            if ch == '!' and self._peek(1) == '=':
                self._advance(); self._advance()
                self._emit(TokenType.NE, '!=')
                continue

            if ch == '<' and self._peek(1) == '=':
                self._advance(); self._advance()
                self._emit(TokenType.LE, '<=')
                continue

            if ch == '>' and self._peek(1) == '=':
                self._advance(); self._advance()
                self._emit(TokenType.GE, '>=')
                continue

            if ch == '&' and self._peek(1) == '&':
                self._advance(); self._advance()
                self._emit(TokenType.SHORT_AND, '&&')
                continue

            if ch == '|' and self._peek(1) == '|':
                self._advance(); self._advance()
                self._emit(TokenType.SHORT_OR, '||')
                continue

            # Single-character operators and delimiters
            single_map = {
                '+': TokenType.PLUS, '-': TokenType.MINUS,
                '*': TokenType.TIMES, '/': TokenType.RDIVIDE,
                '\\': TokenType.LDIVIDE, '^': TokenType.POWER,
                '<': TokenType.LT, '>': TokenType.GT,
                '&': TokenType.AND, '|': TokenType.OR,
                '~': TokenType.NOT, '!': TokenType.NOT,
                '=': TokenType.ASSIGN,
                '(': TokenType.LPAREN, ')': TokenType.RPAREN,
                '[': TokenType.LBRACKET, ']': TokenType.RBRACKET,
                '{': TokenType.LBRACE, '}': TokenType.RBRACE,
                ',': TokenType.COMMA, ';': TokenType.SEMICOLON,
                ':': TokenType.COLON, '@': TokenType.AT,
            }

            if ch in single_map:
                self._advance()
                self._emit(single_map[ch], ch)
                continue

            if ch == '.':
                self._advance()
                self._emit(TokenType.DOT, '.')
                continue

            raise LexerError(f"Unexpected character: {ch!r}", self.line, self.col)

        self._emit(TokenType.EOF, '')
        return self.tokens

    def _read_number(self):
        """Read numeric literal: integer, float, hex, binary, scientific."""
        start = self.pos
        ch = self._peek()

        # Hex
        if ch == '0' and self._peek(1) in ('x', 'X'):
            self._advance(); self._advance()
            while self._peek() in '0123456789abcdefABCDEF_':
                self._advance()
            self._emit(TokenType.NUMBER, self.source[start:self.pos])
            return

        # Binary
        if ch == '0' and self._peek(1) in ('b', 'B'):
            self._advance(); self._advance()
            while self._peek() in '01_':
                self._advance()
            self._emit(TokenType.NUMBER, self.source[start:self.pos])
            return

        # Regular number (int or float)
        while self._peek().isdigit():
            self._advance()

        # Decimal point
        if self._peek() == '.' and self._peek(1) != '.' and not self._peek(1).isalpha() and self._peek(1) not in ('*', '/', '\\', '^', "'"):
            self._advance()
            while self._peek().isdigit():
                self._advance()

        # Scientific notation
        if self._peek() in ('e', 'E'):
            self._advance()
            if self._peek() in ('+', '-'):
                self._advance()
            while self._peek().isdigit():
                self._advance()

        # Imaginary suffix
        if self._peek() in ('i', 'j'):
            self._advance()

        self._emit(TokenType.NUMBER, self.source[start:self.pos])

    def _read_string(self):
        """Read double-quoted string literal."""
        self._advance()  # skip opening "
        value = ''
        while self.pos < len(self.source):
            ch = self._peek()
            if ch == '"':
                if self._peek(1) == '"':  # Escaped quote
                    value += '"'
                    self._advance(); self._advance()
                else:
                    self._advance()  # skip closing "
                    self._emit(TokenType.STRING, value)
                    return
            elif ch == '\\':
                self._advance()
                esc = self._advance()
                esc_map = {'n': '\n', 't': '\t', '\\': '\\', '"': '"', '0': '\0', 'r': '\r', 'b': '\x08', 'f': '\x0c', 'a': '\x07', 'v': '\x0b'}
                value += esc_map.get(esc, '\\' + esc)
            elif ch == '\n':
                raise LexerError("Unterminated string", self.line, self.col)
            else:
                value += self._advance()
        raise LexerError("Unterminated string at end of file", self.line, self.col)

    def _read_char(self):
        """Read single-quoted char array literal."""
        self._advance()  # skip opening '
        value = ''
        while self.pos < len(self.source):
            ch = self._peek()
            if ch == "'":
                if self._peek(1) == "'":  # Escaped quote: '' → '
                    value += "'"
                    self._advance(); self._advance()
                else:
                    self._advance()  # skip closing '
                    self._emit(TokenType.CHAR, value)
                    return
            elif ch == '\n':
                raise LexerError("Unterminated character array", self.line, self.col)
            else:
                value += self._advance()
        raise LexerError("Unterminated character array at end of file", self.line, self.col)

    def _read_ident(self):
        """Read identifier or keyword."""
        start = self.pos
        while self._peek().isalnum() or self._peek() == '_':
            self._advance()
        word = self.source[start:self.pos]
        if word in KEYWORDS:
            self._emit(TokenType.KEYWORD, word)
        else:
            self._emit(TokenType.IDENT, word)


def tokenize(source: str) -> List[Token]:
    """Convenience: tokenize source string."""
    return Lexer(source).tokenize()
