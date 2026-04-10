# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""M-language parser: tokens → AST.

Implements a Pratt (top-down operator precedence) parser for Octave/MATLAB expressions,
plus recursive descent for statements and function definitions.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Any, Union
from forge.engine.lexer import TokenType, Token, tokenize, LexerError


# ============================================================
# AST Node definitions
# ============================================================

@dataclass
class NumberLiteral:
    value: str  # Raw string; evaluated later to handle 0xFF, 3i, etc.

@dataclass
class StringLiteral:
    value: str
    is_char: bool = False  # True for 'single-quoted', False for "double-quoted"

@dataclass
class Identifier:
    name: str

@dataclass
class UnaryOp:
    op: str      # '+', '-', '~', '!'
    operand: Any

@dataclass
class BinaryOp:
    op: str      # '+', '-', '*', '/', '^', '.*', './', '.^', etc.
    left: Any
    right: Any

@dataclass
class CompareOp:
    op: str      # '==', '~=', '<', '>', '<=', '>='
    left: Any
    right: Any

@dataclass
class LogicalOp:
    op: str      # '&', '|', '&&', '||'
    left: Any
    right: Any

@dataclass
class TransposeOp:
    operand: Any
    conjugate: bool = True  # True for ', False for .'

@dataclass
class ColonExpr:
    start: Any
    stop: Any
    step: Optional[Any] = None  # start:step:stop or start:stop

@dataclass
class Index:
    """Parenthesis indexing: A(i,j) or function call f(x)."""
    target: Any
    args: List[Any]

@dataclass
class CellIndex:
    """Brace indexing: C{i,j}."""
    target: Any
    args: List[Any]

@dataclass
class FieldAccess:
    """Dot access: s.field."""
    target: Any
    field: str

@dataclass
class DynamicFieldAccess:
    """Dynamic field: s.(expr)."""
    target: Any
    field_expr: Any

@dataclass
class MatrixLiteral:
    """[a b; c d] — list of rows, each row is a list of expressions."""
    rows: List[List[Any]]

@dataclass
class CellLiteral:
    """{a, b; c, d} — like matrix but heterogeneous."""
    rows: List[List[Any]]

@dataclass
class FunctionHandle:
    """@funcname."""
    name: str

@dataclass
class AnonFunction:
    """@(args) expr."""
    args: List[str]
    body: Any

@dataclass
class EndKeyword:
    """The 'end' used in indexing context: A(1:end)."""
    pass

@dataclass
class BareColon:
    """Standalone : in indexing context: A(:, 2) means select all."""
    pass

@dataclass
class Assignment:
    targets: Any  # Identifier, Index, FieldAccess, or list for [a,b]=...
    value: Any
    suppress: bool = False

@dataclass
class IfStatement:
    condition: Any
    body: List[Any]
    elseifs: List[tuple]  # [(condition, body), ...]
    else_body: Optional[List[Any]] = None

@dataclass
class ForStatement:
    var: str
    iter_expr: Any
    body: List[Any]

@dataclass
class WhileStatement:
    condition: Any
    body: List[Any]

@dataclass
class DoUntilStatement:
    condition: Any
    body: List[Any]

@dataclass
class SwitchStatement:
    expr: Any
    cases: List[tuple]  # [(case_expr, body), ...]
    otherwise_body: Optional[List[Any]] = None

@dataclass
class TryCatchStatement:
    try_body: List[Any]
    catch_var: Optional[str] = None
    catch_body: Optional[List[Any]] = None

@dataclass
class ReturnStatement:
    pass

@dataclass
class BreakStatement:
    pass

@dataclass
class ContinueStatement:
    pass

@dataclass
class FunctionDef:
    name: str
    params: List[str]
    returns: List[str]
    body: List[Any]

@dataclass
class UnwindProtect:
    """unwind_protect ... unwind_protect_cleanup ... end"""
    try_body: list
    cleanup_body: list

@dataclass
class ClassDef:
    """classdef Name < SuperClass"""
    name: str
    superclasses: list  # list of str
    properties: dict    # {name: default_value}
    methods: list       # list of FunctionDef
    is_handle: bool = False

@dataclass
class ExpressionStatement:
    """Bare expression (possibly with ; to suppress output)."""
    expr: Any
    print_result: bool = True  # False if followed by ;

@dataclass
class CommandExpr:
    """Command-style call: cd dir → cd('dir')."""
    name: str
    args: List[str]

@dataclass
class GlobalStatement:
    names: List[str]

@dataclass
class PersistentStatement:
    names: List[str]


# ============================================================
# Operator precedence levels (higher = tighter binding)
# ============================================================
PREC_OR = 10
PREC_AND = 20
PREC_BIT_OR = 30
PREC_BIT_AND = 40
PREC_COMPARE = 50
PREC_COLON = 60
PREC_ADD = 70
PREC_MUL = 80
PREC_UNARY = 90
PREC_POWER = 100
PREC_POSTFIX = 110  # transpose


class ParseError(Exception):
    def __init__(self, msg, token=None):
        if token:
            super().__init__(f"Line {token.line}:{token.col}: {msg}")
        else:
            super().__init__(msg)
        self.token = token


class Parser:
    """Pratt parser for M-language."""

    def __init__(self, tokens: List[Token]):
        # Filter out comments
        self.tokens = [t for t in tokens if t.type != TokenType.COMMENT]
        self.pos = 0
        self._in_matrix = False
        self._paren_depth = 0

    def _peek(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF, '', 0, 0)

    def _advance(self) -> Token:
        tok = self._peek()
        self.pos += 1
        return tok

    def _expect(self, ttype, value=None) -> Token:
        tok = self._peek()
        if tok.type != ttype:
            raise ParseError(f"Expected {ttype.name}, got {tok.type.name} ({tok.value!r})", tok)
        if value is not None and tok.value != value:
            raise ParseError(f"Expected {value!r}, got {tok.value!r}", tok)
        return self._advance()

    def _match(self, ttype, value=None) -> Optional[Token]:
        tok = self._peek()
        if tok.type == ttype and (value is None or tok.value == value):
            return self._advance()
        return None

    def _at(self, ttype, value=None) -> bool:
        tok = self._peek()
        return tok.type == ttype and (value is None or tok.value == value)

    def _skip_newlines(self):
        while self._at(TokenType.NEWLINE):
            self._advance()

    def _at_end_of_statement(self):
        return (self._at(TokenType.NEWLINE) or self._at(TokenType.SEMICOLON)
                or self._at(TokenType.COMMA) or self._at(TokenType.EOF))

    def _is_end_keyword(self):
        """Check if current token is 'end' or end-variant."""
        return self._at(TokenType.KEYWORD) and self._peek().value in (
            'end', 'endif', 'endfor', 'endwhile', 'endfunction', 'endswitch', 'end_try_catch', 'endclassdef', 'end_unwind_protect')

    # ============================================================
    # Top-level parsing
    # ============================================================

    def parse(self) -> List[Any]:
        """Parse entire token stream into list of statements."""
        stmts = []
        self._skip_newlines()
        while not self._at(TokenType.EOF):
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
            self._skip_newlines()
        return stmts

    def parse_function(self) -> FunctionDef:
        """Parse a function definition."""
        return self._parse_function_def()

    # ============================================================
    # Statement parsing
    # ============================================================

    def _parse_statement(self) -> Optional[Any]:
        self._skip_newlines()
        tok = self._peek()

        if tok.type == TokenType.EOF:
            return None

        if tok.type == TokenType.KEYWORD:
            kw = tok.value
            if kw == 'function':
                return self._parse_function_def()
            elif kw == 'if':
                return self._parse_if()
            elif kw == 'for':
                return self._parse_for()
            elif kw == 'parfor':
                return self._parse_for()  # parfor = serial for without parallel toolbox
            elif kw == 'while':
                return self._parse_while()
            elif kw == 'do':
                return self._parse_do_until()
            elif kw == 'switch':
                return self._parse_switch()
            elif kw == 'try':
                return self._parse_try()
            elif kw == 'return':
                self._advance()
                self._consume_terminator()
                return ReturnStatement()
            elif kw == 'break':
                self._advance()
                self._consume_terminator()
                return BreakStatement()
            elif kw == 'continue':
                self._advance()
                self._consume_terminator()
                return ContinueStatement()
            elif kw == 'global':
                return self._parse_global()
            elif kw == 'persistent':
                return self._parse_persistent()

            elif kw == 'classdef':
                return self._parse_classdef()
            elif kw == 'unwind_protect':
                return self._parse_unwind_protect()
            elif kw == 'arguments':
                return self._parse_arguments_block()

        # Expression or assignment
        return self._parse_expr_or_assign()

    def _consume_terminator(self):
        """Consume optional ; or , or newline after statement.
        Comma and newline are non-suppressing; semicolon suppresses output."""
        if self._match(TokenType.SEMICOLON):
            pass
        elif self._match(TokenType.COMMA):
            pass  # comma is non-suppressing (like newline)
        elif self._match(TokenType.NEWLINE):
            pass

    def _parse_classdef(self):
        """Parse classdef Name [< SuperClass] ... end"""
        self._advance()  # consume 'classdef'
        # Check for (Handle)
        is_handle = False
        if self._at(TokenType.LPAREN):
            self._advance()
            if self._at(TokenType.IDENT) and self._peek().value == "Handle":
                self._advance()
                is_handle = True
            self._expect(TokenType.RPAREN)
        # Class name
        name = self._expect(TokenType.IDENT).value
        # Superclasses
        superclasses = []
        if self._at(TokenType.LT):
            self._advance()
            superclasses.append(self._expect(TokenType.IDENT).value)
            while self._at(TokenType.AND):
                self._advance()
                superclasses.append(self._expect(TokenType.IDENT).value)
        self._skip_newlines()
        # Parse blocks: properties, methods, events, enumeration
        properties = {}
        methods = []
        while not self._is_end_keyword():
            self._skip_newlines()
            if self._at(TokenType.EOF):
                break
            if self._at(TokenType.KEYWORD) and self._peek().value == "properties":
                self._advance()
                self._skip_newlines()
                # Properties block
                while not self._is_end_keyword():
                    self._skip_newlines()
                    if self._at(TokenType.EOF) or (self._at(TokenType.KEYWORD) and self._peek().value in ("methods", "events", "enumeration")):
                        break
                    if self._at(TokenType.IDENT):
                        prop_name = self._advance().value
                        default = None
                        if self._at(TokenType.ASSIGN):
                            self._advance()
                            default = self._parse_expression(0)
                        properties[prop_name] = default
                        self._match(TokenType.SEMICOLON)
                        self._match(TokenType.NEWLINE)
                    else:
                        self._advance()  # skip unknown
                if self._is_end_keyword():
                    self._advance()  # consume 'end'
                self._skip_newlines()
            elif self._at(TokenType.KEYWORD) and self._peek().value == "methods":
                self._advance()
                self._skip_newlines()
                while not self._is_end_keyword():
                    self._skip_newlines()
                    if self._at(TokenType.EOF) or (self._at(TokenType.KEYWORD) and self._peek().value in ("properties", "events", "enumeration")):
                        break
                    if self._at(TokenType.KEYWORD) and self._peek().value == "function":
                        methods.append(self._parse_function_def())
                    else:
                        self._advance()
                    self._skip_newlines()
                if self._is_end_keyword():
                    self._advance()  # consume 'end'
                self._skip_newlines()
            elif self._at(TokenType.KEYWORD) and self._peek().value in ("events", "enumeration"):
                self._advance()
                # Skip events/enumeration blocks
                while not self._is_end_keyword() and not self._at(TokenType.EOF):
                    self._advance()
                if self._is_end_keyword():
                    self._advance()
                self._skip_newlines()
            else:
                self._advance()
        if self._is_end_keyword():
            self._advance()
        return ClassDef(name, superclasses, properties, methods, is_handle)

    def _parse_unwind_protect(self):
        """Parse unwind_protect ... unwind_protect_cleanup ... end"""
        self._advance()  # consume 'unwind_protect'
        self._consume_terminator()
        try_body = self._parse_body(['unwind_protect_cleanup', 'end', 'end_unwind_protect'])
        cleanup_body = []
        if self._at(TokenType.KEYWORD) and self._peek().value == "unwind_protect_cleanup":
            self._advance()
            self._consume_terminator()
            cleanup_body = self._parse_body(['end', 'end_unwind_protect'])
        self._expect_end(['end', 'end_unwind_protect'])
        self._consume_terminator()
        return UnwindProtect(try_body, cleanup_body)


    def _parse_arguments_block(self):
        """Parse and skip MATLAB-style arguments...end validation blocks."""
        self._expect(TokenType.KEYWORD, 'arguments')
        self._consume_terminator()
        # Skip everything until we hit 'end'
        depth = 1
        while depth > 0:
            tok = self._peek()
            if tok.type == TokenType.EOF:
                break
            if tok.type == TokenType.KEYWORD and tok.value == 'end':
                depth -= 1
                if depth == 0:
                    self._advance()
                    self._consume_terminator()
                    break
            self._advance()
        # Return a no-op node — arguments blocks are just validation hints
        return None

    def _parse_expr_or_assign(self) -> Any:
        """Parse expression, checking for assignment."""
        # Check for [a, b] = ... (multi-return assignment)
        if self._at(TokenType.LBRACKET):
            saved = self.pos
            try:
                targets = self._parse_multi_assign_targets()
                if self._match(TokenType.ASSIGN):
                    value = self._parse_expression(0)
                    suppress = bool(self._match(TokenType.SEMICOLON))
                    if not suppress:
                        self._match(TokenType.COMMA)  # comma is non-suppressing separator
                    self._match(TokenType.NEWLINE)
                    return Assignment(targets, value, suppress)
                else:
                    # Not an assignment, backtrack
                    self.pos = saved
            except ParseError:
                self.pos = saved

        # Pre-check: path commands like "cd /path/to/dir" before expression parsing
        _PRE_PATH_CMDS = {"cd", "edit", "type", "help", "doc", "load", "save", "run", "source", "addpath", "rmpath", "diary", "which", "lookfor", "print"}
        if (self._at(TokenType.IDENT) and self._peek().value in _PRE_PATH_CMDS):
            _saved_pre = self.pos
            _cmd_name = self._peek().value
            _cmd_pos = self.pos
            self._advance()  # consume the command name
            _nxt = self._peek().type if self.pos < len(self.tokens) else TokenType.EOF
            if _nxt in (TokenType.RDIVIDE, TokenType.DOT, TokenType.NOT):
                # This looks like a path argument - consume rest of line
                parts = []
                while not self._at_end_of_statement() and self._peek().type != TokenType.EOF:
                    tok = self._advance()
                    parts.append(tok.value)
                path_str = "".join(parts).strip()
                cmd_ident = Identifier(_cmd_name)
                if path_str:
                    str_args = [StringLiteral(path_str, is_char=True)]
                else:
                    str_args = []
                expr = Index(cmd_ident, str_args)
                suppress = bool(self._match(TokenType.SEMICOLON))
                if not suppress:
                    self._match(TokenType.COMMA)  # comma is non-suppressing separator
                self._match(TokenType.NEWLINE)
                return ExpressionStatement(expr, not suppress)
            else:
                self.pos = _saved_pre  # backtrack

        # Pre-check: clearvars needs special handling because -except
        # would otherwise be parsed as binary minus by _parse_expression
        if (self._peek().type == TokenType.IDENT
                and self._peek().value == "clearvars"):
            _cv_pos = self.pos
            self._advance()  # consume "clearvars"
            str_args = []
            while not self._at_end_of_statement() and self._peek().type != TokenType.EOF:
                if self._peek().type == TokenType.MINUS:
                    self._advance()  # consume -
                    if self._peek().type == TokenType.IDENT:
                        flag = "-" + self._advance().value
                        str_args.append(StringLiteral(flag, is_char=True))
                elif self._peek().type == TokenType.IDENT:
                    str_args.append(StringLiteral(self._advance().value, is_char=True))
                else:
                    break
            expr = Index(Identifier("clearvars"), str_args)
            suppress = bool(self._match(TokenType.SEMICOLON))
            if not suppress:
                self._match(TokenType.COMMA)  # comma is non-suppressing separator
            self._match(TokenType.NEWLINE)
            return ExpressionStatement(expr, print_result=not suppress)

        expr = self._parse_expression(0)

        # Command-style syntax: hold on, axis equal, cd dir, etc.
        # If expr is an Identifier in the command set and next token is IDENT
        # (not =, not (, not operator), treat subsequent idents as string args.
        _CMD_STYLE_NAMES = {
            "hold", "axis", "format", "grid", "box", "legend",
            "colormap", "shading", "view", "cd", "type", "help",
            "doc", "edit", "dbstop", "dbclear", "dbcont",
            "who", "whos", "clear", "clearvars", "clc", "close", "figure",
            "load", "save", "diary", "more", "pkg", "addpath",
            "rmpath", "which", "lookfor", "run", "source",
            "exist", "methods", "properties", "print",
        }
        _PATH_COMMANDS = {"cd", "edit", "type", "help", "doc", "load",
                          "save", "run", "source", "addpath", "rmpath",
                          "diary", "which", "lookfor", "print"}
        _next_tt = self._peek().type
        _is_cmd_trigger = _next_tt == TokenType.IDENT
        # For path commands, also trigger on / ~ . (common path starts)
        if (not _is_cmd_trigger and isinstance(expr, Identifier)
                and expr.name in _PATH_COMMANDS
                and _next_tt in (TokenType.RDIVIDE, TokenType.LDIVIDE, TokenType.DOT, TokenType.NOT)):
            _is_cmd_trigger = True
        # For clearvars, also trigger on - (e.g. clearvars -except x)
        if (not _is_cmd_trigger and isinstance(expr, Identifier)
                and expr.name == "clearvars"
                and _next_tt == TokenType.MINUS):
            _is_cmd_trigger = True
        if (isinstance(expr, Identifier)
                and expr.name in _CMD_STYLE_NAMES
                and _is_cmd_trigger):
            if expr.name in _PATH_COMMANDS:
                # Consume all tokens until end of statement, join as single path string
                parts = []
                while not self._at_end_of_statement() and self._peek().type != TokenType.EOF:
                    tok = self._advance()
                    parts.append(tok.value)
                path_str = "".join(parts).strip()
                if path_str:
                    str_args = [StringLiteral(path_str, is_char=True)]
                else:
                    str_args = []
            else:
                # Regular command-style: collect identifiers as separate string args
                # Also handle -flag args (e.g. clearvars -except x)
                str_args = []
                while self._peek().type == TokenType.IDENT or (
                    self._peek().type == TokenType.MINUS and expr.name == "clearvars"
                ):
                    if self._peek().type == TokenType.MINUS:
                        self._advance()  # consume -
                        if self._peek().type == TokenType.IDENT:
                            flag = "-" + self._advance().value
                            str_args.append(StringLiteral(flag, is_char=True))
                        continue
                    str_args.append(StringLiteral(self._advance().value, is_char=True))
            expr = Index(expr, str_args)

        # Check for assignment: expr = value
        if self._at(TokenType.ASSIGN):
            if isinstance(expr, (Identifier, Index, FieldAccess, DynamicFieldAccess, CellIndex)):
                self._advance()
                value = self._parse_expression(0)
                suppress = bool(self._match(TokenType.SEMICOLON))
                if not suppress:
                    self._match(TokenType.COMMA)  # comma is non-suppressing separator
                self._match(TokenType.NEWLINE)
                return Assignment(expr, value, suppress)

        suppress = bool(self._match(TokenType.SEMICOLON))
        if not suppress:
            self._match(TokenType.COMMA)  # comma is non-suppressing separator
        self._match(TokenType.NEWLINE)
        return ExpressionStatement(expr, print_result=not suppress)

    def _parse_multi_assign_targets(self) -> List[Any]:
        """Parse [a, b, ~, c] for multi-return assignment. ~ = discard."""
        self._expect(TokenType.LBRACKET)
        targets = []
        while not self._at(TokenType.RBRACKET):
            if self._at(TokenType.NOT):  # ~ tilde = discard
                self._advance()
                targets.append(Identifier("~"))
            else:
                targets.append(Identifier(self._expect(TokenType.IDENT).value))
            if not self._match(TokenType.COMMA):
                break
        self._expect(TokenType.RBRACKET)
        return targets

    # ============================================================
    # Expression parsing (Pratt parser)
    # ============================================================

    def _parse_expression(self, min_prec: int) -> Any:
        """Parse expression with operator precedence."""
        left = self._parse_prefix()
        left = self._parse_postfix(left)

        while True:
            prec = self._infix_precedence()
            if prec <= min_prec:
                break
            left = self._parse_infix(left, prec)
            left = self._parse_postfix(left)

        return left

    def _parse_prefix(self) -> Any:
        """Parse prefix expression (atom or unary operator)."""
        tok = self._peek()

        # Unary operators
        if tok.type in (TokenType.PLUS, TokenType.MINUS):
            op = self._advance()
            operand = self._parse_expression(PREC_UNARY)
            if op.type == TokenType.PLUS:
                return operand  # Unary + is identity
            return UnaryOp('-', operand)

        if tok.type == TokenType.NOT:
            self._advance()
            operand = self._parse_expression(PREC_UNARY)
            return UnaryOp('~', operand)

        # Number
        if tok.type == TokenType.NUMBER:
            return NumberLiteral(self._advance().value)

        # String / char
        if tok.type == TokenType.STRING:
            return StringLiteral(self._advance().value, is_char=False)
        if tok.type == TokenType.CHAR:
            return StringLiteral(self._advance().value, is_char=True)

        # Identifier (or 'end' in indexing context)
        if tok.type == TokenType.IDENT:
            return Identifier(self._advance().value)

        if tok.type == TokenType.KEYWORD and tok.value == 'end':
            self._advance()
            return EndKeyword()

        if tok.type == TokenType.KEYWORD and tok.value in ('true', 'false'):
            return NumberLiteral(self._advance().value)

        # Parenthesized expression
        if tok.type == TokenType.LPAREN:
            self._advance()
            self._paren_depth += 1
            expr = self._parse_expression(0)
            self._expect(TokenType.RPAREN)
            self._paren_depth -= 1
            return expr

        # Matrix literal [...]
        if tok.type == TokenType.LBRACKET:
            return self._parse_matrix_literal()

        # Cell literal {...}
        if tok.type == TokenType.LBRACE:
            return self._parse_cell_literal()

        # Function handle @
        if tok.type == TokenType.AT:
            return self._parse_handle_or_anon()

        raise ParseError(f"Unexpected token: {tok.type.name} ({tok.value!r})", tok)

    def _parse_postfix(self, left: Any) -> Any:
        """Parse postfix: transpose, indexing, field access."""
        while True:
            tok = self._peek()

            # Transpose
            if tok.type == TokenType.TRANSPOSE:
                self._advance()
                left = TransposeOp(left, conjugate=True)
                continue

            if tok.type == TokenType.DOT_TRANSPOSE:
                self._advance()
                left = TransposeOp(left, conjugate=False)
                continue

            # Function call / indexing: A(...)
            if tok.type == TokenType.LPAREN and isinstance(left, (Identifier, Index, FieldAccess, CellIndex)):
                self._advance()
                args = self._parse_arg_list(TokenType.RPAREN)
                self._expect(TokenType.RPAREN)
                left = Index(left, args)
                continue

            # Cell indexing: C{...}
            if tok.type == TokenType.LBRACE and isinstance(left, (Identifier, Index, FieldAccess, CellIndex)):
                self._advance()
                args = self._parse_arg_list(TokenType.RBRACE)
                self._expect(TokenType.RBRACE)
                left = CellIndex(left, args)
                continue

            # Field access: s.field or s.(expr)
            if tok.type == TokenType.DOT and isinstance(left, (Identifier, Index, FieldAccess, CellIndex)):
                self._advance()
                if self._at(TokenType.LPAREN):
                    # Dynamic field: s.(expr)
                    self._advance()
                    field_expr = self._parse_expression(0)
                    self._expect(TokenType.RPAREN)
                    left = DynamicFieldAccess(left, field_expr)
                elif self._at(TokenType.IDENT):
                    field_name = self._advance().value
                    left = FieldAccess(left, field_name)
                else:
                    raise ParseError("Expected field name after '.'", self._peek())
                continue

            break

        return left

    def _infix_precedence(self) -> int:
        """Get precedence of current token as infix operator."""
        tok = self._peek()
        tt = tok.type

        if tt in (TokenType.SHORT_OR,):
            return PREC_OR
        if tt in (TokenType.SHORT_AND,):
            return PREC_AND
        if tt in (TokenType.OR,):
            return PREC_BIT_OR
        if tt in (TokenType.AND,):
            return PREC_BIT_AND
        if tt in (TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE):
            return PREC_COMPARE
        if tt == TokenType.COLON:
            return PREC_COLON
        if tt in (TokenType.PLUS, TokenType.MINUS):
            if self._in_matrix and self._paren_depth == 0 and self.pos > 0 and self.pos + 1 < len(self.tokens):
                cur_tok = self.tokens[self.pos]
                prev_tok = self.tokens[self.pos - 1]
                next_tok = self.tokens[self.pos + 1]
                prev_end = prev_tok.col + len(prev_tok.value)
                cur_end = cur_tok.col + len(cur_tok.value)
                space_before = cur_tok.col > prev_end
                space_after = next_tok.col > cur_end
                # MATLAB rule: space before + no space after = unary (separator)
                # space on both sides or no space = binary operator
                if space_before and not space_after:
                    return 0
            return PREC_ADD
        if tt in (TokenType.TIMES, TokenType.RDIVIDE, TokenType.LDIVIDE,
                  TokenType.DOT_TIMES, TokenType.DOT_RDIVIDE, TokenType.DOT_LDIVIDE):
            return PREC_MUL
        if tt in (TokenType.POWER, TokenType.DOT_POWER):
            return PREC_POWER

        return 0  # Not an infix operator

    def _parse_infix(self, left: Any, prec: int) -> Any:
        """Parse infix operator."""
        tok = self._advance()
        tt = tok.type

        # Left-associative for power (matches Octave: 2^3^2 = (2^3)^2 = 64)
        right_prec = prec

        # Colon is special: a:b or a:b:c
        if tt == TokenType.COLON:
            mid = self._parse_expression(PREC_COLON)
            if self._match(TokenType.COLON):
                # a:b:c → ColonExpr(start=a, stop=c, step=b)
                right = self._parse_expression(PREC_COLON)
                return ColonExpr(start=left, stop=right, step=mid)
            return ColonExpr(start=left, stop=mid)

        right = self._parse_expression(right_prec)

        # Categorize the operator
        op_str = tok.value
        if tt in (TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE):
            return CompareOp(op_str, left, right)
        if tt in (TokenType.SHORT_AND, TokenType.SHORT_OR, TokenType.AND, TokenType.OR):
            return LogicalOp(op_str, left, right)
        return BinaryOp(op_str, left, right)

    def _peek_at(self, offset: int):
        """Peek at token type at current position + offset."""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos].type
        return TokenType.EOF

    def _parse_arg_list(self, close_token: TokenType) -> List[Any]:
        """Parse comma-separated argument list."""
        args = []
        if self._at(close_token):
            return args
        # Handle bare colon as first arg: A(:, ...)
        if self._at(TokenType.COLON) and self._peek_at(1) in (TokenType.COMMA, close_token):
            self._advance()
            args.append(BareColon())
        else:
            args.append(self._parse_expression(0))
        while self._match(TokenType.COMMA):
            # Handle bare colon after comma: A(1, :)
            if self._at(TokenType.COLON) and self._peek_at(1) in (TokenType.COMMA, close_token):
                self._advance()
                args.append(BareColon())
            else:
                args.append(self._parse_expression(0))
        return args

    # ============================================================
    # Compound literals
    # ============================================================

    def _parse_matrix_literal(self) -> MatrixLiteral:
        saved_in_matrix = self._in_matrix
        self._in_matrix = True
        """Parse [a b c; d e f]."""
        self._expect(TokenType.LBRACKET)
        rows = []
        if not self._at(TokenType.RBRACKET):
            rows.append(self._parse_row(TokenType.RBRACKET))
            while self._match(TokenType.SEMICOLON) or self._match(TokenType.NEWLINE):
                self._skip_newlines()
                if self._at(TokenType.RBRACKET):
                    break
                rows.append(self._parse_row(TokenType.RBRACKET))
        self._expect(TokenType.RBRACKET)
        self._in_matrix = saved_in_matrix
        return MatrixLiteral(rows)

    def _parse_cell_literal(self) -> CellLiteral:
        saved_in_matrix = self._in_matrix
        self._in_matrix = True
        """Parse {a, b; c, d}."""
        self._expect(TokenType.LBRACE)
        rows = []
        if not self._at(TokenType.RBRACE):
            rows.append(self._parse_row(TokenType.RBRACE))
            while self._match(TokenType.SEMICOLON) or self._match(TokenType.NEWLINE):
                self._skip_newlines()
                if self._at(TokenType.RBRACE):
                    break
                rows.append(self._parse_row(TokenType.RBRACE))
        self._expect(TokenType.RBRACE)
        self._in_matrix = saved_in_matrix
        return CellLiteral(rows)

    def _parse_row(self, close_token: TokenType) -> List[Any]:
        """Parse a row of space/comma separated expressions."""
        elements = []
        while not self._at(close_token) and not self._at(TokenType.SEMICOLON) and not self._at(TokenType.NEWLINE) and not self._at(TokenType.EOF):
            elements.append(self._parse_expression(0))
            self._match(TokenType.COMMA)  # Optional comma between elements
        return elements

    # ============================================================
    # Function handle / anonymous function
    # ============================================================

    def _parse_handle_or_anon(self) -> Any:
        self._expect(TokenType.AT)
        if self._at(TokenType.LPAREN):
            # Anonymous: @(x, y) expr
            self._advance()
            params = []
            if not self._at(TokenType.RPAREN):
                params.append(self._expect(TokenType.IDENT).value)
                while self._match(TokenType.COMMA):
                    params.append(self._expect(TokenType.IDENT).value)
            self._expect(TokenType.RPAREN)
            body = self._parse_expression(0)
            return AnonFunction(params, body)
        else:
            # Handle: @funcname
            name = self._expect(TokenType.IDENT).value
            return FunctionHandle(name)

    # ============================================================
    # Control flow statements
    # ============================================================

    def _parse_if(self) -> IfStatement:
        self._expect(TokenType.KEYWORD, 'if')
        cond = self._parse_expression(0)
        self._consume_terminator()
        body = self._parse_body(['elseif', 'else', 'end', 'endif'])
        elseifs = []
        while self._at(TokenType.KEYWORD) and self._peek().value == 'elseif':
            self._advance()
            ec = self._parse_expression(0)
            self._consume_terminator()
            eb = self._parse_body(['elseif', 'else', 'end', 'endif'])
            elseifs.append((ec, eb))
        else_body = None
        if self._at(TokenType.KEYWORD) and self._peek().value == 'else':
            self._advance()
            self._consume_terminator()
            else_body = self._parse_body(['end', 'endif'])
        self._expect_end(['end', 'endif'])
        self._consume_terminator()
        return IfStatement(cond, body, elseifs, else_body)

    def _parse_for(self) -> ForStatement:
        tok = self._advance()  # consume 'for' or 'parfor'
        var = self._expect(TokenType.IDENT).value
        self._expect(TokenType.ASSIGN)
        iter_expr = self._parse_expression(0)
        self._consume_terminator()
        body = self._parse_body(['end', 'endfor'])
        self._expect_end(['end', 'endfor'])
        self._consume_terminator()
        return ForStatement(var, iter_expr, body)

    def _parse_while(self) -> WhileStatement:
        self._expect(TokenType.KEYWORD, 'while')
        cond = self._parse_expression(0)
        self._consume_terminator()
        body = self._parse_body(['end', 'endwhile'])
        self._expect_end(['end', 'endwhile'])
        self._consume_terminator()
        return WhileStatement(cond, body)

    def _parse_do_until(self) -> DoUntilStatement:
        self._expect(TokenType.KEYWORD, 'do')
        self._consume_terminator()
        body = self._parse_body(['until'])
        self._expect(TokenType.KEYWORD, 'until')
        cond = self._parse_expression(0)
        self._consume_terminator()
        return DoUntilStatement(cond, body)

    def _parse_switch(self) -> SwitchStatement:
        self._expect(TokenType.KEYWORD, 'switch')
        expr = self._parse_expression(0)
        self._consume_terminator()
        self._skip_newlines()
        cases = []
        otherwise_body = None
        while self._at(TokenType.KEYWORD) and self._peek().value == 'case':
            self._advance()
            case_expr = self._parse_expression(0)
            self._consume_terminator()
            case_body = self._parse_body(['case', 'otherwise', 'end', 'endswitch'])
            cases.append((case_expr, case_body))
        if self._at(TokenType.KEYWORD) and self._peek().value == 'otherwise':
            self._advance()
            self._consume_terminator()
            otherwise_body = self._parse_body(['end', 'endswitch'])
        self._expect_end(['end', 'endswitch'])
        self._consume_terminator()
        return SwitchStatement(expr, cases, otherwise_body)

    def _parse_try(self) -> TryCatchStatement:
        self._expect(TokenType.KEYWORD, 'try')
        self._consume_terminator()
        try_body = self._parse_body(['catch', 'end', 'end_try_catch'])
        catch_var = None
        catch_body = None
        if self._at(TokenType.KEYWORD) and self._peek().value == 'catch':
            self._advance()
            if self._at(TokenType.IDENT):
                catch_var = self._advance().value
            self._consume_terminator()
            catch_body = self._parse_body(['end', 'end_try_catch'])
        self._expect_end(['end', 'end_try_catch'])
        self._consume_terminator()
        return TryCatchStatement(try_body, catch_var, catch_body)

    def _parse_function_def(self) -> FunctionDef:
        self._expect(TokenType.KEYWORD, 'function')
        returns = []
        # Check for return values: function [a, b] = name(...) or function a = name(...)
        saved = self.pos
        try:
            if self._at(TokenType.LBRACKET):
                self._advance()
                while not self._at(TokenType.RBRACKET):
                    returns.append(self._expect(TokenType.IDENT).value)
                    if not self._match(TokenType.COMMA):
                        break
                self._expect(TokenType.RBRACKET)
                self._expect(TokenType.ASSIGN)
            elif self._at(TokenType.IDENT):
                # Could be: function name(...) or function ret = name(...)
                first = self._advance()
                if self._match(TokenType.ASSIGN):
                    returns = [first.value]
                else:
                    # No return, 'first' is the function name
                    name = first.value
                    params = []
                    if self._match(TokenType.LPAREN):
                        if not self._at(TokenType.RPAREN):
                            params.append(self._expect(TokenType.IDENT).value)
                            while self._match(TokenType.COMMA):
                                params.append(self._expect(TokenType.IDENT).value)
                        self._expect(TokenType.RPAREN)
                    self._consume_terminator()
                    body = self._parse_body(['end', 'endfunction'])
                    if self._at(TokenType.KEYWORD) and self._peek().value in ('end', 'endfunction'):
                        self._advance()
                        self._consume_terminator()
                    return FunctionDef(name, params, returns, body)
        except ParseError:
            self.pos = saved
            returns = []

        name = self._expect(TokenType.IDENT).value
        params = []
        if self._match(TokenType.LPAREN):
            if not self._at(TokenType.RPAREN):
                params.append(self._expect(TokenType.IDENT).value)
                while self._match(TokenType.COMMA):
                    params.append(self._expect(TokenType.IDENT).value)
            self._expect(TokenType.RPAREN)
        self._consume_terminator()
        body = self._parse_body(['end', 'endfunction'])
        if self._at(TokenType.KEYWORD) and self._peek().value in ('end', 'endfunction'):
            self._advance()
            self._consume_terminator()
        return FunctionDef(name, params, returns, body)

    def _parse_global(self) -> GlobalStatement:
        self._expect(TokenType.KEYWORD, 'global')
        names = []
        while self._at(TokenType.IDENT):
            names.append(self._advance().value)
        self._consume_terminator()
        return GlobalStatement(names)

    def _parse_persistent(self) -> PersistentStatement:
        self._expect(TokenType.KEYWORD, 'persistent')
        names = []
        while self._at(TokenType.IDENT):
            names.append(self._advance().value)
        self._consume_terminator()
        return PersistentStatement(names)

    # ============================================================
    # Helpers
    # ============================================================

    def _parse_body(self, terminators: List[str]) -> List[Any]:
        """Parse statements until we see a keyword in terminators."""
        stmts = []
        while True:
            self._skip_newlines()
            if self._at(TokenType.EOF):
                break
            if self._at(TokenType.KEYWORD) and self._peek().value in terminators:
                break
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    def _expect_end(self, variants: List[str]):
        """Expect 'end' or variant keyword."""
        tok = self._peek()
        if tok.type == TokenType.KEYWORD and tok.value in variants:
            self._advance()
        else:
            raise ParseError(f"Expected {' or '.join(variants)}, got {tok.value!r}", tok)


def parse(source: str) -> List[Any]:
    """Parse M-language source code into AST."""
    tokens = tokenize(source)
    return Parser(tokens).parse()
