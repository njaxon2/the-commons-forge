# Forge IDE — Architecture Design Document (ADD)

**Document ID:** FORGE-ADD-001
**Version:** 1.0
**Date:** 2026-03-19
**Status:** Draft
**Parent:** FORGE-SRS-001
**Author:** Development Team

---

## 1. Purpose

This document describes the software architecture of Forge IDE, explaining the major components, their interfaces, data flows, and the rationale behind key design decisions. Each architectural component traces to system requirements in FORGE-SRS-001.

## 2. Architectural Overview

Forge is a Python monolith — a single process launched via `python -m forge` that contains the GUI, interpreter engine, and all toolbox functions. There is no client-server split, no IPC, and no network ports. Each launch is a fully independent instance.

```
┌─────────────────────────────────────────────────────────────┐
│                     forge (Python process)                   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    PySide6 GUI Layer                  │   │
│  │  ┌──────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │ Command  │ │ Editor │ │Workspace │ │  File    │  │   │
│  │  │ Widget   │ │ Widget │ │ Browser  │ │ Browser  │  │   │
│  │  └────┬─────┘ └───┬────┘ └────┬─────┘ └──────────┘  │   │
│  │       │           │           │                      │   │
│  │  ┌────┴───────────┴───────────┴──────────────────┐   │   │
│  │  │              Plot System (Matplotlib)          │   │   │
│  │  └───────────────────────────────────────────────┘   │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │ eval(source) → result             │
│  ┌──────────────────────┴───────────────────────────────┐   │
│  │                    Engine Layer                       │   │
│  │  ┌────────┐  ┌────────┐  ┌───────────┐              │   │
│  │  │ Lexer  │→│ Parser │→│ Evaluator │              │   │
│  │  └────────┘  └────────┘  └─────┬─────┘              │   │
│  │                                │                     │   │
│  │  ┌─────────────────────────────┴──────────────────┐  │   │
│  │  │              Session (Workspace + State)        │  │   │
│  │  └─────────────────────────────┬──────────────────┘  │   │
│  │                                │                     │   │
│  │  ┌─────────────────────────────┴──────────────────┐  │   │
│  │  │           Built-in Functions + Toolboxes        │  │   │
│  │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐         │  │   │
│  │  │  │ Math │ │ LinA │ │ Sig  │ │ Img  │ ...     │  │   │
│  │  │  └──────┘ └──────┘ └──────┘ └──────┘         │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Validation Layer (OQE + V&V)            │   │
│  │  ┌──────────┐  ┌───────────┐  ┌─────────────────┐   │   │
│  │  │ OQE DB   │  │ Tolerance │  │ Report Generator│   │   │
│  │  │ (SQLite) │  │ Framework │  │                 │   │   │
│  │  └──────────┘  └───────────┘  └─────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 3. Design Decisions

### DD-001: Python Monolith (no IPC)
**Decision:** Single Python process, no client-server architecture.
**Traces to:** UR-014, SR-400
**Rationale:** The previous Tauri/React architecture suffered from frontend-backend disconnection — the IPC layer added latency, complexity, and made testing difficult. A monolith eliminates these issues. Python instances are lightweight (~50MB idle), so multiple concurrent instances are practical.
**Alternatives Considered:**
- Electron + Python backend (rejected: heavy, IPC complexity)
- Web-based with FastAPI (rejected: can't test GUI programmatically, port conflicts)
**Risk:** Monolith limits future web deployment. Accepted — desktop is the primary target.

### DD-002: PySide6 for GUI
**Decision:** Use PySide6 (Qt6 for Python) for all GUI components.
**Traces to:** UR-017, SR-400-407
**Rationale:** Qt provides native look and feel on all platforms, rich widget set (dockable panels, tree views, table views, syntax highlighting), and is testable via pytest-qt. PySide6 is LGPL-licensed.
**Alternatives Considered:**
- Tkinter (rejected: limited widgets, poor aesthetics)
- wxPython (rejected: smaller community, fewer features)
- PyQt6 (rejected: GPL licensing)

### DD-003: NumPy as Array Backend
**Decision:** ForgeArray wraps numpy.ndarray, adding 1-based indexing and Octave semantics.
**Traces to:** SR-200, SR-201, UR-013
**Rationale:** NumPy provides optimized BLAS/LAPACK-backed array operations. Wrapping rather than reimplementing gives us C-speed computation. The 1-based indexing adapter converts at the API boundary.
**Trade-off:** Thin wrapper adds method call overhead (~microseconds per index). Acceptable for interactive use; hot loops should use vectorized operations.

### DD-004: Pratt Parser for M-Language
**Decision:** Top-down operator precedence (Pratt) parser with recursive descent for statements.
**Traces to:** SR-100, SR-101, SR-102
**Rationale:** Pratt parsing handles operator precedence naturally without a grammar table, is easy to extend with new operators, and handles the transpose-vs-char ambiguity through the `_can_be_transpose` heuristic. Recursive descent for statements maps directly to control flow constructs.
**Alternatives Considered:**
- PLY/Lark grammar-based parser (rejected: harder to handle M-language ambiguities)
- Tree-sitter (rejected: C dependency, harder to customize for Octave quirks)

### DD-005: Session-Based Execution Model
**Decision:** Each interpreter instance is a `Session` object holding workspace, function registry, and output buffer.
**Traces to:** SR-103-105, UR-014
**Rationale:** Session encapsulation enables multiple independent instances (each with own workspace), clean testing (fresh session per test), and future support for parallel sessions.

### DD-006: SciPy for Toolbox Backends
**Decision:** Toolbox functions delegate to scipy.signal, scipy.linalg, scipy.optimize, scipy.integrate, etc.
**Traces to:** SR-503-508
**Rationale:** SciPy provides battle-tested, peer-reviewed implementations of numerical algorithms. Wrapping SciPy functions with Octave-compatible interfaces gives correctness with minimal effort.
**Risk:** SciPy API changes could break wrappers. Mitigated by pinning scipy version in requirements.

### DD-007: SQLite for OQE Telemetry
**Decision:** OQE observations stored in a local SQLite database.
**Traces to:** SR-700
**Rationale:** SQLite is zero-configuration, file-based, included in Python stdlib, and handles concurrent reads well. OQE data is append-heavy with occasional queries — ideal for SQLite.

### DD-008: Matplotlib for Plotting
**Decision:** Use Matplotlib with FigureCanvasQTAgg for embedded plots.
**Traces to:** SR-405, UR-006
**Rationale:** Matplotlib is the de facto Python plotting library, supports all required plot types, embeds natively in Qt, and outputs publication-quality figures. The Handle Graphics object model wraps Matplotlib's Figure/Axes/Artist hierarchy.

## 4. Component Specifications

### 4.1 Engine Layer

#### 4.1.1 Lexer (`forge/engine/lexer.py`)
**Traces to:** SR-100
**Interface:** `tokenize(source: str) -> List[Token]`
**Responsibilities:**
- Convert M-language source to token stream
- Handle transpose-vs-char ambiguity via previous-token heuristic
- Support line continuation (`...`), block comments (`%{ %}`)
- Track line/column numbers for error reporting

#### 4.1.2 Parser (`forge/engine/parser.py`)
**Traces to:** SR-101, SR-102
**Interface:** `parse(source: str) -> List[AST_Node]`
**Responsibilities:**
- Convert token stream to AST using Pratt parser
- 11-level operator precedence matching Octave
- Recursive descent for statements and function definitions
- Backtracking for multi-return assignment detection

**AST Node Types:** NumberLiteral, StringLiteral, Identifier, UnaryOp, BinaryOp, CompareOp, LogicalOp, TransposeOp, ColonExpr, Index, CellIndex, FieldAccess, DynamicFieldAccess, MatrixLiteral, CellLiteral, FunctionHandle, AnonFunction, EndKeyword, Assignment, IfStatement, ForStatement, WhileStatement, DoUntilStatement, SwitchStatement, TryCatchStatement, ReturnStatement, BreakStatement, ContinueStatement, FunctionDef, ExpressionStatement, GlobalStatement, PersistentStatement.

#### 4.1.3 Evaluator (`forge/engine/evaluator.py`)
**Traces to:** SR-103, SR-104, SR-105, SR-106
**Interface:** `Session.eval(source: str) -> Any`
**Responsibilities:**
- Walk AST and evaluate expressions against workspace
- Dispatch function calls (built-in → user-defined → path lookup)
- Manage variable scoping (function-local, global, persistent)
- Execute control flow (if/for/while/switch/try with break/continue/return)
- Capture output (disp, fprintf) into output buffer

**Signal Classes:** BreakSignal, ContinueSignal, ReturnSignal (Python exceptions used for non-local control flow within the evaluator).

### 4.2 Type System

#### 4.2.1 ForgeArray (`forge/engine/types.py`)
**Traces to:** SR-200, SR-201, SR-202
**Key Design:**
- Wraps numpy.ndarray with 1-based indexing conversion at API boundary
- Scalars are 1x1 matrices (Octave semantics)
- 1D arrays are 1xN row vectors
- Single-integer indexing uses column-major linear order (Fortran order)
- Arithmetic operators return ForgeArray
- `_is_char` flag distinguishes char arrays from uint8

#### 4.2.2 Container Types (`forge/engine/containers.py`)
**Traces to:** SR-203, SR-204, SR-205, SR-206, SR-207
- ForgeChar: Subclass of ForgeArray with string semantics
- ForgeCell: List-backed heterogeneous container with {} indexing
- ForgeStruct: OrderedDict-backed named field container
- ForgeMap: OrderedDict with typed keys (containers.Map)
- ForgeSparse: scipy.sparse.csc_matrix wrapper

### 4.3 GUI Layer

#### 4.3.1 MainWindow (`forge/app.py`)
**Traces to:** SR-400
- QMainWindow with QDockWidget layout
- Menu bar, toolbar, status bar
- Dock state save/restore via QSettings

#### 4.3.2 Widget Interfaces
Each widget communicates with the engine through the Session object:
- CommandWidget → `session.eval(text)` → display output
- WorkspaceBrowser → `session.workspace.items()` → populate table
- Editor → `session.eval(file_contents)` → display output
- PlotWidget → matplotlib figure objects created by built-in plot functions

### 4.4 Validation Layer

#### 4.4.1 OQE System (`forge/validation/oqe.py`)
**Traces to:** SR-700
- `@oqe_instrument` decorator captures input/output hashes, timing, anomalies
- SQLite database stores observations and aggregated stats
- Anomaly detection: NaN in output, Inf in output, performance regression

#### 4.4.2 V&V Framework (`forge/validation/framework.py`)
**Traces to:** SR-701
- Tolerance comparison: absolute, relative, ULP-based
- NaN-aware comparison (NaN==NaN for testing purposes)
- assert_close, assert_exact, assert_identity, assert_property
- VVReport: per-function pass/fail collection

## 5. Data Flow

### 5.1 Command Execution
```
User types "x = sin(pi/4)" in CommandWidget
  → CommandWidget.on_enter()
    → Session.eval("x = sin(pi/4)")
      → Lexer.tokenize() → [IDENT'x', ASSIGN'=', IDENT'sin', LPAREN, ...]
      → Parser.parse() → Assignment(Identifier('x'), Index(Identifier('sin'), [BinaryOp(...)]))
      → Evaluator._exec() → Evaluator._exec_assign()
        → Evaluator._eval_expr() → sin(pi/4) → 0.7071...
        → workspace.set('x', ForgeArray(0.7071...))
    → return ForgeArray(0.7071...)
  → WorkspaceBrowser.refresh() → shows 'x  1x1  double  0.7071'
```

### 5.2 Function Call Resolution
```
Identifier 'f' encountered during evaluation:
  1. Check workspace: ws.has('f')? → return variable
  2. Check function registry: session.functions['f']?
     a. If callable (built-in lambda): return callable
     b. If FunctionDef (user function): return FunctionDef
  3. Check path for f.m file: (future) load and parse
  4. Raise NameError
```

## 6. File Layout

```
~/forge/
├── pyproject.toml
├── forge/
│   ├── __init__.py          (version)
│   ├── __main__.py          (entry point)
│   ├── app.py               (MainWindow, QApplication)
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── lexer.py         (SR-100)
│   │   ├── parser.py        (SR-101, SR-102)
│   │   ├── evaluator.py     (SR-103, SR-104, SR-105, SR-106)
│   │   ├── types.py         (SR-200, SR-201, SR-202)
│   │   ├── containers.py    (SR-203-207)
│   │   └── builtins/        (SR-500-508)
│   ├── gui/                  (SR-400-407)
│   │   └── __init__.py
│   └── validation/           (SR-700, SR-701)
│       ├── __init__.py
│       ├── oqe.py
│       └── framework.py
├── tests/
│   ├── conftest.py
│   ├── test_smoke.py        (SR-400)
│   ├── test_types.py        (SR-200, SR-201, SR-202, SR-300, SR-301)
│   ├── test_containers.py   (SR-203-207)
│   ├── test_lexer.py        (SR-100)
│   ├── test_parser.py       (SR-101, SR-102)
│   ├── test_evaluator.py    (SR-103-106, SR-500)
│   ├── test_oqe.py          (SR-700)
│   └── test_framework.py    (SR-701)
├── docs/
│   ├── urd/                  (FORGE-URD-001)
│   ├── srs/                  (FORGE-SRS-001)
│   ├── add/                  (FORGE-ADD-001)
│   ├── dds/                  (per-module detailed design)
│   ├── vvp/                  (V&V Plan)
│   ├── traceability/         (requirements traceability)
│   └── reports/              (V&V reports)
└── scripts/
    └── run_tests.sh
```

## 7. Interface Contracts

### 7.1 Engine → GUI
The GUI layer communicates with the engine exclusively through the `Session` class:
```python
session = Session()
result = session.eval("x = 2 + 3")           # Execute code
vars = session.workspace.items()               # Read workspace
output = session.output_buffer.getvalue()      # Read output text
session.workspace.set("x", ForgeArray(42))     # Set variable
```
No other coupling exists between GUI and engine. The engine has zero Qt dependencies.

### 7.2 Engine → Validation
OQE instrumentation is opt-in via the `@oqe_instrument` decorator. Functions work identically with or without instrumentation. The validation layer has no effect on computation results.

## 8. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-19 | Dev Team | Initial release |
