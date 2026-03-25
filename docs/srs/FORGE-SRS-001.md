# Forge IDE — System Requirements Specification (SRS)

**Document ID:** FORGE-SRS-001
**Version:** 1.0
**Date:** 2026-03-19
**Status:** Draft
**Parent:** FORGE-URD-001
**Author:** Development Team

---

## 1. Purpose

This document derives testable system requirements from the User Requirements Document (FORGE-URD-001). Each requirement has a unique ID, traces to a parent UR, specifies acceptance criteria, and identifies the verification method.

## 2. Verification Methods

| Code | Method | Description |
|------|--------|-------------|
| T | Test | Automated test (unit, integration, or system) |
| I | Inspection | Manual code or document review |
| A | Analysis | Mathematical or logical proof |
| D | Demonstration | Manual walkthrough showing capability |

## 3. Requirement Priority

| Level | Meaning |
|-------|---------|
| M | Mandatory — must be present for release |
| H | High — expected for release, deferral requires justification |
| D | Desirable — implement if schedule permits |

---

## 4. System Requirements

### 4.1 Interpreter Core (traces to UR-001, UR-002, UR-003)

#### SR-100: Lexer
| Field | Value |
|-------|-------|
| **ID** | SR-100 |
| **Parent** | UR-002 |
| **Priority** | M |
| **Description** | The system shall tokenize M-language source code into a stream of typed tokens. |
| **Acceptance Criteria** | All Octave token types recognized: numbers (int, float, hex, binary, scientific, imaginary), strings (single/double quoted), identifiers, keywords (30+), operators (25+), delimiters, comments (line and block), newlines. |
| **Verification** | T — test_lexer.py (64 tests) |

#### SR-101: Parser
| Field | Value |
|-------|-------|
| **ID** | SR-101 |
| **Parent** | UR-002 |
| **Priority** | M |
| **Description** | The system shall parse token streams into an Abstract Syntax Tree (AST) representing the program structure. |
| **Acceptance Criteria** | Correct AST for: expressions with 11-level operator precedence matching Octave, all statement types (if/for/while/switch/try/function), matrix and cell literals, function handles, anonymous functions, multi-return assignment. |
| **Verification** | T — test_parser.py (84 tests) |

#### SR-102: Operator Precedence
| Field | Value |
|-------|-------|
| **ID** | SR-102 |
| **Parent** | UR-002 (AC-002.3) |
| **Priority** | M |
| **Description** | Operator precedence shall match GNU Octave exactly: (1) transpose, (2) power, (3) unary, (4) multiply/divide, (5) add/subtract, (6) colon, (7) relational, (8) element-AND, (9) element-OR, (10) short-circuit AND, (11) short-circuit OR. Power is right-associative; all others left-associative. |
| **Verification** | T — TestPrecedence class (8 tests) |

#### SR-103: Expression Evaluator
| Field | Value |
|-------|-------|
| **ID** | SR-103 |
| **Parent** | UR-001, UR-003 |
| **Priority** | M |
| **Description** | The system shall evaluate AST expressions to produce numerical results matching Octave behavior. |
| **Acceptance Criteria** | Correct evaluation of: arithmetic (+,-,*,/,\,^,.*,./,.^), comparison (==,~=,<,>,<=,>=), logical (&,|,&&,||,~), transpose (',.\'), colon ranges, function calls, indexing, field access. |
| **Verification** | T — test_evaluator.py TestArithmeticEval, TestComparisonEval, TestLogicalEval, TestTransposeEval, TestColonEval (30+ tests) |

#### SR-104: Control Flow
| Field | Value |
|-------|-------|
| **ID** | SR-104 |
| **Parent** | UR-002 (AC-002.1) |
| **Priority** | M |
| **Description** | The system shall execute all Octave control flow constructs: if/elseif/else/end, for/end, while/end, do/until, switch/case/otherwise/end, try/catch/end, break, continue, return. |
| **Verification** | T — test_evaluator.py TestIfEval, TestForEval, TestWhileEval, TestSwitchEval, TestTryCatchEval (15 tests) |

#### SR-105: Function Definition and Dispatch
| Field | Value |
|-------|-------|
| **ID** | SR-105 |
| **Parent** | UR-002 (AC-002.2) |
| **Priority** | M |
| **Description** | The system shall support user-defined functions with parameters, return values, local scope, and recursion. Function resolution order: local variables, subfunctions, private functions, path functions, built-in functions. |
| **Acceptance Criteria** | Single/multi-return functions, recursive calls, workspace isolation (local variables don't leak), anonymous functions with closures, function handles. |
| **Verification** | T — test_evaluator.py TestUserFunctions, TestAnonymousFunction, TestFunctionHandle (10 tests) |

#### SR-106: Error System
| Field | Value |
|-------|-------|
| **ID** | SR-106 |
| **Parent** | UR-015 |
| **Priority** | M |
| **Description** | The system shall provide error/warning mechanisms matching Octave: error(id, msg), warning(id, msg), try/catch with error struct, assert(). |
| **Verification** | T — test_evaluator.py TestIO error tests, TestTryCatchEval (5 tests) |

### 4.2 Data Types (traces to UR-005)

#### SR-200: Numeric Array (ForgeArray)
| Field | Value |
|-------|-------|
| **ID** | SR-200 |
| **Parent** | UR-005 (AC-005.1) |
| **Priority** | M |
| **Description** | The system shall provide an N-dimensional numeric array type supporting double, single, int8/16/32/64, uint8/16/32/64, and logical dtypes with 1-based indexing. Scalars are 1x1, 1D arrays are 1xN row vectors. |
| **Verification** | T — test_types.py TestForgeArrayCreation, TestForgeArrayIndexing, TestTypeConversion (50 tests) |

#### SR-201: 1-Based Indexing
| Field | Value |
|-------|-------|
| **ID** | SR-201 |
| **Parent** | UR-002 (AC-002.4), UR-004 (AC-004.4) |
| **Priority** | M |
| **Description** | All array indexing shall be 1-based. A(1) returns the first element. Single-integer indexing on N-D arrays shall use column-major linear indexing. Index 0 shall raise an error. |
| **Verification** | T — test_types.py TestForgeArrayIndexing (7 tests) |

#### SR-202: Complex Numbers
| Field | Value |
|-------|-------|
| **ID** | SR-202 |
| **Parent** | UR-005 (AC-005.7) |
| **Priority** | M |
| **Description** | The system shall support complex128 arrays. Complex literals (`3+4i`, `2j`) shall parse correctly. Complex arithmetic, conjugate transpose, and functions (real, imag, conj, angle, abs) shall work on complex inputs. |
| **Verification** | T — test_types.py TestComplex (5 tests) |

#### SR-203: Char Arrays
| Field | Value |
|-------|-------|
| **ID** | SR-203 |
| **Parent** | UR-005 (AC-005.3) |
| **Priority** | M |
| **Description** | Single-quoted literals create char arrays (uint8 codes). String comparison functions (strcmp, strcmpi, strncmp, strncmpi) shall be provided. Multi-row char arrays pad with spaces. |
| **Verification** | T — test_containers.py TestCharArrays, TestStringComparison (16 tests) |

#### SR-204: Cell Arrays
| Field | Value |
|-------|-------|
| **ID** | SR-204 |
| **Parent** | UR-005 (AC-005.4) |
| **Priority** | M |
| **Description** | The system shall support heterogeneous cell arrays with {} content access and () cell access. cell(m,n) creates empty cells. cellfun applies a function to each element. |
| **Verification** | T — test_containers.py TestCellArrays (12 tests) |

#### SR-205: Structs
| Field | Value |
|-------|-------|
| **ID** | SR-205 |
| **Parent** | UR-005 (AC-005.5) |
| **Priority** | M |
| **Description** | The system shall support structs with named fields, dynamic field access via s.(name), field manipulation (fieldnames, rmfield, setfield, getfield, orderfields), and nested structs. |
| **Verification** | T — test_containers.py TestStructs (16 tests) |

#### SR-206: Sparse Matrices
| Field | Value |
|-------|-------|
| **ID** | SR-206 |
| **Parent** | UR-005 (AC-005.6) |
| **Priority** | M |
| **Description** | The system shall support sparse matrix creation (from dense, from triplets), arithmetic (+, scalar *, matrix @), and round-trip conversion (sparse/full). |
| **Verification** | T — test_containers.py TestSparseMatrices (12 tests) |

#### SR-207: Containers.Map
| Field | Value |
|-------|-------|
| **ID** | SR-207 |
| **Parent** | UR-005 (AC-005.8) |
| **Priority** | H |
| **Description** | The system shall provide containers.Map with typed keys, CRUD operations (get/set/remove/isKey), and keys()/values() queries. |
| **Verification** | T — test_containers.py TestContainersMap (10 tests) |

### 4.3 Matrix Construction (traces to UR-004)

#### SR-300: Construction Functions
| Field | Value |
|-------|-------|
| **ID** | SR-300 |
| **Parent** | UR-004 |
| **Priority** | M |
| **Description** | The system shall provide matrix construction functions: eye, ones, zeros, rand, randn, randi, true, false, diag, linspace, colon, repmat. Size arguments follow Octave convention: `zeros(n)` = nxn, `zeros(m,n)` = mxn. |
| **Verification** | T — test_types.py TestMatrixConstruction (27 tests) |

#### SR-301: Matrix Literals
| Field | Value |
|-------|-------|
| **ID** | SR-301 |
| **Parent** | UR-004 (AC-004.1, AC-004.5) |
| **Priority** | M |
| **Description** | The system shall support matrix literal syntax: `[1 2; 3 4]` creates a 2x2 matrix. Rows separated by `;` or newline. Elements separated by space or `,`. Empty matrix `[]` creates 0x0. |
| **Verification** | T — test_evaluator.py TestMatrixEval (4 tests), test_parser.py TestMatrixLiteral (3 tests) |

### 4.4 GUI (traces to UR-001, UR-007, UR-008, UR-009, UR-010, UR-011)

#### SR-400: Application Window
| Field | Value |
|-------|-------|
| **ID** | SR-400 |
| **Parent** | UR-001 |
| **Priority** | M |
| **Description** | The application shall launch a main window with menu bar (File, Edit, View, Debug, Window, Help), toolbar, status bar, and dockable panels. Dock layout shall persist across sessions. |
| **Verification** | T — test_smoke.py (5 tests) |

#### SR-401: Command Widget
| Field | Value |
|-------|-------|
| **ID** | SR-401 |
| **Parent** | UR-001 (AC-001.1 through AC-001.5) |
| **Priority** | M |
| **Description** | A REPL widget shall display `>> ` prompt, accept user input, execute it via the interpreter, and display output. History navigation, multi-line continuation, tab completion, and syntax highlighting shall be supported. |
| **Verification** | T + D |

#### SR-402: Code Editor
| Field | Value |
|-------|-------|
| **ID** | SR-402 |
| **Parent** | UR-007 |
| **Priority** | M |
| **Description** | An integrated code editor with M-language syntax highlighting, line numbers, bracket matching, find/replace, run file (F5), run selection (F9), and multiple tabs. |
| **Verification** | T + D |

#### SR-403: Workspace Browser
| Field | Value |
|-------|-------|
| **ID** | SR-403 |
| **Parent** | UR-008 |
| **Priority** | M |
| **Description** | A table view showing all workspace variables with columns: Name, Size, Class, Value. Updates in real-time when workspace changes. Double-click opens variable editor. |
| **Verification** | T + D |

#### SR-404: File Browser
| Field | Value |
|-------|-------|
| **ID** | SR-404 |
| **Parent** | UR-009 |
| **Priority** | H |
| **Description** | A tree view of the file system with current directory display, directory navigation, and context menu (new file, rename, delete). Double-click .m files opens in editor. |
| **Verification** | T + D |

#### SR-405: Plot System
| Field | Value |
|-------|-------|
| **ID** | SR-405 |
| **Parent** | UR-006 |
| **Priority** | M |
| **Description** | Matplotlib-backed plot system with figure windows as dock widgets. Supports 2D (plot, scatter, bar, etc.), 3D (surf, mesh, contour), formatting (title, labels, legend), interactive tools (zoom, pan), and export (PNG, SVG, PDF, EPS). |
| **Verification** | T + D |

#### SR-406: Help/Documentation Viewer
| Field | Value |
|-------|-------|
| **ID** | SR-406 |
| **Parent** | UR-010 |
| **Priority** | M |
| **Description** | help(func) displays docstring. doc(func) opens rich HTML viewer. lookfor(keyword) searches all function descriptions. |
| **Verification** | T |

#### SR-407: Debugger
| Field | Value |
|-------|-------|
| **ID** | SR-407 |
| **Parent** | UR-011 |
| **Priority** | H |
| **Description** | Breakpoint support (editor gutter click), step controls (in/over/out/continue/stop), stack frame viewer, variable inspection at breakpoints. dbstop/dbcont/dbstep/dbquit commands. |
| **Verification** | T + D |

### 4.5 Built-in Functions (traces to UR-002 AC-002.5, UR-012)

#### SR-500: Elementary Math
| Field | Value |
|-------|-------|
| **ID** | SR-500 |
| **Parent** | UR-003, UR-002 |
| **Priority** | M |
| **Description** | abs, sign, sqrt, exp, log, log2, log10, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, asinh, acosh, atanh, ceil, floor, round, fix, mod, rem, hypot, complex, real, imag, conj, angle. All functions operate element-wise on arrays and handle NaN/Inf per IEEE 754. |
| **Verification** | T — per-function tests with known values, edge cases, NaN propagation |

#### SR-501: Extended Trigonometric (elfun)
| Field | Value |
|-------|-------|
| **ID** | SR-501 |
| **Parent** | UR-002 |
| **Priority** | M |
| **Description** | 27 degree/hyperbolic trig functions per V&V traceability sheet EF-001 through EF-027: sind, cosd, tand, acosd, asind, atand, atan2d, cot, cotd, coth, csc, cscd, csch, sec, secd, sech, acot, acotd, acoth, acsc, acscd, acsch, asec, asecd, asech, cospi, sinpi. |
| **Verification** | T — identity tests (e.g., sind(asind(x))=x), exact special values (sind(30)=0.5), domain enforcement |

#### SR-502: Special Functions
| Field | Value |
|-------|-------|
| **ID** | SR-502 |
| **Parent** | UR-002 |
| **Priority** | M |
| **Description** | 21 functions per SF-001 through SF-021: beta, betainc, betaincinv, betaln, cosint, ellipke, expint, factor, factorial, gammainc, gammaincinv, isprime, lcm, legendre, nchoosek, nthroot, primes, reallog, realpow, realsqrt, sinint. |
| **Verification** | T — known values, round-trip tests (betainc/betaincinv), domain enforcement |

#### SR-503: Linear Algebra
| Field | Value |
|-------|-------|
| **ID** | SR-503 |
| **Parent** | UR-012 (AC-012.4) |
| **Priority** | M |
| **Description** | 37 functions per LA-001 through LA-037 plus decompositions (eig, svd, lu, qr, chol, schur, hess) and solvers (mldivide, mrdivide). |
| **Verification** | T — reconstruction tests (A=U*S*V'), identity properties (A*inv(A)=I), known eigenvalues |

#### SR-504: Signal Processing Toolbox
| Field | Value |
|-------|-------|
| **ID** | SR-504 |
| **Parent** | UR-012 (AC-012.1) |
| **Priority** | M |
| **Description** | 300+ functions covering: filter design (butter, cheby1, ellip), spectral analysis (fft, periodogram, pwelch), window functions (hamming, hanning, blackman), convolution, correlation, resampling. |
| **Verification** | T — known filter responses, Parseval's theorem, round-trip filter/defilter |

#### SR-505: Image Processing Toolbox
| Field | Value |
|-------|-------|
| **ID** | SR-505 |
| **Parent** | UR-012 (AC-012.2) |
| **Priority** | M |
| **Description** | 200+ functions covering: spatial filtering, morphological operations, geometric transforms, color conversion, feature detection, image I/O. |
| **Verification** | T — known filter outputs, round-trip transform/inverse, color conversion identities |

#### SR-506: Statistics/ML Toolbox
| Field | Value |
|-------|-------|
| **ID** | SR-506 |
| **Parent** | UR-012 (AC-012.3) |
| **Priority** | M |
| **Description** | 300+ functions covering: descriptive statistics, probability distributions, hypothesis tests, regression, classification, clustering, cross-validation. |
| **Verification** | T — known distribution values, test statistic p-values, regression coefficient recovery |

#### SR-507: ODE Solvers
| Field | Value |
|-------|-------|
| **ID** | SR-507 |
| **Parent** | UR-012 (AC-012.6) |
| **Priority** | M |
| **Description** | ode45, ode23, ode15s, ode23s, ode15i with odeset options, event detection, and mass matrix support. |
| **Verification** | T — known ODE solutions (harmonic oscillator, Van der Pol, Robertson stiff), convergence order |

#### SR-508: Optimization Toolbox
| Field | Value |
|-------|-------|
| **ID** | SR-508 |
| **Parent** | UR-012 (AC-012.5) |
| **Priority** | M |
| **Description** | fzero, fminbnd, fminsearch, fminunc, fsolve, lsqnonneg, linprog, quadprog. |
| **Verification** | T — known optima (Rosenbrock), constraint satisfaction, KKT conditions |

### 4.6 I/O and File System (traces to UR-018)

#### SR-600: File I/O
| Field | Value |
|-------|-------|
| **ID** | SR-600 |
| **Parent** | UR-018 |
| **Priority** | M |
| **Description** | csvread/csvwrite, load/save (.mat), imread/imwrite, audioread/audiowrite, jsonencode/jsondecode, dlmread/dlmwrite. C-style formatted I/O: fprintf, sprintf, fscanf, sscanf. |
| **Verification** | T — round-trip read/write for each format |

### 4.7 Validation & Quality (traces to UR-020)

#### SR-700: OQE System
| Field | Value |
|-------|-------|
| **ID** | SR-700 |
| **Parent** | UR-020 (AC-020.3) |
| **Priority** | M |
| **Description** | Every computational function can be instrumented with the @oqe_instrument decorator, which records: input/output hashes, execution timing, anomaly flags (NaN in output, Inf in output). Data stored in SQLite. |
| **Verification** | T — test_oqe.py (15 tests) |

#### SR-701: V&V Framework
| Field | Value |
|-------|-------|
| **ID** | SR-701 |
| **Parent** | UR-020 (AC-020.1, AC-020.2) |
| **Priority** | M |
| **Description** | Tolerance comparison framework supporting absolute, relative, and ULP tolerances. Handles NaN==NaN, Inf sign matching. Report generation per function. |
| **Verification** | T — test_framework.py (24 tests) |

#### SR-702: Reference Testing
| Field | Value |
|-------|-------|
| **ID** | SR-702 |
| **Parent** | UR-020 (AC-020.2) |
| **Priority** | M |
| **Description** | Automated comparison against GNU Octave reference outputs. Test cases extracted from Octave's 703 test files where applicable. |
| **Verification** | T + A |

### 4.8 Performance (traces to UR-013)

#### SR-800: Startup Time
| Field | Value |
|-------|-------|
| **ID** | SR-800 |
| **Parent** | UR-013 (AC-013.1) |
| **Priority** | H |
| **Description** | Application startup (to interactive prompt) shall complete in under 3 seconds on reference hardware (4-core, 8GB RAM). |
| **Verification** | T — timed startup test |

#### SR-801: REPL Latency
| Field | Value |
|-------|-------|
| **ID** | SR-801 |
| **Parent** | UR-013 (AC-013.2) |
| **Priority** | H |
| **Description** | Simple expression evaluation (e.g., `2+3`) shall complete in under 100ms including parsing and display. |
| **Verification** | T — timed evaluation test |

#### SR-802: Matrix Performance
| Field | Value |
|-------|-------|
| **ID** | SR-802 |
| **Parent** | UR-013 (AC-013.3) |
| **Priority** | H |
| **Description** | Matrix multiply of 1000x1000 double matrices shall complete within 2x of equivalent NumPy operation. |
| **Verification** | T — benchmark test with timing comparison |

---

## 5. Requirements Summary

| Category | Count | Mandatory | High | Desirable |
|----------|-------|-----------|------|-----------|
| Interpreter Core | 7 | 7 | 0 | 0 |
| Data Types | 8 | 7 | 1 | 0 |
| Matrix Construction | 2 | 2 | 0 | 0 |
| GUI | 8 | 6 | 2 | 0 |
| Built-in Functions | 9 | 9 | 0 | 0 |
| I/O | 1 | 1 | 0 | 0 |
| Validation | 3 | 3 | 0 | 0 |
| Performance | 3 | 0 | 3 | 0 |
| **Total** | **41** | **35** | **6** | **0** |

## 6. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-19 | Dev Team | Initial release |
