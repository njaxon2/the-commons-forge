# Forge Development Memory

## Current State: R135
- **Functions**: 1007 registered (milestone crossed!)
- **E2E Tests**: 169 passing, 0 failing
- **Git**: master branch, commit f3112b0
- **VPS**: ubuntu@15.204.8.77, project at ~/forge/

## Architecture
- forge/engine/session.py — Central file, ~2800+ lines, all function registrations
- forge/engine/evaluator.py — AST execution engine
- forge/engine/parser.py — M-language parser
- forge/engine/lexer.py — Tokenizer
- forge/engine/types.py — ForgeArray (numpy wrapper)
- forge/engine/containers.py — ForgeChar, ForgeCell, ForgeStruct
- forge/gui/ — PySide6 GUI (main_window, command_widget, variable_editor, etc.)
- test_e2e.py — End-to-end test suite

## Key Technical Details
- ForgeChar is subclass of ForgeArray — check ForgeChar before ForgeArray in isinstance
- 1-based indexing throughout
- Display uses _format attribute for short/long/shortE etc.
- Constants (pi, eps, Inf, etc.) copied into function local workspaces
- Lexer: keywords like case/return/if suppress transpose interpretation of single-quote
- Logical indexing uses direct boolean mask (no ravel/reshape)
- TIGA builtins: findspan, basisfun, derbasisfun, gaussQuad registered as Python functions
- NumPy 2.4: use np.trapezoid not np.trapz

## Function Categories Added
- R119-R133: Core engine fixes, TIGA, 950+ functions
- R134: 24 common functions (cast, filter, mpower, butter, freqz, lyap, etc.)
- R135: 1007 milestone — gamma, erf, bessel, bitops, validators, system

## Next Steps
- ODE solvers (ode45, ode23, ode15s)
- Optimization (fzero, fminbnd, fminsearch)
- Integral functions (integral, integral2, quad)
- Continue thesis figures
- classdef/OOP support (future)
