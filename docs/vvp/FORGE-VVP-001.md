# Forge IDE — Verification & Validation Plan (VVP)

**Document ID:** FORGE-VVP-001
**Version:** 1.0
**Date:** 2026-03-19
**Status:** Draft
**Parents:** FORGE-URD-001, FORGE-SRS-001, FORGE-ADD-001
**Author:** Development Team

---

## 1. Purpose

This document defines the verification and validation strategy for Forge IDE. It specifies what is tested, how it is tested, at what level, and what constitutes sufficient evidence that each requirement is satisfied.

## 2. V-Model Test Levels

```
    URD (User Requirements)          ←→  Level 5: User Acceptance Tests
         ↓                                       ↑
    SRS (System Requirements)        ←→  Level 4: System Tests
         ↓                                       ↑
    ADD (Architecture Design)        ←→  Level 3: Integration Tests
         ↓                                       ↑
    DDS (Detailed Design)            ←→  Level 2: Component Tests
         ↓                                       ↑
    Code Implementation              ←→  Level 1: Unit Tests
```

### Level 1: Unit Tests
**Scope:** Individual functions, methods, and classes in isolation.
**Tooling:** pytest + pytest-qt (headless via Xvfb)
**Frequency:** Every code change (CI equivalent: run_tests.sh)
**Pass criteria:** 100% of tests pass.
**Artifacts:** test_*.py files, pytest output logs.

### Level 2: Component Tests
**Scope:** Each major component (lexer, parser, evaluator, types, containers, OQE, GUI widgets) tested as a unit with its internal dependencies but mocked external interfaces.
**Tooling:** pytest with fixtures providing isolated Sessions.
**Pass criteria:** Each component's test suite passes independently.
**Artifacts:** Per-component test files with component-level test classes.

### Level 3: Integration Tests
**Scope:** Cross-component interactions: GUI ↔ Engine, Engine ↔ Toolboxes, Parser ↔ Evaluator pipelines.
**Tooling:** pytest-qt for GUI-engine integration, multi-statement eval for engine integration.
**Pass criteria:** End-to-end scenarios produce correct results.
**Artifacts:** test_integration.py (to be created).

### Level 4: System Tests
**Scope:** Full application behavior against SRS requirements. Each SR-xxx has at least one system test.
**Tooling:** pytest-qt for GUI, Session.eval for headless.
**Pass criteria:** Every mandatory SRS requirement (M-priority) has a passing system test.
**Artifacts:** test_system.py organized by SRS requirement ID.

### Level 5: User Acceptance Tests
**Scope:** Real-world user scenarios testing against URD requirements.
**Tooling:** Scripted scenarios executed via Session.eval or pytest-qt.
**Pass criteria:** Each UR-xxx acceptance criterion demonstrated.
**Artifacts:** test_acceptance.py organized by URD requirement ID.

## 3. Test Categories

### 3.1 Correctness Tests
Verify that computational functions produce correct results.
- **Known-value tests:** Compare output against analytically known values (sin(0)=0, eig(I)=[1,1,...]).
- **Identity tests:** Verify mathematical identities (sin(asin(x))=x, A*inv(A)=I).
- **Round-trip tests:** Encode then decode returns original (fft/ifft, sparse/full).
- **Reference tests:** Compare against GNU Octave output for identical inputs.
- **Edge case tests:** Boundary values, empty inputs, single elements, NaN, Inf, very large/small.

### 3.2 Error Handling Tests
Verify that invalid inputs produce appropriate errors.
- Wrong type, wrong size, out-of-range, division by zero.
- Error messages include function name and descriptive text.
- try/catch properly catches errors.

### 3.3 Performance Tests
Verify that operations meet performance targets.
- Startup time < 3s (SR-800)
- REPL latency < 100ms (SR-801)
- Matrix ops within 2x NumPy (SR-802)

### 3.4 GUI Tests
Verify that GUI components work correctly.
- Widget creation and visibility (pytest-qt qtbot.addWidget)
- User interaction simulation (qtbot.keyClicks, qtbot.mouseClick)
- State changes propagate (type command → workspace updates)

### 3.5 OQE (Ongoing Quality Evaluation)
For functions that lack clear analytical validation:
- Instrument with @oqe_instrument
- Collect input/output hashes over time
- Detect anomalies: unexpected NaN, Inf, or performance regression
- Periodic review of OQE dashboards

## 4. Test Naming Convention

```
test_{component}_{requirement}_{scenario}

Examples:
  test_evaluator_SR103_add_scalars
  test_lexer_SR100_block_comment
  test_types_SR201_one_based_indexing_error_on_zero
```

## 5. Test Procedure Template

Each test procedure follows this structure:

```
Test ID:        TP-{SR_ID}-{sequence}
Requirement:    SR-xxx
Level:          1-5
Preconditions:  (setup required before test)
Steps:
  1. (action)
  2. (action)
Expected Result: (what should happen)
Pass Criteria:   (how to determine pass/fail)
```

## 6. Verification Matrix Summary

This table summarizes which test level verifies each SRS requirement. Full traceability is in the traceability spreadsheet (FORGE-TRACE-001.xlsx).

| SRS ID | Description | L1 Unit | L2 Component | L3 Integration | L4 System | L5 Acceptance |
|--------|-------------|---------|--------------|----------------|-----------|---------------|
| SR-100 | Lexer | 64 tests | - | - | - | - |
| SR-101 | Parser | 84 tests | - | - | - | - |
| SR-102 | Operator Precedence | 8 tests | - | - | TP-102 | - |
| SR-103 | Expression Evaluator | 30+ tests | - | TP-103-INT | TP-103-SYS | - |
| SR-104 | Control Flow | 15 tests | - | - | TP-104 | - |
| SR-105 | Function Dispatch | 10 tests | - | TP-105-INT | TP-105 | TP-105-UAT |
| SR-106 | Error System | 5 tests | - | - | TP-106 | - |
| SR-200 | ForgeArray | 50 tests | - | - | - | - |
| SR-201 | 1-Based Indexing | 7 tests | - | - | TP-201 | TP-201-UAT |
| SR-202 | Complex Numbers | 5 tests | - | - | - | - |
| SR-203 | Char Arrays | 16 tests | - | - | - | - |
| SR-204 | Cell Arrays | 12 tests | - | - | - | - |
| SR-205 | Structs | 16 tests | - | - | - | - |
| SR-206 | Sparse | 12 tests | - | - | - | - |
| SR-207 | Containers.Map | 10 tests | - | - | - | - |
| SR-300 | Construction Functions | 27 tests | - | - | - | - |
| SR-301 | Matrix Literals | 7 tests | - | - | - | - |
| SR-400 | Application Window | 5 tests | - | TP-400-INT | TP-400 | TP-400-UAT |
| SR-401-407 | GUI Widgets | pending | pending | pending | pending | pending |
| SR-500 | Elementary Math | 80+ built-in | pending | - | TP-500 | - |
| SR-501-508 | Toolboxes | per V&V sheet | pending | pending | pending | pending |
| SR-600 | File I/O | pending | pending | - | TP-600 | - |
| SR-700 | OQE System | 15 tests | - | TP-700-INT | - | - |
| SR-701 | V&V Framework | 24 tests | - | - | - | - |
| SR-800-802 | Performance | pending | pending | - | TP-800-802 | - |

**Current coverage:** 432 Level 1 tests passing across SR-100 through SR-701.

## 7. Regression Testing

After every development stage:
1. Run `~/forge/scripts/run_tests.sh` (full Level 1 suite)
2. All tests must pass (0 failures)
3. Test count must not decrease (no test deletion without justification)
4. New code must include new tests before merge

## 8. Release Criteria

Before any release:
- [ ] All mandatory SRS requirements have passing L1 and L4 tests
- [ ] No open L4 system test failures
- [ ] OQE anomaly rate < 1% across instrumented functions
- [ ] Performance tests (SR-800-802) passing
- [ ] All V-model documents reviewed and approved
- [ ] Traceability matrix complete (every UR → SR → test link verified)

## 9. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-19 | Dev Team | Initial release |
