# The Commons V-Model Iterative Development Process

## Overview

This document formalizes the V-model development process used on the Forge project within The Commons. It is designed to be followed by Claude instances working on any subproject, ensuring consistent quality, traceability, and continuous validation.

## Core Principle: Task-Driven Iterations

Development is organized around a **driving task** — a real-world use case that exercises the system under development. Rather than building features in isolation and testing later, every feature is motivated by and immediately validated through the driving task.

**Example from Forge**: The TIGA (Isogeometric Analysis) paper serves as the driving task. Each script written in Forge's M-language exercises parser, evaluator, linear algebra, plotting, and other subsystems simultaneously. Bugs are found by doing real work, not by writing contrived tests.

## The Iteration Cycle

Each iteration follows this cycle (typically 20-60 minutes):

```
┌─────────────────────────────────────────────┐
│  1. EXPLORE                                  │
│     Run the driving task in the system       │
│     Observe what works and what breaks       │
├─────────────────────────────────────────────┤
│  2. IDENTIFY                                 │
│     Catalog shortcomings found during (1)    │
│     Distinguish: bug, missing feature,       │
│     performance issue, UX problem            │
├─────────────────────────────────────────────┤
│  3. FORMALIZE                                │
│     Write specific requirements for fixes    │
│     Define acceptance criteria               │
│     Prioritize by impact on driving task     │
├─────────────────────────────────────────────┤
│  4. IMPLEMENT                                │
│     Fix bugs / add features                  │
│     Keep changes minimal and focused         │
│     One logical change per commit            │
├─────────────────────────────────────────────┤
│  5. VERIFY                                   │
│     Re-run the driving task                  │
│     Confirm the fix works                    │
│     Check for regressions                    │
├─────────────────────────────────────────────┤
│  6. COMMIT & ADVANCE                         │
│     Commit with descriptive message          │
│     Extend the driving task to exercise      │
│     the next capability frontier             │
│     → Return to step 1                       │
└─────────────────────────────────────────────┘
```

## Choosing a Driving Task

The driving task should:

- **Be realistic** — something an actual user would do with the system
- **Be progressive** — start simple, add complexity each iteration
- **Exercise multiple subsystems** — not just one module in isolation
- **Have known correct answers** — so verification is objective
- **Be domain-appropriate** — related to the project's purpose

### Examples by Project Type

| Project | Driving Task | What It Exercises |
|---------|-------------|-------------------|
| Forge (Octave clone) | IGA numerical analysis paper | Parser, evaluator, linear algebra, plotting, file I/O |
| Web framework | Build a real CRUD app | Routing, templates, database, auth, API |
| Compiler | Compile and run a real program | Lexer, parser, codegen, runtime, stdlib |
| Data pipeline | Process a real dataset end-to-end | Ingestion, transforms, validation, output |

## Testing Strategy: Headless First, Visual Second

For GUI applications, maintain two testing paths:

1. **Headless verification** (fast, automated): Run the driving task programmatically, check results numerically. This is your primary feedback loop. Example: `ForgeSession.eval("tiga_poisson")` and check L2 error < threshold.

2. **Visual verification** (slower, manual): Run in the actual GUI to verify rendering, interaction, layout. Do this only after headless passes. Keep screenshots as evidence.

The ratio should be ~80% headless, ~20% visual.

## Commit Discipline

Each commit should represent one completed iteration:

```
R{N}: {Brief description of what changed}

{What was found}: Describe the bug or missing feature
{What was done}: Describe the fix or addition
{Verification}: Key metric (e.g., "L2 error 2.58e-04, was 0.4")

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

Use sequential R-numbers (R45, R46, R47...) for traceability across sessions. Include the Co-Authored-By line for all AI-generated commits.

## Session Continuity

Claude instances have finite context windows. To maintain continuity:

1. **MEMORY.md**: Keep a running log of what was accomplished, current state, and next steps
2. **Plan file**: Maintain a high-level plan (like the Forge plan with phases/stages)
3. **Commit history**: Use `git log` to reconstruct recent work
4. **Self-contained scripts**: Each driving task iteration should be a standalone .m/.py/.js file that can be re-run independently

When resuming a session:
- Read MEMORY.md and plan file
- Run `git log --oneline -10` to see recent work
- Run the most recent driving task script to establish baseline
- Continue from where the last session left off

## Escalation Rules

When to escalate vs. work around:

| Situation | Action |
|-----------|--------|
| Bug blocks driving task completely | Fix immediately |
| Bug causes wrong results but workaround exists | Note it, fix when convenient |
| Missing feature needed for next iteration | Implement minimally |
| Performance issue but results correct | Log, defer to optimization pass |
| Architectural issue discovered | Document, discuss with team |

## Quality Metrics

Track these per iteration:

- **Driving task coverage**: How much of the task runs successfully
- **Error magnitude**: For numerical tasks, L2/H1 error norms
- **Regression count**: How many previously-working things broke
- **Time per iteration**: Keep iterations short (20-60 min)
- **Commit frequency**: Aim for 3-6 commits per session

## Applying to Other Commons Projects

To adopt this process on a new project:

1. **Define your driving task** — pick something real and progressive
2. **Set up dual testing** — headless verification + visual confirmation
3. **Create helper scripts** — fast command runners, screenshot tools
4. **Establish commit conventions** — R-numbering, descriptive messages
5. **Write MEMORY.md** — for session continuity
6. **Start simple** — first iteration should be the simplest possible version of the driving task that exercises the core system
