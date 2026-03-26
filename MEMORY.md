# Forge Development Memory

## Current State: R144
- **Functions**: 1077 registered
- **E2E Tests**: 240 passing, 0 failing
- **Git**: master branch, commit 29af0c3
- **VPS**: ubuntu@15.204.8.77, project at ~/forge/
- **Thesis Figures**: 4 generated (Fig 14-17)

## Session R134-R144 Progress
- R134: 24 functions (cast, filter, mpower, butter, freqz, lyap)
- R135: 1007 milestone (gamma, erf, bessel, bitops, validators)
- R136: ODE solvers (ode45/23/15s/23s), optimization (fzero/fminbnd/fminsearch)
- R137: Thesis Fig 15 — ODE validation
- R138: CSV I/O, sparse solvers, data analysis, signal, strings
- R139: Thesis Fig 16 — NURBS vs Standard IGA convergence
- R140: Control systems (lqr/care/dare/place), image, extended trig
- R141: Degree trig, string ops, type conversion, debug stubs
- R142: rref, normpdf/normcdf/norminv, meshgrid, peaks
- R143: Thesis Fig 17 — Statistical IGA analysis
- R144: Probability distributions (7 RNG + 10 PDF/CDF/INV), file ops

## Key Technical Details
- ForgeChar is subclass of ForgeArray — check ForgeChar first
- NumPy 2.4: use np.trapezoid not np.trapz
- Cell length/numel: uses _shape attribute via hasattr check
- Constants copied into function local workspaces
- Lexer: expression-expecting keywords suppress transpose
