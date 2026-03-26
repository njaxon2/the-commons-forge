# Forge MEMORY.md — Session Continuity

## Current State (R99)
- **Last commit**: R99 — 890 registered functions, ODE/optimization/file I/O
- **VPS**: ubuntu@15.204.8.77 (NOT 34.32.100.183 which is production)
- **Display**: :99 port 5900 (GUI), :98 port 5901 (testing)
- **Git branch**: master

## Recent Commits (R89-R99)
- R89: MATLAB-style types and previews in workspace browser GUI
- R90: regexp/strrep/strfind + cellfun/arrayfun/cat/rmfield/cell2mat/num2cell
- R91: basisfunder wrapper + E2E test suite expanded to 59 tests
- R92: MATLAB-style display for cell arrays and structs
- R93: Update MEMORY.md
- R94: File I/O — fopen/fclose/fprintf/fgetl/feof/fileread/dlmwrite/dlmread
- R95: E2E test suite expanded to 62 tests
- R96: deal, structfun, deblank, mat2str, blanks
- R97: ODE solvers — ode45, ode23, ode15s
- R98: Optimization — fzero, fminbnd, fminsearch, fsolve, integral
- R99: Fill gaps — cbrt, pow2, asinh/acosh/atanh, isa, isreal, interp2, save/load

## NOVEL FINDING: Complete Thesis Arc

### Problem: NURBS Weight-Conditioning Tradeoff
- Geometric weight (w=1/√2 for quarter circle) gives exact circle but worsens conditioning
- Conditioning penalty factor: cond(K;w_geo)/cond(K;w=1) ≈ 1.44 (constant across mesh sizes)

### Analysis: Scaling Law
- cond(K;w)/cond(K;1) = (W_max/W_min)^α where W(η) = 1 + 2(w-1)η(1-η)
- α converges to ~2.32 for quarter circle as h→0
- The weight effect is purely spectral: λ_min drops while λ_max unchanged

### Solution: Diagonal Preconditioning
- **D^{-1/2} K D^{-1/2} with D=diag(K) eliminates the entire penalty**
- Preconditioned ratio prec/bsp = 0.93-1.07 across all configurations
- Works for all mesh sizes (nel=2..8) and all arc angles (30°..120°)
- **Generalizes to elliptical geometries** (R80): prec/bsp = 0.76-1.04

### Thesis Figures (in thesis_figures/)
- fig1_weight_tradeoff.png: Main tradeoff curve
- fig2_preconditioning.png: Bar chart, mesh independence
- fig3_arc_angle.png: Arc angle dependence with scaling model
- fig4_ellipse.png: Elliptical geometry generalization

### Key Data Tables

**Mesh independence (R1=0.5, R2=1.5, quarter circle):**
| nel | w* | opt/bsp | geo/bsp | prec/bsp |
|-----|------|---------|---------|----------|
| 2 | 2.158 | 0.685 | 1.513 | 1.033 |
| 4 | 2.368 | 0.672 | 1.491 | 1.035 |
| 8 | 2.306 | 0.686 | 1.444 | 0.968 |

**Arc angle dependence (nel=4):**
| θ | w_geo | geo/bsp | prec/bsp |
|---|-------|---------|----------|
| 30° | 0.966 | 1.035 | 0.941 |
| 60° | 0.866 | 1.174 | 0.932 |
| 90° | 0.707 | 1.491 | 1.035 |
| 120° | 0.500 | 2.313 | 1.065 |

**Elliptical geometries (nel=4):**
| Geometry | geo/bsp | prec/bsp |
|----------|---------|----------|
| Circle | 1.491 | 1.035 |
| Ellipse 2:1 | 1.455 | 0.921 |
| Ellipse 4:1 | 1.345 | 0.763 |
| Non-conformal | 1.427 | 0.800 |

## Forge Features Added (R79-R92)

### Plotting (R81-R82)
- bar(): Full MATLAB calling convention (x, y, width, color)
- legend(): MATLAB keyword args (Location, FontSize), correct handle ordering
- title/xlabel/ylabel: MATLAB keyword args (FontSize, Color, etc.)
- set(gca, "FontSize"): Applies to title, labels, ticks
- xlim/ylim: Accept [lo hi] array form
- text(): Keyword args support
- saveas/print: Working PNG/SVG/PDF export at 150 DPI

### Command Widget (R79, R83)
- Single-pane terminal design (R72 original)
- Tab completion: workspace vars, functions, keywords
- Welcome message and Ctrl+C prompt fix
- History navigation (up/down arrows)

### String Builtins (R83)
- chr(), char(): integer to character
- num2str(), str2num(), str2double()
- lower(), upper(), strtrim()
- strsplit(), strjoin(), strcat()

### Type System (R84-R85)
- class(): Returns MATLAB names (double, single, logical, char, cell, struct)
- whos: Shows MATLAB type names
- ischar(), isfield(), isscalar(), isvector(), ismatrix(), issquare()

### Syntax Highlighting (R88)
- VS Code-style M-code highlighting in command widget
- Keywords (blue bold), strings (orange), numbers (green), comments (green italic), functions (yellow)

### Workspace Browser (R89)
- Shows MATLAB type names: double, char, cell, struct (not numpy names)
- Better value previews for ForgeArray, ForgeChar, ForgeCell, ForgeStruct

### String Operations (R90)
- regexp(), regexpi(), regexprep(): Full regex support with match/tokens/names modes
- strrep(), strfind(): String search and replace

### Container Utilities (R90)
- cellfun(), arrayfun(): Apply functions to cell/array elements
- cat(), num2cell(), cell2mat(), rmfield(): Container manipulation

### Display Formatting (R92)
- Cell arrays: indexed display {[1] val, [2] val, ...}
- Structs: field: value pairs with nested type summaries

### File I/O (R94)
- fopen, fclose, fprintf (to file handles), fgets, fgetl, feof, ftell, fseek
- fileread, dlmwrite, dlmread, csvread, csvwrite
- tempname, tempdir
- fprintf dispatches to file or stdout based on first argument

### ODE Solvers (R97)
- ode45 (Dormand-Prince RK45), ode23 (Bogacki-Shampine RK23), ode15s (BDF for stiff)
- Full MATLAB calling convention: [t,y] = ode45(@f, tspan, y0)
- Anonymous function handles as RHS, scipy.integrate backend

### Optimization (R98)
- fzero: root finding (bracket or initial guess)
- fminbnd: bounded scalar minimization
- fminsearch: unconstrained multivariable (Nelder-Mead)
- fsolve: nonlinear equation system
- integral: numerical integration (quad)

### Function Coverage (R99)
- 890 total registered functions
- 226/226 commonly-used MATLAB functions available
- 62/62 E2E tests passing
- IGA assembly: 31ms (nel=2), 61ms (nel=4)

## GUI State
- Single-pane command widget with tab completion
- saveas/print fixed for MATLAB calling convention
- source() command as alias for run()
- noVNC available on port 6080 (websockify installed)

## User Feedback Log
- Command window must operate like a single command prompt — FIXED R72
- Model user characterization in docs/forge_model_user.md (never summarize)

## Known Issues
- p>2 on mapped geometry needs order elevation support
- 3D array display shows type info instead of values
- Plot window focus stealing in GUI
- No subfunction support in scripts
- IGA assembly: 31ms (nel=2), 61ms (nel=4) — good performance

## Next Steps
1. **Thesis production**: Document with Forge-exported figures (complete)
2. **Forge features**: Syntax highlighting, more plotting options
3. Build order elevation for p>2 on mapped geometry
4. Improve interpreter performance for nested loops
5. Variable editor (double-click workspace variable to inspect)
