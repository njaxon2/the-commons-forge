# Forge MEMORY.md — Session Continuity

## Current State (R85)
- **Last commit**: R85 — MATLAB-style type names in class() and whos
- **VPS**: ubuntu@15.204.8.77 (NOT 34.32.100.183 which is production)
- **Display**: :99 port 5900 (GUI), :98 port 5901 (testing)
- **Git branch**: master

## Recent Commits (R78-R85)
- R78: Update MEMORY.md with complete thesis arc and current state
- R79: Fix command widget welcome message and prompt after Ctrl+C/command
- R80: Diagonal preconditioning generalizes to elliptical geometries
- R81: Enhanced plotting (bar/legend/set/xlim/ylim) + thesis figures
- R82: Publication-quality thesis figures + plotting fixes
- R83: Tab completion + string builtins (chr, char, num2str, etc.)
- R84: Register missing type-checking builtins
- R85: MATLAB-style type names in class() and whos

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

## Forge Features Added (R79-R85)

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
- Interpreter slow for deeply nested loops (~4s per IGA assembly at nel=4)
- fprintf with \n in string causes lexer error (escaped chars in double quotes)

## Next Steps
1. **Thesis production**: Document with Forge-exported figures (complete)
2. **Forge features**: Syntax highlighting, more plotting options
3. Build order elevation for p>2 on mapped geometry
4. Improve interpreter performance for nested loops
5. Fix escaped characters (\n, \t) in double-quoted strings
