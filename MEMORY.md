# Forge MEMORY.md — Session Continuity

## Current State (R77)
- **Last commit**: R77 — Verified diagonal preconditioning across meshes and arc angles
- **VPS**: ubuntu@15.204.8.77 (NOT 34.32.100.183 which is production)
- **Display**: :99 port 5900 (GUI), :98 port 5901 (testing)
- **Git branch**: master

## Recent Commits (R68-R77)
- R68: Annulus h-convergence study, fix saveas/print, improve GUI layout
- R69: Condition number study on mapped annulus, h-p convergence attempt
- R70: NURBS weight effect on conditioning — spectral analysis
- R71: Systematic NURBS weight sweep — accuracy vs conditioning tradeoff
- R72: Weight-conditioning anatomy study + single-pane command widget
- R73: Arc angle study + analytical analysis + thesis-quality plots
- R74: Conditioning exponent analysis — alpha varies with angle and mesh
- R75: Summary thesis figure + source() command + 1D analysis script
- R76: Diagonal preconditioning eliminates NURBS weight conditioning penalty
- R77: Verified diagonal preconditioning across meshes and arc angles

## NOVEL FINDING: Complete Thesis Arc

### Problem: NURBS Weight-Conditioning Tradeoff
- Geometric weight (w=1/√2 for quarter circle) gives exact circle but worsens conditioning
- Conditioning penalty factor: cond(K;w_geo)/cond(K;w=1) ≈ 1.44 (constant across mesh sizes)
- Sharp error minimum at w=1/√2 but conditioning minimum at w≈2.3

### Analysis: Scaling Law
- cond(K;w)/cond(K;1) = (W_max/W_min)^α where W(η) = 1 + 2(w-1)η(1-η)
- W_max/W_min = 2/(1+w) for the standard conic NURBS weight function
- α converges to ~2.32 for quarter circle as h→0
- α → 2 in the small-angle limit
- The weight effect is purely spectral: λ_min drops while λ_max unchanged
- All ratios are mesh-independent (confirmed nel=2,4,8,16)

### Solution: Diagonal Preconditioning
- **D^{-1/2} K D^{-1/2} with D=diag(K) eliminates the entire penalty**
- Preconditioned ratio prec/bsp = 0.93-1.07 across all configurations
- Works for all mesh sizes (nel=2..8) and all arc angles (30°..120°)
- Even for extreme 120° arcs: 2.31x penalty → 1.07x preconditioned
- Zero additional implementation cost — trivial diagonal scaling

### Key Data Tables

**Mesh independence (R1=0.5, R2=1.5, quarter circle):**
| nel | w* | opt/bsp | geo/bsp | prec/bsp |
|-----|------|---------|---------|----------|
| 2 | 2.158 | 0.685 | 1.513 | 1.033 |
| 4 | 2.368 | 0.672 | 1.491 | 1.035 |
| 8 | 2.306 | 0.686 | 1.444 | 0.968 |
| 16 | 2.307 | 0.686 | 1.444 | (not tested) |

**Arc angle dependence (nel=4):**
| θ | w_geo | geo/bsp | prec/bsp |
|---|-------|---------|----------|
| 30° | 0.966 | 1.035 | 0.941 |
| 60° | 0.866 | 1.174 | 0.932 |
| 90° | 0.707 | 1.491 | 1.035 |
| 120° | 0.500 | 2.313 | 1.065 |

## GUI State
- R72: Single-pane command widget (unified terminal, no more two-textbox design)
- R68: saveas/print fixed for MATLAB calling convention
- R75: source() command added as alias for run()

## User Feedback Log
- Command window must operate like a single command prompt — FIXED R72
- Model user characterization in docs/forge_model_user.md (never summarize)

## V-Model Process
- Doc: C:\Users\njaxo\Documents\CLAUDE\documentation\commons_v_model_process.md
- Driving task milestones 10-13: investigation, novel result, thesis, platform limits
- **Milestone 11 (novel result) substantially complete** — problem/analysis/solution arc

## Known Issues
- p>2 on mapped geometry needs order elevation support
- 3D array display shows type info instead of values
- chr() builtin missing
- Plot window focus stealing in GUI
- No subfunction support in scripts (use separate .m files)

## Next Steps
1. **Thesis production**: Generate final publication-quality figures from Forge
2. **Generalization**: Test preconditioning on non-circular geometries (ellipse, multi-patch)
3. **1D verification**: Confirm α=2 scaling in 1D problem
4. **Forge features**: Tab completion, syntax highlighting, script file running from GUI
5. Build order elevation for p>2 on mapped geometry
