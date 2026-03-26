# Forge MEMORY.md — Session Continuity

## Current State (R72)
- **Last commit**: R72 — Weight-conditioning anatomy + single-pane command widget
- **VPS**: ubuntu@15.204.8.77 (NOT 34.32.100.183 which is production)
- **Display**: :99 port 5900 (GUI), :98 port 5901 (testing)
- **Git branch**: master

## Recent Commits (R68-R72)
- R68: Annulus h-convergence study, fix saveas/print, improve GUI layout
- R69: Condition number study on mapped annulus, h-p convergence attempt
- R70: NURBS weight effect on conditioning — spectral analysis
- R71: Systematic NURBS weight sweep — accuracy vs conditioning tradeoff
- R72: Weight-conditioning anatomy study + single-pane command widget

## Novel Finding: NURBS Weight-Conditioning Tradeoff
**This is the main result being developed toward the driving task thesis.**

Key results from tiga_weight_anatomy.m (R72):
1. **Optimal conditioning weight w* ≈ 2.31** (converged as h→0), found via golden section
2. **Mesh-independent conditioning ratios**:
   - cond(w*)/cond(w=1) → 0.686 (optimal weight 31% better than B-spline)
   - cond(1/√2)/cond(w=1) → 1.444 (geometric weight 44% WORSE than B-spline)
3. **w* depends on geometry (R2/R1)**: peaks at ~2.37 for R2/R1=3, decreases for thinner/thicker
4. **Error minimum at w=1/√2** (exact circle): err=0.046 vs err=0.64 at w=1
5. **Fundamental tradeoff**: geometric accuracy optimal at w=1/√2, conditioning optimal at w≈2.3

Data (Part 3 — mesh independence, R1=0.5, R2=1.5):
| nel | w* | opt/bsp | geo/bsp |
|-----|------|---------|---------|
| 2 | 2.158 | 0.6851 | 1.5129 |
| 4 | 2.368 | 0.6715 | 1.4906 |
| 8 | 2.306 | 0.6859 | 1.4443 |
| 16 | 2.307 | 0.6857 | 1.4444 |

## GUI Improvements
- R68: saveas/print fixed for MATLAB calling convention (handle, filename, format)
- R68: Dark terminal styling, monospace font, minimum height for command widget
- R72: **Single-pane command widget** — unified terminal replacing two-textbox design

## User Feedback Log
- Command window must operate like a single command prompt (not two text boxes) — FIXED R72
- Model user characterization stored in docs/forge_model_user.md (never summarize)

## V-Model Process
- Documentation at C:\Users\njaxo\Documents\CLAUDE\documentation\commons_v_model_process.md
- Current phase: Exploration → novel finding development
- Driving task milestones 10-13: investigation, novel result, thesis, platform limits

## Key Bug Fixes (R68-R72 sessions)
- saveas() ForgeArray truth value ambiguous — rewrote with *args
- print() same issue — rewrote with *args
- Command window collapsed to tiny sliver — added minimum height + resizeDocks
- QSettings overriding fresh layout — clear ~/.config/Forge
- p>2 NURBS quarter circle parametrization wrong — NOT YET FIXED (needs order elevation)

## TIGA Scripts Working
- All previous scripts still working
- tiga_annulus_convergence.m — h-convergence on NURBS annulus, rates ~2.5
- tiga_condition_study.m — cond vs h (O(h^-2)), cond vs aspect ratio
- tiga_weight_conditioning.m — B-spline vs NURBS vs extreme weight comparison
- tiga_weight_sweep.m — systematic w=0.01..10 sweep with 2x2 subplot
- tiga_weight_anatomy.m — fine sweep + golden section + geometry/mesh independence
- compute_annulus_cond.m — reusable function for weight studies
- tiga_hp_convergence.m — p=2 works, p=3/4 broken (order elevation needed)

## Known Issues
- p>2 on mapped geometry needs order elevation support
- 3D array display shows type info instead of values
- chr() builtin missing
- Plot window focus stealing in GUI
- source() command not implemented (use addpath + function files)
- No subfunction support in scripts (use separate .m files)

## Next Steps for Novel Finding
1. Investigate analytical form of w* — is it related to curvature, arc length?
2. Test on half-circle, full circle, ellipse to check universality
3. Derive the conditioning ratio analytically from the NURBS weight function
4. Produce thesis document with Forge-exported plots
5. Build order elevation support for p>2 studies
