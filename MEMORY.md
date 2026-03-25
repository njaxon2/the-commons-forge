# Forge MEMORY.md — Session Continuity

## Current State (R61)
- **Last commit**: R61 — Poisson on NURBS-mapped quarter annulus
- **VPS**: ubuntu@15.204.8.77 (NOT 34.32.100.183 which is production)
- **Display**: :99 port 5900 (GUI), :98 port 5901 (testing)
- **Git branch**: master

## Recent Commits
- R54: Fix 2D elasticity body force, auto-call bare callable (t=toc)
- R55: Stress recovery with von Mises visualization
- R56: k-refinement study (C^(p-1) vs C^0)
- R57: Fix cd /path command-form parsing, add helper scripts
- R58: Major surface plot upgrade (surf, mesh, contourf, peaks, sombrero, etc.)
- R59: V-model iterative development process documentation
- R60: NURBS surface geometry (cylinder, sphere), 3D subplot fix
- R61: Poisson on NURBS-mapped quarter annulus (error 2.37e-02, O(h²) convergence)

## Key Bug Fixes
- J^{-T} vs J^{-1}: Inverse Jacobian for gradient transformation must use transpose
  - Wrong: inv_J12 = -dx_deta/detJ, inv_J21 = -dy_dxi/detJ
  - Right: inv_J12 = -dy_dxi/detJ, inv_J21 = -dx_deta/detJ
- Auto-call bare callable in assignment RHS (evaluator.py _exec_assign)
- cd /path command-form parsing (parser.py pre-check before expression parsing)
- 3D subplot preservation (_get_3d_ax keeps subplot geometry)

## TIGA Scripts (ForgeHome/tiga/)
Working:
- tiga_poisson1d.m — 1D Poisson, L2 error ~1e-06
- tiga_convergence.m — h-refinement convergence rates
- tiga_2d_poisson.m — 2D Poisson on unit square
- tiga_elasticity.m — 2D plane stress, L2 error 2.58e-04
- tiga_stress_recovery.m — Stress/strain from displacements
- tiga_k_refinement.m — IGA vs FEA continuity comparison
- tiga_nurbs_surface.m — Quarter cylinder (exact), quarter sphere
- tiga_mapped_solution.m — Poisson on quarter annulus, error 2.37e-02
- tiga_plot_showcase.m — 4-panel surface plot demo

## Known Issues
- help function not registered as standalone callable
- upper/lower string functions not registered
- NURBS quarter sphere has 0.56 radius error (single patch limitation)
- Plot window focus stealing in GUI (use headless for logic testing)

## Workflow
- Headless testing: python3 scripts/run_m.py "script_name"
- Save figures: use Agg backend, fig.savefig()
- GUI testing: DISPLAY=:99 with Forge app
- Fast command: scripts/forge_cmd.sh "command"
- Screenshot: scripts/forge_snap.sh

## Next Steps
- More NURBS-mapped geometries (full annulus, multi-patch)
- Surface plot enhancements (lighting, view angles, colormaps)
- Eigenvalue problems on mapped domains
- Register missing builtins (help, upper, lower)
