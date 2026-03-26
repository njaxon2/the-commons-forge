# Forge MEMORY.md — Session Continuity

## Current State (R115)
- **Last commit**: R115 — 917 registered functions
- **VPS**: ubuntu@15.204.8.77 (NOT 34.32.100.183 which is production)
- **Display**: :99 port 5900 (GUI), :98 port 5901 (testing)
- **Git branch**: master
- **E2E tests**: 76/76 passing
- **Single-quote strings**: Work correctly (tested R115)

## Recent Commits (R103-R115)
- R103: Thesis Fig 5 — multi-order h-convergence (p=2,3,4)
- R103b: Thesis Figs 3,4,6 — basis functions, solution, weight-conditioning
- R104: xline, yline, meshgrid (906 funcs)
- R105: eval, feval, nargin, nargout, strjoin (910 funcs)
- R106: Updated MEMORY.md
- R107: Thesis Figs 7,8 — p-refinement and k-refinement
- R108: sprintf, error, warning, assert, class + char concat fix (911 funcs)
- R109: Thesis Figs 9,10 — NURBS geometry + eigenvalue spectrum
- R110: Fast TIGA 2D assembly built-in (225x speedup)
- R111: TIGA weight-conditioning demo (0.8s from M-code)
- R112: E2E tests expanded to 76
- R113: Type predicates and introspection (913 funcs)
- R114: Thesis Figs 11,12 + GUI screenshot
- R115: schur, hess, balance, conv2, eigs fix, expm/logm/funm (917 funcs)

## Thesis Figures (12 total in thesis_figures/)
1. fig1_weight_tradeoff.png — Main tradeoff curve
2. fig2_preconditioning.png — Bar chart, mesh independence
3. fig3_arc_angle.png — Arc angle dependence
4. fig3_basis_functions.png — B-spline basis p=1,2,3
5. fig4_ellipse.png — Elliptical geometry
6. fig4_solution_comparison.png — IGA vs exact + error
7. fig5_h_convergence.png — Multi-order h-convergence
8. fig6_weight_conditioning.png — Novel 4-panel weight study
9. fig7_p_refinement.png — p-refinement exponential convergence
10. fig8_k_refinement.png — h vs p vs k strategies
11. fig9_nurbs_geometry.png — Control net + physical mesh
12. fig10_eigenspectrum.png — Eigenvalue spectra
13. fig11_weight_function.png — W(eta) analysis
14. fig12_summary.png — Comprehensive 4-panel summary

## Novel Finding: NURBS Weight-Conditioning

### Key Results (Verified R110-R111)
| nel | B-spline κ | NURBS κ | Extreme κ | NURBS/BS | Precond ratio |
|-----|-----------|---------|-----------|----------|--------------|
| 4 | 1.20e+01 | 1.79e+01 | 1.00e+03 | 1.49 | 1.23 |
| 8 | 4.14e+01 | 5.98e+01 | 1.86e+03 | 1.44 | 1.21 |
| 16 | 1.64e+02 | 2.37e+02 | 6.70e+03 | 1.44 | 1.21 |
| 32 | 6.60e+02 | 9.53e+02 | 2.69e+04 | 1.44 | 1.21 |

### h-Convergence (1D Poisson)
| p | Theory | Verified |
|---|--------|----------|
| 2 | O(h³) | ✓ ratio 8.0 |
| 3 | O(h⁴) | ✓ ratio 16.1 |
| 4 | O(h⁵) | ✓ ratio 31.3 |

## Forge Capabilities (917 functions)
- **Core LA**: eig, svd, lu, qr, chol, schur, hess, balance, inv, det, norm, rank
- **Special LA**: expm, logm, funm, sqrtm, condest, eigs
- **Set ops**: unique, setdiff, union, intersect, ismember
- **Polynomials**: polyval, polyfit, roots, poly, conv, deconv
- **ODE**: ode45, ode23, ode15s
- **Optimization**: fzero, fminbnd, fminsearch, fsolve, integral
- **File I/O**: fopen/fclose/fprintf/fgets/fileread/dlmwrite/csvread
- **Strings**: sprintf, regexp, strsplit, strjoin, num2str, str2num
- **Plotting**: plot, loglog, semilogx, semilogy, subplot, scatter, bar, stem,
                contour, contourf, surf, mesh, xline, yline, meshgrid, peaks
- **Types**: class, fieldnames, isfield, isstruct, iscell, ischar, isnumeric
- **Meta**: eval, feval, exist, which, nargin, nargout
- **TIGA**: findspan, basisfun, derbasisfun, gaussQuad, tiga_assemble_2d

## Known Issues
- Cell assignment via {} indexing has deeper evaluator bug
- repmat with char input returns numeric codes (cosmetic)
- Interpreter slow for deeply nested loops (use tiga_assemble_2d for speed)
- No subfunction support in scripts

## Next Steps
1. Continue thesis figure generation if needed
2. Build 2D solution visualization (annulus contour plots)
3. Performance optimization for nested loops
4. Variable editor in GUI
5. More comprehensive E2E test coverage
