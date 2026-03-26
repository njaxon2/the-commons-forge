# Forge MEMORY.md — Session Continuity

## Current State (R126)
- **Last commit**: R126 — 950 registered functions
- **VPS**: ubuntu@15.204.8.77 (NOT 34.32.100.183 which is production)
- **Display**: :99 port 5900 (GUI), :98 port 5901 (testing)
- **Git branch**: master
- **E2E tests**: 120/120 passing
- **Single-quote strings**: Work correctly (tested R115)
- **Cell {} assignment**: Fixed in R119

## Recent Commits (R119-R126)
- R119: Fix cell {} assignment, expand E2E to 108
- R120: sscanf/textscan, sylvester, validation, moving stats (928 funcs)
- R121: Array manipulation, special matrices, del2, divergence (933 funcs)
- R122: Improved display formatting — scientific notation, complex, format cmd
- R123: Variable editor dialog — double-click workspace to inspect
- R124: Tilde output placeholder [~, x] = func(), 120 E2E tests
- R125: Statistics toolkit — 21 functions (947 total)
- R126: Geometry, strings, 950 functions milestone

## Earlier Commits (R103-R118)
- R103-R103b: Thesis Figs 3-6 — basis functions, solution, weight-conditioning
- R104: xline, yline, meshgrid (906 funcs)
- R105-R108: eval/feval, sprintf/error/warning, class, char concat fix
- R109-R110: Thesis Figs 9-10 + fast TIGA 2D assembly (225x speedup)
- R111-R112: Weight-conditioning demo + E2E tests to 76
- R113-R115: Type predicates, LA decompositions (schur, hess, expm, etc.)
- R116-R118: MEMORY update, Thesis Fig 13, help/date/time system

## Thesis Figures (15 total in thesis_figures/)
1-14: As listed in previous MEMORY.md
15. forge_gui_r122.png — Updated GUI screenshot with workspace

## Novel Finding: NURBS Weight-Conditioning (Verified R110-R111)
| nel | B-spline κ | NURBS κ | Extreme κ | NURBS/BS | Precond ratio |
|-----|-----------|---------|-----------|----------|----|
| 4   | 1.20e+01  | 1.79e+01| 1.00e+03  | 1.49     | 1.23 |
| 8   | 4.14e+01  | 5.98e+01| 1.86e+03  | 1.44     | 1.21 |
| 16  | 1.64e+02  | 2.37e+02| 6.70e+03  | 1.44     | 1.21 |
| 32  | 6.60e+02  | 9.53e+02| 2.69e+04  | 1.44     | 1.21 |

## Forge Capabilities (950 functions)
- **Core LA**: eig, svd, lu, qr, chol, schur, hess, balance, inv, det, norm, rank
- **Special LA**: expm, logm, funm, sqrtm, condest, eigs, sylvester
- **Statistics**: mean, std, var, cov, corrcoef, prctile, zscore, skewness, kurtosis
- **Stat tests**: ttest, ttest2, chi2gof, fitlm, regress, kmeans
- **Set ops**: unique, setdiff, union, intersect, ismember
- **Polynomials**: polyval, polyfit, roots, poly, conv, deconv
- **ODE**: ode45, ode23, ode15s
- **Optimization**: fzero, fminbnd, fminsearch, fsolve, integral
- **File I/O**: fopen/fclose/fprintf/fgets/fileread/dlmwrite/csvread
- **Strings**: sprintf, regexp, strsplit, strjoin, num2str, str2num, sscanf, textscan
- **Plotting**: plot, loglog, semilogx, semilogy, subplot, scatter, bar, stem,
                contour, contourf, surf, mesh, xline, yline, meshgrid, peaks
- **Types**: class, fieldnames, isfield, isstruct, iscell, ischar, isnumeric
- **Meta**: eval, feval, exist, which, nargin, nargout, help, lookfor
- **Array**: fliplr, flipud, rot90, circshift, squeeze, permute
- **Moving stats**: movmean, movsum, movstd
- **Geometry**: inpolygon, convhull, delaunay, voronoi, polyarea
- **Validation**: validateattributes, inputParser
- **TIGA**: findspan, basisfun, derbasisfun, gaussQuad, tiga_assemble_2d

## GUI Features
- Command widget with REPL, history, tab completion
- Multi-tab code editor with syntax highlighting
- File browser
- Workspace browser with double-click variable editor (R123)
- Plot system (matplotlib)
- Themes (dark/light)

## Known Issues (Fixed)
- ✅ Cell assignment via {} indexing (R119)
- ✅ Display formatting (R122)
- ✅ Tilde output placeholder (R124)
- ✅ ForgeChar unicode overflow in help (R119)

## Known Issues (Open)
- repmat with char input returns numeric codes (cosmetic)
- Interpreter slow for deeply nested loops (use tiga_assemble_2d for speed)
- No subfunction support in scripts
- No classdef/OOP support yet

## Next Steps
1. Continue thesis figure generation if needed
2. Performance optimization for nested loops
3. classdef/OOP support
4. Package system (+package directories)
5. More comprehensive E2E test coverage
