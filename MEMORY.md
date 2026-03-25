# Forge MEMORY.md — Session Continuity

## Current State (R67)
- **Last commit**: R67 — Multi-patch full annulus with DOF stitching
- **VPS**: ubuntu@15.204.8.77 (NOT 34.32.100.183 which is production)
- **Display**: :99 port 5900 (GUI), :98 port 5901 (testing)
- **Git branch**: master

## Recent Commits (R61-R67)
- R61: Poisson on NURBS-mapped quarter annulus (error 2.37e-02)
- R62: Eigenvalue problem on quarter annulus (λ₁=9.59)
- R63: String builtins (upper/lower/strcmp/sprintf/num2str) + help/which
- R64: Annulus gallery, view/clim/text plot functions
- R65: squeeze/permute/ndims builtins
- R66: Fix N-dimensional indexed assignment (A(i,j,k)=v)
- R67: Multi-patch full annulus (4 patches, DOF stitching, error 1.54e-02)

## Key Bug Fixes This Session
- J^{-T} vs J^{-1}: gradient transformation uses transposed inverse Jacobian
- N-D indexed assignment: changed len(args)==2 to len(args)>=2
- Source term: f=16r²-4(R1²+R2²) for u=(r²-R1²)(R2²-r²)

## TIGA Scripts Working
- All 1D/2D scripts from previous sessions
- tiga_mapped_solution.m — Poisson on quarter annulus, O(h²) convergence
- tiga_mapped_eigen.m — Eigenvalue problem on quarter annulus
- tiga_annulus_gallery.m — 2x3 subplot gallery (Poisson + modes)
- tiga_multi_patch.m — Full annulus from 4 patches, smooth across interfaces

## Known Issues
- 3D array display shows type info instead of values for large arrays
- NURBS quarter sphere has 0.56 radius error (single patch limitation)
- Plot window focus stealing in GUI (use headless for logic testing)

## Next Steps
- Time-dependent problems (heat equation on annulus)
- Higher-order elements (p=3, p=4)
- Error estimation and adaptive refinement on mapped geometry
- More surface plot features (contourf on mapped geometry)
