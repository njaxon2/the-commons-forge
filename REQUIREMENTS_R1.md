# Forge Requirements - Round 1 (Validation-Driven)

## R01 - Semicolon Output Suppression
Statements ending with semicolon SHALL suppress display output.

## R02 - MATLAB-Style Output Formatting
Matrix display SHALL use right-aligned columns, no numpy brackets.

## R03 - Character Array Display
Single-quoted strings SHALL display as text, not ASCII codes.

## R04 - Slash in Single-Quoted Strings
Strings containing / SHALL parse correctly.

## R05 - Command-Style Syntax
Bare commands (who, whos, hold on, axis equal) SHALL invoke, not return refs.

## R06 - Float-to-Int Coercion for Shape Args
reshape, zeros, ones SHALL coerce float args to int.

## R07 - Struct Auto-Creation
s.field = val WHERE s is undefined SHALL create a struct.

## R08 - Sparse Matrix Construction
sparse(m,n) SHALL return zero sparse. sparse(I,J,V,m,n) SHALL work with ForgeArrays.

## R09 - Indexed Assignment with RHS Expressions
K(i,j) = K(i,j) + val SHALL resolve all RHS variables correctly.

## R10 - eig() Return Order
[V,D] = eig(A): V=eigenvectors, D=eigenvalues (diagonal).

## R11 - Nested Function Calls
max(abs([-3 5 -7 2])) SHALL return 7.

## R12 - Multi-Output Functions
[m,idx] = max(v) SHALL return value and index. [r,c] = find(A) row/col vectors.

## R13 - .m File Auto-Discovery
Undefined functions SHALL be searched on path directories for .m files.

## R14 - Script Execution
run(path) SHALL execute .m scripts in current workspace.

## R15 - Missing Core Functions
tic/toc, dot, sub2ind/ind2sub, delaunay, save/load, etc.

## R16 - Figure Window Display
plot() SHALL produce visible figures. Matplotlib SHALL integrate with Qt.
