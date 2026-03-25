%% TIGA 2D Laplacian Eigenvalue Problem
%  Solve -nabla^2 u = lambda*u on [0,1]^2 with u=0 on boundary
%  Exact eigenvalues: lambda = pi^2*(m^2 + n^2) for mode (m,n)
%  Tests: 2D tensor product assembly, sparse-like operations, contourf, colorbar

clear;
fprintf('=== TIGA 2D Laplacian Eigenvalues ===\n\n');

%% Setup
p = 2;
num_el = 6;  % elements per direction

% 1D knot vector
interior = linspace(0, 1, num_el + 1);
interior = interior(2:end-1);
Xi = [zeros(1, p+1), interior, ones(1, p+1)];
n1d = length(Xi) - p - 1;

fprintf('  p=%d, %d elements/dir, %d 1D DOFs, %d 2D DOFs\n', ...
    p, num_el, n1d, n1d*n1d);

% Gauss quadrature
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(Xi);
nel = length(knots_unique) - 1;

%% Assemble 1D stiffness and mass matrices
K1d = zeros(n1d, n1d);
M1d = zeros(n1d, n1d);

for e = 1:nel
    xi_a = knots_unique(e);
    xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14
        continue;
    end
    J_xi = (xi_b - xi_a) / 2;

    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n1d-1, p, xi, Xi);
        ders = derbasisfun(span, xi, p, 1, Xi);
        N_val = ders(1, :);
        dN_dx = ders(2, :);
        wt = gw(q) * J_xi;

        for ii = 0:p
            I = span - p + ii + 1;
            for jj = 0:p
                J = span - p + jj + 1;
                K1d(I, J) = K1d(I, J) + dN_dx(ii+1) * dN_dx(jj+1) * wt;
                M1d(I, J) = M1d(I, J) + N_val(ii+1) * N_val(jj+1) * wt;
            end
        end
    end
end

%% Build 2D matrices via tensor product: K2d = K1d x M1d + M1d x K1d
% Using kron (Kronecker product)
n2d = n1d * n1d;
K2d = kron(K1d, M1d) + kron(M1d, K1d);
M2d = kron(M1d, M1d);

fprintf('  2D system size: %d x %d\n', n2d, n2d);

%% Apply boundary conditions
% Boundary DOFs: first and last in each direction
is_free = ones(n1d, n1d);
is_free(1, :) = 0;
is_free(n1d, :) = 0;
is_free(:, 1) = 0;
is_free(:, n1d) = 0;

free_mask = reshape(is_free, n2d, 1);
free_dofs = find(free_mask);
n_free = length(free_dofs);
fprintf('  Free DOFs: %d (after BCs)\n', n_free);

K_f = zeros(n_free, n_free);
M_f = zeros(n_free, n_free);
for i = 1:n_free
    for j = 1:n_free
        K_f(i, j) = K2d(free_dofs(i), free_dofs(j));
        M_f(i, j) = M2d(free_dofs(i), free_dofs(j));
    end
end

%% Solve eigenvalue problem
lambda_vals = eig(K_f, M_f);
lambda_vals = sort(lambda_vals);

%% Compare with exact eigenvalues
fprintf('\n  First 8 eigenvalues:\n');
fprintf('  Mode | m,n | Computed    | Exact       | Rel Error\n');
fprintf('  -----|-----|-------------|-------------|----------\n');

% Exact eigenvalues sorted: pi^2*(m^2+n^2), m,n=1,2,...
exact_pairs = [];
for m = 1:5
    for nn = 1:5
        exact_pairs = [exact_pairs; pi^2*(m^2 + nn^2), m, nn];
    end
end
[exact_sorted, sort_idx] = sort(exact_pairs(:, 1));

for k = 1:min(8, n_free)
    exact_val = exact_sorted(k);
    m_mode = exact_pairs(sort_idx(k), 2);
    nn_mode = exact_pairs(sort_idx(k), 3);
    rel_err = abs(lambda_vals(k) - exact_val) / exact_val;
    fprintf('  %4d | %d,%d | %11.4f | %11.4f | %.2e\n', ...
        k, m_mode, nn_mode, lambda_vals(k), exact_val, rel_err);
end

fprintf('\n=== 2D Laplacian Eigenvalue Analysis Complete ===\n');
