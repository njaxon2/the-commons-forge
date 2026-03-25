%% TIGA 2D Eigenmode Visualization
%  Compute and plot first 4 eigenmodes of -nabla^2 u = lambda*u on [0,1]^2
%  Tests: contourf, colorbar, meshgrid, 2D B-spline evaluation, reshape

clear;
fprintf('=== TIGA 2D Eigenmode Visualization ===\n\n');

%% Setup
p = 2;
num_el = 8;

% 1D knot vector
interior = linspace(0, 1, num_el + 1);
interior = interior(2:end-1);
Xi = [zeros(1, p+1), interior, ones(1, p+1)];
n1d = length(Xi) - p - 1;
n2d = n1d * n1d;

fprintf('  p=%d, %d elements/dir, %d total DOFs\n', p, num_el, n2d);

% Quadrature
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(Xi);
nel = length(knots_unique) - 1;

%% Assemble 1D K and M
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
                K1d(I, J) = K1d(I, J) + dN_dx(ii+1)*dN_dx(jj+1)*wt;
                M1d(I, J) = M1d(I, J) + N_val(ii+1)*N_val(jj+1)*wt;
            end
        end
    end
end

%% 2D system via Kronecker product
K2d = kron(K1d, M1d) + kron(M1d, K1d);
M2d = kron(M1d, M1d);

% BCs
is_free = ones(n1d, n1d);
is_free(1, :) = 0;
is_free(n1d, :) = 0;
is_free(:, 1) = 0;
is_free(:, n1d) = 0;
free_mask = reshape(is_free, n2d, 1);
free_dofs = find(free_mask);
n_free = length(free_dofs);

K_f = zeros(n_free, n_free);
M_f = zeros(n_free, n_free);
for i = 1:n_free
    for j = 1:n_free
        K_f(i, j) = K2d(free_dofs(i), free_dofs(j));
        M_f(i, j) = M2d(free_dofs(i), free_dofs(j));
    end
end

%% Solve eigenproblem
[V, D] = eig(K_f, M_f);
n_eig = n_free;
lambda_all = zeros(n_eig, 1);
for k = 1:n_eig
    lambda_all(k) = D(k, k);
end
[lambda_sorted, idx] = sort(lambda_all);

fprintf('  First 4 eigenvalues: %.4f, %.4f, %.4f, %.4f\n', ...
    lambda_sorted(1), lambda_sorted(2), lambda_sorted(3), lambda_sorted(4));

%% Evaluate eigenmodes on a grid
n_pts = 40;
x_grid = linspace(0, 1, n_pts);
y_grid = linspace(0, 1, n_pts);

% Precompute 1D basis at grid points
N_at_pts = zeros(n_pts, n1d);
for pt = 1:n_pts
    xi = x_grid(pt);
    if xi > 1 - 1e-14
        xi = 1 - 1e-14;
    end
    span = findspan(n1d-1, p, xi, Xi);
    N = basisfun(span, xi, p, Xi);
    for k = 0:p
        N_at_pts(pt, span - p + k + 1) = N(k + 1);
    end
end

%% Plot first 4 modes
figure(1);
for mode = 1:4
    % Get eigenvector
    phi = real(V(:, idx(mode)));
    phi = phi / max(abs(phi));

    % Full DOF vector
    phi_full = zeros(n2d, 1);
    phi_full(free_dofs) = phi;

    % Reshape to 2D grid of coefficients
    phi_2d = reshape(phi_full, n1d, n1d);

    % Evaluate on grid: u(x,y) = sum_I sum_J N_I(x)*N_J(y)*phi(I,J)
    Z = N_at_pts * phi_2d * N_at_pts';

    subplot(2, 2, mode);
    contourf(x_grid, y_grid, Z', 20);
    colorbar;
    title(sprintf('Mode %d (lambda=%.2f)', mode, lambda_sorted(mode)));
    xlabel('x');
    ylabel('y');
    axis equal;
end
drawnow;

fprintf('\n=== 2D Eigenmode Visualization Complete ===\n');
