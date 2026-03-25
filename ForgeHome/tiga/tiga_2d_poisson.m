%% TIGA 2D Poisson Solver
%  Solve -nabla^2 u = f on [0,1]x[0,1] with u=0 on boundary
%  Using tensor-product B-splines
%  Tests: 2D assembly, Kronecker product, kron, surf plot

clear;
fprintf('=== TIGA 2D Poisson Solver ===\n\n');

% Problem: -u_xx - u_yy = 2*pi^2 * sin(pi*x)*sin(pi*y)
% Exact: u(x,y) = sin(pi*x)*sin(pi*y)

p = 2;
nel_1d = 6;

% Build 1D knot vector
interior = linspace(0, 1, nel_1d + 1);
interior = interior(2:end-1);
Xi = [zeros(1, p+1), interior, ones(1, p+1)];
n_1d = length(Xi) - p - 1;

fprintf('  1D: p=%d, nel=%d, n=%d, len(Xi)=%d\n', p, nel_1d, n_1d, length(Xi));
fprintf('  2D: total DOFs = %d x %d = %d\n', n_1d, n_1d, n_1d*n_1d);

% Quadrature
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(Xi);
nel = length(knots_unique) - 1;

%% Assemble 1D stiffness and mass matrices
K1 = zeros(n_1d, n_1d);
M1 = zeros(n_1d, n_1d);

for e = 1:nel
    xi_a = knots_unique(e);
    xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14
        continue;
    end
    J_xi = (xi_b - xi_a) / 2;
    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n_1d-1, p, xi, Xi);
        ders = derbasisfun(span, xi, p, 1, Xi);
        N_val = ders(1, :);
        dN_dx = ders(2, :);
        wt = gw(q) * J_xi;
        for ii = 0:p
            I = span - p + ii + 1;
            for jj = 0:p
                J = span - p + jj + 1;
                K1(I, J) = K1(I, J) + dN_dx(ii+1)*dN_dx(jj+1)*wt;
                M1(I, J) = M1(I, J) + N_val(ii+1)*N_val(jj+1)*wt;
            end
        end
    end
end

%% 2D stiffness via Kronecker products
% K_2D = K1 kron M1 + M1 kron K1
n_2d = n_1d * n_1d;
K2 = kron(K1, M1) + kron(M1, K1);

fprintf('  Assembled 2D stiffness: %d x %d\n', n_2d, n_2d);

%% Assemble 2D load vector
f2 = zeros(n_2d, 1);

for ex = 1:nel
    xi_a = knots_unique(ex);
    xi_b = knots_unique(ex + 1);
    if xi_b - xi_a < 1e-14
        continue;
    end
    Jx = (xi_b - xi_a) / 2;
    for ey = 1:nel
        eta_a = knots_unique(ey);
        eta_b = knots_unique(ey + 1);
        if eta_b - eta_a < 1e-14
            continue;
        end
        Jy = (eta_b - eta_a) / 2;
        for qx = 1:nqp
            xi = (xi_a + xi_b) / 2 + Jx * gp(qx);
            span_x = findspan(n_1d-1, p, xi, Xi);
            Nx = basisfun(span_x, xi, p, Xi);
            for qy = 1:nqp
                eta = (eta_a + eta_b) / 2 + Jy * gp(qy);
                span_y = findspan(n_1d-1, p, eta, Xi);
                Ny = basisfun(span_y, eta, p, Xi);

                x = xi;
                y = eta;
                f_val = 2*pi^2 * sin(pi*x) * sin(pi*y);
                wt = gw(qx) * Jx * gw(qy) * Jy;

                for ii = 0:p
                    I = span_x - p + ii + 1;
                    for jj = 0:p
                        J = span_y - p + jj + 1;
                        glob = (J - 1) * n_1d + I;
                        f2(glob) = f2(glob) + Nx(ii+1)*Ny(jj+1)*f_val*wt;
                    end
                end
            end
        end
    end
end

%% Apply boundary conditions
% Boundary DOFs: first/last row/column in the 2D grid
bc_dofs = [];
for i = 1:n_1d
    bc_dofs = [bc_dofs, i];
    bc_dofs = [bc_dofs, (n_1d-1)*n_1d + i];
    bc_dofs = [bc_dofs, (i-1)*n_1d + 1];
    bc_dofs = [bc_dofs, (i-1)*n_1d + n_1d];
end
bc_dofs = unique(bc_dofs);

% Build free DOF list
free_dofs = [];
for d = 1:n_2d
    is_bc = 0;
    for b = 1:length(bc_dofs)
        if d == bc_dofs(b)
            is_bc = 1;
            break;
        end
    end
    if is_bc == 0
        free_dofs = [free_dofs, d];
    end
end

fprintf('  Boundary DOFs: %d, Free DOFs: %d\n', length(bc_dofs), length(free_dofs));

%% Solve
K_f = K2(free_dofs, free_dofs);
f_f = f2(free_dofs);
u_f = K_f \ f_f;
u2 = zeros(n_2d, 1);
u2(free_dofs) = u_f;

%% Compute L2 error
err_L2 = 0;
for ex = 1:nel
    xi_a = knots_unique(ex);
    xi_b = knots_unique(ex + 1);
    if xi_b - xi_a < 1e-14
        continue;
    end
    Jx = (xi_b - xi_a) / 2;
    for ey = 1:nel
        eta_a = knots_unique(ey);
        eta_b = knots_unique(ey + 1);
        if eta_b - eta_a < 1e-14
            continue;
        end
        Jy = (eta_b - eta_a) / 2;
        for qx = 1:nqp
            xi = (xi_a + xi_b) / 2 + Jx * gp(qx);
            span_x = findspan(n_1d-1, p, xi, Xi);
            Nx = basisfun(span_x, xi, p, Xi);
            for qy = 1:nqp
                eta = (eta_a + eta_b) / 2 + Jy * gp(qy);
                span_y = findspan(n_1d-1, p, eta, Xi);
                Ny = basisfun(span_y, eta, p, Xi);

                u_h = 0;
                for ii = 0:p
                    I = span_x - p + ii + 1;
                    for jj = 0:p
                        J = span_y - p + jj + 1;
                        glob = (J - 1) * n_1d + I;
                        u_h = u_h + Nx(ii+1)*Ny(jj+1)*u2(glob);
                    end
                end

                u_exact = sin(pi*xi)*sin(pi*eta);
                err_L2 = err_L2 + (u_h - u_exact)^2 * gw(qx)*Jx * gw(qy)*Jy;
            end
        end
    end
end
err_L2 = sqrt(err_L2);
fprintf('\n  L2 error = %.6e\n', err_L2);

%% Evaluate solution on grid for plotting
n_plot = 40;
x_plot = linspace(0, 1-1e-10, n_plot);
y_plot = linspace(0, 1-1e-10, n_plot);
Z_iga = zeros(n_plot, n_plot);
Z_exact = zeros(n_plot, n_plot);

for ix = 1:n_plot
    xi = x_plot(ix);
    span_x = findspan(n_1d-1, p, xi, Xi);
    Nx = basisfun(span_x, xi, p, Xi);
    for iy = 1:n_plot
        eta = y_plot(iy);
        span_y = findspan(n_1d-1, p, eta, Xi);
        Ny = basisfun(span_y, eta, p, Xi);

        u_h = 0;
        for ii = 0:p
            I = span_x - p + ii + 1;
            for jj = 0:p
                J = span_y - p + jj + 1;
                glob = (J - 1) * n_1d + I;
                u_h = u_h + Nx(ii+1)*Ny(jj+1)*u2(glob);
            end
        end
        Z_iga(iy, ix) = u_h;
        Z_exact(iy, ix) = sin(pi*xi)*sin(pi*eta);
    end
end

%% Plot
figure(1);
subplot(1, 3, 1);
surf(x_plot, y_plot, Z_iga);
title('IGA Solution');
xlabel('x'); ylabel('y'); zlabel('u');
colorbar;

subplot(1, 3, 2);
surf(x_plot, y_plot, Z_exact);
title('Exact Solution');
xlabel('x'); ylabel('y'); zlabel('u');
colorbar;

subplot(1, 3, 3);
surf(x_plot, y_plot, abs(Z_iga - Z_exact));
title('Absolute Error');
xlabel('x'); ylabel('y'); zlabel('|error|');
colorbar;

drawnow;

fprintf('\n=== 2D Poisson Complete ===\n');
