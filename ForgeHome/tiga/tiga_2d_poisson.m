%% TIGA 2D Poisson IGA Solver
%  Solve -laplacian(u) = f on [0,1]^2 using tensor product B-splines
%  Exact solution: u(x,y) = sin(pi*x)*sin(pi*y)
%  Source: f(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y)
%  BCs: u = 0 on boundary

clear;
fprintf('=== TIGA 2D Poisson Solver ===\n');

%% Problem setup
p = 2;  % quadratic B-splines
Xi = [0 0 0 0.5 1 1 1];  % knot vector in each direction
n = length(Xi) - p - 1;   % basis functions per direction
N_dof = n * n;             % total DOFs

fprintf('Degree: p = %d\n', p);
fprintf('DOFs per direction: n = %d\n', n);
fprintf('Total DOFs: %d\n', N_dof);

%% Gauss quadrature
nqp = p + 1;
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(Xi);
num_el_1d = length(knots_unique) - 1;
num_el = num_el_1d * num_el_1d;

fprintf('Elements per direction: %d\n', num_el_1d);
fprintf('Total elements: %d\n', num_el);
fprintf('Quad points per element: %d x %d = %d\n', nqp, nqp, nqp*nqp);

%% Assemble global stiffness and load
K = zeros(N_dof, N_dof);
F = zeros(N_dof, 1);

for ex = 1:num_el_1d
    xi_a = knots_unique(ex);
    xi_b = knots_unique(ex + 1);
    if xi_b - xi_a < 1e-14
        continue;
    end
    Jx = (xi_b - xi_a) / 2;

    for ey = 1:num_el_1d
        eta_a = knots_unique(ey);
        eta_b = knots_unique(ey + 1);
        if eta_b - eta_a < 1e-14
            continue;
        end
        Jy = (eta_b - eta_a) / 2;

        for qx = 1:nqp
            xi = (xi_a + xi_b) / 2 + Jx * gp(qx);
            span_x = findspan(n-1, p, xi, Xi);
            ders_x = derbasisfun(span_x, xi, p, 1, Xi);
            Nx = ders_x(1, :);
            dNx = ders_x(2, :);

            for qy = 1:nqp
                eta = (eta_a + eta_b) / 2 + Jy * gp(qy);
                span_y = findspan(n-1, p, eta, Xi);
                ders_y = derbasisfun(span_y, eta, p, 1, Xi);
                Ny = ders_y(1, :);
                dNy = ders_y(2, :);

                % Physical coordinates (identity mapping for [0,1]^2)
                x_phys = xi;
                y_phys = eta;

                % Source term
                f_val = 2 * pi^2 * sin(pi * x_phys) * sin(pi * y_phys);

                % Jacobian
                J_total = Jx * Jy;
                wt = gw(qx) * gw(qy) * J_total;

                % Assembly using tensor product
                for ii = 0:p
                    I_x = span_x - p + ii + 1;
                    for jj = 0:p
                        I_y = span_y - p + jj + 1;
                        I_glob = (I_x - 1) * n + I_y;

                        % Load vector
                        F(I_glob) = F(I_glob) + Nx(ii+1) * Ny(jj+1) * f_val * wt;

                        % Stiffness matrix
                        for kk = 0:p
                            J_x = span_x - p + kk + 1;
                            for ll = 0:p
                                J_y = span_y - p + ll + 1;
                                J_glob = (J_x - 1) * n + J_y;

                                K(I_glob, J_glob) = K(I_glob, J_glob) + ...
                                    (dNx(ii+1) * Ny(jj+1) * dNx(kk+1) * Ny(ll+1) + ...
                                     Nx(ii+1) * dNy(jj+1) * Nx(kk+1) * dNy(ll+1)) * wt;
                            end
                        end
                    end
                end
            end
        end
    end
end

fprintf('\nK symmetry error: %e\n', norm(K - K', 'fro'));
fprintf('K size: %d x %d\n', size(K, 1), size(K, 2));

%% Apply boundary conditions
% Identify boundary DOFs: any DOF where i=1, i=n, j=1, or j=n
bc_dofs = [];
free_dofs = [];
for i = 1:n
    for j = 1:n
        glob = (i - 1) * n + j;
        if i == 1 || i == n || j == 1 || j == n
            bc_dofs = [bc_dofs, glob];
        else
            free_dofs = [free_dofs, glob];
        end
    end
end

fprintf('Boundary DOFs: %d\n', length(bc_dofs));
fprintf('Free DOFs: %d\n', length(free_dofs));

%% Solve
d = zeros(N_dof, 1);
K_free = K(free_dofs, free_dofs);
F_free = F(free_dofs);
d(free_dofs) = K_free \ F_free;

fprintf('\nSolution computed.\n');
fprintf('Max solution value: %f\n', max(d));
fprintf('Expected max (1/pi^2 ~ 0.1013): %f\n', 1/pi^2);

%% Evaluate and compute error
num_eval = 21;
xi_eval = linspace(0, 1, num_eval);
u_h = zeros(num_eval, num_eval);
u_exact = zeros(num_eval, num_eval);
x_grid = zeros(num_eval, num_eval);
y_grid = zeros(num_eval, num_eval);

for ix = 1:num_eval
    xi = xi_eval(ix);
    if xi >= 1
        xi = 1 - 1e-10;
    end
    span_x = findspan(n-1, p, xi, Xi);
    Nx = basisfun(span_x, xi, p, Xi);

    for iy = 1:num_eval
        eta = xi_eval(iy);
        if eta >= 1
            eta = 1 - 1e-10;
        end
        span_y = findspan(n-1, p, eta, Xi);
        Ny = basisfun(span_y, eta, p, Xi);

        u_val = 0;
        for ii = 0:p
            I_x = span_x - p + ii + 1;
            for jj = 0:p
                I_y = span_y - p + jj + 1;
                I_glob = (I_x - 1) * n + I_y;
                u_val = u_val + Nx(ii+1) * Ny(jj+1) * d(I_glob);
            end
        end

        x_grid(ix, iy) = xi;
        y_grid(ix, iy) = eta;
        u_h(ix, iy) = u_val;
        u_exact(ix, iy) = sin(pi * xi) * sin(pi * eta);
    end
end

err = abs(u_h - u_exact);
L_inf = max(max(err));
L2_approx = sqrt(sum(sum(err.^2)) / (num_eval * num_eval));

fprintf('\nL_inf error: %e\n', L_inf);
fprintf('L2 error (approx): %e\n', L2_approx);

%% Plot
figure(1);
surf(x_grid, y_grid, u_h);
title('IGA Solution u_h(x,y)');
xlabel('x');
ylabel('y');
zlabel('u');
drawnow;

figure(2);
surf(x_grid, y_grid, u_exact);
title('Exact Solution sin(pi*x)*sin(pi*y)');
xlabel('x');
ylabel('y');
zlabel('u');
drawnow;

figure(3);
surf(x_grid, y_grid, err);
title('Pointwise Error |u_h - u_{exact}|');
xlabel('x');
ylabel('y');
zlabel('Error');
drawnow;

fprintf('\n=== TIGA 2D Poisson Complete ===\n');
