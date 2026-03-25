%% TIGA IGA on Mapped Domain
%  Solve -laplacian(u) = f on a NURBS-mapped domain
%  Using isoparametric concept: same NURBS basis for geometry and solution
%  This is the core of IGA from tigaSPM13.pdf

clear;
fprintf('=== TIGA IGA Mapped Domain (1D) ===\n');

%% Setup: Solve on [0, L] mapped from [0,1] via NURBS geometry
p = 2;  % quadratic
Xi = [0 0 0 0.25 0.5 0.75 1 1 1];  % knot vector
n = length(Xi) - p - 1;  % number of basis functions

% Control points define the geometry mapping
% Map [0,1] -> [0, pi] (for exact solution u(x) = sin(x))
L = pi;
% Uniformly spaced control points in physical space
ctrl_pts = linspace(0, L, n)';

fprintf('Degree: p = %d\n', p);
fprintf('Basis functions: n = %d\n', n);
fprintf('Domain: [0, %f]\n', L);
fprintf('Control points:\n');
disp(ctrl_pts');

%% Evaluate geometry mapping at test points
nqp = p + 1;  % quadrature points per element
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(Xi);
num_el = length(knots_unique) - 1;

fprintf('\nElements: %d\n', num_el);
fprintf('Quadrature points per element: %d\n', nqp);

%% Assemble stiffness matrix and load vector
% Exact solution: u(x) = sin(x)
% Source: f(x) = sin(x) (since -u'' = sin(x))
% BCs: u(0) = 0, u(pi) = 0

K = zeros(n, n);
F = zeros(n, 1);
M = zeros(n, n);  % mass matrix for L2 error computation

for e = 1:num_el
    xi_a = knots_unique(e);
    xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14
        continue;
    end
    J_xi = (xi_b - xi_a) / 2;

    for q = 1:nqp
        % Parametric coordinate
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);

        % Basis functions and derivatives
        span = findspan(n-1, p, xi, Xi);
        ders = derbasisfun(span, xi, p, 1, Xi);
        N_val = ders(1, :);      % N_i(xi)
        dN_dxi = ders(2, :);     % dN_i/dxi

        % Geometry mapping: x(xi) = sum N_i(xi) * P_i
        x_phys = 0;
        dx_dxi = 0;
        for i_loc = 0:p
            i_glob = span - p + i_loc + 1;
            x_phys = x_phys + N_val(i_loc+1) * ctrl_pts(i_glob);
            dx_dxi = dx_dxi + dN_dxi(i_loc+1) * ctrl_pts(i_glob);
        end

        % Jacobian of geometry mapping
        J_geom = dx_dxi;

        % Derivatives in physical space: dN/dx = dN/dxi * 1/J_geom
        dN_dx = dN_dxi / J_geom;

        % Total Jacobian for integration
        J_total = abs(J_geom) * J_xi;

        % Source term
        f_val = sin(x_phys);

        % Assembly
        for ii = 0:p
            I = span - p + ii + 1;
            F(I) = F(I) + N_val(ii+1) * f_val * J_total * gw(q);
            for jj = 0:p
                J = span - p + jj + 1;
                K(I, J) = K(I, J) + dN_dx(ii+1) * dN_dx(jj+1) * J_total * gw(q);
                M(I, J) = M(I, J) + N_val(ii+1) * N_val(jj+1) * J_total * gw(q);
            end
        end
    end
end

fprintf('\nK symmetry error: %e\n', norm(K - K', 'fro'));
fprintf('M symmetry error: %e\n', norm(M - M', 'fro'));
fprintf('K condition number: %e\n', cond(K));

%% Apply BCs: u(0) = 0, u(pi) = sin(pi) = 0
bc_dofs = [1 n];
free = 2:n-1;

K_free = K(free, free);
F_free = F(free) - K(free, bc_dofs) * [0; 0];

fprintf('\nFree DOFs: %d\n', length(free));

%% Solve
d = zeros(n, 1);
d(free) = K_free \ F_free;

fprintf('Solution DOFs:\n');
disp(d');

%% Evaluate solution and compute error
num_eval = 201;
u_eval = linspace(0, 1, num_eval);
x_pts = zeros(1, num_eval);
u_h = zeros(1, num_eval);
u_exact = zeros(1, num_eval);

for k = 1:num_eval
    xi = u_eval(k);
    if xi >= 1
        xi = 1 - 1e-10;
    end
    span = findspan(n-1, p, xi, Xi);
    N = basisfun(span, xi, p, Xi);

    % Compute physical coordinate
    x_k = 0;
    u_k = 0;
    for j = 0:p
        idx = span - p + j + 1;
        x_k = x_k + N(j+1) * ctrl_pts(idx);
        u_k = u_k + N(j+1) * d(idx);
    end
    x_pts(k) = x_k;
    u_h(k) = u_k;
    u_exact(k) = sin(x_k);
end

% Error measures
err_vec = abs(u_h - u_exact);
L_inf = max(err_vec);
L2_approx = sqrt(sum(err_vec.^2) / num_eval);

fprintf('\nL_inf error: %e\n', L_inf);
fprintf('L2 error (approx): %e\n', L2_approx);

%% Plot results
figure(1);
plot(x_pts, u_exact, 'b-', x_pts, u_h, 'r--');
legend('Exact sin(x)', 'IGA solution');
title('IGA on Mapped Domain: -u'''' = sin(x)');
xlabel('x');
ylabel('u(x)');
grid on;
drawnow;

figure(2);
plot(x_pts, err_vec, 'k-');
title('Pointwise Error |u_h - u_{exact}|');
xlabel('x');
ylabel('Error');
grid on;
drawnow;

%% Geometry mapping verification
fprintf('\n--- Geometry Mapping Check ---\n');
x_check = zeros(1, 11);
xi_check = linspace(0, 1, 11);
for k = 1:11
    xi = xi_check(k);
    if xi >= 1
        xi = 1 - 1e-10;
    end
    span = findspan(n-1, p, xi, Xi);
    N = basisfun(span, xi, p, Xi);
    x_val = 0;
    for j = 0:p
        idx = span - p + j + 1;
        x_val = x_val + N(j+1) * ctrl_pts(idx);
    end
    x_check(k) = x_val;
end
fprintf('xi:  ');
fprintf('%.2f ', xi_check);
fprintf('\n');
fprintf('x:   ');
fprintf('%.4f ', x_check);
fprintf('\n');
fprintf('Expected spacing: ~%.4f\n', L / 10);

fprintf('\n=== TIGA IGA Mapped Complete ===\n');
