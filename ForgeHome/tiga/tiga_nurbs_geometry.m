%% TIGA NURBS Geometry Mapping
%  Demonstrate NURBS-based geometry representation
%  Map parameter space to physical space using weighted basis functions
%  Tests: NURBS weights, geometry mapping, Jacobian, physical-space assembly

clear;
fprintf('=== TIGA NURBS Geometry Mapping ===\n\n');

%% Define a quarter-circle geometry using NURBS
% Control points for quarter circle (radius=1)
% In NURBS, a circle arc needs weights
p = 2;
Xi = [0 0 0 1 1 1];
n = length(Xi) - p - 1;  % 3 control points

% Control points: (x, y)
Px = [1.0, 1.0, 0.0];
Py = [0.0, 1.0, 1.0];

% NURBS weights: w = [1, 1/sqrt(2), 1] for exact circle
w = [1.0, 1.0/sqrt(2), 1.0];

fprintf('  Quarter-circle NURBS:\n');
fprintf('    p = %d, n = %d\n', p, n);
fprintf('    Control points:\n');
for i = 1:n
    fprintf('      P(%d) = (%.4f, %.4f), w = %.4f\n', i, Px(i), Py(i), w(i));
end

%% Evaluate NURBS curve
n_eval = 100;
xi_vals = linspace(0, 1 - 1e-10, n_eval);
x_curve = zeros(1, n_eval);
y_curve = zeros(1, n_eval);

for t = 1:n_eval
    xi = xi_vals(t);
    span = findspan(n - 1, p, xi, Xi);
    N = basisfun(span, xi, p, Xi);

    % NURBS: R_i = N_i * w_i / sum(N_j * w_j)
    W = 0;
    for i = 0:p
        idx = span - p + i + 1;
        W = W + N(i + 1) * w(idx);
    end

    x = 0; y = 0;
    for i = 0:p
        idx = span - p + i + 1;
        R = N(i + 1) * w(idx) / W;
        x = x + R * Px(idx);
        y = y + R * Py(idx);
    end
    x_curve(t) = x;
    y_curve(t) = y;
end

% Check: points should lie on unit circle
r_vals = sqrt(x_curve.^2 + y_curve.^2);
max_r_err = max(abs(r_vals - 1.0));
fprintf('\n  Max radius error: %.2e (should be ~machine eps)\n', max_r_err);

%% Solve Laplace on quarter annulus (r=0.5 to r=1)
% Using NURBS geometry mapping
% -nabla^2 u = 0 with u(r=0.5) = ln(0.5), u(r=1) = 0
% Exact: u(r) = ln(r) in polar, u(x,y) = ln(sqrt(x^2+y^2))

p_sol = 2;
nel_r = 8;  % Elements in radial direction

% Radial knot vector
interior_r = linspace(0, 1, nel_r + 1);
interior_r = interior_r(2:end - 1);
Xi_r = [zeros(1, p_sol + 1), interior_r, ones(1, p_sol + 1)];
n_r = length(Xi_r) - p_sol - 1;

% Control points: radial mapping from parameter to physical
% r = 0.5 at xi=0, r = 1 at xi=1
% Linearly space control points in radius
r_ctrl = linspace(0.5, 1.0, n_r);

fprintf('\n  Laplace on quarter annulus:\n');
fprintf('    Radial: p=%d, nel=%d, n=%d\n', p_sol, nel_r, n_r);

% 1D assembly in parameter space
nqp = p_sol + 2;
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(Xi_r);
nel = length(knots_unique) - 1;

% For 1D radial Laplace in mapped coords:
% The mapped problem involves the Jacobian dr/dxi
K = zeros(n_r, n_r);
f = zeros(n_r, 1);

for e = 1:nel
    xi_a = knots_unique(e);
    xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14
        continue;
    end
    J_xi = (xi_b - xi_a) / 2;
    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n_r - 1, p_sol, xi, Xi_r);
        ders = derbasisfun(span, xi, p_sol, 1, Xi_r);
        N_val = ders(1, :);
        dN_dxi = ders(2, :);

        % Map to physical radius
        r = 0; dr_dxi = 0;
        for i = 0:p_sol
            idx = span - p_sol + i + 1;
            r = r + N_val(i + 1) * r_ctrl(idx);
            dr_dxi = dr_dxi + dN_dxi(i + 1) * r_ctrl(idx);
        end

        % dN/dr = dN/dxi * dxi/dr = dN/dxi / (dr/dxi)
        dN_dr = dN_dxi / dr_dxi;

        % Radial Laplacian in 1D: -(r * du/dr)' = 0
        % Weak form: integral r * du/dr * dv/dr dr = 0
        wt = gw(q) * J_xi * abs(dr_dxi);

        for ii = 0:p_sol
            I = span - p_sol + ii + 1;
            for jj = 0:p_sol
                J = span - p_sol + jj + 1;
                K(I, J) = K(I, J) + r * dN_dr(ii + 1) * dN_dr(jj + 1) * wt;
            end
        end
    end
end

% BCs: u(1) = ln(0.5) at first DOF, u(end) = 0 at last DOF
bc_vals = [log(0.5), 0];
free = 2:n_r - 1;

% Modify RHS for BCs
f_mod = f(free) - K(free, 1) * bc_vals(1) - K(free, n_r) * bc_vals(2);
K_f = K(free, free);
u_f = K_f \ f_mod;
u_sol = zeros(n_r, 1);
u_sol(1) = bc_vals(1);
u_sol(free) = u_f;
u_sol(n_r) = bc_vals(2);

% Evaluate and compute error
n_plot = 200;
xi_plot = linspace(0, 1 - 1e-10, n_plot);
r_plot = zeros(1, n_plot);
u_plot = zeros(1, n_plot);
u_exact_plot = zeros(1, n_plot);

for t = 1:n_plot
    xi = xi_plot(t);
    span = findspan(n_r - 1, p_sol, xi, Xi_r);
    N = basisfun(span, xi, p_sol, Xi_r);

    r_val = 0; u_val = 0;
    for i = 0:p_sol
        idx = span - p_sol + i + 1;
        r_val = r_val + N(i + 1) * r_ctrl(idx);
        u_val = u_val + N(i + 1) * u_sol(idx);
    end
    r_plot(t) = r_val;
    u_plot(t) = u_val;
    u_exact_plot(t) = log(r_val);
end

err_max = max(abs(u_plot - u_exact_plot));
fprintf('    Max error: %.6e\n', err_max);

%% Plot results
figure(1);

% Quarter circle
subplot(2, 2, 1);
plot(x_curve, y_curve, 'b-', 'LineWidth', 2);
hold on;
plot(Px, Py, 'ro-', 'LineWidth', 1);
hold off;
title('NURBS Quarter Circle');
xlabel('x'); ylabel('y');
axis equal;
grid on;
legend('NURBS curve', 'Control polygon');

% Radius accuracy
subplot(2, 2, 2);
plot(xi_vals, abs(r_vals - 1.0), 'b-', 'LineWidth', 1.5);
title('Radius Error (should be ~eps)');
xlabel('xi'); ylabel('|r - 1|');
grid on;

% Annulus solution
subplot(2, 2, 3);
plot(r_plot, u_plot, 'b-', 'LineWidth', 1.5);
hold on;
plot(r_plot, u_exact_plot, 'r--', 'LineWidth', 1);
hold off;
title('Laplace on Annulus');
xlabel('r'); ylabel('u');
legend('IGA', 'Exact: ln(r)');
grid on;

% Error
subplot(2, 2, 4);
plot(r_plot, abs(u_plot - u_exact_plot), 'b-', 'LineWidth', 1.5);
title('Pointwise Error');
xlabel('r'); ylabel('|u_h - u|');
grid on;

drawnow;

fprintf('\n=== NURBS Geometry Complete ===\n');
