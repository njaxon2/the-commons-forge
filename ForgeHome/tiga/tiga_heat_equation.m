%% TIGA Transient Heat Equation
%  Solve u_t = u_xx with IGA + implicit Euler time stepping
%  IC: u(x,0) = sin(pi*x), BC: u(0,t)=u(1,t)=0
%  Exact: u(x,t) = exp(-pi^2*t) * sin(pi*x)
%  Tests: mass matrix, time stepping, transient analysis

clear;
fprintf('=== TIGA Transient Heat Equation ===\n\n');

p = 3;
nel = 10;
dt = 0.005;
t_end = 0.2;
n_steps = round(t_end / dt);

% Build knot vector
interior = linspace(0, 1, nel + 1);
interior = interior(2:end - 1);
Xi = [zeros(1, p + 1), interior, ones(1, p + 1)];
n = length(Xi) - p - 1;

fprintf('  p = %d, nel = %d, n = %d\n', p, nel, n);
fprintf('  dt = %.4f, t_end = %.2f, steps = %d\n', dt, t_end, n_steps);

% Quadrature
nqp = p + 2;
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(Xi);
nel_actual = length(knots_unique) - 1;

% Assemble stiffness K and mass M
K = zeros(n, n);
M = zeros(n, n);

for e = 1:nel_actual
    xi_a = knots_unique(e);
    xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14
        continue;
    end
    J_xi = (xi_b - xi_a) / 2;
    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n - 1, p, xi, Xi);
        ders = derbasisfun(span, xi, p, 1, Xi);
        N_val = ders(1, :);
        dN_dx = ders(2, :);
        wt = gw(q) * J_xi;
        for ii = 0:p
            I = span - p + ii + 1;
            for jj = 0:p
                J = span - p + jj + 1;
                K(I, J) = K(I, J) + dN_dx(ii + 1) * dN_dx(jj + 1) * wt;
                M(I, J) = M(I, J) + N_val(ii + 1) * N_val(jj + 1) * wt;
            end
        end
    end
end

% Initial condition: project sin(pi*x) onto B-spline space
% Solve M * u0 = f_ic where f_ic = integral(N_i * sin(pi*x))
f_ic = zeros(n, 1);
for e = 1:nel_actual
    xi_a = knots_unique(e);
    xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14
        continue;
    end
    J_xi = (xi_b - xi_a) / 2;
    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n - 1, p, xi, Xi);
        N_val = basisfun(span, xi, p, Xi);
        wt = gw(q) * J_xi;
        for ii = 0:p
            I = span - p + ii + 1;
            f_ic(I) = f_ic(I) + N_val(ii + 1) * sin(pi * xi) * wt;
        end
    end
end

% Apply BCs to IC projection
free = 2:n - 1;
u = zeros(n, 1);
u(free) = M(free, free) \ f_ic(free);

% Implicit Euler: (M + dt*K) * u^{n+1} = M * u^n
A = M(free, free) + dt * K(free, free);
M_free = M(free, free);

% Time stepping
fprintf('\n  Time stepping...\n');
t = 0;
err_hist = zeros(1, n_steps + 1);

% Compute initial error
err_L2 = 0;
for e = 1:nel_actual
    xi_a = knots_unique(e);
    xi_b = knots_unique(e + 1);
    if xi_b - xi_a < 1e-14
        continue;
    end
    J_xi = (xi_b - xi_a) / 2;
    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
        span = findspan(n - 1, p, xi, Xi);
        N_val = basisfun(span, xi, p, Xi);
        u_h = 0;
        for k = 0:p
            u_h = u_h + N_val(k + 1) * u(span - p + k + 1);
        end
        u_exact = exp(-pi^2 * t) * sin(pi * xi);
        err_L2 = err_L2 + (u_h - u_exact)^2 * gw(q) * J_xi;
    end
end
err_hist(1) = sqrt(err_L2);

for step = 1:n_steps
    t = step * dt;

    % Implicit Euler solve
    rhs = M_free * u(free);
    u(free) = A \ rhs;

    % Compute L2 error
    err_L2 = 0;
    for e = 1:nel_actual
        xi_a = knots_unique(e);
        xi_b = knots_unique(e + 1);
        if xi_b - xi_a < 1e-14
            continue;
        end
        J_xi = (xi_b - xi_a) / 2;
        for q = 1:nqp
            xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
            span = findspan(n - 1, p, xi, Xi);
            N_val = basisfun(span, xi, p, Xi);
            u_h = 0;
            for k = 0:p
                u_h = u_h + N_val(k + 1) * u(span - p + k + 1);
            end
            u_exact = exp(-pi^2 * t) * sin(pi * xi);
            err_L2 = err_L2 + (u_h - u_exact)^2 * gw(q) * J_xi;
        end
    end
    err_hist(step + 1) = sqrt(err_L2);

    if step == 1 || step == n_steps / 4 || step == n_steps / 2 || step == n_steps
        fprintf('    t=%.4f: L2 err = %.4e, max|u| = %.4f\n', t, sqrt(err_L2), max(abs(u)));
    end
end

%% Plot
figure(1);

% Solution at several times
subplot(1, 2, 1);
n_plot = 200;
x_plot = linspace(0, 1 - 1e-10, n_plot);
t_plot_vals = [0, 0.01, 0.05, 0.1, 0.2];

% Recompute for each time (forward Euler from scratch is slow, just plot exact + final)
for it = 1:length(t_plot_vals)
    t_val = t_plot_vals(it);
    y_exact = exp(-pi^2 * t_val) * sin(pi * x_plot);
    plot(x_plot, y_exact, 'LineWidth', 1);
    hold on;
end

% Plot final IGA solution
y_iga = zeros(1, n_plot);
for ix = 1:n_plot
    xi = x_plot(ix);
    span = findspan(n - 1, p, xi, Xi);
    N_val = basisfun(span, xi, p, Xi);
    for k = 0:p
        y_iga(ix) = y_iga(ix) + N_val(k + 1) * u(span - p + k + 1);
    end
end
plot(x_plot, y_iga, 'k--', 'LineWidth', 2);
hold off;

title('Heat Equation Solutions');
xlabel('x'); ylabel('u');
legend('t=0', 't=0.01', 't=0.05', 't=0.1', 't=0.2', 'IGA final');
grid on;

% Error history
subplot(1, 2, 2);
t_vals = linspace(0, t_end, n_steps + 1);
semilogy(t_vals, err_hist, 'b-', 'LineWidth', 1.5);
xlabel('Time');
ylabel('L2 Error');
title('Error vs Time');
grid on;

drawnow;

fprintf('\n=== Heat Equation Complete ===\n');
