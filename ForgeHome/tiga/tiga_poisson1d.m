%% TIGA Poisson 1D - Isogeometric Analysis
%  Solve -u''(x) = f(x) on [0,1] with u(0)=u(1)=0
%  Exact solution: u(x) = sin(pi*x), f(x) = pi^2 * sin(pi*x)
%  Following the TIGA framework from tigaSPM13.pdf

clear;
clc;
fprintf('=== TIGA 1D Poisson Problem ===\n');

%% Problem setup
p = 2;  % quadratic B-splines

% Open knot vector with interior knots
Xi = [0 0 0 0.25 0.5 0.75 1 1 1];
n = length(Xi) - p - 1;  % number of basis functions

fprintf('Degree p = %d\n', p);
fprintf('Number of basis functions: %d\n', n);

%% Verify partition of unity
fprintf('\n--- Partition of Unity Check ---\n');
test_pts = linspace(0, 1, 21);
max_err = 0;
for k = 1:length(test_pts)
    u = test_pts(k);
    if u == 1
        u = 1 - 1e-10;
    end
    span = findspan(n-1, p, u, Xi);
    N = basisfun(span, u, p, Xi);
    pou = sum(N);
    err = abs(pou - 1.0);
    if err > max_err
        max_err = err;
    end
end
fprintf('Max partition of unity error: %e\n', max_err);

%% Plot basis functions
fprintf('\n--- Plotting Basis Functions ---\n');
num_plot = 101;
u_plot = linspace(0, 1, num_plot);
B = zeros(n, num_plot);

for k = 1:num_plot
    u = u_plot(k);
    if u >= 1
        u = 1 - 1e-10;
    end
    span = findspan(n-1, p, u, Xi);
    N = basisfun(span, u, p, Xi);
    for j = 0:p
        idx = span - p + j + 1;
        B(idx, k) = N(j+1);
    end
end

figure(1);
hold on;
for i = 1:n
    plot(u_plot, B(i,:));
end
hold off;
title('Quadratic B-spline Basis Functions');
xlabel('xi');
ylabel('N_{i,p}(xi)');
grid on;
drawnow;
fprintf('Figure 1: Basis functions plotted\n');

%% Assemble stiffness matrix and load vector using Gauss quadrature
fprintf('\n--- Assembling IGA System ---\n');
nqp = p + 1;  % number of quadrature points per element
[gp, gw] = gaussQuad(nqp);

% Identify element boundaries (unique knot values)
knots_unique = unique(Xi);
num_elements = length(knots_unique) - 1;
fprintf('Number of elements: %d\n', num_elements);

K = zeros(n, n);
F = zeros(n, 1);

for e = 1:num_elements
    xi_a = knots_unique(e);
    xi_b = knots_unique(e+1);

    if xi_b - xi_a < 1e-14
        continue;
    end

    J_geom = (xi_b - xi_a) / 2;

    for q = 1:nqp
        xi = (xi_a + xi_b) / 2 + J_geom * gp(q);
        x = xi;

        span = findspan(n-1, p, xi, Xi);
        ders = derbasisfun(span, xi, p, 1, Xi);
        N_val = ders(1, :);
        dN_dxi = ders(2, :);
        dN_dx = dN_dxi;

        f_val = pi^2 * sin(pi * x);

        for i = 0:p
            I = span - p + i + 1;
            F(I) = F(I) + N_val(i+1) * f_val * J_geom * gw(q);
            for j = 0:p
                J = span - p + j + 1;
                K(I, J) = K(I, J) + dN_dx(i+1) * dN_dx(j+1) * J_geom * gw(q);
            end
        end
    end
end

fprintf('Stiffness matrix K: %d x %d\n', size(K,1), size(K,2));
fprintf('K symmetry error: %e\n', norm(K - K', 'fro'));

%% Apply boundary conditions (u(0) = u(1) = 0)
fprintf('\n--- Applying Boundary Conditions ---\n');
bc_dofs = [1, n];
free_dofs = 2:n-1;

K_free = K(free_dofs, free_dofs);
F_free = F(free_dofs) - K(free_dofs, bc_dofs) * [0; 0];

fprintf('Free DOFs: %d\n', length(free_dofs));
fprintf('K_free condition number: %e\n', cond(K_free));

%% Solve
d = zeros(n, 1);
d(free_dofs) = K_free \ F_free;

fprintf('\n--- Solution ---\n');
fprintf('Control point values:\n');
disp(d');

%% Evaluate solution and compare with exact
fprintf('\n--- Error Analysis ---\n');
x_eval = linspace(0, 1, num_plot);
u_h = zeros(1, num_plot);
u_exact = sin(pi * x_eval);

for k = 1:num_plot
    xi = x_eval(k);
    if xi >= 1
        xi = 1 - 1e-10;
    end
    span = findspan(n-1, p, xi, Xi);
    N = basisfun(span, xi, p, Xi);
    for j = 0:p
        idx = span - p + j + 1;
        u_h(k) = u_h(k) + N(j+1) * d(idx);
    end
end

L2_err = sqrt(sum((u_h - u_exact).^2) / num_plot);
Linf_err = max(abs(u_h - u_exact));

fprintf('L2 error:   %e\n', L2_err);
fprintf('Linf error: %e\n', Linf_err);

%% Plot solution comparison
figure(2);
plot(x_eval, u_exact, 'b-', x_eval, u_h, 'r--');
legend('Exact', 'IGA');
title('1D Poisson: IGA vs Exact');
xlabel('x');
ylabel('u(x)');
grid on;
drawnow;
fprintf('Figure 2: Solution comparison plotted\n');

figure(3);
plot(x_eval, abs(u_h - u_exact), 'k-');
title('Pointwise Error');
xlabel('x');
ylabel('Error');
grid on;
drawnow;
fprintf('Figure 3: Error plotted\n');

%% h-refinement convergence study
fprintf('\n--- h-Refinement Convergence Study ---\n');
num_refs = 4;
h_vals = zeros(1, num_refs);
err_vals = zeros(1, num_refs);

for ref = 1:num_refs
    num_el = 2^(ref+1);
    Xi_ref = zeros(1, num_el + 2*p + 1);
    for kk = 1:p+1
        Xi_ref(kk) = 0;
        Xi_ref(end-kk+1) = 1;
    end
    interior = linspace(0, 1, num_el+1);
    for kk = 2:num_el
        Xi_ref(p + kk) = interior(kk);
    end

    n_ref = length(Xi_ref) - p - 1;
    knots_u = unique(Xi_ref);
    num_el_actual = length(knots_u) - 1;
    h_vals(ref) = 1.0 / num_el_actual;

    K_ref = zeros(n_ref, n_ref);
    F_ref = zeros(n_ref, 1);

    for e = 1:num_el_actual
        xi_a = knots_u(e);
        xi_b = knots_u(e+1);
        if xi_b - xi_a < 1e-14
            continue;
        end
        J_g = (xi_b - xi_a) / 2;

        for q = 1:nqp
            xi_q = (xi_a + xi_b)/2 + J_g * gp(q);
            sp = findspan(n_ref-1, p, xi_q, Xi_ref);
            dr = derbasisfun(sp, xi_q, p, 1, Xi_ref);
            N_v = dr(1,:);
            dN_v = dr(2,:);
            fv = pi^2 * sin(pi * xi_q);

            for ii = 0:p
                II = sp - p + ii + 1;
                F_ref(II) = F_ref(II) + N_v(ii+1) * fv * J_g * gw(q);
                for jj = 0:p
                    JJ = sp - p + jj + 1;
                    K_ref(II,JJ) = K_ref(II,JJ) + dN_v(ii+1)*dN_v(jj+1)*J_g*gw(q);
                end
            end
        end
    end

    bc = [1 n_ref];
    fr = 2:n_ref-1;
    d_ref = zeros(n_ref, 1);
    d_ref(fr) = K_ref(fr,fr) \ F_ref(fr);

    err_sum = 0;
    for k = 1:num_plot
        xi_k = x_eval(k);
        if xi_k >= 1
            xi_k = 1 - 1e-10;
        end
        sp = findspan(n_ref-1, p, xi_k, Xi_ref);
        N_k = basisfun(sp, xi_k, p, Xi_ref);
        u_k = 0;
        for jj = 0:p
            u_k = u_k + N_k(jj+1) * d_ref(sp - p + jj + 1);
        end
        err_sum = err_sum + (u_k - sin(pi*xi_k))^2;
    end
    err_vals(ref) = sqrt(err_sum / num_plot);

    fprintf('  h = %.4f, n = %d, L2 error = %e\n', h_vals(ref), n_ref, err_vals(ref));
end

fprintf('\nConvergence rates:\n');
for ref = 2:num_refs
    rate = log(err_vals(ref-1)/err_vals(ref)) / log(h_vals(ref-1)/h_vals(ref));
    fprintf('  rate = %.2f\n', rate);
end

figure(4);
loglog(h_vals, err_vals, 'bo-', h_vals, h_vals.^3, 'r--');
legend('IGA p=2', 'O(h^3)');
title('h-Refinement Convergence');
xlabel('h');
ylabel('L2 Error');
grid on;
drawnow;
fprintf('Figure 4: Convergence plot\n');

fprintf('\n=== TIGA 1D Poisson Complete ===\n');
