%% TIGA Adaptive Refinement
%  Solve -u'' = f with adaptive h-refinement based on error indicators

clear;
fprintf('=== TIGA Adaptive Refinement ===\n\n');

% Problem: -u'' = pi^2 * sin(pi*x), u(0)=u(1)=0
% Exact: u(x) = sin(pi*x)

p = 2;
tol = 1e-6;
max_iter = 8;

% Initial coarse mesh
Xi = [0 0 0 0.5 1 1 1];

% Track convergence history
n_dof_hist = [];
err_hist = [];

for iter = 1:max_iter
    n = length(Xi) - p - 1;

    % Quadrature
    nqp = p + 2;
    [gp, gw] = gaussQuad(nqp);
    knots_unique = unique(Xi);
    nel = length(knots_unique) - 1;

    fprintf('  Iter %d: n=%d, nel=%d, len(Xi)=%d\n', iter, n, nel, length(Xi));

    % Assemble
    K = zeros(n, n);
    f = zeros(n, 1);

    for e = 1:nel
        xi_a = knots_unique(e);
        xi_b = knots_unique(e + 1);
        if xi_b - xi_a < 1e-14
            continue;
        end
        J_xi = (xi_b - xi_a) / 2;
        for q = 1:nqp
            xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
            span = findspan(n-1, p, xi, Xi);

            ders = derbasisfun(span, xi, p, 1, Xi);
            N_val = ders(1, :);
            dN_dx = ders(2, :);
            wt = gw(q) * J_xi;
            for ii = 0:p
                I = span - p + ii + 1;
                f(I) = f(I) + N_val(ii+1) * pi^2 * sin(pi*xi) * wt;
                for jj = 0:p
                    J = span - p + jj + 1;
                    K(I, J) = K(I, J) + dN_dx(ii+1)*dN_dx(jj+1)*wt;
                end
            end
        end
    end

    % Solve
    free = 2:n-1;
    K_f = K(free, free);
    f_f = f(free);
    u_f = K_f \ f_f;
    u = zeros(n, 1);
    u(free) = u_f;

    % Compute L2 error
    err_L2 = 0;
    for e = 1:nel
        xi_a = knots_unique(e);
        xi_b = knots_unique(e + 1);
        if xi_b - xi_a < 1e-14
            continue;
        end
        J_xi = (xi_b - xi_a) / 2;
        for q = 1:nqp
            xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
            span = findspan(n-1, p, xi, Xi);
            N_val = basisfun(span, xi, p, Xi);
            u_h = 0;
            for k = 0:p
                u_h = u_h + N_val(k+1) * u(span - p + k + 1);
            end
            u_exact = sin(pi*xi);
            err_L2 = err_L2 + (u_h - u_exact)^2 * gw(q) * J_xi;
        end
    end
    err_L2 = sqrt(err_L2);

    n_dof_hist = [n_dof_hist, n];
    err_hist = [err_hist, err_L2];

    fprintf('    L2 error = %.4e', err_L2);

    if err_L2 < tol
        fprintf(' -> CONVERGED!\n');
        break;
    end

    % Compute element error indicators
    elem_err = zeros(1, nel);
    for e = 1:nel
        xi_a = knots_unique(e);
        xi_b = knots_unique(e + 1);
        if xi_b - xi_a < 1e-14
            continue;
        end
        J_xi = (xi_b - xi_a) / 2;
        for q = 1:nqp
            xi = (xi_a + xi_b) / 2 + J_xi * gp(q);
            span = findspan(n-1, p, xi, Xi);
            N_val = basisfun(span, xi, p, Xi);
            u_h = 0;
            for k = 0:p
                u_h = u_h + N_val(k+1) * u(span - p + k + 1);
            end
            u_exact = sin(pi*xi);
            elem_err(e) = elem_err(e) + (u_h - u_exact)^2 * gw(q) * J_xi;
        end
        elem_err(e) = sqrt(elem_err(e));
    end

    % Skip refinement on last iteration (u wouldn't be updated)
    if iter == max_iter
        break;
    end

    % Mark elements with largest error for refinement
    max_err = max(elem_err);
    threshold = 0.5 * max_err;

    % Insert midpoints of marked elements
    new_knots = [];
    for e = 1:nel
        if elem_err(e) > threshold
            mid = (knots_unique(e) + knots_unique(e + 1)) / 2;
            new_knots = [new_knots, mid];
        end
    end

    fprintf(', refining %d/%d elements\n', length(new_knots), nel);

    % Insert new knots one at a time
    for k = 1:length(new_knots)
        xi_bar = new_knots(k);
        % Find insertion point using binary search
        kk = 0;
        for i = 1:length(Xi)-1
            if xi_bar >= Xi(i) && xi_bar < Xi(i+1)
                kk = i;
                break;
            end
        end
        if kk == 0
            kk = length(Xi) - p - 1;
        end
        Xi = [Xi(1:kk), xi_bar, Xi(kk+1:end)];
    end
end

%% Plot convergence
figure(1);
subplot(1, 2, 1);
semilogy(n_dof_hist, err_hist, 'bo-', 'LineWidth', 1.5);
xlabel('DOFs');
ylabel('L2 Error');
title('Adaptive Convergence');
grid on;

% Plot final solution
subplot(1, 2, 2);
n = length(Xi) - p - 1;
n_plot = 200;
x_plot = linspace(0, 1-1e-10, n_plot);
y_plot = zeros(1, n_plot);
for t = 1:n_plot
    xi = x_plot(t);
    span = findspan(n-1, p, xi, Xi);
    N_val = basisfun(span, xi, p, Xi);
    for k = 0:p
        y_plot(t) = y_plot(t) + N_val(k+1) * u(span - p + k + 1);
    end
end
plot(x_plot, y_plot, 'b-', 'LineWidth', 1.5);
hold on;
plot(x_plot, sin(pi*x_plot), 'r--', 'LineWidth', 1);
hold off;
legend('IGA (adaptive)', 'Exact');
xlabel('x');
ylabel('u(x)');
title('Final Solution');
grid on;
drawnow;

fprintf('\n=== Adaptive Refinement Complete ===\n');
