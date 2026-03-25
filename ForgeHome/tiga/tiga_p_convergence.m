%% TIGA p-Refinement Convergence Study
%  Compare convergence rates for polynomial degrees p=1,2,3,4
%  Solve -u'' = pi^2*sin(pi*x), u(0)=u(1)=0
%  Exact: u(x) = sin(pi*x)

clear;
fprintf('=== TIGA p-Refinement Convergence Study ===\n\n');

p_values = [1 2 3 4];
nel_values = [2 4 8 16 32];

% Store all results
all_dofs = zeros(length(p_values), length(nel_values));
all_errs = zeros(length(p_values), length(nel_values));

for ip = 1:length(p_values)
    p = p_values(ip);
    fprintf('  p = %d:\n', p);

    for im = 1:length(nel_values)
        nel_target = nel_values(im);

        % Build knot vector
        interior = linspace(0, 1, nel_target + 1);
        interior = interior(2:end-1);
        Xi = [zeros(1, p + 1), interior, ones(1, p + 1)];
        n = length(Xi) - p - 1;

        % Quadrature
        nqp = p + 2;
        [gp, gw] = gaussQuad(nqp);
        knots_unique = unique(Xi);
        nel = length(knots_unique) - 1;

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
                span = findspan(n - 1, p, xi, Xi);
                ders = derbasisfun(span, xi, p, 1, Xi);
                N_val = ders(1, :);
                dN_dx = ders(2, :);
                wt = gw(q) * J_xi;
                for ii = 0:p
                    I = span - p + ii + 1;
                    f(I) = f(I) + N_val(ii + 1) * pi^2 * sin(pi * xi) * wt;
                    for jj = 0:p
                        J = span - p + jj + 1;
                        K(I, J) = K(I, J) + dN_dx(ii + 1) * dN_dx(jj + 1) * wt;
                    end
                end
            end
        end

        % Solve
        free = 2:n - 1;
        K_f = K(free, free);
        f_f = f(free);
        u_f = K_f \ f_f;
        u = zeros(n, 1);
        u(free) = u_f;

        % L2 error
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
                span = findspan(n - 1, p, xi, Xi);
                N_val = basisfun(span, xi, p, Xi);
                u_h = 0;
                for k = 0:p
                    u_h = u_h + N_val(k + 1) * u(span - p + k + 1);
                end
                u_exact = sin(pi * xi);
                err_L2 = err_L2 + (u_h - u_exact)^2 * gw(q) * J_xi;
            end
        end
        err_L2 = sqrt(err_L2);

        all_dofs(ip, im) = n;
        all_errs(ip, im) = err_L2;

        fprintf('    nel=%2d, n=%3d, L2 err = %.4e\n', nel, n, err_L2);
    end
end

%% Compute convergence rates
fprintf('\n  Convergence rates (log-log slope):\n');
for ip = 1:length(p_values)
    p = p_values(ip);
    % Use last two mesh sizes for rate
    h1 = 1.0 / nel_values(length(nel_values) - 1);
    h2 = 1.0 / nel_values(length(nel_values));
    e1 = all_errs(ip, length(nel_values) - 1);
    e2 = all_errs(ip, length(nel_values));
    if e1 > 0 && e2 > 0
        rate = log(e1 / e2) / log(h1 / h2);
        fprintf('    p=%d: rate = %.2f (expected ~%d)\n', p, rate, p + 1);
    end
end

%% Plot
figure(1);
for ip = 1:length(p_values)
    h_vals = 1.0 ./ nel_values;
    loglog(h_vals, all_errs(ip, :), 'o-', 'LineWidth', 1.5);
    hold on;
end
hold off;

% Add reference slopes
xlabel('h (element size)');
ylabel('L2 Error');
title('IGA p-Refinement Convergence');
legend('p=1', 'p=2', 'p=3', 'p=4');
grid on;
drawnow;

fprintf('\n=== p-Convergence Study Complete ===\n');
