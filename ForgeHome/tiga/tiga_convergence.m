%% TIGA Convergence Study: 1D Poisson with h-refinement
%  Solve -u'' = f on [0,1], u(0)=u(1)=0
%  Exact: u(x) = sin(pi*x), f(x) = pi^2*sin(pi*x)
%  Study convergence rates with h-refinement for p=1,2,3

clear;
fprintf('=== TIGA h-Refinement Convergence Study ===\n');

%% Define knot vectors for successive refinements
% For each degree, create progressively finer meshes
degrees = [1 2 3];
num_levels = 4;  % 2,4,8,16 elements

for p_idx = 1:length(degrees)
    p = degrees(p_idx);
    fprintf('\n--- Degree p = %d ---\n', p);

    h_vals = zeros(1, num_levels);
    err_L2 = zeros(1, num_levels);
    err_Linf = zeros(1, num_levels);

    for level = 1:num_levels
        num_el = 2^level;
        h = 1.0 / num_el;
        h_vals(level) = h;

        % Build open knot vector
        % Interior knots: uniformly spaced
        interior = linspace(0, 1, num_el + 1);
        interior = interior(2:end-1);  % remove endpoints
        Xi = [zeros(1, p+1), interior, ones(1, p+1)];
        n = length(Xi) - p - 1;

        % Gauss quadrature
        nqp = p + 2;  % slightly over-integrate
        [gp, gw] = gaussQuad(nqp);
        knots_unique = unique(Xi);
        nel = length(knots_unique) - 1;

        % Assemble
        K = zeros(n, n);
        F_vec = zeros(n, 1);

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
                dN_dxi = ders(2, :);

                % Identity mapping: x = xi, dx/dxi = 1
                x_phys = xi;
                dN_dx = dN_dxi;

                % Source
                f_val = pi^2 * sin(pi * x_phys);

                % Jacobian
                J_total = J_xi;
                wt = gw(q) * J_total;

                for ii = 0:p
                    I = span - p + ii + 1;
                    F_vec(I) = F_vec(I) + N_val(ii+1) * f_val * wt;
                    for jj = 0:p
                        J = span - p + jj + 1;
                        K(I, J) = K(I, J) + dN_dx(ii+1) * dN_dx(jj+1) * wt;
                    end
                end
            end
        end

        % BCs
        free = 2:n-1;
        d_sol = zeros(n, 1);
        d_sol(free) = K(free, free) \ F_vec(free);

        % Error evaluation
        num_eval = 101;
        xi_pts = linspace(0, 1, num_eval);
        max_err = 0;
        sum_sq = 0;

        for k = 1:num_eval
            xi = xi_pts(k);
            if xi >= 1
                xi = 1 - 1e-10;
            end
            span = findspan(n-1, p, xi, Xi);
            N_b = basisfun(span, xi, p, Xi);

            u_h = 0;
            for j = 0:p
                idx = span - p + j + 1;
                u_h = u_h + N_b(j+1) * d_sol(idx);
            end

            u_ex = sin(pi * xi);
            e_abs = abs(u_h - u_ex);
            if e_abs > max_err
                max_err = e_abs;
            end
            sum_sq = sum_sq + e_abs^2;
        end

        err_Linf(level) = max_err;
        err_L2(level) = sqrt(sum_sq / num_eval);

        fprintf('  h=1/%d (n=%d): L_inf=%e, L2=%e\n', num_el, n, max_err, err_L2(level));
    end

    % Compute convergence rates
    fprintf('  Convergence rates (L_inf):\n');
    for k = 2:num_levels
        rate = log(err_Linf(k-1) / err_Linf(k)) / log(h_vals(k-1) / h_vals(k));
        fprintf('    h=%1.4f -> %1.4f: rate = %.2f\n', h_vals(k-1), h_vals(k), rate);
    end
    fprintf('  Convergence rates (L2):\n');
    for k = 2:num_levels
        rate = log(err_L2(k-1) / err_L2(k)) / log(h_vals(k-1) / h_vals(k));
        fprintf('    h=%1.4f -> %1.4f: rate = %.2f\n', h_vals(k-1), h_vals(k), rate);
    end

    % Store for plotting
    if p == 1
        h1 = h_vals; e1 = err_Linf; e1_L2 = err_L2;
    end
    if p == 2
        h2 = h_vals; e2 = err_Linf; e2_L2 = err_L2;
    end
    if p == 3
        h3 = h_vals; e3 = err_Linf; e3_L2 = err_L2;
    end
end

%% Convergence plot
figure(1);
loglog(h1, e1, 'bo-', h2, e2, 'rs-', h3, e3, 'g^-');
hold on;
% Reference slopes
h_ref = [h1(1) h1(end)];
loglog(h_ref, e1(1) * (h_ref / h_ref(1)).^2, 'b--');
loglog(h_ref, e2(1) * (h_ref / h_ref(1)).^3, 'r--');
loglog(h_ref, e3(1) * (h_ref / h_ref(1)).^4, 'g--');
hold off;
xlabel('h (element size)');
ylabel('L_{inf} error');
title('IGA h-Refinement Convergence');
legend('p=1', 'p=2', 'p=3', 'O(h^2)', 'O(h^3)', 'O(h^4)');
grid on;
drawnow;

fprintf('\n=== TIGA Convergence Study Complete ===\n');
