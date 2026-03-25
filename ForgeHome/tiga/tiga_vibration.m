%% TIGA Vibration Analysis
%  Solve generalized eigenvalue problem K*v = lambda*M*v
%  Free vibration of a bar: exact eigenvalues = (n*pi)^2
%  Tests: eig(K,M), sort, sqrt, abs, relative error, formatted output

clear;
fprintf("=== TIGA Vibration Analysis ===\n\n");

%% Setup: p=2 B-spline, varying mesh density
p = 2;
num_modes_show = 5;

for num_el = [4, 8, 16]
    h = 1.0 / num_el;
    fprintf("--- %d elements (h=1/%d, p=%d) ---\n", num_el, num_el, p);

    % Build knot vector
    interior = linspace(0, 1, num_el + 1);
    interior = interior(2:end-1);
    Xi = [zeros(1, p+1), interior, ones(1, p+1)];
    n = length(Xi) - p - 1;

    % Gauss quadrature
    nqp = p + 2;
    [gp, gw] = gaussQuad(nqp);
    knots_unique = unique(Xi);
    nel = length(knots_unique) - 1;

    % Assemble K and M
    K = zeros(n, n);
    M = zeros(n, n);

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
                for jj = 0:p
                    J = span - p + jj + 1;
                    K(I, J) = K(I, J) + dN_dx(ii+1) * dN_dx(jj+1) * wt;
                    M(I, J) = M(I, J) + N_val(ii+1) * N_val(jj+1) * wt;
                end
            end
        end
    end

    % Apply BCs: fixed-fixed (remove first and last DOFs)
    free = 2:n-1;
    K_f = K(free, free);
    M_f = M(free, free);

    % Solve generalized eigenvalue problem
    lambda_vals = eig(K_f, M_f);
    lambda_vals = sort(lambda_vals);

    % Exact eigenvalues for fixed-fixed bar: (n*pi)^2
    n_modes = min(num_modes_show, length(lambda_vals));
    fprintf("  Mode | Computed lambda |  Exact lambda  | Rel Error\n");
    fprintf("  -----|----------------|----------------|----------\n");

    for m = 1:n_modes
        exact = (m * pi)^2;
        computed = lambda_vals(m);
        rel_err = abs(computed - exact) / exact;
        fprintf("  %4d | %14.6f | %14.6f | %.2e\n", m, computed, exact, rel_err);
    end
    fprintf("\n");
end

fprintf("=== TIGA Vibration Analysis Complete ===\n");
