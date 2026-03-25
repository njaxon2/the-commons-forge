%% TIGA Conditioning Study
%  Analyze condition numbers of IGA stiffness matrices
%  Compare different refinement strategies (h vs p)
%  Tests: eig, cond, max/min operations, formatted output

clear;
fprintf('=== TIGA Conditioning Analysis ===\n\n');

%% Study 1: h-refinement conditioning for fixed p=2
fprintf('--- h-Refinement Conditioning (p=2) ---\n');
p = 2;
num_levels = 5;

h_vals = zeros(1, num_levels);
cond_vals = zeros(1, num_levels);
n_vals = zeros(1, num_levels);
lambda_min_vals = zeros(1, num_levels);
lambda_max_vals = zeros(1, num_levels);

for level = 1:num_levels
    num_el = 2^level;
    h = 1.0 / num_el;
    h_vals(level) = h;

    % Build knot vector
    interior = linspace(0, 1, num_el + 1);
    interior = interior(2:end-1);
    Xi = [zeros(1, p+1), interior, ones(1, p+1)];
    n = length(Xi) - p - 1;
    n_vals(level) = n;

    % Assemble stiffness matrix
    nqp = p + 2;
    [gp, gw] = gaussQuad(nqp);
    knots_unique = unique(Xi);
    nel = length(knots_unique) - 1;

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

    % Extract free DOFs
    free = 2:n-1;
    K_f = K(free, free);
    M_f = M(free, free);

    % Compute eigenvalues
    lambda_K = eig(K_f);
    lambda_K = sort(lambda_K);

    lambda_min_vals(level) = lambda_K(1);
    lambda_max_vals(level) = lambda_K(end);
    cond_vals(level) = lambda_K(end) / lambda_K(1);

    fprintf('  h=1/%2d (n=%2d, free=%2d): cond(K)=%.4e, lambda_min=%.4e, lambda_max=%.4e\n', ...
        num_el, n, length(free), cond_vals(level), lambda_min_vals(level), lambda_max_vals(level));
end

%% Compute scaling rates
fprintf('\n  Condition number scaling:\n');
for k = 2:num_levels
    rate = log(cond_vals(k) / cond_vals(k-1)) / log(h_vals(k-1) / h_vals(k));
    fprintf('    h=%1.4f -> %1.4f: rate = %.2f\n', h_vals(k-1), h_vals(k), rate);
end

%% Study 2: p-refinement conditioning
fprintf('\n--- p-Refinement Conditioning (8 elements) ---\n');
num_el = 8;
max_p = 5;

for p = 1:max_p
    h = 1.0 / num_el;
    interior = linspace(0, 1, num_el + 1);
    interior = interior(2:end-1);
    Xi = [zeros(1, p+1), interior, ones(1, p+1)];
    n = length(Xi) - p - 1;

    nqp = p + 2;
    [gp, gw] = gaussQuad(nqp);
    knots_unique = unique(Xi);
    nel = length(knots_unique) - 1;

    K = zeros(n, n);
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
            dN_dx = ders(2, :);
            wt = gw(q) * J_xi;

            for ii = 0:p
                I = span - p + ii + 1;
                for jj = 0:p
                    J = span - p + jj + 1;
                    K(I, J) = K(I, J) + dN_dx(ii+1) * dN_dx(jj+1) * wt;
                end
            end
        end
    end

    free = 2:n-1;
    K_f = K(free, free);
    lambda_K = eig(K_f);
    lambda_K = sort(lambda_K);
    cond_K = lambda_K(end) / lambda_K(1);

    fprintf('  p=%d (n=%2d, free=%2d): cond(K)=%.4e\n', p, n, length(free), cond_K);
end

%% Plot conditioning results
figure(1);
loglog(h_vals, cond_vals, 'bo-');
hold on;
% Reference: O(h^-2)
h_ref = [h_vals(1) h_vals(end)];
loglog(h_ref, cond_vals(1) * (h_ref / h_ref(1)).^(-2), 'r--');
hold off;
xlabel('h');
ylabel('Condition number');
title('IGA Stiffness Matrix Conditioning (p=2)');
legend('cond(K)', 'O(h^{-2})');
grid on;
drawnow;

fprintf('\n=== TIGA Conditioning Analysis Complete ===\n');
