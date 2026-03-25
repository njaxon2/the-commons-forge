%% TIGA Performance Benchmark
%  Time key IGA operations and compare with theoretical complexity
%  Tests: tic/toc, string formatting, table output, memory estimation

clear;
fprintf('=== TIGA Performance Benchmark ===\n\n');

%% Benchmark 1: Assembly time vs problem size
fprintf('--- Assembly Time vs Problem Size ---\n');
fprintf('  %6s | %6s | %8s | %10s | %10s\n', ...
    'p', 'n_el', 'n_dof', 'Assem(ms)', 'Solve(ms)');
fprintf('  %6s-|-%6s-|-%8s-|-%10s-|-%10s\n', ...
    '------', '------', '--------', '----------', '----------');

p = 2;
el_counts = [4, 8, 16, 32, 64];

for idx = 1:length(el_counts)
    num_el = el_counts(idx);
    h = 1.0 / num_el;

    % Build knot vector
    interior = linspace(0, 1, num_el + 1);
    interior = interior(2:end-1);
    Xi = [zeros(1, p+1), interior, ones(1, p+1)];
    n = length(Xi) - p - 1;

    % Quadrature
    nqp = p + 2;
    [gp, gw] = gaussQuad(nqp);
    knots_unique = unique(Xi);
    nel = length(knots_unique) - 1;

    % Time assembly
    tic;
    K = zeros(n, n);
    M = zeros(n, n);
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
            x_phys = xi;
            span = findspan(n-1, p, xi, Xi);
            ders = derbasisfun(span, xi, p, 1, Xi);
            N_val = ders(1, :);
            dN_dx = ders(2, :);
            wt = gw(q) * J_xi;

            for ii = 0:p
                I = span - p + ii + 1;
                f(I) = f(I) + N_val(ii+1) * sin(pi * x_phys) * wt;
                for jj = 0:p
                    J = span - p + jj + 1;
                    K(I, J) = K(I, J) + dN_dx(ii+1)*dN_dx(jj+1)*wt;
                    M(I, J) = M(I, J) + N_val(ii+1)*N_val(jj+1)*wt;
                end
            end
        end
    end
    t_assem = toc * 1000;

    % Time solve
    free = 2:n-1;
    K_f = K(free, free);
    f_f = f(free);

    tic;
    u_f = K_f \ f_f;
    t_solve = toc * 1000;

    fprintf('  %6d | %6d | %8d | %10.2f | %10.2f\n', ...
        p, num_el, n, t_assem, t_solve);
end

%% Benchmark 2: Assembly time vs polynomial degree
fprintf('\n--- Assembly Time vs Polynomial Degree ---\n');
fprintf('  %6s | %6s | %8s | %10s\n', 'p', 'n_el', 'n_dof', 'Assem(ms)');
fprintf('  %6s-|-%6s-|-%8s-|-%10s\n', '------', '------', '--------', '----------');

num_el = 16;
for p = 1:6
    h = 1.0 / num_el;
    interior = linspace(0, 1, num_el + 1);
    interior = interior(2:end-1);
    Xi = [zeros(1, p+1), interior, ones(1, p+1)];
    n = length(Xi) - p - 1;

    nqp = p + 2;
    [gp, gw] = gaussQuad(nqp);
    knots_unique = unique(Xi);
    nel = length(knots_unique) - 1;

    tic;
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
                    K(I, J) = K(I, J) + dN_dx(ii+1)*dN_dx(jj+1)*wt;
                end
            end
        end
    end
    t_assem = toc * 1000;

    fprintf('  %6d | %6d | %8d | %10.2f\n', p, num_el, n, t_assem);
end

%% Summary
fprintf('\n--- Summary ---\n');
fprintf('  Assembly scales as O(n * p^2 * nqp)\n');
fprintf('  Solve scales as O(n^3) for dense, O(n * bw^2) for banded\n');

fprintf('\n=== TIGA Benchmark Complete ===\n');
