%% TIGA Results with Structs and Cell Arrays
%  Store and organize IGA analysis results using structs and cell arrays
%  Tests: struct creation, field access, cell arrays, cellfun-like ops

clear;
fprintf('=== TIGA Results with Data Structures ===\n\n');

%% Build a struct to hold mesh parameters
mesh.p = 2;
mesh.num_el = 8;
mesh.h = 1.0 / mesh.num_el;

% Build knot vector
interior = linspace(0, 1, mesh.num_el + 1);
interior = interior(2:end-1);
mesh.Xi = [zeros(1, mesh.p+1), interior, ones(1, mesh.p+1)];
mesh.n_dof = length(mesh.Xi) - mesh.p - 1;

fprintf('Mesh parameters:\n');
fprintf('  p = %d\n', mesh.p);
fprintf('  num_el = %d\n', mesh.num_el);
fprintf('  n_dof = %d\n', mesh.n_dof);

%% Solve for different load cases and store in cell array
n_cases = 3;
load_names = {'uniform', 'linear', 'sinusoidal'};

% Quadrature
nqp = mesh.p + 2;
[gp, gw] = gaussQuad(nqp);
knots_unique = unique(mesh.Xi);
nel = length(knots_unique) - 1;
n = mesh.n_dof;
p = mesh.p;
Xi = mesh.Xi;

% Assemble stiffness (same for all cases)
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

fprintf('\nSolving %d load cases...\n', n_cases);

% Store results in struct array
for c = 1:n_cases
    % Assemble load vector
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
            x = xi;
            span = findspan(n-1, p, xi, Xi);
            ders = derbasisfun(span, xi, p, 1, Xi);
            N_val = ders(1, :);
            wt = gw(q) * J_xi;

            % Load function
            if c == 1
                load_val = 1.0;
            elseif c == 2
                load_val = x;
            else
                load_val = sin(pi * x);
            end

            for ii = 0:p
                I = span - p + ii + 1;
                f(I) = f(I) + N_val(ii+1) * load_val * wt;
            end
        end
    end

    % Solve
    free = 2:n-1;
    K_f = K(free, free);
    f_f = f(free);
    u_f = K_f \ f_f;

    % Full solution
    u = zeros(n, 1);
    u(free) = u_f;

    % Store in results struct
    results(c).name = load_names{c};
    results(c).load_vector = f;
    results(c).solution = u;
    results(c).max_disp = max(abs(u));
    results(c).energy = 0.5 * u' * K * u;
end

%% Print results table
fprintf('\n  %-15s | %12s | %12s\n', 'Load Case', 'Max |u|', 'Energy');
fprintf('  %-15s-|-%12s-|-%12s\n', '---------------', '------------', '------------');
for c = 1:n_cases
    fprintf('  %-15s | %12.6e | %12.6e\n', ...
        results(c).name, results(c).max_disp, results(c).energy);
end

%% Plot all solutions
figure(1);
n_plot = 200;
x_plot = linspace(0, 1-1e-10, n_plot);

for c = 1:n_cases
    u = results(c).solution;
    y_plot = zeros(1, n_plot);
    for t = 1:n_plot
        xi = x_plot(t);
        span = findspan(n-1, p, xi, Xi);
        N = basisfun(span, xi, p, Xi);
        val = 0;
        for k = 0:p
            val = val + N(k+1) * u(span - p + k + 1);
        end
        y_plot(t) = val;
    end
    plot(x_plot, y_plot, 'LineWidth', 1.5);
    hold on;
end
hold off;
legend('Uniform', 'Linear', 'Sinusoidal');
xlabel('x');
ylabel('u(x)');
title('IGA Solutions for Different Load Cases');
grid on;
drawnow;

fprintf('\n=== Results Complete ===\n');
