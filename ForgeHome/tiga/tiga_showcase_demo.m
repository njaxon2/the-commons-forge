% TIGA Showcase: Full IGA Workflow in Forge
% Demonstrates 1D Poisson equation on unit interval with B-spline basis
% -u''(x) = f(x), u(0) = 0, u(1) = 0

addpath("/home/ubuntu/forge/ForgeHome/tiga");

%% Problem setup
nel = 8;          % Number of elements
p = 2;            % Polynomial degree
ncp = nel + p;    % Number of control points
ngp = p + 1;      % Gauss points per element

% Open knot vector: p+1 zeros, nel-1 interior knots, p+1 ones
nknots = ncp + p + 1;
Xi = zeros(1, nknots);
for i = 1:nel-1
    Xi(p + 1 + i) = i / nel;
end
for i = nknots - p:nknots
    Xi(i) = 1;
end

fprintf("IGA setup: p=%d, nel=%d, ncp=%d\n", p, nel, ncp);
fprintf("Knot vector: ");
disp(Xi);

%% Assembly
[gp, gw] = gaussQuad(ngp);
K = zeros(ncp, ncp);
F = zeros(ncp, 1);

for e = 1:nel
    xi_lo = Xi(e + p);
    xi_hi = Xi(e + p + 1);
    J_map = (xi_hi - xi_lo) / 2;

    for g = 1:ngp
        xi = xi_lo + (1 + gp(g)) * J_map;
        x_phys = xi;  % Identity mapping for unit interval

        span = findspan(ncp-1, p, xi, Xi);
        N = basisfun(span, xi, p, Xi);
        dN = basisfunder(span, xi, p, Xi, 1);
        dN_dx = dN;  % Identity mapping: dN/dx = dN/dxi

        % Source: f(x) = pi^2 * sin(pi*x)
        f_val = pi^2 * sin(pi * x_phys);

        for a = 1:p+1
            I = span - p + a;
            F(I) = F(I) + N(a) * f_val * J_map * gw(g);
            for b = 1:p+1
                J = span - p + b;
                K(I, J) = K(I, J) + dN_dx(a) * dN_dx(b) * J_map * gw(g);
            end
        end
    end
end

%% Apply BCs: u(0) = 0, u(1) = 0
bc = [1, ncp];
free = 2:ncp-1;
K_free = K(free, free);
F_free = F(free) - K(free, bc) * zeros(length(bc), 1);

%% Solve
d = zeros(ncp, 1);
d(free) = K_free \ F_free;

fprintf("Solution computed: %d DOFs\n", length(free));
fprintf("Condition number: %.2f\n", cond(K_free));

%% Evaluate solution on fine grid
n_eval = 100;
x_plot = linspace(0, 1, n_eval);
u_h = zeros(1, n_eval);

for i = 1:n_eval
    xi = x_plot(i);
    if xi >= 1; xi = 1 - eps; end
    span = findspan(ncp-1, p, xi, Xi);
    N = basisfun(span, xi, p, Xi);
    for a = 1:p+1
        u_h(i) = u_h(i) + N(a) * d(span - p + a);
    end
end

% Exact solution: u(x) = sin(pi*x)
u_exact = sin(pi * x_plot);

%% Plot results
figure(1);
plot(x_plot, u_h);
hold on;
plot(x_plot, u_exact, "--r");
hold off;
title("IGA Solution: -u'' = pi^2 sin(pi x)", "FontSize", 14);
xlabel("x");
ylabel("u(x)");
legend("IGA (p=2, nel=8)", "Exact: sin(pi x)");
grid on;
saveas(1, "/tmp/tiga_solution.png");

%% Error analysis
err = max(abs(u_h - u_exact));
fprintf("Max error: %.6e\n", err);

figure(2);
plot(x_plot, abs(u_h - u_exact));
title("Pointwise Error", "FontSize", 14);
xlabel("x");
ylabel("|u_h - u_{exact}|");
grid on;
saveas(2, "/tmp/tiga_error.png");

fprintf("\nTIGA showcase complete!\n");
