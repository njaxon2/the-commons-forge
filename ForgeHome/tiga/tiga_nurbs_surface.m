%% TIGA NURBS Surface Geometry
%  Construct and visualize NURBS surfaces: cylinder, torus, sphere
%  Tests: 2D NURBS evaluation, surface mapping, exact geometry

clear;
fprintf('=== TIGA NURBS Surface Geometry ===\n\n');

%% Part 1: NURBS Quarter Cylinder
fprintf('Part 1: Quarter Cylinder\n');
p_xi = 2; p_eta = 1;
Xi = [0 0 0 1 1 1];        % circumferential
Eta = [0 0 1 1];             % axial
n_xi = length(Xi) - p_xi - 1;  % 3 control points
n_eta = length(Eta) - p_eta - 1; % 2 control points

% Control points: quarter circle x [0, L] extrusion
R = 1.0; L = 2.0;
% Bottom ring (eta=0)
Px = [R, R, 0;    % x-coords: 3 xi x 2 eta
      R, R, 0];
Py = [0, R, R;    % y-coords
      0, R, R];
Pz = [0, 0, 0;    % z-coords
      L, L, L];
w = [1, 1/sqrt(2), 1;  % weights
     1, 1/sqrt(2), 1];

% Evaluate surface
n_plot = 30;
xi_vals = linspace(0, 1 - 1e-10, n_plot);
eta_vals = linspace(0, 1 - 1e-10, n_plot);
Xs = zeros(n_plot, n_plot);
Ys = zeros(n_plot, n_plot);
Zs = zeros(n_plot, n_plot);

for i = 1:n_plot
    xi = xi_vals(i);
    span_xi = findspan(n_xi - 1, p_xi, xi, Xi);
    N_xi = basisfun(span_xi, xi, p_xi, Xi);
    for j = 1:n_plot
        eta = eta_vals(j);
        span_eta = findspan(n_eta - 1, p_eta, eta, Eta);
        N_eta = basisfun(span_eta, eta, p_eta, Eta);

        % NURBS surface point
        W = 0; x = 0; y = 0; z = 0;
        for a = 0:p_xi
            idx_xi = span_xi - p_xi + a + 1;
            for b = 0:p_eta
                idx_eta = span_eta - p_eta + b + 1;
                Nw = N_xi(a + 1) * N_eta(b + 1) * w(idx_eta, idx_xi);
                W = W + Nw;
            end
        end
        for a = 0:p_xi
            idx_xi = span_xi - p_xi + a + 1;
            for b = 0:p_eta
                idx_eta = span_eta - p_eta + b + 1;
                R_val = N_xi(a + 1) * N_eta(b + 1) * w(idx_eta, idx_xi) / W;
                x = x + R_val * Px(idx_eta, idx_xi);
                y = y + R_val * Py(idx_eta, idx_xi);
                z = z + R_val * Pz(idx_eta, idx_xi);
            end
        end
        Xs(j, i) = x;
        Ys(j, i) = y;
        Zs(j, i) = z;
    end
end

% Verify: all points should satisfy x^2 + y^2 = R^2
r_err = max(max(abs(sqrt(Xs.^2 + Ys.^2) - R)));
fprintf('  Max radius error: %.2e (should be ~eps)\n', r_err);

figure(1);
surf(Xs, Ys, Zs);
title('NURBS Quarter Cylinder');
xlabel('x'); ylabel('y'); zlabel('z');
colorbar;
drawnow;

%% Part 2: NURBS Quarter Sphere (single patch)
fprintf('\nPart 2: Quarter Sphere\n');

% Use degree-elevated NURBS for a quarter sphere
% Parametric: xi = circumferential (0 to pi/2), eta = meridional (0 to pi/2)
p_xi = 2; p_eta = 2;
Xi = [0 0 0 1 1 1];
Eta = [0 0 0 1 1 1];
n_xi = 3; n_eta = 3;
R_s = 1.5;

% Control points for quarter sphere
% Bottom row (eta=0, south pole region)
% Middle row (eta=0.5, equator region)
% Top row (eta=1, north pole)
sq2 = 1 / sqrt(2);

% 3x3 grid of control points (eta x xi)
CPx = [R_s,   R_s,   0;
       R_s*sq2, R_s*sq2, 0;
       0,     0,     0];
CPy = [0,   R_s,   R_s;
       0,   R_s*sq2, R_s*sq2;
       0,     0,     0];
CPz = [0,     0,     0;
       0,     0,     0;
       R_s, R_s,   R_s];
Cw = [1,   sq2,   1;
      sq2, 0.5,   sq2;
      1,   sq2,   1];

n_plot2 = 40;
xi_v = linspace(0, 1 - 1e-10, n_plot2);
eta_v = linspace(0, 1 - 1e-10, n_plot2);
Xsp = zeros(n_plot2, n_plot2);
Ysp = zeros(n_plot2, n_plot2);
Zsp = zeros(n_plot2, n_plot2);

for i = 1:n_plot2
    xi = xi_v(i);
    span_xi = findspan(n_xi - 1, p_xi, xi, Xi);
    N_xi = basisfun(span_xi, xi, p_xi, Xi);
    for j = 1:n_plot2
        eta = eta_v(j);
        span_eta = findspan(n_eta - 1, p_eta, eta, Eta);
        N_eta = basisfun(span_eta, eta, p_eta, Eta);

        W = 0; x = 0; y = 0; z = 0;
        for a = 0:p_xi
            idx_xi = span_xi - p_xi + a + 1;
            for b = 0:p_eta
                idx_eta = span_eta - p_eta + b + 1;
                Nw = N_xi(a + 1) * N_eta(b + 1) * Cw(idx_eta, idx_xi);
                W = W + Nw;
            end
        end
        for a = 0:p_xi
            idx_xi = span_xi - p_xi + a + 1;
            for b = 0:p_eta
                idx_eta = span_eta - p_eta + b + 1;
                Rv = N_xi(a + 1) * N_eta(b + 1) * Cw(idx_eta, idx_xi) / W;
                x = x + Rv * CPx(idx_eta, idx_xi);
                y = y + Rv * CPy(idx_eta, idx_xi);
                z = z + Rv * CPz(idx_eta, idx_xi);
            end
        end
        Xsp(j, i) = x;
        Ysp(j, i) = y;
        Zsp(j, i) = z;
    end
end

% Verify sphere radius
r_sphere = sqrt(Xsp.^2 + Ysp.^2 + Zsp.^2);
r_err_s = max(max(abs(r_sphere - R_s)));
fprintf('  Max radius error: %.2e\n', r_err_s);

figure(2);
surf(Xsp, Ysp, Zsp);
title('NURBS Quarter Sphere');
xlabel('x'); ylabel('y'); zlabel('z');
axis equal;
colorbar;
drawnow;

%% Part 3: Plot both together
figure(3);
subplot(1, 2, 1);
surf(Xs, Ys, Zs);
title('Quarter Cylinder');
xlabel('x'); ylabel('y'); zlabel('z');

subplot(1, 2, 2);
surf(Xsp, Ysp, Zsp);
title('Quarter Sphere');
xlabel('x'); ylabel('y'); zlabel('z');
drawnow;

fprintf('\n=== NURBS Surface Complete ===\n');
