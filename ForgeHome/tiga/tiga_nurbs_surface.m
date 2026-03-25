%% TIGA NURBS Surface Evaluation
%  Construct and evaluate a NURBS surface (quarter cylinder)
%  Tests: 2D array operations, meshgrid, surface evaluation

clear;
fprintf('=== TIGA NURBS Surface: Quarter Cylinder ===\n');

%% Define the NURBS quarter cylinder surface
% Parameters:
%   - Radius R in circumferential direction (xi)
%   - Height H in axial direction (eta)
%   - Quarter circle from 0 to pi/2

R = 2;    % radius
H = 3;    % height

% Circumferential direction: quadratic NURBS quarter circle
p = 2;
Xi = [0 0 0 1 1 1];  % knot vector (single element)
n_xi = 3;  % 3 control points

% Axial direction: linear (or quadratic)
q = 1;
Eta = [0 0 1 1];
n_eta = 2;

% Control points for quarter cylinder (3D: x,y,z)
% Bottom circle: R*cos(0)=R, R*cos(pi/4)=R/sqrt(2), R*cos(pi/2)=0
w = 1/sqrt(2);  % weight for middle control point

% Control net: (n_xi x n_eta) control points
% Bottom row (eta=0, z=0):
P_x = zeros(n_xi, n_eta);
P_y = zeros(n_xi, n_eta);
P_z = zeros(n_xi, n_eta);
W = ones(n_xi, n_eta);

% Column 1 (bottom, eta=0)
P_x(1,1) = R; P_y(1,1) = 0; P_z(1,1) = 0;
P_x(2,1) = R; P_y(2,1) = R; P_z(2,1) = 0;
P_x(3,1) = 0; P_y(3,1) = R; P_z(3,1) = 0;
W(2,1) = w;

% Column 2 (top, eta=1)
P_x(1,2) = R; P_y(1,2) = 0; P_z(1,2) = H;
P_x(2,2) = R; P_y(2,2) = R; P_z(2,2) = H;
P_x(3,2) = 0; P_y(3,2) = R; P_z(3,2) = H;
W(2,2) = w;

fprintf('Control points (x):\n');
disp(P_x);
fprintf('Control points (y):\n');
disp(P_y);
fprintf('Weights:\n');
disp(W);
fprintf('Radius: R = %f\n', R);
fprintf('Height: H = %f\n', H);

%% Evaluate the NURBS surface
num_pts = 21;
xi_vals = linspace(0, 1, num_pts);
eta_vals = linspace(0, 1, num_pts);

S_x = zeros(num_pts, num_pts);
S_y = zeros(num_pts, num_pts);
S_z = zeros(num_pts, num_pts);

for i = 1:num_pts
    xi = xi_vals(i);
    if xi >= 1
        xi = 1 - 1e-10;
    end
    span_xi = findspan(n_xi-1, p, xi, Xi);
    N_xi = basisfun(span_xi, xi, p, Xi);

    for j = 1:num_pts
        eta = eta_vals(j);
        if eta >= 1
            eta = 1 - 1e-10;
        end
        span_eta = findspan(n_eta-1, q, eta, Eta);
        N_eta = basisfun(span_eta, eta, q, Eta);

        % Evaluate NURBS surface point
        x_num = 0; y_num = 0; z_num = 0; w_den = 0;

        for a = 0:p
            I = span_xi - p + a + 1;
            for b = 0:q
                J = span_eta - q + b + 1;
                basis_val = N_xi(a+1) * N_eta(b+1) * W(I, J);
                x_num = x_num + basis_val * P_x(I, J);
                y_num = y_num + basis_val * P_y(I, J);
                z_num = z_num + basis_val * P_z(I, J);
                w_den = w_den + basis_val;
            end
        end

        S_x(i,j) = x_num / w_den;
        S_y(i,j) = y_num / w_den;
        S_z(i,j) = z_num / w_den;
    end
end

%% Verify geometry
% On the cylinder, x^2 + y^2 should equal R^2 at any height
fprintf('\n--- Geometry Verification ---\n');
radius_err = sqrt(S_x.^2 + S_y.^2) - R;
max_radius_err = max(max(abs(radius_err)));
fprintf('Max radius error: %e\n', max_radius_err);

% Height should range from 0 to H
fprintf('Z range: [%f, %f] (expected [0, %f])\n', min(min(S_z)), max(max(S_z)), H);

% Check a few specific points
fprintf('\nSurface point at (xi=0, eta=0): (%f, %f, %f)\n', S_x(1,1), S_y(1,1), S_z(1,1));
fprintf('Expected: (%f, %f, %f)\n', R, 0.0, 0.0);

fprintf('Surface point at (xi=0.5, eta=0.5): (%f, %f, %f)\n', ...
    S_x(11,11), S_y(11,11), S_z(11,11));
expected_angle = pi/4;  % 45 degrees
fprintf('Expected: (%f, %f, %f)\n', R*cos(expected_angle), R*sin(expected_angle), H/2);

fprintf('Surface point at (xi=1, eta=1): (%f, %f, %f)\n', ...
    S_x(end,end), S_y(end,end), S_z(end,end));
fprintf('Expected: (%f, %f, %f)\n', 0.0, R, H);

%% Plot the surface
figure(1);
surf(S_x, S_y, S_z);
xlabel('x');
ylabel('y');
zlabel('z');
title('NURBS Quarter Cylinder');
axis equal;
drawnow;

%% Also plot the control net
figure(2);
surf(S_x, S_y, S_z);
hold on;
% Plot control points
plot3(P_x(:,1), P_y(:,1), P_z(:,1), 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
plot3(P_x(:,2), P_y(:,2), P_z(:,2), 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
% Connect columns
for k = 1:n_xi
    plot3([P_x(k,1) P_x(k,2)], [P_y(k,1) P_y(k,2)], [P_z(k,1) P_z(k,2)], 'r--');
end
hold off;
xlabel('x');
ylabel('y');
zlabel('z');
title('NURBS Surface with Control Net');
axis equal;
drawnow;

fprintf('\n=== TIGA NURBS Surface Complete ===\n');
