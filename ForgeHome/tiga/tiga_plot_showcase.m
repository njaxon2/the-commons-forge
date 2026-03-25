%% Surface Plot Showcase
%  Demonstrates all surface plotting capabilities

fprintf('=== Surface Plot Showcase ===\n\n');

% Demo 1: peaks
figure(1);
[X, Y, Z] = peaks(40);
surf(X, Y, Z);
colorbar;
title('peaks(40) - surf');
xlabel('x'); ylabel('y');

% Demo 2: sombrero
figure(2);
[X, Y, Z] = sombrero(50);
mesh(X, Y, Z);
title('sombrero - mesh');
xlabel('x'); ylabel('y');

% Demo 3: contourf with meshgrid
figure(3);
x = linspace(-3, 3, 60);
y = linspace(-3, 3, 60);
[X, Y] = meshgrid(x, y);
Z = sin(X) .* cos(Y);
contourf(X, Y, Z, 20);
colorbar;
title('contourf: sin(x)*cos(y)');
xlabel('x'); ylabel('y');

% Demo 4: surfc
figure(4);
[X, Y, Z] = peaks(30);
surfc(X, Y, Z);
title('surfc with contour projection');
xlabel('x'); ylabel('y');

drawnow;
fprintf('  4 figure windows created\n');
fprintf('=== Showcase Complete ===\n');
