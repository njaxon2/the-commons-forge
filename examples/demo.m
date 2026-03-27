function result = demo(x, n)
%DEMO  Showcase various Octave/MATLAB syntax elements.
%   result = demo(x, n)
%   Demonstrates keywords, builtins, loops, conditionals,
%   string ops, plotting, and matrix manipulation.
%
%   Example:
%       demo(linspace(0, 2*pi, 100), 5)

    % --- Variables and arrays -------------------------------------------
    greeting = 'Hello from Forge!';
    disp(greeting);

    A = [1, 2, 3; 4, 5, 6; 7, 8, 9];      % 3x3 matrix literal
    b = zeros(1, 10);                        % pre-allocate row vector
    scale_factor = 2.718e0;                  % floating-point number

    % --- Loop: fill vector with scaled values --------------------------
    for k = 1:length(b)
        b(k) = sin(k * pi / 5) * scale_factor;
    endfor

    % --- Conditional: choose operation ---------------------------------
    if nargin < 2
        n = 3;                                % default value
        fprintf("Using default n = %d\n", n);
    elseif n > 10
        warning('n is large; clamping to 10');
        n = 10;
    else
        fprintf("n = %d\n", n);
    endif

    % --- String operations ---------------------------------------------
    label = strcat("Result for n=", num2str(n));
    tokens = strsplit(label, " ");
    disp(tokens);

    % --- Computation ---------------------------------------------------
    result = A^n + eye(3) * sum(b);

    % --- Plotting ------------------------------------------------------
    figure('Name', 'Forge Demo Plot');
    t = linspace(0, 2*pi, 256);
    y1 = sin(t);
    y2 = cos(t) .* exp(-t / (2*pi));

    subplot(2, 1, 1);
    plot(t, y1, 'r-', 'LineWidth', 1.5);
    title('Sine wave');
    xlabel('t'); ylabel('sin(t)');
    grid on;

    subplot(2, 1, 2);
    plot(t, y2, 'b--', 'LineWidth', 1.5);
    title('Damped cosine');
    xlabel('t'); ylabel('y');
    legend('cos(t)*exp(-t/2\pi)');
    grid on;

    printf("Demo complete.  Result size: %dx%d\n", rows(result), columns(result));
endfunction
