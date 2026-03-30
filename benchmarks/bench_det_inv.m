% Benchmark: Matrix determinant and inverse (400x400) [deterministic]
n = 400;
A = reshape(linspace(0.01, 1, n*n), n, n) + eye(n) * n;
tic;
B = inv(A);
t = toc;
result = sum(B(:)) / numel(B);
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
