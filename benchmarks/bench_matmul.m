% Benchmark: Matrix multiplication (500x500) [deterministic]
n = 500;
A = reshape(linspace(0.01, 1, n*n), n, n);
B = reshape(linspace(1, 0.01, n*n), n, n);
tic;
C = A * B;
t = toc;
result = sum(C(:)) / numel(C);
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
