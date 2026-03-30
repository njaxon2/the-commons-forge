% Benchmark: Reshape and transpose operations [deterministic]
n = 2000;
A = reshape(linspace(0, 1, n*n), n, n);
tic;
B = reshape(A, n*n, 1);
C = reshape(B, n, n);
D = C.';
result_val = sum(D(:)) / numel(D);
t = toc;
result = result_val;
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
