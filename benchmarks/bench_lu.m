% Benchmark: LU decomposition of 500x500 matrix [deterministic]
n = 500;
A = reshape(linspace(0.01, 1, n*n), n, n) + eye(n) * n;
tic;
[L, U, P] = lu(A);
t = toc;
result = sum(diag(U)) / n;
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
