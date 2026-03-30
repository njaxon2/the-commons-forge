% Benchmark: SVD of 300x300 matrix [deterministic]
n = 300;
A = reshape(linspace(0.01, 1, n*n), n, n);
tic;
[U, S, V] = svd(A);
t = toc;
result = sum(diag(S)) / n;
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
