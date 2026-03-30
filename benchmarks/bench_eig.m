% Benchmark: Eigenvalues of 300x300 symmetric matrix [deterministic]
n = 300;
A = reshape(linspace(0.01, 1, n*n), n, n);
A = (A + A.') / 2;
tic;
[V, D] = eig(A);
t = toc;
result = sum(diag(D)) / n;
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
