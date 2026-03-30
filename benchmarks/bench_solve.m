% Benchmark: Solve Ax=b for 500x500 system [deterministic]
n = 500;
A = reshape(linspace(0.01, 1, n*n), n, n) + eye(n) * n;
b = linspace(1, 2, n).';
tic;
x = A \ b;
t = toc;
result = norm(A * x - b);
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
