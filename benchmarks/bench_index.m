% Benchmark: Array indexing and assignment (500000 ops) [deterministic]
n = 500000;
A = zeros(1, n);
tic;
for i = 1:n
    A(i) = i * 0.001;
end
t = toc;
result = sum(A) / n;
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
