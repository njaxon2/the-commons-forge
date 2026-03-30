% Benchmark: Find and logical indexing [deterministic]
n = 5000000;
x = linspace(-2, 2, n);
tic;
idx = find(x > 0);
y = x(idx);
result_val = sum(y) / length(y);
t = toc;
result = result_val;
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
