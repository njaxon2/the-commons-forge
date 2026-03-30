% Benchmark: Unique elements in large integer array [deterministic]
n = 2000000;
x = mod(floor(linspace(0, n-1, n) * 7), 10000);
tic;
[y] = unique(x);
t = toc;
result = length(y);
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
