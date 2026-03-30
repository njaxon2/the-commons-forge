% Benchmark: Cumulative sum of large array [deterministic]
n = 5000000;
x = ones(1, n) * 0.1;
tic;
y = cumsum(x);
t = toc;
result = y(end);
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
