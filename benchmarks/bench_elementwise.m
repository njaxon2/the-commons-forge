% Benchmark: Element-wise math on large array [deterministic]
n = 2000000;
x = linspace(0.1, 10, n);
tic;
y = sin(x) + cos(x) + exp(x / 10) + log(x + 1) + sqrt(x);
t = toc;
result = sum(y) / n;
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
