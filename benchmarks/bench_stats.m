% Benchmark: Statistical operations on large array [deterministic]
n = 5000000;
x = sin(linspace(0, 1000*pi, n));
tic;
m = mean(x);
s = std(x);
mx = max(x);
mn = min(x);
t = toc;
result = s;
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
