% Benchmark: Sorting large array [deterministic]
n = 2000000;
x = mod(linspace(0, 999999, n) * 7919, 1000);
tic;
y = sort(x);
t = toc;
result = y(1) + y(end);
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
