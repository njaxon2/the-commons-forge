% Benchmark: Function call overhead (10000 sin calls) [deterministic]
x = linspace(0.1, 10, 10000);
tic;
for i = 1:length(x)
    y = sin(x(i));
end
t = toc;
result = y;
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
