% Benchmark: For-loop with scalar math (interpreter overhead) [deterministic]
n = 100000;
result = 0;
tic;
for i = 1:n
    result = result + sin(i * 0.001) * cos(i * 0.001);
end
t = toc;
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
