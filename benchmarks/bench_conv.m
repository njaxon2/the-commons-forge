% Benchmark: Convolution of two vectors [deterministic]
n = 100000;
a = sin(linspace(0, 50*pi, n));
b = ones(1, 500) / 500;
tic;
c = conv(a, b);
t = toc;
result = sum(abs(c)) / length(c);
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
