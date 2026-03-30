% Benchmark: FFT of large signal (2^20 points) [deterministic]
n = 2^20;
x = sin(linspace(0, 100*pi, n)) + 0.5*cos(linspace(0, 200*pi, n));
tic;
y = fft(x);
t = toc;
result = sum(abs(y(1:100))) / 100;
fprintf("TIME=%.6f\n", t);
fprintf("RESULT=%.10f\n", result);
