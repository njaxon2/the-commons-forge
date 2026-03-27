% signal_processing.m  --  Basic DSP examples in Octave
% Covers signal generation, FFT, filtering, and spectral plotting.

% --- Parameters --------------------------------------------------------
fs = 8000;                          % sampling rate (Hz)
T  = 1.0;                          % duration (seconds)
N  = fs * T;                       % number of samples
t  = linspace(0, T, N);            % time vector

% --- Generate a composite signal ---------------------------------------
f1 = 440;  f2 = 1200;  f3 = 2600; % frequencies
signal = 0.8*sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t) + 0.3*sin(2*pi*f3*t);
noisy  = signal + 0.4 * randn(1, N);

% --- FFT ---------------------------------------------------------------
Y = fft(noisy);
f_axis = (0:N-1) * (fs / N);       % frequency axis
magnitude = abs(Y(1:N/2)) / N;

% --- Simple low-pass FIR filter ----------------------------------------
fc = 1500;                          % cutoff frequency
order = 64;
n_tap = 0:order;
h = sinc(2*fc/fs * (n_tap - order/2)) .* hamming(order + 1)';
h = h / sum(h);                     % normalise

filtered = conv(noisy, h, 'same');

% --- Plot results ------------------------------------------------------
figure('Name', 'Signal Processing Demo');

subplot(3, 1, 1);
plot(t(1:400), noisy(1:400), 'b');
title('Noisy signal (first 50 ms)');
xlabel('Time (s)'); ylabel('Amplitude');

subplot(3, 1, 2);
plot(f_axis(1:N/2), magnitude, 'r');
title('Frequency spectrum');
xlabel('Frequency (Hz)'); ylabel('|Y(f)|');
xlim([0, fs/2]);

subplot(3, 1, 3);
plot(t(1:400), filtered(1:400), 'g', 'LineWidth', 1.2);
title(sprintf('Filtered signal (LP cutoff = %d Hz)', fc));
xlabel('Time (s)'); ylabel('Amplitude');
