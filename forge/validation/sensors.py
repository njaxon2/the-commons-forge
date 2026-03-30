# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Sensor Simulation Framework for Forge.

Accurate simulated sensors for validating instrument control functions.
Provides ADC/DAC models, physical sensor simulators, communication protocol
simulators, and signal integrity degradation utilities.

Target location: forge/validation/sensors.py

Backend: NumPy only (no external hardware dependencies).
"""

from __future__ import annotations

import numpy as np
from typing import Any


# =====================================================================
# Sensor Registry
# =====================================================================

SENSOR_REGISTRY: dict[str, Any] = {}


def _reg(name: str | None = None):
    """Decorator to register a sensor class or function."""
    def decorator(obj):
        key = name or obj.__name__
        SENSOR_REGISTRY[key] = obj
        return obj
    return decorator


# =====================================================================
# Simulated ADC / DAC
# =====================================================================

@_reg()
class SimulatedADC:
    """Simulated Analog-to-Digital Converter with quantization and noise.

    Parameters
    ----------
    bits : int
        Resolution in bits (e.g. 12 for a 12-bit ADC).
    vref : float
        Reference voltage (full-scale range is 0 to vref).
    noise_rms : float
        RMS noise in volts added before quantization.
    sample_rate : float
        Sample rate in Hz (used for timing metadata).
    """

    def __init__(self, bits: int = 12, vref: float = 3.3,
                 noise_rms: float = 0.0, sample_rate: float = 1e6):
        self.bits = int(bits)
        self.vref = float(vref)
        self.noise_rms = float(noise_rms)
        self.sample_rate = float(sample_rate)
        self.levels = 2 ** self.bits
        self.lsb = self.vref / self.levels

    def read(self, analog_signal: np.ndarray) -> np.ndarray:
        """Quantize an analog signal to digital codes.

        Parameters
        ----------
        analog_signal : array_like
            Analog voltage values (should be in [0, vref]).

        Returns
        -------
        ndarray
            Integer digital codes in [0, 2^bits - 1].
        """
        sig = np.asarray(analog_signal, dtype=np.float64)
        # Add noise
        if self.noise_rms > 0:
            noise = np.random.normal(0, self.noise_rms, sig.shape)
            sig = sig + noise
        # Clip to valid range
        sig = np.clip(sig, 0.0, self.vref - self.lsb)
        # Quantize
        codes = np.floor(sig / self.lsb).astype(np.int64)
        return np.clip(codes, 0, self.levels - 1)

    def snr(self) -> float:
        """Theoretical signal-to-noise ratio in dB.

        SNR = 6.02 * bits + 1.76 dB (ideal quantization noise only).
        """
        return 6.02 * self.bits + 1.76

    def enob(self) -> float:
        """Effective number of bits accounting for noise.

        ENOB = (SINAD - 1.76) / 6.02 where SINAD accounts for noise_rms.
        """
        if self.noise_rms <= 0:
            return float(self.bits)
        # Quantization noise RMS = LSB / sqrt(12)
        q_noise = self.lsb / np.sqrt(12.0)
        total_noise = np.sqrt(q_noise ** 2 + self.noise_rms ** 2)
        # Full-scale sine amplitude = vref / (2*sqrt(2))
        signal_rms = self.vref / (2.0 * np.sqrt(2.0))
        sinad_db = 20.0 * np.log10(signal_rms / total_noise)
        return (sinad_db - 1.76) / 6.02

    def codes_to_voltage(self, codes: np.ndarray) -> np.ndarray:
        """Convert digital codes back to voltage."""
        return np.asarray(codes, dtype=np.float64) * self.lsb


@_reg()
class SimulatedDAC:
    """Simulated Digital-to-Analog Converter.

    Parameters
    ----------
    bits : int
        Resolution in bits.
    vref : float
        Reference voltage (full-scale output).
    glitch_energy : float
        Glitch energy in volt-nanoseconds (0 = ideal).
    """

    def __init__(self, bits: int = 12, vref: float = 3.3,
                 glitch_energy: float = 0.0):
        self.bits = int(bits)
        self.vref = float(vref)
        self.glitch_energy = float(glitch_energy)
        self.levels = 2 ** self.bits
        self.lsb = self.vref / self.levels

    def write(self, digital_values: np.ndarray) -> np.ndarray:
        """Reconstruct analog signal from digital codes.

        Parameters
        ----------
        digital_values : array_like
            Integer codes in [0, 2^bits - 1].

        Returns
        -------
        ndarray
            Reconstructed analog voltage values.
        """
        codes = np.asarray(digital_values, dtype=np.int64)
        codes = np.clip(codes, 0, self.levels - 1)
        analog = codes.astype(np.float64) * self.lsb
        # Add glitch transients at code transitions
        if self.glitch_energy > 0 and analog.size > 1:
            transitions = np.abs(np.diff(codes))
            glitch = np.zeros_like(analog)
            # Glitch magnitude proportional to code change
            glitch[1:] = transitions * self.glitch_energy * 1e-9 / self.lsb
            analog = analog + glitch
        return analog

    def snr(self) -> float:
        """Theoretical SNR in dB."""
        return 6.02 * self.bits + 1.76

    def dnl(self, test_codes: np.ndarray | None = None) -> np.ndarray:
        """Compute differential nonlinearity (ideal = 0 for simulated).

        Returns array of DNL values in LSB units.
        """
        if test_codes is None:
            test_codes = np.arange(self.levels)
        output = self.write(test_codes)
        steps = np.diff(output)
        return steps / self.lsb - 1.0

    def inl(self, test_codes: np.ndarray | None = None) -> np.ndarray:
        """Compute integral nonlinearity from cumulative DNL."""
        return np.cumsum(self.dnl(test_codes))


# =====================================================================
# Simulated Physical Sensors
# =====================================================================

@_reg()
class SimulatedThermistor:
    """NTC Thermistor using the Steinhart-Hart / B-parameter model.

    Parameters
    ----------
    R0 : float
        Resistance at reference temperature T0 (ohms).
    B : float
        B-parameter (Kelvin).
    T0 : float
        Reference temperature in Kelvin (default 298.15 = 25 C).
    noise_std : float
        Gaussian noise standard deviation in ohms.
    """

    def __init__(self, R0: float = 10000.0, B: float = 3950.0,
                 T0: float = 298.15, noise_std: float = 0.0):
        self.R0 = float(R0)
        self.B = float(B)
        self.T0 = float(T0)
        self.noise_std = float(noise_std)
        # Steinhart-Hart coefficients (simplified B-parameter model)
        self._a = 1.0 / self.T0
        self._b = 1.0 / self.B

    def read(self, temperature: np.ndarray) -> np.ndarray:
        """Compute resistance for given temperature(s).

        Parameters
        ----------
        temperature : array_like
            Temperature in Kelvin.

        Returns
        -------
        ndarray
            Resistance in ohms.
        """
        T = np.asarray(temperature, dtype=np.float64)
        R = self.R0 * np.exp(self.B * (1.0 / T - 1.0 / self.T0))
        if self.noise_std > 0:
            R = R + np.random.normal(0, self.noise_std, R.shape)
        return np.maximum(R, 0.0)

    def temperature_from_resistance(self, resistance: np.ndarray) -> np.ndarray:
        """Inverse: compute temperature from resistance.

        Returns temperature in Kelvin.
        """
        R = np.asarray(resistance, dtype=np.float64)
        inv_T = (1.0 / self.T0) + (1.0 / self.B) * np.log(R / self.R0)
        return 1.0 / inv_T

    def calibrate(self, temps: np.ndarray,
                  resistances: np.ndarray) -> np.ndarray:
        """Fit Steinhart-Hart coefficients from calibration data.

        Parameters
        ----------
        temps : array_like
            Calibration temperatures in Kelvin (at least 3 points).
        resistances : array_like
            Measured resistances at those temperatures.

        Returns
        -------
        ndarray
            Fitted Steinhart-Hart coefficients [a1, a2, a3].
        """
        T = np.asarray(temps, dtype=np.float64)
        R = np.asarray(resistances, dtype=np.float64)
        y = 1.0 / T
        x = np.log(R)
        # Steinhart-Hart: 1/T = a1 + a2*ln(R) + a3*(ln(R))^3
        A = np.column_stack([np.ones_like(x), x, x ** 3])
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return coeffs


# Thermocouple voltage tables (simplified polynomial approximation)
_THERMOCOUPLE_COEFFICIENTS = {
    # Polynomial coefficients for voltage (uV) = sum(c_i * T^i), T in Celsius
    # Simplified 4th-order approximations for typical ranges
    'K': np.array([0.0, 39.4759, 0.023614, -3.284e-5, 1.02e-8]),
    'J': np.array([0.0, 50.3814, 0.030451, -8.568e-6, 1.33e-9]),
    'T': np.array([0.0, 38.7481, 0.044127, -1.187e-5, 2.00e-9]),
}


@_reg()
class SimulatedThermocouple:
    """Simulated thermocouple (Type K, J, or T).

    Parameters
    ----------
    type_letter : str
        Thermocouple type: 'K', 'J', or 'T'.
    noise_uv : float
        Gaussian noise standard deviation in microvolts.
    """

    def __init__(self, type_letter: str = 'K', noise_uv: float = 0.0):
        self.type_letter = type_letter.upper()
        if self.type_letter not in _THERMOCOUPLE_COEFFICIENTS:
            raise ValueError(
                f"Unsupported thermocouple type: {self.type_letter}. "
                f"Supported: {list(_THERMOCOUPLE_COEFFICIENTS.keys())}")
        self.coeffs = _THERMOCOUPLE_COEFFICIENTS[self.type_letter]
        self.noise_uv = float(noise_uv)

    def read(self, temperature: np.ndarray) -> np.ndarray:
        """Compute thermocouple voltage for given temperature(s).

        Parameters
        ----------
        temperature : array_like
            Temperature in degrees Celsius.

        Returns
        -------
        ndarray
            Voltage in microvolts.
        """
        T = np.asarray(temperature, dtype=np.float64)
        voltage = np.polyval(self.coeffs[::-1], T)
        if self.noise_uv > 0:
            voltage = voltage + np.random.normal(0, self.noise_uv, T.shape)
        return voltage

    def temperature_from_voltage(self, voltage_uv: np.ndarray) -> np.ndarray:
        """Approximate inverse: voltage (uV) to temperature (C).

        Uses Newton's method on the polynomial.
        """
        V = np.asarray(voltage_uv, dtype=np.float64)
        # Initial guess from linear coefficient
        T_est = V / self.coeffs[1]
        for _ in range(20):
            f_val = np.polyval(self.coeffs[::-1], T_est) - V
            # Derivative polynomial
            deriv_coeffs = np.polyder(self.coeffs[::-1])
            f_prime = np.polyval(deriv_coeffs, T_est)
            f_prime = np.where(np.abs(f_prime) < 1e-15, 1e-15, f_prime)
            T_est = T_est - f_val / f_prime
        return T_est


@_reg()
class SimulatedPressureSensor:
    """Simulated pressure sensor with linear output.

    Parameters
    ----------
    range_pa : float
        Full-scale pressure range in Pascals.
    sensitivity : float
        Sensitivity in V/Pa.
    offset : float
        Zero-pressure offset voltage in V.
    noise : float
        Gaussian noise standard deviation in V.
    """

    def __init__(self, range_pa: float = 100000.0,
                 sensitivity: float = 1e-5,
                 offset: float = 0.5, noise: float = 0.0):
        self.range_pa = float(range_pa)
        self.sensitivity = float(sensitivity)
        self.offset = float(offset)
        self.noise = float(noise)

    def read(self, pressure: np.ndarray) -> np.ndarray:
        """Compute output voltage for given pressure(s).

        Parameters
        ----------
        pressure : array_like
            Pressure in Pascals.

        Returns
        -------
        ndarray
            Output voltage in Volts.
        """
        P = np.asarray(pressure, dtype=np.float64)
        voltage = self.offset + self.sensitivity * P
        if self.noise > 0:
            voltage = voltage + np.random.normal(0, self.noise, P.shape)
        return voltage

    def pressure_from_voltage(self, voltage: np.ndarray) -> np.ndarray:
        """Inverse: compute pressure from output voltage."""
        V = np.asarray(voltage, dtype=np.float64)
        return (V - self.offset) / self.sensitivity


@_reg()
class SimulatedAccelerometer:
    """Simulated 3-axis MEMS accelerometer.

    Parameters
    ----------
    sensitivity : float
        Sensitivity in V/g.
    range_g : float
        Full-scale range in g.
    cross_axis : float
        Cross-axis sensitivity as fraction (e.g. 0.02 = 2%).
    noise_density : float
        Noise density in g/sqrt(Hz).
    """

    def __init__(self, sensitivity: float = 0.3,
                 range_g: float = 16.0,
                 cross_axis: float = 0.02,
                 noise_density: float = 300e-6):
        self.sensitivity = float(sensitivity)
        self.range_g = float(range_g)
        self.cross_axis = float(cross_axis)
        self.noise_density = float(noise_density)

    def read(self, accel_xyz: np.ndarray,
             bandwidth: float = 1000.0) -> np.ndarray:
        """Compute output voltages for acceleration input.

        Parameters
        ----------
        accel_xyz : array_like
            Acceleration in g, shape (N, 3) or (3,).
        bandwidth : float
            Measurement bandwidth in Hz (for noise calculation).

        Returns
        -------
        ndarray
            Output voltages, same shape as input.
        """
        a = np.asarray(accel_xyz, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.shape[-1] != 3:
            raise ValueError("Accelerometer input must have 3 components (x, y, z)")

        # Clip to range
        a = np.clip(a, -self.range_g, self.range_g)

        # Cross-axis coupling matrix
        C = np.eye(3) + self.cross_axis * (np.ones((3, 3)) - np.eye(3))
        coupled = a @ C.T

        # Convert to voltage
        voltage = coupled * self.sensitivity

        # Add noise
        noise_rms = self.noise_density * np.sqrt(bandwidth)
        if noise_rms > 0:
            voltage = voltage + np.random.normal(0, noise_rms * self.sensitivity,
                                                  voltage.shape)
        return voltage.squeeze()


@_reg()
class SimulatedStrainGauge:
    """Simulated strain gauge (resistive).

    Parameters
    ----------
    gauge_factor : float
        Gauge factor (typically ~2.0 for metal foil).
    resistance : float
        Nominal resistance in ohms.
    noise : float
        Gaussian noise standard deviation in ohms.
    """

    def __init__(self, gauge_factor: float = 2.0,
                 resistance: float = 350.0, noise: float = 0.0):
        self.gauge_factor = float(gauge_factor)
        self.resistance = float(resistance)
        self.noise = float(noise)

    def read(self, strain: np.ndarray) -> np.ndarray:
        """Compute resistance change for given strain(s).

        Parameters
        ----------
        strain : array_like
            Mechanical strain (dimensionless, e.g. 1e-6 for 1 microstrain).

        Returns
        -------
        ndarray
            Resistance in ohms (R0 + delta_R).
        """
        eps = np.asarray(strain, dtype=np.float64)
        delta_R = self.resistance * self.gauge_factor * eps
        R = self.resistance + delta_R
        if self.noise > 0:
            R = R + np.random.normal(0, self.noise, R.shape)
        return R

    def strain_from_resistance(self, measured_R: np.ndarray) -> np.ndarray:
        """Inverse: compute strain from measured resistance."""
        R = np.asarray(measured_R, dtype=np.float64)
        delta_R = R - self.resistance
        return delta_R / (self.resistance * self.gauge_factor)

    def wheatstone_voltage(self, strain: np.ndarray,
                           v_excitation: float = 5.0) -> np.ndarray:
        """Compute Wheatstone bridge output for quarter-bridge config.

        Parameters
        ----------
        strain : array_like
            Mechanical strain.
        v_excitation : float
            Bridge excitation voltage.

        Returns
        -------
        ndarray
            Bridge output voltage in V.
        """
        eps = np.asarray(strain, dtype=np.float64)
        # Quarter bridge: Vout = Vex * GF * strain / 4
        return v_excitation * self.gauge_factor * eps / 4.0


# =====================================================================
# Communication Protocol Simulators
# =====================================================================

_PARITY_MODES = {'none', 'even', 'odd', 'mark', 'space'}


@_reg()
class SimulatedUART:
    """Simulated UART serial communication.

    Parameters
    ----------
    baud : int
        Baud rate (bits per second).
    data_bits : int
        Number of data bits per frame (5-9).
    parity : str
        Parity mode: 'none', 'even', 'odd', 'mark', 'space'.
    stop_bits : float
        Number of stop bits (1, 1.5, or 2).
    """

    def __init__(self, baud: int = 9600, data_bits: int = 8,
                 parity: str = 'none', stop_bits: float = 1.0):
        self.baud = int(baud)
        self.data_bits = int(data_bits)
        self.parity = parity.lower()
        self.stop_bits = float(stop_bits)
        if self.parity not in _PARITY_MODES:
            raise ValueError(f"Invalid parity: {self.parity}. "
                             f"Must be one of {_PARITY_MODES}")
        self.bit_time = 1.0 / self.baud

    def _compute_parity(self, data_byte: int) -> int:
        """Compute parity bit for a data byte."""
        ones = bin(data_byte).count('1')
        if self.parity == 'none':
            return -1  # no parity bit
        elif self.parity == 'even':
            return ones % 2
        elif self.parity == 'odd':
            return (ones + 1) % 2
        elif self.parity == 'mark':
            return 1
        elif self.parity == 'space':
            return 0
        return -1

    def send(self, data: np.ndarray) -> np.ndarray:
        """Encode data bytes into a UART bitstream.

        Parameters
        ----------
        data : array_like
            Array of byte values (0-255).

        Returns
        -------
        ndarray
            Binary bitstream with start/data/parity/stop bits.
        """
        data = np.asarray(data, dtype=np.uint8).ravel()
        frames = []
        for byte_val in data:
            frame = [0]  # start bit (low)
            # Data bits (LSB first)
            for bit_i in range(self.data_bits):
                frame.append((int(byte_val) >> bit_i) & 1)
            # Parity bit
            parity_bit = self._compute_parity(int(byte_val))
            if parity_bit >= 0:
                frame.append(parity_bit)
            # Stop bits (high)
            n_stop = int(np.ceil(self.stop_bits))
            frame.extend([1] * n_stop)
            frames.extend(frame)
        return np.array(frames, dtype=np.int8)

    def receive(self, bitstream: np.ndarray) -> dict:
        """Decode a UART bitstream back to data bytes.

        Parameters
        ----------
        bitstream : array_like
            Binary bitstream from send() or external source.

        Returns
        -------
        dict
            {'data': ndarray of bytes, 'errors': list of error descriptions}
        """
        bits = np.asarray(bitstream, dtype=np.int8).ravel()
        has_parity = self.parity != 'none'
        frame_len = 1 + self.data_bits + (1 if has_parity else 0) + int(
            np.ceil(self.stop_bits))

        data_bytes = []
        errors = []
        idx = 0
        frame_num = 0

        while idx + frame_len <= len(bits):
            # Check start bit
            if bits[idx] != 0:
                errors.append(f"Frame {frame_num}: invalid start bit")
                idx += 1
                continue

            # Extract data bits
            data_bits_arr = bits[idx + 1: idx + 1 + self.data_bits]
            byte_val = 0
            for bit_i in range(self.data_bits):
                byte_val |= int(data_bits_arr[bit_i]) << bit_i

            # Check parity
            if has_parity:
                rx_parity = int(bits[idx + 1 + self.data_bits])
                expected = self._compute_parity(byte_val)
                if rx_parity != expected:
                    errors.append(
                        f"Frame {frame_num}: parity error "
                        f"(got {rx_parity}, expected {expected})")

            # Check stop bit(s)
            stop_start = idx + 1 + self.data_bits + (1 if has_parity else 0)
            n_stop = int(np.ceil(self.stop_bits))
            for s in range(n_stop):
                if stop_start + s < len(bits) and bits[stop_start + s] != 1:
                    errors.append(f"Frame {frame_num}: framing error (stop bit)")

            data_bytes.append(byte_val)
            idx += frame_len
            frame_num += 1

        return {
            'data': np.array(data_bytes, dtype=np.uint8),
            'errors': errors,
        }

    def inject_error(self, bitstream: np.ndarray,
                     ber: float = 1e-3) -> np.ndarray:
        """Inject random bit errors into a bitstream.

        Parameters
        ----------
        bitstream : array_like
            Input bitstream.
        ber : float
            Bit error rate (probability of flipping each bit).

        Returns
        -------
        ndarray
            Corrupted bitstream.
        """
        bits = np.asarray(bitstream, dtype=np.int8).copy()
        mask = np.random.random(bits.shape) < ber
        bits[mask] = 1 - bits[mask]
        return bits


@_reg()
class SimulatedSPI:
    """Simulated SPI (Serial Peripheral Interface) bus.

    Parameters
    ----------
    clock_hz : float
        Clock frequency in Hz.
    mode : int
        SPI mode (0-3): CPOL | CPHA.
    bit_order : str
        'msb' (MSB first) or 'lsb' (LSB first).
    """

    def __init__(self, clock_hz: float = 1e6, mode: int = 0,
                 bit_order: str = 'msb'):
        self.clock_hz = float(clock_hz)
        self.mode = int(mode) % 4
        self.cpol = (self.mode >> 1) & 1
        self.cpha = self.mode & 1
        self.bit_order = bit_order.lower()
        self._registers: dict[int, int] = {}

    def set_register(self, addr: int, value: int) -> None:
        """Set a simulated peripheral register value."""
        self._registers[int(addr)] = int(value) & 0xFF

    def transfer(self, mosi_data: np.ndarray) -> np.ndarray:
        """Perform a full-duplex SPI transfer.

        Parameters
        ----------
        mosi_data : array_like
            Data bytes to send on MOSI line.

        Returns
        -------
        ndarray
            Data bytes received on MISO line.
        """
        mosi = np.asarray(mosi_data, dtype=np.uint8).ravel()
        miso = np.zeros_like(mosi)

        for i, byte_val in enumerate(mosi):
            # First byte is typically address/command
            if i == 0 and int(byte_val) in self._registers:
                # Read mode: return register value on next byte
                pass
            elif i > 0:
                # Return register content from previous address
                addr = int(mosi[0]) & 0x7F  # mask R/W bit
                miso[i] = self._registers.get(addr, 0xFF)

        return miso

    def clock_period(self) -> float:
        """Return clock period in seconds."""
        return 1.0 / self.clock_hz


@_reg()
class SimulatedI2C:
    """Simulated I2C (Inter-Integrated Circuit) bus.

    Parameters
    ----------
    address : int
        7-bit device address.
    clock_hz : float
        Clock frequency in Hz (standard: 100kHz, fast: 400kHz).
    """

    def __init__(self, address: int = 0x48, clock_hz: float = 100e3):
        self.address = int(address) & 0x7F
        self.clock_hz = float(clock_hz)
        self._registers: dict[int, np.ndarray] = {}

    def set_register(self, register: int, data: np.ndarray) -> None:
        """Pre-load a register with data for read simulation."""
        self._registers[int(register)] = np.asarray(data, dtype=np.uint8).ravel()

    def write(self, register: int, data: np.ndarray) -> bool:
        """Write data to a register.

        Parameters
        ----------
        register : int
            Register address (0-255).
        data : array_like
            Byte data to write.

        Returns
        -------
        bool
            True if ACK received (always True in simulation).
        """
        self._registers[int(register)] = np.asarray(data, dtype=np.uint8).ravel()
        return True  # ACK

    def read(self, register: int, nbytes: int = 1) -> np.ndarray:
        """Read data from a register.

        Parameters
        ----------
        register : int
            Register address.
        nbytes : int
            Number of bytes to read.

        Returns
        -------
        ndarray
            Byte data from the register.
        """
        reg_data = self._registers.get(int(register), np.zeros(nbytes, dtype=np.uint8))
        if len(reg_data) < nbytes:
            # Pad with zeros if register has fewer bytes than requested
            reg_data = np.concatenate([reg_data,
                                       np.zeros(nbytes - len(reg_data), dtype=np.uint8)])
        return reg_data[:nbytes]

    def scan(self) -> list[int]:
        """Return list of addresses that ACK on the bus (just self)."""
        return [self.address]


# =====================================================================
# Signal Integrity Functions
# =====================================================================

@_reg()
def add_jitter(signal: np.ndarray, rms_jitter: float,
               sample_rate: float = 1.0) -> np.ndarray:
    """Add timing jitter to a signal via interpolation.

    Parameters
    ----------
    signal : array_like
        Input signal samples.
    rms_jitter : float
        RMS jitter in seconds.
    sample_rate : float
        Sample rate in Hz.

    Returns
    -------
    ndarray
        Signal with timing jitter applied.
    """
    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)
    t_ideal = np.arange(n) / sample_rate
    # Add random timing offsets
    jitter = np.random.normal(0, rms_jitter, n)
    t_jittered = t_ideal + jitter
    # Interpolate signal at jittered time points
    return np.interp(t_jittered, t_ideal, sig)


@_reg()
def add_crosstalk(signal: np.ndarray, aggressor: np.ndarray,
                  coupling_factor: float = 0.01) -> np.ndarray:
    """Add capacitive crosstalk from an aggressor signal.

    Parameters
    ----------
    signal : array_like
        Victim signal.
    aggressor : array_like
        Aggressor signal (same length).
    coupling_factor : float
        Coupling coefficient (0 to 1).

    Returns
    -------
    ndarray
        Signal with crosstalk added.
    """
    sig = np.asarray(signal, dtype=np.float64)
    agg = np.asarray(aggressor, dtype=np.float64)
    # Capacitive coupling is proportional to derivative of aggressor
    if len(agg) > 1:
        d_agg = np.gradient(agg)
    else:
        d_agg = np.zeros_like(agg)
    return sig + coupling_factor * d_agg


@_reg()
def add_attenuation(signal: np.ndarray, freq: float,
                    cable_length: float,
                    loss_per_meter: float = 0.01) -> np.ndarray:
    """Apply frequency-dependent cable attenuation.

    Models skin-effect loss which increases with sqrt(frequency).

    Parameters
    ----------
    signal : array_like
        Input signal.
    freq : float
        Signal frequency in Hz.
    cable_length : float
        Cable length in meters.
    loss_per_meter : float
        Loss at 1 MHz per meter in dB (default 0.01 dB/m).

    Returns
    -------
    ndarray
        Attenuated signal.
    """
    sig = np.asarray(signal, dtype=np.float64)
    # Skin effect: loss proportional to sqrt(freq)
    loss_db = loss_per_meter * cable_length * np.sqrt(freq / 1e6)
    attenuation = 10.0 ** (-loss_db / 20.0)
    return sig * attenuation


# =====================================================================
# Additional Utility Functions
# =====================================================================

@_reg()
def quantization_noise_power(bits: int, vref: float = 1.0) -> float:
    """Compute quantization noise power for an ideal ADC.

    Returns
    -------
    float
        Noise power = (LSB^2) / 12.
    """
    lsb = vref / (2 ** int(bits))
    return lsb ** 2 / 12.0


@_reg()
def thermal_noise_voltage(resistance: float, temperature: float = 300.0,
                          bandwidth: float = 1e6) -> float:
    """Compute Johnson-Nyquist thermal noise voltage (RMS).

    Parameters
    ----------
    resistance : float
        Resistance in ohms.
    temperature : float
        Temperature in Kelvin.
    bandwidth : float
        Measurement bandwidth in Hz.

    Returns
    -------
    float
        RMS noise voltage in Volts.
    """
    k_B = 1.380649e-23  # Boltzmann constant
    return np.sqrt(4.0 * k_B * temperature * resistance * bandwidth)


@_reg()
def shot_noise_current(dc_current: float,
                       bandwidth: float = 1e6) -> float:
    """Compute shot noise current (RMS).

    Parameters
    ----------
    dc_current : float
        DC bias current in Amperes.
    bandwidth : float
        Measurement bandwidth in Hz.

    Returns
    -------
    float
        RMS noise current in Amperes.
    """
    q = 1.602176634e-19  # electron charge
    return np.sqrt(2.0 * q * abs(dc_current) * bandwidth)


@_reg()
def flicker_noise_psd(frequency: np.ndarray,
                      corner_freq: float = 1e3,
                      white_level: float = 1e-12) -> np.ndarray:
    """Compute 1/f (flicker) noise power spectral density.

    Parameters
    ----------
    frequency : array_like
        Frequency points in Hz (must be > 0).
    corner_freq : float
        1/f corner frequency in Hz.
    white_level : float
        White noise floor level (V^2/Hz).

    Returns
    -------
    ndarray
        PSD in V^2/Hz.
    """
    f = np.asarray(frequency, dtype=np.float64)
    f = np.maximum(f, 1e-30)  # avoid division by zero
    return white_level * (1.0 + corner_freq / f)


@_reg()
def adc_effective_resolution(bits: int, noise_rms: float,
                             vref: float = 1.0) -> float:
    """Compute effective resolution of an ADC given noise.

    Parameters
    ----------
    bits : int
        Nominal ADC resolution.
    noise_rms : float
        Total input-referred noise RMS in volts.
    vref : float
        Reference voltage.

    Returns
    -------
    float
        Effective resolution in bits.
    """
    lsb = vref / (2 ** int(bits))
    if noise_rms <= 0:
        return float(bits)
    noise_free_bits = np.log2(vref / (noise_rms * np.sqrt(12.0)))
    return min(float(bits), noise_free_bits)


@_reg()
def generate_test_signal(signal_type: str = 'sine', n_samples: int = 1024,
                         frequency: float = 100.0,
                         sample_rate: float = 10000.0,
                         amplitude: float = 1.0,
                         offset: float = 0.0) -> np.ndarray:
    """Generate a test signal for sensor validation.

    Parameters
    ----------
    signal_type : str
        'sine', 'square', 'triangle', 'sawtooth', 'noise', 'step'.
    n_samples : int
        Number of samples.
    frequency : float
        Signal frequency in Hz.
    sample_rate : float
        Sample rate in Hz.
    amplitude : float
        Peak amplitude.
    offset : float
        DC offset.

    Returns
    -------
    ndarray
        Generated signal.
    """
    t = np.arange(n_samples) / sample_rate
    sig_type = signal_type.lower()

    if sig_type == 'sine':
        sig = amplitude * np.sin(2.0 * np.pi * frequency * t)
    elif sig_type == 'square':
        sig = amplitude * np.sign(np.sin(2.0 * np.pi * frequency * t))
    elif sig_type == 'triangle':
        phase = (frequency * t) % 1.0
        sig = amplitude * (4.0 * np.abs(phase - 0.5) - 1.0)
    elif sig_type == 'sawtooth':
        phase = (frequency * t) % 1.0
        sig = amplitude * (2.0 * phase - 1.0)
    elif sig_type == 'noise':
        sig = amplitude * np.random.randn(n_samples)
    elif sig_type == 'step':
        sig = np.zeros(n_samples)
        sig[n_samples // 2:] = amplitude
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

    return sig + offset


@_reg()
def compute_sinad(signal: np.ndarray, noise_plus_distortion: np.ndarray) -> float:
    """Compute SINAD (Signal-to-Noise and Distortion ratio) in dB.

    Parameters
    ----------
    signal : array_like
        Clean reference signal.
    noise_plus_distortion : array_like
        Measured signal including noise and distortion.

    Returns
    -------
    float
        SINAD in dB.
    """
    sig = np.asarray(signal, dtype=np.float64)
    measured = np.asarray(noise_plus_distortion, dtype=np.float64)
    error = measured - sig
    sig_power = np.mean(sig ** 2)
    error_power = np.mean(error ** 2)
    if error_power < 1e-30:
        return 300.0  # effectively infinite
    return 10.0 * np.log10(sig_power / error_power)


@_reg()
def compute_thd(signal: np.ndarray, n_harmonics: int = 5) -> float:
    """Compute Total Harmonic Distortion (THD) as a ratio.

    Parameters
    ----------
    signal : array_like
        Input signal (should contain a dominant fundamental).
    n_harmonics : int
        Number of harmonics to include.

    Returns
    -------
    float
        THD as a ratio (not in dB).
    """
    sig = np.asarray(signal, dtype=np.float64)
    N = len(sig)
    spectrum = np.abs(np.fft.rfft(sig)) / N
    # Find fundamental
    fundamental_idx = np.argmax(spectrum[1:]) + 1
    fundamental_power = spectrum[fundamental_idx] ** 2
    # Sum harmonic powers
    harmonic_power = 0.0
    for h in range(2, n_harmonics + 2):
        h_idx = fundamental_idx * h
        if h_idx < len(spectrum):
            harmonic_power += spectrum[h_idx] ** 2
    if fundamental_power < 1e-30:
        return 0.0
    return np.sqrt(harmonic_power / fundamental_power)


@_reg()
def compute_sfdr(signal: np.ndarray) -> float:
    """Compute Spurious-Free Dynamic Range (SFDR) in dB.

    Parameters
    ----------
    signal : array_like
        Input signal.

    Returns
    -------
    float
        SFDR in dB.
    """
    sig = np.asarray(signal, dtype=np.float64)
    N = len(sig)
    spectrum = np.abs(np.fft.rfft(sig)) / N
    # Find fundamental
    fundamental_idx = np.argmax(spectrum[1:]) + 1
    fundamental_mag = spectrum[fundamental_idx]
    # Find largest spur (excluding DC and fundamental)
    spur_spectrum = spectrum.copy()
    spur_spectrum[0] = 0  # exclude DC
    # Exclude fundamental +/- 1 bin
    lo = max(1, fundamental_idx - 1)
    hi = min(len(spur_spectrum), fundamental_idx + 2)
    spur_spectrum[lo:hi] = 0
    largest_spur = np.max(spur_spectrum)
    if largest_spur < 1e-30:
        return 300.0
    return 20.0 * np.log10(fundamental_mag / largest_spur)
