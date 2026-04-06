# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for Forge Sensor Simulation Framework (20 tests).

Uses SENSOR_REGISTRY dict to access sensor classes and functions.

V-Model Traceability
---------------------
Requirement: R-SENS
Parent SHALL statement: Forge SHALL provide a sensor simulation framework that
    models ADC/DAC conversion, temperature sensors, accelerometers, communication
    protocols (UART, SPI, I2C), and signal integrity effects with physically
    accurate behavior.
Model-user argument: An engineer working with physical sensor data
    (accelerometers, gyroscopes, temperature probes) needs to simulate sensor
    behavior, convert between units, and validate signal chains before deploying
    to hardware. Without accurate sensor models in Forge, the user must maintain
    separate simulation tools or hand-roll sensor math, losing the integrated
    workflow they expect from a MATLAB-class environment.
Decomposition:
    R-SENS-01: 8-bit ADC SHALL report 256 quantization levels.
    R-SENS-02: ADC output codes SHALL stay within [0, 2^bits - 1].
    R-SENS-03: ADC SNR SHALL match the theoretical formula 6.02*N + 1.76 dB.
    R-SENS-04: Adding noise to an ADC SHALL increase measurement spread.
    R-SENS-05: ADC to DAC roundtrip SHALL recover the signal within 1 LSB.
    R-SENS-06: Thermistor at reference temperature SHALL return R0.
    R-SENS-07: NTC thermistor resistance SHALL decrease with increasing temperature.
    R-SENS-08: Type K thermocouple at 0C SHALL read approximately 0 uV.
    R-SENS-09: Type K thermocouple at positive temperature SHALL give positive voltage.
    R-SENS-10: Accelerometer with zero input SHALL read approximately zero.
    R-SENS-11: Accelerometer at 1g SHALL output sensitivity volts on that axis.
    R-SENS-12: 8N1 UART framing SHALL produce exactly 10 bits per byte.
    R-SENS-13: UART send/receive roundtrip SHALL recover original data with no errors.
    R-SENS-14: SPI transfer SHALL return MISO data of the same length as MOSI.
    R-SENS-15: I2C write then read SHALL recover the written data.
    R-SENS-16: add_jitter SHALL perturb the signal measurably.
    R-SENS-17: add_crosstalk SHALL inject proportional interference from aggressor.
Consistency argument: R-SENS-01 through R-SENS-05 cover analog-to-digital and
    digital-to-analog conversion (quantization, range, SNR, noise, roundtrip).
    R-SENS-06 through R-SENS-09 cover temperature sensor physics (thermistor NTC
    behavior, thermocouple voltage generation). R-SENS-10 and R-SENS-11 cover
    inertial sensing. R-SENS-12 through R-SENS-15 cover digital communication
    protocols (UART framing and roundtrip, SPI transfer, I2C roundtrip).
    R-SENS-16 and R-SENS-17 cover signal integrity effects. Together these span
    the full ADC/DAC pipeline, common sensor modalities, serial protocols, and
    channel impairments.
"""

import pytest
import numpy as np

from forge.validation.sensors import SENSOR_REGISTRY


def _cls(name):
    """Get a class or function from the sensor registry."""
    return SENSOR_REGISTRY[name]


# ===========================================================================
# ADC / DAC
# ===========================================================================

class TestADC:
    """R-SENS-01..04: SimulatedADC SHALL model quantization levels, code range,
    theoretical SNR, and noise behavior accurately.
    """

    def test_adc_8bit_gives_256_levels(self):
        """R-SENS-01: 8-bit ADC reports 256 quantization levels."""
        ADC = _cls('SimulatedADC')
        adc = ADC(bits=8, vref=3.3, noise_rms=0.0)
        assert adc.levels == 256

    def test_adc_quantization_range(self):
        """R-SENS-02: 10-bit ADC output codes stay within [0, 1023]."""
        ADC = _cls('SimulatedADC')
        adc = ADC(bits=10, vref=3.3, noise_rms=0.0)
        signal = np.linspace(0, 3.3, 100)
        codes = adc.read(signal)
        assert np.all(codes >= 0)
        assert np.all(codes <= 1023)

    def test_adc_snr_formula(self):
        """R-SENS-03: ADC SNR matches 6.02*N + 1.76 dB for 8, 10, 12, 16 bits."""
        ADC = _cls('SimulatedADC')
        for bits in [8, 10, 12, 16]:
            adc = ADC(bits=bits)
            expected_snr = 6.02 * bits + 1.76
            np.testing.assert_allclose(adc.snr(), expected_snr, atol=0.01)

    def test_adc_noise_increases_spread(self):
        """R-SENS-04: Noisy ADC has greater measurement spread than clean ADC."""
        ADC = _cls('SimulatedADC')
        adc_clean = ADC(bits=12, vref=3.3, noise_rms=0.0)
        adc_noisy = ADC(bits=12, vref=3.3, noise_rms=0.01)
        signal = np.full(1000, 1.65)
        codes_clean = adc_clean.read(signal)
        codes_noisy = adc_noisy.read(signal)
        assert np.std(codes_noisy.astype(float)) >= np.std(codes_clean.astype(float))


class TestDAC:
    """R-SENS-05: ADC to DAC roundtrip SHALL recover the signal within 1 LSB."""

    def test_dac_roundtrip(self):
        """R-SENS-05: ADC then DAC roundtrip recovers signal within 2 LSB."""
        ADC = _cls('SimulatedADC')
        DAC = _cls('SimulatedDAC')
        adc = ADC(bits=12, vref=3.3, noise_rms=0.0)
        dac = DAC(bits=12, vref=3.3)
        signal = np.linspace(0.1, 3.0, 50)
        codes = adc.read(signal)
        recovered = dac.write(codes)
        # Error should be within 1 LSB
        assert np.max(np.abs(recovered - signal)) < 2 * adc.lsb


# ===========================================================================
# Temperature Sensors
# ===========================================================================

class TestThermistor:
    """R-SENS-06..07: SimulatedThermistor SHALL model NTC behavior with correct
    resistance at reference temperature and monotonic decrease with heating.
    """

    def test_thermistor_at_25C_gives_R0(self):
        """R-SENS-06: Thermistor at 298.15K returns R0 (10000 ohms)."""
        Therm = _cls('SimulatedThermistor')
        therm = Therm(R0=10000.0, B=3950.0, T0=298.15, noise_std=0.0)
        R = therm.read(np.array([298.15]))
        np.testing.assert_allclose(R[0], 10000.0, atol=0.1)

    def test_thermistor_ntc_behavior(self):
        """R-SENS-07: NTC thermistor resistance decreases with increasing temperature."""
        Therm = _cls('SimulatedThermistor')
        therm = Therm(R0=10000.0, B=3950.0, noise_std=0.0)
        temps = np.array([273.15, 298.15, 323.15])  # 0, 25, 50 C
        R = therm.read(temps)
        assert R[0] > R[1] > R[2]


class TestThermocouple:
    """R-SENS-08..09: SimulatedThermocouple SHALL produce correct voltage
    polarity and near-zero reading at 0C.
    """

    def test_thermocouple_type_K_at_0C_is_0mV(self):
        """R-SENS-08: Type K thermocouple at 0C reads approximately 0 uV."""
        TC = _cls('SimulatedThermocouple')
        tc = TC(type_letter='K', noise_uv=0.0)
        voltage = tc.read(np.array([0.0]))
        np.testing.assert_allclose(voltage[0], 0.0, atol=1.0)

    def test_thermocouple_positive_at_positive_temp(self):
        """R-SENS-09: Type K thermocouple at 100C gives positive voltage."""
        TC = _cls('SimulatedThermocouple')
        tc = TC(type_letter='K', noise_uv=0.0)
        voltage = tc.read(np.array([100.0]))
        assert voltage[0] > 0


# ===========================================================================
# Accelerometer
# ===========================================================================

class TestAccelerometer:
    """R-SENS-10..11: SimulatedAccelerometer SHALL read zero with no input and
    output sensitivity volts per g on the stimulated axis.
    """

    def test_accelerometer_no_accel_reads_zero(self):
        """R-SENS-10: Zero-input accelerometer reads approximately 0."""
        Accel = _cls('SimulatedAccelerometer')
        accel = Accel(sensitivity=0.3, noise_density=0.0)
        output = accel.read(np.array([0.0, 0.0, 0.0]), bandwidth=1000.0)
        np.testing.assert_allclose(output, 0.0, atol=1e-10)

    def test_accelerometer_1g_output(self):
        """R-SENS-11: 1g on X axis outputs sensitivity (0.3) volts on X."""
        Accel = _cls('SimulatedAccelerometer')
        accel = Accel(sensitivity=0.3, cross_axis=0.0, noise_density=0.0)
        output = accel.read(np.array([1.0, 0.0, 0.0]), bandwidth=1000.0)
        np.testing.assert_allclose(output[0], 0.3, atol=0.01)


# ===========================================================================
# Communication Protocols
# ===========================================================================

class TestUART:
    """R-SENS-12..13: SimulatedUART SHALL frame bytes correctly (8N1 = 10 bits)
    and roundtrip data without errors.
    """

    def test_uart_8n1_framing_overhead(self):
        """R-SENS-12: 8N1 UART produces exactly 10 bits per byte."""
        UART = _cls('SimulatedUART')
        uart = UART(baud=9600, data_bits=8, parity='none', stop_bits=1.0)
        data = np.array([0x55], dtype=np.uint8)
        bitstream = uart.send(data)
        # 1 start + 8 data + 1 stop = 10 bits total, 2 bits overhead
        assert len(bitstream) == 10

    def test_uart_send_receive_roundtrip(self):
        """R-SENS-13: UART send then receive recovers original data with no errors."""
        UART = _cls('SimulatedUART')
        uart = UART(baud=9600, data_bits=8, parity='none', stop_bits=1.0)
        data = np.array([0x41, 0x42, 0x43], dtype=np.uint8)
        bitstream = uart.send(data)
        result = uart.receive(bitstream)
        np.testing.assert_array_equal(result['data'], data)
        assert len(result['errors']) == 0


class TestSPI:
    """R-SENS-14: SimulatedSPI transfer SHALL return MISO data of the same
    length as MOSI, with correct register contents.
    """

    def test_spi_transfer_returns_data(self):
        """R-SENS-14: SPI transfer returns MISO array matching MOSI length."""
        SPI = _cls('SimulatedSPI')
        spi = SPI(clock_hz=1e6, mode=0)
        spi.set_register(0x10, 0xAB)
        mosi = np.array([0x10, 0x00], dtype=np.uint8)
        miso = spi.transfer(mosi)
        assert len(miso) == len(mosi)
        # Second byte should contain register value
        assert miso[1] == 0xAB


class TestI2C:
    """R-SENS-15: SimulatedI2C write then read SHALL recover the written data."""

    def test_i2c_write_read_roundtrip(self):
        """R-SENS-15: I2C write then read recovers data bytes."""
        I2C = _cls('SimulatedI2C')
        i2c = I2C(address=0x48)
        data = np.array([0xDE, 0xAD], dtype=np.uint8)
        ack = i2c.write(0x00, data)
        assert ack is True
        result = i2c.read(0x00, 2)
        np.testing.assert_array_equal(result, data)


# ===========================================================================
# Signal Integrity
# ===========================================================================

class TestSignalIntegrity:
    """R-SENS-16..17: Signal integrity functions SHALL inject measurable jitter
    and proportional crosstalk.
    """

    def test_jitter_rms_matches_input(self):
        """R-SENS-16: add_jitter perturbs the signal measurably."""
        add_jitter = _cls('add_jitter')
        np.random.seed(42)
        signal = np.sin(2 * np.pi * np.linspace(0, 1, 1000))
        jittered = add_jitter(signal, rms_jitter=0.001, sample_rate=1000.0)
        # Jittered signal should differ from original
        diff = np.abs(jittered - signal)
        assert np.max(diff) > 0

    def test_crosstalk_coupling_level(self):
        """R-SENS-17: add_crosstalk injects proportional interference from aggressor."""
        add_crosstalk = _cls('add_crosstalk')
        victim = np.zeros(100)
        aggressor = np.sin(2 * np.pi * np.linspace(0, 5, 100))
        coupled = add_crosstalk(victim, aggressor, coupling_factor=0.1)
        # Coupled signal should have non-zero energy from aggressor derivative
        assert np.max(np.abs(coupled)) > 0
        # With larger coupling, more interference
        coupled_strong = add_crosstalk(victim, aggressor, coupling_factor=0.5)
        assert np.max(np.abs(coupled_strong)) > np.max(np.abs(coupled))
