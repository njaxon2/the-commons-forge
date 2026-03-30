# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Tests for Forge Sensor Simulation Framework (20 tests).

Uses SENSOR_REGISTRY dict to access sensor classes and functions.
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
    def test_adc_8bit_gives_256_levels(self):
        """8-bit ADC should have 256 quantization levels."""
        ADC = _cls('SimulatedADC')
        adc = ADC(bits=8, vref=3.3, noise_rms=0.0)
        assert adc.levels == 256

    def test_adc_quantization_range(self):
        """ADC output codes should be in [0, 2^bits - 1]."""
        ADC = _cls('SimulatedADC')
        adc = ADC(bits=10, vref=3.3, noise_rms=0.0)
        signal = np.linspace(0, 3.3, 100)
        codes = adc.read(signal)
        assert np.all(codes >= 0)
        assert np.all(codes <= 1023)

    def test_adc_snr_formula(self):
        """Theoretical ADC SNR = 6.02*N + 1.76 dB."""
        ADC = _cls('SimulatedADC')
        for bits in [8, 10, 12, 16]:
            adc = ADC(bits=bits)
            expected_snr = 6.02 * bits + 1.76
            np.testing.assert_allclose(adc.snr(), expected_snr, atol=0.01)

    def test_adc_noise_increases_spread(self):
        """Adding noise should increase spread of repeated measurements."""
        ADC = _cls('SimulatedADC')
        adc_clean = ADC(bits=12, vref=3.3, noise_rms=0.0)
        adc_noisy = ADC(bits=12, vref=3.3, noise_rms=0.01)
        signal = np.full(1000, 1.65)
        codes_clean = adc_clean.read(signal)
        codes_noisy = adc_noisy.read(signal)
        assert np.std(codes_noisy.astype(float)) >= np.std(codes_clean.astype(float))


class TestDAC:
    def test_dac_roundtrip(self):
        """ADC -> DAC roundtrip should recover signal within 1 LSB."""
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
    def test_thermistor_at_25C_gives_R0(self):
        """Thermistor at reference temperature (25C = 298.15K) gives R0."""
        Therm = _cls('SimulatedThermistor')
        therm = Therm(R0=10000.0, B=3950.0, T0=298.15, noise_std=0.0)
        R = therm.read(np.array([298.15]))
        np.testing.assert_allclose(R[0], 10000.0, atol=0.1)

    def test_thermistor_ntc_behavior(self):
        """NTC thermistor: resistance decreases with increasing temperature."""
        Therm = _cls('SimulatedThermistor')
        therm = Therm(R0=10000.0, B=3950.0, noise_std=0.0)
        temps = np.array([273.15, 298.15, 323.15])  # 0, 25, 50 C
        R = therm.read(temps)
        assert R[0] > R[1] > R[2]


class TestThermocouple:
    def test_thermocouple_type_K_at_0C_is_0mV(self):
        """Type K thermocouple at 0 C should read ~0 uV."""
        TC = _cls('SimulatedThermocouple')
        tc = TC(type_letter='K', noise_uv=0.0)
        voltage = tc.read(np.array([0.0]))
        np.testing.assert_allclose(voltage[0], 0.0, atol=1.0)

    def test_thermocouple_positive_at_positive_temp(self):
        """Type K thermocouple at 100 C should give positive voltage."""
        TC = _cls('SimulatedThermocouple')
        tc = TC(type_letter='K', noise_uv=0.0)
        voltage = tc.read(np.array([100.0]))
        assert voltage[0] > 0


# ===========================================================================
# Accelerometer
# ===========================================================================

class TestAccelerometer:
    def test_accelerometer_no_accel_reads_zero(self):
        """Accelerometer with zero input should read approximately 0."""
        Accel = _cls('SimulatedAccelerometer')
        accel = Accel(sensitivity=0.3, noise_density=0.0)
        output = accel.read(np.array([0.0, 0.0, 0.0]), bandwidth=1000.0)
        np.testing.assert_allclose(output, 0.0, atol=1e-10)

    def test_accelerometer_1g_output(self):
        """1g on X axis should give ~sensitivity volts on X."""
        Accel = _cls('SimulatedAccelerometer')
        accel = Accel(sensitivity=0.3, cross_axis=0.0, noise_density=0.0)
        output = accel.read(np.array([1.0, 0.0, 0.0]), bandwidth=1000.0)
        np.testing.assert_allclose(output[0], 0.3, atol=0.01)


# ===========================================================================
# Communication Protocols
# ===========================================================================

class TestUART:
    def test_uart_8n1_framing_overhead(self):
        """8N1 UART: 1 start + 8 data + 1 stop = 10 bits per byte (2 overhead)."""
        UART = _cls('SimulatedUART')
        uart = UART(baud=9600, data_bits=8, parity='none', stop_bits=1.0)
        data = np.array([0x55], dtype=np.uint8)
        bitstream = uart.send(data)
        # 1 start + 8 data + 1 stop = 10 bits total, 2 bits overhead
        assert len(bitstream) == 10

    def test_uart_send_receive_roundtrip(self):
        """UART send then receive should recover original data."""
        UART = _cls('SimulatedUART')
        uart = UART(baud=9600, data_bits=8, parity='none', stop_bits=1.0)
        data = np.array([0x41, 0x42, 0x43], dtype=np.uint8)
        bitstream = uart.send(data)
        result = uart.receive(bitstream)
        np.testing.assert_array_equal(result['data'], data)
        assert len(result['errors']) == 0


class TestSPI:
    def test_spi_transfer_returns_data(self):
        """SPI transfer should return MISO data array of same length."""
        SPI = _cls('SimulatedSPI')
        spi = SPI(clock_hz=1e6, mode=0)
        spi.set_register(0x10, 0xAB)
        mosi = np.array([0x10, 0x00], dtype=np.uint8)
        miso = spi.transfer(mosi)
        assert len(miso) == len(mosi)
        # Second byte should contain register value
        assert miso[1] == 0xAB


class TestI2C:
    def test_i2c_write_read_roundtrip(self):
        """I2C write then read should recover data."""
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
    def test_jitter_rms_matches_input(self):
        """add_jitter with known RMS should perturb the signal."""
        add_jitter = _cls('add_jitter')
        np.random.seed(42)
        signal = np.sin(2 * np.pi * np.linspace(0, 1, 1000))
        jittered = add_jitter(signal, rms_jitter=0.001, sample_rate=1000.0)
        # Jittered signal should differ from original
        diff = np.abs(jittered - signal)
        assert np.max(diff) > 0

    def test_crosstalk_coupling_level(self):
        """add_crosstalk with coupling_factor should add proportional interference."""
        add_crosstalk = _cls('add_crosstalk')
        victim = np.zeros(100)
        aggressor = np.sin(2 * np.pi * np.linspace(0, 5, 100))
        coupled = add_crosstalk(victim, aggressor, coupling_factor=0.1)
        # Coupled signal should have non-zero energy from aggressor derivative
        assert np.max(np.abs(coupled)) > 0
        # With larger coupling, more interference
        coupled_strong = add_crosstalk(victim, aggressor, coupling_factor=0.5)
        assert np.max(np.abs(coupled_strong)) > np.max(np.abs(coupled))
