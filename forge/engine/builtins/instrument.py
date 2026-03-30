# Copyright 2026 The Commons™
# SPDX-License-Identifier: Apache-2.0
"""Instrument Control Toolbox for Forge.

Simulated instrument communication using the sensor simulation framework.
Provides VISA-like, serial, and TCP/IP connection primitives, plus simulated
test instruments (oscilloscope, DMM, signal generator).

Target location: forge/engine/builtins/instrument.py

Backend: Pure Python (no hardware or pyvisa dependency).
"""

from __future__ import annotations

import numpy as np
from typing import Any


# ── Toolbox function registry ────────────────────────────────────
_FUNCTIONS: dict[str, callable] = {}


def _tb(name: str | None = None):
    """Local decorator to register a toolbox function."""
    def decorator(func):
        fn_name = name or func.__name__
        _FUNCTIONS[fn_name] = func
        return func
    return decorator


# =====================================================================
# Connection State
# =====================================================================

class InstrumentConnection:
    """Base class for simulated instrument connections."""

    def __init__(self, resource_id: str, conn_type: str):
        self.resource_id = resource_id
        self.conn_type = conn_type  # 'serial', 'tcpip', 'visa'
        self.is_open = False
        self.timeout = 5.0  # seconds
        self.terminator = '\n'
        self._instrument: SimulatedInstrument | None = None
        self._rx_buffer: str = ''
        self._tx_buffer: str = ''

    def __repr__(self) -> str:
        state = 'open' if self.is_open else 'closed'
        return (f"InstrumentConnection({self.conn_type!r}, "
                f"{self.resource_id!r}, {state})")


# =====================================================================
# Simulated Instruments
# =====================================================================

class SimulatedInstrument:
    """Base class for simulated instruments."""

    def __init__(self, idn: str):
        self.idn = idn
        self._settings: dict[str, Any] = {}

    def process_command(self, cmd: str) -> str | None:
        """Process a SCPI command and return response (or None for writes)."""
        cmd = cmd.strip()
        if cmd == '*IDN?':
            return self.idn
        if cmd == '*RST':
            self._settings.clear()
            return None
        if cmd == '*OPC?':
            return '1'
        if cmd == '*CLS':
            return None
        return self._handle_command(cmd)

    def _handle_command(self, cmd: str) -> str | None:
        raise NotImplementedError


class SimulatedOscilloscope(SimulatedInstrument):
    """Simulated digital oscilloscope.

    Responds to SCPI commands:
    - *IDN? -> identification string
    - :MEASURE:FREQ? -> simulated frequency measurement
    - :MEASURE:VRMS? -> simulated RMS voltage
    - :MEASURE:VPP? -> simulated peak-to-peak voltage
    - :WAVEFORM:DATA? -> simulated waveform data
    - :TIMEBASE:SCALE <val> -> set timebase
    - :CHANNEL<n>:SCALE <val> -> set vertical scale
    - :TRIGGER:LEVEL <val> -> set trigger level
    """

    def __init__(self):
        super().__init__(
            'Forge Instruments,SIM-OSC-1000,SN123456,1.0.0')
        self._settings = {
            'timebase_scale': 1e-3,
            'ch1_scale': 1.0,
            'ch2_scale': 1.0,
            'trigger_level': 0.0,
            'sample_rate': 1e9,
            'record_length': 1000,
            'frequency': 1e3,
            'amplitude': 1.0,
        }

    def _handle_command(self, cmd: str) -> str | None:
        upper = cmd.upper().strip()

        # Queries
        if upper == ':MEASURE:FREQ?':
            f = self._settings['frequency']
            noise = np.random.normal(0, f * 1e-4)
            return f'{f + noise:.6f}'

        if upper == ':MEASURE:VRMS?':
            a = self._settings['amplitude']
            vrms = a / np.sqrt(2.0)
            noise = np.random.normal(0, vrms * 1e-3)
            return f'{vrms + noise:.6f}'

        if upper == ':MEASURE:VPP?':
            a = self._settings['amplitude']
            vpp = 2.0 * a
            noise = np.random.normal(0, vpp * 1e-3)
            return f'{vpp + noise:.6f}'

        if upper == ':WAVEFORM:DATA?':
            n = self._settings['record_length']
            f = self._settings['frequency']
            sr = self._settings['sample_rate']
            a = self._settings['amplitude']
            t = np.arange(n) / sr
            waveform = a * np.sin(2.0 * np.pi * f * t)
            # Return as comma-separated values
            return ','.join(f'{v:.6e}' for v in waveform)

        if upper.startswith(':WAVEFORM:POINTS'):
            if '?' in upper:
                return str(self._settings['record_length'])
            else:
                parts = cmd.split()
                if len(parts) > 1:
                    self._settings['record_length'] = int(parts[-1])
            return None

        # Settings
        if upper.startswith(':TIMEBASE:SCALE'):
            parts = cmd.split()
            if len(parts) > 1:
                self._settings['timebase_scale'] = float(parts[-1])
            return None

        if upper.startswith(':CHANNEL') and ':SCALE' in upper:
            parts = cmd.split()
            if len(parts) > 1:
                ch = upper[8] if len(upper) > 8 else '1'
                self._settings[f'ch{ch}_scale'] = float(parts[-1])
            return None

        if upper.startswith(':TRIGGER:LEVEL'):
            parts = cmd.split()
            if len(parts) > 1:
                self._settings['trigger_level'] = float(parts[-1])
            return None

        return None


class SimulatedDMM(SimulatedInstrument):
    """Simulated Digital Multimeter.

    Responds to SCPI commands:
    - *IDN? -> identification
    - MEASURE:DC? -> DC voltage measurement
    - MEASURE:AC? -> AC voltage measurement
    - MEASURE:RESISTANCE? -> resistance measurement
    - MEASURE:CURRENT:DC? -> DC current measurement
    - CONFIGURE:VOLTAGE:DC <range> -> set DC voltage range
    - CONFIGURE:VOLTAGE:AC <range> -> set AC voltage range
    """

    def __init__(self):
        super().__init__(
            'Forge Instruments,SIM-DMM-34401,SN789012,2.0.0')
        self._settings = {
            'dc_voltage': 5.0,
            'ac_voltage': 3.536,
            'resistance': 1000.0,
            'dc_current': 0.1,
            'range_dc': 10.0,
            'range_ac': 10.0,
            'nplc': 1.0,
        }

    def _handle_command(self, cmd: str) -> str | None:
        upper = cmd.upper().strip()

        if upper == 'MEASURE:DC?' or upper == 'MEASURE:VOLTAGE:DC?':
            v = self._settings['dc_voltage']
            noise = np.random.normal(0, v * 1e-5)
            return f'{v + noise:.8f}'

        if upper == 'MEASURE:AC?' or upper == 'MEASURE:VOLTAGE:AC?':
            v = self._settings['ac_voltage']
            noise = np.random.normal(0, v * 1e-4)
            return f'{v + noise:.8f}'

        if upper == 'MEASURE:RESISTANCE?' or upper == 'MEASURE:RES?':
            r = self._settings['resistance']
            noise = np.random.normal(0, r * 1e-4)
            return f'{r + noise:.6f}'

        if upper == 'MEASURE:CURRENT:DC?':
            i = self._settings['dc_current']
            noise = np.random.normal(0, i * 1e-5)
            return f'{i + noise:.8f}'

        if upper.startswith('CONFIGURE:VOLTAGE:DC'):
            parts = cmd.split()
            if len(parts) > 1:
                self._settings['range_dc'] = float(parts[-1])
            return None

        if upper.startswith('CONFIGURE:VOLTAGE:AC'):
            parts = cmd.split()
            if len(parts) > 1:
                self._settings['range_ac'] = float(parts[-1])
            return None

        if upper.startswith('SENSE:VOLTAGE:DC:NPLC'):
            parts = cmd.split()
            if len(parts) > 1:
                self._settings['nplc'] = float(parts[-1])
            return None

        return None


class SimulatedSignalGen(SimulatedInstrument):
    """Simulated Signal / Function Generator.

    Responds to SCPI commands:
    - *IDN? -> identification
    - FREQ <value> -> set frequency
    - FREQ? -> query frequency
    - AMPL <value> -> set amplitude
    - AMPL? -> query amplitude
    - OUTPUT ON/OFF -> enable/disable output
    - OUTPUT? -> query output state
    - FUNC <SINE|SQUARE|TRIANGLE|RAMP|NOISE> -> set waveform
    - FUNC? -> query waveform type
    """

    def __init__(self):
        super().__init__(
            'Forge Instruments,SIM-FGEN-33220,SN345678,1.5.0')
        self._settings = {
            'frequency': 1000.0,
            'amplitude': 1.0,
            'offset': 0.0,
            'output': False,
            'function': 'SINE',
        }

    def _handle_command(self, cmd: str) -> str | None:
        upper = cmd.upper().strip()
        parts = cmd.strip().split()

        if upper == 'FREQ?':
            return f'{self._settings["frequency"]:.6f}'
        if upper.startswith('FREQ') and len(parts) > 1:
            self._settings['frequency'] = float(parts[-1])
            return None

        if upper == 'AMPL?':
            return f'{self._settings["amplitude"]:.6f}'
        if upper.startswith('AMPL') and len(parts) > 1:
            self._settings['amplitude'] = float(parts[-1])
            return None

        if upper == 'OUTPUT?':
            return '1' if self._settings['output'] else '0'
        if upper.startswith('OUTPUT'):
            if 'ON' in upper:
                self._settings['output'] = True
            elif 'OFF' in upper:
                self._settings['output'] = False
            return None

        if upper == 'FUNC?':
            return self._settings['function']
        if upper.startswith('FUNC') and len(parts) > 1:
            self._settings['function'] = parts[-1].upper()
            return None

        if upper.startswith('VOLT:OFFS') and len(parts) > 1:
            self._settings['offset'] = float(parts[-1])
            return None
        if upper == 'VOLT:OFFS?':
            return f'{self._settings["offset"]:.6f}'

        return None


# ── Instrument factory lookup ────────────────────────────────────

_INSTRUMENT_TYPES: dict[str, type] = {
    'oscilloscope': SimulatedOscilloscope,
    'osc': SimulatedOscilloscope,
    'scope': SimulatedOscilloscope,
    'dmm': SimulatedDMM,
    'multimeter': SimulatedDMM,
    'siggen': SimulatedSignalGen,
    'fgen': SimulatedSignalGen,
    'generator': SimulatedSignalGen,
}


def _resolve_instrument(resource: str) -> SimulatedInstrument:
    """Resolve a resource string to a simulated instrument.

    Resource format examples:
    - 'TCPIP::192.168.1.1::INSTR' -> default oscilloscope
    - 'GPIB0::22::INSTR' -> default DMM
    - 'USB0::0x1234::0x5678::INSTR' -> default signal generator
    - 'sim://oscilloscope' -> explicit instrument type
    - 'sim://dmm' -> explicit instrument type
    """
    lower = resource.lower()

    # Explicit sim:// protocol
    if lower.startswith('sim://'):
        itype = lower[6:].strip()
        cls = _INSTRUMENT_TYPES.get(itype)
        if cls:
            return cls()
        raise ValueError(f"Unknown simulated instrument type: {itype}. "
                         f"Available: {list(_INSTRUMENT_TYPES.keys())}")

    # Heuristic from VISA resource string
    if 'GPIB' in resource.upper():
        return SimulatedDMM()
    elif 'USB' in resource.upper():
        return SimulatedSignalGen()
    else:
        return SimulatedOscilloscope()


# =====================================================================
# Connection Functions (Forge builtins)
# =====================================================================

@_tb('serial')
def forge_serial(port: str = 'COM1', baud: int = 9600) -> InstrumentConnection:
    """Create a simulated serial instrument connection.

    Parameters
    ----------
    port : str
        Serial port name (e.g. 'COM1', '/dev/ttyUSB0').
    baud : int
        Baud rate.

    Returns
    -------
    InstrumentConnection
        Connection object (not yet open).
    """
    conn = InstrumentConnection(f'{port}@{baud}', 'serial')
    conn._instrument = SimulatedDMM()  # default for serial
    return conn


@_tb('tcpip')
def forge_tcpip(host: str = '192.168.1.1',
                port: int = 5025) -> InstrumentConnection:
    """Create a simulated TCP/IP instrument connection.

    Parameters
    ----------
    host : str
        Hostname or IP address.
    port : int
        TCP port number.

    Returns
    -------
    InstrumentConnection
        Connection object (not yet open).
    """
    conn = InstrumentConnection(f'{host}:{port}', 'tcpip')
    conn._instrument = SimulatedOscilloscope()
    return conn


@_tb('visa')
def forge_visa(resource: str) -> InstrumentConnection:
    """Create a simulated VISA instrument connection.

    Parameters
    ----------
    resource : str
        VISA resource string (e.g. 'TCPIP::192.168.1.1::INSTR',
        'GPIB0::22::INSTR', 'sim://oscilloscope').

    Returns
    -------
    InstrumentConnection
        Connection object (not yet open).
    """
    conn = InstrumentConnection(resource, 'visa')
    conn._instrument = _resolve_instrument(resource)
    return conn


@_tb('fopen_inst')
def forge_fopen_inst(inst: InstrumentConnection) -> InstrumentConnection:
    """Open a simulated instrument connection.

    Parameters
    ----------
    inst : InstrumentConnection
        Connection from forge_serial, forge_tcpip, or forge_visa.

    Returns
    -------
    InstrumentConnection
        The same connection, now in 'open' state.
    """
    if not isinstance(inst, InstrumentConnection):
        raise TypeError("Expected InstrumentConnection object")
    inst.is_open = True
    inst._rx_buffer = ''
    inst._tx_buffer = ''
    return inst


@_tb('fclose_inst')
def forge_fclose_inst(inst: InstrumentConnection) -> None:
    """Close a simulated instrument connection.

    Parameters
    ----------
    inst : InstrumentConnection
        An open connection.
    """
    if not isinstance(inst, InstrumentConnection):
        raise TypeError("Expected InstrumentConnection object")
    inst.is_open = False
    inst._rx_buffer = ''
    inst._tx_buffer = ''


def _check_open(inst: InstrumentConnection) -> None:
    """Raise if connection is not open."""
    if not isinstance(inst, InstrumentConnection):
        raise TypeError("Expected InstrumentConnection object")
    if not inst.is_open:
        raise RuntimeError(
            f"Instrument connection {inst.resource_id!r} is not open. "
            f"Call forge_fopen_inst first.")


@_tb('fprintf_inst')
def forge_fprintf_inst(inst: InstrumentConnection, cmd: str) -> None:
    """Send a command string to a simulated instrument.

    Parameters
    ----------
    inst : InstrumentConnection
        An open connection.
    cmd : str
        SCPI command string.
    """
    _check_open(inst)
    # Process command and store any response in rx buffer
    response = inst._instrument.process_command(str(cmd))
    if response is not None:
        inst._rx_buffer = response + inst.terminator
    else:
        inst._rx_buffer = ''


@_tb('fscanf_inst')
def forge_fscanf_inst(inst: InstrumentConnection,
                      fmt: str = '%s') -> str:
    """Read a response string from a simulated instrument.

    Parameters
    ----------
    inst : InstrumentConnection
        An open connection.
    fmt : str
        Format string (for compatibility; returns string in simulation).

    Returns
    -------
    str
        Response from the instrument.
    """
    _check_open(inst)
    response = inst._rx_buffer.strip()
    inst._rx_buffer = ''
    return response


@_tb('fread_inst')
def forge_fread_inst(inst: InstrumentConnection,
                     size: int = 1024) -> np.ndarray:
    """Binary read from a simulated instrument.

    Parameters
    ----------
    inst : InstrumentConnection
        An open connection.
    size : int
        Number of bytes to read.

    Returns
    -------
    ndarray
        Byte data as uint8 array.
    """
    _check_open(inst)
    data = inst._rx_buffer[:size]
    inst._rx_buffer = inst._rx_buffer[size:]
    return np.frombuffer(data.encode('ascii', errors='replace'),
                         dtype=np.uint8)


@_tb('fwrite_inst')
def forge_fwrite_inst(inst: InstrumentConnection,
                      data: np.ndarray) -> int:
    """Binary write to a simulated instrument.

    Parameters
    ----------
    inst : InstrumentConnection
        An open connection.
    data : array_like
        Byte data to write.

    Returns
    -------
    int
        Number of bytes written.
    """
    _check_open(inst)
    byte_data = np.asarray(data, dtype=np.uint8)
    cmd_str = byte_data.tobytes().decode('ascii', errors='replace')
    response = inst._instrument.process_command(cmd_str.strip())
    if response is not None:
        inst._rx_buffer = response + inst.terminator
    return len(byte_data)


@_tb('query')
def forge_query(inst: InstrumentConnection, cmd: str) -> str:
    """Send a query command and read the response (combined write+read).

    Parameters
    ----------
    inst : InstrumentConnection
        An open connection.
    cmd : str
        SCPI query command (should end with '?').

    Returns
    -------
    str
        Instrument response.
    """
    _check_open(inst)
    response = inst._instrument.process_command(str(cmd))
    if response is None:
        return ''
    return response


# =====================================================================
# Utility Functions
# =====================================================================

@_tb('instrhwinfo')
def forge_instrhwinfo(conn_type: str = 'all') -> dict:
    """Return information about available simulated instruments.

    Parameters
    ----------
    conn_type : str
        Connection type filter: 'serial', 'tcpip', 'visa', or 'all'.

    Returns
    -------
    dict
        Dictionary with instrument information.
    """
    info = {
        'serial': {
            'ports': ['COM1', 'COM2', '/dev/ttyUSB0'],
            'instruments': ['SimulatedDMM'],
        },
        'tcpip': {
            'hosts': ['192.168.1.1', '192.168.1.2'],
            'instruments': ['SimulatedOscilloscope'],
        },
        'visa': {
            'resources': [
                'TCPIP::192.168.1.1::INSTR',
                'GPIB0::22::INSTR',
                'USB0::0x1234::0x5678::INSTR',
                'sim://oscilloscope',
                'sim://dmm',
                'sim://siggen',
            ],
            'instruments': list(_INSTRUMENT_TYPES.keys()),
        },
    }
    if conn_type.lower() == 'all':
        return info
    return info.get(conn_type.lower(), {})


@_tb('instrfind')
def forge_instrfind(conn_type: str | None = None) -> list[str]:
    """Find available simulated instrument resources.

    Parameters
    ----------
    conn_type : str, optional
        Filter by connection type.

    Returns
    -------
    list of str
        Available resource strings.
    """
    resources = [
        'sim://oscilloscope',
        'sim://dmm',
        'sim://siggen',
    ]
    if conn_type:
        ct = conn_type.lower()
        if ct in ('serial', 'com'):
            return ['COM1', 'COM2']
        elif ct in ('tcpip', 'tcp'):
            return ['192.168.1.1:5025']
    return resources


@_tb('set_instrument_value')
def forge_set_instrument_value(inst: InstrumentConnection,
                               setting: str, value: Any) -> None:
    """Directly set an internal setting on a simulated instrument.

    Useful for controlling what measurements the simulated instrument returns.

    Parameters
    ----------
    inst : InstrumentConnection
        An open connection.
    setting : str
        Setting name (e.g. 'frequency', 'amplitude', 'dc_voltage').
    value : any
        Value to set.
    """
    _check_open(inst)
    if setting in inst._instrument._settings:
        inst._instrument._settings[setting] = value
    else:
        raise KeyError(
            f"Unknown setting: {setting!r}. "
            f"Available: {list(inst._instrument._settings.keys())}")


@_tb('get_instrument_value')
def forge_get_instrument_value(inst: InstrumentConnection,
                               setting: str) -> Any:
    """Read an internal setting from a simulated instrument.

    Parameters
    ----------
    inst : InstrumentConnection
        An open connection.
    setting : str
        Setting name.

    Returns
    -------
    any
        Current value.
    """
    _check_open(inst)
    if setting in inst._instrument._settings:
        return inst._instrument._settings[setting]
    raise KeyError(
        f"Unknown setting: {setting!r}. "
        f"Available: {list(inst._instrument._settings.keys())}")


# =====================================================================
# Instrument Registry
# =====================================================================

INSTRUMENT_REGISTRY: dict[str, callable] = dict(_FUNCTIONS)


# ── Registration ─────────────────────────────────────────────────
def _load() -> dict[str, callable]:
    return dict(_FUNCTIONS)


