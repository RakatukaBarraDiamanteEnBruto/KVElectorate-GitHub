"""
KVE Inductive Coil Test Bench — app_v2_0.py
=============================================
State machine controller for the KVE test bench.

Features implemented:
    - State machine: IDLE, PRECHECK_STATIC, PRECHECK_DYNAMIC, RUNNING, COMPLETED, ERROR
    - LED tower + button LED threading
    - 8x MCP9600 thermocouples via TCA9548A multiplexer
    - ADS1115 flow/WFS-temp sensors (4 channels, one mux slot)
    - Temperature conditioning (heater + fans) in IDLE
    - SD card CSV logger (3-section combined file for static tests)
    - Button interrupt handling (emergency stop from any active state)
    - Dynamic mode: TC-only data collection

NEW IN v2_0:
    [GENERATOR]  GeneratorController class — relay-driven generator start/stop via GPIO12
    [GENERATOR]  GENERATOR_START_DELAY_S — adjustable delay before generator fires in RUNNING
    [PICO]       PicoscopeController class — full PicoScope 2204A acquisition and processing
    [PICO]       Bandpass filter 250–500 kHz replacing old high-pass
    [PICO]       ps2000_ready() polling with timeout replacing bare time.sleep()
    [WFS]        Corrected flow/temp conversions: 4–20 mA via 0–3 V I-to-V converter
    [WFS]        channel_type parameter on add_flow_sensor() ('flow' or 'temp')
    [WFS]        rec_wfs_temp storage split from rec_flow
    [CSV]        SDWriter rewritten — single 3-section combined CSV per static test
    [CSV]        Section 1: TC + flow + WFS temp rows
    [CSV]        Section 2: Pico RMS summary (all blocks)
    [CSV]        Section 3: Block 63 full signal (8000 rows)

Notes:
    Tested on RPi 5. Push to preprod after bench verification.
    ADS1115 ADDR pin → GND → address 0x48 (default).
    Generator relay on GPIO12 (Gen_input_pin per GPIO layout sheet).
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Standard library
# ─────────────────────────────────────────────────────────────────────────────
import time
import threading
import csv
import sys
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
#  Hardware / GPIO
# ─────────────────────────────────────────────────────────────────────────────
import board
import busio
import RPi.GPIO as GPIO
import gpiozero
from gpiozero import Button, LED, DigitalOutputDevice

# ─────────────────────────────────────────────────────────────────────────────
#  Sensor libraries
# ─────────────────────────────────────────────────────────────────────────────
import adafruit_tca9548a
from adafruit_mcp9600 import MCP9600
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# ─────────────────────────────────────────────────────────────────────────────
#  Numerical / signal processing
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from scipy.signal import butter, filtfilt

# ──────────────────────────────────────────────────────────────────────────────
#  [PICO] PicoScope SDK
# ──────────────────────────────────────────────────────────────────────────────
import ctypes
from picosdk.ps2000 import ps2000 as ps
from picosdk.functions import adc2mV, assert_pico2000_ok


# =============================================================================
#  [GENERATOR] [PICO]  TOP-LEVEL CONFIGURATION VARIABLES
#  Adjust these to change test behaviour without touching any other code.
# =============================================================================

# ── [GENERATOR] Relay / generator ────────────────────────────────────────────
GENERATOR_PIN           = 12     # GPIO12 — relay output (Gen_input_pin)
GENERATOR_START_DELAY_S = 2.0    # seconds after RUNNING starts before generator fires

# ── Test duration (shared by TC loop and Pico loop) ──────────────────────────
TEST_DURATION_S         = 10     # seconds

# ── [PICO] PicoScope acquisition ─────────────────────────────────────────────
TIMEBASE                = 1
OVERSAMPLE              = 1
N_SAMPLES               = 8000
DT                      = 0.00000002          # 20 ns per sample
MPC                     = 0.000001256637061436
BS                      = 0.545
CP                      = -4.904
PICO_INTERVAL_S         = 0.25   # seconds between block captures
PICO_BLOCK_PLOT         = 63     # which block to store in CSV section 3


# =============================================================================
#  GPIO SETUP
# =============================================================================
GPIO.setmode(GPIO.BCM)

# Switch pins
SWITCH_PIN_1 = 19
SWITCH_PIN_2 = 26
switch1 = Button(SWITCH_PIN_1, pull_up=False)
switch2 = Button(SWITCH_PIN_2, pull_up=False)

# Tower LEDs
LED1 = LED(25)   # PRECHECK
LED2 = LED(16)   # RUNNING STATIC
LED3 = LED(20)   # RUNNING DYNAMIC
LED4 = LED(21)   # ERROR

# Button + button LED
button = Button(5)
bLED   = LED(13)


# =============================================================================
#  MULTIPLEXED THERMOCOUPLE + FLOW SENSOR LOGGER
# =============================================================================
class MultiplexedThermocoupleLogger:
    def __init__(self, mux_address=0x70):
        self.I2C_SDA_PIN  = board.SDA
        self.I2C_SCL_PIN  = board.SCL
        self.MUX_ADDRESS  = mux_address

        self.thermocouple_config = []   # [(mux_ch, addr), ...]
        # [WFS] flow_sensor_config now stores channel_type alongside other fields
        self.flow_sensor_config  = []   # [(mux_ch, ads_addr, ads_ch, channel_type), ...]

        # Data storage
        self.rec_temp     = []
        self.rec_flow     = []
        # [WFS] Separate storage list for WFS temperature channels
        self.rec_wfs_temp = []
        self.rec_time     = []

        self.recording        = False
        self.thermocouples    = []
        self.flow_sensors     = []
        self.i2c              = None
        self.mux              = None
        self.test_start_time  = None

    def add_thermocouple(self, mux_channel, thermo_address):
        if not 0 <= mux_channel <= 7:
            raise ValueError("Multiplexer channel must be 0–7")
        self.thermocouple_config.append((mux_channel, thermo_address))

    # [WFS] channel_type parameter added: 'flow' or 'temp'
    def add_flow_sensor(self, mux_channel, ads_address, ads_channel, channel_type='flow'):
        if not 0 <= mux_channel <= 7:
            raise ValueError("Multiplexer channel must be 0–7")
        if not 0 <= ads_channel <= 3:
            raise ValueError("ADS channel must be 0–3")
        if channel_type not in ('flow', 'temp'):
            raise ValueError("channel_type must be 'flow' or 'temp'")
        self.flow_sensor_config.append((mux_channel, ads_address, ads_channel, channel_type))

    def _format_timestamp(self, seconds):
        return str(timedelta(seconds=seconds))[:-3]

    def _init_tc_with_timeout(self, mux_ch, addr, timeout=2.0):
        """
        Attempt to instantiate MCP9600 in a thread with a hard timeout.
        Needed on RPi 5 where adafruit_tca9548a.try_lock() can block
        indefinitely against a non-responsive device instead of raising.
        Returns (MCP9600_instance_or_None, exception_or_None).
        """
        result = [None]
        error  = [None]

        def _try():
            try:
                result[0] = MCP9600(self.mux[mux_ch], address=addr)
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_try, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            return None, TimeoutError(
                f"MCP9600 timed out after {timeout}s "
                f"(Mux CH{mux_ch}, 0x{addr:02X}) — device not responding"
            )
        if error[0] is not None:
            return None, error[0]
        return result[0], None

    # [WFS] Corrected conversion: 4–20 mA via 0–3 V I-to-V, Q in L/min
    def _convert_voltage_to_flow(self, voltage):
        if voltage is None:
            return None
        i_minus_4 = (voltage / 3.0) * 16.0
        return 0.938 * i_minus_4

    # [WFS] New conversion method for WFS temperature channels
    def _convert_voltage_to_wfs_temp(self, voltage):
        if voltage is None:
            return None
        i_minus_4 = (voltage / 3.0) * 16.0
        return 9.375 * i_minus_4 - 25.0

    def initialize_hardware_static(self):
        try:
            self.i2c = busio.I2C(self.I2C_SCL_PIN, self.I2C_SDA_PIN)
            self.mux = adafruit_tca9548a.TCA9548A(self.i2c, address=self.MUX_ADDRESS)
            print(f"Multiplexer initialised at 0x{self.MUX_ADDRESS:02X}")

            n_tc   = len(self.thermocouple_config)
            # [WFS] Count flow and temp channels separately for storage sizing
            n_flow = sum(1 for c in self.flow_sensor_config if c[3] == 'flow')
            n_wfst = sum(1 for c in self.flow_sensor_config if c[3] == 'temp')

            self.rec_temp     = [[] for _ in range(n_tc)]
            self.rec_flow     = [[] for _ in range(n_flow)]
            self.rec_wfs_temp = [[] for _ in range(n_wfst)]
            self.rec_time     = [[] for _ in range(max(n_tc, len(self.flow_sensor_config)))]

            # Init thermocouples — use timeout wrapper (RPi 5 I²C hang protection)
            self.thermocouples = []
            for idx, (mux_ch, addr) in enumerate(self.thermocouple_config):
                tc, err = self._init_tc_with_timeout(mux_ch, addr)
                if tc is not None:
                    self.thermocouples.append((mux_ch, tc))
                    print(f"TC {idx}: Mux CH{mux_ch}, 0x{addr:02X} — OK")
                else:
                    self.thermocouples.append((mux_ch, None))
                    print(f"TC {idx}: Mux CH{mux_ch}, 0x{addr:02X} — Failed: {err}")

            # [WFS] Init flow sensors — open ADS1115 once per unique (mux_ch, ads_addr)
            self.flow_sensors = []
            ads_cache = {}
            for idx, (mux_ch, addr, ads_ch, ch_type) in enumerate(self.flow_sensor_config):
                key = (mux_ch, addr)
                if key not in ads_cache:
                    try:
                        ads = ADS.ADS1115(self.mux[mux_ch], address=addr)
                        # [WFS] Explicit gain: GAIN_ONE covers 0–3 V without clipping
                        ads.gain = 1
                        ads_cache[key] = ads
                        print(f"ADS1115 Mux CH{mux_ch}, 0x{addr:02X} — OK")
                    except Exception as e:
                        ads_cache[key] = None
                        print(f"ADS1115 Mux CH{mux_ch}, 0x{addr:02X} — Failed: {e}")
                self.flow_sensors.append((mux_ch, ads_cache[key], ads_ch, ch_type))

            return True

        except Exception as e:
            print(f"Static initialisation failed: {e}")
            return False

    def initialize_hardware_DYNAMIC(self):
        try:
            self.i2c = busio.I2C(self.I2C_SCL_PIN, self.I2C_SDA_PIN)
            self.mux = adafruit_tca9548a.TCA9548A(self.i2c, address=self.MUX_ADDRESS)
            print(f"Multiplexer initialised at 0x{self.MUX_ADDRESS:02X}")

            n_tc = len(self.thermocouple_config)
            self.rec_temp = [[] for _ in range(n_tc)]
            self.rec_time = [[] for _ in range(n_tc)]

            self.thermocouples = []
            for idx, (mux_ch, addr) in enumerate(self.thermocouple_config):
                tc, err = self._init_tc_with_timeout(mux_ch, addr)
                if tc is not None:
                    self.thermocouples.append((mux_ch, tc))
                    print(f"TC {idx}: Mux CH{mux_ch}, 0x{addr:02X} — OK")
                else:
                    self.thermocouples.append((mux_ch, None))
                    print(f"TC {idx}: Mux CH{mux_ch}, 0x{addr:02X} — Failed: {err}")

            return any(tc[1] is not None for tc in self.thermocouples)

        except Exception as e:
            print(f"Dynamic initialisation failed: {e}")
            return False

    # [GENERATOR] generator parameter added so the loop can fire/stop it
    def collect_data_static(self, generator, duration_sec=0, max_samples=0):
        if not any(tc[1] for tc in self.thermocouples) and not any(fs[1] for fs in self.flow_sensors):
            print("No sensors initialised!")
            return

        self.recording      = True
        self.test_start_time = time.monotonic()
        sample_count        = 0
        # [GENERATOR] Track whether generator has been started yet
        generator_started   = False

        # [WFS] Per-run index counters for flow and wfs_temp sub-lists
        flow_idx = [i for i, c in enumerate(self.flow_sensor_config) if c[3] == 'flow']
        temp_idx = [i for i, c in enumerate(self.flow_sensor_config) if c[3] == 'temp']

        try:
            while self.recording:
                elapsed = time.monotonic() - self.test_start_time

                if (duration_sec > 0 and elapsed >= duration_sec) or \
                   (max_samples > 0 and sample_count >= max_samples):
                    # [GENERATOR] Stop generator when test duration expires
                    generator.off()
                    print(f"[{self._format_timestamp(elapsed)}] Generator OFF — test duration reached")
                    break

                # [GENERATOR] Fire generator after delay, once only
                if not generator_started and elapsed >= GENERATOR_START_DELAY_S:
                    generator.on()
                    generator_started = True
                    print(f"[{self._format_timestamp(elapsed)}] Generator ON")

                # Read thermocouples
                temp_readings = []
                for idx, (mux_ch, tc) in enumerate(self.thermocouples):
                    if tc is not None:
                        try:
                            temp = tc.temperature
                            self.rec_temp[idx].append(temp)
                        except Exception as e:
                            temp = None
                            self.rec_temp[idx].append(None)
                            print(f"[{self._format_timestamp(elapsed)}] TC CH{mux_ch} error: {e}")
                    else:
                        temp = None
                        self.rec_temp[idx].append(None)
                    temp_readings.append((mux_ch, temp))

                # [WFS] Read flow sensors with correct conversion per channel_type
                flow_readings = []
                fi = 0  # index into rec_flow sub-list
                ti = 0  # index into rec_wfs_temp sub-list
                for idx, (mux_ch, ads, ads_ch, ch_type) in enumerate(self.flow_sensors):
                    value = None
                    if ads is not None:
                        try:
                            chan    = AnalogIn(ads, getattr(ADS, f'P{ads_ch}'))
                            voltage = chan.voltage
                            if ch_type == 'flow':
                                value = self._convert_voltage_to_flow(voltage)
                            else:
                                value = self._convert_voltage_to_wfs_temp(voltage)
                        except Exception as e:
                            print(f"[{self._format_timestamp(elapsed)}] ADS CH{ads_ch} error: {e}")

                    if ch_type == 'flow':
                        self.rec_flow[fi].append(value)
                        fi += 1
                    else:
                        self.rec_wfs_temp[ti].append(value)
                        ti += 1
                    flow_readings.append((mux_ch, ads_ch, ch_type, value))

                # Timestamps
                for i in range(len(self.rec_time)):
                    self.rec_time[i].append(elapsed)

                # Console print
                print(f"[{self._format_timestamp(elapsed)}] ", end='')
                for ch, temp in temp_readings:
                    print(f"T{ch}={temp}°C " if temp is not None else f"T{ch}=X ", end='')
                for _, ach, ctype, val in flow_readings:
                    label = f"Q{ach}" if ctype == 'flow' else f"Tw{ach}"
                    unit  = "L/min" if ctype == 'flow' else "°C"
                    print(f"{label}={val}{unit} " if val is not None else f"{label}=X ", end='')
                print()

                sample_count += 1
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nCollection interrupted")
        finally:
            # [GENERATOR] Safety: ensure generator is always off on exit
            generator.off()
            self.recording = False
            elapsed = time.monotonic() - self.test_start_time
            print(f"Collection complete. Duration: {elapsed:.1f}s, Samples: {sample_count}")

    def collect_data_DYNAMIC(self, duration_sec=0, max_samples=0):
        if not any(tc[1] for tc in self.thermocouples):
            print("No thermocouples initialised!")
            return

        self.recording       = True
        self.test_start_time = time.monotonic()
        sample_count         = 0

        try:
            while self.recording:
                elapsed = time.monotonic() - self.test_start_time

                if (duration_sec > 0 and elapsed >= duration_sec) or \
                   (max_samples > 0 and sample_count >= max_samples):
                    break

                temp_readings = []
                for idx, (mux_ch, tc) in enumerate(self.thermocouples):
                    if tc is not None:
                        try:
                            temp = tc.temperature
                            self.rec_temp[idx].append(temp)
                            self.rec_time[idx].append(elapsed)
                        except Exception as e:
                            self.rec_temp[idx].append(None)
                            self.rec_time[idx].append(elapsed)
                            print(f"[{self._format_timestamp(elapsed)}] TC CH{mux_ch} error: {e}")
                    else:
                        self.rec_temp[idx].append(None)
                        self.rec_time[idx].append(elapsed)
                    temp_readings.append((mux_ch, None if tc is None else self.rec_temp[idx][-1]))

                print(f"[{self._format_timestamp(elapsed)}] ", end='')
                for ch, temp in temp_readings:
                    print(f"T{ch}={temp}°C " if temp is not None else f"T{ch}=X ", end='')
                print()

                sample_count += 1
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nCollection interrupted")
        finally:
            self.recording = False
            elapsed = time.monotonic() - self.test_start_time
            print(f"Collection complete. Duration: {elapsed:.1f}s, Samples: {sample_count}")

    # [WFS] Returns rec_wfs_temp as fourth element
    def get_data_static(self):
        return (self.rec_temp, self.rec_flow, self.rec_wfs_temp, self.rec_time)

    def get_data_DYNAMIC(self):
        return (self.rec_temp, self.rec_time)

    def clear_data(self):
        n_tc   = len(self.thermocouple_config)
        n_flow = sum(1 for c in self.flow_sensor_config if c[3] == 'flow')
        n_wfst = sum(1 for c in self.flow_sensor_config if c[3] == 'temp')
        self.rec_temp     = [[] for _ in range(n_tc)]
        self.rec_flow     = [[] for _ in range(n_flow)]
        # [WFS] Clear wfs_temp storage
        self.rec_wfs_temp = [[] for _ in range(n_wfst)]
        self.rec_time     = [[] for _ in range(max(n_tc, len(self.flow_sensor_config)))]
        self.test_start_time = None


# =============================================================================
#  [GENERATOR]  GENERATOR CONTROLLER
#  New class. Wraps relay output on GENERATOR_PIN.
#  on()  → GPIO HIGH → relay closes → generator starts
#  off() → GPIO LOW  → relay opens  → generator stops
# =============================================================================
class GeneratorController:
    def __init__(self, pin=GENERATOR_PIN):
        self._relay = DigitalOutputDevice(pin, active_high=True, initial_value=False)
        print(f"GeneratorController initialised on GPIO{pin}")

    def on(self):
        self._relay.on()
        print("Generator relay CLOSED — generator running")

    def off(self):
        self._relay.off()
        print("Generator relay OPEN — generator stopped")

    def cleanup(self):
        self.off()
        try:
            self._relay.close()
        except Exception:
            pass


# =============================================================================
#  [PICO]  PICOSCOPE 2204A CONTROLLER
#  New class. Encapsulates all PCPTR logic.
#  Bandpass 250–500 kHz replaces original high-pass.
#  ps2000_ready() polling with 1 s timeout replaces bare time.sleep().
# =============================================================================
class PicoscopeController:
    def __init__(self):
        # Acquisition constants
        self.timebase   = TIMEBASE
        self.oversample = OVERSAMPLE
        self.n_samples  = N_SAMPLES
        self.dt         = DT
        self.mpc        = MPC
        self.bs         = BS
        self.cp         = CP
        self.interval_s = PICO_INTERVAL_S

        # Runtime state
        self.chandle          = None
        self.time_interval_ns = None
        self.captured_blocks  = []
        self.processed_blocks = None
        self.rms_V = self.rms_B = self.rms_H = self.rms_I = None
        self.status           = {}
        # [PICO] Threading event used to stop capture_loop cleanly
        self.stop_event       = threading.Event()

    def open(self):
        """Open PicoScope unit, configure channel A, set trigger off, get timebase."""
        chandle = ps.ps2000_open_unit()
        if chandle <= 0:
            raise RuntimeError(f"ps2000_open_unit failed, handle={chandle}")
        self.chandle = chandle
        print(f"PicoScope 2204A opened, handle={chandle}")

        # Configure Channel A — AC coupling, ±2 V
        self.status["ch_a"] = ps.ps2000_set_channel(
            chandle, 0, 1, 0,
            ps.PS2000_VOLTAGE_RANGE['PS2000_2V']
        )
        assert_pico2000_ok(self.status["ch_a"])

        # Disable Channel B
        self.status["ch_b"] = ps.ps2000_set_channel(chandle, 1, 0, 0, 7)
        assert_pico2000_ok(self.status["ch_b"])

        # Trigger off
        self.status["trigger"] = ps.ps2000_set_trigger(chandle, 0, 0, 0, 0, 0)
        assert_pico2000_ok(self.status["trigger"])

        # Get timebase
        ti_ns      = ctypes.c_int32()
        ti_units   = ctypes.c_int32()
        max_samp   = ctypes.c_int32()
        self.status["timebase"] = ps.ps2000_get_timebase(
            chandle, self.timebase, self.n_samples,
            ctypes.byref(ti_ns), ctypes.byref(ti_units),
            self.oversample, ctypes.byref(max_samp)
        )
        assert_pico2000_ok(self.status["timebase"])
        self.time_interval_ns = ti_ns.value
        print(f"PicoScope timebase OK — {ti_ns.value} ns/sample, max {max_samp.value} samples")

    def capture_block(self):
        """Capture one block. Polls ps2000_ready() with 1 s timeout."""
        buffer_a          = (ctypes.c_int16 * self.n_samples)()
        overflow          = ctypes.c_int16()
        time_indisposed_ms = ctypes.c_int32()

        self.status["runBlock"] = ps.ps2000_run_block(
            self.chandle, self.n_samples, self.timebase,
            self.oversample, ctypes.byref(time_indisposed_ms)
        )
        assert_pico2000_ok(self.status["runBlock"])

        # [PICO] Reinstated ps2000_ready() polling with 1 s timeout
        deadline = time.monotonic() + 1.0
        while True:
            ready = ps.ps2000_ready(self.chandle)
            if ready > 0:
                break
            if ready < 0:
                raise RuntimeError("ps2000_ready: device not attached")
            if time.monotonic() > deadline:
                raise RuntimeError("ps2000_ready: timed out after 1 s")
            time.sleep(0.001)

        sample_count = ps.ps2000_get_values(
            self.chandle,
            ctypes.byref(buffer_a),
            None, None, None,
            ctypes.byref(overflow),
            self.n_samples
        )
        if sample_count <= 0:
            raise RuntimeError("ps2000_get_values returned no samples")

        return np.array(buffer_a[:sample_count], dtype=np.int16)

    def capture_loop(self, stop_event):
        """
        [PICO] Thread target for RUNNING state.
        Captures one block every PICO_INTERVAL_S until stop_event is set
        or TEST_DURATION_S has elapsed.
        Raw blocks appended to self.captured_blocks — no processing here.
        """
        print("Pico capture loop started")
        start     = time.monotonic()
        next_read = start + self.interval_s

        try:
            while not stop_event.is_set():
                now = time.monotonic()
                if now - start >= TEST_DURATION_S:
                    break
                if now >= next_read:
                    try:
                        block = self.capture_block()
                        self.captured_blocks.append(block)
                        print(f"Pico block {len(self.captured_blocks)} captured "
                              f"at t={now - start:.2f}s, shape={block.shape}")
                    except Exception as e:
                        print(f"Pico capture error: {e}")
                    next_read += self.interval_s
                    while next_read <= time.monotonic():
                        next_read += self.interval_s
                else:
                    time.sleep(0.005)
        finally:
            print(f"Pico capture loop ended — {len(self.captured_blocks)} blocks captured")

    def process_all(self):
        """
        [PICO] Process all captured raw blocks.
        Called in COMPLETED state after capture_loop has finished.
        Stores results in self.processed_blocks, self.rms_*.
        """
        if not self.captured_blocks:
            print("Pico: no blocks to process")
            return

        result = self._process_captured_blocks(self.captured_blocks)
        self.processed_blocks, self.rms_V, self.rms_B, self.rms_H, self.rms_I = result
        print(f"Pico processing complete — shape: {self.processed_blocks.shape}")

    def _process_captured_blocks(self, block_list):
        """
        [PICO] Core processing pipeline.
        Shape: (n_blocks, 5, N_SAMPLES)
          Row 0: raw ADC
          Row 1: voltage (V), bandpass filtered 250–500 kHz
          Row 2: B field (trapezoidal integration)
          Row 3: H field (B / MPC)
          Row 4: Current (H * BS)
        Returns processed array + rms_V, rms_B, rms_H, rms_I arrays.
        """
        ns = len(block_list)
        if ns == 0:
            empty = np.empty((0, 5, self.n_samples), dtype=np.float64)
            return empty, np.array([]), np.array([]), np.array([]), np.array([])

        processed = np.zeros((ns, 5, self.n_samples), dtype=np.float64)
        rms_V = np.zeros(ns)
        rms_B = np.zeros(ns)
        rms_H = np.zeros(ns)
        rms_I = np.zeros(ns)

        # [PICO] Build bandpass filter once — 250–500 kHz, 2nd-order Butterworth
        fs      = 1.0 / self.dt
        lowcut  = 250_000
        highcut = 500_000
        b_filt, a_filt = butter(
            2,
            [lowcut / (fs / 2), highcut / (fs / 2)],
            btype='bandpass'
        )

        for i in range(ns):
            raw = np.asarray(block_list[i], dtype=np.float64)
            if raw.shape[0] != self.n_samples:
                raise ValueError(
                    f"Block {i}: length {raw.shape[0]}, expected {self.n_samples}"
                )

            processed[i, 0, :] = raw

            # Voltage: ADC → V, then bandpass filter
            processed[i, 1, :] = raw * 20.0 / 32767.0
            processed[i, 1, :] = filtfilt(b_filt, a_filt, processed[i, 1, :])

            # B field: trapezoidal integration of voltage
            processed[i, 2, 0]  = 0.0
            processed[i, 2, 1:] = np.cumsum(
                0.5 * self.cp * (
                    (processed[i, 1, :-1] + processed[i, 1, 1:]) * self.dt
                )
            )

            # H field: B / MPC
            processed[i, 3, :] = processed[i, 2, :] / self.mpc

            # Current: H * BS
            processed[i, 4, :] = processed[i, 3, :] * self.bs

            rms_V[i] = np.sqrt(np.mean(processed[i, 1, :] ** 2))
            rms_B[i] = np.sqrt(np.mean(processed[i, 2, :] ** 2))
            rms_H[i] = np.sqrt(np.mean(processed[i, 3, :] ** 2))
            rms_I[i] = np.sqrt(np.mean(processed[i, 4, :] ** 2))

        return processed, rms_V, rms_B, rms_H, rms_I

    def get_block_for_csv(self, n=PICO_BLOCK_PLOT):
        """
        [PICO] Return the nth processed block (1-indexed) for CSV section 3.
        Falls back to last block if fewer than n were captured.
        """
        if self.processed_blocks is None or len(self.processed_blocks) == 0:
            return None, None
        idx = min(n - 1, len(self.processed_blocks) - 1)
        actual = idx + 1
        if actual != n:
            print(f"Pico: only {actual} blocks available, using block {actual} for CSV")
        return self.processed_blocks[idx], actual

    def close(self):
        """[PICO] Stop and close device unconditionally."""
        if self.chandle is not None:
            try:
                ps.ps2000_stop(self.chandle)
                ps.ps2000_close_unit(self.chandle)
                print("PicoScope closed")
            except Exception as e:
                print(f"PicoScope close error: {e}")
            finally:
                self.chandle = None


# =============================================================================
#  LED THREAD
# =============================================================================
def ledscpt():
    while True:
        state1 = switch1.is_pressed
        state2 = switch2.is_pressed

        if KVEBench.current_state == State.IDLE:
            LED1.on(); LED2.on(); LED3.on(); LED4.on()
            time.sleep(1)

        if KVEBench.current_state in (State.PRECHECK_DYNAMIC, State.PRECHECK_STATIC):
            LED1.off(); LED2.off(); LED3.off(); LED4.off()

        while KVEBench.current_state in (State.PRECHECK_DYNAMIC, State.PRECHECK_STATIC):
            LED1.off(); time.sleep(0.4)
            LED1.on();  time.sleep(0.4)

        if KVEBench.current_state == State.RUNNING:
            while KVEBench.current_state == State.RUNNING and state1 and not state2:
                LED2.off(); time.sleep(0.2)
                LED2.on();  time.sleep(0.2)
            while KVEBench.current_state == State.RUNNING and state2 and not state1:
                LED3.off(); time.sleep(0.2)
                LED3.on();  time.sleep(0.2)

        if KVEBench.current_state == State.COMPLETED:
            while KVEBench.current_state == State.COMPLETED:
                if state1 and not state2:
                    LED2.on()
                elif state2 and not state1:
                    LED3.on()

        if KVEBench.current_state == State.ERROR:
            while KVEBench.current_state == State.ERROR:
                LED4.on();  time.sleep(0.2)
                LED4.off(); time.sleep(0.2)


# =============================================================================
#  TEMPERATURE CONTROLLER
# =============================================================================
class TemperatureController:
    def __init__(self, thermocouple_logger, fan_pin, heater_pin, heater_fan_pin,
                 active_high=True):
        self.logger         = thermocouple_logger
        self.fan_out        = DigitalOutputDevice(fan_pin)
        self.heater_out     = DigitalOutputDevice(heater_pin)
        self.heater_fan_out = DigitalOutputDevice(heater_fan_pin)
        self.desired_min    = 10.0
        self.desired_max    = 11.0
        self.hysteresis     = 1.0

    def set_temperature_range(self, min_temp, max_temp, hysteresis=1.0):
        self.desired_min = min_temp
        self.desired_max = max_temp
        self.hysteresis  = hysteresis

    def get_average_temperature(self):
        temps = []
        for _, tc in self.logger.thermocouples:
            if tc is not None:
                try:
                    t = tc.temperature
                    if t is not None:
                        temps.append(t)
                except OSError:
                    continue
        return sum(temps) / len(temps) if temps else None

    def control_temperature(self):
        avg = self.get_average_temperature()
        if avg is None:
            print("Error: no thermocouple readable")
            return False

        if self.desired_min <= avg <= self.desired_max:
            self._turn_off_fans(); self._turn_off_heater()
            return self._check_stabilization()

        if avg > self.desired_max + self.hysteresis:
            self._turn_off_heater(); self._turn_on_fans()
        elif avg < self.desired_min - self.hysteresis:
            self._turn_off_fans(); self._turn_on_heater()
        return False

    def _turn_on_fans(self):
        self.fan_out.on(); self.heater_fan_out.on()

    def _turn_off_fans(self):
        self.fan_out.off(); self.heater_fan_out.off()

    def _turn_on_heater(self):
        self.heater_out.on(); self.heater_fan_out.on()

    def _turn_off_heater(self):
        self.heater_out.off(); self.heater_fan_out.off()

    def _check_stabilization(self, check_duration=5, check_interval=1):
        start = time.time()
        while time.time() - start < check_duration:
            avg = self.get_average_temperature()
            if avg is None or not (self.desired_min <= avg <= self.desired_max):
                return False
            time.sleep(check_interval)
            print(f"Stabilisation: {avg:.2f}°C in [{self.desired_min}, {self.desired_max}]")
        return True

    def cleanup(self):
        self._turn_off_fans(); self._turn_off_heater()
        for dev in (self.fan_out, self.heater_out, self.heater_fan_out):
            try: dev.close()
            except Exception: pass


# =============================================================================
#  [CSV]  SD CARD WRITER — rewritten for 3-section combined CSV
# =============================================================================
class SDWriter:
    def __init__(self, base_path="/media/kve/KVELOGGER/LoggedTests"):
        self.base_path = base_path

    def _generate_filename(self, prefix="test"):
        return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S.csv")

    def write_to_csv(self, timestamps, temperatures, flows,
                     wfs_temps=None, pico_rms=None, pico_block=None,
                     pico_block_number=None, pico_time_interval_ns=None):
        """
        [CSV] Write single combined CSV with 3 sections.

        Section 1 — TC + flow + WFS temp (one row per 0.5 s sample)
        Section 2 — Pico RMS summary (one row per block, all blocks)
        Section 3 — Block N full signal (8000 rows: V, B, H, I)

        Parameters
        ----------
        timestamps           : list of lists (per-sensor timestamps)
        temperatures         : list of lists (per-TC readings)
        flows                : list of lists (flow L/min per flow channel)
        wfs_temps            : list of lists (WFS temperature °C per temp channel)
        pico_rms             : tuple (rms_V, rms_B, rms_H, rms_I) — one value per block
        pico_block           : np.ndarray shape (5, N_SAMPLES) — processed block N
        pico_block_number    : int — which block number is stored (for header label)
        pico_time_interval_ns: int — ns per sample from ps2000_get_timebase
        """
        os.makedirs(self.base_path, exist_ok=True)
        filepath = os.path.join(self.base_path, self._generate_filename())

        n_tc    = len(temperatures)
        n_flow  = len(flows) if flows else 0
        n_wfst  = len(wfs_temps) if wfs_temps else 0

        try:
            with open(filepath, 'w', newline='') as f:
                w = csv.writer(f)

                # ── Section 1: TC + flow + WFS temp ──────────────────────────
                w.writerow(["# SECTION 1 — THERMOCOUPLE AND FLOW DATA"])
                header = ["timestamp"]
                header += [f"thermocouple_{i+1}" for i in range(n_tc)]
                header += [f"flow_{i+1}_lpm"     for i in range(n_flow)]
                header += [f"wfs_temp_{i+1}_C"   for i in range(n_wfst)]
                w.writerow(header)

                n_samples = len(timestamps[0]) if timestamps else 0
                for i in range(n_samples):
                    ts   = timestamps[0][i] if timestamps else 0
                    row  = [f"{ts:.3f}"]
                    row += [f"{temperatures[t][i]:.1f}"
                            if i < len(temperatures[t]) and temperatures[t][i] is not None
                            else "" for t in range(n_tc)]
                    row += [f"{flows[f][i]:.3f}"
                            if flows and i < len(flows[f]) and flows[f][i] is not None
                            else "" for f in range(n_flow)]
                    row += [f"{wfs_temps[t][i]:.2f}"
                            if wfs_temps and i < len(wfs_temps[t]) and wfs_temps[t][i] is not None
                            else "" for t in range(n_wfst)]
                    w.writerow(row)

                w.writerow([])

                # ── [CSV] Section 2: Pico RMS summary ────────────────────────
                w.writerow(["# SECTION 2 — PICOSCOPE RMS SUMMARY (ALL BLOCKS)"])
                if pico_rms is not None:
                    rms_V, rms_B, rms_H, rms_I = pico_rms
                    w.writerow(["block_index", "rms_V", "rms_B", "rms_H", "rms_I"])
                    for idx in range(len(rms_V)):
                        w.writerow([
                            idx + 1,
                            f"{rms_V[idx]:.6e}",
                            f"{rms_B[idx]:.6e}",
                            f"{rms_H[idx]:.6e}",
                            f"{rms_I[idx]:.6e}",
                        ])
                else:
                    w.writerow(["# No Pico data available"])

                w.writerow([])

                # ── [CSV] Section 3: Block N full signal ──────────────────────
                bn = pico_block_number or PICO_BLOCK_PLOT
                w.writerow([f"# SECTION 3 — PICOSCOPE BLOCK {bn} FULL SIGNAL (8000 SAMPLES)"])
                if pico_block is not None:
                    w.writerow(["sample_index", "time_s",
                                "voltage_V", "b_field_T", "h_field_Am", "current_A"])
                    dt_s = (pico_time_interval_ns or 20) * 1e-9
                    for s in range(pico_block.shape[1]):
                        w.writerow([
                            s,
                            f"{s * dt_s:.12e}",
                            f"{pico_block[1, s]:.12e}",
                            f"{pico_block[2, s]:.12e}",
                            f"{pico_block[3, s]:.12e}",
                            f"{pico_block[4, s]:.12e}",
                        ])
                else:
                    w.writerow(["# No Pico block data available"])

            print(f"CSV written: {filepath}")
            return True

        except Exception as e:
            print(f"CSV write failed: {e}")
            return False


# =============================================================================
#  STATE ENUM
# =============================================================================
class State:
    IDLE             = "IDLE"
    PRECHECK_DYNAMIC = "PRECHECK_DYNAMIC"
    PRECHECK_STATIC  = "PRECHECK_STATIC"
    RUNNING          = "RUNNING"
    COMPLETED        = "COMPLETED"
    ERROR            = "ERROR"


# =============================================================================
#  STATE MACHINE
# =============================================================================
class StateMachine:
    def __init__(self):
        self.current_state  = None
        self.previous_state = None
        self.error_message  = None

        self.collected_temperatures = None
        self.collected_timestamps   = None
        self.collected_flow         = None
        # [WFS] New collection attribute for WFS temperature data
        self.collected_wfs_temp     = None

        self.valid_transitions = {
            State.IDLE:             [State.PRECHECK_DYNAMIC, State.PRECHECK_STATIC, State.ERROR],
            State.PRECHECK_STATIC:  [State.RUNNING, State.IDLE, State.ERROR],
            State.PRECHECK_DYNAMIC: [State.RUNNING, State.IDLE, State.ERROR],
            State.RUNNING:          [State.COMPLETED, State.ERROR, State.IDLE],
            State.COMPLETED:        [State.IDLE, State.ERROR],
            State.ERROR:            [State.IDLE],
        }
        self._setup_state_actions()

    def _setup_state_actions(self):
        self.state_entry_actions = {
            State.IDLE:             self._on_enter_idle,
            State.PRECHECK_DYNAMIC: self._on_enter_precheck_DYNAMIC,
            State.PRECHECK_STATIC:  self._on_enter_precheck_static,
            State.RUNNING:          self._on_enter_running,
            State.COMPLETED:        self._on_enter_completed,
            State.ERROR:            self._on_enter_error,
        }
        self.state_exit_actions = {
            State.IDLE:             self._on_exit_idle,
            State.PRECHECK_DYNAMIC: self._on_exit_precheck_DYNAMIC,
            State.PRECHECK_STATIC:  self._on_exit_precheck_static,
            State.RUNNING:          self._on_exit_running,
            State.COMPLETED:        self._on_exit_completed,
            State.ERROR:            self._on_exit_error,
        }

    # ── Entry / exit stubs (unchanged from v1_1 unless noted) ────────────────
    def _on_enter_idle(self):            pass
    def _on_exit_idle(self):             pass
    def _on_enter_precheck_DYNAMIC(self): pass
    def _on_exit_precheck_DYNAMIC(self):  pass
    def _on_enter_precheck_static(self):  pass
    def _on_exit_precheck_static(self):   pass
    def _on_enter_running(self):         pass
    def _on_enter_completed(self):       pass
    def _on_exit_completed(self):        pass
    def _on_enter_error(self):           pass
    def _on_exit_error(self):            pass

    def _on_exit_running(self):
        """
        [GENERATOR] [PICO] Unconditional cleanup on RUNNING exit.
        Fires regardless of whether exit is normal, interrupt, or error.
        """
        try:
            self._generator.off()
            print("_on_exit_running: generator OFF")
        except Exception:
            pass
        try:
            self._pico.stop_event.set()
            self._pico.close()
            print("_on_exit_running: Pico stopped and closed")
        except Exception:
            pass

    def transition_to(self, new_state, error_message=None):
        if new_state not in self.valid_transitions[self.current_state]:
            print(f"Invalid transition: {self.current_state} → {new_state}")
            return False

        self.state_exit_actions[self.current_state]()
        self.previous_state = self.current_state
        self.current_state  = new_state

        if new_state == State.ERROR:
            self.error_message = error_message or "Unknown error"

        self.state_entry_actions[self.current_state]()
        print(f"State: {self.previous_state} → {self.current_state}")
        return True

    def get_current_state(self):  return self.current_state
    def get_error_message(self):
        return self.error_message if self.current_state == State.ERROR else None

    # =========================================================================
    #  MAIN RUN LOOP
    # =========================================================================
    def run(self):
        self.error_message = None

        # Button LED thread
        def button_led():
            while True:
                if KVEBench.current_state != State.IDLE:
                    bLED.on();  time.sleep(0.5)
                    bLED.off(); time.sleep(0.5)
                else:
                    bLED.on()

        ledthr  = threading.Thread(target=ledscpt,    daemon=True)
        bledthr = threading.Thread(target=button_led, daemon=True)
        bledthr.start()
        ledthr.start()

        # Thermocouple logger
        logger = MultiplexedThermocoupleLogger()
        logger.add_thermocouple(0, 0x67)
        logger.add_thermocouple(1, 0x67)
        logger.add_thermocouple(2, 0x67)
        logger.add_thermocouple(3, 0x65)
        logger.add_thermocouple(4, 0x66)
        logger.add_thermocouple(5, 0x66)
        logger.add_thermocouple(6, 0x67)
        logger.add_thermocouple(7, 0x67)

        # [WFS] Flow sensor registration — one ADS1115 on mux CH7, address 0x48
        # A0=flow sensor 1, A1=WFS temp sensor 1, A2=flow sensor 2, A3=WFS temp sensor 2
        logger.add_flow_sensor(7, 0x48, 0, 'flow')
        logger.add_flow_sensor(7, 0x48, 1, 'temp')
        logger.add_flow_sensor(7, 0x48, 2, 'flow')
        logger.add_flow_sensor(7, 0x48, 3, 'temp')

        # [GENERATOR] Instantiate generator controller
        generator = GeneratorController(GENERATOR_PIN)
        # Store reference on self so _on_exit_running can reach it
        self._generator = generator

        # [PICO] Instantiate PicoScope controller
        pico = PicoscopeController()
        self._pico = pico

        # SD writer and temperature controller
        sd_writer = SDWriter()
        logtemp   = TemperatureController(logger, fan_pin=17, heater_pin=27, heater_fan_pin=22)

        self.current_state = State.IDLE

        # Button interrupt handler
        def handle_button_press():
            nonlocal logger
            if self.current_state in (
                State.PRECHECK_DYNAMIC, State.PRECHECK_STATIC,
                State.RUNNING, State.COMPLETED
            ):
                print("\nButton pressed — emergency stop")
                logger.recording = False
                # [GENERATOR] Stop generator on emergency interrupt
                generator.off()
                # [PICO] Stop Pico capture thread on emergency interrupt
                pico.stop_event.set()
                self.transition_to(State.IDLE)

        button.when_pressed = handle_button_press

        # ── Main loop ─────────────────────────────────────────────────────────
        while True:

            # ── IDLE ──────────────────────────────────────────────────────────
            if self.current_state == State.IDLE:
                time.sleep(2)
                print("IDLE — waiting for mode selection")

                logger.initialize_hardware_DYNAMIC()

                while not logtemp.control_temperature():
                    print("Conditioning temperature...")
                    time.sleep(1)

                while self.current_state == State.IDLE:
                    state1 = switch1.is_pressed
                    state2 = switch2.is_pressed

                    if state1 and not state2:
                        print("Static mode selected", end='\r')
                        if button.is_pressed:
                            time.sleep(0.3)
                            self.transition_to(State.PRECHECK_STATIC)
                    elif state2 and not state1:
                        print("Dynamic mode selected", end='\r')
                        if button.is_pressed:
                            time.sleep(0.3)
                            self.transition_to(State.PRECHECK_DYNAMIC)
                    else:
                        print("No mode selected", end='\r')

            # ── PRECHECK DYNAMIC ──────────────────────────────────────────────
            elif self.current_state == State.PRECHECK_DYNAMIC:
                print("Starting dynamic precheck...")
                if not logger.initialize_hardware_DYNAMIC():
                    self.transition_to(State.ERROR, "Hardware initialisation failed")
                    continue
                time.sleep(2)
                if self.current_state == State.PRECHECK_DYNAMIC:
                    self.transition_to(State.RUNNING)

            # ── PRECHECK STATIC ───────────────────────────────────────────────
            elif self.current_state == State.PRECHECK_STATIC:
                print("Starting static precheck...")
                if not logger.initialize_hardware_static():
                    self.transition_to(State.ERROR, "Hardware initialisation failed")
                    continue

                # [PICO] Open PicoScope during static precheck
                pico.stop_event.clear()
                pico.captured_blocks = []
                try:
                    pico.open()
                except Exception as e:
                    self.transition_to(State.ERROR, f"PicoScope init failed: {e}")
                    continue

                time.sleep(2)
                if self.current_state == State.PRECHECK_STATIC:
                    self.transition_to(State.RUNNING)

            # ── RUNNING ───────────────────────────────────────────────────────
            elif self.current_state == State.RUNNING:
                print("Starting test run...")
                state1 = switch1.is_pressed
                state2 = switch2.is_pressed

                if state1 and not state2:
                    # [PICO] Spawn Pico capture thread alongside TC/flow collection
                    pico_thread = threading.Thread(
                        target=pico.capture_loop,
                        args=(pico.stop_event,),
                        daemon=True
                    )
                    pico_thread.start()

                    # [GENERATOR] Pass generator into collect_data_static
                    logger.collect_data_static(
                        generator=generator,
                        duration_sec=TEST_DURATION_S,
                        max_samples=0
                    )

                    # Wait for Pico thread to finish
                    pico.stop_event.set()
                    pico_thread.join(timeout=5)

                elif state2 and not state1:
                    logger.collect_data_DYNAMIC(
                        duration_sec=TEST_DURATION_S,
                        max_samples=100
                    )

                if self.current_state == State.RUNNING:
                    if state1 and not state2:
                        (self.collected_temperatures,
                         self.collected_flow,
                         self.collected_wfs_temp,
                         self.collected_timestamps) = logger.get_data_static()
                        print("Static data collected")
                        self.transition_to(State.COMPLETED)
                    elif state2 and not state1:
                        self.collected_temperatures, self.collected_timestamps = \
                            logger.get_data_DYNAMIC()
                        self.collected_flow     = None
                        self.collected_wfs_temp = None
                        print("Dynamic data collected")
                        self.transition_to(State.COMPLETED)

            # ── COMPLETED ─────────────────────────────────────────────────────
            elif self.current_state == State.COMPLETED:
                print("Test complete — saving data")
                state1 = switch1.is_pressed
                state2 = switch2.is_pressed

                print("Temperatures:", self.collected_temperatures)
                print("Timestamps:",   self.collected_timestamps)
                print("Flow:",         self.collected_flow)
                print("WFS Temp:",     self.collected_wfs_temp)

                if state1 and not state2:
                    # [PICO] Process all captured blocks before writing
                    pico.process_all()

                    pico_rms   = None
                    pico_block = None
                    pico_bn    = None
                    ti_ns      = None

                    if pico.processed_blocks is not None and len(pico.processed_blocks) > 0:
                        pico_rms   = (pico.rms_V, pico.rms_B, pico.rms_H, pico.rms_I)
                        pico_block, pico_bn = pico.get_block_for_csv(PICO_BLOCK_PLOT)
                        ti_ns      = pico.time_interval_ns

                    # [CSV] Write single combined 3-section CSV
                    success = sd_writer.write_to_csv(
                        timestamps           = self.collected_timestamps,
                        temperatures         = self.collected_temperatures,
                        flows                = self.collected_flow,
                        wfs_temps            = self.collected_wfs_temp,
                        pico_rms             = pico_rms,
                        pico_block           = pico_block,
                        pico_block_number    = pico_bn,
                        pico_time_interval_ns= ti_ns,
                    )

                elif state2 and not state1:
                    success = sd_writer.write_to_csv(
                        timestamps   = self.collected_timestamps,
                        temperatures = self.collected_temperatures,
                        flows        = None,
                        wfs_temps    = None,
                    )
                else:
                    print("Error: invalid test mode state")
                    success = False

                if success:
                    logger.clear_data()
                    self.transition_to(State.IDLE)
                else:
                    self.transition_to(State.ERROR, "CSV write failed")

            # ── ERROR ─────────────────────────────────────────────────────────
            elif self.current_state == State.ERROR:
                print(f"ERROR: {self.get_error_message()}")
                # [GENERATOR] [PICO] Safety net — ensure both are off in error state
                try: generator.off()
                except Exception: pass
                try: pico.close()
                except Exception: pass
                time.sleep(2)
                self.transition_to(State.IDLE)


# =============================================================================
#  ENTRY POINT
# =============================================================================
KVEBench = StateMachine()
KVEBench.run()
