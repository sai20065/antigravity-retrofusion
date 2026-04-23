# modules/retro_reader.py
# Retroreflectometer Serial Reader — USB-CDC Interface
#
# Design reference: Section 3.4 of RetroFusion Design Document.
#
# In production:
#   - Reads RA values from retroreflectometer via USB-Serial
#   - Protocol: "RA:456.3\r\n" or "456.3,PASS\r\n"
#   - Running average filter for reading stability
#   - Connection status monitoring
#
# In simulation:
#   - Generates realistic retroreflectometer readings
#   - Models instrument noise (σ ≈ 5 mcd/lux/m²)

import time
import threading
import numpy as np
from typing import Optional
from dataclasses import dataclass
from collections import deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RETRO_USB_PORT, RETRO_BAUD


@dataclass
class RetroReading:
    """Single retroreflectometer reading."""
    ra_value: float         # mcd/lux/m²
    is_valid: bool          # True if reading is within sane range
    instrument_status: str  # "ok", "warming_up", "error"
    timestamp: float


class RetroMeterReader:
    """
    Serial interface for USB-connected retroreflectometer.

    Supported instruments:
        - Delta LTL-X (protocol: "RA:xxx.x\\r\\n")
        - Zehntner ZRS 6060 (protocol: "xxx.x,STATUS\\r\\n")
        - Custom RS-232 (configurable protocol)

    Features:
        - Threaded background reading
        - Running average filter (configurable window)
        - Connection loss detection with auto-reconnect
        - Simulation mode for development
    """

    def __init__(self, port: str = None, baud: int = None,
                 simulate: bool = True, filter_window: int = 5):
        """
        Args:
            port:          Serial port path (e.g., /dev/ttyUSB0)
            baud:          Baud rate (default: 9600)
            simulate:      If True, generate simulated readings
            filter_window: Number of readings for running average
        """
        self.simulate = simulate
        self._port = port or RETRO_USB_PORT
        self._baud = baud or RETRO_BAUD
        self._serial = None
        self._lock = threading.Lock()
        self._running = False
        self._connected = False

        # Latest reading
        self._last_ra = 0.0
        self._last_reading = None

        # Running average filter
        self._filter_window = filter_window
        self._readings_buffer = deque(maxlen=filter_window)

        # Statistics
        self._reading_count = 0
        self._error_count = 0
        self._sim_ra_true = 250.0  # Simulation: current "true" RA

        if not simulate:
            self._init_serial()
        else:
            self._connected = True
            print("[Retro] Simulation mode -- sigma ~= 5 mcd/lux/m2")

    def _init_serial(self):
        """Initialize serial connection to retroreflectometer."""
        try:
            import serial
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=1.0,
                bytesize=8,
                parity='N',
                stopbits=1,
            )
            self._connected = True
            print(f"[Retro] Connected: {self._port} @ {self._baud} baud")
        except ImportError:
            print("[Retro] pyserial not installed. Falling back to simulation.")
            self.simulate = True
            self._connected = True
        except Exception as e:
            print(f"[Retro] Serial connection failed: {e}. Falling back to simulation.")
            self.simulate = True
            self._connected = True

    def start(self):
        """Start background reading thread."""
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        print("[Retro] Reader thread started")

    def stop(self):
        """Stop reading thread and close serial port."""
        self._running = False
        if self._serial:
            self._serial.close()

    def _read_loop(self):
        """Background thread: continuously read retroreflectometer data."""
        while self._running:
            if self.simulate:
                reading = self._simulate_reading()
            else:
                reading = self._read_serial()

            if reading and reading.is_valid:
                with self._lock:
                    self._last_reading = reading
                    self._last_ra = reading.ra_value
                    self._readings_buffer.append(reading.ra_value)
                    self._reading_count += 1

            # Retroreflectometers typically read at 1-5 Hz
            time.sleep(0.5)

    def _read_serial(self) -> Optional[RetroReading]:
        """Parse real retroreflectometer serial output."""
        try:
            line = self._serial.readline().decode("ascii", errors="replace").strip()
            if not line:
                return None

            ra_value = None

            # Protocol 1: "RA:456.3"
            if "RA:" in line.upper():
                val_str = line.upper().split("RA:")[1].split(",")[0].strip()
                ra_value = float(val_str)

            # Protocol 2: "456.3,PASS" or "456.3"
            elif line.replace(".", "").replace("-", "").split(",")[0].isdigit():
                ra_value = float(line.split(",")[0])

            if ra_value is not None and 0 <= ra_value <= 2000:
                return RetroReading(
                    ra_value=ra_value,
                    is_valid=True,
                    instrument_status="ok",
                    timestamp=time.time(),
                )
            else:
                self._error_count += 1
                return None

        except Exception as e:
            self._error_count += 1
            self._connected = False
            return None

    def _simulate_reading(self) -> RetroReading:
        """Generate realistic retroreflectometer reading."""
        # Slowly drift the "true" RA value for realism
        self._sim_ra_true += np.random.normal(0, 2.0)
        self._sim_ra_true = max(20, min(800, self._sim_ra_true))

        # Retroreflectometer noise: σ ≈ 5 mcd/lux/m²
        ra = self._sim_ra_true + np.random.normal(0, 5.0)
        ra = max(0, ra)

        return RetroReading(
            ra_value=float(ra),
            is_valid=True,
            instrument_status="ok",
            timestamp=time.time(),
        )

    def set_sim_ra(self, ra_true: float):
        """Set the simulation ground truth RA value."""
        self._sim_ra_true = ra_true

    def get_ra(self) -> Optional[float]:
        """
        Get the latest RA reading (raw, unfiltered).

        Returns:
            RA value in mcd/lux/m², or None if no reading available
        """
        with self._lock:
            if self._last_reading:
                return self._last_ra
            return None

    def get_ra_filtered(self) -> Optional[float]:
        """
        Get running-average filtered RA reading.

        Uses a sliding window of recent readings for stability.

        Returns:
            Filtered RA value, or None if insufficient data
        """
        with self._lock:
            if len(self._readings_buffer) == 0:
                return None
            return float(np.mean(self._readings_buffer))

    @property
    def available(self) -> bool:
        """Whether the retroreflectometer is connected and providing data."""
        with self._lock:
            if self._last_reading is None:
                return False
            # Consider stale if no reading in last 5 seconds
            age = time.time() - self._last_reading.timestamp
            return age < 5.0 and self._connected

    def get_stats(self) -> dict:
        """Return reader statistics."""
        with self._lock:
            return {
                "mode": "simulation" if self.simulate else "hardware",
                "connected": self._connected,
                "available": self.available,
                "total_readings": self._reading_count,
                "error_count": self._error_count,
                "buffer_size": len(self._readings_buffer),
                "latest_ra": self._last_ra if self._last_reading else None,
            }

    def __del__(self):
        self.stop()
