# modules/gps_module.py
# GPS Module — UART NEO-M8N NMEA Parser
#
# In production (Raspberry Pi):
#   - Reads NMEA sentences from u-blox NEO-M8N via UART (/dev/ttyS0)
#   - Parses GPGGA (position) and GPRMC (velocity) sentences
#   - Returns lat, lon, altitude, speed, hdop
#
# In simulation:
#   - Generates GPS fixes along a configurable demo route
#   - Bengaluru ring road route for hackathon demo

import time
import threading
from typing import Optional
from dataclasses import dataclass
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GPS_UART_BAUD


@dataclass
class GPSFix:
    """GPS position fix data."""
    latitude: float        # WGS84 decimal degrees
    longitude: float       # WGS84 decimal degrees
    altitude: float        # meters above sea level
    speed_kmh: float       # speed over ground (km/h)
    hdop: float            # horizontal dilution of precision
    satellites: int        # number of satellites in view
    fix_quality: int       # 0=invalid, 1=GPS, 2=DGPS
    timestamp: float       # Unix timestamp of the fix


# Demo route — Bengaluru ring road waypoints
DEMO_ROUTE = [
    (12.9716, 77.5946),  # Start: MG Road
    (12.9718, 77.5948),
    (12.9720, 77.5950),
    (12.9722, 77.5952),
    (12.9724, 77.5954),
    (12.9726, 77.5956),
    (12.9728, 77.5958),
    (12.9730, 77.5960),
    (12.9732, 77.5962),
    (12.9734, 77.5964),
    (12.9736, 77.5966),  # End: ORR junction
]


class GPSModule:
    """
    GPS module interface with NMEA parsing.

    Design reference: Section 2.1, 3.2 of RetroFusion Design Document.

    In production:
        - UART at 9600 baud on /dev/ttyS0 (RPi hardware UART)
        - Parses $GPGGA and $GPRMC sentences via pynmea2
        - 1Hz update rate (standard for NEO-M8N)

    In simulation:
        - Moves along DEMO_ROUTE at configurable speed
        - Adds GPS noise (±2.5m CEP)
    """

    def __init__(self, port: str = "/dev/ttyS0", baud: int = None,
                 simulation: bool = True):
        self.simulation = simulation
        self._port = port
        self._baud = baud or GPS_UART_BAUD
        self._serial = None
        self._latest_fix = None
        self._lock = threading.Lock()
        self._running = False
        self._route_idx = 0
        self._fix_count = 0

        if not simulation:
            self._init_uart()
        else:
            print(f"[GPS] Simulation mode — demo route with {len(DEMO_ROUTE)} waypoints")

    def _init_uart(self):
        """Initialize UART serial connection to GPS module."""
        try:
            import serial
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=1.0
            )
            print(f"[GPS] UART opened: {self._port} @ {self._baud} baud")
        except ImportError:
            print("[GPS] pyserial not installed. Falling back to simulation.")
            self.simulation = True
        except Exception as e:
            print(f"[GPS] UART init failed: {e}. Falling back to simulation.")
            self.simulation = True

    def start(self):
        """Start background GPS reading thread."""
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        print("[GPS] Reader thread started")

    def stop(self):
        """Stop GPS reading."""
        self._running = False
        if self._serial:
            self._serial.close()

    def _read_loop(self):
        """Background thread: continuously read GPS data."""
        while self._running:
            if self.simulation:
                fix = self._simulate_fix()
            else:
                fix = self._read_nmea()

            if fix:
                with self._lock:
                    self._latest_fix = fix
                    self._fix_count += 1

            time.sleep(1.0)  # 1Hz update rate

    def _read_nmea(self) -> Optional[GPSFix]:
        """Parse real NMEA sentences from GPS module."""
        try:
            import pynmea2

            line = self._serial.readline().decode("ascii", errors="replace").strip()
            if not line:
                return None

            # Parse GPGGA for position + quality
            if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                msg = pynmea2.parse(line)
                if msg.gps_qual > 0:
                    return GPSFix(
                        latitude=msg.latitude,
                        longitude=msg.longitude,
                        altitude=float(msg.altitude or 0),
                        speed_kmh=0.0,  # GPGGA doesn't have speed
                        hdop=float(msg.horizontal_dil or 99),
                        satellites=int(msg.num_sats or 0),
                        fix_quality=int(msg.gps_qual),
                        timestamp=time.time(),
                    )

            # Parse GPRMC for speed
            elif line.startswith("$GPRMC") or line.startswith("$GNRMC"):
                msg = pynmea2.parse(line)
                if msg.status == "A":  # Active fix
                    speed_knots = float(msg.spd_over_grnd or 0)
                    return GPSFix(
                        latitude=msg.latitude,
                        longitude=msg.longitude,
                        altitude=0.0,
                        speed_kmh=speed_knots * 1.852,  # knots → km/h
                        hdop=1.0,
                        satellites=0,
                        fix_quality=1,
                        timestamp=time.time(),
                    )

        except Exception as e:
            pass  # Silent fail, try again next cycle

        return None

    def _simulate_fix(self) -> GPSFix:
        """Generate simulated GPS fix along demo route."""
        # Cycle through route waypoints
        idx = self._route_idx % len(DEMO_ROUTE)
        lat, lon = DEMO_ROUTE[idx]

        # Add realistic GPS noise (±2.5m CEP ≈ ±0.000022°)
        noise_lat = np.random.normal(0, 0.000022)
        noise_lon = np.random.normal(0, 0.000022)

        # Simulate realistic speed (30-60 km/h)
        speed = np.random.uniform(30, 60)

        self._route_idx += 1

        return GPSFix(
            latitude=lat + noise_lat,
            longitude=lon + noise_lon,
            altitude=920.0 + np.random.normal(0, 2),  # Bengaluru ~920m ASL
            speed_kmh=speed,
            hdop=0.8 + np.random.uniform(0, 0.5),
            satellites=np.random.randint(8, 14),
            fix_quality=1,
            timestamp=time.time(),
        )

    def get_fix(self) -> Optional[GPSFix]:
        """Get the latest GPS fix (thread-safe)."""
        with self._lock:
            return self._latest_fix

    def get_position(self) -> Optional[tuple]:
        """Get (lat, lon) tuple or None."""
        fix = self.get_fix()
        if fix:
            return (fix.latitude, fix.longitude)
        return None

    def get_stats(self) -> dict:
        """Return GPS module statistics."""
        fix = self.get_fix()
        return {
            "mode": "simulation" if self.simulation else "hardware",
            "total_fixes": self._fix_count,
            "has_fix": fix is not None,
            "satellites": fix.satellites if fix else 0,
            "hdop": fix.hdop if fix else 99,
            "speed_kmh": round(fix.speed_kmh, 1) if fix else 0,
        }

    def __del__(self):
        self.stop()
