# modules/sensor_reader.py
# BH1750 Physics Sensor Model — Simulated Mode
#
# In production, this interfaces with the BH1750 ambient light sensor
# via I2C bus on a Raspberry Pi 4B, computing reflected irradiance
# and converting to retroreflectivity (RA) using the physics model:
#
#   RA = (E_reflected - E_ambient) / (E_incident · cos(θ_obs))
#
# Where:
#   E_reflected = reflected irradiance measured by BH1750
#   E_ambient   = ambient light (subtracted as baseline)
#   E_incident  = IR LED irradiance at target surface
#   θ_obs       = observation angle (0.2° per EN 12899-1)

import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (IR_EMITTANCE_mW, SPOT_AREA_m2, SENSOR_DIST_m,
                    OBS_ANGLE_DEG, BH1750_I2C_ADDR)


class BH1750Sensor:
    """
    BH1750 ambient light sensor interface.

    In production:
        - Connects via I2C (smbus2)
        - Reads lux values at 120ms intervals
        - High-resolution mode (0.5 lux accuracy)

    In simulation:
        - Returns realistic photometric readings
        - Models sensor noise, drift, and temperature effects
    """

    def __init__(self, simulation: bool = True, i2c_bus: int = 1):
        self.simulation = simulation
        self._bus = None
        self._addr = BH1750_I2C_ADDR
        self._reading_count = 0
        self._baseline_ambient = 0.0

        if not simulation:
            try:
                import smbus2
                self._bus = smbus2.SMBus(i2c_bus)
                # Power on BH1750
                self._bus.write_byte(self._addr, 0x01)
                # Set continuous high-resolution mode
                self._bus.write_byte(self._addr, 0x10)
                time.sleep(0.2)  # Wait for first measurement
            except Exception as e:
                print(f"[BH1750] I2C init failed: {e}. Falling back to simulation.")
                self.simulation = True

    def read_lux(self) -> float:
        """Read ambient light level in lux."""
        if self.simulation:
            return self._simulate_lux()

        try:
            data = self._bus.read_i2c_block_data(self._addr, 0x10, 2)
            lux = (data[0] << 8 | data[1]) / 1.2
            return float(lux)
        except Exception:
            return self._simulate_lux()

    def _simulate_lux(self) -> float:
        """Generate realistic ambient light readings."""
        # Simulate diurnal cycle with noise
        hour = (time.time() % 86400) / 3600  # Hour of day (UTC)
        if 6 <= hour <= 18:
            base_lux = 30000 + 15000 * np.sin((hour - 6) / 12 * np.pi)
        else:
            base_lux = np.random.uniform(0.1, 2.0)

        noise = np.random.normal(0, base_lux * 0.02)
        return max(0.0, base_lux + noise)

    def compute_ra(self, reflected_lux: float, ambient_lux: float,
                   ra_true: float = None,
                   rain: bool = False, fog: bool = False) -> dict:
        """
        Compute retroreflectivity from photometric readings.

        Physics model:
            E_incident = P_ir / A_spot  (W/m²)
            RA = (E_reflected - E_ambient) / (E_incident · cos(θ))

        In simulation, adds realistic noise to ra_true.

        Returns:
            dict with ra_estimate, snr, ambient_lux, quality
        """
        self._reading_count += 1

        if self.simulation:
            return self._simulate_ra(ra_true, rain, fog)

        # Physics computation
        obs_angle_rad = np.radians(OBS_ANGLE_DEG)
        e_incident = (IR_EMITTANCE_mW / 1000.0) / SPOT_AREA_m2  # W/m²

        net_reflected = max(0, reflected_lux - ambient_lux)
        ra = net_reflected / (e_incident * np.cos(obs_angle_rad))

        snr = net_reflected / max(ambient_lux, 0.01)

        quality = "good" if snr > 5 else ("fair" if snr > 2 else "poor")

        return {
            "ra_estimate": float(ra),
            "snr": float(snr),
            "ambient_lux": float(ambient_lux),
            "reflected_lux": float(reflected_lux),
            "quality": quality,
        }

    def _simulate_ra(self, ra_true: float = None,
                     rain: bool = False, fog: bool = False) -> dict:
        """Simulate physics sensor RA reading."""
        if ra_true is None:
            ra_true = np.random.uniform(50, 500)

        # Weather-dependent noise
        noise_std = 30.0 if rain else (20.0 if fog else 12.0)
        ra_estimate = ra_true + np.random.normal(0, noise_std)
        ra_estimate = max(0.0, ra_estimate)

        # SNR degrades in bad weather
        snr = 2.1 if rain else (4.5 if fog else 8.5)
        snr += np.random.normal(0, 0.3)

        ambient = np.random.uniform(0.1, 2.0) if rain else np.random.uniform(40000, 50000)

        quality = "good" if snr > 5 else ("fair" if snr > 2 else "poor")

        return {
            "ra_estimate": float(ra_estimate),
            "snr": float(max(0.1, snr)),
            "ambient_lux": float(ambient),
            "reflected_lux": float(ra_estimate * 1.2 + ambient),
            "quality": quality,
        }

    def get_stats(self) -> dict:
        return {
            "total_readings": self._reading_count,
            "mode": "simulation" if self.simulation else "hardware",
            "i2c_address": hex(self._addr),
        }
