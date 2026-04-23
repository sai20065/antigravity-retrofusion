# modules/simulator.py
# Full virtual prototype — replaces all hardware for demo/development

import numpy as np
import math
import time
import threading
from dataclasses import dataclass
from typing import Optional
import random


# Road asset database (Bengaluru demo route)
ROAD_ASSETS = [
    {"id": "S001", "type": "sign",    "class": "sign_RA2",    "ra_true": 420,
     "lat": 12.9716, "lon": 77.5946, "name": "Speed Limit 60"},
    {"id": "M001", "type": "marking", "class": "marking_R2",  "ra_true": 185,
     "lat": 12.9718, "lon": 77.5948, "name": "Lane Divider"},
    {"id": "U001", "type": "stud",    "class": "stud_typeI",  "ra_true": 55,
     "lat": 12.9720, "lon": 77.5950, "name": "Road Stud Km14"},
    {"id": "S002", "type": "sign",    "class": "sign_RA1",    "ra_true": 62,
     "lat": 12.9722, "lon": 77.5952, "name": "Junction Sign"},
    {"id": "M002", "type": "marking", "class": "marking_R1",  "ra_true": 310,
     "lat": 12.9724, "lon": 77.5954, "name": "Stop Line"},
    {"id": "S003", "type": "sign",    "class": "sign_RA2",    "ra_true": 398,
     "lat": 12.9726, "lon": 77.5956, "name": "Direction Sign"},
    {"id": "U002", "type": "stud",    "class": "stud_typeII", "ra_true": 285,
     "lat": 12.9728, "lon": 77.5958, "name": "Active LED Stud"},
    {"id": "M003", "type": "marking", "class": "marking_R2",  "ra_true": 121,
     "lat": 12.9730, "lon": 77.5960, "name": "Zebra Crossing"},
]


def simulate_measurement(asset: dict, night: bool = False,
                          rain: bool = False, fog: bool = False) -> dict:
    """
    Generate realistic sensor readings for a road asset.

    Retroreflectometer: most accurate, σ ≈ 5 mcd/lux/m²
    AI model:           moderate accuracy, bias from lighting conditions
    Physics sensor:     moderate accuracy, degrades in weather
    """
    ra_true = asset["ra_true"]

    # — RetroMeter (ground truth, tightest noise) —
    retro_ra = ra_true + np.random.normal(0, 5.0)

    # — AI model (has bias + lighting-dependent noise) —
    ai_bias  = 15.0 if night else (8.0 if fog else 5.0)
    ai_noise = 45.0 if rain  else (35.0 if fog else 20.0)
    ai_ra    = ra_true + ai_bias + np.random.normal(0, ai_noise)
    ai_conf  = 0.42 if (rain or fog) else (0.68 if night else 0.82)

    # — Physics sensor (BH1750 photometric model) —
    sensor_noise = 30.0 if rain else (20.0 if fog else 12.0)
    sensor_ra    = ra_true + np.random.normal(0, sensor_noise)
    sensor_snr   = 2.1 if rain else (4.5 if fog else 8.5)

    # Ambient lux (simulated)
    ambient_lux = 0.5 if night else (5.0 if fog else 45000.0)

    weather = "fog" if fog else ("rain" if rain else "clear")

    return {
        "asset":          asset,
        "retro_ra":       max(0, retro_ra),
        "ai_ra":          max(0, ai_ra),
        "ai_confidence":  ai_conf,
        "sensor_ra":      max(0, sensor_ra),
        "sensor_snr":     sensor_snr,
        "ambient_lux":    ambient_lux,
        "weather":        weather,
        "lat":            asset["lat"] + np.random.normal(0, 0.00001),
        "lon":            asset["lon"] + np.random.normal(0, 0.00001),
        "speed_kmh":      random.uniform(30, 60),
        "timestamp":      time.time(),
    }


class StreamingSimulator:
    """
    Continuously streams simulated measurements in a background thread.
    Mimics the real hardware data pipeline.
    """
    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._latest   = None
        self._lock     = threading.Lock()
        self._running  = False
        self._t        = 0
        self._history  = []

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            asset  = random.choice(ROAD_ASSETS)
            night  = (self._t % 100) > 70    # simulate day/night cycle
            rain   = (self._t % 50)  > 40    # occasional rain
            result = simulate_measurement(asset, night=night, rain=rain)
            with self._lock:
                self._latest = result
                self._history.append(result)
                if len(self._history) > 1000:
                    self._history.pop(0)
            self._t += 1
            time.sleep(self._interval)

    def get_latest(self) -> Optional[dict]:
        with self._lock:
            return self._latest

    def get_history(self, n: int = 100) -> list:
        with self._lock:
            return list(self._history[-n:])
