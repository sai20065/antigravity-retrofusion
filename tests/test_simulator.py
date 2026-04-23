# tests/test_simulator.py
# Unit tests for the virtual sensor simulator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
import pytest
from modules.simulator import simulate_measurement, ROAD_ASSETS, StreamingSimulator


class TestSimulateMeasurement:
    """Test the measurement simulation function."""

    def test_returns_all_fields(self):
        """Simulation should return all required fields."""
        asset = ROAD_ASSETS[0]
        result = simulate_measurement(asset)
        assert "retro_ra" in result
        assert "ai_ra" in result
        assert "ai_confidence" in result
        assert "sensor_ra" in result
        assert "sensor_snr" in result
        assert "weather" in result
        assert "lat" in result
        assert "lon" in result
        assert "speed_kmh" in result
        assert "timestamp" in result
        assert "asset" in result

    def test_positive_values(self):
        """All RA values should be non-negative."""
        for asset in ROAD_ASSETS:
            result = simulate_measurement(asset)
            assert result["retro_ra"] >= 0
            assert result["ai_ra"] >= 0
            assert result["sensor_ra"] >= 0

    def test_clear_weather(self):
        """Clear weather should produce 'clear' weather code."""
        asset = ROAD_ASSETS[0]
        result = simulate_measurement(asset, rain=False, fog=False)
        assert result["weather"] == "clear"

    def test_rain_weather(self):
        """Rain should produce 'rain' weather code."""
        asset = ROAD_ASSETS[0]
        result = simulate_measurement(asset, rain=True)
        assert result["weather"] == "rain"

    def test_fog_weather(self):
        """Fog should produce 'fog' weather code."""
        asset = ROAD_ASSETS[0]
        result = simulate_measurement(asset, fog=True)
        assert result["weather"] == "fog"

    def test_night_reduces_confidence(self):
        """Night mode should reduce AI confidence."""
        asset = ROAD_ASSETS[0]
        day = simulate_measurement(asset, night=False)
        night = simulate_measurement(asset, night=True)
        assert night["ai_confidence"] < day["ai_confidence"]

    def test_rain_reduces_confidence(self):
        """Rain should reduce AI confidence."""
        asset = ROAD_ASSETS[0]
        clear = simulate_measurement(asset, rain=False)
        rainy = simulate_measurement(asset, rain=True)
        assert rainy["ai_confidence"] < clear["ai_confidence"]

    def test_gps_noise(self):
        """GPS coordinates should have small noise around true position."""
        asset = ROAD_ASSETS[0]
        result = simulate_measurement(asset)
        assert abs(result["lat"] - asset["lat"]) < 0.001
        assert abs(result["lon"] - asset["lon"]) < 0.001


class TestRoadAssets:
    """Test the road asset database."""

    def test_all_assets_have_required_fields(self):
        """All assets should have required fields."""
        for asset in ROAD_ASSETS:
            assert "id" in asset
            assert "type" in asset
            assert "class" in asset
            assert "ra_true" in asset
            assert "lat" in asset
            assert "lon" in asset
            assert "name" in asset

    def test_asset_types(self):
        """Asset types should be valid."""
        valid_types = {"sign", "marking", "stud"}
        for asset in ROAD_ASSETS:
            assert asset["type"] in valid_types

    def test_ra_true_positive(self):
        """True RA values should be positive."""
        for asset in ROAD_ASSETS:
            assert asset["ra_true"] > 0


class TestStreamingSimulator:
    """Test the streaming simulator."""

    def test_start_stop(self):
        """Simulator should start and stop cleanly."""
        sim = StreamingSimulator(interval=0.1)
        sim.start()
        time.sleep(0.5)
        sim.stop()
        assert True  # No crash

    def test_produces_data(self):
        """Simulator should produce data after starting."""
        sim = StreamingSimulator(interval=0.1)
        sim.start()
        time.sleep(0.5)
        latest = sim.get_latest()
        sim.stop()
        assert latest is not None

    def test_history_grows(self):
        """History should accumulate measurements."""
        sim = StreamingSimulator(interval=0.1)
        sim.start()
        time.sleep(1.0)
        history = sim.get_history(50)
        sim.stop()
        assert len(history) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
