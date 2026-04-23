# tests/test_fusion.py
# Unit tests for the EKF fusion engine

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from modules.fusion_engine import ExtendedKalmanFilter, SensorMeasurement, FusionResult


class TestExtendedKalmanFilter:
    """Test suite for the EKF sensor fusion engine."""

    def test_initialization(self):
        """EKF should initialize with default state."""
        ekf = ExtendedKalmanFilter(dt=0.5)
        assert ekf.x[0] == 200.0  # initial RA guess
        assert ekf.x[1] == 0.0    # initial dRA/dt
        assert ekf.P[0, 0] == 500.0  # high initial uncertainty

    def test_single_update(self):
        """Single update should produce valid fusion result."""
        ekf = ExtendedKalmanFilter(dt=0.5)
        meas = SensorMeasurement(
            ai_ra=250.0, ai_confidence=0.8,
            sensor_ra=240.0, sensor_snr=8.0,
            retro_ra=245.0, retro_available=True,
            weather="clear",
        )
        result = ekf.update(meas)

        assert isinstance(result, FusionResult)
        assert result.final_ra > 0
        assert result.ekf_variance > 0
        assert abs(result.alpha + result.beta + result.gamma - 1.0) < 0.01

    def test_retro_gets_highest_weight(self):
        """RetroMeter should get highest weight when connected."""
        ekf = ExtendedKalmanFilter(dt=0.5)
        meas = SensorMeasurement(
            ai_ra=250.0, ai_confidence=0.8,
            sensor_ra=240.0, sensor_snr=8.0,
            retro_ra=245.0, retro_available=True,
            weather="clear",
        )
        result = ekf.update(meas)
        assert result.alpha > result.beta
        assert result.alpha > result.gamma

    def test_no_retro_increases_ai_weight(self):
        """Without RetroMeter, AI and sensor weights should increase."""
        ekf = ExtendedKalmanFilter(dt=0.5)
        meas = SensorMeasurement(
            ai_ra=250.0, ai_confidence=0.8,
            sensor_ra=240.0, sensor_snr=8.0,
            retro_available=False,
            weather="clear",
        )
        result = ekf.update(meas)
        assert result.alpha == 0.0  # no retro weight
        assert result.beta > 0     # AI has nonzero weight
        assert result.gamma > 0    # Sensor has nonzero weight
        assert abs(result.beta + result.gamma - 1.0) < 0.01  # weights sum to 1

    def test_variance_decreases_over_time(self):
        """EKF variance should decrease with consistent measurements."""
        ekf = ExtendedKalmanFilter(dt=0.5)
        variances = []
        for _ in range(10):
            meas = SensorMeasurement(
                ai_ra=200.0, ai_confidence=0.9,
                sensor_ra=200.0, sensor_snr=9.0,
                retro_ra=200.0, retro_available=True,
                weather="clear",
            )
            result = ekf.update(meas)
            variances.append(result.ekf_variance)

        # Variance should generally decrease
        assert variances[-1] < variances[0]

    def test_weather_increases_noise(self):
        """Rain/fog should increase measurement noise."""
        ekf_clear = ExtendedKalmanFilter(dt=0.5)
        ekf_rain = ExtendedKalmanFilter(dt=0.5)

        meas_clear = SensorMeasurement(
            ai_ra=200.0, ai_confidence=0.8,
            sensor_ra=200.0, sensor_snr=8.0,
            retro_ra=200.0, retro_available=True,
            weather="clear",
        )
        meas_rain = SensorMeasurement(
            ai_ra=200.0, ai_confidence=0.8,
            sensor_ra=200.0, sensor_snr=8.0,
            retro_ra=200.0, retro_available=True,
            weather="rain",
        )

        result_clear = ekf_clear.update(meas_clear)
        result_rain = ekf_rain.update(meas_rain)

        # Rain should result in higher uncertainty
        assert result_rain.ekf_variance > result_clear.ekf_variance

    def test_bias_correction(self):
        """AI bias should be tracked and corrected over time."""
        ekf = ExtendedKalmanFilter(dt=0.5)

        # Feed consistent bias (AI always reads 20 higher than retro)
        for _ in range(50):
            meas = SensorMeasurement(
                ai_ra=220.0, ai_confidence=0.8,
                sensor_ra=200.0, sensor_snr=8.0,
                retro_ra=200.0, retro_available=True,
                weather="clear",
            )
            result = ekf.update(meas)

        # Bias estimate should be negative (to correct AI downward)
        assert result.bias_estimate < 0

    def test_ra_clamped_to_positive(self):
        """RA output should never be negative."""
        ekf = ExtendedKalmanFilter(dt=0.5)
        meas = SensorMeasurement(
            ai_ra=0.0, ai_confidence=0.1,
            sensor_ra=0.0, sensor_snr=0.5,
            retro_ra=0.0, retro_available=True,
            weather="snow",
        )
        result = ekf.update(meas)
        assert result.final_ra >= 0


class TestSensorMeasurement:
    """Test SensorMeasurement dataclass."""

    def test_default_values(self):
        meas = SensorMeasurement(
            ai_ra=100.0, ai_confidence=0.5,
            sensor_ra=95.0, sensor_snr=5.0,
        )
        assert meas.retro_ra is None
        assert meas.retro_available is False
        assert meas.weather == "clear"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
