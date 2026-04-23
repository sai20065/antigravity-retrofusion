# modules/fusion_engine.py
# Extended Kalman Filter + Confidence-Weighted Sensor Fusion
# Math reference: Section 5 of RetroFusion Technical Design Doc v1.0

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (SIGMA_AI, SIGMA_SENSOR, SIGMA_RETRO,
                    SIGMA_RA_PROCESS, BIAS_LAMBDA, WEATHER_MULTIPLIERS)


@dataclass
class SensorMeasurement:
    """Container for a single measurement cycle."""
    ai_ra:         float
    ai_confidence: float          # 0–1 from MobileNetV2 conf head
    sensor_ra:     float
    sensor_snr:    float          # (E_reflected - E_ambient) / E_ambient
    retro_ra:      Optional[float] = None
    retro_available: bool          = False
    weather:       str             = "clear"   # clear|rain|fog|snow


@dataclass
class FusionResult:
    final_ra:     float
    ekf_variance: float
    alpha:        float   # retro weight
    beta:         float   # AI weight
    gamma:        float   # sensor weight
    bias_estimate: float


class ExtendedKalmanFilter:
    """
    State vector:  x = [RA, dRA/dt]ᵀ
    Measurement:   z = [AI_RA, Sensor_RA, RetroMeter_RA]ᵀ  (or subset)

    State transition (constant velocity):
        F = [[1, Δt],
             [0,  1]]

    Process noise Q:
        Q = σ_ra² * [[Δt³/3, Δt²/2],
                     [Δt²/2, Δt  ]]

    Measurement matrix H (all sensors observe RA = state[0]):
        H = [[1, 0],
             [1, 0],
             [1, 0]]
    """

    def __init__(self, dt: float = 0.2):
        self.dt = dt

        # State: [RA, dRA/dt]
        self.x = np.array([200.0, 0.0])          # initial guess
        self.P = np.eye(2) * 500.0               # initial covariance (high uncertainty)

        # State transition matrix
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])

        # Process noise
        s = SIGMA_RA_PROCESS
        self.Q = s**2 * np.array([[dt**3/3, dt**2/2],
                                   [dt**2/2, dt]])

        # Measurement matrix (each sensor measures RA directly)
        self.H = np.array([[1.0, 0.0],
                           [1.0, 0.0],
                           [1.0, 0.0]])

        # Running bias estimate for AI drift correction
        self._bias = 0.0

    def _build_R(self, meas: SensorMeasurement) -> tuple:
        """
        Build dynamic measurement noise matrix R_k.

        σᵢ²(k) = σᵢ,base² / cᵢ(k)²
        Low confidence → high variance → sensor is downweighted in Kalman gain.

        Weather noise inflation applied per config multipliers.
        """
        mult = WEATHER_MULTIPLIERS.get(meas.weather, WEATHER_MULTIPLIERS["clear"])

        # Confidence-adjusted noise
        c_ai     = max(0.01, meas.ai_confidence)
        c_sensor = max(0.01, min(1.0, meas.sensor_snr / 10.0))  # SNR→conf
        c_retro  = 0.95 if meas.retro_available else 0.0

        sigma_ai     = (SIGMA_AI     * mult["ai"])     / c_ai
        sigma_sensor = (SIGMA_SENSOR * mult["sensor"]) / c_sensor
        sigma_retro  = (SIGMA_RETRO  * mult["retro"])  / max(0.01, c_retro)

        active = [True, True, meas.retro_available]
        R_diag = [sigma_ai**2, sigma_sensor**2, sigma_retro**2]
        return np.diag(R_diag), active

    def _ai_bias_correction(self, meas: SensorMeasurement) -> float:
        """
        Online AI drift correction via exponential moving average.

        bias_k = RetroMeter_RA_k - AI_RA_k
        B̂_k   = (1 - λ)·B̂_{k-1} + λ·bias_k
        AI_RA_corrected = AI_RA_raw + B̂_k
        """
        if meas.retro_available and meas.retro_ra is not None:
            instant_bias = meas.retro_ra - meas.ai_ra
            self._bias = (1 - BIAS_LAMBDA) * self._bias + BIAS_LAMBDA * instant_bias
        return meas.ai_ra + self._bias

    def update(self, meas: SensorMeasurement) -> FusionResult:
        """
        Full EKF predict + update cycle.

        Predict:
            x̂_{k|k-1} = F · x̂_{k-1|k-1}
            P_{k|k-1}  = F · P_{k-1|k-1} · Fᵀ + Q

        Innovation:
            ỹ_k = z_k - H · x̂_{k|k-1}
            S_k = H · P_{k|k-1} · Hᵀ + R_k

        Kalman Gain:
            K_k = P_{k|k-1} · Hᵀ · S_k⁻¹

        Update:
            x̂_{k|k} = x̂_{k|k-1} + K_k · ỹ_k
            P_{k|k}  = (I - K_k · H) · P_{k|k-1}
        """
        # ── Bias-corrected AI ──────────────────────────────────────────────
        ai_corrected = self._ai_bias_correction(meas)

        # ── Build measurement vector & noise matrix ────────────────────────
        R_full, active = self._build_R(meas)
        retro_val = meas.retro_ra if (meas.retro_available and meas.retro_ra) else self.x[0]
        z_full = np.array([ai_corrected, meas.sensor_ra, retro_val])

        # Filter to only active sensors
        idx = [i for i, a in enumerate(active) if a]
        z = z_full[idx]
        H = self.H[idx]
        R = R_full[np.ix_(idx, idx)]

        # ── PREDICT ───────────────────────────────────────────────────────
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── INNOVATION ────────────────────────────────────────────────────
        y = z - H @ x_pred                        # Innovation
        S = H @ P_pred @ H.T + R                  # Innovation covariance

        # ── KALMAN GAIN ───────────────────────────────────────────────────
        K = P_pred @ H.T @ np.linalg.inv(S)       # Optimal gain

        # ── UPDATE ────────────────────────────────────────────────────────
        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ H) @ P_pred

        # Clamp RA to physical range
        self.x[0] = max(0.0, self.x[0])

        # ── Compute analytical weights (steady-state approximation) ───────
        # α = (c_R²/σ_R²)/Z,  β = (c_AI²/σ_AI²)/Z,  γ = (c_S²/σ_S²)/Z
        c_ai     = max(0.01, meas.ai_confidence)
        c_sensor = max(0.01, min(1.0, meas.sensor_snr / 10.0))
        c_retro  = 0.95 if meas.retro_available else 0.0

        w_retro  = c_retro**2  / max(1e-9, SIGMA_RETRO**2)
        w_ai     = c_ai**2     / max(1e-9, SIGMA_AI**2)
        w_sensor = c_sensor**2 / max(1e-9, SIGMA_SENSOR**2)
        Z = w_retro + w_ai + w_sensor

        return FusionResult(
            final_ra      = float(self.x[0]),
            ekf_variance  = float(self.P[0, 0]),
            alpha         = float(w_retro  / Z),
            beta          = float(w_ai     / Z),
            gamma         = float(w_sensor / Z),
            bias_estimate = float(self._bias),
        )
