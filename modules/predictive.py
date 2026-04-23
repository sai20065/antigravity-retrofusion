# modules/predictive.py
# Predictive Maintenance — RA Exponential Decay Model
#
# Physics:  RA(t) = RA₀ · exp(-λ_deg · t) + ε(t)
#   RA₀    = initial RA at installation
#   λ_deg  = degradation rate constant (material-dependent)
#   ε(t)   = stochastic noise term
#
# Multi-factor λ: λ = λ_base · f(UV_index, traffic, rainfall)

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEGRADATION_RATES, THRESHOLDS
from typing import Optional
from dataclasses import dataclass


@dataclass
class MaintenanceForecast:
    asset_id:          str
    asset_type:        str
    current_ra:        float
    ra0_estimated:     float
    lambda_decay:      float
    days_to_failure:   Optional[float]
    days_uncertainty:  float
    forecast_confidence: float
    recommended_action: str
    urgency:           str    # "immediate" | "within_30_days" | "within_90_days" | "ok"


def ra_decay_model(t: np.ndarray, ra0: float, lam: float) -> np.ndarray:
    """
    Exponential decay:  RA(t) = RA₀ · exp(−λ · t)
    t in days since first measurement.
    """
    return ra0 * np.exp(-lam * t)


def predict_failure(timestamps: list, ra_values: list,
                    asset_id: str, asset_type: str,
                    material: str = "type_ii") -> MaintenanceForecast:
    """
    Fit exponential decay to historical RA readings.
    Uses scipy curve_fit (Levenberg-Marquardt algorithm).

    Uncertainty in failure date:
        t_fail = -ln(threshold / RA₀) / λ
        σ_t    = t_fail · (σ_λ / λ)   [error propagation]
    """
    threshold_key = asset_type
    thresh = THRESHOLDS.get(threshold_key, {}).get("pass", 100)

    t  = np.array(timestamps, dtype=float)
    t -= t[0]   # relative days
    ra = np.array(ra_values,   dtype=float)

    lambda_prior = DEGRADATION_RATES.get(material, 0.001)
    ra_current   = float(ra[-1])

    if len(t) < 3:
        return MaintenanceForecast(
            asset_id=asset_id, asset_type=asset_type,
            current_ra=ra_current, ra0_estimated=ra[0],
            lambda_decay=lambda_prior, days_to_failure=None,
            days_uncertainty=0, forecast_confidence=0.0,
            recommended_action="Insufficient data — collect more readings",
            urgency="ok"
        )

    try:
        popt, pcov = curve_fit(
            ra_decay_model, t, ra,
            p0=[ra[0], lambda_prior],
            bounds=([0, 1e-6], [2000, 0.1]),
            maxfev=10000
        )
        ra0, lam    = popt
        sigma_ra0   = np.sqrt(max(0, pcov[0, 0]))
        sigma_lam   = np.sqrt(max(0, pcov[1, 1]))

        # Coefficient of variation as confidence proxy
        cv = sigma_lam / max(lam, 1e-9)
        forecast_conf = max(0.0, 1.0 - cv)

        # Pearson correlation as goodness-of-fit
        fitted    = ra_decay_model(t, ra0, lam)
        corr, _   = pearsonr(ra, fitted)
        forecast_conf = float(np.clip(forecast_conf * abs(corr), 0, 1))

        # Failure date: solve RA(t_fail) = threshold
        if lam > 0 and ra0 > thresh:
            t_fail_total  = -np.log(thresh / ra0) / lam
            t_fail_days   = t_fail_total - t[-1]   # days from NOW
            t_fail_err    = t_fail_total * (sigma_lam / max(lam, 1e-9))
        else:
            t_fail_days = None
            t_fail_err  = 0.0

        # Urgency classification
        if ra_current < thresh:
            urgency = "immediate"
            action  = "REPLACE IMMEDIATELY — asset already below threshold"
        elif t_fail_days is not None and t_fail_days < 30:
            urgency = "within_30_days"
            action  = f"Schedule replacement within 30 days (est. {t_fail_days:.0f} days)"
        elif t_fail_days is not None and t_fail_days < 90:
            urgency = "within_90_days"
            action  = f"Plan maintenance within 90 days (est. {t_fail_days:.0f} days)"
        else:
            urgency = "ok"
            action  = "No action needed — asset within compliance"

        return MaintenanceForecast(
            asset_id=asset_id, asset_type=asset_type,
            current_ra=ra_current, ra0_estimated=float(ra0),
            lambda_decay=float(lam), days_to_failure=t_fail_days,
            days_uncertainty=float(t_fail_err),
            forecast_confidence=forecast_conf,
            recommended_action=action, urgency=urgency
        )

    except RuntimeError:
        return MaintenanceForecast(
            asset_id=asset_id, asset_type=asset_type,
            current_ra=ra_current, ra0_estimated=ra[0],
            lambda_decay=lambda_prior, days_to_failure=None,
            days_uncertainty=0, forecast_confidence=0.0,
            recommended_action="Curve fit failed — manual inspection recommended",
            urgency="ok"
        )
