# modules/gemma_analyzer.py
# Google Gemma 4 AI Analyzer — Intelligent Defect Analysis & Recommendations
#
# Uses Google's Gemma 4 (open-source, free via google-generativeai API) for:
#   1. Defect Classification — Analyze detected road assets for damage types
#   2. Maintenance Recommendations — Natural language summaries based on RA trends
#   3. Anomaly Detection — Flag unusual patterns in sensor readings
#   4. Compliance Reports — AI-generated reports per EN standards
#
# Gemma 4 is used as an ENHANCEMENT layer on top of the numeric
# MobileNetV2 + EKF pipeline, providing semantic understanding
# and natural language outputs.

import os
import sys
import time
import json
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class GemmaAnalysis:
    """Result from Gemma 4 analysis."""
    defect_type: str           # "fading", "cracking", "dirt", "vandalism", "none"
    severity: str              # "critical", "moderate", "minor", "none"
    recommendation: str        # Natural language maintenance recommendation
    confidence: float          # Analysis confidence (0-1)
    compliance_status: str     # "compliant", "marginal", "non_compliant"
    narrative: str             # Detailed analysis narrative
    timestamp: float


class GemmaAnalyzer:
    """
    Google Gemma 4 intelligent analysis layer.

    Provides AI-powered semantic analysis of road asset conditions
    beyond numeric RA values:
        - WHY is RA low? (dirt, fading, damage, etc.)
        - WHAT action should be taken?
        - WHEN is maintenance needed?
        - HOW does this compare to standards?

    Uses google-generativeai library with the free Gemma 4 model.
    Falls back to rule-based analysis when API is unavailable.
    """

    def __init__(self, api_key: str = None, model_name: str = "gemma-4"):
        """
        Args:
            api_key:    Google AI Studio API key (or set GOOGLE_API_KEY env var)
            model_name: Gemma model to use (default: gemma-4)
        """
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._model_name = model_name
        self._model = None
        self._available = False
        self._analysis_count = 0
        self._lock = threading.Lock()

        self._init_model()

    def _init_model(self):
        """Initialize the Gemma 4 model via google-generativeai."""
        if not self._api_key:
            print("[Gemma] No API key found. Set GOOGLE_API_KEY env var or pass api_key.")
            print("[Gemma] Using rule-based fallback analysis.")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self._model_name)
            self._available = True
            print(f"[Gemma] Initialized model: {self._model_name}")
        except ImportError:
            print("[Gemma] google-generativeai not installed. Using rule-based fallback.")
        except Exception as e:
            print(f"[Gemma] Model init failed: {e}. Using rule-based fallback.")

    def analyze_asset(self, asset_data: dict) -> GemmaAnalysis:
        """
        Analyze a road asset measurement for defects and recommendations.

        Args:
            asset_data: Dict with keys:
                - asset_id, asset_type, asset_class
                - final_ra, ai_ra, sensor_ra, retro_ra
                - status (PASS/MARGINAL/FAIL)
                - weather, ambient_lux
                - latitude, longitude
                - history (optional): list of past RA readings

        Returns:
            GemmaAnalysis with defect classification and recommendations
        """
        if self._available:
            return self._analyze_with_gemma(asset_data)
        else:
            return self._analyze_rule_based(asset_data)

    def _analyze_with_gemma(self, asset_data: dict) -> GemmaAnalysis:
        """Use Gemma 4 for intelligent analysis."""
        try:
            prompt = self._build_analysis_prompt(asset_data)
            response = self._model.generate_content(prompt)
            result = self._parse_gemma_response(response.text, asset_data)

            with self._lock:
                self._analysis_count += 1

            return result

        except Exception as e:
            print(f"[Gemma] API error: {e}. Falling back to rule-based.")
            return self._analyze_rule_based(asset_data)

    def _build_analysis_prompt(self, data: dict) -> str:
        """Build a structured prompt for Gemma 4 analysis."""
        history_text = ""
        if "history" in data and data["history"]:
            recent = data["history"][-10:]
            history_text = f"""
Historical RA readings (last {len(recent)} measurements):
{json.dumps(recent, indent=2)}
"""

        prompt = f"""You are an expert road safety engineer analyzing retroreflectivity data.

ASSET INFORMATION:
- Asset ID: {data.get('asset_id', 'Unknown')}
- Asset Type: {data.get('asset_type', 'Unknown')}
- Asset Class: {data.get('asset_class', 'Unknown')}
- Location: ({data.get('latitude', 0):.6f}, {data.get('longitude', 0):.6f})

CURRENT MEASUREMENTS:
- Final RA (EKF Fused): {data.get('final_ra', 0):.1f} mcd/lux/m²
- AI Model RA: {data.get('ai_ra', 0):.1f} mcd/lux/m²
- Physics Sensor RA: {data.get('sensor_ra', 0):.1f} mcd/lux/m²
- Retroreflectometer RA: {data.get('retro_ra', 'N/A')}
- Status: {data.get('status', 'Unknown')}
- Weather: {data.get('weather', 'clear')}
- Ambient Light: {data.get('ambient_lux', 0):.0f} lux
{history_text}

COMPLIANCE STANDARDS:
- EN 12899-1 (Signs): RA2 ≥ 150, RA1 ≥ 70 mcd/lux/m²
- EN 1436 (Markings): R2 ≥ 150, R1 ≥ 100 mcd/lux/m²
- EN 1463-1 (Studs): Type I ≥ 70, Type II ≥ 300 mcd/lux/m²

Analyze this asset and provide your response in this exact JSON format:
{{
  "defect_type": "<fading|cracking|dirt|vandalism|weathering|none>",
  "severity": "<critical|moderate|minor|none>",
  "recommendation": "<specific maintenance action>",
  "confidence": <0.0-1.0>,
  "compliance_status": "<compliant|marginal|non_compliant>",
  "narrative": "<2-3 sentence analysis explaining the condition and trend>"
}}

Respond ONLY with the JSON object, no other text."""

        return prompt

    def _parse_gemma_response(self, response_text: str, data: dict) -> GemmaAnalysis:
        """Parse Gemma's JSON response into GemmaAnalysis."""
        try:
            # Extract JSON from response
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(text)

            return GemmaAnalysis(
                defect_type=parsed.get("defect_type", "none"),
                severity=parsed.get("severity", "none"),
                recommendation=parsed.get("recommendation", "No specific recommendation"),
                confidence=float(parsed.get("confidence", 0.5)),
                compliance_status=parsed.get("compliance_status", "unknown"),
                narrative=parsed.get("narrative", "Analysis unavailable"),
                timestamp=time.time(),
            )

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Gemma] Response parse error: {e}")
            return self._analyze_rule_based(data)

    def _analyze_rule_based(self, data: dict) -> GemmaAnalysis:
        """
        Rule-based fallback analysis when Gemma API is unavailable.

        Uses RA thresholds and sensor readings to classify defects
        and generate recommendations.
        """
        final_ra = data.get("final_ra", 0)
        status = data.get("status", "PASS")
        asset_type = data.get("asset_type", "sign")
        asset_class = data.get("asset_class", "sign_RA2")
        weather = data.get("weather", "clear")

        # Defect classification based on RA levels
        if status == "FAIL":
            if final_ra < 30:
                defect_type = "fading"
                severity = "critical"
                recommendation = (f"IMMEDIATE REPLACEMENT REQUIRED. "
                                  f"RA of {final_ra:.0f} is critically below minimum "
                                  f"threshold. Schedule emergency maintenance.")
            elif final_ra < 70:
                defect_type = "weathering"
                severity = "moderate"
                recommendation = (f"Schedule replacement within 30 days. "
                                  f"RA of {final_ra:.0f} indicates significant "
                                  f"surface degradation.")
            else:
                defect_type = "dirt"
                severity = "minor"
                recommendation = (f"Cleaning may restore performance. "
                                  f"If RA remains below threshold after cleaning, "
                                  f"schedule replacement.")
        elif status == "MARGINAL":
            defect_type = "weathering"
            severity = "minor"
            recommendation = (f"Monitor closely. RA of {final_ra:.0f} is approaching "
                              f"minimum threshold. Schedule inspection within 90 days.")
        else:
            defect_type = "none"
            severity = "none"
            recommendation = "Asset is within compliance. No action needed."

        # Compliance status
        compliance = "non_compliant" if status == "FAIL" else (
            "marginal" if status == "MARGINAL" else "compliant"
        )

        # Narrative
        sensor_agreement = ""
        ai_ra = data.get("ai_ra", 0)
        retro_ra = data.get("retro_ra")
        if retro_ra:
            diff = abs(ai_ra - retro_ra)
            if diff > 50:
                sensor_agreement = (f" AI and ground truth readings differ by "
                                    f"{diff:.0f} mcd/lux/m², suggesting possible "
                                    f"calibration drift.")
            else:
                sensor_agreement = " Sensor readings show good agreement."

        narrative = (
            f"Asset {data.get('asset_id', 'Unknown')} ({asset_class}) "
            f"measured at {final_ra:.0f} mcd/lux/m² under {weather} conditions. "
            f"Status: {status}.{sensor_agreement}"
        )

        with self._lock:
            self._analysis_count += 1

        return GemmaAnalysis(
            defect_type=defect_type,
            severity=severity,
            recommendation=recommendation,
            confidence=0.75 if status != "PASS" else 0.90,
            compliance_status=compliance,
            narrative=narrative,
            timestamp=time.time(),
        )

    def generate_report(self, measurements: list) -> str:
        """
        Generate a compliance report summary for a batch of measurements.

        Args:
            measurements: List of measurement dicts

        Returns:
            Formatted report string
        """
        if not measurements:
            return "No measurements available for report generation."

        total = len(measurements)
        pass_count = sum(1 for m in measurements if m.get("status") == "PASS")
        fail_count = sum(1 for m in measurements if m.get("status") == "FAIL")
        marginal_count = total - pass_count - fail_count
        avg_ra = np.mean([m.get("final_ra", 0) for m in measurements])

        if self._available:
            return self._generate_report_gemma(measurements, total, pass_count,
                                                fail_count, marginal_count, avg_ra)
        else:
            return self._generate_report_template(total, pass_count, fail_count,
                                                   marginal_count, avg_ra)

    def _generate_report_gemma(self, measurements, total, pass_count,
                                fail_count, marginal_count, avg_ra) -> str:
        """Generate report using Gemma 4."""
        try:
            prompt = f"""Generate a professional road safety compliance report summary.

DATA:
- Total measurements: {total}
- PASS: {pass_count} ({pass_count/total*100:.1f}%)
- MARGINAL: {marginal_count} ({marginal_count/total*100:.1f}%)
- FAIL: {fail_count} ({fail_count/total*100:.1f}%)
- Average RA: {avg_ra:.1f} mcd/lux/m²

Standards: EN 12899-1, EN 1436, EN 1463-1, ASTM D4956

Write a 3-paragraph report: summary, findings, recommendations. Professional tone."""

            response = self._model.generate_content(prompt)
            return response.text

        except Exception as e:
            return self._generate_report_template(total, pass_count, fail_count,
                                                   marginal_count, avg_ra)

    def _generate_report_template(self, total, pass_count, fail_count,
                                   marginal_count, avg_ra) -> str:
        """Template-based report when Gemma is unavailable."""
        pass_rate = pass_count / max(1, total) * 100
        fail_rate = fail_count / max(1, total) * 100

        report = f"""
═══════════════════════════════════════════════════════════
  RETROFUSION AI+ PRO — COMPLIANCE REPORT
═══════════════════════════════════════════════════════════

SUMMARY
─────────────────────────────────────────────
Total assets surveyed:     {total}
Average RA:                {avg_ra:.1f} mcd/lux/m²
PASS rate:                 {pass_rate:.1f}%
FAIL rate:                 {fail_rate:.1f}%

COMPLIANCE BREAKDOWN
─────────────────────────────────────────────
PASS (EN compliant):       {pass_count} ({pass_rate:.1f}%)
MARGINAL (monitor):        {marginal_count} ({marginal_count/max(1,total)*100:.1f}%)
FAIL (action required):    {fail_count} ({fail_rate:.1f}%)

RECOMMENDATIONS
─────────────────────────────────────────────
"""
        if fail_count > 0:
            report += f"• {fail_count} assets require immediate attention\n"
            report += "• Schedule replacement for FAIL-rated assets within 30 days\n"
        if marginal_count > 0:
            report += f"• {marginal_count} assets are approaching threshold limits\n"
            report += "• Plan maintenance inspection within 90 days\n"
        if pass_rate > 90:
            report += "• Overall infrastructure retroreflectivity is satisfactory\n"

        report += f"""
─────────────────────────────────────────────
Standards: EN 12899-1 | EN 1436 | EN 1463-1
Generated by RetroFusion AI+ Pro
═══════════════════════════════════════════════════════════
"""
        return report

    @property
    def is_available(self) -> bool:
        """Whether Gemma API is available."""
        return self._available

    def get_stats(self) -> dict:
        """Return analyzer statistics."""
        with self._lock:
            return {
                "model": self._model_name,
                "api_available": self._available,
                "total_analyses": self._analysis_count,
                "mode": "gemma" if self._available else "rule_based",
            }
