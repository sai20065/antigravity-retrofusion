# dashboard/pages/6_Hardware_Live.py
# Real-Time Hardware Dashboard — Camera Feed + YOLO + GPS + Sensor Status
#
# This page connects ALL hardware/simulation modules and displays:
#   1. Live camera feed with YOLO bounding box overlays
#   2. Real-time GPS position on map
#   3. Hardware status panel (RPi GPIO, IR LEDs, sensors)
#   4. YOLO detection log with crop previews
#   5. Gemma AI defect analysis results
#   6. Full pipeline metrics (latency, FPS, throughput)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.camera_capture import CameraCapture
from modules.yolo_detector import YOLODetector, ASSET_CLASSES
from modules.ai_inference import RAEstimator
from modules.gps_module import GPSModule, GPSFix
from modules.retro_reader import RetroMeterReader
from modules.rpi_controller import RPiController
from modules.sensor_reader import BH1750Sensor
from modules.fusion_engine import ExtendedKalmanFilter, SensorMeasurement
from modules.data_logger import DataLogger
from modules.gemma_analyzer import GemmaAnalyzer
from config import DB_PATH, THRESHOLDS

st.set_page_config(page_title="Hardware Live -- RetroFusion", page_icon="📡", layout="wide")

# ── Dark Theme CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
  html, body, [data-testid="stAppViewContainer"] {
    background:#0d0f14 !important; color:#e8eaf0 !important;
    font-family:'DM Sans',sans-serif !important;
  }
  [data-testid="stSidebar"] { background:#141720 !important; border-right:1px solid #2a3045 !important; }
  #MainMenu, footer, header { visibility: hidden; }
  .section-title { font-family:'Space Mono',monospace; font-size:0.75em; text-transform:uppercase;
                   letter-spacing:0.1em; color:#8b92a8; margin-bottom:8px; }
  .metric-card { background:#141720; border:1px solid #2a3045; border-radius:8px; padding:16px;
                 margin-bottom:8px; transition:all 0.3s ease; }
  .metric-card:hover { border-color:#f5a623; box-shadow:0 0 20px rgba(245,166,35,0.1); }
  .hw-status { display:inline-flex; align-items:center; gap:6px; padding:4px 12px;
               border-radius:20px; font-family:'Space Mono',monospace; font-size:11px; }
  .hw-online { background:rgba(34,197,94,.15); color:#22c55e; border:1px solid rgba(34,197,94,.3); }
  .hw-sim { background:rgba(96,165,250,.1); color:#60a5fa; border:1px solid rgba(96,165,250,.25); }
  .hw-offline { background:rgba(239,68,68,.1); color:#ef4444; border:1px solid rgba(239,68,68,.25); }
  .detection-box { background:#1a1f2e; border:1px solid #2a3045; border-radius:6px;
                   padding:10px; margin:4px 0; font-family:'Space Mono',monospace; font-size:12px; }
  .ra-value { font-family:'Space Mono',monospace; font-size:1.4em; }
  .pulse { animation: pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE — Initialize hardware modules once
# ══════════════════════════════════════════════════════════════════════════

def init_hardware():
    """Initialize all hardware modules in session state."""
    if "hw_initialized" not in st.session_state:
        st.session_state.camera = CameraCapture(simulation=True, resolution=(640, 480), fps=10)
        st.session_state.yolo = YOLODetector(simulation=True)
        st.session_state.ai = RAEstimator(simulation=True)
        st.session_state.gps = GPSModule(simulation=True)
        st.session_state.retro = RetroMeterReader(simulate=True)
        st.session_state.rpi = RPiController(simulation=True)
        st.session_state.sensor = BH1750Sensor(simulation=True)
        st.session_state.hw_ekf = ExtendedKalmanFilter(dt=0.5)
        st.session_state.hw_logger = DataLogger(DB_PATH)
        st.session_state.gemma = GemmaAnalyzer()

        # Start background threads
        st.session_state.camera.start()
        st.session_state.gps.start()
        st.session_state.retro.start()

        st.session_state.hw_initialized = True
        st.session_state.hw_detections = []
        st.session_state.hw_pipeline_history = []
        st.session_state.hw_gemma_results = []
        st.session_state.hw_measurement_count = 0

init_hardware()


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='display:flex; align-items:center; gap:12px; margin-bottom:20px'>
  <h1 style='font-family:"Space Mono",monospace; font-size:1.2em; color:#f5a623; margin:0;'>
    HARDWARE LIVE FEED
  </h1>
  <span class='hw-sim pulse'>SIMULATION</span>
  <span style='background:rgba(245,166,35,.1); color:#f5a623; border:1px solid rgba(245,166,35,.25);
               padding:3px 10px; border-radius:20px; font-size:11px;
               font-family:"Space Mono",monospace'>YOLO + MobileNetV2 + EKF</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — Hardware Controls
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("<div style='font-family:\"Space Mono\",monospace; font-size:14px; color:#f5a623; text-align:center; padding:12px 0; font-weight:700;'>HARDWARE CONTROLS</div>", unsafe_allow_html=True)

    auto_run = st.toggle("Auto Pipeline (2s)", value=False, key="hw_auto_run")
    run_once = st.button("Run Pipeline Once", key="hw_run_once", use_container_width=True)

    st.markdown("---")
    st.markdown("<p class='section-title'>Environment</p>", unsafe_allow_html=True)
    hw_night = st.toggle("Night Mode", value=False, key="hw_night")
    hw_rain = st.toggle("Rain", value=False, key="hw_rain")
    hw_fog = st.toggle("Fog", value=False, key="hw_fog")

    st.markdown("---")
    st.markdown("<p class='section-title'>IR LED Control</p>", unsafe_allow_html=True)
    ir_manual = st.slider("IR Duty Cycle (%)", 0, 100, 0, key="hw_ir_duty")
    if ir_manual > 0:
        st.session_state.rpi.set_ir_manual(ir_manual)

    st.markdown("---")
    st.markdown("<p class='section-title'>Ground Truth</p>", unsafe_allow_html=True)
    sim_ra = st.slider("Retro True RA", 20, 800, 250, step=10, key="hw_sim_ra")
    st.session_state.retro.set_sim_ra(sim_ra)

    st.markdown("---")
    st.markdown("<p class='section-title'>Pipeline Stats</p>", unsafe_allow_html=True)
    st.caption(f"Measurements: {st.session_state.hw_measurement_count}")
    cam_stats = st.session_state.camera.get_stats()
    st.caption(f"Camera FPS: {cam_stats['actual_fps']}")
    if st.session_state.hw_detections:
        st.caption(f"Last detections: {len(st.session_state.hw_detections)}")


# ══════════════════════════════════════════════════════════════════════════
# RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_full_pipeline():
    """Execute the complete hardware pipeline once."""
    t_start = time.time()

    cam = st.session_state.camera
    yolo = st.session_state.yolo
    ai = st.session_state.ai
    gps = st.session_state.gps
    retro = st.session_state.retro
    rpi = st.session_state.rpi
    sensor = st.session_state.sensor
    ekf = st.session_state.hw_ekf
    logger = st.session_state.hw_logger
    gemma = st.session_state.gemma

    # 1. Camera frame
    frame = cam.get_frame()
    if frame is None:
        return None

    # 2. Ambient light
    lux_result = sensor.read_lux()
    ambient_lux = lux_result if isinstance(lux_result, (int, float)) else lux_result.get("lux", 100)

    # 3. IR brightness
    rpi.set_ir_brightness(ambient_lux)

    # 4. YOLO detection
    detections = yolo.detect(frame)
    st.session_state.hw_detections = detections

    # 5. For each detection: AI RA + Sensor RA + Retro RA + Fusion
    results = []
    for det in detections:
        # Extract crop
        crop = det.get_crop(frame)

        # AI inference
        night = st.session_state.get("hw_night", False)
        rain = st.session_state.get("hw_rain", False)
        fog = st.session_state.get("hw_fog", False)

        ai_result = ai.predict(
            image=None,  # simulation mode
            ra_true=sim_ra,
            night=night, rain=rain, fog=fog,
        )

        # Sensor RA
        sensor_result = sensor.compute_ra(
            reflected_lux=0, ambient_lux=ambient_lux,
            ra_true=sim_ra, rain=rain, fog=fog,
        )

        # Retro RA
        retro_ra = retro.get_ra()

        # Weather
        weather = "rain" if rain else ("fog" if fog else "clear")

        # EKF Fusion
        meas = SensorMeasurement(
            ai_ra=ai_result["ra_estimate"],
            ai_confidence=ai_result["confidence"],
            sensor_ra=sensor_result["ra_estimate"],
            sensor_snr=sensor_result["snr"],
            retro_ra=retro_ra,
            retro_available=retro.available,
            weather=weather,
        )
        fusion = ekf.update(meas)

        # Status
        threshold_key = getattr(det, 'asset_class', 'sign_RA2')
        thresholds = THRESHOLDS.get(threshold_key, {"pass": 100, "marginal": 70})
        if fusion.final_ra >= thresholds["pass"]:
            status = "PASS"
        elif fusion.final_ra >= thresholds["marginal"]:
            status = "MARGINAL"
        else:
            status = "FAIL"

        # LED alert
        rpi.alert_on_fail(status)

        # GPS
        gps_fix = gps.get_fix()

        # Gemma analysis (for FAIL/MARGINAL only)
        gemma_result = None
        if status in ("FAIL", "MARGINAL"):
            gemma_result = gemma.analyze_asset({
                "asset_id": f"DET-{st.session_state.hw_measurement_count}",
                "asset_type": det.class_name,
                "asset_class": threshold_key,
                "final_ra": fusion.final_ra,
                "ai_ra": ai_result["ra_estimate"],
                "sensor_ra": sensor_result["ra_estimate"],
                "retro_ra": retro_ra,
                "status": status,
                "weather": weather,
                "ambient_lux": ambient_lux,
                "latitude": gps_fix.latitude if gps_fix else 12.9716,
                "longitude": gps_fix.longitude if gps_fix else 77.5946,
            })
            st.session_state.hw_gemma_results.append({
                "time": time.strftime("%H:%M:%S"),
                "asset": det.class_name,
                "defect": gemma_result.defect_type,
                "severity": gemma_result.severity,
                "recommendation": gemma_result.recommendation[:100],
            })
            if len(st.session_state.hw_gemma_results) > 20:
                st.session_state.hw_gemma_results = st.session_state.hw_gemma_results[-20:]

        # Log
        sim_raw = {
            "asset": {"id": f"DET-{st.session_state.hw_measurement_count}",
                      "name": det.class_name, "class": threshold_key,
                      "ra_true": sim_ra, "type": det.class_name,
                      "lat": gps_fix.latitude if gps_fix else 12.9716,
                      "lon": gps_fix.longitude if gps_fix else 77.5946},
            "timestamp": time.time(),
            "lat": gps_fix.latitude if gps_fix else 12.9716,
            "lon": gps_fix.longitude if gps_fix else 77.5946,
            "speed_kmh": gps_fix.speed_kmh if gps_fix else 0,
            "ai_ra": ai_result["ra_estimate"],
            "ai_confidence": ai_result["confidence"],
            "sensor_ra": sensor_result["ra_estimate"],
            "sensor_snr": sensor_result["snr"],
            "retro_ra": retro_ra,
            "ambient_lux": ambient_lux,
            "weather": weather,
        }
        logger.log(sim_raw, fusion)

        results.append({
            "detection": det,
            "ai_ra": ai_result["ra_estimate"],
            "ai_conf": ai_result["confidence"],
            "sensor_ra": sensor_result["ra_estimate"],
            "retro_ra": retro_ra,
            "final_ra": fusion.final_ra,
            "variance": fusion.ekf_variance,
            "status": status,
            "alpha": fusion.alpha,
            "beta": fusion.beta,
            "gamma": fusion.gamma,
            "gemma": gemma_result,
        })

        st.session_state.hw_measurement_count += 1

    # Pipeline timing
    t_total = (time.time() - t_start) * 1000

    # Store pipeline history
    st.session_state.hw_pipeline_history.append({
        "time": time.strftime("%H:%M:%S"),
        "detections": len(detections),
        "latency_ms": round(t_total, 1),
        "final_ra": results[0]["final_ra"] if results else 0,
    })
    if len(st.session_state.hw_pipeline_history) > 100:
        st.session_state.hw_pipeline_history = st.session_state.hw_pipeline_history[-100:]

    return {
        "frame": frame,
        "detections": detections,
        "results": results,
        "gps_fix": gps.get_fix(),
        "rpi_state": rpi.get_state(),
        "latency_ms": round(t_total, 1),
        "ambient_lux": ambient_lux,
    }


# ── Execute pipeline ──────────────────────────────────────────────────────
pipeline_output = None
if run_once or auto_run:
    pipeline_output = run_full_pipeline()


# ══════════════════════════════════════════════════════════════════════════
# 1. HARDWARE STATUS BAR
# ══════════════════════════════════════════════════════════════════════════

st.markdown("<p class='section-title'>System Status</p>", unsafe_allow_html=True)

rpi_state = st.session_state.rpi.get_state()
gps_fix = st.session_state.gps.get_fix()

c1, c2, c3, c4, c5, c6 = st.columns(6)

modules_info = [
    (c1, "Camera", st.session_state.camera.simulation, True),
    (c2, "YOLO", st.session_state.yolo.simulation, True),
    (c3, "GPS", st.session_state.gps.simulation, gps_fix is not None),
    (c4, "Retro", st.session_state.retro.simulate, st.session_state.retro.available),
    (c5, "IR LED", rpi_state["simulation"], rpi_state["ir_enabled"]),
    (c6, "Gemma", not st.session_state.gemma.is_available, True),
]

for col, name, is_sim, is_active in modules_info:
    with col:
        if is_sim:
            badge_style = "background:rgba(96,165,250,.1); color:#60a5fa; border:1px solid rgba(96,165,250,.25);"
            badge_text = "SIM"
        elif is_active:
            badge_style = "background:rgba(34,197,94,.15); color:#22c55e; border:1px solid rgba(34,197,94,.3);"
            badge_text = "LIVE"
        else:
            badge_style = "background:rgba(239,68,68,.1); color:#ef4444; border:1px solid rgba(239,68,68,.25);"
            badge_text = "OFF"

        extra = ""
        if name == "IR LED":
            extra = f"Duty: {rpi_state['ir_duty_cycle']}%"
        elif name == "GPS" and gps_fix:
            extra = f"{gps_fix.latitude:.4f}N"

        st.markdown(f"<div style='background:#141720; border:1px solid #2a3045; border-radius:8px; padding:16px; text-align:center;'>"
                    f"<div style='font-family:Space Mono,monospace; font-size:0.75em; text-transform:uppercase; letter-spacing:0.1em; color:#8b92a8; margin-bottom:8px;'>{name}</div>"
                    f"<span style='{badge_style} padding:4px 12px; border-radius:20px; font-family:Space Mono,monospace; font-size:11px;'>{badge_text}</span>"
                    f"<div style='font-size:10px; color:#8b92a8; margin-top:6px;'>{extra}</div>"
                    f"</div>", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# 2. CAMERA FEED + DETECTION RESULTS (side by side)
# ══════════════════════════════════════════════════════════════════════════

if pipeline_output:
    cam_col, det_col = st.columns([3, 2])

    with cam_col:
        st.markdown("<p class='section-title'>Camera Feed + YOLO Detections</p>", unsafe_allow_html=True)
        frame = pipeline_output["frame"]
        detections = pipeline_output["detections"]

        # Draw bounding boxes on frame
        if detections:
            annotated = st.session_state.yolo.draw_detections(frame, detections)
        else:
            annotated = frame

        # Convert BGR to RGB for Streamlit
        import cv2
        display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(display_frame, caption=f"Frame | {len(detections)} detections | {pipeline_output['latency_ms']}ms",
                 use_column_width=True)

    with det_col:
        st.markdown("<p class='section-title'>Detection Results</p>", unsafe_allow_html=True)

        for r in pipeline_output["results"]:
            det = r["detection"]
            status_color = {"PASS": "#22c55e", "MARGINAL": "#f5a623", "FAIL": "#ef4444"}[r["status"]]

            st.markdown(f"""<div class='detection-box'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#60a5fa;'>{det.class_name}</span>
                    <span style='color:{status_color}; font-weight:700;'>{r["status"]}</span>
                </div>
                <div style='margin-top:6px; display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:11px;'>
                    <span style='color:#818cf8;'>AI: {r["ai_ra"]:.0f}</span>
                    <span style='color:#34d399;'>Sensor: {r["sensor_ra"]:.0f}</span>
                    <span style='color:#f472b6;'>Retro: {(f"{r['retro_ra']:.0f}" if r['retro_ra'] else "N/A")}</span>
                    <span style='color:#f5a623; font-weight:700;'>Fused: {r["final_ra"]:.0f}</span>
                </div>
                <div style='margin-top:4px; font-size:10px; color:#8b92a8;'>
                    conf={r["ai_conf"]:.2f} | var={r["variance"]:.1f} | a={r["alpha"]:.2f} b={r["beta"]:.2f} g={r["gamma"]:.2f}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # 3. GPS MAP + PIPELINE METRICS
    # ══════════════════════════════════════════════════════════════════════

    map_col, metrics_col = st.columns([3, 2])

    with map_col:
        st.markdown("<p class='section-title'>GPS Position (Live)</p>", unsafe_allow_html=True)
        gps_fix = pipeline_output["gps_fix"]
        if gps_fix:
            import folium
            from streamlit_folium import folium_static

            lat = float(gps_fix.latitude)
            lon = float(gps_fix.longitude)

            m = folium.Map(location=[lat, lon],
                          zoom_start=16, tiles="CartoDB dark_matter",
                          width=500, height=300)

            # Current position marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=8, color="#f5a623", fill=True, fill_color="#f5a623",
                fill_opacity=0.8, popup=f"Current: {lat:.6f}, {lon:.6f}",
            ).add_to(m)

            # Detection markers with status colors
            for r in pipeline_output["results"]:
                color = {"PASS": "#22c55e", "MARGINAL": "#f5a623", "FAIL": "#ef4444"}[r["status"]]
                det_lat = float(lat + np.random.normal(0, 0.0002))
                det_lon = float(lon + np.random.normal(0, 0.0002))
                folium.CircleMarker(
                    location=[det_lat, det_lon],
                    radius=5, color=color, fill=True, fill_color=color,
                    popup=f"{r['detection'].class_name}: RA={r['final_ra']:.0f} ({r['status']})",
                ).add_to(m)

            folium_static(m, width=700, height=300)
        else:
            st.info("Waiting for GPS fix...")

    with metrics_col:
        st.markdown("<p class='section-title'>Pipeline Metrics</p>", unsafe_allow_html=True)

        # Pipeline latency chart
        hist = st.session_state.hw_pipeline_history
        if len(hist) > 1:
            hist_df = pd.DataFrame(hist[-30:])
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Bar(
                x=list(range(len(hist_df))),
                y=hist_df["latency_ms"],
                marker_color=["#22c55e" if l < 150 else "#f5a623" if l < 300 else "#ef4444"
                              for l in hist_df["latency_ms"]],
            ))
            fig_lat.update_layout(
                template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                height=130, margin=dict(l=10, r=10, t=5, b=25),
                xaxis=dict(showticklabels=False),
                yaxis=dict(title="ms", tickfont=dict(family="Space Mono", size=9, color="#8b92a8"),
                           showgrid=True, gridcolor="#1e2538"),
                showlegend=False,
            )
            st.plotly_chart(fig_lat, use_container_width=True, key="hw_latency_chart")

        # Summary metrics
        st.markdown(f"""<div class='metric-card'>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px; font-family:\"Space Mono\",monospace; font-size:12px;'>
                <div><span style='color:#8b92a8;'>Total:</span> <span style='color:#60a5fa;'>{st.session_state.hw_measurement_count}</span></div>
                <div><span style='color:#8b92a8;'>Latency:</span> <span style='color:#f5a623;'>{pipeline_output["latency_ms"]}ms</span></div>
                <div><span style='color:#8b92a8;'>Detections:</span> <span style='color:#818cf8;'>{len(pipeline_output["detections"])}</span></div>
                <div><span style='color:#8b92a8;'>Ambient:</span> <span style='color:#34d399;'>{pipeline_output["ambient_lux"]:.0f} lux</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # 4. GEMMA AI ANALYSIS LOG
    # ══════════════════════════════════════════════════════════════════════

    if st.session_state.hw_gemma_results:
        st.markdown("<p class='section-title'>Gemma AI Defect Analysis</p>", unsafe_allow_html=True)
        gemma_df = pd.DataFrame(st.session_state.hw_gemma_results)
        st.dataframe(gemma_df, use_container_width=True, hide_index=True)

else:
    # No pipeline data yet — show instructions
    st.markdown("""
    <div class='metric-card' style='text-align:center; padding:40px;'>
        <div style='font-size:3em; margin-bottom:12px;'>📡</div>
        <div style='font-family:"Space Mono",monospace; font-size:1.1em; color:#f5a623; margin-bottom:8px;'>
            Ready to Start Pipeline
        </div>
        <div style='color:#8b92a8; font-size:13px; max-width:500px; margin:0 auto;'>
            Click <b>"Run Pipeline Once"</b> in the sidebar to execute the full detection pipeline,
            or enable <b>"Auto Pipeline"</b> for continuous real-time data.
            <br><br>
            Pipeline: Camera → YOLOv8 → Crop → MobileNetV2 → EKF Fusion → Gemma Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show camera preview even without pipeline
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>Camera Preview</p>", unsafe_allow_html=True)

    frame = st.session_state.camera.get_frame()
    if frame is not None:
        import cv2
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(display, caption="Live camera feed (no detection running)", use_column_width=True)
    else:
        st.info("Camera warming up...")


# ══════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH
# ══════════════════════════════════════════════════════════════════════════

if auto_run:
    time.sleep(2)
    st.rerun()
