# dashboard/app.py
# RetroFusion AI+ Pro — Streamlit Dashboard
# Run: streamlit run dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.simulator import StreamingSimulator, ROAD_ASSETS, simulate_measurement
from modules.fusion_engine import ExtendedKalmanFilter, SensorMeasurement
from modules.data_logger import DataLogger
from modules.predictive import predict_failure, ra_decay_model
from modules.camera_capture import CameraCapture
from modules.yolo_detector import YOLODetector, ASSET_CLASSES
from modules.ai_inference import RAEstimator
from modules.gps_module import GPSModule
from modules.retro_reader import RetroMeterReader
from modules.rpi_controller import RPiController
from modules.gemma_analyzer import GemmaAnalyzer
from config import THRESHOLDS, DB_PATH, WEATHER_MULTIPLIERS


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetroFusion AI+ Pro",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom dark CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark industrial theme */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14 !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  [data-testid="stSidebar"] {
    background: #141720 !important;
    border-right: 1px solid #2a3045 !important;
  }
  .metric-card {
    background: #141720;
    border: 1px solid #2a3045;
    border-radius: 8px;
    padding: 16px;
    position: relative;
    margin-bottom: 8px;
  }
  .status-pass   { color: #22c55e; font-family: 'Space Mono', monospace; font-weight: 700; }
  .status-fail   { color: #ef4444; font-family: 'Space Mono', monospace; font-weight: 700; }
  .status-marg   { color: #f5a623; font-family: 'Space Mono', monospace; font-weight: 700; }
  .ra-value      { font-family: 'Space Mono', monospace; font-size: 1.4em; }
  .section-title { font-family: 'Space Mono', monospace; font-size: 0.75em;
                   text-transform: uppercase; letter-spacing: 0.1em;
                   color: #8b92a8; margin-bottom: 8px; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* Custom metric styling */
  [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: #e8eaf0 !important;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7em !important;
    color: #8b92a8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
  }

  /* Plotly chart background */
  .js-plotly-plot .plotly .bg { fill: #141720 !important; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background-color: #141720;
    border-radius: 8px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #8b92a8;
    border-radius: 6px;
  }
  .stTabs [aria-selected="true"] {
    background-color: #1e2538 !important;
    color: #f5a623 !important;
  }

  /* Dataframe styling */
  .stDataFrame { border-radius: 8px; overflow: hidden; }

  /* Smooth animations */
  .metric-card, .ra-value {
    transition: all 0.3s ease;
  }
  .metric-card:hover {
    border-color: #f5a623;
    box-shadow: 0 0 20px rgba(245,166,35,0.1);
  }
</style>
""", unsafe_allow_html=True)


# ── Initialize session state ───────────────────────────────────────────────
if "ekf"       not in st.session_state:
    st.session_state.ekf      = ExtendedKalmanFilter(dt=0.5)
if "logger"    not in st.session_state:
    st.session_state.logger   = DataLogger(DB_PATH)
if "simulator" not in st.session_state:
    st.session_state.simulator = StreamingSimulator(interval=0.5)
    st.session_state.simulator.start()
if "history"   not in st.session_state:
    st.session_state.history  = []
if "fusion_history" not in st.session_state:
    st.session_state.fusion_history = []

# ── Initialize hardware modules (all start in simulation mode) ─────────────
if "hw_camera" not in st.session_state:
    st.session_state.hw_camera = CameraCapture(simulation=True, resolution=(640, 480), fps=10)
    st.session_state.hw_camera.start()
if "hw_yolo" not in st.session_state:
    st.session_state.hw_yolo = YOLODetector(simulation=True)
if "hw_ai" not in st.session_state:
    st.session_state.hw_ai = RAEstimator(simulation=True)
if "hw_gps" not in st.session_state:
    st.session_state.hw_gps = GPSModule(simulation=True)
    st.session_state.hw_gps.start()
if "hw_retro" not in st.session_state:
    st.session_state.hw_retro = RetroMeterReader(simulate=True)
    st.session_state.hw_retro.start()
if "hw_rpi" not in st.session_state:
    st.session_state.hw_rpi = RPiController(simulation=True)
if "hw_gemma" not in st.session_state:
    st.session_state.hw_gemma = GemmaAnalyzer()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0'>
      <div style='background: linear-gradient(135deg, #f5a623, #e8890c);
                  border-radius:12px; padding:12px;
                  font-family:"Space Mono",monospace; font-size:22px;
                  font-weight:700; color:#000; letter-spacing:-1px;
                  box-shadow: 0 4px 15px rgba(245,166,35,0.3);'>RF</div>
      <p style='font-family:"Space Mono",monospace; font-size:11px;
                color:#8b92a8; margin-top:10px; letter-spacing:0.1em'>
        RETROFUSION AI+ PRO
      </p>
      <p style='font-size:10px; color:#4a5068; margin-top:-8px;'>
        v1.0.0 -- Hybrid RA Measurement
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Hardware status badges
    gps_fix = st.session_state.hw_gps.get_fix()
    rpi_state = st.session_state.hw_rpi.get_state()
    st.markdown(f"""
    <div style='display:flex; flex-wrap:wrap; gap:4px; justify-content:center; margin-bottom:8px;'>
      <span style='background:rgba(96,165,250,.1); color:#60a5fa; border:1px solid rgba(96,165,250,.25);
                   padding:2px 8px; border-radius:12px; font-size:9px; font-family:"Space Mono",monospace;'>
        CAM {'ON' if not st.session_state.hw_camera.simulation else 'SIM'}</span>
      <span style='background:rgba(129,140,248,.1); color:#818cf8; border:1px solid rgba(129,140,248,.25);
                   padding:2px 8px; border-radius:12px; font-size:9px; font-family:"Space Mono",monospace;'>
        YOLO {'ON' if not st.session_state.hw_yolo.simulation else 'SIM'}</span>
      <span style='background:rgba(52,211,153,.1); color:#34d399; border:1px solid rgba(52,211,153,.25);
                   padding:2px 8px; border-radius:12px; font-size:9px; font-family:"Space Mono",monospace;'>
        GPS {'FIX' if gps_fix else 'WAIT'}</span>
      <span style='background:rgba(244,114,182,.1); color:#f472b6; border:1px solid rgba(244,114,182,.25);
                   padding:2px 8px; border-radius:12px; font-size:9px; font-family:"Space Mono",monospace;'>
        IR {rpi_state['ir_duty_cycle']}%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    weather = st.selectbox("🌤 Weather Condition", ["clear", "rain", "fog", "snow"],
                           key="weather_select")
    night_mode = st.toggle("🌙 Night Mode", value=False, key="night_toggle")
    retro_connected = st.toggle("🔬 RetroMeter Connected", value=True, key="retro_toggle")
    auto_refresh = st.toggle("🔄 Auto Refresh (2s)", value=False, key="auto_refresh")

    st.markdown("---")
    st.markdown("<p class='section-title'>Fusion Weights</p>", unsafe_allow_html=True)

    # Compute current weights dynamically
    w_mult = WEATHER_MULTIPLIERS.get(weather, WEATHER_MULTIPLIERS["clear"])
    if retro_connected:
        st.markdown("**α** RetroMeter: `0.65`")
        st.markdown("**β** AI Model:   `0.22`")
        st.markdown("**γ** Sensor:     `0.13`")
    else:
        st.markdown("**α** RetroMeter: `0.00` ⚠️")
        st.markdown("**β** AI Model:   `0.62`")
        st.markdown("**γ** Sensor:     `0.38`")

    st.markdown("---")
    st.markdown("<p class='section-title'>Weather Noise</p>", unsafe_allow_html=True)
    st.caption(f"AI: ×{w_mult['ai']}  |  Sensor: ×{w_mult['sensor']}  |  Retro: ×{w_mult['retro']}")

    st.markdown("---")
    st.markdown("<p class='section-title'>Standards</p>", unsafe_allow_html=True)
    st.caption("EN 12899-1 | EN 1436 | EN 1463-1")
    st.caption("ASTM D4956 | IRC:SP:39")

    st.markdown("---")
    st.markdown("<p class='section-title'>Export</p>", unsafe_allow_html=True)
    df_export = st.session_state.logger.query_df(limit=500)
    if not df_export.empty:
        csv = df_export.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "retrofusion_measurements.csv",
                           "text/csv", key="csv_download")


# ── Main content ───────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; gap:12px; margin-bottom:20px'>
  <h1 style='font-family:"Space Mono",monospace; font-size:1.2em;
             color:#f5a623; margin:0; letter-spacing:-0.5px'>
    RETROFUSION AI+ PRO
  </h1>
  <span style='background:rgba(34,197,94,.15); color:#22c55e;
               border:1px solid rgba(34,197,94,.3); padding:3px 10px;
               border-radius:20px; font-size:11px;
               font-family:"Space Mono",monospace'>● LIVE</span>
  <span style='background:rgba(245,166,35,.1); color:#f5a623;
               border:1px solid rgba(245,166,35,.25); padding:3px 10px;
               border-radius:20px; font-size:11px;
               font-family:"Space Mono",monospace'>FUSION MODE</span>
</div>
""", unsafe_allow_html=True)


# ── Run one fusion step ────────────────────────────────────────────────────
def run_fusion_step(weather_cond, night, retro_on):
    """Get latest simulated measurement and run EKF fusion."""
    import random
    asset = random.choice(ROAD_ASSETS)
    raw   = simulate_measurement(asset, night=night,
                                  rain=(weather_cond=="rain"),
                                  fog=(weather_cond=="fog"))
    meas  = SensorMeasurement(
        ai_ra          = raw["ai_ra"],
        ai_confidence  = raw["ai_confidence"],
        sensor_ra      = raw["sensor_ra"],
        sensor_snr     = raw["sensor_snr"],
        retro_ra       = raw.get("retro_ra") if retro_on else None,
        retro_available= retro_on,
        weather        = weather_cond,
    )
    ekf    = st.session_state.ekf
    result = ekf.update(meas)
    st.session_state.logger.log(raw, result)

    # Track fusion history for charts
    st.session_state.fusion_history.append({
        "timestamp": raw["timestamp"],
        "final_ra": result.final_ra,
        "ai_ra": raw["ai_ra"],
        "sensor_ra": raw["sensor_ra"],
        "retro_ra": raw.get("retro_ra", 0),
        "variance": result.ekf_variance,
        "alpha": result.alpha,
        "beta": result.beta,
        "gamma": result.gamma,
        "bias": result.bias_estimate,
        "asset_id": asset["id"],
        "asset_name": asset["name"],
        "asset_class": asset["class"],
        "weather": weather_cond,
    })
    if len(st.session_state.fusion_history) > 200:
        st.session_state.fusion_history = st.session_state.fusion_history[-200:]

    return raw, result


def get_status(asset_class: str, final_ra: float) -> str:
    thresholds = THRESHOLDS.get(asset_class, {"pass": 100, "marginal": 70})
    if final_ra >= thresholds["pass"]:
        return "PASS"
    elif final_ra >= thresholds["marginal"]:
        return "MARGINAL"
    return "FAIL"


# ── Run fusion step ───────────────────────────────────────────────────────
raw, result = run_fusion_step(weather, night_mode, retro_connected)
status = get_status(raw["asset"]["class"], result.final_ra)

# ═══════════════════════════════════════════════════════════════════════════
# 1. KPI ROW
# ═══════════════════════════════════════════════════════════════════════════
stats = st.session_state.logger.get_stats()
rmse = st.session_state.logger.get_rmse_ai_vs_retro()

from dashboard.components.kpi_cards import render_kpi_row
render_kpi_row(stats, rmse)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 2. LIVE MEASUREMENT PANEL
# ═══════════════════════════════════════════════════════════════════════════
from dashboard.components.fusion_panel import render_live_measurement, render_weight_bars
render_live_measurement(raw, result, status)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3 & 4. RA TIMELINE + SENSOR COMPARISON (side by side)
# ═══════════════════════════════════════════════════════════════════════════
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("<p class='section-title'>📈 RA Timeline (Last 50)</p>", unsafe_allow_html=True)
    fh = st.session_state.fusion_history
    if len(fh) > 1:
        fh_df = pd.DataFrame(fh[-50:])
        fh_df["idx"] = range(len(fh_df))

        fig_timeline = go.Figure()

        # Final RA line
        fig_timeline.add_trace(go.Scatter(
            x=fh_df["idx"], y=fh_df["final_ra"],
            mode='lines+markers',
            name='EKF Fused RA',
            line=dict(color="#f5a623", width=2),
            marker=dict(
                size=6,
                color=[
                    "#22c55e" if ra > 150 else ("#f5a623" if ra > 100 else "#ef4444")
                    for ra in fh_df["final_ra"]
                ],
                line=dict(width=1, color="#0d0f14"),
            ),
        ))

        # AI RA (faded)
        fig_timeline.add_trace(go.Scatter(
            x=fh_df["idx"], y=fh_df["ai_ra"],
            mode='lines',
            name='AI RA',
            line=dict(color="#818cf8", width=1, dash='dot'),
            opacity=0.5,
        ))

        # Threshold lines
        fig_timeline.add_hline(y=150, line_dash="dash", line_color="#22c55e",
                               annotation_text="PASS (150)", annotation_position="right",
                               annotation_font=dict(color="#22c55e", size=10, family="Space Mono"))
        fig_timeline.add_hline(y=100, line_dash="dash", line_color="#f5a623",
                               annotation_text="MARGINAL (100)", annotation_position="right",
                               annotation_font=dict(color="#f5a623", size=10, family="Space Mono"))

        fig_timeline.update_layout(
            template="plotly_dark",
            paper_bgcolor="#141720",
            plot_bgcolor="#141720",
            height=350,
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(
                title="Measurement #",
                showgrid=True, gridcolor="#1e2538",
                tickfont=dict(family="Space Mono", size=10, color="#8b92a8"),
            ),
            yaxis=dict(
                title="RA (mcd/lux/m²)",
                showgrid=True, gridcolor="#1e2538",
                tickfont=dict(family="Space Mono", size=10, color="#8b92a8"),
            ),
            legend=dict(
                font=dict(family="DM Sans", size=10, color="#8b92a8"),
                bgcolor="rgba(0,0,0,0)",
                orientation="h", yanchor="bottom", y=1.02,
            ),
            showlegend=True,
        )
        st.plotly_chart(fig_timeline, use_container_width=True, key="ra_timeline")
    else:
        st.info("Collecting data... refresh to see timeline.")

with chart_col2:
    st.markdown("<p class='section-title'>🔬 AI vs RetroMeter</p>", unsafe_allow_html=True)
    df_scatter = st.session_state.logger.query_df(limit=200)
    if not df_scatter.empty and "retro_ra" in df_scatter.columns:
        scatter_df = df_scatter.dropna(subset=["ai_ra", "retro_ra"])
        if len(scatter_df) > 2:
            # Compute R² and RMSE
            from scipy.stats import pearsonr as pr
            corr, _ = pr(scatter_df["ai_ra"], scatter_df["retro_ra"])
            r2 = corr ** 2
            rmse_val = np.sqrt(((scatter_df["ai_ra"] - scatter_df["retro_ra"]) ** 2).mean())

            fig_scatter = go.Figure()
            colors_map = {"PASS": "#22c55e", "MARGINAL": "#f5a623", "FAIL": "#ef4444"}

            for status_val, color in colors_map.items():
                mask = scatter_df["status"] == status_val
                if mask.any():
                    fig_scatter.add_trace(go.Scatter(
                        x=scatter_df.loc[mask, "retro_ra"],
                        y=scatter_df.loc[mask, "ai_ra"],
                        mode='markers',
                        name=status_val,
                        marker=dict(color=color, size=6, opacity=0.7,
                                    line=dict(width=1, color="#0d0f14")),
                    ))

            # Perfect correlation line
            ra_range = [0, max(scatter_df["retro_ra"].max(), scatter_df["ai_ra"].max()) * 1.1]
            fig_scatter.add_trace(go.Scatter(
                x=ra_range, y=ra_range,
                mode='lines', name='Perfect (1:1)',
                line=dict(color="#8b92a8", width=1, dash='dash'),
            ))

            # R² annotation
            fig_scatter.add_annotation(
                x=0.02, y=0.98, xref="paper", yref="paper",
                text=f"R² = {r2:.3f}<br>RMSE = {rmse_val:.1f}",
                showarrow=False,
                font=dict(family="Space Mono", size=12, color="#f5a623"),
                align="left",
                bgcolor="rgba(20,23,32,0.8)",
                bordercolor="#2a3045",
                borderwidth=1,
                borderpad=8,
            )

            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor="#141720",
                plot_bgcolor="#141720",
                height=350,
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis=dict(
                    title="RetroMeter RA",
                    showgrid=True, gridcolor="#1e2538",
                    tickfont=dict(family="Space Mono", size=10, color="#8b92a8"),
                ),
                yaxis=dict(
                    title="AI Estimated RA",
                    showgrid=True, gridcolor="#1e2538",
                    tickfont=dict(family="Space Mono", size=10, color="#8b92a8"),
                ),
                legend=dict(
                    font=dict(family="DM Sans", size=10, color="#8b92a8"),
                    bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.02,
                ),
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key="ai_vs_retro")
        else:
            st.info("Need more data points for scatter plot...")
    else:
        st.info("No retro data available yet.")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5. FUSION WEIGHTS + EKF STATE
# ═══════════════════════════════════════════════════════════════════════════
weight_col, ekf_col = st.columns(2)

with weight_col:
    st.markdown("<p class='section-title'>⚖️ Live Fusion Weights</p>", unsafe_allow_html=True)
    render_weight_bars(result)

with ekf_col:
    st.markdown("<p class='section-title'>🧮 EKF State</p>", unsafe_allow_html=True)
    from dashboard.components.fusion_panel import render_covariance_heatmap
    P = st.session_state.ekf.P
    render_covariance_heatmap(P.tolist())

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 6. RECENT MEASUREMENTS TABLE
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("<p class='section-title'>📋 Recent Measurements</p>", unsafe_allow_html=True)
df_table = st.session_state.logger.query_df(limit=20)
if not df_table.empty:
    display_cols = ["datetime", "asset_id", "asset_type", "asset_class",
                    "ai_ra", "sensor_ra", "retro_ra", "final_ra",
                    "weather_code", "status"]
    available_cols = [c for c in display_cols if c in df_table.columns]
    st.dataframe(
        df_table[available_cols].head(15),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No measurements yet. Data will appear as fusion runs.")


# ── Auto-refresh ──────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(2)
    st.rerun()
