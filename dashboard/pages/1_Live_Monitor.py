# dashboard/pages/1_Live_Monitor.py
# Real-time EKF Fusion Monitor with auto-refresh

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.simulator import ROAD_ASSETS, simulate_measurement
from modules.fusion_engine import ExtendedKalmanFilter, SensorMeasurement
from modules.data_logger import DataLogger
from config import THRESHOLDS, DB_PATH, WEATHER_MULTIPLIERS

st.set_page_config(page_title="Live Monitor — RetroFusion", page_icon="📡", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
  html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14 !important; color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  [data-testid="stSidebar"] { background: #141720 !important; border-right: 1px solid #2a3045 !important; }
  #MainMenu, footer, header { visibility: hidden; }
  .metric-card { background:#141720; border:1px solid #2a3045; border-radius:8px; padding:16px; margin-bottom:8px; transition: all 0.3s ease; }
  .metric-card:hover { border-color:#f5a623; box-shadow:0 0 20px rgba(245,166,35,0.1); }
  .ra-value { font-family:'Space Mono',monospace; font-size:1.4em; }
  .section-title { font-family:'Space Mono',monospace; font-size:0.75em; text-transform:uppercase; letter-spacing:0.1em; color:#8b92a8; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
if "live_ekf" not in st.session_state:
    st.session_state.live_ekf = ExtendedKalmanFilter(dt=0.5)
if "live_logger" not in st.session_state:
    st.session_state.live_logger = DataLogger(DB_PATH)
if "live_readings" not in st.session_state:
    st.session_state.live_readings = []

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0'>
        <div style='font-family:"Space Mono",monospace; font-size:14px;
                    color:#f5a623; font-weight:700;'>📡 LIVE MONITOR</div>
    </div>
    """, unsafe_allow_html=True)
    weather = st.selectbox("Weather", ["clear", "rain", "fog", "snow"], key="lm_weather")
    night = st.toggle("Night Mode", key="lm_night")
    retro = st.toggle("RetroMeter", value=True, key="lm_retro")
    speed = st.slider("Refresh Rate (sec)", 1, 10, 2, key="lm_speed")
    auto = st.toggle("Auto Refresh", value=True, key="lm_auto")

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; gap:12px; margin-bottom:20px'>
  <h1 style='font-family:"Space Mono",monospace; font-size:1.2em; color:#f5a623; margin:0;'>
    LIVE MONITOR
  </h1>
  <span style='background:rgba(34,197,94,.15); color:#22c55e; border:1px solid rgba(34,197,94,.3);
               padding:3px 10px; border-radius:20px; font-size:11px;
               font-family:"Space Mono",monospace'>● STREAMING</span>
</div>
""", unsafe_allow_html=True)

# ── Run fusion ─────────────────────────────────────────────────────────────
asset = random.choice(ROAD_ASSETS)
raw = simulate_measurement(asset, night=night, rain=(weather=="rain"), fog=(weather=="fog"))
meas = SensorMeasurement(
    ai_ra=raw["ai_ra"], ai_confidence=raw["ai_confidence"],
    sensor_ra=raw["sensor_ra"], sensor_snr=raw["sensor_snr"],
    retro_ra=raw.get("retro_ra") if retro else None,
    retro_available=retro, weather=weather,
)
result = st.session_state.live_ekf.update(meas)
st.session_state.live_logger.log(raw, result)

thresholds = THRESHOLDS.get(asset["class"], {"pass": 100, "marginal": 70})
if result.final_ra >= thresholds["pass"]:
    status = "PASS"
elif result.final_ra >= thresholds["marginal"]:
    status = "MARGINAL"
else:
    status = "FAIL"

st.session_state.live_readings.append({
    "step": len(st.session_state.live_readings),
    "final_ra": result.final_ra, "ai_ra": raw["ai_ra"],
    "sensor_ra": raw["sensor_ra"], "retro_ra": raw.get("retro_ra", 0),
    "variance": result.ekf_variance, "bias": result.bias_estimate,
    "alpha": result.alpha, "beta": result.beta, "gamma": result.gamma,
    "asset": asset["name"], "status": status,
})
if len(st.session_state.live_readings) > 100:
    st.session_state.live_readings = st.session_state.live_readings[-100:]

# ── Current reading ────────────────────────────────────────────────────────
status_color = {"PASS": "#22c55e", "MARGINAL": "#f5a623", "FAIL": "#ef4444"}[status]

st.markdown(f"""
<div style="background:#141720; border:1px solid #2a3045; border-radius:12px;
            padding:20px; margin-bottom:16px; border-left:4px solid {status_color};">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span style="font-family:'Space Mono',monospace; color:#f5a623; font-size:14px;">
                {asset['id']}
            </span>
            <span style="color:#e8eaf0; margin-left:8px;">{asset['name']}</span>
            <span style="color:#8b92a8; margin-left:8px;">({asset['class']})</span>
        </div>
        <div>
            <span style="background:{status_color}20; color:{status_color};
                         border:1px solid {status_color}40; padding:4px 16px;
                         border-radius:20px; font-size:13px;
                         font-family:'Space Mono',monospace; font-weight:700;">
                {status}
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sensor cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card" style="border-left:3px solid #818cf8;">
        <div class="section-title">🤖 AI Model</div>
        <div class="ra-value" style="color:#818cf8;">{raw['ai_ra']:.0f}</div>
        <div style="font-size:10px;color:#8b92a8;">conf: {raw['ai_confidence']:.2f}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card" style="border-left:3px solid #34d399;">
        <div class="section-title">📡 Physics Sensor</div>
        <div class="ra-value" style="color:#34d399;">{raw['sensor_ra']:.0f}</div>
        <div style="font-size:10px;color:#8b92a8;">SNR: {raw['sensor_snr']:.1f}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    rv = raw.get('retro_ra', 0)
    rv_display = f"{rv:.0f}" if rv else "N/A"
    st.markdown(f"""<div class="metric-card" style="border-left:3px solid #f472b6;">
        <div class="section-title">🔬 RetroMeter</div>
        <div class="ra-value" style="color:#f472b6;">{rv_display}</div>
        <div style="font-size:10px;color:#8b92a8;">ground truth</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card" style="border-left:3px solid #f5a623;">
        <div class="section-title">⚡ EKF Fused</div>
        <div class="ra-value" style="color:#f5a623;">{result.final_ra:.0f}</div>
        <div style="font-size:10px;color:#8b92a8;">σ²={result.ekf_variance:.1f} | bias={result.bias_estimate:.1f}</div>
    </div>""", unsafe_allow_html=True)

# ── Real-time charts ──────────────────────────────────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
lr = pd.DataFrame(st.session_state.live_readings)

col_chart, col_weights = st.columns([2, 1])

with col_chart:
    st.markdown("<p class='section-title'>📈 Real-Time RA Trace</p>", unsafe_allow_html=True)
    if len(lr) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lr["step"], y=lr["final_ra"], mode='lines+markers',
                                 name='EKF Fused', line=dict(color="#f5a623", width=2),
                                 marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=lr["step"], y=lr["ai_ra"], mode='lines',
                                 name='AI', line=dict(color="#818cf8", width=1, dash='dot'), opacity=0.5))
        fig.add_trace(go.Scatter(x=lr["step"], y=lr["sensor_ra"], mode='lines',
                                 name='Sensor', line=dict(color="#34d399", width=1, dash='dot'), opacity=0.5))
        fig.add_hline(y=150, line_dash="dash", line_color="#22c55e", opacity=0.5)
        fig.add_hline(y=100, line_dash="dash", line_color="#f5a623", opacity=0.5)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                          height=300, margin=dict(l=10,r=10,t=10,b=30),
                          xaxis=dict(showgrid=True, gridcolor="#1e2538",
                                     tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                          yaxis=dict(title="RA", showgrid=True, gridcolor="#1e2538",
                                     tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                          legend=dict(font=dict(size=10, color="#8b92a8"), bgcolor="rgba(0,0,0,0)",
                                      orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True, key="live_chart")

with col_weights:
    st.markdown("<p class='section-title'>⚖️ Current Weights</p>", unsafe_allow_html=True)
    fig_w = go.Figure(go.Bar(
        x=[result.alpha, result.beta, result.gamma],
        y=["Retro (α)", "AI (β)", "Sensor (γ)"],
        orientation='h',
        marker=dict(color=["#f472b6", "#818cf8", "#34d399"]),
        text=[f"{w:.1%}" for w in [result.alpha, result.beta, result.gamma]],
        textposition='outside',
        textfont=dict(family="Space Mono", size=11, color="#e8eaf0"),
    ))
    fig_w.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                        height=180, margin=dict(l=0,r=60,t=10,b=10),
                        xaxis=dict(range=[0,1], showgrid=True, gridcolor="#1e2538",
                                   tickformat=".0%", tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                        yaxis=dict(tickfont=dict(family="Space Mono", size=11, color="#e8eaf0")),
                        showlegend=False)
    st.plotly_chart(fig_w, use_container_width=True, key="live_weights")

    # EKF variance trend
    st.markdown("<p class='section-title'>📉 EKF Variance</p>", unsafe_allow_html=True)
    if len(lr) > 1:
        fig_v = go.Figure(go.Scatter(x=lr["step"], y=lr["variance"], mode='lines',
                                      fill='tozeroy', line=dict(color="#f5a623", width=1),
                                      fillcolor="rgba(245,166,35,0.1)"))
        fig_v.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                            height=120, margin=dict(l=10,r=10,t=5,b=20),
                            xaxis=dict(showgrid=False, tickfont=dict(size=9, color="#8b92a8")),
                            yaxis=dict(showgrid=True, gridcolor="#1e2538",
                                       tickfont=dict(family="Space Mono", size=9, color="#8b92a8")))
        st.plotly_chart(fig_v, use_container_width=True, key="live_variance")

# Auto-refresh
if auto:
    time.sleep(speed)
    st.rerun()
