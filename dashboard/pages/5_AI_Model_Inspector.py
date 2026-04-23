# dashboard/pages/5_AI_Model_Inspector.py
# Model weights, EKF state, bias tracking, and noise analysis

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.data_logger import DataLogger
from modules.fusion_engine import ExtendedKalmanFilter
from config import (DB_PATH, SIGMA_AI, SIGMA_SENSOR, SIGMA_RETRO,
                    SIGMA_RA_PROCESS, WEATHER_MULTIPLIERS, BIAS_LAMBDA)

st.set_page_config(page_title="AI Inspector — RetroFusion", page_icon="🧠", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────
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
  .metric-card { background:#141720; border:1px solid #2a3045; border-radius:8px; padding:16px; margin-bottom:8px;
                 transition:all 0.3s ease; }
  .metric-card:hover { border-color:#f5a623; box-shadow:0 0 20px rgba(245,166,35,0.1); }
  .ra-value { font-family:'Space Mono',monospace; font-size:1.4em; }
  code { font-family:'Space Mono',monospace !important; background:#1e2538 !important;
         color:#f5a623 !important; padding:2px 6px !important; border-radius:4px !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; gap:12px; margin-bottom:20px'>
  <h1 style='font-family:"Space Mono",monospace; font-size:1.2em; color:#f5a623; margin:0;'>
    🧠 AI MODEL INSPECTOR
  </h1>
  <span style='background:rgba(129,140,248,.1); color:#818cf8; border:1px solid rgba(129,140,248,.25);
               padding:3px 10px; border-radius:20px; font-size:11px;
               font-family:"Space Mono",monospace'>DIAGNOSTICS</span>
</div>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────
logger = DataLogger(DB_PATH)
df = logger.query_df(limit=200)

# ═══════════════════════════════════════════════════════════════════════════
# EKF Configuration Card
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("<p class='section-title'>⚙️ EKF Configuration</p>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='metric-card'>
        <div class='section-title'>σ AI</div>
        <div class='ra-value' style='color:#818cf8;'>{SIGMA_AI}</div>
        <div style='font-size:10px;color:#8b92a8;'>mcd/lux/m² base noise</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-card'>
        <div class='section-title'>σ Sensor</div>
        <div class='ra-value' style='color:#34d399;'>{SIGMA_SENSOR}</div>
        <div style='font-size:10px;color:#8b92a8;'>mcd/lux/m² base noise</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card'>
        <div class='section-title'>σ Retro</div>
        <div class='ra-value' style='color:#f472b6;'>{SIGMA_RETRO}</div>
        <div style='font-size:10px;color:#8b92a8;'>mcd/lux/m² base noise</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='metric-card'>
        <div class='section-title'>σ Process</div>
        <div class='ra-value' style='color:#f5a623;'>{SIGMA_RA_PROCESS}</div>
        <div style='font-size:10px;color:#8b92a8;'>RA rate of change</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════
tab_weights, tab_ekf, tab_bias, tab_noise = st.tabs([
    "⚖️ Fusion Weights", "🧮 EKF State", "📐 Bias Tracking", "🌧 Noise Analysis"
])

# ---- Tab 1: Fusion Weights ----
with tab_weights:
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p class='section-title'>Weight History (α, β, γ)</p>", unsafe_allow_html=True)
            fig_wh = go.Figure()
            fig_wh.add_trace(go.Scatter(y=df["fusion_alpha"].values[::-1], mode='lines',
                                        name='α Retro', line=dict(color="#f472b6", width=2)))
            fig_wh.add_trace(go.Scatter(y=df["fusion_beta"].values[::-1], mode='lines',
                                        name='β AI', line=dict(color="#818cf8", width=2)))
            fig_wh.add_trace(go.Scatter(y=df["fusion_gamma"].values[::-1], mode='lines',
                                        name='γ Sensor', line=dict(color="#34d399", width=2)))
            fig_wh.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                                 height=350, margin=dict(l=10,r=10,t=10,b=30),
                                 yaxis=dict(title="Weight", range=[0,1], showgrid=True, gridcolor="#1e2538",
                                            tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                                 xaxis=dict(title="Measurement #", showgrid=True, gridcolor="#1e2538",
                                            tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                                 legend=dict(font=dict(size=10, color="#8b92a8"), bgcolor="rgba(0,0,0,0)",
                                             orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_wh, use_container_width=True, key="weight_history")

        with col2:
            st.markdown("<p class='section-title'>Average Weight Distribution</p>", unsafe_allow_html=True)
            avg_a = df["fusion_alpha"].mean()
            avg_b = df["fusion_beta"].mean()
            avg_g = df["fusion_gamma"].mean()
            fig_wp = go.Figure(go.Pie(
                labels=["RetroMeter (α)", "AI Model (β)", "Sensor (γ)"],
                values=[avg_a, avg_b, avg_g],
                marker=dict(colors=["#f472b6", "#818cf8", "#34d399"],
                            line=dict(width=2, color="#0d0f14")),
                textfont=dict(family="Space Mono", size=12, color="#e8eaf0"),
                hole=0.5,
            ))
            fig_wp.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                                 height=350, margin=dict(l=10,r=10,t=10,b=10),
                                 legend=dict(font=dict(family="DM Sans", size=11, color="#8b92a8")))
            st.plotly_chart(fig_wp, use_container_width=True, key="weight_pie")
    else:
        st.info("No data available. Run measurements first.")

# ---- Tab 2: EKF State ----
with tab_ekf:
    st.markdown("<p class='section-title'>EKF Mathematical Formulation</p>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#141720; border:1px solid #2a3045; border-radius:10px; padding:20px; margin-bottom:16px;">
        <div style="font-family:'Space Mono',monospace; font-size:12px; color:#e8eaf0; line-height:2;">
            <span style="color:#f5a623;">State:</span> x = [RA, dRA/dt]ᵀ<br>
            <span style="color:#f5a623;">Transition:</span> F = [[1, Δt], [0, 1]]<br>
            <span style="color:#f5a623;">Process Q:</span> σ²·[[Δt³/3, Δt²/2], [Δt²/2, Δt]]<br>
            <span style="color:#f5a623;">Measure H:</span> [[1,0], [1,0], [1,0]]<br>
            <span style="color:#f5a623;">Noise R:</span> diag(σ_AI²/c_AI², σ_S²/c_S², σ_R²/c_R²)<br>
            <br>
            <span style="color:#818cf8;">Predict:</span> x̂⁻ = F·x̂, P⁻ = F·P·Fᵀ + Q<br>
            <span style="color:#818cf8;">Innovate:</span> ỹ = z - H·x̂⁻, S = H·P⁻·Hᵀ + R<br>
            <span style="color:#818cf8;">Gain:</span> K = P⁻·Hᵀ·S⁻¹<br>
            <span style="color:#818cf8;">Update:</span> x̂ = x̂⁻ + K·ỹ, P = (I - K·H)·P⁻
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not df.empty and "ekf_variance" in df.columns:
        st.markdown("<p class='section-title'>EKF Variance Over Time</p>", unsafe_allow_html=True)
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Scatter(
            y=df["ekf_variance"].values[::-1], mode='lines',
            fill='tozeroy', line=dict(color="#f5a623", width=2),
            fillcolor="rgba(245,166,35,0.08)",
        ))
        fig_ev.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                             height=300, margin=dict(l=10,r=10,t=10,b=30),
                             xaxis=dict(title="Measurement #", showgrid=True, gridcolor="#1e2538",
                                        tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                             yaxis=dict(title="σ² (Variance)", showgrid=True, gridcolor="#1e2538",
                                        tickfont=dict(family="Space Mono", size=10, color="#8b92a8")))
        st.plotly_chart(fig_ev, use_container_width=True, key="ekf_var_history")

        # Covariance heatmap using current EKF state
        if "ekf" in st.session_state:
            st.markdown("<p class='section-title'>Current Covariance Matrix P</p>", unsafe_allow_html=True)
            from dashboard.components.fusion_panel import render_covariance_heatmap
            render_covariance_heatmap(st.session_state.ekf.P.tolist())

# ---- Tab 3: Bias Tracking ----
with tab_bias:
    st.markdown("<p class='section-title'>AI Bias Correction (EMA)</p>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#141720; border:1px solid #2a3045; border-radius:10px; padding:16px; margin-bottom:16px;">
        <div style="font-family:'Space Mono',monospace; font-size:12px; color:#e8eaf0; line-height:2;">
            <span style="color:#f5a623;">Formula:</span> B̂ₖ = (1-λ)·B̂ₖ₋₁ + λ·(RetroRA - AI_RA)<br>
            <span style="color:#f5a623;">Lambda:</span> {BIAS_LAMBDA}<br>
            <span style="color:#f5a623;">Corrected AI:</span> AI_RA_corrected = AI_RA_raw + B̂ₖ
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not df.empty and "bias_estimate" in df.columns:
        fig_bias = go.Figure()
        bias_vals = df["bias_estimate"].values[::-1]
        fig_bias.add_trace(go.Scatter(
            y=bias_vals, mode='lines',
            line=dict(color="#818cf8", width=2),
            fill='tozeroy',
            fillcolor="rgba(129,140,248,0.08)",
        ))
        fig_bias.add_hline(y=0, line_dash="dash", line_color="#8b92a8", opacity=0.5)
        fig_bias.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                               height=300, margin=dict(l=10,r=10,t=10,b=30),
                               xaxis=dict(title="Measurement #", showgrid=True, gridcolor="#1e2538",
                                          tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                               yaxis=dict(title="Bias Estimate", showgrid=True, gridcolor="#1e2538",
                                          tickfont=dict(family="Space Mono", size=10, color="#8b92a8")))
        st.plotly_chart(fig_bias, use_container_width=True, key="bias_history")

        # AI error distribution
        st.markdown("<p class='section-title'>AI Prediction Error Distribution</p>", unsafe_allow_html=True)
        scatter_df = df.dropna(subset=["ai_ra", "retro_ra"])
        if len(scatter_df) > 2:
            errors = scatter_df["ai_ra"] - scatter_df["retro_ra"]
            fig_err = go.Figure()
            fig_err.add_trace(go.Histogram(
                x=errors, nbinsx=30,
                marker=dict(color="#818cf8", line=dict(width=1, color="#0d0f14")),
            ))
            fig_err.add_vline(x=0, line_dash="dash", line_color="#22c55e")
            fig_err.add_vline(x=errors.mean(), line_dash="dot", line_color="#f5a623",
                             annotation_text=f"Mean: {errors.mean():.1f}",
                             annotation_font=dict(color="#f5a623", size=10, family="Space Mono"))
            fig_err.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                                  height=300, margin=dict(l=10,r=10,t=10,b=30),
                                  xaxis=dict(title="AI Error (AI_RA - Retro_RA)", showgrid=True, gridcolor="#1e2538",
                                             tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                                  yaxis=dict(title="Count", showgrid=True, gridcolor="#1e2538",
                                             tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                                  showlegend=False)
            st.plotly_chart(fig_err, use_container_width=True, key="ai_error_dist")
    else:
        st.info("No bias data available yet.")

# ---- Tab 4: Noise Analysis ----
with tab_noise:
    st.markdown("<p class='section-title'>Weather Noise Multipliers</p>", unsafe_allow_html=True)

    # Table of noise multipliers
    noise_data = []
    for weather, mults in WEATHER_MULTIPLIERS.items():
        noise_data.append({
            "Weather": weather.title(),
            "AI (×)": f"{mults['ai']:.1f}",
            "Sensor (×)": f"{mults['sensor']:.1f}",
            "Retro (×)": f"{mults['retro']:.1f}",
            "σ_AI eff": f"{SIGMA_AI * mults['ai']:.0f}",
            "σ_Sensor eff": f"{SIGMA_SENSOR * mults['sensor']:.0f}",
            "σ_Retro eff": f"{SIGMA_RETRO * mults['retro']:.0f}",
        })
    noise_df = pd.DataFrame(noise_data)
    st.dataframe(noise_df, use_container_width=True, hide_index=True)

    # Effective noise visualization
    st.markdown("<p class='section-title'>Effective Noise σ by Weather</p>", unsafe_allow_html=True)
    fig_noise = go.Figure()
    weathers = list(WEATHER_MULTIPLIERS.keys())

    for sensor, base_sigma, color in [
        ("AI", SIGMA_AI, "#818cf8"),
        ("Sensor", SIGMA_SENSOR, "#34d399"),
        ("Retro", SIGMA_RETRO, "#f472b6"),
    ]:
        effective = [base_sigma * WEATHER_MULTIPLIERS[w][sensor.lower() if sensor != "Retro" else "retro"]
                     for w in weathers]
        fig_noise.add_trace(go.Bar(
            x=[w.title() for w in weathers], y=effective,
            name=sensor, marker_color=color,
            text=[f"{v:.0f}" for v in effective],
            textposition='outside',
            textfont=dict(family="Space Mono", size=10, color="#e8eaf0"),
        ))

    fig_noise.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                            height=350, margin=dict(l=10,r=10,t=10,b=30), barmode='group',
                            xaxis=dict(tickfont=dict(family="Space Mono", size=11, color="#e8eaf0")),
                            yaxis=dict(title="σ (mcd/lux/m²)", showgrid=True, gridcolor="#1e2538",
                                       tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                            legend=dict(font=dict(size=10, color="#8b92a8"), bgcolor="rgba(0,0,0,0)",
                                        orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_noise, use_container_width=True, key="noise_bars")

    # Confidence to weight mapping visualization
    st.markdown("<p class='section-title'>Confidence → Weight Mapping</p>", unsafe_allow_html=True)
    c_range = np.linspace(0.01, 1.0, 100)
    w_ai = (c_range ** 2) / (SIGMA_AI ** 2)
    w_sensor = (c_range ** 2) / (SIGMA_SENSOR ** 2)
    w_retro_val = (0.95 ** 2) / (SIGMA_RETRO ** 2)

    fig_cw = go.Figure()
    fig_cw.add_trace(go.Scatter(x=c_range, y=w_ai / (w_ai + w_sensor[50] + w_retro_val),
                                mode='lines', name='AI', line=dict(color="#818cf8", width=2)))
    fig_cw.add_trace(go.Scatter(x=c_range, y=w_sensor / (w_ai[50] + w_sensor + w_retro_val),
                                mode='lines', name='Sensor', line=dict(color="#34d399", width=2)))
    fig_cw.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                         height=300, margin=dict(l=10,r=10,t=10,b=30),
                         xaxis=dict(title="Confidence", showgrid=True, gridcolor="#1e2538",
                                    tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                         yaxis=dict(title="Normalized Weight", showgrid=True, gridcolor="#1e2538",
                                    tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                         legend=dict(font=dict(size=10, color="#8b92a8"), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_cw, use_container_width=True, key="conf_weight_map")
