# dashboard/pages/4_Predictive_Maintenance.py
# Decay curves, failure forecasts, and maintenance scheduling

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.data_logger import DataLogger
from modules.predictive import predict_failure, ra_decay_model, MaintenanceForecast
from modules.simulator import ROAD_ASSETS
from config import DB_PATH, THRESHOLDS, DEGRADATION_RATES

st.set_page_config(page_title="Predictive Maintenance — RetroFusion", page_icon="🔧", layout="wide")

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
                 transition: all 0.3s ease; }
  .metric-card:hover { border-color:#f5a623; box-shadow:0 0 20px rgba(245,166,35,0.1); }
  .ra-value { font-family:'Space Mono',monospace; font-size:1.4em; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; gap:12px; margin-bottom:20px'>
  <h1 style='font-family:"Space Mono",monospace; font-size:1.2em; color:#f5a623; margin:0;'>
    🔧 PREDICTIVE MAINTENANCE
  </h1>
  <span style='background:rgba(239,68,68,.1); color:#ef4444; border:1px solid rgba(239,68,68,.25);
               padding:3px 10px; border-radius:20px; font-size:11px;
               font-family:"Space Mono",monospace'>FORECAST</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-family:\"Space Mono\",monospace; font-size:14px; color:#f5a623; text-align:center; padding:12px 0; font-weight:700;'>🔧 MAINTENANCE</div>", unsafe_allow_html=True)
    material = st.selectbox("Material Type", list(DEGRADATION_RATES.keys()), key="pm_material")
    sim_days = st.slider("Simulation Horizon (days)", 30, 730, 365, key="pm_horizon")

# ── Generate forecasts for each asset ─────────────────────────────────────
logger = DataLogger(DB_PATH)
df = logger.query_df(limit=500)

forecasts = []

if not df.empty and "asset_id" in df.columns:
    for asset in ROAD_ASSETS:
        asset_df = df[df["asset_id"] == asset["id"]].sort_values("timestamp")
        if len(asset_df) >= 3:
            timestamps = ((asset_df["timestamp"] - asset_df["timestamp"].min()) / 86400).tolist()
            ra_values = asset_df["final_ra"].tolist()
            forecast = predict_failure(timestamps, ra_values,
                                        asset["id"], asset["class"], material)
            forecasts.append(forecast)
        else:
            # Generate synthetic data for demo
            np.random.seed(hash(asset["id"]) % (2**31))
            ra0 = asset["ra_true"]
            lam = DEGRADATION_RATES.get(material, 0.001)
            t_syn = np.arange(0, 180, 5)
            ra_syn = ra0 * np.exp(-lam * t_syn) + np.random.normal(0, 10, len(t_syn))
            forecast = predict_failure(t_syn.tolist(), ra_syn.tolist(),
                                        asset["id"], asset["class"], material)
            forecasts.append(forecast)

# ── Urgency summary ──────────────────────────────────────────────────────
if forecasts:
    immediate = sum(1 for f in forecasts if f.urgency == "immediate")
    within_30 = sum(1 for f in forecasts if f.urgency == "within_30_days")
    within_90 = sum(1 for f in forecasts if f.urgency == "within_90_days")
    ok_count = sum(1 for f in forecasts if f.urgency == "ok")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card' style='border-left:3px solid #ef4444;'>
            <div class='section-title'>⚠ Immediate</div>
            <div class='ra-value' style='color:#ef4444;'>{immediate}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card' style='border-left:3px solid #f5a623;'>
            <div class='section-title'>⏳ Within 30d</div>
            <div class='ra-value' style='color:#f5a623;'>{within_30}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card' style='border-left:3px solid #60a5fa;'>
            <div class='section-title'>📋 Within 90d</div>
            <div class='ra-value' style='color:#60a5fa;'>{within_90}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card' style='border-left:3px solid #22c55e;'>
            <div class='section-title'>✓ OK</div>
            <div class='ra-value' style='color:#22c55e;'>{ok_count}</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Decay curves ──────────────────────────────────────────────────────────
st.markdown("<p class='section-title'>📉 RA Decay Forecast Curves</p>", unsafe_allow_html=True)

if forecasts:
    fig_decay = go.Figure()
    colors = ["#f5a623", "#818cf8", "#34d399", "#f472b6", "#60a5fa", "#ef4444", "#a78bfa", "#fb923c"]

    for i, forecast in enumerate(forecasts):
        color = colors[i % len(colors)]
        t_future = np.linspace(0, sim_days, 200)
        ra_curve = ra_decay_model(t_future, forecast.ra0_estimated, forecast.lambda_decay)

        fig_decay.add_trace(go.Scatter(
            x=t_future, y=ra_curve,
            mode='lines',
            name=f"{forecast.asset_id}",
            line=dict(color=color, width=2),
        ))

        # Mark failure point
        if forecast.days_to_failure is not None and 0 < forecast.days_to_failure < sim_days:
            thresh = THRESHOLDS.get(forecast.asset_type, {}).get("pass", 100)
            fig_decay.add_trace(go.Scatter(
                x=[forecast.days_to_failure], y=[thresh],
                mode='markers',
                marker=dict(size=10, color="#ef4444", symbol="x", line=dict(width=2, color="#ef4444")),
                name=f"{forecast.asset_id} fail",
                showlegend=False,
            ))

    # Threshold band
    fig_decay.add_hline(y=150, line_dash="dash", line_color="#22c55e", opacity=0.5,
                        annotation_text="PASS", annotation_position="right",
                        annotation_font=dict(color="#22c55e", size=10, family="Space Mono"))
    fig_decay.add_hline(y=100, line_dash="dash", line_color="#f5a623", opacity=0.5,
                        annotation_text="MARGINAL", annotation_position="right",
                        annotation_font=dict(color="#f5a623", size=10, family="Space Mono"))

    fig_decay.update_layout(
        template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
        height=450, margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title="Days from Now", showgrid=True, gridcolor="#1e2538",
                   tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
        yaxis=dict(title="RA (mcd/lux/m²)", showgrid=True, gridcolor="#1e2538",
                   tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
        legend=dict(font=dict(family="DM Sans", size=10, color="#8b92a8"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_decay, use_container_width=True, key="decay_curves")

# ── Forecast table ────────────────────────────────────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>📋 Maintenance Schedule</p>", unsafe_allow_html=True)

if forecasts:
    urgency_colors = {
        "immediate": "#ef4444", "within_30_days": "#f5a623",
        "within_90_days": "#60a5fa", "ok": "#22c55e"
    }
    urgency_labels = {
        "immediate": "⚠ IMMEDIATE", "within_30_days": "⏳ 30 DAYS",
        "within_90_days": "📋 90 DAYS", "ok": "✓ OK"
    }

    for f in sorted(forecasts, key=lambda x: ["immediate", "within_30_days", "within_90_days", "ok"].index(x.urgency)):
        uc = urgency_colors[f.urgency]
        ul = urgency_labels[f.urgency]
        dtf = f"{f.days_to_failure:.0f} days" if f.days_to_failure is not None else "N/A"
        unc = f"±{f.days_uncertainty:.0f}d" if f.days_uncertainty else ""

        st.markdown(f"""
        <div style="background:#141720; border:1px solid #2a3045; border-radius:8px;
                    padding:14px 18px; margin-bottom:8px; border-left:4px solid {uc};
                    display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-family:'Space Mono',monospace; color:#f5a623;
                             font-size:13px; font-weight:700;">{f.asset_id}</span>
                <span style="color:#8b92a8; font-size:12px; margin-left:10px;">{f.asset_type}</span>
            </div>
            <div style="text-align:right;">
                <span style="font-family:'Space Mono',monospace; color:#e8eaf0;
                             font-size:12px;">RA: {f.current_ra:.0f}</span>
                <span style="color:#8b92a8; margin:0 8px;">|</span>
                <span style="font-family:'Space Mono',monospace; color:#e8eaf0;
                             font-size:12px;">Fail: {dtf} {unc}</span>
                <span style="color:#8b92a8; margin:0 8px;">|</span>
                <span style="font-family:'Space Mono',monospace; color:#e8eaf0;
                             font-size:12px;">Conf: {f.forecast_confidence:.0%}</span>
                <span style="color:#8b92a8; margin:0 8px;">|</span>
                <span style="background:{uc}20; color:{uc}; border:1px solid {uc}40;
                             padding:2px 10px; border-radius:12px; font-size:11px;
                             font-family:'Space Mono',monospace; font-weight:700;">
                    {ul}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Export maintenance schedule
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    schedule_data = [{
        "Asset ID": f.asset_id, "Type": f.asset_type,
        "Current RA": f"{f.current_ra:.0f}",
        "RA₀ Est": f"{f.ra0_estimated:.0f}",
        "λ Decay": f"{f.lambda_decay:.6f}",
        "Days to Fail": f"{f.days_to_failure:.0f}" if f.days_to_failure else "N/A",
        "Uncertainty": f"±{f.days_uncertainty:.0f}d" if f.days_uncertainty else "-",
        "Confidence": f"{f.forecast_confidence:.0%}",
        "Urgency": f.urgency,
        "Action": f.recommended_action,
    } for f in forecasts]
    schedule_df = pd.DataFrame(schedule_data)
    csv = schedule_df.to_csv(index=False)
    st.download_button("📥 Export Maintenance Schedule (CSV)", csv,
                       "maintenance_schedule.csv", "text/csv", key="pm_csv")

else:
    st.info("No data available for predictive maintenance. Run measurements first.")
