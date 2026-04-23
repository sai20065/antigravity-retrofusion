# dashboard/pages/3_Analytics.py
# Historical charts, correlation analysis, and statistics

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.data_logger import DataLogger
from config import DB_PATH, THRESHOLDS

st.set_page_config(page_title="Analytics — RetroFusion", page_icon="📊", layout="wide")

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
    📊 ANALYTICS
  </h1>
  <span style='background:rgba(139,92,246,.1); color:#a78bfa; border:1px solid rgba(139,92,246,.25);
               padding:3px 10px; border-radius:20px; font-size:11px;
               font-family:"Space Mono",monospace'>HISTORICAL</span>
</div>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────
logger = DataLogger(DB_PATH)
df = logger.query_df(limit=500)

if df.empty:
    st.warning("No measurement data available. Run the Live Monitor first to collect data.")
    st.stop()

# ── Summary stats ──────────────────────────────────────────────────────────
stats = logger.get_stats()
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class='metric-card'>
        <div class='section-title'>Total</div>
        <div class='ra-value' style='color:#60a5fa;'>{stats['total']:,}</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-card'>
        <div class='section-title'>Pass</div>
        <div class='ra-value' style='color:#22c55e;'>{stats['pass_count']:,}</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card'>
        <div class='section-title'>Marginal</div>
        <div class='ra-value' style='color:#f5a623;'>{stats['marginal_count']:,}</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='metric-card'>
        <div class='section-title'>Fail</div>
        <div class='ra-value' style='color:#ef4444;'>{stats['fail_count']:,}</div></div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class='metric-card'>
        <div class='section-title'>Avg RA</div>
        <div class='ra-value' style='color:#f5a623;'>{stats['avg_ra']:.0f}</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────────
tab_dist, tab_corr, tab_weather, tab_asset = st.tabs([
    "📊 Distribution", "🔗 Correlation", "🌤 Weather Impact", "🏗 By Asset Type"
])

# ---- Tab 1: Distribution ----
with tab_dist:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p class='section-title'>RA Distribution</p>", unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df["final_ra"], nbinsx=40, name="Final RA",
            marker=dict(color="#f5a623", line=dict(width=1, color="#0d0f14")),
        ))
        fig_hist.add_vline(x=150, line_dash="dash", line_color="#22c55e",
                           annotation_text="PASS", annotation_position="top right",
                           annotation_font=dict(color="#22c55e", size=10, family="Space Mono"))
        fig_hist.add_vline(x=100, line_dash="dash", line_color="#f5a623",
                           annotation_text="MARGINAL", annotation_position="top left",
                           annotation_font=dict(color="#f5a623", size=10, family="Space Mono"))
        fig_hist.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                               height=350, margin=dict(l=10,r=10,t=10,b=30),
                               xaxis=dict(title="RA (mcd/lux/m²)", showgrid=True, gridcolor="#1e2538",
                                          tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                               yaxis=dict(title="Count", showgrid=True, gridcolor="#1e2538",
                                          tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                               showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True, key="ra_hist")

    with col2:
        st.markdown("<p class='section-title'>Status Breakdown</p>", unsafe_allow_html=True)
        status_counts = df["status"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=status_counts.index, values=status_counts.values,
            marker=dict(colors=["#22c55e", "#ef4444", "#f5a623"],
                        line=dict(width=2, color="#0d0f14")),
            textfont=dict(family="Space Mono", size=12, color="#e8eaf0"),
            hole=0.55,
        ))
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                              height=350, margin=dict(l=10,r=10,t=10,b=10),
                              legend=dict(font=dict(family="DM Sans", size=11, color="#8b92a8")),
                              annotations=[dict(text=f"{stats['total']}", x=0.5, y=0.5,
                                                font=dict(family="Space Mono", size=24, color="#f5a623"),
                                                showarrow=False)])
        st.plotly_chart(fig_pie, use_container_width=True, key="status_pie")

# ---- Tab 2: Correlation ----
with tab_corr:
    st.markdown("<p class='section-title'>Sensor Correlation Matrix</p>", unsafe_allow_html=True)
    cols = ["ai_ra", "sensor_ra", "retro_ra", "final_ra"]
    avail_cols = [c for c in cols if c in df.columns]
    corr_df = df[avail_cols].dropna()
    if len(corr_df) > 2:
        corr_matrix = corr_df.corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
            colorscale=[[0,"#0d0f14"],[0.5,"#4338ca"],[1,"#f5a623"]],
            text=[[f"{v:.3f}" for v in row] for row in corr_matrix.values],
            texttemplate="%{text}",
            textfont=dict(family="Space Mono", size=13, color="#e8eaf0"),
            showscale=True,
            colorbar=dict(tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
        ))
        fig_corr.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                               height=400, margin=dict(l=10,r=10,t=10,b=10),
                               xaxis=dict(tickfont=dict(family="Space Mono", size=11, color="#e8eaf0")),
                               yaxis=dict(tickfont=dict(family="Space Mono", size=11, color="#e8eaf0")))
        st.plotly_chart(fig_corr, use_container_width=True, key="corr_matrix")
    else:
        st.info("Need more data for correlation analysis.")

    # Box plot of sensors
    st.markdown("<p class='section-title'>Sensor Reading Distribution</p>", unsafe_allow_html=True)
    if len(corr_df) > 2:
        fig_box = go.Figure()
        sensor_colors = {"ai_ra": "#818cf8", "sensor_ra": "#34d399", "retro_ra": "#f472b6", "final_ra": "#f5a623"}
        for col in avail_cols:
            fig_box.add_trace(go.Box(y=df[col].dropna(), name=col.replace("_", " ").title(),
                                     marker_color=sensor_colors.get(col, "#8b92a8"),
                                     line_color=sensor_colors.get(col, "#8b92a8")))
        fig_box.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                              height=300, margin=dict(l=10,r=10,t=10,b=30),
                              yaxis=dict(title="RA (mcd/lux/m²)", showgrid=True, gridcolor="#1e2538",
                                         tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                              showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True, key="sensor_box")

# ---- Tab 3: Weather Impact ----
with tab_weather:
    st.markdown("<p class='section-title'>RA by Weather Condition</p>", unsafe_allow_html=True)
    if "weather_code" in df.columns:
        weather_stats = df.groupby("weather_code").agg(
            avg_ra=("final_ra", "mean"),
            std_ra=("final_ra", "std"),
            count=("final_ra", "count"),
            avg_variance=("ekf_variance", "mean"),
        ).reset_index()

        if not weather_stats.empty:
            weather_colors = {"clear": "#22c55e", "rain": "#60a5fa", "fog": "#f5a623", "snow": "#e8eaf0"}
            fig_weather = go.Figure()
            for _, row in weather_stats.iterrows():
                fig_weather.add_trace(go.Bar(
                    x=[row["weather_code"]], y=[row["avg_ra"]],
                    name=row["weather_code"],
                    marker_color=weather_colors.get(row["weather_code"], "#8b92a8"),
                    error_y=dict(type='data', array=[row["std_ra"]], visible=True,
                                 color="#8b92a8"),
                    text=[f"{row['avg_ra']:.0f}"],
                    textposition='outside',
                    textfont=dict(family="Space Mono", size=12, color="#e8eaf0"),
                ))
            fig_weather.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                                      height=350, margin=dict(l=10,r=10,t=10,b=30),
                                      xaxis=dict(title="Weather", tickfont=dict(family="Space Mono", size=11, color="#e8eaf0")),
                                      yaxis=dict(title="Avg RA", showgrid=True, gridcolor="#1e2538",
                                                 tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                                      showlegend=False)
            st.plotly_chart(fig_weather, use_container_width=True, key="weather_bars")

            # EKF variance by weather
            st.markdown("<p class='section-title'>EKF Variance by Weather</p>", unsafe_allow_html=True)
            fig_var = go.Figure()
            for _, row in weather_stats.iterrows():
                fig_var.add_trace(go.Bar(
                    x=[row["weather_code"]], y=[row["avg_variance"]],
                    marker_color=weather_colors.get(row["weather_code"], "#8b92a8"),
                    text=[f"{row['avg_variance']:.1f}"],
                    textposition='outside',
                    textfont=dict(family="Space Mono", size=12, color="#e8eaf0"),
                ))
            fig_var.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                                  height=300, margin=dict(l=10,r=10,t=10,b=30),
                                  xaxis=dict(tickfont=dict(family="Space Mono", size=11, color="#e8eaf0")),
                                  yaxis=dict(title="Avg σ²", showgrid=True, gridcolor="#1e2538",
                                             tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                                  showlegend=False)
            st.plotly_chart(fig_var, use_container_width=True, key="weather_var")

# ---- Tab 4: By Asset Type ----
with tab_asset:
    st.markdown("<p class='section-title'>RA by Asset Class</p>", unsafe_allow_html=True)
    if "asset_class" in df.columns:
        asset_stats = df.groupby("asset_class").agg(
            avg_ra=("final_ra", "mean"),
            min_ra=("final_ra", "min"),
            max_ra=("final_ra", "max"),
            count=("final_ra", "count"),
            fail_rate=("status", lambda x: (x == "FAIL").mean() * 100),
        ).reset_index()

        if not asset_stats.empty:
            fig_asset = go.Figure()
            fig_asset.add_trace(go.Bar(
                x=asset_stats["asset_class"], y=asset_stats["avg_ra"],
                name="Avg RA", marker_color="#f5a623",
                text=[f"{v:.0f}" for v in asset_stats["avg_ra"]],
                textposition='outside',
                textfont=dict(family="Space Mono", size=11, color="#e8eaf0"),
            ))

            # Add threshold lines for each class
            for _, row in asset_stats.iterrows():
                thresh = THRESHOLDS.get(row["asset_class"], {}).get("pass", 100)
                fig_asset.add_annotation(
                    x=row["asset_class"], y=thresh,
                    text=f"T={thresh}", showarrow=True, arrowhead=2,
                    arrowcolor="#ef4444", arrowwidth=1,
                    font=dict(family="Space Mono", size=9, color="#ef4444"),
                )

            fig_asset.update_layout(template="plotly_dark", paper_bgcolor="#141720", plot_bgcolor="#141720",
                                    height=350, margin=dict(l=10,r=10,t=10,b=30),
                                    xaxis=dict(tickfont=dict(family="Space Mono", size=10, color="#e8eaf0"), tickangle=-30),
                                    yaxis=dict(title="RA", showgrid=True, gridcolor="#1e2538",
                                               tickfont=dict(family="Space Mono", size=10, color="#8b92a8")),
                                    showlegend=False)
            st.plotly_chart(fig_asset, use_container_width=True, key="asset_bars")

            # Fail rate table
            st.markdown("<p class='section-title'>Fail Rate by Asset Class</p>", unsafe_allow_html=True)
            st.dataframe(
                asset_stats[["asset_class", "count", "avg_ra", "min_ra", "max_ra", "fail_rate"]].rename(
                    columns={"asset_class": "Class", "count": "Readings", "avg_ra": "Avg RA",
                             "min_ra": "Min RA", "max_ra": "Max RA", "fail_rate": "Fail %"}
                ),
                use_container_width=True, hide_index=True,
            )

# ── Export ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p class='section-title'>📥 Export</p>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Full Dataset (CSV)", csv, "retrofusion_analytics.csv",
                       "text/csv", key="analytics_csv")
with c2:
    # Generate summary HTML report
    html_report = f"""
    <html><head><style>
    body {{ font-family: 'DM Sans', sans-serif; background:#0d0f14; color:#e8eaf0; padding:20px; }}
    h1 {{ color:#f5a623; font-family:'Space Mono',monospace; }}
    table {{ border-collapse:collapse; width:100%; margin:20px 0; }}
    th,td {{ border:1px solid #2a3045; padding:8px; text-align:left; }}
    th {{ background:#141720; color:#f5a623; font-family:'Space Mono',monospace; }}
    .pass {{ color:#22c55e; }} .fail {{ color:#ef4444; }} .marg {{ color:#f5a623; }}
    </style></head><body>
    <h1>RetroFusion AI+ Pro — Analytics Report</h1>
    <p>Total Measurements: <strong>{stats['total']}</strong></p>
    <p>Pass: <span class='pass'>{stats['pass_count']}</span> |
       Marginal: <span class='marg'>{stats['marginal_count']}</span> |
       Fail: <span class='fail'>{stats['fail_count']}</span></p>
    <p>Avg RA: <strong>{stats['avg_ra']:.0f}</strong> mcd/lux/m²</p>
    <p>Fail Rate: <strong>{stats['fail_rate']:.1f}%</strong></p>
    </body></html>"""
    st.download_button("📄 Download Summary (HTML)", html_report, "retrofusion_report.html",
                       "text/html", key="analytics_html")
