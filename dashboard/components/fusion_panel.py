# dashboard/components/fusion_panel.py
# Sensor Fusion Visualization Panel

import streamlit as st
import plotly.graph_objects as go
import numpy as np


def render_live_measurement(raw: dict, result, status: str):
    """Render the live measurement panel with sensor readings and EKF output."""
    asset = raw["asset"]

    # Asset header
    st.markdown(f"""
    <div style="background:#141720; border:1px solid #2a3045; border-radius:10px;
                padding:16px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-family:'Space Mono',monospace; font-size:13px;
                             color:#f5a623; letter-spacing:0.05em;">
                    {asset['id']}
                </span>
                <span style="color:#8b92a8; font-size:12px; margin-left:8px;">
                    {asset['name']}
                </span>
            </div>
            <div>
                <span style="background:{'rgba(34,197,94,.15)' if status=='PASS' else ('rgba(245,166,35,.1)' if status=='MARGINAL' else 'rgba(239,68,68,.15)')};
                             color:{'#22c55e' if status=='PASS' else ('#f5a623' if status=='MARGINAL' else '#ef4444')};
                             border:1px solid {'#22c55e40' if status=='PASS' else ('#f5a62340' if status=='MARGINAL' else '#ef444440')};
                             padding:4px 14px; border-radius:20px; font-size:12px;
                             font-family:'Space Mono',monospace; font-weight:700;">
                    {status}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sensor readings row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid #818cf8;">
            <div class="section-title">🤖 AI Model</div>
            <div class="ra-value" style="color:#818cf8;">{raw['ai_ra']:.0f}</div>
            <div style="font-size:10px; color:#8b92a8; margin-top:4px;">
                conf: {raw['ai_confidence']:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid #34d399;">
            <div class="section-title">📡 Physics Sensor</div>
            <div class="ra-value" style="color:#34d399;">{raw['sensor_ra']:.0f}</div>
            <div style="font-size:10px; color:#8b92a8; margin-top:4px;">
                SNR: {raw['sensor_snr']:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        retro_val = raw.get('retro_ra', 0)
        retro_display = f"{retro_val:.0f}" if retro_val else "N/A"
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid #f472b6;">
            <div class="section-title">🔬 RetroMeter</div>
            <div class="ra-value" style="color:#f472b6;">{retro_display}</div>
            <div style="font-size:10px; color:#8b92a8; margin-top:4px;">
                ground truth
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid #f5a623;">
            <div class="section-title">⚡ EKF Fused</div>
            <div class="ra-value" style="color:#f5a623;">{result.final_ra:.0f}</div>
            <div style="font-size:10px; color:#8b92a8; margin-top:4px;">
                σ² = {result.ekf_variance:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_weight_bars(result):
    """Render fusion weight bar chart."""
    fig = go.Figure()

    weights = [result.alpha, result.beta, result.gamma]
    labels = ["RetroMeter (α)", "AI Model (β)", "Sensor (γ)"]
    colors = ["#f472b6", "#818cf8", "#34d399"]

    fig.add_trace(go.Bar(
        x=weights,
        y=labels,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"{w:.1%}" for w in weights],
        textposition='outside',
        textfont=dict(family="Space Mono", size=12, color="#e8eaf0"),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#141720",
        plot_bgcolor="#141720",
        height=180,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            range=[0, 1],
            showgrid=True,
            gridcolor="#2a3045",
            tickformat=".0%",
            tickfont=dict(family="Space Mono", size=10, color="#8b92a8"),
        ),
        yaxis=dict(
            tickfont=dict(family="Space Mono", size=11, color="#e8eaf0"),
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, key="weight_bars")


def render_covariance_heatmap(P_matrix):
    """Render EKF covariance matrix as a heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=P_matrix,
        x=["RA", "dRA/dt"],
        y=["RA", "dRA/dt"],
        colorscale=[
            [0, "#0d0f14"],
            [0.25, "#1e1b4b"],
            [0.5, "#4338ca"],
            [0.75, "#818cf8"],
            [1, "#c4b5fd"],
        ],
        text=[[f"{v:.2f}" for v in row] for row in P_matrix],
        texttemplate="%{text}",
        textfont=dict(family="Space Mono", size=14, color="#e8eaf0"),
        showscale=True,
        colorbar=dict(
            tickfont=dict(family="Space Mono", size=10, color="#8b92a8"),
        ),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#141720",
        plot_bgcolor="#141720",
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(
            text="EKF Covariance Matrix P",
            font=dict(family="Space Mono", size=12, color="#8b92a8"),
        ),
        xaxis=dict(tickfont=dict(family="Space Mono", size=11, color="#e8eaf0")),
        yaxis=dict(tickfont=dict(family="Space Mono", size=11, color="#e8eaf0")),
    )

    st.plotly_chart(fig, use_container_width=True, key="cov_heatmap")
