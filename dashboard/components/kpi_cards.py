# dashboard/components/kpi_cards.py
# Premium KPI card components for the RetroFusion dashboard

import streamlit as st


def render_kpi_row(stats: dict, rmse: float):
    """Render a row of 4 premium KPI metric cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="section-title">Total Scanned</div>
            <div class="ra-value" style="color:#60a5fa;">{stats.get('total', 0):,}</div>
            <div style="font-size:11px; color:#8b92a8; margin-top:4px;">
                measurements logged
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fail_rate = stats.get('fail_rate', 0)
        fail_color = "#ef4444" if fail_rate > 20 else ("#f5a623" if fail_rate > 10 else "#22c55e")
        st.markdown(f"""
        <div class="metric-card">
            <div class="section-title">Fail Rate</div>
            <div class="ra-value" style="color:{fail_color};">{fail_rate:.1f}%</div>
            <div style="font-size:11px; color:#8b92a8; margin-top:4px;">
                {stats.get('fail_count', 0)} failed / {stats.get('total', 0)} total
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_ra = stats.get('avg_ra', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="section-title">Avg Final RA</div>
            <div class="ra-value" style="color:#f5a623;">{avg_ra:.0f}</div>
            <div style="font-size:11px; color:#8b92a8; margin-top:4px;">
                mcd/lux/m² (EKF fused)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        rmse_color = "#ef4444" if rmse > 50 else ("#f5a623" if rmse > 25 else "#22c55e")
        st.markdown(f"""
        <div class="metric-card">
            <div class="section-title">AI vs Retro RMSE</div>
            <div class="ra-value" style="color:{rmse_color};">{rmse:.1f}</div>
            <div style="font-size:11px; color:#8b92a8; margin-top:4px;">
                mcd/lux/m² deviation
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_status_badge(status: str) -> str:
    """Return HTML for a colored status badge."""
    colors = {
        "PASS": ("#22c55e", "rgba(34,197,94,.15)"),
        "MARGINAL": ("#f5a623", "rgba(245,166,35,.1)"),
        "FAIL": ("#ef4444", "rgba(239,68,68,.15)"),
    }
    color, bg = colors.get(status, ("#8b92a8", "rgba(139,146,168,.1)"))
    return f"""<span style='background:{bg}; color:{color};
                border:1px solid {color}40; padding:2px 10px;
                border-radius:12px; font-size:11px;
                font-family:"Space Mono",monospace;
                font-weight:700'>{status}</span>"""


def render_urgency_badge(urgency: str) -> str:
    """Return HTML for an urgency badge."""
    styles = {
        "immediate":      ("#ef4444", "rgba(239,68,68,.15)", "⚠ IMMEDIATE"),
        "within_30_days": ("#f5a623", "rgba(245,166,35,.1)", "⏳ 30 DAYS"),
        "within_90_days": ("#60a5fa", "rgba(96,165,250,.1)", "📋 90 DAYS"),
        "ok":             ("#22c55e", "rgba(34,197,94,.15)", "✓ OK"),
    }
    color, bg, label = styles.get(urgency, ("#8b92a8", "rgba(139,146,168,.1)", urgency))
    return f"""<span style='background:{bg}; color:{color};
                border:1px solid {color}40; padding:3px 12px;
                border-radius:12px; font-size:11px;
                font-family:"Space Mono",monospace;
                font-weight:700'>{label}</span>"""
