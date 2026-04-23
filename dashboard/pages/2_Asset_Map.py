# dashboard/pages/2_Asset_Map.py
# Full Folium map with all road assets

import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modules.data_logger import DataLogger
from modules.simulator import ROAD_ASSETS
from config import DB_PATH

st.set_page_config(page_title="Asset Map — RetroFusion", page_icon="🗺️", layout="wide")

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
  .metric-card { background:#141720; border:1px solid #2a3045; border-radius:8px; padding:16px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; gap:12px; margin-bottom:20px'>
  <h1 style='font-family:"Space Mono",monospace; font-size:1.2em; color:#f5a623; margin:0;'>
    🗺️ ASSET MAP
  </h1>
  <span style='background:rgba(96,165,250,.1); color:#60a5fa; border:1px solid rgba(96,165,250,.25);
               padding:3px 10px; border-radius:20px; font-size:11px;
               font-family:"Space Mono",monospace'>GEO VIEW</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar filters ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-family:\"Space Mono\",monospace; font-size:14px; color:#f5a623; text-align:center; padding:12px 0; font-weight:700;'>🗺️ MAP CONTROLS</div>", unsafe_allow_html=True)
    status_filter = st.multiselect("Filter by Status", ["PASS", "MARGINAL", "FAIL"],
                                    default=["PASS", "MARGINAL", "FAIL"], key="map_status")
    asset_filter = st.multiselect("Filter by Type", ["sign", "marking", "stud"],
                                   default=["sign", "marking", "stud"], key="map_type")
    show_heatmap = st.toggle("Show Fail Heatmap", value=True, key="map_heat")

# ── Load data ──────────────────────────────────────────────────────────────
logger = DataLogger(DB_PATH)
df = logger.query_df(limit=500, status_filter=status_filter if status_filter else None)

# Apply type filter
if not df.empty and asset_filter:
    df = df[df["asset_type"].isin(asset_filter)]

# ── Stats row ──────────────────────────────────────────────────────────────
if not df.empty:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='section-title'>Assets Scanned</div>
            <div style='font-family:"Space Mono",monospace; font-size:1.3em; color:#60a5fa;'>
                {df['asset_id'].nunique()}</div></div>""", unsafe_allow_html=True)
    with c2:
        pass_pct = (df['status'] == 'PASS').mean() * 100
        st.markdown(f"""<div class='metric-card'>
            <div class='section-title'>Pass Rate</div>
            <div style='font-family:"Space Mono",monospace; font-size:1.3em; color:#22c55e;'>
                {pass_pct:.1f}%</div></div>""", unsafe_allow_html=True)
    with c3:
        fail_count = (df['status'] == 'FAIL').sum()
        st.markdown(f"""<div class='metric-card'>
            <div class='section-title'>Failures</div>
            <div style='font-family:"Space Mono",monospace; font-size:1.3em; color:#ef4444;'>
                {fail_count}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='section-title'>Total Readings</div>
            <div style='font-family:"Space Mono",monospace; font-size:1.3em; color:#f5a623;'>
                {len(df)}</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── Map ────────────────────────────────────────────────────────────────────
from dashboard.components.map_view import render_asset_map
render_asset_map(df, height=550, key="asset_map_main")

# ── Asset inventory table ─────────────────────────────────────────────────
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>📋 Asset Inventory</p>", unsafe_allow_html=True)

asset_df = pd.DataFrame(ROAD_ASSETS)
asset_df.columns = [c.replace("_", " ").title() for c in asset_df.columns]
st.dataframe(asset_df, use_container_width=True, hide_index=True)
