# dashboard/components/map_view.py
# Folium map component for asset visualization

import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd


STATUS_COLORS = {
    "PASS": "#22c55e",
    "MARGINAL": "#f5a623",
    "FAIL": "#ef4444",
}


def render_asset_map(df: pd.DataFrame, height: int = 500, key: str = "main_map"):
    """
    Render a Folium map with color-coded markers for each measurement.

    Args:
        df: DataFrame with latitude, longitude, final_ra, status, asset_type columns
        height: Map height in pixels
        key: Unique key for Streamlit component
    """
    if df.empty:
        st.info("No measurement data available for map. Run some measurements first.")
        return

    # Center map on mean location
    center_lat = df["latitude"].mean() if "latitude" in df.columns else 12.9716
    center_lon = df["longitude"].mean() if "longitude" in df.columns else 77.5946

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=16,
    )

    # Dark tile layer
    folium.TileLayer(
        tiles="https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png",
        attr="Stadia Maps",
        name="Dark Mode",
        max_zoom=20,
    ).add_to(m)

    # Add markers for each measurement
    for _, row in df.iterrows():
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            continue

        status = row.get("status", "PASS")
        color = STATUS_COLORS.get(status, "#8b92a8")
        ra_val = row.get("final_ra", 0)
        asset_type = row.get("asset_type", "unknown")
        asset_class = row.get("asset_class", "")
        asset_id = row.get("asset_id", "")

        popup_html = f"""
        <div style="font-family:'DM Sans',sans-serif; background:#141720;
                    color:#e8eaf0; padding:12px; border-radius:8px;
                    min-width:180px;">
            <div style="font-family:'Space Mono',monospace; color:#f5a623;
                        font-size:13px; font-weight:700; margin-bottom:6px;">
                {asset_id}
            </div>
            <table style="font-size:11px; width:100%;">
                <tr><td style="color:#8b92a8;">Type</td>
                    <td style="text-align:right;">{asset_type}</td></tr>
                <tr><td style="color:#8b92a8;">Class</td>
                    <td style="text-align:right;">{asset_class}</td></tr>
                <tr><td style="color:#8b92a8;">RA</td>
                    <td style="text-align:right; font-family:'Space Mono',monospace;
                               color:{color}; font-weight:700;">
                        {ra_val:.0f}
                    </td></tr>
                <tr><td style="color:#8b92a8;">Status</td>
                    <td style="text-align:right; color:{color};
                               font-weight:700;">{status}</td></tr>
            </table>
        </div>
        """

        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)

    # Add heatmap for FAIL density
    if "status" in df.columns:
        fail_df = df[df["status"] == "FAIL"]
        if not fail_df.empty and "latitude" in fail_df.columns:
            heat_data = fail_df[["latitude", "longitude"]].dropna().values.tolist()
            if heat_data:
                HeatMap(
                    heat_data,
                    radius=25,
                    blur=15,
                    min_opacity=0.3,
                ).add_to(m)

    st_folium(m, height=height, use_container_width=True)
