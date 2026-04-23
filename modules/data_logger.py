# modules/data_logger.py
import sqlite3, time, os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_PATH, THRESHOLDS


SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    latitude        REAL,
    longitude       REAL,
    speed_kmh       REAL,
    asset_id        TEXT,
    asset_type      TEXT    NOT NULL,
    asset_class     TEXT,
    ai_ra           REAL,
    ai_confidence   REAL,
    sensor_ra       REAL,
    sensor_snr      REAL,
    retro_ra        REAL,
    retro_available INTEGER DEFAULT 0,
    final_ra        REAL    NOT NULL,
    fusion_alpha    REAL,
    fusion_beta     REAL,
    fusion_gamma    REAL,
    ekf_variance    REAL,
    bias_estimate   REAL,
    weather_code    TEXT DEFAULT 'clear',
    ambient_lux     REAL,
    status          TEXT    NOT NULL,
    synced          INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_geo    ON measurements(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_time   ON measurements(timestamp);
CREATE INDEX IF NOT EXISTS idx_status ON measurements(status);
CREATE INDEX IF NOT EXISTS idx_asset  ON measurements(asset_type);
"""


def _get_status(asset_class: str, final_ra: float) -> str:
    thresholds = THRESHOLDS.get(asset_class, {"pass": 100, "marginal": 70})
    if final_ra >= thresholds["pass"]:
        return "PASS"
    elif final_ra >= thresholds["marginal"]:
        return "MARGINAL"
    return "FAIL"


class DataLogger:
    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()

    def log(self, sim_result: dict, fusion_result) -> int:
        asset       = sim_result["asset"]
        asset_class = asset["class"]
        status      = _get_status(asset_class, fusion_result.final_ra)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            INSERT INTO measurements
            (timestamp, latitude, longitude, speed_kmh,
             asset_id, asset_type, asset_class,
             ai_ra, ai_confidence, sensor_ra, sensor_snr,
             retro_ra, retro_available, final_ra,
             fusion_alpha, fusion_beta, fusion_gamma,
             ekf_variance, bias_estimate,
             weather_code, ambient_lux, status)
            VALUES (?,?,?,?, ?,?,?, ?,?,?,?, ?,?,?, ?,?,?, ?,?, ?,?,?)
        """, (
            sim_result["timestamp"],
            sim_result["lat"], sim_result["lon"], sim_result.get("speed_kmh", 0),
            asset["id"], asset["type"], asset_class,
            sim_result["ai_ra"], sim_result["ai_confidence"],
            sim_result["sensor_ra"], sim_result["sensor_snr"],
            sim_result.get("retro_ra"), 1 if sim_result.get("retro_ra") else 0,
            fusion_result.final_ra,
            fusion_result.alpha, fusion_result.beta, fusion_result.gamma,
            fusion_result.ekf_variance, fusion_result.bias_estimate,
            sim_result["weather"], sim_result["ambient_lux"], status
        ))
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        return row_id

    def query_df(self, limit: int = 500, status_filter: list = None):
        import pandas as pd
        conn = sqlite3.connect(self.db_path)
        q = "SELECT * FROM measurements"
        if status_filter:
            placeholders = ",".join("?" * len(status_filter))
            q += f" WHERE status IN ({placeholders})"
            df = pd.read_sql(q + " ORDER BY timestamp DESC LIMIT ?",
                             conn, params=status_filter + [limit])
        else:
            df = pd.read_sql(q + f" ORDER BY timestamp DESC LIMIT {limit}", conn)
        conn.close()
        if not df.empty and "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def get_stats(self) -> dict:
        """Get summary statistics from the database."""
        import pandas as pd
        conn = sqlite3.connect(self.db_path)
        try:
            total = pd.read_sql("SELECT COUNT(*) as cnt FROM measurements", conn).iloc[0]["cnt"]
            if total == 0:
                return {"total": 0, "fail_rate": 0, "avg_ra": 0, "pass_count": 0,
                        "fail_count": 0, "marginal_count": 0}

            stats = pd.read_sql("""
                SELECT
                    COUNT(*) as total,
                    AVG(final_ra) as avg_ra,
                    SUM(CASE WHEN status='FAIL' THEN 1 ELSE 0 END) as fail_count,
                    SUM(CASE WHEN status='PASS' THEN 1 ELSE 0 END) as pass_count,
                    SUM(CASE WHEN status='MARGINAL' THEN 1 ELSE 0 END) as marginal_count
                FROM measurements
            """, conn).iloc[0]

            return {
                "total": int(stats["total"]),
                "fail_rate": float(stats["fail_count"] / stats["total"] * 100) if stats["total"] > 0 else 0,
                "avg_ra": float(stats["avg_ra"]) if stats["avg_ra"] else 0,
                "pass_count": int(stats["pass_count"]),
                "fail_count": int(stats["fail_count"]),
                "marginal_count": int(stats["marginal_count"]),
            }
        finally:
            conn.close()

    def get_rmse_ai_vs_retro(self) -> float:
        """Compute RMSE between AI and RetroMeter readings."""
        import pandas as pd
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql(
                "SELECT ai_ra, retro_ra FROM measurements WHERE retro_available=1 AND retro_ra IS NOT NULL",
                conn
            )
            if df.empty or len(df) < 2:
                return 0.0
            return float(np.sqrt(((df["ai_ra"] - df["retro_ra"]) ** 2).mean()))
        finally:
            conn.close()
