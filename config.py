# config.py — RetroFusion AI+ Pro Central Configuration

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_DIR, "retrofusion.db")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

# ── Hardware GPIO (Raspberry Pi 4B) ───────────────────────────────────────
GPIO_IR_PWM    = 18   # PWM pin for IR LED brightness (1kHz)
GPIO_IR_ENABLE = 17   # IR LED enable (active HIGH)
GPIO_STATUS_LED = 27  # Green status LED
GPIO_ALERT_LED  = 22  # Red alert LED
BH1750_I2C_ADDR = 0x23
GPS_UART_BAUD   = 9600
RETRO_USB_PORT  = "/dev/ttyUSB0"
RETRO_BAUD      = 9600

# ── Camera ─────────────────────────────────────────────────────────────────
CAMERA_WIDTH  = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS    = 30
YOLO_INPUT_SIZE = 640
MOBILENET_INPUT_SIZE = 224

# ── IR LED Physics ─────────────────────────────────────────────────────────
IR_EMITTANCE_mW = 48000   # Total IR power in mW (48W array)
SPOT_AREA_m2    = 0.04    # Illuminated spot: 0.2m × 0.2m
SENSOR_DIST_m   = 0.35    # Sensor-to-target distance
OBS_ANGLE_DEG   = 0.2     # Observation angle for retro geometry

# ── EKF Sensor Noise (mcd/lux/m²) ─────────────────────────────────────────
SIGMA_AI     = 45.0   # AI model base noise
SIGMA_SENSOR = 25.0   # Physics sensor base noise
SIGMA_RETRO  = 8.0    # Retroreflectometer base noise
SIGMA_RA_PROCESS = 15.0  # Process noise (RA rate of change)

# ── RA Thresholds (EN 12899-1, EN 1436, EN 1463-1) ────────────────────────
THRESHOLDS = {
    "sign_RA2":     {"pass": 150, "marginal": 100},  # Highway signs
    "sign_RA1":     {"pass": 70,  "marginal": 50},   # Local road signs
    "marking_R2":   {"pass": 150, "marginal": 100},  # Wet night marking
    "marking_R1":   {"pass": 100, "marginal": 70},   # Dry night marking
    "stud_typeI":   {"pass": 70,  "marginal": 50},   # Passive road stud
    "stud_typeII":  {"pass": 300, "marginal": 200},  # Active LED stud
}

# ── Fusion Weights (steady-state EKF approximation) ───────────────────────
DEFAULT_WEIGHTS = {"retro": 0.65, "ai": 0.22, "sensor": 0.13}

# ── AI Bias Correction ─────────────────────────────────────────────────────
BIAS_LAMBDA = 0.05  # EMA smoothing for bias tracking

# ── Predictive Maintenance ─────────────────────────────────────────────────
DEGRADATION_RATES = {
    "painted":     0.003,   # per month
    "type_i":      0.0015,
    "type_ii":     0.001,
    "type_iii":    0.0008,
    "microprismatic": 0.0006,
}

# ── Weather Noise Multipliers ──────────────────────────────────────────────
WEATHER_MULTIPLIERS = {
    "clear":  {"ai": 1.0, "sensor": 1.0, "retro": 1.0},
    "rain":   {"ai": 2.5, "sensor": 3.0, "retro": 1.2},
    "fog":    {"ai": 4.0, "sensor": 2.0, "retro": 1.1},
    "snow":   {"ai": 5.0, "sensor": 2.5, "retro": 1.3},
}

# ── Cloud API (optional) ───────────────────────────────────────────────────
CLOUD_API_URL = "https://api.retrofusion.cloud/v1/measurements"
CLOUD_SYNC_INTERVAL = 30  # seconds

# ── Dashboard ──────────────────────────────────────────────────────────────
DASHBOARD_REFRESH_SEC = 5
MAX_DISPLAY_ROWS = 500
