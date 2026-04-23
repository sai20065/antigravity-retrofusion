# main.py — RetroFusion AI+ Pro Entry Point
"""
RetroFusion AI+ Pro — Hybrid Retroreflectivity Measurement System

Full hardware pipeline (Raspberry Pi 4B):
    Camera → YOLOv8 Detection → Crop → MobileNetV2 RA Estimation
    → EKF Sensor Fusion → SQLite Logging → Dashboard

Modes:
    dashboard  — Launch Streamlit UI (default)
    headless   — Run fusion engine in terminal mode
    hardware   — Full hardware pipeline with real sensors
    simulate   — Full pipeline with all sensors simulated

Usage:
    python main.py --mode dashboard
    python main.py --mode headless --measurements 100
    python main.py --mode hardware
    python main.py --mode simulate --measurements 200
"""

import os
import sys
import time
import argparse
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.simulator import StreamingSimulator, ROAD_ASSETS, simulate_measurement
from modules.fusion_engine import ExtendedKalmanFilter, SensorMeasurement
from modules.data_logger import DataLogger
from config import DB_PATH


# ══════════════════════════════════════════════════════════════════════════
# HEADLESS MODE — Simulation only (original behavior)
# ══════════════════════════════════════════════════════════════════════════

def run_headless(n_measurements: int = 100, interval: float = 0.2):
    """
    Run fusion engine in headless mode for data collection.

    Args:
        n_measurements: Number of measurements to collect
        interval: Time between measurements in seconds
    """
    import random

    print("=" * 60)
    print("  RetroFusion AI+ Pro — Headless Mode")
    print("=" * 60)
    print(f"  Collecting {n_measurements} measurements...")
    print(f"  Database: {DB_PATH}")
    print("-" * 60)

    ekf = ExtendedKalmanFilter(dt=interval)
    logger = DataLogger(DB_PATH)

    pass_count = 0
    fail_count = 0
    marginal_count = 0

    for i in range(n_measurements):
        asset = random.choice(ROAD_ASSETS)
        night = (i % 100) > 70
        rain = (i % 50) > 40

        raw = simulate_measurement(asset, night=night, rain=rain)

        meas = SensorMeasurement(
            ai_ra=raw["ai_ra"],
            ai_confidence=raw["ai_confidence"],
            sensor_ra=raw["sensor_ra"],
            sensor_snr=raw["sensor_snr"],
            retro_ra=raw.get("retro_ra"),
            retro_available=True,
            weather=raw["weather"],
        )

        result = ekf.update(meas)
        row_id = logger.log(raw, result)

        # Determine status
        from config import THRESHOLDS
        thresholds = THRESHOLDS.get(asset["class"], {"pass": 100, "marginal": 70})
        if result.final_ra >= thresholds["pass"]:
            status = "PASS"
            pass_count += 1
        elif result.final_ra >= thresholds["marginal"]:
            status = "MARGINAL"
            marginal_count += 1
        else:
            status = "FAIL"
            fail_count += 1

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:>4}/{n_measurements}] {asset['id']} | "
                  f"RA={result.final_ra:>6.1f} | var={result.ekf_variance:>6.1f} | "
                  f"{status}")

        time.sleep(interval)

    total = pass_count + fail_count + marginal_count
    print("-" * 60)
    print(f"  Complete! {total} measurements logged.")
    print(f"  PASS: {pass_count} ({pass_count/total*100:.1f}%)")
    print(f"  MARGINAL: {marginal_count} ({marginal_count/total*100:.1f}%)")
    print(f"  FAIL: {fail_count} ({fail_count/total*100:.1f}%)")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
# HARDWARE MODE — Full Raspberry Pi Pipeline
# ══════════════════════════════════════════════════════════════════════════

def run_hardware(n_measurements: int = 0, interval: float = 0.033):
    """
    Full hardware pipeline as described in Design Document Appendix A.2.

    Pipeline per frame:
        1. camera.get_frame()
        2. ambient_lux = sensor.read_lux()
        3. rpi.set_ir_brightness(ambient_lux)
        4. detections = yolo.detect(frame)
        5. For each detection:
            a. crop = frame[det.y:det.y+det.h, det.x:det.x+det.w]
            b. ai_ra, ai_conf = ai.predict(crop)
            c. sensor_ra = sensor.compute_ra(reflected, ambient)
            d. retro_ra = retro.get_ra()
            e. final_ra = fusion.update(ai, sensor, retro)
            f. gps_fix = gps.get_fix()
            g. logger.log(...)
            h. gemma.analyze(...) (async, optional)
        6. rpi.alert_on_fail(status)

    Args:
        n_measurements: Number to collect (0 = infinite)
        interval:       Loop interval (0.033 = ~30fps)
    """
    from modules.rpi_controller import RPiController
    from modules.camera_capture import CameraCapture
    from modules.sensor_reader import BH1750Sensor
    from modules.gps_module import GPSModule
    from modules.retro_reader import RetroMeterReader
    from modules.yolo_detector import YOLODetector
    from modules.ai_inference import RAEstimator
    from modules.fusion_engine import ExtendedKalmanFilter, SensorMeasurement
    from modules.data_logger import DataLogger
    from modules.gemma_analyzer import GemmaAnalyzer
    from config import THRESHOLDS, MODEL_DIR

    print("=" * 60)
    print("  RetroFusion AI+ Pro — HARDWARE MODE")
    print("=" * 60)

    # ── Initialize all modules ────────────────────────────────────────────
    simulation = not os.path.exists("/proc/device-tree/model")

    rpi = RPiController(simulation=simulation)
    camera = CameraCapture(simulation=simulation)
    sensor = BH1750Sensor(simulation=simulation)
    gps = GPSModule(simulation=simulation)
    retro = RetroMeterReader(simulate=simulation)
    ai = RAEstimator(simulation=simulation)
    ekf = ExtendedKalmanFilter(dt=interval)
    logger = DataLogger(DB_PATH)
    gemma = GemmaAnalyzer()

    # Start background threads
    camera.start()
    gps.start()
    retro.start()

    print("\n[Main] All modules initialized:")
    print(f"  Camera:  {'SIM' if camera.simulation else 'HW'}")
    print(f"  Sensor:  {'SIM' if sensor.simulation else 'HW'}")
    print(f"  GPS:     {'SIM' if gps.simulation else 'HW'}")
    print(f"  Retro:   {'SIM' if retro.simulate else 'HW'}")
    print(f"  YOLO:    {'SIM' if (ai._yolo and ai._yolo.simulation) else 'HW'}")
    print(f"  AI:      {'SIM' if ai.simulation else 'HW'}")
    print(f"  Gemma:   {'API' if gemma.is_available else 'RULES'}")
    print(f"  RPi:     {'SIM' if rpi.simulation else 'HW'}")
    print("-" * 60)

    # ── Graceful shutdown ─────────────────────────────────────────────────
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("\n[Main] Shutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    # ── Main processing loop ──────────────────────────────────────────────
    count = 0
    pass_count = 0
    fail_count = 0
    marginal_count = 0

    while running:
        if n_measurements > 0 and count >= n_measurements:
            break

        # 1. Get camera frame
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # 2. Read ambient light
        ambient_lux = sensor.read_lux()

        # 3. Auto-adjust IR LED brightness
        rpi.set_ir_brightness(ambient_lux)

        # 4. Run AI pipeline (YOLO → crop → MobileNetV2)
        night = ambient_lux < 5
        rain = False  # TODO: weather sensor integration
        fog = False

        # Set simulation ground truth based on random asset
        import random
        asset = random.choice(ROAD_ASSETS)

        # For simulation, set the retro ground truth
        if retro.simulate:
            retro.set_sim_ra(asset["ra_true"])

        ai_results = ai.process_frame(
            frame,
            ra_true_per_detection={"speed_sign": asset["ra_true"],
                                    "direction_sign": asset["ra_true"],
                                    "lane_marking": asset["ra_true"],
                                    "stop_line": asset["ra_true"],
                                    "road_stud": asset["ra_true"],
                                    "zebra_crossing": asset["ra_true"]},
            night=night, rain=rain, fog=fog,
        )

        # 5. For each AI result, run sensor fusion
        for ai_result in ai_results:
            # Sensor RA
            sensor_result = sensor.compute_ra(
                reflected_lux=0, ambient_lux=ambient_lux,
                ra_true=asset["ra_true"], rain=rain, fog=fog,
            )

            # Retroreflectometer RA
            retro_ra = retro.get_ra()

            # Weather determination
            weather = "rain" if rain else ("fog" if fog else "clear")

            # EKF Fusion
            meas = SensorMeasurement(
                ai_ra=ai_result.ra_estimate,
                ai_confidence=ai_result.confidence,
                sensor_ra=sensor_result["ra_estimate"],
                sensor_snr=sensor_result["snr"],
                retro_ra=retro_ra,
                retro_available=retro.available,
                weather=weather,
            )
            fusion_result = ekf.update(meas)

            # Determine status
            asset_class = ai_result.class_name
            if asset_class in ["speed_sign", "direction_sign"]:
                threshold_key = "sign_RA2" if asset_class == "speed_sign" else "sign_RA1"
            elif asset_class in ["lane_marking", "zebra_crossing"]:
                threshold_key = "marking_R2"
            elif asset_class == "stop_line":
                threshold_key = "marking_R1"
            elif asset_class == "road_stud":
                threshold_key = "stud_typeI"
            else:
                threshold_key = asset.get("class", "sign_RA2")

            thresholds = THRESHOLDS.get(threshold_key, {"pass": 100, "marginal": 70})
            if fusion_result.final_ra >= thresholds["pass"]:
                status = "PASS"
                pass_count += 1
            elif fusion_result.final_ra >= thresholds["marginal"]:
                status = "MARGINAL"
                marginal_count += 1
            else:
                status = "FAIL"
                fail_count += 1

            # 6. Get GPS fix
            gps_fix = gps.get_fix()

            # 7. Log to database
            sim_result = {
                "asset": asset,
                "timestamp": time.time(),
                "lat": gps_fix.latitude if gps_fix else asset["lat"],
                "lon": gps_fix.longitude if gps_fix else asset["lon"],
                "speed_kmh": gps_fix.speed_kmh if gps_fix else 0,
                "ai_ra": ai_result.ra_estimate,
                "ai_confidence": ai_result.confidence,
                "sensor_ra": sensor_result["ra_estimate"],
                "sensor_snr": sensor_result["snr"],
                "retro_ra": retro_ra,
                "ambient_lux": ambient_lux,
                "weather": weather,
            }
            logger.log(sim_result, fusion_result)

            # 8. Update LEDs
            rpi.alert_on_fail(status)

            count += 1
            if count % 10 == 0 or count == 1:
                print(f"  [{count:>5}] {ai_result.class_name:<16} | "
                      f"AI={ai_result.ra_estimate:>6.1f} | "
                      f"Sensor={sensor_result['ra_estimate']:>6.1f} | "
                      f"Fused={fusion_result.final_ra:>6.1f} | "
                      f"{status}")

        time.sleep(interval)

    # ── Cleanup ───────────────────────────────────────────────────────────
    camera.stop()
    gps.stop()
    retro.stop()
    rpi.cleanup()

    total = pass_count + fail_count + marginal_count
    if total > 0:
        print("\n" + "-" * 60)
        print(f"  Complete! {total} measurements logged.")
        print(f"  PASS: {pass_count} ({pass_count/total*100:.1f}%)")
        print(f"  MARGINAL: {marginal_count} ({marginal_count/total*100:.1f}%)")
        print(f"  FAIL: {fail_count} ({fail_count/total*100:.1f}%)")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD MODE
# ══════════════════════════════════════════════════════════════════════════

def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    subprocess.run(["streamlit", "run", dashboard_path])


# ══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetroFusion AI+ Pro")
    parser.add_argument("--mode", choices=["dashboard", "headless", "hardware", "simulate"],
                        default="dashboard", help="Run mode")
    parser.add_argument("--measurements", type=int, default=100,
                        help="Number of measurements (headless/hardware/simulate mode)")
    parser.add_argument("--interval", type=float, default=0.2,
                        help="Measurement interval seconds")
    args = parser.parse_args()

    if args.mode == "headless":
        run_headless(args.measurements, args.interval)
    elif args.mode in ("hardware", "simulate"):
        run_hardware(args.measurements, args.interval)
    else:
        run_dashboard()
