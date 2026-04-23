# modules/__init__.py
# RetroFusion AI+ Pro — Module Registry
#
# Core modules:
#   fusion_engine   — EKF + Confidence-Weighted Sensor Fusion
#   simulator       — Virtual sensor simulator for demo/dev
#   data_logger     — SQLite operations
#   predictive      — Degradation model + failure prediction
#   ai_inference    — Full AI pipeline: YOLOv8 + MobileNetV2 RA
#   sensor_reader   — BH1750 physics sensor model
#
# Hardware modules (new):
#   camera_capture  — OpenCV frame acquisition + RPi Camera
#   rpi_controller  — GPIO, IR LED PWM, Status LEDs
#   gps_module      — UART NEO-M8N NMEA parser
#   retro_reader    — Retroreflectometer serial interface
#   yolo_detector   — YOLOv8-nano object detection
#   gemma_analyzer  — Gemma 4 AI defect analysis

__all__ = [
    "fusion_engine",
    "simulator",
    "data_logger",
    "predictive",
    "ai_inference",
    "sensor_reader",
    "camera_capture",
    "rpi_controller",
    "gps_module",
    "retro_reader",
    "yolo_detector",
    "gemma_analyzer",
]
