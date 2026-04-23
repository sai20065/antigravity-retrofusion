# modules/yolo_detector.py
# YOLOv8-nano Road Asset Detection Module
#
# Design reference: Section 4.1 of RetroFusion Design Document.
#
# Detection classes (6 road asset categories):
#   0: speed_sign      — Speed limit signs
#   1: direction_sign  — Direction/information signs
#   2: lane_marking    — Lane dividers, center lines
#   3: stop_line       — Stop lines, crosswalk edges
#   4: road_stud       — Passive/active road studs
#   5: zebra_crossing  — Zebra/pedestrian crossings
#
# Pipeline:
#   Camera frame (1920×1080) → Resize (640×640) → YOLOv8-nano
#   → Bounding boxes [x,y,w,h,cls,conf] → Crop extraction
#   → Each crop → MobileNetV2 RA estimation
#
# Performance targets (from design doc):
#   mAP@0.5:        > 0.82
#   Inference (RPi): ~85ms (INT8 TFLite with XNNPACK)
#   Model size:      ~3.2MB (INT8)

import numpy as np
import time
import threading
from typing import List, Optional, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import YOLO_INPUT_SIZE, CAMERA_WIDTH, CAMERA_HEIGHT


# Detection class mapping
ASSET_CLASSES = {
    0: "speed_sign",
    1: "direction_sign",
    2: "lane_marking",
    3: "stop_line",
    4: "road_stud",
    5: "zebra_crossing",
}

# Map detection class to config asset_class for threshold lookup
DETECTION_TO_ASSET_CLASS = {
    "speed_sign":     "sign_RA2",
    "direction_sign": "sign_RA1",
    "lane_marking":   "marking_R2",
    "stop_line":      "marking_R1",
    "road_stud":      "stud_typeI",
    "zebra_crossing": "marking_R2",
}


@dataclass
class Detection:
    """Single object detection result."""
    x: int              # Bounding box top-left x (in original frame coords)
    y: int              # Bounding box top-left y
    w: int              # Bounding box width
    h: int              # Bounding box height
    class_id: int       # Class index (0-5)
    class_name: str     # Human-readable class name
    confidence: float   # Detection confidence (0-1)
    asset_class: str    # Mapped config asset class for threshold lookup

    def get_crop(self, frame: np.ndarray) -> np.ndarray:
        """Extract crop from original frame with padding."""
        fh, fw = frame.shape[:2]
        # Add 10% padding for context
        pad_x = int(self.w * 0.1)
        pad_y = int(self.h * 0.1)
        x1 = max(0, self.x - pad_x)
        y1 = max(0, self.y - pad_y)
        x2 = min(fw, self.x + self.w + pad_x)
        y2 = min(fh, self.y + self.h + pad_y)
        return frame[y1:y2, x1:x2].copy()


class YOLODetector:
    """
    YOLOv8-nano road asset detector.

    Modes:
        1. Ultralytics mode: Uses ultralytics YOLO library (dev/GPU machines)
        2. TFLite mode: Uses tflite-runtime (Raspberry Pi deployment)
        3. Simulation mode: Generates realistic detections for demo

    Design reference: Section 4.1 of RetroFusion Design Document.
    """

    def __init__(self, model_path: str = None,
                 tflite_path: str = None,
                 simulation: bool = True,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.45):
        """
        Args:
            model_path:            Path to YOLOv8 .pt or .onnx model
            tflite_path:           Path to TFLite INT8 model (for RPi)
            simulation:            Generate fake detections
            confidence_threshold:  Minimum confidence to keep detection
            nms_threshold:         NMS IoU threshold
        """
        self.simulation = simulation
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self._model = None
        self._tflite_interpreter = None
        self._inference_count = 0
        self._total_latency = 0.0
        self._lock = threading.Lock()

        if not simulation:
            if tflite_path and os.path.exists(tflite_path):
                self._init_tflite(tflite_path)
            elif model_path and os.path.exists(model_path):
                self._init_ultralytics(model_path)
            else:
                print("[YOLO] No model file found. Falling back to simulation.")
                self.simulation = True
        
        if self.simulation:
            print("[YOLO] Simulation mode — generating synthetic detections")

    def _init_ultralytics(self, model_path: str):
        """Load YOLOv8 model via ultralytics library."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            print(f"[YOLO] Loaded ultralytics model: {model_path}")
        except ImportError:
            print("[YOLO] ultralytics not installed. Falling back to simulation.")
            self.simulation = True
        except Exception as e:
            print(f"[YOLO] Model load failed: {e}. Falling back to simulation.")
            self.simulation = True

    def _init_tflite(self, tflite_path: str):
        """Load TFLite INT8 model for Raspberry Pi deployment."""
        try:
            import tflite_runtime.interpreter as tflite
            self._tflite_interpreter = tflite.Interpreter(
                model_path=tflite_path,
                num_threads=4,
            )
            self._tflite_interpreter.allocate_tensors()
            print(f"[YOLO] Loaded TFLite model: {tflite_path}")
            print(f"       Input: {self._tflite_interpreter.get_input_details()[0]['shape']}")
        except ImportError:
            print("[YOLO] tflite-runtime not installed. Falling back to simulation.")
            self.simulation = True
        except Exception as e:
            print(f"[YOLO] TFLite load failed: {e}. Falling back to simulation.")
            self.simulation = True

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on a camera frame.

        Args:
            frame: BGR ndarray (H, W, 3) — raw camera frame

        Returns:
            List of Detection objects with bounding boxes and classes
        """
        t0 = time.time()

        if self.simulation:
            detections = self._simulate_detections(frame)
        elif self._model is not None:
            detections = self._detect_ultralytics(frame)
        elif self._tflite_interpreter is not None:
            detections = self._detect_tflite(frame)
        else:
            detections = []

        latency = (time.time() - t0) * 1000
        with self._lock:
            self._inference_count += 1
            self._total_latency += latency

        return detections

    def _detect_ultralytics(self, frame: np.ndarray) -> List[Detection]:
        """Run detection using ultralytics YOLO library."""
        results = self._model(frame, conf=self.conf_threshold,
                               iou=self.nms_threshold, verbose=False)
        detections = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if cls_id not in ASSET_CLASSES:
                    continue

                cls_name = ASSET_CLASSES[cls_id]
                asset_cls = DETECTION_TO_ASSET_CLASS.get(cls_name, "sign_RA2")

                detections.append(Detection(
                    x=x1, y=y1,
                    w=x2 - x1, h=y2 - y1,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    asset_class=asset_cls,
                ))

        return detections

    def _detect_tflite(self, frame: np.ndarray) -> List[Detection]:
        """Run detection using TFLite runtime (for Raspberry Pi)."""
        import cv2

        # Preprocess
        input_details = self._tflite_interpreter.get_input_details()
        output_details = self._tflite_interpreter.get_output_details()
        
        input_shape = input_details[0]['shape']
        input_h, input_w = input_shape[1], input_shape[2]
        
        img_resized = cv2.resize(frame, (input_w, input_h))
        
        # Normalize and set input
        input_type = input_details[0]['dtype']
        if input_type == np.int8:
            # INT8 quantized model
            scale, zero_point = input_details[0]['quantization']
            img_input = ((img_resized.astype(np.float32) / 255.0 - 0.5) / scale + zero_point).astype(np.int8)
        else:
            img_input = (img_resized.astype(np.float32) / 255.0)
        
        img_input = np.expand_dims(img_input, axis=0)
        self._tflite_interpreter.set_tensor(input_details[0]['index'], img_input)
        self._tflite_interpreter.invoke()

        # Parse output (format depends on export)
        output = self._tflite_interpreter.get_tensor(output_details[0]['index'])[0]

        detections = []
        fh, fw = frame.shape[:2]
        
        for detection in output:
            if len(detection) >= 6:
                x_center, y_center, w, h = detection[:4]
                conf = detection[4] if len(detection) > 4 else 0
                cls_id = int(detection[5]) if len(detection) > 5 else 0
                
                if conf < self.conf_threshold:
                    continue
                if cls_id not in ASSET_CLASSES:
                    continue

                # Convert from normalized to pixel coordinates
                x1 = int((x_center - w/2) * fw)
                y1 = int((y_center - h/2) * fh)
                bw = int(w * fw)
                bh = int(h * fh)

                cls_name = ASSET_CLASSES[cls_id]
                asset_cls = DETECTION_TO_ASSET_CLASS.get(cls_name, "sign_RA2")

                detections.append(Detection(
                    x=max(0, x1), y=max(0, y1),
                    w=bw, h=bh,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=float(conf),
                    asset_class=asset_cls,
                ))

        return detections

    def _simulate_detections(self, frame: np.ndarray) -> List[Detection]:
        """
        Generate realistic synthetic detections for simulation.

        Simulates:
            - 1-3 detections per frame
            - Realistic bounding box sizes and positions
            - Detection confidence variation by class and position
            - Occasional missed detections (realistic recall)
        """
        fh, fw = frame.shape[:2] if frame is not None else (CAMERA_HEIGHT, CAMERA_WIDTH)
        detections = []

        # Number of detections per frame (weighted toward 1-2)
        n_detections = np.random.choice([0, 1, 2, 3], p=[0.05, 0.45, 0.35, 0.15])

        for _ in range(n_detections):
            cls_id = np.random.choice(list(ASSET_CLASSES.keys()),
                                       p=[0.25, 0.15, 0.25, 0.10, 0.10, 0.15])
            cls_name = ASSET_CLASSES[cls_id]

            # Realistic bounding box sizes per class
            if "sign" in cls_name:
                w = np.random.randint(80, 200)
                h = np.random.randint(80, 200)
                x = np.random.randint(fw // 4, 3 * fw // 4)
                y = np.random.randint(int(fh * 0.1), int(fh * 0.5))
            elif "marking" in cls_name or "stop" in cls_name:
                w = np.random.randint(150, 400)
                h = np.random.randint(30, 80)
                x = np.random.randint(fw // 4, 3 * fw // 4)
                y = np.random.randint(int(fh * 0.5), int(fh * 0.85))
            elif "stud" in cls_name:
                w = np.random.randint(15, 40)
                h = np.random.randint(15, 40)
                x = np.random.randint(fw // 3, 2 * fw // 3)
                y = np.random.randint(int(fh * 0.4), int(fh * 0.8))
            else:  # zebra
                w = np.random.randint(200, 500)
                h = np.random.randint(80, 150)
                x = np.random.randint(fw // 4, fw // 2)
                y = np.random.randint(int(fh * 0.5), int(fh * 0.8))

            # Confidence varies by class and randomness
            base_conf = {
                "speed_sign": 0.85, "direction_sign": 0.78,
                "lane_marking": 0.82, "stop_line": 0.75,
                "road_stud": 0.70, "zebra_crossing": 0.80,
            }
            conf = base_conf.get(cls_name, 0.75) + np.random.uniform(-0.15, 0.10)
            conf = max(self.conf_threshold, min(0.99, conf))

            asset_cls = DETECTION_TO_ASSET_CLASS.get(cls_name, "sign_RA2")

            # Ensure bounding box is within frame
            x = min(x, fw - w - 1)
            y = min(y, fh - h - 1)

            detections.append(Detection(
                x=max(0, x), y=max(0, y),
                w=w, h=h,
                class_id=cls_id,
                class_name=cls_name,
                confidence=float(conf),
                asset_class=asset_cls,
            ))

        # Simulate inference latency (realistic: 80-120ms on RPi, 15-30ms on GPU)
        time.sleep(np.random.uniform(0.015, 0.035))

        return detections

    def draw_detections(self, frame: np.ndarray,
                        detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame for visualization.

        Args:
            frame:      BGR ndarray
            detections: List of Detection objects

        Returns:
            Annotated frame copy
        """
        import cv2

        annotated = frame.copy()
        colors = {
            "speed_sign": (0, 128, 255),      # Orange
            "direction_sign": (255, 128, 0),   # Blue
            "lane_marking": (0, 255, 128),     # Green
            "stop_line": (0, 0, 255),          # Red
            "road_stud": (255, 255, 0),        # Cyan
            "zebra_crossing": (128, 0, 255),   # Purple
        }

        for det in detections:
            color = colors.get(det.class_name, (255, 255, 255))
            cv2.rectangle(annotated, (det.x, det.y),
                           (det.x + det.w, det.y + det.h), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(annotated, label, (det.x, det.y - 8),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return annotated

    def get_stats(self) -> dict:
        """Return detection statistics."""
        with self._lock:
            avg_latency = (self._total_latency / max(1, self._inference_count))
            return {
                "mode": "simulation" if self.simulation else ("tflite" if self._tflite_interpreter else "ultralytics"),
                "total_inferences": self._inference_count,
                "avg_latency_ms": round(avg_latency, 2),
                "confidence_threshold": self.conf_threshold,
                "num_classes": len(ASSET_CLASSES),
                "classes": list(ASSET_CLASSES.values()),
            }
