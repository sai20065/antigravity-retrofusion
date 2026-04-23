# modules/ai_inference.py
# Full AI Pipeline — YOLOv8 Detection → Crop → MobileNetV2 RA Estimation
#
# Design reference: Sections 3.2, 4.1, 4.2 of RetroFusion Design Document.
#
# Pipeline:
#   1. Camera frame (1920×1080 BGR) arrives
#   2. YOLOv8-nano detects road assets → bounding boxes
#   3. Each detection is cropped from the original frame
#   4. Crop is resized to 224×224 and fed to MobileNetV2
#   5. MobileNetV2 outputs: (RA_estimate, confidence)
#   6. Results are returned per-detection for fusion
#
# Modes:
#   - Production: YOLOv8 + MobileNetV2 (PyTorch or TFLite)
#   - Simulation: Realistic noise + bias model (no actual inference)

import numpy as np
import time
import os
import sys
from typing import List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_DIR, MOBILENET_INPUT_SIZE


@dataclass
class AIResult:
    """Result from the full AI pipeline for one detection."""
    ra_estimate: float      # Predicted RA (mcd/lux/m²)
    confidence: float       # Model confidence (0-1)
    class_name: str         # Detected asset class
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) bounding box
    latency_ms: float       # Inference latency
    detection_type: str     # "yolo+mobilenet", "simulation", etc.


class RAEstimator:
    """
    AI-based retroreflectivity estimator.

    Full pipeline:
        Frame → YOLOv8 detection → crop extraction → MobileNetV2 RA regression

    In production mode:
        - Loads MobileNetV2 + custom regression head from checkpoint
        - Runs inference on 224×224 road sign/marking crops
        - Returns RA estimate + confidence score

    In simulation mode:
        - Returns realistic estimates with configurable noise
        - Simulates inference latency
    """

    def __init__(self, model_path: str = None, simulation: bool = True):
        self.simulation = simulation
        self.model = None
        self.device = "cpu"
        self._inference_count = 0
        self._total_latency = 0.0

        # Try to load the YOLO detector
        self._yolo = None
        self._init_yolo()

        if not simulation and model_path and os.path.exists(model_path):
            try:
                import torch
                from training.train_ra_model import MobileNetV2_RA
                self.model = MobileNetV2_RA(pretrained=False)
                checkpoint = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                self.simulation = False
                print(f"[AI] MobileNetV2-RA model loaded from {model_path}")
            except Exception as e:
                print(f"[AI] Failed to load model: {e}. Falling back to simulation.")
                self.simulation = True

    def _init_yolo(self):
        """Initialize YOLO detector if available."""
        try:
            from modules.yolo_detector import YOLODetector

            # Try to find a trained model
            yolo_paths = [
                os.path.join(MODEL_DIR, "yolov8n_signs.pt"),
                os.path.join(MODEL_DIR, "yolov8n_signs.tflite"),
                os.path.join(MODEL_DIR, "best.pt"),
                "runs/detect/road_assets/weights/best.pt",
            ]

            for path in yolo_paths:
                if os.path.exists(path):
                    if path.endswith(".tflite"):
                        self._yolo = YOLODetector(tflite_path=path, simulation=False)
                    else:
                        self._yolo = YOLODetector(model_path=path, simulation=False)
                    print(f"[AI] YOLO detector loaded: {path}")
                    return

            # No model found — use simulation
            self._yolo = YOLODetector(simulation=True)
            print("[AI] YOLO detector: simulation mode")

        except ImportError:
            print("[AI] YOLODetector not available")
            self._yolo = None

    def process_frame(self, frame: np.ndarray,
                      ra_true_per_detection: dict = None,
                      night: bool = False, rain: bool = False,
                      fog: bool = False) -> List[AIResult]:
        """
        Process a full camera frame through the AI pipeline.

        Pipeline:
            1. Run YOLOv8 detection on frame
            2. For each detection, crop the region
            3. Run MobileNetV2 RA estimation on each crop
            4. Return list of AIResult objects

        Args:
            frame:                   BGR ndarray (H, W, 3)
            ra_true_per_detection:   Dict mapping class_name → true RA (simulation only)
            night:                   Night condition flag
            rain:                    Rain condition flag
            fog:                     Fog condition flag

        Returns:
            List[AIResult] — one per detected road asset
        """
        t0 = time.time()
        results = []

        # Step 1: Run YOLO detection
        detections = []
        if self._yolo:
            detections = self._yolo.detect(frame)

        if not detections:
            # If no detections, run single prediction (fallback)
            result = self.predict(
                image=None, ra_true=None, night=night, rain=rain, fog=fog
            )
            results.append(AIResult(
                ra_estimate=result["ra_estimate"],
                confidence=result["confidence"],
                class_name="unknown",
                bbox=(0, 0, 0, 0),
                latency_ms=result["latency_ms"],
                detection_type=result.get("detection_type", "simulation"),
            ))
            return results

        # Step 2 & 3: Crop + MobileNetV2 for each detection
        for det in detections:
            crop = det.get_crop(frame)

            # Get true RA for simulation mode
            ra_true = None
            if ra_true_per_detection:
                ra_true = ra_true_per_detection.get(det.class_name)

            result = self.predict(
                image=crop, ra_true=ra_true,
                night=night, rain=rain, fog=fog
            )

            results.append(AIResult(
                ra_estimate=result["ra_estimate"],
                confidence=result["confidence"] * det.confidence,  # Combined confidence
                class_name=det.class_name,
                bbox=(det.x, det.y, det.w, det.h),
                latency_ms=result["latency_ms"],
                detection_type=f"yolo+{result.get('detection_type', 'mobilenet')}",
            ))

        total_latency = (time.time() - t0) * 1000
        return results

    def predict(self, image=None, ra_true: float = None,
                night: bool = False, rain: bool = False,
                fog: bool = False) -> dict:
        """
        Run inference on an image crop.

        Args:
            image:   PIL Image or numpy array (224×224×3). Ignored in simulation.
            ra_true: Ground truth RA (only used in simulation mode).
            night:   Night condition flag.
            rain:    Rain condition flag.
            fog:     Fog condition flag.

        Returns:
            dict with keys: ra_estimate, confidence, latency_ms, model_version
        """
        t0 = time.time()

        if self.simulation:
            result = self._simulate_inference(ra_true, night, rain, fog)
        else:
            result = self._real_inference(image)

        latency = (time.time() - t0) * 1000
        self._inference_count += 1
        self._total_latency += latency

        result["latency_ms"] = round(latency, 2)
        result["model_version"] = "sim-v1.0" if self.simulation else "mobilenetv2-ra-v1.0"
        result["inference_count"] = self._inference_count
        return result

    def _simulate_inference(self, ra_true: float, night: bool,
                            rain: bool, fog: bool) -> dict:
        """Generate realistic AI inference output with noise and bias."""
        if ra_true is None:
            ra_true = np.random.uniform(50, 500)

        # Condition-dependent noise and bias
        bias = 15.0 if night else (8.0 if fog else 5.0)
        noise_std = 45.0 if rain else (35.0 if fog else 20.0)

        ra_estimate = ra_true + bias + np.random.normal(0, noise_std)
        ra_estimate = max(0.0, ra_estimate)

        # Confidence depends on conditions
        if rain or fog:
            confidence = np.random.uniform(0.35, 0.55)
        elif night:
            confidence = np.random.uniform(0.55, 0.75)
        else:
            confidence = np.random.uniform(0.75, 0.92)

        # Simulate processing time (15-45ms)
        time.sleep(np.random.uniform(0.015, 0.045))

        return {
            "ra_estimate": float(ra_estimate),
            "confidence": float(confidence),
            "detection_type": "simulated",
        }

    def _real_inference(self, image) -> dict:
        """Run actual MobileNetV2 inference on a crop."""
        import torch
        from torchvision import transforms
        import cv2

        preprocess = transforms.Compose([
            transforms.Resize((MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        if image is not None:
            from PIL import Image

            # Handle both numpy arrays and PIL images
            if isinstance(image, np.ndarray):
                # OpenCV BGR → RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)

            input_tensor = preprocess(image).unsqueeze(0).to(self.device)
        else:
            input_tensor = torch.randn(1, 3, MOBILENET_INPUT_SIZE,
                                        MOBILENET_INPUT_SIZE).to(self.device)

        with torch.no_grad():
            ra_pred, conf_pred = self.model(input_tensor)

        return {
            "ra_estimate": float(ra_pred.item()),
            "confidence": float(conf_pred.item()),
            "detection_type": "mobilenetv2",
        }

    def get_yolo_detector(self):
        """Get the internal YOLO detector for direct access."""
        return self._yolo

    def get_stats(self) -> dict:
        """Return inference performance statistics."""
        avg_latency = (self._total_latency / self._inference_count
                       if self._inference_count > 0 else 0)
        stats = {
            "total_inferences": self._inference_count,
            "avg_latency_ms": round(avg_latency, 2),
            "mode": "simulation" if self.simulation else "production",
            "has_yolo": self._yolo is not None,
        }
        if self._yolo:
            stats["yolo_stats"] = self._yolo.get_stats()
        return stats
