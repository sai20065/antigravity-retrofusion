# modules/camera_capture.py
# Camera Frame Acquisition — OpenCV + RPi Camera Module 3
#
# In production (Raspberry Pi):
#   - Pi Camera Module 3 via CSI-2 ribbon cable
#   - OpenCV VideoCapture(0) at 1920×1080 @ 30fps
#   - Threaded capture loop for non-blocking frame access
#   - Auto IR LED brightness adjustment via RPiController
#
# In simulation (development):
#   - Generates synthetic road scene frames with OpenCV
#   - Draws simulated road signs, markings, and studs
#   - Adds realistic noise, blur, and lighting effects

import cv2
import numpy as np
import time
import threading
from typing import Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, YOLO_INPUT_SIZE


class CameraCapture:
    """
    OpenCV-based camera frame acquisition with threading.

    Design reference: Section 3.2 of RetroFusion Design Document.

    Pipeline:
        Camera → CSI/USB → OpenCV VideoCapture → frame buffer (thread-safe)
        → get_frame() returns latest frame → YOLO detection pipeline

    Supports:
        - Pi Camera Module 3 (via CSI-2, appears as /dev/video0)
        - USB webcams (development/testing)
        - Simulation mode (synthetic frames for demo)
    """

    def __init__(self, camera_id: int = 0,
                 resolution: Tuple[int, int] = None,
                 fps: int = None,
                 simulation: bool = False):
        """
        Initialize camera capture.

        Args:
            camera_id:  OpenCV camera index (0 for Pi Camera / first USB cam)
            resolution: (width, height) tuple, defaults to config values
            fps:        Target framerate, defaults to config value
            simulation: If True, generate synthetic frames instead of real capture
        """
        self.width = resolution[0] if resolution else CAMERA_WIDTH
        self.height = resolution[1] if resolution else CAMERA_HEIGHT
        self.fps = fps or CAMERA_FPS
        self.simulation = simulation

        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._frame_count = 0
        self._cap = None
        self._start_time = time.time()

        if not simulation:
            self._init_camera(camera_id)
        else:
            print(f"[Camera] Simulation mode — generating synthetic {self.width}×{self.height} frames")

    def _init_camera(self, camera_id: int):
        """Initialize OpenCV VideoCapture."""
        try:
            self._cap = cv2.VideoCapture(camera_id)
            if not self._cap.isOpened():
                print(f"[Camera] Failed to open camera {camera_id}. Falling back to simulation.")
                self.simulation = True
                return

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Read actual values (camera may not support requested resolution)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))

            print(f"[Camera] Opened camera {camera_id}: {actual_w}×{actual_h} @ {actual_fps}fps")
            self.width = actual_w
            self.height = actual_h

        except Exception as e:
            print(f"[Camera] Camera init failed: {e}. Falling back to simulation.")
            self.simulation = True

    def start(self):
        """Start the background capture thread."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        threading.Thread(target=self._capture_loop, daemon=True).start()
        print("[Camera] Capture thread started")

    def stop(self):
        """Stop the capture thread."""
        self._running = False
        if self._cap and self._cap.isOpened():
            self._cap.release()
            print("[Camera] Camera released")

    def _capture_loop(self):
        """Background thread: continuously capture frames."""
        while self._running:
            if self.simulation:
                frame = self._generate_synthetic_frame()
            else:
                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

            with self._lock:
                self._frame = frame
                self._frame_count += 1

            # Rate limiting
            time.sleep(1.0 / self.fps)

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame (thread-safe).

        Returns:
            BGR ndarray (H, W, 3) or None if no frame available
        """
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def get_frame_for_yolo(self) -> Optional[np.ndarray]:
        """
        Get frame resized for YOLOv8 input (640×640).

        Returns:
            BGR ndarray (640, 640, 3) or None
        """
        frame = self.get_frame()
        if frame is None:
            return None
        return cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))

    def get_frame_count(self) -> int:
        """Return total number of frames captured."""
        return self._frame_count

    def get_fps_actual(self) -> float:
        """Return actual achieved FPS."""
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            return self._frame_count / elapsed
        return 0.0

    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Generate a synthetic road scene frame for simulation.

        Creates a dark road background with:
        - Road lanes with markings
        - Simulated road signs (rectangles with text)
        - Random road studs
        - Realistic noise and lighting variation
        """
        # Create dark road background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Road surface (dark gray with slight variation)
        road_color = np.random.randint(35, 55)
        frame[:, :] = (road_color, road_color, road_color)

        # Road center — lighter strip
        cx = self.width // 2
        cv2.rectangle(frame, (cx - 400, 0), (cx + 400, self.height),
                       (road_color + 10, road_color + 10, road_color + 10), -1)

        # Lane markings (white dashed lines)
        for y in range(0, self.height, 80):
            if (y // 80) % 2 == 0:
                cv2.rectangle(frame,
                              (cx - 3, y), (cx + 3, y + 40),
                              (200, 200, 200), -1)

        # Simulated road sign (retroreflective rectangle)
        t = self._frame_count % 200
        sign_x = int(cx + 200 + 100 * np.sin(t * 0.05))
        sign_y = int(self.height * 0.2 + 50 * np.sin(t * 0.03))
        sign_w, sign_h = 120, 80

        # Sign background (retroreflective — bright when illuminated)
        brightness = np.random.randint(150, 250)
        cv2.rectangle(frame,
                       (sign_x, sign_y), (sign_x + sign_w, sign_y + sign_h),
                       (brightness, brightness, brightness), -1)
        cv2.rectangle(frame,
                       (sign_x, sign_y), (sign_x + sign_w, sign_y + sign_h),
                       (255, 50, 50), 2)  # Red border
        cv2.putText(frame, "60", (sign_x + 30, sign_y + 55),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        # Road studs (small reflective dots)
        for i in range(5):
            stud_x = cx + np.random.randint(-300, 300)
            stud_y = int(self.height * (0.5 + i * 0.1))
            stud_brightness = np.random.randint(100, 200)
            cv2.circle(frame, (stud_x, stud_y), 5,
                        (stud_brightness, stud_brightness, 0), -1)

        # Zebra crossing
        zebra_y = int(self.height * 0.7)
        for i in range(8):
            x = cx - 200 + i * 50
            cv2.rectangle(frame, (x, zebra_y), (x + 30, zebra_y + 60),
                           (220, 220, 220), -1)

        # Add sensor noise
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return frame

    def get_stats(self) -> dict:
        """Return capture statistics."""
        return {
            "mode": "simulation" if self.simulation else "hardware",
            "resolution": f"{self.width}×{self.height}",
            "target_fps": self.fps,
            "actual_fps": round(self.get_fps_actual(), 1),
            "total_frames": self._frame_count,
        }

    def __del__(self):
        self.stop()
