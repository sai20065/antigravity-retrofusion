# training/train_yolo.py
# YOLOv8-nano Training Pipeline for Road Asset Detection
#
# Design reference: Section 4.1 of RetroFusion Design Document.
#
# This script handles:
#   1. Dataset preparation (YOLO format: images/ + labels/)
#   2. Synthetic dataset generation for bootstrapping
#   3. Training YOLOv8-nano with custom road asset classes
#   4. Evaluation and mAP metrics
#   5. Export to TFLite INT8 for Raspberry Pi deployment
#
# Classes:
#   0: speed_sign, 1: direction_sign, 2: lane_marking,
#   3: stop_line, 4: road_stud, 5: zebra_crossing
#
# Usage:
#   python training/train_yolo.py --mode generate   # Generate synthetic dataset
#   python training/train_yolo.py --mode train       # Train YOLOv8-nano
#   python training/train_yolo.py --mode export      # Export to TFLite
#   python training/train_yolo.py --mode all         # Full pipeline

import os
import sys
import argparse
import numpy as np
import shutil
import yaml
import time
from typing import Tuple, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ══════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(output_dir: str = "datasets/road_assets",
                                num_train: int = 2000,
                                num_val: int = 500,
                                img_size: int = 640):
    """
    Generate synthetic road scene images with YOLO-format annotations.

    Each image contains 1-4 randomly placed road assets with known
    bounding boxes. Includes:
        - Varied backgrounds (road textures, lighting)
        - Multiple asset sizes and orientations
        - Weather augmentation (rain streaks, fog overlay)
        - Day/night lighting conditions
    """
    import cv2

    print(f"[YOLO-GEN] Generating synthetic dataset: {num_train} train + {num_val} val")
    print(f"           Output: {output_dir}")

    for split, count in [("train", num_train), ("val", num_val)]:
        img_dir = os.path.join(output_dir, "images", split)
        lbl_dir = os.path.join(output_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(count):
            img, labels = _generate_scene(img_size)
            
            # Save image
            img_path = os.path.join(img_dir, f"scene_{i:05d}.jpg")
            cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save YOLO labels (cls x_center y_center width height — all normalized)
            lbl_path = os.path.join(lbl_dir, f"scene_{i:05d}.txt")
            with open(lbl_path, "w") as f:
                for lbl in labels:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

            if (i + 1) % 500 == 0:
                print(f"  [{split}] {i+1}/{count}")

    # Generate data.yaml for YOLO training
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 6,
        "names": ["speed_sign", "direction_sign", "lane_marking",
                   "stop_line", "road_stud", "zebra_crossing"],
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"[YOLO-GEN] Dataset created: {yaml_path}")
    return yaml_path


def _generate_scene(img_size: int) -> Tuple[np.ndarray, List]:
    """Generate a single synthetic road scene with annotations."""
    import cv2

    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    labels = []

    # Background — road surface with variation
    is_night = np.random.random() < 0.3
    base_gray = np.random.randint(20, 40) if is_night else np.random.randint(60, 100)
    img[:, :] = (base_gray, base_gray, base_gray)

    # Add road texture noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Road lanes
    cx = img_size // 2
    lane_color = (180, 180, 180) if not is_night else (120, 120, 120)
    for y in range(0, img_size, 60):
        if np.random.random() < 0.7:
            cv2.rectangle(img, (cx - 2, y), (cx + 2, y + 30), lane_color, -1)

    # Place 1-4 random objects
    n_objects = np.random.randint(1, 5)
    placed_boxes = []

    for _ in range(n_objects):
        cls_id = np.random.randint(0, 6)
        box, cls_id = _place_object(img, cls_id, img_size, placed_boxes, is_night)
        if box:
            x, y, w, h = box
            # YOLO format: class x_center y_center width height (all normalized)
            xc = (x + w / 2) / img_size
            yc = (y + h / 2) / img_size
            wn = w / img_size
            hn = h / img_size
            labels.append([cls_id, xc, yc, wn, hn])
            placed_boxes.append(box)

    # Weather effects
    if np.random.random() < 0.2:  # Rain
        _add_rain(img)
    if np.random.random() < 0.15:  # Fog
        _add_fog(img)

    return img, labels


def _place_object(img: np.ndarray, cls_id: int, img_size: int,
                   existing: list, is_night: bool):
    """Place a synthetic road asset object on the image."""
    import cv2

    for attempt in range(10):
        if cls_id == 0:  # speed_sign
            w, h = np.random.randint(40, 100), np.random.randint(40, 100)
            x = np.random.randint(img_size // 4, 3 * img_size // 4 - w)
            y = np.random.randint(int(img_size * 0.05), int(img_size * 0.4))
            color = (200, 200, 200) if not is_night else (140, 140, 140)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 200), 2)
            cv2.putText(img, str(np.random.choice([30, 40, 50, 60, 80])),
                         (x + w // 4, y + h * 3 // 4),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            return (x, y, w, h), cls_id

        elif cls_id == 1:  # direction_sign
            w, h = np.random.randint(80, 160), np.random.randint(40, 80)
            x = np.random.randint(0, img_size - w)
            y = np.random.randint(int(img_size * 0.05), int(img_size * 0.35))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 140, 0), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (220, 220, 220), 2)
            return (x, y, w, h), cls_id

        elif cls_id == 2:  # lane_marking
            w = np.random.randint(6, 15)
            h = np.random.randint(40, 100)
            x = np.random.randint(img_size // 4, 3 * img_size // 4)
            y = np.random.randint(int(img_size * 0.3), int(img_size * 0.9))
            cv2.rectangle(img, (x, y), (x + w, y + h), (220, 220, 220), -1)
            return (x, y, w, h), cls_id

        elif cls_id == 3:  # stop_line
            w = np.random.randint(100, 250)
            h = np.random.randint(8, 20)
            x = np.random.randint(img_size // 4, img_size // 2)
            y = np.random.randint(int(img_size * 0.5), int(img_size * 0.85))
            cv2.rectangle(img, (x, y), (x + w, y + h), (220, 220, 220), -1)
            return (x, y, w, h), cls_id

        elif cls_id == 4:  # road_stud
            w = h = np.random.randint(10, 25)
            x = np.random.randint(img_size // 3, 2 * img_size // 3)
            y = np.random.randint(int(img_size * 0.3), int(img_size * 0.8))
            cv2.circle(img, (x + w // 2, y + h // 2), w // 2,
                        (0, 200, 200), -1)
            return (x, y, w, h), cls_id

        elif cls_id == 5:  # zebra_crossing
            w = np.random.randint(120, 250)
            h = np.random.randint(50, 100)
            x = np.random.randint(img_size // 4, img_size // 2)
            y = np.random.randint(int(img_size * 0.4), int(img_size * 0.8))
            for stripe in range(0, w, 20):
                cv2.rectangle(img, (x + stripe, y), (x + stripe + 10, y + h),
                               (220, 220, 220), -1)
            return (x, y, w, h), cls_id

    return None, cls_id


def _add_rain(img: np.ndarray):
    """Add synthetic rain streaks."""
    import cv2
    h, w = img.shape[:2]
    for _ in range(100):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(10, 30)
        cv2.line(img, (x, y), (x - 2, min(h - 1, y + length)),
                  (180, 180, 200), 1)


def _add_fog(img: np.ndarray):
    """Add synthetic fog overlay."""
    fog = np.ones_like(img) * np.random.randint(160, 200)
    alpha = np.random.uniform(0.1, 0.4)
    cv2.addWeighted(img, 1 - alpha, fog.astype(np.uint8), alpha, 0, img)


# ══════════════════════════════════════════════════════════════════════════
# 2. YOLO TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train_yolo(data_yaml: str, epochs: int = 100, img_size: int = 640,
               batch_size: int = 16, model_variant: str = "yolov8n.pt",
               project: str = "runs/detect", name: str = "road_assets"):
    """
    Train YOLOv8-nano on the road asset dataset.

    Uses ultralytics training with:
        - Pre-trained COCO weights (transfer learning)
        - Data augmentation (mosaic, mixup, HSV variation)
        - Early stopping (patience=50)
        - Best model checkpoint saving

    Args:
        data_yaml:    Path to data.yaml
        epochs:       Number of training epochs
        img_size:     Input image size
        batch_size:   Training batch size
        model_variant: Base model (yolov8n.pt for nano)
        project:      Output project directory
        name:         Run name
    """
    from ultralytics import YOLO

    print(f"[YOLO-TRAIN] Starting training")
    print(f"  Model:     {model_variant}")
    print(f"  Dataset:   {data_yaml}")
    print(f"  Epochs:    {epochs}")
    print(f"  Image size: {img_size}")
    print(f"  Batch:     {batch_size}")

    # Load pre-trained YOLOv8-nano
    model = YOLO(model_variant)

    # Train with optimized settings for road assets
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project,
        name=name,
        exist_ok=True,

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,        # final LR fraction
        momentum=0.937,
        weight_decay=0.0005,

        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,     # HSV hue augmentation
        hsv_s=0.7,       # HSV saturation
        hsv_v=0.4,       # HSV value
        degrees=10.0,    # Rotation augmentation
        translate=0.1,
        scale=0.5,
        flipud=0.0,      # No vertical flip for road assets
        fliplr=0.5,

        # Training behavior
        patience=50,     # Early stopping patience
        save=True,
        save_period=10,
        plots=True,
        verbose=True,

        # Hardware
        device="0" if _cuda_available() else "cpu",
        workers=4,
    )

    print(f"\n[YOLO-TRAIN] Training complete!")
    best_path = os.path.join(project, name, "weights", "best.pt")
    print(f"  Best model: {best_path}")

    return best_path, results


def evaluate_yolo(model_path: str, data_yaml: str, img_size: int = 640):
    """Evaluate trained YOLO model and print mAP metrics."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=img_size, verbose=True)

    print(f"\n[YOLO-EVAL] Results:")
    print(f"  mAP@0.5:     {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  Precision:    {results.box.mp:.4f}")
    print(f"  Recall:       {results.box.mr:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# 3. EXPORT TO TFLITE
# ══════════════════════════════════════════════════════════════════════════

def export_to_tflite(model_path: str, img_size: int = 640):
    """
    Export trained YOLOv8 model to TFLite INT8 for Raspberry Pi.

    Pipeline: PyTorch (.pt) → TFLite (.tflite)
    Quantization: INT8 with XNNPACK delegate
    Expected RPi4 inference: ~85ms per frame

    Args:
        model_path: Path to trained .pt model
        img_size:   Input image size
    """
    from ultralytics import YOLO

    print(f"[YOLO-EXPORT] Exporting to TFLite INT8")
    print(f"  Source: {model_path}")

    model = YOLO(model_path)

    # Export to TFLite with INT8 quantization
    tflite_path = model.export(
        format="tflite",
        imgsz=img_size,
        int8=True,       # INT8 quantization
        half=False,
    )

    print(f"[YOLO-EXPORT] TFLite model exported: {tflite_path}")

    # Also export ONNX for cross-platform use
    onnx_path = model.export(format="onnx", imgsz=img_size)
    print(f"[YOLO-EXPORT] ONNX model exported: {onnx_path}")

    return tflite_path


def _cuda_available() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Road Asset Training Pipeline")
    parser.add_argument("--mode", choices=["generate", "train", "evaluate", "export", "all"],
                        default="all", help="Pipeline stage to run")
    parser.add_argument("--data-dir", type=str, default="datasets/road_assets",
                        help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model (for evaluate/export)")
    parser.add_argument("--num-train", type=int, default=2000,
                        help="Number of synthetic training images")
    parser.add_argument("--num-val", type=int, default=500,
                        help="Number of synthetic validation images")
    args = parser.parse_args()

    if args.mode in ("generate", "all"):
        data_yaml = generate_synthetic_dataset(
            args.data_dir, args.num_train, args.num_val, args.img_size)
    else:
        data_yaml = os.path.join(args.data_dir, "data.yaml")

    if args.mode in ("train", "all"):
        model_path, _ = train_yolo(data_yaml, args.epochs, args.img_size, args.batch_size)
    else:
        model_path = args.model_path

    if args.mode in ("evaluate", "all") and model_path:
        evaluate_yolo(model_path, data_yaml, args.img_size)

    if args.mode in ("export", "all") and model_path:
        export_to_tflite(model_path, args.img_size)
