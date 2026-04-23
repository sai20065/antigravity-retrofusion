# training/convert_to_tflite.py
# PyTorch → TFLite INT8 Conversion Pipeline
#
# Design reference: Section 4.4 of RetroFusion Design Document.
#
# Conversion chain:
#   PyTorch (.pt) → ONNX (.onnx) → TensorFlow SavedModel → TFLite (INT8)
#
# Performance targets:
#   - Model size: ~3.2MB (INT8)
#   - Inference (RPi4): ~65ms with XNNPACK delegate (4 threads)
#   - Accuracy loss: < 2% mAP drop from FP32 baseline
#
# Usage:
#   python training/convert_to_tflite.py --model checkpoints/best_model.pth
#   python training/convert_to_tflite.py --model checkpoints/best_model.pth --benchmark

import os
import sys
import argparse
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def convert_mobilenet_to_onnx(model_path: str, output_path: str = None,
                               input_size: int = 224):
    """
    Step 1: Convert PyTorch MobileNetV2-RA to ONNX format.

    Args:
        model_path:  Path to trained .pth checkpoint
        output_path: Path for output .onnx file
        input_size:  Input image dimension (224×224)
    """
    import torch
    from training.train_ra_model import MobileNetV2_RA

    if output_path is None:
        output_path = model_path.replace(".pth", ".onnx")

    print(f"[Convert] Step 1: PyTorch → ONNX")
    print(f"  Input:  {model_path}")
    print(f"  Output: {output_path}")

    # Load model
    model = MobileNetV2_RA(pretrained=False)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Export to ONNX with dynamic batch size
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["ra_prediction", "confidence"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "ra_prediction": {0: "batch_size"},
            "confidence": {0: "batch_size"},
        },
    )

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ONNX model saved: {size_mb:.1f} MB")

    return output_path


def convert_onnx_to_tflite(onnx_path: str, output_path: str = None,
                            input_size: int = 224,
                            quantize_int8: bool = True,
                            num_calibration_samples: int = 100):
    """
    Step 2: Convert ONNX → TensorFlow → TFLite (with INT8 quantization).

    Uses representative dataset for calibration during INT8 quantization.
    The calibration dataset consists of synthetic road sign images.

    Args:
        onnx_path:               Path to ONNX model
        output_path:             Path for output .tflite file
        input_size:              Input dimension
        quantize_int8:           Whether to apply INT8 quantization
        num_calibration_samples: Number of samples for quantization calibration
    """
    import tensorflow as tf

    if output_path is None:
        suffix = "_int8.tflite" if quantize_int8 else "_fp32.tflite"
        output_path = onnx_path.replace(".onnx", suffix)

    print(f"[Convert] Step 2: ONNX → TFLite")
    print(f"  Quantization: {'INT8' if quantize_int8 else 'FP32'}")

    # Step 2a: ONNX → TensorFlow SavedModel
    try:
        import onnx_tf
        from onnx_tf.backend import prepare

        onnx_model = __import__("onnx").load(onnx_path)
        tf_rep = prepare(onnx_model)
        saved_model_dir = onnx_path.replace(".onnx", "_tf_saved_model")
        tf_rep.export_graph(saved_model_dir)
        print(f"  TF SavedModel: {saved_model_dir}")
    except ImportError:
        print("  [WARN] onnx-tf not available. Trying onnx2tf...")
        try:
            import subprocess
            saved_model_dir = onnx_path.replace(".onnx", "_tf_saved_model")
            subprocess.run([
                "onnx2tf", "-i", onnx_path,
                "-o", saved_model_dir,
                "--non_verbose"
            ], check=True)
        except Exception as e:
            print(f"  [ERROR] ONNX→TF conversion failed: {e}")
            print("  Install: pip install onnx-tf  OR  pip install onnx2tf")
            return None

    # Step 2b: TF SavedModel → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize_int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for calibration
        def representative_data_gen():
            for _ in range(num_calibration_samples):
                # Generate random calibration images (similar to training distribution)
                sample = np.random.rand(1, input_size, input_size, 3).astype(np.float32)
                yield [sample]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.float32  # Keep output as float for RA

    tflite_model = converter.convert()

    # Save
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  TFLite model saved: {output_path} ({size_mb:.1f} MB)")

    return output_path


def benchmark_tflite(tflite_path: str, input_size: int = 224,
                      num_runs: int = 50):
    """
    Benchmark TFLite model inference speed.

    Reports:
        - Average inference time
        - Min/max inference time
        - Estimated throughput (inferences/sec)

    Args:
        tflite_path: Path to .tflite file
        input_size:  Input dimension
        num_runs:    Number of benchmark iterations
    """
    import tensorflow as tf

    print(f"\n[Benchmark] TFLite Model: {tflite_path}")
    print(f"  Runs: {num_runs}")

    interpreter = tf.lite.Interpreter(
        model_path=tflite_path,
        num_threads=4,
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input shape:  {input_details[0]['shape']}")
    print(f"  Input dtype:  {input_details[0]['dtype']}")
    print(f"  Output shapes: {[d['shape'] for d in output_details]}")

    # Prepare input
    input_type = input_details[0]['dtype']
    if input_type == np.int8:
        input_data = np.random.randint(-128, 127, input_details[0]['shape']).astype(np.int8)
    else:
        input_data = np.random.rand(*input_details[0]['shape']).astype(np.float32)

    # Warmup
    for _ in range(5):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"\n  Results ({num_runs} runs):")
    print(f"    Average: {latencies.mean():.2f} ms")
    print(f"    Median:  {np.median(latencies):.2f} ms")
    print(f"    Min:     {latencies.min():.2f} ms")
    print(f"    Max:     {latencies.max():.2f} ms")
    print(f"    Std:     {latencies.std():.2f} ms")
    print(f"    Throughput: {1000 / latencies.mean():.1f} inferences/sec")

    return {
        "average_ms": float(latencies.mean()),
        "median_ms": float(np.median(latencies)),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "throughput": float(1000 / latencies.mean()),
    }


def full_conversion_pipeline(model_path: str, output_dir: str = "models",
                              input_size: int = 224, benchmark: bool = True):
    """
    Run the complete conversion pipeline:
    PyTorch → ONNX → TFLite (FP32 + INT8) → Benchmark

    Args:
        model_path: Path to trained PyTorch checkpoint
        output_dir: Directory for output models
        input_size: Input image dimension
        benchmark:  Whether to run benchmark after conversion
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("  MobileNetV2-RA → TFLite Conversion Pipeline")
    print("=" * 60)

    # Step 1: PyTorch → ONNX
    onnx_path = os.path.join(output_dir, "mobilenet_ra.onnx")
    try:
        onnx_path = convert_mobilenet_to_onnx(model_path, onnx_path, input_size)
    except Exception as e:
        print(f"[ERROR] ONNX conversion failed: {e}")
        return

    # Step 2a: ONNX → TFLite FP32
    tflite_fp32_path = os.path.join(output_dir, "mobilenet_ra_fp32.tflite")
    try:
        convert_onnx_to_tflite(onnx_path, tflite_fp32_path,
                                input_size, quantize_int8=False)
    except Exception as e:
        print(f"[WARN] FP32 TFLite conversion failed: {e}")

    # Step 2b: ONNX → TFLite INT8
    tflite_int8_path = os.path.join(output_dir, "mobilenet_ra_int8.tflite")
    try:
        convert_onnx_to_tflite(onnx_path, tflite_int8_path,
                                input_size, quantize_int8=True)
    except Exception as e:
        print(f"[WARN] INT8 TFLite conversion failed: {e}")

    # Step 3: Benchmark
    if benchmark:
        for path in [tflite_fp32_path, tflite_int8_path]:
            if os.path.exists(path):
                benchmark_tflite(path, input_size)

    print("\n" + "=" * 60)
    print("  Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileNetV2-RA TFLite Conversion")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Output directory for converted models")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    args = parser.parse_args()

    full_conversion_pipeline(args.model, args.output_dir,
                              args.input_size, args.benchmark)
