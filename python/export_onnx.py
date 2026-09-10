"""
VisionTrack - ONNX Model Export

Export a trained YOLO26 model to ONNX format for cross-platform deployment.

Usage:
    python export_onnx.py --weights runs/detect/train/weights/best.pt
    python export_onnx.py --weights yolo26n.pt --imgsz 640 --simplify
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO26 model to ONNX format"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to .pt weights file",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX graph with onnxslim")
    parser.add_argument("--half", action="store_true", help="Export in FP16 (half precision)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes")
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for exported model",
    )
    return parser.parse_args()


def export(args: argparse.Namespace) -> str:
    """Export YOLO26 model to ONNX format. Returns path to exported model."""
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print(f"Exporting {weights_path.name} to ONNX...")
    print(f"  Image size: {args.imgsz}")
    print(f"  Opset: {args.opset}")
    print(f"  Simplify: {args.simplify}")
    print(f"  FP16: {args.half}")
    print(f"  Dynamic shapes: {args.dynamic}")

    # Load model
    model = YOLO(str(weights_path))

    # Export to ONNX (exports next to the weights by default)
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        half=args.half,
        dynamic=bool(args.dynamic),
    )

    # Move to the requested output directory if different
    onnx_file = Path(export_path)
    output_dir = Path(args.output)
    if onnx_file.parent.resolve() != output_dir.resolve():
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / onnx_file.name
        onnx_file.replace(final_path)
        export_path = str(final_path)

    print(f"\nExport complete!")
    print(f"  ONNX model: {export_path}")

    # Print model size comparison
    pt_size = weights_path.stat().st_size / (1024 * 1024)
    onnx_path = Path(export_path)
    onnx_size = onnx_path.stat().st_size / (1024 * 1024) if onnx_path.exists() else 0
    print(f"  .pt size:   {pt_size:.1f} MB")
    print(f"  .onnx size: {onnx_size:.1f} MB")
    if pt_size > 0:
        print(f"  Compression: {(1 - onnx_size / pt_size) * 100:.1f}%")

    return str(export_path)


def validate_export(onnx_path: str, imgsz: int = 640) -> None:
    """Validate exported ONNX model loads and runs correctly."""
    try:
        import numpy as np
        import onnxruntime as ort

        print("\nValidating ONNX model...")
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"  Input: {input_name}, shape: {input_shape}")

        # Run dummy inference
        dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        print(f"  Output shapes: {[o.shape for o in outputs]}")
        print("  Validation passed!")
    except ImportError:
        print("  Skipped: onnxruntime not installed")
    except Exception as e:
        print(f"  Validation failed: {e}")


def main() -> None:
    args = parse_args()
    export_path = export(args)
    validate_export(export_path, args.imgsz)


if __name__ == "__main__":
    main()
