"""
VisionTrack - YOLO26 Training Pipeline

Fine-tune YOLO26 on custom datasets or train from scratch using Ultralytics.

Usage:
    python train.py --data data/custom.yaml --model yolo26n --epochs 100
    python train.py --data coco.yaml --model yolo26s --epochs 50 --imgsz 640
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO26 object detection model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Base model: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file (e.g., data/custom.yaml or coco.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto)")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="", help="Device: '' for auto, 'cpu', '0', '0,1'")
    parser.add_argument("--project", type=str, default="runs/detect", help="Save directory")
    parser.add_argument("--name", type=str, default="train", help="Experiment name")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Run YOLO26 training with the given arguments."""
    print(f"Starting VisionTrack training...")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        device=args.device or None,
        project=args.project,
        name=args.name,
        patience=args.patience,
        pretrained=args.pretrained,
        resume=args.resume,
        exist_ok=True,
        verbose=True,
    )

    # Print results summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best weights: {args.project}/{args.name}/weights/best.pt")
    print(f"  Last weights: {args.project}/{args.name}/weights/last.pt")

    # Validate the best model
    print("\nRunning validation on best model...")
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        best_model = YOLO(str(best_weights))
        metrics = best_model.val(data=args.data)
        print(f"  mAP50:    {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall:    {metrics.box.mr:.4f}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
