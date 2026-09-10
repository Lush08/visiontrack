"""
VisionTrack - CLI Detection & Tracking Pipeline

Runs detection + tracking on images or videos using ONNX Runtime.

Usage:
    python demo_tracker.py --source image.jpg --output result.jpg
    python demo_tracker.py --source video.mp4 --output result.mp4 --show
    python demo_tracker.py --source video.mp4 --output result.mp4 --webcam
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running as `python demo_tracker.py` from project root
sys.path.insert(0, str(Path(__file__).parent))

from detector import Detector
from tracker import ByteTracker
from utils.preprocessing import Preprocessor
from utils.visualization import COCO_CLASSES, draw_detections, draw_fps, draw_track_trail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionTrack: real-time detection and tracking"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input image/video path, or 'webcam' for camera input",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: source_basename_tracked.ext)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo26n.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Model input size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--classes", type=int, default=80, help="Number of classes")
    parser.add_argument("--show", action="store_true", help="Show output live")
    parser.add_argument("--save", action="store_true", default=True, help="Save output")
    parser.add_argument("--draw-trails", action="store_true", help="Draw track trails")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA acceleration if available")
    return parser.parse_args()


def process_image(args: argparse.Namespace) -> None:
    """Detect objects in a single image."""
    detector = Detector(
        args.model, input_size=args.imgsz, num_classes=args.classes,
        conf_threshold=args.conf, iou_threshold=args.iou, use_cuda=args.gpu,
        class_names=COCO_CLASSES,
    )
    preprocessor = Preprocessor(args.imgsz)

    image = cv2.imread(args.source)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.source}")

    # Inference
    tensor, scale, padding = preprocessor.preprocess(image)
    output = detector.infer(tensor)
    boxes_model, scores, class_ids = detector.postprocess(output)
    boxes = preprocessor.scale_boxes(
        boxes_model, scale, padding, image.shape
    )

    # Annotate
    annotated = draw_detections(
        image, boxes, scores, class_ids, class_names=COCO_CLASSES,
        conf_threshold=args.conf,
    )

    # Save
    output_path = args.output or str(
        Path(args.source).with_name(Path(args.source).stem + "_detected.png")
    )
    cv2.imwrite(output_path, annotated)
    print(f"Saved: {output_path}")
    print(f"  Detected: {len(boxes)} objects")


def process_video(args: argparse.Namespace, use_camera: bool = False) -> None:
    """Detect + track objects in a video or webcam stream."""
    detector = Detector(
        args.model, input_size=args.imgsz, num_classes=args.classes,
        conf_threshold=args.conf, iou_threshold=args.iou, use_cuda=args.gpu,
        class_names=COCO_CLASSES,
    )
    preprocessor = Preprocessor(args.imgsz)
    tracker = ByteTracker()

    # Open video source
    if use_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {args.source}")

    # Video writer
    writer = None
    output_path = args.output or str(
        Path(args.source).with_name(Path(args.source).stem + "_tracked.mp4")
    )

    if use_camera:
        output_path = "webcam_tracked.mp4"

    fps_display = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Infer
        tensor, scale, padding = preprocessor.preprocess(frame)
        output = detector.infer(tensor)
        boxes_model, scores, class_ids = detector.postprocess(output)
        boxes = preprocessor.scale_boxes(boxes_model, scale, padding, frame.shape)

        # Track
        tracker.update(boxes, scores, class_ids)
        track_ids = np.array([t.id for t in tracker.tracks.values()], dtype=int)

        # Annotate
        annotated = draw_detections(
            frame, boxes, scores, class_ids, class_names=COCO_CLASSES,
            track_ids=track_ids,
            conf_threshold=args.conf,
        )

        if args.draw_trails:
            annotated = draw_track_trail(annotated, tracker.track_history)

        # Measure FPS
        frame_count += 1
        elapsed = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if elapsed > 0 and frame_count > 1:
            fps_display = frame_count / elapsed
        annotated = draw_fps(annotated, fps_display)

        # Display
        if args.show:
            cv2.imshow("VisionTrack", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Write output
        if args.save and writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
        if writer is not None:
            writer.write(annotated)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")
    print(f"  Saved: {output_path}")
    print(f"  Active tracks: {len(tracker.tracks)}")


def main() -> None:
    args = parse_args()

    if args.source == "webcam":
        process_video(args, use_camera=True)
    else:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Error: {args.source} not found", file=sys.stderr)
            sys.exit(1)

        # Check file extension
        img_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if source_path.suffix.lower() in img_ext:
            process_image(args)
        else:
            process_video(args)


if __name__ == "__main__":
    main()