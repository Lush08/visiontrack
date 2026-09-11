"""
VisionTrack - Gradio Web Application

Interactive demo for real-time object detection and tracking.
Supports image upload, video upload, and live webcam modes.

Usage:
    python demo_gradio.py
    python demo_gradio.py --port 7860 --share
    python demo_gradio.py --model models/yolo26n.onnx
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from detector import Detector
from tracker import ByteTracker
from utils.preprocessing import Preprocessor
from utils.visualization import COCO_CLASSES, draw_detections, draw_fps, draw_track_trail


# ---------------------------------------------------------------------------
# Theme + branding (matched to the static landing page in site/)
# ---------------------------------------------------------------------------
def _build_theme() -> gr.themes.Base:
    """Dark theme consistent with the site/ landing page."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=["Inter", "Segoe UI", "system-ui", "sans-serif"],
        font_mono=["JetBrains Mono", "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_md,
    )


CUSTOM_CSS = """
/* Surfaces — dark background consistent with site/ landing page */
.gradio-container {
    background: #0b0f14 !important;
    color: #e6edf5 !important;
}
.wrap {
    background: transparent !important;
}
footer { display: none !important; }

/* Typography */
h1, h2, h3, h4, .prose { color: #e6edf5 !important; }

/* Header banner */
.vt-header { text-align: center; padding: 1.75rem 0 0.5rem; }
.vt-header .vt-brand { font-size: 1.9rem; font-weight: 750; letter-spacing: -0.02em; color: #e6edf5; }
.vt-header .vt-brand span { color: #3b82f6; }
.vt-header .vt-sub { color: #94a3b8; font-size: 0.95rem; margin-top: 0.35rem; }

/* Tabs */
.tab-nav button { color: #94a3b8 !important; }
.tab-nav button.selected { color: #e6edf5 !important; border-color: #3b82f6 !important; }

/* Cards / panels */
.gr-box, .form, .gr-panel {
    background: #151b24 !important;
    border-color: #232d3a !important;
    color: #e6edf5 !important;
}
.form .label, .gr-box label { color: #94a3b8 !important; }

/* Buttons */
.gr-button { border-radius: 10px !important; }
.gr-button-primary {
    background: #3b82f6 !important; border-color: #3b82f6 !important; color: #0b0f14 !important;
}
.gr-button-primary:hover { background: #2563eb !important; }

/* Sliders */
.gr-slider input[type="range"] { background: #232d3a; }
.gr-slider input[type="range"]::-webkit-slider-thumb { background: #3b82f6; }

/* Accordion */
.gr-accordion { border-color: #232d3a !important; }
.gr-accordion .label-wrap { color: #94a3b8 !important; }

/* About page links */
.vt-about a { color: #3b82f6; text-decoration: none; }
.vt-about a:hover { text-decoration: underline; }
"""


# ---------------------------------------------------------------------------
# Globals (lazy-loaded)
# ---------------------------------------------------------------------------
_detector: Detector | None = None
_preprocessor: Preprocessor | None = None
_model_path: str = "models/yolo26n.onnx"  # resolved relative to project root
_imgsz: int = 640
_num_classes: int = 80
_use_gpu: bool = False


def _get_detector(
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Detector:
    """Return a cached Detector, re-created only when thresholds change."""
    global _detector, _preprocessor
    if _detector is None or _detector.conf_threshold != conf_threshold or _detector.iou_threshold != iou_threshold:
        _detector = Detector(
            _model_path,
            input_size=_imgsz,
            num_classes=_num_classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_names=COCO_CLASSES,
            use_cuda=_use_gpu,
        )
        _preprocessor = Preprocessor(_imgsz)
    return _detector


# ---------------------------------------------------------------------------
# Image detection
# ---------------------------------------------------------------------------
def detect_image(
    image: np.ndarray | None,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[np.ndarray | None, str]:
    """Run detection on a single image and return annotated result."""
    if image is None:
        return None, "Please upload an image."

    detector = _get_detector(conf_threshold, iou_threshold)
    preprocessor = Preprocessor(detector.input_size)

    # Gradio gives RGB; convert to BGR for OpenCV
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    tensor, scale, padding = preprocessor.preprocess(bgr)
    output = detector.infer(tensor)
    boxes_model, scores, class_ids = detector.postprocess(output)
    boxes = preprocessor.scale_boxes(boxes_model, scale, padding, bgr.shape)

    annotated = draw_detections(
        bgr, boxes, scores, class_ids,
        class_names=COCO_CLASSES,
        conf_threshold=conf_threshold,
    )

    # Build summary
    counts: dict[str, int] = {}
    for cid in class_ids:
        name = COCO_CLASSES[int(cid)] if int(cid) < len(COCO_CLASSES) else f"class_{int(cid)}"
        counts[name] = counts.get(name, 0) + 1
    summary_parts = [f"{n}: {c}" for n, c in sorted(counts.items(), key=lambda x: -x[1])]
    summary = f"Detected {len(boxes)} objects — " + ", ".join(summary_parts) if boxes.size else "No objects detected."

    # Back to RGB for Gradio
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), summary


# ---------------------------------------------------------------------------
# Video detection + tracking
# ---------------------------------------------------------------------------
def detect_video(
    video_path: str | None,
    conf_threshold: float,
    iou_threshold: float,
    draw_trails: bool,
) -> str | None:
    """Process a video file with detection + tracking, return output path."""
    if video_path is None:
        return None

    detector = _get_detector(conf_threshold, iou_threshold)
    preprocessor = Preprocessor(detector.input_size)
    tracker = ByteTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = str(Path(video_path).with_name(Path(video_path).stem + "_tracked.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor, scale, padding = preprocessor.preprocess(frame)
        output = detector.infer(tensor)
        boxes_model, scores, class_ids = detector.postprocess(output)
        boxes = preprocessor.scale_boxes(boxes_model, scale, padding, frame.shape)

        tracker.update(boxes, scores, class_ids)
        track_ids = np.array([t.id for t in tracker.tracks.values()], dtype=int)

        annotated = draw_detections(
            frame, boxes, scores, class_ids,
            class_names=COCO_CLASSES,
            track_ids=track_ids,
            conf_threshold=conf_threshold,
        )

        if draw_trails:
            annotated = draw_track_trail(annotated, tracker.track_history)

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Processed {frame_idx}/{total_frames} frames → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Webcam live detection + tracking
# ---------------------------------------------------------------------------
_webcam_running = False
_webcam_tracker: ByteTracker | None = None


def _webcam_frame(
    frame: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    draw_trails: bool,
) -> np.ndarray:
    """Process a single webcam frame with detection + tracking."""
    global _webcam_tracker

    detector = _get_detector(conf_threshold, iou_threshold)
    preprocessor = Preprocessor(detector.input_size)

    if _webcam_tracker is None:
        _webcam_tracker = ByteTracker()

    # Gradio sends RGB
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    tensor, scale, padding = preprocessor.preprocess(bgr)
    output = detector.infer(tensor)
    boxes_model, scores, class_ids = detector.postprocess(output)
    boxes = preprocessor.scale_boxes(boxes_model, scale, padding, bgr.shape)

    _webcam_tracker.update(boxes, scores, class_ids)
    track_ids = np.array([t.id for t in _webcam_tracker.tracks.values()], dtype=int)

    annotated = draw_detections(
        bgr, boxes, scores, class_ids,
        class_names=COCO_CLASSES,
        track_ids=track_ids,
        conf_threshold=conf_threshold,
    )

    if draw_trails:
        annotated = draw_track_trail(annotated, _webcam_tracker.track_history)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def webcam_reset():
    """Reset the webcam tracker state."""
    global _webcam_tracker
    _webcam_tracker = ByteTracker()
    return "Tracker reset."


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
def build_demo(model_path: str = "models/yolo26n.onnx", imgsz: int = 640, num_classes: int = 80, use_gpu: bool = False):
    """Create and return the Gradio Blocks demo."""
    global _model_path, _imgsz, _num_classes, _use_gpu
    _model_path = model_path
    _imgsz = imgsz
    _num_classes = num_classes
    _use_gpu = use_gpu

    with gr.Blocks(
        title="VisionTrack — Real-Time Object Detection & Tracking",
    ) as demo:
        gr.HTML(
            '<div class="vt-header">'
            '<div class="vt-brand">Vision<span>Track</span></div>'
            '<div class="vt-sub">Real-Time Multi-Object Detection &amp; Tracking — YOLO26 + ByteTrack + ONNX Runtime</div>'
            '</div>'
        )

        with gr.Tabs():
            # ---- Tab 1: Image Detection ----
            with gr.TabItem("🖼️ Image Detection"):
                gr.Markdown("Upload an image to detect objects with bounding boxes and class labels.")
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(type="numpy", label="Upload Image")
                        with gr.Accordion("⚙️ Settings", open=False):
                            img_conf = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="Confidence Threshold")
                            img_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU Threshold")
                        img_detect_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        img_output = gr.Image(type="numpy", label="Detection Result")
                        img_summary = gr.Textbox(label="Summary", interactive=False)

                img_detect_btn.click(
                    fn=detect_image,
                    inputs=[img_input, img_conf, img_iou],
                    outputs=[img_output, img_summary],
                )

            # ---- Tab 2: Video Processing ----
            with gr.TabItem("🎬 Video Processing"):
                gr.Markdown("Upload a video for frame-by-frame detection and tracking with persistent IDs.")
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_input = gr.Video(label="Upload Video")
                        with gr.Accordion("⚙️ Settings", open=False):
                            vid_conf = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="Confidence Threshold")
                            vid_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU Threshold")
                            vid_trails = gr.Checkbox(label="Draw Track Trails", value=True)
                        vid_process_btn = gr.Button("▶️ Process Video", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        vid_output = gr.Video(label="Tracked Output")

                vid_process_btn.click(
                    fn=detect_video,
                    inputs=[vid_input, vid_conf, vid_iou, vid_trails],
                    outputs=[vid_output],
                )

            # ---- Tab 3: Live Webcam ----
            with gr.TabItem("📹 Live Webcam"):
                gr.Markdown("Real-time detection and tracking from your webcam. Objects get persistent track IDs.")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("⚙️ Settings", open=False):
                            wc_conf = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="Confidence Threshold")
                            wc_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU Threshold")
                            wc_trails = gr.Checkbox(label="Draw Track Trails", value=True)
                        wc_reset_btn = gr.Button("🔄 Reset Tracker")

                    with gr.Column(scale=2):
                        wc_output = gr.Image(label="Live Detection Feed")

                # Webcam processing function that wraps the frame processor
                def webcam_process(frame, conf, iou, trails):
                    if frame is None:
                        return frame
                    return _webcam_frame(frame, conf, iou, trails)

                wc_input = gr.Image(
                    label="Webcam Input (auto-capture)",
                    sources=["webcam"],
                    streaming=True,
                )

                wc_input.stream(
                    fn=webcam_process,
                    inputs=[wc_input, wc_conf, wc_iou, wc_trails],
                    outputs=[wc_output],
                )

                wc_reset_btn.click(fn=webcam_reset, outputs=[])

            # ---- Tab 4: About ----
            with gr.TabItem("ℹ️ About"):
                gr.HTML(
                    '<div class="vt-about">'
                    '<h2>VisionTrack</h2>'
                    '<p>A real-time multi-object detection and tracking system '
                    'built to demonstrate the full ML deployment path — training, ONNX export, '
                    'inference in Python and C++, and an interactive web demo.</p>'
                    '<h3>Tech stack</h3>'
                    '<ul>'
                    '<li><strong>Detection</strong> — YOLO26 (Ultralytics, 2026)</li>'
                    '<li><strong>Inference</strong> — ONNX Runtime (Python + C++17)</li>'
                    '<li><strong>Tracking</strong> — ByteTrack (IoU-based)</li>'
                    '<li><strong>Frontend</strong> — Gradio</li>'
                    '</ul>'
                    '<h3>How it works</h3>'
                    '<ol>'
                    '<li>Input is letterbox-resized to 640×640 and normalized.</li>'
                    '<li>YOLO26 runs via ONNX Runtime inference.</li>'
                    '<li>ByteTrack assigns persistent IDs across frames using IoU matching.</li>'
                    '<li>Results are annotated with bounding boxes, labels, and track IDs.</li>'
                    '</ol>'
                    '<h3>Author</h3>'
                    '<p><strong>Armaan Dhall</strong> — 2nd Year CSE AI/ML Student<br>'
                    'GitHub: <a href="https://github.com/Lush08" target="_blank">Lush08</a></p>'
                    '</div>'
                )

    return demo


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionTrack Gradio Web App")
    parser.add_argument("--model", type=str, default="models/yolo26n.onnx", help="ONNX model path")
    parser.add_argument("--imgsz", type=int, default=640, help="Model input size")
    parser.add_argument("--classes", type=int, default=80, help="Number of classes")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if available")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve model path relative to project root (parent of python/)
    project_root = Path(__file__).resolve().parent.parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("  Export a model first: python export_onnx.py --weights yolo26n.pt")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    demo = build_demo(
        model_path=str(model_path),
        imgsz=args.imgsz,
        num_classes=args.classes,
        use_gpu=args.gpu,
    )

    print(f"\nStarting Gradio server on port {args.port}...")
    demo.launch(
        server_port=args.port,
        share=args.share,
        theme=_build_theme(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
