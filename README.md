# VisionTrack

**Real-Time Multi-Object Detection & Tracking System** — Python training, C++ ONNX inference, and a live Gradio web demo.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/c++-17-blue.svg)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.16+-orange.svg)](https://onnxruntime.ai/)
[![Tests](https://img.shields.io/badge/tests-23%20passing-brightgreen.svg)](tests/)

## Overview

VisionTrack is an end-to-end computer vision pipeline that demonstrates:

- **Object Detection** using [YOLO26](https://docs.ultralytics.com/models/yolo26/) (Ultralytics, 2026) — the latest state-of-the-art model
- **Multi-Object Tracking** using IoU-based ByteTrack for consistent object IDs across frames
- **Dual Inference Backends**: Python (ONNX Runtime) for convenience, C++ (ONNX Runtime) for production speed
- **Interactive Web Demo** via Gradio with image upload, video processing, and live webcam support
- **Training Pipeline** for fine-tuning YOLO26 on custom datasets
- **23 Unit Tests** covering preprocessing, postprocessing, and tracking logic

## Demo

> Live demo coming soon on HuggingFace Spaces!

```
Upload Image → YOLO26 Detection → Bounding Boxes + Labels + Track IDs
     ↓
Video Processing → Frame-by-frame detection + ByteTrack ID assignment
     ↓
Live Webcam → Real-time detection + tracking at 30+ FPS
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VisionTrack Pipeline                   │
├──────────────┬──────────────────┬───────────────────────┤
│   Training   │    Inference     │      Demo Layer       │
│   (Python)   │  (Python/C++)    │     (Gradio)          │
├──────────────┼──────────────────┼───────────────────────┤
│ Ultralytics  │ ONNX Runtime     │ Image Upload          │
│ YOLO26       │ Letterbox Resize  │ Video Processing      │
│ Custom Data  │ NMS + Decode     │ Live Webcam           │
│ ONNX Export  │ ByteTrack        │ Confidence Controls   │
└──────────────┴──────────────────┴───────────────────────┘
```

**Detection Pipeline:**
```
Input → Letterbox (640×640) → Float32 Tensor (NCHW) → ONNX Runtime
     → Raw Output (1,84,8400) → Decode + NMS → Bounding Boxes
     → Scale to Original Coords → Annotated Output
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Detection Model | YOLO26 (Ultralytics) | Latest SOTA, Jan 2026 |
| Training | Python, PyTorch | Standard ML stack |
| Model Format | ONNX (simplified, opset 17) | Cross-platform deployment |
| C++ Inference | ONNX Runtime C++ API | Production-speed inference |
| Object Tracking | ByteTrack (IoU-based) | State-of-the-art, clean impl |
| Web Demo | Gradio | Fast to build, HuggingFace deploy |
| Build System | CMake (C++), pip (Python) | Professional tooling |
| CI/CD | GitHub Actions | Automated testing |
| Testing | pytest (Python) | 23 unit tests |

## Quick Start

### Python Setup

```bash
# Clone the repo
git clone https://github.com/Lush08/visiontrack.git
cd visiontrack

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Gradio Demo

```bash
cd python
python demo_gradio.py
# Open http://localhost:7860
```

**Gradio Demo Features:**
- **Image Detection**: Upload any image → instant object detection with bounding boxes
- **Video Processing**: Upload video → frame-by-frame detection + tracking with persistent IDs
- **Live Webcam**: Real-time detection and tracking from your camera
- **Settings Panel**: Adjustable confidence and IoU thresholds

### Run Inference on an Image

```bash
python python/demo_tracker.py --source assets/test_bus.jpg --output result.jpg
```

### Run Inference on Video

```bash
python python/demo_tracker.py --source video.mp4 --output result.mp4 --show --draw-trails
```

### C++ Build (Windows)

```bash
# Build with CMake
cd cpp
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -DONNXRUNTIME_DIR=C:/path/to/onnxruntime
cmake --build . --config Release

# Run image detection
.\Release\visiontrack.exe detect ..\..\assets\test_bus.jpg --output result.jpg

# Run video tracking
.\Release\visiontrack.exe detect ..\..\video.mp4 --output result.mp4

# Benchmark
.\Release\visiontrack.exe benchmark --iterations 200
```

## Training on Custom Data

```bash
# Step 1: Prepare your dataset in YOLO format
# Step 2: Train YOLO26 on your dataset
python python/train.py --data data/custom.yaml --model yolo26n --epochs 100 --imgsz 640

# Step 3: Export trained model to ONNX
python python/export_onnx.py --weights runs/detect/train/weights/best.pt --simplify

# Step 4: Run inference with your custom model
python python/demo_tracker.py --model models/best.onnx --source image.jpg
```

See [notebooks/02_training.ipynb](notebooks/02_training.ipynb) for a full training walkthrough with explanations.

## Benchmarks

> Run `python python/demo_tracker.py --benchmark` or `./visiontrack benchmark` to generate your own.

| Backend | Model | Input | Latency (ms) | FPS |
|---------|-------|-------|--------------|-----|
| Python ONNX Runtime | YOLO26n | 640×640 | ~18 | ~55 |
| C++ ONNX Runtime | YOLO26n | 640×640 | TBD | TBD |

## Project Structure

```
visiontrack/
├── python/
│   ├── train.py              # YOLO26 fine-tuning script
│   ├── export_onnx.py        # ONNX model export
│   ├── demo_gradio.py        # Gradio web application (3 modes)
│   ├── demo_tracker.py       # CLI detection + tracking pipeline
│   ├── detector.py           # ONNX Runtime detector wrapper
│   ├── tracker.py            # ByteTrack multi-object tracker
│   └── utils/
│       ├── preprocessing.py  # Letterbox, tensor conversion
│       ├── postprocessing.py # NMS, box decoding
│       ├── visualization.py  # Drawing boxes, labels, trails
│       └── dataset.py        # Dataset YAML helpers
├── cpp/
│   ├── CMakeLists.txt        # C++ build configuration
│   └── src/
│       ├── main.cpp           # CLI driver (detect/benchmark)
│       ├── detector.hpp/cpp   # ONNX Runtime inference
│       ├── tracker.hpp/cpp    # ByteTrack C++ implementation
│       ├── preprocessor.hpp/cpp  # Image preprocessing
│       └── postprocessor.hpp/cpp # NMS and decoding
├── tests/
│   ├── test_detector.py      # 14 tests: preprocessing + postprocessing
│   └── test_tracker.py       # 9 tests: tracker lifecycle
├── notebooks/
│   ├── 01_exploration.ipynb  # Data exploration & EDA
│   ├── 02_training.ipynb     # Training walkthrough
│   └── 03_benchmark.ipynb    # Python vs C++ benchmarks
├── models/                   # ONNX model weights (gitignored)
├── assets/                   # Demo images and GIFs
└── .github/workflows/ci.yml  # GitHub Actions CI
```

## What This Project Demonstrates

| Skill | Evidence |
|-------|----------|
| **Machine Learning** | YOLO26 training, fine-tuning, evaluation metrics |
| **Computer Vision** | Object detection, tracking, preprocessing pipelines |
| **Python** | Type-annotated code, clean architecture, docstrings, pytest |
| **C++** | ONNX Runtime integration, memory management, CMake build |
| **Model Deployment** | ONNX export, cross-platform inference (Python + C++) |
| **Web Development** | Gradio interactive demo with 3 input modes |
| **Software Engineering** | 23 unit tests, CI/CD pipeline, modular design |
| **Documentation** | Professional README, Jupyter notebooks, code comments |

## Running Tests

```bash
cd python
python -m pytest ../tests/ -v
```

```
tests/test_detector.py::TestPreprocessor::test_letterbox_square_image PASSED
tests/test_detector.py::TestPreprocessor::test_letterbox_preserves_aspect_ratio PASSED
tests/test_detector.py::TestPreprocessor::test_to_tensor_shape PASSED
tests/test_detector.py::TestPostprocessing::test_decode_detections_single_detection PASSED
tests/test_tracker.py::TestByteTracker::test_persistence_same_position PASSED
tests/test_tracker.py::TestByteTracker::test_stale_track_removal PASSED
...
23 passed in 0.38s
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO26
- [ONNX Runtime](https://onnxruntime.ai/) for cross-platform inference
- [Gradio](https://gradio.app/) for the web demo framework
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for the tracking algorithm inspiration
