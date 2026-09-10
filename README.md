# VisionTrack

**Real-Time Multi-Object Detection & Tracking System** with Python training, C++ ONNX inference, and a live Gradio web demo.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/c++-17-blue.svg)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.16+-orange.svg)](https://onnxruntime.ai/)

## Overview

VisionTrack is an end-to-end computer vision pipeline that demonstrates:

- **Object Detection** using [YOLO26](https://docs.ultralytics.com/models/yolo26/) (Ultralytics, Jan 2026) -- the latest state-of-the-art model
- **Multi-Object Tracking** using IoU-based ByteTrack for consistent object IDs across frames
- **Dual Inference Backends**: Python (ONNX Runtime) for convenience, C++ (ONNX Runtime) for production speed
- **Interactive Web Demo** via Gradio with image upload, video processing, and live webcam support
- **Training Pipeline** for fine-tuning on custom datasets

## Demo

> Live demo coming soon on HuggingFace Spaces!

![Demo Screenshot](assets/demo.gif)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VisionTrack Pipeline                   │
├──────────────┬──────────────────┬───────────────────────┤
│   Training   │    Inference     │      Demo Layer       │
│   (Python)   │  (Python/C++)    │     (Gradio)          │
├──────────────┼──────────────────┼───────────────────────┤
│ Ultralytics  │ ONNX Runtime     │ Image Upload          │
│ YOLO26       │ Preprocessing    │ Video Processing      │
│ Custom Data  │ NMS + Decode     │ Live Webcam           │
│ ONNX Export  │ ByteTrack        │ Confidence Controls   │
└──────────────┴──────────────────┴───────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection Model | YOLO26 (Ultralytics) |
| Training | Python, PyTorch |
| Model Format | ONNX (simplified, opset 17) |
| C++ Inference | ONNX Runtime C++ API |
| Object Tracking | ByteTrack (IoU-based) |
| Web Demo | Gradio + HuggingFace Spaces |
| Build System | CMake (C++), pip (Python) |
| CI/CD | GitHub Actions |

## Quick Start

### Python Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/visiontrack.git
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
python python/demo_gradio.py
```

Open http://localhost:7860 in your browser.

### Run Inference on an Image

```bash
python python/demo_tracker.py --source image.jpg --output result.jpg
```

### Run Inference on Video

```bash
python python/demo_tracker.py --source video.mp4 --output result.mp4 --show
```

### C++ Build (Windows)

```bash
# Build with CMake
cd cpp
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release

# Run inference
.\Release\visiontrack.exe --model models/yolo26n.onnx --source image.jpg
```

## Training on Custom Data

```bash
# Train YOLO26 on your dataset
python python/train.py --data data/custom.yaml --model yolo26n --epochs 100

# Export trained model to ONNX
python python/export_onnx.py --weights runs/detect/train/weights/best.pt
```

See [notebooks/02_training.ipynb](notebooks/02_training.ipynb) for a full training walkthrough.

## Benchmarks

> Benchmarks coming soon after implementation.

| Backend | Model | Input Size | Latency (ms) | FPS |
|---------|-------|-----------|--------------|-----|
| Python ONNX | YOLO26n | 640x640 | TBD | TBD |
| C++ ONNX | YOLO26n | 640x640 | TBD | TBD |

## Project Structure

```
visiontrack/
├── python/                # Python inference, training, and demo
│   ├── train.py           # YOLO26 fine-tuning script
│   ├── export_onnx.py     # ONNX model export
│   ├── demo_gradio.py     # Gradio web application
│   ├── demo_tracker.py    # CLI detection + tracking pipeline
│   └── utils/             # Visualization and dataset helpers
├── cpp/                   # C++ ONNX Runtime inference engine
│   ├── src/               # Detector, tracker, preprocessor
│   ├── include/           # Public headers
│   └── CMakeLists.txt     # C++ build config
├── notebooks/             # Jupyter notebooks (training, benchmarks)
├── tests/                 # Unit tests
├── models/                # Model weights (gitignored)
├── data/                  # Training data (gitignored)
└── assets/                # Demo GIFs, architecture diagrams
```

## What This Project Demonstrates

| Skill | Evidence |
|-------|----------|
| Machine Learning | YOLO26 training, fine-tuning, evaluation metrics |
| Computer Vision | Object detection, tracking, preprocessing pipelines |
| Python | Type-annotated code, clean architecture, docstrings |
| C++ | ONNX Runtime integration, memory management, CMake |
| Model Deployment | ONNX export, cross-platform inference |
| Web Development | Gradio interactive demo |
| Software Engineering | Unit tests, CI/CD, modular design |
| Documentation | Professional README, Jupyter notebooks |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO26
- [ONNX Runtime](https://onnxruntime.ai/) for cross-platform inference
- [Gradio](https://gradio.app/) for the web demo framework
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for the tracking algorithm inspiration
