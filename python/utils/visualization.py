"""
VisionTrack - Visualization Utilities

Drawing bounding boxes, labels, track IDs, and annotations on images/video frames.
"""

from typing import Optional

import cv2
import numpy as np

# COCO class names (80 classes) - used as fallback when no custom labels provided
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Distinct colors for up to 64 track IDs
TRACK_COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 0), (0, 128, 128),
]


def get_color(track_id: int) -> tuple[int, int, int]:
    """Get a consistent color for a given track ID."""
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: Optional[list[str]] = None,
    track_ids: Optional[np.ndarray] = None,
    conf_threshold: float = 0.25,
    draw_labels: bool = True,
    draw_confidence: bool = True,
    draw_tracks: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes, labels, and track IDs on a frame.

    Args:
        frame: BGR image (H, W, 3)
        boxes: Bounding boxes in xyxy format (N, 4)
        scores: Confidence scores (N,)
        class_ids: Class indices (N,)
        class_names: List of class name strings
        track_ids: Optional track IDs (N,)
        conf_threshold: Minimum confidence to draw
        draw_labels: Whether to draw class labels
        draw_confidence: Whether to draw confidence scores
        draw_tracks: Whether to draw track IDs

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    if class_names is None:
        class_names = COCO_CLASSES

    for i in range(len(boxes)):
        if scores[i] < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, boxes[i])
        class_id = int(class_ids[i])
        score = scores[i]

        # Determine color
        if track_ids is not None and draw_tracks:
            color = get_color(int(track_ids[i]))
            label_parts = [f"ID:{int(track_ids[i])}"]
        else:
            color = get_color(class_id)
            label_parts = []

        # Add class label
        if draw_labels:
            name = class_names[class_id] if class_id < len(class_names) else f"cls_{class_id}"
            label_parts.append(name)

        # Add confidence
        if draw_confidence:
            label_parts.append(f"{score:.2f}")

        label = " ".join(label_parts)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        if label:
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return annotated


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter on frame."""
    label = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_track_trail(
    frame: np.ndarray,
    track_history: dict[int, list[tuple[int, int]]],
    max_trail_length: int = 50,
) -> np.ndarray:
    """
    Draw track trails (motion history) for each tracked object.

    Args:
        frame: BGR image
        track_history: Dict mapping track_id -> list of center points
        max_trail_length: Maximum number of points to keep in trail
    """
    annotated = frame.copy()

    for track_id, points in track_history.items():
        if len(points) < 2:
            continue

        color = get_color(track_id)
        recent_points = points[-max_trail_length:]

        for j in range(1, len(recent_points)):
            # Fade effect: older points are more transparent
            alpha = j / len(recent_points)
            thickness = max(1, int(3 * alpha))
            cv2.line(
                annotated,
                recent_points[j - 1],
                recent_points[j],
                color,
                thickness,
            )

    return annotated
