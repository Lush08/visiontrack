"""
VisionTrack - Postprocessing Utilities

NMS, box decoding, and score filtering for YOLO ONNX outputs.
Handles both raw output format and NMS-embedded output.
"""

import numpy as np


def decode_detections(
    output: np.ndarray,
    num_classes: int = 80,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    input_size: int = 640,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode YOLO model output into detections.

    Handles two common output formats:
    1. Raw: (1, 4 + num_classes, num_anchors) - requires manual NMS
    2. NMS-embedded: (num_dets, [x1, y1, x2, y2, score, class]) rows

    Args:
        output: Raw ONNX model output
        num_classes: Number of classes
        conf_threshold: Minimum confidence to keep a detection
        iou_threshold: IoU threshold for NMS
        input_size: Model input size

    Returns:
        (boxes_xyxy, scores, class_ids) filtered and NMS'd
    """
    output = np.asarray(output)

    # Squeeze leading singleton batch dimension if present
    out = output[0] if output.ndim == 3 and output.shape[0] == 1 else output

    # ---------------------------------------------------------------
    # Format detection
    #   Raw format:        (1, 4 + nc, num_anchors)  → dim1 small, dim2 large
    #   NMS-embedded:      (1, num_dets, 6)          → dim2 is exactly 6
    #   Transposed raw:    (1, num_anchors, 4 + nc)  → dim1 large
    # ---------------------------------------------------------------
    is_raw = (
        out.ndim == 2
        and out.shape[0] == 4 + num_classes
        and out.shape[1] > 4 + num_classes
    )

    nms_embedded = (
        out.ndim == 2
        and out.shape[1] == 6
        and (out.shape[0] == 0 or out.shape[0] < 1000)  # detections, not anchors
    )

    # Format 1: NMS-embedded - each row is one final detection (N, 6)
    if nms_embedded:
        dets = out
        boxes = dets[:, :4]
        scores = dets[:, 4]
        class_ids = dets[:, 5].astype(int)
        keep = scores >= conf_threshold
        return boxes[keep], scores[keep], class_ids[keep]

    # Format 2: Raw - (4 + nc, num_anchors) or (num_anchors, 4 + nc)
    if is_raw:
        # (4 + nc, num_anchors) → (num_anchors, 4 + nc)
        pred = out.T
    elif out.ndim == 2 and out.shape[1] == 4 + num_classes:
        # Already transposed: (num_anchors, 4 + nc)
        pred = out
    else:
        # Unknown format: try generic fallback
        pred = out

    if len(pred) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # Extract box coords (cx, cy, w, h) and class scores
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:4 + num_classes]

    # Convert to xyxy
    cx, cy, w, h = (
        boxes_xywh[:, 0],
        boxes_xywh[:, 1],
        boxes_xywh[:, 2],
        boxes_xywh[:, 3],
    )
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Get best class per anchor
    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    # Filter by confidence
    keep = scores >= conf_threshold
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores, class_ids

    # Apply class-wise NMS
    keep_indices = nms_multiclass(boxes_xyxy, scores, class_ids, iou_threshold)
    return boxes_xyxy[keep_indices], scores[keep_indices], class_ids[keep_indices]


def nms(iou_matrix: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """
    Single-class NMS using a precomputed IoU matrix.
    Simple O(n^2) greedy suppression.
    """
    n = iou_matrix.shape[0]
    suppressed = np.zeros(n, dtype=bool)
    keep: list[int] = []

    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(i)
        # Suppress all not-yet-suppressed boxes whose IoU with i exceeds threshold
        suppressed[i] = True
        suppressed[~suppressed & (iou_matrix[i] > iou_threshold)] = True

    return keep


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of xyxy boxes."""
    n1, n2 = len(boxes1), len(boxes2)
    iou = np.zeros((n1, n2))

    for i in range(n1):
        x1, y1, x2, y2 = boxes1[i]
        area1 = max(0, x2 - x1) * max(0, y2 - y1)
        if area1 <= 0:
            continue
        for j in range(n2):
            x1b, y1b, x2b, y2b = boxes2[j]
            area2 = max(0, x2b - x1b) * max(0, y2b - y1b)
            if area2 <= 0:
                continue
            xx1 = max(x1, x1b)
            yy1 = max(y1, y1b)
            xx2 = min(x2, x2b)
            yy2 = min(y2, y2b)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            union = area1 + area2 - inter
            iou[i, j] = inter / union if union > 0 else 0.0

    return iou


def nms_multiclass(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """
    Class-aware NMS. Boxes of different classes are not suppressed.

    Args:
        boxes: (N, 4) xyxy
        scores: (N,)
        class_ids: (N,)
        iou_threshold: IoU threshold

    Returns:
        Indices of kept boxes
    """
    keep_all: list[int] = []
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        cls_boxes = boxes[idx]
        cls_scores = scores[idx]

        iou = compute_iou_matrix(cls_boxes, cls_boxes)
        # Sort by score descending for determinism
        order = np.argsort(-cls_scores)

        suppressed = np.zeros(len(idx), dtype=bool)
        for i in order:
            if suppressed[i]:
                continue
            keep_all.append(int(idx[i]))
            # Suppress higher-overlap, lower-score boxes
            for j in order:
                if not suppressed[j] and i != j and iou[i, j] > iou_threshold:
                    if cls_scores[j] < cls_scores[i]:
                        suppressed[j] = True

    return np.array(sorted(keep_all))