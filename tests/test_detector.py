"""Tests for VisionTrack detector and preprocessing."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add python/ to path so imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from utils.preprocessing import Preprocessor
from utils.postprocessing import decode_detections, compute_iou_matrix, nms_multiclass


class TestPreprocessor:
    """Tests for the YOLO preprocessing pipeline."""

    def setup_method(self):
        self.pre = Preprocessor(input_size=640)

    def test_letterbox_square_image(self):
        """Square image should not need padding."""
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        padded, scale, padding = self.pre.letterbox(img, 640)
        assert padded.shape == (640, 640, 3)
        assert padding == (0, 0)
        assert abs(scale[0] - 1.0) < 1e-6

    def test_letterbox_preserves_aspect_ratio(self):
        """Wide image should be letterboxed with top/bottom padding."""
        img = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)
        padded, scale, (top, left) = self.pre.letterbox(img, 640)
        assert padded.shape == (640, 640, 3)
        assert top > 0
        assert left == 0

    def test_letterbox_tall_image(self):
        """Tall image should be letterboxed with left/right padding."""
        img = np.random.randint(0, 255, (600, 300, 3), dtype=np.uint8)
        padded, scale, (top, left) = self.pre.letterbox(img, 640)
        assert padded.shape == (640, 640, 3)
        assert top == 0
        assert left > 0

    def test_to_tensor_shape(self):
        """Tensor should be (1, 3, H, W) float32."""
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        tensor = self.pre.to_tensor(img)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32

    def test_to_tensor_range(self):
        """Values should be in [0, 1]."""
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        tensor = self.pre.to_tensor(img)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_preprocess_end_to_end(self):
        """Full pipeline should produce correct shape."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, scale, padding = self.pre.preprocess(img)
        assert tensor.shape == (1, 3, 640, 640)

    def test_scale_boxes_identity(self):
        """No padding + scale=1 should return same boxes."""
        boxes = np.array([[100, 100, 200, 200], [50, 50, 300, 300]], dtype=np.float32)
        scaled = self.pre.scale_boxes(boxes, (1.0, 1.0), (0, 0), (640, 640))
        np.testing.assert_allclose(scaled, boxes, atol=1.0)

    def test_scale_boxes_with_padding(self):
        """Boxes should be shifted by padding offset, clipped to image bounds."""
        # Box [10,10,50,50] in model space with 20px top, 30px left padding
        # After removing padding: [-20, -10, 20, 30] → clipped to [0, 0, 20, 30]
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scaled = self.pre.scale_boxes(boxes, (1.0, 1.0), (20, 30), (1000, 1000))
        expected = np.array([[0, 0, 20, 30]], dtype=np.float32)
        np.testing.assert_allclose(scaled, expected, atol=1.0)


class TestPostprocessing:
    """Tests for NMS and detection decoding."""

    def test_decode_detections_empty(self):
        """Empty output should return empty detections."""
        output = np.zeros((84, 100), dtype=np.float32)
        boxes, scores, class_ids = decode_detections(output)
        assert len(boxes) == 0

    def test_decode_detections_single_detection(self):
        """A strong detection should survive NMS."""
        num_anchors = 100
        nc = 80
        output = np.zeros((nc + 4, num_anchors), dtype=np.float32)
        # Place a detection at anchor 0: box (320,320,100,100), class 0 score=0.9
        output[0, 0] = 320  # cx
        output[1, 0] = 320  # cy
        output[2, 0] = 100  # w
        output[3, 0] = 100  # h
        output[4, 0] = 0.9  # class 0 score

        boxes, scores, class_ids = decode_detections(output, num_classes=nc, conf_threshold=0.25)
        assert len(boxes) == 1
        assert scores[0] == pytest.approx(0.9, abs=0.01)
        assert class_ids[0] == 0

    def test_nms_suppresses_overlapping(self):
        """Two overlapping boxes of same class — keep only the best."""
        boxes = np.array([
            [100, 100, 200, 200],
            [110, 110, 210, 210],  # high overlap with first
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        class_ids = np.array([0, 0])

        keep = nms_multiclass(boxes, scores, class_ids, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_nms_keeps_different_classes(self):
        """Overlapping boxes of different classes should both survive."""
        boxes = np.array([
            [100, 100, 200, 200],
            [105, 105, 205, 205],  # high overlap but different class
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        class_ids = np.array([0, 1])

        keep = nms_multiclass(boxes, scores, class_ids, iou_threshold=0.5)
        assert len(keep) == 2

    def test_compute_iou_matrix(self):
        """IoU of identical boxes should be 1.0."""
        boxes = np.array([
            [100, 100, 200, 200],
            [100, 100, 200, 200],
        ], dtype=np.float32)
        iou = compute_iou_matrix(boxes, boxes)
        assert iou[0, 0] == pytest.approx(1.0)
        assert iou[0, 1] == pytest.approx(1.0)

    def test_compute_iou_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0."""
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[100, 100, 110, 110]], dtype=np.float32)
        iou = compute_iou_matrix(boxes1, boxes2)
        assert iou[0, 0] == pytest.approx(0.0)
