"""Tests for VisionTrack ByteTracker."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from tracker import ByteTracker


class TestByteTracker:
    """Tests for IoU-based multi-object tracking."""

    def setup_method(self):
        self.tracker = ByteTracker(iou_threshold=0.3, max_missed=5)

    def _make_detection(self, x1, y1, x2, y2, score=0.9, class_id=0):
        """Helper to create a single detection."""
        return (
            np.array([[x1, y1, x2, y2]], dtype=np.float32),
            np.array([score], dtype=np.float32),
            np.array([class_id], dtype=int),
        )

    def _make_multi_detection(self, boxes, scores=None, class_ids=None):
        """Helper for multiple detections."""
        boxes = np.array(boxes, dtype=np.float32)
        n = len(boxes)
        if scores is None:
            scores = np.full(n, 0.9, dtype=np.float32)
        if class_ids is None:
            class_ids = np.zeros(n, dtype=int)
        return boxes, scores, class_ids

    def test_single_detection_creates_track(self):
        """First detection should create track with ID 0."""
        boxes, scores, cids = self._make_detection(100, 100, 200, 200)
        tracks = self.tracker.update(boxes, scores, cids)
        assert len(tracks) == 1
        assert 0 in tracks

    def test_persistence_same_position(self):
        """Same detection twice should keep same track ID."""
        boxes, scores, cids = self._make_detection(100, 100, 200, 200)
        self.tracker.update(boxes, scores, cids)
        self.tracker.update(boxes, scores, cids)

        tracks = self.tracker.get_active_tracks()
        assert len(tracks) == 1
        assert tracks[0].id == 0
        assert tracks[0].matches == 2

    def test_two_objects_separate_ids(self):
        """Two distinct detections should get different IDs."""
        b1, s1, c1 = self._make_detection(10, 10, 50, 50)
        b2, s2, c2 = self._make_detection(300, 300, 400, 400)
        boxes = np.vstack([b1, b2])
        scores = np.concatenate([s1, s2])
        cids = np.concatenate([c1, c2])

        tracks = self.tracker.update(boxes, scores, cids)
        assert len(tracks) == 2
        ids = {t.id for t in tracks.values()}
        assert ids == {0, 1}

    def test_aging_unmatched_track(self):
        """Unmatched track should have missed incremented."""
        # Create track
        boxes, scores, cids = self._make_detection(100, 100, 200, 200)
        self.tracker.update(boxes, scores, cids)

        # Send different detection — original track ages
        boxes2, scores2, cids2 = self._make_detection(400, 400, 500, 500)
        self.tracker.update(boxes2, scores2, cids2)

        # Check the original track aged
        old_track = self.tracker.tracks.get(0)
        assert old_track is not None
        assert old_track.missed >= 1

    def test_stale_track_removal(self):
        """Track should be removed after max_missed frames."""
        # Create track
        boxes, scores, cids = self._make_detection(100, 100, 200, 200)
        self.tracker.update(boxes, scores, cids)

        # Send no detections for max_missed + 1 frames
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_scores = np.empty(0, dtype=np.float32)
        empty_cids = np.empty(0, dtype=int)

        for _ in range(self.tracker.max_missed + 2):
            self.tracker.update(empty_boxes, empty_scores, empty_cids)

        assert 0 not in self.tracker.tracks

    def test_reappearance_gets_new_id(self):
        """Object that disappears and reappears gets a new ID."""
        # Create and remove track
        boxes, scores, cids = self._make_detection(100, 100, 200, 200)
        self.tracker.update(boxes, scores, cids)

        empty = np.empty((0, 4), dtype=np.float32)
        empty_s = np.empty(0, dtype=np.float32)
        empty_c = np.empty(0, dtype=int)
        for _ in range(self.tracker.max_missed + 2):
            self.tracker.update(empty, empty_s, empty_c)

        # Reappear
        self.tracker.update(boxes, scores, cids)
        tracks = self.tracker.get_active_tracks()
        assert len(tracks) == 1
        assert tracks[0].id != 0  # new ID

    def test_reset(self):
        """Reset should clear all state."""
        boxes, scores, cids = self._make_detection(100, 100, 200, 200)
        self.tracker.update(boxes, scores, cids)
        assert len(self.tracker.tracks) == 1

        self.tracker.reset()
        assert len(self.tracker.tracks) == 0
        assert self.tracker.next_id == 0
        assert self.tracker.frame_count == 0

    def test_no_detections_ages_all(self):
        """Empty frame should age all existing tracks."""
        boxes, scores, cids = self._make_detection(100, 100, 200, 200)
        self.tracker.update(boxes, scores, cids)

        empty = np.empty((0, 4), dtype=np.float32)
        empty_s = np.empty(0, dtype=np.float32)
        empty_c = np.empty(0, dtype=int)
        self.tracker.update(empty, empty_s, empty_c)

        assert self.tracker.tracks[0].missed == 1

    def test_track_history(self):
        """Track history should record center positions."""
        boxes, scores, cids = self._make_detection(100, 100, 200, 200)
        self.tracker.update(boxes, scores, cids)

        history = self.tracker.get_track_centers(0)
        assert len(history) == 1
        # Center of (100,100,200,200) is (150, 150)
        assert history[0] == (150, 150)
