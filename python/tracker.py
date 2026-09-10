"""
VisionTrack - Multi-Object Tracker (ByteTrack-based)

A simplified ByteTrack implementation using IoU-based association.
Objects are matched between frames and assigned persistent track IDs.

Track lifecycle:
    Active → (no match for `max_missed` frames) → Lost → (removed)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.postprocessing import compute_iou_matrix


@dataclass
class Track:
    """A single tracked object with persistent identity."""

    id: int
    box: np.ndarray            # xyxy
    score: float
    class_id: int
    matches: int = 0           # Number of successful associations
    missed: int = 0            # Frames without a match
    history: list[np.ndarray] = field(default_factory=list)  # Past boxes

    @property
    def center(self) -> tuple[int, int]:
        """Center point of the bounding box."""
        x1, y1, x2, y2 = self.box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


class ByteTracker:
    """
    Simplified ByteTrack: greedy IoU-based association with track lifecycle.

    Keeps it simple but principled:
      - Match detected boxes to existing tracks by IoU, highest-confidence first
      - Unmatched high-confidence detections spawn new tracks
      - Tracks that miss too many frames are removed
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 30,
        min_matches_for_stable: int = 2,
    ):
        """
        Args:
            iou_threshold: Minimum IoU to consider a detection→track match
            max_missed: Frames a track can go unmatched before removal
            min_matches_for_stable: Matches needed before a track is "stable"
        """
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_matches_for_stable = min_matches_for_stable
        self.tracks: dict[int, Track] = {}
        self.next_id = 0
        self.frame_count = 0
        self.track_history: dict[int, list[tuple[int, int]]] = {}

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> dict[int, Track]:
        """
        Associate detections with existing tracks and update state.

        Args:
            boxes: Detected boxes in xyxy format (N, 4)
            scores: Confidence scores (N,)
            class_ids: Class IDs (N,)

        Returns:
            Updated {track_id: Track} map
        """
        self.frame_count += 1

        # No detections: age every existing track by one frame
        if len(boxes) == 0:
            for track in self.tracks.values():
                track.missed += 1
            self._remove_stale_tracks()
            return self.tracks

        # Snapshot of tracks that existed before this frame;
        # only these may be aged (newly-created tracks start fresh)
        existing_ids = set(self.tracks.keys())

        # Keep only active tracks for matching
        active_ids = [tid for tid, t in self.tracks.items() if t.missed == 0]
        if active_ids:
            active_boxes = np.array([self.tracks[tid].box for tid in active_ids])
            iou_matrix = compute_iou_matrix(boxes, active_boxes)
        else:
            iou_matrix = None

        matched_detections = set()
        matched_tracks = set()

        # Match detections to tracks: greedy by detection confidence
        if iou_matrix is not None:
            # Sort detections by confidence descending
            det_order = np.argsort(-scores)

            for det_idx in det_order:
                # Pick best track by IoU among unmatched tracks
                track_scores = iou_matrix[det_idx]
                # Restrict to unmatched tracks
                track_scores = np.array([
                    track_scores[j] if active_ids[j] not in matched_tracks else -1.0
                    for j in range(len(active_ids))
                ])
                best_track_j = int(np.argmax(track_scores))
                best_iou = track_scores[best_track_j]

                if best_iou >= self.iou_threshold:
                    track_id = active_ids[best_track_j]
                    track = self.tracks[track_id]
                    track.box = boxes[det_idx].copy()
                    track.score = float(scores[det_idx])
                    track.class_id = int(class_ids[det_idx])
                    track.matches += 1
                    track.missed = 0
                    track.history.append(track.box.copy())
                    self.track_history.setdefault(track_id, []).append(track.center)

                    matched_detections.add(int(det_idx))
                    matched_tracks.add(track_id)

        # Create new tracks for unmatched detections
        for det_idx in range(len(boxes)):
            if int(det_idx) in matched_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            new_track = Track(
                id=track_id,
                box=boxes[det_idx].copy(),
                score=float(scores[det_idx]),
                class_id=int(class_ids[det_idx]),
                matches=1,
                history=[boxes[det_idx].copy()],
            )
            self.tracks[track_id] = new_track
            self.track_history.setdefault(track_id, []).append(new_track.center)

        # Age unmatched pre-existing tracks (new tracks start with missed=0)
        for track_id in existing_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id].missed += 1

        # Remove tracks that exceeded max_missed
        self._remove_stale_tracks()

        return self.tracks

    def _remove_stale_tracks(self) -> None:
        """Remove tracks that exceeded max_missed frames."""
        stale = [
            tid for tid, t in self.tracks.items()
            if t.missed > self.max_missed
        ]
        for tid in stale:
            del self.tracks[tid]

    def get_active_tracks(self) -> list[Track]:
        """Return tracks that currently have a detection (missed == 0)."""
        return [t for t in self.tracks.values() if t.missed == 0]

    def reset(self) -> None:
        """Clear all tracks and reset state."""
        self.tracks.clear()
        self.track_history.clear()
        self.next_id = 0
        self.frame_count = 0

    def get_track_centers(self, track_id: int) -> list[tuple[int, int]]:
        """Center history for a track (used for trail drawing)."""
        return self.track_history.get(track_id, [])