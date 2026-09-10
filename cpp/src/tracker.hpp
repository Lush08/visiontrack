#pragma once
/// VisionTrack — Multi-Object Tracker (C++)
/// ByteTrack-inspired IoU-based association.

#include <vector>
#include <array>
#include <unordered_map>

#include "postprocessor.hpp"

namespace vt {

struct Track {
    int id;
    std::array<float, 4> box;  // xyxy
    float score;
    int class_id;
    int matches = 0;
    int missed = 0;
    std::vector<std::array<float, 2>> history;  // center points
};

class ByteTracker {
public:
    ByteTracker(float iou_threshold = 0.3f, int max_missed = 30);

    /// Update tracker with new detections. Returns all active tracks.
    std::vector<Track> update(const std::vector<Detection>& detections);

    /// Get tracks with missed == 0 (currently visible).
    std::vector<Track> get_active_tracks() const;

    /// Clear all state.
    void reset();

    std::unordered_map<int, Track> tracks;
    int next_id = 0;
    int frame_count = 0;
    std::unordered_map<int, std::vector<std::array<float, 2>>> track_history;

private:
    float iou_threshold_;
    int max_missed_;
    void remove_stale();
};

}  // namespace vt
