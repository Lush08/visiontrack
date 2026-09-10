/// VisionTrack — Multi-Object Tracker (C++)

#include "tracker.hpp"
#include <algorithm>
#include <numeric>
#include <unordered_set>

namespace vt {

ByteTracker::ByteTracker(float iou_threshold, int max_missed)
    : iou_threshold_(iou_threshold), max_missed_(max_missed) {}

static float iou_of(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    float x1 = std::max(a[0], b[0]);
    float y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]);
    float y2 = std::min(a[3], b[3]);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    float union_ = area_a + area_b - inter;
    return (union_ > 0) ? inter / union_ : 0.0f;
}

static std::array<float, 2> center(const std::array<float, 4>& box) {
    return {(box[0] + box[2]) / 2.0f, (box[1] + box[3]) / 2.0f};
}

std::vector<Track> ByteTracker::update(const std::vector<Detection>& detections) {
    frame_count++;

    if (detections.empty()) {
        for (auto& [id, t] : tracks) t.missed++;
        remove_stale();
        return get_active_tracks();
    }

    // Snapshot existing track IDs
    std::vector<int> existing_ids;
    std::vector<int> active_ids;
    for (auto& [id, t] : tracks) {
        existing_ids.push_back(id);
        if (t.missed == 0) active_ids.push_back(id);
    }

    // Build IoU matrix (detections × active tracks)
    std::vector<std::vector<float>> iou_matrix;
    std::vector<bool> matched_det(detections.size(), false);
    std::unordered_set<int> matched_tracks_set;

    if (!active_ids.empty()) {
        iou_matrix.resize(detections.size(), std::vector<float>(active_ids.size()));
        for (size_t d = 0; d < detections.size(); ++d) {
            for (size_t t = 0; t < active_ids.size(); ++t) {
                iou_matrix[d][t] = iou_of(detections[d].box, tracks[active_ids[t]].box);
            }
        }

        // Greedy matching: sort detections by score descending
        std::vector<int> det_order(detections.size());
        std::iota(det_order.begin(), det_order.end(), 0);
        std::sort(det_order.begin(), det_order.end(), [&](int a, int b) {
            return detections[a].score > detections[b].score;
        });

        for (int d : det_order) {
            int best_t = -1;
            float best_iou = iou_threshold_;
            for (size_t t = 0; t < active_ids.size(); ++t) {
                if (matched_tracks_set.count(active_ids[t])) continue;
                if (iou_matrix[d][t] > best_iou) {
                    best_iou = iou_matrix[d][t];
                    best_t = static_cast<int>(t);
                }
            }
            if (best_t >= 0) {
                int track_id = active_ids[best_t];
                auto& track = tracks[track_id];
                track.box = detections[d].box;
                track.score = detections[d].score;
                track.class_id = detections[d].class_id;
                track.matches++;
                track.missed = 0;
                track.history.push_back(center(detections[d].box));
                track_history[track_id].push_back(center(detections[d].box));

                matched_det[d] = true;
                matched_tracks_set.insert(track_id);
            }
        }
    }

    // New tracks for unmatched detections
    for (size_t d = 0; d < detections.size(); ++d) {
        if (matched_det[d]) continue;
        int id = next_id++;
        Track t;
        t.id = id;
        t.box = detections[d].box;
        t.score = detections[d].score;
        t.class_id = detections[d].class_id;
        t.matches = 1;
        t.history.push_back(center(detections[d].box));
        tracks[id] = t;
        track_history[id].push_back(center(detections[d].box));
    }

    // Age unmatched pre-existing tracks
    for (int id : existing_ids) {
        if (matched_tracks_set.find(id) == matched_tracks_set.end()) {
            tracks[id].missed++;
        }
    }

    remove_stale();
    return get_active_tracks();
}

std::vector<Track> ByteTracker::get_active_tracks() const {
    std::vector<Track> result;
    for (auto& [id, t] : tracks) {
        if (t.missed == 0) result.push_back(t);
    }
    return result;
}

void ByteTracker::reset() {
    tracks.clear();
    track_history.clear();
    next_id = 0;
    frame_count = 0;
}

void ByteTracker::remove_stale() {
    std::vector<int> stale;
    for (auto& [id, t] : tracks) {
        if (t.missed > max_missed_) stale.push_back(id);
    }
    for (int id : stale) tracks.erase(id);
}

}  // namespace vt
