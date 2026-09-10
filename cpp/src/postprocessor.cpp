/// VisionTrack — Postprocessing (C++)

#include "postprocessor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace vt {

float compute_iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
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

static std::vector<int> nms_single_class(
    const std::vector<std::array<float, 4>>& boxes,
    const std::vector<float>& scores,
    float iou_threshold)
{
    // Sort by score descending
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });

    std::vector<bool> suppressed(scores.size(), false);
    std::vector<int> keep;

    for (int i : order) {
        if (suppressed[i]) continue;
        keep.push_back(i);
        for (int j : order) {
            if (j == i || suppressed[j]) continue;
            if (compute_iou(boxes[i], boxes[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return keep;
}

std::vector<int> nms_multiclass(
    const std::vector<std::array<float, 4>>& boxes,
    const std::vector<float>& scores,
    const std::vector<int>& class_ids,
    float iou_threshold)
{
    std::vector<int> keep_all;

    // Group by class
    std::vector<int> classes;
    for (int cid : class_ids) {
        if (std::find(classes.begin(), classes.end(), cid) == classes.end()) {
            classes.push_back(cid);
        }
    }

    for (int cls : classes) {
        std::vector<int> indices;
        for (int i = 0; i < (int)class_ids.size(); ++i) {
            if (class_ids[i] == cls) indices.push_back(i);
        }

        std::vector<std::array<float, 4>> cls_boxes;
        std::vector<float> cls_scores;
        for (int i : indices) {
            cls_boxes.push_back(boxes[i]);
            cls_scores.push_back(scores[i]);
        }

        auto local_keep = nms_single_class(cls_boxes, cls_scores, iou_threshold);
        for (int k : local_keep) {
            keep_all.push_back(indices[k]);
        }
    }

    std::sort(keep_all.begin(), keep_all.end());
    return keep_all;
}

std::vector<Detection> decode_detections(
    const float* output,
    int output_size,
    int num_classes,
    float conf_threshold,
    float iou_threshold)
{
    // YOLO output format: (1, 4 + num_classes, num_anchors) → (84, 8400)
    // We treat the data as column-major: output[c * num_anchors + anchor_idx]
    const int nc = num_classes;
    const int num_anchors = output_size / (4 + nc);

    std::vector<std::array<float, 4>> all_boxes;
    std::vector<float> all_scores;
    std::vector<int> all_class_ids;

    all_boxes.reserve(num_anchors / 4);
    all_scores.reserve(num_anchors / 4);
    all_class_ids.reserve(num_anchors / 4);

    for (int a = 0; a < num_anchors; ++a) {
        float cx = output[0 * num_anchors + a];
        float cy = output[1 * num_anchors + a];
        float w  = output[2 * num_anchors + a];
        float h  = output[3 * num_anchors + a];

        // Find best class
        float max_score = 0.0f;
        int best_class = 0;
        for (int c = 0; c < nc; ++c) {
            float s = output[(4 + c) * num_anchors + a];
            if (s > max_score) {
                max_score = s;
                best_class = c;
            }
        }

        if (max_score < conf_threshold) continue;

        // xywh → xyxy
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        all_boxes.push_back({x1, y1, x2, y2});
        all_scores.push_back(max_score);
        all_class_ids.push_back(best_class);
    }

    if (all_boxes.empty()) return {};

    // Class-aware NMS
    auto keep = nms_multiclass(all_boxes, all_scores, all_class_ids, iou_threshold);

    std::vector<Detection> detections;
    detections.reserve(keep.size());
    for (int k : keep) {
        detections.push_back({all_boxes[k], all_scores[k], all_class_ids[k]});
    }
    return detections;
}

}  // namespace vt
