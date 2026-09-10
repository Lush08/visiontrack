#pragma once
/// VisionTrack — Postprocessing (C++)
/// Decode YOLO output, NMS, confidence filtering.

#include <array>
#include <vector>

namespace vt {

struct Detection {
    std::array<float, 4> box;  // xyxy
    float score;
    int class_id;
};

/// Decode raw YOLO output (1, 84, 8400) into detections.
std::vector<Detection> decode_detections(
    const float* output,
    int output_size,           // total elements, e.g. 1*84*8400
    int num_classes = 80,
    float conf_threshold = 0.25f,
    float iou_threshold = 0.45f);

/// Compute IoU between two boxes.
float compute_iou(const std::array<float, 4>& a, const std::array<float, 4>& b);

/// Class-aware NMS.
std::vector<int> nms_multiclass(
    const std::vector<std::array<float, 4>>& boxes,
    const std::vector<float>& scores,
    const std::vector<int>& class_ids,
    float iou_threshold = 0.45f);

}  // namespace vt
