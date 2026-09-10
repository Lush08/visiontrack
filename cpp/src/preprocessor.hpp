#pragma once
/// VisionTrack — Image Preprocessing (C++)
/// Letterbox resize, BGR→RGB, HWC→CHW, normalize [0,1].

#include <opencv2/core.hpp>
#include <tuple>
#include <vector>

namespace vt {

struct LetterboxResult {
    cv::Mat image;
    double scale_x;
    double scale_y;
    int pad_top;
    int pad_left;
};

/// Letterbox resize preserving aspect ratio with gray padding.
LetterboxResult letterbox(const cv::Mat& image, int input_size = 640,
                           cv::Scalar pad_color = cv::Scalar(114, 114, 114));

/// Convert letterboxed BGR image to NCHW float32 tensor [0,1].
/// Returns flat vector of size 1*3*H*W.
std::vector<float> to_tensor(const cv::Mat& letterboxed);

/// Map boxes from model-input space back to original image coordinates.
std::vector<std::array<float, 4>> scale_boxes(
    const std::vector<std::array<float, 4>>& boxes,
    double scale_x, double scale_y,
    int pad_top, int pad_left,
    int orig_h, int orig_w);

}  // namespace vt
