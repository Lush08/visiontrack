/// VisionTrack — Image Preprocessing (C++)

#include "preprocessor.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace vt {

LetterboxResult letterbox(const cv::Mat& image, int input_size, cv::Scalar pad_color) {
    int h = image.rows;
    int w = image.cols;

    double ratio = std::min(static_cast<double>(input_size) / h,
                            static_cast<double>(input_size) / w);
    int new_w = static_cast<int>(std::round(w * ratio));
    int new_h = static_cast<int>(std::round(h * ratio));

    int dw = input_size - new_w;
    int dh = input_size - new_h;
    int top  = dh / 2;
    int left = dw / 2;

    cv::Mat resized;
    if (new_h != h || new_w != w) {
        cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = image;
    }

    cv::Mat padded;
    cv::copyMakeBorder(resized, top, dh - top, left, dw - left,
                       cv::BORDER_CONSTANT, pad_color, padded);

    return {padded, ratio, ratio, top, left};
}

std::vector<float> to_tensor(const cv::Mat& letterboxed) {
    int h = letterboxed.rows;
    int w = letterboxed.cols;

    // BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

    // HWC float32, normalize [0,1]
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // CHW layout
    std::vector<float> tensor(1 * 3 * h * w);
    const auto* data = reinterpret_cast<const float*>(rgb.data);
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < h * w; ++i) {
            tensor[c * h * w + i] = data[i * 3 + c];
        }
    }
    return tensor;
}

std::vector<std::array<float, 4>> scale_boxes(
    const std::vector<std::array<float, 4>>& boxes,
    double scale_x, double scale_y,
    int pad_top, int pad_left,
    int orig_h, int orig_w)
{
    std::vector<std::array<float, 4>> result;
    result.reserve(boxes.size());

    for (auto& b : boxes) {
        float x1 = (b[0] - pad_left) / scale_x;
        float y1 = (b[1] - pad_top)  / scale_y;
        float x2 = (b[2] - pad_left) / scale_x;
        float y2 = (b[3] - pad_top)  / scale_y;

        x1 = std::clamp(x1, 0.0f, static_cast<float>(orig_w));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(orig_h));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(orig_w));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(orig_h));

        result.push_back({x1, y1, x2, y2});
    }
    return result;
}

}  // namespace vt
