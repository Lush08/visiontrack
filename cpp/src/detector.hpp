#pragma once
/// VisionTrack — ONNX Runtime Detector (C++)

#include <string>
#include <vector>
#include <array>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

#include "postprocessor.hpp"

namespace vt {

class Detector {
public:
    Detector(const std::string& model_path,
             int input_size = 640,
             int num_classes = 80,
             float conf_threshold = 0.25f,
             float iou_threshold = 0.45f,
             bool use_cuda = false);

    /// Run inference on a preprocessed NCHW tensor. Returns raw output.
    std::vector<float> infer(const std::vector<float>& tensor, int channels, int height, int width);

    /// Postprocess raw output into detections (still in model-input coords).
    std::vector<Detection> postprocess(const std::vector<float>& output);

    int input_size() const { return input_size_; }
    int num_classes() const { return num_classes_; }

private:
    int input_size_;
    int num_classes_;
    float conf_threshold_;
    float iou_threshold_;

    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::vector<int64_t> input_shape_;
};

}  // namespace vt
