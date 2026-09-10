/// VisionTrack — ONNX Runtime Detector (C++)

#include "detector.hpp"
#include <iostream>
#include <stdexcept>

namespace vt {

Detector::Detector(const std::string& model_path,
                   int input_size,
                   int num_classes,
                   float conf_threshold,
                   float iou_threshold,
                   bool use_cuda)
    : input_size_(input_size),
      num_classes_(num_classes),
      conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold),
      env_(ORT_LOGGING_LEVEL_WARNING, "VisionTrack"),
      session_(env_, model_path.c_str(), Ort::SessionOptions{})
{
    // Get input name and shape
    auto* input = session_.GetInputInfo(0);
    input_name_ = input->GetName();

    auto* type_info = input->GetTypeInfo();
    auto* tensor_info = type_info->GetTensorTypeAndShapeInfo();
    input_shape_ = tensor_info->GetShape();

    std::cout << "Model loaded: " << model_path << "\n"
              << "  Input: " << input_name_ << " [";
    for (size_t i = 0; i < input_shape_.size(); ++i) {
        std::cout << input_shape_[i] << (i + 1 < input_shape_.size() ? "," : "");
    }
    std::cout << "]\n"
              << "  Output: " << session_.GetOutputCount() << " tensor(s)\n";

    // Update input_size from model if dynamic
    if (input_shape_.size() == 4 && input_shape_[2] > 0) {
        input_size_ = static_cast<int>(input_shape_[2]);
    }
}

std::vector<float> Detector::infer(const std::vector<float>& tensor,
                                    int channels, int height, int width)
{
    std::vector<int64_t> shape = {1, channels, height, width};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(tensor.data()), tensor.size(),
        shape.data(), shape.size());

    const char* input_name = input_name_.c_str();
    auto output_tensors = session_.Run(
        Ort::RunOptions{}, &input_name, &input_tensor, 1, nullptr, 0);

    // Extract output data
    auto& out = output_tensors[0];
    auto* out_data = out.GetTensorMutableData<float>();
    auto out_shape = out.GetTensorTypeAndShapeInfo().GetShape();
    size_t out_size = 1;
    for (auto d : out_shape) out_size *= d;

    return std::vector<float>(out_data, out_data + out_size);
}

std::vector<Detection> Detector::postprocess(const std::vector<float>& output) {
    return decode_detections(output.data(), static_cast<int>(output.size()),
                             num_classes_, conf_threshold_, iou_threshold_);
}

}  // namespace vt
