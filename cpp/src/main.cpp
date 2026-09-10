/// VisionTrack — C++ CLI Inference Driver
///
/// Usage:
///   visiontrack detect image.jpg --output result.jpg
///   visiontrack detect video.mp4 --output result.mp4
///   visiontrack benchmark --iterations 100

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "detector.hpp"
#include "tracker.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"

// COCO class names (80 classes)
static const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
};

static const std::vector<cv::Scalar> COLORS = {
    {230, 25, 75}, {60, 180, 75}, {255, 225, 25}, {0, 130, 200},
    {245, 130, 48}, {145, 30, 180}, {70, 240, 240}, {240, 50, 230},
    {210, 245, 60}, {250, 190, 212}, {0, 128, 128}, {220, 190, 255},
    {170, 110, 40}, {255, 250, 200}, {128, 0, 0}, {170, 255, 195},
};

struct Args {
    std::string command = "detect";
    std::string source;
    std::string output;
    std::string model = "models/yolo26n.onnx";
    int imgsz = 640;
    float conf = 0.25f;
    float iou = 0.45f;
    int num_classes = 80;
    bool gpu = false;
    bool draw_trails = false;
    int benchmark_iters = 100;
};

static void print_usage() {
    std::cout <<
        "VisionTrack — Real-Time Object Detection & Tracking (C++)\n\n"
        "Usage:\n"
        "  visiontrack detect <image|video> [options]\n"
        "  visiontrack benchmark [options]\n\n"
        "Options:\n"
        "  --output <path>      Output file path\n"
        "  --model <path>       ONNX model path (default: models/yolo26n.onnx)\n"
        "  --imgsz <int>        Input size (default: 640)\n"
        "  --conf <float>       Confidence threshold (default: 0.25)\n"
        "  --iou <float>        NMS IoU threshold (default: 0.45)\n"
        "  --gpu                Use CUDA if available\n"
        "  --draw-trails        Draw motion trails\n"
        "  --iterations <int>   Benchmark iterations (default: 100)\n";
}

static Args parse_args(int argc, char* argv[]) {
    Args args;
    if (argc < 2) { print_usage(); exit(1); }
    args.command = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) args.output = argv[++i];
        else if (arg == "--model" && i + 1 < argc) args.model = argv[++i];
        else if (arg == "--imgsz" && i + 1 < argc) args.imgsz = std::stoi(argv[++i]);
        else if (arg == "--conf" && i + 1 < argc) args.conf = std::stof(argv[++i]);
        else if (arg == "--iou" && i + 1 < argc) args.iou = std::stof(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) args.benchmark_iters = std::stoi(argv[++i]);
        else if (arg == "--gpu") args.gpu = true;
        else if (arg == "--draw-trails") args.draw_trails = true;
        else if (args.source.empty() && arg[0] != '-') args.source = arg;
    }
    return args;
}

/// Annotate a frame with detections.
static cv::Mat annotate(const cv::Mat& frame,
                        const std::vector<vt::Detection>& dets,
                        const std::vector<int>& track_ids = {}) {
    cv::Mat out = frame.clone();
    for (size_t i = 0; i < dets.size(); ++i) {
        auto& d = dets[i];
        auto& box = d.box;
        int x1 = static_cast<int>(box[0]);
        int y1 = static_cast<int>(box[1]);
        int x2 = static_cast<int>(box[2]);
        int y2 = static_cast<int>(box[3]);

        cv::Scalar color = COLORS[d.class_id % COLORS.size()];
        cv::rectangle(out, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        std::string label;
        if (!track_ids.empty()) {
            label = "ID:" + std::to_string(track_ids[i]) + " ";
        }
        label += (d.class_id < (int)COCO_CLASSES.size() ? COCO_CLASSES[d.class_id] : "?");
        label += " " + std::to_string(d.score).substr(0, 4);

        int baseline = 0;
        auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(out, cv::Point(x1, y1 - sz.height - 8), cv::Point(x1 + sz.width, y1), color, -1);
        cv::putText(out, label, cv::Point(x1, y1 - 4), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    return out;
}

static void detect_image(const Args& args) {
    cv::Mat frame = cv::imread(args.source);
    if (frame.empty()) {
        std::cerr << "Error: Cannot read image: " << args.source << "\n";
        exit(1);
    }

    vt::Detector detector(args.model, args.imgsz, args.num_classes, args.conf, args.iou, args.gpu);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto lb = vt::letterbox(frame, detector.input_size());
    auto tensor = vt::to_tensor(lb.image);
    auto raw_out = detector.infer(tensor, 3, lb.image.rows, lb.image.cols);
    auto dets = detector.postprocess(raw_out);
    auto boxes = vt::scale_boxes(
        [&]() {
            std::vector<std::array<float, 4>> b;
            for (auto& d : dets) b.push_back(d.box);
            return b;
        }(),
        lb.scale_x, lb.scale_y, lb.pad_top, lb.pad_left, frame.rows, frame.cols);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Update detection boxes to original coords
    for (size_t i = 0; i < dets.size(); ++i) dets[i].box = boxes[i];

    auto annotated = annotate(frame, dets);

    std::string out_path = args.output.empty()
        ? args.source.substr(0, args.source.rfind('.')) + "_detected.png"
        : args.output;
    cv::imwrite(out_path, annotated);
    std::cout << "Detected " << dets.size() << " objects in " << ms << " ms\n"
              << "  Saved: " << out_path << "\n";
}

static void detect_video(const Args& args) {
    cv::VideoCapture cap(args.source);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video: " << args.source << "\n";
        exit(1);
    }

    vt::Detector detector(args.model, args.imgsz, args.num_classes, args.conf, args.iou, args.gpu);
    vt::ByteTracker tracker;

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 20.0;
    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::string out_path = args.output.empty()
        ? args.source.substr(0, args.source.rfind('.')) + "_tracked.mp4"
        : args.output;

    cv::VideoWriter writer(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(w, h));
    int frame_count = 0;

    cv::Mat frame;
    while (cap.read(frame)) {
        auto lb = vt::letterbox(frame, detector.input_size());
        auto tensor = vt::to_tensor(lb.image);
        auto raw_out = detector.infer(tensor, 3, lb.image.rows, lb.image.cols);
        auto dets = detector.postprocess(raw_out);
        auto boxes = vt::scale_boxes(
            [&]() {
                std::vector<std::array<float, 4>> b;
                for (auto& d : dets) b.push_back(d.box);
                return b;
            }(),
            lb.scale_x, lb.scale_y, lb.pad_top, lb.pad_left, frame.rows, frame.cols);
        for (size_t i = 0; i < dets.size(); ++i) dets[i].box = boxes[i];

        tracker.update(dets);
        auto active = tracker.get_active_tracks();

        std::vector<int> track_ids;
        std::vector<vt::Detection> active_dets;
        for (auto& t : active) {
            track_ids.push_back(t.id);
            active_dets.push_back({t.box, t.score, t.class_id});
        }

        auto annotated = annotate(frame, active_dets, track_ids);
        writer.write(annotated);
        frame_count++;

        if (frame_count % 50 == 0) {
            std::cout << "  Frame " << frame_count << "...\r" << std::flush;
        }
    }

    cap.release();
    writer.release();
    std::cout << "\nProcessed " << frame_count << " frames → " << out_path << "\n";
}

static void benchmark(const Args& args) {
    vt::Detector detector(args.model, args.imgsz, args.num_classes, args.conf, args.iou, args.gpu);

    // Create dummy input
    std::vector<float> tensor(1 * 3 * args.imgsz * args.imgsz, 0.5f);

    // Warmup
    std::cout << "Warming up (10 iterations)...\n";
    for (int i = 0; i < 10; ++i) {
        detector.infer(tensor, 3, args.imgsz, args.imgsz);
    }

    // Benchmark
    std::cout << "Running " << args.benchmark_iters << " iterations...\n";
    std::vector<double> latencies;
    latencies.reserve(args.benchmark_iters);

    for (int i = 0; i < args.benchmark_iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto out = detector.infer(tensor, 3, args.imgsz, args.imgsz);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // Stats
    std::sort(latencies.begin(), latencies.end());
    double sum = 0;
    for (double l : latencies) sum += l;
    double mean = sum / latencies.size();
    double median = latencies[latencies.size() / 2];
    double p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];

    std::cout << "\n=== Benchmark Results ===\n"
              << "  Model:    " << args.model << "\n"
              << "  Input:    " << args.imgsz << "x" << args.imgsz << "\n"
              << "  Iters:    " << args.benchmark_iters << "\n"
              << "  Mean:     " << mean << " ms (" << 1000.0 / mean << " FPS)\n"
              << "  Median:   " << median << " ms (" << 1000.0 / median << " FPS)\n"
              << "  P95:      " << p95 << " ms\n"
              << "  P99:      " << p99 << " ms\n"
              << "  Min:      " << latencies.front() << " ms\n"
              << "  Max:      " << latencies.back() << " ms\n";
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    try {
        if (args.command == "detect") {
            if (args.source.empty()) {
                std::cerr << "Error: No input source specified.\n";
                print_usage();
                return 1;
            }
            // Check if image or video
            auto ext = std::filesystem::path(args.source).extension().string();
            std::vector<std::string> img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"};
            bool is_image = false;
            for (auto& e : img_exts) {
                if (ext == e) { is_image = true; break; }
            }
            if (is_image) detect_image(args);
            else detect_video(args);
        } else if (args.command == "benchmark") {
            benchmark(args);
        } else {
            std::cerr << "Unknown command: " << args.command << "\n";
            print_usage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
