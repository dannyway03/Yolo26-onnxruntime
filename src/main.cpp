#include "yolox_detector.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "instrumentation/global_metrics.hpp"
#include "instrumentation/profiler.hpp"
namespace fs = std::filesystem;

#include <fstream>
#include <iomanip>

void
write_mot17_dets(std::ofstream& ofs, int frame_id, const std::vector<Detection>& dets)
{
  for (const auto& d : dets)
  {
    float w = d.x2 - d.x1;
    float h = d.y2 - d.y1;
    if (w <= 0.f || h <= 0.f)
      continue;

    ofs << frame_id << ",-1," << std::fixed << std::setprecision(2) << d.x1 << "," << d.y1 << "," << w << "," << h
        << "," << d.score << ",-1,-1,-1\n";
  }
}

void
overlay_detections(cv::Mat& image, const std::vector<Detection>& detections, float score_thresh = 0,
                   const cv::Scalar& box_color = {0, 255, 0}, int thickness = 2, bool draw_score = true)
{
  // cv::resize(image, image, cv::Size(640, 480));
  for (const auto& det : detections)
  {
    if (det.score < score_thresh)
      continue;

    int x1 = static_cast<int>(std::round(det.x1));
    int y1 = static_cast<int>(std::round(det.y1));
    int x2 = static_cast<int>(std::round(det.x2));
    int y2 = static_cast<int>(std::round(det.y2));

    // Clamp for visualization only
    x1 = std::max(0, std::min(x1, image.cols - 1));
    y1 = std::max(0, std::min(y1, image.rows - 1));
    x2 = std::max(0, std::min(x2, image.cols - 1));
    y2 = std::max(0, std::min(y2, image.rows - 1));

    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), box_color, thickness);

    if (draw_score)
    {
      std::ostringstream ss;
      ss << std::fixed << std::setprecision(2) << det.score;
      std::string label = ss.str();

      int baseline = 0;
      cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

      int tx = x1;
      int ty = std::max(0, y1 - text_size.height - 4);

      cv::rectangle(image, cv::Rect(tx, ty, text_size.width + 6, text_size.height + 6), box_color, cv::FILLED);

      cv::putText(image, label, cv::Point(tx + 3, ty + text_size.height + 2), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
  }
}

void
print_usage(const char* prog)
{
  std::cout << "Usage: " << prog << " [options]\n"
            << "Options:\n"
            << "  -m, --model <path>   ONNX model path\n"
            << "  --yolo26             Use YOLO26 mode (xyxy output, no normalization)\n"
            << "  --mot17 <path>       MOT17 dataset path\n"
            << "  -h, --help           Show this help\n"
            << "\nLayout (NCHW/NHWC) and input size are auto-detected from the model.\n"
            << "Set NO_VIZ=1 to disable visualization.\n";
}

int
main(int argc, char** argv)
{
  bool show_viz = std::getenv("NO_VIZ") == nullptr;
  bool is_yolo26 = false;
  std::string onnx_model = "models/bytetrack-n_mot17_1x480x640x3.onnx";
  std::string mot17_base = "/home/briox/Software/datasets/MOT17/train";

  // Parse command line args
  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];
    if (arg == "--yolo26")
    {
      is_yolo26 = true;
    }
    else if ((arg == "-m" || arg == "--model") && i + 1 < argc)
    {
      onnx_model = argv[++i];
    }
    else if (arg == "--mot17" && i + 1 < argc)
    {
      mot17_base = argv[++i];
    }
    else if (arg == "-h" || arg == "--help")
    {
      print_usage(argv[0]);
      return 0;
    }
  }

  std::cout << "Model: " << onnx_model << (is_yolo26 ? " (YOLO26 mode)" : " (ByteTrack mode)") << std::endl;

  // Input size and layout auto-detected from ONNX model
  YoloXNano detector(onnx_model, is_yolo26);

  for (auto& seq : fs::directory_iterator(mot17_base))
  {
    PROFILE_SCOPE("OVERALL");
    if (!fs::is_directory(seq))
      continue;
    fs::create_directories(seq.path() / "byteyolo_det");

    int frame_id = 1;
    std::ofstream ofs(seq.path() / "byteyolo_det/det.txt");
    std::cout << "Processing " << seq.path().stem().string() << std::endl;

    std::string pattern = mot17_base + "/" + seq.path().stem().string() + "/img1/%06d.jpg";
    cv::VideoCapture cap(pattern);

    while (true)
    {
      PROFILE_SCOPE(seq.path().c_str());
      cv::Mat frame;
      cap >> frame;
      if (frame.empty())
        break;

      std::vector<Detection> dets;
      {
        PROFILE_SCOPE("INFERENCE");
        detector.infer(frame, dets);
      }

      write_mot17_dets(ofs, frame_id, dets);
      frame_id++;

      if (show_viz)
      {
        overlay_detections(frame, dets);
        cv::imshow("byteyolo_det", frame);
        if ((char)cv::waitKey(1) == 'q')
          break;
      }

      if (frame_id % 100 == 0)
        std::cout << "  frame " << frame_id << " Overall latency: " << PROFILER_GET_AVG_STAT(seq.path().c_str(), "Wall")
                  << " ms/frame - Inference: " << PROFILER_GET_AVG_STAT("INFERENCE", "Wall") << std::endl;
    }
    ofs.close();
  }
  std::cout << PROFILER_GET_AVG_STAT("OVERALL", "Walltime") << " ms/frame" << std::endl;

  return 0;
}
