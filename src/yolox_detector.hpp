#pragma once
#include "utils.hpp"

#include <vector>

#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>

class YoloXNano
{
public:
  YoloXNano(const std::string& model_path, bool isXYXY = false);
  ~YoloXNano();
  float
  iou(const Detection& a, const Detection& b);
  void
  nms(const std::vector<Detection>& input, std::vector<Detection>& output, float iou_thresh);
  void
  infer(const cv::Mat& frame, std::vector<Detection>& detections);

private:
  Ort::Env env_;
  std::unique_ptr<Ort::IoBinding> io_binding_;
  std::shared_ptr<Ort::Session> session_;
  Ort::SessionOptions session_opts_;

  float* input_buffer_;  // aligned NHWC input
  float* output_buffer_; // aligned output
  Ort::Value input_tensor_;
  Ort::Value output_tensor_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  int H, W;
  InputLayout layout;
  resizeParams preproc_params_;
  bool xyxy;

  void
  allocate_buffers();
};

YoloXNano::YoloXNano(const std::string& model_path, bool isXYXY) :
  env_(ORT_LOGGING_LEVEL_WARNING, "YoloX"), session_opts_(), xyxy(isXYXY)
{
  session_opts_.SetIntraOpNumThreads(1);
  session_opts_.SetInterOpNumThreads(1);
  session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // OpenVINO EP
  OrtOpenVINOProviderOptions ov_options{};
  ov_options.device_type = "CPU";
  ov_options.num_of_threads = 4;
  ov_options.enable_dynamic_shapes = false;

  session_opts_.AppendExecutionProvider_OpenVINO(ov_options);

  session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(), session_opts_);

  Ort::AllocatorWithDefaultOptions allocator;
  Ort::AllocatedStringPtr name = session_->GetInputNameAllocated(0, allocator);
  input_names_.push_back(strdup(name.get()));
  name = session_->GetOutputNameAllocated(0, allocator);
  output_names_.push_back(strdup(name.get()));

  allocate_buffers();  // H, W, layout auto-detected from model

  io_binding_ = std::make_unique<Ort::IoBinding>(*session_);
  io_binding_->BindInput(input_names_[0], input_tensor_);
  io_binding_->BindOutput(output_names_[0], output_tensor_);
}

void
YoloXNano::allocate_buffers()
{
  // Extract input shape from ONNX model
  auto in_typeinfo = session_->GetInputTypeInfo(0);
  auto in_info = in_typeinfo.GetTensorTypeAndShapeInfo();
  input_shape_ = in_info.GetShape();

  // Detect layout and extract H, W from model input shape
  // NCHW: [N, C, H, W] - channel (1 or 3) at index 1
  // NHWC: [N, H, W, C] - channel (1 or 3) at index 3
  if (input_shape_.size() == 4)
  {
    if (input_shape_[1] == 3 || input_shape_[1] == 1)
    {
      layout = InputLayout::NCHW;
      H = static_cast<int>(input_shape_[2]);
      W = static_cast<int>(input_shape_[3]);
    }
    else if (input_shape_[3] == 3 || input_shape_[3] == 1)
    {
      layout = InputLayout::NHWC;
      H = static_cast<int>(input_shape_[1]);
      W = static_cast<int>(input_shape_[2]);
    }
    else
    {
      throw std::runtime_error("Cannot detect layout: channel dim not at index 1 or 3");
    }
  }
  else
  {
    throw std::runtime_error("Expected 4D input, got " + std::to_string(input_shape_.size()) + "D");
  }

  std::cout << "Model input: " << (layout == InputLayout::NCHW ? "NCHW" : "NHWC") << " ["
            << input_shape_[0] << "," << input_shape_[1] << "," << input_shape_[2] << "," << input_shape_[3]
            << "] H=" << H << " W=" << W << std::endl;

  size_t input_size = static_cast<size_t>(H * W * 3);
  input_buffer_ = aligned_alloc_buffer(input_size);
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  input_tensor_ =
    Ort::Value::CreateTensor<float>(mem_info, input_buffer_, input_size, input_shape_.data(), input_shape_.size());

  // Extract output shape from ONNX model
  auto out_typeinfo = session_->GetOutputTypeInfo(0);
  auto out_info = out_typeinfo.GetTensorTypeAndShapeInfo();
  auto out_shape = out_info.GetShape();

  // Handle dynamic shapes (-1) with sensible defaults
  int64_t num_preds = (out_shape.size() > 1 && out_shape[1] > 0) ? out_shape[1] : 8400;
  int64_t num_attrs = (out_shape.size() > 2 && out_shape[2] > 0) ? out_shape[2] : 6;

  output_shape_ = {1, num_preds, num_attrs};
  output_buffer_ = aligned_alloc_buffer(num_preds * num_attrs);

  output_tensor_ = Ort::Value::CreateTensor<float>(mem_info, output_buffer_, num_preds * num_attrs,
                                                   output_shape_.data(), output_shape_.size());
}

YoloXNano::~YoloXNano()
{
  free(input_buffer_);
  free(output_buffer_);
}

float
YoloXNano::iou(const Detection& a, const Detection& b)
{
  float x1 = std::max(a.x1, b.x1);
  float y1 = std::max(a.y1, b.y1);
  float x2 = std::min(a.x2, b.x2);
  float y2 = std::min(a.y2, b.y2);
  float w = std::max(0.f, x2 - x1);
  float h = std::max(0.f, y2 - y1);
  float inter = w * h;
  float union_area = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return inter / union_area;
}

void
YoloXNano::nms(const std::vector<Detection>& input, std::vector<Detection>& output, float iou_thresh)
{
  std::vector<Detection> dets = input;
  std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b)
            {
              return a.score > b.score;
            });

  std::vector<bool> suppressed(dets.size(), false);
  for (size_t i = 0; i < dets.size(); i++)
  {
    if (suppressed[i])
      continue;
    output.push_back(dets[i]);
    for (size_t j = i + 1; j < dets.size(); j++)
    {
      if (suppressed[j])
        continue;
      if (dets[i].class_id == dets[j].class_id && iou(dets[i], dets[j]) > iou_thresh)
        suppressed[j] = true;
    }
  }
}

void
YoloXNano::infer(const cv::Mat& frame, std::vector<Detection>& detections)
{
  cv::Mat resized;
  preproc_params_ = resizeLetterBox(frame, resized, W, H);

  static constexpr float MEAN_NORM[3] = {0.485f, 0.456f, 0.406f};
  static constexpr float STD_NORM[3] = {0.229f, 0.224f, 0.225f};
  static constexpr float MEAN_RAW[3] = {0.f, 0.f, 0.f};
  static constexpr float STD_RAW[3] = {1.f, 1.f, 1.f};

  const float* MEAN = xyxy ? MEAN_RAW : MEAN_NORM;
  const float* STD = xyxy ? STD_RAW : STD_NORM;

  float* ptr = input_buffer_; // NHWC: [1,480,640,3]

  if (layout == InputLayout::NHWC)
    preprocess_nhwc(resized, ptr, H, W, MEAN, STD);
  else
    preprocess_nchw(resized, ptr, H, W, MEAN, STD);

  session_->Run(Ort::RunOptions{nullptr}, *io_binding_);

  std::vector<Detection> raw_dets;

  for (size_t i = 0; i < output_shape_[1]; ++i)
  {
    float xc = output_buffer_[i * 6 + 0];
    float yc = output_buffer_[i * 6 + 1];
    float w = output_buffer_[i * 6 + 2];
    float h = output_buffer_[i * 6 + 3];
    float score = xyxy ? output_buffer_[i * 6 + 4] : output_buffer_[i * 6 + 4] * output_buffer_[i * 6 + 5];

    if (score < 0.2)
      continue;

    int class_id = 0; // static_cast<int>(output_buffer_[i * 6 + 5]);
    float x1, y1, x2, y2;
    if (xyxy)
    {
      x1 = xc * preproc_params_.scale_x - preproc_params_.left_pad;
      y1 = yc * preproc_params_.scale_y - preproc_params_.top_pad;
      x2 = w * preproc_params_.scale_x - preproc_params_.left_pad;
      y2 = h * preproc_params_.scale_y - preproc_params_.top_pad;
    }
    else
    {
      scale_boxes(xc, yc, w, h, preproc_params_);
      x1 = xc - w / 2.f;
      y1 = yc - h / 2.f;
      x2 = xc + w / 2.f;
      y2 = yc + h / 2.f;
    }
    // x1 = std::max(0.0f, std::min(w, x1)); // x1
    // y1 = std::max(0.0f, std::min(h, y1)); // y1
    // x2 = std::max(0.0f, std::min(w, x2)); // x2
    // y2 = std::max(0.0f, std::min(h, y2)); // y2

    raw_dets.emplace_back(x1, y1, x2, y2, score, class_id);
  }

  nms(raw_dets, detections, 0.4f);
}
