#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>

struct Detection
{
  float x1, y1, x2, y2, score;
  int class_id;
};

struct resizeParams
{
  int top_pad;
  int left_pad;
  float scale_x, scale_y;
};

enum class InputLayout
{
  NCHW,
  NHWC
};

static float*
aligned_alloc_buffer(size_t size, size_t alignment = 64)
{
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, sizeof(float) * size) != 0)
    throw std::bad_alloc();
  return reinterpret_cast<float*>(ptr);
}

// Preprocess the image (letterbox resize, normalize, and prepare tensor)
resizeParams
resizeLetterBox(const cv::Mat& input_image, cv::Mat& resized_img, int input_width, int input_height)
{
  resizeParams result;

  // Get the height and width of the input image
  int img_height = input_image.rows;
  int img_width = input_image.cols;

  // Calculate letterbox padding
  int obj_shape = std::max(img_height, img_width);
  int top_pad = (obj_shape - img_height) / 2;
  int bottom_pad = obj_shape - img_height - top_pad;
  int left_pad = (obj_shape - img_width) / 2;
  int right_pad = obj_shape - img_width - left_pad;
  result.top_pad = top_pad;
  result.left_pad = left_pad;

  // Add padding
  cv::Mat padded_img;
  cv::copyMakeBorder(input_image, padded_img, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  cv::Mat rgb_img;
  cv::cvtColor(padded_img, rgb_img, cv::COLOR_BGR2RGB);

  // Resize to input size
  cv::resize(padded_img, resized_img, cv::Size(input_width, input_height));

  // scale from model input back to padded image
  result.scale_x = static_cast<float>(obj_shape) / input_width;
  result.scale_y = static_cast<float>(obj_shape) / input_height;

  return result;
}

void
preprocess_nchw(const cv::Mat& img, float* dst, int H, int W, const float* mean, const float* std)
{
  const int hw = H * W;

  for (int y = 0; y < H; ++y)
  {
    const uchar* row = img.ptr<uchar>(y);
    for (int x = 0; x < W; ++x)
    {
      int idx = y * W + x;
      dst[idx] = (row[3 * x + 2] / 255.f - mean[0]) / std[0];          // R
      dst[idx + hw] = (row[3 * x + 1] / 255.f - mean[1]) / std[1];     // G
      dst[idx + 2 * hw] = (row[3 * x + 0] / 255.f - mean[2]) / std[2]; // B
    }
  }
}

void
preprocess_nhwc(const cv::Mat& img, float* dst, int H, int W, const float* mean, const float* std)
{
  int idx = 0;
  for (int y = 0; y < H; ++y)
  {
    const uchar* row = img.ptr<uchar>(y);
    for (int x = 0; x < W; ++x)
    {
      dst[idx++] = (row[3 * x + 2] / 255.f - mean[0]) / std[0];
      dst[idx++] = (row[3 * x + 1] / 255.f - mean[1]) / std[1];
      dst[idx++] = (row[3 * x + 0] / 255.f - mean[2]) / std[2];
    }
  }
}

//// Scale boxes from model output to original image size
// input_shape: [height, width] of the image the boxes are in (model input: [640, 640])
// boxes: boxes in xywh format
// img_shape: [height, width] of the target image (original image)
void
scale_boxes(float& xc, float& yc, float& w, float& h, resizeParams p)
{
  // Scale by gain
  float x = (xc - w / 2.f) * p.scale_x - p.left_pad; // x1
  float y = (yc - h / 2.f) * p.scale_y - p.top_pad;  // ye
  w *= p.scale_x;                                    // w
  h *= p.scale_y;                                    // h

  // Adjust for padding if enable
  xc = x + w / 2.f;
  yc = y + h / 2.f;
}
