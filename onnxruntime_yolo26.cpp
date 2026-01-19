#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>

// COCO class names
const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

// Structure to hold preprocessing results
struct PreprocessResult {
    std::vector<float> image_data;
    int real_shape;
    int top_pad;
    int left_pad;
};

// Preprocess the image (letterbox resize, normalize, and prepare tensor)
PreprocessResult preprocess(const cv::Mat& input_image, int input_width, int input_height) {
    PreprocessResult result;
    
    // Get the height and width of the input image
    int img_height = input_image.rows;
    int img_width = input_image.cols;
    
    // Calculate letterbox padding
    int obj_shape = std::max(img_height, img_width);
    result.real_shape = obj_shape;
    int top_pad = (obj_shape - img_height) / 2;
    int bottom_pad = obj_shape - img_height - top_pad;
    int left_pad = (obj_shape - img_width) / 2;
    int right_pad = obj_shape - img_width - left_pad;
    result.top_pad = top_pad;
    result.left_pad = left_pad;
    
    // Add padding
    cv::Mat padded_img;
    cv::copyMakeBorder(input_image, padded_img, top_pad, bottom_pad, left_pad, right_pad,
                       cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    
    // Convert BGR to RGB
    cv::Mat rgb_img;
    cv::cvtColor(padded_img, rgb_img, cv::COLOR_BGR2RGB);
    
    // Resize to input size
    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(input_width, input_height));
    
    // Normalize and prepare tensor (HWC -> CHW, normalize to [0, 1])
    result.image_data.resize(1 * 3 * input_height * input_width);
    
    // Convert to float and normalize
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);
    
    // Transpose from HWC to CHW format
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                result.image_data[c * input_height * input_width + h * input_width + w] =
                    resized_img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    
    return result;
}

// Clip boxes to image boundaries
void clip_boxes(std::vector<float>& boxes, int height, int width) {
    for (size_t i = 0; i < boxes.size(); i += 4) {
        boxes[i] = std::max(0.0f, std::min((float)width, boxes[i]));      // x1
        boxes[i + 1] = std::max(0.0f, std::min((float)height, boxes[i + 1])); // y1
        boxes[i + 2] = std::max(0.0f, std::min((float)width, boxes[i + 2]));  // x2
        boxes[i + 3] = std::max(0.0f, std::min((float)height, boxes[i + 3])); // y2
    }
}

// Scale boxes from model output to original image size
// img1_shape: [height, width] of the image the boxes are in (model input: [640, 640])
// boxes: boxes in xyxy format (when xywh=False)
// img0_shape: [height, width] of the target image (original image)
// padding: if True, adjust for YOLO-style padding
std::vector<float> scale_boxes(const std::vector<float>& boxes, int img1_height, int img1_width,
                                int img0_height, int img0_width, bool padding = true) {
    // Calculate gain and pad (matching Python logic)
    // gain = min(old/new) where old is img1, new is img0
    float gain = std::min((float)img1_height / img0_height, (float)img1_width / img0_width);
    
    // Calculate padding (wh padding)
    float pad_x = std::round((img1_width - img0_width * gain) / 2.0f - 0.1f);
    float pad_y = std::round((img1_height - img0_height * gain) / 2.0f - 0.1f);
    
    std::vector<float> scaled_boxes = boxes;
    
    // Adjust for padding if enabled
    if (padding) {
        for (size_t i = 0; i < scaled_boxes.size(); i += 4) {
            scaled_boxes[i] -= pad_x;      // x1
            scaled_boxes[i + 1] -= pad_y;  // y1
            scaled_boxes[i + 2] -= pad_x;  // x2
            scaled_boxes[i + 3] -= pad_y;  // y2
        }
    }
    
    // Scale by gain
    for (size_t i = 0; i < scaled_boxes.size(); ++i) {
        scaled_boxes[i] /= gain;
    }
    
    // Clip to image boundaries
    clip_boxes(scaled_boxes, img0_height, img0_width);
    
    return scaled_boxes;
}

// Draw single detection on image (matches Python version)
void draw_detections(cv::Mat& img, const std::vector<float>& box, float score, int class_id) {
    // Generate color palette (static, generated once)
    static std::vector<cv::Scalar> color_palette;
    if (color_palette.empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);
        for (size_t i = 0; i < CLASS_NAMES.size(); ++i) {
            color_palette.push_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
        }
    }
    
    // Extract box coordinates (xywh format: [x1, y1, w, h])
    float x1 = box[0];
    float y1 = box[1];
    float w = box[2];
    float h = box[3];
    
    // Get color for this class
    cv::Scalar color = color_palette[class_id % color_palette.size()];
    
    // Draw bounding box
    cv::rectangle(img, cv::Point((int)x1, (int)y1),
                 cv::Point((int)(x1 + w), (int)(y1 + h)), color, 2);
    
    // Create label
    std::string label = CLASS_NAMES[class_id] + ": " + std::to_string(score).substr(0, 4);
    
    // Get text size
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    
    // Calculate label position
    int label_x = (int)x1;
    int label_y = (y1 - 10 > text_size.height) ? (int)(y1 - 10) : (int)(y1 + 10);
    
    // Draw label background
    cv::rectangle(img, cv::Point(label_x, label_y - text_size.height),
                 cv::Point(label_x + text_size.width, label_y + text_size.height),
                 color, cv::FILLED);
    
    // Draw label text
    cv::putText(img, label, cv::Point(label_x, label_y),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// Postprocess model outputs
void postprocess(cv::Mat& input_image, const std::vector<Ort::Value>& outputs, int real_shape,
                 int input_width, int input_height, int img_height, int img_width,
                 float confidence_thres = 0.35f) {
    // Get output tensor (output[0][0] in Python)
    // Python: outputs = output[0][0] - gets first batch, first element
    // The output shape is [1, num_detections, features] where features >= 6
    auto output_tensor = outputs[0].GetTensorData<float>();
    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    
    // Output shape is [1, num_detections, 6] or [1, num_detections, num_classes+5]
    // We access the first batch element (batch=0), so shape[1] is num_detections
    if (output_shape.size() != 3 || output_shape[0] != 1) {
        std::cerr << "Unexpected output shape. Expected [1, num_detections, features], got [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cerr << output_shape[i];
            if (i < output_shape.size() - 1) std::cerr << ", ";
        }
        std::cerr << "]" << std::endl;
        return;
    }
    
    int rows = (int)output_shape[1];  // num_detections
    int cols = (int)output_shape[2];  // features (should be at least 6)
    
    if (cols < 6) {
        std::cerr << "Output tensor has insufficient features: " << cols << " (expected at least 6)" << std::endl;
        return;
    }
    
    std::vector<float> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    
    // Process each detection
    for (int i = 0; i < rows; ++i) {
        // Access first batch element: output[0][i]
        int offset = i * cols;
        float max_score = output_tensor[offset + 4];
        
        if (max_score >= confidence_thres) {
            int class_id = (int)output_tensor[offset + 5];
            
            // Extract box coordinates directly (outputs[i,:4] in Python)
            // Python passes outputs[i,:4] directly to scale_boxes with xywh=False
            // This means the model outputs are treated as xyxy format [x1, y1, x2, y2]
            std::vector<float> box_raw = {
                output_tensor[offset + 0],
                output_tensor[offset + 1],
                output_tensor[offset + 2],
                output_tensor[offset + 3]
            };
            
            // Scale boxes from model input size [640, 640] to original image size
            // Matching Python: scale_boxes([640,640], outputs[i,:4], (img_height, img_width), xywh=False)
            std::vector<float> scaled_box = scale_boxes(box_raw, input_height, input_width,
                                                        img_height, img_width, true);
            
            // Convert to xywh format for drawing (as Python code does: [x1, y1, w, h])
            // Python: boxes.append([int(new_bbox[0]),int(new_bbox[1]),int(new_bbox[2]-new_bbox[0]),int(new_bbox[3]-new_bbox[1])])
            float scaled_x1 = scaled_box[0];
            float scaled_y1 = scaled_box[1];
            float scaled_w = scaled_box[2] - scaled_box[0];
            float scaled_h = scaled_box[3] - scaled_box[1];
            
            boxes.push_back(scaled_x1);
            boxes.push_back(scaled_y1);
            boxes.push_back(scaled_w);
            boxes.push_back(scaled_h);
            
            scores.push_back(max_score);
            class_ids.push_back(class_id);
        }
    }
    
    // Draw detections (one at a time like Python code)
    for (size_t i = 0; i < class_ids.size(); ++i) {
        // Boxes are already stored in xywh format: [x1, y1, w, h]
        std::vector<float> box_xywh = {
            boxes[i * 4],      // x1
            boxes[i * 4 + 1],  // y1
            boxes[i * 4 + 2],  // w
            boxes[i * 4 + 3]   // h
        };
        draw_detections(input_image, box_xywh, scores[i], class_ids[i]);
    }
}

// Load model and run inference
void run_inference(const std::string& model_path, const std::string& image_path,
                   int input_width = 640, int input_height = 640) {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO26");
    Ort::SessionOptions session_options;
    
    // Load model
    std::cout << "Loading model from: " << model_path << std::endl;
    
    // Convert string to wchar_t* for Windows (ONNX Runtime on Windows requires wide strings)
#ifdef _WIN32
    std::wstring model_path_wide(model_path.begin(), model_path.end());
    Ort::Session session(env, model_path_wide.c_str(), session_options);
#else
    Ort::Session session(env, model_path.c_str(), session_options);
#endif
    
    std::cout << "Model loaded successfully." << std::endl;
    
    // Load and preprocess image
    cv::Mat input_image = cv::imread(image_path);
    if (input_image.empty()) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return;
    }
    
    PreprocessResult preprocessed = preprocess(input_image, input_width, input_height);
    
    // Get input/output names dynamically (like Python: session.get_inputs()[0].name)
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    
    // Prepare input tensor
    std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
    std::vector<const char*> input_names = {input_name.get()};
    std::vector<const char*> output_names = {output_name.get()};
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, preprocessed.image_data.data(), preprocessed.image_data.size(),
        input_shape.data(), input_shape.size());
    
    // Run inference
    std::cout << "Running inference..." << std::endl;
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                               output_names.data(), 1);
    std::cout << "Inference completed." << std::endl;
    
    // Postprocess
    int img_height = input_image.rows;
    int img_width = input_image.cols;
    postprocess(input_image, outputs, preprocessed.real_shape, input_width, input_height,
                img_height, img_width);
    
    // Save result
    cv::imwrite("result.jpg", input_image);
    std::cout << "Result saved to result.jpg" << std::endl;
    
    // Display result (optional, comment out if not needed)
    cv::imshow("YOLO26 Inference with Letterbox", input_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
    // Default paths
    std::string model_path = "yolo26n.onnx";
    std::string image_path = "bus.jpg";
    
    // Parse command line arguments if provided
    if (argc >= 2) {
        model_path = argv[1];
    }
    if (argc >= 3) {
        image_path = argv[2];
    }
    
    try {
        run_inference(model_path, image_path);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
