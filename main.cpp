/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
/* for OpenCV */
#include <opencv2/opencv.hpp>
#ifdef ARMNN_DELEGATE
// for armnn delegate
#include <armnn_delegate.hpp>
#endif

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

using namespace tflite;
/* Model parameters */
#define WORK_DIR                      RESOURCE_DIR
#define MODEL_NAME  "yolov5n-fp16.tflite"
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input_1"
#define INPUT_DIMS  { 1, 640, 640, 3 }
#define IS_NCHW     false   //batch channel height width
#define IS_RGB      true
#define OUTPUT_NAME "Identity"
static constexpr int32_t kGridScaleList[] = { 8, 16, 32 };
static constexpr int32_t kGridChannel = 3;
static constexpr int32_t kNumberOfClass = 80;
static constexpr int32_t kElementNumOfAnchor = kNumberOfClass + 5;    // x, y, w, h, bbox confidence, [class confidence]
#define DEFAULT_INPUT_IMAGE RESOURCE_DIR"/kite.jpg"
#define LABEL_NAME   "label_coco_80.txt"


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

/**
 * @param   org                 输入图像
 * @param   dst                 输出图像    这个是根据tensor info 来设置的
 * @param   crop_x              裁剪的左上角 0 , 0
 * @param   crop_w              裁剪的长宽   960,717
 * @param   is_rgb              默认值true
 * @param   crop_type           kCropTypeExpand
 * @param   resize_by_linear    默认true  这个就是选择什么样的resize方法而已
 * @brief   1. resize
 *          2. crop
 *          这里是做了一个保持长款比的缩放
 */
void CropResizeCvt(const cv::Mat& org, cv::Mat& dst, int32_t& crop_x, int32_t& crop_y, int32_t& crop_w, int32_t& crop_h) {
  printf("orig image height: %d\n", org.rows);
  printf("orig image width : %d\n", org.cols);
  cv::Mat src = org(cv::Rect(crop_x, crop_y, crop_w, crop_h));
  printf("crop_x: %d\n", crop_x);
  printf("crop_y: %d\n", crop_y);
  printf("crop_w: %d\n", crop_w);
  printf("crop_h: %d\n", crop_h);

  printf("dst height: %d\n", dst.rows);
  printf("dst width : %d\n", dst.cols);

  float aspect_ratio_src = static_cast<float>(src.cols) / src.rows;
  float aspect_ratio_dst = static_cast<float>(dst.cols) / dst.rows;
  printf("aspect_ratio_src: %f\n", aspect_ratio_src);
  printf("aspect_ratio_dst: %f\n", aspect_ratio_dst);

  cv::Rect target_rect(0, 0, dst.cols, dst.rows);
  if (aspect_ratio_src > aspect_ratio_dst) {
    target_rect.height = static_cast<int32_t>(target_rect.width / aspect_ratio_src);
    target_rect.y = (dst.rows - target_rect.height) / 2;
  } else {
    target_rect.width = static_cast<int32_t>(target_rect.height * aspect_ratio_src);
    target_rect.x = (dst.cols - target_rect.width) / 2;
  }
  cv::Mat target = dst(target_rect);
  printf("target x: %d\n", target_rect.x);
  printf("target y: %d\n", target_rect.y);
  printf("target width: %d\n", target_rect.width);
  printf("target height: %d\n", target_rect.height);

  cv::resize(src, target, target.size(), 0, 0, cv::INTER_LINEAR);
  // crop_x -= target_rect.x * crop_w / target_rect.width;
  // crop_y -= target_rect.y * crop_h / target_rect.height;
  // crop_w = dst.cols * crop_w / target_rect.width;
  // crop_h = dst.rows * crop_h / target_rect.height;
  // printf("crop_x: %d\n", crop_x);
  // printf("crop_y: %d\n", crop_y);
  // printf("crop_w: %d\n", crop_w);
  // printf("crop_h: %d\n", crop_h);
  printf("converting bgr to rgb!\n");
  cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
}

void PreProcessImage(const cv::Mat input_img, float* dst) {
  const int32_t img_width = input_img.cols;
  const int32_t img_height = input_img.rows;
  const int32_t img_channel = input_img.channels();
  uint8_t* src = (uint8_t*)(input_img.data);
  for (int32_t i = 0; i < img_width * img_height; i++) {
    for (int32_t c = 0; c < img_channel; c++) {
      dst[i * img_channel + c] = (src[i * img_channel + c]) / 255.;
    }
  }
}

class BoundingBox {
public:
    BoundingBox()
        :class_id(0), label(""), score(0), x(0), y(0), w(0), h(0)
    {}

    BoundingBox(int32_t _class_id, std::string _label, float _score, int32_t _x, int32_t _y, int32_t _w, int32_t _h)
        :class_id(_class_id), label(_label), score(_score), x(_x), y(_y), w(_w), h(_h)
    {}

    int32_t     class_id;
    std::string label;
    float       score;
    int32_t     x;
    int32_t     y;
    int32_t     w;
    int32_t     h;
};

void GetBoundingBox(const float* data, float scale_x, float  scale_y, int32_t grid_w, int32_t grid_h, std::vector<BoundingBox>& bbox_list)
{
    int32_t index = 0;
    for (int32_t grid_y = 0; grid_y < grid_h; grid_y++) {
        for (int32_t grid_x = 0; grid_x < grid_w; grid_x++) {
            for (int32_t grid_c = 0; grid_c < kGridChannel; grid_c++) {
                float box_confidence = data[index + 4];
                if (box_confidence >= 0.4) {
                    int32_t class_id = 0;
                    float confidence = 0;
                    for (int32_t class_index = 0; class_index < kNumberOfClass; class_index++) {
                        float confidence_of_class = data[index + 5 + class_index];
                        if (confidence_of_class > confidence) {
                            confidence = confidence_of_class;
                            class_id = class_index;
                        }
                    }

                    if (confidence >= 0.4) {
                        int32_t cx = static_cast<int32_t>((data[index + 0] + 0) * scale_x);     // no need to + grid_x
                        int32_t cy = static_cast<int32_t>((data[index + 1] + 0) * scale_y);     // no need to + grid_y
                        int32_t w = static_cast<int32_t>(data[index + 2] * scale_x);            // no need to exp
                        int32_t h = static_cast<int32_t>(data[index + 3] * scale_y);            // no need to exp
                        int32_t x = cx - w / 2;
                        int32_t y = cy - h / 2;
                        bbox_list.push_back(BoundingBox(class_id, "", confidence, x, y, w, h));
                    }
                }
                index += kElementNumOfAnchor;
            }
        }
    }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

#ifdef ARMNN_DELEGATE
    // Create the ArmNN Delegate
  std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    // std::string backends = "GpuAcc";
  armnnDelegate::DelegateOptions delegateOptions(backends);
  std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
                        theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                                         armnnDelegate::TfLiteArmnnDelegateDelete);
  // Modify armnnDelegateInterpreter to use armnnDelegate
  interpreter->ModifyGraphWithDelegate(theArmnnDelegate.get());
#endif

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors
  // Note: The buffer of the input tensor with index `i` of type T can
  // 按照新的输入张量的大小重新分配内存
  interpreter->AllocateTensors();
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // 输入张量信息,获取其编号
  int input_tensor_id;
  for (auto i : interpreter->inputs()) {
    const TfLiteTensor* tensor = interpreter->tensor(i);
    if (std::string(tensor->name) == INPUT_NAME) {
      input_tensor_id = i; // 编号
      printf("input tensor id: %d\n", input_tensor_id);
      std::cout << "name: " << std::string(tensor->name) << std::endl;
      std::cout << "input tensor type: " <<  tensor->type << std::endl;// 1-->float32 这里其实是要把mat都除以255，变成float的，在processimage里做了
    }
  }
  // 输出张量信息，获取其数据
  float* out_tensor_data_0;
  for (auto i : interpreter->outputs()) {
    const TfLiteTensor* tensor = interpreter->tensor(i);
    if (std::string(tensor->name) == OUTPUT_NAME) {
      out_tensor_data_0 = interpreter->typed_tensor<float>(i); // 数据
      printf("output tensor id:: %d\n", i );
      std::cout << "name: " << std::string(tensor->name) << std::endl;
      std::cout << "out tensor type: " <<  tensor->type << std::endl;// 1-->float32
    }
  }

  // 加载图片
  cv::Mat origin_img = cv::imread(DEFAULT_INPUT_IMAGE);
  int crop_w = origin_img.cols;
  int crop_h = origin_img.rows;
  int crop_x = 0;
  int crop_y = 0;
  cv::Mat img_src = cv::Mat::zeros(640, 640, CV_8UC3);
  CropResizeCvt(origin_img, img_src, crop_x, crop_y, crop_w, crop_h);
  // cv::namedWindow("img_src", 0);
  // cv::imshow("img_src", img_src);
  // cv::waitKey(0);

    // /* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
    // float mean[3];// 0
    // float norm[3];// 1
    // mean[0] = 0.0f;
    // mean[1] = 0.0f;
    // mean[2] = 0.0f;
    // norm[0] = 1.0f / 255.0f;
    // norm[1] = 1.0f / 255.0f;
    // norm[2] = 1.0f / 255.0f;
    // for (int32_t i = 0; i < 3; i++) {
    //     mean[i] *= 255.0f;
    //     norm[i] *= 255.0f;
    //     norm[i] = 1.0f / norm[i];
    // }

  // 灌入数据
  printf("input_tensor_id: %d\n", input_tensor_id);
  float* dst = interpreter->typed_tensor<float>(input_tensor_id); 
  PreProcessImage(img_src, dst);

  // Run inference
  const auto& time_start = std::chrono::steady_clock::now();
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  const auto& time_end = std::chrono::steady_clock::now();
  double time_used = (time_end - time_start).count() / 1000000.0;
  std::cout << "time used for inference: " << time_used << std::endl;
  // Read output buffers
  // TODO(user): Insert getting data out code.
  // 读取结果
  // std::vector<BoundingBox> bbox_list;
  // float* output_data = out_tensor_data_0;
  //   for (const auto& scale : kGridScaleList) {
  //       int32_t grid_w = 640 / scale;
  //       int32_t grid_h = 640 / scale;
  //       float scale_x = static_cast<float>(crop_w);      /* scale to original image */
  //       float scale_y = static_cast<float>(crop_h);
  //       GetBoundingBox(output_data, scale_x, scale_y, grid_w, grid_h, bbox_list);
  //       output_data += grid_w * grid_h * kGridChannel * kElementNumOfAnchor;
  //   }
  //   std::cout << "-----------------------------------------------------" << std::endl;
  //   std::cout << "Get bbox size: " << bbox_list.size() << std::endl;
  // cv::namedWindow("result", 0);
  // cv::imshow("result", input_image);
  // cv::waitKey(0);
  return 0;
}
