/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/selectors/dw_convolution_selector.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/depthwise_conv.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/depthwise_conv_3x3.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::unique_ptr<GPUOperation> SelectDWConvolutionAdreno(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def) {
  if (IsDepthwiseConv3x3Supported(attr)) {
    return absl::make_unique<DepthwiseConv3x3>(
        CreateDepthwiseConv3x3(gpu_info, op_def, attr));
  } else {
    return absl::make_unique<GPUOperation>(
        CreateDepthwiseConvolution2D(gpu_info, op_def, attr));
  }
}

std::unique_ptr<GPUOperation> SelectDWConvolutionPowerVR(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def) {
  if (IsDepthwiseConv3x3Supported(attr)) {
    return absl::make_unique<DepthwiseConv3x3>(
        CreateDepthwiseConv3x3(gpu_info, op_def, attr));
  } else {
    return absl::make_unique<GPUOperation>(
        CreateDepthwiseConvolution2D(gpu_info, op_def, attr));
  }
}

std::unique_ptr<GPUOperation> SelectDWConvolutionMali(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def) {
  const auto storage_type = op_def.src_tensors[0].storage_type;
  bool buffer_type = storage_type == TensorStorageType::BUFFER ||
                     storage_type == TensorStorageType::IMAGE_BUFFER;
  const MaliInfo mali_info = gpu_info.mali_info;
  if (IsDepthwiseConv3x3Supported(attr) && !mali_info.IsMidgard() &&
      !buffer_type && op_def.precision != CalculationsPrecision::F32) {
    return absl::make_unique<DepthwiseConv3x3>(
        CreateDepthwiseConv3x3(gpu_info, op_def, attr));
  } else {
    return absl::make_unique<GPUOperation>(
        CreateDepthwiseConvolution2D(gpu_info, op_def, attr));
  }
}
}  // namespace

std::unique_ptr<GPUOperation> SelectDWConvolution(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def) {
  if (gpu_info.IsAdreno()) {
    return SelectDWConvolutionAdreno(attr, gpu_info, op_def);
  } else if (gpu_info.IsPowerVR()) {
    return SelectDWConvolutionPowerVR(attr, gpu_info, op_def);
  } else if (gpu_info.IsMali()) {
    return SelectDWConvolutionMali(attr, gpu_info, op_def);
  } else {
    return SelectDWConvolutionAdreno(attr, gpu_info, op_def);
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
