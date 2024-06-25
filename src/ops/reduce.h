#ifndef __VKLLAMA_REDUCE_H__
#define __VKLLAMA_REDUCE_H__

#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
#include "src/ops/op.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <cstdio>
#include <memory>
#include <numeric>
#include <vector>

namespace vkllama
{
// 0: sum 1: max 2: min 3: mean
class Reduce : public Op
{
public:
  Reduce (GPUDevice *gpu, Command *command, int op_type,
          VkTensor::DType const dtype = VkTensor::FP32)
      : Op (gpu, command), op_type_ (op_type), dtype_ (dtype)
  {
  }

  VkResult
  init () noexcept override
  {
    if (dtype_ == VkTensor::FP16 && !dev_->support_16bit_storage ())
      {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
      }

    Pipeline::ShaderInfo stage0Info = { 1, 2, 3 * sizeof (int), 64, 4, 1 };
    Pipeline::ShaderInfo stage1Info
        = { 1, 2, 3 * sizeof (int) + sizeof (float), 1, 64, 1 };

    const auto *spv_code = dtype_ == VkTensor::FP16
                               ? __get_reduce_stage0_fp16_comp_spv_code ()
                               : __get_reduce_stage0_comp_spv_code ();
    auto spv_size = dtype_ == VkTensor::FP16
                        ? __get_reduce_stage0_fp16_comp_spv_size ()
                        : __get_reduce_stage0_comp_spv_size ();

    if (dtype_ == VkTensor::FP16 && dev_->support_fp16_arithmetic ())
      {
        spv_code = __get_reduce_stage0_fp16a_comp_spv_code ();
        spv_size = __get_reduce_stage0_fp16a_comp_spv_size ();
      }

    const auto *spv_code1 = dtype_ == VkTensor::FP16
                                ? __get_reduce_stage1_fp16_comp_spv_code ()
                                : __get_reduce_stage1_comp_spv_code ();
    auto spv_size1 = dtype_ == VkTensor::FP16
                         ? __get_reduce_stage1_fp16_comp_spv_size ()
                         : __get_reduce_stage1_comp_spv_size ();

    if (dtype_ == VkTensor::FP16 && dev_->support_fp16_arithmetic ())
      {
        spv_code1 = __get_reduce_stage1_fp16a_comp_spv_code ();
        spv_size1 = __get_reduce_stage1_fp16a_comp_spv_size ();
      }

    auto op_type = op_type_ == 3 ? 0 : op_type_;
    stage0_.reset (
        new Pipeline (dev_, spv_code, spv_size, { op_type }, stage0Info));

    stage1_.reset (
        new Pipeline (dev_, spv_code1, spv_size1, { op_type }, stage1Info));

    VkResult ret = stage0_->init ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    return stage1_->init ();
  }

  uint64_t
  time () noexcept override
  {
    return stage0_->time () + stage1_->time ();
  };

  VkResult
  operator() (VkTensor a, VkTensor &b)
  {
    if (a.dtype () != dtype_)
      {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
      }

    b = VkTensor (a.channels (), a.height (), 1, dev_, dtype_);
    VkResult ret = b.create ();
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    uint32_t group_x = (a.width () + 63) / 64, group_y = (a.height () + 3) / 4,
             group_z = a.channels ();
    ret = stage0_->set_group (group_x, group_y, group_z);
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    stage0_output_ = VkTensor (
        a.channels (), a.height (), group_x, dev_,
        dev_->support_fp16_arithmetic () ? VkTensor::FP16 : VkTensor::FP32);
    if ((ret = stage0_output_.create ()) != VK_SUCCESS)
      {
        return ret;
      }

    ret = command_->record_pipeline (
        *stage0_, { a, stage0_output_ },
        { (int)a.channels (), (int)a.height (), (int)a.width () });
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    stage0_output_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
    stage0_output_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    float mean_scale;
    if (op_type_ == 3)
      {
        mean_scale = 1.0f / static_cast<float> (a.width ());
      }
    else
      {
        mean_scale = 1.0f;
      }

    ShaderConstants constants
        = { (int)a.channels (), (int)a.height (), (int)group_x };

    if (dtype_ == VkTensor::FP16 && dev_->support_fp16_arithmetic ())
      {
        constants.push_back (__fp32_to_fp16 (mean_scale));
        constants.push_back (__fp32_to_fp16 (.0f)); // padding
      }
    else
      {
        constants.push_back (mean_scale);
      }

    ret = stage1_->set_group (1, group_y, group_z);
    ret = command_->record_pipeline (*stage1_, { stage0_output_, b },
                                     constants);
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    b.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
    b.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    return VK_SUCCESS;
  }

private:
  std::unique_ptr<Pipeline> stage0_;
  std::unique_ptr<Pipeline> stage1_;
  const int op_type_;
  VkTensor stage0_output_;
  const VkTensor::DType dtype_;
};
}

#endif
