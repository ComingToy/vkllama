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

    Pipeline::ShaderInfo stage0Info = { 1,
                                        2,
                                        3 * sizeof (int) + sizeof (float),
                                        (uint32_t)dev_->subgroup_size (),
                                        1,
                                        1 };

    const auto *spv_code = dtype_ == VkTensor::FP16
                               ? __get_reduce_fp16_comp_spv_code ()
                               : __get_reduce_comp_spv_code ();
    auto spv_size = dtype_ == VkTensor::FP16
                        ? __get_reduce_fp16_comp_spv_size ()
                        : __get_reduce_comp_spv_size ();

    auto op_type = op_type_ == 3 ? 0 : op_type_;
    stage0_.reset (
        new Pipeline (dev_, spv_code, spv_size, { op_type }, stage0Info));

    return stage0_->init ();
  }

  uint64_t
  time () noexcept override
  {
    return stage0_->time ();
  };

  VkResult
  operator() (VkTensor a, VkTensor &b)
  {
    if (a.dtype () != dtype_)
      {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
      }

    uint32_t group_x
        = (a.width () + dev_->subgroup_size () - 1) / dev_->subgroup_size (),
        group_y = a.height (), group_z = a.channels ();

    auto ret = stage0_->set_group (group_x, group_y, group_z);
    if (ret != VK_SUCCESS)
      {
        return ret;
      }

    b = VkTensor (a.channels (), a.height (), 1, dev_, dtype_);
    if ((ret = b.create ()) != VK_SUCCESS)
      {
        return ret;
      }

    float mean_scale
        = op_type_ != 3 ? 1.0f : 1.0f / static_cast<float> (a.width ());

    ret = command_->record_pipeline (
        *stage0_, { a, b },
        { (int)a.channels (), (int)a.height (), (int)a.width (), mean_scale });
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
  const int op_type_;
  const VkTensor::DType dtype_;
};
}

#endif
