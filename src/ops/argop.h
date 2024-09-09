#ifndef __VKLLAMA_ARGOP_H__
#define __VKLLAMA_ARGOP_H__

#include "src/core/command.h"
#include "src/ops/op.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <cstdio>
#include <memory>
#include <vector>

namespace vkllama
{
template <int op_type> class ArgOp : public Op
{
public:
  ArgOp (GPUDevice *gpu, Command *command,
         const VkTensor::DType dtype = VkTensor::FP32)
      : Op (gpu, command), dtype_ (dtype)
  {
  }

  absl::Status
  init () noexcept override
  {
    Pipeline::ShaderInfo info0 = { 1, 2, sizeof (uint32_t) * 3, 32, 4, 1 };
    Pipeline::ShaderInfo info1 = { 1, 2, sizeof (uint32_t) * 3, 1, 128, 1 };

    if (dtype_ == VkTensor::FP16 && !dev_->support_16bit_storage ())
      {
        return absl::InvalidArgumentError (
            "fp16 dtype is unsupported on device");
      }

    const uint8_t *spv_code0 = nullptr;
    size_t spv_size0 = 0;
    const uint8_t *spv_code1 = nullptr;
    size_t spv_size1 = 0;
    if (dtype_ == VkTensor::FP16 && dev_->support_fp16_arithmetic ())
      {
        spv_code0 = __get_argmax_stage0_fp16a_comp_spv_code ();
        spv_size0 = __get_argmax_stage0_fp16a_comp_spv_size ();
        spv_code1 = __get_argmax_stage1_fp16a_comp_spv_code ();
        spv_size1 = __get_argmax_stage1_fp16a_comp_spv_size ();
      }
    else if (dtype_ == VkTensor::FP16)
      {
        spv_code0 = __get_argmax_stage0_fp16_comp_spv_code ();
        spv_size0 = __get_argmax_stage0_fp16_comp_spv_size ();
        spv_code1 = __get_argmax_stage1_comp_spv_code ();
        spv_size1 = __get_argmax_stage1_comp_spv_size ();
      }
    else if (dtype_ == VkTensor::FP32)
      {
        spv_code0 = __get_argmax_stage0_comp_spv_code ();
        spv_size0 = __get_argmax_stage0_comp_spv_size ();
        spv_code1 = __get_argmax_stage1_comp_spv_code ();
        spv_size1 = __get_argmax_stage1_comp_spv_size ();
      }
    else
      {
        return absl::InvalidArgumentError (
            absl::StrFormat ("dtype %i is unsupported", int (dtype_)));
      }

    pipeline0_.reset (
        new Pipeline (dev_, spv_code0, spv_size0, { op_type }, info0));

    pipeline1_.reset (
        new Pipeline (dev_, spv_code1, spv_size1, { op_type }, info1));

    auto ret = pipeline0_->init ();
    if (!ret.ok ())
      {
        return ret;
      }

    return pipeline1_->init ();
  }

  absl::Status
  operator() (VkTensor in, VkTensor &out) noexcept
  {
    if (in.dtype () != dtype_)
      {
        return absl::InvalidArgumentError (
            absl::StrFormat ("operator defined as %d dtype but %d given.",
                             int (dtype_), int (in.dtype ())));
      }

    out = VkTensor (in.channels (), in.height (), 1, dev_, VkTensor::UINT32);
    auto ret = out.create ();
    if (!ret.ok ())
      {
        return ret;
      }

    uint32_t group_x = (in.width () + 31) / 32,
             group_y = (in.height () + 3) / 4, group_z = in.channels ();

    if (!(ret = pipeline0_->set_group (group_x, group_y, group_z)).ok ())
      {
        return ret;
      }

    stage0_output_
        = VkTensor (in.channels (), in.height (), group_x * 2, dev_);

    if (!(ret = stage0_output_.create ()).ok ())
      {
        return ret;
      }

    ShaderConstants shape = { static_cast<uint32_t> (in.channels ()),
                              static_cast<uint32_t> (in.height ()),
                              static_cast<uint32_t> (in.width ()) };

    ret = command_->record_pipeline (*pipeline0_, { in, stage0_output_ },
                                     shape);
    if (!ret.ok ())
      {
        return ret;
      }

    stage0_output_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
    stage0_output_.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);

    group_y = (in.height () + 127) / 128;
    ret = pipeline1_->set_group (1, group_y, group_z);
    if (!ret.ok ())
      {
        return ret;
      }

    ShaderConstants shape1 = { static_cast<uint32_t> (in.channels ()),
                               static_cast<uint32_t> (in.height ()), group_x };

    ret = command_->record_pipeline (*pipeline1_, { stage0_output_, out },
                                     shape1);

    if (!ret.ok ())
      {
        return ret;
      }

    out.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
    out.set_pipeline_stage (VK_SHADER_STAGE_COMPUTE_BIT);

    return absl::OkStatus ();
  }

  uint64_t
  time () noexcept override
  {
    return pipeline0_->time () + pipeline1_->time ();
  }

private:
  std::unique_ptr<Pipeline> pipeline0_;
  std::unique_ptr<Pipeline> pipeline1_;
  VkTensor stage0_output_;
  VkTensor::DType dtype_;
};

using ArgMax = ArgOp<0>;
using ArgMin = ArgOp<1>;

}

#endif
