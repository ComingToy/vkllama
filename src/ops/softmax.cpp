#include "src/ops/softmax.h"
#include "src/core/command.h"
#include "src/core/common.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"
#include "src/ops/reduce.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include <cstdio>
#include <memory>
#include <vector>

namespace vkllama
{
Softmax::Softmax (GPUDevice *dev, Command *command, bool seq_mask, float temp,
                  const Tensor::DType dtype)
    : Op (dev, command), seq_mask_ (seq_mask), dtype_ (dtype), temp_ (temp)
{
}

absl::Status
Softmax::init () noexcept
{
  if (dtype_ != FP16)
    {
      return absl::InvalidArgumentError (
          "Softmax op: only fp16 dtype is supported");
    }

  Pipeline::ShaderInfo info0 = { 1,
                                 2,
                                 sizeof (ShapeConstant) + sizeof (uint32_t),
                                 (uint32_t)dev_->subgroup_size (),
                                 2,
                                 1 };

  auto spv_code = __get_softmax_fp16_comp_spv_code ();
  auto spv_size = __get_softmax_fp16_comp_spv_size ();

  softmax0_.reset (
      new Pipeline (dev_, spv_code, spv_size, { (int)seq_mask_ }, info0));

  return softmax0_->init ();
}

uint64_t
Softmax::time () noexcept
{
  return softmax0_->time ();
}

absl::StatusOr<Tensor>
Softmax::operator() (Tensor a, size_t offset) noexcept
{
  if (a.dtype () != dtype_)
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "softmax op defined with %d dtype but input0's dtype = %d",
          int (dtype_), int (a.dtype ())));
    }

  if (out_.channels () != a.channels () || out_.height () != a.height ()
      || out_.width () != a.width ())
    {
      out_ = Tensor::like (a);
      VKLLAMA_STATUS_OK (out_.create ());
    }

  uint32_t group_x = 1, group_y = (a.height () + 1) / 2,
           group_z = a.channels ();

  auto ret = softmax0_->set_group (group_x, group_y, group_z);
  if (!ret.ok ())
    {
      return ret;
    }

  auto constants = a.shape_constant ();
  constants.push_back ((uint32_t)offset);

  ret = command_->record_pipeline (*softmax0_, { a, out_ }, constants);

  if (!ret.ok ())
    {
      return ret;
    }

  out_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  return out_;
}
}

