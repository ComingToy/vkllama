#include "src/ops/softmax.h"
#include "src/core/command.h"
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
  if (dtype_ != Tensor::FP16)
    {
      return absl::InvalidArgumentError (
          "Softmax op: only fp16 dtype is supported");
    }

  Pipeline::ShaderInfo info0 = {
    1, 2, 4 * sizeof (uint32_t), (uint32_t)dev_->subgroup_size (), 1, 1
  };

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

  auto b = Tensor::like (a);
  if (auto ret = b.create (); !ret.ok ())
    {
      return ret;
    }

  uint32_t group_x
      = (a.width () + dev_->subgroup_size () - 1) / dev_->subgroup_size (),
      group_y = a.height (), group_z = a.channels ();

  auto ret = softmax0_->set_group (group_x, group_y, group_z);
  if (!ret.ok ())
    {
      return ret;
    }

  ShaderConstants constants = { (uint32_t)a.channels (), (uint32_t)a.height (),
                                (uint32_t)a.width (), (uint32_t)offset };

  ret = command_->record_pipeline (*softmax0_, { a, b }, constants);

  if (!ret.ok ())
    {
      return ret;
    }

  b.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  b.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  return b;
}
}

