#include "rms_norm.h"
#include "src/core/command.h"
#include "src/core/pipeline.h"
#include "src/shaders/vkllama_comp_shaders.h"

namespace vkllama
{
RMSNorm::RMSNorm (GPUDevice *dev, Command *command, Tensor weight,
                  const float eps_, const Tensor::DType dtype)
    : Op (dev, command), weight_ (weight), dtype_ (dtype)
{
  Pipeline::ShaderInfo info = {
    2, 3, sizeof (ShapeConstant), (uint32_t)dev_->subgroup_size (), 4, 1
  };

  const auto *spv_code = __get_rms_norm_fp16_comp_spv_code ();
  const auto spv_size = __get_rms_norm_fp16_comp_spv_size ();

  pipeline_.reset (
      new Pipeline (dev_, spv_code, spv_size, { 2.0f, eps_ }, info));
}

absl::Status
RMSNorm::init () noexcept
{
  if (dtype_ != FP16)
    {
      return absl::InvalidArgumentError (
          "RMSNorm op: only fp16 dtype is supported.");
    }

  if (weight_.dtype () != FP32)
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "rms_norm op defined with %d dtype but the dtype of weight is %d",
          int (dtype_), int (weight_.dtype ())));
    }

  auto ret = pipeline_->init ();
  if (!ret.ok ())
    {
      return ret;
    }

  if (!(ret = pipeline_->update_bindings ({ weight_ }, { 1 })).ok ())
    {
      return ret;
    }

  return absl::OkStatus ();
}

uint64_t
RMSNorm::time () noexcept
{
  return pipeline_->time ();
}

absl::StatusOr<Tensor>
RMSNorm::operator() (Tensor x) noexcept
{
  if (x.dtype () != dtype_)
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "rms_norm op defined with %d dtype but the dtype of input0 is %d",
          int (dtype_), int (x.dtype ())));
    }

  auto output
      = Tensor (x.channels (), x.height (), x.width (), dev_, dtype_, false);
  auto ret = output.create ();
  if (!ret.ok ())
    {
      return ret;
    }

  ret = pipeline_->set_group (1, (x.height () + 3) / 4, x.channels ());
  if (!ret.ok ())
    {
      return ret;
    }

  ret = command_->record_pipeline (*pipeline_, { x, output }, { 0, 2 },
                                   x.shape_constant ());
  if (!ret.ok ())
    {
      return ret;
    }

  output.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  output.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return output;
}
}

