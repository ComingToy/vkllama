#include "src/ops/cast.h"
#include "src/core/command.h"
#include "src/shaders/vkllama_comp_shaders.h"

namespace vkllama
{
Cast::Cast (GPUDevice *gpu, Command *command, const Tensor::DType from,
            const Tensor::DType to)
    : Op (gpu, command), from_ (from), to_ (to)
{
}

absl::Status
Cast::init () noexcept
{
  const uint8_t *spv_code = nullptr;
  size_t spv_size = 0;
  if (from_ == FP32 && to_ == FP16)
    {
      spv_code = __get_cast_fp32_to_fp16_comp_spv_code ();
      spv_size = __get_cast_fp32_to_fp16_comp_spv_size ();
    }
  else if (from_ == FP16 && to_ == FP32)
    {
      spv_code = __get_cast_fp16_to_fp32_comp_spv_code ();
      spv_size = __get_cast_fp16_to_fp32_comp_spv_size ();
    }
  else
    {
      return absl::InvalidArgumentError (
          "only fp32 -> fp16 and fp16 -> fp32 are supported.");
    }

  Pipeline::ShaderInfo info = { 0, 2, sizeof (uint32_t), 128, 1, 1 };
  pipeline_.reset (new Pipeline (dev_, spv_code, spv_size, {}, info));

  return pipeline_->init ();
}

uint64_t
Cast::time () noexcept
{
  return pipeline_->time ();
}

absl::StatusOr<Tensor>
Cast::operator() (Tensor from) noexcept
{
  if (from.dtype () != from_)
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "operator defined as casting from %d dtype but %d given",
          int (from_), int (from.dtype ())));
    }

  Tensor to (from.channels (), from.height (), from.width (), dev_, to_);
  auto ret = to.create ();
  if (!ret.ok ())
    {
      return ret;
    }

  ret = pipeline_->set_group ((from.size () + 127) / 128, 1, 1);
  if (!ret.ok ())
    {
      return ret;
    }

  ShaderConstants N = { static_cast<uint32_t> (from.size ()) };
  ret = command_->record_pipeline (*pipeline_, { from, to }, N);
  to.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  to.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  return to;
}
}

