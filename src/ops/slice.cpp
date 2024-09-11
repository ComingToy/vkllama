#include "src/ops/slice.h"
#include "src/core/command.h"
#include "src/shaders/vkllama_comp_shaders.h"

namespace vkllama
{
Slice::Slice (GPUDevice *gpu, Command *command, Tensor::DType dtype)
    : Op (gpu, command), dtype_ (dtype)
{
}

absl::Status
Slice::init () noexcept
{
  constexpr int binding_count = 9;
  Pipeline::ShaderInfo info
      = { 0, binding_count, sizeof (uint32_t) * binding_count, 8, 8, 4 };

  const auto spv_code = dtype_ == Tensor::FP32
                            ? __get_slice_comp_spv_code ()
                            : __get_slice_fp16_comp_spv_code ();
  const auto spv_size = dtype_ == Tensor::FP32
                            ? __get_slice_comp_spv_size ()
                            : __get_slice_fp16_comp_spv_size ();
  pipeline_.reset (new Pipeline (dev_, spv_code, spv_size, {}, info));

  return pipeline_->init ();
}

absl::StatusOr<Tensor>
Slice::operator() (Tensor in, const std::array<uint32_t, 3> &starts,
                   const std::array<uint32_t, 3> &extents) noexcept
{
  if (starts[0] + extents[0] > in.channels ()
      || starts[1] + extents[1] > in.height ()
      || starts[2] + extents[2] > in.width ())
    {
      return absl::OutOfRangeError (absl::StrFormat (
          "slice starts = (%zu, %zu, %zu) from (%zu, %zu, %zu) shape tensor",
          size_t (starts[0]), size_t (starts[1]), size_t (starts[2]),
          in.channels (), in.height (), in.width ()));
    }

  ShaderConstants constants = { (uint32_t)in.channels (),
                                (uint32_t)in.height (),
                                (uint32_t)in.width (),
                                starts[0],
                                starts[1],
                                starts[2],
                                extents[0],
                                extents[1],
                                extents[2] };

  auto out = Tensor (extents[0], extents[1], extents[2], dev_, dtype_);
  auto ret = out.create ();
  if (!ret.ok ())
    {
      return ret;
    }

  uint32_t groupz = (extents[0] + 3) / 4, groupy = (extents[1] + 7) / 8,
           groupx = (extents[2] + 7) / 8;

  ret = pipeline_->set_group (groupx, groupy, groupz);
  if (!ret.ok ())
    {
      return ret;
    }

  ret = command_->record_pipeline (*pipeline_, { in, out }, constants);
  if (!ret.ok ())
    {
      return ret;
    }

  out.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return out;
}

uint64_t
Slice::time () noexcept
{
  return pipeline_->time ();
}

}

