#include "src/ops/transpose.h"
#include "src/core/command.h"
#include "src/core/common.h"
#include "src/ops/op.h"
#include "src/shaders/vkllama_comp_shaders.h"

namespace vkllama
{
Transpose::Transpose (GPUDevice *gpu, Command *command, const int type,
                      Tensor::DType const dtype)
    : Op (gpu, command), dtype_ (dtype), trans_type_ (type)
{
}

absl::Status
Transpose::init () noexcept
{
  if (dtype_ != FP16)
    {
      return absl::InvalidArgumentError (
          "Transpose op: only fp16 is supported.");
    }

  Pipeline::ShaderInfo info = { 0, 2, 2 * sizeof (ShapeConstant), 8, 2, 2 };
  const auto *spv_code = __get_transpose_type0_fp16_comp_spv_code ();
  auto spv_size = __get_transpose_type0_fp16_comp_spv_size ();

  pipeline_.reset (new Pipeline (dev_, spv_code, spv_size, {}, info));

  return pipeline_->init ();
}

absl::StatusOr<Tensor>
Transpose::operator() (Tensor in) noexcept
{
  if (in.dtype () != dtype_)
    {
      return absl::InvalidArgumentError (absl::StrFormat (
          "transpose op defined with %d dtype but the dtype of input0 is %d",
          int (dtype_), int (in.dtype ())));
    }

  if (trans_type_ != 0)
    {
      return absl::UnimplementedError (
          "only transpose type 0 is supported now.");
    }

  if (out_.channels () != in.height () || out_.height () != in.channels ()
      || out_.width () != in.width ())
    {

      out_ = Tensor (in.height (), in.channels (), in.width (), dev_,
                     in.dtype ());
      VKLLAMA_STATUS_OK (out_.create ());
    }

  uint32_t group_x = (out_.width () + 7) / 8,
           group_y = (out_.height () + 1) / 2,
           group_z = (out_.channels () + 1) / 2;

  absl::Status ret;
  if (!(ret = pipeline_->set_group (group_x, group_y, group_z)).ok ())
    {
      return ret;
    }

  auto shape = in.shape_constant () + out_.shape_constant ();

  ret = command_->record_pipeline (*pipeline_, { in, out_ }, shape);
  if (!ret.ok ())
    {
      return ret;
    }

  out_.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  out_.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return out_;
};

uint64_t
Transpose::time () noexcept
{
  return pipeline_->time ();
}
}
